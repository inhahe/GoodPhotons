// RGB -> reflectance spectrum upsampling (Jakob & Hanika, "A Low-Dimensional
// Function Space for Efficient Spectral Upsampling", EG 2019).
//
// A reflectance is modelled as a sigmoid of a quadratic in wavelength:
//     S(λ) = 1/2 + p / (2·sqrt(1 + p²)),   p = c0·t² + c1·t + c2,
// with t a normalized wavelength. This always yields S ∈ (0,1) (a physical
// reflectance) and is smooth. Given a target linear-sRGB colour we fit the three
// coefficients (c0,c1,c2) at load time with a few Gauss-Newton steps so that the
// spectrum, viewed under D65 through the CIE observer, reproduces that colour.
//
// This replaces the crude three-lobe placeholder; it round-trips sRGB accurately
// (see checkUpsample()). No large precomputed table — the per-colour fit is a
// handful of 3×3 solves, done once per material/texel, not per photon.
#pragma once
#include <cmath>
#include <array>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "color.h"
#include "spectrum.h"
#include "lights.h"
#include "meng_table.h"
#include "parallel.h"
#include "stochtile.h"   // STOCH_JH_N / stochJhCoeff — the device shares this table

namespace upsample {

// Normalized wavelength coordinate for the quadratic (keeps coefficients O(1)).
inline double tOf(double lambda) { return (lambda - 595.0) / 235.0; }

inline double sigmoid(double p) { return 0.5 + 0.5 * p / std::sqrt(1.0 + p * p); }
inline double dSigmoid(double p) { double s = 1.0 + p * p; return 0.5 / (s * std::sqrt(s)); }

// Reflectance value at λ for coefficients c.
inline double reflAt(const std::array<double, 3>& c, double lambda) {
    double t = tOf(lambda);
    double p = c[0] * t * t + c[1] * t + c[2];
    return sigmoid(p);
}


// Precomputed integration weights: wX/Y/Z(λ) = k·D65(λ)·CMF(λ)·dλ, sampled on a
// fixed grid, with k chosen so a unit reflectance integrates to the D65 white
// point (Y = 1). Built once.
struct Basis {
    static constexpr int N = 95;          // (830-360)/5 + 1
    static constexpr double step = 5.0;
    double lam[N];
    double wX[N], wY[N], wZ[N];
    Basis() {
        Spectrum d65 = daylight(6504.0);
        double kY = 0.0;
        for (int i = 0; i < N; ++i) {
            double w = LAMBDA_MIN + i * step;
            lam[i] = w;
            double e = std::max(0.0, d65(w));
            wX[i] = e * cieX(w) * step;
            wY[i] = e * cieY(w) * step;
            wZ[i] = e * cieZ(w) * step;
            kY += wY[i];
        }
        double k = (kY > 0) ? 1.0 / kY : 1.0;
        for (int i = 0; i < N; ++i) { wX[i] *= k; wY[i] *= k; wZ[i] *= k; }
    }
    // XYZ of a unit-vs-reflectance under D65.
    void integrate(const std::array<double, 3>& c, double& X, double& Y, double& Z) const {
        X = Y = Z = 0.0;
        for (int i = 0; i < N; ++i) {
            double s = reflAt(c, lam[i]);
            X += s * wX[i]; Y += s * wY[i]; Z += s * wZ[i];
        }
    }
};

inline const Basis& basis() { static Basis b; return b; }

// Soft-clip a fitted coefficient vector so the quadratic it encodes stays bounded,
// WITHOUT disturbing the part of the curve that is actually doing any work.
//
// Why it is needed: the model is R = sigmoid(p(t)), so a colour whose reflectance is
// pinned at 0 or 1 across the band can only be expressed by |p| -> infinity. The fitter
// duly returns |c| in the millions at the white and black corners. That is harmless for
// a single evaluation but fatal for a LOOKUP TABLE, because interpolating between a
// corner like that and its moderate neighbour sweeps p through every intermediate value.
//
// How: soft-compress p through pMax*tanh(p/pMax) — which is the identity to within a
// part in 1e5 while |p| < pMax/5, so an unsaturated curve is untouched, and which keeps
// every ROOT of p exactly where it was, so the wavelengths at which the reflectance
// crosses 1/2 do not move — then least-squares a quadratic back through the compressed
// curve. The weight is dSigmoid(p), the sensitivity of reflectance to p: minimising
// sum w*(q - p')^2 is a first-order stand-in for minimising sum (R(q) - R(p'))^2, which
// is what we actually care about. A floor keeps the normal equations conditioned where
// the whole curve is saturated and the weights would otherwise all be ~1e-6.
//
// Two simpler things were tried first and are wrong, both measured:
//   * Scale c uniformly until max|p| <= pMax. Preserves roots, but shrinks the
//     UNSATURATED part of the curve along with the saturated part — a dark saturated red
//     came out with |dR| = 0.97.
//   * Read p at three wavelengths, clamp each, and rebuild the quadratic through them
//     (exact wherever it does not clamp, which is seductive). It moves the roots, and
//     near white the fit puts its roots within a nanometre or two of 400 and 700 nm —
//     precisely where one wants to put the outer nodes — so white lost the 400-430 and
//     670-700 nm ends of its spectrum: |XYZ - target| = 0.37.
inline std::array<double, 3> clampSaturated(const std::array<double, 3>& c,
                                            double pMax = 60.0) {
    auto p = [&](double t) { return c[0] * t * t + c[1] * t + c[2]; };
    // Cheap exit: if the curve never saturates hard, it is already representable.
    {
        const double tLo = tOf(380.0), tHi = tOf(730.0);
        double m = std::max(std::fabs(p(tLo)), std::fabs(p(tHi)));
        if (std::fabs(c[0]) > 1e-300) {
            const double tv = -c[1] / (2.0 * c[0]);
            if (tv > tLo && tv < tHi) m = std::max(m, std::fabs(p(tv)));
        }
        if (!(m > pMax)) return c;              // also catches NaN: leave it alone
    }
    // Weighted least squares of {t^2, t, 1} against the compressed curve, over the
    // visible band only (outside it the original fit is unconstrained, so its values
    // there are noise and must not be fitted to). Weighting by the CMF as well was tried
    // and measures worse (worst |XYZ - target| 0.037 -> 0.088): it lets the band edges,
    // where the observer is nearly blind but the curve still has to stay high for a
    // white, drift away.
    double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, rhs[3] = {0, 0, 0};
    for (double lam = 380.0; lam <= 730.0 + 1e-9; lam += 5.0) {
        const double t  = tOf(lam);
        const double pc = pMax * std::tanh(p(t) / pMax);
        const double w  = std::max(dSigmoid(pc), 1e-3);
        const double bs[3] = {t * t, t, 1.0};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) A[a][b] += w * bs[a] * bs[b];
            rhs[a] += w * bs[a] * pc;
        }
    }
    auto det3 = [](const double m[3][3]) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    };
    const double D = det3(A);
    if (!(std::fabs(D) > 1e-30)) return c;
    std::array<double, 3> out{};
    for (int k = 0; k < 3; ++k) {
        double M[3][3];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) M[a][b] = (b == k) ? rhs[a] : A[a][b];
        out[k] = det3(M) / D;
    }
    return out;
}

// Linear sRGB (D65) -> XYZ. Inverse of color.h's xyzToLinearSrgb matrix.
inline void linSrgbToXyz(double r, double g, double b, double& X, double& Y, double& Z) {
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b;
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b;
}

// Fit sigmoid coefficients so the modelled spectrum, integrated against the given
// weight set (wX/Y/Z(λ)·dλ, already built), reproduces the target XYZ. Gauss-Newton
// with a Cramer's-rule 3×3 solve. Shared by the reflectance fit (D65-weighted basis)
// and the illuminant fit (bare-observer basis) — identical solver, different weights.
//
// The step is BACKTRACKED. An undamped Gauss-Newton step here does not merely converge
// slowly on hard targets — it diverges outright, because the model is a sigmoid of a
// quadratic and a dark saturated colour needs |c| in the hundreds (the sigmoid has to
// be pinned near 0 across most of the band and lifted in a narrow window). Out there
// dSigmoid is ~0, so the Jacobian is nearly singular, Cramer's rule returns an enormous
// Δ, and the iterate is thrown somewhere with an even worse residual; 40 such steps
// wander and the returned coefficients are unrelated to the request. Measured before
// this guard, with a 95-sample D65 basis: pure red at luminance 0.001 came back with
// |XYZ - target| = 1.41 and pure blue at 0.01 with 1.40 — i.e. a *black* red rendered
// with the tristimulus of something bright. Greys and mid-tones were unaffected, which
// is why it survived: it only bites where the colour is both dark and saturated.
// Halving the step until the residual actually decreases costs a few extra spectrum
// evaluations on the hard cases and nothing on the easy ones, and turns those same two
// fits into 1e-9. See `-checkupsample` §fit.
inline std::array<double, 3> fitSigmoid(int N, const double* lam,
                                        const double* wX, const double* wY, const double* wZ,
                                        double tX, double tY, double tZ, double pMax = 0.0) {
    std::array<double, 3> c{0.0, 0.0, 0.0};   // start at S ≡ 0.5 (mid gray)
    // ||XYZ(c) - target||^2, and (optionally) the Jacobian at c.
    auto eval = [&](const std::array<double, 3>& cc, double J[3][3]) {
        double X = 0, Y = 0, Z = 0;
        if (J) for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) J[a][b] = 0.0;
        for (int i = 0; i < N; ++i) {
            const double t = tOf(lam[i]);
            const double p = cc[0] * t * t + cc[1] * t + cc[2];
            const double s = sigmoid(p);
            X += s * wX[i]; Y += s * wY[i]; Z += s * wZ[i];
            if (J) {
                const double ds = dSigmoid(p);
                const double dc[3] = {t * t, t, 1.0};
                for (int j = 0; j < 3; ++j) {
                    J[0][j] += ds * dc[j] * wX[i];
                    J[1][j] += ds * dc[j] * wY[i];
                    J[2][j] += ds * dc[j] * wZ[i];
                }
            }
        }
        const double rX = X - tX, rY = Y - tY, rZ = Z - tZ;
        return rX * rX + rY * rY + rZ * rZ;
    };

    double J[3][3];
    double err = eval(c, J);
    for (int iter = 0; iter < 60; ++iter) {
        if (err < 1e-14) break;
        // Solve J·Δ = residual (3×3) via Cramer's rule. Recomputing the residual here
        // is redundant with eval(), so eval() reports err and we re-derive the residual
        // vector from the same pass by evaluating XYZ once more only when stepping.
        double X = 0, Y = 0, Z = 0;
        for (int i = 0; i < N; ++i) {
            const double t = tOf(lam[i]);
            const double s = sigmoid(c[0] * t * t + c[1] * t + c[2]);
            X += s * wX[i]; Y += s * wY[i]; Z += s * wZ[i];
        }
        const double rhs[3] = {X - tX, Y - tY, Z - tZ};
        const double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
                         - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
                         + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
        if (std::fabs(det) < 1e-30) break;
        double d[3];
        for (int col = 0; col < 3; ++col) {
            double M[3][3];
            for (int a = 0; a < 3; ++a) for (int bb = 0; bb < 3; ++bb) M[a][bb] = J[a][bb];
            for (int a = 0; a < 3; ++a) M[a][col] = rhs[a];
            const double dc = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
            d[col] = dc / det;
        }
        // Backtracking line search: accept the first step that lowers the residual.
        bool moved = false;
        double relax = 1.0;
        for (int bt = 0; bt < 30; ++bt, relax *= 0.5) {
            std::array<double, 3> t{c[0] - relax * d[0], c[1] - relax * d[1],
                                    c[2] - relax * d[2]};
            // PROJECTED Gauss-Newton (pMax > 0, used only when building the LUT): pull the
            // trial point back into the bounded set before scoring it, so what the line
            // search accepts is the best BOUNDED colour match rather than an unbounded one
            // that is then mangled after the fact. Soft-clipping the finished fit instead
            // costs 0.037 of |XYZ - target| near white; doing it inside the loop lets the
            // remaining freedom compensate for the clip.
            if (pMax > 0.0) t = clampSaturated(t, pMax);
            double Jt[3][3];
            const double e = eval(t, Jt);
            if (e < err) {
                c = t; err = e;
                for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) J[a][b] = Jt[a][b];
                moved = true;
                break;
            }
        }
        if (!moved) break;      // no downhill step exists along this direction
    }
    return c;
}

// Fit sigmoid coefficients for a linear-sRGB *reflectance* colour (D65-weighted basis).
inline std::array<double, 3> fit(double r, double g, double b, double pMax = 0.0) {
    const Basis& B = basis();
    double tX, tY, tZ; linSrgbToXyz(r, g, b, tX, tY, tZ);
    return fitSigmoid(B.N, B.lam, B.wX, B.wY, B.wZ, tX, tY, tZ, pMax);
}

// --- Bulk fit: fit(...) for a whole image, DEDUPLICATED and THREADED ---------------
// A single fit is ~40 Gauss-Newton iterations over a 95-sample basis, i.e. a few
// microseconds; run once per texel, serially, that is seconds per megapixel and it lands
// squarely in scene-load latency (a 10-texture gallery scene spent ~45 s of its ~47 s
// startup right here, before anything appeared on screen).
//
// Two exact accelerations, in this order:
//   * DEDUPLICATE. The fit is a pure function of the colour, and an 8-bit source image
//     decodes through a 256-entry per-channel table, so equal texels are BIT-equal
//     Vec3s — hashing the raw bit patterns collapses them safely (no tolerance, no
//     quantisation, so the output is bit-identical to fitting every texel). Real images
//     collapse hard: the project's own procedural marble maps run 96 to 80 k distinct
//     colours over 0.05-1.05 M texels, a 13x-to-10000x reduction. A photograph or an
//     HDR float image dedups little, and then this costs one hash probe per texel.
//   * THREAD the remaining distinct fits. They are independent and their cost varies a
//     lot per colour (saturated colours never hit the residual bail-out and burn all 40
//     iterations), so parallelFor's atomic chunk cursor is what keeps the tail even.
//
// `basis()` is touched before any thread starts so its function-local static is already
// constructed — magic statics are thread-safe, but paying a guarded init on every worker
// probe is pointless when the caller can do it once.
//
// Returns false if a clean stop (`ftrace -stop`, Ctrl-C) was requested while it ran, in
// which case `out` is only partially written and the caller must abandon the load rather
// than render with a half-filled coefficient table.
[[nodiscard]] inline bool fitMany(const Vec3* in, size_t n, std::array<double, 3>* out,
                                  double pMax = 0.0) {
    if (!in || !out || n == 0) return true;
    (void)basis();

    // Key = the three doubles' raw bits. memcpy (not a reinterpret_cast) so this stays
    // free of strict-aliasing UB; -0.0 simply fails to dedup against +0.0, which is
    // harmless (it fits to the same coefficients anyway, just twice).
    struct Key {
        uint64_t a, b, c;
        bool operator==(const Key& o) const { return a == o.a && b == o.b && c == o.c; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            uint64_t h = k.a * 0x9E3779B97F4A7C15ull;
            h ^= k.b + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
            h ^= k.c + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
            return (size_t)h;
        }
    };
    std::unordered_map<Key, uint32_t, KeyHash> seen;
    seen.reserve(std::min<size_t>(n, 1u << 16));
    std::vector<uint32_t> which(n);
    std::vector<Vec3> uniq;
    uniq.reserve(std::min<size_t>(n, 1u << 16));
    for (size_t i = 0; i < n; ++i) {
        // One hash probe per texel is fast, but "fast per texel" is exactly how the
        // serial fit got here — poll the stop flag once per 64 k so even a huge image
        // can't sit through the dedup pass ignoring a stop.
        if ((i & 0xFFFF) == 0 && ft::stopRequested()) return false;
        Key k;
        std::memcpy(&k.a, &in[i].x, sizeof(double));
        std::memcpy(&k.b, &in[i].y, sizeof(double));
        std::memcpy(&k.c, &in[i].z, sizeof(double));
        auto it = seen.find(k);
        if (it == seen.end()) {
            uint32_t id = (uint32_t)uniq.size();
            seen.emplace(k, id);
            uniq.push_back(in[i]);
            which[i] = id;
        } else {
            which[i] = it->second;
        }
    }

    std::vector<std::array<double, 3>> uc(uniq.size());
    if (!ft::parallelFor(uniq.size(), 64, [&](size_t i) {
            uc[i] = fit(uniq[i].x, uniq[i].y, uniq[i].z, pMax);
        }))
        return false;
    return ft::parallelFor(n, 65536, [&](size_t i) { out[i] = uc[which[i]]; });
}

// --- Global linear-RGB -> coefficient LUT ------------------------------------
// `fit` is a 60-iteration Gauss-Newton with a line search — microseconds, fine once
// per distinct texel at load, hopeless per shading point. Stochastic tiling needs
// exactly that, though: it *invents* a colour at every hit (a blend of three crops,
// see stochtile.h) and then has to hand the renderer a spectrum for it.
//
// So: tabulate. The table is a pure function of the colour, shared by every texture
// and every backend, and is built lazily on first use — a scene with no stochastic
// texture never pays for it.
//
// Two details that are not free choices:
//
//   * The grid is warped by sqrt (i.e. indexed by sqrt(channel), inverted by squaring).
//     A LINEAR grid is badly wrong: the coefficients move fastest near black, where a
//     dark colour needs the sigmoid pinned low across the whole band. Measured on the
//     demo image's own colours, 48^3 linear gives mean |dR| 1.0e-2 and worst 2.4e-2 —
//     visibly off. The same 48^3 with the sqrt warp gives mean 1.4e-4, worst 4.3e-3,
//     and a full round trip through the LUT and back out to linear sRGB reproduces the
//     un-tabulated result to within 1e-4. sqrt specifically (rather than the sRGB
//     exponent 1/2.2, which measures the same) because the lookup is on the hot path
//     and sqrt is one instruction where pow is a library call.
//     The residual worst case now sits near WHITE, not black, and is the model's own
//     limit rather than the table's — see STOCH_JH_N's comment for the resolution sweep.
//
//   * `STOCH_JH_N` lives in stochtile.h, not here, because the DEVICE has to do the
//     same trilinear lookup into the same table and stochtile.h is the header both
//     sides share. This function only *builds* it.
//
// THE ENTRIES MUST BE BOUNDED, and this is not cosmetic: the fitter returns |c| = 1.6e6
// at the white corner, and trilinear interpolation between a node like that and its
// moderate neighbour is meaningless — a colour a few percent off white came out with
// |dR| = 0.81. So the table is built with the fit's PROJECTED variant (fit(..., pMax)),
// which keeps every iterate inside |p| <= STOCH_JH_PMAX via clampSaturated. Bounding the
// finished fit instead is measurably worse (worst |XYZ - target| near white 0.037 vs
// 0.005) because the clip then has no chance to be compensated for.
inline const std::vector<float>& coeffLut() {
    static const std::vector<float> tbl = [] {
        const int N = STOCH_JH_N;
        std::vector<Vec3> rgb((size_t)N * N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                for (int k = 0; k < N; ++k) {
                    // Undo the sqrt warp: the grid is uniform in sqrt(channel).
                    const double r = (double)i / (N - 1), g = (double)j / (N - 1),
                                 b = (double)k / (N - 1);
                    rgb[((size_t)i * N + j) * N + k] = Vec3{r * r, g * g, b * b};
                }
        std::vector<std::array<double, 3>> c(rgb.size());
        // A stop during this build leaves `c` partly written; the entries are
        // value-initialised to {0,0,0} (mid grey), which is a harmless placeholder for
        // a load that is being abandoned anyway.
        (void)fitMany(rgb.data(), rgb.size(), c.data(), STOCH_JH_PMAX);
        std::vector<float> out(rgb.size() * 3);
        for (size_t i = 0; i < rgb.size(); ++i)
            for (int k = 0; k < 3; ++k) out[i * 3 + k] = (float)c[i][k];
        return out;
    }();
    return tbl;
}

// --- RGB -> illuminant (emission) spectrum upsampling ------------------------
// The Jakob-Hanika *illuminant* variant. The reflectance fit above pre-multiplies
// its integration weights by the D65 illuminant and clamps the result to a physical
// (0,1) reflectance — right for a surface seen under a light, wrong for a light
// itself. An emitter's own SPD is what the observer integrates *directly*, so here
// (a) the weights are the bare CIE observer (no D65), and (b) the SPD is unbounded:
// we model it as A·sigmoid(quadratic), where the sigmoid ∈ (0,1) carries the shape
// (chromaticity) and the scalar A carries the magnitude, so any brightness — even a
// saturated primary or an over-unity light — is representable. Meant for lights
// (`spd rgbillum r g b`).
struct IllumBasis {
    static constexpr int N = 95;          // (830-360)/5 + 1
    static constexpr double step = 5.0;
    double lam[N];
    double wX[N], wY[N], wZ[N];
    IllumBasis() {
        double kY = 0.0;
        for (int i = 0; i < N; ++i) {
            double w = LAMBDA_MIN + i * step;
            lam[i] = w;
            wX[i] = cieX(w) * step;       // bare observer, NO D65 pre-weighting
            wY[i] = cieY(w) * step;
            wZ[i] = cieZ(w) * step;
            kY += wY[i];
        }
        // Normalize so a flat unit spectrum (S ≡ 1) integrates to Y = 1 — i.e. the
        // equal-energy white E (the natural neutral for a self-luminous source).
        double k = (kY > 0) ? 1.0 / kY : 1.0;
        for (int i = 0; i < N; ++i) { wX[i] *= k; wY[i] *= k; wZ[i] *= k; }
    }
    // XYZ of a unit-amplitude sigmoid emission (multiply by A for the true SPD).
    void integrate(const std::array<double, 3>& c, double& X, double& Y, double& Z) const {
        X = Y = Z = 0.0;
        for (int i = 0; i < N; ++i) {
            double s = reflAt(c, lam[i]);
            X += s * wX[i]; Y += s * wY[i]; Z += s * wZ[i];
        }
    }
};

inline const IllumBasis& illumBasis() { static IllumBasis b; return b; }

// Fit sigmoid coefficients for a *normalized* emission chromaticity against the
// bare-observer basis. The caller pre-divides the target XYZ by the amplitude A
// (so the fitted sigmoid ∈ (0,1) reproduces the normalized target), then scales
// the SPD back up by A. Same solver as the reflectance fit, different weights.
inline std::array<double, 3> fitIllum(const IllumBasis& B, double tX, double tY, double tZ) {
    return fitSigmoid(B.N, B.lam, B.wX, B.wY, B.wZ, tX, tY, tZ);
}

// --- RGB -> dominant wavelength (spectral-locus construction) ----------------
// Distinct from the reflectance upsample above: this maps a colour to a *single*
// dominant wavelength for near-monochromatic (line) emission. Draw a ray from the
// D65 white point through the sample's xy chromaticity; where it crosses the
// spectral locus is the dominant wavelength. If it crosses the "line of purples"
// instead (magentas have no real dominant wavelength), report that with a blend
// between the violet/red endpoints so the caller can emit a two-line mix.

constexpr double D65_x = 0.31272, D65_y = 0.32903;   // CIE 1931 D65 chromaticity
constexpr double LOCUS_LO = 400.0, LOCUS_HI = 700.0; // visible locus span sampled

struct LocusPt { double x, y, lam; };

// The chromaticity horseshoe (spectral locus) sampled at 1 nm, closed by the
// line of purples (last->first). Convex, so a ray from the interior white point
// exits through exactly one edge. Built once.
inline const std::vector<LocusPt>& locusPolygon() {
    static const std::vector<LocusPt> poly = [] {
        std::vector<LocusPt> p;
        for (double lam = LOCUS_LO; lam <= LOCUS_HI + 1e-9; lam += 1.0) {
            double X = cieX(lam), Y = cieY(lam), Z = cieZ(lam);
            double s = X + Y + Z;
            if (s <= 0) continue;
            p.push_back({X / s, Y / s, lam});
        }
        return p;
    }();
    return poly;
}

struct DomWL {
    double lambda = 560.0;   // dominant wavelength (nm); unused when complementary
    double purity = 0.0;     // excitation purity in [0,1] (white=0, locus=1)
    bool   complementary = false;  // true for purples/magentas (no real dominant λ)
    double purpleBlend = 0.0;      // for purples: 0 at violet end, 1 at red end
};

// Map a linear-sRGB colour to its dominant wavelength.
inline DomWL rgbToDominantWavelength(double r, double g, double b) {
    DomWL out;
    double X, Y, Z; linSrgbToXyz(r, g, b, X, Y, Z);
    double s = X + Y + Z;
    if (s <= 1e-9) return out;                 // black -> ill-defined, purity 0
    double sx = X / s, sy = Y / s;
    double dx = sx - D65_x, dy = sy - D65_y;   // ray direction: white -> sample
    if (dx * dx + dy * dy < 1e-10) return out;  // (near) white -> purity 0

    const auto& L = locusPolygon();
    const int n = (int)L.size();
    for (int i = 0; i < n; ++i) {
        const LocusPt& P0 = L[i];
        const LocusPt& P1 = L[(i + 1) % n];
        double ex = P1.x - P0.x, ey = P1.y - P0.y;
        double det = ex * dy - dx * ey;
        if (std::fabs(det) < 1e-15) continue;
        // Solve  t*D - u*E = P0 - W  for ray param t (>=0) and edge param u (0..1).
        double bx = P0.x - D65_x, by = P0.y - D65_y;
        double t = (-bx * ey + ex * by) / det;
        double u = (dx * by - dy * bx) / det;
        if (t <= 1e-9 || u < -1e-9 || u > 1.0 + 1e-9) continue;
        out.purity = std::clamp(1.0 / t, 0.0, 1.0);   // sample sits at t=1
        if ((i + 1) % n == 0) {                        // the closing (purple) edge
            out.complementary = true;
            out.purpleBlend = std::clamp(u, 0.0, 1.0);
        } else {
            out.lambda = P0.lam + std::clamp(u, 0.0, 1.0) * (P1.lam - P0.lam);
        }
        return out;
    }
    return out;   // no crossing found (degenerate) -> defaults
}

// --- Smits 1999 RGB->reflectance basis ------------------------------------
// Brian Smits, "An RGB-to-Spectrum Conversion for Reflectances" (1999): seven
// tabulated basis reflectances (white / C M Y / R G B) sampled at 10 wavelengths
// evenly spanning [380,720] nm. An RGB triple is decomposed additively —
// white·min + one secondary + one primary — so the reconstruction stays smooth
// and (unlike a naive box) round-trips RGB reasonably. Classic, cheaper and
// lower-fidelity than Jakob-Hanika; offered as a selectable alternative (K1).
struct SmitsBasis {
    static constexpr int N = 10;
    double lam[N];
    double white[N], cyan[N], magenta[N], yellow[N], red[N], green[N], blue[N];
    SmitsBasis() {
        const double W[N]  = {1.0000,1.0000,0.9999,0.9993,0.9992,0.9998,1.0000,1.0000,1.0000,1.0000};
        const double C[N]  = {0.9710,0.9426,1.0007,1.0007,1.0007,1.0007,0.1564,0.0000,0.0000,0.0000};
        const double M[N]  = {1.0000,1.0000,0.9685,0.2229,0.0000,0.0458,0.8369,1.0000,1.0000,0.9959};
        const double Y[N]  = {0.0001,0.0000,0.1088,0.6651,1.0000,1.0000,0.9996,0.9586,0.9685,0.9840};
        const double R[N]  = {0.1012,0.0515,0.0000,0.0000,0.0000,0.0000,0.8325,1.0149,1.0149,1.0149};
        const double G[N]  = {0.0000,0.0000,0.0273,0.7937,1.0000,0.9418,0.1719,0.0000,0.0000,0.0025};
        const double B[N]  = {1.0000,1.0000,0.8916,0.3323,0.0000,0.0000,0.0003,0.0369,0.0483,0.0496};
        for (int i = 0; i < N; ++i) {
            lam[i] = 380.0 + (720.0 - 380.0) * i / (N - 1);
            white[i]=W[i]; cyan[i]=C[i]; magenta[i]=M[i]; yellow[i]=Y[i];
            red[i]=R[i]; green[i]=G[i]; blue[i]=B[i];
        }
    }
};
inline const SmitsBasis& smitsBasis() { static SmitsBasis b; return b; }

// Additive Smits decomposition of a clamped RGB triple into the 10-sample lattice.
inline std::array<double, SmitsBasis::N> smitsCombine(double r, double g, double b) {
    const SmitsBasis& B = smitsBasis();
    std::array<double, SmitsBasis::N> ret{};   // value-initialised to 0
    auto add = [&](const double* basis, double w) {
        for (int i = 0; i < SmitsBasis::N; ++i) ret[i] += w * basis[i];
    };
    if (r <= g && r <= b) {
        add(B.white, r);
        if (g <= b) { add(B.cyan, g - r);    add(B.blue,  b - g); }
        else        { add(B.cyan, b - r);    add(B.green, g - b); }
    } else if (g <= r && g <= b) {
        add(B.white, g);
        if (r <= b) { add(B.magenta, r - g); add(B.blue,  b - r); }
        else        { add(B.magenta, b - g); add(B.red,   r - b); }
    } else {
        add(B.white, b);
        if (r <= g) { add(B.yellow, r - b);  add(B.green, g - r); }
        else        { add(B.yellow, g - b);  add(B.red,   r - g); }
    }
    for (auto& v : ret) v = std::clamp(v, 0.0, 1.0);   // guarantee a physical reflectance
    return ret;
}

// --- Plain 3-box RGB->reflectance ------------------------------------------
// The simplest possible upsampler: three fixed rectangular reflectance bands
// (B [400,500), G [500,600), R [600,700) nm). Rather than dumping r,g,b straight
// into the boxes (which would badly mis-reproduce the colour), the three band
// heights are *calibrated* — each unit band's response under D65 through the CIE
// observer is precomputed as a column of a 3x3, and its inverse maps a target
// linear-sRGB triple to the heights that reconstruct it. So a plain box still
// round-trips as well as a 3-primary basis can (heights clamped to [0,1], so very
// saturated colours degrade gracefully). Cheapest option; sharp band edges make
// it handy for testing dispersion/spectral response (K1).
struct BoxBasis {
    static constexpr double edges[4] = {400.0, 500.0, 600.0, 700.0};   // B, G, R
    double Minv[9];   // linear-sRGB target -> (hBlue, hGreen, hRed)
    BoxBasis() {
        const Basis& B = basis();
        double M[9];   // columns: linear-sRGB response of each unit band
        for (int band = 0; band < 3; ++band) {
            double lo = edges[band], hi = edges[band + 1];
            double X = 0, Y = 0, Z = 0;
            for (int i = 0; i < B.N; ++i) {
                if (B.lam[i] >= lo && B.lam[i] < hi) { X += B.wX[i]; Y += B.wY[i]; Z += B.wZ[i]; }
            }
            Vec3 lin = xyzToLinearSrgb(Vec3{X, Y, Z});
            M[0 * 3 + band] = lin.x; M[1 * 3 + band] = lin.y; M[2 * 3 + band] = lin.z;
        }
        // 3x3 inverse (cofactor / determinant); identity fallback if singular.
        double d = M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]);
        if (std::fabs(d) < 1e-12) { for (int i = 0; i < 9; ++i) Minv[i] = (i % 4 == 0) ? 1.0 : 0.0; return; }
        double id = 1.0 / d;
        Minv[0] =  (M[4]*M[8]-M[5]*M[7]) * id;
        Minv[1] = -(M[1]*M[8]-M[2]*M[7]) * id;
        Minv[2] =  (M[1]*M[5]-M[2]*M[4]) * id;
        Minv[3] = -(M[3]*M[8]-M[5]*M[6]) * id;
        Minv[4] =  (M[0]*M[8]-M[2]*M[6]) * id;
        Minv[5] = -(M[0]*M[5]-M[2]*M[3]) * id;
        Minv[6] =  (M[3]*M[7]-M[4]*M[6]) * id;
        Minv[7] = -(M[0]*M[7]-M[1]*M[6]) * id;
        Minv[8] =  (M[0]*M[4]-M[1]*M[3]) * id;
    }
};
inline const BoxBasis& boxBasis() { static BoxBasis b; return b; }

// --- Meng 2015 "smoothest spectrum" grid -----------------------------------
// Meng, Simon, Hanika & Dachsbacher, "Physically Meaningful Rendering using
// Tristimulus Colours" (EGSR 2015): rather than fitting an analytic shape (JH)
// or mixing fixed basis curves (Smits/box), tabulate the *smoothest* reflectance
// — the one minimising roughness sum (s[i+1]-s[i])^2 — that realises a given
// chromaticity at the greatest attainable brightness, then interpolate the table
// and rescale to the requested luminance. Smoothness matters because a smooth
// reflectance is what real pigments look like, so re-illuminating it under a
// non-D65 light (or dispersing it) behaves plausibly instead of ringing.
//
// The table (meng_table.h) is OURS: the paper's supplemental data carries no
// licence, so tools/bake_meng.py re-solves the same optimisation from scratch
// against ftrace's own observer/D65. Three departures from the paper's grid,
// all of which make the result *exact* rather than approximate (details and
// derivations in bake_meng.py):
//   * the lattice is barycentric over the sRGB primary triangle rather than a
//     rotated grid over the whole locus — legitimate because every colour
//     ftrace upsamples comes from `rgb r g b`, so the enclosing cell follows in
//     closed form from the colour itself: no search, no inside/outside test;
//   * vertex k is weighted by bary_k/T_k with T_k = X+Y+Z of its spectrum,
//     which lands the mix on the requested chromaticity exactly (a chromaticity
//     is an (X+Y+Z)-weighted mean, so unweighted blending drifts off-hue);
//   * the tabulated spectra are normalised to unit luminance and solved with NO
//     upper bound, so scaling by the requested Y is exactly optimal (the cone
//     {s >= 0} is scale-invariant; the box {0 <= s <= 1} is not, and clamping
//     the table to it costs real smoothness — see bake_meng.py's smoothest()).
// Measured against a from-scratch solve of the same colour, the result is the
// true global minimum-roughness reflectance to ~0.2%, with zero colour error.
//
// X+Y+Z of each unit sRGB primary — the factor converting a linear-sRGB
// component into its share of the chromaticity mix (column sums of linSrgbToXyz).
inline constexpr double MENG_PRIM_SUM[3] = {
    0.4124 + 0.2126 + 0.0193,
    0.3576 + 0.7152 + 0.1192,
    0.1805 + 0.0722 + 0.9505,
};

// Row-major index of lattice point (a,b), a+b <= MENG_ORDER. Clamped so a
// degenerate cell on the triangle's edge can never index out of the table (the
// offending corner always carries weight 0 there anyway).
inline int mengVertex(int a, int b) {
    a = std::clamp(a, 0, MENG_ORDER);
    b = std::clamp(b, 0, MENG_ORDER - a);
    return a * (MENG_ORDER + 1) - (a * (a - 1)) / 2 + b;
}

// The interpolated, luminance-matched reflectance samples for a linear-sRGB
// colour, on the table's own 5 nm lattice. Empty (all-zero) for black.
inline std::array<double, MENG_N> mengSamples(double r, double g, double b) {
    std::array<double, MENG_N> out{};
    r = std::clamp(r, 0.0, 1.0); g = std::clamp(g, 0.0, 1.0); b = std::clamp(b, 0.0, 1.0);

    // Barycentric coordinates in the primary triangle: component * primary sum.
    double lr = r * MENG_PRIM_SUM[0], lg = g * MENG_PRIM_SUM[1], lb = b * MENG_PRIM_SUM[2];
    double tot = lr + lg + lb;
    if (tot < 1e-12) return out;                       // black -> zero reflectance
    lr /= tot; lg /= tot;

    // Locate the enclosing sub-triangle of the order-N lattice.
    double u = lr * MENG_ORDER, v = lg * MENG_ORDER;
    int i = std::min((int)u, MENG_ORDER - 1);
    int j = std::min((int)v, MENG_ORDER - 1);
    double fu = u - i, fv = v - j;
    int   ia[3]; int jb[3]; double wt[3];
    if (fu + fv <= 1.0) {                              // lower ("upright") triangle
        ia[0]=i;   jb[0]=j;   wt[0]=1.0 - fu - fv;
        ia[1]=i+1; jb[1]=j;   wt[1]=fu;
        ia[2]=i;   jb[2]=j+1; wt[2]=fv;
    } else {                                           // upper ("inverted") triangle
        ia[0]=i+1; jb[0]=j;   wt[0]=1.0 - fv;
        ia[1]=i;   jb[1]=j+1; wt[1]=1.0 - fu;
        ia[2]=i+1; jb[2]=j+1; wt[2]=fu + fv - 1.0;
    }

    // Mix, dividing each vertex by its own X+Y+Z so the result's chromaticity is
    // exactly the barycentric mix of the corners' (a chromaticity is an
    // (X+Y+Z)-weighted mean, so unweighted blending would drift off-hue).
    double wsum = 0.0;
    for (int k = 0; k < 3; ++k) {
        if (wt[k] <= 0.0) continue;
        int idx = mengVertex(ia[k], jb[k]);
        double w = wt[k] / MENG_SUM[idx];
        for (int s = 0; s < MENG_N; ++s) out[s] += w * MENG_SPECTRA[idx][s];
        wsum += w;
    }
    if (wsum <= 0.0) return out;
    for (double& s : out) s /= wsum;

    // Rescale to the requested luminance, then clamp to a physical reflectance.
    // The vertices carry Y = 1, so this is a pure scale — and because they were
    // solved without an upper bound over the scale-invariant cone {s >= 0}, the
    // scaled result is still the exact optimum. Clamping (the paper's own
    // simplest fix-up) therefore only bites for colours brighter than ANY smooth
    // reflectance of that chromaticity can be — e.g. pure sRGB white, whose
    // chromaticity differs from that of a flat reflectance under ftrace's D65.
    double tX, tY, tZ; linSrgbToXyz(r, g, b, tX, tY, tZ);
    const Basis& B = basis();
    double mixY = 0.0;
    for (int n = 0; n < B.N; ++n) {
        int s = std::clamp((int)std::lround((B.lam[n] - MENG_LAMBDA_MIN) / MENG_LAMBDA_STEP),
                           0, MENG_N - 1);
        mixY += out[s] * B.wY[n];
    }
    if (mixY < 1e-12) { out.fill(0.0); return out; }
    double scale = tY / mixY;
    for (double& s : out) s = std::clamp(s * scale, 0.0, 1.0);
    return out;
}

} // namespace upsample

// Build a near-monochromatic *emission* Spectrum from a linear-sRGB triple: a
// narrow Gaussian at the colour's dominant wavelength (K3). Saturation controls
// line width (a vivid colour -> tight spike; a pale one -> broad band that tends
// back to white). Purples/magentas have no single dominant λ, so they become a
// two-line violet+red mix. `sigmaOverride` (nm, >0) forces the line width.
inline Spectrum rgbToLineEmission(double r, double g, double b, double sigmaOverride = -1.0) {
    upsample::DomWL d = upsample::rgbToDominantWavelength(r, g, b);
    if (d.complementary) {
        double blend = d.purpleBlend;                 // 0 violet .. 1 red
        double sig = (sigmaOverride > 0.0) ? sigmaOverride : 14.0;
        Spectrum lo = gaussianBand(upsample::LOCUS_LO, sig, 1.0 - blend);
        Spectrum hi = gaussianBand(upsample::LOCUS_HI, sig, blend);
        return [lo, hi](double w) { return lo(w) + hi(w); };
    }
    double sigma = (sigmaOverride > 0.0) ? sigmaOverride
                                         : 5.0 + 125.0 * (1.0 - d.purity);
    return gaussianBand(d.lambda, sigma, 1.0);
}

// Build a reflectance Spectrum from a linear-sRGB triple (Smits 1999). The 10
// tabulated samples are combined additively then linearly interpolated in λ;
// outside [380,720] nm the endpoint value is held.
inline Spectrum rgbToReflectanceSmits(double r, double g, double b) {
    r = std::clamp(r, 0.0, 1.0); g = std::clamp(g, 0.0, 1.0); b = std::clamp(b, 0.0, 1.0);
    auto vals = upsample::smitsCombine(r, g, b);
    const upsample::SmitsBasis& B = upsample::smitsBasis();
    std::array<double, upsample::SmitsBasis::N> lam;
    for (int i = 0; i < upsample::SmitsBasis::N; ++i) lam[i] = B.lam[i];
    return [vals, lam](double w) -> double {
        const int N = upsample::SmitsBasis::N;
        if (w <= lam[0])     return vals[0];
        if (w >= lam[N - 1]) return vals[N - 1];
        int i = 0; while (i < N - 1 && w > lam[i + 1]) ++i;
        double t = (w - lam[i]) / (lam[i + 1] - lam[i]);
        return vals[i] * (1.0 - t) + vals[i + 1] * t;
    };
}

// Build a reflectance Spectrum from a linear-sRGB triple (plain calibrated 3-box).
// Three rectangular bands whose heights are solved to reproduce the colour under
// D65; heights clamped to [0,1]. Zero outside [400,700) nm.
inline Spectrum rgbToReflectanceBox(double r, double g, double b) {
    r = std::clamp(r, 0.0, 1.0); g = std::clamp(g, 0.0, 1.0); b = std::clamp(b, 0.0, 1.0);
    const upsample::BoxBasis& BB = upsample::boxBasis();
    std::array<double, 3> h;
    for (int i = 0; i < 3; ++i)
        h[i] = std::clamp(BB.Minv[i*3+0]*r + BB.Minv[i*3+1]*g + BB.Minv[i*3+2]*b, 0.0, 1.0);
    return [h](double w) -> double {
        if (w >= 400.0 && w < 500.0) return h[0];   // blue band
        if (w >= 500.0 && w < 600.0) return h[1];   // green band
        if (w >= 600.0 && w < 700.0) return h[2];   // red band
        return 0.0;
    };
}

// Build a reflectance Spectrum from a linear-sRGB triple (Meng 2015 smoothest-
// spectrum grid). The 81 tabulated samples are linearly interpolated in lambda;
// outside [380,780] nm the endpoint value is held — which is exactly the
// convention tools/bake_meng.py folded into the weights it solved against, so
// the round-trip through reflectanceToLinearSrgbD65 is exact (not approximate).
inline Spectrum rgbToReflectanceMeng(double r, double g, double b) {
    auto vals = upsample::mengSamples(r, g, b);
    return [vals](double w) -> double {
        constexpr int N = upsample::MENG_N;
        double t = (w - upsample::MENG_LAMBDA_MIN) / upsample::MENG_LAMBDA_STEP;
        if (t <= 0.0)      return vals[0];
        if (t >= N - 1)    return vals[N - 1];
        int i = (int)t;
        double f = t - i;
        return vals[i] * (1.0 - f) + vals[i + 1] * f;
    };
}

// Build a reflectance Spectrum from a linear-sRGB triple (Jakob-Hanika fit).
inline Spectrum rgbToReflectanceJH(double r, double g, double b) {
    r = std::clamp(r, 0.0, 1.0); g = std::clamp(g, 0.0, 1.0); b = std::clamp(b, 0.0, 1.0);
    std::array<double, 3> c = upsample::fit(r, g, b);
    return [c](double lambda) { return upsample::reflAt(c, lambda); };
}

// Build an *emission* Spectrum from a linear-sRGB triple (Jakob-Hanika illuminant
// variant). Unlike rgbToReflectanceJH, the SPD is unbounded: we factor it as
// A·sigmoid(quadratic), fitting the sigmoid shape to the chromaticity (target XYZ
// normalized by A) and letting the scalar A = 2·max(X,Y,Z) carry the magnitude, so
// even a saturated primary or an over-unity light is representable. Black -> zero.
inline Spectrum rgbToIlluminantJH(double r, double g, double b) {
    r = std::max(0.0, r); g = std::max(0.0, g); b = std::max(0.0, b);   // emitters unbounded above
    double tX, tY, tZ; upsample::linSrgbToXyz(r, g, b, tX, tY, tZ);
    double mx = std::max({tX, tY, tZ});
    if (mx < 1e-9) return [](double) { return 0.0; };   // black -> no emission
    double A = 2.0 * mx;   // headroom so the normalized target's max component is 0.5
    std::array<double, 3> c = upsample::fitIllum(upsample::illumBasis(), tX / A, tY / A, tZ / A);
    return [c, A](double lambda) { return A * upsample::reflAt(c, lambda); };
}

// Reduce an arbitrary reflectance Spectrum to a linear-sRGB triple by integrating
// it under D65 through the CIE observer (the same basis rgbToReflectanceJH inverts,
// so the two round-trip). Used to interpolate colour stops in linear RGB. Not
// clamped — the caller may lerp several of these before upsampling back.
inline Vec3 reflectanceToLinearSrgbD65(const Spectrum& refl) {
    const upsample::Basis& B = upsample::basis();
    double X = 0, Y = 0, Z = 0;
    for (int i = 0; i < B.N; ++i) {
        double s = refl ? std::max(0.0, refl(B.lam[i])) : 0.0;
        X += s * B.wX[i]; Y += s * B.wY[i]; Z += s * B.wZ[i];
    }
    return xyzToLinearSrgb({X, Y, Z});
}
