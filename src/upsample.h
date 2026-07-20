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
#include "color.h"
#include "spectrum.h"
#include "lights.h"

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

// Linear sRGB (D65) -> XYZ. Inverse of color.h's xyzToLinearSrgb matrix.
inline void linSrgbToXyz(double r, double g, double b, double& X, double& Y, double& Z) {
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b;
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b;
}

// Fit sigmoid coefficients for a linear-sRGB colour via Gauss-Newton.
inline std::array<double, 3> fit(double r, double g, double b) {
    const Basis& B = basis();
    double tX, tY, tZ; linSrgbToXyz(r, g, b, tX, tY, tZ);
    std::array<double, 3> c{0.0, 0.0, 0.0};   // start at S ≡ 0.5 (mid gray)
    for (int iter = 0; iter < 40; ++iter) {
        // Residual and 3×3 Jacobian dXYZ/dc.
        double X = 0, Y = 0, Z = 0;
        double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (int i = 0; i < B.N; ++i) {
            double t = tOf(B.lam[i]);
            double p = c[0] * t * t + c[1] * t + c[2];
            double s = sigmoid(p), ds = dSigmoid(p);
            X += s * B.wX[i]; Y += s * B.wY[i]; Z += s * B.wZ[i];
            double dc[3] = {t * t, t, 1.0};
            for (int j = 0; j < 3; ++j) {
                J[0][j] += ds * dc[j] * B.wX[i];
                J[1][j] += ds * dc[j] * B.wY[i];
                J[2][j] += ds * dc[j] * B.wZ[i];
            }
        }
        double rX = X - tX, rY = Y - tY, rZ = Z - tZ;
        if (rX * rX + rY * rY + rZ * rZ < 1e-14) break;
        // Solve J·Δ = residual (3×3) via Cramer's rule.
        double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
                   - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
                   + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
        if (std::fabs(det) < 1e-18) break;
        double rhs[3] = {rX, rY, rZ};
        double d[3];
        for (int col = 0; col < 3; ++col) {
            double M[3][3];
            for (int a = 0; a < 3; ++a) for (int bb = 0; bb < 3; ++bb) M[a][bb] = J[a][bb];
            for (int a = 0; a < 3; ++a) M[a][col] = rhs[a];
            double dc = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                      - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                      + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
            d[col] = dc / det;
        }
        // Damped step for stability on saturated colours.
        const double relax = 1.0;
        c[0] -= relax * d[0]; c[1] -= relax * d[1]; c[2] -= relax * d[2];
    }
    return c;
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

// Build a reflectance Spectrum from a linear-sRGB triple (Jakob-Hanika fit).
inline Spectrum rgbToReflectanceJH(double r, double g, double b) {
    r = std::clamp(r, 0.0, 1.0); g = std::clamp(g, 0.0, 1.0); b = std::clamp(b, 0.0, 1.0);
    std::array<double, 3> c = upsample::fit(r, g, b);
    return [c](double lambda) { return upsample::reflAt(c, lambda); };
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
