// Spectral power / reflectance distributions and emission importance sampling.
// Phase 0 uses std::function for flexibility (CPU only). A POD/tagged form will
// replace this when we port the hot loop to GPU.
#pragma once
#include <functional>
#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>
#include <string>
#include <memory>
#include "color.h"
#include "rng.h"

using Spectrum = std::function<double(double)>; // lambda (nm) -> value

// --- Shared-callable spectra (spectrum IDENTITY) ------------------------------
// A Spectrum is a std::function, so two spectra built from the same source text are
// two independent closures with no way to tell they compute the same curve. That
// matters for performance, not aesthetics: the backward renderer refills a per-sample
// table of SPD(lambda) for EVERY emitter, so a room lit by 256 panels that all say
// `spd preset:bb4000` pays 256 Planck evaluations per wavelength per sample to obtain
// 256 copies of one number. (Measured: at 256 lights that table, not the shadow rays,
// is what is left of the many-lights cost on the CPU once the light BVH is in.)
//
// `sharedSpectrum` wraps a curve in a reference-counted forwarder, so every Spectrum
// copied from one call shares one underlying callable and `spectrumIdentity` returns
// the same non-null pointer for all of them. Equal identity is then a PROOF that two
// spectra are the same function — not a heuristic like comparing sampled tables — and
// the renderer may evaluate one and copy the result to the others bit-for-bit.
// Identity is null for any spectrum not built this way, and null is never assumed
// equal to anything, so the fallback is simply the old per-emitter evaluation.
struct SharedSpectrum {
    std::shared_ptr<const Spectrum> fn;
    double operator()(double w) const { return (*fn)(w); }
};
inline Spectrum sharedSpectrum(Spectrum s) {
    if (s.target<SharedSpectrum>()) return s;          // already shared: don't nest
    return SharedSpectrum{std::make_shared<const Spectrum>(std::move(s))};
}

// A shared curve times a constant. This exists because the constant is exactly what
// makes N otherwise-identical lights different: `lumens 78` normalises each emitter's
// SPD by its own factor (ftsl absPower), so without this the 256 panels of one fixture
// would be 256 unrelated closures again. Keeping the factor OUTSIDE the shared base
// means the renderer can evaluate the base once and recover every emitter's value as
// base*k — and because that is literally how operator() computes it, the recovered
// value is bit-identical, not merely equal to rounding.
struct ScaledSpectrum {
    std::shared_ptr<const Spectrum> fn;
    double k;
    double operator()(double w) const { return (*fn)(w) * k; }
};
inline Spectrum scaledSpectrum(const Spectrum& base, double k) {
    if (const SharedSpectrum* s = base.target<SharedSpectrum>()) return ScaledSpectrum{s->fn, k};
    if (const ScaledSpectrum* s = base.target<ScaledSpectrum>())
        // Do NOT fold k into s->k: (v*k0)*k and v*(k0*k) can differ in the last bit,
        // and this must reproduce the composed curve exactly. Share the composite whole.
        return ScaledSpectrum{std::make_shared<const Spectrum>(base), k};
    return ScaledSpectrum{std::make_shared<const Spectrum>(base), k};
}

// Decompose `s` into a shared base curve and a constant factor such that
// s(w) == (*base)(w) * factor holds BIT-FOR-BIT. Returns null for a spectrum that was
// not built through sharedSpectrum/scaledSpectrum — the caller must then treat it as
// its own unique curve, which is always safe (it just shares nothing).
inline const Spectrum* spectrumBase(const Spectrum& s, double& factor) {
    if (const ScaledSpectrum* t = s.target<ScaledSpectrum>()) { factor = t->k;  return t->fn.get(); }
    if (const SharedSpectrum* t = s.target<SharedSpectrum>()) { factor = 1.0;   return t->fn.get(); }
    factor = 1.0; return nullptr;
}

// --- Builtins ---------------------------------------------------------------
inline Spectrum constantSpectrum(double v) { return [v](double) { return v; }; }

// Planck spectral radiance at temperature `kelvin` and wavelength `lambdaNm`
// (SI, W·m⁻³·sr⁻¹ up to the usual constant). Free function so hot loops
// (per-voxel volumetric blackbody emission, fire) can evaluate it inline
// without constructing a std::function per point.
inline double blackbodyRadiance(double kelvin, double lambdaNm) {
    const double h = 6.62607015e-34, c = 2.99792458e8, kb = 1.380649e-23;
    double l = lambdaNm * 1e-9;
    double e = std::exp((h * c) / (l * kb * kelvin)) - 1.0;
    return (2.0 * h * c * c) / (std::pow(l, 5.0) * e);
}

inline Spectrum blackbody(double kelvin) {
    // Planck's law (relative). Returns spectral radiance up to a constant.
    return [kelvin](double lambdaNm) { return blackbodyRadiance(kelvin, lambdaNm); };
}

// Blackbody radiance normalized against a fixed 6500 K / 560 nm reference, so a
// volumetric emitter's `emission_scale ~ O(1)` maps to a usable glow for typical
// flame temperatures (~1000–3000 K) while PRESERVING the physical relative
// brightness (Stefan–Boltzmann T⁴) and hue shift (Wien's law) between voxels of
// different temperature. The denominator is a compile-once constant.
inline double blackbodyEmissionRadiance(double kelvin, double lambdaNm) {
    static const double kRef = blackbodyRadiance(6500.0, 560.0);
    return blackbodyRadiance(kelvin, lambdaNm) / kRef;
}

// Smooth "colored wall" reflectances (plausible, not measured — fine for Phase 0).
inline Spectrum whiteWall(double r = 0.75) { return constantSpectrum(r); }
inline Spectrum redWall() {
    return [](double w) { return 0.05 + 0.70 / (1.0 + std::exp(-(w - 600.0) * 0.08)); };
}
inline Spectrum greenWall() {
    return [](double w) { double t = (w - 550.0) / 45.0; return 0.05 + 0.70 * std::exp(-0.5 * t * t); };
}

// A Gaussian band centred at `center` nm with std-dev `sigma`, peak `amp`. Used
// for fluorescent emission spectra (the re-radiated glow) and other smooth bands.
inline Spectrum gaussianBand(double center, double sigma, double amp = 1.0) {
    return [=](double w) { double t = (w - center) / sigma; return amp * std::exp(-0.5 * t * t); };
}

// A logistic roll-off high below `edge` nm and low above it (slope>0), scaled by
// `amp`. Used for fluorescent excitation/absorption: short wavelengths (blue/UV)
// excite the dye, long wavelengths pass through it.
inline Spectrum shortPass(double edge, double slope, double amp = 1.0) {
    return [=](double w) { return amp / (1.0 + std::exp((w - edge) * slope)); };
}

// --- Dispersion: wavelength-dependent index of refraction -------------------
// Sellmeier equation: n^2(l) = 1 + sum_i Bi*l^2 / (l^2 - Ci), with l in micrometres.
// Single-wavelength photons make this "free" dispersion — each lambda bends by its
// own n, so a glass object separates colours with no special-casing.
inline Spectrum sellmeier(double B1, double B2, double B3, double C1, double C2, double C3) {
    return [=](double lambdaNm) {
        double l2 = (lambdaNm * 1e-3) * (lambdaNm * 1e-3); // um^2
        double n2 = 1.0 + B1 * l2 / (l2 - C1) + B2 * l2 / (l2 - C2) + B3 * l2 / (l2 - C3);
        return std::sqrt(n2 > 1.0 ? n2 : 1.0);
    };
}
// Cauchy dispersion n(l) = A + B/l^2 (l in micrometres). A compact two-term fit for
// weakly-dispersive materials (water, ice, common plastics) where full Sellmeier
// coefficients are overkill; captures the visible-range trend to ~1e-3.
inline Spectrum cauchy(double A, double B) {
    return [=](double lambdaNm) {
        double um = lambdaNm * 1e-3;
        return A + B / (um * um);
    };
}

// Constant (non-dispersive) index of refraction.
inline Spectrum iorConstant(double n) { return [n](double) { return n; }; }

// NOTE: the per-glass dispersion DATA (Sellmeier/Cauchy coefficients for BK7, SF10,
// fused silica, sapphire, diamond, water, ice, acrylic, polycarbonate) used to live
// here as `iorBK7()` etc. + a `resolveGlassIor()` table. That DATA now lives in
// external files (data/glass/*.glass) and is loaded by `resolveGlassIor()` in
// spectral_library.h, which feeds the coefficients to the native `sellmeier()` /
// `cauchy()` evaluators above. Only the data moved out; the evaluators stay here.

// Piecewise-linear measured curve from (wavelength nm, value) pairs. The pairs are
// sorted by wavelength at build time; sampling clamps to the endpoints outside the
// measured range and linearly interpolates within. This is the ingestion point for
// measured reflectance/SPD data (FTSL `table { 400:0.05 450:0.12 ... }`).
// A named type rather than a lambda so that a consumer holding only the Spectrum can recover
// its knots (spectrumKnots) -- the device does, to read a table exactly instead of off its grid.
struct TabulatedLinear {
    std::vector<std::pair<double, double>> pairs;   // sorted by wavelength
    double operator()(double w) const {
        if (pairs.empty()) return 0.0;
        if (w <= pairs.front().first) return pairs.front().second;
        if (w >= pairs.back().first)  return pairs.back().second;
        // Binary search for the bracketing interval.
        size_t lo = 0, hi = pairs.size() - 1;
        while (lo + 1 < hi) {
            size_t mid = (lo + hi) / 2;
            (pairs[mid].first <= w ? lo : hi) = mid;
        }
        double w0 = pairs[lo].first, w1 = pairs[lo + 1].first;
        double v0 = pairs[lo].second, v1 = pairs[lo + 1].second;
        double f = (w1 > w0) ? (w - w0) / (w1 - w0) : 0.0;
        return v0 + (v1 - v0) * f;
    }
};
inline Spectrum tabulatedSpectrum(std::vector<std::pair<double, double>> pairs) {
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return TabulatedLinear{std::move(pairs)};
}

// Monotone piecewise-cubic (Fritsch-Carlson / PCHIP) measured curve — the opt-in
// `interp=cubic` alternative to the linear `tabulatedSpectrum` above. It is C1-smooth
// yet SHAPE-PRESERVING: unlike a natural/plain cubic spline it never overshoots, so an
// interpolated value stays within its two bracketing samples (a reflectance/absorption
// curve can't ring negative or exceed its neighbours — which a plain spline would, and
// which is unphysical). Best for sparse hand-authored control points (`table { … }`)
// where linear kinks are visible; for dense measured data it matches linear closely.
// Reduces to linear for 2 points; clamps to the endpoints outside the measured range
// (no extrapolation — same tail policy as tabulatedSpectrum).
// The monotone-cubic table's evaluator: knots x, values y, Hermite tangents m. Named for the
// same reason as TabulatedLinear.
struct TabulatedMono {
    std::vector<double> x, y, m;
    double operator()(double w) const {
        size_t n = x.size();
        if (n == 0) return 0.0;
        if (n == 1 || w <= x.front()) return y.front();
        if (w >= x.back()) return y.back();
        size_t lo = 0, hi = n - 1;
        while (lo + 1 < hi) { size_t mid = (lo + hi) / 2; (x[mid] <= w ? lo : hi) = mid; }
        double hlen = x[lo + 1] - x[lo];
        double t = (w - x[lo]) / hlen, t2 = t * t, t3 = t2 * t;
        double h00 =  2 * t3 - 3 * t2 + 1;   // Hermite basis
        double h10 =      t3 - 2 * t2 + t;
        double h01 = -2 * t3 + 3 * t2;
        double h11 =      t3 -     t2;
        return h00 * y[lo] + h10 * hlen * m[lo] + h01 * y[lo + 1] + h11 * hlen * m[lo + 1];
    }
};
inline Spectrum tabulatedSpectrumMono(std::vector<std::pair<double, double>> pairs) {
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    // Collapse exact-duplicate wavelengths (a zero-width interval would divide by 0).
    pairs.erase(std::unique(pairs.begin(), pairs.end(),
                    [](const auto& a, const auto& b) { return a.first == b.first; }),
                pairs.end());
    size_t n = pairs.size();
    std::vector<double> x(n), y(n), m(n, 0.0);
    for (size_t i = 0; i < n; ++i) { x[i] = pairs[i].first; y[i] = pairs[i].second; }
    if (n >= 2) {
        std::vector<double> h(n - 1), d(n - 1);           // interval widths, secant slopes
        for (size_t i = 0; i + 1 < n; ++i) { h[i] = x[i + 1] - x[i]; d[i] = (y[i + 1] - y[i]) / h[i]; }
        // Interior tangents: weighted harmonic mean of the two adjacent secants, forced
        // to 0 at a local extremum (sign change) — this is what kills overshoot.
        for (size_t i = 1; i + 1 < n; ++i) {
            if (d[i - 1] * d[i] <= 0.0) { m[i] = 0.0; continue; }
            double w1 = 2.0 * h[i] + h[i - 1];
            double w2 = h[i] + 2.0 * h[i - 1];
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
        }
        // Endpoints: one-sided 3-point estimate with the standard shape-preserving limiter.
        auto endTangent = [](double h0, double h1, double d0, double d1) {
            double m0 = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
            if (m0 * d0 <= 0.0) return 0.0;                              // don't flip sign
            if (d0 * d1 <= 0.0 && std::fabs(m0) > 3.0 * std::fabs(d0)) return 3.0 * d0; // cap
            return m0;
        };
        if (n == 2) { m[0] = d[0]; m[1] = d[0]; }
        else {
            m[0]     = endTangent(h[0],     h[1],     d[0],     d[1]);
            m[n - 1] = endTangent(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
        }
    }
    return TabulatedMono{std::move(x), std::move(y), std::move(m)};
}

// The knots of a piecewise-linear curve that reproduces `s`, for a consumer that reads curves off
// a grid (the device's SPEC_N tables, 4.9 nm apart) and must not lose what falls between its
// points: a window a few nm wide, an emission line. Exact for a linear table (TabulatedLinear),
// also when shared or scaled (SharedSpectrum, ScaledSpectrum); a monotone-cubic table
// (TabulatedMono) is split into kMonoSub linear pieces per interval. False for anything else --
// a constant, an analytic curve (smooth on the grid's scale), a lambda nobody can see into -- and
// the caller keeps its grid. `x` comes back ascending, `y` the matching values. `id`, when given,
// receives the table's own address and `scale` the factor folded into `y`: two spectra with the
// same pair have the same knots, so a consumer can store them once. `kMonoSub`: the linear pieces a
// monotone-cubic table's span is split into (1: its own knots).
inline bool spectrumKnots(const Spectrum& s, std::vector<double>& x, std::vector<double>& y,
                          const void** id = nullptr, double* scale = nullptr, int kMonoSub = 16) {
    x.clear(); y.clear();
    double k = 1.0;
    const Spectrum* cur = &s;
    for (int guard = 0; guard < 16 && cur && *cur; ++guard) {   // see through shares and scales
        if (const SharedSpectrum* t = cur->target<SharedSpectrum>()) { cur = t->fn.get(); continue; }
        if (const ScaledSpectrum* t = cur->target<ScaledSpectrum>()) { k *= t->k; cur = t->fn.get(); continue; }
        break;
    }
    if (!cur || !*cur) return false;
    if (id) *id = nullptr;
    if (scale) *scale = k;
    if (const TabulatedLinear* t = cur->target<TabulatedLinear>()) {
        if (t->pairs.size() < 2) return false;
        for (const auto& p : t->pairs) { x.push_back(p.first); y.push_back(p.second * k); }
        if (id) *id = t;
        return true;
    }
    if (const TabulatedMono* t = cur->target<TabulatedMono>()) {
        const size_t n = t->x.size();
        if (n < 2) return false;
        for (size_t i = 0; i + 1 < n; ++i)
            for (int j = 0; j < kMonoSub; ++j) {
                const double w = t->x[i] + (t->x[i + 1] - t->x[i]) * (double)j / kMonoSub;
                x.push_back(w); y.push_back((*t)(w) * k);
            }
        x.push_back(t->x[n - 1]); y.push_back(t->y[n - 1] * k);
        if (id) *id = t;
        return true;
    }
    return false;
}

// RGB -> reflectance upsampling now lives in src/upsample.h
// (`rgbToReflectanceJH`, a Jakob-Hanika 2019 sigmoid fit that round-trips linear
// sRGB under D65). The earlier three-lobe placeholder was removed once the
// proper fit landed; `rgb r g b` in FTSL routes through the JH upsampler.

// The spectral carriers' uniform bins (JSpec::K; SpecThr's cells split them, SpecCells), and so the
// wavelengths at which the question "does this vary over the band?" is asked.
constexpr int kBandProbeBins = 24;
// Bin k's centre on that grid (JSpec::lamOf, the mode-J merge weights' tables).
inline double bandBinLambda(int k) {
    return LAMBDA_MIN + (k + 0.5) * (LAMBDA_MAX - LAMBDA_MIN) / (double)kBandProbeBins;
}
// The bin of that grid `lam` (nm) falls in, the end bins below and above the band.
inline int bandBinOf(double lam) {
    const int b = (int)((lam - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN) * kBandProbeBins);
    return b < 0 ? 0 : (b >= kBandProbeBins ? kBandProbeBins - 1 : b);
}

// --- Emission importance sampling ------------------------------------------
// Precomputes a CDF over [LAMBDA_MIN, LAMBDA_MAX] to sample lambda ~ SPD, and
// exposes the integral so photon weights stay physically consistent.
struct EmissionSampler {
    std::vector<double> cdf;   // size N+1
    double step = 1.0;
    double integral = 0.0;     // integral of SPD over range
    // The density's mass in each of the uniform kBandProbeBins bins, which mode J's support rule divides
    // by (JSpecRun::supportShare, the device's c_jBinQ; mode M's and S's take SpecCells::mass). Filled by
    // build(), once, so a walk reads it rather than walking the CDF.
    double binMass[kBandProbeBins] = {};

    void build(const Spectrum& spd, double stepNm = 1.0) {
        step = stepNm;
        int n = static_cast<int>((LAMBDA_MAX - LAMBDA_MIN) / step);
        cdf.assign(n + 1, 0.0);
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            double w = LAMBDA_MIN + (i + 0.5) * step;
            acc += std::max(0.0, spd(w)) * step;
            cdf[i + 1] = acc;
        }
        integral = acc;
        if (acc > 0) for (auto& c : cdf) c /= acc; // normalise to [0,1]
        for (int k = 0; k < kBandProbeBins; ++k) {
            const double lo = LAMBDA_MIN + k * (LAMBDA_MAX - LAMBDA_MIN) / (double)kBandProbeBins;
            const double hi = LAMBDA_MIN + (k + 1) * (LAMBDA_MAX - LAMBDA_MIN) / (double)kBandProbeBins;
            binMass[k] = cdfAt(hi) - cdfAt(lo);
        }
    }

    // Invert the CDF at a supplied uniform u in [0,1). Sets pdf (per nm) and
    // returns lambda. Factored out of sample() so hero-wavelength sampling can
    // draw stratified secondaries from the *same* CDF without a fresh rng draw.
    double sampleAt(double u, double& pdf) const {
        int lo = 0, hi = static_cast<int>(cdf.size()) - 1;
        while (lo + 1 < hi) { int mid = (lo + hi) / 2; (cdf[mid] <= u ? lo : hi) = mid; }
        double c0 = cdf[lo], c1 = cdf[lo + 1];
        double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
        double w = LAMBDA_MIN + (lo + frac) * step;
        double binProb = c1 - c0;         // probability mass of this bin
        pdf = binProb / step;             // convert to density per nm
        return w;
    }

    // Returns lambda and sets pdf (per nm). p(lambda) = SPD(lambda)/integral.
    double sample(Pcg32& rng, double& pdf) const {
        return sampleAt(rng.uniform(), pdf);
    }

    // The density (per nm) sample() draws `lambda` with: its bin's probability mass over the
    // bin width. 0 outside the range.
    double pdfAt(double lambda) const {
        const int n = static_cast<int>(cdf.size()) - 1;
        if (n <= 0) return 0.0;
        const double t = (lambda - LAMBDA_MIN) / step;
        if (!(t >= 0.0) || t >= (double)n) return 0.0;
        const int b = static_cast<int>(t);
        return (cdf[b + 1] - cdf[b]) / step;
    }

    // The density's mass over [a, b] (nm), and in `m1` its first moment there (the integral of lambda
    // p(lambda)): constant over each `step` nm, as sample() draws it.
    double massMoment(double a, double b, double& m1) const {
        m1 = 0.0;
        const int n = static_cast<int>(cdf.size()) - 1;
        if (n <= 0 || !(b > a)) return 0.0;
        const int j0 = std::max(0, static_cast<int>(std::floor((a - LAMBDA_MIN) / step)));
        const int j1 = std::min(n - 1, static_cast<int>(std::floor((b - LAMBDA_MIN) / step)));
        double m = 0.0;
        for (int j = j0; j <= j1; ++j) {
            const double x0 = std::max(a, LAMBDA_MIN + j * step), x1 = std::min(b, LAMBDA_MIN + (j + 1) * step);
            if (!(x1 > x0)) continue;
            const double d = (cdf[(size_t)j + 1] - cdf[(size_t)j]) / step;
            m += d * (x1 - x0);
            m1 += d * 0.5 * (x1 * x1 - x0 * x0);
        }
        return m;
    }

    // The probability that sample() draws below `lambda`: the CDF, linear within a bin as the
    // draw is. 0 below the range, 1 above it.
    double cdfAt(double lambda) const {
        const int n = static_cast<int>(cdf.size()) - 1;
        if (n <= 0) return 0.0;
        const double t = (lambda - LAMBDA_MIN) / step;
        if (!(t > 0.0)) return 0.0;
        if (t >= (double)n) return 1.0;
        const int b = static_cast<int>(t);
        return cdf[b] + (t - b) * (cdf[b + 1] - cdf[b]);
    }
};

// ---- The cells mode M's and mode S's spectral carrier is read in (SPECGATHER-BIN-RESOLUTION) ----
// Their camera walk carries its spectral factors over a grid (SpecThr; the device's DSpecThr), and a
// photon reads the cell its own wavelength falls in. Read at one wavelength, a cell gives its value to
// every photon in it, and its whole walk density to whichever side of a support edge that wavelength
// is on. So the grid is the kBandProbeBins uniform bins, each split where a factor the walk folds is
// not near linear across it (chooseSpecCuts) -- a scene with no such factor has exactly the uniform bins --
// and each cell is read where the light's wavelengths in it lie (setMass, SPECGATHER-CELL-CENTROID).
constexpr int kSpecCutsMax = 16;
// A walk whose branches' probability, at some cell, is under this share of its likeliest cell's is partial
// (SpecThr::p, the device's DSpecThr): the exact ratio p(lamP) / p(lamC) is unbiased only where walks drawn
// at lamC do take the branches, and a branch a thousandth as likely is, for a finite render, one they never
// take -- its light would come only from the walks drawn elsewhere, weighed as if they were all. The
// averaged probability is unbiased whatever the spread. Taken on the walk's accumulated product, so three
// coins each under the ratio but compounding past it count as one would.
constexpr double kCoinRare = 1e-3;
constexpr int kSpecCellsMax = kBandProbeBins + kSpecCutsMax;
struct SpecCells {
    int n = kBandProbeBins;
    double lo[kSpecCellsMax], hi[kSpecCellsMax];   // each cell's ends, ascending, the last's hi LAMBDA_MAX
    double lam[kSpecCellsMax];                     // the wavelength it is read at (setMass)
    double mass[kSpecCellsMax];                    // the walk density's mass in it (setMass)
    unsigned char first[kBandProbeBins];           // each uniform bin's first cell...
    unsigned char count[kBandProbeBins];           // ...and how many it is split into
    SpecCells() { build({}); }

    // The grid with cuts at `cuts` (nm): one cell a bin and no cut but the bins' own ends where there
    // are none. A cut outside the band, or within 0.05 nm of a bin's end or of a cut kept, is dropped
    // (a step that narrow is one edge); past kSpecCutsMax the rest are (Scene::buildSpecCells orders
    // them, the costliest first). An unsplit bin keeps the uniform grid's own expressions for its
    // ends. Each cell is read at its middle until setMass places it.
    void build(std::vector<double> cuts) {
        const double W = (LAMBDA_MAX - LAMBDA_MIN) / (double)kBandProbeBins;
        std::vector<double> keep;
        for (double c : cuts) {
            if ((int)keep.size() == kSpecCutsMax) break;
            if (!(c > LAMBDA_MIN && c < LAMBDA_MAX)) continue;
            const double t = (c - LAMBDA_MIN) / W;
            if (std::fabs(t - std::round(t)) * W < 0.05) continue;
            bool twin = false;   // (not `near`: <windows.h> defines that away)
            for (double k : keep) twin = twin || std::fabs(k - c) < 0.05;
            if (!twin) keep.push_back(c);
        }
        std::sort(keep.begin(), keep.end());
        n = 0;
        size_t ci = 0;
        for (int k = 0; k < kBandProbeBins; ++k) {
            const double b0 = LAMBDA_MIN + k * (LAMBDA_MAX - LAMBDA_MIN) / (double)kBandProbeBins;
            const double b1 = LAMBDA_MIN + (k + 1) * (LAMBDA_MAX - LAMBDA_MIN) / (double)kBandProbeBins;
            first[k] = (unsigned char)n;
            double a = b0;
            while (ci < keep.size() && keep[ci] < b1) {
                lo[n] = a; hi[n] = keep[ci]; lam[n] = 0.5 * (a + keep[ci]); ++n;
                a = keep[ci++];
            }
            lo[n] = a; hi[n] = b1;
            lam[n] = a == b0 ? bandBinLambda(k) : 0.5 * (a + b1);
            ++n;
            count[k] = (unsigned char)(n - first[k]);
        }
        for (int i = 0; i < n; ++i) mass[i] = 0.0;
    }
    // The walk density's mass in each cell, as EmissionSampler::binMass has it in each bin, and the
    // wavelength the cell is read at: the density's centroid in it, where the light's power in the cell
    // sits (a line's own wavelength), so a factor near linear across the cell -- what chooseSpecCuts
    // leaves it -- gives its photons the value their wavelengths average to; the middle where the
    // cell holds none of it.
    void setMass(const EmissionSampler& walk) {
        for (int i = 0; i < n; ++i) {
            mass[i] = walk.cdfAt(hi[i]) - walk.cdfAt(lo[i]);
            double m1 = 0.0;
            const double m = walk.massMoment(lo[i], hi[i], m1);
            if (m > 0.0) lam[i] = std::min(hi[i], std::max(lo[i], m1 / m));
        }
    }
    // The cell `w` (nm) falls in; the end cells below and above the band. A cut belongs to the cell
    // above it.
    int cellOf(double w) const {
        const int b = bandBinOf(w);
        int c = first[b];
        for (int e = c + count[b] - 1; c < e && w >= hi[c]; ++c) {}
        return c;
    }
    bool split() const { return n != kBandProbeBins; }
};

// A spectrum the camera walk folds, for the carrier's cuts (Scene::buildSpecCells). `support`: its zeros
// are holes in the walk's support (a reflectance, a filter's transmittance), not a coefficient's value
// (an absorption, an index).
struct SpecCutCurve { const Spectrum* s; bool support; };

// The carrier's cuts for `curves`: from the uniform bins, it splits the cell worst for any curve at the
// point that lowers the worse of its two halves most, until every cell is within `tol` or kSpecCutsMax
// cuts are taken. A cell's error for a curve, over the curve's largest magnitude: the larger of how far
// the curve is from linear across it (the RMS residual of its own least-squares line) and its mean from
// its value at the cell's wavelength; and for a support curve, the share of the cell on the other side
// of a zero from that wavelength. Linear across a cell, a factor read at its centre is right to first
// order for every photon's weight; a step, a bend or a narrow band is not, wherever in the cell it
// sits (the centre's bias alone misses a step centred in its cell). The curves are sampled kSpecCutSub times a bin, so a bin's ends are sample
// boundaries; a cut snaps to a table's knot within a sample of it, so a step there reads exactly.
// Returns the cuts (nm, in no order); `left` gets the worst error the budget left (0 when none did).
constexpr int kSpecCutSub = 80;
inline std::vector<double> chooseSpecCuts(const std::vector<SpecCutCurve>& curves, double tol, double& left) {
    left = 0.0;
    const int N = kBandProbeBins * kSpecCutSub;
    const double W = (LAMBDA_MAX - LAMBDA_MIN) / (double)kBandProbeBins, dl = W / kSpecCutSub;
    // Each curve's prefix sums over the samples: of f, f^2 and j f (its linear fit), and of its zeros.
    struct Curve { const Spectrum* s; bool support; double top; std::vector<double> P, Q, J, Z, knots; };
    std::vector<Curve> cs;
    for (const SpecCutCurve& c : curves) {
        if (!c.s || !*c.s) continue;
        Curve cv{c.s, c.support, 0.0, std::vector<double>((size_t)N + 1, 0.0),
                 std::vector<double>((size_t)N + 1, 0.0), std::vector<double>((size_t)N + 1, 0.0), {}, {}};
        if (c.support) cv.Z.assign((size_t)N + 1, 0.0);
        double f0 = 0.0;
        bool flat = true;
        for (int j = 0; j < N; ++j) {
            const double f = (*c.s)(LAMBDA_MIN + (j + 0.5) * dl);
            if (j == 0) f0 = f;
            else if (f != f0) flat = false;
            cv.top = std::max(cv.top, std::fabs(f));
            cv.P[(size_t)j + 1] = cv.P[(size_t)j] + f;
            cv.Q[(size_t)j + 1] = cv.Q[(size_t)j] + f * f;
            cv.J[(size_t)j + 1] = cv.J[(size_t)j] + (double)j * f;
            if (c.support) cv.Z[(size_t)j + 1] = cv.Z[(size_t)j] + (f == 0.0 ? 1.0 : 0.0);
        }
        if (flat || !(cv.top > 0.0)) continue;
        std::vector<double> x, y;
        if (spectrumKnots(*c.s, x, y, nullptr, nullptr, 1)) cv.knots = x;
        cs.push_back(std::move(cv));
    }
    std::vector<double> cuts;
    if (cs.empty()) return cuts;
    // A cell as sample boundaries [a, b); read at a bin's centre where it is a whole bin, else its middle.
    auto lamOf = [&](int a, int b) {
        return (a % kSpecCutSub == 0 && b == a + kSpecCutSub) ? bandBinLambda(a / kSpecCutSub)
                                                                : LAMBDA_MIN + 0.5 * (a + b) * dl;
    };
    auto err = [&](int a, int b) {
        const double lam = lamOf(a, b), n = (double)(b - a);
        const double stt = n * (n * n - 1.0) / 12.0;   // sum of (j - the cell's middle)^2
        double e = 0.0;
        for (const Curve& c : cs) {
            const double at = (*c.s)(lam);
            const double s1 = c.P[(size_t)b] - c.P[(size_t)a], s2 = c.Q[(size_t)b] - c.Q[(size_t)a];
            const double m = s1 / n, stf = (c.J[(size_t)b] - c.J[(size_t)a]) - 0.5 * (a + b - 1) * s1;
            const double ss = s2 - n * m * m - (stt > 0.0 ? stf * stf / stt : 0.0);   // the fit's residual
            double ec = std::max(std::fabs(m - at), std::sqrt(std::max(0.0, ss) / n)) / c.top;
            if (c.support) {
                const double z = (c.Z[(size_t)b] - c.Z[(size_t)a]) / n;
                ec += (at == 0.0) ? 1.0 - z : z;
            }
            e = std::max(e, ec);
        }
        return e;
    };
    std::vector<std::pair<int, int>> cells;
    for (int k = 0; k < kBandProbeBins; ++k) cells.push_back({k * kSpecCutSub, (k + 1) * kSpecCutSub});
    for (int taken = 0;; ++taken) {
        size_t worst = 0;
        double we = -1.0;
        for (size_t i = 0; i < cells.size(); ++i) {
            const double e = err(cells[i].first, cells[i].second);
            if (e > we) { we = e; worst = i; }
        }
        if (we <= tol) return cuts;
        if (taken == kSpecCutsMax) { left = we; return cuts; }
        const int a = cells[worst].first, b = cells[worst].second;
        int best = -1;
        double be = 0.0;
        for (int c = a + 1; c < b; ++c) {
            const double e = std::max(err(a, c), err(c, b));
            if (best < 0 || e < be) { best = c; be = e; }
        }
        if (best < 0) { left = we; return cuts; }   // a one-sample cell: nothing finer to cut
        double at = LAMBDA_MIN + best * dl;
        for (const Curve& c : cs)
            for (double kx : c.knots)
                if (std::fabs(kx - at) < dl && kx > LAMBDA_MIN + a * dl && kx < LAMBDA_MIN + b * dl) at = kx;
        cuts.push_back(at);
        cells[worst].second = best;
        cells.insert(cells.begin() + (std::ptrdiff_t)worst + 1, {best, b});
    }
}
