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
inline Spectrum tabulatedSpectrum(std::vector<std::pair<double, double>> pairs) {
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return [pairs](double w) -> double {
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
    };
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
    return [x, y, m](double w) -> double {
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
    };
}

// RGB -> reflectance upsampling now lives in src/upsample.h
// (`rgbToReflectanceJH`, a Jakob-Hanika 2019 sigmoid fit that round-trips linear
// sRGB under D65). The earlier three-lobe placeholder was removed once the
// proper fit landed; `rgb r g b` in FTSL routes through the JH upsampler.

// --- Emission importance sampling ------------------------------------------
// Precomputes a CDF over [LAMBDA_MIN, LAMBDA_MAX] to sample lambda ~ SPD, and
// exposes the integral so photon weights stay physically consistent.
struct EmissionSampler {
    std::vector<double> cdf;   // size N+1
    double step = 1.0;
    double integral = 0.0;     // integral of SPD over range

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
};
