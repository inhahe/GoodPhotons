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
#include "color.h"
#include "rng.h"

using Spectrum = std::function<double(double)>; // lambda (nm) -> value

// --- Builtins ---------------------------------------------------------------
inline Spectrum constantSpectrum(double v) { return [v](double) { return v; }; }

inline Spectrum blackbody(double kelvin) {
    // Planck's law (relative). Returns spectral radiance up to a constant.
    return [kelvin](double lambdaNm) {
        const double h = 6.62607015e-34, c = 2.99792458e8, kb = 1.380649e-23;
        double l = lambdaNm * 1e-9;
        double e = std::exp((h * c) / (l * kb * kelvin)) - 1.0;
        return (2.0 * h * c * c) / (std::pow(l, 5.0) * e);
    };
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
