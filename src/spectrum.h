// Spectral power / reflectance distributions and emission importance sampling.
// Phase 0 uses std::function for flexibility (CPU only). A POD/tagged form will
// replace this when we port the hot loop to GPU.
#pragma once
#include <functional>
#include <vector>
#include <cmath>
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

    // Returns lambda and sets pdf (per nm). p(lambda) = SPD(lambda)/integral.
    double sample(Pcg32& rng, double& pdf) const {
        double u = rng.uniform();
        int lo = 0, hi = static_cast<int>(cdf.size()) - 1;
        while (lo + 1 < hi) { int mid = (lo + hi) / 2; (cdf[mid] <= u ? lo : hi) = mid; }
        double c0 = cdf[lo], c1 = cdf[lo + 1];
        double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
        double w = LAMBDA_MIN + (lo + frac) * step;
        double binProb = c1 - c0;         // probability mass of this bin
        pdf = binProb / step;             // convert to density per nm
        return w;
    }
};
