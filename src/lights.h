// Built-in light source spectral power distributions.
// Emission already samples an arbitrary SPD, so a light is fully described by its
// spectral envelope. These builders return Spectrum (lambda nm -> relative power).
//
// The color-temperature / incandescent sources are physically exact (Planck).
// The daylight and artificial (LED / fluorescent) models are plausible analytic
// approximations, not measured data — real measured SPDs (D65, solar, specific
// lamps) are intended to load from data files via the Python tooling later.
#pragma once
#include <cmath>
#include "spectrum.h"
#include "color.h"

inline double gaussLobe(double x, double mu, double sigma) {
    double t = (x - mu) / sigma;
    return std::exp(-0.5 * t * t);
}

// Planckian (thermal) radiator at temperature T (Kelvin). Exact.
// Covers "specify a color temperature" and incandescent/tungsten sources.
inline Spectrum colorTemperature(double kelvin) { return blackbody(kelvin); }

// CIE Standard Illuminant A (tungsten filament), exactly Planckian at 2856 K.
inline Spectrum illuminantA() { return blackbody(2856.0); }

// Daylight / sunlight approximation. Real daylight is the CIE D-series; here we
// approximate with a Planckian at the requested correlated colour temperature
// (D65 ~ 6504 K, noon sun ~ 5778 K). Good enough for mood; swap for measured
// data when the SPD loader lands.
inline Spectrum daylight(double kelvin = 6504.0) { return blackbody(kelvin); }
inline Spectrum sunlight() { return blackbody(5778.0); }

// White LED: a blue pump peak plus a broad phosphor hump. `warm` in [0,1] shifts
// the phosphor redward and lowers the blue peak (0 = cool, 1 = warm).
inline Spectrum ledWhite(double warm = 0.3) {
    double bluePeak = 460.0;
    double blueAmp  = 1.0 - 0.4 * warm;
    double phosMu   = 560.0 + 40.0 * warm;
    return [=](double w) {
        return blueAmp * gaussLobe(w, bluePeak, 18.0)
             + 1.0     * gaussLobe(w, phosMu, 90.0);
    };
}

// Trichromatic fluorescent: a low phosphor continuum with mercury emission lines.
// Illustrative model of the spiky spectrum, not a measured F-series.
inline Spectrum fluorescent() {
    return [](double w) {
        double cont = 0.08 + 0.05 * gaussLobe(w, 560.0, 120.0);
        double lines = 1.00 * gaussLobe(w, 436.0, 4.0)   // Hg blue
                     + 1.10 * gaussLobe(w, 546.0, 4.0)   // Hg green
                     + 0.75 * gaussLobe(w, 611.0, 6.0)   // phosphor red
                     + 0.35 * gaussLobe(w, 488.0, 6.0);
        return cont + lines;
    };
}

// Scale an SPD so its integral over the visible range is 1 (comparable brightness
// across illuminants). Auto-exposure hides absolute scale, but this keeps relative
// intensities sane when mixing multiple lights.
inline Spectrum normalizePower(const Spectrum& spd) {
    double s = 0;
    for (double w = LAMBDA_MIN; w <= LAMBDA_MAX; w += 1.0) s += std::max(0.0, spd(w));
    double inv = (s > 0) ? 1.0 / s : 1.0;
    return [spd, inv](double w) { return spd(w) * inv; };
}
