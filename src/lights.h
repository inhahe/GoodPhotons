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
#include <string>
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
// Illustrative model of the spiky spectrum, not a measured F-series. For real
// tabulated fluorescent SPDs use the CIE F-series builders below (f2/f7/f11).
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

// ---------------------------------------------------------------------------
// Measured / spectroscopic artificial sources
// ---------------------------------------------------------------------------

// Build a piecewise-linear SPD from evenly-spaced samples starting at `startNm`
// with spacing `stepNm`. Used to embed measured tables (CIE F-series below).
inline Spectrum sampledSPD(double startNm, double stepNm, const std::vector<double>& v) {
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        pairs.emplace_back(startNm + stepNm * static_cast<double>(i), v[i]);
    return tabulatedSpectrum(std::move(pairs));
}

// Sum of narrow emission lines: (center nm, relative peak) pairs, each a Gaussian
// of width `sigma` nm. Gas-discharge lamps (sodium, mercury, metal halide) are
// dominated by such lines; positions and relative strengths are from spectroscopy.
inline Spectrum emissionLines(std::vector<std::pair<double, double>> lines, double sigma = 3.0) {
    return [lines, sigma](double w) {
        double s = 0.0;
        for (const auto& L : lines) s += L.second * gaussLobe(w, L.first, sigma);
        return s;
    };
}

// CIE Standard Illuminant F-series: real tabulated relative SPDs, 380-780 nm at
// 5 nm. F2 = cool white halophosphate (CCT ~4230 K, CRI ~64); F7 = broadband
// "daylight" fluorescent, a D65 simulator (CCT ~6500 K, CRI ~90); F11 = narrow-band
// triphosphor (CCT ~4000 K, CRI ~83). Source: CIE 15 tabulated illuminant data,
// verified against colour-science's transcription and mirrored (for the future
// data-file loader) in data/spd/cie_f2.csv / cie_f7.csv / cie_f11.csv.
inline Spectrum fluorescentF2() {
    static const std::vector<double> d = {
        1.18, 1.48, 1.84, 2.15, 3.44, 15.69, 3.85, 3.74, 4.19, 4.62,   // 380-425
        5.06, 34.98, 11.81, 6.27, 6.63, 6.93, 7.19, 7.40, 7.54, 7.62,  // 430-475
        7.65, 7.62, 7.62, 7.45, 7.28, 7.15, 7.05, 7.04, 7.16, 7.47,    // 480-525
        8.04, 8.88, 10.01, 24.88, 16.64, 14.59, 16.16, 17.56, 18.62, 21.47, // 530-575
        22.79, 19.29, 18.66, 17.73, 16.54, 15.21, 13.80, 12.36, 10.95, 9.65, // 580-625
        8.40, 7.32, 6.31, 5.43, 4.68, 4.02, 3.45, 2.96, 2.55, 2.19,    // 630-675
        1.89, 1.64, 1.53, 1.27, 1.10, 0.99, 0.88, 0.76, 0.68, 0.61,    // 680-725
        0.56, 0.54, 0.51, 0.47, 0.47, 0.43, 0.46, 0.47, 0.40, 0.33, 0.27 // 730-780
    };
    return sampledSPD(380.0, 5.0, d);
}
inline Spectrum fluorescentF7() {
    static const std::vector<double> d = {
        2.56, 3.18, 3.84, 4.53, 6.15, 19.37, 7.37, 7.05, 7.71, 8.41,   // 380-425
        9.15, 44.14, 17.52, 11.35, 12.00, 12.58, 13.08, 13.45, 13.71, 13.88, // 430-475
        13.95, 13.93, 13.82, 13.64, 13.43, 13.25, 13.08, 12.93, 12.78, 12.60, // 480-525
        12.44, 12.33, 12.26, 29.52, 17.05, 12.44, 12.58, 12.72, 12.83, 15.46, // 530-575
        16.75, 12.83, 12.67, 12.45, 12.19, 11.89, 11.60, 11.35, 11.12, 10.95, // 580-625
        10.76, 10.42, 10.11, 10.04, 10.02, 10.11, 9.87, 8.65, 7.27, 6.44, // 630-675
        5.83, 5.41, 5.04, 4.57, 4.12, 3.77, 3.46, 3.08, 2.73, 2.47,    // 680-725
        2.25, 2.06, 1.90, 1.75, 1.62, 1.54, 1.45, 1.32, 1.17, 0.99, 0.81 // 730-780
    };
    return sampledSPD(380.0, 5.0, d);
}
inline Spectrum fluorescentF11() {
    static const std::vector<double> d = {
        0.91, 0.63, 0.46, 0.37, 1.29, 12.68, 1.59, 1.79, 2.46, 3.33,   // 380-425
        4.49, 33.94, 12.13, 6.95, 7.19, 7.12, 6.72, 6.13, 5.46, 4.79,  // 430-475
        5.66, 14.29, 14.96, 8.97, 4.72, 2.33, 1.47, 1.10, 0.89, 0.83,  // 480-525
        1.18, 4.90, 39.59, 72.84, 32.61, 7.52, 2.83, 1.96, 1.67, 4.43, // 530-575
        11.28, 14.76, 12.73, 9.74, 7.33, 9.72, 55.27, 42.58, 13.18, 13.16, // 580-625
        12.26, 5.11, 2.07, 2.34, 3.58, 3.01, 2.48, 2.14, 1.54, 1.33,   // 630-675
        1.46, 1.94, 2.00, 1.20, 1.35, 4.10, 5.58, 2.51, 0.57, 0.27,    // 680-725
        0.23, 0.21, 0.24, 0.24, 0.20, 0.24, 0.32, 0.26, 0.16, 0.12, 0.09 // 730-780
    };
    return sampledSPD(380.0, 5.0, d);
}

// Low-pressure sodium (LPS/SOX): the near-monochromatic sodium D doublet at 589.0 /
// 589.6 nm — the classic deep-orange streetlight, essentially zero colour rendering.
inline Spectrum sodiumLow() {
    return [](double w) {
        return 1.00 * gaussLobe(w, 589.0, 1.2)
             + 0.60 * gaussLobe(w, 589.6, 1.2)
             + 0.004;                             // faint background
    };
}

// High-pressure sodium (HPS/SON): the sodium D resonance is pressure-broadened and
// self-reversed (a central dip) into a wide warm band, with weaker Na lines and a
// rising red continuum — the amber-white streetlight, low but non-zero CRI.
inline Spectrum sodiumHigh() {
    return [](double w) {
        double dband = 1.00 * gaussLobe(w, 589.0, 9.0)
                     - 0.55 * gaussLobe(w, 589.3, 2.0);   // self-reversal notch
        double lines = 0.22 * gaussLobe(w, 568.5, 3.0)
                     + 0.18 * gaussLobe(w, 615.7, 3.5)
                     + 0.10 * gaussLobe(w, 498.3, 3.0)
                     + 0.10 * gaussLobe(w, 515.4, 3.0);
        double cont  = 0.05 + 0.10 / (1.0 + std::exp(-(w - 600.0) * 0.02)); // warm rise
        return std::max(0.0, dband) + lines + cont;
    };
}

// Mercury vapour: strong discrete lines at 405/436 (violet-blue), 546 (green) and
// the 577/579 yellow doublet, with almost no red — the cold blue-green cast of
// older street/industrial lamps.
inline Spectrum mercuryVapor() {
    return [](double w) {
        double lines = 0.40 * gaussLobe(w, 405.0, 2.5)
                     + 1.00 * gaussLobe(w, 436.0, 2.5)
                     + 1.10 * gaussLobe(w, 546.0, 2.5)
                     + 0.55 * gaussLobe(w, 577.0, 2.0)
                     + 0.55 * gaussLobe(w, 579.0, 2.0);
        return lines + 0.03;                      // weak continuum (poor red)
    };
}

// Metal halide: mercury lines plus additive-metal lines (In ~451, Tl ~535, Na ~589)
// over a broad rare-earth quasi-continuum — much whiter and higher-CRI than plain
// mercury vapour.
inline Spectrum metalHalide() {
    return [](double w) {
        double lines = 0.50 * gaussLobe(w, 436.0, 3.0)   // Hg blue
                     + 0.45 * gaussLobe(w, 451.0, 3.0)   // In blue
                     + 0.70 * gaussLobe(w, 535.0, 3.0)   // Tl green
                     + 0.60 * gaussLobe(w, 546.0, 3.0)   // Hg green
                     + 0.55 * gaussLobe(w, 589.0, 3.0)   // Na yellow
                     + 0.30 * gaussLobe(w, 611.0, 4.0);
        double cont  = 0.20 + 0.18 * gaussLobe(w, 560.0, 130.0); // Dy/Ho/Tm haze
        return lines + cont;
    };
}

// White phosphor LED tuned to a target correlated colour temperature. Blue InGaN
// pump (~455-465 nm) plus a broad YAG:Ce phosphor hump; cooler CCT raises the blue
// peak and shifts the phosphor bluer, warmer CCT lowers it and shifts it redder.
inline Spectrum ledCCT(double kelvin) {
    double warm = (6500.0 - kelvin) / (6500.0 - 2700.0);
    warm = std::max(0.0, std::min(1.2, warm));
    double bluePeak = 465.0 - 8.0 * warm;
    double blueAmp  = 1.0 - 0.45 * warm;
    double phosMu   = 555.0 + 45.0 * warm;
    double phosSig  = 95.0 + 10.0 * warm;
    return [=](double w) {
        return blueAmp * gaussLobe(w, bluePeak, 17.0)
             + 1.0     * gaussLobe(w, phosMu, phosSig);
    };
}

// Resolve a light/illuminant preset name to an emission SPD. Returns true and sets
// `out` if the name is recognized; returns false for unknown names so each caller
// picks its own fallback (main.cpp -> 6500 K blackbody; FTSL loader -> parse error).
// This is the single source of truth shared by the `-light` CLI flag and the FTSL
// `preset:<name>` expression — keep new sources here, not duplicated per caller.
inline bool resolveLightPreset(const std::string& name, Spectrum& out) {
    auto num = [](const std::string& s) -> double {
        try { return std::stod(s); } catch (...) { return 0.0; }
    };
    // "bbNNNN" -> Planckian at NNNN K (e.g. bb3200).
    if (name.rfind("bb", 0) == 0 && name.size() > 2) {
        double k = num(name.substr(2));
        if (k > 0) { out = blackbody(k); return true; }
    }
    // "ledNNNNk" / "led-NNNNk" -> phosphor LED at a colour temperature (e.g. led4000k).
    if (name.rfind("led", 0) == 0 && name.size() > 3) {
        std::string p = name.substr(3);
        if (!p.empty() && p[0] == '-') p.erase(0, 1);
        double k = num(p);                       // stod stops at trailing 'k'
        if (k > 100.0) { out = ledCCT(k); return true; }
    }
    if (name == "sun")                          { out = sunlight();       return true; }
    if (name == "daylight" || name == "d65")    { out = daylight(6504.0); return true; }
    if (name == "a" || name == "incandescent")  { out = illuminantA();    return true; }
    if (name == "led")                          { out = ledWhite(0.3);    return true; }
    if (name == "led-warm")                     { out = ledWhite(1.0);    return true; }
    if (name == "fluorescent" || name == "cfl") { out = fluorescent();    return true; }
    // CIE F-series fluorescents (measured tabulated SPDs).
    if (name == "f2"  || name == "cool-white")  { out = fluorescentF2();  return true; }
    if (name == "f7"  || name == "daylight-fl") { out = fluorescentF7();  return true; }
    if (name == "f11" || name == "triphosphor") { out = fluorescentF11(); return true; }
    // Gas-discharge lamps (spectroscopic line models).
    if (name == "hps" || name == "sodium")      { out = sodiumHigh();     return true; }
    if (name == "lps" || name == "sodium-low")  { out = sodiumLow();      return true; }
    if (name == "mercury" || name == "hg")      { out = mercuryVapor();   return true; }
    if (name == "metal-halide" || name == "mh") { out = metalHalide();    return true; }
    return false;
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
