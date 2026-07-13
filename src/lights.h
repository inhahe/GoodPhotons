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
#include "spectral_library.h"
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
// tabulated fluorescent SPDs load the CIE F-series (f2/f7/f11) — those measured
// tables now live in data/illuminant/*.csv and are resolved by
// resolveTabulatedIlluminant() (spectral_library.h), not baked in this file.
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

// NOTE: the CIE Standard Illuminant F-series (F2 cool-white, F7 daylight/D65-sim,
// F11 triphosphor) used to be baked here as `fluorescentF2/F7/F11()` sampledSPD
// tables. Those measured tabulated SPDs now live in data/illuminant/{f2,f7,f11}.csv
// and are loaded at runtime by `resolveTabulatedIlluminant()` (spectral_library.h);
// resolveLightPreset() below delegates the f2/f7/f11 names there. Only the DATA
// moved out — the sampledSPD/emissionLines builders and the analytic line models
// (sodium/mercury/metal-halide/LED) are algorithms and stay native.

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

// Resolve a light `spd <expr>` token list to an emission SPD. First tries the shared
// data-oriented vocabulary (`blackbody K`, `const N`, `file:...`, `gaussian …` — see
// spectral_library.h), then the native light MODELS defined above (which live here,
// above the library, so they can't go in the shared resolver): the phosphor-LED and
// gas-discharge line models. Returns false on an unrecognized head.
inline bool resolveLightSpd(const std::vector<std::string>& w, Spectrum& out) {
    if (speclib::resolveSpectrumTokens(w, out)) return true;
    if (w.empty()) return false;
    const std::string& h = w[0];
    auto num = [](const std::string& s) -> double {
        try { return std::stod(s); } catch (...) { return 0.0; }
    };
    if (h == "led-white")                        { out = ledWhite(w.size() > 1 ? num(w[1]) : 0.3); return true; }
    if (h == "led-cct" && w.size() > 1)          { out = ledCCT(num(w[1]));  return true; }
    if (h == "fluorescent")                      { out = fluorescent();      return true; }
    if (h == "sodium-high" || h == "sodium")     { out = sodiumHigh();       return true; }
    if (h == "sodium-low")                       { out = sodiumLow();        return true; }
    if (h == "mercury")                          { out = mercuryVapor();     return true; }
    if (h == "metal-halide")                     { out = metalHalide();      return true; }
    return false;
}

// Interpret a data/light/<name>.light bundle into an emission SPD. A light asset
// currently groups a single `spd <expr>` field (plus room for future intrinsic
// fields — angular/goniometric envelope, size — grouped under one name). Aliases are
// handled by the library's `# aliases:` header scan. Returns true when a file exists
// and yields an SPD.
inline bool resolveLightBundle(const std::string& name, Spectrum& out) {
    speclib::Bundle b;
    if (!speclib::loadBundle("light", name, b)) return false;
    for (const auto& f : b.fields)
        if (f.key == "spd") return resolveLightSpd(f.args, out);
    return false;
}

// Resolve a light/illuminant preset name to an emission SPD. Returns true and sets
// `out` if the name is recognized; returns false for unknown names so each caller
// picks its own fallback (main.cpp -> 6500 K blackbody; FTSL loader -> parse error).
// This is the single source of truth shared by the `-light` CLI flag and the FTSL
// `preset:<name>` expression — keep new sources here, not duplicated per caller.
//
// Resolution order: parametric names (computed from the name) -> data/light/*.light
// bundles (sun / daylight / incandescent / led / led-warm — simple parameter bindings
// to the native models, now externalized) -> data/illuminant/*.csv measured SPDs ->
// the native gas-discharge line models (analytic shaping = algorithm, kept in source).
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
    // Externalized named presets (data/light/*.light) — sun / daylight / incandescent
    // / led / led-warm and any drop-in additions.
    if (resolveLightBundle(name, out)) return true;
    if (name == "fluorescent" || name == "cfl") { out = fluorescent();    return true; }
    // CIE F-series fluorescents (measured tabulated SPDs, loaded from
    // data/illuminant/*.csv via the spectral library — aliases handled there).
    if (resolveTabulatedIlluminant(name, out)) return true;
    // Gas-discharge lamps (spectroscopic line models — analytic shaping stays native).
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
