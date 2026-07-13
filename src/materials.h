// Whole-material presets for common real-world materials.
//
// NOTE: both the measured spectral DATA and the hand-tuned iridescent RECIPES that
// used to be baked here now live in external files, loaded at runtime:
//   - metal reflectances / natural reflectances -> data/metal/*.csv,
//     data/reflectance/*.csv (resolveMetalReflectance / resolveNaturalReflectance).
//   - whole-material recipes (`material { preset <name> }`) -> data/material/*.material
//     bundle files (a MatType + spectral envelopes + intrinsic scalars grouped into
//     one named asset). The soap-film / oil-slick / anodised-Ti / Morpho / beetle /
//     nacre configs — layer stacks, film thickness/index, substrate extinction — are
//     just tuned numeric parameters, so they are DATA and move out to files.
// Only DATA moved out; the ALGORITHMS stay native — the interference / Abeles-matrix
// / Fresnel evaluators (render.h) and the sellmeier/cauchy dispersion functions are
// unchanged. This file now just interprets a bundle manifest into a Material.
#pragma once
#include <string>
#include "spectrum.h"
#include "spectral_library.h"
#include "scene.h"

// Map a `type <name>` bundle field to a MatType. Returns false on an unknown type.
inline bool parseMatType(const std::string& t, MatType& out) {
    std::string s = speclib::lower(t);
    if      (s == "diffuse")          out = MatType::Diffuse;
    else if (s == "dielectric")       out = MatType::Dielectric;
    else if (s == "mirror")           out = MatType::Mirror;
    else if (s == "halfmirror")       out = MatType::HalfMirror;
    else if (s == "glossy")           out = MatType::Glossy;
    else if (s == "fluorescent")      out = MatType::Fluorescent;
    else if (s == "thinfilm")         out = MatType::ThinFilm;
    else if (s == "grating")          out = MatType::Grating;
    else if (s == "multilayer")       out = MatType::Multilayer;
    else if (s == "layered")          out = MatType::Layered;
    else if (s == "diffusetransmit")  out = MatType::DiffuseTransmit;
    else if (s == "filter")           out = MatType::Filter;
    else return false;
    return true;
}

// --- Whole-material recipe bundles ------------------------------------------
// Interpret a data/material/<name>.material manifest into a complete Material. A
// bundle groups several spectral envelopes plus intrinsic scalars under one name:
//   type <matType>                     required
//   reflect|emit|ior|absorb|transmit|substrate_k  <spectrum expr>   (token resolver)
//   roughness|film_ior|film_thickness  <number>
//   layer <n> <k> <thickness_nm>       one per stack layer (accumulates, outer first)
// Spectrum-valued fields accept the same primitive vocabulary as the scene language
// (`const N`, `metal:Au`, `glass:BK7`, `reflectance:leaf`, `file:...`, `blackbody K`,
// `ior N`, `gaussian …`). Returns true and fills `out` when a bundle file exists.
inline bool resolveMaterialBundle(const std::string& name, Material& out) {
    speclib::Bundle b;
    if (!speclib::loadBundle("material", name, b)) return false;
    Material m;
    bool haveType = false;
    auto num = [](const std::string& s) { return std::strtod(s.c_str(), nullptr); };
    auto spec = [&](const std::vector<std::string>& a, Spectrum& dst) {
        Spectrum s; if (speclib::resolveSpectrumTokens(a, s)) dst = s;
    };
    for (const auto& f : b.fields) {
        const auto& a = f.args;
        if (f.key == "type") {
            if (a.empty() || !parseMatType(a[0], m.type)) return false;
            haveType = true;
        }
        else if (f.key == "reflect")      spec(a, m.reflect);
        else if (f.key == "emit")         spec(a, m.emit);
        else if (f.key == "ior")          spec(a, m.ior);
        else if (f.key == "absorb")       spec(a, m.absorb);
        else if (f.key == "transmit")     spec(a, m.transmit);
        else if (f.key == "substrate_k")  spec(a, m.substrateK);
        else if (f.key == "roughness"      && !a.empty()) m.roughness     = num(a[0]);
        else if (f.key == "film_ior"       && !a.empty()) m.filmIor       = num(a[0]);
        else if (f.key == "film_thickness" && !a.empty()) m.filmThickness = num(a[0]);
        else if (f.key == "layer" && a.size() >= 3) {
            m.layerN.push_back(num(a[0]));
            m.layerK.push_back(num(a[1]));
            m.layerThick.push_back(num(a[2]));
        }
        // Unknown keys are ignored (forward-compatible with future material fields).
    }
    if (!haveType) return false;
    out = m; return true;
}

// `material "x" { preset <name> }` fills a complete Material. Resolution order:
//   1. an explicit data/material/<name>.material bundle (the curated recipes,
//      including all iridescent structural-colour materials);
//   2. a generic CONVENTION for bare primitives, so any metal/glass in the library
//      (including future drop-in files) works as a material with no bundle needed:
//      a metal name -> lightly-polished Glossy tinted by its reflectance; a glass
//      name -> clear Dielectric with its dispersion (`glass` aliases to BK7).
// Returns true and sets `out` on a known name. (Common knobs — roughness, film
// thickness/index, reflect, ior — may still be overridden by the caller afterwards.)
inline bool resolveMaterialPreset(const std::string& name, Material& out) {
    if (resolveMaterialBundle(name, out)) return true;
    Material m;
    Spectrum s;
    // Polished metals -> glossy with the measured reflectance tint.
    if (resolveMetalReflectance(name, s)) {
        m.type = MatType::Glossy;
        m.reflect = s;
        m.roughness = 0.05;
        out = m; return true;
    }
    // Transparent dielectrics -> refractive glass with the right dispersion.
    std::string glassName = (name == "glass") ? "BK7" : name;
    if (resolveGlassIor(glassName, s)) {
        m.type = MatType::Dielectric;
        m.ior = s;
        m.roughness = 0.0;                        // clear glass (opt into frosting explicitly)
        out = m; return true;
    }
    return false;
}
