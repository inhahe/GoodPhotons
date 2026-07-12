// Built-in material recipes for common real-world materials.
//
// NOTE: the measured spectral DATA that used to be baked here — the metal
// reflectances (`metal:<name>`, Johnson & Christy / Rakic R(lambda) tables) and the
// natural diffuse reflectances (`reflectance:<name>`, USGS splib07 curves) — now
// lives in external files (data/metal/*.csv, data/reflectance/*.csv) and is loaded
// at runtime by `resolveMetalReflectance()` / `resolveNaturalReflectance()` in
// spectral_library.h. Only the DATA moved out; this file keeps the ALGORITHMIC
// content:
//   Whole-material recipes (`material { preset <name> }`) — a MatType plus tuned
//   parameters, so one keyword yields a realistic gold / diamond / soap-film /
//   Morpho material. Iridescent recipes are physically-motivated film/stack
//   configurations (parameter recipes, not measured spectra), so they stay native.
#pragma once
#include <string>
#include "spectrum.h"
#include "spectral_library.h"
#include "scene.h"

// --- Whole-material recipes -------------------------------------------------
// `material "x" { preset <name> }` fills a complete Material. Metals default to a
// lightly-polished glossy lobe (override with `roughness`); iridescent recipes are
// physically-motivated film/stack configs (override `film_thickness` to retune the
// colour). Returns true and sets `out` on a known name.
inline bool resolveMaterialPreset(const std::string& name, Material& out) {
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
    // (Map the material name to the glass IOR of the same substance.)
    std::string glassName = name;
    if (name == "glass") glassName = "BK7";
    if (resolveGlassIor(glassName, s)) {
        m.type = MatType::Dielectric;
        m.ior = s;
        m.roughness = 0.0;                        // clear glass (opt into frosting explicitly)
        out = m; return true;
    }
    // Iridescent / structural colour.
    if (name == "soap-bubble" || name == "soap_bubble" || name == "bubble") {
        m.type = MatType::ThinFilm;               // water film in air, both sides transparent
        m.ior = iorConstant(1.0);                 // "substrate" = air behind the film
        m.filmIor = 1.33; m.filmThickness = 380.0;
        m.substrateK = constantSpectrum(0.0);
        out = m; return true;
    }
    if (name == "oil-slick" || name == "oil_slick" || name == "oil") {
        m.type = MatType::ThinFilm;               // oil film on dark wet asphalt (absorbing)
        m.ior = iorConstant(1.5);
        m.filmIor = 1.47; m.filmThickness = 320.0;
        m.substrateK = constantSpectrum(2.0);     // absorbing substrate -> opaque iridescence
        out = m; return true;
    }
    if (name == "anodized-ti" || name == "anodized_ti" || name == "anodized-titanium") {
        m.type = MatType::ThinFilm;               // TiO2 film on titanium metal
        m.ior = iorConstant(2.5);
        m.filmIor = 2.30; m.filmThickness = 250.0;
        m.substrateK = constantSpectrum(3.0);
        out = m; return true;
    }
    if (name == "morpho") {
        m.type = MatType::Multilayer;             // chitin/air quarter-wave stack tuned to ~450 nm blue
        m.ior = iorConstant(1.56);
        m.substrateK = constantSpectrum(0.5);     // melanin backing -> opaque, saturated
        for (int i = 0; i < 6; ++i) {
            m.layerN.push_back(1.56); m.layerK.push_back(0.0); m.layerThick.push_back(72.0);   // chitin
            m.layerN.push_back(1.00); m.layerK.push_back(0.0); m.layerThick.push_back(112.0);  // air
        }
        out = m; return true;
    }
    if (name == "beetle" || name == "jewel-beetle") {
        m.type = MatType::Multilayer;             // high/low chitin stack tuned to green
        m.ior = iorConstant(1.6);
        m.substrateK = constantSpectrum(0.4);
        for (int i = 0; i < 6; ++i) {
            m.layerN.push_back(1.70); m.layerK.push_back(0.0); m.layerThick.push_back(75.0);
            m.layerN.push_back(1.40); m.layerK.push_back(0.0); m.layerThick.push_back(95.0);
        }
        out = m; return true;
    }
    if (name == "nacre" || name == "mother-of-pearl") {
        m.type = MatType::Multilayer;             // aragonite/conchiolin platelets -> pastel iridescence
        m.ior = iorConstant(1.68);
        m.substrateK = constantSpectrum(0.0);     // translucent
        for (int i = 0; i < 5; ++i) {
            m.layerN.push_back(1.68); m.layerK.push_back(0.0); m.layerThick.push_back(300.0);  // aragonite
            m.layerN.push_back(1.53); m.layerK.push_back(0.0); m.layerThick.push_back(120.0);  // conchiolin
        }
        out = m; return true;
    }
    return false;
}
