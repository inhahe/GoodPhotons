// Built-in material data and recipes for common real-world materials.
//
// Three kinds of built-in live here:
//   1. Metal spectral reflectances (`metal:<name>`) — normal-incidence R(lambda)
//      computed from published measured complex refractive indices: Johnson &
//      Christy 1972 for Au/Ag/Cu, Rakic 1995/1998 for Al/Cr (all CC0 via
//      refractiveindex.info; regenerate with tools/ri_nk_to_reflectance.py).
//      Brass has no single canonical dataset and remains an alloy fit. Feed a
//      `mirror`/`glossy` material's `reflect`.
//   2. Natural diffuse reflectances (`reflectance:<name>`) — representative curves
//      for vegetation, skin, snow, etc. These capture the characteristic spectral
//      SHAPE (chlorophyll dip, haemoglobin W, flat snow) but are illustrative, not
//      a specific measured sample — see known-issues.md.
//   3. Whole-material recipes (`material { preset <name> }`) — a MatType plus tuned
//      parameters, so one keyword yields a realistic gold / diamond / soap-film /
//      Morpho material. Iridescent recipes are physically-motivated film/stack
//      configurations, not measured spectra.
#pragma once
#include <string>
#include "spectrum.h"
#include "scene.h"

// --- Metal spectral reflectance (normal incidence) --------------------------
// R(lambda) = ((n-1)^2 + k^2) / ((n+1)^2 + k^2), computed from published complex
// refractive indices n,k (see tools/ri_nk_to_reflectance.py). Au/Ag/Cu use the
// canonical Johnson & Christy 1972 measurements at their native sample points;
// Al/Cr use the Rakic 1995/1998 datasets resampled to 20 nm. All data is CC0
// (refractiveindex.info). Wavelengths in nm; tabulatedSpectrum interpolates.
inline Spectrum metalGold() {   // Johnson & Christy 1972 (Au), R from n,k
    return tabulatedSpectrum({
        {354.2,0.383},{367.9,0.392},{381.5,0.403},{397.4,0.407},{413.3,0.409},{430.5,0.408},{450.9,0.408},{471.4,0.401},
        {495.9,0.447},{520.9,0.643},{548.6,0.787},{582.1,0.882},{616.8,0.931},{659.5,0.963},{704.5,0.971},{756,0.974},
        {821.1,0.976},{892,0.98}
    });
}
inline Spectrum metalSilver() {   // Johnson & Christy 1972 (Ag), R from n,k
    return tabulatedSpectrum({
        {354.2,0.876},{367.9,0.928},{381.5,0.956},{397.4,0.963},{413.3,0.968},{430.5,0.978},{450.9,0.98},{471.4,0.979},
        {495.9,0.981},{520.9,0.984},{548.6,0.983},{582.1,0.987},{616.8,0.987},{659.5,0.991},{704.5,0.993},{756,0.996},
        {821.1,0.995},{892,0.996}
    });
}
inline Spectrum metalCopper() {   // Johnson & Christy 1972 (Cu), R from n,k
    return tabulatedSpectrum({
        {354.2,0.41},{367.9,0.426},{381.5,0.446},{397.4,0.464},{413.3,0.492},{430.5,0.518},{450.9,0.539},{471.4,0.555},
        {495.9,0.576},{520.9,0.591},{548.6,0.619},{582.1,0.726},{616.8,0.9},{659.5,0.943},{704.5,0.956},{756,0.959},
        {821.1,0.963},{892,0.966}
    });
}
inline Spectrum metalAluminium() {   // Rakic 1995 (Al), R from n,k, 20 nm grid
    return tabulatedSpectrum({
        {360,0.925},{380,0.925},{400,0.924},{420,0.923},{440,0.923},{460,0.921},{480,0.92},{500,0.919},
        {520,0.918},{540,0.916},{560,0.915},{580,0.913},{600,0.911},{620,0.91},{640,0.907},{660,0.904},
        {680,0.901},{700,0.897},{720,0.894},{740,0.888},{760,0.882},{780,0.875},{800,0.868},{820,0.866}
    });
}
inline Spectrum metalChromium() {   // Rakic 1998 (Cr, LD model), R from n,k, 20 nm grid
    return tabulatedSpectrum({
        {360,0.647},{380,0.65},{400,0.652},{420,0.654},{440,0.654},{460,0.654},{480,0.654},{500,0.653},
        {520,0.652},{540,0.65},{560,0.649},{580,0.647},{600,0.646},{620,0.644},{640,0.643},{660,0.641},
        {680,0.639},{700,0.638},{720,0.636},{740,0.635},{760,0.634},{780,0.633},{800,0.631},{820,0.63}
    });
}
inline Spectrum metalBrass() {   // Cu-Zn alloy: a paler, less saturated gold
    return tabulatedSpectrum({
        {380,0.40},{420,0.41},{460,0.43},{500,0.50},{520,0.58},{540,0.68},
        {560,0.76},{580,0.80},{600,0.83},{640,0.85},{680,0.86},{720,0.87},{760,0.88}
    });
}

// Resolve a `metal:<name>` reflectance preset. Shared by the FTSL `metal:`
// expression and the material recipes below.
inline bool resolveMetalReflectance(const std::string& name, Spectrum& out) {
    if (name == "Au" || name == "gold")      { out = metalGold();      return true; }
    if (name == "Ag" || name == "silver")    { out = metalSilver();    return true; }
    if (name == "Cu" || name == "copper")    { out = metalCopper();    return true; }
    if (name == "Al" || name == "aluminium" ||
        name == "aluminum")                  { out = metalAluminium(); return true; }
    if (name == "Cr" || name == "chromium" ||
        name == "chrome")                    { out = metalChromium();  return true; }
    if (name == "brass")                     { out = metalBrass();     return true; }
    return false;
}

// --- Natural / everyday diffuse reflectances --------------------------------
// Representative spectral SHAPES (see file header + known-issues). Range extends
// to 830 nm so vegetation's red-edge NIR rise is captured.
inline Spectrum reflectanceLeaf() {   // green vegetation: green bump, chlorophyll dip, red-edge
    return tabulatedSpectrum({
        {400,0.05},{450,0.05},{500,0.08},{550,0.15},{570,0.13},{600,0.08},
        {640,0.05},{670,0.04},{690,0.06},{700,0.22},{720,0.42},{750,0.50},
        {780,0.52},{830,0.53}
    });
}
inline Spectrum reflectanceSkinLight() {   // haemoglobin W-dips at 540/576 nm
    return tabulatedSpectrum({
        {400,0.20},{440,0.28},{480,0.36},{520,0.42},{540,0.40},{560,0.44},
        {576,0.41},{600,0.55},{640,0.63},{680,0.66},{720,0.68},{780,0.70},{830,0.71}
    });
}
inline Spectrum reflectanceSkinDark() {
    return tabulatedSpectrum({
        {400,0.06},{440,0.08},{480,0.11},{520,0.14},{560,0.16},{600,0.22},
        {640,0.29},{680,0.34},{720,0.38},{780,0.42},{830,0.44}
    });
}
inline Spectrum reflectanceSnow() {   // near-flat high in visible, drops in NIR
    return tabulatedSpectrum({
        {400,0.95},{500,0.95},{600,0.94},{680,0.92},{720,0.86},{780,0.75},{830,0.60}
    });
}
inline Spectrum reflectanceSoil() {   // smooth reddish-brown rise
    return tabulatedSpectrum({
        {400,0.05},{450,0.07},{500,0.10},{550,0.14},{600,0.19},{650,0.24},
        {700,0.29},{750,0.33},{800,0.36},{830,0.37}
    });
}
inline Spectrum reflectanceRedBrick() {
    return tabulatedSpectrum({
        {400,0.06},{450,0.07},{500,0.09},{550,0.12},{600,0.20},{620,0.28},
        {650,0.34},{700,0.40},{760,0.44},{830,0.46}
    });
}
inline Spectrum reflectanceConcrete() {   // fairly flat mid-grey
    return tabulatedSpectrum({
        {400,0.30},{450,0.33},{500,0.36},{550,0.38},{600,0.40},{650,0.41},
        {700,0.42},{760,0.43},{830,0.43}
    });
}

// Resolve a `reflectance:<name>` diffuse preset.
inline bool resolveNaturalReflectance(const std::string& name, Spectrum& out) {
    if (name == "leaf" || name == "vegetation") { out = reflectanceLeaf();      return true; }
    if (name == "skin" || name == "skin-light") { out = reflectanceSkinLight(); return true; }
    if (name == "skin-dark")                     { out = reflectanceSkinDark();  return true; }
    if (name == "snow")                          { out = reflectanceSnow();      return true; }
    if (name == "soil" || name == "dirt")        { out = reflectanceSoil();      return true; }
    if (name == "brick" || name == "red-brick")  { out = reflectanceRedBrick();  return true; }
    if (name == "concrete")                      { out = reflectanceConcrete();  return true; }
    return false;
}

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
