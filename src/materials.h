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
// Four of these are measured samples from the USGS Spectral Library v7 (splib07,
// public domain, DOI 10.5066/F7RR1WDJ), extracted via tools/splib_to_reflectance.py:
//   leaf     = Oak_Oak-Leaf-1_fresh (ASD, 10 nm)   -- record 13172-ish, green leaf
//   snow     = Melting_snow_mSnw01a (ASD, 20 nm)
//   brick    = Brick_GDS353_Building_MedRed (ASD, 20 nm)
//   concrete = Concrete_GDS375_Lt_Gry_Road (ASD, 20 nm)
// skin (light/dark) and soil have no clean splib sample and remain representative
// SHAPES (see file header + known-issues). Range extends to 830 nm so vegetation's
// red-edge NIR rise is captured.
inline Spectrum reflectanceLeaf() {   // USGS Oak green leaf: green bump, chlorophyll dip, red-edge
    return tabulatedSpectrum({
        {360,0.104},{370,0.099},{380,0.097},{390,0.096},{400,0.096},{410,0.095},{420,0.096},{430,0.096},
        {440,0.097},{450,0.097},{460,0.097},{470,0.098},{480,0.098},{490,0.098},{500,0.099},{510,0.104},
        {520,0.119},{530,0.144},{540,0.163},{550,0.173},{560,0.174},{570,0.16},{580,0.142},{590,0.131},
        {600,0.127},{610,0.122},{620,0.115},{630,0.112},{640,0.11},{650,0.105},{660,0.101},{670,0.098},
        {680,0.098},{690,0.104},{700,0.149},{710,0.263},{720,0.402},{730,0.546},{740,0.668},{750,0.752},
        {760,0.798},{770,0.818},{780,0.828},{790,0.833},{800,0.837},{810,0.84},{820,0.843},{830,0.846}
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
inline Spectrum reflectanceSnow() {   // USGS melting snow: high-flat in visible, drops in NIR
    return tabulatedSpectrum({
        {360,0.821},{380,0.807},{400,0.819},{420,0.831},{440,0.834},{460,0.833},{480,0.833},{500,0.832},
        {520,0.832},{540,0.832},{560,0.833},{580,0.833},{600,0.83},{620,0.827},{640,0.823},{660,0.82},
        {680,0.817},{700,0.816},{720,0.813},{740,0.809},{760,0.799},{780,0.782},{800,0.772},{820,0.774}
    });
}
inline Spectrum reflectanceSoil() {   // smooth reddish-brown rise
    return tabulatedSpectrum({
        {400,0.05},{450,0.07},{500,0.10},{550,0.14},{600,0.19},{650,0.24},
        {700,0.29},{750,0.33},{800,0.36},{830,0.37}
    });
}
inline Spectrum reflectanceRedBrick() {   // USGS medium-red building brick
    return tabulatedSpectrum({
        {360,0.082},{380,0.08},{400,0.078},{420,0.077},{440,0.078},{460,0.079},{480,0.08},{500,0.082},
        {520,0.085},{540,0.091},{560,0.113},{580,0.167},{600,0.213},{620,0.229},{640,0.234},{660,0.238},
        {680,0.243},{700,0.249},{720,0.253},{740,0.254},{760,0.253},{780,0.249},{800,0.244},{820,0.24}
    });
}
inline Spectrum reflectanceConcrete() {   // USGS light-grey road concrete
    return tabulatedSpectrum({
        {360,0.155},{380,0.168},{400,0.184},{420,0.199},{440,0.216},{460,0.229},{480,0.239},{500,0.251},
        {520,0.266},{540,0.28},{560,0.292},{580,0.302},{600,0.308},{620,0.311},{640,0.313},{660,0.315},
        {680,0.316},{700,0.317},{720,0.318},{740,0.318},{760,0.318},{780,0.318},{800,0.317},{820,0.316}
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
