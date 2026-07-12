# Spectral asset library

The renderer loads its measured / tabulated spectral **data** from files here at
runtime, instead of compiling it into the binary. Each subdirectory is a *category*;
the lookup key for a named preset is the lowercased filename stem, plus any extra
names a file declares in a `# aliases: a b c` header line. The library is drop-in
extensible: add a file to a category directory and it resolves by name with **no
rebuild**. Loading is handled by `src/spectral_library.h`.

Only measured/tabulated **data** lives here. The **algorithms** that consume it stay
in the source: the Sellmeier/Cauchy dispersion evaluators (`src/spectrum.h`), the
piecewise-linear curve builder (`tabulatedSpectrum`), Planck blackbody, the LED /
gas-discharge line models, and the BSDF / thin-film / iridescent recipes.

## File formats

- **Curve files** (`.csv`, categories `metal/`, `reflectance/`, `illuminant/`):
  comment lines start with `#`, one header row `wavelength_nm,<value column>`, then
  `wavelength_nm,value` rows. Comma **or** whitespace delimited. Values are relative
  unless the column name says otherwise (an emission SPD's absolute scale is
  irrelevant — the power law renormalises it; a reflectance should already be 0..1).
  Ingested by `tabulatedSpectrum` (piecewise-linear, clamped at the endpoints).
- **Glass files** (`.glass`, category `glass/`): dispersion **coefficients** fed to
  the native evaluators, not a sampled curve, so the lens/Abbe math stays exact.
  A `form` line names the evaluator and the rest supply its coefficients:
  - `form sellmeier` + `B b1 b2 b3` + `C c1 c2 c3`  → `sellmeier(B…,C…)`
  - `form cauchy` + `A a` + `B b`                    → `cauchy(a,b)`
  - `form constant` + `n 1.5`                        → `iorConstant(n)`

## Categories & resolvers

| Directory        | FTSL expression        | Resolver (spectral_library.h)   |
|------------------|------------------------|---------------------------------|
| `glass/`         | `glass:<name>`         | `resolveGlassIor`               |
| `metal/`         | `metal:<name>`         | `resolveMetalReflectance`       |
| `reflectance/`   | `reflectance:<name>`   | `resolveNaturalReflectance`     |
| `illuminant/`    | `preset:<name>` (light)| `resolveTabulatedIlluminant`    |

`material { preset <name> }` and the built-in defaults (BK7/SF10 for `dielectric`
and the lens presets) route through these same resolvers.

### `file:<path>` — ad-hoc curves
Any `<spectrum>` slot in FTSL also accepts `file:<path>`, which loads an arbitrary
curve file directly (`speclib::loadSpdCsv` → `tabulatedSpectrum`), bypassing the
named-preset index. E.g. `spd file:data/illuminant/f2.csv`, demonstrated by
`scenes/measured_spd.ftsl` (which renders identically to `spd preset:f2`, since the
preset loads that very file — the end-to-end proof).

## Present data

### `glass/*.glass` — dispersion coefficients
BK7 (`crown`), SF10 (`flint`), fused-silica (`silica`/`quartz`), sapphire, diamond,
water, ice, acrylic (`pmma`), polycarbonate (`pc`). Schott catalog & Malitson/Peter
Sellmeier fits for the crystalline glasses; two-term Cauchy fits for the weakly-
dispersive materials (water/ice/plastics). Verified `n_d`: BK7 1.5168, SF10 1.7283,
water 1.333.

### `metal/*.csv` — normal-incidence reflectance R(λ)
Au (`gold`), Ag (`silver`), Cu (`copper`), Al (`aluminium`/`aluminum`), Cr
(`chromium`/`chrome`), brass. `R = ((n-1)²+k²)/((n+1)²+k²)` from published complex
refractive indices: Johnson & Christy 1972 for Au/Ag/Cu, Rakic 1995/1998 for Al/Cr
(all CC0 via refractiveindex.info; regenerate with `tools/ri_nk_to_reflectance.py`).
Brass has no single canonical dataset and remains a Cu-Zn alloy fit.

### `reflectance/*.csv` — natural diffuse reflectance
leaf (`vegetation`), skin-light (`skin`), skin-dark, snow, soil (`dirt`), red-brick
(`brick`), concrete. leaf/snow/red-brick/concrete are USGS Spectral Library v7
samples (splib07, public domain, DOI 10.5066/F7RR1WDJ). skin-light/skin-dark/soil
are representative *shapes* (illustrative, not a specific measured sample — see
known-issues.md).

### `illuminant/*.csv` — measured SPDs
f2 (`cool-white`), f7 (`daylight-fl`), f11 (`triphosphor`): CIE standard illuminant
F-series relative SPDs, 380-780 nm at 5 nm. Transcribed from CIE 15:2004 fluorescent
illuminant tables via colour-science (github.com/colour-science/colour, BSD-3; the
CIE tables themselves are public reference data).

## Pending (loader exists; better data still to be fetched + wired)

Closing each of these is just: drop a `wavelength_nm,value` CSV into the right
category directory (add `# aliases:` if you want extra names) and it resolves by
name. No rebuild. Each entry lists an authoritative, openly-licensed source.

### Discharge lamps (HPS, LPS, metal-halide, mercury)
Currently `src/lights.h` `sodiumHigh/sodiumLow/mercuryVapor/metalHalide()` are
spectroscopic *line models*, not measurements. To swap in measurements, drop a
measured SPD into `illuminant/` (e.g. `hps.csv`) and reference it as `preset:hps`.
- **LSPDD** — Lamp Spectral Power Distribution Database (lspdd.org): measured SPDs
  of real market lamps; per-lamp CSV export.
- **LICA-UCM lamps spectral database v2.6** (guaix.fis.ucm.es): measured lamp SPDs.

### Human skin reflectance (`skin`, `skin-dark`)
Currently `reflectance/skin-light.csv` / `skin-dark.csv` are representative
haemoglobin-dip shapes.
- **NIST "Reference Data Set of Human Skin Reflectance"**, J. Res. NIST 122.026:
  100 measured spectra, 250-2500 nm, CSV supplement. Average a light- and a
  dark-skin subset; clip/resample to 380-780 nm and overwrite the two files.

### Soil / loam reflectance (`soil`)
Currently `reflectance/soil.csv` is a representative reddish rise (USGS splib07's
"soils" are mineral sands, not generic loam).
- **NASA JPL ECOSTRESS Spectral Library v1.0** (speclib.jpl.nasa.gov): a
  loam/dark-brown soil record as ASCII, resampled to 380-780 nm.
- **ISRIC Globally Distributed Soil Spectral Library** (data.isric.org).

### Iridescent recipes (`soap-bubble`, `oil-slick`, `anodized-ti`, `morpho`, …)
These are thin-film-interference *models* (layer index/thickness), so there is no
single measured SPD to mirror; they stay as native recipes in `src/materials.h`.
