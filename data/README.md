# Measured spectral data

Public-domain / open measured spectra, mirrored here so the runtime measured-SPD
loader and the `tools/` regeneration scripts have a canonical local copy. Format
for `.csv` files: comment lines start with `#`, one header row
`wavelength_nm,<value column>`, then `wavelength_nm,value` rows. Values are
relative unless a column name says otherwise (an emission SPD's absolute scale is
irrelevant — the power law renormalises it; a reflectance should already be 0..1).

**Runtime loader.** Any `<spectrum>` slot in FTSL accepts `file:<path>`, which
loads one of these files into a piecewise-linear curve (`loadSpdCsv` in
`src/ftsl.h` → `tabulatedSpectrum`). The parser is liberal about comma-vs-whitespace
delimiters and skips non-numeric header rows. E.g. `spd file:data/spd/cie_f2.csv`,
demonstrated by `scenes/measured_spd.ftsl` (which renders pixel-identically to the
baked `preset:f2`, the end-to-end proof).

## Present

### `spd/cie_f2.csv`, `spd/cie_f7.csv`, `spd/cie_f11.csv`
CIE standard illuminant F-series relative SPDs, 380-780 nm at 5 nm.
- **Source:** CIE 15:2004 fluorescent illuminant tables, transcribed via
  colour-science (github.com/colour-science/colour, BSD-3; the CIE tables
  themselves are public reference data).
- **Used by:** `src/lights.h` `fluorescentF2/F7/F11()` (`f2`/`cool-white`,
  `f7`/`daylight-fl`, `f11`/`triphosphor`). The baked tables now match these
  files exactly (an earlier F7 tail 685-780 nm was mis-transcribed "from memory"
  and has been corrected against this data).

## Pending (loader exists; data still to be fetched + wired)

The `file:<path>` loader now exists, so closing each of these is just: download the
source below into `data/spd/` (or `data/reflectance/`) as a `wavelength_nm,value`
CSV, then point the relevant preset (or a scene) at it. Each entry lists an
authoritative, openly-licensed source and how to extract a clean table.

### Discharge lamps (HPS, LPS, metal-halide, mercury)
Currently `src/lights.h` `sodiumHigh/sodiumLow/mercuryVapor/metalHalide()` are
spectroscopic *line models*, not measurements.
- **LSPDD** — Lamp Spectral Power Distribution Database (lspdd.org): measured SPDs
  of real market lamps; per-lamp CSV export. Pick representative HPS / MH / Hg /
  LPS entries.
- **LICA-UCM lamps spectral database v2.6** (guaix.fis.ucm.es / researchgate
  312596512): measured lamp SPDs, ASCII.

### Human skin reflectance (`skin`, `skin-dark`)
Currently `src/materials.h` `reflectanceSkinLight/Dark()` are representative
haemoglobin-dip shapes.
- **NIST "Reference Data Set of Human Skin Reflectance"**, J. Res. NIST 122.026
  (nvlpubs.nist.gov/nistpubs/jres/122/jres.122.026.pdf): 100 measured spectra,
  250-2500 nm, distributed as a CSV supplement. Average a light-skin and a
  dark-skin subset for the two presets; clip/resample to 380-780 nm.

### Soil / loam reflectance (`soil`)
Currently `src/materials.h` `reflectanceSoil()` is a representative reddish rise
(USGS splib07's "soils" are mineral sands, not generic loam).
- **NASA JPL ECOSTRESS Spectral Library v1.0** (speclib.jpl.nasa.gov): download a
  loam/dark-brown soil record as ASCII, resample to 380-780 nm.
- **ISRIC Globally Distributed Soil Spectral Library** (data.isric.org): VIS-NIR
  diffuse reflectance, many loam samples.

### Iridescent recipes (`soap-bubble`, `oil-slick`, `anodized-ti`, `morpho`, ...)
These are thin-film-interference *models* (layer index/thickness), so there is no
single measured SPD to mirror; validate against specimen photographs instead.
