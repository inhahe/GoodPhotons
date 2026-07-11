# Scene Description Language (FTSL) — Design Spec

> **Status: Phase 1 IMPLEMENTED; Phases 2–3 still design proposal.** The loader
> lives in `src/ftsl.h` and is wired to the `-in <file.ftsl>` CLI flag. It parses
> the block format below, evaluates spectrum expressions (constant, `blackbody`,
> `gaussian`, `shortpass`, `ior`, `rgb`, `whitewall`/`redwall`/`greenwall`,
> `glass:`, `preset:`, `spectrum:` refs, and `table { }`), builds materials
> (all eight `MatType`s), geometry (`sphere`/`quad`/`triangle`/`mesh` with full
> translate+rotate+non-uniform-scale transforms), any number of `light` blocks
> (area/sphere/cylinder/spot/env/collimated), a `medium`, a `camera`, and a `render` block (overridable by CLI).
> `scenes/cornell.ftsl` reproduces the hard-coded `buildCornell` **bit-for-bit**.
> Phase 2a is also done: the `scene { units … }` length unit
> (meters/centimeters/millimeters/inches/feet) is scaled to internal metres at
> load time, so a scene authored in any unit renders identically. Phase 2c is
> done too: `rgb r g b` now upsamples through a **Jakob-Hanika 2019** sigmoid
> fit (`src/upsample.h`) that round-trips linear sRGB under D65 to <1e-3 for
> unsaturated colours (validated by `ftrace -checkupsample`). Phase 2b is done:
> the engine now supports **multiple emitters** (any number of `light` blocks)
> with a power-weighted selection CDF in the forward tracer (CPU + CUDA) and an
> emitter-summing backward reference; validated by `scenes/twolight.ftsl` under
> mode V. Phase 2d is done: the `type mix` material stochastically picks among
> named child materials per photon (weights sum ≤ 1, remainder absorbs), shared by
> the forward tracer, backward reference, and CUDA kernel; validated by
> `scenes/mixmat.ftsl` under mode V. Phase 3a (partial) is done: any number of named
> `camera` blocks render one image each (per-camera film resolution + mode), with
> `-camera <name>` selection, validated by `scenes/twocam.ftsl`; a `camera_path`
> block expands into keyframe-interpolated frame cameras (`scenes/dolly.ftsl`); and
> physical film `size` (mm) → focal length, `fstop` → aperture radius, plus relative
> exposure compensation via `iso`/`shutter`/`exposure` (`scenes/expo.ftsl`).
> Phase 3b (partial) is done: a `texture` block loads a PPM/PFM image and binds it
> to a diffuse material's albedo via `reflect texture:<name>`; per-vertex UVs flow
> through quads (auto-generated corners) and OBJ meshes (`uv use_mesh` reads `vt`),
> and each texel is Jakob-Hanika–upsampled to a reflectance spectrum at the sampled
> wavelength (`scenes/textured.ftsl`, `scenes/uvmesh.ftsl`). Textured scenes run on
> the CPU (the CUDA kernel bakes only one reflect spectrum, so it defers to the CPU
> tracer). The still-unimplemented
> pieces (configurable spectral *range*, absolute light power/units — which also
> gates absolute-EV film sensitivity — the full physical `layered` material, the
> shared multi-camera mode-B pass, textures on non-albedo
> parameters) remain tagged
> **[needs engine work]** below. Alongside them, constructs
> the loader already handles are tagged **[maps 1:1]**; the spec doubles as the
> implementation checklist (§11).
>
> Before Phase 1 the renderer had **no** scene file format — scenes were the
> hard-coded C++ builders (`buildCornell`, `buildPrism`, `buildGrating`,
> `buildMaterials` in `src/main.cpp`) selected with `-scene <name>`. Those still
> exist and work; `-in` is the data-driven alternative.

---

## 0. Why a text format at all, and which one

The renderer already has a clean in-memory scene model (`Scene` in
`src/scene.h`): a material palette, triangles + spheres in a BVH, one area
light with a spectral power distribution, an optional homogeneous medium, a
contact-sensor film, and a camera. A scene file's only job is to **populate
that model from text** instead of from a hand-written C++ function. So the
language is a thin, declarative serialization of `Scene` + `Camera` + render
intent — not a general programming language.

**Format choice: a small block-structured text format (call it FTSL).** Not
JSON (too noisy for hand-authoring spectra and vectors, no comments), not a
scripting language (overkill, unsafe). The model to imitate is Mitsuba/PBRT:
typed blocks with named parameters, references by name, `#` comments. It parses
with a ~300-line hand-written tokenizer + recursive descent — no dependency.

```
# a line comment
blocktype "optional-name" {
    key   value
    key   value value value      # vectors are just space-separated numbers
    key   spectrum:someName      # reference another named block
}
```

Design rules:

- **One canonical unit system, declared up front** (see §7). Physical length
  units (metres) and physical wavelengths (nanometres) throughout. This is what
  makes the physics — diffraction, thin-film, fog, depth of field — unambiguous.
- **Spectra are first-class** (§2). The whole point of this renderer is spectral
  correctness; the format must make it easy to say "blackbody 6500 K", "this
  measured reflectance curve", or "plain RGB, upsample it for me".
- **References by name, never by index.** Materials, spectra, and cameras get
  names; geometry refers to them. (The C++ model uses integer `matId`; the
  loader resolves names → indices at load time.)
- **The file describes the *scene*; render *controls* stay on the CLI** but may
  be overridden in an optional `render { }` block (§8). This keeps one scene
  file reusable across quick previews and long final renders.

---

## 1. File structure

A file is an unordered list of top-level blocks. Recommended order for
readability: `scene` → `spectrum`s → `material`s → geometry → `light`s →
`medium` → `camera`s → `render`. The loader does two passes: collect all named
`spectrum`/`material`/`camera` blocks, then resolve references while building
geometry.

Top-level block types:

| Block        | Count      | Purpose                                             |
|--------------|------------|-----------------------------------------------------|
| `scene`      | 0 or 1     | Global units, spectral sampling range               |
| `spectrum`   | 0+         | Named reusable spectral curve                       |
| `material`   | 0+         | Named material                                      |
| `sphere`     | 0+         | Analytic sphere                                     |
| `quad`       | 0+         | Rectangle (two triangles) — walls, panels           |
| `triangle`   | 0+         | Single triangle                                     |
| `mesh`       | 0+         | OBJ instance with transform                         |
| `light`      | 1+         | Emitter (area, sphere, cylinder, spot, env, or collimated) |
| `medium`     | 0 or 1     | Global homogeneous fog                              |
| `camera`     | 1+         | Viewpoint + film + measurement model                |
| `render`     | 0 or 1     | Optional render controls (overridable by CLI)       |

---

## 2. Spectra (the heart of the format)

Everything colored — reflectances, indices of refraction, light SPDs, fog
coefficients, fluorescence curves — is a **spectrum**: a function of wavelength
λ in nm over the working range (default [360, 830] nm). In C++ this is
`using Spectrum = std::function<double(double)>` (`src/spectrum.h:11`). In FTSL
a spectrum is written inline anywhere a spectral value is expected, **or**
declared once as a named `spectrum` block and referenced as `spectrum:name`.

### 2.1 Spectrum expression forms

| Form                                   | Meaning                                                         | Backing (`src/spectrum.h`)     |
|----------------------------------------|----------------------------------------------------------------|--------------------------------|
| `0.75`                                 | Constant (gray) value at every λ                               | `constantSpectrum` — **[maps 1:1]** |
| `blackbody 6500`                       | Planck's law at 6500 K (normalized)                            | `blackbody` — **[maps 1:1]**   |
| `gaussian center=550 sigma=40 amp=0.8` | Gaussian band (emission lines, fluorescence)                   | `gaussianBand` — **[maps 1:1]** |
| `shortpass edge=500 slope=0.2 amp=1`   | Logistic high-pass (excitation filters)                        | `shortPass` — **[maps 1:1]**   |
| `glass:BK7`, `glass:diamond`, ...      | Named dispersion curve (refractive index) — see list below    | `resolveGlassIor`, `src/spectrum.h` — **[maps 1:1]** |
| `metal:Au`, `metal:copper`, ...        | Named metal reflectance R(λ) from measured n,k                | `resolveMetalReflectance`, `src/materials.h` — **[maps 1:1]** |
| `reflectance:leaf`, `reflectance:skin`, ... | Named natural diffuse reflectance (representative)        | `resolveNaturalReflectance`, `src/materials.h` — **[maps 1:1]** |
| `ior 1.5`                              | Constant refractive index                                     | `iorConstant` — **[maps 1:1]** |
| `table { 400:0.05 450:0.12 ... }`      | Piecewise-linear measured curve (λnm:value pairs)             | `tabulatedSpectrum` — **[maps 1:1]** |
| `file:data/spd/cie_f2.csv`             | Piecewise-linear curve loaded from an external data file (see below) | `loadSpdCsv` → `tabulatedSpectrum` — **[maps 1:1]** |
| `rgb 0.63 0.06 0.05`                   | Convenience: upsample an sRGB triple to a smooth reflectance   | `rgbToReflectanceJH` (Jakob-Hanika sigmoid fit, `src/upsample.h`) — **[maps 1:1]**; validated by `-checkupsample` |
| `spectrum:name`                        | Reference a named `spectrum` block                            | name resolution                |
| `preset:D65`, `preset:led`, ...        | Named illuminant SPD (see §5)                                 | `src/lights.h` — **[maps 1:1]** |

**`glass:<name>` dispersion curves** (refractive index vs λ, `src/spectrum.h`):
`BK7`/`crown`, `SF10`/`flint`, `silica`/`fused-silica`/`quartz`, `sapphire`,
`diamond`, `water`, `ice`, `acrylic`/`pmma`, `polycarbonate`/`pc`. Sellmeier for
the glasses/crystals, Cauchy fits for water/ice/plastics.

**`metal:<name>` reflectances** (normal-incidence R(λ) from measured complex index,
`src/materials.h`): `Au`/`gold`, `Ag`/`silver`, `Cu`/`copper`, `Al`/`aluminium`,
`Cr`/`chrome`, `brass`. Feed a `mirror`/`glossy` `reflect`.

**`reflectance:<name>` natural diffuse curves** (representative spectral shapes, *not*
a specific measured sample — see known-issues): `leaf`/`vegetation`, `skin`/`skin-light`,
`skin-dark`, `snow`, `soil`/`dirt`, `brick`/`red-brick`, `concrete`.

**`file:<path>` — measured spectra from disk.** Any `<spectrum>` slot accepts
`file:<path>` to load a piecewise-linear curve (an SPD, a reflectance, or an n(λ)
table) from an external CSV/whitespace data file — the runtime ingestion point for the
authoritative measured data mirrored under `data/` (see `data/README.md`). The parser
is liberal: lines beginning with `#` are comments, fields may be comma- **or**
whitespace-separated, and any line whose first two fields are not both numeric (e.g. a
`wavelength_nm,relative_power` header row) is skipped. The first numeric field is the
wavelength in nm and the second is the value (extra columns are ignored); values are
taken verbatim (an emission SPD's absolute scale is irrelevant — the power law
renormalises it — and a reflectance file should already be in 0..1). Paths resolve
relative to the current working directory (the same convention as `texture`/`mesh`
file refs), and repeated references to one path share a cached curve. Because the
built-in `preset:f2` blackbody-table was transcribed from `data/spd/cie_f2.csv`,
`spd file:data/spd/cie_f2.csv` renders pixel-identically to `spd preset:f2` — the
end-to-end loader proof (`scenes/measured_spd.ftsl`).

### 2.2 Named spectrum blocks

```
spectrum "sf10"       = glass:SF10
spectrum "warmwhite"  = blackbody 2856
spectrum "leaf_green" = gaussian center=550 sigma=35 amp=0.7
spectrum "gold"       = table { 400:0.35 500:0.42 550:0.62 600:0.90 700:0.95 }
spectrum "excite"     = shortpass edge=490 slope=0.15 amp=1.0
spectrum "emit_green" = gaussian center=560 sigma=25 amp=1.0
```

Wherever the grammar shows a `<spectrum>` you may write any inline form from
§2.1 or a `spectrum:name` reference.

> **Note on measured material spectra** (todo item "can we find spectral
> envelopes for any common materials?"): **mostly done** — built-in `metal:<name>`
> reflectances, an expanded `glass:<name>` dispersion set, `reflectance:<name>`
> natural curves, and whole-material `preset`s (§3.1) ship for common materials, and
> most are now **real measured data**:
>
> - `metal:Au|Ag|Cu|Al|Cr` — normal-incidence R computed from published complex
>   indices (Johnson & Christy 1972, Rakić 1995/1998; CC0 via refractiveindex.info).
> - `glass:*` — real Sellmeier / Cauchy dispersion.
> - `reflectance:leaf|snow|brick|concrete` — measured samples from the USGS Spectral
>   Library v7 (splib07, public domain, DOI 10.5066/F7RR1WDJ).
>
> Still representative (not per-sample measurements, see known-issues.md): `brass`,
> `reflectance:skin|skin-dark|soil`, and the iridescent recipes.
>
> Three stdlib-only converters in `tools/` ingest published data into scenes — each
> can emit an FTSL `table` block or a C++ `tabulatedSpectrum` initializer:
>
> `tools/ri_nk_to_reflectance.py` — refractiveindex.info tabulated-nk YAML →
> normal-incidence reflectance `R=((n-1)²+k²)/((n+1)²+k²)` (how the built-in metals
> were generated):
>
> ```
> python tools/ri_nk_to_reflectance.py Au_Johnson.yml --name gold_measured
> python tools/ri_nk_to_reflectance.py Al_Rakic.yml --format cpp --resample 20
> ```
>
> `tools/splib_to_reflectance.py` — USGS splib07 spectrum + wavelength file →
> reflectance (how the built-in leaf/snow/brick/concrete were generated):
>
> ```
> python tools/splib_to_reflectance.py splib07a_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt \
>     -w splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt --format cpp --resample 10
> ```
>
> `tools/csv_to_table.py` — a generic two-column CSV/TSV of (wavelength, value) →
> a `table { }` block, for any reflectance/SPD/n(λ) dataset:
>
> ```
> # a reflectance CSV (nm, value) -> a named spectrum block
> python tools/csv_to_table.py gold_reflectance.csv -n gold_measured
>
> # refractiveindex.info export in micrometres, n is column 1 -> nm
> python tools/csv_to_table.py bk7.csv --x-scale 1000 --y-col 1 -n bk7_ior
>
> # resample to a uniform 5 nm grid over the visible, clamp to [0,1]
> python tools/csv_to_table.py leaf.csv --resample 5 --min 380 --max 780 --clamp01
> ```
>
> Both auto-detect the delimiter, skip headers/comments, sort by wavelength, and
> emit either a `spectrum "name" = table { … }` block or (with `--bare`) just the
> `table { … }` expression to paste into any `<spectrum>` slot.

---

## 3. Materials

One block per material; `type` selects the model, remaining keys are that
model's parameters. All eight `MatType`s (`src/scene.h:10`) are expressible.
Any `<spectrum>`-typed parameter takes an inline spectrum or `spectrum:name`.

```
material "white"   { type diffuse    reflect 0.75 }
material "red"     { type diffuse    reflect rgb 0.63 0.06 0.05 }
material "green"   { type diffuse    reflect spectrum:leaf_green }
material "glass"   { type dielectric ior glass:SF10 }
material "mirror"  { type mirror     reflect 0.95 }
material "beam50"  { type halfmirror reflect 0.5 }
material "brushed" { type glossy     reflect spectrum:gold  roughness 0.2 }
material "bubble"  { type thinfilm   ior 1.5  film_ior 1.33  film_thickness 380 }
material "anodized"{ type thinfilm   ior 1.2  film_ior 1.45  film_thickness 430  substrate_k 3.0 }
material "beetle"  { type multilayer ior 1.5                        # Bragg / dichroic stack
                     layer 2.30 0.0 59.8   layer 1.38 0.0 99.6      # (repeat quarter-wave pairs) }
material "grating" { type grating    reflect 0.9  groove_spacing 1000  groove_dir 1 0 0  max_order 3 }
material "glow"    { type fluorescent  absorb spectrum:excite  emit spectrum:emit_green
                     yield 0.9  reflect 0.1 }
```

### 3.1 Per-type parameter reference

| `type`        | Phenomenon                        | Parameters (→ `Material` field)                                                                                  |
|---------------|-----------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `diffuse`     | Lambertian **color / translucence** | `reflect <spectrum>` → `reflect` (albedo). Reflectance < 1 already models absorption.                            |
| `dielectric`  | **transparency / refraction / dispersion** | `ior <spectrum>` → `ior` (Sellmeier curve gives wavelength-dependent bending = dispersion).                      |
| `mirror`      | **specular reflection**           | `reflect <spectrum>` → `reflect` (metallic tint; probability of reflect-vs-absorb per photon).                   |
| `halfmirror`  | **semi-mirror / beamsplitter**    | `reflect <spectrum>` → `reflect` used as reflection **probability**; the rest passes straight through (semi-transparency). |
| `glossy`      | **glossiness / brushed metal**    | `reflect <spectrum>` → `reflect`; `roughness <0..1>` → `roughness` (power-cosine lobe width).                    |
| `thinfilm`    | **iridescence** (Airy interference) | `ior <spectrum>` (substrate) → `ior`; `film_ior <n>` → `filmIor`; `film_thickness <nm>` → `filmThickness`; `substrate_k <spectrum>` → `substrateK` (substrate extinction κ; **0 = transparent** substrate → lossless reflect-or-refract; **>0 = absorbing/metallic** substrate → **opaque** structural colour, transmitted light absorbed). |
| `multilayer`  | **structural colour** (N-layer Abelès transfer matrix) | `ior <spectrum>` (substrate) → `ior`; `substrate_k <spectrum>` → `substrateK` (substrate κ, same rule as `thinfilm`); one or more ordered `layer <n> <k> <thickness_nm>` statements → `layerN`/`layerK`/`layerThick` (**layer 0 = outermost**, nearest incident air). A stack of alternating high/low-index quarter-wave layers is a Bragg/dichroic mirror (Morpho, jewel beetle, nacre, dielectric mirror). Any absorbing layer (`k > 0`) or absorbing substrate makes it **opaque** (reflect-or-absorb); a fully lossless stack over a transparent substrate is dichroic (reflect-or-refract). A single lossless layer reduces exactly to `thinfilm` Airy. GPU caps the stack at 16 layers; deeper stacks fall back to CPU. |
| `grating`     | **diffraction** (vector grating eq.) | `reflect <spectrum>` → `reflect`; `groove_spacing <nm>` → `grooveSpacing`; `groove_dir x y z` → `grooveDir`; `max_order <int>` → `gratingMaxOrder`. |
| `fluorescent` | **fluorescence** (wavelength shift) | `absorb <spectrum>` → `fluoAbsorb` (excitation ε(λ)); `emit <spectrum>` → `fluoEmit` (re-emission M(λ′), auto-baked into `fluoEmitSampler`); `yield <0..1>` → `fluoYield` (quantum yield Q); `reflect <spectrum>` → `reflect` (elastic base). |
| `mix`         | **stochastic blend of materials** | `layer "<name>" <weight>` (repeatable) → `mixChildren`/`mixWeights`. Per photon, pick child `k` with prob `weight_k`; leftover `1 − Σweight` absorbs. Children are named non-mix materials. See §3.2. |

#### Built-in material recipes (`preset <name>`)

Instead of `type` + parameters, a material block may name a **built-in recipe** that
fills a complete material (a `MatType` plus tuned parameters) for a common real-world
substance — one keyword for a realistic gold, diamond, or soap film. A few knobs
(`roughness`, `film_ior`, `film_thickness`, `reflect`, `ior`) may follow the preset to
retune it. Recipes live in `src/materials.h` (`resolveMaterialPreset`).

```
material "ring"   { preset gold }                 # polished gold (glossy + measured reflectance)
material "brushed"{ preset gold  roughness 0.4 }  # same, rougher
material "gem"    { preset diamond }              # dielectric, n≈2.42 dispersion
material "film"   { preset soap-bubble  film_thickness 300 }
```

| Preset name(s) | Expands to | Notes |
|---|---|---|
| `gold`, `silver`, `copper`, `aluminium`/`aluminum`, `chrome`, `brass` | `glossy` + `metal:<name>` reflectance, `roughness 0.05` | measured normal-incidence reflectance; **specular — reflects its surroundings, so it needs an environment/other geometry to look right (near-black in an empty pinhole box; use mode A or an env light)** |
| `glass`, `crown`, `flint`, `water`, `diamond`, `sapphire`, `silica`/`fused-silica`/`quartz`, `acrylic`/`pmma`, `polycarbonate`/`pc`, `ice` | `dielectric` + `glass:<name>` IOR | real dispersion (Sellmeier/Cauchy); `glass` = BK7 crown |
| `soap-bubble`/`bubble` | `thinfilm`, 1.33 film / 380 nm, transparent | classic reflected-only interference film |
| `oil-slick`/`oil` | `thinfilm`, 1.47 film / 320 nm, absorbing substrate | opaque oil-on-asphalt sheen |
| `anodized-ti`/`anodized-titanium` | `thinfilm`, TiO₂ 2.30 / 250 nm on Ti | anodised-metal structural colour |
| `morpho` | `multilayer`, 6× chitin/air quarter-wave (blue) | Morpho butterfly |
| `beetle`/`jewel-beetle` | `multilayer`, 6× high/low chitin (green) | jewel-beetle elytra |
| `nacre`/`mother-of-pearl` | `multilayer`, aragonite/conchiolin platelets | pastel mother-of-pearl |

Iridescent recipes are physically-motivated film/stack configurations, not measured
spectra — override `film_thickness` (thinfilm) to shift the colour.

### 3.2 Combining effects on one surface — the `layered` material

**Can a single material be semi-mirror + glossy + iridescent + (body color /
fluorescence) at once? Yes — via the `layered` material (implemented on the CPU;
`transmit`/`subsurface` body lobes are the only part still [needs engine work]).**
The important correction is that these are **not** independent, additively-stacked
flags. Physically a surface is a **two-layer stack — one specular *interface* on
top of a *body*** — and most of the "effects" are different knobs on the *same*
lobe:

**The interface** (the boundary the photon hits first). One specular lobe that
splits incoming light into a reflected and a transmitted part:

- `roughness` — 0 gives a **mirror**-sharp reflection; > 0 gives **glossy**. These
  are not two effects to combine; they are one slider. ("mirror + glossy" is a
  category error — you just pick a roughness.)
- reflectance model — plain **Fresnel** (from `ior`) *or* thin-film **Airy**. The
  Airy option *is* **iridescence**: it's a wavelength/angle-dependent replacement
  for the interface's reflectance, not a separate layer. So "rough iridescence"
  (oil-sheen / soap-film glint) = thin-film reflectance on a rough interface.
- reflect-vs-transmit split — Fresnel-governed (physical), or a manual
  `specular`/reflectance weight. A partial, angle-independent reflectance *is*
  the **semi-mirror**; the part that isn't reflected transmits inward, which is
  where **transparency** begins. "Semi-mirror + semi-transparent" is literally
  one dielectric interface.

**The body** (what happens to the transmitted part). A weighted choice, summing
with the interface to ≤ 1 (energy conservation):

- `diffuse reflect <spectrum>` — opaque **color** under the interface (paint under
  clear-coat).
- `transmit` with interior `absorb <spectrum>` — **transparency**; Beer-Lambert
  absorption inside gives a tinted glass **translucence-by-absorption**.
- `subsurface` — diffuse transmission / random-walk **translucence** (wax, skin,
  marble).
- `fluorescent { absorb / emit / yield }` — **fluorescence** (wavelength shift)
  living in the body, under any interface.

So the meaningful combinations all compose as a stack — e.g. *a rough iridescent
coat over a tinted-transparent, subsurface, faintly-fluorescent body* is one
coherent material. The only "combinations" that don't exist are the ones that
are secretly the same lobe (mirror ≡ glossy at roughness 0).

**In a photon tracer this is just weighted lobe selection per photon**, which is
exactly how `halfmirror` already works (reflect-or-transmit by probability,
`src/render.h:322`) and how dielectric/thinfilm pick reflect-vs-refract by
Fresnel/Airy probability. The `layered` material generalizes that: pick the
interface reflection vs. entering the body by probability, then pick the body
lobe by weight.

```
# SUPPORTED: one physically-layered material (CPU forward + backward)
material "wall_body_a" { type diffuse reflect rgb 0.80 0.25 0.20 }
material "wall_body_b" { type diffuse reflect rgb 0.20 0.55 0.80 }
material "lacquered_shell" {
    type layered
    coat {                        # the specular interface
        reflectance   thinfilm    # fresnel | thinfilm(=iridescent) | manual
        roughness     0.15        # 0 = mirror, >0 = glossy (roughness_map allowed)
        ior           1.5         # coat/body effective index (fresnel & thinfilm)
        film_ior      1.4         # (thinfilm only)
        film_thickness 380        # nm  (thinfilm only; film_thickness_map allowed)
        # specular    0.5         # manual model only: flat partial reflectance
    }
    # The body is a mix-style weighted list of NAMED materials (weights sum ≤ 1;
    # remainder absorbs). One lobe is chosen per photon that enters the coat.
    layer "wall_body_a" 0.55
    layer "wall_body_b" 0.45
}
```

The coat interface reflects with probability `R` (Fresnel from `ior`, thin-film
Airy from `film_ior`/`film_thickness`, or a flat `specular`) as a glossy lobe
about the mirror direction (lossless); otherwise the photon enters and one body
`layer` is picked exactly like a `mix`. Coat `R` + body weights partition the
photon, so the surface is energy-consistent by construction, and the same split
runs in the forward tracer and the backward reference. **Scope:** the body lobes
are ordinary named materials (`diffuse`, `glossy`, `fluorescent`, …); the spec's
inline `transmit`/`subsurface` body lobes are **[needs engine work]** (those
`MatType`s don't exist yet). `layered` is **CPU-only** — GPU forward/backward fall
back to the CPU tracer and BDPT (mode D) refuses a layered scene (use mode B/P or
R). As with the standalone `fluorescent` type, a layered material whose body
includes a `fluorescent` lobe stays forward + backward on the CPU; the backward
reference still can't fully validate fluorescence (see known-issues).

A `mix` of whole named materials (probabilistic pick among sub-materials) is a
simpler, less-physical alternative that the same machinery supports; `layered`
is preferred because the coat/body split is energy-consistent and matches how
real surfaces work.

**`mix` is implemented** (Phase 2d). A photon (or backward path) that hits a mix
picks child `k` with probability `weight_k`, then behaves *exactly* as that child
material; the leftover `1 − Σ weight_k` is absorption. This is unbiased per-photon
lobe selection shared verbatim by the forward tracer, the backward reference, and
the CUDA kernel, so `mix` scenes validate with mode V. Children are named
materials resolved by name (declared before or after the mix); nesting a mix
inside a mix is rejected. The CUDA path supports up to 8 child lobes (more falls
back to CPU).

```
# SUPPORTED: a stochastic mix of named materials (weights sum ≤ 1; remainder absorbs)
material "warm"  { type diffuse reflect rgb 0.85 0.55 0.30 }
material "cool"  { type diffuse reflect rgb 0.30 0.45 0.85 }
material "blend" {
    type mix
    layer "warm" 0.5       # pick 'warm' with prob 0.5
    layer "cool" 0.3       # pick 'cool' with prob 0.3
    # leftover 0.2 ⇒ absorbed
}
```

The physical `layered` material (Fresnel/Airy/manual coat over a weighted body of
named material lobes) is **implemented** on the CPU (forward + backward,
`scenes/layered.ftsl`); only the inline `transmit`/`subsurface` body-lobe types
remain **[needs engine work]**.

---

## 4. Geometry

Coordinates are in the declared length unit (§7). Normals are computed
geometrically (`Scene::build()` recomputes triangle `gn`); winding order sets
the front face. Every primitive names a `material`.

```
sphere   { center 0 1 0   radius 0.5   material glass }

quad     { origin -1 0 -1   u 2 0 0   v 0 0 2   material white }   # floor
# quad spans origin + s*u + t*v for s,t in [0,1]; normal = normalize(u × v)

triangle { v0 0 0 0   v1 1 0 0   v2 0 1 0   material red }

mesh     { file "bunny.obj"   material white
           translate 0 0 0    scale 1.0 }
```

- `sphere` / `quad` / `triangle` — **[maps 1:1]** onto `Sphere` / two `Tri` /
  one `Tri` (`src/geometry.h`).
- `mesh` — **[maps 1:1]** via `loadObj` (`src/mesh.h`), which reads `v`/`f`, the
  full affine `translate` + `rotate` (Euler XYZ, degrees) + non-uniform `scale`
  transform (Phase 1e); when `uv use_mesh` is set, per-vertex texture coordinates
  from `vt` (Phase 3b, §9); and when `usemtl use_names` is set, per-face material
  switching — each OBJ `usemtl <name>` group is matched to the FTSL material of the
  same name (unmatched → the mesh's default `material`). Two-token maps can't
  survive the statement splitter, so name-matching is the convention (mirrors
  `uv use_mesh`). **[needs engine work]** only for vertex normals (`vn` ignored;
  geometric normals are recomputed). See §9 for the UV/texture/skin discussion.

```
mesh { file "head.obj"  material skin
       uv use_mesh          # read OBJ vt texture coordinates
       usemtl use_names }   # switch material per OBJ usemtl group (name-matched)
```

**Full transform syntax** (implemented — `rotate` is Euler XYZ in degrees, `scale`
is a single uniform value or a vec3):

```
mesh { file "teapot.obj" material brushed
       translate 0 0.5 -1   rotate 0 30 0   scale 0.5 0.5 0.5 }
```

---

## 5. Lights

The scene supports **any number of emitters** (Phase 2b). Each `light` block adds
one `Emitter` (`src/scene.h`): a rectangular area light with a spectral power
distribution, a **spherical area light** (a glowing ball, Phase 3c), a
**cylindrical area light** (a glowing tube / fluorescent lamp), a **point
spotlight** (a cone with a soft penumbra, Phase 3c), an infinite environment, or a
collimated beam (prism/grating demos). `Scene::emitters` holds
the list; `finalizeEmitters()` computes each emitter's `power = emitIntegral *
area * PI`, a power-weighted selection CDF (`emitterCdf`/`totalPower`), and a
combined wavelength sampler (`emitSampler`) for the backward reference. The
forward tracer selects one emitter per photon proportional to power (so every
photon carries `beta = totalPower`, keeping the estimator unbiased); a single
emitter draws no selection randomness, so single-light scenes render
bit-identically to the pre-multi-light engine. The backward reference sums
next-event estimation over all emitters. Multiple lights are validated by
`scenes/twolight.ftsl` (mode V: forward agrees with backward, energy conserves,
CPU==GPU).

```
light area {
    origin 0.3 1.99 0.3   u 0.4 0 0   v 0 0 0.4   normal 0 -1 0
    spd blackbody 6500              # any <spectrum>, or preset:<name>
    # power  100                    # optional: absolute radiant flux in watts
    # lumens 1500                   # optional: absolute luminous flux in lumens
}

light collimated {
    dir 1 0 0
    spd preset:sun
}

light sphere {                     # Phase 3c: a glowing ball
    center 0.5 0.75 0.5   radius 0.12
    spd preset:bb6500
}

light cylinder {                   # a glowing tube (fluorescent lamp)
    center 0.5 0.85 0.5   axis 1 0 0
    length 0.7   radius 0.05       # optional: segments 48 (wall tessellation)
    caps off                       # optional: `caps on` also emits the two end discs
    spd preset:bb6500
}

light spot {                       # Phase 3c: a cone with a soft penumbra
    origin 0.5 0.98 0.5   dir 0 -1 0
    inner_angle 18   outer_angle 30   # half-angles in degrees
    spd preset:bb6500
}
```

A `light spot` registers a **point** emitter (`shape = EmitterShape::Spot`) that
radiates only into a cone about `dir`, with a cubic-smoothstep falloff between the
inner and outer half-angles (a hard cone if they are equal). Because a point has
no area, its geometric weight is the falloff-weighted solid angle `spotOmega =
π·(2 − cos θᵢ − cos θₒ)`, so `power = emitIntegral · spotOmega` and the peak
intensity per unit SPD is 1. The forward tracer samples a direction uniformly in
the outer cone and reweights the photon by `falloff · Ω_outer / spotOmega`
(analog MC); the backward reference connects each shading point straight to the
light point and weights by the cone falloff toward that point (`I(ω)·cosθ/d²`,
no area or light-side cosine). No emissive geometry is added — a point light is
infinitely small, so it has no direct-view term. Validated by
`scenes/spotlight.ftsl` (mode V: forward agrees with backward; CPU==GPU energy).

A `light sphere` registers a spherical `Emitter` (`shape = EmitterShape::Sphere`,
`area = 4·π·r²`) and also drops an emissive sphere into the geometry so photons
that strike it are absorbed and it is visible in the photon-catch camera modes
(mirroring how `light area` adds its emissive quad). Both the forward tracer and
the backward reference sample a **uniform point on the sphere surface** and use
that point's outward normal for the one-sided Lambertian cosine — so exactly the
hemisphere facing the receiver contributes, and the `1/area` point pdf and the
`power = emitIntegral · area · π` power law are unchanged from the quad case.
Sampling shares `Emitter::samplePoint()` (quad draws are byte-for-byte identical,
so existing quad scenes stay bit-identical). Validated by `scenes/spherelight.ftsl`
(mode V: forward agrees with backward; CPU==GPU energy).

A `light cylinder` registers a cylindrical `Emitter` (`shape =
EmitterShape::Cylinder`, `area = 2·π·r·L` — the **lateral** wall only; by default the
end caps are not emissive) for a fluorescent-lamp-shaped tube. `center` is the tube
midpoint, `axis` its direction, `length`/`radius` its size, and the optional
`segments` (default 48) sets how finely the loader tessellates the emissive wall into
triangles (dropped into the geometry so the tube is visible and absorbs returning
photons, mirroring the sphere light). Both tracers sample a **uniform point on the
lateral surface** — `u₁` slides along the axis, `u₂` picks the angle around it — and
use that point's outward radial normal for the one-sided Lambertian cosine; the
`1/area` point pdf and the `power = emitIntegral · area · π` law carry over
unchanged from the quad/sphere cases, so the cylinder needs no special-casing in the
forward tracer, the backward reference, or BDPT. The analytic sampling surface is
the true cylinder while the rendered wall is faceted; with the default segment count
the difference is far below Monte-Carlo noise (and only affects the geometric shape,
not the MIS pdfs, which are analytic). Validated by `scenes/cylinderlight.ftsl`
(mode V: forward agrees with backward; forward/BDPT and CPU/GPU agree to MC noise).

Add **`caps on`** to also emit from the two circular end discs — a **closed glowing
capsule** rather than an open tube. This is worth it for short, fat emissive
cylinders (a glowing puck/can, a neon segment seen end-on) where the caps are a
meaningful fraction of the emitting surface, or for any view straight down the tube's
axis where the grazing lateral wall would otherwise nearly vanish; it is *off* by
default because a real fluorescent lamp's ends are non-emissive metal end-caps and,
for a long thin tube, the caps are a negligible `~r/L` of the total. With caps the
sampling area becomes `2·π·r·L + 2·π·r²`, `samplePoint` draws all three regions
(lateral wall + both discs) with probability proportional to their area so the
`1/area` pdf and power law still hold, the loader tessellates two extra emissive
disc fans, and GPU parity is maintained (device `caps` flag). One consequence: the
backward reference's visible-arc importance sampler (`sampleCylinderVisible`, which
covers only the lateral wall) is bypassed for capped tubes, falling back to the
plain uniform `samplePoint` (still unbiased, just noisier) — a fair trade since caps
are used precisely for the short/fat cylinders where the lateral-arc win is smallest.
Validated by `scenes/cylindercaps.ftsl` (mode V forward-vs-backward scale ≈ 0.996;
forward B, BDPT D, and CPU/GPU all agree to MC noise).

```
light env { spd 0.5 }              # Phase 3c: uniform infinite environment
```

A `light env` registers an **infinite constant environment** (`shape =
EmitterShape::Env`): a uniform radiance `L(λ) = spd(λ)` arriving from every
direction, with no local position. It illuminates open scenes with a flat "sky" and
is visible directly in the background where the camera sees past all geometry. Its
geometric weight is `envGeom = 4·π²·R²` (with `R` the scene bounding-sphere radius),
so `power = emitIntegral · envGeom` — exactly the flux a uniform environment injects
through the bounding sphere. The forward tracer spawns each photon by sampling a
direction uniformly on the sphere (pdf `1/4π`) and an entry point on a disk of radius
`R` perpendicular to that direction (pdf `1/πR²`); the joint `1/(4π²R²) = 1/envGeom`
makes the spawn exactly analog (no reweight). The backward reference adds
`L(λ)·invPdfλ` on any ray that escapes the geometry. Because the forward emission is
isotropic and most photons miss an open scene, forward (mode B) env images are
**high-variance** (chromatic noise) and need large `-n`; the backward reference
(mode R) is clean. Validated by `scenes/envlight.ftsl` (mode V: forward converges to
the backward reference on a unit radiance scale; CPU and GPU energy agree). The
constant environment runs on the **GPU** forward tracer as well (the device kernel
emits env photons from the bounding sphere and the directly-viewed sky is added by
the backend-agnostic background pass).

```
light env { file "sky.hdr"  rotate 30  intensity 1.5 }   # image-based (lat-long)
```

Giving `light env` a **`file`** instead of an `spd` registers an **image-based
environment**: an equirectangular (lat-long) HDR map — a Radiance `.hdr`, a float
`.pfm`, or any LDR image the texture loader handles — becomes an infinite directional
emitter. Each texel's linear RGB is upsampled to a physical emission spectrum
`L(λ) = scale · S_JH(chroma)(λ) · illum(λ)`, where `S_JH` is the texel's Jakob-Hanika
sigmoid fit and `illum` is a normalized 6504 K illuminant (so the spectrum reproduces
the texel colour under the CIE observer, PBRT's RGB-illuminant convention); `scale`
carries the HDR brightness. `rotate` spins the map about the vertical axis (degrees)
and `intensity` scales its brightness. Directions are **importance-sampled** from a 2D
luminance CDF over the map (marginal over rows × conditional over columns, each
weighted by `sin θ`), so forward photons and the backward miss/NEE draw bright parts
of the sky in proportion to the radiance they carry — the key variance reduction for
peaked skies (a sun). The emitter's power and wavelength CDF use the map's
`sin θ`-weighted **mean** radiance spectrum, and the forward photon's flat power is
reweighted by `L(dir,λ)/(4π·pdf_ω·meanSpd(λ))` so it represents the radiance actually
arriving from the sampled direction (this factor is exactly 1 for a constant env, so
those scenes stay bit-identical). The directly-viewed background uses each texel's
spectrally-integrated XYZ, matching the backward camera-ray miss term. Validated by
`scenes/envmap.ftsl` + `scenes/sky.pfm` (mode V: forward converges to the backward
reference on a unit radiance scale). Direction convention: `θ` from `+y` (up), row 0
at the top; `φ = atan2(z,x)`, `u = φ/2π + ½`. The image environment runs on the
**GPU forward tracer** as well: the per-texel JH coeff/scale, the mean coeff/scale,
and the flattened 2D luminance CDF (marginal + per-row conditional) are uploaded, and
the direction sampler + beta reweight are ported to the device (the reweight's shared
illuminant cancels in `L/avgSpd`, so no illuminant table is needed on-device). GPU and
CPU agree to Monte-Carlo noise (energy conserves, mean RGB within ~0.5%). The backward
reference does **env next-event estimation** at every
diffuse and fog-scatter vertex — it samples a sky direction from the map's luminance
CDF, shadow-rays past the scene bounds, and **MIS-combines** (balance heuristic) that
connection with the BSDF-sampled continuation that reaches the sky on a ray miss (the
miss term is added at full weight only on a camera/specular arrival, and MIS-weighted
otherwise, so nothing is double-counted). This keeps a strongly peaked map — e.g. a
sun disk — low-variance in the reference. All env-NEE work is skipped when the scene
has no env light, so non-env scenes keep a bit-identical RNG stream / backward image.

> **Absolute-radiance camera convention.** The model-B forward light tracer now
> measures **absolute radiance** — a pixel viewing radiance `L` reads `L` (the
> pinhole importance normalizes by the *per-pixel* image-plane area, not the whole
> plane). This is what lets the directly-viewed environment background composite
> with the photon-traced surface illumination on one consistent scale, and it makes
> the mode-V / mode-P best-fit scale land at ~1 instead of an arbitrary constant.
> Auto-exposed outputs are unchanged (a global scale is invisible after exposure).

#### Absolute emitter power (`power` / `lumens`)

By default an emitter's SPD sets only its **relative** brightness and the film is
auto-exposed per image (a p99 anchor), so doubling every light's `spd` produces the
same picture. Give any light block an **absolute flux** to pin the emitter to a real
physical output instead:

```
light area {
    origin 0.35 0.999 0.35   u 0.3 0 0   v 0 0 0.3   normal 0 -1 0
    spd    preset:bb6500
    power  100                 # 100 W of radiant flux (radiometric)
}
```

- **`power <watts>`** — total **radiant** flux Φₑ in watts (radiometric). The loader
  scales the emitter SPD by `k = watts / (∫spd(λ)dλ · geomW)`, where `geomW` is the
  same geometric weight the power law already uses (`area·π` for a surface,
  `spotOmega` for a spot, `envGeom` for env), so the emitter's authored SPD shape is
  preserved and its integrated flux becomes exactly `watts`.
- **`lumens <lm>`** — total **luminous** flux Φᵥ in lumens (photometric). The loader
  solves `Φᵥ = 683 · geomW · ∫spd(λ)·V(λ)dλ` for the same scale `k` (using the CIE
  `ȳ` colour-matching function as the luminous-efficiency curve `V`), so e.g. a
  `lumens 1500` lamp emits 1500 lm regardless of its colour temperature.

Authoring either keyword on **any** light switches the whole scene into **absolute
mode**: every emitter SPD flows through the transport un-renormalised, the film is
physically linear, and `writeFilm` replaces the per-image auto-exposure with a
**fixed sensor gain** (`ABS_EXPOSURE_GAIN`) combined with the photographic controls.
In this mode `iso` / `shutter` / `exposure` become true absolute stops — doubling
`power` (or `iso`, or `shutter`) makes the image exactly one stop brighter, and a
dimmer lamp renders darker rather than being auto-normalised back. The integrals use
the same midpoint quadrature (`binWidth` bins over `[LAMBDA_MIN, LAMBDA_MAX]`) as the
emission sampler, so the absolute scale is exact. `power`/`lumens` on a `light env` is
rejected (an infinite environment has no finite total flux to author). Demonstrated by
`scenes/absolute.ftsl`.

### 5.1 Built-in illuminant SPDs (`preset:<name>`)

All resolve through the existing `-light` presets (`src/lights.h`,
`src/main.cpp:44`) — **[maps 1:1]**:

| `preset:` name                     | Physical model                                          |
|------------------------------------|---------------------------------------------------------|
| `bb<K>` (e.g. `preset:bb6500`)     | Blackbody at K kelvin (Planck)                          |
| `sun`                              | 5778 K blackbody                                        |
| `daylight` / `d65`                 | ~6504 K (CIE D65 approximation)                        |
| `a` / `incandescent`               | 2856 K tungsten (CIE Illuminant A)                     |
| `led`                              | Blue pump 460 nm + phosphor 560 nm                     |
| `led-warm`                         | Redshifted phosphor (~600 nm)                          |
| `led<K>k` (e.g. `preset:led4000k`) | Phosphor LED tuned to K kelvin (blue pump + YAG hump)   |
| `fluorescent` / `cfl`              | Mercury lines (436/546/611 nm) + continuum (illustrative) |
| `f2` / `cool-white`                | **CIE F2** cool-white fluorescent — measured SPD (~4230 K) |
| `f7` / `daylight-fl`               | **CIE F7** broadband daylight fluorescent — measured SPD (~6500 K, D65 sim) |
| `f11` / `triphosphor`              | **CIE F11** narrow-band triphosphor — measured SPD (~4000 K) |
| `hps` / `sodium`                   | High-pressure sodium — broadened, self-reversed Na D band + warm continuum |
| `lps` / `sodium-low`               | Low-pressure sodium — near-monochromatic Na D doublet (589 nm) |
| `mercury` / `hg`                   | Mercury vapour — 405/436/546/577/579 nm lines, red-deficient |
| `metal-halide` / `mh`              | Metal halide — Hg + In/Tl/Na additive lines over a rare-earth continuum |

The `f2` / `f7` / `f11` entries are **real tabulated CIE illuminant data** (380–780 nm
at 5 nm); the sodium / mercury / metal-halide entries are **spectroscopic line models**
(accurate line positions and relative strengths, analytic continua). The plain
`fluorescent` / `cfl` model remains an illustrative approximation — prefer `f2`/`f7`/`f11`
for a faithful fluorescent SPD.

Or supply any `<spectrum>` directly (`spd blackbody 3000`, `spd spectrum:myLED`,
`spd table { … }`).

### 5.2 Open design points for lights

- **Multiple / typed lights.** **[done — Phase 2b]** Any number of `light` blocks
  accumulate; the forward tracer uses a power-weighted selection CDF in the photon
  spawn path (CPU and CUDA), and the backward reference sums NEE over all emitters.
  Sphere area lights, spotlights, a uniform constant environment, and image-based
  HDRI environments are all done (Phase 3c) — see below.
- **Absolute power / units.** **[needs engine work]** Today emission is normalized by the SPD integral
  and the light area — good enough for relative imagery, but there is no
  radiometric "this bulb is 800 lumens / 10 W". A `power <watts>` (radiant) or
  `luminous <lm>` key is the place to add physically-absolute output. Until the
  engine tracks absolute units this is documentation-only.
- **Other shapes.** Sphere area lights **[done — Phase 3c]** (`light sphere {
  center … radius … }`), point spotlights **[done — Phase 3c]** (`light spot {
  dir … inner_angle … outer_angle … }`), and a uniform constant environment
  **[done — Phase 3c]** (`light env { spd … }`), and image-based HDRI environments
  **[done — Phase 3c]** (`light env { file "sky.hdr" }` with a 2D luminance CDF +
  per-texel JH spectral upsampling, on both the CPU and GPU forward tracers).

---

## 6. Medium (fog / participating media)

One optional global homogeneous medium (`Scene::medium`, `src/scene.h:84`).
**[maps 1:1]** onto the existing `-fog*` controls.

```
medium {
    # Either give sigma_t + albedo (matches the CLI), ...
    sigma_t 0.5           # extinction at 550 nm, per length unit
    albedo  0.9           # single-scattering albedo sigma_s/sigma_t
    # ... or give the two coefficient spectra directly:
    # sigma_a spectrum:...   sigma_s spectrum:...
    g       0.3           # Henyey-Greenstein anisotropy [-1..1]
    rayleigh true         # sigma_s(λ) ∝ (550/λ)^4  (sky-blue scattering)
}
```

Mapping: `sigma_t`/`albedo`/`g`/`rayleigh` correspond to `-fog`, `-fogalbedo`,
`-fogg`, `-fograyleigh`. The coefficient **units are inverse length** (1/m if
the scene is in metres), which is exactly why the length unit must be declared
(§7): `sigma_t 0.5` means something different at cm scale vs. m scale.

---

## 7. Units & scale — answering the question directly

> *"scene scale (relevant to diffraction if nothing else; or is that already
> handled by specifying everything in physical length units?)"*

Two separate things are going on, and the answer is different for each:

**(a) Wave-optics phenomena (diffraction, thin-film iridescence) are already
scale-safe — because they're computed from physical wavelengths.** The grating
equation uses the dimensionless ratio `λ / grooveSpacing`, with **both in nm**
(`src/scene.h`: `grooveSpacing` is nanometres; λ is nanometres). The thin-film
phase is `φ = 4π·n·d·cosθ / λ` with `d` (`filmThickness`) and `λ` both in nm.
The `multilayer` transfer matrix is the same: each layer's phase `δ = (2π/λ)·q·t`
uses `t` (`layerThick`) and `λ` in nm. So these effects depend only on absolute
wavelength vs. absolute feature size —
**the size of your room in metres is irrelevant to them.** You do *not* need a
global "scene scale" knob to get diffraction right; you need the groove spacing
and film thickness specified in real nm, which the format does.

**(b) Ray-geometry phenomena DO care about the length unit — and today that
unit is undeclared, which is a latent trap.** Fog extinction is per-length
(`sigma_t` in 1/unit); aperture radius, focus distance, and focal length are in
scene units; the free-flight distance `-ln(1-u)/sigma_t` comes out in scene
units. The current code has **no declared unit** — coordinates are whatever
number you type, and you must keep `sigma_t`, `aperture`, `focus` mentally
consistent with them. That's error-prone.

**Recommendation:** pin a canonical unit in the `scene` block. Metres for
length, nanometres for wavelength (fixed by physics), and let the loader do the
bookkeeping:

```
scene {
    units    meters        # length unit for ALL coordinates/distances/fog
    spectral 360 830 1      # working wavelength range + bin width (nm)
}
```

`units meters | centimeters | millimeters | inches | feet` scales coordinates,
radii, camera distances (eye/look_at/aperture/focus), mesh transforms, light
geometry, and fog coefficients (per-length, so divided by the factor) into a
single internal unit (metres) at load time — so a scene authored in cm and one in
m render identically. Wavelength / groove / film-thickness stay in nm regardless.
This makes scale *explicit* without any per-object "scale" fudge factor.
**[IMPLEMENTED, Phase 2a]** in `src/ftsl.h` (`Builder::L_`); validated: a
×100 centimetre copy of `scenes/cornell.ftsl` renders bit-for-bit identical to
the metre original.

`spectral 360 830 1` mirrors the hard-coded `[360,830]` nm range and the
`EmissionSampler` step. The **bin width** (third number) is applied to the
emission sampler today; **widening the range** is still **[needs engine work]**
(the `[360,830]` bounds are compile-time constants in `color.h`/`spectrum.h`),
so a non-default range prints a warning and is clamped to the engine range.

---

## 8. Cameras, film, and the measurement model

```
camera "hero" {
    eye     0 1 3
    look_at 0 1 0
    up      0 1 0
    fov_y   40                 # vertical field of view, degrees
    lens    50                 # OR: focal length (mm) ⇒ fov_y from film height (overrides fov_y)
    zoom    1                  # focal-length multiplier (2 = 2× tele; 0.5 = wider) — §8.5
    projection rectilinear     # lens map: rectilinear | fisheye | … (§8.5)

    aperture 0.02              # aperture radius (scene units); 0 ⇒ pinhole-ish
    focus    3.0               # focus distance ⇒ thin-lens focal length
    fstop    2.8               # f-number ⇒ apertureR = focal/(2N) (overrides aperture)
    mode     A                 # A | B | C  (measurement model, §8.2; A = physical camera)

    film {
        res    512 512         # output pixels
        format full-frame      # named sensor size (§8.1); OR: size 36 24 (mm)
        iso    100             # exposure compensation (relative) — §8.1
    }
}
```

Maps onto `Camera` (`src/camera.h`) via `lookAt(eye, target, up, fovYDeg, rx, ry)`
and `setFocus(focus)`; `aperture` → `apertureR`; `mode` picks the forward
measurement model. **[maps 1:1]**, plus the physical-film fields below. Authoring a
`lens`/`fstop` seats the finite-lens modes (A/C) as a **physically-seated** camera:
the film is placed at the real image distance and the thin lens gets the true focal
length, so the f-number yields correct depth of field (§8.1). **Mode A is the
recommended physical camera** — it images the same finite aperture + thin lens +
film as mode C, but via next-event estimation of the lens (it splats every diffuse
bounce through a sampled pupil point onto the film), so it converges in millions of
photons where the brute-force catch needs billions.

### 8.1 Film — present vs. proposed

- `res W H` — output resolution. **[maps 1:1]** (`Film::resX/resY`). Non-square films
  (W≠H) are fully supported across every tracer (CPU/GPU forward A/B/C, backward R,
  BDPT D, composite P, validate V) and checkpoint/resume. The horizontal field of view
  is derived from the width (`tanHalfX = tanHalfY·W/H`), so a wider film shows more of
  the scene rather than stretching it. The CLI `-r W H` sets a non-square film too
  (`-r N` stays square).
- `format <name>` — **[done — Phase 3a]**: a named sensor/film preset that expands
  to a physical `size` (mm). Case-, space-, hyphen-, and underscore-insensitive, so
  `full-frame`, `full frame`, and `fullframe` are the same. Recognised names:

  | name(s)                                             | size (mm, W×H) |
  |-----------------------------------------------------|----------------|
  | `full-frame` `35mm` `135` `ff`                      | 36 × 24        |
  | `half-frame`                                         | 24 × 18        |
  | `super35` `s35`                                      | 24.89 × 18.66  |
  | `academy`                                            | 21.95 × 16.0   |
  | `aps-c`                                              | 23.6 × 15.6    |
  | `aps-h`                                              | 28.7 × 19.0    |
  | `micro-four-thirds` `mft` `m43` `four-thirds`       | 17.3 × 13.0    |
  | `1inch` `1in`                                        | 13.2 × 8.8     |
  | `medium-format` `645` `6x45`                        | 56 × 41.5      |
  | `6x6`                                               | 56 × 56        |
  | `6x7`                                               | 70 × 56        |
  | `6x9`                                               | 84 × 56        |
  | `digital-medium-format` `gfx`                       | 43.8 × 32.9    |
  | `large-format` `4x5` `5x4`                          | 127 × 101.6    |
  | `8x10`                                              | 254 × 203.2    |

  An explicit `size w h` below overrides a `format`.
- `size <w> <h>` (mm) — physical sensor dimensions. **[done — Phase 3a]**: the
  focal length is derived from the film **height** and `fov_y`
  (`f = filmH / (2·tan(fov_y/2))`, in metres) and used for f-stop → aperture. A
  35 mm "full frame" is `size 36 24` (or `format full-frame`). *(When unspecified a
  24 mm full-frame height is assumed wherever a physical length is needed. Pair the
  sensor with a matching-aspect `res W H` — e.g. 3:2 — for a physically faithful
  horizontal fov and no crop.)*
- `lens <mm>` — **[done — Phase 3a]**: focal length in millimetres. Photographers
  pick a lens far more often than an angle, so `lens 50` sets the vertical field of
  view directly from the focal length and film **height**
  (`fov_y = 2·atan(filmH / (2·f))`) and **overrides** any `fov_y`. Because the fov
  follows the film height, the *same* focal length frames wider on a taller sensor
  (a 50 mm lens is "normal" on full-frame but wide on medium format) — exactly like
  real cameras. On a full-frame sensor `lens 50` ≡ `fov_y 26.99`.
- `zoom <x>` — **[done]**: a focal-length multiplier layered on `lens`/`fov_y`
  (`zoom 2` = 2× tele/narrower, `zoom 0.5` = wider). It's the animatable zoom knob;
  see §8.5 (lens/zoom) and §8.3 (per-key fov + dolly zoom).
- **f-stop authoring** — **[done — Phase 3a]**: `fstop 2.8` ⇒
  `apertureR = focal / (2·N)` at load time (overrides any `aperture` radius). When a
  `lens` **or** `fstop` is authored the finite-lens modes (A/C) become physically
  seated: the film sits at the real image distance (`1/si = 1/f − 1/focus`, or
  `si = f` when `focus` is 0/at infinity) and the thin lens takes the true focal
  length, so the f-number produces *correct* depth of field rather than a
  unit-relative blur. Legacy cameras (no `lens`/`fstop`, just an `aperture`/`focus`
  radius in scene units) keep their previous unit-film behaviour unchanged. Mode B
  (the pinhole limit) ignores the aperture entirely, so it is unaffected either way.
- `iso` / `shutter` / `exposure` — **[done — Phase 3a (relative); absolute 2026-07-11]**:
  by default the film's radiometric scale is not absolute, so images are auto-exposed
  (99th-percentile anchor) and these act as an exposure **compensation** on top of that
  anchor: `comp = exposure · (iso/100) · shutter` (each factor defaults to 1), e.g.
  `iso 200` is exactly one stop brighter than `iso 100`. Aperture is deliberately
  *not* folded in (in A/C a smaller aperture already darkens the image physically;
  in B the aperture is virtual). **True absolute EV is now available**: author an
  absolute emitter flux (`power <watts>` / `lumens <lm>`, see §5) and the whole scene
  renders in **absolute mode** — the auto-exposure anchor is replaced by a fixed
  sensor gain (`ABS_EXPOSURE_GAIN`) and `comp` becomes a real absolute-stop multiplier
  on top of it, so `iso`/`shutter`/`exposure` and the light's authored watts/lumens all
  compose as exact photographic stops (no per-image renormalisation).
- **Exposure lock** — **[done — 2026-07-11]**: because each image is auto-exposed
  independently, a moving `camera_path` can flicker as the scene brightness under the
  anchor shifts frame-to-frame. Author `exposure_lock` inside a `camera_path` block
  (a bare keyword, or `exposure_lock on`; `off`/`false`/`0` disables) to compute the
  auto-exposure anchor **once from the first frame** and reuse it for every frame of
  that path — steady exposure, no flicker. The CLI flag `-exposure-lock` forces the
  same behaviour across *all* rendered cameras (e.g. to match exposure between several
  standalone `camera` blocks). Per-frame `iso`/`shutter`/`exposure` compensation still
  applies on top of the locked anchor. This is a *relative* lock (it fixes the shared
  auto-exposure anchor); in absolute mode (`power`/`lumens`) the exposure is already
  fixed by the sensor gain, so a path renders flicker-free without needing the lock.

### 8.2 Measurement model (`mode`)

| `mode` | Model                              | Notes                                                                       |
|--------|------------------------------------|-----------------------------------------------------------------------------|
| `A`    | Physical camera (finite-lens NEE)  | The real camera: finite aperture + thin lens + film, real depth of field from `aperture`/`fstop`/`focus`. Splats every diffuse bounce through a sampled pupil point, so it converges fast. GPU-accelerated. **Rectilinear only** (author a fisheye/panoramic lens with mode B). |
| `B`    | Pinhole limit (connect/splat, default) | The `aperture → 0` limit of A: infinitely sharp (no DOF), fastest, GPU-accelerated. Ignores `aperture`/`focus`. Handles fisheye/panoramic projections (§8.5). |
| `C`    | Brute-force catch (oracle)         | Same physical lens as A, but only photons that *physically* fly through the pupil are caught — unbiased but very slow (billions of photons). Mainly a validation oracle for A. |

The three are one physical camera seen three ways: **A** importance-samples the
lens (correct **and** fast — the recommended default for a real camera), **B** is
its zero-aperture pinhole limit, **C** is the brute-force ground truth A is
validated against. (Modes R/V/P are reference/validation/composite tooling, not
scene-facing; they stay CLI-only.)

> **The old contact-sensor mode A** (a flat film wall the photons landed on) has
> been retired: with no aperture it integrated the whole hemisphere per pixel and so
> could not form an image. The `deposit()`/`Scene::sensor` machinery survives for
> irradiance-map diagnostics but is no longer wired to any camera `mode`.

### 8.3 Multiple cameras & camera paths

**Multiple named cameras are implemented** (Phase 3a). Any number of `camera`
blocks accumulate; one render invocation produces one image per camera. Selection:

- default (no `-camera`): render **every** declared camera;
- `-camera <name>`: render just that one (errors listing the available names if
  unknown);
- `-camera all`: explicit "render every camera".

Each camera has its own **film resolution** (`film { res W H }`) and its own
measurement `mode` (`A`/`B`/`C`), used unless a CLI `-r`/`-mode` forces the value
globally. With several cameras the output filename gets a `_<name>` suffix before
the extension (`-o out.ppm` → `out_hero.ppm`, `out_side.ppm`); a single camera
writes straight to `-o`. Validated by `scenes/twocam.ftsl` (a hero 256² view + an
oblique 192² view of one Cornell box; both energy-conserve).

**Current limitation (optimization, not correctness):** the cameras are rendered
as **independent forward passes** today, each re-tracing the photon set. The spec's
"same render for efficiency" — a *single shared mode-B photon pass* that connects
every diffuse bounce to all cameras' pupils at once — is a natural future extension
of `connect()` (photons are camera-independent until the splat) and is logged in
known-issues.

A `camera_path` for motion **[done — Phase 3a; `scenes/dolly.ftsl`]**:

```
camera_path "dolly" {
    look_at 0.5 0.5 0.5   up 0 1 0   fov_y 40
    mode B
    film   { res 128 128 }
    frames 5
    key 0.0   0.5 0.5 2.0            # key <t> <ex> <ey> <ez>
    key 1.0   0.5 0.5 3.6            #   optional trailing <lx> <ly> <lz> per-key look_at
}
```

A `camera_path` block expands, at load time, into `frames` ordinary named cameras
(`dolly0`, `dolly1`, … — the base name plus a zero-padded index), each an
independent forward pass just like a hand-written `camera` block. The `key`
statements give sampled `(t, eye)` control points (with an optional per-key
`look_at`); `t` is an arbitrary monotonic parameter (the keys are sorted by `t`).
For each of the `frames` output frames the parameter is stepped uniformly from the
first key's `t` to the last, and `eye`/`look_at` are **piecewise-linearly**
interpolated between the bracketing keys. The shared block-level `look_at`, `up`,
`fov_y`/`lens`, `mode`, `aperture`/`fstop`, `focus`, `projection`, and
`film { res, format/size }` apply to every frame.
`-camera dolly2` selects a single frame. The grammar is deliberately *numbers-only*
(`key <t> <ex> <ey> <ez> [<lx> <ly> <lz>] [<fov_deg>]`) because the FTSL statement
splitter breaks a statement on the next bareword, so inline keywords like `eye`/`t=`
inside a one-line `key` are not available.

**Animating the zoom (per-key fov) and the dolly zoom.** A `key` may carry a
trailing **fov** (degrees), disambiguated by field count: `t ex ey ez` (4),
`t ex ey ez fov` (5), `t ex ey ez lx ly lz` (7), or `t ex ey ez lx ly lz fov` (8).
The fov is piecewise-linearly interpolated per frame just like the eye — that gives
a **zoom** across the shot. A bare `dolly_zoom` (or `dolly_zoom on`; `off`/`false`/`0`
disables) turns on the **dolly-zoom / Vertigo effect**: the fov is instead *solved*
each frame to hold the subject's on-screen size constant while the camera dollies,
so the subject stays fixed while the background compression morphs. The subject is
each frame's `look_at` point, and the reference size is anchored on the first frame
(`distance · tan(fov/2) = const`). Example — `scenes/fisheye.ftsl`'s `vertigo`
path pulls the eye back from z = 1.0 → 2.2 while the fov auto-narrows 60° → ~22°.

**Steady exposure across a path (`exposure_lock`).** Add a bare `exposure_lock`
(or `exposure_lock on`) inside the `camera_path` block to lock the auto-exposure
anchor to the first frame's value for the whole path, so the shot doesn't flicker as
the framing changes (see §8.1 *Exposure lock*). The CLI `-exposure-lock` forces the
same across every rendered camera.

For a shared photon pass (a future optimization), the engine would connect each
diffuse bounce to *every* frame/camera's pupil (mode B) in one trace — a natural
extension of the existing `connect()` since photons are camera-independent until
the splat.

### 8.4 Progressive rendering — photon budget, time budget, and resume

The forward camera models (`mode A/B/C`) accumulate radiance one photon at a time,
so the render can be sized three ways and picked up again later. **Brightness is
independent of the photon count** — `writeFilm` divides the film by the cumulative
photons — so adding photons only *lowers the graininess*; it never changes exposure.

| CLI flag | Meaning |
|---|---|
| `-n <photons>` | Trace exactly this many photons, then stop (the default sizing). |
| `-time <seconds>` | Trace in batches until the wall-clock budget elapses. `-n` becomes the **batch size** (checkpoint granularity; default 2 000 000). Runs at least one batch and stops on the first batch boundary past the budget. |
| `-noise <percent>` | Trace in batches until the estimated graininess falls to `<= percent`, then stop and save. The metric is the same **`~X% noise`** figure the progress line reports — `100 / sqrt(mean per-lit-pixel photon count)`, the Monte-Carlo relative error at a typical lit pixel — so `-noise 2` means "keep going until it reads about 2 %". It's a global-average proxy (bright regions converge first; deep shadows may still be grainier), not a per-pixel guarantee. Alone it traces until converged; combine with `-time` to also cap the wall clock (**stops at whichever comes first**). `-n` is the batch size. Ctrl-C still stops early. |
| `-forever` | Trace indefinitely, refining the image, until you interrupt it. The first **Ctrl-C** (or Ctrl-Break) finishes the current batch, writes a final image + checkpoint, and exits cleanly; a second Ctrl-C force-quits. Implies checkpointing, so a later `-resume` picks up exactly where you stopped. |
| `-resume` | Before rendering, reload the accumulated film from the checkpoint sidecar (below) and keep adding photons to it — combine with `-n` (add that many more), `-time` (that many more seconds), or `-forever`. |
| `-checkpoint` | On a plain `-n` render, also write the checkpoint sidecar so a later `-resume` can continue it. (`-time`, `-forever`, and `-resume` imply checkpointing.) |
| `-preview` | During `-time`/`-noise`/`-forever`, redraw a live ANSI-colour thumbnail of the current image in the terminal at each periodic update (in place, over the previous frame). Needs a truecolour-capable terminal. |
| `-interval <seconds>` | Seconds between periodic image writes / preview refreshes during `-time`/`-noise`/`-forever` (default 15). The output image file is rewritten at this cadence too, so pointing an auto-reloading image viewer at it gives a live display without `-preview`. |
| `-exposure-lock` | Compute the auto-exposure anchor once (from the first rendered camera/frame) and reuse it for **every** camera rendered this invocation, so a multi-frame `camera_path` or a set of `camera` blocks share one exposure (no flicker). A per-`camera_path` `exposure_lock` keyword locks just that path (§8.1 / §8.3). |

**Checkpoint sidecar.** Because the 8-bit tone-mapped image is exposure-anchored and
gamma-quantised, it cannot be resumed from faithfully. Alongside `-o out.png` the
renderer therefore writes `out.png.ftbuf`: the raw linear XYZ film, the per-pixel hit
counts, the cumulative photon count, the energy tally, and a small identity hash of
the scene/mode/resolution. `-resume` reloads it; if the hash or resolution disagrees
with the current invocation it refuses to blend (printing a message and starting
fresh) so a stale file can never silently corrupt an image. During a `-time` or
`-forever` render the sidecar (and output image) are re-written every `-interval`
seconds (default ~15 s), so an interrupted run loses at most that much work.

**Live display.** Two ways to watch a long render progress, composable with any of the
above: (1) the output image file is rewritten every `-interval` seconds, so any
auto-reloading image viewer pointed at `out.png` is a live preview; (2) `-preview`
draws a coarse in-terminal ANSI-colour thumbnail (upper-half-block glyphs, two pixels
per character cell), redrawn in place each interval, using the same auto-exposure as
the written image so it tracks the final look. `-forever` + `-preview` is the "run and
watch until it looks good, then Ctrl-C" workflow.

Each accumulation batch is seeded with an RNG offset equal to the cumulative photon
count, so every batch — and every resume — draws a **statistically independent**
photon stream: doubling the total photons drops RMSE by ~√2, exactly as a single
render of the combined count would. A fresh (non-resumed) `-n` render uses offset 0
and is bit-for-bit identical to the historical single-shot path. These flags apply
only to the forward models `A/B/C`; the spp-based reference/BDPT (`R/V/D`) and the
`P` composite are not resumable this way (they warn and ignore the flags).

```
# render for two minutes, then add another minute later:
ftrace -in scene.ftsl -mode B -time 120 -o out.png
ftrace -in scene.ftsl -mode B -time 60  -o out.png -resume   # out.png now = 180 s of photons

# run and watch until it looks clean, then Ctrl-C to stop (image + checkpoint saved):
ftrace -in scene.ftsl -mode B -forever -preview -interval 5 -o out.png
ftrace -in scene.ftsl -mode B -forever -o out.png -resume     # keep refining later

# stop automatically once the image is clean enough (~2% estimated noise):
ftrace -in scene.ftsl -mode B -noise 2 -o out.png
ftrace -in scene.ftsl -mode B -noise 1 -time 300 -o out.png   # ...but never longer than 5 min
```

### 8.5 Lens projection & zoom — rectilinear vs. fisheye/panoramic

Two independent "focal" axes, often confused:

- **`zoom <x>` / `lens <mm>` / `fov_y` — how *much* you see (the angular slice).**
  `zoom` is a plain focal-length multiplier that composes on whatever `lens`/`fov_y`
  you set: `zoom 2` doubles the focal length (a 2× tele — *narrower* fov), `zoom 0.5`
  halves it (wider). It's the animatable "zoom ring"; per-keyframe fov on a
  `camera_path` (§8.3) animates it across a shot. **Note:** perspective/"compression"
  is *not* a focal-length property — it's set by camera **distance**. Zooming alone
  just crops; the cinematic *dolly zoom* (§8.3 `dolly_zoom`) morphs the look by
  changing distance and fov together while holding the subject size fixed.
- **`projection <name>` — *how* the angle maps to the film (the lens geometry).**
  The default `rectilinear` is a normal perspective lens: straight lines stay
  straight, but it can't reach 180° and stretches the corners. The fisheye/panoramic
  projections trade straight lines for very wide (≥ 180°) fields, each with a
  different angle-to-radius law `r(θ)`:

  | `projection` | `r(θ)` | character |
  |---|---|---|
  | `rectilinear` (default) | `tan θ` | straight lines; < 180°, corner stretch |
  | `equidistant` / `fisheye` | `θ` | classic "true" fisheye; angle ∝ radius |
  | `equisolid` | `2 sin(θ/2)` | most consumer fisheyes; preserves area |
  | `stereographic` | `2 tan(θ/2)` | "little planet"; preserves local shape |
  | `orthographic` | `sin θ` | hemispherical; 180° max |

  Shorthand: bare `fisheye` ≡ `projection equisolid`; `fisheye stereographic` etc.
  also work. With a fisheye you'll usually want a wide `fov_y` (e.g. `fov_y 160`);
  the image is a circle inscribed in the square film, so the corners fall dark.
  Validated by `scenes/fisheye.ftsl` (a rectilinear/equisolid/zoom/dolly-zoom set).

  **Implementation status:** fisheye lives in the **pinhole splat (mode B)** and the
  reference tracer (mode R/V) — those map a ray angle through the projection. The
  finite-lens modes **A and C** are **rectilinear only**: a real fisheye is a
  wide-angle *lens element*, which the single thin-lens model can't form, so author
  fisheye with **mode B** (its projection-correct splat handles the per-pixel
  solid-angle Jacobian, so images are radiometrically right, not just geometrically).
  Fisheye is also **CPU-only** — the CUDA megakernels replicate only the rectilinear
  pinhole, so a non-rectilinear camera falls back to the CPU (`-device gpu` prints a
  notice) — and mode **D (BDPT)** rejects a fisheye lens outright (its MIS camera
  importance is the rectilinear convention). All logged in `known-issues.md`.

---

## 9. Skins / textures — import, mapping, and spectral color

The wishlist: *"provide ways of mapping skins to meshes (to get as evenly
distributed / without warp as possible, such as UV mapping)? can we also get
skins with spectral envelopes somehow defined for their various colors?"*

**Current state: base-color texturing works (Phase 3b).** A `texture` block loads
a PNG / JPG / BMP / TGA / HDR (via stb_image) or PPM / PFM (built-in) image;
`reflect texture:<name>` binds it to a `diffuse` material's albedo; per-vertex UVs
flow through quads (auto corners) and OBJ meshes (`uv use_mesh` reads `vt`),
barycentric-interpolated at the hit (`src/geometry.h`); and each texel is
Jakob-Hanika–upsampled to a reflectance spectrum at the sampled wavelength
(`src/texture.h`). Validated by `scenes/textured.ftsl` (quad) and `scenes/uvmesh.ftsl`
(mesh). Procedural UV projections (planar/spherical/cylindrical, baked into
per-vertex UVs at load) and `triplanar` box projection (a per-hit blend on the
material) are implemented on both CPU and GPU; validated by `scenes/triplanar.ftsl`.
Textures on **non-albedo** scalar parameters — a **roughness map** on `glossy` and a
**film-thickness map** on `thinfilm` (§9.4) — are implemented on both backends;
validated by `scenes/scalarmap.ftsl` (CPU/GPU forward exposure and mean agree to
<0.1%). **Indexed-spectral palettes** (§9.3) are implemented on the CPU (a `palette`
block maps red-channel indices to named spectra; `scenes/palette.ftsl`); GPU falls
back to CPU for them. **Still [needs engine work]:** non-albedo maps for the remaining
parameters (mix weight, ior). GPU BDPT (mode D) falls back to the
CPU for scenes using roughness/thickness maps (the device kernel's MIS pdf/eval use
the constant parameter; the CPU BDPT threads the hit UV through so it stays unbiased).
Textured albedo runs on both backends; the CUDA kernel ports the
texture sampler (`dDiffuseRho`) so GPU and CPU agree. The section breaks into three pieces: **(9.1) importing the
image, (9.2) mapping it onto geometry, (9.3) turning its colors into spectra.**

### 9.1 Importing a skin (the image)

A `texture` block names an image and its sampling parameters. **Implemented now:**
PNG / JPG / BMP / TGA and Radiance `.hdr` (via the vendored **stb_image**, a single
public-domain header compiled once in `src/stb_image_impl.cpp`), plus PPM (P6/P3)
and PFM (PF/Pf float) through the built-in loader (`src/texture.h`). The `load()`
dispatch handles PPM/PFM itself and defers everything else to stb.

```
texture "face_albedo" {
    file     scenes/face_albedo.png
    encoding srgb            # srgb | linear  — how to decode the file  [implemented]
    filter   bilinear        # nearest | bilinear  (texel interpolation) [implemented]
    wrap     repeat          # repeat | clamp | mirror                   [implemented]
}
```

**Color management matters for physical correctness:** art PNG/JPGs are
sRGB-**display-encoded** (gamma). `encoding srgb` linearizes each texel before
use (`srgbToLinear`, `src/color.h`); data maps (roughness, masks, thickness) are
`encoding linear` and skip it. HDR/`.hdr`/`.pfm` are already linear (the PFM
loader forces `encoding linear`).

### 9.2 Mapping the skin onto the mesh (the "even / without warp" question)

A texture is sampled at a `(u,v)` produced from the surface hit. How that
`(u,v)` is chosen is the mapping method, set per mesh:

```
mesh "head" { file "head.obj"  material face
    uv use_mesh              # use the OBJ's own vt coordinates                 [implemented]
    # uv triplanar scale=4   # box projection from 3 axes, blended by normal    [implemented]
    # uv planar axis=y       # single-axis projection                          [implemented]
    # uv spherical           # lat/long — globes, eyeballs                      [implemented]
    # uv cylindrical         # bottles, limbs                                   [implemented]
}
```

> **Sub-parameter syntax.** The optional `uv` argument (`scale`, `axis`) must be a
> value the parser keeps attached to the `uv` statement: a bare number
> (`uv triplanar 4`) or a `key=val` param (`uv triplanar scale=4`, `uv planar axis=x`).
> A bareword (`uv triplanar scale 4` or `uv planar x`) is *not* accepted — the parser
> treats a trailing bareword as the start of the next statement, so `scale 4` would
> silently become a second `scale` statement and clobber the mesh's own `scale`
> transform. Default triplanar scale is `1.0`; default axis is `y`.

- **`use_mesh` (UV mapping) — the low-warp answer. [implemented]** Reads OBJ `vt`,
  stores per-vertex UVs on `Tri`, barycentric-interpolates at the hit. **Crucial point:
  "evenly distributed / without warp" is a property of the UV *unwrap* authored
  in the modeling tool** (Blender/Maya minimize stretch + seams) — it is *not*
  something the renderer solves. The renderer's job is to faithfully sample the
  UVs the mesh already carries (with correct filtering/wrap). So the answer to
  "how do I avoid warp?" is: author a good unwrap and export `vt`; the renderer
  will honor it. This is exactly what UV mapping is for.
- **`triplanar` — the renderer-side way to avoid warp with NO/bad UVs.** Project
  the texture from the ±X/±Y/±Z axes and blend the three samples by the surface
  normal. Gives seam-free, low-stretch mapping on organic shapes without any
  unwrap — the best default when a mesh has no usable UVs. (Costs 3 texture
  lookups; not tileable-artistically but distortion-free.)
- **`planar` / `spherical` / `cylindrical`** — cheap procedural projections for
  simple/known geometry; they warp on curvature (planar) or at poles
  (spherical), so they're fallbacks, not general solutions.

Per-vertex UVs on `Tri` and the `Texture` type are implemented; a tangent frame
from UV derivatives is still future work (only needed once normal/bump maps arrive).

### 9.3 Spectral envelopes for the skin's colors (the key question)

*Yes — a skin can carry proper spectra for its colors, two ways:*

- **RGB → reflectance upsampling (general, for any color image). [implemented]**
  Each linearized texel is run through the **reflectance upsampler** — the
  Jakob-Hanika 2019 sigmoid fit in `src/upsample.h` — to produce a smooth,
  physically-plausible reflectance *spectrum* for that color. The coefficients are
  precomputed per texel at load (`Texture::buildReflCoeff`), then bilinearly
  interpolated and evaluated at the sampled wavelength per hit — the standard
  Jakob-Hanika coefficient interpolation, so per-hit cost is a bilerp + sigmoid,
  not a Gauss-Newton fit. This is the same machine as the inline `rgb …` spectrum
  in §2.1. It lets ordinary painted skins participate correctly in the spectral
  pipeline (proper metamerism, correct colour under non-D65 lights) without
  hand-authoring curves. Upsampling is automatic — no `upsample` keyword needed:

  ```
  texture "face_albedo" { file scenes/face_albedo.ppm  encoding srgb }
  material "face" { type diffuse  reflect texture:face_albedo }
  ```

- **Indexed-spectral (precise, for known pigments / scientific skins). [implemented,
  CPU]** The image stores *indices* (in the **red channel**, quantized to 0..255),
  and a `palette` maps each index to a named spectrum (measured pigment, dye, metal).
  Exact where you know the actual materials — e.g. a flag or a chart of paint chips.

  ```
  texture "flag" { file "flag_index.png"  encoding linear  filter nearest
      palette { 0 spectrum:navy   1 spectrum:crimson   2 spectrum:offwhite } }
  ```

  The palette is resolved to reflectance spectra at parse time and looked up
  **nearest** (indices are categorical — never bilinearly blended). Because a palette
  entry can be *any* measured spectrum (not an RGB colour), there is no Jakob-Hanika
  upsampling and the value is used directly — so `encoding linear` + `filter nearest`
  are the right choices. Palette maps evaluate on the **CPU**; the GPU forward path
  only bakes the RGB→coeff upsampler, so palette-bound albedos fall back to the CPU
  tracer (`cudaForwardSupported`). Validated by `scenes/palette.ftsl` (a four-index
  swatch chart). **Note:** an index map with more than 256 entries needs a wider
  channel (16-bit PNG) — not yet supported; 8-bit / ≤256 indices for now.

- **True spectral images** (per-texel measured spectra, e.g. hyperspectral
  captures) are the most faithful but rarely available and storage-heavy; the
  `texture` block could grow a `spectral` encoding later. Upsampling covers the
  99% case.

### 9.4 Textures drive *any* parameter, not just base color

Because a texture resolves to a value at `(u,v)`, it can bind to any material
parameter — spectral or scalar. That is what makes skins expressive:

```
material "face" {
    type layered
    coat { roughness texture:face_rough   reflectance fresnel  film_ior 1.45 }
    body {
        diffuse    { reflect texture:face_albedo }
        subsurface { reflect texture:sss_map  weight texture:sss_mask }
    }
}
```

e.g. a **roughness map** (oily forehead vs. matte cheek), a **mix/weight mask**
(where a coat or subsurface applies), or a spatially-varying **film-thickness
map** driving §3.2 iridescence for a peacock/beetle skin. All the same texture
machinery.

**Implemented now (both backends):** a roughness map and a film-thickness map:

```
# glossy: grayscale texel = roughness directly (both 0..1)
material "wall" { type glossy   reflect whitewall 0.9  roughness texture:rough_map }

# thinfilm: film_thickness is the nominal peak (nm); the 0..1 map scales it per-hit
material "wing" { type thinfilm ior 1.5 film_ior 1.30
                  film_thickness 400   film_thickness_map texture:thick_map }
```

A **mix blend-mask** works the same way — a 2-child `mix` takes `weight_map
texture:<name>`, and the map value `t` at the hit is the probability of child 0 (child
1 gets `1-t`, no absorption), so material A shows where the mask is bright and B where
it is dark (wear masks, decals, patterns):

```
material "wear" {
    type mix
    layer "paint" 0.5
    layer "rust"  0.5
    weight_map texture:wear_mask
}
```

The scalar sample is the **mean of the linear RGB** (`Texture::scalarAt`, mirrored on
the GPU by `dTexScalarAt`), so grayscale (`encoding linear`) maps are exact and colour
maps degrade to a luminance-ish mean. Bilerp-of-means equals mean-of-bilerp, so the
CPU and GPU forward paths agree by construction (`scenes/scalarmap.ftsl`,
`scenes/maskblend.ftsl`). Because roughness/thickness are sampled per-hit and enter the
BSDF, MIS pdf/eval must see the **same** value: the CPU BDPT threads the hit UV through
`bsdfPdf`/`bsdfF`; the GPU BDPT does not, so scenes with those maps fall back to the CPU
BDPT (mode D). (A mix weight is a stochastic RR-style selection that does *not* enter
the BSDF pdf, so a blend-mask is unbiased in every tracer — but the GPU BDPT mix-pick
still uses constant weights, so masked mixes also take the CPU-BDPT fallback.) **Still
[needs engine work]:** binding a map to `ior`.

**Build order for this section:** (1) stb_image + `texture` block + `use_mesh`
UVs — makes ordinary albedo maps work; (2) the reflectance upsampler (shared
with §2.1 `rgb`) — makes them spectrally correct; (3) triplanar + procedural
projections — covers un-UV'd meshes; (4) parameter-driving + indexed/spectral
textures — full expressiveness.

---

## 10. A complete example — Cornell box in FTSL

This reproduces `buildCornell` (default scene) as a file, to show the whole
language together:

```
scene {
    units    meters
    spectral 360 830 1
}

# --- spectra ---
spectrum "wall_white" = 0.75
spectrum "wall_red"   = rgb 0.63 0.06 0.05
spectrum "wall_green" = rgb 0.14 0.45 0.09

# --- materials ---
material "white" { type diffuse    reflect spectrum:wall_white }
material "red"   { type diffuse    reflect spectrum:wall_red   }
material "green" { type diffuse    reflect spectrum:wall_green }
material "glass" { type dielectric ior    glass:SF10 }

# --- box (unit cube, light on ceiling) ---
quad { origin -1 0 -1   u 2 0 0   v 0 0 2   material white }   # floor
quad { origin -1 2 -1   u 2 0 0   v 0 0 2   material white }   # ceiling
quad { origin -1 0 -1   u 2 0 0   v 0 2 0   material white }   # back
quad { origin -1 0 -1   u 0 2 0   v 0 0 2   material red   }   # left
quad { origin  1 0 -1   u 0 2 0   v 0 0 2   material green }   # right

sphere { center 0 0.5 0   radius 0.5   material glass }

# --- light ---
light area {
    origin -0.3 1.99 -0.3   u 0.6 0 0   v 0 0 0.6   normal 0 -1 0
    spd preset:bb6500
}

# --- camera ---
camera "cam" {
    eye 0 1 3   look_at 0 1 0   up 0 1 0   fov_y 40
    mode B
    film { res 512 512 }
}

# --- optional render controls (else from CLI) ---
render {
    photons 200000000
    device  auto
}
```

Invocation once the loader exists: `ftrace -in cornell.ftsl -o cornell.ppm`,
with CLI flags still able to override anything in `render { }` (e.g. `-n`,
`-device`, `-mode`).

---

## 11. Implementation checklist (what building this actually costs)

Ordered by dependency; the first group is a working loader for scenes the
engine *already* renders, later groups are engine features the format is
designed to grow into.

**Phase 1 — loader for existing capabilities [mostly maps 1:1]**
1. Tokenizer + block parser (~300 lines, no deps).
2. Spectrum expression evaluator wrapping the existing `src/spectrum.h`
   builders; add `tabulatedSpectrum` for `table { }`.
3. Build `Scene`/`Camera` from blocks; resolve names → `matId`/indices.
4. `-in <file.ftsl>` CLI flag; `render { }` overridable by existing flags.
5. Full mesh transform (rotate + non-uniform scale) in `loadObj`.

**Phase 2 — near-term engine features the format already anticipates**
6. `units` scaling + configurable `spectral` range.
7. Multiple lights (emitter list + power-weighted selection CDF). **[done — Phase 2b; `scenes/twolight.ftsl`]**
8. RGB→reflectance upsampler (unlocks `rgb` spectra and later textures). **[done — `src/upsample.h`, Jakob-Hanika sigmoid fit, `-checkupsample`]**
9. `mix`/layered materials (generalize `halfmirror`). **[`mix` done — Phase 2d; per-photon lobe selection in forward + backward + CUDA; `scenes/mixmat.ftsl`. Physical `layered` done 2026-07-11 — CPU forward + backward coat/body split; `scenes/layered.ftsl`; only inline `transmit`/`subsurface` body lobes remain.]**

**Phase 3 — larger features**
10. Multiple cameras / `camera_path`; per-camera films; physical film size +
    f-stop + sensitivity. **[mostly done — Phase 3a: multiple named cameras +
    `-camera` selection + per-camera film resolution + per-camera mode
    (`scenes/twocam.ftsl`); `camera_path` keyframe interpolation
    (`scenes/dolly.ftsl`); physical film `size` (mm) → focal length, `fstop` →
    aperture radius, relative exposure compensation via `iso`/`shutter`/
    `exposure` (`scenes/expo.ftsl`), a `camera_path`/CLI exposure **lock**, and
    non-square films (`res W H` / `-r W H`, resX≠resY through every tracer).
    Remaining: shared mode-B multi-camera pass and absolute-EV/sensitivity (needs
    absolute light power).]**
11. UVs + spectral/RGB textures; per-face materials from OBJ `usemtl`.
12. Additional light shapes (sphere/spot/HDRI environment).

Each phase is independently useful: Phase 1 alone replaces the hard-coded
scene builders with real data files, which is the actual ask.

---

## 12. Cross-references

- In-memory model: `src/scene.h` (`Scene`, `Material`, `MatType`, `Medium`,
  `Sensor`), `src/geometry.h` (`Tri`, `Sphere`), `src/camera.h` (`Camera`),
  `src/scene_film.h` (`Film`).
- Spectra: `src/spectrum.h` (`Spectrum`, builders, `EmissionSampler`),
  `src/lights.h` (illuminant presets), `src/color.h` (CIE, [360,830] nm range).
- Mesh loader: `src/mesh.h` (`loadObj`).
- Current CLI (until `-in` exists): `src/main.cpp:888` onward.
