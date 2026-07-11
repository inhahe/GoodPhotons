# ftrace — a spectral forward + backward photon raytracer

`ftrace` is a physically-based, **spectral** light-transport renderer written in
C++20. Unlike an RGB path tracer, it transports light one **wavelength** at a
time, so dispersion, chromatic aberration, thin-film iridescence, fluorescence
and diffraction all fall out of the physics for free — there is no special-casing
of colour. It has both a **forward** (light-tracing / photon) core and an
independent **backward** (path-traced) reference, a CUDA GPU backend for the
forward pinhole mode, and a small scene-description language (**FTSL**).

---

## Highlights

- **Spectral transport** — single-wavelength photons over a configurable band
  (e.g. `spectral 360 830 1`); per-wavelength refraction gives dispersion and
  chromatic aberration with no extra code.
- **Forward *and* backward** engines that validate each other (mode `V` reports
  the residual between them).
- **Realistic cameras** — from a simple pinhole to a **physical multi-element
  lens** (real glass prescription: depth of field, vignetting, spherical &
  chromatic aberration all emergent), plus fisheye/panoramic projections.
- **Rich material set** — dielectrics with real glass dispersion, metals from
  measured data, rough microfacet, thin-film & multilayer interference,
  diffraction gratings, fluorescence, and stochastic mixes.
- **Wave-optical effects** — thin-film Airy interference, Abelès multilayer
  stacks, and reflective diffraction gratings.
- **Participating media** — homogeneous fog with Henyey–Greenstein or Rayleigh
  scattering.
- **CUDA GPU backend** for the forward pinhole splat (mode `B`), megakernel or
  wavefront, with CPU fallback.
- **Long-running renders** — time / noise / forever budgets, live ANSI preview,
  and checkpoint/resume.

---

## Building

Requires a C++20 compiler and CMake. CUDA is **optional** (auto-detected).

```sh
cmake -B build -S .
cmake --build build --config Release --target ftrace
```

The binary lands at `build/bin/ftrace` (`.exe` on Windows).

Useful CMake options:

| Option | Meaning |
|---|---|
| *(CUDA auto-detected)* | If a CUDA toolkit is found, the GPU backend (`src/render_cuda.cu`) is compiled and `-device gpu` becomes available; otherwise a CPU-only binary is built. |
| `-DFTRACE_CUDA_ARCH=native\|all-major\|"75;86;89"` | GPU architectures to target (default `native`). |
| `-DFTRACE_GPU_FP32=ON\|OFF` | GPU transport in float with double film accumulation (default `ON`); `OFF` = full FP64 (slower, CPU-matching). |

---

## Quick start

```sh
# Built-in Cornell box, forward pinhole splat (mode B), 512²
ftrace -scene cornell -n 200000000 -r 512 -o cornell.png

# Render an FTSL scene file
ftrace -in scenes/cornell.ftsl -n 200000000 -o out.png

# Physical-lens camera demo (depth of field from real optics; forces mode R)
ftrace -in scenes/realcam.ftsl -n 6000000 -o realcam.png

# Render until a wall-clock budget with a live preview
ftrace -in scenes/group.ftsl -time 120 -preview -o group.png
```

---

## Render modes (`-mode`, or per-camera `mode`)

| Mode | Name | What it does | Backend |
|---|---|---|---|
| `A` | Contact sensor | Pure forward photon catch on a front-wall sensor (no lens) | CPU |
| `B` | Pinhole splat *(default)* | Light-tracing splat to a pinhole camera; fast, independent photons | CPU + **GPU** |
| `C` | Finite-aperture catch | Forward photon catch through a thin lens (real depth of field) | CPU |
| `R` | Backward reference | Backward path-traced reference image; drives the physical-lens camera | CPU |
| `V` | Validate | Runs `B` and `R` and reports the best-fit residual between them | CPU |
| `P` | Composite | Forward `B` for diffuse/caustic pixels + a backward camera ray for specular/coated surfaces | CPU |
| `D` | BDPT | Bidirectional path tracing with MIS over all light×camera connections | CPU |

Choose the hardware with `-device auto|cpu|gpu` (default `auto`, falls back to CPU
if CUDA is unavailable). `-wavefront` selects the streaming GPU backend instead of
the megakernel.

---

## Cameras

Defined with a `camera "name" { … }` block (or the built-in scene camera).

**Basics:** `eye`, `look_at`, `up`, `fov_y`, `mode`, and a `film { res N M … }`
block. Film size can be a preset **format** — `full-frame`, `aps-c`,
`micro-four-thirds`, `super35`, `medium-format`, `6x6`, `6x7`, `large-format`,
`4x5`, `8x10` — or an explicit `size W H` in millimetres.

**Projections** (`projection …`): `rectilinear` (default perspective),
`equidistant` and `equisolid` fisheye, `stereographic` ("little planet"), and
`orthographic`. These are analytic remaps available in the forward pinhole mode.

**Analytic depth of field:** `aperture`, `focus`, `lens` (focal length, mm),
`fstop`, and `zoom` give a thin-lens camera with a real focus plane and bokeh.

**Physical (realistic) lens** — `lens { … }`:

A camera can carry a real **lens prescription**: a stack of spherical/planar
glass interfaces plus an aperture stop. The backward tracer samples a film point
and a point on the rear element and traces the ray *through the actual glass*
(per-wavelength Snell refraction), so **depth of field, distortion, spherical &
chromatic aberration, field curvature and vignetting all emerge from the
geometry** — no thin-lens approximation. A physical lens automatically renders in
mode `R`.

```ftsl
camera "real" {
    eye 0 0.55 -1.6   look_at 0 0.35 2.4   up 0 1 0
    focus 2.4
    film { res 512 512   format full-frame }
    lens {
        preset achromat   # singlet | biconvex | achromat | doublet | telephoto | wide
        focal 50          # mm
        fstop 2.8
        glass BK7
    }
}
```

- **Presets** are physically derived (lensmaker equation for the singlet, Abbe-number
  power split for the achromatic doublet), so focal length and colour correction are
  correct by construction; render-time dispersion uses real Sellmeier glass indices.
- Or paste an **arbitrary real prescription** as repeated
  `surface <radius_mm> <thickness_mm> <ior> <semi_aperture_mm> [stop]` lines
  (PBRT lens-file convention). See `scenes/realcam.ftsl` for a working demo.

*Current limits:* the physical lens is backward-only (mode `R`, CPU), renders to a
square film (the 3:2 sensor is cropped to the output frame), and does not model
inter-element flare/ghosting or shaped-iris bokeh.

---

## Materials

Declared with `material "name" { type <type> … }`.

| Type | Description | Key parameters |
|---|---|---|
| `diffuse` | Lambertian reflector | `reflect` (spectrum or `texture:<name>`) |
| `dielectric` | Refractive glass with dispersion | `ior` (Sellmeier glass or constant) |
| `mirror` | Perfect specular reflector | `reflect` |
| `halfmirror` | 50/50 beamsplitter | `reflect` |
| `glossy` | Rough microfacet reflector | `reflect`, `roughness` |
| `thinfilm` | Single-layer interference (iridescence) | `ior`, `film_ior`, `film_thickness` (nm), `substrate_k` |
| `multilayer` | N-layer Abelès transfer-matrix stack | `ior`, `substrate_k`, repeated `layer <n> <k> <nm>` |
| `grating` | Reflective diffraction grating | `reflect`, `groove_spacing` (nm), `groove_dir`, `max_order` |
| `fluorescent` | Stokes-shifted fluorescence | `reflect`, `absorb`, `emit`, `yield` |
| `mix` | Stochastic blend of materials | repeated `layer <material> <weight>` |

**Whole-material presets** (`preset <name>`): metals (gold/Au, silver/Ag,
copper/Cu, aluminium/Al, chromium/Cr, brass), glasses, and iridescent recipes
(soap-bubble, oil-slick, anodised-Ti, morpho, beetle, nacre).

---

## Spectra (SPDs, reflectances, indices)

Anywhere a spectrum is expected (`spd`, `reflect`, `ior`, …) you can write:

- **`preset:<name>`** — illuminants and light sources: `bb<K>` blackbody (e.g.
  `bb6500`), `sun`, `d65`/`daylight`, `a`/`incandescent`, `led`, `led-warm`,
  `led<K>k`, CIE F-series fluorescents (`f2`/`cool-white`, `f7`, `f11`), and
  gas-discharge lamps (`hps`/`sodium`, `mercury`, `metal-halide`).
- **`rgb r g b`** — Jakob–Hanika sigmoid upsampling to a reflectance spectrum
  (round-trips under D65).
- **`table { 400:0.05 450:0.12 … }`** — a measured/tabulated spectrum
  (piecewise-linear).
- **`glass:<name>`** — dispersive index via Sellmeier: `BK7`/crown, `SF10`/flint,
  `silica`/fused-silica, `sapphire`, `diamond`, plus Cauchy fits for `water`,
  `ice`, `acrylic`/PMMA, `polycarbonate`.
- **`metal:<name>`** and **`reflectance:<name>`** — measured metal reflectance
  (Au/Ag/Cu/Al/Cr) and natural-material reflectances (leaf, skin, snow, soil,
  brick, concrete).
- **`spectrum "name" { … }`** blocks to define and reuse a named SPD.

---

## Lights

`light <subtype> { … }`:

| Subtype | Description | Key parameters |
|---|---|---|
| *(default)* area | Rectangular area light | `origin`, `u`, `v`, `normal`, `spd` |
| `sphere` | Spherical area light | `center`, `radius`, `spd` |
| `cylinder` | Cylindrical tube light | `center`, `axis`, `length`, `radius`, `caps`, `spd` |
| `spot` | Cone spotlight with penumbra | `origin`, `dir`, `inner_angle`, `outer_angle`, `spd` |
| `collimated` | Thin parallel pencil beam | `origin`, `dir`, `spd` |
| `env` | Environment / IBL light | `file` (lat-long HDR) or `spd`, `rotate`, `intensity` |

---

## Geometry

`sphere`, `quad` (parallelogram), `triangle`, and `mesh` (OBJ import, with
`usemtl use_names` for per-face materials and `uv use_mesh` for mesh UVs).
`group { translate … rotate … scale … <children> }` composes transform
hierarchies (baked to world space at load). Everything is accelerated by a BVH.

## Textures

`texture "name" { file <path> encoding srgb|linear filter nearest|bilinear wrap
repeat|clamp|mirror }` loads PNG / JPG / HDR / PPM / PFM images; bind one to a
diffuse albedo with `reflect texture:<name>`. Each texel is Jakob–Hanika
upsampled to a reflectance spectrum. (Base-colour maps today; normal/roughness
maps are future work.)

## Participating media / fog

`medium { sigma_t <v> albedo <v> g <v> rayleigh <bool> }`, or from the CLI with
`-fog <sigma_t> -fogalbedo <a> -fogg <g> [-fograyleigh]`. Henyey–Greenstein phase
function by default; Rayleigh optional.

---

## Scene language (FTSL)

An FTSL file is a list of blocks. Top-level block types: `scene` (the
`units …` / `spectral …` header), `material`, `texture`, `spectrum`, `sphere`,
`quad`, `triangle`, `mesh`, `light`, `group`, `medium`, `camera`, `camera_path`
(keyframed camera animation), and `render` (render-setting overrides). See the
`scenes/` directory for worked examples (`cornell.ftsl`, `fisheye.ftsl`,
`spotlight.ftsl`, `envlight.ftsl`, `material_presets.ftsl`, `realcam.ftsl`, …).

---

## Command-line reference

**Core**

| Flag | Meaning |
|---|---|
| `-in <path>` | Load an FTSL scene file |
| `-scene <name>` | Built-in scene (e.g. `cornell`) |
| `-n <photons>` | Trace exactly this many photons/samples |
| `-r <res>` | Square output resolution (overrides scene default) |
| `-o <path>` | Output image (`.png` / `.jpg` / `.ppm` by extension) |
| `-mode <A..D>` | Render mode (default `B`) |
| `-camera <name>` | Select a named camera |
| `-t <threads>` | CPU thread count |
| `-device auto\|cpu\|gpu` | Hardware backend |
| `-wavefront` | Streaming GPU backend instead of megakernel |

**Camera / physics overrides**

| Flag | Meaning |
|---|---|
| `-light <preset>` | Override light SPD by preset |
| `-aperture <r>` / `-focus <d>` | Thin-lens aperture radius / focus distance |
| `-mesh <path>` / `-meshscale <s>` | Load & scale an OBJ into the built-in scene |
| `-fog <σt>` / `-fogalbedo <a>` / `-fogg <g>` / `-fograyleigh` | Fog controls |
| `-filmthickness <nm>` / `-filmior <n>` | Thin-film iridescence demo params |
| `-diffraction <mode>` / `-nodiffraction` | Enable/disable grating & thin-film diffraction |
| `-spp <n>` | Samples per pixel for mode `V` |

**Long-running / output**

| Flag | Meaning |
|---|---|
| `-time <s>` | Render until a wall-clock budget |
| `-noise <pct>` | Render until the noise floor drops below `pct` % |
| `-forever` | Refine indefinitely (Ctrl-C stops gracefully) |
| `-preview` | Live ANSI thumbnail while rendering |
| `-interval <s>` | Periodic image write / preview refresh (default 15 s) |
| `-resume` / `-checkpoint` | Resume from / always write a `<out>.ftbuf` checkpoint |

**Diagnostics / self-tests:** `-checkbvh`, `-bvhstats`, `-checklens`,
`-checkfluoro`, `-checkfog`, `-checkthinfilm`, `-checkmultilayer`,
`-thinfilmswatch`, `-checkgrating`, `-checkupsample`.

---

## Output

Images are written as **PNG**, **JPEG** (q95), or binary **PPM (P6)**, chosen by
the output file extension, tone-mapped from the internal linear spectral film to
8-bit sRGB. Long renders can checkpoint to `<out>.ftbuf` and resume
deterministically.

---

## Known issues & roadmap

Open limitations and technical debt are tracked in `known-issues.md` — including
the physical-lens camera's remaining gaps (forward-catch / GPU / BDPT support,
non-square film, inter-element flare) and the shared multi-camera pass.
