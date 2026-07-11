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

Modes `A`/`B`/`C`/`P` trace **identical forward physics** and differ only in how
the camera *measures* the light (splat vs. physical catch vs. composite). `R` and
`D` are separate backward / bidirectional estimators. All modes are unbiased and
converge to the same image; they trade **speed, noise character, and which light
paths they can capture at all**.

| Mode | Name | What it does | Backend |
|---|---|---|---|
| `A` | Finite-lens camera | Forward next-event splat through a finite aperture + thin lens (true depth of field, efficient) | CPU + GPU |
| `B` | Pinhole splat *(default)* | Light-tracing splat to a pinhole camera; independent photons | CPU + **GPU** |
| `C` | Finite-aperture catch | Forward photon catch through a thin lens (real depth of field) | CPU + GPU |
| `R` | Backward reference | Backward path-traced reference image; drives the physical-lens camera | CPU + **GPU** |
| `V` | Validate | Runs `B` and `R` and reports the best-fit residual between them | CPU (+GPU forward pass) |
| `P` | Composite | Forward `B` for diffuse/caustic pixels + a backward camera ray for specular/coated surfaces | CPU + **GPU** |
| `D` | BDPT | Bidirectional path tracing with MIS over every light×camera connection | CPU + **GPU** |

### Speed / accuracy / ability tradeoffs

- **`B` — pinhole splat (default, fastest).** Every photon that hits a
  camera-visible surface splats to the pinhole, so essentially no photons are
  wasted — **orders of magnitude faster** than physically catching photons through
  an aperture. GPU-accelerated. *Cost:* a pinhole has no depth of field, and it
  **cannot render specular-first pixels** (a mirror/glass surface seen directly
  splats nothing and stays black — use `P`, `D`, or `R` for those). Best default
  for diffuse and caustic-heavy scenes.
- **`A` — finite-lens camera (efficient depth of field).** A physical finite
  aperture + thin lens + film, but imaged by **next-event splatting** each photon to
  the lens pupil (like `B`'s splat, through a real aperture instead of a pinhole).
  This gives **true thin-lens depth of field and bokeh** at a fraction of `C`'s cost,
  because photons don't have to physically hit the aperture. `B` is the `aperture→0`
  pinhole limit of this camera; rectilinear only (a fisheye needs a wide-angle element
  the single thin lens can't form). Use when you want DoF without `C`'s noise.
- **`C` — finite-aperture catch (brute-force DoF oracle, slow).** Photons must
  physically pass through the aperture to be counted, giving true thin-lens depth of
  field and bokeh — but it is **catch-starved** (most photons miss the aperture), so
  it is **much noisier / slower** than `A`/`B` for the same photon budget. Mainly the
  ground-truth `A` is validated against; use directly only when you want the
  unapproximated forward-catch.
- **`R` — backward reference (unbiased, general).** Traces from the camera, so it
  renders **any** first-hit surface including specular, and is the **quiet, reliable
  reference** for camera-visible lighting. It also renders **fluorescence** — the
  backward tracer is bispectral, sampling a separate excitation wavelength and
  reradiating with the material's emission colour (so `V` can validate the forward
  fluorescent tracer). GPU-accelerated (its own backward megakernel, which also drives
  the **physical multi-element lens** camera on the GPU). It gets **noisy on caustics**
  (light focused through glass/water is hard to find backward). *GPU scope:* the
  megakernel covers area/sphere/cylinder Lambertian lights and all the
  specular/textured materials; scenes using fog, environment lights, spot/collimated
  lights, or fluorescence fall back to the CPU tracer automatically.
- **`V` — validate.** Runs `B` and `R` and reports their residual; a correctness
  check, not a production renderer (roughly twice the work).
- **`P` — composite (fills in what `B` misses).** Uses fast forward `B` for
  diffuse-first pixels and caustics, and a backward camera ray for
  specular/coated surfaces that `B` leaves black — a good "best of both" for scenes
  that mix diffuse lighting with mirrors/coatings. Both layers are GPU-accelerated
  (the forward layer via the `B` megakernel, the camera-side via `R`'s backward
  megakernel) when the scene is within the backward-GPU scope; otherwise the
  camera-side layer falls back to the CPU. *Cost:* more expensive than plain `B`;
  there can be a subtle seam between the two layers. With a **physical lens** the
  pinhole-splat forward pass can't form the lens image, so `P` automatically routes
  to the lens-aware BDPT (`D`) — or, if the scene is outside BDPT scope (fog / env /
  spot / fluorescence), falls back to the backward realistic camera (`R`).
- **`D` — BDPT (most general, slowest per sample).** One unbiased estimator that
  traces a light *and* a camera subpath and MIS-combines every connection, so it
  captures **specular-first pixels and diffuse caustics in a single pass** on the
  absolute-radiance scale (no composite seam). GPU-accelerated (its own megakernel).
  It also supports the **physical (realistic) lens on its camera subpath** — the
  camera ray is traced through the real glass while forward light transport keeps its
  caustic efficiency (the light-image splat strategy is disabled, since a multi-element
  lens has no closed-form sensor projection; runs on the CPU **and GPU**). *Cost:* highest
  cost per sample, and it **does not support fluorescence, participating media, or
  spot & env lights** (use `B`/`P` or `R` for those).

The **forward modes (`A`/`B`/`C`, and the forward pass of `V`)** are progressive and
GPU-eligible, **`D` has its own GPU BDPT megakernel**, and **`R` (including the
physical-lens camera) has its own GPU backward megakernel** — which the **`P`
composite reuses for its camera-side layer**, so both of `P`'s layers run on the GPU
when the scene is within the backward-GPU scope. Brightness is photon-count-
independent, so more photons only reduce graininess. Outside that scope `P`'s
camera-side layer, and `V`'s backward reference (kept on the CPU as a stable ground
truth), remain CPU-only.

### Backends & performance (`-device`, `-wavefront`)

- **`-device auto` (default, recommended).** Uses the GPU when a supported CUDA
  device is present *and* the render is one it can handle (forward modes
  `A`/`B`/`C` on a non-fluorescent scene, mode `D`'s BDPT megakernel, mode `R`'s
  backward megakernel — including the physical-lens camera — or both layers of the
  mode-`P` composite); otherwise the CPU. Prints its choice.
- **`-device gpu` / `cpu`.** Force the backend. The GPU **falls back to the CPU**
  for the mode-`P` camera-side layer and for `R`/`D` scenes outside their GPU scope
  (fog/env/spot/collimated lights, fluorescence), and for fluorescent/oversized-mix
  forward scenes. It also falls back for any scene using the newer CPU-only features —
  **implicit surfaces (`isosurface`), procedural patterns, and dielectric translucency
  (frosted/colored glass)** — whose device port is pending. `cpu` is fully
  deterministic and is used for reference/validation baselines.
- **`-wavefront` vs. the default megakernel** (GPU forward renders only). Both run
  identical, exactly energy-conserving physics. The **megakernel** runs each
  photon's whole path in one thread and is usually fastest on **shallow, uniform
  scenes on a big GPU**. The **wavefront** splits the trace into coherent
  extend/shade passes over a persistent photon pool and wins on **divergent /
  deep-path scenes and smaller GPUs**. Their RNG streams differ, so their images
  match only to within Monte-Carlo noise.

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
This is the **fast, approximate** option: an ideal paraxial thin lens with a
circular aperture, evaluated analytically, so it runs in the forward modes (`B`
splat / `C` catch) with no per-element ray tracing. It gives correct focus-plane
placement and blur size but **no optical aberrations** (no spherical/chromatic
aberration, distortion, or field curvature) and a perfectly circular bokeh.

**Analytic projections** (fisheye/panoramic/orthographic, above) are likewise a
cheap closed-form remap in the forward pinhole mode — a true wide field of view
with none of the aberration or vignetting a real objective would add.

**Physical (realistic) lens** — `lens { … }`:

A camera can carry a real **lens prescription**: a stack of spherical/planar
glass interfaces plus an aperture stop. The backward tracer samples a film point
and a point on the rear element and traces the ray *through the actual glass*
(per-wavelength Snell refraction), so **depth of field, distortion, spherical &
chromatic aberration, field curvature and vignetting all emerge from the
geometry** — no thin-lens approximation. A physical lens renders in mode `R`
(backward realistic camera) by default, or in mode `D` (BDPT with the lens on the
camera subpath) when you want forward light transport's caustic efficiency through
the glass; mode `P` routes to whichever of those fits the scene.

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

**Analytic vs. simulated — the tradeoff:** the physical lens is the **accurate but
slower** option. It captures real optical behaviour the thin lens cannot
(aberrations, distortion, field curvature, natural vignetting, dispersion-driven
colour fringing, and aperture-shaped bokeh), but it traces every camera ray
through the glass stack, so it is more expensive per sample than the analytic thin
lens. In both mode `R` and mode `D` it **runs on the GPU** (the backward and BDPT
megakernels each refract the camera ray through the glass stack on the device via the
same lens tracer), so within the GPU-supported scope it is still fast. Reach for the
analytic lens/projection when you want speed and a clean ideal image, and the physical
lens when you want a specific real objective's look.

*Current limits:* the lens attaches to the **camera subpath** — mode `R` (backward),
mode `D` (BDPT, keeping forward caustics but with the light-image splat disabled), or
mode `P` (which routes to `D`/`R`). It maps the sensor across the film width, so a
film whose aspect matches the sensor (e.g. `res 360 240` for a 3:2 sensor) covers it
without cropping, while a mismatched aspect crops. It does not model inter-element
flare/ghosting or shaped-iris bokeh. On the GPU it inherits its mode's scope (no
fog/env/spot/fluorescence, and at most 16 lens surfaces); outside that scope it falls
back to the CPU automatically.

---

## Materials

Declared with `material "name" { type <type> … }`.

| Type | Description | Key parameters |
|---|---|---|
| `diffuse` | Lambertian reflector | `reflect` (spectrum or `texture:<name>`) |
| `dielectric` | Refractive glass with dispersion, optional **frosting** and **colored-glass tint** | `ior` (Sellmeier glass or constant); `roughness` (constant or `pattern:`/`texture:` map) frosts the reflected & transmitted lobes; `absorb` (spectrum, σₐ per metre) tints via Beer–Lambert interior absorption |
| `mirror` | Perfect specular reflector | `reflect` |
| `halfmirror` | Lossless beamsplitter; `reflect` is the reflect probability (default 0.5 = 50/50). A spectral `reflect` gives a wavelength-dependent (dichroic) split | `reflect` |
| `glossy` | Rough microfacet reflector | `reflect`, `roughness` (constant or `texture:<name>` map) |
| `thinfilm` | Single-layer interference (iridescence) | `ior`, `film_ior`, `film_thickness` (nm), `film_thickness_map texture:<name>`, `substrate_k` |
| `multilayer` | N-layer Abelès transfer-matrix stack | `ior`, `substrate_k`, repeated `layer <n> <k> <nm>` |
| `grating` | Reflective diffraction grating | `reflect`, `groove_spacing` (nm), `groove_dir`, `max_order` |
| `fluorescent` | Stokes-shifted fluorescence | `reflect`, `absorb`, `emit`, `yield` |
| `mix` | Stochastic blend of materials | repeated `layer <material> <weight>`; optional `weight_map texture:<name>` **or `weight_map pattern:<name>`** (2-child spatial blend mask — with a pattern this becomes a math-driven *per-point material selection*, see Procedural patterns) |
| `layered` | Physical coat over a weighted body: reflect off the coat with prob R, else enter and pick one body lobe (energy-consistent). CPU only | `coat { reflectance fresnel\|thinfilm\|manual, ior, roughness[/roughness_map], film_ior, film_thickness[/film_thickness_map], specular }` + repeated body `layer <material> <weight>` |

**Whole-material presets** (`preset <name>`) fill a complete `Material` from a name:

- **Metals** (polished glossy lobe, override with `roughness`): `gold`/`Au`,
  `silver`/`Ag`, `copper`/`Cu`, `aluminium`/`aluminum`/`Al`,
  `chromium`/`chrome`/`Cr`, `brass`.
- **Glasses** (dispersive `dielectric`): `glass` (=BK7), plus every `glass:<name>`
  below (`BK7`/`crown`, `SF10`/`flint`, `silica`, `sapphire`, `diamond`, `water`,
  `ice`, `acrylic`, `polycarbonate`).
- **Iridescent / structural colour** (thin-film or multilayer stacks): `soap-bubble`,
  `oil-slick`, `anodized-ti`/`anodized-titanium`, `morpho`, `beetle`/`jewel-beetle`,
  `nacre`/`mother-of-pearl`.

**Translucency (dielectrics).** Beyond perfectly clear glass, a `dielectric` supports
two physically-motivated translucency controls (both compose with dispersion):

- **Frosted glass** — a `roughness` (0..1) puts a microfacet lobe on *both* the
  reflected and the refracted ray, so light scatters as it passes through. It accepts a
  constant, a `texture:<name>` map, or a `pattern:<name>` (so frosting can vary over the
  surface — see `scenes/procedural.ftsl`, whose height-banded glass sphere is clear at
  the bottom and frosted at the top).
- **Colored glass** — an `absorb` spectrum (absorption coefficient σₐ per metre)
  attenuates throughput by `exp(-σₐ(λ)·d)` over each in-glass path segment
  (Beer–Lambert), so thick regions tint more than thin edges. Authored like any
  spectrum (e.g. `absorb gaussian center=470 sigma=60 amp=14` for amber). Interior
  absorption is threaded through all three CPU transport loops (forward, backward,
  BDPT); see `scenes/translucency.ftsl`. *(GPU: CPU-only for now — see Known issues.)*

---

## Spectra (SPDs, reflectances, indices)

Anywhere a spectrum is expected (`spd`, `reflect`, `ior`, …) you can write:

- **`preset:<name>`** — illuminants and light sources:
  - **Blackbody / daylight:** `bb<K>` Planckian (e.g. `bb6500`), `sun`,
    `d65`/`daylight`, `a`/`incandescent`.
  - **LED:** `led` (neutral), `led-warm`, and `led<K>k` phosphor LED at a colour
    temperature (e.g. `led4000k`).
  - **Fluorescent:** `fluorescent`/`cfl` (generic compact-fluorescent model) plus the
    measured CIE F-series `f2`/`cool-white`, `f7`/`daylight-fl`, `f11`/`triphosphor`.
  - **Gas-discharge lamps:** `hps`/`sodium` (high-pressure sodium),
    `lps`/`sodium-low` (low-pressure sodium), `mercury`/`hg` (mercury vapor),
    `metal-halide`/`mh`.
- **`rgb r g b`** — Jakob–Hanika sigmoid upsampling to a reflectance spectrum
  (round-trips under D65).
- **`table { 400:0.05 450:0.12 … }`** — a measured/tabulated spectrum
  (piecewise-linear).
- **`file:<path>`** — load a measured curve (SPD, reflectance, or n(λ)) from an
  external CSV/whitespace data file (`#` comments, a header row, `wavelength_nm,value`
  rows); the runtime ingestion point for the data mirrored under `data/`. E.g.
  `spd file:data/spd/cie_f2.csv` (see `scenes/measured_spd.ftsl`).
- **`glass:<name>`** — dispersive index via Sellmeier: `BK7`/crown, `SF10`/flint,
  `silica`/fused-silica, `sapphire`, `diamond`, plus Cauchy fits for `water`,
  `ice`, `acrylic`/PMMA, `polycarbonate`.
- **`metal:<name>`** — measured metal reflectance: `Au`/`gold`, `Ag`/`silver`,
  `Cu`/`copper`, `Al`/`aluminium`, `Cr`/`chromium`, `brass`.
- **`reflectance:<name>`** — measured natural-material diffuse reflectances:
  `leaf`/`vegetation`, `skin`/`skin-light`, `skin-dark`, `snow`, `soil`/`dirt`,
  `brick`/`red-brick`, `concrete`.
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

**Absolute power.** Any non-env light may author a real physical output —
`power <watts>` (radiometric radiant flux) or `lumens <lm>` (photometric luminous
flux, via `Φᵥ = 683·∫spd·V dλ`) — instead of relying on the per-image
auto-exposure. Authoring either on *any* light puts the whole scene in **absolute
mode**: the film is physically linear, the auto-exposure is replaced by a fixed
sensor gain, and `iso`/`shutter`/`exposure` become true absolute stops (doubling
`power` is exactly one stop brighter). See `scenes/absolute.ftsl`.

---

## Geometry

`sphere`, `quad` (parallelogram), `triangle`, and `mesh` (OBJ import, with
`usemtl use_names` for per-face materials and `uv use_mesh` for mesh UVs).
Meshes without their own `vt` coordinates can be textured via a procedural
projection — `mesh { uv planar|spherical|cylindrical [x|y|z] }` synthesizes UVs
at load time from the mesh's world-space bounding box (the optional token is the
projection/up axis, default `y`).
`group { translate … rotate … scale … <children> }` composes transform
hierarchies (baked to world space at load). Everything is accelerated by a BVH.

### Implicit surfaces (`isosurface`)

Besides the explicit primitives above, geometry can be defined *implicitly* as the
zero set of a signed-distance field and rendered by **sphere-tracing** (see
`src/implicit.h`). An `isosurface { material <m>  <one field element> }` block contains
exactly one root **field element**, which is either a **leaf** primitive or a **CSG
combinator** whose children are themselves field elements:

| Leaf | Parameters |
|---|---|
| `sphere` | `center`, `radius` |
| `ellipsoid` | `center`, `radius <rx> <ry> <rz>` (a non-uniformly scaled sphere) |
| `box` | `center`, `size <x> <y> <z>`, `round` (corner-rounding radius, 0 = sharp) |
| `torus` | `major`, `minor` (ring / tube radii; axis = local +y) |
| `cylinder` | `radius`, `height` (axis = local +y) |
| `cone` | `radius` (bottom), `radius2` (top, 0 = pointed), `height` |
| `plane` | `normal`, `offset` |

| Combinator | Meaning |
|---|---|
| `union` / `intersect` / `difference` | hard boolean CSG (min / max / subtract) |
| `smooth_union` / `smooth_intersect` / `smooth_difference` | filleted boolean; blend radius `k` softens the seam |
| `blob` | alias for `smooth_union` — with `k`, children fuse like **metaballs** |

Every element (leaf *or* combinator) may carry its own `translate` / `rotate` / `scale`.
To rotate a leaf **in place**, position it with `translate` (applied outside the
rotation) rather than `center` (applied inside — it would orbit the world origin).
Non-uniform scale is supported (the field stays a valid Lipschitz-1 SDF by de-rating
the step to the smallest axis scale), so an `ellipsoid`, a squashed `box`/`torus`, etc.
all work. Surface normals come from the analytic field gradient. A worked example with
metaballs, drilled CSG, and a tilted torus is in `scenes/implicit.ftsl`. *(GPU: implicit
surfaces are CPU-only for now — see Known issues.)*

## Textures

`texture "name" { file <path> encoding srgb|linear filter nearest|bilinear wrap
repeat|clamp|mirror }` loads PNG / JPG / HDR / PPM / PFM images; bind one to a
diffuse albedo with `reflect texture:<name>`. Each texel is Jakob–Hanika
upsampled to a reflectance spectrum. UVs come from quad corners, OBJ `vt`
(`uv use_mesh`), a procedural `planar`/`spherical`/`cylindrical` projection, or
per-hit `triplanar` box projection for un-UV'd meshes (see Geometry). Besides
base-colour albedo, a texture can also drive a **scalar** parameter: a grayscale
**roughness map** on `glossy` (`roughness texture:<name>`) or a **film-thickness
map** on `thinfilm` (`film_thickness_map texture:<name>`, a 0..1 profile × the
nominal `film_thickness`). All of these run on both the CPU and GPU forward paths.
A texture can also be an **indexed-spectral palette** — `palette { 0 spectrum:navy
1 spectrum:crimson … }` maps red-channel indices (0..255) to named reflectance
spectra, looked up nearest (CPU only; GPU falls back). A 2-child `mix` can take a
**blend mask** (`weight_map texture:<name>`) that selects child 0 vs child 1 per hit.
A scalar map on `ior` remains future work.

## Procedural patterns (math-driven materials)

A `pattern "name" { … }` block compiles a **scalar field** — a function of the hit
point evaluated per shading sample — that can drive any scalar material parameter
*procedurally*, without a texture image. The variables available to a pattern are the
world-space position `x y z`, the implicit field value `f` (the SDF value at the hit,
`~0` on an isosurface; `0` for explicit geometry), the surface normal `nx ny nz`, and
the radius `r = √(x²+y²+z²)`. Two authoring forms:

- **Free-form expression** — `expr "0.5 + 0.5*sin(40*y)"` (must be quoted). Compiled by
  a shunting-yard parser to a postfix scalar VM. Supports `+ - * / ^ %`, comparison-free
  math, `pi`, and functions `abs sqrt sin cos tan exp log floor fract sign saturate min
  max atan2 step pow clamp mix smoothstep noise`.
- **Named generator** — `type <gen>` plus params (mirrors material syntax):

  | Generator | Parameters | Result |
  |---|---|---|
  | `axis`    | `axis <x\|y\|z>` `[scale]` `[offset]` | a coordinate ramp |
  | `radial`  | `[center <x y z>]` `[scale]` | distance from a point |
  | `bands`   | `axis <x\|y\|z>` `[freq]` `[phase]` | `0.5+0.5·sin(2π·freq·coord+phase)` stripes |
  | `checker` | `[size]` | 0/1 world-space checkerboard |
  | `noise`   | `[freq]` | deterministic value noise in [0,1] (CPU/GPU bit-identical) |
  | `field`   | `[scale]` | the raw implicit field value `f`, scaled |

Bind a pattern anywhere a scalar `texture:<name>` map is accepted, using
`pattern:<name>` instead: **roughness** (`dielectric`/`glossy`/preset/`layered` coat),
**film thickness** (`thinfilm`, preset), and a 2-child `mix` **`weight_map`**. The
`weight_map` case is the powerful one: because a `mix` blends whole materials, a pattern
weight makes the *material itself* — colour **and** BSDF type — vary from point to
point (checkerboard of red vs green diffuse, noise-selected metal vs glass, …). See
`scenes/procedural.ftsl`. *(GPU: patterns are CPU-only for now — see Known issues.)*

## Participating media / fog

`medium { sigma_t <v> albedo <v> g <v> rayleigh <bool> }`, or from the CLI with
`-fog <sigma_t> -fogalbedo <a> -fogg <g> [-fograyleigh]`. Henyey–Greenstein phase
function by default; Rayleigh optional.

---

## Scene language (FTSL)

An FTSL file is a list of blocks. Top-level block types: `scene` (the
`units …` / `spectral …` header), `material`, `texture`, `pattern` (procedural scalar
field), `spectrum`, `sphere`, `quad`, `triangle`, `mesh`, `isosurface` (implicit SDF
surface / CSG / metaballs), `light`, `group`, `medium`, `camera`, `camera_path`
(keyframed camera animation), and `render` (render-setting overrides). See the
`scenes/` directory for worked examples (`cornell.ftsl`, `fisheye.ftsl`,
`spotlight.ftsl`, `envlight.ftsl`, `material_presets.ftsl`, `realcam.ftsl`,
`implicit.ftsl`, `procedural.ftsl`, `translucency.ftsl`, …).

### Importing Mitsuba scenes

`tools/mitsuba_to_ftsl.py` converts a Mitsuba (0.6 / 2 / 3) XML scene to FTSL:

```
python tools/mitsuba_to_ftsl.py scene.xml scene.ftsl
ftrace -in scene.ftsl -mode D -o out.png
```

Mitsuba is also a spectral, physically-based renderer, so most constructs map
almost 1:1: the `perspective`/`thinlens` sensor → an FTSL `camera` (`thinlens`
becomes mode `A` with aperture + focus), `diffuse`/`conductor`/`roughconductor`/
`dielectric`/`plastic` BSDFs → `diffuse`/`mirror`/`glossy`/`dielectric` materials,
and `area`/`constant`/`envmap` emitters → FTSL lights. `rectangle`, `cube`,
`sphere`, and `obj` shapes are supported (with full `to_world` transforms); RGB
reflectances ride FTSL's Jakob–Hanika upsampling and measured/blackbody spectra
pass through losslessly. Constructs outside FTSL's scope (rough transmission,
bump/normal maps, `.ply`/`.serialized` meshes, mesh area-emitters) degrade to a
documented approximation and are flagged with `# WARN:` comments in the output.
Since **Blender can export directly to Mitsuba XML** (via the `mitsuba-blender`
add-on), this doubles as a Blender → FTSL path.

---

## Command-line reference

**Core**

| Flag | Meaning |
|---|---|
| `-in <path>` | Load an FTSL scene file |
| `-scene <name>` | Built-in scene (e.g. `cornell`) |
| `-n <photons>` | Trace exactly this many photons/samples |
| `-r <res>` / `-r <W> <H>` | Output resolution (overrides scene default); one value = square, two = non-square film |
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
| `-exposure-lock` | Share one auto-exposure anchor across all rendered cameras (no `camera_path` flicker); a per-path `exposure_lock` keyword locks just that path |

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
the physical-lens camera's remaining gaps (inter-element flare/ghosting,
shaped-iris bokeh) and the shared multi-camera pass.
