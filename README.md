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
- **Participating media** — one or many coexisting (superposed) fog regions with
  Henyey–Greenstein or Rayleigh scattering; box / sphere / **named-object** bounds
  (fog shaped to a sphere, isosurface field, or mesh AABB) and heterogeneous
  **density fields** — either formula-defined blobs with soft edges *or* imported
  **`.nvdb` (NanoVDB) volumes** (`density vdb:<file>`) — via unbiased delta/ratio
  tracking on the forward modes (CPU and GPU).
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

> **Windows + CUDA gotcha.** With the Visual Studio generator, CUDA auto-detection
> needs the CUDA **VS integration** (MSBuild props), not just `nvcc` on `PATH`. If
> configure prints `CUDA not found; building CPU-only`, point the toolset at the CUDA
> install directly and select the VS instance that has the integration, e.g.:
> ```sh
> cmake -B build -S . -G "Visual Studio 17 2022" -A x64 \
>   -T "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" \
>   -DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Community"
> ```
> A CUDA-linked `ftrace.exe` is ~3 MB vs ~0.8 MB for a CPU-only build — a quick size
> check tells you which you got.

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
| `M` | Photon map | Builds a **view-independent** photon map once, then gathers the camera image from it — a direct radius density estimate at the first diffuse hit, or a Jensen final gather one bounce away with `-pmfg <K>` (reusable across cameras) | CPU |
| `S` | SPPM | Stochastic **progressive** photon mapping: repeated photon passes with a shrinking per-pixel radius — converges (unbiased in the limit), bounded memory, excels at caustics | CPU |
| `U` | VCM/UPS | Vertex **connection and merging**: BDPT vertex connections **and** SPPM photon merging combined under one MIS weight — robust across diffuse GI, glossy, and caustics in a single estimator | CPU |

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
  lens has no closed-form sensor projection; runs on the CPU **and GPU**). It renders
  **participating media of every kind** — global haze, multiple superposed media,
  box/sphere/object-bounded fog, and **heterogeneous `density`-field blobs** — with volume
  in-scatter vertices, HG-phase connections and transmittance-weighted edges (subpath
  medium vertices placed by delta tracking, connections weighted by ratio-tracking
  transmittance), so fog *inside a glass shell* images correctly here (a case the
  next-event modes leave dark). *Cost:* highest cost per sample, and it **does not support
  fluorescence or spot & env lights** (use `B`/`P` or `R` for those).
- **`M` — photon map (view-independent, reusable).** Traces a forward photon pass
  **once** and stores every diffuse deposit in a **view-independent photon map** (a
  uniform hash grid), then forms the camera image by a backward camera pass. By default
  it uses a **direct density query**: each camera ray walks through specular surfaces
  until it lands on a diffuse one, where a radius density estimate over the nearby
  photons gives the radiance directly at that surface. Optionally, `-pmfg <K>` switches
  to a **Jensen final gather**: at the first diffuse hit it shoots `K` cosine-weighted
  hemisphere sub-rays, traces one bounce each, and queries the map at *those* points —
  so the density estimate's blur lives one bounce away instead of on the visible surface
  (direct light at the visible point is recovered by gather rays that strike an emitter
  directly). Because the map is independent of the camera, it can be **built once and
  reused across every frame of a flythrough** (or every camera of a multi-camera render)
  — the cost of the photon pass amortizes over all views. *Cost:* with the direct query
  the density estimate **blurs sharp contact shadows** at large gather radii (bias
  controlled by `-pmradius` / `-pmradiusfrac`); final gather (`-pmfg`) keeps those
  contact shadows and fine detail **sharp** while still smoothing indirect light, at
  roughly `K`× the per-sample cost (so pair it with fewer `-spp`). Directly-viewed
  emitters carry a little chromatic speckle at low spp. Best when many cameras share one
  lighting solution. CPU only. (Matches the forward splat modes `A`/`B`/`C` — same
  forward physics, just measured from a stored map.)
- **`S` — SPPM (progressive, caustic-strong).** Stochastic progressive photon mapping
  (Hachisuka 2008/2009): instead of one fixed-radius map, it runs **repeated bounded
  photon passes** and **shrinks each pixel's gather radius** over iterations, so the
  estimate is **unbiased in the limit** with **flat memory** (the map is rebuilt small
  each pass, never grown). Each pass also re-samples the camera subpaths, which
  anti-aliases and makes it robust for depth of field / glossy. Its stand-out strength is
  **caustics and SDS paths** — light focused through glass or off metal, which the
  backward tracer (`R`) and even BDPT (`D`) resolve slowly but a photon method captures
  directly. `-n` is photons **per pass**, `-spp` is the **number of passes** (or use a
  `-time`/`-noise`/`-forever` budget); the radius-shrink rate is `-sppmalpha` (default
  `0.7`) and the initial radius reuses `-pmradius`/`-pmradiusfrac`. A single pass reduces
  exactly to mode `M`. CPU only. *Cost:* many passes to converge; the running preview
  starts blurry (large radius) and sharpens as the radius shrinks.
- **`U` — VCM/UPS (the "have it all" estimator).** Vertex Connection and Merging
  (Georgiev et al. 2012, a.k.a. Unified Path Sampling): each pass traces a **light
  subpath and a camera subpath per pixel**, and combines **every** BDPT-style vertex
  **connection** (what `D` does — great for diffuse/glossy interreflection connected
  directly to the light) **and** every SPPM-style photon **merge** (what `S` does — great
  for caustics / SDS focusing) under **one multiple-importance-sampling (balance-
  heuristic) weight**. That single weighting makes it robust across the whole gamut: it
  matches the backward tracer on diffuse GI *and* resolves caustics like a photon method,
  with no per-scene mode picking. Like SPPM it is **progressive** and **unbiased in the
  limit**, shrinking the merge radius as `r_i = R0·i^((alpha-1)/2)` across passes. `-n` is
  **ignored** (light-path count follows the film resolution); `-spp` is the **number of
  passes** (or a `-time`/`-noise`/`-forever` budget); the radius-shrink rate is
  `-vcmalpha` (default `0.75`) and the initial radius reuses `-pmradius`/`-pmradiusfrac`.
  CPU only. *Cost:* the heaviest per-pass (both a full light pass and a full camera pass,
  plus a grid build), but the most consistent quality per pass — at equal time it beats
  SPPM on caustics *and* stays as clean as BDPT on diffuse GI. (Single-wavelength note:
  connections pair a camera path with its **own** light path so they share one wavelength
  and are exact; merges gather photons from other paths, so like modes `M`/`S` they use
  the standard spectral-photon-mapping XYZ estimate.)

The **image-forming modes are all progressive** — the forward camera models
(`A`/`B`/`C`), the backward reference (`R`), the bidirectional tracer (`D`), and the
composite (`P`) each refine an image whose brightness is fixed while only graininess
falls, so they share the same live progress and budget flags (`-time` / `-noise` /
`-forever` / `-preview` / `-interval`, and periodic crash-safe writes) on **both** the CPU
and the GPU. They're all GPU-eligible too: **`A`/`B`/`C` and the forward pass of `V`** via
the forward megakernel, **`D`** via its own GPU BDPT megakernel, and **`R` (including the
physical-lens camera)** via its own GPU backward megakernel — which the **`P` composite
reuses for its camera-side layer**, so both of `P`'s layers run on the GPU when the scene
is within the backward-GPU scope. Outside that scope `P`'s camera-side layer, and `V`'s
backward reference (kept on the CPU as a stable ground truth), remain CPU-only. The
composite `P` classifies its pixels once, then alternates forward and backward batches
into two accumulating films, re-fitting the forward→backward scale and re-blending each
interval. **Disk `-resume`/`-checkpoint` now cover `A`/`B`/`C` (photon-count checkpoint),
`R`/`D` (spp-count checkpoint), and `P` (dual forward+backward film)** — a resumed render
draws a decorrelated sample stream so its added samples genuinely reduce variance. Only the
persistent-state photon modes `M`/`S`/`U` (whose per-pass state a film alone can't restore)
stay non-resumable.

### Backends & performance (`-device`, `-wavefront`)

- **`-device auto` (default, recommended).** Uses the GPU when a supported CUDA
  device is present *and* the render is one it can handle (forward modes
  `A`/`B`/`C` on a non-fluorescent scene, mode `D`'s BDPT megakernel, mode `R`'s
  backward megakernel — including the physical-lens camera — or both layers of the
  mode-`P` composite); otherwise the CPU. Prints its choice.
- **`-device gpu` / `cpu`.** Force the backend. The GPU **falls back to the CPU**
  for the mode-`P` camera-side layer and for `R`/`D` scenes outside their GPU scope
  (env/spot/collimated lights, fluorescence; any fog for `R`), and for
  fluorescent/oversized-mix forward scenes. Mode `D`'s GPU BDPT megakernel renders
  **all** participating media — haze, superposed, bounded, and heterogeneous
  `density`-field fog — directly on the device. Implicit surfaces / `isosurface`, **procedural patterns**, and
  **dielectric translucency** (frosting + Beer–Lambert colored-glass tint) are all
  GPU-accelerated now — the device sphere-traces the same field expressions, runs the
  same pattern VM, and threads the interior-absorption medium through both the forward
  and backward tracers. GPU **BDPT** (mode `D`) still falls back for any pattern-driven
  material *or* frosted/colored glass, whose per-hit BSDF its MIS kernel can't yet
  reproduce. `cpu` is fully deterministic and is used for reference/validation baselines.
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
| `translucent` | Two-sided Lambertian (**diffuse transmission** / thin-subsurface look) — light diffuses THROUGH the surface, so a backlit sheet glows softly. Front hemisphere scatters `reflect`, back hemisphere scatters `transmit`; non-specular, so it connects/renders in every mode (A/B/C/R/V/D/P). CPU only. Alias `diffuse_transmit` | `reflect` (spectrum or `texture:<name>`), `transmit` (spectrum); the two are energy-clamped so `reflect+transmit ≤ 1` |
| `dielectric` | Refractive glass with dispersion, optional **frosting** and **colored-glass tint** | `ior` (Sellmeier glass or constant); `roughness` (constant or `pattern:`/`texture:` map) frosts the reflected & transmitted lobes; `absorb` (spectrum, σₐ per metre) tints via Beer–Lambert interior absorption |
| `mirror` | Perfect specular reflector | `reflect` |
| `halfmirror` | Lossless beamsplitter; `reflect` is the reflect probability (default 0.5 = 50/50). A spectral `reflect` gives a wavelength-dependent (dichroic) split | `reflect` |
| `filter` | Colored **gel / Wratten filter**: a thin non-scattering absorber. Light passes straight through (no reflection or refraction), surviving with probability `transmit`(λ) — the per-wavelength transmittance T(λ) ∈ [0,1] — and is absorbed otherwise. Like clear glass it isn't lit directly; you see its effect on whatever is behind it | `transmit` (spectrum: `filter:<name>`, `file:<path>`, or a primitive like `gaussian`) |
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

Each iridescent preset is a **bundle file** (`data/material/<name>.material`) that
groups the material's several spectral envelopes (`ior`, `substrate_k`) and its tuned
film/stack geometry (`film_thickness`/`film_ior` or `layer <n> <k> <nm>` rows) under
one name — so new structural-colour materials drop in with **no rebuild**. Metals and
glasses need no file: a bare `metal:`/`glass:` name resolves by the generic convention
above. (The interference math stays native; only the parameters are data.)

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
  BDPT); see `scenes/translucency.ftsl`. *(GPU: forward + backward `R` accelerate both
  frosting and colored-glass tint; mode-`D` BDPT still falls back to the CPU.)*

---

## Spectra (SPDs, reflectances, indices)

Anywhere a spectrum is expected (`spd`, `reflect`, `ior`, …) you can write:

- **`preset:<name>`** — illuminants and light sources:
  - **Blackbody / daylight:** `bb<K>` Planckian (e.g. `bb6500`), `sun`,
    `d65`/`daylight`, `a`/`incandescent`.
  - **White LED:** `led` (neutral), `led-warm`, and `led<K>k` phosphor LED at a colour
    temperature (e.g. `led4000k`).
  - **Colored LED:** single-die narrow-band emitters `led-violet`, `led-royal-blue`,
    `led-blue`, `led-cyan`, `led-green`, `led-amber`, `led-red`, `led-deep-red`
    (measured die SPDs, Brendel 2021, CC BY-SA 4.0).
  - **Fluorescent:** `fluorescent`/`cfl` (generic compact-fluorescent model) plus the
    measured CIE F-series `f2`/`cool-white`, `f7`/`daylight-fl`, `f11`/`triphosphor`.
  - **Gas-discharge lamps:** `hps`/`sodium` (high-pressure sodium),
    `lps`/`sodium-low` (low-pressure sodium), `mercury`/`hg` (mercury vapor),
    `metal-halide`/`mh` (analytic line model), plus measured ceramic-metal-halide
    SPDs `cmh`/`cmh-3000k` (warm white) and `cmh-4200k` (cool white), digitized from
    the GE ConstantColor CMH G12 datasheet and colour-matched to its published
    chromaticity.
- **`rgb r g b`** — Jakob–Hanika sigmoid upsampling to a reflectance spectrum
  (round-trips under D65).
- **`table { 400:0.05 450:0.12 … }`** — a measured/tabulated spectrum
  (piecewise-linear).
- **`file:<path>`** — load a measured curve (SPD, reflectance, or n(λ)) from an
  external CSV/whitespace data file (`#` comments, a header row, `wavelength_nm,value`
  rows); the runtime ingestion point for the data under `data/`. E.g.
  `spd file:data/illuminant/f2.csv` (see `scenes/measured_spd.ftsl`).
- **`glass:<name>`** — dispersive index via Sellmeier: `BK7`/crown, `SF10`/flint,
  `silica`/fused-silica, `sapphire`, `diamond`, plus Cauchy fits for `water`,
  `ice`, `acrylic`/PMMA, `polycarbonate`.
- **`metal:<name>`** — measured metal reflectance: `Au`/`gold`, `Ag`/`silver`,
  `Cu`/`copper`, `Al`/`aluminium`, `Cr`/`chromium`, `brass`.
- **`reflectance:<name>`** — measured natural-material diffuse reflectances:
  `leaf`/`vegetation`, `skin`/`skin-light`, `skin-dark`, `snow`, `soil`/`dirt`,
  `brick`/`red-brick`, `concrete`.
- **`filter:<name>`** — gel/Wratten filter transmittances T(λ) (for a `filter`
  material's `transmit`): the **complete 84-filter Kodak Wratten set**, named
  `wratten-<n>` (e.g. `wratten-25`, `wratten-34a`, `wratten-47b`). Descriptive
  aliases resolve too: `red-25` (`red`), `deep-red-29`, `orange-21`, `yellow-12`,
  `green-58`, `blue-47`, `deep-blue-47b`, `magenta`, `cyan`, ….
- **`spectrum "name" { … }`** blocks to define and reuse a named SPD.

The `glass:`, `metal:`, `reflectance:`, `filter:` and `preset:` (illuminant) presets —
plus the whole-material `preset <name>` recipes and the named light presets — are a
**drop-in spectral asset library**: their data lives in external files under
`data/{glass,metal,reflectance,illuminant,filter,material,light}/`, loaded at runtime — add a
file to a category directory and it resolves by name with **no rebuild** (the
lowercased filename is the preset name; a `# aliases:` header line adds more). The
`material/` and `light/` files are *bundles* that group several envelopes plus scalars
into one named asset (a thin-film material owns an index curve, a substrate-extinction
curve and film thickness/index at once). Only the data is external; the dispersion
evaluators, interference/BSDF math and light models stay in the renderer. See
`data/README.md`.

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

`sphere`, `quad` (parallelogram), `triangle`, and `mesh` (**OBJ and glTF 2.0 /
GLB** import — the loader dispatches on file extension). glTF brings its node
transform hierarchy, per-vertex normals/UVs, and `pbrMetallicRoughness` materials
(base color upsampled to a reflectance spectrum, metallic → glossy tint, roughness
→ lobe width; `import_materials no` forces the FTSL `material` instead). OBJ
supports `usemtl use_names` for per-face materials and `uv use_mesh` for mesh UVs.
OBJ **vertex normals (`vn`) are read as smooth shading normals** — a hit
barycentric-interpolates them (CPU and GPU) for smooth-shaded curved meshes, with
no visible faceting; a mesh with no `vn` stays exactly flat-shaded (geometric
normal). Meshes without their own `vt` coordinates can be textured via a procedural
projection — `mesh { uv planar|spherical|cylindrical [x|y|z] }` synthesizes UVs
at load time from the mesh's world-space bounding box (the optional token is the
projection/up axis, default `y`).
`group { translate … rotate … scale … <children> }` composes transform
hierarchies (baked to world space at load).

**Instancing.** `mesh_asset "name" { file … material … }` loads a mesh once into
its local space; `mesh_instance { of "name"  translate … rotate … scale …
[material …] }` places that shared geometry through a per-copy affine. Instances
share one triangle set and one bottom-level BVH — a **two-level BVH** (TLAS over
instances → shared BLAS) — so N copies cost N affines instead of N triangle sets,
and a per-instance `material` can override the asset's own materials. Works in
every render mode, and the memory sharing holds on **both** the CPU and the GPU: the
device also uses a true two-level BVH (shared per-BLAS pools + an instance table that
transforms the ray into BLAS space), so device memory scales with unique geometry, not
with the instance count. Everything is accelerated by a BVH.

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
| `function` | `expr "f(x,y,z)"` — arbitrary formula leaf (see below) |

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
metaballs, drilled CSG, and a tilted torus is in `scenes/implicit.ftsl`. Implicit
surfaces are sphere-traced on **both the CPU and the GPU** (the device port matches the
CPU to Monte-Carlo noise).

#### Arbitrary-formula isosurfaces (`function`)

The analytic leaves above are all built-in signed-distance fields. To render the
surface of an **arbitrary equation** `f(x,y,z) = 0` — a gyroid, a Goursat/heart shape,
any hand-typed formula — use a `function` leaf:

```
isosurface {
    material gold
    function {
        translate 0.5 0.5 0.45           # (optional) place / rotate / scale the field frame
        expr "sin(28*x)*cos(28*y) + sin(28*y)*cos(28*z) + sin(28*z)*cos(28*x)"
    }
    contained_by { min 0.3 0.3 0.25   max 0.7 0.7 0.65 }   # REQUIRED bound box
    max_gradient 48                       # (optional) Lipschitz bound; auto-estimated if omitted
    accuracy 1e-4                         # (optional) march-step floor, world units
}
```

The `expr` string is compiled by the **same math VM as procedural patterns** (variables
`x y z` and `r = |p|`, plus `sin cos tan exp log sqrt abs floor fract sign min max pow
atan2 clamp mix smoothstep noise`, and the constant `pi`). Because an arbitrary field is
**not** a signed distance and has no analytic bound, a `function` isosurface **must**
supply a `contained_by { min <x y z>  max <x y z> }` box (the region the surface is
marched inside). Safe sphere-tracing needs a **Lipschitz bound** `L ≥ max|∇f|` so a step
of `|f|/L` never overshoots the first zero crossing; give it explicitly with
`max_gradient`, or omit it and the loader auto-estimates it by sampling `|∇f|` over the
box (padded ×1.3). `accuracy` overrides the march-step floor. A `function` leaf also
composes inside CSG combinators (`union`, `difference`, `blob`, …) like any other leaf.
The worked gyroid example is in `scenes/function.ftsl`; expression isosurfaces run on
**both the CPU and the GPU** (the device evaluates the identical formula VM — an
expression sphere matches the analytic `sphere` leaf to RMSE ≈ 0.15 % on the same
backend).

**Container shape and caps.** The container can be a box **or a sphere**, and you can
choose whether the container *seals* the solid it cuts:

```
isosurface {
    material gold
    function { expr "f_enneper(x, y, z, 1)"  scale 1.7  translate 0 1.2 0 }
    contained_by { sphere { center 0 1.2 0  radius 1.7 } }   # curved boundary
    max_gradient 20
    open                                                     # (optional) don't cap
}
```

`contained_by { sphere { center <x y z>  radius r } }` clips the ray along a **smooth
curved boundary** instead of the axis-aligned `min`/`max` box, so an *unbounded* surface
(e.g. `f_enneper`, or a solid that pokes out of the container) reads as a natural rounded
edge rather than hard box facets. The sphere `center`/`radius` are taken in the field's
frame and transformed to world (exact under uniform scale; a conservative bounding sphere
under rotation/shear). Where the container wall slices through **solid** material
(`f < 0`), it is **capped** by default — sealed with a flat/curved face of the isosurface
material (a cleanly sawn-off solid). The **`open`** keyword suppresses those caps, leaving
the surface's cut edge and a see-through opening into the interior (`open off` forces the
default). Caps only affect surfaces that actually reach the container wall; a fully
bounded surface never touches it, so the choice is moot. Both the container shape and the
cap policy run identically on CPU and GPU.

##### POV-Ray internal functions (`f_torus`, `f_heart`, …)

The formula VM also ships the **complete set of POV-Ray's built-in isosurface
functions** — the classic `functions.inc` library (`f_torus`, `f_heart`,
`f_klein_bottle`, `f_superellipsoid`, `f_dupin_cyclid`, `f_helix1`, `f_spiral`,
`f_kummer_surface_v1/v2`, `f_boy_surface`, `f_steiners_roman`, and ~60 more). They are
**exact ports of POV-Ray's own C++ source** (`source/vm/fnintern.cpp`), so a scene using
them evaluates to the same field POV-Ray computes — call them straight from any `expr`
string, no `#include` needed:

```
isosurface {
    material copper
    function {
        expr "f_torus(x, y, z, 0.28, 0.10)"    # major R = 0.28, minor r = 0.10
        translate 0.5 0.5 0.5   rotate 62 0 0
    }
    contained_by { min 0.05 0.05 0.05   max 0.95 0.95 0.95 }
    max_gradient 1.5
}
```

Exactly as in POV-Ray, the **first three arguments are the coordinates** (`x, y, z`,
which you may pre-transform) and the rest are the function's parameters — e.g.
`f_torus(x,y,z, majorR, minorR)`, `f_heart(x,y,z, strength)`,
`f_superellipsoid(x,y,z, e, n)`. Distance-like functions (`f_torus`, `f_sphere`,
`f_rounded_box`, `f_helix1`) are ~unit-Lipschitz and march robustly; the many
**polynomial** surfaces (`f_heart`, `f_klein_bottle`, `f_dupin_cyclid`, …) are not signed
distances, so give them a `max_gradient` (POV clamps most of them to ±10, so a bound in
the 5–50 range usually works).

The **noise-based** entries — `f_noise3d`, `f_noise_generator`, `f_ridge`, `f_ridged_mf`,
`f_hetero_mf` — are supported too, driven by an **exact host+device port of POV-Ray's
Perlin `Noise()`** (`src/pov_noise.h`, generated by `tools/pov_noise_gen.py`): POV's init
tables are re-derived and baked in, so a scene like
`sqrt(x*x+z*z) - 1 + 0.5*f_noise3d(3*x,3*y,3*z)` yields the same bumpy field POV produces,
identically on CPU and GPU (all three noise generators: 1=Original, 2=RangeCorrected
[default], 3=Perlin). Only `f_pattern` (which needs POV's full pattern/pigment engine)
remains unported — tracked in `known-issues.md`. The functions are generated by
`tools/pov_functions_gen.py` into `src/pov_functions.h` and evaluated by the same
host+device VM, so they render identically on CPU and GPU.

##### Ray-march strategy (`method`, `refine`)

Any `isosurface` (analytic *or* `function`) chooses how the ray finds the field's first
zero crossing:

| Key | Values | Meaning |
|---|---|---|
| `method` | `adaptive` (default) / `sample` | how the ray steps toward the surface |
| `samples` | `<n>` | *sample mode only* — number of fixed steps across the container's diagonal (default 256; or size the step with `accuracy`) |
| `refine` | `bisect` (default) / `regula_falsi` | how a bracketed sign change is refined to the root |

- **`adaptive`** steps by `max(|f|/max_gradient, accuracy)` — sphere-tracing for a true
  SDF (`max_gradient = 1`), or a Lipschitz-bounded march for a `function` field. With a
  correct `max_gradient` it **provably cannot skip** the first crossing (across one step
  `f` can change by at most the step size, so it can't dip through zero and back), and it
  slows down only near surfaces. This is the right choice almost always.
- **`sample`** ignores `|f|` and marches by a **fixed** world step (POV-Ray's sampling
  mode). It needs **no** Lipschitz bound, so it's the fallback when `max_gradient` can't be
  trusted (spiky/near-unbounded gradients where the auto-estimate is unreliable) — but a
  feature thinner than one step *between two samples* can be missed, so raise `samples`
  until the surface is clean.
- **`refine`** only changes root-polishing speed, not the result: `bisect` is
  unconditionally robust (linear); `regula_falsi` (secant with the Illinois safeguard)
  converges faster on smooth brackets. Both land on the same root to ~1e-12.

Validated: on a clean surface the `sample` and `adaptive` marchers agree to RMSE ≈ 0.4/255
(CPU) / 0.01/255 (GPU) — same geometry, both backends. `scenes/function.ftsl` uses the
adaptive default; `method sample` + `samples`/`refine` are shown in the scraps test
scenes.

##### Exporting an isosurface to a mesh (`-export-mesh`)

Any scene's isosurfaces can be **polygonised into a watertight triangle mesh** and written
as an OBJ (for import into Unreal, Blender, etc.) instead of being rendered:

```
ftrace -in scene.ftsl -export-mesh out.obj -mesh-res 192
ftrace -in scene.ftsl -export-mesh out.obj -mesh-res 256 -mesh-adaptive -mesh-decimate 0.35
```

| Flag | Meaning |
|---|---|
| `-export-mesh <file.obj>` | polygonise every `isosurface` in the scene (marching **tetrahedra**), write an OBJ, then exit (no render). Each isosurface becomes one OBJ object (`o isosurface_k`). |
| `-mesh-res <N>` | **fineness** — grid cells along the longest bounds axis (default 128). The other axes get proportional counts so cells stay ~cubic. Higher = more triangles / finer detail. |
| `-mesh-adaptive` | after marching, run a curvature-adaptive **quadric-error decimation** pass. |
| `-mesh-decimate <f>` | adaptive target: keep this fraction of triangles (default 0.5; implies `-mesh-adaptive`). |

The exporter reuses the **exact field the renderer sees** — `f(x,y,z)` for edge crossings and
`∇f` for normals — so the mesh matches the rendered surface. It uses **marching tetrahedra**
(Kuhn/Freudenthal 6-tet split of each cell) rather than marching cubes: tetrahedra have no
face-ambiguous cases, so the output is a guaranteed **watertight 2-manifold** (marching cubes
can leave holes / non-manifold edges). The field is **intersected with its `contained_by`
domain box** (a CSG `max(f, boxSDF)` over a lattice padded a couple cells beyond the box), so a
surface that reaches the boundary is sealed with a flat cap into a **closed solid** instead of
leaving an open rim. Vertices are welded by a canonical grid-edge id (adjacent cells reference
one vertex ⇒ **no cracks**), crossings are refined by bisection on the real field, per-vertex
normals come from the field gradient (box-face normals on caps), and each triangle is wound so
its geometric normal points outward.

The **adaptive** pass collapses cheap edges first: the quadric error is near-zero on flat
regions (a vertex can slide freely) and large where the surface curves, so triangles thin out
on flat areas and stay dense on detailed ones — the requested curvature-driven tessellation. A
**link-condition** test plus foldover rejection keep the mesh a watertight 2-manifold through
the collapses. (The mesher runs on the CPU; it reads `Implicit::eval`/`gradient` from
`src/isomesh.h`.)

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
`~0` on an isosurface; `0` for explicit geometry), the surface normal `nx ny nz`, the
radius `r = √(x²+y²+z²)`, and the **surface UV coordinates `u v`** (mesh-interpolated,
or a native-primitive wrap — see below). Two authoring forms:

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
`scenes/procedural.ftsl`. *(GPU: patterns run on the device forward and backward
paths, including a roughness pattern on a `dielectric` (frosted glass). GPU BDPT is the
exception — its MIS kernel falls back to the CPU for any pattern or frosted/colored
glass.)*

**UV on native primitives.** The `u v` pattern variables aren't limited to meshes.
A native `sphere {}` carries built-in equirectangular (lat/long) UVs, a `quad {}`
maps its `u`/`v` edges to (0,0)→(1,1), and an `isosurface` can request a procedural
wrap with `uv planar|spherical|cylindrical [axis=x|y|z]` — the **same projection used
for un-`vt`'d meshes**, referenced to the surface's world bounds. So a checker or
stripe authored in `(u,v)` space wraps *around* the object (a globe, tiles converging
at the poles, a grid on box faces) instead of slicing through world space. See
`scenes/uv_native.ftsl`.

## Participating media / fog

`medium { sigma_t <v> albedo <v> g <v> rayleigh <bool> }`, or from the CLI with
`-fog <sigma_t> -fogalbedo <a> -fogg <g> [-fograyleigh]`. Henyey–Greenstein phase
function by default; Rayleigh optional.

**Multiple, overlapping media.** Author as many `medium` blocks as you like — they
coexist as independent regions (e.g. two differently-tinted fog orbs plus a faint
global haze). The forward tracer superposes them physically: extinction adds (total
transmittance is the *product* of the per-medium transmittances) and each scatter is
drawn from the *earliest* of the media's independent free-flights. A single-`medium`
scene is bit-identical to before.

**Bounded, heterogeneous fog (blobs).** The medium isn't limited to a single global
haze. Add `bounds { min <x y z> max <x y z> }` to confine it to an axis-aligned box —
or `bounds { center <x y z> radius <r> }` to confine it to a **sphere** region, i.e.
simple *per-object* fog like **the whole inside of a glass sphere** (author the same
center/radius as the sphere). Or shape the fog to a **named object** with
`bounds { object "<name>" }`: a named `sphere` gives its exact analytic bound, a named
`isosurface` fills the field's interior (the fog takes the metaball/SDF silhouette
exactly, carved per-point during tracking), and a named `mesh` uses the mesh's world
AABB (a box approximation; true mesh containment is deferred). An *open* fog sphere is directly viewable in every mode.
Fog (and any diffuse surface) seen through a **glass sphere** *is* imaged directly by the
pinhole splat `B`, via the **analytic specular connection**: for each glowing haze in-scatter
(or Lambertian surface) vertex the renderer solves the refracted eye ray that reaches the
camera through the sphere in closed form (a planar reduction to a 1-D root solve, with a
ray-differential Jacobian for the splat weight), so a lantern glowing inside a fogged glass
orb — and the fly-through *through* that orb — renders correctly rather than black. The
solve evaluates the ior at the photon's own wavelength, so the refraction is dispersive for
free. It runs on both CPU and GPU. This currently covers **glass spheres in mode `B`** (both
surface and volume vertices); the finite-lens splat `A`, photon-catch `C`, and non-spherical
dielectric shells are not yet covered by the analytic path — for those, seeing the fog
through the curved glass is a refracted (specular↔volume) path that the straight camera
connection can't bend, so that direct view renders black (the fog still correctly **lights
the surrounding room** indirectly). **BDPT `D` images fog-through-glass for any shape**: its
camera subpath refracts through the shell (specular vertices) to a volume in-scatter vertex,
then MIS-connects bidirectionally to the light. Photon-catch `C` traces the same path but
far more slowly (the fog-scattered photon must refract out and hit the pupil).
Add `density "<expr>"` (or `density pattern:<name>`) —
a scalar field over world `x y z` (the same infix expression language as isosurface
`function` fields) that scales `sigma_t` per point — for **fog blobs with soft, formula-defined
boundaries**: e.g. a smooth radial falloff `pow(saturate(1 - dist/R), 2)` renders a
glowing sphere of haze whose edge fades gradually instead of a hard surface. Sampling is
unbiased **delta (Woodcock) tracking** for scattering and **ratio tracking** for shadow
transmittance — exact, no voxelization. A majorant `density_max` is auto-estimated over
`bounds` (or set explicitly). Heterogeneous/bounded fog is honored by the **forward**
modes (A/B/C) on **both the CPU and the GPU** (the device runs the identical density VM +
delta/ratio tracking). **BDPT `D`** renders media of **every kind** — global haze,
multiple superposed media, box/sphere/object-**bounded** fog, and **heterogeneous
`density`-field blobs** — unbiased on both the CPU and the GPU: subpath medium vertices are
placed by delta tracking (analog throughput) and connection edges weighted by ratio-tracking
transmittance, exactly as the forward tracer samples them. (The MIS weights omit the
heterogeneous distance-pdf / transmittance — a variance-only simplification per PBRT-v3;
the balance heuristic is a partition of unity so the estimator stays unbiased regardless.)
The backward reference (R/V) and the P composite treat the medium as a single global homogeneous haze
and warn if you author `density`/`bounds` for them. See `FTSL.md` §12.1.

**Imported volumes (`.nvdb`).** Instead of a formula, point the density field at a real
sparse volume: `density vdb:<path.nvdb>` imports a NanoVDB **FloatGrid** (the compact,
GPU-friendly form of an OpenVDB volume — convert a `.vdb` with OpenVDB's `nanovdb_convert`,
uncompressed). On load the grid is **baked into a dense lattice** plus a world→index affine,
so the *identical* trilinear sampler runs on the CPU and the GPU (`dMedDensityAt` reads the
uploaded lattice) and any affine map (translation/scale/rotation) is honored. The grid's
world AABB auto-seeds the medium bound and its peak value the delta-tracking majorant — so
`medium { sigma_t 40  albedo 0.9  density vdb:cloud.nvdb }` is all it takes to light an
imported cloud. Values are treated as a dimensionless density multiplier on `sigma_t`, so
you still dial the optical thickness with `sigma_t`. Only **float** grids are supported and
the bake is **dense** (memory ~ the grid's index-space bounding box), so very large sparse
volumes are bounded by a safety cap; a native sparse device sampler is a future
optimization. Works in the forward modes (A/B/C) and BDPT `D` on CPU and GPU, exactly like a
`density` formula. Generate a test asset with `scraps/make_nvdb.cpp`.

---

## Scene language (FTSL)

> **Full reference: [`FTSL.md`](FTSL.md)** — the complete grammar (every block, key,
> default, spectrum/pattern/material form, and parsing quirk). The overview below is a
> quick tour; `FTSL.md` is the authoritative spec.

An FTSL file is a list of blocks. Top-level block types: `scene` (the
`units …` / `spectral …` header), `material`, `texture`, `pattern` (procedural scalar
field), `spectrum`, `sphere`, `quad`, `triangle`, `mesh`, `isosurface` (implicit SDF
surface / CSG / metaballs / arbitrary `function` formulas), `light`, `group`, `medium`,
`camera`, `camera_path` (keyframed camera animation), `camera_orbit` (turntable /
fly-around: N frames on a circle around a `center`, for MP4 orbits), `camera_curve`
(spline fly-through with variable speed), and `render` (render-setting overrides). See the `scenes/` directory for worked examples
(`cornell.ftsl`, `fisheye.ftsl`, `spotlight.ftsl`, `envlight.ftsl`,
`material_presets.ftsl`, `realcam.ftsl`, `implicit.ftsl`, `function.ftsl`,
`procedural.ftsl`, `uv_native.ftsl`, `showcase_orbit.ftsl`, `translucency.ftsl`,
`gallery.ftsl` (a large room packed with varied materials around a gold gyroid), …).

### Camera animation (`camera_path`, `camera_orbit`)

Both expand into a sequence of frames sharing look_at/up/fov/mode/film/lens; a
multi-camera render writes one file per frame (`_<name>` inserted before the
extension), which ffmpeg concatenates into a video.

- **`camera_path "name" { … key <t> <ex ey ez> [<lx ly lz>] [<fov>] … frames N }`** —
  keyframed fly-through: the eye (and optionally look_at / fov) is linearly
  interpolated across `key` frames. Optional `dolly_zoom` holds the subject's
  on-screen size (Vertigo effect); optional `exposure_lock` shares frame 0's exposure.
- **`camera_orbit "name" { center <x y z> radius <m> [height <m>] [axis x|y|z] frames N
  [start_deg <d>] [sweep_deg <d>] [look_at <x y z>] [exposure_lock] }`** — a turntable /
  fly-around whose eye rides a circle around `center` (the default look_at). The circle
  lies in the plane perpendicular to `axis` (default y); `height` offsets the eye along
  the axis. A full 360° sweep is sampled so frame N == frame 0 (seamless loop); a
  partial sweep spans its endpoints. See `scenes/showcase_orbit.ftsl` (an orbit tuned
  to fly straight *through* a glass sphere).
- **`camera_curve "name" { point <x y z> … [frames N] [density <ρ> | density_at <t> <ρ> …]
  [look tangent | look_at <x y z> | look curve + look_point <x y z> …] [closed] }`** — a
  fly-through along a **Catmull-Rom spline** that passes through the `point` control
  points. Camera placement is either a fixed `frames` count (uniform arc length) or a
  **density** (cameras per unit length) that can vary along the curve via `density_at`
  keyframes — this is the camera's *speed*: high density = many closely-spaced frames =
  slow dwell, low density = fast. Aim along the travel tangent (default), at a fixed
  `look_at`, or at a second `look curve`. **Orientation and lens can also be animated**
  per frame over the normalized timeline `t ∈ [0,1]` (`t=0` first frame, `t=1` last),
  each keyframed by `<name>_at <t> <value>` (piecewise-linear, flat-clamped at the ends,
  just like `density_at`) or held constant by the bare keyword: **`roll[_at]`** banks the
  camera about its view axis (the third orientation degree of freedom), and
  **`fov_at` / `zoom_at` / `fstop_at` / `focus_at`** animate the vertical field of view,
  focal-length multiplier, f-number, and focus distance. (`fstop`/`focus` change depth of
  field only in the physical catch modes `A`/`C`; in the pinhole splat `B` the aperture is
  virtual, so there `roll`/`fov`/`zoom` are the visible ones. Lens *projection*/fisheye is
  a discrete whole-flight mode, not a continuous track — set it once with `projection`.)

### Multi-camera shared photon pass (modes `A`, `B`, and `M`)

When several cameras render at once (multiple `camera` blocks, or the frames a
`camera_path`/`orbit`/`curve` expands into) in a **forward next-event** mode, the
tracer flies **one** photon set and splats every vertex to **all** cameras of that
mode at once, instead of re-flying the photons per camera. This is the "many cameras
for one photon set" win — emission, BVH traversal, and scattering are paid once. It
runs on **both the CPU and the GPU** (`-device gpu`), and applies to the two forward
splat models:

- **`B` (pinhole splat)** — `connect()` draws no random numbers, so the shared pass is
  **bit-identical** to rendering each camera on its own.
- **`A` (finite-lens camera)** — each camera samples its own aperture pupil (draws
  RNG), so the shared photon flight is **unbiased per camera** but matches a standalone
  render **in distribution**, not bit-for-bit. Rectilinear cameras only.

**Mode `M` (photon map) shares even more cheaply.** Because the photon map is
**view-independent**, a multi-camera mode-`M` render builds the map **once** and runs
each camera's backward density gather against that one shared map — the whole forward
photon flight amortizes across every frame. Unlike `A`/`B` (which reuse a photon
*flight*, so every camera inherits the *same* fixed noise), each mode-`M` camera gathers
with its **own** independent backward samples, so frames share only the underlying
radiance solution, **not** the noise. That makes `M` safe to share across
**exposure-locked** `camera_path` frames too (it isn't restricted to per-frame
auto-exposed cameras the way `A`/`B` sharing is) — the ideal mode for a flythrough of a
static scene.

The `A`, `B`, and `M` cameras form **separate** shared passes (`A` perturbs the RNG
stream during the trace, `B` doesn't, and `M` gathers backward instead of splatting).
`A`/`B` sharing applies to plain `-n` renders with per-frame auto-exposure; `M` sharing
applies to any plain `-n` render (including exposure-locked paths). The budget flags
(`-time`/`-noise`/`-forever`/`-resume`/`-preview`) render per camera in every mode.

**Shared vs. independent randomness across cameras (matters for video and for
side-by-side cameras).** This is the key per-mode difference in how randomness is
distributed *between* cameras. Note that **a "frame" here is simply a camera in the same
scene**: a `camera_path`/`orbit`/`curve` expands into one `camera` per frame, and they all
render together in a single scene exactly like several hand-authored `camera` blocks — so
everything below applies identically whether you wrote the cameras out by hand or generated
them as animation frames:

- **`B`** — every camera is splatted from the *same* photon set, so they share identical
  random paths: a camera's noise is **correlated** with every other camera's (and a camera
  rendered in the group is bit-identical to rendering it alone). Across an animation the
  grain drifts coherently frame-to-frame rather than reshuffling.
- **`A`** — cameras share the photon *flight* but each draws its own aperture-pupil
  samples, so each carries **independent** randomness on top of the shared paths (unbiased
  per camera; correlated only through the shared flight).
- **`M`** — cameras share the photon *map* (the radiance solution) but each runs its own
  backward density gather, so each frame's noise is **independent** — the best of both:
  the expensive forward pass is paid once, yet frames don't inherit a shared grain.
- **`C`/`R`/`D`/`P`/`V`** — each camera is traced **fully independently** with its own
  sample budget, so their randomness (and noise) is **uncorrelated** by construction.

Mode `B`'s correlation is usually invisible (and cheaper), but if you want independent,
film-grain-like noise per camera/frame, render them separately (e.g. via a budget flag,
which falls back to per-camera passes) so each draws its own photons.

> **Other modes do NOT save time with multiple cameras.** `C` (finite-aperture catch)
> consumes each photon at the first aperture it hits, so it can't share a photon set; and
> `R`, `D`, `P`, and `V` are camera-anchored estimators that trace **from** each camera —
> a multi-camera render of those modes simply renders **each camera independently**
> (re-tracing the full sample budget per camera), so it costs the same as running them
> one at a time. Only `A`, `B`, and `M` amortise the forward trace across cameras.

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
| `-topng <in> <out.png>` | Convert an existing `.ppm` or `.ftbuf` to a 24-bit PNG (no rendering); see **Output** |
| `-mode <A..D,M,S,U,P,R,V>` | Render mode (default `B`) |
| `-pmradius <r>` / `-pmradiusfrac <f>` | Mode `M`/`S`/`U` photon-map/merge gather radius (initial radius for `S`/`U`): absolute world units, or a fraction of the scene radius (default `0.02`). Smaller = sharper contact shadows but noisier |
| `-pmfg <K>` | Mode `M` final gather: `K` cosine-weighted hemisphere sub-rays per sample, querying the map one bounce away for sharp contact shadows / fine detail (default `0` = off, direct density query). ~`K`× per-sample cost — pair with fewer `-spp` |
| `-sppmalpha <a>` | Mode `S` radius-shrink rate (default `0.7`; smaller shrinks faster) |
| `-vcmalpha <a>` | Mode `U` (VCM) radius-shrink rate (default `0.75`; smaller shrinks faster) |
| `-camera <sel>` | Pick which camera(s) to render (and thus what `-window`/`-preview` shows). `<sel>` is `all`, an exact name (`hero`, `fly137`), an index `#N` into the declared cameras (0-based, `#-1` = last), or `near=X,Y,Z` (the camera whose eye is closest to that point). The index / nearest forms make it easy to aim the live view at one frame of a long `camera_curve` without hunting for its frame name. |
| `-view EX,EY,EZ/LX,LY,LZ[/FOV]` | Render a brand-new ad-hoc camera (eye → look, optional vertical FOV; `,` and `/` are interchangeable separators) instead of the scene's cameras — a quick way to preview a scene from an arbitrary angle. Works with `-in` scenes and built-in `-scene`s. |
| `-t <threads>` | CPU thread count |
| `-device auto\|cpu\|gpu` | Hardware backend |
| `-wavefront` | Streaming GPU backend instead of megakernel |

**Camera / physics overrides**

| Flag | Meaning |
|---|---|
| `-light <preset>` | Override light SPD by preset |
| `-aperture <r>` / `-focus <d>` | Thin-lens aperture radius / focus distance |
| `-mesh <path>` / `-meshscale <s>` | Load & scale an OBJ into the built-in scene |
| `-export-mesh <out.obj>` | Polygonise the scene's isosurfaces into a watertight OBJ mesh (marching tetrahedra, box-capped) and exit, instead of rendering — for Unreal / Blender import (see **Exporting an isosurface to a mesh**) |
| `-mesh-res <N>` | Mesh export fineness: grid cells along the longest bounds axis (default 128) |
| `-mesh-adaptive` / `-mesh-decimate <f>` | Curvature-adaptive QEM decimation of the exported mesh; `<f>` = triangle fraction to keep (default 0.5) |
| `-fog <σt>` / `-fogalbedo <a>` / `-fogg <g>` / `-fograyleigh` | Fog controls |
| `-filmthickness <nm>` / `-filmior <n>` | Thin-film iridescence demo params |
| `-diffraction <mode>` / `-nodiffraction` | Enable/disable grating & thin-film diffraction |
| `-spp <n>` | Samples per pixel for modes `R`, `D`, `M`, and `V`; **number of passes** for SPPM (`S`) and VCM (`U`) |
| `-n <photons>` (mode `S`) | Photons traced **per pass** (SPPM rebuilds a bounded map each pass). *(Mode `U` ignores `-n` — its light-path count follows the film resolution.)* |

**Long-running / output** — `-time` / `-noise` / `-forever` / `-preview` / `-window` /
`-interval` apply to every image-forming mode (forward `A`/`B`/`C`, the spp modes `R`/`D`,
the composite `P`, and the photon modes `M`/`S`/`U`), on both CPU and GPU. `-resume` /
`-checkpoint` cover `A`/`B`/`C` (photon-count checkpoint), `R`/`D` (spp-count checkpoint),
and `P` (dual forward+backward film) — `M`/`S`/`U` keep persistent per-pass state a film
alone can't restore, so they are not disk-resumable.

| Flag | Meaning |
|---|---|
| `-time <s>` | Render until a wall-clock budget |
| `-noise <pct>` | Render until the noise floor drops below `pct` % |
| `-forever` | Refine indefinitely (Ctrl-C stops gracefully) |
| `-preview` | Live ANSI thumbnail while rendering |
| `-window` | Open a real OS window (Win32 GDI; no-op off Windows) showing the actual tone-mapped pixels, refreshed each `-interval` tick. Full-resolution, unlike `-preview`'s terminal thumbnail; runs on its own UI thread. A plain fixed-`-n` forward render is auto-chunked so the view converges live, and closing the window stops the render (final image is still written). |
| `-interval <s>` | Periodic image write / preview / window refresh (default 15 s) |
| `-resume` / `-checkpoint` | Resume from / always write a `<out>.ftbuf` checkpoint (modes `A`/`B`/`C`, `R`/`D`, and `P`) |
| `-exposure-lock` | Share one auto-exposure anchor across all rendered cameras (no `camera_path` flicker); a per-path `exposure_lock` keyword locks just that path |

**Diagnostics / self-tests:** `-checkbvh`, `-bvhstats`, `-checklens`,
`-checkfluoro`, `-checkfog`, `-checkthinfilm`, `-checkmultilayer`,
`-thinfilmswatch`, `-checkgrating`, `-checkupsample`.

---

## Output

Images are written as **PNG** (24-bit RGB, 8 bits/channel — no alpha), **JPEG**
(q95), or binary **PPM (P6)**, chosen by the output file extension, tone-mapped from
the internal linear spectral film to 8-bit sRGB. Long renders can checkpoint to
`<out>.ftbuf` and resume deterministically.

**Converting existing artifacts to PNG** — `ftrace -topng <in> <out.png>` re-encodes
an artifact to a 24-bit PNG *without re-rendering*:

- a **`.ppm`** (binary P6, 8-bit) is copied to PNG losslessly;
- a **`.ftbuf`** resume-checkpoint has its raw linear film tone-mapped (with the
  default p99 auto-exposure — the sidecar doesn't store the exposure mode, so an
  absolute/lumens scene may read brighter or darker than its original `-o` image;
  re-render for an exposure-exact PNG).

A `.ftsl` is a *scene*, not an image — render it with `-in scene.ftsl -o out.png`.
Three drag-and-drop Windows helpers in the repo root wrap this: **`ppm_to_png.bat`**,
**`ftbuf_to_png.bat`** (both call `-topng`), and **`ftsl_to_png.bat`** (renders the
scene). Drop a file on one, or run `ppm_to_png.bat input.ppm [output.png]`.

---

## Known issues & roadmap

Open limitations and technical debt are tracked in `known-issues.md` — including
the physical-lens camera's remaining gaps (inter-element flare/ghosting,
shaped-iris bokeh).
