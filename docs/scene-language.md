# Scene Description Language (FTSL) — Design Spec

> **Status: DESIGN PROPOSAL, not yet implemented.** As of this writing the
> renderer has **no** scene file format. Scenes are hard-coded C++ builders
> (`buildCornell`, `buildPrism`, `buildGrating`, `buildMaterials` in
> `src/main.cpp`) selected with `-scene <name>`, and tuned only through CLI
> flags; `-mesh file.obj` just swaps one OBJ mesh in for the Cornell sphere.
> That is fine for developing the physics but useless for describing arbitrary
> scenes. This document designs the language we need. Every construct below is
> annotated **[maps 1:1]** (a loader would just fill an existing struct) or
> **[needs engine work]** (the renderer doesn't support it yet), so the spec
> doubles as an implementation checklist.

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
| `light`      | 1+         | Emitter (area or collimated)                        |
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
| `glass:BK7`, `glass:SF10`              | Named Sellmeier dispersion curve (refractive index)           | `iorBK7`, `iorSF10` — **[maps 1:1]** |
| `ior 1.5`                              | Constant refractive index                                     | `iorConstant` — **[maps 1:1]** |
| `table { 400:0.05 450:0.12 ... }`      | Piecewise-linear measured curve (λnm:value pairs)             | **[needs engine work]** — add a `tabulatedSpectrum(pairs)` builder (trivial: linear interp) |
| `rgb 0.63 0.06 0.05`                   | Convenience: upsample an sRGB triple to a smooth reflectance   | **[needs engine work]** — add a reflectance upsampler (e.g. Scott Burns / Jakob-Hanika); see §9 |
| `spectrum:name`                        | Reference a named `spectrum` block                            | name resolution                |
| `preset:D65`, `preset:led`, ...        | Named illuminant SPD (see §5)                                 | `src/lights.h` — **[maps 1:1]** |

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
> envelopes for any common materials?"): the `table { }` form is exactly the
> ingestion point for published datasets (e.g. the RPMK / Vos-measured pigment
> and metal reflectances, or the refractiveindex.info database). A small
> converter tool could turn a CSV of λ,value into a `spectrum "..." = table {…}`
> block. This is the recommended path for real material fidelity.

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
| `thinfilm`    | **iridescence** (Airy interference) | `ior <spectrum>` (substrate) → `ior`; `film_ior <n>` → `filmIor`; `film_thickness <nm>` → `filmThickness`.       |
| `grating`     | **diffraction** (vector grating eq.) | `reflect <spectrum>` → `reflect`; `groove_spacing <nm>` → `grooveSpacing`; `groove_dir x y z` → `grooveDir`; `max_order <int>` → `gratingMaxOrder`. |
| `fluorescent` | **fluorescence** (wavelength shift) | `absorb <spectrum>` → `fluoAbsorb` (excitation ε(λ)); `emit <spectrum>` → `fluoEmit` (re-emission M(λ′), auto-baked into `fluoEmitSampler`); `yield <0..1>` → `fluoYield` (quantum yield Q); `reflect <spectrum>` → `reflect` (elastic base). |

**Combining effects.** The user's wishlist mentions "combine semi-mirror with
semi-transparency", mirror + glossy, etc. Today each surface is exactly one
`MatType`, so a *layered* material (e.g. glossy clear-coat over diffuse, or
partial mirror + partial transmit + partial diffuse in arbitrary ratios) is
**[needs engine work]**: it requires a `layered`/`mix` material that stores
child materials + per-photon selection probabilities. The format should reserve
the syntax now so the engine can grow into it:

```
# PROPOSED (not yet supported): probabilistic mix of sub-materials
material "coated" { type mix
    layer material:brushed  weight 0.1     # 10% of photons take the glossy coat
    layer material:white    weight 0.9     # 90% hit the diffuse base
}
```

`halfmirror` is the one built-in "mix" today (reflect-or-transmit by
probability); `mix` generalizes it.

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
- `mesh` — **[maps 1:1 for the basics]** via `loadObj` (`src/mesh.h:27`), which
  reads `v`/`f` and applies `translate` + uniform `scale`. **[needs engine
  work]** for: non-uniform scale + rotation (loader only does translate+uniform
  scale today — add a full 4×4 transform), per-face / per-group materials (OBJ
  `usemtl` is ignored — currently one `matId` for the whole mesh), and vertex
  normals/UVs (ignored). See §6 for the UV/texture/skin discussion.

**Reserved transform syntax** (for when the loader gains a full transform):

```
mesh { file "teapot.obj" material brushed
       translate 0 0.5 -1   rotate_y 30   scale 0.5 0.5 0.5 }
```

---

## 5. Lights

The scene has **one emitter today** — a rectangular area light with a spectral
power distribution, or a collimated beam (prism/grating demos). The
`Scene` fields are `lightOrigin/U/V/normal`, `lightArea`, `lightSpd`
(an `EmissionSampler` CDF), `lightEmitIntegral`, and `collimated`/`beamDir`.

```
light area {
    origin 0.3 1.99 0.3   u 0.4 0 0   v 0 0 0.4   normal 0 -1 0
    spd blackbody 6500              # any <spectrum>, or preset:<name>
    # power/scale TBD — see note below
}

light collimated {
    dir 1 0 0
    spd preset:sun
}
```

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
| `fluorescent` / `cfl`              | Mercury lines (436/546/611 nm) + continuum            |

Or supply any `<spectrum>` directly (`spd blackbody 3000`, `spd spectrum:myLED`,
`spd table { … }`).

### 5.2 Open design points for lights **[needs engine work]**

- **Multiple / typed lights.** The engine currently supports exactly one area
  light (photon emission samples one quad). Supporting `light` blocks *plural*
  requires: a list of emitters + a light-selection CDF (pick which light to emit
  from, weighted by power) in the photon spawn path. The format already allows
  multiple `light` blocks; the engine must catch up.
- **Absolute power / units.** Today emission is normalized by the SPD integral
  and the light area — good enough for relative imagery, but there is no
  radiometric "this bulb is 800 lumens / 10 W". A `power <watts>` (radiant) or
  `luminous <lm>` key is the place to add physically-absolute output. Until the
  engine tracks absolute units this is documentation-only.
- **Other shapes.** Sphere/point/spot/environment(HDRI) emitters are all
  future; the `light <type>` tag leaves room (`light sphere { … }`,
  `light env { file "sky.hdr" }`).

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
So these effects depend only on absolute wavelength vs. absolute feature size —
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

`units meters | centimeters | millimeters` would scale coordinates, fog
coefficients, and camera distances into a single internal unit (metres) at load
time — so a scene authored in cm and one in m render identically. Wavelength /
groove / film-thickness stay in nm regardless. This makes scale *explicit*
without adding any per-object "scale" fudge factor. **[needs engine work]** —
just a load-time multiply; the physics doesn't change.

`spectral 360 830 1` mirrors the hard-coded `[360,830]` nm range and the 1 nm
`EmissionSampler` step; exposing it lets a UV-fluorescence or IR study widen the
band. **[needs engine work]** to thread the bounds through `color.h`/`spectrum.h`
(currently compile-time constants).

---

## 8. Cameras, film, and the measurement model

```
camera "hero" {
    eye     0 1 3
    look_at 0 1 0
    up      0 1 0
    fov_y   40                 # vertical field of view, degrees

    aperture 0.02              # aperture radius (scene units); 0 ⇒ pinhole-ish
    focus    3.0               # focus distance ⇒ thin-lens focal length
    mode     B                 # A | B | C  (measurement model, §8.2)

    film {
        res  512 512           # output pixels
        # size 36 24           # PROPOSED physical sensor size (mm) — see §8.1
        # iso  100             # PROPOSED film sensitivity — see §8.1
    }
}
```

Maps onto `Camera` (`src/camera.h`) via `lookAt(eye, target, up, fovYDeg, rx, ry)`
and `setFocus(focus)`; `aperture` → `apertureR`; `mode` picks the forward
measurement model. **[maps 1:1]**, except the commented film fields.

### 8.1 Film — present vs. proposed

- `res W H` — output resolution. **[maps 1:1]** (`Film::resX/resY`).
- `size <w> <h>` (mm) — physical sensor dimensions. **[needs engine work]**: the
  camera today derives the image plane purely from `fov_y` + aspect; a physical
  film size (35 mm "full frame", medium/large format) would let f-stop and
  circle-of-confusion be *physically* meaningful instead of unit-relative. This
  is the todo item about "film dimensions (33mm, medium format, …)".
- `iso` / sensitivity, spectral response curve — **[needs engine work]**: the
  film accumulates linear XYZ; a sensitivity/response model (and per-camera
  exposure) would live here. Todo item "film sensitivity".
- **f-stop authoring.** Photographers set an f-number, not an aperture radius.
  `fstop 2.8` ⇒ `apertureR = focalLength / (2·N)`. **[needs engine work]** (a
  load-time conversion once physical focal length / film size exist).

### 8.2 Measurement model (`mode`)

| `mode` | Model                              | Notes                                                            |
|--------|------------------------------------|------------------------------------------------------------------|
| `A`    | Contact-sensor forward catch       | Physically literal; needs a `sensor`/film surface in the scene.  |
| `B`    | Pinhole connect/splat (default)    | Unbiased estimator of an ideal pinhole; fast.                    |
| `C`    | Finite-aperture thin-lens catch    | Real depth of field; uses `aperture`/`focus`.                    |

(Modes R/V/P are reference/validation/composite tooling, not scene-facing;
they stay CLI-only.)

### 8.3 Multiple cameras & camera paths **[needs engine work]**

The wishlist wants "many cameras at once (possibly along a path)… same render
for efficiency". The format supports it by allowing multiple named `camera`
blocks (render all, or `-camera hero`); and a `camera_path` for motion:

```
# PROPOSED (not yet supported):
camera_path "dolly" {
    look_at 0 1 0   up 0 1 0   fov_y 40   mode B
    key t=0.0  eye 0 1 4
    key t=0.5  eye 2 1 3
    key t=1.0  eye 3 1 0
    frames 60
}
```

The engine would need per-camera films and, for a shared photon pass, connect
each diffuse bounce to *every* camera's pupil (mode B) — a natural extension of
the existing `connect()` since photons are camera-independent until the splat.

---

## 9. Skins / UV mapping / textures — where this lands

The wishlist asks about mapping "skins" to meshes (UV) and getting per-color
spectral envelopes for them. Current state: **none of this exists** — meshes
carry one material, no UVs, no textures (`src/mesh.h` ignores `vt`/`vn`).

Design direction when it's built **[needs engine work]**:

- **UVs:** read OBJ `vt` and store per-vertex UVs on `Tri`; barycentric-interp
  at the hit point.
- **Spectral textures:** a texture is a function `(u,v) → spectrum`. The honest
  version stores a *spectral* image or, pragmatically, an sRGB image plus the
  reflectance-upsampler from §2.1 (`rgb → smooth reflectance`) so ordinary
  albedo maps become physically plausible spectra. Proposed syntax:

  ```
  # PROPOSED:
  spectrum_texture "skin_albedo" { file "face_albedo.png"  upsample reflectance }
  material "face" { type diffuse  reflect texture:skin_albedo }
  ```

- **Even distribution / low warp:** that's an authoring-tool concern (the UV
  layout in the mesh), not a renderer concern — the renderer just samples
  whatever UVs the mesh provides.

The reflectance-upsampler (Jakob-Hanika 2019 or Scott Burns' method) is the
single most useful piece to build here: it unlocks both `rgb …` spectra (§2.1)
and future RGB textures, letting non-spectral art assets participate in the
spectral pipeline without hand-authoring curves.

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
7. Multiple lights (emitter list + power-weighted selection CDF).
8. RGB→reflectance upsampler (unlocks `rgb` spectra and later textures).
9. `mix`/layered materials (generalize `halfmirror`).

**Phase 3 — larger features**
10. Multiple cameras / `camera_path`; per-camera films; physical film size +
    f-stop + sensitivity.
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
