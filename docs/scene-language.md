# Scene Description Language (FTSL) — Design Spec

> **Status: Phase 1 IMPLEMENTED; Phases 2–3 still design proposal.** The loader
> lives in `src/ftsl.h` and is wired to the `-in <file.ftsl>` CLI flag. It parses
> the block format below, evaluates spectrum expressions (constant, `blackbody`,
> `gaussian`, `shortpass`, `ior`, `rgb`, `whitewall`/`redwall`/`greenwall`,
> `glass:`, `preset:`, `spectrum:` refs, and `table { }`), builds materials
> (all eight `MatType`s), geometry (`sphere`/`quad`/`triangle`/`mesh` with full
> translate+rotate+non-uniform-scale transforms), any number of `light` blocks
> (area or collimated), a `medium`, a `camera`, and a `render` block (overridable by CLI).
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
> The still-unimplemented
> pieces (configurable spectral *range*, absolute light power/units — which also
> gates absolute-EV film sensitivity — the full physical `layered` material, the
> shared multi-camera mode-B pass, non-square films, textures/UVs, extra light
> shapes) remain tagged
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
| `rgb 0.63 0.06 0.05`                   | Convenience: upsample an sRGB triple to a smooth reflectance   | `rgbToReflectanceJH` (Jakob-Hanika sigmoid fit, `src/upsample.h`) — **[maps 1:1]**; validated by `-checkupsample` |
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
| `mix`         | **stochastic blend of materials** | `layer "<name>" <weight>` (repeatable) → `mixChildren`/`mixWeights`. Per photon, pick child `k` with prob `weight_k`; leftover `1 − Σweight` absorbs. Children are named non-mix materials. See §3.2. |

### 3.2 Combining effects on one surface — the `layered` material

**Can a single material be semi-mirror + glossy + transparent + translucent +
iridescent + fluorescent at once? Yes — but only under the right model, and it
is [needs engine work] (today a surface is exactly one `MatType`).** The
important correction is that these are **not** independent, additively-stacked
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
# PROPOSED (not yet supported): one physically-layered material
material "lacquered_shell" {
    type layered
    coat {                       # the specular interface
        roughness     0.15       # 0 = mirror, >0 = glossy
        reflectance   thinfilm   # fresnel | thinfilm(=iridescent)
        film_ior      1.4        # (thinfilm only)
        film_thickness 380       # nm  (thinfilm only)
        # specular   0.5         # optional manual partial reflectance ⇒ semi-mirror
    }
    body {                       # what the transmitted light does; weights sum ≤ 1
        diffuse      { reflect rgb 0.2 0.5 0.9   weight 0.5 }
        transmit     { ior glass:BK7  absorb spectrum:amber_tint   weight 0.3 }
        subsurface   { reflect 0.8   weight 0.1 }         # translucence
        fluorescent  { absorb spectrum:excite  emit spectrum:emit_green
                       yield 0.9   weight 0.1 }
    }
}
```

A `mix` of whole named materials (probabilistic pick among sub-materials) is a
simpler, less-physical alternative that the same machinery supports; `layered`
is preferred because the coat/body split is energy-consistent and matches how
real surfaces work. (Note: the existing backward reference tracer can't validate
a body with `fluorescent` — see known-issues — so `layered`/`mix` materials that
include a fluorescent child stay forward-only, same restriction as the standalone
`fluorescent` type; such scenes also fall back to the CPU forward tracer.)

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

The full physical `layered` material (Fresnel/Airy-weighted coat over exotic body
lobes) is still **[needs engine work]**.

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

The scene supports **any number of emitters** (Phase 2b). Each `light` block adds
one `Emitter` (`src/scene.h`): a rectangular area light with a spectral power
distribution, or a collimated beam (prism/grating demos). `Scene::emitters` holds
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

### 5.2 Open design points for lights

- **Multiple / typed lights.** **[done — Phase 2b]** Any number of `light` blocks
  accumulate; the forward tracer uses a power-weighted selection CDF in the photon
  spawn path (CPU and CUDA), and the backward reference sums NEE over all emitters.
  Typed shapes (sphere/spot/env) are still future — see below.
- **Absolute power / units.** **[needs engine work]** Today emission is normalized by the SPD integral
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

    aperture 0.02              # aperture radius (scene units); 0 ⇒ pinhole-ish
    focus    3.0               # focus distance ⇒ thin-lens focal length
    fstop    2.8               # f-number ⇒ apertureR = focal/(2N) (overrides aperture)
    mode     B                 # A | B | C  (measurement model, §8.2)

    film {
        res  512 512           # output pixels
        size 36 24             # physical sensor size (mm) — §8.1
        iso  100               # exposure compensation (relative) — §8.1
    }
}
```

Maps onto `Camera` (`src/camera.h`) via `lookAt(eye, target, up, fovYDeg, rx, ry)`
and `setFocus(focus)`; `aperture` → `apertureR`; `mode` picks the forward
measurement model. **[maps 1:1]**, plus the physical-film fields below.

### 8.1 Film — present vs. proposed

- `res W H` — output resolution. **[maps 1:1]** (`Film::resX/resY`). *Note: the
  forward/backward tracers currently allocate a **square** film, so only the first
  value is used; non-square sensors are a follow-up.*
- `size <w> <h>` (mm) — physical sensor dimensions. **[done — Phase 3a]**: the
  focal length is derived from the film **height** and `fov_y`
  (`f = filmH / (2·tan(fov_y/2))`, in metres) and used for f-stop → aperture. A
  35 mm "full frame" is `size 36 24`. *(Because the film is square today the width
  is not yet used for a true horizontal fov; when unspecified a 24 mm full-frame
  height is assumed wherever a physical length is needed.)*
- **f-stop authoring** — **[done — Phase 3a]**: `fstop 2.8` ⇒
  `apertureR = focal / (2·N)` at load time (overrides any `aperture` radius), so
  depth of field in the finite-aperture catch modes (A/C) is physically meaningful
  instead of unit-relative.
- `iso` / `shutter` / `exposure` — **[done (relative) — Phase 3a]**: the film's
  radiometric scale is not absolute, so images are always auto-exposed (99th-
  percentile anchor). These act as an exposure **compensation** on top of that
  anchor: `comp = exposure · (iso/100) · shutter` (each factor defaults to 1), e.g.
  `iso 200` is exactly one stop brighter than `iso 100`. Aperture is deliberately
  *not* folded in (in A/C a smaller aperture already darkens the image physically;
  in B the aperture is virtual). **True absolute EV / a physical sensitivity+
  response model still needs engine work** — it depends on absolute light power
  (watts/lumens), which is a separate deferred feature (see §7 / known-issues). A
  fixed exposure *lock* across `camera_path` frames (compute the anchor once, reuse
  it) is a natural follow-up on top of this.

### 8.2 Measurement model (`mode`)

| `mode` | Model                              | Notes                                                            |
|--------|------------------------------------|------------------------------------------------------------------|
| `A`    | Contact-sensor forward catch       | Physically literal; needs a `sensor`/film surface in the scene.  |
| `B`    | Pinhole connect/splat (default)    | Unbiased estimator of an ideal pinhole; fast.                    |
| `C`    | Finite-aperture thin-lens catch    | Real depth of field; uses `aperture`/`focus`.                    |

(Modes R/V/P are reference/validation/composite tooling, not scene-facing;
they stay CLI-only.)

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
`fov_y`, `mode`, `aperture`, `focus`, and `film { res }` apply to every frame.
`-camera dolly2` selects a single frame. The grammar is deliberately *numbers-only*
(`key <t> <ex> <ey> <ez> [<lx> <ly> <lz>]`) because the FTSL statement splitter
breaks a statement on the next bareword, so inline keywords like `eye`/`t=` inside
a one-line `key` are not available.

For a shared photon pass (a future optimization), the engine would connect each
diffuse bounce to *every* frame/camera's pupil (mode B) in one trace — a natural
extension of the existing `connect()` since photons are camera-independent until
the splat.

---

## 9. Skins / textures — import, mapping, and spectral color

The wishlist: *"provide ways of mapping skins to meshes (to get as evenly
distributed / without warp as possible, such as UV mapping)? can we also get
skins with spectral envelopes somehow defined for their various colors?"*

**Current state: none of this exists.** Meshes carry one material, there are no
UVs, no image loader, and no texture concept (`src/mesh.h` reads only `v`/`f`,
ignores `vt`/`vn`; the project only reads/writes PPM). Everything in this
section is **[needs engine work]**. It breaks into three independent pieces:
**(9.1) importing the image, (9.2) mapping it onto geometry, (9.3) turning its
colors into spectra.**

### 9.1 Importing a skin (the image)

Add a `texture` block and an image loader. Recommend **stb_image** (single
public-domain header; PNG/JPG/TGA/BMP + `.hdr`) — no heavy dependency, matches
the project's "no external deps" style.

```
texture "face_albedo" {
    file     "face_albedo.png"
    encoding srgb            # srgb | linear  — how to decode the file
    filter   bilinear        # nearest | bilinear  (texel interpolation)
    wrap     repeat          # repeat | clamp | mirror
}
```

**Color management matters for physical correctness:** art PNG/JPGs are
sRGB-**display-encoded** (gamma). `encoding srgb` linearizes each texel before
use; data maps (roughness, masks, thickness) are `encoding linear` and skip it.
HDR/`.hdr`/`.pfm` are already linear.

### 9.2 Mapping the skin onto the mesh (the "even / without warp" question)

A texture is sampled at a `(u,v)` produced from the surface hit. How that
`(u,v)` is chosen is the mapping method, set per mesh:

```
mesh "head" { file "head.obj"  material face
    uv use_mesh              # use the OBJ's own vt coordinates (preferred)
    # uv triplanar scale 1.0 # box projection from 3 axes, blended by normal
    # uv planar axis y       # single-axis projection
    # uv spherical           # lat/long — globes, eyeballs
    # uv cylindrical         # bottles, limbs
}
```

- **`use_mesh` (UV mapping) — the low-warp answer.** Read OBJ `vt`, store
  per-vertex UVs on `Tri`, barycentric-interpolate at the hit. **Crucial point:
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

Needed engine plumbing: per-vertex UVs on `Tri` + a `Texture` type + a tangent
frame from UV derivatives (only once normal/bump maps arrive).

### 9.3 Spectral envelopes for the skin's colors (the key question)

*Yes — a skin can carry proper spectra for its colors, two ways:*

- **RGB → reflectance upsampling (general, for any color image).** Run each
  linearized texel through the **reflectance upsampler** — implemented as the
  Jakob-Hanika 2019 sigmoid fit in `src/upsample.h` (`rgbToReflectanceJH`) — to
  produce a smooth, physically-plausible reflectance *spectrum* for that color.
  This is the same machine as the inline `rgb …` spectrum in §2.1 (already
  wired), to be applied per texel once textures land. It lets ordinary painted skins
  participate correctly in the spectral pipeline (proper metamerism, correct
  colour under non-D65 lights) without hand-authoring curves.

  ```
  texture "face_albedo" { file "face_albedo.png"  encoding srgb  upsample reflectance }
  material "face" { type diffuse  reflect texture:face_albedo }
  ```

- **Indexed-spectral (precise, for known pigments / scientific skins).** The
  image stores *indices*, and a palette maps each index to a named spectrum
  (measured pigment, dye, metal). Exact where you know the actual materials —
  e.g. a flag or a chart of paint chips.

  ```
  texture "flag" { file "flag_index.png"  encoding linear
      palette { 0 spectrum:navy   1 spectrum:crimson   2 spectrum:offwhite } }
  ```

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
9. `mix`/layered materials (generalize `halfmirror`). **[`mix` done — Phase 2d; per-photon lobe selection in forward + backward + CUDA; `scenes/mixmat.ftsl`. Full physical `layered` still needs engine work.]**

**Phase 3 — larger features**
10. Multiple cameras / `camera_path`; per-camera films; physical film size +
    f-stop + sensitivity. **[mostly done — Phase 3a: multiple named cameras +
    `-camera` selection + per-camera film resolution + per-camera mode
    (`scenes/twocam.ftsl`); `camera_path` keyframe interpolation
    (`scenes/dolly.ftsl`); physical film `size` (mm) → focal length, `fstop` →
    aperture radius, and relative exposure compensation via `iso`/`shutter`/
    `exposure` (`scenes/expo.ftsl`). Remaining: shared mode-B multi-camera pass,
    non-square films, and absolute-EV/sensitivity (needs absolute light power).]**
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
