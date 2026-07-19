# FTSL — the Forward-Tracer Scene Language

FTSL is the block-structured text format the raytracer loads with `-in scene.ftsl`.
It populates the in-memory `Scene` + `Camera(s)` instead of a hand-written C++ builder.
This is the complete reference; the loader lives in `src/ftsl.h`.

> **Scope note.** This documents what the parser in `src/ftsl.h` actually accepts.
> Where a keyword is easy to get subtly wrong (value continuations, `key=val` vs
> bareword, `mod` not being a function, …) it is called out explicitly.

---

## 1. Lexical structure

- **Comments**: `#` to end of line (anywhere).
- **Whitespace**: spaces / tabs / newlines separate tokens. Newlines terminate
  statements, but several `key value` pairs may share one line.
- **Strings**: `"double quoted"` — used for block names and quoted expressions.
- **Braces**: `{ … }` open a block body or a nested sub-block.
- **Barewords**: any run of non-space, non-brace, non-quote, non-`#` characters. A
  bareword is a *number* iff `strtod` consumes all of it (`-1`, `0.999`, `1e30`).

### 1.1 Statements and value continuations

A statement is `key value…`. The parser reads the **first** value token
unconditionally, then keeps consuming *continuation* tokens — a token that is a
**number** or contains an `=` (a `key=val` named param) — and stops at the next
bareword, which begins the next statement's key.

Consequences you must know:

- Vectors work because they are numbers: `origin 0 0 0  u 1 0 0` parses as two
  statements on one line.
- A trailing option that is a **bareword** starts a *new* statement and will NOT be
  folded back into the previous one. This is why axis/scale options use `key=val`
  form: `uv planar axis=y` (correct) vs `uv planar y` (the `y` is silently a stray
  statement). See §9.
- A `{` after a value opens a nested block whose *type* is the preceding word (or the
  statement key if none): `film { … }`, `table { … }`, `coat { … }`, `lens { … }`.

### 1.2 Top-level block syntax

```
blocktype ["name"] [subtype] { … }
```

- `spectrum` is special: `spectrum "name" = <spectrum-expr>` (needs the `=`, no braces).
- `light` takes a **subtype** bareword: `light area { … }`, `light spot { … }`, etc.
- All other blocks are `type ["name"] { … }`.

### 1.3 Conditional blocks (`prefer { … } else { … }`)

A top-level `prefer` node holds ordered **branches**, each a complete list of top-level
blocks. The loader trial-builds them in order and keeps the first branch that is
**renderable under the active render mode**, splicing its blocks into the scene as if
they'd been written inline; if none qualify, the last branch is used as the fallback.

```
prefer {
    camera "cam" { … mode D … }
    medium  "gas" { … }            # plain (non-GRIN)
} else {
    camera "cam" { … mode B … }
    medium  "gas" { … ior … }      # GRIN version
}
```

Rules:

- `else` chains **flat**: `prefer { A } else { B } else { C }`. A branch may **not**
  contain a nested `prefer` (the parser errors).
- Each branch must have at least one block.
- Only **cameras** and **media** carry mode-support constraints today — GRIN media
  (`medium … { ior … }`) and non-rectilinear cameras (`projection fisheye/…`) are the
  features rejected by the bidirectional modes (`D` BDPT, `U` VCM). The support test
  wraps `bdptUnsupportedFeature` / the fisheye/media checks; everything else always
  builds.
- Resolution logs `[prefer] branch N rejected (<reason>); trying the next` and
  `[prefer] using branch N of M`.

The CLI flag `-on-unsupported error|fallback|strip` is a separate, lower-priority
safety net applied *after* prefer resolution, for when the finally-selected mode still
can't render a feature (see the command-line reference).

---

## 2. Units and the `scene` block

```
scene { units meters  spectral 360 830 1 }
```

| key | values | default | meaning |
|---|---|---|---|
| `units` | `meters`/`m`, `centimeters`/`cm`, `millimeters`/`mm`, `inches`/`in`, `feet`/`ft` | `meters` | all authored **lengths/positions** are scaled to internal metres at load time |
| `spectral` | `<lo> <hi> <binWidth>` | `360 830 1` | only the **bin width** is applied; the engine range is fixed at 360–830 nm (a warning prints if `lo/hi` differ) |
| `default_mode` | a mode letter (`A`/`B`/`C`/`D`/`U`/`M`/`R`/`P`/…) | *(none)* | the render mode used when nothing else picks one. Resolution order: `-mode` (CLI) → a camera's own `mode` → `default_mode` → built-in `B` |
| `fps` | `<n>` | *(none)* | default playback rate for flyby animations, read by assembly tooling (e.g. `showcase_flyby.py` when `--fps` is omitted). Overridable per-flyby with `fps <n>` on the `camera_curve`/`camera_path`/`camera_orbit` block. Playback hint only — does not affect rendering |

Directions (`up`, `normal`, `dir`, `axis`) are **not** unit-scaled — only points and
lengths are. There is one `scene` block (extra ones are also scanned but the last
`units`/`spectral` win).

---

## 3. Spectrum expressions

A spectrum expression appears after `spd`, `reflect`, `ior`, `absorb`, `emit`,
`substrate_k`, `sigma_a`, `sigma_s`, and in `spectrum "name" = …`.

| form | meaning |
|---|---|
| `<number>` | constant (flat) spectrum |
| `blackbody <K>` | Planckian radiator at K kelvin (default 6500) |
| `ior <n>` | constant index of refraction (default 1.5) |
| `whitewall [r]` | neutral diffuse reflectance (default 0.75) |
| `redwall` / `greenwall` | the Cornell-box side-wall reflectances |
| `gaussian center=<nm> sigma=<nm> amp=<a>` | a Gaussian emission/reflectance band |
| `shortpass edge=<nm> slope=<nm> amp=<a>` | a soft short-pass edge |
| `rgb <r> <g> <b>` | sRGB-linear triple → reflectance via Jakob-Hanika upsampling |
| `glass:<name>` | dispersive glass IOR curve (see §3.1) |
| `metal:<name>` | measured metal reflectance (see §3.2) |
| `reflectance:<name>` | measured natural diffuse reflectance (see §3.3) |
| `preset:<illuminant>` | a named light SPD (see §3.4) |
| `spectrum:<name>` | reference to a previously declared `spectrum "name"` |
| `file:<path>` | a measured CSV curve `wavelength_nm,value` (comma or whitespace, `#` comments, header rows skipped) |
| `table { <λ>:<v> <λ>:<v> … }` | inline piecewise-linear tabulated curve |

`gaussian`/`shortpass` use `key=val` params (their values are continuations). The
`table` body is a flat list of `λ:value` tokens.

### 3.1 `glass:<name>` IOR curves

`BK7`/`crown`, `SF10`/`flint`, `silica`/`fused-silica`/`quartz`, `sapphire`,
`diamond`, `water`, `ice`, `acrylic`/`pmma`, `polycarbonate`/`pc`.

### 3.2 `metal:<name>` reflectances

`Au`/`gold`, `Ag`/`silver`, `Cu`/`copper`, `Al`/`aluminium`/`aluminum`,
`Cr`/`chromium`/`chrome`, `brass`.

### 3.3 `reflectance:<name>` natural diffuse

`leaf`/`vegetation`, `skin`/`skin-light`, `skin-dark`, `snow`, `soil`/`dirt`,
`brick`/`red-brick`, `concrete`.

### 3.4 `preset:<illuminant>` light SPDs

- Parametric: `bb<K>` (e.g. `bb6500`, Planckian), `led<K>k` / `led-<K>k` (phosphor LED
  at a colour temperature, e.g. `led4000k`).
- Named: `sun`, `daylight`/`d65`, `a`/`incandescent`, `led`, `led-warm`,
  `fluorescent`/`cfl`, `f2`/`cool-white`, `f7`/`daylight-fl`, `f11`/`triphosphor`,
  `hps`/`sodium`, `lps`/`sodium-low`, `mercury`/`hg`, `metal-halide`/`mh`.

---

## 4. `spectrum` — named reusable curves

```
spectrum "warm" = blackbody 3200
spectrum "leafy" = reflectance:leaf
```

Reference it anywhere a spectrum is expected with `spectrum:warm`. References resolve
lazily (a spectrum may be declared after its use); cycles up to depth 16 are caught.

---

## 5. `texture` — image maps

```
texture "wood" {
    file "maps/wood.png"
    encoding srgb            # srgb (default) | linear
    filter bilinear          # bilinear (default) | nearest
    wrap repeat              # repeat (default) | clamp | mirror
    palette { 0 spectrum:navy  1 metal:gold  2 rgb 0.8 0.1 0.1 }   # optional §9.3
}
```

- `file` is required (unless `rgb` is given, below); the path resolves relative to
  the working directory.
- Reflectance coefficients (Jakob-Hanika) are precomputed at load.
- `palette { <index> <spectrum-expr> … }` turns the texture's red channel into an
  indexed spectral lookup (nearest, no upsampling). Indices 0–255. (File textures only.)

Bind a texture to a material with `reflect texture:wood` (albedo), or to a scalar
parameter with `roughness texture:<name>` / `film_thickness_map texture:<name>` /
`weight_map texture:<name>`.

### 5.1 Procedural (function-defined) UV skins

Instead of a bitmap `file`, a texture's albedo may be defined by **three ftsl
expressions** of the surface UV coordinates — a UV-space procedural:

```
texture "grad" {
    rgb "u" "v" "0.5+0.5*sin(2*pi*4*u)"   # r(u,v)  g(u,v)  b(u,v), each quoted
    res 512                                # bake resolution (default 512, 1–8192)
    filter bilinear                        # bilinear (default) | nearest
    wrap clamp                             # repeat | clamp (default) | mirror
}
```

- The three expressions use the [`pattern`](#6-pattern--procedural-scalar-fields)
  infix grammar (`sin cos sqrt min max clamp mix step smoothstep …`, the constant
  `pi`); their variables are the surface `u, v` (world-space `x y z` carry no value
  in UV space). Each output is clamped to `[0,1]` and interpreted as **linear** RGB.
- The skin is baked **once at load** to a `res`×`res` linear grid, then flows through
  the *exact same* pipeline as an image texture — UV-wrap, Jakob-Hanika spectral
  upsampling, triplanar, GPU and raster paths all apply, and a material binds it
  unchanged with `reflect texture:<name>`.
- `res`, `filter`, `wrap` behave as for file textures. `palette` and `encoding` do
  not apply (the grid is already linear RGB). loom: `loom.ProcTexture` /
  `loom.func_skin(name, r, g, b, …)`.

---

## 6. `pattern` — procedural scalar fields

Two authoring forms:

**Infix formula (must be quoted):**
```
pattern "waves" { expr "0.5 + 0.5*sin(20*x)" }
```

**Named generator:**
```
pattern "stripes" { type bands axis x freq 8 phase 0 }
```

| `type` | params (defaults) |
|---|---|
| `axis` | `axis <x\|y\|z>` `scale`(1) `offset`(0) |
| `radial` | `center <x y z>`(0,0,0) `scale`(1) |
| `bands` | `axis <x\|y\|z>` `freq`(1) `phase`(0) |
| `checker` | `size`(1) |
| `noise` | `freq`(1) |
| `field` | `scale`(1) |

### 6.1 Expression language

Bind a pattern to a scalar material parameter with `pattern:<name>` (roughness,
film thickness, mix weight). Patterns are evaluated per hit.

**Variables:** `x y z` (world position), `f` (field value, for isosurfaces),
`nx ny nz` (surface normal), `r` (`|p|`), `u v` (surface texture coordinates — on
meshes and on native primitives that declare a `uv` wrap, see §9). Constant `pi`.

**Functions:** `abs sqrt sin cos tan exp log floor fract sign saturate` (1 arg);
`min max pow atan2 step` (2 args); `clamp mix smoothstep noise` (3 args).

**POV-Ray internal functions:** the whole classic `functions.inc` isosurface library is
built in — `f_torus`, `f_heart`, `f_klein_bottle`, `f_superellipsoid`, `f_dupin_cyclid`,
`f_helix1`, `f_spiral`, `f_boy_surface`, `f_kummer_surface_v1/v2`, … (~73 functions, exact
ports of POV-Ray's `source/vm/fnintern.cpp`). As in POV-Ray, the **first three arguments
are the coordinates** and the rest are parameters: `f_torus(x,y,z, majorR, minorR)`,
`f_heart(x,y,z, strength)`, `f_superellipsoid(x,y,z, e, n)`. They shine as `isosurface`
`function { expr "…" }` leaves (§ isosurface / README) but work in any pattern `expr`.
The ~5 noise-/pattern-based entries (`f_noise3d`, `f_ridged_mf`, `f_hetero_mf`, `f_ridge`,
`f_pattern`) are not yet available. Regenerate via `tools/pov_functions_gen.py`.

**Operators:** `+ - * / % ^` and unary `-`. `%` is floating-point modulo.

> **There is no `mod()` function** — use the `%` operator: `(floor(u*8)+floor(v*4)) % 2`.
> Unknown identifiers are a hard error ("unknown identifier '…'").

---

## 7. `material` — surfaces

```
material "name" { type <kind>  <params> }
material "name" { preset <recipe>  [overrides] }
```

### 7.1 `preset <recipe>` (whole-material recipes)

Fills a complete material; a few knobs may be overridden afterward
(`roughness`, `film_ior`, `film_thickness`, `film_thickness_map`, `reflect`, `ior`).

- **Metals** (glossy, roughness 0.05): every `metal:` name — `gold`, `silver`,
  `copper`, `aluminium`, `chromium`/`chrome`, `brass` (also `Au`/`Ag`/`Cu`/… ).
- **Glass** (dielectric): `glass` (=BK7) and every `glass:` name (`BK7`, `SF10`,
  `diamond`, `water`, …).
- **Structural colour**: `soap-bubble`/`bubble`, `oil-slick`/`oil`,
  `anodized-ti`/`anodized-titanium`, `morpho`, `beetle`/`jewel-beetle`,
  `nacre`/`mother-of-pearl`.

### 7.2 `type <kind>`

| type | key params (defaults) |
|---|---|
| `diffuse` | `reflect <spec>`(whitewall 0.75); `reflect texture:<n>` for a spatially-varying albedo |
| `translucent` | `reflect <spec>`(0.4); `transmit <spec>`(0.4) — two-sided Lambertian (diffuse transmission / thin-SSS look). Front hemisphere scatters `reflect`, back hemisphere scatters `transmit`; light diffuses THROUGH the surface. `reflect texture:<n>` allowed. Alias: `diffuse_transmit`. `reflect`+`transmit` are energy-clamped to ≤1. |
| `mirror` | `reflect <spec>`(0.95) |
| `halfmirror` | `reflect <spec>`(0.5) |
| `glossy` | `reflect <spec>`(0.9); `roughness <r>`(0.2) or `roughness pattern:/texture:<n>` |
| `dielectric` | `ior <spec>`(BK7); `roughness`(0)/map; `absorb <spec>`(0) Beer-Lambert tint per metre |
| `thinfilm` | `ior`(1.5); `film_ior`(1.30); `film_thickness <nm>`(300)/`film_thickness_map`; `substrate_k <spec>`(0) |
| `grating` | `reflect`(0.9); `groove_spacing <nm>`(1000); `groove_dir <x y z>`(0,1,0); `max_order`(3) |
| `fluorescent` | `reflect`(0.1); `absorb <spec>`; `emit <spec>`; `yield`(1) |
| `multilayer` | `ior`(1.5); `substrate_k`(0); ordered `layer <n> <k> <thickness_nm>` list (outermost first) |
| `mix` | stochastic blend of children — see §7.3 |
| `layered` | specular coat over a weighted body — see §7.4 |

Scalar-parameter maps: `roughness` / `film_thickness_map` / `weight_map` accept
`pattern:<name>` (math over x,y,z,normal,u,v) or `texture:<name>` (grayscale UV map).

### 7.3 `mix` — stochastic material blend

```
material "speckle" {
    type mix
    layer "gold" 0.5
    layer "ink"  0.5
    weight_map pattern:uvcheck    # optional, requires exactly 2 layers
}
```

- Each `layer "child" weight` names another material and its selection probability;
  weights must be ≥ 0 and sum to ≤ 1.
- Children may be declared before or after the mix (resolved in a second pass).
  **Nesting is not allowed** (a layer may not itself be a mix/layered).
- `weight_map pattern:/texture:<n>` drives child-0's probability per hit (child 1 =
  1 − map); requires **exactly two** layers.

### 7.4 `layered` — physical coat + body

```
material "carpaint" {
    type layered
    ior 1.5                       # body / effective index
    coat {
        reflectance fresnel       # fresnel (default) | thinfilm | manual
        roughness 0.05            # glossy coat lobe; map allowed
        ior glass:BK7             # coat index (fresnel/thinfilm)
        film_ior 1.30  film_thickness 300  film_thickness_map texture:t   # thinfilm coat
        specular 0.05             # flat reflectance (manual model)
    }
    layer "base" 1.0              # body lobes, resolved like a mix
}
```

### 7.5 `record` — parametric slot LUTs

A **record** is a named bank of per-channel look-up tables over a shared scalar
domain `[lo,hi]`. A per-hit **driver** scalar samples every channel at once; each
channel whose name matches a real material slot fills that slot at the driven value.
This turns a single expression into a coordinated sweep across a material's slots.

```
# palette that ramps a diffuse albedo steel -> gold -> copper across the domain 0..1
grad = range 0-1 [
    reflect   spectrum:steel  spectrum:gold  spectrum:copper
    roughness 0.05  p:0.7 0.4  0.6
    interp    smooth
]
```

- **Declaration:** `NAME = range LO-HI [ … ]` (top-level, like `spectrum`/`pattern`).
  The domain is `LO-HI` or `LO HI` with `HI > LO`.
- **Channels** are one line each, `name stop stop …`. The channel **name is matched
  to a material slot by destination** — `reflect` drives the diffuse albedo,
  `roughness` drives the glossy roughness. A name that matches no slot is simply not
  auto-bound (reserved for later stages); a slot with no channel keeps its constant.
- **Stops** are laid out evenly across the domain. Prefix any stop with `p:<pos>` to
  **pin** it to an explicit domain position; unpinned runs redistribute between the
  pinned anchors. Positions must stay in `[lo,hi]` and be non-decreasing.
- A channel is a **colour** LUT iff its stops are prefixed spectrum refs (they contain
  `:`, e.g. `spectrum:steel`, `metal:copper`); otherwise it is a **scalar** LUT whose
  stops are pattern expressions (a literal, or math over `x y z nx ny nz r u v f` and
  functions like `noise(…)` — the §6.1 language). A channel may not mix the two.
- **`interp nearest|linear|smooth`** selects the sampling mode (default `linear`).
  `smooth` is a monotone (Fritsch–Carlson) cubic — no overshoot. Colour channels
  interpolate in linear RGB, then upsample back to a reflectance spectrum.

**Binding a record to geometry** uses the inline `material NAME(driver)` form in any
primitive's `material` field, where `driver` is a pattern expression evaluated per
hit:

```
sphere { center 0 0 0  radius 1  material grad(u) }              # sweep along u
sphere { center 2 0 0  radius 1  material grad(noise(9*x,9*y,9*z)) }  # mottled by noise
```

**Override blocks** — a named `material "m" { … }` block can *import* a record in
bulk with `from R(driver)` and then override individual slots. Statements resolve in
written order, **last-write-wins** per slot:

```
material "swept" {
    from palette(u)                     # bulk import: every matching channel binds,
}                                        #   all driven by u
material "mixed" {
    type glossy
    from palette(noise(7*x,7*y,7*z))    # record drives reflect (mottled by noise)…
    roughness = 0.5*(1+sin(v*12.56))    # …but roughness comes from a direct expression
}
material "picked" {
    from palette(u)
    reflect = palette.reflect[2]        # constant stop selector: pin to the 3rd stop
}
```

A slot RHS after `=` may be:
- a **math expression** (scalar slots only) — `roughness = 0.5*(1+sin(v*12.56))`;
- a **bare imported channel name** — binds that channel using the driver it was
  imported with;
- **`RECORD.channel`** — the channel of a record, driven by the `from RECORD(…)`
  driver in the same block (error if that record was never imported);
- **`RECORD.channel[i]`** (or **`self.channel[i]`**) — a **constant stop selector**
  that pins the slot to the channel's `i`-th stop (0-based), ignoring the driver.

A later `from` re-imports and its driver wins for the slots it covers. Type rules
still apply: `reflect` needs a colour channel, `roughness` a scalar; assigning a
scalar expression to `reflect` is an error.

**Record refs as plain values** — a record channel can also be read *anywhere a value
is read*, not just inside an override block, as long as the reference resolves to a
**constant** (a record is a curve, so a value site needs one point on it). Two forms:

```
spectrum "cu" = palette.reflect[2]              # top-level spectrum = the 3rd reflect stop
material "pick" { type diffuse reflect palette.reflect[2] }   # colour slot = a stop, verbatim
material "samp" { type diffuse reflect palette.reflect(0.0) } # colour slot = sample at driver 0.0
material "frost" { type glossy reflect steel roughness palette.gloss(0.5) }  # scalar slot = sample
```

- **`RECORD.channel[i]`** — the channel's `i`-th stop (0-based), used verbatim.
- **`RECORD.channel(c)`** — the channel sampled at a **constant** driver `c`
  (interpolated per the record's `interp`). The channel must be explicit here — the
  bare-`RECORD(c)` shorthand only exists inside a material where the destination slot
  names the channel.

A colour site (a `spectrum`, a `reflect`/`transmit`/`emit` slot) needs a **spectrum**
channel; a scalar site needs a **scalar** channel — a type mismatch is an error, as is
a stop index out of range.

**Scope check.** The driver in `RECORD.channel(c)` is an ordinary pattern expression,
but a value site only permits the driver *variables that are in scope there*. A material
slot is a per-hit surface site, so surface intrinsics (`x y z nx ny nz r u v f`) are in
scope — but a **standalone value** (a top-level `spectrum`, a light SPD, a camera scalar)
is evaluated once at load time, so **no** per-hit variable is in scope. Writing
`palette.reflect(u)` at such a site is a **scope error** (`u` has no meaning there); only
a constant driver like `palette.reflect(0.3)` is allowed. This is the general rule behind
"can a light carry a `from`?" — yes, but only a constant one.

> **GPU note:** record-driven materials currently render on the **CPU only** — a scene
> that binds a record falls back from the GPU forward/backward tracer automatically
> (`[device] … -> CPU (…parametric record…)`). GPU parity is a later stage.

---

## 8. Geometry primitives

Positions are unit-scaled; direction/edge vectors (`u`, `v`, `normal`) are scaled as
directions. Every primitive needs a `material "name"` (declared earlier or later).

### 8.1 `sphere`
```
sphere { center 0 0 0  radius 1  material white }
```
Only translate + rotation + **uniform** scale is allowed on a sphere (a non-uniform
scale would need an ellipsoid — use an `isosurface { ellipsoid … }` or a mesh).

### 8.2 `quad`
```
quad { origin 0 0 0  u 1 0 0  v 0 0 1  material white }
```
A parallelogram from `origin` spanning edge vectors `u`, `v`. Emits two triangles with
UVs origin=(0,0), +u=(1,0), +v=(0,1).

### 8.3 `triangle`
```
triangle { v0 0 0 0  v1 1 0 0  v2 0 1 0  material white }
```

### 8.4 `mesh`
```
mesh "bunny" {
    file "meshes/bunny.obj"
    material white
    translate 0 0 0   rotate 0 45 0   scale 1        # scale: uniform value or `sx sy sz`
    uv use_mesh            # use OBJ vt coords; or planar|spherical|cylindrical|triplanar (§9)
    usemtl use_names       # switch material per OBJ usemtl group by name match
}
```

OBJ **vertex normals (`vn`) are read automatically** as smooth shading normals:
if the OBJ supplies `vn` (face tokens like `f v//vn` or `v/vt/vn`), the hit
barycentric-interpolates them for smooth shading (no per-facet look), transformed
by the mesh transform's inverse-transpose. A mesh with **no `vn` stays exactly
flat-shaded** (geometric normal), so it's a no-op for older assets. No FTSL key is
needed — it's driven entirely by the OBJ contents.

**glTF 2.0 / GLB.** `file` may point at a `.gltf` (JSON, embedded/external/base64
buffers) or a `.glb` (binary container); the loader dispatches on the extension.
It bakes the glTF node transform hierarchy (matrix or TRS) under the mesh block's
own `translate/rotate/scale`, reads `POSITION` / `NORMAL` / `TEXCOORD_0` +
indices, and imports `pbrMetallicRoughness` materials — `baseColorFactor` is
upsampled to a reflectance spectrum, `metallicFactor ≥ 0.5` → a glossy (metal)
BSDF tinted by the base color, else diffuse, with `roughnessFactor` as the lobe
width. Add `import_materials no` to ignore glTF's materials and paint every
primitive with the block's FTSL `material` instead. The block `material` is always
the fallback for primitives that carry no material. *Not supported* (see
known-issues): textures, KHR extensions (transmission/clearcoat/…), skinning,
morph targets, sparse accessors, animation, and non-triangle primitives (skipped).

### 8.5 `mesh_asset` + `mesh_instance` — instancing (two-level BVH)

A `mesh` bakes its triangles into the scene, so ten copies cost ten triangle
sets. **Instancing** loads the geometry once and places it many times through
per-placement affines — the copies share one set of triangles and one
bottom-level BVH (a two-level / TLAS-over-BLAS acceleration structure).

```
mesh_asset "ball" {           # load ONCE into local space (no world transform)
    file "meshes/sphere.obj"  # .obj / .gltf / .glb (same loaders as `mesh`)
    material ivory            # fallback material for primitives without one
    uv use_mesh               # optional: read OBJ vt
    usemtl use_names          # optional: per-usemtl-group material by name
    import_materials no       # optional (glTF): ignore glTF materials
}

mesh_instance {               # cheap placement of a named asset
    of "ball"                 # references the mesh_asset by name
    translate 0.5 0.3 0.5     # placement transform (like `mesh`); composes with
    rotate 0 45 0             #   an enclosing group{} and the scene unit scale
    scale 0.22                # uniform value or `sx sy sz`
    material gold             # optional: override the asset's materials for THIS
}                             #   placement (omit to keep the asset's own materials)
```

- The asset is stored in its **authored (local) space** — no transform is baked
  in, so one asset serves differently scaled/rotated placements.
- A `mesh_instance` may appear at top level or inside a `group{}` (its transform
  composes with the group's, exactly like `mesh`).
- **CPU:** true instancing — instances share the BLAS triangles, so N copies add
  only N affines to memory. All render modes (A/B/C/R/D/P and the photon modes)
  traverse the two-level BVH. **GPU:** instances are expanded to world-space
  triangles at upload (flat device memory — the memory saving is CPU-only; images
  are identical). See known-issues.

---

## 9. UV wraps on native primitives and meshes

Pattern/texture expressions can see surface texture coordinates `u`, `v`. Where they
come from:

- **`quad` / `triangle`**: built-in UVs (parallelogram / barycentric).
- **native `sphere`**: built-in equirectangular (lat/long) UV.
- **`mesh`**: `uv use_mesh` (OBJ `vt`), or a synthesized projection
  `uv planar|spherical|cylindrical [axis=x|y|z]`, or `uv triplanar [scale=<s>]`.
- **`isosurface`**: `uv planar|spherical|cylindrical [axis=x|y|z]` — the SAME
  projection meshes use, referenced to the primitive's world AABB. Default axis `y`.

> The projection **axis must be `key=val`**: `uv planar axis=z`. A bareword
> (`uv planar z`) would be parsed as a separate statement and the axis silently
> ignored (§1.1). Likewise `uv triplanar scale=4` (or a bare number `uv triplanar 4`).

---

## 10. `isosurface` — analytic SDFs, CSG, and functions

An `isosurface { material <m>  <one root field element> }` builds an implicit surface.
The root is **exactly one** leaf or CSG combinator (wrap multiple shapes in a
`union { … }`). Every element may carry `translate`/`rotate`/`scale` that composes
down the tree.

### 10.1 Leaves
| leaf | params |
|---|---|
| `sphere` | `center` `radius`(1) |
| `ellipsoid` | `center` `radius <rx ry rz>` |
| `box` | `size <x y z>`(1,1,1) `round`(0 corner radius) |
| `torus` | `major`(1) `minor`(0.25) |
| `cylinder` | `radius`(0.5) `height`(1) — axis = local y |
| `cone` | `radius`(0.5 bottom) `radius2`(0 top) `height`(1) |
| `plane` | `normal <x y z>`(0,1,0) `offset`(0) |
| `function` | `expr "f(x,y,z)"` — needs `contained_by` (see §10.3) |

#### The `function` expression language

`expr "…"` is compiled by the **same** math parser as material patterns (§6.1) — a
full infix expression compiler (`src/pattern.h`, shunting-yard → a flat postfix VM that
runs identically on CPU and GPU). The surface is the zero set `f = 0`; the value is a
raw field (not necessarily a true distance), so a `function` leaf needs `contained_by`
and a Lipschitz bound (`max_gradient`, see §10.3).

- **Variables (live in a field expression):** `x y z` (leaf-local position, after any
  `translate`/`rotate`/`scale`), `r` = `sqrt(x²+y²+z²)`, and the constant `pi`. The
  other pattern variables (`f`, the normal `nx ny nz`, and UV `u v`) are meaningless
  when *defining* a surface and read `0`.
- **Functions:** `abs sqrt sin cos tan exp log floor fract sign saturate` (1 arg);
  `min max pow atan2 step` (2 args); `clamp mix smoothstep noise` (3 args). `noise` is
  deterministic 3-D value noise in `[0,1]` (same on CPU/GPU).
- **Operators:** `+ - * / % ^` and unary `-`. `^` is `pow` (right-assoc), `%` is
  floating-point modulo. There is **no** `mod()` function — use `%`. Unknown
  identifiers are a hard error.

```
# a gyroid shell, and a noise-warped sphere
isosurface { material gold
    function { expr "abs(sin(6*x)*cos(6*y)+sin(6*y)*cos(6*z)+sin(6*z)*cos(6*x)) - 0.4" }
    contained_by { min -1 -1 -1  max 1 1 1 }  max_gradient 12 }
isosurface { material wax
    function { expr "r - 0.8 - 0.15*noise(3*x, 3*y, 3*z)" }
    contained_by { min -1.2 -1.2 -1.2  max 1.2 1.2 1.2 }  max_gradient 3 }
```

### 10.2 Combinators
`union`, `intersect`/`intersection`, `difference`/`subtract`, and the smooth
variants `smooth_union`, `smooth_intersect`, `smooth_difference` (each takes a blend
radius `k <len>`), plus `blob` (= smooth union). Children are nested blocks; they fold
pairwise in order.

```
isosurface {
    material gold
    intersect {
        function { translate 0.4 0.37 0.45
                   expr "abs(sin(40*x)*cos(40*y)+sin(40*y)*cos(40*z)+sin(40*z)*cos(40*x)) - 0.5" }
        sphere { center 0.4 0.37 0.45  radius 0.32 }
    }
    contained_by { min 0.06 0.03 0.11  max 0.74 0.71 0.79 }
    max_gradient 80
}
```

### 10.3 Marching controls
| key | values | notes |
|---|---|---|
| `contained_by` | `{ min <x y z>  max <x y z> }` | **required** for any `function` field (marched only inside it) |
| `max_gradient` | `<L>` | Lipschitz bound for a function field (else auto-estimated ×1.3) |
| `accuracy` | `<len>` | min march step (function fields) / fixed step (sample method) |
| `method` | `adaptive`(default) / `sample`(=`fixed`) | POV-Ray-style fixed marching for untrusted bounds |
| `samples` | `<n>` | intervals across the box diagonal (sample method) |
| `refine` | `bisect`(default) / `regula_falsi`(=`falsi`/`secant`) | root refinement once bracketed |
| `uv` | `planar\|spherical\|cylindrical [axis=…]` | §9 UV wrap |

Analytic SDF leaves + CSG stay unit-Lipschitz; only `function` fields need
`contained_by` + a gradient bound.

### 10.4 Exporting an isosurface to a mesh (CLI, not FTSL)

An isosurface scene can be **polygonised to a watertight OBJ** instead of rendered, for
import into Unreal / Blender. This is a command-line action, not scene syntax:

```
ftrace -in scene.ftsl -export-mesh out.obj -mesh-res 192
ftrace -in scene.ftsl -export-mesh out.obj -mesh-res 256 -mesh-adaptive -mesh-decimate 0.35
```

- `-export-mesh <out.obj>` — polygonise every `isosurface` in the scene (each becomes one OBJ
  object) with **marching tetrahedra** (no ambiguous cases ⇒ guaranteed watertight 2-manifold),
  then exit. Uses the exact `f(x,y,z)` for crossings and `∇f` for normals, intersects the field
  with its `contained_by` box so boundary-reaching surfaces seal into closed solids (flat caps),
  welds vertices by a canonical grid-edge id (no cracks), and winds triangles outward.
- `-mesh-res <N>` — fineness: cells along the longest bounds axis (default 128); other axes
  scale proportionally so cells stay ~cubic.
- `-mesh-adaptive` / `-mesh-decimate <f>` — curvature-adaptive quadric-error decimation:
  triangles thin out on flat regions and stay dense where the surface curves. `<f>` is the
  triangle fraction to keep (default 0.5; passing `-mesh-decimate` implies `-mesh-adaptive`).

See README → **Exporting an isosurface to a mesh** for details; the code is `src/isomesh.h`.

---

## 11. `light` — emitters

Every scene needs at least one `light`. All lights take `spd <spectrum-expr>`
(default `blackbody 6500`). Surface/tube/spot emitters can be given an absolute flux
with `power <watts>` (radiant) or `lumens <lm>` (photometric) — this flips the whole
scene to fixed-exposure output (`power` wins if both given). Env lights reject
`power`/`lumens`.

| subtype | keys (defaults) |
|---|---|
| `area` (default) | `origin` `u` `v` `normal`(from u×v) `spd` — a rectangle |
| `collimated` | `dir`(0,0,-1) `origin`(0.5,0.5,0.95) `spd` — a thin pencil beam |
| `sphere` | `center` `radius`(0.1) `spd` — a glowing ball (also dropped into geometry) |
| `cylinder` | `center` `axis`(0,1,0) `length`(0.5) `radius`(0.05) `segments`(48) `caps`(off) `spd` — a tube/fluorescent |
| `spot` | `origin`(0.5,0.99,0.5) `dir`(0,-1,0) `inner_angle`(20°) `outer_angle`(30°) `spd` |
| `env` | constant: `spd`. Image-based: `file "sky.hdr"` `rotate`(0°) `intensity`(1) |

`caps on`/`true`/`yes` closes the cylinder (emissive end discs). `spot` angles are
half-angles in degrees with a smoothstep penumbra between inner and outer.

---

## 12. `medium` — participating volume

```
medium { sigma_t 0.5  albedo 0.9  g 0  rayleigh false }
medium { sigma_a <spec>  sigma_s <spec>  g 0 }     # spectral form
```

- `sigma_t` + `albedo` splits into scattering/absorption; or give spectral
  `sigma_a`/`sigma_s` directly (per authored-unit length; converted to 1/metre).
- `g` is the Henyey-Greenstein anisotropy; `rayleigh true` gives a λ⁻⁴ scattering
  tilt (blue-sky falloff).

**Angular scattering model — `phase`.** By default a medium scatters through the
smooth single-parameter Henyey-Greenstein lobe (`g` above). A `phase` statement swaps
in a different angular model:

```
medium {
    sigma_t 0.0012  albedo 0.99
    bounds { min -260 -260 12   max 260 260 190 }
    phase rainbow {
        droplet_um    500      # water-drop RADIUS in microns (default 500 = 0.5 mm rain)
        secondary     on       # p=3 secondary bow            (default on)
        supernumerary on       # Airy side-maxima / supernumerary arcs (default on)
        strength      1.0      # weight of the bows over the smooth forward haze (default 1)
        forward_g     0.55     # HG anisotropy of the smooth forward-scatter background
        secondary_ratio 0.43   # secondary brightness relative to primary
    }
}
```

- `phase hg` (or no `phase` statement) is the default Henyey-Greenstein lobe — nothing
  changes; a bare `phase hg` is only for making the choice explicit.
- **`phase rainbow { .. }`** replaces the lobe with a physically-tabulated **water-droplet
  phase** (Airy theory of the rainbow, `rainbow.h`). A fog/rain medium then shows a real
  **primary bow (~42°) + secondary bow (~51°)**, wavelength dispersion (red outer / violet
  inner on the primary, *reversed* on the secondary), **Alexander's dark band** between
  them, and **supernumerary arcs**. Its physical features are **on by default**; the block
  knobs are overrides (turn a feature off or retune it). When set, the rainbow phase
  **overrides `g`** for this medium.
- Smaller drops broaden and desaturate the bow toward a white **fogbow** (try
  `droplet_um 10`); 0.5–1 mm rain drops give the crispest bows and supernumeraries.
- **Geometry that shows a bow:** the sun must be *behind the camera* and effectively far
  away (parallel rays → sharp bow); aim the camera at the **antisolar point** (the shadow
  of your head) and the bow appears as a ring at ~42° radius around it. Keep the fog
  **thin** (optical depth ≲ 0.3) so single scattering — which carries the bow — dominates
  over the multiply-scattered veil.
- **Mode support:** the tabulated rainbow phase is evaluated by the **CPU** tracers
  (forward A/B/C, backward R, BDPT D). The **GPU** volume path only knows the analytic HG
  lobe, so a scene with a rainbow-phase medium automatically **falls back to the CPU**
  tracer (rather than silently dropping the bow to a smooth haze).

**Multiple media.** You may author several `medium` blocks; they coexist as
independent, possibly overlapping regions (e.g. two differently-tinted fog orbs plus
a faint global haze). The forward tracer superposes them physically: extinction adds,
so total transmittance is the *product* of the per-medium transmittances, and each
collision is drawn from the *earliest* of the media's independent free-flights (with
the winning medium's albedo/`g` driving the scatter). A scene with a single `medium`
is bit-identical to before. *(Superposition is a forward-mode feature — see the mode
note at the end of §12.1.)*

### 12.1 Bounded and heterogeneous fog (blobs)

By default the medium is a single global homogeneous haze filling the whole scene.
Two optional sub-parts localize it and give it internal structure, so the fog forms
discrete **blobs with soft, formula-defined boundaries**:

```
medium {
    sigma_t 4.0   albedo 0.95   g 0.0
    bounds  { min 0.7 0.1 1.9   max 3.3 2.7 4.5 }        # (alias: contained_by)
    density "pow( saturate( 1 - sqrt((x-2)^2+(y-1.4)^2+(z-3.2)^2)/1.2 ), 2 )"
}
```

- **`bounds { min <x y z>  max <x y z> }`** — an axis-aligned box (authored units →
  metres) that the fog is confined to. A ray's fog interaction is clipped to its
  overlap with the box, so nothing scatters outside it. (Alias: `contained_by`.)
- **`bounds { center <x y z>  radius <r> }`** — a **sphere** region instead of a box:
  the fog fills exactly a ball, e.g. **the whole inside of a glass sphere**. Author the
  same `center`/`radius` as your sphere geometry and the fog is clipped precisely to
  that sphere (round silhouette). This is the simple *per-object* fog: co-locate the
  region with an object to fill it. (A `density` field works inside a sphere bound too —
  its majorant grid uses the sphere's AABB.) *Note (accuracy limitation):* fog inside an
  actual `dielectric` shell is **not imaged directly** by the next-event modes — the
  pinhole splat (`B`) *and* the finite-lens splat (`A`) connect the fog to the camera with a
  **straight** ray, which the glass occludes (and could not refract anyway), so seeing the
  fog *through* the glass renders black. The fog still correctly **lights the surrounding
  room** in those modes. **BDPT (`D`) images fog-through-glass correctly** — its camera
  subpath refracts through the shell to a volume vertex and MIS-connects to the light, so a
  lantern glowing inside a fogged glass sphere renders as a bright disc; photon-catch (`C`)
  traces the same path but far more slowly. An *open* fog sphere (no glass shell) is
  directly viewable in every mode.
- **`bounds { object "<name>" }`** — shape the fog to a **named scene object** instead
  of authoring a box/sphere by hand. Give any `sphere`, `isosurface`, or `mesh` a
  `"name"` and reference it here:
  - a named **sphere** → the exact analytic sphere bound (its world center/radius);
  - a named **isosurface** → **field membership** — the fog fills the field's interior
    (a point is inside when the field is negative, auto-detected from the field's
    sign at its bounding-box center), so the fog takes the metaball/SDF silhouette
    exactly (carved per-point during delta/ratio tracking over the field's AABB). A
    `density` field still multiplies on top, shaping the fog *within* the iso-shape.
  - a named **mesh** → the mesh's world **AABB** (a box approximation; true mesh
    containment is deferred — see `known-issues.md`).

  The object may be authored anywhere in the file (media are resolved after all
  geometry). The named object's own material/visibility is unaffected — only its
  *shape* is borrowed for the fog bound.
- **`density <expr>`** or **`density pattern:<name>`** — a scalar field, ≥ 0, that
  multiplies `sigma_t` (and hence both `sigma_a` and `sigma_s`) at each point. Uses
  the same infix expression language as isosurface `function` fields and `pattern`
  leaves (§6.1 / §10.1): variables `x y z` (world position, metres), `r` (distance
  from the world origin), and the constant `pi` are live; `f`, the normals, and `u v`
  read 0. A smooth radial falloff like the example above gives a blob whose edge fades
  gradually rather than a hard surface. Albedo stays spatially constant (density scales
  absorption and scattering together), so only the *amount* of fog varies in space,
  not its color.
- **`density vdb:<path>`** — instead of a formula, sample the density from an imported
  **NanoVDB** volume. `<path>` is an **unquoted** bareword path to a `.nvdb` file
  (uncompressed, float grid): `density vdb:scraps/cloud.nvdb`. Convert a `.vdb` to
  `.nvdb` with OpenVDB's `nanovdb_convert` (or generate a test asset with
  `scraps/make_nvdb.cpp`). On load the grid is baked into a dense lattice + world→index
  transform and sampled trilinearly — the *same* sampler on CPU and GPU. The grid's world
  AABB auto-seeds the medium **bound** and its peak value the **majorant**, so no `bounds`
  or `density_max` is needed (either still overrides). Values scale `sigma_t` just like the
  formula form, so dial optical thickness with `sigma_t`. Only float grids; the dense bake
  is memory-capped (see `known-issues.md`).
- **`density_max <v>`** — the delta/ratio-tracking majorant (an upper bound on the
  density over the region). If omitted it is auto-estimated on a 24³ grid over
  `bounds` (×1.3 safety), so a heterogeneous medium needs either a `bounds` box or an
  explicit `density_max`. Set it explicitly if your field can spike between grid
  samples.

The sampler uses unbiased **delta (Woodcock) tracking** for scattering and **ratio
tracking** for shadow-ray transmittance, so the result is exact (no voxelization). A
plain homogeneous `medium` (no `density`, no `bounds`) is unchanged and bit-identical
to before.

> **Mode support:** *heterogeneous* (`density`-field) fog is honored by the **forward**
> light tracer — modes **A/B/C** (and the forward layers of V/P) — **and by BDPT (D)**, on
> **both the CPU and the GPU** (`-device gpu` runs the identical density VM + delta/ratio
> tracking). **BDPT (D)** renders every kind of medium — global haze, superposed media,
> box/sphere/object-**bounded** fog, and **heterogeneous `density`-field blobs** — unbiased
> on both the CPU and the GPU: subpath medium vertices are placed by delta tracking and
> connections weighted by ratio-tracking transmittance, exactly as the forward tracer
> samples them. (The MIS weights omit the heterogeneous transmittance — a variance-only
> PBRT-v3 simplification; the balance heuristic is a partition of unity, so the estimator
> stays unbiased regardless.) The backward reference (R/V) and the camera-side layer of the
> P composite treat the medium as a single global homogeneous haze and **ignore** `density`
> and `bounds` (the renderer warns when you do this). Render heterogeneous fog for those
> modes with a forward mode instead.

---

## 13. `group` — transform hierarchy

```
group {
    translate 0 0.5 0   rotate 0 30 0   scale 2      # scale: uniform or `sx sy sz`
    sphere { center 0 0 0  radius 0.2  material gold }
    group { translate 0.5 0 0   mesh "m" { … } }     # nesting recurses
    light area { … }
}
```

A group's transform composes with its parent's (parent applied last) and is baked
into every child at load time (no runtime scene graph). Allowed children: `sphere`,
`quad`, `triangle`, `mesh`, `light`, nested `group`. A light anywhere in the tree
satisfies the "scene needs a light" check.

---

## 14. Cameras

Any number of `camera` blocks accumulate. The CLI selects one (or renders all).
Multi-camera renders write one file per camera, inserting `_<camname>` before the
output extension (`-o png/out.png` → `png/out_<name>.png`).

### 14.1 `camera`
```
camera "cam" {
    eye 0.8 0.62 2.35   look_at 0.8 0.28 0.55   up 0 1 0
    fov_y 42            # vertical field of view (degrees)
    aperture 0.02       # thin-lens aperture radius (legacy DoF)
    focus 2.3           # focus distance (0 = infinity)
    mode R              # A|B|C|R|D — per-camera measurement model
    lens 50  fstop 2.8  zoom 1    # photographic authoring (overrides fov_y/aperture)
    projection rectilinear        # or `fisheye [type]` (§14.4)
    film { res 1200 800  format full-frame  iso 100 shutter 1 exposure 1 }
    lens { … }          # optional physical multi-element lens (§14.5)
}
```

| key | default | notes |
|---|---|---|
| `eye` / `look_at` / `up` | (0,1,3)/(0,1,0)/(0,1,0) | `up` is a direction (unscaled) |
| `fov_y` | 40 | vertical FOV in degrees |
| `aperture` | 0.02 | thin-lens radius (metres) |
| `focus` | 0 (∞) | focus distance |
| `mode` | inherit CLI | `A` finite-lens forward splat, `B` pinhole splat, `C` finite-aperture catch, `R` backward reference, `D` BDPT |
| `lens <mm>` | — | focal length; sets fov_y = 2·atan(filmH/2f); overrides `fov_y` |
| `fstop <N>` | — | aperture radius = focal/2N; seats film at image distance for A/C DoF |
| `zoom <x>` | 1 | multiplies focal length (x>1 tele/narrower) |

### 14.2 `film { … }`

| key | notes |
|---|---|
| `res W [H]` | resolution; `res W` is square (H=W). CLI `-r` can override |
| `format <name>` | named sensor → physical mm (see §14.3); `size` overrides it |
| `size <Wmm> <Hmm>` | explicit physical sensor in millimetres |
| `iso` / `shutter` / `exposure` | exposure *compensation* on top of auto-exposure: `comp = exposure·(iso/100)·shutter` |

### 14.3 Film formats
`full-frame`/`35mm`/`135`/`ff`, `half-frame`, `super35`/`s35`, `academy`,
`aps-c`, `aps-h`, `micro-four-thirds`/`mft`/`m43`/`four-thirds`, `1-inch`/`1in`,
`medium-format`/`645`/`6x45`, `6x6`, `6x7`, `6x9`, `digital-medium-format`/`gfx`,
`large-format`/`4x5`/`5x4`, `8x10`. (Case/space/underscore/hyphen-insensitive.)

### 14.4 Lens projection
`projection <name>` or the `fisheye [type]` shorthand (bare `fisheye` = equisolid).
Names: `rectilinear`/`perspective`/`normal`/`pinhole`, `equidistant`/`fisheye`,
`equisolid`/`equal-area`, `stereographic`, `orthographic`/`ortho`. The fisheye modes
allow `fov_y ≥ 180`.

### 14.5 Physical `lens { … }` block

Renders through the backward realistic-camera path (mode R), tracing real glass
interfaces (radii/thicknesses/apertures in **millimetres**).

```
lens { preset achromat  focal 50  fstop 2.8  glass BK7 }
lens { surface <radius_mm> <thickness_mm> <ior> <semi_aperture_mm> [stop]  … }
```

- Explicit `surface` lines take priority (paste a real prescription; `ior` = glass
  name or a number; `stop` marks the aperture stop).
- Otherwise a `preset`: `singlet`/`biconvex`/`simple`, `achromat`/`doublet`,
  `telephoto`, `wide` (default = achromatic doublet at the derived focal/f-number).
- Sensor size comes from the camera `film` (default full-frame); autofocuses at `focus`.

---

## 15. Camera animation

Both expand into N `CamSpec` frames sharing `look_at`/`up`/`fov_y`/`mode`/`film`/
`lens`. Each frame's output file is `<base><zero-padded index>`, so a multi-camera
render writes one PNG per frame ready for ffmpeg.

### 15.1 `camera_path` — keyframed dolly
```
camera_path "dolly" {
    look_at 0 1 0   up 0 1 0   fov_y 40   mode B   frames 60
    film { res 256 256 }
    key <t> <ex ey ez> [<lx ly lz>] [<fov>]      # >= 2 keys, t in [0,1]
    dolly_zoom            # optional: hold subject size (Vertigo), trade fov vs distance
    exposure_lock         # optional: share frame-0 auto-exposure anchor (no flicker)
}
```
Keyframe field count disambiguates: `4`=t,eye · `5`=t,eye,fov · `7`=t,eye,look ·
`8`=t,eye,look,fov. Frame i samples t = i/(frames−1) with piecewise-linear
interpolation between bracketing keys.

### 15.2 `camera_orbit` — turntable / fly-around
```
camera_orbit "spin" {
    center 0.40 0.37 0.45   radius 0.45   height -0.20   axis y
    up 0 1 0   fov_y 78   mode R   frames 120
    look_at 0.40 0.37 0.45      # optional (defaults to center)
    start_deg 0   sweep_deg 360 # optional
    exposure_lock               # optional
    film { res 900 900 }
}
```
The eye rides a circle of `radius` in the plane perpendicular to `axis` (`x|y|z`,
default y), offset `height` along the axis from `center`. A full 360° sweep is sampled
at i/frames (frame N == frame 0, not duplicated → seamless loop); a partial sweep
spans its endpoints via i/(frames−1). `center` and `radius > 0` are required.

### 15.3 `camera_curve` — spline fly-through with variable speed
```
camera_curve "fly" {
    point 0.3 0.6 2.2   point 0.8 0.6 1.6   point 1.3 0.6 2.2   point 0.8 0.6 2.6
    up 0 1 0   fov_y 45   mode R
    frames 90                   # fixed count (uniform arc length if no density)
    density 20                  # OR: cameras per unit length (constant)
    density_at 0 6   density_at 0.5 30   density_at 1 6    # OR: variable density
    look_at 0.8 0.3 0.6         # orientation: fixed target
    up_frame travel             # optional: bank the up axis into the curve's RMF
    closed                      # optional: loop the curve seamlessly
    exposure_lock               # optional
    film { res 900 600 }
}
```
The eye rides a **Catmull-Rom spline** that passes through every `point` control point
(≥ 2 required). Where cameras sit is set by:

- **`frames N`** — a fixed count. With no density, they are spaced at **uniform arc
  length** (constant speed). With a density, `frames` still fixes the count but the
  density **distributes** those N cameras (more where density is high).
- **`density <ρ>`** / **`density_at <t> <ρ>`** — cameras per unit length. `density`
  is constant; `density_at` keyframes it (piecewise-linear over normalized position
  `t ∈ [0,1]`, t=0 first point, t=1 last). Without `frames`, the count is the integral
  of ρ over the curve. **High density = many closely-spaced frames = slow dwell**
  through that stretch; low density = fast. This is the camera's "speed" curve.

**Orientation** (`look …`):

| mode | how |
|---|---|
| `look tangent` (default) | aim along the direction of travel (curve tangent) |
| `look_at <x y z>` | a fixed target for every frame |
| `look curve` + `look_point <x y z> …` | aim at a **second** Catmull-Rom spline (≥ 2 `look_point`s), sampled in step with the eye |

`closed` loops the curve (wrap-around Catmull-Rom, sampled i/N so frame N == frame 0);
an open curve spans both endpoints via i/(N−1). All frames share
up/mode/film; `exposure_lock` shares the frame-0 exposure anchor.

**Two-axis orientation (forward + up).** The camera basis is fixed by two authored
axes; `right` is always derived from them, so you never set it. Each axis can be read
in one of two **reference frames**:

- **`frame world|travel`** — default for *both* axes. `world` = the fixed world axes
  (the classic behavior). `travel` = the curve's **rotation-minimizing frame** (RMF), a
  twist-free moving basis carried along the path by parallel transport (double-reflection
  method), so the shot banks naturally into turns instead of the Frenet frame's abrupt
  flips. On a `closed` loop the RMF's residual holonomy is distributed evenly so the
  frame closes seamlessly (frame N == frame 0).
- **`fwd_frame world|travel`** / **`up_frame world|travel`** — override the reference
  frame for the forward and up axes independently (each defaults to `frame`).

The **forward** axis (2 DOF — where the camera points) is, in precedence order:
`fwd_at <t> <x y z>` direction keyframes (piecewise-linear, `t ∈ [0,1]`), else
`look_at`/`look curve` aim, else the path tangent. The **up** axis (1 DOF — roll about
forward) is `up_at <t> <x y z>` vector keyframes, else `roll`/`roll_at` an angle
(degrees) about the reference up, else the reference up itself.

A `fwd_at`/`up_at` vector is interpreted **in the reference frame of its axis**: under
`travel` its components are `(x=right, y=up, z=forward)` in the RMF basis (so a constant
`up_at 0 0 1 0 … up_at 1 0 0 1` tips the camera from RMF-up toward RMF-forward relative
to the moving path); under `world` it is a plain world-space direction. Omitting all of
`fwd_at`/`up_at`/`frame`/`fwd_frame`/`up_frame` reproduces the legacy world-up behavior
byte-for-byte.

**Animated camera scalars.** `fov`, `roll`, `zoom`, `fstop` and `focus` may vary along
the flyby as a function of the normalized timeline `t ∈ [0,1]` (`t=0` at the first
frame, `t=1` at the last). Two mechanisms, both keyed on `t`:

- **Keyframe track — `<scalar>_at <t> <value>`** (repeatable): piecewise-linear
  interpolation between keyframes, e.g. `fov_at 0 60   fov_at 1 30` sweeps the field of
  view 60°→30° across the flyby. Also `roll_at`, `zoom_at`, `fstop_at`, `focus_at`.
- **Record track — `<scalar>_from RECORD.channel[(driver)]`**: drive the scalar from a
  parametric [`record`](#75-record--parametric-slot-luts) channel (records-as-keyframe-
  tracks). The channel is sampled at the driver, which defaults to the raw timeline `t`
  and may be any expression **in `t`** — e.g. `fov_from zoom.fov` (linear over `t`) or
  `fov_from zoom.fov(t*t)` (ease-in). The record's own `interp` (nearest/linear/smooth)
  controls the curve shape, so `smooth` gives eased motion for free. The record's stops
  must be constant (no per-hit surface variables), and the driver may reference **only**
  `t` — surface variables like `u`/`x`/`noise` are out of scope here and error.

Precedence per scalar: a `_from` record track wins over an `_at` keyframe track, which
wins over the authored base constant (`fov_y`, `roll`, etc.). A linear record track is
exactly equivalent to the matching linear `_at` keyframes — e.g. `fov_from zoom.fov`
with

```
zoom = range 0-1 [
    fov 60 30
    interp linear
]
```

renders frame-for-frame identical to `fov_at 0 60   fov_at 1 30`. These camera scalars
are baked into each frame at load time, so record camera tracks work in every render
mode (no GPU restriction).

---

## 16. `render` — defaults (CLI overrides)

```
render { photons 1000000  device gpu  mode R  out png/out.png  res 900 }
```

| key | meaning |
|---|---|
| `photons` | photon budget (forward modes) |
| `device` | `cpu` / `gpu` / `auto` |
| `mode` | global measurement mode (per-camera `mode` still wins for that camera) |
| `out` | output path (`.png` → `png/`, `.ppm` → `ppm/` per repo convention) |
| `res` | global resolution |

Any of these are overridden by the matching CLI flag.

---

## 17. Load order and validation

The loader runs in passes: scene units/spectral → spectra → textures → patterns →
materials (+ mix/layered resolve) → geometry/lights/medium/cameras/render →
`Scene::build()`. Notable hard errors: unknown block/material/type/preset, a scene
with **no light**, a `mix` with nested children or weights summing > 1, a `function`
isosurface without `contained_by`, an isosurface without exactly one root element, and
any unknown spectrum/preset/identifier.

---

## 18. Worked example

```
scene { units meters  spectral 360 830 1 }

material "white" { type diffuse reflect whitewall 0.75 }
material "gold"  { preset gold }
material "glass" { type dielectric ior 1.5 }

quad { origin -0.4 0 -0.4  u 1.8 0 0  v 0 0 2.4  material white }   # floor
# … remaining Cornell walls …

isosurface { material glass  sphere { center 0.77 0.17 0.70  radius 0.17 } }

light area { origin 0.22 1.199 0.22  u 0.56 0 0  v 0 0 0.56  normal 0 -1 0  spd preset:bb6500 }

camera "cam" {
    eye 0.8 0.62 2.35  look_at 0.8 0.28 0.55  up 0 1 0  fov_y 42
    mode R  film { res 900 600 }
}
```

See `scenes/` for complete examples (`showcase.ftsl`, `uv_native.ftsl`,
`showcase_orbit.ftsl`, `mirror_selfie.ftsl`, and many feature demos).
