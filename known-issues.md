# Known Issues & Technical Debt

Running log of unsolved bugs and accumulated tech debt. Fix items here as soon
as practical; this file is the fallback for what can't be addressed immediately.

## Open issues

### TECH-DEBT (2026-07-20): forward mode B barely converges the water-droplet rainbow (`scraps/rainbow_test.ftsl`); mode D works well
The Airy rainbow phase model (`src/rainbow.h`) is **verified correct** — `ftrace -rainbow-selftest`
reports textbook geometry (primary 40.7°–42.5° antisolar, secondary 50.1°–53.4° with reversed
colour order, water dispersion violet n=1.343 > red n=1.330, phase normalised 2π∫p dμ=1.000). But
rendering the bounded rain-curtain scene `scraps/rainbow_test.ftsl` (thin fog slab + distant sun,
camera toward the antisolar point):
- **mode B (forward pinhole splat):** after 450 M photons / 75 s the image is essentially **black**
  (`sensor=0.0000`, ~95 % noise, auto-exposure ~1.9e-10). The single-scatter-to-camera path over the
  scene's hundreds-of-metres scale is too rare for the forward splat to accumulate the bow.
- **mode D (BDPT):** converges the **full, clearly-coloured bow** in the same 75 s (~7.5 % noise) —
  primary + secondary + Alexander's dark band all visible (`png/rainbow_rain.png`). The fogbow
  variant (`scraps/fogbow_test.ftsl`, 10 µm droplets) likewise renders correctly as a broad,
  supernumerary-dominated pale bow (`png/fogbow.png`).
So this is not a physics bug — the machine and mode D are fine. It is a **forward-mode
efficiency gap**: mode B (and likely A/C) can't practically image a thin-medium single-scatter bow
at this scale. **Proper fix (deferred):** give the forward medium-scatter path a next-event /
camera-connection term for participating media (so each scatter vertex connects to the camera with a
proper importance weight, like BDPT does) instead of relying on the plain pinhole splat, or document
that rainbow/fog-bow scenes should use mode D. Low priority — mode D covers the use case.

### BUG (2026-07-19; root-caused 2026-07-20): `scenes/gallery_settled.ftsl` OOMs — but ONLY when the 600-frame flyby is in the camera selection
`error: bad allocation` (exit 1) rendering `scenes/gallery_settled.ftsl`.
**Root-caused 2026-07-20 — it is NOT a generic "scene too heavy" OOM.** With 26 GB
free on a 64-bit build, ~1.5 M scene tris and a 1280×720 film cannot exhaust
memory; the `bad_alloc` is a single *pathological* allocation that scales with the
**number of selected cameras**, and the scene defines a `camera_curve "fly"` of
**600 frames**. Reproduction matrix (all on this machine, after the 2026-07-20
Klein→gyroid-lite swap):
- `-camera cam` (single still)  → **works** in every path: mode D GPU render
  (1280×720, 5.4 s), CPU `-raster -seethrough` (1440×810, 5.08 M tessellated tris),
  and the interactive `-explore -seethrough` fly viewer.
- no `-camera` (selects `cam` + all 600 `fly` frames) → `bad_alloc` right after
  `[selftest]`.
- `-camera fly` (the 600-frame flyby alone) → `bad_alloc` right after
  `[camera] path 'fly' -> 600 frames`.
So the old "OOMs at default resolution" claim was misleading: the *still camera*
renders fine at default res; only pulling the whole flyby into one selection blows
up. The allocation is proportional to the frame count (each selected camera gets
its own `Film` accumulator — see `renderForwardShared`, `main.cpp` ~1240/5590, and
the "~3 GB of films" note at ~5845), so 600 simultaneous films (or a similar
per-camera structure built during camera expansion) overflow.
**Workaround (works today):** always pass a single camera, e.g.
`ftrace -in scenes/gallery_settled.ftsl -camera cam -explore -seethrough`, or
`-camera fly` **with** a frame selector so only a handful of frames are live at
once. **Proper fix (deferred):** chunk the multi-camera / flyby film allocation so
a long `camera_curve` renders in bounded-size batches (render K frames per photon
flight, recycle the film buffers) instead of allocating one film per frame up
front — mirror the "one-frame of host RAM" strategy already noted at ~5845 for the
budgeted path. Not blocking (the still camera renders the showcase fine); the
flyby just needs the batching fix before it can render all 600 frames in one
invocation on this machine.

### RESOLVED (2026-07-19): stale `absorb 3 0.5 0.3` example in `src/ftsl.h` — untagged spectrum triples don't parse
The comment above the `dielectric` `absorb` handling (`src/ftsl.h` ~1838) read
`e.g. `absorb 3 0.5 0.3` (per-channel, upsampled)`, implying a bare/untagged numeric triple is a valid
spectrum expression. It is **not**: `evalSpectrum` only accepts a *single* number as a constant;
a 3-word run with a numeric head (`3 0.5 0.3`) fell through every branch to
`fail("unrecognized spectrum expression '3'")`. Verified empirically — a scene with `absorb 3 0.5 0.3`
(and likewise `reflect 0.8 0.7 0.2`) errored out with `[ftsl] unrecognized spectrum expression '<head>'`.
The valid spellings are a scalar (`absorb 0.5`), a tagged colour (`absorb rgb 3 0.5 0.3`), or a named/ref
spectrum. **Fixed:** (1) corrected the comment to the tagged example plus an explicit "the `rgb` tag is
required" note; (2) added a targeted parser hint — when the head is numeric and every word in the run is a
number (`w.size() >= 3`), `evalSpectrum` now fails with `… — a bare numeric triple isn't a spectrum; tag it,
e.g. \`rgb 3 0.5 0.3\`` instead of the opaque `unrecognized spectrum expression '3'`. (VERSION 0.9.2 → 0.9.3.)

### RESOLVED (2026-07-19): loom's shared-grammar `reflect`/`roughness`/`*_map` fields are now shape-validated
The grammar-backed reader (`tools/loom/loom/grammar/reader.py`) shape-checks *purely-spectral* fields
(`ior`/`transmit`/`absorb`/`substrate_k`/`emit` on materials, `spd` on lights) against the shared spectrum
grammar (`loom/grammar/spectrum.py`). The binding-union fields `reflect`, `roughness`, and any `*_map` binder
are now validated too, via new `loom/grammar/bindings.py` — a faithful mirror of ftrace's `bindReflectTexture`
/ `bindScalarTexture` / `bindScalarPattern` + `spectrumParam` / `dblParam` (`src/ftsl.h` `buildMaterial`):
`as_color_binding` (`reflect` = `texture:<name>` | spectrum — reflect binds only a UV texture, never a
`pattern:`), `as_scalar_binding` (`roughness` = `pattern:` | `texture:` | one scalar number, not a spectrum),
and `as_map_binding` (`film_thickness_map`/`weight_map` = `pattern:` | `texture:` only). Wired into
`_build_material` (`_validate_bindings`); shape-only (a bound name's scene membership stays a later,
scene-aware check). An untagged triple `reflect 0.8 0.7 0.2` is now rejected exactly as ftrace rejects it.
`tests/test_grammar_bindings.py` (10 cases) + `test_grammar_material.py` additions; loom suite 823 passed.
**Remaining scope (not this item):** the record-driven *whole-material override* block (`from R(...)` +
`slot = REC.chan`, ftrace's `isRecordOverrideBlock`) is a distinct block form loom does not emit, and the
final step is mirroring the `value`/`spectrum`/binding grammar into ftrace's C++ front-end at the J3c port.

### RESOLVED (2026-07-19): loom `Light(color=…, size=…, turbidity=…)` emitted fields ftrace's `addLight` ignored
loom's `Light` accepted free-form props and emitted them verbatim (`light { kind …  color 0.9 0.8 0.7  size 2 2 }`),
but ftrace's `addLight` reads emission from `spd` (a spectrum) plus per-subtype geometry
(`origin`/`u`/`v`/`center`/`radius`/…) — it has **no** `color`, `size`, or `turbidity` field, so those loom props
were silently dropped at render. **Fixed:** loom's `Light` now maps `color=(r,g,b)` → `spd rgb r g b` (a spectral
emission, since ftrace lights are spectral) and otherwise passes only ftrace-valid props through; the test battery
dropped its `size`/`turbidity` fixtures and uses ftrace's real light schema (`u`/`v` area geometry,
`center`/`radius` sphere light). `size`/`turbidity` were demo-only and are gone — if loom wants to author a light
it uses ftrace's language. (`test_light_color_emits_spd_rgb` locks the mapping; full loom suite green, 824 passed.)
The analytic-sky *feature* that would give `turbidity` meaning is deferred — see TODO ("Analytic physical sky").

### RESOLVED (2026-07-19): `gallery_settled.ftsl` — several objects mis-positioned (resting on the floor)
**Root cause:** the free-settle physics bake tumble, not a transform/collider/floor bug. Three of the
five settled pieces landed correctly; klein (COM y≈0.37) and heart (y≈0.14) had tipped, rolled past
the rim of their NARROW museum pedestals, and fallen to the floor. Verified by recovering each piece's
settled COM height from the baked `group` delta (`pos_sim = R·c_auth + t`; see
`scraps/check_settle_heights.py`) — confirming the transform math and colliders were correct and only
the two tippy shapes had walked off their caps. **Fix:** added a during-sim horizontal restoring
spring to `tools/settle_scene.py` (`--tether [k]`, default k=150) applied AT each body's COM every
step, pulling it back toward its authored XZ. Because it acts at the COM it exerts **zero torque**, so
a piece is still free to tip/rotate onto its cap but cannot walk off it sideways — the result is a
genuine physics rest pose that stayed home. At k=150 all five pieces settle onto their stands in one
pass (klein rescued 0.37→1.38, heart 0.14→1.12). The committed `gallery_settled.ftsl` was repaired by
swapping the five tumbled `group` deltas for the tether-settled values. (Distinct from the separate
OPEN mode-D GPU-BDPT launch-failure on the same scene, logged 2026-07-15.)

### TECH DEBT (2026-07-19): `scenes/gallery_settled.ftsl` has diverged from its authored source `scenes/gallery.ftsl`
The generated settled file was hand-edited by three later commits (4c86261 prefer{}/else{} camera
mode-D/B fallback, f0786d6 flyby `frames 600`, bc9d664 scene-level `default_mode` + fps hint) that were
**never back-ported** to the authored `scenes/gallery.ftsl` (still plain `camera mode D`, `frames 144`).
So the "generated" file is now the de-facto source of truth for those features, and re-running
`tools/settle_scene.py --scene scenes/gallery.ftsl` would silently REGRESS them. That is why the
2026-07-19 floor-bug fix above patched the five `group` deltas in place instead of regenerating. **Proper
fix:** reconcile the two — back-port the camera/frames/default_mode authoring into `gallery.ftsl` so the
settled file is fully regeneratable from source, then regenerate with `--tether`. Blocked on confirming
authoring intent (is `frames 600` or `144` canonical? is the prefer/else wrapper wanted in source?).

### TECH DEBT (2026-07-19): settle sim slow — non-manifold stand colliders won't decimate below ~150–390k tris
`tools/settle_scene.py` decimates static concave colliders to `STATIC_TRI_CAP` (4000) via
`trimesh.simplify_quadric_decimation`, but the marching-cubes museum-stand meshes are multi-shell /
non-manifold (unions of boxes), so quadric edge-collapse gives up early and only reduces them ~65%
(e.g. stand meshes 500k–1M → 150k–390k tris), keeping per-step collision cost high. Clean closed meshes
(gyroid, lamp) hit 4000 fine. Worked around by validating at `--mesh-res 64`. **Proper fix candidates:**
(a) VHACD-decompose the static stands into convex compounds too (fast convex-vs-convex collision), or
(b) vertex-clustering / voxel-remesh decimation that ignores topology, or (c) approximate each stand
with authored box/cylinder primitive colliders instead of the polygonised isosurface.

### FEATURE REQUEST (2026-07-19): cache ftrace's per-scene preprocessing before rasterizing
Add an option to **cache the scene-derived data ftrace computes at load** (tessellation / BVH /
material+texture bake / whatever the rasterizer consumes) to disk, keyed on the scene (hash of the
`.ftsl` + referenced assets), so re-opening the same scene in the raster preview skips the rebuild.
Motivation: interactive raster/preview startup on a heavy scene repeats expensive preprocessing every
launch. Design notes: invalidate on any input change (scene text, mesh/texture/vdb asset mtimes,
relevant CLI flags that affect the baked data); store next to the scene or in a sidecar cache dir;
make it opt-in (`-scene-cache` or similar) at first. Overlaps conceptually with the `.ftbuf`
checkpoint sidecar but is about *input* preprocessing, not *output* accumulation.

### PERF (2026-07-19): preview rasterizer much slower than an equivalent WebGL renderer
Our software/CUDA preview rasterizer is far slower than a WebGL rendition of a comparable scene.
Needs profiling before any fix. Likely factors to measure: (a) we rasterize on the CPU in double
precision by default (GPU path is opt-in via `-device gpu|auto`), where WebGL is GPU float hardware
with fixed-function raster; (b) per-frame host readback + p99 auto-exposure on the host each frame
(known-issues "Per-frame readback"); (c) we re-tessellate / rebuild scene data per launch (see the
scene-cache request above); (d) no persistent GPU-resident vertex buffers / we may re-upload. **Next
step:** profile a representative scene (CPU vs `-raster-gpu`) to find the actual bottleneck rather
than guessing; compare against what a trivial WebGL draw of the same triangle count costs. This is a
"why is it slow" investigation, not a confirmed single bug.

### FEATURE REQUEST (2026-07-19): option for curve-editor curve to be occluded by geometry in front of it
The camera-path / curve overlay shown in the curve editor currently draws over everything (an
always-on-top helper). Add an **option** to instead depth-test the curve against the scene so objects
in front of it occlude it (more spatially truthful), while keeping the always-visible mode available
(often you *want* to see the whole path through geometry). Implementation: test the curve fragments'
depth against the opaque z-buffer the rasterizer already produces; expose as a toggle (CLI flag +/or
editor control). Low risk — the z-buffer is already there.

### DONE (2026-07-19): loom RBF scatter field rebuilt the interpolator every frame
`RbfScatterField` / `VecRbfScatterField` (`loom/interp.py`, `_RbfEngine`) used to rebuild the
`scipy.interpolate.RBFInterpolator` once per frame, gated only on the frame number, even when
the sample positions and values were completely static — pure waste for the very common case
of a fixed scatter field queried along a moving path over a long animation.
**Fixed** by replacing the bare per-frame gate with change detection: `_RbfEngine` now caches
the last-built interpolator together with the exact position/value arrays it was built from,
and on a frame advance re-evaluates the sample arrays and reuses the cached interpolator
verbatim (bit-for-bit identical output) unless the positions or values actually changed. A
static field now builds exactly once regardless of animation length; animated values still
rebuild per changed frame (correct, and cheap at realistic scatter sizes). Tests in
`tests/test_rbf.py` assert build-once for static fields (scalar + vector), bit-identical reuse
vs. a fresh rebuild, and correct rebuild-on-change for animated values.
Deliberately **not** done: the deeper "static positions, animated values → reuse the O(M³)
factorization, re-solve only the RHS" optimization. scipy exposes no public refactor-with-new-RHS
API, so it would require coupling to scipy's private compiled internals
(`_rbfinterp_np._build_system`) — a version-fragile dependency for a gain that is negligible at
the small scatter sizes loom fields realistically use. `neighbors=` (local k-NN RBF) still
mitigates cost for large point sets.

### DEFERRED (2026-07-18): loom VDB generator/wrapper — author sparse voxel grids from loom
**Status: intentionally not built (documented for later).** `loom.Volume` (added 2026-07-18)
can *reference* an existing NanoVDB grid via `density="vdb:<path>"`, but loom has no way to
*generate* a `.nvdb` from a Python field or to wrap OpenVDB's tooling.

- **What it would be.** A `loom` helper that bakes a loom `SpatialExpr` / `Grid` density field
  into a NanoVDB `FloatGrid` on disk (dense or sparse), so a procedural cloud authored in loom
  could ship as a real sparse asset instead of an inline `density "<expr>"`. Optionally a thin
  wrapper over OpenVDB's `nanovdb_convert` for `.vdb → .nvdb`.
- **Why we skip it now.** The procedural path already covers loom's sweet spot: an animated
  density formula emits straight into `medium { density "<expr>" }` and renders unbiased on CPU
  and GPU (validated by `scraps/_volume_test.py`). Baking to voxels only helps when a field is
  too costly to evaluate per-sample or must interop with external VDB assets — neither is a
  current need. The engine already imports `.nvdb` (`scraps/make_nvdb.cpp` makes test assets).
- **When to revisit.** When a loom scene needs a genuinely sparse, prebaked cloud (huge extent,
  expensive field, or sharing an asset with a DCC pipeline).

### DEFERRED (2026-07-18): `PatOp::MatMulAdd` — a fused matrix·vec+offset pattern opcode (future optimization)
**Status: intentionally not built. This is an optimization of an already-working path, not a
missing capability.** Why we may want it someday, and why we don't need it now:

- **The need it *seems* to fill is already met.** loom's N-D isosurfaces (`gyroid_nd`) rotate the
  lattice in higher dimensions by an affine map `A·(x,y,z) + c` fed into the field — and that matrix
  multiply **must** live inside the isosurface function ftrace evaluates. It already does: `_arg_expr()`
  (and `_pov_nd_coords`/`_pov_affine_coords`) in `tools/loom/examples/gyroid_nd.py` emit each matrix row
  as a plain scalar expression `(coeff*((a)*x+(b)*y+(c)*z)+phase)`. ftrace's pattern parser
  (`src/pattern.h`) compiles that straight into `Const/VarX/VarY/VarZ/Mul/Add` postfix bytecode and
  evaluates it directly on both CPU and GPU (the `-raster-gpu` iso preview ray-marches D=8 tumble gyroids
  today with the matrix baked in). So there is **no field ftrace currently fails to render** for lack of
  a matrix op.
- **What MatMulAdd would actually buy.** Purely a more compact *encoding*: one fused opcode replacing the
  ~6 scalar ops per emitted matrix row (`Const, VarX, Mul, VarY, Mul, Add, …`). Same math, bit-identical
  result — fewer `PatNode`s in the compiled program and a slightly cheaper inner eval. It is **not** a new
  capability.
- **Why we skip it now.** Per-frame pattern evaluation is not the bottleneck (the sin/cos/`PovFn` terms
  and the sphere-march dominate the field eval), and the postfix programs are well within any practical
  size. The payoff is marginal; the cleanest correct implementation still isn't free (see below).
- **When to revisit.** If a real workload ever makes pattern-eval node count or throughput a measured
  problem — e.g. very high-D fields with many coupling edges producing enormous postfix programs, or a
  profile showing the linear ops as a hot fraction of field eval.
- **How to build it (the design fork), if revisited.** The pattern VM is a single-scalar-stack machine:
  every `PatNode` is a POD `{PatOp op; double a;}` that pops N and pushes **exactly one** scalar. A
  matrix·vec+offset is 12 coefficients in → a **3-vector** out, which doesn't fit that contract. Two ways:
  - **Option A — single-output "matrow" + coefficient pool (preferred).** Add a `MatRow` op computing one
    output component `m0·x + m1·y + m2·z + off`; emit 3 per point. Preserves the one-push-per-node
    invariant and the POD/GPU-uploadable node, but needs `PatNode` to gain a side-array index (the single
    `double a` can't hold 4 coeffs). Contained; the five standard PatOp touch-points (opcode enum
    `pattern.h:29`, CPU eval switch `pattern.h:112`, device eval switch `render_cuda.cu` ~2754, the
    parser, and PatNode storage) plus the coefficient pool.
  - **Option B — true multi-output (invasive, not recommended for an ergonomics gain).** Give the stack
    vector semantics so a node can push 3 values. Cleanest for "transform a point," but rewrites the VM's
    fundamental one-scalar-out contract across CPU eval, device eval, arity accounting, and every consumer.

### ~~TECH DEBT (2026-07-18): `-raster-gpu` iso preview shades flat per-material albedo (no textures)~~ — FIXED 2026-07-19 (G5)
The GPU primary-ray isosurface preview (G2, `kIsoPreview` in `src/render_cuda.cu`,
wired as `-raster-gpu`) cast one ray/pixel with the shared `closestHit` and shaded
each hit with a **flat per-material solid colour** (`raster::materialColor` baked into
`matCol[]`), unlike the *triangle* GPU rasterizer — so a textured mesh previewed as flat
colour. **FIXED 2026-07-19 (TODO G5):** ported the CPU rasterizer's textured-preview path
(`Texture::sampleRgb`/`sampleRgbTriplanar`) into `kIsoPreview`. A shared flattened
linear-RGB texel array + per-texture `DPTex` meta + per-material `matTex`/`matTri` binding
(mirroring `raster.h` buildScene's rule exactly) are uploaded alongside `matCol`; the
kernel samples the hit `(u,v)` (or world triplanar) and replaces the flat albedo for
non-emitter hits. Covers **image** skins and, because E1 procedural (formula) skins bake
to `rgb` at load, **formula** skins too — one path. Flat (no-texture) hits are unchanged
(`matTex==-1` skips the sampler). Validated: `-raster-gpu` matches CPU `-raster` within
edge-coverage tolerance on `procskin.ftsl` (formula), `textured.ftsl` (image UV) and
`triplanar.ftsl` (mean channel diff ~0.03/255, ~1033 shared box-edge px); flat
`implicit.ftsl` unchanged. The device sampler is a private twin of `raster_cuda.cu`'s
(separate translation unit).

### TECH DEBT (2026-07-18): FBX import (C8) consumes geometry only — no materials/skinning/animation
The new FBX loader (`src/fbx_load.cpp`, vendored `ufbx`) imports **baked triangle
geometry + generated-if-missing normals + the first UV set**, applying every mesh
instance's `geometry_to_world`. It does **not** yet consume:
- **FBX materials** (Phong/Lambert/PBR) — every triangle takes the mesh block's
  FTSL `material`. glTF already maps `pbrMetallicRoughness`; FBX should get a similar
  material bridge (ufbx exposes `ufbx_material` + `pbr`/`fbx` property maps).
- **Skinning / blend shapes** — `ufbx_load_opts` could `evaluate` a bind/rest pose,
  but the loader currently reads the static mesh. A future path could bake a chosen
  animation time (ufbx supports `ufbx_evaluate_scene`).
- **Animation** — no per-frame FBX animation sampling (loom emits frames instead).
- **Multiple UV sets / per-face materials / vertex colors** — only `vertex_uv` set 0
  is read; `face_material` segmentation is ignored.
These are additive follow-ups; the geometry path is validated (`scenes/cube.fbx` →
8 verts / 12 tris, `scenes/fbxcube.ftsl`). The proper fix for materials is a
`ufbx_material` → spectral-BSDF mapping mirroring `gltf.h`'s material import.

### TECH DEBT (2026-07-17): GPU preview rasterizer — feature parity with the CPU rasterizer

The GPU preview rasterizer (`src/raster_cuda.{h,cu}`, wired into `main.cpp`'s
`-raster` block via the `rasterOne` dispatcher, gated on `-device gpu|auto`) now
accelerates **all camera projections** (rectilinear + fisheye/panoramic), **opaque +
textured (skinned)** previews, and **see-through (clear-glass)** compositing on the GPU.
It only falls back to the CPU rasterizer (`raster::renderFrame`) on a device failure
(the `renderFrame` returning empty). Remaining deferred work is limited to bit-exactness
and readback (below). History:

- **Fisheye / panoramic projections (M2). — DONE (2026-07-17).** `kProject` now branches
  rectilinear (`x/z` + near-plane Sutherland-Hodgman clip → ≤2 sub-tris) vs angular
  (the same `projRadius(projection, θ)/rEdge` map as `raster::projectVtx`, behind-camera
  reject-clip → 1 sub-tri) via `DCam.projection`/`DCam.rEdge`; `kRaster`/`kShade`/
  `exposeAndEncode` were untouched (projection-agnostic as predicted). Validated GPU vs
  CPU on `scenes/fisheye.ftsl` (fish camera: mean abs diff 0.015/255, 515/2.07 M edge
  pixels — tighter than the rectilinear `rect` frame's 0.034/1200), and rectilinear
  regression unchanged.
- **See-through (`-see-through`) on GPU. — DONE (2026-07-17).** `kProject` now propagates a
  per-triangle `clear` flag; `kRaster` skips clear surfaces when `seeThrough` (so only opaque
  geometry wins the visibility buffer); a new `kClear` device pass mirrors `fillTriangleClear`
  — it rasterizes each clear sub-triangle against the opaque `zbuf` written by `kShade`, and
  for every fragment IN FRONT multiplies that pixel's `clearT` (transmittance) and `milkT`
  (1 − per-surface milk, incl. the grazing/rim term) via a CAS-based `atomicMulF` (the product
  is commutative → order-independent, so races are safe). `renderFrame` resets `clearT`/`milkT`
  to 1 with `kFillF`, runs `kClear`, downloads both, and feeds the shared `exposeAndEncode`
  (which already composites them). Validated GPU vs CPU on `scenes/cornell.ftsl -see-through`:
  mean abs diff 0.020/255, 0.034 % edge pixels — same float-vs-double edge gap as opaque.
- **Image skins (textured `reflect texture:<name>` albedo) on GPU. — DONE (2026-07-17).** Each
  `Scene::textures` entry's linear-RGB buffer is flattened into one shared device texel array
  (`DTex` metadata: w/h/filter/wrap/offset) in `upload`; `DPTri`/`DSTri`/`DVtxCS` carry `uv`,
  `tex` and `triplanarScale` (with UV lerped through the near-plane clip); `kShade` samples via
  device twins `dSampleRgb`/`dSampleRgbTri` (nearest/bilinear, v-flip, wrap modes; triplanar
  |n|^4 axis blend) exactly mirroring the host `Texture::sampleRgb`/`sampleRgbTriplanar`.
  Indexed-palette textures sample their raw index-map RGB here, identical to the CPU preview's
  `sampleRgb` path. Validated GPU vs CPU on `scenes/textured.ftsl` (UV skin: mean 0.019/255,
  0.029 % edge) and `scenes/triplanar.ftsl` (world triplanar: mean 0.018/255, 0.028 % edge).
- **Parity is visual, not bit-exact.** The device geometry/shading is single precision
  vs the CPU's double, so silhouette-edge pixels can differ by one pixel of coverage
  (measured ~0.03 % of pixels on cornell/implicit, all on color boundaries, mean abs
  diff ~0.02/255). The exposure + tone-map tail IS shared host double code
  (`raster::exposeAndEncode`), so brightness/lock never drift. Acceptable for a preview;
  a full-FP64 device path would close the edge gap at a large speed cost (not worth it).
- **Per-frame readback.** Each frame downloads the HDR accum + z + emis buffers and runs
  the p99 + encode on the host. Fine today; if it ever bottlenecks, move the tone map
  onto the device and read back only RGB8 (would then need the anchor computed on-device
  or shared explicitly for the exposure-lock case).

### TECH DEBT (2026-07-17): GPU BDPT (mode D) can't texture — falls back to CPU BDPT

The forward (A/B/C) and backward-reference (R) GPU megakernels sample image skins per
hit via `dDiffuseRho` (device twin of `diffuseReflectance`), so a textured
`reflect texture:<name>` albedo renders on the GPU for those modes. The GPU BDPT kernel
(`kBdpt` in `src/render_cuda.cu`), however, drops the surface-local `(u,v)` from its
`DVertex` and its diffuse vertices sample only the constant `reflect` spectrum — so
`cudaBdptSupported` (render_cuda.cu ~5443, `usesTexOrFluoro`: `m.reflectTex >= 0`) rejects
any textured scene and `-mode D -device gpu` falls back to the (correct) CPU BDPT. Same
for fluorescence, roughness/film-thickness/pattern-driven params and mix masks. Verified:
`scenes/textured.ftsl -mode D -device gpu` prints "BDPT-GPU-unsupported feature … using
CPU" and matches the CPU BDPT image. Proper fix: thread the hit `(u,v)` through `DVertex`
+ upload the textures to the device (as the forward path already does), then have the
BDPT `dBsdfF`/`dBsdfPdf`/`dConnect` sample `dDiffuseRho` at the vertex instead of the
constant spectrum. Non-trivial (MIS pdfs must stay consistent), low priority (CPU BDPT is
correct; textured caustic-heavy scenes are rare).

### TECH DEBT (2026-07-17): loom preview server (`-serve`) is resident-process only

The M12 preview server (ftrace `-serve` in `src/main.cpp` `runServe`, loom
`loom/preview.py` `PreviewServer`) delivers only the *resident-process* win: it
keeps the process, live window, CUDA context, and spectral tables alive across
frames, re-rendering each `.ftsl` path streamed on stdin. What it does **not** yet
do (DESIGN.md §11.9 — the real interactivity speedup):

- **Per-frame delta push.** Loom bakes a whole new scene per frame; only a handful
  of constants actually change between adjacent frames. `-serve` still re-parses the
  full ftsl and rebuilds everything each frame. Proper fix: a delta protocol
  (loom sends only changed baked constants; ftrace patches them in place).
- **Static-geometry / BVH caching.** Geometry that doesn't move between frames is
  re-tessellated and its accel structure rebuilt every frame. Needs primitive
  identity + an incremental/cached BVH so unchanged geometry is reused.
- **Preview LOD.** No reduced-fidelity fast path (e.g. coarser isosurface fineness /
  fewer samples) distinct from the final render budget.

Also: the resident live window keeps the *first* frame's resolution for the whole
session (`liveWindowUpdate` / raster window create are guarded by `!g_liveWin`), so a
preview run must hold `-r` constant. Fine for a fixed-size scrub; revisit if
per-frame resolution changes are ever needed.

### DONE (2026-07-16): raster see-through for clear objects (`-see-through`)

Opt-in preview transparency for the `-raster` viewer, so clear materials read as
see-through instead of the old solid pale ghost — *without* refraction. Implemented
entirely in `raster.h` + a CLI flag in `main.cpp`:

- **Model.** Each clear surface (dielectric / thin-film / filter / diffuse-transmit,
  via `isClearPreviewType`) between the camera and the opaque background multiplies a
  per-pixel transmittance (`clarity`, default 0.85) and accumulates a milk product, so
  N crossed surfaces give `clarity^N` dimming + growing haze (a closed ball = 2
  crossings). A grazing-angle Fresnel-ish term adds silhouette milk so edges read.
  Composited in display-linear space in the tone-map pass:
  `c = c*clearT + milkColor*(1 - milkT)`.
- **Architecture.** `PTri`/`STri` gained a `clear` flag (set in `tessellate` from the
  material type). Opaque pass skips clear tris; a second **order-independent**
  band-parallel pass (`fillTriangleClear`) accumulates transmittance/milk against the
  finished opaque z-buffer — the product is commutative so no transparent depth sort is
  needed. Auto-exposure is still computed on the opaque-only accum (glass doesn't skew
  exposure).
- **CLI.** `-see-through` / `-seethrough` / `-glass` enable it; `-glass-clarity <0..1>`
  sets the per-surface transmittance (implies the flag). Wired into the main raster
  path and the interactive fly-viewer `renderFrame` calls (metering pass stays opaque).
- **Possible follow-ups (not done):** per-channel coloured transmittance for tinted
  glass/filters (currently neutral dimming); modelling thickness so a thin edge dims
  less than a thick centre (currently every crossed triangle counts equally).

### DONE (2026-07-16): camera_curve editor — all five phases + rough edges landed

The in-viewer `camera_curve` editor (main.cpp fly-viewer, Rec / +Pt / Ins / Del / Save)
landed as **Phase 1** (core point authoring + recording + live spline overlay + Save a
`camera_curve` block with a `look curve`). All planned follow-ons are now DONE:

- **Phase 2 — speed painting. DONE (2026-07-16).** Additive wheel brush modulates
  per-control-point speed (inverse density) in Paint mode; emitted as a `density_at`
  track and retimes live playback. Paint/Flat panel controls + speed readout.
- **Phase 3 — orientation painting. DONE (2026-07-16).** Mouse-look in Paint mode steers
  the nearest control points' `fwd`, reshaping the `look curve` (WYSIWYG in the overlay
  and saved block).
- **Phase 4 — rendered-sequence source. DONE (2026-07-16).** `ftrace -review <base>`
  plays a directory of rendered frames (`<base><digits>.<ext>`; reads png/jpg/bmp/tga/ppm)
  on the live window/timeline, scrub/Play, re-times via the Paint-mode speed brush, and
  Save writes a re-paced copy into `<dir>/retimed/` + an ffmpeg hint. `reviewMode()` in
  main.cpp.
- **Phase 5 — round-trip. DONE (2026-07-16).** Opening a scene with an existing
  `camera_curve` under `-explore`/`-fly` seeds the editor's `editPts` from that curve's
  control points (eye + look direction from look curve/look_at/tangent + per-point speed
  from the `density` track). Captured at load in `ftsl.h` (`AuthoredCurve` on `Loaded`,
  filled in `addCameraCurve`), consumed in `main.cpp`'s viewer. Save re-emits a revised
  curve. The loaded flyby still plays at full fidelity until the first edit.

Rough edges — all addressed (2026-07-16):
- **(a) look-spline bowing. FIXED.** Saved `look_point`s are now placed one MEAN control-
  point spacing ahead along each point's fwd (scene-relative, clamped), instead of a fixed
  1 world-unit. A larger, consistent offset keeps `(lookSample − eyeSample)` well away from
  the two splines' interpolation noise, so the aim spline stays smooth between sparse points.
  Direction at each control point is preserved exactly (any positive distance along the same
  fwd). In `saveCurveFn` (main.cpp).
- **(b) explicit point selection. FIXED.** A `selectedPoint()` helper drives both the red
  overlay highlight and Del. Locked to the path it follows the timeline (the control point
  nearest the scrub position — scrub to select), and in free flight it's the point nearest
  the eye. Also fixed a latent bug: Ins now finds its segment via `bracket()` (normalizes by
  the ACTUAL explorePath length) instead of `pathPos / kPreviewPerSeg`, which was wrong for a
  freshly-loaded curve whose frame count isn't a multiple of the preview sampling rate.
- **(c) multi-curve round-trip. FIXED.** The viewer records the flyby's base name (frame
  "beta00" → "beta") and the editor seeds from the authored curve whose name matches, so
  `-camera <name>` selects which curve is edited — not blindly the first. In main.cpp
  (`exploreCurveName` + the Phase-5 seeding block).

### OPEN: interactive raster fly-viewer can peg all cores / grow RAM when orphaned or on a heavy scene

The interactive fly-camera viewer loop (`main.cpp`, ~line 4037) only re-rasterizes
when `changed` is true and sleeps 15 ms when `!nav.any()`, so a *normal* idle viewer
is cheap. But an **orphaned** ftrace (e.g. a `-explore` test whose parent `timeout`
sent a signal the GUI process ignored, leaving it running with no console) was observed
pegging **all** CPU cores and climbing from ~2 GB to ~8 GB working set on
`scenes/gallery_settled.ftsl` — i.e. it was re-rendering flat-out (`changed` stuck true)
with no user input. Exact trigger unconfirmed (likely a teardown-state artifact where
`clientSize`/`drainNav` return values that keep flipping `changed`), and it wasn't
reproduced because doing so re-pegs the machine. Two things to consider as the proper fix:
(1) **bound the re-render rate** in the viewer loop (pace the loop to ~60 fps regardless
of `changed`/`nav.any()`), so even a stuck-`changed` runaway or a fast light scene can't
burn 100% of every core for no visible benefit; (2) make the loop exit on the same
signals as a normal render (so `-explore`/interactive processes die cleanly on SIGTERM,
not just window-close), and audit the resize-follow (`fitRes` vs `clientSize`) for any
size oscillation that would flip `changed` every iteration. Operational note: kill
interactive/`-window` test processes **explicitly** (PowerShell `Stop-Process -Force`) —
`timeout`-wrapping a GDI window app does not reliably terminate it.

### OPEN: rainbow phase — `SpecVtx::term` (through-glass-sphere fog connection) is HG-only

The new **rainbow droplet phase** (`rainbow.h`, tabulated Airy/Mie spectral phase) is
dispatched everywhere a medium's phase is evaluated **except one spot**: the specialised
"trace a photon *through a glass sphere* and connect its interior fog to the sensor" path
in `render.h` (`SpecVtx::term`, ~line 598) still calls `hgPhase(dot(wIn, wP), g)` directly
instead of `Medium::phaseValue(...)`. That code path predates the phase abstraction and
uses a flattened per-vertex `g` (no `mediumId`/λ), so it can't see a `RainbowPhase`. Impact
is tiny — it only affects fog that sits *inside* a refracting glass sphere viewed on the
special two-refraction connection — but a rainbow medium placed there would silently fall
back to the smooth HG lobe. **Proper fix:** give `SpecVtx` the owning `mediumId` (and thread
λ into `term`) so it can call `scene.media[mediumId].phaseValue(cosθ, λ)` like every other
site. Left HG-only for now to avoid reworking that specialised connector in the same change.

### DONE (2026-07-15): rainbow (water-droplet) phase — implemented, wired, and validated end-to-end

A medium can now scatter through a physically-tabulated **Airy water-droplet phase**
(`rainbow.h`) via FTSL `phase rainbow { .. }`, instead of the smooth Henyey-Greenstein lobe.
- **Physics core** (`rainbow.h`): Airy theory of the rainbow tabulated on a (λ×μ) grid with
  per-λ CDF importance sampling; normalised so `2π∫p dμ = 1` per λ. Self-test confirms exact
  Airy values, textbook Descartes angles, and unit normalisation.
- **Data model** (`scene.h`): `Medium::rainbowPhase` (shared_ptr, null for the common HG case
  → HG media stay bit-identical); `phaseValue`/`phaseSample` dispatch to it when set (overrides
  `g`). Symmetric phase → forward pdf == reverse pdf.
- **Wiring:** phase dispatch threaded through CPU forward (`render.h`), backward (`backward.h`),
  and BDPT (`bdpt.h` `phaseF`/`mediumScatterF`, `-dot(wo,wi)` scattering cosine). GPU volume
  path is HG-only, so `cudaForwardSupported`/`cudaBdptSupported` now **refuse rainbow media**
  and let the render fall back to the CPU (rather than silently dropping the bow to HG).
- **FTSL grammar** (`ftsl.h addMedium`): `phase hg` (default) / `phase rainbow { droplet_um,
  secondary, supernumerary, strength, forward_g, secondary_ratio }`. Features on by default.
- **Parser bug found & fixed:** `phase rainbow { .. }` — the subtype bareword `rainbow` before
  `{` is consumed by `parseValue` as the nested block's **`type`**, so `val.words` was empty and
  `kind` silently defaulted to HG (no bow, no error). Fixed by reading the kind from
  `ph->val.block->type` when `val.words` is empty. This was the reason early validation renders
  showed only a smooth veil despite correct physics.
- **Validated:** `scraps/rainbow_ring.ftsl` (centred-ring geometry, mode D, CPU) analysed with
  `scraps/radial_profile.py` shows the full signature — a **primary bow at ~42°** (violet inner /
  red outer), **Alexander's dark band** (~43–50°), and a **secondary bow at ~51–53°** with
  **reversed colours** (red inner / blue outer). Peak/median luminance ratio ~3.5× and climbing
  with samples.

Remaining rainbow tech debt: the `SpecVtx::term` glass-sphere-interior connector is still
HG-only (see the OPEN note above).

### DONE (2026-07-15): gradient-index (GRIN) media — Phase 2 wired through forward (CPU+GPU) + backward

**Phase 1 (landed earlier):** a `medium { ior "<expr over x y z r>" bounds { .. } }`
defines a **gradient-index region**: rays entering its bound bend continuously via a
symplectic Eikonal march (`d/ds(n·dr/ds)=∇n`) instead of travelling straight. Data model
(`Medium::ior`/`iorStep`, `nAt`/`gradNAt`/`insideBound` in `scene.h`), ftsl parsing
(`ior` / `ior_step` in `ftsl.h addMedium`), and the CPU backward marcher were done.

**Phase 2 (this commit):** the one canonical marcher now lives in **`grin.h`**
(`grin::sceneHasGrin` + `grin::march`, extracted verbatim from the backward tracer) and is
shared by **CPU backward** (`backward.h`, mode R), **CPU forward** (`render.h tracePhoton`,
modes A/B/C) and the **GPU forward megakernel + wavefront** (`render_cuda.cu` `dGrinMarch`,
`dMedInside`/`dMedNAt`/`dMedGradN`; `DMedium.ior`/`iorN`/`iorStep` uploaded; gated by
`DScene::hasGrin`). All bend rays identically; `ior`-free scenes stay bit-identical (the
march is only entered when `sceneHasGrin`). Validated: `scraps/_grin_lens.ftsl` warps a
checker in mode R; `scraps/_grin_caustic.ftsl` (a converging GRIN sphere over a floor)
shows the same lens redistribution in CPU mode B **and** GPU mode B, and a smooth
unperturbed pool in mode R (backward's straight NEE shadow ray can't bend — see below).

**BDPT (mode D) deliberately REFUSES GRIN.** BDPT's connection geometric term, area-measure
pdf conversion and MIS weights all assume STRAIGHT connecting segments, so a bent path would
bias the estimator. `main.cpp bdptUnsupportedFeature()` returns a GRIN message (mode D errors
out with "use mode A/B/C or R"), and `cudaBdptSupported()` rejects GRIN as defense-in-depth.
GPU backward (mode R) already falls back to the CPU for *any* medium (`cudaBackwardSupported`
rejects `anyMedium()`), so GRIN mode R runs on the GRIN-aware CPU backward tracer — correct.

Remaining GRIN tech debt is tracked as its own OPEN entry below.

### OPEN: GRIN tech debt (Phase-1 semantics carried forward — not regressions)

Follow-on work for the GRIN Phase-2 wiring above. None of these are bugs in the shipped
behavior; they are known limits of the current bend-only model.

1. **Only PRIMARY rays bend; connection/NEE rays are still straight.** Each tracer bends only
   its primary ray — the backward camera ray and the forward photon path. Its *connection/NEE
   ray* is straight: mode R's shadow ray to a light can't bend through a GRIN region (so it
   misses GRIN caustics — that's why `_grin_caustic` is dark in R), and the forward
   camera-splat (modes A/B) is straight (so imaging a surface *through* a GRIN lens via splat
   doesn't warp — use mode R for "camera looks through a GRIN lens", forward for "GRIN caustic
   onto a surface viewed directly"). Mode C (forward-catch) is fully unbiased but
   sample-starved. Curved-path connections (bending the shadow/splat ray) are a future
   enhancement.
2. **Exterior IOR inside a GRIN region is hard-coded to 1.0.** At a dielectric interface
   *inside* a GRIN region the exterior IOR should be `nAt(hit)` not 1.0 (the current code
   assumes GRIN regions sit in open air). **Directly relevant to the pending xenon-lamp work:**
   its bulb is a nested dielectric whose interior gas index differs from air, so a GRIN
   gradient over that same region would need the correct surrounding index at each interface.
3. **Absorbing/scattering GRIN not integrated along the curve.** A GRIN medium that is *also*
   absorbing/scattering isn't integrated along the curved path (treated as clear — the classic
   use). Fixed-step RK1 (`iorStep`, default bound/64) suits smooth fields; steep gradients may
   want RK4 / adaptive stepping.

**GRIN in BDPT (mode D) — deferred, may implement someday.** Mode D refuses GRIN today (see
the DONE entry above: its connection G-term, area-measure pdf conversion and MIS weights all
assume straight edges). Two tiers exist if we revisit it: (1) **cheap** — let the camera/light
subpaths bend on their PRIMARY march (they already can elsewhere) and pdf-consistently *skip*
any connection whose straight edge crosses a GRIN region; stays unbiased, only under-samples
pure-GRIN-caustic paths (negligible for a weak gradient like the lamp gas), and would let mode
D literally accept a GRIN scene. (2) **research-grade** — true curved connections (solve the
two-point boundary-value problem for the connecting geodesic, generalized geometric/Jacobian
term, consistent MIS); expensive and numerically nasty near a focus, poor ROI. Keep this on
the radar; neither is built.

**Showcase (`scenes/gallery_settled.ftsl`) render-mode note.** The hero stays in **mode D**
for now (its documented command). If we want the xenon lamp to carry a GRIN gas gradient *and*
have the whole hero render, the pragmatic route is to render the showcase in **mode B**
(forward) instead — B supports GRIN and captures *all* the effects — accepting that it's
**slower to converge** than D on this mixed scene. Tracked so we remember the trade-off:
mode D now (fast, no lamp GRIN) vs. mode B later (slow, full effects incl. lamp GRIN), unless/
until tier-1 "GRIN in mode D" above is built. The showcase now encodes this trade-off directly:
its still camera is wrapped in `prefer { mode D } else { mode B }`, so mode D wins today and the
loader auto-falls back to mode B the day a mode-D-hostile feature (a GRIN lamp-gas field) is added.

**Showcase idea: a fog rainbow that MOVES with the camera (`phase rainbow`).** Now that the
water-droplet phase is wired end-to-end (FTSL `phase rainbow { .. }`, validated: primary +
secondary bows, Alexander's dark band, reversed secondary colours — see the DONE note below),
we could dress the showcase (or a dedicated flyby) with a thin rain curtain that produces a
real rainbow. The compelling part is that **the bow is centred on the antisolar point (the
anti-sun direction), not on any object** — so as a moving camera pans/dollies, the bow slides
across the frame and *follows the view* exactly the way a real rainbow "runs away" from you.
That motion is the giveaway that it's genuine scattering physics, not a painted arc. Recipe
to keep on the radar for a flyby (own subdir `png/<set>/` per the flyby rule):
- Distant sun **behind** the camera's general travel direction (parallel rays → sharp bow);
  keep the sun *outside* the fog slab so it isn't extincted before lighting the drops.
- A **bounded, thin** rain curtain in front (`sigma_t`~0.001–0.002, `albedo` ~0.99, optical
  depth ≲ 0.3) so single scattering — which carries the bow — dominates the multiply-scattered
  veil. `phase rainbow { droplet_um 500 secondary on supernumerary on }`.
- A `camera_curve` that pans the antisolar point across frame (e.g. yaw the look direction, or
  translate laterally) so the ring visibly tracks the camera. Render **mode B** (forward) or
  **mode D** (BDPT, bounded fog) on the **CPU** — the GPU volume path is HG-only and auto-falls
  back to CPU for rainbow media, so a big flyby is CPU-bound (budget accordingly).
- Validated seed scenes to crib geometry/params from: `scraps/rainbow_ring.ftsl` (centred
  ring, mode D) and `scraps/rainbow_test.ftsl` (antisolar-aimed slab).

**Minor tech debt: `prefer{}/else{}` trial builds reload meshes.** Resolving a `prefer` node
trial-builds each candidate branch to test renderability (`ftsl::load`, `tryBuild` lambda). The
common **single-node** case is optimized — the accepted trial's `Loaded` is reused as the final
scene, so meshes load exactly once (verified on `gallery_settled.ftsl`). But when a node has
several branches that get *rejected* before one is accepted, each rejected trial still re-parses
and RE-LOADS every mesh; and a **multi-node** `prefer` does one extra final rebuild on top of the
per-node trials. For a heavy scene (600k-tri OBJs) that multiplies OBJ-load time. Proper fix if it
ever bites: build the shared/non-`prefer` blocks (all the meshes) *once* and only re-resolve the
small mode-sensitive delta per branch, instead of flattening + full-building the whole block list
each trial. Negligible today (branches carry only a camera + medium), so left as-is.

### DONE (2026-07-15): `exposure_lock` selector meter pre-pass now covers every render mode

Previously the real-render `exposure_lock <selector>` meter pre-pass only metered the
forward models (A/B/C) and the backward reference (R); modes D/M/P silently fell back to
locking on whichever frame rendered first, *ignoring the selector*. Fixed: the meter
pre-pass (`meterAnchor` lambda in main.cpp, the `meterPlan` loop just after the `-raster`
block) now renders the selector-chosen viewpoint in its **own** mode — `renderBdpt` for
D, a lazily-built shared reduced photon map + `renderPhotonCamera` for M, and
`classifyComposite`+forward+backward+`compositeFromFilms` for P — and any other mode
(S/U/V/…) falls back to a **general forward mode-B light-trace** (still a correct
scene-brightness anchor, never an arbitrary frame). Because the p99 anchor is a property
of the radiance, not the integrator, every mode yields a consistent anchor. There is now
**no silent frame-0 fallback anywhere**; a bare `exposure_lock` also defaults to the path
**average** rather than the first frame. Validated on scraps/lock_test.ftsl in modes
B/M/D/P with `index`/`average` selectors (all honoured, all frames flicker-free).

### DONE (2026-07-15): A/B/C now agree in absolute brightness at equal `power` (finite-lens catch modes were near-black)

Previously, `ABS_EXPOSURE_GAIN` (main.cpp ~927, value `6.0`) was calibrated **only for
mode B** (the pinhole splat). The finite-lens catch modes **A** (`connectLens`) and **C**
(forward pupil catch) produced a film scale ~10^3–10^4× dimmer at the same gain, so an
absolute scene shot in mode A/C came out essentially **black**. Root cause: mode B's
`connect()` records **radiance** (i.e. it is implicitly divided by the pixel solid angle
Ω_pix = `pixelPlaneArea()·cosCam³`), whereas the A/C splat records **flux-per-cell**
(`× R²/dist²`, with no cell-area normalisation) — a dimensionally different quantity.

**Fix (the F = 1/A_cell factor).** Fold a single per-camera constant
`F = 1/(pixelPlaneArea()·filmDist²) = 1/A_cell` into the A/C splat, converting its
flux deposit into film irradiance on the same scale mode B uses. Derivation: the raw A/C
splat = `B·(π R²·A_pix)`; the target (camera equation) = `B·(π R²/filmDist²)`; on-axis
`A_pix = pixelPlaneArea()`, so the correcting ratio is exactly `1/A_cell`. Because it is a
per-camera constant it cancels under p99 auto-exposure (auto-exposed scenes stay
byte-identical) and only re-seats the *absolute* level. Applied at four sites, CPU + GPU:
- `render.h` `connectLens` and `connectLensVolume` (`contrib *= 1/(pixelPlaneArea()·filmDist²)`),
- `render.h` mode-C catch (`cCell` factor on the film `add`),
- `render_cuda.cu` `connectLens`, `connectLensVolume`, and CAM_C catch (same factor).

Modes A/C use plain gain 6 (`comp = 1`); mode B keeps its aperture `camEq = (π/4)/N²` fold
(the separate DONE issue below) — the two paths each carry the `1/N²` once, never doubled.

**Validated** (`scraps/abs_calib.ftsl` / `_acalib_wide.ftsl`, Cornell box + 100 W area
light), tone-mapped 8-bit whole-image mean, GPU **and** CPU:
- f/4:   A `6.79` vs B `7.16` (95%).
- f/1.4: GPU A `25.54` vs B `25.90` (**98.6%**); CPU A `25.49` vs B `25.86` (**98.6%**).
- Mode C converges to B as its (pupil-catch) noise falls — mean `16.81 → 20.70 → 21.54`
  as noise `82% → 35.8% → 31.6%` (residual gap is tone-map clamp bias on fireflies, not a
  scale error). A and C now land at the same absolute brightness as B instead of black.

NOTE: the `-raster` preview still uses its relative reference aperture (Rref=0.02) for the
aperture-brightness *ratio*; that remains correct. Known wart (minor, logged for later):
the **`-aperture` CLI override** changes A/C's physical pupil `R` but does **not** feed
mode B's `camEq` comp (which reads the scene's `fstop`/`lens`), so cross-mode comparison
via `-aperture` desyncs B — always set `fstop` in the scene for an apples-to-apples A/B/C
comparison. Not a render-correctness bug (a normal single-mode render is unaffected).

### DONE (2026-07-15): absolute EV — mode B now applies the aperture's exposure (light-gathering)

**Fixed** in `src/main.cpp` (~3380, the ftsl per-camera render setup): when a scene
is in absolute EV *and* a physical aperture was actually authored (`fstop`/`lens`,
detected by `c.lensF > 0`), a mode-B render now folds the camera-equation aperture
term `camEq = (π/4)/N²` (with `N = c.lensF/(2·c.apertureR)`) into the exposure comp.
This adds only the **radiometric** light-gathering factor — the pinhole keeps zero
DoF. Gated tightly so it only *darkens* a mode-B camera that opted into an f-number;
with no aperture authored (`c.lensF == 0`, e.g. shipped `scenes/absolute.ftsl`) the
branch is skipped and the pinhole stays the pure radiance reference (comp unchanged,
byte-identical). Modes A/C are untouched (they must NOT double-apply `1/N²` — their
gross-scale mis-seat is the separate issue above, now DONE via `F = 1/A_cell`).

**Validated** (`scraps/abs_calib.ftsl`, Cornell box + 100 W area light, mode B, GPU):
rendering the same scene at f/2 vs f/8 now separates by exactly 4 stops —
patch linear-luminance mean `2.2225e-2` (f/2) vs `1.3919e-3` (f/8), ratio **15.97×**
(≈ 16× = 4 stops); whole-image mean ratio 15.99×. The startup log shows the comp
scaling correctly: `exposure=1.18 (absolute: gain 6 x 0.196 comp)` for f/2, where
`0.196 = (π/4)/2²`. Author `fstop`/`lens` at the **camera-block** level (not inside
`film{}` — the film block only reads res/size/format/iso/shutter/exposure).

The other half of the "unification" (making A/B/C agree in *absolute* brightness at
equal power) is now also DONE — see the "A/B/C now agree in absolute brightness"
issue above (the `F = 1/A_cell` A/C re-seat). Mode B is internally correct wrt
aperture, and A/B/C now match within noise at equal `power`.

The old rationale for excluding aperture from the exposure comp ("in splat mode B
the aperture is virtual, so an f-number term would double-count / be an artifact",
CamSpec/main.cpp ~921) held **only for modes A/C**, which already carry the physical
`R²` in their splat weight (render.h `connectLens`). Mode B has **no** `R²`, so a
virtual-aperture exposure term there is clean and non-redundant — hence the fix above
adds it in B only. When the A/C gross-scale gain is eventually re-seated (issue above),
keep the two paths consistent: A/C get `1/N²` from the pupil-area `R²` splat weight
(fix the gain, don't add `camEq`), B gets it as the pure exposure factor added here —
do NOT double-apply. Ideal end state is one camera-equation absolute model (add
`cos⁴θ` natural vignetting too) that makes A, B, C agree at equal `power`.

### OPEN (2026-07-15): mode D (GPU BDPT) — data-dependent "unspecified launch failure" on gallery_settled.ftsl

Rendering `scenes/gallery_settled.ftsl` in **mode D on the GPU** crashes with
`[cuda] bdpt kernel failed: unspecified launch failure` reproducibly at **spp 14**
(~232 s in; earlier spp complete fine and write correct images). It is an illegal
memory access inside the BDPT megakernel (`kBdpt`, `render_cuda.cu` ~4283), NOT a TDR
timeout (chunks are ~0.15 s) and NOT GPU contention (single process).

**Ruled out by inspection:** the per-thread `eye[]`/`light[]` subpath arrays
(`BDPT_MAXV=11`), the `DMediumStack` (CAP 8, push guarded), the media free-flight
loops, and the `double st[64]` pattern/field VM stacks are all bounds-safe. The fault
is **data-dependent** (RNG seeded by the global sample index `gidx`, so it reproduces
deterministically regardless of timing) — some specific path at spp 14 indexes out of
bounds or dereferences a bad pointer, likely a rare geometric/CSG/medium configuration
hit only by that sample's random walk.

**Investigation status (2026-07-15 update):** a **bounded** `compute-sanitizer --tool
memcheck` run — mode-D BDPT pinned to the single still camera (`-camera cam -spp 6`, so
it terminates instead of rolling onto the 144-frame flyby) — completed with **ZERO
memory errors**. So the fault does **not** reproduce on the still frame at low spp; it is
either **flyby-camera-position specific** (a geometric configuration only some moving-cam
viewpoint hits) or was already mitigated by unrelated fixes since the crash was first
seen. The earlier unbounded attempt (`-noise 3`, no `-camera`) never reached spp 14
(memcheck ~65× slowdown) and also rolled onto the flyby — avoid that; always bound it.
**Next step:** reproduce at the *specific* crashing spp/camera (drive to spp 14 on the
flyby camera under a bounded memcheck) to catch the exact `render_cuda.cu:<line>`, then
fix the OOB. Earlier context: build has `-lineinfo`, so memcheck reports the exact line.
Repro (headless — sanitizer runs instrumented). Use the real `compute-sanitizer.exe`
(in the CUDA `compute-sanitizer/` subdir), NOT the `bin/compute-sanitizer.bat` wrapper —
the `.bat` exits 127 (no useful output) when launched from the bash tool:
`"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/compute-sanitizer/compute-sanitizer.exe" --tool memcheck --log-file scraps/_sanit.log build_cuda2/bin/ftrace.exe -in scenes/gallery_settled.ftsl -mode D -device gpu -noise 3 -o png/_sanit.png`
Mode B on the same scene is stable, and the new `-raster` preview is unaffected.

### DONE (2026-07-15): Forward modes now smooth-shade interpolated normals — Veach adjoint correction applied

A smooth-shaded mesh (authored `vn` **or** crease-smoothed via `mesh { smooth }`) used
to render smooth in the backward reference mode R but **faceted in the forward modes**.
Root cause was the **shading-normal adjoint asymmetry** (Veach §5.3): a backward
estimator integrates the incident cosine through the *shading* normal, so interpolated
normals smooth the shading for free; a forward/particle tracer deposits irradiance per
**geometric** area and stays faceted.

**Fix shipped.** `shadingAdjointCorr(wi, wo, ns, ng)` in `geometry.h`
(`corr = |cos(wi,Ns)·cos(wo,Ng)| / |cos(wi,Ng)·cos(wo,Ns)|`, guarded denom, exactly 1
when Ns==Ng) is multiplied into the **particle throughput** at every non-specular
continuation and every camera connection of the LIGHT/importance subpath:
- `render.h` modes A/B/C — `connect`/`connectLens` splats + Diffuse/Fluorescent/
  DiffuseTransmit continuations (shared `tracePhoton` walk, so the M/S deposit inherits it).
- `bdpt.h` mode D — `randomWalk` (gated `mode == Importance`) + `connectBDPT` light-side `f`.
- `vcm.h` mode U — light-subpath continuation + light-image splat + the VC connection's
  light-vertex `fLit`. The eye/Radiance side is never corrected (it smooth-shades for free).
- GPU twins in `render_cuda.cu` — `dShadingAdjointCorr` threaded through `connect`/
  `connectLens`/`splatSurfaceAll` (forward tracer) and `dRandomWalk`(importance flag)/
  `dConnectBDPT` (GPU BDPT).

Validated smooth against mode R on `scraps/_smooth_on.ftsl` and the harsher
`scraps/_iso_sphere.ftsl` (directional-lit sphere, no indirect): modes B, D (CPU+GPU),
and U all match the mode-R gradient. Exactly 1 when Ns==Ng ⇒ every flat/analytic scene is
bit-identical (whole existing validation suite untouched).

**Surprising finding — photon-density gathers (modes M/S) did NOT need a gather-side
correction.** An initial hypothesis added a `cos_s/cos_g` gather reweight to the mode-M
photon-map and mode-S SPPM density estimates. Empirically this was WRONG: mode M/S already
smooth-shade (verified matching mode R by center-column brightness profile on both the
Cornell and isolated directional scenes, at gather radii from 0.004 to 0.017), and adding
the reweight *introduced* facet banding. That speculative correction was reverted; M/S
gathers are left uncorrected. See the tech-debt note below for the one place a gather-side
correction WAS needed (VCM's vertex-merge) and why the asymmetry isn't fully understood.

### TECH DEBT: VCM vertex-merge needs a gather-side shading-normal reweight that M/S don't — asymmetry not fully understood 2026-07-15

Direct consequence of the DONE entry above. The mode-M photon-map and mode-S SPPM
photon-density gathers **smooth-shade correctly with no gather-side correction** (verified
against mode R). But mode U's **VCM vertex-merge (VM) strategy** genuinely *facets* on the
same smooth meshes — strong full faceting on `scraps/_iso_sphere.ftsl` — even though its VM
density estimate is mathematically the same kind of radius gather. The current fix is a
**scoped, file-local `vmGatherCorr(wp, ns, ng) = |cos(wp,Ns)| / |cos(wp,Ng)|`** in `vcm.h`,
multiplied into the merge's camera-side `fCam` only (the `vmNorm` carries no geometric
cosine, so the raw VM value is as smooth as mode M — the faceting instead enters through the
per-facet MIS coupling between the VM and VC strategies).

**Why the asymmetry exists is not fully understood.** Best current hypothesis: the VM/VC
MIS weights assume normal consistency between the merged light-vertex and camera-vertex
BSDF evaluations, and the shading/geometric-normal mismatch breaks that assumption for VM in
a way the standalone M/S estimators (no competing MIS strategy) never see. The **cleaner
future fix** is to make the VM↔VC MIS-weight derivation consistent under interpolated
normals (so no ad-hoc `fCam` reweight is needed), rather than patching `fCam`. The current
`vmGatherCorr` is a no-op on flat tris / analytic spheres (`Ns==Ng` ⇒ ratio 1), so it
cannot regress any existing (non-smooth) scene. Low priority — mode U smooth-shades
correctly today; this is about *why* and a tidier derivation.

### DONE (2026-07-15): Shading-normal geometric-hemisphere clamp propagated to all connection sites (+ fixed a pre-existing GPU mode-R light leak)

The geometric-hemisphere clamp (stop a smoothed shading normal from leaking light in
through the geometric back face; `orientedGeoN()` in `geometry.h`) was previously only in
the backward reference (`backward.h` `neeLight`/`neeEnv`) and the forward tracer's camera
connection (`render.h` `connect`/`connectLens`). It is now applied at **every** NEE /
connection site: `bdpt.h` (mode D — the t==1 splat, s==1 NEE, and interior connection, each
endpoint), `vcm.h` (mode U — light-image splat, NEE, and both VC-connection endpoints), and
the GPU twins in `render_cuda.cu` (`connect`/`connectLens` for A/B/C, `dConnectBDPT` for
mode D, and `bkNeeLight` for backward). **Per-site recipe applied:** alongside the existing
`dot(Ns, wi) <= 0` shading-side test, also require `dot(ngo, wi) > 0` (ngo = geo normal
oriented to Ns) and offset the shadow/connection ray along `ngo`. In `bdpt.h`/`vcm.h` the
clamp is **guarded by `!isTwoSidedMat`** so transmissive (glass) connections through the
back hemisphere are not wrongly killed; GPU BDPT is reflect-only (v1) so the clamp is
unconditional there. No-op for flat tris / analytic spheres (`ngo == Ns`), so every
non-smooth scene is bit-identical.

**Modes M/S need nothing:** mode M's direct lighting reuses `bw.neeLight` (already clamped)
and its photon gather has no shadow ray; SPPM's visible-point walk does no NEE at all.

**Pre-existing bug fixed as a side effect.** The GPU backward NEE (`bkNeeLight`) was missing
the clamp its CPU twin (`backward.h neeLight`) already had, so **GPU mode R silently leaked
light through geometric back faces** — on `scraps/_iso_sphere.ftsl` GPU-R showed a smooth
(leaking) terminator while CPU-R showed the correct leak-free (faceted) one. Adding the
clamp to `bkNeeLight` makes CPU-R and GPU-R identical.

**Caveat — shadow terminator (NOW SOFTENED, see DONE entry below).** A *hard* geometric
clamp reveals the shadow-terminator problem: on a low-poly smooth-normal sphere under grazing
light the terminator shows the underlying facets (hard dark slivers) rather than a smooth
gradient. This was the mode-R reference's existing behavior too, so the clamp made the forward
modes *consistent with the reference*. The hard cutoff has since been replaced everywhere by
Chiang et al. 2019 softening — see the next DONE entry.

### DONE (2026-07-15): Shadow-terminator softening (Chiang et al. 2019) replaces the hard geometric clamp everywhere

The hard geometric-hemisphere cutoff from the entry above carved dark facet slivers at the
terminator of low-poly smooth-normal meshes under grazing light (the classic shadow-terminator
artifact). Replaced the hard `dot(ngo, wi) <= 0 ? reject` at **every** clamp site with a smooth
ramp: `shadowTerminatorG(wi, ns, ng)` in `geometry.h` (Chiang, Li, Burley & Hovhannisyan 2019,
"Taming the Shadow Terminator"; same cubic as Cycles' `bump_shadowing_term`).

    g = cos(Ng,wi) / (cos(Ns,wi)·cos(Ng,Ns)),  softened by  -g³ + g² + g  on (0,1)

Returns a `[0,1]` factor multiplied into the surface response (NEE/connection contrib): still
**exactly 0** when `wi` is behind the true geometry (no back-face leak — leak-free is preserved),
but ramps up smoothly off the geometric horizon instead of a step. Applied **uniformly to all
modes including the mode-R reference** so R softens too and every mode stays mutually consistent:
- `backward.h` mode R — `neeLight` (spot + sphere-cone + area/quad sites) and `neeEnv`.
- `render.h` modes A/B/C — `connect` and `connectLens` camera splats.
- `bdpt.h` mode D — t==1 splat, s==1 NEE, and both interior-connection endpoints (`!isTwoSidedMat` guarded).
- `vcm.h` mode U — light-image splat, NEE, and both VC-connection endpoints (`!isTwoSidedMat` guarded).
- GPU twins in `render_cuda.cu` — `dShadowTerminatorG` threaded through `connect`/`connectLens`,
  `dConnectBDPT` (3 subsites), and `bkNeeLight`.

**Bit-identity guard.** `shadowTerminatorG` short-circuits to a plain leak-free step (return
exactly 1.0 when in front of the geometry, 0.0 behind — identical to the old hard clamp) whenever
`dot(Ng,Ns) >= 1 - 1e-7`, i.e. the shading and geometric normals coincide (flat tris, analytic
spheres). Without this guard a re-normalized `ns` differs from `ng` in the last bit, the cubic
returns ~1 (not bit-exactly 1), and every flat/analytic scene would drift by ~1e-7. With it, the
softening engages **only** once `ns` and `ng` genuinely diverge (a real smooth/crease-smoothed
mesh), so the whole flat-scene validation suite stays bit-identical.

Validated on `scraps/_iso_sphere.ftsl` (grazing directional-lit low-poly smooth sphere): the
terminator is now a smooth gradient (no facet slivers) and mutually consistent across CPU R/D/U
and GPU R/D; flat `scenes/cornell.ftsl` renders unchanged.

**Diagnostic note — disabling the clamp entirely (the "option 2" that was considered).** A debug
toggle to *fully disable* the geometric-hemisphere clamp (reverting to a pure shading-normal test,
which leaks light through geometric back faces but never facets) was considered and **deliberately
not implemented**: softening is the correct fix, so a disable toggle is only ever a diagnostic, and
plumbing a runtime flag through the CUDA kernels (device-constant, kernel signatures, host parsing)
isn't worth the surface area for a debug-only path. If ever needed to isolate a back-face-leak vs.
terminator issue, edit `shadowTerminatorG` (and `dShadowTerminatorG`) to `return dot(ng,wi) > 0 ? 1
: 1` (always 1 → no clamp, no softening) or `return 1` unconditionally, rebuild, and compare — a
one-line local change, no scene/CLI plumbing.

### Headless-spawned `-window` render on gallery.ftsl hangs with no output — NEEDS INVESTIGATION 2026-07-14

A `ftrace -in scenes/gallery.ftsl -mode R -n 1500000 -spp 8 -window` invocation *spawned
as a background process without an interactive window station* ran for 15+ min burning
~7 CPU cores (3400+ CPU-seconds) at **1% GPU**, producing **zero stdout and no image /
checkpoint**, and never exited. Meanwhile the same scene stripped to room+lights (no
isosurfaces/fog), rendered with `-mode R -device gpu -spp 64 -preview` (NO `-window`, no
`-n`), built and rendered in seconds on the GPU concurrently. Two suspects, not yet
isolated: (a) `-window` can't create its Win32 GDI window when the process has no window
station (background/detached spawn) and the code spins instead of erroring; (b) passing
`-n <photons>` to **mode R** (which is spp-based, not photon-count) mis-budgets into a
huge CPU loop. Likely (a). **Repro to confirm:** run the heavy scene once with `-window`
+ no `-n`, and once with `-n` + no `-window`, from a detached shell. If (a): make the
`-window` init detect a missing/invalid window station and fall back to `-preview`
(or error cleanly) instead of hanging. NB: do NOT `taskkill /F` a live ftrace — the
nvlddmkm teardown BSOD (below) has fired after an abrupt kill.

### `look tangent` pitches hard at path folds (gallery fly cusps) — FIXED 2026-07-14

`camera_curve` `look tangent` aims at a point a FIXED arc-length ahead (`sTgt =
sHere + 0.045*Smax`, ftsl.h ~2726). Where the path makes a horizontal U-turn (a
"fold") while also changing height, that look-ahead reaches across the fold to a
point at a very different y, so the view pitches sharply — a visible frame-to-frame
flick. The gallery `fly` curve had two folds: the dive turnaround (frames ~117-120,
old peak pitch **+24.5°** staring UP into the y=4.48 ceiling lights — the jerk the
user reported) and the loop closure (frames ~171-175, old peak **-70°** staring DOWN
at the floor, pre-existing/unreported). **Fix (scene-level):** keep y as flat as
possible ACROSS each fold and push the height change onto the straight opening
corridor — return apex lowered from y~2.6 to ~2.2. New peaks +10.6° / -15°, worst
frame-to-frame pitch swing ~30° → 6.5° (measured with `scraps/_cam_curve.py`, a
faithful re-impl of the ftsl.h sampler). Commit d46cacb. **Engine-side fix (also
done):** the latent general issue is now fixed in the `look tangent` branch. Root
cause pinned down numerically: at a fold the look-ahead chord's HORIZONTAL reach
collapses (e.g. closure frame 171: dy only -0.46 but h=0.17), so `asin(dy/L)` blows
the pitch up to -70°. Two defences, both only touching near-fold frames (well-
conditioned frames incl. legit steep dives keep their reach → byte-identical):
(1) `min_reach <frac>` (default 0.5) floors the horizontal reach used for the pitch
at `frac * lookAheadChord`; (2) `look_smooth <sigma_frames>` (default 0/off) does a
wrap-aware Gaussian smooth of the decomposed yaw+pitch so a fold's unavoidable fast
pan (the flight genuinely reverses direction) is spread over frames instead of
snapping. Implemented as a pre-pass before the frame loop (ftsl.h ~2720). Validated
A/B on the ORIGINAL unfixed control points (`scraps/_engine_fix_test.ftsl` vs
`_engine_fix_legacy.ftsl`): legacy frame 11 rakes into a ceiling light, fixed frame
11 is level. Numerically (`scraps/_cam_smooth_test.py`) the original -69.7° rake
becomes a bounded ±30° near-level pan with `min_reach 0.5 look_smooth 2`. NOTE: the
scene-level control-point fix is still the best-LOOKING result for the gallery (it
removes the sharp reversal geometrically → jerk 6.5°); the engine fix is the general
safety net so aggressive future paths degrade to a bounded pan instead of a rake.

### Mode-M dense photon map makes per-frame gather slow — PERF NOTE 2026-07-14

With a very dense saved map (the 60M-photon gallery map deposits ~58.3M photons), each
per-camera density-estimate gather is expensive (~90–120 s/frame at 960×540, 48 spp on a
4090), so a 180-frame flythrough runs ~3–4.5 h. The dense map already yields a smooth
density estimate, so most of the per-frame spp is spent on anti-aliasing rather than noise
reduction. **Tuning opportunity:** once the map is saved (`-savemap`), re-gather via
`-loadmap` at reduced spp (~16–20) for roughly a 2–3× speedup with near-identical quality
(the map deposit — the physically expensive part — is skipped entirely). Not a bug; a
knob worth remembering when iterating on camera angles / radius on a fixed map.

### Shared FORWARD (A/B) multi-camera pass writes all frames only at the end — FIXED 2026-07-14

The shared photon-map path (mode M, both CPU and GPU) writes each frame to disk the moment
its gather completes (crash-safe incremental output). The shared **forward** models A/B
(`renderForwardShared` / `renderForwardSharedCuda`, dispatched from `main.cpp runSharedGroup`)
now do too: `runSharedGroup` was rewritten to chunk the N-photon pass (folding a per-chunk
`seedBase` into the RNG so successive chunks draw independent photons and seedBase==0 stays
bit-identical), accumulate into per-camera SUM films, write all current films + per-camera
`.ftbuf` checkpoints every `-interval` seconds, and support `-resume` from them — mirroring
the single-camera chunked modes. Verified GPU+CPU bit-identity vs standalone, checkpoint,
resume, time budget, and energy conservation (sum/emitted = 1.000000). See commit d43fb6b.

### System BSOD/reboot on GPU context teardown (nvlddmkm.sys driver bug) — MITIGATED 2026-07-14

Twice, the whole machine bugchecked and rebooted **a couple of seconds after an ftrace
render window closed** (once after an abrupt `taskkill /F`). Forensics (minidumps copied
from `C:\Windows\Minidump` via elevated PowerShell, analyzed with `cdb -z <dump> -c
"!analyze -v"`):

- Both dumps fault **inside `nvlddmkm.sys`** (NVIDIA kernel driver, build ~Jan 2026,
  driver 591.86) at the **same function** (+0xcff9xx), at **IRQL 2 (DISPATCH_LEVEL / DPC
  context)** — bugcheck `0xBE` (ATTEMPTED_WRITE_TO_READONLY_MEMORY) and `0xD1`
  (DRIVER_IRQL_NOT_LESS_OR_EQUAL). A DPC-level fault has **no attributable user process**;
  it is the driver's own asynchronous context-teardown DPC, running *after* our process
  has exited. `nvlddmkm` Event 13 (Xid 13, "Graphics FECS Exception") also appears in the
  System log during render sessions. This is a **driver bug**, widely reported for
  RTX 4090 + nvlddmkm across driver versions — not something ftrace causes directly.
- **ftrace's own CUDA kernels are clean.** `compute-sanitizer --tool memcheck` and
  `--tool initcheck` on a mode-M render (`scenes/gallery.ftsl -mode M -camera fly090 -n
  150000 -spp 1 -r 64 48`) both report **0 errors** — no out-of-bounds or uninitialized
  device memory access. So the crash is not an app-side OOB feeding the driver a bad
  pointer; it is purely the driver's teardown path.

**Root enabler found: TDR was disabled.** `HKLM\System\CurrentControlSet\Control\
GraphicsDrivers\TdrLevel = 0` (Timeout Detection & Recovery **off** — no `TdrDelay`
override). With TDR off, a wedged GPU op / driver fault has **no recovery path** and
escalates straight to a bugcheck instead of the driver being reset. (TDR was likely
disabled so long compute kernels wouldn't be killed by the default 2 s watchdog.)

**Mitigations:**
1. **App-side (done):** added `cudaGracefulShutdown()` (`render_cuda.cu`) — a
   `cudaDeviceSynchronize()` + `cudaDeviceReset()` called from `main()` on every exit path
   (normal or exception, guarded by `HAVE_CUDA`). This destroys the CUDA context
   **synchronously, in-process, while quiescent**, instead of leaving it for the driver to
   reap asynchronously from a DPC after `main()` returns — closing the exact window in
   which the fault fires. Build now also compiles device code with `-lineinfo` (free at
   runtime) so any future GPU fault maps to `file:line`.
2. **Operational (done):** always shut renders down **gracefully** (close the live window /
   Ctrl-C / let it finish) — **never `taskkill /F`** a live CUDA process, which yanks the
   context mid-flight and is the most reliable way to hit the teardown fault.
   - **Teardown tracer:** set `FTRACE_TEARDOWN_LOG=<path>` to have `cudaGracefulShutdown()`
     append a flushed line to `<path>` around each driver call (`cudaDeviceSynchronize` /
     `cudaDeviceReset` enter+return). Each line is written with reopen+`fflush` so a hard
     reboot mid-teardown leaves the **failing step as the last line on disk**. Reading it
     after a crash tells us whether the fault is *inside* our `cudaDeviceReset` call (last
     line = "cudaDeviceReset enter", no "returned") or purely the driver's post-exit DPC
     (last line = "cudaDeviceReset returned") — which we can't touch from user space.
3. **OS-side (APPLIED 2026-07-14, takes effect after a reboot):** re-enabled TDR with a
   generous delay so the OS *recovers* a hung GPU instead of bugchecking, while still not
   killing legitimate multi-second kernels. Set via elevated `reg add` under
   `HKLM\System\CurrentControlSet\Control\GraphicsDrivers`:
   `TdrLevel=3 (REG_DWORD)`, `TdrDelay=60` (0x3c s), `TdrDdiDelay=60`. (Was `TdrLevel=0`,
   TDR fully disabled.) Verified present in the registry; **requires a reboot** to take
   effect. Revert with `TdrLevel=0`. Also worth doing: update the NVIDIA driver (591.86 is
   months old) or DDU clean-reinstall.
4. **App-side follow-up (done 2026-07-14):** the shared **mode-M** gather
   (`main.cpp runSharedPhotonMap`) was the ONE render path that never installed a SIGINT
   handler — every other mode wraps its loop in `signal(SIGINT, onInterrupt)`, but this one
   relied on the default terminate action. A backgrounded / headless Ctrl-C therefore
   abruptly killed the live CUDA context mid-gather instead of routing through
   `g_stopRequested` + `cudaGracefulShutdown()` — the exact abrupt-teardown scenario above.
   Fixed with a `SigGuard` RAII (installs SIGINT+SIGBREAK on entry, restores on every exit
   path incl. the GPU early-return) plus stop-checks: the GPU path's `writeFrame`/`liveProg`
   callbacks already return `g_stopRequested`, and the CPU fallback gather loop now breaks on
   `g_stopRequested` between frames. So an interrupt now finishes the current frame, writes
   it, and returns for the orderly teardown.

## Recently fixed

### Scene grammar: two value-less flag keywords can't share a line — FIXED (docs+scene) 2026-07-14

`camera_curve "fly" { … closed   exposure_lock … }` silently dropped the exposure lock, so
the gallery flythrough flickered (every frame auto-exposed independently). Cause: in the
FTSL grammar (`ftsl.h parseValue`) a statement's value is *always* the next token — required
for value-bearing barewords like `material white`, `look tangent`, `caps on`. So
`closed exposure_lock` parses as key `closed` with value `"exposure_lock"`; there is no
separate `exposure_lock` statement, `find(b,"exposure_lock")` returns null, and the lock
never applies (`closed` still works by accident since any non-`off` value is truthy). The
grammar genuinely can't tell `closed exposure_lock` from `material white`, so this is a
by-design limitation, **not** a parser bug to "fix." **Resolution:** put each value-less
flag on its own line — done in `scenes/gallery.ftsl`, and the misleading one-line example in
the `camera_curve` doc comment (`ftsl.h`) is corrected with an explicit NOTE. The CLI
`-exposure-lock` (global override) was always a reliable alternative. Workaround for authors:
one flag keyword per line.

### Mode `M` photon map ported to the GPU (direct density query) — ADDED 2026-07-14

Mode `M` was **CPU-only** — a serious backend gap, since the shared photon map (build
once, gather every camera) is exactly the workload a GPU wins at (e.g. a 90-frame
flythrough). Ported the **direct density query** to CUDA (`render_cuda.cu`
`renderPhotonMapSharedCuda`), reusing the existing pieces: the forward deposit runs on
the **same** `kTrace`/`shadeStep` megakernel (added a `depositPhoton` branch at every
`Diffuse`/`DiffuseTransmit` vertex, gated by a device deposit buffer — a two-pass
count-then-fill for exact sizing), the grid build reuses the tested host
`PhotonMap::build` (download hits → build → re-upload sorted photons + `cellStart`), and
the gather is a new `kGather`/`dPhotonGather` kernel mirroring the CPU `photonGather`
(specular walk, 3×3×3 grid query, per-photon-wavelength XYZ density estimate,
cross-surface reject, emitter term, Beer–Lambert interior; no env term). Gather is
spp-chunked to stay under the Windows TDR watchdog. Gated by `cudaPhotonMapSupported`
(POD-bakeable materials, no env light) + pinhole cameras + no `-pmfg`; otherwise the CPU
path runs. Host dispatch in `main.cpp runSharedPhotonMap` honors the shared
`-exposure-lock` anchor so a flythrough doesn't flicker.

**Validation** (`scenes/_gpumtest.ftsl`, dispersive-glass Cornell box, 2 mode-M cameras):
energy conserves exactly (sum/emitted = 1.000000). CPU-vs-GPU converges *together* as
photons rise (the signature of a correct port whose only difference is the RNG noise
realisation): 4M/8spp → linear relRMSE 51%, Pearson 0.836; 40M/32spp → 28%, 0.954. After
a heavy Gaussian blur to remove Monte-Carlo noise, the two radiance fields agree to
**1.7% / 3.8% linear RMSE at Pearson 0.999** with a best-fit scale of **1.00** (no
systematic brightness bias). The residual per-pixel error is pure caustic/light-source
noise (bright in linear space, slow to converge). **`-pmfg` final gather and env-lit
scenes still fall back to the CPU** — porting the final-gather sub-ray pass is future
work. `README.md` mode-`M`/CUDA/`-device` sections updated.

### Mode `M` optional Jensen final gather (`-pmfg`) — ADDED 2026-07-14

Mode `M` was a **direct** radius density query at the visible point, which inherits the
density estimate's low-frequency blur *at that surface* (softening contact shadows / fine
detail at large gather radii). Added an optional true **final gather** (`-pmfg <K>`,
`g_pmFinalGather`): at the first diffuse hit it shoots `K` cosine-weighted hemisphere
sub-rays (`photonGatherSub` in `photonmap_render.h`), traces one bounce each, and queries
the map at *those* points — so the blur now lives one bounce away, decoupling the visible-
surface sharpness from the gather radius. **Direct** lighting at the visible point is done
with low-variance next-event estimation (`BackwardRenderer::neeLight`) instead of relying
on gather rays randomly striking the light; the gather rays therefore collect indirect
(+ env + specular-direct via a `specularSeen` gate) only, so there is no double-count. The
cosine/pdf and Lambertian `1/pi` cancel to `rho(x)`, folded per-photon at the gather hit
(spectrally, at each photon's wavelength) so two-bounce colour bleed stays correct.
`K = 0` (default) keeps the original direct query — a pure superset. Validated on the
diffuse Cornell box: direct query reproduces the prior numbers (M/R=0.989, relRMSE 4.7%,
r=0.998) and final gather matches mode `R` in energy (diffuse-mask M/R=1.010). The point
of the feature shows up at a large gather radius (`-pmradius 0.06`), in the darkest 10% of
the reference (contact shadows / corner creases): the direct query suffers the classic
photon-map **boundary/corner-darkening bias** (the `1/(pi r^2)` normalisation overshoots
where the gather disc runs off the surface or into shadow) reading **M/R=0.929**, while
final gather is essentially unbiased at **M/R=0.994** (see `scraps/_shadow_bias.py`). Cost
~`K`× per sample, so pair with fewer `-spp`. `DiffuseTransmit`/`Fluorescent` visible points
fall back to the direct query. `README.md` mode-`M` description + CLI table updated. (A
secondary-hemisphere final gather is what this file previously listed as future work.)

### GPU forward camera-splat out-of-bounds write (illegal memory access) — FIXED 2026-07-12

The GPU forward/light-tracing kernel (modes A/B/C, and the splat in M/S/U) could
crash with `[cuda] forward kernel failed: an illegal memory access was encountered`.
**Root cause:** `DCamera::project()` / `lensImage()` compute the splat pixel as
`px = (int)((ix*0.5+0.5)*resX)` in **FP32** (`Real`). The on-film rejection test only
guarantees `ix,iy < 1`, but the gap between the largest float below 1 and 1.0 is
~6e-8, so for a photon landing within that gap of the film edge, `(ix*0.5f+0.5f)`
rounds to exactly `1.0f` and `px` becomes `resX` (likewise `py==resY`). `filmAdd()`
indexes `py*resX+px` with no bounds check → out-of-bounds write. Data-dependent, so
it manifested only for some scenes/resolutions and always eventually with enough
photons (longer renders reliably tripped it). The CPU `Camera::project()` uses
`double` and never rounds up this way, which is why CPU renders were unaffected.
**Fix:** clamp `px∈[0,resX-1]`, `py∈[0,resY-1]` right after the cast in all three GPU
projection sites (`project()` rectilinear + fisheye/panoramic branches, `lensImage()`;
`catchPhoton()` routes through `lensImage()`). The rejection test already guarantees
the point is on-film, so clamping the boundary roundup to the last pixel is exact.
See `render_cuda.cu` ~line 555/577/624.

## Open bugs

### Klein glass mesh had a non-manifold pinch vertex — FIXED 2026-07-14
The Klein mesh (`scraps/klein_hunyuan_clean.obj` and its staged copy `klein_staged.obj`, used
by `settle_test_settled.ftsl` / `klein_glass_ior152.ftsl` / `klein_glass_ior242.ftsl`) failed
the new `-check-watertight` audit. Diagnosis: in RAW OBJ indexing the mesh is a *perfect* closed
2-manifold (951420 edges, every one shared by exactly 2 faces, zero boundary, zero non-manifold).
The defect was a single **3-sheet pinch vertex** — three distinct vertices (ids 151608/151609/
153154, 7+5+5=17 incident faces) snapped by the AI mesh generator to the *same* point
(~(-0.008, 0.331, -0.453), within 9.5e-8) — which only shows as a non-manifold vertex once the
audit welds coincident vertices (weld eps = bbox_diag·1e-7). The audit reported different counts
per instance (3 at full scale, 8–9 on the smaller staged copy) because the weld eps scales with
each mesh's post-transform bbox diagonal. Not a hole and not a broad self-intersection, so
MeshFix ("could not fix everything", changed nothing) was the wrong tool. **Fixed** with
`tools/repair_mesh.py` (MeshLab engine): merge-close-vertices → repair-non-manifold-edges
(remove faces) → repair-non-manifold-vertices → close-holes, taking `klein_hunyuan_clean.obj`
and `klein_staged.obj` to `[OK] … watertight, dielectric` (317038 v / 634076 f, −102 v / −204 f).
Pre-repair copies kept locally as `*.orig.obj` (scraps/ is git-ignored, so the meshes aren't
versioned — only the repair tool is). NB: this pinch was measure-zero and its render impact was
negligible; the audit is just strict about it.

_(former `light cylinder` entry moved to Resolved — it was a misdiagnosis.)_

## Tech debt

### Mesh repair exists (`tools/repair_mesh.py`); isosurface cap-at-polygonise still TODO — 2026-07-14 — MESH PART DONE
`-check-watertight` DETECTS non-airtight geometry and **`tools/repair_mesh.py`** now FIXES meshes
(MeshLab engine by default: merge-close-vertices → repair non-manifold edges/vertices → close
holes; `--engine meshfix` for Attene's MeshFix on self-intersection/hole-heavy meshes; `--place-like`
re-applies a derived copy's transform). Used it to make the Klein glass mesh airtight (see the
FIXED bug entry above).
Still open — **isosurfaces**: `-check-watertight` polygonises the field and audits that, but a
`contained_by` box that clips the surface open would need a fix at *polygonise* time — emit the
flat cap on the clip plane so the marched mesh is closed by construction (marching cubes is already
watertight otherwise). `repair_mesh.py` could also just be run on an exported isosurface OBJ, but
the cap-at-source approach is cleaner. Also optional: a `-repair-mesh` CLI wrapper if we ever want
it in-process (currently repair is a separate Python step, which is fine).

### `-export-mesh` QEM decimation is pathologically slow on huge/self-intersecting meshes — 2026-07-13
`isomesh::decimateAdaptive` (QEM edge-collapse) is fine at small/medium counts but effectively
hangs on multi-million-triangle inputs. Meshing the Klein bottle `a=1.2 b=0.6 c=3.0 d=12.7`
at `-mesh-res 224` produces ~2.19 M tris (the neck self-intersects, so there's a lot of
interior surface); `-mesh-adaptive -mesh-decimate 0.18` on it ran for minutes with flat memory
(~469 MB) and made no visible progress before it was killed. Workaround for now: march at a
lower `-mesh-res` (e.g. 128 → ~716 k tris) to get a lighter mesh directly instead of decimating
a huge one. Proper fix: profile the collapse loop — likely the priority-queue / cost-update or
the link-condition neighbour scan is super-linear on high-valence, self-touching vertices; add
a progress log and a spatial cap, or switch to a vertex-clustering pre-pass before QEM.

### POV-Ray pattern/pigment/spline internal functions not ported — 2026-07-13
`src/pov_functions.h` (generated by `tools/pov_functions_gen.py` from POV-Ray's
`source/vm/fnintern.cpp`) now ports **78 of POV-Ray's ~79 internal isosurface functions**
as exact formulas, shared by the CPU (`pattern.h`/`patternEval`) and GPU
(`render_cuda.cu`/`dPatternEval`).

**DONE — Perlin noise ported (2026-07-13):** `f_noise3d` (76), `f_noise_generator` (78),
`f_hetero_mf` (29), `f_ridge` (58), `f_ridged_mf` (59) are now supported. `src/pov_noise.h`
(generated by `tools/pov_noise_gen.py`) is an exact host+device port of POV's `Noise()`:
its three init tables (`hashTable[8192]`, gradient `RTable`, and the Perlin
`NoisePermutation`/`NoiseGradients` lattice) are re-derived by replicating POV's
deterministic 32-bit-LCG init procedures and baked in as constant data, so the CPU and GPU
evaluate byte-identical noise with no runtime init. All three generators are supported
(1=Original, 2=RangeCorrected [default], 3=Perlin). Device storage uses `__device__`
globals (via `#ifdef __CUDA_ARCH__`) so the ~130 KB of tables sidestep the 64 KB
constant-memory limit. Validated visually: `sqrt(x²+z²)-1 + 0.5*f_noise3d(3x,3y,3z)`
renders a coherent Perlin-lumped cylinder identical in character to POV's f_noise3d.

**Still not ported** (the `EXCLUDE` set in the generator): `f_pattern` (77) — plus the
S-table `f_pigment`/`f_transform`/`f_spline`. These reference a whole
`TPATTERN`/`PIGMENT`/`TRANSFORM`/`Spline` object via `private_data`; they are
function-*wrappers* around POV's texturing engine, not standalone math. Out of scope until
(if ever) that engine is ported. Parser rejects any unported name as an "unknown
identifier", so scenes fail loudly rather than silently.

### Isosurface `contained_by` is box-only — add a sphere/curved container — 2026-07-13 — DONE 2026-07-13
**DONE:** `contained_by { sphere { center <x y z>  radius r } }` is now accepted (`ftsl.h`
`addIsosurface`), storing `Container::Sphere` + world `sphereCenter`/`sphereRadius` on the
`Implicit` (box stays the default). `intersectImplicit` (`implicit.h`) and the device twin
(`render_cuda.cu`) clip the ray against the actual container (sphere → quadratic; box →
face-tracking slab) and carry the container's outward normals for cap shading. The AABB
`im.bounds` is still the BVH-leaf/broad bound (set to the sphere's AABB for sphere
containers). Validated on `f_enneper`/`f_klein_bottle` (rounded clip vs box facets) — see
`scraps/gen_container_test.py` → `png/iso_container_grid.png`.

**What:** an isosurface's `contained_by { min <x y z>  max <x y z> }` is the *only*
container shape we support — an axis-aligned box (see `ftsl.h` `addIsosurface`
~line 1545; the 8 corners are transformed to world and reduced to an AABB stored as
`im.bounds`). POV-Ray also lets the container be a `sphere` (and in fact any shape).
**Why it matters:** for a surface that reaches the container wall (any *unbounded*
surface like `f_enneper`, or a solid lump that pokes out), a box clips it along **flat
planes**, so the cut reads as hard angular facets. A **sphere** container clips along a
smooth curved boundary, so the unavoidable cut looks like a natural rounded edge instead
of a sawn plane — this is why hand-tuned POV enneper/klein renders frame cleanly and ours
show flat patches. It's container ergonomics, not a math gap: both engines must clip an
infinite surface *somewhere*; the sphere just hides the seam.
**Where / proper fix:** `ftsl.h` `addIsosurface` — accept `contained_by { sphere {
center <x y z> radius r } }` (keep `min`/`max` box as the default). Store the container
shape on the `Implicit` (currently just `im.bounds`, an AABB used to clip the ray in
`implicit.h intersectImplicit` ~line 246). The ray-clip step must then intersect the ray
with the actual container (sphere slab → quadratic) rather than the AABB, and the CUDA
mirror (`render_cuda.cu` `dIntersectImplicit`) needs the same. AABB stays as the BVH-leaf
bound regardless.

### Isosurface container has no cap/`open` control (and no proper cap at all) — 2026-07-13 — DONE 2026-07-13
**DONE:** `intersectImplicit` (CPU `implicit.h` + device `render_cuda.cu`) now caps the
container. In the **default capped** mode a ray that enters the container already inside
the solid (`f < 0` at the near clip) registers a hit on the container's near face (a NEAR
cap); a ray that reaches the container exit still inside the solid registers a hit on the
far face (a FAR cap, only when the far clip is the container itself, so bounce/transmission/
shadow rays originating inside the solid seal correctly). Both use the container's outward
normal and the isosurface material. The **`open`** keyword on the `isosurface {}` block
(`ftsl.h`, default `capped = true` for expr fields) suppresses both caps, revealing the
cut edge. Fully-bounded surfaces (`f > 0` at entry) never trigger a cap, so SDF/CSG leaves
are byte-identical. Validated: `f_enneper`/`f_klein_bottle` render as cleanly sealed solids
by default and open shells with `open`.

**What:** where an isosurface's solid interior (`f < 0`) is sliced by the container wall,
we render **neither** a clean sealed cap **nor** a clean open edge. `intersectImplicit`
(`implicit.h` ~line 245) clips the ray to the container and reports the first field
*sign change* inside it; it never treats the container faces as geometry. So a solid cut
by the box returns the next interior crossing (its back/inner wall) or passes straight
through — reading as odd flat interior patches or see-through holes.
**Background (what a "cap" is):** convention is `f < 0` = solid inside, `f > 0` = outside.
When the container plane cuts through solid material you must choose: **capped/"closed"**
(POV default) draws that slice as a flat face of the object's material, sealing the solid
flush with the wall (looks cleanly sawn off); **`open`** (POV keyword) omits the wall so
the surface just ends and you see into/through the interior. Only matters for surfaces
that actually *reach* the container (`f_enneper`, the klein bottle's outer shell); a fully
bounded surface never touches the wall so the choice is moot.
**Why it matters:** without a real capped mode, box-cut solids can't be shown as clean
solids; without an `open` option, thin-shell / hollow looks aren't authorable. Today's
behavior is effectively a broken third option.
**Where / proper fix:** in `intersectImplicit` (CPU) and `dIntersectImplicit`
(`render_cuda.cu`), detect the case where the ray enters the container already inside the
solid (`f < 0` at the near clip `t0`, or exits the far clip `t1` still `f < 0`) and, in the
**default capped** mode, register a hit on the container face itself (position = clip
point, normal = the container's inward face normal, material = the isosurface material).
Add an `open` toggle to the `isosurface {}` block (`ftsl.h`) that suppresses these caps
(current behavior). Pairs naturally with the sphere-container item above (a sphere cap is
a spherical patch with the sphere's radial normal). Validate on `f_enneper` (should read
as a cleanly-capped solid by default, an open shell with `open`).

### glTF/GLB loader is a static-geometry subset — 2026-07-12
The new glTF 2.0 loader (`src/gltf.h` + `src/third_party/json.h`) covers the common
static-mesh case but deliberately omits a number of glTF features. Each is a scoped
follow-up, not a bug:
- **No textures.** Only `baseColorFactor`/`metallicFactor`/`roughnessFactor` *scalars*
  are read; `baseColorTexture`/`metallicRoughnessTexture`/`normalTexture` are ignored.
  Proper fix: decode referenced images (glTF images are PNG/JPEG — the renderer already
  vendors stb_image), register them as `Scene::textures`, and set `reflectTex`/UV set.
- **No KHR material extensions** (transmission, clearcoat, volume, ior, emissive
  strength, sheen, specular). A glass glTF loads as an opaque glossy/diffuse, not a
  dielectric. Proper fix: read `extensions.KHR_materials_transmission`/`_ior` → map to
  `MatType::Dielectric` with the given ior; other extensions as feasible.
- **No `emissiveFactor` import.** Emissive glTF materials load unlit. (Intentionally
  skipped for now: setting `emit` without registering the tris as a sampled light would
  desync NEE; doing it right means adding mesh-emitter area lights — tied to ROADMAP §5
  "emissive triangles".)
- **No skinning, morph targets, animation, or sparse accessors.** Static bind pose only.
- **Non-triangle primitives** (points/lines/strips/fans, `mode != 4`) are skipped with a
  note; only `mode 4` (TRIANGLES) is baked.
- Materials are created **per glTF material, not deduplicated across meshes/files**. (A
  `mesh` still bakes its triangles into `Scene::tris`; use `mesh_asset`/`mesh_instance`
  for shared instanced geometry — see below.)
The core path (buffers/GLB, node transforms, POSITION/NORMAL/TEXCOORD_0, indexed +
non-indexed tris, metallic-roughness → BSDF) is validated on CPU and GPU.

### Instancing memory saving is CPU-only (GPU expands instances) — 2026-07-12 — DONE 2026-07-13
`mesh_asset`/`mesh_instance` (§5c) give a true two-level BVH on the CPU: instances share
one BLAS (triangles + BVH), so N copies cost N affines. **The GPU has no two-level
traversal** — `buildUploadScene` (`render_cuda.cu`) EXPANDS every instance into
world-space triangles, appends them to the flat device tri list, and rebuilds a single
flat BVH over the whole set at upload. Images are identical to the CPU, but device memory
scales with total instanced triangles (no sharing), so a huge instanced scene that fits on
the CPU can OOM on the GPU. Proper fix: a device two-level BVH — upload per-BLAS
node/tri/primIdx pools + an instance table (toLocal affine + blasId + matOverride) and add
an instance-leaf branch to the device `traverseClosest`/`traverseAny` that transforms the
ray into BLAS space (parametric `t` is preserved, exactly as on the CPU). Deferred because
it touches the hottest device kernel; the expand-at-upload path is correct and low-risk.

**RESOLVED 2026-07-13 — device two-level BVH.** `render_cuda.cu` now mirrors the CPU.
`Scene::bvh` (TLAS) is uploaded **verbatim** in all cases; its prim-index layout
`[tris | spheres | implicits | instances]` is understood by the device leaf dispatch in
both `closestHit` and `occluded` (a prim index `>= nTris+nSph+nImplicits` is an instance
leaf). New device structs `DBlas { nodeOff, triOff, primOff }` and `DInstance { Lm[9],
Lt[3] (world→local affine), Nm[9] ((toWorld)⁻ᵀ normal matrix, host-precomputed), blasId,
matOverride }`. Each `Blas` contributes its local-space tris/BVH-nodes/primIdx to three
**concatenated shared pools** (`blasTris`/`blasNodes`/`blasPrim`) uploaded ONCE, and a
`DInstance` places it via an affine — so N copies cost one `DInstance` each, not N× tris.
Device `blasClosest`/`blasOccluded` walk the shared sub-BVH in BLAS-local space (48-deep
local stack); `affPoint`/`affDir` transform the ray (dir NOT renormalized, so local `t` ==
world `t`, matching the host `Blas`); `instanceHitToWorld` maps the hit back (normal by
`Nm`, shading normal re-oriented, matOverride applied). Validated with
`scraps/instance_test.ftsl` (4 tori sharing one 16 384-tri BLAS, incl. a material override
and a mirror): GPU mode B matches CPU backward reference mode R at Pearson r=0.996 (MAE
~1/255; residual is the forward-vs-backward mirror-highlight difference). Implicit scenes
still render correctly (the leaf-dispatch bounds change is a no-op with no instances).
Device geometry memory is now flat in instance count.

### Forward modes render ~5% brighter than the backward reference (`R`) — 2026-07-12
On a pure-diffuse Cornell box (`scraps/cornell_diffuse.ftsl`) the forward splat modes
and the new photon-map mode agree with each other but sit **~5% brighter** than the
backward path tracer:
- `R` (backward): mean 61.05  →  `B` (forward splat): 63.96  →  `M` (photon map): 63.84.

Mode `M` matching mode `B` to within 0.2% is the *expected* result (both measure the
same forward light transport, just from a stored map vs. a live splat) and **confirms
the photon map is correct**. The open question is the **pre-existing forward-vs-backward
discrepancy** — `B` and `R` should converge to the same image but don't quite. Likely
suspects: a subtle difference in area-light emission normalization / solid-angle pdf
between the forward emit sampler and the backward NEE light pdf, or a `cos`/pdf factor
at the light or first diffuse bounce. Not introduced by this work; surfaced by the mode-M
validation. **Proper fix:** derive both estimators' light-vertex measure on paper for the
1-bounce diffuse case and reconcile the constant (check `emitSampler` power vs. `sampleLight`
radiance × pdf). Until then `V`'s residual bakes this ~5% in.

### No bounded / per-object participating medium (fog is global-only) — 2026-07-12 — CPU + GPU forward DONE 2026-07-12 (box/sphere/implicit bounds, density fields, multi-medium superposition, object-name bounds)
**Resolved on the CPU forward tracer.** The `medium` block now takes an optional
`bounds { min/max }` box (AABB the fog is clipped to) and an optional `density <expr>`
scalar field (same infix expression VM as isosurface `function` fields — variables
`x y z r`, constant `pi`) that multiplies `sigma_t` per point, so fog forms blobs with
soft, formula-defined boundaries. Majorant is `density_max` (explicit or auto-estimated
on a 24³ grid over `bounds`). Sampling uses unbiased **delta (Woodcock) tracking** for
scattering and **ratio tracking** for shadow transmittance; a plain homogeneous medium is
bit-identical to before (one RNG draw in the free-flight, exact `exp` transmittance).
Implemented in `scene.h` (`Medium` struct: `density`/`densityMax`/`bounded`/`bmin`/`bmax`
+ `densityAt`/`clipToBounds`/`heterogeneous`), `ftsl.h` `addMedium` (bounds/density/
density_max parsing), `render.h` (`sampleMediumCollision`/`mediumTransmittance` + connect
updates). Validated with `scraps/fogblob.ftsl` (a soft glowing sphere blob, mode B).

**GPU forward — DONE 2026-07-12.** The density field + bounds + delta/ratio tracking are
now ported to `render_cuda.cu`: `DMedium` carries `heterogeneous`/`density` (a device
`PatNode` pool + `densityN`)/`densityMax`/`bounded`/`bmin`/`bmax`; `dMedDensityAt` (postfix
VM twin of `densityAt`), `dMedClip` (twin of `clipToBounds`), `dMedSampleCollision` (delta/
Woodcock tracking twin of `sampleMediumCollision`), and `dMedTransmittance` (ratio-tracking
twin of `mediumTransmittance`) drive the forward `shadeStep` free-flight and every camera
splat (`connect`/`connectVolume`/`connectLens`/`connectLensVolume` — the two RNG-less
connects now take `rng` for ratio tracking). A homogeneous medium keeps the exact analytic
path (no extra RNG draw). Validated on an RTX 4090 mode B: `scraps/fogblob.ftsl` GPU-vs-CPU
16×16-block RMSE 1.07/255 with ~0 bias (per-pixel diff is pure MC noise from the 0.95-albedo
fog; means 38.57 vs 38.56), and a homogeneous regression (`scraps/foghom.ftsl`) block RMSE
2.45/255, bias −0.009.

**Per-object (sphere) fog bound — DONE 2026-07-12.** `bounds` now also accepts
`{ center <x y z> radius <r> }`, confining the fog to a **sphere** region — the simple
per-object case ("the whole inside of a glass sphere"): author the same center/radius as
the object. Added `MediumBound { Box, Sphere }` + `boundShape`/`bcenter`/`bradius` to the
`Medium` struct with a ray∩sphere interval in `clipToBounds` (heterogeneous density works
inside a sphere too — the majorant grid uses the sphere's AABB, filled in by the parser).
Mirrored on the GPU (`DMedium.boundShape`/`bcenter`/`bradius`, `dMedClip` sphere branch,
upload path). Validated on an RTX 4090 mode B: open glowing orb `scraps/fogorb.ftsl`
GPU-vs-CPU block RMSE 0.96/255 (bias 0.024), glass-shell `scraps/fogsphere.ftsl` block RMSE
0.57/255 (bias −0.010); box/heterogeneous path unchanged (fogblob block RMSE still 1.07).
*Limitations:* (1) **Fog inside a `dielectric` shell is not imaged directly by the
next-event modes — an accuracy (bias) issue, empirically confirmed 2026-07-12.** The
direct view of the fog is a specular↔volume (SDS-type) path: the camera sees the fog
*through* the curved glass, i.e. along a *refracted* line. The next-event/splat modes
connect a fog vertex to the camera with a **straight** ray, which (a) is occluded by the
glass surface (`occluded()` treats every surface as opaque) and (b) could not bend even if
it weren't — so the contribution is structurally **zero**, not merely noisy. This affects
both the **pinhole splat (`B`)** and the **finite-lens splat (`A`)** — both are NEE-based,
and both render the fog-through-glass as **black** (verified: `scraps/fogsphere.ftsl` mode
B whole-image mean 6.6 but the fog-sphere center box mean 0.000; mode A identical). Only the
**physically-tracing modes** — photon-catch (`C`) and BDPT (`D`) — can sample the path at
all, because a real photon scatters in the fog, **refracts** out through the glass, and
lands on a finite aperture. For `C` that path is extraordinarily improbable (a fog-scattered
photon must exit heading almost exactly at the pupil), so at practical sample counts `C` is
effectively black too (60 M photons, aperture 0.45: fog-sphere center still mean 0.000) — an
**efficiency** problem on top of the accuracy one. **BDPT `D` — RESOLVED 2026-07-12** (volumetric
BDPT, below): its camera subpath refracts through the shell (specular vertices) to a volume
in-scatter vertex, then MIS-connects to the light, so a lantern inside a fogged glass sphere
images as a bright disc — `scraps/fogsphere.ftsl` mode D fog-sphere center box mean 0.22
(saturating) vs mode B's 0.00, at the same absolute exposure.
The fog still correctly **lights the surrounding room** (indirect, via NEE off the walls),
and an **open** fog sphere (no glass shell) is directly viewable in every forward mode
(`scraps/fogorb.ftsl` mode B center box mean 135.8). A naive "let connect rays pass through
glass" hack is wrong (it draws the fog along a straight line, with no lensing, in the wrong
place) and is deliberately avoided — the correct fix is the analytic specular connection below.

**Mode-B analytic specular connection through glass SPHERES — DONE 2026-07-12.** The proper
refractive/manifold next-event estimation is now implemented for the tractable case: a
**glass sphere** in the **pinhole splat (`B`)**. For each fog in-scatter vertex (and each
diffuse surface vertex), the renderer solves — in closed form — the refracted eye ray that
leaves the vertex, bends through the sphere, and reaches the pinhole: a planar reduction of
the two-refraction manifold to a **1-D root solve**, with a ray-differential Jacobian
(`G = eps²/|ax·by − ay·bx|`) supplying the splat weight, and Fresnel-transmittance ×
Beer-Lambert interior absorption × medium transmittance along the two glass segments. The
sphere's ior is evaluated at the photon's **own wavelength**, so the refraction is dispersive
for free. Unified surface/volume vertices via a `SpecVtx`/`DSpecVtx` `term(wP)` (Lambertian
`rho/π·cosSurf` vs HG-phase·albedo). Implemented on **CPU** (`render.h`:
`connectSpecularSphere`/`connectSpecularSphereInside`, `camSpecularSplatAll`/`…VolumeAll`)
and **GPU** (`render_cuda.cu`: `dConnectSpecularSphere`/`…Inside`, `camSpecularSplatAll`/
`camSpecularSplatVolumeAll`), validated GPU-vs-CPU and vs BDPT/mode-P ground truth. So a
lantern glowing inside — and a fly-through *through* — a clear glass orb now images correctly
in mode B (see `scenes/lanterns.ftsl`). *Still out of scope for now:* the finite-lens splat
(`A`), photon-catch (`C`), and **non-spherical** dielectric shells — those still render the
direct fog-through-glass view black in the forward splat modes (use BDPT `D`, which handles
any shape). Mode A and flat-plane (window/pane) analytic connections are the next tracked
items.

**Multiple coexisting media (superposition) — DONE 2026-07-12.** `Scene::medium` is now a
vector `Scene::media` of independent, possibly overlapping media; several `medium {}` blocks
coexist (e.g. two tinted fog orbs + a global haze). The forward tracer superposes them
physically: extinction adds, so total transmittance is the **product** of the per-medium
transmittances (`Renderer::mediaTransmittance` / `dMediaTransmittance`), and the first
collision is the **earliest** of the media's independent free-flights, with the winning
medium's albedo/`g` driving the scatter (`sampleMediaCollision` / `dMediaSampleCollision` —
Poisson superposition). A single-medium scene stays bit-identical. `Scene::backwardMedium()`
returns the first medium for the homogeneous-only backward/BDPT path. Implemented in
`scene.h`, `render.h`, `ftsl.h` (`addMedium` appends), `render_cuda.cu` (`DScene.media`/
`mediaN` + a `DMedium` array). Validated on an RTX 4090 mode B: `scraps/fogmulti.ftsl`
(warm + cool disjoint orbs + global haze) GPU-vs-CPU 16×16-block RMSE 1.40/255 (bias 0.012,
means match to 0.02%); single-medium regression (`scraps/fogorb.ftsl`) block RMSE 1.07,
unchanged.

**Object-name / implicit-shape fog bounds — DONE 2026-07-12.** `bounds { object "<name>" }`
shapes the fog to a **named** scene object: a named `sphere` → its exact analytic sphere
bound; a named `isosurface` → **field membership** (a new `MediumBound::Implicit`: the fog
fills the field interior via `fieldEval < 0` — inside-sign auto-detected from the field's
value at its AABB center — carved per-point inside delta/ratio tracking over the field's
AABB, reusing the same field VM as isosurface rendering); a named `mesh` → the mesh's world
**AABB** (box approximation; true mesh containment deferred). Media are resolved in a
deferred second sweep so the object may be authored anywhere. Implemented in `scene.h`
(`Medium::boundField`/`boundFieldExpr`/`boundInsideNeg` + `insideField`/`densityAt`/
`heterogeneous`), `ftsl.h` (name registries populated by `addSphere`/`addIsosurface`/
`addMesh`; `object` branch in `addMedium`), `render_cuda.cu` (`DMedium.boundField`/
`boundFieldN`/`boundFieldExpr`/`boundInsideNeg`, `dMedDensityAt` membership carve-out,
`appendFieldProgram` bakes the medium field into the shared device field pool). Validated on
an RTX 4090 mode B: metaball-`blob`-shaped glowing fog in a glass shell (`scraps/fogimplicit.ftsl`)
GPU-vs-CPU energy identical (absorbed 0.9978) and indirect room lighting agreeing within the
(large) dim-caustic noise floor. *(Same fog-inside-glass direct-view limitation as above
applies — an implicit-shaped fog is enclosed by its own isosurface, so its direct camera view
is a refracted SDS path; it lights the room correctly.)*

**Remaining gap (BDPT fully closed):**
- **BDPT (mode D) — ALL media DONE 2026-07-12 (CPU + GPU), incl. heterogeneous.** `bdpt.h`
  and the GPU BDPT megakernel (`render_cuda.cu` `kBdpt`) handle media of every kind —
  global haze, multiple superposed media, box/sphere/object-**bounded** fog, **and
  heterogeneous `density`-field blobs** — with volume in-scatter (`VType::Medium` /
  `BV_MEDIUM`) vertices, HG-phase connections and transmittance-weighted edges. Both
  `bdptUnsupportedFeature` (CPU) and `cudaBdptSupported` (GPU) now accept any medium.
  Validation (homogeneous): global haze CPU-vs-GPU whole-image mean 0.04698 vs 0.04702
  (+0.09%); bounded fog-through-glass (`scraps/fogsphere.ftsl`) CPU-vs-GPU center 0.237 vs
  0.242. Validation (heterogeneous): `scraps/fogblob.ftsl` (soft-edged density blob) mode D
  GPU vs mode B forward reference mean 0.04213 vs 0.04247 (−0.8%), centerMean 0.30009 vs
  0.30211 (−0.7%) — within the ~6% MC noise floor, confirming unbiased. See the resolved
  entry below for why the homogeneous cancellation is *not* required for correctness.
- **Backward modes (R/V) + P camera layer still treat it as homogeneous** (on BOTH
  backends). `backward.h` (modes R/V) and the camera-side layer of the P composite still
  use the medium as a single global homogeneous haze and ignore `density`/`bounds`; on the
  GPU, `cudaBackwardSupported` rejects *any* medium so R/V fall back to the CPU tracer,
  which shares that homogeneous-only limitation. `main.cpp` `runRender` **warns** when a
  heterogeneous/bounded medium is rendered in R/V/P. Proper fix: port delta/ratio tracking
  into the backward volume march too (then mirror it on the GPU).

### Heterogeneous (density-field) media in BDPT (mode D) — DONE 2026-07-12 (CPU + GPU)
**What:** BDPT (mode D) now renders **heterogeneous** (`density`-field) media unbiasedly on
both backends, using the *same* code path as homogeneous/bounded media — no null-scattering
rewrite was needed. Both `bdptUnsupportedFeature` (CPU) and `cudaBdptSupported` (GPU) accept
any medium; the heterogeneous rejections were removed.

**Why the earlier "cancellation breaks → biased" reasoning was wrong (corrected):** the
balance-heuristic MIS weights `w_s = p̂_s / Σ_i p̂_i` are a **partition of unity for any
consistent positive pdfs** — `Σ_s w_s = 1` holds identically, regardless of what each `p̂_i`
is. The estimator `E[ Σ_s w_s · f/p_s ] = ∫ f · (Σ_s w_s) dx = ∫ f dx` is therefore
**unbiased** whenever (a) the *sampled* strategy's throughput `f/p_s` is exact, and (b) the
weights sum to 1. Omitting the heterogeneous distance-pdf / transmittance from the MIS
weights (the homogeneous bookkeeping we reuse) only makes the `p̂_i` a *different but still
consistent* set of positive numbers — it changes the **variance**, never the bias. This is
exactly what PBRT-v3 does for heterogeneous media. The homogeneous σt·exp/transmittance
cancellation is a variance nicety, **not** a correctness requirement.

**Why the sampled-strategy throughput stays exact:** subpath medium vertices are placed by
**delta (Woodcock) tracking** (`sampleMediaCollision`) with **analog throughput** (β
unchanged; RR-absorb on albedo) — the same unbiased sampler validated mode B uses.
Connection edges are weighted by **ratio-tracking transmittance** (`mediaTransmittance`),
which appears *linearly* in the connection throughput, so its unbiased estimate keeps the
connection estimate unbiased. Albedo and phase `g` are spatially constant (only density
varies), so a medium vertex's `mediumId`/`mediumG` fully determine phase + albedo and
`vertexPdf` recomputes the cosine-free phase-direction density consistently forward/reverse
regardless of heterogeneity.

**Implementation:** removed the `heterogeneous()` guards in `bdptUnsupportedFeature`
(`main.cpp`) and `cudaBdptSupported` (`render_cuda.cu`); the existing `randomWalk` /
`dRandomWalk` medium-event blocks and `connectBDPT` / `dConnectBDPT` transmittance-weighted
connections already handle spatially-varying σt (they call the same delta/ratio-tracking
helpers the forward tracer uses). No path-budget or MIS changes were required.

**Validation:** `scraps/fogblob.ftsl` (soft-edged density blob, absolute exposure): mode D
GPU vs mode B forward reference — mean 0.04213 vs 0.04247 (−0.8%), centerMean(30%) 0.30009 vs
0.30211 (−0.7%); mode D CPU vs GPU — mean 0.04204 vs 0.04213 (−0.2%), centerMean 0.29972 vs
0.30009 (−0.1%). All within the ~6–9% MC noise floor → unbiased and backend-consistent.

**Optional future variance work (not correctness):** the null-scattering path-integral
formulation (Miller/Georgiev/Jarosz 2019; UPBP, Křivánek et al. 2014) would put the omitted
heterogeneous transmittance *into* the MIS weights, reducing variance in optically-thick
heterogeneous media. Purely a variance optimization — the current estimator is already
unbiased.

### Diffuse-transmission material — CPU DONE 2026-07-12 (GPU port pending)
Added `type translucent` (alias `diffuse_transmit`): a two-sided Lambertian BSDF — the
front hemisphere scatters the `reflect` albedo, the back hemisphere scatters the `transmit`
albedo, so light diffuses THROUGH the surface (soft "waxy"/"paper"/thin-skin look). Because
both lobes are non-specular it renders/connects in **every** CPU mode: forward A/B/C
(`render.h` two-sided `camSplatAll` — flip the normal, wrong side self-rejects), backward
R/V (`backward.h` two NEE calls, one per hemisphere via a normal-flipped `Hit`), and BDPT D
(`bdpt.h` — added to `isConnectibleMat`, `bsdfF`/`bsdfPdf` two-lobe eval, scatter lobe
selection, and — critically — the connection cosine guards in `connectBDPT` now allow the
back hemisphere for two-sided materials via `isTwoSidedMat`, using `|cos|` in the geometry
term with `bsdfF>0` as the real gate; `lambda` is now threaded through
`bsdfPdf`→`vertexPdf`→`misWeight` so the wavelength-dependent lobe-selection pdf is exact).
`reflect`+`transmit` are energy-clamped so their sum ≤ 1. Validated: `scraps/translucent_panel.ftsl`
(backlit warm panel) renders consistently across modes B, R, and D.
**GPU — DONE 2026-07-12.** `render_cuda.cu` now handles `translucent`: `D_DIFFUSETRANSMIT`
(enum aligned to `MatType` with a `D_LAYERED` placeholder), a `DMaterial::transmit[SPEC_N]`
field baked on upload, the two-lobe splat/scatter in the forward `shadeStep` (megakernel +
wavefront share it) and the backward reference `bkRadiance` (GPU mode R). The restricted GPU
BDPT kernel (`kBdpt`) has no two-sided strategy, so translucent scenes fall back to the
validated CPU BDPT via `cudaBdptSupported` (same pattern as frosted glass / textures /
fluorescence). Validated on an RTX 4090: forward B and backward R GPU-vs-CPU RMSE = 3.82/255
(pure MC noise, matching means), and GPU mode D renders the panel correctly through the CPU
fallback.
**Remaining:** a true **BSSRDF / dipole / random-walk subsurface** model (for thick solid SSS
with proper mean-free-path blurring) is still not implemented — this material is a thin
diffuse-transmission approximation, not volumetric SSS.

## Resolved

### Nested-dielectric exterior IOR hardcoded to 1.0 (glass-in-water wrong) — DONE 2026-07-14 (Level 0)
Every dielectric interface previously assumed the exterior medium was vacuum (IOR 1.0), so
glass-in-water refracted 1.0↔1.52 instead of 1.33↔1.52 and nested/overlapping solids were
wrong in **all** modes. Fixed at ROADMAP §7 **Level 0** (Schmidt & Budge 2002 priority
field): each path now carries a tiny LIFO medium stack (`src/medium_stack.h`, device
`DMediumStack` in `render_cuda.cu`); at every dielectric hit the exterior IOR comes from the
enclosing highest-priority medium, and the lower-priority surface inside an overlap is
suppressed (ray passes straight through). Wired through **all** integrators: CPU R/A/B/C/D/M/S/U
(`backward.h`, `render.h`, `bdpt.h`, `photonmap_render.h`, `sppm_render.h`, `vcm.h`) and GPU
forward/backward/BDPT/photon (`render_cuda.cu`). **Safe fallback:** the priority rule fires
only when *both* sides carry an explicit `priority` (air/empty stack always valid at 1.0), so
priority-free scenes render bit-identically. An ahead-of-time scene audit warns when two
dielectric bounding volumes overlap and either lacks a `priority` (isosurfaces compared by
their `contained_by` container bounds — conservative, never misses a real overlap). Validated:
mode-R priority vs no-priority differ across 33.6% of pixels (mean 5.5/255) — well above render
noise; GPU-priority matches CPU-priority reference to mean 1.3/255; modes C/M energy-conserving.
- **Remaining gap (pre-existing, not a regression):** device BDPT (`dRandomWalk` in
  `render_cuda.cu`) still does **not** apply Beer-Lambert interior absorption across in-glass
  segments — it never did, and Level 0 only added the priority-driven exterior-IOR resolution
  there, not absorption. GPU BDPT already falls back to CPU for frosted/colored glass anyway
  (`cudaBdptSupported`), so colored-glass absorption is exercised on the CPU path; clear-glass
  device BDPT is unaffected. Proper fix: thread `stk.topMat()` absorption into `dRandomWalk`'s
  segment loop the way the forward/backward device paths do.
- **Future (ROADMAP §7 Levels 1/2, deferred):** true physical stacking of co-located media and
  interpenetrating volumes remain opt-in tiers beyond Level 0.

### Mode `P` composite is not progressive; `R`/`D` have no disk resume — DONE 2026-07-13
Both gaps closed. `-time`/`-noise`/`-forever`/`-preview`/`-interval` and `-resume`/
`-checkpoint` now cover **all** the accumulating image modes — the forward camera models
`A`/`B`/`C`, the spp reference modes `R` (backward) / `D` (BDPT), and the composite `P`.
- **Mode `P` is now progressive** (`runCompositeProgressive`, `main.cpp` ~line 1986). The
  view-dependent first-hit pixel classification is computed **once** (`classifyComposite`)
  and reused; the driver then alternates forward (model-B, `N` photons) and backward
  (camera-side, `spp`) batches into two persistent SUM films, adapting the batch toward
  ~0.5 s so early frames appear fast. After each batch it re-fits the forward→backward
  scale `s` and re-blends (`compositeFromFilms`), writing the image + a status line every
  `-interval`. The old single-shot `renderComposite` wrapper was deleted.
- **`R`/`D` now disk-resume** through `runSppProgressive`, reusing the single-film
  `Checkpoint`/`writeCheckpoint`/`readCheckpoint` format keyed on **spp** (the mode byte is
  folded into the identity guard so an `R` checkpoint can't be loaded as `D`, verified).
- **Mode `P` gets a dual-film checkpoint** (`CompositeCheckpoint`, magic `FTPCM02`) storing
  the forward SUM + backward SUM + their counts + the forward energy tally.
- **Seed decorrelation on resume:** fresh samples are biased past the loaded ones via
  `SppProgress::sampleBase` (CPU: added to the per-chunk seed; GPU: XORed into the megakernel
  seed base) so continued samples are an independent noise realization. Validated: `R`
  58 196→116 545 spp with noise tracking 100/√spp exactly (0.41 %→0.29 %); `D`
  1016→2052 spp (3.14 %→2.21 %); `P` 36.2 M photons/4636 spp→56.5 M/7228 spp with the
  diffuse-side residual falling 0.0281→0.0226 — proving the resumed samples reduce variance
  rather than re-tracing identical paths.

### `light cylinder` "emits no illumination" — NOT A BUG (misdiagnosis) — RESOLVED 2026-07-13
The original 2026-07-11 report claimed a `light cylinder` glows but lights nothing on
both CPU and GPU, citing an "auto-exposure 1.54e-14, i.e. zero contribution." That
inference was wrong on two counts, and re-testing shows the cylinder light works
correctly on **both** backends.
- **A ~1e-14 auto-exposure is normal here, not "zero light."** This renderer uses
  physically-scaled blackbody SPDs whose absolute radiance is ~1e13 W/m²/sr, so the
  content-based auto-exposure lands around 1e-14 for *any* such scene — the stock
  Cornell box (`-scene cornell`) reports `auto-exposure=8.87e-14`.
- **The isolation scene had no explicit `power`**, so `absPower` was a no-op and the
  emitter surface kept the raw (astronomically bright) blackbody radiance. A directly-
  visible emitter that bright dominates the auto-exposure anchor and crushes the
  genuinely-lit wall to near-black in the tonemap. **A `light sphere` in the identical
  no-`power` isolation scene behaves the same way** — so it was never cylinder-specific.
- **Controlled proof.** In absolute mode (each light given an explicit `power`, so a
  fixed sensor gain is used instead of content-based auto-exposure), a cylinder and a
  sphere light of equal power illuminate the wall essentially identically: at `power 30`
  wall-region mean ≈ 1.32 (cyl) vs 1.11 (sph); at `power 4000`, 21.33 (cyl) vs 19.65
  (sph). GPU forward (mode B) matches the CPU backward (mode R): wall mean 21.60 vs
  21.33. The `neeLight`/`neeVolume` switches in `backward.h` already dispatch the
  Cylinder shape (`sampleCylinderVisible` for the un-capped front-facing arc; uniform
  `samplePoint` for capped capsules), and forward photon emission selects it via the
  power CDF — all correct.
- **Repro (now shows a properly lit wall):** `scraps/cyl_test.ftsl` was updated to give
  the tube `power 4000`; `ftrace -in scraps/cyl_test.ftsl -mode R -device cpu -spp 128
  -o png/cyl_test.png` shows the wall lit with correct falloff around the tube.
- **Lesson for the tracker:** don't read a tiny auto-exposure as "black"; verify with
  absolute-`power` lighting or by measuring HDR/PNG pixels of a receiver away from the
  directly-visible emitter.

### Unified live progress across all image modes (`R`/`D` join `A`/`B`/`C`) — DONE 2026-07-12
- **What:** modes `R` (backward reference) and `D` (BDPT) previously ran as a single
  monolithic launch with **no progress output, no periodic image write, and no way to stop
  early** — a multi-hour reference render showed nothing until it finished (and a killed
  render lost everything). Now every image-forming mode shares one progress driver: a
  status line (or `-preview` ANSI thumbnail) with a `~noise%` estimate, a periodic
  crash-safe image rewrite every `-interval` seconds, and `-time`/`-noise`/`-forever`
  budgeting with clean Ctrl-C — on both CPU and GPU.
- **How:** `R`/`D` films accumulate a **SUM over samples-per-pixel** (CPU `renderBdpt` was
  changed from ÷spp to SUM to match `renderBackward`/the GPU), so they chunk exactly like
  the forward photon-count films. The GPU kernels (`kBackward`/`kBdpt`) take
  `chunkSpp`/`sppTotal`/`sampleBase` and seed the RNG on the **global sample index**
  (`gidx = pix*sppTotal + sampleBase + local`), so any chunking draws the same union of
  streams as one `sppTotal` pass — **bit-identical** to the old single-shot for a given spp.
  `gpuSppChunks` (device) and `cpuSppChunks` (host, via a `seedOffset` on the CPU renderers)
  own the chunk loop; `runSppProgressive` (`main.cpp`) is the shared reporter, reused by the
  mode-`R` and mode-`D` dispatch. A time/noise/forever budget opens the spp target to a
  capped `UNBOUNDED_SPP=1e9` (keeps `pix*sppTotal` inside int64). New: `render_progress.h`
  (`SppProgress` callback).

### Concurrent GPU renders silently wrote a black PNG (all-black, `auto-exposure=1`) — DONE 2026-07-11
- **What:** running two or more `ftrace ... -device gpu` processes at once could make
  one emit a **completely black** image logging `auto-exposure=1` (the fallback used
  when the 99th-percentile luminance comes back zero), with exit code 0 — silently.
  The symptom was non-deterministic and non-monotonic in spp (e.g. for
  `scenes/mirror_selfie.ftsl`: 512 spp OK, 2048 spp black, 4096 spp OK), depending on
  which renders happened to overlap on the GPU. Re-running the black case **alone**
  renders correctly.
- **Root cause (our bug, not a driver mystery):** `render_cuda.cu` never checked the
  return codes of its ~15 `cudaMalloc` / `cudaMemset` / download `cudaMemcpy` calls
  nor the kernel-launch status on every path. Under contention an alloc or copy fails
  (or a launch returns `unspecified launch failure`), but the code carried on and wrote
  the zero-initialized host film out as a black PNG — no error printed because the
  failing call's status was never inspected. Earlier notes claiming "CUDA still reported
  `cudaSuccess`" were wrong: the errors were there, we just weren't reading them. This
  is process-local — MMU/context isolation means it cannot corrupt another process's GPU
  state.
- **Fix (root cause):** every CUDA call in `render_cuda.cu` is now wrapped in a
  `CUDA_CHECK(...)` macro (checks the returned `cudaError_t`, and on failure prints
  `[cuda] <call> failed at <file>:<line>: <msg>` and `std::exit(EXIT_FAILURE)`), and
  every kernel launch is followed by `cudaCheckKernel(<what>)`
  (`cudaGetLastError` + `cudaDeviceSynchronize`, same loud-exit on error). This covers
  `uploadVec`, the wavefront path, and the forward/BDPT/backward render entries — so a
  failed alloc/copy/launch aborts **before** any framebuffer is downloaded or written.
- **Verified:** built and reproduced contention by running 4 concurrent
  `-device gpu -mode R` renders of `mirror_selfie.ftsl` at 2048 spp — three rendered
  correctly (valid `auto-exposure`), one hit contention and failed loudly with
  `[cuda] backward kernel failed: unspecified launch failure` + exit 1 and wrote **no**
  PNG. No silent black image.
- **Safety net removed:** the earlier `filmIsValid()` gate in `writeFilm`
  (`src/main.cpp`) — which rejected all-zero/NaN framebuffers before tone-mapping — was
  removed now that failures are caught at the source. `writeFilm` still returns a bool
  and callers still propagate a non-zero exit, but only for a genuine image-encoder
  failure. **Tradeoff:** `CUDA_CHECK` catches CUDA-reported errors, not a numerically
  produced NaN that returns `cudaSuccess`; if such a case ever appears it would tonemap
  to black again, and the fix would be a targeted NaN check, not the blanket gate.
- **Residual caveat:** contention still wastes work (one job aborts). Prefer to
  **render GPU jobs one at a time**; the difference is a contended render now fails
  loudly (non-zero exit, no PNG) instead of silently overwriting a good image with black.

### Missing/unknown light spectrum silently fell back to 6500 K white — DONE 2026-07-12
- **What:** an explicit spectrum resource that failed to load rendered the scene with
  a silent default illuminant instead of erroring. `speclib::resolveSpectrumTokens`
  returned `false` for a failed `file:`/`glass:`/`metal:`/`reflectance:`/`illuminant:`/
  `filter:` reference — but `false` also means "not my token, try the next resolver",
  so the failure cascaded to `main.cpp resolveLight`'s `return blackbody(6500.0)`. A
  typo'd path or a `-light` name with no matching preset produced a plausible-looking
  white render with exit 0 — the wrong image, no warning. (This is also why the
  measured-LED presets appeared to "work as white" on a stale binary.)
- **Root cause:** overloaded `false` return (fall-through vs. hard failure) on the
  explicit-prefix branches, plus a catch-all 6500 K fallback for unknown `-light`.
- **Fix:** explicit resource prefixes now **throw** `std::runtime_error` with a clear
  message on load failure (`spectral_library.h`); `resolveLight` throws on an
  unrecognized explicit `-light` name (the built-in `bb6500` default always resolves
  via the parametric path, so only genuine typos trip it); `main()` wraps `run()` in a
  `try/catch` that prints `error: <msg>` and exits 1. Bare, unprefixed names still
  return `false` so the resolver layering (bb<K>, gas-discharge models, illuminant
  lookup) is unchanged. Verified: valid `file:` → exit 0; missing `file:` and unknown
  `-light` → `error:` + exit 1; no-`-light` default → exit 0.

### UV coordinates (`u`,`v`) on native primitives for pattern materials — DONE 2026-07-11
- **What:** the procedural-pattern math VM now exposes the surface texture coordinates
  `u`,`v` (previously mesh-only) to expressions on **native** objects too, so a UV-space
  checker/stripe wraps *around* a sphere/box/isosurface instead of slicing through world
  space. Native `sphere {}` (equirectangular) and `quad {}` (edge-mapped) already filled
  `hit.u/v`; an `isosurface` now accepts `uv planar|spherical|cylindrical [axis=x|y|z]`
  to synthesize a wrap from its world bounds using the **same `projectUV` used for
  un-`vt`'d meshes**.
- **Implementation:** `pattern.h` — added `PatOp::VarU/VarV`, `PatCtx.u/v`, `makePatCtx`
  u/v params, `patternEval` cases, and `u`/`v` in `varOp`. `geometry.h` — hoisted
  `UvProjection`/`parseUvProjection`/`projectUV` out of `mesh.h` (both include geometry.h)
  so implicits reuse them. `implicit.h` — `Implicit.uvProj/uvAxis/uvBounds`; `intersectImplicit`
  projects UV at the hit when enabled. `scene.h` — `patCtxFromHit` threads `h.u/h.v`.
  `ftsl.h` — `addIsosurface` parses `uv <mode> [axis=]`. GPU twins in `render_cuda.cu`:
  `DImplicit.uvProj/uvAxis/uvLo/uvHi`, device `dProjectUV`, `dPatternEval`/`dPatternScalarAt`
  thread `u,v` (and the DF_EXPR field call passes 0,0). Demo: `scenes/uv_native.ftsl`.

### `camera_curve` block (spline fly-through with variable speed) — DONE 2026-07-11
- **What:** a new top-level `camera_curve "name" { point … [frames N] [density <ρ> |
  density_at <t> <ρ> …] [look tangent|look_at|curve+look_point] [closed] [exposure_lock] … }`
  expands into N CamSpec frames whose eye rides a **Catmull-Rom spline** through the
  `point` control points (interpolating — passes through each). Placement is either a
  fixed `frames` count (uniform arc length) or a **density** (cameras per unit length)
  that can vary via `density_at` keyframes — the camera's *speed*: high density = many
  closely-spaced frames = slow dwell; low density = fast. This answers the original
  "how do you specify camera speed as a separate curve" question: density ρ(t) is
  integrated over arc length to a cumulative count C, and camera i is placed by
  inverting C at the target fraction. Orientation: travel tangent (default), a fixed
  `look_at`, or a second `look curve` (its own spline sampled in step).
- **Implementation:** `ftsl.h` `catmullRomAt()` (interpolating spline eval, open clamps
  neighbours / closed wraps) + `addCameraCurve()` (dense arc-length + density sampling,
  cumulative-count inversion for placement, tangent/fixed/curve look) + dispatch entry.
  Reuses the `camera_path`/`camera_orbit` machinery (shared CamSpec, `base<NNN>` naming,
  `pathGroup`/`exposureLock`, multi-camera render loop). Validated on CPU
  (`scraps/curve_test.ftsl`, 3 frames — eye rides the spline, holds the look_at). Same
  GPU caveat as `camera_orbit`: one camera per launch, frames render sequentially (fine).

### `camera_curve` animatable orientation + lens tracks — DONE 2026-07-12
- **What:** `camera_curve` gained the two remaining animatable degrees of freedom it was
  missing — **orientation roll** and **lens properties**. `roll[_at]` banks the camera
  about its view axis (the third orientation DOF beyond eye position and look target);
  `fov_at` / `zoom_at` / `fstop_at` / `focus_at` animate vertical field of view, focal
  multiplier, f-number and focus distance. Each is a keyframe track over the normalized
  timeline `t ∈ [0,1]` (piecewise-linear, flat-clamped at the ends — same idiom as
  `density_at`), or a constant via the bare keyword. Lens *projection*/fisheye stays a
  discrete whole-flight mode (not a continuous track), documented as such.
- **Implementation:** `ftsl.h` — new `ScalarTrack` helper (sorted `{t,v}` keys +
  flat-clamped `sample()`), `rotateAboutAxis()` (Rodrigues) for the roll bank, and a
  static `deriveCameraOptics()` factored out of `readFilmExposure()` so the per-frame loop
  can re-derive focal/fov/aperture/film-distance from the sampled tracks starting from the
  authored base values (no double-apply of zoom). `addCameraCurve()` parses the tracks,
  samples them at each frame's timeline `fr`, re-derives optics when any lens track is
  active, and applies roll to `up` about the final view direction. Demoed in
  `scenes/crystalloop.ftsl` (roll banks into the oval's turns; fov widens for the crystal
  plunge). Note: `fstop`/`focus`/DoF only bite in the physical catch modes (A/C); in the
  pinhole splat mode B the aperture is virtual, so roll/fov/zoom are the visible tracks.

### `camera_orbit` block (turntable / fly-around for MP4s) — DONE 2026-07-11
- **What:** a new top-level `camera_orbit "name" { center radius [height] [axis] frames
  [start_deg] [sweep_deg] [look_at] [exposure_lock] … }` expands into N CamSpec frames
  whose eye rides a circle around `center` (the default look_at), for stitching an orbit
  MP4. A full 360° sweep samples `i/frames` (frame N == frame 0, seamless loop); a partial
  sweep spans endpoints via `i/(frames-1)`. Reuses the `camera_path` machinery (shared
  CamSpec, per-frame naming `base<NNN>`, `pathGroup`/`exposureLock`, the multi-camera
  render loop + `_<name>` file naming).
- **Implementation:** `ftsl.h` `addCameraOrbit` (basis vectors U,W ⟂ axis; eye = center +
  axis·height + (U·cosθ + W·sinθ)·radius) + dispatch entry. Demo: `scenes/showcase_orbit.ftsl`
  (orbit tuned so its circle flies straight through the glass sphere). NOTE: the forward
  splat models A/B share one photon set across all frames (see the shared multi-camera
  entry below), but `-mode R` is camera-anchored (it traces *from* each camera) so an orbit
  on `-mode R`/`-device gpu` renders frames sequentially — which is fine, the per-frame
  cost dominates.

### Isosurface → watertight mesh export (`-export-mesh`) — DONE 2026-07-13
- **What:** any scene's `isosurface`es can be polygonised to an OBJ (`-export-mesh out.obj`)
  for Unreal / Blender import instead of being rendered. `-mesh-res <N>` sets fineness (cells
  along the longest bounds axis); `-mesh-adaptive` / `-mesh-decimate <f>` run a
  curvature-adaptive quadric-error decimation that thins triangles on flat regions and keeps
  them dense where the surface curves. Reuses the renderer's `Implicit::eval`/`gradient`.
- **Implementation:** `src/isomesh.h` (`marchImplicit`, `decimateAdaptive`, `writeObj`);
  CLI + export hook in `src/main.cpp` (~line 2644). Runs on the CPU.
- **Watertightness (proper fix, not a hack):** started on marching **cubes** → left holes /
  non-manifold edges from its face-ambiguous cases. Replaced entirely with marching
  **tetrahedra** (Kuhn/Freudenthal 6-tet split, no ambiguous cases). Surfaces that reach the
  `contained_by` domain box were leaving an **open rim**; fixed by intersecting the field with
  the box SDF (`max(f, boxSDF)`) over a lattice padded 2 cells beyond the box, sealing them
  into flat-capped closed solids (cap normals from central differences of the augmented field).
  Decimation was introducing **non-manifold edges**; fixed with a **link-condition** test
  (collapse only when the endpoints' common neighbours are exactly the shared-face opposites)
  plus foldover rejection.
- **Container-aware caps (2026-07-13):** the mesher originally *always* box-capped at
  `im.bounds` (the AABB), ignoring the isosurface's `contained_by` shape and `open` flag — a
  sphere-container isosurface would mesh with flat AABB caps that bulge toward the box corners
  instead of a clean spherical cut. `marchImplicit` now switches the cap SDF on `im.container`
  (`Container::Sphere` → `sphereSDF(center,radius)`, else box), so the mesh boundary matches
  what the ray tracer / `klein_explorer.html` show. When the isosurface is `open`
  (`im.capped == false`) the field is left un-sealed (`augEval = im.eval`), so the surface's own
  cut edge stays an open rim rather than being force-capped. `src/isomesh.h` ~line 84.
- **Verification:** heart (genus-0) exports at V−E+F=2, 0 boundary, 0 non-manifold — uniform
  *and* adaptive (keep 30%). Gyroid TPMS shell → Euler −34 (genus-18), csg_mech → −4 (genus-3),
  metaballs → 2, all with 0 boundary + 0 non-manifold edges (Euler correctly tracks genus).
  Round-trip: re-rendering `heart_test.obj` via `-mesh` shows a clean solid heart with correct
  outward normals.

### Arbitrary-formula isosurfaces (`function` leaf, `f(x,y,z)=0`) — DONE 2026-07-11
- **What:** an `isosurface` can now contain a `function { expr "f(x,y,z)" }` leaf that
  renders the zero set of a hand-typed equation (gyroid, Goursat, etc.), distinct from
  the built-in analytic SDF leaves. The formula is compiled by the **same shunting-yard /
  postfix VM as procedural patterns** (`compilePatternExpr`, vars `x y z` and `r=|p|`).
- **Implementation:** `implicit.h` gained a `FieldOp::Expr` leaf (indices a per-`Implicit`
  `exprNodes` PatNode pool via `exprOff/exprN`); `fieldLeafSDF`/`fieldEval`/`fieldGradient`
  thread `const PatNode* exprPool`; new helpers `fieldHasExpr` +
  `estimateFieldLipschitz` (samples `|∇f|` on a 24³ grid over the container box).
  `ftsl.h` `addFunctionLeaf` + rewritten `addIsosurface` parse `function`,
  `contained_by { min max }`, optional `max_gradient` (Lipschitz bound; auto-estimated
  ×1.3 when omitted), and optional `accuracy` (march-step floor). GPU port in
  `render_cuda.cu`: `DF_EXPR` op, `DFieldNode.exprOff/exprN`, a flat device
  `fieldExprNodes` PatNode pool (`DScene::fieldExprNodes`), and `dFieldLeafSDF`/
  `dFieldEval`/`dFieldGradient` thread the pool + call `dPatternEval` for the Expr case
  (forward-declared above the field VM).
- **Why a container box is required:** an arbitrary field is **not** a signed distance
  and has no analytic AABB, so the marcher needs (1) a region to march inside and (2) a
  Lipschitz bound `L ≥ max|∇f|` so a step of `|f|/L` never overshoots the first zero.
- **Validation:** an expression sphere (`x*x+y*y+z*z-0.04`) matches the analytic `sphere`
  leaf to **RMSE 0.37/255 (0.15 %) on the same backend** (the ~12.6 CPU↔GPU RMSE is the
  inherent FP32/RNG divergence — the analytic sphere shows the same 12.58). `scenes/
  function.ftsl` (gyroid) renders correctly on both CPU and GPU. Means match ~1 %.
- **Ray-march strategy selector — DONE 2026-07-11 (follow-up):** any `isosurface` now
  picks how the ray finds the first zero crossing via `method adaptive|sample`,
  `samples <n>`, and `refine bisect|regula_falsi`. `implicit.h` gained `MarchMethod` /
  `RootRefine` enums + `Implicit.sampleStep`; `intersectImplicit` branches the march
  (fixed `sampleStep/dlen` vs the `|f|/lipschitz` adaptive step) and the refinement
  (bisection vs Illinois-safeguarded regula-falsi, tracking both bracket endpoints).
  `ftsl.h` `addIsosurface` parses the three keys (sample step = box diagonal / samples,
  else `accuracy`, else diag/256). GPU twins in `render_cuda.cu`: `DImplicit.method/
  refine/sampleStep` + the identical branch in the device `intersectImplicit`. The
  `adaptive` method (default) provably can't skip the first crossing given a correct
  `max_gradient`; `sample` needs no Lipschitz bound but can miss features thinner than one
  step. Validated: on the clean expression sphere `sample` vs `adaptive` agree to RMSE
  0.41/255 (CPU) / **0.01/255 (GPU)** — identical geometry; regula-falsi and bisection
  land on the same root. Bad `method`/`refine` values are rejected with a clear error.

### `-o foo.png` wrote a PPM (P6), not a PNG — extension was ignored [RESOLVED 2026-07-10]
- **What was wrong:** the image writer (`writePPM` in `src/main.cpp`) always
  emitted binary PPM (P6) regardless of the output extension, so `ftrace -o
  group.png` produced a file starting with `P6\n256 256\n255` but named `.png`.
- **Why it mattered:** anything that trusts the extension mis-handled the file.
  Concretely it softlocked a Claude session: reading the mislabeled `.png` sent
  it to the vision API as `image/png`; the API rejected the PPM bytes (`Image
  format image/png not supported`), and the bad image stayed pinned in the
  conversation, so *every* subsequent request 400'd until a fresh session.
- **Fix:** vendored `stb_image_write.h` (compiled once in `stb_image_impl.cpp`)
  and added `writeImage()`, which dispatches on the output extension — `.png` ->
  PNG, `.jpg`/`.jpeg` -> JPEG (q95), everything else -> PPM (P6). The tone-map
  writer was renamed `writePPM` -> `writeFilm` since it no longer only writes PPM.
  Verified: `-o x.png` / `x.jpg` / `x.ppm` produce correct magic bytes and a
  real `.png` now loads in the vision API without error.

## Limitations (by design, tracked for future work)

### BDPT connection edges through colored glass are not absorption-weighted
- **What:** Beer-Lambert interior absorption (`Material::absorb`, colored glass)
  is threaded through all three CPU transport loops via an `interior` medium
  pointer (forward `tracePhoton`, backward `radiance`, BDPT `randomWalk`). In
  BDPT this attenuates only the **subpath walk** — the camera/light subpaths that
  are traced by ray marching. A **connection edge** (`connectBDPT`, the
  deterministic segment joining a camera vertex to a light vertex) that happens
  to cross a dielectric is treated as unoccluded transmittance = 1, so it picks
  up no absorption tint.
- **Why it matters:** BDPT (mode D) images of scenes with colored glass will be
  slightly biased along light↔eye connections that pass through the glass — the
  glass tints direct-walk contributions correctly but not the connected ones.
  Forward (A/B/C) and backward (R) modes are unaffected (they have no connection
  edges), so the primary/reference renders are correct.
- **Proper fix:** accumulate optical depth along the connection ray by
  intersecting it against dielectric boundaries (or track the medium a connection
  endpoint sits in) and multiply the connection throughput by the resulting
  `exp(-sigma_a*dist)`. Deferred until BDPT-through-glass accuracy is needed.

### VCM (mode `U`): merges use a spectral XYZ estimate; connections through glass share BDPT's absorption gap; CPU-only
- **What:** Mode `U` (VCM/UPS, `src/vcm.h`) is single-wavelength like the rest of the
  renderer. Its **vertex connections** pair a camera subpath with its **own** light
  subpath, so both share one wavelength and the connection is exact (like BDPT). Its
  **merges**, however, gather light vertices from *other* paths (each carrying its own
  sampled wavelength), so — exactly like the photon map (modes `M`/`S`) — the merge builds
  the estimate directly in XYZ using `cie(λ_photon)` and the camera BSDF evaluated at the
  photon's wavelength. This is the standard spectral-photon-mapping approximation, valid
  because this renderer's MIS pdfs are wavelength-independent (diffuse cosine / glossy lobe
  densities don't depend on λ), but it is not a spectrally-exact merge.
- **Why it matters:** For strongly dispersive caustics (wavelength-dependent focusing) the
  merged contribution is approximated in XYZ rather than resolved per-wavelength, just as in
  modes `M`/`S`. Connections remain exact, so the diffuse/glossy portion is unaffected.
- **Also:** VCM connection edges that cross colored glass inherit the same absorption gap
  documented above for BDPT (the deterministic connect segment isn't Beer-Lambert weighted).
  And mode `U` is **CPU-only** — no GPU path yet.
- **Proper fix (if needed):** per-wavelength (hero-wavelength or spectral-bin) merging, and
  optical-depth accumulation along connection rays through dielectrics. Deferred until a
  dispersive-caustic VCM render demands it.

### `.nvdb` volume import (`density vdb:<file>`): dense bake, float-only, uncompressed
- **What:** `medium { density vdb:cloud.nvdb }` imports a NanoVDB FloatGrid (`src/vdbgrid.cpp`,
  the only TU that includes the vendored `NanoVDB.h`). On load the sparse grid is **baked into a
  dense float lattice** covering its active index-space bounding box (`VdbGrid`, `src/vdbgrid.h`).
  A CPU+GPU-shared trilinear sampler reads that lattice.
- **Limitations:**
  1. **Dense memory** — RAM/VRAM scales with the index-space AABB (nx·ny·nz·4 bytes), not the
     active voxel count, so a large but mostly-empty sparse volume can blow up. A safety cap
     (512 M voxels) rejects pathological grids with a clear error rather than OOM-ing.
  2. **Float grids only** — non-float builds (Fp4/Fp8/Fp16/level-set index grids) are rejected
     with a message. Convert to a float fog volume first.
  3. **Uncompressed `.nvdb` only** — Blosc/ZIP-compressed files are rejected (we deliberately
     don't vendor zlib/blosc). Re-export uncompressed (`nanovdb_convert`, or NanoVDB's
     `writeUncompressedGrids`).
  4. **Quoted path not accepted** — the FTSL value grammar takes one bareword token, so the path
     must be unquoted: `density vdb:scraps/cloud.nvdb` (no spaces). A quoted `vdb:"..."` form
     would need a small `parseValue` change to consume a trailing String.
  5. No emission/temperature grids (fire), no motion-blur/velocity grids.
- **Proper fix (if needed):** a native NanoVDB **sparse** device accessor (sample the tree
  directly on CPU+GPU instead of baking dense) to drop the memory cost and support huge volumes;
  fp16 dense option; a second float grid for blackbody emission. Deferred until an asset needs it.

### GPU parity for §1–4 features — DONE (implicits + patterns + translucency)
- **What:** the whole §1–4 CPU feature set is now ported to the GPU forward + backward
  tracers: **implicit surfaces** (5a), **procedural patterns** (5b), and **dielectric
  translucency** (5c — frosted glass = roughness lobe on both dielectric lobes; colored
  glass = Beer–Lambert `absorb` interior tint). The only remaining fallback is **GPU BDPT**
  (mode `D`), whose MIS kernel still can't reproduce per-hit pattern BSDFs or frosted/
  colored glass, so those scenes fall back to the CPU BDPT.
- **Implicit surfaces — DONE (2026-07-11, step 5a):** `render_cuda.cu` gained device
  twins `DFieldNode`/`DImplicit`, a postfix field evaluator (`dFieldEval`/`dFieldLeafSDF`/
  `dFieldGradient`, all FP64 for sphere-trace bisection robustness), and
  `intersectImplicit` (a direct port of the CPU sphere-trace). `buildUpload` flattens
  every `Implicit`'s `FieldNode` array into one device pool and uploads a `DImplicit`
  descriptor per primitive; `closestHit`/`occluded` dispatch BVH prims with index
  `>= nTris+nSph` to `intersectImplicit` (matching the CPU prim ordering). Validated:
  `scenes/implicit.ftsl` (metaballs + drilled CSG + tilted torus) renders on the GPU
  mode-R backward megakernel with GPU-vs-CPU RMSE 9.9/255 at 512 spp — *lower* than the
  cornell baseline (12.7/255) at the same settings, i.e. pure Monte-Carlo noise, no
  implicit-specific bias; mean brightness matches to ~1%.
- **Procedural patterns — DONE (2026-07-11, step 5b):** `render_cuda.cu` gained a device
  pattern VM — `DPattern` slices into a flat `PatNode` pool (`DScene::patNodes`), plus
  `dPatHash3`/`dPatValueNoise`/`dPatternEval`/`dPatternScalarAt`, exact ports of
  `pattern.h` (POD `PatNode`/`PatOp` uploaded verbatim; the field variable `f` is 0 at
  surfaces, matching the CPU). `DMaterial` carries `roughnessPat`/`filmThicknessPat`/
  `mixWeightPat`; `dMatRoughness`/`dMatFilmThickness`/`dMixResolveChild` consult a bound
  pattern (highest priority, above textures). `buildUpload` flattens `Scene::patterns` and
  sets the per-material indices. `cudaForwardSupported()` no longer gates patterns (only
  frosted/colored glass), so the forward + backward paths render them on-device; **GPU
  BDPT still falls back** for any pattern-driven material (`cudaBdptSupported`), because
  its MIS pdf/eval kernel (`kBdpt`) uses the constant params. Validated: `scraps/patval.ftsl`
  (checker/noise `mixWeightPat` spheres + a glossy `roughnessPat` sphere) GPU-vs-CPU RMSE
  12.9/255 at 512 spp → 7.2/255 at 2048 spp (falls as 1/√spp — pure noise, no bias);
  mean brightness matches to ~1%.
- **Dielectric translucency — DONE (2026-07-11, step 5c):** the device `refractOrReflect`
  gained a frosting lobe (jitter both the reflected and refracted directions by a
  power-cosine lobe when per-hit `dMatRoughness` > 1e-3, rejecting jitters that cross to the
  wrong side); `DMaterial` gained a baked `absorb[SPEC_N]` table; and an `interior` medium
  index (the dielectric material a photon/ray is inside, -1 = vacuum) is threaded through
  both forward paths — `shadeStep` (megakernel `kTrace` local + wavefront `WFState::interior`
  SoA slot) — and the backward `bkRadiance`, applying `beta/thr *= exp(-absorb(λ)·dSeg)` over
  each in-glass segment. `cudaForwardSupported()` no longer gates frosted/colored glass, so
  `-device gpu`/`auto` renders them on the forward + backward tracers; **GPU BDPT still falls
  back** (the `frostedOrColoredGlass` gate moved into `cudaBdptSupported`, alongside the
  pattern gate). Validated: `translucency.ftsl` (colored glass) GPU-vs-CPU RMSE 16.9/255 →
  9.5/255 at 512→2048 spp (falls ~1/√spp; mean matches 1.3%→0.7%); `procedural.ftsl` (frosted
  height-banded glass + patterns) RMSE 21.7 → 13.0 at 512→2048 spp (mean matches 0.06%→0.2%);
  forward megakernel vs wavefront agree on mean to 0.15%; BDPT falls back with the correct
  message; `cornell.ftsl` (clear glass) + `implicit.ftsl` still run on GPU (no regression).
- **Status:** DONE — logged 2026-07-11; implicit surfaces (5a), procedural patterns (5b), and
  dielectric translucency (5c) all landed the same day. Full §1–4 GPU forward/backward parity
  achieved; only GPU BDPT retains feature-scoped fallbacks (patterns, frosted/colored glass).

### Multi-camera renders re-trace photons per camera (RESOLVED — shared pass for modes A/B, CPU + GPU)
- **What:** Phase 3a implements multiple named `camera` blocks: one render
  invocation emits one image per camera (`scenes/twocam.ftsl`), with `-camera
  <name>` selection and per-camera film resolution + mode. But each camera is a
  **separate forward pass** — the photon set is re-traced from scratch for every
  camera (`runRender` is called in a loop in `src/main.cpp`).
- **Why it matters:** the wishlist's framing is "many cameras at once… *same
  render for efficiency*". For N cameras this is N× the photon work instead of 1×.
- **CPU shared pass — DONE 2026-07-11.** `tracePhoton` now takes a list of
  `CamTarget{Camera,Film}` and splats each diffuse/emitter/volume vertex to *every*
  camera at once (`camSplatAll`/`camSplatVolumeAll`). Because model-B `connect()` draws
  no RNG, adding cameras never perturbs the photon's RNG stream: the single-camera
  overload is bit-identical to the old path, and an N-camera shared pass reproduces N
  independent single-camera renders exactly. `renderForwardShared()` (src/main.cpp) runs
  one CPU photon trace feeding one film per camera; the multi-camera loop groups the
  eligible cameras (plain `-n`, per-frame auto-exposure) into that single pass and renders
  the rest per-camera as before (the GPU shared pass, below, later removed the CPU-only
  restriction). **Validated:** `twocam.ftsl`
  `-device cpu` shared vs. per-`-camera` solo renders are pixel-identical (max abs diff
  0, both films). The fluoro reradiation λ' is sampled once (camera-independent), and
  mode-A aperture RNG is drawn once, so those single-camera streams are preserved too.
- **GPU shared pass — DONE 2026-07-12.** The forward device code was refactored around a
  `DCamSet` (device pointer to a `DCamera` array + per-camera film/hit buffer arrays +
  `nCam`) that unifies single- and multi-camera tracing, so the ~240-line `shadeStep`
  isn't duplicated (single-camera is just `nCam==1`, bit-identical). `genPhoton`/`shadeStep`
  splat via `splatSurfaceAll`/`splatVolumeAll`; `buildUpload` was split into a scene-only
  bake plus a per-camera `bakeCamera`, and `renderForwardSharedCuda()` (render_cuda.cu)
  bakes the scene once, bakes N cameras, allocates one film/hit buffer per camera, and
  launches a single trace. **Validated 2026-07-12:** GPU model-B shared vs. single-camera
  GPU render pixel-identical (`cmp` clean); CPU model-B shared vs. single also identical;
  the megakernel and wavefront backends both drive the shared pass.
- **Mode A shared pass — DONE 2026-07-12.** Mode A (finite-lens splat) now joins the
  shared pass on both CPU and GPU. Because `connectLens()` draws an aperture sample per
  camera, an N-camera mode-A trace perturbs the RNG stream, so it is **unbiased per camera
  but matches a standalone render in distribution, not bit-for-bit** (validated: shared vs.
  standalone auto-exposure agree to noise). The A- and B-cameras run as **separate** shared
  passes (mode A draws RNG mid-trace, mode B doesn't, so their photon paths diverge). Mode
  C (forward catch) stays inherently per-camera (a photon is consumed by one aperture), and
  the dispatch (`main.cpp`) partitions eligible cameras into A- and B-groups, sharing only
  when a group has ≥2 members.
- **Status:** DONE 2026-07-12 — CPU + GPU shared pass for both forward splat models
  (A and B), validated. Mode C and the camera-anchored modes (R/D/P/V) render per camera
  by construction (documented in README: "Other modes do NOT save time with multiple
  cameras").

### Absolute-EV film sensitivity, non-square films, shared multi-camera pass
- **What (remaining):** three camera/film pieces are still open:
  1. ~~**Absolute EV / physical sensitivity.** `iso`/`shutter`/`exposure` act as a
     *relative* exposure compensation on top of the per-image auto-exposure (the
     film's radiometric scale is arbitrary). A true absolute exposure needs absolute
     light power (watts/lumens) on emitters.~~ **DONE 2026-07-11.** A `light` block
     may author an absolute emitted flux — `power <watts>` (radiometric) or
     `lumens <lm>` (photometric, via Φ_v = 683·∫SPD·V(λ)dλ with cieY as V) — on
     area/sphere/cylinder/spot/collimated lights. The FTSL loader (`absPower()` in
     ftsl.h) scales the emitter SPD so `power = emitIntegral·geomW` equals that flux;
     because photon β and the film accumulation are linear in the SPD, the film
     becomes physically linear. When any light is absolute, `Scene::absolute` is set
     and `writeFilm` swaps the 99th-percentile auto-exposure for a FIXED sensor gain
     (`ABS_EXPOSURE_GAIN`) times the photographic compensation, so scene power flows
     through un-renormalised and iso/shutter/exposure give exact absolute stops.
     Validated on `scenes/absolute.ftsl`: `power 100`→`200` brightens the diffuse
     walls ~2× (light patch clips), the `lumens` path engages absolute mode, and
     non-absolute scenes stay bit-identical (auto-exposure, `Scene::absolute=false`).
     Env lights reject `power`/`lumens` (their phase-space weight needs scene bounds;
     use `intensity`). Not yet metrologically calibrated to cd/m² — the single
     `ABS_EXPOSURE_GAIN` sets the sensor zero-point (tuned so ~100 W in a unit box at
     the neutral triple is mid-tone); relative stops and power ratios are exact.
     **Exposure-lock DONE 2026-07-11:** a `camera_path` can
     author `exposure_lock` (or the CLI `-exposure-lock` forces it across *all*
     rendered cameras) so the auto-exposure anchor is computed once from the first
     frame and reused for the rest — no dolly/zoom flicker. Implemented by an optional
     `double* lockAnchor` threaded `runRender → writeFilm` and a per-lock-group
     `std::map<int,double>` anchor in the multi-camera loop (`CamSpec.pathGroup/
     exposureLock`, `RenderCam.expGroup`); only the final converged write sets/reuses
     the anchor, so progressive intermediate saves don't poison it. Validated on
     `scenes/dolly.ftsl`: unlocked frames swing 2.0e-14→5.2e-13 (25×, visible
     flicker), locked frames all hold the frame-0 anchor 2.02e-14. Standalone cameras
     (no lock) stay bit-identical (null anchor → per-frame auto-exposure as before).
     True *absolute* EV still needs absolute emitter power (deferred, §7).
  2. ~~**Non-square films.** `film { res W H }` only uses the first value; the
     forward/backward tracers (and CUDA) allocate a square film.~~ **DONE
     2026-07-11.** `film { res W H }` (and the CLI `-r W H`) now flow resX≠resY
     through every tracer: CPU/GPU forward (A/B/C), backward (R), BDPT (D), composite
     (P), validate (V), plus checkpoint/resume (the identity guard mixes resY, so a
     mismatched height is rejected instead of silently poisoning the image). The
     camera already carried the true horizontal fov from width
     (`tanHalfX = tanHalfY·rx/ry`); only the render-entry plumbing had collapsed to a
     single square `res`. `renderForward/Backward/Bdpt/Composite`, `runRender`,
     `readCheckpoint`, and the CUDA entry points + `kBackward`/`kBdpt` kernels all
     take resX,resY now. Validated at 320×180: mode V PASSES (bulk RMSE 3%,
     firefly-dominated top-1%), CPU vs GPU auto-exposure agree (4.68e-13 vs 4.62e-13),
     resume accumulates 1M→2M correctly, and the guard rejects a 200×120→200×140
     mismatch.
  3. ~~**Shared multi-camera mode-B pass**~~ (CPU shared pass DONE 2026-07-11 — see the
     multi-camera entry above; GPU shared pass still deferred) — one photon trace
     splatting to every camera pupil.
- **Proper fix:** (1) ~~add a per-`camera_path` exposure-lock flag~~ (DONE
  2026-07-11); ~~absolute emitter power + a sensitometric film model~~ (absolute
  power + fixed-gain exposure DONE 2026-07-11; full cd/m² sensitometry still open).
  (2) ~~thread resX/resY through `renderForward`/`renderBackward`/CUDA and
  `writePPM`~~ (DONE 2026-07-11). (3) see multi-camera (CPU shared pass done).
- **Status:** OPEN (design captured) — logged 2026-07-10; **exposure-lock done
  2026-07-11**, **non-square films done 2026-07-11**, **absolute-EV done
  2026-07-11**, **CPU shared multi-camera pass done 2026-07-11**; only the GPU shared
  pass (and optional mode-A sharing) remains.
- **Done (2026-07-10, Phase 3a):**
  - `camera_path` keyframed motion — expands at load time into a sequence of
    `CamSpec` frames with piecewise-linear `eye`/`look_at` interpolation between
    sorted `key` control points; the multi-camera loop renders the generated list
    with frame-numbered output names. Grammar is numbers-only
    (`key <t> <ex> <ey> <ez> [<lx> <ly> <lz>]`). Validated by `scenes/dolly.ftsl`.
  - Physical film `size <w> <h>` (mm) → focal length `f = filmH/(2·tan(fov_y/2))`
    (metres); `fstop N` → `apertureR = f/(2N)` at load time (overrides `aperture`),
    giving physically-meaningful DOF in modes A/C. `iso`/`shutter`/`exposure` →
    relative exposure compensation `comp = exposure·(iso/100)·shutter` applied over
    the auto-exposure anchor in `writePPM`. Validated by `scenes/expo.ftsl` (ISO 200
    is exactly 2.0× ISO 100 in linear space).

### Fisheye/panoramic lenses: GPU mode-B done; unsupported by BDPT (mode D) and by the finite-lens modes (A/C)
- **What:** `projection <name>` / `fisheye` (equidistant, equisolid, stereographic,
  orthographic) is implemented on the **CPU** for the forward light tracer (modes
  A/B/C), the backward reference (R), and validation/composite (V/P), and on the
  **GPU** for the mode-B pinhole-splat path (see below). The mode-B splat importance
  is projection-correct (the camera computes the per-pixel solid angle
  `Camera::pixelSolidAngle`, replacing the rectilinear `1/(A_pix·cos⁴)`).
- **GPU mode-B fisheye (done 2026-07-11):** the device `DCamera` (`render_cuda.cu`)
  now carries a `projection` enum plus `halfFovY`/`rEdge`, with `HD dProjRadius` /
  `dProjRadiusDeriv` helpers mirroring `Camera::projRadius`/`projRadiusDeriv`.
  `DCamera::project()` branches rectilinear vs fisheye (azimuth + normalised
  `projRadius/rEdge`, no `cz>0` reject), and `pixelSolidAngle()` returns the
  projection-general solid angle (`aNorm·sinθ·rEdge²/(r·dr)`), keeping the
  rectilinear branch bit-identical (`A_pix·cos³`). `connect`/`connectVolume` divide
  by that solid angle. The device path only **splats** (never generates camera
  rays), so no inverse map (`projRadiusInv`) is needed on-device. Fisheye B/V/P now
  run on the GPU (no CPU fallback); validated GPU-vs-CPU on `scenes/fisheye.ftsl`
  2026-07-11 — the equisolid-160° `fish` frame matches CPU at RMSE 3.0/255 (8.8%
  rel, same noise floor as the rectilinear `rect` frame) and mean brightness within
  0.25%. Two gaps remain:
  1. **Finite-lens modes (A/C) reject fisheye.** A thin-lens/aperture camera cannot
     *form* a fisheye projection analytically, so modes A and C error out for a
     non-rectilinear lens (guarded in `src/main.cpp`). A true wide-angle physical
     camera needs the mesh-lens forward-catch mode (see the mesh-lens camera entry).
  2. **BDPT (mode D) rejects fisheye.** `bdpt.h`'s `cameraWe`/`cameraPdfDir` are the
     rectilinear pinhole convention (`1/(A·cos⁴)`, `1/(A·cos³)`) and feed the MIS
     balance heuristic; a fisheye lens there would give subtly-wrong weights, so
     mode D errors out for a non-rectilinear camera rather than lie.
- **Proper fix (future):** generalise the BDPT camera importance + its
  importance-sampling pdf to the projection's Jacobian so the MIS weights stay
  consistent (the pdfDir must match the actual sampling density over the fisheye
  image). The GPU mode-B port is complete.
- **Status:** OPEN (acceptable) — CPU fisheye done + validated (`scenes/fisheye.ftsl`)
  2026-07-11; **GPU mode-B fisheye done + validated 2026-07-11**; BDPT (mode D) and
  finite-lens (A/C) support deferred (the latter belongs to the mesh-lens camera).

### Physical (realistic) lens camera — backward realistic-camera formulation [IMPLEMENTED 2026-07-11]
- **What (done):** a camera can now carry a real **lens prescription** — a stack of
  spherical/planar refracting interfaces plus an aperture stop (`src/lens.h`,
  `LensSystem`). The backward reference tracer (mode R) samples a film point and a
  point on the rear element, traces that ray *through the actual glass interfaces*
  (per-wavelength Snell refraction, so dispersion → chromatic aberration is
  automatic) out into the scene, then path-traces. Depth of field, distortion,
  spherical aberration, coma, field curvature and **vignetting** (clipped/TIR rays
  contribute nothing) all emerge from the geometry — no thin-lens or projection
  model. Survivors carry a PBRT-style radiometric weight (cos⁴θ·A_rear/Z_rear²).
  Wired via an FTSL `camera { lens { … } }` block (`readLens` in `src/ftsl.h`);
  a physical lens forces the camera to mode R (`src/main.cpp`). Autofocus shifts the
  film plane with a paraxial probe (`focusAt`). Demo: `scenes/realcam.ftsl`
  (validated: the focus-plane sphere is sharp, near/far spheres blur; the `singlet`
  preset visibly softens from spherical aberration).
- **Presets & generators (`src/lens.h`):** `makeSinglet` (biconvex, lensmaker
  R=2(n−1)f), `makeAchromat` (cemented crown+flint doublet, powers split by Abbe
  numbers to cancel first-order CA); `resolveLensPreset` names: `singlet`/`biconvex`,
  `achromat`/`doublet`, `telephoto`, `wide`. All physically derived (not fabricated
  data), so focal length + achromatisation are correct by construction; dispersion at
  render time uses the real Sellmeier glass indices. Users can also paste an arbitrary
  real prescription as repeated `surface <radius_mm> <thickness_mm> <ior> <semi_ap_mm>
  [stop]` lines (PBRT lens-file convention: +radius ⇒ centre of curvature on the scene
  side; lens works in millimetres, scene in metres).
- **Sign-convention gotcha (fixed):** the geometry stores curvature as `centre =
  vertex + radius` with +z toward the scene (identical to PBRT's
  `IntersectSphericalElement`). The lensmaker/achromat generators emit radii in the
  opposite object→image convention, so their radii are **negated** at construction
  (see comments in `makeSinglet`/`makeAchromat`). Without the negation the doublet
  *diverges* and the autofocus places the sensor on the scene side (all rays miss) —
  the first-cut symptom was a fully black image.
- **Remaining gaps (OPEN, deferred):**
  1. **Backward-only.** No forward-catch (mode C-style) or forward-splat (A/B)
     realistic-lens path yet. A physical lens always renders in mode R. **GPU: DONE**
     (Plan A, 2026-07-11) — a dedicated GPU backward megakernel (GPU mode R) runs the
     physical lens as a ray-generation front-end (`kBackward` in `render_cuda.cu`, the
     lens bit-for-bit ported to `dGenLensRay`/`dLensTrace` with per-surface sensor-side
     ior baked into a device table). `-device auto`/`gpu` selects it. v1 scope
     (`cudaBackwardSupported`): no participating media, no environment light, no
     spot/collimated emitters, no fluorescence (all fall back to the CPU backward
     tracer), and ≤ `D_MAXLENS` (16) lens surfaces; textured albedo IS supported. The
     device RNG differs from the CPU, so the image is an independent noise realization
     that agrees to within Monte-Carlo noise.
  2. **Sensor mapping.** `genLensRay` maps the sensor width across the film width and
     derives the vertical extent from the output pixel aspect. Now that the film
     pipeline carries resX≠resY (non-square films, DONE 2026-07-11), rendering the
     physical lens into a film whose aspect matches the sensor (e.g. 3:2) covers the
     sensor with square pixels and no crop; a mismatched aspect still crops as before.
  3. **No inter-element flare/ghosting** (rays refract, they don't also partially
     reflect at each interface), **no enclosure/body geometry**, and the aperture is a
     circular clear-diameter clip (no shaped-iris bokeh).
  4. ~~**Not in BDPT (mode D).**~~ **DONE 2026-07-11 (Plan B, below).** The lens now
     also rides on the BDPT camera subpath (mode D), and mode P routes to it.
- **Status:** IMPLEMENTED (backward realistic camera, 2026-07-11; GPU backward
  megakernel / Plan A, 2026-07-11; **BDPT camera-subpath lens / Plan B, CPU + GPU,
  2026-07-11**). Supersedes the analytical thin-lens for "arbitrary real camera" use;
  the remaining gaps above (inter-element flare, shaped-iris bokeh) are follow-ups.

#### Plan B — realistic lens on the camera subpath of BDPT (mode D) and composite (mode P) [DONE 2026-07-11]
- **Why it's wanted:** the physical lens currently rides on **pure mode R** (backward
  everything), so it inherits mode R's weaknesses — noisy on in-frame caustics, and no
  fluorescence. The lens only ever lives on the *camera subpath*; in principle you can
  keep forward light transport lighting the scene (caustics on surfaces) while a
  backward lens ray samples that lit scene through the glass — i.e. attach the lens to
  the camera subpath of a bidirectional/composite estimator instead of forcing pure R.
  That would recover *some* of the forward tracer's caustic efficiency while keeping the
  physical optics.
- **The catch (why it's only a partial win, and deferred):** the multi-element lens map
  has **no closed-form inverse**, and both D and P need that inverse for the parts that
  would buy the forward advantage:
  1. **BDPT (mode D).** BDPT's power comes from light→camera connection strategies. The
     **t=1 strategy** (splat a light-subpath vertex directly onto the film) requires
     projecting a world point onto the sensor *through the glass stack* — the lens
     inversion. PBRT disables the camera-connection strategies for realistic cameras for
     exactly this reason. So a realistic lens in D must run with t=1 disabled: you keep
     the *scene-side* connections (a camera-subpath vertex out in the scene connects to
     light-subpath vertices), which recovers part of the caustic efficiency, but not the
     full forward win. It's a substantial, delicate, per-wavelength change layered on a
     mode D that already lacks fog / env / spot / fluorescence support.
  2. **Composite (mode P).** Worse fit: P's forward pass **splats to a pinhole** — it
     fundamentally assumes a pinhole camera. Routing that forward pass through a physical
     lens is the ill-posed forward-through-lens problem again. So P does not cleanly
     extend to a realistic lens without solving the same inversion.
- **Bottom line:** the clean, buildable step was **Plan A** — a dedicated GPU backward
  megakernel (GPU mode R) with the lens as a ray-generation front-end (**DONE**
  2026-07-11; see gap #1 above). Plan B (D and P) is genuinely more general but only a
  *partial* caustic recovery, and can't do the light→film splat through the glass.

- **DONE 2026-07-11 (the resolution that made Plan B clean and rigorous):** the t=1
  light-image splat is simply **disabled** for a lensed camera (no closed-form lens
  inverse), and the key realisation is that this needs **no lens direction-pdf** at all:
  1. `generateCameraSubpath` (`src/bdpt.h`): when `cam.hasLens()`, the first camera ray
     is generated by `Camera::genLensRay` (film point + rear-pupil point traced through
     the real glass, identical to mode R). The camera vertex sits at the ray's
     **scene-entry point** with `beta = wLens` (the lens radiometric weight), so a pure
     eye path measures `L·wLens` — matching mode R's film `add`. The camera vertex is
     flagged **`delta = true`**.
  2. `connectBDPT` t==1 branch returns 0 for `cam.hasLens()` (the splat needs the
     sensor projection we don't have). The retained strategies are the scene-side
     connections (s≥0, t≥2: pure path trace, NEE, interior light↔eye), which keep the
     forward tracer's caustic efficiency through the physical lens.
  3. **Why the lens direction pdf is unnecessary (the rigorous part):** `misWeight`'s
     camera loop runs `i = t-1 … 1` and only *adds* a hypothetical strategy's term when
     `!eye[i].delta && !eye[i-1].delta`. The i==1 term (the t=1 splat) is gated on
     `!eye[0].delta` — which is now false — so it is **excluded from the MIS sum**, and
     since the loop never reaches i==0, `eye[1].pdfFwd` (the only place the camera
     direction density would enter) never affects any *retained* ratio. The surviving
     strategies therefore still form a **partition of unity** ⇒ the estimator is
     **unbiased** regardless of the (unused) camera-ray direction pdf. `eye[1].pdfFwd`
     is seeded with the pinhole `cameraPdfDir` purely as an inert placeholder.
  - **Mode P (composite):** its forward pass splats to a **pinhole** and can't form the
    lens image, so a lensed camera in mode P **routes to the lens-aware BDPT (mode D)**
    when the scene is within BDPT scope, else **falls back to mode R** (fog / env / spot /
    fluorescence / layered — which R supports and D doesn't). Wired in `src/main.cpp`
    (`bdptUnsupportedFeature` helper shared by the mode-D gate and the P routing).
  - **GPU: DONE 2026-07-11.** The GPU BDPT megakernel (`kBdpt`) now takes the lens on its
    camera subpath too: `dGenCameraSubpath` generates the first ray via `dGenLensRay` (the
    same device lens tracer Plan A ported for GPU mode R), sets the camera vertex `beta =
    wLens` and `delta = 1`, and `dConnectBDPT`'s t==1 branch returns 0 for a lensed camera
    — a bit-for-bit mirror of the CPU path. The DCamera lens is already uploaded by
    `buildUpload`. So `-mode D -device gpu` on a lensed scene runs entirely on-device; the
    old CPU-force guard in `src/main.cpp` is removed.
  - **Validation** (`scenes/realcam.ftsl`, achromat 50 mm f/2.8, full-frame): mode D vs
    mode R with the same lens agree on absolute radiance and the residual is **pure Monte-
    Carlo noise**. CPU-D↔GPU-D↔R all agree (median per-pixel ratios 0.987–1.010 @ 256–512
    spp). Unbiasedness: high-spp GPU-D vs R gives median **1.0003** with the ratio IQR
    narrowing [0.905,1.096] → [0.967,1.035] as spp goes 512 → 8192 (≈√16 narrowing), and
    the auto-exposures converge (GPU-D 1.07e-11 vs R 1.08e-11). No bias, CPU or GPU. Mode P
    lens routing (→ D) and out-of-scope fallback (→ R, tested with `-fog`) both verified;
    lensless mode D/P unchanged (cornell regression).
  - **Remaining follow-up:** the true t=1 splat through an approximate lens inverse
    (PBRT-style exit-pupil sampling) for the extra light-tracing strategy — optional, since
    scene-side connections already recover the main forward win.
- **Status:** DONE 2026-07-11 (CPU + GPU BDPT + composite routing).

### Texturing is base-color only, `use_mesh`/quad UVs only (Phase 3b partial)
- **What (done 2026-07-10):** a `texture "name" { file … encoding srgb|linear
  filter nearest|bilinear wrap repeat|clamp|mirror }` block loads an image into
  `Scene::textures` (`src/texture.h`); `reflect texture:<name>` on a `diffuse`
  material binds it (`Material::reflectTex`); per-vertex UVs live on `Tri`
  (`src/geometry.h`), auto-generated for quads and read from OBJ `vt` when a mesh
  sets `uv use_mesh`; each texel is Jakob-Hanika–upsampled to a reflectance
  spectrum (coefficients precomputed at load via `Texture::buildReflCoeff`, bilerped
  + sigmoid-evaluated per hit through `diffuseReflectance()`). Shared by the forward
  tracer and the backward reference. Image formats: PNG/JPG/BMP/TGA + Radiance
  `.hdr` via the vendored stb_image (`src/third_party/stb_image.h`, compiled once in
  `src/stb_image_impl.cpp`), plus built-in PPM/PFM. Validated by
  `scenes/textured.ftsl` (quad) and `scenes/uvmesh.ftsl` (mesh) — the checker maps
  with correct orientation (blue band at v≈1/top, yellow at u≈0/left) and spectral
  colour; a PNG copy of the checker renders bit-identically to the PPM (RMSE 0.0),
  confirming the stb sRGB decode + orientation.
- **Remaining [needs engine work]:**
  1. **UV projections.** ~~Only `uv use_mesh` (OBJ `vt`) and quad corners exist.~~
     **PARTLY DONE 2026-07-11:** the analytic projections `uv planar|spherical|
     cylindrical [x|y|z]` (spec §9.2) are now synthesized at load time from the
     world-space vertex AABB (`UvProjection`/`projectUV` in `src/mesh.h`, parsed in
     `src/ftsl.h`), normalised to [0,1] across the mesh so the map wraps once by
     default. Because they fill the same per-vertex `Tri.uv{0,1,2}` slots as
     `use_mesh`, both tracers **and the GPU** interpolate them with no shading change
     (validated on `torus.obj`, which carries no `vt` — the checker maps onto the
     torus via the spherical projection; `scraps/uvproc.ftsl`). **DONE 2026-07-11:**
     `triplanar` — unlike the three analytic maps it can't be baked into per-vertex
     UVs (it blends three world-axis planar samples per hit, weighted by |n|^4), so it
     lives on the bound material as `Material::triplanarScale` (world->texture repeat
     rate) and is evaluated per hit in `Texture::reflectanceTriplanar`, called from
     `diffuseReflectance()` (CPU, shared by forward/backward/BDPT) and the device twin
     `dTexReflTriplanar` from `dDiffuseRho()` (GPU) — the two agree by construction.
     Parsed from `mesh { uv triplanar [<s>|scale=<s>] }` in `src/ftsl.h`. Validated by
     `scenes/triplanar.ftsl`: CPU vs GPU exposures match to 3 digits (4.87e-13 vs
     4.88e-13) and the box-projected checker is visually identical on both backends.
     **Parser gotcha fixed:** the scale/axis argument must be a bare number or a
     `key=val` param — a bareword `scale`/axis letter starts a *new* statement and
     clobbered the mesh's own `scale` transform (this caused an all-black render while
     the torus ballooned to 4x and occluded the box; the same fix now applies to
     `uv planar axis=x`).
  2. **Non-albedo parameters.** ~~A texture can only bind to diffuse `reflect`.~~
     **DONE 2026-07-11 (roughness + film-thickness maps, both backends):** `glossy`
     takes `roughness texture:<name>` (grayscale = roughness directly) and `thinfilm`
     takes `film_thickness_map texture:<name>` (0..1 profile × nominal `film_thickness`
     nm). Bound in `src/ftsl.h` via `bindScalarTexture`; sampled per-hit by
     `materialRoughness`/`materialFilmThickness` (`src/scene.h`) → `Texture::scalarAt`
     (mean of linear RGB). All three CPU tracers (forward/backward/BDPT) and the GPU
     forward path (megakernel + wavefront, via `dMatRoughness`/`dMatFilmThickness` +
     `dTexScalarAt` over an uploaded per-texel `gray` array) use it. **MIS
     correctness:** the CPU BDPT threads the hit UV through `bsdfPdf`/`bsdfF` so the
     sampled and evaluated roughness match; the GPU BDPT does not, so `cudaBdptSupported`
     rejects roughness/thickness-map scenes → CPU BDPT fallback. Validated by
     `scenes/scalarmap.ftsl`: CPU vs GPU forward exposure 7.3e-13 vs 7.29e-13, mean
     agrees to <0.1%, signed diff ~0.04% (unbiased). **Also DONE 2026-07-11 (mix
     blend-mask):** a 2-child `mix` takes `weight_map texture:<name>` — the map value t
     at the hit is the probability of child 0 (child 1 = 1-t, no absorption), a spatial
     A/B blend. `Material::mixWeightTex` + `mixResolveChild` (scene.h), threaded through
     all three CPU tracers and the GPU forward path (`dMixResolveChild`). Mix selection
     is a stochastic RR pick that doesn't enter the BSDF pdf, so it's unbiased in every
     tracer; the GPU BDPT mix-pick still uses constant weights, so masked mixes take the
     CPU-BDPT fallback (`cudaBdptSupported`). Validated by `scenes/maskblend.ftsl`.
     **Still deferred:** a map on `ior` (spatially-varying refractive index — rare, and
     better served by measured dispersion data than a grayscale map; low priority).
  3. ~~**GPU.** Textured scenes force the CPU tracer.~~ **DONE 2026-07-11:** the
     forward CUDA path now ports textured diffuse reflectance. `buildUpload()` uploads
     each texture's per-texel Jakob-Hanika coeff table (`DTexture`, flattened `3*w*h`)
     plus per-tri UVs (`DTri.uv0/1/2`); `intersectTri`/`intersectSphere` interpolate
     the hit UV into `DHit.u/v`; `dTexReflAt()` is the device twin of
     `Texture::reflectanceAt` (wrap + nearest/bilinear + sigmoid), used via
     `dDiffuseRho()` in the `shadeStep` diffuse/fluoro branches. `cudaForwardSupported()`
     no longer rejects `reflectTex >= 0`. Validated GPU-vs-CPU on `textured.ftsl` and
     `uvmesh.ftsl`: energy matches (absorbed 0.7066 vs 0.7068) and images agree to
     within Monte-Carlo noise (RMSE ~6/255); the wavefront backend matches the
     megakernel. The BDPT kernel (mode D) still lacks a textured vertex, so
     `cudaBdptSupported()` explicitly rejects textured scenes → they use the CPU BDPT.
  4. **Indexed-spectral palettes** (§9.3). ~~An index image + name→spectrum palette —
     not implemented.~~ **DONE 2026-07-11 (CPU):** a `texture { ... palette { <idx>
     spectrum:<name> ... } }` block resolves each index to a named reflectance spectrum
     at parse time (`Texture::palette`, `src/ftsl.h addTexture`). The red channel,
     quantized to 0..255, selects an entry per texel — nearest only (indices are
     categorical, never bilerped) via `Texture::paletteReflectanceAt`. No JH upsampling
     (palette entries are arbitrary measured spectra used directly), so `buildReflCoeff`
     skips palette maps and the GPU forward path (`cudaForwardSupported`) rejects them →
     CPU fallback. Validated by `scenes/palette.ftsl` (a 4-index swatch chart). **Limit:**
     8-bit index channel → ≤256 entries; 16-bit index maps are future work.
- **Status:** OPEN (acceptable) — base-color texturing + stb image import done
  2026-07-10; GPU port done 2026-07-11; **analytic UV projections (planar/spherical/
  cylindrical) + triplanar box projection done 2026-07-11 (CPU + GPU)**; **non-albedo
  roughness + film-thickness maps + mix blend-mask done 2026-07-11 (CPU all tracers +
  GPU forward; GPU BDPT falls back to CPU)**; **indexed-spectral palettes done 2026-07-11
  (CPU)**. Only an `ior` map (item 2) remains deferred (rare; low priority).

### Light shapes: sphere + spot done, HDRI environment deferred (Phase 3c partial)
- **What (done 2026-07-10):** two new emitter shapes on the shared `Emitter`
  (`src/scene.h`), both routed through forward + backward + CUDA:
  - **Spherical area light** — `light sphere { center x y z  radius r  spd … }` —
    `shape = EmitterShape::Sphere`, `area = 4·π·r²`. Forward and backward both call
    `Emitter::samplePoint()` (uniform surface point + outward normal; quad draws are
    byte-identical so quad scenes stay bit-identical). The FTSL loader also drops an
    emissive sphere into the geometry (mirroring the area-light quad). Validated by
    `scenes/spherelight.ftsl` (mode V PASS at 60M photons / 1024 spp; CPU==GPU).
  - **Point spotlight** — `light spot { origin … dir … inner_angle d  outer_angle d
    spd … }` — `shape = EmitterShape::Spot`, a cone with a cubic-smoothstep
    penumbra. Geometric weight is the falloff-weighted solid angle `spotOmega =
    π·(2−cosᵢ−cosₒ)`, so `power = emitIntegral·spotOmega`, peak intensity/SPD = 1.
    Forward samples a direction uniformly in the outer cone and reweights beta by
    `falloff·Ω_outer/spotOmega`; backward connects straight to the light point with
    a cone-falloff weight (`I(ω)·cosθ/d²`, no area/light-cosine). No emissive
    geometry (a point). Validated by `scenes/spotlight.ftsl` (mode V PASS at 200M
    photons / 4096 spp → 3.8% RMSE; CPU==GPU energy).
  - `finalizeEmitters()` now keys the power law / combined wavelength sampler off a
    per-emitter `geomWeight()` (area·PI for surfaces, spotOmega for spots); the
    area/sphere branch keeps the exact `emitIntegral·area·PI` expression so those
    renders stay bit-identical (verified: cornell FTSL==C++==pre-3c hash).
- **Constant environment done (2026-07-10, increment 1a):** `light env { spd … }`
  registers a uniform infinite emitter (`shape = EmitterShape::Env`,
  `geomWeight = envGeom = 4·π²·R²` with `R` the scene bounding-sphere radius set in
  `Scene::build()` from the BVH root AABB). Forward emission spawns each photon from
  a disk of radius `R` perpendicular to a uniformly-sampled sphere direction (joint
  pdf `1/envGeom` → exactly analog `beta = emitIntegral·envGeom`); backward adds
  `L(λ)·invPdfλ` on ray-miss; a per-pixel background pass (`addEnvBackground`) supplies
  the directly-viewed sky in forward mode B. Validated by `scenes/envlight.ftsl`
  (mode V: forward converges to backward on a **unit** radiance scale — best-fit
  s→1).
- **GPU constant env done (2026-07-10, increment 1b):** the device forward kernel
  now emits env photons (`DEmitter::shape == 3`) from the scene bounding sphere
  (`DScene::sceneCenter`/`sceneRadius`) exactly like the CPU path, and the
  directly-viewed sky is added by the backend-agnostic `addEnvBackground()` pass in
  `main.cpp` — so `cudaForwardSupported()` no longer rejects `envIndex ≥ 0` and env
  scenes run on the GPU. Verified CPU==GPU on `envlight.ftsl` mode V (both best-fit
  s≈0.97, absorbed≈0.129, RMSE≈58% at 8M — deltas are independent RNG streams).
  - **Absolute-radiance We fix (same change):** the model-B pinhole importance was
    normalizing by the *whole* image-plane area (`imagePlaneArea()`), making the
    forward tracer measure `radiance / (resX·resY)` — an arbitrary global constant
    that modes V/P best-fit away and auto-exposure hid. This blocked compositing the
    (true-radiance) env background with the (scaled) photon surface illumination.
    Fixed by normalizing by the **per-pixel** image-plane area
    (`Camera::pixelPlaneArea() = imagePlaneArea()/(resX·resY)`) in `connect()` /
    `connectVolume()` on **both** CPU (`render.h`) and GPU (`render_cuda.cu`). Now
    forward measures absolute radiance (mode V/P best-fit s → ~1). Displayed outputs
    are unchanged (a global scale is invisible after auto-exposure; mode-P
    `fwd·invF/s` and mode-V RMSE are scale-invariant); verified cornell mode V still
    PASSes (s 5.8e-5 → 0.98) and CPU==GPU film scale holds.
  - **Forward env is high-variance (acceptable limitation):** the env photon
    emission is isotropic over 4π, so in an open scene the vast majority of photons
    escape without hitting geometry (~87% on `envlight.ftsl`). Combined with
    single-wavelength spectral spikes, forward mode-B env images are heavily
    chromatic-noisy and need large `-n` to converge (mode V RMSE falls as 1/√N with
    s≈1 — variance, not bias: 58%@8M → 27%@60M). Clean env images come from the
    **backward** reference (mode R). A future variance reduction would importance-sample
    the emission toward the actual geometry (not just the bounding sphere) and/or
    trace multiple wavelengths per photon (hero-wavelength); deferred.
  - **Mode P + env: sky background — DONE 2026-07-11.** The directly-viewed sky is
    now composited in `renderComposite()` (mode P), not just modes B/V. The pixel
    classifier became three-way — SPEC (specular-first → backward layer), SKY (camera
    ray escapes an env scene → env radiance) and DIFF (everything else → forward
    layer) — and the env radiance (`envXYZForDir`, already in the composite's
    display-radiance units) is written on SKY pixels. Critically, **SKY pixels are now
    excluded from the forward→backward scale fit**: they are measured by env radiance
    directly (forward film ≈ 0, backward film = full bright sky), so including them
    dragged the best-fit `s` toward 0 — exactly the bias mode V avoids by adding the
    sky to `fwd` before its `compareFilms` fit. Verified on `envlight.ftsl` mode P:
    excluding sky pixels restores s 0.27 → 0.957 (matching mode V's ~0.97), the sky
    renders behind the geometry, and a non-env specular scene (`group.ftsl`) is
    unaffected (s 0.965, no env line, DIFF/SPEC split unchanged).
- **Image-based HDRI environment (2026-07-10, increment 2a — DONE, CPU):** `light env
  { file "sky.hdr"  rotate deg  intensity s }` registers an equirectangular (lat-long)
  environment. `src/envmap.h` (`EnvMap`) loads the map (via the existing `Texture`
  loader — `.hdr`/`.pfm`/LDR), upsamples each texel to a physical emission spectrum
  `L(λ) = scale·S_JH(chroma)(λ)·illum(λ)` (Jakob-Hanika chroma fit × normalized 6504 K
  illuminant, PBRT RGB-illuminant convention; `scale` carries HDR brightness), and
  builds a 2D luminance CDF (`Distribution2D`: marginal rows × conditional cols,
  `sin θ`-weighted) for importance-sampled directions. Wired through `Scene`
  (`envMap` shared_ptr, `addEnvLight(map)`, direction-dependent `envRadiance(dir,λ)` /
  `envXYZForDir(dir)` / `sampleEnvDir`), `render.h` (forward emission importance-samples
  the direction and reweights the flat power by `L(dir,λ)/(4π·pdf_ω·meanSpd(λ))` — a
  factor that is exactly 1 for a constant env, so those stay bit-identical), `backward.h`
  (miss term uses the escape direction), `main.cpp` (`addEnvBackground` uses per-texel
  spectral XYZ), and `ftsl.h` (`file`/`rotate`/`intensity` parse). The emitter power +
  wavelength CDF use the map's `sin θ`-weighted mean radiance spectrum. Validated by
  `scenes/envmap.ftsl` + `scenes/sky.pfm` (mode V: best-fit s→~0.95 and climbing with
  samples — the residual is Monte-Carlo variance from the sun glow, not bias;
  forward/backward auto-exposure agree to ~3%; energy conserves). Constant env
  (`envlight.ftsl`) stays **bit-identical** (mode-V scale 0.971252, unchanged).
  - **Increment 2b — DONE (2026-07-10):** backward env **NEE** at every diffuse and
    fog-scatter vertex (`neeEnv`/`neeEnvVolume` in `backward.h`): sample `ω` from the
    map's luminance CDF, shadow-ray past the scene bounds, and **MIS-combine** (balance
    heuristic) with the BSDF-sampled continuation that reaches the sky on a ray miss.
    The miss term is added at full weight only on a camera/specular arrival and MIS-
    weighted otherwise (gated on `specularArrival`), so nothing is double-counted;
    `envMap->pdf(d)` provably equals `sample()`'s reported pdfW, so the weights sum to
    1 (unbiased — verified: `envmap.ftsl` mode-V scale stays ~0.947 at 60M/512, energy
    conserves, residual broadly distributed). All env-NEE work is gated on
    `scene.envIndex >= 0`, so non-env scenes keep a **bit-identical** RNG stream /
    backward image (cornell mode V unchanged).
  - **Increment 2c — DONE (2026-07-10):** GPU port of the lat-long sampler
    (`render_cuda.cu`). The host flattens the EnvMap into device buffers — per-texel JH
    `coeff`/`scale`, the mean `avgCoeff`/`avgScale`, and the 2D luminance CDF (marginal
    `Distribution1D` over rows + one conditional per row) — and the device gets
    `dReflAt`/`dSample1D`/`dEnvSample`/`dEnvTexel` so the `shape==3` emission branch
    importance-samples the map and reweights beta by `L(dir,λ)/(4π·pdfW·avgSpd)`. The
    reweight's shared illuminant cancels in `L/avgSpd`, so no illuminant table is
    uploaded. `cudaForwardSupported()` now returns true for image env; the constant-env
    device path is untouched (`sc.env.scale == nullptr`). Verified: GPU vs CPU forward on
    `envmap.ftsl` agree — energy conserves (sum/emitted=1.0, escaped 0.8893 vs 0.8894),
    mean RGB within ~0.5%, auto-exposure 53.5 vs 53.8; constant env + all other GPU
    scenes unchanged.
- **Deferred (still future):**
  1. **HDRI env** — image-based environment lighting (`light env { file … }`) is fully
     done: increments 2a (CPU forward + backward miss/background), 2b (backward env-NEE
     with MIS), and 2c (GPU forward port) are all complete. Original 7-step plan below,
     all steps done, kept for reference:
     **Concrete plan (each sub-step
     independently
     buildable + validatable):**
     1. *Scene bounding sphere.* Add `Vec3 sceneCenter; double sceneRadius;`
        computed in `Scene::build()` from the BVH root AABB (`center`, `0.5·diag`).
        The env disk/emission and the "to infinity" shadow-ray length key off this.
     2. *Env data + importance sampler.* Store the lat-long map as linear RGB +
        per-texel JH coeffs (reuse `Texture`). Precompute a 2D luminance CDF
        (marginal over rows, conditional over columns, each row weighted by
        `sin θ`) → `sampleEnvDir(u1,u2, pdfω)` and `envPdf(ω)`. `envRadiance(ω,λ)`
        = `reflAt(coeff(ω), λ)` scaled by an intensity factor.
     3. *Backward first (easiest to validate in mode R).* In `radiance()`, on
        `!h.valid` return `L + thr·envRadiance(ray.d,λ)·invPdfLambda` when
        `specularArrival` (direct/mirror view of the sky). Add env NEE at diffuse
        vertices: sample `ω~envPdf`, shadow-ray to `sceneRadius`, add
        `f·envRadiance(ω,λ)·cosSurf·invPdfLambda/envPdf(ω)`. Fold the env into the
        combined wavelength sampler `g(λ)` (its geomWeight ≈ `π·sceneRadius²·avgLum`).
     4. *Forward emission.* New branch in `tracePhoton`: `shape == Env` emits a
        photon FROM the sky — importance-sample `ω~envPdf`, pick a point on the disk
        of radius `sceneRadius` perpendicular to `-ω` tangent to the bounding sphere,
        fire along `-ω`, `beta = envPower · envRadiance(ω)/(avgLum·envPdf(ω))`.
        `envPower = π·sceneRadius²·∫envRadiance dω` feeds the selection CDF like any
        other emitter's `power`.
     5. *Mode-B background.* In forward mode B a camera ray isn't traced, so the sky
        isn't directly visible via `connect()`. Add a per-pixel background pass:
        for each pixel, project the pinhole ray, and if it escapes, splat
        `envRadiance(dir,λ)` (spectrally integrated) — a cheap deterministic add,
        analogous to the existing direct-emitter `connect`.
     6. *CUDA.* Upload the env RGB+coeff tables + the marginal/conditional CDFs;
        port `sampleEnvDir`/`envPdf`/`envRadiance` and the forward disk-emission
        branch. Escaped photons already terminate; only emission + (optional) the
        mode-B background pass need device code. Keep `cudaForwardSupported()`
        returning true for env scenes once ported (else fall back to CPU).
     7. *Validation.* `scenes/envlight.ftsl` (a diffuse box open to the sky):
        mode V forward-vs-backward RMSE < 5%; energy conserves; CPU==GPU. A constant
        (single-colour) env is the smallest first milestone — it exercises steps
        1/3/4/5/6 with a trivial step 2 (uniform pdf), so land that before the full
        image-based 2D CDF.
  2. **Sphere-light importance sampling.** *(DONE 2026-07-10.)* The backward
     reference's sphere NEE now does cone/solid-angle importance sampling of only
     the visible cap toward the receiver (`Emitter::sampleSphereCone`, PBRT's
     `Sphere::Sample`): sample `cosθ` uniformly in `[cosθmax, 1]` about the
     centre-to-point axis (`sinθmax = r/dc`), find the near intersection, and weight
     in solid-angle measure with `pdfW = 1/(2π(1−cosθmax))` — so no draws land on
     the far, self-occluded, back-facing hemisphere. A receiver inside the sphere
     (`dc ≤ r`) falls back to uniform `samplePoint`. Applies to both surface
     (`neeLight`) and fog-vertex (`neeVolume`) NEE. Quad lights are untouched and
     keep a bit-identical RNG stream. Validation: `spherelight.ftsl` mode V best-fit
     scale → 0.9997 (unbiased) at 80M/1024 spp, RMSE 2.5% bulk; sphere+`-fog 0.5`
     scale 0.988; cornell (quad) unaffected. Only the backward reference changed —
     the forward tracer emits sphere photons omnidirectionally as before, so the
     GPU/CUDA path is unaffected.
  3. **Spot penumbra sampling.** The forward spot samples uniformly in the outer
     cone then reweights by falloff, so photons in the dark penumbra edge carry
     small weights (mild variance). Exact CDF sampling of the smoothstep band would
     be lower-variance but needs a quartic inverse; uniform+reweight is correct.
- **Status:** OPEN (acceptable) — sphere + spot done 2026-07-10; **constant
  environment (`light env { spd … }`) done 2026-07-10 (increments 1a CPU + 1b GPU)**
  incl. the absolute-radiance We fix and on-device env emission; **image-based HDRI
  (`light env { file … }`, 2D luminance CDF + per-texel JH spectral upsampling) fully
  done 2026-07-10 (increments 2a CPU forward+backward miss/background + 2b backward
  env-NEE with MIS + 2c GPU forward port)**; **sphere-light cone importance sampling
  in the backward reference done 2026-07-10**; only spot penumbra CDF sampling (item
  3) still deferred.

### Built-in artificial-light SPDs: F-series transcribed from memory, discharge lamps are models not measurements
- **What (added 2026-07-11):** `src/lights.h` now provides spectral envelopes for
  artificial light sources, wired into `resolveLight()` / `preset:<name>`:
  - **CIE F-series fluorescents** — `fluorescentF2/F7/F11()` (`f2`/`cool-white`,
    `f7`/`daylight-fl`, `f11`/`triphosphor`), tabulated 380–780 nm at 5 nm via
    `sampledSPD()` → `tabulatedSpectrum()`.
  - **Gas-discharge lamps** — `sodiumHigh()` (`hps`/`sodium`), `sodiumLow()`
    (`lps`/`sodium-low`), `mercuryVapor()` (`mercury`/`hg`), `metalHalide()`
    (`metal-halide`/`mh`).
  - **CCT-tuned phosphor LED** — `ledCCT(kelvin)` via the `led<K>k` name (e.g.
    `led4000k`).
- **The honesty caveats (tech debt, not a bug):**
  1. ~~**The F2/F7/F11 tables were transcribed from the canonical CIE 15 illuminant
     data by hand/from memory.**~~ **VERIFIED & CORRECTED 2026-07-11; FULLY
     EXTERNALIZED 2026-07-12.** The baked tables were diffed against the authoritative
     CIE 15:2004 F-series (via colour-science), which caught a real bug:
     `fluorescentF7()`'s tail (685–780 nm) was wrong (it wiggled back up to 4.34 at
     765 nm instead of decaying smoothly). F7 was corrected; F2/F11 already matched
     exactly. **As of 2026-07-12 the baked `fluorescentF2/F7/F11()` tables are DELETED
     from `src/lights.h`** — the measured SPDs live only in
     `data/illuminant/{f2,f7,f11}.csv` and `resolveLightPreset()` resolves the
     `f2`/`f7`/`f11`/`cool-white`/`daylight-fl`/`triphosphor` names through
     `resolveTabulatedIlluminant()` (spectral_library.h) at load time. `preset:f2`
     and `spd file:data/illuminant/f2.csv` now load the *same file* — verified
     identical by `scenes/measured_spd.ftsl`, and the loader round-trips the old baked
     values exactly (e.g. F2 P(545 nm)=24.88). This sub-item is DONE.
  2. **The sodium / mercury / metal-halide entries are deliberately *illustrative*
     spectroscopic models, not per-lamp measurements** — correct line positions and
     plausible relative strengths (from spectroscopy references) over analytic
     continua, tuned to give the right visual cast. They are not a specific
     manufacturer's lamp and are not radiometrically calibrated. Same intended
     upgrade path: swap for measured SPDs when the data-file loader lands.
- **Proper fix:** the spectral asset library now exists (`data/<category>/<name>` +
  `src/spectral_library.h`, DONE 2026-07-12), and `resolveLightPreset()` already reads
  the F-series from `data/illuminant/`. To upgrade the discharge lamps to measurements:
  drop a measured lamp SPD (LSPDD / LICA-UCM, see `data/README.md`) into
  `data/illuminant/` (e.g. `hps.csv`, alias `sodium`) — it then resolves by name with
  no rebuild, and can shadow the analytic model. Only the discharge-lamp measured CSVs
  remain to be fetched; the preset-reads-CSV wiring is now generic and done.
- **Status:** OPEN (acceptable, reduced) — library + F-series data done and the
  built-in F-series presets now read the CSVs at load time; only discharge-lamp
  measurements remain (the analytic line models stay as the default until then).

### Built-in material presets: skin/soil & iridescent recipes are representative (metals + most natural curves now measured)
- **What (added 2026-07-11):** `src/materials.h` adds built-in common-material data
  and recipes, plus expanded glasses in `src/spectrum.h`:
  - **Metals** — `metal:Au|Ag|Cu|Al|Cr|brass` reflectance R(λ) (`metalGold()` etc.).
    Au/Ag/Cu/Al/Cr now computed from published measured n,k (Johnson & Christy 1972,
    Rakić 1995/1998; CC0 via refractiveindex.info) with
    `tools/ri_nk_to_reflectance.py` — see resolved item below. `brass` is still an
    alloy fit (no single canonical dataset).
  - **Glasses/crystals** — `glass:` gained `silica`/`fused-silica`/`quartz`,
    `sapphire`, `diamond`, `water`, `ice`, `acrylic`/`pmma`, `polycarbonate`/`pc`
    (Sellmeier for glass/crystal, `cauchy()` fits for water/ice/plastics), unified
    behind `resolveGlassIor()`.
  - **Natural diffuse** — `reflectance:leaf|skin|skin-dark|snow|soil|brick|concrete`.
    `leaf`/`snow`/`brick`/`concrete` are now measured USGS splib07 samples (see item 2);
    `skin`/`skin-dark`/`soil` remain representative shapes.
  - **Whole-material recipes** — `material { preset <name> }` via
    `resolveMaterialPreset()`: metals (glossy), glasses (dielectric), and iridescent
    `soap-bubble`/`oil-slick`/`anodized-ti`/`morpho`/`beetle`/`nacre`.
- **The honesty caveats (tech debt, not a bug):**
  1. *(RESOLVED 2026-07-11)* Metal reflectances were hand-transcribed and coarse;
     now regenerated from the canonical measured n,k datasets (Johnson & Christy
     1972 for Au/Ag/Cu at native sample points, Rakić 1995/1998 for Al/Cr at 20 nm)
     via `tools/ri_nk_to_reflectance.py`, which computes normal-incidence
     R=((n-1)²+k²)/((n+1)²+k²). Colours re-validated on diffuse-lit spheres
     (gold/copper/salmon/neutral-silver/neutral-chrome). Only `brass` remains an
     alloy fit — no single canonical dataset exists for it.
  2. *(PARTLY RESOLVED 2026-07-11)* Most `reflectance:` natural curves are now
     measured samples from the USGS Spectral Library v7 (splib07, public domain,
     DOI 10.5066/F7RR1WDJ), extracted with `tools/splib_to_reflectance.py`:
     `leaf` = fresh green Oak leaf (ASD, 10 nm — captures the real chlorophyll dip
     and steep red-edge), `snow` = melting snow mSnw01a, `brick` = medium-red
     building brick GDS353, `concrete` = light-grey road concrete GDS375 (all ASD).
     Still representative shapes: `skin`/`skin-dark` (human skin isn't in splib) and
     `soil` (splib's Soils chapter is mineral mixtures/sand, not a generic loam).
     Real vegetation/skin still vary enormously sample-to-sample.
  3. **The iridescent recipes (`soap-bubble`, `oil-slick`, `anodized-ti`, `morpho`,
     `beetle`, `nacre`) are physically-motivated film/stack *configurations*, not
     measured spectra** — layer indices/thicknesses tuned to give the right colour
     family, not matched to a specimen.
- **Renderer note (not a preset bug):** metal/glass presets are *specular*, so they
  show colour only through what they reflect/transmit. In a closed pinhole (mode B)
  box with nothing bright around them they read near-black — expected forward-tracer
  behaviour (same as the existing `mirror`/`glossy`/`dielectric` types); use mode A,
  an environment light, or surrounding geometry to see them. Their reflectance data
  is correct (verified by putting the same `metal:` spectra on a diffuse surface).
- **Proper fix (remaining):** three offline generators exist — `tools/csv_to_table.py`
  (generic CSV→`table`), `tools/ri_nk_to_reflectance.py` (refractiveindex.info
  n,k→reflectance), and `tools/splib_to_reflectance.py` (USGS splib07→reflectance) —
  plus, as of 2026-07-11, a *runtime* `file:<path>` loader so a reflectance CSV can be
  bound directly (`reflect file:data/reflectance/skin-light.csv`) without re-baking
  source. Metals and leaf/snow/brick/concrete are done. Remaining debt is finding
  measured samples for `skin`/`skin-dark` (a skin-optics dataset, e.g. NIST JRES
  122.026) and `soil` (a loam/dirt reflectance, e.g. ECOSTRESS/ISRIC) and overwriting
  the placeholder files in `data/reflectance/`, plus optionally validating the
  iridescent recipes against specimens.
- **DATA EXTERNALIZED 2026-07-12:** the baked `metalGold()..metalBrass()`,
  `reflectanceLeaf()..reflectanceConcrete()` tables (`src/materials.h`) and the
  `iorBK7()..iorPolycarbonate()` dispersion functions (`src/spectrum.h`) are **deleted
  from source**. They now live as data files — `data/metal/*.csv`,
  `data/reflectance/*.csv`, `data/glass/*.glass` (Sellmeier/Cauchy coefficients) — and
  are loaded by the same-named resolvers (`resolveMetalReflectance` /
  `resolveNaturalReflectance` / `resolveGlassIor`) now living in
  `src/spectral_library.h`. Every call site is unchanged (only the data source and
  includes moved); a category is a directory of files keyed by lowercased filename
  stem + `# aliases:` header, so new metals/glasses/reflectances drop in with no
  rebuild. The `sellmeier()`/`cauchy()`/`tabulatedSpectrum()` evaluators and the
  iridescent recipes stayed native (algorithms, not data). Verified: standalone loader
  test round-trips n_d (BK7 1.5168, SF10 1.7283, water 1.333) and R(λ) (Au R(700)=0.970)
  from the files, matching the old baked values.
- **RECIPES EXTERNALIZED (bundles) 2026-07-12:** the whole-material `preset` recipes
  and named light presets are now **composite asset bundle files**, not baked C++.
  `data/material/*.material` (soap-bubble, oil-slick, anodized-ti, morpho, beetle,
  nacre) and `data/light/*.light` (sun, daylight/d65, incandescent/a, led, led-warm)
  group a `type` + several spectral envelopes (`ior`/`substrate_k`/`spd`…) + intrinsic
  scalars (`film_ior`/`film_thickness`, `layer <n> <k> <nm>` rows) under one name.
  `resolveMaterialBundle` (materials.h) / `resolveLightBundle` (lights.h) interpret the
  manifest; spectrum-valued fields reuse the scene language's primitive vocabulary via
  a new shared `speclib::resolveSpectrumTokens`. So the tuned iridescent layer stacks
  are now DATA (retune/extend with no rebuild) while the interference/Abeles/Fresnel
  evaluators (render.h) and the LED/gas-discharge line models (lights.h) stay native.
  `resolveMaterialPreset` reads a bundle first, then a generic metal→glossy /
  glass→dielectric convention so bare primitive names still resolve with no file. The
  bit-for-bit faithfulness holds: `const N` == the old `iorConstant(N)`
  (`[N](double){return N;}`), so the bundles reproduce the baked recipes exactly.
- **Status:** OPEN (acceptable, much reduced) — metals + 4 natural curves are now
  measured data, and ALL spectral data AND the whole-material / named-light recipes now
  load from `data/` files rather than baked source; skin/soil and iridescent recipes
  remain representative. All presets load on CPU==GPU and render the right colours.

### Colored-LED light bundles + `filter` gel material — DONE 2026-07-12
- **Colored LEDs (data only).** A direct-emission LED die is a single narrow band, and
  the light-bundle vocabulary already had `gaussian center=… sigma=…`, so seven colored
  LEDs (`data/light/led-royal-blue`…`led-deep-red`) are pure-data `.light` bundles — no
  native code. Representative InGaN/AlInGaP peaks + FWHMs (sigma = FWHM/2.355). Measured
  die SPDs (slightly asymmetric) can drop into `illuminant/` later (pending in
  data/README). Verified `preset:led-red` renders pure red.
- **`filter` material (option A).** New `MatType::Filter`: a thin non-scattering absorber
  (colored gel / Wratten). The photon passes straight through (direction unchanged) and
  survives with probability T(λ) = `transmit`(λ), else absorbs — Russian roulette on the
  transmittance, β unchanged, unbiased. Specular straight-through so it makes no camera
  connection (like clear glass — it colors what's behind it). Threaded through EVERY
  tracer: forward CPU (`render.h`), forward GPU (`render_cuda.cu` `D_FILTER`), backward
  CPU (`backward.h`) + GPU, and BDPT CPU (`bdpt.h`) + GPU (delta vertex, throughput
  ×= T). `type filter` in FTSL (ftsl.h) reads `transmit`; `parseMatType` adds it for
  bundles. New `filter/` data category + `filter:<name>` token + `resolveFilterTransmittance`.
- **Data (RESOLVED 2026-07-12; full set digitized 2026-07-12):** `data/filter/wratten-*.csv`
  is now the **complete 84-filter Kodak Wratten set**, each **digitized from the numeric
  percent-transmittance tables** in *Kodak Wratten Filters for Scientific and Technical Use*,
  22nd ed. (pub. B-3), 400–700 nm at 10 nm, 31 samples (book dashes = negligible → 0).
  Files are named `wratten-<n>` (letter suffix lowercased) with `# aliases:` headers keeping
  the old descriptive names (red-25, deep-red-29, orange-21, yellow-12, green-58, blue-47,
  deep-blue-47b, …) resolvable; the 7 old descriptive-named CSVs were deleted. Text-layer
  pages were coordinate-extracted (word x → column, y → row); the eight image-only pages
  (28,29,31,33,35,38,41,47) were visually transcribed from high-DPI crops. Extraction/
  transcription scripts live in `scraps/` (`wratten_extract.py`, `wratten_manual.py`,
  `wratten_all.py`; gitignored). Renders confirm correct per-filter tints. Finer spacing/
  more gels can drop into `filter/` later (Rosco `.sed` / LEE / CRC), no rebuild.

### Full physical `layered` material [IMPLEMENTED 2026-07-11]
- **What:** both the FTSL `type mix` material (stochastic per-photon pick among named
  child materials, weights ≤ 1, remainder absorbs — Phase 2d, `scenes/mixmat.ftsl`,
  mode V PASS, CPU==GPU) and the richer physical `layered` material (spec §3.2) now
  ship. `layered` is a specular *coat* interface over a weighted *body*: on each hit a
  photon reflects off the coat with probability R, else it enters and one body lobe is
  chosen from a `mix`-style `layer "name" weight` list (leftover weight absorbs). Coat R
  + body weights partition the photon, so the surface is energy-consistent (validated:
  `scenes/layered.ftsl`, forward mode B and backward mode R, `absorbed+escaped=1.0`,
  residual 0).
- **Coat models:** `coat { reflectance … }` selects the interface reflectance:
  `fresnel` (plain dielectric Fresnel from the coat `ior`, rises toward grazing —
  clearcoat sheen), `thinfilm` (Airy multiple-beam reflectance from `film_ior` /
  `film_thickness` over the body index — soap-bubble iridescence), or `manual` (a flat
  `specular` fraction). The coat reflection is a glossy lobe about the mirror direction
  (`roughness` / `roughness_map`, lossless), and `film_thickness_map` gives spatially
  varying iridescence just like a `thinfilm` material.
- **Constraints of `mix`/`layered` (by design):** children/body lobes must be non-mix,
  non-layered materials (nesting rejected by the parser to keep resolution single-step
  and the CUDA CDF bounded); the CUDA path supports ≤ 8 child lobes (more → CPU
  fallback); a mix containing a fluorescent child is forward-only but as of 2026-07-11
  runs on the GPU forward path (the device fluoro port resolves the mix child before
  dispatch and the `D_FLUORESCENT` `shadeStep` branch handles it — see the
  GPU-fluorescence note below); a textured child is likewise fine.
- **Scope / fallbacks:** `layered` is CPU-only (forward + backward). GPU forward/backward
  fall back to the CPU tracer (`cudaForwardSupported` rejects any Layered material, like
  indexed palettes); BDPT (mode D) refuses a Layered scene with a clear message
  (`render it with mode B/P or mode R`) rather than dropping the surface via the
  randomWalk `default: terminate`. A per-lobe BDPT vertex strategy for `layered`
  (mirroring the forward split) is possible future work but not required for the
  reference/forward validation paths.
- **Status:** DONE — `mix` 2026-07-10; `layered` 2026-07-11 (CPU forward + backward).

### Backward reference tracer now validates fluorescence [RESOLVED 2026-07-11]
- **What (was):** `src/backward.h` had no Fluorescent case — a fluorescent material
  fell through to the Diffuse branch, so modes R (reference) and V (validate)
  silently mis-rendered `-scene fluoro`; fluoro scenes were forward-only.
- **Fix applied:** added a bispectral reradiation case to `BackwardRenderer::radiance`
  — the unbiased backward adjoint of the forward tracer's `fluoroInteract()`:
  1. **Elastic channel** — diffuse NEE (+ RR continuation) at the output wavelength
     with the small elastic base `rho(lambda)`, exactly as before.
  2. **Fluorescent DIRECT NEE** — a *second* excitation wavelength `lambdaIn` is drawn
     from the combined emission distribution (reusing `scene.emitSampler` /
     `invPdfLambda`, so multi-light SPDs weight correctly). The lights are connected at
     `lambdaIn` with a reradiation "albedo" `aEff(lambdaIn)*Q` (shared `fluoroWeights`),
     and the result is tinted by the emission colour at the OUTPUT wavelength
     `gOut = (M(lambda)/∫M) * invPdfLambda` — the `invPdfLambda` factor deconvolves the
     camera-path wavelength-sampling density so the reradiated colour follows `M(lambda)`
     and not the light SPD used to sample `lambda`.
  3. **Indirect excitation** — a single stochastic continuation splits between an elastic
     bounce at `lambda` (prob `rho`) and a wavelength-switched bounce to `lambdaIn`
     (prob `pF ~ gOut*aEff*Q`, throughput `*= wFluo/pF`), so light that bounces before
     exciting the dye (light→wall→dye→camera) is captured without double-counting the
     direct term (`specularArrival=false` suppresses the emission-on-hit term).
- **Validation (mode V, forward mode-B vs backward, `-scene fluoro`):** best-fit
  backward→forward scale = **0.996** (≈1, i.e. the two agree on ABSOLUTE scale — a wrong
  bispectral normalisation would not), residual 94% firefly-concentrated (variance not
  bias), and the bulk RMSE (ex. top-1%) scales as **1/sqrt(N)**: 2.05% at 40M/400spp →
  **1.02%** at 160M/1600spp (4× samples ⇒ exactly 2× reduction), which a transport bias
  could not produce. `-checkfluoro` (deterministic sampler/branch/Stokes-shift self-test)
  is retained as a fast complementary check.
- **Remaining (BDPT / mode D):** bidirectional bispectral fluorescence (a wavelength
  change inside a light↔camera connection needs hero-wavelength MIS, à la Mojžík et al.
  2018) is still deferred — mode D refuses fluoro with a clear message pointing to modes
  B/P (forward) or R (backward). The backward reference now covers fluorescence
  validation, so this is low priority.
- **Status:** RESOLVED 2026-07-11 for modes R/V; BDPT (mode D) bispectral vertices
  remain future work.

### RESOLVED: Backward reference tracer now validates participating media (fog)
- **What (was):** `src/backward.h` ignored `scene.medium` — its camera rays didn't
  sample volume free-flight or in-scattering, so `-fog` with modes R/V would have
  compared a volumetric forward image against a vacuum backward image (garbage
  residual). Fog was therefore forward-only and never set in refMode.
- **Fix applied:** added a homogeneous-medium path to `BackwardRenderer::radiance`
  that mirrors the forward tracer exactly:
  1. **Free-flight sampling** competes with the surface hit each bounce
     (`tMed = -ln(1-u)/sigma_t`; on `tMed < dSurf` a volume collision occurs).
  2. **`neeVolume()`** — phase-function next-event estimation at the collision
     vertex: the surface BRDF/cosine are replaced by the single-scattering albedo
     and the HG phase function `hgPhase(dot(wIn, wi), g)`, with fog transmittance
     `exp(-sigma_t*dist)` on the shadow ray (the backward mirror of the forward
     `connectVolume`). The phase angle uses reciprocal conventions to the forward
     side, both equal to the physical `dot(prop_in, prop_out)`.
  3. **Analog scatter/absorb** continuation: survive with prob = albedo, then
     `sampleHG` a new direction (throughput unchanged); otherwise absorb.
  4. **Beer-Lambert on surface NEE too:** `neeLight` now attenuates its shadow ray
     by `exp(-sigma_t*dist)` (took a new `lambda` parameter).
- **Validation (mode V, forward vs backward, identical fog):** the best-fit
  backward→forward scale agrees to ~4 sig figs across no-fog / fog-g0.3 /
  fog-rayleigh, which a transport bug could not produce. The raw-linear residual is
  firefly-dominated (top-1% pixels hold 77–95% of it) from the unbounded 1/dist^2
  light connection, so full RMSE plateaus but the **bulk RMSE (ex. top-1%) scales
  as ~1/sqrt(N)**, proving variance not bias: for fog g=0.3 alb=0.85 at 256^2, bulk
  RMSE went 7.67% (120M/800spp) -> 4.53% (480M/3200spp) [1.69x ~ ideal 2x], with
  firefly concentration held constant at ~86% and the 4x run reporting PASS.
  No-fog bulk RMSE is 1.2% (95% firefly-concentrated). The firefly-vs-bias
  diagnostic (residual concentration + bulk RMSE) was added to `compareFilms` in
  `src/main.cpp` specifically to make this distinction rigorous.
- **Status:** RESOLVED 2026-07-10. `-fog` can now be combined with modes R/V.
  `-checkfog` (deterministic transmittance / HG mean-cosine / phase-normalization
  self-test) is retained as a fast complementary check.

### Model A redefined as the finite-lens physical camera (GPU port landed)
- **What:** as of 2026-07-11 **mode A is the physical finite-lens camera**: a finite
  aperture + thin lens + film imaged by next-event estimation of the pupil
  (`Renderer::connectLens`/`camera.h::lensImage`). It replaces the old contact-sensor
  "mode A" (a flat film wall — no aperture, so it integrated the whole hemisphere per
  pixel and could not form an image; retired). Mode B is now the pinhole (`aperture→0`)
  limit; mode C is the brute-force forward-catch oracle A is validated against
  (matching framing/DOF/scale — auto-exposure within ~2.5% at equal aperture/focus).
- **GPU port (done):** the CUDA `DCamera` now has `lensImage` (thin-lens `u' = u − ρ/f`,
  shared with the mode-C `catchPhoton`) and `kTrace` runs device `connectLens`/
  `connectLensVolume` splats under `camMode 'A'` — emitter-direct, diffuse-vertex, and
  fog-in-scatter, mirroring the CPU. `renderForward` selects `camMode 'A'` on the GPU
  and `runRender` treats mode A as a GPU-forward mode. Validated vs CPU (Cornell, 192²,
  wide aperture 0.25/focus 2.2, 40M photons): energy to 4 sig figs, auto-exposure 2.99e-8
  GPU vs 2.95e-8 CPU (1.4%), image RMSE 2.6/255 (pure MC noise — the GPU is an
  independent realization); the tiny-aperture A→B pinhole limit holds on-device (sharp,
  RMSE 3.16/255 vs mode B). The old contact-sensor GPU `deposit` path was removed with
  the port. Mode A remains **rectilinear only** (a real fisheye needs a wide-angle lens
  element the single thin-lens can't form) — a fisheye+A/C camera is rejected, and a
  fisheye lens still falls back to the CPU even on `-device gpu` (see GPU-fisheye entry).

### GPU backend (`-device gpu`) covers forward camera models A/B/C
- **What:** the CUDA backend (`src/render_cuda.cu`, `renderForwardCuda`) implements
  the finite-lens next-event splat (A), the pinhole splat (B), and the finite-aperture
  thin-lens forward catch (C), selected by the `camMode` parameter. It is used for
  `-mode A/B/C` and the forward pass of `-mode V`. It also tracks per-pixel photon
  **hit counts** on-device (a `d_hits` buffer incremented in `filmAdd`, downloaded into
  `Film::hits`) — matching the CPU `Film::add`. This fixed a latent bug where the GPU
  never populated `hits`, so the progressive `~X% noise` graininess estimate (and the
  new `-noise` stop) read a constant **0%** for any `-device gpu` render. The backward
  tracer is now on-device too: **mode R has its own GPU backward megakernel** (`kBackward`,
  Plan A, 2026-07-11 — including the physical mesh-lens as a ray-gen front-end), and the
  **mode-P composite reuses it for its camera-side layer** (`renderComposite` calls
  `renderBackwardCuda` when `cudaBackwardSupported`), so both of P's layers run on the GPU
  within the backward-GPU scope. Only scenes outside that scope (fog/env/spot/collimated/
  fluorescence) — and mode V's backward reference, kept on the CPU by design as a stable
  ground truth — still use the CPU backward tracer. **Fluorescence is now ported on-device (done
  2026-07-11):** each Fluorescent material bakes its excitation spectrum
  (`DMaterial.fluoAbsorb`) and emission-SPD CDF (a flat `fluoCdfAll` slice, per-material
  `fluoCdfOffset/N/step`); the `shadeStep` `D_FLUORESCENT` branch splats the elastic
  channel at lambda and the glow channel at a Stokes-shifted lambda' (`sampleFluoEmit`)
  with albedo `aEff*fluoYield`, then stochastically continues (elastic / reemit / absorb)
  exactly like `render.h`'s `fluoroInteract`. `shadeStep`'s `lambda` became a reference
  (Stokes shift mutates it), and the wavefront `kWfShade` now writes `st.lambda[slot]`
  back on the continue branch. `cudaForwardSupported()` no longer rejects Fluorescent
  materials. Validated GPU-vs-CPU (fluoro scene, mode B and mode A): energy conserves
  (`sum/emitted=1.0`, absorbed 0.7034 vs 0.7036), mean RGB matches to ~0.1/255, image
  RMSE 3.5/255, wavefront matches the megakernel, and `-checkfluoro` PASSes. The BDPT
  kernel (mode D) has no fluorescent vertex strategy, so `cudaBdptSupported()` explicitly
  rejects fluorescence → CPU fallback (mode D already refuses fluoro scene-wide anyway).
- **Why acceptable / validated:** model B is the default and the one mode V
  validates. The kernel `kTrace` mirrors `Renderer::tracePhoton` exactly and gates the
  camera-specific work on `camMode`: emitter/diffuse/in-scatter `connectLens` runs for
  A, the pinhole `connect`/`connectVolume` for B, and `catchPhoton` (thin lens
  `u' = u - rho/f`) for C. (Mode A validation is in the redefinition entry above; the
  historical **Mode A bullet below** records the now-removed contact-sensor GPU
  `deposit` path, not the current finite-lens next-event mode A.)
  Validation vs CPU (Cornell, 128²):
  - **Mode B:** image RMSE ≈ 0.85/255 at 200M photons (pure MC noise); `-mode V
    -device gpu` PASSes vs the backward reference (bulk RMSE 4.17% ≈ CPU 4.22%);
    ~14× speedup (400M @256²: 153s CPU → 10.9s GPU on an RTX 4090).
  - **Mode A (retired contact-sensor GPU path — historical):** energy report matched
    to 4 sig figs (sensor 0.3298 vs 0.3299); image RMSE scaled as √N — 11.18/255 @40M
    → 5.11/255 @200M (5× photons, ideal 2.24×, measured 2.19×), proving variance not
    bias. This validated the old flat-film-wall mode A, which has since been replaced
    by the finite-lens next-event camera (validated separately above); the device
    `deposit` path it exercised has been removed. Retained only as a historical record.
  - **Mode C:** energy report matches to 4 sig figs; with a wide aperture (0.25,
    focus 2.2) the caught fraction matches exactly (sensor=0.0058) and per-image
    auto-exposure agrees (1.60e-8 vs 1.59e-8). Image RMSE scales as √N —
    15.70/255 @200M → 8.10/255 @800M (4× photons, ideal 2×, measured 1.94×) —
    proving variance not bias. (The CPU is deterministic across runs, so GPU is the
    only independent noise realization; the small default aperture is catch-starved
    and its tone-mapped RMSE is dominated by per-image auto-exposure.)
- **Spectral baking:** device materials/fog sample each `std::function` Spectrum into
  a fixed 96-entry table over [360,830] nm with linear interpolation (`SPEC_N=96`).
  Smooth reflectances/Sellmeier indices make this accurate to within MC noise; a
  pathologically spiky spectrum would need a finer table. CIE CMFs are ported
  analytically (no table).
- **Precision (mixed FP32/FP64, default float transport):** consumer GeForce GPUs run
  FP64 at ~1/64 the FP32 rate, so the megakernel computes all geometry/BRDF/spectral
  transport in a compile-time `Real` scalar (`float` by default) while accumulating the
  film and energy in `double` (`atomicAdd` on `double*`). Build with
  `-DFTRACE_GPU_FP32=OFF` for a full-FP64 device path (bit-closer to the CPU, far slower
  on GeForce; sensible on datacenter cards or for precision debugging). The CPU renderer
  is always `double` and remains the ground-truth. Float-safe self-intersection epsilons
  (`RAY_EPS=1e-4`, `DET_EPS=1e-6`) replace the FP64 `1e-6`/`1e-9`. **FP32 validated vs
  FP64 CPU (Cornell, RTX 4090):** energy conserves exactly (`sum/emitted=1.000000`,
  residual=0 on A/B/C/V — no self-intersection leak from the float epsilons); fractions
  converge to 4 sig figs (retired contact-sensor mode A sensor 0.3298 vs 0.3301, mode C 0.0059 vs 0.0058); mode
  V PASSES (bulk RMSE 2.89%, firefly-dominated); ~14× faster than FP64 (400M @256² in
  0.76s vs 10.9s). The DVec3 3-arg ctor deliberately keeps `double` params so host
  brace-init from `double` Scene coords is a widening (legal) conversion, never
  narrowing; spectral/CDF tables stay `double` (host-baked, tiny, cached).
- **Portable build (multi-arch) + HIP-ready:** `-DFTRACE_CUDA_ARCH=` selects the device
  arch set — `native` (default; the local GPU only, fast builds), `all-major` (a
  redistributable fat binary: one cubin per major arch + forward-compatible PTX so newer
  GPUs JIT at load), `all`, or an explicit `"75;86;89"` list. The device kernel is
  written in the portable CUDA/HIP subset (`__global__`/`__device__`, grid-stride, double
  `atomicAdd`, `<<<>>>` launches); the only vendor-specific surface — the host runtime API
  (device query, malloc/memcpy/memset/free, error strings, synchronize) — is isolated
  behind a compat block at the top of `render_cuda.cu` that maps `cuda*` → `hip*` under
  `-DFTRACE_USE_HIP`/`__HIP_PLATFORM_AMD__`. Porting to AMD ROCm is therefore a
  build-system change (compile this one file with `hipcc`), not a code rewrite. **CUDA is
  the supported GPU backend today; HIP is a near-drop-in future target (untested — no AMD
  hardware here).**
- **Proper fix — DONE (2026-07-11):** the backward tracer is now ported to CUDA — mode R
  has its own GPU backward megakernel (`kBackward`, including the physical-lens ray-gen
  front-end), and the mode-P composite reuses it for its camera-side layer. (The device
  fluorescence path and textured-albedo path are done too — see above.) Remaining CPU-only
  backward work: scenes outside the backward-GPU scope (fog/env/spot/collimated/
  fluorescence) and mode V's reference (kept on the CPU by design).
- **Status:** OPEN (acceptable) — logged 2026-07-10; A/C, mixed-precision FP32, portable
  multi-arch build, and the HIP compat layer added same day. Requires a CUDA toolkit at
  configure time; without one the project builds CPU-only and `-device gpu` warns and
  uses the CPU.

### GPU scaling path: megakernel vs. wavefront (IMPLEMENTED — both backends ship)
- **Status:** both backends now exist. The **megakernel** (`kTrace`) is the default; the
  **wavefront** (streaming) backend is opt-in via `-wavefront`. They share the exact same
  device physics — `genPhoton()` (emitter sample + direct connect) and `shadeStep()` (one
  bounce: medium/catch/material dispatch) are `__device__` functions called by both — so
  only the *scheduling* differs. Because of that shared code, adding the wavefront left the
  megakernel bit-for-bit identical (validated: cornell/materials mode B/C images and energy
  reports unchanged after the extraction refactor).
- **Context:** the **megakernel** is one `kTrace` launch where each thread runs an entire
  photon path (emit → bounce loop → connect/catch/deposit) start to finish. This is the
  right choice for *this* renderer's typical case: an RTX 4090 has huge register/occupancy
  headroom, Cornell-class scenes are shallow, and a single kernel keeps all state in
  registers with no round-trips to global memory. It hits ~500M+ photons/s in FP32.
- **The known limitation (thread divergence):** in a megakernel, threads in a warp that
  take different material branches (a dielectric refraction next to a diffuse bounce next
  to a grating), or that terminate after wildly different path lengths, **serialize** —
  the warp runs at the speed of its slowest/most-divergent lane, and finished lanes sit
  idle while others keep bouncing. The megakernel also carries the register footprint of
  *every* material's code path in *every* thread, capping occupancy. Both effects get
  worse as scenes gain more material variety and deeper paths, and they bite harder on
  smaller GPUs (fewer SMs / less latency-hiding to absorb the idle lanes).
- **What we shipped (wavefront / path-regeneration):** the tracer is split into two
  coherent stages that alternate over a **persistent pool of photon slots** (SoA state:
  ro/rd/beta/lambda/rng/bounce/alive/hit, W = min(N, 1M) slots): **extend** (`kWfExtend`,
  one `closestHit` per live slot) then **shade** (`kWfShade`, one `shadeStep` per live
  slot). A warp's threads therefore execute the same stage together instead of diverging
  on per-photon path length. When a path terminates (or hits the bounce cap), its slot
  **immediately regenerates a fresh photon** (`wfSpawn` claims the next index from an
  atomic budget counter) — path compaction by regeneration — so SIMD lanes stay full until
  all N photons are traced. The host loop (`wavefrontTrace`) reads a live-slot counter each
  pass and stops when the pool drains. **Phase 2 not yet done:** a *sort/compaction by
  material* before the shade stage would additionally kill BSDF-branch divergence (every
  thread in a warp running the same material) — that's the remaining coherence win over
  what's implemented, and the natural next step if a material-diverse scene proves
  shade-divergence-bound.
- **The cost:** the wavefront reads/writes the whole photon state to global memory every
  bounce and launches two kernels per pass, so it's *not* a win for shallow, uniform scenes
  on a big GPU — the megakernel's register-resident state wins there. Measured on an RTX
  4090, materials mode B, 400M photons: megakernel 0.79 s vs wavefront 1.60 s (~2× slower),
  exactly the expected regime. The wavefront's payoff is on divergent / deep-path scenes and
  smaller GPUs; energy conservation and image agreement (to within Monte-Carlo noise) hold
  across every scene tested (cornell, materials A/B/C, spotlight, envlight, thin-film,
  multilayer, mix, fog).
- **Re: "wavefront helps divergent scenes AND small GPUs" (notes/todo.txt question):** it's
  *both*, and they're related. (1) *Divergent scenes* — many materials and/or highly
  variable path lengths — benefit from the per-material sort (kills branch divergence) and
  compaction (kills path-length divergence). (2) *Small GPUs* benefit because they have
  less occupancy/latency-hiding headroom to paper over idle lanes and high per-thread
  register pressure, so keeping warps coherent and full matters more there. A big GPU on a
  shallow uniform scene (our current case) is the one regime where the megakernel clearly
  wins, which is why we ship it first.
- **Decision:** the megakernel stays the default/recommended path for the scenes this
  renderer targets (shallow, uniform, big GPU); the wavefront ships as an opt-in
  (`-wavefront`) for the divergent/deep-path/small-GPU regime. Remaining optional work: the
  Phase 2 per-material shade sort (above), and a heuristic for `-device auto` to pick the
  backend by scene material variety + path depth rather than always defaulting to the
  megakernel.

## Performance

### RESOLVED: Diffuse-mesh renders were ~60× slower per photon (degenerate BVH)
- **Symptom:** The Cornell + diffuse torus (`-mesh torus.obj`) traced at
  ~34–37 µs/photon, vs ~0.55 µs/photon for the Cornell + glass sphere. 3M photons
  took ~112s. Made mesh scenes impractical.
- **Root cause (found by instrumentation):** The BVH build was leaving giant
  leaves. Added a `-bvhstats` diagnostic (nodes/leaf-tests per ray + leaf-size
  histogram) which showed the 16k-tri torus BVH had only **245 nodes / 123 leaves,
  max leaf = 9334 primitives**, and each ray did **~1091 leaf primitive tests**.
  The culprit was the SAH termination in `Bvh::buildRecursive` (`src/bvh.h`):
  `if (bestSplit < 0 || bestCost >= leafCost) makeLeaf();`. Object-SAH on a
  ring-like shape hits a top-level pathology — splitting a torus through its
  centre yields two C-shaped halves whose AABBs each nearly equal the *whole*
  box, so every candidate split has cost ≈ the leaf cost. The greedy "only split
  if it lowers SAH" test therefore gave up immediately at the top and dumped
  most of the mesh into one leaf. (Path length was a red herring: the diffuse
  Cornell walls dominate bounce count regardless of the mesh.)
- **Fix applied:**
  1. **BVH (the real fix):** use SAH only to *choose* the split plane, and always
     recurse down to `LEAF_SIZE`, falling back to a median (`nth_element`) split
     when SAH finds no usable partition. Result: 245→**10425 nodes**, max leaf
     9334→**4**, leaf-tests/ray 1091→**0.7**. Torus 3M render 112s→**1.3s**
     (0.43 µs/photon — now *faster* than the glass-sphere reference).
  2. **Front-to-back traversal ordering** in `traverseClosest` (descend nearer
     child first; cull children against `tMax` at push time). Minor on its own
     (~8%) but correct and keeps the win robust.
  3. **Russian roulette** for Diffuse/Mirror/Glossy in `Renderer::tracePhoton`
     (`src/render.h`): terminate with prob `1-reflectance`, keep `beta` unchanged
     on survival. Unbiased; caps path length (residual now 0.0000) and removed the
     now-dead `betaCutoff`. `maxBounce=32` kept as a hard safety cap.
- **Validation:** `-checkbvh` still reports 0 mismatches on cornell/materials/
  prism/torus; energy conserves exactly (`sum/emitted=1.000000`) on all scenes.
- **Status:** RESOLVED 2026-07-10. Logged & fixed same day.

## Scene interoperability / importers

- **Mitsuba XML → FTSL: DONE 2026-07-11** (`tools/mitsuba_to_ftsl.py`). Mitsuba
  0.6/2/3 is also a spectral PBR renderer, so the mapping is nearly 1:1
  (perspective/thinlens sensor → camera, diffuse/conductor/roughconductor/
  dielectric/plastic/blendbsdf → materials, area/constant/envmap/point/directional
  emitters → lights, rectangle/cube/sphere/obj shapes with full `to_world`
  transforms, `<ref>`/`<default>` resolution). Validated on a converted Cornell
  scene (glass + gold spheres, colored walls, area light) rendering correctly in
  modes B and D. **Because Blender exports to Mitsuba XML via `mitsuba-blender`,
  this is also the Blender→FTSL path.**
  - **Known approximations (flagged with `# WARN:` in output):** roughdielectric →
    smooth dielectric (no rough transmission in FTSL); plastic/roughplastic →
    glossy (diffuse+specular coat merged); bumpmap/normalmap dropped to base BSDF;
    mask opacity ignored; `.ply`/`.serialized` meshes emitted as `mesh` lines but
    ftrace's loader is OBJ-only (convert first); mesh area-emitters have no FTSL
    equivalent (emitted as lit geometry, emission dropped); mesh `to_world` with
    rotation/shear only partly expressible (translate+scale + euler).
  - **Possible follow-ups:** map `.ply` via an auto OBJ conversion; emissive-mesh
    support (needs an emissive-triangle light primitive in the core); rough
    transmission material.
- **POV-Ray: DECLINED (2026-07-11), rationale logged.** The SDL is genuinely nice
  (programmable, exact CSG, implicit `isosurface`), but a poor fit here: (1) faithful
  parsing = writing a Turing-complete interpreter (macros/loops/functions), far more
  than an XML parse, and POV-Ray has no mesh *export* to lean on; (2) we're a
  triangle/quad/sphere renderer with no analytic CSG or implicit intersection, so
  importing means **tessellating** everything (marching cubes for isosurfaces/blobs,
  mesh-booleans for CSG) — which discards POV-Ray's exact-surface advantage, its whole
  point; (3) RGB/non-spectral/non-physical-camera means re-authoring the physics anyway.
- **Alternative worth its own feature (deferred):** a **native SDF / implicit-surface
  primitive** (sphere-traced, GPU-portable) would give metaballs/isosurfaces *exactly*
  without lossy tessellation — useful independent of any importer, and the right way to
  ever support POV-Ray-style implicit geometry. Not started.

## Mode-M shared photon-map deposit/build hangs on the full gallery scene (4M photons)

- **Symptom:** `renderPhotonMapSharedCuda` on `scenes/gallery_settled.ftsl` (via
  `scraps/gallery_fly.ftsl`) with `-n 4000000 -spp 6 -r 320 180` ran **67 min pegging
  ~6 CPU cores (24,255 CPU-s) with GPU util ~1-6% and never wrote the `-savemap` file**
  (i.e. the one-time deposit+build phase never completed). Working set stayed ~1 GB, so
  it is NOT a huge-photon-set RAM blowup.
- **Contrast:** the identical pipeline with `-device gpu -n 2000000 -spp 4` on the SAME
  scene completed fully (deposit+build+2 gathers+mode-D still) in ~2 min. So 4M is >30x
  slower than 2M — wildly super-linear, pointing at a pathological host-side phase
  (suspect `PhotonMap::build` counting sort or the device->host photon download/convert
  loop in render_cuda.cu ~5680-5698), not simple scaling.
- **Repro:** `ftrace -in scraps/gallery_fly.ftsl -n 4000000 -spp 6 -r 320 180 -window
  -savemap gallery/hero_map.ftpmap -o png/fly/fly.png`
- **Proper fix (TODO):** profile the deposit/build with 2M vs 4M; find why host CPU
  scales super-linearly (likely an O(n^2) or lock-contended path, or grid cellSize
  degenerating so build buckets explode). Until fixed, cap flyby photons at ~2M.

## Access-violation popup when closing the live-preview window (intermittent) — PARTIALLY ADDRESSED (2026-07-17)

- **Symptom (user report, 2026-07-17):** closing two `ftrace.exe` live-preview windows
  produced two "access violation" WER popups on exit. The renders were mode-R gyroid
  batches launched with `-window -keepwindow` (studio-env HDR variants).
- **Could NOT reproduce despite faithful attempts.** Tried under **cdb** (break on AV /
  heap-corruption `0xC0000374` / fastfail `0xC0000409`) and, to defeat any debugger
  Heisenbug, under **procdump** (`-e -ma`, full native speed) across: (a) mode-B GPU
  render closed after finishing, (b) mode-R GPU render (`scenes/implicit.ftsl`) closed
  **mid-render**, and an isolated `src/livewindow.cpp` harness (`scraps/livewin_repro.cpp`)
  in three lifecycle modes — plain batch close, destructor-closes-a-still-live-window,
  and a no-sleep `update()`/`setTitle()` "hammer" race closed externally. **Every path
  tore down cleanly (exit 0, no dump).** So the isolated live-window lifecycle is NOT
  the fault on the paths tested.
- **Prime suspect: CUDA/driver async teardown.** The binary imports `nvcuda.dll`; closing
  a `-keepwindow` window is exactly what unblocks the whole shutdown (`main()` hold loop →
  `cudaGracefulShutdown()` → CRT static dtors incl. `g_liveWin`). `main.cpp` (~5568)
  already documents a related async `nvlddmkm` DPC teardown fault they mitigate. A
  userspace AV popup would be a fault inside `nvcuda.dll`/driver teardown, which our code
  can't fully fix — but capturing a real dump would confirm the module.
- **PARTIAL FIX applied (`src/livewindow.cpp`):** hardened a genuine latent cross-thread
  hazard found by inspection — `impl_->hwnd` was never cleared after the window was
  destroyed, yet it is read from the render thread (`setTitle`/`clientSize`/`enablePanel`/
  `setPathCount`) and from `~LiveWindow` (`PostMessageW(WM_CLOSE)`). Windows **recycles
  HWND values**, so a since-reused handle belonging to another window/thread could receive
  our `WM_CLOSE`/`WM_SETTEXT`. Made `hwnd` `std::atomic<HWND>`, null it on `WM_DESTROY`,
  and made every cross-thread caller (and the dtor) `load()` once + null-check so nothing
  marshals to a stale/recycled handle after close. Verified no regression via the isolated
  harness + a real mode-R render. This removes a real defect but is **not confirmed to be
  THE crash** (couldn't reproduce the original AV).
- **Next step if it recurs:** capture a full dump of the actual fault to pin the module —
  e.g. attach `procdump -ma -e` to the live PID, or `procdump -i -ma <dir>` (admin) to
  install a system postmortem catcher — then analyze `.dmp` (`!analyze -v`) to confirm
  whether it is `nvcuda`/driver teardown vs. app code.

## FIXED (2026-07-19): `-n` ignored scientific notation (`-n 2e8` → 2 photons)
`-n <count>` was parsed with `std::atoll`, which stops at the first non-digit — so the
common shorthand `-n 2e8` / `-n 8e7` (used in README examples) silently parsed as just the
leading integer (`2`, `8`), rendering a near-black image from a handful of photons. Fixed
in `run()` (src/main.cpp): tokens containing `e`/`E`/`.` are now parsed as a double and
rounded to the nearest count; plain integers still go through `atoll` exactly. Found while
validating `-stereo`.
