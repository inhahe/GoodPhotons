# Loom — procedural animation / geometry toolkit for Good Photons

> **Status:** design locked, implementation not started. This document is the
> authoritative plan. Working name **"loom"** (weaving curves/ribbons/loops over
> space and time) — provisional, easy to rename before first release.

Loom is a **programmatic-first** toolkit for building 3-D scenes and **seamless
looping animations** out of composable *modulators*, *curves/grids/scatter data*,
*sweeps* (ribbons/tubes/blobs), and *N-D-transformed isosurfaces*. It targets the
Good Photons forward raytracer by **emitting `.ftsl` per frame**, but is written to
stand alone (usable outside Good Photons, e.g. to drive any renderer or a preview).

---

## 1. Guiding principles

1. **Programmatic-first.** The authoring surface is a Python API. Any GUI is a
   later *add-on* — first a passive **viewer** (see results while editing), and only
   much later an interactive **editor** (drag control points). The Python model is
   the single source of truth; every front-end is just a reader/writer of it.
2. **Functions/fields over time; discretize LAST, per frame.** Never transport a
   discretization (mesh, point cloud, frame) through time. Animate the continuous
   thing (a `Signal`, a field, a curve) and re-sample/re-mesh every frame. This is
   what keeps N-D-rotated isosurfaces contiguous and keeps everything composable.
3. **One mechanism, unlimited depth.** Modulators are a DAG of pure functions;
   "modulators modulating modulators" is just more edges, not more machinery.
4. **Seamless loops are structural, not patched.** A loop is a *closed* path in some
   space (space, time, or a modulator's value). Built closed, it needs no seam fixup.
5. **Reuse before rebuild.** The modulation core already exists in
   `soundshop/juce_client/signals/core.py`; we vendor and generalize it rather than
   reimplement.
6. **Emit-ftsl-first.** Prefer letting the renderer mesh/root-find isosurfaces from
   emitted `.ftsl`; add an in-tool mesher only where we must bake a field to geometry.

---

## 2. Locked decisions

| # | Decision |
|---|---|
| 1 | **Language: Python**, reusing soundshop's `signals` DSL as the modulation core. |
| 2 | **Meshing: emit-`.ftsl`-first** — the renderer meshes/root-finds isosurfaces. An in-tool **adaptive marching cubes** is added only for fields that must be baked to a mesh (e.g. a scatter volume the renderer can't evaluate directly). |
| 3 | **Home:** `forward raytracer/tools/loom/` as a self-contained Python package; soundshop's `signals` is **vendored** (a trimmed copy) so Loom ships with Good Photons and also stands alone. |

---

## 3. Architecture (layers, bottom → top)

```
┌───────────────────────────────────────────────────────────────────────┐
│ 6. Drivers / IO:  render a frame range → .ftsl per frame → ftrace       │
│                   live viewer (emit → raster preview); GIF/MP4 assembly  │
├───────────────────────────────────────────────────────────────────────┤
│ 5. Scene:  geometry instances + materials + camera, all Animatable      │
│            serialize/round-trip (source of truth for a future GUI)       │
├───────────────────────────────────────────────────────────────────────┤
│ 4. Geometry:  sweep engine (+ ribbon/tube/blob/fan presets),            │
│               isosurface + N-D domain slicer, function-driven materials  │
├───────────────────────────────────────────────────────────────────────┤
│ 3. Data + interpolation:  point-path | grid | scatter  (N-D),           │
│               interpolators: scribbles3-closed-curve | grid | scatter    │
│               (each interpolator is ITSELF a Signal node)                │
├───────────────────────────────────────────────────────────────────────┤
│ 2. Math:  N-D vectors/matrices, Givens-rotation builder, the slicer      │
│           P = O + a·u + b·v + c·w                                         │
├───────────────────────────────────────────────────────────────────────┤
│ 1. Modulation DAG:  Signal graph (vendored soundshop), generalized to    │
│    N-D vector signals; cycle detection; per-frame cached evaluation      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer 1 — Modulation DAG (vendor + generalize)

**Source:** `soundshop/juce_client/signals/core.py` (vendored to
`tools/loom/loom/signals/`), trimmed of audio-specific bits (MIDI/param/plugin
dataclasses, beat/tempo, wavetable-osc phase machinery if unused).

**Keep as-is (reused):**
- `Signal` base (a DAG node = pure function of a clock), `children()`, per-block
  cache, operator overloading (`+ - * neg`), `Const`, `TimeFn`.
- `Add/Sub/Mul/Neg/Clamp/Rectify/Power/MapRange/Mix/Smooth`.
- `RefSignal` (shared/named sub-graph).
- **`detect_signal_cycle(root)`** — the loop detector (3-color DFS →
  `SignalCycleError`). **Runs before every render** so a bad graph fails loudly
  instead of hanging / stack-overflowing.
- `GlobalCanonicalizer` (CSE) + `ControlCache` (memoize shared modulators).

**Generalize (new):**
- Rename the clock from audio "sample" to a **normalized time / loop parameter
  `t ∈ [0,1)`** (one loop). Keep an optional real-seconds/`fps` mapping for export.
- **`VecSignal`** — an N-D vector whose components are each a `Signal` (or a single
  `Signal` broadcast). All vector math (`+ - *`, dot, matrix-apply) works on it, and
  it participates in the same DAG + cycle detection + cache.
- **Periodic leaves** for seamless loops: `Sine(freq,phase)`, `LoopCurve(...)`
  (the scribbles3 closed curve, see Layer 3) — everything periodic in `t` ⇒ the
  whole scene loops with no seam.
- **Deterministic randomness**: a seeded `Rand`/`Noise` leaf (repeatable loops).

**Open item (documented, not built yet):** *feedback / "elastic" modulators* whose
output depends on their own past (springs, relaxation). Those need **state across
time**, which a pure per-frame DAG forbids (it would be a cycle). Planned as a
separate **stateful evaluation mode** (integrate frame-by-frame), NOT an extension of
the pure DAG. Deferred.

---

## 5. Layer 2 — N-D math

- `Vec` (N-D), `Mat` (M×N), multiply, transpose, basic linear algebra.
- **`rotation(plane_i, plane_j, angle)`** → an N×N Givens rotation (rotate in one
  coordinate plane); compose several to rotate "on any number of axes." Angle may be
  an `Animatable`.
- **Slicer** `slice3(O, u, v, w)` → maps a scene point `(a,b,c)` to the N-D query
  point `P = O + a·u + b·v + c·w`. `O,u,v,w` are N-D `VecSignal`s (animatable).
  This is the general "rotate + take a 3-D slice" operator we derived; feeding the
  first `k` components to a `k`-input function gives the honest behavior (a 3-input
  field → affine tilt/shear/drift; a genuinely `k`-D field → true morph).
- Everything here is `Animatable` (matrices/vectors can be modulated over time).

---

## 6. Layer 3 — Data structures + interpolation

Three **datasets**, each N-D, each with **every value feedable by a modulator**
(a stored value may be a `Signal`, so control points animate).  Each dataset is
itself a **node in the modulation DAG** (it carries an `id` + `children()`), so it
is both *modulable* (its stored values are driven by modulators) **and** a
*modulator* (an interpolator over it is a `Signal`), and `detect_signal_cycle`
walks *through* the dataset — a control point / grid value / scatter position or
value that loops back is caught:

1. **Point-path** — an ordered sequence of N-D points (a curve's control points).
   The **points themselves are modulable** (each is a `VecSignal`; any coordinate
   may be a `Signal`), so control points animate over the loop.
2. **Grid** — N-D values on a regular lattice of *arbitrary rarity* (resolution).
   Only the **values** are modulable; the lattice **positions are deliberately
   fixed**. That regular structure is the whole point of a Grid — it is what buys
   the fast **separable N-linear interpolation** — so animating node positions is
   explicitly *not* a Grid feature. If you want moving sample *positions*, that is
   exactly what **Scatter** is for.
3. **Scatter** — N-D values at arbitrary positions (no lattice). **Both** the
   sample **positions** (each a `VecSignal`) **and** their **values** are modulable,
   so a scatter point can drift *and* pulse; `ScatterField` re-reads every position
   and value per frame, and both are walked by the cycle detector.

Plus one composite dataset built on the point-path:

- **`TrackedPath`** — a point-path that carries **Y extra per-waypoint tracks**
  keyed at the *same* control points (the toolkit analog of a `camera_curve`: one
  sequence bundling position + a speed/density track + an orientation track + any
  other scalar/vector track you key). Each track is one value per control point,
  scalar or N-D vector, animatable like everything else.

Three **interpolators**, each exposed **as a `Signal`/field** (so an interpolator's
output can feed another modulator — "it's just another function"):

1. **`LoopCurve` (scribbles3 curve, generalized to N-D).** Port the midpoint
   quadratic-Bézier construction: each control point `B` with neighbors `A,C`
   produces an arc from `mid(A,B)` through `B` to `mid(B,C)`; wrap with modulo for a
   **seamless closed** curve (no seam angle to choose). Verified to generalize to any
   dimension (the construction is per-component). Open/closed both supported.
2. **`GridField`** — grid interpolation → a value anywhere in the volume.
   `interp="linear"` (default, N-linear) or `interp="cubic"` (separable Catmull-Rom
   / tricubic; smoother C1, may overshoot). Boundary phantoms are linearly
   extrapolated so cubic reproduces linear ramps to the edge.
3. **`ScatterField`** — Shepard inverse-distance interpolation of scatter values
   (robust, C0, flattens toward the mean far from samples).
3b. **`RbfScatterField`** — radial-basis interpolation (`scipy.interpolate.RBFInterpolator`,
   an *optional* dep, lazily imported). Smooth, exact at samples, meshless, any N-D.
   Default kernel `thin_plate_spline` (parameter-free); `on_outside="clamp"` guards the
   convex-hull extrapolation. Rebuilt **only when the sampled positions/values actually
   change** (change-detection cache in `_RbfEngine`): a static field builds once and is
   reused verbatim across the whole animation even as the query moves.

Grids and scatters may hold **vector** values (a `VecSignal` per sample, optionally
with named `channels=`). `VecGridField` / `VecScatterField` / `VecRbfScatterField`
interpolate those as a `VecSignal`: the **domain weights (or RBF factorization) are
computed once** and applied to every channel (RBF uses a multi-column RHS — one solve
for all channels), so a vector field's `.channel(name|idx)` equals the scalar field over
that channel, and the single-valued field is just the one-channel case. Grid fields also
take `interp="linear"|"cubic"` (see §2 above).

And, over a `TrackedPath`, one multi-curve sampler:

4. **`TrackedCurve`** — samples a `TrackedPath`'s position **and every track** on
   one shared seamless parameter `u` (each track is just another `LoopCurve` riding
   the same `u`), exactly the way a camera flyby's speed and look-direction curves
   ride along its position curve. `TrackedCurve.traveling(tracked, s, density=...)`
   retimes traversal through **`Reparam`** — an inverse-CDF over equal `u`-bins that
   maps a uniform travel param `s` to a `u` that *dwells* where the density track is
   large (the distinguishing behavior of a camera-curve speed curve).

And one **field-routed** curve:

5. **`FieldCurve`** — a curve routed **through a field**. `FieldCurve(curve, field_builder, u)`
   pairs a position curve with a field built over that curve's point (`field_builder` is a
   callable `q -> field`, so any grid/scatter field composes in). `.position`, `.value` and
   `.channel(name|idx)` are DAG nodes that can drive scene variables; `.sample(u, clock)`
   polls at an explicit progression index → `(coords, {channel: value})`.

Because interpolators are `Signal`s, you can: feed a modulator into a control point;
*or* feed an N-D value into an interpolator to read a value out and pass it onward;
*or* chain modulators through interpolators arbitrarily. All one DAG, all cycle-checked.

---

## 7. Layer 4 — Geometry

### 7a. Sweep engine + presets
One engine: **`sweep(spine, profile, frame, scale, twist, linkage)`** — carry a
cross-section (`profile`) along a `spine` curve, orienting it (`frame`), resizing it
(`scale`), rotating it (`twist`), and skinning consecutive cross-sections (`linkage`
= straight | curved). Needs a **frame field**: rotation-minimizing frame
(double-reflection method) for a stable orientation with no flips; for closed spines,
distribute the residual twist so the ribbon closes seamlessly.

**Four named presets (your original ideas), each a thin wrapper over `sweep`:**
| Preset | = sweep with |
|---|---|
| `ribbon(spine, width, rotation)` | profile = 2-point line; `scale`=width; `twist`=rotation |
| `tube(spine, radius)` | profile = closed circle; frame ⟂ spine |
| `blob(spine, profile, ...)` | closed profile, **curved** cross-links |
| `fan(spine, rotations[], distance)` | profile = points placed by rotation/distance curves |

Raw `sweep` stays exposed as the power-user escape hatch.

### 7b. Isosurface + N-D slicer
Emit an ftsl `isosurface`/`function` block whose input coordinates are pre-transformed
by the Layer-2 slicer (rotate/scale/shear/drift, or true N-D slice if the function
declares ≥4 inputs). Parameters (frequency, threshold, N-D rotation angles, slice
anchor) are all `Animatable`.

### 7c. Function-driven materials
Reuse Good Photons' existing material-props-by-function (reflectance/color/IOR/etc.
over `x,y,z`/UV). Loom emits those expressions; adding `t` makes any property animate.

---

## 8. Layer 5 — Scene + serialization

- A `Scene` = geometry instances + materials + camera, each field `Animatable`.
- **`evaluate(scene, t)`** → a concrete, non-animated snapshot (numbers, not Signals).
- **Serialize / round-trip** the whole model (JSON). This is the discipline that keeps
  "GUI as add-on" cheap: the GUI is just another reader/writer of this format.

### 8a. Parametric records (`Record`) — a loom twin of the FTSL record
A **`Record`** (`record.py`) mirrors ftrace's parametric record (`src/record.h`,
`ROADMAP_records.md`): a bank of named per-channel curves over a shared scalar driver
domain `[lo, hi]`, sampled `nearest`/`linear`/`smooth`. Each **channel** is named
after a destination slot and holds ordered **stops**; a channel is either *scalar*
(`D==1`; numeric-literal or pattern-expression stops) or *colour* (`D==3`;
`spectrum:`/`metal:`/`rgb:` refs) — the two arities ftrace materializes today. Stops
carry a raw `token` (preserved verbatim) and an optional `p:<pos>` pin; unpinned stops
redistribute evenly between anchors exactly as ftrace does.
- **Emit** produces the `NAME = range LO-HI [ … ]` block (routed through `Scene`
  before the materials that bind it); **`Record.parse` / `parse_all`** read one (or
  every) record block back out of `.ftsl` text — the round-trip that lets loom *copy*
  an existing scene's records (J3a; round-tripped against `scenes/_record_*.ftsl`).
- **`Record.sample(channel, d)`** is a numeric sampler (Fritsch–Carlson monotone cubic
  for `smooth`) for all-numeric scalar channels; colour and *expression* stops are
  represented and re-emitted faithfully but not evaluated (that needs the pattern VM —
  deferred to the full-scene parser, J3c). Higher channel arities are the loom-only
  superset (J3b).

**Ladder parser (`ladder.py`, J3b item 2).** The generalized stop grammar
(`ROADMAP_records.md` §3.1) authors arbitrary-arity nested values with a delimiter
*precedence ladder* — **whitespace binds like `×`, comma like `+`, brackets are parens**
— so `1 1 1, 2 2 2` parses like `(1·1·1)+(2·2·2)` and structure is recoverable from the
delimiters alone. `parse_ladder(str)` → nested `list`/`str` tree (leaves are raw tokens);
`emit_ladder(v)` renders it back canonically (space-join a flat vector, comma-join with
bracket-wrapped multi-level children — round-trips); `shape(v)` reports rectangular dims.
Parens `( )` are **not** a ladder delimiter (reserved for expressions / the §3.2
application surface) — a parenthesised run is an opaque atom, so `clamp(x,0,1)` is one
leaf. This is loom-only authoring; current ftrace's tokenizer is not comma-aware and
cannot parse it.

**Arbitrary-arity (vector) channels (J3b item 1).** `RecordStop` now holds
`.components` (a `D`-tuple of tokens; `.token` is the single component of an arity-1
stop, `.arity`/`.as_vector()` the vector view), so a `RecordChannel.kind` is `scalar`
(arity 1) / `colour` (`:`-refs) / **`vector`** (arity `D` ≥ 2, homogeneous). `Record`
gains `sample_vec(name, d)` (per-component interpolation; scalar `sample` still returns a
float). Current-FTSL uses *whitespace* to separate stops but the ladder uses it for
*components*, so `emit`/`parse` are **one backward-compatible ladder grammar** (an
*additive superset*, not a breaking change) that dispatches per channel line on the
presence of a top-level comma: a comma-free line is the J3a whitespace form (every word
a scalar stop — `metal steel gold copper` = three stops), while a line with a top-level
comma is the ladder form (comma-separated stops, space-separated components — `tint 0 0
0, 1 1 1` = two arity-3 vector stops). A *lone* vector stop is written with a trailing
comma (`tint 0 0 0,`) so it can't be misread as N scalar stops. `emit` picks the form
per channel automatically (whitespace for scalar/colour, comma for vector); records with
no vector channel emit byte-identically to before. ftrace's own tokenizer is still not
comma-aware, so a record that actually uses comma lines stays loom-only until J3c.

**Inline-colour channels + lowering (J3b item 1, done).** A colour channel can be
authored *inline* with a leading `rgb`/`hsv`/`hsl` **tag** word instead of a chain of
`spectrum:<name>` refs — `reflect  rgb 0.55 0.57 0.60, 0.90 0.75 0.30` is a two-stop rgb
colour channel (`RecordChannel.space` carries the tag; `.kind == "colour"`, `.arity == 3`).
The tag fixes arity 3, so each comma-group is one colour stop and a lone tagged stop
(`reflect  rgb .5 .5 .5`) needs no trailing comma. An `rgb` channel is numerically
sampleable (`sample_vec` interpolates the components = ftrace's linear-RGB colour interp);
`hsv`/`hsl` channels reject sampling until lowered. `Record.lower_colours()` rewrites every
inline-colour channel to the ftrace-native form: it returns `(decls, lowered_record)` where
`decls` are synthesized `spectrum "<name>" = rgb r g b` declarations (one per **unique**
colour, deduped across the record; `hsv`/`hsl` converted to rgb via loom's own hue maths)
and the channels now hold `spectrum:<name>` refs (pins preserved). `lower_ftsl()` returns
the decls + record as one self-contained parseable block. The remaining J3b item-1 piece
is wiring these synthesized spectra into a full-scene emit path (part of J3c).

---

## 9. Layer 6 — Drivers / IO

- **`render_range(scene, frames)`** → for each frame `k`: `t=k/frames`, run cycle
  check, evaluate, emit `.ftsl`, invoke `ftrace` (with the mandatory `-window` /
  crash-safe flags per project rules), collect PNG.
- **Live viewer** (cheap GUI value): emit → raster preview so you can watch loops
  while tuning. Passive; no editing.
- **Assembly**: reuse existing `tools/obj_sequence_to_video.py`-style helpers to build
  a seamless GIF/MP4.
- **Determinism**: a global `--seed`; a given seed reproduces a loop exactly.
- **Optional in-tool adaptive marching cubes** (only where a field must be baked to a
  mesh): octree/dual-contouring that subdivides more where the field changes fast and
  emits fewer faces in flat regions (`configurable fineness` + `adaptive` flag).

---

## 10. Directory layout

```
tools/loom/
  DESIGN.md                 (this file)
  README.md                 (user-facing; written when the API stabilizes)
  loom/
    __init__.py
    signals/                (vendored + generalized soundshop signals)
      core.py               scalar Signal graph + cycle detector (reused)
      vector.py             VecSignal, N-D vector ops (new)
      periodic.py           Sine, LoopCurve leaves, seeded Rand/Noise (new)
    mathnd.py               Vec/Mat, Givens rotation, slice3 (new)
    data.py                 PointPath / Grid / Scatter datasets (new)
    interp.py               LoopCurve(N-D) / GridField / ScatterField (new)
    sweep.py                sweep engine + frame field + 4 presets (new)
    iso.py                  isosurface + N-D slicer emit (new)
    material.py             function-driven material emit (new)
    record.py               parametric record twin: emit + parse + sample (J3a)
    ladder.py               delimiter-precedence-ladder parser (J3b item 2)
    scene.py                Scene, evaluate(), serialize/round-trip (new)
    ftsl_emit.py            snapshot → .ftsl text (new)
    drive.py                render_range, viewer, assembly, seed (new)
    mcubes.py               marching cubes: bake a field to a mesh (M7)
    xvideo.py               two-pass spacetime transform video (M11)
    preview.py              resident ftrace -serve preview client (M12)
  examples/                 runnable scripts (ribbon loop, gyroid slice, scribbles3-in-3D)
  tests/                    unit tests (cycle detection, closed-curve seamlessness, slicer)
```

---

## 11. Open items / risks (design on purpose, don't stumble in)

1. **Feedback / elastic modulators** need state-across-time → separate stateful mode,
   deferred (§4).
2. **Scatter→volume interpolation quality** (inverse-distance vs RBF vs natural
   neighbor) is a real quality/speed tradeoff; start simple, revisit.
3. **Adaptive meshing** is the heaviest new algorithm; kept optional and last because
   emit-ftsl covers most isosurface needs.
4. **Aperiodic vs seamless** slices: irrational N-D slice angles never repeat (nice
   quasicrystal look) but break looping; the "loop" flag forces commensurate/closed
   motion.
5. **Performance**: Python is fine for authoring + preview + emitting scenes; heavy
   geometry (meshing) may later warrant calling ftrace's C++ mesher instead.
6. **Looping is opt-in, not baked (decision, post-M6).** The impression that loom is
   "always seamlessly periodic" came from exactly one line — `Clock.at_frame`'s
   `t=(frame % frames)/frames` (modulo wrap + division by `frames`, so frame N == frame
   0). The DAG engine itself is timeline-neutral. Periodicity actually lives in *what
   you compose*: periodic leaves (`Sine`/`LoopNoise`/`phase_drift`→`sin`) + a **closed**
   `LoopCurve`. `LoopCurve` is already the opt-in "seamless because I chose a closed
   curve" mechanism; the clock just overrode the choice. Fix (M6.5): give `Clock` an
   open vs. closed mode — closed keeps `(frame % frames)/frames`; open uses
   `frame/(frames-1)` (no modulo, endpoints distinct, no phantom duplicate frame) — and
   let seamlessness be a property of the composed leaves/curves, not an imposed
   invariant. Closed stays the default so M1–M6 are untouched.
7. **POV-function N-D honesty.** ftrace exposes ~78 POV-Ray isosurface functions as expr
   builtins (`src/pov_functions.h`, `povFnLookup` name→(id, arity); `f_name(x,y,z,
   ...params)`, 3 coords + up to 10 params; wired into both `implicit.h` and
   `pattern.h`/`PatOp::PovFn`; 8 are explicit `_2d` variants). loom wraps them as
   *field/pattern templates* (not DAG nodes — they're functions of space, not `t`); their
   **params** are the DAG hook (Signal-driven, baked per frame). Only the algebraically
   symmetric subset (`f_sphere`, `f_ellipsoid`, `f_superellipsoid`, `f_paraboloid`,
   `f_ovals_of_cassini`, the quartics, TPMS) generalizes to a *genuine* extra dimension;
   the bespoke named surfaces (`f_heart`, `f_klein_bottle`, `f_boy_surface`, …) are 3-D
   artifacts — they can only be affine-sliced (tilt/shear/drift), never honestly morphed.
   Docs must not over-promise "N-D heart."
8. **Spacetime rotation needs a two-pass model + a torus constraint.** Rotating a plane
   that includes the time axis breaks "each frame is a pure function of `t`" — a rotated
   frame depends on a *range* of times. Doing it honestly means materializing the whole
   temporal extent into a 4-D block, rotating, then re-slicing (time-caching / freezing,
   two passes) → a **separate "transform video" script**, never the streaming emitter.
   Seam caveat: loop-time is a *circle* (S¹); rotating a periodic axis into a
   non-periodic spatial axis is no longer periodic, so seamless output requires *both*
   coupled axes periodic (a rotation on a 2-torus). Natural layering: open clip →
   transform → open clip out is the general/default case; looped output is the
   constrained special case.
9. **Preview bottleneck is scene *rebuild*, not the raster pass.** For a 480² preview on
   a modern GPU the rasterizer is not the cost — re-parsing ftsl + re-tessellating
   isosurfaces + rebuilding accel structures each frame is. So the interactivity win is a
   **resident ftrace preview server** that takes per-frame *deltas* (only the changed
   baked constants), plus static-geometry caching and preview LOD — not a hand-rolled
   faster rasterizer (which would only lose fidelity). Reuse ftrace's raster for the 80/20
   viewer today; resident-server is the real speedup later.
10. **Naming: keep "loom".** The weaving metaphor is earned (threading a DAG, sweeping
   ribbons/tubes, skinning meshes — `skin_rings`/`MixMaterial("skin")` already in code).
   Rejected "Snakecraft"/"Snakeskin" — snake puns are overdone and renaming a working,
   committed, tested codebase for a pun isn't worth the churn. ("Snakeskin" could name the
   2D backend if a pun is ever wanted.)
11. **Coordinates do NOT belong in the time-DAG.** The Signal DAG is a function of the
   *clock*, cached per `(node, frame)` — one value per node per frame. That invariant is
   the whole reason the cache works, and a *spatial* input (`x`, `y`, `z`) breaks it: a
   node would have one value per **pixel**, not per frame, so the frame-keyed cache is
   simply wrong. So a **field** (function of space) lives in a *separate* spatial algebra,
   not the temporal DAG. The two axes stay factored: **loom owns time, the field owns
   space**, and a *time-varying* field is their product — a spatial expression whose
   *coefficients* are temporal Signals baked per frame (exactly how the 3-D pattern/iso
   emitter already animates a static x/y/z formula). Two authoring styles for a field are
   genuinely distinct and both kept: an **opaque numpy callable** `f(x,y,clock)` (fast,
   imperative, but loom can't introspect or re-emit it) vs a **symbolic spatial-expr tree**
   (loom can evaluate it numerically *and* emit it as an ftsl string, and can *introspect*
   it — e.g. auto-detect a time-independent field and bake its raster once). Time-dependence
   is orthogonal to authoring style; don't conflate "static→numpy, animated→tree". The tree
   is the **shared 2-D-numeric / 3-D-emitted pattern layer** (`loom/spatial.py`, M10.5).

---

## 12. Build order (milestones)

- **M1 — Foundation.** Vendor + trim `signals/core.py`; verify cycle detector; add
  `VecSignal`; port `LoopCurve` (scribbles3 curve) to N-D; the three datasets +
  three interpolators. Tests: cycle detection fires; closed curve is seamless;
  interpolators evaluate as Signals.
- **M2 — Math + slicer.** `Vec/Mat`, Givens `rotation`, `slice3`. Test: 3-input
  gyroid tilts/shears/drifts; ≥4-input gyroid genuinely morphs.
- **M3 — Emit + drive + viewer.** `Scene`, `evaluate`, `ftsl_emit`, `render_range`,
  live viewer, seed, GIF assembly. Milestone demo: **a seamless looping GIF** (a
  scribbles3-style closed curve, now in 3-D, rendered by Good Photons).
- **M4 — Sweep.** Frame field + `sweep` engine + 4 presets. Demo: a looping ribbon.
- **M5 — Isosurface animation.** `iso.py` + slicer wired to ftsl. Demo: a gyroid
  whose N-D rotation/params modulate over a seamless loop.
- **M6 — Function materials.** Animated reflectance/color/IOR over space+time. ✅ done.
- **M6.5 — Opt-in looping.** ✅ done. Make seamless looping a *choice*, not a baked invariant
  (§11.6). `Clock` gains open vs. closed mode: closed keeps `(frame % frames)/frames`;
  open uses `frame/(frames-1)` (no modulo, distinct endpoints). Add an **open-curve
  interpolator** (non-wrapping spline through a `PointPath`, symmetric with the closed
  `LoopCurve`) and a couple of **non-periodic leaves** (linear ramp, ease-in/out
  envelope) so the open-timeline kit exists. `render_range(..., loop=True|False)` picks
  the sampling and whether the seam-equality assertion applies. Closed stays the default
  so M1–M6 are untouched. Tests: open clock endpoints distinct (no phantom frame N);
  open path is *not* seamless while a closed curve still is; a ramp leaf differs frame 0
  vs last under open mode.
- **M7 — Marching cubes (bake a field to a mesh).** ✅ done (`loom/mcubes.py`,
  `loom.IsoMesh`). ftrace root-finds isosurfaces directly, so most fields stay an
  `Isosurface` (emitted `function { expr }`); `IsoMesh` is for the minority case where
  a field must become geometry (a numpy-only field with no ftsl twin, a sampled volume,
  a mesh for another tool). `mesh_field(field, bounds, res, iso, adaptive)` turns a
  `SpatialExpr` (baked at the clock) or a vectorised `f(X,Y,Z)` into `(verts, faces)` via
  scikit-image's crack-free Lewiner marching cubes (lazy/optional import). `IsoMesh` bakes
  one OBJ per frame via `ctx.asset_path` and emits `mesh { file ... }`; a time-independent
  field is baked once and cached. **Adaptive = honest narrow-band *sampling*** (not variable
  triangle density): a coarse pass finds blocks straddling the iso and the fine grid is
  evaluated only there (+ a one-block skirt), far blocks filled with a same-sign sentinel,
  then **one** global MC runs — crack-free and identical to a dense fine mesh near the
  surface while skipping ~O(res³) far cells (measured 5.6% of dense evals on a thin
  surface). True variable-density DC/QEF output stays future work; MC emits uniform density
  by construction and we don't pretend otherwise. Tests: `tests/test_mcubes.py` (sphere-radius
  accuracy, 2-manifold edges, adaptive==dense, fewer evals, empty-box, callable+SpatialExpr,
  morphing field, IsoMesh emit/static-cache/roots). Demo: `examples/mesh_bake.py` (a breathing
  smooth-min metaball union baked per frame; still validated in ftrace).
- **M8 — Affine composition.** ✅ done. Collapse an arbitrarily long chain of N-D Givens
  rotations **+ translations** into one baked `(Mat, offset)` affine per frame (extend
  `rotations()` to homogeneous coords). Win: one affine in the emitted expr instead of a
  sequential chain (fewer ops in ftrace's per-hit eval). Pin the order/convention (row
  vs column, pre vs post) once; test associativity vs a reference. Small, low-risk.
- **M9 — POV-function library.** ✅ done. Wrap ftrace's ~78 POV isosurface builtins as
  parametric field/pattern templates driven off a mirrored `povFnLookup` table
  (name→arity); validate param count in Python; params are Signal-drivable (baked per
  frame). Golden-value tests per function against known shapes. Honesty per §11.7:
  affine-slice all, genuine N-D only for the symmetric subset.
- **M10 — 2D backend.** ✅ done (`loom/canvas.py`). A parallel output driver over the
  *same* dimension-agnostic DAG — **not** a fork. The core primitive is the user's model:
  `Canvas2D.plot(x, y, color, ...)` plots an RGB at an (x, y) **at the current clock**,
  so a single call traces a moving/colour-cycling marker over the loop (seamless from
  periodic leaves, open under `loop=False`). Two output formats (both, per the user):
  **SVG** = resolution-independent vector primitives (markers + strokes); **raster PNG**
  (Pillow/numpy) = pixels, so it also renders a full-canvas per-pixel `field(fn)` and
  assembles a seamless GIF. `stroke()` polylines a point list; `curve_points()` samples a
  `LoopCurve` to a stroke (sweeps→strokes). y-up world `view` box; colours RGB in [0,1].
  Honesty: SVG has no per-pixel surface, so it omits `field`. Tests: `tests/test_canvas.py`
  (mapping, per-frame animation, seamless wrap vs open endpoints, field, strokes, cycles).
- **Colour model — RGB, HSV *and* HSL** (`loom/color.py`). ✅ done. A `Color` is a
  3-component `VecSignal` that *is* its resolved **RGB** (an HSV/HSL colour is converted
  in the graph via `hsv_to_rgb` / `hsl_to_rgb`), so it drops into 2-D (`Canvas2D`
  markers/strokes/field) **and** 3-D (`Material` colours) with no special casing, and —
  remembering how it was authored — emits the matching `.ftsl` colour token
  (`rgb r g b` / `hsv h s v` / `hsl h s l`), which ftrace's scene loader now parses
  natively. Both cylindrical models are kept: **HSV** (value; `v=1` most vivid) matches
  painterly pickers, **HSL** (lightness; `l=0.5` pure hue, `l→1` white, `l→0` black)
  matches CSS — they share the same hue wheel. Hue is in `[0,1]` and **wraps**, so a hue
  driven by a 1-periodic leaf cycles the whole wheel and returns bit-for-bit at the loop
  seam (seamless colour cycling). Tests: `tests/test_dag_and_color.py`.
- **Image skins — `Texture` + `skin()`** (`loom/scene.py`). ✅ done. An image file
  applied to a surface as a spatially-varying diffuse albedo. `Texture("name", "img.png",
  encoding=…, filter=…, wrap=…)` emits a `.ftsl` `texture "name" { file "…" … }` block;
  `skin("name", "img.png", **material_props)` is the one-call convenience returning the
  `Texture` *and* a `Material` bound via `reflect texture:name`. The Scene emits texture
  blocks before the materials that bind them. (Sweep's mesh ring-skinning is now
  `skin_rings` to free the `skin` name.) Tests: `tests/test_dag_and_color.py`.
- **M10.5 — Shared spatial-expression pattern layer.** ✅ done (`loom/spatial.py`). One
  pattern **defined once, used two ways** (§11.11): a `SpatialExpr` tree over coordinate
  leaves `X`/`Y`/`Z` + loop phase `T`, with temporal `Signal` coefficients baked per frame.
  `eval_np` evaluates it numerically over numpy grids (the 2-D raster `field`); `emit`
  renders it as an ftsl string in x/y/z (the 3-D isosurface/pattern). Every emitted builtin
  (`sin`/`sign`/`clamp`/`mix`/…) is a real `src/pattern.h` op and isosurfaces share that
  same `patternEval` engine, so one string is valid for both; the numpy twins compute the
  same maths (`noise` deliberately omitted — no bit-identical numpy value-noise). It plugs
  into `Isosurface`/`FuncPattern` through their **existing** `build()`/`param_signals()`
  duck-typed protocol — zero changes there. `Canvas2D.field` now type-dispatches (SpatialExpr
  / 3-tuple of them / opaque callable) and **bakes a time-independent field once** (auto for
  a tree via `uses_time`; a `static=True` flag for opaque callables). Enabling fix: `Signal`
  arithmetic returns `NotImplemented` for un-coercible operands so `Signal * SpatialExpr`
  defers to the reflected op. Tests: `tests/test_spatial.py` (deterministic emit, numeric
  semantics vs ftrace ops, `uses_time`/`time_signals`, one expr → both backends, static
  bake, iso integration). Demo: `examples/shared_pattern.py` (a drifting gyroid as both a
  2-D loop and a 3-D isosurface loop).
- **M11 — "transform video" script.** ✅ done (`loom/xvideo.py`). Separate two-pass
  offline tool (§11.8), kept out of the streaming emitter: **materialize** a clip into a
  4-D block `(T,H,W,C)` (`Clip.from_array` / `.from_frames` / `.from_canvas`), **transform**
  it under a spacetime map, **re-slice** to frames (`Clip.save` → PNGs + GIF). Two honest
  cases: `spacetime_rotate(clip, angle, axis, coupling, mode)` is the **general/default open**
  case — a metric rotation of the (axis, t) plane that synthesizes motion from time (a static
  stripe sweeps across); it does **not** loop (open boundaries held/blanked), because rotating
  the periodic S¹ time axis into a non-periodic spatial axis isn't periodic. `spacetime_shear(
  clip, axis, winding)` is the **constrained seamless-loop** case — an integer-winding shear on
  the 2-torus: over one loop, time advances one period while the coupled (tiling) spatial axis
  scrolls `winding` whole periods, so both axes wrap and the output is **bit-seamless** (a
  non-integer winding is rejected). Interpolation via SciPy `map_coordinates`. Tests:
  `tests/test_xvideo.py` (materialize, rotate motion/identity/boundary modes, bit-exact shear
  seam, winding-zero static, integer-winding + axis validation). Demo:
  `examples/transform_video.py` (`--rotate` open sweep, `--shear` seamless torus scroll).
- **M12 — resident preview server.** ✅ done (first increment). ftrace gained a
  `-serve` mode (`src/main.cpp`, `runServe`): instead of exiting after one render it keeps
  the process — and with it the live window, CUDA context, and spectral / spectral-upsampling
  tables — resident, re-rendering whenever a new scene path arrives on stdin (one path per
  line; `[serve] ready` / `[serve] done <path>` / `quit` protocol). The loom side is
  `loom/preview.py`: `PreviewServer` (a context-managed resident process) and `preview_range`
  (animate a `Scene` through it), which stream one `.ftsl` per frame so the live window
  updates *in place* with no per-frame process churn. Tests: `tests/test_preview.py`
  (command assembly, budget precedence, protocol handshake, frame naming, clean shutdown —
  driven by a fake in-memory server so they run headless); end-to-end smoke via
  `scraps/preview_smoke.py`. Demo: `examples/preview_server.py --preview`.
  **Honest scope:** this delivers the *resident-process* win only — skipping per-frame
  process spawn + window/CUDA/table init. It does **not** yet do the per-frame *delta*
  push (only changed baked constants), static-geometry / BVH caching between frames, or a
  reduced preview LOD (§11.9). Each frame is still a full independent render, and the live
  window keeps the first frame's resolution for the session. Those remain the real future
  speedup; `-serve` is the bounded, correct increment they build on.

- **M13 — `CameraCurve` element + two-axis orientation.** ✅ done (`loom/scene.py`
  `CameraCurve`). loom gained a genuine ftrace `camera_curve` element: unlike `Camera`
  (re-baked to a static `camera` block every frame by loom's clock), a `CameraCurve` is
  emitted **once** and *ftrace* expands the N flyby frames — pass it in place of the camera
  (`Scene(camera=CameraCurve(...))`). It mirrors ftrace's grammar 1:1: `points` spline,
  `frames`/`density`/`density_at` speed, `look_at`/`look_points` aim, and the animatable
  lens/orientation tracks. The orientation half also required the underlying ftrace feature
  (`src/ftsl.h` `addCameraCurve`): a **two-axis** model where forward (`fwd_at`, else
  `look_at`/tangent) and up (`up_at`, else `roll`, else reference up) are authored and
  `right` is derived, each read in a `world` or `travel` **reference frame**. `travel` is a
  rotation-minimizing frame (RMF) built by double-reflection parallel transport (`Vec3Track`
  + the RMF pre-pass) with closed-loop holonomy distributed for a seamless loop, so shots
  bank into turns. Authoring no orientation keywords is byte-identical to the legacy world-up
  framing. Tests: `tests/test_emit.py` (`test_camera_curve_*` — golden emit, orientation
  axes, scalar tracks, validation, in-scene); ftrace-side validation scene
  `scenes/_camA_travel.ftsl`.

Each milestone: keep `known-issues.md` current, commit at green checkpoints, never
`git push`. Update this doc if the plan changes.
