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
- **Time as a value + retiming** (`signals/retime.py`). Because a `Signal` is a
  *pure function of a `Clock`*, evaluating one at some other phase is already
  well-defined — the only thing that ever assumed one-value-per-node-per-frame was
  the memo. So the whole family comes from two additions:
  - **`Phase`** — a leaf that returns `clock.t` *as a value*, making time a
    first-class input rather than an ambient one.
  - **`Retime(x, when)` / `VecRetime`** — evaluate `x` against
    `retimed_clock(clock, when)`. Sugar: `freeze(x, at)` (`x` pinned at one phase,
    and `at` may itself be animated ⇒ a scrubbable hold), `delay(x, dt)`
    (`x(t−dt)`; wraps on a closed clock so a delayed seamless loop *stays*
    seamless — negative `dt` looks ahead, which is equally well-defined for a pure
    function), `warp(x, g)` (`x(g(t))` for a Signal or a plain `f(t)->t'`).
    `wrap` defaults to *wrap iff `clock.loop`*, keeping an open timeline honest
    off the end.
  - **The cache.** `Cache` keys on `(node id, frame)`; a retimed subtree is
    evaluated at a *different phase inside the same frame*, so its values must not
    land in that store. Every retime evaluates its child through
    **`Cache.scope(key)`** — a nested `Cache` keyed `(node id, frame, sample
    phase)`. Sharing still works *within* one sample point; nothing leaks between
    sample points or out to the frame. (Chosen over widening the global key, which
    would have touched every `at()` call site; `scope()` is purely additive and
    behaviour is unchanged when no retime node exists.)
  - **Cycles.** Both edges — the retimed subtree *and* the phase driver — are
    ordinary structural edges reported by `children()`, so `detect_signal_cycle`
    keeps owning them. A retime is **not** a recurrence: it reads a pure function
    at another point, it does not read its own past (see the open item below).
  - The 4-D **time-shear** — a *spatially varying* sample phase — needs a
    coordinate in scope, so it lives on the spatial tier as `spatial.SigAt`
    (§ M10.5).

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
   exactly what **Scatter** is for. Constructor is `Grid(values, *, shape=None,
   lo=None, hi=None, channels=None)` (values first; everything else keyword-only).
   **`shape` is optional** — the *nesting of `values` carries it*: `[[0 1 2][3 4 5]]`
   is a `(2, 3)` grid, no `shape=` needed. Bare `list`/`tuple`s always mean *axes*;
   a stored **vector** value is a `vec(...)` (never a bare list), so a nested grid of
   `vec`s is a vector grid. A flat list is 1-D unless you pass `shape=` to fold it
   (`Grid([0,1,2,3,4,5], shape=(2,3))`); ragged nesting is rejected (use **Scatter**).
   **`lo`/`hi` are optional and broadcastable**: `lo=None`→all-zeros, `lo=<scalar>`→broadcast;
   `hi=None`→a unit-spacing index lattice (`hi[a]=lo[a]+shape[a]-1`, so a coordinate
   equals a sample index), `hi=<scalar>` (or length-1)→pins **axis 0** and derives every
   other axis as a **uniform lattice** — one isotropic spacing `h=(hi-lo[0])/(shape[0]-1)`,
   `hi[a]=lo[a]+h·(shape[a]-1)` (the mathematically pure "single lattice constant" reading,
   so the interpolated field is geometrically isotropic; a full `hi` tuple gives the exact
   box, including deliberately anisotropic cells).
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
   extrapolated so cubic reproduces linear ramps to the edge. Out-of-domain policy
   `on_outside="clamp"` (default, edge-extend) / `"raise"` / `"wrap"` (periodic fold)
   / `"extrapolate"` (linearly extend off the boundary cell).
3. **`ScatterField`** — Shepard inverse-distance interpolation of scatter values
   (robust, C0, flattens toward the mean far from samples).
3b. **`RbfScatterField`** — radial-basis interpolation (`scipy.interpolate.RBFInterpolator`,
   an *optional* dep, lazily imported). Smooth, exact at samples, meshless, any N-D.
   Default kernel `thin_plate_spline` (parameter-free); `on_outside="clamp"` guards the
   convex-hull extrapolation. Rebuilt **only when the sampled positions/values actually
   change** (change-detection cache in `_RbfEngine`): a static field builds once and is
   reused verbatim across the whole animation even as the query moves.

**Datasets are callable as a field of position** (sugar that mirrors ftsl's `n(x,y)`):
`grid(x, y)` / `scatter(x, y)` — scalar args, `Signal`s, or a single `vec(...)` — builds
the matching interpolator node (`GridField`/`VecGridField` or `ScatterField`/
`VecScatterField`), a lazy `Signal` you evaluate with `.at(clock)`; `interp=`/`on_outside=`
(grid) and `power=`/`eps=` (scatter) pass through. `ds.sample(x, y[, clock=...])` is the
**eager** form — it builds the field and evaluates it in one call, returning a `float`
(scalar) or component `tuple` (vector), defaulting to a static frame-0 clock so a
non-animated dataset reads with no ceremony. (`__call__` lazily imports from `interp` to
sidestep the `data`↔`interp` cycle, the same pattern `_Transformable.transformed` uses.)

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

**Dataset placement transform (decoupled sampling curve).** A `Grid` / `Scatter` may
carry an optional local→world **`Transform`** (`.transformed(translate=, rotate=, scale=,
skew=)`, from §7d) that places its fixed *local* frame in world space. The stored samples
do **not** move; instead the field **inverse-maps** a world-space query into the dataset's
local frame (`Transform.inverse_apply`, threaded through `_local_query` in `interp.py` for
all six field classes) before interpolating. This is what **decouples the sampling curve
from the data object**: the curve lives in world space and stays put, so moving / resizing
/ skewing the data object changes *which local coordinate each world curve point lands on*
— and therefore the value it reads back (moving the object actually does something). The
inverse exactly round-trips the ftrace forward map; every transform parameter is a Signal
(so the remap animates and threads into the DAG). Applies to 2-D or 3-D datasets (2-D uses
the in-plane parameters only: translate/scale XY, rotate about Z, skew X-along-Y).

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

**Placement (J2).** An `Isosurface` carries an animatable `placement` `VecSignal`
(default origin). It offsets **both** the coordinate frame (read as `freq*(M·(x −
placement)) + drift`) **and** the `contained_by` box/sphere, so the container tracks
the pattern as a blob drifts/tumbles around a scene over the loop (`placement=(0,0,0)`
stays byte-identical to the un-placed emission). A `Room`/`Group` element gathers
several placed isosurfaces under one animatable rigid `Affine` frame `P`; on emit it
folds `P` into each child (frame's rotation into `M` as `M_eff = M·Pᵀ`, its translation
into the world placement `p_eff = P·p_local + T`), namespaces child names (`room/child`,
stacking for nested rooms), and emits a box container under a rotating room as the
conservative world AABB of the rotated local box. The frame must stay **rigid**
(orthonormal linear part) since the fold assumes `Pᵀ` inverts `P`; seamlessness comes
from translating on *closed* curves and rotating by integer turns. Factory pattern:
`make_gyroid(**params) -> Isosurface` + a `Room` driver (see
`examples/room_of_gyroids.py`).

### 7c. Function-driven materials
Reuse Good Photons' existing material-props-by-function (reflectance/color/IOR/etc.
over `x,y,z`/UV). Loom emits those expressions; adding `t` makes any property animate.

### 7d. Per-object transform (`Transform`)
A general **`Transform`** (`transform.py`) bundles four independently *animatable* fields —
`translate` (position), `scale` (size, per-axis or uniform), `rotate` (Euler XYZ degrees)
and `skew` (unit-diagonal upper-triangular shear `x'=x+a·y+b·z, y'=y+c·z`). Every field is
a `Signal`/`VecSignal`, so all of position/size/rotation/skew modulate over time. Attach one
to **any** element with `element.transformed(translate=, rotate=, scale=, skew=)` (or bundle
several children under a `Group`); it emits an ftsl `group { translate/rotate/scale/shear … }`
whose composed affine ftrace bakes into world-space prims at load. Order matches ftrace
(`src/mesh.h` `MeshXform`, `src/ftsl.h` `addGroup`): `world = translate + Rz·Ry·Rx·(scale ⊙
(shear · local))`. **`shear`** is a real ftsl `group` statement (added alongside
translate/rotate/scale); skew works for quad/tri/mesh/sweep geometry, and a `sphere{}` under a
non-uniform scale or shear is auto-tessellated by ftrace into a smooth-normal ellipsoid /
sheared quadric (uniform-scaled spheres keep the fast analytic path).
The same `Transform` doubles as a **dataset placement transform** via
`Transform.inverse_apply` (see §6): a `Grid`/`Scatter` carrying one is sampled by
inverse-mapping the world query into the dataset's local frame.

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
    vdbio.py                bake a field to a dense grid + write/read .vdb, read .nvdb (E4)
    axes.py                 axis-typed signals: broadcast/pin/mod + sample/reduce + lower-to-value-site (E5)
    anim.py                 curve→scene-variable go-between: config + sidecar + fan-out + named slots + live pipe (E2 s1–2)
    xvideo.py               two-pass spacetime transform video (M11)
    preview.py              resident ftrace -serve preview client (M12)
    viewer.py               native-viewer contract: build() loader + scene-introspection sidecar (F1) + .ftsl source emission (F7) + live re-introspection/emit server (F4/F7)
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
- **E4 — Volume `.vdb` write *and* read.** ✅ done (`loom/vdbio.py`). Bake any loom field to a
  dense lattice and serialise it as a **loom-native OpenVDB `.vdb`** grid that ftrace ingests
  directly (`density vdb:<path>` / `temperature vdb:<path>`), and read back both loom's own
  output and a useful slice of what real DCC tools emit — with **no OpenVDB/NanoVDB dependency**
  on either end. Tree is `Tree_float_5_4_3` (positive voxels vectorised into 8³ leaves under
  Internal<4>/<5>, empty leaves dropped).
  - **Write API:** `write_vdb(path, [VolumeGrid(name, values, box|transform=…), …])`,
    `bake_field(field, box, res)` (field/isosurface/callable → dense `<f4` + world box, reusing
    the `mcubes` samplers), `write_volume(path, *, box, res, **fields)` (several named fields
    over one box/res, e.g. a `density`+`temperature` fire pair). Codecs: `COMPRESS_ACTIVE_MASK`
    always, optionally `half=` (16-bit, `_HalfFloat` grid type), `zip=` (interchange only —
    ftrace has no ZIP) or `blosc=` (DCC-standard, and on ftrace's render path). Defaults are
    **byte-for-byte** the original ACTIVE_MASK/float32 output (test-asserted).
  - **Read API — two entry points, because the return type is the whole question.**
    `read_vdb(path)` → `{name: (array, box6)}` as always; `read_vdb_grids(path)` →
    `{name: ReadGrid}` carrying the index-space array, its `index_lo`, the full
    `VdbTransform` (index→world `A·i + t`, mirroring ftrace's `readTransform`) and the grid
    `background`.
  - **NanoVDB `.nvdb` read** (read-only; loom has no NanoVDB writer). `read_nvdb(path)` handles
    v32.6 float `5_4_3` in both accepted layouts — a `FileHeader` multi-grid container and a bare
    raw grid buffer — and `read_vdb_grids` **dispatches on magic**, so a caller wanting "read
    whatever volume this is" needn't know which it holds. A `.nvdb` is a *memory image*, not a
    stream: 32-byte-aligned PODs linked by signed byte offsets (`GridData / TreeData / RootData+
    tiles / Internal<5> / Internal<4> / Leaf<3>`), so the reader indexes at fixed offsets and
    walks. It mirrors ftrace's `src/vdbgrid.cpp` and therefore produces a **faithful dense bake**
    — inactive voxels take `background`, every non-child tile is expanded — deliberately unlike
    `read_vdb`, which keeps only active-positive voxels because it round-trips loom's own writer.
    That difference is load-bearing: `LeafData::getValue` ignores the value mask and
    `InternalNode::getValue` returns a tile's value active-or-not, so a mask-filtered read would
    drop real data (the `cloud.nvdb` sample carries 10 active lower-level tiles).
  - **Transforms.** All of OpenVDB's linear maps are decoded: the diagonal ones (Scale /
    Translate / UniformScale and combinations) and the general `AffineMap`/`UnitaryMap`.
    A **rotation costs the samples nothing** — an OpenVDB tree is a regular lattice in *index*
    space regardless, and the map only says where that lattice sits — so the dense array is
    unaffected and only the axis-aligned `box6` becomes inexpressible. Hence `read_vdb` still
    **rejects** a rotated grid (an approximate box would silently misplace every voxel, the
    worst failure mode) while `read_vdb_grids` reads it fine. `VolumeGrid(…, transform=…)`
    writes one, as an `AffineMap`. `is_diagonal` compares each off-diagonal against its own
    row's scale, so it is unit-free and tolerates the ~1e-17 crumbs a DCC leaves when it
    composes a 90°/180° rotation in floating point.
  - **Tests:** `tests/test_vdbio.py` (49) — bit-exact round-trip, world-box↔linspace positions,
    multi-grid named selection, sparse-empty-leaf drop, duplicate-name rejection, bake+write,
    each codec, four real third-party sample files, the rotated set (round-trip through
    `AffineMap`, `read_vdb` refusal, diagonal-`AffineMap` still yielding a box, and the two read
    entry points agreeing on diagonal files), and the NanoVDB set against the real
    `scraps/cloud.nvdb` — header metadata, `RootData` statistics, an **independent**
    breadth-first leaf walk (via `mNodeOffset[0]`, not the reader's child-offset descent) checking
    per-leaf min/max and every active voxel's landing index, the root-tile stride, the raw-buffer
    layout, magic dispatch, and the three rejections (non-NanoVDB, compressed, non-float). All
    **11 layout-constant mutations are caught** — two initially were not, which is what drove the
    independent-walk and tile-stride tests. Cross-validated through ftrace on CPU+GPU
    (`scraps/make_loom_vdb.py` → `scraps/loom_smoke.vdb` via `scraps/loom_vdb.ftsl`; sparse
    device path `1000/1000 bricks active`, energy `sum/emitted=1.000000`), and the rotated path
    by rendering one asymmetric volume under a diagonal vs a 45°-about-Y map
    (`scraps/vdb_rot_make.py`). The NanoVDB path was likewise checked end-to-end
    (`scraps/nvdb_roundtrip.py`): `cloud.nvdb` read by loom, re-emitted as a `.vdb`, and rendered
    in the same scene through ftrace's *independent* OpenVDB reader — means agree to 0.007%, and
    the per-pixel diff halves at 4× photons (8.65% → 4.35%), i.e. √N noise from the diverged RNG
    streams rather than a volume difference.
  - **Volume transforms — a volume is a *term*, not an API.** `loom.spatial.VolumeField` is the
    3-D twin of the `Image` leaf: a scalar `SpatialExpr` whose value at a world point is an
    imported grid's trilinearly-interpolated density (`ReadGrid.sample`, a port of ftrace's
    `VdbGrid::sample`). That one decision is what makes E4's *read → transform → write* "basis
    workflow" fall out of machinery that already existed — value ops and modulation are the
    spatial algebra, warping is the rebindable `x`/`y`/`z` children (as `Image` has `u`/`v`),
    meshing is `mcubes` (it takes any callable), and resampling is `bake_field`. Named
    `VolumeField` because `loom.scene.Volume` is already the `medium { }` scene element.
    - **Placement is lossless.** `translated`/`scaled`/`rotated`/`fitted` compose a world-space
      affine onto the grid's own index→world map (`VdbTransform.premultiplied`: `A' = M·A`,
      `t' = M·t + d`) instead of resampling — a VDB tree is a regular lattice in *index* space, so
      moving it costs nothing and touches no voxel (`v.rotated(37).read_grid.values is g.values`).
      Interpolation error enters exactly once, at the final bake: "discretize last", applied to
      volumes. Test-asserted both ways — 4×90°, 360°, translate-and-back and scale-and-back all
      reproduce the original array (~1e-15), while a single 37° rotation must *change* the field
      (so the round-trips can't pass vacuously) and conserve mass to 5%.
    - **`emit()` deliberately raises.** ftrace's pattern VM has no volume-sampling opcode, so
      there is no honest ftsl string; a `VolumeField` is bake-only and the error names
      `write_volume`. It is one of the two single-backend leaves in `spatial.py` (the
      other is `SigAt`, below).
    - Support added alongside: `VdbTransform.inverse_linear`/`to_index`/`premultiplied`,
      `ReadGrid.world_box` (AABB of the eight index-box corners — defined for a rotated grid,
      unlike `.box`) and `ReadGrid.with_transform` (shares the array).
    - **Found an ftrace bug** (fixed in v0.84.2): `VdbGrid::sample` and its CUDA twin clamped the
      stencil *indices* but not the interpolation *fraction*, so the half-voxel shell below index
      0 was dominated by the **second** voxel. Caught because a 360° rotation — necessarily a
      no-op — shifted the baked field by 0.32. See `known-issues.md`.
  - **Still open:** **Vec3** grids (blocked — no real vec3 file to validate against, and no
    consumer: ftrace is scalar-float-only); sparse *storage* as a backing, and transforms authored
    directly against it (now a storage optimisation, not a missing capability — reads of sparse
    files and every transform on them already work through the dense path).
    Writing `.nvdb` is deliberately not built (ftrace reads loom's `.vdb` directly).
- **E5 (foundation) — Axis-typed signals (one influence model).** ✅ done (`loom/axes.py`). Resolves E5's
  deferred open-q (node taxonomy + axis-set representation) with a small additive layer *on top of* the
  scalar `Signal` DAG (no churn; 891 prior tests stay green). An `AxSignal` is a pure function of a **point**
  (`dict[str,float]`, axis→coord) carrying `.axes` (`frozenset[str]`, its free variables). Composition unions
  axes ⇒ **broadcast** on unshared axes is implicit (a `{t}` node ignores a point's `s`), **pointwise** on
  shared ones, and the illegal cross-`t` op is *inexpressible* (no detector, no spatial/temporal type split).
  Nodes: `Ax`/`AConst`/`Lift` (bridge a legacy `{t}` `Signal`) leaves; `AFn`+arithmetic; `Sample(fn,arg)`
  (continuous `curve(t)`) + `.comp(i)` (`curve(t).y`) + `select(items,i)` (discrete `R.chan[i]`); the **only**
  cross-axis node `Reduce(body,axis,samples,op)` (`axes = body.axes − {axis}`, explicit); and the pin/mod edge
  model `Target(kind,[Binding(source,mode,gain)],base)` with target-declared neutrals (`ADDITIVE` 0 / `GAIN` 1
  / `BIPOLAR` ½). Reuses `alloc_id`/`children`/`detect_signal_cycle`/`walk`. **Follow-up 1 done** — the
  sample grammar now folds loom's own clock-parameterized producers: `CurveSample(curve,arg,*,clock_axis='t')`
  binds a `LoopCurve`/`FieldCurve`/`TrackedCurve`'s param axis *and* threads the clock (so an animated spatial
  curve is honestly `{s,t}`, a static one broadcasts over `t`); `RecordSample(record,channel,arg)` binds a
  `Record`'s static driver axis (`{driver}`, no clock); and `sample(obj,arg,…)` dispatches by duck-type. Both
  thread the loom node into the axis-layer `walk` like `Lift`.
  **Follow-up 2 done — scene value-sites route through `Target`.** `Lower`/`LowerVec` are the exact inverse of
  `Lift`: they bind an `AxSignal` back down to a clock-parameterized `Signal`/`VecSignal`, which is what every
  loom scene value-site (`Sphere.radius`, `Isosurface.iso`, a material colour, a camera position, …) consumes —
  so a `Target` reaches a scene variable *through here*, and E5's influence model becomes the authoring model.
  The site's clock axis (default `'t'`) is fed `clock.t`; every **other** axis the node reads must be pinned via
  `bind={'s': <coord or Signal>}` — a constant reads one arclength of a spatial curve, a `Signal` sweeps along
  it over the loop. Records-5a's scope rule ("a node's free variables ⊆ the axes in scope here") is enforced at
  **construction**, naming the unbound axes, rather than failing deep inside a render. `lower(node)` picks the
  scalar/vector form by probing at `t=0`; `LowerVec` evaluates the axis graph *once* per frame (the per-component
  `Lower` nodes still exist so `walk`/`detect_signal_cycle` and ordinary `VecSignal` math see a normal vector).
  Routing is **one memoised hook**, `signals.core.lower_axsignal`, consumed by `as_signal`, `VecSignal.of`,
  `ftsl_emit.site_node` (→ `num`/`vecn`/`value_token`) and `Element.roots()` — so no element constructor changed,
  and *every* value-site accepts an axis node uniformly. Memoising the lowered node on the axis node is required,
  not cosmetic: node identity is the per-frame `Cache` key **and** `roots()` must hand the cycle detector the very
  node emission will evaluate. Sugar: `mod(src, gain)`/`pin(src, gain)` build `Binding`s, `Binding` coerces its
  source via `as_ax` (which now also `Lift`s a legacy `Signal`), and a `GAIN` target with a negative source now
  raises a domain error instead of silently producing a complex number. Tests: `tests/test_axes.py` (55).
  **Follow-up 3 done — the on-disk projection of axis annotations.** The resolution of the deferred open-q's
  second half is a *decision plus an implementation*. **`.ftsl` carries no axis annotation, deliberately**: it
  is a **bound**, per-frame projection — by emit time the clock axis is fixed to `clock.t` and every other axis
  pinned by `bind=`, so an `{s,t}` node has already collapsed to a number. ftrace renders one frame and has no
  notion of an axis; annotating its language would make `.ftsl` an animation format and move the animation
  authority out of loom (core ideas 2 and 5). The on-disk projection that *does* need the annotation is the
  **viewer introspection sidecar** (F1/F5) — what an editor reads. `loom.axes.axis_annotation(node)` and
  `binding_edges(target)` are the model's own serialisers (the model owns its projection; `loom.viewer` just
  merges the dicts), feeding sidecar **v2**: a DAG node carries its free `axes`, plus `target_kind`/`neutral`
  (a `Target`'s declared quantity), `reduces`/`reduce_op`/`samples` (the one cross-axis node), `component` /
  `channel` / `leaf_axis`, and — on the two bridge nodes — `site`, `clock_axis`, `bound_axes` and `source_axes`
  (the value-site's whole axis scope). An edge out of a `Target` carries the `mode` (`pin`/`mod`) and `gain`
  that a plain child list *cannot* express (sources hang off `Binding` records, so a generic walk saw only
  anonymous inputs), and is named `mod[i]`/`pin[i]`. `src/viewer_gui.cpp`'s F5 panel renders all of it — an
  axis chip (`axes {s,t}`), a one-line kind/scope caption, and `mod[0] x0.8` on the input pin. Purely additive:
  a v1 reader ignores the new keys. Tests: 8 in `tests/test_viewer.py`. **E5 is complete.**
- **E2 (slices 1–2) — N-D curve → scene-variable go-between.** ✅ done (`loom/anim.py`). The channel-a config
  model + JSON sidecar + channel-b value fan-out — the pure-Python core of the animation go-between (resolves
  E2 OPEN Q1/Q2: config in a loom struct with a serialized sidecar; go-between = loom). `CurveDrive(dims,
  points, bindings, mode, closed)` holds the dimension count, the static starting control points (point
  *modulation is out* — the editor owns the time axis), and `ChannelBinding(channel, target, mode, gain,
  kind)` associations whose `mode`/`gain`/`kind` are exactly the E5 pin/mod edge attributes (so E2's
  value-routing *is* the E5 influence model — the "E5 unifies E2/E4" tie-in). Sidecar `save`/`load` is
  versioned JSON with an **atomic** temp+`os.replace` write ("scene proposes, editor disposes" round-trip).
  `sample(t)` is a uniform Catmull-Rom (loom-side preview; ftrace's editor is the live sampling authority);
  `apply(values, bases)`/`frame(t)` fan the sampled channels out to `{target: value}`, composing
  multi-channel targets through an E5 `Target` of the declared kind.
  **Slice 2** resolves "how a binding target names a real scene value-site" via **named animatable slots**
  (option b, `RefSignal`-style): a `Slot(name, default)` *is* a `Signal`, so dropping it in any signal-valued
  scene param means the scene's own `roots()`/`walk`/`emit` discovers and bakes it — zero emit-path change.
  `collect_slots(scene)` groups slots by name; `SceneDriver(scene, drive, bases, strict)` fans a `CurveDrive`
  out into the named slots (`set_values` pushes, `emit_frame` emits with a **fresh** `Cache` since a slot's
  value is pushed, not clock-derived) and its `default` doubles as the target's `mod` base. `LiveSession` +
  `serve_live(session, in, out)` are the editor↔loom **live-value channel**: a newline-delimited-JSON stdio
  loop (the `PreviewServer` precedent, editor→loom direction) with `frame`/`config`/`bindings`/`points`/
  `save`/`quit` commands, each a pure `dict`→`dict` `handle()` so the protocol is unit-testable without a
  pipe. Tests: `tests/test_anim.py` (19) + `tests/test_anim_live.py` (23).
  Remaining slice: (3) the interactive ftrace `camera_curve` **editor** generalization (seed from / write
  back the sidecar, drive arbitrary scene variables) — the C++ part, best done with the user present.
- **F1 (native viewer — the loom↔viewer data contract).** ✅ done (`loom/viewer.py`). The §F native viewer is
  a C++ process; loom is Python, so (per the locked architecture) loom exposes a scene via a **`build()`
  load contract** and a **JSON introspection sidecar**, not in-process sharing. `build(clock=None, **params)
  -> Scene` returns a *fresh* Scene per call (side-effect-free at import — no module-level `scene`), so the
  viewer re-derives geometry live (scrub/param/re-tessellate). `load_build(path)` imports a scene file and
  returns its `build`; `ViewerModel(build, **params)` (or `.from_file`) wraps it — `.scene(clock)` builds,
  `.declared_params()` surfaces the build's keyword defaults as UI controls, `.introspect(clock)` /
  `.save_sidecar` produce the sidecar. `introspect(scene)` enumerates: `objects` (geometry elements, Groups
  recursed, each linking the `datasets` it references by node id), `datasets` (every `PointPath`/
  `TrackedPath`/`Grid`/`Scatter` reachable — dim/shape/channels/etc.), minimal `camera`/`lights`, and the
  modulator `dag` (`nodes` = id+op+label, `edges` = child→parent; per-edge **param** labels are §F5's job).
  Tests: `tests/test_viewer.py` (22, incl. the F2-slice-A curve geometry). **F2 slice A (loom)** extends the
  sidecar so every path/tracked_path dataset carries real geometry: `control_points` (animated control points
  at the frame's clock) + a display `polyline` (96 samples through the engine's own `eval_curve`, closed
  spines wrapping). **F2 is complete:** the C++ host is a **`-viewer <sidecar.json>`** mode of the ftrace
  binary (Win32 + Direct3D 11 + Dear ImGui, `src/viewer_gui.cpp`): it reads the sidecar with ftrace's
  vendored `minijson` and shows Scene/Objects/Datasets panels + the full N-D curve pane — a **3-of-N dim
  picker** (view-only re-projection over the curve's full N-D coordinates), **index markers** along the
  curve, and **stereo** (mono / red-cyan anaglyph / wall-eyed L|R / cross-eyed R|L, with an eye-separation
  slider). **F3 is complete:** `tracked_path` datasets now also emit a `channels` array (each track
  sampled along the same curve parameter as the polyline), and the viewer compiles vendored **ImPlot** to
  draw **scroll-locked strip charts** below the 3-D pane — one per curve dimension + one per channel
  component, sharing a linked X axis and a draggable index line wired to the 3-D index dot. **F5 is
  complete:** `_describe_dag` tags every edge with the destination `param` it feeds (identity-matched to
  the attribute the child is stored under), and the viewer compiles vendored **imnodes** to draw a
  **Modulator DAG** panel — nodes titled `<op> #<id>`, one labelled input pin per incoming edge, longest-
  path layering from leaves to driven params. The layering **wraps each level into sub-columns** against
  the pane's available height (measured from the node rects imnodes actually produced — read back via
  `GetNodeDimensions()` after `EndNodeEditor()` and re-wrapped once), so a graph with dozens of leaves
  doesn't run off the bottom; and since this imnodes build has no zoom, the panel supplies one by scaling
  the ImGui font size + `NodePadding` (wheel 15–300%, `fit` = iterative width-fit solve, `100%`,
  `re-layout`, and a full-window `maximize` / Esc mode). **F6 is complete:** `_describe_dataset` now emits real field
  geometry — scatter `points`+`values`, grid `axes`+flat C-order `values` (scalar values normalised to
  1-lists) — and the viewer's **Fields tab** (`collectFields`/`drawFieldPane`) renders those sample points
  in the shared 3-D orbit view (grid node positions reconstructed from axes+shape in C-order), with 3-of-N
  dim pickers, a heatmap-channel / ch0·1·2→RGB colour selector, click-to-inspect, and per-extra-dim slice
  sliders for N-D grids. **F4 core is complete:** `_describe_element` emits each `SweptMesh`'s tessellated
  `mesh` (`vertices`/`faces`/`uvs`, from `sweep_rings`+`skin_rings` at the clock), and the viewer's
  **Meshes tab** (`collectMeshes`/`drawMeshPane`) draws it as a shaded, painter's-depth-sorted triangle
  surface with two-sided lambert lighting, a wireframe overlay, and grey/per-object/UV-checker/**texture**
  colouring (orbiting is view-only). **F4 textures are complete:** `introspect` emits a `materials` list
  (type/props + the resolved `texture` each binds) and a `textures` list — image skins as
  `file`/`encoding`/`filter`/`wrap`, formula skins as their three `r`/`g`/`b` UV expressions + `res`.
  `_describe_texture` **bakes** a formula channel to its ftsl source at the clock (`ProcTexture._chan_str`
  + an `EmitCtx`), because a material *bundle* whose colour slot is a `SpatialExpr` lowers to a
  `ProcTexture` holding live expression objects — those are neither JSON-serialisable (the sidecar dump
  used to die outright) nor compilable by the viewer. In C++, `SkinLib` decodes each texture into a D3D11
  SRV — images through ftrace's own `Texture::load`, formulas baked on the CPU through ftrace's own
  `compilePatternExpr`/`patternEval` (with a `PatTexScope` so `tex:<name>(u,v)` resolves against images
  declared above, exactly as in `FtslLoader::addTexture`) — and the Meshes tab draws each triangle at its
  interpolated per-vertex UVs. Unusable skins degrade to grey with a printed reason. Still deferred:
  **off-thread re-tessellation when rotating into a parameter dim** (needs the live viewer↔loom channel,
  since the static sidecar can't re-bake geometry).
  **F7's MC-mesh fallback is complete:** `_describe_element` bakes each `IsoMesh`'s field to a
  marching-cubes mesh (`_iso_mesh_geometry`→`mcubes.mesh_field`) into the object's `mesh` key, so the
  existing Meshes tab draws the isosurface with no C++ change. **F7's primary path is also complete:**
  `save_sidecar` now emits the scene's `.ftsl` beside the JSON (via `scene.emit`) and records its path
  under a `source` key (`emit_source=True` by default); `ViewerSession` gains an **`emit`** command that
  bakes the scene to `.ftsl` for a given clock/params. In C++ the viewer parses that `.ftsl` with ftrace's
  own `ftsl::load` and adds a **Render tab** that raymarches the real field in-process via
  `renderIsoPreviewCuda` (the `-raster-gpu` sphere-tracer — no tessellation), driven by an orbit camera and
  blitted into a D3D11 texture. This closes the last big open §F piece; the `emit` command also lays the
  groundwork for F4 off-thread re-tessellation over the live channel.
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
  - **J3b item 3 — Surface leaf family + binding-by-substitution.** ✅ (spatial.py
    foundation). The coordinate leaf `_Coord` is generalised to a public `Surface` family:
    `X`/`Y`/`Z` (axes 0/1/2, both backends) plus the *surface* params `U`/`V` (the ftrace
    pattern vars `u`/`v` — **emit-only**, no numpy twin) and the material *albedo* `A` (a
    pure binding placeholder — ftrace's VM has no `a` variable, so emitting a bare `A`
    raises; it must be substituted or defaulted first). `SpatialExpr.free_inputs()` reports
    the bindable named leaves (`{u,v,a}`; `include_coords=True` adds `x/y/z`) and
    `SpatialExpr.substitute({name: expr})` rewrites them by name (functional — the original
    tree is untouched). This is the substrate for **materials-as-bundles** (`gold(u=v, a=1)`
    / `(a=x*.5)` authoring): loom resolves every binding to a concrete field in real ftrace
    variables **at emit**, so it never writes literal bundle syntax and every intermediate
    stays renderable — needing **zero** ftrace C++ changes. Tests: `test_spatial.py`
    (uv-emit, no-numpy-twin, albedo-raises-until-bound, free-inputs, substitute rewrite/
    partial/nested).
  - **J3b item 3 — materials-as-bundles.** ✅ (`scene.py`). A `Material` property may now
    be a `SpatialExpr` field (scalar) or a tuple of them (a colour), making the material a
    **parameterized bundle**: `Material.free_inputs()` is the union of its fields' bindable
    leaves and `mat(u=v, a=1)` / `mat(expr)` (positional, sole free input) **applies** it by
    substituting across every field — pure functional rewrite, so a bound material is an
    ordinary one whose fields are concrete formulas. Unbound `u`/`v` stay the surface params;
    an unbound albedo `A` resolves to the material's `albedo_default`. Adding a bundle to a
    `Scene` **expands** each field to a renderable companion (`Material.expand`): a colour
    slot (`reflect`/`transmit`/`emit`/…) → a `ProcTexture` baked over surface `u`/`v` (now
    accepts `SpatialExpr` channels, re-baked per frame with time coefficients folded in, its
    `roots()` surfacing them for cycle-check); a scalar slot → a live `FuncPattern`. The two
    slot kinds have genuinely different coordinate reach, and only the colour one is
    restricted: a colour skin *bakes* into an image indexed by `u`/`v`, so world `x/y/z` in a
    colour field raises. A scalar slot stays live and ftrace evaluates it via `patCtxFromHit`
    (`src/scene.h`), which supplies world position, the field value, the hit normal **and**
    surface `u`/`v` — so a scalar field may mix all of them (`scenes/uv_native.ftsl` ships
    `weight_map pattern:uvcheck8` over `floor(u*8)`). Validated: a bundle scene emits
    byte-for-byte renderable `.ftsl` that ftrace parses & renders identically to the
    hand-authored `func_skin` path. Tests: `tests/test_material_bundle.py` (free-inputs union,
    keyword/positional/partial application, arbitrary-expr RHS, colour→ProcTexture /
    scalar→FuncPattern lowering, colour-slot world-coordinate rejection, scalar-slot u/v
    acceptance, Scene expansion ordering, animated re-bake, roots).
  - **J3b item 3(b) — `Image`, a photograph as a TERM in a formula.** ✅ (`spatial.py`
    + `scene.py`, and a new ftrace opcode). The complement of `skin()`: `skin()` binds a
    whole image to a material slot, whereas `Image("bark.png")` is a scalar *leaf* that
    composes — `0.05 + 0.9 * Image("grime.png") * (0.5 + 0.5*sin(30*X))`. This is the one
    part of item 3 that was **not** zero-cost on the renderer: no pattern-VM op could
    sample a texture, so emitting one would have produced un-renderable `.ftsl`. Added
    `PatOp::Tex` (`src/pattern.h`), spelled `tex:<name>(u, v)`, with the texture index
    carried in the existing `PatNode::a` double (the trick `PatOp::PovFn` already uses),
    so `PatNode` still uploads to the device verbatim. `pattern.h` stays free of
    `texture.h` — sampling goes through an opaque `PatCtx::texFn` hook that `scene.h`
    installs (`bindPatTex`). Compile-time resolution is opt-in per value site via a
    `PatTexScope`, granted only where a shading context exists (pattern blocks, record
    stops/drivers/overrides, procedural texture channels), so `tex:` in a field formula
    or medium program is a **clear compile error, never a silent zero**. GPU twin in
    `dPatternEval` (`render_cuda.cu`) over the existing `dTexScalarAt`.
    On the loom side: `Image`'s `u`/`v` are ordinary sub-expressions (so the lookup is
    warpable and `substitute` reaches into it, which is how a bundle's `u=`/`v=` binding
    flows in); `_auto_name()` derives a deterministic `img_<stem>_<hash>` from path +
    sampler settings, so identical images share one declaration and differing samplers
    don't collide; `SpatialExpr.image_textures()` collects the needed `Texture` blocks
    and `Scene.add` declares them automatically (an explicit declaration of the same name
    wins in either order) — otherwise every image term would emit a dangling reference.
    `eval_np` is a faithful port of `Texture::sampleRgb`/`scalarAt` (half-texel offset,
    `v`-flip, repeat/clamp/mirror, mean-of-linear-RGB), so the two-backend rule holds.
    `encoding` defaults to `"linear"`, not `"srgb"`: a value used as a *number* wants the
    stored levels. Validated end-to-end: the loom-emitted `.ftsl` renders **bit-for-bit
    identically** to the hand-authored `scraps/texop_pattern.ftsl`, and the composed case
    (`scraps/loom_image_mixed.ftsl`) matches CPU↔GPU to MC noise.
    Tests: `tests/test_image_term.py` (emit shape, warped/rebound coordinates, auto-name
    determinism & separation, `image_textures` dedup incl. nested leaves, Scene
    auto-declaration + ordering + explicit-wins, scalar-slot acceptance, and `eval_np`
    nearest/bilinear/wrap/sRGB/shape against hand-computed values).
  - **Retime / 4-D time-shear — `SigAt`.** ✅ (`spatial.py`, on top of
    `signals/retime.py`, § 4). The spatial half of retiming, and the reason the whole
    feature was worth building. A bare `Signal` coerced into the spatial algebra becomes
    `_Sig`, which bakes **one number per frame** — the entire field shares the modulator's
    current value. `SigAt(sig, when)` instead reads the modulator at a phase that is
    *itself a field*, so different points of space see different **moments**:
    `SigAt(Sine(cycles=3), T - X/4.0)` is a wave whose phase lags with distance. It is an
    ordinary `SpatialExpr` leaf, so warping, `substitute`, meshing (`mesh_field`) and
    baking (`bake_field`/`write_volume`) all apply unchanged.
    - **Single-backend, deliberately.** `emit()` raises: ftrace evaluates a pattern per hit
      and has no access to loom's modulator DAG, so a per-point signal read has no ftsl
      spelling — and baking one number would silently *drop the shear*, which is the whole
      effect. The error names the discretise-then-render route (`mesh_field` /
      `bake_field` / `write_volume`), the same workflow `VolumeField` uses.
    - **Cost is bounded and stated.** `eval_np` groups the phase field with
      `np.unique(..., return_inverse=True)`, so the signal graph is evaluated once per
      *distinct* phase, each inside its own `Cache.scope`; `quantize=k` snaps phases to `k`
      levels and caps it at `k`. (Wrapped phases that coincide share a scope — on a closed
      clock `t=1` and `t=0` are one sample.)
    - Tests: `tests/test_retime.py` (27) — shear-vs-flat anti-vacuity, constant phase ≡ a
      plain coefficient, quantize call-counts, loop seamlessness, emit-raises, `_is_time`
      /`_rebuild`, non-finite phase field, and **cache non-poisoning** for a *sub-frame*
      shear where every sample lands on `clock.frame` (the only case a frame-keyed memo
      cannot survive). Mutation-checked: degrading `Cache.scope` to the parent cache, or
      dropping the driver edge from `children()`, each fails a test.
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
