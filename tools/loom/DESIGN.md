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
(a stored value may be a `Signal`, so control points animate):

1. **Point-path** — an ordered sequence of N-D points (a curve's control points).
2. **Grid** — N-D values on a regular lattice of *arbitrary rarity* (resolution).
3. **Scatter** — N-D values at arbitrary positions (no lattice).

Three **interpolators**, each exposed **as a `Signal`/field** (so an interpolator's
output can feed another modulator — "it's just another function"):

1. **`LoopCurve` (scribbles3 curve, generalized to N-D).** Port the midpoint
   quadratic-Bézier construction: each control point `B` with neighbors `A,C`
   produces an arc from `mid(A,B)` through `B` to `mid(B,C)`; wrap with modulo for a
   **seamless closed** curve (no seam angle to choose). Verified to generalize to any
   dimension (the construction is per-component). Open/closed both supported.
2. **`GridField`** — N-linear interpolation of grid values → a value anywhere in the
   volume.
3. **`ScatterField`** — smooth interpolation of scatter values (inverse-distance /
   RBF; **quality/speed tradeoff is an open tuning item**, see §11).

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
    scene.py                Scene, evaluate(), serialize/round-trip (new)
    ftsl_emit.py            snapshot → .ftsl text (new)
    drive.py                render_range, viewer, assembly, seed (new)
    mesh.py                 (deferred) adaptive marching cubes
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
- **M6 — Function materials.** Animated reflectance/color/IOR over space+time.
- **M7 (deferred) — Adaptive marching cubes**, if/when a field must be baked.

Each milestone: keep `known-issues.md` current, commit at green checkpoints, never
`git push`. Update this doc if the plan changes.
```
