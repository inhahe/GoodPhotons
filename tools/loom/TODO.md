# Loom — TODO

## Emit ftrace's native `curve` fiber primitive

**✅ DONE.** ftrace 0.150 added `curve { }` (FTSL §8.6) — control points + a radius,
flattened at load into a watertight chain of round cones — and 0.151 put it on the GPU.
Loom now emits it directly as a **`Strand`** element (`loom/scene.py`), with `strand()`
and `hair()` factories and the control-point maths in `loom/strands.py`. That retires two
workarounds for the missing primitive: `tube()` tessellated a circle into a per-frame OBJ,
and `Beads` approximated a curve with a string of spheres (its own docstring called itself
"the simplest way to render a 3-D closed curve before the sweep engine exists"). Both
remain — `SweptMesh` is still the right answer for a non-circular profile, a twist, or
when the triangles are the product — but a round fiber is now the renderer's job.

The load-bearing decision is the **basis**, and it is core principle 4 again. An open
spine emits `catmull_rom`, which interpolates, so the strand passes exactly through the
sampled spine. A closed spine **cannot** use Catmull-Rom: `curve.h` builds an end span's
missing neighbour by clamping (`at(i) = pts[clamp(i)]`), so the seam span's tangent comes
from a duplicated phantom and the loop is C1-broken at exactly one place. So a closed
spine emits a **periodic uniform cubic B-spline** — control points wrapped by 3, giving
exactly N spans, never touching the clamp, C2 *through* the seam. Because a B-spline only
approximates, the control points are **solved** for (a cyclic tridiagonal system, by
Sherman–Morrison) rather than copied, so the loop is seamless *and* on the spine instead
of one or the other. Tests re-implement `curve.h`'s span evaluators and check the round
trip, with a deliberate control that the naive wrapped Catmull-Rom does kink
(`tests/test_strand.py`); the design record is `DESIGN.md` §7a′.

Also: `Strand._check_seam` warns once per element when a closed fiber's radius doesn't
close (`radius_tip != radius`, or a non-periodic `radius_profile`), mirroring
`SweptMesh._check_turns` — warn, never snap. And `eval_curve` now **clamps** an open path
instead of wrapping it, so an open strand can sample its endpoint and reach its tip.
Example: `examples/strand_loop.py`.

The native viewer keeps up (ftrace 0.153.0): the sidecar gains a `strand` record (spine
points + per-sample radii + `closed`/`basis`) and `viewer_gui.cpp`'s `strandToMesh` tubes
it for the **Meshes** tab along a rotation-minimising frame, spreading a closed fiber's
frame holonomy evenly round the loop so the preview tube closes without a sheared band at
the seam. That tube is preview-only — the renderer still traces the analytic cone chain.

---

## Per-object Transform (size / position / rotation / x·y·z skew), signal-modulatable

**✅ DONE (Phases A + B).** Loom now has a general `Transform` (`loom/transform.py`)
settable on **any** `Element` via `.transformed(translate=, rotate=, scale=, skew=)`
(or a whole `Group`); every field is a `Signal`/`VecSignal` so position/size/rotation/
skew all animate. It emits an ftsl `group { translate/rotate/scale/shear … }`. ftrace's
`group` grammar gained a `shear a b c` statement (`src/ftsl.h` `addGroup`), verified
bit-identical to hand-sheared geometry (`scenes/_shear_{a,b,c}.ftsl`; A==B mean|diff|=0,
A≠C). **Phase C is also done** — a `sphere{}` under a non-uniform scale or shear is
auto-tessellated by ftrace (`ftsl.h` `addTessellatedSphere`) into a smooth-normal
ellipsoid / sheared quadric mesh at load; uniform-scaled spheres keep the fast analytic
path. Verified: `scenes/_ellipsoid_test.ftsl` renders a flattened ellipsoid + a leaning
sheared sphere. (A *true* analytic ray-quadric ellipsoid — no tessellation, exact
silhouette, dual-backend — remains a possible future optimization, but is not needed for
the feature.)

<details><summary>Original plan (historical)</summary>

**Status today:** loom has **no** general per-object transform. Each geometry class
(`Sphere`, `Beads`, `SweptMesh`, `IsoMesh`, `Raw`, `Volume`, `Light`, …) emits its own
fixed `.ftsl` positioning; there is no size/position/skew wrapper that applies to "any
object." `SweptMesh` has its own `scale`/`twist`/`turns`, but that is sweep-local, not a
general transform.

### What to build

1. **`Transform` value type** carrying:
   - `translate` — `VecSignal` (position)
   - `scale` — `VecSignal` or scalar `Signal` (size; per-axis)
   - `rotate` — `VecSignal` (Euler degrees)
   - `skew` — x/y/z shear factors as a `VecSignal` (or a small shear-triple)

   Every field is a `Signal`/`VecSignal`, so all of position/size/rotation/skew animate.
   Roots must be exposed via `Element.roots()` for `detect_signal_cycle`.

2. **Apply it to any element.** Two options (do the wrapper; the `Group` is a nice-to-have):
   - Optional `.transform` / `xf` on the base `Element`; `emit()` wraps the child block in
     an `.ftsl` `group { translate … rotate … scale … <child block> }`.
   - A `Group(*children, translate=, rotate=, scale=, skew=)` element that emits one `group`
     around several children (ftsl `group` already nests and composes parent∘child).

3. **Emission target:** ftrace's `.ftsl` `group { translate <v> rotate <v> scale <v> … }`
   (see `src/ftsl.h` `addGroup`, ~line 2331). Position→`translate`, size→`scale` (per-axis
   supported), rotation→`rotate`. The engine bakes the composed affine into world-space
   prims at load — no per-frame scene-graph cost.

### ⚠️ Objections / constraints (read before implementing skew)

- **ftsl has no skew/shear statement.** The `group` grammar reads **only**
  translate/rotate/scale into its `Affine`. The engine's internal `Affine` (`src/linalg.h`,
  full 3×3 + translation) *can* represent shear, but there is **no `.ftsl` syntax to author
  it**. So x/y/z skew is **not emittable today** without an ftrace-engine change.
- **Analytic `sphere{}` rejects non-uniform scale.** ftsl explicitly fails with
  *"sphere under non-uniform scale would be an ellipsoid"* (`src/ftsl.h` `addSphere`,
  ~line 2044). So even non-uniform **size** — never mind skew — cannot apply to an analytic
  sphere. Only translate + *uniform* scale works for spheres.
- **Quads / triangles / meshes transform arbitrarily** — their vertices are pushed through
  the full affine, so per-axis scale *and* skew already work for tri/mesh geometry (incl.
  loom's `SweptMesh` sweeps). The gap is analytic primitives + the missing grammar.

**Consequence — "skew for *any* object" needs ftrace-side work:**
  1. Add a `shear` (or `matrix`) statement to the ftsl `group` grammar and thread it into
     the `Affine` (small C++ change in `ftsl.h`; rebuild + `VERSION` bump).
  2. Give analytic spheres a quadric/ellipsoid path (or have loom tessellate a skewed
     sphere to a mesh). Otherwise skewed/squashed spheres are impossible.

**Recommended phased scope:**
  - **Phase A (loom-only, ships now):** emit `group` translate/rotate/(per-axis)scale.
    Covers position + size + rotation for meshes/quads/sweeps; spheres limited to uniform
    scale. No skew yet.
  - **Phase B (small ftrace change):** add the `shear` grammar statement → real skew for
    mesh/quad/sweep geometry.
  - **Phase C (optional):** ellipsoid/quadric sphere so non-uniform scale + skew apply to
    analytic spheres too.

</details>

---

## Decouple Grid / Scatter sampling curves from the data object

**✅ DONE.** `Grid` / `Scatter` now carry an optional placement `Transform` via
`.transformed(...)` (`loom/data.py` `_Transformable`). The stored samples stay in the
dataset's fixed *local* frame; the field inverse-maps a world-space query into that frame
(`Transform.inverse_apply`, wired through `_local_query` in `interp.py` for all six field
classes — Grid/Vec/Scatter/Vec + RBF scalar/vec). So a world-space sampling curve stays
put while moving/resizing/skewing the data object changes the values it reads back. The
inverse is exact (round-trips the ftrace forward map) and every transform param may be a
Signal (threads into the DAG, cycle-checked). Tests: `tests/test_dataset_transform.py`.
Note: transforms apply to 2-D or 3-D datasets (2-D uses in-plane params only).

<details><summary>Original plan (historical)</summary>

**Problem today:** a `FieldCurve` (`interp.py` ~line 893) queries a `GridField` /
`ScatterField` whose query point lives in the **same coordinate frame** as the dataset
(`self.value = field(self.position)`; `GridField.children()` includes the grid). If the
"data object" were transformed as a unit (dataset + curve moving together), the sampled
values would not change — moving/resizing/skewing the data object does nothing.

**Goal:** the sampling curve should stay put in **world space** while the data object
transforms, so transforming the data object actually changes what values the curve reads
back.

### What to build

1. Give `Grid` / `Scatter` their own `Transform` (from the section above). The lattice /
   points live in the **data object's local frame**; the object's transform maps local→world.
2. The sampling curve's control points remain **independent world-space `VecSignal`s** — the
   curve is **not** a child of the data object's transform (do not tie the curve to the
   dataset `id` for positioning).
3. In `FieldCurve` / `GridField` / `ScatterField`, map the curve's world-space query point
   through the dataset's **inverse** transform into the dataset's local frame before
   interpolating: `local_q = dataset.xf.inverse().apply(world_q)`.

**Result:** translating/scaling/skewing the data object moves the field *under* the fixed
world-space curve, so the values the curve returns change — which is the intended behavior.
(`Affine.inverse()` already exists engine-side for reference; loom needs the equivalent in
`mathnd.py` / the `Transform` type.)

</details>
