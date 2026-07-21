# Loom — TODO

## Per-object Transform (size / position / rotation / x·y·z skew), signal-modulatable

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

---

## Decouple Grid / Scatter sampling curves from the data object

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
