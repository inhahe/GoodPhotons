# `gyroid_nd.py` — unified `--oscillate` grammar (design spec + migration plan)

Status: **design, pre-implementation.** This replaces the `--transform` system
(and its satellite flags) with one namespace of animatable "change-axes" driven by
`--oscillate` / `--lock`. Nothing here is coded yet.

---

## 1. The core idea

Everything that can vary over a video loop is a **change-axis** — an axis of variation
in an abstract "morph space." Dimension indices and named motions are the *same class
of thing*; there is no structural difference between them.

```
axes  =  { 0, 1, 2, … , N-1 }            spatial dims (the N-D lattice basis)
       ∪ { drift, rotate, tumble, bloom } orientation / envelope motions
       ∪ { freq, threshold, thickness }   scalar surface parameters (all surfaces)
       ∪ { …per-surface shape params… }   e.g. f_torus's major/minor radius (§7)
```

The last row is **surface-dependent**: the chosen `--surface` contributes its own
named shape parameters as extra axes (see §7). So the axis namespace — and therefore
what `--oscillate`/`--lock` accept — changes with the surface, which is exactly why
the help must be able to *list a surface's parameters* (§7).

The axis namespace is shared; it is acted on by **three verbs**, split along the honest
space-vs-time line (see §6 for why field and motion are separate commands, not one):

* **`--oscillate`** — *motion* (time): this axis *moves* over the loop.
* **`--lock`** — this axis is *held fixed* (pinned; the opposite of oscillate).
* **`--couple`** — *field* (space): which spatial dims share shape-terms in a frozen
  frame. Generalizes and retires `--coupling`/`--pair` (see §6).

`--transform` is **removed**. `drift`/`rotate`/`tumble`/`bloom` survive only as axis
names you drop into an `--oscillate` group.

**One comma never means two things.** `--oscillate`'s comma = "share one oscillator";
`--couple`'s comma = "share shape-terms." Because the two kinds of binding live in
*different commands*, no `f:`/`m:` prefix tagging is ever needed, and a mixed
`m:0,f:1` group — which is undefined anyway (a static field item has no clock to ride
an oscillator) — simply can't be written.

### Groups = composite oscillators

An `--oscillate` argument is a **space-separated list of groups**; each group is a
**comma-joined list of weighted items**:

```
--oscillate  <group>  <group>  …
     group  =  item , item , …  [ rate <expr> ] [ phase <expr> ]
     item   =  [ amp * ] axisname
```

* **comma** combines axes into **one composite direction** — a single oscillator
  swinging along the diagonal of its member axes. The members always peak together
  (one degree of freedom, a straight line through morph space).
* **space** separates **independent** oscillators — each group has its own clock, so
  its members drift in and out of sync with other groups (N degrees of freedom, the
  trajectory fills a torus / Lissajous figure).

So:

```
--oscillate tumble,bloom          # ONE oscillator on the tumble+bloom diagonal
--oscillate tumble bloom          # TWO independent oscillators (a torus)
```

### Amplitude = slope (per item); rate/phase = the clock (per group)

There is exactly **one** per-item magnitude knob and **one** shared clock per group:

* **`amp *`** (per item) — the axis's **amplitude**, i.e. its component of the
  composite direction (the *slope* of the diagonal). `2*tumble,bloom` → direction
  `(2, 1)`; `2*tumble,1.5*bloom` → `(2, 1.5)` = ratio 4:3. Default `1`. This is the
  same knob written `weight *` in earlier drafts — there is **no** separate
  "rate-scale."
* **`rate <expr>`** (per group) = full cycles of the shared clock over the loop.
  Default `1`.
* **`phase <expr>`** (per group) = starting offset in **radians** (`2*pi` = one cycle).
  Default `0`. `rate`/`phase` accept arithmetic (`pi/2`, `1/3`, …).

One group = one oscillator = one clock (rate+phase); each member merely *responds* to
that clock scaled by its amplitude.

`rate` and `phase` are **reserved words**: after a group, tokens are greedily absorbed
as `rate <expr>` / `phase <expr>` (either order) until the next token that isn't
`rate`/`phase`, which begins a new group. No axis may be named `rate` or `phase`.

### Worked examples

```
# tumble+bloom diagonal, 2 cycles, phase pi/2
--oscillate bloom,tumble rate 2 phase pi/2

# steeper toward tumble (direction (2,1.5)), 2 cycles, phase 0
--oscillate 2*tumble,1.5*bloom rate 2

# two independent oscillators, each its own clock
--oscillate  bloom,tumble rate 2 phase pi/2   drift,3 rate 1
#            └──── group 1 ────┘               └ group 2 ┘
```

`--lock` reuses the identical `item`/`group` grammar to pin axes (weights/rate/phase
are ignored there — a lock is just "held").

---

## 2. Axis catalog — what each axis is wired to

The CLI treats all axes identically; they differ only in *what they touch* and in
**kind**, which decides only whether `amp` is an independent degree of freedom:

| axis | kind | wired to | `amp` means |
|---|---|---|---|
| `0…N-1` | winder | phase of that spatial dim (slice drifts through it) | turns (≡ rate; leave 1) |
| `drift` | winder | phase advance of the group's dims | turns (≡ rate; leave 1) |
| `rotate` | winder | each wavevector rotates out of the 3-D slice | turns (≡ rate; leave 1) |
| `tumble` | winder | rigid Givens rotation of the whole slice basis | turns (≡ rate; leave 1) |
| `bloom` | swinger | classic↔full N-D crossfade envelope `sin²(πt)` | swing amplitude |
| `freq` | swinger | spatial frequency pulse | swing amplitude |
| `threshold` | swinger | level-set shift | swing amplitude |
| `thickness` | swinger | sheet half-width swell | swing amplitude |

**Two kinds — same knobs, one degenerate:** every axis takes `amp` (item) + `rate`
and `phase` (group), uniformly. The only difference is how many of those are
*independent*:

* **swinger** — value `= amp·sin(2π(rate·t + phase))`. `amp` (how far) and `rate`
  (how fast) are independent. `bloom`, `freq`, `threshold`, `thickness`.
* **winder** — value is an angle `θ = 2π(amp·rate·t + phase)` that cycles through whole
  turns and wraps. A rotation has no separate "size," so `amp` and `rate` collapse into
  one quantity (turns) — normally leave `amp` at 1 and steer with `rate`. This is
  today's integer `winding`. `drift`, `rotate`, `tumble`, and bare dims.

A composite group may mix kinds (`drift,freq`): one shared clock (rate/phase), the
winder part turns, the swinger part swings, at the same cadence.

---

## 3. Migration map (old flag → new grammar)

| old | new |
|---|---|
| `--transform drift` (default) | `--oscillate drift` (becomes the default group) |
| `--transform rotate` | `--oscillate rotate` |
| `--transform tumble` | `--oscillate tumble` |
| `--transform drift,tumble` | `--oscillate drift,tumble` (layered = one composite) |
| `--transform bloom` | `--oscillate bloom` |
| `--transform bloom --bloom freq,threshold` | `--oscillate bloom,freq,threshold` |
| `--transform bloom --bloom dims` | `--oscillate bloom` (the `dims` crossfade *is* `bloom`) |
| `--bloom-amp 1.5` (on `freq`) | `--oscillate 1.5*freq` (amplitude) |
| `--tumble-mode rotate` (default) | `--oscillate tumble` |
| `--tumble-mode slide --tumble-amp 0.3` | `--oscillate 0.3*tumble` treated as a bounded swing (see Q3) |
| `--tumble-lock 0,1` | omit `0,1` from the tumble group (optionally `--lock 0,1`) |
| per-dim `winding` (RNG-assigned) | `rate` on the group, or left random when unspecified |
| `--coupling cyclic` (default) | `--couple` default (auto cyclic ring over active dims) |
| `--coupling all` | one `--couple` cluster over all active dims (see §6) |
| `--coupling none --pair 0,1:on --pair 1,2:on` | `--couple 0,1 1,2` (explicit clusters) |
| `--pair 0,3:off` | drop `0,3` from its cluster / split the cluster |

Backward-compat: keep `--transform` as a **deprecated alias** that desugars to the
equivalent `--oscillate` string and prints a one-line deprecation notice, so existing
batch scripts and the pytest suite keep working through at least one release.

---

## 4. Open questions

**Q1 — Winder vs swinger unification. [RESOLVED]** Every axis takes the *same* knobs:
`amp` (item) + `rate`/`phase` (group). There is no separate "rate-scale." The only
difference is that a winder (a rotation) has no independent amplitude, so for winders
`amp` and `rate` collapse into one quantity (turns) and `amp` is normally left at 1.
See §2. Uniform grammar, one degenerate degree of freedom — no `swing`/`wind` modifier.

**Q2 — Does grouping fold in `--coupling`/`--pair`? [RESOLVED: separate `--couple`
command, phase 2 to build].** Folding field-coupling into `--oscillate`'s comma would
overload one comma with two meanings — *field* coupling (which terms exist in a frozen
frame — spatial, present even in a still) **and** *motion* coupling (which axes share
one oscillator — temporal). A mixed `m:0,f:1` group is undefined anyway (field items
have no clock). So field and motion get **separate verbs**: `--oscillate` (motion) and
`--couple` (field, §6). Each command's comma has exactly one meaning; no `f:`/`m:`
prefixes. Implementation still deferred to **phase 2** (it touches the field
machinery); `--coupling`/`--pair` keep working until then.

**Q3 — Randomization vs. explicit spec. [RESOLVED: base+override]** `gyroid_nd.py` is a
*variant generator*: it RNG-fills geometry, and `--oscillating <count>` / `--axis` are
base+override knobs over that randomness. `--oscillate <groups>` behaves the same way
as `--axis-default` + `--axis`: axes named in `--oscillate`/`--lock` are pinned to the
given amp/rate/phase; every unnamed axis stays RNG-randomized.

---

## 5. Staging plan (keep tests green at every step)

1. **Parser + model, no behavior change.** Add an `--oscillate`/`--lock` grammar parser
   producing a list of `Group{items:[(amp,axis)], rate, phase}`. Unit-test the
   parser in isolation (grouping, amplitudes, rate/phase, reserved words, errors).
2. **Desugar `--transform` → groups.** Route the *existing* transform code through the
   new group model (transform names → default groups with today's semantics). All 106
   existing tests must still pass unchanged.
3. **Wire swinger axes** (`freq`/`threshold`/`thickness`/`bloom`) so `amp` = amplitude,
   replacing `--bloom`/`--bloom-amp`. Add tests; keep old flags as aliases.
4. **Wire winder axes** (`drift`/`rotate`/`tumble`/bare dims) with per-group `rate`
   (= winding) and `phase`. Replace `--tumble-*`. Add tests; keep aliases.
5. **Flip the default** so `--oscillate` is primary and `--transform` prints a
   deprecation notice. Update README, module docstring, epilog, `--help`.
6. **(Phase 2) `--couple` cluster command** (§6): parse clusters → the existing
   `coupling_pairs()` edge set, retiring `--coupling`/`--pair` (kept as aliases).
7. **(Phase 3) Surface library** (§7): author the per-surface param-metadata table
   (+ generator + test), add `--list-surfaces` / `--surface-help`, then widen
   `--surface` to the full `iso.py` TPMS + `pov.py` `POV_FUNCS` set with the N-D and
   seamless-motion guards. Per-surface shape params become `--oscillate`/`--lock` axes.

Each phase is independently committable and leaves the tool fully working.

---

## 6. `--couple` — the field (spatial-coupling) command

`--couple` chooses **which dims share shape-terms in a frozen frame** — the spatial
structure of the isosurface, independent of any motion. It is the honest counterpart
to `--oscillate`: same axis namespace, but its comma binds dims in **space** (shared
terms) rather than **time** (shared oscillator).

### Grammar

```
--couple   CLUSTER   CLUSTER   …            ← spaces separate clusters
   CLUSTER  =  dim , dim , …                ← a set of mutually-coupled dims
```

No `rate`/`phase`/`amp` — a field relationship is static, so it has no clock and no
amplitude. A cluster is just an unordered set of dim indices. Space-separated clusters
are **disjoint coupling groups** (a dim belongs to at most one cluster).

### What a cluster means per surface

* **gyroid** — a cluster `{a,b,c,…}` contributes the `sin(u_i)·cos(u_j)` terms *among
  its own members*. Which pairs within the cluster is the cluster's internal scheme,
  defaulting to the **cyclic ring** (`(a,b),(b,c),…,(z,a)`) — today's `--coupling
  cyclic` restricted to the cluster; a `full` cluster would emit every unordered pair
  (today's `--coupling all`). A singleton cluster contributes nothing (a gyroid term
  needs two dims).
* **primitive (Schwarz P)** — no edges; each dim in any cluster contributes its own
  `cos(u_d)` node term. Clustering is therefore a **no-op** for primitive (membership,
  not pairing, is what matters) — consistent with today's "`--coupling`/`--pair` do not
  apply to primitive" warning.

### Defaults & override model (base+override, like `--axis`)

* **No `--couple`** → today's behavior: one automatic cyclic ring over all active dims
  (`--coupling cyclic`). The common "couple these N dims and animate them" case needs
  only `--oscillate`; coupling is implied.
* **`--couple 0,1,2  3,4`** → two disjoint clusters; the auto-ring is replaced by
  exactly these. Dims not named in any cluster are uncoupled (contribute no gyroid
  term) unless the RNG/`--axis` defaults add them — same base+override rule as `--axis`.

### Supersedes `--coupling`/`--pair`

| old | new |
|---|---|
| `--coupling cyclic` (default) | *(no flag — auto ring)* |
| `--coupling all` | `--couple <all active dims>` with the `full` internal scheme |
| `--coupling none` | `--couple` with no cluster covering a dim (empty field) |
| `--pair 0,1:on` | put `0` and `1` in the same cluster |
| `--pair 0,3:off` | keep `0` and `3` in *different* clusters |

`--coupling`/`--pair` remain as **deprecated aliases** that desugar to `--couple`
clusters, so existing scripts and the coupling tests keep passing through phase 2.

### Open sub-question (decide when building phase 2)

The per-cluster internal scheme (`cyclic` ring vs `full` clique) needs a surface for
its own syntax — e.g. a trailing tag `--couple 0,1,2,3:full` — versus a global
`--couple-scheme` default. Recommend a global default (`cyclic`) plus an optional
per-cluster `:full`/`:cyclic` tag, mirroring how `--coupling` is global today.

---

## 7. Surface library & per-surface parameters

`--surface` currently offers only `gyroid` / `primitive`. It will expand to the
**whole isosurface library already in the repo**, in two collections:

* **Periodic TPMS** — `tools/loom/loom/iso.py` `FIELDS`: `gyroid`, `schwarz_p`
  ("primitive"), `schwarz_d` ("diamond"), `neovius`. These are `2π`-periodic per axis
  and genuinely **N-D-generalizable** (symmetric sums), so the full N-D slice machinery
  (drift/rotate/tumble/bloom, coupling clusters) applies. They take **no shape params**
  beyond the shared `freq`/`threshold`/`thickness`.
* **POV-Ray builtins** — `tools/loom/loom/pov.py` `POV_FUNCS` (~78 functions:
  `f_torus`, `f_heart`, `f_superellipsoid`, `f_helix1`, `f_klein_bottle`, …), mirrored
  from `src/pov_functions.h` (auto-generated by `tools/pov_functions_gen.py` from
  POV-Ray's `fnintern.cpp`). ftrace evaluates these natively; each is called
  `f_name(x, y, z, P0, P1, …)` with **`arity − 3` shape parameters**.

### Two honesty caveats (from `pov.py` / `iso.py` / DESIGN.md §11.7)

1. **True N-D structure only for the generalizable subset.** The periodic TPMS above,
   plus the POV set `POV_ND_GENERALIZABLE` (`f_sphere`, `f_ellipsoid`,
   `f_superellipsoid`, `f_paraboloid`, `f_quartic_paraboloid`, `f_ovals_of_cassini`,
   `f_isect_ellipsoids`, `f_cross_ellipsoids`, `f_poly4`), actually fold into higher
   dimensions. Every other POV function is a 3-D field; its "N-D slice" is only an
   **affine remap** of (x,y,z) — tilt/shear/scale/drift — not new topology.
2. **Seamless motion needs periodicity.** A phase *drift* loops seamlessly only for
   periodic fields (the TPMS). Non-periodic POV functions loop seamlessly only under a
   coordinate transform that *returns to itself* over the loop — e.g. a Givens rotation
   through a whole `2π` (`rotate`/`tumble`) — not a linear drift ramp. So for those,
   `drift` is disallowed/degenerate and `rotate`/`tumble` are the valid motions.

### Per-surface params become axes

Each POV function's `arity − 3` shape params are **surface-specific axes** in the
unified namespace: with `--surface f_torus` you can `--oscillate <majorR>,<minorR>`
(swingers) exactly like `freq`/`threshold`. So selecting a surface *extends* the axis
set; the shared axes (`freq`/`threshold`/`thickness`, the motions, the dims) are always
present, the shape-param axes are added on top.

### Listing params in `--help` (the piece that needs building)

`POV_FUNCS`/`pov_functions.h` store only **arity** (the param *count*), not param
**names, meanings, defaults, or ranges**. POV-Ray's docs describe them (e.g.
`f_torus(x,y,z, P0=major radius, P1=minor radius)`), so we must **author a
param-metadata table** — `{func: [(name, description, default, [lo,hi]), …]}` — most
practically emitted by extending `tools/pov_functions_gen.py` to scrape/annotate the
POV docs, with a hand-maintained fallback. A test should assert every `POV_FUNCS` entry
has metadata with exactly `arity − 3` params (same drift-guard discipline the arity
table already uses).

With that table, add discovery commands:

* **`--list-surfaces`** — print every surface (grouped: periodic TPMS / N-D POV /
  affine-only POV), one line each with its param count and N-D status.
* **`--surface-help NAME`** (or `--help NAME`) — print one surface's full parameter
  list: each param's axis name, meaning, default, range, and swinger/winder kind, so the
  user knows exactly what they can drop into `--oscillate`/`--lock` for that surface.
* The main `--help` gains a pointer to both (since the axis namespace is
  surface-dependent, a static `--help` can't enumerate every surface's params inline).

### Staging

This is **phase 3+** work (after the core `--oscillate` grammar lands). Order:
(a) author the param-metadata table + generator + test; (b) `--list-surfaces` /
`--surface-help`; (c) widen `--surface` choices to the full library with the N-D and
seamless-motion guards from the caveats above.

---

## 8. GPU isosurface rendering — kill the per-frame tessellation cost

### 8.0 The problem this solves

Video frames go through the `-raster` path: `gyroid_nd._render_frame` runs
`ftrace -in frame.ftsl -o frame.png -raster`. That path is **CPU-tessellate → GPU
rasterize**: every frame, ftrace rebuilds the isosurface into a triangle mesh on the
CPU (`isomesh::marchImplicit`, marching **tetrahedra** — Kuhn/Freudenthal, watertight),
then hands the mesh to the CUDA rasterizer (`raster_cuda::renderFrame`). The CPU
tessellation is the bottleneck: ~3.15 s/frame at the default grid res (~2 M tris), redone
from scratch every single frame even though only a few parameters (and an affine
transform of x,y,z) change between frames.

Measured fact that reframes the whole thing: the **Python `.ftsl` generation is
negligible** — ~1.1 ms/frame total (build_scene 0.17, emit 0.06, write 0.87 ms over 60
frames; see `scraps/time_ftsl_gen.py`). So a C++/Cython rewrite of loom or a
marching-cubes `.pyd` would be pointless. **All the cost is inside ftrace (already C++),
in the CPU tessellation step.**

### 8.1 The key recon finding — the GPU already raymarches isosurfaces directly

ftrace's **forward GPU path already sphere-traces arbitrary isosurfaces on-device**, no
mesh required:

* `intersectImplicit` (device, `render_cuda.cu:1319`) — sphere-traces an implicit field.
* `dFieldEval` (`render_cuda.cu:1251`) / `dPatternEval` (`render_cuda.cu:2754`) — evaluate
  an **arbitrary ftsl expression** on-device from the postfix bytecode (not just the
  `f_*` builtins), so any surface we can write into `.ftsl` already runs on the GPU.
* `dFieldGradient` — analytic/numeric normals on-device.
* `povFnEval` (`POV_HD`, host **and** device) — the whole POV builtin library is already
  callable on the GPU.
* Isosurfaces do **not** trigger the CPU fallback on the forward GPU path.

So the user's idea — "feed the transformed isofunctions directly, skip tessellation, and
put the per-frame matrix transform of (x,y,z) into the function itself" — is **~80%
already built.** What's missing is a *deterministic primary-ray preview kernel* that uses
this machinery for the fast look-dev/video path the way `-raster` does today.

### 8.2 The plan (3 steps, increasing scope)

1. **Interim, free win — expose `-raster-iso` in gyroid_nd. ✅ DONE 2026-07-18.**
   ftrace already has `-raster-iso <n>` (grid resolution for the raster tessellation,
   default 96, `main.cpp:3187`). `gyroid_nd` never exposed it, so every frame tessellated
   at 96³. Added a passthrough flag `--raster-iso N` threaded `main → make_video →
   _render_frame`, which appends `-raster-iso N` on the raster path so a lower grid cuts the
   CPU tessellation cost *today*, with zero engine changes. Verified end-to-end (coarse
   gyroid at res 40) + 268 loom tests green.

2. **The real fix — a GPU deterministic primary-ray isosurface preview kernel.**
   Add a CUDA kernel in `raster_cuda` (a sibling to `renderFrame`) that, per pixel, casts
   the primary camera ray and calls the **existing** `intersectImplicit` +
   `dFieldGradient` + shading, i.e. a direct sphere-traced preview with **no
   tessellation at all**. Wire a mode (e.g. `-raster-gpu` / an isosurface fast path) so
   `gyroid_nd` video frames render this way. This removes the per-frame CPU mesh build
   entirely — the frame cost becomes GPU raymarch time, and the pipeline reuses code
   ftrace already ships. **This also makes "GPU marching cubes" moot for the preview
   bottleneck** — we don't need to port marching cubes to CUDA to speed up frames, because
   we skip meshing altogether.

3. **Follow-on — `PatOp::MatMulAdd` matrix intrinsic.**
   Today per-frame transforms are inlined as scalar arithmetic in the postfix bytecode
   (PatNode); there's no matrix op. Adding one opcode `PatOp::MatMulAdd` (matrix · vec +
   offset) makes the "bake the frame's affine transform of (x,y,z) into the function"
   approach clean and cheap instead of emitting a dozen scalar mul/adds per axis. Small,
   mechanical change touching the same five spots any new PatOp does: opcode enum
   (`pattern.h:29-50`), CPU eval switch (`pattern.h:112-172`), device eval switch
   (`render_cuda.cu:2754-2816`), the parser, and PatNode storage. Independent of steps
   1–2; do it when the transform-baking path wants it.

### 8.3 Is there still any case where ftrace WANTS to tessellate? — **Yes.**

Removing tessellation from the *video preview* path does **not** retire tessellation from
the engine. It's still needed for:

* **Actual mesh output / export.** The `marchImplicit` calls at `main.cpp:3522` and
  `main.cpp:3623`, together with `decimateAdaptive(mesh, ratio, im)` (`isomesh.h`), exist
  to produce **real triangle geometry** (STL/OBJ export, decimated meshes, anything that
  consumes polygons downstream). A direct raymarcher produces **pixels, not geometry**, so
  it fundamentally cannot serve these — if you want a mesh out, you must march.
* **The current `-raster` path itself**, until/unless the step-2 GPU raymarch kernel fully
  replaces it. Any workflow that keeps rasterizing meshes (or wants the exact raster look)
  still tessellates.

So **GPU marching cubes stays a legitimate — but lower-priority — future want**, purely to
*accelerate mesh export*, not to fix the per-frame video bottleneck (step 2 handles that by
not meshing at all). Don't build GPU marching cubes for the video path; do keep it on the
list for the export path if mesh-export throughput ever becomes a pain point.

### 8.4 Staging

Independent of the `--oscillate` grammar work. Order: (1) `--raster-iso` passthrough
(trivial, do anytime); (2) GPU direct-iso preview kernel (the real win — the per-frame
bottleneck fix); (3) `PatOp::MatMulAdd` (when transform-baking wants it). GPU marching
cubes: deferred, export-only, build only if mesh-export speed becomes a problem.
