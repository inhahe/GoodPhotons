# TODO — consolidated undone work

Single source of truth for everything planned-but-not-done, pulled together from the
project's scattered plan files. Check items off (`[x]`) as they land, and mirror the
status back into the originating file (`DESIGN.md`, `ROADMAP.md`, `OSCILLATE_GRAMMAR.md`,
`ROADMAP_heroroom.md`) when a whole section closes.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.
Origin tags point at the authoritative design text for each item.

---

## A. Camera-curve bridge + orientation axes  *(design locked in conversation; NOT previously written to any file — captured here so it isn't lost)*

**Context.** loom's `Camera` (`tools/loom/loom/scene.py` `Camera.emit`, ~line 396) currently
**bakes** a static per-frame `camera "name" { eye … look_at … up … fov_y … mode … film {res} }`
block — it animates by re-emitting numbers every frame and does **not** emit a real
`.ftsl camera_curve`. ftrace's `camera_curve` (`src/ftsl.h` `addCameraCurve`, ~line 2883)
today does a Catmull-Rom **position** spline + arc-length/density reparam + scalar tracks
(`roll_at`/`fov_at`/`zoom_at`/`fstop_at`/`focus_at`) + look modes (tangent default / `look_at`
/ `look_curve`+`look_point`) + world `up` + fold-robustness (`min_reach`/`look_smooth`). It does
**roll + aim**, not two free orientation axes.

**Goal.** (1) a loom `CameraCurve` element that emits a genuine `.ftsl camera_curve`; and
(2) ftrace-side **orientation axes** — a forward-direction curve and an up curve — with a
per-curve reference frame.

### The orientation model (locked)
Full 3-D camera rotation = 3 DOF. We author it as two independent axes; the third is derived:

- **forward** — 2 DOF (pointing direction). Authored **one** of three ways:
  - `fwd_at` **direction vector** (normalized), or
  - an **aim-point** (`look_at` fixed world point, or `look_curve` = a second spline of
    look-points) → `forward = normalize(target − eye)`, or
  - **omitted** → the path **tangent** (today's default, with `min_reach`/`look_smooth`).
  - *Direction-mode and aim-point-mode are two authoring conventions for the SAME forward
    axis — not redundant with each other, and neither is "up rotated 90°".*
- **up** — 1 DOF (roll about forward). Authored **one** of:
  - `up_at` **vector** curve (re-orthogonalized against forward), or
  - scalar `roll_at` angle about the frame's reference up (today's behavior), or
  - **omitted** → the frame's reference up.
- **right** — 0 DOF, **always derived**: `right = normalize(forward × up)`, then up is
  re-orthogonalized `up = right × forward`. Never authored.

### Reference frame (locked: **per orientation axis**, `travel | world`)
An orthogonal choice of what "reference up / straight ahead" *mean* before the `fwd_at`/`up_at`/
`roll_at` overrides apply:
- **`world`** — fixed world axes (a global up vector; today's behavior).
- **`travel`** — curve-relative **rotation-minimizing frame (RMF)** built by parallel transport
  (double-reflection / Bishop frame, **not** Frenet — no torsion flips). For a **closed** loop the
  RMF has holonomy (closure twist); **distribute the residual twist** evenly along the loop so the
  orientation returns to itself seamlessly (same technique the sweep engine's closed-spine frame
  uses, `DESIGN.md` §7a).

**This is an ftrace decision; loom mirrors it 1:1.** The orientation math lives in ftrace's
`camera_curve`; loom's bridge only emits `.ftsl` text and can express exactly what ftrace parses.
ftrace does **none** of this today (only tangent-look + world `up` + scalar `roll_at`), so we are
choosing ftrace's new behavior — not matching an existing one. **Decision: the frame is chosen
per orientation axis** (`fwd_at` and `up_at` each carry their own optional `frame travel|world`,
with a curve-level default), *not* one switch for the whole camera. Per-axis is strictly more
expressive — a single global frame is just "both axes set the same" — and it's the only way to
express e.g. *forward locked to a fixed world subject across the room while up rides the travel
frame so the shot still banks into turns*. It costs ftrace's parser one optional keyword per curve
instead of per block. loom exposes the same per-axis `frame` and emits it into each track.

### Tasks
- [ ] **ftrace: `fwd_at` vector track** on `camera_curve` — parse + store a per-keyframe 3-vector
      forward direction; sample on the same `u` as position; normalize; fall back to tangent/aim
      when absent.
- [ ] **ftrace: `up_at` vector track** — parse + store a per-keyframe 3-vector up; re-orthogonalize
      against forward; fall back to reference up (`roll_at` still composes on top).
- [ ] **ftrace: per-axis `frame travel|world`** keyword — `fwd_at` and `up_at` each select world
      axes vs RMF reference independently (curve-level default); global frame = both set the same.
- [ ] **ftrace: RMF construction** (double-reflection parallel transport) + **closed-loop twist
      distribution** for seamless closed curves.
- [ ] **ftrace: `right = forward × up` derivation** + up re-orthogonalization, roll composed on top.
- [ ] **ftrace: back-compat** — with no `fwd_at`/`up_at`/`frame` authored, behaves **bit-identically**
      to today (tangent look + world up + `roll_at`).
- [ ] **loom: `CameraCurve` scene element** — emit a real `camera_curve` from a `TrackedCurve`/points:
      position → `point`, speed/density track → `density_at`, roll track → `roll_at`, orientation
      tracks → `fwd_at`/`up_at`, per-axis `frame`. Mirrors ftrace's grammar 1:1 (no orientation
      semantics loom can't emit).
- [ ] **Docs** — README (ftrace camera_curve grammar) + loom docstrings; update `DESIGN.md` with a
      milestone (M13) once landed.
- [ ] **Tests** — loom `CameraCurve` emit golden; ftrace parse of `fwd_at`/`up_at`/`frame`; RMF +
      closed-loop seam; bit-compat when no axes authored.

---

## B. `gyroid_nd.py` unified `--oscillate` grammar  *(origin: `tools/loom/examples/OSCILLATE_GRAMMAR.md` — "Nothing here is coded yet")*

Replaces `--transform`/`--bloom*`/`--tumble*`/`--coupling`/`--pair` with one `--oscillate`/`--lock`/
`--couple` axis grammar. Each phase is independently committable and keeps tests green.

### Phase 1 — the `--oscillate`/`--lock` core (§5 steps 1–5)
- [x] **P1.1 Parser + model, no behavior change.** `--oscillate`/`--lock` grammar →
      `Group{items:[(amp,axis)], rate, phase}`. Unit-test parser in isolation (grouping,
      amplitudes, rate/phase, reserved words `rate`/`phase`, error cases).
      *Done 2026-07-18:* `OscGroup` dataclass + `parse_oscillate`/`parse_lock_axes` +
      safe arithmetic evaluator (`pi`/`tau`/`e`, `+ - * / ** %`) in `gyroid_nd.py`; pure
      parser, not yet wired to behavior. 17 unit tests in `test_gyroid_nd.py` (grouping,
      amplitudes incl. `2*pi*x`, rate/phase either order, reserved-word/duplicate/empty/
      bad-expr errors, lock flatten+dedup). 285 loom tests green.
- [x] **P1.2 Desugar `--transform` → groups.** ✅ 2026-07-18. Added `transform_to_oscillate(...)`
      + `oscillate_spec(...)` in `gyroid_nd.py`: behavior-preserving bridge re-expressing today's
      `--transform`/`--bloom`/`--bloom-amp`/`--tumble-*` flags as one canonical composite
      `OscGroup` per §3 migration map. Pure model, execution path untouched; 13 new tests, 298
      loom tests green.
- [x] **P1.3 Wire swinger axes** (`freq`/`threshold`/`thickness`/`bloom`), `amp` = amplitude;
      replace `--bloom`/`--bloom-amp` (kept as aliases). ✅ 2026-07-18.
      - Per-axis `Variant.bloom_amps: Dict[str,float]` + `_swing_amp()`; the three swinger functions
        read the per-axis override, falling back to the shared `bloom_amp`.
      - `--oscillate`/`--lock` argparse flags + idempotent `resolve_oscillate(args)` that maps the
        parsed group model onto the canonical `transform`/`bloom`/`bloom_amps`/`tumble_*` fields
        (the exact inverse of `transform_to_oscillate`), so `pick_variant` needs no new path.
        `--transform` default → `None` for clean mutual-exclusion; conflict guards for `--transform`
        + the legacy satellite flags; `amp*tumble` → slide mode; `--lock <dims>` → tumble lock.
      - **Validated:** 14 new tests incl. field-expression equivalence to the legacy `--transform`
        invocations; a real `--oscillate bloom,freq` render through the full CLI→ftrace pipeline;
        and a byte-for-byte `.ftsl` diff (`--oscillate` ≡ legacy, incl. the `1.5*freq` amp case).
        314 loom tests green.
      - *Deferred to P1.4 (guarded with clear "not yet" errors):* per-group `rate`/`phase` (the
        shared clock / winding override) and bare spatial-dim-index axes.
- [x] **P1.4 Wire winder axes** (`drift`/`rotate`/`tumble`/bare dims), per-group `rate` (= winding) +
      `phase`; replace `--tumble-*` (keep aliases). ✅ 2026-07-18.
      - `resolve_oscillate` emits three winder-clock outputs the picker honors, all no-ops on the
        legacy `--transform` path (those variants stay bit-identical):
        - `args.osc_dim_windings` — a **bare dim index** is the atomic winder: forced on and pinned
          to an **exact** integer winding `round(amp*rate)` (`--oscillate 3 rate 2` → dim 3 winds
          twice; `2*3` is identical since `amp≡rate`). Raises the dim floor; off-lock conflict errors.
        - `args.osc_max_winding` — an explicit `rate` on a **motion** group is the **ceiling** of the
          RNG-varied `1..N` winding cycle (overrides `--max-winding`, keeps the distinct-rate spread
          — "how fast, at most", consistent with a lone dim's exact rate).
        - `args.osc_phase` — a constant radians offset (`2π` = one turn) on the shared winding clock
          (drift/rotate/tumble), from a group's `phase`; shifts the loop start, keeps `t=0==t=1`.
      - One shared winding clock ⇒ conflicting motion rates/phases across groups are rejected, as is
        `rate`/`phase` on a swinger (fixed `sin²(πt)` envelope — later step). Bare dims with no named
        motion default to `drift`.
      - **Validated:** 12 new tests + a real `--oscillate drift rate 3` video through the CLI→ftrace
        pipeline. 324 loom tests green.
- [x] **P1.5 Make `--oscillate` the documented surface** ✅ 2026-07-18. `--transform` +
      satellites (`--bloom`/`--bloom-amp`/`--tumble-*`) hidden from `--help` (`argparse.SUPPRESS`)
      but still fully supported; passing `--transform` prints a plain one-line deprecation note.
      Migrated the module docstring examples/prose + epilog to the grammar, and the test suite's
      incidental `--transform` setup usages to `--oscillate` (23 via
      `scraps/convert_transform_to_oscillate.py` + 2 hand edits; the `bloom_amps`-representation
      and deliberate desugaring/equivalence tests intentionally stay on `--transform`). Default
      motion (neither flag given) is still `drift`. 324 loom green.

### Phase 2 — `--couple` field-coupling command (§6)
- [x] **P2.1** `--couple CLUSTER CLUSTER…` (comma-joined dims, space-disjoint) with per-cluster
      `cyclic`/`full` scheme (global `--couple-scheme` default + optional `:full`/`:cyclic` tag).
      `parse_couple`/`resolve_couple` → `couple_clusters` + forced-on `couple_axes` (fed to
      `forced_on`/`max_forced_axis` like `--pair …:on`, no new RNG draws). `coupling_pairs()`
      refactored around a shared `_scheme_edges()` helper: cluster path emits ring/clique edges over
      oscillating members in CLI order; empty clusters fall through to the legacy `--coupling`/`--pair`
      base-graph path bit-identically. **Decided: kept `--coupling`/`--pair` on their own path (they
      resolve over the post-RNG active set; `--couple` names dims at parse time — no clean desugar), so
      `--couple` is mutually exclusive with a non-default `--coupling`/any `--pair`.** `coupling_desc`
      + primitive-surface warning updated; docstring/epilog/help + OSCILLATE_GRAMMAR.md §6 updated.
      11 new tests, 335 green.

### Phase 3 — surface library & per-surface params (§7)
- [x] **P3.1** Author per-surface param-metadata table `{func:[(name,desc,default,[lo,hi]),…]}`
      (extend `tools/pov_functions_gen.py` + hand fallback) + a test asserting every `POV_FUNCS`
      entry has exactly `arity−3` params.
      *Done 2026-07-18:* metadata lives in `loom/pov.py` (Python side, for `--surface-help`), not the
      C header (the VM only needs arity). `_AUTHORED_PARAMS` hand-authors real
      `(name,desc,default,(lo,hi))` for the well-documented / N-D-core shapes (f_sphere, f_ellipsoid,
      f_superellipsoid, f_paraboloid, f_quartic_paraboloid, f_rounded_box, f_torus, f_heart,
      f_noise_generator) + the 0-param helpers (f_r/f_th/f_ph/f_noise3d); every other `POV_FUNCS` entry
      falls back to honest generic `p0..` placeholders via `_generic_params`. `POV_PARAMS` is built to
      match `arity−3` by construction; `pov_params(name)` accessor returns a copy. Exported from
      `loom/__init__`. 6 new tests (completeness drift-guard, well-formedness: valid/unique axis names +
      default∈[lo,hi], spot-checks, unknown-name reject, copy-safety), 349 loom green.
- [x] **P3.2** `--list-surfaces` + `--surface-help NAME` discovery commands; main `--help` pointer.
      *Done 2026-07-18:* surface catalog in `gyroid_nd.py` (`_TPMS_CATALOG` + `POV_FUNCS`), grouped by
      N-D honesty class (`surface_group`: periodic / nd_pov / affine_pov). `--list-surfaces` prints all
      82 surfaces (4 periodic TPMS with `[nd] [loop]`, 9 N-D-generalizable POV, 69 affine-only POV) with
      each one's shape-param count; `--surface-help NAME` prints one surface's params via `pov_params`
      (axis name/meaning/default/range) or the shared-axis note for a param-free TPMS, resolving the
      `schwarz_p`→`primitive` alias. Both are early-exit (return 0 before any generation), ASCII-safe
      for the Windows console, and cross-referenced from `--surface` help + the epilog. 11 new tests,
      360 loom green.
- [x] **P3.2b** Generalize the swinger envelope to carry its own `rate`/`phase`, uniform with
      winders/bloom. *Done 2026-07-18:* each swinger's bloom is now
      `w(t) = 0.5·(1 − cos(2π·rate·t + phase))` (`_bloom_env_p`), keyed by `Variant.bloom_rates`
      / `bloom_phases` (`"dims"` for `bloom`, own name for freq/threshold/thickness). `rate`/`phase`
      are read from the swinger's group and no longer rejected. Default (rate 1 / phase 0) is
      byte-identical to the legacy fixed `sin²(πt)` envelope; integer rate loops seamlessly, a
      non-integer rate pulses faster but breaks the loop and `main()` warns. 6 new tests, 365 loom green.
- [ ] **P3.3** Widen `--surface` to the full `iso.py` TPMS (`gyroid`/`schwarz_p`/`schwarz_d`/`neovius`)
      + `pov.py` `POV_FUNCS`, with the N-D (`POV_ND_GENERALIZABLE`) and seamless-motion (periodic-only
      `drift`) guards. Per-surface shape params become `--oscillate`/`--lock` axes.
      **Design locked (2026-07-18), building as a parallel POV emission path:**
      (1) *Solid vs shell* — POV shapes render **solid** (`f - threshold`, no abs); a small tagged set of
          genuinely-thin surfaces (klein_bottle, boy_surface, enneper, the `*_2d` curves, ...) render thin;
          a `--shell` flag forces any shape hollow. TPMS keep the abs()-shell.
      (2) *Gradient bound* — **per-function table** derived with SymPy + `mpmath.iv` (interval arithmetic)
          from the exact bodies in `src/pov_functions.h` (auto-generated exact POV ports); render-test for
          holes. Many are near-SDF (f_sphere/f_torus have |grad|~1); only the polynomial ones need work.
      (3) *N-D* — **affine remap** of x/y/z for all 78 now (extra dims only reorient via tumble/rotate);
          hand-written **true-N-D** forms for the 9 `POV_ND_GENERALIZABLE` deferred to **P3.4**. Named params
          (from `pov_params`) become `--oscillate`/`--lock` axes inheriting P3.2b rate/phase.
      (4) *Container* — per-function **bbox table** sizes bounded shapes; explicit `--radius` clips unbounded
          ones (paraboloid/cylinders/helices). POV coords are **not** freq-scaled (unit authored scale).
      (5) *Unspecified params* — default to a **random draw within the authored (lo,hi) range** per seed
          (consistent with unnamed dims), governed by `--axis-default` (off=freeze at default, on=swing).
      Build order (small green slices): **(S1) done 2026-07-18** — `--surface` accepts any POV name
      (validated at runtime via `resolve_surface`: `schwarz_p` alias resolved, catalog-only
      `schwarz_d`/`neovius` + unknown names rejected). POV emits as a **solid** (`(f)-(threshold)`, no
      abs-shell) at dims=3 with authored default params, a per-function `max_gradient` from `_POV_GRAD_BOUND`
      (f_sphere/f_torus = 1.0, conservative `_POV_GRAD_DEFAULT` 8.0 otherwise), wired through `build_scene`
      via a shared `_assemble_iso_scene` helper. New Variant field `pov_values`; early POV branch in
      `field_expr` (all transforms are no-ops on a POV field for now); 13 new tests (378 loom green);
      smoke-rendered f_sphere + f_torus as clean solids. **S1 follow-up done 2026-07-18** — solid
      orientation + natural isolevel: most clamped builtins are `r = -(poly)` (positive-inside), so the
      naive `{f<0}` rendered their *exterior* (heart came out as a sphere with heart craters). Added
      `_POV_SOLID_META = {name:(sign,level)}`; emit `sign·(f − (level+threshold))` — positive-inside funcs
      negated (sign flip leaves `|∇f|`/`max_gradient` unchanged), non-zero-level funcs (f_ellipsoid, surface
      at level 1) shifted first; un-tabulated funcs fall back to honest `(+1,0)`. Validated: f_heart renders a
      solid valentine, f_ellipsoid a solid unit sphere (both were broken). 6 more tests (384 loom green).
      **(S2) done 2026-07-18 (Option B)** — tight active-band gradient bound in new `loom/pov_grad.py`,
      wired into `build_scene` via `_pov_grad_bound(name, values, box)`. A **correctness** fix, not just
      speed: POV algebraic builtins are `clamp(P0·r, ±10)` and a high-degree `r` has a huge gradient far
      from the surface — f_hunt_surface's true `|∇f|` ≈ 11000, so the old `8.0` default was a catastrophic
      under-estimate (marcher oversteps → holes). Bound `|∇f|` only over the un-railed **active band**
      (`|P0·r|<10`): rigorous (crossing ±10→0 takes ≥ `10/bound` of travel, so sphere-tracing never
      oversteps) and tight (skips the railed tails). Impl: vectorised adaptive interval branch-and-bound
      (numpy `_IV` intervals on the *factored* derivatives w/ exact even-power handling; certify a lower
      bound from band sample points, discard sub-boxes provably below it, octasect survivors, stop within
      tol; ×1.02 safety). Closed forms for SDF-like primitives (sphere/torus→1, ellipsoid→max|semi-axis|);
      returns None for noise/atan2/rotation → caller keeps default. Cross-checked vs dense numeric sample:
      rigorous + ≈1.05× tight; render-validated f_hunt_surface hole-free. 13 more tests (397 loom green).
      **(S3) done 2026-07-18** — POV container auto-sizing + `--shell`. Rather than a hand-authored 78-entry
      bbox table, `loom.pov_grad.surface_bbox(name, params, level)` *derives* each surface's natural extent
      by grid-sampling the transcribed field `f(x,y,z)` (new `_FIELD_BUILDERS` for the SDF/norm builtins
      f_sphere/f_torus/f_ellipsoid; the algebraic builtins reuse `P0·r`) and finding where it crosses the
      isolevel; returns `(half_extent, bounded)` (bounded=False when the surface runs to the search
      boundary — an unbounded paraboloid/cylinder/helix). `build_scene` now defaults `--radius` to None and
      calls `_pov_container_radius(name, values, level, radius_arg)`: explicit `--radius` wins (and *clips*
      unbounded shapes), else auto-size to the padded bbox (×1.08), else the 1.3 default (unbounded / no
      transcribed field). Fixes f_hunt_surface (surface at r≈3.67 — was a clipped disk at 1.3, now clip
      radius 3.96 / box 4.16) and f_ellipsoid's long lobes. `--shell` carves any POV shape hollow
      (`abs(sheet) − thickness`); a tagged thin set (`_POV_THIN_SURFACES`: klein_bottle/boy_surface/enneper/
      cross_cap/… + any `*_2d` curve) shells by default. TPMS keep their own abs-shell and the 1.3 default,
      untouched by `--shell`/None-radius. 20 new tests (417 loom green); render-validated f_hunt_surface
      shows its full surface (not a clipped disk).
      **(S4) done 2026-07-18** — POV shape params pinnable via `--lock NAME=VALUE`. Each POV surface has
      named shape params (`pov_params(name)` → `(axis, desc, default, (lo,hi))`, e.g. f_torus: `major`/
      `minor`; f_ellipsoid: `rx`/`ry`/`rz`); `--lock major=1.6` overrides that param's default. Rides on
      the existing `--lock` flag but stays unambiguous: the motion grammar never uses `=`, so any
      `NAME=VALUE` token is a param pin and everything else (commas, `tumble`, `spin`, …) keeps its
      motion meaning — `resolve_pov_param_locks(args)` splits the two, pins go to `args.pov_param_locks`,
      the rest stays on `args.lock` (collapsing to None if only pins were given). Space-separates multiple
      pins (`--lock "rx=2 rz=0.5"`); values are full `_osc_eval_num` expressions. Validation: pin on a
      non-POV surface, or an unknown param name, errors (SystemExit, lists valid names); out-of-range
      value warns but is honored. `pick_variant` applies pins onto `pov_default_values` before emit, so a
      pinned semi-axis both flows into the emitted `f_*` call *and* resizes the S3 auto-sized container.
      13 new tests (430 loom green).
      Still: (S5) params as `--oscillate` swinger axes; (S6) affine remap for dims>3; (S7) `--axis-default`
      random-draw for unspecified params. **Note for S5:** the S2 bound is per (name, params, box) and
      cached — animating a param means recomputing it per frame (or bounding once over the param range).
      **Note for S4/S5:** container auto-size is also per (name, params) and cached — a param that moves the
      surface's extent (e.g. ellipsoid semi-axes) needs the same per-frame / range recompute as the bound.
- [ ] **P3.4** True-N-D forms for the 9 `POV_ND_GENERALIZABLE` funcs (hand-written symmetric N-D FTSL,
      bypassing the 3-coord `f_*` builtins; must match the `f_*` call at N=3). Makes the nd_pov/affine_pov
      split real. After P3.3.
- [ ] **P3.5** Ordered / overlapping N-D tumble (design captured 2026-07-18; do *after* P3.3, it's
      orthogonal to the surface library). Today's `tumble` is confined to a set of **disjoint** Givens
      planes (`pick_variant` lines ~1328-1353) — i.e. a **maximal torus of SO(N)**, a commuting abelian
      subgroup where rotation order is a no-op *by construction*. Generalize to an **ordered word** of
      possibly-overlapping planes, where list order = composition order and non-commutativity yields
      genuinely richer reorientation paths the disjoint set can't reach. Key facts that make this cheap:
      (1) the evaluator `_tumbled_directions` **already composes planes sequentially in list order** — the
      restriction lives *only* in the construction, not the eval; (2) **seamlessness survives ordering** —
      each whole-turn factor returns to identity at t=1, so the product is identity at t=1 regardless of
      order/overlap; (3) the **only real cost is the Lipschitz bound**: disjoint planes cap `|rotated dir|
      <= sqrt(2)` (the current `coef *= sqrt(2)` shortcut, line ~2052), but overlapping planes can grow a
      row toward `sqrt(#coupled rows)`, so the general path must compute the true worst-case row norm.
      **Design decision (agreed 2026-07-18, revised): "supersede", not "alongside"** — the user chose the
      more elegant single path and explicitly **waived byte-identity of existing tumble renders** ("we'll
      just re-render them"). Scope of the waiver: *only* renders that use `tumble` — non-tumble seeds never
      build tumble planes, so they stay untouched. So replace the disjoint construction with ONE general
      path: tumble is an **ordered word** of `(axis_i, axis_j, winding)` Givens planes (list order =
      composition order; planes may overlap), evaluated by the existing sequential `_tumbled_directions`
      (already order-honoring), with a **single general bound** = the true worst-case visible-row norm of
      the composed rotation. That bound *auto-returns* `sqrt(2)` for a disjoint word and grows only for
      overlapping ones, so there is **no special-case code and no speed loss on the disjoint case** — the
      `coef *= sqrt(2)` shortcut is subsumed, not duplicated. No legacy/opt-in branch. **Open sub-question:**
      the seed-driven **default word** when the user gives no explicit one — keep it the current disjoint
      pairing (each visible axis <-> one hidden dim; clean, predictable, tight bound) re-expressed in the
      general framework, or make the default itself a richer overlapping draw. **Decided (2026-07-18): default
      stays disjoint-clean; the richer interacting motion is opt-in via an explicit ordered word.** Good UX
      + tight bound + guaranteed the slice tips out of the 3-space; the full-group richness is one explicit
      word away. Deliverables: (a) grammar for the ordered plane word via a new **`--tumble-sequence`** flag
      (provisional name; fits the `--tumble-mode`/`--tumble-lock`/`--tumble-amp` family) — an ordered,
      comma-separated list of axis pairs like `0-3,3-4,0-4` (each optionally carrying a turn count, e.g.
      `0-3x2`); **list order is significant** and pairs may overlap (that's what unlocks order-dependent
      motion). Plain `--oscillate tumble` with no `--tumble-sequence` keeps the tidy automatic default; (b)
      the single general
      construction; (c) the general row-norm bound (>= sqrt(2), only ever safer; `max_gradient` affects
      march step / hole-safety, never the converged image, so this is safe); (d) tests: an overlapping word
      produces motion a disjoint set can't, seamless-loop preservation (product = I at t=1 regardless of
      order), bound-never-under-estimates, and the default word still meaningfully reorients the slice.

### §8 — GPU isosurface rendering (kill per-frame tessellation; independent track)
- [x] **G1** `--raster-iso <n>` passthrough in `gyroid_nd._render_frame` → ftrace's existing
      `-raster-iso` (grid res, default 96). Zero engine changes; cuts CPU tessellation cost today.
      *Done 2026-07-18:* `--raster-iso N` CLI flag → `make_video` → `_render_frame` appends
      `-raster-iso N` on the raster path. Verified end-to-end (2-frame render at res 40 → coarse
      gyroid) and all 268 loom tests green.
- [ ] **G2** GPU deterministic primary-ray isosurface **preview kernel** in `raster_cuda` (sibling to
      `renderFrame`): per-pixel cast primary ray → existing `intersectImplicit` + `dFieldGradient` +
      shading, **no tessellation**. Wire a mode (e.g. `-raster-gpu`); route `gyroid_nd` video frames
      through it.
- [ ] **G3** `PatOp::MatMulAdd` intrinsic (matrix·vec + offset) so per-frame affine transforms bake
      cleanly instead of a dozen scalar mul/adds. Five standard PatOp touch-points.
- [ ] **G4 (deferred, export-only)** GPU marching cubes — *only* to accelerate mesh export, not the
      video path. Build only if mesh-export throughput becomes a pain point.

---

## C. Renderer roadmap follow-ups  *(origin: `ROADMAP.md` — main items DONE; these remain)*

- [ ] **C1 Mode M true final gather.** Mode M does a *direct density query* at the first diffuse hit,
      not a secondary-hemisphere final gather. A real final-gather pass is the future enhancement
      (noted in `known-issues.md`).
- [ ] **C2 VDB: native sparse device sampler.** Today the NanoVDB grid is baked to a **dense** float
      lattice for the device sampler; a native sparse GPU sampler is the follow-up.
- [ ] **C3 VDB: fp16 + emission/temperature grids** (fire) — currently float density grids only.
- [ ] **C4 VDB: native `.vdb` front-end** — validated against a downloaded official OpenVDB sample
      (only `.nvdb` is ingested today; `.vdb→.nvdb` is a manual step).
- [ ] **C5 Mesh: emissive triangles** (mesh area lights).
- [ ] **C6 Mesh: tangent-space normal maps.**
- [ ] **C7 Mesh: watertight ray–triangle test** to kill grazing-edge cracks.
- [ ] **C8 FBX import via `ufbx`** (decided: vendor MIT single-file `ufbx` into `src/third_party/`,
      like the glTF/JSON headers). Consumes baked geometry/normals/UVs/materials/skinning/animation.
- [ ] **C9 Alembic (`.abc`) import** — heavy SDK (Imath + HDF5/Ogawa); **deferred**, decide if an
      OBJ/glTF/FBX sequence suffices before taking the build weight.

---

## D. Hero-room showcase scene  *(origin: `ROADMAP_heroroom.md`)*

- [~] **D1 Flyby photon-map render** — GPU shared photon-map path (build once, gather all 144 frames),
      `-savemap gallery/hero_map.ftpmap`. (Was in progress.)
- [ ] **D2 Verify still** — raster + a real photon-mapped render frame; confirm all pieces read.
- [ ] **D3 Verify flyby** — render frames + assemble; confirm gyroid thread + glass pass + seamless loop.

---

## Progress log
- 2026-07-18: file created; consolidated undone items from DESIGN.md, OSCILLATE_GRAMMAR.md, ROADMAP.md,
  ROADMAP_heroroom.md, and the just-designed camera-curve bridge (§A). Starting on item G1
  (`--raster-iso` passthrough — the trivial, zero-engine-change win).
- 2026-07-18: **G1 done.** `--raster-iso` flag threaded through `gyroid_nd`; verified end-to-end
  (coarse gyroid at res 40) + 268 loom tests green. Next: P1.1 (the `--oscillate` parser + model).
- 2026-07-18: **P1.1 done.** Standalone `--oscillate`/`--lock` grammar parser + `OscGroup` model +
  safe arithmetic evaluator in `gyroid_nd.py`; 17 parser unit tests; 285 loom tests green. No
  behavior wired yet (that's P1.2 — desugar `--transform` through the group model). Next: P1.2.
- 2026-07-18: **P1.2 done.** `transform_to_oscillate(...)` + `oscillate_spec(...)` desugaring bridge
  maps today's `--transform`/`--bloom`/`--bloom-amp`/`--tumble-*` to one canonical composite
  `OscGroup` (§3 migration map). Pure model — execution path untouched, all existing tests pass
  unchanged. 13 new tests; 298 loom tests green. Next: P1.3 (wire swinger axes freq/threshold/
  thickness/bloom to real behavior — the deterministic, non-RNG-sensitive half).
- 2026-07-18: **P1.3 foundation (partial).** Added per-axis `Variant.bloom_amps` + `_swing_amp()`;
  the three swinger functions read a per-axis amp override, falling back to the shared `bloom_amp`
  (empty dict ⇒ byte-identical to the legacy path). 2 new tests; 300 loom green.
- 2026-07-18: **P1.3 done.** Wired the `--oscillate`/`--lock` flags via an idempotent
  `resolve_oscillate(args)` that maps the group model onto the canonical transform/bloom/tumble
  fields — the inverse of `transform_to_oscillate`, so `pick_variant` gets no new path. `--transform`
  default → None for mutual-exclusion; conflict + "not yet wired" (rate/phase, bare dims) guards.
  Validated three ways: 14 field-expression-equivalence/guard tests, a real `--oscillate bloom,freq`
  render through the full CLI→ftrace pipeline (live preview), and a byte-identical `.ftsl` diff vs
  the legacy `--transform` form (incl. `1.5*freq`). 314 loom green. (Corrected an earlier bad call:
  the live-preview rule never blocked rendering-to-validate — CLAUDE.md reworded to say so.)
  Next: P1.4 (wire winder `rate`/`phase` + bare-dim axes — the RNG-order-sensitive winding piece).
- 2026-07-18: **P1.4 done.** Wired the winder axes. `resolve_oscillate` now emits `osc_dim_windings`
  (bare dim index → exact `round(amp*rate)` winding, forced on), `osc_max_winding` (a motion group's
  `rate` = the ceiling of the varied `1..N` cycle, per the user's option-2 call — consistent with a
  lone dim's exact rate), and `osc_phase` (a constant radians offset on the shared winding clock);
  the picker applies windings after the RNG cycle (no draw consumed) and threads phase into
  `field_expr`/`_tumbled_directions`, all no-ops on the legacy path so existing variants stay
  bit-identical. Single shared clock ⇒ conflicting motion rates/phases and swinger `rate`/`phase`
  are rejected. 12 new tests + a real `--oscillate drift rate 3` video (CLI→ftrace). 324 loom green.
  Next: P1.5 (flip default — `--oscillate` primary, `--transform` deprecation notice + docs).
- 2026-07-18: **P1.5 done.** `--oscillate` is now the single documented motion surface.
  `--transform` + its `--bloom`/`--bloom-amp`/`--tumble-*` satellites are hidden from `--help`
  (`argparse.SUPPRESS`) but stay fully supported; explicitly passing `--transform` prints a plain
  one-line deprecation note (checked before `resolve_oscillate` synthesizes it). Migrated the
  module docstring examples/prose + epilog quickstart to the grammar (`--oscillate bloom,1.5*freq`,
  `--oscillate 0.3*tumble`, `--oscillate tumble --lock 0,1`) and rewrote the test suite's incidental
  `--transform` setup usages to `--oscillate` (a one-shot `scraps/convert_transform_to_oscillate.py`
  did 23; 2 hand edits for the dynamic `tr` loop + `base` list). The two `bloom_amps`
  legacy-representation tests and the deliberate desugaring/equivalence references stay on
  `--transform` by design. Default motion (neither flag) is still `drift`. 324 loom green.
  Phase 1 complete — next: P2.1 (`--couple` cluster command).
- 2026-07-18: **P2.1 done.** `--couple CLUSTER CLUSTER…` — the spatial (field) counterpart of
  `--oscillate`: each cluster is comma-joined dims sharing sin*cos terms; spaces separate disjoint
  clusters (a dim in ≤1). Per-cluster `:full`/`:cyclic` tag over a global `--couple-scheme` default.
  `parse_couple`/`resolve_couple` build `couple_clusters` + a forced-on `couple_axes` set fed into
  `forced_on`/`max_forced_axis` exactly like a `--pair …:on` endpoint (no new RNG draws).
  `coupling_pairs()` refactored around a shared `_scheme_edges(dims, scheme)` helper — cluster path
  emits each cluster's ring/clique edges (over its oscillating members) in CLI order; empty
  `couple_clusters` falls through to the legacy `--coupling`/`--pair` base-graph path bit-identically.
  Kept `--coupling`/`--pair` on their own resolution path (they act over the post-RNG active set,
  `--couple` names explicit dims at parse time — no clean desugar), so `--couple` is mutually
  exclusive with a non-default `--coupling` / any `--pair`. `coupling_desc` summarizes clusters;
  primitive warning lists `--couple`; docstring/epilog/help + OSCILLATE_GRAMMAR.md §6 updated.
  11 new tests, 335 loom green. Next: Phase 3 (P3.1 surface library) or another TODO track.
- 2026-07-18: **P3.1 done.** Per-surface shape-param metadata table in `loom/pov.py` (Python side —
  it feeds the future `--surface-help`; the C header stays arity-only for the VM). `_AUTHORED_PARAMS`
  hand-documents real `(name, description, default, (lo, hi))` tuples for the well-understood /
  N-D-core shapes and the 0-param spherical/noise helpers; `_generic_params(n)` supplies honest
  `p0..p{n-1}` placeholders for every other `POV_FUNCS` entry. `POV_PARAMS` is built by comprehension
  so its per-function count always equals `arity−3`; `pov_params(name)` returns a defensive copy and
  raises on unknown names. Exported from `loom/__init__`. 6 new tests mirror the arity drift-guard
  discipline (set-equality with `POV_FUNCS`, exact `arity−3` count, valid+unique axis names,
  default∈[lo,hi] with lo<hi, authored spot-checks, unknown-name reject, copy-safety). 349 loom green.
  Next: P3.2 (`--list-surfaces` / `--surface-help NAME` discovery commands).
- 2026-07-18: **P3.2 done.** Surface-library discovery commands in `gyroid_nd.py`. A catalog
  (`_TPMS_CATALOG` for the 4 periodic minimal-surface families + `POV_FUNCS` for the 78 POV builtins)
  is grouped by N-D honesty class via `surface_group()` → periodic / nd_pov / affine_pov.
  `--list-surfaces` prints the whole library (82 surfaces) with per-surface shape-param counts and
  `[nd]`/`[loop]` tags; `--surface-help NAME` prints one surface's shape params (name, meaning,
  default, range from `pov_params`) or the shared-axis note for a param-free TPMS, resolving the
  `schwarz_p`→`primitive` alias and raising on unknowns. Both early-exit before any generation and
  emit ASCII-only text (Windows-console-safe). Cross-referenced from `--surface` help + epilog.
  11 new tests, 360 loom green. Next: P3.3 (widen `--surface` to the full library — the design-heavy
  step: map POV builtins into the N-D slice machinery with the N-D + seamless-motion guards).
- 2026-07-18: **P3.2b done.** Generalized the swinger envelope to carry an independent clock,
  making swingers uniform with winders/bloom (the user's insight: there was no good reason for
  freq/threshold/thickness to lack a rate/phase once seamless-looping was demoted from a hard
  requirement). Each swinger's bloom is now `w(t) = 0.5·(1 − cos(2π·rate·t + phase))` via the new
  `_bloom_env_p(v, key, t)`, reading `Variant.bloom_rates`/`bloom_phases` (keyed `"dims"` for the
  dimensional crossfade, own name for the scalar swingers). `resolve_oscillate` records the swinger
  group's `rate`/`phase` instead of rejecting them. Default rate 1 / phase 0 is byte-for-byte the
  legacy fixed `sin²(πt)` envelope (the `--transform` path leaves both tables empty, so all existing
  seeds reproduce exactly). An integer rate loops seamlessly for any phase; a non-integer rate pulses
  faster but breaks the loop, so `main()` prints a one-line "won't loop seamlessly" note. 6 new tests
  (rate stored + peaks at t=¼,¾; default byte-identity; integer-rate seamless; phase flips the bump
  but still loops; `bloom`→`dims` keying; non-integer warning via `main`), 365 loom green.
  Next: P3.3.
