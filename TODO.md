# TODO — consolidated undone work

Single source of truth for everything planned-but-not-done, pulled together from the
project's scattered plan files. Check items off (`[x]`) as they land, and mirror the
status back into the originating file (`DESIGN.md`, `ROADMAP.md`, `OSCILLATE_GRAMMAR.md`,
`ROADMAP_heroroom.md`) when a whole section closes.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.
Origin tags point at the authoritative design text for each item.

---

## 0. Parametric records — FTSL data structure  *(design locked; full spec in `ROADMAP_records.md`)*

A named record over a scalar domain whose channels are named after real material slots,
sampled by a per-hit driver expression, with nearest/linear/smooth interpolation and
ordered last-write-wins `from` composition. **See [`ROADMAP_records.md`](ROADMAP_records.md)
for the authoritative spec and the 6-stage build plan.**

- [x] **Stage 1** — tokenizer `[` `]` + `NAME = range LO-HI [ … ]` declaration parse & data model. *(committed 0e24f07-precursor)*
- [x] **Stage 2** — channel eval (nearest/linear/smooth + expr stops + spectrum RGB-lerp→Jakob–Hanika) → slots. *(0e24f07)*
- [x] **Stage 3** — driver binding + inline `material NAME(driver)` in geometry. *(b3f42ce)*
- [x] **Stage 4** — `material "m" { from R(d) … slot=expr/channel }` ordered last-write-wins + selectors + record-aware specular reflect. *(989f21f)*
- [ ] **Stage 5 — all-scope value sites** *(in progress; split in `ROADMAP_records.md §4`)*:
  - [ ] **5a** — record refs as *constant* values (`R.chan[i]`, `R(const)`) at any spectrum/scalar value site, + a free-variable scope check that errors on out-of-scope drivers (each site publishes its in-scope driver axes; a load-time constant site publishes ∅).
  - [ ] **5b** *(optional; gated on user go-ahead)* — camera-curve `t`-driver: publish flyby param `t` as an in-scope axis so a record can drive fov/roll/zoom/fstop/focus along a `camera_curve`.
- [ ] **Stage 6** — GPU parity (bake like `ProcTexture`).

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
- [x] **P3.3** (DONE 2026-07-18 — all slices S1–S7 shipped; see the per-slice notes below) Widen
      `--surface` to the full `iso.py` TPMS (`gyroid`/`schwarz_p`/`schwarz_d`/`neovius`)
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
          (consistent with unnamed dims), governed by `--param-default {default,random}` (shipped as S7;
          renamed from the provisional `--axis-default` to avoid the existing axis-polarity flag).
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
      13 new tests (430 loom green). Also fixed an ordering bug found by render-validation: the pin
      extraction ran *after* `resolve_oscillate`, so `--lock minor=0.4` crashed (the motion grammar
      rejects `=`); moved it before, made the `_pv`/`_resolved_args` test helpers faithfully run
      `resolve_oscillate` (which had hidden the bug), +1 regression test (431 green).
      **(S5) done 2026-07-18** — POV shape params are now `--oscillate` swinger axes. With a POV `--surface`
      the grammar's axis set gains that surface's named params: `--surface f_torus --oscillate minor` sweeps
      the tube radius over the loop. Semantics are range-aware so amp is intuitive: `p(t) = clamp(base +
      amp*span*env(t), lo, hi)` with `span = (hi-base)` for amp≥0 else `(base-lo)`, and `env` the shared
      sin²(πt) bump — so `amp=1` reaches the param's authored range *edge* exactly at mid-loop (no plateau,
      seamless return to base), `amp<0` sweeps the other way, `|amp|>1` over-drives and clamps. Recorded in
      a new `Variant.pov_swing = {param: amp}` (kept apart from the gyroid dims-bloom swingers since it drives
      `pov_values` per frame, not the dims cross-fade), sharing the `_bloom_env_p` clock (`bloom_rates`/
      `bloom_phases`). `field_expr`/`build_scene` evaluate params via `_pov_values_at(v, t)`, so the S2
      gradient bound and the S3 auto-sized container **recompute per frame** from the swept values (an
      animated ellipsoid semi-axis grows its container as it lengthens; a torus's SDF bound stays 1). Grammar
      plumbing: `--surface` now resolves *before* `resolve_oscillate` so a param-name axis classifies against
      it; a param-only `--oscillate` names a benign `drift` (POV ignores transform) instead of erroring "no
      motion axes"; a bad axis on a POV surface hints the valid param names. 12 new tests (443 loom green);
      render-validated (torus `minor` sweeps 0.25→2.0→0.25, container 1.44→3.06→1.44).
      **(S7) done 2026-07-18** — `--param-default {default,random}` gives POV batches actual variety. A POV
      surface ignores the randomized dims/freq/harmonics (its shape is the `f_*` call args, not the N-D
      field), so a plain `-n N --surface f_torus` batch was N *identical* images. With `--param-default
      random`, every UNSPECIFIED shape param (not pinned by `--lock NAME=VALUE`, not animated by
      `--oscillate NAME`) is drawn uniformly in its authored `[lo,hi]` per variant seed, so each variant is a
      distinct shape; `default` (the flag's default) keeps the current single authored shape. The draw runs
      *last* in `pick_variant`'s RNG stream (after the hidden-offset / tumble draws) so it never perturbs the
      field's reproducibility, and it's a no-op on a TPMS (no `pov_values`) — a TPMS's shape already varies
      via its randomized freq/threshold. Explicit pins and swingers opt their param out of the draw, so
      `--lock major=1.6 --param-default random` fixes the major radius while the minor still varies. (Named
      `--param-default`, not `--axis-default`, to avoid colliding with the existing `--axis-default`
      on/off/random axis-polarity flag.) 7 new tests (450 loom green); smoke-validated (`-n 3 --surface
      f_torus --param-default random` → 3 distinct `f_torus(...)` calls; `default` → one shared default).
      **(S6) done 2026-07-18** — affine N-D remap: a POV surface's `(x,y,z)` now pass through a per-frame
      affine `M·p + b` before the `f_*` call, the honest realization of "an N-D slice of a 3-D POV field is
      an affine remap of x/y/z" (design confirmed by the user: *full affine* + *allow drift*). Rows 0/1/2 of
      `M` are the three visible slice axes' world directions, composed from the same motion layers as the
      periodic field but read as coordinate axes: **tumble** rotates the whole slice basis in N-D (a visible
      axis mixes with a hidden dim, tilting/foreshortening the shape out of the rendered 3-space and back —
      the marquee dims>3 effect), **rotate** turns each axis edge-on independently (its row scales by
      `cos α`, gaining a `hidden_offset·sin α` translation), **drift** pans each axis by `winding·t` world
      units. tumble/rotate return to identity at t=0,1 (seamless); drift is deliberately *non-seamless* for a
      non-periodic POV shape (the user opted in). New `_pov_affine(v,t,transform)` builds `(M,b)`;
      `_mat3_singular_extremes` (analytic 3×3-symmetric eigenvalues, pure stdlib) gives σ_min/σ_max so the
      render stays rigorous: the emitted field is `f(M·p+b)` whose gradient is `Mᵀ∇f`, so the S2 marcher
      bound is scaled by σ_max and the S3 container grows by `1/σ_min` (σ_min floored at 0.15 so a near-edge-
      on axis can't blow the container up unbounded; an explicit `--radius` clips instead of auto-growing).
      Gated on a new `Variant.pov_motion` (set only by a *real* explicit motion — a named drift/rotate/tumble
      or `--transform`; a pov_swing-only spec's benign filler `drift` and the default both leave it False), so
      a plain `--surface f_torus` stays the static `f(x,y,z)` (exact pre-S6 behavior). 16 new tests (466 loom
      green); render-validated (`--dims 5 --oscillate tumble --surface f_torus`: t=0 face-on torus, t=0.40
      tilted ring, t=0.25 near-edge-on sliver — all hole-free, no clipping, seamless at the loop ends).
      **P3.3 complete.** (Known refinement: a flat shape like the torus foreshortens hard mid-tumble, so the
      auto-container can jump several× within a few frames — pin `--radius` for a steadier camera.)
- [x] **P3.4** True-N-D forms for the 9 `POV_ND_GENERALIZABLE` funcs (hand-written symmetric N-D FTSL,
      bypassing the 3-coord `f_*` builtins; must match the `f_*` call at N=3). Makes the nd_pov/affine_pov
      split real. *Done 2026-07-18:* new module `loom/pov_nd.py` supplies, for each of the nine funcs,
      an honest `D`-coordinate field `F(ξ_0…ξ_{D-1})` (`nd_field_expr` FTSL emission + `nd_field_eval`
      numeric twin) that at `N=3` reduces **bit-for-bit** to the `f_*` builtin (verified to machine
      precision against direct ports of the C bodies in `src/pov_functions.h`), plus `nd_grad_bound_xi`,
      a rigorous conservative bound on `|∇_ξ F|` over the coord box (numerically confirmed to never
      under-estimate; returns `None` for `f_superellipsoid`'s non-Lipschitz corners → caller falls back
      to the per-function default). Integrated into `gyroid_nd.py`: `_pov_nd_embedding(v,t,transform)`
      builds the per-frame `D×3` slice Jacobian `A` (rest = `e_i` for the three visible dims, `0` for
      hidden dims; `c=0`) and folds hidden axes in via the same tumble Givens rotations as the affine
      path (plus rotate cos-scaling / drift pan); the emitted field is `F(A·p+c)`. Rigor mirrors S6 with
      the `D×3` Jacobian: `_matn3_singular_extremes(A)` (shares the new `_sym3_eig_extremes` helper with
      `_mat3_singular_extremes`) gives σ_min/σ_max from the `3×3` Gram `AᵀA`, so `|∇_p F| ≤ σ_max·|∇_ξ F|`
      (marcher bound) and the container grows by `(nat_rad+|c|)/max(0.15,σ_min)`. **Gated** on
      `_pov_use_nd`: `pov_motion ∧ tumble ∧ dims>3 ∧ surface∈POV_ND_GENERALIZABLE` — every other case
      (no motion, drift/rotate-only, `D≤3`, affine_pov) keeps the exact pre-P3.4 S6 path (byte-identical).
      38 new tests (504 loom green); render-validated (`--dims 5 --oscillate tumble --surface f_ellipsoid
      --lock rx=1.8 ry=0.6 rz=1.0`: t=0 face-on ellipsoid, t=0.25 the x-axis folded into a hidden dim —
      hole-free, seamless at the loop ends). **P3.4 complete.**
- [x] **P3.5** Ordered / overlapping N-D tumble via **`--tumble-sequence`** — DONE 2026-07-18.
      Implemented exactly the agreed "supersede, not alongside" single-path design. `--tumble-sequence
      i-j[xN],…` (`_parse_tumble_sequence`) parses an **ordered** word of `(i,j,winding)` Givens planes
      whose list order = composition order and whose pairs may **overlap** (share an axis); it overrides
      `--tumble-lock` and, when absent, plain `--oscillate tumble` keeps the tidy disjoint default.
      pick_variant (~1404) branches to the explicit word when given, else the existing disjoint draw. The
      one rigor change is the periodic-field Lipschitz bound: `coef *= sqrt(2)` → `coef *=
      _tumble_rownorm_factor(v)` = **sqrt(max connected-component size)** of the plane graph (union-find;
      Cauchy–Schwarz — a row draws amplitude only from its component). That **auto-returns sqrt(2) for any
      disjoint word** (each plane its own size-2 component, so the disjoint default's bound is *byte-
      identical* to the old shortcut — the waiver on tumble byte-identity was never even needed for the
      default) and grows only for overlapping words (`0-3,3-4,0-4` → component {0,3,4} → sqrt(3)). The POV
      affine (S6) and N-D (P3.4) paths already compute σ_max from the **exact** per-frame matrix via
      `_tumbled_directions`/direct plane iteration, so they honor overlapping words with **zero** changes.
      Deliverables all met: (a) `--tumble-sequence` flag + parser with full validation (axis range, self-
      pair, turn count); (b) single general construction (no legacy branch); (c) general row-norm bound;
      (d) 11 new tests (parse+validation, component-size bound incl. disjoint=sqrt2 / triangle=sqrt3 /
      chain=2, exact plane wiring, lock-override, seamless+starts-from-base, **overlap is order-dependent /
      disjoint is order-independent**, bound-never-underestimates on the composed rotation, default still
      reorients, and the N-D POV path honoring an overlapping word). **515 loom tests green** (was 504).
      Docs: OSCILLATE_GRAMMAR.md §7.y, `--tumble-sequence` help text. **P3.5 complete.**
  - **P3.5 design notes (historical, for reference — superseded by the DONE entry above):** Ordered / overlapping N-D tumble (design captured 2026-07-18; do *after* P3.3, it's
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
- [x] **G2** GPU deterministic primary-ray isosurface **preview kernel** — per-pixel cast primary ray
      → existing `closestHit` (which sphere-traces implicits via `intersectImplicit`) + `dFieldGradient`
      shading, **no tessellation**. Wired as `-raster-gpu`; `gyroid_nd` frames route through it.
      *Done 2026-07-18:* kernel `kIsoPreview` lives in `render_cuda.cu` (where `closestHit`/`DScene`/
      `buildUpload` already are — a device twin of `raster::renderFrame`'s shading: flat per-material
      albedo, ambient + Σ weighted N·L keys + headlight fill), downloads linear-RGB + depth/emitter
      masks and calls the **shared** host `raster::exposeAndEncode` so output matches `-raster` and
      honours a camera_path's locked auto-exposure anchor. `-raster-gpu` (main.cpp) falls back to the
      CPU rasterizer when the GPU can't handle the config (no CUDA device, `-see-through`/clarity, or a
      physical mesh-lens camera). `gyroid_nd --raster-gpu` swaps the per-frame flag (`--raster-iso` moot
      — no marching cubes). Validated on `scenes/implicit.ftsl` (metaballs + CSG + torus render
      identically to `-raster`, cleaner surfaces) and a 3-frame gyroid video.
- [ ] **G3 (deferred — optimization only, not an enabler)** `PatOp::MatMulAdd` intrinsic (matrix·vec +
      offset). *Decided 2026-07-18: skip for now.* The N-D rotation loom bakes into each isosurface
      **already renders correctly** via existing scalar ops — `_arg_expr()` emits each matrix row as
      `(a)*x+(b)*y+(c)*z`, which ftrace compiles straight to `Const/VarX/Mul/Add` bytecode and evaluates
      directly (including on the GPU: `-raster-gpu` ray-marches D=8 tumble gyroids today). So MatMulAdd
      only *compresses* the encoding (one fused opcode vs ~6 scalar ops per row) — a compactness /
      marginal-speed win, **not** a new capability. Revisit only if per-frame pattern eval becomes a
      real bottleneck (it isn't — sin/cos/PovFn + the sphere-march dominate). See known-issues.md
      "Deferred: `PatOp::MatMulAdd`". Prefer the contained single-output "matrow" form (Option A) if so.
- [ ] **G4 (deferred, export-only)** GPU marching cubes — *only* to accelerate mesh export, not the
      video path. Build only if mesh-export throughput becomes a pain point.

---

## C. Renderer roadmap follow-ups  *(origin: `ROADMAP.md` — main items DONE; these remain)*

- [x] **C1 Mode M true final gather.** DONE 2026-07-14 (`-pmfg <K>` / `g_pmFinalGather`). At the first
      diffuse hit mode M now shoots `K` cosine-weighted hemisphere sub-rays (`photonGatherSub`,
      `photonmap_render.h`), traces one bounce each, and queries the map at *those* points, so the
      density-estimate blur lives one bounce away — the standard Jensen secondary-hemisphere final
      gather. Direct light uses low-variance NEE (`neeLight`); gather rays collect indirect/env only
      (no double-count). `K=0` keeps the original direct query (a pure superset). Validated on the
      diffuse Cornell box: final gather matches mode R in energy (diffuse-mask M/R=1.010) and is
      essentially unbiased at a large gather radius (M/R=0.994 vs. the direct query's 0.929
      corner-darkening). See `known-issues.md` "Mode M optional Jensen final gather". *Remaining
      GPU caveat (separate, lesser item): the shared GPU mode-M path still falls back to CPU when
      `-pmfg` is set — porting the final-gather sub-ray pass to CUDA is future work, tracked in
      known-issues.md.*
- [ ] **C2 VDB: native sparse device sampler.** Today the NanoVDB grid is baked to a **dense** float
      lattice for the device sampler; a native sparse GPU sampler is the follow-up.
- [ ] **C3 VDB: fp16 + emission/temperature grids** (fire) — currently float density grids only.
- [ ] **C4 VDB: native `.vdb` front-end** — validated against a downloaded official OpenVDB sample
      (only `.nvdb` is ingested today; `.vdb→.nvdb` is a manual step).
- [ ] **C5 Mesh: emissive triangles** (mesh area lights).
- [ ] **C6 Mesh: tangent-space normal maps.**
- [x] **C7 Mesh: watertight ray–triangle test** to kill grazing-edge cracks.  **DONE 2026-07-18**.
      Replaced Möller–Trumbore with the Woop/Benthin/Wald/Áfra watertight test (JCGT 2013) on BOTH the
      CPU double path (`src/geometry.h`) and the GPU float path (`src/render_cuda.cu`). Per-ray the test
      picks the dominant axis of the ray direction, permutes the other two (swapping them when the
      dominant component is negative to preserve winding), and precomputes shear constants (`TriShear` /
      `DTriShear`, built by `makeTriShear`); per-triangle it shears the relative vertices into the ray
      frame and forms the three scaled barycentric edge functions U,V,W. A hit needs the edge signs to
      agree (two-sided: all-nonneg OR all-nonpos), with an exact-zero fallback in higher precision so a
      grazing edge lands deterministically on exactly one of the two triangles sharing it — no cracks
      (background leaking through a closed mesh) and no dropped hits. The shear is **hoisted once per ray**
      at every BVH leaf loop (5 host call sites in `scene.h`, 4 device sites in `render_cuda.cu`) so the
      per-triangle cost is only the shear+edge math; an interface-preserving `intersectTri(ray, tri, …)`
      overload that builds the shear inline remains for one-off callers. The barycentric convention
      (U,V,W weight v0,v1,v2 ⇔ old w0,u,v) matches the retired M–T code, so UVs and interpolated shading
      normals are unchanged. **Validated:** `scenes/triplanar.ftsl` (16 384-tri closed torus, the shape
      whose silhouette used to crack at grazing angles) renders a clean continuous silhouette with no
      background leak on BOTH the GPU float path (where M–T's independent per-triangle edge signs cracked
      worst) and the CPU double path, with byte-identical energy (`absorbed=0.7794`).
- [x] **C8 FBX import via `ufbx`**  **DONE 2026-07-18**. Vendored the MIT / public-domain single-file
      `ufbx` (v0.23.0: `src/third_party/ufbx.{h,c}` + `ufbx-LICENSE`) and confined it to one TU
      (`src/fbx_load.cpp`, mirroring `vdbgrid.cpp`/`stb_image_impl.cpp`) behind a lightweight
      `src/fbx.h` declaration so the 220-KB header stays out of every other TU. `loadFbx` walks each
      mesh-instance node, triangulates faces with `ufbx_triangulate_face`, bakes world positions via
      ufbx's `geometry_to_world` (+ inverse-transpose for normals), then applies the mesh block's
      authored affine on top — filling the SAME `Tri` position/normal/UV slots the OBJ/glTF paths use,
      so smooth shading + texturing come free. Load opts normalize to right-handed **Y-up metres** and
      `generate_missing_normals`, so FBX lands in the engine's convention. Wired `.fbx` into `addMesh`
      **and** `addMeshAsset` extension dispatch (CMake gained `LANGUAGES … C` for `ufbx.c`). Validated:
      hand-authored `scenes/cube.fbx` → `loadFbx: … 8 verts, 12 tris`, `scenes/fbxcube.ftsl`
      render-checked. **Scope now:** baked triangle geometry + normals + first UV set. **Not yet
      consumed** (follow-ups, logged in known-issues): FBX materials, skinning/blend-shapes, animation,
      multiple UV sets, per-face materials.
- [ ] **C9 Alembic (`.abc`) import** — heavy SDK (Imath + HDF5/Ogawa); **deferred**, decide if an
      OBJ/glTF/FBX sequence suffices before taking the build weight.

---

## D. Hero-room showcase scene  *(origin: `ROADMAP_heroroom.md`)*

> **BLOCKED on user sign-off.** None of D2/D3 (the expensive verify renders) proceed until the
> user has personally verified — in the interactive rasterizer flyby-definition tool (the
> camera_curve editor) — that they *like the room and the flyby*. The look-dev of the room
> composition and the camera path is a human aesthetic decision, so don't burn photon-map renders
> on a room/flyby that hasn't been approved. Once the user says "I like it", D2/D3 are unblocked.

- [~] **D1 Flyby photon-map render** — GPU shared photon-map path (build once, gather all 144 frames),
      `-savemap gallery/hero_map.ftpmap`. (Was in progress.)
- [ ] **D2 Verify still** — raster + a real photon-mapped render frame; confirm all pieces read.
- [ ] **D3 Verify flyby** — render frames + assemble; confirm gyroid thread + glass pass + seamless loop.

---

## E. Feature ideas captured 2026-07-18  *(user-proposed; design-captured, not yet scheduled)*

### E1 — Procedural (function-defined) skin, UV-space  *(ftrace; small–medium, self-contained)*  **DONE 2026-07-18**
**Implemented (option b — three r/g/b sub-expressions baked as a texture).** A `texture "name"` block
may now give `rgb "r(u,v)" "g(u,v)" "b(u,v)"` (three quoted ftsl pattern expressions of the surface
`u,v`, constant `pi`) in place of `file`. ftrace compiles them with `compilePatternExpr` and bakes them
**once at load** to a `res`×`res` (default 512, 1–8192) **linear** RGB grid via `patternEval` over the
UV grid (matching `sampleRgb`'s `(1-v)` flip; each output clamped to `[0,1]`), then runs `buildReflCoeff`
— so the result flows through the *exact same* texture pipeline as an image skin (UV-wrap, Jakob-Hanika
spectral upsampling, triplanar, GPU, raster; `reflect texture:<name>` binds it unchanged) with **zero
`render_cuda.cu` changes** and no per-hit fit. This chose the bake-to-grid path over per-hit JH fit (far
too slow — 40-iter Gauss-Newton) and on-demand eval (no benefit for bounded UV). Fills the third square
of the skin matrix: image skins × 3-D-space procedural patterns × **UV-space procedurals**. `src/ftsl.h`
`addTexture` branches on the `rgb` statement; `scenes/procskin.ftsl` render-validated (red=u L→R,
green=v bottom→top, four blue `sin(2π4u)` stripes — all orientation checks pass). loom: `ProcTexture` /
`func_skin(name, r, g, b, …)` in `scene.py` (routed into the texture bucket so it emits before its
material), exported from `loom`, 5 new emit tests (550 loom green). Docs: FTSL.md §5.1,
docs/scene-language.md §9.1, README Textures.

**Idea.** Let a skin be defined by a *function* `f(u,v)` evaluated on demand instead of a pre-drawn
image, but applied through the **exact same UV-wrap machinery** an image skin uses — poll `f(u,v)` in
place of `image(u,v)` for each hit's interpolated UV. **Verdict: worth adding.** It's a genuine gap:
ftrace today has (a) UV-mapped *image* skins (`texture "name" { file … }`, sampled at each hit's UV —
`loom.Texture`/`skin`) and (b) *3-D-space* procedural patterns (`FuncPattern`/`SpatialExpr`, evaluated
at the surface point's world/object XYZ via `dPatternEval`). What's missing is the third square of the
matrix — a **UV-space procedural**: an arbitrary ftsl expression whose variables are the surface `u,v`
(and, cheaply, its derivatives / the hit's other channels), bound like a texture. The evaluator already
exists (`dPatternEval` runs arbitrary postfix bytecode on host+device); the work is (1) expose `u,v` as
pattern inputs when a pattern is bound in a *texture* slot, (2) a `texture "name" { expr … }` (or
`pattern uv:<name>`) grammar so a material's `reflect texture:<name>` resolves to the function instead
of a bitmap, (3) loom `FuncSkin`/`skin(expr=…)` emit, (4) tests + a render. Low risk, high reuse. Open
sub-q: also expose bump/normal-from-UV-gradient for free (the derivative is analytic on the bytecode).

### E2 — General N-D curve → scene-variable animation via the rasterizer curve editor  *(loom + ftrace; LARGE, design; extends §A)*
**Idea.** Generalize ftrace's existing interactive **camera_curve editor** (drop control points,
scrub/play, paint local speed, edit-in-place, save a real `camera_curve` block — `main.cpp` ~4473+)
from "edit a camera flyby" into "edit an **N-D curve through a grid/scatterplot** whose curve variables
can drive **any** scene variable," with **loom as the go-between** (`.ftsl` can't express animation, so
the animation binding must live in loom, which emits the per-frame `.ftsl`). **Verdict: worth capturing
as a design item; it's big and overlaps §A — schedule after §A lands.** The locked-in pieces of the
user's design:
- **Two authoring modes, chosen up front:** *pure flyby* vs. *true animation*. For most render modes the
  distinction "costs nothing." In **flyby** mode everything sampled — `curve(which, frame, dim)` or
  `grid/scatter(curve-coords, dim)` — collapses to just **camera position + orientation at time t**. In
  **animation** mode any sampled value can map to **any** scene variable (e.g. in the gyroid_nd
  isosurface example, any isosurface function parameter).
- **One exceedingly-simple binding API** (lives in the loom go-between, not `.ftsl`): *plug any curve
  variable into any scene variable* — camera position/orientation, or a surface param, etc.
- **The API has TWO distinct channels** (don't conflate them):
  - **(a) whole-video config** — the persistent authoring info: number of curve dimensions, the
    dimension↔scene-parameter *associations* (which sampled channel drives which variable), and the
    starting control points. This is authored **once for the entire animation**.
  - **(b) per-frame live values** — while the user scrubs/plays in the editor, the **rasterizer must be
    able to push the go-between the *current sampled curve values* at the scrub position** so it can
    generate/preview *that one frame*. This is a transient per-frame data flow, **separate from** (a):
    (a) decides *what maps to what* for the whole video, (b) supplies *the numbers right now* for one
    frame. The API must expose both.
- **The scene informs the editor**, through that same API, of: the curve's dimensionality, how many
  curves are tacked onto it, and the full array of **starting control points** to seed the editor with.
- **Scene proposes, editor disposes.** The scene sets the *initial* dimension count and the initial
  dimension↔scene-parameter associations, but the **editor may change them** — doing so just edits the
  original info stored in the animation definition (the persisted (a) config). So the associations aren't
  a one-way scene→editor push; they round-trip.
- **Modulable curve points are OUT for the editor.** The user resolved this: the rasterizer *already*
  owns the time dimension via curve points, so passing loom-modulable (time-varying) control points would
  introduce a *second* time axis — incoherent. So the editor receives a **static starting array** of
  control points; modulation of the points, if any, stays a loom-side concern that is *not* round-tripped
  through the editor.
- **Likely simplification (open q the user leaned toward "yes"):** there may be **no real distinction**
  between higher-D aspects of the curve itself (a 4-D curve) and extra dimensions "tacked on" (e.g. camera
  density), because the editor ignores every spatial dimension past the first three anyway — so the API
  and editing UX can treat them uniformly (one flat list of per-point dimensions).
- **Relation to §A:** §A already covers "loom emits a real `camera_curve` + ftrace orientation axes." E2
  is the strict generalization — same editor, same emit path, but the curve's sampled channels fan out to
  arbitrary scene variables, not only camera pose. Build §A first (it nails the camera/orientation case
  and the emit grammar), then E2 widens the binding target set and the editor's scene-driven seeding.
- **OPEN Q1 — where does the config (a) live: a loom in-memory data structure, or a separate animation
  definition file?** *Leaning: BOTH, at different layers — they aren't alternatives.* The **authoritative
  in-memory model is a loom data structure** (an `Animation`/`CurveDrive` object holding the dimension
  count, the channel→param bindings, and the control points). But because the editor is a **separate ftrace
  C++ process**, the config also needs a **serialized form** the editor can read to seed itself and write
  back when the user edits associations/dimensions/points — i.e. a small persisted **animation-definition
  sidecar** (JSON or an ftsl-adjacent block). loom owns the struct; the sidecar is its on-disk projection
  for the round-trip with the editor. (Note this sidecar is exactly "the animation info" that (a)-edits
  mutate, and it is *not* the `.ftsl` — the `.ftsl` stays per-frame and animation-free.)
- **OPEN Q2 — is the go-between loom, or a separate program?** *Leaning: loom.* loom is already the Python
  program that models a scene and emits per-frame `.ftsl`; it already has the curve system (`TrackedCurve`/
  `LoopCurve`/`Grid`/`Scatter`) and the scene-variable graph. A separate go-between would duplicate all of
  that. So the go-between = loom, exposing the two-channel API above (config in/out + per-frame live-value
  in → `.ftsl` out).
- **OPEN Q3 — transport for the two channels (editor C++ ↔ loom Python).** *Analysis (2026-07-18):* the two
  channels have different needs, so pick per channel:
  - **Config channel (a):** written rarely (once per edit), not latency-sensitive → the **serialized
    sidecar file** from Q1 is fine (use atomic write/rename to avoid half-read races).
  - **Live-value channel (b):** per-frame during scrub → **latency-sensitive, so NOT file-poll** (polling
    lag + disk I/O + half-written-read races). Ranking:
    1. **Anonymous stdin/stdout pipe (preferred to start).** *There is already working precedent:*
       `loom.PreviewServer` spawns a resident `ftrace -serve` child and streams it one `.ftsl` path per
       frame over stdin, reading status over stdout (`preview.py` `_build_cmd`/`show`). Anonymous stdio
       pipes are very cross-platform (subprocess stdin/stdout is identical on Windows/Linux/macOS) and
       **not** fragile in the parent-child model (coupled lifetime = child dies with parent, no ports, no
       firewall). Caveat: E2's live flow is *editor→loom* (push curve values) then *loom→ftrace* (`.ftsl`),
       i.e. more bidirectional than PreviewServer's one-way drive — doable over two pipes, slightly more
       plumbing. **Extend this channel first.**
    2. **TCP-loopback socket (`127.0.0.1`)** — reach for this *only if* E2's UX needs **decoupled,
       restartable** processes (editor restarts without killing loom) or a cleaner bidirectional protocol.
       Most portable socket option (identical Berkeley/Winsock API everywhere), decoupled lifetimes,
       reconnection; costs bind/listen/accept + port mgmt + occasional Windows firewall prompt; sub-ms
       loopback latency is negligible here.
    3. **Named pipe / Unix-domain socket — AVOID.** This is where the real cross-platform pain lives
       (`mkfifo` vs `\\.\pipe\…`; `AF_UNIX` patchy on Windows). No advantage over 1/2 for this use.
  - **Net:** live values over the existing **stdio-pipe** path (upgrade to TCP-loopback only if
    decoupled/restartable processes are wanted); config over the **sidecar file**. Decide the final wire
    format when E2 is scheduled.

### E3 — loom procedural audio: one buffer back-end, per-tick as a thin front-end  *(loom; medium; **DONE 2026-07-18**)*
**Idea / decision.** loom should be able to *generate audio files* procedurally. Two candidate output
models — (1) emit one sample value per time tick, vs. (2) random-access a sample array (`buf[t] += v`,
`=`, `*=`, …) and serialize at the end. **Decision (from `loomsound.txt`): build ONE back-end — the
random-access sample buffer as the single source of truth — and make "one sample per tick" a thin cursor
wrapper on top (`emit(v)` ≡ `buf[cursor++] += v`), NOT a second parallel pipeline.** Rationale: the
buffer model strictly subsumes streaming (it enables mixing multiple voices, overlap-add, reverb/delay
tails past a note's end, range fades, whole-file normalize-before-write, revision) — additive/subtractive
synthesis *is* the buffer model; per-tick streaming is just the buffer with a monotone write cursor and
no look-back/ahead. Two separate systems would duplicate dithering/clip/normalize/interleave/format-write
(divergent-code-path tech debt). Concrete shape: **core** = a per-channel float sample buffer (read/write/
accumulate at any index); **producers** write however they like (per-tick cursor *or* scatter-write
ranges); a single **`finalize()`** does gain/normalize/dither/clip → format-encode → write. "Per-tick" and
"whole-file" become two front-ends over one back-end. **The one genuine fork** that would force a separate
path is *real-time / unbounded* output (live to speakers, or an effectively-infinite stream you can't hold
in RAM) — then you must flush fixed-size blocks and can't revise the past; even then, share everything
below "how samples are produced" (mixer, format, dither, clip/normalize, writer). **DECIDED
2026-07-18 — OFFLINE ONLY: build just the buffer model** (no real-time/streaming path), for three
reasons the user gave: (1) loom is meant to generate **static products**, not do anything in real time;
(2) **Python is too slow** to synthesize audio in real time anyway; and (3) real-time **wouldn't even
work here** — the buffer model's whole point is that producers edit arbitrary past/future indices (mix,
overlap-add, tails, normalize), which is fundamentally incompatible with a commit-as-you-go stream. So:
one per-channel float **sample buffer** as the single source of truth; `emit_next(v)` is a thin
`buf[cursor++] += v` cursor helper for sequential generators; a single `finalize()` (gain/normalize/
dither/clip → encode → write). No second pipeline, no streaming fork. *Note: loom has no audio
subsystem today, so this is a new capability, not a refactor.*

**DONE 2026-07-18.** Implemented as `loom/audio.py` → `SampleBuffer` (exported from `loom`). One
per-channel `array('d')` back-end is the single source of truth. Random-access ops (`add`/`set`/`mul`/
`get`, out-of-range silently ignored), range ops (`add_range` overlap-add, `mul_range`, `fade` linear
ramp, `mix` another buffer with channel routing + offset). Per-tick front-end is the thin cursor
wrapper promised (`emit_next(v)` ≡ `buf[cursor] += v; cursor += 1`, plus `seek`/`tell`). Producers:
`render_fn(fn(i, t_seconds))` and `render_signal(loom Signal)` (audio-rate sampling via
`Clock.at_frame`, seamless-loop aware), each with add/set/mul modes + gain + start/count windows.
Analysis: `peak`/`rms`/`channel`. Single `finalize(path)` = gain → normalize → dither (TPDF, seeded,
default-on for 16-bit) → clip → PCM-encode → WAV (16/24-bit, stdlib `wave`). 30 tests in
`tests/test_audio.py` (round-trip WAV verify for 16/24-bit mono+stereo, dither determinism,
normalize, cursor≡add equivalence, seamless-loop signal render); 545 loom green. Smoke-validated a
real 1 s 220+660 Hz WAV.

### E4 — loom volume transforms: read and write as independent capabilities  *(loom; medium; design-captured 2026-07-18)*
**Idea / decision (user changed their mind 2026-07-18).** loom should be able to **transform volumes** —
both **sparse** (NanoVDB-style / scatter) and **dense** (regular lattice) grids. Originally the user was
wary of loom being able to *output a volume on its own* (i.e. author a grid from nothing and serialize
it), preferring only the coupled form "use an existing volume as a **basis**, transform it, then emit the
result." **Reversed:** forcing that coupling — requiring every volume *write* to be fed by a volume *read*
— is actually **more** machinery than leaving them orthogonal, so the two stay **independent, freely
composable capabilities**:
- **Read** a volume (sparse or dense) as an input field — sample it, feed it into the signal/field DAG,
  use it as a basis for a transform, drive geometry/materials from it, etc.
- **Write / output** a volume (sparse or dense) — serialize a field to a grid on disk — **without
  requiring** that field to have originated from a volume read. The source can be anything the DAG can
  produce (an isosurface function, a procedural field, an expression, a transformed read of *another*
  volume, …).
- Because reading and writing are decoupled, all four combinations are valid: read-only (sample a volume
  into the scene), write-only (bake a procedural/function field to a grid), read→transform→write
  (the "basis" workflow that motivated this), and neither.

**Transforms in scope:** the same field-domain operations the "keep everything as functions; discretize
last" principle already implies (see `loom.txt` claude-analysis) — N-D rotate-and-slice of the domain,
warps/remaps, per-voxel value ops, resampling between sparse↔dense, and modulation by other DAG signals.
Sparse and dense are two storage backings of the *same* logical field type, so a transform is authored
once against the field abstraction and the read/write ends pick the backing (a dense read can emit sparse
and vice-versa). **Open q (defer to scheduling):** on-disk formats for the write end (`.nvdb` to match
ftrace's ingest; dense raw/`.vdb`?), and whether sparse-write goes through an OpenVDB/NanoVDB dependency
or a loom-native sparse encoder.

### E5 — Axis-typed signals: one influence model (broadcast / pointwise / reduce) + mod·pin + sample·select grammar  *(loom; LARGE, design; unifies E2/E4 and records-5a)*
**Idea / decision (design-captured 2026-07-18, from a design bounce).** The whole "what can modulate what,
and does t-influencing-t break?" question collapses into **one** model: every value-producing node in the
loom signal DAG is **a function of a named set of axes** (its free variables) — e.g. a purely spatial
curve depends on `{s}` (arclength/param), a time-curve on `{t}`, an animated spatial curve on `{s,t}`, a
surface field on `{u,v}`, an N-D grid on `{a,b,c,…}`. "A influences B" = **evaluate A at the point where B
is being evaluated**, and the axis sets alone decide how:

- **Broadcast** on axes A lacks: A:`{t}` driving B:`{s,t}` contributes `A(t)`, identical for every `s`
  (⇒ "a time-curve shifts the whole elevation of a spatial curve over time"). Free, pure.
- **Pointwise** on axes A and B share: two things both depending on `t` combine at the *same* t. This is
  the "lockstep" constraint — but it is **not a rule to detect/enforce**; a function-of-t simply *cannot*
  see any t but the current one, so the illegal "run over the whole of B across time" op is
  **inexpressible**, not caught-after-the-fact. **⇒ Do NOT build a t-influences-t detector, and do NOT
  split signals into separate spatial-vs-temporal data types** (that duplicates every op, can't type the
  mixed `{s,t}` / `{u,v,t}` cases, and forbids the legal broadcast). The single axis-set-typed signal
  (the `Animatable<T>` DAG, refined so each node carries *which axes it depends on*) subsumes all of them;
  it's the tensor/shader-broadcast / Houdini-CHOPs model.

**The real (and only) expensive line — pointwise-at-P vs. cross-index-along-an-axis.** Output at eval
point P is **free/pure/streaming** iff it depends only on inputs *at P* (same `s`, same `t`). This
includes `t` (or a t-varying value) appearing inside *each point's own formula* — e.g.
`B.y(s) = f(s, some_curve(t))` reshapes the *whole* of B over time yet is still evaluated pointwise in `s`
and emits exactly **one whole spatial `.ftsl` per tick**; nothing is materialized (you pass a *scalar at
the current t*, not "the whole curve"). It also includes a spatial rotation `R(t)·p` (mixes x/y/z but at
fixed t, independent per point). The **only** cases that need materialization / caching are genuine
**cross-index** ops, where output-at-P reads inputs at *other* points along an axis:
- **Reduce over `s`** — arc length, centroid, an integral, "all of B's points at once as a set." Needs B
  materialized over all `s`. Must be an **explicit reduction node** (never smuggled in implicitly).
- **A transform mixing a spatial axis *with* `t`** — output frame t then reads input across a *range* of
  t′ ⇒ time-caching / two passes. **This is exactly the existing 4-D space-time "video node"** (`loom.txt`
  ~line 61). The test that separates it from the free case is one question: *does output-t read any t but
  the current one?* No ⇒ free (t-in-each-formula). Yes ⇒ it's the video node, pay the caching cost knowingly.

**Two orthogonal edge attributes.** A DAG edge carries `(combine-mode) × (broadcast, implied by axis sets)`:
- **combine-mode = `pin` | `mod`** — `pin` replaces (last-write-wins); `mod` accumulates toward the
  **target's identity element**, which depends on the target's quantity type: neutral **0** + `y += gain·x`
  for additive/unbounded quantities (position, elevation), neutral **1** + `y *= x` for gains/scales,
  **½-centered** `y = clamp((y−½)+gain·(x−½), 0,1)` for bipolar-[0,1] quantities. So "mod" is *one mode*
  at the authoring surface but resolves to the domain-correct operator; the edge carries `mode` + a
  **gain**, and the **target** declares its neutral/normalization (don't hardcode the ½/[0,1] assumption).
- Broadcast/pointwise is *not* an author choice — it falls out of the axis sets (above). mode and axes
  compose without interacting: axes decide *where* combining happens, mode decides *how* it combines there.

**One sample/select grammar everywhere (records, curves, grids, scatters).** A serial structure is
**sampled** with `(...)` (continuous, interpolated) and **indexed** with `[...]` (discrete constant
selector); `.name` picks a named component/channel. This is the *same* grammar records already set
(`R(driver)` sample, `R.chan[i]` stop-select, `R.chan` channel):
```
some_curve(t)          # sample the curve at parameter t (interpolated between control points)
some_curve(t).y        # …take its y component
some_curve.y(t)        # component-first spelling of the same
some_curve.dim[3](t)   # dim 3 as a discrete channel pick, then sampled at t
```
Deliberately **avoid `some_curve[t]`** for the temporal index — brackets already mean "pick a fixed
discrete stop" in records, so `[t]` would overload them; `(t)` reads as "sample here, interpolate," which
is the intended semantics. Because `some_curve(t)` yields a scalar/fixed-vector *at the current t*, it
broadcasts across the target's other axes ⇒ lands on the free side by construction.

**The unifying one-liner (shared with records-5a's free-variable scope check).** *Everything that produces
a value declares the axes it depends on. Composition broadcasts on unshared axes and combines (pin/mod)
pointwise on shared ones. Crossing an axis you don't own requires an explicit reduction (over `s`) or is
the cached space-time video node (over `t`).* Records-5a is the same mechanism seen at a value site: a
driver's free variables must be ⊆ the axes in scope there (`R(u)` errors in a light SPD because `u` isn't
in that site's axis set). **Open q (defer to scheduling):** the concrete `Animatable<T>` node taxonomy and
how axis-set inference/annotation is represented in the loom struct + the on-disk projection; where the
explicit reduction node and the video node sit in that taxonomy.

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
- 2026-07-18: **P3.4 + P3.5 done** (see §B entries) — true N-D forms for the 9 generalizable POV solids,
  then ordered/overlapping tumble via `--tumble-sequence`. 515 loom green; both render-validated.
- 2026-07-18: **Housekeeping.** Verified **C1 (mode-M final gather) was already done** (2026-07-14,
  `-pmfg`) and marked it off (GPU-port of the sub-ray pass remains a lesser follow-up). Added a
  **BLOCKED-on-user-sign-off** gate to §D (no D2/D3 verify renders until the user approves the room +
  flyby in the rasterizer camera_curve editor). Captured three user-proposed features as §E: E1
  UV-space procedural skin (ftrace, small), E2 general N-D-curve→scene-variable animation via the
  rasterizer curve editor (loom+ftrace, large, extends §A), E3 loom procedural audio (one buffer
  back-end, per-tick as a thin front-end — decided).
- 2026-07-18: **E3 done.** New `loom/audio.py` → `SampleBuffer`: one per-channel `array('d')`
  back-end as the single source of truth; random-access `add`/`set`/`mul`/`get` (out-of-range
  ignored), range ops (`add_range` overlap-add, `mul_range`, `fade`, `mix` with channel routing),
  the thin per-tick cursor (`emit_next` ≡ `buf[cursor]+=v; cursor+=1`, `seek`/`tell`), producers
  `render_fn(fn(i,t_sec))` + `render_signal(Signal)` (audio-rate, seamless-loop aware, add/set/mul
  modes), `peak`/`rms`, and one `finalize()` (gain→normalize→dither→clip→PCM→WAV, 16/24-bit via
  stdlib `wave`). Exported from `loom`. 30 new tests (WAV round-trips, dither determinism, cursor≡add,
  seamless-loop render); 545 loom green; real 220+660 Hz WAV smoke-validated. Next: E1 (UV-space
  procedural skin).
- 2026-07-18: **E1 done** (see §E1) — UV-space procedural color skin, option b (three r/g/b
  sub-expressions baked to a linear RGB grid at load, then run through the whole existing texture
  pipeline; zero GPU changes). `src/ftsl.h` `addTexture` `rgb`-branch + `compilePatternExpr`/`patternEval`
  bake; loom `ProcTexture`/`func_skin`; `scenes/procskin.ftsl` render-validated (all orientation checks
  pass). 5 new loom tests, 550 loom green. Next: G3 (PatOp::MatMulAdd matrix intrinsic).
- 2026-07-18: **C8 done.** FBX mesh import via vendored ufbx (single-file, confined to `fbx_load.cpp`);
  `mesh { file "*.fbx" }` triangulates + bakes world positions; no unit conversion (raw cm coords, size
  via `scale`). `scenes/fbxcube.ftsl` validated in raster + forward mode B. Geometry-only (no FBX
  materials/skins/anim) — logged in known-issues. Committed 3d6dd65.
- 2026-07-18: **G2 done.** `-raster-gpu`: GPU deterministic primary-ray isosurface preview (`kIsoPreview`
  in `render_cuda.cu`, reusing `closestHit`/`buildUpload` + shared `raster::exposeAndEncode`); no
  tessellation. main.cpp falls back to CPU raster on unsupported configs; `gyroid_nd --raster-gpu` routes
  video frames through it. Fixed a vertical-flip bug (dGenRay py=0 is image bottom, accum row 0 is top).
  Validated on `scenes/implicit.ftsl` + a gyroid video. Next: G3 (PatOp::MatMulAdd — needs a design call).
