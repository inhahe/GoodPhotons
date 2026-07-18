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
- [ ] **P2.1** Parse clusters → the existing `coupling_pairs()` edge set; per-cluster `cyclic`/`full`
      scheme (global `--couple-scheme` default + optional `:full`/`:cyclic` tag). Retire
      `--coupling`/`--pair` (keep as deprecated aliases). Tests.

### Phase 3 — surface library & per-surface params (§7)
- [ ] **P3.1** Author per-surface param-metadata table `{func:[(name,desc,default,[lo,hi]),…]}`
      (extend `tools/pov_functions_gen.py` + hand fallback) + a test asserting every `POV_FUNCS`
      entry has exactly `arity−3` params.
- [ ] **P3.2** `--list-surfaces` + `--surface-help NAME` discovery commands; main `--help` pointer.
- [ ] **P3.3** Widen `--surface` to the full `iso.py` TPMS (`gyroid`/`schwarz_p`/`schwarz_d`/`neovius`)
      + `pov.py` `POV_FUNCS`, with the N-D (`POV_ND_GENERALIZABLE`) and seamless-motion (periodic-only
      `drift`) guards. Per-surface shape params become `--oscillate`/`--lock` axes.

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
