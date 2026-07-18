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

1. **Parser + model, no behavior change. ✅ DONE 2026-07-18.** Added an `--oscillate`/`--lock`
   grammar parser producing a list of `OscGroup{items:[(amp,axis)], rate, phase}`
   (`parse_oscillate`/`parse_lock_axes` in `examples/gyroid_nd.py`), with a safe arithmetic
   evaluator (`pi`/`tau`/`e`, `+ - * / ** %`, parens) for amplitudes and rate/phase. Pure
   parser — not yet wired to any behavior. Unit-tested in isolation (17 tests in
   `tests/test_gyroid_nd.py`: grouping, amplitudes incl. `2*pi*x`, rate/phase either order,
   reserved-word/duplicate/empty/bad-expr errors, `--lock` flatten+dedup). 285 loom tests green.
2. **Desugar `--transform` → groups. ✅ DONE 2026-07-18.** Added
   `transform_to_oscillate(transform, *, bloom_params, bloom_amp, tumble_mode,
   tumble_amp) -> List[OscGroup]` (and the inverse `oscillate_spec(groups) -> str`)
   in `examples/gyroid_nd.py`: the behavior-preserving bridge that re-expresses
   today's `--transform` + `--bloom`/`--bloom-amp`/`--tumble-mode`/`--tumble-amp`
   flags as one canonical composite `OscGroup` per §3's migration map (winders amp 1,
   tumble-slide → `tumble_amp*tumble`, `dims`→`bloom`, scalar swingers at amp
   `bloom_amp`, scalars only under an active bloom). Pure model — the execution path is
   untouched, so all existing tests pass unchanged. 13 new tests (298 loom tests green).
3. **Wire swinger axes** (`freq`/`threshold`/`thickness`/`bloom`) so `amp` = amplitude,
   replacing `--bloom`/`--bloom-amp`. ✅ **DONE 2026-07-18.** Added per-axis
   `Variant.bloom_amps` + `_swing_amp()` (each swinger uses its own amp, else the shared
   `bloom_amp`), the `--oscillate`/`--lock` argparse flags, and an idempotent
   `resolve_oscillate(args)` that maps the parsed group model onto the canonical
   `transform`/`bloom`/`bloom_amps`/`tumble_*` fields — the exact inverse of
   `transform_to_oscillate`, so `pick_variant` needs no new code path. `--transform` default
   became `None` (clean mutual-exclusion with `--oscillate`); the legacy `--bloom`/`--bloom-amp`
   /`--tumble-*` flags stay as `--transform`-only aliases (a conflict guard rejects mixing).
   `amp*tumble` selects slide mode; `--lock <dims>` → tumble lock. Per-group `rate`/`phase` and
   bare dim indices raise a clear "not yet wired" error (they land in step 4). Validated by 14
   new tests (field-expression equivalence to the legacy `--transform` forms + guards), a real
   `--oscillate bloom,freq` render through the full CLI→ftrace pipeline, and a byte-identical
   `.ftsl` diff (incl. the `1.5*freq` per-axis-amp case). 314 loom tests green.
4. **Wire winder axes** (`drift`/`rotate`/`tumble`/bare dims) with per-group `rate`
   (= winding) and `phase`. ✅ **DONE 2026-07-18.** `resolve_oscillate` now emits three
   winder-clock outputs the picker honors (all no-ops on the legacy `--transform` path,
   so those variants stay bit-identical): `args.osc_dim_windings` — a bare dim index is
   the atomic winder, forced on and pinned to an **exact** integer winding
   `round(amp*rate)` (`--oscillate 3 rate 2` → dim 3 winds twice; `2*3` is the same since
   `amp≡rate`); `args.osc_max_winding` — an explicit `rate` on a **motion** group
   (`drift`/`rotate`/`tumble`) is the **ceiling** of the RNG-varied `1..N` winding cycle
   (overrides `--max-winding`, keeps the distinct-rate spread — "how fast, at most",
   consistent with a lone dim's exact rate); `args.osc_phase` — a constant radians offset
   (`2*pi` = one turn) added to the shared winding clock (drift/rotate/tumble), from a
   winder group's `phase`, which shifts where the loop starts while keeping `t=0==t=1`.
   The engine has one shared winding clock, so conflicting motion rates/phases across
   independent groups are rejected. Bare dims with no named motion
   default to `drift`. Validated by 12 new tests (motion-rate ≡ `--max-winding`, bare-dim
   exact/amp winding, dim-floor + off-lock conflicts, seamless phase offset, rate/phase
   conflict + swinger-rate guards) and a real `--oscillate drift rate 3` video through the
   CLI→ftrace pipeline. 324 loom tests green.
   - ✅ **DONE (P3.2b)** — the swinger envelope now carries its own clock, uniform with
     winders/bloom. Each swinger's bloom is `w(t) = 0.5·(1 − cos(2π·rate·t + phase))`
     (`_bloom_env_p`), where `rate` (`Variant.bloom_rates[key]`, default 1) is how many
     bumps it makes over the loop and `phase` (`Variant.bloom_phases[key]`, radians,
     default 0) offsets it; `bloom`/dims is keyed `"dims"`, the scalar swingers by their
     own name. `rate`/`phase` are read from the swinger's group and no longer rejected.
     Default rate 1 / phase 0 is byte-identical to the legacy fixed `sin²(πt)` envelope
     (the `--transform` path leaves both tables empty). An **integer** rate loops
     seamlessly for any phase; a **non-integer** rate pulses faster but breaks the loop
     (last frame ≠ first), so `main()` prints a one-line "won't loop seamlessly" note.
     Validated by 6 new tests (rate stored + peaks at t=¼,¾; default byte-identity;
     integer-rate seamless; phase offset flips the bump but still loops; `bloom`→`dims`
     keying; non-integer-rate warning via `main`). 365 loom tests green.
5. ✅ **DONE (P1.5)** — made `--oscillate` the single documented surface. `--transform`
   and its satellites (`--bloom`/`--bloom-amp`/`--tumble-mode`/`--tumble-amp`/
   `--tumble-lock`) are hidden from `--help` (`argparse.SUPPRESS`) but stay fully
   supported; passing `--transform` now prints a plain one-line deprecation note (before
   `resolve_oscillate` can synthesize it). The module docstring examples + conceptual
   prose and the epilog quickstart were migrated to the grammar (e.g. `--oscillate
   bloom,1.5*freq`, `--oscillate 0.3*tumble`, `--oscillate tumble --lock 0,1`). The test
   suite's incidental `--transform` setup usages were migrated to `--oscillate` (23
   auto-converted via `scraps/convert_transform_to_oscillate.py` + 2 hand edits for the
   dynamic `tr` loop and the `base` list); the two `bloom_amps`-representation tests and
   the deliberate desugaring/equivalence references stay on `--transform` on purpose. The
   default motion when *neither* flag is given is still `drift` (unchanged). 324 tests
   green. (The loom README references gyroid_nd only generically — no flag change needed.)
6. ✅ **DONE (P2.1) — `--couple` cluster command** (§6): `--couple CLUSTER CLUSTER…`
   (each cluster `d,d,…` comma-joined, spaces disjoint) with a per-cluster `:full`/
   `:cyclic` tag over a global `--couple-scheme` default (cyclic). `parse_couple` +
   `resolve_couple` produce `couple_clusters` (tuple of `(dims, scheme)`) and
   `couple_axes` (flat forced-on set, fed to `forced_on`/`max_forced_axis` like a
   `--pair …:on` endpoint — no new RNG draws). `coupling_pairs()` was refactored around a
   shared `_scheme_edges(dims, scheme)` helper: when `couple_clusters` is set it emits each
   cluster's ring/clique edges (over its *oscillating* members) in CLI order; otherwise it
   falls through to the legacy `--coupling`/`--pair` base-graph path unchanged. **Decision
   (differs from the earlier "deprecated alias" sketch below):** rather than desugaring
   `--coupling`/`--pair` into `--couple` clusters, they keep their own resolution path and
   `--couple` is **mutually exclusive** with a non-default `--coupling` / any `--pair`
   (they resolve at different times — `--coupling` acts over the post-RNG active set,
   `--couple` names explicit dims at parse time — so a clean desugar isn't possible without
   changing behavior). Empty `couple_clusters` ⇒ legacy path is bit-identical. `coupling_desc`
   summarizes clusters (`couple {0,1,2}:clique {3,4}:ring (N terms)`); the primitive-surface
   warning now also lists `--couple`. 11 new tests; 335 green.
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

**As built (P2.1):** `--coupling`/`--pair` are *not* desugared into `--couple` clusters —
they keep their own base-graph resolution path (which acts over the post-RNG active set,
whereas `--couple` names explicit dims at parse time, so a behavior-preserving desugar
isn't possible). Instead `--couple` is **mutually exclusive** with a non-default
`--coupling` or any `--pair` (a `SystemExit` nudges you to put the two dims in one cluster
instead of a `--pair` chord). Both paths stay fully working; existing scripts and the
legacy coupling tests are untouched and bit-identical.

### Resolved sub-question — internal scheme syntax

**Resolved:** a global `--couple-scheme {cyclic,full}` default (default `cyclic`) plus an
optional per-cluster trailing tag `:full`/`:cyclic` that overrides it for that cluster —
e.g. `--couple 0,1,2:full 3,4` (clique cluster + default-ring cluster).

---

## 7. Surface library & per-surface parameters

`--surface` now selects the **whole isosurface library already in the repo** (P3.3
slice S1, 2026-07-18): the two native TPMS families plus every POV builtin, each sliced
at its authored default shape params. The library is two collections:

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
param-metadata table** — `{func: [(name, description, default, [lo,hi]), …]}`.

**P3.1 (done 2026-07-18):** that table now lives in `loom/pov.py` as `POV_PARAMS`,
reachable via `pov_params(name)`. It is *not* emitted from `tools/pov_functions_gen.py`
after all — the C header only needs arity for the VM, and the metadata (names/docs/
ranges) is Python-side consumer data for `--surface-help`, so hand-authoring it in
`pov.py` next to `POV_FUNCS`/`POV_ND_GENERALIZABLE` keeps it where it's used. Real
`(name, description, default, (lo, hi))` tuples are authored for the well-documented and
N-D-core shapes (f_sphere, f_ellipsoid, f_superellipsoid, f_paraboloid,
f_quartic_paraboloid, f_rounded_box, f_torus, f_heart, f_noise_generator) plus the
0-param spherical/noise helpers (f_r/f_th/f_ph/f_noise3d); every remaining `POV_FUNCS`
entry falls back to honest generic `p0..p{n-1}` placeholders (`_generic_params`). The
completeness is guaranteed *by construction* (a comprehension over `POV_FUNCS`), and a
test asserts every entry has exactly `arity − 3` params with valid, unique axis names and
in-range defaults — the same drift-guard discipline the arity table uses.

With that table, add discovery commands. **P3.2 (done 2026-07-18):** both landed in
`gyroid_nd.py`, backed by a surface catalog (`_TPMS_CATALOG` for the periodic families +
`POV_FUNCS`) grouped by `surface_group()` into `periodic` / `nd_pov` / `affine_pov`:

* **`--list-surfaces`** — prints every surface (82 total) grouped periodic TPMS / N-D POV
  / affine-only POV, one line each with its shape-param count and `[nd]`/`[loop]` status.
* **`--surface-help NAME`** — prints one surface's parameter list from `pov_params(name)`
  (each param's axis name, meaning, default, range), or the "no shape params — use
  freq/threshold/thickness" note for a param-free TPMS; resolves the `schwarz_p`→
  `primitive` alias and errors (listing the catalog) on an unknown name. Both are
  early-exit (return 0 before generation) and ASCII-only for the Windows console.
* The main `--help` (`--surface` help text + epilog) points at both.

### Staging

This is **phase 3+** work (after the core `--oscillate` grammar lands). Order:
(a) ~~author the param-metadata table + generator + test~~ **done (P3.1)**;
(b) ~~`--list-surfaces` / `--surface-help`~~ **done (P3.2)**; (c) widen `--surface` to the
full library with the N-D and seamless-motion guards from the caveats above (P3.3).

**P3.3 slice S1 (done 2026-07-18):** `--surface` now accepts any POV builtin (validated
at runtime, `schwarz_p` alias resolved, catalog-only `schwarz_d`/`neovius` and unknown
names rejected with a clear message). A POV surface emits as a **solid** isosurface —
the interior `{f(x,y,z, defaults…) < threshold}` bounded by the field's own level set (no
`abs()`-shell, no frequency-scaled Lipschitz bound; a per-function `max_gradient` from a
small table, conservative default `8.0` until S2 tabulates the rest). Shape params take
their authored defaults; the N-D slice machinery (drift/rotate/tumble/bloom) is a no-op
on a POV field for now. Smoke-rendered `f_sphere` (exact SDF, bound 1.0) and `f_torus`
(bound 1.0) as clean solids.

**S1 follow-up — solid orientation + natural isolevel (done 2026-07-18):** POV builtins
split into two sign conventions and the naive `{f < 0}` render inverts half of them. The
SDF-like helpers (`f_sphere`, `f_torus`) are *negative* inside and cross zero on the
surface, but most clamped algebraic builtins are built as `r = -(polynomial)` then
clamped, so they are *positive* inside and rail to `-10` far away — rendering `{f < 0}`
gives the *exterior* (a shape-shaped crater, e.g. the heart came out as a sphere with
heart dimples). A per-function `_POV_SOLID_META = {name: (sign, level)}` table now records
each builtin's inside-sign and natural isolevel: the emitted field is `sign·(f − (level +
threshold))`, so a positive-inside function is negated (a sign flip leaves `|∇f|`, hence
`max_gradient`, unchanged) and a function whose surface lives at a non-zero level (e.g.
`f_ellipsoid`, `≥ 0` everywhere with the surface at level 1) is shifted before the sign
test. Un-tabulated functions fall back to the honest `(+1, 0)` passthrough. Validated:
`f_heart` now renders as a solid valentine and `f_ellipsoid` as a solid unit sphere (both
were broken before).

**P3.3 slice S2 — tight active-band gradient bound (done 2026-07-18, Option B):** the
`max_gradient` a POV solid emits is now a *rigorous, tight* per-function ceiling computed by
`loom.pov_grad.active_band_grad_bound`, not the hand-picked `8.0` default. This is a
**correctness** fix, not just a speed one: most POV algebraic builtins are
`f = clamp(P0·r(x,y,z), −10, +10)`, and a high-degree `r` has an enormous gradient far from
the surface — `f_hunt_surface`'s true `|∇f|` ceiling is ≈ 11000, so the old `8.0` default was
a catastrophic *under*-estimate that would make ftrace's sphere-marcher overstep and punch
holes. The key insight (Option B): the gradient only matters on the **active band** where the
field is un-railed (`|P0·r| < 10`); in the railed tails `f` is pinned to ±10 and flat, so a
band-only bound is both rigorous (sphere-tracing never oversteps: crossing from ±10 to 0 takes
at least `10/bound` of travel) and tight (ignores the railed high-gradient tails). The bounder
runs a **vectorised adaptive interval branch-and-bound**: cut the box into a grid, enclose
`|∇f|` per sub-box with natural interval arithmetic on the *factored* derivatives (with exact
even-power handling so the dependency blow-up never happens), certify a lower bound from
active-band sample points, discard every sub-box provably below it, octasect the survivors, and
stop when the max outstanding interval-sup is within tol of the certified max — times a small
safety factor so it never dips under the true sup. Closed-form fast paths cover the SDF-like
primitives (`f_sphere`/`f_torus` → 1, `f_ellipsoid` → `max|semi-axis|`). Noise / `atan2` /
rotation builtins aren't analyzable → the bounder returns `None` and the caller keeps the
conservative default. Cross-checked against a dense numerical gradient sample: rigorous (never
under) and tight (≈ 1.05×) for `f_heart`/`f_hunt_surface`/`f_kummer_surface_v1`; render-validated
`f_hunt_surface` as a clean hole-free solid. New module `loom/pov_grad.py` (+ `test_pov_grad.py`);
397 loom green. Still to come: S3 bbox containers + `--shell`, S4 `--lock NAME=VALUE`, S5 params
as `--oscillate` swingers, S6 affine N-D remap, S7 `--axis-default` random param draw.

**P3.3 slice S3 — auto-sized containers + `--shell` (done 2026-07-18):** a POV surface no longer
renders inside a fixed radius-1.3 ball (which *clipped* any shape bigger than that — `f_hunt_surface`
came out a featureless disk because its surface sits at radius ≈ 3.67). `--radius` now defaults to
**None**, and the container is **auto-sized to each surface's own bounding box**. Rather than a
hand-authored 78-entry table, `loom.pov_grad.surface_bbox(name, params, level)` *derives* the extent:
it grid-samples the transcribed field `f(x,y,z)` (new `_FIELD_BUILDERS` give the closed forms for the
SDF/norm builtins `f_sphere`/`f_torus`/`f_ellipsoid`; the algebraic builtins reuse `P0·r`), finds where
it crosses the isolevel, and returns `(half_extent, bounded)` — `bounded=False` when the surface runs
to the search boundary (a genuinely unbounded paraboloid / cylinder / helix, or a non-compact quartic
like `f_kummer_surface_v1`). `build_scene` calls `_pov_container_radius(name, values, level, radius_arg)`:
an explicit `--radius` always wins (and is how you *clip* an unbounded shape to a finite view);
otherwise the padded bbox (×1.08) sizes the clip sphere + `contained_by` box; a shape with no
transcribed field or an unbounded one falls back to the 1.3 default. So `f_hunt_surface` now sizes to
clip radius ≈ 3.96 / box ≈ 4.16 and shows its full surface, and `f_ellipsoid`'s long lobes aren't
clipped. `--shell` carves *any* POV shape hollow — `abs(sheet) − thickness` — and a tagged set of
genuinely-thin surfaces (`_POV_THIN_SURFACES`: `f_klein_bottle`/`f_boy_surface`/`f_enneper`/`f_cross_cap`/…
plus any `*_2d` planar curve) shell by default (a solid fill would just be a lumpy ball). TPMS are
untouched: they keep their own `abs()`-shell and the 1.3 default regardless of `--shell` / a None radius.
20 new tests (417 loom green); render-validated `f_hunt_surface` shows its full surface, not a clipped
disk. Still to come: S4 `--lock NAME=VALUE`, S5 params as `--oscillate` swingers, S6 affine N-D remap,
S7 `--axis-default` random param draw. (Note: like the S2 bound, the auto-size is cached per
`(name, params)`, so an animated param that moves the surface's extent will need a per-frame recompute.)

**P3.3 slice S4 — pin shape params with `--lock NAME=VALUE` (done 2026-07-18):** every POV surface
carries named shape params (`pov_params(name)` → `(axis, desc, default, (lo, hi))`; e.g. `f_torus`
has `major` / `minor`, `f_ellipsoid` has `rx` / `ry` / `rz`), and you can now override any of them at
render time: `--lock minor=0.4`, `--lock "rx=2 rz=0.5"`. This **reuses the existing `--lock` flag**
without colliding with its motion grammar, because that grammar never uses `=` — so any token of the
form `NAME=VALUE` is unambiguously a *param pin*, while everything else (commas, `tumble`, `spin`,
axis names…) keeps its motion meaning. `resolve_pov_param_locks(args)` splits the `--lock` string into
the two worlds: `=`-tokens become `args.pov_param_locks` (a `{name: value}` map), the remaining tokens
stay on `args.lock` (collapsing to `None` if only pins were given, so lock-motion is untouched). Pins
space-separate (`"rx=2 rz=0.5"`) and their values are full `_osc_eval_num` expressions, so
`--lock "minor=0.1+0.2"` works. Validation is strict: a pin on a **non-POV** surface errors, an
**unknown** param name errors (and lists the valid names for that surface), and an **out-of-range**
value warns but is still honored (you may deliberately push a param past its authored range).
`pick_variant` applies the pins onto `pov_default_values(surface)` before emitting, so a pinned
semi-axis both flows into the emitted `f_*(...)` call **and** feeds S3's `surface_bbox`, resizing the
auto-sized container to match (e.g. pinning `rz=0.3` shrinks the clip sphere accordingly). 13 new
tests (430 loom green). *(A follow-up fixed an ordering bug: the pin extraction ran after
`resolve_oscillate`, which parses the rest of `--lock` through the motion grammar and rejects `=`;
it now runs first. `--surface` is also resolved before both, so the grammar can classify a token
against the surface's params.)*

**P3.3 slice S5 — shape params as `--oscillate` swingers (done 2026-07-18):** selecting a POV
`--surface` *extends* the `--oscillate` axis set with that surface's named shape params, exactly as
the doc's "params become axes" promised. `--surface f_torus --oscillate minor` sweeps the tube
radius over the loop; `--oscillate 0.5*minor` sweeps it half as far; comma-joining with a motion
(`--oscillate tumble,minor`) shares one clock. The amplitude is **range-aware** so it reads
intuitively:

```
p(t) = clamp( base + amp * span * env(t),  lo, hi ),   span = (hi - base) if amp >= 0 else (base - lo)
```

with `base` the param's default (plus any `--lock` pin) and `env(t) = sin²(πt)` the shared bump (its
own rate/phase via `bloom_rates`/`bloom_phases`). So `amp = 1` reaches the param's authored **range
edge** exactly at mid-loop — a single-instant touch, no plateau — and returns to `base` at the loop
ends (seamless); `amp < 0` sweeps toward `lo` instead; `|amp| > 1` over-drives and is clamped. This
lives in a new `Variant.pov_swing = {param: amp}`, deliberately **separate** from the gyroid
dims-bloom swingers (`freq`/`threshold`/`thickness`): a POV surface has no dims cross-fade, so a
param swing drives `pov_values` per frame rather than the bloom envelope. `field_expr` and
`build_scene` evaluate the params through `_pov_values_at(v, t)`, which means the S2 gradient bound
and the S3 auto-sized container **recompute per frame from the swept values** — an ellipsoid whose
`rz` lengthens grows its container as it goes, while a torus's SDF bound correctly stays 1. Grammar
plumbing: a param-only `--oscillate` (no winder/dims axis) names a benign `drift` (POV ignores
`transform`) so it doesn't trip the "names no motion axes" guard; a swing on a non-POV surface, or a
bad axis name, errors and (for a POV surface) hints the valid param names. 12 new tests (443 loom
green); render-validated — a torus `minor` sweep runs `0.25 → 2.0 → 0.25` with the container tracking
`1.44 → 3.06 → 1.44`. Still to come: S6 affine N-D remap.

**P3.3 slice S7 — `--param-default random` for POV batch variety (done 2026-07-18):** a POV surface's
shape is its `f_*` call arguments, *not* the N-D field, so it ignores the randomized dims / freq /
harmonics that give TPMS batches their variety — a plain `-n N --surface f_torus` batch was N
*identical* images. The new `--param-default {default,random}` flag fixes this: with `random`, every
**unspecified** shape param — one the user neither pinned with `--lock NAME=VALUE` (S4) nor animated
with `--oscillate NAME` (S5) — is drawn uniformly within its authored `[lo,hi]` range per variant
seed, so each variant is a genuinely distinct shape; `default` (the flag's default) keeps the single
authored default shape (the prior behavior). The draw is the **last** consumer of `pick_variant`'s
per-variant RNG stream (after the hidden-offset and tumble draws), so turning it on never shifts the
field's other random choices — a given seed's dims/freq/tumble are byte-identical with or without it.
It is a no-op on a TPMS (which has no `pov_values`, and already varies via freq/threshold), and an
explicit pin or swing opts that param out of the draw, so `--lock major=1.6 --param-default random`
freezes the major radius while the minor still varies from variant to variant. The flag is named
`--param-default` (not `--axis-default`) to avoid colliding with the pre-existing `--axis-default`
{random,on,off} axis-polarity flag. 7 new tests (450 loom green); smoke-validated — `-n 3 --surface
f_torus --param-default random` emits three distinct `f_torus(x,y,z,·,·)` calls where `default` emits
one shared `f_torus(x,y,z,1,0.25)`.

**P3.3 slice S6 — affine N-D remap of a POV surface (done 2026-07-18):** the honest realization of
caveat 1 above — "a non-generalizable POV function's N-D slice is only an affine remap of (x,y,z)".
When the user asks for a slice **motion** on a POV surface, its three coordinates now pass through a
per-frame affine `M·p + b` before the `f_*` call: the emitted field becomes `f(M₀·p+b₀, M₁·p+b₁,
M₂·p+b₂, params)`. Rows 0/1/2 of `M` are the three visible slice axes' world directions, composed from
the same motion layers as the periodic field but read as **coordinate axes** rather than wavevectors:
**tumble** rotates the whole slice basis in N-D (`_tumbled_directions`), so a visible axis mixes with a
hidden dim — its row tilts and foreshortens out of the rendered 3-space and back, which is the marquee
`--dims>3` effect (an ellipsoid rotates, an asymmetric shape shears); **rotate** turns each visible
axis edge-on independently (its row scales by `cos α`, gaining a `hidden_offset·sin α` translation);
**drift** pans each axis by `winding·t` world units. tumble and rotate return to the identity at t=0,1
so they loop **seamlessly**; drift, being a linear pan of a *non-periodic* shape, does **not** return
at t=1 — it is deliberately non-seamless, the user's opt-in (the user chose *full affine* + *allow
drift*). The render stays rigorous because the emitted field is `f(M·p+b)` whose gradient is `Mᵀ∇f`:
`_mat3_singular_extremes(M)` (analytic 3×3-symmetric eigenvalues, pure stdlib) gives σ_min/σ_max, the
S2 marcher bound is scaled by **σ_max**, and the S3 container grows by **1/σ_min** (σ_min floored at
0.15 so a near-edge-on axis can't inflate the container without bound; an explicit `--radius` clips
instead of auto-growing). The whole remap is gated on a new `Variant.pov_motion`, set **only** by a
real explicit motion — a named `drift`/`rotate`/`tumble`, or the legacy `--transform` — so the default
`drift` the pipeline installs and the benign filler `drift` a shape-param-only `--oscillate` needs both
leave it False, and a plain `--surface f_torus` stays the static `f(x,y,z)` (exact pre-S6 behavior). 16
new tests (466 loom green); render-validated with `--dims 5 --oscillate tumble --surface f_torus` — t=0
a face-on torus, t=0.40 a tilted ring, t=0.25 a near-edge-on sliver, all hole-free and un-clipped, with
t=0 and t=1 identical. **This completes P3.3.**

### 7.x P3.4 — true N-D forms for the generalizable POV solids

The S6 affine remap above rotates a POV solid *rigidly* — it is honest, but a sphere under
tumble stays a sphere, because `f(M·p+b)` can only apply a 3×3 map. The nine
`POV_ND_GENERALIZABLE` builtins (`f_sphere`, `f_ellipsoid`, `f_superellipsoid`, `f_paraboloid`,
`f_quartic_paraboloid`, `f_ovals_of_cassini`, `f_isect_ellipsoids`, `f_cross_ellipsoids`,
`f_poly4`) instead get a genuinely N-dimensional field: for one of them at `--dims>3` under
**tumble**, loom emits a hand-written expansion `F(ξ₀…ξ_{D-1})` where `ξ = A·p + c` and `A` is the
`D×3` slice Jacobian (rows 0/1/2 the visible axes, hidden rows folded in by the same Givens planes).
At `t=0`/`t=1` the rest embedding (`A_i=e_i`, hidden rows 0, `c=0`) makes `F` reduce **bit-for-bit** to
the base `f_*(x,y,z)` — so the loop is seamless — but mid-loop the extra dims genuinely participate
(a sphere can bulge into an ovoid, an ellipsoid's axis folds away). Rigor: `|∇_p F| ≤ σ_max(A)·|∇_ξ F|`
with σ from the 3×3 Gram `AᵀA` (`_matn3_singular_extremes`), a per-function ξ-space gradient bound
(`nd_grad_bound_xi`, with a fallback for `f_superellipsoid` which has none), and a container grown by
`(nat_rad+|c|)/max(0.15, σ_min)`. Gated by `_pov_use_nd` (pov_motion ∧ tumble ∧ dims>3 ∧ surface∈the
nine); every other case keeps the exact pre-P3.4 S6 affine path byte-for-byte. **This completes P3.4.**

### 7.y P3.5 — ordered / overlapping tumble words (`--tumble-sequence`)

The automatic tumble default pairs each visible axis with a distinct hidden dim in **disjoint**
Givens planes (order-independent, each direction row mixes ≤2 unit rows so `|row| ≤ √2`).
`--tumble-sequence i-j[xN],…` replaces it with an explicit **ordered word** of planes — list order is
the composition order and pairs **may overlap** (share an axis), e.g. `0-3,3-4,0-4`. Overlapping planes
don't commute, so the word reaches order-dependent reorientation the disjoint set cannot (swap two
overlapping planes → a different mid-loop slice; a disjoint word is unchanged by any reordering). Each
plane is still a whole-turn factor (`xN` = N turns), so the loop stays seamless for **any** ordering.
The only rigor change is the periodic-field Lipschitz bound: the static `coef *= √2` shortcut becomes
`coef *= _tumble_rownorm_factor(v)` = **√(max connected-component size)** of the plane graph (union-find;
Cauchy–Schwarz: a row draws amplitude only from its component). That auto-returns √2 for a disjoint word
(every plane its own size-2 component → the disjoint default's bound is byte-identical), and grows only
when planes overlap (`0-3,3-4,0-4` → component {0,3,4} → √3). The POV affine (S6) and N-D (P3.4) paths
already compute σ_max from the **exact** per-frame matrix, so they honor an overlapping word with no
change. `--tumble-sequence` overrides `--tumble-lock`; plain `--oscillate tumble` keeps the tidy
default. **This completes P3.5.**

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
