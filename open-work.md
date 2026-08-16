# Open work — the short list

`TODO.md` is ~4000 lines and is now mostly a *record of what shipped*: entries there are
long prose blocks whose opening paragraph reads like a plan but whose later
`**STATUS (date) … DONE**` sub-paragraph says it landed. That makes "what's actually left?"
expensive to answer.

**This file is the actionable extract, as of 2026-08-15 (ftrace v0.186.0).** It carries only
work that is genuinely undone *and* not explicitly ruled out. `TODO.md` remains the
authoritative design text — every item below names its section/item ID there, and the full
rationale, prior art and scoping live in that entry, not here.

> **Status, 2026-08-15: section 1 is empty — nothing is both unblocked and undone.**
> The O (procedural noise), P2/P3 (fur), C (mesh/VDB), E, K, F and B batches have all
> landed; §J3b's only remaining item is the one the user excluded. What is left is
> section 2 (waiting on *you*, not on code) and the standing exclusions at the bottom.
> The one piece of *new* work identified since is not a TODO.md item at all but a
> `known-issues.md` entry: **many-lights importance sampling** (a light BVH / Conty–Kulla
> tree, optionally ReSTIR DI on top), logged when mesh area lights shipped in v0.186.0.
>
> **Update 2026-08-16 (v0.187.0):** the emitter-level light tree has landed on both
> devices (75.9 s → 2.7 s GPU / 524.7 s → 7.0 s CPU on the 256-light benchmark, unbiased).
> What remains of that entry is the *mesh-triangle* level, using the tree for
> forward/BDPT/VCM emitter selection, and ReSTIR DI — see section 1.
>
> **Update 2026-08-16 (v0.188.0):** forward mode `B`'s black-mirror gap is now partly
> closed — flat mirror panels and mirror spheres image via analytic specular camera
> connections. Mirrored *meshes*, mirror-in-mirror, the 64-plane cap and the finite-lens
> modes `A`/`C` remain; they are the second entry in section 1.
>
> **Update 2026-08-16 (v0.190.0):** the world-space diffuse radiance cache (`-radcache`,
> `src/radcache.h`) shipped for mode `R` on the CPU. It is opt-in and approximate — see the
> first `known-issues.md` entry for the measured bias. What it leaves open is the third
> entry in section 1: the GPU kernel and the scalar `radiance()` path have no cache, and a
> `-radcache` run on the GPU is a silent no-op that should at minimum become an error.

Keep the two in sync: when an item below lands, mark it DONE in **both** files (or delete it
here and record the DONE in `TODO.md`). When a new open item appears in `TODO.md`, add it here.

Excluded by explicit decision — see the bottom of this file, and don't re-litigate them
without asking.

---

## 1. Unblocked — nothing external is stopping these

### Many-lights importance sampling — a light BVH, then optionally ReSTIR DI  *(ftrace)*
*Not a TODO.md item — logged in `known-issues.md` (first Open entry) when mesh area lights
shipped in v0.186.0. The only genuinely open, unblocked engineering work as of 2026-08-15.*

> **PARTLY DONE 2026-08-16 (v0.187.0): the emitter-level tree shipped, on both devices.**
> `src/lighttree.h` + `src/lighttree_build.h` + `Scene::buildLightTree`, consumed by
> `BackwardRenderer::pickEmitters` and `dPickEmitters`. The 256-panel benchmark below went
> **75.9 s → 2.7 s** on the GPU and **524.7 s → 7.0 s** on the CPU (which needed a second
> fix: the per-sample emitter-SPD table, now factored over distinct spectra — see the
> `lighttree.h` module entry in `design.md`). Unbiased: `-no-lighttree` reproduces the old
> binary bit-for-bit on a 10-scene corpus, and the deliberately hard 40 m-corridor case
> lands within 0.07 % of its reference. Flags: `-no-lighttree` / `-lighttree` /
> `-light-split` / `-light-samples`.
>
> **Still open, which is why this entry stays in section 1:** (a) forward/BDPT/VCM still
> select an emitter from the power CDF (`selectEmitter`, `render_cuda.cu`) — the tree is
> already built and uploaded, so this is a small change; (b) a **mesh** emitter's own
> triangles are still sampled uniformly by area, which is the occlusion problem the
> paragraph below measures, and is the phase the tree was designed to extend into (descend
> from an emitter leaf into that emitter's triangle tree); (c) ReSTIR DI, unstarted.

Emitter selection is a **power CDF** and, within an emitter, the draw is **uniform** (by area
for `Mesh`, by surface for quad/cylinder). Neither consults the shading point, so any part of
a light that is backfacing or shadowed still spends its share of the NEE budget and returns
zero. Measured in a controlled A/B (`scraps/occl_one.ftsl` vs `scraps/occl_two.ftsl`): an
emitter half of which is permanently occluded costs **1.31× the noise at equal spp**, and the
factor grows with the number of independently-visible patches. A compact convex emitter is
fine — a 2304-tri emissive sphere matches the analytic cone-sampled `light sphere` exactly —
so this is specifically about occlusion and orientation diversity.

**Bigger, and separately measured (2026-08-15):** the backward renderer does not *select* an
emitter at all — `BackwardRenderer::neeLight` (`src/backward.h:709`) and its CUDA mirrors
(`src/render_cuda.cu:7419/7466/7506`) **split over every emitter at every non-specular
vertex**. Same room, same total 20 000 lm, mode R 256 spp 256² (`scraps/gen_manylights.py`):
1 light 0.4 s, 16 lights 3.3 s, 64 lights 15.1 s, 256 lights **75.9 s** — all at an identical
6.25 % noise. 190× the cost for the same image. That is the largest known avoidable cost in
mode R, and the same light tree fixes it (select one importance-weighted emitter, or a few
via adaptive splitting, instead of all N).

Fix: a Conty–Kulla adaptive light tree (Arnold / Cycles / PBRT-v4) built to serve **both**
levels — a tree over emitters whose mesh-emitter leaves descend into that emitter's own
triangle tree — returning a selection pdf so `emitterGeom`'s `pdf_area` becomes
`pdfSelect / triArea`. Self-contained: `samplePoint` + the pdf in `emitterGeom`/BDPT, not the
transport; needs a device mirror. ReSTIR DI is the larger follow-on. See the known-issues
entry for the full write-up and the measurement scenes.

### Analytic specular camera connections — the rest of the mirrors  *(ftrace)*
*Not a TODO.md item — the remainder of the `known-issues.md` entry opened when the flat-panel
and mirror-sphere connectors shipped in v0.188.0.*

> **PARTLY DONE 2026-08-16 (v0.188.0).** Forward mode `B` used to render **every** mirror pure
> black: a specular vertex has a delta BSDF, so its pinhole connection pdf is zero and
> `tracePhoton` skipped the camera connect. Two analytic connectors now construct that vertex
> deterministically — `connectSpecularPlane` (exact eye-unfolding across a plane) and
> `connectSpecularSphereMirror` (Alhazen scan + bisection, ray-differential geometry factor),
> both with CUDA twins. Mirror-box agreement with the mode-`R` reference went 0.00028 → 1.0078
> against a 1.0060 outside-box control, GPU matching CPU. See `design.md`, *Analytic specular
> camera connections*.

**Still open, in rough order of value:**
* **Mirrored meshes.** `Scene::buildMirrorPlanes()` collects only *coplanar* world-space mirror
  triangles, so a curved or faceted mirrored mesh — `gallery_rain`'s chrome ring is the
  standing example — is not collected and still renders black. Wants either a per-triangle
  connector or a curved-mesh Alhazen solve generalising the sphere's. Instanced / BLAS mirrors
  are likewise skipped.
* **Lift the 64-plane cap.** `kMaxMirrorPlanes = 64` exists because the per-photon-vertex loop
  is O(#mirror surfaces) with no spatial index. The planes already carry AABBs; indexing them
  removes both the cap and the linear scan.
* **Mirror-in-mirror.** One specular vertex per connection today; a chain of two unfoldings
  would cover the common hall-of-mirrors case.
* **A `halfmirror`'s transmitted side.** Only the reflected leg is built, so whatever is
  behind a beamsplitter stays black — measured at a 19 % deficit on
  `scenes/_mirror_mats_fwd.ftsl` before that scene grew its black backing. The fix is the
  mirror-image of the plane connector: unfold *through* the plane rather than across it.
* **Modes `A`/`C`.** Pinhole-only — `lensMode`/`forwardCatch` return early, so the finite-lens
  physical camera still shows black mirrors. Needs the connection integrated over the aperture
  rather than through a point.

### `-radcache` reaches only one code path — the GPU kernel and the scalar `radiance()` have no cache  *(ftrace)*
*Not a TODO.md item — logged when the radiance cache shipped in v0.190.0. See the first
`known-issues.md` entry for the measured accuracy picture and the reasons it is opt-in.*

The world-space diffuse radiance cache (`src/radcache.h`) is read from exactly one place:
`BackwardRenderer::radianceHeroLoop` in `src/backward.h`, on the CPU, in mode `R`. Everything
else silently ignores `-radcache`. Two of those gaps are worth closing; the third is a
statement to make, not code to write.

* **The GPU backward kernel.** `bkRadianceHeroLoop` (`src/render_cuda.cu`, ~8637 / 8661 /
  8822) is the device twin of the loop that reads the cache, and it has no read site. This is
  why `-device cpu` is currently mandatory (a `-radcache` run on the GPU renders correctly and
  reports nothing, which is the worst kind of no-op — it looks like it worked). The table
  itself ports cleanly: it is a flat open-addressed array of POD cells with a 64-bit key, so
  the natural shape is upload-after-update / read-only-during-chunk, with the update pass
  staying on the host between chunks. What does *not* port trivially is reader verification —
  `radBank->vals.push_back` is a per-thread `std::vector`, and the device needs either a
  bump-allocated append buffer or an atomic-counter slab, drained back to the host each chunk.
  **The silence is fixed (v0.190.1); the port is not.** `runRender` now prints
  `[radcache] IGNORED: the GPU backward megakernel has no cache -- pass -device cpu` at the
  point the device is resolved, alongside the existing `[medium]` warning. This mattered more
  than it sounds: `-device auto` picks the GPU for mode `R` on any machine that has one, so
  the natural spelling `ftrace scene -mode R -radcache` was doing *nothing at all* on the
  development machine, correctly and without complaint. The warning is the feature's device
  story until the kernel gains a read site.
* **The scalar `radiance()` path** in `src/backward.h`. Taken whenever `heroC == 1`, and
  whenever a path enters participating media, a GRIN medium, or the finite-lens camera. The
  read placement is the same (after this vertex's NEE, before the continuation roulette) and
  the partition argument is identical, so this is mostly a transcription — but the media case
  needs thought: a cell records radiance leaving a *surface*, and a path that terminates into
  it from inside a medium must still carry the medium's transmittance along the tail it did
  not trace, which the cell does not know about. Simplest correct answer is to keep the cache
  read disabled while `inMedium` and enable it only on the surface-to-surface segments.
* **Modes other than `R`.** Forward `A`/`B`/`C`, BDPT, VCM and SPPM have no cache and are not
  obviously improved by one (their vertices are light-side, so the "what lies beyond this
  vertex" quantity a cell stores is the wrong quantity). No work planned; recorded so the
  absence reads as a decision rather than an oversight.

### ~~K1 remainder — user-supplied named RGB→spectral mapping~~  **DONE 2026-07-28 (v0.90.0)**
*TODO.md §K, item K1 — now closed.*

Shipped as `upsample "<name>" { expr "f(r, g, b, w)" }`, named by the colon head
`rgb:<name> r g b` (also `hsv:`/`hsl:`). The body is a pattern-VM expression over a vocabulary
disjoint from the surface one (`r` is RED here, not radius — surface names are rejected by
name), plus `spec:<spectrum>(w)`, which is what makes a *measured basis* expressible rather
than only closed-form arithmetic. Pinned by `-checkupsample` section (h); scene
`scenes/_upsample.ftsl`; loom twins `NamedSpectrum`/`Upsample`/`UserSpec`/`is_colour_space`.

### ~~Array-literal formals + keyword rebind — `[0 1](a)` … `(a=u)`~~  **DONE 2026-07-28 (v0.91.0)**
*TODO.md "DECISION — color-vector / array syntax" — now closed; see the increment-2 remainder
STATUS there.*

The v0.73.0 deferral was stale in both directions. The feature itself had already arrived with
the §3.3 material bundles (v0.87.0) and §3.2 per-property access (v0.89.0) — `[0 1](a)` compiled
to a program with a free `a`, and `mat(a=u)` / `src.reflect(a=u)` / `src.reflect(u)` rebound it —
but nothing pinned it, and the one genuinely open spelling was still a raw lexer error.

Pinned semantics: a literal's axes are anonymous and bind by **position**, so its "formals" are
just the driver names in its own tuple (which is what a rebind substitutes; hence the simultaneous
2-D swap `(u=v, v=u)` transposes). `formal=driver` *inside* a literal's own call is refused with
both working spellings named, generated `__arrN` blocks are re-attributed to the authoring site,
the unsaturated message now names the `(a)` deferral route, loom's `values.py` refuses the same
spelling, and `-checkarray` + `scenes/_array_formal.ftsl` pin the identities (with explicit
non-vacuity checks).

### ~~Composing array literals — `[0 1]([0.2 0.8](u))`~~  **DONE 2026-07-28 (v0.100.0)**
*TODO.md "DECISION — color-vector / array syntax", composition STATUS — now closed. This was
the last unimplemented arm of that section's grammar sketch (`coord = NAME | NUMBER | value`).*

The blocker was the **lexer**, not the loader: `splitCallArgs` already tracked bracket depth, but
`PARENWORD`'s interior class excluded `[` / `]`, so the inner literal's brackets split the token.
Widening the class is safe because the terminal's balance guarantee rests entirely on `(` / `)`
staying excluded. Composition nests to any depth, works on one axis of a multi-axis call, and works
as a term inside coordinate arithmetic; a composed literal emits only a `grid` (no `pattern`
wrapper), since `grid:__arrN(coords)` is already a legal expression term. Brackets inside a call are
captured but not balance-checked by the lexer, so the loader re-parses the argument text and reports
a malformed inner literal against the author's source. Pinned by `-checkarray` section (h).

### ~~`NAME axistuple` at a value site — `reflect grid:ramp(u)`~~  **DONE 2026-07-28 (v0.101.0)**
*Corrected a wrong claim in TODO.md's increment-2 `Deferred:` clause — now closed by that
section's table-call STATUS block.*

A **value site is not an expression site**: the slot readers only recognised `pattern:<name>`, so
`reflect grid:ramp(u)` was "unrecognized spectrum expression". The two per-hit readers
(`bindScalarPattern`, `patternedSpectrumParam`) now route a `grid:` / `scatter:` head through
`Builder::tableCallPattern`, which compiles it with the ordinary `compilePatternExpr` and appends
to `scene.patterns` — the slot holds exactly the index a hand-written one-line `pattern` wrapper
would have produced. The two load-time-constant readers (`dblParam`, `evalSpectrum`) refuse,
naming the slots that can take a per-hit value. Only the **scoped** spelling is accepted, since a
bare `ramp(u)` already means a material-bundle application; a call-less `grid:ramp` is refused with
the `(u)` to add. A composed array literal works inside a table call too
(`grid:ramp([0.2 0.8](u))`), which needed `WORD`'s balanced-group alternative widened to match
`PARENWORD`'s body. Pinned by `-checkarray` section (i).

### ~~N4a — bit-exact host-vs-device sweep of the mode-W sample lattices~~  **DONE 2026-08-05 (v0.137.0)**
*TODO.md §N item N4 — now closed, both parts.*

Shipped as `ftrace -checklattice`: structural contracts for the digit-scrambled radical inverse
(bijectivity, `π(0)=0`, grid-permutation, low-spp coverage, `rot05`, `gridUV` tiling, "sample 0 is
the canonical outcome"), then a bit-pattern comparison of **2 169 156** values — 65 732 sample
indices × 33 lattice columns — host against device. It found a real divergence on its first run:
nvcc contracts `r += digit * f` into an FMA and MSVC doesn't, so 1.6 % of values differed by 1 ULP.
Fixed with `__dmul_rn` / `__dadd_rn`; logged in `known-issues.md`.

### ~~N5 — re-measure spectral vs `-rgb` at mode W's 1 spp, then judge an RGB mode W~~  **DONE 2026-08-05 (v0.138.0)**
*TODO.md §N item N5 — measured and declined.*

The verdict is **no**, by a much wider margin than the recorded 1.27–1.7× reasoning suggested —
because that figure is a **mode-R** number and does not transfer to mode W. Re-measured on the
RTX 4090: mode R's spectral penalty has actually *widened* (1.61× Cornell, 2.30× gyroids), but a
`-heroc` sweep shows it is entirely **bundle width** — a spectral render at C=1 costs exactly what
the RGB kernel costs on Cornell, so the fixed cost of being spectral measures as zero. Mode W,
being traversal-bound (4×4 shadow rays per light per hit), is nearly flat in bundle width: on a
15-second frame the full 8-wavelength default costs **2.7%** over one wavelength, versus 61%/46%
in mode R. So a second hand-written RGB megakernel could win ~1–2%, in exchange for a permanent
bit-exactness obligation (now *tested*, via `-checklattice`) and `cudaBackwardRGBSupported`'s scope
gate blanking the deterministic preview on media / thin-film / gratings / multilayer / layered /
fluorescence / textured albedo. Not built.

Also confirmed `-rgb`'s one structural edge is closed: spectral mode W is 1-spp-clean on the
Cornell SF10 sphere at **0.82 pp** chroma error (reproducing N1's 0.80 pp). And it turned up a real
bug — see below.

### ~~`-mode W -heroc 1` silently reproduced the de-hero collapse~~  **DONE 2026-08-05 (v0.138.0)**
*Found while measuring N5.*

`-heroc 1` turns the hero bundle off, so mode W's fixed spectral quadrature collapses to one
wavelength — and on a dispersive scene that isn't approximate, it's flatly wrong: the Cornell SF10
sphere renders **46.85 pp** off in chroma (a flat green ball), with nothing printed. The viewer
absorbs this by accumulating passes (`wNeedSpp`); a batch `-spp 1` render has nothing to average.
`warnWhittedHeroCollapse` now names the offending material class. `whittedNeedsBundle` scans only
materials attached to geometry, and calls a dielectric dispersive only if its `ior` Spectrum
actually varies over 400–700 nm, so a constant-IOR dielectric doesn't nag. Print-only; the mode-W
image is byte-identical to 0.137.0.

### ~~O — procedural-texture / noise roadmap~~  **DONE 2026-08-10 (O1–O8 complete)**
*TODO.md §O (the "third bullet point" batch, user-greenlit 2026-08-07) — now closed.*

- **O2** vector noise for domain warping — v0.158.0 (`dnoisex/y/z`, `dturbx/y/z`).
- **O1** cellular / Worley / Voronoi — v0.159.0 (`worley`/`worley2`/`worleyd`/`worleyid`).
- **O3** non-stationary randomness — 2026-08-10, all four sub-items.
- **O4** anisotropic / flow-aligned noise — v0.165.0 (`gabor(...)`).
- **O5** blue noise / sparse-convolution placement — v0.166.0.
- **O6** reaction–diffusion — v0.167.0 (`texture "n" { reaction { … } }`).
- **O7** by-example synthesis / stochastic tiling — v0.170.0 (Heitz–Neyret).
- **O8** band-limiting / antialiasing — v0.168.0 (stage 1) + v0.169.0 (stage 2).

### ~~P2 / P3 — fur follow-ons~~  **DONE (P3 v0.174.0, P2 v0.178.0)**
*TODO.md §P (P1 shipped v0.150.0: the native `curve` primitive; the `fur` groom generator
shipped v0.152.0) — now closed.*

- **P3** fiber BCSDF — v0.174.0, all four stages: Marschner R/TT/TRT core (v0.171.0), scene +
  renderer wiring (v0.172.0), Yan's double-cylinder medulla (v0.173.0), Zinke dual
  scattering (v0.174.0).
- **P2** sub-pixel variance + aggregate-BSDF LOD — v0.178.0: the fiber density/orientation
  grid (v0.175.0) then the aggregate volumetric far tier, incl. `-fur-volume`.

### ~~J3b — loom N-D / generalized-grammar superset~~  **effectively closed**
*TODO.md §J3b.* Items 1 (arbitrary channel arity), 2 (generalized stop grammar / delimiter
ladder) and 3 (uniform named-input binding, parts a–d) all shipped, in ftrace as well as
loom (v0.86.0 / v0.87.0 / v0.57.0). The checkbox stays open only because of **item 4**
(N-D record *input* domain), which the user put out of scope on 2026-07-25 — see the
exclusions table. Nothing here is actionable without reversing that decision.

---

## 2. Blocked on a user decision, not on code

### D1 / D2 / D3 — hero-room showcase renders  *(ftrace)*
*TODO.md §D.*

Gated on **your sign-off**, by design: D2/D3 don't proceed until you've personally verified — in
the interactive rasterizer flyby-definition tool (the camera_curve editor) — that you like the
room composition and the camera path. That's a human aesthetic call, and the point of the gate
is not to burn expensive photon-map renders on a room/flyby that hasn't been approved.

- **D1** *(in progress)* — flyby photon-map render: GPU shared photon-map path (build once,
  gather all 144 frames), `-savemap gallery/hero_map.ftpmap`.
- **D2** — verify still: raster + one real photon-mapped frame; confirm all pieces read.
- **D3** — verify flyby: render frames + assemble; confirm the gyroid thread, the glass pass and
  a seamless loop.

Once you say "I like it", D2/D3 unblock.

### E4 — Vec3 volume grids  *(loom)*
*TODO.md §E4, "Still open".*

Blocked on **validation data**: none of the four real OpenVDB sample files on hand carry a Vec3
grid, so there's nothing to check a decoder against. Compounding it, there's **no downstream
consumer** — ftrace supports scalar float grids only, so a Vec3 read would serve loom-internal
use alone. Everything else in E4 shipped (write side, read codecs incl. half/ZIP/blosc, rotated
`AffineMap`, NanoVDB `.nvdb` ingest, and the read→transform→write capability the item was
actually about).

Also listed there as still open: sparse *storage* (sparse-source *reads* already work).

---

## 3. Open but with no current driver

### ~~E2 slice 3b — the editor's LIVE channel + binding panel~~  **DONE 2026-07-28 (v0.95.0)**
*TODO.md §E2 — E2 is now closed end to end.*

Slices 1–2 landed 2026-07-24 (config model + the live-value channel: `collect_slots`,
`SceneDriver`, `LiveSession`/`serve_live`). Slice 3a landed 2026-07-28 (v0.94.0): `ftrace
-anim <sidecar.json>` makes the fly editor edit loom's N-D *drive*.

Slice 3b closed the interactive half:
- **live values** — `ftrace -anim <sidecar> -loom <scene.py>` spawns
  `python -X utf8 -u -m loom.anim <scene.py> --config <sidecar>` and pushes a `frame` message per
  scrub position (frames latest-wins on one slot; `points`/`bindings`/`dims` on a FIFO that is
  drained first and never drops). Loom samples the curve — ftrace sends control points and asks by
  parameter `t` — so the preview cannot drift from the video loom renders. The returned `.ftsl`
  replaces the scene wholesale, so the viewport shows the *bound scene variables* moving, not just
  the camera. Verified with a deliberately static camera: identical eye/dir at every scrub point,
  and the ball still breathes (`ball_r` 0.20 → 0.95 → 0.20).
- **binding panel** — a fourth panel row: channel combo → slot combo (fed by `slots`, pick-only,
  `(none)` first), Bind / Unbind, a `chans:` grow/shrink box, and a live status readout
  (`ch 0 → ball_r | live — 4 baked, 2 ms`). Save writes the edited dims *and* bindings back to the
  sidecar; shrinking dims drops the bindings on the vanished channels, matching what loom does.

Name-keyed incremental re-tessellation (instead of the wholesale scene swap) is a later,
*measured* optimization — a cost/continuity question about derived state (`plight`, `prims`, GPU
baked triangles, the resident RGB session), not a semantics one.

### ~~J3c second half — `.ftsl` → loom Element tree~~  **DONE 2026-07-28 (v0.93.0)**
*TODO.md §J3c — now closed, both halves.*

The **emitter-reconciliation** half shipped 2026-07-26: the unused-key warning
(`Stmt::used` → `collectUnusedKeys` → an `[ftsl] warning` from `loadSource`) turned a silent
drift into a loud one, and the audit came back clean on all 11 element kinds and all 78
checked-in scenes, fixing real drift on the way (`Isosurface` couldn't emit
`samples`/`accuracy`/`refine`/`uv`; a misplaced `priority`; 6 dead `contained_by` lines).

The **reader** half shipped 2026-07-28. It is not emit's inverse — it can't be, since loom
bakes Signals at a clock — so the property proven instead is **round-trip fidelity**:
`parse_document(src).emit(ctx)` reproduces the source byte for byte (layout, alignment,
comments, blank lines, and the text *between* elements), which is what an editor actually
needs. Faithful kinds build their real class; baked kinds fall back to the new
layout-preserving `loom/block.py` (`Block`/`Stmt`/`Document`). All 11 loom-emitted kinds
round-trip byte-identically, 65 of 97 corpus files parse and 64 of those re-emit exactly (the
other 32 are full-ftrace-language forms `ftsl.epeg` deliberately doesn't model — see the
scope-boundary note in TODO.md §J3c and loom's `design.md` §8b), 1243 loom tests green.

### ~~loom `Grid` has no `.ftsl` emitter — grids can only be sampled in Python~~  **DONE 2026-08-05 (loom-only, no `VERSION` bump)**
*Noticed 2026-07-28 while auditing the loom element emitters.*

ftsl has first-class **`grid { shape / lo / hi / outside / data }`** and
**`scatter { dim / power / eps / data }`** dataset blocks (loaded in Pass 1a, `src/ftsl.h`
`addGrid`/`addScatter`) sampled from any pattern expression as `grid:<name>(c0, …)` /
`scatter:<name>(c0, …)` (`PatOp::Grid`/`PatOp::Scatter`, `src/pattern.h`). loom had the
datasets and the interpolators but **no path from one to the other**: `grid(X, Y)` raised,
and a grid-driven field could only reach `.ftsl` fully baked per frame.

Fixed as designed, and extended to `Scatter` — leaving one dataset renderable and its sibling
not would have been exactly the asymmetry that becomes debt later:

- **`GridSample` / `ScatterSample`** (`loom/spatial.py`) — new `SpatialExpr` leaves.
  `Grid.__call__`/`Scatter.__call__` are now dual-tier: a temporal query still builds the
  `GridField`/`ScatterField` Signal, a query containing any `SpatialExpr` builds the
  renderable leaf. `emit()` writes the table call; `eval_np()` is a vectorised port of
  `patGridSample`/`patScatterSample`, so the raster preview, the temporal field and the
  render all agree (checked to ~1e-15 against loom's own interpolators).
- **`GridDecl` / `ScatterDecl`** (`loom/scene.py`) — the companion blocks, values (and a
  scatter's positions) baked at `ctx.clock`. `Scene.add` collects them automatically via
  `SpatialExpr.table_decls()`, deduped by name, explicit declaration wins — the same
  mechanism `Image` → `Texture` already used.
- **Placement folds into the query.** ftsl's `grid` block has no transform, so a
  `.transformed()` dataset inverse-maps the *coordinates* instead. `Transform.inverse_apply`
  and the new `inverse_apply_spatial` now share one body (`_inverse_map(c, d, wrap)`), so the
  two tiers cannot drift.
- **Honest refusals** where ftrace cannot follow: vector-valued datasets (`PatGrid` stores
  scalar floats), `interp="cubic"` (`patGridSample` is N-linear only), `on_outside="raise"`
  (a renderer cannot throw per-sample) and > 4 axes (`PAT_ND_MAX_DIM`).

Verified: 48 new tests in `tools/loom/tests/test_grid_term.py` (1328 loom tests green), and
ftrace loads and renders the emitted `.ftsl` with no warnings — an A/B render against an
all-constant grid moves the two table-driven channels while the deliberately-constant channel
stays at ratio exactly 1.000.  The rendered image also carries the field itself: dividing each
grid-driven channel by the constant one cancels the cosine shading that otherwise dominates the
eye, leaving R/B spanning 1.00–1.77 and G/B spanning 1.10–1.57 across the sphere — the wider
red spread is exactly what the narrower green albedo range (`0.15 + 0.6*g`) predicts.

### FUTURE — loom full `.ftsl` read support  *(loom; large)*
*TODO.md, the `FUTURE` bullet under §J3c.*

The bigger version of the above: a complete `.ftsl` → `Scene` reader (not just per-element
round-trip) — the whole-file `scene { … }` wrapper rule plus a `Scene` builder that reassembles
textures/patterns/records/materials/geometry/lights/camera into a live `Scene`, plus the lossy
cases (`mesh { file … }` → a lightweight `MeshRef` that re-emits the same block; `medium`,
`pattern`, `camera_curve`).

Explicitly parked: *"the grammar's real job is ftrace's parser — so this waits until a concrete
editor need exists."* The motivating consumer would be an editor/GUI (load an existing `.ftsl`,
manipulate in loom's object model, re-emit).

### ~~F4 item 2 — re-tessellation when rotating into a parameter dimension~~  **DONE 2026-07-28 (v0.92.0)**
*TODO.md §F4 item (2) — now closed, which closes §F4 entirely.*

The loom half had been done since 2026-07-24 (`ViewerSession`/`serve_viewer` + the
`python -m loom.viewer <scene.py>` CLI). The C++ half shipped as `-loom <scene.py>` alongside
`-viewer`: `LoomLink` holds the child process, `LoomBridge` is one worker thread with a
**one-slot** pending job so `post()` overwrites anything unstarted (latest-wins — a fast drag
costs one bake of the final value, and the UI never blocks). A **Live (loom)** panel exposes
the clock, one typed control per declared keyword param, `auto` / `re-derive now`, and
`posted / baked` counters; marking a continuous param the **sweep axis** makes a right-drag on
any 3-D pane rotate into that dimension. Each bake returns a sidecar *and* an `.ftsl`, so
Curves/Fields/Meshes and F7's Render pane all refresh, preserving orbit/zoom/tab/DAG layout.

Two real bugs surfaced during validation and were fixed (written up in `known-issues.md`): the
imgui #7543 `EndGroup()` change corrupting imnodes node rects into a multi-gigabyte
`PrimReserve`, and the DAG pane re-packing itself from degenerate measurements when clipped to
zero height.

---

## Excluded — explicitly ruled out, listed so they aren't rediscovered

Full reasoning in `TODO.md` at each item. Don't pick these up without asking.

| Item | Status |
|---|---|
| **C9** Alembic (`.abc`) import | **DON'T DO FOR NOW** (user, 2026-07-24). Heavy SDK; OBJ/glTF/FBX/STL/PLY suffice. |
| **J3b item 4** N-D record *input* domain | **NOT SCHEDULED** (user, 2026-07-25). Also entangled with the axis-labelled-array work — doing it first would build a competing spelling. |
| **G3** `PatOp::MatMulAdd` intrinsic | Skip for now (2026-07-18). Deferred on **Amdahl**, not opcode mix: field eval is only 7–12% of end-to-end. Revisit past ~800 pattern nodes. |
| **G4** GPU marching cubes | Deferred. Measured 2026-07-27: a res-160 export was 55% *ASCII OBJ write*, not marching — so a free GPU march capped at 1.8×. G4b fixed the writer instead (2.5×). Revisit only for repeated/batch export or res ≥ 384. |
| **F7** `-serve` streaming path | Only needed if the raymarch is ever pushed to a separate process. |
| Mode-R / hero wavefront scheduler | Hero forces the megakernel there; single-λ by design. |
| GPU SPPM / GPU photon paths | No current demand. |
