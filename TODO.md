# TODO — in flight

Live working plan. Each item says what, why, how it gets validated, and where it stands, so an
interruption costs the work in progress and not the plan. Finished items keep one line here and
their measurements in `known-issues.md`; `design.md` gets the architecture.

**Read the "Recently finished" list before starting anything.** On 2026-09-17 I reported VOLCACHE's
deposit split, GPU-VARIANCE and GPU-BEAM-TAIL as outstanding. **All three were already closed** —
I had carried them forward from a stale task list without re-reading their entries. Checking status
costs one `grep`; chasing finished work costs a session.

---

## 0. IN PROGRESS — real hair for Alice: `include`, curves of curves, guide-driven fur

Asked 2026-09-17: convert the hair in `meshes/alice.glb` to real strands, matching the sculpted
flow by eye, without authoring every strand — via a general "curves of curves" structure. Four
pieces, built in this order because each is testable on its own and the later ones depend on it.

**What was found first (all measured, none guessed):**
- Alice is ONE fused mesh, one material, 407 792 tris. There is no hair mesh; the hair is sculpted
  into the same shell as the head. So step one is segmentation.
- **Colour cannot segment it**: forehead and hair are painted the same (hue 34 vs 36 deg, sat 0.49
  vs 0.52, val 0.91 vs 0.92). **Surface roughness cannot either**: the sculpted hair is smoother
  (5.3 deg mean dihedral) than the face (6.7 forehead, 15 cheeks). The roughness map is flat.
- What worked: warm-coloured triangles above the dress, minus a face ellipsoid placed off the nose
  tip (0, 0.723, 0.221), minus the throat, then the LARGEST CONNECTED COMPONENT — after **welding
  vertices by position** (Meshy duplicates every vertex along every UV seam; index connectivity
  shatters into 53 pieces, welded it is 8 with the hair mass at 91 634 tris). Verified from five
  angles. `scraps/alice_scalp.obj`, 0.775 m^2; the segmentation lives in session scratch and
  must be promoted to `tools/` when the groom is committed (see the standing constraint).
- She faces +z; +x is HER right. Crown / bow at (0.053, 0.894, -0.018). Hair spans y 0.41..0.90.
- The flow, read off the sculpt: everything radiates from the crown; back falls straight; sides
  fall down and slightly back over the ears, flaring at the tips; the fringe sweeps across the
  forehead from her right to her left; ends curl outward. Estimate: 4-5 curves-of-curves, ~15
  guides, 150-300k strands ~0.48 m long.
- **glTF has no curve primitive**, so the deliverable is an FTSL scene, not a GLB.

**Design decisions (agreed 2026-09-17):**
- Curves use the CAMERA's evaluator (`ftsl.h catmullRomAt`: Catmull-Rom, `spline
  uniform|centripetal|chordal`, `closed`, arc-length placement with `count`/`density`/`density_at`)
  and `camera_curve`'s syntax. The three curve implementations in the codebase are all
  Catmull-Rom family (camera: alpha-parameterised; `curve.h` strands and loom: uniform); the
  camera's is the superset. `spline` is also to be exposed on `curve`/`fur`, defaulting to
  `uniform` so nothing existing changes.
- **One recursive rule, any depth:** a `curve` is a sequence of children; a child is a `point` or a
  `curve` (inline, or by name). Children are points -> a strand (today's `curve`, unchanged).
  Children are curves -> the path is the Catmull-Rom through the children's ROOTS, the shape at a
  parameter is the point-wise Catmull-Rom blend of the children's control points on ROOT-RELATIVE
  offsets, and `count`/`density`/`density_at` place instances along the path by arc length.
  Children with different point counts are resampled by arc length to a common count at load.
  Two levels is as simple as one because a leaf is just `point x y z`. The grammar is already
  generic and recursive, so this costs no grammar change.
- A recursive `curve` with a `material` renders directly (rows of grass, a braid). Referenced from
  `fur ... guides "<name>"` it is the SHAPE FIELD that area-uniform roots on the scalp interpolate
  (k nearest guides, inverse-distance, on root-relative offsets), with the existing
  jitter/clump/curl/length_jitter applied on top.
- `include "file.ftsl"`: paths resolve relative to the INCLUDING file, then the root scene's
  directory, then cwd (the existing asset order); cycles are an error naming the chain; blocks
  splice in place so names cross files (a `fur` in the main scene can grow on a scalp defined in
  the included one). Needs one grammar rule (there is no brace-less top-level form today),
  regenerating `src/gpda/ftsl_scene.gen.cpp` via `python -m loom.grammar.emit_cpp`, a reducer
  case, and a post-parse splice in `loadSource`.

### 0.1 `include` — **DONE (v0.325.0)**
Grammar rule + reducer case + post-parse splice (`expandIncludes` in `ftsl.h`), regenerated
`ftsl_scene.gen.cpp`. `tools/include_rig.py`: split, nested-through-a-subdirectory and
inside-`prefer` all render **byte-identical** to the one-file scene; a cycle fails naming the full
chain; a missing file fails naming the including file and line. Two limits recorded in
`known-issues.md` (an included file's own asset paths resolve via the root scene; loom's reader
does not know the keyword). FTSL §1.5.

### 0.2 Recursive `curve` + `spline` on strands — **DONE (v0.326.0)**
`flattenCurveNode` in `ftsl.h`; `splineArcParams` / `resampleStrandCR` / `parseSplineAlpha`
shared helpers; `tessellateCurve` takes the knot alpha; `fur` takes `spline`; named curves
register in `curveByName_` (material-less = definition). `-dumpcurves <file>` is the exact
oracle. `tools/curve_rig.py` passes all seven checks: midpoint = average, identity bit-for-bit
(and render byte-identical), resample ends exact, level-3 product, by-name = inline, closed
spacing, spline knob wired with `uniform` byte-identical. FTSL §8.6.

### 0.3 `fur ... guides` — **DONE (v0.327.0)**
`FurSpec::Guide` + `furBuildStrandGuided` (fur.h), `guides` / `guide_blend` / `guide_falloff`
parsed in `addFur` against `curveByName_`. `tools/guide_rig.py` (deterministic roots on a 2-µm
quad, read back with `-dumpcurves`): reproduce-at-root, midpoint = average, nearest-only, and the
unguided path untouched. FTSL §8.7 "Guided grooms".

### 0.4 Alice's groom — IN PROGRESS (v0.327.0 lands the inputs; the look is not yet judged)
- `tools/alice_scalp.py` -> `meshes/alice_scalp.obj` (91 634 tris, 0.775 m^2). `meshes/` is an
  UNTRACKED asset store (`*.obj` and the `.glb`s are git-ignored — even `alice.glb`), so the OBJ
  is regenerated, not committed; the scene header says so.
- `tools/alice_guides.py` -> `scenes/alice_guides.ftsl`: guides are STREAMLINES traced down the
  sculpted hair surface (gravity on the tangent plane, re-projected onto the scalp each step), from
  four horizontal rings of roots, the last at the hairline (a root on the face snaps to the scalp's
  rim, which is the hairline). 63 guides in 4 closed rings, as curves of curves. Visualised over a
  ghost body in `png/alicehair/guides_*.png`: back and sides hug the sculpt; the first version had
  almost nothing at the front, which is why the hairline ring exists.
- `scenes/alice_hair.ftsl`: doll + `shape_only` scalp + `include "alice_guides.ftsl"` +
  `fur { guides "alice_hair" ... }`, ~190k strands. Harness: `scraps/alice_hair_test.ftsl`.
- Three real-use catches on the way, all now fixed: the emitter wrote `closed  spline centripetal`
  on one line (the strict `closed` refused it -- the check paid for itself on its first scene); a
  count-less curve of rings with 10/14/18/21 guides was refused by the equal-strand-count rule,
  which only belongs to BLENDING (fixed: a group takes any); the glb mesh block needs a fallback
  `material` even though the glb carries its own.
- **Iterations so far** (`png/alicehair/groom{1,2,3,4}_*.png`):
  1. groom1: rainbow speckle pinned to a 1e15 maximum -- NOT the hair, NOT the guides: the
     harness sun was `preset:d65 intensity 90`, a Planckian 1e14 times the real sun, hidden by
     auto-exposure in every earlier render. Five probes cleared the fibers before the one that
     mattered (no fur at all: floor at 4.7e14 in R and D). Fixed in 0.328.0/0.328.1 with a load
     guard; the quick-view key had the same fault. Written up in known-issues.
  2. groom2 (sane sun): the strands follow the sculpt's flow and hang like hair -- but the CROWN
     WAS BALD: `root_offset -0.002` buried roots, and blending surface-hugging guides on a convex
     head puts the chord INSIDE the surface. Fix: guides floated off the sculpt along the normal
     (4 mm at the root -> 16 mm at the tip), `root_offset +0.001`.
  3. groom3: crown covered, radial flow from the bow reads well on the back. Still: the fringe
     drapes over the face (hairline streamlines run downhill, the sculpt sweeps them sideways);
     fibers read grey; spectral speckle persists at 96 spp with 0.6 mm fibers (sub-pixel at this
     framing -- the documented aggregate-LOD gap). `-fur-volume` is smoother but far too coarse
     for a close-up (128^3 voxels over the head); fine for distance.
  4. groom4 (rendering): sideways sweep 2.5 on front roots, hairline ring traced as SHORT BANGS
     (~0.14 m), brighter blonde `reflect rgb 0.96 0.82 0.52  beta_m 0.2`, `-denoise`, 128 spp.
  5. groom5/6: the sparkle survives mode R's `-rgb` path (byte-identical on a fiber ball), so it is
     hair-lobe Monte Carlo variance, not spectral noise: 8x spp cuts chroma noise by the expected
     sqrt(8); softer lobes and the denoiser help; NOT a reason to fatten fibers.
  6. Machinery verified VISUALLY on toy scenes (`png/curvevis/`): nine strands morph straight ->
     curled; a level-3 node blends rows; a fur ball rendered puffball / comb-over / whorl from
     hand-placed guides. Alice's look is the groom's problem, not the tools'.
  7. The dome diagnosis: roots were spread over the WHOLE sculpted hair mass (0.775 m^2), so hair
     grew out of hair. Rebuilt (groom7): roots on a SCALP CAP (the mass within 16 cm of the skull
     centre, above the nape, ~0.1 m^2 -- `tools/alice_scalp.py` now writes
     `meshes/alice_scalp_cap.obj`); doll numbers, since she is a ~7-inch doll at 10.5x life:
     15 000 strands, radius 0.5 mm (rooted doll fiber ~0.1 mm at 1:1), rooted rows as locks
     (`clump 0.7  clump_size 0.012`), `count` not `density` so a later true-size `group { scale }`
     keeps the groom; wide framing (whole hair to the shoulders).
  8. groom7 (four wide views, `png/alicehair/groom7_*.png`): the first result that reads as a
     DOLL -- face clear, hair rooted on the cap and hanging as a fine wig over the molded base, the
     sculpt's flared curls showing through as the underlayer. 12 683 strands after the face cull,
     ~40 s a view. Faults left: the face zone reached y 0.87 and ate the bangs (forehead bare);
     the fiber reads silver (white R lobe over a pale `reflect`); 1.2 cm locks invisible.
  9. groom8: face zone lowered to a top of y 0.82 (bangs may cover the forehead, not the eyes),
     `reflect rgb 0.88 0.60 0.25`, 2.5 cm locks, `curl 0.04 curl_freq 1.5`.
- NEXT: judge groom8, commit the iteration, show the user; then colour/lighting for blonde, the
  part line / fringe shape, and what the mode-M flyby does with fibers (mode M gathers on
  strand surfaces -- its noise behaviour is a different question from mode R's).

---

## 1. The `gallery_rain` 960x540 flyby — THE DELIVERABLE, and it has never been launched

1147 frames, `camera_curve "fly"`, mode M. Everything below it (`-sunnee`, VOLCACHE, the spectral
fixes, the beam work) exists to make this frame cost and this image quality possible. The single
frame currently renders in ~96 s.

**Three known blockers, all already logged — read them before launching, not after:**

- **The machine's COMMIT limit, not its RAM, is the ceiling** (`known-issues.md`, "the `gallery_rain`
  600-frame mode-M flyby cannot start"). The verified showcase command died in the film allocation.
  That entry has the exact command and the diagnosis; the flyby is now *1147* frames rather than
  600, so the allocation is larger, not smaller.
- **A mode-M GPU gather can die with `unspecified launch failure` under concurrent GPU load — and
  the batch carries on as if it had not** (logged 2026-09-16). On a 1147-frame run that silently
  produces a hole in the sequence. Decide how the run detects this *before* starting it.
- **There is no `-frames A B`**, so a run that dies cannot be resumed at the frame it died on
  (item 7).

**Before launching:** render a handful of scattered frames (`-camera fly0000`, `fly0400`, `fly0555`,
`fly1146`) at final settings and look at them. Cheap insurance against discovering a framing or
exposure problem 900 frames in.

## 2. glTF per-texel metalness -> a `mixWeightTex`-driven mix — the likely remaining Meshy gap

glTF's metalness is **per texel**; ftrace's material type is **per material**. Alice's
metallicRoughness map has mean metalness 0.28 but **p90 0.53** — parts of that one material are
properly metallic, and the importer, typing the whole thing by the mean, renders them as a 4 %
dielectric. A 4 % coat is genuinely subtle; a metal is not.

**The fix is already scoped:** import a metalness-mapped material as a **two-child mix driven by the
map** — `Material::mixWeightTex` exists and does exactly this (a per-hit blend mask on a 2-child
mix) — with a metal `glossy` child and the `layered` dielectric child. Not built.

This is the most likely remaining difference against the viewer the user compares to, and the user
has asked about Alice's dress looking glossy more than once. Full write-up in `known-issues.md`.

## 3. An emissive mesh that is not PLANAR is silently re-oriented outward (logged 2026-09-17)

So an emissive **enclosure** — a furnace, a cove, the inside of a softbox or a lampshade — renders
black, with no diagnostic. Found while building the coat validation rig, where it cost two wrong
measurements before it was understood.

Cause: an emissive mesh's signed volume about its centroid is measured and, if negative past
`-1e-6 * area^1.5`, every triangle's winding is reversed so emission points outward. The intent is
right (an inward-wound import like `torus.obj` would otherwise glow into its own hollow) but an
enclosure is indistinguishable from that case. **The real rule is planarity, not closure**: any
emissive mesh with triangles in more than one plane is at risk.

**Fix:** an explicit opt-out on the mesh block — `emit_orient keep` alongside the current `auto` —
**not** a cleverer heuristic, because no geometric test can tell a lampshade interior from a torus
wound the wrong way. The author knows which they meant; the loader cannot.
**Workaround meanwhile:** one `mesh` block per planar face (what `tools/furnace_rig.py` does).

## 4. Mode M's CPU and GPU gathers disagree by ~9 % on `gallery_rain`'s floor grid (2026-09-16)

Open, on the surface photon-map gather. It matters *because of item 1*: the flyby is mode M, and a
9 % backend disagreement on a large visible surface means one of the two is wrong in the frames
being shipped. Full entry in `known-issues.md`.

## 5. A3 / the explicit multi-bounce layered BSDF — trigger FIRED, design named, not built

The analytic coated body is built and validated (A1 0.323.0, A2 0.324.0). What is missing is a
**directional** body under a coat, and it is no longer a judgement call — the deferred trigger's
condition 2 has fired with numbers:

- Up to **-37 % in directional albedo and -34 % in lobe width** on a glossy body under a smooth
  coat, against the "a few percent" bar the trigger itself set. `tools/a3_snell.py`.
- Both controls pass, so that number means what it says: no coat -> 0.13 %, Lambertian body ->
  0.23 % (which independently reconfirms A1's derivation).
- **No cheap fudge exists**: the lobe-width error *changes sign* with roughness, so no roughness
  rescale fixes both ends.

**An exact result worth keeping in mind:** Snell in-and-out of a *smooth* body is the **identity**
(`sin t_out = n sin t_t = sin t_i`). So this is not "the coat bends the light" — it is only about the
body's MICROFACET normal disagreeing with the coat's, which exists only for a rough body, which is
exactly where TIR traps part of the lobe and no closed form survives. The two are inseparable.

**Why not built:** `f(wi,wo)` and `pdf(wo|wi)` have no closed form; NEE needs the first at every
shading point and BDPT/VCM need both at every vertex. That is the stochastic-evaluation BSDF
(Guo/Hasan/Zhao 2018) the trigger already names — real architecture, with an MIS decision to make up
front rather than halfway through.

**Next, if picked up:** build the brute-force reference IN THE RENDERER (mode R or A/B/C with the
coat traced explicitly, which needs no MIS pdf), in an enclosure, and confirm the Python numbers end
to end before committing to the architecture.

## 6. The heterogeneous spectral-media tier has no end-to-end number

The three-tier spectral media fix (C, v0.322.0) is validated for the flat and homogeneous tiers
(coloured fog: 113 % channel spread -> 1.0 %). The **coloured-AND-heterogeneous** combination is not
covered: those renders exceeded 20-40 minutes and were stopped.

**Do not re-derive the wrong conclusion from that.** I twice called the stochastic tier
"impractically slow" and was wrong both times — 24 bins vs 8 changed nothing, narrowing the spectral
spread changed nothing, and the control I should have run first settled it: the same scene with a
FLAT spectrum, taking the scalar fast path (i.e. pre-fix behaviour exactly), is **equally slow**.
The cost is that scene's heterogeneous medium, not the spectral vector. What is missing is a **cheap
scene** that exercises the tier, not a redesign.

## 7. No CLI flag renders a RANGE of a `camera_curve` (logged 2026-09-02, partly stale)

**Correction found 2026-09-17:** a single frame *can* be selected — the curve's frames are ordinary
named cameras, so `-camera fly0555` works. (`-frame N`, `-camera fly#555` and `-res` do not exist;
the flag is `-r W H`.) The entry should be narrowed to what is genuinely missing: a **range**
(`-frames A B`), which is what would let a stopped flyby resume at the frame it died on — see item 1.

## 8. BLOCKED: the paired-timing rule into `CLAUDE.md`

"Time paired within a repetition, warm-up discarded" is a standing measurement rule that lives only
in session context. It belongs in `CLAUDE.md` — which currently carries the user's own uncommitted
changes and **is not to be touched**. Do this only if that tree becomes clean, and ask first.

## 9. Characterised, no action decided: mode D's heavy noise tail

Mode D's per-frame noise falls as roughly `spp^-0.15`. The firefly framing was largely retired
(2026-09-12) — both modes peak at the same pixel at every seed — leaving a 4 %-energy, 1669x-peaked
**connection** residual that is shared BDPT machinery rather than anything mode-specific. Recorded
in `known-issues.md`; no fix proposed, and it is not blocking anything.

---

## Recently finished (one line each; measurements in `known-issues.md`, architecture in `design.md`)

- **A1 — the coat's exit interface (v0.323.0).** A coated body's albedo becomes
  `a(1-F_dr)/(1-a F_dr)`, so a coated colour deepens like varnish instead of being washed out.
  White body 0.9998 vs 1.0000 predicted; grey 0.5 -> 0.3159 vs 0.3159. `tools/furnace_rig.py`.
- **A2 — absorption in the coat layer (v0.324.0).** New FTSL `coat { absorb <spd> depth <m> }`;
  Beer-Lambert in, out, and on every internal round trip. Measured 0.3475 vs 0.3476 analytic.
  Also fixed a host/device disagreement shipped in 0.323.0 (glossy body under a coat, ~1.6x) by
  giving the coat exactly one funnel.
- **B — mode M mis-coloured coloured speculars (v0.321.0).**
- **C — mode M mis-rendered coloured media (v0.322.0).** 113 % channel spread -> 1.0 %.
- **VOLCACHE deposit split — BUILT AND MEASURED (v0.300.0)**, 64 % faster and energy-correct.
  *Not outstanding.*
- **GPU-VARIANCE — ROOT CAUSE FOUND AND PROVEN (v0.300.2):** the two backends average a different
  number of light-side realizations at the same spp. Not a noisier deposit. *Not outstanding.*
- **GPU-BEAM-TAIL — CLOSED (2026-09-14)** as a narrow edge case: needs `-beams` *and* `sigma_t`
  between 6 and 20, one scene in the repo. *Not outstanding.*

---

## Standing constraints (so a fresh session does not have to be told)

- Bump `VERSION` in the same commit as any observable change; no rebuild → no bump. Commit freely,
  **never `git push`**.
- Every render passes `-window-min` (never bare `-window`) and launches with
  `dangerouslyDisableSandbox`. Stop a render with `ftrace -stop <pid>`, **never** `taskkill /F`.
  Never blanket-kill a shared runtime by image name (python/node/dotnet/java) — only the exact PID.
- Scratch in `scraps/`, PNGs in `png/`; a flyby series gets its own `png/<setname>/`.
- **A rig that is cited as evidence does not belong in git-ignored `scraps/`** — promote it to
  `tools/` and rewrite the references in the same commit, or the table is not re-derivable from a
  clean clone.
- `CLAUDE.md` is the user's and is not to be touched.

## Measurement discipline (each of these cost something to learn)

- **Run the controls first, and let them veto the result.** A rig that cannot see the effect
  produces confident numbers that are pure artefact: a furnace built from `light area {}` renders a
  MIRROR as exactly 0.0000, and I nearly reported that blindness as a 4 % energy bug in a material.
  `tools/furnace_rig.py` prints its three known answers before it prints anything else.
- **Never measure reflection in an open scene** — use an enclosure. This is what made the
  "mode M renders glossy 2.7x darker" claim wrong, and it had the same tell (a mirror reading 0).
- **Run the baseline control before attributing a difference**, especially before concluding that
  something new is slow (see item 6).
- Score per-ROI, never whole-frame; separate variance from systematic; hold confounds fixed by
  construction; time paired within a repetition with the warm-up discarded.
- When a quantity must be applied at N sites, **N > 1 is the bug** — build the funnel. The 0.323.0
  host/device slip was a four-site change made at three sites.
