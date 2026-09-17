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
  10. TRUE SIZE (2026-09-17, user's numbers): she is a 10-inch doll and the compote's bowl is
      6 inches. `tools/glb_rescale.py` rewrote both GLBs' root-node transforms (binary untouched,
      `.orig.glb` backups beside each): alice.glb 1.8988 -> 0.2540 m (x0.13377); compote 0.1697 m
      bowl -> 0.1524 m (x0.898, both copies -- the redcup original and meshes/ were byte-identical).
      gallery_rain shows both at the hall's display scale 3.0, so 0.762 m of doll beside 0.457 m
      of bowl (10:6); Alice's block went from scale 0.85 / 1.614 m to 3.0 / 0.762 m, seated at
      translate y 1.281363; the compote's thirteen `absorb` sigmas x1.1136 so the authored optical
      depth survives the 10 % smaller bowl. TRADE-OFF recorded in the scene: her head used to clear
      every exhibit (the siting notes' reason she was visible at all); at 0.762 m it no longer does.
  11. The hair follows the frame: alice_scalp.py applies the GLB's node transform at output, the
      guide tool measures the frame scale from the OBJ and scales its constants, the fur numbers are
      true size (radius 0.05 mm, 3.3 mm locks), the fiber is the SCULPT'S colour (linear 0.79 0.44
      0.155), the part is at her right with the fringe swept to her left (like the doll). groom9.
  12. For the hall: `tools/alice_guides.py --place TX TY TZ RY S` wraps the guides in a group with
      the doll's placement; `scenes/alice_hair_gallery.ftsl` carries the placed scalp mesh, those
      guides, and the fur at x3 (face zone transformed into hall coordinates); gallery_rain includes
      it right after its `mesh "alice"` block.
  13. MODE M test frame (true size, 30M photons, 720^2): the fibers render, but the photon gather
      lands on them as violet speckle (few photons per hair, spectral bins), and the frame cost
      510 s against ~40 s in mode R. For the flyby the hair is a small region of a 960x540 frame
      under `-denoise`; whether that is acceptable needs a real gallery frame.
  14. The gallery placement check found a real loader bug (v0.328.2): a by-name curve reference
      inside a group re-applied the group's transform once per reference level, so the chained
      guide -> ring -> hair definitions landed at (122, 20, -24). Fixed by flattening in the
      authored frame and applying a node's transform once (`curveApplyXf`); both reference cases
      are now in `tools/curve_rig.py` (11 checks). Also measured: `rotate 0 20 0` swings +z toward
      +x, the opposite of the gallery's siting note; the face zone uses the measured sign. After
      the fix the gallery culls 1703 strands to the harness's 1720, and the close-up shows the
      hair on her head, face clear, side part, blonde matching (`gallery_alice_check2.png`).
  15. The gallery still in mode M at flyby settings (`gallery_still_modeM.png`): she IS visible at
      0.762 m -- far right on her plinth, ~45 px tall, a blonde doll with no visible speckle after
      the denoiser; the compote below her at the right proportion. COST, paired on the same frame:
      **522 s with the hair vs 112 s without (4.7x)**. Not the photon trace (both ~1 s a chunk):
      the camera pass runs ~17 s/spp against ~4, because a camera path that enters the hair mass
      bounces strand to strand (the fiber BCSDF is glossy, so the gather does not land) before it
      reaches a diffuse surface. On a 1147-frame flyby that is ~33 h -> ~166 h.
- NEXT (the user's call): accept the cost; or cut it -- `points 5  segments 1` (a third of the
  segments), fewer strands, a mode-M cap on strand-to-strand bounces (a renderer change), or hair
  only in the frames that come near her. And the look itself.

---

### 0.5 THE PLAN for hair that looks real AND a flyby that renders in reasonable time

Everything below was measured on the current groom; each step says what it fixes and how it is
judged. Order matters: the look at DISTANCE is what most flyby frames see, so it comes first.

**A. The distant look (what 1147 frames mostly see).** At the flyby's scale (~45 px of doll) the
strand version reads OLIVE-GREEN and sparkly while the molded base reads warm blonde
(`png/alicehair/gallery_alice_with_vs_without.png`): the fiber lobes forward-scatter the hall's
green grid light through the mass, and rare strand-to-strand paths make fireflies.
  1. Diagnose the green: render her in the hall over a grey floor; if the cast goes, it is the
     grid's light through the fibers and the fix is the fiber material (more absorption in the
     green -- the TT lobe's colour is the `reflect` inverted), not the lighting.
     **MEASURED (2026-09-17, `scraps/_a1/`, `png/alicehair/a1_compare.png`)**: the same still with
     the grid floor swapped for grey diffuse 0.5 (everything else the same: mode M, 2 M photons,
     24 spp, 960x540; ROI = her hair, x 852..879 y 281..307). Over the GRID: strands mean RGB
     (63, 72, 46), hue 79 deg, vs the molded base (95, 76, 60), hue 27 deg -- olive and a third
     darker. Over GREY: strands (111, 113, 88), hue 64 deg, vs base (115, 108, 91), hue 43 deg --
     the same brightness, a 21 deg residual toward yellow-green.
     **RETRACTED the same day -- this comparison was confounded and its conclusion was wrong.**
     Every WITH-strands frame ran mode M on the CPU and every WITHOUT frame on the GPU (a hair
     scene was refused the GPU photon map until 0.335.0), and the two devices disagree on the
     molded base by as much as the effect being measured (known-issues, "Mode M renders Alice ...
     far from mode D"). So "the grid's green light" was credited with a gap that was mostly
     DEVICE, and the darkness was not the floor at all: it was the starved photon map (2 M over a
     45 m hall) plus the fibers' own direct light, which the walk never took (HAIR-NEE, 0.334.0).
     What the grey test does still show, device held fixed, is a real residual hue difference
     between strands and base -- ~11 deg -- which is the fiber material and belongs to B.
     The lesson is the file's own header: hold the confound fixed BY CONSTRUCTION. Swapping an
     EMISSIVE floor for a bright diffuse one also changes two things at once (removes a green
     source, adds a neutral bounce card); a black non-emissive floor would have been the clean
     single-variable test. The grey-floor scenes (`scraps/_a1/grey_*.ftsl`, frames
     `png/alicehair/a1_grey_*.png`) are kept only as that diagnostic -- their off-white ground is
     the substitution, not a render of the gallery; every frame of the REAL scene
     (`a3_grid_*`, `gallery_still_*`, `png/flyframes/*`) has the green grid.
  2. Fireflies: Russian roulette after N strand-to-strand bounces (continue with probability p,
     weight 1/p). UNBIASED, so no darkening; expected cost falls; variance rises. A hard cap is
     NOT acceptable: light hair IS multiple scattering, and a cap darkens the body of the mass
     while leaving the sheen, i.e. it changes the character, not the exposure.
     **SUPERSEDED (2026-09-17):** the "fireflies" were chroma variance of the monochromatic walk
     and few-sample colour, not rare long chains -- the fold, HAIR-NEE, 20 M photons and the
     chroma filter removed them (C). Roulette would have added variance to a walk whose cost is
     now dominated by the GPU gather; not built.
  3. Judge by the same with/without pair at flyby scale: the aggregate must match the base to
     within the frame noise (mean |diff| ~11/255 is the floor) AND look the same to the eye.
     Only then is an LOD switch (hair off at distance) invisible.

**B. The close-up look (the user's eye, in the live window).** Pending feedback. Levers: guide
placement (the tool below), lock size, fringe/part, colour calibration on a fiber ball under the
HALL's light rather than the harness suns, `beta_m`/`beta_n`.

**C. Mode M on fibers.** The photon gather lands on strands as violet speckle (few photons per
hair, spectral bins); the harness frame cost 510 s vs ~40 s in mode R. Candidates, in order of
cost: more photons + the denoiser (cheap, may suffice at flyby scale); gather AT the fiber hit
instead of continuing through it (a change to mode M's rule for hair hits -- read that path
first); the aggregate-medium tier is backward-only today and would need a mode-M twin.
  **READ AND MEASURED (2026-09-17).** Mode M never gathers at a fiber: the Hair case of
  `photonGather` samples the BCSDF and continues (up to 32 bounces) until a diffuse hit gathers.
  Two things followed. (1) The walk is MONOCHROMATIC and the hair factor was folded into the
  scalar `thr` only -- the SPECGATHER bias at its worst. Fixed in 0.333.0 (the Hair case folds
  into `SpecThr`): hue 64 -> 54 deg vs the base's 43 over grey; the speckle did not move (8.95 ->
  8.74 chroma-speckle, base floor 5.1) so it is not the camera wavelength. (2) The speckle is
  therefore the density estimate at the surfaces UNDER the mass, which few photons reach: the
  next test is photons x10 (`-n 20000000`, the photon pass costs ~1 s per 2 M) -- if the speckle
  falls toward the floor, C's answer is "more photons", and the map is built once per flyby.
  **x10 photons: no** (8.74 -> 8.10; and the mean shifted +11 % brighter, hue 54 -> 33 deg: the
  per-query gather sharpens its radius with population, a kernel bias that moves with `-n` --
  not a speckle lever). What is left is the walk's own path variance: every camera sample exits
  the mass somewhere else and gathers a different surface, and the multiple-scatter transmission
  is steep across wavelength, so a pixel's colour at 24 spp is a few effective spectral samples.
  Levers that address THAT: spp, and the chroma denoiser (`-denoise`, made for single-lambda
  speckle) with `-fireflies k` for the isolated dots. Both being measured on the grey still.
  **DENOISE: yes.** `-denoise -fireflies 3` at the same 24 spp: chroma-speckle 8.74 -> 1.34
  (the base's own denoised floor is 0.90), luma high-frequency 13.7 -> 10.3 (base 9.9 -> 8.0),
  the mean colour untouched (hue 54.7 vs 53.5) -- for ~1 % of the render time. THIS is C's
  answer for the flyby: the map once, 24 spp, the chroma filter. **96 spp without it**: 6.08
  (from 8.74 -- 1.4x for 4x the cost; pure variance would have given 2x, so part of the speckle
  is systematic per pixel and only the filter reaches the floor). Next: A.3, the with/without
  pair over the real grid floor with the fold and the filter -- the aggregate must match the
  molded base to within the frame noise before an LOD switch can be invisible.
  **A.3 MEASURED, and a confound found (2026-09-17, `scraps/_a1/compare_a3.py`).** A scene with
  `type hair` is refused by `cudaPhotonMapSupported` (the device gather shades every query as
  Lambertian), so every WITH-strands frame ran mode M on the CPU while every WITHOUT frame ran
  on the GPU -- and the two paths DISAGREE on the molded base itself: the same no-strand scene
  renders her hair ORANGE on the GPU (hue 27 deg, RGB 95/76/60) and YELLOW-GREEN on the CPU (hue
  81 deg, RGB 79/95/51), mean|diff| 22 in the hair ROI, 16 over the doll. The "olive strands vs
  orange base" that opened A.1 was mostly device, not hair. On ONE device (CPU, no filter):
  strands hue 70 deg vs base 81 deg, luminance 0.79 of the base, doll-level mean|diff| 5.9 (the
  filter's own bias on the base is 10-13, so that floor is an upper bound). A mode-D reference
  of the base is rendering to say which device is right; the loser gets a known-issues entry.
  The chroma filter at this scale bleeds the surrounding green into a 45-px doll (the base's hue
  moved 27 -> 74 deg under it on the GPU frame) -- `-denoise-levels 2` / a lower chroma
  tolerance need measuring before the filter goes into the flyby recipe.
  **THE REFERENCE (mode D, 1786 spp, `png/alicehair/a3_reference.png`)** says BOTH mode-M paths
  are wrong on the doll, hair or no hair: D has her hair warm (hue 48 deg, luminance 95) and the
  apron white; M-GPU is too orange and 19 % dark, M-CPU too green and 21 % dark, the apron
  grey-pink / green. Logged in known-issues ("Mode M renders Alice ... far from mode D"). The
  hair question therefore has two halves: (i) strands vs base WITHIN the flyby's own path
  (M-CPU): hue within 11 deg, luminance 0.79, doll mean|diff| 5.9 -- close; (ii) mode M's own
  fidelity at a small exhibit, which x10 photons may largely fix (the map is built once per
  flyby, and 20 M photons trace in ~10 s) -- measuring now on both devices. A mode-D render WITH
  strands (CPU BDPT, 15 min budget) is the true target for how the hair should look in the hall.
  **MEASURED (`scraps/_a1/compare_ref.py`, `png/alicehair/a3_photons.png`): 20 M photons is the
  fix.** M-CPU base at 20 M: hair RGB 111/103/63, hue 50 deg, luminance 92.5 -- against D's
  112/104/70, 48 deg, 95.4: the mean matches to within the frame noise (at 2 M it was 79/95/51,
  hue 81, luminance 75). M-GPU at 20 M: hue 54, luminance 86 (closer; still 9 % dark). And mode D
  WITH strands (97 spp): hair 109/108/75, hue 58 deg, luminance 97 -- the real mass is as bright
  as the molded base and a little yellower, so the 21 % darkening of the strands in M-CPU-2M was
  the starved gather, not the hair. Cost: the 20 M map traces in ~10 s and the 24-spp frame took
  no longer (4:48 vs 5:09 on the grey scene). Decisive check running: strands in M-CPU at 20 M,
  with and without the chroma filter, against D-with-strands.
  **Strands at 20 M: hair 79/75/49, hue 51 deg, luminance 68 -- still 30 % under D's strands
  (97), while the base at 20 M matches D.** So with strands mode M lacks something the base does
  not need. Diagnosis from the code: mode M's walk collects radiance ONLY at the diffuse surface a
  chain finally lands on (fibers are never gathered on, and no light is connected at a fiber),
  whereas mode D's camera paths take direct light at every fiber vertex -- the mass's own lit glow.
  Mode M already does exactly that for glossy continuations (GLOSSY-NEE: `bwNee.neeLight` at the
  vertex, the continuation MIS-weighted through `gmis` when it reaches an emitter). The fix is the
  same at a fiber: NEE with the hair BCSDF at each hair hit, the sampled continuation weighted.
  Target: D-with-strands, hair luminance 97, hue 58 deg (`png/alicehair/a3_final.png`).
  BUILT as HAIR-NEE (0.334.0, `scraps/hair_nee.py`): both mode-M walks connect to the lights at
  every fiber vertex (`neeLight` with `hs`, mode R's split: an emitter the continuation then
  hits is not counted again; env escapes stay with the continuation). **Validated** on the grid
  still at 20 M photons, 24 spp: strands hair RGB 98/95/63, hue 55 deg, luminance 85 -- from 68
  without it, against D-with-strands' 97 / 58 deg (`png/alicehair/a3_nee.png`): 60 % of the
  gap closed, 12 % still under D. Candidates for the rest: the walk's 32-bounce cap on a deep
  chain, and NEE's single light sample per fiber vertex on a hall whose largest emitter is a
  45 m quad lit only along its gridlines. Cost, paired -- CORRECTED as for the fold: on the
  head-filling view the camera pass is 1:25 without HAIR-NEE and 1:41-1:58 with it (+20-40 %;
  two shadow rays per fiber vertex through the mass); the earlier "within noise" was a frame with
  almost no hair in it. The NEE
  term is monochromatic (camera wavelength), so it adds coloured speckle the chroma filter
  removes; a SPECTRAL NEE at fibers (shared shadow ray, BCSDF x spd at the SpecThr grid) is the
  clean follow-up and would let the flyby drop the filter's chroma bleed at small exhibits.
  (Machine note: CUDA 13.4 was installed on this box at 14:53-15:03 today -- that is what filled
  D: -- and a shell older than the install lacks `CUDA_PATH_V13_4`, which the regenerated
  project's `CUDA 13.4.props` needs: "The CUDA Toolkit directory '' does not exist". Set it, or
  open a new shell.)
  **The fold's cost, paired -- CORRECTED.** The first measurement ("+15 %") used the harness's
  default camera, which shows the whole floor with a 60-px doll: barely any hair pixels. On a
  HEAD-FILLING view (`-view 0.25,0.14,0.30/0,0.09,0/20`, 960x540, 24 spp, 1 M photons, CPU) the
  camera pass is 1:03 without the fold and 1:41-1:58 with it (+60-85 %): 27 BCSDF builds per
  fiber bounce is expensive where the frame is all hair. The angular terms of the BCSDF do not
  depend on wavelength -- only the lobe attenuations do -- so a per-lobe fold (angular terms
  once, attenuations per bin) should cut that to a few percent. At flyby scale (hair a few % of
  the frame) it is still negligible.

**D. Flyby cost.** Paired on the still: 522 s with hair vs 112 s without (4.7x); the camera pass
(~17 s/spp vs ~4) because a path entering the mass bounces strand to strand before it lands on a
diffuse surface. 1147 frames: ~33 h -> ~166 h. Levers, cheapest first, each MEASURED PAIRED:
  1. `segments 1  points 5` (a third of the segments) and fewer strands -- geometry only.
  2. Russian roulette (A.2) -- fewer bounces on average.
  3. Gather at the fiber (C) -- turns a long path into one gather.
  4. LOD: hair off beyond a distance, with a CAMERA-SIDE stochastic fade (a per-frame
     probability that a camera ray ignores a strand hit) so the switch is seamless -- NOT a
     per-frame `count` ramp, which would force a photon-map rebuild per frame and cost more
     than the hair (the flyby's whole economy is one map, 1147 gathers). Only valid once A.3
     holds, or the fade itself is a visible colour shift.
  5. The flyby's own blockers still stand (item 1 below): COMMIT limit, silent launch failure,
     no `-frames` range.
  **MEASURED 2026-09-17 (after 0.334.0).** The flyby frame with hair is mode M on the CPU
  (`sceneUsesHairMaterial` refuses the GPU photon map): 960x540, 24 spp, 20 M photons, NEE ->
  5:04 alone on the machine, i.e. ~97 h for 1147 frames. The SAME frame without hair gathers on
  the GPU in ~67 s (5.7 M of 12.4 M samples in 31 s), 4.5x faster. So lever 0, ahead of the four
  above: a device Hair case for mode M's GPU gather (sample the device BCSDF and continue, the
  DSpecThr fold, NEE at the fiber), then drop the gate for mode M and prove GPU == CPU on a hair
  scene. The device forward tracer already runs hair (the deposit pass), so the sampler exists.
  **BUILT (0.335.0, `scraps/gpu_hair.py`).** GPU vs CPU on a head-filling view: luminance ratio
  1.05-1.09, hue within 5-7 deg, per-pixel diff at the frames' noise level -- the device offset the
  base shows too, nothing hair-specific. **The filter for the flyby, measured on the GPU base at
  20 M** (`scraps/_a1/compare_dn.py`): `-denoise -fireflies 3 -denoise-levels 2` keeps the hue
  (54.0 -> 53.2 deg), +4 % luminance, chroma speckle 10.4 -> 4.7; `-denoise-levels 1` and
  `-denoise-chroma 1` shift the hue to 33 deg (worse) -- so the recipe is levels 2, not the default
  3 (the earlier "bleed to 74 deg" was the 2 M map's own colour as much as the filter). The
  gallery still with strands at 20 M photons: 5.75 s per
  spp on the GPU -> ~2.3 min a 24-spp frame -> **~44 h for 1147 frames** (was ~97 h on the CPU).
  Its hair reads luminance 81, hue 71 deg (CPU+NEE 85 / 55, D 97 / 58).
  **The fold's per-lobe form -- BUILT (0.336.0)** in three steps, each proven equal to the last
  (GPU bit-identical, CPU within rounding): the BCSDF rebuild replaced by one exp + Ap() per bin
  (no gain -- the rebuild was not the cost), the colour inversion's three pow() hoisted and its
  24-bin table cached per material (no gain on a busy machine), then the angular products taken
  from the sample's own f() and the Fresnel term hoisted so a bin is one exp and a four-lobe
  recurrence: OFF 1:12 / 1:28 vs ON 1:47 / 1:47 on the head view, i.e. +20-50 % against the earlier
  +60-85 % -- load-limited numbers (the machine was in use); a quiet re-measure is owed.

**E. The hair-authoring GUI tool** (agreed 2026-09-17; design in 0.6 below).

### 0.6 THE HAIR-AUTHORING GUI (agreed 2026-09-17) -- design, on what already exists

**Why it is feasible.** ftrace already has: an imgui + Direct3D 11 viewer shell (`src/viewer_gui.cpp`,
`-viewer`) with tabs, orbit/zoom 3-D panes, SOLID and WIREFRAME rasterizer states, mesh vertex
buffers, a strand -> display-mesh converter (`strandToMesh`), click-to-inspect in its Fields pane,
and a Render pane that path-traces the loaded scene in-process; a raster preview with a host z-buffer
(`-explore`, `raster.h`); the fur generator (`generateFur`) and the recursive curve flattener as
plain functions; a `camera_curve` editor that already round-trips authored control points through
`Loaded::authoredCurves`. The tool is those pieces pointed at `curve` / `fur` blocks.

**What is NEW:** (a) surface picking -- a ray from the clicked pixel against the target mesh (its
BVH), so a plotted point lands ON the scalp; (b) a curve-hierarchy editor (curves of curves to any
level, each level its own colour); (c) an FTSL writer/reader for nested `curve` blocks and the
`fur` block, keeping the AUTHORED points (the loader flattens them away; retain them the way
`authoredCurves` does for cameras).

**Entry:** `ftrace -groom <scene.ftsl>` (or a bare mesh: auto-scalp = the whole object) opens the
viewer shell with a Groom tab. Everything in the scene renders in the 3-D pane; the target of the
`fur` block is what you pick on.

**Phase 1 -- VIEW (read-only, immediately useful) -- BUILT 0.329.0, `ftrace -groom <scene>`:** load a scene; draw the target object solid or
wireframe; draw every `curve` as a polyline, LEVEL-COLOURED (leaf guides / rings / the curve of rings
...), control points as dots; a tree panel of the hierarchy (name, level, count/density, closed,
spline); a toggle "generate hair" that runs the real `generateFur` on the fur block and shows the
strands (as tubes, via strandToMesh) -- and hides them again, because the hair hides the curves.
Judge: open `scenes/alice_hair.ftsl` and see her scalp, the four rings in four colours, and the
groom on demand.

**Phase 2 -- AUTHOR points -- BUILT 0.330.0 (`src/groom.h`; the round trip proven by `tools/groom_rig.py`):** Verified by driving the GUI (`tools/gui_drive.ps1`): N, two plotted points on the cap, an off-cap click refused, a surface drag, Ctrl+S -> the guides file rewritten with only the new strand added (rest byte-identical), the scene reloads with 70 curves and the same 13 280 fur strands. click the surface to plot a control point (picked on the mesh, so the
root is on the skin); drag to move along the surface; a modifier to pull a point OFF the surface
along the normal (tips hang in the air) or in the screen plane; Del; N for a new leaf curve; radius
per point; undo. Save writes the hair file (curves + fur) and reload round-trips it exactly.

**Phase 3 -- HIERARCHY -- BUILT 0.331.0 (live preview through `flattenCurveForTool`; `-groom-check` in the rig). Verified by driving the GUI: Ctrl-click two guides, G -> `curve_1 [L1]` with both as children; `+` on count four times -> four blended instances previewed; Ctrl+S -> `curve "curve_1" { count 4 curve "g_0_0" curve "g_0_1" }` appended, `-groom-check` PASS on all 71 curves:** select curves -> "group into a curve of curves" (a new node with them as
children, next level up, next colour); node parameters: `count` / `density` / `density_at t rho`
keyframes (drawn as ticks along the node's path), `closed`, `spline`; the node's PATH (through the
children's roots) drawn in the node's colour; the blended instances previewed live as thin polylines
when `count` is set, so you see the interpolation before any hair exists. Any depth: a curve of
rings of guides is three colours.

**Phase 4 -- FUR + RENDER -- BUILT 0.332.0 (statements edited as text; bald zones picked and drawn; the fur block patched into its span in a mixed file; `save + render` spawns a real ftrace with the live window). Verified by driving the GUI: `count` retyped to 12000 and a bald centre picked on the forehead -> only the fur block's lines changed in the copy of alice_hair.ftsl (its inline comments lost, everything around it byte-identical), the patched scene loads (12 664 strands, 2 336 culled by the moved zone), and `save + render` produced a real mode-M frame from the pane's framing (`png/groom/scene_groom.png`):** the `fur` block's parameters in a panel (count, radius, guide_blend,
clump, curl, jitter, spline, seed); `bald` zones placed by clicking a centre and dragging a radius,
drawn as wire spheres; follicles are automatic and area-uniform over the `on` object (existing
behaviour) with an option to show the root dots; "Render" hands the saved scene to the viewer's
in-process path tracer so the real hair can be judged without leaving the tool.

**Order and cost:** Phase 1 first (a day: it is display + the tree, no editing), then 2, 3, 4. The
mode-M / flyby-cost work (0.5) is independent of the tool and can interleave.

**All four phases are built (0.329.0 -- 0.332.0).** Left for polish: a live fur regenerate
without a reload (needs the parse half of `addFur` split out so a scratch Builder can run it),
named-sphere `bald "name"` zones drawn, the `on` mesh chosen from a list, per-point radius
dragging, and a `-groom` on a bare mesh with no scene (auto-scalp).
## 1. The `gallery_rain` 960x540 flyby — THE DELIVERABLE, and it has never been launched

1147 frames, `camera_curve "fly"`, mode M. Everything below it (`-sunnee`, VOLCACHE, the spectral
fixes, the beam work) exists to make this frame cost and this image quality possible. The single
frame currently renders in ~96 s.

**Three known blockers, all already logged — read them before launching, not after:**

- **The machine's COMMIT limit, not its RAM, is the ceiling** (`known-issues.md`, "the `gallery_rain`
  600-frame mode-M flyby cannot start"). The verified showcase command died in the film allocation.
  That entry has the exact command and the diagnosis; the flyby is now *1147* frames rather than
  600, so the allocation is larger, not smaller.
  **Re-read 2026-09-17: the film allocation it died in no longer exists on the GPU path** -- the
  shared mode-M device render hands each frame to the host through `onFrame` and releases it
  (main.cpp ~24725: "a flythrough runs in one frame of host RAM"), so the run holds one 960x540
  film plus the photon map (~1 GB at 20 M photons), not 1147 films. The commit charge of other
  processes on the machine is still what it is; check `commit available` before launching.
- **A mode-M GPU gather can die with `unspecified launch failure` under concurrent GPU load — and
  the batch carries on as if it had not** (logged 2026-09-16). On a 1147-frame run that silently
  produces a hole in the sequence. Decide how the run detects this *before* starting it.
- ~~There is no `-frames A B`~~ **Built (0.337.0)**: `-camera fly -frames A B` resumes at any frame.
- The "batch carried on" half of the launch-failure entry is the chain's, not ftrace's: a failed
  kernel exits 1 with `[cuda] ... kernel failed: ...` as the last line (`cudaCheckKernel`); a
  chain that checks exit codes and re-runs `-frames N end` recovers the sequence.

**Before launching:** render a handful of scattered frames (`-camera fly0000`, `fly0400`, `fly0555`,
`fly1146`) at final settings and look at them. Cheap insurance against discovering a framing or
exposure problem 900 frames in.
**DONE 2026-09-17 (`png/flyframes/sheet.png`).** The four frames at the final recipe --
`-mode M -device gpu -n 20000000 -spp 24 -r 960 540 -denoise -fireflies 3 -denoise-levels 2`
(with hair on the GPU since 0.335.0) -- render clean: the hall, the gyroid, the compote, the grid
all as in the stills; Alice is at the frame's right edge in fly0000/fly1146 and a 30-50 px blonde
figure in fly0400/fly0555. Gathering one frame alone takes ~2 min (5.75 s/spp), so the whole
flyby is ~40-45 h of GPU time; `-frames A B` resumes a stopped run. **Decisions that are the
user's:** launching the run (it owns the GPU for two days; other GPU jobs alongside a long
mode-M gather are what killed the cloud-circuit run), and whether to LOD the hair -- on the GPU
the strands read luminance 81 / hue 71 deg against the molded base's 86 / 54 deg, so a switch
would show a small step; without one the cost is as above.

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

## 7. DONE (0.337.0): `-frames A B` renders a RANGE of a `camera_curve`

**Correction found 2026-09-17:** a single frame *can* be selected — the curve's frames are ordinary
named cameras, so `-camera fly0555` works. (`-frame N`, `-camera fly#555` and `-res` do not exist;
the flag is `-r W H`.) The range is built: `-camera fly -frames 642 699` keeps only the frames
whose number lies in 642..699 (an empty selection is an error naming the path's range). Tested on
gallery_rain's `fly`: frames 5..7 render three files; 2000..2100 refuses.

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
