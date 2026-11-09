# TODO — in flight

Live working plan: only what is still to do. Each item says what, why, and where it stands, so an
interruption costs the work in progress and not the plan. **Finished work moves to
`TODO-completed.md`** (verbatim, with its reasoning); its measurements stay in `known-issues.md` and
its architecture in `design.md`.

**Read `TODO-completed.md`'s "Recently finished" list before starting anything.** On 2026-09-17 I
reported VOLCACHE's deposit split, GPU-VARIANCE and GPU-BEAM-TAIL as outstanding. **All three were
already closed** — I had carried them forward from a stale task list without re-reading their
entries. Checking status costs one `grep`; chasing finished work costs a session.

---

## 1. Alice's hair — what is left (the history is `TODO-completed.md` §0)

The groom tools are built (`include`, curves of curves, guided `fur`, the groom GUI's seven phases),
Alice2 carries her v3 hair in `gallery_rain`, and her hair renders on the GPU in mode M. Open:

- **The look is the user's call**: lock size, fringe/part, colour calibrated under the hall's light
  rather than the harness suns, `beta_m`/`beta_n`. Pending feedback.
- **Flyby cost**: ~44 h of GPU time for 1147 frames with hair (5.75 s/spp at 20 M photons). Levers,
  each to be measured paired: `segments 1 points 5` and fewer strands; an LOD with a CAMERA-SIDE
  stochastic fade (never a per-frame `count` ramp, which rebuilds the map per frame) — only once
  the strands match the molded base at flyby scale, or the fade itself is a visible colour step
  (GPU strands luminance 81 / hue 71 deg against the base's 86 / 54).
- **A quiet re-measure of the per-lobe fold's cost** (0.336.0's +20-50 % was taken on a busy machine).
- **Groom tool polish**: a live fur regenerate without a reload (split the parse half of `addFur` out
  so a scratch Builder can run it); named-sphere `bald "name"` zones drawn; the `on` mesh chosen from
  a list; per-point radius dragging; `-groom` on a bare mesh (auto-scalp); a depth pre-pass so a
  closed see-through shell does not blend twice; snapping a whole strand inside a part; moving a box
  selection together. (Its untested drags are GROOM-DRAG-UNTESTED, in §4.)

## 2. The `gallery_rain` 960x540 flyby — the deliverable, waiting on the user

1147 frames, `camera_curve "fly"`, mode M at the final recipe (`-mode M -device gpu -n 20000000
-spp 24 -r 960 540 -denoise -fireflies 3 -denoise-levels 2`); four scattered frames render clean
(`png/flyframes2/sheet.png`). Decided 2026-10-16: the flight path stays as it is, Alice keeps her
`rotate 0 20 0` turn, and she wears the sparkle dress (`import_metal mix`, 0.382.0).
**Still the user's decisions:** launching it (it owns the GPU for about two days, and another GPU
job beside a long mode-M gather is what killed an earlier run), and whether to LOD the hair (§1).
The command must now name the path (`-camera fly`, since 0.383.0 a plain render is the still).
Before launching, re-read the two run hazards in `known-issues.md`: the
machine's COMMIT limit (the GPU path now holds one frame of host film, so check `commit available`
rather than assume), and a gather that dies with `unspecified launch failure` under concurrent GPU
load — ftrace exits 1 with a `[cuda] ... kernel failed` line, so the chain must check exit codes
and re-run `-frames N end`.

## 3. The A3 LOBE-WIDTH claim is untested — the rig's two sides are not geometrically identical

**The lobe-width claim is untested — its own control fails at +10.2 % with no coat present, so those
numbers can't be cited. Measuring it needs a rig whose two sides are geometrically identical (a
direction sweep against a BSDF evaluation, not two renders of different scenes).**

The detail, from `tools/a3_reference.py`. Each A3 claim is gated by its own `coat ior 1.0` control —
with no index contrast there is no coat, so both constructions must agree. The energy claim passes
that control at **−0.2 %** (explicit 0.6210 vs analytic 0.6197) and is established. The lobe claim
fails it at **+10.2 %** (r50 0.322 vs 0.355), which is *larger than several of the differences the
sweep attributes to the coat* (−5.5 % at roughness 0.02). The cause is structural: the explicit
scene carries a second sphere that the analytic scene does not, so `r50` measures that geometric
difference as much as the layered model.

Two renders of different scenes can never settle this. The replacement rig evaluates **one**
geometry: sweep the incident direction explicitly and compare a brute-force layered simulation with
the analytic BSDF at the same (wi, wo) pairs, with no scene, no camera and no fireflies.

*(Keeping the failing control is deliberate — deleting it would restore the appearance of a
measurement. A test that says "this instrument cannot answer this question" is the useful output.)*

## 4. Everything open in `known-issues.md` that should be fixed

Listed 2026-10-16 from an entry-by-entry read of `known-issues.md` (its 2026-09-10 audit is the last
time every entry was checked against the code); the entries opened since, to 2026-11-04, added then.
Each line names its entry by ID or by a phrase of its heading, so `grep` finds it; the entry has the
detail. Mark a line `[x]` with the version when it is fixed, and mark the entry DONE in the same
commit.

### 4.0 Order of work (2026-11-04: the user asked for them most important first)

Wrong images in the modes most used come first, then the build, then speed, then debt and scope.
Each names the section its line is in.

1. ~~**MERGE-PROXIMITY** (4.2) — every photon gather (M, S, U, J) takes photons from any surface in
   its radius: a table top gathers the floor's light.~~ Done, 0.431.0.
2. ~~**HAZE-SUN-B** (4.2) — mode B lights an unbounded sunlit haze at 0.16 of mode R.~~ Decided
   (2026-11-05): R, D and J stay, and the modes whose haze comes from forward photons warn. Done,
   0.434.15 (A, B, C) and 0.434.16-18 (P, V, M); D and J's own 1-2 % is HAZE-BLUR-DOMAIN (18).
3. ~~**VCM-SPEC** and **SPPM-SPECTRAL** (4.2) — modes U and S merge at the camera's wavelength, so
   a coloured surface seen before a merge washes out (the fix mode J got in 0.386.1, JSPEC).~~ Done:
   mode S 0.432.0, mode U 0.433.0.
4. ~~**MODEM-GPU-CORNER** and **MODEM-COMPOTE-NAN** (4.2)~~ — the NaN done 0.433.1; the corner was
   noise (0.433.2), and what is left of the entry, a 1 % red residual on `flat`, moves to the end.
5. **CURVE-GATHER** (4.2) — modes M, S and U read a coat of diffuse fur at 0.7-2.4 of mode R; never
   gather at a curve vertex, as mode J does. Modes M and S done, 0.433.3; mode U left
   (VCM-THIN-CURVES).
6. ~~**VCOL-BDPT-GPU** (4.2) — GPU modes D and J connect to a vertex-coloured mesh without its colours.~~
   Done, 0.433.4.
7. ~~**MODEM-BEAM-CARRIER**~~ (0.433.5), ~~**JSPEC-GAPS**' support~~ (0.433.11) and
   ~~**MODEM-REDPASS**~~ (not reproduced) (4.2) — mode M's beams, and mode J's last camera-side
   factors, at the camera's wavelength. Done.
8. ~~**UPBP-BEAM-RADIUS**~~ (0.434.0) with **J-FOG-REALCAM** (4.3) — mode J's beam kernel shrinks
   per epoch now; J-FOG-REALCAM's CPU-only extra deficit is what is left.
9. ~~**GLTF-ROUGHNESS-MAP**~~ (0.434.4) and ~~**GLTF-MAP-MEAN-GUTTERS**~~ (0.434.7) (4.2) — glTF
   imports rougher than glTF means, and their mean metalness and roughness counting the atlas
   gutters. Done.
10. ~~**FWD-GLOSSY-INDIRECT**~~ (0.434.8) (4.2) — modes A and B leave a rough reflector black in a
    mirror or through glass, and splat no clearcoat sheen. Done; LAYERED-BODY-COIN and
    FWD-SMOOTH-COAT logged.
11. ~~**CPU-ONLY-BUILD** (4.4) — a build without CUDA does not compile.~~ Done.
12. ~~**BLUR-EPOCH-COST** (4.5) — a blurred clock-budgeted mode M or J spends its clock rebuilding.~~
    Done, 0.434.12.
13. ~~**UPBP-W-CHROMATIC** (4.3) — mode J's beam weight mixes two wavelengths in a chromatic medium.~~
    Done, 0.434.19.
14. ~~**FEATHER-IS-A-THRESHOLD** (4.2) — `feather` shrinks a cloud instead of ramping it; the fix
    changes `gallery_rain`'s cloud edges, so show the user before it lands.~~ Shown, approved
    (2026-11-05), done 0.434.17.
15. ~~**UPBP-W-RAINBOW** (4.3) — mode J's merge weights read a rainbow phase's camera-side
    densities at the camera's wavelength (`gallery_rain`'s rain).~~ Measured paired on the now
    deterministic CPU: +0.05-0.07 % of the frame, 0.04 % in colour. Too small to fix (WONTFIX).
16. ~~**J-CPU-NONDET** (4.4) — CPU mode J renders a heterogeneous medium differently run to run
    (found validating UPBP-W-CHROMATIC); the CPU is the validation baseline.~~ Its epochs ended on
    the clock; they end on work counts since 0.434.21 (the user's option b).
17. ~~**MODEM-CPU-NONDET** (4.4) — CPU mode M's refresh still ends on the clock; mode J's work
    counts, for its loop (found in review of 0.434.21).~~ Done, 0.434.23 (and its camera pass's
    per-call wavelength strata, a second clock dependence); its review's follow-ups 0.434.24 (a
    golden-ratio wavelength sequence in place of the strata, a resume's fresh first light side).
18. ~~**HAZE-BLUR-DOMAIN** (4.2) — modes D and J weigh together strategies that take different
    unbounded haze (first found under a blur, for a mover outside the instant's sphere).~~ Done
    0.434.29, the user's option (a); with it J-BEAM-FARLIMIT and REFRESH-EMPTY-STOP.
19. ~~**GLOSSY-SAMPLE-SITES** (4.4) — the interference factor applied by hand at two dozen sites.~~
    Done 2026-11-06: `drawGlossy` / `dDrawGlossy` return the direction with its factor.
19a. **J-FIELD-SLOW** (4.5) — mode J in an unbounded density field (found in HAZE-BLUR-DOMAIN's
    check). Its GPU hit queue no longer spills (0.434.33-34, 126 s -> 26 s for the check's 2 spp);
    left: the beam budget's probes overshoot the knee, and fattened beams make every walk long.
20. **JSPEC-CARRIER-SIZE** (4.5) — every mode-J GPU thread reserves the spectral carrier.
21. **RADCACHE-BATCHES** (4.5) — `-radcache`'s fixed batches outlast `-stop` and the interval.
    0.434.27: a budget no longer hangs in them and `-stop` lands mid-batch (modes R and P); the
    interval writes and a resume's schedule need a canonical cache merge first (known-issues).
22. **P-HEROSPLIT-FIRST**, **ROUGH-IRID-AIR** (4.3) — narrow estimator gaps.
23. **REMAP0-IMPOSSIBLE** (4.4) — the MIS sums read every zero density as a delta's.
24. **BEAMEPOCHS-CHUNK-PIN** (4.5) — `-beamepochs` pins cheap scenes' GPU chunks to 1 spp.
25. **DEFORM-LIMITS**, **MOTION-LIGHT-RIGID**, **PER-OBJECT-MODES** (4.6) — what is left of them.
26. **BLACK-PIXELS-ONCE** (4.2) — one render's black pixels, never reproduced.

Then the rest of 4.2-4.6 in their order, 4.7's bookkeeping alongside whatever touches its entries.

### 4.1 Decisions the user owes (each blocks its fix)

- [x] **HAZE-BLUR-DOMAIN** — decided 2026-11-06: (a), done in 0.434.29 (`tools/check_haze_domain.py`).
  (Measured on 0.434.24.) In an unbounded sunlit haze, modes
  D and J's sun subpaths walk the haze from the entry disc's plane while their NEE legs, and mode
  R's, count it out to `|centre - p| + sceneRadius`. So D reads 1-2 % brighter than R in a still
  scene (1.0175 on a sunlit ball, 1.011 on the haze), and more for a mover far past the instant's
  sphere under a blur (1.0275 at 32 m). (a) Weigh the subpaths' first leg by the NEE leg's haze:
  R stays, D and J move 1-2 % dimmer to meet it. (b) End every NEE leg at the entry plane's slab,
  the domain the subpaths and the forward photons already use: R (and D and J's NEE strategies)
  move 1-2 % brighter. Recommended: (a), since R is the reference and HAZE-SUN-B kept its domain.
- [x] **J-CPU-NONDET** — decided 2026-11-06: (b), done in 0.434.21 (`tools/check_jdeterminism.py`).
  CPU mode J is not repeatable at a fixed `-spp`, because its light-side
  epochs end on the clock (renderChunked: an epoch lasts its rebuild time over `-beamrefresh`'s
  share). (a) Keep that, document that a deterministic render needs `-beamepochs K`, and pass it in
  the validation tools; (b) on the CPU at a fixed `-spp`, end epochs on work counts (subpaths and
  beams against camera samples) instead of seconds -- adaptive and repeatable; or (c) fixed epochs
  by default there -- repeatable, no longer adaptive. Recommended: (b).
- [x] **TILTED-LIGHT-EMIT** (0.384.0) — a tilted quad light was measured by its patch on the camera
  side and by its authored normal on the light side. The user's decision (2026-10-16), instead of
  the proposed snap: a leaning normal AIMS the light and the rectangle keeps its orientation. Every
  strategy on both backends now reads one model (`Emitter::aimCos`); `check_tilted_light.py`.
- [x] **`scenes/gallery_settled.ftsl` has diverged from `scenes/gallery.ftsl`** — closed by the
  user's answer (2026-10-16): the scene in use is `gallery_rain.ftsl`; no back-port.
- [x] **`textures/marble_dumbbell.png` derives from a WATERMARKED stock preview** — and so did four
  more caps: sources 1 (Vecteezy) and 2 (pngtree) carry tiled watermarks, legible in the shipped
  textures. All five re-quarried from the clean sheets 7 and 8 (2026-10-16, `make_marble_caps.py`
  now refuses the three).
- [x] **Rendering a scene without `-camera` also renders its 600-frame flyby** — left to judgement;
  stills by default, paths by name (0.383.0).
- [ ] **COMPOTE-NO-CAUSTIC** — under the 20° sun the gem compote's ruby bowl absorbs the light that
  made its caustic under the old 45° sun, and no cap fixes it: move the compote, change the bowl,
  or accept it?
- [x] **HAZE-SUN-B** — decided 2026-11-05: (b), done in 0.434.15 (`tools/check_haze_warning.py`).
  Which haze does an unbounded medium under a sun hold? A forward mode starts
  its sun photons on the disc the scene's bounding sphere casts, so mode B lights only the haze in
  that cylinder, while modes R, D and J light it all the way out along each camera ray. On
  `scenes/_haze_sun_b_vs_r.ftsl` (0.431.0, against mode R): haze over the floor D 1.008, J 0.981,
  B 0.144; the floor D 1.015, J 1.025, B 0.910. (a) Clip an unbounded medium to a sphere around
  the scene and its cameras in every mode: all agree, but R, D and J lose the haze glow wherever
  the camera looks past the scene. (b) Keep R, D and J as they are, and have the forward modes
  warn at load that they light an unbounded haze only near the scene. Recommended: (b), since the
  three that agree are the modes scenes are rendered with.
- [ ] **GLOW-LIGHT-MODES** — a glow with no emitter lights nothing in R/M/S/B, part of itself in
  D/J, all of itself in U; mode P runs away on it (x368). The fix is to register emitters for
  emissive analytic primitives (4.6's first line), which makes `gallery_rain`'s grid lines light
  the scene: does the grid keep glowing unlit (a material flag), or does the scene get rebalanced?

### 4.2 Wrong, missing or silent output

- [x] **MERGE-PROXIMITY** (0.431.0) — every gather keeps only the photons near its tangent plane
  (`gatherplane.h`), except on a curve; mode M's sheet lit from below read 21x, now 0.
- [x] **HAZE-SUN-B** (0.434.15, 0.434.16) — mode B lights an unbounded sunlit haze at 0.163 of
  mode R, its floor 0.846; the modes whose haze comes from forward photons (A, B, C, P, V,
  M) say so (the user's choice, 4.1).
- [x] **HAZE-BLUR-DOMAIN** (0.434.29) — in an unbounded sunlit haze modes D and J's sun subpaths
  walked less haze than their NEE leg counts; they take the NEE leg's now (option (a)).
- [x] **VCM-SPEC** (0.433.0) — mode U's camera walk carries mode J's spectral carrier, and each
  merge reads it at the light vertex's wavelengths.
- [ ] **MODEM-GPU-CORNER**, what is left — the corner was noise (0.433.2); GPU mode M reads
  check_bitid's `flat` red 1 % low against the CPU (3.5 sigma).
- [ ] **CURVE-GATHER**, mode U's half (VCM-THIN-CURVES) — modes M and S done (0.433.3); mode U
  still merges on a `type diffuse` strand. Was: modes M, S and U misread a coat of `type diffuse` curves
  (`tools/check_fur_gather.py`), M's two backends differently; VCM-THIN-CURVES is its mode-U half.
- [ ] **PROBE-PARITY-LIMITS** — the footprint probe's crossing count misreads a photon-holding open
  surface hovering within the radius, and an open solid taller than it. Closed solids (analytic
  spheres, closed meshes) say their own side (0.434.5, PROBE-SOLID-SIDE); since 0.434.9 so do
  inward-wound meshes and quads closing a box, and open sheets that hold no photon count for
  nothing; since 0.434.10 a closed solid taller than the radius is known from the path to each
  probe's top (PROBE-PATH); since 0.434.11 a closed isosurface says its side too; since 0.434.13 a
  sheet hovering over the floor leaves its floor covered and an open shell's stays a miss (the low
  leg). Left: an open solid taller than the radius.
- [x] **PROBE-UNDERSIDE** (0.434.10) — beside a ball resting on the floor the probe ended on the
  ball's tilted underside, seen from inside, where `-gageom`'s march counts the floor under it. It
  passes that face to the floor now; the diffuse ball's mode S 1.08-1.11 -> 1.00.
- [ ] **MODEM-CPU-DARKGLASS-BLUE** — CPU mode M reads blue 15 % low through strongly absorbing
  glass, where GPU mode M and both mode S do not.
- [x] **MODEM-COMPOTE-NAN** (0.433.1) — the spectral carrier's ratio overflowed a float in the
  device's gather (and mode S's per-pixel copy); the float side now takes ratio / peak.
- [x] **VCOL-BDPT-GPU** (0.433.4) — `DVertex` and `DVcmLV` carry the hit's vertex colour; GPU
  modes D, J and U had read a vertex-coloured sphere near grey.
- [ ] **GPU-PHOTON-LAUNCH-FAIL** — one GPU photon pass died with "unspecified launch failure"
  (mode M, a mix wall, beside another GPU process); six reruns clean. Watch for a recurrence.
- [x] **COMPOUND-MERGE-PICK** (0.434.3) — modes U and J's merges took a compound's lobe probability
  twice (a half-weight mix wall 0.71-0.75 of mode R in mode U, 0.63 in mode J); each stored vertex now
  carries the merges' share of it.
- [x] **COMPOUND-DOUBLE-PICK** (0.433.10) — modes M and S took a mix's or a layered coat's lobe
  probability on the photon's side and again on the camera's (a half-weight mix 0.50 of mode R, a
  clearcoat 0.89); found testing the review's coat-coin finding.
- [x] **MODEM-BEAM-CARRIER** (0.433.5) — mode M's beam gather reads its walk's spectral carrier and
  the glass around the segment at each beam's wavelength (fog through tinted glass: R 0.54, G 2.63
  of mode R before).
- [x] **JSPEC-GAPS**' support (0.433.11) — modes J and U divide the merges past a factor that is 0
  somewhere by the share of the camera wavelength's density left open (a fog through a gel read
  0.64 of mode R in mode J, a wall behind one 0.71 in mode U). Left by design: dispersive
  refraction and a real lens keep the camera's direction unless `-jdispall`.
- [x] **MODEM-REDPASS** — not reproduced on 0.433.10 (scenes/_redpass.ftsl recreates it; green
  within 1.4 % of mode R in modes M and J on both backends).
- [x] **FWD-GLOSSY-INDIRECT** — modes A and B splat a rough reflector only where it is seen directly
  (and a layered clearcoat's sheen on neither backend). 0.434.8.
- [ ] **LAYERED-BODY-COIN** — a forward mode weighs a coated body by the photon's coin, at the light's
  angle (5 % on the car paint).
- [ ] **FWD-SMOOTH-COAT** — a smooth clearcoat shows no sheen in modes A and B.
- [x] **GLTF-MAP-MEAN-GUTTERS** — the importer's mean metalness and roughness count the atlas's
  gutters.
- [x] **FEATHER-IS-A-THRESHOLD** (0.434.17) — a mesh-bound medium's `feather` shrinks the silhouette
  instead of ramping the density (`tools/check_feather.py`).
- [ ] **BLACK-PIXELS-ONCE** — one GPU mode-R render had pure-black pixels four reruns did not.

- [ ] **D-SILENT-EXIT** — one host mode-D render ended with no output and no image during a CUDA
  build; 13 reruns clean. The checks now print the exit code; reproduce under memory pressure.
- [ ] **SPECGATHER-BIN-RESOLUTION** — mode M's 24-bin carrier misreads a spectral edge or kink inside
  a bin; the 555-585 nm ramp's green excess is unexplained.
- [x] **SPPM-SPECTRAL** (0.432.0) — mode S's visible point carries its walk's spectral factors per
  bin (SpecThr), and each gathered photon is weighted at its own wavelength.
- [ ] **BIRTH-EMPTY-STACK** — a light walk born inside a dielectric does not know it is inside.
- [ ] **GRIN-RADIANCE-SCALE** — a camera path ending inside a GRIN region misses the n² radiance
  scale.
- [ ] **STUBBLE-DEVICE-UNDERSIDE** — mode W on a stubble coat: device and host 9.6 % apart in its
  dark underside (to investigate).
- [ ] **GATHERAREA-TRANSPARENT's remaining limits** — an opaque, photon-holding surface hovering
  within the radius above another; Mix compounds.
- [ ] **EMISSIVE-VOLUME-SILENT, what is left** — GPU mode D falls back to the CPU on a fire; modes J
  and U refuse a fire (their light passes need fire-born subpaths, J's beams their own births); mode
  P refused until its composite is checked with a fire.
- [ ] **SPECTRAL-BUNDLE, what is left** — the other modes' media gates (the device's mode D among
  them); a fiber with a dispersive index (`hair.h` takes one `eta`).
- [ ] **ALICE2-CHEEK-DOT** — a pinprick dark dot on the Alice2 face in front of each ear.
- [x] **GLTF-ROUGHNESS-MAP** — a glTF roughness is fed to the Phong lobe raw, so imports render
  rougher than glTF means. 0.434.4.
- [ ] **"a quad's emission is invisible from the side its winding faces away from"** — re-check
  against AREA-LIGHT-SIDE (0.377.2), which fixed a neighbouring case.
- [ ] **"`-dual-scatter` drops the coat's indirect illumination"** (its part 2).
- [ ] **"`curv` reads 0 on an ISOSURFACE"** — a diagnostic names it now; the Hessian is the fix.
- [ ] **"`curv` reads 0 in the PREVIEW rasterizer"** — mechanical.
- [ ] **"`cavity` reads 0 in BOTH preview rasterizers"** — not mechanical.
- [ ] **"mode `W` is NOT noise-free at `-spp 1` when the scene has a medium"** — the fog term is
  still a Monte-Carlo free flight.
- [x] **"freshly CMake-configured build dirs produce a GPU-silently-dead ftrace.exe"** (0.385.0) —
  the host STL's float `copysign` compiled to a device function returning 0; `build.bat` runs the
  device probe after every build.
- [ ] **"Headless-spawned `-window` render on gallery.ftsl hangs with no output"** — to investigate
  (compare CLAUDE.md's note on the sandbox's non-interactive window station).
- [ ] **"`scenes/gallery_settled.ftsl` OOMs — but ONLY when the 600-frame flyby is in the camera
  selection"** — re-test: the GPU path now renders a frame at a time; the CPU path may not.
- [ ] **"Forward modes render ~5% brighter than the backward reference (`R`)"** — from 2026-07-12;
  re-measure first, then reconcile the constant.
- [ ] **"rainbow phase — `SpecVtx::term` ... is HG-only"**.
- [ ] **"`phase rainbow` — the 2048-bin uniform-in-mu table under-resolves large droplets"**, and
  monodisperse supernumeraries read as a white arc.
- [ ] **"forward mode B barely converges the water-droplet rainbow"**.
- [ ] **"interactive raster fly-viewer can peg all cores / grow RAM when orphaned"**.
- [ ] **"native `-viewer` Meshes tab ignores a skin's `clamp`/`mirror` wrap mode"**.
- [ ] **"Mode M renders Alice ... far from mode D, differently on each device"** — 20 M photons
  closed most of it (the GPU still ~9 % dark): re-measure, then fix or close.
- [ ] **"Residual: hue lost at true underflow"** — see-through on a very deep stack (minor).
- [ ] **"a `curve`'s azimuthal `v` is a PER-SEGMENT frame, not parallel-transported"**.
- [ ] **"an `sdf` over a self-overlapping mesh is exact outside but not inside"**.
- [ ] **"`settle.drop()` falls back to the surface's TOP PLANE when the footprint 'misses'"** — find
  why the down-rays miss.
- [ ] **"an `include`d file's own asset paths resolve through the ROOT scene"**.
- [ ] **"`tools/_gemsweep.py` — `spread` is not resolution-stable"**.
- [ ] **"`tools/_capchroma.py` scores marble VEINS as a caustic"** — refuse a textured cap.
- [ ] **"`python -m loom.viewer` prints a runpy double-import RuntimeWarning"** (cosmetic).

### 4.3 Estimator approximations and accuracy debt

- [x] **UPBP-BEAM-RADIUS** (0.434.0) — mode J's beam kernel shrinks per light-side epoch under
  -vcmalpha (_fog_cornell -0.70 % of mode D fixed, -0.43 % at 256 epochs, -0.07 % per sample).
- [ ] **J-FOG-REALCAM**, what is left — CPU mode J reads realcam's fog up to 2.8 % below GPU mode J.
- [x] **UPBP-W-CHROMATIC** (0.434.19) — in a chromatic medium a beam merge's weight mixed two
  wavelengths and the frame tilted in colour (`tools/check_jchroma.py`).
- [x] **UPBP-W-RAINBOW** — the same mix in a rainbow-phase medium, through the camera side's phase
  densities: measured at +0.05-0.07 % of the frame, too small to fix.
- [ ] **P-HEROSPLIT-FIRST** — blurred mode P under `-herosplit` files a sample by the hero's first
  event alone.
- [ ] **ROUGH-IRID-AIR** — a rough interference reflector always interferes against air.

- [ ] **BDPT-MIS-TR** — mode D's MIS weights take every connection edge's transmittance as 1 in a
  medium.
- [ ] **UPBP-W** — three approximations in mode J's merge weight (variance only).
- [ ] **"BDPT connection edges through colored glass are not absorption-weighted"**, and mode U's
  connections share it (**"VCM (mode `U`): merges use a spectral XYZ estimate"**).
- [ ] **"VCM vertex-merge needs a gather-side shading-normal reweight that M/S don't"** — the
  asymmetry is not understood.
- [ ] **"`-radcache` is opt-in because it is *approximate*"** — residual on caustics/specular; not
  deterministic across frames.
- [ ] **"mode-R GPU GRIN has a small bent-region float-vs-double residual"**.
- [ ] **"GRIN tech debt (Phase-1 semantics carried forward)"** — only primary rays bend, and the
  rest of that entry's list.
- [ ] **"mode `W` picks a dielectric's dominant branch, it does not fork"**.
- [ ] **"the tone map hard-clips with no shoulder"** — a real defect the anchor studies assumed away.
- [ ] **"the sigmoid upsampling model itself is ~0.019 off at pure white"**.
- [ ] **"`fw` is 0 at every secondary bounce"** — `fnoise` does not band-limit in a reflection.
- [ ] **"a non-uniformly scaled instance's `curv` is approximated by |det|^(1/3)"**.
- [ ] **"GPU photon mapping isn't run-to-run bit-stable"** (~3 % under `-beams`), and **"GPU raster
  see-through output isn't run-to-run bit-stable"**.
- [ ] **"a `hair` coat is very noisy in the stochastic modes"** — path variance through the mass;
  the denoiser is today's answer.
- [ ] **"`type hair` cannot be gathered on in modes M / S"** — the photon record has no incident
  direction.
- [ ] **Mode D's heavy noise tail** — per-frame noise falls as ~`spp^-0.15`; a 4 %-energy,
  1669x-peaked connection residual in shared BDPT machinery. Characterised, no fix proposed yet.

### 4.4 Tech debt

- [x] **CPU-ONLY-BUILD** — a build without CUDA did not compile (`main.cpp` named device-only types
  outside `#ifdef HAVE_CUDA`). The two device headers are included in every build now.
- [x] **J-CPU-NONDET** (0.434.21) — CPU mode J rendered a heterogeneous medium 2 % differently run
  to run: its epochs ended on the clock; they end on work counts now.
- [x] **MODEM-CPU-NONDET** (0.434.23) — CPU mode M's light-side refresh ended on the clock, and its
  camera pass stratified wavelengths per clock-sized call; both deterministic now. 0.434.24: the
  wavelengths follow a golden-ratio sequence by absolute sample, which no resume or stop tilts.
- [x] **GLOSSY-SAMPLE-SITES** — the interference factor is applied by hand at every glossy sampler.
  Done 2026-11-06: one draw helper per backend carries it.
- [ ] **REMAP0-IMPOSSIBLE** — the MIS sums read a zero density as a delta's, whatever made it zero
  (only the sun's is special-cased).

- [ ] **GPU-STACK-DRIFT** — every kernel's stack reservation moves when unrelated device code changes.
- [ ] **SLAB-DDA-COPIES** — the box clip and the grid walk are written out several times.
- [ ] **GMIS-NULL-BY-HAND** — every caller keeps the glossy MIS record across a pass-through by hand.
- [ ] **GROOM-SILENT-OOM** — the groom tool can exit without a word when memory is short.
- [ ] **GROOM-DRAG-UNTESTED** — the groom tool's drags have no automated test.
- [ ] **"mode `U` (VCM) still uses a dense cell grid with the same `(int)(nCells + 1)` overflow"** —
  check what 0.381.7's cell cap left of it.
- [ ] **"The 12 render modes are 5 estimators wearing 12 letters"** — thin aliases over five entry
  points, and close the U/J gap that costs image quality.
- [ ] **"Checked-in scenes cite ~22 derivation scripts that live in gitignored `scraps/`"** — a fresh
  clone has none of them (the output cleaner keeps them locally, nothing more).
- [ ] **"a 'render every scene at a uniform budget' sweep cannot distinguish correct from broken"**
  — needs stored per-scene references.
- [ ] **"`isomesh.h` caps a *rotated* container against its world AABB"**.
- [ ] **"`fieldLeafSDF` computes `c.r` ... unconditionally"** (2-4 % of `patternEval`).
- [ ] **"loom's `Isosurface` cannot emit a CSG field tree"**.
- [ ] **"`grid:<name>(…)` / `scatter:<name>(…)` are *surface-pattern* samplers only"**.
- [ ] **"`fur` cannot grow on a `mesh_instance`"**, and a huge groom peaks at 2x its final memory.
- [ ] **"`CurveSeg` is 80 bytes"** — a real groom is ~1.3 GB of segments.
- [ ] **"the loom viewer's prebake cache costs ~6.4 MB per frame"**.
- [ ] **"Fast RGB backward (`-rgb`) omits media + textured/record albedo"**.
- [ ] **"GPU BDPT (mode D) can't texture"**.
- [ ] **"GPU preview rasterizer — feature parity with the CPU rasterizer"** — what that entry lists
  as left.
- [ ] **"loom preview server (`-serve`) is resident-process only"**.
- [ ] **"HIP alias block deliberately omits `__shfl_sync`"** (HIP is untested anyway).

### 4.5 Performance

- [x] **BLUR-EPOCH-COST** (0.434.12) — a blurred clock epoch of mode M or J rebuilt its whole
  light side every 4 samples; it draws what its light side can afford against its gather now, the
  caustic neighbour count scaled with it.
- [ ] **J-FIELD-SLOW** — mode J in an unbounded density field: `probeGatherCount`'s random chords
  of the beams' box gather 32 beams where the camera's segments gather 10^4-10^5, so the knee
  overshoots and the `-beamk` floor fattens the beams. The GPU wavefront no longer spills on it
  (0.434.33 shrinking waves, 0.434.34 resume passes); the walk is still one thread a segment.
- [ ] **JSPEC-CARRIER-SIZE** — every mode-J GPU thread reserves a 120-byte spectral carrier per
  segment and per eye vertex, chromatic scene or not.
- [ ] **RADCACHE-BATCHES** — under `-radcache` a fixed budget's CPU batches follow its samples, not
  the clock: the periodic writes wait, and a resume reschedules. (0.434.27: `-stop` lands
  mid-batch, and a `-time`/`-noise`/`-forever` budget takes the clock instead of hanging.)
- [ ] **BEAMEPOCHS-CHUNK-PIN** — `-beamepochs` pins the GPU chunks to 1 spp, which costs a cheap
  scene.

- [ ] **FIRE-NEE-STRATA-COST** — one fire light sample costs up to 32 traversals.
- [ ] **Fire births** (EMISSIVE-VOLUME-SILENT's efficiency debt) — drawn uniformly over the box;
  drawing them from mode R's emission grid (`EmissiveVolume::cellW`) makes beta nearly constant.
- [ ] **"mode `D` has no resident multi-camera session"** — a flyby re-uploads the scene per frame.
- [ ] **"the beam-map build is host-only and single-threaded"**.
- [ ] **"a mesh-bound medium's voxelisation is 86 % of `gallery_rain`'s load"** — its triangle pass
  is serial.
- [ ] **"`-mode W` inherits mode R's `-spp 256` default"**, though deterministic at 1.
- [ ] **"a GPU render can be starved to a standstill by ANOTHER process"** (and its RECURRENCE entry,
  which has the detector) — no way to yield today.
- [ ] **"on `gallery_rain` the aimed caustic pass cannot concentrate"**.
- [ ] **"the preview re-marshals every texel of every texture, every frame"**.
- [ ] **"gallery GPU configs are now dominated by the ~8.4 s fixed CPU-side setup"**.
- [ ] **"preview rasterizer much slower than an equivalent WebGL renderer"** — profile first.
- [ ] **"`-export-mesh` QEM decimation is pathologically slow"** on huge or self-intersecting meshes.
- [ ] **"scene loading is down 5×, but the graph walk ... is what's left"** — only if load shows up
  again.

### 4.6 Missing features and scoped follow-ups

- [ ] **CURVE-CAUSTIC** — modes M and S miss a point or spot light's caustic on a `type diffuse`
  strand (0.433.3 stopped depositing there); the fix is a strand-shaped estimate on the caustic map.
- [ ] **DEFORM-LIMITS** — a dynamic elastic must be first in its mesh's stack.
- [ ] **MOTION-LIGHT-RIGID** — a spot, beam or aimed light its keys stretch unevenly, and a sun or
  sky in a moving group, stay at the instant.
- [ ] **PER-OBJECT-MODES** — emissive surfaces and bare mesh instances cannot be tagged; only D and
  J route (U and the unidirectional modes are separate renderers).

- [ ] **An emissive `quad` / isosurface / CSG solid glows but does not illuminate** (two entries:
  "KNOWN LIMITATION" and "LIMITATION (2026-08-02)") — register emitters for them. Waits on the
  GLOW-LIGHT-MODES decision (4.1).
- [ ] **Mode S in participating media** — refused since 0.384.2 (SPPM-LIGHT-SIDE); a volume
  estimator (photon beams, as mode M's `-beams`) would lift it.
- [ ] **"the live window cannot repaint mid-image"**.
- [ ] **"cache ftrace's per-scene preprocessing before rasterizing"** (feature request).
- [ ] **"option for curve-editor curve to be occluded by geometry"** (feature request).
- [ ] **"FBX import (C8) consumes geometry only"**.
- [ ] **"glTF/GLB loader is a static-geometry subset"** — what that entry lists as left.
- [ ] **"`.nvdb` / `.vdb` volume import: dense bake, float-only"**.
- [ ] **"Fisheye/panoramic lenses"** — unsupported by mode D and the finite-lens modes A/C.
- [ ] **"Absolute-EV film sensitivity, non-square films, shared multi-camera pass"** — its open items.
- [ ] **"Texturing is base-color only"** — only the `ior` map is left.
- [ ] **"Light shapes"** — spot penumbra CDF sampling is left.
- [ ] **"POV-Ray pattern/pigment/spline internal functions not ported"** — the remainder.
- [ ] **"the ~11 % of cloud beams that a dispersive event disqualifies"** — check first: 0.257.0's
  per-member bundle weights may be this (then it is 5.7).

### 4.7 Bookkeeping: entries whose heading still says OPEN for finished work

Verify each against the code, then mark it DONE (or retitle it) in `known-issues.md`.

- [ ] Both **`camera_curve` frame-range** entries in "Open bugs" — `-frames A B` shipped in 0.337.0.
- [ ] **"`-savemap` / `-loadmap` are silently ignored on the CPU"** — `photonmap_io.h` serves both.
- [ ] **"an emissive mesh that is not PLANAR is silently re-oriented OUTWARD"** — its own body says
  FIXED (0.338.0, `emit_orient`).
- [ ] **VOLCACHE** (2026-09-06) — the deposit split shipped in 0.300.0.
- [ ] **"a mesh area light is sampled UNIFORMLY BY AREA"** — the penalty no longer reproduces; the
  emitter-level tree shipped in 0.187.0.
- [ ] **BVH-BUILD** — its heading says not worth parallelising; its body says done in 0.292.0.
- [ ] **UPBP-CONV** (claims overturned), **FURDIM** (dissolved), and the **PLANNED UPBP** entry (first
  half done in 0.214.0 — check what is left of the second).
- [ ] **"the `gallery_rain` 600-frame mode-M flyby cannot start — the machine's COMMIT limit"** —
  re-test on the frame-at-a-time GPU path; it may be closed.

### 4.8 Not listed, on purpose

Notes and by-design entries (the GitHub `<video>` NOTE, the p99 anchor BY-DESIGN, `SetWindowTextW`,
`-gi-clamp`, the CIE side-table "don't retry", the timed-out-shell NOTE), work deliberately not
done (`PatOp::MatMulAdd`, GPU marching cubes, sweeps as an FTSL primitive), the environment (Nsight's
admin-locked counters, the one-off `build.bat` failure, the nvlddmkm teardown BSOD — a driver bug,
mitigated), and data whose entries call it acceptable (the F-series SPDs, the material presets).

## 5. The paired-timing rule into `CLAUDE.md`

"Time paired within a repetition, warm-up discarded" is a standing measurement rule that lives only
in session context and in this file's last section. It belongs in `CLAUDE.md`. (The old block —
the user's uncommitted changes there — no longer applies: `CLAUDE.md` is local-only and git-ignored.
It is still the user's file; ask first.)

---

## Standing constraints (so a fresh session does not have to be told)

- Bump `VERSION` in the same commit as any observable change; no rebuild → no bump. Commit freely,
  **never `git push`**.
- Every render passes `-window-min` (never bare `-window`) and launches with
  `dangerouslyDisableSandbox`. Stop a render with `ftrace -stop <pid>`, **never** `taskkill /F`.
  Never blanket-kill a shared runtime by image name (python/node/dotnet/java) — only the exact PID.
- Scratch in `scraps/`, PNGs in `png/`; a flyby series gets its own `png/<setname>/`. What outlives
  its use there is pruned by `tools/clean_outputs.py` after every build (REFERENCE.md, *How long
  output is kept*).
- **A rig that is cited as evidence does not belong in git-ignored `scraps/`** — promote it to
  `tools/` and rewrite the references in the same commit, or the table is not re-derivable from a
  clean clone.
- `CLAUDE.md` is the user's (and local-only): change it only when asked.

## Measurement discipline (each of these cost something to learn)

- **Run the controls first, and let them veto the result.** A rig that cannot see the effect
  produces confident numbers that are pure artefact: a furnace built from `light area {}` renders a
  MIRROR as exactly 0.0000, and I nearly reported that blindness as a 4 % energy bug in a material.
  `tools/furnace_rig.py` prints its three known answers before it prints anything else.
- **Never measure reflection in an open scene** — use an enclosure. This is what made the
  "mode M renders glossy 2.7x darker" claim wrong, and it had the same tell (a mirror reading 0).
- **Run the baseline control before attributing a difference**, especially before concluding that
  something new is slow (see `TODO-completed.md` §6).
- Score per-ROI, never whole-frame; separate variance from systematic; hold confounds fixed by
  construction; time paired within a repetition with the warm-up discarded.
- When a quantity must be applied at N sites, **N > 1 is the bug** — build the funnel. The 0.323.0
  host/device slip was a four-site change made at three sites.
