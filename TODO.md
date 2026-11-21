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
Re-checked 2026-11-11: each of the 42 `## OPEN` entries is listed here, and no older-style `###`
entry has opened since.
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
5. ~~**CURVE-GATHER** (4.2) — modes M, S and U read a coat of diffuse fur at 0.7-2.4 of mode R; never
   gather at a curve vertex, as mode J does.~~ Modes M and S done, 0.433.3; mode U 0.439.2
   (VCM-THIN-CURVES).
6. ~~**VCOL-BDPT-GPU** (4.2) — GPU modes D and J connect to a vertex-coloured mesh without its colours.~~
   Done, 0.433.4.
7. ~~**MODEM-BEAM-CARRIER**~~ (0.433.5), ~~**JSPEC-GAPS**' support~~ (0.433.11) and
   ~~**MODEM-REDPASS**~~ (not reproduced) (4.2) — mode M's beams, and mode J's last camera-side
   factors, at the camera's wavelength. Done.
8. ~~**UPBP-BEAM-RADIUS**~~ (0.434.0) with ~~**J-FOG-REALCAM**~~ (4.3) — mode J's beam kernel shrinks
   per epoch now; J-FOG-REALCAM's CPU-only extra deficit was the CPU's fewer counted epochs
   (closed 2026-11-12: at equal epochs the devices agree).
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
19a. ~~**J-FIELD-SLOW** (4.5) — mode J in an unbounded density field (found in HAZE-BLUR-DOMAIN's
    check).~~ Done: the GPU hit queue no longer spills (0.434.33-36), 0.434.40's beam domain ended
    the outward ratchet of 0.434.31's beam limit (J-BEAM-FARLIMIT) and made the field fast, and
    0.435.0's `-jbeams` made a fixed 1000-beam map the default, and 0.436.8's camera probe
    (J-AUTO-PROBE) put `auto` on it in unbounded hazes.
19b. ~~**J-FIELD-SLOW's open ends** (4.5), found measuring it 2026-11-09/10: a `-time` budget cannot
    end a GPU mode D/J chunk.~~ Done, 0.436.6. (The 2 000-beam brightness it also found was the
    shared seeds' noise, 4.3; the resume passes' re-walk no longer costs anything measurable, 4.5.)
20. ~~**JSPEC-CARRIER-SIZE** (4.5) — every mode-J GPU thread reserves the spectral carrier.~~
    Deferred 2026-11-14: option (1) (decided that morning, the agent's pick) turned out to reach
    only all-grey scenes -- the carrier folds every coloured reflectance of the camera prefix -- so
    it is not worth the compile time; (2), half-precision bins, is the step if VRAM gets tight
    (4.1), and helps scenes with colour. (The ~3x camera pass measured against it was the MIS
    weights' transmittances, 0.436.5.)
21. ~~**RADCACHE-BATCHES** (4.5) — `-radcache`'s fixed batches outlast `-stop` and the interval.~~
    Done, 0.439.3: the cache's own schedule over absolute samples, a canonical merge, the chunks
    on the clock.
22. ~~**P-HEROSPLIT-FIRST**~~ (0.439.4), ~~**ROUGH-IRID-AIR**~~ (0.439.6) (4.3) — narrow estimator gaps.
23. ~~**REMAP0-IMPOSSIBLE** (4.4) — the MIS sums read every zero density as a delta's.~~ Done,
    0.439.7 (which also counts a glow at full weight in D and J, GLOW-LIGHT-MODES).
23b. ~~**BUMP-MODES** (4.2) — on a normal-mapped surface U reads 16 % under R, M 12 % over, D/J/B
    1-2 %: one inconsistency at a vertex below its shading hemisphere.~~ Done in three stages
    (D, J and U in 0.439.9; the photon modes in 0.439.10; glossy and sky 0.440.6, translucent
    0.440.8, the analytic mirror / glass-sphere splat 0.440.10).
23d. ~~**TRANSLUCENT-MS** (4.2) — modes M and S read a backlit translucent panel 10 % dark: their
    gathers drop the light it transmits.~~ Done, 0.440.1.
23e. ~~**GLOSSY-FLAT-DJU** (4.3) — D, J and U read a flat glossy room about 1 % bright.~~ Done,
    0.440.2: a light walk's glossy bounce now carries the lobe's adjoint.
23f. ~~**GLOSSY-ROOM-M** (4.3) — mode M reads a room of glossy walls 0.5 % bright on the host.~~
    Closed: twelve seeds read 1.0017 +- 0.0007, the 1.0050 was three high seeds.
23g. ~~**LAYERED-MODES** (4.2) — a `layered` material reads 1-9 % off mode R in the other modes, even
    flat: the coat's coin is taken at the camera's angle in R and the light's elsewhere.~~ Done,
    0.444.0: (b), the coat reciprocal in every mode -- the body takes it both ways, the coat's
    reflection its Fresnel term at the half vector (tools/check_layered_modes.py).
23i. ~~**GLOSSY-DEPTH-CAP** (4.3) — at `-max-bounce`, mode R's glossy NEE keeps only its MIS share:
    the last vertex's lobe-sampled emitter hit is never traced.~~ Done, 0.440.4 (the tail).
23j. ~~**THIN-GAP-GATHER** (4.2) — mode M at a forced large radius gathers a lit surface a few
    millimetres behind as the same surface (MERGE-PROXIMITY's flat allowance).~~ Done, 0.440.13:
    no allowance, and a slack for the device's float rounding.
23k. ~~**GLOSSY-ADJ-VARIANCE** (4.3) — the glossy lobe's adjoint is unbounded at grazing arrival.
    Decided 2026-11-14: a GGX microfacet lobe.~~ Done, 0.445.0: GGX with its multiple scattering.
23h. ~~**S-GPU-GLOSSY-BUMP** (4.3) — device mode S reads normal-mapped glossy walls 1.043 of R (host
    0.997); see BUMP-MODES.~~ Done, 0.440.6: R's own glossy strategies disagreed (contSoftening).
23l. ~~**B-ENV-WALLS** (4.2) — mode B reads walls lit by a sky about 1 % bright, flat or bumped.~~
    Done, 0.440.11: the sky it sees directly took one ray per pixel's centre.
23c. ~~**B-GPU-MAXBOUNCE** (4.2) — GPU mode B ignores `-max-bounce` (its kernels get a literal 32).~~
    Done, 0.439.9.
24. ~~**BEAMEPOCHS-CHUNK-PIN** (4.5) — `-beamepochs` pins cheap scenes' GPU chunks to 1 spp.~~ Done,
    0.438.2.
25. ~~**DEFORM-LIMITS**~~ (0.440.0), **MOTION-LIGHT-RIGID**, **PER-OBJECT-MODES** (4.6) — what is left of them.
26. ~~**BLACK-PIXELS-ONCE** (4.2) — one render's black pixels, never reproduced.~~ Closed:
    28 clean renders of its scene (scraps/_black_pixels.py).
27. ~~**SPPM-LOWPASS-DEVHOST** (4.2) — mode S's device and host differ at low pass counts.~~
    Closed: noise in one high-variance tile (16 seeds a backend agree).
28. ~~**HAZE-BLUR-INVERSE-TR** (4.3) — a blurred mover far out in a density-field haze: modes D and
    J weigh its sun subpaths by a quadrature, not an unbiased estimate.~~ Done, 0.436.4.

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
- [x] **COMPOTE-NO-CAUSTIC** — decided 2026-11-14: accepted. Under the 20° sun the gem compote's
  ruby bowl absorbs the light that made its caustic under the old 45° sun; the compote stays a
  piece to look at, not a caustic source.
- [x] **LAYERED-MODES** — decided 2026-11-14: (b). A `layered` material's coin is taken at the
  camera's angle in mode R and the light's elsewhere, so the modes read it 1-9 % apart. (a) Keep R's
  integrand and re-weight every light-side walk; (b) make the coat physical, (1 - R(in)) (1 - R(out))
  on every body scattering in every mode: reciprocal, and a layered body renders about 8 % darker
  at normal incidence for ior 1.5.
- [x] **GLOSSY-ADJ-VARIANCE** — decided 2026-11-14: a GGX microfacet lobe. The glossy lobe divides
  by the light side's cosine, so its adjoint is unbounded at grazing arrival; a reciprocal lobe has
  no such tail, and changes how every glossy material renders.
- [x] **JSPEC-CARRIER-SIZE** — decided 2026-11-14: (1), a template parameter that sizes mode J's
  device spectral carrier to nothing where it is off (achromatic scene, `FTRACE_JSPEC=0`). Kept for
  if VRAM gets tight: (2), the carrier's bins as `__half` normalised by their maximum (48 B instead
  of 96). The two combine -- (1) removes the carrier where it is off, (2) halves it where it is on,
  which is the only help for a chromatic scene like `gallery_rain` -- and (2) costs precision
  (about 5e-4 a bin, to be measured on `_lay_chroma` and `gallery_rain` against the float carrier
  before it replaces it) and a conversion at every read and write.
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
- [x] **GLOW-LIGHT-MODES** — decided 2026-11-14 and done in 0.442.0: every emissive surface has an
  emitter (an isosurface's sampled by the lines through its box), `glow_only on` keeps one a glow
  that lights nothing, the same way in every mode. Owed: the user's choice of `gallery_rain`'s grid
  and stands (`gridground`, `wirecage`: lit, or `glow_only`), from a frame rendered each way. A glow with no emitter lights nothing in
  R/M/S/B, all of itself in D/J/U; mode P runs away on it (x368).

### 4.2 Wrong, missing or silent output

- [x] **BUMP-MODES** (0.440.10) — on a strongly normal-mapped room mode U read 0.844 of R, M 1.148,
  D/J/B 1-2 % off: a vertex below its shading hemisphere was lit in R and not consistently elsewhere.
  Diffuse (0.439.9, 0.439.10), glossy and sky (0.440.2 - 0.440.6), translucent (0.440.8), the mirror /
  glass-sphere splat's camera side (0.440.10).
- [x] **TRANSLUCENT-MS** (0.440.1) — modes M and S read a backlit translucent panel 10 % dark (the
  gathers dropped its transmitted light).
- [x] **GLOSSY-FLAT-DJU** (0.440.2) — modes D, J and U read a flat glossy room about 1 % bright (a
  light walk's glossy bounce took r, not the adjoint's r cos(out)/cos(in)).
- [x] **GLOSSY-ROOM-M** — mode M reads a room of glossy walls 0.5 % bright on the host. Closed: no
  bias beyond 0.2 % (twelve seeds, 1.0017 +- 0.0007).
- [x] **LAYERED-MODES** (0.444.0) — a `layered` material read 1-9 % off mode R elsewhere, even flat
  (the coat's coin was non-reciprocal): the coat is reciprocal now, (b).
- [x] **S-GPU-GLOSSY-BUMP** (0.440.6) — device mode S read normal-mapped glossy walls 1.043 of R:
  R's own lobe and light sampling weighed different integrands (contSoftening).
- [x] **B-ENV-WALLS** (0.440.11) — mode B read walls lit by a sky about 1 % bright: the sky the
  forward modes see directly took one ray through each pixel's centre.
- [x] **THIN-GAP-GATHER** (0.440.13) — mode M at a forced large radius gathered a lit surface
  millimetres behind: the gather band's flat allowance is gone.
- [x] **GLOSSY-ADJ-VARIANCE** (0.445.0) — the glossy lobe's adjoint was unbounded at grazing
  arrival: the lobe is GGX with its multiple scattering (reciprocal; roughness keeps its width).
- [x] **GLOSSY-DEPTH-CAP** (0.440.4) — at the bounce cap mode R's glossy NEE kept only its MIS share;
  R now traces the ray its last vertex samples, for the light it reaches.
- [x] **B-GPU-MAXBOUNCE** (0.439.9) — GPU mode B ignored `-max-bounce` (a literal 32 reached its
  kernels).

- [x] **REVIEW-0444-FABLE** (0.444.2) -- a mapped thin film's T-bar at the hit's thickness; a coated
  fluorescent body normalised at its re-emitted wavelength in mode R; device mode W's layered weight;
  manual coats uncoated; the coat factor prepared once per pair (CoatEval) and one continuation helper.
- [x] **REVIEW-0442-FABLE** (0.442.4, 0.443.0) -- a hit finds its emitter by its primitive (`Hit::emitter`),
  not its material: no glow material copies (their solid identity, the instancing refusal, the
  retag, the degenerate group, the spectrum wrap, the load cost); the docs' "one light per material".
- [x] **GLOW-MIS-J-HOST** — closed 0.443.2: two high seeds of a noisy estimator (eleven seeds:
  host J 1.0009 +- 0.0023 of mode D).
- [x] **KEEPWINDOW-VRAM** (0.442.3) — a finished render held open by `-keepwindow` kept all its device
  memory, so the next GPU render paged (~45x slower); it gives the device back as the render ends.
- [x] **J-THIN-GAP-LEAK** (0.445.3) — J's and U's merges keep their own surface's photons over its
  share of the disc (the footprint): the gap's light stays off the plate, and object bases read true.
- [x] **REVIEW-0445B** (0.446.1) — the Opus (max) and Fable (xhigh) reviews of 0.445.4-0.445.9:
  where walks start (tube lights, `-no-env`, open sheets, mixes, strands, the device's cap), the
  legs left unabsorbed (mode C, M's glossy NEE, `-rgb`, a glow seen, a camera in a glass ball,
  `-radcache`, `-herosplit`), flat mirror planes; known-issues REVIEW-0445B.
- [x] **REVIEW-0448C** (0.448.5) — the Opus (xhigh) review of 0.448.4; known-issues REVIEW-0448C.
- [x] **REVIEW-0448B** (0.448.4) — the Opus (max) and Fable (xhigh) reviews of 0.448.1-0.448.3;
  known-issues REVIEW-0448B.
- [x] **REVIEW-0448** (0.448.1) — the Opus (max) and Fable (xhigh) reviews of 0.447.2-0.448.0;
  known-issues REVIEW-0448.
- [x] **BALL-EYE-HALO** (0.448.0) — mode B's glass-sphere connection draws every chain of surface
  events (reflections outside and inside, to 16; 128 since 0.448.1), its rim and its folds
  (known-issues).
- [x] **GPU-SPHERE-CONNECT-SLOW** (0.450.42-0.450.44) — the connection closes on its roots and splats after its
  scan, and the connections are queued into a kernel of their own: 17.5 -> 2.9 s on the outside view at
  5e6 photons (the CPU's 32 threads 11.7); known-issues.
- [ ] **SPH-SCAN-FP32** — the glass-sphere connection's scan in fp32 and its roots refined in fp64 (the
  FP64 pipe is what bounds kSphConnect): time it on the outside view, and hold check_ball_images and
  check_sphere_queue to it.
- [x] **MODEB-SPHERE-ROOM** (0.448.1) — the check's depth cap, which mode R counts the ball's events
  against and mode B's connection did not (known-issues).
- [x] **MODEM-BALL-LAMP** (no change) — the lamp's edge pixels' coverage noise at 64 spp, the same
  two seeds' camera samples in every run (known-issues).
- [x] **GPU-TIR-DRIFT** (0.448.2) — on the GPU a ray held inside a glass ball by total internal
  reflection drifted out, its origin pushed inward at each reflection (known-issues).
- [x] **MODEB-DEPTH-CONNECT** (0.448.3) — mode B's specular connections count their vertices against
  -max-bounce (known-issues).
- [x] **REVIEW-0447R** (0.447.2) — the Opus (xhigh) review of 0.447.1's fixes; known-issues REVIEW-0447R.
- [x] **LEG-TRANSMITTANCE** — one call for a leg's whole transmittance (media and its solid's
  absorption) at every connection site, host and device, and a walk's starting stack a required
  argument rather than GiCtx::stk0's pointer (REVIEW-0445B's last two findings).
- [x] **REVIEW-0446** (0.447.1) — the Opus (max) and Fable (xhigh) reviews of 0.446.0-0.447.0;
  known-issues REVIEW-0446.
- [x] **EMIT-TWO-SIDED** (0.447.0) — `emit_two_sided on` for emissive materials (a glowing quad or mesh sheet
  lit both ways), the back face decided by material at the hit's finalizer rather than by triangle,
  so instances take it too; known-issues "a quad's emission is invisible from the side its winding
  faces away from".
- [x] **COINCIDENT-LAMP-FACES** (0.446.0) — `light area { two_sided on }`, and a load-time warning
  for area lights overlapping on one plane; `tools/check_two_sided.py`.
- [ ] **GAP-MERGE-RADIUS** — beside a millimetre gap's mouth a merge's disc averages in the gap's
  trapped light at the default radius (J 2-3.6x mode B): the kernel's blur across a 1000x step, a
  bias that converges as E^(-1/4). The remedy, an adaptive merge radius, was put to the user with its
  downsides and recommended against (known-issues GAP-MERGE-RADIUS): left until the user wants it.
- [x] **MODEM-GPU-CORNER** (no change) — the corner and `flat`'s red residual were noise: at new
  seeds GPU over CPU mode M reads 1.0001 in red (known-issues).
- [x] **CURVE-GATHER**, mode U's half (VCM-THIN-CURVES) — 0.439.2: mode U merges at no curve
  vertex (`vcmMergeAt` / `dVcmMergeAt`); its coat 2.10 / 2.36 -> 1.000 / 1.014 of mode R.
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
- [x] **MODEM-CPU-DARKGLASS-BLUE** (no change) — seeds 1 and 2's noise in mode M and in the mode R
  reference alike (known-issues); check_dark_glass judges six seeds now.
- [x] **MODEM-COMPOTE-NAN** (0.433.1) — the spectral carrier's ratio overflowed a float in the
  device's gather (and mode S's per-pixel copy); the float side now takes ratio / peak.
- [x] **VCOL-BDPT-GPU** (0.433.4) — `DVertex` and `DVcmLV` carry the hit's vertex colour; GPU
  modes D, J and U had read a vertex-coloured sphere near grey.
- [ ] **GPU-PHOTON-LAUNCH-FAIL** — the forward photon kernel died with "unspecified launch failure"
  three times (modes M and B, 0.433.10, 0.434.8, 0.450.0), each beside another process's GPU
  renders; reruns alone clean, memcheck clean. Reproduce under the same load, then memcheck it there.
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
- [x] **LAYERED-BODY-COIN** (0.444.0) — a forward mode weighed a coated body by the photon's coin, at
  the light's angle (5 % on the car paint): closed by LAYERED-MODES (mode B 0.987-1.000 of R now).
- [x] **FWD-SMOOTH-COAT** (0.445.4) — mode B's mirror connections take a smooth coat as a mirror of
  its own reflectance R(theta): the smooth-coat rooms read 0.997-0.999 of R (0.90 before).
- [x] **GLTF-MAP-MEAN-GUTTERS** — the importer's mean metalness and roughness count the atlas's
  gutters.
- [x] **FEATHER-IS-A-THRESHOLD** (0.434.17) — a mesh-bound medium's `feather` shrinks the silhouette
  instead of ramping the density (`tools/check_feather.py`).
- [x] **BLACK-PIXELS-ONCE** (closed 2026-11-13) — one GPU mode-R render's black pixels: not
  reproduced in 24 more runs (28 in all); reopen if seen, with the image.
- [ ] **D-SILENT-EXIT** — one host mode-D render ended with no output and no image during a CUDA
  build; 13 reruns clean. The checks now print the exit code; reproduce under memory pressure.
- [x] **S-IRID-SPHERES** (0.445.1) — no bias: the studies scored two seeds through a `stats()` that
  read only its first two images; over every seed S holds R within 2 sigma. check_rough_irid's
  `stats()` takes every seed, and mode S runs eight.
- [x] **POV-FMOD-INT** (0.445.2) — `f_hex_x` / `f_hex_y` were flat on the GPU (MSVC's host-only
  `fmod<double, int>` in device code); `3.0` takes the double overload on both.
- [x] **SPECGATHER-BIN-RESOLUTION** (0.451.0) — mode M's and mode S's carrier bins split where the scene's
  tables step, bend or leave zero (SpecCells); the ramp's excess was its bend inside bin 11; known-issues.
- [x] **SPPM-SPECTRAL** (0.432.0) — mode S's visible point carries its walk's spectral factors per
  bin (SpecThr), and each gathered photon is weighted at its own wavelength.
- [x] **BIRTH-EMPTY-STACK** (0.445.5) — every light walk starts inside the solids enclosing its
  birth (Scene::birthStack, dBirthStack); `tools/check_birth_stack.py`.
- [x] **SPPM-GPU-LAMP** (no change) — one fluke at seeds 1-8, measured again at the same seeds:
  1.006 +- 0.008 over 16 GPU seeds (known-issues).
- [x] **GPU-U-GLOSSY-SLOW** (0.450.0) — GPU mode U's merges are queued for a warp each
  (VCM-WAVEFRONT): at -pmradius 0.05 62.5 -> 7.4 s (the CPU 9.5), at 640x480 204 -> 50 s;
  `tools/check_vcm_wavefront.py`.
- [x] **VCM-MERGE-LOCALITY** (0.450.1) — the merges read the light vertices in cell order, a copy
  in the light slab's memory (`DVcmGrid::byCell`); bit-identical. On an idle GPU it moved the
  640x480 gather little (44.6 -> 43.8 s), the gather being bound elsewhere (VCM-GATHER-SPILLS).
- [x] **REVIEW-0450** (0.450.1) — the Opus (max) review of 0.450.0: the merge's coat read at the
  walk's hit, waves sized by the last pass's per-block profile (check_vcm_wavefront's sky case), the
  grid's worst kept free beside the queue, the shared helpers; known-issues REVIEW-0450.
- [x] **VCM-GATHER-SPILLS** (0.450.5) — a merge site's lobe (DMergeLobe): the hit and roughness once a
  site, one GGX pair a light direction for its value and both pdfs, the reflectance alone a
  wavelength; byte for byte 0.450.4's images. 640x480 bowl: gather 43.4 -> 15.7 s; Nsight 10.5 -> 1.5
  G local requests, 26.6 -> 15.6 G FP64. Its four pattern readers forced inline, which made mode D
  1.75x faster too (0.450.4 had dGlossyF out of line at 168 registers).
- [x] **REVIEW-0450H** (0.450.18) — the Opus (max) and Fable (xhigh) reviews of 0.450.17: the wavefront
  pool by the batch traced and one list, mode J's plain merge kernel kept, the noise tables in ggx.h's
  layout, the library note's reach and cost; known-issues REVIEW-0450H.
- [x] **REVIEW-0450G** (0.450.17) — the Opus (max) and Fable (xhigh) reviews of 0.450.15-0.450.16:
  `FT_UNIT_INLINE` keyed by the unit again (separate compilation forbids `__CUDA_ARCH__` there),
  renderBdptCuda launching through the route's own kernel pointer, mode J's fallback weighed, the
  wavefront pool counted, the lattice's float rounding, and the gates' and messages' slips;
  known-issues REVIEW-0450G.
- [x] **PREFLIGHT-ROUTE-PLAN** (0.450.16) — the preflight weighs routes, each the kernels its launcher
  will run from the launcher's own decision (render_cuda.h `GpuRoute`); mode P's camera-side layers
  and `-rgb` weighed as they run; known-issues.
- [x] **VCM-SPARSE-LATTICE** (0.450.19) — modes U and J bin their merges in a hashed lattice sized
  from the point count (merge_grid.h), on both devices; the CPU's merges bit-identical to the dense
  lattice's; tools/check_merge_table.py.
- [x] **PM-BUCKET-TWICE** (0.450.21) — mode M's and SPPM's gathers could read a bucket twice when two
  of their cells shared it: the photon map takes merge_grid.h's bucket, host and device.
- [x] **GATHER-BASE-MIRROR-M** — the check's CPU mode M ring was noise at 4 samples: they take 16.
- [x] **MIRROR-BASE-GPU** — not the renderer: the check's mirror ring held the pixels straddling the
  box top's edge (the lamp in the mirror, 14x the floor), whose 16-sample noise 8 seeds did not
  average away; the box rings leave them out (`box_floor`), and the devices agree to 0.1 %.
- [x] **PM-BUILD-NOINIT** (0.450.28) — the photon map's bulk arrays are `ftalloc::BulkVec`s, whose
  resize constructs nothing; known-issues. The entry as logged: the photon map's arrays (`pos`, `photons`, `cie`, the binning's
  destination) are `std::vector`s of types whose members zero themselves, so each resize zero-fills
  on one thread what the next pass overwrites: 0.8 s for the first binning's 7 GB destination and
  0.3 s for the CIE table at 129 M photons. An allocator that leaves elements unconstructed (the
  storage written before it is read) lets the parallel passes take the pages themselves.
- [x] **PM-PROBE-GEOMETRY** (0.450.27) — the radius probe samples the cells the geometry picks (the
  lowest-hashed 4096), not the table's buckets, on every core; check_merge_table renders mode M
  unpinned; known-issues. The entry as logged: mode M's automatic radius samples the map by occupied BUCKET
  (`medianNeighborCount`), so every change to the table's size or hash moves it and every unpinned
  mode M render with it (0.450.23 and 0.450.24 did; `FTRACE_MERGE_TABLE` does, so
  check_merge_table pins the radius). Sample by CELL instead, with a set the geometry alone
  decides -- the cells whose coordinates hash lowest (a bottom-k sample), each represented by its
  lexicographically smallest photon -- which keeps the order-independence `-loadmap` needs and
  drops the table from the radius (the review of 0.450.22-0.450.24).
- [x] **GPU-MERGE-FLOAT-LATTICE** (0.450.22) — the device's merge maps carry their lattice in double
  (`MergeLattice`), as their points were binned; the device probe holds it.
- [x] **REVIEW-0450F** (0.450.15) — the Opus (max) and Fable (xhigh) reviews of 0.450.12-0.450.14, and
  0.450.14's mode-U slowdown (+18 %) found by timing and bisected: one albedo slot, the merge's own
  lobe; the scanner's levels and headers, the gates in CMake, FT_UNIT_INLINE per device pass in
  twelve headers and the POV generators, four more unit-local structs, the lattice shared, the
  library hint everywhere; known-issues REVIEW-0450F.
- [x] **REVIEW-0450E** (0.450.14) — the Opus (max) and Fable (xhigh) reviews of 0.450.10-0.450.11: the
  last unsequenced draws (selW's among them) and a scanner that sees them, modes V and P and the
  meter weighed by the kernels they launch, a flight's posed cameras, one reservation rule, the
  linkage check over every GPU object and twenty more structs unit-local, mode U's lobes shared;
  known-issues REVIEW-0450E.
- [x] **REVIEW-0450D** (0.450.10) — the Opus (max) and Fable (xhigh) reviews of 0.450.7-0.450.9: the
  split's linkage rule (static helpers, tools/check_device_linkage.py in build.bat), the preflight for
  shared flights and the meter, modes A/B/C weighed, mode S charged its session, the M/S deposits and
  mode U's store and cell table kept off the stacks' memory, a composed slot clamped per factor;
  known-issues REVIEW-0450D.
- [x] **REVIEW-0450C** (0.450.9) — the Opus (max) and Fable (xhigh) reviews of 0.450.5-0.450.6: a
  spectral property refused as a scalar map, a slot's pattern composed with its map, tryDevCap's
  growth, the cell table and the store kept off the sort's and the stacks' memory, FTSL.md's map
  rule, stale words; known-issues REVIEW-0450C.
- [x] **REVIEW-0450B** (0.450.6) — the Opus (max) and Fable (xhigh) reviews of 0.450.1-0.450.4: mode U's VRAM preflight,
  the beam seed's draws sequenced, the checks' GPU and history fixes, an empty site skipped first;
  DVertHitState stays 16 B and three of Fable's camera-kernel edits were declined, each breaking the
  device link (GPU-LINK-FRAGILITY); known-issues REVIEW-0450B.
- [x] **VRAM-PREFLIGHT-MS** (0.450.7 mode S, 0.450.8 mode M) — the preflight is a free function
  (`vramPreflight`: the mode, the render's device buffers, the batch's held scene). Mode M asks it
  once in run() for its largest camera's films (`mGpuOk`), and the seven sites that route a mode-M
  camera to the device (the meter, the budget and blur tests, two notes, the shared path) read it.
- [x] **GPU-LINK-FRAGILITY** (0.450.9) — kBdptT (`render_cuda_bdpt.cu`) and kJSurfLightT
  (`render_cuda_jsurf.cu`) compile in units of their own under `-maxrregcount` at their launch
  bounds (168, 128), the device code they share in `render_cuda_dev.cuh` / `render_cuda_dev2.cuh`
  (design.md, Build & release). Under `-rdc` ptxas compiles an out-of-line helper to its own count,
  not its callers' bounds, and an edit anywhere in the one unit -- a host-only one at the end --
  could put one out at 238; now every function in a capped unit fits it (cuobjdump: none over 168).
  Whole-program compilation also linked, but in 77 minutes of ptxas, and no faster. Next: land the
  declined edits (known-issues REVIEW-0450B) and the two below.
- [x] **MERGE-LOBE-SHARED** (0.450.11) — DVertexLobe is the one connectible-BSDF implementation:
  dBsdfF and dBsdfPdf a lobe for one direction, the hair case in it, the glossy reflectance shared
  with dGlossyF (dGlossyR), the hit read at its first need; mode U's camera kernel keeps a camera
  lobe for NEE and every connection, and one a light vertex for its two pdfs (design.md).
- [x] **DVERTHITSTATE-8B** (0.450.32) — the deform barycentrics restored from the vertex's own `vb0q/vb1q`:
  DVertHitState 8 B, DVertex 152 B; no speed moved (register-bound), 35-211 MB of reserved local
  memory freed; known-issues.
- [x] **MERGE-LAZY-TIMING** (no change) — eager against lazy at HEAD, paired: 0.998-1.004 on every mode J
  case; the lazy lobe stays; known-issues.
- [x] **MERGE-HASH-TIMING** (0.450.24) — the hashed merge lattice paired against the dense one on an
  idle machine: no dense case slower by more than GPU U's 1.3 %, the sparse box 5.5x faster on the
  CPU; on the way, modes U and J tag each point's block (the strays out of the counts), exact
  slots, the key forced inline and the radius probe counted in a register; known-issues.
- [x] **PM-BUILD-PARALLEL** (0.450.26) — from a million points the host's binning runs on every core in
  the one-thread sort's order exactly, and the photon map's second binning reuses the first's
  destination; known-issues. The entry as logged: a large photon map is built on one thread: at 129 M photons (mode M,
  _caustic_box at -n 20M) buildAuto's two binnings, probe and CIE pass take about 6.5 s of an 11 s
  map phase whose tracing takes 3.5 s on 32 threads. Each binning zero-fills two fresh scatter
  buffers (7 GB, 0.8 s: Vec3's member initialisers), counts its keys (0.7 s) and scatters (0.75 s
  to 1.25 s) serially, and the second allocates all of it again. Make it parallel and keep its
  order -- keys computed in parallel, then a two-level counting sort (each thread's range split by
  the bucket's high bits, each part sorted on its own), which is the serial sort's order, so no
  render moves -- and let the second binning reuse the first's buffers unfilled. The same helps
  mode U's CPU passes, which bin their light vertices every pass.
- [x] **RNG-ARG-ORDER** (0.450.11) — every call that took two or more RNG draws as arguments (89,
  the renderers' camera, lens and hair draws, the layered body's, the caustic aim's twins, the
  self-tests') draws them first into a declaration list or a braced initializer, which C++
  sequences; hair::sampleWith draws a fiber's four (design.md, "The draws a render takes"). Every
  render's noise realization moved once.
- [x] **VCM-GATHER-DIVERGENCE** (0.450.33) — the queued gather's lanes test their entries, and the warp
  weighs the ones that pass 32 at a time (dSiteListWarp): 16.8 -> 28.5 active lanes, the 640x480
  bowl's gather 11.55 -> 8.49 s, its render 21 % faster; known-issues.
- [x] **REVIEW-0450N** (0.450.35-0.450.36) — the Opus (max) and Fable (xhigh) reviews of 0.450.31-0.450.33;
  known-issues REVIEW-0450N.
- [x] **J-SURF-GATHER-DIVERGENCE** (0.450.37) — mode J's queued surface gather tests a photon a lane and weighs
  32 at a time (dSiteListWarp): at five times the radius its gather 33 % faster, at its own unchanged
  (its sites hold a warp's worth of photons or fewer); known-issues.
- [x] **BUILD-LOCK** — build.bat now and then failed on a locked output (LNK1104 / C1083): the holder is
  LithicBackup's continuous backup copying the tree's fresh files; the local build.bat builds again on a
  held file (twice at most) and ends every failure on `[build] NOT UPDATED`; known-issues BUILD-LOCK.
- [x] **REVIEW-0451** (0.451.1) — the Opus (max) and Fable (xhigh) reviews of 0.451.0; known-issues REVIEW-0451.
- [x] **MODEB-PASSTHROUGH**, the direct legs (0.451.3) — modes A and B's camera legs cross gels and half
  mirrors (camLegThrough / dCamLegThrough); check_passthrough.
- [x] **REVIEW-0453** (0.451.7) — the Opus (max) and Fable (xhigh) reviews of 0.451.4-0.451.6: instanced gels,
  hide_camera past a gel, mode P's composite sky through fog, unbounded media in the sky pass and mode M,
  one traversal a blocked leg; known-issues REVIEW-0453.
- [x] **REVIEW-0452** (0.451.4) — the Opus (max) and Fable (xhigh) reviews of 0.451.2-0.451.3: mode P's split,
  the device's fiber legs, crossings charged to -max-bounce, the cumulative kCoinRare, `tile_verdict` live where
  either image is lit; known-issues REVIEW-0452.
- [x] **MODEB-PASSTHROUGH, the sky** (0.451.5) — the sky, sun and glows modes A and B see directly, through a gel
  or half mirror (skySeenThroughXYZ); known-issues.
- [x] **MODEB-PASSTHROUGH, compounds** (0.451.8) — a `mix` / `layered` surface whose child is a gel or half
  mirror, on the camera legs and in the sky pass (StraightCross); known-issues.
- [ ] **MODEB-PASSTHROUGH, the specular connections** — their legs (glass sphere, mirrors) through a gel or half
  mirror; known-issues.
- [x] **SKY-THROUGH-FOG** (0.451.5) — modes A and B showed the sky through a fog undimmed (1.000 where mode R
  and Beer-Lambert read 0.341); known-issues.
- [x] **M-BEAMS-ABSORBER** (0.451.6) — mode M `-beams` dropped a camera ray's extinction through a medium that
  stores no beams (a pure absorber); check_sky_fog; known-issues.
- [ ] **THINLENS-VIGNETTING** (needs the user's decision) — through a thin lens mode A records irradiance with
  natural vignetting, the other modes radiance: one camera equation for all, or none; known-issues.
- [x] **HOST-HALFMIRROR-ANALYTIC** (0.451.2) — a coin far rarer at some cell than at its peak turns mode M's
  and S's walk partial (`kCoinRare`): a half mirror falling to 1e-123 read half bright on the CPU; known-issues.
- [ ] **SPECGATHER-CELL-CENTROID** — read each carrier cell at the emission's centroid in it, not its middle
  (a line-lit scene behind a sloped factor reads ~20 % dark in modes M and S); known-issues.
- [ ] **JSPEC-BIN-RESOLUTION** — the beam gather's carrier and mode J's on the scene's cells too, with
  JSPEC-CARRIER-SIZE's half-precision bins; known-issues.
- [x] **REVIEW-0450R** (0.450.45) — the Opus (max) and Fable (xhigh) reviews of 0.450.43-0.450.44;
  known-issues REVIEW-0450R.
- [x] **REVIEW-0450Q** (0.450.43) — the Opus (max) and Fable (xhigh) reviews of 0.450.42; known-issues REVIEW-0450Q.
- [x] **REVIEW-0450P** (0.450.41) — the Opus (xhigh) review of 0.450.38-0.450.40; known-issues REVIEW-0450P.
- [x] **REVIEW-0450O** (0.450.40) — the Opus (max) and Fable (xhigh) reviews of 0.450.34-0.450.37;
  known-issues REVIEW-0450O.
- [x] **SHADOW-WARNINGS** (0.450.39) — 34 hidden names renamed or removed, 3 dead parameters, the narrowings
  explicit; hiding (C4456-C4459) and C4701 are errors; a full build prints only nvcc's stub's four
  warnings; known-issues.
- [x] **VCM-LOBE-FP32** (0.450.34) — the merges' lobe takes its GGX pair in the device's precision
  (DMergeLobe), connections keep double: the gather's FP64 pipe 69 -> 24 % busy, the 640x480 bowl's
  render 12 % faster, its images double's to 4e-5 a pixel; known-issues.
- [x] **REVIEW-0450-FABLE** (0.450.3) — the Fable (xhigh) review of 0.450.0: the queue sized to its
  smallest wave, mode U in the local-memory preflight, one warp-gather tail, the seed's draws
  sequenced, check_bitid's floor for GPU J and U; known-issues REVIEW-0450-FABLE.
- [x] **LAYERED-FILM-PATTERN** (0.450.2) — a layered coat's `film_thickness_map pattern:` was dropped,
  and a map naming neither a pattern nor a texture loaded; `tools/check_layered_film_map.py`.
- [x] **GPU-DVERTEX-HIT-STATE** (0.450.4) — a GPU connection's rebuilt hit carries the walk's
  time and cavity (DVertHitState): patterns on movers stay put, the cavity is probed once;
  `tools/check_vertex_hit_state.py`.
- [x] **CAMERA-EMPTY-STACK** (0.445.8) — every camera walk starts inside the solids around the eye,
  and every connection leg takes its medium's absorption; `tools/check_camera_in_water.py`.
- [x] **REVIEW-0445** (0.445.6, 0.445.7) — the Opus (max) and Fable (xhigh) reviews of
  2613698..b004c3b: the J/U probe stops on glossy faces (check_merge_glossy), solid faces marked for
  J/U (check_merge_base's tower), roughness patterns reach 2 (check_rough_range), and the cleanups;
  known-issues REVIEW-0445. The lane-0 probes: WARP-PROBES (0.445.9), 2.0-2.2x -> 1.16-1.24x.
- [x] **REVIEW-0449** (0.449.2) — the Fable (xhigh) and Opus (xhigh) reviews of 0.448.6-0.449.1: the
  GRIN rule's gaps, BDPT big-primitive offsets, mode S's coverage argument, sessions' scale (known-issues).
- [x] **REVIEW-0448D** (0.449.1) — the Opus (max) review of 0.448.6-0.448.7: big primitives near the
  origin (check_big_primitive), per-primitive offsets, probe tmins, tool gates (known-issues).
- [x] **CPU-SMALL-IN-BIG** (0.449.6) — a host ray's offset and tmin come from the feature it leaves
  (Scene::hitEps over Hit::size; Ray::tmin carries it): the stubble ball 100x smaller on a 400 m
  ground reads x1.0001 of itself in mode W on the CPU, was x1.129 (check_scale_invariance; known-issues).
- [x] **REVIEW-0449B** (0.449.4) — the Opus (xhigh) review of 0.449.2-0.449.3 (known-issues).
- [x] **ROUTE-TAG-SOLIDS** (0.449.5) — a mode-D camera with objects tagged J merged without the
  solids' sides marked; check_route_tags had failed since 0.445.6 (known-issues).
- [x] **The 0.449 sweep's long checks** — check_ball_images, check_motion, check_rough_irid and
  check_two_sided pass on 0.449.6 (1.6, 1.6, 2.0 and 1.2 hours), run detached from the harness.
- [x] **GRIN-RADIANCE-SCALE** (0.449.0) — radiance over n² along a camera walk's march.
- [x] **GRIN-STRAIGHT-CONNECT** (0.449.0) — no straight connection crosses a bending region; mode R
  matches mode B outside the lens and mode C through it (check_grin_connect).
- [x] **GPU-TMIN-SCALE** (0.448.6) — a device trace's tmin is 32 ulps of its origin, not an absolute
  1e-4; fixed STUBBLE-DEVICE-UNDERSIDE (check_modew_fiber's stubble case).
- [x] **GPU-MINIATURE-SCALE** (0.448.7) — every absolute floor on the device scales with the scene
  (`c_lenScale`); the stubble ball 100x smaller renders at x1.0001 of itself (check_scale_invariance).
- [x] **CPU-MINIATURE-SCALE** (0.448.8) — the host's distances read `Scene::eps` (1e-6 x lenScale);
  the ball 100x smaller reads x1.0022 of itself on the CPU (check_scale_invariance).
- [x] **SPPM-LOWPASS-DEVHOST** (closed 2026-11-13) — mode S's device read check_coat_modes' coated
  balls 16 % bluer than its host at 16 passes: noise in one tile seen through the thin-film bubble;
  sixteen seeds a backend agree within 0.5 sigma.
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
- [x] **"a quad's emission is invisible from the side its winding faces away from"** — done in 0.447.0
  (EMIT-TWO-SIDED, `emit_two_sided on`; check_two_sided).
- [ ] **"`-dual-scatter` drops the coat's indirect illumination"** (its part 2).
- [x] **"`curv` reads 0 on an ISOSURFACE"** (0.449.8) — from the field's Hessian, where the material reads it.
- [x] **"`curv` reads 0 in the PREVIEW rasterizer"** (0.449.8) — per pixel from the vertex normals, both previews.
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
- [x] **"Forward modes render ~5% brighter than the backward reference (`R`)"** — re-measured
  2026-11-19: B / R x1.0001, M / R x1.0005; D's 0.2 % is its default 8-bounce depth (x1.0003 at 32).
- [ ] **"rainbow phase — `SpecVtx::term` ... is HG-only"**.
- [ ] **"`phase rainbow` — the 2048-bin uniform-in-mu table under-resolves large droplets"**, and
  monodisperse supernumeraries read as a white arc.
- [ ] **"forward mode B barely converges the water-droplet rainbow"**.
- [ ] **"interactive raster fly-viewer can peg all cores / grow RAM when orphaned"**.
- [x] **"native `-viewer` Meshes tab ignores a skin's `clamp`/`mirror` wrap mode"** (0.449.12) — a sampler per
  wrap/filter; built, still to be seen in a live viewer.
- [ ] **"Mode M renders Alice ... far from mode D, differently on each device"** — 20 M photons
  closed most of it (the GPU still ~9 % dark): re-measure, then fix or close.
- [ ] **"Residual: hue lost at true underflow"** — see-through on a very deep stack (minor).
- [ ] **"a `curve`'s azimuthal `v` is a PER-SEGMENT frame, not parallel-transported"**.
- [ ] **"an `sdf` over a self-overlapping mesh is exact outside but not inside"**.
- [x] **"`settle.drop()` falls back to the surface's TOP PLANE when the footprint 'misses'"** — the piece
  started under its stand; drop() lifts it clear first (check_settle_drop).
- [x] **"an `include`d file's own asset paths resolve through the ROOT scene"** (0.449.11) — beside the included file.
- [ ] **loom's reader does not parse `include`** — `tools/loom/loom/grammar/reader.py` parses its own typed
  `ftsl.epeg`, so a hand-authored scene with an `include` cannot round-trip through loom's editor.
- [x] **"`tools/_gemsweep.py` — `spread` is not resolution-stable"** — metered on a fixed-world cell
  (`GEMCELL`); the orb's caustic reads alike at 480 and 960 px.
- [x] **"`tools/_capchroma.py` scores marble VEINS as a caustic"** — meters over the frame's albedo
  and position passes; refuses a textured cap without them.
- [x] **"`python -m loom.viewer` prints a runpy double-import RuntimeWarning"** (cosmetic): the
  package re-exports the viewer and anim names lazily.

### 4.3 Estimator approximations and accuracy debt

- [x] **UPBP-BEAM-RADIUS** (0.434.0) — mode J's beam kernel shrinks per light-side epoch under
  -vcmalpha (_fog_cornell -0.70 % of mode D fixed, -0.43 % at 256 epochs, -0.07 % per sample).
- [x] **J-FOG-REALCAM**, what is left — CPU mode J read realcam's fog below GPU mode J: its counted
  epochs are a third of the GPU's, its kernel shrinks less; at equal epochs they agree (2026-11-12).
- [x] **UPBP-W-CHROMATIC** (0.434.19) — in a chromatic medium a beam merge's weight mixed two
  wavelengths and the frame tilted in colour (`tools/check_jchroma.py`).
- [x] **UPBP-W-RAINBOW** — the same mix in a rainbow-phase medium, through the camera side's phase
  densities: measured at +0.05-0.07 % of the frame, too small to fix.
- [x] **P-HEROSPLIT-FIRST** (0.439.4) — blurred mode P under `-herosplit` filed a sample by the hero's first
  event alone.
- [x] **ROUGH-IRID-AIR** (0.439.6) — a rough interference reflector always interfered against air.
- [x] **J-FIELD-SLOW's 2 000-beam brightness** — not a bias: with eight seeds an arm, disjoint
  across arms (`scraps/probe/biascheck.py`, 0.435.1, 1024 spp), `_fog_thick`'s mean against the
  mode D reference is 0.998 +- 0.018 at 250 beams, 0.990 +- 0.007 at 1000, 0.994 +- 0.013 at
  2000, 1.000 +- 0.011 at 4000, 0.990 +- 0.008 under the knee rule and 0.999 +- 0.002 with no
  beams. The 0.434.39 reading shared its seeds between arms.
- [x] **HAZE-BLUR-INVERSE-TR** (0.436.4: the Poisson estimator, `invTransmittancePoisson` and its
  device twin; `-checksunleg`) — HAZE-BLUR-DOMAIN's remaining approximation: where a blur's sun
  subpath begins further out than the NEE leg reaches (a mover outside the instant's disc), the
  first leg's factor is the INVERSE of a transmittance, exact in a homogeneous haze but taken by
  the deterministic quadrature (trDetMedium) in a density field, so biased by the quadrature's
  error. Known-issues calls it undrawable unbiased; it is not: the Poisson estimator,
  exp(tau) = E[e^lambda * prod_{i<=K} tau_i / lambda] with K ~ Poisson(lambda) and tau_i
  independent unbiased estimates of the optical depth (each >= 0 here, so the estimate stays
  positive), is unbiased, its variance moderate for lambda near tau. Host `sunLegFactor` and
  device `dSunLegFactor`; check_haze_domain's `farfield` is the test. Watch: on 0.434.41, six GPU
  seeds each, farfield's haze read mode D 0.9976 +- 0.0031 of R but mode J 0.9895 +- 0.0058 (the
  ball 0.999 and 0.997) -- within J's noise, but recheck J there with more seeds when this is
  worked.

- [ ] **BDPT-MIS-TR** — mode D's MIS weights take every connection edge's transmittance as 1 in a
  medium.
- [ ] **UPBP-W** — three approximations in mode J's merge weight (variance only).
- [x] **"BDPT connection edges through colored glass are not absorption-weighted"**, and mode U's
  connections (0.445.8, with CAMERA-EMPTY-STACK: every connection leg takes its medium's absorption).
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
- [x] **REMAP0-IMPOSSIBLE** (0.439.7) — the MIS sums read a zero density as a delta's, whatever
  made it zero (only the sun's was special-cased).

- [x] **GPU-STACK-DRIFT** (0.450.17) — `--split-compile` made the device build nondeterministic (mode D 11 %
  between two builds of one source); dropped: repeatable builds, R/D/U 12-24 % faster.
- [x] **J-NOSPLIT** (0.450.31) — not inlining: the L1. Every route kernel but mode D's prefers the
  largest L1 (`preferL1`): mode J through a medium 36 % faster, mode R 17 %, the rest unchanged; known-issues.
- [x] **REVIEW-0450M** (0.450.30) — the Opus (max) and Fable (xhigh) reviews of 0.450.29: `-t` read
  before the load and held by every threaded build, the CIE table in the scratch's storage, one
  joined releaser thread, BandPool exceptions rethrown, a test of every table entry; known-issues.
- [x] **REVIEW-0450L** (0.450.29) — the Opus (max) and Fable (xhigh) reviews of 8851601..0.450.28: the
  radius probe's upside-down heap and the binning scratch never freed (both this batch's), a test
  that holds the probe's sample itself, parallel.h's passes exception-safe and held to `-t`, the
  photon map's table unfilled; known-issues.
- [x] **REVIEW-0450K** (0.450.25) — the Opus (max) and Fable (xhigh) reviews of 0.450.22-0.450.24: mode
  U's budget counting its table and tags, SurfMap's accounting, the capped site count, a bin with no
  copy of its table, the host's 27 keys in the caller's array (a shared device tag filter measured
  and declined: GPU U 20 % slower); known-issues.
- [x] **REVIEW-0450I** (0.450.20) — the Opus (max) and Fable (xhigh) reviews of SLAB-DDA-COPIES: the
  volume caches' clips, a NaN-safe and monotone walk, `-checkcellwalk`, one fire pdf integral; known-issues.
- [x] **SLAB-DDA-COPIES** — the box clip and the grid walk were written out several times: one
  `cell_walk.h` for host and device (eight clips, six walks), renders byte-identical.
- [x] **GMIS-NULL-BY-HAND** (0.450.15) — one pass-through restore a walk: the scalar walks' interaction
  functions, one `passedThrough` per hero loop (the device's three sites folded), each mode-M walk's
  coverage test; the clear-on-real-vertex inversion declined (known-issues).
- [ ] **GROOM-SILENT-OOM** — the groom tool can exit without a word when memory is short.
- [ ] **GROOM-DRAG-UNTESTED** — the groom tool's drags have no automated test.
- [x] **"mode `U` (VCM) still uses a dense cell grid with the same `(int)(nCells + 1)` overflow"**
  (0.450.12) — the device capped since 0.381.7; the host's VcmGrid takes the same cap now (a fine
  radius over a large box had CPU mode U exit 1 silently); tools/check_vcm_grid_cap.py.
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
- [x] **RASTER-GPU-ISO-FLAT** (0.449.9) — `-raster-gpu` resolves pattern-driven mixes and reflect patterns.
- [ ] **"loom preview server (`-serve`) is resident-process only"**.
- [ ] **"HIP alias block deliberately omits `__shfl_sync`"** (HIP is untested anyway).

### 4.5 Performance

- [x] **BLUR-EPOCH-COST** (0.434.12) — a blurred clock epoch of mode M or J rebuilt its whole
  light side every 4 samples; it draws what its light side can afford against its gather now, the
  caustic neighbour count scaled with it.
- [x] **J-FIELD-SLOW** (0.434.33-0.435.0) — mode J in an unbounded density field: the wavefront
  no longer spills, the beams end at the scene's beam domain, and the default map is a fixed 1000
  beams (`-jbeams`), the knee rule trailing it 1.5-3.5x on the fog scenes.
- [x] **J-AUTO-PROBE** (done, 0.436.8: camera rays where a medium is unbounded, which lands an
  unbounded haze's `auto` on the fixed map; known-issues has the measurements) — what is left of
  J-FIELD-SLOW under `-jbeams auto`: `probeGatherCount`'s
  random chords of the beams' box gather 32 beams where a camera segment in an unbounded haze
  gathers far more, so the knee overshoots. A probe drawn as the camera draws its rays would say
  whether `auto` can land where the fixed map does; measure with the beam domain's scale (and
  whether the domain should hold the camera), on `scraps/probe/sweep4.sh`'s scenes. (Every
  campaign scene's camera sits inside its domain, so that last question did not arise.)
- [ ] **AUTO-BEAMK** — what J-AUTO-PROBE left between `-jbeams auto` and the fixed map: the knee
  is where a probe gathers `-beamk` beams, 32 by default, but the best arms of every campaign
  scene gather 1-20 a camera segment, and in bounded media `auto` still trails 1000 beams by
  ~2x. Sweep `-beamk` (4, 8, 16) under each scene's probe with `scraps/probe/sweep6.sh`'s scenes
  and `gallery_rain` (whose cloud and rain are what `auto` is for), and move the default if one
  target serves them all.
- [ ] **GPU-BDPT-LOCALMEM** — Nsight Compute (2026-11-10): the GPU camera pass of modes D and J is
  bound by its threads' local memory (94 % of L1 traffic, mostly the vertex stacks, not spills; 5-6
  of 32 threads busy a warp), and mode J's surface merges by divergence (2.3 of 32 on
  `_fog_cornell`, 3.5x mode D's instructions). The surface gather is queued since 0.437.0 (1.7x
  on `_fog_cornell`'s surface merges). Left: a smaller SoA vertex record, then `__launch_bounds__`
  again.
- [ ] **JSPEC-CARRIER-SIZE** — every mode-J GPU thread reserves a 120-byte spectral carrier per
  segment and per eye vertex, chromatic scene or not: VRAM (~0.5 GB, ~3.2 GB on the deep kernel),
  not speed. The slow camera pass measured against it was the MIS weights' per-connection
  transmittances, computed once a sample since 0.436.5 (known-issues). Left: the template
  parameter that sizes the carrier to nothing in an achromatic render costs two more variants of
  the largest kernel -- decided (1) (4.1), with (2)'s half-precision bins kept for if VRAM gets
  tight.
- [x] **RADCACHE-BATCHES** — 0.439.3: the cache advances at its own schedule's boundaries over
  absolute samples, its merge canonical; the chunks follow the clock, and the image is the same
  at any thread count, chunking or stop and resume (`tools/check_radcache_det.py`).
- [x] **BEAMEPOCHS-CHUNK-PIN** — `-beamepochs` pinned the GPU chunks to 1 spp, which cost a cheap
  scene. Done, 0.438.2: the surface realizations follow sample blocks, one a sample under the flag,
  and the chunks stay on the clock.
- [x] **J-FIELD-SLOW's "A `-time` budget cannot end a GPU chunk"** — a GPU mode D/J chunk is at least
  one sample of the whole frame and the budget is read between chunks, so a slow sample overran it
  (the field at 512x384 ran one chunk over 55 minutes against `-time 120`). Done, 0.436.6: the
  budget's deadline cuts the chunk in flight once the film holds a sample (known-issues says why
  the part-sample chunk proposed here was not the fix).
- [x] **J-FIELD-SLOW's resume passes re-walk from the root** — closed by measurement, 2026-11-10
  (0.436.3): across `_fog_cornell`, `_fog_thick`, the still haze, the density field and `_slab_ss`,
  each at 128x96 and 512x384 under the fixed map and the knee rule (20 s each), and `gallery_rain`
  at 640x360 (120 s), one render took one resume pass (`_fog_cornell`, knee rule, 512x384: 67 942
  segments, well under a second); the beam domain and the smaller maps ended the overflows, so the
  saved-stack resume would not repay its code. Was: a segment the full hit queue turned
  away is walked again from the beam tree's root in a resume pass (~0.8 s a pass on the field; at
  96x72 every one of its 51 waves overflowed once). Saving each deferred segment's traversal stack
  in a double-buffered side store, the re-walk kept as the fallback when it fills, removes that.
  After the beam-map fixes (0.434.40's domain, 0.435.0's `-jbeams`), measure overflow on the
  reference scenes and build it if any still overflow.

- [ ] **FIRE-NEE-STRATA-COST** — one fire light sample costs up to 32 traversals.
- [ ] **Fire births** (EMISSIVE-VOLUME-SILENT's efficiency debt) — drawn uniformly over the box;
  drawing them from mode R's emission grid (`EmissiveVolume::cellW`) makes beta nearly constant.
- [ ] **"mode `D` has no resident multi-camera session"** — a flyby re-uploads the scene per frame.
- [ ] **"the beam-map build is host-only and single-threaded"**.
- [ ] **"a mesh-bound medium's voxelisation is 86 % of `gallery_rain`'s load"** — its triangle pass
  is serial.
- [x] **"`-mode W` inherits mode R's `-spp 256` default"** (0.449.7) — mode W defaults to 16.
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
- [x] **DEFORM-LIMITS** (0.440.0) — a dynamic elastic had to be first in its mesh's stack.
- [ ] **MOTION-LIGHT-RIGID** — a spot, beam or aimed light its keys stretch unevenly, and a sun or
  sky in a moving group, stay at the instant.
- [ ] **ROUTE-D-GPU-STREAM** — on the GPU, mode J's renderer with every path routed D draws other
  random streams than mode D in a scene without media (unbiased; the host is byte-identical).
- [ ] **PER-OBJECT-MODES** — ~~emissive surfaces and bare mesh instances cannot be tagged~~ (done,
  0.441.0); only D and
  J route (U and the unidirectional modes are separate renderers).

- [x] **An emissive `quad` / isosurface / CSG solid glows but does not illuminate** — done,
  0.442.0: every emissive surface gets an emitter at load (an isosurface's sampled by the lines
  through its box), `glow_only on` keeps one a glow that lights nothing (GLOW-LIGHT-MODES).
- [ ] **Emission-pattern importance sampling** — a patterned emitter (gallery_rain's 2000 m^2
  ground, whose grid lines are 3 % of it) is chosen by the light its pattern lets out since
  0.442.0, but sampled uniformly over its area, so most of its samples land where it is dark.
  Sampling the pattern itself needs its density at a hit: the hit's triangle and barycentrics in
  every mode, host and device.
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

- [x] Both **`camera_curve` frame-range** entries in "Open bugs" — `-frames A B` shipped in 0.337.0.
- [x] **"`-savemap` / `-loadmap` are silently ignored on the CPU"** — `photonmap_io.h` serves both.
- [x] **"an emissive mesh that is not PLANAR is silently re-oriented OUTWARD"** — its own body says
  FIXED (0.338.0, `emit_orient`).
- [x] **VOLCACHE** (2026-09-06) — the deposit split shipped in 0.300.0.
- [x] **"a mesh area light is sampled UNIFORMLY BY AREA"** — the penalty no longer reproduces; the
  emitter-level tree shipped in 0.270.0; measured ~1 % at 90 % waste, retitled not worth the fix.
- [x] **BVH-BUILD** — its heading says not worth parallelising; its body says done in 0.292.0.
- [x] **UPBP-CONV** (claims overturned), **FURDIM** (dissolved), and the **PLANNED UPBP** entry (first
  half done in 0.214.0 — check what is left of the second).
- [ ] **"the `gallery_rain` 600-frame mode-M flyby cannot start — the machine's COMMIT limit"** —
  re-test on the frame-at-a-time GPU path; it may be closed.

### 4.8 Not listed, on purpose

Notes and by-design entries (the GitHub `<video>` NOTE, the p99 anchor BY-DESIGN, `SetWindowTextW`,
`-gi-clamp`, the CIE side-table "don't retry", the timed-out-shell NOTE), work deliberately not
done (`PatOp::MatMulAdd`, GPU marching cubes, sweeps as an FTSL primitive), the environment (Nsight's
admin-locked counters, the one-off `build.bat` failure, the nvlddmkm teardown BSOD — a driver bug,
mitigated — and BUGCHECK-INTSTEER, two blue screens from a driver corrupting kernel memory, ftrace
ruled out; the MediaTek drivers updated and full kernel dumps on, should it recur), and data whose entries call it acceptable (the F-series SPDs, the material presets).

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
