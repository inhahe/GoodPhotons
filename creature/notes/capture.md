# Capture rig

*(Created 2026-08-06. Referenced from `todo.md` P5 and P9, which pointed at this file before it
existed. Extended 2026-08-12 with the operational half — placement, calibration, sync, shot list,
budget — which the original covered only as "4 cameras at ~3 m".)*

**If you are about to shoot, read these four in this order:**
[What each thing you capture actually unlocks](#what-each-thing-you-capture-actually-unlocks) ·
[The session: shot list, time, and the physical limits](#the-session-shot-list-time-and-the-physical-limits) ·
[Where the cameras go](#where-the-cameras-go) ·
[Sync — the on-site procedure](#sync--the-on-site-procedure).
The rest of the page is *why*. The one rule that must not be violated on the day is
[the two-mode split](#the-rig-serves-two-jobs-and-they-must-not-share-a-mode): motion and groom
never share a recording.

**What happens to the footage afterwards is `notes/pipeline.md`** — the command sequence from four
camera cards to a trained policy — and **`notes/training.md`** for the training commands
specifically. Read pipeline.md *before* shooting, because it says plainly which stages are unbuilt:
the ingest, calibration and keypoint-import glue, and the anatomy/motion fits that consume this
page's output, do not exist yet.

The ordering decision that affects *when you shoot*: run the self-test first, so a broken fit is
discovered before an animal, an owner and a two-hour session have been spent on footage the
pipeline cannot use. Half of it now exists and is worth running the day you calibrate —

```bash
python tools/fit_selftest.py --calib sessions/.../calib.json
```

— which projects known poses through *your* calibrated cameras and checks the solver recovers
them. It has already earned its keep: it found a stale-Jacobian bug that on real footage would
have been misdiagnosed as a bad calibration. What it does **not** yet do is the part its name
refers to — replaying public mocap of a *real* dog, which is the only version that tests whether a
real animal is representable by the rig at all, rather than testing the optimiser against poses
the rig can trivially reach.

## The rig serves two jobs, and they must not share a mode

The single most important thing on this page: **motion is dynamic, groom is static.** They want
opposite camera settings, and trying to get both from one recording gets you a bad version of each.
The same *hardware* does both jobs fine — just not at the same time.

| | **Motion mode** (P5, P9) | **Groom/appearance mode** (P7) |
|---|---|---|
| what it's for | triangulating joint centres over time | fur direction, pattern, density, length |
| frame rate | ≥120 fps (fast gait, minimal blur) | stills; fps irrelevant |
| resolution | whatever survives the fps budget | **full sensor**, every pixel available |
| framing | whole animal in frame, ~3 m | close range, body-part at a time |
| lighting | ambient / whatever the space gives | **controlled**, deliberately raking |
| subject | moving, un-cooperative | still, sleeping, sedated — or a pelt |
| cameras | ≥4 (quadruped self-occludes badly) | 1 is enough; more is convenience |

Minimum for motion: **4 cameras** for a quadruped — a quadruped self-occludes far worse than a human,
and a leg on the far side is invisible to fewer than that. Calibration target + synced trigger.

## Why groom cannot be inferred from the motion footage

Worth writing down because it looks like a free win and isn't:

- **Resolution.** A dog at ~3 m filling ~600 px of a 1080p frame, fur diameter 50–100 µm ⇒ one strand
  projects to roughly **0.01 px**. Individual fibers are not merely hard to see, they are far below
  the sampling limit. You are recovering *texture*, not strands.
- **The fps/resolution trade fights you.** 120 fps forces resolution down exactly when you need it up.
- **Motion blur is a systematic bias, not just noise.** Blur smears apparent fur orientation along the
  direction of travel, so a naive estimate measures where the animal is *going*, not which way the
  coat *lies*.
- **The surface is deforming**, so multi-view stereo has no rigid frame to work in. (Sequencing helps
  — P5 solves the body first — but it doesn't recover the resolution.)

Human hair capture (Paris et al.; the newer neural-strand methods) works because it violates every one
of those: close range, many megapixels, controlled light, stationary subject. Reproduce those
conditions in a separate stills session and most of the difficulty evaporates.

## What is actually recoverable, in descending order of realism

1. **Coat pattern / albedo — easy, highest value per unit effort.** Multi-view gives the unwrap;
   spots, stripes, ticking come out as a texture. Not research.
2. **Direction field — the good one.** Oriented filter banks (Gabor at multiple orientations) recover
   2-D flow from the *aggregate anisotropic texture*, so it survives at texture scale where
   per-strand methods die. Lift to 3-D by cross-view consistency. **The anisotropic specular
   highlight is a strong second cue** — it runs perpendicular to the strands — which is the concrete
   reason groom mode wants *controlled, raking* light rather than ambient.
3. **Density** — weakly, from how much skin shows through.
4. **Length** — only at silhouettes, where the fuzz profile against the background is a decent proxy.
   Interior length is occluded and essentially unrecoverable.
5. **Clumping** — never directly. See below; it comes back as a statistic.

## The reframing that makes this tractable: fit parameters, don't reconstruct geometry

P7 locks the groom to **procedural and anatomy-parameterised**. That changes the problem completely:
there is nothing to reconstruct, there is a **~10–30 number parameter vector to fit**. So:

- **Analysis-by-synthesis on summary statistics**, not per-strand matching. Render the procedural
  groom at the current parameters and compare against the captured stills on:
  - orientation histograms (direction field, anisotropy strength),
  - the **spatial-frequency spectrum of the texture** — this is where clump scale lives, and it is how
    clumping is recovered *as a statistic* even though a clump is never individually visible,
  - the silhouette fuzz profile (length, density).
- ftrace is **not** differentiable, so use derivative-free optimisation. Over ~20 parameters CMA-ES is
  entirely adequate; don't build a differentiable renderer for this.

## The version that's actually novel

Don't fit *the groom*. Fit the **anatomy → groom mapping**.

We will have the anatomical model *and* images of the real animal, so direction-field-as-a-function-of-
muscle-topology is fittable, rather than a direction field baked into UV space. That **generalises
across morphs**, which is exactly what P4 needs and exactly what a fitted texture map would fail to
do — a baked map goes stale the moment the body changes. This is the "we own both layers" claim
showing up somewhere concrete and testable.

## Risk

Human hair capture is well-trodden. **Short animal fur, at distance, uncontrolled, is not**, and there
is no solved prior art to point at. So groom inference sits in the same risk bucket as P5, *not* in the
known-good pile.

What the stills-session reframing buys is a clean split: it pulls the easy 80% (pattern, direction
field) out of the risk bucket entirely, and leaves only the genuinely uncertain part — the
anatomy→groom mapping — inside it. Do the easy 80% first and treat the mapping as a research bet.

## The decided rig — 4× GoPro HERO 12 Black  *(decided 2026-08-07; supersedes the open
hardware question that used to sit here — the webcam/phone option is retired for motion mode)*

Target: markerless capture of a real animal, **in a messy real environment**, at ~140 fps
(user spec). The HERO 12's native high-fps ladder is 4K@120 / 2.7K@240 / 1080@240 — either side
of 140 works; 4K@120 buys keypoint resolution, 2.7K@240 buys sync margin and less motion blur.
**Decide by test on the actual animal at the actual distance**, scoring 2D keypoint confidence,
not by taste.

### Settings that are load-bearing (each one silently corrupts geometry if wrong)

- **HyperSmooth OFF. Non-negotiable.** EIS is a per-frame software reprojection — it *warps the
  image to fake a steadier camera*, which makes the extrinsics time-varying and unknowable. A
  calibration taken once is then wrong for every frame. Rigid-mount the cameras and turn all
  stabilisation off.
- **Capture native wide; never in-camera Linear/SuperView dewarp.** The lens is markedly
  fisheye-ish. In-camera "Linear" is a baked, opaque resampling that discards the edges and blurs
  the corners — and you cannot calibrate through it as accurately as you can calibrate the raw
  projection itself. Fit a real **fisheye/omnidirectional model** (OpenCV `cv::fisheye` or the
  omnidir module) to the native image in calibration, and undistort *keypoints*, not frames.
- **Calibrate per camera, at the exact capture mode/FOV**, with a large board, covering the frame
  **edges and corners** where fisheye distortion is strongest — an animal will constantly transit
  them. Re-check intrinsics if the mode changes; they are per-mode, not per-camera.
- **Protune on, everything locked**: fixed shutter (fast enough to freeze a distal limb —
  1/480 s or faster; budget light accordingly), fixed ISO ceiling, fixed WB. Auto-exposure mid-clip
  changes blur and appearance, which keypoint detectors read as jitter.
- **Rolling shutter persists** regardless of settings — a fast-moving limb is skewed within each
  frame. High fps shrinks per-frame motion and therefore the skew; treat the residual as part of
  the noise floor rather than trying to model it (until P5's error budget says otherwise).

### Sync: timecode, not genlock — know the difference

GoPro's wireless timecode (Quik app / GoPro Labs) aligns the cameras' *clocks*, so frames are
**labelled** on a common timeline. It does **not** genlock the *shutters*: each camera free-runs,
so a residual sub-frame offset of up to one frame interval remains between any two cameras.
The mitigation is frame rate itself — at 240 fps the worst half-frame offset is ~2 ms, during
which a galloping paw travels ~1 cm; at 120 fps double that. This is exactly why the fps spec
exists, and why triangulation should use a solver tolerant of small per-camera time offsets
(or interpolate keypoint tracks to a common clock, which the timecode makes possible).

- **Re-sync every session** — the internal clocks drift over hours.
- **Clap/flash at the start of every recording anyway**: one frame-exact event visible to all
  four cameras verifies the timecode and is the fallback alignment if it fails.
- **GoPro Labs firmware** is worth installing regardless: QR-code configuration makes "identical
  settings on all four cameras" a scan instead of a menu pilgrimage, and it exposes finer sync
  and trigger control than the consumer stack.

### The messy environment is a keypoint problem, not a rig problem

No chroma screen, no markers, clutter and other moving things in frame — that is
DeepLabCut/SLEAP's normal operating regime (they were built for lab cages and field footage), so
the answer is training data and view redundancy, not set dressing:

- **Label frames from the real environment**, including occluded/edge/motion-blurred cases —
  the detector generalises to the clutter it was shown.
- **Four views is what makes outlier rejection work**: triangulate with RANSAC over view subsets;
  a keypoint hallucinated in one view (on a branch, a shadow, another animal) is outvoted by the
  other three. This robustness argument — not just occlusion coverage — is why the minimum is 4
  and not 3.
- Reprojection-error gating + temporal smoothness (P5's regularisers) mop up what RANSAC misses.

## Where the cameras go  *(added 2026-08-12 — this page named a camera count and a distance and then stopped; the geometry was never written down)*

### The number that decides the layout: depth error

For a triangulated point at range `Z` from a pair with baseline `B`, the along-depth uncertainty is
roughly

```
sigma_Z  ~  (Z / B) * p * sigma_px          p = the pixel footprint at the subject
```

At the working distance this page already specifies (~3 m) with the native wide lens (~92° H),
4K gives **p ≈ 1.6 mm/px**, and P5's Huber δ = 2 px is a fair stand-in for `sigma_px`. So:

| pair separation around the subject | baseline at Z = 3 m | sigma_Z |
|---|---|---|
| 90° | 4.2 m | **~2.3 mm** — parity with lateral error |
| 60° | 3.0 m | ~3.2 mm |
| 40° | 2.1 m | ~4.6 mm |
| 20° | 1.0 m | ~9.2 mm — 4× worse for no saving |

That table is the whole argument for spreading the cameras rather than clustering them: below ~40°
separation, depth is the dominant error and bone-length estimates inherit it. Against femur lengths
of 0.115–0.285 m, 2–3 mm is a **~1–2 % length measurement**, which is the accuracy the anatomy fit
actually runs on.

### The layout

**Four cameras at the corners of a square, ~90° apart, on a circle of radius ~3 m, aimed at its
centre.** Then every pair is either 90° (excellent) or 180° (degenerate on its own, but each
camera has two 90° partners, so no keypoint is ever left with only the bad pair).

- **Stagger the heights: two at ~1.6 m tilted down ~20°, two at ~0.7 m tilted down ~5°.** The high
  pair sees floor contacts and the far-side legs over the body; the low pair sees the leg segments
  near-side-on, which is where limb keypoints are most confidently detected. Elevation is an
  *occlusion* argument, not a conditioning one — a quadruped occludes its far legs with its own
  trunk, and that is the failure the 4-camera minimum exists to fix.
- **Never put all four on one line or one arc.** Two lateral views 20° apart are, per the table
  above, one lateral view with extra storage cost.
- **90° also breaks left/right limb confusion**, which is the characteristic DLC/SLEAP failure on a
  purely lateral view — the near and far legs cross the same image region and the detector swaps
  them. A view down the animal's long axis makes the swap geometrically impossible, and P5's RANSAC
  gate then outvotes it rather than averaging it in.
- **Front/rear views are what make `hip_width` and `shoulder_width` identifiable at all.** Those two
  morph params are almost invisible from the side. If the layout drifts toward "two lateral pairs",
  those knobs stop being fit and start being guessed.
- Rigid mounts, not tripod-with-a-loose-pan-head. §Settings turns EIS off precisely so extrinsics
  are constant; a camera that gets nudged mid-session invalidates every frame after the nudge.
  Mark the floor positions with tape so a knocked tripod can be put back and re-verified rather
  than re-calibrated from scratch.

### Capture-volume sizing — and why gallop does not fit in it

At 3 m with ~92° H FOV a frame covers ~6.2 m across, but the volume where the animal is in **all
four** views is much smaller: roughly a **3 × 3 m** footprint. Set that against how far a dog
travels per stride:

| gait | speed | stride length | strides per pass through a 3 m volume |
|---|---|---|---|
| walk | ~1.2 m/s | ~1.2 m | ~2.5 |
| trot | ~2.5–3 m/s | ~1.6 m | ~2 |
| gallop | ~8–10 m/s | ~2.5–3.5 m | **~1** |

So the arena as specified is sized for **walk and trot**, and gallop gets about one stride per pass.
There is no free fix; pick one deliberately and write down which:

- **more passes** — the cheapest option, and the default. ~10 clean gallop strides then means ~10+
  usable passes rather than ~5.
- **pull the cameras back to ~5 m** — doubles the volume, but `p` grows to ~2.7 mm/px and `sigma_Z`
  to ~4 mm; the anatomy fit degrades from ~1–2 % to ~3 %.
- **the corridor layout** — abandon the circle for a straight runway with two cameras per side,
  staggered along the run and toed in so their fields overlap across a ~6 m working segment. Buys
  length at the cost of the 90° property over part of the run (pairs across the corridor stay wide;
  pairs along it do not). Worth it only if gallop specifically is the goal.

Decide this per session and record it in the session log — extrinsics are per-layout, so switching
layouts mid-session means re-calibrating.

## Calibration — the on-site procedure

Intrinsics are **per camera per capture mode**; extrinsics are per layout and die the moment a
tripod moves. Both are cheap to redo and catastrophic to get wrong, so both happen every session.

1. **Board.** ChArUco, not a plain checkerboard — partial views still solve, which matters because
   the frame *edges and corners* are where fisheye distortion is strongest and where you most need
   coverage. Size it for the volume: **~0.6–1.0 m** on a side. A laptop-sized board at 3 m is a few
   hundred pixels and will not constrain the distortion tails.
2. **Intrinsics, per camera, in the exact mode you will shoot** (4K120 and 2.7K240 are *different
   intrinsics*): 30–60 stills or a slow sweep, board deliberately pushed into all four corners and
   tilted hard. Fit `cv::fisheye` (or omnidir). **Never in-camera Linear** — §Settings.
3. **Extrinsics:** wave the board slowly through the whole working volume so it is seen by
   overlapping camera pairs, then bundle-adjust (this is exactly what Anipose's calibration does).
   Cover the volume at multiple heights, including floor level.
4. **Scale and floor.** The board's known square size is what makes the whole reconstruction
   *metric* — without it the fit is shape-up-to-scale and `body_scale` is meaningless. Also lay the
   board flat on the floor for a few frames: that fixes the ground plane, which E_phys's
   floor-penetration and foot-slip gates need.
5. **Verify before the animal arrives:** triangulate the board corners and check the recovered
   square size against the printed one, and check reprojection RMS is ≲1 px. A calibration that is
   wrong is not detectable from the footage afterwards.
6. **Re-shoot calibration at the END of the session too.** If a tripod drifted, the two calibrations
   disagree and you know which takes to distrust — this costs two minutes and is the only thing
   that turns a silent corruption into a detected one.

## Sync — the on-site procedure

§Sync above explains *why* timecode is not genlock. The steps:

1. **GoPro Labs firmware on all four cameras** (already recommended above for QR-code config).
2. **Set all four from one QR code** — mode, fps, Protune lock, RAW audio. Identical settings is a
   correctness requirement, not tidiness: mismatched shutter changes blur, and mismatched mode
   changes intrinsics.
3. **Wireless timecode sync at the start of the session, and again every ~2 hours** — the internal
   clocks drift.
4. **Clap-and-flash at the head of every single take.** One frame-exact event in all four views (and
   all mic tracks, §Audio). This is the verification of the timecode and the fallback if it failed;
   it costs one second and there is no way to reconstruct it later.
5. **Roll all four before the clap and stop all four after the take.** Do not start a camera late.
6. **Residual offset is up to one frame interval** between any two cameras, and no procedure removes
   it — this is why the fps spec exists. Interpolate keypoint tracks onto a common clock (the
   timecode makes that possible) rather than assuming frame *n* is simultaneous across cameras.

## What each thing you capture actually unlocks

The question this page kept getting asked and never answered: *what do I have to shoot to turn which
knob?* Against `canis.ftcl`'s 26-parameter morph vector and the downstream phases:

| capture | what it makes identifiable / possible | how much |
|---|---|---|
| **Calm standing square**, all four feet planted, animal settled | the 8 `stance_*` params — nearly a third of the morph vector — plus the cleanest frames for the length params. The single highest-value clip on this list. | 3–5 takes × 5–10 s |
| **Slow walk, straight, both directions** | the 12 length params (`trunk/neck/head/tail/femur/tibia/hmeta/scapula/humerus/radius/fmeta/paw`) by triangulated-segment medians; θ₀ initialisation; AMP walk demos | ≥10 clean strides ⇒ ~5 passes/direction |
| **Trot** | AMP trot demos; the gait the P2-mocap yardstick comparison is run on | ≥10 clean strides ⇒ ~5 passes/direction |
| **Gallop** | AMP gallop demos; the flight phase is the strongest E_phys signal | ≥10 strides ⇒ ~10+ passes (see volume sizing) |
| **Front-on and rear-on passes** | `hip_width`, `shoulder_width` — weakly constrained from lateral views | 2–3 passes each |
| **Turns / circles, both directions** | left-right asymmetry check (a healthy animal should fit symmetric; a large asymmetry means a bad calibration or a limp); P10's yaw channel | 2–3 circles each way |
| **Sit, lie down, get up, shake, scratch, head turns** | P10 §Postural transitions and the channel inventory; also the *actual* joint ranges, which the authored `.ftcl` limits are currently a guess at | 3–5 takes each |
| **Silhouette-friendly passes** (animal clearly separable from background) | `trunk_radius`, `limb_gracility` — these live **only** in E_sil, never in keypoints | woven through the above; prefer an uncluttered backdrop for a few passes |
| **The animal on a bathroom scale** | `body_mass` — kinematically invisible, no amount of footage recovers it (todo.md P5 §Identifiability honesty) | one number, once |
| **A tape measure on 2–3 bones** | an independent check that the metric scale is right before you trust 26 fitted numbers | 5 minutes |
| **Groom stills session** (separate; §two-mode rule) | P7's ~10–30 groom params: pattern, direction field, density, silhouette-length | own session, ~30–45 min |
| **Vocal session** (separate; §Audio) | P12's sample library | own session |

Two caveats that belong here rather than downstream:

- **`body_scale` is a gauge freedom.** It multiplies every length, so it is degenerate with the
  twelve length params unless the fit fixes a convention (fit lengths at `body_scale = 1`, or fit
  `body_scale` from total length and the rest as ratios). Nothing you capture resolves this; it is
  a parameterisation choice and the fit has to make it explicitly.
- **Fur inflates the silhouette.** E_sil measures the *coat*, so `limb_gracility` and `trunk_radius`
  fit from a fluffy animal describe the fluff, not the bone. On a long-coated subject expect a
  systematic overestimate — and note that the groom session is the thing that can eventually
  measure and subtract it.

### "What do I capture to make a *new* creature?" — nothing, and that is the design

Worth stating plainly because it is the most natural wrong assumption. You do **not** capture a
second animal to get a second creature. The capture produces `θ_animal` — a point in the
pre-authored template's morph space — and every other creature in that body plan is *another point
in the same space*, reachable by dragging sliders in P8's editor with no new footage at all. See
todo.md **P6 §Converting a captured animal**: the order is author template → capture + fit θ → edit
anatomy → train **once**, conditioned on morph and randomised over a *generous neighbourhood* of θ.

The capture-side consequence is a single instruction: **capture the animal that sits nearest the
centre of the region you eventually want to explore**, and set the morph ranges generously before
training. What a new capture *is* needed for is a **different body plan** — a different template,
different topology, and P4's morph manifold explicitly does not span those (you cannot interpolate a
leg into a wing; see P11).

## The session: shot list, time, and the physical limits

A realistic first motion session, in order:

| # | step | time |
|---|---|---|
| 1 | rig placement, tape the floor marks, power/cards in | 20 min |
| 2 | QR-config all four, timecode sync, RAW audio on | 5 min |
| 3 | calibration: intrinsics (if mode changed) + extrinsics + floor + verify | 15–20 min |
| 4 | weigh the animal; tape-measure a couple of bones; log it | 5 min |
| 5 | **standing square** takes | 5 min |
| 6 | walk passes, both directions | 10–15 min |
| 7 | trot passes, both directions | 10–15 min |
| 8 | gallop passes (if the space allows) | 15–20 min |
| 9 | front/rear passes, circles | 10 min |
| 10 | postural transitions, shake, scratch, head turns | 10–15 min |
| 11 | closing calibration re-shoot | 5 min |
| | **total on site** | **~2 hours** |

What that yields, roughly: ~15–25 min of *recorded* footage, of which — with an uncooperative
subject — expect **10–25 % usable** after culling for occlusion, frame-edge exits, detector failure
and the E_phys gate. So ~3–5 min of usable footage, of which the anatomy fit needs only ~300 frames
(todo.md P5: "every 4th frame of a few standing/walking clips") and the AMP demos need perhaps
30–60 s. **The binding constraint is variety and cleanliness, not duration** — this is not a
"record for hours" problem, it is a "get ten clean strides of each gait" problem.

Three physical limits that will end a session early if unplanned for:

- **Thermal.** A HERO 12 at 4K120 in a warm room overheats in roughly 20–30 min of continuous
  recording. Short per-take rolls (which the clap-per-take protocol already implies) are the
  mitigation; continuous rolling is not viable.
- **Storage.** 4K120 runs ~100–120 Mbps ⇒ ~0.8 GB/min/camera ⇒ **~3.4 GB/min for the rig**. Twenty
  minutes of recording is ~70 GB. 256 GB cards minimum, V30 or better, and offload before the
  session ends — a card swap mid-session is also a chance to knock a tripod.
- **Power.** High-fps recording drains a HERO battery in well under an hour. External USB power on
  all four, or the session ends at step 8.

**Log every session**: layout used, camera modes, calibration files, the animal's mass, what was
shot in each take, and anything that went wrong (a nudged tripod, a failed timecode). A take with no
provenance is a take you will not trust in six months.

### Groom mode hardware note

Groom/appearance mode is unchanged by this decision: stills of a stationary subject want full
sensor resolution, close range and controlled raking light — a phone (or the GoPros in 27 MP
photo mode, though a phone focuses closer) remains entirely adequate. The two modes still must
never share a recording.

## Audio — the stream the rig was about to throw away  *(added 2026-08-09)*

The pages above capture what the animal *does* and what it *looks like* and silently discard what
it *sounds like*. That omission is expensive in exactly one direction: every downstream consumer
of sound (todo.md **P12**) is deferrable, but the recording is not — a session that ran without
audio cannot be re-run. So the protocol change is immediate even though the consumers come later:
**from the first motion session onward, sound is recorded.**

### What comes free: four synced GoPro tracks

Every motion-mode recording already contains four audio tracks, timecode-labelled with the video
by construction. Enable Protune **RAW audio (.wav sidecar)** on all four cameras — it is a QR
scan away given GoPro Labs is already installed. Their quality is honestly mediocre (distance,
AGC history, and above all **wind** — fit foam/deadcat covers; wind is the difference between a
usable track and a useless one), but two of the three uses of audio need *timing*, not fidelity:

- **Footfall onsets.** A footfall is a broadband transient; spectral-flux onset detection on a
  48 kHz track localises it to well under a millisecond, against 4–8 ms video frames *plus* the
  sub-frame shutter offsets §Sync already documents. Contact timing is precisely where video is
  weakest (a paw at the ground is where occlusion, blur and rolling-shutter skew all concentrate)
  and precisely what P5's physics-plausibility term wants. One correction is mandatory: sound
  travels at ~343 m/s ≈ **2.9 ms per metre**, so at 10 m the propagation delay is ~29 ms — several
  frames' worth, far larger than the error being fixed. The fitted trajectory knows where the
  animal is at every frame, so the per-event delay to each camera is closed-form: solve pose
  coarsely, correct the onsets, and the sharpened contacts feed back into the fit's contact
  regulariser. Gate the whole signal by SNR — hooves on hard ground are a strong signal, a dog on
  grass is not, and an onset that isn't clearly there should count for nothing.
- **Vocalization *timing* and pairing.** Which bark happened during which stride, at which breath
  phase — the pairing datum P12's breath-gated scheduler is fit against. The GoPro tracks
  timestamp the bark; they do not have to make it pretty.

### What does not come free: the library recorder

The third use — the **vocalization library** (the samples the creature will actually play) — is
the one the GoPro tracks cannot serve: too far, too windy, too compressed. Add exactly one piece
of hardware: a **32-bit-float field recorder with a directional (super-cardioid/shotgun) mic**
— Zoom F2/F3 class. 32-bit float removes gain-setting from the protocol entirely (nothing to
clip, nothing to ride during a session with an uncooperative subject), which fits a rig philosophy
that already trades operator attention away wherever possible. Sync is the solved dual-system
problem: the **clap/flash that already starts every recording** serves double duty (flash for the
cameras, clap for every mic), and cross-correlation against the GoPro tracks recovers alignment
even when the clap is missed. Recorder drift over a take at these durations is a few ms — fine
for a library sample, irrelevant for pairing.

### The two-mode rule extends to audio, with the same shape

Motion-mode audio (far, synced, timing-first) cannot yield library vocals, and a library session
cannot yield gait pairing — the same "opposite settings, same hardware" doctrine, applied to
sound:

| | **motion mode** (audio side) | **vocal session** (audio's groom mode) |
|---|---|---|
| what it's for | onset timing; (vocal, stride, breath-phase) pairing | the sample library itself |
| mic | 4× GoPro tracks + recorder wherever it fits | shotgun close (≤1 m if the animal allows) |
| environment | whatever the space gives | quiet, wind-sheltered, controlled |
| subject | moving, incidental vocalizations | provoked/opportunistic — doorbell, play, food |
| cameras | all four (that's the point) | **1–2 close** — see below |

The vocal session should still run one or two cameras close: the whole-animal 3 m framing of
motion mode cannot resolve the jaw/lip/throat detail that must co-animate with a vocalization,
and the close session captures exactly that (sound, face/head motion) pair at high resolution.
That is the audio analogue of "groom needs its own stills session" — and like groom mode, it is
cheap, unhurried, and can be repeated on a calm subject.
