# Capture rig

*(Created 2026-08-06. Referenced from `todo.md` P5 and P9, which pointed at this file before it
existed.)*

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

### Groom mode hardware note

Groom/appearance mode is unchanged by this decision: stills of a stationary subject want full
sensor resolution, close range and controlled raking light — a phone (or the GoPros in 27 MP
photo mode, though a phone focuses closer) remains entirely adequate. The two modes still must
never share a recording.
