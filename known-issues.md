# Known Issues & Technical Debt

Running log of unsolved bugs and accumulated tech debt. Fix items here as soon
as practical; this file is the fallback for what can't be addressed immediately.

## Limitations (by design, tracked for future work)

### Backward reference tracer cannot validate fluorescence
- **What:** `src/backward.h` has no Fluorescent case — a fluorescent material
  falls through to the Diffuse branch, so modes R (reference) and V (validate)
  would silently mis-render `-scene fluoro`.
- **Why:** Backward tracing a wavelength-shifting material requires the full
  bispectral reradiation matrix (integrate the camera-side path over all possible
  input wavelengths for each output wavelength). Forward single-wavelength tracing
  handles fluorescence trivially (sample lambda' ~ emission SPD, M/pdf cancels).
- **Mitigation in place:** `-scene fluoro` is a forward-only (model A/B/C) scene;
  it is never selected in refMode. Fluorescence correctness is instead validated
  deterministically by `-checkfluoro` (emission-sampler mean, epsilon*Q branch
  fraction, Stokes shift). Energy conservation holds (`sum/emitted=1.000000`).
- **Proper fix (future):** implement a bispectral backward estimator (reradiation
  matrix / Mojzik-style hero-wavelength reweighting) if we ever want R/V to cover
  fluorescent scenes. Not needed for the forward tracer's own correctness.
- **Status:** OPEN (acceptable) — logged 2026-07-10.

### Backward reference tracer cannot validate participating media (fog)
- **What:** `src/backward.h` ignores `scene.medium` — its camera rays don't sample
  volume free-flight or in-scattering, so `-fog` with modes R/V would compare a
  volumetric forward image against a vacuum backward image (garbage residual).
- **Why:** A backward volumetric estimator needs free-flight distance sampling
  along the camera ray plus phase-function next-event estimation to the light
  (and transmittance on the shadow ray). ~30–40 lines, but non-trivial to get the
  MIS/analog weights right.
- **Mitigation in place:** fog is forward-only; `-fog` is never set in refMode.
  Correctness is validated deterministically by `-checkfog` (Beer-Lambert
  transmittance, HG mean-cosine, phase normalization). Energy conserves
  (`sum/emitted=1.000000`) on foggy renders.
- **Proper fix (future):** add a homogeneous-medium path to `BackwardRenderer::
  radiance` (sample collision, phase-NEE + HG continuation) so mode V can
  cross-validate fog against the forward tracer.
- **Status:** OPEN (acceptable) — logged 2026-07-10.

## Performance

### RESOLVED: Diffuse-mesh renders were ~60× slower per photon (degenerate BVH)
- **Symptom:** The Cornell + diffuse torus (`-mesh torus.obj`) traced at
  ~34–37 µs/photon, vs ~0.55 µs/photon for the Cornell + glass sphere. 3M photons
  took ~112s. Made mesh scenes impractical.
- **Root cause (found by instrumentation):** The BVH build was leaving giant
  leaves. Added a `-bvhstats` diagnostic (nodes/leaf-tests per ray + leaf-size
  histogram) which showed the 16k-tri torus BVH had only **245 nodes / 123 leaves,
  max leaf = 9334 primitives**, and each ray did **~1091 leaf primitive tests**.
  The culprit was the SAH termination in `Bvh::buildRecursive` (`src/bvh.h`):
  `if (bestSplit < 0 || bestCost >= leafCost) makeLeaf();`. Object-SAH on a
  ring-like shape hits a top-level pathology — splitting a torus through its
  centre yields two C-shaped halves whose AABBs each nearly equal the *whole*
  box, so every candidate split has cost ≈ the leaf cost. The greedy "only split
  if it lowers SAH" test therefore gave up immediately at the top and dumped
  most of the mesh into one leaf. (Path length was a red herring: the diffuse
  Cornell walls dominate bounce count regardless of the mesh.)
- **Fix applied:**
  1. **BVH (the real fix):** use SAH only to *choose* the split plane, and always
     recurse down to `LEAF_SIZE`, falling back to a median (`nth_element`) split
     when SAH finds no usable partition. Result: 245→**10425 nodes**, max leaf
     9334→**4**, leaf-tests/ray 1091→**0.7**. Torus 3M render 112s→**1.3s**
     (0.43 µs/photon — now *faster* than the glass-sphere reference).
  2. **Front-to-back traversal ordering** in `traverseClosest` (descend nearer
     child first; cull children against `tMax` at push time). Minor on its own
     (~8%) but correct and keeps the win robust.
  3. **Russian roulette** for Diffuse/Mirror/Glossy in `Renderer::tracePhoton`
     (`src/render.h`): terminate with prob `1-reflectance`, keep `beta` unchanged
     on survival. Unbiased; caps path length (residual now 0.0000) and removed the
     now-dead `betaCutoff`. `maxBounce=32` kept as a hard safety cap.
- **Validation:** `-checkbvh` still reports 0 mismatches on cornell/materials/
  prism/torus; energy conserves exactly (`sum/emitted=1.000000`) on all scenes.
- **Status:** RESOLVED 2026-07-10. Logged & fixed same day.
