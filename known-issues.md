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

### RESOLVED: Backward reference tracer now validates participating media (fog)
- **What (was):** `src/backward.h` ignored `scene.medium` — its camera rays didn't
  sample volume free-flight or in-scattering, so `-fog` with modes R/V would have
  compared a volumetric forward image against a vacuum backward image (garbage
  residual). Fog was therefore forward-only and never set in refMode.
- **Fix applied:** added a homogeneous-medium path to `BackwardRenderer::radiance`
  that mirrors the forward tracer exactly:
  1. **Free-flight sampling** competes with the surface hit each bounce
     (`tMed = -ln(1-u)/sigma_t`; on `tMed < dSurf` a volume collision occurs).
  2. **`neeVolume()`** — phase-function next-event estimation at the collision
     vertex: the surface BRDF/cosine are replaced by the single-scattering albedo
     and the HG phase function `hgPhase(dot(wIn, wi), g)`, with fog transmittance
     `exp(-sigma_t*dist)` on the shadow ray (the backward mirror of the forward
     `connectVolume`). The phase angle uses reciprocal conventions to the forward
     side, both equal to the physical `dot(prop_in, prop_out)`.
  3. **Analog scatter/absorb** continuation: survive with prob = albedo, then
     `sampleHG` a new direction (throughput unchanged); otherwise absorb.
  4. **Beer-Lambert on surface NEE too:** `neeLight` now attenuates its shadow ray
     by `exp(-sigma_t*dist)` (took a new `lambda` parameter).
- **Validation (mode V, forward vs backward, identical fog):** the best-fit
  backward→forward scale agrees to ~4 sig figs across no-fog / fog-g0.3 /
  fog-rayleigh, which a transport bug could not produce. The raw-linear residual is
  firefly-dominated (top-1% pixels hold 77–95% of it) from the unbounded 1/dist^2
  light connection, so full RMSE plateaus but the **bulk RMSE (ex. top-1%) scales
  as ~1/sqrt(N)**, proving variance not bias: for fog g=0.3 alb=0.85 at 256^2, bulk
  RMSE went 7.67% (120M/800spp) -> 4.53% (480M/3200spp) [1.69x ~ ideal 2x], with
  firefly concentration held constant at ~86% and the 4x run reporting PASS.
  No-fog bulk RMSE is 1.2% (95% firefly-concentrated). The firefly-vs-bias
  diagnostic (residual concentration + bulk RMSE) was added to `compareFilms` in
  `src/main.cpp` specifically to make this distinction rigorous.
- **Status:** RESOLVED 2026-07-10. `-fog` can now be combined with modes R/V.
  `-checkfog` (deterministic transmittance / HG mean-cosine / phase-normalization
  self-test) is retained as a fast complementary check.

### GPU backend (`-device gpu`) covers model B only
- **What:** the CUDA backend (`src/render_cuda.cu`, `renderForwardCudaMB`) implements
  only the forward model-B light trace (connect/splat to the pinhole). It is used
  for `-mode B` and the forward pass of `-mode V`; it silently falls back to the CPU
  for modes A (contact sensor), C (finite-aperture forward catch), R (backward
  reference), and for the mode-P camera-side/backward layer. Fluorescent scenes are
  rejected on-device (fall back to CPU) because the emission-sampler reradiation
  path is not ported — `cudaForwardSupported()` checks whether any *geometry* uses a
  Fluorescent material (not just the palette, which buildCornell always populates).
- **Why acceptable:** model B is the default forward mode and the one that dominates
  render cost; mode R/backward and mode C are validation/creative paths that run at
  lower sample counts. Validated: GPU vs CPU image RMSE ≈ 0.85/255 at 200M photons
  (pure MC noise), energy report matches to 4 sig figs, and `-mode V -device gpu`
  PASSes against the independent backward reference (bulk RMSE 4.17% ≈ CPU 4.22%).
  Measured ~14× speedup (400M photons @256²: 153s CPU → 10.9s GPU on an RTX 4090).
- **Spectral baking:** device materials/fog sample each `std::function` Spectrum into
  a fixed 96-entry table over [360,830] nm with linear interpolation (`SPEC_N=96`).
  Smooth reflectances/Sellmeier indices make this accurate to within MC noise; a
  pathologically spiky spectrum would need a finer table. CIE CMFs are ported
  analytically (no table).
- **Proper fix (future):** port models A/C (photon aperture catch) and the backward
  tracer to CUDA if those paths ever become the bottleneck; add a device
  fluorescence path (bake `fluoEmitSampler`'s CDF) to lift the fluoro restriction.
- **Status:** OPEN (acceptable) — logged 2026-07-10. Requires a CUDA toolkit at
  configure time; without one the project builds CPU-only and `-device gpu` warns
  and uses the CPU.

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
