# Known Issues & Technical Debt

Running log of unsolved bugs and accumulated tech debt. Fix items here as soon
as practical; this file is the fallback for what can't be addressed immediately.

## Limitations (by design, tracked for future work)

### Multi-camera renders re-trace photons per camera (no shared pass yet)
- **What:** Phase 3a implements multiple named `camera` blocks: one render
  invocation emits one image per camera (`scenes/twocam.ftsl`), with `-camera
  <name>` selection and per-camera film resolution + mode. But each camera is a
  **separate forward pass** — the photon set is re-traced from scratch for every
  camera (`runRender` is called in a loop in `src/main.cpp`).
- **Why it matters:** the wishlist's framing is "many cameras at once… *same
  render for efficiency*". For N cameras this is N× the photon work instead of 1×.
- **Proper fix (future):** a single **shared mode-B photon pass** that connects
  each diffuse/emitter vertex to *every* camera's pupil at once. Photons are
  camera-independent until the `connect()` splat, so `tracePhoton` would take a
  list of (Camera, Film) targets and call `connect`/`connectVolume` once per
  camera per vertex; per-thread films become per-thread × per-camera. Modes A/C
  (contact-sensor / thin-lens forward catch) are inherently per-camera and would
  stay single-camera or need their own catch loop; the CUDA kernel would also need
  the camera list (currently one `DCamera`). Scoped as a follow-up so the initial
  multi-camera feature (correct, just not yet shared) could land validated.
- **Status:** OPEN (acceptable) — multi-camera done 2026-07-10; shared pass deferred.

### Absolute-EV film sensitivity, non-square films, shared multi-camera pass
- **What (remaining):** three camera/film pieces are still open:
  1. **Absolute EV / physical sensitivity.** `iso`/`shutter`/`exposure` are wired
     but act as a *relative* exposure **compensation** on top of the per-image
     auto-exposure (the film's radiometric scale is arbitrary). A true absolute
     exposure (a given ISO+shutter+f-number yielding a physically-determined
     brightness) needs **absolute light power** (watts/lumens) on emitters, which is
     itself deferred (§7). A cheaper intermediate win: an exposure **lock** across
     `camera_path` frames — compute the auto anchor once and reuse it so a dolly
     doesn't flicker frame-to-frame.
  2. **Non-square films.** `film { res W H }` only uses the first value; the
     forward/backward tracers (and CUDA) allocate a square film. Non-square sensors
     (and a true horizontal fov from film **width**) need the tracers to carry
     resX≠resY.
  3. **Shared multi-camera mode-B pass** (already logged above under the multi-camera
     entry) — one photon trace splatting to every camera pupil.
- **Proper fix:** (1) add absolute emitter power + a sensitometric film model; add a
  per-`camera_path` exposure-lock flag as the near-term step. (2) thread resX/resY
  through `renderForward`/`renderBackward`/CUDA and `writePPM`. (3) see multi-camera.
- **Status:** OPEN (design captured) — logged 2026-07-10.
- **Done (2026-07-10, Phase 3a):**
  - `camera_path` keyframed motion — expands at load time into a sequence of
    `CamSpec` frames with piecewise-linear `eye`/`look_at` interpolation between
    sorted `key` control points; the multi-camera loop renders the generated list
    with frame-numbered output names. Grammar is numbers-only
    (`key <t> <ex> <ey> <ez> [<lx> <ly> <lz>]`). Validated by `scenes/dolly.ftsl`.
  - Physical film `size <w> <h>` (mm) → focal length `f = filmH/(2·tan(fov_y/2))`
    (metres); `fstop N` → `apertureR = f/(2N)` at load time (overrides `aperture`),
    giving physically-meaningful DOF in modes A/C. `iso`/`shutter`/`exposure` →
    relative exposure compensation `comp = exposure·(iso/100)·shutter` applied over
    the auto-exposure anchor in `writePPM`. Validated by `scenes/expo.ftsl` (ISO 200
    is exactly 2.0× ISO 100 in linear space).

### Texturing is base-color only, `use_mesh`/quad UVs only (Phase 3b partial)
- **What (done 2026-07-10):** a `texture "name" { file … encoding srgb|linear
  filter nearest|bilinear wrap repeat|clamp|mirror }` block loads an image into
  `Scene::textures` (`src/texture.h`); `reflect texture:<name>` on a `diffuse`
  material binds it (`Material::reflectTex`); per-vertex UVs live on `Tri`
  (`src/geometry.h`), auto-generated for quads and read from OBJ `vt` when a mesh
  sets `uv use_mesh`; each texel is Jakob-Hanika–upsampled to a reflectance
  spectrum (coefficients precomputed at load via `Texture::buildReflCoeff`, bilerped
  + sigmoid-evaluated per hit through `diffuseReflectance()`). Shared by the forward
  tracer and the backward reference. Image formats: PNG/JPG/BMP/TGA + Radiance
  `.hdr` via the vendored stb_image (`src/third_party/stb_image.h`, compiled once in
  `src/stb_image_impl.cpp`), plus built-in PPM/PFM. Validated by
  `scenes/textured.ftsl` (quad) and `scenes/uvmesh.ftsl` (mesh) — the checker maps
  with correct orientation (blue band at v≈1/top, yellow at u≈0/left) and spectral
  colour; a PNG copy of the checker renders bit-identically to the PPM (RMSE 0.0),
  confirming the stb sRGB decode + orientation.
- **Remaining [needs engine work]:**
  1. **UV projections.** Only `uv use_mesh` (OBJ `vt`) and quad corners exist. The
     procedural projections in the spec (§9.2 triplanar/planar/spherical/cylindrical)
     are not built — meshes without `vt` fall back to zero UVs.
  2. **Non-albedo parameters.** A texture can only bind to diffuse `reflect` today.
     Spec §9.4 wants textures on roughness, mix weights, ior, thickness, etc. — each
     needs the corresponding material param to accept a per-hit texture lookup.
  3. **GPU.** Textured scenes force the CPU tracer (`cudaForwardSupported()` returns
     false): the CUDA kernel bakes one reflect spectrum per material and has no
     per-hit texture sampler. A device port would upload texel coeff tables + UVs.
  4. **Indexed-spectral palettes** (§9.3) — an index image + name→spectrum palette —
     not implemented.
- **Status:** OPEN (acceptable) — base-color texturing + stb image import done
  2026-07-10; the four items above deferred.

### Light shapes: sphere + spot done, HDRI environment deferred (Phase 3c partial)
- **What (done 2026-07-10):** two new emitter shapes on the shared `Emitter`
  (`src/scene.h`), both routed through forward + backward + CUDA:
  - **Spherical area light** — `light sphere { center x y z  radius r  spd … }` —
    `shape = EmitterShape::Sphere`, `area = 4·π·r²`. Forward and backward both call
    `Emitter::samplePoint()` (uniform surface point + outward normal; quad draws are
    byte-identical so quad scenes stay bit-identical). The FTSL loader also drops an
    emissive sphere into the geometry (mirroring the area-light quad). Validated by
    `scenes/spherelight.ftsl` (mode V PASS at 60M photons / 1024 spp; CPU==GPU).
  - **Point spotlight** — `light spot { origin … dir … inner_angle d  outer_angle d
    spd … }` — `shape = EmitterShape::Spot`, a cone with a cubic-smoothstep
    penumbra. Geometric weight is the falloff-weighted solid angle `spotOmega =
    π·(2−cosᵢ−cosₒ)`, so `power = emitIntegral·spotOmega`, peak intensity/SPD = 1.
    Forward samples a direction uniformly in the outer cone and reweights beta by
    `falloff·Ω_outer/spotOmega`; backward connects straight to the light point with
    a cone-falloff weight (`I(ω)·cosθ/d²`, no area/light-cosine). No emissive
    geometry (a point). Validated by `scenes/spotlight.ftsl` (mode V PASS at 200M
    photons / 4096 spp → 3.8% RMSE; CPU==GPU energy).
  - `finalizeEmitters()` now keys the power law / combined wavelength sampler off a
    per-emitter `geomWeight()` (area·PI for surfaces, spotOmega for spots); the
    area/sphere branch keeps the exact `emitIntegral·area·PI` expression so those
    renders stay bit-identical (verified: cornell FTSL==C++==pre-3c hash).
- **Constant environment done (2026-07-10, increment 1a):** `light env { spd … }`
  registers a uniform infinite emitter (`shape = EmitterShape::Env`,
  `geomWeight = envGeom = 4·π²·R²` with `R` the scene bounding-sphere radius set in
  `Scene::build()` from the BVH root AABB). Forward emission spawns each photon from
  a disk of radius `R` perpendicular to a uniformly-sampled sphere direction (joint
  pdf `1/envGeom` → exactly analog `beta = emitIntegral·envGeom`); backward adds
  `L(λ)·invPdfλ` on ray-miss; a per-pixel background pass (`addEnvBackground`) supplies
  the directly-viewed sky in forward mode B. Validated by `scenes/envlight.ftsl`
  (mode V: forward converges to backward on a **unit** radiance scale — best-fit
  s→1).
- **GPU constant env done (2026-07-10, increment 1b):** the device forward kernel
  now emits env photons (`DEmitter::shape == 3`) from the scene bounding sphere
  (`DScene::sceneCenter`/`sceneRadius`) exactly like the CPU path, and the
  directly-viewed sky is added by the backend-agnostic `addEnvBackground()` pass in
  `main.cpp` — so `cudaForwardSupported()` no longer rejects `envIndex ≥ 0` and env
  scenes run on the GPU. Verified CPU==GPU on `envlight.ftsl` mode V (both best-fit
  s≈0.97, absorbed≈0.129, RMSE≈58% at 8M — deltas are independent RNG streams).
  - **Absolute-radiance We fix (same change):** the model-B pinhole importance was
    normalizing by the *whole* image-plane area (`imagePlaneArea()`), making the
    forward tracer measure `radiance / (resX·resY)` — an arbitrary global constant
    that modes V/P best-fit away and auto-exposure hid. This blocked compositing the
    (true-radiance) env background with the (scaled) photon surface illumination.
    Fixed by normalizing by the **per-pixel** image-plane area
    (`Camera::pixelPlaneArea() = imagePlaneArea()/(resX·resY)`) in `connect()` /
    `connectVolume()` on **both** CPU (`render.h`) and GPU (`render_cuda.cu`). Now
    forward measures absolute radiance (mode V/P best-fit s → ~1). Displayed outputs
    are unchanged (a global scale is invisible after auto-exposure; mode-P
    `fwd·invF/s` and mode-V RMSE are scale-invariant); verified cornell mode V still
    PASSes (s 5.8e-5 → 0.98) and CPU==GPU film scale holds.
  - **Forward env is high-variance (acceptable limitation):** the env photon
    emission is isotropic over 4π, so in an open scene the vast majority of photons
    escape without hitting geometry (~87% on `envlight.ftsl`). Combined with
    single-wavelength spectral spikes, forward mode-B env images are heavily
    chromatic-noisy and need large `-n` to converge (mode V RMSE falls as 1/√N with
    s≈1 — variance, not bias: 58%@8M → 27%@60M). Clean env images come from the
    **backward** reference (mode R). A future variance reduction would importance-sample
    the emission toward the actual geometry (not just the bounding sphere) and/or
    trace multiple wavelengths per photon (hero-wavelength); deferred.
  - **Mode P + env: no sky background (minor gap).** The directly-viewed sky is
    supplied by `addEnvBackground()`, which is wired into modes B and V but *not*
    into `renderComposite()` (mode P). An env scene rendered in mode P therefore
    shows the environment illumination on surfaces but a black background on the
    forward-side (diffuse) pixels. Mode P targets specular scenes (its whole purpose
    is the camera-side layer for S* paths), so env + mode P is niche; the proper fix
    (add the background to the composite's forward layer, mindful of the best-fit
    `s` interaction — now easy since the We fix makes s≈1) is deferred.
- **Image-based HDRI environment (2026-07-10, increment 2a — DONE, CPU):** `light env
  { file "sky.hdr"  rotate deg  intensity s }` registers an equirectangular (lat-long)
  environment. `src/envmap.h` (`EnvMap`) loads the map (via the existing `Texture`
  loader — `.hdr`/`.pfm`/LDR), upsamples each texel to a physical emission spectrum
  `L(λ) = scale·S_JH(chroma)(λ)·illum(λ)` (Jakob-Hanika chroma fit × normalized 6504 K
  illuminant, PBRT RGB-illuminant convention; `scale` carries HDR brightness), and
  builds a 2D luminance CDF (`Distribution2D`: marginal rows × conditional cols,
  `sin θ`-weighted) for importance-sampled directions. Wired through `Scene`
  (`envMap` shared_ptr, `addEnvLight(map)`, direction-dependent `envRadiance(dir,λ)` /
  `envXYZForDir(dir)` / `sampleEnvDir`), `render.h` (forward emission importance-samples
  the direction and reweights the flat power by `L(dir,λ)/(4π·pdf_ω·meanSpd(λ))` — a
  factor that is exactly 1 for a constant env, so those stay bit-identical), `backward.h`
  (miss term uses the escape direction), `main.cpp` (`addEnvBackground` uses per-texel
  spectral XYZ), and `ftsl.h` (`file`/`rotate`/`intensity` parse). The emitter power +
  wavelength CDF use the map's `sin θ`-weighted mean radiance spectrum. Validated by
  `scenes/envmap.ftsl` + `scenes/sky.pfm` (mode V: best-fit s→~0.95 and climbing with
  samples — the residual is Monte-Carlo variance from the sun glow, not bias;
  forward/backward auto-exposure agree to ~3%; energy conserves). Constant env
  (`envlight.ftsl`) stays **bit-identical** (mode-V scale 0.971252, unchanged).
  - **Increment 2b — DONE (2026-07-10):** backward env **NEE** at every diffuse and
    fog-scatter vertex (`neeEnv`/`neeEnvVolume` in `backward.h`): sample `ω` from the
    map's luminance CDF, shadow-ray past the scene bounds, and **MIS-combine** (balance
    heuristic) with the BSDF-sampled continuation that reaches the sky on a ray miss.
    The miss term is added at full weight only on a camera/specular arrival and MIS-
    weighted otherwise (gated on `specularArrival`), so nothing is double-counted;
    `envMap->pdf(d)` provably equals `sample()`'s reported pdfW, so the weights sum to
    1 (unbiased — verified: `envmap.ftsl` mode-V scale stays ~0.947 at 60M/512, energy
    conserves, residual broadly distributed). All env-NEE work is gated on
    `scene.envIndex >= 0`, so non-env scenes keep a **bit-identical** RNG stream /
    backward image (cornell mode V unchanged).
  - **Increment 2c — DONE (2026-07-10):** GPU port of the lat-long sampler
    (`render_cuda.cu`). The host flattens the EnvMap into device buffers — per-texel JH
    `coeff`/`scale`, the mean `avgCoeff`/`avgScale`, and the 2D luminance CDF (marginal
    `Distribution1D` over rows + one conditional per row) — and the device gets
    `dReflAt`/`dSample1D`/`dEnvSample`/`dEnvTexel` so the `shape==3` emission branch
    importance-samples the map and reweights beta by `L(dir,λ)/(4π·pdfW·avgSpd)`. The
    reweight's shared illuminant cancels in `L/avgSpd`, so no illuminant table is
    uploaded. `cudaForwardSupported()` now returns true for image env; the constant-env
    device path is untouched (`sc.env.scale == nullptr`). Verified: GPU vs CPU forward on
    `envmap.ftsl` agree — energy conserves (sum/emitted=1.0, escaped 0.8893 vs 0.8894),
    mean RGB within ~0.5%, auto-exposure 53.5 vs 53.8; constant env + all other GPU
    scenes unchanged.
- **Deferred (still future):**
  1. **HDRI env** — image-based environment lighting (`light env { file … }`) is fully
     done: increments 2a (CPU forward + backward miss/background), 2b (backward env-NEE
     with MIS), and 2c (GPU forward port) are all complete. Original 7-step plan below,
     all steps done, kept for reference:
     **Concrete plan (each sub-step
     independently
     buildable + validatable):**
     1. *Scene bounding sphere.* Add `Vec3 sceneCenter; double sceneRadius;`
        computed in `Scene::build()` from the BVH root AABB (`center`, `0.5·diag`).
        The env disk/emission and the "to infinity" shadow-ray length key off this.
     2. *Env data + importance sampler.* Store the lat-long map as linear RGB +
        per-texel JH coeffs (reuse `Texture`). Precompute a 2D luminance CDF
        (marginal over rows, conditional over columns, each row weighted by
        `sin θ`) → `sampleEnvDir(u1,u2, pdfω)` and `envPdf(ω)`. `envRadiance(ω,λ)`
        = `reflAt(coeff(ω), λ)` scaled by an intensity factor.
     3. *Backward first (easiest to validate in mode R).* In `radiance()`, on
        `!h.valid` return `L + thr·envRadiance(ray.d,λ)·invPdfLambda` when
        `specularArrival` (direct/mirror view of the sky). Add env NEE at diffuse
        vertices: sample `ω~envPdf`, shadow-ray to `sceneRadius`, add
        `f·envRadiance(ω,λ)·cosSurf·invPdfLambda/envPdf(ω)`. Fold the env into the
        combined wavelength sampler `g(λ)` (its geomWeight ≈ `π·sceneRadius²·avgLum`).
     4. *Forward emission.* New branch in `tracePhoton`: `shape == Env` emits a
        photon FROM the sky — importance-sample `ω~envPdf`, pick a point on the disk
        of radius `sceneRadius` perpendicular to `-ω` tangent to the bounding sphere,
        fire along `-ω`, `beta = envPower · envRadiance(ω)/(avgLum·envPdf(ω))`.
        `envPower = π·sceneRadius²·∫envRadiance dω` feeds the selection CDF like any
        other emitter's `power`.
     5. *Mode-B background.* In forward mode B a camera ray isn't traced, so the sky
        isn't directly visible via `connect()`. Add a per-pixel background pass:
        for each pixel, project the pinhole ray, and if it escapes, splat
        `envRadiance(dir,λ)` (spectrally integrated) — a cheap deterministic add,
        analogous to the existing direct-emitter `connect`.
     6. *CUDA.* Upload the env RGB+coeff tables + the marginal/conditional CDFs;
        port `sampleEnvDir`/`envPdf`/`envRadiance` and the forward disk-emission
        branch. Escaped photons already terminate; only emission + (optional) the
        mode-B background pass need device code. Keep `cudaForwardSupported()`
        returning true for env scenes once ported (else fall back to CPU).
     7. *Validation.* `scenes/envlight.ftsl` (a diffuse box open to the sky):
        mode V forward-vs-backward RMSE < 5%; energy conserves; CPU==GPU. A constant
        (single-colour) env is the smallest first milestone — it exercises steps
        1/3/4/5/6 with a trivial step 2 (uniform pdf), so land that before the full
        image-based 2D CDF.
  2. **Sphere-light importance sampling.** *(DONE 2026-07-10.)* The backward
     reference's sphere NEE now does cone/solid-angle importance sampling of only
     the visible cap toward the receiver (`Emitter::sampleSphereCone`, PBRT's
     `Sphere::Sample`): sample `cosθ` uniformly in `[cosθmax, 1]` about the
     centre-to-point axis (`sinθmax = r/dc`), find the near intersection, and weight
     in solid-angle measure with `pdfW = 1/(2π(1−cosθmax))` — so no draws land on
     the far, self-occluded, back-facing hemisphere. A receiver inside the sphere
     (`dc ≤ r`) falls back to uniform `samplePoint`. Applies to both surface
     (`neeLight`) and fog-vertex (`neeVolume`) NEE. Quad lights are untouched and
     keep a bit-identical RNG stream. Validation: `spherelight.ftsl` mode V best-fit
     scale → 0.9997 (unbiased) at 80M/1024 spp, RMSE 2.5% bulk; sphere+`-fog 0.5`
     scale 0.988; cornell (quad) unaffected. Only the backward reference changed —
     the forward tracer emits sphere photons omnidirectionally as before, so the
     GPU/CUDA path is unaffected.
  3. **Spot penumbra sampling.** The forward spot samples uniformly in the outer
     cone then reweights by falloff, so photons in the dark penumbra edge carry
     small weights (mild variance). Exact CDF sampling of the smoothstep band would
     be lower-variance but needs a quartic inverse; uniform+reweight is correct.
- **Status:** OPEN (acceptable) — sphere + spot done 2026-07-10; **constant
  environment (`light env { spd … }`) done 2026-07-10 (increments 1a CPU + 1b GPU)**
  incl. the absolute-radiance We fix and on-device env emission; **image-based HDRI
  (`light env { file … }`, 2D luminance CDF + per-texel JH spectral upsampling) fully
  done 2026-07-10 (increments 2a CPU forward+backward miss/background + 2b backward
  env-NEE with MIS + 2c GPU forward port)**; **sphere-light cone importance sampling
  in the backward reference done 2026-07-10**; only spot penumbra CDF sampling (item
  3) still deferred.

### Full physical `layered` material not yet implemented (`mix` is)
- **What:** the FTSL `type mix` material (stochastic per-photon pick among named
  child materials, weights ≤ 1, remainder absorbs) is implemented and validated
  (Phase 2d — `scenes/mixmat.ftsl`, mode V PASS, CPU==GPU). The richer physical
  `layered` material from the spec (§3.2) — a Fresnel/Airy-weighted specular *coat*
  over a weighted *body* of diffuse/transmit/subsurface/fluorescent lobes with
  energy-consistent coat↔body coupling — is **not** built yet.
- **Why acceptable:** `mix` covers the "blend two finished materials" use case with
  the same unbiased lobe-selection machinery; `layered` adds physically-correct
  interface/substrate coupling (the transmitted fraction enters the body, internal
  reflection, etc.) which is a larger transport change. The spec documents it as the
  preferred long-term form.
- **Constraints of `mix` (by design):** children must be non-mix materials (nesting
  rejected by the parser to keep resolution single-step and the CUDA CDF bounded);
  the CUDA path supports ≤ 8 child lobes (more → CPU fallback); a mix containing a
  fluorescent child is forward-only and CPU-only (same fluorescence restriction as
  the standalone type — see below).
- **Proper fix (future):** implement `layered` as a coat interface (reuse
  thinfilm/Fresnel reflect-or-enter) feeding a body lobe selector, with the body's
  transmitted radiance re-emerging through the coat. Forward-first; backward support
  follows the same per-lobe pattern except for fluorescent bodies.
- **Status:** OPEN (acceptable) — `mix` done 2026-07-10; `layered` deferred.

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

### GPU backend (`-device gpu`) covers all three forward camera models (A/B/C)
- **What:** the CUDA backend (`src/render_cuda.cu`, `renderForwardCuda`) implements
  the three forward camera models — A (contact-sensor deposit), B (connect/splat to
  the pinhole), and C (finite-aperture thin-lens forward catch) — selected by the
  `camMode` parameter. It is used for `-mode A/B/C` and the forward pass of `-mode V`.
  It still falls back to the CPU for mode R (backward reference) and the mode-P
  camera-side/backward layer (no backward tracer on-device). Fluorescent scenes are
  rejected on-device (fall back to CPU) because the emission-sampler reradiation
  path is not ported — `cudaForwardSupported()` checks whether any *geometry* uses a
  Fluorescent material (not just the palette, which buildCornell always populates).
- **Why acceptable / validated:** model B is the default and the one mode V
  validates. The kernel `kTrace` mirrors `Renderer::tracePhoton` exactly and gates the
  camera-specific work on `camMode`: emitter→pinhole connect, in-scatter
  `connectVolume`, and diffuse-vertex `connect` run only for B; `catchPhoton` (thin
  lens `u' = u - rho/f`) runs only for C; the sensor-plane `deposit` runs only for A.
  Validation vs CPU (Cornell, 128²):
  - **Mode B:** image RMSE ≈ 0.85/255 at 200M photons (pure MC noise); `-mode V
    -device gpu` PASSes vs the backward reference (bulk RMSE 4.17% ≈ CPU 4.22%);
    ~14× speedup (400M @256²: 153s CPU → 10.9s GPU on an RTX 4090).
  - **Mode A:** energy report matches to 4 sig figs (sensor 0.3298 vs 0.3299); image
    RMSE scales as √N — 11.18/255 @40M → 5.11/255 @200M (5× photons, ideal 2.24×,
    measured 2.19×), proving variance not bias.
  - **Mode C:** energy report matches to 4 sig figs; with a wide aperture (0.25,
    focus 2.2) the caught fraction matches exactly (sensor=0.0058) and per-image
    auto-exposure agrees (1.60e-8 vs 1.59e-8). Image RMSE scales as √N —
    15.70/255 @200M → 8.10/255 @800M (4× photons, ideal 2×, measured 1.94×) —
    proving variance not bias. (The CPU is deterministic across runs, so GPU is the
    only independent noise realization; the small default aperture is catch-starved
    and its tone-mapped RMSE is dominated by per-image auto-exposure.)
- **Spectral baking:** device materials/fog sample each `std::function` Spectrum into
  a fixed 96-entry table over [360,830] nm with linear interpolation (`SPEC_N=96`).
  Smooth reflectances/Sellmeier indices make this accurate to within MC noise; a
  pathologically spiky spectrum would need a finer table. CIE CMFs are ported
  analytically (no table).
- **Precision (mixed FP32/FP64, default float transport):** consumer GeForce GPUs run
  FP64 at ~1/64 the FP32 rate, so the megakernel computes all geometry/BRDF/spectral
  transport in a compile-time `Real` scalar (`float` by default) while accumulating the
  film and energy in `double` (`atomicAdd` on `double*`). Build with
  `-DFTRACE_GPU_FP32=OFF` for a full-FP64 device path (bit-closer to the CPU, far slower
  on GeForce; sensible on datacenter cards or for precision debugging). The CPU renderer
  is always `double` and remains the ground-truth. Float-safe self-intersection epsilons
  (`RAY_EPS=1e-4`, `DET_EPS=1e-6`) replace the FP64 `1e-6`/`1e-9`. **FP32 validated vs
  FP64 CPU (Cornell, RTX 4090):** energy conserves exactly (`sum/emitted=1.000000`,
  residual=0 on A/B/C/V — no self-intersection leak from the float epsilons); fractions
  converge to 4 sig figs (mode A sensor 0.3298 vs 0.3301, mode C 0.0059 vs 0.0058); mode
  V PASSES (bulk RMSE 2.89%, firefly-dominated); ~14× faster than FP64 (400M @256² in
  0.76s vs 10.9s). The DVec3 3-arg ctor deliberately keeps `double` params so host
  brace-init from `double` Scene coords is a widening (legal) conversion, never
  narrowing; spectral/CDF tables stay `double` (host-baked, tiny, cached).
- **Portable build (multi-arch) + HIP-ready:** `-DFTRACE_CUDA_ARCH=` selects the device
  arch set — `native` (default; the local GPU only, fast builds), `all-major` (a
  redistributable fat binary: one cubin per major arch + forward-compatible PTX so newer
  GPUs JIT at load), `all`, or an explicit `"75;86;89"` list. The device kernel is
  written in the portable CUDA/HIP subset (`__global__`/`__device__`, grid-stride, double
  `atomicAdd`, `<<<>>>` launches); the only vendor-specific surface — the host runtime API
  (device query, malloc/memcpy/memset/free, error strings, synchronize) — is isolated
  behind a compat block at the top of `render_cuda.cu` that maps `cuda*` → `hip*` under
  `-DFTRACE_USE_HIP`/`__HIP_PLATFORM_AMD__`. Porting to AMD ROCm is therefore a
  build-system change (compile this one file with `hipcc`), not a code rewrite. **CUDA is
  the supported GPU backend today; HIP is a near-drop-in future target (untested — no AMD
  hardware here).**
- **Proper fix (future):** port the backward tracer (modes R and the mode-P
  camera-side layer) to CUDA if those paths ever become the bottleneck; add a device
  fluorescence path (bake `fluoEmitSampler`'s CDF) to lift the fluoro restriction.
- **Status:** OPEN (acceptable) — logged 2026-07-10; A/C, mixed-precision FP32, portable
  multi-arch build, and the HIP compat layer added same day. Requires a CUDA toolkit at
  configure time; without one the project builds CPU-only and `-device gpu` warns and
  uses the CPU.

### GPU scaling path (future): megakernel vs. wavefront
- **Context:** the current GPU backend is a **megakernel** — one `kTrace` launch where
  each thread runs an entire photon path (emit → bounce loop → connect/catch/deposit)
  start to finish. This is the right choice for *this* renderer today: an RTX 4090 has
  huge register/occupancy headroom, the Cornell-class scenes are shallow, and a single
  kernel keeps all state in registers with no round-trips to global memory. It already
  hits ~500M+ photons/s in FP32.
- **The known limitation (thread divergence):** in a megakernel, threads in a warp that
  take different material branches (a dielectric refraction next to a diffuse bounce next
  to a grating), or that terminate after wildly different path lengths, **serialize** —
  the warp runs at the speed of its slowest/most-divergent lane, and finished lanes sit
  idle while others keep bouncing. The megakernel also carries the register footprint of
  *every* material's code path in *every* thread, capping occupancy. Both effects get
  worse as scenes gain more material variety and deeper paths, and they bite harder on
  smaller GPUs (fewer SMs / less latency-hiding to absorb the idle lanes).
- **The alternative (wavefront / path-regeneration):** split the tracer into stages —
  generate, extend (intersect), shade-per-material, connect — each its own kernel, with
  photon state held in global "ray queues" between stages. A **sort/compaction by
  material** before the shade stage makes each shading kernel branch-coherent (every
  thread in a warp runs the same BSDF), and terminated paths are **compacted out** so
  every thread always has live work (path regeneration keeps the SIMD lanes full). This
  is how production GPU renderers (PBRT-v4's `wavefront`, OptiX path guiding) scale to
  many-material, deep-path scenes. The cost: extra global-memory bandwidth for the queues
  and more kernel-launch overhead, which is why it's *not* a win for shallow, uniform
  scenes on a big GPU (the megakernel's register-resident state wins there).
- **Re: "wavefront helps divergent scenes AND small GPUs" (todo.txt question):** it's
  *both*, and they're related. (1) *Divergent scenes* — many materials and/or highly
  variable path lengths — benefit from the per-material sort (kills branch divergence) and
  compaction (kills path-length divergence). (2) *Small GPUs* benefit because they have
  less occupancy/latency-hiding headroom to paper over idle lanes and high per-thread
  register pressure, so keeping warps coherent and full matters more there. A big GPU on a
  shallow uniform scene (our current case) is the one regime where the megakernel clearly
  wins, which is why we ship it first.
- **Decision:** keep the megakernel as the default and recommended path for the scenes
  this renderer targets. Add a wavefront backend only if/when profiling on a genuinely
  material-diverse, deep-path scene (or a small GPU) shows the megakernel is
  divergence-bound. Documented here so the scaling path is on record; no code owed now.

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
