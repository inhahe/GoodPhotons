# GPU backward parity + fast RGB backward — plan & decisions

Working doc for the initiative to (1) give the GPU backward tracer (mode R,
`renderBackwardCuda` / `bkRadiance` in `src/render_cuda.cu`) full feature parity
with the CPU backward tracer (`renderBackward` in `src/backward.h`), (2) add a
**fast RGB (non-spectral) backward renderer** whose original purpose is *speed*,
and (3) add scene-ignore flags (like the rasterizer) to skip expensive features.

## Motivation

The backward tracer was meant to be the *fast* previewer, but two things make it
slow today:

- **Single-wavelength spectral sampling.** Each backward sample traces ONE
  wavelength and deposits `CIE(λ)·L` into the film, so a clean *color* image needs
  many samples (noise ∝ 1/√spp, plus the spectral dimension on top).
- **CPU fallback on real scenes.** `cudaBackwardSupported` rejects participating
  media, environment light, fluorescence, and spot/collimated emitters, so any
  scene using them (e.g. `gallery_settled` with its haze/isosurface media) drops to
  the CPU tracer — measured ≫1 min/spp, unusable interactively.

## Decisions

### RGB representation: **Option B — RGB throughput** (chosen 2026-07-23)

The fast renderer carries an **RGB throughput triple** (not a single wavelength).
Each material's reflectance is reduced to a **precomputed per-material RGB albedo**
once at scene build (integrate the spectral reflectance under the colour basis), so
a path does ONE intersection walk and produces a **full-colour** result every
sample. This is the fastest, most POV-Ray-like path.

- **Chosen over Option A (3-wavelength combine)** because the whole point is speed;
  A would trace ~3 wavelengths/path (≈3× intersection cost) to keep dispersion.
- **Accepted tradeoff:** RGB-throughput drops true dispersion, thin-film
  iridescence, and spectral fluorescence — the same spectral effects the ignore
  flags would drop anyway. Scenes needing physically-exact spectra use the spectral
  backward (mode R) or the reference modes (D/M/P).

## Stages

1. **GPU backward feature parity** — port into `bkRadiance` / `renderBackwardCuda`,
   reusing existing device code from sibling kernels:
   - Participating media (homogeneous + heterogeneous): reuse the GPU **BDPT**
     kernel's device delta-tracking / ratio-tracking transmittance in the backward
     walk (free-flight collision vs. surface hit, phase-function NEE, Beer-Lambert).
   - Fluorescence: the GPU **forward** path already uploads per-material
     fluorescence CDFs (`fluoCdfAll`); add the Stokes-shift vertex strategy.
   - Environment light: env-miss radiance + env-NEE at each vertex.
   - Spot / collimated / env emitters: add their sampling + pdf to backward NEE.
   - Relax `cudaBackwardSupported` per-feature as each lands (feature-flagged so
     partial progress stays shippable and testable vs. the CPU reference).

2. **Fast RGB backward path** — Option B above. New non-spectral device path.

3. **Scene-ignore flags** (rasterizer-style): `-no-media`, `-no-env`, `-no-fluoro`,
   `-max-bounce N`, `-direct-only` (Whitted: direct + specular recursion only, no
   diffuse indirect — converges in ~1 spp). A scene-sanitize/param pass usable in
   both batch renders and `-explore`.

4. **Wire fast backward into `-explore`** — toggle so the interactive viewer uses
   RGB backward + ignore-flags, converging live (the original motivation).

## Status

- [~] Stage 1: GPU backward feature parity
  - [x] **1a — participating media (homogeneous + heterogeneous).** Ported into
    `bkRadiance`: free-flight collision competes with the surface hit
    (`dMediaSampleCollision`), volume NEE (`bkNeeVolume`, HG-phase-weighted, skips
    beam/spot/env emitters), Beer-Lambert on surface NEE + throughput via
    `dMediaTransmittance`, albedo-absorption Russian roulette, HG scatter
    (`sampleHG`). `cudaBackwardSupported` now accepts media (rejects only GRIN and
    rainbow/dispersive media, which stay CPU). **Validated** vs CPU reference on
    `_fog_cornell` (g=0) and `_fog_cornell_g` (g=0.6, exercises HG sign): raw film
    radiance (from `.ftbuf`, so no auto-exposure confound) agrees to **0.1%** in
    absolute luminance, per-channel XYZ within 0.4%, and the scale-normalized
    block-averaged structural residual falls monotonically toward zero
    (1.75%→1.02%→0.56% at 8/16/32-px blocks) = noise only, no bias. Compare tools:
    `scraps/cmp_ftbuf.py` (raw radiance) and `scraps/cmp_fog.py` (tone-mapped PNGs).
  - [x] **1b — fluorescence (bispectral Stokes-shift adjoint).** Ported the
    `D_FLUORESCENT` branch into `bkRadiance` (was falling through to plain diffuse):
    elastic base NEE at the output wavelength, a separately-sampled excitation
    wavelength `lambdaIn` (`dSampleSceneLambda` + `dInvPdfLambda`, the machinery the
    initial wavelength already uses), the fluoro direct-NEE weighted by
    `gOut = (M(lambda)/Mint)*invPdf` and `rhoFluo = min(eps(lambdaIn),1-rho)*yield`,
    then a stochastic elastic/fluoro/absorb continuation with the Stokes-shift
    wavelength switch. Added `fluoEmitSpec[SPEC_N]` (baked continuous emission SPD)
    + `fluoMint` to `DMaterial` so the adjoint can evaluate `M(lambda)` at a *fixed*
    output wavelength (the forward path samples from the CDF where `M/pdf` cancels).
    `cudaBackwardSupported` now accepts fluorescence. **Validated** vs CPU on
    `_fluo_cornell` (blue-excited green-emitting dye sphere): raw film radiance
    agrees to **0.02%** in absolute luminance, per-channel XYZ within 0.1%, block
    residual 0.34%→0.19%→0.09% (noise only).
  - [x] **1c — environment light (constant env: env-miss + env-NEE, MIS'd).** Ported
    into `bkRadiance`: `bkNeeEnv` (surface-vertex env-NEE, uniform-sphere sample pdf
    1/4π, shadow-terminator gate, `dMediaTransmittance` to the scene exit, balance-
    heuristic MIS vs the cosine continuation) called at diffuse / diffuse-transmit
    (both lobes) / fluoro (elastic + fluoro-direct) vertices; `bkNeeEnvVolume` (fog-
    vertex env-NEE, albedo·HG-phase in place of BRDF·cos, HG-phase MIS) at media
    collisions; env-miss at ray escape (full weight on specular/camera arrival,
    else MIS-weighted against the previous vertex's env-NEE via a tracked
    `contBsdfPdf` = cosine pdf after diffuse/fluoro, HG-phase pdf after a fog
    scatter). Added `int envIndex` to `DScene` and the env geomWeight (4π²R²) to
    `dInvPdfLambda`. `cudaBackwardSupported` now accepts a **constant** env (image env
    → CPU) and no longer rejects the env-shape emitter. **Validated** vs CPU on
    `_env_cornell` (open-top Cornell, constant d65 sky, no area light): raw film
    radiance agrees to **0.17%** in absolute luminance, per-channel XYZ within 0.3%,
    block residual 1.35%→0.91%→0.37% (noise only); and on `_env_fog` (same + g=0.6
    haze, exercises `bkNeeEnvVolume` + HG-phase env-miss MIS): **0.23%** luminance,
    XYZ within 0.3%, block residual 1.82%→1.10%→0.49%. Compare tool
    `scraps/cmp_ftbuf.py`.
  - [~] **1d — spot / collimated / image-env emitter sampling + pdf in backward NEE.**
    - [x] **Point-spot lights.** Ported into `bkNeeLight` (surface) and `bkNeeVolume`
      (fog vertex): deterministic connect to `em.origin`, `spotFalloff` cone weight,
      Beer-Lambert shadow transmittance, no rng draw (matches the CPU `emitterGeom` /
      `neeVolume` spot branch). Added the spot geomWeight (`spotOmega`) to
      `dInvPdfLambda` and relaxed the gate to accept spot emitters. **Validated** vs CPU
      on `_spot_cornell` (single down-aimed spot, cone 18°/32°): raw film radiance
      agrees to **0.17%** in absolute luminance, per-channel XYZ within 0.2% (no scale
      bias → the spotOmega weight + falloff are exact). Structural residual is noisier
      (4.1%→2.4%→1.4% at 8/16/32 px) because a spot casts a hard-edged high-contrast
      pool, but falls monotonically = noise only.
    - [ ] **Collimated beams** — intentionally kept on the CPU: a zero-solid-angle beam
      is not NEE-samplable (the CPU tracer also skips it in NEE), so a backward path can
      essentially never connect to it; the gate still routes collimated scenes to CPU.
    - [ ] **Image-based environment** (lat-long map): env-NEE via the luminance CDF
      (`dEnvSample`), image env-miss with `envPdfDir`, and the per-texel radiance
      reweight. Deferred; a constant env already works (1c).
- [x] **Stage 2: fast RGB backward (Option B).** New non-spectral device path
  (`renderBackwardRGBCuda` / `kBackwardRGB` / `bkRadianceRGB` in `src/render_cuda.cu`),
  selected by `-rgb` on mode R when the GPU backward is eligible
  (`cudaBackwardRGBSupported`). Carries a `DVec3` linear-RGB throughput `beta` and
  accumulator `L` instead of a single wavelength, so one intersection walk produces a
  full-colour sample. Baking (at scene build, host side, `namespace rgbbake`):
  - **Emitter/env radiance** → `xyzToLinearSrgb(∫CIE(λ)·emitSpd(λ)dλ)` (`DEmitter.rgbEmit`,
    `DScene.rgbEnv`). No area/invPdf term — the spectral estimator's `p(λ)·invPdf = 1`
    regardless of emitter geometry, so a directly-viewed emitter's film XYZ is exactly
    that integral; the geometry factors stay in NEE.
  - **Material reflectance** → linear-RGB albedo under equal-energy white, clamped per
    channel to [0,1] (`reflToRgb`; flat white → (1,1,1)). Stored as `rgbAlbedo` /
    `rgbTransmit`; colored glass keeps a 3-tap Beer-Lambert `rgbAbsorb` (σ_a at
    610/550/465 nm) and `rgbIor = ior(550)`.
  - **Deposit** converts `L` (linear-RGB) → XYZ via `dRgbToXyz` (inverse of
    `color.h`'s `xyzToLinearSrgb`) before adding to the film.
  - **Achromatic specular** uses a representative `LREP_RGB = 550 nm` for Fresnel /
    refraction via the existing `dDielectricStep`; **RGB Russian roulette** survives
    with `q = rgbLuma(albedo)` then `beta = hadamard(beta, albedo)/q` (unbiased,
    colored, lower variance than the spectral RR).
  - **Scope gate** (`cudaBackwardRGBSupported`): forward-eligible, no GRIN, **no media**
    (deferred), no image-env, no collimated/env-shape emitters, only material types
    Diffuse/Dielectric/Mirror/HalfMirror/Glossy/Mix/DiffuseTransmit/Filter, no textured
    or record-driven reflectance (deferred). Ineligible scenes fall back to the spectral
    backward with a warning.
  - **Validated** vs the spectral backward on `scenes/_rgb_neutral.ftsl` (flat-spectrum
    Cornell: reflectances `0.75`/`0.5`, equal-energy-round-tripping light — no
    metamerism, so the RGB path must match the spectral estimator's *absolute*
    luminance): raw film radiance (from `.ftbuf`, no auto-exposure confound) agrees to
    **0.07%** in mean luminance (ratio B/A = 1.0007), per-channel XYZ within 0.3%
    (X 1.0005 / Y 1.0007 / Z 1.0034), block residual 1.38%→1.04%→0.41% at 8/16/32 px
    (monotonic → noise only, no bias). Colored scenes carry the accepted Option-B
    metamerism approximation (no dispersion / thin-film / spectral fluorescence).
    Compare tool `scraps/cmp_ftbuf.py`.
  - **Deferred follow-ups:** participating media in the RGB walk (gated out); textured /
    record-driven albedo in RGB (gated out). Logged in `known-issues.md`.
- [x] **Stage 3: scene-ignore flags.** Rasterizer-style knobs to strip expensive scene
  features so the fast RGB (or spectral) backward runs faster. Two kinds:
  - **Scene mutations** (`Scene::applyIgnoreFlags(noMedia, noEnv, noFluoro)` in
    `src/scene.h`, applied host-side once after scene load, before any upload):
    - `-no-media` / `-nomedia` — clears `media` (drops all participating-media volumes;
      also un-gates the RGB fast path, since media are a fast-path gate).
    - `-no-env` / `-noenv` — erases the environment emitter (`emitters[envIndex]`), resets
      `envIndex=-1` / `envMap` / `envXYZ=0`, then `finalizeEmitters()` to rebuild the
      emitter CDF/sampler. Safe because `envIndex` is the only persistently-stored emitter
      index.
    - `-no-fluoro` / `-nofluoro` — demotes every `MatType::Fluorescent` material to
      `MatType::Diffuse` (falls back to the elastic `reflect` albedo — correct by design).
    Each strip prints `[ignore] stripped: <summary>` (e.g. `1 medium/media`,
    `environment light`, `1 fluorescent material`).
  - **Render params** (threaded via globals `g_maxBounceOverride` / `g_directOnly` in
    `main.cpp`, mirroring the `g_heroC` pattern; on GPU carried as `DScene.bkMaxBounce` /
    `bkDirectOnly`, set on the `renderBackward*Cuda` wrappers since `render_cuda.cu` is a
    separate TU):
    - `-max-bounce <N>` — caps path depth (`BackwardRenderer::maxBounce`, forward
      `Renderer::maxBounce`, and GPU `bkMaxBounce`). `<0` = tracer default (32).
    - `-direct-only` / `-directonly` — Whitted mode: after a **non-specular** vertex
      (Diffuse / DiffuseTransmit / elastic-Fluorescent / fog single-scatter) does its
      direct-lighting NEE, terminate — no diffuse indirect continuation. Specular chains
      (mirror / dielectric / glossy / filter) still recurse. **Scoped to the camera path
      tracers** (backward R spectral + RGB, P's backward side); forward B and the
      photon/bidirectional modes (M/S/D) honour `maxBounce` but ignore `directOnly`.
  - **Validated:** all three strip flags print the correct `[ignore] stripped:` summaries
    and render without crashing; `-direct-only` produces textbook Whitted images on **both
    GPU and CPU** (no colour bleeding, black shadows, dark ceiling) — verified on the
    Cornell box (`png/_direct_cornell.png` GPU, `png/_cpu_direct.png` CPU); `-max-bounce`
    parses and caps depth.
- [x] **Stage 4: `-explore` integration.** The interactive fly-viewer (`-explore`/`-fly`)
  can toggle a **live path-traced preview** with the key **`T`**, swapping the flat raster
  still for a progressively-converging fast-RGB backward render:
  - **Resident session** (`BackwardRGBSession` in `src/render_cuda.cu`, decls in
    `render_cuda.h`): `backwardRGBSessionBegin` runs `buildUploadScene` ONCE (scene bake is
    the expensive part) and keeps a persistent double-precision SUM film; `...SetCamera`
    re-bakes only the cheap POD camera (`bakeCamera`, no per-aim alloc for a pinhole) and
    zeroes the film; `...Accumulate(spp)` launches the same validated `kBackwardRGB` kernel
    adding a batch into the resident film (advancing `sampleBase`, fixed `kSppCap = 2^22` so
    the per-pixel RNG streams stay unique across the whole idle convergence); `...Download`
    fetches the running SUM; `...End` frees. The Stage-3 knobs bake in via `bkMaxBounce`/
    `bkDirectOnly` on begin().
  - **Viewer wiring** (`src/main.cpp` interactive loop; `T` key added through
    `livewindow.{h,cpp}` → `NavInput::toggleTrace`): while the camera **holds still** the
    loop accumulates a small spp batch each idle tick and presents the converging image via
    the shared `filmToRgb8` (auto-exposure anchor locked per pose so it doesn't flicker as
    noise settles); a camera **move** shows the responsive raster and flags a `setCamera`
    re-aim; the idle-sleep is suppressed while refining so it converges without a busy spin.
    The session is (re)created on a window resize and freed on close.
  - **Availability** mirrors the batch `-rgb` scope exactly (`cudaBackwardRGBSupported`), so
    the viewer prints whether `T` is usable at startup; the **scene-ignore flags** apply for
    free (they mutate the `Scene` host-side before the session bakes it; `-max-bounce` /
    `-direct-only` are passed into begin()).
  - **Validated:** on `scenes/_rgb_neutral.ftsl` (in scope) `T` engages the preview, the
    window title's `spp` climbs and the image converges to a correct neutral Cornell
    (captured live at 1536 spp); `T` again returns to the raster; no crash and the session
    frees on exit. On the built-in `cornell` (out of RGB scope) the viewer correctly reports
    the preview unavailable, matching the batch `-rgb` fallback.
