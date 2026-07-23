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
  - [ ] 1c — environment light (env-miss radiance + env-NEE)
  - [ ] 1d — spot / collimated / env emitter sampling + pdf in backward NEE
- [ ] Stage 2: fast RGB backward (Option B)
- [ ] Stage 3: scene-ignore flags
- [ ] Stage 4: `-explore` integration
