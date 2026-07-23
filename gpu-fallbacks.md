# GPU fallbacks & CPU-only modes — audit

Enumeration of every render mode's CPU fallbacks: features that currently force the
CPU path (or are entirely CPU-only), classified by how portable they are to the GPU.
Source of truth is the `cuda*Supported()` predicates in `src/render_cuda.cu` and the
mode dispatch in `src/main.cpp`. Companion actionable checklist lives in `todo.md`
("GPU fallback closure").

Status legend: **portable** (worth doing) · **portable-hard** (large but feasible) ·
**inherently-CPU** (spectral/geometry reasons; leave on CPU).

---

## Modes with a GPU path + per-feature CPU fallbacks

### Forward A/B/C — `cudaForwardSupported` (render_cuda.cu:6281)
| Feature | Why CPU today | Class |
|---|---|---|
| Indexed-spectral palette maps | device only bakes the Jakob-Hanika coeff path; palette resolves per-texel to a named reflectance spectrum | **portable** (upload per-texel palette spectra) |
| Layered material (coat over weighted body) | no device `shadeStep` Layered branch | **portable** (port the branch) |
| Spectral rainbow-phase media | device volume path only knows the analytic HG lobe; the λ×µ CDF table (`rainbow.h`) isn't uploaded | **portable-hard** (per-λ importance sampling on device); also affects backward + BDPT |
| Oversized multilayer (>`D_MAXLAYERS`) | fixed device cap; CPU has no limit | portable but **low value** (raise cap) |
| Oversized Mix (>`D_MIXMAX` children) | fixed device cap | portable but **low value** |
| Driven record scalar >64 stops | overflows device interp array `vs[64]` | portable but **vanishingly rare** |

### Backward R (spectral) — `cudaBackwardSupported` (render_cuda.cu:7467)
Everything forward rejects, **plus**:
| Feature | Why CPU today | Class |
|---|---|---|
| ~~Image-based env NEE (lat-long map)~~ | **DONE (M1, 2026-07-23)** — added `dEnvRadiance`/`dEnvPdf`, uploaded the illuminant table, wired the device env sampler into `bkNeeEnv`/`bkNeeEnvVolume` + MIS'd env-miss; dropped the `envMap` reject. Validated GPU==CPU to 0.14% at 8192 spp. | ✅ |
| GRIN media | `bkRadiance` has no Eikonal ray-marcher | **portable-hard** |
| Collimated beams / stray Env-shape emitter | not NEE-samplable even on CPU | **inherently-CPU** (low value) |
| Lens deeper than `D_MAXLENS` | fixed device cap | low value |

### Backward R `-rgb` (fast, Option B) — `cudaBackwardRGBSupported` (render_cuda.cu:7545)
Rejects everything spectral-backward rejects **plus**:
| Feature | Why CPU/spectral today | Class |
|---|---|---|
| Any participating media | not yet ported to the RGB walk | **portable** (deferred tech-debt) |
| Textured / record-driven albedo | baked `rgbAlbedo` doesn't capture per-texel/per-hit albedo | **portable** (deferred tech-debt) |
| Dispersion materials (thin-film / grating / multilayer / layered / fluorescence) | effects can't survive an RGB throughput | **inherently-CPU/spectral** (correct as-is) |

### BDPT D — `cudaBdptSupported` (render_cuda.cu:7793)
**M9 (2026-07-23):** the GPU BDPT kernel now stores per-hit texcoords in each `DVertex`
(`u,v`) and reconstructs a `DHit` (`dVertHit`) so its connection BSDF `dBsdfF`/`dBsdfPdf`
(and the random walk) evaluate per-hit-driven throughput slots consistently in BOTH the
sampler and the pdf/eval — MIS-safe. Now **on-device**: textured/patterned/record diffuse
albedo & glossy reflect, per-hit glossy roughness + thin-film thickness maps, mix blend
masks, and Beer-Lambert colored-glass interior absorption (delta vertex → throughput only).
Validated GPU==CPU on `textured.ftsl` (mean 0.06%, background 0.00%, per-pixel diff halves
8.2%→4.3% at 4× spp = unbiased) and `mixmat.ftsl` (mean 0.21%, background 0.12%).
Remaining CPU-only (no device strategy yet):
| Feature | Why CPU today | Class |
|---|---|---|
| Diffuse-transmit vertices | no two-lobe / back-hemisphere connection strategy | **portable** (thread Hit through `dConnect`) |
| Frosted (rough) glass | kernel treats every dielectric as smooth (no microfacet dielectric BSDF) | **portable** |
| Fluorescence | no re-emission vertex strategy | **portable-hard** |
| Spot/env/collimated emitters | no light-subpath strategy for them | **portable-hard** |
| GRIN / rainbow media | straight-segment MIS assumptions / HG-only phase | **inherently-CPU** / rainbow portable-hard |

### Photon map M — `cudaPhotonMapSupported` (render_cuda.cu:7830)
Forward scope plus:
| Feature | Why CPU today | Class |
|---|---|---|
| ~~Any environment light~~ | **DONE (M2, 2026-07-23)** — deposit already emits env photons (indirect); added env's direct term on gather-ray escape in `dPhotonGather` (constant + image via `dEnvRadiance`); dropped the `envIndex >= 0` reject. Validated GPU==CPU mean 0.18%, background 0.04%. | ✅ |
| ~~**Final gather** (`g_pmFinalGather > 0`)~~ | **DONE (M4, 2026-07-23)** — added device `dPhotonGatherSub` (specular walk → one-bounce density query folding `rho(y)*rho(vis)` per photon, + env-on-escape / specular-arrival emitter reflected off the visible point) and a `fgRays>0` branch in `dPhotonGather` (NEE direct via `bkNeeLight` + K cosine-hemisphere sub-rays); threaded `fgRays` through `kGather`→`renderPhotonMapSharedCuda` and dropped the `g_pmFinalGather==0` caller gates (main.cpp meter + flyby). Validated GPU==CPU mean 0.43%, background 0.98%, per-pixel diff √-scales 22%→11.5% at 4× spp (unbiased). | ✅ |
| Physical-lens cameras | caller-gated | secondary |

### Composite P — main.cpp:2534
Camera-side layer falls back to CPU when outside backward-GPU scope. Improves for free
as backward-GPU scope widens (esp. image-env above). No separate predicate.

### Iso preview / `-raster-gpu` — `cudaIsoPreviewSupported` (render_cuda.cu:7705)
Just defers to `cudaForwardSupported`; no independent fallbacks.

---

## Modes with **no GPU path at all** (fully CPU)

| Mode | File | Portability |
|---|---|---|
| ~~**S — SPPM**~~ | `sppm_render.h` | **DONE (M3, 2026-07-23)** — resident device SPPM session (`SppmSession`): per-pixel `tau`/`radius`/`nAcc`/`directSum` stay on-device across passes; each pass reuses the mode-M forward deposit + a per-pixel visible-point/gather/update kernel trio (`kSppmVisiblePoint`/`kSppmGather`/`kSppmResolve`). Wired into main.cpp mode-S (`-device gpu/auto`). Validated GPU==CPU on a Cornell glass-sphere caustic (mean 0.2–1.2%, background 0.3%, per-pixel diff shrinks with passes). ✅ |
| **U — VCM/UPS** (vertex connection + merging) | `vcm.h` | **portable-hard, lowest priority**: needs GPU BDPT correctness *and* photon merging under one MIS weight. |

---

## Scheduled work (greenlit by user 2026-07-23) — quickest wins first

1. ~~**Image-based env NEE in GPU backward** (M1)~~ — **DONE 2026-07-23.** Added `dEnvRadiance`/`dEnvPdf`, uploaded the illuminant table, wired the device env sampler into `bkNeeEnv`/`bkNeeEnvVolume` + MIS'd env-miss; dropped the `envMap` reject. Validated GPU==CPU to 0.14% at 8192 spp. Also unblocks mode P camera-side.
2. ~~**Env term in the mode-M GPU gather** (M2)~~ — **DONE 2026-07-23.** Deposit already emits env photons (indirect); added env's direct term on gather-ray escape in `dPhotonGather` (constant + image env); dropped the `envIndex >= 0` reject. Validated GPU==CPU mean 0.18%, background 0.04%.
3. ~~**GPU SPPM** (M3)~~ — **DONE 2026-07-23.** Resident device SPPM session reusing the mode-M deposit + a per-pixel visible-point/gather/update kernel trio; per-pixel progressive state stays on-device across passes. Validated GPU==CPU on a Cornell glass-sphere caustic (mean 0.2–1.2%, background 0.3%).
4. ~~**Mode-M final gather on GPU** (M4)~~ — **DONE 2026-07-23.** Device `dPhotonGatherSub` (specular walk → one-bounce density query folding `rho(y)*rho(vis)` per photon; env/specular-emitter reflected off the visible point) + a `fgRays>0` branch in `dPhotonGather` (NEE direct + K cosine sub-rays); `fgRays` threaded through `kGather`/`renderPhotonMapSharedCuda`, `g_pmFinalGather==0` caller gates dropped. Validated GPU==CPU mean 0.43%, background 0.98%, per-pixel noise √-scales with spp.
5. **Per-hit BSDFs in GPU BDPT** (M9) — **first increment DONE 2026-07-23.** DVertex now stores per-hit `u,v`; `dVertHit` reconstructs a `DHit` so `dBsdfF`/`dBsdfPdf`/`dRandomWalk` evaluate textured/patterned/record diffuse albedo & glossy reflect, per-hit glossy roughness + thin-film maps, mix masks, and colored-glass Beer-Lambert — all MIS-safe (same per-hit value in sampler and pdf). Gate `cudaBdptSupported` relaxed accordingly. Validated GPU==CPU on `textured.ftsl` (mean 0.06%, per-pixel diff √-scales with spp) and `mixmat.ftsl` (mean 0.21%). Still deferred: diffuse-transmit two-lobe connection, frosted (rough) dielectric microfacet BSDF, fluorescence vertex, spot/env light-subpath strategies.
6. Longer tail: **rainbow media** on device (M10); **GRIN marcher** on device backward (M11); **GPU VCM** (M12).

### Descoped by user (2026-07-23) — NOT scheduled
Left on their current CPU/spectral fallbacks: **indexed-spectral palette maps** on device forward,
**Layered material** on device, **participating media in the RGB fast path**, and **textured/record
albedo in the RGB fast path**.

## Left on CPU by design (do NOT port)
Collimated beams (not NEE-samplable), dispersion-dependent materials in the RGB fast
path (inherently spectral), fixed-cap overflows (oversized multilayer/mix, >64-stop
driven records, over-deep lens) — too rare or semantically CPU/spectral to justify.
