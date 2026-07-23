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

### BDPT D — `cudaBdptSupported` (render_cuda.cu:7312)
Forward scope plus MIS-correctness limits (the GPU connection BSDF `dBsdfF`/`dBsdfPdf`
has no `Hit` in scope, so a per-hit BSDF would bias MIS):
| Feature | Why CPU today | Class |
|---|---|---|
| Diffuse-transmit vertices | no two-lobe / back-hemisphere connection strategy | **portable** (thread Hit through `dConnect`) |
| Frosted / colored glass | kernel treats every dielectric as smooth & non-absorbing | **portable** |
| Textured albedo, fluorescence, roughness/film tex+pat, mix mask, records | connection uses constant params → MIS bias | **portable-hard** (each needs Hit threaded) |
| Spot/env/collimated emitters | no light-subpath strategy for them | **portable-hard** |
| GRIN / rainbow media | straight-segment MIS assumptions / HG-only phase | **inherently-CPU** / rainbow portable-hard |

### Photon map M — `cudaPhotonMapSupported` (render_cuda.cu:7830)
Forward scope plus:
| Feature | Why CPU today | Class |
|---|---|---|
| **Any environment light** | device `kGather` has no env term; CPU `photonGather` adds env on escape / at diffuse hits | **portable** (add env term to gather kernel) |
| **Final gather** (`g_pmFinalGather > 0`) | device does only the direct density estimate; final gather stays CPU (caller-gated, main.cpp:6034/6441) | **portable-hard, high value** (a second bounce of gather rays on device) |
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
| **S — SPPM** (stochastic progressive photon mapping) | `sppm_render.h` | **portable — most feasible large port**: repeated bounded photon deposits + radius-shrinking gather; the GPU already has the forward deposit pass and `kGather` from mode M, so it's largely a loop around existing kernels. High value. |
| **U — VCM/UPS** (vertex connection + merging) | `vcm.h` | **portable-hard, lowest priority**: needs GPU BDPT correctness *and* photon merging under one MIS weight. |

---

## Scheduled work (greenlit by user 2026-07-23) — quickest wins first

1. ~~**Image-based env NEE in GPU backward** (M1)~~ — **DONE 2026-07-23.** Added `dEnvRadiance`/`dEnvPdf`, uploaded the illuminant table, wired the device env sampler into `bkNeeEnv`/`bkNeeEnvVolume` + MIS'd env-miss; dropped the `envMap` reject. Validated GPU==CPU to 0.14% at 8192 spp. Also unblocks mode P camera-side.
2. **Env term in the mode-M GPU gather** (M2) — small kernel addition; unblocks env-lit photon maps.
3. **GPU SPPM** (M3) — reuses existing deposit + gather kernels; biggest quality-mode win.
4. **Mode-M final gather on GPU** (M4) — high value, more work.
5. Longer tail: **per-hit BSDFs in GPU BDPT** (M9); **rainbow media** on device (M10); **GRIN marcher** on device backward (M11); **GPU VCM** (M12).

### Descoped by user (2026-07-23) — NOT scheduled
Left on their current CPU/spectral fallbacks: **indexed-spectral palette maps** on device forward,
**Layered material** on device, **participating media in the RGB fast path**, and **textured/record
albedo in the RGB fast path**.

## Left on CPU by design (do NOT port)
Collimated beams (not NEE-samplable), dispersion-dependent materials in the RGB fast
path (inherently spectral), fixed-cap overflows (oversized multilayer/mix, >64-stop
driven records, over-deep lens) — too rare or semantically CPU/spectral to justify.
