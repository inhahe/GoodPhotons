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

**M9 second increment (2026-07-23, 0.35.0):** two-sided **diffuse-transmit** (translucent)
vertices now render on-device in mode D — both lobes (front-hemisphere `reflect`,
back-hemisphere `transmit`, energy-clamped) plus the two-sided back-hemisphere connection
strategy. `dBsdfPdf`/`dVertexPdfF`/`dMisWeight` gained a `lambda` argument because the
lobe-selection pdf (`pSel = rhoR/tot` vs `rhoT/tot`) is wavelength-dependent; `dConnectBDPT`
uses two-sided guards (allow back hemisphere, skip shadow-terminator, `|cos|` in G) mirroring
the CPU reference. Validated GPU==CPU on `scraps/dtrans.ftsl` (mean B/A=1.0009 at 512 spp,
background 1.0026, per-pixel diff halves 8.42%→4.39% at 4× spp = unbiased).

**M9 third increment (2026-07-23, 0.36.0):** **frosted (rough) dielectric** is now on-device
in mode D. This one only needed the gate relaxed: the device `refractOrReflect` already
jittered the chosen reflect/refract lobe by the per-hit `dMatRoughness` (keeping it on the
intended side), and `dDielectricStep` in the BDPT random walk already routed through it, so a
rough dielectric is the same **stochastic-delta** vertex on GPU as on CPU (`bdpt.h` treats
every dielectric as a non-connectable delta and only jitters its scattered direction). The old
"kernel treats every dielectric as smooth" note was stale. Validated GPU==CPU on
`scraps/frosted.ftsl` (Cornell box + rough BK7 sphere): mean B/A=0.9991 at 512 spp, and the
per-pixel abs diff halves 10.86%→5.73% at 4× spp = unbiased.

**M9 wrap-up (2026-07-23, 0.36.0): the per-hit-BSDF GPU-vs-CPU parity gaps in mode D are
now all closed.** The items that were previously listed here as "remaining" — **fluorescence**
and **spot/env/collimated lights** — turned out NOT to be GPU-vs-CPU gaps at all. BDPT can't
render either on *any* backend: `main.cpp bdptUnsupportedFeature()` flags them (along with
layered materials) at the mode-D guard, which **refuses the render** (or demotes D → B with
`-on-unsupported fallback`) before any BDPT dispatch, so a
fluorescent/spot/env scene never reaches the BDPT path (CPU or GPU) — the CPU BDPT's
"elastic-base-only" fluorescent handling in `bdpt.h` is itself unreachable dead code for
whole-scene fluorescence. The stale per-material rejects in `cudaBdptSupported` (which the
demotion made unreachable) were removed; the gate now carries no material reject and only
mirrors the emitter/GRIN/rainbow refusals as belt-and-suspenders. True fluorescence / spot /
env rendering is a mode B/P/R feature on both CPU and GPU.
Genuinely still CPU-only in mode D (real device-volume limits, not per-BSDF):
| Feature | Why CPU today | Class |
|---|---|---|
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
5. **Per-hit BSDFs in GPU BDPT** (M9) — **three increments DONE 2026-07-23.** (1) DVertex now stores per-hit `u,v`; `dVertHit` reconstructs a `DHit` so `dBsdfF`/`dBsdfPdf`/`dRandomWalk` evaluate textured/patterned/record diffuse albedo & glossy reflect, per-hit glossy roughness + thin-film maps, mix masks, and colored-glass Beer-Lambert — all MIS-safe (same per-hit value in sampler and pdf). Validated GPU==CPU on `textured.ftsl` (mean 0.06%) and `mixmat.ftsl` (mean 0.21%). (2) Two-sided **diffuse-transmit** (translucent) now on-device — both lobes + back-hemisphere connection; `lambda` threaded through `dBsdfPdf`/`dVertexPdfF`/`dMisWeight` for the wavelength-dependent lobe-selection pdf. Validated GPU==CPU on `scraps/dtrans.ftsl` (mean B/A=1.0009 at 512 spp, per-pixel diff halves 8.42%→4.39% at 4× spp = unbiased). (3) **Frosted (rough) dielectric** now on-device — only the gate needed relaxing; `refractOrReflect`/`dDielectricStep` already jittered the lobe by per-hit roughness (stochastic-delta, same as `bdpt.h`). Validated GPU==CPU on `scraps/frosted.ftsl` (mean B/A=0.9991 at 512 spp, per-pixel diff halves 10.86%→5.73% at 4× spp = unbiased). Gate `cudaBdptSupported` relaxed accordingly. **M9 COMPLETE:** the two items once listed as deferred (fluorescence, spot/env light-subpaths) are not GPU-vs-CPU gaps — BDPT can't render them on any backend, so the `main.cpp` mode-D guard refuses those scenes (or demotes D → B with `-on-unsupported fallback`) before dispatch. The stale unreachable per-material rejects in `cudaBdptSupported` were removed. Only genuine device-volume limits (GRIN/rainbow, handled by M10/M11) keep a mode-D scene on the CPU.
6. Longer tail: **rainbow media** on device (M10); **GRIN marcher** on device backward (M11); **GPU VCM** (M12).

### Descoped by user (2026-07-23) — NOT scheduled
Left on their current CPU/spectral fallbacks: **indexed-spectral palette maps** on device forward,
**Layered material** on device, **participating media in the RGB fast path**, and **textured/record
albedo in the RGB fast path**.

## Left on CPU by design (do NOT port)
Collimated beams (not NEE-samplable), dispersion-dependent materials in the RGB fast
path (inherently spectral), fixed-cap overflows (oversized multilayer/mix, >64-stop
driven records, over-deep lens) — too rare or semantically CPU/spectral to justify.
