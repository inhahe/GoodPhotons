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
| ~~Spectral rainbow-phase media~~ | **DONE (M10, 0.37.0)** — λ×µ phase table + per-λ CDF uploaded per-medium; `dMedPhase`/`dMedPhaseSample` dispatch rainbow (bilinear `dRbEval` / CDF sample) vs HG on forward + backward + BDPT | ✔ on device |
| Oversized multilayer (>`D_MAXLAYERS`) | fixed device cap; CPU has no limit | portable but **low value** (raise cap) |
| Oversized Mix (>`D_MIXMAX` children) | fixed device cap | portable but **low value** |
| Driven record scalar >64 stops | overflows device interp array `vs[64]` | portable but **vanishingly rare** |

### Backward R (spectral) — `cudaBackwardSupported` (render_cuda.cu:7467)
Everything forward rejects, **plus**:
| Feature | Why CPU today | Class |
|---|---|---|
| ~~Image-based env NEE (lat-long map)~~ | **DONE (M1, 2026-07-23)** — added `dEnvRadiance`/`dEnvPdf`, uploaded the illuminant table, wired the device env sampler into `bkNeeEnv`/`bkNeeEnvVolume` + MIS'd env-miss; dropped the `envMap` reject. Validated GPU==CPU to 0.14% at 8192 spp. | ✅ |
| ~~GRIN media~~ | **DONE (M11, 0.38.0)** — `dGrinMarch` (device twin of `grin::march`, double-accumulated running state) now advances each backward bounce's ray through the Eikonal marcher before `closestHit`, gated by `sc.hasGrin`; the `cudaBackwardSupported` GRIN reject was removed. Validated GPU==CPU on the linear-gradient lens (SSIM 0.99, Pearson 0.99); small bent-region float-vs-double residual documented in known-issues.md. **Mode-D BDPT and the RGB fast path still reject GRIN.** | ✅ |
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
mirrors the emitter/GRIN refusals as belt-and-suspenders (rainbow is now on device — M10).
True fluorescence / spot / env rendering is a mode B/P/R feature on both CPU and GPU.
Genuinely still CPU-only in mode D (real device-volume limits, not per-BSDF):
| Feature | Why CPU today | Class |
|---|---|---|
| GRIN media | straight-segment MIS assumptions | **inherently-CPU** |

(Rainbow media in BDPT: **DONE (M10, 0.37.0)** — `dPhaseF`/`dPhasePdf`/`dMediumScatterF` now
dispatch the tabulated Airy phase; the rainbow reject in `cudaBdptSupported` was relaxed.)

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
| ~~**U — VCM/UPS**~~ (vertex connection + merging) | `vcm.h` | **DONE (M12, 2026-07-23, 0.39.0)** — resident device `VcmSession` mirroring `vcmPass`: `kVcmLight` traces one light subpath/pixel into a per-path slab (+ connect-to-camera splats), the host compacts it into per-path ranges and counting-sort-builds the merge grid (cell = radius), then `kVcmCamera` does emission/NEE/paired-path connection/grid merge under one balance-heuristic MIS. Reuses M9's device BDPT BSDFs + M3's device grid; gate `cudaBdptSupported && media.empty()` (surfaces-only, pinhole only). Wired into main.cpp mode-U (`-device gpu/auto`). Validated GPU==CPU on `absolute.ftsl` (fixed-gain radiance) at 500 passes: mean luminance ratio 0.9993, per-channel bias ≤0.5%, per-pixel median 3.0% at the MC noise floor. ✅ |

---

## Scheduled work (greenlit by user 2026-07-23) — quickest wins first

1. ~~**Image-based env NEE in GPU backward** (M1)~~ — **DONE 2026-07-23.** Added `dEnvRadiance`/`dEnvPdf`, uploaded the illuminant table, wired the device env sampler into `bkNeeEnv`/`bkNeeEnvVolume` + MIS'd env-miss; dropped the `envMap` reject. Validated GPU==CPU to 0.14% at 8192 spp. Also unblocks mode P camera-side.
2. ~~**Env term in the mode-M GPU gather** (M2)~~ — **DONE 2026-07-23.** Deposit already emits env photons (indirect); added env's direct term on gather-ray escape in `dPhotonGather` (constant + image env); dropped the `envIndex >= 0` reject. Validated GPU==CPU mean 0.18%, background 0.04%.
3. ~~**GPU SPPM** (M3)~~ — **DONE 2026-07-23.** Resident device SPPM session reusing the mode-M deposit + a per-pixel visible-point/gather/update kernel trio; per-pixel progressive state stays on-device across passes. Validated GPU==CPU on a Cornell glass-sphere caustic (mean 0.2–1.2%, background 0.3%).
4. ~~**Mode-M final gather on GPU** (M4)~~ — **DONE 2026-07-23.** Device `dPhotonGatherSub` (specular walk → one-bounce density query folding `rho(y)*rho(vis)` per photon; env/specular-emitter reflected off the visible point) + a `fgRays>0` branch in `dPhotonGather` (NEE direct + K cosine sub-rays); `fgRays` threaded through `kGather`/`renderPhotonMapSharedCuda`, `g_pmFinalGather==0` caller gates dropped. Validated GPU==CPU mean 0.43%, background 0.98%, per-pixel noise √-scales with spp.
5. **Per-hit BSDFs in GPU BDPT** (M9) — **three increments DONE 2026-07-23.** (1) DVertex now stores per-hit `u,v`; `dVertHit` reconstructs a `DHit` so `dBsdfF`/`dBsdfPdf`/`dRandomWalk` evaluate textured/patterned/record diffuse albedo & glossy reflect, per-hit glossy roughness + thin-film maps, mix masks, and colored-glass Beer-Lambert — all MIS-safe (same per-hit value in sampler and pdf). Validated GPU==CPU on `textured.ftsl` (mean 0.06%) and `mixmat.ftsl` (mean 0.21%). (2) Two-sided **diffuse-transmit** (translucent) now on-device — both lobes + back-hemisphere connection; `lambda` threaded through `dBsdfPdf`/`dVertexPdfF`/`dMisWeight` for the wavelength-dependent lobe-selection pdf. Validated GPU==CPU on `scraps/dtrans.ftsl` (mean B/A=1.0009 at 512 spp, per-pixel diff halves 8.42%→4.39% at 4× spp = unbiased). (3) **Frosted (rough) dielectric** now on-device — only the gate needed relaxing; `refractOrReflect`/`dDielectricStep` already jittered the lobe by per-hit roughness (stochastic-delta, same as `bdpt.h`). Validated GPU==CPU on `scraps/frosted.ftsl` (mean B/A=0.9991 at 512 spp, per-pixel diff halves 10.86%→5.73% at 4× spp = unbiased). Gate `cudaBdptSupported` relaxed accordingly. **M9 COMPLETE:** the two items once listed as deferred (fluorescence, spot/env light-subpaths) are not GPU-vs-CPU gaps — BDPT can't render them on any backend, so the `main.cpp` mode-D guard refuses those scenes (or demotes D → B with `-on-unsupported fallback`) before dispatch. The stale unreachable per-material rejects in `cudaBdptSupported` were removed. Only genuine device-volume limits (GRIN, handled by M11) keep a mode-D scene on the CPU.
6. ~~**Rainbow media** on device (M10)~~ — **DONE 2026-07-23 (0.37.0).** λ×µ phase table + per-λ CDF uploaded
   per-medium; unified `dMedPhase`/`dMedPhaseSample` dispatch (bilinear `dRbEval` / CDF importance-sample vs
   analytic HG) across forward + backward + BDPT; rejects relaxed in all three gates. Validated by isolating
   the phase from a pre-existing, phase-independent GPU↔CPU media brightness discrepancy (see known-issues.md):
   in clean mode-D BDPT the rainbow and a plain-HG control give the *same* GPU↔CPU B/A=2.41 (3 s.f.), so the
   rainbow adds zero bias beyond HG's; forward-bulk median 1.02 (rainbow) / 1.00 (HG); bows visually correct.
7. ~~**GRIN marcher** on device backward (M11)~~ — **DONE 2026-07-23 (0.38.0).** `dGrinMarch` (device twin of
   `grin::march`; running Eikonal state carried in double) marches each backward bounce's ray before
   `closestHit`, gated by `sc.hasGrin`; `bkRadiance` marches the primary camera ray at the top of its bounce
   loop, and the GRIN reject in `cudaBackwardSupported` was dropped (mode-D BDPT + RGB fast path still reject).
   Validated GPU==CPU on the linear-gradient lens (`scraps/grin_lin.ftsl`): SSIM 0.99 / Pearson 0.99, both
   backends bend identically; a small bent-region float(GPU)-vs-double(CPU) residual (~2.7% disc on linear,
   up to ~17% on a strong radial caustic, does not converge with spp) is documented in known-issues.md as the
   accepted device float-precision envelope amplified through the lens.
8. ~~**GPU VCM** (M12, mode U)~~ — **DONE 2026-07-23 (0.39.0).** Resident device `VcmSession` mirroring
   `vcmPass`: per-pass `kVcmLight` (light subpaths → per-path vertex slab + connect-to-camera splats) →
   host compaction into per-path ranges + counting-sort grid build (cell = radius) → `kVcmCamera`
   (emission/NEE/paired-path connection/grid merge under one balance-heuristic MIS) accumulating the running
   sum. Reuses M9's device BDPT BSDFs (`dBsdfF`/`dBsdfPdf`/`DVertex`) + M3's device grid pattern; gate
   `cudaBdptSupported && media.empty()` (surfaces-only, pinhole only). Validated GPU==CPU on `absolute.ftsl`
   (Cornell + dielectric sphere, fixed-gain absolute mode) at 500 passes: mean linear-luminance ratio 0.9993
   (−0.07%), per-channel bias ≤0.5%, per-pixel median rel error 3.0% at the ~4.5% independent-MC noise floor.

**All scheduled GPU-fallback-closure work (M1–M4, M9–M12) is now complete.** The only modes/features left on
the CPU are those descoped by the user or left on CPU by design (below).

### Descoped by user (2026-07-23) — NOT scheduled
Left on their current CPU/spectral fallbacks: **indexed-spectral palette maps** on device forward,
**Layered material** on device, **participating media in the RGB fast path**, and **textured/record
albedo in the RGB fast path**.

## Left on CPU by design (do NOT port)
Collimated beams (not NEE-samplable), dispersion-dependent materials in the RGB fast
path (inherently spectral), fixed-cap overflows (oversized multilayer/mix, >64-stop
driven records, over-deep lens) — too rare or semantically CPU/spectral to justify.
