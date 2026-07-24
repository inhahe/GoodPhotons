# ftrace — design & architecture

Physically-based **spectral** renderer (C++17, single exe `ftrace.exe`), Windows /
MSVC / CMake, with a CUDA backend (RTX-class, tested sm_89). Photons are traced
**forward from the lights** in the flagship modes (hence "forward raytracer"), but
backward path tracing, BDPT, photon mapping, SPPM, VCM and a z-buffer preview
rasterizer are all built in. `README.md` is the exhaustive user-facing manual;
this file records the *internal* architecture. `known-issues.md` tracks bugs/debt.

## Render modes (dispatch in `main.cpp`)

| Mode | What | Core |
|---|---|---|
| `A` | forward + finite-lens physical camera (photons hit the lens) | `render.h` |
| `B` | forward light tracing, splat through pinhole/lens to film (flagship) | `render.h` |
| `C` | forward + contact sensor | `render.h` |
| `R` | backward (unidirectional) path tracer — the reference | `backward.h` |
| `P` | composite: forward B + backward R passes merged | `main.cpp` orchestration |
| `D` | bidirectional path tracer (BDPT, MIS) | `bdpt.h` |
| `M` | photon map (deposit pass + per-pixel density gather; optional `-pmfg` final gather) | `photonmap.h`, `photonmap_render.h` |
| `S` | SPPM (progressive photon mapping, shrinking radius) | `sppm_render.h` |
| `U` | VCM (vertex connection & merging) | `vcm.h` |
| `V` | validation: renders B and R, reports residual | `main.cpp` |
| `-raster` | z-buffer preview rasterizer + interactive fly viewer (`-explore`) | `raster.h`, `raster_cuda.cu` |

CPU is the default device; `-device gpu|auto` enables CUDA for forward A/B/C,
M-deposit/gather, R, D (untextured), and the raster preview (`-raster-gpu`).

## Module map (src/)

- **`main.cpp`** (~6200) — CLI parsing, mode dispatch, chunking/progressive loop
  (`cpuSppChunks`, `chunkFixed = !progressive && g_showWindow` — a bare fixed `-n`
  with no `-window`/budget flag runs monolithically with output only at the end),
  periodic write/`-interval`, checkpoint/resume (`.ftbuf`), multi-camera shared
  passes, exposure-lock metering pre-pass (device-aware: each mode meters through
  its own GPU entry point when `-device gpu|auto` and the mode's support predicate
  allow, CPU fallback otherwise; all-M pinhole groups meter in one batched
  `renderPhotonMapSharedCuda` pass with `MeterConverge` early-stop via `onFrame`),
  PNG/PPM output.
- **`scene.h` / `ftsl.h`** — scene model and the FTSL scene-language parser
  (cameras, camera_curve/path/orbit, materials, lights, media, implicits, meshes).
  `FTSL.md` documents the language.
- **`geometry.h` / `bvh.h`** — primitives + SAH BVH (split plane by SAH, always
  recurse to LEAF_SIZE, median fallback; front-to-back traversal, ray-slab test
  unrolled; `tEnter` pruning).
- **`mesh.h`** (+ `gltf.h`, `fbx.h`/`fbx_load.cpp`) — OBJ (custom fast parser:
  single fread, in-place float/int scan), glTF/GLB subset, FBX geometry-only.
- **`implicit.h` / `isomesh.h`** — implicit/isosurface evaluation and marching-cubes
  tessellation. `marchImplicit` is staged **fill → discover → resolve → wind**:
  parallel lattice `val[]` fill and parallel per-vertex bisection refine + gradient
  (pure per-slot), with the order-sensitive weld-map sweep and winding pass kept
  serial — bit-identical to the old serial code by construction. Per-implicit
  marching also runs in parallel across objects.
- **`render.h`** — CPU forward tracer (modes A/B/C + photon deposit for M/S/P):
  per-photon loop, Russian roulette, sphere-scan cos/sin tables, splatting.
- **`backward.h`** — CPU backward reference tracer (`radianceHero`).
- **`bdpt.h`** — BDPT with MIS; vertices stored by **index** (never `Vertex&`
  across `push_back` — a use-after-free lived here once; see known-issues).
- **`vcm.h`**, **`sppm_render.h`**, **`photonmap.h`/`photonmap_render.h`** — U/S/M.
  PhotonMap::build precomputes per-photon CIE X/Y/Z (the 3.65× mode-M win); VCM
  caches CIE lookups; kd/grid structures for gathers.
- **`spectrum.h` / `spectral_library.h` / `upsample.h` / `color.h` / `hero.h`** —
  spectral core: measured SPDs/materials, RGB→spectrum upsampling, CIE tables,
  hero-wavelength sampling (`kHeroC=4`: hero λ + 3 stratified secondaries) used by
  R, A/B/C, M/S on CPU and the GPU forward megakernel. Emitter SPD sampling is
  cached per light. Tabulated curves (FTSL `table { }` / `file:`) build a
  `Spectrum` via `tabulatedSpectrum` (piecewise-linear, default) or
  `tabulatedSpectrumMono` (opt-in `interp=cubic`: monotone Fritsch–Carlson/PCHIP —
  C¹ but shape-preserving, no overshoot, so an interpolated reflectance/absorption
  can't ring outside its neighbouring samples). Both clamp to the endpoints outside
  the sampled range (no extrapolation); `loadSpdFile` warns once if a `file:` curve
  doesn't span the 360–830 nm render range. Since `Spectrum` is
  `std::function<double(double)>` evaluated at each photon's exact λ, the interpolant
  shape shows directly (there's no pre-binning), which is why overshoot matters.
- **`camera.h` / `lens.h`** — camera models incl. finite thin-lens, fisheye/pano,
  realistic multi-element lens; `scene_film.h` film/EV/auto-exposure (p99),
  exposure-lock anchors.
- **`materials.h` / `pattern.h` / `texture.h` / `layered`** — BSDFs (diffuse,
  mirror, glossy, dielectric w/ nested IOR, diffuse-transmission, filter gels,
  fluorescence, layered), procedural patterns (POV-derived `pov_noise.h` /
  `pov_functions.h`), UV texturing.
- **`medium_stack.h` / `phase.h` / `grin.h` / `rainbow.h` / `vdbgrid.*`** —
  participating media (bounded, density fields, superposition), HG + water-droplet
  (rainbow) phase functions, gradient-index bending, NanoVDB density import.
- **`rng.h`** — Pcg32 + `seedUnit(rng, unitIndex, salt)` splitmix64 mixing:
  **every work unit (photon or pixel-sample) seeds its own stream**, so results are
  independent of chunk splits / thread count / banding / `-resume` boundaries.
- **`render_cuda.cu`** (~7000) — the whole GPU backend: megakernel + wavefront
  forward paths, GPU R and BDPT, M deposit/gather, device twins of hero sampling.
  GPU backward (`bkRadiance`) supports **participating media** natively since
  0.23.0 (free-flight `dMediaSampleCollision` competing with the surface hit,
  volume NEE `bkNeeVolume`, Beer–Lambert `dMediaTransmittance` on NEE + throughput,
  HG scatter + albedo Russian roulette) — homogeneous *and* heterogeneous.
  Spectral **rainbow-phase** media run on the device too since 0.37.0 (M10):
  a per-medium λ×µ Airy phase table + per-λ CDF is uploaded, and the unified
  `dMedPhase`/`dMedPhaseSample` dispatch (bilinear table eval / CDF importance-sample
  vs analytic HG) replaces the raw `hgPhase` calls across forward, backward, and BDPT.
  **GRIN (gradient-index) media** run on the backward reference (mode `R`) on-device
  too since 0.38.0 (M11): `dGrinMarch` (render_cuda.cu) is the device twin of
  `grin::march`, carrying its running Eikonal state in double, and `bkRadiance` marches
  each bounce's ray before `closestHit` (gated by `sc.hasGrin`); only mode-`D` BDPT and
  the RGB fast path still route GRIN scenes to the CPU. Since
  0.24.0 it also does **fluorescence** (bispectral Stokes-shift adjoint: elastic +
  excitation-wavelength NEE, `gOut = M(lambda)/Mint * invPdf`, stochastic
  elastic/reemit/absorb continuation — baked `fluoEmitSpec`/`fluoMint` on the
  device material). Since 0.25.0 it does a **constant environment light**
  (`bkNeeEnv` surface-vertex + `bkNeeEnvVolume` fog-vertex env-NEE, uniform-sphere
  sample, balance-heuristic MIS against a tracked `contBsdfPdf`, MIS'd env-miss on
  ray escape; `envIndex` + env geomWeight added to `DScene`/`dInvPdfLambda`). Since
  0.30.0 (M1) it also handles an **IMAGE-based (lat-long HDR) env** on the GPU
  backward: `bkNeeEnv`/`bkNeeEnvVolume` branch on `sc.env.scale != nullptr` and
  importance-sample the map on-device via `dEnvSample` (the forward path's luminance
  2-D CDF), with `dEnvRadiance` (per-texel JH `coeff`·`scale`·`illum`) and `dEnvPdf`
  for MIS-consistent env-miss; the previously-canceled illuminant table is uploaded
  as `DEnvMap::illum`. Validated GPU==CPU to 0.14% in linear radiance at 8192 spp.
  The same `dEnvRadiance` also serves **mode-M (photon map) env** on the GPU (M2,
  0.31.0): the deposit already emits env photons (env's indirect bounces land in the
  map), and `dPhotonGather` adds env's direct term on gather-ray escape — so
  `cudaPhotonMapSupported` no longer rejects env scenes (validated GPU==CPU mean 0.18%,
  background sky 0.04%). Since 0.32.0 (M3) there is a resident **GPU SPPM** session
  (`SppmSession`, `cudaSppmSupported == cudaPhotonMapSupported`): per-pixel progressive
  state (`tau`/`radius`/`nAcc`/`directSum` + this pass's visible point) lives on the device
  across passes, and each pass runs `kSppmVisiblePoint` (resample the camera visible point +
  direct term via the specular walk), reuses the mode-M forward deposit (`launchForward`,
  fresh seed = cumulative emitted), builds the grid **on-device** at the largest current
  per-pixel radius (0.39.1: deposits stay resident in a grow-only device buffer; rMax via
  `transform_reduce`, `kSppmCellKey` + stable sort + `lower_bound` cell ranges,
  `kSppmGatherConvert` bakes the photon records with double-precision cie tables — per-pass
  PCIe traffic drops from the full photon slab to a few bytes; ~13× faster passes), then
  `kSppmGather` (query + Hachisuka shared-statistics radius/flux update) and
  `kSppmResolve` (`L = directSum/passes + tau/(pi R^2 Nemit)`). The SPPM photon record bakes
  `pX = cie(lambda)·power/pi` with NO area/nEmitted fold (those depend on the current
  per-pixel radius, applied at resolve) — unlike mode M, which folds them in. Validated
  GPU==CPU on a Cornell glass-sphere caustic (mean 0.2–1.2%, background wall 0.3%, per-pixel
  diff shrinking with passes). Since 0.33.0 (M4) the mode-M **`-pmfg` Jensen final gather**
  also runs on the device: `dPhotonGather` gained an `fgRays>0` branch (NEE direct term via
  `bkNeeLight` + `K` cosine-hemisphere sub-rays), each sub-ray handled by `dPhotonGatherSub`
  (device twin of `photonGatherSub`) — it follows specular surfaces, then at the first diffuse
  hit y does a radius density query folding `rho(y)·rho(vis)` per photon wavelength (spectral
  two-bounce colour bleed), plus env-on-escape / specular-arrival emitters reflected off the
  visible point. `fgRays` is threaded through `kGather`→`renderPhotonMapSharedCuda` and the
  `g_pmFinalGather==0` caller gates in `main.cpp` were dropped. Validated GPU==CPU on a Cornell
  glass-sphere+diffuse-walls box (mean 0.43%, background 0.98%, per-pixel noise √-scaling with
  spp — unbiased). Since 0.39.0 (M12) there is a resident **GPU VCM/UPS** session (mode `U`,
  `VcmSession`, `cudaVcmSupported == cudaBdptSupported && media.empty()` — surfaces-only, pinhole
  only) mirroring `vcm.h`'s `vcmPass`: each pass (1) `kVcmLight` traces one light subpath per pixel,
  storing connectible vertices into a **per-path slab** (`lvSlab[i·vcmCap+k]`, no cross-thread
  atomics) and splatting the connect-to-camera (t=1) light-image contributions (atomic into a
  per-pass double buffer); (2) compacts the slab **on-device** (0.39.1: thrust scans over the
  per-path counts → `pathBegin`/`pathEnd` + `kVcmCompactScatter`; only the 4-byte total vertex
  count crosses PCIe per pass, vs the former ~69 MB slab download) into contiguous per-path
  ranges so the same-λ vertex CONNECTION reads its PAIRED light subpath exactly
  (single-wavelength spectral BDPT); (3) sort-builds the uniform hash grid on-device
  (`kVcmCellKey` + `thrust::sequence` + stable sort + `lower_bound` — order-identical to the
  former host counting sort; cell = merge radius, reusing the M3 device-grid query layout;
  ~4× faster passes, byte-identical output); (4) `kVcmCamera` traces one camera
  subpath per pixel doing emission (s=0) / NEE (s=1) / paired-path vertex connection (c) / grid merge
  from ALL paths (d) under one **balance-heuristic** MIS (SmallVCM `dVCM`/`dVC`/`dVM` bookkeeping,
  misArrival/misScatter inlined with Mis=identity), accumulating the running per-pixel XYZ sum
  (camera radiance + the light splat); the resolve divides by the pass count. Reuses M9's device BDPT
  BSDFs (`dBsdfF`/`dBsdfPdf`/`DVertex`); `dVcmScatter` is the device twin of `scatterSample`. The
  merge builds the density estimate in XYZ (cie(λ) per merged light vertex, other paths' λ) exactly
  like modes M/S. main.cpp mode-U mirrors mode-S (auto/gpu device, radius schedule
  `r_i=R0·i^((α−1)/2)`). Validated GPU==CPU on `absolute.ftsl` (Cornell + dielectric sphere,
  fixed-gain absolute mode to bypass per-image auto-exposure) at 500 passes: mean linear-luminance
  ratio 0.9993 (−0.07%), per-channel bias ≤0.5% (R −0.43%, G −0.06%, B +0.20%), per-pixel median rel
  error 3.0% at the ~4.5% independent-MC noise floor — no systematic bias. Since 0.34.0 (M9) the **GPU BDPT** kernel (mode `D`) threads the **per-hit
  surface point** through its connection BSDF: each `DVertex` stores the interpolated texcoords
  (`u,v`) and `dVertHit` reconstructs a minimal `DHit`, so `dBsdfF`/`dBsdfPdf` and the random
  walk evaluate per-hit-driven throughput slots consistently in BOTH the sampler and the
  pdf/eval (MIS-safe: a textured albedo changes only `f`, not the cosine pdf; per-hit glossy
  roughness feeds the same `dMatRoughness` into sampler and pdf). On-device now: textured/
  patterned/record diffuse albedo & glossy reflect, per-hit glossy roughness + thin-film maps,
  mix blend masks, and Beer–Lambert **colored-glass** interior absorption (delta vertex →
  throughput only, mirroring `bdpt.h`'s `curAbsorb`). Since 0.35.0 two-sided **diffuse-transmit**
  (translucent) also renders on-device — both lobes (front-hemisphere `reflect`, back-hemisphere
  `transmit`, energy-clamped) plus the two-sided back-hemisphere connection (allow back hemisphere,
  skip the shadow-terminator softening, `|cos|` in G); `lambda` is threaded through
  `dBsdfPdf`/`dVertexPdfF`/`dMisWeight` because the lobe-selection pdf is wavelength-dependent.
  Since 0.36.0 **frosted (rough) dielectric** also renders on-device: `refractOrReflect`/
  `dDielectricStep` already jitter the chosen reflect/refract lobe by the per-hit roughness, so a
  rough dielectric is the same non-connectable **stochastic-delta** vertex on GPU as in `bdpt.h`
  (only the gate needed relaxing). With that, **all genuine per-hit-BSDF GPU-vs-CPU parity gaps in
  mode D are closed** (M9 complete): `cudaBdptSupported` carries no per-material reject. The things
  BDPT still can't render — **fluorescence**, **layered stacks**, **spot/env/collimated lights** —
  are *not* GPU gaps: `main.cpp`'s mode-D guard (`bdptUnsupportedFeature`) refuses those scenes (or
  demotes D→B with `-on-unsupported fallback`) on both backends before any BDPT dispatch, so they
  never reach the device path; only GRIN media (curved paths) keep an in-scope mode-D scene
  on the CPU (spectral rainbow-phase media now render on-device in mode D since M10/0.37.0). Validated GPU==CPU on `textured.ftsl` (mean 0.06%,
  per-pixel diff halving 8.2%→4.3% at 4× spp — unbiased), `mixmat.ftsl` (mean 0.21%),
  `scraps/dtrans.ftsl` (mean B/A=1.0009 at 512 spp, per-pixel diff halving 8.42%→4.39% at 4× spp),
  and `scraps/frosted.ftsl` (mean B/A=0.9991 at 512 spp, per-pixel diff halving 10.86%→5.73%). Since
  0.26.0 it also does
  **point-spot lights** (deterministic connect + `spotFalloff` cone weight in
  `bkNeeLight`/`bkNeeVolume`, `spotOmega` geomWeight in `dInvPdfLambda`); only
  collimated beams (not NEE-samplable) still force the CPU backward tracer.
  Since 0.27.0 there is a separate **fast RGB backward** path
  (`renderBackwardRGBCuda`/`kBackwardRGB`/`bkRadianceRGB`, selected by `-rgb` on mode R
  via `cudaBackwardRGBSupported`): an Option-B non-spectral tracer carrying a `DVec3`
  linear-RGB throughput so one intersection walk yields a full-colour sample. Materials
  bake to a linear-RGB albedo under equal-energy white and emitters/env to
  `xyzToLinearSrgb(∫CIE·spd)` (host `namespace rgbbake`, at scene build); deposit maps
  linear-RGB→XYZ via `dRgbToXyz`. Achromatic specular uses `LREP_RGB=550nm` through the
  existing `dDielectricStep`; colored glass keeps a 3-tap Beer-Lambert `rgbAbsorb`.
  RGB Russian roulette survives with `q=rgbLuma(albedo)`. Matches the spectral backward's
  absolute luminance to noise on flat-spectrum scenes; drops dispersion/thin-film/
  fluorescence (Option-B). Gate excludes media, image-env, textured/record albedo, and
  collimated/env-shape emitters (fall back to the spectral backward).
  Since 0.28.0 there are **scene-ignore speed flags** (rasterizer-style): host-side
  `Scene::applyIgnoreFlags(noMedia,noEnv,noFluoro)` (in `scene.h`, run once at load)
  strips media (`-no-media`), the environment emitter (`-no-env`; erases
  `emitters[envIndex]`, resets `envIndex`/`envMap`/`envXYZ`, `finalizeEmitters()`), or
  demotes fluorescent → diffuse (`-no-fluoro`); each prints an `[ignore] stripped:`
  summary. Depth/Whitted params are threaded via globals `g_maxBounceOverride`
  (`-max-bounce N`) and `g_directOnly` (`-direct-only`) — mirroring the `g_heroC`
  pattern — into `BackwardRenderer::maxBounce`/`directOnly`, forward `Renderer::maxBounce`,
  and (on GPU) `DScene.bkMaxBounce`/`bkDirectOnly` set on the `renderBackward*Cuda`
  wrappers. `directOnly` (terminate after the first non-specular NEE, specular chains
  still recurse) is scoped to the camera path tracers (R spectral + RGB, P's backward
  layer); forward B and the photon/BDPT modes honour only `maxBounce`.
  Since 0.29.0 the interactive `-explore` fly-viewer can toggle (key **`T`**) a live
  **path-traced preview** using the fast RGB backward tracer instead of the flat raster:
  a resident `BackwardRGBSession` (render_cuda.cu) bakes/uploads the scene ONCE
  (`buildUploadScene`) and keeps a persistent SUM film; while the camera holds still the
  main loop calls `backwardRGBSessionAccumulate(batch)` each idle tick (advancing the RNG
  `sampleBase`, fixed large `kSppCap` so streams stay unique) and presents the converging
  image via the shared `filmToRgb8` (auto-exposure anchor locked per pose); a camera move
  shows the responsive raster and marks the session for a `setCamera()` re-aim (which
  re-bakes only the cheap POD `bakeCamera` and zeroes the film). Availability mirrors the
  batch `-rgb` scope (`cudaBackwardRGBSupported`); the scene-ignore flags carry in for free
  (scene already stripped host-side; maxBounce/directOnly passed to `...SessionBegin`).
  (`traceHeroPhoton`/`shadeStepHero`), scene upload into `__constant__`/device
  buffers. FP32 by default (`FTRACE_GPU_FP32=ON`). Implicit sphere-tracing
  (`intersectImplicit`) marches + root-refines in FP32 on pre-converted mirror
  pools (`DFieldNodeF`/`PatNodeF`, float VM twins `dFieldEvalF`/`dPatternEvalF`)
  since 0.19.14 — the committed hit is float anyway and FP64 VM ops serialize on
  consumer GPUs' 1/64-rate FP64 pipe (~2× on implicit-heavy scenes); normals
  (`dFieldGradient`) and media bound-fields stay FP64 on the original pools.
  Photon-beams (`-beams`, since 0.19.17) run on the GPU too: `DCamSet::beamGather`
  drives a `shadeStep` branch that (with an independent per-photon `DRng crng` seeded
  in `kTrace`) crosses the medium straight and has each shared camera resample its own
  single-scatter in-scatter point — decorrelated per-frame flyby noise, megakernel-only.
  The shared multi-camera pass runs through a **resident GPU session** (0.20.2):
  `sharedForwardGpuBegin/Batch/Hits0/Download/End` (render_cuda.h) bake+upload the
  scene, bake all cameras, and allocate device films/hits/energy **once**; each
  progressive batch is then a bare `launchForward` accumulating in place, and host
  films are only downloaded lazily (`syncAcc` in `runSharedGroup`) at `-interval` /
  status / final boundaries — download REPLACES host `acc[]`/`accE` with running
  totals rather than merging. `-resume` seeds the device accumulators from the
  loaded checkpoint at Begin. Between intervals a `-noise` budget polls only cam-0's
  hits plane (`sharedForwardGpuHits0`). The old wrapper paid full
  upload/alloc/download/convert/merge/free per ~2M-photon batch, which throttled the
  loop (measured: +80% photons/s on 16 cams @ 640×360, +22% on 2 cams @ 320×240);
  `renderForwardSharedCuda` survives as a one-shot wrapper over the session.
  `raster_cuda.cu` = GPU raster (own section below).
- **`livewindow.*`** — Win32 GDI live preview (`-window`/`-keepwindow`), interactive
  fly viewer input, camera-path timeline panel.
- **`record.h` / `render_progress.h`** — run records, live status line
  (`[live] … photons, ~N% noise`), noise estimation for `-noise` budgets.

## GPU raster pipeline (`raster_cuda.cu`)

Powers `-raster -device gpu` and the interactive explorer's per-frame redraws;
steady-state cost (independent of launch/scene build/upload) is measured with
`-raster-bench N`. Four device passes per frame:

- **A — project** (`kProject`): one thread per slot, register-resident 8-case
  near-plane clip. Geometry is split hot/cold: `DGeo` (36 B — screen verts, invd,
  flags' companion) is written for every surviving slot, `DAttr` (120 B — the
  attribute payload) only for clipped slots that needed new vertices. A dense int
  `flags[]` (valid / clear / clipped) drives later passes.
- **B — classify + raster** (`kClassify`, `kRasterSmall/Med/Large`): slots are
  binned by clamped bbox pixel count (≤128 small, ≤16 384 med, else large) into
  device lists; raster kernels merge into a 64-bit packed `(invd_bits<<32)|slot`
  visibility buffer via `atomicMax` (order-independent ⇒ bit-identical under any
  thread mapping). Kernels read bin counts **from device memory** (`dbinCnt`,
  5 ints: 3 counts + med/large ticket counters), so the host never reads counts
  back and the whole frame enqueues without a mid-frame WDDM flush. Work mapping
  per bin: **small** = 1:1 thread↔item under an upper-bound grid (the hardware
  block scheduler load-balances millions of variable-cost items better than any
  grid-stride loop); **med** = warp-level ticket queue (lane 0 `atomicAdd` +
  `__shfl_sync` broadcast, 32 lanes stride rows); **large** = block-level ticket
  queue (shared-mem ticket, block strides rows). Heavy variable-cost bins need
  dynamic balancing — static grid-stride created straggler warps (+0.5 ms on
  gallery); never ticket the small bin (millions of atomics would serialize).
- **C — shade** (`kShade`): one thread per pixel resolves `vis` → shaded float
  RGB. Optional see-through mode then runs a fill+clear pass (`kFillF`+`kClear`);
  its `atomicMulF` has a benign 1-px race (see known-issues).
- **D — expose + encode**: device luminance histogram rounds give an *exact* p99
  white point (readbacks only on the first frame; later frames reuse the cached
  exposure unless the histogram shifts), then `kToneMap` encodes RGB8 on device;
  one pinned-memory D2H of the final image.

The frame is **sync-free**: no `cudaDeviceSynchronize` anywhere; only real data
dependencies block (first-frame histogram readbacks, final image download). Errors
surface through the blocking copies' return codes plus one sticky
`cudaGetLastError()` sweep per frame. Per-pass profiling (`-raster-bench`'s
breakdown) records CUDA events into the stream between passes and resolves them
once after the download — zero overhead when disabled.

Perf state (2026-07 campaign, opts 1–8, RTX 4090 @1600×900): cornell **1.97 ms**
(508 fps), gallery (5.08 M tris) **~4.45 ms** (~225 fps), glassgal **5.12 ms** —
~22–25× vs the 0.19.0 baseline. Passes sit near memory-bandwidth floors; the
remaining ~1.1 ms is host-side (result-vector copy, WDDM submit, bench loop).
HIP portability note: the alias block deliberately does **not** alias
`__shfl_sync`, so a HIP build fails loudly at kRasterMed instead of silently
mis-broadcasting on wave64 GPUs (see known-issues).

**GPU clock keep-warm (interactive explorer).** The explorer re-renders one frame
per camera move, then idle-sleeps — a bursty, low-duty submission pattern the
NVIDIA driver's DVFS reads as "idle", parking the card in its lowest power state
(measured RTX 4090: **P8 @ 210 MHz** vs **P0/P2 @ 2520–2775 MHz** under load, a
~13× clock drop; ~33× for a cold first frame). That made each fresh mouse-look
burst pay a cold-clock penalty until continuous motion finally ramped the clocks.
Fix (`main.cpp` explorer loop): for `kWarmGraceSec` (2.5 s) after the last real
interaction the loop holds the boost clock with discarded "warm-only" `rasterOne`
frames (never touch the window), then past the grace window falls back to the 15 ms
passive sleep and the card powers down to P8. Two thresholds tune it: a warm frame
fires only once `idleFor` passes `kWarmGapSec` (0.10 s) — i.e. a GENUINE pause — and
then runs *continuously* (no nap) so the clock actually stays up (a sparse rate-
limited trickle was measured too weak — the card sat at P8). During an active mouse-
look or timeline-scrub drag the sub-frame gaps between input events stay under the
gap, so warm frames are suppressed and every loop slot samples the next scrub
position; otherwise a warm frame landing between two events would steal that slot and
the timeline would "chunk" by several cameras per drag (0.22.0 regression, fixed
0.22.1; see known-issues). Between events inside the gap the loop naps 3 ms (prompt
drain, no busy spin). Gated on the discrete-GPU path (`gpuRaster != nullptr`); CPU
raster unaffected.

## Threading model (CPU)

Band/chunk parallelism via `std::thread` pools sized by `hardware_concurrency`;
work units pull atomically from a shared counter in chunks. Determinism comes from
per-unit RNG seeding (above) plus order-independent accumulation per band/tile;
film merges are structured so paired runs differ only by summation-order ulps at
worst (mode R) or are bit-identical (fixed splits).

## Benchmarks & perf discipline

- `scraps/bench.py` — 19 standard configs (13 CPU + 6 GPU; cornell + gallery
  scenes); min-of-reps timing, sha1 of PPM outputs, `fuzzy_ppm_diff` for GPU.
- `scraps/bench_ab.py` — interleaved A/B harness (exeA/exeB alternate per config so
  machine drift cancels); used for the 2026-07 optimization campaign report
  (`scraps/bench_final_ab.json`).
- Rule: any hot-path optimization must be **bit-identical** (CPU sha1) or
  visually/fuzzy identical (GPU) vs. the pre-change exe before committing, one
  commit per optimization so any regression can be reverted alone.

## Build & release

- `build.bat` → CMake/VS2022 x64 Release into `build_cuda2/`, copies
  `ftrace.exe` to the repo root. **Warning:** freshly-configured build dirs
  currently produce a GPU-silently-dead exe (see known-issues, 2026-07-22) — build
  in the long-lived `build_cuda2`.
- `VERSION` (single `MAJOR.MINOR.PATCH` line) bumps with every observable rebuild;
  `release.bat` publishes repo-root `ftrace.exe` as GitHub release `v<VERSION>`
  (refuses on duplicate tag).
- Output conventions: renders → `ppm/`/`png/` (flyby series in `png/<set>/`),
  scratch scripts → `scraps/`. Renders always launched with `-keepwindow`
  (+ `-checkpoint`/`-interval`) and outside the Bash sandbox so the live window is
  visible.
