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
  HG scatter + albedo Russian roulette) — homogeneous *and* heterogeneous; only
  GRIN and rainbow/dispersive media still route the backward pass to CPU.
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
