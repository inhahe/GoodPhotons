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

### `camera_path`, physical film size, f-stop, and ISO not implemented
- **What:** the spec (§8.1/§8.3) proposes `camera_path` keyframed motion, a physical
  film `size <w> <h>` (mm), f-stop authoring (`fstop N` → aperture radius via focal
  length), and film `iso`/sensitivity. None are built; the camera still derives its
  image plane from `fov_y` + aspect and takes an `aperture` radius directly.
- **Proper fix (future):** add physical focal length from film size + fov, convert
  `fstop` → `apertureR = f/(2N)` at load time, apply `iso` as a per-camera exposure
  scale in the film write, and expand a `camera_path` into a sequence of `CamSpec`
  frames (the multi-camera loop in `main.cpp` already renders a list, so a path is
  just a generated `CamSpec` list + frame-numbered output names).
- **Status:** OPEN (design captured) — logged 2026-07-10.

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
