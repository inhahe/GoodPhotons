// CUDA backend for the forward light tracer (model B).
//
// This is a plain-C++ interface so main.cpp (compiled by MSVC) can call into the
// CUDA translation unit (render_cuda.cu, compiled by nvcc) without ever seeing a
// __device__ symbol. The .cu file's HOST side includes the project headers, bakes
// the std::function-based Scene into POD device tables (sampled spectra, flat BVH,
// POD materials), uploads them, launches a megakernel that mirrors
// Renderer::tracePhoton, then downloads the accumulated film.
//
// The GPU covers the three forward camera models: A (contact sensor deposit),
// B (connect/splat to the pinhole — the default, and the one mode V validates), and
// C (finite-aperture thin-lens forward catch). Fluorescence is NOT supported on the
// device (it needs the emission-sampler reradiation path); a scene containing a
// Fluorescent material must fall back to the CPU. The caller is responsible for that
// check via cudaForwardSupported().
#pragma once
#include "scene.h"
#include "camera.h"
#include "render.h"   // EnergyReport, Film
#include "render_progress.h"   // SppProgress (chunked progress for modes R/D)

// True if a usable CUDA device is present (driver + at least one device). Cheap to
// call; result is cached after the first query.
bool cudaAvailable();

// Human-readable name of the primary CUDA device (or "none").
const char* cudaDeviceName();

// Orderly CUDA teardown: synchronize any outstanding device work, then destroy the
// primary context (cudaDeviceReset) while the process is still in a clean, quiescent
// state. Call this once, at the very end of a successful (or failed) run, BEFORE the
// process exits. Rationale: leaving the context to be reclaimed implicitly at process
// exit means the NVIDIA kernel driver tears it down asynchronously, from a DPC, after
// main() has already returned — and buggy nvlddmkm builds can fault there and bugcheck
// the whole machine (observed as BSOD/reboot "a couple seconds after the window closed",
// worst with an abrupt kill). Draining + resetting synchronously here removes that
// window: by the time we return there is no live GPU work and no context left to reap.
// No-op (and safe to call) when CUDA is unavailable or was never initialised.
void cudaGracefulShutdown();

// True if this scene can be rendered on the GPU (no unsupported material such as
// Fluorescent). When false, the caller must use the CPU renderer.
bool cudaForwardSupported(const Scene& scene);

// GPU forward light trace. camMode selects the camera model: 'A' (contact-sensor
// deposit), 'B' (connect/splat to the pinhole), or 'C' (finite-aperture forward
// catch). Traces N photons and returns the accumulated camera film (same
// units/convention as the CPU renderForward for the matching mode), so
// writeFilm(film, N) and the mode-V comparison work unchanged. Fills eOut with the
// same energy report. Requires cudaAvailable() && cudaForwardSupported(scene);
// otherwise returns an empty film.
// `seedBase` offsets the RNG stream so successive calls (render-in-chunks for a time
// budget, or resuming an accumulated film) draw statistically-independent photons;
// pass the cumulative photon count already traced. seedBase==0 reproduces the
// original single-shot stream bit-for-bit.
// `wavefront` selects the streaming (path-regeneration) GPU backend instead of the
// default megakernel. Both run identical physics and conserve energy exactly; the
// wavefront scheduler keeps SIMD lanes full on divergent / deep-path scenes and small
// GPUs, at the cost of extra memory traffic (the RNG stream — and thus the exact image
// noise — differs, but the two agree to within Monte-Carlo noise).
// `heroC` (>1) enables hero-wavelength spectral sampling in the forward megakernel: each
// photon path carries C stratified wavelengths through one shared BVH walk, cutting
// chromatic noise. It requires no participating media / GRIN (gated on the device) and
// forces the megakernel backend (hero physics is not in the wavefront scheduler); heroC<=1
// reproduces the original single-λ stream bit-for-bit.
Film renderForwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                       long long N, EnergyReport& eOut, bool diffraction,
                       char camMode, unsigned long long seedBase = 0,
                       bool wavefront = false, int heroC = 1);

// GPU multi-camera shared forward trace (models A and B — the device twin of the CPU
// renderForwardShared). Traces ONE set of N photons and splats each vertex to ALL
// cameras at once, returning one film per camera (each at its own resX[c] x resY[c]).
// This is the "many cameras for one photon set" win — the whole scene is baked and the
// photons flown once, instead of re-tracing per camera. camMode must be 'A' or 'B' (mode
// C consumes the photon per camera and can't share). Model B is bit-identical to
// per-camera renders (connect draws no RNG); model A shares the photon flight but each
// camera samples its own pupil (connectLens draws RNG), so its images match a standalone
// render in distribution, not bit-for-bit. eOut is the single shared-pass energy report.
// Requires cudaAvailable() && cudaForwardSupported(scene); otherwise returns empty films.
std::vector<Film> renderForwardSharedCuda(const Scene& scene,
                                          const std::vector<Camera>& cams,
                                          const std::vector<int>& resX,
                                          const std::vector<int>& resY,
                                          long long N, EnergyReport& eOut, bool diffraction,
                                          char camMode, unsigned long long seedBase = 0,
                                          bool wavefront = false, int heroC = 1,
                                          bool beamGather = false);

// ---- Resident shared forward session (the batched-accumulation fast path) ----
//
// renderForwardSharedCuda pays the full bake/upload/alloc/download/free round trip on
// EVERY call — fine for a one-shot render, ruinous for the progressive shared loop in
// main.cpp, which calls it ~50+ times/second in ~2M-photon batches (each batch re-baked
// and re-uploaded the whole scene, re-baked every camera, re-allocated every film, then
// downloaded and host-merged every film even though images are only consumed once per
// -interval). A session makes everything RESIDENT: begin() bakes/uploads the scene and
// cameras and allocates per-camera device films ONCE; batch() is then a bare kernel
// launch that accumulates into the resident films/energy; download() fetches the running
// TOTALS only when the host actually needs images (interval writes, noise checks, final);
// end() frees. Semantics match a sequence of renderForwardSharedCuda calls with the same
// per-batch seedBase (identical RNG streams), minus the per-batch overhead.
//
// Resume support: begin() can seed the device films/hits/energy from checkpoint state
// (`seedFilms`/`seedEnergy`, both nullable), so download() always returns full running
// totals (checkpoint + everything traced this session) and the caller can simply REPLACE
// its accumulators instead of merging.
struct SharedGpuSession;   // opaque; lives in render_cuda.cu
SharedGpuSession* sharedForwardGpuBegin(const Scene& scene,
                                        const std::vector<Camera>& cams,
                                        const std::vector<int>& resX,
                                        const std::vector<int>& resY,
                                        char camMode, bool wavefront, int heroC,
                                        bool beamGather,
                                        const std::vector<Film>* seedFilms = nullptr,
                                        const EnergyReport* seedEnergy = nullptr);
// Trace N more photons into the resident films (async on the device; downloads sync).
// seedBase must be the cumulative photon count already traced (same convention as
// renderForwardSharedCuda) so every batch draws an independent stream.
void sharedForwardGpuBatch(SharedGpuSession* s, long long N,
                           unsigned long long seedBase, bool diffraction);
// Cheap per-batch noise poll: download ONLY camera 0's per-pixel hit counts (running
// totals) into `hits` (must already be sized to camera 0's pixel count).
void sharedForwardGpuHits0(SharedGpuSession* s, std::vector<double>& hits);
// Download the running TOTAL films (xyz + hits, including any resume seed) into
// `films[c]` (each must already be alloc'd at its camera's resolution) and REPLACE
// `e` with the total energy report.
void sharedForwardGpuDownload(SharedGpuSession* s, std::vector<Film>& films,
                              EnergyReport& e);
void sharedForwardGpuEnd(SharedGpuSession* s);

// True if this scene can be rendered by the GPU photon-map path (mode M). Requires the
// same POD-bakeable materials as the forward path (cudaForwardSupported) and no
// environment light (the device gather has no env term). Final gather and physical-lens
// cameras are render-config properties gated by the caller, not here.
bool cudaPhotonMapSupported(const Scene& scene);

// GPU shared photon-map render (mode M): build ONE view-independent photon map on the
// device (forward deposit pass over N photons) and gather every camera from it — the
// device twin of the CPU shared runSharedPhotonMap. `radius` is the gather radius (== grid
// cell size). Returns one film per camera (each at resX[c] x resY[c]), accumulated as a
// SUM over `spp` (display divides by spp: writeFilm(film, spp)). Only the DIRECT density
// estimate is implemented (final gather stays on the CPU). Requires cudaAvailable() &&
// cudaPhotonMapSupported(scene) and pinhole cameras; otherwise returns empty films.
// When `prog` is non-null the per-camera gather reports its converging film through
// prog->report(sumFilm, sppDone, final) — the same hook modes R/D use — so the host can
// drive the live window / preview as each frame builds up (final is true on the chunk that
// completes a camera). Returning true from report() stops the render after the current
// chunk (e.g. the live window was closed). A null `prog` renders silently as before.
//
// `onFrame` (when non-null) is called ONCE per camera, right after that camera's gather
// fully completes, with its local index and finished film — so the host can write that
// frame to disk IMMEDIATELY (crash-safe incremental output, matching the CPU mode-M path
// which writes each frame as it finishes) instead of holding all films to the end. When
// `onFrame` is supplied the returned vector's films are RELEASED as they are handed off
// (each returned Film[c] is left empty after the callback), so the whole render runs in
// roughly one-frame of host memory rather than accumulating every frame — the caller must
// therefore consume frames via `onFrame`, not the return value. Returning true from
// `onFrame` stops the render after the current frame. A null `onFrame` keeps every film in
// the returned vector for the caller to write at the end (the historical behaviour).
//
// `mapLoad`/`mapSave` (when non-null) drive the view-independent photon-map cache file. The
// deposited map is the expensive result of the forward photon trace and is independent of
// camera and gather radius, so it is worth persisting: `mapSave` writes it after the deposit,
// and `mapLoad` reloads it and SKIPS the deposit entirely — re-gathering new camera angles /
// a new radius for free, without re-tracing a photon. The file (magic "FTPMP01\n") holds the
// raw photon set + emitted count + energy; the grid is rebuilt on load at the requested
// `radius`, so one file serves any radius. A scene-identity guard rejects a stale map built
// for a different scene (it falls back to a fresh deposit). Both default null (no caching).
std::vector<Film> renderPhotonMapSharedCuda(const Scene& scene, const std::vector<Camera>& cams,
                                            const std::vector<int>& resX, const std::vector<int>& resY,
                                            long long N, double radius, EnergyReport& eOut,
                                            bool diffraction, long long spp,
                                            const SppProgress* prog = nullptr,
                                            const std::function<bool(int, const Film&)>* onFrame = nullptr,
                                            const char* mapLoad = nullptr, const char* mapSave = nullptr,
                                            int heroC = 1, int fgRays = 0);

// True if this scene can be rendered by the GPU BDPT megakernel (mode D). Stricter
// than cudaForwardSupported: also requires no participating media and only area/sphere/
// cylinder Lambertian emitters (no spot/env/collimated) — the BDPT scope. When false, the
// caller must use the CPU BDPT renderer.
bool cudaBdptSupported(const Scene& scene);

// GPU bidirectional path trace (mode D). Renders `spp` samples per pixel at the given
// resolution and returns the film accumulated as a SUM over spp (display divides by
// spp: writeFilm(film, spp)). maxDepth is the maximum path length in edges (clamped to
// the device capacity). Requires cudaAvailable() && cudaBdptSupported(scene); otherwise
// returns an empty film.
// When `prog` is non-null the spp is rendered in chunks: after each chunk prog->report()
// receives the running SUM film and the spp done so far, and can rewrite the image / show
// progress / request an early stop. A null `prog` renders all spp in one launch (the
// historical path). Either way the result is bit-identical for a given spp (the RNG is
// seeded on the global sample index, so chunking never changes the image).
Film renderBdptCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                    long long spp, int maxDepth, bool diffraction,
                    const SppProgress* prog = nullptr);

// True if this scene + camera can be rendered by the GPU backward reference megakernel
// (mode R), including the physical (mesh-lens) camera as a ray-generation front-end.
// Current scope: participating media (homogeneous + heterogeneous, incl. spectral-rainbow
// Airy phase; minus GRIN), fluorescence, a CONSTANT environment light, textured albedo, and
// area/sphere/cylinder Lambertian + point-spot emitters; a lens no deeper than the device
// cap (D_MAXLENS). Image-based env, collimated beams and GRIN media fall back to the CPU tracer.
bool cudaBackwardSupported(const Scene& scene, const Camera& cam);

// GPU backward reference trace (mode R). Renders `spp` samples per pixel at the given
// resolution and returns the film accumulated over spp (same convention as the CPU
// renderBackward: writeFilm(film, spp) for display). Handles the physical lens exactly
// as the CPU path (film + rear-pupil sample refracted through the glass stack, with the
// cos^4*A/Z^2 radiometric weight). Requires cudaAvailable() && cudaBackwardSupported();
// otherwise returns an empty film. The device RNG differs from the CPU, so the image is
// an independent noise realization that agrees to within Monte-Carlo noise.
// When `prog` is non-null the spp is rendered in chunks: after each chunk prog->report()
// receives the running SUM film and the spp done so far, and can rewrite the image / show
// progress / request an early stop. A null `prog` renders all spp in one launch (the
// historical path). Either way the result is bit-identical for a given spp (the RNG is
// seeded on the global sample index, so chunking never changes the image).
// maxBounce (< 1 => leave the device default of 32) caps the backward path depth
// (-max-bounce); directOnly (-direct-only) renders direct + specular recursion only.
Film renderBackwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                        long long spp, bool diffraction,
                        const SppProgress* prog = nullptr,
                        int maxBounce = 32, bool directOnly = false);

// True if this scene + camera can be rendered by the FAST RGB backward megakernel
// (mode R `-rgb`, Option B in gpu-backward-fast.md): the reduced non-spectral tracer
// that carries a linear-sRGB throughput and produces a full-colour image in ONE walk
// per sample. Scope is narrower than cudaBackwardSupported — no participating media,
// no dispersion-dependent materials (thin-film / grating / multilayer / layered /
// fluorescence), only constant (untextured, non-record) per-material reflectance, no
// image env / collimated emitters. When false the caller falls back to the spectral
// backward (renderBackwardCuda) or the CPU tracer.
bool cudaBackwardRGBSupported(const Scene& scene, const Camera& cam);

// Fast RGB backward trace (mode R `-rgb`). Same spp/chunk/checkpoint conventions and
// film accumulation as renderBackwardCuda, but each sample does a single colour walk
// (no wavelength dimension), so a clean colour image converges far faster. A neutral
// (spectrally flat) scene matches the spectral estimator's absolute luminance; colour
// carries the Option-B metamerism approximation. Requires cudaBackwardRGBSupported();
// otherwise returns an empty film.
Film renderBackwardRGBCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                           long long spp, bool diffraction,
                           const SppProgress* prog = nullptr,
                           int maxBounce = 32, bool directOnly = false);

// ---- Resident fast-RGB backward PREVIEW session (interactive -explore) ----------------
// The batch renderBackwardRGBCuda re-uploads the whole scene, re-bakes the camera, allocs
// films, launches, downloads and frees on every call — fine for one shot, wasteful when an
// interactive viewer wants to progressively converge a still while the camera holds, then
// re-aim and start over on the next move. A session makes the scene bake RESIDENT: begin()
// uploads the (already ignore-flag-stripped) scene and allocs a persistent SUM film ONCE;
// setCamera() re-bakes just the pinhole camera (cheap POD) and zeroes the film to start a
// fresh accumulation; accumulate() launches kBackwardRGB adding `spp` more samples into the
// resident film (running total, distinct RNG streams per batch); download() fetches the
// running SUM (the caller divides by samples() and tone-maps via filmToRgb8, exactly as the
// batch path does); end() frees. Scene-ignore flags need no special handling — they mutate
// the Scene host-side before begin(), so the baked device scene already reflects them.
struct BackwardRGBSession;   // opaque; lives in render_cuda.cu
// Returns nullptr if CUDA is unavailable. `maxBounce` (<1 => device default 32) and
// `directOnly` are the Stage-3 knobs, baked into the resident DScene. The scene must pass
// cudaBackwardRGBSupported() for the chosen camera(s); callers gate on that first.
BackwardRGBSession* backwardRGBSessionBegin(const Scene& scene, int resX, int resY,
                                            int maxBounce = 32, bool directOnly = false);
// Re-aim: bake `cam` into the resident scene and reset the accumulation (samples()->0).
void backwardRGBSessionSetCamera(BackwardRGBSession* s, const Camera& cam);
// Trace `spp` more samples-per-pixel into the resident SUM film; returns the new running
// total spp. `diffraction` matches the batch path's flag (unused by the RGB walk today).
long long backwardRGBSessionAccumulate(BackwardRGBSession* s, long long spp, bool diffraction);
// Download the running SUM film (xyz + hits) into `out` (already alloc'd at resX*resY).
void backwardRGBSessionDownload(BackwardRGBSession* s, Film& out);
// Samples-per-pixel accumulated since the last setCamera() (0 right after a re-aim).
long long backwardRGBSessionSamples(const BackwardRGBSession* s);
void backwardRGBSessionEnd(BackwardRGBSession* s);

// ---- GPU SPPM (mode S) ----------------------------------------------------------------
// Resident device SPPM session: keeps per-pixel progressive state (tau / radius / nAcc /
// directSum + this pass's visible point) on the device across passes, so the mode-S driver
// calls one pass per iteration and resolves the current radiance whenever it wants a
// preview / checkpoint frame. Each pass (1) resamples camera visible points, (2) deposits a
// bounded photon set via the SAME forward tracer as mode M and host-builds the grid at the
// largest current per-pixel radius, and (3) gathers + progressively updates every pixel.
// Device twin of sppm_render.h; validated statistically against the CPU (independent MC).
struct SppmSession;   // opaque; lives in render_cuda.cu
// True when the scene is device-bakeable for SPPM: same scope as the photon map (mode M,
// which now includes constant + image env). Pinhole cameras only (dGenRay) — the caller
// gates the camera choice.
bool cudaSppmSupported(const Scene& scene);
// Returns nullptr if CUDA is unavailable or the scene is out of scope. `R0` is the initial
// per-pixel gather radius, `maxBounce` (<1 => device default 32) the specular-walk cap.
SppmSession* sppmSessionBegin(const Scene& scene, const Camera& cam, int resX, int resY,
                              double R0, bool diffraction, int maxBounce, int heroC);
// Run one SPPM pass: resample visible points, deposit `photonsPerPass` fresh photons, gather
// at each pixel's current radius, and apply the shared-statistics radius/flux update.
void sppmSessionPass(SppmSession* s, long long photonsPerPass, double alpha);
// Resolve the current accumulated state into `out` (radiance L, exactly like sppmResolve).
void sppmSessionResolve(SppmSession* s, Film& out);
long long sppmSessionPasses(const SppmSession* s);
long long sppmSessionEmitted(const SppmSession* s);
void sppmSessionEnd(SppmSession* s);

// ---- GPU VCM / UPS (mode U) ------------------------------------------------------------
// Resident device VCM session mirroring vcm.h's vcmPass. Each pass traces one light + one
// camera subpath per pixel, combining BDPT vertex CONNECTIONS with photon-map vertex MERGING
// under one balance-heuristic MIS. Light vertices are stored into a per-path slab on the
// device, downloaded and compacted host-side into contiguous per-path ranges, and a uniform
// hash grid (cell = merge radius) is built over them (byte-for-byte mirror of vcm.h
// VcmGrid::build); the camera kernel then does emission/NEE/connection/merge and accumulates
// the running per-pixel sum. Resolve divides by the pass count. Validated statistically
// against the CPU (independent MC).
struct VcmSession;   // opaque; lives in render_cuda.cu
// True when the scene is device-bakeable for VCM: the BDPT device scope (mode D) PLUS a
// no-participating-media restriction (mode U is surfaces-only). Pinhole cameras only
// (cam.project) — the caller gates the camera choice.
bool cudaVcmSupported(const Scene& scene);
// Returns nullptr if CUDA is unavailable or the scene is out of scope. `maxDepth` (<1 =>
// default 8) is the full path length in edges (also the per-light-subpath stored-vertex cap).
VcmSession* vcmSessionBegin(const Scene& scene, const Camera& cam, int resX, int resY,
                            bool diffraction, int maxDepth);
// Run one VCM pass at the given merge `radius` (the caller shrinks it per the progressive
// schedule r_i = R0 * i^((alpha-1)/2)), accumulating into the resident per-pixel sum.
void vcmSessionPass(VcmSession* s, double radius);
// Resolve the current accumulated state into `out` (running average = accum / passes).
void vcmSessionResolve(VcmSession* s, Film& out);
long long vcmSessionPasses(const VcmSession* s);
void vcmSessionEnd(VcmSession* s);

// True if this scene + camera can be rendered by the GPU isosurface PREVIEW kernel
// (G2, `-raster-gpu`): a usable CUDA device, a POD-bakeable scene (cudaForwardSupported),
// and a non-physical camera (dGenRay handles pinhole + fisheye/panoramic; a mesh-lens
// camera falls back to the CPU raster). When false the caller must use raster::renderFrame.
bool cudaIsoPreviewSupported(const Scene& scene, const Camera& cam);

// GPU deterministic primary-ray isosurface PREVIEW (G2). Casts one pixel-centre primary
// ray per pixel, finds the nearest surface with the shared closestHit (which sphere-traces
// implicit isosurfaces directly — NO tessellation), and shades it once with the same solid
// preview model as raster::renderFrame (flat per-material albedo, ambient + Σ weighted N·L
// keys + a headlight fill). Returns W*H*3 RGB8 (row 0 = image top), tone-mapped by the
// SHARED raster::exposeAndEncode so it matches the CPU `-raster` output and honours a
// camera_path's locked auto-exposure anchor. Returns an EMPTY vector on any device failure
// or unsupported config so the caller can fall back to the CPU rasterizer.
#include <vector>
#include <cstdint>
std::vector<uint8_t> renderIsoPreviewCuda(const Scene& scene, const Camera& cam,
                                          int W, int H, int nThreads, double exposure = 1.0,
                                          bool autoExpose = true, double* lockAnchor = nullptr);
