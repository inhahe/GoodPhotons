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
                                          bool wavefront = false, int heroC = 1);

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
                                            int heroC = 1);

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
// v1 scope: no participating media, no environment light, only area/sphere/cylinder
// Lambertian emitters (no spot/env/collimated), no fluorescence, and a lens no deeper
// than the device cap (D_MAXLENS). Textured albedo IS supported. When false, the
// caller must use the CPU backward tracer (which has no such restrictions).
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
Film renderBackwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                        long long spp, bool diffraction,
                        const SppProgress* prog = nullptr);

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
