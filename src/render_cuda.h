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
namespace bdpt { struct SurfMap; }   // surfmerge.h: mode J's point x point merge map
struct CudaSceneHold;                 // cudaHoldScene, below: a device copy of a scene kept across calls

// True if a usable CUDA device is present (driver + at least one device). Cheap to
// call; result is cached after the first query.
bool cudaAvailable();

// Human-readable name of the primary CUDA device (or "none").
const char* cudaDeviceName();

// Free / total device memory, in bytes. False (leaving the outputs untouched) if there is
// no device or the query fails. Note `free` is the WHOLE card's free pool, i.e. it moves
// with every other process on the machine, not just this one.
bool cudaMemInfo(size_t* freeB, size_t* totalB);

// THE KERNELS A GPU RENDER LAUNCHES, for the VRAM preflight to weigh. A route lists them as its
// launcher chooses them, each from the decision the launcher itself runs (render_cuda.cu: the forward
// walk's backend, BDPT's instantiation, mode U's bundle), so that what is weighed and what launches
// cannot part. Routes add (mode P's two layers).
struct GpuRoute {
    std::vector<const void*> kernels;
    GpuRoute& operator+=(const GpuRoute& o) {
        kernels.insert(kernels.end(), o.kernels.begin(), o.kernels.end());
        return *this;
    }
};
// The device memory a route's kernels reserve for per-thread LOCAL storage alone: the most of their
// local sizes times the threads the card holds resident. This is real VRAM and it is often
// gigabytes, and the context keeps it after the kernel returns, for the next launch to reuse. If the
// card cannot hold it, the driver silently spills it to host RAM over PCIe and the kernel runs 2-3
// orders of magnitude slower at 100% reported utilisation -- see main.cpp's vramPreflight. Returns
// 0 for "unknown" (no device, or an empty route), which callers treat as no opinion, not as free.
size_t cudaRouteLocalBytes(const GpuRoute& r);
// The headroom the preflight keeps beside a render's kernels and buffers for the scene upload and the
// driver's own context, and a session sizing a buffer from the free memory keeps too.
constexpr size_t kVramHeadroomBytes = (size_t)256 * 1024 * 1024;
// The forward photon walk (modes A, B, C, V, mode P's forward layer, a shared flight of `nCam`
// cameras gathering photon beams by `beamGather`): the megakernel, or the wavefront's three kernels
// where `wavefront` (-wavefront) is asked and nothing keeps the megakernel -- the hero bundle the
// scene keeps of `heroC`, a flight's beams, mode P's stable seed (`stableSeed`).
GpuRoute cudaForwardRoute(const Scene& scene, bool wavefront, int heroC, bool beamGather = false,
                          int nCam = 1, bool stableSeed = false);
// Whether that walk runs on the wavefront backend (cudaForwardRoute's choice).
bool cudaForwardRunsWavefront(const Scene& scene, bool wavefront, int heroC, bool beamGather = false,
                              int nCam = 1, bool stableSeed = false);
// The backward walk (modes R and W, mode P's camera-side layers): the RGB previewer's kernel where
// `rgb` (-rgb, in its scope on the device), the spectral one otherwise.
GpuRoute cudaBackwardRoute(bool rgb);
// BDPT (modes D and J, the meter's D and J frames): renderBdptCuda's instantiation for `maxDepth`
// and `heroC` (the bundle off through a medium, GRIN or a camera `lens`: bdptPick) or, where mode J
// merges (`merging`), the merge instantiation -- with the beam queue's variant and kernels where the
// beams queue (`beamQueue`), and -jsurf's light pass and surface gather (`surfMerge`).
GpuRoute cudaBdptRoute(const Scene& scene, bool lens, int maxDepth, int heroC, bool merging = false,
                       bool beamQueue = false, bool surfMerge = false);
// Mode M: the photon walk and the gather. Mode S: the walk, the visible points and the gather.
GpuRoute cudaPhotonMapRoute();
GpuRoute cudaSppmRoute();
// Mode U: the session's light, camera and merge kernels, at the bundle it carries for `heroC`.
GpuRoute cudaVcmRoute(int heroC);

// Device memory mode J's beam queue takes beside the kernel's local memory, where beams gather.
// (The point merges' queue sizes itself to the room left, or stays off.)
size_t cudaModeJBeamQueueBytes();

// The local-memory reservation this process's context holds now: its per-thread stack limit
// (which the driver raises to the largest kernel launched so far) times the threads the card
// holds resident. A kernel launched later reuses it. 0 if unknown.
size_t cudaLocalReservedBytes();
// The device memory a mode-S session keeps a pixel (sppmSessionBegin), for the VRAM preflight.
size_t cudaSppmPixelBytes();

// Orderly CUDA teardown: synchronize any outstanding device work, then destroy the
// primary context (cudaDeviceReset) while the process is still in a clean, quiescent
// state. Call this once, at the very end of a successful (or failed) run, BEFORE the
// process exits. Rationale: leaving the context to be reclaimed implicitly at process
// exit means the NVIDIA kernel driver tears it down asynchronously, from a DPC, after
// main() has already returned — and buggy nvlddmkm builds can fault there and bugcheck
// the whole machine (observed as BSOD/reboot "a couple seconds after the window closed",
// worst with an abrupt kill). Draining + resetting synchronously here removes that
// window: by the time we return there is no live GPU work and no context left to reap.
// No-op (and safe to call) when CUDA is unavailable or was never initialised -- unless
// `force`, which says another user of the driver in this process (OIDN on the GPU) has had a
// context: the device is then brought up and reset anyway, so no context is left to reap.
void cudaGracefulShutdown(bool force = false);

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
// `pSplit`: mode P's blurred split -- the forward layer makes no splat the camera side draws (no
// analytic specular connection, no glossy lobe's on a specular-type surface; DCamSet::pSplit).
// `stableSeed`: each photon's stream from its absolute index (`seedBase` the photons before this
// call), so a render split into calls anywhere traces the same photons (mode P's batches); it
// runs on the megakernel. `held`, when non-null, is cudaHoldScene's copy of this same `scene`: the
// call renders against it instead of uploading the scene itself (mode P's batches).
Film renderForwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                       long long N, EnergyReport& eOut, bool diffraction,
                       char camMode, unsigned long long seedBase = 0,
                       bool wavefront = false, int heroC = 1, bool pSplit = false,
                       bool stableSeed = false, const CudaSceneHold* held = nullptr,
                       int maxBounce = 32);

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
                                          bool beamGather = false, int maxBounce = 32);

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
                                        bool beamGather, int maxBounce,
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
// completes, with its local index, its film, and the samples-per-pixel that ACTUALLY landed
// in it — so the host can write that frame to disk IMMEDIATELY (crash-safe incremental
// output, matching the CPU mode-M path which writes each frame as it finishes) instead of
// holding all films to the end. `sppDone` is normally the requested spp, and is less on the
// frame an `ftrace -stop` / a closed window interrupts; the film is a SUM over samples, so
// the host must normalise by `sppDone` and not by the requested count or an interrupted
// frame is written darkened by exactly the fraction it never gathered. `sppDone == 0` means
// the stop landed before a single complete sample existed: there is nothing to write, and
// the callback is NOT invoked at all in that case. When
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
// a new radius for free, without re-tracing a photon. The file (magic "FTPMP02\n") holds the
// raw photon set + emitted count + energy; the grid is rebuilt on load at the requested
// `radius`, so one file serves any radius. A scene-identity guard rejects a stale map built
// for a different scene (it falls back to a fresh deposit). Both default null (no caching).
//
// `autoK` > 0 turns on the density-adaptive gather radius: `radius` becomes only a STARTING
// point, and the grid is rebuilt at whatever radius makes a typical gather see the target
// population (PhotonMap::buildAuto, photonmap.h — see there for why the target scales as the
// cube root of the stored count). 0 = off, use `radius` exactly. This must be honoured on the
// GPU too, not just the CPU: the shared-map path IS the high-photon-count path, which is
// exactly where a count-independent radius collapses. The gather reads pm.radius (not this
// argument) after the build, so the adapted value flows through with no further plumbing.
//
// `beams` (when non-null) turns on the photon-BEAM volume pass (`-beams`) — the
// view-independent single-scatter gather that is the only thing making a participating medium
// visible in mode M (the surface map stores no volume records). The beams MUST come from the
// SAME forward trace as the surface map, because `-beams` changes the transport: a photon
// crosses a medium straight instead of scattering (Renderer::doBeamStraight). So the device
// deposits both in one pass and downloads the crossings into `beams->map`.
//
// Everything scene-scale about the beam map stays on the HOST: the exact unbiased trim to
// `beams->target` (-beamcount), then `beams->build`, which owns the kernel radius, the
// area-optimal split and the BVH (BeamMap::buildAuto). That reasoning has no place in a kernel
// and is shared with the CPU path verbatim rather than reimplemented. After the callback
// returns, the built map is uploaded once and every camera's gather sees it.
//
// `stage` (when non-null) reports the phases that precede the first pixel — chiefly the
// photon deposit, with a running photon count — so the live window's title bar keeps moving
// through them instead of freezing on one caption for minutes. It is also what makes the
// deposit CHUNKED: with no reporter the whole `-n` goes out as a single kernel launch (the
// historical, bit-identical path), while a reporter splits it into adaptively-sized launches
// so a count can be read between them. See the deposit loop in render_cuda.cu for why
// chunking is safe (photon power is absolute, so the normalisation does not depend on the
// split; the atomic deposit cursor accumulates across launches; and each chunk draws its own
// seed, with chunk 0 reproducing the single-shot stream exactly).
//
// `causticK` > 0 turns on the Jensen two-map CAUSTIC SPLIT: every deposit reached by an
// L.S+.D path (at least one focusing vertex, no scattering one — see dPhotonVertexBit) is
// partitioned into a SECOND photon map, built at its own adaptive radius targeting `causticK`
// photons per gather. That is the whole point of the split: a single radius is sized by the
// majority population, which is the broad ambient wash, and it convolves a caustic — a thin,
// high-contrast concentration — into a flat smear. The two sets are disjoint and share
// nEmitted, so the gather simply adds their density estimates. 0 = off (one map, as before).
//
// `aim` + `nAimed` add Jensen's PROJECTION-MAP half on top of that split (causticaim.h): a
// SECOND deposit launch of `nAimed` photons whose emission is importance-sampled towards the
// scene's focusing geometry, storing into the caustic map only. It is a pure variance
// reduction and not a replacement — both launches' caustic deposits carry the same
// balance-heuristic weight, computed at emission from the ratio nAimed/N, and the caustic
// map's nEmitted stays at the MAIN launch's count — so an incomplete or over-eager target set
// costs efficiency, never correctness, and `nAimed == 0` leaves every deposit bit-for-bit what
// it was. Host twin: the aimed pass in tracePhotonPass (photonmap_render.h).
//
// `causticAdaptK` switches the caustic map's gather from one radius per MAP to one radius PER
// QUERY (PhotonMap::adaptiveRadius / dPmAdaptiveRadius): 0 = off (fixed radius, as before),
// < 0 = on with the same target `causticK` implies, > 0 = on with this explicit target. The
// split alone is not enough on a scene whose caustic map ALSO holds a broad specular wash —
// the map-wide probe is decided by the wash and the filament is smeared anyway. Host twin:
// buildCausticMap in main.cpp.
struct BeamPass {
    BeamMap*  map    = nullptr;   // receives the deposited (or -loadmap'd) beams, then built
    long long target = 0;         // -beamcount: exact unbiased trim; <= 0 keeps every crossing
    std::function<void(BeamMap&)> build;   // host build (radius + split + BVH); required
    bool      loadedMissing = false;       // out: -loadmap succeeded but the file held no beams
};

// PINNED GATHER RADII FOR THE LIGHT-SIDE REFRESH (main.cpp's mode-M epoch loop, 0.253.0).
//
// Default-constructed and handed to the FIRST pass: that pass adapts its radii exactly as it
// always did and records what it settled on here. Every later pass is given the same struct
// back, sees non-zero radii, and re-bins at them instead of re-probing.
//
// This is a correctness requirement, not a tidiness one. Photon mapping is BIASED at a finite
// radius, so epochs that each re-adapt would be estimating slightly DIFFERENT quantities and
// their average would be a mixture of estimators rather than variance reduction on one. Pinned,
// every epoch is the same estimator, the bias stays exactly where the first pass put it, and
// the average is plain 1/sqrt(epochs). (Host twin: rebuildPhotonMapAt / rebuildCausticMapAt in
// main.cpp, which pin the CPU mode-M path for the same reason.)
//
// `kGather` has to travel too: it is the caustic map's per-query adaptive target, PhotonMap
// does not derive it from the radius, and a re-bin that dropped it would silently return the
// caustic gather to a fixed radius.
struct PmRadiiPin {
    double radius  = 0.0;   // global map gather radius (0 = not decided yet: adapt and record)
    double radiusC = 0.0;   // caustic map gather radius
    double kGather = 0.0;   // caustic per-query adaptive target (0 = fixed radius)
    // Out, every pass: seconds from the call to its first gather, the light side's cost (upload,
    // deposit, map and beam builds), which a clock blur's epochs are sized by (BLUR-EPOCH-COST).
    double lightSec = 0.0;
};

// (g_gpuQuietRebuild — the refresh-epoch narration gate — lives in render_progress.h, because
// mode J's epoch loop sets it from code that is not inside #ifdef HAVE_CUDA.)

// `held`, when non-null, is cudaHoldScene's copy of this same `scene` (below): the call gathers
// against it instead of uploading the scene itself, as the refresh loop's epochs do.
struct CudaSceneHold;
std::vector<Film> renderPhotonMapSharedCuda(const Scene& scene, const std::vector<Camera>& cams,
                                            const std::vector<int>& resX, const std::vector<int>& resY,
                                            long long N, double radius, EnergyReport& eOut,
                                            bool diffraction, long long spp,
                                            const SppProgress* prog = nullptr,
                                            const std::function<bool(int, const Film&, long long)>* onFrame = nullptr,
                                            const char* mapLoad = nullptr, const char* mapSave = nullptr,
                                            int heroC = 1, int fgRays = 0, double autoK = 0.0,
                                            BeamPass* beams = nullptr,
                                            const StageProgress* stage = nullptr,
                                            double causticK = 0.0,
                                            const caim::AimMap* aim = nullptr,
                                            long long nAimed = 0,
                                            double causticAdaptK = 0.0,
                                            PmRadiiPin* pin = nullptr,
                                            const CudaSceneHold* held = nullptr);

// True if this scene can be rendered by the GPU BDPT megakernel (mode D). Stricter
// than cudaForwardSupported: also requires no participating media and only area/sphere/
// cylinder Lambertian emitters (no spot/env/collimated) — the BDPT scope. When false, the
// caller must use the CPU BDPT renderer.
// `-jhostlight`: keep mode J's surface light side on the HOST, redrawn once per epoch, which is
// what it did before 0.272.0. The default is the device pass redrawn every few samples -- 11.2x less
// whole-frame variance at equal wall clock on a surfaces-only scene. This is the switch for
// reproducing pre-0.272.0 output.
void cudaSetJHostLight(bool on);
// Per-object render modes (0.386.0, Material::route / Medium::route): routing on, the camera's
// mode (ROUTE_D / ROUTE_J), whether any medium routes J, and each mode's sample multiplier
// (`mode_spp`). runRender sets it per D/J render.
void cudaSetRouting(int on, int routeDefault, int anyMedJ, double sppD, double sppJ);
// The aimed sun's target capacity on the device (its constant arrays). The host clusters the
// targets to bdpt::SunAim::kMaxTargets, which main.cpp asserts equals this.
constexpr int kSunAimMaxTargets = 64;
// JSPEC's dispersive window (0.386.2, bdpt g_jDispWinNm / g_jDispAll): its full width in nm, and
// whether dispersive refraction and a real lens open it as a grating's non-zero order does.
void cudaSetJDisp(double winNm, int all);
// Mode J's estimator switches, set per render from the host's own (bdpt.h holds them; the
// device has no copy of their environment strings or tests): the JSPEC carrier
// (bdpt::jSpecEnabled), whether some medium's sigma_t is chromatic
// (bdpt::anyMediumSigmaTChromatic), a delta bounce's light density read as 1
// (bdpt::g_jDeltaMerge), and the half-render diagnostic (bdpt::jHalfMode: 0 full, 1
// connections, 2 merges; the device has no merges-raw).
void cudaSetJSwitches(int jSpec, int trSpec, int deltaMerge, int half, int chroma);
// The aimed sun for modes D/J (MODE-J-SDS, 0.387.0; bdpt.h SunAim): the target spheres (centres
// relative to the scene centre as x,y,z triples, radii), SUM r^2 and the aimed share. n == 0 or
// f == 0: the uniform disc.
void cudaSetSunAim(const double* wxyz, const double* r, int n, double sumR2, double f);

bool cudaBdptSupported(const Scene& scene);

// A device copy of one Scene, kept across several renderBdptCuda calls. Uploading a scene is a
// fixed cost per call (1.6 s on gallery_rain, more than a 640x360 spp of rendering), and mode J
// calls once per light-side epoch, so a frame that redraws its light side often pays it over and
// over for a scene that has not changed. The copy is read-only to every kernel (DScene holds
// only const pointers), so reusing it cannot carry state from one call to the next. The caller
// must not modify `scene` while the hold lives: the copy does not follow it. Returns null when
// no device is available. cudaReleaseScene(null) is a no-op.
struct CudaSceneHold;
CudaSceneHold* cudaHoldScene(const Scene& scene);
void cudaReleaseScene(CudaSceneHold* held);
// Whether `held` is a copy of `scene` as it is now: the same object, the same load
// (Scene::generation) and the same build (Scene::buildCount). False for null.
bool cudaSceneHoldFits(const CudaSceneHold* held, const Scene& scene);
// The device memory `held` occupies (0 for null): this process's own, for the VRAM preflight.
size_t cudaSceneHoldBytes(const CudaSceneHold* held);

// Whether renderBdptCuda traces and grids mode J's surface photon map on the device itself (the
// default; -jhostlight or FTRACE_JDEVLIGHT=0 keep it on the host). Then it reads only the host
// map's photon count, emission count and radius, so the host need not grid it
// (bdpt::SurfMap::radiusOnly).
bool cudaJSurfOnDevice();
// ...and whether it needs nothing of the host's map but its count and radius: not under the
// diagnostic arms that compare the host map with the device's (FTRACE_JDEVLIGHT=2 or 4), which
// need the host to have traced and stored it. Mode J's device-drawn surface map
// (BeamBudgetReq::hostSurf) asks this.
bool cudaJSurfHostless();

// The fixed cost of the last renderBdptCuda call, in seconds: everything but its chunks (the
// setup before the first launch and the teardown after the last). Mode J's epoch loop reads
// it to weigh one light-side realization per sample against the samples themselves.
double cudaLastBdptOverheadSec();
// The camera-side work of the last renderBdptCuda call, in device units (render_cuda.cu
// g_dWork: the vertices it traced, the connections it could evaluate, the photons and beam hits
// its merges weighed). Deterministic: mode J's counted refresh reads it, weighed by
// bdpt::kJWorkDeviceUnit, where the CPU reads bdpt::jWork().cam (GPU-J-CLOCK).
long long cudaLastBdptWork();
// ...of which the call's own surface map, drawn for its setup (one a call, whatever its samples):
// the light side's, which the counted refresh charges there. Set once that map is drawn, so a
// call's own SppProgress callbacks read it too.
long long cudaLastBdptSetupWork();
// cudaLastBdptWork's count so far, from inside a renderBdptCuda call -- its SppProgress
// callbacks, between chunks: mode J's epoch 0 prices a sample on its first chunk. 0 outside one.
long long cudaBdptCallWork();

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
// `heroC` (`-heroc N`) > 1 carries a hero wavelength plus C-1 stratified secondaries down
// BOTH subpaths of every sample (one shared BVH walk, one shared MIS weight, sharply lower
// chroma noise). It is clamped to hero::kHeroMax and silently dropped to 1 for scenes the
// hero walk does not cover (participating media, GRIN, a physical lens) — the same gate the
// CPU BDPT applies. heroC <= 1 reproduces the original single-λ kernel bit-for-bit.
//
// `bmap` non-null is MODE J (UPBP): the caller's already-built photon-beam map, uploaded with
// its MIS partials and MIS-combined with the connections inside the kernel. Null is mode D,
// and then the kernel is bit-for-bit the mode-D one (no segment recording, no merge terms —
// the merge machinery is a template parameter, not a runtime branch, so mode D does not even
// pay the occupancy). A non-null but EMPTY map (`-nobeams`) also degenerates to exactly mode
// D, which is validation gate 1. `stage` reports progress during the (potentially very long)
// host-to-device conversion of a multi-million-sub-beam map.
// `smap` is mode J's OTHER merge kind, the point x point surface merges of `-jsurf` (surfmerge.h);
// null or empty leaves the estimator exactly two-technique, which is validation gate 1's premise.
// `held`, when non-null, is cudaHoldScene's copy of this same `scene`: the call renders against
// it instead of uploading the scene itself.
Film renderBdptCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                    long long spp, int maxDepth, bool diffraction,
                    const SppProgress* prog = nullptr, int heroC = 1,
                    const BeamMap* bmap = nullptr, const StageProgress* stage = nullptr,
                    const bdpt::SurfMap* smap = nullptr, const CudaSceneHold* held = nullptr);

// True if this scene + camera can be rendered by the GPU backward reference megakernel
// (mode R), including the physical (mesh-lens) camera as a ray-generation front-end.
// Current scope: participating media (homogeneous + heterogeneous, incl. spectral-rainbow
// Airy phase; minus GRIN), fluorescence, a CONSTANT environment light, textured albedo, and
// area/sphere/cylinder Lambertian + point-spot emitters, and a physical lens of any depth.
// Image-based env, collimated beams and GRIN media fall back to the CPU tracer.
bool cudaBackwardSupported(const Scene& scene, const Camera& cam);

// Motion blur on the device (ROADMAP 11): for a camera that sees motion (Camera::seesMotion),
// rendering in `mode`, null when the device can carry the blur, else why it renders on the CPU.
// Every mode but V: every primitive kind moves (rigid keys and deformers), and the camera and the
// lights may, each camera path (mode D: each sample's two subpaths; A, B and C: each photon and its
// splats) at its own time in the shutter, and each pass of S and U and each epoch of J and M at one
// time. Where the run's routing, not the mode, keeps a blurred render off the device -- a shared
// forward flight (kSharedFlightBlurWhy), a moving scene's shared or saved mode-M map -- main.cpp
// says so itself.
const char* cudaBlurUnsupported(const Scene& scene, const Camera& cam, char mode);
// Device memory a blurred single-camera forward render (modes A, B and C) of a camera that moves
// takes for its per-thread posed copies of the camera (DCamSet::cam), on the backend
// renderForwardCuda will run (`wavefront` asked, `heroC` the hero bundle asked): one a thread of
// the megakernel's grid, or one a slot of the wavefront's pool for an N-photon trace -- for the
// VRAM preflight.
size_t cudaForwardPosedCameraBytes(const Scene& scene, bool wavefront, int heroC, long long N,
                                   bool stableSeed = false);
// Device memory the wavefront backend's photon pool takes for an N-photon trace, where the forward
// walk runs there (cudaForwardRunsWavefront's arguments), else 0 -- for the VRAM preflight.
size_t cudaForwardWavefrontBytes(const Scene& scene, bool wavefront, int heroC, long long N,
                                 bool beamGather = false, int nCam = 1);
// The same for a blurred shared forward flight of `nCam` cameras (sharedForwardGpuBegin's arguments)
// whose set moves: every camera a thread, sized once for the session's largest batch.
size_t cudaSharedFlightPosedBytes(const Scene& scene, int nCam, bool wavefront, int heroC, bool beamGather);

// ---- Mode W (deterministic Whitted / POV-Ray-style preview) ----------------------------
// Device twin of BackwardRenderer's mode-W fields (src/backward.h). Passing a non-null
// WhittedOpts to renderBackwardCuda replaces every stochastic estimator on the backward
// path with fixed quadrature, exactly as `-mode W` does on the CPU: an N*N light grid
// instead of one random emitter point, a fixed throughput cutoff (POV-Ray's adc_bailout)
// instead of Russian roulette, the dominant branch at every stochastic material fork, and
// radical-inverse lattices (shared by every pixel, indexed by the ABSOLUTE sample index so
// the image is chunk-split-independent) for the subpixel offset, the wavelength and each
// glossy lobe.
struct WhittedOpts {
    int    grid      = 4;    // -whitted-grid N: N*N quadrature points per area light
    int    giDirs    = 0;    // -gi N: deterministic one-bounce gather directions (0 = ambient only)
    int    giGrid    = 1;    // light grid used INSIDE the gather (cheaper than `grid`)
    int    giBounce  = 4;    // bounce cap inside the gather
    bool   heroSplit = true; // mode W forces the split-at-dispersion walk (de-hero would
                             // collapse the whole frame onto one lambda: the lattice is shared).
                             // Ported to the device in v0.111.0 as bkRadianceHeroLoop<true>;
                             // plain mode R takes the same policy from `-herosplit`.
    double ambient   = 0.0;  // -ambient, ALREADY pre-scaled by Scene::ambientRef()
    double giClamp   = 0.0;  // -gi-clamp: per-lambda firefly ceiling on ONE gather ray's
                             // returned radiance, 0 = off. Also pre-scaled by ambientRef().
                             // Caps the caustic-through-the-gather contour aliasing; the
                             // rationale is on BackwardRenderer::giClamp (src/backward.h).
    double fwPerDist = 0.0;  // O8 stage 2: Camera::footprintPerDist(spp) — the world-space
                             // shading-footprint diameter at unit distance, head-on. The
                             // kernel scales it by the hit distance and the grazing factor
                             // to fill `fw` at the camera segment. 0 = leave `fw` unknown,
                             // i.e. fnoise() unfiltered. Twin of BackwardRenderer::fwPerDist.
};

// True if this scene + camera can be rendered by the GPU mode-W megakernel with the given
// options. As of v0.116.0 NOTHING mode-W-specific narrows this any more: it just forwards to
// cudaBackwardSupported. Dispersive materials stopped gating in v0.111.0 (the device carries
// the split-at-dispersion walk) and `-gi N` stopped gating in v0.116.0 (the one-bounce gather
// is a compile-time-bounded recursion, bkRadianceHeroLoop<AllowSplit, GiDepth>).
bool cudaBackwardWhittedSupported(const Scene& scene, const Camera& cam,
                                  const WhittedOpts& w);

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
// heroC (`-heroc N`) > 1 carries a hero wavelength plus C-1 stratified secondaries down
// each camera path (one shared BVH walk, sharply lower chroma noise); it is clamped to
// hero::kHeroMax and silently dropped to 1 for scenes the hero walk does not cover
// (participating media, GRIN, a physical lens) — the same gate the CPU tracer applies.
// `whitted` non-null selects the deterministic mode-W estimator (see WhittedOpts above) and
// additionally requires cudaBackwardWhittedSupported(); otherwise an empty film is returned.
// Note the "independent noise realization" caveat above does NOT apply in mode W: it has no
// noise, so the CPU and GPU images agree up to device fp32 precision and the intersection
// epsilon, with no structural difference.
// `specOut`, when non-null (mode P's blurred split), receives the samples whose own first event is
// a specular-side surface or the sky (an env scene's); the returned film holds the rest.
// `held`: as renderForwardCuda's.
Film renderBackwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                        long long spp, bool diffraction,
                        const SppProgress* prog = nullptr,
                        int maxBounce = 32, bool directOnly = false, int heroC = 1,
                        const WhittedOpts* whitted = nullptr, Film* specOut = nullptr,
                        const CudaSceneHold* held = nullptr);

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
// Resident device VCM session mirroring vcm.h's vcmPass. Each pass traces one camera subpath
// per pixel and nLight light subpaths -- one per pixel, or fewer when their vertices would not
// fit the light-vertex store's memory budget, each then shared by several camera paths --
// combining BDPT vertex CONNECTIONS with photon-map vertex MERGING under one balance-heuristic
// MIS. The light vertices are compacted on the device into contiguous per-path ranges and a
// uniform hash grid (cell = merge radius) is built over them (a mirror of vcm.h
// VcmGrid::build); the camera kernel then does emission/NEE/connection/merge and accumulates
// the running per-pixel sum. Resolve divides by the pass count.
struct VcmSession;   // opaque; lives in render_cuda.cu
// True when the scene is device-bakeable for VCM: the BDPT device scope (mode D) PLUS a
// no-participating-media restriction (mode U is surfaces-only). Pinhole cameras only
// (cam.project) — the caller gates the camera choice.
bool cudaVcmSupported(const Scene& scene);
// Returns nullptr if CUDA is unavailable, the scene is out of scope, or the GPU has no room
// for the film and one light subpath's store. `maxDepth` (<= 1 => 9) is the longest path in
// EDGES (-max-bounce + 1); a light subpath stores at most maxDepth - 1 vertices.
// `heroC` > 1 carries a hero-wavelength bundle of that width along BOTH subpaths (clamped to
// hero::kHeroMax); 1 is the classic single-λ session, which the templated kernels reproduce
// bit-identically and without allocating the per-vertex secondary slab at all.
VcmSession* vcmSessionBegin(const Scene& scene, const Camera& cam, int resX, int resY,
                            bool diffraction, int maxDepth, int heroC = 1);
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

// Optional per-call phase timing (milliseconds), so "why is the Render pane slow?" is
// answered by measurement instead of intuition. The three phases are kept separate
// because they scale with DIFFERENT things and react differently to another process
// contending for the card:
//   upload   host marshalling of the WHOLE scene (tris, BVH, materials, texels) + the
//            H2D copies. Scales with SCENE size, not pixels. Mostly CPU + DMA, so it is
//            the phase least distorted by a foreign process saturating the SMs.
//   kernel   the raymarch (launch + sync). Scales with PIXELS. The only SM-bound phase,
//            so it is the one inflated by GPU contention -- do not rank it against the
//            others without first checking the card is idle.
//   readback D2H of the accum/z/emissive buffers + the host tone map. Pixels; mostly CPU.
// A struct rather than one number so no phase can hide inside another -- the same reason
// the viewer's play breakdown shows its residual explicitly.
struct IsoPreviewTiming {
    double msUpload   = 0.0;
    double msKernel   = 0.0;
    double msReadback = 0.0;
};

namespace raster { struct PreviewLight; }   // raster.h (0.368.0: passed in, derived once)
// `showHidden`: draw `hide_camera` flats too (-raster-hidden); by default the primary ray skips
// them as the render's does (0.368.2).
std::vector<uint8_t> renderIsoPreviewCuda(const Scene& scene, const Camera& cam,
                                          int W, int H, int nThreads, double exposure = 1.0,
                                          bool autoExpose = true, double* lockAnchor = nullptr,
                                          IsoPreviewTiming* timing = nullptr,
                                          const raster::PreviewLight* light = nullptr,
                                          bool showHidden = false);

// ---------------------------------------------------------------------------
// N4a — mode-W deterministic-lattice bit-exactness probe (`-checklattice`).
//
// Evaluates the DEVICE twins of backward.h's sample lattices over a caller-supplied index
// sweep and hands back the raw bits for the host to compare against `BackwardRenderer`'s own
// statics. `out` receives `n * kLatticeProbeCols` doubles, row-major — see lattice_probe.h
// for the column layout and for why this is the one CPU/GPU comparison held to bit exactness.
//
// Returns false (leaving `out` untouched) if there is no usable CUDA device or the launch
// fails; the caller then reports the device half as SKIPPED rather than failed.
#include "lattice_probe.h"
bool cudaLatticeProbe(const unsigned long long* idx, int n, double* out);

// `-checkjsurfw` (jsurf_tap.h): mode J's point-merge weights, held against the host's.
// Arm: the next renderBdptCuda call records its first `n` samples (absolute indices below `n`):
// each one's camera subpath, up to `stride` vertices (at most kJSurfTapMaxV; its walk holds its
// depth + 3), and the camera half of every point merge's weight it built on it.
// Take: hands the record back and disarms -- per sample its vertices (`stride` apart), their
// count (0: not recorded) and its camera wavelength, and per vertex the half (camVert 0 where
// none was built). False if nothing was armed or there is no device.
#include "jsurf_tap.h"
void cudaJSurfTapArm(int n, int stride);
bool cudaJSurfTapTake(std::vector<JSurfTapVertex>& eye, std::vector<int>& nE,
                      std::vector<double>& lam, std::vector<JSurfTapHalf>& halves, int& stride);
// The device's own weight (DSurfMergeW, its bins read by dSpecBinAt) of each state against the
// half it names: out[i] for states[i]. False if there is no device or the launch failed.
bool cudaJSurfWEval(const std::vector<JSurfTapHalf>& halves,
                    const std::vector<JSurfTapState>& states, std::vector<double>& out);
// `-rainbow-selftest`'s device half (rainbow::RainbowPhase::samplerFit): `n` draws of the device's
// rainbow phase sampler (dMedPhaseSample) about +z at `lambda`, mu = cos theta histogrammed one bin
// per table cell into `counts`. False if there is no device, or it failed.
bool cudaRainbowSampleHist(const rainbow::RainbowPhase& rp, double lambda, long long n,
                           std::vector<double>& counts);

// Run a batch of pattern programs through BOTH device VMs and hand the results back for
// the host to compare against its own `patternEval` — the parity check behind `-checkpatops`.
//
// It exists because the pattern VM is implemented THREE times: `patternEval` on the host,
// the fp64 template `dPatternEval`, and `dPatternEvalF`, an FP32 twin that is what the
// sphere-trace march actually runs. Their switches have no `default:`, so an opcode one of
// them has never heard of is not a compile error and not a crash — the node is skipped, the
// stack is left short, and the program quietly returns the wrong slot. When that happened to
// an isosurface field it read 0 everywhere and the surface simply was not there, which looks
// like an empty scene rather than a broken shader. This turns that into a named failure.
//
// `prog` is a flat pool of `nProg` one-opcode programs described by `off`/`len`; `outF64`
// and `outF32` each receive `nProg` doubles. Returns false (leaving both untouched) if
// there is no usable CUDA device or the launch fails, so the caller reports SKIPPED rather
// than failed — the same contract cudaLatticeProbe keeps.
// The probe's inputs, in one struct so the host reference and the two device twins cannot
// drift apart: a check whose three sides disagree about what `y` is would report mismatches
// that mean nothing. Every value is distinctive and non-zero on purpose — an opcode a VM has
// never heard of is SKIPPED, which leaves the stack one short, and `patternEval` returns 0.0
// for an empty stack. Zero inputs would make "skipped" and "evaluated" look identical.
struct PatOpProbeIn {
    double x, y, z, f;
    double nx, ny, nz, r;
    double u, v, curv, cavity, fw;
    double d[9];              // d4..d12, the N-D slice coordinates
};

bool cudaPatOpProbe(const PatNode* prog, const int* off, const int* len, int nProg,
                    const PatOpProbeIn& in, double* outF64, double* outF32);

// -checkdevhits: rays traced on the device against `scene` (uploaded for the call), each through
// the device's closestHit (kind < 0) or through one intersector alone, handed a record full of
// another primitive's values as a closest-hit loop's reused record carries them (kind 0 sphere,
// 1 triangle, 2 curve segment, 3 isosurface; `index` into that DScene array, which keeps the
// host Scene's order). The hit's fields and the pattern inputs at it come back for the host to
// hold against its own intersectors, closestHit and patCtxFromHit.
struct DevHitProbeRay { double o[3], d[3]; int kind, index; };
struct DevHitProbeOut {
    int    valid, matId, back, patOwnSide, vcol, solid;
    double t, p[3], n[3], ng[3], pscale, fiberRadius, curv, tangentZ;
    double q[3], qn[3], qcurv;   // dPatIn: the point, normal and curvature a pattern reads there
    double cavity;               // dCavityOf: its `cavity`, where the material reads it
    double sng[3];               // dShadedNg: the geometric normal on the side being shaded
};
bool cudaHitProbe(const Scene& scene, const DevHitProbeRay* rays, int n, DevHitProbeOut* out);

// `sizeof(Real)` — the device's working float width (4 with the default FTRACE_GPU_FP32,
// 8 in an exact-FP64 build). The lattice helpers are `double` either way, but `gridUV`
// hands its result back as `Real`, so `-checklattice` must narrow the host's double to the
// same width before comparing bits. Also worth printing in any CPU/GPU parity report, since
// it is the single biggest reason whole-image bit-exactness is not on the table.
int cudaRealBytes();
