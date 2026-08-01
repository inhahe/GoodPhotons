// Forward spectral photon tracer — Phase 0 (+ model B camera).
//   -mode A : finite-lens physical camera — forward next-event splat through a finite
//             aperture + thin lens (true DoF; B is the aperture->0 pinhole limit)
//   -mode B : pinhole camera outside the box, light-tracing splat (default)
//   -mode C : finite-aperture forward catch (thin-lens depth of field)
//   -mode R : backward path-traced reference (independent validation)
//   -mode V : validate — run B and R and report the best-fit residual
//   -mode P : forward + camera-side composite — model B for diffuse-first pixels
//             (and caustics), a backward camera-side ray path for specular/coated
//             surfaces (which model B alone leaves black). See renderComposite.
//   -mode D : bidirectional path tracing (BDPT) — one unbiased estimator that traces
//             a light AND a camera subpath and MIS-combines every connection. Renders
//             specular-first pixels directly (no composite seam) AND diffuse caustics
//             in a single pass, on the absolute-radiance scale. Renders participating
//             media of every kind — global haze, bounded, and heterogeneous density
//             fields (volume in-scatter vertices + transmittance-weighted connections).
//             GPU-accelerated (its own BDPT megakernel; see renderBdptCuda). Does not
//             support fluorescence / spot & env lights (use B/P or R for those). See
//             renderBdpt / bdpt.h.
//   -mode M : photon map — trace a forward photon pass once, store diffuse deposits in
//             a view-independent uniform-hash-grid photon map, then final-gather the
//             camera image from it (backward camera ray through specular, radius density
//             estimate at the first diffuse hit). View-independent, so the map can be
//             built once and reused across every camera / flythrough frame. Gather
//             radius set by -pmradius (absolute) or -pmradiusfrac (fraction of scene
//             radius, default 0.02). CPU only. See photonmap.h / photonmap_render.h.
//   -mode S : stochastic progressive photon mapping (SPPM) — repeated bounded photon
//             passes with a per-pixel shrinking gather radius (Hachisuka 2008/2009), so
//             the estimate converges (unbiased in the limit) with flat memory and nails
//             caustics / SDS paths. -n = photons per pass, -spp = number of passes (or a
//             -time/-noise budget); radius-shrink rate -sppmalpha (default 0.7), initial
//             radius from -pmradius/-pmradiusfrac. CPU only. See sppm_render.h.
//   -mode U : vertex connection and merging (VCM/UPS, Georgiev 2012) — combines BDPT
//             vertex connections with SPPM photon merging under one multiple-importance
//             -sampling (balance-heuristic) weight, so it robustly handles both the
//             diffuse/glossy paths BDPT is good at and the caustic/SDS paths photon
//             mapping is good at. Each pass traces resX*resY light subpaths + one camera
//             subpath per pixel; -n is ignored (light-path count follows the film). -spp
//             = number of passes (or a -time/-noise budget); radius-shrink rate -vcmalpha
//             (default 0.75), initial radius from -pmradius/-pmradiusfrac. CPU only.
//             See vcm.h.
// Modes A/B/C/P trace identical forward physics; B/C/P differ only in how the
// camera measures (splat / aperture catch / composite with the camera-side path).
//
// -device auto|cpu|gpu selects the backend (default auto):
//   auto — use the GPU when a supported CUDA device is present and the render is a
//          forward trace the GPU handles (models A/B/C on a non-fluorescent scene);
//          otherwise use the CPU. Prints which it chose and why. Recommended.
//   gpu  — force the GPU; warns and falls back to the CPU if it can't be used.
//   cpu  — force the CPU (deterministic; used for reference/validation baselines).
// The GPU runs the forward light trace (models A/B/C, and the forward pass of mode
// V) as a CUDA megakernel, and mode D as its own BDPT megakernel; it falls back to
// the CPU for the backward tracer (mode R, the mode-P camera-side layer) and
// fluorescent scenes. The CUDA backend is optional
// at build time (see CMakeLists.txt / FTRACE_CUDA_ARCH); without a CUDA toolkit the
// renderer is CPU-only and -device gpu/auto use the CPU.
//
// -wavefront selects the streaming GPU backend instead of the default megakernel (only
// affects a forward GPU render; ignored otherwise). Both run identical physics and
// conserve energy exactly; the megakernel runs each photon's whole path in one thread,
// while the wavefront splits the trace into coherent extend/shade passes over a
// persistent photon pool and regenerates finished paths to keep SIMD lanes full. The
// wavefront helps on divergent / deep-path scenes and small GPUs; the megakernel is
// usually faster on shallow, uniform scenes on a big GPU (its default). The RNG stream
// differs, so images match the megakernel only to within Monte-Carlo noise.
//
// Progressive rendering & live progress. Every image-forming mode — the forward camera
// models A/B/C (photon-count-independent brightness), the backward reference R, and the
// bidirectional D (both accumulate a SUM over samples-per-pixel) — refines an image whose
// brightness is fixed and whose graininess only falls with more samples. So all of them
// report the same live progress (periodic crash-safe image write, a status line or
// -preview thumbnail, and a ~noise% estimate) and accept the same budget flags:
//   -n <photons>   forward A/B/C: trace exactly this many photons (default). For R/D the
//                  sample budget is -spp; -n is only the forward batch granularity.
//   -time <sec>    render until the wall-clock budget elapses, then stop and save. Works
//                  for A/B/C (photon batches) and R/D (spp chunks) alike.
//   -noise <pct>   render until the estimated graininess falls to <= pct percent (the same
//                  "~X% noise" figure the progress line reports: for A/B/C 100/sqrt(mean
//                  per-lit-pixel photon count); for R/D 100/sqrt(spp done)), then stop.
//                  Combine with -time to also cap the wall-clock ("stop at whichever
//                  comes first"); alone it traces until converged (Ctrl-C stops early).
//   -forever       render indefinitely, refining, until interrupted (Ctrl-C): the first
//                  Ctrl-C finishes the current batch/chunk, writes a final image, and
//                  exits cleanly (a second Ctrl-C force-quits). For A/B/C it implies the
//                  checkpoint, so a later -resume picks up where you stopped.
//   -resume        (modes A/B/C, R/D, P) reload the accumulated film from the "<out>.ftbuf"
//                  checkpoint and keep adding samples (with -n/-spp/-time/-forever).
//   -checkpoint    (modes A/B/C, R/D, P) on a plain -n/-spp render, also write the checkpoint
//                  so a later -resume can continue it (-time/-forever/-resume imply it). Each
//                  batch/resume draws an independent RNG stream (seed offset = cumulative
//                  photons for A/B/C, cumulative spp for R/D/P), so the result matches a single
//                  render of the combined count; a fresh render (offset 0) is bit-identical to
//                  the historical path. R/D store a SUM-over-spp film + spp count; P stores a
//                  dual forward+backward film (magic FTPCM02). M/S/U keep persistent per-pass
//                  state that a film alone can't restore, so they are not disk-resumable.
//   -savemap <f>   (mode M, GPU) after the forward deposit pass, write the view-independent
//                  photon map to <f> (magic FTPMP01). The map is the expensive result of the
//                  photon trace and is independent of camera and gather radius.
//   -loadmap <f>   (mode M, GPU) load a photon map saved with -savemap and SKIP the deposit
//                  entirely — re-gather new camera angles / a different -pmradius for free,
//                  without re-tracing a single photon. A scene-identity guard rejects a map
//                  built for a different scene (falls back to a fresh deposit).
//   -preview       during a progress render, redraw a live ANSI colour thumbnail of the
//                  current image in the terminal at each periodic update (in place).
//   -window        open a real OS window (Win32 GDI on Windows; no-op elsewhere) that shows
//                  the actual tone-mapped pixels, refreshed at each -interval tick. Unlike
//                  -preview's coarse terminal thumbnail this is the full-resolution image.
//                  A plain fixed -n forward render with -window is auto-chunked so the view
//                  updates as it converges; closing the window stops the render (final image
//                  is still written). Runs on its own UI thread, so it stays responsive.
//   -interval <s>  seconds between periodic image writes / preview refreshes (default 15).
//                  The output image file is rewritten at this cadence, so an auto-reloading
//                  image viewer is also a live display. Applies to every mode above (a plain
//                  fixed -spp R/D render also rewrites the image and prints "[spp] x/total"
//                  progress as its chunks land).

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <csignal>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>
#include <thread>
#include <map>
#include <set>
#include <memory>
#include <atomic>              // -stop: cross-thread flags for the external stop channel
#include <filesystem>          // -review: scan a directory of rendered frames
#include "scene.h"
#include "isomesh.h"            // -export-mesh: isosurface -> watertight OBJ (marching tetrahedra)
#include "watertight.h"         // -check-watertight: report non-airtight meshes/isosurfaces
#include "airtight.h"           // -check-airtight: ray-parity audit of the marched isosurface field
#include "priority_audit.h"     // ahead-of-time nested-dielectric priority ambiguity warning
#include "camera.h"
#include "raster.h"             // -raster: fast solid-shaded preview rasterizer (no light transport)
#include "render.h"
#include "rainbow.h"            // Airy-theory droplet phase function (rainbows in droplet media)
#include "backward.h"
#include "bdpt.h"
#include "photonmap_render.h"   // mode M: photon-mapped final gather (ROADMAP item 1)
#include "sppm_render.h"        // mode S: stochastic progressive photon mapping (item 2)
#include "vcm.h"                // mode U: vertex connection and merging (VCM/UPS, item 3)
#include "lights.h"
#include "mesh.h"
#include "ftsl.h"
#include "curvedrive.h"         // -anim: loom CurveDrive JSON sidecar (E2 channel a) read/write
#include "animlive.h"           // -anim -loom: the live editor<->loom value channel (E2 channel b)
#include "livewindow.h"         // -window: real OS live-preview window (Win32 GDI)
#include "viewer_gui.h"         // -viewer: loom native viewer host (Dear ImGui + Win32/D3D11)
#include "render_progress.h"   // SppProgress — used unconditionally below; the CUDA
                               // header also pulls it in, but CPU-only builds need it too
#ifdef HAVE_CUDA
#include "render_cuda.h"
#include "raster_cuda.h"   // GPU preview rasterizer (device twin of raster::renderFrame)
#endif
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX               // keep std::min/std::max (windows.h else macro-clobbers them)
#include <windows.h>          // -preview: enable ANSI VT processing in a plain console
#endif

// stb_image_write encoders (implementation compiled once in stb_image_impl.cpp).
// Only the two we use for 8-bit RGB output; see writeImage().
extern "C" {
    int stbi_write_png(const char* filename, int w, int h, int comp, const void* data, int stride_bytes);
    int stbi_write_jpg(const char* filename, int w, int h, int comp, const void* data, int quality);
}

// stb_image decoder (implementation compiled once in stb_image_impl.cpp). Used by
// -review to load already-rendered PNG/JPG/BMP/TGA frames; PPM P6 is read by a small
// custom loader (stb_image doesn't decode PPM).
extern "C" {
    unsigned char* stbi_load(const char* filename, int* x, int* y, int* channels_in_file, int desired_channels);
    void           stbi_image_free(void* retval_from_stbi_load);
}

// Case-insensitive test for a filename ending in `ext` (e.g. ".png").
static bool endsWithCI(const std::string& s, const char* ext) {
    size_t n = std::strlen(ext);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i)
        if (std::tolower((unsigned char)s[s.size() - n + i]) != std::tolower((unsigned char)ext[i]))
            return false;
    return true;
}

// Write an 8-bit RGB buffer (row-major, top row first) to `path`, choosing the
// encoder from the file extension: .png -> PNG, .jpg/.jpeg -> JPEG (q=95), and
// anything else (incl. .ppm / no extension) -> binary PPM (P6). This honours the
// requested format instead of always emitting PPM bytes — a mislabeled .png
// (PPM bytes in a .png file) breaks any consumer that trusts the extension.
// Returns true on success. `label` is the human name printed by the caller.
static bool writeImage(const std::string& path, int W, int H, const std::vector<uint8_t>& img) {
    if (endsWithCI(path, ".png"))
        return stbi_write_png(path.c_str(), W, H, 3, img.data(), W * 3) != 0;
    if (endsWithCI(path, ".jpg") || endsWithCI(path, ".jpeg"))
        return stbi_write_jpg(path.c_str(), W, H, 3, img.data(), 95) != 0;
    std::ofstream fo(path, std::ios::binary);
    if (!fo) return false;
    fo << "P6\n" << W << ' ' << H << "\n255\n";
    fo.write((const char*)img.data(), (std::streamsize)img.size());
    return (bool)fo;
}

// Resolve a -light name to an emission SPD. Delegates to the shared resolver in
// lights.h (the same one the FTSL `preset:<name>` expression uses). An explicitly
// named light that resolves to nothing is a fatal error (the user asked for a
// specific source), NOT a silent fall-through to white — the built-in default
// ("bb6500") always resolves via the parametric bb<K> path, so this only fires on
// a genuinely unrecognized name (typo / missing data file).
static Spectrum resolveLight(const char* name) {
    if (!name) return blackbody(6500.0);
    Spectrum s;
    if (resolveLightPreset(name, s)) return s;
    throw std::runtime_error("unknown -light preset '" + std::string(name) +
        "' — not a bb<K>/led<K>k parametric, a data/light or data/illuminant file/alias, "
        "or a built-in lamp model (sodium/mercury/metal-halide/fluorescent/led-white)");
}

static void addQuad(Scene& s, Vec3 a, Vec3 b, Vec3 c, Vec3 d, int mat, int sensorId = -1) {
    s.tris.push_back(Tri{a, b, c, mat, sensorId, {}});
    s.tris.push_back(Tri{a, c, d, mat, sensorId, {}});
}
static void addTri(Scene& s, Vec3 a, Vec3 b, Vec3 c, int mat) {
    s.tris.push_back(Tri{a, b, c, mat, -1, {}});
}

// White box + dispersive glass prism + collimated white beam -> rainbow on the floor.
static Scene buildPrism(int res) {
    (void)res; // geometry is resolution-independent; camera res set by caller
    Scene s;
    Material white; white.reflect = whiteWall(0.75);            s.mats.push_back(white); // 0
    Material glass; glass.type = MatType::Dielectric;
    glass.roughness = 0.0;
    Spectrum sf10; if (!resolveGlassIor("SF10", sf10)) sf10 = iorConstant(1.7283);
    glass.ior = sf10;                                            s.mats.push_back(glass); // 1

    addQuad(s, {0,0,0},{1,0,0},{1,0,1},{0,0,1}, 0);   // floor
    addQuad(s, {0,1,0},{0,1,1},{1,1,1},{1,1,0}, 0);   // ceiling
    addQuad(s, {0,0,0},{0,1,0},{1,1,0},{1,0,0}, 0);   // back
    addQuad(s, {0,0,0},{0,0,1},{0,1,1},{0,1,0}, 0);   // left
    addQuad(s, {1,0,0},{1,1,0},{1,1,1},{1,0,1}, 0);   // right

    // Equilateral-ish triangular prism, apex up, axis along z.
    Vec3 T0{0.5,0.75,0.35}, L0{0.30,0.35,0.35}, R0{0.70,0.35,0.35};
    Vec3 T1{0.5,0.75,0.65}, L1{0.30,0.35,0.65}, R1{0.70,0.35,0.65};
    addTri(s, T0,L0,R0, 1); addTri(s, T1,R1,L1, 1);   // caps
    addQuad(s, L0,T0,T1,L1, 1);                        // left face
    addQuad(s, T0,R0,R1,T1, 1);                        // right face
    addQuad(s, R0,L0,L1,R1, 1);                        // bottom face

    // Collimated white beam entering the left face, travelling +x. Equal-energy
    // SPD -> even rainbow. Thin 3cm pencil cross-section.
    s.addAreaLight(/*origin*/{0.05, 0.54, 0.49}, /*u*/{0, 0.03, 0}, /*v*/{0, 0, 0.03},
                   /*normal*/{1, 0, 0}, /*area*/0.03 * 0.03, constantSpectrum(1.0), 1.0,
                   /*collimated*/true, /*beamDir*/{1, 0, 0});
    s.finalizeTris();
    return s;
}

// White box + reflective diffraction grating + collimated white beam -> a
// diffracted spectrum fanned across the side walls. A collimated beam (like the
// prism scene) is essential: under an area light every groove point is lit from
// all directions and the orders overlap into white, so the dispersion only reads
// cleanly for a single well-defined incidence. The beam strikes a grating patch on
// the back wall; the 0th order retro-reflects (white) while the +/-1 orders fan in
// +/-x by wavelength, painting symmetric rainbows on the left and right walls.
static Scene buildGrating(int res, bool diffraction) {
    (void)res;
    Scene s;
    Material white; white.reflect = whiteWall(0.75);            s.mats.push_back(white); // 0
    Material grating; grating.type = MatType::Grating;
    grating.reflect = constantSpectrum(0.95);   // overall reflectivity
    grating.grooveSpacing = 1000.0;             // 1 um period -> strong visible spread
    grating.grooveDir = {0, 1, 0};              // vertical grooves -> horizontal dispersion (x)
    grating.gratingMaxOrder = 3;
    (void)diffraction;                          // toggled on the Renderer, not the material
    s.mats.push_back(grating);                                                           // 1

    addQuad(s, {0,0,0},{1,0,0},{1,0,1},{0,0,1}, 0);   // floor
    addQuad(s, {0,1,0},{0,1,1},{1,1,1},{1,1,0}, 0);   // ceiling
    addQuad(s, {0,0,0},{0,1,0},{1,1,0},{1,0,0}, 0);   // back (plain white border)
    addQuad(s, {0,0,0},{0,0,1},{0,1,1},{0,1,0}, 0);   // left
    addQuad(s, {1,0,0},{1,1,0},{1,1,1},{1,0,1}, 0);   // right
    // Grating patch on the back wall (z ~ 0), facing into the room (+z). Sits just
    // in front of the wall so it is the first surface the beam meets.
    addQuad(s, {0.3,0.3,0.002},{0.7,0.3,0.002},{0.7,0.7,0.002},{0.3,0.7,0.002}, 1);

    // Collimated white beam entering from the front, travelling -z into the
    // grating. Equal-energy SPD -> even rainbow. Thin 3cm pencil cross-section.
    s.addAreaLight(/*origin*/{0.485, 0.485, 0.95}, /*u*/{0.03, 0, 0}, /*v*/{0, 0.03, 0},
                   /*normal*/{0, 0, -1}, /*area*/0.03 * 0.03, constantSpectrum(1.0), 1.0,
                   /*collimated*/true, /*beamDir*/{0, 0, -1});
    s.finalizeTris();
    return s;
}

// The front of the box is left open so the external camera (any image-forming
// mode, including the model-A finite-lens camera) can see in. `mode` is retained
// for API compatibility but no longer changes the geometry.
static Scene buildCornell(int res, char mode, const Spectrum& lightSpd,
                          const char* meshPath = nullptr, double meshScale = 1.0,
                          bool diffuseSphere = false, bool fluoroSphere = false,
                          bool thinFilmSphere = false,
                          double filmThickness = 300.0, double filmIor = 1.30) {
    (void)res;   // camera resolution is set by the caller; scene geometry is res-free
    Scene s;
    Material white; white.reflect = whiteWall(0.75);            s.mats.push_back(white); // 0
    Material red;   red.reflect   = redWall();                   s.mats.push_back(red);   // 1
    Material green; green.reflect = greenWall();                 s.mats.push_back(green); // 2
    Material light; light.reflect = constantSpectrum(0.0);
    light.emit = lightSpd; light.isLight = true;                 s.mats.push_back(light); // 3
    Material glass; glass.type = MatType::Dielectric;
    glass.roughness = 0.0;
    Spectrum sf10; if (!resolveGlassIor("SF10", sf10)) sf10 = iorConstant(1.7283);
    glass.ior = sf10;                                            s.mats.push_back(glass); // 4
    Material mesh;  mesh.reflect  = whiteWall(0.8);              s.mats.push_back(mesh);  // 5 (diffuse)
    s.mats.push_back(makeFluoroMaterial());                                              // 6 (fluorescent)
    Material film;  film.type = MatType::ThinFilm;
    film.ior = iorConstant(1.5);                // dispersion-free glass-like substrate
    film.filmIor = filmIor; film.filmThickness = filmThickness;
    s.mats.push_back(film);                                                              // 7 (thin film)

    addQuad(s, {0,0,0},{1,0,0},{1,0,1},{0,0,1}, 0);            // floor
    addQuad(s, {0,1,0},{0,1,1},{1,1,1},{1,1,0}, 0);            // ceiling
    addQuad(s, {0,0,0},{0,1,0},{1,1,0},{1,0,0}, 0);            // back
    addQuad(s, {0,0,0},{0,0,1},{0,1,1},{0,1,0}, 1);            // left (red)
    addQuad(s, {1,0,0},{1,1,0},{1,1,1},{1,0,1}, 2);            // right (green)
    (void)mode;   // front left open for the external camera in every mode

    const double lx0 = 0.35, lx1 = 0.65, lz0 = 0.35, lz1 = 0.65, ly = 0.999;
    addQuad(s, {lx0,ly,lz0},{lx1,ly,lz0},{lx1,ly,lz1},{lx0,ly,lz1}, 3);

    // A loaded mesh (diffuse) replaces the glass sphere when -mesh is given;
    // otherwise the dispersive glass sphere casts a spectral caustic on the floor.
    if (meshPath && meshPath[0]) {
        loadObj(s, meshPath, /*mat*/5, /*translate*/{0.5, 0.4, 0.5}, meshScale);
    } else {
        // Sphere material: thin-film (7) for the iridescent demo; fluorescent (6)
        // for the fluoro demo; diffuse (5) for the reference/validation modes so
        // there is no specular black-glass mismatch; otherwise the dispersive
        // glass sphere (4) casts a spectral caustic.
        int sphMat = thinFilmSphere ? 7 : (fluoroSphere ? 6 : (diffuseSphere ? 5 : 4));
        s.spheres.push_back(Sphere{{0.5, 0.32, 0.4}, 0.25, sphMat});
    }

    s.addAreaLight(/*origin*/{lx0, ly, lz0}, /*u*/{lx1 - lx0, 0, 0}, /*v*/{0, 0, lz1 - lz0},
                   /*normal*/{0, -1, 0}, /*area*/(lx1 - lx0) * (lz1 - lz0), s.mats[3].emit, 1.0,
                   /*collimated*/false, /*beamDir*/{1, 0, 0}, /*matId*/3);
    s.build();

    return s;
}

// Cornell box (model-B only) with the reflective material types side by side:
// a near-perfect mirror, a rough glossy metal, and a half-mirror (beamsplitter).
// All three are specular, so under pure light tracing (model B) they appear BLACK
// from the camera: a specular vertex has zero probability of connecting to the
// pinhole (the SDS limitation, same as the glass sphere in the Cornell scene).
// The physics is still exercised — photons reflect off them and illuminate the
// diffuse walls, and energy conserves — but seeing the spheres' mirrored image
// directly requires a camera-side ray path (mode R/P/D).
static Scene buildMaterials(int res, const Spectrum& lightSpd) {
    (void)res;
    Scene s;
    Material white; white.reflect = whiteWall(0.75);            s.mats.push_back(white); // 0
    Material red;   red.reflect   = redWall();                   s.mats.push_back(red);   // 1
    Material green; green.reflect = greenWall();                 s.mats.push_back(green); // 2
    Material light; light.reflect = constantSpectrum(0.0);
    light.emit = lightSpd; light.isLight = true;                 s.mats.push_back(light); // 3
    Material mirror; mirror.type = MatType::Mirror;
    mirror.reflect = constantSpectrum(0.95);                     s.mats.push_back(mirror);// 4
    Material glossy; glossy.type = MatType::Glossy;
    glossy.reflect = constantSpectrum(0.9); glossy.roughness = 0.25;
                                                                 s.mats.push_back(glossy);// 5
    Material half; half.type = MatType::HalfMirror;
    half.reflect = constantSpectrum(0.5);                        s.mats.push_back(half);  // 6

    addQuad(s, {0,0,0},{1,0,0},{1,0,1},{0,0,1}, 0);            // floor
    addQuad(s, {0,1,0},{0,1,1},{1,1,1},{1,1,0}, 0);            // ceiling
    addQuad(s, {0,0,0},{0,1,0},{1,1,0},{1,0,0}, 0);            // back
    addQuad(s, {0,0,0},{0,0,1},{0,1,1},{0,1,0}, 1);            // left (red)
    addQuad(s, {1,0,0},{1,1,0},{1,1,1},{1,0,1}, 2);            // right (green)

    const double lx0 = 0.35, lx1 = 0.65, lz0 = 0.35, lz1 = 0.65, ly = 0.999;
    addQuad(s, {lx0,ly,lz0},{lx1,ly,lz0},{lx1,ly,lz1},{lx0,ly,lz1}, 3);

    s.spheres.push_back(Sphere{{0.26, 0.20, 0.35}, 0.18, 4}); // mirror
    s.spheres.push_back(Sphere{{0.74, 0.20, 0.35}, 0.18, 5}); // glossy
    s.spheres.push_back(Sphere{{0.50, 0.22, 0.68}, 0.20, 6}); // half-mirror

    s.addAreaLight(/*origin*/{lx0, ly, lz0}, /*u*/{lx1 - lx0, 0, 0}, /*v*/{0, 0, lz1 - lz0},
                   /*normal*/{0, -1, 0}, /*area*/(lx1 - lx0) * (lz1 - lz0), s.mats[3].emit, 1.0,
                   /*collimated*/false, /*beamDir*/{1, 0, 0}, /*matId*/3);
    s.finalizeTris();
    return s;
}

static void selfTestColor() {
    Vec3 xyz{};
    for (double w = LAMBDA_MIN; w <= LAMBDA_MAX; w += 1.0)
        xyz += Vec3(cieX(w), cieY(w), cieZ(w));
    xyz = xyz / cieYIntegral();
    Vec3 lin = xyzToLinearSrgb(xyz);
    std::printf("[selftest] equal-energy XYZ=(%.3f,%.3f,%.3f) Y=%.3f  linsRGB=(%.3f,%.3f,%.3f)\n",
                xyz.x, xyz.y, xyz.z, xyz.y, lin.x, lin.y, lin.z);
}

// Fire random rays through the scene and assert the BVH agrees with the linear
// scan (same hit distance, material, sensor). Guards against BVH build/traversal
// bugs that would silently corrupt the image.
static int checkBvh(const Scene& scene, long long rays) {
    Pcg32 rng; rng.seed(1234567u, 0xABCDEFu);
    int mismatches = 0;
    for (long long i = 0; i < rays; ++i) {
        // Random ray: origin in a box around the scene, random direction.
        Vec3 o{rng.uniform() * 3 - 1, rng.uniform() * 3 - 1, rng.uniform() * 3 - 1};
        double z = rng.uniform() * 2 - 1, phi = 2 * PI * rng.uniform();
        double rr = std::sqrt(std::max(0.0, 1 - z * z));
        Vec3 d = normalize(Vec3{rr * std::cos(phi), rr * std::sin(phi), z});
        Ray r{o, d};
        Hit a = scene.closestHit(r);
        Hit b = scene.closestHitLinear(r);
        bool ok = (a.valid == b.valid) &&
                  (!a.valid || (std::fabs(a.t - b.t) < 1e-7 &&
                                a.matId == b.matId && a.sensorId == b.sensorId));
        if (!ok) ++mismatches;
    }
    std::printf("[checkbvh] %lld rays, %d mismatches -> %s\n",
                rays, mismatches, mismatches == 0 ? "PASS" : "FAIL");
    return mismatches;
}

// Implicit-surface (SDF sphere trace) self-test. A unit-Lipschitz SDF sphere must
// reproduce the analytic sphere intersection: for many random rays we compare the
// sphere-traced hit (distance + geometric normal) against intersectSphere on the
// same geometry. Rays grazing the silhouette (impact parameter within a couple of
// surface epsilons of the radius) are ambiguous hit/miss and are excluded — the
// surface itself, not the razor-thin tangent, is what must match.
static int checkImplicit(long long rays) {
    const Vec3 c{0.3, -0.1, 0.2};
    const double r = 0.7;
    Implicit im = makeSphereImplicit(c, r, 0);
    Sphere sp{c, r, 0};
    Pcg32 rng; rng.seed(0xC0FFEEu, 0x1234u);
    int mismatches = 0; long long compared = 0, grazed = 0;
    double maxdt = 0, maxdn = 0;
    for (long long i = 0; i < rays; ++i) {
        Vec3 o{rng.uniform() * 4 - 2, rng.uniform() * 4 - 2, rng.uniform() * 4 - 2};
        double z = rng.uniform() * 2 - 1, phi = 2 * PI * rng.uniform();
        double rr = std::sqrt(std::max(0.0, 1 - z * z));
        Vec3 d = normalize(Vec3{rr * std::cos(phi), rr * std::sin(phi), z});
        Ray ray{o, d};
        // Impact parameter: perpendicular distance from the sphere center to the ray.
        Vec3 oc = c - o;
        double proj = dot(oc, d);
        double b2 = std::max(0.0, dot(oc, oc) - proj * proj);
        double impact = std::sqrt(b2);
        // Skip rays that are ambiguous at the surface epsilon: tangent grazes
        // (impact ~= r) and origins that start ON the surface (|o-c| ~= r) — both
        // are sub-epsilon hit/miss coin-flips, not a test of the surface itself.
        bool grazing = std::fabs(impact - r) < 1e-3 || std::fabs(length(oc) - r) < 1e-3;
        Hit ha; ha.t = DBL_MAX; bool hitA = intersectSphere(ray, sp, 1e-6, ha);
        Hit hi; hi.t = DBL_MAX; bool hitI = intersectImplicit(ray, im, 1e-6, hi);
        if (grazing) { ++grazed; continue; }
        if (hitA != hitI) { ++mismatches; continue; }
        if (!hitA) continue;
        ++compared;
        double dt = std::fabs(ha.t - hi.t);
        double dn = length(ha.ng - hi.ng);
        maxdt = std::max(maxdt, dt); maxdn = std::max(maxdn, dn);
        if (dt > 1e-3 || dn > 2e-2) ++mismatches;
    }
    std::printf("[checkimplicit] %lld rays (%lld surface, %lld grazing skipped), "
                "%d mismatches, max|dt|=%.2e max|dn|=%.2e -> %s\n",
                rays, compared, grazed, mismatches, maxdt, maxdn,
                mismatches == 0 ? "PASS" : "FAIL");
    return mismatches;
}

// Deterministic thin-lens (mode C) self-test. Forward catch is far too photon-
// inefficient to validate the lens by rendering, so instead we fire rays from a
// fixed scene point through many aperture positions and measure the circle of
// confusion (pixel spread) on the film. A correct thin lens collapses the CoC to
// ~0 for a point at the focus distance and spreads it for off-focus points; the
// in-focus image must also land where the pinhole project() puts it.
static int checkLens() {
    const int res = 256;
    Camera cam;
    cam.lookAt({0.5, 0.5, 2.7}, {0.5, 0.5, 0.5}, {0, 1, 0}, 40.0, res, res);
    cam.apertureR = 0.15;
    const double focusDist = 2.2;      // focus plane == box centre (z = 0.5)
    cam.setFocus(focusDist);
    Pcg32 rng; rng.seed(99u, 7u);

    // Fire rays from X through many aperture disc samples; return the film pixel
    // bounding-box spread (max CoC in x/y), catch count, and mean pixel.
    auto measure = [&](const Vec3& X, int& spread, double& mx, double& my, int& caught) {
        int minx = 1 << 30, maxx = -(1 << 30), miny = 1 << 30, maxy = -(1 << 30);
        long long sx = 0, sy = 0; caught = 0;
        for (int k = 0; k < 40000; ++k) {
            double rr = cam.apertureR * std::sqrt(rng.uniform());
            double a = 2 * PI * rng.uniform();
            Vec3 P = cam.eye + cam.u * (rr * std::cos(a)) + cam.v * (rr * std::sin(a));
            Vec3 d = normalize(P - X);
            int px, py;
            if (!cam.catchPhoton(Ray{X, d}, 1e30, px, py)) continue;
            ++caught;
            minx = std::min(minx, px); maxx = std::max(maxx, px);
            miny = std::min(miny, py); maxy = std::max(maxy, py);
            sx += px; sy += py;
        }
        spread = std::max(maxx - minx, maxy - miny);
        mx = caught ? (double)sx / caught : -1; my = caught ? (double)sy / caught : -1;
    };

    // On-axis point exactly on the focus plane -> sharp (CoC ~ 0), at film centre.
    Vec3 Xfocus = cam.eye + cam.w * focusDist;
    int sF, cF; double mxF, myF; measure(Xfocus, sF, mxF, myF, cF);
    // A point well in front of the focus plane -> defocused (large CoC).
    Vec3 Xnear = cam.eye + cam.w * (focusDist * 0.5);
    int sN, cN; double mxN, myN; measure(Xnear, sN, mxN, myN, cN);
    // Off-axis focus-plane point must image where the pinhole projection predicts.
    Vec3 Xoff = cam.eye + cam.w * focusDist + cam.u * 0.25 + cam.v * (-0.15);
    int sO, cO; double mxO, myO; measure(Xoff, sO, mxO, myO, cO);
    int ppx, ppy; double cc, d2;
    bool proj = cam.project(Xoff, ppx, ppy, cc, d2);
    double projErr = proj ? std::max(std::fabs(mxO - ppx), std::fabs(myO - ppy)) : 1e9;

    std::printf("[checklens] focus CoC=%dpx (caught %d), defocus CoC=%dpx (caught %d)\n",
                sF, cF, sN, cN);
    std::printf("[checklens] off-axis focus mean=(%.1f,%.1f) vs pinhole project=(%d,%d) err=%.1fpx\n",
                mxO, myO, ppx, ppy, projErr);
    bool pass = (cF > 0 && cN > 0 && cO > 0) && (sF <= 1) && (sN >= 8) && (projErr <= 1.5);
    std::printf("[checklens] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Fire random rays and report average BVH work per ray (nodes visited, leaf
// primitive tests). Confirms tree quality independent of image correctness.
static void bvhStats(const Scene& scene, long long rays) {
    Pcg32 rng; rng.seed(2468013u, 0x13579u);
    long long totNodes = 0, totLeaf = 0, hits = 0;
    for (long long i = 0; i < rays; ++i) {
        Vec3 o{rng.uniform() * 3 - 1, rng.uniform() * 3 - 1, rng.uniform() * 3 - 1};
        double z = rng.uniform() * 2 - 1, phi = 2 * PI * rng.uniform();
        double rr = std::sqrt(std::max(0.0, 1 - z * z));
        Vec3 d = normalize(Vec3{rr * std::cos(phi), rr * std::sin(phi), z});
        TraversalStats st;
        Hit h = scene.closestHit(Ray{o, d}, 1e-6, &st);
        totNodes += st.nodeVisits; totLeaf += st.leafTests; if (h.valid) ++hits;
    }
    long long prims = (long long)scene.tris.size() + (long long)scene.spheres.size();
    // Leaf-size histogram to gauge tree balance.
    long long leaves = 0, maxLeaf = 0, primsInLeaves = 0;
    for (const auto& nd : scene.bvh.nodes)
        if (nd.isLeaf()) { ++leaves; primsInLeaves += nd.count; maxLeaf = std::max<long long>(maxLeaf, nd.count); }
    std::printf("[bvhstats] %lld prims, %lld nodes, %lld leaves (max %lld, avg %.1f prims/leaf)\n",
                prims, (long long)scene.bvh.nodes.size(), leaves, maxLeaf,
                leaves ? (double)primsInLeaves / leaves : 0.0);
    std::printf("[bvhstats] per ray: %.1f nodes, %.1f leaf-tests, %.1f%% hit\n",
                (double)totNodes / rays, (double)totLeaf / rays, 100.0 * hits / rays);
}

// Deterministic fluorescence self-test. Forward fluorescence has no analytic image
// to compare against, so instead we validate the reradiation primitives directly:
//   (a) the emission sampler is unbiased  -> its mean lambda matches the SPD's;
//   (b) the branch probabilities are correct -> the re-emission fraction at a
//       strongly-excited input wavelength equals epsilon*Q (= aEff*Q);
//   (c) the Stokes shift is physical -> re-emitted lambda' is longer than the
//       excitation lambda and centred on the emission band.
// It exercises the same fluoroInteract()/fluoEmitSampler used by the renderer, so a
// bug in the transport math surfaces here without needing a full render.
static int checkFluoro() {
    Material m = makeFluoroMaterial();

    // (a) Analytic emission mean vs Monte-Carlo sampler mean.
    double num = 0, den = 0;
    for (double w = LAMBDA_MIN; w <= LAMBDA_MAX; w += 0.5) {
        double v = std::max(0.0, m.fluoEmit(w)); num += v * w; den += v;
    }
    double meanAnalytic = num / den;
    Pcg32 rng; rng.seed(0xF10E5Cu, 0x1234u);
    const long long Ns = 4'000'000;
    double sMean = 0; for (long long i = 0; i < Ns; ++i) { double pf; sMean += m.fluoEmitSampler.sample(rng, pf); }
    sMean /= Ns;

    // (b,c) Branch statistics at a strongly-excited input wavelength.
    const double lin = 450.0;                 // blue excitation
    double rho, aEff; fluoroWeights(m, lin, rho, aEff);
    double expectFrac = aEff * m.fluoYield;
    long long reemit = 0, elastic = 0, absorb = 0; double meanOut = 0;
    const long long Nt = 4'000'000;
    for (long long i = 0; i < Nt; ++i) {
        FluoroResult r = fluoroInteract(m, lin, rng);
        if      (r.event == FluoroEvent::Reemit)  { ++reemit; meanOut += r.lambdaOut; }
        else if (r.event == FluoroEvent::Elastic) ++elastic;
        else                                       ++absorb;
    }
    double frac = (double)reemit / Nt;
    double moOut = reemit ? meanOut / reemit : 0.0;

    // (d) The EXCITATION sampler (Material::fluoInSampler, built from
    //     absorb(lambda)*illuminant(lambda)). The backward tracer's reradiation NEE
    //     weight is E[aEff(lambda_in)*Q * spd(lambda_in) / pdf(lambda_in)], which must
    //     equal Q*integral(aEff*spd) NO MATTER which pdf is used -- so estimating it
    //     both ways (illuminant-only, the pre-0.115.0 sampler, vs absorb*illuminant)
    //     is a direct unbiasedness test. The variance ratio is reported too, though
    //     note it is a BEST case: this integrand is exactly the new sampler's target,
    //     so its estimator is constant and only CDF discretisation is left. In a real
    //     render the NEE geometry/visibility factor rides along and keeps variance.
    //     Also checks the mode-W 1-spp draw (u = rot05(0) = 0.5, i.e. the CDF median)
    //     lands inside the absorption band -- what makes a narrow dye right at 1 spp.
    Spectrum illum = blackbody(6500.0);
    Spectrum prodS = [&m, illum](double w) { return clamp01(m.fluoAbsorb(w)) * illum(w); };
    EmissionSampler illumS, inS;
    illumS.build(illum, 1.0);
    inS.build(prodS, 1.0);
    double refI = 0.0;   // analytic integral(aEff(lambda)*illum(lambda)) by fine quadrature
    for (double w = LAMBDA_MIN + 0.25; w < LAMBDA_MAX; w += 0.5) {
        double r2, a2; fluoroWeights(m, w, r2, a2);
        refI += a2 * illum(w) * 0.5;
    }
    const long long Nx = 400'000;
    double sumOld = 0, sumOld2 = 0, sumNew = 0, sumNew2 = 0;
    for (long long i = 0; i < Nx; ++i) {
        double p, r2, a2;
        double w = illumS.sample(rng, p);
        double f = 0.0;
        if (p > 0.0) { fluoroWeights(m, w, r2, a2); f = a2 * illum(w) / p; }
        sumOld += f; sumOld2 += f * f;
        w = inS.sample(rng, p);
        f = 0.0;
        if (p > 0.0) { fluoroWeights(m, w, r2, a2); f = a2 * illum(w) / p; }
        sumNew += f; sumNew2 += f * f;
    }
    double eOld = sumOld / Nx, eNew = sumNew / Nx;
    double vOld = sumOld2 / Nx - eOld * eOld, vNew = sumNew2 / Nx - eNew * eNew;
    double pMed; double lamMed = inS.sampleAt(0.5, pMed);   // mode W's 1-spp lambda_in
    double rMed, aMed; fluoroWeights(m, lamMed, rMed, aMed);

    bool passA = std::fabs(sMean - meanAnalytic) < 1.0;
    bool passB = std::fabs(frac - expectFrac) < 0.005;
    bool passC = (moOut > lin) && std::fabs(moOut - meanAnalytic) < 1.5;
    // Both estimators unbiased to 2%, the new one strictly lower-variance, and the
    // mode-W median draw actually excites the dye (aEff > half its peak-ish 0.1).
    bool passD = refI > 0.0 && std::fabs(eOld - refI) < 0.02 * refI &&
                 std::fabs(eNew - refI) < 0.02 * refI && vNew < vOld && aMed > 0.1;
    bool pass = passA && passB && passC && passD;

    std::printf("[checkfluoro] emission mean: sampler=%.2f analytic=%.2f nm  (%s)\n",
                sMean, meanAnalytic, passA ? "ok" : "BAD");
    std::printf("[checkfluoro] reemit fraction @%.0fnm: measured=%.4f expected(eps*Q)=%.4f  (%s)\n",
                lin, frac, expectFrac, passB ? "ok" : "BAD");
    std::printf("[checkfluoro] Stokes shift: in=%.0f -> out_mean=%.2f nm (elastic=%.3f absorb=%.3f)  (%s)\n",
                lin, moOut, (double)elastic / Nt, (double)absorb / Nt, passC ? "ok" : "BAD");
    std::printf("[checkfluoro] excitation NEE weight: analytic=%.5g  illuminant-sampled=%.5g"
                "  absorb*illuminant-sampled=%.5g  var %.3g -> %.3g (%.1fx)  (%s)\n",
                refI, eOld, eNew, vOld, vNew, (vNew > 0.0 ? vOld / vNew : 0.0),
                passD ? "ok" : "BAD");
    std::printf("[checkfluoro] mode-W 1-spp lambda_in (CDF median) = %.1f nm, aEff there = %.3f\n",
                lamMed, aMed);
    std::printf("[checkfluoro] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Deterministic participating-media self-test. Validates the three primitives the
// fog transport relies on, each against a closed-form answer:
//   (a) free-flight sampling -> transmittance matches Beer-Lambert exp(-sigma_t*d);
//   (b) Henyey-Greenstein sampling -> mean scattered cosine equals g;
//   (c) phase-function normalization -> integral over the sphere equals 1.
static int checkFog() {
    Pcg32 rng; rng.seed(0xF0602Au, 0x5151u);

    // (a) Transmittance from exponential free-flight.
    const double st = 2.0, d = 0.75;
    const long long Nt = 8'000'000; long long through = 0;
    for (long long i = 0; i < Nt; ++i) {
        double tMed = -std::log(1.0 - rng.uniform()) / st;
        if (tMed >= d) ++through;
    }
    double Tmeasured = (double)through / Nt, Tanalytic = std::exp(-st * d);
    bool passA = std::fabs(Tmeasured - Tanalytic) < 0.001;

    // (b) HG mean cosine equals g.
    const double g = 0.6; Vec3 wi{0, 0, 1};
    const long long Ns = 4'000'000; double mc = 0;
    for (long long i = 0; i < Ns; ++i) mc += dot(wi, sampleHG(wi, g, rng));
    mc /= Ns;
    bool passB = std::fabs(mc - g) < 0.002;

    // (c) Phase function integrates to 1 over the sphere: int p 2pi dcos = 1.
    double integ = 0; const int NB = 200000; double dc = 2.0 / NB;
    for (int i = 0; i < NB; ++i) { double c = -1.0 + (i + 0.5) * dc; integ += hgPhase(c, g) * 2.0 * PI * dc; }
    bool passC = std::fabs(integ - 1.0) < 1e-3;

    bool pass = passA && passB && passC;
    std::printf("[checkfog] transmittance @sigma_t*d=%.2f: measured=%.4f analytic=%.4f  (%s)\n",
                st * d, Tmeasured, Tanalytic, passA ? "ok" : "BAD");
    std::printf("[checkfog] HG mean cosine @g=%.2f: measured=%.4f  (%s)\n", g, mc, passB ? "ok" : "BAD");
    std::printf("[checkfog] HG sphere integral: %.5f (want 1)  (%s)\n", integ, passC ? "ok" : "BAD");
    std::printf("[checkfog] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Deterministic thin-film / iridescence self-test. Validates the interference
// reflectance against closed-form expectations, each independent of the renderer:
//   (a) R stays physically bounded in [0,1] across a wide angle/wavelength sweep;
//   (b) at normal incidence R matches the hand-computed two-beam expression;
//   (c) R is periodic in the interference phase (equal at phi and phi+2pi), the
//       signature of true interference;
//   (d) R varies with wavelength (max-min gap) -> the surface is actually
//       iridescent, not a flat reflector.
static int checkThinFilm() {
    const double n0 = 1.0, n1 = 1.30, n2 = 1.50, d = 300.0;

    // (a) reflectance in range across the visible band and all incidence angles.
    bool inRange = true; double rmin = 1e9, rmax = -1e9;
    for (double lam = 380.0; lam <= 720.0; lam += 2.0)
        for (double ci = 0.05; ci <= 1.0; ci += 0.05) {
            double R = thinFilmReflectance(n0, n1, n2, 0.0, d, ci, lam);
            if (R < -1e-9 || R > 1.0 + 1e-9) inRange = false;
            rmin = std::min(rmin, R); rmax = std::max(rmax, R);
        }
    bool passA = inRange;

    // (b) normal incidence matches the closed-form Airy (multiple-beam)
    //     reflectance R = (r01^2+r12^2+2 r01 r12 cos phi) /
    //                     (1+r01^2 r12^2+2 r01 r12 cos phi).
    double r01 = (n0 - n1) / (n0 + n1), r12 = (n1 - n2) / (n1 + n2);
    double lam0 = 550.0, phi0 = 4.0 * PI * n1 * d / lam0;
    double cphi0 = std::cos(phi0);
    double num0 = r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * cphi0;
    double den0 = 1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * cphi0;
    double Ranalytic = clamp01(num0 / den0);
    double Rcode = thinFilmReflectance(n0, n1, n2, 0.0, d, 1.0, lam0);
    bool passB = std::fabs(Ranalytic - Rcode) < 1e-9;

    // (c) periodicity in phase: pick two wavelengths whose phase differs by 2*pi
    //     (phi = 4*pi*n1*d/lambda at normal incidence -> lambda = 4*pi*n1*d/phi).
    double phiA = 6.0, phiB = phiA + 2.0 * PI;
    double lamA = 4.0 * PI * n1 * d / phiA, lamB = 4.0 * PI * n1 * d / phiB;
    double RA = thinFilmReflectance(n0, n1, n2, 0.0, d, 1.0, lamA);
    double RB = thinFilmReflectance(n0, n1, n2, 0.0, d, 1.0, lamB);
    bool passC = std::fabs(RA - RB) < 1e-9;

    // (d) the film is genuinely iridescent: reflectance varies with wavelength.
    bool passD = (rmax - rmin) > 0.02;

    bool pass = passA && passB && passC && passD;
    std::printf("[checkthinfilm] reflectance range over sweep: [%.4f, %.4f]  (%s)\n",
                rmin, rmax, passA ? "in [0,1]" : "OUT OF RANGE");
    std::printf("[checkthinfilm] normal-incidence R: code=%.6f analytic=%.6f  (%s)\n",
                Rcode, Ranalytic, passB ? "ok" : "BAD");
    std::printf("[checkthinfilm] phase periodicity: R(phi)=%.6f R(phi+2pi)=%.6f  (%s)\n",
                RA, RB, passC ? "ok" : "BAD");
    std::printf("[checkthinfilm] iridescence (max-min R)=%.4f  (%s)\n",
                rmax - rmin, passD ? "ok" : "BAD");
    std::printf("[checkthinfilm] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Deterministic multilayer self-test. The general Abeles transfer-matrix
// multilayerReflectance() must reduce EXACTLY to the closed-form single-film
// Airy reflectance (thinFilmReflectance) when the stack is a single lossless
// layer over a lossless substrate — this is the strong correctness anchor for
// the N-layer material. Also checks (b) a symmetric quarter-wave dielectric
// mirror is highly reflective at its design wavelength, and (c) energy stays in
// [0,1] over a full angle/wavelength sweep of a multi-layer stack.
static int checkMultilayer() {
    const double n0 = 1.0, n1 = 1.30, n2 = 1.50, d = 300.0;

    // (a) single lossless layer == closed-form Airy across the full sweep.
    double maxDiff = 0.0;
    for (double lam = 380.0; lam <= 720.0; lam += 2.0)
        for (double ci = 0.05; ci <= 1.0; ci += 0.05) {
            double nL[1] = { n1 }, kL[1] = { 0.0 }, dL[1] = { d };
            double Rml = multilayerReflectance(n0, ci, lam, nL, kL, dL, 1, n2, 0.0);
            double Rtf = thinFilmReflectance(n0, n1, n2, 0.0, d, ci, lam);
            maxDiff = std::max(maxDiff, std::fabs(Rml - Rtf));
        }
    bool passA = maxDiff < 1e-9;

    // (b) a quarter-wave Bragg stack (alternating high/low index, each layer an
    //     optical quarter-wave at the design wavelength) is a strong reflector at
    //     that wavelength. n_H=2.30, n_L=1.38, design lam0=550nm, 8 pairs.
    const double nH = 2.30, nL_ = 1.38, lam0 = 550.0;
    const double dH = lam0 / (4.0 * nH), dL_ = lam0 / (4.0 * nL_);
    const int pairs = 8, NL = 2 * pairs;
    std::vector<double> sn(NL), sk(NL, 0.0), sd(NL);
    for (int p = 0; p < pairs; ++p) {
        sn[2 * p] = nH; sd[2 * p] = dH;
        sn[2 * p + 1] = nL_; sd[2 * p + 1] = dL_;
    }
    double Rdesign = multilayerReflectance(1.0, 1.0, lam0, sn.data(), sk.data(),
                                           sd.data(), NL, 1.52, 0.0);
    bool passB = Rdesign > 0.95;

    // (c) energy stays in [0,1] across a full sweep of the Bragg stack.
    bool inRange = true; double rmin = 1e9, rmax = -1e9;
    for (double lam = 380.0; lam <= 720.0; lam += 2.0)
        for (double ci = 0.05; ci <= 1.0; ci += 0.05) {
            double R = multilayerReflectance(1.0, ci, lam, sn.data(), sk.data(),
                                             sd.data(), NL, 1.52, 0.0);
            if (R < -1e-9 || R > 1.0 + 1e-9) inRange = false;
            rmin = std::min(rmin, R); rmax = std::max(rmax, R);
        }
    bool passC = inRange;

    bool pass = passA && passB && passC;
    std::printf("[checkmultilayer] single-layer vs Airy: max|dR|=%.3e  (%s)\n",
                maxDiff, passA ? "match" : "MISMATCH");
    std::printf("[checkmultilayer] quarter-wave Bragg stack R@%.0fnm=%.4f  (%s)\n",
                lam0, Rdesign, passB ? "high-reflect" : "TOO LOW");
    std::printf("[checkmultilayer] Bragg-stack sweep range: [%.4f, %.4f]  (%s)\n",
                rmin, rmax, passC ? "in [0,1]" : "OUT OF RANGE");
    std::printf("[checkmultilayer] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Deterministic diffraction-grating self-test. Validates that gratingDiffract
// obeys the exact vector grating equation, conserves the propagating-order set,
// reduces to specular reflection at m=0 (and with diffraction disabled), and is
// reciprocal. No scene / rendering needed — it probes the material sampler directly.
static int checkGrating() {
    Renderer r;
    Material m; m.type = MatType::Grating; m.reflect = constantSpectrum(1.0);
    m.grooveSpacing = 1000.0; m.grooveDir = {1, 0, 0}; m.gratingMaxOrder = 3;

    // Flat grating with normal +y at the origin; a downward-and-forward incident ray.
    Hit h; h.valid = true; h.p = {0, 0, 0}; h.ng = {0, 1, 0}; h.n = {0, 1, 0}; h.matId = 0;
    Vec3 din = normalize(Vec3{0.30, -0.80, 0.10});
    double lambda = 550.0;
    Pcg32 rng; rng.seed(42u, 7u);

    // Reconstruct the surface frame exactly as gratingDiffract does, to recover m.
    Vec3 nl = {0, 1, 0};
    Vec3 g  = normalize(Vec3{1, 0, 0});
    Vec3 t  = normalize(cross(nl, g));
    Vec3 ut = din - nl * dot(din, nl);
    double lod = lambda / m.grooveSpacing;

    // (a) diffraction OFF collapses to specular reflection (m=0 only).
    r.diffraction = false;
    bool ab; Ray r0 = r.gratingDiffract(m, h, din, lambda, rng, ab);
    Vec3 spec = normalize(reflect(din, nl));
    bool passA = !ab && length(r0.d - spec) < 1e-9;

    // Expected propagating-order set: |ut + m*(lambda/d)*t| < 1.
    bool expSeen[7] = {false};
    for (int mi = -3; mi <= 3; ++mi) { Vec3 a = ut + t * (mi * lod); if (dot(a, a) < 1.0) expSeen[mi + 3] = true; }

    // (b) every sampled order obeys the grating equation (integer m in range, unit
    //     outgoing vector on the incidence side); (c) exactly the propagating set is
    //     produced (coverage, and no evanescent order ever appears).
    r.diffraction = true;
    bool orderSeen[7] = {false};
    bool eqOK = true, unitOK = true, sideOK = true;
    const int Ns = 300000;
    for (int i = 0; i < Ns; ++i) {
        bool a2; Ray v = r.gratingDiffract(m, h, din, lambda, rng, a2);
        if (a2) continue;
        Vec3 vt = v.d - nl * dot(v.d, nl);
        double mrec = dot(vt - ut, t) / lod;
        int mi = (int)std::lround(mrec);
        if (std::fabs(mrec - (double)mi) > 1e-6) eqOK = false;
        if (mi < -3 || mi > 3) { eqOK = false; continue; }
        orderSeen[mi + 3] = true;
        if (std::fabs(length(v.d) - 1.0) > 1e-9) unitOK = false;
        if (dot(v.d, nl) <= 0.0) sideOK = false;
    }
    bool passB = eqOK && unitOK && sideOK;
    bool passC = true;
    for (int k = 0; k < 7; ++k) if (orderSeen[k] != expSeen[k]) passC = false;

    // (d) reciprocity: reversing the m=+1 outgoing ray reproduces the incident
    //     direction via the SAME order (the equation's difference term is sign-stable
    //     under direction reversal, so u<->v swap uses m, not -m).
    bool passD = true;
    Vec3 a1 = ut + t * lod;
    if (dot(a1, a1) < 1.0) {
        Vec3 v1  = normalize(a1 + nl * std::sqrt(1.0 - dot(a1, a1)));
        Vec3 rin = -v1;
        Vec3 rut = rin - nl * dot(rin, nl);
        Vec3 ra  = rut + t * lod;
        Vec3 rv  = normalize(ra + nl * std::sqrt(std::max(0.0, 1.0 - dot(ra, ra))));
        passD = length(rv - (-din)) < 1e-9;
    }

    bool pass = passA && passB && passC && passD;
    int nExp = 0; for (int k = 0; k < 7; ++k) nExp += expSeen[k];
    std::printf("[checkgrating] diffraction-off = specular reflection  (%s)\n", passA ? "ok" : "BAD");
    std::printf("[checkgrating] grating equation (integer orders, unit dirs)  (%s)\n", passB ? "ok" : "BAD");
    std::printf("[checkgrating] propagating orders produced=%d expected=%d  (%s)\n",
                [&]{int c=0;for(int k=0;k<7;++k)c+=orderSeen[k];return c;}(), nExp, passC ? "ok" : "BAD");
    std::printf("[checkgrating] reciprocity (reverse ray -> incident)  (%s)\n", passD ? "ok" : "BAD");
    std::printf("[checkgrating] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Deterministic RGB -> reflectance upsampling self-test (Jakob-Hanika sigmoid
// fit, src/upsample.h). Each colour is fitted to sigmoid coefficients, the
// resulting reflectance is integrated under D65 through the CIE observer, and
// converted back to linear sRGB. Validates, independent of the renderer:
//   (a) the fitted spectrum round-trips every test colour to small error;
//   (b) the reflectance stays physical (S in [0,1]) across the visible band;
//   (c) neutral greys round-trip essentially exactly.
// Note: pure white (1,1,1) is an inherently unreachable target for a sigmoid
// (S=1 everywhere requires p->inf), so it carries a small residual (~1-2%) that
// is expected and not a failure. Every non-saturated colour hits <1e-3.
static int checkUpsample() {
    struct C { double r, g, b; const char* name; };
    const C tests[] = {
        {0.0, 0.0, 0.0, "black"},   {1.0, 1.0, 1.0, "white"},
        {0.5, 0.5, 0.5, "grey"},    {0.8, 0.1, 0.1, "red"},
        {0.1, 0.7, 0.2, "green"},   {0.15, 0.2, 0.85, "blue"},
        {0.9, 0.8, 0.1, "yellow"},  {0.1, 0.75, 0.8, "cyan"},
        {0.8, 0.15, 0.75, "magenta"}, {0.7, 0.45, 0.2, "tan"},
    };
    const upsample::Basis& B = upsample::basis();

    double maxErr = 0.0, whiteErr = 0.0; bool physical = true;
    for (const C& c : tests) {
        std::array<double, 3> co = upsample::fit(c.r, c.g, c.b);
        // Reflectance stays in [0,1] over the band.
        for (int i = 0; i < B.N; ++i) {
            double s = upsample::reflAt(co, B.lam[i]);
            if (s < -1e-9 || s > 1.0 + 1e-9) physical = false;
        }
        // Integrate under D65 -> XYZ -> linear sRGB.
        double X, Y, Z; B.integrate(co, X, Y, Z);
        Vec3 lin = xyzToLinearSrgb(Vec3{X, Y, Z});
        double e = std::max({std::fabs(lin.x - c.r), std::fabs(lin.y - c.g), std::fabs(lin.z - c.b)});
        bool isWhite = (c.r == 1.0 && c.g == 1.0 && c.b == 1.0);
        if (isWhite) whiteErr = e; else maxErr = std::max(maxErr, e);
        std::printf("[checkupsample] %-8s (%.2f %.2f %.2f) -> (%.4f %.4f %.4f)  err=%.5f\n",
                    c.name, c.r, c.g, c.b, lin.x, lin.y, lin.z, e);
    }

    bool passA = maxErr < 1e-3;          // non-saturated colours are near-exact
    bool passB = physical;
    bool passW = whiteErr < 0.02;        // pure white: sigmoid can only asymptote to 1
    // (c) mid grey round-trips essentially exactly.
    std::array<double, 3> cg = upsample::fit(0.5, 0.5, 0.5);
    double gX, gY, gZ; B.integrate(cg, gX, gY, gZ);
    Vec3 gl = xyzToLinearSrgb(Vec3{gX, gY, gZ});
    double greyErr = std::max({std::fabs(gl.x - 0.5), std::fabs(gl.y - 0.5), std::fabs(gl.z - 0.5)});
    bool passC = greyErr < 1e-4;

    // (d) Illuminant (emission) upsample round-trips: build the A·sigmoid emission
    // SPD, integrate it under the *bare* CIE observer (illumBasis, no D65), convert
    // XYZ -> linear sRGB, and compare to the input colour. Unlike reflectance, the
    // magnitude is unbounded (carried by A), so even saturated primaries and the
    // 1,1,1 white round-trip accurately.
    const upsample::IllumBasis& IB = upsample::illumBasis();
    double illumErr = 0.0;
    for (const C& c : tests) {
        if (c.r == 0.0 && c.g == 0.0 && c.b == 0.0) continue;   // black -> zero SPD
        Spectrum spd = rgbToIlluminantJH(c.r, c.g, c.b);
        double X = 0, Y = 0, Z = 0;
        for (int i = 0; i < IB.N; ++i) {
            double s = spd(IB.lam[i]);
            X += s * IB.wX[i]; Y += s * IB.wY[i]; Z += s * IB.wZ[i];
        }
        Vec3 lin = xyzToLinearSrgb(Vec3{X, Y, Z});
        double e = std::max({std::fabs(lin.x - c.r), std::fabs(lin.y - c.g), std::fabs(lin.z - c.b)});
        illumErr = std::max(illumErr, e);
        std::printf("[checkupsample] illum %-8s (%.2f %.2f %.2f) -> (%.4f %.4f %.4f)  err=%.5f\n",
                    c.name, c.r, c.g, c.b, lin.x, lin.y, lin.z, e);
    }
    bool passD = illumErr < 2e-3;

    // (e) Smits 1999 reflectance upsample round-trips: build the tabulated basis
    // reflectance, integrate under D65 through the CIE observer, and compare to the
    // input. Smits is an approximate classic (lower fidelity than Jakob-Hanika), so
    // the tolerance is deliberately looser — this just guards gross regressions.
    double smitsErr = 0.0; bool smitsPhysical = true;
    for (const C& c : tests) {
        Spectrum spd = rgbToReflectanceSmits(c.r, c.g, c.b);
        for (int i = 0; i < B.N; ++i) {
            double s = spd(B.lam[i]);
            if (s < -1e-9 || s > 1.0 + 1e-9) smitsPhysical = false;
        }
        Vec3 lin = reflectanceToLinearSrgbD65(spd);
        double e = std::max({std::fabs(lin.x - c.r), std::fabs(lin.y - c.g), std::fabs(lin.z - c.b)});
        smitsErr = std::max(smitsErr, e);
        std::printf("[checkupsample] smits %-8s (%.2f %.2f %.2f) -> (%.4f %.4f %.4f)  err=%.5f\n",
                    c.name, c.r, c.g, c.b, lin.x, lin.y, lin.z, e);
    }
    bool passE = smitsPhysical && smitsErr < 0.20;   // approximate by design

    // (f) Plain calibrated 3-box reflectance round-trips: since the band heights are
    // solved from the inverse response matrix, unsaturated colours reconstruct nearly
    // exactly; saturated ones clamp (heights in [0,1]) and drift. Guards the calibration.
    double boxErr = 0.0; bool boxPhysical = true;
    for (const C& c : tests) {
        Spectrum spd = rgbToReflectanceBox(c.r, c.g, c.b);
        for (int i = 0; i < B.N; ++i) {
            double s = spd(B.lam[i]);
            if (s < -1e-9 || s > 1.0 + 1e-9) boxPhysical = false;
        }
        Vec3 lin = reflectanceToLinearSrgbD65(spd);
        double e = std::max({std::fabs(lin.x - c.r), std::fabs(lin.y - c.g), std::fabs(lin.z - c.b)});
        boxErr = std::max(boxErr, e);
        std::printf("[checkupsample] box   %-8s (%.2f %.2f %.2f) -> (%.4f %.4f %.4f)  err=%.5f\n",
                    c.name, c.r, c.g, c.b, lin.x, lin.y, lin.z, e);
    }
    bool passF = boxPhysical && boxErr < 0.30;   // crude basis; saturated colours clamp

    // (g) Meng 2015 smoothest-spectrum grid. Because the interpolation weights are
    // divided by each vertex's X+Y+Z, the mix lands on the requested chromaticity
    // *exactly*, so the only error is the brightness clamp — near-zero for anything
    // a smooth reflectance can actually be that bright. The second check is the one
    // that matters for the method: the result must be SMOOTHER (lower sum of squared
    // first differences) than the Jakob-Hanika fit of the same colour, since that is
    // the entire property being tabulated.
    double mengErr = 0.0, mengWhiteErr = 0.0; bool mengPhysical = true, mengSmoother = true;
    for (const C& c : tests) {
        if (c.r == 0.0 && c.g == 0.0 && c.b == 0.0) continue;   // black -> zero
        Spectrum spd = rgbToReflectanceMeng(c.r, c.g, c.b);
        Spectrum jh  = rgbToReflectanceJH(c.r, c.g, c.b);
        double roughM = 0.0, roughJ = 0.0, prevM = spd(B.lam[0]), prevJ = jh(B.lam[0]);
        for (int i = 0; i < B.N; ++i) {
            double s = spd(B.lam[i]);
            if (s < -1e-9 || s > 1.0 + 1e-9) mengPhysical = false;
            double dM = s - prevM, dJ = jh(B.lam[i]) - prevJ;
            roughM += dM * dM; roughJ += dJ * dJ;
            prevM = s; prevJ = jh(B.lam[i]);
        }
        if (roughM > roughJ) mengSmoother = false;
        Vec3 lin = reflectanceToLinearSrgbD65(spd);
        double e = std::max({std::fabs(lin.x - c.r), std::fabs(lin.y - c.g), std::fabs(lin.z - c.b)});
        if (c.r == 1.0 && c.g == 1.0 && c.b == 1.0) mengWhiteErr = e; else mengErr = std::max(mengErr, e);
        std::printf("[checkupsample] meng  %-8s (%.2f %.2f %.2f) -> (%.4f %.4f %.4f)  err=%.5f  rough=%.5f (jh %.5f)\n",
                    c.name, c.r, c.g, c.b, lin.x, lin.y, lin.z, e, roughM, roughJ);
    }
    // Tabulated + interpolated, so allow a hair more slack than the analytic fits;
    // white is capped by the brightest smooth reflectance of that chromaticity.
    bool passG = mengPhysical && mengSmoother && mengErr < 5e-3 && mengWhiteErr < 0.02;

    // (h) USER-DECLARED upsamplers (K1 remainder): `upsample "n" { expr "f(r,g,b,w)" }`
    // named as `rgb:n` / `hsv:n` / `hsl:n`. Unlike (a)-(g), which test closed-form fits
    // that can be evaluated directly, this one has to run the LOADER — the feature IS the
    // binding of a name to a colour head, so what's under test is a property of loading.
    // Every assert compares against a value computed here in C++ from the same inputs, so
    // the test pins the semantics (which variable is which, when the space conversion
    // happens, what `spec:` samples) rather than a number that could drift with defaults.
    bool passH = true;
    {
        auto uchk = [&](const char* what, bool cond) {
            if (!cond) { std::printf("[checkupsample] user %-46s BAD\n", what); passH = false; }
        };
        // Load a fragment and hand back the probe material's reflect spectrum. The probe
        // quad always uses `probe`, so the resolved index comes off its first triangle.
        auto probeReflect = [&](const std::string& decls, Spectrum& out) -> bool {
            std::string src =
                "scene { units meters }\n" + decls + "\n"
                "quad { origin 0 0 0  u 1 0 0  v 0 1 0  material probe }\n"
                "light area { origin 0 0.99 0.1  u 1 0 0  v 0 0 0.4  normal 0 -1 0  spd preset:bb6500 }\n"
                "camera \"c\" { eye 0.5 0.5 2  look_at 0.5 0.5 0  up 0 1 0  fov_y 32  film { res 8 8 } }\n";
            ftsl::Loaded L; std::string e;
            if (!ftsl::loadSource(src, "<checkupsample>", L, e)) {
                std::printf("[checkupsample] user load FAILED: %s\n", e.c_str());
                return false;
            }
            int mi = L.scene.tris.empty() ? 0 : L.scene.tris[0].matId;
            out = L.scene.mats[mi].reflect;
            // NOTE: `L` dies here. The returned Spectrum must still work — that is the
            // whole point of applyUpsample capturing shared_ptrs rather than the Builder,
            // and every sample below is taken AFTER this scope exits.
            return true;
        };
        // A scene that must NOT load, and whose error mentions `needle`.
        auto uReject = [&](const char* what, const std::string& decls, const char* needle) {
            std::string src =
                "scene { units meters }\n" + decls + "\n"
                "quad { origin 0 0 0  u 1 0 0  v 0 1 0  material probe }\n"
                "light area { origin 0 0.99 0.1  u 1 0 0  v 0 0 0.4  normal 0 -1 0  spd preset:bb6500 }\n"
                "camera \"c\" { eye 0.5 0.5 2  look_at 0.5 0.5 0  up 0 1 0  fov_y 32  film { res 8 8 } }\n";
            ftsl::Loaded L; std::string e;
            if (ftsl::loadSource(src, "<checkupsample>", L, e)) { uchk(what, false); return; }
            if (e.find(needle) == std::string::npos) {
                std::printf("[checkupsample] user %-46s BAD (error was: %s)\n", what, e.c_str());
                passH = false;
            }
        };
        const double lams[] = { 380.0, 450.0, 550.0, 632.8, 780.0 };

        // h1. A constant body is that constant at every wavelength — the smallest possible
        //     proof that the program is compiled, stored and evaluated at all.
        {
            Spectrum s;
            if (probeReflect("upsample \"k\" { expr \"0.375\" }\n"
                             "material \"probe\" { type diffuse  reflect rgb:k 0.2 0.5 0.9 }", s)) {
                bool okc = true;
                for (double w : lams) okc = okc && std::fabs(s(w) - 0.375) < 1e-12;
                uchk("constant body is constant in wavelength", okc);
            } else passH = false;
        }

        // h2. r/g/b are the LINEAR sRGB triple and w is the wavelength in nm, each landing
        //     in its own slot. The formula weights all four differently so a swapped pair
        //     cannot pass by coincidence.
        {
            Spectrum s;
            const double R = 0.2, G = 0.5, B = 0.9;
            if (probeReflect("upsample \"m\" { expr \"r*1 + g*10 + b*100 + w*0.001\" }\n"
                             "material \"probe\" { type diffuse  reflect rgb:m 0.2 0.5 0.9 }", s)) {
                bool okc = true;
                for (double w : lams)
                    okc = okc && std::fabs(s(w) - (R + G * 10 + B * 100 + w * 0.001)) < 1e-9;
                uchk("r,g,b,w each reach their own slot", okc);
            } else passH = false;
        }

        // h3. The space conversion runs BEFORE the body, so an `hsv:` head and the `rgb:`
        //     head fed the converted triple are indistinguishable. This is the invariant
        //     that lets an author pick a colour space without re-reading the upsampler.
        {
            Spectrum a, b;
            Vec3 c = hsvToRgb(0.3, 0.8, 0.6);
            char rgbDecl[256];
            std::snprintf(rgbDecl, sizeof rgbDecl,
                          "upsample \"m\" { expr \"r*1 + g*10 + b*100\" }\n"
                          "material \"probe\" { type diffuse  reflect rgb:m %.17g %.17g %.17g }",
                          c.x, c.y, c.z);
            if (probeReflect("upsample \"m\" { expr \"r*1 + g*10 + b*100\" }\n"
                             "material \"probe\" { type diffuse  reflect hsv:m 0.3 0.8 0.6 }", a) &&
                probeReflect(rgbDecl, b)) {
                uchk("hsv: head converts to linear sRGB before the body",
                     std::fabs(a(550.0) - b(550.0)) < 1e-9);
            } else passH = false;
        }

        // h4. `spec:<name>(w)` samples a declared spectrum AT THE PASSED WAVELENGTH. The
        //     gaussian is checked against its own closed form, so this pins both that the
        //     right curve was resolved and that `w` really is the argument (a constant
        //     would match at the centre only).
        {
            Spectrum s;
            const double R = 0.8;
            if (probeReflect("spectrum \"g\" = gaussian center=550 sigma=30 amp=1\n"
                             "upsample \"basis\" { expr \"r*spec:g(w)\" }\n"
                             "material \"probe\" { type diffuse  reflect rgb:basis 0.8 0.1 0.1 }", s)) {
                bool okc = true;
                for (double w : lams) {
                    double t = (w - 550.0) / 30.0;
                    okc = okc && std::fabs(s(w) - R * std::exp(-0.5 * t * t)) < 1e-9;
                }
                uchk("spec:<name>(w) samples the named spectrum at w", okc);
            } else passH = false;
        }

        // h5. A MEASURED BASIS — the reason `spec:` exists at all. Three named spectra
        //     weighted by the three channels is the classic linear-basis upsampler, and it
        //     must equal the same sum computed here. Also proves the spectrum vector stays
        //     valid across several resolved indices (it is append-only for exactly this).
        {
            Spectrum s;
            const double R = 0.3, G = 0.6, B = 0.1;
            if (probeReflect("spectrum \"sr\" = gaussian center=620 sigma=40 amp=1\n"
                             "spectrum \"sg\" = gaussian center=540 sigma=40 amp=1\n"
                             "spectrum \"sb\" = gaussian center=460 sigma=40 amp=1\n"
                             "upsample \"basis\" { expr \"r*spec:sr(w) + g*spec:sg(w) + b*spec:sb(w)\" }\n"
                             "material \"probe\" { type diffuse  reflect rgb:basis 0.3 0.6 0.1 }", s)) {
                bool okc = true;
                for (double w : lams) {
                    auto gau = [&](double c0) { double t = (w - c0) / 40.0; return std::exp(-0.5 * t * t); };
                    okc = okc && std::fabs(s(w) - (R * gau(620) + G * gau(540) + B * gau(460))) < 1e-9;
                }
                uchk("three-spectrum measured basis matches by hand", okc);
            } else passH = false;
        }

        // h6. Loud refusals. Each of these has a specific failure the author needs named,
        //     and the alternative in every case is a silent wrong answer.
        uReject("an unknown upsampler names itself",
                "material \"probe\" { type diffuse  reflect rgb:nope 0.2 0.5 0.9 }",
                "unknown upsampler");
        uReject("an upsample with no expr is refused",
                "upsample \"m\" { }\n"
                "material \"probe\" { type diffuse  reflect rgb:m 0.2 0.5 0.9 }",
                "no `expr`");
        // The trap this exists for: `r` is RADIUS in the surface vocabulary and RED here.
        // Any surface name must be rejected BY NAME, never silently read as something else.
        uReject("a surface variable in an upsample body is refused",
                "upsample \"m\" { expr \"x + y\" }\n"
                "material \"probe\" { type diffuse  reflect rgb:m 0.2 0.5 0.9 }",
                "surface/shading variable");
        uReject("an unknown identifier lists the upsample vocabulary",
                "upsample \"m\" { expr \"q*2\" }\n"
                "material \"probe\" { type diffuse  reflect rgb:m 0.2 0.5 0.9 }",
                "unknown identifier");
        uReject("spec: naming a missing spectrum says which",
                "upsample \"m\" { expr \"spec:ghost(w)\" }\n"
                "material \"probe\" { type diffuse  reflect rgb:m 0.2 0.5 0.9 }",
                "unknown spectrum 'ghost'");
        uReject("an uncalled spec: reference asks for the wavelength",
                "spectrum \"g\" = gaussian center=550 sigma=30 amp=1\n"
                "upsample \"m\" { expr \"spec:g\" }\n"
                "material \"probe\" { type diffuse  reflect rgb:m 0.2 0.5 0.9 }",
                "must be called with a wavelength");
        // `tex:` has no hit point to sample at here; refusing beats silently returning 0.
        uReject("a texture sample in an upsample body is out of scope",
                "texture \"t\" { file scenes/graychecker.ppm  encoding linear }\n"
                "upsample \"m\" { expr \"tex:t(0.5, 0.5)\" }\n"
                "material \"probe\" { type diffuse  reflect rgb:m 0.2 0.5 0.9 }",
                "out of scope");
        // And the same `spec:` is NOT in scope in an ordinary pattern, which has a hit
        // point but no wavelength — the symmetric half of the scope rule.
        uReject("spec: outside an upsample body is out of scope",
                "spectrum \"g\" = gaussian center=550 sigma=30 amp=1\n"
                "pattern \"p\" { expr \"spec:g(550)\" }\n"
                "material \"probe\" { type diffuse  reflect pattern:p }",
                "out of scope");
    }

    bool pass = passA && passB && passW && passC && passD && passE && passF && passG && passH;
    std::printf("[checkupsample] round-trip max error (excl. white) = %.5f  (%s)\n", maxErr, passA ? "ok" : "BAD");
    std::printf("[checkupsample] reflectance in [0,1]  (%s)\n", passB ? "ok" : "BAD");
    std::printf("[checkupsample] pure-white residual = %.5f (<0.02 expected)  (%s)\n", whiteErr, passW ? "ok" : "BAD");
    std::printf("[checkupsample] mid-grey round-trip = %.6f  (%s)\n", greyErr, passC ? "ok" : "BAD");
    std::printf("[checkupsample] illuminant round-trip max error = %.5f  (%s)\n", illumErr, passD ? "ok" : "BAD");
    std::printf("[checkupsample] smits round-trip max error = %.5f (<0.20 expected)  (%s)\n", smitsErr, passE ? "ok" : "BAD");
    std::printf("[checkupsample] box round-trip max error = %.5f (<0.30 expected)  (%s)\n", boxErr, passF ? "ok" : "BAD");
    std::printf("[checkupsample] meng round-trip max error = %.5f (excl. white %.5f); smoother than JH: %s  (%s)\n",
                mengErr, mengWhiteErr, mengSmoother ? "yes" : "NO", passG ? "ok" : "BAD");
    std::printf("[checkupsample] user-declared `upsample` blocks  (%s)\n", passH ? "ok" : "BAD");
    std::printf("[checkupsample] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Deterministic N-D grid sampler self-test (src/pattern.h: PatGrid / patGridSample,
// reached from a pattern expression as `grid:<name>(c0, …)`). Validates, with no
// scene and no renderer:
//   (a) sample points reproduce the stored samples EXACTLY (no off-by-one, no drift);
//   (b) C-order flattening — axis 0 is the OUTERMOST axis, matching loom's data.Grid
//       and the nesting-is-the-shape authoring rule;
//   (c) separable N-linear interpolation is exact for a multilinear function, in 1-D
//       through 4-D (the strongest available analytic check on the corner weights);
//   (d) the three out-of-box policies: clamp (edge-extend), wrap (period hi-lo, with
//       sample n-1 aliasing sample 0) and extrapolate (the boundary cell continues);
//   (e) the compile path: `grid:<name>(…)` resolves through a PatTableScope, takes the
//       GRID's own dimensionality as its arity, and pushes coordinates in axis order.
static int checkGrid() {
    auto mk = [](int ndim, const int* shape, const double* lo, const double* hi,
                 PatGridOutside os, int off, int count) {
        PatGrid g;
        g.ndim = ndim;
        for (int a = 0; a < ndim; ++a) { g.shape[a] = shape[a]; g.lo[a] = lo[a]; g.hi[a] = hi[a]; }
        g.outside = os; g.off = off; g.count = count;
        return g;
    };
    double worst = 0.0;
    auto chk = [&](const char* what, double got, double want, double tol) {
        double e = std::fabs(got - want);
        if (e > worst) worst = e;
        if (e > tol)
            std::printf("[checkgrid] %-34s got %.9f want %.9f  err=%.3g  BAD\n", what, got, want, e);
        return e <= tol;
    };
    bool ok = true;

    // ---- (a)+(b) exact sample recovery and C-order on a 2x3 grid -------------
    // data laid out row-major: value at (i, j) == i*3 + j.
    std::vector<float> pool;
    const int off23 = (int)pool.size();
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) pool.push_back((float)(i * 3 + j));
    const int    sh23[2] = {2, 3};
    const double lo23[2] = {0, 0}, hi23[2] = {1, 2};   // unit-spacing index lattice
    PatGrid g23 = mk(2, sh23, lo23, hi23, PatGridOutside::Clamp, off23, 6);
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) {
        double c[2] = {(double)i, (double)j};
        ok &= chk("2x3 sample recovery (C order)", patGridSample(g23, pool.data(), (int)pool.size(), c),
                  (double)(i * 3 + j), 1e-12);
    }
    // Midpoint between (0,0)=0 and (0,1)=1 is 0.5; between (0,0) and (1,0)=3 is 1.5.
    { double c[2] = {0.0, 0.5}; ok &= chk("2x3 bilinear mid (axis 1)", patGridSample(g23, pool.data(), (int)pool.size(), c), 0.5, 1e-12); }
    { double c[2] = {0.5, 0.0}; ok &= chk("2x3 bilinear mid (axis 0)", patGridSample(g23, pool.data(), (int)pool.size(), c), 1.5, 1e-12); }

    // ---- (c) N-linear exactness for a multilinear function, 1-D .. 4-D -------
    // f(t0..t_{n-1}) = prod(0.3 + 0.7*t_a) is multilinear, so N-linear interpolation
    // of its corner values must reproduce it EXACTLY at every interior point.
    for (int nd = 1; nd <= PAT_ND_MAX_DIM; ++nd) {
        int    shape[PAT_ND_MAX_DIM];
        double lo[PAT_ND_MAX_DIM], hi[PAT_ND_MAX_DIM];
        int    n = 1;
        for (int a = 0; a < nd; ++a) { shape[a] = 2; lo[a] = -1.0; hi[a] = 3.0; n *= 2; }
        const int offN = (int)pool.size();
        for (int c = 0; c < n; ++c) {
            // C order: axis 0 is the OUTERMOST, so its index is the high bit.
            double f = 1.0;
            for (int a = 0; a < nd; ++a) {
                int up = (c >> (nd - 1 - a)) & 1;
                f *= 0.3 + 0.7 * (double)up;
            }
            pool.push_back((float)f);
        }
        PatGrid g = mk(nd, shape, lo, hi, PatGridOutside::Clamp, offN, n);
        const double ts[3] = {0.125, 0.5, 0.875};
        for (int s = 0; s < 3; ++s) {
            double co[PAT_ND_MAX_DIM], want = 1.0;
            for (int a = 0; a < nd; ++a) {
                double t = ts[(s + a) % 3];
                co[a] = lo[a] + t * (hi[a] - lo[a]);
                want *= 0.3 + 0.7 * t;
            }
            char lbl[64]; std::snprintf(lbl, sizeof lbl, "%d-D multilinear exactness", nd);
            ok &= chk(lbl, patGridSample(g, pool.data(), (int)pool.size(), co), want, 1e-6);
        }
    }

    // ---- (d) out-of-box policies on a 1-D ramp over [0,1] --------------------
    const int offR = (int)pool.size();
    pool.push_back(0.25f); pool.push_back(0.75f);       // data { 0.25 0.75 }, lo 0, hi 1
    const int    shR[1] = {2};
    const double loR[1] = {0.0}, hiR[1] = {1.0};
    PatGrid gClamp = mk(1, shR, loR, hiR, PatGridOutside::Clamp, offR, 2);
    PatGrid gExtra = mk(1, shR, loR, hiR, PatGridOutside::Extrapolate, offR, 2);
    { double c[1] = {0.5};  ok &= chk("clamp: interior",       patGridSample(gClamp, pool.data(), (int)pool.size(), c), 0.50, 1e-9); }
    { double c[1] = {-2.0}; ok &= chk("clamp: below lo",       patGridSample(gClamp, pool.data(), (int)pool.size(), c), 0.25, 1e-9); }
    { double c[1] = {5.0};  ok &= chk("clamp: above hi",       patGridSample(gClamp, pool.data(), (int)pool.size(), c), 0.75, 1e-9); }
    { double c[1] = {-1.0}; ok &= chk("extrapolate: below lo", patGridSample(gExtra, pool.data(), (int)pool.size(), c), -0.25, 1e-9); }
    { double c[1] = {2.0};  ok &= chk("extrapolate: above hi", patGridSample(gExtra, pool.data(), (int)pool.size(), c), 1.25, 1e-9); }

    // Wrap: 5 samples over [0,1] with sample 4 aliasing sample 0 -> a period-1 sawtooth.
    const int offW = (int)pool.size();
    const float saw[5] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    for (float f : saw) pool.push_back(f);
    const int    shW[1] = {5};
    const double loW[1] = {0.0}, hiW[1] = {1.0};
    PatGrid gWrap = mk(1, shW, loW, hiW, PatGridOutside::Wrap, offW, 5);
    { double c[1] = {0.375};  ok &= chk("wrap: interior",  patGridSample(gWrap, pool.data(), (int)pool.size(), c), 0.375, 1e-6); }
    { double c[1] = {1.375};  ok &= chk("wrap: +1 period", patGridSample(gWrap, pool.data(), (int)pool.size(), c), 0.375, 1e-6); }
    { double c[1] = {-0.625}; ok &= chk("wrap: -1 period", patGridSample(gWrap, pool.data(), (int)pool.size(), c), 0.375, 1e-6); }
    { double c[1] = {1.0};    ok &= chk("wrap: seam aliases sample 0", patGridSample(gWrap, pool.data(), (int)pool.size(), c), 0.0, 1e-6); }

    // ---- (e) the `grid:<name>(…)` compile + eval path ------------------------
    // A two-entry scope: "ramp" is the 1-D clamp ramp, "tbl" is the 2x3 C-order grid.
    struct Scope {
        static int lookup(const void*, PatTableKind kind, const char* name, int* ndim) {
            if (kind != PatTableKind::Grid) return -1;
            if (!std::strcmp(name, "ramp")) { if (ndim) *ndim = 1; return 0; }
            if (!std::strcmp(name, "tbl"))  { if (ndim) *ndim = 2; return 1; }
            return -1;
        }
    };
    PatTableScope scope; scope.self = nullptr; scope.lookup = &Scope::lookup;
    PatGrid grids[2] = {gClamp, g23};

    struct Case { const char* expr; double u, v; double want; };
    const Case cases[] = {
        {"grid:ramp(u)",             0.5,  0.0, 0.50},
        {"grid:ramp(2*u)",           0.25, 0.0, 0.50},
        {"grid:tbl(0, 1)",           0.0,  0.0, 1.00},   // C order: (i=0, j=1) -> 1
        {"grid:tbl(1, 0)",           0.0,  0.0, 3.00},   // (i=1, j=0) -> 3
        {"grid:tbl(u, v) + 1",       1.0,  2.0, 6.00},   // corner (1,2) = 5
        {"grid:ramp(grid:tbl(0,0))", 0.0,  0.0, 0.25},   // nested: tbl(0,0)=0 -> ramp(0)
    };
    for (const Case& cs : cases) {
        std::vector<PatNode> prog; std::string perr;
        if (!compilePatternExpr(cs.expr, prog, perr, false, nullptr, &scope)) {
            std::printf("[checkgrid] compile `%s` FAILED: %s\n", cs.expr, perr.c_str());
            ok = false; continue;
        }
        PatCtx c;
        c.u = cs.u; c.v = cs.v;
        c.grids = grids; c.nGrids = 2;
        c.dataPool = pool.data(); c.dataPoolN = (int)pool.size();
        char lbl[80]; std::snprintf(lbl, sizeof lbl, "expr %s", cs.expr);
        ok &= chk(lbl, patternEval(prog.data(), (int)prog.size(), c), cs.want, 1e-9);
    }
    // Arity is the GRID's dimensionality, so a wrong argument count must be an error,
    // and an out-of-scope / unknown grid must not silently compile to 0.
    struct Bad { const char* expr; const char* why; };
    const Bad bads[] = {
        {"grid:tbl(u)",     "2-D grid called with 1 argument"},
        {"grid:ramp(u, v)", "1-D grid called with 2 arguments"},
        {"grid:nope(u)",    "unknown grid name"},
        {"grid:ramp",       "grid referenced without a call"},
        {"grid(u)",         "bare `grid` with no name"},
        {"scatter:ramp(u)", "a grid name is NOT visible in the scatter namespace"},
    };
    for (const Bad& bd : bads) {
        std::vector<PatNode> prog; std::string perr;
        if (compilePatternExpr(bd.expr, prog, perr, false, nullptr, &scope)) {
            std::printf("[checkgrid] `%s` compiled but should be rejected (%s)  BAD\n", bd.expr, bd.why);
            ok = false;
        }
    }
    // With no grid scope at all (a load-time constant site), `grid:` must be refused.
    {
        std::vector<PatNode> prog; std::string perr;
        if (compilePatternExpr("grid:ramp(u)", prog, perr, false, nullptr, nullptr)) {
            std::printf("[checkgrid] `grid:ramp(u)` compiled with no grid scope  BAD\n");
            ok = false;
        }
    }

    // ---- (f) the not-found guard ABANDONS the program, it does not corrupt the stack --
    // A `grid:` node whose tables aren't bound used to push 0 WITHOUT popping its
    // operands, so patternEval returned st[0] — the first COORDINATE — as the sample.
    // That was reachable for real (`medium { density pattern:<p> }` copies a
    // table-scoped pattern's nodes into a medium), and it silently rendered a mirrored
    // field. Pin it: an unbound table evaluates to 0, never to a coordinate.
    {
        struct Unbound { const char* expr; double u, v; };
        const Unbound ubs[] = {
            {"grid:ramp(u)",    0.8, 0.0},     // stack-corrupt answer would be 0.8
            {"grid:tbl(u, v)",  0.7, 1.0},     // ... and 0.7 for the 2-D call
        };
        for (const Unbound& b : ubs) {
            std::vector<PatNode> prog; std::string perr;
            if (!compilePatternExpr(b.expr, prog, perr, false, nullptr, &scope)) {
                std::printf("[checkgrid] compile `%s` FAILED: %s\n", b.expr, perr.c_str());
                ok = false; continue;
            }
            PatCtx c;
            c.u = b.u; c.v = b.v;                       // grids deliberately left unbound
            c.dataPool = pool.data(); c.dataPoolN = (int)pool.size();
            char lbl[96];
            std::snprintf(lbl, sizeof lbl, "unbound `%s` -> 0, not a coord", b.expr);
            ok &= chk(lbl, patternEval(prog.data(), (int)prog.size(), c), 0.0, 1e-12);
        }
    }

    // ---- (g) end-to-end through a medium's density field ---------------------
    // `density "grid:ramp(x)"` is a SAMPLED volume, and Scene::patTables() is the view
    // that carries the tables all the way to Medium::densityAt. Verify the hand-off,
    // and that omitting it fails as a clean 0 rather than as the x coordinate.
    {
        Scene sc;
        sc.dataPool.assign(pool.begin(), pool.end());
        sc.grids.push_back(gClamp);                     // index 0 == "ramp" in `scope`
        sc.grids.push_back(g23);                        // index 1 == "tbl"
        Medium med; std::string perr;
        if (!compilePatternExpr("grid:ramp(x)", med.density, perr, false, nullptr, &scope)) {
            std::printf("[checkgrid] compile medium density FAILED: %s\n", perr.c_str());
            ok = false;
        } else {
            const PatTables tabs = sc.patTables();
            const Vec3 p(0.8, 0.0, 0.0);                // ramp(0.8) = 0.25 + 0.8*0.5
            ok &= chk("medium density `grid:ramp(x)`", med.densityAt(p, &tabs), 0.65, 1e-6);
            ok &= chk("medium density, tables omitted", med.densityAt(p, nullptr), 0.0, 1e-12);
        }
    }

    std::printf("[checkgrid] worst absolute error = %.3g\n", worst);
    std::printf("[checkgrid] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Deterministic N-D scatter sampler self-test (src/pattern.h: PatScatter /
// patScatterSample, reached from a pattern expression as `scatter:<name>(c0, …)`).
// The ragged sibling of -checkgrid; validates, with no scene and no renderer:
//   (a) EXACT reproduction at each sample position — the property that makes Shepard
//       an interpolant rather than merely an approximation (and the branch that
//       removes the 1/0 singularity there);
//   (b) partition of unity: a constant-valued sample set reads back as that constant
//       EVERYWHERE, in 1-D through 4-D. This is the analytic check on normalisation;
//   (c) symmetry — equidistant samples blend to their plain mean, independent of
//       `power`, which pins the weight formula's distance handling;
//   (d) `power` actually sharpens: a higher exponent pulls a query nearer the closer
//       sample, checked against the closed-form two-sample weight;
//   (e) far-field behaviour: at large distance the blend tends to the plain mean
//       (it flattens rather than diverging — a scatter's answer to a grid's `clamp`);
//   (f) the compile path: `scatter:<name>(…)` resolves through a PatTableScope, takes
//       the SCATTER's own dimensionality as its arity, and lives in a namespace
//       separate from `grid:`.
static int checkScatter() {
    double worst = 0.0;
    bool ok = true;
    auto chk = [&](const char* what, double got, double want, double tol) {
        double e = std::fabs(got - want);
        if (e > worst) worst = e;
        if (e > tol)
            std::printf("[checkscatter] %-40s got %.9f want %.9f  err=%.3g  BAD\n", what, got, want, e);
        return e <= tol;
    };
    // Build a scatter from (position, value) tuples into the shared flat pool.
    std::vector<float> pool;
    auto add = [&](int ndim, const std::vector<double>& flat, double power) {
        PatScatter s;
        s.ndim = ndim; s.power = power; s.eps = 1e-9;
        s.off = (int)pool.size();
        s.count = (int)(flat.size() / (size_t)(ndim + 1));
        for (double d : flat) pool.push_back((float)d);
        return s;
    };
    auto smp = [&](const PatScatter& s, const double* q) {
        return patScatterSample(s, pool.data(), (int)pool.size(), q);
    };

    // ---- (a) exact reproduction at every sample position ---------------------
    // Four samples in 2-D with deliberately unequal values; each must read back exactly.
    const std::vector<double> quad = {
        0.0, 0.0,  0.10,
        1.0, 0.0,  0.90,
        0.0, 1.0,  0.40,
        1.0, 1.0,  0.70,
    };
    PatScatter s2 = add(2, quad, 2.0);
    for (int i = 0; i < 4; ++i) {
        double q[2] = {quad[i * 3 + 0], quad[i * 3 + 1]};
        char lbl[80]; std::snprintf(lbl, sizeof lbl, "exact at sample %d", i);
        // 1e-6, not 0: the pool stores FLOATS, so an authored 0.10 is only float-exact.
        // "Exact" here means "the stored sample, with no interpolation error on top".
        ok &= chk(lbl, smp(s2, q), quad[i * 3 + 2], 1e-6);
    }

    // ---- (b) partition of unity, 1-D .. 4-D ----------------------------------
    // Every sample carries the SAME value, so a normalised blend must return it at any
    // query — including one far outside the samples' own extent.
    const double kConst = 0.375;
    for (int nd = 1; nd <= PAT_ND_MAX_DIM; ++nd) {
        std::vector<double> flat;
        // 2^nd samples on the unit cube's corners, all valued kConst.
        for (int c = 0; c < (1 << nd); ++c) {
            for (int a = 0; a < nd; ++a) flat.push_back(((c >> a) & 1) ? 1.0 : 0.0);
            flat.push_back(kConst);
        }
        PatScatter s = add(nd, flat, 2.0);
        const double qs[3][4] = {{0.5, 0.5, 0.5, 0.5}, {0.2, 0.7, 0.1, 0.9}, {7.0, -3.0, 5.0, 2.0}};
        for (int k = 0; k < 3; ++k) {
            char lbl[80]; std::snprintf(lbl, sizeof lbl, "%d-D partition of unity, q%d", nd, k);
            ok &= chk(lbl, smp(s, qs[k]), kConst, 1e-9);
        }
    }

    // ---- (c) symmetry: equidistant samples blend to the plain mean -----------
    // Midpoint of the 2-D quad above: all four are equidistant, so the answer is the
    // mean of the values regardless of the exponent.
    const double quadMean = (0.10 + 0.90 + 0.40 + 0.70) / 4.0;
    { double q[2] = {0.5, 0.5}; ok &= chk("2-D midpoint == mean (power 2)", smp(s2, q), quadMean, 1e-6); }
    {
        PatScatter s2p = add(2, quad, 6.0);
        double q[2] = {0.5, 0.5};
        ok &= chk("2-D midpoint == mean (power 6)", smp(s2p, q), quadMean, 1e-6);
    }

    // ---- (d) `power` sharpens, matching the closed-form two-sample weight -----
    // Samples at 0 and 1 valued 0 and 1: at q the weights are q^-p and (1-q)^-p, so the
    // result is (1-q)^p / (q^p + (1-q)^p) — an independent formula, not a re-derivation
    // of the implementation.
    const std::vector<double> pair = {0.0, 0.0,   1.0, 1.0};
    const double q1 = 0.25;
    for (double p : {1.0, 2.0, 3.0, 8.0}) {
        PatScatter s1 = add(1, pair, p);
        double q[1] = {q1};
        const double wa = std::pow(q1, -p), wb = std::pow(1.0 - q1, -p);
        const double want = wb / (wa + wb);          // value 0 at a, 1 at b
        char lbl[80]; std::snprintf(lbl, sizeof lbl, "1-D two-sample, power %.0f", p);
        ok &= chk(lbl, smp(s1, q), want, 1e-9);
    }
    // Sharper exponent must move the answer TOWARD the nearer sample (value 0 at 0.0).
    {
        PatScatter sSoft = add(1, pair, 1.0), sHard = add(1, pair, 8.0);
        double q[1] = {q1};
        if (!(smp(sHard, q) < smp(sSoft, q))) {
            std::printf("[checkscatter] higher power did not sharpen toward the nearer sample  BAD\n");
            ok = false;
        }
    }

    // ---- (e) far field tends to the plain mean -------------------------------
    // At a great distance every sample is essentially equidistant, so the blend
    // flattens to the unweighted mean instead of diverging.
    { double q[2] = {1e6, 1e6}; ok &= chk("far field -> mean", smp(s2, q), quadMean, 1e-4); }

    // ---- (f) the `scatter:<name>(…)` compile + eval path ----------------------
    // "pts" is the 2-D quad; "line" is the 1-D pair. Grid lookups must MISS: the two
    // datatypes share the scope object but not the namespace.
    struct Scope {
        static int lookup(const void*, PatTableKind kind, const char* name, int* ndim) {
            if (kind != PatTableKind::Scatter) return -1;
            if (!std::strcmp(name, "pts"))  { if (ndim) *ndim = 2; return 0; }
            if (!std::strcmp(name, "line")) { if (ndim) *ndim = 1; return 1; }
            return -1;
        }
    };
    PatTableScope scope; scope.self = nullptr; scope.lookup = &Scope::lookup;
    PatScatter tables[2] = {s2, add(1, pair, 2.0)};

    struct Case { const char* expr; double u, v; double want; };
    const double wa = std::pow(0.25, -2.0), wb = std::pow(0.75, -2.0);
    const Case cases[] = {
        {"scatter:pts(0, 0)",              0.0,  0.0, 0.10},        // exact at a sample
        {"scatter:pts(u, v)",              1.0,  1.0, 0.70},        // ... via variables
        {"scatter:pts(0.5, 0.5)",          0.0,  0.0, quadMean},    // equidistant -> mean
        {"scatter:line(0.25)",             0.0,  0.0, wb / (wa + wb)},
        {"scatter:line(u) * 2",            0.25, 0.0, 2.0 * wb / (wa + wb)},
        {"scatter:line(scatter:pts(0,0))", 0.0,  0.0,               // nested: pts(0,0) = 0.1
             std::pow(0.1, -2.0) * 0.0 / (std::pow(0.1, -2.0) + std::pow(0.9, -2.0)) +
             std::pow(0.9, -2.0) * 1.0 / (std::pow(0.1, -2.0) + std::pow(0.9, -2.0))},
    };
    for (const Case& cs : cases) {
        std::vector<PatNode> prog; std::string perr;
        if (!compilePatternExpr(cs.expr, prog, perr, false, nullptr, &scope)) {
            std::printf("[checkscatter] compile `%s` FAILED: %s\n", cs.expr, perr.c_str());
            ok = false; continue;
        }
        PatCtx c;
        c.u = cs.u; c.v = cs.v;
        c.scatters = tables; c.nScatters = 2;
        c.dataPool = pool.data(); c.dataPoolN = (int)pool.size();
        char lbl[96]; std::snprintf(lbl, sizeof lbl, "expr %s", cs.expr);
        ok &= chk(lbl, patternEval(prog.data(), (int)prog.size(), c), cs.want, 1e-6);
    }
    struct Bad { const char* expr; const char* why; };
    const Bad bads[] = {
        {"scatter:pts(u)",      "2-D scatter called with 1 argument"},
        {"scatter:line(u, v)",  "1-D scatter called with 2 arguments"},
        {"scatter:nope(u)",     "unknown scatter name"},
        {"scatter:pts",         "scatter referenced without a call"},
        {"scatter(u)",          "bare `scatter` with no name"},
        {"grid:pts(u, v)",      "a scatter name is NOT visible in the grid namespace"},
    };
    for (const Bad& bd : bads) {
        std::vector<PatNode> prog; std::string perr;
        if (compilePatternExpr(bd.expr, prog, perr, false, nullptr, &scope)) {
            std::printf("[checkscatter] `%s` compiled but should be rejected (%s)  BAD\n", bd.expr, bd.why);
            ok = false;
        }
    }
    // With no table scope at all (a load-time constant site), `scatter:` must be refused.
    {
        std::vector<PatNode> prog; std::string perr;
        if (compilePatternExpr("scatter:pts(u, v)", prog, perr, false, nullptr, nullptr)) {
            std::printf("[checkscatter] `scatter:pts(u,v)` compiled with no table scope  BAD\n");
            ok = false;
        }
    }

    std::printf("[checkscatter] worst absolute error = %.3g\n", worst);
    std::printf("[checkscatter] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Deterministic self-test for NAMED-INPUT BINDING BY SUBSTITUTION (src/pattern.h;
// ROADMAP_records.md §3.2/§3.3). No scene file and no renderer — it pins the algebraic
// properties the whole material-application feature rests on:
//   (a) substitution is a pure SPLICE — binding an input to an expression gives the same
//       number as textually inlining that expression, for every input;
//   (b) it is SIMULTANEOUS, not sequential: `(u=v, v=u)` swaps the two inputs instead of
//       collapsing both onto one, which a naive left-to-right rewrite would get wrong;
//   (c) it is IDENTITY when nothing is bound (the additive-superset guarantee: a material
//       nobody applies must be bit-identical to before);
//   (d) introspection agrees with the program — patternCollectVars finds exactly the
//       inputs present, and varName/varOp round-trip;
//   (e) `a` (albedo) parses only where a material can resolve it, and binding it to a
//       constant is what turns a symbolic program into a concrete one.
static int checkBind() {
    using namespace pattern_detail;
    bool ok = true;
    auto chk = [&](const char* what, bool cond) {
        if (!cond) { std::printf("[checkbind] %-52s BAD\n", what); ok = false; }
    };
    auto compile = [&](const char* src, std::vector<PatNode>& out, bool allowA) {
        std::string e;
        bool good = compilePatternExpr(src, out, e, /*allowT=*/false, nullptr, nullptr, allowA);
        if (!good) std::printf("[checkbind] compile `%s` FAILED: %s\n", src, e.c_str());
        return good;
    };
    // A context with distinct, non-degenerate values so an accidental swap is visible.
    PatCtx c{};
    c.x = 0.37; c.y = -0.81; c.z = 1.23; c.u = 0.19; c.v = 0.64;
    c.nx = 0.0; c.ny = 1.0; c.nz = 0.0; c.r = 0.5; c.f = 0.25;
    auto ev = [&](const std::vector<PatNode>& p) {
        return patternEval(p.data(), (int)p.size(), c);
    };

    // (a) splice == textual inlining, for every bindable input.
    struct { const char* var; const char* arg; } cases[] = {
        {"u", "0.5*v+0.25"}, {"v", "x*2"},   {"x", "sin(u)"},
        {"y", "z-1"},        {"z", "r*3"},   {"r", "abs(y)"},
        {"f", "u*v"},        {"nx", "0.5"},  {"ny", "nz+1"}, {"nz", "0.125"},
    };
    for (auto& cs : cases) {
        std::string body = std::string("2.0*") + cs.var + " + sin(" + cs.var + ") + 1.5";
        std::string inl  = std::string("2.0*(") + cs.arg + ") + sin(" + cs.arg + ") + 1.5";
        std::vector<PatNode> prog, want, arg;
        if (!compile(body.c_str(), prog, false) || !compile(inl.c_str(), want, false) ||
            !compile(cs.arg, arg, false)) { ok = false; continue; }
        PatOp var;
        chk("varOp resolves the input name", varOp(cs.var, var));
        chk("varName round-trips", varName(var) && !std::strcmp(varName(var), cs.var));
        std::vector<PatBind> b{{var, arg}};
        double got = ev(patternSubstitute(prog, b)), wanted = ev(want);
        if (std::fabs(got - wanted) > 1e-12) {
            std::printf("[checkbind] bind %-3s <- %-12s got %.12g want %.12g  BAD\n",
                        cs.var, cs.arg, got, wanted);
            ok = false;
        }
    }

    // (b) SIMULTANEOUS: `u*10 + v` with (u=v, v=u) must become `v*10 + u`, NOT `u*10+u`
    //     (sequential rewriting) and NOT `v*10+v`.
    {
        std::vector<PatNode> prog, swapped, uOnly, vOnly, au, av;
        if (compile("u*10 + v", prog, false) && compile("v*10 + u", swapped, false) &&
            compile("u*10 + u", uOnly, false) && compile("v*10 + v", vOnly, false) &&
            compile("u", au, false) && compile("v", av, false)) {
            std::vector<PatBind> b{{PatOp::VarU, av}, {PatOp::VarV, au}};
            double got = ev(patternSubstitute(prog, b));
            chk("simultaneous bind swaps u and v", std::fabs(got - ev(swapped)) < 1e-12);
            chk("swap is not a sequential u-collapse", std::fabs(got - ev(uOnly)) > 1e-9);
            chk("swap is not a sequential v-collapse", std::fabs(got - ev(vOnly)) > 1e-9);
        } else ok = false;
    }

    // (c) identity when nothing binds — same nodes, not merely the same value.
    {
        std::vector<PatNode> prog, arg;
        if (compile("u*2 + sin(v)", prog, false) && compile("9.0", arg, false)) {
            std::vector<PatBind> none;
            std::vector<PatBind> unrelated{{PatOp::VarZ, arg}};   // z does not appear
            auto a = patternSubstitute(prog, none), b2 = patternSubstitute(prog, unrelated);
            chk("empty bind list is identity", a.size() == prog.size());
            chk("binding an absent input is identity", b2.size() == prog.size());
            chk("identity preserves the value", std::fabs(ev(b2) - ev(prog)) < 1e-15);
        } else ok = false;
    }

    // (d) introspection matches the program.
    {
        std::vector<PatNode> prog;
        if (compile("u*v + u - 3", prog, false)) {
            std::vector<PatOp> vars;
            patternCollectVars(prog, vars);
            chk("collectVars finds exactly {u, v}", vars.size() == 2);
            chk("collectVars is in order of first use",
                vars.size() == 2 && vars[0] == PatOp::VarU && vars[1] == PatOp::VarV);
            chk("collectVars dedupes a repeated input", !vars.empty() && vars[0] == PatOp::VarU);
            chk("usesVar agrees (present)", patternUsesVar(prog, PatOp::VarU));
            chk("usesVar agrees (absent)",  !patternUsesVar(prog, PatOp::VarZ));
        } else ok = false;
    }

    // (e) `a` is scoped, and binding it to a constant concretises the program.
    {
        std::vector<PatNode> prog, unusedProg;
        std::string e;
        chk("`a` is rejected where no material can resolve it",
            !compilePatternExpr("0.5*a", unusedProg, e, false, nullptr, nullptr, /*allowA=*/false));
        if (compile("0.5*a", prog, /*allowA=*/true)) {
            std::vector<PatOp> vars;
            patternCollectVars(prog, vars);
            chk("`a` shows up as a free input", vars.size() == 1 && vars[0] == PatOp::VarA);
            PatNode k; k.op = PatOp::Const; k.a = 0.8;
            std::vector<PatBind> b{{PatOp::VarA, {k}}};
            auto bound = patternSubstitute(prog, b);
            std::vector<PatOp> after;
            patternCollectVars(bound, after);
            chk("binding `a` leaves no free inputs", after.empty());
            chk("bound `a` evaluates to 0.5*0.8", std::fabs(ev(bound) - 0.4) < 1e-12);
        } else ok = false;
    }

    std::printf("[checkbind] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Deterministic self-test for PER-PROPERTY ACCESS (`MATERIAL.slot` / `MATERIAL.slot(args)`;
// ROADMAP_records.md §3.2). Unlike checkBind, which is pure algebra over pattern programs,
// this one has to run the LOADER — the whole point of the feature is that a value site
// resolves a reference to another material's slot, so the property under test is a property
// of loading, not of substitution. Scenes are built in memory and compared against
// hand-written TWINS, so every assert is "the reference produced exactly what writing it out
// by hand produces" rather than a hard-coded number that could drift with the defaults.
static int checkProp() {
    bool ok = true;
    auto chk = [&](const char* what, bool cond) {
        if (!cond) { std::printf("[checkprop] %-56s BAD\n", what); ok = false; }
    };
    // Load a scene fragment. Every fragment gets the same trivial camera + light so the
    // loader's renderability checks are satisfied; only the materials differ.
    auto loadMats = [&](const char* body, ftsl::Loaded& L) -> bool {
        std::string src =
            "scene { units meters }\n"
            + std::string(body) + "\n"
            "quad { origin 0 0 0  u 1 0 0  v 0 1 0  material probe }\n"
            "light area { origin 0 0.99 0.1  u 1 0 0  v 0 0 0.4  normal 0 -1 0  spd preset:bb6500 }\n"
            "camera \"c\" { eye 0.5 0.5 2  look_at 0.5 0.5 0  up 0 1 0  fov_y 32  film { res 8 8 } }\n";
        std::string e;
        if (!ftsl::loadSource(src, "<checkprop>", L, e)) {
            std::printf("[checkprop] load FAILED: %s\n", e.c_str());
            return false;
        }
        return true;
    };
    // A scene that must NOT load, and whose error must mention `needle`.
    auto mustReject = [&](const char* what, const char* body, const char* needle) {
        ftsl::Loaded L;
        std::string src =
            "scene { units meters }\n" + std::string(body) + "\n"
            "quad { origin 0 0 0  u 1 0 0  v 0 1 0  material probe }\n"
            "light area { origin 0 0.99 0.1  u 1 0 0  v 0 0 0.4  normal 0 -1 0  spd preset:bb6500 }\n"
            "camera \"c\" { eye 0.5 0.5 2  look_at 0.5 0.5 0  up 0 1 0  fov_y 32  film { res 8 8 } }\n";
        std::string e;
        bool loaded = ftsl::loadSource(src, "<checkprop>", L, e);
        if (loaded)                                   { chk(what, false); return; }
        if (e.find(needle) == std::string::npos) {
            std::printf("[checkprop] %-56s BAD (error was: %s)\n", what, e.c_str());
            ok = false;
        }
    };
    // Find a material by name is not possible post-load (names are not kept on Material),
    // so the fragments below always put the material under test LAST among the declared
    // ones and reach it through the probe. Instead of guessing indices, compare the two
    // scenes' probe materials: `probe` is always the material the quad uses.
    auto probeOf = [&](ftsl::Loaded& L) -> const Material& {
        // The quad's two triangles carry the resolved material index.
        int mi = L.scene.tris.empty() ? 0 : L.scene.tris[0].matId;
        return L.scene.mats[mi];
    };
    // Evaluate a material's reflect slot as (base spectrum at 550nm) * (pattern at ctx).
    PatCtx c{};
    c.x = 0.37; c.y = -0.81; c.z = 1.23; c.u = 0.19; c.v = 0.64;
    c.nx = 0.0; c.ny = 1.0; c.nz = 0.0; c.r = 0.5; c.f = 0.25;
    auto reflectAt = [&](ftsl::Loaded& L, const Material& m) {
        double base = m.reflect(550.0);
        if (m.reflectPat >= 0 && m.reflectPat < (int)L.scene.patterns.size()) {
            const auto& p = L.scene.patterns[m.reflectPat].nodes;
            base *= patternEval(p.data(), (int)p.size(), c);
        }
        return base;
    };
    // The core assert shape: two scenes whose probe materials must be indistinguishable.
    auto sameReflect = [&](const char* what, const char* refBody, const char* twinBody) {
        ftsl::Loaded A, B;
        if (!loadMats(refBody, A) || !loadMats(twinBody, B)) { chk(what, false); return; }
        double a = reflectAt(A, probeOf(A)), b = reflectAt(B, probeOf(B));
        if (std::fabs(a - b) > 1e-12) {
            std::printf("[checkprop] %-56s BAD (%.12g vs %.12g)\n", what, a, b);
            ok = false;
        }
    };

    // (a) A bare reference reproduces the source slot — BOTH halves of it, the flat-1.0
    //     base a lone `pattern:` leaves behind AND the pattern itself.
    sameReflect("bare ref reproduces a pattern-driven reflect slot",
        "pattern \"p\" { expr \"0.05+0.9*u\" }\n"
        "material \"src\" { type diffuse  reflect pattern:p }\n"
        "material \"probe\" { type diffuse  reflect src.reflect }",
        "pattern \"p\" { expr \"0.05+0.9*u\" }\n"
        "material \"probe\" { type diffuse  reflect pattern:p }");

    // (b) A reference to a plain-spectrum slot carries the spectrum and no pattern.
    sameReflect("bare ref reproduces a constant reflect slot",
        "material \"src\" { type diffuse  reflect 0.37 }\n"
        "material \"probe\" { type diffuse  reflect src.reflect }",
        "material \"probe\" { type diffuse  reflect 0.37 }");

    // (c) Rebinding inside the reference is the SAME machinery §3.3 uses at a use site:
    //     `src.reflect(u=v)` must equal a hand-written program with u replaced by v.
    sameReflect("ref rebinding u=v == the hand-written twin",
        "pattern \"p\" { expr \"0.05+0.9*u\" }\n"
        "material \"src\" { type diffuse  reflect pattern:p }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(u=v) }",
        "pattern \"q\" { expr \"0.05+0.9*v\" }\n"
        "material \"probe\" { type diffuse  reflect pattern:q }");

    // (d) An unbound `a` resolves against the SOURCE material's albedo_default, not the
    //     consumer's and not the system 1.0 — the reference does not change whose default
    //     applies. Twin: the same program with `a` written out as the source's 0.4.
    sameReflect("unbound `a` falls back to the SOURCE albedo_default",
        "pattern \"p\" { expr \"a*(0.05+0.9*u)\" }\n"
        "material \"src\" { type diffuse  reflect pattern:p  albedo_default 0.4 }\n"
        "material \"probe\" { type diffuse  reflect src.reflect }",
        "pattern \"q\" { expr \"0.4*(0.05+0.9*u)\" }\n"
        "material \"probe\" { type diffuse  reflect pattern:q }");
    sameReflect("`a` bound at the reference overrides that default",
        "pattern \"p\" { expr \"a*(0.05+0.9*u)\" }\n"
        "material \"src\" { type diffuse  reflect pattern:p  albedo_default 0.4 }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(a=1) }",
        "pattern \"q\" { expr \"1*(0.05+0.9*u)\" }\n"
        "material \"probe\" { type diffuse  reflect pattern:q }");

    // (e) A reference COMPOSES with the consumer's own `_map` instead of clobbering it.
    //     Both spellings mean "a per-hit multiplier on the slot", so the answer is their
    //     product — the case a naive assignment would silently get wrong in one direction
    //     or the other depending on statement order.
    sameReflect("ref composes with the consumer's own reflect_map",
        "pattern \"p\" { expr \"0.05+0.9*u\" }\n"
        "pattern \"h\" { expr \"0.5\" }\n"
        "material \"src\" { type diffuse  reflect pattern:p }\n"
        "material \"probe\" { type diffuse  reflect src.reflect  reflect_map pattern:h }",
        "pattern \"q\" { expr \"(0.05+0.9*u)*0.5\" }\n"
        "material \"probe\" { type diffuse  reflect pattern:q }");

    // (f) Cross-slot references are legal as long as the TYPE matches: transmit is a
    //     spectral slot like reflect, so reading one into the other is fine.
    sameReflect("cross-slot spectral ref (transmit -> reflect)",
        "material \"src\" { type translucent  reflect 0.1  transmit 0.62 }\n"
        "material \"probe\" { type diffuse  reflect src.transmit }",
        "material \"probe\" { type diffuse  reflect 0.62 }");

    // (g) Scalar properties, read back through the scalar ladder
    //     (bindScalarPattern -> bindScalarTexture -> dblParam).
    {
        ftsl::Loaded A, B;
        if (loadMats("material \"src\" { type glossy  reflect 0.6  roughness 0.35 }\n"
                     "material \"probe\" { type glossy  reflect 0.6  roughness src.roughness }", A) &&
            loadMats("material \"probe\" { type glossy  reflect 0.6  roughness 0.35 }", B)) {
            chk("scalar property read back == the authored literal",
                std::fabs(probeOf(A).roughness - probeOf(B).roughness) < 1e-12);
        } else ok = false;
    }
    {
        ftsl::Loaded A;
        if (loadMats("material \"src\" { type thinfilm  film_ior 1.42  film_thickness 275 }\n"
                     "material \"probe\" { type thinfilm  film_ior src.film_ior  "
                     "film_thickness src.film_thickness }", A)) {
            chk("film_ior read back", std::fabs(probeOf(A).filmIor - 1.42) < 1e-12);
            chk("film_thickness read back", std::fabs(probeOf(A).filmThickness - 275.0) < 1e-12);
        } else ok = false;
    }
    // A pattern-driven scalar property carries its pattern through the reference.
    {
        ftsl::Loaded A, B;
        if (loadMats("pattern \"p\" { expr \"0.2+0.5*u\" }\n"
                     "material \"src\" { type glossy  reflect 0.6  roughness pattern:p }\n"
                     "material \"probe\" { type glossy  reflect 0.6  roughness src.roughness }", A) &&
            loadMats("pattern \"p\" { expr \"0.2+0.5*u\" }\n"
                     "material \"probe\" { type glossy  reflect 0.6  roughness pattern:p }", B)) {
            const Material& ma = probeOf(A);
            const Material& mb = probeOf(B);
            chk("pattern-driven scalar property keeps its pattern",
                ma.roughnessPat >= 0 && mb.roughnessPat >= 0);
            if (ma.roughnessPat >= 0 && mb.roughnessPat >= 0) {
                const auto& pa = A.scene.patterns[ma.roughnessPat].nodes;
                const auto& pb = B.scene.patterns[mb.roughnessPat].nodes;
                chk("...and evaluates identically",
                    std::fabs(patternEval(pa.data(), (int)pa.size(), c) -
                              patternEval(pb.data(), (int)pb.size(), c)) < 1e-12);
            }
        } else ok = false;
    }

    // (h) A no-op reference is SHARED, not duplicated: `src.reflect` written twice must
    //     not grow the material table, since applyMaterial memoises on (material, args).
    {
        ftsl::Loaded A, B;
        if (loadMats("pattern \"p\" { expr \"0.05+0.9*u\" }\n"
                     "material \"src\" { type diffuse  reflect pattern:p }\n"
                     "material \"probe\" { type diffuse  reflect src.reflect(u=v) }\n"
                     "material \"probe2\" { type diffuse  reflect src.reflect(u=v) }", A) &&
            loadMats("pattern \"p\" { expr \"0.05+0.9*u\" }\n"
                     "material \"src\" { type diffuse  reflect pattern:p }\n"
                     "material \"probe\" { type diffuse  reflect src.reflect(u=v) }", B)) {
            // One extra declared material, but the APPLIED material and its substituted
            // pattern are shared, so the pattern table must be the same size.
            chk("identical references share one applied material + pattern",
                A.scene.patterns.size() == B.scene.patterns.size() &&
                A.scene.mats.size() == B.scene.mats.size() + 1);
        } else ok = false;
    }

    // (i) The type system is real, in both directions, and unknown properties are named.
    mustReject("a scalar property is refused at a spectral slot",
        "material \"src\" { type glossy  reflect 0.6  roughness 0.35 }\n"
        "material \"probe\" { type diffuse  reflect src.roughness }", "scalar property");
    mustReject("a spectral property is refused at a scalar slot",
        "material \"src\" { type diffuse  reflect 0.6 }\n"
        "material \"probe\" { type glossy  reflect 0.6  roughness src.reflect }",
        "spectral property");
    mustReject("an unknown property names the material and lists the slots",
        "material \"src\" { type diffuse  reflect 0.6 }\n"
        "material \"probe\" { type diffuse  reflect src.colour }", "has no property");
    // A pattern-carrying source at a slot that cannot apply one is refused, not flattened.
    mustReject("a pattern-carrying property is refused where it cannot be applied",
        "pattern \"p\" { expr \"0.05+0.9*u\" }\n"
        "material \"src\" { type diffuse  reflect pattern:p }\n"
        "material \"probe\" { type dielectric  ior src.reflect }", "per-hit pattern");
    // An unbalanced argument list is an error rather than a run-on, which is the whole
    // point of lexing the group BALANCED (see the v0.88.0 grammar change).
    mustReject("an unbalanced argument list is an error, not a silent run-on",
        "material \"src\" { type diffuse  reflect 0.6 }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(a=1 }", "unbalanced");
    // A texture-bound slot is an IMAGE sampled at the hit UV; a property reference carries
    // a spectrum plus a per-hit pattern and has nowhere to put a texture binding. Refusing
    // it beats handing the consumer the fallback constant, which looks like it worked.
    mustReject("a texture-bound property is refused rather than flattened",
        "texture \"t\" { file scenes/graychecker.ppm  encoding linear }\n"
        "material \"src\" { type diffuse  reflect texture:t }\n"
        "material \"probe\" { type diffuse  reflect src.reflect }", "bound to a texture");
    // A material application error inside the reference surfaces as itself, not swallowed.
    mustReject("a bad rebinding inside a reference reports the binding error",
        "pattern \"p\" { expr \"0.05+0.9*u\" }\n"
        "material \"src\" { type diffuse  reflect pattern:p }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(bogus=1) }",
        "not a bindable input");

    // (j) Records keep priority over materials on a name clash, so an existing scene's
    //     `R.chan` cannot change meaning just because a material was named `R`.
    {
        ftsl::Loaded A;
        if (loadMats("R = range 0-1 [\n  k  0.2 0.8\n]\n"
                     "material \"R\" { type diffuse  reflect 0.9 }\n"
                     "material \"probe\" { type glossy  reflect 0.6  roughness R.k(0.0) }", A)) {
            chk("a record wins a name clash with a material",
                std::fabs(probeOf(A).roughness - 0.2) < 1e-9);
        } else ok = false;
    }

    std::printf("[checkprop] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Deterministic self-test for INLINE ARRAY LITERAL sample calls — the axis tuple on
// `[0 1](…)` (TODO "DECISION — color-vector / array syntax", the increment-2 `Deferred:`
// clause and the `ADDENDUM — call = sample`).
//
// The claim under test is that an array literal's axis tuple takes DRIVERS, and that
// leaving an axis open for the material's *user* is spelled by naming the one free input
// (`[0 1](a)`) and binding it where the material is used (`material mat(a=u)`) — not by a
// separate formal namespace. That claim is only worth anything if it is an IDENTITY, so
// every positive case here compares two independently authored scenes whose probe
// materials must be indistinguishable, exactly like checkProp; nothing is hard-coded.
//
// The refusals matter just as much: the one form the design text left open —
// `formal=driver` *inside a literal's own tuple* — is refused rather than approximated,
// because an inline literal's axes are anonymous and positional, so there is no formal to
// bind and accepting it would have to invent a per-material default that two literals in
// one material could contradict. Each refusal also pins its MESSAGE, because the message
// is the whole value of a refusal.
static int checkArray() {
    bool ok = true;
    auto chk = [&](const char* what, bool cond) {
        if (!cond) { std::printf("[checkarray] %-58s BAD\n", what); ok = false; }
    };
    // `quadMat` is the geometry field's material *reference*, so a case can exercise the
    // OTHER use site a bind can appear at — `material probe(a=u)` on the quad itself.
    auto wrap = [](const char* body, const char* quadMat = "probe") {
        return "scene { units meters }\n" + std::string(body) + "\n"
               "quad { origin 0 0 0  u 1 0 0  v 0 1 0  material " + quadMat + " }\n"
               "light area { origin 0 0.99 0.1  u 1 0 0  v 0 0 0.4  normal 0 -1 0  spd preset:bb6500 }\n"
               "camera \"c\" { eye 0.5 0.5 2  look_at 0.5 0.5 0  up 0 1 0  fov_y 32  film { res 8 8 } }\n";
    };
    auto loadMats = [&](const char* body, ftsl::Loaded& L, const char* quadMat = "probe") -> bool {
        std::string e;
        if (!ftsl::loadSource(wrap(body, quadMat), "<checkarray>", L, e)) {
            std::printf("[checkarray] load FAILED: %s\n", e.c_str());
            return false;
        }
        return true;
    };
    auto mustReject = [&](const char* what, const char* body, const char* needle) {
        ftsl::Loaded L; std::string e;
        if (ftsl::loadSource(wrap(body), "<checkarray>", L, e)) { chk(what, false); return; }
        if (e.find(needle) == std::string::npos) {
            std::printf("[checkarray] %-58s BAD (error was: %s)\n", what, e.c_str());
            ok = false;
        }
    };
    auto probeOf = [&](ftsl::Loaded& L) -> const Material& {
        int mi = L.scene.tris.empty() ? 0 : L.scene.tris[0].matId;
        return L.scene.mats[mi];
    };
    // Five probe points, so an identity that only holds on the diagonal (u == v, the
    // classic way a rebind test passes for the wrong reason) cannot slip through.
    PatCtx pts[5]{};
    const double us[5] = {0.07, 0.31, 0.50, 0.83, 0.96};
    const double vs[5] = {0.62, 0.11, 0.50, 0.24, 0.78};
    for (int k = 0; k < 5; ++k) {
        pts[k].x = 0.37; pts[k].y = -0.81; pts[k].z = 1.23;
        pts[k].nx = 0.0; pts[k].ny = 1.0; pts[k].nz = 0.0; pts[k].r = 0.5; pts[k].f = 0.25;
        pts[k].u = us[k]; pts[k].v = vs[k];
    }
    // The grid/scatter POOL has to be bound into the context, or `PatOp::Grid` bails out
    // and returns 0.0 — and since a desugared array literal is nothing BUT a grid sample,
    // every comparison here would then be 0 == 0 and pass vacuously. That is why the
    // `varies` assertions below exist at all.
    auto reflectAt = [&](ftsl::Loaded& L, const Material& m, const PatCtx& base_c) {
        PatCtx c = base_c;
        bindPatScene(c, L.scene);
        double base = m.reflect(550.0);
        if (m.reflectPat >= 0 && m.reflectPat < (int)L.scene.patterns.size()) {
            const auto& p = L.scene.patterns[m.reflectPat].nodes;
            base *= patternEval(p.data(), (int)p.size(), c);
        }
        return base;
    };
    // `tol` defaults to the bit-identity the rebind cases demand: those twins are the SAME
    // grid sampled two ways, so any difference at all is a real one. A twin that spells the
    // coordinate ARITHMETICALLY instead (section h) is a different claim — grid samples are
    // stored as float32 while an expression evaluates in double, so the two agree only to
    // float precision (~1e-8 here), and demanding more would be pinning the storage format
    // rather than the semantics.
    auto sameReflect = [&](const char* what, const char* refBody, const char* twinBody,
                           double tol = 1e-12) {
        ftsl::Loaded A, B;
        if (!loadMats(refBody, A) || !loadMats(twinBody, B)) { chk(what, false); return; }
        for (int k = 0; k < 5; ++k) {
            double a = reflectAt(A, probeOf(A), pts[k]), b = reflectAt(B, probeOf(B), pts[k]);
            if (std::fabs(a - b) > tol) {
                std::printf("[checkarray] %-58s BAD (pt %d: %.12g vs %.12g)\n", what, k, a, b);
                ok = false; return;
            }
        }
    };
    // Also assert the pair is not accidentally CONSTANT — a literal that failed to sample
    // would compare equal to another that failed the same way.
    auto varies = [&](const char* what, const char* body) {
        ftsl::Loaded A;
        if (!loadMats(body, A)) { chk(what, false); return; }
        double lo = 1e300, hi = -1e300;
        for (int k = 0; k < 5; ++k) {
            double r = reflectAt(A, probeOf(A), pts[k]);
            lo = std::fmin(lo, r); hi = std::fmax(hi, r);
        }
        if (hi - lo <= 1e-6) {
            std::printf("[checkarray] %-58s BAD (constant: lo=%.12g hi=%.12g, pat=%d)\n",
                        what, lo, hi, probeOf(A).reflectPat);
            ok = false;
        }
    };

    // (a) The baseline the rest is measured against: a literal really is the grid+pattern
    //     it desugars to, and it really does vary with its driver.
    sameReflect("`[0 1](u)` == the hand-written grid + pattern twin",
        "material \"probe\" { type diffuse  reflect [0 1](u) }",
        "grid \"g\" { shape 2  lo 0  hi 1  data { 0 1 } }\n"
        "pattern \"p\" { expr \"grid:g(u)\" }\n"
        "material \"probe\" { type diffuse  reflect pattern:p }");
    varies("...and the sampled value actually tracks u",
        "material \"probe\" { type diffuse  reflect [0 1](u) }");

    // (b) THE ITEM. Naming the free input `a` leaves the axis open; binding it at the use
    //     site must reproduce spending it inline. Both spellings of the bind — the
    //     keyword form and the positional one — and both use sites (a geometry `material`
    //     field and a property reference) have to agree with the inline literal.
    sameReflect("formal `[0 1](a)` + keyword bind `(a=u)` == `[0 1](u)`",
        "material \"src\" { type diffuse  reflect [0 1](a) }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(a=u) }",
        "material \"probe\" { type diffuse  reflect [0 1](u) }");
    sameReflect("formal `[0 1](a)` + POSITIONAL bind `(u)` == the keyword form",
        "material \"src\" { type diffuse  reflect [0 1](a) }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(u) }",
        "material \"src\" { type diffuse  reflect [0 1](a) }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(a=u) }");
    {
        // The other use site a bind can appear at: the geometry field itself,
        // `material probe(a=u)`. It must land on the same answer as the inline literal.
        ftsl::Loaded A, B;
        if (loadMats("material \"probe\" { type diffuse  reflect [0 1](a) }", A, "probe(a=u)") &&
            loadMats("material \"probe\" { type diffuse  reflect [0 1](u) }", B)) {
            bool same = true;
            for (int k = 0; k < 5; ++k)
                if (std::fabs(reflectAt(A, probeOf(A), pts[k]) -
                              reflectAt(B, probeOf(B), pts[k])) > 1e-12) same = false;
            chk("binding a formal at a geometry `material` field agrees too", same);
        } else ok = false;
    }
    varies("a bound formal axis actually varies with its driver",
        "material \"src\" { type diffuse  reflect [0 1](a) }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(a=u) }");

    // (c) The driver is an ordinary expression, not just a variable name — so the formal
    //     route composes with arithmetic exactly like the inline route does.
    sameReflect("a formal bound to an EXPRESSION == that expression written inline",
        "material \"src\" { type diffuse  reflect [0 1](a) }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(a=0.25+0.5*v) }",
        "material \"probe\" { type diffuse  reflect [0 1](0.25+0.5*v) }");

    // (d) Multi-axis: the formals of a 2-D literal ARE the driver names in its own tuple
    //     (the design text's `(u=a, v=x)` case), so a simultaneous swap must transpose the
    //     lookup — and NOT collapse the way a sequential u->v, v->u rebind would.
    sameReflect("2-D literal, simultaneous swap `(u=v, v=u)` == the transposed literal",
        "material \"src\" { type diffuse  reflect [[0 0.3][0.6 1]](u,v) }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(u=v, v=u) }",
        "material \"probe\" { type diffuse  reflect [[0 0.3][0.6 1]](v,u) }");
    {
        // Negative twin for the same case: the swap must NOT equal the unswapped literal,
        // or the assert above would pass for a rebind that silently did nothing.
        ftsl::Loaded A, B;
        const char* swapped =
            "material \"src\" { type diffuse  reflect [[0 0.3][0.6 1]](u,v) }\n"
            "material \"probe\" { type diffuse  reflect src.reflect(u=v, v=u) }";
        const char* plain = "material \"probe\" { type diffuse  reflect [[0 0.3][0.6 1]](u,v) }";
        if (loadMats(swapped, A) && loadMats(plain, B)) {
            bool differs = false;
            for (int k = 0; k < 5; ++k)
                if (std::fabs(reflectAt(A, probeOf(A), pts[k]) -
                              reflectAt(B, probeOf(B), pts[k])) > 1e-9) differs = true;
            chk("...and the swap is not a silent no-op", differs);
        } else ok = false;
    }

    // (e) THE PINNED REFUSAL. `formal=driver` inside a literal's own tuple has no formal
    //     to bind. The message must carry BOTH escapes, because an author who wrote it
    //     meant one of exactly two things.
    mustReject("`[0 1](a=u)` is refused, not approximated",
        "material \"probe\" { type diffuse  reflect [0 1](a=u) }", "no formal to bind");
    mustReject("...and the refusal offers the spend-it-here spelling",
        "material \"probe\" { type diffuse  reflect [0 1](a=u) }", "(u)`");
    mustReject("...and the leave-it-free spelling",
        "material \"probe\" { type diffuse  reflect [0 1](a=u) }", "material mat(a=u)");
    mustReject("a keyword arg is refused after a positional one too",
        "material \"probe\" { type diffuse  reflect [[0 0.3][0.6 1]](u, b=v) }",
        "no formal to bind");

    // (f) Errors name what the AUTHOR wrote. `__arrN` is a symbol they never chose and
    //     cannot search the file for, so no message about a literal may mention it.
    {
        ftsl::Loaded L; std::string e;
        ftsl::loadSource(wrap("material \"probe\" { type diffuse  reflect [0 1](nope) }"),
                         "<checkarray>", L, e);
        chk("a bad coordinate reports the author's site, not `__arrN`",
            e.find("__arr") == std::string::npos &&
            e.find("inline array literal's sample call") != std::string::npos &&
            e.find("unknown identifier 'nope'") != std::string::npos);
    }

    // (g) The remaining shape errors, each naming the thing that is wrong.
    mustReject("an unsaturated literal names the formal-axis escape",
        "material \"probe\" { type diffuse  reflect [0 1] }", "`[0 1](a)`");
    mustReject("an empty axis is named by index",
        "material \"probe\" { type diffuse  reflect [[0 0.3][0.6 1]](u,) }", "axis 1");
    mustReject("coordinate count is checked against the nesting",
        "material \"probe\" { type diffuse  reflect [0 1](u,v) }", "one per nesting level");
    mustReject("an unbindable formal name is named at the use site",
        "material \"src\" { type diffuse  reflect [0 1](a) }\n"
        "material \"probe\" { type diffuse  reflect src.reflect(b=u) }",
        "not a bindable input");

    // (h) COMPOSITION — a coordinate may itself be a sampled value, so a literal can be
    //     written directly into another's sample call. Every identity here is checked
    //     against a twin whose coordinate is spelled out ARITHMETICALLY, which is the
    //     strongest available statement: the composed form is not merely self-consistent,
    //     it agrees with what the inner grid's interpolation is defined to mean.
    //     (A 2-sample grid over lo=0 hi=1 interpolates linearly, so `[0.5 1](u)` IS
    //     `0.5+0.5*u` — that equivalence is what makes these twins non-circular.)
    sameReflect("`[0 1]([0.2 0.8](u))` == the composed coordinate written inline",
        "material \"probe\" { type diffuse  reflect [0 1]([0.2 0.8](u)) }",
        "material \"probe\" { type diffuse  reflect [0 1](0.2+0.6*u) }", 1e-6);
    varies("...and a composed literal actually tracks its innermost driver",
        "material \"probe\" { type diffuse  reflect [0 1]([0.2 0.8](u)) }");
    // A NON-identity outer array, so the case cannot pass by the outer grid being a
    // no-op that returns its own coordinate: here the outer is a 3-sample tent, and
    // sampling it over the inner's [0.5,1] half gives the falling edge, `1-u`.
    sameReflect("a non-identity outer array composes correctly (`[0 1 0]([0.5 1](u))`)",
        "material \"probe\" { type diffuse  reflect [0 1 0]([0.5 1](u)) }",
        "material \"probe\" { type diffuse  reflect [1 0](u) }", 1e-6);
    sameReflect("composition nests to depth 3",
        "material \"probe\" { type diffuse  reflect [0 1]([0 1]([0.2 0.8](u))) }",
        "material \"probe\" { type diffuse  reflect [0.2 0.8](u) }", 1e-6);
    sameReflect("a composed literal is one AXIS of a multi-axis call, not the whole call",
        "material \"probe\" { type diffuse  reflect [[0 0.3][0.6 1]]([0.5 1](u), v) }",
        "material \"probe\" { type diffuse  reflect [[0 0.3][0.6 1]](0.5+0.5*u, v) }", 1e-6);
    sameReflect("a composed literal is a TERM in a coordinate expression",
        "material \"probe\" { type diffuse  reflect [0 1](0.5*[0.5 1](u)+0.25) }",
        "material \"probe\" { type diffuse  reflect [0 1](0.5*(0.5+0.5*u)+0.25) }", 1e-6);
    // The refusals. A composed literal reaches the loader as raw TEXT inside one token
    // (the lexer cannot balance-check brackets it is deliberately holding together), so
    // the loader is the only place these can be diagnosed — and it must diagnose them
    // against the author's own source rather than let a mangled name reach the compiler.
    mustReject("a composed literal needs its own sample call",
        "material \"probe\" { type diffuse  reflect [0 1]([0.2 0.8]) }",
        "needs its own trailing call");
    mustReject("an unbalanced composed literal is named, not silently re-lexed",
        "material \"probe\" { type diffuse  reflect [0 1]([0.2 0.8 (u)) }",
        "unbalanced");
    mustReject("the composed literal's OWN arity is checked against its nesting",
        "material \"probe\" { type diffuse  reflect [0 1]([[0 1][2 3]](u)) }",
        "one per nesting level");
    mustReject("a literal glued to an identifier is a typo, not a composition",
        "material \"probe\" { type diffuse  reflect [0 1](x[0.2 0.8](u)) }",
        "stand on its own");
    {
        // Same rule as (f), one level deeper: the generated names multiply under
        // composition, so this is exactly where a leak would show up first.
        ftsl::Loaded L; std::string e;
        ftsl::loadSource(wrap("material \"probe\" { type diffuse  reflect [0 1]([0.2 0.8](nope)) }"),
                         "<checkarray>", L, e);
        chk("a bad coordinate INSIDE a composition still names the author's site",
            e.find("__arr") == std::string::npos &&
            e.find("unknown identifier 'nope'") != std::string::npos);
    }

    // (i) A NAMED table sampled directly at a value site — `reflect grid:ramp(u)`. This is
    //     the same statement as an inline literal written the other way round: an inline
    //     `[0 1](u)` desugars to precisely this expression wrapped in an anonymous pattern,
    //     so the two spellings must be INDISTINGUISHABLE. That is the claim under test, and
    //     it is why these pins live in `-checkarray` rather than a suite of their own.
    //     Only the *scoped* spelling is accepted: a bare `ramp(u)` at a value site already
    //     means "apply the material `ramp`", so `grid:` / `scatter:` is what keeps the
    //     meaning from depending on which namespace happens to hold the name.
    const std::string gramp = "grid \"g\" { shape 2  lo 0  hi 1  data { 0 1 } }\n";
    const std::string ghalf = "grid \"h\" { shape 2  lo 0  hi 1  data { 0.5 1 } }\n";
    const std::string gcall = gramp + "material \"probe\" { type diffuse  reflect grid:g(u) }";
    sameReflect("`reflect grid:g(u)` == the inline literal it desugars to",
        gcall.c_str(), "material \"probe\" { type diffuse  reflect [0 1](u) }");
    varies("...and it actually tracks its coordinate", gcall.c_str());
    sameReflect("a 2-D table call matches the 2-D literal",
        "grid \"g2\" { shape 2 2  lo 0  hi 1  data { 0 0.3  0.6 1 } }\n"
        "material \"probe\" { type diffuse  reflect grid:g2(u,v) }",
        "material \"probe\" { type diffuse  reflect [[0 0.3][0.6 1]](u,v) }");
    sameReflect("a `scatter:` call is accepted at a value site too",
        "scatter \"s\" { dim 1  power 2  data { 0 0   1 1 } }\n"
        "material \"probe\" { type diffuse  reflect scatter:s(u) }",
        "scatter \"s\" { dim 1  power 2  data { 0 0   1 1 } }\n"
        "pattern \"p\" { expr \"scatter:s(u)\" }\n"
        "material \"probe\" { type diffuse  reflect pattern:p }");
    // The deferral route works here for the same reason it works for a literal: `a` is an
    // ordinary coordinate that survives to the use site, where the bundle substitution
    // rebinds it. Nothing about it is special-cased for tables.
    {
        const std::string deferred = gramp +
            "material \"src\" { type diffuse  reflect grid:g(a) }\n"
            "material \"probe\" { type diffuse  reflect src.reflect(a=u) }";
        sameReflect("`grid:g(a)` + a use-site bind == spending the axis inline",
            deferred.c_str(), gcall.c_str());
    }
    // Composition runs BOTH directions — a literal inside a named table's call and a table
    // call inside a literal's — because both go through the one `desugarNestedLiterals`.
    // The second uses a non-identity outer table, so it cannot pass by the outer being a
    // no-op: `[0 1 0]` sampled over `h`'s [0.5,1] half is the tent's falling edge, `1-u`.
    {
        const std::string litIn = gramp +
            "material \"probe\" { type diffuse  reflect grid:g([0.5 1](u)) }";
        sameReflect("a literal composes INTO a named table's call", litIn.c_str(),
            "material \"probe\" { type diffuse  reflect [0 1](0.5+0.5*u) }", 1e-6);
        const std::string callIn = ghalf +
            "material \"probe\" { type diffuse  reflect [0 1 0](grid:h(u)) }";
        sameReflect("...and a named table's call composes INTO a literal", callIn.c_str(),
            "material \"probe\" { type diffuse  reflect [1 0](u) }", 1e-6);
    }
    {
        // A SCALAR slot: `roughness` takes the sampled table as its per-hit pattern, exactly
        // as the inline literal does. Compared by evaluating the two bound patterns, since
        // `reflectAt` only reaches the reflect slot.
        auto roughAt = [&](ftsl::Loaded& L, const PatCtx& base_c) {
            PatCtx c = base_c;
            bindPatScene(c, L.scene);
            int pat = probeOf(L).roughnessPat;
            if (pat < 0 || pat >= (int)L.scene.patterns.size()) return -1e300;
            const auto& p = L.scene.patterns[pat].nodes;
            return patternEval(p.data(), (int)p.size(), c);
        };
        ftsl::Loaded A, B;
        const std::string rc = gramp + "material \"probe\" { type glossy  roughness grid:g(u) }";
        if (loadMats(rc.c_str(), A) &&
            loadMats("material \"probe\" { type glossy  roughness [0 1](u) }", B)) {
            bool same = true, live = false;
            const double a0 = roughAt(A, pts[0]);
            for (int k = 0; k < 5; ++k) {
                double a = roughAt(A, pts[k]), b = roughAt(B, pts[k]);
                if (a < -1e299 || std::fabs(a - b) > 1e-12) same = false;
                if (k && std::fabs(a - a0) > 1e-6) live = true;
            }
            chk("a table call binds a SCALAR slot's pattern like the literal does", same);
            chk("...and that scalar pattern is not a constant", live);
        } else ok = false;
    }
    // The refusals. Two of them are the whole point of hooking four separate value sites
    // rather than one: a slot that cannot hold a per-hit value has to SAY so, instead of
    // reading `grid:g(u)` as the number zero or as an unrecognized spectrum expression.
    {
        const std::string iorCall = gramp +
            "material \"probe\" { type dielectric  ior grid:g(u) }";
        mustReject("a per-hit table is refused at a load-time SPECTRAL slot",
            iorCall.c_str(), "fixed at load time");
        const std::string filmCall = gramp +
            "material \"probe\" { type thinfilm  film_ior grid:g(u) }";
        mustReject("a per-hit table is refused at a load-time SCALAR slot",
            filmCall.c_str(), "fixed at load time");
        const std::string bare = gramp + "material \"probe\" { type diffuse  reflect grid:g }";
        mustReject("naming a table without sampling it is refused, not defaulted",
            bare.c_str(), "does not sample it");
        const std::string arity = gramp +
            "material \"probe\" { type diffuse  reflect grid:g(u,v) }";
        mustReject("the call's arity is checked against the table's own dimensionality",
            arity.c_str(), "expects 1 arg");
    }
    mustReject("an unknown table is named",
        "material \"probe\" { type diffuse  reflect grid:nope(u) }", "unknown grid");

    std::printf("[checkarray] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Deterministic distant-sun self-test (EmitterShape::Sun; src/scene.h addSunLight /
// sampleCone / inCone / geomWeight). No scene file, no renderer, no RNG-seeded image —
// it pins the four invariants the emitter's correctness rests on:
//   (a) the shared spot field reuse really does produce the cone solid angle:
//       spotOmega == 2*PI*(1 - cos theta) for every authored `angle`;
//   (b) EXPOSURE INVARIANCE — the authored `spd` is perpendicular irradiance, so the
//       stored radiance times the solid angle must reproduce it exactly, independent of
//       the angular diameter. This is the property that lets you widen `angle` to soften
//       a penumbra without re-grading the shot;
//   (c) sampleCone is uniform in SOLID ANGLE about its axis — every direction lands
//       inside the cone, the mean cosine matches (1+cos theta)/2, and the fraction inside
//       an inner sub-cone matches its solid-angle share. Forward emission (axis = beamDir)
//       and NEE (axis = -beamDir) both consume this, so a bias here breaks the B/R match;
//   (d) inCone agrees with sampleCone (every sampled NEE direction reads back as ON the
//       disc, and a direction just outside the rim reads back as off it) — the direct-view
//       miss term and the NEE estimator must see the same disc or the split double-counts.
static int checkSun() {
    double worst = 0.0;
    bool ok = true;
    auto chk = [&](const char* what, double got, double want, double tol) {
        double e = std::fabs(got - want);
        if (e > worst) worst = e;
        if (e > tol)
            std::printf("[checksun] %-42s got %.9f want %.9f  err=%.3g  BAD\n", what, got, want, e);
        return e <= tol;
    };

    // A flat unit "irradiance" spectrum makes the radiometry checks read directly:
    // stored radiance must come back as 1/Omega at every wavelength.
    Spectrum flat = [](double) { return 1.0; };
    const double toSun[3][3] = {{0.0, 1.0, 0.0}, {0.6, 0.5, 0.3}, {-0.2, 0.05, -1.0}};
    const double diamDeg[] = {0.53, 2.0, 8.0, 45.0, 120.0};

    for (int a = 0; a < 3; ++a) {
        for (double dd : diamDeg) {
            Scene sc;
            const double theta = 0.5 * dd * PI / 180.0;
            Vec3 aim = normalize(Vec3{toSun[a][0], toSun[a][1], toSun[a][2]});
            sc.addSunLight(aim, theta, flat, 1.0);
            const Emitter& e = sc.emitters.back();
            char lbl[96];

            // (a) the spot-field reuse must land exactly on the cone solid angle.
            const double omega = 2.0 * PI * (1.0 - std::cos(theta));
            std::snprintf(lbl, sizeof lbl, "spotOmega == cone SA (%.2f deg)", dd);
            ok &= chk(lbl, e.spotOmega, omega, 1e-12);

            // beamDir is the TRAVEL direction: exactly opposite the authored aim.
            std::snprintf(lbl, sizeof lbl, "beamDir == -toSun (%.2f deg)", dd);
            ok &= chk(lbl, dot(e.beamDir, aim), -1.0, 1e-12);

            // (b) radiance * Omega == the authored perpendicular irradiance, at any
            // angular diameter. This is the exposure-invariance guarantee.
            for (double lam : {380.0, 550.0, 780.0}) {
                std::snprintf(lbl, sizeof lbl, "L*Omega == E_perp @%.0fnm (%.2f deg)", lam, dd);
                ok &= chk(lbl, e.spdFn(lam) * e.spotOmega, 1.0, 1e-12);
            }

            // (c) sampleCone: uniform in solid angle about BOTH the forward axis
            // (beamDir) and the NEE axis (-beamDir). Stratified u1/u2 over the unit
            // square, so this is deterministic — no RNG, no flaky tolerance.
            const int NS = 200;                       // 200x200 = 40000 strata
            const double ci = std::cos(theta);
            const double inner = 0.5 * (1.0 + ci);    // an inner sub-cone: cos = midpoint
            for (int side = 0; side < 2; ++side) {
                Vec3 axis = (side == 0) ? e.beamDir : e.beamDir * -1.0;
                double sumCos = 0.0; long long nIn = 0, nInner = 0, nOnDisc = 0;
                for (int i = 0; i < NS; ++i)
                    for (int j = 0; j < NS; ++j) {
                        double u1 = (i + 0.5) / NS, u2 = (j + 0.5) / NS;
                        Vec3 d = sc.emitters.back().sampleCone(axis, u1, u2);
                        double c = dot(d, axis);
                        sumCos += c;
                        if (c >= ci - 1e-12) ++nIn;
                        if (c >= inner - 1e-12) ++nInner;
                        // (d) a NEE draw (axis = -beamDir, so `d` already points TOWARD
                        // the sun — the same sense as an escaping camera ray) must read
                        // back as ON the disc.
                        if (side == 1 && sc.emitters.back().inCone(d)) ++nOnDisc;
                        // unit length is what makes every cosine above meaningful
                        if (std::fabs(length(d) - 1.0) > 1e-9) ok = false;
                    }
                const double N = (double)NS * NS;
                const char* wh = (side == 0) ? "fwd" : "nee";
                std::snprintf(lbl, sizeof lbl, "%s: all draws inside cone (%.2f deg)", wh, dd);
                ok &= chk(lbl, (double)nIn / N, 1.0, 1e-12);
                // Uniform-in-solid-angle => cos is uniform on [cos theta, 1].
                std::snprintf(lbl, sizeof lbl, "%s: mean cos == (1+cos)/2 (%.2f deg)", wh, dd);
                ok &= chk(lbl, sumCos / N, 0.5 * (1.0 + ci), 1e-3);
                // ...so the inner sub-cone's share is its solid-angle share, = 1/2 here.
                std::snprintf(lbl, sizeof lbl, "%s: inner sub-cone share (%.2f deg)", wh, dd);
                ok &= chk(lbl, (double)nInner / N, 0.5, 2e-3);
                if (side == 1) {
                    std::snprintf(lbl, sizeof lbl, "nee draw reads back inCone (%.2f deg)", dd);
                    ok &= chk(lbl, (double)nOnDisc / N, 1.0, 1e-12);
                }
            }

            // (d) the rim is where the direct-view term and NEE must agree. A direction
            // a hair INSIDE the rim is on the disc; a hair outside is not.
            {
                Vec3 t, b; onb(aim, t, b);
                for (double eps : {-1e-4, 1e-4}) {
                    double th = theta + eps;
                    Vec3 look = aim * std::cos(th) + t * std::sin(th);   // ray TOWARD the sun
                    bool want = (eps < 0.0);
                    if (sc.emitters.back().inCone(look) != want) {
                        std::printf("[checksun] rim test failed at %.2f deg, eps=%+.0e  BAD\n", dd, eps);
                        ok = false;
                    }
                }
            }
        }
    }

    // A degenerate scene bound must not make the emitter's power NaN/inf: geomWeight
    // is Omega*PI*R^2 and is only filled by build(), so an unbuilt emitter reads 0.
    {
        Scene sc;
        sc.addSunLight(Vec3{0, 1, 0}, 0.5 * 0.53 * PI / 180.0, flat, 1.0);
        ok &= chk("unbuilt sun geomWeight == 0", sc.emitters.back().geomWeight(), 0.0, 0.0);
    }

    std::printf("[checksun] worst absolute error = %.3g\n", worst);
    std::printf("[checksun] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// The film accumulates radiance in an arbitrary (non-absolute) radiometric scale
// that depends on photon count, light power, etc., so the image is always anchored
// by an auto-exposure that maps the 99th luminance percentile to ~0.9. `expComp`
// is the camera's photographic exposure *compensation* (from iso/shutter/exposure,
// with 1.0 = neutral): the final exposure is auto * expComp, so e.g. ISO 200
// (expComp 2.0) is exactly one stop brighter than ISO 100. This is a *relative*
// control — true absolute EV needs absolute light power (watts/lumens), which is a
// separate deferred feature (see docs §8.1 / known-issues). expComp <= 0 means
// "not authored" -> neutral auto-exposure.
//
// The tone-mapped 8-bit RGB result is written via writeImage(), which picks the
// encoder from `path`'s extension (.png/.jpg/.jpeg, else PPM) — so `-o foo.png`
// yields a real PNG, not PPM bytes in a .png file.
// `lockAnchor` (optional) implements a shared auto-exposure anchor across the frames
// of a `camera_path` (see the exposure-lock feature): when non-null and *lockAnchor
// > 0 the stored `eAuto` is reused (so a dolly doesn't flicker frame-to-frame); when
// non-null but still 0 the freshly-computed `eAuto` is written back for the next
// frame. Null => per-frame auto-exposure (the default).
// Absolute-exposure sensor gain. In absolute mode (a scene with `power`/`lumens`
// emitters) the film's radiometric scale is physically meaningful, so instead of
// the content-dependent auto-exposure anchor we apply a FIXED gain times the
// photographic compensation `expComp = exposure*(iso/100)*shutter`. Aperture is
// not in expComp on purpose (the physical A/C modes already darken by passing
// fewer photons; see CamSpec). This constant is the sensor's absolute sensitivity
// calibration: it was chosen so a ~100 W area light in a unit (Cornell-scale) box
// at the neutral triple (ISO 100, 1 s, exposure 1) exposes to mid-tone. Changing
// lamp wattage/lumens then brightens or darkens the image (no auto-renormalise),
// and iso/shutter/exposure give exact photographic stops on top.
constexpr double ABS_EXPOSURE_GAIN = 6.0;

// Tone-map a film into an 8-bit RGB buffer (W*H*3, row 0 = image top; +y flipped to
// image-top to match writeImage). Shared by writeFilm (PNG/PPM output) and the live
// preview window so both see identical pixels. Auto-exposure mirrors writeFilm:
// absolute EV uses a fixed sensor gain; otherwise a p99 anchor (locked via lockAnchor
// if non-null, else recomputed per frame). Optionally reports the chosen gain/exposure.
static std::vector<uint8_t> filmToRgb8(const Film& f, double N, double expComp,
                                       bool absolute, double* lockAnchor,
                                       double* outEAuto = nullptr,
                                       double* outExposure = nullptr) {
    const int W = f.resX, H = f.resY;
    std::vector<Vec3> lin((size_t)W * H);
    double norm = 1.0 / (N * cieYIntegral());
    std::vector<double> lum; lum.reserve((size_t)W * H);
    for (size_t i = 0; i < lin.size(); ++i) {
        lin[i] = xyzToLinearSrgb(f.xyz[i] * norm);
        lum.push_back(std::max({lin[i].x, lin[i].y, lin[i].z, 0.0}));
    }
    double eAuto;
    double exposure;
    if (absolute) {
        // Absolute EV: fixed sensor gain, no per-image normalisation. Scene power
        // (watts/lumens) flows straight through; the auto-exposure anchor and the
        // camera_path exposure-lock are bypassed.
        eAuto = ABS_EXPOSURE_GAIN;
        exposure = eAuto * (expComp > 0.0 ? expComp : 1.0);
    } else {
    if (lockAnchor && *lockAnchor > 0.0) {
        eAuto = *lockAnchor;                       // reuse the path's locked anchor
    } else {
        std::vector<double> sorted = lum; std::sort(sorted.begin(), sorted.end());
        double p99 = sorted[(size_t)(0.99 * (sorted.size() - 1))];
        eAuto = (p99 > 0) ? 0.9 / p99 : 1.0;
        if (lockAnchor) *lockAnchor = eAuto;       // first frame sets the anchor
    }
    exposure = eAuto * (expComp > 0.0 ? expComp : 1.0);
    }

    std::vector<uint8_t> img((size_t)W * H * 3);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        size_t src = (size_t)(H - 1 - y) * W + x;       // flip so +y is image-top
        size_t dst = ((size_t)y * W + x) * 3;
        for (int c = 0; c < 3; ++c) {
            double v = (&lin[src].x)[c] * exposure;
            img[dst + c] = (uint8_t)std::clamp(srgbGamma(v) * 255.0 + 0.5, 0.0, 255.0);
        }
    }
    if (outEAuto)    *outEAuto = eAuto;
    if (outExposure) *outExposure = exposure;
    return img;
}

// Returns true on success, false if the image encoder failed. Callers that own the
// process exit code should propagate a non-zero status on false. (GPU renders that
// fail — driver TDR, device-memory/scheduling contention — are now caught at the
// source: every CUDA call in render_cuda.cu is checked via CUDA_CHECK/cudaCheckKernel,
// which fails loudly with a non-zero exit before any framebuffer is downloaded, so an
// all-zero/black film never reaches this function.)
static bool writeFilm(const char* path, const Film& f, double N, double expComp = 0.0,
                      bool quiet = false, double* lockAnchor = nullptr,
                      bool absolute = false) {
    const int W = f.resX, H = f.resY;
    double eAuto = 0.0, exposure = 0.0;
    std::vector<uint8_t> img = filmToRgb8(f, N, expComp, absolute, lockAnchor,
                                          &eAuto, &exposure);
    if (!writeImage(path, W, H, img)) {
        std::fprintf(stderr, "error: could not write %s\n", path);
        return false;
    }
    if (quiet) return true;
    if (absolute)
        std::printf("wrote %s (%dx%d), exposure=%.3g (absolute: gain %.3g x %.3g comp)\n",
                    path, W, H, exposure, eAuto, (expComp > 0.0 ? expComp : 1.0));
    else if (expComp > 0.0)
        std::printf("wrote %s (%dx%d), exposure=%.3g (auto %.3g x %.3gEV-comp)\n",
                    path, W, H, exposure, eAuto, expComp);
    else
        std::printf("wrote %s (%dx%d), auto-exposure=%.3g\n", path, W, H, exposure);
    return true;
}

// --- Live terminal preview ----------------------------------------------------
// Downsample the display film to a small ANSI-truecolour thumbnail and redraw it in
// place (cursor moved back up over the previous frame) so -time/-forever renders show
// a coarse live view without any external viewer. Uses the upper-half-block glyph so
// each character cell carries two vertical pixels (fg = upper, bg = lower), doubling
// the effective vertical resolution. Tone-mapping mirrors writeFilm (same p99 auto-
// exposure + sRGB gamma) so the thumbnail tracks what the written image looks like.
static int g_previewRows = 0;   // terminal lines the last preview occupied (for redraw)

// Photon-map (mode M) gather-radius controls. The density-estimation radius is
// g_pmRadiusAbs when >0 (absolute world units, CLI -pmradius), else a fraction
// g_pmRadiusFactor of the scene bounding-sphere radius (CLI -pmradiusfrac). Larger
// radius = smoother but blurrier estimate.
static double g_pmRadiusAbs = 0.0;
static double g_pmRadiusFactor = 0.02;

// Density-adaptive gather radius (mode M). ON by default: the radius above is only a
// starting point, and PhotonMap::buildAuto measures how many photons a typical gather
// actually sees and rescales the radius so that population lands on
// `g_pmAutoCount * cbrt(stored/1e6)`. Without this the radius is independent of `-n`, so
// photons-per-cell — and gather time — grows linearly with the photon count and mode M
// gets *slower per sample* the more photons you ask for (see known-issues.md). The target
// grows sublinearly on purpose so that noise AND bias both still converge; see the long
// note on PhotonMap::buildAuto for the exponent argument.
//
// An explicit `-pmradius <r>` means "use exactly this radius" and turns the adaptation off
// (that was the documented workaround for the scaling problem, so it must keep working);
// `-pmauto` forces it back on, `-nopmauto` off, `-pmcount <k>` sets the target and implies on.
static bool   g_pmAutoRadius = true;
static double g_pmAutoCount  = 200.0;   // calibrated to today's look — see PhotonMap::buildAuto

// Bin a freshly-deposited (or freshly-loaded) photon map, honouring the adaptive-radius
// setting, and say out loud what radius it settled on — the radius printed before the
// deposit is only the starting point, and a silently-different one would be baffling when
// comparing renders. Returns the radius actually used.
static double buildPhotonMap(PhotonMap& pm, double radius, const char* tag) {
    if (!g_pmAutoRadius) { pm.build(radius); return radius; }
    double nProbe = 0.0, kTarget = 0.0;
    const double r = pm.buildAuto(radius, g_pmAutoCount, &nProbe, &kTarget);
    std::printf("%s adaptive gather radius: %.4g -> %.4g (a typical gather saw %.0f photons "
                "at the starting radius; target %.0f for %zu stored)\n",
                tag, radius, r, nProbe, kTarget, pm.photons.size());
    return r;
}

// Mode-M final gather (CLI -pmfg <K>). 0 = off: read the density estimate directly at the
// visible point (fast, but the estimate's blur softens contact shadows / fine detail right
// at that surface). K > 0 = Jensen final gather: shoot K cosine-weighted hemisphere
// sub-rays from the visible point and query the map ONE bounce away, decoupling visible-
// surface sharpness from the gather radius (sharper contact shadows, at ~K x the cost, so
// pair with fewer spp). See photonmap_render.h (photonGatherSub).
static int g_pmFinalGather = 0;

// Mode-M photon-map cache file (CLI -savemap / -loadmap). The deposited map is view-
// independent — the expensive result of the forward photon trace, gatherable by any camera
// at any radius — so it is worth persisting. -savemap writes it after the GPU deposit pass;
// -loadmap reloads it and SKIPS the deposit, re-gathering new angles / a new radius without
// re-tracing a photon. Empty = disabled. GPU shared mode-M only (see renderPhotonMapSharedCuda).
static std::string g_pmapSave;
static std::string g_pmapLoad;

// SPPM (mode S) radius-shrink rate alpha (Hachisuka 2008; CLI -sppmalpha). Smaller =
// faster radius shrink (less bias sooner, more variance); 0.7 is the paper default. The
// initial radius R0 reuses the mode-M -pmradius / -pmradiusfrac controls above.
static double g_sppmAlpha = 0.7;

// VCM (mode U) radius-shrink rate alpha (Georgiev 2012; CLI -vcmalpha). The per-pass merge
// radius follows r_i = R0 * i^((alpha-1)/2), i = pass index (1-based); 0.75 is the
// SmallVCM default. Initial radius R0 reuses the mode-M -pmradius / -pmradiusfrac controls.
static double g_vcmAlpha = 0.75;

// Hero-wavelength bundle size (CLI -heroc N). Number of wavelengths carried per path
// (hero + N-1 stratified secondaries) on the CPU hero tracers (modes A/B/C, R, M/S).
// Set once at arg-parse; clamped to [1, hero::kHeroMax]. N==1 turns hero off (bit-identical
// single-λ). Defaults to hero::kHeroC (4). GPU / BDPT / VCM paths ignore it (still single-λ).
static int g_heroC = hero::kHeroC;
// Was -heroc given explicitly? Mode W raises the default to the full kHeroMax bundle
// (the C wavelengths ride ONE shared BVH walk, so a wider bundle is very nearly free,
// while it is the only thing that buys spectral accuracy at 1 spp) — but never over an
// explicit user choice.
static bool g_heroCSet = false;

// Scene-ignore render params (Stage 3), set once at arg-parse and read by the tracer
// wrappers (like g_heroC). g_maxBounceOverride < 0 leaves each tracer's own default
// (32 for the unidirectional tracers, 8 for the bidirectional D/U estimators, whose
// connection cost grows ~depth^2); >= 1 SETS the path-depth loop and is honoured
// UNIVERSALLY (forward B, backward R/RGB, BDPT D, VCM U, photon/SPPM, P). Note that for
// D/U it can RAISE the depth as well as cap it, which is what a specular-only cavity
// needs. g_directOnly renders direct lighting + specular recursion
// only (no diffuse indirect bounce) — a Whitted-style near-1-spp preview — in the CAMERA
// path tracers where it is well-defined: backward R, the RGB fast path, and the backward
// camera side of the P composite. The forward light tracer (B) and the photon /
// bidirectional modes (M/S/D) honour maxBounce but ignore directOnly.
static int  g_maxBounceOverride = -1;
static bool g_directOnly = false;

// -mode W: the DETERMINISTIC Whitted preview. g_directOnly alone still leaves every
// estimator stochastic (one random light point, one random glossy direction, a Russian-
// roulette coin per specular bounce), so it needs tens of spp to look clean and buys
// only ~3x over full GI. g_whitted additionally swaps all three for their deterministic
// equivalents, which is what lets it converge at ONE sample per pixel -- the actual
// order-of-magnitude win, and what POV-Ray does. g_whittedGrid is the NxN shadow-ray
// lattice per area light; g_ambient is the flat GI stand-in (POV-Ray's `ambient`),
// without which a CLOSED room previews with black shadows, since everything there that
// isn't facing the key light is lit purely by the bounce this mode drops.
static bool g_whitted = false;
static int  g_whittedGrid = 4;
static double g_ambient = 0.0;

// -gi: mode W's deterministic ONE-BOUNCE GATHER, the real thing g_ambient only stands in
// for. A flat constant cannot reproduce contact darkening (it lights a crevice exactly as
// much as an exposed face) or colour bleeding (it is grey, where light that has bounced
// off gold is not), and no single value fixes both -- raising it to fill the crevices
// blows out the open faces. g_gi > 0 instead traces that many rays from every diffuse
// vertex along a FIXED lattice and takes whatever deterministic Whitted radiance they
// find. Unlike POV-Ray's radiosity there is no irradiance cache, so nothing depends on
// render order or on which sample points the geometry happened to trigger -- which is
// what makes it safe for an animated loop. See BackwardRenderer::giDirs.
static int g_gi = 0;
static int g_giGrid = 1;
static int g_giBounce = 4;
// -gi-clamp, dimensionless like -ambient (a multiple of one light's own radiance, scaled by
// Scene::ambientRef() at the two hand-off sites below). 0 = off. Caps one gather ray's
// returned radiance, which is what tames the caustic-through-the-gather contour aliasing.
// See BackwardRenderer::giClamp for the full rationale.
static double g_giClamp = 0.0;

// PHOTON-BEAMS gather for the shared multi-camera forward pass (CLI -beams). When set,
// the shared A/B pass has each camera resample its own medium in-scatter point per beam
// segment, so a volumetric FLYBY (rainbow/fogbow/fog) gets independent per-frame noise
// instead of one frozen speckle pattern, while the photon flight is still traced once.
// Only affects groups of >1 shared camera with participating media; single stills and
// media-free scenes are byte-for-byte unchanged. Forces the CPU forward path (the GPU
// shared kernel doesn't implement the per-camera resample yet).
static bool g_beamGather = false;

// Enable ANSI/virtual-terminal escape processing so the preview renders in a plain
// Windows console (conhost/cmd), not only in Windows Terminal. No-op elsewhere.
static void enableAnsiTerminal() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD m = 0;
    if (h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &m))
        SetConsoleMode(h, m | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}
static void ansiPreview(const Film& f, double N, double expComp, const char* status) {
    const int W = f.resX, H = f.resY;
    const int gw = std::min(W, 72);                   // thumbnail width in characters
    int gh = std::max(2, (int)std::llround((double)H / W * gw)); // thumbnail pixel rows
    if (gh % 2) ++gh;                                 // even so half-blocks pair cleanly

    double norm = 1.0 / (N * cieYIntegral());
    // Box-downsample film -> linear sRGB grid, tracking p99 for auto-exposure.
    std::vector<Vec3> grid((size_t)gw * gh);
    std::vector<double> lum; lum.reserve(grid.size());
    for (int gy = 0; gy < gh; ++gy) for (int gx = 0; gx < gw; ++gx) {
        int x0 = gx * W / gw, x1 = std::max(x0 + 1, (gx + 1) * W / gw);
        int y0 = gy * H / gh, y1 = std::max(y0 + 1, (gy + 1) * H / gh);
        Vec3 s{}; double n = 0.0;
        for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) {
            s += f.xyz[(size_t)y * W + x]; n += 1.0;
        }
        Vec3 c = xyzToLinearSrgb((s / std::max(1.0, n)) * norm);
        grid[(size_t)gy * gw + gx] = c;
        lum.push_back(std::max({c.x, c.y, c.z, 0.0}));
    }
    std::vector<double> sorted = lum; std::sort(sorted.begin(), sorted.end());
    double p99 = sorted.empty() ? 0.0 : sorted[(size_t)(0.99 * (sorted.size() - 1))];
    double eAuto = (p99 > 0) ? 0.9 / p99 : 1.0;
    double exposure = eAuto * (expComp > 0.0 ? expComp : 1.0);

    auto px = [&](int gx, int gy, int& r, int& g, int& b) {
        // Flip vertically so +y is image-top, matching writeFilm.
        const Vec3& c = grid[(size_t)(gh - 1 - gy) * gw + gx];
        r = (int)std::clamp(srgbGamma(c.x * exposure) * 255.0 + 0.5, 0.0, 255.0);
        g = (int)std::clamp(srgbGamma(c.y * exposure) * 255.0 + 0.5, 0.0, 255.0);
        b = (int)std::clamp(srgbGamma(c.z * exposure) * 255.0 + 0.5, 0.0, 255.0);
    };

    std::string out;
    if (g_previewRows > 0) { out += "\033["; out += std::to_string(g_previewRows); out += "A"; }
    for (int gy = 0; gy < gh; gy += 2) {
        for (int gx = 0; gx < gw; ++gx) {
            int tr, tg, tb, br, bg, bb;
            px(gx, gy, tr, tg, tb);
            px(gx, std::min(gy + 1, gh - 1), br, bg, bb);
            char buf[64];
            std::snprintf(buf, sizeof buf, "\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm\xE2\x96\x80",
                          tr, tg, tb, br, bg, bb);
            out += buf;
        }
        out += "\033[0m\033[K\n";
    }
    out += "\033[K"; out += (status ? status : ""); out += "\n";
    g_previewRows = gh / 2 + 1;
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
}

// Deterministic, noise-free visualisation of the thin-film structural colour: a
// swatch whose rows sweep coating thickness (dMin..dMax nm) and whose columns
// sweep incidence angle (0..~85 deg). Each cell integrates the interference
// reflectance R(lambda) against the CIE curves under a flat (equal-energy)
// illuminant, so the pixel colour is exactly the colour the coating reflects at
// that thickness/angle. Reuses thinFilmReflectance -> a single source of truth
// with the renderer, and makes the iridescent colour bands unmistakable without
// the Monte-Carlo noise of a forward-catch render.
static void thinFilmSwatch(double n1, double n2) {
    const int W = 512, H = 256;
    const double dMin = 100.0, dMax = 800.0, thetaMax = 85.0 * PI / 180.0;
    Film f; f.resX = W; f.resY = H; f.alloc();
    for (int y = 0; y < H; ++y) {
        double d = dMin + (dMax - dMin) * (y + 0.5) / H;             // thickness (row)
        for (int x = 0; x < W; ++x) {
            double cosI = std::cos(thetaMax * (x + 0.5) / W);        // angle (column)
            Vec3 xyz{};
            for (double lam = LAMBDA_MIN; lam <= LAMBDA_MAX; lam += 1.0) {
                double R = thinFilmReflectance(1.0, n1, n2, 0.0, d, cosI, lam);
                xyz += Vec3(cieX(lam), cieY(lam), cieZ(lam)) * R;    // flat illuminant
            }
            f.add(x, y, xyz);
        }
    }
    // N=1: writeFilm's 1/(N*cieYIntegral) makes a perfect (R=1) reflector map to
    // white, so the swatch colours are physical reflectances (up to auto-exposure).
    writeFilm("thinfilm_swatch.ppm", f, 1.0);
    std::printf("[thinfilm] swatch n1=%.2f n2=%.2f: rows=thickness %.0f-%.0fnm, cols=angle 0-85deg\n",
                n1, n2, dMin, dMax);
}

// Add the directly-viewed environment background to a forward (model-B) film. For
// each pixel whose pixel-center camera ray escapes all geometry, deposit N times the
// escape direction's env XYZ (constant for a flat env, the lat-long map colour for
// an image env), so that after writeFilm's 1/(N*cieYIntegral) normalisation the pixel
// shows the environment radiance in XYZ — matching the backward tracer's ray-miss
// term (which adds L_env(dir)*invPdfLambda). Forward photons carry the env *illumination* of
// surfaces; this pass supplies the *direct view* of the sky behind the geometry.
// No-op unless the scene has an env light. Silhouette pixels are classified by the
// pixel center (a sub-pixel edge approximation, like mode P's classifier).
// A distant `sun` light is handled here too: a forward photon fired into the solar
// cone never travels *toward* the camera (the disc is at infinity), so the direct
// view of the solar disc — like the sky behind the geometry — is a pure camera-ray
// term. `sunXYZForDir` returns 0 outside every sun's cone, so this costs nothing
// for a scene without one.
static void addEnvBackground(Film& film, const Scene& scene, const Camera& cam, long long N) {
    const bool haveEnv = scene.envIndex >= 0, haveSun = scene.sunCount > 0;
    if (!haveEnv && !haveSun) return;
    for (int py = 0; py < film.resY; ++py)
        for (int px = 0; px < film.resX; ++px) {
            Ray r = cam.genRay(px, py, 0.5, 0.5);
            Hit h = scene.closestHit(r);
            if (!h.valid) {
                Vec3 bg{0, 0, 0};
                if (haveEnv) bg += scene.envXYZForDir(r.d);
                if (haveSun) bg += scene.sunXYZForDir(r.d);
                film.add(px, py, bg * (double)N);
            }
        }
}

// Forward photon trace (models A/B/C) into a merged film. Accumulates the energy
// report across threads. Factored out so mode V can reuse it alongside the
// backward reference.
// `seedBase` offsets the per-thread RNG streams so that rendering the image in
// several accumulation passes (a wall-clock time budget, or resuming a saved film)
// draws statistically-independent photons each pass; pass the cumulative photon
// count already traced. seedBase==0 reproduces the original single-shot streams
// bit-for-bit, so a plain `-n` render is unchanged.
static Film renderForward(const Scene& scene, const Camera* cam, int resX, int resY, long long N,
                          int nThreads, bool forwardCatch, bool lensMode, bool useCamera,
                          EnergyReport& eOut, bool diffraction = true, bool useGpu = false,
                          uint64_t seedBase = 0, bool wavefront = false) {
#ifdef HAVE_CUDA
    // GPU path covers all three finite-lens camera models: the pinhole splat (B), the
    // brute-force catch (C), and the finite-lens next-event splat (A). Fluorescent
    // scenes fall back to the CPU (the reradiation sampler is not ported). `wavefront`
    // selects the streaming backend over the default megakernel (same physics/energy).
    if (useGpu && cam && cudaAvailable() && cudaForwardSupported(scene)) {
        char camMode = lensMode ? 'A' : forwardCatch ? 'C' : 'B';
        return renderForwardCuda(scene, *cam, resX, resY, N, eOut, diffraction, camMode, seedBase, wavefront, g_heroC);
    }
#else
    (void)useGpu; (void)wavefront;
#endif
    std::vector<Film> films(nThreads);
    std::vector<EnergyReport> reports(nThreads);
    for (auto& f : films) { f.resX = resX; f.resY = resY; f.alloc(); }

    // Hero-wavelength sampling: on when C>1 and the scene has no participating media /
    // GRIN (dispersive interfaces de-hero mid-path). Forward modes A/B/C all qualify —
    // the finite-lens pupil is achromatic, so the C wavelengths share the connection.
    const bool heroOn = (g_heroC > 1) && scene.media.empty() && !grin::sceneHasGrin(scene);
    auto worker = [&](int tid) {
        Renderer r; r.forwardCatch = forwardCatch; r.lensMode = lensMode; r.diffraction = diffraction;
        r.useHero = heroOn; r.heroC = g_heroC;
        if (g_maxBounceOverride >= 1) r.maxBounce = g_maxBounceOverride;
        // Photon i draws from its own stream keyed by the ABSOLUTE photon index
        // seedBase+i (seedBase = cumulative photons of earlier batches), so the traced
        // set is independent of batch splits and thread count (see rng.h seedUnit).
        Pcg32 rng;
        long long lo = N * tid / nThreads, hi = N * (tid + 1) / nThreads;
        Film* sensorFilm = useCamera ? nullptr : &films[tid];
        const Camera* camPtr = useCamera ? cam : nullptr;
        Film* camFilm = useCamera ? &films[tid] : nullptr;
        for (long long i = lo; i < hi; ++i) {
            seedUnit(rng, seedBase + (uint64_t)i, 0x9E3779B97F4A7C15ULL);
            r.tracePhoton(scene, camPtr, sensorFilm, camFilm, rng, reports[tid]);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    Film out; out.resX = resX; out.resY = resY; out.alloc();
    for (int t = 0; t < nThreads; ++t) { out.merge(films[t]); }
    for (auto& rp : reports) {
        eOut.emitted += rp.emitted; eOut.absorbed += rp.absorbed; eOut.sensor += rp.sensor;
        eOut.escaped += rp.escaped; eOut.residual += rp.residual;
    }
    return out;
}

// Shared multi-camera forward pass (models A and B). Traces ONE set of N photons and
// splats every diffuse/emitter/volume vertex to *all* cameras at once, returning one
// film per camera — the "many cameras for 1x photon work" win instead of re-tracing
// per camera. With `lensMode` false (model B) connect() draws no RNG, so the per-thread
// RNG streams and photon paths are identical to a single-camera renderForward and each
// camera's film is bit-for-bit an independent renderForward for that camera (validated).
// With `lensMode` true (model A) each camera samples its own aperture (connectLens draws
// RNG), so the shared photon flight is still a valid unbiased estimate for every camera
// but the per-camera images match a standalone render in distribution only, not bit-for-
// bit. Model C is never shared (it consumes the photon per camera). The GPU twin is
// renderForwardSharedCuda; each camera keeps its own resolution/projection/exposure.
static std::vector<Film> renderForwardShared(const Scene& scene,
                                             const std::vector<Camera>& cams,
                                             const std::vector<int>& resX,
                                             const std::vector<int>& resY,
                                             long long N, int nThreads,
                                             EnergyReport& eOut, bool diffraction = true,
                                             bool lensMode = false,
                                             unsigned long long seedBase = 0,
                                             bool beamGather = false) {
    int nc = (int)cams.size();
    // Per-thread × per-camera films (each thread accumulates into its own copies to
    // avoid shared-pixel races; merged per camera at the end).
    std::vector<std::vector<Film>> films(nThreads, std::vector<Film>(nc));
    std::vector<EnergyReport> reports(nThreads);
    for (int t = 0; t < nThreads; ++t)
        for (int c = 0; c < nc; ++c) { films[t][c].resX = resX[c]; films[t][c].resY = resY[c]; films[t][c].alloc(); }

    const bool heroOn = (g_heroC > 1) && scene.media.empty() && !grin::sceneHasGrin(scene);
    auto worker = [&](int tid) {
        Renderer r; r.forwardCatch = false; r.lensMode = lensMode; r.diffraction = diffraction;
        r.useHero = heroOn; r.heroC = g_heroC; r.beamGather = beamGather;
        if (g_maxBounceOverride >= 1) r.maxBounce = g_maxBounceOverride;
        // Identical per-photon seeding to renderForward (absolute index seedBase+i via
        // seedUnit). For model B this keeps each camera's shared film bit-identical to
        // its standalone single-camera render at the same seedBase; for model A the
        // aperture draws perturb the stream, so it matches in distribution. `seedBase`
        // (the cumulative photon count) makes a checkpointed / resumed / budgeted
        // shared render draw fresh photons each pass, independent of the batch split.
        Pcg32 rng;
        long long lo = N * tid / nThreads, hi = N * (tid + 1) / nThreads;
        std::vector<CamTarget> targets(nc);
        for (int c = 0; c < nc; ++c) { targets[c].cam = &cams[c]; targets[c].film = &films[tid][c]; }
        for (long long i = lo; i < hi; ++i) {
            seedUnit(rng, seedBase + (uint64_t)i, 0x9E3779B97F4A7C15ULL);
            r.tracePhoton(scene, targets.data(), nc, /*sensorFilm*/nullptr, rng, reports[tid]);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    std::vector<Film> out(nc);
    for (int c = 0; c < nc; ++c) {
        out[c].resX = resX[c]; out[c].resY = resY[c]; out[c].alloc();
        for (int t = 0; t < nThreads; ++t) out[c].merge(films[t][c]);
    }
    for (auto& rp : reports) {
        eOut.emitted += rp.emitted; eOut.absorbed += rp.absorbed; eOut.sensor += rp.sensor;
        eOut.escaped += rp.escaped; eOut.residual += rp.residual;
    }
    return out;
}

// Backward reference: `spp` samples per pixel, threads render disjoint row bands
// of a shared film (no shared-pixel writes, so no race).
// `sampleBase` is the ABSOLUTE index of the first sample: a chunked/progressive
// render passes its running spp count so successive chunks render successive
// per-(pixel,sample) streams — the realization is identical for ANY chunk split,
// thread count, or resume boundary (see renderRows / rng.h seedUnit).
// `forceWhitted` renders this one film in deterministic mode W even when the run's mode is
// something else. That is for the interactive viewer's live preview, which wants mode W's
// noise-free-at-1-spp frame regardless of what the batch render is set to; a global flip
// would leak into every other call, so the override is per-call.
//
// `rowBegin`/`rowEnd` (rowEnd < 0 = to the end) render only a BAND of the film, with the
// thread pool splitting that band rather than the whole frame. The interactive preview uses
// it to render a pose a slice at a time so a mode-W frame that costs seconds (a gyroid
// labyrinth is ~26s at 960x600, vs 0.4s for a Cornell box) can still be shown as it fills
// and abandoned the moment the camera moves. Rows outside the band are left untouched, so
// the caller owns the film across calls.
static Film renderBackward(const Scene& scene, const Camera& cam, int resX, int resY,
                           long long spp, int nThreads, bool diffraction = true,
                           unsigned long long sampleBase = 0, bool forceWhitted = false,
                           Film* into = nullptr, int rowBegin = 0, int rowEnd = -1) {
    Film out;
    if (!into) { out.resX = resX; out.resY = resY; out.alloc(); }
    Film& film = into ? *into : out;
    const int bandLo = std::clamp(rowBegin, 0, resY);
    const int bandHi = std::clamp(rowEnd < 0 ? resY : rowEnd, bandLo, resY);
    const int bandN  = bandHi - bandLo;
    if (bandN <= 0) return out;
    if (bandN < nThreads) nThreads = bandN;   // don't hand a thread an empty row range
    auto worker = [&](int tid) {
        BackwardRenderer br; br.diffraction = diffraction; br.heroC = g_heroC;
        if (g_maxBounceOverride >= 1) br.maxBounce = g_maxBounceOverride;
        br.directOnly = g_directOnly || forceWhitted;   // mode W is direct-only by construction
        br.whitted = g_whitted || forceWhitted; br.lightGrid = g_whittedGrid;
        // Mode W turns split-at-dispersion ON by default. Its λ lattice is shared by every
        // pixel, so a de-hero would collapse the WHOLE FRAME onto one wavelength and mistint
        // every dielectric -- a deterministic error, not noise, so no amount of spp fixes it.
        // Mode R is stochastic and averages the collapse away, so there it stays opt-in
        // (`-herosplit`, which reaches the backward tracer via BackwardRenderer::heroSplit).
        br.heroSplit = hero::gSplit || br.whitted;
        // -ambient is dimensionless (fraction of a light's own radiance); convert to
        // this scene's absolute radiance scale here. See Scene::ambientRef().
        br.ambient = g_ambient * scene.ambientRef();
        br.giDirs = g_gi; br.giGrid = g_giGrid; br.giBounce = g_giBounce;
        br.giClamp = g_giClamp * scene.ambientRef();   // same scaling as -ambient above
        int y0 = bandLo + bandN * tid / nThreads, y1 = bandLo + bandN * (tid + 1) / nThreads;
        br.renderRows(scene, cam, film, y0, y1, spp, sampleBase);
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    return out;
}

// Mode D: bidirectional path tracing. Each thread renders a band of rows into its own
// camera-image film (t>=2 connections, current-pixel) and light-image film (t==1
// splats to the projected raster position), then all bands are merged. BDPT produces
// one absolute-radiance image: normalise the camera image by spp (the per-pixel
// radiance convention shared with the backward reference, mode R) and the light image
// by the total light-subpath count (W*H*spp), matching mode B's splat convention. The
// two normalised films sum to the final radiance; writeFilm(...,1.0) then only divides
// by cieYIntegral for display, exactly like mode P's composite.
static Film renderBdpt(const Scene& scene, const Camera& cam, int resX, int resY,
                       long long spp, int nThreads, int maxDepth, bool diffraction = true,
                       unsigned long long sampleBase = 0) {
    std::vector<Film> camBands(nThreads), splatBands(nThreads);
    auto worker = [&](int tid) {
        bdpt::BdptRenderer br; br.maxDepth = maxDepth; br.diffraction = diffraction;
        br.heroC = g_heroC;   // renderRows applies the media/GRIN/lens gate itself
        Film& cf = camBands[tid]; cf.resX = resX; cf.resY = resY; cf.alloc();
        Film& sf = splatBands[tid]; sf.resX = resX; sf.resY = resY; sf.alloc();
        int y0 = resY * tid / nThreads, y1 = resY * (tid + 1) / nThreads;
        br.renderRows(scene, cam, cf, sf, y0, y1, spp, sampleBase);
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    Film cam_film; cam_film.resX = resX; cam_film.resY = resY; cam_film.alloc();
    Film splat_film; splat_film.resX = resX; splat_film.resY = resY; splat_film.alloc();
    for (int t = 0; t < nThreads; ++t) { cam_film.merge(camBands[t]); splat_film.merge(splatBands[t]); }

    // Combine onto one radiance scale as a SUM over samples (display divides by spp via
    // writeFilm(out, spp), matching the GPU renderBdptCuda convention and letting a
    // chunked/progressive render accumulate batches by summing). Both halves share the
    // per-pixel sample count spp: the camera image (t>=2) is a per-pixel radiance
    // estimate; the light image (t==1 splats) uses the full-image-plane camera importance
    // We (see bdpt.h cameraWe), for which (1/spp)*We(A_full) is exactly the light-tracing
    // scale (equivalently (1/(W*H*spp))*We(A_pixel) — the mode-B convention).
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    for (size_t i = 0; i < out.xyz.size(); ++i)
        out.xyz[i] = cam_film.xyz[i] + splat_film.xyz[i];
    return out;
}

// Mode P: forward light tracing (model B) composited with a camera-side ray path,
// so specular/coated surfaces are finally visible directly from the camera.
//
// Model B renders diffuse-first pixels (and the caustics specular surfaces cast
// onto diffuse ones), but leaves specular/coated surfaces BLACK: a specular
// last-vertex-before-camera has zero connect pdf (the SDS limitation). The
// camera-side path is a backward path trace from the camera — it follows the
// specular chain deterministically and does NEE + GI at the first diffuse vertex —
// which fills exactly those specular-first pixels. The two path sets are DISJOINT,
// partitioned by the first camera-ray hit (diffuse-side vs specular-side), so
// compositing them double-counts nothing; it is the BDPT observation that forward
// wins on L(S)*D*E caustics while camera-side wins on directly-viewed E*S* paths.
//
// The forward film measures radiance x a single global constant s (the model-B
// camera measurement convention). We recover s by best-fit against the backward
// radiance over the diffuse-side pixels (where both estimators agree), then convert
// forward to radiance as F/(N*s) and drop in the backward radiance R/spp on the
// specular-side pixels. The result is one true-radiance image.
//
// NOTE: fluorescence is unsupported here (the backward tracer can't reradiate) —
// same caveat as modes R/V. Classification uses the pixel-centre camera ray, so
// silhouette pixels are assigned wholesale to one side (a sub-pixel edge approx).
// Per-pixel first-hit classification for the mode-P composite (view-dependent, so it is
// computed ONCE and reused across every progressive pass). Each pixel is DIFF (forward
// model-B layer), SPEC (camera-side backward layer fills the SDS specular gap), or SKY
// (directly-viewed environment). `skyXYZ` holds the env radiance per SKY pixel.
struct CompositeClass {
    enum { DIFF = 0, SPEC = 1, SKY = 2 };
    std::vector<char> cls;
    std::vector<Vec3> skyXYZ;   // populated only for env scenes
    long long nSpec = 0, nSky = 0;
};

static CompositeClass classifyComposite(const Scene& scene, const Camera& cam,
                                        int resX, int resY) {
    CompositeClass cc;
    cc.cls.assign((size_t)resX * resY, CompositeClass::DIFF);
    if (scene.envIndex >= 0) cc.skyXYZ.assign(cc.cls.size(), {});
    for (int py = 0; py < resY; ++py)
        for (int px = 0; px < resX; ++px) {
            size_t i = (size_t)py * resX + px;
            Ray r = cam.genRay(px, py, 0.5, 0.5);
            Hit h = scene.closestHit(r);
            if (!h.valid) {
                if (scene.envIndex >= 0) {
                    cc.cls[i] = CompositeClass::SKY;
                    cc.skyXYZ[i] = scene.envXYZForDir(r.d); ++cc.nSky;
                }
                // (no env => leave as DIFF; the forward film is legitimately black there)
            } else if (h.sensorId < 0 && isSpecularType(scene.mats[h.matId].type)) {
                cc.cls[i] = CompositeClass::SPEC; ++cc.nSpec;
            }
        }
    return cc;
}

// Composite the accumulated forward (SUM over N photons) and backward (SUM over spp)
// films into one true-radiance image using the fixed classification `cc`. Fits the
// forward->backward scale s over the DIFF pixels, then blends: forward F/(N*s) on DIFF,
// backward R/spp on SPEC, env radiance on SKY. Result is in writeFilm(...,1.0) units.
// When `verbose` prints the scale/residual diagnostics (once, at the final write).
static Film compositeFromFilms(const Film& fwd, long long N, const Film& ref, long long spp,
                               const CompositeClass& cc, bool envScene, bool verbose) {
    using CC = CompositeClass;
    const double invF = 1.0 / (double)N, invR = 1.0 / (double)spp;
    const auto& cls = cc.cls;

    // Best-fit forward->backward scale over the DIFF pixels only: Fval ~ s*Rval,
    // so s = sum(Fval.Rval)/sum(Rval.Rval) (same convention as compareFilms).
    double sfr = 0, srr = 0;
    for (size_t i = 0; i < cls.size(); ++i) {
        if (cls[i] != CC::DIFF) continue;
        Vec3 f = fwd.xyz[i] * invF, rv = ref.xyz[i] * invR;
        sfr += dot(f, rv); srr += dot(rv, rv);
    }
    double s = (srr > 0) ? sfr / srr : 1.0;
    if (s <= 0) s = 1.0;

    double num = 0, den = 0;
    for (size_t i = 0; i < cls.size(); ++i) {
        if (cls[i] != CC::DIFF) continue;
        Vec3 fr = fwd.xyz[i] * (invF / s), rv = ref.xyz[i] * invR;
        Vec3 dd = fr - rv;
        num += dot(dd, dd); den += dot(fr, fr);
    }
    double rmse = (den > 0) ? std::sqrt(num / den) : 0.0;

    Film comp; comp.resX = fwd.resX; comp.resY = fwd.resY; comp.alloc();
    for (size_t i = 0; i < cls.size(); ++i)
        comp.xyz[i] = (cls[i] == CC::SPEC) ? ref.xyz[i] * invR
                    : (cls[i] == CC::SKY)  ? cc.skyXYZ[i]
                                           : fwd.xyz[i] * (invF / s);
    if (verbose) {
        std::printf("[composite] forward->radiance scale s=%.6g  specular-first pixels=%lld/%lld\n",
                    s, cc.nSpec, (long long)cls.size());
        if (envScene)
            std::printf("[composite] env background on %lld escaped (sky) pixels\n", cc.nSky);
        std::printf("[composite] diffuse-side residual (forward/s vs backward) rel RMSE=%.4f\n", rmse);
    }
    return comp;
}

// Compare forward vs backward films in raw linear-XYZ radiance. Because the two
// estimators measure the same image under different conventions, we solve for the
// single best-fit scale s (backward -> forward) and report the relative RMSE of
// the residual. A small RMSE validates the forward transport/camera math; a large
// or structured residual flags a bug.
static void compareFilms(const Film& fwd, long long Nfwd, const Film& ref, long long spp) {
    const double invF = 1.0 / (double)Nfwd, invR = 1.0 / (double)spp;
    double sfr = 0, srr = 0, sff = 0;
    size_t n = fwd.xyz.size();
    for (size_t i = 0; i < n; ++i) {
        Vec3 f = fwd.xyz[i] * invF, r = ref.xyz[i] * invR;
        sfr += dot(f, r); srr += dot(r, r); sff += dot(f, f);
    }
    double s = (srr > 0) ? sfr / srr : 0.0;
    double num = 0;
    std::vector<double> perPix(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        Vec3 f = fwd.xyz[i] * invF, r = ref.xyz[i] * invR;
        Vec3 d = f - r * s; double sq = dot(d, d);
        perPix[i] = sq; num += sq;
    }
    double rmse = (sff > 0) ? std::sqrt(num / sff) : 0.0;
    // Firefly vs bias diagnostic: what fraction of the squared residual is held by
    // the worst 1% of pixels? A high concentration means a few high-variance pixels
    // (fireflies from the unbounded 1/dist^2 connect near the light) dominate — a
    // sampling-quality issue, not a transport bug. A low concentration with high
    // RMSE means a broad, structured residual — the fingerprint of an actual bug.
    std::vector<double> sortedPix = perPix;
    std::sort(sortedPix.begin(), sortedPix.end(), std::greater<double>());
    size_t top = std::max<size_t>(1, n / 100);
    double topSum = 0; for (size_t i = 0; i < top; ++i) topSum += sortedPix[i];
    double concentration = (num > 0) ? topSum / num : 0.0;
    // Bulk RMSE excludes the top-1% highest-residual pixels. If the bulk agrees
    // (small) while the full RMSE is large, the disagreement is confined to a few
    // firefly pixels — the transport is correct and only variance remains.
    double bulkNum = num - topSum;
    double bulkRmse = (sff > 0) ? std::sqrt(bulkNum / sff) : 0.0;
    bool pass = (rmse < 0.05) || (concentration > 0.5 && bulkRmse < 0.05);
    std::printf("[validate] best-fit scale (backward->forward) = %.6g\n", s);
    std::printf("[validate] relative RMSE after scale = %.3f%%  (full) / %.3f%% (bulk, ex. top-1%%)\n",
                100.0 * rmse, 100.0 * bulkRmse);
    std::printf("[validate] residual concentration: top-1%% pixels hold %.1f%% of it (%s)\n",
                100.0 * concentration,
                concentration > 0.5 ? "firefly/variance-dominated" : "broadly distributed");
    std::printf("[validate] %s\n", pass
                ? "PASS: forward light tracer agrees with backward reference."
                : "review: residual above 5% — increase -n/-spp (if firefly-dominated) or investigate transport.");
}

// --- Graceful interrupt (Ctrl-C) for indefinite / long renders ----------------
// -forever (and any long -time render) traps SIGINT so the first Ctrl-C requests a
// clean stop -- the batch loop notices the flag, writes a final image + checkpoint,
// and returns -- while a second Ctrl-C restores the default handler and force-quits.
// A sig_atomic_t flag is the only state a signal handler may touch portably.
static volatile std::sig_atomic_t g_stopRequested = 0;
static void onInterrupt(int sig) {
    if (g_stopRequested) { std::signal(sig, SIG_DFL); std::raise(sig); return; }
    g_stopRequested = 1;
}

// --- External graceful stop (`ftrace -stop [<pid>|all]`) -----------------------
// Ctrl-C only reaches a render that owns the console it was started from. A render
// launched detached, or from a tool that isn't its parent, previously had no way to
// be stopped except `taskkill /F` -- and killing ftrace while CUDA kernels are in
// flight is a well-known way to wedge the NVIDIA display driver (TDR / bugcheck).
// "Just kill it" is therefore not an acceptable stop for this program. This channel
// delivers the exact same CLEAN stop Ctrl-C delivers -- finish the current chunk,
// write the final image + .ftbuf checkpoint, unwind through cudaGracefulShutdown()
// -- but triggerable from outside the process.
//
// The channel is a sentinel FILE, deliberately, rather than a named event or socket:
// renders run in the interactive Console session while whatever wants to stop them
// may live in a different session / window station (the same split that makes
// -window invisible when launched from a sandboxed shell), and `Local\` kernel
// objects are per-session. The filesystem is the one namespace both sides share.
//
//   <temp>/ftrace/<pid>.run    exists while this process runs; holds a one-line
//                              "scene -> output" description. `-stop` with no
//                              argument lists these, `-stop all` targets them all.
//                              Removed on exit; one left behind by a hard kill is
//                              reaped by the next -stop (the pid is probed first).
//   <temp>/ftrace/<pid>.stop   created by `ftrace -stop <pid>`. The target's watcher
//                              thread sees it within ~250 ms, deletes it, and raises
//                              the very same g_stopRequested flag Ctrl-C raises.
//                              Nothing is force-killed, ever.
static std::atomic<bool>     g_extStopRequested{false};  // also breaks the -keepwindow hold
static std::atomic<bool>     g_stopWatchQuit{false};
static std::thread           g_stopWatchThread;
static std::filesystem::path g_stopRunFile, g_stopSentinelFile;

// <temp>/ftrace, created on demand. Empty path = no usable temp dir (channel disabled).
static std::filesystem::path stopChannelDir() {
    std::error_code ec;
    std::filesystem::path d = std::filesystem::temp_directory_path(ec);
    if (ec) return {};
    d /= "ftrace";
    std::filesystem::create_directories(d, ec);
    return ec ? std::filesystem::path{} : d;
}

static long ftraceCurrentPid() {
#ifdef _WIN32
    return (long)GetCurrentProcessId();
#else
    return (long)getpid();
#endif
}

// Is that pid still alive? Only used to reap a .run file whose owner died hard, so a
// conservative "yes" off Windows just means stale entries linger in the -stop listing.
static bool ftraceProcessAlive(long pid) {
#ifdef _WIN32
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)pid);
    if (!h) return false;                                   // gone (or not ours to touch)
    bool alive = (WaitForSingleObject(h, 0) == WAIT_TIMEOUT);
    CloseHandle(h);
    return alive;
#else
    (void)pid; return true;
#endif
}

// Publish this process in the stop channel and start watching for its sentinel.
static void stopChannelStart(const std::string& what) {
    std::filesystem::path dir = stopChannelDir();
    if (dir.empty()) return;                     // no temp dir: run without the channel
    const long pid = ftraceCurrentPid();
    g_stopRunFile      = dir / (std::to_string(pid) + ".run");
    g_stopSentinelFile = dir / (std::to_string(pid) + ".stop");
    std::error_code ec;
    // A sentinel already sitting here belongs to a dead process whose pid we've been
    // recycled into; clear it so we don't stop the instant we start.
    std::filesystem::remove(g_stopSentinelFile, ec);
    { std::ofstream f(g_stopRunFile); f << what << "\n"; }
    g_stopWatchQuit.store(false);
    g_stopWatchThread = std::thread([] {
        while (!g_stopWatchQuit.load(std::memory_order_relaxed)) {
            std::error_code e;
            if (std::filesystem::exists(g_stopSentinelFile, e)) {
                std::filesystem::remove(g_stopSentinelFile, e);
                // Deliberately true whether a render is in flight (finish the chunk, write,
                // exit) or the process is just holding a -keepwindow preview open.
                std::printf("\n[stop] external stop requested — stopping cleanly "
                            "(any render in progress writes its image + checkpoint first).\n");
                std::fflush(stdout);
                g_extStopRequested.store(true);
                g_stopRequested = 1;   // the one flag every render loop already polls
                return;                // one-shot: never re-arm behind a later frame
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    });
}

static void stopChannelEnd() {
    g_stopWatchQuit.store(true);
    if (g_stopWatchThread.joinable()) g_stopWatchThread.join();
    std::error_code ec;
    if (!g_stopRunFile.empty()) std::filesystem::remove(g_stopRunFile, ec);
}

// `ftrace -stop [<pid>|all]`: list running renders, or ask one/all of them to finish
// cleanly. Never loads a scene, never touches the GPU; returns a process exit code.
static int runStopCommand(const char* who) {
    std::filesystem::path dir = stopChannelDir();
    if (dir.empty()) {
        std::fprintf(stderr, "error: -stop cannot reach the ftrace stop-channel directory\n");
        return 1;
    }
    // Every live render, reaping .run files whose owner is gone (hard kill / crash).
    std::vector<std::pair<long, std::string>> live;
    std::error_code ec;
    for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
        if (de.path().extension() != ".run") continue;
        const long pid = std::strtol(de.path().stem().string().c_str(), nullptr, 10);
        if (pid <= 0) continue;
        if (!ftraceProcessAlive(pid)) { std::filesystem::remove(de.path(), ec); continue; }
        std::string what;
        { std::ifstream f(de.path()); std::getline(f, what); }
        live.emplace_back(pid, what);
    }
    std::sort(live.begin(), live.end());

    if (!who) {                                   // bare -stop: just list what's running
        if (live.empty()) { std::printf("[stop] no ftrace renders are running.\n"); return 0; }
        std::printf("[stop] running renders — stop one with `ftrace -stop <pid>`, "
                    "all with `ftrace -stop all`:\n");
        for (const auto& p : live) std::printf("    pid %-7ld %s\n", p.first, p.second.c_str());
        return 0;
    }

    std::vector<long> targets;
    if (!std::strcmp(who, "all")) {
        for (const auto& p : live) targets.push_back(p.first);
        if (targets.empty()) { std::printf("[stop] no ftrace renders are running.\n"); return 0; }
    } else {
        char* end = nullptr;
        const long pid = std::strtol(who, &end, 10);
        if (!end || *end || pid <= 0) {
            std::fprintf(stderr, "error: -stop takes a pid or 'all' (got \"%s\")\n", who);
            return 1;
        }
        bool known = false;
        for (const auto& p : live) if (p.first == pid) known = true;
        if (!known)
            std::printf("[stop] warning: pid %ld isn't a running ftrace render "
                        "(dropping the sentinel anyway)\n", pid);
        targets.push_back(pid);
    }

    for (long pid : targets) {
        const std::filesystem::path s = dir / (std::to_string(pid) + ".stop");
        std::ofstream f(s);
        if (!f) { std::fprintf(stderr, "error: cannot write %s\n", s.string().c_str()); return 1; }
        f << "stop\n";
        std::printf("[stop] asked pid %ld to finish and exit cleanly.\n", pid);
    }
    std::fflush(stdout);

    // Wait for them to actually go. A render only notices at a chunk boundary and then
    // still has to write its image + checkpoint, so this can legitimately take up to
    // one -interval (default 15 s) plus the write; give it a generous ceiling and say
    // so rather than leaving the caller guessing whether the stop took.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    std::vector<long> pending = targets;
    while (!pending.empty() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        std::vector<long> still;
        for (long pid : pending)
            if (ftraceProcessAlive(pid) &&
                std::filesystem::exists(dir / (std::to_string(pid) + ".run"), ec))
                still.push_back(pid);
        pending.swap(still);
    }
    if (pending.empty()) { std::printf("[stop] done — stopped cleanly.\n"); return 0; }
    std::printf("[stop] still running after 120s:");
    for (long pid : pending) std::printf(" %ld", pid);
    std::printf("\n[stop] it may be mid-write, or in a long non-chunked batch "
                "(a bare -n render with no -window/-time/-noise budget writes only at the end).\n");
    return 2;
}

// --- Live preview window (-window) --------------------------------------------
// When enabled, the render drivers periodically push the current tone-mapped frame
// to a real OS window (Win32 GDI; no-op stub off Windows) so the image is watched as
// it converges, instead of only landing in a PNG. The window is created lazily on the
// first update at the film's resolution and torn down at process exit. Closing the
// window sets g_stopRequested so the render stops cleanly (writing its final image).
static bool                        g_showWindow = false;
// -keepwindow / -hold: don't auto-close the live preview when the render finishes.
// Normally g_liveWin is torn down at process exit (right after the render's last frame),
// so the window vanishes the instant rendering completes. With this set, main() blocks
// after run() returns until the user closes the window themselves, so a finished image
// stays on screen to inspect.
static bool                        g_keepWindow = false;
static std::unique_ptr<LiveWindow> g_liveWin;
// Base window title identifying WHAT is being rendered — set in main() to
// "ftrace - <scene> -> <output>" (see makeWindowTitle). The current render mode
// (g_windowMode, below) and the live status (spp / noise) are appended per frame so
// the title bar shows the subject, the transport mode, and its progress.
static std::string                 g_windowTitle = "ftrace live preview";
// Short label for the mode currently driving the live window, e.g. "mode B (pinhole)".
// Each render dispatch (runRender / runSharedGroup / runSharedPhotonMap) stamps it so a
// multi-camera flight with per-camera modes always shows the mode of the frame on screen.
static std::string                 g_windowMode;
// Human-readable name for a transport mode char (title bar + diagnostics).
static const char* modeLabel(char m) {
    switch (m) {
        case 'A': return "mode A (finite-lens)";
        case 'B': return "mode B (pinhole)";
        case 'C': return "mode C (aperture-catch)";
        case 'R': return "mode R (backward ref)";
        case 'V': return "mode V (validate)";
        case 'P': return "mode P (composite)";
        case 'D': return "mode D (BDPT)";
        case 'M': return "mode M (photon map)";
        case 'S': return "mode S (SPPM)";
        case 'U': return "mode U (VCM)";
        default:  return "";
    }
}
static void liveWindowUpdate(const Film& f, double N, double expComp, bool absolute,
                             const char* status = nullptr) {
    if (!g_showWindow || N <= 0.0) return;
    if (!g_liveWin)
        g_liveWin = std::make_unique<LiveWindow>(f.resX, f.resY, g_windowTitle.c_str());
    // Per-frame auto-expose (nullptr anchor) so the live view tracks the converging
    // image the same way the ANSI preview does.
    std::vector<uint8_t> rgb = filmToRgb8(f, N, expComp, absolute, nullptr);
    g_liveWin->update(f.resX, f.resY, rgb);
    // Reflect the render subject + mode + live progress in the title bar.
    std::string t = g_windowTitle;
    if (!g_windowMode.empty())     t += "  \xE2\x80\x94  " + g_windowMode;
    if (status && *status)         t += "  \xE2\x80\x94  " + std::string(status);
    g_liveWin->setTitle(t);
    if (g_liveWin->closed()) g_stopRequested = 1;
}

// --- Resumable-render checkpoint (.ftbuf sidecar) -----------------------------
// A forward render accumulates radiance photon-by-photon into a Film, so it can be
// stopped and continued: brightness scales with photon count and only graininess
// changes. The 8-bit tone-mapped image cannot be resumed from faithfully (it is
// exposure-anchored and gamma-quantised), so alongside `-o out.png` we persist the
// raw linear film + cumulative photon count + energy tally to `out.png.ftbuf`.
// `-resume` reloads it and keeps adding photons; a fresh render overwrites it.
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Film XYZ blob assumes packed Vec3");

struct Checkpoint {
    Film film;
    long long N = 0;          // cumulative photons already accumulated in `film`
    EnergyReport energy;      // cumulative energy tally
};

// A cheap identity hash so a resume refuses to blend photons from a different scene,
// mode, or resolution into the saved film (which would silently corrupt the result).
static uint64_t checkpointGuard(const Scene& scene, char mode, int res, int resY) {
    uint64_t h = 14695981039346656037ULL;                 // FNV-1a offset basis
    auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ULL; };
    mix((uint64_t)scene.tris.size());
    mix((uint64_t)scene.spheres.size());
    mix((uint64_t)scene.emitters.size());
    uint64_t tp; std::memcpy(&tp, &scene.totalPower, sizeof tp); mix(tp);
    mix((uint64_t)(unsigned char)mode);
    mix((uint64_t)(unsigned)res);
    mix((uint64_t)(unsigned)resY);
    return h;
}

static std::string checkpointPath(const std::string& outPath) { return outPath + ".ftbuf"; }

// "png/foo.png" + "_forward" -> "png/foo_forward.png". Inserts a suffix before the
// extension, keeping the directory (so a companion image lands next to -o rather than in
// the CWD) and the format (writeImage dispatches on the extension). No extension, or a
// dot that belongs to a directory component, appends instead.
static std::string pathWithSuffix(const std::string& path, const char* suffix) {
    size_t dot = path.find_last_of('.');
    size_t sep = path.find_last_of("/\\");
    if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
        return path + suffix;
    return path.substr(0, dot) + suffix + path.substr(dot);
}

static bool writeCheckpoint(const std::string& outPath, const Checkpoint& c,
                            uint64_t guard, char mode) {
    std::ofstream o(checkpointPath(outPath), std::ios::binary);
    if (!o) return false;
    const char magic[8] = {'F','T','B','U','F','0','1','\n'};
    int32_t rx = c.film.resX, ry = c.film.resY, m = (int32_t)(unsigned char)mode;
    o.write(magic, 8);
    o.write((const char*)&rx, 4); o.write((const char*)&ry, 4); o.write((const char*)&m, 4);
    o.write((const char*)&c.N, 8);
    double en[5] = {c.energy.emitted, c.energy.absorbed, c.energy.sensor,
                    c.energy.escaped, c.energy.residual};
    o.write((const char*)en, sizeof en);
    o.write((const char*)&guard, 8);
    o.write((const char*)c.film.xyz.data(), c.film.xyz.size() * sizeof(Vec3));
    o.write((const char*)c.film.hits.data(), c.film.hits.size() * sizeof(double));
    return (bool)o;
}

// Load a checkpoint for resume. Returns false (leaving `c` untouched) if the sidecar
// is missing, malformed, or its identity guard/resolution disagrees with this render
// (a clear message is printed for the mismatch cases so a stale file never silently
// poisons the image).
static bool readCheckpoint(const std::string& outPath, int res, int resY, uint64_t guard,
                           char mode, Checkpoint& c) {
    std::ifstream in(checkpointPath(outPath), std::ios::binary);
    if (!in) return false;
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "FTBUF01\n", 8) != 0) {
        std::fprintf(stderr, "[resume] %s is not a recognised checkpoint; ignoring\n",
                     checkpointPath(outPath).c_str());
        return false;
    }
    int32_t rx = 0, ry = 0, m = 0; long long N = 0;
    in.read((char*)&rx, 4); in.read((char*)&ry, 4); in.read((char*)&m, 4);
    in.read((char*)&N, 8);
    double en[5] = {0,0,0,0,0}; in.read((char*)en, sizeof en);
    uint64_t g = 0; in.read((char*)&g, 8);
    if (!in) return false;
    if (rx != res || ry != resY || m != (int32_t)(unsigned char)mode || g != guard) {
        std::fprintf(stderr, "[resume] checkpoint %s does not match this render "
                             "(scene/mode/resolution differ); starting fresh\n",
                     checkpointPath(outPath).c_str());
        return false;
    }
    c.film.resX = rx; c.film.resY = ry; c.film.alloc();
    c.N = N;
    c.energy = {en[0], en[1], en[2], en[3], en[4]};
    in.read((char*)c.film.xyz.data(), c.film.xyz.size() * sizeof(Vec3));
    in.read((char*)c.film.hits.data(), c.film.hits.size() * sizeof(double));
    if (!in) { std::fprintf(stderr, "[resume] checkpoint %s truncated; starting fresh\n",
                            checkpointPath(outPath).c_str()); return false; }
    return true;
}

// --- Mode-P composite checkpoint (dual-film .ftbuf sidecar) --------------------
// Mode P blends TWO accumulators — a forward SUM film (over N photons) and a backward
// SUM film (over spp) — so its resumable sidecar stores both, plus their two counts.
// The per-pixel classification is view-dependent and cheap, so it is recomputed on
// resume rather than serialized. Magic "FTPCM02\n" distinguishes it from the single-film
// "FTBUF01\n" format so a stale/mismatched sidecar can never be misread.
struct CompositeCheckpoint {
    Film fwd;                 // SUM over N photons (forward model-B layer)
    Film ref;                 // SUM over spp (backward camera-side layer)
    long long N = 0;          // cumulative forward photons in `fwd`
    long long spp = 0;        // cumulative backward samples in `ref`
    EnergyReport energy;      // cumulative forward energy tally
};

static bool writeCompositeCheckpoint(const std::string& outPath, const CompositeCheckpoint& c,
                                     uint64_t guard) {
    std::ofstream o(checkpointPath(outPath), std::ios::binary);
    if (!o) return false;
    const char magic[8] = {'F','T','P','C','M','0','2','\n'};
    int32_t rx = c.fwd.resX, ry = c.fwd.resY;
    o.write(magic, 8);
    o.write((const char*)&rx, 4); o.write((const char*)&ry, 4);
    o.write((const char*)&c.N, 8); o.write((const char*)&c.spp, 8);
    double en[5] = {c.energy.emitted, c.energy.absorbed, c.energy.sensor,
                    c.energy.escaped, c.energy.residual};
    o.write((const char*)en, sizeof en);
    o.write((const char*)&guard, 8);
    o.write((const char*)c.fwd.xyz.data(), c.fwd.xyz.size() * sizeof(Vec3));
    o.write((const char*)c.fwd.hits.data(), c.fwd.hits.size() * sizeof(double));
    o.write((const char*)c.ref.xyz.data(), c.ref.xyz.size() * sizeof(Vec3));
    o.write((const char*)c.ref.hits.data(), c.ref.hits.size() * sizeof(double));
    return (bool)o;
}

static bool readCompositeCheckpoint(const std::string& outPath, int res, int resY,
                                    uint64_t guard, CompositeCheckpoint& c) {
    std::ifstream in(checkpointPath(outPath), std::ios::binary);
    if (!in) return false;
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "FTPCM02\n", 8) != 0) {
        std::fprintf(stderr, "[resume] %s is not a recognised composite checkpoint; ignoring\n",
                     checkpointPath(outPath).c_str());
        return false;
    }
    int32_t rx = 0, ry = 0; long long N = 0, spp = 0;
    in.read((char*)&rx, 4); in.read((char*)&ry, 4);
    in.read((char*)&N, 8); in.read((char*)&spp, 8);
    double en[5] = {0,0,0,0,0}; in.read((char*)en, sizeof en);
    uint64_t g = 0; in.read((char*)&g, 8);
    if (!in) return false;
    if (rx != res || ry != resY || g != guard) {
        std::fprintf(stderr, "[resume] composite checkpoint %s does not match this render "
                             "(scene/resolution differ); starting fresh\n",
                     checkpointPath(outPath).c_str());
        return false;
    }
    c.fwd.resX = rx; c.fwd.resY = ry; c.fwd.alloc();
    c.ref.resX = rx; c.ref.resY = ry; c.ref.alloc();
    c.N = N; c.spp = spp;
    c.energy = {en[0], en[1], en[2], en[3], en[4]};
    in.read((char*)c.fwd.xyz.data(), c.fwd.xyz.size() * sizeof(Vec3));
    in.read((char*)c.fwd.hits.data(), c.fwd.hits.size() * sizeof(double));
    in.read((char*)c.ref.xyz.data(), c.ref.xyz.size() * sizeof(Vec3));
    in.read((char*)c.ref.hits.data(), c.ref.hits.size() * sizeof(double));
    if (!in) { std::fprintf(stderr, "[resume] composite checkpoint %s truncated; starting fresh\n",
                            checkpointPath(outPath).c_str()); return false; }
    return true;
}

// --- Standalone artifact -> PNG conversion (`-topng`) --------------------------
// Turn an existing render artifact into a 24-bit PNG without re-rendering, so the
// ppm/ outputs and *.ftbuf resume checkpoints can be shared as PNGs with the same
// binary. Dispatched from main() before any scene/CLI setup, so it is a pure,
// dependency-free utility path. Handles:
//   * .ppm  — binary P6, 8-bit — re-encoded as PNG (a lossless RGB copy).
//   * .ftbuf — the raw linear film checkpoint — loaded and tone-mapped to PNG. The
//     sidecar does not persist the exposure mode, so this uses the same p99 auto-
//     exposure as a non-absolute render; an absolute (power/lumens) scene may look
//     brighter/darker than its original -o image. Re-run the render for an
//     exposure-exact PNG. A trailing `-ev <c>` scales that auto-exposure, which is
//     how you re-develop a finished render brighter without paying for it again.
//   * .ftsl — NOT handled here (it is a scene, not an image): render it with -in.

// Read a binary P6 (8-bit) PPM into a top-row-first RGB byte buffer. Returns false
// for ASCII (P3), non-8-bit (maxval != 255) or malformed files.
static bool readBinaryPPM(const std::string& path, int& W, int& H,
                          std::vector<uint8_t>& rgb) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char m0 = 0, m1 = 0;
    in.get(m0); in.get(m1);
    if (!in || m0 != 'P' || m1 != '6') return false;
    // Read an unsigned int, skipping leading whitespace and '#' comments (PPM grammar).
    auto readUInt = [&](int& v) -> bool {
        int c;
        for (;;) {
            c = in.get();
            if (c == EOF) return false;
            if (c == '#') { while ((c = in.get()) != EOF && c != '\n') {} continue; }
            if (std::isspace((unsigned char)c)) continue;
            break;
        }
        if (!std::isdigit((unsigned char)c)) return false;
        v = 0;
        do { v = v * 10 + (c - '0'); c = in.get(); }
        while (c != EOF && std::isdigit((unsigned char)c));
        return true;   // the one trailing non-digit byte consumed here is the single
                       // whitespace separator before the pixel data (P6 convention).
    };
    int maxv = 0;
    if (!readUInt(W) || !readUInt(H) || !readUInt(maxv)) return false;
    if (W <= 0 || H <= 0 || maxv != 255) return false;   // only 8-bit binary PPM
    rgb.assign((size_t)W * H * 3, 0);
    in.read((char*)rgb.data(), (std::streamsize)rgb.size());
    return (bool)in && in.gcount() == (std::streamsize)rgb.size();
}

// `expComp` is the -exposure/-ev multiplier applied on top of the auto-exposure
// (<= 0 means "plain auto"). It only affects a .ftbuf, whose linear film is still
// tone-mapped here; a .ppm is already 8-bit sRGB and is copied through verbatim.
static int convertToPng(const std::string& inPath, const std::string& outPath,
                        double expComp = 0.0) {
    if (endsWithCI(inPath, ".ppm")) {
        int W = 0, H = 0; std::vector<uint8_t> rgb;
        if (!readBinaryPPM(inPath, W, H, rgb)) {
            std::fprintf(stderr, "error: %s is not a readable binary (P6) 8-bit PPM\n",
                         inPath.c_str());
            return 1;
        }
        if (!writeImage(outPath, W, H, rgb)) return 1;
        std::printf("converted %s -> %s (%dx%d, 24-bit RGB)\n",
                    inPath.c_str(), outPath.c_str(), W, H);
        return 0;
    }
    if (endsWithCI(inPath, ".ftbuf")) {
        std::ifstream in(inPath, std::ios::binary);
        if (!in) { std::fprintf(stderr, "error: cannot open %s\n", inPath.c_str()); return 1; }
        char magic[8]; in.read(magic, 8);
        if (!in || std::memcmp(magic, "FTBUF01\n", 8) != 0) {
            std::fprintf(stderr, "error: %s is not a recognised FTBUF checkpoint\n",
                         inPath.c_str());
            return 1;
        }
        int32_t rx = 0, ry = 0, m = 0; long long Nph = 0;
        in.read((char*)&rx, 4); in.read((char*)&ry, 4); in.read((char*)&m, 4);
        in.read((char*)&Nph, 8);
        double en[5]; in.read((char*)en, sizeof en);
        uint64_t g = 0; in.read((char*)&g, 8);
        if (!in || rx <= 0 || ry <= 0) {
            std::fprintf(stderr, "error: %s header malformed\n", inPath.c_str());
            return 1;
        }
        Film f; f.resX = rx; f.resY = ry; f.alloc();
        in.read((char*)f.xyz.data(),  (std::streamsize)(f.xyz.size()  * sizeof(Vec3)));
        in.read((char*)f.hits.data(), (std::streamsize)(f.hits.size() * sizeof(double)));
        if (!in) { std::fprintf(stderr, "error: %s truncated\n", inPath.c_str()); return 1; }
        // Tone-map with the p99 auto-exposure (see note above), scaled by -ev if given.
        // writeFilm prints the "wrote <out> ..." line, including the comp when != 1.
        return writeFilm(outPath.c_str(), f, (double)std::max<long long>(Nph, 1), expComp) ? 0 : 1;
    }
    std::fprintf(stderr,
        "error: -topng converts .ppm and .ftbuf inputs; got '%s'.\n"
        "       A .ftsl is a scene, not an image — render it with: "
        "ftrace -in scene.ftsl -o out.png\n", inPath.c_str());
    return 1;
}

// Is anything in this scene outside BDPT's (mode D) transport scope? Returns a human
// description of the first unsupported feature, or nullptr if the scene is fully BDPT-
// renderable. BDPT here covers Lambertian/glossy scatter + quad/sphere area emission
// only; fluorescence, participating media, spot/env/collimated lights and the layered
// stack would silently drop their contribution, so the caller refuses (mode D) or
// routes elsewhere (mode P with a lens falls back to mode R). Only materials actually
// referenced by geometry are flagged (built-in palettes carry spare unused entries);
// Mix children are expanded since a used Mix can pick e.g. a fluorescent child.
static const char* bdptUnsupportedFeature(const Scene& scene) {
    // Participating media — homogeneous AND heterogeneous (density-field / bounded) — are
    // supported by volumetric BDPT. Subpath medium vertices are placed by delta (Woodcock)
    // tracking with analog throughput and connection edges are weighted by ratio-tracking
    // transmittance (both unbiased estimators, exactly as the forward tracer does). The MIS
    // weights omit the heterogeneous distance-pdf / transmittance — a variance-only
    // simplification (PBRT-v3 convention): the balance heuristic is a partition of unity for
    // any consistent pdfs, so the estimator stays unbiased regardless (only the sampled
    // strategy's throughput must be exact, which analog + ratio tracking guarantee).
    //
    // GRADIENT-INDEX (GRIN) media are the one exception: they bend rays along curved
    // Eikonal paths, which breaks BDPT's straight-edge assumptions (the geometric term G,
    // area-measure pdf conversion and MIS all assume the connecting segment is a line). The
    // forward (A/B/C) and backward (R) tracers march GRIN correctly; BDPT would need curved
    // connections to stay unbiased, so we refuse GRIN scenes here rather than ship a subtly
    // wrong image. (See known-issues.md: curved-path BDPT is a future enhancement.)
    for (const auto& md : scene.media)
        if (md.enabled && md.grin())
            return "gradient-index (GRIN) media (use mode A/B/C or R)";

    std::vector<char> matUsed(scene.mats.size(), 0);
    // Mark a material and (one level, since Mix children can't themselves be Mix) its
    // Mix children, which a used Mix can pick at runtime.
    auto markUsed = [&](int id) {
        if (id < 0 || id >= (int)scene.mats.size() || matUsed[id]) return;
        matUsed[id] = 1;
        if (scene.mats[id].type == MatType::Mix)
            for (int c : scene.mats[id].mixChildren)
                if (c >= 0 && c < (int)scene.mats.size()) matUsed[c] = 1;
    };
    for (const auto& tr : scene.tris) markUsed(tr.matId);
    for (const auto& sp : scene.spheres) markUsed(sp.matId);
    for (size_t i = 0; i < scene.mats.size(); ++i)
        if (matUsed[i] && scene.mats[i].type == MatType::Fluorescent)
            return "fluorescent materials";
    for (size_t i = 0; i < scene.mats.size(); ++i)
        if (matUsed[i] && scene.mats[i].type == MatType::Layered)
            return "layered materials";
    for (const auto& em : scene.emitters)
        if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env ||
            em.shape == EmitterShape::Sun || em.collimated)
            return "spot / environment / sun / collimated lights";
    return nullptr;
}

// VCM (mode U) scope guard. VCM reuses BDPT's transport scope but ADDITIONALLY excludes
// participating media (surfaces-only merging in this build) and the fisheye/lensed cameras
// (its camera importance assumes the rectilinear pinhole). Returns a reason string or null.
static const char* vcmUnsupportedFeature(const Scene& scene, const Camera& cam) {
    if (const char* r = bdptUnsupportedFeature(scene)) return r;
    if (!scene.media.empty()) return "participating media (mode U is surfaces-only)";
    if (cam.hasLens()) return "a realistic multi-element lens";
    if (cam.projection != CAM_RECTILINEAR) return "a non-rectilinear (fisheye/panoramic) camera";
    return nullptr;
}

// --- Unsupported-feature POLICY (`-on-unsupported`) + the `prefer{}/else{}` predicate ---
// A scene may ask a mode to render something it can't (e.g. GRIN media in mode D). The
// policy decides what happens: error out (default, historical), fall back to a mode that
// CAN render it (backward reference R), or strip the offending feature and render anyway.
enum class OnUnsupported { Error, Fallback, Strip };
static OnUnsupported g_onUnsupported = OnUnsupported::Error;

// Core capability check: return a reason string if `mode` cannot render `scene` with a
// camera of the given `projection`, else nullptr. Only modes with real restrictions (D
// BDPT, U VCM) gate anything; the general modes (A/B/C/R/M/S/P) render everything here.
static const char* modeFeatureUnsupported(const Scene& scene, char mode, int projection) {
    if (mode == 'D') {
        if (const char* r = bdptUnsupportedFeature(scene)) return r;
        if (projection != CAM_RECTILINEAR)
            return "a non-rectilinear (fisheye/panoramic) camera in mode D";
    } else if (mode == 'U') {
        if (const char* r = bdptUnsupportedFeature(scene)) return r;
        if (!scene.media.empty()) return "participating media (mode U is surfaces-only)";
        if (projection != CAM_RECTILINEAR)
            return "a non-rectilinear (fisheye/panoramic) camera in mode U";
    }
    return nullptr;
}

// `prefer{}/else{}` predicate handed to ftsl::load: a branch is renderable iff EVERY
// camera it declares can render the scene in its effective mode. `cliMode` (0 = none)
// is a `-mode` override that forces the mode for all cameras.
static const char* sceneModeUnsupported(const ftsl::Loaded& L, char cliMode) {
    auto effOf = [&](char camMode) -> char {
        if (cliMode) return cliMode;
        if (camMode) return camMode;
        return L.mode ? L.mode : 'B';
    };
    if (!L.cameras.empty()) {
        for (const auto& cs : L.cameras)
            if (const char* r = modeFeatureUnsupported(L.scene, effOf(cs.mode), cs.projection))
                return r;
    } else {
        if (const char* r = modeFeatureUnsupported(L.scene, effOf(0), CAM_RECTILINEAR))
            return r;
    }
    return nullptr;
}

// Best-effort feature stripping for `-on-unsupported strip`. Today only GRIN is
// strippable (clear the index field -> the medium is treated as homogeneous/clear).
// Mutates `scene` only if that fully resolves the conflict; otherwise restores it and
// returns false (so the caller falls back to a supported mode instead). Returns true iff
// the scene now renders in `mode`.
static bool stripUnsupportedFeature(Scene& scene, char mode, int projection) {
    if (modeFeatureUnsupported(scene, mode, projection) == nullptr) return true;
    std::vector<Medium> saved = scene.media;
    bool anyGrin = false;
    for (auto& m : scene.media) if (m.grin()) { m.ior.clear(); m.iorStep = 0.0; anyGrin = true; }
    if (anyGrin && modeFeatureUnsupported(scene, mode, projection) == nullptr) return true;
    scene.media = saved;   // couldn't fully strip -> leave the scene intact
    return false;
}

// Apply the `-on-unsupported` policy for one camera. Returns the (possibly changed) mode;
// may mutate `scene` (strip). `proceed` is set false only when Error policy should abort
// this camera's render. Prints a notice describing what happened.
static char applyUnsupportedPolicy(Scene& scene, char mode, int projection,
                                   const char* camName, bool& proceed) {
    proceed = true;
    const char* why = modeFeatureUnsupported(scene, mode, projection);
    if (!why) return mode;
    if (g_onUnsupported == OnUnsupported::Error) {
        std::fprintf(stderr, "[mode %c] camera '%s' uses %s, which that mode can't render; "
                             "use mode A/B/C/R, add a prefer{}/else{} fallback, or pass "
                             "-on-unsupported fallback|strip.\n", mode, camName, why);
        proceed = false;
        return mode;
    }
    if (g_onUnsupported == OnUnsupported::Strip &&
        stripUnsupportedFeature(scene, mode, projection)) {
        std::printf("[on-unsupported=strip] camera '%s': stripped %s; rendering in mode %c "
                    "anyway.\n", camName, why, mode);
        return mode;
    }
    // Fallback (or strip couldn't resolve it): the backward reference (R) renders every
    // feature BDPT/VCM refuse (GRIN, media, fisheye, fluorescence, ...).
    std::printf("[on-unsupported=fallback] camera '%s': %s unsupported in mode %c -> mode R "
                "(backward reference).\n", camName, why, mode);
    return 'R';
}

// CPU counterpart of the GPU gpuSppChunks helper: render `sppTarget` samples-per-pixel
// in adaptive chunks so a CPU mode-R/D render gets the same live progress as the GPU.
// `renderOne(chunkSpp, seedOffset)` renders one chunk and returns its SUM film; the
// chunks are merged into a running SUM and reported after each. A null/empty prog does
// the historical single-shot render. Returns the accumulated SUM film (writeFilm divides
// by the completed spp). Chunk size adapts toward ~0.4s so early frames appear quickly
// and the per-chunk thread-spawn overhead stays negligible.
static Film cpuSppChunks(long long sppTarget, const SppProgress* prog, int resX, int resY,
                         const std::function<Film(long long, unsigned long long)>& renderOne) {
    if (!prog || !prog->report) return renderOne(sppTarget, 0);
    using clk = std::chrono::steady_clock;
    // On resume the loaded film already holds `sampleBase` spp; continue from that
    // absolute sample index. renderOne(c, base) renders the absolute samples
    // [base, base+c) — per-(pixel,sample) seeding downstream makes the realization
    // identical no matter how this loop happens to split the chunks.
    const unsigned long long seedBias = (unsigned long long)prog->sampleBase;
    Film acc; acc.resX = resX; acc.resY = resY; acc.alloc();
    long long done = 0, chunk = 1;
    // Debug aids (determinism triage): FTRACE_CHUNK_SPP=K pins every chunk to K
    // samples (making the normally wall-clock-adaptive split sequence exactly
    // reproducible); FTRACE_CHUNK_DEBUG=1 logs the sequence actually used.
    long long forcedChunk = 0;
    if (const char* e = std::getenv("FTRACE_CHUNK_SPP")) forcedChunk = std::atoll(e);
    if (forcedChunk > 0) chunk = forcedChunk;
    const bool chunkDebug = std::getenv("FTRACE_CHUNK_DEBUG") != nullptr;
    while (done < sppTarget) {
        long long c = chunk; if (c > sppTarget - done) c = sppTarget - done;
        auto t0 = clk::now();
        Film f = renderOne(c, seedBias + (unsigned long long)done);
        acc.merge(f);
        done += c;
        double dt = std::chrono::duration<double>(clk::now() - t0).count();
        if (chunkDebug) std::fprintf(stderr, "[chunk] c=%lld base=%llu dt=%.3f\n",
                                     c, seedBias + (unsigned long long)(done - c), dt);
        if (forcedChunk <= 0 && dt > 1e-4) {   // retarget chunk toward ~0.4s of work
            long long next = (long long)((double)c * (0.4 / dt));
            if (next < 1) next = 1;
            if (next > c * 8 + 1) next = c * 8 + 1;   // ramp up gently
            chunk = next;
        }
        if (prog->report(acc, done, done >= sppTarget)) break;
    }
    return acc;
}

// Unified progress driver for the samples-per-pixel image modes (R backward reference,
// D bidirectional). `renderChunked(sppTarget, prog)` is the mode-specific renderer that
// accumulates a SUM film in chunks and calls prog->report() after each; this function
// supplies that callback so every mode gets the SAME live progress as the forward camera
// models: a periodic image rewrite (crash-safe), a status line (or -preview ANSI
// thumbnail) every `intervalSec`, a ~noise% estimate, and clean Ctrl-C / -time / -noise /
// -forever stopping. `sppReq` is the requested spp; when a time/noise/forever budget is
// set the target is opened up (UNBOUNDED_SPP) and the stop is driven by the budget. The
// display divides the SUM film by the spp completed, so brightness is constant and only
// graininess falls as more samples land. Returns the process exit code (0 ok, 1 on a
// write failure).
static int runSppProgressive(
        const std::string& outPath, long long sppReq,
        double manualExposure, double* exposureAnchor, bool absolute,
        double timeBudgetSec, double noiseTarget, bool runForever,
        double intervalSec, bool preview,
        const std::function<Film(long long, const SppProgress*)>& renderChunked,
        int res = 0, int resY = 0,
        bool resume = false, bool wantCheckpoint = false,
        uint64_t guard = 0, char mode = '?') {
    using clk = std::chrono::steady_clock;
    // A time/noise/forever budget renders "until the budget", so open the spp target to a
    // large-but-safe cap (keeps pixel*sppTotal seed indices well inside int64). A plain
    // fixed-spp render just targets sppReq and still shows progress along the way.
    const long long UNBOUNDED_SPP = 1'000'000'000LL;
    const bool budgeted = (timeBudgetSec > 0.0 || noiseTarget > 0.0 || runForever);
    const long long sppTarget = budgeted ? UNBOUNDED_SPP : sppReq;
    if (intervalSec <= 0.0) intervalSec = 15.0;

    // --- disk resume (mode R / D): reload a saved SUM film + spp count from the .ftbuf
    // sidecar and keep adding samples on top. The freshly-traced samples are biased past
    // the loaded ones (prog.sampleBase) so they form an independent realization; the
    // display/checkpoint always uses the COMBINED film (base + fresh) and the COMBINED
    // spp, so brightness is constant and only graininess falls (exactly like A/B/C).
    Checkpoint base;
    long long baseSpp = 0;
    if (resume && readCheckpoint(outPath, res, resY, guard, mode, base)) {
        baseSpp = base.N;
        std::printf("[resume] loaded %s: %lld spp accumulated so far\n",
                    checkpointPath(outPath).c_str(), baseSpp);
    }
    const bool haveBase = baseSpp > 0 && base.film.resX == res && base.film.resY == resY;

    if (preview) { enableAnsiTerminal(); g_previewRows = 0; }
    // Trap Ctrl-C (and Windows Ctrl-Break) so a long render stops cleanly with the
    // accumulated image saved, instead of losing everything since the last periodic write.
    auto prev = std::signal(SIGINT, onInterrupt);
#ifdef SIGBREAK
    auto prevBrk = std::signal(SIGBREAK, onInterrupt);
#endif

    const auto t0 = clk::now();
    auto lastSave = t0;
    bool writeOk = true;
    bool metNoise = false;
    long long finalSpp = 0;

    SppProgress prog;
    prog.sampleBase = baseSpp;   // bias fresh seeds past the loaded samples
    prog.report = [&](const Film& film, long long sppDone, bool final) -> bool {
        long long totalSpp = baseSpp + sppDone;
        finalSpp = totalSpp;
        double elapsed   = std::chrono::duration<double>(clk::now() - t0).count();
        double sinceSave = std::chrono::duration<double>(clk::now() - lastSave).count();
        // Every pixel receives exactly totalSpp samples, so the Monte-Carlo relative error
        // ~ 1/sqrt(samples) gives an honest graininess ballpark straight from the count.
        double noisePct = totalSpp > 0 ? 100.0 / std::sqrt((double)totalSpp) : 0.0;
        // Mode W has no Monte-Carlo noise to report (every estimator is a fixed
        // quadrature), so quoting 1/sqrt(spp) there would be pure fiction; spp only
        // buys antialiasing and spectral resolution. Say so instead.
        char nz[32];
        if (g_whitted) std::snprintf(nz, sizeof nz, "deterministic");
        else           std::snprintf(nz, sizeof nz, "~%.2f%% noise", noisePct);
        bool stopped  = g_stopRequested != 0;
        bool timeUp   = (!runForever && timeBudgetSec > 0.0 && elapsed >= timeBudgetSec);
        bool noiseMet = (!g_whitted && noiseTarget > 0.0 && totalSpp > 0 && noisePct <= noiseTarget);
        if (noiseMet) metNoise = true;
        bool stop = stopped || timeUp || noiseMet;
        bool done = stop || final;
        if (done || sinceSave >= intervalSec) {
            // Combine the loaded base film (if resuming) with the fresh SUM before display.
            const Film* shown = &film;
            Film combined;
            if (haveBase) { combined = film; combined.merge(base.film); shown = &combined; }
            // The converged/stopping frame owns the exposure anchor; intermediate frames
            // auto-expose independently (they only refine, never lock the anchor).
            writeOk = writeFilm(outPath.c_str(), *shown, (double)totalSpp, manualExposure,
                                /*quiet*/preview, done ? exposureAnchor : nullptr, absolute);
            if (wantCheckpoint) {
                Checkpoint save; save.film = *shown; save.N = totalSpp;
                if (!writeCheckpoint(outPath, save, guard, mode))
                    std::fprintf(stderr, "[checkpoint] could not write %s\n",
                                 checkpointPath(outPath).c_str());
            }
            lastSave = clk::now();
            const char* why = stopped ? " (stopping)" : noiseMet ? " (noise target met)" : "";
            char st[220];
            if (runForever)
                std::snprintf(st, sizeof st, "[forever] %.1fs, %lld spp, %s%s",
                              elapsed, totalSpp, nz, why);
            else if (timeBudgetSec > 0.0)
                std::snprintf(st, sizeof st, "[time] %.1fs / %.3gs, %lld spp, %s%s",
                              elapsed, timeBudgetSec, totalSpp, nz, why);
            else if (noiseTarget > 0.0)
                std::snprintf(st, sizeof st, "[noise] target ~%.2g%%, %.1fs, %lld spp, %s%s",
                              noiseTarget, elapsed, totalSpp, nz, why);
            else
                std::snprintf(st, sizeof st, "[spp] %lld / %lld, %.1fs, %s",
                              totalSpp, baseSpp + sppReq, elapsed, nz);
            if (preview) ansiPreview(*shown, (double)totalSpp, manualExposure, st);
            else { std::printf("%s\n", st); std::fflush(stdout); }
            liveWindowUpdate(*shown, (double)totalSpp, manualExposure, absolute, st);
        }
        return stop;
    };

    renderChunked(sppTarget, &prog);

    std::signal(SIGINT, prev);
#ifdef SIGBREAK
    std::signal(SIGBREAK, prevBrk);
#endif
    if (g_stopRequested)
        std::printf("\n[stop] interrupted at %lld spp — image saved.\n", finalSpp);
    else if (metNoise)
        std::printf("[noise] reached the ~%.2g%% target at %lld spp — image saved.\n",
                    noiseTarget, finalSpp);
    if (wantCheckpoint)
        std::printf("[checkpoint] %s holds %lld spp — rerun with -resume to add more\n",
                    checkpointPath(outPath).c_str(), finalSpp);
    return writeOk ? 0 : 1;
}

// Progressive driver for mode P (forward + camera-side composite). Grows BOTH the forward
// SUM film (over photons, seedBase = cumulative photons) and the backward SUM film (over
// spp, decorrelated per batch) in lockstep at the requested N:spp ratio, re-compositing
// and rewriting the image every `intervalSec` so the render is watchable and crash-safe —
// the same live-progress + .ftbuf resume the forward camera models A/B/C already have,
// but over the composite's two accumulators (dual-film sidecar). Handles -time / -noise /
// -forever budgets, a fixed N/spp target, Ctrl-C, -window / -preview, and -resume.
static int runCompositeProgressive(
        const Scene& scene, const Camera& cam, int res, int resY,
        long long N, long long spp, int nThreads, bool diffraction, bool useGpu, bool wavefront,
        const std::string& outPath, double manualExposure, double* exposureAnchor,
        double timeBudgetSec, double noiseTarget, bool runForever, double intervalSec,
        bool preview, bool resume, bool wantCheckpoint, uint64_t guard) {
    using clk = std::chrono::steady_clock;
    if (intervalSec <= 0.0) intervalSec = 15.0;
    const bool absolute = scene.absolute;
    const bool envScene = scene.envIndex >= 0;
    const bool budgeted = timeBudgetSec > 0.0 || noiseTarget > 0.0 || runForever;
    // Photons traced per backward sample-per-pixel, so the two halves grow at the ratio the
    // user requested (N photons alongside spp samples). Defaults keep both halves nonzero.
    const long long Nreq   = (N   > 0) ? N   : 2'000'000;
    const long long sppReq = (spp > 0) ? spp : 64;
    const double perSpp = (double)Nreq / (double)sppReq;
#ifdef HAVE_CUDA
    const bool gpuBackward = useGpu && cudaBackwardSupported(scene, cam);
#else
    const bool gpuBackward = false;
#endif

    // View-dependent first-hit classification: computed once, reused every pass.
    CompositeClass cc = classifyComposite(scene, cam, res, resY);

    CompositeCheckpoint acc;
    acc.fwd.resX = res; acc.fwd.resY = resY; acc.fwd.alloc();
    acc.ref.resX = res; acc.ref.resY = resY; acc.ref.alloc();
    if (resume && readCompositeCheckpoint(outPath, res, resY, guard, acc))
        std::printf("[resume] loaded %s: %lld photons + %lld spp accumulated so far\n",
                    checkpointPath(outPath).c_str(), acc.N, acc.spp);

    std::printf("mode P: forward+camera-side composite, target %lld photons / %lld spp "
                "at %dx%d on %s (light=%s)%s ...\n",
                Nreq, sppReq, res, resY,
                useGpu ? "GPU" : (std::to_string(nThreads) + " threads").c_str(),
                scene.envIndex >= 0 ? "env" : "lit",
                (resume && (acc.N > 0 || acc.spp > 0)) ? " [resuming]" : "");

    if (preview) { enableAnsiTerminal(); g_previewRows = 0; }
    auto prev = std::signal(SIGINT, onInterrupt);
#ifdef SIGBREAK
    auto prevBrk = std::signal(SIGBREAK, onInterrupt);
#endif

    const auto t0 = clk::now();
    auto lastSave = t0;
    bool writeOk = true;
    bool metNoise = false;
    long long batchSpp = 1;   // adapts toward ~0.5 s of combined work per iteration

    auto writeOut = [&](bool done) {
        Film comp = compositeFromFilms(acc.fwd, std::max(acc.N, 1LL), acc.ref,
                                       std::max(acc.spp, 1LL), cc, envScene, /*verbose*/done);
        writeOk = writeFilm(outPath.c_str(), comp, 1.0, manualExposure, /*quiet*/preview,
                            done ? exposureAnchor : nullptr, absolute);
        if (wantCheckpoint && !writeCompositeCheckpoint(outPath, acc, guard))
            std::fprintf(stderr, "[checkpoint] could not write %s\n",
                         checkpointPath(outPath).c_str());
        return comp;
    };

    for (;;) {
        long long dSpp = batchSpp;
        long long dN   = std::max(1LL, (long long)std::llround((double)batchSpp * perSpp));
        if (!budgeted) {                              // cap each half to its remaining budget
            long long remSpp = sppReq - acc.spp; if (remSpp < 0) remSpp = 0;
            long long remN   = Nreq   - acc.N;   if (remN   < 0) remN   = 0;
            dSpp = std::min(dSpp, remSpp);
            dN   = std::min(dN,   remN);
            if (dSpp == 0 && dN == 0) { writeOut(/*done*/true); break; }  // both budgets met
        }
        auto tb = clk::now();
        if (dN > 0) {   // forward model-B layer (seedBase = cumulative photons, like A/B/C)
            EnergyReport e;
            Film f = renderForward(scene, &cam, res, resY, dN, nThreads,
                                   /*forwardCatch*/false, /*lensMode*/false, /*useCamera*/true,
                                   e, diffraction, useGpu, (uint64_t)acc.N, wavefront);
            acc.fwd.merge(f); acc.N += dN;
            acc.energy.emitted += e.emitted; acc.energy.absorbed += e.absorbed;
            acc.energy.sensor  += e.sensor;  acc.energy.escaped  += e.escaped;
            acc.energy.residual += e.residual;
        }
        if (dSpp > 0) {  // backward camera-side layer, decorrelated per batch by acc.spp
            Film r;
#ifdef HAVE_CUDA
            if (gpuBackward) {
                SppProgress bp; bp.sampleBase = acc.spp;   // mixes into the device seed
                bp.report = [](const Film&, long long, bool) { return false; };
                r = renderBackwardCuda(scene, cam, res, resY, dSpp, diffraction, &bp,
                                       g_maxBounceOverride, g_directOnly, g_heroC);
            } else
#endif
                r = renderBackward(scene, cam, res, resY, dSpp, nThreads, diffraction,
                                   (uint64_t)acc.spp);
            acc.ref.merge(r); acc.spp += dSpp;
        }
        // Adapt the batch toward ~0.5 s so early frames appear fast and overhead stays low.
        double dt = std::chrono::duration<double>(clk::now() - tb).count();
        if (dt > 1e-4) {
            long long next = (long long)((double)batchSpp * (0.5 / dt));
            if (next < 1) next = 1;
            if (next > batchSpp * 8 + 1) next = batchSpp * 8 + 1;
            batchSpp = next;
        }

        double elapsed   = std::chrono::duration<double>(clk::now() - t0).count();
        double sinceSave = std::chrono::duration<double>(clk::now() - lastSave).count();
        double noisePct  = acc.spp > 0 ? 100.0 / std::sqrt((double)acc.spp) : 0.0;
        bool stopped  = g_stopRequested != 0;
        bool timeUp   = (!runForever && timeBudgetSec > 0.0 && elapsed >= timeBudgetSec);
        bool noiseMet = (noiseTarget > 0.0 && acc.spp > 0 && noisePct <= noiseTarget);
        if (noiseMet) metNoise = true;
        bool done = stopped || timeUp || noiseMet;
        bool wantStatus = sinceSave >= intervalSec;
        if (done || wantStatus) {
            Film comp = writeOut(done);
            lastSave = clk::now();
            const char* why = stopped ? " (stopping)" : noiseMet ? " (noise target met)" : "";
            char st[220];
            if (runForever)
                std::snprintf(st, sizeof st, "[forever] %.1fs, %lld photons / %lld spp, ~%.2f%% noise%s",
                              elapsed, acc.N, acc.spp, noisePct, why);
            else if (timeBudgetSec > 0.0)
                std::snprintf(st, sizeof st, "[time] %.1fs / %.3gs, %lld photons / %lld spp, ~%.2f%% noise%s",
                              elapsed, timeBudgetSec, acc.N, acc.spp, noisePct, why);
            else if (noiseTarget > 0.0)
                std::snprintf(st, sizeof st, "[noise] target ~%.2g%%, %.1fs, %lld photons / %lld spp, ~%.2f%% noise%s",
                              noiseTarget, elapsed, acc.N, acc.spp, noisePct, why);
            else
                std::snprintf(st, sizeof st, "[spp] %lld / %lld spp (%lld / %lld photons), %.1fs, ~%.2f%% noise",
                              acc.spp, sppReq, acc.N, Nreq, elapsed, noisePct);
            if (preview) ansiPreview(comp, 1.0, manualExposure, st);
            else { std::printf("%s\n", st); std::fflush(stdout); }
            liveWindowUpdate(comp, 1.0, manualExposure, absolute, st);
        }
        if (done) break;
    }

    std::signal(SIGINT, prev);
#ifdef SIGBREAK
    std::signal(SIGBREAK, prevBrk);
#endif
    if (g_stopRequested)
        std::printf("\n[stop] interrupted at %lld photons / %lld spp — image saved.\n", acc.N, acc.spp);
    else if (metNoise)
        std::printf("[noise] reached the ~%.2g%% target at %lld spp — image saved.\n", noiseTarget, acc.spp);
    if (wantCheckpoint)
        std::printf("[checkpoint] %s holds %lld photons / %lld spp — rerun with -resume to add more\n",
                    checkpointPath(outPath).c_str(), acc.N, acc.spp);
    return writeOk ? 0 : 1;
}

// Render one camera into `outPath`. Resolves the -device request for THIS mode,
// runs the mode dispatch (R/V backward+validate, P composite, or A/B/C forward),
// and writes the result. Factored out of main so any number of cameras (Phase 3a
// multi-camera) share exactly one render path. `res` is the camera's own film
// resolution; `cam` must already be built at that resolution.
static int runRender(const Scene& scene, const Camera& cam, char mode,
                     long long N, int res, int resY, long long spp, int nThreads,
                     const char* device, bool diffraction,
                     const char* lightLabel, const std::string& outPath,
                     double manualExposure = 0.0,
                     double timeBudgetSec = 0.0, bool resume = false,
                     bool wantCheckpointFlag = false, bool runForever = false,
                     bool preview = false, double intervalSec = 15.0,
                     double noiseTarget = 0.0, bool wavefront = false,
                     double* exposureAnchor = nullptr, bool rgbBackward = false,
                     int maxBounceOverride = -1, bool directOnly = false) {
    g_windowMode = modeLabel(mode);   // title bar shows the transport mode of this frame
    const bool refMode      = (mode == 'R' || mode == 'V');
    const bool useCamera    = (mode == 'A' || mode == 'B' || mode == 'C' || mode == 'P' || mode == 'D' || refMode);
    const bool forwardCatch = (mode == 'C');
    const bool lensMode     = (mode == 'A');   // finite-lens next-event splat (physical camera)

    // -time / -noise / -forever now drive progress for the spp image modes too (R
    // backward reference, D bidirectional): those accumulate a SUM-over-samples film in
    // chunks, so a wall-clock/noise/indefinite budget just keeps adding samples exactly
    // like the forward camera models. -resume / -checkpoint apply to the forward models
    // A/B/C (photon-count checkpoint), the SUM-over-spp reference modes R and D
    // (spp-count checkpoint — the .ftbuf stores the SUM film + spp; resume adds decorrelated
    // samples on top), and the composite mode P (dual-film checkpoint — forward SUM + backward
    // SUM). The persistent-state modes M/S/U (photon-map / SPPM / VCM) can't be
    // resumed from a film alone, so keep those gated with a warning.
    if ((resume || wantCheckpointFlag) &&
        !(mode == 'A' || mode == 'B' || mode == 'C' || mode == 'R' || mode == 'D' || mode == 'P')) {
        std::fprintf(stderr, "[render] -resume/-checkpoint apply only to modes A/B/C "
                             "(forward), R/D (reference), and P (composite); ignoring for mode %c\n", mode);
        resume = false; wantCheckpointFlag = false;
    }
    if ((timeBudgetSec > 0.0 || noiseTarget > 0.0 || runForever) &&
        !(mode == 'A' || mode == 'B' || mode == 'C' || mode == 'R' || mode == 'D' ||
          mode == 'P' || mode == 'M' || mode == 'S' || mode == 'U')) {
        std::fprintf(stderr, "[render] -time/-noise/-forever apply only to modes A/B/C (forward), "
                             "R/D (reference/BDPT), P (composite), and M/S/U (photon map / SPPM / VCM); ignoring for mode %c\n", mode);
        timeBudgetSec = 0.0; noiseTarget = 0.0; runForever = false;
    }
    if (intervalSec <= 0.0) intervalSec = 15.0;

    // Heterogeneous / bounded participating media (a `density` field or a `bounds`
    // box on `medium`) are honored only by the FORWARD light tracer (modes A/B/C, and
    // the forward layers of V/P). The backward reference (R/V) and the camera-side layer
    // of the P composite still treat the medium as a single global HOMOGENEOUS haze —
    // they ignore the density field and the bounds box. Warn loudly rather than silently
    // render a different fog than authored. (Tracked in known-issues.md: heterogeneous
    // media in backward modes.) Mode D (volumetric BDPT) is EXCLUDED here: it handles
    // multiple superposed, box/sphere/object-bounded AND heterogeneous (density-field)
    // media correctly (over the full scene.media vector) — subpath medium vertices are
    // placed by delta tracking and connections weighted by ratio-tracking transmittance —
    // so it never needs this "single global haze" warning.
    bool mediaNeedForward = scene.media.size() > 1;   // >1 medium is forward-only (R/V/P)
    for (const Medium& m : scene.media)
        if (m.heterogeneous() || m.bounded) mediaNeedForward = true;
    if (scene.anyMedium() && mediaNeedForward &&
        (mode == 'R' || mode == 'V' || mode == 'P')) {
        std::fprintf(stderr,
            "[medium] mode %c uses the backward tracer, which treats participating "
            "media as a SINGLE global HOMOGENEOUS haze (the first authored medium); any "
            "additional media, `density` fields and `bounds` regions (box/sphere/object) are "
            "IGNORED here. Render multi/heterogeneous/bounded fog with a forward mode "
            "(A/B/C) or volumetric BDPT (mode D) for correct results.\n", mode);
    }

    // Resolve the -device request (auto|cpu|gpu) to a concrete GPU flag. The GPU
    // covers the forward light trace (models A/B/C, the forward pass of mode V, and
    // the forward layer of the mode-P composite) AND the backward tracer (mode R, and
    // the mode-P camera-side layer) when the scene is within the backward-GPU scope
    // (renderBackwardCuda / cudaBackwardSupported — Lambertian/textured/specular,
    // point-spot lights, participating media, fluorescence, and a constant env light;
    // image-based env, collimated beams and GRIN/rainbow media still fall back to the
    // CPU backward tracer); otherwise the backward layer falls back to the CPU. Mode V keeps
    // its backward reference on the CPU by design. Fisheye/panoramic lenses run on the
    // GPU too (the device camera's project()/pixelSolidAngle() port the analytic
    // projection remap) for the pinhole-splat modes (B/V/P).
    const bool gpuForwardMode =
        (mode == 'A' || mode == 'B' || mode == 'C' || mode == 'V' || mode == 'P');
    const bool gpuBdptMode = (mode == 'D');   // GPU BDPT megakernel (own support check)
    // GPU backward reference megakernel (own check). -mode W runs here too: the device
    // megakernel carries a full twin of the deterministic estimators (the bkWhitted /
    // bkGrid / bkGi* / bkHeroSplit / bkAmbient DScene knobs + the dWhitted* lattice helpers
    // in render_cuda.cu, and the split-at-dispersion walk bkRadianceHeroLoop<true>), so it
    // reproduces the CPU's noise-free image rather than the noisy one the mode exists to
    // avoid. The device twin is not yet complete, though -- cudaBackwardWhittedSupported()
    // rejects -gi, and those scenes fall back to the CPU mode-W tracer (which they can
    // afford, being ~1 spp).
    const bool gpuBackwardMode = (mode == 'R');
    const bool wantGpu  = !std::strcmp(device, "gpu");
    const bool wantAuto = !std::strcmp(device, "auto");
    const bool fisheyeCam = (cam.projection != CAM_RECTILINEAR);
    // BDPT's camera importance (bdpt.h cameraWe/cameraPdfDir) is the rectilinear
    // pinhole convention and feeds the MIS balance heuristic; a fisheye lens there
    // would give subtly-wrong weights, so mode D rejects it rather than lie.
    if (fisheyeCam && mode == 'D') {
        std::fprintf(stderr, "[camera] mode D (BDPT) does not support a fisheye/panoramic "
                             "lens; render this camera with mode B (forward pinhole) or R "
                             "(reference) instead.\n");
        return 1;
    }
    // Model A/C image through a single rectilinear thin lens (lensImage uses
    // tanHalfX/Y), so they cannot form a fisheye — that needs a wide-angle lens
    // element. A fisheye is authored via the pinhole splat (mode B) or reference.
    if (fisheyeCam && (mode == 'A' || mode == 'C')) {
        std::fprintf(stderr, "[camera] mode %c (finite-lens camera) is rectilinear only; a "
                             "fisheye/panoramic lens can't be formed by the thin-lens model. "
                             "Render this camera with mode B (pinhole splat) or R (reference).\n",
                     mode);
        return 1;
    }
#ifdef HAVE_CUDA
    // Mode W's device knobs — the exact twin of the BackwardRenderer setup in the mode-R CPU
    // worker above (renderBackward's lambda), so the two estimators are configured
    // identically. Built here because BOTH the -device gate below and the mode-R dispatch
    // need it. Only read when g_whitted.
    WhittedOpts whittedOpts;
    whittedOpts.grid      = g_whittedGrid;
    whittedOpts.giDirs    = g_gi;
    whittedOpts.giGrid    = g_giGrid;
    whittedOpts.giBounce  = g_giBounce;
    whittedOpts.heroSplit = hero::gSplit || g_whitted;
    whittedOpts.ambient   = g_ambient * scene.ambientRef();
    whittedOpts.giClamp   = g_giClamp * scene.ambientRef();   // same scaling as ambient
#endif
    bool useGpu = false;
    if (!wantGpu && !wantAuto && std::strcmp(device, "cpu"))
        std::fprintf(stderr, "[device] unknown -device '%s'; using CPU "
                             "(valid: auto|cpu|gpu)\n", device);
    if (wantGpu || wantAuto) {
#ifdef HAVE_CUDA
        if (!cudaAvailable()) {
            if (wantGpu) std::fprintf(stderr, "[device] no CUDA device found; using CPU\n");
            else         std::printf("[device] auto -> CPU (no CUDA device found)\n");
        } else if (gpuBdptMode) {
            // Mode D has its own (stricter) GPU support check: BDPT scope only. A realistic
            // lens on the camera subpath (Plan B) is supported on-device too — the BDPT
            // kernel generates the lens ray via dGenLensRay, exactly as the GPU mode-R
            // backward megakernel does.
            if (!cudaBdptSupported(scene)) {
                const char* why = "scene has a BDPT-GPU-unsupported feature "
                                  "(fluorescent/oversized-mix material, fog, "
                                  "spot/env/collimated light, an `emit pattern:` emission "
                                  "profile, or a per-hit BSDF the GPU BDPT can't MIS: a "
                                  "procedural pattern or frosted/colored glass)";
                if (wantGpu) std::fprintf(stderr, "[device] %s; using CPU\n", why);
                else         std::printf("[device] auto -> CPU (%s)\n", why);
            } else {
                useGpu = true;
                std::printf("[device] %s -> GPU: %s\n", wantAuto ? "auto" : "gpu",
                            cudaDeviceName());
            }
        } else if (gpuBackwardMode) {
            // Mode R has its own GPU support check: the backward reference megakernel
            // (with the physical mesh-lens as a ray-gen front-end) covers area/sphere/
            // cylinder Lambertian AND point-spot lights, textured/specular materials,
            // participating media (homog+heterog, incl. rainbow phase — M10), GRIN
            // marching (M11), fluorescence, and BOTH a constant and an image-based env
            // light (M1). Collimated beams still fall back to the CPU backward tracer.
            // -mode W adds its own narrower check on top: the deterministic device twin
            // does not yet cover the -gi gather, so those scenes stay on the CPU mode-W
            // tracer (cheap there — mode W is ~1 spp).
            const bool bwOk = g_whitted
                            ? cudaBackwardWhittedSupported(scene, cam, whittedOpts)
                            : cudaBackwardSupported(scene, cam);
            if (!bwOk) {
                const char* why = g_whitted
                    ? "mode W scene is outside the deterministic GPU scope (-gi, or a "
                      "backward-GPU-unsupported feature)"
                    : "scene has a backward-GPU-unsupported feature "
                      "(collimated light, an `emit pattern:` emission "
                      "profile, or a lens deeper than the device cap)";
                if (wantGpu) std::fprintf(stderr, "[device] %s; using CPU\n", why);
                else         std::printf("[device] auto -> CPU (%s)\n", why);
            } else {
                useGpu = true;
                std::printf("[device] %s -> GPU: %s\n", wantAuto ? "auto" : "gpu",
                            cudaDeviceName());
            }
        } else if (!gpuForwardMode) {
            // Modes M (shared/flyby gather), S (SPPM) and U (VCM) run their OWN device gating
            // in their dispatch blocks below, so don't claim "CPU-only" here — that would be
            // wrong for an S/U render that then picks the GPU. Stay quiet and let the mode decide.
            if (mode != 'M' && mode != 'S' && mode != 'U') {
                const char* why = "unsupported mode - CPU-only path";
                if (wantGpu) std::fprintf(stderr,
                    "[device] GPU can't accelerate this render: %s; using CPU\n", why);
                else         std::printf("[device] auto -> CPU (%s)\n", why);
            }
        } else if (!cudaForwardSupported(scene)) {
            const char* why = "GPU-unsupported feature (layered material, indexed "
                              "palette, parametric record, oversized multilayer/mix "
                              "material, or an emissive 'fire' volume)";
            if (wantGpu) std::fprintf(stderr, "[device] scene has a %s; using CPU\n", why);
            else         std::printf("[device] auto -> CPU (%s)\n", why);
        } else {
            useGpu = true;
            const char* pSuffix = "";
            if (mode == 'P')
                pSuffix = cudaBackwardSupported(scene, cam)
                        ? " (forward + camera-side layers)"
                        : " (forward layer; camera-side stays CPU — outside backward-GPU scope)";
            std::printf("[device] %s -> GPU: %s%s\n", wantAuto ? "auto" : "gpu",
                        cudaDeviceName(), pSuffix);
        }
#else
        if (wantGpu)
            std::fprintf(stderr, "[device] built without CUDA; using CPU "
                                 "(reconfigure with a CUDA toolkit for -device gpu)\n");
#endif
    }

    // The wavefront (streaming) backend only applies to a forward render on the GPU.
    if (wavefront) {
        if (useGpu && gpuForwardMode)
            std::printf("[device] GPU backend: wavefront (streaming, path regeneration)\n");
        else
            std::fprintf(stderr, "[device] -wavefront ignored: it only applies to a forward "
                                 "GPU render (megakernel/CPU otherwise)\n");
    }

    // --- Backward reference (mode R) ---
    // Renders through the unified progress driver: the reference film accumulates as a
    // SUM over samples-per-pixel, so it chunks exactly like the forward camera models and
    // gets the same live status line / -preview thumbnail / periodic crash-safe write and
    // -time / -noise / -forever budgeting. GPU when in scope (backward megakernel, incl.
    // the physical lens), CPU otherwise — both chunk internally.
    if (mode == 'R') {
        const bool gpuBackward = useGpu;
        // Fast RGB backward (-rgb): the non-spectral Option-B previewer. Only on the GPU
        // and only when the scene is within its reduced scope; otherwise fall back to the
        // spectral backward (a warning is printed so the flag isn't silently ignored).
        bool rgbFast = false;
#ifdef HAVE_CUDA
        if (rgbBackward) {
            // The RGB kernel is a separate reduced tracer with no deterministic twin, so
            // -mode W keeps the spectral estimator (the whole point of W is a noise-free
            // image; the RGB kernel would hand back a noisy one).
            if (g_whitted)
                std::fprintf(stderr, "[render] -rgb ignored in -mode W: the fast RGB backward "
                                     "has no deterministic estimator; using the spectral "
                                     "mode-W tracer\n");
            else if (gpuBackward && cudaBackwardRGBSupported(scene, cam)) rgbFast = true;
            else std::fprintf(stderr, "[render] -rgb (fast RGB backward) not applicable to this "
                                      "render (%s); using the spectral backward tracer\n",
                              gpuBackward ? "scene outside the RGB fast-path scope"
                                          : "RGB fast path is GPU-only");
        }
#else
        if (rgbBackward)
            std::fprintf(stderr, "[render] -rgb ignored: built without CUDA (RGB fast path is GPU-only)\n");
#endif
        std::printf("mode R: backward %s at %dx%d on %s (light=%s) ...\n",
                    rgbFast ? "RGB fast preview" : "reference",
                    res, resY,
                    gpuBackward ? "GPU" : (std::to_string(nThreads) + " CPU threads").c_str(),
                    lightLabel);
        auto renderChunked = [&](long long sppTarget, const SppProgress* p) -> Film {
#ifdef HAVE_CUDA
            if (rgbFast)      return renderBackwardRGBCuda(scene, cam, res, resY, sppTarget, diffraction, p,
                                                           g_maxBounceOverride, g_directOnly);
            if (gpuBackward)  return renderBackwardCuda(scene, cam, res, resY, sppTarget, diffraction, p,
                                                        g_maxBounceOverride, g_directOnly, g_heroC,
                                                        g_whitted ? &whittedOpts : nullptr);
#endif
            return cpuSppChunks(sppTarget, p, res, resY,
                [&](long long c, unsigned long long off) {
                    return renderBackward(scene, cam, res, resY, c, nThreads, diffraction, off);
                });
        };
        // Disk resume/checkpoint (like A/B/C): a budgeted or -checkpoint render writes a
        // resumable .ftbuf sidecar; -resume continues it. The film is a SUM over spp.
        const bool ckpt = resume || wantCheckpointFlag ||
                          timeBudgetSec > 0.0 || noiseTarget > 0.0 || runForever;
        return runSppProgressive(outPath, spp, manualExposure, exposureAnchor, scene.absolute,
                                 timeBudgetSec, noiseTarget, runForever, intervalSec, preview,
                                 renderChunked, res, resY, resume, ckpt,
                                 checkpointGuard(scene, mode, res, resY), mode);
    }

    // --- Validation (mode V) ---
    // Mode V keeps its backward reference single-shot on the CPU as the stable ground
    // truth, then cross-checks it against a forward light-trace pass. No progressive
    // budgeting here (it renders a fixed spp / photon count to compare).
    if (mode == 'V') {
        std::printf("mode V: backward reference %lld spp at %dx%d on %d CPU threads (light=%s) ...\n",
                    spp, res, resY, nThreads, lightLabel);
        Film ref = renderBackward(scene, cam, res, resY, spp, nThreads, diffraction);

        std::printf("mode V: forward light tracer %lld photons for cross-check ...\n", N);
        EnergyReport e;
        Film fwd = renderForward(scene, &cam, res, resY, N, nThreads,
                                 /*forwardCatch*/false, /*lensMode*/false, /*useCamera*/true, e,
                                 diffraction, useGpu, /*seedBase*/0, wavefront);
        addEnvBackground(fwd, scene, cam, N);   // directly-viewed sky (env scenes)
        double tot = e.absorbed + e.sensor + e.escaped + e.residual;
        std::printf("[energy] absorbed=%.4f sensor=%.4f escaped=%.4f residual=%.4f (sum/emitted=%.6f)\n",
                    e.absorbed / e.emitted, e.sensor / e.emitted, e.escaped / e.emitted,
                    e.residual / e.emitted, tot / e.emitted);
        compareFilms(fwd, N, ref, spp);
        // Mode V produces a PAIR of images (the two independent estimates), so it derives
        // `<out>_forward` / `<out>_backward` from -o instead of writing one -o file. It
        // used to hard-code `validate_forward.ppm` / `validate_backward.ppm` in the CWD,
        // which ignored -o entirely and scattered output into the repo root.
        const std::string vFwd = pathWithSuffix(outPath, "_forward");
        const std::string vBk  = pathWithSuffix(outPath, "_backward");
        writeFilm(vFwd.c_str(), fwd, (double)N);
        writeFilm(vBk.c_str(),  ref, (double)spp);
        return 0;
    }

    // --- Bidirectional path tracing (mode D) ---
    if (mode == 'D') {
        // Refuse scenes outside BDPT's transport scope rather than render a subtly wrong
        // image (a realistic lens on the camera subpath IS supported — see bdpt.h).
        if (const char* unsupported = bdptUnsupportedFeature(scene)) {
            std::fprintf(stderr, "[mode D] this scene uses %s, which BDPT (mode D) does not "
                                 "support; render it with mode B/P (forward) or mode R "
                                 "(backward) instead.\n", unsupported);
            return 1;
        }
        // Path length in edges; connection cost grows ~depth^2, so the default stays
        // low. `-max-bounce` raises it: a specular-only cavity (a mirror-lined sphere,
        // a kaleidoscope, nested dielectrics) needs far more than 8 edges before the
        // recursive images stop truncating to black, and specular vertices are cheap
        // because a delta BSDF has no connection to make.
        int maxDepth = (g_maxBounceOverride >= 1) ? g_maxBounceOverride : 8;
        std::printf("mode D: bidirectional path tracing at %dx%d on %s (maxDepth=%d, light=%s) ...\n",
                    res, resY, useGpu ? "GPU" : (std::to_string(nThreads) + " CPU threads").c_str(),
                    maxDepth, lightLabel);
        // BDPT accumulates a SUM over spp (cam image + light-splat image), so it chunks
        // through the same unified progress driver as the forward and mode-R renders.
        auto renderChunked = [&](long long sppTarget, const SppProgress* p) -> Film {
#ifdef HAVE_CUDA
            // g_heroC > 1 enables the hero-wavelength bundle on both subpaths; the kernel
            // applies the media/GRIN/lens gate itself (matching the CPU BdptRenderer).
            if (useGpu) return renderBdptCuda(scene, cam, res, resY, sppTarget, maxDepth, diffraction, p, g_heroC);
#endif
            return cpuSppChunks(sppTarget, p, res, resY,
                [&](long long c, unsigned long long off) {
                    return renderBdpt(scene, cam, res, resY, c, nThreads, maxDepth, diffraction, off);
                });
        };
        // Disk resume/checkpoint (like A/B/C): a budgeted or -checkpoint render writes a
        // resumable .ftbuf sidecar; -resume continues it. The film is a SUM over spp.
        const bool ckpt = resume || wantCheckpointFlag ||
                          timeBudgetSec > 0.0 || noiseTarget > 0.0 || runForever;
        return runSppProgressive(outPath, spp, manualExposure, exposureAnchor, scene.absolute,
                                 timeBudgetSec, noiseTarget, runForever, intervalSec, preview,
                                 renderChunked, res, resY, resume, ckpt,
                                 checkpointGuard(scene, mode, res, resY), mode);
    }

    // --- Photon-mapped final gather (mode M) — ROADMAP item 1 ---------------------
    // Build a view-independent photon map ONCE (forward light-trace with the camera
    // splat off, depositing a record at every diffuse vertex), then run a backward
    // camera pass that estimates diffuse radiance by a radius density query into the
    // map. Specular/direct reach the diffuse surface normally; the map supplies the
    // (direct + indirect) diffuse illumination. The map is reusable across cameras of
    // a static scene — the flythrough win (see the multi-camera path below).
    if (mode == 'M') {
        double radius = (g_pmRadiusAbs > 0.0) ? g_pmRadiusAbs
                                              : scene.sceneRadius * g_pmRadiusFactor;
        std::printf("mode M: photon map — tracing %lld photons on %d CPU threads "
                    "(light=%s), gather radius %.4g ...\n",
                    N, nThreads, lightLabel, radius);
        if (g_pmFinalGather > 0)
            std::printf("mode M: final gather ON — %d hemisphere sub-rays/sample "
                        "(density query one bounce away)\n", g_pmFinalGather);
        PhotonMap pm;
        auto tp0 = std::chrono::steady_clock::now();
        tracePhotonPass(scene, N, nThreads, diffraction, pm, g_heroC);
        radius = buildPhotonMap(pm, radius, "mode M:");
        double buildSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        std::printf("mode M: deposited %zu photons from %lld emitted in %.1fs; "
                    "grid %dx%dx%d. Gathering camera pass at %dx%d ...\n",
                    pm.photons.size(), pm.nEmitted, buildSec, pm.nx, pm.ny, pm.nz, res, resY);
        if (pm.photons.empty())
            std::fprintf(stderr, "[mode M] warning: 0 photons deposited — no diffuse "
                                 "surfaces reached? The image will be black.\n");
        auto renderChunked = [&](long long sppTarget, const SppProgress* p) -> Film {
            return cpuSppChunks(sppTarget, p, res, resY,
                [&](long long c, unsigned long long off) {
                    return renderPhotonCamera(scene, cam, res, resY, pm, c, nThreads,
                                              diffraction, /*maxBounce*/32, off, g_pmFinalGather);
                });
        };
        return runSppProgressive(outPath, spp, manualExposure, exposureAnchor, scene.absolute,
                                 timeBudgetSec, noiseTarget, runForever, intervalSec, preview,
                                 renderChunked, res, resY);
    }

    // --- Stochastic progressive photon mapping (mode S) — ROADMAP item 2 ----------
    // Repeated bounded photon passes with a per-pixel shrinking gather radius. Persistent
    // per-pixel state (flux/radius/count) lives across passes in `st`; each pass re-samples
    // the camera visible points (stochastic PPM), traces N photons, gathers, and updates the
    // radius/flux. -n = photons PER PASS, -spp = number of passes (or a -time/-noise budget).
    if (mode == 'S') {
        double R0 = (g_pmRadiusAbs > 0.0) ? g_pmRadiusAbs
                                          : scene.sceneRadius * g_pmRadiusFactor;
#ifdef HAVE_CUDA
        // GPU SPPM (M3): per-pixel progressive state (tau/radius/nAcc/directSum + this pass's
        // visible point) stays resident on the device across passes; each pass deposits a
        // bounded photon set with the SAME forward tracer as mode M, host-builds the grid at
        // the largest current radius, and gathers on the device. Pinhole cameras only (dGenRay)
        // and the photon-map-supported scene scope; anything else falls through to the CPU.
        {
            const bool wantGpu  = !std::strcmp(device, "gpu");
            const bool wantAuto = !std::strcmp(device, "auto");
            if ((wantGpu || wantAuto) && !cam.hasLens() &&
                cudaAvailable() && cudaSppmSupported(scene)) {
                SppmSession* sess = sppmSessionBegin(scene, cam, res, resY, R0, diffraction,
                                                     /*maxBounce*/32, g_heroC);
                if (sess) {
                    std::printf("mode S: SPPM on %s — %lld photons/pass, R0=%.4g, alpha=%.2f "
                                "at %dx%d (light=%s) ...\n",
                                cudaDeviceName(), N, R0, g_sppmAlpha, res, resY, lightLabel);
                    auto renderChunked = [&](long long passTarget, const SppProgress* p) -> Film {
                        Film disp; disp.resX = res; disp.resY = resY; disp.alloc();
                        for (long long pass = 0; pass < passTarget; ++pass) {
                            sppmSessionPass(sess, N, g_sppmAlpha);
                            sppmSessionResolve(sess, disp);
                            long long passes = sppmSessionPasses(sess);
                            for (auto& v : disp.xyz) v = v * (double)passes;   // undone by /sppDone
                            if (p->report(disp, passes, passes >= passTarget)) break;
                        }
                        return disp;
                    };
                    int rc = runSppProgressive(outPath, spp, manualExposure, exposureAnchor,
                                               scene.absolute, timeBudgetSec, noiseTarget,
                                               runForever, intervalSec, preview,
                                               renderChunked, res, resY);
                    sppmSessionEnd(sess);
                    return rc;
                }
                std::fprintf(stderr, "[device] SPPM GPU session failed to start; using CPU\n");
            } else if (wantGpu) {
                const char* why = cam.hasLens()        ? "a physical-lens camera (pinhole only)"
                                : !cudaAvailable()     ? "no CUDA device found"
                                : "a GPU-unsupported scene feature";
                std::fprintf(stderr, "[device] mode S GPU path unavailable (%s); using CPU\n", why);
            }
        }
#endif
        std::printf("mode S: SPPM — %lld photons/pass, R0=%.4g, alpha=%.2f at %dx%d on "
                    "%d CPU threads (light=%s) ...\n",
                    N, R0, g_sppmAlpha, res, resY, nThreads, lightLabel);
        SPPMState st; st.init(res, resY, R0);
        // renderChunked runs the pass loop, reporting L*passes so the progress driver's
        // divide-by-sppDone recovers the resolved radiance L. Persistent state means we
        // ignore the chunk's sppTarget granularity and just step one pass per iteration.
        auto renderChunked = [&](long long passTarget, const SppProgress* p) -> Film {
            Film disp; disp.resX = res; disp.resY = resY; disp.alloc();
            for (long long pass = 0; pass < passTarget; ++pass) {
                sppmPass(scene, cam, st, N, nThreads, diffraction, g_sppmAlpha,
                         /*maxBounce*/32, (uint64_t)(pass + 1), g_heroC);
                disp = sppmResolve(st);
                for (auto& v : disp.xyz) v = v * (double)st.passes;   // undone by /sppDone
                if (p->report(disp, st.passes, st.passes >= passTarget)) break;
            }
            return disp;
        };
        return runSppProgressive(outPath, spp, manualExposure, exposureAnchor, scene.absolute,
                                 timeBudgetSec, noiseTarget, runForever, intervalSec, preview,
                                 renderChunked, res, resY);
    }

    // --- Vertex Connection and Merging (mode U) — ROADMAP item 3 ------------------
    // VCM/UPS: runs BDPT vertex connections AND photon-map vertex merging under one MIS
    // balance heuristic, so it is robust across diffuse GI, glossy, and specular caustics
    // in a single unbiased-in-the-limit estimator. Persistent per-pass accumulation lives
    // in `st`; each pass traces one light + one camera subpath per pixel, shrinking the
    // merge radius as r_i = R0 * i^((alpha-1)/2). -n is ignored (paths are per-pixel);
    // -spp = number of passes (or a -time/-noise budget). -vcmalpha = radius-shrink rate.
    if (mode == 'U') {
        if (const char* unsupported = vcmUnsupportedFeature(scene, cam)) {
            std::fprintf(stderr, "[mode U] this scene uses %s, which VCM (mode U) does not "
                                 "support; render it with mode B/P (forward), R (backward), "
                                 "or D (BDPT) instead.\n", unsupported);
            return 1;
        }
        int maxDepth = (g_maxBounceOverride >= 1) ? g_maxBounceOverride : 8;  // full path length in edges
        double R0 = (g_pmRadiusAbs > 0.0) ? g_pmRadiusAbs
                                          : scene.sceneRadius * g_pmRadiusFactor;
        // Hero-wavelength bundle (Wilkie 2014). Mode U's scene scope already excludes
        // everything the hero gate would reject — vcmUnsupportedFeature refuses media,
        // GRIN and lens cameras — so `-heroc N > 1` is the whole condition. CPU only for
        // now; the GPU session below is still the single-wavelength estimator.
        const int vcmHeroC = g_heroC;
#ifdef HAVE_CUDA
        // GPU VCM (M12): resident device session mirroring vcm.h's vcmPass. Each pass traces
        // one light + one camera subpath per pixel, combining BDPT vertex connections with
        // photon-map vertex merging under one balance-heuristic MIS; light vertices are stored
        // in a per-path slab, downloaded + compacted host-side, and gridded (cell = radius),
        // then the camera kernel does emission/NEE/connection/merge. Pinhole cameras only and
        // the BDPT-supported, media-free scene scope; anything else falls through to the CPU.
        {
            const bool wantGpu  = !std::strcmp(device, "gpu");
            const bool wantAuto = !std::strcmp(device, "auto");
            if ((wantGpu || wantAuto) && !cam.hasLens() &&
                cudaAvailable() && cudaVcmSupported(scene)) {
                VcmSession* sess = vcmSessionBegin(scene, cam, res, resY, diffraction, maxDepth,
                                                   vcmHeroC);
                if (sess) {
                    std::printf("mode U: VCM/UPS on %s — connections + merging, R0=%.4g, "
                                "alpha=%.2f at %dx%d (maxDepth=%d, light=%s, %s) ...\n",
                                cudaDeviceName(), R0, g_vcmAlpha, res, resY, maxDepth, lightLabel,
                                vcmHeroC > 1
                                    ? (std::string("hero C=") + std::to_string(vcmHeroC)).c_str()
                                    : "single-lambda");
                    auto renderChunked = [&](long long passTarget, const SppProgress* p) -> Film {
                        Film disp; disp.resX = res; disp.resY = resY; disp.alloc();
                        for (long long pass = 0; pass < passTarget; ++pass) {
                            double it = (double)(vcmSessionPasses(sess) + 1);
                            double radius = R0 * std::pow(it, 0.5 * (g_vcmAlpha - 1.0));
                            if (radius <= 0.0) radius = R0;
                            vcmSessionPass(sess, radius);
                            vcmSessionResolve(sess, disp);
                            long long passes = vcmSessionPasses(sess);
                            for (auto& v : disp.xyz) v = v * (double)passes;   // undone by /sppDone
                            if (p->report(disp, passes, passes >= passTarget)) break;
                        }
                        return disp;
                    };
                    int rc = runSppProgressive(outPath, spp, manualExposure, exposureAnchor,
                                               scene.absolute, timeBudgetSec, noiseTarget,
                                               runForever, intervalSec, preview,
                                               renderChunked, res, resY);
                    vcmSessionEnd(sess);
                    return rc;
                }
                std::fprintf(stderr, "[device] VCM GPU session failed to start; using CPU\n");
            } else if (wantGpu) {
                const char* why = cam.hasLens()        ? "a physical-lens camera (pinhole only)"
                                : !cudaAvailable()     ? "no CUDA device found"
                                : "a GPU-unsupported scene feature";
                std::fprintf(stderr, "[device] mode U GPU path unavailable (%s); using CPU\n", why);
            }
        }
#endif
        std::printf("mode U: VCM/UPS — connections + merging, R0=%.4g, alpha=%.2f at %dx%d on "
                    "%d CPU threads (maxDepth=%d, light=%s, %s) ...\n",
                    R0, g_vcmAlpha, res, resY, nThreads, maxDepth, lightLabel,
                    vcmHeroC > 1 ? (std::string("hero C=") + std::to_string(vcmHeroC)).c_str()
                                 : "single-lambda");
        vcm::VcmState st; st.init(res, resY);
        auto renderChunked = [&](long long passTarget, const SppProgress* p) -> Film {
            Film disp; disp.resX = res; disp.resY = resY; disp.alloc();
            for (long long pass = 0; pass < passTarget; ++pass) {
                // Progressive radius schedule (Georgiev/SmallVCM): shrink from R0.
                double it = (double)(st.passes + 1);
                double radius = R0 * std::pow(it, 0.5 * (g_vcmAlpha - 1.0));
                if (radius <= 0.0) radius = R0;
                vcm::vcmPass(scene, cam, st, radius, nThreads, diffraction, maxDepth,
                             (uint64_t)(st.passes + 1), vcmHeroC);
                disp = vcm::vcmResolve(st);
                for (auto& v : disp.xyz) v = v * (double)st.passes;   // undone by /sppDone
                if (p->report(disp, st.passes, st.passes >= passTarget)) break;
            }
            return disp;
        };
        return runSppProgressive(outPath, spp, manualExposure, exposureAnchor, scene.absolute,
                                 timeBudgetSec, noiseTarget, runForever, intervalSec, preview,
                                 renderChunked, res, resY);
    }

    // --- Forward + camera-side composite (mode P) ---
    // Progressive: alternates forward (model-B) and backward (camera-side) batches into
    // two persistent SUM films, recompositing + writing periodically. A budgeted or
    // -checkpoint render writes a resumable dual-film .ftbuf sidecar (magic FTPCM02);
    // -resume continues both halves, decorrelating fresh samples from the loaded ones.
    if (mode == 'P') {
        const bool ckpt = resume || wantCheckpointFlag ||
                          timeBudgetSec > 0.0 || noiseTarget > 0.0 || runForever;
        return runCompositeProgressive(scene, cam, res, resY, N, spp, nThreads, diffraction,
                                       useGpu, wavefront, outPath, manualExposure, exposureAnchor,
                                       timeBudgetSec, noiseTarget, runForever, intervalSec, preview,
                                       resume, ckpt, checkpointGuard(scene, mode, res, resY));
    }

    // --- Forward camera models A/B/C ---------------------------------------
    // These accumulate radiance photon-by-photon, so the render can be split into
    // batches for a wall-clock time budget (-time) and/or resumed from a saved film
    // (-resume): brightness tracks the cumulative photon count and only graininess
    // changes. Every batch uses seedBase = cumulative photons so it draws an
    // independent stream; the checkpoint stores the PURE-photon film (the direct view
    // of the sky is re-added deterministically at write time, never accumulated, so a
    // resume can't double-count it).
    const bool progressive = timeBudgetSec > 0.0 || runForever || noiseTarget > 0.0;   // batch loop modes
    const bool wantCheckpoint = resume || progressive || wantCheckpointFlag;
    const uint64_t guard = checkpointGuard(scene, mode, res, resY);
    const std::string backend = useGpu ? std::string("GPU")
                                       : (std::to_string(nThreads) + " CPU threads");

    Checkpoint acc;
    acc.film.resX = res; acc.film.resY = resY; acc.film.alloc();
    if (resume && readCheckpoint(outPath, res, resY, guard, mode, acc))
        std::printf("[resume] loaded %s: %lld photons accumulated so far\n",
                    checkpointPath(outPath).c_str(), acc.N);

    // `useAnchor` gates the camera_path exposure-lock: only the final converged write
    // should set/reuse the shared anchor (a premature intermediate save would lock in
    // a noisy anchor for the frame and every later path frame).
    bool writeOk = true;   // tracks the most recent writeFilm result (drives exit code)
    auto writeOut = [&](bool announceCheckpoint, bool quiet = false, bool useAnchor = true) {
        Film disp = acc.film;                        // display copy (+ direct sky view)
        if (useCamera && !forwardCatch) addEnvBackground(disp, scene, cam, acc.N);
        writeOk = writeFilm(outPath.c_str(), disp, (double)acc.N, manualExposure, quiet,
                  useAnchor ? exposureAnchor : nullptr, scene.absolute);
        if (wantCheckpoint) {
            if (writeCheckpoint(outPath, acc, guard, mode)) {
                if (announceCheckpoint)
                    std::printf("[checkpoint] wrote %s (%lld photons) — rerun with -resume to continue\n",
                                checkpointPath(outPath).c_str(), acc.N);
            } else {
                std::fprintf(stderr, "[checkpoint] could not write %s\n",
                             checkpointPath(outPath).c_str());
            }
        }
    };

    auto runBatch = [&](long long batchN) {
        EnergyReport e;
        Film b = renderForward(scene, &cam, res, resY, batchN, nThreads, forwardCatch,
                               lensMode, useCamera, e, diffraction, useGpu, (uint64_t)acc.N, wavefront);
        acc.film.merge(b);
        acc.N += batchN;
        acc.energy.emitted  += e.emitted;  acc.energy.absorbed += e.absorbed;
        acc.energy.sensor   += e.sensor;   acc.energy.escaped  += e.escaped;
        acc.energy.residual += e.residual;
    };

    using clk = std::chrono::steady_clock;
    // A plain fixed-N render with -window (no time/noise/forever budget) still wants a
    // live view, so chunk N into pieces and stop at the total. `chunkFixed` marks that
    // mode: the batch loop runs but the stop is the fixed photon total, not a budget.
    const bool chunkFixed = !progressive && g_showWindow;
    if (progressive || chunkFixed) {
        long long batchN = chunkFixed ? std::max(1LL, ((N > 0) ? N : 2'000'000) / 16)
                                      : ((N > 0) ? N : 2'000'000);  // -n is the granularity
        const char* resumeTag = (resume && acc.N > 0) ? " [resuming]" : "";
        char noiseSuffix[64] = "";                    // appended when -noise adds a floor
        if (noiseTarget > 0.0)
            std::snprintf(noiseSuffix, sizeof noiseSuffix, " or until ~%.2g%% noise", noiseTarget);
        if (chunkFixed)
            std::printf("mode %c: tracing %lld photons in %lld-photon batches at %dx%d on %s "
                        "(light=%s)%s — live window; Ctrl-C to stop early ...\n",
                        mode, N, batchN, res, resY, backend.c_str(), lightLabel, resumeTag);
        else if (runForever)
            std::printf("mode %c: tracing indefinitely in %lld-photon batches at %dx%d on %s "
                        "(light=%s)%s%s — press Ctrl-C to stop ...\n",
                        mode, batchN, res, resY, backend.c_str(), lightLabel, resumeTag, noiseSuffix);
        else if (timeBudgetSec > 0.0)
            std::printf("mode %c: tracing for %.3gs%s in %lld-photon batches at %dx%d on %s "
                        "(light=%s)%s (Ctrl-C to stop early) ...\n",
                        mode, timeBudgetSec, noiseSuffix, batchN, res, resY, backend.c_str(),
                        lightLabel, resumeTag);
        else   // -noise only: trace until the graininess estimate reaches the target
            std::printf("mode %c: tracing until ~%.2g%% noise in %lld-photon batches at %dx%d on %s "
                        "(light=%s)%s (Ctrl-C to stop early) ...\n",
                        mode, noiseTarget, batchN, res, resY, backend.c_str(), lightLabel, resumeTag);
        if (preview) { enableAnsiTerminal(); g_previewRows = 0; }  // fresh preview per render
        // Trap Ctrl-C so a long/indefinite render stops cleanly (final image +
        // checkpoint) instead of losing the batch since the last periodic save.
        auto prev = std::signal(SIGINT, onInterrupt);
#ifdef SIGBREAK
        auto prevBrk = std::signal(SIGBREAK, onInterrupt);  // Windows Ctrl-Break too
#endif
        auto t0 = clk::now();
        auto lastSave = t0;
        long long batches = 0;
        bool metNoise = false;
        for (;;) {
            runBatch(batchN); ++batches;
            double elapsed   = std::chrono::duration<double>(clk::now() - t0).count();
            double sinceSave = std::chrono::duration<double>(clk::now() - lastSave).count();
            bool stopped = g_stopRequested != 0;
            bool timeUp  = (!runForever && timeBudgetSec > 0.0 && elapsed >= timeBudgetSec);
            bool wantStatus = sinceSave >= intervalSec;
            // Cheap graininess estimate: Monte-Carlo relative error at an illuminated
            // pixel falls as 1/sqrt(samples), and the per-pixel photon (hit) count is
            // that sample count, so 100/sqrt(mean hits over lit pixels) is an honest
            // ballpark for how noisy the image still is. It drives both the status line
            // and the -noise stop. Computed every batch only when -noise is active
            // (needed to test the floor); otherwise just when we're about to report.
            double noisePct = 0.0, meanHits = 0.0;
            if (noiseTarget > 0.0 || wantStatus || stopped || timeUp) {
                double sumHits = 0.0; long long lit = 0;
                for (double h : acc.film.hits) if (h > 0.0) { sumHits += h; ++lit; }
                meanHits = lit ? sumHits / (double)lit : 0.0;
                noisePct = meanHits > 0.0 ? 100.0 / std::sqrt(meanHits) : 0.0;
            }
            // The estimate is only trustworthy once lit pixels have real coverage, so
            // require meanHits > 0 before honouring the floor (guards a degenerate
            // black frame from "converging" at 0% on the very first batch).
            bool noiseMet = (noiseTarget > 0.0 && meanHits > 0.0 && noisePct <= noiseTarget);
            if (noiseMet) metNoise = true;
            bool totalDone = chunkFixed && N > 0 && acc.N >= N;   // fixed-N window render
            bool done = stopped || timeUp || noiseMet || totalDone;
            if (done || wantStatus) {   // periodic crash-safe checkpoint + preview
                writeOut(/*announceCheckpoint*/false, /*quiet*/preview, /*useAnchor*/done);
                lastSave = clk::now();
                const char* why = stopped ? " (stopping)"
                                : noiseMet ? " (noise target met)"
                                : totalDone ? " (done)" : "";
                char st[220];
                if (chunkFixed)
                    std::snprintf(st, sizeof st, "[live] %.1fs, %lld / %lld photons, ~%.1f%% noise%s",
                                  elapsed, acc.N, N, noisePct, why);
                else if (runForever)
                    std::snprintf(st, sizeof st, "[forever] %.1fs elapsed, %lld batches, %lld photons, ~%.1f%% noise%s",
                                  elapsed, batches, acc.N, noisePct, why);
                else if (timeBudgetSec > 0.0)
                    std::snprintf(st, sizeof st, "[time] %.1fs / %.3gs, %lld batches, %lld photons, ~%.1f%% noise%s",
                                  elapsed, timeBudgetSec, batches, acc.N, noisePct, why);
                else
                    std::snprintf(st, sizeof st, "[noise] target ~%.2g%%, %.1fs, %lld batches, %lld photons, ~%.1f%% noise%s",
                                  noiseTarget, elapsed, batches, acc.N, noisePct, why);
                if (preview || g_showWindow) {
                    Film disp = acc.film;
                    if (useCamera && !forwardCatch) addEnvBackground(disp, scene, cam, acc.N);
                    if (preview) ansiPreview(disp, (double)acc.N, manualExposure, st);
                    else { std::printf("%s\n", st); std::fflush(stdout); }
                    liveWindowUpdate(disp, (double)acc.N, manualExposure, scene.absolute, st);
                } else { std::printf("%s\n", st); std::fflush(stdout); }
            }
            if (done) break;
        }
        std::signal(SIGINT, prev);                    // restore prior handler
#ifdef SIGBREAK
        std::signal(SIGBREAK, prevBrk);
#endif
        if (g_stopRequested) std::printf("\n[stop] interrupted — image and checkpoint saved.\n");
        else if (metNoise) std::printf("[noise] reached the ~%.2g%% target at %lld photons — image saved.\n",
                                       noiseTarget, acc.N);
        if (wantCheckpoint)
            std::printf("[checkpoint] %s holds %lld photons — rerun with -resume to add more\n",
                        checkpointPath(outPath).c_str(), acc.N);
    } else {
        // Fixed photon count: one batch of N. A fresh (non-resumed) render uses
        // seedBase 0, so it is bit-identical to the historical single-shot path.
        std::printf("mode %c: tracing %lld photons at %dx%d on %s (light=%s)%s ...\n",
                    mode, N, res, resY, backend.c_str(), lightLabel,
                    (resume && acc.N > 0) ? " [resuming]" : "");
        runBatch(N);
        writeOut(/*announceCheckpoint*/true);
    }

    double tot = acc.energy.absorbed + acc.energy.sensor + acc.energy.escaped + acc.energy.residual;
    if (acc.energy.emitted > 0.0)
        std::printf("[energy] absorbed=%.4f sensor=%.4f escaped=%.4f residual=%.4f (sum/emitted=%.6f)\n",
                    acc.energy.absorbed / acc.energy.emitted, acc.energy.sensor / acc.energy.emitted,
                    acc.energy.escaped / acc.energy.emitted, acc.energy.residual / acc.energy.emitted,
                    tot / acc.energy.emitted);
    return writeOk ? 0 : 1;
}

// --- -review: rendered-sequence review player -------------------------------
// Load an 8-bit RGB image (row 0 = top) from a rendered frame on disk. Handles PPM
// P6 with a tiny custom reader (stb_image can't decode PPM — ftrace's default output)
// and PNG/JPG/BMP/TGA via stb_image. Returns false on any failure.
static bool loadImageRGB(const std::string& path, int& w, int& h, std::vector<uint8_t>& rgb) {
    if (endsWithCI(path, ".ppm")) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::string magic; f >> magic;
        if (magic != "P6") return false;
        // Read three integers (width, height, maxval), skipping '#' comment lines.
        auto readInt = [&](long& v) -> bool {
            for (;;) {
                int c = f.peek();
                if (c == EOF) return false;
                if (std::isspace((unsigned char)c)) { f.get(); continue; }
                if (c == '#') { std::string junk; std::getline(f, junk); continue; }
                break;
            }
            f >> v; return (bool)f;
        };
        long W = 0, H = 0, mx = 0;
        if (!readInt(W) || !readInt(H) || !readInt(mx)) return false;
        if (W <= 0 || H <= 0 || mx != 255) return false;
        f.get();  // single whitespace after maxval precedes the pixel block
        rgb.assign((size_t)W * H * 3, 0);
        f.read(reinterpret_cast<char*>(rgb.data()), (std::streamsize)rgb.size());
        if (!f) return false;
        w = (int)W; h = (int)H;
        return true;
    }
    int nc = 0;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &nc, 3);
    if (!px) return false;
    rgb.assign(px, px + (size_t)w * h * 3);
    stbi_image_free(px);
    return true;
}

// `ftrace -review <base>` — play a directory of already-rendered frames on the same
// live window + timeline used by the fly viewer, so you can watch an actual rendered
// flyby, scrub/play it, RE-TIME it by painting local speed (wheel in Paint mode), and
// Save a re-paced copy. `base` is a filename stem with an optional directory: frames
// are files named `<stem><digits>.<ext>` (ftrace appends a zero-padded index), e.g.
// `-review png/swoop/swoop` matches swoop000.png, swoop001.png, ... Numeric-sorted.
// Self-contained utility path (no scene load).
static int reviewMode(const std::string& base) {
    namespace fs = std::filesystem;
    fs::path bpath(base);
    fs::path dir = bpath.has_parent_path() ? bpath.parent_path() : fs::path(".");
    std::string prefix = bpath.filename().string();
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        std::fprintf(stderr, "-review: '%s' is not a directory\n", dir.string().c_str());
        return 2;
    }
    auto knownExt = [](const std::string& e) {
        static const char* exts[] = {"png","jpg","jpeg","bmp","tga","ppm"};
        std::string lo; for (char c : e) lo += (char)std::tolower((unsigned char)c);
        for (const char* x : exts) if (lo == x) return true;
        return false;
    };
    // Collect matching frames: name = prefix + digits + '.' + ext.
    std::vector<std::pair<long, std::string>> frames;  // (index, full path)
    for (const auto& de : fs::directory_iterator(dir, ec)) {
        if (!de.is_regular_file()) continue;
        std::string name = de.path().filename().string();
        if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0) continue;
        size_t i = prefix.size();
        size_t d0 = i;
        while (i < name.size() && std::isdigit((unsigned char)name[i])) ++i;
        if (i == d0) continue;                     // need at least one digit
        if (i >= name.size() || name[i] != '.') continue;
        std::string ext = name.substr(i + 1);
        if (!knownExt(ext)) continue;
        long idx = std::strtol(name.substr(d0, i - d0).c_str(), nullptr, 10);
        frames.emplace_back(idx, de.path().string());
    }
    if (frames.size() < 1) {
        std::fprintf(stderr, "-review: no frames matching '%s<digits>.<ext>' in %s\n",
                     prefix.c_str(), dir.string().c_str());
        return 2;
    }
    std::sort(frames.begin(), frames.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });
    const int nFrames = (int)frames.size();
    std::printf("[review] %d frames: %s ... %s\n", nFrames,
                fs::path(frames.front().second).filename().string().c_str(),
                fs::path(frames.back().second).filename().string().c_str());
    std::fflush(stdout);

    // Load the first frame to size the window.
    int fw = 0, fh = 0; std::vector<uint8_t> cur;
    if (!loadImageRGB(frames[0].second, fw, fh, cur)) {
        std::fprintf(stderr, "-review: failed to load %s\n", frames[0].second.c_str());
        return 2;
    }
    std::string title = "ftrace review — " + prefix;
    LiveWindow win(fw, fh, title.c_str());
    const double defFps = 30.0;
    win.enablePanel(nFrames, defFps, "n/a");
    win.setPanelState(0, /*playing*/false, /*pathMode*/true, "n/a");

    // Per-frame local speed multiplier (Paint-mode wheel brush is additive, clamped).
    std::vector<double> speed(nFrames, 1.0);
    auto speedAt = [&](double pos) {
        if (nFrames == 0) return 1.0;
        int i = std::clamp((int)std::floor(pos), 0, nFrames - 1);
        int j = std::min(i + 1, nFrames - 1);
        double f = pos - i;
        return speed[i] * (1.0 - f) + speed[j] * f;
    };
    auto paintSpeed = [&](double pos, double notches) {
        int i = std::clamp((int)std::floor(pos), 0, nFrames - 1);
        int j = std::min(i + 1, nFrames - 1);
        double f = pos - i;
        double delta = notches * 0.15;
        speed[i] = std::clamp(speed[i] + delta * (1.0 - f), 0.1, 10.0);
        speed[j] = std::clamp(speed[j] + delta * f,         0.1, 10.0);
    };

    using clock = std::chrono::steady_clock;
    auto prevT = clock::now();
    double pos = 0.0;           // fractional frame index
    int    shown = -1;          // frame currently displayed
    bool   playing = false;
    double camPerSec = defFps;  // frames/second when playing (before speed scaling)
    int    strideN = 1;
    bool   rateMode = true;
    double lastSpdSent = -1.0;
    int    lastIdxSent = -1;
    bool   lastPlaying = false;

    auto display = [&](int idx) {
        if (idx == shown) return;
        int w2 = 0, h2 = 0; std::vector<uint8_t> rgb;
        if (loadImageRGB(frames[idx].second, w2, h2, rgb)) {
            win.update(w2, h2, rgb);
            shown = idx;
        }
    };
    display(0);

    std::printf("[review] scrub/Play the timeline; Paint + wheel re-times (speed); "
                "Flat resets; Save writes a re-paced copy. Close the window to finish.\n");
    std::fflush(stdout);

    while (!win.closed()) {
        NavInput nav = win.drainNav();
        auto nowT = clock::now();
        double dt = std::chrono::duration<double>(nowT - prevT).count();
        prevT = nowT;
        if (dt > 0.25) dt = 0.25;

        if (nav.stride    >= 1)  strideN   = nav.stride;
        if (nav.camPerSec > 0.0) camPerSec = nav.camPerSec;
        rateMode = nav.rateMode;

        if (nav.togglePlay) {
            playing = !playing;
            if (playing && pos >= nFrames - 1 - 1e-9) pos = 0.0;
        }
        if (nav.scrubTo >= 0) {
            playing = false;
            pos = std::clamp((double)nav.scrubTo, 0.0, (double)(nFrames - 1));
        }
        if (nav.reset) { pos = 0.0; playing = false; }

        // Paint mode: wheel paints local speed (re-timing brush); otherwise wheel dollies
        // the timeline one frame per notch.
        bool wheelPainted = false;
        if (nav.paintMode && nav.wheel != 0.0) {
            paintSpeed(pos, nav.wheel);
            wheelPainted = true;
        }
        if (nav.speedReset) std::fill(speed.begin(), speed.end(), 1.0);

        // Advance playback (speed-scaled) or step by a painted/plain wheel notch.
        if (playing) {
            double rate = rateMode ? (camPerSec * dt) : (double)strideN;
            pos += rate * speedAt(pos);
            if (pos >= nFrames - 1) { pos = nFrames - 1; playing = false; }
        }
        if (!wheelPainted && nav.wheel != 0.0)
            pos = std::clamp(pos + nav.wheel, 0.0, (double)(nFrames - 1));

        display(std::clamp((int)std::llround(pos), 0, nFrames - 1));

        // Save: re-pace the sequence by the painted speed profile. Fast-painted regions
        // yield fewer output frames (skimmed), slow regions more (dwelt on). We resample
        // nFrames output slots uniformly in cumulative DWELL time (dwell = 1/speed), then
        // copy the chosen source file into <dir>/retimed/.
        if (nav.saveCurve) {
            std::vector<double> cum(nFrames + 1, 0.0);
            for (int i = 0; i < nFrames; ++i) cum[i + 1] = cum[i] + 1.0 / std::max(1e-3, speed[i]);
            double total = cum[nFrames];
            fs::path outDir = dir / "retimed";
            std::error_code mec; fs::create_directories(outDir, mec);
            int written = 0;
            for (int j = 0; j < nFrames; ++j) {
                double target = (nFrames > 1) ? (double)j / (nFrames - 1) * total : 0.0;
                int src = 0;
                while (src < nFrames - 1 && cum[src + 1] < target) ++src;
                fs::path sp(frames[src].second);
                char nm[64];
                std::snprintf(nm, sizeof(nm), "%s%03d%s", prefix.c_str(), j,
                              sp.extension().string().c_str());
                fs::path dst = outDir / nm;
                std::error_code cec;
                fs::copy_file(sp, dst, fs::copy_options::overwrite_existing, cec);
                if (!cec) ++written;
            }
            std::printf("[review] re-timed %d frames -> %s\n", written, outDir.string().c_str());
            std::printf("[review] assemble e.g.: ffmpeg -framerate %g -i \"%s/%s%%03d.png\" -pix_fmt yuv420p %s_retimed.mp4\n",
                        defFps, outDir.string().c_str(), prefix.c_str(), prefix.c_str());
            std::fflush(stdout);
        }

        // Mirror live state onto the panel (no feedback edges).
        int idxNow = std::clamp((int)std::llround(pos), 0, nFrames - 1);
        if (idxNow != lastIdxSent || playing != lastPlaying) {
            win.setPanelState(idxNow, playing, /*pathMode*/true, "n/a");
            lastIdxSent = idxNow; lastPlaying = playing;
        }
        double sp = speedAt(pos);
        if (std::fabs(sp - lastSpdSent) > 5e-3) { win.setSpeedLabel(sp); lastSpdSent = sp; }

        std::this_thread::sleep_for(std::chrono::milliseconds(playing ? 8 : 20));
    }
    std::printf("[review] window closed.\n");
    return 0;
}

// --- Stereoscopic (3-D) output helpers (-stereo) ------------------------------
// The two eyes render as two ordinary rectilinear cameras (offset along the right
// axis u, each with an off-axis sheared frustum via Camera::frustumShiftX), then this
// post-pass composites their PNGs into one side-by-side or anaglyph image. Reusing the
// full render pipeline per eye means checkpoints, budgets, GPU and the live window all
// work unchanged; only the compositing lives here.
enum StereoMode { STEREO_OFF = 0, STEREO_SBS, STEREO_CROSS, STEREO_ANAGLYPH_RC, STEREO_ANAGLYPH_GM };

// Best-effort screen DPI for `-dpi auto`. On Windows this is the LOGICAL system DPI
// (usually 96 unless the user scaled the desktop), NOT the monitor's physical pixel
// pitch — reading true physical DPI needs the panel's EDID, which we don't parse. So it
// is only a rough hint; pass a measured -dpi (or rely on the -view-dist / FOV mapping,
// the default) for a physically exact baseline. Returns 0 when unknown.
#if defined(_WIN32)
extern "C" __declspec(dllimport) unsigned int __stdcall GetDpiForSystem(void);
#endif
static double stereoDetectDpi() {
#if defined(_WIN32)
    unsigned int d = GetDpiForSystem();
    return (d > 0) ? (double)d : 0.0;
#else
    return 0.0;
#endif
}

// Dubois least-squares anaglyph mixing matrices (row-major 3x3, applied to sRGB in
// [0,1]). Far less ghosting / retinal rivalry than a naive channel split. From Eric
// Dubois' optimised projections (the same matrices bino/3dtv use). out = ML*left +
// MR*right, then clamp. Red-cyan (default) and green-magenta variants.
static const double kDuboisRC_L[9] = {
     0.437,  0.449,  0.164,
    -0.062, -0.062, -0.024,
    -0.048, -0.050, -0.017 };
static const double kDuboisRC_R[9] = {
    -0.011, -0.032, -0.007,
     0.377,  0.761,  0.009,
    -0.026, -0.093,  1.234 };
static const double kDuboisGM_L[9] = {
    -0.062, -0.158, -0.039,
     0.284,  0.668,  0.143,
    -0.015, -0.027,  0.021 };
static const double kDuboisGM_R[9] = {
     0.529,  0.705,  0.024,
    -0.016, -0.015, -0.065,
     0.009,  0.075,  0.937 };

// Composite two already-rendered eye PNGs into one stereo image at `outPath`.
// Returns true on success. `left`/`right` are the on-disk eye files.
static bool stereoComposite(int mode, const std::string& left, const std::string& right,
                            const std::string& outPath) {
    int lw = 0, lh = 0, lc = 0, rw = 0, rh = 0, rc = 0;
    unsigned char* lp = stbi_load(left.c_str(),  &lw, &lh, &lc, 3);
    unsigned char* rp = stbi_load(right.c_str(), &rw, &rh, &rc, 3);
    if (!lp || !rp) {
        std::fprintf(stderr, "[stereo] could not load eye images (%s / %s)\n",
                     left.c_str(), right.c_str());
        if (lp) stbi_image_free(lp);
        if (rp) stbi_image_free(rp);
        return false;
    }
    if (lw != rw || lh != rh) {
        std::fprintf(stderr, "[stereo] eye images differ in size (%dx%d vs %dx%d)\n",
                     lw, lh, rw, rh);
        stbi_image_free(lp); stbi_image_free(rp);
        return false;
    }
    const int W = lw, H = lh;
    bool ok = false;
    if (mode == STEREO_SBS || mode == STEREO_CROSS) {
        // Side-by-side: wall-eyed puts Left|Right, cross-eyed swaps to Right|Left.
        const unsigned char* halfL = (mode == STEREO_SBS) ? lp : rp;   // shown on the left
        const unsigned char* halfR = (mode == STEREO_SBS) ? rp : lp;   // shown on the right
        std::vector<uint8_t> out((size_t)W * 2 * H * 3);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                size_t s = ((size_t)y * W + x) * 3;
                size_t dL = ((size_t)y * (2 * W) + x) * 3;
                size_t dR = ((size_t)y * (2 * W) + (W + x)) * 3;
                for (int c = 0; c < 3; ++c) { out[dL + c] = halfL[s + c]; out[dR + c] = halfR[s + c]; }
            }
        }
        ok = writeImage(outPath, 2 * W, H, out);
    } else {
        const double* ML = (mode == STEREO_ANAGLYPH_GM) ? kDuboisGM_L : kDuboisRC_L;
        const double* MR = (mode == STEREO_ANAGLYPH_GM) ? kDuboisGM_R : kDuboisRC_R;
        std::vector<uint8_t> out((size_t)W * H * 3);
        for (size_t i = 0; i < (size_t)W * H; ++i) {
            const unsigned char* Lc = lp + i * 3;
            const unsigned char* Rc = rp + i * 3;
            double l0 = Lc[0] / 255.0, l1 = Lc[1] / 255.0, l2 = Lc[2] / 255.0;
            double r0 = Rc[0] / 255.0, r1 = Rc[1] / 255.0, r2 = Rc[2] / 255.0;
            for (int c = 0; c < 3; ++c) {
                double v = ML[c*3+0]*l0 + ML[c*3+1]*l1 + ML[c*3+2]*l2
                         + MR[c*3+0]*r0 + MR[c*3+1]*r1 + MR[c*3+2]*r2;
                out[i * 3 + c] = (uint8_t)std::clamp(v * 255.0 + 0.5, 0.0, 255.0);
            }
        }
        ok = writeImage(outPath, W, H, out);
    }
    stbi_image_free(lp); stbi_image_free(rp);
    if (ok) std::printf("[stereo] composited -> %s (%s)\n", outPath.c_str(),
                        mode == STEREO_SBS   ? "side-by-side wall-eyed" :
                        mode == STEREO_CROSS ? "side-by-side cross-eyed" :
                        mode == STEREO_ANAGLYPH_GM ? "green-magenta anaglyph"
                                                   : "red-cyan anaglyph");
    else std::fprintf(stderr, "[stereo] failed to write %s\n", outPath.c_str());
    return ok;
}

// Curated `-h` / `--help` usage summary. Covers the common flags grouped by task;
// the exhaustive list (fog, thin-film, mesh export, physics diagnostics, …) lives in
// README.md, which this points at rather than duplicating.
static void printHelp(const char* prog) {
    std::printf(
"ftrace — spectral forward + backward photon raytracer\n"
"\n"
"Usage:\n"
"  %s -in <scene.ftsl> [options]         render a scene file\n"
"  %s <scene.ftsl>                       quick raster preview in a live window\n"
"  %s <model.glb|.obj|.gltf|.fbx>        quick-view a bare mesh (auto-lit, auto-framed)\n"
"  %s [options]                          render the built-in demo scene\n"
"  %s -topng <in.ppm|in.ftbuf> <out.png> convert an artifact to PNG (no render)\n"
"  %s -review <base>                     play a rendered frame sequence\n"
"\n"
"Scene & camera:\n"
"  -in <file>            FTSL scene file (.ftsl/.scene); a bare positional path works too\n"
"                        (a bare mesh path .obj/.gltf/.glb/.fbx/.stl/.ply auto-lights & views it)\n"
"  -scene <name>         built-in demo scene (default: cornell)\n"
"  -light <name>         built-in light preset (default: bb6500)\n"
"  -camera <sel>         pick FTSL camera(s): <name>|<pathbase>|all|#N|near=X,Y,Z\n"
"  -view EX,EY,EZ/LX,LY,LZ[/FOV]   ad-hoc eye/look-at[/fovY] camera; renders just it\n"
"  -exposure|-ev <c>     override every camera's exposure compensation\n"
"  -exposure-lock        one shared auto-exposure anchor across all rendered cameras\n"
"\n"
"Render mode & budget:\n"
"  -mode <letter>        transport mode (default B; A/B/C forward, R/V/D backward — see README)\n"
"  -mode W               deterministic POV-Ray-style preview: mode R with every estimator\n"
"                        replaced by a fixed quadrature, so it is noise-free at -spp 1\n"
"                        (CPU or GPU). See -whitted-grid / -ambient below\n"
"  -n <count>            photon/sample count (accepts 2e8, 1.5e9)\n"
"  -r <W> [H]            resolution (square if H omitted)\n"
"  -time <sec>           wall-clock budget (progressive)\n"
"  -noise <pct>          stop at target graininess (progressive)\n"
"  -forever              trace until Ctrl-C (progressive)\n"
"  -spp <n>              samples/pixel for backward modes R/V\n"
"  -beams|-photonbeams   decorrelated photon-beams gather for shared multi-camera flybys\n"
"                        (single-scatter volumetrics; CPU or GPU; kills frozen speckle)\n"
"  -device auto|cpu|gpu  compute device (default: auto); -wavefront = streaming GPU backend\n"
"  -rgb                  mode R fast RGB (non-spectral) backward preview on the GPU (much\n"
"                        faster; drops dispersion/thin-film/fluorescence — Option B)\n"
"  -heroc <N>            hero-wavelength bundle size, 1..8 (default 4); 1 = single-λ, hero off\n"
"  -herosplit            at a dispersive interface fan the bundle into N monochromatic\n"
"                        sub-paths (crisp prism/rainbow caustics) instead of de-hero'ing;\n"
"                        costs ~N× traversal past the split (CPU forward A/B/C, M/S, backward\n"
"                        R; mode W always splits — it is what makes glass right at 1 spp)\n"
"  -t <n>                CPU thread count\n"
"\n"
"Scene-ignore (faster preview — strip expensive features, like the rasterizer):\n"
"  -no-media             drop all participating media (haze/fog/volumes)\n"
"  -no-env               remove the environment (sky/IBL) light\n"
"  -no-fluoro            demote fluorescent materials to plain diffuse\n"
"  -max-bounce <n>       set path depth to n bounces (default 32; modes D/U default 8,\n"
"                        where raising it is what a mirror-lined cavity needs)\n"
"  -direct-only          Whitted: direct + specular recursion only, no diffuse indirect\n"
"                        (near-1-spp preview; camera modes R/RGB and P's backward side)\n"
"\n"
"Mode W (deterministic preview) tuning:\n"
"  -whitted-grid <n>     n×n fixed shadow rays per area light (default 4 = 16 rays);\n"
"                        this is what makes a soft shadow smooth instead of noisy\n"
"  -ambient|-amb <v>     flat ambient fill, as a fraction of a light's own radiance\n"
"                        (default 0; try 0.02..0.2). The stand-in for the diffuse GI\n"
"                        mode W drops — without it a CLOSED room previews with black\n"
"                        shadows, since everything there is lit by bounce\n"
"  -gi|-radiosity <n>    replace the flat ambient with a REAL deterministic one-bounce\n"
"                        gather: n rays per diffuse vertex along a fixed lattice\n"
"                        (default 0 = off; try 16..64). Brings back what a constant\n"
"                        cannot — contact darkening in crevices, and colour bleeding\n"
"                        (a gold object actually tints the room). Costs roughly n/6×\n"
"                        the frame time. Has NO irradiance cache, so unlike POV-Ray's\n"
"                        radiosity it is safe on animation: nothing depends on render\n"
"                        order, so a seamless loop cannot flicker. -ambient still\n"
"                        applies, now as the far-field fill a gather ray sees when it\n"
"                        escapes the geometry\n"
"  -gi-grid <n>          n×n shadow rays at a GATHER vertex (default 1). Cheap detail\n"
"                        knob; the gather averages over n directions anyway\n"
"  -gi-bounce <n>        max bounces along one gather ray (default 4). Bounds the cost\n"
"                        of a specular chain inside a highly reflective lattice\n"
"  -gi-clamp <x>         firefly ceiling on ONE gather ray, as a multiple of one light's\n"
"                        own radiance (same units as -ambient; 0 = off, the default).\n"
"                        Fixes the thin bright dashed curves a glass ball or mirror puts\n"
"                        on nearby diffuse surfaces at low -spp: those are gather rays\n"
"                        that reach the lamp THROUGH the specular surface, carrying its\n"
"                        full radiance, and the shared direction lattice turns the\n"
"                        on/off boundary into a contour instead of noise. Try 0.05-0.2.\n"
"                        Keep it ABOVE -ambient: the clamp also caps the far-field tail an\n"
"                        escaping gather ray returns, so the gather's fill is effectively\n"
"                        min(-ambient, x) and a smaller x just darkens the whole scene\n"
"\n"
"Output, preview & checkpointing:\n"
"  -o <file.ppm|.png>    output path (default: cornell.ppm)\n"
"  -window               live OS preview window, refreshed as it converges\n"
"  -keepwindow|-hold     like -window but hold the final image until you close it\n"
"  -preview              live ANSI thumbnail in the terminal\n"
"  -interval <sec>       periodic image-write / preview cadence (default: 15)\n"
"  -checkpoint           write a resumable .ftbuf sidecar next to -o (modes A/B/C)\n"
"  -resume               continue an accumulated render from its .ftbuf checkpoint\n"
"  -parseonly            load the scene, print a contents summary, exit (no render)\n"
"  -stop [<pid>|all]     ask a RUNNING ftrace to finish cleanly (image + checkpoint\n"
"                        written, CUDA torn down) instead of killing it; bare -stop\n"
"                        lists running renders. Never force-kill a CUDA render.\n"
"\n"
"Raster preview & interactive explore (no light transport):\n"
"  -raster               fast solid-shaded preview; -raster-gpu = GPU isosurface preview\n"
"  -raster-iso <n>       marching-cubes resolution for isosurfaces (0 = skip)\n"
"  -explore | -fly       interactive fly-camera viewer (implies -keepwindow -no-meter); press T to cycle\n"
"                        the lit preview: raster -> mode W (deterministic, CPU, any scene) -> path-traced (GPU)\n"
"  -noclip|-nocollide    start the fly viewer with wall collision off\n"
"  -anim <file.json>     edit a loom CurveDrive sidecar in the fly viewer (implies -explore);\n"
"                        control points seed from it and Save writes the reshaped curve back\n"
"  -see-through|-glass   render clear dielectrics as see-through; -glass-clarity <0..1>\n"
"\n"
"Stereoscopic 3-D output:\n"
"  -stereo sbs|cross|anaglyph|anaglyph-gm   stereo pair / anaglyph composite\n"
"  -eye-sep <m>          interocular distance (default: 0.063)\n"
"  -view-dist <m>        viewing distance (default: 0.6)\n"
"  -dpi <n|auto>         screen pixel density; -convergence <m> = convergence-plane distance\n"
"\n"
"Utilities (exit after running):\n"
"  -topng|-convert <in> <out.png> [-ev <c>]   convert .ppm/.ftbuf to PNG\n"
"                        (-ev re-develops a .ftbuf brighter/darker, no re-render)\n"
"  -review <base>        play a rendered frame sequence on the live window\n"
"  -export-mesh <o.obj> [-mesh-res N] [-mesh-adaptive]   isosurface -> mesh\n"
"  -serve                resident loop: re-render scene paths streamed on stdin\n"
"  -viewer <s.json>      open the loom native viewer on a scene-introspection sidecar\n"
"  -loom <scene.py>      with -viewer: re-derive geometry live from this loom build\n"
"  -h | --help           show this help and exit\n"
"\n"
"See README.md for the complete flag list (fog, thin-film, meshes, diagnostics, …).\n",
        prog, prog, prog, prog, prog, prog);
}

static int run(int argc, char** argv) {
    // Normalize GNU-style double-dash options to the single-dash spellings the parser
    // uses, so EVERY flag accepts either form (`--window` == `-window`, `--beams` ==
    // `-beams`, etc.). We simply advance the pointer past one leading dash for any token
    // that starts with "--" and has more characters (leaving a lone "--" or "-" alone).
    // This runs before every downstream scan (help, -topng, the main parse loop, the
    // tab-completion helper), so a double-dash flag is never mistaken for an unknown
    // option, and genuinely unrecognized flags still fall through to the loud error at
    // the end of the parse loop (no invalid parameter is ever silently ignored).
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-' && argv[i][1] == '-' && argv[i][2] != '\0')
            argv[i] += 1;
    }
    // `-h` / `--help` (also `-help` / `help`) anywhere on the command line: print the
    // usage summary and exit, before any scene setup or the default render. Scanned
    // across all args (not just argv[1]) so `ftrace foo --help` still helps.
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help") ||
            !std::strcmp(argv[i], "-help") || !std::strcmp(argv[i], "help")) {
            printHelp(argv[0]);
            return 0;
        }
    }
    // Standalone artifact -> PNG conversion (no rendering): `ftrace -topng <in> <out>`
    // (`-convert` is an alias). Handles .ppm (P6 8-bit) and .ftbuf (raw linear film
    // checkpoint). Kept before all scene/CLI setup so it is a pure utility path.
    if (argc >= 2 && (!std::strcmp(argv[1], "-topng") || !std::strcmp(argv[1], "-convert"))) {
        if (argc < 4) {
            std::fprintf(stderr, "usage: %s -topng <input.ppm|input.ftbuf> [-ev <c>] <output.png>\n",
                         argv[0]);
            return 2;
        }
        // This branch runs before the main parse loop, so -exposure/-ev has to be picked
        // up here or it is silently ignored (it was, until 0.102.1). Only meaningful for
        // .ftbuf, which still holds linear film and is tone-mapped on the way out; a .ppm
        // is already 8-bit sRGB and is copied through untouched.
        double convExp = 0.0;   // <=0 = plain p99 auto-exposure
        for (int i = 4; i + 1 < argc; ++i)
            if (!std::strcmp(argv[i], "-exposure") || !std::strcmp(argv[i], "-ev"))
                convExp = std::atof(argv[++i]);
        if (convExp > 0.0 && endsWithCI(argv[2], ".ppm"))
            std::fprintf(stderr, "warning: -ev ignored for a .ppm input (already 8-bit sRGB); "
                                 "it only applies to a .ftbuf's linear film\n");
        return convertToPng(argv[2], argv[3], convExp);
    }
    // Rendered-sequence review player (no rendering): `ftrace -review <base>`.
    // Plays a directory of `<base><digits>.<ext>` frames on the live window/timeline,
    // with scrub/Play and speed re-timing. Pure utility path (no scene load).
    if (argc >= 2 && !std::strcmp(argv[1], "-review")) {
        if (argc < 3) {
            std::fprintf(stderr, "usage: %s -review <base>   (e.g. -review png/swoop/swoop)\n", argv[0]);
            return 2;
        }
        return reviewMode(argv[2]);
    }
    // Rainbow (Airy droplet phase) physics self-test: prints the primary/secondary
    // Descartes angles across the spectrum + Airy/normalisation checks, then exits.
    if (argc >= 2 && !std::strcmp(argv[1], "-rainbow-selftest")) {
        rainbow::RainbowPhase::selfTest();
        return 0;
    }
    long long N = 2'000'000;
    int res = 256;
    char mode = 'B';
    int nThreads = (int)std::thread::hardware_concurrency();
    const char* out = "cornell.ppm";
    const char* sceneName = "cornell";
    const char* lightName = "bb6500";
    double apertureR = 0.02;  // mode C aperture radius (scene units)
    double focusDist = 0.0;   // mode C thin-lens focus distance (0 = no lens)
    bool checkBvhOnly = false;
    bool checkImplicitOnly = false;
    bool bvhStatsOnly = false;
    bool checkLensOnly = false;
    bool checkFluoroOnly = false;
    const char* meshPath = nullptr;
    double meshScale = 1.0;
    const char* exportMeshPath = nullptr;  // -export-mesh <file.obj>: isosurface -> mesh
    int    exportMeshRes = 128;            // -mesh-res <N>: cells along longest bounds axis
    bool   exportMeshAdaptive = false;     // -mesh-adaptive: curvature-driven QEM decimation
    double exportMeshDecimate = 0.5;       // -mesh-decimate <f>: keep this fraction of triangles
    bool   checkWatertight = false;        // -check-watertight: audit every mesh/isosurface + exit
    bool   checkAirtight = false;          // -check-airtight: ray-parity audit of the marched field + exit
    long long airtightRays = 4000;         // -check-airtight chord count per isosurface
    long long spp = 256;      // backward reference samples/pixel (modes R and V)
    double fogSigmaT = 0.0;   // fog extinction coeff (0 = no fog); at 550nm if Rayleigh
    double fogAlbedo = 0.9;   // single-scattering albedo sigma_s/sigma_t
    double fogG = 0.0;        // Henyey-Greenstein anisotropy
    bool fogRayleigh = false; // wavelength-dependent scattering ~1/lambda^4
    bool checkFogOnly = false;
    double filmThickness = 300.0; // thin-film coating thickness (nm) for -scene iridescent
    double filmIor = 1.30;        // thin-film coating refractive index
    bool checkThinFilmOnly = false;
    bool checkMultilayerOnly = false;
    bool thinFilmSwatchOnly = false;
    bool diffraction = true;      // MatType::Grating diffraction on/off (-diffraction)
    bool checkGratingOnly = false;
    bool checkUpsampleOnly = false;
    bool checkGridOnly = false;
    bool checkScatterOnly = false;
    bool checkBindOnly = false;
    bool checkPropOnly = false;
    bool checkArrayOnly = false;
    bool checkSunOnly = false;
    const char* device = "auto";  // -device auto|cpu|gpu (auto = GPU when it helps)
    bool wavefront = false;       // -wavefront: streaming GPU backend (else megakernel)
    bool rgbBackward = false;      // -rgb: fast RGB (non-spectral) backward preview (mode R, GPU)
    // Scene-ignore flags (Stage 3): rasterizer-style feature stripping for a faster
    // preview. noMedia/noEnv/noFluoro mutate the scene (Scene::applyIgnoreFlags);
    // maxBounceOverride caps path depth (<0 = leave the tracer default of 32);
    // directOnly renders direct lighting + specular recursion only (no diffuse indirect).
    bool noMedia = false, noEnv = false, noFluoro = false, directOnly = false;
    int  maxBounceOverride = -1;
    const char* cameraSel = nullptr; // -camera <name>|<pathbase>|all|#N|near=X,Y,Z (FTSL multi-camera select)
    bool   haveView = false;         // -view: an ad-hoc CLI camera (renders/previews just it)
    Vec3   viewEye{0,0,0}, viewLook{0,0,0}, viewUp{0,1,0};
    double viewFov = 40.0;
    bool forceExposureLock = false;  // -exposure-lock: one shared auto-exposure anchor across all rendered cameras
    double timeBudgetSec = 0.0;   // -time <sec>: wall-clock render budget (modes A/B/C forward, R/D spp)
    double noiseTarget = 0.0;     // -noise <pct>: stop when estimated graininess falls to this % (A/B/C, R/D)
    bool resume = false;          // -resume: continue an accumulated render from its .ftbuf checkpoint (A/B/C)
    bool wantCheckpointFlag = false; // -checkpoint: save a resumable .ftbuf sidecar next to -o (A/B/C)
    bool runForever = false;      // -forever: trace until Ctrl-C (modes A/B/C forward, R/D spp)
    bool preview = false;         // -preview: live ANSI thumbnail during a progress render
    double intervalSec = 15.0;    // -interval <sec>: periodic image-write / preview cadence
    bool modeFromCli = false;     // did the CLI force a global -mode? (else per-camera)
    bool resFromCli  = false;     // did the CLI force a global -r?   (else per-camera)
    int  resYCli     = -1;        // optional height from `-r W H` (-1 = square, use res)
    bool doRaster    = false;     // -raster: fast solid-shaded preview (no light transport)
    bool exploreMode = false;     // -explore/-fly: raster + interactive fly viewer seeded at the first selected frame (no full render)
    bool noMeter     = false;     // -no-meter/-nometer: skip the exposure-lock metering pre-pass (frames auto-expose instead)
    bool viewerNoclip = false;    // -noclip/-nocollide: start the interactive fly-viewer with collision OFF (fly through walls)
    std::string animSidecar;      // -anim <file.json>: loom CurveDrive sidecar the curve editor seeds from / saves back to (E2 channel a)
    std::string animLoomScene;    // -loom <scene.py>: with -anim, the build file the LIVE channel re-derives from (E2 channel b)
    int  rasterIso   = 96;        // -raster-iso <n>: marching-cubes resolution for isosurfaces (0 = skip)
    bool rasterGpu   = false;     // -raster-gpu: GPU deterministic primary-ray iso preview (G2; NO tessellation)
    int  rasterBench = 0;         // -raster-bench <n>: render the first camera n times, report steady-state ms/frame (explorer metric)
    bool rasterSeeThrough = false; // -see-through/-glass: render clear (dielectric) objects as see-through (dim + milky haze, no refraction)
    double rasterClarity  = 0.85; // -glass-clarity <0..1>: per-surface transmittance for see-through mode (higher = clearer)
    double exposureCli = -1.0;    // -exposure/-ev <comp>: override every camera's exposure compensation (>0; <=0 = use authored)
    // --- Stereoscopic (3-D) output (-stereo) ---
    int    stereoMode    = STEREO_OFF; // -stereo sbs|cross|anaglyph|anaglyph-gm
    double stereoEyeSep  = 0.063;      // -eye-sep <m>: interocular distance (default 63 mm)
    double stereoViewDist= 0.6;        // -view-dist <m>: viewing distance (default 60 cm)
    double stereoDpi     = 0.0;        // -dpi <n|auto>: screen pixel density (0 = derive screen width from view-dist + FOV)
    double stereoConverge= 0.0;        // -convergence <m>: convergence-plane distance in scene units (0 = look-at target)
    bool   stereoKeepEyes= false;      // -stereo-keep-eyes: keep the intermediate per-eye PNGs (else deleted after compositing)

    // --- FTSL scene file (-in <file>) --------------------------------------
    // Load the scene from a file *before* parsing the rest of argv, so any explicit
    // CLI flag (-n, -r, -mode, -device, -o) still overrides what the file's
    // render {} block specified. Pre-scan for -in only; the full parse follows.
    const char* inFile = nullptr;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "-in") && i + 1 < argc) { inFile = argv[i + 1]; break; }
    // -parseonly: load the scene, report what it contains, exit 0 — or exit 1 with the
    // load error. Nothing renders and no device is touched. Prescanned here (rather than
    // with the other flags below) because the scene load happens before the main argv
    // pass, and the whole point is to stop the instant that load returns.
    //
    // This exists because "does every scene in the tree still load?" is the regression
    // question a front-end or loader change actually needs answered, and the only way to
    // ask it before was to render all ~200 of them.
    bool parseOnly = false;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "-parseonly")) { parseOnly = true; break; }
    // Positional scene / mesh file: `ftrace scene.ftsl` or `ftrace model.glb` (e.g. a
    // double-click / drag-drop) with no -in. Accept a bare token that ends in a scene
    // extension (loaded directly) OR a mesh extension (.obj/.gltf/.glb/.fbx/.stl/.ply —
    // wrapped in a synthesized, auto-lit quick-viewer scene below). Only these extensions
    // qualify, so this never swallows a flag value (no flag takes such a path argument).
    // A bare token that LOOKS like a file (has an extension or a path separator) but isn't
    // a recognized scene/mesh is remembered so we ERROR instead of silently rendering the
    // built-in demo — the old, confusing behavior (`ftrace foo.glb` used to draw cornell).
    bool positionalScene = false;
    bool positionalMesh  = false;
    const char* unknownPositional = nullptr;
    if (!inFile) {
        auto lower = [](const char* s){ std::string t = s; for (auto& c : t) c = (char)std::tolower((unsigned char)c); return t; };
        auto ends  = [](const std::string& t, const char* e){ size_t n = std::strlen(e); return t.size() >= n && t.compare(t.size()-n, n, e) == 0; };
        auto hasSceneExt = [&](const char* s){ std::string t = lower(s); return ends(t,".ftsl") || ends(t,".scene") || ends(t,".fts"); };
        auto hasMeshExt  = [&](const char* s){ std::string t = lower(s);
            return ends(t,".obj") || ends(t,".gltf") || ends(t,".glb") || ends(t,".fbx") || ends(t,".stl") || ends(t,".ply"); };
        auto looksLikeFile = [](const char* s){ std::string t = s; size_t sl = t.find_last_of("/\\");
            std::string base = (sl == std::string::npos) ? t : t.substr(sl + 1);
            return base.find('.') != std::string::npos || sl != std::string::npos; };
        for (int i = 1; i < argc; ++i) {
            if (argv[i][0] == '-') continue;                 // a flag
            const bool couldBeFlagValue = (i > 0 && argv[i-1][0] == '-');
            if (couldBeFlagValue) {                          // a flag's value: only claim it if it's a scene/mesh path
                if (!hasSceneExt(argv[i]) && !hasMeshExt(argv[i])) continue;
            }
            if (hasSceneExt(argv[i])) { inFile = argv[i]; positionalScene = true; break; }
            if (hasMeshExt(argv[i]))  { inFile = argv[i]; positionalScene = true; positionalMesh = true; break; }
            // Not a recognized scene/mesh. If it looks like a file path (and isn't a flag
            // value), flag it as an error candidate rather than silently ignoring it.
            if (!couldBeFlagValue && !unknownPositional && looksLikeFile(argv[i])) unknownPositional = argv[i];
        }
    }
    if (!inFile && unknownPositional) {
        std::fprintf(stderr,
            "[ftrace] unrecognized argument '%s': not a scene (.ftsl/.scene/.fts) or a mesh "
            "(.obj/.gltf/.glb/.fbx/.stl/.ply).\n"
            "  To render a scene file:   ftrace <scene.ftsl>\n"
            "  To quick-view a mesh:      ftrace <model.glb>\n"
            "  For the built-in demos:    ftrace -scene <cornell|materials|prism|fluoro|...>\n",
            unknownPositional);
        return 2;
    }
    // Pre-scan the two flags that affect `prefer{}/else{}` branch selection (which the
    // loader resolves up-front): a `-mode` override forces the mode a branch is judged
    // against, and `-on-unsupported` sets the global policy. Pre-scanning mirrors how
    // -in is found above; the full CLI loop below re-parses them normally.
    // Two more flags used to be pre-scanned here for the same reason — `-legacy-parser`
    // and `-validate-grammar` selected between the shared grammar and a hand-written
    // parser, and the scene is parsed just below, long before the full CLI loop runs.
    // 0.79.0 deleted that parser, so both are retired: still ACCEPTED in the CLI loop
    // below (a script that passes one keeps working) but announced as a no-op rather
    // than silently ignored.
    char cliModePrescan = 0;
    for (int i = 1; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "-mode")) {
            cliModePrescan = argv[i + 1][0];
            // -mode W is the deterministic Whitted preview. It is not a separate
            // transport: it reuses the backward tracer's traversal wholesale and only
            // swaps the stochastic estimators for deterministic ones, so it normalises
            // to 'R' HERE -- before any of the dozen `mode == 'R'` capability checks
            // downstream -- and carries its difference in g_whitted instead.
            if (cliModePrescan == 'W' || cliModePrescan == 'w') cliModePrescan = 'R';
        }
        else if (!std::strcmp(argv[i], "-on-unsupported")) {
            std::string v = argv[i + 1];
            if      (v == "fallback" || v == "fall") g_onUnsupported = OnUnsupported::Fallback;
            else if (v == "strip"    || v == "ignore") g_onUnsupported = OnUnsupported::Strip;
            else                                     g_onUnsupported = OnUnsupported::Error;
        }
    }

    // The prefer/else resolver asks this predicate whether a branch renders; when the
    // policy is fallback/strip we accept every branch (the policy handles it later at
    // render time), so the FIRST/most-preferred branch always wins. Function-scoped
    // because the initial load is not the only one: the fly editor's loom live channel
    // (-anim -loom) re-loads an emitted .ftsl mid-flight and must resolve prefer/else
    // exactly the way the scene it is replacing did.
    ftsl::SupportFn supportFn = (g_onUnsupported == OnUnsupported::Error)
        ? ftsl::SupportFn([cliModePrescan](const ftsl::Loaded& L) -> const char* {
              return sceneModeUnsupported(L, cliModePrescan);
          })
        : ftsl::SupportFn{};

    ftsl::Loaded ftslScene;
    bool fromFtsl = false;
    if (positionalMesh) {
        // ---- Quick mesh viewer -------------------------------------------------------
        // `ftrace model.glb` (or .obj/.gltf/.fbx/.stl/.ply) with no scene file wraps the
        // bare mesh in a synthesized, auto-lit FTSL scene and renders it with an
        // auto-framed camera. The mesh keeps its own materials when the format carries
        // them (glTF/GLB import materials by default); a neutral clay fallback covers
        // primitives/faces with none. Lit by a soft uniform environment so any mesh reads
        // with shape. Bare invocation then defaults to the fast raster preview in a live
        // window (see the positional-preview block below); pass -mode/-n/etc. to force a
        // real light-transport render of the same auto-lit scene.
        std::string mp = inFile;
        for (char& c : mp) if (c == '\\') c = '/';    // FTSL file strings use forward slashes
        std::string src;
        src += "scene { units meters spectral 360 830 1 }\n";
        src += "material \"clay\" { type diffuse reflect whitewall 0.6 }\n";
        src += "mesh { file \"" + mp + "\"  material clay }\n";
        src += "light env { spd 0.5 }\n";
        std::string ferr;
        if (!ftsl::loadSource(src, std::string("<mesh-viewer:") + inFile + ">", ftslScene, ferr)) {
            std::fprintf(stderr, "[ftrace] could not load mesh '%s': %s\n", inFile, ferr.c_str());
            return 1;
        }
        fromFtsl = true;
        std::printf("[viewer] quick-view scene for mesh %s (%zu triangles)\n",
                    inFile, ftslScene.scene.tris.size());
        // Auto-frame the camera on the scene bounding sphere from a 3/4 front-high angle,
        // far enough that the sphere fits the vertical FOV (with a little margin). Skip if
        // the user pinned their own -view.
        if (!haveView) {
            Vec3 ctr = ftslScene.scene.sceneCenter;
            double rad = (ftslScene.scene.sceneRadius > 0.0) ? ftslScene.scene.sceneRadius : 1.0;
            const double fovDeg = 40.0, half = fovDeg * 0.5 * PI / 180.0;
            double dist = (rad / std::sin(half)) * 1.15;
            Vec3 dir = {0.55, 0.42, 1.0};
            { double L = std::sqrt(dot(dir, dir)); dir = dir * (1.0 / L); }
            viewEye = ctr + dir * dist; viewLook = ctr; viewUp = {0, 1, 0}; viewFov = fovDeg;
            haveView = true;
            std::printf("[viewer] auto-framed: center (%.3f,%.3f,%.3f) radius %.3f -> eye (%.3f,%.3f,%.3f)\n",
                        ctr.x, ctr.y, ctr.z, rad, viewEye.x, viewEye.y, viewEye.z);
        }
    } else if (inFile) {
        std::string ferr;
        if (!ftsl::load(inFile, ftslScene, ferr, supportFn)) {
            std::fprintf(stderr, "[ftsl] %s\n", ferr.c_str());
            return 1;
        }
        fromFtsl = true;
        std::printf("[ftsl] loaded scene from %s\n", inFile);
        // Ahead-of-time nested-dielectric priority audit: warn where two overlapping
        // dielectric solids can't be disambiguated (missing/equal `priority`), so the
        // exterior IOR in the overlap would be picked arbitrarily. Read-only; renders
        // still proceed (Level-0 uses priority where present, else assumes exterior air).
        for (const std::string& w : pri::audit(ftslScene.scene))
            std::fprintf(stderr, "[priority] WARNING: %s\n", w.c_str());
        if (parseOnly) {
            const Scene& sc = ftslScene.scene;
            std::printf("[parseonly] ok: %zu materials, %zu records, %zu emitters, "
                        "%zu spheres, %zu tris, %zu implicits, %zu textures, "
                        "%zu patterns, %zu cameras\n",
                        sc.mats.size(), sc.records.size(), sc.emitters.size(),
                        sc.spheres.size(), sc.tris.size(), sc.implicits.size(),
                        sc.textures.size(), sc.patterns.size(),
                        ftslScene.cameras.size());
            return 0;
        }
        if (ftslScene.photons >= 0)       N = ftslScene.photons;
        if (ftslScene.res > 0)            res = ftslScene.res;
        if (ftslScene.mode)               mode = ftslScene.mode;
        // A scene-level `default_mode` is the authoritative fallback for cameras that don't
        // author their own `mode`. It takes precedence over the incidental global `mode`
        // above (which just trails the last camera/render block), but a per-camera `mode`
        // (via effMode) and a CLI -mode still override it.
        if (ftslScene.defaultMode)        mode = ftslScene.defaultMode;
        if (!ftslScene.device.empty())    device = ftslScene.device.c_str();
        if (!ftslScene.out.empty())       out = ftslScene.out.c_str();
    }

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            // Photon count. Accept both plain integers ("200000000") and scientific /
            // float shorthand ("2e8", "1.5e9") — atoll stops at the 'e', so parse the
            // token as a double when it contains one and round to the nearest count.
            const char* s = argv[++i];
            if (std::strpbrk(s, "eE.")) N = (long long)std::llround(std::atof(s));
            else                        N = std::atoll(s);
        }
        else if (!std::strcmp(argv[i], "-r") && i + 1 < argc) {
            res = std::atoi(argv[++i]); resFromCli = true;
            // Optional second numeric token makes a non-square film: `-r W H`.
            if (i + 1 < argc && argv[i + 1][0] != '-' && std::isdigit((unsigned char)argv[i + 1][0]))
                resYCli = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "-mode") && i + 1 < argc) {
            mode = argv[++i][0]; modeFromCli = true;
            if (mode == 'W' || mode == 'w') { mode = 'R'; g_whitted = true; }   // see the prescan
        }
        else if (!std::strcmp(argv[i], "-whitted-grid") && i + 1 < argc) {
            g_whittedGrid = std::max(1, std::atoi(argv[++i]));
        }
        else if ((!std::strcmp(argv[i], "-ambient") || !std::strcmp(argv[i], "-amb")) && i + 1 < argc) {
            g_ambient = std::max(0.0, std::atof(argv[++i]));
        }
        else if ((!std::strcmp(argv[i], "-gi") || !std::strcmp(argv[i], "-radiosity")) && i + 1 < argc) {
            g_gi = std::max(0, std::atoi(argv[++i]));
        }
        else if (!std::strcmp(argv[i], "-gi-grid") && i + 1 < argc) {
            g_giGrid = std::max(1, std::atoi(argv[++i]));
        }
        else if (!std::strcmp(argv[i], "-gi-bounce") && i + 1 < argc) {
            g_giBounce = std::max(1, std::atoi(argv[++i]));
        }
        else if (!std::strcmp(argv[i], "-gi-clamp") && i + 1 < argc) {
            g_giClamp = std::max(0.0, std::atof(argv[++i]));
        }
        else if (!std::strcmp(argv[i], "-on-unsupported") && i + 1 < argc) { ++i; /* pre-scanned into g_onUnsupported */ }
        // An explicit absolute radius pins the radius: don't then adapt it out from under
        // the user (this was the documented workaround for mode M's scaling problem).
        // -pmradiusfrac only rescales the STARTING radius, so it leaves adaptation on.
        else if (!std::strcmp(argv[i], "-pmradius") && i + 1 < argc) { g_pmRadiusAbs = std::atof(argv[++i]); g_pmAutoRadius = false; }
        else if (!std::strcmp(argv[i], "-pmradiusfrac") && i + 1 < argc) g_pmRadiusFactor = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-pmauto")) g_pmAutoRadius = true;
        else if (!std::strcmp(argv[i], "-nopmauto")) g_pmAutoRadius = false;
        else if (!std::strcmp(argv[i], "-pmcount") && i + 1 < argc) { g_pmAutoCount = std::atof(argv[++i]); g_pmAutoRadius = true; }
        else if (!std::strcmp(argv[i], "-pmfg") && i + 1 < argc) { g_pmFinalGather = std::atoi(argv[++i]); if (g_pmFinalGather < 0) g_pmFinalGather = 0; }
        else if (!std::strcmp(argv[i], "-savemap") && i + 1 < argc) g_pmapSave = argv[++i];
        else if (!std::strcmp(argv[i], "-loadmap") && i + 1 < argc) g_pmapLoad = argv[++i];
        else if (!std::strcmp(argv[i], "-sppmalpha") && i + 1 < argc) g_sppmAlpha = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-vcmalpha") && i + 1 < argc) g_vcmAlpha = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-heroc") && i + 1 < argc) {
            g_heroC = std::atoi(argv[++i]);
            if (g_heroC < 1) g_heroC = 1;
            if (g_heroC > hero::kHeroMax) g_heroC = hero::kHeroMax;
            g_heroCSet = true;
        }
        // Split-at-dispersion instead of de-hero. A single global policy flag read by
        // every CPU forward tracer via Renderer::heroSplit's default initialiser, so it
        // needs no plumbing through the renderer entry points (see hero.h).
        else if (!std::strcmp(argv[i], "-herosplit")) hero::gSplit = true;
        else if ((!std::strcmp(argv[i], "-exposure") || !std::strcmp(argv[i], "-ev")) && i + 1 < argc) exposureCli = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-camera") && i + 1 < argc) cameraSel = argv[++i];
        else if (!std::strcmp(argv[i], "-view") && i + 1 < argc) {
            // Ad-hoc preview/render camera: EX,EY,EZ/LX,LY,LZ[/FOV] (',' and '/'
            // are interchangeable separators). Renders and previews just this
            // camera, ignoring any authored/curve cameras.
            const char* s = argv[++i];
            double v[7]; int nv = 0;
            for (const char* p = s; *p && nv < 7; ) {
                char* e = nullptr; double val = std::strtod(p, &e);
                if (e == p) break;
                v[nv++] = val; p = e;
                while (*p == ',' || *p == '/' || *p == ' ') ++p;
            }
            if (nv < 6) { std::fprintf(stderr, "error: -view needs EX,EY,EZ/LX,LY,LZ[/FOV]\n"); return 1; }
            viewEye = {v[0], v[1], v[2]}; viewLook = {v[3], v[4], v[5]};
            if (nv >= 7) viewFov = v[6];
            haveView = true;
        }
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) nThreads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-scene") && i + 1 < argc) sceneName = argv[++i];
        else if (!std::strcmp(argv[i], "-light") && i + 1 < argc) lightName = argv[++i];
        else if (!std::strcmp(argv[i], "-aperture") && i + 1 < argc) apertureR = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-focus") && i + 1 < argc) focusDist = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-checkbvh")) checkBvhOnly = true;
        else if (!std::strcmp(argv[i], "-checkimplicit")) checkImplicitOnly = true;
        else if (!std::strcmp(argv[i], "-bvhstats")) bvhStatsOnly = true;
        else if (!std::strcmp(argv[i], "-checklens")) checkLensOnly = true;
        else if (!std::strcmp(argv[i], "-checkfluoro")) checkFluoroOnly = true;
        else if (!std::strcmp(argv[i], "-mesh") && i + 1 < argc) meshPath = argv[++i];
        else if (!std::strcmp(argv[i], "-meshscale") && i + 1 < argc) meshScale = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-export-mesh") && i + 1 < argc) exportMeshPath = argv[++i];
        else if (!std::strcmp(argv[i], "-mesh-res") && i + 1 < argc) exportMeshRes = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-mesh-adaptive")) exportMeshAdaptive = true;
        else if (!std::strcmp(argv[i], "-check-watertight") || !std::strcmp(argv[i], "-airtight")) checkWatertight = true;
        else if (!std::strcmp(argv[i], "-check-airtight")) checkAirtight = true;
        else if (!std::strcmp(argv[i], "-check-airtight-rays") && i + 1 < argc) airtightRays = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "-mesh-decimate") && i + 1 < argc) { exportMeshDecimate = std::atof(argv[++i]); exportMeshAdaptive = true; }
        else if (!std::strcmp(argv[i], "-spp") && i + 1 < argc) spp = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "-fog") && i + 1 < argc) fogSigmaT = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-fogalbedo") && i + 1 < argc) fogAlbedo = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-fogg") && i + 1 < argc) fogG = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-fograyleigh")) fogRayleigh = true;
        else if (!std::strcmp(argv[i], "-checkfog")) checkFogOnly = true;
        else if (!std::strcmp(argv[i], "-filmthickness") && i + 1 < argc) filmThickness = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-filmior") && i + 1 < argc) filmIor = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-checkthinfilm")) checkThinFilmOnly = true;
        else if (!std::strcmp(argv[i], "-checkmultilayer")) checkMultilayerOnly = true;
        else if (!std::strcmp(argv[i], "-thinfilmswatch")) thinFilmSwatchOnly = true;
        else if (!std::strcmp(argv[i], "-diffraction") && i + 1 < argc) {
            const char* v = argv[++i];
            diffraction = !(std::strcmp(v, "off") == 0 || std::strcmp(v, "0") == 0);
        }
        else if (!std::strcmp(argv[i], "-nodiffraction")) diffraction = false;
        else if (!std::strcmp(argv[i], "-checkgrating")) checkGratingOnly = true;
        else if (!std::strcmp(argv[i], "-checkupsample")) checkUpsampleOnly = true;
        else if (!std::strcmp(argv[i], "-checkgrid")) checkGridOnly = true;
        else if (!std::strcmp(argv[i], "-checkscatter")) checkScatterOnly = true;
        else if (!std::strcmp(argv[i], "-checkbind")) checkBindOnly = true;
        else if (!std::strcmp(argv[i], "-checkprop")) checkPropOnly = true;
        else if (!std::strcmp(argv[i], "-checkarray")) checkArrayOnly = true;
        else if (!std::strcmp(argv[i], "-checksun")) checkSunOnly = true;
        else if (!std::strcmp(argv[i], "-device") && i + 1 < argc) device = argv[++i];
        else if (!std::strcmp(argv[i], "-wavefront")) wavefront = true;
        else if (!std::strcmp(argv[i], "-rgb")) rgbBackward = true;
        else if (!std::strcmp(argv[i], "-no-media") || !std::strcmp(argv[i], "-nomedia")) noMedia = true;
        else if (!std::strcmp(argv[i], "-no-env") || !std::strcmp(argv[i], "-noenv")) noEnv = true;
        else if (!std::strcmp(argv[i], "-no-fluoro") || !std::strcmp(argv[i], "-nofluoro")) noFluoro = true;
        else if (!std::strcmp(argv[i], "-direct-only") || !std::strcmp(argv[i], "-directonly")) directOnly = true;
        else if (!std::strcmp(argv[i], "-max-bounce") && i + 1 < argc) maxBounceOverride = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-time") && i + 1 < argc) timeBudgetSec = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-noise") && i + 1 < argc) noiseTarget = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-forever")) runForever = true;
        else if (!std::strcmp(argv[i], "-preview")) preview = true;
        else if (!std::strcmp(argv[i], "-beams") || !std::strcmp(argv[i], "-photonbeams")) g_beamGather = true;
        else if (!std::strcmp(argv[i], "-window")) g_showWindow = true;
        else if (!std::strcmp(argv[i], "-keepwindow") || !std::strcmp(argv[i], "-hold")) { g_showWindow = true; g_keepWindow = true; }
        else if (!std::strcmp(argv[i], "-raster")) doRaster = true;
        else if (!std::strcmp(argv[i], "-raster-gpu")) { doRaster = true; rasterGpu = true; }
        else if (!std::strcmp(argv[i], "-raster-bench") && i + 1 < argc) { rasterBench = std::atoi(argv[++i]); doRaster = true; }
        else if (!std::strcmp(argv[i], "-explore") || !std::strcmp(argv[i], "-fly")) {
            // Interactive fly-through: start at the first selected camera frame and let
            // the user explore with the raster viewer instead of rendering every frame.
            // The exposure-lock metering pre-pass is pointless here (the viewer auto-exposes
            // per frame), and metering a whole flyby's frames just to fly one is wasteful,
            // so explore implies -no-meter.
            exploreMode = true; doRaster = true; g_showWindow = true; g_keepWindow = true; noMeter = true;
        }
        else if (!std::strcmp(argv[i], "-anim") && i + 1 < argc) {
            // Edit a loom `CurveDrive` sidecar (E2 channel a) instead of a bare camera
            // path: the editor's control points ARE the drive's N-D curve, and Save
            // writes the reshaped curve back to this file. Implies the fly editor.
            animSidecar = argv[++i];
            exploreMode = true; doRaster = true; g_showWindow = true; g_keepWindow = true; noMeter = true;
        }
        else if ((!std::strcmp(argv[i], "-loom") || !std::strcmp(argv[i], "--loom")) && i + 1 < argc) {
            // With -anim, the loom build file to open the E2 **live** channel against:
            // the editor spawns `python -m loom.anim <scene.py> --config <sidecar>` and
            // every scrub position becomes a freshly-emitted .ftsl. (With -viewer the
            // same flag names the F4 re-introspection scene; that pre-scan runs earlier.)
            animLoomScene = argv[++i];
        }
        else if (!std::strcmp(argv[i], "-no-meter") || !std::strcmp(argv[i], "-nometer")) noMeter = true;
        else if (!std::strcmp(argv[i], "-noclip") || !std::strcmp(argv[i], "-nocollide")) viewerNoclip = true;
        else if (!std::strcmp(argv[i], "-raster-iso") && i + 1 < argc) rasterIso = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-see-through") || !std::strcmp(argv[i], "-seethrough") || !std::strcmp(argv[i], "-glass")) rasterSeeThrough = true;
        else if (!std::strcmp(argv[i], "-glass-clarity") && i + 1 < argc) { rasterClarity = std::clamp(std::atof(argv[++i]), 0.0, 1.0); rasterSeeThrough = true; }
        else if (!std::strcmp(argv[i], "-exposure-lock")) forceExposureLock = true;
        else if (!std::strcmp(argv[i], "-stereo") && i + 1 < argc) {
            std::string m = argv[++i];
            for (auto& c : m) c = (char)std::tolower((unsigned char)c);
            if      (m=="sbs"||m=="side-by-side"||m=="wall"||m=="walleye"||m=="wall-eyed") stereoMode = STEREO_SBS;
            else if (m=="cross"||m=="sbs-cross"||m=="crosseye"||m=="cross-eyed")           stereoMode = STEREO_CROSS;
            else if (m=="anaglyph"||m=="rc"||m=="red-cyan"||m=="redcyan")                  stereoMode = STEREO_ANAGLYPH_RC;
            else if (m=="anaglyph-gm"||m=="gm"||m=="green-magenta"||m=="greenmagenta")     stereoMode = STEREO_ANAGLYPH_GM;
            else { std::fprintf(stderr, "error: -stereo mode '%s' unknown "
                                        "(sbs|cross|anaglyph|anaglyph-gm)\n", m.c_str()); return 1; }
        }
        else if (!std::strcmp(argv[i], "-eye-sep") && i + 1 < argc) stereoEyeSep = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-view-dist") && i + 1 < argc) stereoViewDist = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-dpi") && i + 1 < argc) {
            const char* v = argv[++i];
            if (!std::strcmp(v, "auto")) {
                stereoDpi = stereoDetectDpi();
                if (stereoDpi > 0.0)
                    std::printf("[stereo] -dpi auto -> %.0f (logical system DPI; pass a measured "
                                "-dpi for a physically exact baseline)\n", stereoDpi);
                else
                    std::fprintf(stderr, "[stereo] -dpi auto: could not detect DPI; using the "
                                         "-view-dist/FOV screen-width mapping instead\n");
            } else stereoDpi = std::atof(v);
        }
        else if (!std::strcmp(argv[i], "-convergence") && i + 1 < argc) stereoConverge = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-stereo-keep-eyes")) stereoKeepEyes = true;
        else if (!std::strcmp(argv[i], "-interval") && i + 1 < argc) intervalSec = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-resume")) resume = true;
        else if (!std::strcmp(argv[i], "-checkpoint")) wantCheckpointFlag = true;
        else if (!std::strcmp(argv[i], "-in") && i + 1 < argc) ++i; // handled in pre-scan
        else if (!std::strcmp(argv[i], "-serve")) { /* resident loop; driven by main(), ignored here */ }
        // Retired in 0.79.0 along with the hand-written parser they selected. Accepted so
        // an existing script does not hit the unknown-flag error, but SAID OUT LOUD — a
        // flag that quietly stopped doing anything is worse than one that is gone.
        else if (!std::strcmp(argv[i], "-legacy-parser") ||
                 !std::strcmp(argv[i], "-validate-grammar")) {
            std::fprintf(stderr, "ftrace: %s was retired in 0.79.0 — the shared grammar "
                                 "is the only .ftsl front end now; ignoring\n", argv[i]);
        }
        else if (argv[i][0] == '-') {
            // Any remaining dash-prefixed token is an unrecognized (or malformed, e.g.
            // value-less) option. Fail loudly instead of silently falling through to the
            // default demo render — a typo'd flag should never masquerade as a real run.
            // (Non-dash positionals, e.g. a scene file, were consumed by the -in pre-scan.)
            std::fprintf(stderr, "ftrace: unknown option '%s' (try -h / --help)\n", argv[i]);
            return 2;
        }
    }
    if (nThreads < 1) nThreads = 1;

    // Bare-invocation quick preview: `ftrace scene.ftsl` (double-click / drag-drop, no
    // other flags) defaults to a fast raster preview shown in a live window — no light
    // transport, no stray output file. If the user asked for any real-render control
    // (a mode/budget/device/camera flag, an explicit -raster, etc.) we respect that and
    // don't force preview.
    if (positionalScene && !doRaster) {
        // A scene file's preview yields to ANY render-control flag. The quick MESH viewer
        // is fundamentally a preview, so it stays a raster preview even with presentation
        // flags (-window/-o/-r/-camera/-view) and only yields to a genuine light-transport
        // request (-mode and the budget/device/map flags). So `ftrace model.glb -window`
        // shows a raster preview in a window, while `ftrace model.glb -mode D -n 1e8`
        // renders it for real.
        static const char* kSceneRenderFlags[] = {
            "-mode","-n","-time","-noise","-forever","-preview","-spp","-device",
            "-camera","-view","-savemap","-loadmap","-wavefront","-o","-r","-window"
        };
        static const char* kMeshRenderFlags[] = {
            "-mode","-n","-time","-noise","-forever","-preview","-spp","-device",
            "-savemap","-loadmap","-wavefront"
        };
        bool explicitControl = false;
        auto scan = [&](const char* const* flags, size_t nflags) {
            for (int i = 1; i < argc && !explicitControl; ++i)
                for (size_t k = 0; k < nflags; ++k)
                    if (!std::strcmp(argv[i], flags[k])) { explicitControl = true; break; }
        };
        if (positionalMesh) scan(kMeshRenderFlags, sizeof(kMeshRenderFlags)/sizeof(*kMeshRenderFlags));
        else                scan(kSceneRenderFlags, sizeof(kSceneRenderFlags)/sizeof(*kSceneRenderFlags));
        if (!explicitControl) {
            doRaster = true;
            g_showWindow = true;
            g_keepWindow = true;   // double-click preview: hold the image open until the
                                   // user closes the window (don't flash-and-vanish)
            // Don't drop a stray cornell.ppm next to the cwd: send the preview PNG to a
            // temp path derived from the scene name. (Window is the real deliverable.)
            if (!std::strcmp(out, "cornell.ppm")) {
                const char* tmp = std::getenv("TEMP");
                if (!tmp) tmp = std::getenv("TMPDIR");
                if (!tmp) tmp = ".";
                std::string base = inFile;
                size_t slash = base.find_last_of("/\\");
                if (slash != std::string::npos) base = base.substr(slash + 1);
                size_t dot = base.find_last_of('.');
                if (dot != std::string::npos) base = base.substr(0, dot);
                static std::string previewOut = std::string(tmp) + "/ftrace_preview_" + base + ".png";
                out = previewOut.c_str();
            }
        }
    }

    // Name the live-preview window after what it is rendering: "ftrace — <scene> → <out>"
    // (em dash + right-arrow are UTF-8; livewindow decodes them properly). The scene is
    // the -in file when given, else the built-in scene name; the output is the -o target.
    {
        std::string scene = inFile ? inFile : sceneName;
        g_windowTitle = "ftrace  \xE2\x80\x94  " + scene + "  \xE2\x86\x92  " + out;
    }
    if (checkImplicitOnly) return checkImplicit(500'000) == 0 ? 0 : 1; // deterministic, no scene needed
    if (checkLensOnly)     return checkLens();     // deterministic, no scene needed
    if (checkFluoroOnly)   return checkFluoro();   // deterministic, no scene needed
    if (checkFogOnly)      return checkFog();      // deterministic, no scene needed
    if (checkThinFilmOnly) return checkThinFilm(); // deterministic, no scene needed
    if (checkMultilayerOnly) return checkMultilayer(); // deterministic, no scene needed
    if (thinFilmSwatchOnly) { thinFilmSwatch(filmIor, 1.5); return 0; } // visual diagnostic
    if (checkGratingOnly)  return checkGrating();  // deterministic, no scene needed
    if (checkUpsampleOnly) return checkUpsample(); // deterministic, no scene needed
    if (checkGridOnly)     return checkGrid();     // deterministic, no scene needed
    if (checkScatterOnly)  return checkScatter();  // ditto (the ragged sibling)
    if (checkBindOnly)     return checkBind();     // deterministic, no scene needed
    if (checkPropOnly)     return checkProp();     // ditto (loads in-memory scenes only)
    if (checkArrayOnly)    return checkArray();    // ditto
    if (checkSunOnly)      return checkSun();      // deterministic, no scene needed

    // --- every output directory must exist BEFORE a single photon is traced ----------
    // Otherwise a mistyped/not-yet-created output directory used to be discovered only
    // by the first writer: the render ran to completion, printed "error: could not
    // write ..." at every -interval tick, and exited having thrown the entire
    // accumulated film away. Resolve the parents once, here, where `out` is final (the
    // bare-invocation preview path above can still rewrite it).
    //
    // `-o` covers most of it: the `.ftbuf` checkpoint sidecar (`out + ".ftbuf"`), the
    // per-camera `outFor()` variants and the stereo eye pair all live beside it.
    // `-savemap` is the one independent path, and a discarded photon map costs just as
    // much as a discarded film.
    //
    // Policy: create the directory. Renders are routinely aimed at a fresh per-series
    // subdirectory (png/<setname>/), and refusing to make one would be a pointless
    // extra step. But say so on stdout, so a typo shows up as a surprise directory in
    // the log rather than silently; and if creation fails, bail NOW with a clear
    // message instead of rendering into the void.
    {
        namespace fs = std::filesystem;
        auto ensureOutDir = [](const char* what, const std::string& file) -> bool {
            std::error_code ec;
            fs::path parent = fs::path(file).parent_path();
            if (parent.empty() || fs::is_directory(parent, ec)) return true;
            if (fs::exists(parent, ec)) {
                std::fprintf(stderr, "ftrace: %s path '%s' exists but is not a directory "
                                     "(from %s)\n",
                             what, parent.string().c_str(), file.c_str());
                return false;
            }
            ec.clear();
            fs::create_directories(parent, ec);
            if (ec || !fs::is_directory(parent)) {
                std::fprintf(stderr, "ftrace: %s directory '%s' does not exist and could "
                                     "not be created: %s\n",
                             what, parent.string().c_str(),
                             ec ? ec.message().c_str() : "unknown error");
                return false;
            }
            std::printf("[out] created %s directory %s\n", what, parent.string().c_str());
            return true;
        };
        if (!ensureOutDir("output", out)) return 2;
        if (!g_pmapSave.empty() && !ensureOutDir("-savemap", g_pmapSave)) return 2;
    }

    bool prism     = !std::strcmp(sceneName, "prism");
    bool materials = !std::strcmp(sceneName, "materials");
    bool fluoro    = !std::strcmp(sceneName, "fluoro");
    bool iridescent = !std::strcmp(sceneName, "iridescent");
    bool grating    = !std::strcmp(sceneName, "grating");

    selfTestColor();

    // Modes R (backward reference) and V (validate: forward vs backward) need an
    // all-diffuse scene so the known model-B specular limitation doesn't pollute
    // the comparison — use a diffuse sphere when no mesh is supplied. Mode D (BDPT)
    // uses the same all-diffuse built-in cornell so it can be diffed directly against
    // mode R as the primary validation; to exercise D on specular/glossy surfaces use
    // -scene materials (which builds the mirror+glossy scene for every mode).
    const bool refMode = (mode == 'R' || mode == 'V');
    const bool diffuseScene = refMode || mode == 'D';
    Scene scene = fromFtsl  ? std::move(ftslScene.scene)
                : prism     ? buildPrism(res)
                : grating   ? buildGrating(res, diffraction)
                : materials ? buildMaterials(res, resolveLight(lightName))
                            : buildCornell(res, mode, resolveLight(lightName),
                                           fluoro ? nullptr : meshPath, meshScale,
                                           /*diffuseSphere*/diffuseScene, /*fluoroSphere*/fluoro,
                                           /*thinFilmSphere*/iridescent, filmThickness, filmIor);

    // Optional global fog / participating medium (-fog sigma_t). With -fograyleigh
    // the scattering coefficient varies as (550/lambda)^4, so short wavelengths
    // scatter far more — a bluish haze that transmits red (a spectral sky/sunset).
    if (fogSigmaT > 0.0) {
        Medium fog;
        fog.enabled = true;
        fog.g = fogG;
        double ss = fogAlbedo * fogSigmaT, sa = (1.0 - fogAlbedo) * fogSigmaT;
        if (fogRayleigh) {
            fog.sigma_s = [ss](double w) { double r = 550.0 / w; double r2 = r * r; return ss * r2 * r2; };
            fog.sigma_a = constantSpectrum(sa);
        } else {
            fog.sigma_s = constantSpectrum(ss);
            fog.sigma_a = constantSpectrum(sa);
        }
        scene.media.push_back(std::move(fog));
    }

    // Scene-ignore flags (Stage 3): strip expensive features for a faster preview,
    // "like the rasterizer does". Applied after all scene construction (incl. -fog)
    // so it sees the final scene. Pure mutation; maxBounce / directOnly are render
    // params threaded into runRender below.
    if (noMedia || noEnv || noFluoro) {
        std::string removed = scene.applyIgnoreFlags(noMedia, noEnv, noFluoro);
        if (!removed.empty())
            std::printf("[ignore] stripped: %s\n", removed.c_str());
    }
    // Publish the depth cap / direct-only mode to the tracer wrappers (globals, like
    // g_heroC), so every render (incl. the meter pre-pass) honours them.
    g_maxBounceOverride = maxBounceOverride;
    // An authored `mode W` is normalised to 'R' by the loader (ftsl::normMode); this is
    // the half that turns the deterministic estimators on. A CLI `-mode <x>` forces every
    // camera, so it also overrides the scene's choice of W — otherwise `-mode R` on a W
    // scene would silently still render the biased preview.
    if (ftslScene.whitted && !modeFromCli) g_whitted = true;
    g_directOnly = directOnly || g_whitted;   // -mode W implies it
    if (maxBounceOverride >= 1) std::printf("[ignore] max bounce = %d\n", maxBounceOverride);
    // The interactive viewer renders its live preview in mode W whatever the run's mode is
    // (press T), so mode W's settings have to be honoured in an -explore run too -- else
    // `-explore -gi 32` would silently drop the gather and the preview would come out with
    // a 1-wavelength bundle. There is no batch render in an explore run, so widening the
    // bundle here can't slow anything else down.
    const bool wPreview = exploreMode;
    if (g_whitted || wPreview) {
        // Widen the spectral bundle by default (see g_heroCSet): at 1 spp the C hero
        // wavelengths ARE the whole spectral quadrature, and they share one BVH walk,
        // so this is the cheapest accuracy in the mode.
        if (!g_heroCSet) g_heroC = hero::kHeroMax;
    }
    if (g_whitted) {
        std::printf("[mode W] deterministic Whitted preview: %dx%d shadow rays/light, "
                    "%d wavelengths/sample%s, ambient %.3g\n",
                    g_whittedGrid, g_whittedGrid, g_heroC,
                    g_heroC > 1 ? " (split at dispersion)" : "", g_ambient);
        if (g_gi > 0) {
            std::printf("[mode W] one-bounce gather: %d rays/diffuse vertex, %dx%d shadow "
                        "rays at gather vertices, <=%d bounces/gather ray (cacheless, so "
                        "temporally stable)\n", g_gi, g_giGrid, g_giGrid, g_giBounce);
            if (g_giClamp > 0.0)
                std::printf("[mode W] gather firefly clamp: one gather ray capped at %.3g "
                            "of a light's own radiance\n", g_giClamp);
        }
    }
    else if (directOnly) std::printf("[ignore] direct-only (no diffuse indirect)\n");
    // Kept out of the chain above: rejecting -gi is independent of whether the run is
    // also direct-only, and folding it in would swallow that notice when both are given.
    // Mode R already carries real multi-bounce GI; the gather is mode W's substitute for
    // it, so silently accepting -gi anywhere else would just be misleading.
    // (wPreview spares it: the viewer's T preview IS mode W, so -gi is live there.)
    if (!g_whitted && !wPreview && g_gi > 0) {
        std::printf("[ignore] -gi %d needs -mode W (other modes either have real GI or no "
                    "diffuse transport at all)\n", g_gi);
        g_gi = 0;
    }
    // -gi-clamp only ever reads inside the gather, so it is dead without -gi. Say so rather
    // than letting someone tune a value that cannot do anything (and note that g_gi may have
    // just been zeroed above, which is exactly one of the ways to get here).
    if (g_giClamp > 0.0 && g_gi == 0)
        std::printf("[ignore] -gi-clamp %.3g does nothing without -gi (it caps a GATHER "
                    "ray; the flat -ambient fill is not clamped)\n", g_giClamp);
    // -herosplit reaches the CPU forward tracer, the photon maps, and the backward tracer
    // (modes R/W) on BOTH the CPU and the GPU (the device split is bkRadianceHeroLoop<true>,
    // v0.111.0). The GPU FORWARD megakernel still de-heros, so name the layers rather than
    // silently ignoring it, and point out that it is a no-op without a bundle to split. Mode W
    // enables it itself (see BackwardRenderer::heroSplit), so it is reported there instead.
    if (hero::gSplit && !g_whitted) {
        if (g_heroC <= 1)
            std::printf("[hero] -herosplit has no effect with -heroc 1 (no secondaries to split)\n");
        else
            std::printf("[hero] split-at-dispersion ON (C=%d fan-out; backward R/W on CPU+GPU, "
                        "forward modes A/B/C and photon-map M/S on the CPU)\n", g_heroC);
    }

    if (checkBvhOnly) {
        // Bound the linear-reference work (~O(rays * prims)) so the self-test
        // stays fast even for big meshes: ~5e8 primitive tests, clamped.
        long long prims = (long long)scene.tris.size() + (long long)scene.spheres.size();
        long long rays = 500'000'000LL / (prims > 0 ? prims : 1);
        rays = std::clamp(rays, 20'000LL, 2'000'000LL);
        return checkBvh(scene, rays) == 0 ? 0 : 1;
    }
    if (bvhStatsOnly) { bvhStats(scene, 500'000); return 0; }

    // -check-watertight (alias -airtight): audit every named mesh and every isosurface
    // for a closed, consistently-oriented surface, warn about any that aren't, then exit.
    // Non-watertight geometry (holes / non-manifold edges) or flipped-normal facets break
    // the renderer's dielectric enter/exit + interior-medium tracking, so glass built on
    // such a surface refracts wrong and can splash artifacts elsewhere in the scene. The
    // check is informational for opaque materials but emphasised (!) for dielectrics.
    // Isosurfaces are polygonised at -mesh-res (default 128) before checking.
    if (checkWatertight) {
        auto dielectric = [&](int matId) {
            return matId >= 0 && matId < (int)scene.mats.size() &&
                   scene.mats[matId].type == MatType::Dielectric;
        };
        int failures = 0, checked = 0;
        std::printf("[check-watertight] auditing %zu mesh object(s) and %zu isosurface(s)\n",
                    scene.meshGroups.size(), scene.implicits.size());
        auto report = [&](const std::string& kind, const std::string& name, int matId,
                          const watertight::Report& r) {
            ++checked;
            bool glass = dielectric(matId);
            if (r.ok()) {
                std::printf("  [OK]   %-11s \"%s\"  (%zu tris, %zu verts, watertight%s)\n",
                            kind.c_str(), name.c_str(), r.tris, r.verts,
                            glass ? ", dielectric" : "");
                return;
            }
            ++failures;
            std::printf("  [WARN%s] %-9s \"%s\"  NOT airtight (%zu tris):\n",
                        glass ? "!" : " ", kind.c_str(), name.c_str(), r.tris);
            if (r.boundary)    std::printf("           - %zu boundary edge(s) (holes / open border)\n", r.boundary);
            if (r.nonManifold) std::printf("           - %zu non-manifold edge(s) (3+ faces share an edge)\n", r.nonManifold);
            if (r.flipped)     std::printf("           - %zu inconsistently-wound edge(s) (some normals point inward)\n", r.flipped);
            if (glass)         std::printf("           ! this object is DIELECTRIC (glass) — refraction WILL be wrong\n");
        };
        for (const auto& g : scene.meshGroups) {
            watertight::Report r = (g.blasId >= 0 && g.blasId < (int)scene.blasList.size())
                ? watertight::checkTris(scene.blasList[g.blasId].tris.data(), scene.blasList[g.blasId].tris.size())
                : watertight::checkTris(scene.tris.data() + g.triStart, g.triCount);
            report("mesh", g.name, g.matId, r);
        }
        for (size_t k = 0; k < scene.implicits.size(); ++k) {
            const Implicit& im = scene.implicits[k];
            std::string name = im.name.empty() ? ("isosurface_" + std::to_string(k)) : im.name;
            isomesh::Options mo; mo.res = std::max(2, exportMeshRes);
            const PatTables tabs = scene.patTables();
            isomesh::Mesh m = isomesh::marchImplicit(im, mo, &tabs);
            watertight::Report r = watertight::check(m.pos, m.tri);
            report("isosurface", name, im.matId, r);
        }
        if (checked == 0)
            std::printf("[check-watertight] scene has no named mesh or isosurface objects to check\n");
        else if (failures == 0)
            std::printf("[check-watertight] all %d object(s) are airtight.\n", checked);
        else
            std::printf("[check-watertight] %d of %d object(s) are NOT airtight (see warnings above).\n",
                        failures, checked);
        return failures ? 1 : 0;
    }

    // -check-airtight: ray-parity audit of every isosurface's *marched* field — the
    // exact zero level-set the renderer sphere-traces, not a polygonised proxy. Fires
    // random exterior->exterior chords; a closed solid crosses the boundary an even
    // number of times, so any ODD count is a leak (an open cap on an uncapped surface,
    // or a Lipschitz/thin-feature overshoot the renderer would show as a light leak).
    if (checkAirtight) {
        auto dielectric = [&](int matId) {
            return matId >= 0 && matId < (int)scene.mats.size() &&
                   scene.mats[matId].type == MatType::Dielectric;
        };
        if (scene.implicits.empty()) {
            std::printf("[check-airtight] scene has no isosurfaces to audit\n");
            return 0;
        }
        std::printf("[check-airtight] auditing %zu isosurface(s) with %lld chords each "
                    "(probing the marched field directly)\n",
                    scene.implicits.size(), airtightRays);
        int failures = 0;
        for (size_t k = 0; k < scene.implicits.size(); ++k) {
            const Implicit& im = scene.implicits[k];
            std::string name = im.name.empty() ? ("isosurface_" + std::to_string(k)) : im.name;
            const PatTables tabs = scene.patTables();
            airtight::Report r = airtight::check(im, airtightRays, 0x9E3779B97F4A7C15ull + k, &tabs);
            bool glass = dielectric(im.matId);
            if (r.degenerate) {
                std::printf("  [SKIP] \"%s\"  degenerate/unbounded container — no valid chords\n",
                            name.c_str());
                continue;
            }
            if (r.airtight()) {
                std::printf("  [OK]   \"%s\"  airtight  (%lld chords, %s, %s%s)\n",
                            name.c_str(), r.chords, r.open ? "open" : "capped",
                            r.overshoot ? "marcher-clean" : "no overshoot",
                            glass ? ", dielectric" : "");
                if (r.overshoot)
                    std::printf("           note: %.2f%% of chords show the marcher finding fewer\n"
                                "           crossings than a dense reference — thin features near the\n"
                                "           march step; parity still even. Consider raising max_gradient.\n",
                                100.0 * r.overFrac());
                continue;
            }
            ++failures;
            std::printf("  [WARN%s] \"%s\"  NOT airtight  (%lld chords):\n",
                        glass ? "!" : " ", name.c_str(), r.chords);
            if (r.oddParity)
                std::printf("           - %.2f%% of chords cross the boundary an ODD number of times\n"
                            "             (%lld/%lld) — the interior connects to the exterior (a leak)\n",
                            100.0 * r.oddFrac(), r.oddParity, r.chords);
            if (r.open && r.boundaryInside)
                std::printf("           - the solid touches the container wall on %.2f%% of boundary\n"
                            "             samples (%lld/%lld, worst f=%.3g) while the surface is OPEN —\n"
                            "             an open cap. Add `capped` or shrink `contained_by` to seal it.\n",
                            100.0 * r.capFrac(), r.boundaryInside, r.boundarySamples, r.worstMinF);
            if (r.overshoot)
                std::printf("           - %.2f%% of chords: marcher misses crossings the dense reference\n"
                            "             finds — Lipschitz/max_gradient overshoot or sub-step features\n",
                            100.0 * r.overFrac());
            if (glass)
                std::printf("           ! this object is DIELECTRIC (glass) — refraction WILL leak light\n");
        }
        if (failures == 0)
            std::printf("[check-airtight] all %zu isosurface(s) are airtight.\n", scene.implicits.size());
        else
            std::printf("[check-airtight] %d of %zu isosurface(s) are NOT airtight (see above).\n",
                        failures, scene.implicits.size());
        return failures ? 1 : 0;
    }

    // -export-mesh <file.obj>: polygonise every isosurface in the scene into a
    // watertight triangle mesh (marching cubes) and write an OBJ for import into
    // Unreal / Blender / etc., then exit. -mesh-res sets fineness (cells along the
    // longest bounds axis). -mesh-adaptive runs a curvature-driven QEM decimation
    // pass so triangles concentrate where the surface bends and thin out where it
    // is flat, while staying watertight.
    if (exportMeshPath) {
        if (scene.implicits.empty()) {
            std::fprintf(stderr, "[export-mesh] ERROR: scene has no isosurface to export\n");
            return 1;
        }
        isomesh::Options mo;
        mo.res = std::max(2, exportMeshRes);
        mo.adaptive = exportMeshAdaptive;
        mo.decimate = std::clamp(exportMeshDecimate, 0.01, 1.0);
        auto logfn = [](const std::string& s) { std::printf("%s\n", s.c_str()); };
        std::vector<std::pair<std::string, isomesh::Mesh>> groups;
        for (size_t k = 0; k < scene.implicits.size(); ++k) {
            std::printf("[export-mesh] marching isosurface %zu/%zu at res %d ...\n",
                        k + 1, scene.implicits.size(), mo.res);
            const PatTables tabs = scene.patTables();
            isomesh::Mesh m = isomesh::marchImplicit(scene.implicits[k], mo, &tabs);
            std::printf("[export-mesh]   marched: %zu verts, %zu tris\n",
                        m.pos.size(), m.tri.size() / 3);
            if (mo.adaptive && !m.tri.empty()) {
                size_t before = m.tri.size() / 3;
                isomesh::decimateAdaptive(m, mo.decimate, scene.implicits[k]);
                std::printf("[export-mesh]   decimated: %zu -> %zu tris (target %.0f%%)\n",
                            before, m.tri.size() / 3, mo.decimate * 100.0);
            }
            // Name the OBJ group after the object's authored ftsl name when it has
            // one; fall back to isosurface_<k> for unnamed blocks. Sanitise to a safe
            // OBJ group token (OBJ `g` names can't contain whitespace) and de-duplicate.
            std::string gname = scene.implicits[k].name;
            for (char& c : gname) { if (std::isspace((unsigned char)c)) c = '_'; }
            if (gname.empty()) gname = "isosurface_" + std::to_string(k);
            {
                std::string base = gname; int dup = 1;
                auto taken = [&](const std::string& n) {
                    for (const auto& g : groups) if (g.first == n) return true;
                    return false;
                };
                while (taken(gname)) gname = base + "_" + std::to_string(++dup);
            }
            groups.emplace_back(std::move(gname), std::move(m));
        }
        bool ok = isomesh::writeObj(exportMeshPath, groups, logfn);
        return ok ? 0 : 1;
    }

    // Build the list of cameras to render. FTSL scenes may declare several; a
    // built-in scene has exactly one. Each render camera carries its own effective
    // mode and film resolution (per-camera FTSL values, unless a CLI -mode/-r forces
    // them globally). All cameras share the single already-built scene.
    const char* lightLabel = (prism || grating) ? "beam" : lightName;
    auto effMode = [&](char camMode) -> char {
        if (modeFromCli) return mode;         // CLI -mode forces every camera
        return camMode ? camMode : mode;      // else per-camera, else the global default
    };
    struct RenderCam { std::string name; Camera cam; char mode; int res; int resY; double exposure; int expGroup;
                       Vec3 lookAt{0,0,0}; Vec3 up{0,1,0}; double fovY = 40.0; };  // lookAt/up/fovY: for the interactive raster viewer
    std::vector<RenderCam> toRender;

    // Raster previews are cheap to compute, so unless the user pinned a size with -r,
    // scale each preview camera UP so its long edge is at least kRasterPreviewLong px
    // (aspect preserved — the same scale on both axes, so the camera's tanHalfX/Y still
    // match). A 256²-authored test camera then previews large and readable in the live
    // window instead of tiny; already-large cameras are left untouched, and real
    // (light-transport) renders always keep their authored resolution.
    const int kRasterPreviewLong = 1440;
    auto previewUpscale = [&](int& rx, int& ry) {
        if (!doRaster || resFromCli) return;
        int lo = std::max(rx, ry);
        if (lo > 0 && lo < kRasterPreviewLong) {
            double s = (double)kRasterPreviewLong / lo;
            rx = std::max(1, (int)std::lround(rx * s));
            ry = std::max(1, (int)std::lround(ry * s));
        }
    };

    // -view against a loaded (-in) scene: inject an ad-hoc 'view' CamSpec and
    // render only it (so the live -window/-preview shows exactly that angle).
    // A built-in scene has no CamSpec list; its view override is applied in the
    // else branch below.
    if (haveView && fromFtsl) {
        ftsl::CamSpec vc;
        vc.name = "view"; vc.eye = viewEye; vc.look = viewLook; vc.up = viewUp; vc.fov = viewFov;
        ftslScene.cameras.push_back(vc);
        cameraSel = "view";
    }

    if (fromFtsl && !ftslScene.cameras.empty()) {
        // Select which cameras to render. `-camera` accepts:
        //   all           every camera (default when several are declared)
        //   <name>        exact camera/frame name (e.g. hero, fly137)
        //   <pathbase>    a whole camera_path/curve/orbit by its base name (e.g.
        //                 `fly` selects fly000..fly143 but not an unrelated still)
        //   #N            the Nth declared camera, 0-based (#-1 = last)
        //   near=X,Y,Z    the camera whose eye is closest to (X,Y,Z)
        // The index and nearest selectors make it easy to aim the live window at
        // one frame of a long camera_curve without hunting for its frame name.
        std::vector<const ftsl::CamSpec*> sel;
        if (cameraSel && std::strcmp(cameraSel, "all") != 0) {
            const std::string q = cameraSel;
            if (!q.empty() && q[0] == '#') {
                int count = (int)ftslScene.cameras.size();
                int n = std::atoi(q.c_str() + 1);
                if (n < 0) n += count;
                if (n < 0 || n >= count) {
                    std::fprintf(stderr, "[camera] index '%s' out of range (have %d cameras: 0..%d)\n",
                                 q.c_str(), count, count - 1);
                    return 1;
                }
                sel.push_back(&ftslScene.cameras[n]);
                std::printf("[camera] index %s -> '%s'\n", q.c_str(), ftslScene.cameras[n].name.c_str());
            } else if (q.rfind("near=", 0) == 0 || q.rfind("near:", 0) == 0) {
                double xyz[3] = {0,0,0}; int nv = 0;
                for (const char* p = q.c_str() + 5; *p && nv < 3; ) {
                    char* e = nullptr; double val = std::strtod(p, &e);
                    if (e == p) break;
                    xyz[nv++] = val; p = e;
                    while (*p == ',' || *p == ' ') ++p;
                }
                if (nv < 3) { std::fprintf(stderr, "[camera] -camera near= needs X,Y,Z\n"); return 1; }
                Vec3 target{xyz[0], xyz[1], xyz[2]};
                const ftsl::CamSpec* best = nullptr; double bestD2 = 1e300;
                for (const auto& cs : ftslScene.cameras) {
                    Vec3 d = cs.eye - target; double d2 = dot(d, d);
                    if (d2 < bestD2) { bestD2 = d2; best = &cs; }
                }
                sel.push_back(best);
                std::printf("[camera] nearest to (%.3f,%.3f,%.3f) is '%s' (eye %.3f,%.3f,%.3f, dist %.3f)\n",
                            target.x, target.y, target.z, best->name.c_str(),
                            best->eye.x, best->eye.y, best->eye.z, std::sqrt(bestD2));
            } else {
                for (const auto& cs : ftslScene.cameras)
                    if (cs.name == cameraSel) sel.push_back(&cs);
                // No exact hit? Treat the query as a camera_path base name and select
                // every frame named "<q>NNN" (q followed by digits only) — so
                // `-camera fly` grabs the whole `camera_curve "fly"` (fly000..fly143)
                // while excluding an unrelated still like `cam`.
                if (sel.empty()) {
                    for (const auto& cs : ftslScene.cameras) {
                        if (cs.name.size() > q.size() && cs.name.compare(0, q.size(), q) == 0) {
                            bool allDigits = true;
                            for (size_t k = q.size(); k < cs.name.size(); ++k)
                                if (!std::isdigit((unsigned char)cs.name[k])) { allDigits = false; break; }
                            if (allDigits) sel.push_back(&cs);
                        }
                    }
                    if (!sel.empty()) {
                        // Resolve the flyby's playback fps hint (per-camera, else scene
                        // default) so a user running ftrace directly sees the authored rate
                        // the video tooling will assemble at; 0 => none authored.
                        double pfps = (sel.front()->fps > 0.0) ? sel.front()->fps
                                                               : ftslScene.defaultFps;
                        if (pfps > 0.0)
                            std::printf("[camera] path '%s' -> %zu frames (%s..%s) @ %g fps\n",
                                        q.c_str(), sel.size(), sel.front()->name.c_str(),
                                        sel.back()->name.c_str(), pfps);
                        else
                            std::printf("[camera] path '%s' -> %zu frames (%s..%s)\n",
                                        q.c_str(), sel.size(), sel.front()->name.c_str(),
                                        sel.back()->name.c_str());
                    }
                }
                if (sel.empty()) {
                    std::fprintf(stderr, "[camera] no camera named '%s' (have:", cameraSel);
                    for (const auto& cs : ftslScene.cameras)
                        std::fprintf(stderr, " %s", cs.name.c_str());
                    std::fprintf(stderr, ")\n");
                    return 1;
                }
            }
        } else {
            for (const auto& cs : ftslScene.cameras) sel.push_back(&cs);
        }
        for (const ftsl::CamSpec* cs : sel) {
            int cresX = resFromCli ? res : (cs->res  > 0 ? cs->res  : res);
            int cresY = resFromCli ? (resYCli > 0 ? resYCli : res)
                                   : (cs->resY > 0 ? cs->resY : cresX);
            previewUpscale(cresX, cresY);   // big, readable raster preview (no-op for real renders)
            Camera c;
            c.lookAt(cs->eye, cs->look, cs->up, cs->fov, cresX, cresY);
            c.setProjection(cs->projection);   // rectilinear (default) or a fisheye/panoramic lens
            c.apertureR = cs->aperture;
            if (cs->filmDist_m > 0.0) { c.filmDist = cs->filmDist_m; c.lensF = cs->lensF_m; }  // physical-optics (lens/fstop): film at image distance, real focal
            else                      { c.setFocus(cs->focus); }                                // legacy unit-film camera
            char cmode = effMode(cs->mode);
            if (cs->lens) {
                // Physical multi-element lens: the realistic-camera ray-gen (genLensRay)
                // traces film->scene through the real glass. The analytic pinhole/thin-
                // lens forward modes (A/B/C) and the pinhole-splat composite (P) can't
                // form that image, so they render in mode R (backward realistic camera).
                // Mode D keeps the lens on its camera subpath (Plan B): the backward lens
                // ray still lights through the glass while forward light transport keeps
                // its caustic efficiency (the light-image splat, t=1, is disabled).
                // Mode P routes to that lens-aware BDPT, since P's forward pass splats to
                // a pinhole and can't be pushed through the lens.
                c.lens = cs->lens;
                double flmm = cs->lens->focalLengthMM();
                double fw = cs->lens->filmW_mm, fh = cs->lens->filmH_mm;
                if (cmode == 'D') {
                    std::printf("[camera] '%s' has a physical lens -> mode D (BDPT) with "
                                "the lens on the camera subpath (Plan B; light-image splat "
                                "disabled); f=%.1fmm, sensor %.1fx%.1fmm\n",
                                cs->name.c_str(), flmm, fw, fh);
                } else if (cmode == 'P') {
                    // The composite's forward pass splats to a pinhole and can't be
                    // pushed through the lens, so route to the lens-aware BDPT (mode D)
                    // when the scene is within BDPT scope; otherwise (fog/fluorescence/
                    // spot-env/layered) fall back to the backward realistic camera (R),
                    // which supports those — matching the pre-Plan-B behavior.
                    if (const char* why = bdptUnsupportedFeature(scene)) {
                        std::printf("[camera] '%s' has a physical lens -> mode P falls back "
                                    "to mode R (backward realistic camera): the composite "
                                    "can't route its pinhole splat through the lens, and the "
                                    "scene uses %s (outside lens-aware BDPT scope); "
                                    "f=%.1fmm, sensor %.1fx%.1fmm\n",
                                    cs->name.c_str(), why, flmm, fw, fh);
                        cmode = 'R';
                    } else {
                        std::printf("[camera] '%s' has a physical lens -> mode P routes to "
                                    "the lens-aware BDPT (mode D): the composite's pinhole-"
                                    "splat forward pass can't form the lens image; f=%.1fmm, "
                                    "sensor %.1fx%.1fmm\n", cs->name.c_str(), flmm, fw, fh);
                        cmode = 'D';
                    }
                } else if (cmode != 'R') {
                    std::printf("[camera] '%s' has a physical lens -> rendering in mode R "
                                "(backward realistic camera); f=%.1fmm, sensor %.1fx%.1fmm\n",
                                cs->name.c_str(), flmm, fw, fh);
                    cmode = 'R';
                }
            }
            // Exposure-lock group: a global -exposure-lock forces one shared anchor
            // (group 0) across every camera; otherwise a per-path `exposure_lock`
            // locks only that path's frames (group = its pathGroup); -1 = per-frame.
            int eg = forceExposureLock ? 0 : (cs->exposureLock ? cs->pathGroup : -1);
            double cexp = (exposureCli > 0.0) ? exposureCli : cs->exposureMul;   // -exposure/-ev overrides the authored comp
            // Absolute-EV aperture exposure (camera equation E = L·(π/4)/N²). The pinhole
            // splat (mode B) measures scene RADIANCE and, unlike the finite-lens catch
            // modes A/C (whose splat weight already carries the pupil area R²∝1/N²),
            // ignores the aperture entirely — so an absolute mode-B render is identically
            // bright at f/2 and f/8 when a real sensor separates them by 4 stops. Fold the
            // camera-equation aperture term into the exposure comp, but ONLY when a
            // physical aperture was actually authored (c.lensF>0 ⟺ an `fstop`/`lens` was
            // given). With no aperture authored the pinhole stays the pure radiance
            // reference (byte-identical to before — e.g. scenes/absolute.ftsl), so this
            // only ever darkens a mode-B camera that opted into an f-number. Modes A/C are
            // untouched here (they must NOT double-apply 1/N²; their gross-scale mis-seat
            // is the separate issue #1). No effect outside absolute EV (auto-exposure's
            // p99 anchor cancels any uniform aperture scale anyway).
            if (scene.absolute && cmode == 'B' && c.lensF > 0.0 && c.apertureR > 0.0) {
                double N = c.lensF / (2.0 * c.apertureR);           // f-number = focal / (2·apertureR)
                if (N > 0.0) {
                    double camEq = (PI / 4.0) / (N * N);            // (π/4)/N² image-side irradiance factor
                    cexp = (cexp > 0.0 ? cexp : 1.0) * camEq;
                }
            }
            // Unsupported-feature policy (-on-unsupported): if this camera's mode still
            // can't render the scene (e.g. GRIN media in mode D and no prefer{}/else{}
            // branch selected one), apply the policy — error (abort), fall back to mode R,
            // or strip the offending feature and render anyway.
            {
                bool proceed = true;
                cmode = applyUnsupportedPolicy(scene, cmode, cs->projection, cs->name.c_str(), proceed);
                if (!proceed) return 1;
            }
            toRender.push_back({cs->name, c, cmode, cresX, cresY, cexp, eg, cs->look, cs->up, cs->fov});
        }
    } else {
        // Built-in scene: one camera. Every image-forming mode (A/B/C/P/D/M/S/U/ref)
        // uses the same camera frame; only the old contact-sensor diagnostic did not.
        const bool useCamera = (mode == 'A' || mode == 'B' || mode == 'C' ||
                                mode == 'P' || mode == 'D' || mode == 'M' ||
                                mode == 'S' || mode == 'U' || refMode);
        int fresX = res, fresY = (resYCli > 0) ? resYCli : res;
        previewUpscale(fresX, fresY);   // big, readable raster preview (no-op for real renders)
        // Demo-camera eye/target/up/fov (captured for the interactive raster viewer).
        Vec3 cEye  = haveView ? viewEye  : (prism ? Vec3{0.5, 0.5, 2.4} : Vec3{0.5, 0.5, 2.7});
        Vec3 cLook = haveView ? viewLook : (prism ? Vec3{0.5, 0.45, 0.5} : Vec3{0.5, 0.5, 0.5});
        Vec3 cUp   = haveView ? viewUp   : Vec3{0, 1, 0};
        double cFov = haveView ? viewFov : (prism ? 45.0 : 40.0);
        Camera c;
        if (useCamera) {
            c.lookAt(cEye, cLook, cUp, cFov, fresX, fresY);
            c.apertureR = apertureR;
            c.setFocus(focusDist);   // thin lens for the finite-aperture modes A/C (0 = camera obscura)
        }
        toRender.push_back({"", c, mode, fresX, fresY, (exposureCli > 0.0 ? exposureCli : 0.0), forceExposureLock ? 0 : -1, cLook, cUp, cFov});
    }

    // -explore/-fly: seed the interactive raster viewer at the first selected frame
    // instead of rendering the whole flyby. We KEEP a copy of every selected frame's
    // camera as a "path" the viewer can lock onto (its timeline / lock-to-path panel),
    // then trim toRender to a single frame so the raster loop draws one frame before the
    // fly viewer takes over (window is held open).
    struct PathFrame { Vec3 eye; Vec3 fwd; Vec3 up; double fov; };
    std::vector<PathFrame> explorePath;    // one entry per flyby frame (empty for a lone camera)
    double explorePathFps = 0.0;           // authored playback rate hint (0 = none)
    std::string exploreCurveName;          // base name of the selected flyby (frame "swoop007" -> "swoop"),
                                           //   used to pick the matching authored camera_curve for round-trip edit
    if (exploreMode && toRender.size() > 1) {
        explorePath.reserve(toRender.size());
        for (const auto& rc : toRender) {
            Vec3 f = rc.lookAt - rc.cam.eye;
            double L = std::sqrt(dot(f, f));
            f = (L > 1e-9) ? f * (1.0 / L) : Vec3{0, 0, -1};
            explorePath.push_back({rc.cam.eye, f, rc.up, rc.fovY});
        }
        // Recover the flyby's base name by stripping the trailing zero-padded frame index
        // off the first frame's camera name (e.g. "swoop007" -> "swoop").
        { const std::string& fn = toRender.front().name;
          size_t end = fn.size();
          while (end > 0 && std::isdigit((unsigned char)fn[end - 1])) --end;
          if (end < fn.size()) exploreCurveName = fn.substr(0, end); }
        explorePathFps = ftslScene.defaultFps;   // scene-authored fps seeds the cam/sec box
        std::printf("[explore] starting interactive fly viewer at '%s' with a %zu-frame camera path"
                    " (timeline + lock-to-path enabled)\n",
                    toRender.front().name.empty() ? "<camera>" : toRender.front().name.c_str(),
                    explorePath.size());
        toRender.resize(1);
    }

    // Output naming: a single camera writes to `out`; several cameras write one file
    // each, inserting `_<name>` before the extension (so out=r.ppm -> r_hero.ppm).
    auto outFor = [&](const std::string& name) -> std::string {
        if (toRender.size() <= 1 || name.empty()) return out;
        std::string base = out;
        auto dot = base.find_last_of('.');
        std::string stem = (dot == std::string::npos) ? base : base.substr(0, dot);
        std::string ext  = (dot == std::string::npos) ? std::string(".ppm") : base.substr(dot);
        return stem + "_" + name + ext;
    };

    // --- Stereoscopic expansion (-stereo): each camera -> a Left/Right eye pair --------
    // Off-axis method: two PARALLEL rectilinear cameras offset ±baseline/2 along the M13
    // camera right axis u, each with a horizontally SHEARED frustum (Camera::frustumShiftX)
    // so the convergence plane sits at zero parallax — no toe-in, hence no vertical
    // parallax / eye strain. Baseline & convergence come from the physical viewing
    // geometry (interocular, screen width from -dpi or the -view-dist/FOV mapping, and the
    // convergence distance). Both eyes share one exposure group so they (and, for an
    // exposure-locked path, every frame) tone-map identically. After the render loop the
    // two eye PNGs are composited (side-by-side or Dubois anaglyph) into the -o path.
    struct StereoPair { std::string finalPath, leftPath, rightPath; };
    std::vector<StereoPair> stereoPairs;
    if (stereoMode != STEREO_OFF) {
        if (doRaster || exploreMode) {
            std::fprintf(stderr, "[stereo] ignored: -stereo needs a light-transport render "
                                 "(not -raster/-explore)\n");
        } else {
            // Final composite path for each ORIGINAL camera, resolved before we expand
            // toRender (outFor keys off toRender.size()).
            std::vector<std::string> finalPaths;
            finalPaths.reserve(toRender.size());
            for (const auto& rc : toRender) finalPaths.push_back(outFor(rc.name));

            std::vector<RenderCam> expanded;
            expanded.reserve(toRender.size() * 2);
            int syntheticGroup = 1000000;   // per-pair exposure groups for unlocked (per-frame) cameras
            std::vector<std::pair<size_t, size_t>> pairIdx;   // (leftIdx,rightIdx) in `expanded`; SIZE_MAX = mono
            bool announced = false;
            for (size_t k = 0; k < toRender.size(); ++k) {
                const RenderCam& rc = toRender[k];
                if (rc.cam.projection != CAM_RECTILINEAR) {
                    std::fprintf(stderr, "[stereo] '%s' is fisheye/panoramic; stereo needs a "
                                         "rectilinear lens — rendering it mono\n",
                                 rc.name.empty() ? "<camera>" : rc.name.c_str());
                    pairIdx.push_back({expanded.size(), SIZE_MAX});
                    expanded.push_back(rc);
                    continue;
                }
                Vec3 toTgt = rc.lookAt - rc.cam.eye;
                double C = (stereoConverge > 0.0) ? stereoConverge : std::sqrt(dot(toTgt, toTgt));
                if (!(C > 0.0)) C = 1.0;                                   // no target -> unit convergence
                // Screen width in metres: from a measured DPI, else assume the screen shows
                // the camera's horizontal field at the viewing distance (W = 2·d·tanHalfX).
                double W = (stereoDpi > 0.0) ? ((double)rc.res * 0.0254 / stereoDpi)
                                             : (2.0 * stereoViewDist * rc.cam.tanHalfX);
                double S = (W > 0.0) ? stereoEyeSep / W : 0.0;            // shear: infinity -> interocular on screen
                double b = 2.0 * C * rc.cam.tanHalfX * S;                  // baseline (scene units): b/C = eyeSep/screenW
                int grp = (rc.expGroup >= 0) ? rc.expGroup : syntheticGroup++;
                RenderCam L = rc, R = rc;
                L.cam.eye = rc.cam.eye - rc.cam.u * (b * 0.5); L.cam.frustumShiftX = +S; L.expGroup = grp;
                R.cam.eye = rc.cam.eye + rc.cam.u * (b * 0.5); R.cam.frustumShiftX = -S; R.expGroup = grp;
                std::string tag = rc.name.empty() ? std::string("stereo") : rc.name;
                L.name = tag + "__eyeL"; R.name = tag + "__eyeR";
                pairIdx.push_back({expanded.size(), expanded.size() + 1});
                expanded.push_back(L); expanded.push_back(R);
                if (!announced) {
                    std::printf("[stereo] eye-sep %.4g m, view-dist %.4g m, screen width %.4g m (%s); "
                                "'%s' -> baseline %.4g, convergence %.4g (scene units), shear %.4f\n",
                                stereoEyeSep, stereoViewDist, W,
                                stereoDpi > 0.0 ? "from -dpi" : "from -view-dist/FOV",
                                rc.name.empty() ? "<camera>" : rc.name.c_str(), b, C, S);
                    announced = true;
                }
            }
            toRender.swap(expanded);   // now size>1 so outFor yields the per-eye file paths
            for (size_t p = 0; p < pairIdx.size(); ++p) {
                if (pairIdx[p].second == SIZE_MAX) continue;   // mono camera: no composite
                stereoPairs.push_back({ finalPaths[p],
                                        outFor(toRender[pairIdx[p].first].name),
                                        outFor(toRender[pairIdx[p].second].name) });
            }
        }
    }

    // Shared auto-exposure anchors, one per exposure-lock group (see RenderCam.expGroup).
    // A group's anchor is normally computed by the first frame rendered and reused by the
    // rest (no dolly flicker); the meter pre-pass below can instead *pre-populate* it from
    // a chosen metering frame so every frame locks to that viewpoint's exposure. A null
    // anchor (group -1) = per-frame auto.
    std::map<int, double> expAnchors;

    // --- Exposure-lock metering plan (which frame each locked group meters from) --------
    // The `exposure_lock <selector>` on a camera_path/orbit/curve chooses the viewpoint the
    // whole group locks to (see CamSpec::EXPLOCK_*). Here we resolve that selector to the
    // concrete metering camera(s) for every locked group actually being rendered, so the
    // meter pre-pass (preview: raster; real: a reduced-sample render) can compute each
    // group's shared anchor up front. Skipped for absolute-EV scenes (fixed sensor gain,
    // no auto-exposure to lock) and when a global -exposure-lock is forcing one anchor.
    struct MeterCam { Camera cam; char mode; int res; int resY; std::string name; };
    std::map<int, std::vector<MeterCam>> meterPlan;   // group -> metering camera(s) (>1 = average)
    std::set<int> meterAdaptive;                      // groups whose plan is metered ADAPTIVELY

    // Low-discrepancy metering ORDER over a path's N frames. Instead of picking evenly
    // spaced frames (which can ALIAS with a periodic exposure swing along the path — an
    // orbit that passes a light once per revolution, say — biasing the averaged anchor),
    // we walk the frames in van der Corput (base-2 radical-inverse / bit-reversal) order:
    // 0, N/2, N/4, 3N/4, N/8, …  Every prefix of this sequence is uniformly spread over
    // the whole path AND non-periodic, so an adaptive meter that stops after k frames has
    // still sampled the entire flyby evenly with no periodic bias — and it is deterministic
    // (reproducible), which a purely random jitter would not be. Rounding collisions are
    // resolved by probing to the nearest unused index, so the result is a permutation.
    auto meterOrder = [](int n) {
        std::vector<int> order; order.reserve(std::max(0, n));
        if (n <= 0) return order;
        std::vector<char> used(n, 0);
        for (int r = 0; (int)order.size() < n; ++r) {
            unsigned b = (unsigned)r; double f = 0.0, base = 0.5;   // radical inverse base 2 of r
            while (b) { f += (b & 1u) * base; b >>= 1; base *= 0.5; }
            int i = (int)(f * n); if (i >= n) i = n - 1;
            while (used[i]) i = (i + 1) % n;                        // nearest unused (dedupe rounding)
            used[i] = 1; order.push_back(i);
        }
        return order;
    };

    // Adaptive stop for an AVERAGED exposure meter. We don't know a path's exposure
    // variance up front, so rather than metering a fixed frame count we meter in the
    // low-discrepancy order above and watch the running mean converge. The anchor is a
    // brightness, so convergence is judged in STOPS (log2): at successive power-of-two
    // checkpoints (8,16,32,…) we compare the running mean-of-log2 to the previous
    // checkpoint and stop once it moves less than `tolStops`. Bounded to [kMin, kMax]
    // valid samples, so a smooth path settles in ~16 frames while a wildly varying one
    // keeps going (up to kMax) for a faithful average — the count adapts to the DATA
    // instead of a guessed constant. The returned anchor is the ARITHMETIC mean of the
    // metered per-frame anchors (unchanged from the non-adaptive path, so short paths
    // that meter every frame are bit-identical to before).
    struct MeterConverge {
        int    kMin, kMax; double tolStops;
        int    k = 0; double sumLin = 0.0, sumLog2 = 0.0;
        int    nextCheck; double lastMean = 0.0; bool haveLast = false;
        MeterConverge(int kmn, int kmx, double tol)
            : kMin(kmn), kMax(kmx), tolStops(tol), nextCheck(std::max(kmn, 8)) {}
        // Feed one per-frame anchor (>0 to count). Returns true once enough frames are in.
        bool add(double a) {
            if (a > 0.0) { sumLin += a; sumLog2 += std::log2(a); ++k; }
            if (k >= kMax) return true;
            if (k >= kMin && k >= nextCheck) {
                double mean = sumLog2 / k;
                bool conv = haveLast && std::fabs(mean - lastMean) <= tolStops;
                lastMean = mean; haveLast = true; nextCheck *= 2;
                if (conv) return true;
            }
            return false;
        }
        double anchor() const { return k > 0 ? sumLin / k : 0.0; }
        int    used()   const { return k; }
    };
    // Adaptive-meter bounds: never fewer than kMeterMin nor more than kMeterMax frames.
    constexpr int    kMeterMin = 8, kMeterMax = 64;
    constexpr double kMeterTolStops = 0.02;   // stop when the mean moves < 0.02 stop

    if (noMeter) {
        bool anyLocked = false;
        for (const auto& rc : toRender) if (rc.expGroup >= 0) { anyLocked = true; break; }
        if (anyLocked)
            std::printf("[meter] skipped (-no-meter%s): frames auto-expose per frame "
                        "instead of metering the exposure-lock group.\n",
                        exploreMode ? " via -explore" : "");
    }
    if (!noMeter && fromFtsl && !ftslScene.cameras.empty() && !scene.absolute && !forceExposureLock) {
        // Build a camera the same way the render loop does, minus the verbose lens logging.
        auto buildMeterCam = [&](const ftsl::CamSpec& cs) -> MeterCam {
            MeterCam m;
            m.res  = resFromCli ? res : (cs.res  > 0 ? cs.res  : res);
            m.resY = resFromCli ? (resYCli > 0 ? resYCli : res) : (cs.resY > 0 ? cs.resY : m.res);
            m.cam.lookAt(cs.eye, cs.look, cs.up, cs.fov, m.res, m.resY);
            m.cam.setProjection(cs.projection);
            m.cam.apertureR = cs.aperture;
            if (cs.filmDist_m > 0.0) { m.cam.filmDist = cs.filmDist_m; m.cam.lensF = cs.lensF_m; }
            else                     { m.cam.setFocus(cs.focus); }
            m.mode = effMode(cs.mode);
            if (cs.lens) { m.cam.lens = cs.lens; if (m.mode != 'D' && m.mode != 'P') m.mode = 'R'; }
            m.name = cs.name;
            return m;
        };
        // Which locked groups are actually in the render set?
        std::map<int, int> activeGroups;   // group -> count (presence)
        for (const auto& rc : toRender) if (rc.expGroup >= 0) ++activeGroups[rc.expGroup];
        for (const auto& [g, cnt] : activeGroups) {
            (void)cnt;
            // Gather this path's frames (pathGroup == g) in file order, and its selector.
            std::vector<const ftsl::CamSpec*> members;
            for (const auto& cs : ftslScene.cameras)
                if (cs.exposureLock && cs.pathGroup == g) members.push_back(&cs);
            if (members.empty()) continue;               // e.g. a forced/standalone group
            const ftsl::CamSpec& rep = *members.front(); // all frames share the selector
            std::vector<MeterCam>& plan = meterPlan[g];
            auto addFrame = [&](const ftsl::CamSpec& cs) { plan.push_back(buildMeterCam(cs)); };
            switch (rep.expLockSel) {
                case ftsl::CamSpec::EXPLOCK_AVERAGE: {
                    // Average the per-frame anchors — but metering every frame of a long
                    // flyby is wasteful. Each meter frame projects the WHOLE scene (the
                    // dominant, resolution-independent cost), and the locked anchor is a
                    // smooth statistic of the path. Rather than a fixed subsample count, we
                    // queue ALL frames in low-discrepancy (van der Corput) order and let the
                    // meter loop stop ADAPTIVELY once the running average converges (see
                    // MeterConverge). That kills the periodic-aliasing bias of even spacing
                    // and spends frames in proportion to how variable the path actually is:
                    // a smooth dolly settles in ~16 frames, a wild orbit meters up to kMax.
                    const int n = (int)members.size();
                    for (int idx : meterOrder(n)) addFrame(*members[(size_t)idx]);
                    meterAdaptive.insert(g);
                    break;
                }
                case ftsl::CamSpec::EXPLOCK_INDEX: {
                    int n = (int)members.size(), i = rep.expLockIndex;
                    if (i < 0) i += n;
                    if (i < 0 || i >= n) {
                        std::fprintf(stderr, "[exposure] lock index %d out of range for '%s' "
                                     "(%d frames); metering the first frame instead\n",
                                     rep.expLockIndex, rep.name.c_str(), n);
                        i = 0;
                    }
                    addFrame(*members[i]);
                    break;
                }
                case ftsl::CamSpec::EXPLOCK_NEAR: {
                    const ftsl::CamSpec* best = members.front(); double bd2 = 1e300;
                    for (const auto* cs : members) {
                        Vec3 d = cs->eye - rep.expLockPoint; double d2 = dot(d, d);
                        if (d2 < bd2) { bd2 = d2; best = cs; }
                    }
                    addFrame(*best);
                    break;
                }
                case ftsl::CamSpec::EXPLOCK_CAMERA: {
                    const ftsl::CamSpec* named = nullptr;
                    for (const auto& cs : ftslScene.cameras)
                        if (cs.name == rep.expLockCam) { named = &cs; break; }
                    if (!named) {
                        std::fprintf(stderr, "[exposure] lock camera '%s' not found for '%s'; "
                                     "metering the first frame instead\n",
                                     rep.expLockCam.c_str(), rep.name.c_str());
                        addFrame(rep);
                    } else addFrame(*named);
                    break;
                }
                case ftsl::CamSpec::EXPLOCK_FIRST:
                default:
                    addFrame(rep);
                    break;
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Fast solid-shaded PREVIEW (-raster). Bypass ALL light transport: tessellate the
    // whole scene once (spheres, isosurfaces, instanced meshes) and z-buffer each
    // selected camera with plain diffuse+headlight shading. No transparency, mirrors,
    // caustics or GI — just the composition and, for a camera_curve, the flyby motion,
    // in a fraction of a second per frame. Honours the same camera list / -camera
    // selection / -window live view as the real renderer.
    // -----------------------------------------------------------------------------
    if (doRaster) {
        // -raster-gpu (G2): render implicit isosurfaces by casting a deterministic primary
        // ray per pixel on the GPU (renderIsoPreviewCuda) instead of marching-cubes
        // tessellation. Requires CUDA + a POD-bakeable scene; see-through mode and a
        // physical-lens camera aren't covered, so those fall back to CPU tessellation.
#ifdef HAVE_CUDA
        bool useGpuIso = rasterGpu && !rasterSeeThrough && cudaAvailable() && cudaForwardSupported(scene);
        if (rasterGpu && !useGpuIso) {
            if (rasterSeeThrough)      std::fprintf(stderr, "[raster] -raster-gpu doesn't support see-through; using CPU tessellation\n");
            else if (!cudaAvailable()) std::fprintf(stderr, "[raster] -raster-gpu: no CUDA device; using CPU tessellation\n");
            else                       std::fprintf(stderr, "[raster] -raster-gpu: scene not GPU-bakeable; using CPU tessellation\n");
        }
#else
        const bool useGpuIso = false;
        if (rasterGpu) std::fprintf(stderr, "[raster] -raster-gpu needs a CUDA build; using CPU tessellation\n");
#endif
        if (useGpuIso)
            std::printf("[raster] GPU iso preview: primary-ray isosurface render on the GPU (no tessellation)\n");
        else
            std::printf("[raster] solid-shaded preview: tessellating scene (iso res %d) ...\n", rasterIso);
        if (rasterSeeThrough)
            std::printf("[raster] see-through: clear objects dim/haze what's behind them (clarity %.2f, no refraction)\n", rasterClarity);
        std::fflush(stdout);

        // Pop the live window up IMMEDIATELY (before the potentially-slow tessellation)
        // so heavy scenes don't sit with a blank screen while the isosurfaces march.
        // Size it to the first camera we'll render; fill a dark placeholder frame and
        // show a "tessellating…" title, then update N/M progress as each implicit is
        // marched (see the tessellate() callback below).
        if (g_showWindow && !toRender.empty() && !g_liveWin) {
            int pw = toRender.front().res, ph = toRender.front().resY;
            std::vector<uint8_t> placeholder((size_t)pw * ph * 3);
            for (size_t i = 0; i < placeholder.size(); i += 3) {
                placeholder[i] = 24; placeholder[i + 1] = 26; placeholder[i + 2] = 30;
            }
            g_liveWin = std::make_unique<LiveWindow>(pw, ph, g_windowTitle.c_str());
            g_liveWin->update(pw, ph, placeholder);
            if (useGpuIso) {
                g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  GPU iso preview\xE2\x80\xA6");
            } else {
                const size_t nImp = scene.implicits.size();
                g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  tessellating" +
                                    (nImp ? " (0/" + std::to_string(nImp) + ")" : "\xE2\x80\xA6"));
            }
        }

        raster::PreviewLight plight = raster::deriveLight(scene);
        std::vector<raster::PTri> prims;   // tessellated lazily (empty in pure GPU-iso mode)
        bool tessellated = false;
        // Tessellate on demand: the CPU / GPU-triangle path calls this immediately; the GPU
        // iso path skips it entirely and only tessellates if a frame must fall back (e.g. a
        // physical-lens camera the primary-ray kernel can't handle).
        auto ensurePrims = [&]() {
            if (tessellated) return;
            tessellated = true;
            auto rt0 = std::chrono::steady_clock::now();
            // Progress callback: update the window title (and a periodic stdout line) as the
            // heavy isosurface/CSG implicits are marched one by one.
            auto lastTick = std::chrono::steady_clock::now();
            auto tessProgress = [&](int done, int total) {
                if (total <= 0) return;
                int pct = (int)std::lround(100.0 * done / total);
                if (g_liveWin && !g_liveWin->closed()) {
                    g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  tessellating (" +
                                        std::to_string(done) + "/" + std::to_string(total) +
                                        ", " + std::to_string(pct) + "%)");
                }
                auto now = std::chrono::steady_clock::now();
                if (done == 0 || done == total ||
                    std::chrono::duration<double>(now - lastTick).count() >= 1.0) {
                    std::printf("[raster] tessellating implicit %d/%d (%d%%)\n", done, total, pct);
                    std::fflush(stdout);
                    lastTick = now;
                }
            };
            prims = raster::tessellate(scene, rasterIso, tessProgress);
            auto rt1 = std::chrono::steady_clock::now();
            std::printf("[raster] %zu triangles in %.2fs; rendering %zu camera(s) on %d threads%s\n",
                        prims.size(), std::chrono::duration<double>(rt1 - rt0).count(),
                        toRender.size(), nThreads, g_showWindow ? " — live window" : "");
            std::fflush(stdout);
        };
        if (!useGpuIso) {
            ensurePrims();
        } else {
            std::printf("[raster] rendering %zu camera(s) on the GPU (primary-ray iso)%s\n",
                        toRender.size(), g_showWindow ? " — live window" : "");
            std::fflush(stdout);
        }

        // GPU preview rasterizer (-device gpu|auto). Bake the world triangles + image skins
        // to the device ONCE (reused for every camera / flyby frame), then each frame runs
        // ENTIRELY on the GPU — projection + raster + shade (+ clear-accumulation pass when
        // see-through) + a device twin of the host exposure/tonemap tail (exact p99 anchor
        // via float-bit histograms, double-precision tonemap, shared sRGB LUT) — verified
        // byte-identical to the CPU path's frames. The GPU covers all camera projections
        // (rectilinear + fisheye/panoramic), opaque + textured (skinned) previews, and
        // see-through (clear-glass) compositing. Any device failure falls back per-frame.
#ifdef HAVE_CUDA
        raster_cuda::Scene* gpuRaster = nullptr;
        {
            const bool wantGpu  = !std::strcmp(device, "gpu");
            const bool wantAuto = !std::strcmp(device, "auto");
            if (!useGpuIso && (wantGpu || wantAuto) && raster_cuda::available()) {
                gpuRaster = raster_cuda::upload(prims, plight, &scene.textures);
                if (gpuRaster)
                    std::printf("[raster] GPU rasterizer: frames on the GPU "
                                "(all projections; skins + see-through supported)\n");
                else if (wantGpu)
                    std::fprintf(stderr, "[raster] GPU upload failed; using CPU\n");
            } else if (wantGpu && !raster_cuda::available()) {
                std::fprintf(stderr, "[raster] no CUDA device found; using CPU\n");
            }
            std::fflush(stdout);
        }
#endif
        // Render one preview frame: GPU when it's baked (any projection / skins / see-through),
        // else the CPU rasterizer. A GPU device failure returns empty -> CPU fallback too.
        auto rasterOne = [&](const Camera& cam, int W, int H, double ev, bool autoExp,
                             double* lock) -> std::vector<uint8_t> {
#ifdef HAVE_CUDA
            // G2: cast primary rays straight at the implicit on the GPU (no tessellation).
            // A physical-lens camera isn't covered by the pinhole/fisheye ray-gen, so it
            // falls through to the tessellated path (built lazily on first need).
            if (useGpuIso && !cam.hasLens()) {
                std::vector<uint8_t> img =
                    renderIsoPreviewCuda(scene, cam, W, H, nThreads, ev, autoExp, lock);
                if (!img.empty()) return img;
            }
            if (gpuRaster) {
                std::vector<uint8_t> img =
                    raster_cuda::renderFrame(gpuRaster, cam, W, H, nThreads, ev, autoExp, lock,
                                             rasterSeeThrough, rasterClarity);
                if (!img.empty()) return img;
            }
#endif
            ensurePrims();   // lazy fallback (also the sole path when the GPU is unavailable)
            return raster::renderFrame(prims, cam, W, H, plight, nThreads, ev, autoExp, lock,
                                       rasterSeeThrough, rasterClarity, &scene.textures);
        };

        // Exposure-lock meter pre-pass: for each locked group, raster its selected metering
        // frame(s) and pre-populate expAnchors[group] (averaging for EXPLOCK_AVERAGE), so
        // every frame of the group previews at the chosen viewpoint's exposure — mirroring
        // what the meter pre-pass does for the real render. Uses the same raster shading, so
        // the anchor matches the frames' own pipeline exactly.
        //
        // This pass can dwarf the tessellation for an averaged lock (e.g. a 145-frame
        // camera_path meters ~144 frames), so it drives its own progress: a throttled
        // stdout percentage + a window title, and it pushes each freshly-metered frame to
        // the live window so the preview animates through the metering instead of sitting
        // blank on the last tessellation frame.
        size_t meterDone = 0;
        auto   meterTick = std::chrono::steady_clock::now();
        for (const auto& [g, cams] : meterPlan) {
            if (cams.empty()) continue;
            const bool adaptive = meterAdaptive.count(g) != 0;
            const int  N   = (int)cams.size();
            const int  kmx = adaptive ? std::min(N, kMeterMax) : N;
            const int  kmn = adaptive ? std::min(N, kMeterMin) : N;
            MeterConverge conv(kmn, kmx, kMeterTolStops);
            for (const auto& mc : cams) {
                if (g_liveWin && g_liveWin->closed()) { g_stopRequested = 1; break; }
                double a = 0.0;
                std::vector<uint8_t> mimg =
                    rasterOne(mc.cam, mc.res, mc.resY, /*exposure*/1.0, /*autoExpose*/true, &a);
                bool stop = conv.add(a);
                ++meterDone;
                // Show the metering pass converging + a throttled running count so the
                // window/console isn't silent while this (often long) pre-pass runs. The
                // shown frames are per-frame auto-exposed — a rough, NOT-yet-exposure-locked
                // preview whose brightness varies frame to frame — so the title says as much
                // to avoid the impression that this flickering sweep is the final look.
                auto now = std::chrono::steady_clock::now();
                if (meterDone == 1 || stop ||
                    std::chrono::duration<double>(now - meterTick).count() >= 1.0) {
                    if (g_liveWin && !g_liveWin->closed()) {
                        g_liveWin->update(mc.res, mc.resY, mimg);
                        g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  metering exposure "
                                            "(preview NOT locked yet) " +
                                            std::to_string(meterDone));
                    }
                    std::printf("[raster] metering exposure %zu\n", meterDone);
                    std::fflush(stdout);
                    meterTick = now;
                }
                if (stop) break;
            }
            if (g_stopRequested) break;
            if (conv.used() > 0) {
                expAnchors[g] = conv.anchor();
                if (adaptive)
                    std::printf("[raster] exposure lock: group %d meters the average of %d/%d "
                                "frames (anchor %.4g)\n", g, conv.used(), N, expAnchors[g]);
                else
                    std::printf("[raster] exposure lock: group %d meters '%s' (anchor %.4g)\n",
                                g, cams.front().name.c_str(), expAnchors[g]);
            }
        }
        std::fflush(stdout);

        // -raster-bench N: steady-state per-frame cost — the interactive explorer's
        // metric (it re-renders one frame per camera move) — measured independently of
        // process launch, scene build/tessellation and the GPU upload, which have all
        // already happened by this point. Renders the FIRST selected camera N times
        // through the same rasterOne path the explorer uses (per-frame auto-exposure,
        // no lock anchor), reports min/median/mean ms per frame and, when the GPU
        // rasterizer ran, a per-pass breakdown. Writes the last frame to -o so
        // backends/builds can be byte-compared.
        if (rasterBench > 0 && !toRender.empty()) {
            const RenderCam& rc = toRender.front();
            const int W = rc.res, H = rc.resY;
            double ev = (rc.exposure > 0.0) ? rc.exposure : 1.0;
            if (scene.absolute && (rc.mode == 'A' || rc.mode == 'C')) {
                const double Rref = 0.02; double R = rc.cam.apertureR;
                if (R > 0.0) ev *= (R * R) / (Rref * Rref);
            }
            const bool autoExp = !scene.absolute;
            // Warmup frame (not timed): first-frame scratch allocs, GPU clock ramp.
            std::vector<uint8_t> img = rasterOne(rc.cam, W, H, ev, autoExp, nullptr);
#ifdef HAVE_CUDA
            raster_cuda::profEnable(true);
            (void)raster_cuda::profTake();
#endif
            std::vector<double> ms;
            std::vector<double> tail;                 // host-side present cost, timed separately
            ms.reserve(rasterBench);
            tail.reserve(rasterBench);
            for (int it = 0; it < rasterBench && !g_stopRequested; ++it) {
                auto t0 = std::chrono::steady_clock::now();
                img = rasterOne(rc.cam, W, H, ev, autoExp, nullptr);
                ms.push_back(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0).count());
                if (g_showWindow) {
                    if (!g_liveWin) g_liveWin = std::make_unique<LiveWindow>(W, H, g_windowTitle.c_str());
                    // The present tail used to dwarf the render itself (a per-pixel RGB->BGRA
                    // repack plus a HALFTONE StretchDIBits, both charged to the render thread),
                    // so report it: a faster backend is only a real speedup if this stays small.
                    auto t1 = std::chrono::steady_clock::now();
                    g_liveWin->update(W, H, img);
                    tail.push_back(std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - t1).count());
                    if (g_liveWin->closed()) g_stopRequested = 1;
                }
            }
#ifdef HAVE_CUDA
            raster_cuda::Prof profHost = raster_cuda::profTake();   // phase 1's per-pass tally
            // ---- Phase 2: the same frame delivered ZERO-COPY -----------------------------
            // Identical render work, but the tonemap writes the live window's D3D11 texture
            // in place, so there is no device->host download, no copy out of the pinned
            // buffer, and no re-upload. Timed end to end (render + show) so it can be read
            // straight against phase 1's render time + present tail, which is the same job
            // routed through host memory.
            std::vector<double> zc;
            bool zcTried = false;
            if (gpuRaster && g_showWindow && g_liveWin) {
                zcTried = true;
                zc.reserve(rasterBench);
                for (int it = 0; it < rasterBench && !g_stopRequested; ++it) {
                    auto t0 = std::chrono::steady_clock::now();
                    bool ok = g_liveWin->renderShared(W, H, [&](void* dev, void* tex) -> bool {
                        if (!raster_cuda::bindPresentTarget(gpuRaster, dev, tex, W, H)) return false;
                        return raster_cuda::renderFrameToTarget(gpuRaster, rc.cam, W, H, nThreads,
                                                                ev, autoExp, nullptr,
                                                                rasterSeeThrough, rasterClarity);
                    });
                    if (!ok) { zc.clear(); break; }   // no interop here: report it, don't fake it
                    zc.push_back(std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - t0).count());
                    if (g_liveWin->closed()) g_stopRequested = 1;
                }
            }
            raster_cuda::Prof profZc = raster_cuda::profTake();     // phase 2's per-pass tally
            raster_cuda::profEnable(false);
#endif
            if (!ms.empty()) {
                std::vector<double> s = ms;
                std::sort(s.begin(), s.end());
                double mean = 0;
                for (double v : ms) mean += v;
                mean /= ms.size();
                double mn = s.front(), md = s[s.size() / 2];
                std::printf("[raster-bench] %zu frames %dx%d: min %.2f ms  median %.2f ms  "
                            "mean %.2f ms  (%.1f fps @ median)\n",
                            ms.size(), W, H, mn, md, mean, md > 0 ? 1000.0 / md : 0.0);
            }
            if (!tail.empty()) {
                std::vector<double> s = tail;
                std::sort(s.begin(), s.end());
                double mean = 0;
                for (double v : tail) mean += v;
                mean /= tail.size();
                std::printf("[raster-bench] live-window present tail: min %.2f ms  median %.2f ms  "
                            "mean %.2f ms\n", s.front(), s[s.size() / 2], mean);
            }
#ifdef HAVE_CUDA
            {
                auto passLine = [](const char* what, const raster_cuda::Prof& p) {
                    if (p.frames <= 0) return;
                    double f = 1.0 / p.frames;
                    std::printf("[raster-bench] GPU per-pass avg ms (%s): clearvis %.2f  project %.2f  "
                                "raster %.2f  shade %.2f  clear %.2f  expose+encode %.2f  "
                                "download %.2f\n",
                                what, p.clearvis_ms * f, p.project_ms * f, p.raster_ms * f,
                                p.shade_ms * f, p.clear_ms * f, p.expose_ms * f,
                                p.download_ms * f);
                };
                passLine("download", profHost);
                if (!zc.empty()) {
                    std::vector<double> s = zc;
                    std::sort(s.begin(), s.end());
                    double mean = 0;
                    for (double v : zc) mean += v;
                    mean /= zc.size();
                    std::printf("[raster-bench] zero-copy render+present: min %.2f ms  median %.2f ms  "
                                "mean %.2f ms  (%.1f fps @ median)\n",
                                s.front(), s[s.size() / 2], mean,
                                s[s.size() / 2] > 0 ? 1000.0 / s[s.size() / 2] : 0.0);
                    passLine("zero-copy", profZc);
                } else if (zcTried) {
                    std::printf("[raster-bench] zero-copy present unavailable "
                                "(GDI window, or D3D on a different adapter than the CUDA device)\n");
                }
            }
#endif
            std::string path = outFor(rc.name);
            if (!writeImage(path, W, H, img))
                std::fprintf(stderr, "[raster-bench] failed to write %s\n", path.c_str());
            else
                std::printf("[raster-bench] wrote %s (%dx%d)\n", path.c_str(), W, H);
            std::fflush(stdout);
#ifdef HAVE_CUDA
            raster_cuda::destroy(gpuRaster);
#endif
            return 0;
        }

        int frame = 0;
        auto ft0 = std::chrono::steady_clock::now();
        for (const auto& rc : toRender) {
            if (g_stopRequested) break;
            int W = rc.res, H = rc.resY;
            // Effective preview brightness = photographic exposure comp * aperture term.
            double ev = (rc.exposure > 0.0) ? rc.exposure : 1.0;   // iso*shutter*exposure (<=0 = neutral)
            // Aperture only changes OUTPUT brightness in absolute-EV scenes shot through
            // a physical finite-lens catch mode (A/C): there the mode-A splat deposits
            // energy ∝ pupil area R² (render.h connectLens: contrib *= R*R) and the fixed
            // absolute sensor gain does NOT renormalise it, so a wider aperture is
            // genuinely brighter (∝ 1/N²). In the default auto-exposed pipeline the
            // 99th-percentile anchor divides that uniform R² scale straight back out
            // (aperture then affects only depth of field + noise), and mode B is a pure
            // pinhole (the authored aperture is virtual) — so in both those cases the
            // real render's brightness is aperture-independent and we leave it neutral.
            if (scene.absolute && (rc.mode == 'A' || rc.mode == 'C')) {
                const double Rref = 0.02;               // engine default aperture radius = neutral
                double R = rc.cam.apertureR;
                if (R > 0.0) ev *= (R * R) / (Rref * Rref);   // brightness ∝ pupil area
            }
            // Emulate the real renderer's tone map: non-absolute scenes get the p99
            // auto-exposure (aperture cancels out; iso/shutter/exposure stay as stops),
            // absolute EV bypasses it so power/aperture brightness survives. Honour the
            // same exposure-lock groups as filmToRgb8 so a camera_curve preview doesn't
            // flicker frame-to-frame (shared anchor per group; per-frame when expGroup<0).
            const bool autoExp = !scene.absolute;
            double* lockAnchor = (autoExp && rc.expGroup >= 0) ? &expAnchors[rc.expGroup] : nullptr;
            std::vector<uint8_t> img = rasterOne(rc.cam, W, H, ev, autoExp, lockAnchor);
            std::string path = outFor(rc.name);
            if (!writeImage(path, W, H, img)) {
                std::fprintf(stderr, "[raster] failed to write %s\n", path.c_str());
                return 1;
            }
            if (g_showWindow) {
                if (!g_liveWin) g_liveWin = std::make_unique<LiveWindow>(W, H, g_windowTitle.c_str());
                g_liveWin->update(W, H, img);
                std::string title = g_windowTitle + "  \xE2\x80\x94  raster " +
                                    (rc.name.empty() ? std::string("preview") : rc.name) + " (" +
                                    std::to_string(frame + 1) + "/" + std::to_string(toRender.size()) + ")";
                g_liveWin->setTitle(title);
                if (g_liveWin->closed()) g_stopRequested = 1;
            }
            if (toRender.size() > 1) {
                if (frame % 15 == 0 || frame + 1 == (int)toRender.size())
                    std::printf("[raster] frame %d/%zu -> %s\n", frame + 1, toRender.size(), path.c_str());
            } else {
                std::printf("[raster] wrote %s (%dx%d)\n", path.c_str(), W, H);
            }
            std::fflush(stdout);
            ++frame;
        }
        auto ft1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(ft1 - ft0).count();
        std::printf("[raster] done: %d frame(s) in %.2fs (%.1f fps).\n",
                    frame, secs, frame > 0 ? frame / std::max(secs, 1e-6) : 0.0);

        // ---------------------------------------------------------------------------
        // Interactive raster viewer. For a single still camera shown in a live window,
        // fly the camera with the keyboard and read off the eye/look_at to author a
        // .ftsl camera. Six controls, all along WORLD axes: the camera EYE (x,y,z) and
        // a LOOK-AT TARGET (x,y,z) which the camera always points at and which is drawn
        // as a red crosshair.
        //
        // A multi-camera flyby animates all its frames first (the loop above) and is NOT
        // interactive during the animation. But once it finishes, if the window is being
        // held open (-keepwindow), we hand control to the user too — seeded from the LAST
        // frame's camera (the one still on screen) — so the flyby doesn't just freeze on
        // its final frame with no way to look around. Without -keepwindow a flyby is a
        // batch sequence render, so we leave it non-interactive and let the process exit.
        if (g_showWindow && g_liveWin && !g_stopRequested &&
            (toRender.size() == 1 || g_keepWindow)) {
            const RenderCam& rc0 = toRender.back();   // == the only / last-shown camera
            const int    W = rc0.res, H = rc0.resY, proj = rc0.cam.projection;
            const Vec3   eye0 = rc0.cam.eye, tgt0 = rc0.lookAt, up = rc0.up;
            const double fovY = rc0.fovY;
            double ev = (rc0.exposure > 0.0) ? rc0.exposure : 1.0;
            if (scene.absolute && (rc0.mode == 'A' || rc0.mode == 'C')) {
                const double Rref = 0.02; double R = rc0.cam.apertureR;
                if (R > 0.0) ev *= (R * R) / (Rref * Rref);
            }
            const bool autoExp = !scene.absolute;   // per-frame auto-exposure while navigating
            // Unified fly-camera state: an eye position and a normalized look direction
            // `fwd` (no separate orientation target — you always travel where you look).
            // The world up is fixed (no roll), so mouse-look is a yaw about worldUp plus a
            // clamped pitch about the camera's right axis. `lookDist` is only used to place
            // the look_at when printing a camera block (the eye+fwd ray is what matters).
            const Vec3 worldUp = up;
            Vec3   eye = eye0;
            Vec3   fwd = tgt0 - eye0;
            { double L = std::sqrt(dot(fwd, fwd)); fwd = (L > 1e-9) ? fwd * (1.0 / L) : Vec3{0, 0, -1}; }
            double lookDist = std::sqrt(dot(tgt0 - eye0, tgt0 - eye0));
            if (lookDist < 1e-4) lookDist = (scene.sceneRadius > 0.0 ? scene.sceneRadius : 1.0);
            const double sceneR = (scene.sceneRadius > 0.0 ? scene.sceneRadius : 1.0);
            // Motion is FEEDBACK-LOCKED, not wall-clock-based: each held-key frame (and each
            // wheel notch) advances the eye by this fixed `step` in world units, and exactly
            // one frame is rendered per move. So travel rate auto-scales with render speed
            // (heavy scene -> careful crawl, light scene -> quick) and you can never skip past
            // geometry between two frames you didn't see. `step` is the per-move distance,
            // adjustable live with Ctrl+wheel.
            double       step   = sceneR * 0.02;     // held-key per-frame travel, world units
            // The plain wheel is a quick DOLLY, so a notch moves several fly-steps (a held
            // key is the fine cruise; the wheel repositions in a few flicks). Still tied to
            // `step` so Ctrl+wheel scales both together, and still collision-feedback-locked
            // (resolveMove stops at surfaces), so a coarse notch can't punch through geometry.
            const double kWheelDolly = 8.0;           // fly-steps travelled per plain-wheel notch
            // Hover-look turn RATES: the cursor's dead-zoned offset from the window centre
            // (nav.lookX/lookY, -1..+1) is multiplied by these AND the wall-clock frame time
            // to turn the view. Full deflection = kYaw/kPitch radians per SECOND, integrated
            // by dt, so the turn speed is FRAME-RATE INDEPENDENT: a light scene that raster-
            // previews at hundreds of fps turns at the same comfortable rate as a heavy one,
            // instead of spinning the view off-screen. (Translation stays feedback-locked
            // per-frame below — that's the collision-safety part; rotating in place can never
            // fling the eye through geometry, so it has no reason to be frame-locked.)
            const double kYaw   = 2.6;               // max yaw   rad/sec (~150 deg/s) at full pointer deflection
            const double kPitch = 2.0;               // max pitch rad/sec (~115 deg/s) at full pointer deflection
            // Rodrigues rotation of v about a UNIT axis by `ang` radians.
            auto rotAxis = [](const Vec3& v, const Vec3& axis, double ang) -> Vec3 {
                double c = std::cos(ang), s = std::sin(ang);
                return v * c + cross(axis, v) * s + axis * (dot(axis, v) * (1.0 - c));
            };
            auto norml = [](const Vec3& v) -> Vec3 {
                double L = std::sqrt(dot(v, v)); return (L > 1e-9) ? v * (1.0 / L) : v;
            };
            // Collision: keep the eye out of solid geometry so you can't fly through a wall.
            //   SLIDE  — stop at the wall but let the remaining motion slide along it, so
            //            holding forward against a wall carries you around a corner into open
            //            space (the default; also the least "stuck"-feeling).
            //   STOP   — halt dead at the wall (no sideways drift).
            //   OFF    — no collision (ghost through anything; for placing a camera outside
            //            the room or inside glass). `-noclip` starts here.
            enum CollideMode { COLLIDE_SLIDE, COLLIDE_STOP, COLLIDE_OFF };
            CollideMode collide = viewerNoclip ? COLLIDE_OFF : COLLIDE_SLIDE;
            auto collideName = [](CollideMode m) {
                return m == COLLIDE_SLIDE ? "slide" : (m == COLLIDE_STOP ? "stop" : "off (noclip)");
            };
            // Compact label for the panel's Clip button (fits the narrow button width).
            auto collideShort = [](CollideMode m) {
                return m == COLLIDE_SLIDE ? "slide" : (m == COLLIDE_STOP ? "stop" : "noclip");
            };
            // Resolve a proposed eye move against the scene. Casts along the motion with the
            // engine's own BVH (scene.closestHit); keeps a `skin` standoff so the near plane
            // never pokes through a surface. SLIDE iterates a few times so a corner (two walls)
            // doesn't leak. Returns the collision-safe new position.
            const double kSkin = sceneR * 0.02;       // standoff kept between eye and any wall
            auto resolveMove = [&](Vec3 pos, Vec3 delta) -> Vec3 {
                if (collide == COLLIDE_OFF) return pos + delta;
                for (int iter = 0; iter < 4; ++iter) {
                    double len = std::sqrt(dot(delta, delta));
                    if (len < 1e-9) break;
                    Vec3 dir = delta * (1.0 / len);
                    Hit h = scene.closestHit(Ray{pos, dir}, 1e-6);
                    if (!h.valid || h.t > len + kSkin) { pos = pos + delta; break; }  // clear path
                    double advance = h.t - kSkin; if (advance < 0.0) advance = 0.0;   // stop short
                    pos = pos + dir * advance;
                    if (collide == COLLIDE_STOP) break;
                    // Slide: strip the into-wall component from the leftover motion. h's
                    // geometric normal oriented toward us (orientedGeoN) is the wall plane's.
                    Vec3 n = orientedGeoN(h);
                    Vec3 remain = dir * (len - advance);
                    delta = remain - n * dot(remain, n);
                }
                return pos;
            };
            // Interactive render resolution IS the live window: the raster renders at the
            // image area's OWN pixel dimensions, so the preview always FILLS the window with
            // no letterbox bars, and resizing in ANY direction changes the pixel count (drag
            // smaller for a faster nav on a heavy scene, larger for a crisper view). The
            // camera's horizontal FOV follows the window aspect while fov_y stays fixed
            // (lookAt derives tanHalfX = tanHalfY * VW/VH), exactly like a game viewport:
            // a wider window simply reveals more to the sides, with square pixels (no stretch).
            // Two guards: never render past the authored longest edge (growing the window
            // beyond the film res would only supersample the preview, not add real detail),
            // and never shrink the long edge below kMinLong. The eye/look_at readout and the
            // world-scaled crosshair stay resolution-independent. Recomputed every loop so a
            // live resize retunes it.
            auto fitRes = [&](int& outW, int& outH) {
                int cw = 0, ch = 0;
                if (!g_liveWin->clientSize(cw, ch)) { outW = W; outH = H; return; }
                int vw = std::max(1, cw), vh = std::max(1, ch);   // fill the window (its aspect)
                const int kMaxLong = std::max(W, H);   // cap at authored detail (no supersampling)
                int lo = std::max(vw, vh);
                if (lo > kMaxLong) {
                    double s = (double)kMaxLong / lo;
                    vw = std::max(1, (int)std::lround(vw * s));
                    vh = std::max(1, (int)std::lround(vh * s));
                }
                const int kMinLong = 160;   // guard against an absurdly tiny render
                lo = std::max(vw, vh);
                if (lo < kMinLong) {
                    double up = (double)kMinLong / lo;
                    vw = std::max(1, (int)std::lround(vw * up));
                    vh = std::max(1, (int)std::lround(vh * up));
                }
                outW = vw; outH = vh;
            };
            int VW = W, VH = H;
            fitRes(VW, VH);
            auto fmt3 = [](const Vec3& p) {
                char b[64]; std::snprintf(b, sizeof b, "%.2f, %.2f, %.2f", p.x, p.y, p.z);
                return std::string(b);
            };
            // ---- Control panel + camera-path (timeline) state ------------------------
            // The window hosts a strip of controls below the image (Clip / Reset always;
            // plus a timeline, Play/Pause, Path-lock toggle and two traversal-speed inputs
            // when a flyby path is present). enablePanel with pathCount<2 shows just the two
            // buttons. Path playback rides the SAME camera-index cursor the timeline scrubs.
            int pathCount = (int)explorePath.size();   // mutable: the editor rebuilds the path
            g_liveWin->enablePanel(pathCount, explorePathFps, collideShort(collide));
            bool   pathMode = false;    // locked to the camera path (orientation + travel follow it)
            bool   playing  = false;    // auto-advancing along the path
            double pathPos  = 0.0;      // fractional camera index (continuous; render uses the nearest)
            int    strideN  = 1;        // stride mode: cameras advanced per RENDERED frame
            double camPerSec = (explorePathFps > 0.0) ? explorePathFps : 30.0;  // rate mode: cameras / wall-second
            bool   rateMode  = true;    // true = cam/sec (wall clock), false = stride (per update)
            // Last values mirrored to the panel, so we only re-push on an actual change.
            int    lastIdxSent = -1; bool lastPlaySent = false, lastPathSent = false;
            CollideMode lastCollideSent = collide;
            double lastSpdSent = -1.0;   // last painted-speed readout pushed to the panel
            auto clampPos = [&](double p) { return std::clamp(p, 0.0, (double)std::max(0, pathCount - 1)); };
            using clock = std::chrono::steady_clock;
            auto prevT = clock::now();   // wall-clock delta for rate-mode traversal

            // ---- Interactive camera_curve EDITOR state --------------------------------
            // The user flies free, records/hand-places control points (position + look
            // direction), scrubs the spline built through them, inserts/deletes points, and
            // Saves a real camera_curve .ftsl block. `editPts` are the authored control
            // points; `explorePath` is REGENERATED by sampling a centripetal Catmull-Rom
            // spline through them (the same math ftsl uses for camera_curve), so the preview
            // is WYSIWYG with what the renderer will produce. Recording captures the free
            // flight as raw samples that are optionally RDP-simplified into control points.
            const int kPreviewPerSeg = 24;   // preview spline samples per control-point segment
            std::vector<PathFrame> editPts;  // authored control points (pose per point)
            bool   recording = false;        // "Rec": auto-sampling the free flight
            double recTol    = 0.0;          // simplify tolerance in world units (0 = keep raw)
            bool   recRaw    = false;        // "raw" checkbox: keep every sample (ignore tol)
            std::vector<PathFrame> recRawBuf;// raw samples captured in the current recording pass
            Vec3   lastRecPos{0, 0, 0}; bool haveRecPos = false;
            // Per-control-point traversal-SPEED multiplier (Phase 2 speed painting): 1.0 = the
            // curve's natural pace, >1 faster / <1 slower. Kept in lockstep with editPts and
            // exported on Save as `density_at` keyframes (camera density = inverse speed).
            std::vector<double> ptSpeed;
            // ---- loom CurveDrive sidecar (-anim, E2 "channel a") ----------------------
            // With `-anim <file.json>` the editor is reshaping loom's N-dimensional DRIVE
            // curve, not merely a camera path: the control points ARE the drive's points.
            // Channels 0..2 are what the viewport draws and the mouse moves (for a flyby
            // drive that is literally the camera eye); channels 3.. are values no 3-D
            // viewport can show, so they ride along per point in `ptExtra` and are written
            // back untouched. `animDrive` holds everything the editor does NOT own — the
            // drive's name, mode, closed flag and its channel->scene-variable bindings — so
            // saving a reshaped curve never drops associations loom put there.
            curvedrive::Drive animDrive;
            bool animActive = false;                   // -anim given (sidecar loaded or to be created)
            int  animDims   = 0;                       // drive channel count (0 = not driving a sidecar)
            std::vector<std::vector<double>> ptExtra;  // per point, channels 3.. (each animDims-3 long)
            // Set by every curve mutation (rebuildPath); consumed by the live channel to
            // resend the points before the next frame. Declared here, with the rest of the
            // anim state, so rebuildPath() can reach it — the live bridge itself is set up
            // much further down, just before the interactive loop.
            bool animPtsDirty = false;
            auto animExtraCount = [&]() -> size_t { return (size_t)std::max(0, animDims - 3); };
            // editPts carries two parallel per-point side tracks (the painted speed
            // multiplier and the anim extra channels). Every add/insert/erase goes through
            // these two helpers so a track can never drift out of alignment with the points
            // it annotates — a silent misalignment would mis-assign speeds and channels to
            // the wrong points on the next Save.
            auto trackInsert = [&](size_t i) {
                ptSpeed.insert(ptSpeed.begin() + std::min(i, ptSpeed.size()), 1.0);
                size_t ne = animExtraCount();
                if (!ne) { ptExtra.insert(ptExtra.begin() + std::min(i, ptExtra.size()), std::vector<double>()); return; }
                // A new point inherits its unseen channels from its neighbours (midpoint in
                // the middle, a copy at either end); zeroing them would silently punch a
                // hole in every non-spatial channel the drive carries.
                const std::vector<double>* a = (i > 0 && i - 1 < ptExtra.size()) ? &ptExtra[i - 1] : nullptr;
                const std::vector<double>* b = (i < ptExtra.size()) ? &ptExtra[i] : nullptr;
                std::vector<double> ex(ne, 0.0);
                for (size_t c = 0; c < ne; ++c) {
                    double va = (a && c < a->size()) ? (*a)[c] : 0.0;
                    double vb = (b && c < b->size()) ? (*b)[c] : 0.0;
                    ex[c] = (a && b) ? 0.5 * (va + vb) : (a ? va : vb);
                }
                ptExtra.insert(ptExtra.begin() + std::min(i, ptExtra.size()), std::move(ex));
            };
            auto trackErase = [&](size_t i) {
                if (i < ptSpeed.size()) ptSpeed.erase(ptSpeed.begin() + i);
                if (i < ptExtra.size()) ptExtra.erase(ptExtra.begin() + i);
            };
            // Current free pose as a control-point frame.
            auto poseNow = [&]() -> PathFrame { return PathFrame{eye, fwd, worldUp, fovY}; };
            // Regenerate the preview path (explorePath) + timeline from the control points.
            auto rebuildPath = [&]() {
                ptSpeed.resize(editPts.size(), 1.0);   // safety: keep the side tracks sized to the points
                ptExtra.resize(editPts.size(), std::vector<double>(animExtraCount(), 0.0));
                animPtsDirty = true;   // the drive curve moved: the live channel must resend it
                int oldCount = pathCount;
                explorePath.clear();
                int n = (int)editPts.size();
                if (n == 0) { pathCount = 0; if (oldCount != 0) g_liveWin->setPathCount(0); return; }
                if (n == 1) { explorePath.push_back(editPts[0]); pathCount = 1; if (oldCount != 1) g_liveWin->setPathCount(1); return; }
                std::vector<Vec3> P, Lk; P.reserve(n); Lk.reserve(n);
                for (const auto& e : editPts) { P.push_back(e.eye); Lk.push_back(e.eye + e.fwd); }
                const bool closed = false; const int nSeg = n - 1;
                const int total = nSeg * kPreviewPerSeg;
                explorePath.reserve((size_t)total + 1);
                for (int k = 0; k <= total; ++k) {
                    double g = (double)nSeg * k / total;
                    Vec3 pe = ftsl::catmullRomAt(P,  closed, g, 0.5);
                    Vec3 pl = ftsl::catmullRomAt(Lk, closed, g, 0.5);
                    Vec3 f = pl - pe; { double L = std::sqrt(dot(f, f)); f = (L > 1e-9) ? f * (1.0 / L) : editPts[0].fwd; }
                    int si = (int)g; if (si >= nSeg) si = nSeg - 1; double tt = g - si;
                    Vec3 uu = editPts[si].up * (1.0 - tt) + editPts[si + 1].up * tt;
                    { double lu = std::sqrt(dot(uu, uu)); if (lu > 1e-9) uu = uu * (1.0 / lu); }
                    double fv = editPts[si].fov * (1.0 - tt) + editPts[si + 1].fov * tt;
                    explorePath.push_back({pe, f, uu, fv});
                }
                pathCount = (int)explorePath.size();
                if (pathCount != oldCount) g_liveWin->setPathCount(pathCount);   // only notify on a real change
            };
            // Ramer-Douglas-Peucker simplify of a pose polyline by eye position (keeps ends).
            auto simplify = [&](const std::vector<PathFrame>& in, double tol) -> std::vector<PathFrame> {
                int n = (int)in.size();
                if (n < 3 || tol <= 0.0) return in;
                std::vector<char> keep((size_t)n, 0); keep[0] = keep[(size_t)n - 1] = 1;
                std::vector<std::pair<int,int>> stk; stk.push_back({0, n - 1});
                while (!stk.empty()) {
                    auto seg = stk.back(); stk.pop_back();
                    int a = seg.first, b = seg.second;
                    if (b <= a + 1) continue;
                    Vec3 A = in[(size_t)a].eye, B = in[(size_t)b].eye, AB = B - A;
                    double abl = std::sqrt(dot(AB, AB));
                    double maxd = -1.0; int mi = -1;
                    for (int i = a + 1; i < b; ++i) {
                        Vec3 AP = in[(size_t)i].eye - A; double d;
                        if (abl < 1e-12) d = std::sqrt(dot(AP, AP));
                        else { Vec3 c = cross(AB, AP); d = std::sqrt(dot(c, c)) / abl; }
                        if (d > maxd) { maxd = d; mi = i; }
                    }
                    if (maxd > tol && mi > 0) { keep[(size_t)mi] = 1; stk.push_back({a, mi}); stk.push_back({mi, b}); }
                }
                std::vector<PathFrame> out;
                for (int i = 0; i < n; ++i) if (keep[(size_t)i]) out.push_back(in[(size_t)i]);
                return out;
            };
            // Round-trip (Phase 5): if the scene came from an existing `camera_curve`, seed the
            // editor's control points from it so the curve can be EDITED in place rather than
            // starting from an empty editor. We keep the loaded `explorePath` (the fully expanded
            // flyby) for high-fidelity playback and only populate `editPts` / `ptSpeed` here — the
            // overlay's control-point markers appear immediately, and the first authoring action
            // refines the loaded points instead of replacing the path. Speed round-trips from the
            // curve's `density` as a relative multiplier (mean/rho), so Save re-emits the profile.
            if (!ftslScene.authoredCurves.empty()) {
                // With several camera_curves in one scene, seed from the one the viewer is
                // actually flying (matched by the recovered base name), not blindly the first.
                const auto* acp = &ftslScene.authoredCurves.front();
                if (!exploreCurveName.empty()) {
                    for (const auto& c : ftslScene.authoredCurves)
                        if (c.name == exploreCurveName) { acp = &c; break; }
                }
                const auto& ac = *acp;
                int n = (int)ac.eyes.size();
                editPts.clear(); editPts.reserve((size_t)n);
                for (int i = 0; i < n; ++i)
                    editPts.push_back(PathFrame{ac.eyes[(size_t)i], ac.fwds[(size_t)i], ac.up, ac.fov});
                ptSpeed.assign((size_t)n, 1.0);
                if ((int)ac.density.size() == n && n > 0) {
                    double mean = 0.0; for (double r : ac.density) mean += r; mean /= n;
                    if (mean > 1e-12)
                        for (int i = 0; i < n; ++i)
                            ptSpeed[(size_t)i] = std::clamp(mean / std::max(1e-12, ac.density[(size_t)i]), 0.1, 10.0);
                }
                if (g_liveWin) g_liveWin->setEditState(false, n);
                std::printf("[editor] loaded %d control points from camera_curve \"%s\" (edit in place)\n",
                            n, ac.name.c_str());
                std::fflush(stdout);
            }
            // -anim: seed from the loom CurveDrive sidecar. This runs AFTER the camera_curve
            // seed above so an explicitly-named drive wins over whatever curve the scene
            // happened to carry. A sidecar that does not exist yet is not an error — that is
            // how you START a drive from the editor: keep whatever points the scene seeded
            // (or none) and let the first Save create the file.
            if (!animSidecar.empty()) {
                animActive = true;
                std::string err;
                curvedrive::Drive d;
                if (curvedrive::load(animSidecar, d, err)) {
                    animDrive = d;
                    animDims  = d.dims;
                    int n = (int)d.points.size();
                    editPts.clear(); editPts.reserve((size_t)n);
                    ptExtra.assign((size_t)n, std::vector<double>(animExtraCount(), 0.0));
                    for (int i = 0; i < n; ++i) {
                        const std::vector<double>& p = d.points[(size_t)i];
                        Vec3 e{p.size() > 0 ? p[0] : 0.0, p.size() > 1 ? p[1] : 0.0, p.size() > 2 ? p[2] : 0.0};
                        editPts.push_back(PathFrame{e, fwd, worldUp, fovY});
                        for (size_t c = 3; c < p.size(); ++c) ptExtra[(size_t)i][c - 3] = p[c];
                    }
                    // A drive is a curve of VALUES, so the sidecar stores no orientation. Aim
                    // each point down the chord to its successor (the last one keeps its
                    // predecessor's aim) — the same direction `look curve` would produce, and
                    // any of it can be re-aimed by orientation painting.
                    for (int i = 0; i < n; ++i) {
                        int a = (i + 1 < n) ? i : i - 1, b = (i + 1 < n) ? i + 1 : i;
                        if (a < 0 || b < 0) break;
                        Vec3 ch = editPts[(size_t)b].eye - editPts[(size_t)a].eye;
                        if (dot(ch, ch) > 1e-18) editPts[(size_t)i].fwd = norml(ch);
                    }
                    ptSpeed.assign((size_t)n, 1.0);
                    rebuildPath();
                    if (g_liveWin) g_liveWin->setEditState(false, n);
                    std::printf("[editor] -anim: drive \"%s\" (%s) — %d points x %d channels, %zu binding(s) from %s\n",
                                d.name.c_str(), d.mode.c_str(), n, d.dims,
                                d.bindings.size(), animSidecar.c_str());
                    if (!d.bindings.empty()) {
                        for (const auto& b : d.bindings)
                            std::printf("[editor]   ch%d -> %s (%s, gain %.4g, %s)\n",
                                        b.channel, b.target.c_str(), b.mode.c_str(), b.gain, b.kind.c_str());
                    }
                    if (d.dims > 3)
                        std::printf("[editor]   channels 3..%d are not spatial — carried per point and saved unchanged\n",
                                    d.dims - 1);
                } else {
                    // Fresh drive: the editor's own points are the camera path, so label it a
                    // flyby (loom's "channels collapse to camera pose") with the three spatial
                    // channels. Bindings can be added later by loom or a future panel.
                    animDims = 3;
                    animDrive = curvedrive::Drive();
                    animDrive.dims = 3;
                    animDrive.mode = curvedrive::kModeFlyby;
                    { std::string b = inFile ? std::string(inFile) : std::string("scene");
                      size_t sl = b.find_last_of("/\\"); if (sl != std::string::npos) b = b.substr(sl + 1);
                      size_t dt = b.find_last_of('.');   if (dt != std::string::npos) b = b.substr(0, dt);
                      animDrive.name = b + "_drive"; }
                    ptExtra.assign(editPts.size(), std::vector<double>());
                    std::printf("[editor] -anim: starting a new drive \"%s\" (%s) — Save writes %s\n",
                                animDrive.name.c_str(), err.c_str(), animSidecar.c_str());
                }
                std::fflush(stdout);
            }
            // Write the authored control points as a camera_curve .ftsl block, next to the
            // scene file AND echoed to stdout so it can be pasted straight into a scene.
            auto saveCurveFn = [&]() {
                if (editPts.size() < 2) {
                    std::printf("[editor] need >= 2 control points to save a camera_curve (have %zu)\n", editPts.size());
                    std::fflush(stdout); return;
                }
                std::string scenePath = inFile ? std::string(inFile) : std::string("scene");
                std::string dir, base = scenePath;
                { size_t sl = scenePath.find_last_of("/\\"); if (sl != std::string::npos) { dir = scenePath.substr(0, sl + 1); base = scenePath.substr(sl + 1); } }
                { size_t dt = base.find_last_of('.'); if (dt != std::string::npos) base = base.substr(0, dt); }
                std::string curveName = base + "_edit";
                // Pick a non-clobbering output filename.
                std::string outPath;
                for (int k = 0; k < 1000; ++k) {
                    std::string cand = dir + base + "_curve" + (k ? std::to_string(k) : std::string()) + ".ftsl";
                    std::ifstream test(cand);
                    if (!test.good()) { outPath = cand; break; }
                }
                if (outPath.empty()) outPath = dir + base + "_curve.ftsl";
                int frames = std::max(2, (int)explorePath.size());
                char hdr[256];
                std::string blk;
                blk += "camera_curve \"" + curveName + "\" {\n";
                blk += "    spline centripetal\n";
                { const Vec3& u0 = editPts.front().up;
                  std::snprintf(hdr, sizeof hdr, "    up %.6g %.6g %.6g\n", u0.x, u0.y, u0.z); blk += hdr; }
                std::snprintf(hdr, sizeof hdr, "    fov_y %.6g\n", editPts.front().fov); blk += hdr;
                std::snprintf(hdr, sizeof hdr, "    mode %c\n", rc0.mode); blk += hdr;
                std::snprintf(hdr, sizeof hdr, "    frames %d\n", frames); blk += hdr;
                if (explorePathFps > 0.0) { std::snprintf(hdr, sizeof hdr, "    fps %.6g\n", explorePathFps); blk += hdr; }
                blk += "    look curve\n";
                for (const auto& e : editPts) {
                    std::snprintf(hdr, sizeof hdr, "    point %.6g %.6g %.6g\n", e.eye.x, e.eye.y, e.eye.z); blk += hdr;
                }
                // The `look curve` is a SECOND Catmull-Rom spline through these look targets. Its
                // aim direction only stays smooth between control points when the targets sit a
                // reasonable distance ahead: too close and (lookSample - eyeSample) shrinks toward
                // the two splines' interpolation noise, making the aim swing/bow. Placing each
                // target one MEAN control-point spacing ahead (scene-relative, clamped) keeps the
                // look spline a smooth parallel-ish offset of the eye path. Direction at each
                // control point is preserved exactly (any positive distance along the same fwd).
                double lookAhead = 0.0;
                for (size_t i = 1; i < editPts.size(); ++i)
                    lookAhead += std::sqrt(dot(editPts[i].eye - editPts[i - 1].eye, editPts[i].eye - editPts[i - 1].eye));
                lookAhead = (editPts.size() > 1) ? lookAhead / (editPts.size() - 1) : 0.0;
                if (!(lookAhead > 1e-4)) lookAhead = (sceneR > 1e-4 ? sceneR * 0.1 : 1.0);
                for (const auto& e : editPts) {
                    Vec3 lp = e.eye + e.fwd * lookAhead;   // look target ~one segment ahead along the view ray
                    std::snprintf(hdr, sizeof hdr, "    look_point %.6g %.6g %.6g\n", lp.x, lp.y, lp.z); blk += hdr;
                }
                // Painted speed -> camera density (density = 1/speed). Emitted only when the
                // pace is non-uniform; with `frames N` fixed, ftsl distributes the N cameras by
                // this rho profile (absolute scale is irrelevant — only the relative shape).
                bool nonUniform = false;
                if (ptSpeed.size() == editPts.size())
                    for (double s : ptSpeed) if (std::fabs(s - 1.0) > 1e-3) { nonUniform = true; break; }
                if (nonUniform) {
                    int np = (int)editPts.size();
                    for (int i = 0; i < np; ++i) {
                        double t = (np > 1) ? (double)i / (np - 1) : 0.0;
                        double rho = 1.0 / std::max(ptSpeed[(size_t)i], 1e-3);
                        std::snprintf(hdr, sizeof hdr, "    density_at %.4g %.4g\n", t, rho); blk += hdr;
                    }
                }
                blk += "}\n";
                std::ofstream of(outPath);
                if (of.good()) { of << blk; of.close();
                    std::printf("[editor] saved camera_curve (%zu points, %d frames) to %s\n",
                                editPts.size(), frames, outPath.c_str());
                } else {
                    std::printf("[editor] FAILED to write %s — block echoed below only\n", outPath.c_str());
                }
                std::printf("%s", blk.c_str());
                // -anim: write the reshaped drive back to its sidecar (E2 channel a). The
                // editor owns only the point LIST — each point's spatial channels come from
                // its eye, its non-spatial channels from the values it carried in. Name,
                // mode, closed flag, dims and every channel->variable binding are copied
                // from whatever the sidecar last held, so an editing pass here reshapes the
                // curve without ever dropping an association loom authored.
                if (animActive) {
                    curvedrive::Drive d = animDrive;
                    d.dims = std::max(1, animDims);
                    d.points.clear(); d.points.reserve(editPts.size());
                    for (size_t i = 0; i < editPts.size(); ++i) {
                        std::vector<double> row((size_t)d.dims, 0.0);
                        const Vec3& e = editPts[i].eye;
                        if (d.dims > 0) row[0] = e.x;
                        if (d.dims > 1) row[1] = e.y;
                        if (d.dims > 2) row[2] = e.z;
                        for (int c = 3; c < d.dims; ++c)
                            row[(size_t)c] = (i < ptExtra.size() && (size_t)(c - 3) < ptExtra[i].size())
                                           ? ptExtra[i][(size_t)(c - 3)] : 0.0;
                        d.points.push_back(std::move(row));
                    }
                    std::string serr;
                    if (curvedrive::save(animSidecar, d, serr)) {
                        animDrive = d;   // the sidecar and the editor now agree
                        std::printf("[editor] saved drive \"%s\" (%zu points x %d channels, %zu binding(s)) to %s\n",
                                    d.name.c_str(), d.points.size(), d.dims, d.bindings.size(),
                                    animSidecar.c_str());
                    } else {
                        std::printf("[editor] FAILED to write sidecar %s: %s\n", animSidecar.c_str(), serr.c_str());
                    }
                }
                std::fflush(stdout);
            };
            // The currently SELECTED control point — the target Del removes and the overlay
            // highlights red. When locked to the path (scrubbing/playing) the selection follows
            // the timeline: it's the control point nearest the current scrub position, so you
            // scrub to a point to select it. In free flight it's the point nearest the eye.
            // Returns -1 when there are no points.
            auto selectedPoint = [&]() -> int {
                int n = (int)editPts.size();
                if (n == 0) return -1;
                if (n == 1) return 0;
                if (pathMode && pathCount >= 2) {
                    double t = pathPos / (double)(pathCount - 1);         // 0..1 along the timeline
                    int k = (int)std::llround(t * (double)(n - 1));       // nearest control point
                    return std::clamp(k, 0, n - 1);
                }
                int best = 0; double bd = 1e300;
                for (int i = 0; i < n; ++i) { Vec3 d = editPts[(size_t)i].eye - eye; double dd = dot(d, d); if (dd < bd) { bd = dd; best = i; } }
                return best;
            };
            // Draw the control-point markers + the live spline polyline over the tone-mapped
            // frame. Camera::project gives py with +up = larger py; the RGB buffer is row-0-top,
            // so the screen row is (h-1-py). Segments are drawn only when both ends project.
            auto drawOverlay = [&](const Camera& c, int w, int h, std::vector<uint8_t>& img) {
                if (explorePath.size() < 2 && editPts.empty()) return;
                auto putpx = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
                    if (x < 0 || y < 0 || x >= w || y >= h) return;
                    size_t i = ((size_t)y * w + x) * 3; img[i] = r; img[i + 1] = g; img[i + 2] = b;
                };
                auto proj = [&](const Vec3& p, int& sx, int& sy) -> bool {
                    int px, py; double cc, d2; if (!c.project(p, px, py, cc, d2)) return false;
                    sx = px; sy = h - 1 - py; return true;
                };
                auto line = [&](const Vec3& a, const Vec3& b, uint8_t r, uint8_t g, uint8_t bl) {
                    int x0, y0, x1, y1; if (!proj(a, x0, y0) || !proj(b, x1, y1)) return;
                    int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
                    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy;
                    for (;;) { putpx(x0, y0, r, g, bl); if (x0 == x1 && y0 == y1) break;
                        int e2 = 2 * err; if (e2 >= dy) { err += dy; x0 += sx; } if (e2 <= dx) { err += dx; y0 += sy; } }
                };
                auto marker = [&](const Vec3& p, uint8_t r, uint8_t g, uint8_t b) {
                    int sx, sy; if (!proj(p, sx, sy)) return;
                    for (int yy = -2; yy <= 2; ++yy) for (int xx = -2; xx <= 2; ++xx) putpx(sx + xx, sy + yy, r, g, b);
                };
                for (size_t i = 1; i < explorePath.size(); ++i)
                    line(explorePath[i - 1].eye, explorePath[i].eye, 40, 220, 90);   // green spline
                // The selected control point (Del target) is highlighted red; the rest yellow.
                int sel = selectedPoint();
                for (size_t i = 0; i < editPts.size(); ++i)
                    marker(editPts[i].eye, 255, ((int)i == sel) ? 60 : 220, ((int)i == sel) ? 60 : 40);  // yellow / red-selected
            };
            // ---- ZERO-COPY present (CUDA <-> Direct3D 11) -----------------------------
            // The fastest way to show a GPU-rastered frame is not to move it: the tonemap
            // writes its bytes straight into the texture the live window's D3D11 presenter
            // samples, so the image never crosses the PCIe bus and no host code ever touches
            // a pixel. That skips, per frame, the device->host download, the vector copy out
            // of the pinned buffer, and the host->device re-upload the presenter would do.
            //
            // It only applies when nothing needs the pixels on the HOST — i.e. the curve
            // editor's overlay (control-point markers + spline polyline, drawn with putpx into
            // the RGB buffer) is not active. Every other case, and any failure at any step
            // (no CUDA, CPU rasterizer, GDI window, D3D on a different adapter than the CUDA
            // device), returns false and the caller renders the ordinary way.
            int zeroCopyOn = -1;   // last reported state: -1 = unreported, 0 = host path, 1 = zero-copy
            auto rasterPresent = [&](const Camera& c, int w, int h, double expo, bool autoExp_) -> bool {
#ifdef HAVE_CUDA
                if (!g_liveWin || !gpuRaster) return false;
                if (useGpuIso && !c.hasLens()) return false;      // implicit-ray preview owns this frame
                if (!(explorePath.size() < 2 && editPts.empty())) return false;   // overlay needs host pixels
                bool zc = g_liveWin->renderShared(w, h, [&](void* dev, void* tex) -> bool {
                    if (!raster_cuda::bindPresentTarget(gpuRaster, dev, tex, w, h)) return false;
                    return raster_cuda::renderFrameToTarget(gpuRaster, c, w, h, nThreads, expo,
                                                            autoExp_, nullptr,
                                                            rasterSeeThrough, rasterClarity);
                });
                // Say which way the pixels are actually flowing — whether the interop engaged
                // is invisible otherwise (both paths show the same image), and it can legitimately
                // be off (GDI window, or D3D on a different adapter than the CUDA device).
                if ((int)zc != zeroCopyOn) {
                    zeroCopyOn = (int)zc;
                    std::printf(zc ? "[raster] zero-copy present: the tonemap writes the live window's "
                                     "D3D11 texture directly (no host readback)\n"
                                   : "[raster] zero-copy present off: frames travel through host memory\n");
                    std::fflush(stdout);
                }
                return zc;
#else
                (void)c; (void)w; (void)h; (void)expo; (void)autoExp_;
                return false;
#endif
            };
            // ---- Speed / orientation PAINTING (Phase 2/3) -----------------------------
            // The painted tracks are anchored to the CONTROL POINTS: speed is a per-point
            // multiplier (ptSpeed) and orientation is each point's own look direction. The
            // brush at a scrub position distributes its effect to the two bracketing control
            // points, weighted by proximity — so scrubbing/playing while painting shapes a
            // smooth track that the exported density_at / look curve reproduce.
            auto bracket = [&](double pos, int& i, double& f) {   // scrub index -> (control seg, frac)
                int n = (int)editPts.size();
                int len = (int)explorePath.size();
                double t = (len > 1) ? pos / (len - 1) : 0.0;   // normalized 0..1 along the curve
                double g = t * std::max(0, n - 1);
                i = (int)g; if (i < 0) i = 0; if (i > n - 2) i = std::max(0, n - 2);
                f = (n >= 2) ? g - i : 0.0;
            };
            auto speedAt = [&](double pos) -> double {
                int n = (int)ptSpeed.size();
                if (n == 0) return 1.0;
                if (n == 1) return ptSpeed[0];
                int i; double f; bracket(pos, i, f);
                return ptSpeed[(size_t)i] * (1.0 - f) + ptSpeed[(size_t)i + 1] * f;
            };
            auto paintSpeed = [&](double pos, double notches) {
                int n = (int)ptSpeed.size();
                if (n == 0) return;
                double delta = notches * 0.15;   // additive per wheel notch
                auto bump = [&](int k, double w) {
                    if (k < 0 || k >= n || w <= 0.0) return;
                    ptSpeed[(size_t)k] = std::clamp(ptSpeed[(size_t)k] + delta * w, 0.1, 10.0);
                };
                if (n == 1) { bump(0, 1.0); return; }
                int i; double f; bracket(pos, i, f); bump(i, 1.0 - f); bump(i + 1, f);
            };
            auto paintOrient = [&](double pos, double lx, double ly) {
                int n = (int)editPts.size();
                if (n == 0) return;
                double yaw = -lx * kYaw, pitch = -ly * kPitch;
                auto steer = [&](int k, double w) {
                    if (k < 0 || k >= n || w <= 1e-6) return;
                    Vec3 d = norml(rotAxis(editPts[(size_t)k].fwd, worldUp, yaw * w));
                    Vec3 right = cross(d, worldUp); double rl = std::sqrt(dot(right, right));
                    if (rl > 1e-9) { right = right * (1.0 / rl);
                        Vec3 cand = norml(rotAxis(d, right, pitch * w));
                        if (std::fabs(dot(cand, worldUp)) < 0.9995) d = cand; }
                    editPts[(size_t)k].fwd = d;
                };
                if (n == 1) { steer(0, 1.0); rebuildPath(); return; }
                int i; double f; bracket(pos, i, f); steer(i, 1.0 - f); steer(i + 1, f);
                rebuildPath();
            };
            if (pathCount >= 2)
                std::printf("[viewer] camera path: %d frames on the timeline"
                            " (Play/scrub/lock via the panel below the image)\n", pathCount);
            std::printf(
              "[viewer] interactive fly-camera — fly around, then copy the printed camera block:\n"
              "         move:   Space or +  = fly forward     Shift or -  = fly backward   (you travel where you look)\n"
              "         dolly:  mouse wheel up/down = dolly forward/back one notch (each notch renders — no overshoot; Ctrl+wheel scales it)\n"
              "         look:   move the mouse off-centre to steer — offset from centre = turn rate (centre holds still); cursor stays visible; leave the window to stop\n"
              "         step:   Ctrl + mouse wheel = bigger/smaller step (now %.3g u; travel scales with render speed)\n"
              "         collide: C cycles wall collision (now: %s) — slide along walls / stop dead / noclip\n"
              "         trace:  T toggles a live PATH-TRACED preview (fast RGB backward) — holds still to converge, re-aims on move (GPU, if the scene is in RGB scope)\n"
              "         panel:  Clip / Reset buttons below the image%s\n"
              "         editor: Rec records your flight into a camera_curve; +Pt appends the current pose;\n"
              "                 Ins inserts at the scrub point; Del removes the nearest point; Save writes a camera_curve block\n"
              "         paint:  Paint (path mode) — wheel paints local speed (density) at the scrub point, mouse steers orientation; Flat resets speed\n"
              "         0 = reset view    P = print camera block    (close the window to finish)\n"
              "         resize the window to change the preview resolution — the render fills the window (no bars): smaller = faster on a heavy scene, larger = crisper; the horizontal view widens/narrows with the window (fov_y fixed)\n",
              step, collideName(collide),
              pathCount >= 2 ? "; timeline + Play/Pause + Path-lock + cams/upd | cams/s speed switch"
                             : "");
            std::fflush(stdout);

            bool changed = true;   // render one frame immediately
            // ---- GPU clock keep-warm (fixes the "bursty explorer feels slow" problem) --
            // The explorer re-renders ONE frame per camera move, then idle-sleeps. On a
            // discrete GPU the driver's DVFS reads that bursty, low-duty submission pattern
            // as "idle" and parks the card in its lowest power state (measured on an RTX
            // 4090: P8 @ 210 MHz vs a 2520-2760 MHz boost under load — a ~13x clock drop,
            // and ~33x slower for a first cold frame once first-frame allocs are added).
            // So each fresh mouse-look burst pays a cold-clock penalty until a second or two
            // of continuous motion finally ramps the clocks — exactly the "slow, then
            // suddenly fast, then slow again after relaunch" the card's power management
            // produces. To keep exploration responsive we hold the GPU warm during an
            // ACTIVE session: for a short grace window after the last real interaction we
            // keep submitting GPU render work (which is what the driver needs to hold the
            // boost clock) even when the frame hasn't changed, WITHOUT touching the display.
            // Once the user is truly idle past the grace window we fall back to the passive
            // sleep and let the card power all the way down. warmOne() reuses rasterOne but
            // discards its result — it exists purely to keep the clocks up. GPU only.
            auto lastActiveT = clock::now();
            const double kWarmGraceSec = 2.5;   // hold clocks this long after the last move
            // A warm-only frame fires only once the user has genuinely PAUSED — i.e. after
            // `idleFor` passes this gap. During an active scrub/mouse drag the sub-frame gaps
            // between input events stay under it, so a warm frame never lands between two
            // events and can't steal the loop slot that samples the next scrub position (that
            // stolen slot was the timeline "chunking" by several cameras per drag). Once past
            // the gap (a real pause) warm frames run CONTINUOUSLY to actually hold the boost
            // clock — a sparse rate-limited trickle was measured too weak (card stayed at P8).
            const double kWarmGapSec = 0.10;
            bool gpuWarmKeep = false;
#ifdef HAVE_CUDA
            gpuWarmKeep = (gpuRaster != nullptr);   // only meaningful on the discrete GPU path
#endif
            // ---- Interactive LIT preview, cycled with 'T' -------------------------------
            // The explorer normally shows the flat-shaded raster (instant, for navigation).
            // 'T' cycles that for a real lit render of the pose you are standing at:
            //
            //   RASTER  ->  W  ->  PT  ->  RASTER ...
            //
            //   W  = mode W, the deterministic Whitted preview, on the CPU. ONE render per
            //        pose, noise-free at 1 spp, and it works on ANY scene (full spectral
            //        walk, all materials, media, env, and the -gi one-bounce gather if asked
            //        for). This is the preview that is always available.
            //   PT = progressively PATH-TRACE the pose with the fast RGB backward tracer
            //        (Stage 2) into a resident GPU session: it keeps converging while you
            //        hold still, so it ends up more correct than W (real multi-bounce GI),
            //        but it needs a CUDA GPU and a scene inside the fast-RGB scope, and it
            //        starts noisy. Skipped in the cycle when unavailable.
            //
            // Either way the instant the camera moves we drop back to the responsive raster,
            // then re-render/re-aim once you settle. The scene-ignore flags (-no-media/-env/
            // -fluoro, -max-bounce, -direct-only) already apply — they mutated the Scene
            // before this point, so both previews inherit them.
            // `-explore -mode W` opens straight into the mode-W preview: asking for the
            // deterministic preview mode AND the interactive viewer in the same command can
            // only mean you want to fly around the lit image, not the flat raster. Plain
            // -explore still opens on the raster, which is what you want for navigating.
            enum PreviewMode { PV_RASTER = 0, PV_WHITTED, PV_PT };
            PreviewMode pvMode = g_whitted ? PV_WHITTED : PV_RASTER;   // 'T' cycles this
            bool  traceAvail = false;             // scene+camera in fast-RGB scope on this GPU (PV_PT)
            bool  traceDirty = true;              // camera moved -> re-render / re-aim + restart accumulation
            double traceAnchor = 0.0;             // locked auto-exposure anchor for the current pose (0 = recompute)
            const long long kTraceBatchSpp = 4;   // spp accumulated per idle iteration (responsive batches)
            const long long kTraceCapSpp   = 4096;// stop refining once this converged (idle after)
            // PV_WHITTED progressive state. A mode-W frame is one deterministic pass, but it
            // can cost seconds, so it is rendered as a stack of row BANDS with one band per
            // viewer iteration: input keeps being drained between bands, and moving the camera
            // simply abandons the unfinished rows. Band height is retuned from the measured
            // cost of the previous band to hold ~kWBandSec, which is what lets the same code
            // stay responsive on both a 0.4s Cornell box and a 26s gyroid labyrinth.
            Film wFilm;                           // full-res mode-W film for the current pose (accumulates)
            std::vector<uint8_t> wImg;            // tone-mapped RGB8 shown so far (coarse pass, then bands)
            int    wRow      = 0;                 // film rows still to render are [0, wRow); bands come off the top
            int    wBandRows = 16;                // adaptive band height (rows)
            int    wPass     = 0;                 // absolute sample index of the pass being rendered
            int    wResX = 0, wResY = 0;          // wFilm size (rebuilt on a resize)
            const double kWBandSec = 0.10;        // target wall-time per band: responsiveness vs. overhead
            const int    kWCoarse  = 16;          // first pass is 1/16 linear (1/256 the pixels)
            // Mode W is exact at 1 spp only while the whole path stays in the hero BUNDLE, which
            // carries heroC wavelengths at once. Anything that DE-HEROES the path onto a single
            // wavelength breaks that, because mode W's wavelength lattice is a function of the
            // sample index alone -- shared by every pixel -- so at 1 spp the entire object is
            // rendered at ONE wavelength and comes out strongly mistinted. Extra passes are the
            // fallback; they are only worth taking when the scene actually contains such a
            // material, since anywhere else they are bit-for-bit identical work.
            //
            // The DISPERSIVE materials (dielectric / thin-film / multilayer / grating /
            // half-mirror / fluorescent) used to be the main offender -- a Cornell SF10 ball
            // rendered flat green until ~16 spp. They no longer are: mode W now SPLITS the bundle
            // at a dispersive vertex into C monochromatic sub-paths, each on its own Snell
            // direction (BackwardRenderer::heroSplit), so glass is colour-correct at 1 spp.
            // `Layered` used to be the remaining offender for the same reason (it de-hero'd
            // unconditionally, so a clearcoat previewed monochromatic); as of v0.115.1 its coat
            // reflectance is applied as a PER-λ weight with the bundle intact, and only a
            // genuinely chromatic coat -- one where the R >= 0.5 dominant branch differs across
            // λ -- fans out, so layered scenes are 1-spp-clean too. What still de-heroes is the
            // scalar (bundle-free) path taken for media / GRIN / heroC 1.
            // (Thin-film and multilayer needed a SECOND fix beyond the split, in v0.112.0: the
            // split gives each λ its own direction, but the reflect-or-transmit choice at the
            // interface was still a coin flip with no `whitted` branch, so those two stayed
            // grainy here even at C > 1. They now take the dominant branch weighted, like
            // Dielectric -- see thinFilmInterface's whittedWeight.)
            // (No hasLens() term: the viewer builds its camera fresh from the pose each frame,
            // so the preview camera is always a plain one even if the scene authored a lens.)
            const int kWSppCap = 16;
            bool wNeedSpp = (g_heroC <= 1) || scene.backwardMedium().enabled ||
                            grin::sceneHasGrin(scene);
            // Rough GLOSSY is deliberately NOT on that list, even though its lobe is likewise
            // resolved across samples (whittedGlossyDir) rather than within one. The difference
            // is what a single pass looks like: a de-hero'd dielectric is flatly WRONG (a green
            // ball), whereas a one-direction lobe is merely SHARP -- it reads as a shinier
            // metal, not as an error. Making every satin surface cost 16 passes to settle would
            // trade the viewer's whole reason for existing against a subtle look difference, so
            // resolving the lobe is left to an explicit `-spp` on a batch render.
#ifdef HAVE_CUDA
            BackwardRGBSession* traceSess = nullptr;   // resident RGB-backward preview (lazy)
            int   traceResX = 0, traceResY = 0;        // session film size (recreated on a resize)
            Film  traceFilm;                           // scratch download film (lazily sized)
            if (gpuRaster != nullptr)
                traceAvail = cudaBackwardRGBSupported(scene, rc0.cam);   // same scope the batch -rgb uses
#endif
            std::printf("[viewer] press 'T' to cycle the lit preview: raster -> mode W "
                        "(deterministic, CPU, any scene) -> %s%s\n",
                        traceAvail ? "path-traced (fast RGB, GPU)"
                                   : "(path-traced GPU preview unavailable: no CUDA raster, or "
                                     "scene/camera outside fast-RGB scope)",
                        pvMode == PV_WHITTED ? "  [starting in mode W]" : "");
            std::fflush(stdout);
            // ---- loom LIVE channel (-anim + -loom, E2 "channel b") --------------------
            // With a loom build file the editor stops being a sidecar editor and becomes a
            // live one: each scrub position is pushed to a resident `python -m loom.anim`,
            // which applies the drive's channel->variable bindings and emits that frame's
            // .ftsl; we load it and preview the actual animated scene.
            //
            // LOOM samples the curve — we push our control points and ask by parameter `t`
            // (see animlive.h). Sampling here instead would risk the preview disagreeing
            // with the video loom finally renders, which is the whole point of the channel.
            animlive::Bridge animBridge;
            bool animLive        = false;   // the live channel is up
            double animLastT     = -1.0;    // last curve position we asked loom for
            long long animBaked  = 0;       // frames loom has emitted this session
            double animLastMs    = 0.0;
            std::string animLastErr;
            // Last bind-row state pushed to the panel, so the mirror only fires on a real change.
            std::string animLastStatus;
            std::vector<std::string> animLastTargets;
            int animLastBindCh = -2;        // -2 = "never sent" (-1 is a legitimate "no selection")
            if (animActive && !animLoomScene.empty()) {
                std::string lerr;
                if (animBridge.start(animLoomScene, animSidecar, lerr)) {
                    animLive = true;
                    animPtsDirty = true;    // seed loom with the curve we actually loaded
                    std::printf("[anim] live channel up: %s\n", animBridge.command().c_str());
                    const auto& sl = animBridge.slots();
                    std::printf("[anim] %zu bindable scene variable(s)%s", sl.size(),
                                sl.empty() ? "\n" : ": ");
                    for (size_t i = 0; i < sl.size() && i < 12; ++i)
                        std::printf("%s%s", sl[i].first.c_str(),
                                    (i + 1 < sl.size() && i < 11) ? ", " : "\n");
                    if (sl.size() > 12) std::printf("[anim]   (+%zu more)\n", sl.size() - 12);
                    // Reveal the bind row: only now do we know what the scene actually exposes,
                    // and a pick-list is the only honest way to offer it (a typed name would
                    // just be a typo loom silently ignores).
                    if (g_liveWin) {
                        std::vector<std::string> names;
                        names.reserve(sl.size());
                        for (const auto& s : sl) names.push_back(s.first);
                        g_liveWin->enableBindRow(names, std::max(1, animDims));
                    }
                } else {
                    std::fprintf(stderr, "[anim] live channel unavailable: %s\n", lerr.c_str());
                    std::fprintf(stderr, "[anim] continuing as a sidecar-only editor "
                                         "(the curve still saves; the preview stays static)\n");
                }
                std::fflush(stdout);
            } else if (animActive) {
                std::printf("[anim] no -loom <scene.py>: sidecar-only editor "
                            "(add -loom to preview the animation live)\n");
                std::fflush(stdout);
            }
            // Push the editor's control points to loom (all channels: eye + the carried
            // extras). A control message, not a sample — it must never be dropped, or the
            // next frame would be emitted against a curve that no longer exists.
            auto animSendPoints = [&]() {
                if (!animLive) return;
                int dims = std::max(1, animDims);
                std::string js = "{\"cmd\":\"points\",\"points\":[";
                for (size_t i = 0; i < editPts.size(); ++i) {
                    if (i) js += ",";
                    js += "[";
                    const Vec3& e = editPts[i].eye;
                    for (int c = 0; c < dims; ++c) {
                        if (c) js += ",";
                        double v = (c == 0) ? e.x : (c == 1) ? e.y : (c == 2) ? e.z
                                 : ((i < ptExtra.size() && (size_t)(c - 3) < ptExtra[i].size())
                                        ? ptExtra[i][(size_t)(c - 3)] : 0.0);
                        char b[40]; std::snprintf(b, sizeof b, "%.17g", v);
                        js += b;
                    }
                    js += "]";
                }
                js += "]}";
                animBridge.control(js);
                animPtsDirty = false;
            };
            // Push `animDrive.bindings` to loom. Sent WHOLESALE rather than as a delta: the
            // binding set is tiny and loom's `bindings` command replaces it outright, so there
            // is no incremental protocol to get out of step with.
            auto animSendBindings = [&]() {
                if (!animLive) return;
                std::string js = "{\"cmd\":\"bindings\",\"bindings\":[";
                for (size_t i = 0; i < animDrive.bindings.size(); ++i) {
                    const curvedrive::Binding& b = animDrive.bindings[i];
                    char g[40]; std::snprintf(g, sizeof g, "%.17g", b.gain);
                    if (i) js += ",";
                    js += "{\"channel\":" + std::to_string(b.channel)
                        + ",\"target\":\"" + loomlink::jsonEsc(b.target) + "\""
                        + ",\"mode\":\"" + loomlink::jsonEsc(b.mode) + "\""
                        + ",\"gain\":" + g
                        + ",\"kind\":\"" + loomlink::jsonEsc(b.kind) + "\"}";
                }
                js += "]}";
                animBridge.control(js);
            };
            // The bind row's per-channel view of the drive: entry c is channel c's target, or
            // "" when nothing binds it. Rebuilt from `animDrive` (the editor's authority for
            // everything that is not the curve's geometry) rather than cached, so it cannot
            // drift from what a Save would write.
            auto animTargets = [&]() {
                std::vector<std::string> t((size_t)std::max(1, animDims));
                for (const curvedrive::Binding& b : animDrive.bindings)
                    if (b.channel >= 0 && (size_t)b.channel < t.size()) t[(size_t)b.channel] = b.target;
                return t;
            };
            // Bind (or re-target) one channel. An empty target UNBINDS it — the "(none)" entry
            // in the slot pick-list — so Bind and Unbind are the same operation and cannot
            // disagree about what "no binding" means.
            auto animBind = [&](int channel, const std::string& target) {
                if (channel < 0 || channel >= std::max(1, animDims)) return;
                auto& bs = animDrive.bindings;
                auto it = std::find_if(bs.begin(), bs.end(),
                                       [&](const curvedrive::Binding& b) { return b.channel == channel; });
                if (target.empty()) {
                    if (it == bs.end()) return;                  // already unbound: nothing to say
                    std::printf("[anim] ch%d unbound (was %s)\n", channel, it->target.c_str());
                    bs.erase(it);
                } else if (it != bs.end()) {
                    if (it->target == target) return;            // idempotent: no rebake for a no-op
                    std::printf("[anim] ch%d -> %s (was %s)\n", channel, target.c_str(), it->target.c_str());
                    it->target = target;
                } else {
                    curvedrive::Binding b;                       // defaults: pin / gain 1 / additive
                    b.channel = channel;
                    b.target  = target;
                    std::printf("[anim] ch%d -> %s (%s, gain %.4g, %s)\n",
                                channel, target.c_str(), b.mode.c_str(), b.gain, b.kind.c_str());
                    bs.push_back(b);
                }
                std::fflush(stdout);
                animSendBindings();
                animLastT = -1.0;    // force a rebake: the same curve position now means something else
            };
            // Grow or shrink the drive's channel count. Growing appends zeroed channels to every
            // control point; SHRINKING DISCARDS them — and loom drops any binding that lived on a
            // channel that no longer exists, so we mirror that here rather than let the sidecar
            // keep a binding loom has already forgotten.
            auto animSetDims = [&](int nd) {
                nd = std::max(1, nd);
                if (nd == animDims) return;
                int old = animDims;
                animDims = nd;
                size_t ne = animExtraCount();
                for (auto& ex : ptExtra) ex.resize(ne, 0.0);
                animDrive.dims = nd;
                size_t before = animDrive.bindings.size();
                animDrive.bindings.erase(
                    std::remove_if(animDrive.bindings.begin(), animDrive.bindings.end(),
                                   [&](const curvedrive::Binding& b) { return b.channel >= nd; }),
                    animDrive.bindings.end());
                size_t dropped = before - animDrive.bindings.size();
                // Keep the suffix in a NAMED string: building it inline as an argument would
                // hand printf a c_str() into a temporary already destroyed at the sequence point.
                std::string note = dropped ? "  (" + std::to_string(dropped) + " binding(s) dropped)"
                                           : std::string();
                std::printf("[anim] channels %d -> %d%s\n", old, nd, note.c_str());
                std::fflush(stdout);
                if (animLive) {
                    animBridge.control("{\"cmd\":\"dims\",\"dims\":" + std::to_string(nd) + "}");
                    animSendBindings();
                    animSendPoints();
                    animLastT = -1.0;
                }
            };
            // Swap in a scene loom just emitted. Everything downstream of `scene` is
            // derived state and every bit of it has to be dropped: the preview light, the
            // tessellation, the GPU rasterizer's baked triangles, and the resident
            // RGB-backward session (which bakes the scene at begin() — setCamera only
            // re-aims, so a swap needs a full End/Begin). The user's POSE is deliberately
            // untouched: the scene changed under them, they did not move.
            auto animAdoptScene = [&](const std::string& path, std::string& aerr) -> bool {
                ftsl::Loaded nl;
                if (!ftsl::load(path, nl, aerr, supportFn)) return false;
                scene = std::move(nl.scene);
                plight = raster::deriveLight(scene);
                prims.clear();
                tessellated = false;
#ifdef HAVE_CUDA
                // Re-upload the GPU rasterizer only if it was the path in use; the CPU
                // and primary-ray-iso paths both want `prims` left lazy so a scrub the
                // user immediately supersedes never pays for a tessellation.
                if (gpuRaster) {
                    raster_cuda::destroy(gpuRaster);
                    gpuRaster = nullptr;
                    ensurePrims();
                    gpuRaster = raster_cuda::upload(prims, plight, &scene.textures);
                    if (!gpuRaster)
                        std::fprintf(stderr, "[anim] GPU re-upload failed; using the CPU rasterizer\n");
                }
                // The RGB-backward session bakes the scene at begin() — setCamera only
                // re-aims — so a scene swap needs a full End/Begin, not a re-aim.
                if (traceSess) { backwardRGBSessionEnd(traceSess); traceSess = nullptr; }
#endif
                // Outside the ifdef: the mode-W preview traces the live `scene` band by band,
                // so a scene swap must discard its half-finished frame too, CUDA or not.
                traceDirty = true;
                return true;
            };
            while (!g_liveWin->closed() && !g_stopRequested) {
                // Match the render resolution to the live window: a user resize re-renders
                // at the new size (smaller = faster, larger = crisper).
                { int nvw = VW, nvh = VH; fitRes(nvw, nvh);
                  if (nvw != VW || nvh != VH) {
                      VW = nvw; VH = nvh; changed = true;
                      std::printf("[viewer] preview resolution %dx%d\n", VW, VH);
                      std::fflush(stdout);
                  } }
                NavInput nav = g_liveWin->drainNav();

                // Fold in whatever loom finished since the last iteration. The editor
                // never blocks on an emit: it keeps flying the scene it already has and
                // adopts the new one on whatever iteration it lands.
                if (animLive) {
                    animlive::Result ar;
                    if (animBridge.take(ar)) {
                        animLastMs = ar.ms;
                        if (ar.ok) {
                            std::string aerr;
                            if (animAdoptScene(ar.ftslPath, aerr)) {
                                ++animBaked;
                                animLastErr.clear();
                                changed = true;          // repaint on the new geometry
                            } else {
                                // A re-derived scene ftrace cannot load is a real error and
                                // has to be said out loud, not silently left on stale
                                // geometry the user would read as "my edit did nothing".
                                animLastErr = "ftsl: " + aerr;
                                std::fprintf(stderr, "[anim] emitted scene did not load: %s\n",
                                             aerr.c_str());
                            }
                        } else {
                            animLastErr = ar.err;
                            std::fprintf(stderr, "[anim] %s\n", ar.err.c_str());
                        }
                        animBridge.reap(ar.ftslPath);
                        std::fflush(stderr);
                    }
                    if (!animBridge.linkUp() && animLastErr != animBridge.deadReason()) {
                        animLastErr = animBridge.deadReason();
                        animLive = false;
                        std::fprintf(stderr, "[anim] live channel lost: %s\n", animLastErr.c_str());
                        std::fprintf(stderr, "[anim] continuing as a sidecar-only editor\n");
                        std::fflush(stderr);
                    }
                }

                // Wall-clock delta for rate-mode (cameras/second) path traversal.
                auto nowT = clock::now();
                double dt = std::chrono::duration<double>(nowT - prevT).count();
                prevT = nowT;
                if (dt > 0.25) dt = 0.25;   // clamp a hitch so playback can't leap the whole path

                // Panel traversal-speed inputs (current values; 0 = leave unchanged).
                if (nav.stride    >= 1)   strideN   = nav.stride;
                if (nav.camPerSec > 0.0)  camPerSec = nav.camPerSec;
                rateMode = nav.rateMode;   // radio switch: true = cam/sec, false = stride/update

                // Path-lock toggle (panel "Path" button): snap the fly camera onto the path.
                if (pathCount >= 2 && nav.togglePath) {
                    pathMode = !pathMode;
                    if (!pathMode) playing = false;   // leaving the path stops playback
                    changed = true;
                    std::printf("[viewer] path lock %s\n", pathMode ? "ON" : "OFF"); std::fflush(stdout);
                }
                // Play/Pause (panel button): engage path lock and toggle auto-advance.
                if (pathCount >= 2 && nav.togglePlay) {
                    if (!pathMode) pathMode = true;
                    playing = !playing;
                    if (playing && pathPos >= pathCount - 1 - 1e-9) pathPos = 0.0;   // restart from the top
                    changed = true;
                }
                // Timeline scrub/jump (panel trackbar): engage path lock, pause, seek.
                if (pathCount >= 2 && nav.scrubTo >= 0) {
                    pathMode = true; playing = false;
                    pathPos = clampPos((double)nav.scrubTo);
                    changed = true;
                }

                // Ctrl+wheel adjusts the STEP SIZE (up = bigger), clamped to a sane band.
                if (nav.wheelSpeed != 0.0) {
                    step = std::clamp(step * std::pow(1.15, nav.wheelSpeed), sceneR * 1e-3, sceneR * 2.0);
                    std::printf("[viewer] step %.3g u\n", step); std::fflush(stdout);
                }
                // C cycles the collision response: slide -> stop -> off -> slide.
                if (nav.cycleCollide) {
                    collide = (CollideMode)((collide + 1) % 3);
                    std::printf("[viewer] collision: %s\n", collideName(collide)); std::fflush(stdout);
                }
                // T cycles the lit preview: raster -> mode W -> path-traced -> raster.
                // The GPU path-trace stage is SKIPPED (not just refused) when unavailable, so
                // on a CPU-only box or an out-of-scope scene T is a plain raster<->W toggle
                // rather than a key that prints an error and does nothing.
                if (nav.toggleTrace) {
                    if (pvMode == PV_RASTER)       pvMode = PV_WHITTED;
                    else if (pvMode == PV_WHITTED) pvMode = traceAvail ? PV_PT : PV_RASTER;
                    else                           pvMode = PV_RASTER;
                    traceDirty = true;   // re-render / restart accumulation at the current pose
                    changed = true;      // repaint immediately (raster if off; re-aim if on)
                    std::printf("[viewer] preview: %s\n",
                                pvMode == PV_RASTER  ? "raster (flat, instant)"
                              : pvMode == PV_WHITTED ? "mode W (deterministic, CPU)"
                                                     : "path-traced (fast RGB backward, GPU)");
                    std::fflush(stdout);
                }
                // Reset is the reliable "put me back to a normal, steerable state" escape:
                // ALWAYS return to free flight at the authored pose. Previously, resetting
                // while locked to the path only rewound the timeline but LEFT you locked —
                // with mouse-look suspended — so a user who got locked (e.g. by an accidental
                // click on the timeline slider, which snaps+locks onto the path) was stuck:
                // the view wouldn't steer and Reset didn't help. Now Reset also RELEASES the
                // path lock (and stops playback), so it dependably restores free-flight look.
                // Rewinding-while-locked is still available via the timeline / Play-from-top.
                if (nav.reset) {
                    if (pathMode) {
                        pathMode = false; playing = false;
                        std::printf("[viewer] path lock OFF (reset -> free flight)\n"); std::fflush(stdout);
                    }
                    pathPos = 0.0;
                    eye = eye0; fwd = norml(tgt0 - eye0);
                    lookDist = std::sqrt(dot(tgt0 - eye0, tgt0 - eye0));
                    if (lookDist < 1e-4) lookDist = sceneR;
                    changed = true;
                }

                // ---- Camera_curve EDITOR controls ---------------------------------
                // Author control points by hand or by recording the free flight, then Save a
                // camera_curve block. Every mutating action regenerates the preview path.
                if (nav.simplifyTol >= 0.0) recTol = nav.simplifyTol;   // panel tolerance box
                recRaw = nav.rawRecord;                                 // "raw" checkbox
                if (nav.recToggle) {
                    recording = !recording;
                    if (recording) { recRawBuf.clear(); haveRecPos = false;
                        std::printf("[editor] recording flythrough (fly around; press Rec again to stop)\n");
                    } else {
                        std::vector<PathFrame> got = (!recRaw && recTol > 0.0) ? simplify(recRawBuf, recTol) : recRawBuf;
                        for (const auto& g : got) { editPts.push_back(g); trackInsert(editPts.size() - 1); }
                        rebuildPath();
                        std::printf("[editor] recorded %zu control points from %zu raw samples (tol %.4g, %s)\n",
                                    got.size(), recRawBuf.size(), recTol, recRaw ? "raw" : "simplified");
                    }
                    g_liveWin->setEditState(recording, (int)editPts.size());
                    std::fflush(stdout); changed = true;
                }
                if (nav.addPoint) {
                    editPts.push_back(poseNow()); trackInsert(editPts.size() - 1);
                    rebuildPath();
                    g_liveWin->setEditState(recording, (int)editPts.size());
                    std::printf("[editor] +point %zu at eye(%s)\n", editPts.size(), fmt3(eye).c_str());
                    std::fflush(stdout); changed = true;
                }
                if (nav.insPoint) {
                    if (editPts.size() < 2) { editPts.push_back(poseNow()); trackInsert(editPts.size() - 1); }
                    else {
                        // Insert between the two control points bracketing the current scrub
                        // position. bracket() normalizes by the ACTUAL explorePath length, so this
                        // is correct for a freshly-loaded curve (whose frame count isn't a multiple
                        // of kPreviewPerSeg) as well as an editor-rebuilt preview.
                        int seg; double fr; bracket(pathPos, seg, fr);
                        seg = std::clamp(seg, 0, (int)editPts.size() - 2);
                        editPts.insert(editPts.begin() + seg + 1, poseNow());
                        trackInsert((size_t)seg + 1);
                    }
                    rebuildPath();
                    g_liveWin->setEditState(recording, (int)editPts.size());
                    std::printf("[editor] inserted point (now %zu)\n", editPts.size());
                    std::fflush(stdout); changed = true;
                }
                if (nav.delPoint && !editPts.empty()) {
                    int best = selectedPoint();   // the highlighted (selected) point — scrub to choose it
                    if (best < 0) best = 0;
                    editPts.erase(editPts.begin() + best);
                    trackErase((size_t)best);
                    if (pathPos > std::max(0, pathCount - 1)) pathPos = std::max(0, pathCount - 1);
                    rebuildPath();
                    pathPos = clampPos(pathPos);
                    g_liveWin->setEditState(recording, (int)editPts.size());
                    std::printf("[editor] deleted selected point %d (now %zu)\n", best, editPts.size());
                    std::fflush(stdout); changed = true;
                }
                if (nav.saveCurve) saveCurveFn();
                // "Flat" button: reset the painted speed track back to a uniform pace.
                if (nav.speedReset) {
                    std::fill(ptSpeed.begin(), ptSpeed.end(), 1.0);
                    std::printf("[editor] speed reset to flat (1.00x everywhere)\n"); std::fflush(stdout);
                    changed = true;
                }
                // ---- loom bind row: retarget/unbind a channel, or resize the drive ----------
                // Only reachable when the row exists (it is built only for a live -loom editor),
                // so these are inert in a sidecar-only session.
                if (animActive) {
                    if (nav.bindApply) animBind(nav.bindChannel, nav.bindTarget);
                    if (nav.bindClear) animBind(nav.bindChannel, std::string());
                    // The box reports its CURRENT value every drain, so act only on a real change.
                    if (nav.dimsReq >= 1 && nav.dimsReq != animDims) animSetDims(nav.dimsReq);
                }

                // The render camera's up vector and fov: fixed authored values while flying
                // free; the current path frame's own up/fov while locked to the path.
                Vec3   rUp  = worldUp;
                double rFov = fovY;

                if (!pathMode) {
                    // ---- FREE FLIGHT --------------------------------------------------
                    // Accumulate this frame's translation from all sources (plain-wheel dolly +
                    // held throttle), then resolve it ONCE against the scene so collision (and its
                    // slide) sees the true combined motion. Plain wheel DOLLIES one `step` per notch
                    // along the view ray (up = forward); held keys advance one `step`/frame.
                    Vec3 moveDelta{0, 0, 0};
                    if (nav.wheel != 0.0) moveDelta = moveDelta + fwd * (step * kWheelDolly * nav.wheel);
                    // Mouse-look STEERS at a RATE set by how far the cursor sits from the window
                    // centre (joystick/hover-look): each rendered frame turns by that offset x the
                    // max rate, so the view keeps turning while you hold the pointer off-centre and
                    // holds still in the central dead zone (where you can see the scene). Horizontal
                    // offset yaws about world up, vertical offset pitches about the camera right
                    // axis, pitch clamped shy of the poles so the view can't flip over (no roll).
                    // Per-frame (feedback-locked): a heavy scene turns in careful steps you actually
                    // see rather than spinning past.
                    if (nav.lookX != 0.0 || nav.lookY != 0.0) {
                        double yaw   = -nav.lookX * kYaw   * dt;   // pointer right -> turn right (rad/sec x dt)
                        double pitch = -nav.lookY * kPitch * dt;   // pointer down  -> look down  (rad/sec x dt)
                        fwd = norml(rotAxis(fwd, worldUp, yaw));
                        Vec3 right = cross(fwd, worldUp);
                        double rl = std::sqrt(dot(right, right));
                        if (rl > 1e-9) {
                            right = right * (1.0 / rl);
                            Vec3 cand = norml(rotAxis(fwd, right, pitch));
                            if (std::fabs(dot(cand, worldUp)) < 0.9995) fwd = cand;   // clamp near poles
                        }
                        changed = true;
                    }
                    // Held throttle: advance ONE `step` per RENDERED frame while Space/+ (forward)
                    // or Shift/- (backward) is down. Deliberately NOT wall-clock-integrated —
                    // tying the move to the render cadence means every position you pass through
                    // is actually drawn, so a slow scene can't fling you through a wall between two
                    // frames you never saw. Travel rate = step x render-fps (faster scene = quicker).
                    if (nav.fwd)  moveDelta = moveDelta + fwd * step;
                    if (nav.back) moveDelta = moveDelta - fwd * step;
                    // Apply the combined move through collision (no-op when collision is OFF).
                    if (dot(moveDelta, moveDelta) > 0.0) { eye = resolveMove(eye, moveDelta); changed = true; }
                } else {
                    // ---- LOCKED TO THE CAMERA PATH ------------------------------------
                    // Travel is along the timeline (camera index), not through free space, and
                    // the orientation/up/fov come straight from the path frames. Forward/back
                    // (or Play auto-advance) move the cursor; the two speed modes decide how far
                    // per frame: rate mode = cameras/second on the wall clock (may skip frames on
                    // a slow render); stride mode = a fixed number of cameras per RENDERED frame.
                    // PAINT mode (panel "Paint"): the plain wheel PAINTS local traversal speed
                    // at the scrub position (additive brush) and mouse-look STEERS the nearest
                    // control points' orientation — authoring the density_at + look curve live.
                    // Outside paint mode the wheel nudges and mouse-look is suspended (as before).
                    bool wheelPainted = false;
                    if (nav.paintMode) {
                        if (nav.wheel != 0.0 && !editPts.empty()) { paintSpeed(pathPos, nav.wheel); wheelPainted = true; changed = true; }
                        if ((nav.lookX != 0.0 || nav.lookY != 0.0) && !editPts.empty()) { paintOrient(pathPos, nav.lookX, nav.lookY); changed = true; }
                    }
                    double dir = 0.0;
                    if (playing)  dir += 1.0;
                    if (nav.fwd)  dir += 1.0;
                    if (nav.back) dir -= 1.0;
                    double advance = 0.0;
                    if (dir != 0.0)
                        advance = (rateMode ? (dir * camPerSec * dt) : (dir * (double)strideN)) * speedAt(pathPos);
                    if (!wheelPainted) advance += nav.wheel;   // plain wheel nudges one camera per notch (precise)
                    if (advance != 0.0) {
                        double np = clampPos(pathPos + advance);
                        if (np != pathPos) { pathPos = np; changed = true; }
                        // Auto-play stops when it runs off either end of the timeline.
                        if (playing && dir > 0.0 && pathPos >= pathCount - 1 - 1e-9) playing = false;
                        if (playing && dir < 0.0 && pathPos <= 1e-9)                 playing = false;
                    }
                    int i = (int)std::lround(pathPos);
                    eye = explorePath[i].eye; fwd = explorePath[i].fwd;
                    rUp = explorePath[i].up;  rFov = explorePath[i].fov;
                }

                // Recording sampler: while Rec is armed and we're flying free, capture the
                // pose whenever the eye has moved a small min-distance since the last sample
                // (distance-gated so a stationary pause never spams the buffer). These raw
                // samples become control points — optionally RDP-simplified — when Rec stops.
                if (recording && !pathMode) {
                    bool take = !haveRecPos;
                    if (haveRecPos) { Vec3 d = eye - lastRecPos; take = dot(d, d) >= (sceneR * 0.008) * (sceneR * 0.008); }
                    if (take) { recRawBuf.push_back(poseNow()); lastRecPos = eye; haveRecPos = true; }
                }

                // Ask loom for the frame at the current curve position. Latest-wins, so a
                // fast drag leaves at most one emit in flight and one waiting; the
                // positions swept through in between are dropped rather than queued into
                // a backlog the user would have to sit through. Only post when the
                // position (or the curve itself) actually changed — an idle editor must
                // not spin loom re-emitting the same frame forever.
                if (animLive && pathCount >= 2) {
                    double denom = (double)(pathCount - 1);
                    double t = (denom > 0.0) ? std::clamp(pathPos / denom, 0.0, 1.0) : 0.0;
                    // A reshaped curve means the frame we are showing was emitted against
                    // points that no longer exist, so re-ask even if the position is
                    // unchanged (t is in [0,1]; -1 can never compare equal).
                    if (animPtsDirty) { animSendPoints(); animLastT = -1.0; }
                    if (t != animLastT) {
                        animlive::Job j;
                        j.t      = t;
                        j.frame  = (int)std::lround(pathPos);
                        j.frames = pathCount;
                        animBridge.post(j);
                        animLastT = t;
                    }
                }

                Vec3 tgt = eye + fwd * lookDist;   // look_at point on the view ray (for readout/print)
                // A real, display-changing frame vs a GPU keep-warm-only frame. `changed`
                // is set by any actual input; a keep-warm frame renders solely to hold the
                // boost clock during the grace window and never touches the window/overlay.
                bool active = changed || nav.any() || playing;
                if (active) lastActiveT = clock::now();
                double idleFor = std::chrono::duration<double>(clock::now() - lastActiveT).count();
                // Warm-only frame: hold the boost clock, but ONLY during a genuine pause
                // (idleFor past the gap) and never during an active drag — so it can't steal
                // the slot that samples the next scrub position (the timeline-"chunking" bug).
                bool warmOnly = gpuWarmKeep && !changed &&
                                idleFor >= kWarmGapSec && idleFor < kWarmGraceSec;
                // `tracingNow` is set below when the path-traced preview is actively refining
                // this idle pose; it suppresses the raster warm-frame and the idle sleep so the
                // image keeps converging (the accumulate() launch already holds the GPU warm).
                bool tracingNow = false;
                if (pvMode == PV_WHITTED) {
                    Camera c; c.projection = proj;
                    c.lookAt(eye, tgt, rUp, rFov, VW, VH);
                    if (wResX != VW || wResY != VH) {          // first use / window resize
                        wFilm.resX = VW; wFilm.resY = VH; wFilm.alloc();
                        wResX = VW; wResY = VH;
                        traceDirty = true;
                    }
                    if (changed) {
                        // Camera moved: show the responsive raster and drop the stale rows.
                        if (!rasterPresent(c, VW, VH, ev, autoExp)) {
                            std::vector<uint8_t> img = rasterOne(c, VW, VH, ev, autoExp, nullptr);
                            drawOverlay(c, VW, VH, img);
                            g_liveWin->update(VW, VH, img);
                        }
                        g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  eye(" + fmt3(eye) +
                                            ")  dir(" + fmt3(fwd) + ")  [mode W: stop to render]");
                        traceDirty = true;
                        changed = false;
                    } else if (traceDirty) {
                        // First still frame at a new pose: a COARSE full-frame mode-W pass.
                        // It lands almost immediately (1/256 the pixels) so there is a real lit
                        // image to look at at once, and — the reason it is full-frame rather
                        // than just the first band — its p99 gives a globally representative
                        // auto-exposure anchor. Anchoring on band 0 instead would expose the
                        // whole frame off one strip of it and blow out everything that follows.
                        const int cw = std::max(1, VW / kWCoarse), ch = std::max(1, VH / kWCoarse);
                        Camera cc; cc.projection = proj;
                        cc.lookAt(eye, tgt, rUp, rFov, cw, ch);
                        Film cf = renderBackward(scene, cc, cw, ch, 1, nThreads, /*diffraction*/false,
                                                 0, /*forceWhitted*/true);
                        traceAnchor = 0.0;                     // recompute the anchor for this pose
                        std::vector<uint8_t> small =
                            filmToRgb8(cf, 1.0, ev, scene.absolute, &traceAnchor);
                        wImg.assign((size_t)VW * VH * 3, 0);   // nearest-neighbour up to full size
                        for (int y = 0; y < VH; ++y) {
                            const int sy = std::min(ch - 1, y * ch / VH);
                            for (int x = 0; x < VW; ++x) {
                                const int sx = std::min(cw - 1, x * cw / VW);
                                std::memcpy(&wImg[((size_t)y * VW + x) * 3],
                                            &small[((size_t)sy * cw + sx) * 3], 3);
                            }
                        }
                        std::vector<uint8_t> show = wImg;
                        drawOverlay(c, VW, VH, show);
                        g_liveWin->update(VW, VH, show);
                        // wFilm ACCUMULATES across passes, so it has to be cleared for the new pose.
                        std::fill(wFilm.xyz.begin(), wFilm.xyz.end(), Vec3(0.0, 0.0, 0.0));
                        wRow = VH;                             // now refine full-res, top band first
                        wBandRows = 16;
                        wPass = 0;
                        traceDirty = false;
                        tracingNow = true;
                    } else if (wRow > 0) {
                        // Refine one band. Film row 0 is the image BOTTOM (filmToRgb8 flips), so
                        // taking bands off the high end of the film fills the picture downwards.
                        const int y1 = wRow, y0 = std::max(0, wRow - wBandRows);
                        auto t0 = clock::now();
                        renderBackward(scene, c, VW, VH, 1, nThreads, /*diffraction*/false,
                                       /*sampleBase*/(unsigned long long)wPass,
                                       /*forceWhitted*/true, &wFilm, y0, y1);
                        double secs = std::chrono::duration<double>(clock::now() - t0).count();
                        // Tone-map JUST this band, with the pose's locked anchor, and splice it
                        // into the shown image over the coarse pixels it replaces. N is the pass
                        // count these rows have received, not the frame's -- rows further down are
                        // still one pass behind until the sweep reaches them.
                        Film bf; bf.resX = VW; bf.resY = y1 - y0; bf.alloc();
                        std::memcpy(bf.xyz.data(), &wFilm.xyz[(size_t)y0 * VW],
                                    sizeof(Vec3) * (size_t)VW * (y1 - y0));
                        std::vector<uint8_t> bimg =
                            filmToRgb8(bf, (double)(wPass + 1), ev, scene.absolute, &traceAnchor);
                        for (int r = 0; r < y1 - y0; ++r) {
                            // bimg row r is film row (y1-1-r); the shown image has film row f at
                            // image row VH-1-f.
                            const int dst = VH - 1 - (y1 - 1 - r);
                            std::memcpy(&wImg[(size_t)dst * VW * 3], &bimg[(size_t)r * VW * 3],
                                        (size_t)VW * 3);
                        }
                        wRow = y0;
                        // Retune for the next band. Clamped below at 1 row (a heavy scene must
                        // still make progress) and above at 1/4 of the frame (so an easy scene
                        // does not swallow the whole image in one unresponsive gulp).
                        if (secs > 1e-4) {
                            double scale = kWBandSec / secs;
                            wBandRows = (int)std::clamp((double)wBandRows * scale, 1.0,
                                                        std::max(1.0, VH / 4.0));
                        }
                        // Pass complete. A bundle-only scene is EXACT here, so stop; otherwise
                        // start the next pass to fill in the spectrum (see wNeedSpp).
                        if (wRow == 0 && wNeedSpp && wPass + 1 < kWSppCap) {
                            ++wPass;
                            wRow = VH;
                        }
                        std::vector<uint8_t> show = wImg;
                        drawOverlay(c, VW, VH, show);
                        g_liveWin->update(VW, VH, show);
                        tracingNow = (wRow > 0);   // keep spinning until the frame is complete
                        const std::string sppTag = wNeedSpp ? " " + std::to_string(wPass + 1) + " spp" : "";
                        if (tracingNow)
                            g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  mode W" + sppTag + " " +
                                                std::to_string(100 * (VH - wRow) / std::max(1, VH)) +
                                                "%  eye(" + fmt3(eye) + ")");
                        else
                            g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  mode W" + sppTag +
                                                "  eye(" + fmt3(eye) + ")  dir(" + fmt3(fwd) + ")");
                    }
                }
#ifdef HAVE_CUDA
                if (pvMode == PV_PT && traceAvail) {
                    Camera c; c.projection = proj;
                    c.lookAt(eye, tgt, rUp, rFov, VW, VH);
                    // (Re)create the resident session on first use or after a resize.
                    if (!traceSess || traceResX != VW || traceResY != VH) {
                        if (traceSess) backwardRGBSessionEnd(traceSess);
                        traceSess = backwardRGBSessionBegin(scene, VW, VH, g_maxBounceOverride, g_directOnly);
                        traceResX = VW; traceResY = VH;
                        traceFilm.resX = VW; traceFilm.resY = VH; traceFilm.alloc();
                        traceDirty = true;
                    }
                    if (changed) {
                        // Camera moved: show the responsive raster and mark the trace stale.
                        if (!rasterPresent(c, VW, VH, ev, autoExp)) {
                            std::vector<uint8_t> img = rasterOne(c, VW, VH, ev, autoExp, nullptr);
                            drawOverlay(c, VW, VH, img);
                            g_liveWin->update(VW, VH, img);
                        }
                        g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  eye(" + fmt3(eye) +
                                            ")  dir(" + fmt3(fwd) + ")  [trace: move to re-aim]");
                        traceDirty = true;
                        changed = false;
                    } else if (traceSess) {
                        // Idle: re-aim on the first still frame after a move, then keep adding
                        // sample batches until the pose is well converged.
                        if (traceDirty) {
                            backwardRGBSessionSetCamera(traceSess, c);
                            traceAnchor = 0.0;   // re-lock auto-exposure for the new pose
                            traceDirty = false;
                        }
                        if (backwardRGBSessionSamples(traceSess) < kTraceCapSpp) {
                            long long spp = backwardRGBSessionAccumulate(traceSess, kTraceBatchSpp, /*diffraction*/false);
                            backwardRGBSessionDownload(traceSess, traceFilm);
                            std::vector<uint8_t> img = filmToRgb8(traceFilm, (double)spp, ev,
                                                                  scene.absolute, &traceAnchor);
                            drawOverlay(c, VW, VH, img);
                            g_liveWin->update(VW, VH, img);
                            g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  path-trace " +
                                                std::to_string(spp) + " spp  eye(" + fmt3(eye) + ")");
                            tracingNow = (spp < kTraceCapSpp);   // more to refine -> keep spinning
                        }
                    }
                }
#endif
                if (pvMode == PV_RASTER && (changed || warmOnly)) {
                    Camera c; c.projection = proj;
                    c.lookAt(eye, tgt, rUp, rFov, VW, VH);
                    // A warm-only frame renders solely to hold the boost clock and must NOT
                    // repaint, so the zero-copy present (which renders AND shows) is for real
                    // changes only; the warm frame keeps taking the ordinary render path.
                    if (changed && rasterPresent(c, VW, VH, ev, autoExp)) {
                        g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  eye(" + fmt3(eye) +
                                            ")  dir(" + fmt3(fwd) + ")");
                    } else {
                        std::vector<uint8_t> img =
                            rasterOne(c, VW, VH, ev, autoExp, nullptr);
                        if (changed) {   // only a real change repaints the window
                            drawOverlay(c, VW, VH, img);   // control-point markers + live spline polyline
                            g_liveWin->update(VW, VH, img);
                            g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  eye(" + fmt3(eye) +
                                                ")  dir(" + fmt3(fwd) + ")");
                        }
                    }
                    changed = false;
                }
                if (nav.print) {
                    std::printf("camera \"cam\" { eye %.4g %.4g %.4g   look_at %.4g %.4g %.4g"
                                "   up %.4g %.4g %.4g   fov_y %.4g }\n",
                                eye.x, eye.y, eye.z, tgt.x, tgt.y, tgt.z,
                                rUp.x, rUp.y, rUp.z, rFov);
                    std::fflush(stdout);
                }
                // Mirror the live viewer state back onto the panel controls (timeline slider,
                // Play/Pause label, Path toggle, Clip label) whenever they change, so the panel
                // always reflects reality — e.g. the slider tracks playback and the toggles
                // follow keyboard/auto changes. setPanelState never re-emits a NavInput edge.
                {
                    int idxNow = pathMode ? (int)std::lround(pathPos)
                                          : (lastIdxSent < 0 ? 0 : lastIdxSent);
                    if (idxNow != lastIdxSent || playing != lastPlaySent ||
                        pathMode != lastPathSent || collide != lastCollideSent) {
                        g_liveWin->setPanelState(idxNow, playing, pathMode, collideShort(collide));
                        lastIdxSent = idxNow; lastPlaySent = playing;
                        lastPathSent = pathMode; lastCollideSent = collide;
                    }
                    // Mirror the painted local-speed multiplier at the current scrub position.
                    if (pathMode) {
                        double sp = speedAt(pathPos);
                        if (std::fabs(sp - lastSpdSent) > 5e-3) { g_liveWin->setSpeedLabel(sp); lastSpdSent = sp; }
                    }
                    // Mirror the loom bind row: what the selected channel drives, plus the live
                    // channel's health. Recomputed and diffed rather than pushed on every event,
                    // because the SELECTION also changes it and a combo pick raises no edge here.
                    if (animActive) {
                        std::string st;
                        if (!animLive)
                            st = animLastErr.empty() ? "offline (sidecar only)" : "offline: " + animLastErr;
                        else if (!animLastErr.empty())
                            st = animLastErr;
                        else {
                            char b[96];
                            // Explicit UTF-8 bytes, not \u2014: a narrow literal escape would be
                            // transcoded to cp1252 (C4566) and land in the panel as junk.
                            std::snprintf(b, sizeof b, "live \xE2\x80\x94 %lld baked, %.0f ms",
                                          animBaked, animLastMs);
                            st = b;
                        }
                        std::vector<std::string> tg = animTargets();
                        // The row also reports the SELECTED channel's binding, and picking a
                        // channel raises no edge out here — so the selection is part of the diff.
                        if (st != animLastStatus || tg != animLastTargets ||
                            nav.bindChannel != animLastBindCh) {
                            g_liveWin->setBindState(tg, st.c_str());
                            animLastStatus = st; animLastTargets = tg;
                            animLastBindCh = nav.bindChannel;
                        }
                    }
                }
                // Sleep policy. While a throttle key is held, the mouse is steering, or the
                // path is auto-playing we loop at full raster speed for smooth motion. A
                // warm-only frame already ran the GPU this iteration (continuous during a
                // pause to hold the clock) so it never sleeps. Otherwise: inside the grace
                // window we take only a SHORT 3 ms nap — short enough that the next scrub/
                // drag event drains promptly (the timeline tracks the thumb without chunking)
                // yet not a busy spin; past the grace window we sleep the full idle interval
                // and let the card power down.
                if (!nav.any() && !playing && !warmOnly && !tracingNow) {
                    bool inGrace = gpuWarmKeep && idleFor < kWarmGraceSec;
                    std::this_thread::sleep_for(std::chrono::milliseconds(inGrace ? 3 : 15));
                }
            }
#ifdef HAVE_CUDA
            if (traceSess) backwardRGBSessionEnd(traceSess);   // free the resident preview session
#endif
            g_stopRequested = 1;   // window closed → done
        }
#ifdef HAVE_CUDA
        raster_cuda::destroy(gpuRaster);
#endif
        return 0;
    }

    // -----------------------------------------------------------------------------
    // Exposure-lock meter pre-pass (REAL render). For every locked group being
    // rendered, meter its selector-chosen frame(s) with a quick *reduced-sample* CPU
    // render and pre-populate expAnchors[group] (averaging for EXPLOCK_AVERAGE) BEFORE
    // any full frame runs. Because filmToRgb8's p99 anchor is sample-count-invariant
    // (norm = 1/(N*cieYIntegral) cancels the photon/spp count), a cheap metering render
    // yields the same eAuto the full frame would — just noisier, and p99 is noise-robust.
    // Once expAnchors[g] > 0, every render path reuses it untouched (no per-frame
    // recompute, no dolly flicker), so the whole group locks to the chosen viewpoint's
    // exposure — the selector is ALWAYS honoured (there is no silent frame-0 fallback).
    // Skipped for absolute-EV scenes and forced global locks (meterPlan is empty then).
    //
    // The anchor is a measure of scene brightness AT the viewpoint — a property of the
    // radiance, not of the integrator — so every render mode yields the same value in
    // expectation. We meter each frame in its OWN mode where a cheap pass exists
    // (A/B/C forward, R backward, D BDPT, M photon-map, P composite) and fall back to a
    // general forward mode-B light-trace for anything else (S/U/V/…, which still converge
    // to the same radiance). One reduced, view-independent photon map (built lazily once)
    // serves every mode-M meter.
    PhotonMap meterPmap; bool meterPmapBuilt = false;
#ifdef HAVE_CUDA
    // Meter on the device the run asked for. The meter is a REAL reduced render, so when
    // a mode's GPU path supports this scene it must use it: metering on the CPU while the
    // user asked for -device gpu used to front-load the whole pre-pass as silent CPU work
    // — tens of minutes on a big scene (the mode-M "shared deposit hang" in known-issues
    // was exactly this meter, misdiagnosed as the GPU build). Every branch below gates on
    // the same support predicate its real render uses and falls back to the CPU renderer
    // otherwise, so -device cpu runs are bit-identical to before.
    const bool meterGpu = (!std::strcmp(device, "gpu") || !std::strcmp(device, "auto")) &&
                          cudaAvailable();
#else
    const bool meterGpu = false;
#endif
    auto meterAnchor = [&](const MeterCam& mc) -> double {
        const int W = mc.res, H = mc.resY;
        // Reduced budgets: enough coverage for a clean p99 without paying for a full render.
        const long long meterN   = std::clamp((long long)W * H * 40LL, 500000LL, 4000000LL);
        const long long meterSpp = 16;
        char mode = mc.mode;
        // A scene outside BDPT's transport scope can't meter in mode D (the real render
        // will itself refuse it later, loudly) — meter it with the general forward pass.
        if (mode == 'D' && bdptUnsupportedFeature(scene)) mode = 'B';
        Film mf; double eAuto = 0.0;
        switch (mode) {
            case 'A': case 'B': case 'C': {
                EnergyReport e;
                // renderForward self-gates on cudaForwardSupported and falls back to CPU.
                mf = renderForward(scene, &mc.cam, W, H, meterN, nThreads,
                                   /*forwardCatch*/mode == 'C', /*lensMode*/mode == 'A',
                                   /*useCamera*/true, e, diffraction, /*useGpu*/meterGpu);
                addEnvBackground(mf, scene, mc.cam, meterN);
                filmToRgb8(mf, (double)meterN, 1.0, false, nullptr, &eAuto);
                break;
            }
            case 'R': {
                bool onGpu = false;
#ifdef HAVE_CUDA
                if (meterGpu && cudaBackwardSupported(scene, mc.cam)) {
                    mf = renderBackwardCuda(scene, mc.cam, W, H, meterSpp, diffraction, nullptr,
                                            g_maxBounceOverride, g_directOnly, g_heroC);
                    onGpu = true;
                }
#endif
                if (!onGpu)
                    mf = renderBackward(scene, mc.cam, W, H, meterSpp, nThreads, diffraction);
                filmToRgb8(mf, (double)meterSpp, 1.0, false, nullptr, &eAuto);
                break;
            }
            case 'D': {
                bool onGpu = false;
#ifdef HAVE_CUDA
                if (meterGpu && cudaBdptSupported(scene)) {
                    mf = renderBdptCuda(scene, mc.cam, W, H, meterSpp, /*maxDepth*/8, diffraction);
                    onGpu = true;
                }
#endif
                if (!onGpu)
                    mf = renderBdpt(scene, mc.cam, W, H, meterSpp, nThreads, /*maxDepth*/8, diffraction);
                filmToRgb8(mf, (double)meterSpp, 1.0, false, nullptr, &eAuto);
                break;
            }
            case 'M': {
                // CPU fallback only: mode-M groups that pass the GPU gates are metered in
                // ONE batched renderPhotonMapSharedCuda call in the group loop below
                // (shared device map + GPU gathers), never per-frame here.
                if (!meterPmapBuilt) {
                    double radius = (g_pmRadiusAbs > 0.0) ? g_pmRadiusAbs
                                                          : scene.sceneRadius * g_pmRadiusFactor;
                    tracePhotonPass(scene, meterN, nThreads, diffraction, meterPmap, g_heroC);
                    buildPhotonMap(meterPmap, radius, "[meter]");
                    meterPmapBuilt = true;
                }
                mf = renderPhotonCamera(scene, mc.cam, W, H, meterPmap, meterSpp, nThreads,
                                        diffraction, /*maxBounce*/32, 0, g_pmFinalGather);
                filmToRgb8(mf, (double)meterSpp, 1.0, false, nullptr, &eAuto);
                break;
            }
            case 'P': {
                // Composite = forward (model B) + backward, combined at radiance scale 1.0.
                CompositeClass cc = classifyComposite(scene, mc.cam, W, H);
                EnergyReport e;
                Film fwd = renderForward(scene, &mc.cam, W, H, meterN, nThreads,
                                         false, false, true, e, diffraction, meterGpu);
                Film ref;
                bool refGpu = false;
#ifdef HAVE_CUDA
                if (meterGpu && cudaBackwardSupported(scene, mc.cam)) {
                    ref = renderBackwardCuda(scene, mc.cam, W, H, meterSpp, diffraction, nullptr,
                                             g_maxBounceOverride, g_directOnly, g_heroC);
                    refGpu = true;
                }
#endif
                if (!refGpu)
                    ref = renderBackward(scene, mc.cam, W, H, meterSpp, nThreads, diffraction);
                mf = compositeFromFilms(fwd, meterN, ref, meterSpp, cc,
                                        scene.envIndex >= 0, /*verbose*/false);
                filmToRgb8(mf, 1.0, 1.0, false, nullptr, &eAuto);
                break;
            }
            default: {   // S/U/V and any future mode: general forward mode-B light-trace
                EnergyReport e;
                mf = renderForward(scene, &mc.cam, W, H, meterN, nThreads,
                                   false, false, true, e, diffraction, meterGpu);
                addEnvBackground(mf, scene, mc.cam, meterN);
                filmToRgb8(mf, (double)meterN, 1.0, false, nullptr, &eAuto);
                break;
            }
        }
        return eAuto;
    };
    for (const auto& [g, cams] : meterPlan) {
        if (cams.empty() || g_stopRequested) continue;
        const bool adaptive = meterAdaptive.count(g) != 0;
        const int  N   = (int)cams.size();
        const int  kmx = adaptive ? std::min(N, kMeterMax) : N;
        const int  kmn = adaptive ? std::min(N, kMeterMin) : N;
        MeterConverge conv(kmn, kmx, kMeterTolStops);
        bool metered = false;
#ifdef HAVE_CUDA
        // Batched GPU meter for a mode-M group: ONE device photon map + GPU gathers for
        // the (up to kmx) meter frames, early-stopped by the same convergence test via
        // the shared path's per-frame onFrame hook. Per-frame metering can't reuse a
        // device map across meterAnchor calls, and the CPU version of this (one CPU map
        // + up to kMeterMax full-res CPU gathers) is the pre-pass that used to take tens
        // of minutes while the GPU idled. Gated exactly like runSharedPhotonMap's GPU
        // branch; any miss falls through to the per-frame loop below unchanged.
        if (meterGpu && cudaPhotonMapSupported(scene)) {
            bool allM = true, allPinhole = true;
            for (const auto& mc : cams) {
                if (mc.mode != 'M')    allM = false;
                if (mc.cam.hasLens()) allPinhole = false;
            }
            if (allM && allPinhole) {
                std::vector<Camera> mcams; std::vector<int> rxs, rys;
                for (int i = 0; i < kmx; ++i) {
                    mcams.push_back(cams[i].cam);
                    rxs.push_back(cams[i].res); rys.push_back(cams[i].resY);
                }
                const long long meterN = std::clamp((long long)cams[0].res * cams[0].resY * 40LL,
                                                    500000LL, 4000000LL);
                const long long meterSpp = 16;   // matches meterAnchor's reduced budget
                double radius = (g_pmRadiusAbs > 0.0) ? g_pmRadiusAbs
                                                      : scene.sceneRadius * g_pmRadiusFactor;
                std::printf("[meter] exposure lock: group %d metering on %s (shared "
                            "photon map, up to %d frame(s)) ...\n",
                            g, cudaDeviceName(), kmx);
                std::fflush(stdout);
                EnergyReport e;
                std::function<bool(int, const Film&)> onFrame =
                    [&](int, const Film& f) -> bool {
                        double eAuto = 0.0;
                        filmToRgb8(f, (double)meterSpp, 1.0, false, nullptr, &eAuto);
                        return conv.add(eAuto) || g_stopRequested != 0;
                    };
                renderPhotonMapSharedCuda(scene, mcams, rxs, rys, meterN, radius, e,
                                          diffraction, meterSpp, nullptr, &onFrame,
                                          nullptr, nullptr, g_heroC, g_pmFinalGather,
                                          g_pmAutoRadius ? g_pmAutoCount : 0.0);
                metered = true;   // a black meter falls into the no-anchor warning below
            }
        }
#endif
        if (!metered)
            for (const auto& mc : cams) {
                if (conv.add(meterAnchor(mc))) break;   // adaptive early-stop once converged
            }
        if (conv.used() > 0) {
            expAnchors[g] = conv.anchor();
            if (adaptive)
                std::printf("[meter] exposure lock: group %d meters the average of %d/%d frame(s) "
                            "(anchor %.4g)\n", g, conv.used(), N, expAnchors[g]);
            else
                std::printf("[meter] exposure lock: group %d meters '%s' (anchor %.4g)\n",
                            g, cams.front().name.c_str(), expAnchors[g]);
        } else {
            std::fprintf(stderr, "[meter] exposure lock: group %d produced no valid anchor "
                         "(all-black meter?); its frames will meter individually\n", g);
        }
        std::fflush(stdout);
    }

    // Shared multi-camera forward pass. When several plain-`-n` forward cameras of the
    // same camera model render at once, trace ONE photon set and splat every vertex to
    // all of them (renderForwardShared / renderForwardSharedCuda) instead of re-tracing
    // per camera — the "many cameras for 1x photon work" win. It applies to the two
    // forward next-event models:
    //   * model B (pinhole splat): connect() draws no RNG, so a shared pass is
    //     bit-identical to per-camera renders.
    //   * model A (finite-lens physical camera): connectLens() samples each camera's own
    //     pupil (draws RNG), so the shared photon flight is un-biased per camera but
    //     matches a standalone render in distribution, not bit-for-bit. Rectilinear only
    //     (the thin-lens model can't form a fisheye — see the mode A/C guard in runRender).
    // Model C consumes the photon at the first aperture it hits, so it can't be shared.
    // The A- and B-groups are SEPARATE passes: mode A perturbs the RNG stream during the
    // trace and mode B does not, so their photon paths diverge and can't ride one flight.
    // Both groups run on the GPU when the device/scene allow (renderForwardSharedCuda),
    // else on the CPU. Sharing applies only to per-frame-auto-exposed cameras (an
    // exposure-locked camera_path is an animation, better left un-shared so its frames
    // don't all carry the same fixed noise realisation).
    bool useGpuForward = false;
#ifdef HAVE_CUDA
    {
        const bool wantGpu  = !std::strcmp(device, "gpu");
        const bool wantAuto = !std::strcmp(device, "auto");
        if ((wantGpu || wantAuto) && cudaAvailable() && cudaForwardSupported(scene))
            useGpuForward = true;   // -beams per-camera resample is supported on the GPU too
    }
#endif
    (void)useGpuForward;   // only read under HAVE_CUDA; keep CPU-only builds warning-clean
    const bool plainRender = !(timeBudgetSec > 0.0 || noiseTarget > 0.0 || resume ||
                               wantCheckpointFlag || runForever || preview);
    std::vector<int> groupB, groupA, groupM, restIdx;
    for (int i = 0; i < (int)toRender.size(); ++i) {
        const RenderCam& rc = toRender[i];
        // Forward A/B sharing no longer requires `plainRender`: the shared pass itself
        // now chunks the photons, drives the live window, and writes a per-camera .ftbuf
        // so it is crash-safe / resumable / budgetable exactly like the single-camera
        // path (Feature B). Only the per-frame-auto-exposure requirement remains (an
        // exposure-locked camera_path animation is still rendered un-shared so its frames
        // don't all carry the same fixed noise realisation).
        bool base = (rc.expGroup < 0);
        if (base && rc.mode == 'B')                                           groupB.push_back(i);
        else if (base && rc.mode == 'A' && rc.cam.projection == CAM_RECTILINEAR) groupA.push_back(i);
        // Mode M (photon map): the map is view-INDEPENDENT, so build it once and gather
        // every camera from it — the flythrough win. Unlike A/B sharing (which reuses one
        // photon *flight* and so imposes the same fixed noise on every frame), the mode-M
        // gather is an independent backward pass per camera, so frames don't share noise —
        // only the underlying radiance solution. That makes it safe to share even across
        // exposure-locked camera_path frames, so it isn't gated on `expGroup < 0`.
        else if (rc.mode == 'M' && plainRender)                               groupM.push_back(i);
        else                                                                  restIdx.push_back(i);
    }
    // A single-camera forward group has nothing to share — fold it back into the per-camera
    // path (models A/B still get the GPU there via renderForwardCuda).
    if (groupB.size() < 2) { for (int i : groupB) restIdx.push_back(i); groupB.clear(); }
    if (groupA.size() < 2) { for (int i : groupA) restIdx.push_back(i); groupA.clear(); }
    // Mode M is different: the per-camera fallback is CPU-only, so the shared photon-map path
    // is the ONLY GPU route for mode M and it handles a single camera fine. Keep even one
    // plain mode-M camera here so `-camera #N`/`near=`/name can aim the live window at one
    // frame of a long camera_curve and still render it on the GPU.
    std::sort(restIdx.begin(), restIdx.end());

    bool sharedWriteFail = false;
    // Shared forward A/B pass with the SAME crash-safety machinery as the single-camera
    // path (Feature B): the group's ONE photon flight is traced in accumulation chunks,
    // each chunk seeded off the cumulative photon count so it draws independent photons;
    // every camera keeps its own SUM-film accumulator; and periodically we write each
    // camera's image plus a per-camera .ftbuf checkpoint so a crash/Ctrl-C loses at most
    // one interval and `-resume` continues from the saved films. This mirrors the forward
    // A/B/C loop in runRender, generalised to N cameras riding one shared flight.
    auto runSharedGroup = [&](const std::vector<int>& idx, char groupMode) {
        if (idx.empty() || g_stopRequested) return;
        g_windowMode = modeLabel(groupMode);   // title bar shows this shared group's mode
        const int nc = (int)idx.size();
        std::vector<Camera> cams; std::vector<int> rxs, rys;
        for (int i : idx) { cams.push_back(toRender[i].cam); rxs.push_back(toRender[i].res); rys.push_back(toRender[i].resY); }

        // Per-camera SUM-film accumulators sharing one photon count + energy tally (all
        // cameras see the same flight, so accN / energy are group-wide).
        std::vector<Film> acc(nc);
        for (int c = 0; c < nc; ++c) { acc[c].resX = rxs[c]; acc[c].resY = rys[c]; acc[c].alloc(); }
        long long accN = 0;
        EnergyReport accE;

        const bool progressive   = timeBudgetSec > 0.0 || runForever || noiseTarget > 0.0;
        const bool wantCheckpoint = resume || progressive || wantCheckpointFlag;

        // Resume: load every camera's sidecar. A shared flight can only resume as a whole
        // (all cameras must be at the same photon count), so any missing / mismatched /
        // inconsistent sidecar falls the whole group back to a fresh start.
        if (resume) {
            std::vector<Checkpoint> cks(nc);
            bool ok = true; long long n0 = -1;
            for (int c = 0; c < nc && ok; ++c) {
                uint64_t g = checkpointGuard(scene, groupMode, rxs[c], rys[c]);
                if (!readCheckpoint(outFor(toRender[idx[c]].name), rxs[c], rys[c], g, groupMode, cks[c])) ok = false;
                else if (n0 < 0) n0 = cks[c].N;
                else if (cks[c].N != n0) ok = false;
            }
            if (ok && n0 > 0) {
                for (int c = 0; c < nc; ++c) acc[c] = cks[c].film;
                accN = n0; accE = cks[0].energy;
                std::printf("[resume] loaded shared model-%c group (%d cameras): %lld photons accumulated so far\n",
                            groupMode, nc, accN);
            } else if (n0 > 0) {
                // Camera 0 loaded but a later camera was missing / mismatched: a shared
                // flight can't resume half-done, so drop it and start fresh. (acc[] was never
                // populated with the loaded films — that only happens in the success branch —
                // so it is still zero here; an all-missing set stays silent like single-cam.)
                std::fprintf(stderr, "[resume] shared model-%c group is inconsistent across cameras; starting fresh\n",
                             groupMode);
            }
        }

        const std::string backend =
#ifdef HAVE_CUDA
            useGpuForward ? std::string(cudaDeviceName()) :
#endif
            (std::to_string(nThreads) + " CPU threads");

#ifdef HAVE_CUDA
        // Resident GPU session for the whole group render: the scene and every camera are
        // baked/uploaded ONCE and the per-camera films accumulate on the device, so each
        // batch below is a bare kernel launch instead of a full upload/download round trip
        // (the old per-batch renderForwardSharedCuda wrapper dominated wall-clock on
        // multi-camera groups). Films/energy are downloaded only when the host actually
        // needs them (syncAcc: interval writes, window refresh, final). On -resume the
        // checkpoint films seed the device accumulators so downloads are full totals.
        SharedGpuSession* gses = useGpuForward
            ? sharedForwardGpuBegin(scene, cams, rxs, rys, groupMode, wavefront, g_heroC,
                                    g_beamGather, (accN > 0) ? &acc : nullptr,
                                    (accN > 0) ? &accE : nullptr)
            : nullptr;
        bool accFresh = true;   // do host acc[]/accE match the device accumulators?
#endif

        // One accumulation chunk of `batchN` photons across the whole group.
        auto runBatch = [&](long long batchN) {
#ifdef HAVE_CUDA
            if (gses) {
                // Resident path: launch and go. seedBase = cumulative photon count, same
                // stream convention as the CPU path / the old per-batch wrapper.
                sharedForwardGpuBatch(gses, batchN, (unsigned long long)accN, diffraction);
                accN += batchN;
                accFresh = false;
                return;
            }
#endif
            EnergyReport e;
            std::vector<Film> films =
                renderForwardShared(scene, cams, rxs, rys, batchN, nThreads, e, diffraction,
                                    groupMode == 'A', (unsigned long long)accN, g_beamGather);
            for (int c = 0; c < nc; ++c) acc[c].merge(films[c]);
            accN += batchN;
            accE.emitted += e.emitted; accE.absorbed += e.absorbed; accE.sensor += e.sensor;
            accE.escaped += e.escaped; accE.residual += e.residual;
        };

        // Pull the running totals off the device into acc[]/accE (no-op on the CPU path,
        // where runBatch merges host-side; no-op when already fresh).
        auto syncAcc = [&] {
#ifdef HAVE_CUDA
            if (gses && !accFresh) { sharedForwardGpuDownload(gses, acc, accE); accFresh = true; }
#endif
        };

        // Write every camera's image (+ optional .ftbuf). `quiet` suppresses the per-file
        // announce for intermediate saves; these groups are per-frame-auto-exposed
        // (expGroup < 0) so no shared exposure anchor is involved.
        auto writeOut = [&](bool quiet) {
            syncAcc();
            for (int c = 0; c < nc; ++c) {
                const RenderCam& rc = toRender[idx[c]];
                Film disp = acc[c];
                addEnvBackground(disp, scene, rc.cam, accN);   // directly-viewed sky (env scenes)
                std::string op = outFor(rc.name);
                if (!quiet && toRender.size() > 1)
                    std::printf("[camera] '%s' (mode %c, %dx%d) -> %s\n",
                                rc.name.c_str(), groupMode, rc.res, rc.resY, op.c_str());
                if (!writeFilm(op.c_str(), disp, (double)accN, rc.exposure, quiet, nullptr, scene.absolute))
                    sharedWriteFail = true;
                if (wantCheckpoint) {
                    Checkpoint ck; ck.film = acc[c]; ck.N = accN; ck.energy = accE;
                    if (!writeCheckpoint(op, ck, checkpointGuard(scene, groupMode, rxs[c], rys[c]), groupMode))
                        std::fprintf(stderr, "[checkpoint] could not write %s\n", checkpointPath(op).c_str());
                }
            }
        };

        using clk = std::chrono::steady_clock;
        // A plain fixed-N render with -window (no budget) still wants a live view, so chunk
        // N and stop at the total; a budgeted render loops until its time/noise/forever stop.
        const bool chunkFixed = !progressive && g_showWindow;
        if (progressive || chunkFixed) {
            long long batchN = chunkFixed ? std::max(1LL, ((N > 0) ? N : 2'000'000) / 16)
                                          : ((N > 0) ? N : 2'000'000);
            const char* resumeTag = (resume && accN > 0) ? " [resuming]" : "";
            char noiseSuffix[64] = "";
            if (noiseTarget > 0.0) std::snprintf(noiseSuffix, sizeof noiseSuffix, " or until ~%.2g%% noise", noiseTarget);
            if (chunkFixed)
                std::printf("[camera] shared model-%c pass: %d cameras, %lld photons in %lld-photon "
                            "batches on %s (light=%s)%s — live window; Ctrl-C to stop early ...\n",
                            groupMode, nc, N, batchN, backend.c_str(), lightLabel, resumeTag);
            else if (runForever)
                std::printf("[camera] shared model-%c pass: %d cameras, tracing indefinitely in "
                            "%lld-photon batches on %s (light=%s)%s%s — Ctrl-C to stop ...\n",
                            groupMode, nc, batchN, backend.c_str(), lightLabel, resumeTag, noiseSuffix);
            else if (timeBudgetSec > 0.0)
                std::printf("[camera] shared model-%c pass: %d cameras, tracing for %.3gs%s in "
                            "%lld-photon batches on %s (light=%s)%s (Ctrl-C to stop early) ...\n",
                            groupMode, nc, timeBudgetSec, noiseSuffix, batchN, backend.c_str(), lightLabel, resumeTag);
            else
                std::printf("[camera] shared model-%c pass: %d cameras, tracing until ~%.2g%% noise in "
                            "%lld-photon batches on %s (light=%s)%s (Ctrl-C to stop early) ...\n",
                            groupMode, nc, noiseTarget, batchN, backend.c_str(), lightLabel, resumeTag);
            if (preview) { enableAnsiTerminal(); g_previewRows = 0; }
            auto prev = std::signal(SIGINT, onInterrupt);
#ifdef SIGBREAK
            auto prevBrk = std::signal(SIGBREAK, onInterrupt);
#endif
            auto t0 = clk::now();
            auto lastSave = t0;
            long long batches = 0;
            bool metNoise = false;
            for (;;) {
                runBatch(batchN); ++batches;
                double elapsed   = std::chrono::duration<double>(clk::now() - t0).count();
                double sinceSave = std::chrono::duration<double>(clk::now() - lastSave).count();
                bool stopped = g_stopRequested != 0;
                bool timeUp  = (!runForever && timeBudgetSec > 0.0 && elapsed >= timeBudgetSec);
                bool wantStatus = sinceSave >= intervalSec;
                // Graininess estimate from camera 0's lit-pixel hit count (mirrors the
                // single-camera path); drives the status line and the -noise stop.
                double noisePct = 0.0, meanHits = 0.0;
                if (noiseTarget > 0.0 || wantStatus || stopped || timeUp) {
#ifdef HAVE_CUDA
                    if (gses && !accFresh) {
                        // Resident GPU path: the hit counts live on the device. A status /
                        // stop boundary wants the films anyway, so do the full download;
                        // a bare -noise poll between intervals fetches ONLY camera 0's
                        // hits (npix doubles) instead of every film.
                        if (wantStatus || stopped || timeUp) syncAcc();
                        else sharedForwardGpuHits0(gses, acc[0].hits);
                    }
#endif
                    double sumHits = 0.0; long long lit = 0;
                    for (double h : acc[0].hits) if (h > 0.0) { sumHits += h; ++lit; }
                    meanHits = lit ? sumHits / (double)lit : 0.0;
                    noisePct = meanHits > 0.0 ? 100.0 / std::sqrt(meanHits) : 0.0;
                }
                bool noiseMet = (noiseTarget > 0.0 && meanHits > 0.0 && noisePct <= noiseTarget);
                if (noiseMet) metNoise = true;
                bool totalDone = chunkFixed && N > 0 && accN >= N;
                bool done = stopped || timeUp || noiseMet || totalDone;
                if (done || wantStatus) {
                    writeOut(/*quiet*/preview);
                    lastSave = clk::now();
                    const char* why = stopped ? " (stopping)" : noiseMet ? " (noise target met)"
                                    : totalDone ? " (done)" : "";
                    char st[240];
                    if (chunkFixed)
                        std::snprintf(st, sizeof st, "[live] %.1fs, %lld / %lld photons, %d cams, ~%.1f%% noise%s",
                                      elapsed, accN, N, nc, noisePct, why);
                    else if (runForever)
                        std::snprintf(st, sizeof st, "[forever] %.1fs, %lld batches, %lld photons, %d cams, ~%.1f%% noise%s",
                                      elapsed, batches, accN, nc, noisePct, why);
                    else if (timeBudgetSec > 0.0)
                        std::snprintf(st, sizeof st, "[time] %.1fs / %.3gs, %lld photons, %d cams, ~%.1f%% noise%s",
                                      elapsed, timeBudgetSec, accN, nc, noisePct, why);
                    else
                        std::snprintf(st, sizeof st, "[noise] ~%.2g%% target, %.1fs, %lld photons, %d cams, ~%.1f%% noise%s",
                                      noiseTarget, elapsed, accN, nc, noisePct, why);
                    if (preview || g_showWindow) {
                        Film disp = acc[0];
                        addEnvBackground(disp, scene, toRender[idx[0]].cam, accN);
                        if (preview) ansiPreview(disp, (double)accN, toRender[idx[0]].exposure, st);
                        else { std::printf("%s\n", st); std::fflush(stdout); }
                        liveWindowUpdate(disp, (double)accN, toRender[idx[0]].exposure, scene.absolute, st);
                    } else { std::printf("%s\n", st); std::fflush(stdout); }
                }
                if (done) break;
            }
            std::signal(SIGINT, prev);
#ifdef SIGBREAK
            std::signal(SIGBREAK, prevBrk);
#endif
            if (g_stopRequested) std::printf("\n[stop] interrupted — images and checkpoints saved.\n");
            else if (metNoise) std::printf("[noise] reached the ~%.2g%% target at %lld photons — images saved.\n",
                                           noiseTarget, accN);
            if (wantCheckpoint)
                std::printf("[checkpoint] shared model-%c group holds %lld photons — rerun with -resume to add more\n",
                            groupMode, accN);
        } else {
            // Fixed photon count, no window: one batch of N (seedBase 0 unless resumed),
            // bit-identical to the historical single-shot shared pass.
            std::printf("[camera] shared model-%c pass: %d cameras, %lld photons on %s (light=%s)%s ...\n",
                        groupMode, nc, N, backend.c_str(), lightLabel, (resume && accN > 0) ? " [resuming]" : "");
            runBatch(N);
            writeOut(/*quiet*/false);
        }
#ifdef HAVE_CUDA
        // Tear the resident session down. Every exit path above ends in a writeOut(),
        // whose syncAcc() already pulled the final films/energy into acc[]/accE.
        if (gses) { sharedForwardGpuEnd(gses); gses = nullptr; }
#endif

        double tot = accE.absorbed + accE.sensor + accE.escaped + accE.residual;
        if (accE.emitted > 0.0)
            std::printf("[energy] absorbed=%.4f sensor=%.4f escaped=%.4f residual=%.4f (sum/emitted=%.6f)\n",
                        accE.absorbed / accE.emitted, accE.sensor / accE.emitted, accE.escaped / accE.emitted,
                        accE.residual / accE.emitted, tot / accE.emitted);
    };
    runSharedGroup(groupB, 'B');
    runSharedGroup(groupA, 'A');

    // Shared photon-map pass (mode M): build ONE view-independent map, gather every
    // camera from it. This is where the photon map pays off over per-camera backward
    // tracing — the (expensive) forward photon flight amortizes across all frames.
    auto runSharedPhotonMap = [&](const std::vector<int>& idx) {
        if (idx.empty() || g_stopRequested) return;
        g_windowMode = modeLabel('M');   // title bar shows the shared photon-map mode
        double radius = (g_pmRadiusAbs > 0.0) ? g_pmRadiusAbs
                                              : scene.sceneRadius * g_pmRadiusFactor;
        // Trap Ctrl-C for the whole mode-M gather. Without this the default SIGINT action
        // terminates the process, which on a -window-less / backgrounded run would abruptly
        // kill a live CUDA context mid-gather — the exact scenario cudaGracefulShutdown()
        // exists to avoid (async nvlddmkm teardown BSOD). With the handler, an interrupt
        // just sets g_stopRequested; the gather finishes the current frame (the writeFrame /
        // liveProg callbacks and the CPU loop below both poll it), returns, and main() runs
        // the orderly teardown. The RAII guard restores the prior handlers on every exit
        // path (including the GPU branch's early return). A window close already routes
        // through the same g_stopRequested via liveWindowUpdate.
        struct SigGuard {
            void (*prevInt)(int) = std::signal(SIGINT, onInterrupt);
#ifdef SIGBREAK
            void (*prevBrk)(int) = std::signal(SIGBREAK, onInterrupt);
#endif
            ~SigGuard() { std::signal(SIGINT, prevInt);
#ifdef SIGBREAK
                          std::signal(SIGBREAK, prevBrk);
#endif
            }
        } sigGuard;
#ifdef HAVE_CUDA
        // GPU photon map: build the map once on the device and gather every frame there —
        // the same amortization as the CPU shared path, but the (expensive) gather runs on
        // the GPU. Only the DIRECT density estimate is ported, so a final-gather render, a
        // lens camera, an env scene, or an unsupported material falls back to the CPU below.
        {
            const bool wantGpu  = !std::strcmp(device, "gpu");
            const bool wantAuto = !std::strcmp(device, "auto");
            bool allPinhole = true;
            for (int i : idx) if (toRender[i].cam.hasLens()) allPinhole = false;
            if ((wantGpu || wantAuto) && allPinhole &&
                cudaAvailable() && cudaPhotonMapSupported(scene)) {
                std::vector<Camera> cams; std::vector<int> rxs, rys;
                for (int i : idx) { cams.push_back(toRender[i].cam); rxs.push_back(toRender[i].res); rys.push_back(toRender[i].resY); }
                std::printf("[camera] shared photon map (mode M) on %s: %zu cameras, %lld "
                            "photons, radius %.4g (light=%s)%s ...\n",
                            cudaDeviceName(), cams.size(), N, radius, lightLabel,
                            g_pmFinalGather > 0 ? " [final gather]" : "");
                EnergyReport e;
                // Drive the live window (per the always-`-window` rule): the shared gather
                // reports each frame's converging film here so the window shows it build up
                // and, on a flythrough, flips through the frames as they complete. Only armed
                // when a window is open so a headless batch pays no extra device->host copies.
                SppProgress liveProg;
                if (g_showWindow) {
                    const double liveExp = toRender[idx[0]].exposure;
                    liveProg.report = [&, liveExp](const Film& f, long long sppDone, bool) -> bool {
                        liveWindowUpdate(f, (double)sppDone, liveExp, scene.absolute);
                        return g_stopRequested != 0;   // window closed -> stop after this chunk
                    };
                }
                // Write each frame to disk the instant its gather completes (crash-safe
                // incremental output, same as the CPU mode-M path below): a flythrough of
                // hundreds of frames can run for many minutes, and batching every write to
                // the very end means an interrupt / crash / power loss throws away ALL of it.
                // Writing per frame also lets the device path free each film as it goes, so a
                // long render stays near one-frame of host RAM instead of ~3 GB of films.
                std::function<bool(int, const Film&)> writeFrame =
                    [&](int k, const Film& f) -> bool {
                        const RenderCam& rc = toRender[idx[k]];
                        std::string op = outFor(rc.name);
                        if (toRender.size() > 1)
                            std::printf("[camera] '%s' (mode M/GPU, %dx%d) -> %s\n",
                                        rc.name.c_str(), rc.res, rc.resY, op.c_str());
                        double* anchor = (rc.expGroup >= 0) ? &expAnchors[rc.expGroup] : nullptr;
                        if (!writeFilm(op.c_str(), f, (double)spp, rc.exposure, false, anchor, scene.absolute))
                            sharedWriteFail = true;
                        return g_stopRequested != 0;   // window closed / Ctrl-C -> stop after this frame
                    };
                renderPhotonMapSharedCuda(scene, cams, rxs, rys, N, radius, e,
                                          diffraction, spp,
                                          g_showWindow ? &liveProg : nullptr, &writeFrame,
                                          g_pmapLoad.empty() ? nullptr : g_pmapLoad.c_str(),
                                          g_pmapSave.empty() ? nullptr : g_pmapSave.c_str(), g_heroC,
                                          g_pmFinalGather,
                                          g_pmAutoRadius ? g_pmAutoCount : 0.0);
                if (e.emitted > 0.0)
                    std::printf("[energy] absorbed=%.4f escaped=%.4f residual=%.4f (sum/emitted=%.6f)\n",
                                e.absorbed / e.emitted, e.escaped / e.emitted, e.residual / e.emitted,
                                (e.absorbed + e.sensor + e.escaped + e.residual) / e.emitted);
                return;
            }
        }
#endif
        std::printf("[camera] shared photon map (mode M): %zu cameras, %lld photons, "
                    "radius %.4g on %d CPU threads (light=%s)%s ...\n",
                    idx.size(), N, radius, nThreads, lightLabel,
                    g_pmFinalGather > 0 ? " [final gather]" : "");
        PhotonMap pm;
        auto tp0 = std::chrono::steady_clock::now();
        tracePhotonPass(scene, N, nThreads, diffraction, pm, g_heroC);
        buildPhotonMap(pm, radius, "[camera]");
        double buildSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        std::printf("[camera] photon map: %zu photons from %lld emitted in %.1fs, "
                    "grid %dx%dx%d — gathering %zu cameras ...\n",
                    pm.photons.size(), pm.nEmitted, buildSec, pm.nx, pm.ny, pm.nz, idx.size());
        if (pm.photons.empty())
            std::fprintf(stderr, "[mode M] warning: 0 photons deposited — images "
                                 "will be black.\n");
        for (size_t k = 0; k < idx.size(); ++k) {
            // Poll the interrupt between frames: a window close / Ctrl-C sets g_stopRequested,
            // and we stop after finishing (writing) the current frame rather than abandoning
            // the whole flythrough or abruptly terminating a live render mid-gather.
            if (g_stopRequested) break;
            const RenderCam& rc = toRender[idx[k]];
            Film f = renderPhotonCamera(scene, rc.cam, rc.res, rc.resY, pm, spp,
                                        nThreads, diffraction, /*maxBounce*/32, 0, g_pmFinalGather);
            std::string op = outFor(rc.name);
            if (toRender.size() > 1)
                std::printf("[camera] '%s' (mode M, %dx%d) -> %s\n",
                            rc.name.c_str(), rc.res, rc.resY, op.c_str());
            double* anchor = (rc.expGroup >= 0) ? &expAnchors[rc.expGroup] : nullptr;
            if (!writeFilm(op.c_str(), f, (double)spp, rc.exposure, false, anchor, scene.absolute))
                sharedWriteFail = true;
        }
    };
    runSharedPhotonMap(groupM);

    for (int i : restIdx) {
        const RenderCam& rc = toRender[i];
        if (toRender.size() > 1)
            std::printf("[camera] rendering '%s' (mode %c, %dx%d) -> %s\n",
                        rc.name.c_str(), rc.mode, rc.res, rc.resY, outFor(rc.name).c_str());
        double* anchor = (rc.expGroup >= 0) ? &expAnchors[rc.expGroup] : nullptr;
        int rv = runRender(scene, rc.cam, rc.mode, N, rc.res, rc.resY, spp, nThreads,
                           device, diffraction, lightLabel, outFor(rc.name), rc.exposure,
                           timeBudgetSec, resume, wantCheckpointFlag, runForever,
                           preview, intervalSec, noiseTarget, wavefront, anchor, rgbBackward,
                           maxBounceOverride, directOnly);
        if (rv != 0) return rv;
    }

    // --- Stereoscopic compositing (-stereo): fuse each eye pair into the -o image ------
    // Every eye rendered to its own PNG (sharing an exposure anchor for identical tone-
    // mapping); combine them into the requested output (side-by-side or Dubois anaglyph),
    // then delete the intermediate eye files (+ any .ftbuf) unless -stereo-keep-eyes.
    if (!stereoPairs.empty() && !g_stopRequested) {
        for (const StereoPair& sp : stereoPairs) {
            if (!stereoComposite(stereoMode, sp.leftPath, sp.rightPath, sp.finalPath))
                sharedWriteFail = true;
            if (!stereoKeepEyes) {
                std::remove(sp.leftPath.c_str());
                std::remove(sp.rightPath.c_str());
                std::remove((sp.leftPath  + ".ftbuf").c_str());
                std::remove((sp.rightPath + ".ftbuf").c_str());
            }
        }
    }
    return sharedWriteFail ? 1 : 0;
}

// --- Resident preview server (-serve) -----------------------------------------
// `ftrace -serve -in <scene.ftsl> [render flags…]` keeps the process — and with it
// the live window, CUDA context, spectral/upsampling tables — resident, re-rendering
// whenever a new scene path arrives on stdin (one path per line). This skips the
// per-frame cost of spawning a fresh process and re-initialising all of that global
// state, which is the dominant fixed overhead for cheap preview frames.
//
// Protocol (line-oriented, both directions):
//   stdout  "[serve] ready"              once, before the first frame
//   stdin   <path/to/frame.ftsl>\n       request: render this scene, reusing all flags
//   stdout  "[serve] done <path>"        after each frame completes (or errors)
//   stdin   "quit" / "exit" / EOF        end the loop
//   stdout  "[serve] shutdown"           on exit
//
// Every rendered scene reuses the *same* CLI flags (-mode/-n/-r/-window/-o/…) given
// on the -serve command line; only the -in path is swapped per frame. Honest scope:
// this delivers the resident-process win only. It does NOT yet do incremental delta
// rendering, static-geometry/BVH caching between frames, or a reduced preview LOD —
// each frame is a full independent render. The live window is created lazily on the
// first frame and keeps that first frame's resolution for the session.
static int runServe(int argc, char** argv, int inValPos) {
    std::printf("[serve] ready\n");
    std::fflush(stdout);
    int rc = 0;
    // Initial render for whatever -in was on the command line (if any).
    if (inValPos >= 0) {
        rc = run(argc, argv);
        std::printf("[serve] done %s\n", argv[inValPos]);
        std::fflush(stdout);
    }
    // Stream subsequent scene paths from stdin, one per line. std::fgets (not iostream)
    // keeps this dependency-free and blocks until a line or EOF.
    std::string pathBuf;
    char buf[4096];
    while (true) {
        // A closed live window means the user dismissed the preview: stop serving.
        if (g_showWindow && g_liveWin && g_liveWin->closed()) break;
        if (!std::fgets(buf, sizeof(buf), stdin)) break;             // EOF
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' ||
                                 line.back() == ' '  || line.back() == '\t'))
            line.pop_back();
        size_t s = line.find_first_not_of(" \t");
        if (s == std::string::npos) continue;                        // blank line
        line = line.substr(s);
        if (line == "quit" || line == "exit") break;
        if (inValPos < 0) {
            std::fprintf(stderr, "[serve] no -in slot to swap; ignoring '%s'\n", line.c_str());
            continue;
        }
        // Point the -in argv slot at the new path and re-render. pathBuf owns the
        // storage for the duration of this run() call.
        pathBuf = line;
        argv[inValPos] = const_cast<char*>(pathBuf.c_str());
        // An EXTERNAL stop (-stop) means "shut this process down", not "abandon this
        // frame", so it must not be cleared and re-entered like a per-frame Ctrl-C.
        if (g_extStopRequested.load()) break;
        g_stopRequested = 0;   // clear any prior clean-stop request before the new frame
        try {
            rc = run(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[serve] error: %s\n", e.what());
            rc = 1;
        }
        std::printf("[serve] done %s\n", pathBuf.c_str());
        std::fflush(stdout);
    }
    std::printf("[serve] shutdown\n");
    std::fflush(stdout);
    return rc;
}

// Thin wrapper: turn a fatal configuration error (e.g. an explicit `file:`/`glass:`/
// `illuminant:` reference whose target is missing or malformed — thrown by the
// spectral-library resolver) into a clean message + non-zero exit, instead of a
// silent fall-through to a default illuminant that would render the wrong thing.
int main(int argc, char** argv) {
    // `-stop [<pid>|all]`: talk to ALREADY-RUNNING renders and exit. Handled before
    // anything else so it works from a bare command line -- it loads no scene, opens
    // no window and creates no CUDA context, so there is nothing here to tear down.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-stop") && std::strcmp(argv[i], "--stop")) continue;
        const char* who = (i + 1 < argc && argv[i + 1][0] != '-') ? argv[i + 1] : nullptr;
        return runStopCommand(who);
    }
    // Tear the CUDA context down synchronously, in-process, on EVERY exit path (normal
    // return or exception). Leaving it for the driver to reclaim implicitly after main()
    // returns triggers an asynchronous nvlddmkm DPC teardown that, on buggy driver
    // builds, can fault and bugcheck the machine (the "reboot a few seconds after the
    // window closed" BSOD). Draining + resetting here closes that window.
    int rc;
    try {
        // Native loom viewer (-viewer <sidecar.json>): open the ImGui/D3D11 GUI on a
        // scene-introspection sidecar instead of rendering. Short-circuits the renderer.
        // `-loom <scene.py>` names the loom build file the viewer opens its LIVE
        // re-introspection channel against (§F4 item 2); without it the viewer falls
        // back to the sidecar's own `build` provenance key, and without that too it
        // shows the sidecar frozen.
        {
            const char* viewerSidecar = nullptr;
            const char* viewerLoom    = nullptr;
            for (int i = 1; i < argc; ++i) {
                if (!std::strcmp(argv[i], "-viewer") || !std::strcmp(argv[i], "--viewer")) {
                    if (i + 1 >= argc) {
                        std::fprintf(stderr, "error: -viewer needs a sidecar .json path\n");
                        return 1;
                    }
                    viewerSidecar = argv[++i];
                } else if (!std::strcmp(argv[i], "-loom") || !std::strcmp(argv[i], "--loom")) {
                    if (i + 1 >= argc) {
                        std::fprintf(stderr, "error: -loom needs a loom scene .py path\n");
                        return 1;
                    }
                    viewerLoom = argv[++i];
                }
            }
            if (viewerSidecar)
                return runViewerGui(viewerSidecar, viewerLoom ? viewerLoom : "");
            if (viewerLoom) {
                // -loom names a live channel, and there are two of them: the viewer's F4
                // re-introspection (-viewer) and the fly editor's E2 value channel
                // (-anim), which the renderer's own arg loop parses. Only reject the flag
                // when NEITHER host asked for it, rather than letting it fall through to
                // be silently eaten as an unknown flag.
                bool withAnim = false;
                for (int i = 1; i < argc; ++i)
                    if (!std::strcmp(argv[i], "-anim")) { withAnim = true; break; }
                if (!withAnim) {
                    std::fprintf(stderr, "error: -loom <scene.py> is only meaningful with "
                                         "-viewer <sidecar.json> or -anim <drive.json>\n");
                    return 1;
                }
            }
        }
        // Resident preview server (-serve): keep the process alive and re-render each
        // scene path streamed on stdin. Find the -in value slot to swap per frame.
        bool serve = false;
        int inValPos = -1;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "-serve")) serve = true;
            else if (!std::strcmp(argv[i], "-in") && i + 1 < argc) inValPos = i + 1;
        }
        // Publish this render in the stop channel (see `-stop` above) so it can be asked
        // to finish cleanly from outside, and keep the watcher alive across the whole
        // run INCLUDING the -keepwindow hold below. stopChannelEnd() unpublishes it on
        // every exit path, normal or exceptional.
        {
            const char* inPath = nullptr; const char* outPath = nullptr;
            for (int i = 1; i < argc; ++i) {
                if (!std::strcmp(argv[i], "-in") && i + 1 < argc)      inPath  = argv[i + 1];
                else if (!std::strcmp(argv[i], "-o") && i + 1 < argc)  outPath = argv[i + 1];
            }
            stopChannelStart(std::string(inPath ? inPath : "(no -in)") + " -> " +
                             (outPath ? outPath : "(default output)"));
        }
        rc = serve ? runServe(argc, argv, inValPos) : run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        rc = 1;
    }
    // -keepwindow / -hold: keep the finished image on screen. The live window runs its
    // own UI thread, so we just block here until the user closes it (or it's already gone)
    // rather than letting process exit tear it down the instant the render completes.
    // (An external -stop means "exit now", so it skips the hold entirely rather than
    // announcing a wait it's about to break out of.)
    if (g_keepWindow && g_liveWin && !g_liveWin->closed() && !g_extStopRequested.load()) {
        std::printf("[window] render done — close the preview window to exit "
                    "(or run: ftrace -stop %ld).\n", ftraceCurrentPid());
        std::fflush(stdout);
        // The hold ends on the window closing OR on an external -stop, so a held window
        // on an unattended machine is never a reason to reach for taskkill.
        while (!g_liveWin->closed() && !g_extStopRequested.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    stopChannelEnd();
#ifdef HAVE_CUDA
    cudaGracefulShutdown();
#endif
    return rc;
}
