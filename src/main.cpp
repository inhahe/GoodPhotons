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
#include "livewindow.h"         // -window: real OS live-preview window (Win32 GDI)
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

    bool passA = std::fabs(sMean - meanAnalytic) < 1.0;
    bool passB = std::fabs(frac - expectFrac) < 0.005;
    bool passC = (moOut > lin) && std::fabs(moOut - meanAnalytic) < 1.5;
    bool pass = passA && passB && passC;

    std::printf("[checkfluoro] emission mean: sampler=%.2f analytic=%.2f nm  (%s)\n",
                sMean, meanAnalytic, passA ? "ok" : "BAD");
    std::printf("[checkfluoro] reemit fraction @%.0fnm: measured=%.4f expected(eps*Q)=%.4f  (%s)\n",
                lin, frac, expectFrac, passB ? "ok" : "BAD");
    std::printf("[checkfluoro] Stokes shift: in=%.0f -> out_mean=%.2f nm (elastic=%.3f absorb=%.3f)  (%s)\n",
                lin, moOut, (double)elastic / Nt, (double)absorb / Nt, passC ? "ok" : "BAD");
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

    bool pass = passA && passB && passW && passC && passD;
    std::printf("[checkupsample] round-trip max error (excl. white) = %.5f  (%s)\n", maxErr, passA ? "ok" : "BAD");
    std::printf("[checkupsample] reflectance in [0,1]  (%s)\n", passB ? "ok" : "BAD");
    std::printf("[checkupsample] pure-white residual = %.5f (<0.02 expected)  (%s)\n", whiteErr, passW ? "ok" : "BAD");
    std::printf("[checkupsample] mid-grey round-trip = %.6f  (%s)\n", greyErr, passC ? "ok" : "BAD");
    std::printf("[checkupsample] illuminant round-trip max error = %.5f  (%s)\n", illumErr, passD ? "ok" : "BAD");
    std::printf("[checkupsample] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
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
static void addEnvBackground(Film& film, const Scene& scene, const Camera& cam, long long N) {
    if (scene.envIndex < 0) return;
    for (int py = 0; py < film.resY; ++py)
        for (int px = 0; px < film.resX; ++px) {
            Ray r = cam.genRay(px, py, 0.5, 0.5);
            Hit h = scene.closestHit(r);
            if (!h.valid) film.add(px, py, scene.envXYZForDir(r.d) * (double)N);
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
static Film renderBackward(const Scene& scene, const Camera& cam, int resX, int resY,
                           long long spp, int nThreads, bool diffraction = true,
                           unsigned long long sampleBase = 0) {
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    auto worker = [&](int tid) {
        BackwardRenderer br; br.diffraction = diffraction; br.heroC = g_heroC;
        int y0 = resY * tid / nThreads, y1 = resY * (tid + 1) / nThreads;
        br.renderRows(scene, cam, out, y0, y1, spp, sampleBase);
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
//     exposure-exact PNG.
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

static int convertToPng(const std::string& inPath, const std::string& outPath) {
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
        // Tone-map with the default p99 auto-exposure (see note above). writeFilm prints
        // the "wrote <out> ..." line.
        return writeFilm(outPath.c_str(), f, (double)std::max<long long>(Nph, 1)) ? 0 : 1;
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
        if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env || em.collimated)
            return "spot / environment / collimated lights";
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
        bool stopped  = g_stopRequested != 0;
        bool timeUp   = (!runForever && timeBudgetSec > 0.0 && elapsed >= timeBudgetSec);
        bool noiseMet = (noiseTarget > 0.0 && totalSpp > 0 && noisePct <= noiseTarget);
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
                std::snprintf(st, sizeof st, "[forever] %.1fs, %lld spp, ~%.2f%% noise%s",
                              elapsed, totalSpp, noisePct, why);
            else if (timeBudgetSec > 0.0)
                std::snprintf(st, sizeof st, "[time] %.1fs / %.3gs, %lld spp, ~%.2f%% noise%s",
                              elapsed, timeBudgetSec, totalSpp, noisePct, why);
            else if (noiseTarget > 0.0)
                std::snprintf(st, sizeof st, "[noise] target ~%.2g%%, %.1fs, %lld spp, ~%.2f%% noise%s",
                              noiseTarget, elapsed, totalSpp, noisePct, why);
            else
                std::snprintf(st, sizeof st, "[spp] %lld / %lld, %.1fs, ~%.2f%% noise",
                              totalSpp, baseSpp + sppReq, elapsed, noisePct);
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
                r = renderBackwardCuda(scene, cam, res, resY, dSpp, diffraction, &bp);
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
                     double* exposureAnchor = nullptr) {
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
    // the mode-P camera-side layer) when the scene is within the backward-GPU v1 scope
    // (renderBackwardCuda / cudaBackwardSupported — no fog/env/spot/collimated/
    // fluorescence); otherwise the backward layer falls back to the CPU. Mode V keeps
    // its backward reference on the CPU by design. Fisheye/panoramic lenses run on the
    // GPU too (the device camera's project()/pixelSolidAngle() port the analytic
    // projection remap) for the pinhole-splat modes (B/V/P).
    const bool gpuForwardMode =
        (mode == 'A' || mode == 'B' || mode == 'C' || mode == 'V' || mode == 'P');
    const bool gpuBdptMode = (mode == 'D');   // GPU BDPT megakernel (own support check)
    const bool gpuBackwardMode = (mode == 'R');   // GPU backward reference megakernel (own check)
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
                                  "spot/env/collimated light, or a per-hit BSDF the GPU "
                                  "BDPT can't MIS: a procedural pattern or frosted/colored "
                                  "glass)";
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
            // cylinder Lambertian lights and textured/specular materials, but not fog,
            // env light, spot/collimated lights, or fluorescence (v1 scope).
            if (!cudaBackwardSupported(scene, cam)) {
                const char* why = "scene has a backward-GPU-unsupported feature "
                                  "(fog, env light, spot/collimated light, fluorescence, "
                                  "or a lens deeper than the device cap)";
                if (wantGpu) std::fprintf(stderr, "[device] %s; using CPU\n", why);
                else         std::printf("[device] auto -> CPU (%s)\n", why);
            } else {
                useGpu = true;
                std::printf("[device] %s -> GPU: %s\n", wantAuto ? "auto" : "gpu",
                            cudaDeviceName());
            }
        } else if (!gpuForwardMode) {
            const char* why = "unsupported mode - CPU-only path";
            if (wantGpu) std::fprintf(stderr,
                "[device] GPU can't accelerate this render: %s; using CPU\n", why);
            else         std::printf("[device] auto -> CPU (%s)\n", why);
        } else if (!cudaForwardSupported(scene)) {
            const char* why = "GPU-unsupported feature (layered material, indexed "
                              "palette, parametric record, or oversized multilayer/mix "
                              "material)";
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
        std::printf("mode R: backward reference at %dx%d on %s (light=%s) ...\n",
                    res, resY,
                    gpuBackward ? "GPU" : (std::to_string(nThreads) + " CPU threads").c_str(),
                    lightLabel);
        auto renderChunked = [&](long long sppTarget, const SppProgress* p) -> Film {
#ifdef HAVE_CUDA
            if (gpuBackward) return renderBackwardCuda(scene, cam, res, resY, sppTarget, diffraction, p);
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
        writeFilm("validate_forward.ppm", fwd, (double)N);
        writeFilm("validate_backward.ppm", ref, (double)spp);
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
        int maxDepth = 8;   // path length in edges; connection cost grows ~depth^2
        std::printf("mode D: bidirectional path tracing at %dx%d on %s (maxDepth=%d, light=%s) ...\n",
                    res, resY, useGpu ? "GPU" : (std::to_string(nThreads) + " CPU threads").c_str(),
                    maxDepth, lightLabel);
        // BDPT accumulates a SUM over spp (cam image + light-splat image), so it chunks
        // through the same unified progress driver as the forward and mode-R renders.
        auto renderChunked = [&](long long sppTarget, const SppProgress* p) -> Film {
#ifdef HAVE_CUDA
            if (useGpu) return renderBdptCuda(scene, cam, res, resY, sppTarget, maxDepth, diffraction, p);
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
        pm.build(radius);
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
        int maxDepth = 8;   // full path length in edges
        double R0 = (g_pmRadiusAbs > 0.0) ? g_pmRadiusAbs
                                          : scene.sceneRadius * g_pmRadiusFactor;
        std::printf("mode U: VCM/UPS — connections + merging, R0=%.4g, alpha=%.2f at %dx%d on "
                    "%d CPU threads (maxDepth=%d, light=%s) ...\n",
                    R0, g_vcmAlpha, res, resY, nThreads, maxDepth, lightLabel);
        vcm::VcmState st; st.init(res, resY);
        auto renderChunked = [&](long long passTarget, const SppProgress* p) -> Film {
            Film disp; disp.resX = res; disp.resY = resY; disp.alloc();
            for (long long pass = 0; pass < passTarget; ++pass) {
                // Progressive radius schedule (Georgiev/SmallVCM): shrink from R0.
                double it = (double)(st.passes + 1);
                double radius = R0 * std::pow(it, 0.5 * (g_vcmAlpha - 1.0));
                if (radius <= 0.0) radius = R0;
                vcm::vcmPass(scene, cam, st, radius, nThreads, diffraction, maxDepth,
                             (uint64_t)(st.passes + 1));
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
"  -n <count>            photon/sample count (accepts 2e8, 1.5e9)\n"
"  -r <W> [H]            resolution (square if H omitted)\n"
"  -time <sec>           wall-clock budget (progressive)\n"
"  -noise <pct>          stop at target graininess (progressive)\n"
"  -forever              trace until Ctrl-C (progressive)\n"
"  -spp <n>              samples/pixel for backward modes R/V\n"
"  -device auto|cpu|gpu  compute device (default: auto); -wavefront = streaming GPU backend\n"
"  -t <n>                CPU thread count\n"
"\n"
"Output, preview & checkpointing:\n"
"  -o <file.ppm|.png>    output path (default: cornell.ppm)\n"
"  -window               live OS preview window, refreshed as it converges\n"
"  -keepwindow|-hold     like -window but hold the final image until you close it\n"
"  -preview              live ANSI thumbnail in the terminal\n"
"  -interval <sec>       periodic image-write / preview cadence (default: 15)\n"
"  -checkpoint           write a resumable .ftbuf sidecar next to -o (modes A/B/C)\n"
"  -resume               continue an accumulated render from its .ftbuf checkpoint\n"
"\n"
"Raster preview & interactive explore (no light transport):\n"
"  -raster               fast solid-shaded preview; -raster-gpu = GPU isosurface preview\n"
"  -raster-iso <n>       marching-cubes resolution for isosurfaces (0 = skip)\n"
"  -explore | -fly       interactive fly-camera viewer (implies -keepwindow -no-meter)\n"
"  -noclip|-nocollide    start the fly viewer with wall collision off\n"
"  -see-through|-glass   render clear dielectrics as see-through; -glass-clarity <0..1>\n"
"\n"
"Stereoscopic 3-D output:\n"
"  -stereo sbs|cross|anaglyph|anaglyph-gm   stereo pair / anaglyph composite\n"
"  -eye-sep <m>          interocular distance (default: 0.063)\n"
"  -view-dist <m>        viewing distance (default: 0.6)\n"
"  -dpi <n|auto>         screen pixel density; -convergence <m> = convergence-plane distance\n"
"\n"
"Utilities (exit after running):\n"
"  -topng|-convert <in> <out.png>   convert .ppm/.ftbuf to PNG\n"
"  -review <base>        play a rendered frame sequence on the live window\n"
"  -export-mesh <o.obj> [-mesh-res N] [-mesh-adaptive]   isosurface -> mesh\n"
"  -serve                resident loop: re-render scene paths streamed on stdin\n"
"  -h | --help           show this help and exit\n"
"\n"
"See README.md for the complete flag list (fog, thin-film, meshes, diagnostics, …).\n",
        prog, prog, prog, prog, prog, prog);
}

static int run(int argc, char** argv) {
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
            std::fprintf(stderr, "usage: %s -topng <input.ppm|input.ftbuf> <output.png>\n",
                         argv[0]);
            return 2;
        }
        return convertToPng(argv[2], argv[3]);
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
    const char* device = "auto";  // -device auto|cpu|gpu (auto = GPU when it helps)
    bool wavefront = false;       // -wavefront: streaming GPU backend (else megakernel)
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
    char cliModePrescan = 0;
    for (int i = 1; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "-mode")) cliModePrescan = argv[i + 1][0];
        else if (!std::strcmp(argv[i], "-on-unsupported")) {
            std::string v = argv[i + 1];
            if      (v == "fallback" || v == "fall") g_onUnsupported = OnUnsupported::Fallback;
            else if (v == "strip"    || v == "ignore") g_onUnsupported = OnUnsupported::Strip;
            else                                     g_onUnsupported = OnUnsupported::Error;
        }
    }

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
        // The prefer/else resolver asks this predicate whether a branch renders; when the
        // policy is fallback/strip we accept every branch (the policy handles it later at
        // render time), so the FIRST/most-preferred branch always wins.
        ftsl::SupportFn supportFn = (g_onUnsupported == OnUnsupported::Error)
            ? ftsl::SupportFn([cliModePrescan](const ftsl::Loaded& L) -> const char* {
                  return sceneModeUnsupported(L, cliModePrescan);
              })
            : ftsl::SupportFn{};
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
        else if (!std::strcmp(argv[i], "-mode") && i + 1 < argc) { mode = argv[++i][0]; modeFromCli = true; }
        else if (!std::strcmp(argv[i], "-on-unsupported") && i + 1 < argc) { ++i; /* pre-scanned into g_onUnsupported */ }
        else if (!std::strcmp(argv[i], "-pmradius") && i + 1 < argc) g_pmRadiusAbs = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-pmradiusfrac") && i + 1 < argc) g_pmRadiusFactor = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-pmfg") && i + 1 < argc) { g_pmFinalGather = std::atoi(argv[++i]); if (g_pmFinalGather < 0) g_pmFinalGather = 0; }
        else if (!std::strcmp(argv[i], "-savemap") && i + 1 < argc) g_pmapSave = argv[++i];
        else if (!std::strcmp(argv[i], "-loadmap") && i + 1 < argc) g_pmapLoad = argv[++i];
        else if (!std::strcmp(argv[i], "-sppmalpha") && i + 1 < argc) g_sppmAlpha = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-vcmalpha") && i + 1 < argc) g_vcmAlpha = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-heroc") && i + 1 < argc) {
            g_heroC = std::atoi(argv[++i]);
            if (g_heroC < 1) g_heroC = 1;
            if (g_heroC > hero::kHeroMax) g_heroC = hero::kHeroMax;
        }
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
        else if (!std::strcmp(argv[i], "-device") && i + 1 < argc) device = argv[++i];
        else if (!std::strcmp(argv[i], "-wavefront")) wavefront = true;
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
        else if (!std::strcmp(argv[i], "-validate-grammar")) ftsl_shim::enabled_flag() = true;
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
            isomesh::Mesh m = isomesh::marchImplicit(im, mo);
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
            airtight::Report r = airtight::check(im, airtightRays, 0x9E3779B97F4A7C15ull + k);
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
            isomesh::Mesh m = isomesh::marchImplicit(scene.implicits[k], mo);
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
            ms.reserve(rasterBench);
            for (int it = 0; it < rasterBench && !g_stopRequested; ++it) {
                auto t0 = std::chrono::steady_clock::now();
                img = rasterOne(rc.cam, W, H, ev, autoExp, nullptr);
                ms.push_back(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0).count());
                if (g_showWindow) {
                    if (!g_liveWin) g_liveWin = std::make_unique<LiveWindow>(W, H, g_windowTitle.c_str());
                    g_liveWin->update(W, H, img);
                    if (g_liveWin->closed()) g_stopRequested = 1;
                }
            }
#ifdef HAVE_CUDA
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
#ifdef HAVE_CUDA
            {
                raster_cuda::Prof p = raster_cuda::profTake();
                if (p.frames > 0) {
                    double f = 1.0 / p.frames;
                    std::printf("[raster-bench] GPU per-pass avg ms: clearvis %.2f  project %.2f  "
                                "raster %.2f  shade %.2f  clear %.2f  expose+encode %.2f  "
                                "download %.2f\n",
                                p.clearvis_ms * f, p.project_ms * f, p.raster_ms * f,
                                p.shade_ms * f, p.clear_ms * f, p.expose_ms * f,
                                p.download_ms * f);
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
            const double kYaw   = 1.6;               // max yaw   rad/sec at full pointer deflection
            const double kPitch = 1.2;               // max pitch rad/sec at full pointer deflection
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
            // Current free pose as a control-point frame.
            auto poseNow = [&]() -> PathFrame { return PathFrame{eye, fwd, worldUp, fovY}; };
            // Regenerate the preview path (explorePath) + timeline from the control points.
            auto rebuildPath = [&]() {
                ptSpeed.resize(editPts.size(), 1.0);   // safety: keep the speed track sized to the points
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
                // Reset: in free flight, restore the authored eye + look direction; while
                // locked to the path, jump back to the start of the timeline and pause.
                if (nav.reset) {
                    if (pathMode) { pathPos = 0.0; playing = false; }
                    else {
                        eye = eye0; fwd = norml(tgt0 - eye0);
                        lookDist = std::sqrt(dot(tgt0 - eye0, tgt0 - eye0));
                        if (lookDist < 1e-4) lookDist = sceneR;
                    }
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
                        for (const auto& g : got) { editPts.push_back(g); ptSpeed.push_back(1.0); }
                        rebuildPath();
                        std::printf("[editor] recorded %zu control points from %zu raw samples (tol %.4g, %s)\n",
                                    got.size(), recRawBuf.size(), recTol, recRaw ? "raw" : "simplified");
                    }
                    g_liveWin->setEditState(recording, (int)editPts.size());
                    std::fflush(stdout); changed = true;
                }
                if (nav.addPoint) {
                    editPts.push_back(poseNow()); ptSpeed.push_back(1.0);
                    rebuildPath();
                    g_liveWin->setEditState(recording, (int)editPts.size());
                    std::printf("[editor] +point %zu at eye(%s)\n", editPts.size(), fmt3(eye).c_str());
                    std::fflush(stdout); changed = true;
                }
                if (nav.insPoint) {
                    if (editPts.size() < 2) { editPts.push_back(poseNow()); ptSpeed.push_back(1.0); }
                    else {
                        // Insert between the two control points bracketing the current scrub
                        // position. bracket() normalizes by the ACTUAL explorePath length, so this
                        // is correct for a freshly-loaded curve (whose frame count isn't a multiple
                        // of kPreviewPerSeg) as well as an editor-rebuilt preview.
                        int seg; double fr; bracket(pathPos, seg, fr);
                        seg = std::clamp(seg, 0, (int)editPts.size() - 2);
                        editPts.insert(editPts.begin() + seg + 1, poseNow());
                        ptSpeed.insert(ptSpeed.begin() + std::min((size_t)seg + 1, ptSpeed.size()), 1.0);
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
                    if ((size_t)best < ptSpeed.size()) ptSpeed.erase(ptSpeed.begin() + best);
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

                Vec3 tgt = eye + fwd * lookDist;   // look_at point on the view ray (for readout/print)
                if (changed) {
                    Camera c; c.projection = proj;
                    c.lookAt(eye, tgt, rUp, rFov, VW, VH);
                    std::vector<uint8_t> img =
                        rasterOne(c, VW, VH, ev, autoExp, nullptr);
                    drawOverlay(c, VW, VH, img);   // control-point markers + live spline polyline
                    g_liveWin->update(VW, VH, img);
                    g_liveWin->setTitle(g_windowTitle + "  \xE2\x80\x94  eye(" + fmt3(eye) +
                                        ")  dir(" + fmt3(fwd) + ")");
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
                }
                // Sleep only when truly idle; while a throttle key is held, the mouse is
                // steering, or the path is auto-playing we loop at full raster speed for
                // smooth continuous motion.
                if (!nav.any() && !playing)
                    std::this_thread::sleep_for(std::chrono::milliseconds(15));
            }
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
                    mf = renderBackwardCuda(scene, mc.cam, W, H, meterSpp, diffraction);
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
                    meterPmap.build(radius);
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
                    ref = renderBackwardCuda(scene, mc.cam, W, H, meterSpp, diffraction);
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
        if (meterGpu && g_pmFinalGather == 0 && cudaPhotonMapSupported(scene)) {
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
                                          nullptr, nullptr, g_heroC);
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
        if ((wantGpu || wantAuto) && cudaAvailable() && cudaForwardSupported(scene) && !g_beamGather)
            useGpuForward = true;   // -beams per-camera resample is CPU-only for now
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

        // One accumulation chunk of `batchN` photons across the whole group.
        auto runBatch = [&](long long batchN) {
            EnergyReport e;
            std::vector<Film> films;
#ifdef HAVE_CUDA
            if (useGpuForward)
                films = renderForwardSharedCuda(scene, cams, rxs, rys, batchN, e, diffraction,
                                                groupMode, (unsigned long long)accN, wavefront, g_heroC);
            else
#endif
                films = renderForwardShared(scene, cams, rxs, rys, batchN, nThreads, e, diffraction,
                                            groupMode == 'A', (unsigned long long)accN, g_beamGather);
            for (int c = 0; c < nc; ++c) acc[c].merge(films[c]);
            accN += batchN;
            accE.emitted += e.emitted; accE.absorbed += e.absorbed; accE.sensor += e.sensor;
            accE.escaped += e.escaped; accE.residual += e.residual;
        };

        // Write every camera's image (+ optional .ftbuf). `quiet` suppresses the per-file
        // announce for intermediate saves; these groups are per-frame-auto-exposed
        // (expGroup < 0) so no shared exposure anchor is involved.
        auto writeOut = [&](bool quiet) {
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
            if ((wantGpu || wantAuto) && g_pmFinalGather == 0 && allPinhole &&
                cudaAvailable() && cudaPhotonMapSupported(scene)) {
                std::vector<Camera> cams; std::vector<int> rxs, rys;
                for (int i : idx) { cams.push_back(toRender[i].cam); rxs.push_back(toRender[i].res); rys.push_back(toRender[i].resY); }
                std::printf("[camera] shared photon map (mode M) on %s: %zu cameras, %lld "
                            "photons, radius %.4g (light=%s) ...\n",
                            cudaDeviceName(), cams.size(), N, radius, lightLabel);
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
                                          g_pmapSave.empty() ? nullptr : g_pmapSave.c_str(), g_heroC);
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
        pm.build(radius);
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
                           preview, intervalSec, noiseTarget, wavefront, anchor);
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
    // Tear the CUDA context down synchronously, in-process, on EVERY exit path (normal
    // return or exception). Leaving it for the driver to reclaim implicitly after main()
    // returns triggers an asynchronous nvlddmkm DPC teardown that, on buggy driver
    // builds, can fault and bugcheck the machine (the "reboot a few seconds after the
    // window closed" BSOD). Draining + resetting here closes that window.
    int rc;
    try {
        // Resident preview server (-serve): keep the process alive and re-render each
        // scene path streamed on stdin. Find the -in value slot to swap per frame.
        bool serve = false;
        int inValPos = -1;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "-serve")) serve = true;
            else if (!std::strcmp(argv[i], "-in") && i + 1 < argc) inValPos = i + 1;
        }
        rc = serve ? runServe(argc, argv, inValPos) : run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        rc = 1;
    }
    // -keepwindow / -hold: keep the finished image on screen. The live window runs its
    // own UI thread, so we just block here until the user closes it (or it's already gone)
    // rather than letting process exit tear it down the instant the render completes.
    if (g_keepWindow && g_liveWin && !g_liveWin->closed()) {
        std::printf("[window] render done — close the preview window to exit.\n");
        std::fflush(stdout);
        while (!g_liveWin->closed())
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
#ifdef HAVE_CUDA
    cudaGracefulShutdown();
#endif
    return rc;
}
