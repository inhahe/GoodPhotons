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
#include <memory>
#include "scene.h"
#include "isomesh.h"            // -export-mesh: isosurface -> watertight OBJ (marching tetrahedra)
#include "camera.h"
#include "render.h"
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

    bool pass = passA && passB && passW && passC;
    std::printf("[checkupsample] round-trip max error (excl. white) = %.5f  (%s)\n", maxErr, passA ? "ok" : "BAD");
    std::printf("[checkupsample] reflectance in [0,1]  (%s)\n", passB ? "ok" : "BAD");
    std::printf("[checkupsample] pure-white residual = %.5f (<0.02 expected)  (%s)\n", whiteErr, passW ? "ok" : "BAD");
    std::printf("[checkupsample] mid-grey round-trip = %.6f  (%s)\n", greyErr, passC ? "ok" : "BAD");
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

// SPPM (mode S) radius-shrink rate alpha (Hachisuka 2008; CLI -sppmalpha). Smaller =
// faster radius shrink (less bias sooner, more variance); 0.7 is the paper default. The
// initial radius R0 reuses the mode-M -pmradius / -pmradiusfrac controls above.
static double g_sppmAlpha = 0.7;

// VCM (mode U) radius-shrink rate alpha (Georgiev 2012; CLI -vcmalpha). The per-pass merge
// radius follows r_i = R0 * i^((alpha-1)/2), i = pass index (1-based); 0.75 is the
// SmallVCM default. Initial radius R0 reuses the mode-M -pmradius / -pmradiusfrac controls.
static double g_vcmAlpha = 0.75;

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
        return renderForwardCuda(scene, *cam, resX, resY, N, eOut, diffraction, camMode, seedBase, wavefront);
    }
#else
    (void)useGpu; (void)wavefront;
#endif
    std::vector<Film> films(nThreads);
    std::vector<EnergyReport> reports(nThreads);
    for (auto& f : films) { f.resX = resX; f.resY = resY; f.alloc(); }

    auto worker = [&](int tid) {
        Renderer r; r.forwardCatch = forwardCatch; r.lensMode = lensMode; r.diffraction = diffraction;
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 1,
                            (0x9e3779b97f4a7c15ULL ^ (uint64_t)tid) + seedBase * 0x9e3779b97f4a7c15ULL);
        long long lo = N * tid / nThreads, hi = N * (tid + 1) / nThreads;
        Film* sensorFilm = useCamera ? nullptr : &films[tid];
        const Camera* camPtr = useCamera ? cam : nullptr;
        Film* camFilm = useCamera ? &films[tid] : nullptr;
        for (long long i = lo; i < hi; ++i)
            r.tracePhoton(scene, camPtr, sensorFilm, camFilm, rng, reports[tid]);
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
                                             bool lensMode = false) {
    int nc = (int)cams.size();
    // Per-thread × per-camera films (each thread accumulates into its own copies to
    // avoid shared-pixel races; merged per camera at the end).
    std::vector<std::vector<Film>> films(nThreads, std::vector<Film>(nc));
    std::vector<EnergyReport> reports(nThreads);
    for (int t = 0; t < nThreads; ++t)
        for (int c = 0; c < nc; ++c) { films[t][c].resX = resX[c]; films[t][c].resY = resY[c]; films[t][c].alloc(); }

    auto worker = [&](int tid) {
        Renderer r; r.forwardCatch = false; r.lensMode = lensMode; r.diffraction = diffraction;
        // Identical seeding to renderForward (seedBase 0). For model B this makes each
        // camera's shared film bit-identical to its standalone single-camera render; for
        // model A the aperture draws perturb the stream, so it matches in distribution.
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 1, 0x9e3779b97f4a7c15ULL ^ (uint64_t)tid);
        long long lo = N * tid / nThreads, hi = N * (tid + 1) / nThreads;
        std::vector<CamTarget> targets(nc);
        for (int c = 0; c < nc; ++c) { targets[c].cam = &cams[c]; targets[c].film = &films[tid][c]; }
        for (long long i = lo; i < hi; ++i)
            r.tracePhoton(scene, targets.data(), nc, /*sensorFilm*/nullptr, rng, reports[tid]);
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
// `seedOffset` decorrelates the RNG stream so a chunked/progressive render can call
// this repeatedly (each chunk with a distinct offset) and merge the SUM films for an
// independent, ever-refining estimate. seedOffset==0 reproduces the original stream.
static Film renderBackward(const Scene& scene, const Camera& cam, int resX, int resY,
                           long long spp, int nThreads, bool diffraction = true,
                           unsigned long long seedOffset = 0) {
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    auto worker = [&](int tid) {
        BackwardRenderer br; br.diffraction = diffraction;
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 7,
                            0xD1B54A32D192ED03ULL ^ (uint64_t)tid
                              ^ (seedOffset * 0x9E3779B97F4A7C15ULL));
        int y0 = resY * tid / nThreads, y1 = resY * (tid + 1) / nThreads;
        br.renderRows(scene, cam, out, y0, y1, spp, rng);
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
                       unsigned long long seedOffset = 0) {
    std::vector<Film> camBands(nThreads), splatBands(nThreads);
    auto worker = [&](int tid) {
        bdpt::BdptRenderer br; br.maxDepth = maxDepth; br.diffraction = diffraction;
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 11,
                            0x9E3779B97F4A7C15ULL ^ (uint64_t)tid
                              ^ (seedOffset * 0xD1B54A32D192ED03ULL));
        Film& cf = camBands[tid]; cf.resX = resX; cf.resY = resY; cf.alloc();
        Film& sf = splatBands[tid]; sf.resX = resX; sf.resY = resY; sf.alloc();
        int y0 = resY * tid / nThreads, y1 = resY * (tid + 1) / nThreads;
        br.renderRows(scene, cam, cf, sf, y0, y1, spp, rng);
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
static std::unique_ptr<LiveWindow> g_liveWin;
static void liveWindowUpdate(const Film& f, double N, double expComp, bool absolute) {
    if (!g_showWindow || N <= 0.0) return;
    if (!g_liveWin)
        g_liveWin = std::make_unique<LiveWindow>(f.resX, f.resY, "ftrace — live preview");
    // Per-frame auto-expose (nullptr anchor) so the live view tracks the converging
    // image the same way the ANSI preview does.
    std::vector<uint8_t> rgb = filmToRgb8(f, N, expComp, absolute, nullptr);
    g_liveWin->update(f.resX, f.resY, rgb);
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
    // On resume the loaded film already holds `sampleBase` spp; bias the fresh seeds past
    // it so the continued samples are an independent realization (see SppProgress).
    const unsigned long long seedBias = (unsigned long long)prog->sampleBase;
    Film acc; acc.resX = resX; acc.resY = resY; acc.alloc();
    long long done = 0, chunk = 1;
    while (done < sppTarget) {
        long long c = chunk; if (c > sppTarget - done) c = sppTarget - done;
        auto t0 = clk::now();
        Film f = renderOne(c, seedBias + (unsigned long long)(done + 1));
        acc.merge(f);
        done += c;
        double dt = std::chrono::duration<double>(clk::now() - t0).count();
        if (dt > 1e-4) {                       // retarget chunk toward ~0.4s of work
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
            liveWindowUpdate(*shown, (double)totalSpp, manualExposure, absolute);
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
                                   (uint64_t)acc.spp + 1);
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
            liveWindowUpdate(comp, 1.0, manualExposure, absolute);
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
                              "palette, or oversized multilayer/mix material)";
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
        tracePhotonPass(scene, N, nThreads, diffraction, pm);
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
                         /*maxBounce*/32, (uint64_t)(pass + 1));
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
                    liveWindowUpdate(disp, (double)acc.N, manualExposure, scene.absolute);
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

static int run(int argc, char** argv) {
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
    const char* cameraSel = nullptr; // -camera <name>|all|#N|near=X,Y,Z (FTSL multi-camera select)
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

    // --- FTSL scene file (-in <file>) --------------------------------------
    // Load the scene from a file *before* parsing the rest of argv, so any explicit
    // CLI flag (-n, -r, -mode, -device, -o) still overrides what the file's
    // render {} block specified. Pre-scan for -in only; the full parse follows.
    const char* inFile = nullptr;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "-in") && i + 1 < argc) { inFile = argv[i + 1]; break; }
    ftsl::Loaded ftslScene;
    bool fromFtsl = false;
    if (inFile) {
        std::string ferr;
        if (!ftsl::load(inFile, ftslScene, ferr)) {
            std::fprintf(stderr, "[ftsl] %s\n", ferr.c_str());
            return 1;
        }
        fromFtsl = true;
        std::printf("[ftsl] loaded scene from %s\n", inFile);
        if (ftslScene.photons >= 0)       N = ftslScene.photons;
        if (ftslScene.res > 0)            res = ftslScene.res;
        if (ftslScene.mode)               mode = ftslScene.mode;
        if (!ftslScene.device.empty())    device = ftslScene.device.c_str();
        if (!ftslScene.out.empty())       out = ftslScene.out.c_str();
    }

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) N = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "-r") && i + 1 < argc) {
            res = std::atoi(argv[++i]); resFromCli = true;
            // Optional second numeric token makes a non-square film: `-r W H`.
            if (i + 1 < argc && argv[i + 1][0] != '-' && std::isdigit((unsigned char)argv[i + 1][0]))
                resYCli = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "-mode") && i + 1 < argc) { mode = argv[++i][0]; modeFromCli = true; }
        else if (!std::strcmp(argv[i], "-pmradius") && i + 1 < argc) g_pmRadiusAbs = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-pmradiusfrac") && i + 1 < argc) g_pmRadiusFactor = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-pmfg") && i + 1 < argc) { g_pmFinalGather = std::atoi(argv[++i]); if (g_pmFinalGather < 0) g_pmFinalGather = 0; }
        else if (!std::strcmp(argv[i], "-sppmalpha") && i + 1 < argc) g_sppmAlpha = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-vcmalpha") && i + 1 < argc) g_vcmAlpha = std::atof(argv[++i]);
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
        else if (!std::strcmp(argv[i], "-window")) g_showWindow = true;
        else if (!std::strcmp(argv[i], "-exposure-lock")) forceExposureLock = true;
        else if (!std::strcmp(argv[i], "-interval") && i + 1 < argc) intervalSec = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-resume")) resume = true;
        else if (!std::strcmp(argv[i], "-checkpoint")) wantCheckpointFlag = true;
        else if (!std::strcmp(argv[i], "-in") && i + 1 < argc) ++i; // handled in pre-scan
    }
    if (nThreads < 1) nThreads = 1;
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
            groups.emplace_back("isosurface_" + std::to_string(k), std::move(m));
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
    struct RenderCam { std::string name; Camera cam; char mode; int res; int resY; double exposure; int expGroup; };
    std::vector<RenderCam> toRender;

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
            toRender.push_back({cs->name, c, cmode, cresX, cresY, cs->exposureMul, eg});
        }
    } else {
        // Built-in scene: one camera. Every image-forming mode (A/B/C/P/D/M/S/U/ref)
        // uses the same camera frame; only the old contact-sensor diagnostic did not.
        const bool useCamera = (mode == 'A' || mode == 'B' || mode == 'C' ||
                                mode == 'P' || mode == 'D' || mode == 'M' ||
                                mode == 'S' || mode == 'U' || refMode);
        const int resY = (resYCli > 0) ? resYCli : res;
        Camera c;
        if (useCamera) {
            if (haveView)   c.lookAt(viewEye, viewLook, viewUp, viewFov, res, resY);   // -view overrides the demo camera
            else if (prism) c.lookAt({0.5, 0.5, 2.4}, {0.5, 0.45, 0.5}, {0, 1, 0}, 45.0, res, resY);
            else            c.lookAt({0.5, 0.5, 2.7}, {0.5, 0.5, 0.5}, {0, 1, 0}, 40.0, res, resY);
            c.apertureR = apertureR;
            c.setFocus(focusDist);   // thin lens for the finite-aperture modes A/C (0 = camera obscura)
        }
        toRender.push_back({"", c, mode, res, resY, 0.0, forceExposureLock ? 0 : -1});
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

    // Shared auto-exposure anchors, one per exposure-lock group (see RenderCam.expGroup).
    // The first frame in a group computes its anchor and stores it here; later frames
    // in the same group reuse it (no dolly flicker). A null anchor = per-frame auto.
    std::map<int, double> expAnchors;

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
            useGpuForward = true;
    }
#endif
    (void)useGpuForward;   // only read under HAVE_CUDA; keep CPU-only builds warning-clean
    const bool plainRender = !(timeBudgetSec > 0.0 || noiseTarget > 0.0 || resume ||
                               wantCheckpointFlag || runForever || preview);
    std::vector<int> groupB, groupA, groupM, restIdx;
    for (int i = 0; i < (int)toRender.size(); ++i) {
        const RenderCam& rc = toRender[i];
        bool base = (rc.expGroup < 0) && plainRender;
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
    // A single-camera group has nothing to share — fold it back into the per-camera path.
    if (groupB.size() < 2) { for (int i : groupB) restIdx.push_back(i); groupB.clear(); }
    if (groupA.size() < 2) { for (int i : groupA) restIdx.push_back(i); groupA.clear(); }
    if (groupM.size() < 2) { for (int i : groupM) restIdx.push_back(i); groupM.clear(); }
    std::sort(restIdx.begin(), restIdx.end());

    bool sharedWriteFail = false;
    auto runSharedGroup = [&](const std::vector<int>& idx, char groupMode) {
        if (idx.empty()) return;
        std::vector<Camera> cams; std::vector<int> rxs, rys;
        for (int i : idx) { cams.push_back(toRender[i].cam); rxs.push_back(toRender[i].res); rys.push_back(toRender[i].resY); }
#ifdef HAVE_CUDA
        if (useGpuForward)
            std::printf("[camera] shared model-%c pass: %zu cameras, %lld photons on %s (light=%s) ...\n",
                        groupMode, cams.size(), N, cudaDeviceName(), lightLabel);
        else
#endif
            std::printf("[camera] shared model-%c pass: %zu cameras, %lld photons on %d CPU threads (light=%s) ...\n",
                        groupMode, cams.size(), N, nThreads, lightLabel);
        EnergyReport e;
        std::vector<Film> films;
#ifdef HAVE_CUDA
        if (useGpuForward)
            films = renderForwardSharedCuda(scene, cams, rxs, rys, N, e, diffraction, groupMode, 0, wavefront);
        else
#endif
            films = renderForwardShared(scene, cams, rxs, rys, N, nThreads, e, diffraction, groupMode == 'A');
        double tot = e.absorbed + e.sensor + e.escaped + e.residual;
        if (e.emitted > 0.0)
            std::printf("[energy] absorbed=%.4f sensor=%.4f escaped=%.4f residual=%.4f (sum/emitted=%.6f)\n",
                        e.absorbed / e.emitted, e.sensor / e.emitted, e.escaped / e.emitted,
                        e.residual / e.emitted, tot / e.emitted);
        for (size_t k = 0; k < idx.size(); ++k) {
            const RenderCam& rc = toRender[idx[k]];
            Film disp = films[k];
            addEnvBackground(disp, scene, rc.cam, N);     // directly-viewed sky (env scenes)
            std::string op = outFor(rc.name);
            if (toRender.size() > 1)
                std::printf("[camera] '%s' (mode %c, %dx%d) -> %s\n",
                            rc.name.c_str(), groupMode, rc.res, rc.resY, op.c_str());
            if (!writeFilm(op.c_str(), disp, (double)N, rc.exposure, false, nullptr, scene.absolute))
                sharedWriteFail = true;
        }
    };
    runSharedGroup(groupB, 'B');
    runSharedGroup(groupA, 'A');

    // Shared photon-map pass (mode M): build ONE view-independent map, gather every
    // camera from it. This is where the photon map pays off over per-camera backward
    // tracing — the (expensive) forward photon flight amortizes across all frames.
    auto runSharedPhotonMap = [&](const std::vector<int>& idx) {
        if (idx.empty()) return;
        double radius = (g_pmRadiusAbs > 0.0) ? g_pmRadiusAbs
                                              : scene.sceneRadius * g_pmRadiusFactor;
        std::printf("[camera] shared photon map (mode M): %zu cameras, %lld photons, "
                    "radius %.4g on %d CPU threads (light=%s)%s ...\n",
                    idx.size(), N, radius, nThreads, lightLabel,
                    g_pmFinalGather > 0 ? " [final gather]" : "");
        PhotonMap pm;
        auto tp0 = std::chrono::steady_clock::now();
        tracePhotonPass(scene, N, nThreads, diffraction, pm);
        pm.build(radius);
        double buildSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        std::printf("[camera] photon map: %zu photons from %lld emitted in %.1fs, "
                    "grid %dx%dx%d — gathering %zu cameras ...\n",
                    pm.photons.size(), pm.nEmitted, buildSec, pm.nx, pm.ny, pm.nz, idx.size());
        if (pm.photons.empty())
            std::fprintf(stderr, "[mode M] warning: 0 photons deposited — images "
                                 "will be black.\n");
        for (size_t k = 0; k < idx.size(); ++k) {
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
    return sharedWriteFail ? 1 : 0;
}

// Thin wrapper: turn a fatal configuration error (e.g. an explicit `file:`/`glass:`/
// `illuminant:` reference whose target is missing or malformed — thrown by the
// spectral-library resolver) into a clean message + non-zero exit, instead of a
// silent fall-through to a default illuminant that would render the wrong thing.
int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
