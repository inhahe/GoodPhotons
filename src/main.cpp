// Forward spectral photon tracer — Phase 0 (+ model B camera).
//   -mode A : contact sensor on the front wall (pure forward catch, no lens)
//   -mode B : pinhole camera outside the box, light-tracing splat (default)
// Both trace identical physics; B just also connects each vertex to the camera.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <thread>
#include "scene.h"
#include "camera.h"
#include "render.h"
#include "lights.h"

// Resolve a -light name to an emission SPD. "bbNNNN" means a Planckian at NNNN K
// (e.g. bb3200). Unknown names fall back to a 6500 K blackbody.
static Spectrum resolveLight(const char* name) {
    if (!name) return blackbody(6500.0);
    if (!std::strncmp(name, "bb", 2) && name[2]) {
        double k = std::atof(name + 2);
        if (k > 0) return blackbody(k);
    }
    if (!std::strcmp(name, "sun"))          return sunlight();
    if (!std::strcmp(name, "daylight") ||
        !std::strcmp(name, "d65"))          return daylight(6504.0);
    if (!std::strcmp(name, "a") ||
        !std::strcmp(name, "incandescent")) return illuminantA();
    if (!std::strcmp(name, "led"))          return ledWhite(0.3);
    if (!std::strcmp(name, "led-warm"))     return ledWhite(1.0);
    if (!std::strcmp(name, "fluorescent") ||
        !std::strcmp(name, "cfl"))          return fluorescent();
    return blackbody(6500.0);
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
    glass.ior = iorSF10();                                       s.mats.push_back(glass); // 1

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
    s.finalizeTris();

    // Collimated white beam entering the left face, travelling +x.
    s.collimated = true;
    s.beamDir = {1, 0, 0};
    s.lightOrigin = {0.05, 0.54, 0.49};
    s.lightU = {0, 0.03, 0};      // thin pencil cross-section
    s.lightV = {0, 0, 0.03};
    s.lightNormal = {1, 0, 0};
    s.lightArea = 0.03 * 0.03;
    s.lightSpd.build(constantSpectrum(1.0), 1.0); // equal-energy -> even rainbow
    s.lightEmitIntegral = s.lightSpd.integral;
    return s;
}

// mode 'A' builds a sensor front wall; mode 'B' leaves the front open.
static Scene buildCornell(int res, char mode, const Spectrum& lightSpd) {
    Scene s;
    Material white; white.reflect = whiteWall(0.75);            s.mats.push_back(white); // 0
    Material red;   red.reflect   = redWall();                   s.mats.push_back(red);   // 1
    Material green; green.reflect = greenWall();                 s.mats.push_back(green); // 2
    Material light; light.reflect = constantSpectrum(0.0);
    light.emit = lightSpd; light.isLight = true;                 s.mats.push_back(light); // 3
    Material glass; glass.type = MatType::Dielectric;
    glass.ior = iorSF10();                                       s.mats.push_back(glass); // 4

    addQuad(s, {0,0,0},{1,0,0},{1,0,1},{0,0,1}, 0);            // floor
    addQuad(s, {0,1,0},{0,1,1},{1,1,1},{1,1,0}, 0);            // ceiling
    addQuad(s, {0,0,0},{0,1,0},{1,1,0},{1,0,0}, 0);            // back
    addQuad(s, {0,0,0},{0,0,1},{0,1,1},{0,1,0}, 1);            // left (red)
    addQuad(s, {1,0,0},{1,1,0},{1,1,1},{1,0,1}, 2);            // right (green)
    if (mode == 'A')
        addQuad(s, {0,0,1},{1,0,1},{1,1,1},{0,1,1}, 0, /*sensor*/0); // front = sensor
    // mode 'B': front left open so the external camera can see in.

    const double lx0 = 0.35, lx1 = 0.65, lz0 = 0.35, lz1 = 0.65, ly = 0.999;
    addQuad(s, {lx0,ly,lz0},{lx1,ly,lz0},{lx1,ly,lz1},{lx0,ly,lz1}, 3);

    // Dispersive glass sphere -> casts a spectral caustic on the floor.
    s.spheres.push_back(Sphere{{0.5, 0.32, 0.4}, 0.25, 4});

    s.finalizeTris();

    if (mode == 'A') {
        s.sensor.origin = {0,0,1}; s.sensor.uAxis = {1,0,0}; s.sensor.vAxis = {0,1,0};
        s.sensor.film.resX = res; s.sensor.film.resY = res; s.sensor.alloc();
    }

    s.lightOrigin = {lx0, ly, lz0};
    s.lightU = {lx1 - lx0, 0, 0};
    s.lightV = {0, 0, lz1 - lz0};
    s.lightNormal = {0, -1, 0};
    s.lightArea = (lx1 - lx0) * (lz1 - lz0);
    s.lightSpd.build(s.mats[3].emit, 1.0);
    s.lightEmitIntegral = s.lightSpd.integral;
    return s;
}

// Cornell box (model-B only) with the reflective material types side by side:
// a near-perfect mirror, a rough glossy metal, and a half-mirror (beamsplitter).
// All three are specular, so under pure light tracing (model B) they appear BLACK
// from the camera: a specular vertex has zero probability of connecting to the
// pinhole (the SDS limitation, same as the glass sphere in the Cornell scene).
// The physics is still exercised — photons reflect off them and illuminate the
// diffuse walls, and energy conserves — but seeing the spheres' mirrored image
// directly requires the future camera-side ray path (or model A's contact catch).
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

    s.finalizeTris();

    s.lightOrigin = {lx0, ly, lz0};
    s.lightU = {lx1 - lx0, 0, 0};
    s.lightV = {0, 0, lz1 - lz0};
    s.lightNormal = {0, -1, 0};
    s.lightArea = (lx1 - lx0) * (lz1 - lz0);
    s.lightSpd.build(s.mats[3].emit, 1.0);
    s.lightEmitIntegral = s.lightSpd.integral;
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

static void writePPM(const char* path, const Film& f, double N) {
    const int W = f.resX, H = f.resY;
    std::vector<Vec3> lin((size_t)W * H);
    double norm = 1.0 / (N * cieYIntegral());
    std::vector<double> lum; lum.reserve((size_t)W * H);
    for (size_t i = 0; i < lin.size(); ++i) {
        lin[i] = xyzToLinearSrgb(f.xyz[i] * norm);
        lum.push_back(std::max({lin[i].x, lin[i].y, lin[i].z, 0.0}));
    }
    std::vector<double> sorted = lum; std::sort(sorted.begin(), sorted.end());
    double p99 = sorted[(size_t)(0.99 * (sorted.size() - 1))];
    double exposure = (p99 > 0) ? 0.9 / p99 : 1.0;

    std::vector<uint8_t> img((size_t)W * H * 3);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        size_t src = (size_t)(H - 1 - y) * W + x;       // flip so +y is image-top
        size_t dst = ((size_t)y * W + x) * 3;
        for (int c = 0; c < 3; ++c) {
            double v = (&lin[src].x)[c] * exposure;
            img[dst + c] = (uint8_t)std::clamp(srgbGamma(v) * 255.0 + 0.5, 0.0, 255.0);
        }
    }
    std::ofstream fo(path, std::ios::binary);
    fo << "P6\n" << W << ' ' << H << "\n255\n";
    fo.write((const char*)img.data(), (std::streamsize)img.size());
    std::printf("wrote %s (%dx%d), auto-exposure=%.3g\n", path, W, H, exposure);
}

int main(int argc, char** argv) {
    long long N = 2'000'000;
    int res = 256;
    char mode = 'B';
    int nThreads = (int)std::thread::hardware_concurrency();
    const char* out = "cornell.ppm";
    const char* sceneName = "cornell";
    const char* lightName = "bb6500";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) N = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "-r") && i + 1 < argc) res = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "-mode") && i + 1 < argc) mode = argv[++i][0];
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) nThreads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-scene") && i + 1 < argc) sceneName = argv[++i];
        else if (!std::strcmp(argv[i], "-light") && i + 1 < argc) lightName = argv[++i];
    }
    if (nThreads < 1) nThreads = 1;
    bool prism     = !std::strcmp(sceneName, "prism");
    bool materials = !std::strcmp(sceneName, "materials");

    selfTestColor();

    Scene scene = prism     ? buildPrism(res)
                : materials ? buildMaterials(res, resolveLight(lightName))
                            : buildCornell(res, mode, resolveLight(lightName));
    Camera cam;
    if (mode == 'B') {
        if (prism) cam.lookAt({0.5, 0.5, 2.4}, {0.5, 0.45, 0.5}, {0, 1, 0}, 45.0, res, res);
        else       cam.lookAt({0.5, 0.5, 2.7}, {0.5, 0.5, 0.5}, {0, 1, 0}, 40.0, res, res);
    }
    const bool modeB = (mode == 'B');

    std::printf("mode %c: tracing %lld photons at %dx%d on %d threads (light=%s) ...\n",
                mode, N, res, res, nThreads, prism ? "beam" : lightName);

    // Per-thread films + energy reports, merged after. Each thread gets a
    // distinct RNG stream so photons are independent.
    std::vector<Film> films(nThreads);
    std::vector<EnergyReport> reports(nThreads);
    for (auto& f : films) { f.resX = res; f.resY = res; f.alloc(); }

    auto worker = [&](int tid) {
        Renderer r;
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 1, 0x9e3779b97f4a7c15ULL ^ (uint64_t)tid);
        long long lo = N * tid / nThreads, hi = N * (tid + 1) / nThreads;
        Film* sensorFilm = modeB ? nullptr : &films[tid];
        Camera* camPtr   = modeB ? &cam : nullptr;
        Film* camFilm    = modeB ? &films[tid] : nullptr;
        for (long long i = lo; i < hi; ++i)
            r.tracePhoton(scene, camPtr, sensorFilm, camFilm, rng, reports[tid]);
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    // Merge.
    Film out_film; out_film.resX = res; out_film.resY = res; out_film.alloc();
    EnergyReport e;
    for (int t = 0; t < nThreads; ++t) {
        out_film.merge(films[t]);
        e.emitted += reports[t].emitted; e.absorbed += reports[t].absorbed;
        e.sensor += reports[t].sensor;   e.escaped += reports[t].escaped;
        e.residual += reports[t].residual;
    }

    double tot = e.absorbed + e.sensor + e.escaped + e.residual;
    std::printf("[energy] absorbed=%.4f sensor=%.4f escaped=%.4f residual=%.4f (sum/emitted=%.6f)\n",
                e.absorbed / e.emitted, e.sensor / e.emitted, e.escaped / e.emitted,
                e.residual / e.emitted, tot / e.emitted);

    writePPM(out, out_film, (double)N);
    return 0;
}
