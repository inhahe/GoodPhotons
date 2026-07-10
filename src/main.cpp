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
#include "backward.h"
#include "lights.h"
#include "mesh.h"

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
static Scene buildCornell(int res, char mode, const Spectrum& lightSpd,
                          const char* meshPath = nullptr, double meshScale = 1.0,
                          bool diffuseSphere = false) {
    Scene s;
    Material white; white.reflect = whiteWall(0.75);            s.mats.push_back(white); // 0
    Material red;   red.reflect   = redWall();                   s.mats.push_back(red);   // 1
    Material green; green.reflect = greenWall();                 s.mats.push_back(green); // 2
    Material light; light.reflect = constantSpectrum(0.0);
    light.emit = lightSpd; light.isLight = true;                 s.mats.push_back(light); // 3
    Material glass; glass.type = MatType::Dielectric;
    glass.ior = iorSF10();                                       s.mats.push_back(glass); // 4
    Material mesh;  mesh.reflect  = whiteWall(0.8);              s.mats.push_back(mesh);  // 5 (diffuse)

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

    // A loaded mesh (diffuse) replaces the glass sphere when -mesh is given;
    // otherwise the dispersive glass sphere casts a spectral caustic on the floor.
    if (meshPath && meshPath[0]) {
        loadObj(s, meshPath, /*mat*/5, /*translate*/{0.5, 0.4, 0.5}, meshScale);
    } else {
        // Diffuse sphere (mat 5) for the reference/validation modes so there is no
        // specular black-glass mismatch; the dispersive glass sphere (mat 4)
        // otherwise casts a spectral caustic on the floor.
        s.spheres.push_back(Sphere{{0.5, 0.32, 0.4}, 0.25, diffuseSphere ? 5 : 4});
    }

    s.build();

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

// Forward photon trace (models A/B/C) into a merged film. Accumulates the energy
// report across threads. Factored out so mode V can reuse it alongside the
// backward reference.
static Film renderForward(const Scene& scene, const Camera* cam, int res, long long N,
                          int nThreads, bool forwardCatch, bool useCamera, EnergyReport& eOut) {
    std::vector<Film> films(nThreads);
    std::vector<EnergyReport> reports(nThreads);
    for (auto& f : films) { f.resX = res; f.resY = res; f.alloc(); }

    auto worker = [&](int tid) {
        Renderer r; r.forwardCatch = forwardCatch;
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 1, 0x9e3779b97f4a7c15ULL ^ (uint64_t)tid);
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

    Film out; out.resX = res; out.resY = res; out.alloc();
    for (int t = 0; t < nThreads; ++t) { out.merge(films[t]); }
    for (auto& rp : reports) {
        eOut.emitted += rp.emitted; eOut.absorbed += rp.absorbed; eOut.sensor += rp.sensor;
        eOut.escaped += rp.escaped; eOut.residual += rp.residual;
    }
    return out;
}

// Backward reference: `spp` samples per pixel, threads render disjoint row bands
// of a shared film (no shared-pixel writes, so no race).
static Film renderBackward(const Scene& scene, const Camera& cam, int res,
                           long long spp, int nThreads) {
    Film out; out.resX = res; out.resY = res; out.alloc();
    auto worker = [&](int tid) {
        BackwardRenderer br;
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 7, 0xD1B54A32D192ED03ULL ^ (uint64_t)tid);
        int y0 = res * tid / nThreads, y1 = res * (tid + 1) / nThreads;
        br.renderRows(scene, cam, out, y0, y1, spp, rng);
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    return out;
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
    for (size_t i = 0; i < n; ++i) {
        Vec3 f = fwd.xyz[i] * invF, r = ref.xyz[i] * invR;
        Vec3 d = f - r * s; num += dot(d, d);
    }
    double rmse = (sff > 0) ? std::sqrt(num / sff) : 0.0;
    std::printf("[validate] best-fit scale (backward->forward) = %.6g\n", s);
    std::printf("[validate] relative RMSE after scale = %.3f%%  (lower = better agreement)\n",
                100.0 * rmse);
    std::printf("[validate] %s\n", rmse < 0.05
                ? "PASS: forward light tracer agrees with backward reference."
                : "review: residual above 5% — increase -n/-spp, or investigate transport.");
}

int main(int argc, char** argv) {
    long long N = 2'000'000;
    int res = 256;
    char mode = 'B';
    int nThreads = (int)std::thread::hardware_concurrency();
    const char* out = "cornell.ppm";
    const char* sceneName = "cornell";
    const char* lightName = "bb6500";
    double apertureR = 0.02;  // mode C aperture radius (scene units)
    bool checkBvhOnly = false;
    bool bvhStatsOnly = false;
    const char* meshPath = nullptr;
    double meshScale = 1.0;
    long long spp = 256;      // backward reference samples/pixel (modes R and V)
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) N = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "-r") && i + 1 < argc) res = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "-mode") && i + 1 < argc) mode = argv[++i][0];
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) nThreads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-scene") && i + 1 < argc) sceneName = argv[++i];
        else if (!std::strcmp(argv[i], "-light") && i + 1 < argc) lightName = argv[++i];
        else if (!std::strcmp(argv[i], "-aperture") && i + 1 < argc) apertureR = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-checkbvh")) checkBvhOnly = true;
        else if (!std::strcmp(argv[i], "-bvhstats")) bvhStatsOnly = true;
        else if (!std::strcmp(argv[i], "-mesh") && i + 1 < argc) meshPath = argv[++i];
        else if (!std::strcmp(argv[i], "-meshscale") && i + 1 < argc) meshScale = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-spp") && i + 1 < argc) spp = std::atoll(argv[++i]);
    }
    if (nThreads < 1) nThreads = 1;
    bool prism     = !std::strcmp(sceneName, "prism");
    bool materials = !std::strcmp(sceneName, "materials");

    selfTestColor();

    // Modes R (backward reference) and V (validate: forward vs backward) need an
    // all-diffuse scene so the known model-B specular limitation doesn't pollute
    // the comparison — use a diffuse sphere when no mesh is supplied.
    const bool refMode = (mode == 'R' || mode == 'V');
    Scene scene = prism     ? buildPrism(res)
                : materials ? buildMaterials(res, resolveLight(lightName))
                            : buildCornell(res, mode, resolveLight(lightName), meshPath,
                                           meshScale, /*diffuseSphere*/refMode);

    if (checkBvhOnly) {
        // Bound the linear-reference work (~O(rays * prims)) so the self-test
        // stays fast even for big meshes: ~5e8 primitive tests, clamped.
        long long prims = (long long)scene.tris.size() + (long long)scene.spheres.size();
        long long rays = 500'000'000LL / (prims > 0 ? prims : 1);
        rays = std::clamp(rays, 20'000LL, 2'000'000LL);
        return checkBvh(scene, rays) == 0 ? 0 : 1;
    }
    if (bvhStatsOnly) { bvhStats(scene, 500'000); return 0; }
    // mode A: contact sensor (no camera). mode B: connect/splat. mode C: finite-
    // aperture forward catch. mode R: backward reference. mode V: validate B vs R.
    const bool useCamera    = (mode == 'B' || mode == 'C' || refMode);
    const bool forwardCatch = (mode == 'C');
    Camera cam;
    if (useCamera) {
        if (prism) cam.lookAt({0.5, 0.5, 2.4}, {0.5, 0.45, 0.5}, {0, 1, 0}, 45.0, res, res);
        else       cam.lookAt({0.5, 0.5, 2.7}, {0.5, 0.5, 0.5}, {0, 1, 0}, 40.0, res, res);
        cam.apertureR = apertureR;
    }

    // --- Backward reference (mode R) and validation (mode V) ---
    if (refMode) {
        std::printf("mode %c: backward reference %lld spp at %dx%d on %d threads (light=%s) ...\n",
                    mode, spp, res, res, nThreads, lightName);
        Film ref = renderBackward(scene, cam, res, spp, nThreads);
        if (mode == 'R') { writePPM(out, ref, (double)spp); return 0; }

        // mode V: also run the forward light tracer (model B) and compare.
        std::printf("mode V: forward light tracer %lld photons for cross-check ...\n", N);
        EnergyReport e;
        Film fwd = renderForward(scene, &cam, res, N, nThreads,
                                 /*forwardCatch*/false, /*useCamera*/true, e);
        double tot = e.absorbed + e.sensor + e.escaped + e.residual;
        std::printf("[energy] absorbed=%.4f sensor=%.4f escaped=%.4f residual=%.4f (sum/emitted=%.6f)\n",
                    e.absorbed / e.emitted, e.sensor / e.emitted, e.escaped / e.emitted,
                    e.residual / e.emitted, tot / e.emitted);
        compareFilms(fwd, N, ref, spp);
        writePPM("validate_forward.ppm", fwd, (double)N);
        writePPM("validate_backward.ppm", ref, (double)spp);
        return 0;
    }

    std::printf("mode %c: tracing %lld photons at %dx%d on %d threads (light=%s) ...\n",
                mode, N, res, res, nThreads, prism ? "beam" : lightName);

    EnergyReport e;
    Film out_film = renderForward(scene, &cam, res, N, nThreads, forwardCatch, useCamera, e);

    double tot = e.absorbed + e.sensor + e.escaped + e.residual;
    std::printf("[energy] absorbed=%.4f sensor=%.4f escaped=%.4f residual=%.4f (sum/emitted=%.6f)\n",
                e.absorbed / e.emitted, e.sensor / e.emitted, e.escaped / e.emitted,
                e.residual / e.emitted, tot / e.emitted);

    writePPM(out, out_film, (double)N);
    return 0;
}
