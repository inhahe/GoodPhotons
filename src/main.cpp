// Forward spectral photon tracer — Phase 0.
// Renders a spectral Cornell box onto a contact sensor (camera model A, no lens)
// and reports energy conservation. This validates the physics on CPU before we
// add a real camera, GPU, and the exotic optics.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include "scene.h"
#include "render.h"

static void addQuad(Scene& s, Vec3 a, Vec3 b, Vec3 c, Vec3 d, int mat, int sensorId = -1) {
    Tri t1{a, b, c, mat, sensorId, {}}; s.tris.push_back(t1);
    Tri t2{a, c, d, mat, sensorId, {}}; s.tris.push_back(t2);
}

static Scene buildCornell(int res) {
    Scene s;
    // Materials: 0 white, 1 red, 2 green, 3 light.
    Material white; white.reflect = whiteWall(0.75);            s.mats.push_back(white); // 0
    Material red;   red.reflect   = redWall();                   s.mats.push_back(red);   // 1
    Material green; green.reflect = greenWall();                 s.mats.push_back(green); // 2
    Material light; light.reflect = constantSpectrum(0.0);
    light.emit = blackbody(6500.0); light.isLight = true;        s.mats.push_back(light); // 3

    // Box [0,1]^3. Front face (z=1) is the sensor.
    addQuad(s, {0,0,0},{1,0,0},{1,0,1},{0,0,1}, 0);            // floor
    addQuad(s, {0,1,0},{0,1,1},{1,1,1},{1,1,0}, 0);            // ceiling
    addQuad(s, {0,0,0},{0,1,0},{1,1,0},{1,0,0}, 0);            // back
    addQuad(s, {0,0,0},{0,0,1},{0,1,1},{0,1,0}, 1);            // left (red)
    addQuad(s, {1,0,0},{1,1,0},{1,1,1},{1,0,1}, 2);            // right (green)
    addQuad(s, {0,0,1},{1,0,1},{1,1,1},{0,1,1}, 0, /*sensor*/0); // front = sensor

    // Ceiling area light.
    const double lx0 = 0.35, lx1 = 0.65, lz0 = 0.35, lz1 = 0.65, ly = 0.999;
    addQuad(s, {lx0,ly,lz0},{lx1,ly,lz0},{lx1,ly,lz1},{lx0,ly,lz1}, 3);

    s.finalizeTris();

    // Sensor: front plane, u=+x, v=+y.
    s.sensor.origin = {0,0,1}; s.sensor.uAxis = {1,0,0}; s.sensor.vAxis = {0,1,0};
    s.sensor.resX = res; s.sensor.resY = res; s.sensor.alloc();

    // Light sampling data.
    s.lightOrigin = {lx0, ly, lz0};
    s.lightU = {lx1 - lx0, 0, 0};
    s.lightV = {0, 0, lz1 - lz0};
    s.lightNormal = {0, -1, 0};
    s.lightArea = (lx1 - lx0) * (lz1 - lz0);
    s.lightSpd.build(s.mats[3].emit, 1.0);
    s.lightEmitIntegral = s.lightSpd.integral;
    return s;
}

// --- Colour-pipeline self test ---------------------------------------------
static void selfTestColor() {
    // Equal-energy spectrum -> XYZ should have Y ~ 1 and be near-neutral.
    Vec3 xyz{};
    for (double w = LAMBDA_MIN; w <= LAMBDA_MAX; w += 1.0)
        xyz += Vec3(cieX(w), cieY(w), cieZ(w));
    xyz = xyz / cieYIntegral();
    Vec3 lin = xyzToLinearSrgb(xyz);
    std::printf("[selftest] equal-energy XYZ = (%.3f, %.3f, %.3f)  Y=%.3f\n", xyz.x, xyz.y, xyz.z, xyz.y);
    std::printf("[selftest] equal-energy linear sRGB = (%.3f, %.3f, %.3f)\n", lin.x, lin.y, lin.z);
}

static void writePPM(const char* path, const Sensor& s, double N) {
    const int W = s.resX, H = s.resY;
    std::vector<Vec3> lin((size_t)W * H);
    double norm = 1.0 / (N * cieYIntegral());
    // Auto-exposure from the 99th-percentile luminance.
    std::vector<double> lum; lum.reserve((size_t)W * H);
    for (size_t i = 0; i < lin.size(); ++i) {
        Vec3 rgb = xyzToLinearSrgb(s.xyz[i] * norm);
        lin[i] = rgb;
        lum.push_back(std::max({rgb.x, rgb.y, rgb.z, 0.0}));
    }
    std::vector<double> sorted = lum; std::sort(sorted.begin(), sorted.end());
    double p99 = sorted[(size_t)(0.99 * (sorted.size() - 1))];
    double exposure = (p99 > 0) ? 0.9 / p99 : 1.0;

    std::vector<uint8_t> img((size_t)W * H * 3);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        // Flip vertically so +y (world up) is image-top.
        size_t src = (size_t)(H - 1 - y) * W + x;
        size_t dst = ((size_t)y * W + x) * 3;
        for (int c = 0; c < 3; ++c) {
            double v = (&lin[src].x)[c] * exposure;
            img[dst + c] = (uint8_t)std::clamp(srgbGamma(v) * 255.0 + 0.5, 0.0, 255.0);
        }
    }
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << ' ' << H << "\n255\n";
    f.write((const char*)img.data(), (std::streamsize)img.size());
    std::printf("wrote %s (%dx%d), auto-exposure=%.3g\n", path, W, H, exposure);
}

int main(int argc, char** argv) {
    long long N = 2'000'000;
    int res = 256;
    const char* out = "cornell.ppm";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) N = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "-r") && i + 1 < argc) res = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
    }

    selfTestColor();

    Scene scene = buildCornell(res);
    Renderer r;
    Pcg32 rng; rng.seed(1u, 0x1234u);
    EnergyReport e;

    std::printf("tracing %lld photons at %dx%d ...\n", N, res, res);
    long long tick = std::max<long long>(1, N / 10);
    for (long long i = 0; i < N; ++i) {
        r.tracePhoton(scene, rng, e);
        if ((i + 1) % tick == 0) std::printf("  %lld%%\n", (i + 1) * 100 / N);
    }

    double tot = e.absorbed + e.sensor + e.escaped + e.residual;
    std::printf("\n[energy] emitted=%.6g\n", e.emitted);
    std::printf("[energy] absorbed=%.4f sensor=%.4f escaped=%.4f residual=%.4f (sum/emitted=%.6f)\n",
                e.absorbed / e.emitted, e.sensor / e.emitted, e.escaped / e.emitted,
                e.residual / e.emitted, tot / e.emitted);

    writePPM(out, scene.sensor, (double)N);
    return 0;
}
