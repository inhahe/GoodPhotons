// meshbench — split loadObj into its two halves for the live-viewer mesh path.
//
// The viewer reloads the same swept mesh every frame, and `[play] ftsl … assets N`
// is the cost.  Two very different things live in there:
//
//   * TEXT PARSING     — which a binary mesh handoff deletes outright.
//   * CREASE SMOOTHING — which it does NOT: `smooth 1` on a mesh with no `vn` welds
//                        vertices and synthesizes angle-weighted normals, and that
//                        has to happen for any new geometry however it arrived.
//
// Knowing the split decides whether a binary format is worth it, so measure before
// designing.  Reports min per rep, not mean (background load can only make a sample
// slower; see tools/ftslbench.cpp).  Build with tools/meshbench.bat.
//
// It also takes a `.ftmesh` (the binary format that came out of this measurement),
// dispatching on the extension, so the two paths can be compared directly and their
// FINGERPRINTS diffed -- both loaders share `meshFinishTris`, so the smoothed
// normals must agree bit-for-bit wherever f32 storage happens to round to the same
// value as the OBJ text did.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "mesh.h"

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// Bit-exact fingerprint of everything loadObj produced, so an optimization of the
// smoothing loop can be PROVEN identical to the baseline rather than eyeballed.
// FNV-1a over the raw bytes of each triangle's positions, normals and UVs.
static unsigned long long fingerprint(const Scene& s, int triStart, int nTris) {
    unsigned long long h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    for (int i = 0; i < nTris; ++i) {
        const Tri& t = s.tris[triStart + i];
        mix(&t.v0, sizeof(t.v0)); mix(&t.v1, sizeof(t.v1)); mix(&t.v2, sizeof(t.v2));
        mix(&t.n0, sizeof(t.n0)); mix(&t.n1, sizeof(t.n1)); mix(&t.n2, sizeof(t.n2));
        mix(&t.uv0, sizeof(t.uv0));
        mix(&t.uv1, sizeof(t.uv1));
        mix(&t.uv2, sizeof(t.uv2));
    }
    return h;
}

// Load one mesh with crease smoothing, whatever its format.
static int loadAny(Scene& s, const std::string& path, const Affine& xf, double crease) {
    bool binary = path.size() > 7 && path.compare(path.size() - 7, 7, ".ftmesh") == 0;
    if (!binary)
        return loadObj(s, path.c_str(), 0, xf, false, nullptr,
                       UvProjection::None, 1, crease);
    std::string err;
    int n = loadFtmesh(s, path.c_str(), 0, xf, false, err,
                       UvProjection::None, 1, crease);
    if (n == 0 && !err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(1); }
    return n;
}

// `--compare A B`: load the same mesh through two formats and report how far apart
// the RESULTS are. A fingerprint diff only says "not identical", which is the
// expected answer when one side stores f32 and the other stores `%.6g` decimal —
// the question that actually matters is whether the difference is at the storage
// epsilon (fine) or somewhere structural (a bug). Reports worst-case position
// error against the mesh size, and worst-case shading-normal deviation in degrees.
static int compareMain(const std::string& pa, const std::string& pb, double crease) {
    Affine xf = MeshXform{{0, 0, 0}, {1, 1, 1}, {0, 0, 0}}.toAffine();
    Scene sa, sb;
    int na = loadAny(sa, pa, xf, crease);
    int nb = loadAny(sb, pb, xf, crease);
    std::fflush(stdout);
    if (na != nb) {
        std::fprintf(stderr, "\nMISMATCH: %s has %d tris, %s has %d\n",
                     pa.c_str(), na, pb.c_str(), nb);
        return 1;
    }
    Vec3 lo = sa.tris[0].v0, hi = sa.tris[0].v0;
    for (int i = 0; i < na; ++i)
        for (const Vec3& v : {sa.tris[i].v0, sa.tris[i].v1, sa.tris[i].v2}) {
            lo = Vec3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
            hi = Vec3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
        }
    Vec3 ext = hi - lo;
    double diag = std::sqrt(dot(ext, ext));

    double maxPos = 0.0, maxAng = 0.0;
    for (int i = 0; i < na; ++i) {
        const Tri& A = sa.tris[i];
        const Tri& B = sb.tris[i];
        const Vec3 pv[6] = {A.v0, A.v1, A.v2, B.v0, B.v1, B.v2};
        for (int k = 0; k < 3; ++k) {
            Vec3 d = pv[k] - pv[k + 3];
            maxPos = std::max(maxPos, std::sqrt(dot(d, d)));
        }
        const Vec3 nv[6] = {A.n0, A.n1, A.n2, B.n0, B.n1, B.n2};
        for (int k = 0; k < 3; ++k) {
            const Vec3& x = nv[k];
            const Vec3& y = nv[k + 3];
            double lx = std::sqrt(dot(x, x)), ly = std::sqrt(dot(y, y));
            if (lx < 1e-12 || ly < 1e-12) continue;   // "no normal" on both sides
            double c = dot(x, y) / (lx * ly);
            c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
            maxAng = std::max(maxAng, std::acos(c) * 180.0 / 3.14159265358979323846);
        }
    }
    std::fprintf(stderr, "\ncompare (crease %.3g deg): %d tris\n", crease, na);
    std::fprintf(stderr, "  A %s\n  B %s\n", pa.c_str(), pb.c_str());
    std::fprintf(stderr, "  mesh bbox diagonal        %.6g\n", diag);
    std::fprintf(stderr, "  worst vertex displacement %.3g  (%.3g x the diagonal)\n",
                 maxPos, diag > 0 ? maxPos / diag : 0.0);
    std::fprintf(stderr, "  worst shading-normal tilt %.4g degrees\n", maxAng);
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--compare")
        return compareMain(argv[2], argv[3], (argc > 4) ? std::atof(argv[4]) : 40.0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: meshbench <mesh.obj|mesh.ftmesh> [reps] [creaseDeg]\n"
                     "       meshbench --compare <a> <b> [creaseDeg]\n");
        return 2;
    }
    const int    reps   = (argc > 2) ? std::atoi(argv[2]) : 40;
    const double crease = (argc > 3) ? std::atof(argv[3]) : 40.0;

    Affine xf = MeshXform{{0, 0, 0}, {1, 1, 1}, {0, 0, 0}}.toAffine();

    std::string path = argv[1];
    bool binary = path.size() > 7 && path.compare(path.size() - 7, 7, ".ftmesh") == 0;

    // One call, either loader, so the two are timed through identical scaffolding.
    auto load = [&](Scene& s, double creaseDeg) -> int {
        if (!binary)
            return loadObj(s, path.c_str(), 0, xf, false, nullptr,
                           UvProjection::None, 1, creaseDeg);
        std::string err;
        int n = loadFtmesh(s, path.c_str(), 0, xf, false, err,
                           UvProjection::None, 1, creaseDeg);
        if (n == 0 && !err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(1); }
        return n;
    };

    double flatMin = 1e300, smoothMin = 1e300;
    double flatSum = 0.0,   smoothSum = 0.0;
    int    tris = 0;
    unsigned long long fp = 0;

    // Silence the loaders' per-call summary lines so they don't dominate the timing.
    std::fflush(stdout);

    for (int i = 0; i < reps; ++i) {
        {   // no crease smoothing: pure read + decode
            Scene s;
            auto t0 = std::chrono::steady_clock::now();
            tris = load(s, /*creaseAngleDeg*/ -1.0);
            double d = ms_since(t0);
            flatSum += d; if (d < flatMin) flatMin = d;
        }
        {   // same, plus crease smoothing
            Scene s;
            auto t0 = std::chrono::steady_clock::now();
            int n = load(s, crease);
            double d = ms_since(t0);
            smoothSum += d; if (d < smoothMin) smoothMin = d;
            fp = fingerprint(s, 0, n);
        }
    }

    std::fprintf(stderr, "\n%s: %d tris, %d reps\n", argv[1], tris, reps);
    std::fprintf(stderr, "  read+%-6s only   min %6.2f ms  mean %6.2f ms\n",
                 binary ? "decode" : "parse", flatMin, flatSum / reps);
    std::fprintf(stderr, "  + crease smooth   min %6.2f ms  mean %6.2f ms\n",
                 smoothMin, smoothSum / reps);
    std::fprintf(stderr, "  => smoothing costs      %6.2f ms  (%.0f%% of the loaded cost)\n",
                 smoothMin - flatMin, 100.0 * (smoothMin - flatMin) / smoothMin);
    std::fprintf(stderr, "  smoothed fingerprint  %016llx\n", fp);
    return 0;
}
