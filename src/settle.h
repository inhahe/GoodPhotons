#pragma once
// ============================================================================================
// HAIR SETTLING — a relaxation over the flattened strands, run once per load (v0.346.0).
//
// WHY IT LIVES HERE AND NOT IN THE GROOM TOOL. The `.ftsl` file carries curve *definitions*,
// and a settled groom cannot be written as one: `count N` spaces its instances by ARC LENGTH
// along the path through its children's roots, and no arc-length rule can express "except
// where another strand already is". The settled positions only exist once the definitions have
// been flattened, so the settle has to be a stage of the LOAD. (Baking the result back as
// explicit `point`s would work and would also destroy the procedural groom and grow the file by
// 1.2 M points, which is not a trade worth making.)
//
// It runs on `Scene::curveSegs` — after every curve, curve-of-curves and deferred `fur` block
// has been flattened and tessellated, and before the BVH is built over the result. That single
// placement is what makes the request "no matter how we define our curves" achievable rather
// than a per-feature fix: every authoring route in the language funnels through this one pool.
//
// WHAT THE MEASUREMENT SAID, and how it shaped the solver (known-issues HAIR-PENETRATION):
// 72.67 % of Alice's 1.2 M segments lie inside another strand, against 0.97 % for `fur`
// scattered at the same 20 000 strands — so it is the blend, not density. But the median
// penetration is only **66 um**, i.e. 4.7 % of one segment length. Removing it is a LOCAL
// de-overlap of tens of microns, not a dynamics problem, which is why a quasi-static projection
// scheme converges here in tens of iterations rather than needing a stable time integrator.
//
// THE SCHEME is position-based and velocity-free: each iteration injects a small gravity
// displacement and then projects four constraint sets, in the order they must dominate —
//   1. inextensibility  (a hair does not stretch; Gauss-Seidel along each strand)
//   2. shape            (stiffness: hold the authored form, stiffest at the
//                        root, slackest at the tip — this is what stops a settle turning a
//                        sculpted groom into a wet mop, and it is the "stiffness" in the ask)
//   3. separation       (strand-strand, exact segment-segment distance against r_i + r_j)
//   4. volume           (the COLLECTIVE term -- see below)
//   5. collision        (stay outside named mesh groups)
//
// SHAPE: `global` (default) pulls each particle toward its AUTHORED POSITION. That position is
// by definition the configuration that overlaps, so the spring opposes precisely the motion
// that would fix the overlap -- stiffness 0.40 leaves 63.12 % of segments overlapping, 0.00
// leaves 52.24 %, and the only way to weaken it is to let the whole groom sag (3.46 mm mean).
//
// `shape local` was built to escape that -- preserve the strand's SHAPE rather than its
// POSITION, so it can translate and swing to clear a neighbour while keeping its curl -- and
// IT DRIFTS. Measured: local at stiffness 0.05 gives 52.68 % at 3.34 mm, which is what NO
// shape constraint does (global at 0.00 gives 52.24 % at 3.46 mm), and local at 0.20 is WORSE
// at 70.41 % while moving the groom FURTHER (4.75 mm mean, 19 mm max) -- which no restoring
// force can do. The defect is that the frame R below is re-derived from the BLENDED result
// each sweep, so the rest shape follows the current shape: the constraint targets wherever the
// strand already drifted to, and at high stiffness tracks that tightly enough to pump motion
// in rather than damp it out. A correct version would either carry a per-particle frame as
// simulation STATE (a quaternion advanced by the rest transform, TressFX-style) or use a
// per-strand RIGID fit against the REST shape, which cannot drift by construction. Kept
// because it is the A/B control that produced this measurement.
//
// WHY 4 EXISTS, measured rather than assumed. Separation alone applies the MEAN of a segment's
// contact directions, and for a fiber overlapped on all sides those vectors cancel to nearly
// zero however deep the overlaps are. Measured on Alice at 0.346.0: with stiffness AND gravity
// both at zero -- nothing opposing separation at all -- the share of overlapping segments moved
// only 72.67 % -> 70.09 %. Local pairwise pushing cannot expand a bundle; only a collective
// term can. So the fibers are also splatted into a density grid and pushed DOWN its gradient
// wherever the local volume fraction exceeds `packing`, inflating an over-dense neighbourhood
// as a whole.
//
// IT IS OFF BY DEFAULT, because on Alice it measurably made things WORSE: the binary share of
// overlapping segments went to 73.97 %, against 70.09 % with separation alone. The reason is
// scale. A field can only separate structures it RESOLVES, and the overlapping groups here are
// finer than one 0.35 mm cell, so the gradient translates a whole cluster rather than expanding
// it. Resolving them would need ~0.1 mm cells, which is 830 M cells on this groom -- a sparse
// structure, not a dense grid. Kept because it is correct and cheap when a groom genuinely is
// jammed at a coarser scale; enable with `volume`.
//
// What actually limits the de-overlap is the UPDATE, measured after the fact: a penetrating
// segment has a median of 4 overlapping partners (mean 7.7) and local packing is 0.21 against
// 0.82 for random close packing, so there is room and four partners is locally resolvable. The
// old separation divided the summed violation by `hits` and then again by `cw`, i.e. applied
// ~6 % of it per sweep. That is Jacobi averaging, which crawls on contact problems.
//
// The grid cell is ~8 fiber radii (0.35 mm on Alice), NOT the 1-2 mm 'lock scale' the original
// design named -- because the packing measurement said the clustering is finer than a lock:
// 7 distinct strands within 0.25 mm at a local volume fraction of only 0.21. A 1.5 mm cell
// holds barely one segment and would move whole locks apart while leaving the fibers inside
// them merged. Segments are splatted ALONG their length (a segment is 1.47 mm, four cells),
// not at their endpoints, or the field aliases into stripes.
//
// The accumulation is FIXED-POINT in uint64, not float: integer addition is associative, so the
// grid is identical however the threads interleave. Float atomics would reintroduce exactly the
// thread-count dependence the determinism control exists to catch.
// with roots pinned throughout. The fixed point is where gravity balances stiffness and
// contact, which is the "equilibrium" asked for, reached without a dt to tune.
//
// THE DEFAULTS ARE MEASURED, not guessed (tools/settle_rig.py + scraps/sweep_settle.py on
// Alice; baseline 72.67 % of segments overlapping, median depth 66.4 um):
//
//   stiffness   overlapping   depth     groom moved (mean/max), segment is 1.47 mm
//      0.40       63.12 %    57.1 um      0.250 / 1.361 mm
//      0.20       60.04 %    55.4 um      0.518 / 2.774 mm
//      0.05       55.11 %    51.6 um      1.629 / 6.316 mm   <- default
//      0.00       52.24 %    50.9 um      3.463 / 9.048 mm   (fails the groom-survives guard)
//
// `stiffness` is the whole trade: it buys the authored silhouette back at the cost of leaving
// overlaps in. 0.05 is the lowest setting whose displacement still passes the rig's
// groom-survives guard (mean under two segment lengths). Separation slack 1.8 likewise:
// 1.25 -> 61.99 %, 1.8 -> 60.04 %, 2.5 -> 58.62 %, but the depth statistic gets steadily
// WORSE as the slack grows (51.5 -> 55.4 -> 63.5 um) because a larger target puts more pairs
// in violation and spreads the effort, so 2.5 is buying the headline with the other metric.
//
// Iterations are NOT the limiter and raising them is wasted: 150 sweeps gains 0.2 points over
// 60 (59.82 vs 60.04 %). The solver converges; what is left is a genuine constraint conflict.
//
// DETERMINISM IS A HARD REQUIREMENT, not a nicety: the CPU and CUDA backends must trace
// byte-identical geometry, and a flyby re-loads the scene per frame. So the separation pass is
// ONE-SIDED — segment i accumulates only into itself, scanning every neighbour j — which costs
// each pair twice and buys a result with no atomics, no race and no thread-count dependence.
// ============================================================================================
#include "scene.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace settle {

struct Params {
    bool   on        = false;
    int    iters     = 60;      // projection sweeps
    double droop     = 0.5;     // gravity injected per sweep, as a fraction of 0.1 % of the groom's diagonal
    double stiffRoot = 0.05;    // hold to the authored shape at the root ...
    double stiffTip  = 0.01;    // ... and at the tip  (both MEASURED, see the table below)
    double sepScale  = 1.8;     // separation target as a multiple of (r_i + r_j); 0 disables
    double margin    = 0.0;     // extra clearance held against colliders
    int    maxNbr    = 32;      // neighbour-list cap per segment (memory bound)
    int    refresh   = 10;      // rebuild the neighbour list every N sweeps (0 = once only)
    int    shapeMode = 0;       // 0 = global position spring, 1 = local (DRIFTS), 2 = rigid fit
    double volume    = 0.0;     // COLLECTIVE density-gradient push; OFF by default, see below
    double packing   = 0.30;    // volume fraction above which a neighbourhood expands
    double cellSize  = 0.0;     // density-grid cell in metres; 0 = auto (8 x mean fiber radius)
    double maxStep   = 1.0;     // per-sweep separation displacement cap, in fiber radii
    std::vector<std::string> collide;   // mesh-group names to stay outside of
    std::string cachePath;              // sidecar for the settled positions; empty = no cache
};

namespace detail {

struct Grid {
    double cell = 1.0;
    std::vector<uint64_t>  key;     // sorted cell key per entry
    std::vector<uint32_t>  idx;     // the entry's item index, in key order
    static uint64_t code(int x, int y, int z) {
        const uint64_t ux = (uint64_t)(x + 1048576) & 0x1FFFFF;
        const uint64_t uy = (uint64_t)(y + 1048576) & 0x1FFFFF;
        const uint64_t uz = (uint64_t)(z + 1048576) & 0x1FFFFF;
        return (ux << 42) | (uy << 21) | uz;
    }
    uint64_t keyOf(const Vec3& p) const {
        return code((int)std::floor(p.x / cell), (int)std::floor(p.y / cell), (int)std::floor(p.z / cell));
    }
    // [lo, hi) of the run with this key
    void range(uint64_t k, size_t& lo, size_t& hi) const {
        lo = (size_t)(std::lower_bound(key.begin(), key.end(), k) - key.begin());
        hi = lo;
        while (hi < key.size() && key[hi] == k) ++hi;
    }
};

inline void buildGrid(Grid& g, const std::vector<Vec3>& mid, double cell) {
    g.cell = cell;
    const size_t n = mid.size();
    std::vector<std::pair<uint64_t, uint32_t>> e(n);
    for (size_t i = 0; i < n; ++i) e[i] = { g.keyOf(mid[i]), (uint32_t)i };
    std::sort(e.begin(), e.end());
    g.key.resize(n); g.idx.resize(n);
    for (size_t i = 0; i < n; ++i) { g.key[i] = e[i].first; g.idx[i] = e[i].second; }
}

// Closest points on two segments (Ericson). Returns the squared distance and the two points.
inline double segSeg(const Vec3& p1, const Vec3& q1, const Vec3& p2, const Vec3& q2,
                     Vec3& c1, Vec3& c2) {
    const Vec3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    const double a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    double s = 0.0, t = 0.0;
    if (a <= 1e-30 && e <= 1e-30) { c1 = p1; c2 = p2; const Vec3 d = c1 - c2; return dot(d, d); }
    if (a <= 1e-30) { s = 0.0; t = std::min(std::max(f / e, 0.0), 1.0); }
    else {
        const double c = dot(d1, r);
        if (e <= 1e-30) { t = 0.0; s = std::min(std::max(-c / a, 0.0), 1.0); }
        else {
            const double b = dot(d1, d2), den = a * e - b * b;
            s = (den > 1e-30) ? std::min(std::max((b * f - c * e) / den, 0.0), 1.0) : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0)      { t = 0.0; s = std::min(std::max(-c / a, 0.0), 1.0); }
            else if (t > 1.0) { t = 1.0; s = std::min(std::max((b - c) / a, 0.0), 1.0); }
        }
    }
    c1 = p1 + d1 * s; c2 = p2 + d2 * t;
    const Vec3 d = c1 - c2;
    return dot(d, d);
}

inline Vec3 closestOnTri(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 ab = b - a, ac = c - a, ap = p - a;
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    const Vec3 bp = p - b;
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + ab * (d1 / (d1 - d3));
    const Vec3 cp = p - c;
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + ac * (d2 / (d2 - d6));
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    const double den = 1.0 / (va + vb + vc);
    return a + ab * (vb * den) + ac * (vc * den);
}

// ---- the sidecar cache -------------------------------------------------------------------
// FNV-1a over every input that can change the answer. Not a cryptographic hash and does not need
// to be: the failure mode is a stale groom, and the payload carries its own particle count and a
// magic as a second gate.
inline uint64_t fnv(uint64_t h, const void* data, size_t n) {
    const unsigned char* b = (const unsigned char*)data;
    for (size_t i = 0; i < n; ++i) { h ^= (uint64_t)b[i]; h *= 1099511628211ull; }
    return h;
}

static const uint64_t kCacheMagic = 0x46545253544c4531ull;   // "FTRSTLE1"

// Minimal rotation taking unit `a` onto unit `b`, composed onto the 3x3 `R` from the left
// (Rodrigues). The antiparallel case picks an arbitrary perpendicular axis; a single sweep's
// motion cannot produce it, but a doubled-back authored strand can.
inline void rotateOnto(const Vec3& a, const Vec3& b, double R[9]) {
    const Vec3 v = cross(a, b);
    const double c = dot(a, b);
    const double s2 = dot(v, v);
    double D[9];
    if (s2 < 1e-24) {
        if (c > 0.0) return;                       // already aligned
        Vec3 ax = (std::fabs(a.x) < 0.9) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
        ax = cross(a, ax);
        const double l = length(ax);
        if (l < 1e-18) return;
        ax = ax * (1.0 / l);
        D[0] = 2*ax.x*ax.x - 1; D[1] = 2*ax.x*ax.y;     D[2] = 2*ax.x*ax.z;
        D[3] = 2*ax.y*ax.x;     D[4] = 2*ax.y*ax.y - 1; D[5] = 2*ax.y*ax.z;
        D[6] = 2*ax.z*ax.x;     D[7] = 2*ax.z*ax.y;     D[8] = 2*ax.z*ax.z - 1;
    } else {
        const double k = 1.0 / (1.0 + c);
        D[0] = 1 - (v.y*v.y + v.z*v.z)*k; D[1] = -v.z + v.x*v.y*k;          D[2] =  v.y + v.x*v.z*k;
        D[3] =  v.z + v.x*v.y*k;          D[4] = 1 - (v.x*v.x + v.z*v.z)*k; D[5] = -v.x + v.y*v.z*k;
        D[6] = -v.y + v.x*v.z*k;          D[7] =  v.x + v.y*v.z*k;          D[8] = 1 - (v.x*v.x + v.y*v.y)*k;
    }
    double O[9];
    for (int r = 0; r < 3; ++r)
        for (int cc = 0; cc < 3; ++cc)
            O[r*3 + cc] = D[r*3 + 0]*R[0*3 + cc] + D[r*3 + 1]*R[1*3 + cc] + D[r*3 + 2]*R[2*3 + cc];
    for (int i = 0; i < 9; ++i) R[i] = O[i];
}
inline Vec3 applyR(const double R[9], const Vec3& v) {
    return Vec3{ R[0]*v.x + R[1]*v.y + R[2]*v.z,
                 R[3]*v.x + R[4]*v.y + R[5]*v.z,
                 R[6]*v.x + R[7]*v.y + R[8]*v.z };
}

// 3x3 helpers for the rigid shape fit. Row-major, R[r*3+c].
inline void mat3mul(const double* A, const double* B, double* O) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            O[r*3+c] = A[r*3+0]*B[0*3+c] + A[r*3+1]*B[1*3+c] + A[r*3+2]*B[2*3+c];
}
inline double mat3det(const double* A) {
    return A[0]*(A[4]*A[8] - A[5]*A[7]) - A[1]*(A[3]*A[8] - A[5]*A[6]) + A[2]*(A[3]*A[7] - A[4]*A[6]);
}
// Transpose of the inverse, which is what the polar iteration actually wants.
inline bool mat3invT(const double* A, double* O) {
    const double d = mat3det(A);
    if (std::fabs(d) < 1e-300) return false;
    const double s = 1.0 / d;
    O[0] = (A[4]*A[8] - A[5]*A[7]) * s;  O[3] = -(A[1]*A[8] - A[2]*A[7]) * s;  O[6] = (A[1]*A[5] - A[2]*A[4]) * s;
    O[1] = -(A[3]*A[8] - A[5]*A[6]) * s; O[4] = (A[0]*A[8] - A[2]*A[6]) * s;   O[7] = -(A[0]*A[5] - A[2]*A[3]) * s;
    O[2] = (A[3]*A[7] - A[4]*A[6]) * s;  O[5] = -(A[0]*A[7] - A[1]*A[6]) * s;  O[8] = (A[0]*A[4] - A[1]*A[3]) * s;
    return true;
}
// The rotation factor of A, by Higham's iteration R <- (R + R^-T)/2. Quadratic, and eight
// steps is far past convergence for the near-rotations a settle produces. Returns false when
// A is rank-deficient (a straight strand), which the caller handles rather than papers over.
inline bool polarRotation(const double* A, double* R) {
    for (int i = 0; i < 9; ++i) R[i] = A[i];
    double invT[9], nxt[9];
    for (int it = 0; it < 8; ++it) {
        if (!mat3invT(R, invT)) return false;
        for (int i = 0; i < 9; ++i) nxt[i] = 0.5 * (R[i] + invT[i]);
        double d = 0.0;
        for (int i = 0; i < 9; ++i) d += std::fabs(nxt[i] - R[i]);
        for (int i = 0; i < 9; ++i) R[i] = nxt[i];
        if (d < 1e-14) break;
    }
    return mat3det(R) > 0.0;   // a reflection is not a pose
}

} // namespace detail

// Runs the settle in place. Returns false only if `ftrace -stop` cancelled it mid-way (the
// caller must then abandon the load: a half-relaxed groom is not a scene anyone asked for).
inline bool run(Scene& sc, const Params& p, std::string& report) {
    report.clear();
    if (!p.on || sc.curves.empty() || sc.curveSegs.empty()) return true;

    // ---- 1. strands -> particles ----------------------------------------------------------
    // Consecutive CurveSegs share an endpoint by construction (tessellateCurve carries `prevP`),
    // so a strand of N segs is a polyline of N+1 particles.
    const size_t nC = sc.curves.size();
    std::vector<int> first((size_t)nC), cnt((size_t)nC);
    size_t nP = 0;
    for (size_t c = 0; c < nC; ++c) {
        const int n = sc.curves[c].segCount;
        first[c] = (int)nP; cnt[c] = (n > 0) ? n + 1 : 0;
        nP += (size_t)cnt[c];
    }
    if (nP == 0) return true;

    std::vector<Vec3>   pos(nP), rest(nP);
    std::vector<double> rad(nP), rlen(nP, 0.0);
    std::vector<float>  stiff(nP, 0.0f);
    std::vector<int>    owner(nP, -1);
    for (size_t c = 0; c < nC; ++c) {
        const Curve& cu = sc.curves[c];
        if (cu.segCount <= 0) continue;
        const int b = first[c], n = cu.segCount;
        for (int k = 0; k < n; ++k) {
            const CurveSeg& s = sc.curveSegs[(size_t)cu.firstSeg + (size_t)k];
            if (k == 0) { pos[(size_t)b] = s.p0; rad[(size_t)b] = s.r0; }
            pos[(size_t)b + (size_t)k + 1] = s.p1;
            rad[(size_t)b + (size_t)k + 1] = s.r1;
        }
        for (int k = 0; k <= n; ++k) {
            const size_t i = (size_t)b + (size_t)k;
            owner[i] = (int)c;
            rest[i]  = pos[i];
            const double t = (n > 0) ? (double)k / (double)n : 0.0;
            stiff[i] = (float)(p.stiffRoot + (p.stiffTip - p.stiffRoot) * t);
            if (k < n) rlen[i] = length(pos[i + 1] - pos[i]);
        }
    }

    // ---- 1b. the cache key, and a hit short-circuits the whole solve ----------------------
    uint64_t key = 1469598103934665603ull;
    {
        const double pv[] = { (double)p.iters, p.droop, p.stiffRoot, p.stiffTip, p.sepScale,
                              p.margin, (double)p.maxNbr, p.volume, p.packing, p.cellSize,
                              p.maxStep, (double)p.refresh, (double)p.shapeMode };
        key = detail::fnv(key, pv, sizeof pv);
        for (const std::string& c : p.collide) key = detail::fnv(key, c.data(), c.size());
        key = detail::fnv(key, pos.data(), pos.size() * sizeof(Vec3));   // the AUTHORED geometry
        key = detail::fnv(key, rad.data(), rad.size() * sizeof(double));
    }
    if (!p.cachePath.empty()) {
        std::FILE* f = std::fopen(p.cachePath.c_str(), "rb");
        if (f) {
            uint64_t magic = 0, k = 0, n = 0;
            const bool hdr = std::fread(&magic, sizeof magic, 1, f) == 1 &&
                             std::fread(&k, sizeof k, 1, f) == 1 &&
                             std::fread(&n, sizeof n, 1, f) == 1;
            if (hdr && magic == detail::kCacheMagic && k == key && n == (uint64_t)nP &&
                std::fread(pos.data(), sizeof(Vec3), nP, f) == nP) {
                std::fclose(f);
                for (size_t c = 0; c < nC; ++c) {
                    const Curve& cu = sc.curves[c];
                    for (int kk = 0; kk < cu.segCount; ++kk) {
                        const size_t i = (size_t)first[c] + (size_t)kk;
                        CurveSeg& sg = sc.curveSegs[(size_t)cu.firstSeg + (size_t)kk];
                        sg.p0 = pos[i]; sg.p1 = pos[i + 1];
                    }
                }
                char cb[200];
                std::snprintf(cb, sizeof cb, "%zu strand(s), %zu particle(s) restored from %s",
                              nC, nP, p.cachePath.c_str());
                report += cb;
                return true;
            }
            std::fclose(f);
        }
    }

    // groom extent, for the gravity step and the grid cell
    Vec3 lo = pos[0], hi = pos[0];
    double lenSum = 0.0;
    for (size_t i = 0; i < nP; ++i) {
        lo.x = std::min(lo.x, pos[i].x); lo.y = std::min(lo.y, pos[i].y); lo.z = std::min(lo.z, pos[i].z);
        hi.x = std::max(hi.x, pos[i].x); hi.y = std::max(hi.y, pos[i].y); hi.z = std::max(hi.z, pos[i].z);
        lenSum += rlen[i];
    }
    const size_t nSeg = sc.curveSegs.size();
    const double meanSeg = (nSeg > 0) ? lenSum / (double)nSeg : 1e-3;
    const double diag = length(hi - lo);
    const double gstep = p.droop * 0.001 * diag;     // injected downward displacement per sweep

    // ---- 2. the neighbour list ------------------------------------------------------------
    // Built ONCE. The displacements this solver applies are ~66 um against a cell of ~1.4 mm,
    // so a rebuild per sweep would buy nothing and cost the sort each time.
    std::vector<uint32_t> nbrOff, nbrIdx;
    size_t capped = 0;
    const bool doSep = (p.sepScale > 0.0);
    std::vector<Vec3>     mid;
    std::vector<uint32_t> flat, nn;
    std::vector<uint8_t>  over;
    detail::Grid g;
    if (doSep) {
        mid.resize(nSeg);
        flat.assign((size_t)nSeg * (size_t)p.maxNbr, 0u);
        nn.assign(nSeg, 0u);
        over.assign(nSeg, 0u);
        nbrOff.assign(nSeg + 1, 0u);
    }
    // Rebuilt DURING the settle, not only before it. Measured: of the contacts surviving a
    // 60-sweep settle, 51.8 % were never in the shortlist -- as the solver separates its 32
    // nearest, pairs that ranked 33+ move up and become real contacts it never sees. The cell
    // stays valid as positions move (~66 um against 1.47 mm); the RANKING does not, which is the
    // distinction the original "build it once" justification missed.
    auto buildNbrs = [&]() -> bool {
        for (size_t c = 0; c < nC; ++c) {
            const Curve& cu = sc.curves[c];
            for (int k = 0; k < cu.segCount; ++k) {
                const size_t i = (size_t)first[c] + (size_t)k;
                mid[(size_t)cu.firstSeg + (size_t)k] = (pos[i] + pos[i + 1]) * 0.5;
            }
        }
        detail::buildGrid(g, mid, std::max(meanSeg, 8.0 * (nSeg ? rad[0] : 1e-4)));
        const double reach = g.cell;
        // RANK BY THE REAL SEGMENT-SEGMENT DISTANCE, not by midpoint distance.
        //
        // Midpoint distance is the obvious cheap proxy and it is close to USELESS here, measured:
        // of a segment's ~2253 candidates in reach and its median 7 actual contacts, a
        // nearest-32-by-midpoint shortlist contained 3 and missed 4 -- 33.0 % of contacts overall.
        // A lock is full of near-parallel neighbours whose midpoints nearly coincide but which do
        // not touch, while two segments crossing at an angle touch with their midpoints a
        // segment-length apart. The solver was faithfully resolving the wrong pairs, which is why
        // three separate strengthenings of the push moved nothing.
        if (!ft::parallelFor(nC, 16, [&](size_t c) {
                const Curve& cu = sc.curves[c];
                const int K = p.maxNbr;
                std::vector<std::pair<double, uint32_t>> best;
                best.reserve((size_t)K + 1);
                for (int k = 0; k < cu.segCount; ++k) {
                    const size_t si = (size_t)cu.firstSeg + (size_t)k;
                    const size_t ia = (size_t)first[c] + (size_t)k, ib = ia + 1;
                    const Vec3 m = mid[si];
                    const int cx = (int)std::floor(m.x / g.cell), cy = (int)std::floor(m.y / g.cell), cz = (int)std::floor(m.z / g.cell);
                    best.clear();
                    uint32_t seen = 0;
                    for (int dx = -1; dx <= 1; ++dx)
                    for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        size_t a, b;
                        g.range(detail::Grid::code(cx + dx, cy + dy, cz + dz), a, b);
                        for (size_t e = a; e < b; ++e) {
                            const uint32_t j = g.idx[e];
                            if ((size_t)j == si) continue;
                            const int c2 = sc.curveSegs[j].curveId;
                            if (c2 == (int)c || c2 < 0 || c2 >= (int)nC) continue;   // same strand
                            const Vec3 dm = mid[j] - m;
                            if (dot(dm, dm) > 4.0 * reach * reach) continue;
                            const int k2 = (int)((size_t)j - (size_t)sc.curves[(size_t)c2].firstSeg);
                            const size_t ja = (size_t)first[c2] + (size_t)k2, jb = ja + 1;
                            Vec3 x1, x2;
                            const double dd = detail::segSeg(pos[ia], pos[ib], pos[ja], pos[jb], x1, x2);
                            ++seen;
                            if ((int)best.size() == K && dd >= best.back().first) continue;
                            auto it = std::lower_bound(best.begin(), best.end(), std::make_pair(dd, j));
                            best.insert(it, std::make_pair(dd, j));
                            if ((int)best.size() > K) best.pop_back();
                        }
                    }
                    for (size_t q = 0; q < best.size(); ++q) flat[si * (size_t)p.maxNbr + q] = best[q].second;
                    nn[si] = (uint32_t)best.size();
                    over[si] = (seen > (uint32_t)K) ? 1 : 0;
                }
            })) return false;
        capped = 0;
        for (size_t i = 0; i < nSeg; ++i) { if (over[i]) ++capped; nbrOff[i + 1] = nbrOff[i] + nn[i]; }
        nbrIdx.resize(nbrOff[nSeg]);
        for (size_t i = 0; i < nSeg; ++i)
            for (uint32_t k = 0; k < nn[i]; ++k) nbrIdx[nbrOff[i] + k] = flat[i * (size_t)p.maxNbr + k];
        return true;
    };
    if (doSep && !buildNbrs()) return false;

    // ---- 3. colliders ----------------------------------------------------------------------
    std::vector<Tri> ctris;
    for (const std::string& nm : p.collide) {
        const MeshGroup* g = nullptr;
        for (const MeshGroup& mg : sc.meshGroups) if (mg.name == nm) { g = &mg; break; }
        if (!g) { report += "collide \"" + nm + "\": no such mesh group; ignored. "; continue; }
        if (g->blasId >= 0) { report += "collide \"" + nm + "\": instanced (BLAS) geometry is not yet a collider; ignored. "; continue; }
        for (size_t t = 0; t < g->triCount; ++t) ctris.push_back(sc.tris[g->triStart + t]);
    }
    detail::Grid cg;
    std::vector<Vec3> ccent;
    if (!ctris.empty()) {
        ccent.resize(ctris.size());
        double ext = 0.0;
        for (size_t t = 0; t < ctris.size(); ++t) {
            ccent[t] = (ctris[t].v0 + ctris[t].v1 + ctris[t].v2) * (1.0 / 3.0);
            ext = std::max(ext, std::max(length(ctris[t].v1 - ctris[t].v0), length(ctris[t].v2 - ctris[t].v0)));
        }
        detail::buildGrid(cg, ccent, std::max(ext, meanSeg));
    }

    // ---- 3b. the density grid ---------------------------------------------------------------
    const bool doVol = (p.volume > 0.0 && p.packing > 0.0);
    double cell = p.cellSize;
    if (doVol && cell <= 0.0) {
        double rs = 0.0;
        for (size_t i = 0; i < nP; ++i) rs += rad[i];
        cell = 8.0 * (rs / (double)nP);
    }
    if (cell <= 0.0) cell = 1.0;
    const double PI_D = 3.14159265358979323846;
    const double VSCALE = 1e21;        // m^3 -> fixed point; one fiber-cell ~2e9, uint64 holds 1.8e19
    Vec3 glo = lo, ghi = hi;
    int gx = 1, gy = 1, gz = 1;
    size_t gN = 1;
    std::unique_ptr<std::atomic<uint64_t>[]> dens;
    if (doVol) {
        const double padv = 4.0 * cell + 0.02 * diag;
        const Vec3 pad{padv, padv, padv};
        glo = lo - pad; ghi = hi + pad;
        gx = (int)std::ceil((ghi.x - glo.x) / cell) + 1;
        gy = (int)std::ceil((ghi.y - glo.y) / cell) + 1;
        gz = (int)std::ceil((ghi.z - glo.z) / cell) + 1;
        gN = (size_t)gx * (size_t)gy * (size_t)gz;
        if (gN > (size_t)400000000) {  // refuse rather than swap the machine
            report += "volume: grid would need too many cells; raise cell. ";
            gN = 1; gx = gy = gz = 1;
        } else {
            dens.reset(new std::atomic<uint64_t>[gN]);
        }
    }
    const bool volOn = doVol && (dens != nullptr);
    const double invCell = 1.0 / cell;
    const double cellVol = cell * cell * cell;
    auto splat = [&](const Vec3& q, uint64_t w) {
        const double fx = (q.x - glo.x) * invCell, fy = (q.y - glo.y) * invCell, fz = (q.z - glo.z) * invCell;
        const int ix = (int)std::floor(fx), iy = (int)std::floor(fy), iz = (int)std::floor(fz);
        if (ix < 0 || iy < 0 || iz < 0 || ix + 1 >= gx || iy + 1 >= gy || iz + 1 >= gz) return;
        const double tx = fx - ix, ty = fy - iy, tz = fz - iz;
        for (int dz = 0; dz < 2; ++dz) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            const double wgt = (dx ? tx : 1.0 - tx) * (dy ? ty : 1.0 - ty) * (dz ? tz : 1.0 - tz);
            const uint64_t add = (uint64_t)(wgt * (double)w);
            if (!add) continue;
            const size_t id = ((size_t)(iz + dz) * (size_t)gy + (size_t)(iy + dy)) * (size_t)gx + (size_t)(ix + dx);
            dens[id].fetch_add(add, std::memory_order_relaxed);
        }
    };
    auto sampleRho = [&](const Vec3& q) -> double {
        const double fx = (q.x - glo.x) * invCell, fy = (q.y - glo.y) * invCell, fz = (q.z - glo.z) * invCell;
        const int ix = (int)std::floor(fx), iy = (int)std::floor(fy), iz = (int)std::floor(fz);
        if (ix < 0 || iy < 0 || iz < 0 || ix + 1 >= gx || iy + 1 >= gy || iz + 1 >= gz) return 0.0;
        const double tx = fx - ix, ty = fy - iy, tz = fz - iz;
        double acc = 0.0;
        for (int dz = 0; dz < 2; ++dz) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            const double wgt = (dx ? tx : 1.0 - tx) * (dy ? ty : 1.0 - ty) * (dz ? tz : 1.0 - tz);
            const size_t id = ((size_t)(iz + dz) * (size_t)gy + (size_t)(iy + dy)) * (size_t)gx + (size_t)(ix + dx);
            acc += wgt * (double)dens[id].load(std::memory_order_relaxed);
        }
        return (acc / VSCALE) / cellVol;
    };

    // ---- 4. the sweeps ---------------------------------------------------------------------
    std::vector<Vec3> corr(nP);
    std::vector<float> cw(nP);
    for (int it = 0; it < p.iters; ++it) {
        if (doSep && p.refresh > 0 && it > 0 && (it % p.refresh) == 0 && !buildNbrs()) return false;

        // (a) gravity, into every unpinned particle (k == 0 is the follicle)
        if (!ft::parallelFor(nC, 64, [&](size_t c) {
                for (int k = 1; k < cnt[c]; ++k) pos[(size_t)first[c] + (size_t)k].y -= gstep;
            })) return false;

        // (b) inextensibility, Gauss-Seidel from the root out (the root is fixed, so each
        //     correction lands entirely on the outboard particle -- which is what makes a
        //     single sweep exact for a strand rather than iterative)
        if (!ft::parallelFor(nC, 64, [&](size_t c) {
                const int b = first[c];
                for (int k = 0; k + 1 < cnt[c]; ++k) {
                    const size_t i = (size_t)b + (size_t)k, j = i + 1;
                    Vec3 d = pos[j] - pos[i];
                    const double L = length(d);
                    if (L < 1e-12) continue;
                    pos[j] = pos[i] + d * (rlen[i] / L);
                }
            })) return false;

        // (c) shape. Sequential along each strand (the frame is carried outward from the root),
        //     parallel ACROSS strands -- which is also what keeps it deterministic, since a
        //     strand owns its particles outright.
        if (p.shapeMode == 1) {
            if (!ft::parallelFor(nC, 32, [&](size_t c) {
                    const int b = first[c], n = cnt[c];
                    if (n < 2) return;
                    double R[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
                    for (int k = 0; k + 1 < n; ++k) {
                        const size_t i = (size_t)b + (size_t)k, j = i + 1;
                        const Vec3 want = detail::applyR(R, rest[j] - rest[i]);
                        // toward where the authored offset says it belongs RELATIVE to its
                        // predecessor, so the strand may translate and swing freely
                        pos[j] = pos[j] + ((pos[i] + want) - pos[j]) * (double)stiff[j];
                        // carry the frame by whatever the segment actually ended up doing
                        const Vec3 got = pos[j] - pos[i];
                        const double lw = length(want), lg = length(got);
                        if (lw > 1e-15 && lg > 1e-15)
                            detail::rotateOnto(want * (1.0 / lw), got * (1.0 / lg), R);
                    }
                })) return false;
        } else if (p.shapeMode == 2) {
            // RIGID: one best-fit rotation per strand, solved against the REST shape every
            // sweep, so there is no state for error to accumulate into -- if the strand
            // returns to its authored pose, R returns to identity. The root is pinned, so
            // rotation about it is the only rigid freedom, and a rigid motion changes no
            // inter-particle distance: the authored curl is preserved EXACTLY while the strand
            // is free to swing aside and clear a neighbour.
            if (!ft::parallelFor(nC, 32, [&](size_t c) {
                    const int b = first[c], n = cnt[c];
                    if (n < 3) return;
                    const Vec3 p0 = pos[(size_t)b], r0 = rest[(size_t)b];
                    double A[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                    for (int k = 1; k < n; ++k) {
                        const Vec3 q = pos[(size_t)b + (size_t)k] - p0;
                        const Vec3 w = rest[(size_t)b + (size_t)k] - r0;
                        A[0] += q.x*w.x; A[1] += q.x*w.y; A[2] += q.x*w.z;
                        A[3] += q.y*w.x; A[4] += q.y*w.y; A[5] += q.y*w.z;
                        A[6] += q.z*w.x; A[7] += q.z*w.y; A[8] += q.z*w.z;
                    }
                    double R[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
                    if (!detail::polarRotation(A, R)) {
                        // rank-deficient: a straight strand, whose twist about its own axis is
                        // undetermined AND irrelevant. The minimal rotation carrying the rest
                        // root->tip onto the current one is the right answer there.
                        for (int i = 0; i < 9; ++i) R[i] = (i % 4 == 0) ? 1.0 : 0.0;
                        const Vec3 wr = rest[(size_t)b + (size_t)n - 1] - r0;
                        const Vec3 wc = pos[(size_t)b + (size_t)n - 1] - p0;
                        const double lr = length(wr), lc = length(wc);
                        if (lr > 1e-15 && lc > 1e-15)
                            detail::rotateOnto(wr * (1.0 / lr), wc * (1.0 / lc), R);
                    }
                    for (int k = 1; k < n; ++k) {
                        const size_t j = (size_t)b + (size_t)k;
                        const Vec3 tgt = p0 + detail::applyR(R, rest[j] - r0);
                        pos[j] = pos[j] + (tgt - pos[j]) * (double)stiff[j];
                    }
                })) return false;
        } else {
            // the A/B control: the old global position spring
            if (!ft::parallelFor(nP, 1024, [&](size_t i) {
                    if (owner[i] < 0) return;
                    pos[i] = pos[i] + (rest[i] - pos[i]) * (double)stiff[i];
                })) return false;
        }

        // (d) separation. One-sided by construction: segment `si` reads every neighbour and
        //     writes only its own two particles, so no two threads touch the same slot.
        if (doSep) {
            std::fill(corr.begin(), corr.end(), Vec3{0, 0, 0});
            std::fill(cw.begin(), cw.end(), 0.0f);
            // PARALLEL OVER STRANDS, not segments. "One-sided" is not by itself enough to make
            // this race-free, and the rig caught me assuming it was: consecutive segments of a
            // strand SHARE a particle (seg k's p1 is seg k+1's p0), so two threads splitting a
            // strand's segments both write that slot. A strand owns its particles outright, so
            // dispatching by strand removes the race by construction and keeps the result
            // independent of the thread count -- which is the requirement, since the CPU and CUDA
            // backends must trace identical geometry.
            if (!ft::parallelFor(nC, 32, [&](size_t c) {
                    const Curve& cu = sc.curves[c];
                    for (int k = 0; k < cu.segCount; ++k) {
                        const size_t si = (size_t)cu.firstSeg + (size_t)k;
                        const size_t ia = (size_t)first[c] + (size_t)k, ib = ia + 1;
                        Vec3 acc{0, 0, 0};
                        int hits = 0;
                        for (uint32_t e = nbrOff[si]; e < nbrOff[si + 1]; ++e) {
                            const uint32_t sj = nbrIdx[e];
                            const int c2 = sc.curveSegs[sj].curveId;
                            if (c2 < 0 || c2 >= (int)nC) continue;
                            const int k2 = (int)((size_t)sj - (size_t)sc.curves[(size_t)c2].firstSeg);
                            const size_t ja = (size_t)first[c2] + (size_t)k2, jb = ja + 1;
                            Vec3 x1, x2;
                            const double d2 = detail::segSeg(pos[ia], pos[ib], pos[ja], pos[jb], x1, x2);
                            const double want = p.sepScale * (0.5 * (rad[ia] + rad[ib]) + 0.5 * (rad[ja] + rad[jb]));
                            if (d2 >= want * want) continue;
                            const double d = std::sqrt(std::max(d2, 0.0));
                            Vec3 nrm = (d > 1e-12) ? (x1 - x2) * (1.0 / d)
                                                   : Vec3{0.0, 1.0, 0.0};   // exactly coincident
                            acc = acc + nrm * (want - d);
                            ++hits;
                        }
                        if (hits) {
                            // SUM, not mean. Dividing by `hits` is what made this crawl: with four
                            // partners it applied ~6 % of the violation per sweep. The clamp below
                            // is what buys back the stability that averaging was providing.
                            const Vec3 half = acc * 0.5;
                            corr[ia] = corr[ia] + half; cw[ia] += 1.0f;
                            corr[ib] = corr[ib] + half; cw[ib] += 1.0f;
                        }
                    }
                })) return false;
            // Apply with a per-sweep CLAMP rather than an average: a summed violation can be
            // arbitrarily large where many fibers coincide, and an unclamped projection would throw
            // those strands across the groom. One fiber radius per sweep still allows 60 sweeps to
            // move a strand 2.6 mm, far more than the ~33 um a typical de-overlap needs.
            if (!ft::parallelFor(nP, 1024, [&](size_t i) {
                    if (!(cw[i] > 0.0f)) return;
                    Vec3 d = corr[i] * (1.0 / (double)cw[i]);
                    const double dl = length(d);
                    const double cap = p.maxStep * rad[i];
                    if (dl > cap && dl > 1e-30) d = d * (cap / dl);
                    pos[i] = pos[i] + d;
                })) return false;
        }

        // (d2) THE COLLECTIVE TERM: expand wherever the local volume fraction is too high.
        if (volOn) {
            for (size_t i = 0; i < gN; ++i) dens[i].store(0, std::memory_order_relaxed);
            if (!ft::parallelFor(nC, 32, [&](size_t c) {
                    const Curve& cu = sc.curves[c];
                    for (int k = 0; k < cu.segCount; ++k) {
                        const size_t ia = (size_t)first[c] + (size_t)k, ib = ia + 1;
                        const Vec3 a = pos[ia], b = pos[ib];
                        const Vec3 ab = b - a;
                        const double len = length(ab);
                        if (len < 1e-12) continue;
                        const double rr = 0.5 * (rad[ia] + rad[ib]);
                        const int ns = (int)std::ceil(len * invCell);
                        const int nsub = (ns < 1) ? 1 : ((ns > 64) ? 64 : ns);
                        const double vol = PI_D * rr * rr * len / (double)nsub;
                        const uint64_t q = (uint64_t)(vol * VSCALE);
                        if (q == 0) continue;
                        for (int t = 0; t < nsub; ++t)
                            splat(a + ab * (((double)t + 0.5) / (double)nsub), q);
                    }
                })) return false;
            // one write per particle, so the gather is deterministic by construction
            if (!ft::parallelFor(nP, 512, [&](size_t i) {
                    const double r0 = sampleRho(pos[i]);
                    if (!(r0 > p.packing)) return;
                    const Vec3 ex{cell, 0, 0}, ey{0, cell, 0}, ez{0, 0, cell};
                    const Vec3 g{ sampleRho(pos[i] + ex) - sampleRho(pos[i] - ex),
                                  sampleRho(pos[i] + ey) - sampleRho(pos[i] - ey),
                                  sampleRho(pos[i] + ez) - sampleRho(pos[i] - ez) };
                    const double gl = length(g);
                    if (gl < 1e-30) return;
                    const double over = (r0 - p.packing) / p.packing;
                    pos[i] = pos[i] - g * ((p.volume * (over < 1.0 ? over : 1.0) * cell) / gl);
                })) return false;
        }

        // (e) colliders
        if (!ctris.empty()) {
            if (!ft::parallelFor(nP, 512, [&](size_t i) {
                    const Vec3 q = pos[i];
                    const int cx = (int)std::floor(q.x / cg.cell), cy = (int)std::floor(q.y / cg.cell), cz = (int)std::floor(q.z / cg.cell);
                    double best = 1e300; Vec3 bp{0, 0, 0}, bn{0, 1, 0};
                    for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
                        size_t a, b;
                        cg.range(detail::Grid::code(cx + dx, cy + dy, cz + dz), a, b);
                        for (size_t e = a; e < b; ++e) {
                            const Tri& t = ctris[cg.idx[e]];
                            const Vec3 cp = detail::closestOnTri(q, t.v0, t.v1, t.v2);
                            const Vec3 dv = q - cp;
                            const double dd = dot(dv, dv);
                            // Tri::gn is filled by Scene::finalize(), which runs in scene.build()
                            // AFTER this pass, so the normal has to be computed here.
                            if (dd < best) { best = dd; bp = cp; bn = cross(t.v1 - t.v0, t.v2 - t.v0); }
                        }
                    }
                    if (best > 1e299) return;
                    const double nl = length(bn);
                    if (nl < 1e-18) return;
                    bn = bn * (1.0 / nl);
                    const Vec3 dv = q - bp;
                    const double sd = dot(dv, bn);
                    const double want = rad[i] + p.margin;
                    if (sd < want) pos[i] = bp + bn * want;      // outside, by the fiber radius
                })) return false;
        }

        // (f) roots never move
        if (!ft::parallelFor(nC, 256, [&](size_t c) {
                if (cnt[c] > 0) pos[(size_t)first[c]] = rest[(size_t)first[c]];
            })) return false;
    }

    // ---- 5. write back ---------------------------------------------------------------------
    double moved = 0.0, movedMax = 0.0;
    for (size_t c = 0; c < nC; ++c) {
        const Curve& cu = sc.curves[c];
        for (int k = 0; k < cu.segCount; ++k) {
            const size_t i = (size_t)first[c] + (size_t)k;
            CurveSeg& s = sc.curveSegs[(size_t)cu.firstSeg + (size_t)k];
            s.p0 = pos[i]; s.p1 = pos[i + 1];
        }
    }
    for (size_t i = 0; i < nP; ++i) {
        const double d = length(pos[i] - rest[i]);
        moved += d; movedMax = std::max(movedMax, d);
    }
    if (!p.cachePath.empty()) {
        // temp-then-rename: an interrupted write must not leave a truncated cache behind, because
        // the key hashes the INPUTS and so cannot notice a damaged payload.
        const std::string tmp = p.cachePath + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (f) {
            const uint64_t n = (uint64_t)nP;
            bool ok = std::fwrite(&detail::kCacheMagic, sizeof(uint64_t), 1, f) == 1 &&
                      std::fwrite(&key, sizeof key, 1, f) == 1 &&
                      std::fwrite(&n, sizeof n, 1, f) == 1 &&
                      std::fwrite(pos.data(), sizeof(Vec3), nP, f) == nP;
            std::fclose(f);
            std::error_code ec;
            if (ok) std::filesystem::rename(tmp, p.cachePath, ec);
            if (!ok || ec) std::filesystem::remove(tmp, ec);
        }
    }
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "%zu strand(s), %zu particle(s), %d sweep(s); moved mean %.3f mm, max %.3f mm%s",
                  nC, nP, p.iters, 1e3 * moved / (double)nP, 1e3 * movedMax,
                  capped ? "" : "");
    report += buf;
    if (capped)
        std::snprintf(buf, sizeof buf, "; %.1f %% of segments had more than %d candidates in reach"
                      " (the nearest %d are kept)", 100.0 * (double)capped / (double)nSeg, p.maxNbr, p.maxNbr);
    if (capped) report += buf;
    return true;
}

} // namespace settle
