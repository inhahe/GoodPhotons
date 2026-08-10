// Fur / groom generator — scatter strands over a surface.  (TODO.md §P1, stage 2)
//
// WHY THIS EXISTS.  `curve.h` gave ftrace a strand primitive; this file gives it a way to
// AUTHOR a groom.  Those are different problems.  A curve block is one hair written out by
// hand, and a coat is 10^4-10^6 hairs — nobody writes that as text, so without a generator
// the primitive can only ever be used for the handful of wires and guide hairs you are
// willing to type.  The generator is what turns "ftrace has fibers" into "ftrace has fur".
//
// WHAT IT IS.  A pure function of (surface, parameters, seed) -> strands.  There is no
// simulation and no solver: every strand is built independently from its own deterministic
// RNG stream, so the groom is reproducible, order-independent, and embarrassingly parallel.
// That is a deliberate ceiling — a real groom pipeline has collision, styling curves and
// interactive brushing — but it is the right first tier, because the shapes fur actually
// needs (grow along the normal, comb, sag under gravity, curl, clump into tufts) are all
// closed-form displacements of a straight strand, and closed-form is what parallelises.
//
// HOW A STRAND IS BUILT.  Roots are sampled AREA-UNIFORMLY over the target surface (a
// triangle CDF, or the analytic sphere), which is the only sampling that makes `density`
// mean anything and the only one that does not thin out over large triangles.  Each root
// then grows `points` control points along a direction derived from the surface normal,
// displaced by four independent closed-form terms — gravity droop, comb, helical curl and
// clump — and handed to `tessellateCurve` exactly as a hand-authored `curve` would be.  A
// generated strand is therefore not a second kind of geometry: it IS a Curve, and every
// downstream consumer (BVH, CPU tracer, CUDA megakernel, raster preview) sees strands it
// already knows how to draw, with no new code path anywhere.
//
// CLUMPING gets a real spatial structure rather than a hash, because the difference is
// visible: hashing the root cell gives cube-shaped tufts on a grid, while nearest-guide
// assignment gives Voronoi tufts, which is what hair does.  Guides are scattered by the
// same area-uniform sampler and looked up through a CSR uniform grid, so assignment is
// O(1) per strand and the whole generator stays linear in strand count.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "curve.h"
#include "geometry.h"
#include "linalg.h"
#include "parallel.h"
#include "rng.h"

namespace furdet { inline constexpr double kPi = 3.14159265358979323846; }
using furdet::kPi;

// ---------------------------------------------------------------------------
// The surface strands are scattered over.
// ---------------------------------------------------------------------------
// Either a triangle range (a named `mesh`/`quad`/`triangle` object's world triangles) or
// an analytic sphere.  The sphere is not tessellated: a `sphere` target samples the real
// surface with the real normal, so a furred ball has no faceting in its root distribution
// and no tessellation-density bias in its coat.
struct FurSurface {
    const Tri* tris  = nullptr;      // world-space triangles (Scene::tris + triStart)
    size_t     nTris = 0;
    bool       isSphere = false;
    Vec3       center{0, 0, 0};      // sphere target only
    double     radius = 1.0;
};

// One authored `fur { }` block, already converted to internal (metre) units.
struct FurSpec {
    int         matId = 0;
    std::string name;                // authored block name (diagnostics)

    // Count.  `count` wins when > 0; otherwise `density` (strands per authored unit^2)
    // times the target's area.  Density is the parameter that survives editing the
    // model — it keeps the coat the same fineness when the creature is rescaled.
    long long   count   = 0;
    double      density = 0.0;       // strands per INTERNAL unit^2 (loader converts)
    uint64_t    seed    = 0;

    int         points = 5;          // control points per strand (>= 2)
    int         subdiv = 2;          // round cones per span
    CurveBasis  basis  = CurveBasis::CatmullRom;

    double      length = 0.05;       // root->tip length
    double      lengthJitter = 0.2;  // +/- fraction of `length`, uniform

    double      radius    = 0.0008;  // root radius
    double      radiusTip = -1.0;    // tip radius; < 0 => 0.25 * radius (fur tapers)

    double      lift  = 1.0;         // 1 = straight out along N, 0 = flat along the surface
    double      jitter = 0.15;       // random tilt of the growth direction

    Vec3        comb{0, 0, 0};       // world combing direction (zero = no comb)
    double      combAmount = 0.35;   // tip displacement along `comb`, as a fraction of length

    Vec3        gravity{0, -1, 0};   // world "down"
    double      droop = 0.25;        // tip sag along `gravity`, as a fraction of length

    double      curl = 0.0;          // helix radius, as a fraction of length
    double      curlFreq = 3.0;      // turns over the strand's length

    double      clump = 0.0;         // 0 = none, 1 = tips meet the guide exactly
    double      clumpSize = 0.02;    // tuft radius (one guide per pi*clumpSize^2 of area)

    double      rootOffset = 0.0;    // push the root along N (bed the strand into the skin)

    // BALD ZONES — spheres no strand may enter.  A coat is grown per BODY PART, but the
    // features that must stay bare (an eye, a nose leather, a scar) are separate little
    // spheres sitting ON that part, and the part's own fur does not know they are there:
    // it roots area-uniformly over the whole target, including the disc the eye overlaps,
    // and every strand rooted around that disc then grows straight across the eyeball.
    // The result is an eye peppered with hair — visibly wrong in a way no combination of
    // length / lift / comb can fix, because the problem is WHERE the roots are, not which
    // way they point.
    //
    // The cull is per strand and tests the WHOLE strand, not just its root: a hair rooted
    // outside a zone that arcs through it under droop/comb/clump is exactly the hair that
    // shows on an eyeball, so keeping it would defeat the parameter.  Culling cannot bias
    // the surviving coat either — roots are still drawn area-uniformly and every survivor
    // is untouched, so the fur outside a zone is bit-for-bit what it was without one.
    struct BaldZone { Vec3 center{0, 0, 0}; double radius = 0.0; };
    std::vector<BaldZone> bald;
};

// Does a strand (a polyline of `n` control points, thickened by `rad`) reach into any bald
// zone?  Tested SPAN-wise rather than point-wise: a small zone can sit entirely between two
// control points, and a point test would happily let the hair skewer it.
inline bool furStrandHitsBald(const FurSpec& spec, const Vec3* cp, int n, double rad) {
    if (spec.bald.empty()) return false;
    for (const FurSpec::BaldZone& z : spec.bald) {
        const double rr = z.radius + rad, rr2 = rr * rr;
        const Vec3 d0 = cp[0] - z.center;
        if (dot(d0, d0) <= rr2) return true;              // root inside: the common case
        for (int k = 0; k + 1 < n; ++k) {
            const Vec3 a = cp[k], ab = cp[k + 1] - a, ac = z.center - a;
            const double denom = dot(ab, ab);
            double t = (denom > 1e-24) ? dot(ac, ab) / denom : 0.0;
            t = std::min(std::max(t, 0.0), 1.0);
            const Vec3 q = ac - ab * t;                   // centre -> closest point on span
            if (dot(q, q) <= rr2) return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Area-uniform root sampling
// ---------------------------------------------------------------------------
// A prefix CDF over triangle area.  Built once per `fur` block; a strand costs one binary
// search.  Sampling proportional to area (rather than picking a triangle uniformly) is
// what makes a coat even over a mesh whose tessellation is not: a low-poly belly and a
// dense face must grow the same hairs per square centimetre, or the model's topology
// leaks into the look.
struct FurAreaCdf {
    std::vector<double> cdf;   // size nTris+1, cdf[0] = 0, cdf[nTris] = total area
    double total = 0.0;

    void build(const FurSurface& s) {
        cdf.assign(s.nTris + 1, 0.0);
        for (size_t i = 0; i < s.nTris; ++i) {
            const Tri& t = s.tris[i];
            const double a = 0.5 * length(cross(t.v1 - t.v0, t.v2 - t.v0));
            cdf[i + 1] = cdf[i] + a;
        }
        total = s.nTris ? cdf[s.nTris] : 0.0;
    }
    // Index of the triangle owning cumulative area `x` in [0, total).
    size_t pick(double x) const {
        size_t lo = 0, hi = cdf.size() - 1;          // invariant: cdf[lo] <= x < cdf[hi]
        while (hi - lo > 1) {
            size_t mid = (lo + hi) >> 1;
            if (cdf[mid] <= x) lo = mid; else hi = mid;
        }
        return lo;
    }
};

// Total area of the target (drives `density`, and is reported at load).
inline double furSurfaceArea(const FurSurface& s, const FurAreaCdf& cdf) {
    if (s.isSphere) return 4.0 * kPi * s.radius * s.radius;
    return cdf.total;
}

// Same, standalone — for callers that only want the number (the loader's log line).
inline double furTargetArea(const FurSurface& s) {
    if (s.isSphere) return 4.0 * kPi * s.radius * s.radius;
    double a = 0.0;
    for (size_t i = 0; i < s.nTris; ++i) {
        const Tri& t = s.tris[i];
        a += 0.5 * length(cross(t.v1 - t.v0, t.v2 - t.v0));
    }
    return a;
}

// One area-uniform root: position + unit shading normal.
//
// The barycentric map is the standard sqrt warp (b0 = 1-sqrt(u)); the normal is the
// BARYCENTRIC-INTERPOLATED shading normal, not the geometric one, so fur on a
// smooth-shaded low-poly mesh follows the smooth surface instead of stepping at every
// facet edge — the single most visible thing this function gets to decide.
inline void furSampleRoot(const FurSurface& s, const FurAreaCdf& cdf, Pcg32& rng,
                          Vec3& p, Vec3& n) {
    if (s.isSphere) {
        // Uniform on the sphere: z uniform in [-1,1], azimuth uniform (Archimedes).
        const double z = 1.0 - 2.0 * rng.uniform();
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double ph = 2.0 * kPi * rng.uniform();
        n = Vec3(r * std::cos(ph), r * std::sin(ph), z);
        p = s.center + n * s.radius;
        return;
    }
    const size_t ti = cdf.pick(rng.uniform() * cdf.total);
    const Tri& t = s.tris[ti];
    double u = rng.uniform(), v = rng.uniform();
    const double su = std::sqrt(u);
    const double b0 = 1.0 - su, b1 = su * (1.0 - v), b2 = su * v;
    p = t.v0 * b0 + t.v1 * b1 + t.v2 * b2;
    // LOAD-ORDER TRAP, and it is silent: Tri::finalize() — which computes `gn` and back-
    // fills absent shading normals — does not run until Scene::build(), long AFTER the
    // loader's fur sweep. So `t.gn` is still (0,0,0) here for every primitive that did not
    // author normals (all quads and triangles, and any mesh without `vn`). Reading it
    // would normalize a zero vector, make the whole strand NaN, and then lose it in
    // tessellateCurve's `dot(dp,dp) > 0` coincidence test — an entire groom silently
    // generating ZERO strands, with no error anywhere. Derive the geometric normal from
    // the vertices instead, and use shading normals only when they are actually present.
    Vec3 sn = t.n0 * b0 + t.n1 * b1 + t.n2 * b2;
    if (dot(sn, sn) < 1e-24) sn = cross(t.v1 - t.v0, t.v2 - t.v0);
    if (dot(sn, sn) < 1e-24) sn = Vec3(0, 1, 0);      // degenerate triangle
    n = normalize(sn);
}

// ---------------------------------------------------------------------------
// Strand shaping
// ---------------------------------------------------------------------------
// Fill `out` with `spec.points` world control points for one strand rooted at (p, n).
//
// Every displacement is a closed-form function of the normalised arc parameter t, applied
// on top of a straight strand — which is what lets a strand be built in isolation with no
// neighbour queries and no iteration.  Growth is LINEAR in t and every bend is QUADRATIC
// in t, so the root leaves the skin along the growth direction (a strand that bends at its
// root reads as broken) while the tip carries the full displacement.
inline void furBuildStrand(const FurSpec& spec, const Vec3& p, const Vec3& n,
                           Pcg32& rng, Vec3* out) {
    Vec3 T, B; onb(n, T, B);

    // Growth direction: the normal tipped toward a random tangential azimuth by (1-lift),
    // then perturbed within a small cone by `jitter`.  Both are drawn BEFORE any other
    // use of the stream so a strand's direction does not change when an unrelated
    // parameter (e.g. curl) is switched on — stable seeds are what makes a groom editable.
    const double az = 2.0 * kPi * rng.uniform();
    const Vec3 flat = T * std::cos(az) + B * std::sin(az);
    Vec3 dir = n * spec.lift + flat * (1.0 - spec.lift);
    if (spec.jitter > 0.0) {
        const double jx = (2.0 * rng.uniform() - 1.0) * spec.jitter;
        const double jy = (2.0 * rng.uniform() - 1.0) * spec.jitter;
        dir = dir + T * jx + B * jy;
    }
    if (dot(dir, dir) < 1e-24) dir = n;
    dir = normalize(dir);
    // Never grow INTO the surface: a strand whose direction points below the tangent plane
    // spends its whole length inside the skin and renders as a shadow with no hair.  Lift
    // it back to a shallow grazing angle rather than rejecting the sample (rejection would
    // silently bias the root distribution, which is the one thing this file must not do).
    const double minCos = 0.08;
    if (dot(dir, n) < minCos) dir = normalize(dir + n * (minCos - dot(dir, n)));

    const double jl = 1.0 + (2.0 * rng.uniform() - 1.0) * spec.lengthJitter;
    const double L  = spec.length * std::max(0.05, jl);
    const double phase = 2.0 * kPi * rng.uniform();          // per-strand curl phase

    // Curl rides in the plane perpendicular to the growth direction.
    Vec3 U, V; onb(dir, U, V);
    const double curlR = spec.curl * L;
    const double c0 = std::cos(phase), s0 = std::sin(phase);

    const Vec3 root = p + n * spec.rootOffset;
    const int  N = spec.points;
    for (int k = 0; k < N; ++k) {
        const double t = (N > 1) ? (double)k / (double)(N - 1) : 0.0;
        Vec3 q = root + dir * (L * t);
        if (spec.droop != 0.0) q = q + spec.gravity * (spec.droop * L * t * t);
        if (spec.combAmount != 0.0) q = q + spec.comb * (spec.combAmount * L * t * t);
        if (curlR > 0.0) {
            // Subtract the t=0 value so the helix leaves the root where the root is;
            // without it the whole strand is translated off the surface by `curlR`.
            const double a = 2.0 * kPi * spec.curlFreq * t + phase;
            q = q + U * (curlR * (std::cos(a) - c0)) + V * (curlR * (std::sin(a) - s0));
        }
        out[k] = q;
    }
}

// ---------------------------------------------------------------------------
// Clump guides + a CSR uniform grid for nearest-guide lookup
// ---------------------------------------------------------------------------
// Guides are ordinary strands (same sampler, same shaping, their own RNG salt).  A strand
// blends toward its nearest guide with weight `clump * t`, so roots stay exactly where the
// area-uniform sampler put them — clumping must change the SHAPE of a coat, never its
// density — while tips converge into a tuft.
struct FurGuides {
    std::vector<Vec3> pts;      // guideIdx*points + k
    std::vector<Vec3> roots;
    int points = 0;

    // CSR uniform grid over guide roots.
    Vec3   gmin{0, 0, 0};
    double cell = 1.0;
    int    nx = 1, ny = 1, nz = 1;
    std::vector<int> cellStart;   // size nx*ny*nz+1
    std::vector<int> cellItem;    // guide indices, bucketed

    int cellIndex(int ix, int iy, int iz) const { return (iz * ny + iy) * nx + ix; }
    void clampCell(int& ix, int& iy, int& iz) const {
        ix = std::min(std::max(ix, 0), nx - 1);
        iy = std::min(std::max(iy, 0), ny - 1);
        iz = std::min(std::max(iz, 0), nz - 1);
    }

    void buildGrid(double wantCell) {
        const size_t G = roots.size();
        gmin = Vec3(1e300, 1e300, 1e300);
        Vec3 gmax(-1e300, -1e300, -1e300);
        for (const Vec3& r : roots) {
            gmin = Vec3(std::min(gmin.x, r.x), std::min(gmin.y, r.y), std::min(gmin.z, r.z));
            gmax = Vec3(std::max(gmax.x, r.x), std::max(gmax.y, r.y), std::max(gmax.z, r.z));
        }
        if (G == 0) { gmin = Vec3(0, 0, 0); gmax = Vec3(1, 1, 1); }
        const Vec3 ext = gmax - gmin;
        cell = std::max(wantCell, 1e-9);
        // Cap the grid so a tiny clump_size on a large model cannot allocate a
        // billion-cell array: coarsening the grid only costs a slightly wider search.
        const double maxCells = 2.0e6;
        for (int guard = 0; guard < 64; ++guard) {
            double cx = std::floor(ext.x / cell) + 1.0;
            double cy = std::floor(ext.y / cell) + 1.0;
            double cz = std::floor(ext.z / cell) + 1.0;
            if (cx * cy * cz <= maxCells) break;
            cell *= 2.0;
        }
        nx = std::max(1, (int)std::floor(ext.x / cell) + 1);
        ny = std::max(1, (int)std::floor(ext.y / cell) + 1);
        nz = std::max(1, (int)std::floor(ext.z / cell) + 1);
        const size_t nc = (size_t)nx * ny * nz;
        cellStart.assign(nc + 1, 0);
        std::vector<int> ci(G);
        for (size_t i = 0; i < G; ++i) {
            const Vec3 d = roots[i] - gmin;
            int ix = (int)std::floor(d.x / cell), iy = (int)std::floor(d.y / cell),
                iz = (int)std::floor(d.z / cell);
            clampCell(ix, iy, iz);
            ci[i] = cellIndex(ix, iy, iz);
            ++cellStart[ci[i] + 1];
        }
        for (size_t c = 0; c < nc; ++c) cellStart[c + 1] += cellStart[c];
        cellItem.assign(G, 0);
        std::vector<int> cursor(cellStart.begin(), cellStart.end() - 1);
        for (size_t i = 0; i < G; ++i) cellItem[cursor[ci[i]]++] = (int)i;
    }

    // Nearest guide root to `p`, or -1 if there are none.  Rings outward until a hit is
    // found and then one ring further, because the nearest guide can sit in a ring beyond
    // the first non-empty one (the classic off-by-one in grid nearest-neighbour search).
    int nearest(const Vec3& p) const {
        if (roots.empty()) return -1;
        const Vec3 d = p - gmin;
        int cx = (int)std::floor(d.x / cell), cy = (int)std::floor(d.y / cell),
            cz = (int)std::floor(d.z / cell);
        clampCell(cx, cy, cz);
        int best = -1; double bestD2 = 1e300;
        const int maxR = std::max(nx, std::max(ny, nz));
        int stopAfter = -1;
        for (int r = 0; r <= maxR; ++r) {
            for (int iz = cz - r; iz <= cz + r; ++iz) {
                if (iz < 0 || iz >= nz) continue;
                for (int iy = cy - r; iy <= cy + r; ++iy) {
                    if (iy < 0 || iy >= ny) continue;
                    for (int ix = cx - r; ix <= cx + r; ++ix) {
                        if (ix < 0 || ix >= nx) continue;
                        // Only the shell of the (2r+1)^3 box is new at radius r.
                        const bool shell = (std::abs(ix - cx) == r || std::abs(iy - cy) == r ||
                                            std::abs(iz - cz) == r);
                        if (!shell) continue;
                        const int c = cellIndex(ix, iy, iz);
                        for (int k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                            const Vec3 q = roots[cellItem[k]] - p;
                            const double d2 = dot(q, q);
                            if (d2 < bestD2) { bestD2 = d2; best = cellItem[k]; }
                        }
                    }
                }
            }
            if (best >= 0 && stopAfter < 0) stopAfter = r + 1;
            if (stopAfter >= 0 && r >= stopAfter) break;
        }
        return best;
    }
};

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------
// Appends `spec.count` strands to `curves`/`segs`.  Returns the number of strands added,
// or -1 if the caller should abandon the load (`ftrace -stop` during generation).
//
// PARALLELISM.  Strand k is a pure function of (spec, surface, k), so the build is a
// parallelFor over strands into a PREALLOCATED slice — no locks, no per-strand allocation
// in the common case, and a result that does not depend on how the work was scheduled.
// `tessellateCurve` may legitimately emit fewer cones than the upper bound (it drops
// exactly-coincident samples), so each strand records its own count and a serial
// compaction pass closes the gaps; when nothing was dropped — which is every ordinary
// groom — the compaction degenerates to a single bulk append.
inline long long generateFur(const FurSpec& specIn, const FurSurface& surf,
                             std::vector<Curve>& curves, std::vector<CurveSeg>& segs,
                             std::string* err = nullptr, long long* baldCulled = nullptr) {
    FurSpec spec = specIn;
    // Drop no-op zones up front so the per-strand test never sees one, and so a typo'd
    // `bald "name"` that resolved to nothing costs zero per hair.
    spec.bald.erase(std::remove_if(spec.bald.begin(), spec.bald.end(),
                                   [](const FurSpec::BaldZone& z) { return !(z.radius > 0.0); }),
                    spec.bald.end());
    if (baldCulled) *baldCulled = 0;
    if (spec.points < 2) spec.points = 2;
    if (spec.subdiv < 1) spec.subdiv = 1;
    if (spec.subdiv > 256) spec.subdiv = 256;
    if (spec.radiusTip < 0.0) spec.radiusTip = 0.25 * spec.radius;
    if (spec.lift < 0.0) spec.lift = 0.0;
    if (spec.lift > 1.0) spec.lift = 1.0;
    if (spec.clump < 0.0) spec.clump = 0.0;
    if (spec.clump > 1.0) spec.clump = 1.0;
    if (dot(spec.gravity, spec.gravity) > 0.0) spec.gravity = normalize(spec.gravity);
    if (dot(spec.comb, spec.comb) > 0.0) spec.comb = normalize(spec.comb);
    else spec.combAmount = 0.0;

    const int spans = curveSpanCount(spec.basis, spec.points);
    if (spans <= 0) {
        if (err) *err = "fur: " + std::to_string(spec.points) + " points per strand is not a valid " +
                        curveBasisName(spec.basis) + " strand";
        return -1;
    }
    if (spec.basis == CurveBasis::Linear) spec.subdiv = 1;
    const int segsPerStrand = spans * spec.subdiv;

    FurAreaCdf cdf;
    if (!surf.isSphere) cdf.build(surf);
    const double area = furSurfaceArea(surf, cdf);
    if (!(area > 0.0)) {
        if (err) *err = "fur: target surface has zero area";
        return -1;
    }

    long long count = spec.count;
    if (count <= 0) count = (long long)std::llround(spec.density * area);
    if (count <= 0) {
        if (err) *err = "fur: needs a positive `count` or `density`";
        return -1;
    }

    // --- clump guides -------------------------------------------------------
    // One guide per pi*clumpSize^2 of surface, i.e. `clump_size` is the tuft RADIUS. That
    // is the parameter an artist can predict from looking at the model, unlike a guide
    // count, which changes meaning the moment the model is rescaled.
    FurGuides guides;
    const bool doClump = spec.clump > 0.0 && spec.clumpSize > 0.0;
    if (doClump) {
        long long G = (long long)std::llround(area / (kPi * spec.clumpSize * spec.clumpSize));
        G = std::min(std::max(G, 1LL), count);
        guides.points = spec.points;
        guides.pts.resize((size_t)G * spec.points);
        guides.roots.resize((size_t)G);
        for (long long g = 0; g < G; ++g) {
            Pcg32 rng;
            rng.seed(mix64(spec.seed ^ 0x9E3779B97F4A7C15ULL), mix64((uint64_t)g * 2 + 1));
            Vec3 p, n;
            furSampleRoot(surf, cdf, rng, p, n);
            guides.roots[(size_t)g] = p;
            furBuildStrand(spec, p, n, rng, &guides.pts[(size_t)g * spec.points]);
        }
        guides.buildGrid(spec.clumpSize);
    }

    // --- strands ------------------------------------------------------------
    const size_t base = segs.size();
    const size_t curveBase = curves.size();
    segs.resize(base + (size_t)count * segsPerStrand);
    std::vector<int> nseg((size_t)count, 0);
    // A bald cull and a degenerate strand both land as nseg == 0, but only one of them is
    // something the author asked for, so they are tallied apart — a `bald` that quietly
    // ate the whole coat has to be visible in the load log, not inferred from a low count.
    // Only paid for when a zone exists (one byte per strand).
    std::vector<uint8_t> baldFlag(spec.bald.empty() ? 0 : (size_t)count, 0);

    const bool ok = ft::parallelFor((size_t)count, 256, [&](size_t i) {
        // Scratch reused across every strand this thread builds: tessellateCurve appends
        // to a vector, and one malloc per hair would dominate a million-strand groom.
        static thread_local std::vector<CurveSeg> scratch;
        static thread_local std::vector<Vec3>     cp;
        static thread_local std::vector<double>   radii;
        scratch.clear();
        cp.resize((size_t)spec.points);
        radii.resize((size_t)spec.points);

        Pcg32 rng;
        rng.seed(mix64((uint64_t)i * 2 + 1), mix64(spec.seed + (uint64_t)i));
        Vec3 p, n;
        furSampleRoot(surf, cdf, rng, p, n);
        furBuildStrand(spec, p, n, rng, cp.data());

        if (doClump) {
            const int g = guides.nearest(p);
            if (g >= 0) {
                const Vec3* gpts = &guides.pts[(size_t)g * guides.points];
                for (int k = 0; k < spec.points; ++k) {
                    const double t = (spec.points > 1) ? (double)k / (spec.points - 1) : 0.0;
                    const double w = spec.clump * t;
                    cp[(size_t)k] = cp[(size_t)k] * (1.0 - w) + gpts[k] * w;
                }
            }
        }
        // Bald zones are tested AFTER clumping, because clumping is what pulls a tip
        // sideways into a zone its own root pointed clear of. `nseg[i] = 0` routes the
        // strand into the same compaction path a degenerate strand already takes, so
        // there is no second way for a hair to disappear.
        if (furStrandHitsBald(spec, cp.data(), spec.points, spec.radius)) {
            nseg[i] = 0; baldFlag[i] = 1; return;
        }
        for (int k = 0; k < spec.points; ++k) {
            const double t = (spec.points > 1) ? (double)k / (spec.points - 1) : 0.0;
            radii[(size_t)k] = spec.radius + (spec.radiusTip - spec.radius) * t;
        }
        const int added = tessellateCurve(cp, radii, spec.basis, spec.subdiv, spec.matId,
                                          (int)(curveBase + i), scratch);
        nseg[i] = added;
        if (added > 0)
            std::memcpy(&segs[base + i * (size_t)segsPerStrand], scratch.data(),
                        (size_t)added * sizeof(CurveSeg));
    });
    if (!ok) { segs.resize(base); return -1; }
    if (baldCulled) {
        long long nb = 0;
        for (uint8_t f : baldFlag) nb += f;
        *baldCulled = nb;
    }

    // Compaction.  The fast path (nothing dropped) is the overwhelmingly common one and
    // costs a single comparison per strand.
    bool dense = true;
    for (long long i = 0; i < count; ++i) if (nseg[(size_t)i] != segsPerStrand) { dense = false; break; }
    size_t write = base;
    if (dense) {
        write = segs.size();
    } else {
        for (long long i = 0; i < count; ++i) {
            const size_t rd = base + (size_t)i * segsPerStrand;
            const int m = nseg[(size_t)i];
            if (m > 0 && rd != write) std::memmove(&segs[write], &segs[rd], (size_t)m * sizeof(CurveSeg));
            write += (size_t)m;
        }
        segs.resize(write);
    }

    curves.reserve(curves.size() + (size_t)count);
    size_t first = base;
    long long live = 0;
    for (long long i = 0; i < count; ++i) {
        const int m = nseg[(size_t)i];
        if (m <= 0) continue;                       // a fully degenerate strand: drop it
        Curve c;
        c.firstSeg = (int)first;
        c.segCount = m;
        c.matId    = spec.matId;
        c.basis    = spec.basis;
        c.name     = spec.name;
        curves.push_back(std::move(c));
        first += (size_t)m;
        ++live;
    }
    // A dropped strand shifts every later strand's Curve index, so the curveId baked into
    // the segments would no longer point at its owner.  Rather than leave a stale id (the
    // kind of bug that only shows up in a diagnostic months later), renumber.
    if (live != count) {
        for (size_t ci = curveBase; ci < curves.size(); ++ci) {
            const Curve& c = curves[ci];
            for (int k = 0; k < c.segCount; ++k) segs[(size_t)c.firstSeg + k].curveId = (int)ci;
        }
    }
    return live;
}
