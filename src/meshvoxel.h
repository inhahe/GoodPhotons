// meshvoxel.h — SOLID voxelization of a triangle mesh: fog containment (VdbGrid) and
// signed-distance bakes (PatGrid).
//
// Why this exists: `medium { bounds { object "<name>" } }` shapes fog to a named scene
// object. A named `sphere` gets its exact analytic bound and a named `isosurface` gets
// exact field membership, but a named MESH used to fall back to its world AABB — a fog
// BOX wearing the mesh's name, which is useless for the thing meshes are actually for
// (a cloud, a body of smoke, any imported silhouette). This bakes true containment.
//
// The result is deliberately a `VdbGrid`: that is already the renderer's dense-volume
// vehicle, already sampled trilinearly by identical CPU/GPU code, and already uploaded
// to the device as a sparse brick lattice. Routing mesh containment through it means the
// GPU, the majorant machinery and delta/ratio tracking all work with no new plumbing.
//
// Method — SIGNED-CROSSING (generalized winding) x-scanlines, not parity:
// for every lattice row (j,k) we collect the x of each triangle the +x ray through that
// row's voxel CENTRES crosses, tagged +1 when the triangle faces the ray (entering) and
// -1 when it faces away (leaving). Sorting by x and accumulating gives the winding
// number along the row; voxels where it is nonzero are inside. Parity would be simpler
// but breaks on exactly the meshes people import: a model built from several closed
// bodies (cloud1.glb has two) or with self-intersecting shells double-toggles and comes
// out hollow in the overlap. Winding treats those as a union, which is what the eye
// expects. Triangles are visited once each and scattered into the rows they cover, so
// the cost is O(tris + covered rows) rather than O(rows x tris) — a 1.9M-triangle cloud
// bakes in about a second.
//
// Occupancy is stored 1 inside / 0 outside with a one-voxel zero shell around the whole
// lattice, so the trilinear sample ramps 1 -> 0 across the boundary voxel and the
// membership threshold (0.5) lands on a smooth interpolated isosurface rather than on
// stair-stepped voxel faces.
//
// Accuracy, measured against exact lattice-point counts (scraps/_voxexpect.py): a sphere,
// two overlapping spheres and two disjoint spheres all reproduce their expected solid
// fraction to within the mesh's own faceting error (<= 0.1 pt at res 200). The one case
// that differs is a surface lying EXACTLY on a lattice-centre plane (an axis-aligned box
// face): membership there is a floating-point coin flip, and this errs consistently on the
// side of EROSION, losing the outermost plane -- the safe direction for a fog bound, which
// should never leak outside the mesh, and sub-voxel against a boundary the trilinear ramp
// deliberately softens anyway.
//
// FEATHERING (`feather <metres>`): the trilinear ramp above spans exactly ONE voxel, so the
// bake is effectively a hard silhouette -- a cloud bounded this way has a crisp cut-out edge
// where real cloud has a diffuse one. `featherGrid` replaces the 0/1 occupancy with
// `saturate(distance-to-the-outside / feather)`, i.e. the fog reaches full density only
// `feather` metres INSIDE the surface and ramps smoothly to zero at it. It is a pure
// post-pass on the lattice, so nothing downstream changes: the same fp16 grid, the same
// trilinear sample, the same majorant (the maximum is still 1). Distance comes from an exact
// Euclidean distance transform (Felzenszwalb & Huttenlocher 2012's separable lower-envelope
// algorithm, three O(n) passes over the axes), not a chamfer approximation -- an approximate
// metric shows up as faceted banding in the very gradient the feature exists to produce.
//
// SECOND TENANT — `bakeSignedDistance` (bottom of this file), which backs FTSL's `sdf`
// element. It answers a different question ("how far is this point from that mesh, and am I
// inside it?") and produces a plain float lattice for `PatGrid` rather than a `VdbGrid`, but
// it lives here because it needs the SIGN, and the signed-crossing scanline above is the one
// piece of genuinely delicate code that neither the fog bound nor an SDF should re-derive.
// `voxelizeSolidInto` is the shared core, factored out so each caller can pick its own
// lattice (the fog bound wants the mesh AABB plus a one-voxel shell; an SDF wants a box
// padded far enough that the field is useful outside the object).
//
// Note that it does NOT reuse the feather pass's EDT, and must not: F&H is a distance
// transform only for BINARY seeds. See the comment on `bakeSignedDistance` for the measured
// 34 %-short field that resulted from assuming otherwise.
#pragma once
#include "geometry.h"
#include "parallel.h"
#include "vdbgrid.h"
#include <algorithm>
#include <vector>
#include <cmath>

namespace meshvox {

// One ray-vs-triangle crossing along a +x scanline: where it happened and whether it
// entered (+1) or left (-1) the solid.
struct Crossing {
    float x;
    int8_t w;
    bool operator<(const Crossing& o) const { return x < o.x; }
};

// Inside/outside of `tris[triStart .. triStart+triCount)` on an EXPLICIT lattice: sample
// (i,j,k) sits at world `w0 + (i,j,k)*h`, and `occ` (nx*ny*nz, C order with x fastest) is
// written 1 inside / 0 outside. Factored out of `voxelizeSolid` so the signed-distance
// bake below can reuse the winding logic on a lattice of its own choosing — the medium
// bound wants the mesh's own AABB plus a one-voxel shell, an SDF wants a box PADDED far
// enough out that the field is useful outside the object, and this is the one piece of
// genuinely delicate code (see the `det` note below) that neither should re-derive.
inline void voxelizeSolidInto(const Tri* tris, size_t triStart, size_t triCount,
                              const Vec3& w0, double h, int nx, int ny, int nz,
                              std::vector<uint8_t>& occ) {
    occ.assign((size_t)nx * ny * nz, 0);
    if (!tris || triCount == 0 || !(h > 0.0)) return;

    // --- Scatter every triangle into the scanlines it covers -----------------
    // Row (j,k) is the +x ray through world y = w0.y + j*h, z = w0.z + k*h.
    std::vector<std::vector<Crossing>> rows((size_t)ny * nz);
    for (size_t t = 0; t < triCount; ++t) {
        const Tri& tr = tris[triStart + t];
        const Vec3 &a = tr.v0, &b = tr.v1, &c = tr.v2;
        // Edge vectors in the (y,z) projection; the ray direction is +x, so the
        // projected area is the determinant below.
        const double e1y = b.y - a.y, e1z = b.z - a.z;
        const double e2y = c.y - a.y, e2z = c.z - a.z;
        // det is exactly cross(v1-v0, v2-v0).x, i.e. the x-component of the UNNORMALISED
        // geometric normal: |det| = 2*area projected along +x (zero when the triangle is
        // edge-on and crosses nothing) and its SIGN is the facing.
        const double det = e1y * e2z - e1z * e2y;
        if (std::fabs(det) < 1e-20) continue;          // edge-on to +x: contributes no crossing
        const double inv = 1.0 / det;
        // Winding contribution: +1 when the triangle faces the ray (normal pointing back
        // along +x => the ray is ENTERING the solid), -1 when it faces away.
        //
        // Read from `det`, deliberately NOT from `tr.gn`: Tri::gn is filled in by
        // Tri::finalize(), which Scene::build() only runs once ALL geometry is loaded —
        // long after the loader's medium sweep calls this. Using gn here silently read
        // (0,0,0) for every triangle, so every crossing got the same sign, the winding
        // never returned to zero, and each scanline filled solid from its first crossing
        // to its last: a mesh's x-convex hull instead of the mesh. det is the same
        // quantity computed from the vertices in hand, so it is correct at any load stage.
        const int8_t wsign = (det < 0.0) ? (int8_t)+1 : (int8_t)-1;

        double tlo_y = std::min(a.y, std::min(b.y, c.y)), thi_y = std::max(a.y, std::max(b.y, c.y));
        double tlo_z = std::min(a.z, std::min(b.z, c.z)), thi_z = std::max(a.z, std::max(b.z, c.z));
        int j0 = (int)std::ceil((tlo_y - w0.y) / h), j1 = (int)std::floor((thi_y - w0.y) / h);
        int k0 = (int)std::ceil((tlo_z - w0.z) / h), k1 = (int)std::floor((thi_z - w0.z) / h);
        j0 = std::max(j0, 0); j1 = std::min(j1, ny - 1);
        k0 = std::max(k0, 0); k1 = std::min(k1, nz - 1);

        for (int k = k0; k <= k1; ++k) {
            const double pz = w0.z + k * h;
            for (int j = j0; j <= j1; ++j) {
                const double py = w0.y + j * h;
                // Barycentric solve in the (y,z) projection.
                const double ry = py - a.y, rz = pz - a.z;
                const double u = (ry * e2z - rz * e2y) * inv;
                const double v = (rz * e1y - ry * e1z) * inv;
                if (u < 0.0 || v < 0.0 || u + v > 1.0) continue;
                const double px = a.x + u * (b.x - a.x) + v * (c.x - a.x);
                rows[(size_t)k * ny + j].push_back({(float)px, wsign});
            }
        }
    }

    // --- Fill each row by accumulated winding --------------------------------
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            auto& cr = rows[(size_t)k * ny + j];
            if (cr.size() < 2) continue;
            std::sort(cr.begin(), cr.end());
            int wind = 0;
            for (size_t m = 0; m + 1 < cr.size(); ++m) {
                wind += cr[m].w;
                if (wind == 0) continue;                       // outside this span
                // Voxels whose centre lies in (cr[m].x, cr[m+1].x] are solid.
                int i0 = (int)std::ceil((cr[m].x - w0.x) / h);
                int i1 = (int)std::floor((cr[m + 1].x - w0.x) / h);
                i0 = std::max(i0, 0); i1 = std::min(i1, nx - 1);
                uint8_t* dst = &occ[((size_t)k * ny + j) * nx];
                for (int i = i0; i <= i1; ++i) dst[i] = 1;
            }
        }
    }
}

// Solid-voxelize `tris[triStart .. triStart+triCount)` into an occupancy VdbGrid.
// `res` is the target voxel count along the LONGEST axis (cubic voxels; the other two
// axes get however many voxels that implies). Returns an empty grid if the range is
// empty or degenerate.
inline VdbGrid voxelizeSolid(const Tri* tris, size_t triStart, size_t triCount, int res) {
    VdbGrid g;
    if (!tris || triCount == 0) return g;
    if (res < 4) res = 4;

    // --- Bounds of the range -------------------------------------------------
    Vec3 lo(1e300, 1e300, 1e300), hi(-1e300, -1e300, -1e300);
    for (size_t t = 0; t < triCount; ++t) {
        const Tri& tr = tris[triStart + t];
        for (const Vec3& v : {tr.v0, tr.v1, tr.v2}) {
            lo.x = std::min(lo.x, v.x); lo.y = std::min(lo.y, v.y); lo.z = std::min(lo.z, v.z);
            hi.x = std::max(hi.x, v.x); hi.y = std::max(hi.y, v.y); hi.z = std::max(hi.z, v.z);
        }
    }
    Vec3 ext = hi - lo;
    double maxExt = std::max(ext.x, std::max(ext.y, ext.z));
    if (!(maxExt > 0.0)) return g;

    const double h = maxExt / res;                 // cubic voxel edge (world units)
    // One-voxel zero shell on every side: the first voxel CENTRE sits at lo - h, so the
    // trilinear ramp to zero happens inside the lattice instead of being clamped at it.
    const Vec3 w0 = lo - Vec3(h, h, h);
    const int nx = (int)std::ceil(ext.x / h) + 3;
    const int ny = (int)std::ceil(ext.y / h) + 3;
    const int nz = (int)std::ceil(ext.z / h) + 3;
    if ((long long)nx * ny * nz > 400LL * 1000 * 1000) return g;   // refuse absurd lattices

    std::vector<uint8_t> occ;
    voxelizeSolidInto(tris, triStart, triCount, w0, h, nx, ny, nz, occ);

    g.nx = nx; g.ny = ny; g.nz = nz;
    g.data.assign((size_t)nx * ny * nz, 0);
    const uint16_t ONE = floatToHalfBits(1.0f);
    for (size_t t = 0; t < occ.size(); ++t) if (occ[t]) g.data[t] = ONE;

    // --- Transform + metadata ------------------------------------------------
    const double s = 1.0 / h;
    g.ainv[0] = s; g.ainv[1] = 0; g.ainv[2] = 0;
    g.ainv[3] = 0; g.ainv[4] = s; g.ainv[5] = 0;
    g.ainv[6] = 0; g.ainv[7] = 0; g.ainv[8] = s;
    g.w0 = w0;
    g.imin = Vec3(0, 0, 0);
    g.wmin = w0;
    g.wmax = w0 + Vec3((nx - 1) * h, (ny - 1) * h, (nz - 1) * h);
    g.maxVal = 1.0f;
    return g;
}

// --- Feathering ---------------------------------------------------------------------
// One pass of the exact separable EDT (Felzenszwalb & Huttenlocher): given per-column
// squared distances `f[0..n)`, write back the lower envelope of the parabolas
// (x - q)^2 + f[q]. O(n) via the standard stack of intersection points. `f`, `d`, `v` and
// `z` are caller-owned scratch so the 3-axis sweep allocates once, not per column.
inline void edt1d(const std::vector<double>& f, std::vector<double>& d, int n,
                  std::vector<int>& v, std::vector<double>& z) {
    const double BIG = 1e30;                     // envelope bound, NOT the seed (see below)
    // Intersection abscissa of the parabolas rooted at p and q.
    auto sect = [&](int q, int p) {
        return ((f[q] + (double)q * q) - (f[p] + (double)p * p)) / (2.0 * (double)(q - p));
    };
    int k = 0;
    v[0] = 0; z[0] = -BIG; z[1] = BIG;
    for (int q = 1; q < n; ++q) {
        double s = sect(q, v[k]);
        while (k > 0 && s <= z[k]) { --k; s = sect(q, v[k]); }
        ++k; v[k] = q; z[k] = s; z[k + 1] = BIG;
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < (double)q) ++k;
        const double dq = (double)q - (double)v[k];
        d[q] = dq * dq + f[v[k]];
    }
}

// Turn a 0/1 occupancy lattice into a feathered one: each solid voxel's value becomes
// `saturate(distance-to-the-nearest-empty-voxel / featherVox)`, so density ramps from 0 at
// the silhouette to 1 once `featherVox` voxels inside. `featherVox` is in VOXELS (the caller
// converts the author's metres by dividing by the voxel edge). A value <= 0 is a no-op.
//
// Distances are measured to the nearest EMPTY voxel, and the bake already surrounds the
// lattice with a one-voxel zero shell, so a solid region touching the lattice edge still
// feathers rather than being cut flat.
inline void featherGrid(VdbGrid& g, double featherVox) {
    if (g.data.empty() || !(featherVox > 0.0)) return;
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const size_t n = (size_t)nx * ny * nz;
    auto idx = [nx, ny](int i, int j, int k) { return ((size_t)k * ny + (size_t)j) * nx + i; };

    // Seed: 0 at every EMPTY voxel, "unreachable" at every solid one — the EDT then gives
    // each solid voxel its squared distance to the outside. The seed is a large FINITE
    // value, not 1e300: F&H's parabola intersection subtracts two seeds, and with 1e300 on
    // both sides that difference is pure cancellation noise. One more than the longest
    // squared diagonal the lattice can hold is unreachable in practice and keeps every
    // intermediate well-conditioned.
    const double UNREACH = (double)nx * nx + (double)ny * ny + (double)nz * nz + 1.0;
    std::vector<double> dist(n);
    for (size_t t = 0; t < n; ++t) dist[t] = g.data[t] ? UNREACH : 0.0;

    const int nmax = std::max(nx, std::max(ny, nz));
    std::vector<double> f((size_t)nmax), d((size_t)nmax), z((size_t)nmax + 1);
    std::vector<int> v((size_t)nmax);

    for (int k = 0; k < nz; ++k)                                   // along x
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) f[i] = dist[idx(i, j, k)];
            edt1d(f, d, nx, v, z);
            for (int i = 0; i < nx; ++i) dist[idx(i, j, k)] = d[i];
        }
    for (int k = 0; k < nz; ++k)                                   // along y
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) f[j] = dist[idx(i, j, k)];
            edt1d(f, d, ny, v, z);
            for (int j = 0; j < ny; ++j) dist[idx(i, j, k)] = d[j];
        }
    for (int j = 0; j < ny; ++j)                                   // along z
        for (int i = 0; i < nx; ++i) {
            for (int k = 0; k < nz; ++k) f[k] = dist[idx(i, j, k)];
            edt1d(f, d, nz, v, z);
            for (int k = 0; k < nz; ++k) dist[idx(i, j, k)] = d[k];
        }

    // Smoothstep rather than a linear ramp: a linear ramp leaves a visible crease where it
    // reaches 1, which on a cloud edge reads as a second, softer silhouette.
    const double inv = 1.0 / featherVox;
    for (size_t t = 0; t < n; ++t) {
        if (!g.data[t]) continue;
        double u = std::sqrt(dist[t]) * inv;
        if (u >= 1.0) continue;                       // already full density; leave the 1
        if (u < 0.0) u = 0.0;
        g.data[t] = floatToHalfBits((float)(u * u * (3.0 - 2.0 * u)));
    }
    g.maxVal = 1.0f;
}

// Fraction of the lattice that came out solid — a cheap sanity signal for the loader's
// report (a mesh that voxelizes to ~0% is inside-out, open, or far below the resolution
// it needs, and the fog would silently vanish).
inline double solidFraction(const VdbGrid& g) {
    if (g.data.empty()) return 0.0;
    size_t n = 0;
    for (uint16_t v : g.data) if (v) ++n;
    return (double)n / (double)g.data.size();
}

// --- Signed distance field (O3 stage 3) ----------------------------------------------
//
// Squared distance from `p` to the triangle (a,b,c) — the exact closest-point routine
// (Ericson, *Real-Time Collision Detection* §5.1.5): test the three vertex Voronoi
// regions, then the three edge regions, then fall through to the face interior. Written
// out rather than reduced to "project and clamp the barycentrics" because that shortcut
// is wrong on obtuse triangles, which is exactly what an imported mesh is full of.
inline double pointTriDistSq(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 ab = b - a, ac = c - a, ap = p - a;
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return dot(ap, ap);                       // vertex a
    const Vec3 bp = p - b;
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return dot(bp, bp);                        // vertex b
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {                            // edge ab
        const double v = (d1 - d3 != 0.0) ? d1 / (d1 - d3) : 0.0;
        const Vec3 q = a + ab * v; const Vec3 e = p - q; return dot(e, e);
    }
    const Vec3 cp = p - c;
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return dot(cp, cp);                        // vertex c
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {                            // edge ac
        const double w = (d2 - d6 != 0.0) ? d2 / (d2 - d6) : 0.0;
        const Vec3 q = a + ac * w; const Vec3 e = p - q; return dot(e, e);
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {              // edge bc
        const double den = (d4 - d3) + (d5 - d6);
        const double w = (den != 0.0) ? (d4 - d3) / den : 0.0;
        const Vec3 q = b + (c - b) * w; const Vec3 e = p - q; return dot(e, e);
    }
    const double den = va + vb + vc;                                      // face interior
    if (!(den > 0.0)) return dot(ap, ap);                                 // degenerate triangle
    const double inv = 1.0 / den;
    const Vec3 q = a + ab * (vb * inv) + ac * (vc * inv);
    const Vec3 e = p - q;
    return dot(e, e);
}

// A baked signed distance field over a regular lattice. Sample (i,j,k) sits at
// `lo + (i,j,k)*h` and holds the signed distance in WORLD units, negative inside.
struct SdfBake {
    int nx = 0, ny = 0, nz = 0;
    Vec3 lo;                      // world position of sample (0,0,0)
    double h = 0;                 // cubic voxel edge, world units
    // nx*ny*nz samples, flat index ((i*ny) + j)*nz + k — C order with **x OUTERMOST
    // and z fastest**. That is deliberately NOT the x-fastest order the voxelizer and
    // the sweeps work in internally: it is `PatGrid`'s convention ("axis 0 outermost",
    // pattern.h), so the bake can be appended to `Scene::dataPool` verbatim and read by
    // `grid:<name>(x, y, z)` with no transpose at the call site. The transpose happens
    // once, in stage 4. Getting this backwards is invisible in any aggregate statistic
    // (min, max, histogram) and shows up only as a field that looks plausible but is
    // rotated — so `-checksdf` §7 samples the finished PatGrid, not this array.
    std::vector<float> d;
    bool empty() const { return d.empty(); }
    // Index of sample (i,j,k) within `d`.
    size_t at(int i, int j, int k) const { return ((size_t)i * ny + (size_t)j) * nz + (size_t)k; }
    Vec3 hiCorner() const { return lo + Vec3((nx - 1) * h, (ny - 1) * h, (nz - 1) * h); }
};

// Bake the signed distance to `tris[triStart .. triStart+triCount)` over that range's
// AABB expanded by `pad` world units on every side, at `res` samples along the padded
// box's LONGEST axis.
//
// Three stages:
//
//  1. SIGN comes from `voxelizeSolidInto` — signed-crossing (generalized winding)
//     scanlines, so a model made of several closed bodies reads as their union instead of
//     hollowing where they overlap. Sign is the only part of an SDF that a mesh can be
//     genuinely ambiguous about, and this is the one place in the codebase that has
//     already been argued out and measured (see the fog bound above).
//  2. NARROW BAND. Every triangle visits the lattice samples within `kBand` voxels of its
//     own AABB and records the EXACT point-triangle distance, plus WHICH triangle was
//     closest. Parallelised over z-slabs after bucketing each triangle into the slabs it
//     touches, so no two threads ever write the same sample and no atomics are needed.
//  3. PROPAGATION by Bridson's closest-point sweep (the `SDFGen` algorithm): eight sweeps
//     over the lattice, one per octant direction, each sample adopting a neighbour's
//     closest TRIANGLE if re-measuring against that triangle beats what it has. Distance
//     stays exact-per-triangle everywhere it reaches; only the *search* is approximated,
//     and two rounds of eight sweeps close it for any geometry that is not pathological.
//
// The tempting shortcut here is wrong and was implemented and measured before it was
// caught, so it is worth naming: the exact separable EDT the feather pass uses computes
// min over q of (|p-q|^2 + seed[q]), which is a distance transform ONLY for BINARY seeds.
// Feed it the exact squared distances from stage 2 and it adds SQUARES where the true
// distance adds LENGTHS — the two legs p->q and q->surface are collinear in the case that
// matters, so the answer should be (|p-q| + d_q)^2, not |p-q|^2 + d_q^2. On a torus of
// tube radius 71.5 mm that read the deepest interior as -47 mm instead of -71.5 mm: a
// smooth, plausible, systematically shallow field. Propagating the closest PRIMITIVE
// rather than a distance is what makes the composition legal.
inline SdfBake bakeSignedDistance(const Tri* tris, size_t triStart, size_t triCount,
                                  int res, double pad) {
    SdfBake out;
    if (!tris || triCount == 0) return out;
    if (res < 4) res = 4;
    if (!(pad >= 0.0)) pad = 0.0;

    Vec3 lo(1e300, 1e300, 1e300), hi(-1e300, -1e300, -1e300);
    for (size_t t = 0; t < triCount; ++t) {
        const Tri& tr = tris[triStart + t];
        for (const Vec3& v : {tr.v0, tr.v1, tr.v2}) {
            lo.x = std::min(lo.x, v.x); lo.y = std::min(lo.y, v.y); lo.z = std::min(lo.z, v.z);
            hi.x = std::max(hi.x, v.x); hi.y = std::max(hi.y, v.y); hi.z = std::max(hi.z, v.z);
        }
    }
    if (!(hi.x >= lo.x)) return out;
    lo = lo - Vec3(pad, pad, pad);
    hi = hi + Vec3(pad, pad, pad);
    const Vec3 ext = hi - lo;
    const double maxExt = std::max(ext.x, std::max(ext.y, ext.z));
    if (!(maxExt > 0.0)) return out;

    const double h = maxExt / (res - 1);           // res SAMPLES span the longest axis
    const int nx = (int)std::floor(ext.x / h) + 1;
    const int ny = (int)std::floor(ext.y / h) + 1;
    const int nz = (int)std::floor(ext.z / h) + 1;
    if ((long long)nx * ny * nz > 120LL * 1000 * 1000) return out;   // refuse absurd lattices
    const size_t n = (size_t)nx * ny * nz;

    // --- 1. Sign ------------------------------------------------------------
    std::vector<uint8_t> occ;
    voxelizeSolidInto(tris, triStart, triCount, lo, h, nx, ny, nz, occ);

    // --- 2. Exact narrow band -----------------------------------------------
    // `dist` is the running best SQUARED distance in world units; `nearTri` is the triangle
    // that produced it (-1 = none yet), which is what stage 3 propagates. (Not `near`:
    // that is still a legacy 16-bit-model macro in the Windows headers.)
    const double kUnset = 1e300;
    std::vector<double> dist(n, kUnset);
    std::vector<int>    nearTri(n, -1);
    const int kBand = 2;                            // voxels of exact seeding around a face
    const double invH = 1.0 / h;

    // Bucket triangles by the z-slabs they touch, so the band loop can be parallelised
    // over slabs with every thread owning its samples exclusively.
    std::vector<std::vector<int>> slab((size_t)nz);
    auto triSlabRange = [&](const Tri& tr, int& k0, int& k1) {
        const double zl = std::min(tr.v0.z, std::min(tr.v1.z, tr.v2.z));
        const double zh = std::max(tr.v0.z, std::max(tr.v1.z, tr.v2.z));
        k0 = (int)std::floor((zl - lo.z) * invH) - kBand;
        k1 = (int)std::ceil ((zh - lo.z) * invH) + kBand;
        k0 = std::max(k0, 0); k1 = std::min(k1, nz - 1);
    };
    for (size_t t = 0; t < triCount; ++t) {
        int k0, k1; triSlabRange(tris[triStart + t], k0, k1);
        for (int k = k0; k <= k1; ++k) slab[(size_t)k].push_back((int)t);
    }

    (void)ft::parallelFor((size_t)nz, 1, [&](size_t ks) {
        const int k = (int)ks;
        const double pz = lo.z + k * h;
        for (int ti : slab[(size_t)k]) {
            const Tri& tr = tris[triStart + (size_t)ti];
            const double xl = std::min(tr.v0.x, std::min(tr.v1.x, tr.v2.x));
            const double xh = std::max(tr.v0.x, std::max(tr.v1.x, tr.v2.x));
            const double yl = std::min(tr.v0.y, std::min(tr.v1.y, tr.v2.y));
            const double yh = std::max(tr.v0.y, std::max(tr.v1.y, tr.v2.y));
            int i0 = std::max((int)std::floor((xl - lo.x) * invH) - kBand, 0);
            int i1 = std::min((int)std::ceil ((xh - lo.x) * invH) + kBand, nx - 1);
            int j0 = std::max((int)std::floor((yl - lo.y) * invH) - kBand, 0);
            int j1 = std::min((int)std::ceil ((yh - lo.y) * invH) + kBand, ny - 1);
            for (int j = j0; j <= j1; ++j) {
                const double py = lo.y + j * h;
                const size_t row = ((size_t)k * ny + (size_t)j) * nx;
                for (int i = i0; i <= i1; ++i) {
                    const double q = pointTriDistSq(Vec3(lo.x + i * h, py, pz),
                                                    tr.v0, tr.v1, tr.v2);
                    if (q < dist[row + i]) { dist[row + i] = q; nearTri[row + i] = ti; }
                }
            }
        }
    });

    // --- 3. Closest-triangle sweeps -----------------------------------------
    auto at = [nx, ny](int i, int j, int k) { return ((size_t)k * ny + (size_t)j) * nx + i; };
    // Try adopting the closest triangle of the already-swept neighbour `(ni,nj,nk)`.
    auto relax = [&](int i, int j, int k, int ni, int nj, int nk) {
        const size_t s = at(ni, nj, nk);
        const int ti = nearTri[s];
        if (ti < 0) return;
        const size_t d0 = at(i, j, k);
        if (nearTri[d0] == ti) return;                 // already measured against this one
        const Tri& tr = tris[triStart + (size_t)ti];
        const double q = pointTriDistSq(Vec3(lo.x + i * h, lo.y + j * h, lo.z + k * h),
                                        tr.v0, tr.v1, tr.v2);
        if (q < dist[d0]) { dist[d0] = q; nearTri[d0] = ti; }
    };
    // One directional sweep: walk each axis in the given sign and relax against the three
    // axis neighbours already visited in this order. Eight sign combinations cover every
    // octant; Bridson runs the set twice, which is what closes the cases where the closest
    // triangle is only reachable around a concavity.
    auto sweep = [&](int si, int sj, int sk) {
        const int i0 = si > 0 ? 1 : nx - 2, i1 = si > 0 ? nx : -1;
        const int j0 = sj > 0 ? 1 : ny - 2, j1 = sj > 0 ? ny : -1;
        const int k0 = sk > 0 ? 1 : nz - 2, k1 = sk > 0 ? nz : -1;
        for (int k = k0; k != k1; k += sk)
            for (int j = j0; j != j1; j += sj)
                for (int i = i0; i != i1; i += si) {
                    relax(i, j, k, i - si, j, k);
                    relax(i, j, k, i, j - sj, k);
                    relax(i, j, k, i, j, k - sk);
                }
    };
    for (int round = 0; round < 2; ++round)
        for (int s = 0; s < 8; ++s)
            sweep((s & 1) ? +1 : -1, (s & 2) ? +1 : -1, (s & 4) ? +1 : -1);

    // --- 4. Sign, and the transpose into PatGrid order -----------------------
    out.nx = nx; out.ny = ny; out.nz = nz; out.lo = lo; out.h = h;
    out.d.resize(n);
    // A sample the sweeps never reached can only happen on an empty triangle range, but if
    // it does it must not read 0 — for a distance field that means "exactly on the surface",
    // the most consequential value there is. The box diagonal is the honest lower bound.
    const double unreached = length(Vec3((nx - 1) * h, (ny - 1) * h, (nz - 1) * h));
    // Written destination-sequential (k innermost): `out.d` is the array that gets copied
    // into the shared sample pool, so it is the side worth keeping contiguous.
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            float* dst = &out.d[((size_t)i * ny + (size_t)j) * nz];
            for (int k = 0; k < nz; ++k) {
                const size_t s = at(i, j, k);          // working (x-fastest) layout
                const double m = (dist[s] >= kUnset) ? unreached : std::sqrt(dist[s]);
                dst[k] = (float)(occ[s] ? -m : m);
            }
        }
    return out;
}

}  // namespace meshvox
