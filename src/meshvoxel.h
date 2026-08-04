// meshvoxel.h — SOLID voxelization of a triangle mesh into a dense VdbGrid.
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
#pragma once
#include "geometry.h"
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
    g.nx = nx; g.ny = ny; g.nz = nz;
    g.data.assign((size_t)nx * ny * nz, 0);
    const uint16_t ONE = floatToHalfBits(1.0f);
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
                uint16_t* dst = &g.data[((size_t)k * ny + j) * nx];
                for (int i = i0; i <= i1; ++i) dst[i] = ONE;
            }
        }
    }

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

// Fraction of the lattice that came out solid — a cheap sanity signal for the loader's
// report (a mesh that voxelizes to ~0% is inside-out, open, or far below the resolution
// it needs, and the fog would silently vanish).
inline double solidFraction(const VdbGrid& g) {
    if (g.data.empty()) return 0.0;
    size_t n = 0;
    for (uint16_t v : g.data) if (v) ++n;
    return (double)n / (double)g.data.size();
}

}  // namespace meshvox
