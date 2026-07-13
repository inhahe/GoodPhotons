// vdbgrid.h — lightweight, NanoVDB-free interface to a loaded .nvdb volume.
//
// ROADMAP item 4: import OpenVDB/NanoVDB sparse volumes as heterogeneous fog.
// The heavy NanoVDB.h template header is confined to ONE translation unit
// (vdbgrid.cpp); everywhere else (scene.h, ftsl.h, the CUDA host uploader) sees
// only this small plain-old-data grid. On load we BAKE the sparse FloatGrid into
// a DENSE float lattice plus a world->index affine, so the exact same trilinear
// sampler runs on the CPU (VdbGrid::sample) and on the GPU (a device twin over
// the uploaded arrays). This keeps the renderer's NanoVDB footprint to a single
// vendored header and makes the sampled field trivially portable.
#pragma once
#include "linalg.h"
#include <vector>
#include <string>

// A dense scalar volume baked from a .nvdb FloatGrid. Values are stored on the
// grid's own integer voxel lattice covering its active index bounding box; a
// world point is mapped to fractional lattice coordinates by the stored affine
// (`ainv` * (p - w0) - imin) and trilinearly interpolated. Density outside the
// baked box reads 0 (the medium simply does not exist there).
struct VdbGrid {
    int nx = 0, ny = 0, nz = 0;         // dense lattice dimensions
    std::vector<float> data;            // nx*ny*nz values, index [(k*ny + j)*nx + i]
    double ainv[9] = {1,0,0, 0,1,0, 0,0,1}; // world->index linear map (row-major 3x3)
    Vec3   w0{0,0,0};                   // world position of index origin (0,0,0)
    Vec3   imin{0,0,0};                 // integer min-corner of the baked lattice
    Vec3   wmin{0,0,0}, wmax{0,0,0};    // world-space AABB of the active voxels
    float  maxVal = 0.0f;               // sup of the field (delta/ratio-tracking majorant)

    bool empty() const { return data.empty(); }

    // Fractional lattice coordinate of a world point (before the imin shift).
    void toLattice(const Vec3& p, double& fi, double& fj, double& fk) const {
        double rx = p.x - w0.x, ry = p.y - w0.y, rz = p.z - w0.z;
        fi = ainv[0]*rx + ainv[1]*ry + ainv[2]*rz - imin.x;
        fj = ainv[3]*rx + ainv[4]*ry + ainv[5]*rz - imin.y;
        fk = ainv[6]*rx + ainv[7]*ry + ainv[8]*rz - imin.z;
    }

    // Trilinearly sample the dense field at a world point (>= 0). Points outside
    // the baked lattice return 0.
    double sample(const Vec3& p) const {
        if (data.empty()) return 0.0;
        double fi, fj, fk;
        toLattice(p, fi, fj, fk);
        // Reject clearly-outside points (half a voxel margin); the medium bound
        // already clips rays to the AABB, so this only guards interpolation edges.
        if (fi < -0.5 || fj < -0.5 || fk < -0.5 ||
            fi > nx - 0.5 || fj > ny - 0.5 || fk > nz - 0.5) return 0.0;
        // Clamp the interpolation stencil to the valid lattice.
        auto clampi = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
        int i0 = clampi((int)std::floor(fi), nx - 1), i1 = clampi(i0 + 1, nx - 1);
        int j0 = clampi((int)std::floor(fj), ny - 1), j1 = clampi(j0 + 1, ny - 1);
        int k0 = clampi((int)std::floor(fk), nz - 1), k1 = clampi(k0 + 1, nz - 1);
        double tx = fi - std::floor(fi), ty = fj - std::floor(fj), tz = fk - std::floor(fk);
        if (tx < 0) tx = 0; else if (tx > 1) tx = 1;
        if (ty < 0) ty = 0; else if (ty > 1) ty = 1;
        if (tz < 0) tz = 0; else if (tz > 1) tz = 1;
        auto at = [&](int i, int j, int k) -> double {
            return (double)data[(size_t(k) * ny + j) * nx + i];
        };
        double c00 = at(i0,j0,k0)*(1-tx) + at(i1,j0,k0)*tx;
        double c10 = at(i0,j1,k0)*(1-tx) + at(i1,j1,k0)*tx;
        double c01 = at(i0,j0,k1)*(1-tx) + at(i1,j0,k1)*tx;
        double c11 = at(i0,j1,k1)*(1-tx) + at(i1,j1,k1)*tx;
        double c0 = c00*(1-ty) + c10*ty;
        double c1 = c01*(1-ty) + c11*ty;
        double v = c0*(1-tz) + c1*tz;
        return v > 0.0 ? v : 0.0;
    }
};

// Load a .nvdb (uncompressed NanoVDB) file, bake its first FloatGrid into `out`.
// Returns false and fills `err` on failure. Implemented in vdbgrid.cpp (the only
// TU that includes NanoVDB.h).
bool loadVdbGrid(const std::string& path, VdbGrid& out, std::string& err);
