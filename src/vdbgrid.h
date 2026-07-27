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
#include <cstdint>
#include <cstring>

// IEEE-754 binary16 (half) <-> binary32 (float) conversion. The dense VDB
// lattice is stored as fp16 to halve host RAM and GPU VRAM for large volumes
// (density fields tolerate the ~0.05% relative error of half precision easily).
// Round-to-nearest-even on store; full subnormal/inf handling both ways. The GPU
// device sampler (render_cuda.cu) mirrors these with matching __device__ inlines.
inline float halfBitsToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                          // +/- 0
        } else {                                  // subnormal half -> normal float
            exp = 1;
            while ((man & 0x400u) == 0) { man <<= 1; --exp; }
            man &= 0x3FFu;
            bits = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13);  // inf / nan
    } else {
        bits = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (man << 13);
    }
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

inline uint16_t floatToHalfBits(float f) {
    uint32_t bits; std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t man  = bits & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;     // too small -> signed zero
        man |= 0x800000u;                         // restore implicit 1
        int shift = 14 - exp;
        uint32_t half = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) half++;
        return (uint16_t)(sign | half);
    } else if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);        // overflow -> inf
    }
    uint16_t half = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    uint32_t rem = man & 0x1FFFu;                 // round-to-nearest-even
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;  // carry into exp is correct
    return half;
}

// A dense scalar volume baked from a .nvdb FloatGrid. Values are stored on the
// grid's own integer voxel lattice covering its active index bounding box; a
// world point is mapped to fractional lattice coordinates by the stored affine
// (`ainv` * (p - w0) - imin) and trilinearly interpolated. Density outside the
// baked box reads 0 (the medium simply does not exist there).
struct VdbGrid {
    int nx = 0, ny = 0, nz = 0;         // dense lattice dimensions
    std::vector<uint16_t> data;         // nx*ny*nz fp16 values, index [(k*ny + j)*nx + i]
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
        // Clamp the sample COORDINATE to the lattice, then interpolate. Clamping
        // only the stencil indices (as this used to) leaves the fraction near 1
        // in the outer half-voxel shell below index 0, so the sample would be
        // dominated by the SECOND voxel and get wronger the further out it went;
        // clamping the coordinate makes that shell read the edge voxel, which is
        // what the clamp was always meant to do. Inside [0, n-1] this is
        // bit-identical to the old code (and drops three floor() calls).
        auto clampd = [](double v, double hi) { return v < 0.0 ? 0.0 : (v > hi ? hi : v); };
        double ci = clampd(fi, nx - 1), cj = clampd(fj, ny - 1), ck = clampd(fk, nz - 1);
        int i0 = (int)ci, j0 = (int)cj, k0 = (int)ck;   // ci >= 0, so trunc == floor
        int i1 = i0 + 1 < nx ? i0 + 1 : nx - 1;
        int j1 = j0 + 1 < ny ? j0 + 1 : ny - 1;
        int k1 = k0 + 1 < nz ? k0 + 1 : nz - 1;
        double tx = ci - i0, ty = cj - j0, tz = ck - k0;
        auto at = [&](int i, int j, int k) -> double {
            return (double)halfBitsToFloat(data[(size_t(k) * ny + j) * nx + i]);
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

    // Partition the dense lattice into B^3 bricks for the native sparse GPU
    // sampler (ROADMAP C2). Only bricks holding at least one nonzero voxel are
    // emitted: `brickIndex` (bx*by*bz int32s) maps a brick coordinate to its slot
    // in `brickData` (contiguous B^3 fp16 voxels per active brick), or -1 for an
    // all-empty brick that the device sampler reads as density 0. B must be a
    // power of two. The result samples BIT-FOR-BIT identically to dense `data`:
    // the trilinear stencil is always clamped to [0,n-1] before lookup, so the
    // per-brick padding voxels (beyond nx/ny/nz when a dim isn't a multiple of B)
    // are never addressed. Returns the number of active bricks.
    int buildBricks(int B, int& bx, int& by, int& bz,
                    std::vector<int32_t>& brickIndex,
                    std::vector<uint16_t>& brickData) const {
        bx = (nx + B - 1) / B; by = (ny + B - 1) / B; bz = (nz + B - 1) / B;
        brickIndex.assign((size_t)bx * by * bz, -1);
        const size_t B3 = (size_t)B * B * B;
        int active = 0;
        // Pass 1: flag bricks with any nonzero voxel, assign compact slots.
        for (int bk = 0; bk < bz; ++bk)
        for (int bj = 0; bj < by; ++bj)
        for (int bi = 0; bi < bx; ++bi) {
            bool any = false;
            for (int lk = 0; lk < B && !any; ++lk) { int k = bk*B+lk; if (k>=nz) break;
              for (int lj = 0; lj < B && !any; ++lj) { int j = bj*B+lj; if (j>=ny) break;
                for (int li = 0; li < B; ++li) { int i = bi*B+li; if (i>=nx) break;
                  if (data[((size_t)k*ny + j)*nx + i] != 0) { any = true; break; } } } }
            if (any) brickIndex[((size_t)bk*by + bj)*bx + bi] = active++;
        }
        // Pass 2: copy voxels into their brick slot (padding voxels stay 0).
        brickData.assign((size_t)active * B3, 0);
        for (int bk = 0; bk < bz; ++bk)
        for (int bj = 0; bj < by; ++bj)
        for (int bi = 0; bi < bx; ++bi) {
            int slot = brickIndex[((size_t)bk*by + bj)*bx + bi];
            if (slot < 0) continue;
            uint16_t* dst = &brickData[(size_t)slot * B3];
            for (int lk = 0; lk < B; ++lk) { int k = bk*B+lk; if (k>=nz) continue;
              for (int lj = 0; lj < B; ++lj) { int j = bj*B+lj; if (j>=ny) continue;
                for (int li = 0; li < B; ++li) { int i = bi*B+li; if (i>=nx) continue;
                  dst[((size_t)lk*B + lj)*B + li] = data[((size_t)k*ny + j)*nx + i]; } } }
        }
        return active;
    }
};

// Load a sparse volume file, bake a float grid into `out`. Dispatches on the
// file's magic: a native OpenVDB `.vdb` (magic "VDB ") is parsed by
// loadOpenVDBGrid (vdb_openvdb.cpp, no NanoVDB), otherwise it is treated as an
// (uncompressed) NanoVDB `.nvdb` and baked in vdbgrid.cpp (the only TU that
// includes NanoVDB.h). When `wantName` is empty the FIRST float grid is baked
// (density); otherwise the float grid whose name matches `wantName`
// (case-insensitive) is selected — used to pull the "temperature" grid out of a
// multi-grid fire `.vdb` alongside its "density" grid. Returns false and fills
// `err` on failure (including a clear message when a requested grid is absent).
bool loadVdbGrid(const std::string& path, VdbGrid& out, std::string& err,
                 const std::string& wantName = "");

// Native OpenVDB `.vdb` reader (vdb_openvdb.cpp): parses the file container, tree
// topology and BLOSC/ACTIVE_MASK/HalfFloat leaf buffers by hand (vendored LZ4 for
// blosc's LZ4 codec) and bakes a float grid into a dense VdbGrid. No
// OpenVDB/NanoVDB dependency. `wantName` selects a named grid (empty => first
// float grid). Returns false and fills `err` on failure (including a clear
// "re-export with LZ4" message for the unsupported blosc codecs).
bool loadOpenVDBGrid(const std::string& path, VdbGrid& out, std::string& err,
                     const std::string& wantName = "");
