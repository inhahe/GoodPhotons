// View-independent photon map (ROADMAP item 1).
//
// A stored, camera-independent spatial structure of photon records that a backward
// camera pass can query by radius (density estimation). Light transport is computed
// ONCE (the forward photon pass deposits a record at each diffuse/volume vertex) and
// reused — both across the pixels of one frame and, crucially, across many camera
// frames of a static scene (the flythrough case: scene fixed, only the camera moves).
//
// Acceleration structure: a UNIFORM HASH GRID (cell size = gather radius), not a
// kd-tree. The flat-array grid is the CUDA-friendly structure the shared photon /
// light-vertex passes reuse (build = bin into cells, counting-sort by cell id,
// prefix-sum offsets). See ROADMAP.md §(1).
//
// Spectral note: photons are monochromatic here (one wavelength sampled per photon
// from the emitter SPD, carried as a scalar power `beta`). The camera-side estimate
// accumulates each photon's contribution weighted by the CIE response at the PHOTON's
// wavelength — i.e. the density estimate is built directly in XYZ, exactly like the
// forward light-tracer's per-splat `cie(lambda)*contrib`. This sidesteps monochromatic
// spectral-irradiance reconstruction (see the ROADMAP open question) and is exact for
// a directly-viewed diffuse surface.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <utility>
#include "linalg.h"

// One deposited photon: where light landed, where it came from, and how much power it
// carried at what wavelength. `wi` (incident direction, pointing back toward the source
// of the photon = -ray.d) and `n` (shading normal) are stored for kernel weighting and
// cross-surface leak rejection; a pure Lambertian estimate only needs pos/power/lambda.
struct Photon {
    Vec3  pos;        // deposit position (world space)
    Vec3  wi;         // incident direction at deposit (unit, = -photon travel dir)
    Vec3  n;          // shading normal at the deposit surface (unit)
    float power;      // monochromatic power (beta) carried by this photon
    float lambda;     // wavelength (nm)
};

// Uniform hash grid over deposited photons. After build(), `photons` is reordered into
// cell-contiguous runs and `cellStart` gives each cell's [begin,end) slice — a flat,
// pointer-free layout (counting sort) that ports directly to the GPU.
struct PhotonMap {
    std::vector<Photon> photons;   // reordered to cell order by build()
    long long nEmitted = 0;        // total photons EMITTED in the pass (normalization)
    double    radius   = 0.02;     // gather radius (world units); == grid cell size

    // grid geometry
    Vec3   lo{0, 0, 0};
    double cellSize = 0.02;
    int    nx = 1, ny = 1, nz = 1;
    std::vector<int> cellStart;    // size nCells+1; cell c occupies [cellStart[c], cellStart[c+1])

    long long cellCount() const { return (long long)nx * ny * nz; }

    int cellIndex(int ix, int iy, int iz) const {
        return (iz * ny + iy) * nx + ix;
    }
    // Clamp a world point to a valid grid cell coordinate.
    void cellCoord(const Vec3& p, int& ix, int& iy, int& iz) const {
        ix = (int)std::floor((p.x - lo.x) / cellSize);
        iy = (int)std::floor((p.y - lo.y) / cellSize);
        iz = (int)std::floor((p.z - lo.z) / cellSize);
        ix = std::min(std::max(ix, 0), nx - 1);
        iy = std::min(std::max(iy, 0), ny - 1);
        iz = std::min(std::max(iz, 0), nz - 1);
    }

    // Bin the deposited photons into a uniform grid of cell size `r` (== gather radius,
    // so a radius-r query touches only the 3x3x3 neighbourhood) via a counting sort.
    void build(double r) {
        radius = r;
        cellSize = (r > 0.0) ? r : 1e-6;
        if (photons.empty()) { nx = ny = nz = 1; cellStart.assign(2, 0); return; }

        Vec3 mn = photons[0].pos, mx = photons[0].pos;
        for (const Photon& ph : photons) {
            mn.x = std::min(mn.x, ph.pos.x); mn.y = std::min(mn.y, ph.pos.y); mn.z = std::min(mn.z, ph.pos.z);
            mx.x = std::max(mx.x, ph.pos.x); mx.y = std::max(mx.y, ph.pos.y); mx.z = std::max(mx.z, ph.pos.z);
        }
        // Pad by half a cell so floor() never underflows at the low edge.
        lo = mn - Vec3{cellSize, cellSize, cellSize} * 0.5;
        Vec3 ext = (mx - lo) + Vec3{cellSize, cellSize, cellSize} * 0.5;
        nx = std::max(1, (int)std::ceil(ext.x / cellSize));
        ny = std::max(1, (int)std::ceil(ext.y / cellSize));
        nz = std::max(1, (int)std::ceil(ext.z / cellSize));
        const long long nCells = cellCount();

        // Pass 1: count photons per cell.
        std::vector<int> cellOf(photons.size());
        cellStart.assign((size_t)nCells + 1, 0);
        for (size_t i = 0; i < photons.size(); ++i) {
            int ix, iy, iz; cellCoord(photons[i].pos, ix, iy, iz);
            int c = cellIndex(ix, iy, iz);
            cellOf[i] = c;
            ++cellStart[c + 1];
        }
        // Prefix sum -> cellStart[c] = begin offset of cell c.
        for (long long c = 0; c < nCells; ++c) cellStart[c + 1] += cellStart[c];

        // Pass 2: scatter into cell-contiguous order.
        std::vector<Photon> sorted(photons.size());
        std::vector<int> cursor(cellStart.begin(), cellStart.end() - 1);
        for (size_t i = 0; i < photons.size(); ++i)
            sorted[cursor[cellOf[i]]++] = photons[i];
        photons.swap(sorted);
    }

    // Invoke fn(const Photon&, double dist2) for every photon within `radius` of p.
    template <class F>
    void query(const Vec3& p, F&& fn) const { queryR(p, radius, std::forward<F>(fn)); }

    // Same, but with an explicit query radius `r`. The 3x3x3 cell neighbourhood only
    // covers radii up to the grid cell size, so the CALLER MUST ensure r <= cellSize
    // (true for PPM/SPPM, where the grid is built at the largest current per-pixel
    // radius and every pixel's radius only shrinks from there).
    template <class F>
    void queryR(const Vec3& p, double r, F&& fn) const {
        if (photons.empty()) return;
        int ix, iy, iz; cellCoord(p, ix, iy, iz);
        const double r2 = r * r;
        for (int dz = -1; dz <= 1; ++dz) {
            int cz = iz + dz; if (cz < 0 || cz >= nz) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int cy = iy + dy; if (cy < 0 || cy >= ny) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int cx = ix + dx; if (cx < 0 || cx >= nx) continue;
                    int c = cellIndex(cx, cy, cz);
                    for (int k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                        const Photon& ph = photons[k];
                        Vec3 d = p - ph.pos;
                        double d2 = dot(d, d);
                        if (d2 <= r2) fn(ph, d2);
                    }
                }
            }
        }
    }
};
