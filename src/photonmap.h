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
#include <thread>
#include "linalg.h"
#include "color.h"

// One deposited photon's PAYLOAD — everything the gather reads *after* a candidate has
// passed the distance test. The deposit POSITION deliberately lives in a separate
// `PhotonMap::pos` array (see below); this struct is what `pos[k]` indexes into.
//
// `n` (shading normal) is here for cross-surface leak rejection; a pure Lambertian
// estimate only needs power/lambda. There is no `wi`: the incident direction was stored
// for years and never read by any gather (the estimate is Lambertian, so it only needs
// the normal), which cost 24 dead bytes on every one of what can be tens of millions of
// records. Don't re-add a field here without a reader — see the size note below.
struct Photon {
    Vec3  n;          // shading normal at the deposit surface (unit)
    float power;      // monochromatic power (beta) carried by this photon
    float lambda;     // wavelength (nm)
};

// A per-thread deposit bank: the split (position, payload) pair that a photon pass appends
// to, mirroring PhotonMap's own split layout so the concatenation into the map is a plain
// append of both arrays. The two vectors are always the same length.
struct PhotonBank {
    std::vector<Vec3>   pos;
    std::vector<Photon> payload;
    size_t size() const { return payload.size(); }
    void push(const Vec3& p, const Vec3& n, float power, float lambda) {
        pos.push_back(p);
        payload.push_back(Photon{n, power, lambda});
    }
};

// Uniform hash grid over deposited photons. After build(), the three per-photon arrays
// are reordered together into cell-contiguous runs and `cellStart` gives each cell's
// [begin,end) slice — a flat, pointer-free layout (counting sort) that ports directly to
// the GPU. Index k addresses the same photon in `pos[k]` / `photons[k]` / `cie[k]`.
//
// LAYOUT IS STRUCTURE-OF-ARRAYS ON PURPOSE, and it is a load-bearing perf decision.
// A radius-r query scans the 3x3x3 cell neighbourhood, i.e. a (3r)^3 box, but only keeps
// what falls inside the radius-r sphere: (4/3 pi r^3)/(27 r^3) = 15% of it. So ~85% of
// every query's work is a distance test that touches the position and NOTHING else. With
// position embedded in a fat record, that scan strided over the whole record and pulled
// cache lines it used a quarter of; splitting positions out makes the reject scan a dense
// 24 B/photon stream. This matters most exactly where mode M hurts: the record arrays run
// to gigabytes on a dense map (tens of millions of photons), far past any cache, so the
// gather is DRAM-bandwidth-bound and the win is close to the bandwidth ratio. Keep `pos`
// separate, and keep `Photon` free of anything the gather doesn't read.
struct PhotonMap {
    std::vector<Vec3>   pos;       // deposit positions (world); the distance-test stream
    std::vector<Photon> photons;   // payloads, parallel to pos[]; reordered by build()
    // Per-photon CIE XYZ response at photons[i].lambda, filled by build(). The gather
    // estimate weights every photon by cie(lambda_p); evaluating the analytic CIE
    // multi-Gaussians (several exp() each) per photon PER QUERY dominated mode-M render
    // time (~74% profiled), so build() computes each photon's triple exactly once and
    // queries index cie[k] instead. The stored values are the exact same doubles the
    // direct call produced (same function, same float->double promoted lambda), so
    // gather results are bit-identical.
    std::vector<Vec3>   cie;
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
        if (photons.empty()) { nx = ny = nz = 1; cellStart.assign(2, 0); cie.clear(); pos.clear(); return; }

        Vec3 mn = pos[0], mx = pos[0];
        for (const Vec3& p : pos) {
            mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
            mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
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
            int ix, iy, iz; cellCoord(pos[i], ix, iy, iz);
            int c = cellIndex(ix, iy, iz);
            cellOf[i] = c;
            ++cellStart[c + 1];
        }
        // Prefix sum -> cellStart[c] = begin offset of cell c.
        for (long long c = 0; c < nCells; ++c) cellStart[c + 1] += cellStart[c];

        // Pass 2: scatter into cell-contiguous order. pos[] and photons[] are permuted by
        // the SAME cursor walk, so index k keeps addressing one photon across both.
        std::vector<Photon> sorted(photons.size());
        std::vector<Vec3>   sortedPos(pos.size());
        std::vector<int> cursor(cellStart.begin(), cellStart.end() - 1);
        for (size_t i = 0; i < photons.size(); ++i) {
            const int d = cursor[cellOf[i]]++;
            sorted[d]    = photons[i];
            sortedPos[d] = pos[i];
        }
        photons.swap(sorted);
        pos.swap(sortedPos);

        // Precompute each (sorted) photon's CIE XYZ triple once — see `cie` above.
        // Embarrassingly parallel and worth threading: a large map costs several exp()
        // per photon, which would otherwise serialize ~1s+ onto the build.
        const size_t nPh = photons.size();
        cie.resize(nPh);
        auto fill = [&](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i) {
                const double l = photons[i].lambda;
                cie[i] = Vec3{cieX(l), cieY(l), cieZ(l)};
            }
        };
        unsigned nT = std::max(1u, std::thread::hardware_concurrency());
        if (nPh < 65536 || nT == 1) {
            fill(0, nPh);
        } else {
            std::vector<std::thread> pool;
            for (unsigned t = 0; t < nT; ++t)
                pool.emplace_back(fill, nPh * t / nT, nPh * (t + 1) / nT);
            for (auto& th : pool) th.join();
        }
    }

    // Invoke fn(const Photon&, double dist2, int index) for every photon within
    // `radius` of p. `index` addresses the photon's row in photons[] / cie[].
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
                    // The reject scan reads ONLY pos[] (see the layout note on PhotonMap):
                    // ~85% of the candidates in this 3x3x3 box fail the test, and for those
                    // the fat payload record is never touched at all.
                    const Vec3* __restrict pp = pos.data();
                    for (int k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                        Vec3 d = p - pp[k];
                        double d2 = dot(d, d);
                        if (d2 <= r2) fn(photons[k], d2, k);
                    }
                }
            }
        }
    }
};
