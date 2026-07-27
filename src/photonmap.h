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
    // so a radius-r query touches only the 3x3x3 neighbourhood) via a counting sort,
    // then precompute the per-photon CIE triples.
    //
    // Split into buildGrid + fillCie because buildAuto() below needs to bin TWICE (once at
    // the probe radius, once at the chosen one) but only ever needs the CIE pass once —
    // and on a large map that pass is several exp() per photon, i.e. the expensive half.
    void build(double r) { buildGrid(r); fillCie(); }

    void buildGrid(double r) {
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
    }

    // Precompute each (sorted) photon's CIE XYZ triple once — see `cie` above.
    // Embarrassingly parallel and worth threading: a large map costs several exp()
    // per photon, which would otherwise serialize ~1s+ onto the build.
    void fillCie() {
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

    // MEDIAN number of OTHER photons inside the current gather radius of a deposited
    // photon — i.e. how many photons a typical gather actually sums. Requires a built grid.
    //
    // MEDIAN rather than mean: a caustic can hold orders of magnitude more photons per unit
    // area than the walls around it, and a mean would be dragged up by that tail and shrink
    // the radius until the dim parts of the image went to noise. Median asks "how dense is it
    // where photons typically are", which is the quantity the estimator is tuned on.
    //
    // THE SAMPLE MUST NOT DEPEND ON ARRAY ORDER. The obvious implementation — stride over
    // photons[] and query around every n-th one — is order-dependent: the counting sort is
    // stable, so photons within a cell keep their arrival order, and a map that has been
    // saved (already sorted at one radius) and reloaded re-sorts to a different within-cell
    // order than the original deposit did. Same photons, same grid, different sample, and
    // hence a slightly different chosen radius — so `-loadmap` would no longer reproduce the
    // `-savemap` run that wrote the file. (Caught 2026-07-26: 1401 vs 1402 median, r
    // 0.008981 vs 0.008977, images differed.)
    //
    // So sample by CELL — cells are fixed by the bbox and cell size, i.e. by geometry alone —
    // striding over occupied cells, and inside each take the lexicographically smallest
    // position as the representative. That is a set-minimum, so it names the same photon no
    // matter how the array is ordered (coincident positions would tie, but then the neighbour
    // count is identical anyway). Querying an actual photon rather than the cell centre
    // matters: a cell the surface merely clips at a corner has its centre off the surface and
    // would report a spuriously empty neighbourhood.
    //
    // Cell-striding also makes the statistic area-weighted rather than photon-weighted, which
    // is if anything the better match: camera gathers land on visible surface points, spread
    // over area, not in proportion to local photon density.
    double medianNeighborCount(int sampleTarget = 4096) const {
        const long long nCells = cellCount();
        if (photons.empty() || nCells <= 0) return 0.0;

        // How many cells actually hold photons — needed to pick a stride that lands near
        // sampleTarget. O(nCells) over a flat int array (~20 ms at 22M cells, against a
        // build that already costs seconds at that size).
        long long occupied = 0;
        for (long long c = 0; c < nCells; ++c)
            if (cellStart[c + 1] > cellStart[c]) ++occupied;
        if (occupied == 0) return 0.0;

        const long long stride = std::max<long long>(1, occupied / std::max(1, sampleTarget));
        std::vector<int> counts;
        counts.reserve((size_t)(occupied / stride) + 1);
        long long seenOccupied = 0;
        for (long long c = 0; c < nCells; ++c) {
            const int b = cellStart[c], e = cellStart[c + 1];
            if (e <= b) continue;
            if ((seenOccupied++ % stride) != 0) continue;
            int rep = b;
            for (int k = b + 1; k < e; ++k) {
                const Vec3& a = pos[k];
                const Vec3& m = pos[rep];
                if (a.x != m.x ? a.x < m.x
                  : a.y != m.y ? a.y < m.y
                               : a.z <  m.z) rep = k;
            }
            int n = 0;
            queryR(pos[rep], radius, [&](const Photon&, double, int) { ++n; });
            counts.push_back(n - 1);          // drop the representative itself (dist2 == 0)
        }
        if (counts.empty()) return 0.0;
        const size_t mid = counts.size() / 2;
        std::nth_element(counts.begin(), counts.begin() + mid, counts.end());
        return (double)counts[mid];
    }

    // Build the grid at a gather radius CHOSEN FROM THE MEASURED PHOTON DENSITY, instead of
    // from the scene size alone. Returns the radius actually used.
    //
    // WHY: a radius picked independently of the photon count is the reason mode M scales
    // badly. `build`'s grid is sized at cellSize == r, so with r fixed the grid resolution
    // is frozen no matter how many photons land in it (the same scene reports the same
    // 59x59x59 grid at 500k emitted and at 8M), photons-per-cell grows linearly with `-n`,
    // and every pixel's 3x3x3 scan grows with it — the render gets slower per sample the
    // more photons you ask for, which is the opposite of what a user expects.
    //
    // THE EXPONENT MATTERS, and "keep the cost flat" is the wrong target. Holding the disc
    // population constant means r ~ N^(-1/2), which holds *variance* constant too: the
    // image would then never converge in noise however many photons you traced, only in
    // bias. So the target population is grown deliberately but sublinearly,
    //
    //     k(M) = kAt1M * cbrt(M / 1e6),          M = stored photons
    //
    // which gives r ~ M^(-1/3): per-query cost grows as M^(1/3) (so total gather cost is
    // mild), noise falls as k^(-1/2) ~ M^(-1/6), and bias falls as r^2 ~ M^(-2/3). Both
    // error terms go to zero, which is the property r ~ N^(-1/2) lacks. (The MSE-optimal
    // 2-D kernel bandwidth is r ~ N^(-1/6); that converges too but leaves cost growing as
    // N^(2/3), which is most of the original problem. N^(-1/3) is the engineering pick.)
    //
    // The disc population is measured, not assumed: bin once at the requested radius r0,
    // ask medianNeighborCount() how many photons a typical gather sees, then rescale by
    // sqrt(k/n) (a surface density estimate is 2-D, so population ~ r^2) and re-bin. Two
    // counting sorts, one CIE pass — see build().
    // CALIBRATION of the default kAt1M: it is deliberately set to roughly what the old
    // fixed radius already delivered at ordinary photon counts, so turning this on does not
    // restyle everybody's existing renders. Measured on scraps/abs_herosplit.ftsl at the
    // default -pmradiusfrac, the old radius gave a typical gather 311 / 1256 / 5045 / 20221
    // photons at 1.67M / 6.7M / 26.7M / 107M stored — i.e. ~185 at 1M stored, hence 200.
    // The win is in the growth: those same four points become ~237 / 377 / 598 / 949, so the
    // population is roughly preserved where people actually render and only the runaway tail
    // is cut. Equal-time quality is strictly better even at the small end (see README).
    double buildAuto(double r0, double kAt1M = 200.0, double* nProbeOut = nullptr,
                     double* kTargetOut = nullptr) {
        if (photons.empty()) { build(r0); return radius; }
        buildGrid(r0);
        const double n0 = medianNeighborCount();
        const double k  = kAt1M * std::cbrt((double)photons.size() / 1.0e6);
        if (nProbeOut)  *nProbeOut  = n0;
        if (kTargetOut) *kTargetOut = k;
        if (n0 <= 0.0 || k <= 0.0) { fillCie(); return radius; }   // too sparse to judge

        double r1 = r0 * std::sqrt(k / n0);
        // Never wander more than a couple of octaves from what was asked for: the probe is
        // a median over a sample, and a pathological scene (all photons in one caustic, or
        // a single lit texel) shouldn't be allowed to pick an absurd grid.
        r1 = std::min(std::max(r1, r0 / 64.0), r0 * 4.0);
        // Hard memory guard: cellStart is one int per cell and cellIndex() is int math, so
        // the cell count has to stay well inside 2^31 regardless of what the probe wants.
        // Grow r1 back until the predicted grid fits.
        {
            // The probe binning already measured the padded bbox: nx*cellSize etc. is it
            // (rounded up), so no second pass over 10s of millions of positions is needed.
            const double ex = nx * cellSize, ey = ny * cellSize, ez = nz * cellSize;
            // Budget the cell array against the MAP, not against an absolute number: photons
            // live on surfaces (a 2-D sheet through a 3-D grid), so most cells are empty and
            // the count runs as (L/r)^3 while the useful occupancy runs as (L/r)^2 — an
            // absolute cap therefore binds at exactly the large-map sizes this function
            // exists to serve. 2 cells/photon is 8 B/photon of cellStart against 56 B/photon
            // of map (~14% overhead), with a floor so small maps can still refine freely.
            // At the calibrated default this guard never binds; it is here for -pmcount
            // extremes and for pathological (near-planar, one-caustic) scenes.
            const double kMaxCells = std::max(1.6e7, 2.0 * (double)photons.size());
            for (int guard = 0; guard < 64; ++guard) {
                const double cells = std::ceil(ex / r1) * std::ceil(ey / r1)
                                   * std::ceil(ez / r1);
                if (cells <= kMaxCells) break;
                r1 *= 1.25;
            }
        }
        if (std::abs(r1 / r0 - 1.0) > 0.05) buildGrid(r1);   // else keep the probe binning
        fillCie();
        return radius;
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
