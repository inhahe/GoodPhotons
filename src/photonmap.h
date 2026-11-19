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
#include <cstdlib>          // photonMaxBounce: std::getenv
#include <utility>
#include <queue>
#include <unordered_map>
#include "linalg.h"
#include "gatherplane.h"   // onGatherSurfaceSided: which photons a gather keeps
#include "merge_grid.h"    // mergeLatticeKey: the hashed lattice
#include "parallel.h"      // ft::parallelForAll, ft::workerCount: the build's threads
#include "color.h"
#include "allocreport.h"   // OOM that names the buffer, its size and the flag that sizes it
#include "workcount.h"    // tlWork: a query's candidates, for the CPU refresh (MODEM-CPU-NONDET)

// FTRACE_PHOTONBOUNCE=<n> (`-photon-bounce <n>`): cap the LIGHT path's bounce count in the
// photon passes of modes M and S, host (tracePhotonPass) and device (launchForward). It exists
// because nothing else could reach it. `-max-bounce` governs the CAMERA path, and in mode M that
// path stops at the first diffuse hit, so `-max-bounce 2`, `32` and `64` produce byte-identical
// mode-M images -- a perfectly clean, perfectly meaningless null for any question about how far
// LIGHT travels before it is deposited. Default 32. Here, in the header both backends include, so
// they read one value.
inline int photonMaxBounce() {
    static const int m = [] {
        const char* e = std::getenv("FTRACE_PHOTONBOUNCE");
        const int v = e ? std::atoi(e) : 0;
        return v >= 1 ? v : 32;
    }();
    return m;
}

// ---- Caustic split: the one tunable of the FOCUS/SCATTER classification ---------------
// Mode M partitions its deposits into a global map and a caustic map by the classic L·S⁺·D
// rule; the classifier itself is photonVertexKind (render.h) and its device twin
// dPhotonVertexBit (render_cuda.cu). This threshold is the only judgement call in it, and it
// lives HERE, in the one header both of them include, because a CPU and a GPU render that
// disagreed about which glossy surfaces can focus would produce two different images from one
// scene — a discrepancy with no symptom other than "the GPU looks wrong".
//
// `roughness` is the lobe's width (0 = perfect mirror; ggx.h, ROUGHNESS). 0.15 is about where
// the lobe stops being able to hold a caustic together over a scene-scale throw: GGX alpha 0.097,
// a lobe some 7 degrees to half maximum in the reflected direction, with GGX's long tail
// beyond. Anything broader produces a diffuse
// glow that the global map already represents perfectly well — and folding that broad
// population into the caustic map would drag ITS density estimate back up, which is exactly
// what the split exists to prevent.
inline constexpr double kCausticGlossRoughness = 0.15;

// One deposited photon's PAYLOAD — everything the gather reads *after* a candidate has
// passed the distance test. The deposit POSITION deliberately lives in a separate
// `PhotonMap::pos` array (see below); this struct is what `pos[k]` indexes into.
//
// `n` (the true normal on the side the photon arrived from) is here for cross-surface leak
// rejection -- the true one, so a photon on a sheet's other face can never pass for this one's,
// however a normal map tilts the shading normals; a pure Lambertian
// estimate only needs power/lambda. There is no `wi`: the incident direction was stored
// for years and never read by any gather (the estimate is Lambertian, so it only needs
// the normal), which cost 24 dead bytes on every one of what can be tens of millions of
// records. Don't re-add a field here without a reader — see the size note below.
struct Photon {
    Vec3  n;          // true normal at the deposit, on the side the photon came from (unit)
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

// The photon map's lattice is the merge maps' hashed one (merge_grid.h, mergeLatticeKey), shared with
// the device's gathers (dPmNeighborhood) and its SPPM build (mergeBinDevice), which must find each
// photon in the bucket the host's counting sort put it in.
//
// WHY THE GRID IS HASHED AND NOT DENSE. A dense grid indexes cells as (iz*ny+iy)*nx+ix and
// therefore has to ALLOCATE nx*ny*nz ints, which grows as (L/r)^3 in the scene size L over the
// gather radius r — while the photons themselves live on surfaces, a 2-D sheet whose occupancy
// only grows as (L/r)^2. So on any large scene the empty cells, not the photons, set the memory
// bill, and the map had to defend itself with a guard that GREW r back until the cell array fit
// (kMaxCells, removed in 0.199.6). That guard is what made small radii unreachable exactly where
// they matter: on gallery_rain (scene radius 32.7 m) a requested r of 0.031 m was inflated to
// 0.094 m, and a ~3 cm caustic under a ~10 cm kernel is smeared into a colourless grey veil —
// the reported "no colourful caustics anywhere" (2026-09-01).
//
// Hashing removes the volume term completely: the table is sized from the PHOTON COUNT, so the
// cost per photon is a constant ~8 B no matter how fine the cells are, and r is free to follow
// the measured density wherever it leads. A bucket holds a cell of every block that hashes to its
// slot; the query distance-tests every candidate, so a stray is simply rejected, and with about two
// buckets a photon that is ~13 extra distance tests across a 3x3x3 neighbourhood, against the
// hundreds of genuine candidates a tuned gather visits. A gather's 27 cells always fall in 27
// different buckets, so none is read twice -- two sharing one would count its photons twice.
//
// The hash grid over deposited photons. After build(), the three per-photon arrays
// are reordered together into bucket-contiguous runs and `cellStart` gives each bucket's
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
    // The bulk arrays are BulkVecs: a resize leaves new elements unwritten, so whatever resizes one
    // writes every element before a read -- the deposit's concatenation, -loadmap's read, the
    // device's download, the binning, fillCie.
    ftalloc::BulkVec<Vec3>   pos;       // deposit positions (world); the distance-test stream
    ftalloc::BulkVec<Photon> photons;   // payloads, parallel to pos[]; reordered by build()
    // Per-photon CIE XYZ response at photons[i].lambda, filled by build(). The gather
    // estimate weights every photon by cie(lambda_p); evaluating the analytic CIE
    // multi-Gaussians (several exp() each) per photon PER QUERY dominated mode-M render
    // time (~74% profiled), so build() computes each photon's triple exactly once and
    // queries index cie[k] instead. The stored values are the exact same doubles the
    // direct call produced (same function, same float->double promoted lambda), so
    // gather results are bit-identical.
    ftalloc::BulkVec<Vec3>   cie;
    long long nEmitted = 0;        // total photons EMITTED in the pass (normalization)
    double    radius   = 0.02;     // gather radius (world units); == grid cell size

    // PER-QUERY ADAPTIVE GATHER (0 = off; see adaptiveRadius below). When > 0 this is the
    // number of photons a single gather wants to see, and `radius` becomes a MAXIMUM rather
    // than the radius actually used. Set by buildAuto to the same k it solved the global
    // radius for; only the CAUSTIC map turns it on (see main.cpp -pmadaptive).
    double    kGather  = 0.0;

    // grid geometry: cells the gather radius wide (merge_grid.h)
    MergeLattice lat = mergeLatticeAt(0, 0, 0, 0.02, 1);
    // Padded bbox cell dims. PURELY DIAGNOSTIC since 0.199.6 — the lattice is hashed, so
    // nothing indexes through these and their product is never formed (it overflows a long
    // long on a fine grid over a large scene, which is the whole reason the dense grid went).
    // buildAuto still reads them as the measured bbox EXTENT in cells, and the mode-M log
    // prints them so "how fine is the lattice" stays visible.
    int    nx = 1, ny = 1, nz = 1;
    // Bucket runs: bucket b occupies [cellStart[b], cellStart[b+1]). Size bucketCount() + 1. A BulkVec,
    // as the arrays above: a binning on several threads writes every entry.
    ftalloc::BulkVec<int> cellStart;

    long long bucketCount() const { return (long long)mergeBucketCount(lat.slots); }

    // Bin the deposited photons into a uniform grid of cell size `r` (== gather radius,
    // so a radius-r query touches only the 3x3x3 neighbourhood) via a counting sort,
    // then precompute the per-photon CIE triples.
    //
    // Split into buildGrid + fillCie because buildAuto() below needs to bin TWICE (once at
    // the probe radius, once at the chosen one) but only ever needs the CIE pass once —
    // and on a large map that pass is several exp() per photon, i.e. the expensive half.
    void build(double r) {
        {
            BinScratch s;
            buildGrid(r, s);
            cie.swap(s.pos);   // the CIE table's storage (BinScratch)
        }
        fillCie();
    }

    // A binning's destination, the photons' size: what buildGrid moves the photons into it swaps with
    // photons[] and pos[], so a second binning given the same scratch (buildAuto bins twice) finds it
    // allocated -- at 129 M photons 7 GB it would otherwise fault in again. Every element is written by
    // the binning (BulkVec: a resize leaves them as they were). A local of build and buildAuto, so it
    // goes on every way out. Once the binnings are done its positions array, the CIE table's type and
    // size, is swapped into `cie`, so fillCie allocates nothing while the rest is still being freed.
    struct BinScratch {
        ftalloc::BulkVec<Photon> ph;
        ftalloc::BulkVec<Vec3>   pos;
        // Freed in the background (ftalloc::releaseInBackground): the build does not wait the
        // 0.6-0.9 s the system takes to release 7 GB of pages.
        ~BinScratch() { ftalloc::releaseInBackground(ph); ftalloc::releaseInBackground(pos); }
    };

    void buildGrid(double r, BinScratch& s) {
        radius = r;
        const double cellSize = (r > 0.0) ? r : 1e-6;
        if (photons.empty()) {
            nx = ny = nz = 1; lat = mergeLatticeAt(0, 0, 0, cellSize, 1);
            cellStart.assign(mergeBucketCount(1) + 1, 0);
            cie.clear(); pos.clear(); return;
        }

        Vec3 mn = pos[0], mx = pos[0];
        for (const Vec3& p : pos) {
            mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
            mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
        }
        lat = mergeLatticeOver(mn.x, mn.y, mn.z, cellSize, photons.size());
        // Diagnostic only, and clamped into int: a fine lattice over a large scene can want
        // billions of cells per axis, which is exactly the situation the hash exists to make
        // affordable — it must not overflow the number we merely PRINT. The box half a cell
        // past the photons each side.
        const Vec3 ext = (mx - mn) + Vec3{cellSize, cellSize, cellSize};
        auto dim = [](double e, double c) {
            const double d = std::ceil(e / c);
            return (int)std::min(std::max(d, 1.0), 2.0e9);
        };
        nx = dim(ext.x, cellSize); ny = dim(ext.y, cellSize); nz = dim(ext.z, cellSize);

        // Bin, scattering pos[] and photons[] to the SAME positions, so index k keeps
        // addressing one photon across both.
        ftalloc::resize(s.ph, photons.size(), "the photon map payloads (binning)", "-n");
        ftalloc::resize(s.pos, pos.size(), "the photon map positions (binning)", "-n");
        mergeBin(lat, photons.size(), [&](size_t i) -> const Vec3& { return pos[i]; },
                 [&](size_t i, size_t k) { s.ph[k] = photons[i]; s.pos[k] = pos[i]; }, cellStart, nullptr,
                 "the photon grid bucket table", "-n");
        photons.swap(s.ph);
        pos.swap(s.pos);
    }

    // Precompute each (sorted) photon's CIE XYZ triple once — see `cie` above.
    // Embarrassingly parallel and worth threading: a large map costs several exp()
    // per photon, which would otherwise serialize ~1s+ onto the build.
    void fillCie() {
        const size_t nPh = photons.size();
        ftalloc::resize(cie, nPh, "the photon CIE table", "-n");
        ft::parallelForAll(nPh, 32768, [&](size_t i) {
            const double l = photons[i].lambda;
            cie[i] = Vec3{cieX(l), cieY(l), cieZ(l)};
        });
    }

    // The radius probe's cell of a point (medianNeighborCount): its coordinates in the lattice and their
    // 64-bit mix, which orders the probe's sample (probeBefore).
    struct ProbeCell { int c[3]; unsigned long long h; };
    ProbeCell probeCellOf(const Vec3& p) const {
        ProbeCell k;
        k.c[0] = mergeCellCoord(p.x, lat.lo[0], lat.invCell);
        k.c[1] = mergeCellCoord(p.y, lat.lo[1], lat.invCell);
        k.c[2] = mergeCellCoord(p.z, lat.lo[2], lat.invCell);
        unsigned long long h = (unsigned long long)(unsigned int)k.c[0] * 0x9E3779B97F4A7C15ull ^
                               (unsigned long long)(unsigned int)k.c[1] * 0xC2B2AE3D27D4EB4Full ^
                               (unsigned long long)(unsigned int)k.c[2] * 0x165667B19E3779F9ull;
        h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ull;
        h ^= h >> 27; h *= 0x94D049BB133111EBull;
        h ^= h >> 31;
        k.h = h;
        return k;
    }
    // The sample's order: by the mix, a tie (two cells mixing alike) by the cell.
    static bool probeBefore(const ProbeCell& a, const ProbeCell& b) {
        if (a.h != b.h) return a.h < b.h;
        if (a.c[0] != b.c[0]) return a.c[0] < b.c[0];
        if (a.c[1] != b.c[1]) return a.c[1] < b.c[1];
        return a.c[2] < b.c[2];
    }
    static bool probeLexLess(const Vec3& a, const Vec3& m) {
        return a.x != m.x ? a.x < m.x : a.y != m.y ? a.y < m.y : a.z < m.z;
    }
    // The probe's sample: the `sampleTarget` occupied cells whose coordinates hash lowest, in
    // probeBefore's order, each with its representative, its lexicographically smallest photon.
    std::vector<std::pair<ProbeCell, Vec3>> probeSample(int sampleTarget = 4096) const {
        const size_t n = photons.size();
        if (n == 0 || sampleTarget <= 0) return {};
        struct CellHash { size_t operator()(const ProbeCell& k) const { return (size_t)k.h; } };
        struct CellEq {
            bool operator()(const ProbeCell& a, const ProbeCell& b) const {
                return a.c[0] == b.c[0] && a.c[1] == b.c[1] && a.c[2] == b.c[2];
            }
        };
        using Reps = std::unordered_map<ProbeCell, Vec3, CellHash, CellEq>;
        const size_t K = (size_t)sampleTarget;
        const unsigned T = mergeThreadsFor(n);
        std::vector<Reps> kept(T);
        ft::parallelForAll(T, 1, [&](size_t t) {
            Reps& reps = kept[t];
            reps.reserve(2 * K);
            // The share's bottom k so far, the last of them in probeBefore's order on top (a priority
            // queue keeps its greatest by the order it is given at the top).
            std::priority_queue<ProbeCell, std::vector<ProbeCell>, decltype(&probeBefore)> heap(&probeBefore);
            for (size_t i = n * t / T, e = n * (t + 1) / T; i < e; ++i) {
                const Vec3& p = pos[i];
                const ProbeCell k = probeCellOf(p);
                if (heap.size() == K && probeBefore(heap.top(), k)) continue;   // past the bottom k
                const auto it = reps.find(k);
                if (it != reps.end()) {
                    if (probeLexLess(p, it->second)) it->second = p;
                    continue;
                }
                reps.emplace(k, p);
                heap.push(k);
                if (heap.size() > K) {
                    reps.erase(heap.top());
                    heap.pop();
                }
            }
        });
        Reps all = std::move(kept[0]);
        for (unsigned t = 1; t < T; ++t)
            for (const auto& kv : kept[t]) {
                const auto it = all.find(kv.first);
                if (it == all.end()) all.emplace(kv.first, kv.second);
                else if (probeLexLess(kv.second, it->second)) it->second = kv.second;
            }
        std::vector<std::pair<ProbeCell, Vec3>> cells(all.begin(), all.end());
        const size_t m = std::min(K, cells.size());
        std::partial_sort(cells.begin(), cells.begin() + (ptrdiff_t)m, cells.end(),
                          [](const std::pair<ProbeCell, Vec3>& a, const std::pair<ProbeCell, Vec3>& b) {
                              return probeBefore(a.first, b.first);
                          });
        cells.resize(m);
        return cells;
    }
    // MEDIAN number of OTHER photons inside the current gather radius of a deposited
    // photon — i.e. how many photons a typical gather actually sums. Requires a built grid.
    //
    // MEDIAN rather than mean: a caustic can hold orders of magnitude more photons per unit
    // area than the walls around it, and a mean would be dragged up by that tail and shrink
    // the radius until the dim parts of the image went to noise. Median asks "how dense is it
    // where photons typically are", which is the quantity the estimator is tuned on.
    //
    // THE SAMPLE IS A FUNCTION OF THE PHOTONS' POSITIONS ALONE: not of their order in the array,
    // nor of the table they are binned in. Not of the order, because the counting sort is stable,
    // so photons within a cell keep their arrival order, and a map that has been saved (already
    // sorted at one radius) and reloaded re-sorts to a different within-cell order than the
    // original deposit did: a sample that strided over photons[] would differ, and `-loadmap`
    // would no longer reproduce the `-savemap` run that wrote the file. Not of the table, because then
    // every change to the table's size or hash would move the chosen radius, and every render
    // that leaves it automatic with it.
    //
    // So it samples CELLS: the `sampleTarget` occupied cells whose coordinates hash lowest (a
    // bottom-k sample of the cells mergeCellCoord places the photons in, under a 64-bit mix), each
    // represented by its lexicographically smallest photon. Both are set functions of the
    // positions (coincident positions would tie, but then the neighbour count is identical
    // anyway). Querying an actual photon rather than the cell centre matters: a cell the surface
    // merely clips at a corner has its centre off the surface and would report a spuriously empty
    // neighbourhood.
    //
    // Cell sampling also makes the statistic area-weighted rather than photon-weighted, which
    // is if anything the better match: camera gathers land on visible surface points, spread
    // over area, not in proportion to local photon density.
    //
    // On mergeThreadsFor's threads: each thread keeps the bottom k of the cells its share of the
    // photons lies in -- a cell of the whole bottom k is in the bottom k of every share it appears in,
    // so the shares' union holds it, its representative the least of theirs -- and the k queries are
    // split among the threads, their candidates added to this thread's tally as one walk adds them.
    double medianNeighborCount(int sampleTarget = 4096) const {
        const std::vector<std::pair<ProbeCell, Vec3>> cells = probeSample(sampleTarget);
        const size_t m = cells.size();
        if (m == 0) return 0.0;
        const unsigned T = mergeThreadsFor(photons.size());
        std::vector<int> counts(m);
        const unsigned Tq = (m >= 64) ? T : 1u;
        // Each part's candidates taken out of whichever thread's tally ran it and put in the caller's,
        // as one walk on the caller's thread would have left them.
        std::vector<long long> cand(Tq, 0);
        ft::parallelForAll(Tq, 1, [&](size_t t) {
            const long long before = bdpt::tlCandidates();
            for (size_t j = m * t / Tq, e = m * (t + 1) / Tq; j < e; ++j)
                counts[j] = countR(cells[j].second, radius) - 1;   // not the representative itself
            cand[t] = bdpt::tlCandidates() - before;
            bdpt::tlCandidates() = before;
        });
        for (unsigned t = 0; t < Tq; ++t) bdpt::tlCandidates() += cand[t];
        const size_t mid = counts.size() / 2;
        std::nth_element(counts.begin(), counts.begin() + (ptrdiff_t)mid, counts.end());
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
    //
    // `rMax` > 0 caps the chosen radius. Its one caller is the CAUSTIC map of the two-map
    // split, which passes the GLOBAL map's radius — because a caustic map that ends up
    // blurrier than the map it was separated from is strictly worse than not splitting at
    // all. That happens whenever the caustic population is very sparse (a scene where only a
    // thousandth of the flux reaches a diffuse surface through glass): k(M) shrinks with the
    // count, the probe sees almost nothing, and sqrt(k/n) asks to GROW the radius — smearing
    // the one thing the split exists to keep sharp. Capped, the worst case degenerates to
    // "same radius as before, just estimated from two disjoint sets", which is exactly the
    // unsplit answer.
    double buildAuto(double r0, double kAt1M = 200.0, double* nProbeOut = nullptr,
                     double* kTargetOut = nullptr, double rMax = 0.0) {
        if (photons.empty()) { build(r0 > 0.0 && rMax > 0.0 ? std::min(r0, rMax) : r0);
                               return radius; }
        {
            BinScratch s;   // both binnings'; its positions array becomes the CIE table's storage
            buildGrid(r0, s);
            const double n0 = medianNeighborCount();
            const double k  = kAt1M * std::cbrt((double)photons.size() / 1.0e6);
            if (nProbeOut)  *nProbeOut  = n0;
            if (kTargetOut) *kTargetOut = k;
            if (n0 > 0.0 && k > 0.0) {   // else too sparse to judge: the probe binning stays
                double r1 = r0 * std::sqrt(k / n0);
                // Never wander more than a couple of octaves from what was asked for: the probe is
                // a median over a sample, and a pathological scene (all photons in one caustic, or
                // a single lit texel) shouldn't be allowed to pick an absurd grid.
                r1 = std::min(std::max(r1, r0 / 64.0), r0 * 4.0);
                if (rMax > 0.0 && r1 > rMax) r1 = rMax;   // see the rMax note above
                // No memory bound on r1: the table is sized by the PHOTON COUNT (merge_grid.h), not
                // by the lattice's volume, so a fine radius over a large scene costs no more table
                // than a coarse one, and the only limit is the two-octave clamp above.
                if (std::abs(r1 / r0 - 1.0) > 0.05) buildGrid(r1, s);   // else keep the probe binning
            }
            cie.swap(s.pos);
        }
        fillCie();
        return radius;
    }

    // Invoke fn(const Photon&, double dist2, int index) for every photon within
    // `radius` of p. `index` addresses the photon's row in photons[] / cie[].
    template <class F>
    void query(const Vec3& p, F&& fn) const { queryR(p, radius, std::forward<F>(fn)); }
    // The slack of this map's plane test (onGatherSurfaceSided) for a gather point on a curve of radius
    // `fiberR` (0 off one): one value for the adaptive radius's count and the gather's sum.
    double gatherSlack(double fiberR) const { return gatherBandSlack(radius, fiberR); }

    // Same, but with an explicit query radius `r`. The 3x3x3 cell neighbourhood only
    // covers radii up to the grid cell size, so the CALLER MUST ensure r <= lat.cell
    // (true for PPM/SPPM, where the grid is built at the largest current per-pixel
    // radius and every pixel's radius only shrinks from there).
    template <class F>
    void queryR(const Vec3& p, double r, F&& fn) const {
        if (photons.empty()) return;
        const double r2 = r * r;
        long long cand = 0;   // the candidates tested: a gather's cost, for the CPU refresh
        // No per-axis bounds tests: the lattice is hashed, so an out-of-bbox cell coordinate is
        // a legal bucket lookup that simply finds no photon within r (or finds a stray and
        // rejects it on distance, below).
        // The reject scan reads ONLY pos[] (see the layout note on PhotonMap): ~85% of the
        // candidates in this 3x3x3 box fail the test, and for those the fat payload record is
        // never touched at all. The scan reads locals alone (the query point copied in, not read
        // through `p`): the callback stores into its captures each match (a gather's sums), and a
        // store the compiler cannot place would have it reload anything read through a reference
        // (5 % of a gather's walk).
        const Vec3 q = p;
        const Vec3* __restrict pp = pos.data();
        const Photon* ph = photons.data();
        const int* cs = cellStart.data();
        MergeKey keys[27];
        mergeNeighbourKeys(lat, q.x, q.y, q.z, keys);
        for (const MergeKey& c : keys) {
            const int b = cs[c.bucket], e = cs[c.bucket + 1];
            cand += e - b;
            for (int k = b; k < e; ++k) {
                const Vec3 d = q - pp[k];
                const double d2 = dot(d, d);
                if (d2 <= r2) fn(ph[k], d2, k);
            }
        }
        bdpt::tlCandidates() += cand;
    }

    // The number of photons within `r` of p -- what a queryR counting in its callback finds, and
    // the candidates it adds to the CPU refresh's tally -- counted in a local. A count through a
    // callback's capture is a store to one memory word a match, each waiting on the last: on a dense
    // probe (medianNeighborCount, tens of thousands of matches a query) that chain took four times
    // the walk's own cost.
    int countR(const Vec3& p, double r) const {
        if (photons.empty()) return 0;
        const double r2 = r * r;
        const Vec3 q = p;
        const Vec3* __restrict pp = pos.data();
        const int* cs = cellStart.data();
        MergeKey keys[27];
        mergeNeighbourKeys(lat, q.x, q.y, q.z, keys);
        long long cand = 0;
        int n = 0;
        for (const MergeKey& c : keys) {
            const int b = cs[c.bucket], e = cs[c.bucket + 1];
            cand += e - b;
            for (int k = b; k < e; ++k) {
                const Vec3 d = q - pp[k];
                if (dot(d, d) <= r2) ++n;
            }
        }
        bdpt::tlCandidates() += cand;
        return n;
    }

    // ------------------------- PER-QUERY ADAPTIVE GATHER RADIUS -------------------------
    // Returns the radius this particular gather should use: the radius of the smallest disc
    // around `p` that holds `kGather` photons of its own surface (onGatherSurfaceSided), capped at
    // `radius`.
    // `kGather <= 0` returns `radius` unchanged, so every caller is bit-identical when the
    // feature is off.
    //
    // WHY. A single map-wide radius cannot serve a scene whose photon density spans orders of
    // magnitude, and a CAUSTIC map is exactly that scene: a gem's focused light lands as a thin
    // filament thousands of times denser than the specular wash covering the rest of the room.
    // The two-map split (see main.cpp) was supposed to fix this by giving caustics their own
    // radius — but one radius per MAP is still one radius, and buildAuto's median-density probe
    // is dominated by the wash, so on gallery_rain the caustic map's chosen radius pinned to
    // the global map's (0.1775 m) — 5-10x wider than the features it exists to preserve. The
    // measured consequence: the axicon cap's caustic peaked at 2.9x its own median where the
    // mode-D reference peaks at 9.7x, i.e. the caustic was still there, still the right colour,
    // and smeared into a flat pastel wash. Forcing the map-wide radius down instead recovers
    // the peak (8.1x at r/8) and turns the whole rest of the image into chromatic speckle,
    // because each sparse-region gather then catches one or two SPECTRAL photons and paints a
    // random saturated hue. Adapting PER QUERY is the only thing that gets both: the filament
    // shrinks its own kernel to its own scale, the wash keeps the wide one it needs.
    //
    // HOW, in ONE neighbourhood walk. A kNN search would need a heap (and a heap per thread is
    // exactly what a GPU gather cannot afford), and iterating "count, shrink, recount" costs a
    // full 3x3x3 walk per iteration. Instead the single walk histograms each candidate's
    // squared distance into geometric shells t = d^2/r^2 in (2^-(i+1), 2^-i], so the suffix sum
    // C_i = sum_{j>=i} hist[j] is the population inside radius r*2^(-i/2) — a whole radius
    // profile from one pass over the same candidates the gather was going to visit anyway.
    // Take the tightest shell that still holds k, then interpolate inside it under the local
    // uniform-density assumption (count grows as r^2): r_q^2 = r_i^2 * k / C_i.
    //
    // The surface test (onGatherSurfaceSided: facing, and near the tangent plane) is applied HERE and
    // not only in the caller's sum, deliberately. Counting a wall's photons while gathering the
    // floor beside it would shrink the radius for a population the sum then rejects, and the
    // estimate is divided by pi*r_q^2 either way — so the mismatch would print a dark seam along
    // every surface junction.
    static const int kAdaptBins = 16;   // covers r down to r * 2^-8 = r/256

    // `fiberR`: the gather point's curve radius (0 off one); its sum takes gatherSlack(fiberR) too.
    // `twoSided`: a translucent gather point, whose sum takes its far side's photons too
    // (onGatherSurfaceSided), so the radius counts them as well.
    double adaptiveRadius(const Vec3& p, const Vec3& n, double fiberR, bool twoSided) const {
        if (kGather <= 0.0 || photons.empty() || radius <= 0.0) return radius;
        const double slack = gatherSlack(fiberR);
        int hist[kAdaptBins] = {0};
        const double r2 = radius * radius;
        const double invR2 = 1.0 / r2;
        long long cand = 0;   // the candidates tested, as queryR counts them
        // The scan reads locals alone, as queryR's does: each match stores into hist.
        const Vec3 q = p, nq = n;
        const Vec3* __restrict pp = pos.data();
        const Photon* ph = photons.data();
        const int* cs = cellStart.data();
        MergeKey keys[27];
        mergeNeighbourKeys(lat, q.x, q.y, q.z, keys);
        for (const MergeKey& c : keys) {
            const int b = cs[c.bucket], e = cs[c.bucket + 1];
            cand += e - b;
            for (int k = b; k < e; ++k) {
                Vec3 d = q - pp[k];
                const double d2 = dot(d, d);
                if (d2 > r2) continue;
                bool farSide;
                if (!onGatherSurfaceSided(q, nq, pp[k], ph[k].n, d2, slack, twoSided, farSide)) continue;
                // Shell index: t in (2^-(i+1), 2^-i] -> i. ilogb(t) is the exponent e
                // with t = m*2^e, m in [1,2), so i = -e-1. d2 == 0 lands in the
                // innermost shell (ilogb(0) is FP_ILOGB0, clamped below).
                const double t = d2 * invR2;
                int i = (t > 0.0) ? -std::ilogb(t) - 1 : (kAdaptBins - 1);
                if (i < 0) i = 0;
                if (i >= kAdaptBins) i = kAdaptBins - 1;
                ++hist[i];
            }
        }
        bdpt::tlCandidates() += cand;
        // Largest i (smallest shell) whose cumulative population still reaches k.
        long long C = 0;
        for (int i = kAdaptBins - 1; i >= 0; --i) {
            C += hist[i];
            if ((double)C >= kGather) {
                // r_i^2 = r^2 * 2^-i; interpolate to exactly k photons inside it.
                double rq2 = r2 * std::ldexp(1.0, -i) * (kGather / (double)C);
                if (rq2 >= r2) return radius;
                const double rMin = radius * (1.0 / 256.0);
                double rq = std::sqrt(rq2);
                return (rq < rMin) ? rMin : rq;
            }
        }
        return radius;   // fewer than k photons in the whole disc — nothing to tighten
    }
};
