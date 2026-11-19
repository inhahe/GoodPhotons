// merge_grid.h -- the hashed lattice the merge maps bin their points into, on the host and the
// device: mode U's light vertices (VcmGrid, DVcmGrid), mode J's surface photons (SurfMap, DSurfMap)
// and the photon map of modes M and S (PhotonMap, DPhotonMap, SPPM's device grid). Cells are as
// wide as the merge radius, so a query reads the 3x3x3 cells around it, and the table of buckets is
// sized from the POINT COUNT, never from the box the points span: a dense lattice takes (L / r)^3
// cells over a box L wide, where points on surfaces fill only (L / r)^2 of them, so a fine radius
// over a large scene asked for gigabytes or coarsened its cells (and every merge with them); here a
// point costs about two ints of table however fine the cells are.
//
// The cells go in blocks of 3x3x3, a block's 27 cells in 27 consecutive buckets -- the block hashed,
// the cell's place in it (its coordinates mod 3) the low part -- so a query's 27 cells span at most
// eight blocks, two cache lines of table each, as a dense lattice keeps neighbours together. A
// bucket holds one cell of every block that hashes to its slot, and a query rejects the strays by
// the distance test it makes anyway. It must never read one bucket twice, or the points of two of
// its cells that share it would each count twice: any two cells of a 3x3x3 neighbourhood differ by 1
// or 2 along some axis, so their places in their blocks differ, and a query's 27 cells always fall
// in 27 different buckets. Within a bucket the points keep the order they were binned in, so a query
// meets each of its cells' points in the order a dense lattice would hand them over.
#pragma once
#include "unit_linkage.h"   // FT_UNIT_INLINE

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>
#include "allocreport.h"    // ftalloc::resize: a table too large names itself
#include "parallel.h"      // ft::parallelForAll, ft::workerCount: a large binning's threads

// Forced inline: left to itself MSVC keeps mergeCellKey out of line in mergeBin's count loop, a call
// a point, and the count of a large map's binning takes half as long again or more.
#ifdef __CUDACC__
#define MG_HD FT_UNIT_INLINE __host__ __device__ __forceinline__
#elif defined(_MSC_VER)
#define MG_HD FT_UNIT_INLINE __forceinline
#else
#define MG_HD FT_UNIT_INLINE __attribute__((always_inline))
#endif

// The cell coordinate, along one axis, of a point at `p` in a lattice from `lo` whose cells' width has
// the reciprocal `invCell`: floor((p - lo) * invCell) -- a multiply, every bin and query computing a
// point's cell the same way -- held within +-2^30 so that a point far off the lattice (or a NaN)
// converts defined and its neighbours' coordinates do not overflow.
MG_HD int mergeCellCoord(double p, double lo, double invCell) {
    const double f = floor((p - lo) * invCell);
    return f > 1073741824.0 ? 1073741824 : (f >= -1073741824.0 ? (int)f : -1073741824);
}

// The block of cells a coordinate lies in, floor(i / 3).
MG_HD int mergeBlockOf(int i) {
    const int q = i / 3;
    return q * 3 > i ? q - 1 : q;
}

// The mix of block (bx, by, bz): the coordinates' products with three large odd constants, xor-ed and
// passed through murmur3's 32-bit finaliser, so that neighbouring blocks scatter. Its high half,
// scaled to the table's slots ((h * slots) >> 32: any count of slots, two buckets a point exactly),
// is the block's slot, its low byte the block's tag. 32-bit throughout: a GPU multiplies 64-bit
// integers in software (the scaling's 32 x 32 -> 64 product it does in one instruction).
MG_HD unsigned int mergeMix(int bx, int by, int bz) {
    unsigned int h = (unsigned int)bx * 0x8DA6B343u ^ (unsigned int)by * 0xD8163841u ^ (unsigned int)bz * 0xCB1AB31Fu;
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

// A cell's bucket and its block's TAG. A bucket holds the points of one cell of every block hashed
// to its slot -- its own cell's and about 13 strays a site at two buckets a point -- so where a map's
// sites are sparse, the strays outnumber what they hold: modes U and J (VcmGrid, SurfMap and their
// device twins) keep each point's block's tag (the mix's low byte; the slot is its high bits, scaled),
// one byte a point in bucket order, and a query takes only the entries tagged as its own cell's block
// (mergeForEachEntry): of the strays it reads a byte each, and only those of a block whose tag matches
// too (one in 256) go on to its distance test. A point of the cell always carries its tag, so none is
// missed, and the counts those merges are gated and paced by count the tagged entries. The photon map
// keeps none: a gather there weighs hundreds of photons, beside which the strays are a percent or
// two, and its hundred-million-photon builds would pay a third random scatter for them.
struct MergeKey {
    unsigned int bucket;
    unsigned char tag;
};

// The slot, among `slots`, of a block whose mix is `h`.
MG_HD unsigned int mergeSlotOf(unsigned int h, unsigned int slots) {
    return (unsigned int)(((unsigned long long)h * slots) >> 32);
}

// The key of cell (ix, iy, iz) in a table of 27 * slots buckets (mergeSlotCount): the bucket 27 times
// its block's slot plus its place in the block, the tag its block's.
MG_HD MergeKey mergeCellKey(int ix, int iy, int iz, unsigned int slots) {
    const int bx = mergeBlockOf(ix), by = mergeBlockOf(iy), bz = mergeBlockOf(iz);
    const unsigned int h = mergeMix(bx, by, bz);
    return MergeKey{27u * mergeSlotOf(h, slots) + (unsigned int)((ix - 3 * bx) + 3 * (iy - 3 * by) + 9 * (iz - 3 * bz)),
                    (unsigned char)h};
}

// A lattice as the device's maps carry it (DPhotonMap, DSurfMap, DVcmGrid): its corner and cell
// width in double, as its points were binned -- a gather that places a site in another rounding of
// them takes the cell beside its own near a cell plane, and misses the points two cells away -- and
// its table's slots.
struct MergeLattice {
    double lo[3];
    double cell, invCell;   // the cells' width, and its reciprocal, by which a point's cell is found
    unsigned int slots;     // the table's slots, 27 buckets each
};

// The lattice from corner (lx, ly, lz) of cells `cell` wide, its table `slots` slots.
MG_HD MergeLattice mergeLatticeAt(double lx, double ly, double lz, double cell, unsigned int slots) {
    return MergeLattice{{lx, ly, lz}, cell, 1.0 / cell, slots};
}

// The key of the cell holding (x, y, z) in lattice `L`: how a map bins a point, and its tag.
MG_HD MergeKey mergeLatticeKey(const MergeLattice& L, double x, double y, double z) {
    return mergeCellKey(mergeCellCoord(x, L.lo[0], L.invCell), mergeCellCoord(y, L.lo[1], L.invCell),
                        mergeCellCoord(z, L.lo[2], L.invCell), L.slots);
}

// The bucket of the cell holding (x, y, z) in lattice `L`.
MG_HD unsigned int mergeLatticeBucket(const MergeLattice& L, double x, double y, double z) {
    return mergeLatticeKey(L, x, y, z).bucket;
}

// The buckets of a table of `slots` slots.
inline size_t mergeBucketCount(unsigned int slots) { return 27 * (size_t)slots; }

// The slots of the table for `n` points: two buckets a point, ceil(2n / 27), from 64 (or 1 for an
// empty map) to 2^26 (27 * 2^26 buckets index below 2^31). FTRACE_MERGE_TABLE=<k> holds them to at most
// k (1 leaves one block of 27 buckets that every block shares), for the checks that an exact query
// renders the same at any table size (tools/check_merge_table.py); read once per process.
inline unsigned int mergeSlotCount(size_t n) {
    static const unsigned long long cap = [] {
        const char* e = std::getenv("FTRACE_MERGE_TABLE");
        const long long k = e ? std::atoll(e) : 0;
        return k > 0 ? (unsigned long long)k : (1ull << 26);
    }();
    unsigned long long t = n ? (2ull * (unsigned long long)n + 26) / 27 : 1;
    if (n && t < 64) t = 64;
    if (t > (1ull << 26)) t = 1ull << 26;
    if (t > cap) t = cap;
    return (unsigned int)t;
}

// The lattice of cells `cell` wide over points whose box's low corner is (mnx, mny, mnz), `n` of them:
// the corner half a cell below it, the table mergeSlotCount(n)'s. Every map's rule (VcmGrid, SurfMap,
// PhotonMap and the device's builds of theirs).
inline MergeLattice mergeLatticeOver(double mnx, double mny, double mnz, double cell, size_t n) {
    return mergeLatticeAt(mnx - cell * 0.5, mny - cell * 0.5, mnz - cell * 0.5, cell, mergeSlotCount(n));
}

// From this many points a binning (mergeBin) runs on the threads the program may use (ft::workerCount);
// below it the threads' start would cost more than they save -- mode U's CPU passes bin a few thousand
// light vertices each.
constexpr size_t kMergeBinThreadsFrom = (size_t)1 << 20;

// The counting sort of `n` points into lattice `L`: `cellStart` (a vector of int, mergeBucketCount(L.slots)
// + 1 entries, bucket b holding positions cellStart[b] .. cellStart[b + 1]), `place(i, k)` for each point,
// to put point i at position k (bucket-contiguous, each bucket's points in index order), and, where `tag`
// is given, (*tag)[k] that point's tag (MergeKey). `posOf(i)` is point i's position (x, y, z); `what`
// and `flags` name the table, and what sizes the point count, to an out-of-memory report. From
// kMergeBinThreadsFrom points it runs on ft::workerCount() threads, in the same order to the point:
// `posOf` and `place` are then called from several threads at once, `place` for distinct k.
template <class PosOf, class Place, class CellStart>
inline void mergeBin(const MergeLattice& L, size_t n, PosOf&& posOf, Place&& place, CellStart& cellStart,
                     std::vector<unsigned char>* tag, const char* what, const char* flags) {
    const size_t nBuckets = mergeBucketCount(L.slots);
    const unsigned T = (n >= kMergeBinThreadsFrom) ? std::min(ft::workerCount(), 64u) : 1u;
    cellStart.clear();
    // On one thread one entry over, dropped at the end: the table is its own cursor (below), its counts
    // zeroed first. On several every entry is written (4), so a table that leaves a resize's elements
    // unwritten (ftalloc::BulkVec, the photon map's) is not zero-filled for nothing.
    ftalloc::resize(cellStart, nBuckets + (T > 1 ? 1 : 2), what, flags);
    if (T == 1) std::fill(cellStart.begin(), cellStart.end(), 0);
    // Each point's key, computed once, in scratch of four bytes a point (five with tags) left unfilled:
    // a map of a hundred million photons bins twice in buildAuto.
    std::unique_ptr<unsigned int[]> bucketOf;
    std::unique_ptr<unsigned char[]> tagOf;
    try {
        bucketOf.reset(new unsigned int[n]);
        if (tag) tagOf.reset(new unsigned char[n]);
    } catch (const std::bad_alloc&) {
        ftalloc::reportOom("a binning's key of each point", n, tag ? 5 : 4, flags);
    }
    int* const cs = cellStart.data();
    if (tag) ftalloc::resize(*tag, n, "a binning's tags", flags);
    if (T == 1) {
        // Bucket b's points counted at cs[b + 2], so that after the prefix sum cs[b + 1] is where bucket b
        // begins: the scatter takes it as b's next position, which leaves it where bucket b + 1 begins --
        // every start in its place, with no copy of the table to scatter through.
        for (size_t i = 0; i < n; ++i) {
            const auto& p = posOf(i);
            const MergeKey key = mergeLatticeKey(L, p.x, p.y, p.z);
            bucketOf[i] = key.bucket;
            if (tag) tagOf[i] = key.tag;
            ++cs[(size_t)key.bucket + 2];
        }
        for (size_t b = 2; b < nBuckets + 2; ++b) cs[b] += cs[b - 1];
        for (size_t i = 0; i < n; ++i) {
            const size_t k = (size_t)cs[(size_t)bucketOf[i] + 1]++;
            if (tag) (*tag)[k] = tagOf[i];
            place(i, k);
        }
        cellStart.pop_back();
        return;
    }
    // On T threads, in the one-thread sort's order exactly. The buckets go in P contiguous ranges
    // ("parts"); thread t takes points [n t / T, n (t + 1) / T). (1) Each thread computes its points'
    // keys and counts them by part. (2) The parts' starts, and each thread's place within each part, by
    // a prefix over (part, thread). (3) Each thread files its points (index and bucket) into their
    // parts in index order, so a part's list runs thread by thread, which is index order. (4) Each
    // part, by one thread, counts its list by bucket, writes its buckets' starts and gives its points
    // their positions in list order -- a bucket's points in index order, as the one-thread sort leaves
    // them -- in the key scratch, which (3) is done with. (5) Each thread places its points there.
    // (3) and (4) read their lists in order and write four bytes a point; only (5) moves the points,
    // reading them in order.
    const size_t P = (size_t)T * 64;
    // A bucket's part by a multiply, not a division a point (twice a point, in (1) and (3)): any
    // non-decreasing map of buckets to parts keeps the order, and each part's first bucket is found
    // from that same map, by bisection, so the two agree whatever the rounding.
    const double partScale = (double)P / (double)nBuckets;
    auto partOf = [&](unsigned int b) {
        const size_t p = (size_t)((double)b * partScale);
        return p < P ? p : P - 1;
    };
    std::vector<size_t> partFirst(P + 1);   // part p's buckets: [partFirst[p], partFirst[p + 1])
    for (size_t p = 0; p <= P; ++p) {
        size_t lo = 0, hi = nBuckets;       // the first bucket whose part is p or later
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (partOf((unsigned int)mid) >= p) hi = mid; else lo = mid + 1;
        }
        partFirst[p] = lo;
    }
    auto partLo = [&](size_t p) { return partFirst[p]; };
    std::vector<size_t> cnt((size_t)T * P, 0), pStart(P + 1, 0);
    struct Filed { unsigned int i, bucket; };
    std::unique_ptr<Filed[]> list;
    try { list.reset(new Filed[n]); }
    catch (const std::bad_alloc&) { ftalloc::reportOom("a binning's list of the points", n, sizeof(Filed), flags); }
    ft::parallelForAll(T, 1, [&](size_t t) {                                                       // (1)
        size_t* c = cnt.data() + (size_t)t * P;
        for (size_t i = n * t / T, e = n * (t + 1) / T; i < e; ++i) {
            const auto& p = posOf(i);
            const MergeKey key = mergeLatticeKey(L, p.x, p.y, p.z);
            bucketOf[i] = key.bucket;
            if (tag) tagOf[i] = key.tag;
            ++c[partOf(key.bucket)];
        }
    });
    for (size_t p = 0, at = 0; p < P; ++p) {                                                   // (2)
        pStart[p] = at;
        for (unsigned t = 0; t < T; ++t) {
            const size_t c = cnt[(size_t)t * P + p];
            cnt[(size_t)t * P + p] = at;
            at += c;
        }
        pStart[p + 1] = at;
    }
    ft::parallelForAll(T, 1, [&](size_t t) {                                                       // (3)
        size_t* at = cnt.data() + (size_t)t * P;
        for (size_t i = n * t / T, e = n * (t + 1) / T; i < e; ++i) {
            const unsigned int b = bucketOf[i];
            list[at[partOf(b)]++] = Filed{(unsigned int)i, b};
        }
    });
    unsigned int* const dest = bucketOf.get();   // a point's position, once (4) has given it one
    std::atomic<size_t> nextPart{0};
    ft::parallelForAll(T, 1, [&](size_t) {                                                         // (4)
        std::vector<int> local;
        for (size_t p; (p = nextPart.fetch_add(1, std::memory_order_relaxed)) < P;) {
            const size_t b0 = partLo(p), b1 = partLo(p + 1);
            local.assign(b1 - b0 + 1, 0);
            const Filed* li = list.get() + pStart[p];
            const size_t m = pStart[p + 1] - pStart[p];
            for (size_t j = 0; j < m; ++j) ++local[li[j].bucket - b0 + 1];
            int run = (int)pStart[p];
            for (size_t b = b0; b < b1; ++b) { run += local[b - b0]; local[b - b0] = run; cs[b] = run; }
            // local[b - b0] is now bucket b's start: the next position of its points.
            for (size_t j = 0; j < m; ++j) dest[li[j].i] = (unsigned int)local[li[j].bucket - b0]++;
        }
    });
    ft::parallelForAll(T, 1, [&](size_t t) {                                                       // (5)
        for (size_t i = n * t / T, e = n * (t + 1) / T; i < e; ++i) {
            const size_t k = dest[i];
            if (tag) (*tag)[k] = tagOf[i];
            place(i, k);
        }
    });
    cs[nBuckets] = (int)n;
}

// The keys of the 27 cells around the cell holding (x, y, z) in lattice `L`, into `keys` in walk order
// (dz outermost, dx innermost), the at most eight blocks they lie in hashed once each: the buckets a
// host query reads, which it then scans in a loop of its own, from its own locals. (The device's walk,
// DGridCell, takes a cell at a time: a lane there advances by a runtime index, which would put an
// array like this one in local memory.)
inline void mergeNeighbourKeys(const MergeLattice& L, double x, double y, double z, MergeKey keys[27]) {
    const int c[3] = {mergeCellCoord(x, L.lo[0], L.invCell), mergeCellCoord(y, L.lo[1], L.invCell),
                      mergeCellCoord(z, L.lo[2], L.invCell)};
    // Per axis and offset -1, 0, 1: which of the axis's (at most two) blocks the cell is in, and its
    // place in that block.
    int sel[3][3], place[3][3], blk[3][2];
    for (int a = 0; a < 3; ++a) {
        blk[a][0] = mergeBlockOf(c[a] - 1);
        blk[a][1] = mergeBlockOf(c[a] + 1);
        for (int k = 0; k < 3; ++k) {
            const int b = mergeBlockOf(c[a] + k - 1);
            sel[a][k] = b == blk[a][0] ? 0 : 1;
            place[a][k] = c[a] + k - 1 - 3 * b;
        }
    }
    unsigned int mix[2][2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) mix[i][j][k] = mergeMix(blk[0][k], blk[1][j], blk[2][i]);
    int n = 0;
    for (int dz = 0; dz < 3; ++dz)
        for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx) {
                const unsigned int h = mix[sel[2][dz]][sel[1][dy]][sel[0][dx]];
                keys[n++] = MergeKey{27u * mergeSlotOf(h, L.slots) +
                                         (unsigned int)(place[0][dx] + 3 * place[1][dy] + 9 * place[2][dz]),
                                     (unsigned char)h};
            }
}

// fn(k) for each entry k of the 27 cells' buckets around (x, y, z) in lattice `L` that is its own
// cell's point -- tagged as its cell's block (`tag`, in bucket order; MergeKey) -- in walk order:
// what a tagged map's query weighs (VcmGrid, SurfMap).
template <class F>
inline void mergeForEachEntry(const MergeLattice& L, const int* cellStart, const unsigned char* tag,
                              double x, double y, double z, F&& fn) {
    MergeKey keys[27];
    mergeNeighbourKeys(L, x, y, z, keys);
    for (const MergeKey& c : keys) {
        const int e = cellStart[c.bucket + 1];
        for (int k = cellStart[c.bucket]; k < e; ++k)
            if (tag[k] == c.tag) fn(k);
    }
}
