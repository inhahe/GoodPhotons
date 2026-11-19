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

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include "allocreport.h"    // ftalloc::resize: a table too large names itself

#ifdef __CUDACC__
#define MG_HD __host__ __device__ FT_UNIT_INLINE
#else
#define MG_HD FT_UNIT_INLINE
#endif

// The cell coordinate, along one axis, of a point at `p` in a lattice of cells `cell` wide from `lo`:
// floor((p - lo) / cell), held within +-2^30 so that a point far off the lattice (or a NaN) converts
// defined and its neighbours' coordinates do not overflow.
MG_HD int mergeCellCoord(double p, double lo, double cell) {
    const double f = floor((p - lo) / cell);
    return f > 1073741824.0 ? 1073741824 : (f >= -1073741824.0 ? (int)f : -1073741824);
}

// The block of cells a coordinate lies in, floor(i / 3).
MG_HD int mergeBlockOf(int i) {
    const int q = i / 3;
    return q * 3 > i ? q - 1 : q;
}

// The slot of block (bx, by, bz) in a table of mask + 1 slots: the coordinates' products with three
// large odd constants, xor-ed and passed through murmur3's 32-bit finaliser, so that neighbouring
// blocks scatter. 32-bit throughout: a GPU multiplies 64-bit integers in software.
MG_HD unsigned int mergeSlot(int bx, int by, int bz, unsigned int mask) {
    unsigned int h = (unsigned int)bx * 0x8DA6B343u ^ (unsigned int)by * 0xD8163841u ^ (unsigned int)bz * 0xCB1AB31Fu;
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h & mask;
}

// The bucket of cell (ix, iy, iz) in a table of 27 * (mask + 1) buckets (mergeBucketMask): 27 times
// its block's slot plus its place in the block.
MG_HD unsigned int mergeBucket(int ix, int iy, int iz, unsigned int mask) {
    const int bx = mergeBlockOf(ix), by = mergeBlockOf(iy), bz = mergeBlockOf(iz);
    return 27u * mergeSlot(bx, by, bz, mask) + (unsigned int)((ix - 3 * bx) + 3 * (iy - 3 * by) + 9 * (iz - 3 * bz));
}

// A lattice as the device's maps carry it (DPhotonMap, DSurfMap, DVcmGrid): its corner and cell
// width in double, as its points were binned -- a gather that places a site in another rounding of
// them takes the cell beside its own near a cell plane, and misses the points two cells away -- and
// its table's mask.
struct MergeLattice {
    double lo[3];
    double cell;
    unsigned int mask;
};

// The bucket of the cell holding (x, y, z) in lattice `L`: how a map bins a point.
MG_HD unsigned int mergeLatticeBucket(const MergeLattice& L, double x, double y, double z) {
    return mergeBucket(mergeCellCoord(x, L.lo[0], L.cell), mergeCellCoord(y, L.lo[1], L.cell),
                       mergeCellCoord(z, L.lo[2], L.cell), L.mask);
}

// The buckets of a table: 27 * (mask + 1).
inline size_t mergeBucketCount(unsigned int mask) { return 27 * ((size_t)mask + 1); }

// The mask of the table for `n` points: about two buckets a point, mask + 1 (the blocks' slots) a power
// of two from 64 to 2^26 (27 * 2^26 buckets index below 2^31). FTRACE_MERGE_TABLE=<k> holds mask + 1
// to at most k (a power of two; 1 leaves one block of 27 buckets that every block shares), for the
// checks that an exact query renders the same at any table size (tools/check_merge_table.py); read
// once per process.
inline unsigned int mergeBucketMask(size_t n) {
    static const unsigned long long cap = [] {
        const char* e = std::getenv("FTRACE_MERGE_TABLE");
        const long long k = e ? std::atoll(e) : 0;
        return k > 0 ? (unsigned long long)k : (1ull << 26);
    }();
    const unsigned long long want = (2ull * (unsigned long long)n + 26) / 27;
    unsigned long long t = 64;
    while (t < want && t < (1ull << 26)) t <<= 1;
    while (t > cap && t > 1) t >>= 1;
    return (unsigned int)(t - 1);
}

// The lattice of cells `cell` wide over points whose box's low corner is (mnx, mny, mnz), `n` of them:
// the corner half a cell below it, the table mergeBucketMask(n)'s. Every map's rule (VcmGrid, SurfMap,
// PhotonMap and the device's builds of theirs).
inline MergeLattice mergeLatticeOver(double mnx, double mny, double mnz, double cell, size_t n) {
    return MergeLattice{{mnx - cell * 0.5, mny - cell * 0.5, mnz - cell * 0.5}, cell, mergeBucketMask(n)};
}

// The counting sort of `n` points into lattice `L`: `cellStart` (mergeBucketCount(L.mask) + 1 entries,
// bucket b's points order[cellStart[b] .. cellStart[b + 1])) and `order` (the point indices,
// bucket-contiguous, each bucket's in index order). `posOf(i)` is point i's position (x, y, z);
// `what` and `flags` name the table, and what sizes the point count, to an out-of-memory report.
template <class PosOf>
inline void mergeBin(const MergeLattice& L, size_t n, PosOf&& posOf, std::vector<int>& cellStart,
                     std::vector<int>& order, const char* what, const char* flags) {
    const size_t nBuckets = mergeBucketCount(L.mask);
    cellStart.clear();
    ftalloc::resize(cellStart, nBuckets + 1, what, flags);   // zeroed
    std::vector<unsigned int> bucketOf(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& p = posOf(i);
        const unsigned int b = mergeLatticeBucket(L, p.x, p.y, p.z);
        bucketOf[i] = b;
        ++cellStart[(size_t)b + 1];
    }
    for (size_t b = 0; b < nBuckets; ++b) cellStart[b + 1] += cellStart[b];
    order.assign(n, 0);
    std::vector<int> cursor(cellStart.begin(), cellStart.end() - 1);
    for (size_t i = 0; i < n; ++i) order[(size_t)cursor[bucketOf[i]]++] = (int)i;
}

// fn(bucket) for each of the 27 cells around the cell holding (x, y, z) in lattice `L`, in walk order
// (dz outermost, dx innermost), the at most eight blocks they lie in hashed once each.
template <class F>
inline void mergeForEachBucket(const MergeLattice& L, double x, double y, double z, F&& fn) {
    const int c[3] = {mergeCellCoord(x, L.lo[0], L.cell), mergeCellCoord(y, L.lo[1], L.cell),
                      mergeCellCoord(z, L.lo[2], L.cell)};
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
    unsigned int slot[2][2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) slot[i][j][k] = mergeSlot(blk[0][k], blk[1][j], blk[2][i], L.mask);
    for (int dz = 0; dz < 3; ++dz)
        for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx)
                fn(27u * slot[sel[2][dz]][sel[1][dy]][sel[0][dx]] +
                   (unsigned int)(place[0][dx] + 3 * place[1][dy] + 9 * place[2][dz]));
}
