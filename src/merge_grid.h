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

// The bucket of cell (ix, iy, iz) in a table of 27 * (mask + 1) buckets (mergeBucketMask): 27 times a
// SplitMix64-style mix of its block's coordinates, masked, plus its place in the block.
MG_HD unsigned int mergeBucket(int ix, int iy, int iz, unsigned int mask) {
    const int bx = mergeBlockOf(ix), by = mergeBlockOf(iy), bz = mergeBlockOf(iz);
    unsigned long long h = (unsigned long long)(unsigned int)bx * 0x9E3779B97F4A7C15ull;
    h ^= (unsigned long long)(unsigned int)by * 0xC2B2AE3D27D4EB4Full;
    h ^= (unsigned long long)(unsigned int)bz * 0x165667B19E3779F9ull;
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 27; h *= 0x94D049BB133111EBull;
    h ^= h >> 31;
    return 27u * ((unsigned int)h & mask) + (unsigned int)((ix - 3 * bx) + 3 * (iy - 3 * by) + 9 * (iz - 3 * bz));
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
