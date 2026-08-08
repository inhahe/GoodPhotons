// worley.h — 3-D cellular (Worley / Voronoi) noise, host + device.
//
// One feature point per integer lattice cell, jittered uniformly inside the
// cell by a deterministic integer hash (murmur3 finalizer — full avalanche, so
// neighbouring cells are uncorrelated and CPU/GPU agree bit-for-bit). A query
// returns the distances to the nearest (F1) and second-nearest (F2) feature
// points under a selectable metric, plus a per-cell random id in [0,1) taken
// from the F1 cell's hash (for per-cell colour/roughness randomisation).
//
//   metric 0 = Euclidean   (round organic cells — stone, cobble, cells)
//   metric 1 = Manhattan   (diamond / pyramid facets)
//   metric 2 = Chebyshev   (square / techy panels)
//
// F1 renders as bubbly cells, F2-F1 as the crack network between cells
// (cracked mud, crocodile skin), id as flat per-cell randomisation.
// Distances are raw (cell size 1): F1 is ~[0, 1.1], mean ~0.65 (Euclidean).
//
// EXACTNESS. The common 3x3x3-neighbourhood Worley is subtly wrong: the query
// point sits anywhere in its cell, so the true F1 can be up to sqrt(3) away
// while an unsearched cell two rings out can hold a point at distance 1. We
// search concentric Chebyshev rings and stop only when the nearest possible
// point of the next ring (ring r's cells are all >= r-1 away, in ALL three
// metrics, since each metric >= the Chebyshev point distance) cannot beat the
// current F2 — so F1/F2 are mathematically exact, at the average cost of the
// same 27 cells the approximate version reads. `-checkworley` §1 compares
// against a fixed 13^3 brute force with no early-out.
//
// The ring cap 8 is unreachable math (worst-case F2 after ring 1 is < 6, the
// Manhattan diameter of the 3x3x3 block, and 8-1 > 6) — it only guards NaN
// inputs, whose comparisons are all false, from looping forever.
#pragma once

#if defined(__CUDACC__)
  #ifndef WORLEY_HD
  #define WORLEY_HD __host__ __device__
  #endif
#else
  #ifndef WORLEY_HD
  #define WORLEY_HD
  #endif
#endif

#include <math.h>

// murmur3 fmix32: full-avalanche 32-bit finalizer.
WORLEY_HD inline unsigned int patWorleyMix(unsigned int h) {
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Cell -> base hash. The three coordinates are combined with large odd
// constants before the finalizer so permuted cells don't collide.
WORLEY_HD inline unsigned int patWorleyCellHash(int cx, int cy, int cz) {
    unsigned int h = (unsigned int)cx * 374761393u
                   ^ (unsigned int)cy * 668265263u
                   ^ (unsigned int)cz * 2246822519u;
    return patWorleyMix(h);
}

// Worley query at (x, y, z) under `metric` (0 Euclid / 1 Manhattan / 2 Chebyshev).
// out[0] = F1, out[1] = F2, out[2] = id of the F1 cell in [0,1).
WORLEY_HD inline void patWorley(double x, double y, double z, int metric,
                                double out[3]) {
    const double fx = floor(x), fy = floor(y), fz = floor(z);
    const int bx = (int)fx, by = (int)fy, bz = (int)fz;
    double f1 = 1e300, f2 = 1e300;
    unsigned int idHash = 0u;
    for (int r = 0; r <= 8; ++r) {
        // Ring r's cells all lie >= (r-1) away in every supported metric
        // (each metric >= Chebyshev distance, and the query is inside the
        // centre cell). Once that bound can't beat F2, no farther ring can.
        if ((double)(r - 1) >= f2) break;
        for (int dz = -r; dz <= r; ++dz)
        for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            // shell only: skip interior cells already visited by smaller r
            int adx = dx < 0 ? -dx : dx, ady = dy < 0 ? -dy : dy,
                adz = dz < 0 ? -dz : dz;
            int cheb = adx > ady ? adx : ady; if (adz > cheb) cheb = adz;
            if (cheb != r) continue;
            const int cx = bx + dx, cy = by + dy, cz = bz + dz;
            const unsigned int h1 = patWorleyCellHash(cx, cy, cz);
            const unsigned int h2 = patWorleyMix(h1 + 0x9e3779b9u);
            const unsigned int h3 = patWorleyMix(h2 + 0x9e3779b9u);
            // feature point, uniform in the cell
            const double px = (double)cx + (double)h1 * (1.0 / 4294967296.0);
            const double py = (double)cy + (double)h2 * (1.0 / 4294967296.0);
            const double pz = (double)cz + (double)h3 * (1.0 / 4294967296.0);
            const double ddx = x - px, ddy = y - py, ddz = z - pz;
            double d;
            if (metric == 1) {          // Manhattan
                d = fabs(ddx) + fabs(ddy) + fabs(ddz);
            } else if (metric == 2) {   // Chebyshev
                double ax = fabs(ddx), ay = fabs(ddy), az = fabs(ddz);
                d = ax > ay ? ax : ay; if (az > d) d = az;
            } else {                    // Euclidean
                d = sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
            }
            if (d < f1) {
                f2 = f1;
                f1 = d;
                idHash = patWorleyMix(h3 + 0x9e3779b9u);
            } else if (d < f2) {
                f2 = d;
            }
        }
    }
    out[0] = f1;
    out[1] = f2;
    out[2] = (double)idHash * (1.0 / 4294967296.0);   // [0,1)
}
