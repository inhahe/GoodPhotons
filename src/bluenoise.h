// bluenoise.h — blue-noise (Poisson-disk) point set, host + device.
//
//   patBlueNoise(x, y, z, r, out)  ->  out[0] = F1 (distance to the nearest
//                                      point of the set), out[1] = F2, out[2] =
//                                      that F1 point's random id in [0,1).
//
// WHAT THIS IS FOR, AND WHY `worley` IS NOT IT. Worley's feature points are a
// JITTERED LATTICE: exactly one point per cell, uniform inside it. Two points
// in adjacent cells can therefore be arbitrarily close (both jitter to the
// shared face), and elsewhere the lattice leaves gaps. Threshold F1 to draw
// spots and the clumping is immediately legible as "computer texture": pairs of
// freckles fused into peanuts, bald patches next to them. Real scattered
// features — freckles, pores, seeds, stomata, rain spatter, lichen — have a
// MINIMUM SEPARATION and are otherwise evenly spread. That is a blue-noise
// point set, and no amount of post-processing recovers one from a lattice.
//
// THE HARD PART. Classical Poisson-disk sampling (dart throwing, Bridson) is
// inherently SEQUENTIAL: whether a dart is kept depends on every dart accepted
// before it, so you cannot answer "what is the nearest point to p" without
// simulating the whole plane out to p. A shader needs the opposite: O(1) work
// at an arbitrary point of an unbounded domain, no bake, no global pass, and
// bit-identical answers on CPU and GPU.
//
// THE FIX is to replace the sequential order with a RANDOM PRIORITY and make
// acceptance a purely local predicate:
//
//     every cell holds one candidate (jittered position + a 32-bit rank);
//     a candidate is ACCEPTED iff no candidate within distance r outranks it.
//
// That is one round of Luby's maximal-independent-set algorithm on the conflict
// graph — equivalently, a Matern type-II hardcore thinning with a JITTERED
// LATTICE parent instead of the usual Poisson one. Because `precedes` below is a
// STRICT TOTAL ORDER (rank, then cell coordinates — so a rank collision cannot
// make two candidates mutually accept), two accepted points can never lie within
// r of each other: one of the pair would have seen the other and stood down. The
// minimum separation is therefore a THEOREM, not a statistic — `-checkbluenoise`
// §3 verifies it over millions of pairs, and §1 verifies acceptance against a
// brute force.
//
// THE DENSITY, and why there is exactly ONE candidate per cell. A candidate
// survives iff it outranks all N others inside its exclusion ball, which happens
// with probability 1/(N+1). E[N] is NOT the ball volume, because the candidate's
// own cell holds no competitor: the cell's self-overlap comes out of it, and
// with that overlap's kernel being the per-axis triangle (1-|w|) it integrates
// in closed form for r <= 1 (where the ball never leaves the +-1 block):
//
//     E[N] = INT_{|w|<r} [ 1 - PROD_a (1-|w_a|) ] dw
//          = 3/2 pi r^4  -  8/5 r^5  +  1/6 r^6            ( = 3.2791 at r = 1)
//
// the ball volume 4/3 pi r^3 having cancelled against the overlap's leading
// term. The measured accepted density at r = 1 is 0.2665 points per cell.
//
// Now the comparison that decides the design. Matern-II thinning of a POISSON
// parent of intensity L retains (1-exp(-L*V))/V per unit volume, rising to
// 1/V = 3/(4 pi) = 0.2387 as L -> inf. So one stratified candidate per cell does
// not merely approximate a dense candidate process — it BEATS the dense limit by
// 12%, because stratification has already removed the close candidate pairs that
// would otherwise consume the rank competition for nothing. Adding candidates
// per cell would make the set sparser, not denser. (§5 measures E[N] against the
// closed form, and asserts the ceiling really is beaten.)
//
// WHAT IT STILL COSTS: the set is independent but not MAXIMAL — there is room a
// sequential dart-thrower would have filled. Getting past that needs a second
// MIS round, and each round widens the dependency neighbourhood by r (round 2
// reads +-2 cells, round 3 +-3), so the cost grows cubically for a shrinking
// gain. One round, honestly documented, is the right point on that curve.
//
// The rejected alternative is a baked Poisson-disk tile (Wang tiles, Cohen et
// al. 2003). It is cheaper per query, but it REPEATS — and a repeat in a field
// of freckles is exactly the artifact the whole primitive exists to avoid — and
// it would need resident data on the device, which nothing else in the pattern
// VM does.
//
// THE `r` KNOB, and why the primitive is a strict generalisation. r is the
// exclusion radius in cell units, clamped to [0,1]:
//
//     r = 0    no conflicts, every candidate accepted  ==  the jittered lattice
//              (i.e. exactly Worley's point set, up to the hash), density 1
//     r = 1    maximum blue, density ~0.235, F1 in ~[0, 1.6]
//
// so r sweeps continuously from white-noise placement to blue-noise placement
// and the author can dial "how evenly spread" independently of "how big". The
// r <= 1 clamp is what keeps the acceptance neighbourhood EXACT at 3x3x3: a
// candidate in a cell two rings out is strictly more than 2-1 = 1 >= r away in
// one axis alone, so it can never conflict. (Distinct hash constants from
// worley.h and gabor.h, so the three noises do not correlate when layered.)
//
// COST. Two early-outs keep this near Worley's price rather than 100x it.
// (a) Per cell, a pure-arithmetic lower bound — the distance from the query to
// the nearest POINT OF THE CELL — is tested before any hashing, so the outer
// rings collapse to a few flops each. (b) Inside the acceptance test the RANK
// is compared before the POSITION is computed: rank is free (it is the base
// cell hash), position costs three more mixes, and half the neighbours are
// outranked and skipped for one mix. A rejected candidate usually exits after a
// handful of neighbours. `-checkbluenoise` §8 reports the measured cell-visit
// count against Worley's.
//
// F1 is a raw distance (cell size 1). Thresholding F1 draws discs; dividing by
// a value derived from `id` before thresholding varies each disc's size, which
// is what makes a freckle field read as freckles rather than as polka dots.
// F2-F1 is the crack network of the blue-noise Voronoi diagram — more even than
// Worley's, because the sites are.
#pragma once

#if defined(__CUDACC__)
  #ifndef PAT_BN_HD
  #define PAT_BN_HD __host__ __device__
  #endif
  // The query is by far the largest single op in the pattern VM (a ring search whose
  // body contains a second, nested 3x3x3 search). Inlined into dPatternEval it pushed
  // that function's register count over the BDPT kernels' budget and ptxas refused the
  // whole build, so on the device it stays OUT OF LINE: its frame is then paid only by
  // programs that actually call it, instead of by every pattern evaluation everywhere.
  #define PAT_BN_NOINLINE __noinline__
#else
  #ifndef PAT_BN_HD
  #define PAT_BN_HD
  #endif
  #define PAT_BN_NOINLINE
#endif

#include <math.h>

// Ring cap. Reaching ring R means a ball of radius R-1 around the query holds
// no accepted point; with density ~0.235 the expected count inside ring 8 is
// ~1150, so ring 3 is already unreachable in practice (§7 measures the true
// maximum over millions of queries and asserts it stays tiny). The cap exists
// so that a pathological input cannot loop unboundedly; outputs are clamped to
// it, so a capped query degrades to a large finite distance, never to 1e150.
#define PAT_BN_RINGCAP 8

// Hash salts. Distinct from worley.h's and gabor.h's so the point sets of the
// three primitives are independent when an author layers them.
#define PAT_BN_SX  0x27d4eb2fu
#define PAT_BN_SY  0x165667b1u
#define PAT_BN_SZ  0xd3a2646cu
#define PAT_BN_SID 0xfd7046c5u

// murmur3 fmix32: full-avalanche 32-bit finalizer.
PAT_BN_HD inline unsigned int patBNMix(unsigned int h) {
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Cell -> base hash. Doubles as the candidate's RANK, which is what makes the
// rank-before-position early-out free.
PAT_BN_HD inline unsigned int patBNCellHash(int cx, int cy, int cz) {
    unsigned int h = (unsigned int)cx * 2135587861u
                   ^ (unsigned int)cy * 805459861u
                   ^ (unsigned int)cz * 3266489917u;
    return patBNMix(h ^ 0x7feb352du);
}

// Candidate position: uniform in its cell.
PAT_BN_HD inline void patBNPos(unsigned int h0, int cx, int cy, int cz,
                               double& px, double& py, double& pz) {
    px = (double)cx + (double)patBNMix(h0 ^ PAT_BN_SX) * (1.0 / 4294967296.0);
    py = (double)cy + (double)patBNMix(h0 ^ PAT_BN_SY) * (1.0 / 4294967296.0);
    pz = (double)cz + (double)patBNMix(h0 ^ PAT_BN_SZ) * (1.0 / 4294967296.0);
}

// STRICT TOTAL ORDER on candidates: lower rank wins, ties broken by cell
// coordinates. Totality is what guarantees the minimum separation — under a
// merely partial order two candidates with colliding ranks (probability ~n/2^32,
// i.e. certain somewhere in an unbounded domain) could each fail to outrank the
// other and both be accepted while overlapping.
PAT_BN_HD inline bool patBNPrecedes(unsigned int ra, int ax, int ay, int az,
                                    unsigned int rb, int bx, int by, int bz) {
    if (ra != rb) return ra < rb;
    if (az != bz) return az < bz;
    if (ay != by) return ay < by;
    return ax < bx;
}

// Distance from a point to the nearest point of the unit cell based at c, per
// axis. Zero when the coordinate is already inside the slab.
PAT_BN_HD inline double patBNAxisGap(double v, int c) {
    const double lo = (double)c;
    if (v < lo) return lo - v;
    const double hi = lo + 1.0;
    return v > hi ? v - hi : 0.0;
}

// Is the candidate of cell (cx,cy,cz) — base hash h0, position p — accepted?
// Exact for r in [0,1]: a candidate two cells out is > 1 >= r away in one axis.
PAT_BN_HD PAT_BN_NOINLINE inline bool patBNAccept(int cx, int cy, int cz, unsigned int h0,
                                                  double px, double py, double pz, double r) {
    if (!(r > 0.0)) return true;               // r = 0: the jittered lattice
    const double r2 = r * r;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        const int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        // (a) geometric reject, no hashing: can this cell hold a conflict at all?
        const double ex = patBNAxisGap(px, nx);
        const double ey = patBNAxisGap(py, ny);
        const double ez = patBNAxisGap(pz, nz);
        if (ex * ex + ey * ey + ez * ez >= r2) continue;
        // (b) rank reject, one mix: outranked neighbours can never veto us.
        const unsigned int nh = patBNCellHash(nx, ny, nz);
        if (!patBNPrecedes(nh, nx, ny, nz, h0, cx, cy, cz)) continue;
        // (c) only now is the position worth three more mixes.
        double qx, qy, qz; patBNPos(nh, nx, ny, nz, qx, qy, qz);
        const double ddx = qx - px, ddy = qy - py, ddz = qz - pz;
        if (ddx * ddx + ddy * ddy + ddz * ddz < r2) return false;
    }
    return true;
}

// Blue-noise query. out[0] = F1, out[1] = F2, out[2] = id of the F1 point.
// `visited`, when non-null, receives the number of cells whose candidate was
// hashed (the cost metric §8 reports); pass null in the renderer.
PAT_BN_HD PAT_BN_NOINLINE inline void patBlueNoiseC(double x, double y, double z, double r,
                                                    double out[3], int* visited) {
    if (visited) *visited = 0;
    // Non-finite guard. Every comparison against a NaN is false, so without
    // this the ring loop would run to the cap and do ~5000 acceptance tests.
    if (!(x > -1e300 && x < 1e300) || !(y > -1e300 && y < 1e300) ||
        !(z > -1e300 && z < 1e300)) { out[0] = out[1] = out[2] = 0.0; return; }
    if (!(r > 0.0)) r = 0.0; else if (r > 1.0) r = 1.0;

    const int bx = (int)floor(x), by = (int)floor(y), bz = (int)floor(z);
    double f1s = 1e300, f2s = 1e300;           // squared; sqrt once at the end
    unsigned int idH = 0u;

    for (int R = 0; R <= PAT_BN_RINGCAP; ++R) {
        // Ring R's cells all lie >= R-1 from the query (it sits in ring 0's
        // cell), so once that cannot beat F2 no farther ring can either.
        if (R >= 1) { const double lb = (double)(R - 1); if (lb * lb >= f2s) break; }
        for (int dz = -R; dz <= R; ++dz)
        for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
            const int adx = dx < 0 ? -dx : dx, ady = dy < 0 ? -dy : dy,
                      adz = dz < 0 ? -dz : dz;
            int cheb = adx > ady ? adx : ady; if (adz > cheb) cheb = adz;
            if (cheb != R) continue;           // shell only
            const int cx = bx + dx, cy = by + dy, cz = bz + dz;
            // Per-cell lower bound, before any hashing: the nearest point this
            // cell could possibly hold. Kills most of every outer ring.
            const double ex = patBNAxisGap(x, cx);
            const double ey = patBNAxisGap(y, cy);
            const double ez = patBNAxisGap(z, cz);
            if (ex * ex + ey * ey + ez * ez >= f2s) continue;
            const unsigned int h0 = patBNCellHash(cx, cy, cz);
            if (visited) ++*visited;
            double px, py, pz; patBNPos(h0, cx, cy, cz, px, py, pz);
            const double ddx = x - px, ddy = y - py, ddz = z - pz;
            const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (d2 >= f2s) continue;           // cannot improve either slot
            if (!patBNAccept(cx, cy, cz, h0, px, py, pz, r)) continue;
            if (d2 < f1s) { f2s = f1s; f1s = d2; idH = patBNMix(h0 ^ PAT_BN_SID); }
            else          { f2s = d2; }
        }
    }
    const double capd = (double)PAT_BN_RINGCAP;
    const double cap2 = capd * capd;
    if (f1s > cap2) f1s = cap2;
    if (f2s > cap2) f2s = cap2;
    out[0] = sqrt(f1s);
    out[1] = sqrt(f2s);
    out[2] = (double)idH * (1.0 / 4294967296.0);
}

PAT_BN_HD PAT_BN_NOINLINE inline void patBlueNoise(double x, double y, double z, double r,
                                                   double out[3]) {
    patBlueNoiseC(x, y, z, r, out, (int*)0);
}
