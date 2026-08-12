// fur_grid.h — voxelised fiber density + mean-orientation grid.  (TODO.md §P2)
//
// WHY THIS EXISTS.  Two separate problems in this repo want the same data structure, and
// both of them are "answer a question about a coat of hair without touching the strands":
//
//   1. Dual scattering (§P3 stage 4, `-dual-scatter`) needs, per shadow ray, only the
//      NUMBER of fibers crossed and their inclinations — never their identities.  It gets
//      those today by riding `Scene::walkFibers`, which visits every primitive overlapping
//      the segment.  That is exact, but it cannot early-out (a fiber does not block, it
//      accumulates), so on an absorbing coat it made dual scattering 1.6x SLOWER than the
//      brute-force path tracing it replaces.  This is Zinke et al. 2008 §4.1.2 — the
//      density-grid alternative to their §4.1.1 ray shooting.
//   2. The aggregate-BSDF level of detail (§P2): past some distance individual fibers must
//      give way to a volumetric medium, and a medium needs a density field.
//
// WHAT A CELL STORES.  A fiber of radius r, length l and unit tangent t presents a
// cross-section of `2 r l sin(angle(d,t))` = `2 r l sqrt(1 - (d.t)^2)` to direction d.  So
// the exact directional extinction of a cell of volume V is
//
//      sigma_t(d) = (1/V) SUM_i 2 r_i l_i sqrt(1 - (d.t_i)^2)
//
// which would need every strand.  Pull the square root outside the sum (Jensen) and it
// collapses onto two quantities that DO aggregate:
//
//      c = (2/V) SUM_i r_i l_i                         a scalar, 1/length
//      T = SUM_i r_i l_i t_i t_i^T / SUM_i r_i l_i     a normalised symmetric 3x3
//      sigma_t(d) ~= c * sqrt(1 - d^T T d)
//
// because `d^T T d` is exactly the density-weighted mean of `(d.t)^2`.  This is EXACT when
// the cell's tangents are aligned (T is then a rank-1 projector and the sqrt is a constant
// that factors out), and worst at perfect isotropy, where it gives sqrt(2/3) = 0.8165
// against the true <sin> = pi/4 = 0.7854 — a 4.0% over-estimate of extinction, i.e. the
// error is bounded and one-signed over the whole space of grooms.  Real fur is combed, so
// cells sit much nearer the exact end than the 4% end.
//
// The orientation tensor is also what makes this useful to a fiber BCSDF and not just to a
// density tracker: `d^T T d` is `<sin^2 theta_i>` in Marschner's frame directly, since that
// frame's longitudinal axis IS the tangent (`hairShadowCross` computes `sin theta` as
// `dot(w, t)`).  So one quadratic form gives both the extinction and the inclination to
// look the dual-scattering tables up at.
//
// THE SIGN OF THE TANGENT is not in `T` at all: `t t^T == (-t)(-t)^T`, so the second moment
// cannot tell a strand from its reverse, and `sqrt(d^T T d)` is `|sin theta|`.  Extinction
// does not care — a fiber blocks the same either way — so `sigma_t`, `march` and the
// dual-scattering read are all correctly sign-blind, and marginalise over the two signs
// rather than invent one (see `furAvgSigned`).  SCATTERING does care: the cuticle tilt
// `alpha` tips the R and TRT lobes toward the root, and reversing the tangent tips them the
// other way, which moved the aggregate far tier's response by 27% against the explicit
// population it stands for even for a perfectly parallel cell (`-checkfurvol` §6).  So each
// cell also carries the FIRST moment, `v = sum(r l t) / sum(r l)`, packed into the 4 bytes
// that `tzz` used to occupy — `tzz` is redundant because `T` is unit-trace by construction,
// so `tzz == 1 - txx - tyy` exactly.  `|v|` is the sign coherence: 1 for a combed cell, 0
// for a groom with no preferred direction, and the fraction of mass pointing along `v` is
// `(1 + |v|) / 2` exactly for a two-delta population, which is the rule `FurODF::sample`
// uses.  The grid therefore stores a first AND a second moment for the same 32 bytes.
//
// COST: 32 bytes per cell (see FurCell), so a 64^3 grid is 8 MB and a 128^3 grid 64 MB,
// against ~1.3 GB for the `CurveSeg` pool of a real groom.  The build transiently needs
// another 12 bytes per cell for the mean-tangent accumulator, freed before it returns.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "linalg.h"
#include "curve.h"

// ---------------------------------------------------------------------------------------
// Octahedral direction + coherence in one word: 12 bits per octahedral coordinate (about
// 0.05 degrees, far finer than a cell's tangent spread) and 8 bits of |v| in [0,1].  Zero
// means "no preferred direction", which is what a default-constructed cell must decode to.
inline uint32_t furPackMean(const Vec3& v, double coh) {
    const double l1 = std::fabs(v.x) + std::fabs(v.y) + std::fabs(v.z);
    if (!(l1 > 0.0) || !(coh > 0.0)) return 0;
    double x = v.x / l1, y = v.y / l1;
    if (v.z < 0.0) {
        const double ax = std::fabs(x), ay = std::fabs(y);
        const double nx = (1.0 - ay) * (x >= 0.0 ? 1.0 : -1.0);
        const double ny = (1.0 - ax) * (y >= 0.0 ? 1.0 : -1.0);
        x = nx; y = ny;
    }
    const uint32_t qx = (uint32_t)std::lround(std::min(1.0, std::max(0.0, x * 0.5 + 0.5)) * 4095.0);
    const uint32_t qy = (uint32_t)std::lround(std::min(1.0, std::max(0.0, y * 0.5 + 0.5)) * 4095.0);
    uint32_t qc = (uint32_t)std::lround(std::min(1.0, coh) * 255.0);
    if (qc == 0) qc = 1;                     // a nonzero |v| must not decode as "unoriented"
    return (qc << 24) | (qy << 12) | qx;
}
inline double furMeanCoherence(uint32_t p) { return (double)(p >> 24) * (1.0 / 255.0); }
inline Vec3 furMeanDir(uint32_t p) {
    if (!p) return Vec3{0, 0, 0};
    double x = (double)(p & 0xFFFu) * (1.0 / 4095.0) * 2.0 - 1.0;
    double y = (double)((p >> 12) & 0xFFFu) * (1.0 / 4095.0) * 2.0 - 1.0;
    double z = 1.0 - std::fabs(x) - std::fabs(y);
    if (z < 0.0) {
        const double ax = std::fabs(x), ay = std::fabs(y);
        const double nx = (1.0 - ay) * (x >= 0.0 ? 1.0 : -1.0);
        const double ny = (1.0 - ax) * (y >= 0.0 ? 1.0 : -1.0);
        x = nx; y = ny;
    }
    return normalize(Vec3{x, y, z});
}

// One voxel.  Kept to 32 bytes so a fine grid stays in a size worth having: `c` is the
// extinction scale, the five floats are the upper triangle of the normalised symmetric
// orientation tensor minus its redundant `tzz` (unit trace: `tzz == 1 - txx - tyy`), `mdir`
// is the packed mean tangent that buys back the sign the tensor cannot hold, and `matId` is
// the material with the most fiber mass in the cell (a coat is authored per-material, so
// cells are overwhelmingly pure; the dual-scattering tables are per-material and something
// must be picked).
struct FurCell {
    float    c   = 0.0f;                // (2/V) * sum(r*l)   [1/length]
    float    txx = 0.0f, tyy = 0.0f;    // tzz is NOT stored — see furTzz()
    float    txy = 0.0f, txz = 0.0f, tyz = 0.0f;
    uint32_t mdir  = 0;                 // 12:12 octahedral mean tangent + 8-bit coherence
    int32_t  matId = -1;
};
static_assert(sizeof(FurCell) == 32, "FurCell is meant to be 32 bytes");

// The component the unit trace makes redundant.  Clamped at 0 because float accumulation of
// 10^5 deposits can put the recovered value a few ulp negative in a cell whose tangents all
// lie in the xy plane.
inline double furTzz(const FurCell& fc) {
    return std::max(0.0, 1.0 - (double)fc.txx - (double)fc.tyy);
}

// What a march through the grid measured.
struct FurMarch {
    double tau   = 0.0;   // optical depth == EXPECTED NUMBER OF FIBER CROSSINGS (see below)
    double sin2  = 0.0;   // tau-weighted mean of sin^2(theta) at the crossings
    int    matId = -1;    // dominant material along the ray, -1 if the ray saw no fiber
    bool   hit   = false; // did the ray overlap the grid's occupied region at all
};

// The grid itself.  Build once per scene (fibers do not move within a frame), query from
// any thread — everything after `build` is const.
struct FurGrid {
    Vec3  lo{0, 0, 0}, hi{0, 0, 0};     // world AABB of the fiber population
    int   nx = 0, ny = 0, nz = 0;
    Vec3  cell{1, 1, 1}, invCell{1, 1, 1};
    double cellVol = 1.0;
    std::vector<FurCell> cells;
    // Diagnostics / self-test hooks.
    double totalRL   = 0.0;             // sum over all fibers of r*l, as deposited
    double totalL    = 0.0;             // sum over all fibers of l, as deposited
    int    occupied  = 0;               // cells with any fiber mass
    bool   valid     = false;

    int index(int ix, int iy, int iz) const { return (iz * ny + iy) * nx + ix; }

    // Length-weighted mean fiber RADIUS, world units.  `totalRL/totalL`, not a plain average
    // over segments, because a coat's strands are tessellated into wildly different segment
    // counts -- a curly guard hair into many short pieces, a straight undercoat fiber into a
    // few long ones -- so an unweighted mean would be a mean over TESSELLATION rather than
    // over fur.  This is the scale `-fur-lod` measures the pixel footprint against: it is the
    // width of the silhouette detail the aggregate tier is throwing away.
    double meanRadius() const { return totalL > 0.0 ? totalRL / totalL : 0.0; }

    // Directional extinction in a cell, 1/length.  `d` must be unit.
    static double sigmaT(const FurCell& fc, const Vec3& d) {
        if (fc.c <= 0.0f) return 0.0;
        const double s2 = furSin2(fc, d);
        return fc.c * std::sqrt(std::max(0.0, 1.0 - s2));
    }
    // `d^T T d` — the density-weighted mean of `cos^2(angle to tangent)`, which in
    // Marschner's fiber frame is `sin^2(theta)`.  Clamped: T is PSD and unit-trace by
    // construction, so the form lies in [0,1] mathematically, but float accumulation of
    // 10^5 deposits can push it a few ulp outside.
    static double furSin2(const FurCell& fc, const Vec3& d) {
        const double v = d.x * d.x * fc.txx + d.y * d.y * fc.tyy + d.z * d.z * furTzz(fc)
                       + 2.0 * (d.x * d.y * fc.txy + d.x * d.z * fc.txz + d.y * d.z * fc.tyz);
        return std::min(1.0, std::max(0.0, v));
    }

    // -----------------------------------------------------------------------------------
    // BUILD.  `segs` is the scene's flat CurveSeg pool; `isFiber(matId)` selects which of
    // them are hair (grass and wire are curves too, and have no business in a coat's
    // density field).  `targetCells` is a budget, not a resolution: the axis counts are
    // chosen proportional to the AABB extents so cells come out roughly cubic, which is
    // what keeps the DDA step count uniform and the tensor meaningful.
    //
    // Deposition is by SUB-SAMPLING each round cone along its length at a fraction of a
    // cell, depositing each sub-segment's whole `r*dl` into the cell containing its
    // midpoint.  Exact clipping of a capsule against a voxel lattice is a great deal of
    // code for a quantity that is already a mean-field approximation; point deposition of
    // sub-cell pieces converges to the same field and — the property actually worth having
    // — conserves total `sum(r*l)` EXACTLY, which is what `-checkfurgrid` asserts.
    template <class IsFiber>
    void build(const std::vector<CurveSeg>& segs, IsFiber&& isFiber,
               int targetCells = 64 * 64 * 64, int maxAxis = 512) {
        *this = FurGrid{};
        // Pass 1: bounds over fiber segments only, radius-inflated so no strand's surface
        // pokes out of the grid it is supposed to be summarised by.
        Vec3 blo{ 1e300,  1e300,  1e300}, bhi{-1e300, -1e300, -1e300};
        size_t nFiber = 0;
        for (const CurveSeg& s : segs) {
            if (!isFiber(s.matId)) continue;
            ++nFiber;
            const double r = std::max(s.r0, s.r1);
            blo = Vec3{std::min(blo.x, std::min(s.p0.x, s.p1.x) - r),
                       std::min(blo.y, std::min(s.p0.y, s.p1.y) - r),
                       std::min(blo.z, std::min(s.p0.z, s.p1.z) - r)};
            bhi = Vec3{std::max(bhi.x, std::max(s.p0.x, s.p1.x) + r),
                       std::max(bhi.y, std::max(s.p0.y, s.p1.y) + r),
                       std::max(bhi.z, std::max(s.p0.z, s.p1.z) + r)};
        }
        if (!nFiber) return;
        // A perfectly flat coat (all strands in a plane) would give a zero extent and a
        // zero cell volume; pad degenerate axes rather than special-casing them.
        const double ext = std::max(bhi.x - blo.x, std::max(bhi.y - blo.y, bhi.z - blo.z));
        const double pad = std::max(1e-9, ext * 1e-4);
        blo = blo - Vec3{pad, pad, pad};
        bhi = bhi + Vec3{pad, pad, pad};
        lo = blo; hi = bhi;

        const Vec3 d = hi - lo;
        // Cubic cells: pick the side length that puts `targetCells` of them in the box.
        const double vol = std::max(1e-30, d.x * d.y * d.z);
        double side = std::cbrt(vol / std::max(1, targetCells));
        nx = std::max(1, std::min(maxAxis, (int)std::lround(d.x / side)));
        ny = std::max(1, std::min(maxAxis, (int)std::lround(d.y / side)));
        nz = std::max(1, std::min(maxAxis, (int)std::lround(d.z / side)));
        cell    = Vec3{d.x / nx, d.y / ny, d.z / nz};
        invCell = Vec3{1.0 / cell.x, 1.0 / cell.y, 1.0 / cell.z};
        cellVol = cell.x * cell.y * cell.z;
        cells.assign((size_t)nx * ny * nz, FurCell{});

        // Pass 2: deposit.  Accumulate raw sums in the FurCell fields (they are normalised
        // in pass 3), using `c` as the running `sum(r*l)` so no second array is needed.
        // `matId` is resolved by keeping the best-so-far mass in a side table only when
        // the scene actually mixes fiber materials, which is rare and cheap to detect.
        // The one accumulator that has nowhere to live in the packed cell: the mean tangent
        // is a signed vector and `mdir` is a quantised unit direction, so it is summed here
        // in full precision and packed once in pass 3.  Freed on return.
        std::vector<Vec3> meanAcc(cells.size(), Vec3{0, 0, 0});
        std::vector<float> matMass;
        int firstMat = -1; bool mixed = false;
        for (const CurveSeg& s : segs) {
            if (!isFiber(s.matId)) continue;
            if (firstMat < 0) firstMat = s.matId;
            else if (s.matId != firstMat) { mixed = true; break; }
        }
        if (mixed) matMass.assign(cells.size(), 0.0f);

        const double subStep = 0.5 * std::min(cell.x, std::min(cell.y, cell.z));
        for (const CurveSeg& s : segs) {
            if (!isFiber(s.matId)) continue;
            const Vec3 ab = s.p1 - s.p0;
            const double len = length(ab);
            if (!(len > 0.0)) continue;
            const Vec3 t = ab / len;
            const int nSub = std::max(1, (int)std::ceil(len / std::max(1e-12, subStep)));
            const double dl = len / nSub;
            for (int k = 0; k < nSub; ++k) {
                const double u = (k + 0.5) / nSub;
                const Vec3 p = s.p0 + ab * u;
                const double r = s.r0 + (s.r1 - s.r0) * u;
                const double w = r * dl;
                totalRL += w;
                totalL  += dl;
                int ix = (int)((p.x - lo.x) * invCell.x);
                int iy = (int)((p.y - lo.y) * invCell.y);
                int iz = (int)((p.z - lo.z) * invCell.z);
                ix = std::min(nx - 1, std::max(0, ix));
                iy = std::min(ny - 1, std::max(0, iy));
                iz = std::min(nz - 1, std::max(0, iz));
                const size_t ci = (size_t)index(ix, iy, iz);
                FurCell& fc = cells[ci];
                meanAcc[ci] = meanAcc[ci] + t * w;
                fc.c   += (float)w;
                fc.txx += (float)(w * t.x * t.x);
                fc.tyy += (float)(w * t.y * t.y);
                fc.txy += (float)(w * t.x * t.y);
                fc.txz += (float)(w * t.x * t.z);
                fc.tyz += (float)(w * t.y * t.z);
                if (mixed) {
                    float& m = matMass[(size_t)index(ix, iy, iz)];
                    // Winner-take-all by mass, tracked as "mass of the current leader minus
                    // everything else": the leader flips only when a rival overtakes it.
                    if (fc.matId == s.matId || fc.matId < 0) { fc.matId = s.matId; m += (float)w; }
                    else if ((m -= (float)w) < 0.0f) { fc.matId = s.matId; m = -m; }
                } else {
                    fc.matId = firstMat;
                }
            }
        }

        // Pass 3: normalise.  `T /= sum(r*l)` makes it unit-trace; `c = 2*sum(r*l)/V` turns
        // the mass into an extinction coefficient.
        const double invV = 1.0 / cellVol;
        for (size_t ci = 0; ci < cells.size(); ++ci) {
            FurCell& fc = cells[ci];
            const double m = fc.c;
            if (!(m > 0.0)) { fc = FurCell{}; continue; }
            ++occupied;
            const float inv = (float)(1.0 / m);
            fc.txx *= inv; fc.tyy *= inv;
            fc.txy *= inv; fc.txz *= inv; fc.tyz *= inv;
            const Vec3   mv  = meanAcc[ci] * (1.0 / m);
            const double coh = std::min(1.0, length(mv));
            fc.mdir = furPackMean(mv, coh);
            fc.c = (float)(2.0 * m * invV);
        }
        valid = true;
    }

    size_t bytes() const { return cells.size() * sizeof(FurCell); }

    // Point query: the cell containing `p`, or null outside the grid. Exposed because the
    // far-tier medium samples the field at free-flight positions, and because a naive
    // fine-step Riemann sum over it is the independent reference `-checkfurgrid` §5 checks
    // the DDA against.
    const FurCell* at(const Vec3& p) const {
        if (!valid) return nullptr;
        const int ix = (int)std::floor((p.x - lo.x) * invCell.x);
        const int iy = (int)std::floor((p.y - lo.y) * invCell.y);
        const int iz = (int)std::floor((p.z - lo.z) * invCell.z);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) return nullptr;
        return &cells[(size_t)index(ix, iy, iz)];
    }
    double sigmaAt(const Vec3& p, const Vec3& d) const {
        const FurCell* fc = at(p);
        return fc ? sigmaT(*fc, d) : 0.0;
    }

    // -----------------------------------------------------------------------------------
    // MARCH.  Amanatides-Woo 3D-DDA from `o` along unit `dir` for at most `maxDist`,
    // accumulating optical depth and the depth-weighted mean `sin^2 theta`.
    //
    // WHAT `tau` MEANS, and why it is not merely "an optical depth".  A ray of length L
    // through a volume V whose cross-section it sees as A = V/L hits fiber i with
    // probability `2 r_i l_i sin(theta_i) / A`, so the EXPECTED NUMBER OF CROSSINGS over
    // that length is `L * (1/V) * sum_i 2 r_i l_i sin(theta_i)` = `L * sigma_t(d)`.  The
    // integral of sigma_t is therefore the mean fiber count the ray-shooting walk would
    // have returned — the very quantity Zinke's `n` is — with no primitive tests and no
    // sampling noise.  That identity is what lets this replace `Scene::walkFibers`.
    FurMarch march(const Vec3& o, const Vec3& dir, double maxDist) const {
        FurMarch m;
        if (!valid || !(maxDist > 0.0)) return m;
        // Slab clip against the grid box.
        double t0 = 0.0, t1 = maxDist;
        for (int a = 0; a < 3; ++a) {
            const double od = (&dir.x)[a], oo = (&o.x)[a];
            const double bl = (&lo.x)[a], bh = (&hi.x)[a];
            if (std::fabs(od) < 1e-15) { if (oo < bl || oo > bh) return m; continue; }
            const double inv = 1.0 / od;
            double na = (bl - oo) * inv, fa = (bh - oo) * inv;
            if (na > fa) std::swap(na, fa);
            t0 = std::max(t0, na); t1 = std::min(t1, fa);
            if (t0 > t1) return m;
        }
        m.hit = true;
        // Start cell: nudge inside so a ray entering exactly on a face lands in the box.
        const Vec3 pe = o + dir * (t0 + 1e-9);
        int ix = std::min(nx - 1, std::max(0, (int)((pe.x - lo.x) * invCell.x)));
        int iy = std::min(ny - 1, std::max(0, (int)((pe.y - lo.y) * invCell.y)));
        int iz = std::min(nz - 1, std::max(0, (int)((pe.z - lo.z) * invCell.z)));
        int step[3]; double tMax[3], tDelta[3];
        for (int a = 0; a < 3; ++a) {
            const double od = (&dir.x)[a];
            const int    ii = (a == 0 ? ix : a == 1 ? iy : iz);
            const double cs = (&cell.x)[a], bl = (&lo.x)[a];
            if (std::fabs(od) < 1e-15) { step[a] = 0; tMax[a] = 1e300; tDelta[a] = 1e300; continue; }
            step[a]   = od > 0.0 ? 1 : -1;
            tDelta[a] = cs / std::fabs(od);
            const double bound = bl + (ii + (od > 0.0 ? 1 : 0)) * cs;
            tMax[a]   = (bound - (&o.x)[a]) / od;
        }
        double t = t0;
        double bestMat = 0.0;   // depth of the leading material, as a winner-take-all margin
        double s2Acc = 0.0;
        int    guard = 4 * (nx + ny + nz) + 8;   // DDA cannot outlive the lattice; belt and braces
        while (t < t1 && guard-- > 0) {
            const int    axis = (tMax[0] < tMax[1]) ? ((tMax[0] < tMax[2]) ? 0 : 2)
                                                    : ((tMax[1] < tMax[2]) ? 1 : 2);
            const double tNext = std::min(tMax[axis], t1);
            const double seg   = tNext - t;
            if (seg > 0.0) {
                const FurCell& fc = cells[(size_t)index(ix, iy, iz)];
                if (fc.c > 0.0f) {
                    const double s2 = furSin2(fc, dir);
                    const double dt = fc.c * std::sqrt(std::max(0.0, 1.0 - s2)) * seg;
                    m.tau  += dt;
                    s2Acc  += dt * s2;
                    if (m.matId == fc.matId || m.matId < 0) { m.matId = fc.matId; bestMat += dt; }
                    else if ((bestMat -= dt) < 0.0) { m.matId = fc.matId; bestMat = -bestMat; }
                }
            }
            t = tNext;
            if (t >= t1) break;
            const int i = (axis == 0 ? (ix += step[0]) : axis == 1 ? (iy += step[1]) : (iz += step[2]));
            const int n = (axis == 0 ? nx : axis == 1 ? ny : nz);
            if (i < 0 || i >= n) break;
            tMax[axis] += tDelta[axis];
        }
        m.sin2 = m.tau > 0.0 ? std::min(1.0, std::max(0.0, s2Acc / m.tau)) : 0.0;
        return m;
    }
};

// The sign-marginalised read of a table the grid can only index by |theta|.  The
// orientation tensor cannot distinguish `t` from `-t`, so a march yields `|sin theta|` and
// nothing more.  Picking a sign would be inventing information; averaging the table over
// both is the correct marginalisation given a second moment, and is what this does.  The
// dual-scattering curves are near-even in theta anyway (only the ~3 degree cuticle tilt
// `alpha` breaks the symmetry), so the two reads normally agree to a fraction of a percent
// and this costs one extra lerp.
template <class Lookup>
inline double furAvgSigned(Lookup&& lk, double absTheta) {
    return 0.5 * (lk(absTheta) + lk(-absTheta));
}
