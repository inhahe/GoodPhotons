// SURFACE PHOTONS for mode J (UPBP) — the point x point merge, i.e. VCM's vertex merging.
//
// WHY THIS EXISTS, GIVEN photonmap.h ALREADY STORES SURFACE PHOTONS
// -----------------------------------------------------------------
// Mode M's `PhotonMap` is a fine density-estimate cache and mode J cannot use it, for the
// same reason mode J traces its own light subpaths instead of borrowing mode M's photon pass
// (bdpt.h, "WHY MODE J DOES NOT REUSE tracePhotonPass"): a balance-heuristic weight is a
// ratio of the densities with which two competing techniques would have produced the SAME
// path, and a photon out of `Renderer::tracePhoton` carries no densities at all — nor is it
// even drawn from the same distributions as the connections it would be weighed against.
// A merge here has to come out of the same `randomWalk` that builds `light[]`, carrying the
// same per-vertex pdfFwd/pdfRev bookkeeping, or the ratio is between things that are not the
// same kind of thing.
//
// So this file is to `photonmap.h` what `BeamMis` is to `PhotonBeam`: the same physical
// record, plus the light half of every merge weight it can take part in, precomputed while
// the subpath that made it is still in hand.
//
// WHAT ONE RECORD IS
// ------------------
// A light-subpath surface vertex y_j that a photon could be gathered from — precisely the
// vertices `bdpt::surfMergeSite` accepts, and that predicate is shared with the MIS weight on
// purpose: a site the weight counts but the map never fills under-weights every competing
// technique, and a site the map fills but the weight ignores double-counts. The record keeps
//
//   p, wo      where it is, and the direction back toward the previous light vertex — the
//              fixed argument of the CAMERA vertex's BSDF at gather time. (The photon's own
//              material is never consulted: a merge shades with the camera vertex's BSDF.)
//   beta       the subpath's throughput carried to here at its own wavelength, emission and
//              1/p(lambda) included, exactly as a beam's stored power is
//   lambda     that wavelength, and its CIE triple cached — the gather reads a photon once
//              per nearby camera vertex, so evaluating three Gaussian sums there would repay
//              the cost per gather instead of once per photon (the same trade vcm.h makes)
//   misIdx     into the parallel `SurfMis` array
//
// THE ESTIMATOR
// -------------
//   L(eye[k]) += beta_cam * (1 / (n_m * pi r^2)) * SUM_photons  w * f_cam(wo_cam, wo_p) * Phi_p
//
// a plain Jensen disc estimate, with `w` the balance-heuristic weight that is this whole
// file's reason for existing. See bdpt::mergeEtaPrimeSurf for the weight's derivation and
// SurfMergeWeight for its assembly.
//
// MEMORY. 72 bytes of photon + 40 of MIS, and the MIS array is separate rather
// than inlined so that the day a non-MIS consumer wants these photons it does not pay for
// them (`BeamMis` is split from `PhotonBeam` for the same reason). Unlike beams there is no
// split step, so the two arrays are indexed 1:1 and `misIdx` is currently the identity — it
// is kept explicit anyway because a future trim/subsample would break that and a silently
// wrong index is the one bug class this file cannot survive.
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "linalg.h"
#include "allocreport.h"

namespace bdpt {

// Gather-side shading-normal correction at a merge point (cos_s/cos_g), which rebalances the
// geometric-cosine density estimate onto the shading cosine so a smooth mesh merges smoothly
// like mode R:
//
//   gcorr = |cos(wp, Ns)| / |cos(wp, Ng)|
//
// EXACTLY 1 when Ns==Ng (flat tris, analytic spheres), so every non-smooth scene is
// bit-identical with or without it. The grazing `cos(wp,Ng)` denominator is guarded
// (measure-zero, ~0 flux) so a degenerate sample falls back to no correction. NOTE: the
// standalone photon-map / SPPM gathers (modes M/S) already smooth-shade in this renderer and
// must NOT use this — only a MIS-COUPLED merge needs it. Why the coupling makes the
// difference is logged as tech debt in known-issues.md.
//
// It lives here rather than in vcm.h (where it was written, for mode U's VM) because mode J's
// point merge is the same estimator and needs the same correction; vcm.h now pulls it back in
// by `using bdpt::vmGatherCorr`, so the two modes cannot drift apart.
inline double vmGatherCorr(const Vec3& wp, const Vec3& ns, const Vec3& ng) {
    double denom = std::fabs(dot(wp, ng));
    if (denom <= 1e-8) return 1.0;
    return std::fabs(dot(wp, ns)) / denom;
}

// The light half of a point merge's weight. Same telescoped-ratio idea as `BeamMis`, but
// shifted one strategy, and the shift is the whole difference between the two files: a beam
// merge INSERTS a vertex neither walk had, so its reference connection C1 splits AFTER the
// beam's origin vertex y_j; a point merge IDENTIFIES a vertex both walks already have, so
// (keeping bdpt.h's convention that C1 is "the merge site is the LAST CAMERA vertex, joined
// to the one before it" — see mergeEtaPrimeSurf) C1 splits after y_{j-1} instead. Every
// accumulator below is therefore read at j-1 where BeamMis reads at j.
//
// Writing y_0..y_j for the light subpath (the photon sits at y_j), gate_i for "connecting
// y_{i-1} to y_i is a legal strategy" (neither delta), eta'_i for the merge/connection
// density ratio at y_i less that KIND's constant, and
//     B_i = PROD_{u=i}^{j-2} pdfRev(y_u)/pdfFwd(y_u)      (B_{j-1} = 1, empty product)
//
//   gateC1 = gate_j                        — is C1 itself, the ratio-1 reference, legal?
//   sumC   = SUM_{i<=j-1} gate_i * B_i     — every light-side connection strategy
//   sumMb  = SUM_{i<=j-1} B_i * eta'_i     — light-side BEAM merges (medium vertices)
//   sumMs  = SUM_{i<=j-1} B_i * eta'_i     — light-side POINT merges (stored photons)
//
// (the i = j-1 term of the two merge sums has B_{j-1} = 1 and so is simply eta'(y_{j-1}),
// which the light pass adds in by hand — its own recurrence would only reach it one step
// later. The merge AT y_j is the estimator's own term and is in neither sum.)
//
// EVERY ONE of those is multiplied at the gather by ONE ratio, `R` — the density of y_{j-1}
// seen from y_j when the path leaves y_j toward the CAMERA rather than the way the light walk
// actually continued. That is the only light-side density a point merge invalidates, and it
// is why `rCoef` below exists: everything in R except the camera vertex's own BSDF pdf is
// light-side geometry and is finished here, at store time.
struct SurfMis {
    double sumC  = 0.0;
    double sumMb = 0.0;
    double sumMs = 0.0;
    double pdfFwdA = 0.0;      // pl_j: the light-side AREA density of the photon's vertex.
                               // The numerator of the whole weight (eta = kappaS * pl_j) and
                               // of the ratio D1 that carries it onto the camera side, so it
                               // is a double: it is not a geometric quantity but a density,
                               // and densities in this renderer span the dynamic range of the
                               // whole path.
    float rCoef = 0.0f;        // cos(y_{j-1}) / (|y_j - y_{j-1}|^2 * remap0(pdfFwd(y_{j-1})))
                               // — the merge-independent half of R = pdfRev*(y_{j-1}) /
                               // pdfFwd(y_{j-1}). The gather multiplies in the camera
                               // vertex's BSDF pdf toward the photon's `wo`, which is the
                               // only factor of R it could not know here. Exactly BeamMis::
                               // rCoef's role, one vertex earlier.
    unsigned char gateC1 = 0;  // 1 if connecting y_{j-1} to y_j is a legal strategy. y_j is
                               // never delta (surfMergeSite), so this asks about y_{j-1}.
    unsigned short vert = 0;   // j — the light subpath index, for the depth cap only. Merging
                               // light vertex j with camera vertex k builds a path of depth
                               // j+k, and the connection loop refuses depth > maxDepth, so
                               // the merge must refuse it too (cf. BeamMis::vert, whose beam
                               // merge inserts a vertex and so caps at j+k+1).
};

struct SurfPhoton {
    Vec3  p{0, 0, 0};
    Vec3  wo{0, 0, 0};         // unit, toward the PREVIOUS light vertex
    float lambda = 0.0f;
    float beta   = 0.0f;       // throughput carried here at `lambda` (emission + 1/p(lam) in)
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;   // cie{X,Y,Z}(lambda), cached at store time
    unsigned misIdx = 0;
};

// A per-thread bank, merged into the map at the end of the light pass (mirrors BeamBank).
struct SurfBank {
    std::vector<SurfPhoton> pts;
    std::vector<SurfMis>    mis;
    size_t size() const { return pts.size(); }
    void clear() { pts.clear(); mis.clear(); }
};

// Uniform hash grid over the stored photons, counting-sort layout (photonmap.h's).
//
// `cellCount()` and every cell index are `long long`, NOT int. A 512^3 grid is 134 M cells
// and `(int)(nCells + 1)` overflows — which is exactly the bug still open against vcm.h's
// grid in known-issues.md ("mode U still uses a dense cell grid with the same overflow").
// This one is written the right way round from the start, and the resolution is additionally
// capped so a scene with one distant stray photon cannot ask for a petabyte of cells.
struct SurfMap {
    std::vector<SurfPhoton> pts;
    std::vector<SurfMis>    mis;
    long long nEmitted = 0;    // light subpaths this map was built from — the merge
                               // technique's sample count n_m in the balance heuristic
    double radius = 0.0;       // gather radius r_s (the acceptance disc's)

    Vec3 lo{0, 0, 0};
    double cell = 1.0;
    long long nx = 1, ny = 1, nz = 1;
    std::vector<int> cellStart;   // size nCells+1
    std::vector<int> order;       // photon indices, cell-contiguous

    bool empty() const { return pts.empty(); }
    size_t size() const { return pts.size(); }

    // The MIS partials of photon `i`, or null when the map carries none. A null return is
    // the gather's signal to skip the merge entirely rather than weight it 1: unlike a beam
    // map (whose weight-1 fallback degrades to mode M's estimator, over-bright but still a
    // picture), an unweighted point merge would be added ON TOP of a complete BDPT sum and
    // so double-counts every path it touches. Refusing is the honest failure.
    const SurfMis* misOf(size_t i) const {
        if (mis.empty() || i >= pts.size()) return nullptr;
        const size_t j = (size_t)pts[i].misIdx;
        return j < mis.size() ? &mis[j] : nullptr;
    }
    long long cellCount() const { return nx * ny * nz; }
    long long cellIndex(long long ix, long long iy, long long iz) const {
        return (iz * ny + iy) * nx + ix;
    }
    void cellCoord(const Vec3& p, long long& ix, long long& iy, long long& iz) const {
        ix = (long long)std::floor((p.x - lo.x) / cell);
        iy = (long long)std::floor((p.y - lo.y) / cell);
        iz = (long long)std::floor((p.z - lo.z) / cell);
        ix = std::min(std::max(ix, 0LL), nx - 1);
        iy = std::min(std::max(iy, 0LL), ny - 1);
        iz = std::min(std::max(iz, 0LL), nz - 1);
    }

    // `r` is the gather radius; the cell is sized to it so a query touches 3x3x3 cells.
    // A grid that would exceed `maxCells` is coarsened instead of refused — a coarser cell
    // only makes the query visit more photons, which costs time and changes no result.
    void build(double r, long long maxCells = 64LL << 20) {
        radius = (r > 0.0) ? r : 1e-6;
        cell = radius;
        const size_t n = pts.size();
        if (n == 0) { nx = ny = nz = 1; cellStart.assign(2, 0); order.clear(); return; }
        Vec3 mn = pts[0].p, mx = pts[0].p;
        for (const SurfPhoton& q : pts) {
            mn.x = std::min(mn.x, q.p.x); mn.y = std::min(mn.y, q.p.y); mn.z = std::min(mn.z, q.p.z);
            mx.x = std::max(mx.x, q.p.x); mx.y = std::max(mx.y, q.p.y); mx.z = std::max(mx.z, q.p.z);
        }
        // Grow the cell until the grid fits the budget. Doubling terminates fast (the count
        // falls 8x a step) and keeps `cell >= radius`, which the 3x3x3 query relies on.
        Vec3 ext0 = (mx - mn) + Vec3{cell, cell, cell} * 2.0;
        for (int guard = 0; guard < 64; ++guard) {
            long long ax = std::max(1LL, (long long)std::ceil(ext0.x / cell));
            long long ay = std::max(1LL, (long long)std::ceil(ext0.y / cell));
            long long az = std::max(1LL, (long long)std::ceil(ext0.z / cell));
            if (ax <= maxCells && ay <= maxCells && az <= maxCells &&
                (double)ax * (double)ay * (double)az <= (double)maxCells) {
                nx = ax; ny = ay; nz = az; break;
            }
            cell *= 2.0;
            ext0 = (mx - mn) + Vec3{cell, cell, cell} * 2.0;
            nx = ny = nz = 1;
        }
        lo = mn - Vec3{cell, cell, cell};
        const long long nCells = cellCount();
        std::vector<long long> cellOf(n);
        // The one allocation here sized by a flag rather than by the scene: `-jsurf-radius`
        // (through `maxCells`) decides how many cells there are. Name it, so an OOM says
        // which knob to turn instead of "bad allocation" between two progress lines.
        cellStart.clear();
        ftalloc::resize(cellStart, (size_t)nCells + 1, "the surface-merge grid", "-jsurf-radius");
        std::fill(cellStart.begin(), cellStart.end(), 0);
        for (size_t i = 0; i < n; ++i) {
            long long ix, iy, iz; cellCoord(pts[i].p, ix, iy, iz);
            const long long c = cellIndex(ix, iy, iz);
            cellOf[i] = c; ++cellStart[(size_t)c + 1];
        }
        for (long long c = 0; c < nCells; ++c)
            cellStart[(size_t)c + 1] += cellStart[(size_t)c];
        order.assign(n, 0);
        std::vector<int> cursor(cellStart.begin(), cellStart.end() - 1);
        for (size_t i = 0; i < n; ++i) order[(size_t)cursor[(size_t)cellOf[i]]++] = (int)i;
    }

    // fn(photonIndex) for every photon within `radius` of p.
    template <class F>
    void query(const Vec3& p, F&& fn) const {
        if (order.empty()) return;
        long long ix, iy, iz; cellCoord(p, ix, iy, iz);
        const double r2 = radius * radius;
        for (long long dz = -1; dz <= 1; ++dz) { const long long cz = iz + dz; if (cz < 0 || cz >= nz) continue;
        for (long long dy = -1; dy <= 1; ++dy) { const long long cy = iy + dy; if (cy < 0 || cy >= ny) continue;
        for (long long dx = -1; dx <= 1; ++dx) { const long long cx = ix + dx; if (cx < 0 || cx >= nx) continue;
            const long long c = cellIndex(cx, cy, cz);
            for (int k = cellStart[(size_t)c]; k < cellStart[(size_t)c + 1]; ++k) {
                const int idx = order[(size_t)k];
                const Vec3 d = p - pts[(size_t)idx].p;
                if (dot(d, d) <= r2) fn(idx);
            }
        }}}
    }
};

} // namespace bdpt
