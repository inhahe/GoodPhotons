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
// THE FOOTPRINT (J-THIN-GAP-LEAK). With the gather footprint on (`-gatherarea`, the default), the sum
// takes only the photons on the camera vertex's own surface -- near its tangent plane and facing its
// way (gatherplane.h onGatherSurfaceSided), as a mode M gather does -- and divides by the share of the
// disc that surface covers (gathercoverage.h gatherCoverage), probed only where the sum found light.
// The light trapped in a millimetre gap under a plate is then not the plate's, and the floor at the
// base of a box standing on it loses neither the wall's photons nor the area they stood for. The MIS
// weights keep the whole disc's kappa: they need only sum to one over the techniques.
//
// MEMORY. 88 bytes of photon + 40 of MIS, and the MIS array is separate rather
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
#include "merge_grid.h"   // the hashed lattice, one with the device's grid

namespace bdpt {

// A merge's gather-side shading-normal correction is geometry.h's arrivalAdjointCorr, which modes
// M and S's deposits take too: one function, so no two estimators can drift apart on a smooth mesh.

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
    unsigned char gateC1 = 0;  // 1 if connecting y_{j-1} to y_j is a legal strategy. In the
                               // merged path y_j scatters through the camera's lobe, never a
                               // delta, so this asks about y_{j-1}.
    unsigned char pad_ = 0;    // explicit, so the struct has no implicit padding a new field could
                               // hide in: the device upload copies every field and asserts the size
    unsigned short vert = 0;   // j — the light subpath index, for the depth cap only. Merging
                               // light vertex j with camera vertex k builds a path of depth
                               // j+k-1 (SurfMergeWeight's SHAPE), and the connection loop
                               // refuses depth > maxDepth, so the merge must refuse it too
                               // (cf. BeamMis::vert, whose beam merge inserts a vertex and
                               // keeps y_j, and so caps at j+k+1).
};

struct SurfPhoton {
    Vec3  p{0, 0, 0};
    Vec3  wo{0, 0, 0};         // unit, toward the PREVIOUS light vertex
    float lambda = 0.0f;
    float beta   = 0.0f;       // throughput carried here at `lambda` (emission + 1/p(lam) in)
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;   // cie{X,Y,Z}(lambda), cached at store time
    unsigned misIdx = 0;
    // The true normal of the surface it lies on, turned to the side its light came from (bdpt.h
    // vertexGeoN), as mode M's photons carry theirs: the footprint above.
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;
};
static_assert(sizeof(SurfPhoton) == 88, "SurfPhoton's size is quoted in -jsurf-count's memory notes");

// A per-thread bank, merged into the map at the end of the light pass (mirrors BeamBank).
struct SurfBank {
    std::vector<SurfPhoton> pts;
    std::vector<SurfMis>    mis;
    size_t size() const { return pts.size(); }
    void clear() { pts.clear(); mis.clear(); }
};

// The stored photons and the hashed lattice over them (merge_grid.h, photonmap.h's counting-sort
// layout): cells the gather radius wide, so a query reads the 3x3x3 neighbourhood, and buckets
// about two a photon, whatever box the photons span. The device's map (buildJSurfMapDevice,
// DSurfMap) is the same lattice.
struct SurfMap {
    std::vector<SurfPhoton> pts;
    std::vector<SurfMis>    mis;
    long long nEmitted = 0;    // light subpaths this map was built from — the merge
                               // technique's sample count n_m in the balance heuristic
    double radius = 0.0;       // gather radius r_s (the acceptance disc's)
    // A map the device draws itself (cudaJSurfOnDevice) from other subpaths than the host
    // traced -- mode J's fixed beam map or its beams off (BeamBudgetReq::hostSurf): how many it
    // draws, and the photon slots it reserves (from the pilot's photons a subpath). 0: this map's
    // own nEmitted, and twice its size.
    long long devPaths = 0;
    long long devCap = 0;

    MergeLattice lat = mergeLatticeAt(0, 0, 0, 1.0, 1);   // cells the gather radius wide
    ftalloc::BulkVec<int> cellStart;   // bucket b's photons are order[cellStart[b] .. cellStart[b + 1]) (mergeBin)
    std::vector<int> order;       // photon indices, bucket-contiguous
    std::vector<unsigned char> tag;   // order[k]'s block's tag (MergeKey)
    // Bytes held by the containers above (their capacity), for a cache that budgets maps by size.
    size_t heapBytes() const {
        return pts.capacity() * sizeof(SurfPhoton) + mis.capacity() * sizeof(SurfMis) +
               cellStart.capacity() * sizeof(int) + order.capacity() * sizeof(int) + tag.capacity();
    }
    // Bytes the containers above actually use (their size).
    size_t usedBytes() const {
        return pts.size() * sizeof(SurfPhoton) + mis.size() * sizeof(SurfMis) +
               cellStart.size() * sizeof(int) + order.size() * sizeof(int) + tag.size();
    }
    // Release the containers' spare capacity (see BeamMap::shrinkToFit).
    void shrinkToFit() {
        pts.shrink_to_fit(); mis.shrink_to_fit(); cellStart.shrink_to_fit(); order.shrink_to_fit();
        tag.shrink_to_fit();
    }

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
    // The radius alone, without the grid: for a map the device grids itself (render_cuda.h
    // cudaJSurfOnDevice), which reads only its photon count, emission count and radius.
    void radiusOnly(double r) {
        radius = (r > 0.0) ? r : 1e-6;
        lat = mergeLatticeAt(0, 0, 0, radius, 1);
        cellStart.clear(); cellStart.shrink_to_fit();
        order.clear(); order.shrink_to_fit();
        tag.clear(); tag.shrink_to_fit();
    }
    // `r` is the gather radius; the cell is sized to it so a query touches 3x3x3 cells.
    void build(double r) {
        radius = (r > 0.0) ? r : 1e-6;
        const size_t n = pts.size();
        Vec3 mn = n ? pts[0].p : Vec3{0, 0, 0};
        for (const SurfPhoton& q : pts) {
            mn.x = std::min(mn.x, q.p.x); mn.y = std::min(mn.y, q.p.y); mn.z = std::min(mn.z, q.p.z);
        }
        lat = mergeLatticeOver(mn.x, mn.y, mn.z, radius, n);
        order.assign(n, 0);
        mergeBin(lat, n, [&](size_t i) -> const Vec3& { return pts[i].p; },
                 [&](size_t i, size_t k) { order[k] = (int)i; }, cellStart, &tag,
                 "the surface-merge grid", "-jsurf-count");
    }

    // fn(photonIndex) for every photon within `radius` of p.
    template <class F>
    void query(const Vec3& p, F&& fn) const {
        if (order.empty()) return;
        const double r2 = radius * radius;
        const int* ord = order.data();
        mergeForEachEntry(lat, cellStart.data(), tag.data(), p.x, p.y, p.z, [&](int k) {
            const int idx = ord[k];
            const Vec3 d = p - pts[(size_t)idx].p;
            if (dot(d, d) <= r2) fn(idx);
        });
    }
};

} // namespace bdpt
