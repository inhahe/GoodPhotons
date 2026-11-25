#pragma once
// The gather footprint (M-GATHERAREA): what share of a density estimate's disc is the gather point's own
// surface, for the photon gathers of modes M and S (photonmap_render.h, sppm_render.h) and the surface
// merges of modes J and U (bdpt.h, vcm.h), which divide their sums by it.
#include "scene.h"
#include "geometry.h"
#include "bsdf_eval.h"    // holdsMergeVertex: the faces a merge footprint measures
#include "hair_shade.h"   // hairOpacityAt: a strand's coin in the probe
#include "gatherplane.h"
#include "gafootprint.h"
#include "rng.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---- GATHER FOOTPRINT (M-GATHERAREA, `-gatherarea <M>`) --------------------------------------
// The direct density estimate divides by pi*r^2, the area of the full gather disc, while
// collecting only from the part of that disc that is real, same-facing surface. Where the disc
// overhangs -- a cap edge, a fold of cloth, a hair strand -- the divisor is too big and the
// estimate is dark in proportion. Measured on `gallery_rain`: flat ground 0 %, a cap edge -33 %,
// Alice's dress -42 %, her hair -71 %.
//
// `gatherCoverage` measures the fraction of the tangent-plane disc that has same-facing surface
// under it, by probing M points along -n. Returns 1.0 when the feature is off, so the estimate
// is bit-identical then.
//
// WHY A PROBE AND NOT AN ANALYTIC CLIP: the entry prescribes clipping each same-facing primitive
// to the disc, which is exact for triangles and IMPOSSIBLE for everything else in this scene --
// fur (the biggest single loss, mats 38-41), isosurfaces, CSG solids. One intersector call
// handles them all, and it is the same intersector the render already trusts.
// ON BY DEFAULT at 8 probes since v0.268.0. `-gatherarea 0` restores the pre-0.267 estimator.
// 8 is where the sweep plateaus: it recovers 91 % of `alice_hair`'s -68 % for 1.3-1.7x the
// gather cost, and 16 buys only a few more points. Lower is NOT better despite scoring well on
// cloth -- see the Jensen note in known-issues.md.

// FTRACE_GAGEOM=1 (`-gageom 1`): coverage from the geometric footprint rather than from probe
// rays. OFF by default -- it is a prototype, it is CPU-only, and it is inert on flat geometry by
// construction (footprint 1.0000 there), so switching it on changes only the places the probe was
// already guessing at. `-gageom-disc` / `-gageom-curve` set its sampling.
inline bool gaGeomOn() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_GAGEOM");
        return e && *e && *e != '0';
    }();
    return on;
}
inline int gaGeomDisc() {
    static const int n = [] {
        const char* e = std::getenv("FTRACE_GAGEOMDISC");
        const int v = e ? std::atoi(e) : 0;
        return v >= 1 ? v : 64;
    }();
    return n;
}
// Max surfaces counted along one disc ray. The footprint marches past each hit and keeps
// going, so this bounds the work on a tangle -- a coat's chord can cross dozens of strands.
// Hitting the cap sets `incomplete` and the estimator falls back rather than divide by an
// under-count.
inline int gaGeomLayers() {
    static const int n = [] {
        const char* e = std::getenv("FTRACE_GAGEOMLAYERS");
        const int v = e ? std::atoi(e) : 0;
        return v >= 1 ? v : 32;
    }();
    return n;
}

inline int gatherAreaSamples() {
    static const int m = [] {
        const char* e = std::getenv("FTRACE_GATHERAREA");
        return e ? std::max(0, std::atoi(e)) : 8;   // below 0: none, as 0
    }();
    return m;
}
// FTRACE_GADIAG=1: per-material tally of WHY a probe contributed nothing. See the M-GATHERAREA
// fur case -- a probe that hits empty space and a probe that hits geometry facing the wrong way
// both add 0 to the coverage, but they mean opposite things, and the shipped estimator cannot
// tell them apart. Diagnostic only: off (the default) nothing below is touched and the estimate
// is bit-identical.
// FTRACE_GAREJECT=<pct>: suppress the footprint correction for a gather whose probes REJECT at
// least `pct` percent of their hits on the normal test -- the tangle signature. 0 = off, and off
// is bit-identical to the pre-0.273.6 estimator. See M-GATHERAREA: dense fur is accurate
// UNCORRECTED and +48 % corrected, because the probe sees the nearest layer while the query
// gathers from the whole ball, so on a tangle the correction has the wrong SIGN.
// ON BY DEFAULT AT 30 since v0.274.0. It was opt-in only because it was host-only, and
// defaulting a host-only correction would have split `-device gpu` from `-device cpu` on any
// scene with dense fur; the device twin landed in v0.273.10 and the two agree on the fur to 0.4
// points, so that reason is retired. Measured: fur -17.7 +- 2.2 points, collateral <= 1 point on
// every other ROI, +0.047 % on pure truncation, inert on flat ground. `-tanglegate 0` restores
// the pre-0.274.0 estimator exactly.
inline int gaRejectPct() {
    static const int p = [] {
        const char* e = std::getenv("FTRACE_GAREJECT");
        return e ? std::atoi(e) : 30;
    }();
    return p;
}
// FTRACE_GAREJW=<pct>: the area a REJECTED probe contributes, as a percentage of the 1.0 an
// accepted flat-on probe contributes. 0 = the pre-0.273.7 behaviour (a reject counts as empty
// space, identical to a miss) and is the default. See M-GATHERAREA: treating "there is surface
// here, facing the wrong way" as "there is no surface here" is what makes a tangle read as low
// coverage and so drives the correction the wrong way on dense fur.
inline int gaRejWeightPct() {
    static const int p = [] {
        const char* e = std::getenv("FTRACE_GAREJW");
        return e ? std::atoi(e) : 0;
    }();
    return p;
}
// FTRACE_GADEPTH=1: require NEGATIVE mean probe depth as well as a high reject rate before the
// tangle gate fires. Depth is measured below the tangent plane, so negative means the geometry
// found by the probes sits ABOVE it -- the shading point is inside a packed coat. Sparse strands
// give positive depth (probes fall through the gaps), and they NEED the correction. See
// M-GATHERAREA: the reject rate alone cannot tell fur from hair, because both are tangles.
inline bool gaDepthGateOn() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_GADEPTH");
        return e && *e && *e != '0';
    }();
    return on;
}
// THE FIBER GATE, ON BY DEFAULT since 0.277.0 (`-fibergate 0` restores the old estimator).
// Skips the coverage correction where the gather point is ON A FIBER, on both backends.
//
// Measured on gallery_rain, four GPU seeds, one binary, fixed -spp: mean absolute error on the fur
// ROI 48.6 % -> 13.0 %, an improvement at 4/4 seeds and never the wrong sign, with the other four
// ROIs reading the SAME value in both arms at every seed. The `off` spread across those seeds is
// +22.3..+67.5 %, which is why four realizations were needed to claim anything.
//
// A gather point on a 0.64 mm strand has no surface footprint for a tangent-plane disc to be
// clipped against, so `coverage` there measures how much of a disc neighbouring strands happen to
// intersect, which is not the quantity the density estimate divides by. The fiber% column of
// FTRACE_GADIAG shows the test separates fur from mesh 100 % to 0 % on two scenes, which neither
// the reject rate nor the depth statistic could do. See M-GATHERAREA.
inline bool gaFiberSkipOn() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_GAFIBER");
        return !(e && *e == '0');       // ON by default since 0.277.0; `-fibergate 0` turns it off
    }();
    return on;
}
// FTRACE_GABIAS=1 (`-gabias 1`): the BIAS-CORRECTED coverage, `(area + 1) / (M + 1)` instead of
// `area / M`. PROTOTYPE, off by default.
//
// The estimate divides by coverage, so it is `1/c-hat` of a noisy `c-hat`, and E[1/c-hat] >
// 1/E[c-hat] by Jensen -- the correction reads too bright, the more so the fewer probes. That is
// not a convergence error that more probes fix cheaply: measured, `alice_dress` reads -12.3 % at
// M = 8 against -15.4 % at M = 32 and has stopped moving between 16 and 32, so four times the
// rays buys three points of bias and nothing else.
//
// A pseudo-count removes it for free. For the plain binomial case (every probe flat-on, so
// `area` is a hit count) `(M+1)/(k+1)` is the textbook near-unbiased estimator of `1/p`. Here
// `area` carries the projection Jacobian and so is not a count, but the same shrinkage applies
// and the three properties that matter are structural:
//   * at `area == M` it is EXACTLY 1.0, so full coverage stays inert and flat ground stays
//     bit-identical -- which is what the whole feature rests on;
//   * it is bounded by M+1, so `gatherAreaScale`'s `cov < 0.05 -> 1.0` cliff is unnecessary --
//     and that cliff points the WRONG WAY, since a gather that found almost no surface is the
//     one that needs the LARGEST correction, not none;
//   * it is monotone in `area`, so it cannot reorder two gathers the raw estimator ranked.
// The prediction to test it against is that the M = 8 and M = 32 results should CONVERGE.
inline bool gaBiasOn() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_GABIAS");
        return !(e && *e == '0');       // ON by default since 0.278.0
    }();
    return on;
}
// FTRACE_GAGATE=<n> (`-gagate <n>`): hold the flat-interior early-out at a FIXED n probes
// instead of `M/4`. 0 = the M/4 behaviour and is the default, so this is bit-identical off.
//
// The gate returns coverage 1.0 -- no correction -- when its first `probe0` probes all land on
// flat-on surface. With `probe0 = M/4` that is 2-of-2 at M = 8 but 8-of-8 at M = 32, so the SAME
// disc passes the SAME test at two different rates purely because the budget changed (0.90
// against 0.66 on a 95 %-covered disc). Measured by elimination: `-gabias` removes the Jensen
// half of the M-dependence and collapsed `alice_hair` 4.5x and `alice_dress` 6.7x, while
// `cap_gyroid` -- an edge strip, which is exactly the geometry this gate misjudges -- got 2.0x
// WORSE. What survives the removal of one mechanism is the other one.
inline int gaGateProbes() {
    static const int n = [] {
        const char* e = std::getenv("FTRACE_GAGATE");
        return e ? std::atoi(e) : -1;   // -1 = no early-out, the default since 0.278.0
    }();
    return n;
}
// FTRACE_GABALL=1 (`-gaball 1`): accept a probe only where its hit lies inside the same BALL the
// photon query uses, not merely inside the probe's cylinder. PROTOTYPE, off by default.
//
// The probe starts `r` above the tangent plane and accepts `h.t <= 2r`, so it accepts surface
// anywhere in a cylinder of radius r and height 2r. The numerator is `queryR(p, r)` -- photons
// within 3D distance r, a BALL. A surface point at tangent offset `rr` and height `dz` sits at
// distance sqrt(rr^2 + dz^2) >= rr, so on anything non-flat the probe counts rim surface the
// query can never reach: the area comes out too big, the correction too small, and the estimate
// too dark. That is the sign of the entire residual left after `-gabias` and `-gagate`, and that
// residual is the same size on three quite different geometries, which a footprint-shaped error
// would not be.
//
// FLAT GROUND CANNOT MOVE, by construction: a flat hit lands at `h.t == r` exactly, so `dz == 0`
// and the test becomes `rr <= r`, true for every probe. `grid_ground` is also the one ROI with no
// residual to explain, so it is a control that cannot move rather than one that merely did not.
inline bool gaBallOn() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_GABALL");
        return !(e && *e == '0');       // ON by default since 0.278.0
    }();
    return on;
}
inline bool gaDiagOn() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_GADIAG");
        return e && *e && *e != '0';
    }();
    return on;
}
struct GaDiagMat {
    std::atomic<long long> miss{0}, reject{0}, accept{0};
    // DISTRIBUTION of the per-gather-point coverage, 8 equal bins over [0,1] with everything at
    // or above 1 in the last. The per-material MEAN cannot be compared against an ROI's required
    // coverage, because a material spans both flat interior (coverage ~1, nothing to correct) and
    // truncated edge (the part an edge-strip ROI actually scores), and mixing them dilutes exactly
    // the effect. A histogram lets the two sub-populations separate themselves without a spatial
    // gate to configure. See M-GATHERAREA / cap_gyroid.
    std::atomic<long long> covHist[8]{};
    // OCCUPANCY, which orientation alone cannot give: how far below the tangent plane the first
    // surface sits. `depthSum` is in units of the gather radius; `deep` counts hits past 0.25 r.
    // A packed shell (fur) hits shallow and tight; sparse strands (hair) let probes fall through
    // the gaps and hit something far below. See M-GATHERAREA, the fur-vs-hair split.
    std::atomic<long long> deep{0};
    std::atomic<long long> depthMilli{0};   // sum of 1000*depth/r, integral so it can be atomic
    // Is the GATHER POINT itself on a fiber? Counted so the type test can be verified to separate
    // fur from mesh before anything is gated on it -- every statistic tried so far (reject rate,
    // depth, deep%) failed to. Per material, so the table shows the split directly.
    std::atomic<long long> fiber{0};
    // Gather POINTS, not probes. `fiber` is incremented once per gatherCoverage call, so it must
    // be normalised against this and not against miss+reject+accept -- each point fires up to M
    // probes (and fewer when the adaptive early-out trips), so dividing by the probe total gives
    // a number capped near 1/M that looks like a low rate and is not one. That mistake read
    // "84 % of fur gather points are not fibers" off a ceiling of 12.5 %.
    std::atomic<long long> points{0};
};
inline std::vector<GaDiagMat>& gaDiag() {
    static std::vector<GaDiagMat> t(1024);      // matId is small; 1024 is far past any scene
    return t;
}


// The estimator itself. Wrapped by `gatherCoverage` below, which only records the result -- the
// wrapper exists so that every early return here (flat-interior gate, tangle gate, fiber gate)
// is histogrammed without editing any of them.
// `slack` is the gather's plane-test slack (gatherBandSlack at its photon map's radius), the one
// its photon test takes, and `holds` the faces its photons lie on: the footprint must describe the
// population the sum counts.
inline double gatherCoverageRaw(const Scene& scene, const Vec3& p, const Vec3& n,
                             double r, double slack, Pcg32& rng, int M, int matId = -1,
                             double fiberR = 0.0, GatherHolds holds = holdsPhotons) {
    if (M <= 0 || !(r > 0.0)) return 1.0;
    // `-gageom 1`: take the coverage from the GEOMETRY instead of from probe rays. The probe
    // fires M nearest-hit rays and so sees only the first surface along each; the geometric
    // footprint tests every primitive the ball contains. Measured against analytic answers:
    // exactly 1.0000 on a flat plane (so this is inert there, and flat ground stays
    // bit-identical) and 0.5986 on floor within 0.25 m of a wall, where a point ON the
    // junction must see half a disc and the minimum observed is 0.5156.
    //
    // DELIBERATELY NOT APPLIED ON FIBERS, and the fiber gate below is left to handle them.
    // On a coat the measured footprint is ~2.9x pi r^2 -- a tangle really does hold that much
    // surface -- so using it as a divisor would make fur about three times DARKER, and fur
    // already reads -17 % against truth. The footprint is right and the DIVISION is wrong
    // there: the density estimate assumes the ball meets one locally flat surface, and with
    // ~600 strands in it the numerator is already the wrong region. See M-GATHERAREA.
    if (gaGeomOn() && !(fiberR > 0.0 && gaFiberSkipOn())) {
        const double denom = 3.14159265358979323846 * r * r;
        // If the ball held geometry this cannot measure, the area is an UNDER-count and the
        // divide would over-brighten by up to kDisc-fold. Fall through to the probe instead:
        // a coarser estimate beats a confidently wrong one. gallery_rain is the scene that
        // forced this -- its caps are implicit isosurfaces and the first build of this path
        // reported footprint 0.0000 on a solid marble cap, which is impossible.
        bool incomplete = false;
        const double raw = gatherFootprintArea(scene, p, n, r, gaGeomDisc(), gaGeomLayers(),
                                               slack, &incomplete, holds);
        if (!incomplete) {
        const double cov = raw / denom;
        // THE COVERAGE MUST BE BOUNDED AWAY FROM ZERO. The estimate divides by it, and a
        // measured footprint of exactly 0 is not rare -- `skin` on fur_creature reports
        // min 0.0000, a gather that found no same-facing surface at all. The first build of
        // this path returned that straight through and produced a pixel 4.7e+11 times too
        // bright on `_ga_corner`, which would have wrecked any render it touched. The probe
        // path never had the problem because its pseudo-count `(area+1)/(M+1)` is bounded
        // below by construction; this path bypassed that.
        //
        // Clamped to one disc cell rather than snapped to 1.0. Returning 1.0 would be the
        // `cov < 0.05 -> 1.0` cliff that gaBias removed, and that cliff points the WRONG WAY:
        // a gather that found almost no surface is the one needing the LARGEST correction,
        // not none. A floor of 1/kDisc is the smallest non-zero area this method can resolve
        // -- one disc ray hitting -- so it bounds the correction at kDisc-fold while staying
        // monotone in the measurement.
        const double floorCov = 1.0 / (double)gaGeomDisc();
        return cov < floorCov ? floorCov : cov;
        }
    }
    // Is the gather point itself on a fiber? TALLY ONLY -- nothing is gated on it, because the
    // tally is what showed it cannot be: see M-GATHERAREA. A gather point on a 0.64 mm strand has
    // no surface footprint for a disc to be clipped against, so skipping the correction there is
    // the right idea, but `cr_coat` (used by `fur` blocks and nothing else) reports fiberRadius > 0
    // on only 15.9 % of its probes, so the test cannot reach the other 84 %.
    if (gaDiagOn() && matId >= 0 && matId < (int)gaDiag().size()) {
        gaDiag()[matId].points.fetch_add(1, std::memory_order_relaxed);
        if (fiberR > 0.0) gaDiag()[matId].fiber.fetch_add(1, std::memory_order_relaxed);
    }

    Vec3 t, b; onb(n, t, b);
    double area = 0.0;                 // in units of the full disc, so 1.0 == fully covered
    int   nRej = 0;                    // probes that FOUND geometry facing the wrong way
    double depthSum = 0.0;             // sum of (hit depth below the tangent plane) / r
    int    nHit = 0;                   // probes that found anything at all
    // THE SILHOUETTE GATE, as an adaptive early-out rather than a separate heuristic. The entry
    // proposes "only worth doing when the gather is near a silhouette or a small-feature
    // primitive", and the honest way to know that is to ask the same estimator with fewer
    // samples: probe a quarter of the budget first, and if every one of them lands on surface
    // that is flat-on (cos ~ 1), this disc is in the interior of a plane and the remaining
    // probes can only confirm it. That costs 4 rays instead of 16 on the ground plane and the
    // caps -- which is most of a frame -- while any disc that is actually truncated shows a
    // miss almost immediately and pays the full budget.
    //
    // Deliberately NOT a photon-count test: this entry already establishes that no photon
    // statistic can separate geometry from illumination, and a gate built on one would skip
    // exactly the dim truncated gathers that need correcting most.
    // `max(2, M/4)` and never M itself: at M = 4 the old form set probe0 = 4, so the check sat
    // at an index the loop never reaches and the early-out silently never fired -- which is why
    // M = 4 cost as much as M = 8 in the first sweep.
    // `-gagate n` holds this at n regardless of M (see gaGateProbes). Clamped to M-1 so the
    // check index stays inside the loop: probe0 == M is the silent-no-op the comment above warns
    // about, since `i == probe0` is then never reached.
    int probe0 = (M >= 4) ? ((M / 4 < 2) ? 2 : M / 4) : M;
    if (const int gp = gaGateProbes()) probe0 = (gp < M) ? gp : (M > 1 ? M - 1 : M);
    const double pathEps = std::max(1e-9, r * 1e-4);
    // Whether a closed solid may lie within reach of a probe's segment from the gather point to its
    // top, which stays within r * sqrt(2) of it: asked once, the first time a probe needs its side.
    int solidsNear = -1;
    for (int i = 0; i < M; ++i) {
        if (i == probe0 && area >= (double)probe0 * 0.995)
            return 1.0;                // interior of a flat patch: nothing to correct
        // STRATIFIED in the disc, and the stratification is not a refinement -- it attacks a
        // BIAS. The estimate divides by the measured coverage, and E[1/cov] > 1/E[cov] by
        // Jensen, so noise in `cov` makes the correction too BRIGHT, the more so the fewer
        // samples. Measured: `alice_dress` reads -5.9 % at M = 4 against -15.2 % at M = 16, and
        // the M = 4 figure is not the better one -- it is a bias cancelling the layering
        // under-count below. Cutting the variance of `cov` at fixed M shrinks that bias for
        // free, and a disc stratifies exactly: equal-area rings x equal angle sectors, jittered
        // inside each cell so it stays unbiased.
        //
        // sqrt(u) within the ring puts equal expected samples per unit AREA; a linear radius
        // would over-weight the middle and report a truncated disc as fuller than it is.
        // INDEPENDENT, not stratified, and that is a decision with a measurement behind it.
        // Stratifying the radius to fight the Jensen bias below is incompatible with the
        // early-out above: `u1 = (i + xi)/M` walks the rings from the centre outwards, so the
        // gate's first M/4 probes all land in the MIDDLE of the disc, which is covered almost
        // by definition -- the gate then fires on nearly every gather and the correction stops
        // happening. Measured at M = 16: `cap_gyroid` -4.3 % independent against -16.9 %
        // radius-stratified, `alice_hair` +1.5 % against -14.8 %. (Stratifying BOTH dimensions
        // off one index is worse still, -32.9 %, because it correlates radius with angle and
        // puts every sample on a spiral.) Independent samples are spread over the whole disc by
        // construction, which is exactly what the gate needs to see.
        //
        // sqrt(u) puts equal expected samples per unit AREA; a linear radius would over-weight
        // the middle and report a truncated disc as fuller than it is.
        const double rr = r * std::sqrt(rng.uniform());
        const double ph = 2.0 * PI * rng.uniform();
        const Vec3 q = p + t * (rr * std::cos(ph)) + b * (rr * std::sin(ph));
        // Probe from r ABOVE the tangent plane straight down. `2r` of travel is what lets a
        // curved surface still count: within the disc it deviates from the plane by at most
        // ~r^2/(2R), far inside this window for any radius worth gathering at.
        // COVERAGE (hair `opacity` < 1): a strand is present with probability o, so a probe that
        // lands on one passes straight through with probability (1 - o) -- the coin every walk
        // tosses at a fiber -- instead of counting a tube the transport itself may never see.
        // At opacity 0 the probe cannot tell fur from air, which is what the invisibility null
        // demands (measured before this: mode S +5.6 % / M +2.3 % on skin under opacity-0 fur,
        // all of it from this divisor, none from transport). The tube is cleared with the
        // curve-only tmin and the distance carried forward, so `h.t` stays depth from launch.
        //
        // A photon-holding face off the gather's tangent-plane band (withinGatherBand at the photon
        // test's slack) is another surface's, whose photons the query drops. Below the band, none
        // of this surface lies under the probe: a miss. Above it -- a shelf, a table top over the
        // floor point -- the probe goes on through, counting the faces it crosses that light cannot
        // (passesLight), photon-holding or not: where it then meets this surface the column may
        // lie inside an opaque solid resting on it (a box's own bottom face, or the floor inside a
        // bottomless one), where no photon can be, and is a miss too. Counted as covered, it would
        // ring the base of every object standing on a surface with a dark halo (a 3 cm box on a
        // floor at radius 0.1: 0.75 of mode R within 5 cm). A closed solid's face says which side
        // the probe is on (PROBE-SOLID-SIDE); for the rest an odd count of them means inside.
        Hit h;
        {
            Ray pr{q + n * r, n * -1.0};
            double tAcc = 0.0;
            // The opaque faces (passesLight) passed above the plane say whether the column ends
            // inside a solid resting on this surface. One that bounds a closed solid (Hit::solid)
            // says itself whether the probe went into it or out of it, so the last such, the
            // nearest the plane, says whether the column ends in one (`lastSolid`: +1 into, -1 out
            // of, 0 none). Where it crosses none, the side the probe began on decides (PROBE-PATH):
            // the side walkSegment finds along the segment from the gather point to the probe's
            // top, taken only when it is asked (`kAsk`). So a column inside a solid taller than r
            // is inside though the probe crosses none of its faces. An opaque face that cannot say
            // counts toward a parity (`crossed`). Either puts the column inside only where the low
            // leg -- from just over the gather point to the column's foot -- agrees that a wall
            // stands between the two (inside()): a bottomless box's lid or an open cup's top over a
            // floor its walls close in, a box resting on the floor, not a sheet hovering over it nor
            // a room the gather point lies in.
            constexpr int kAsk = 2;
            int crossed = 0, lastSolid = kAsk;
            int probeNet1 = 0;   // closed solids (+-1) the probe enters coming down, less those it leaves
            // The segment from the gather point to the probe's top (PROBE-PATH), walked once when asked.
            bool topWalked = false;
            SegmentCrossings topWalk;
            const auto walkTop = [&]() -> const SegmentCrossings& {
                if (!topWalked) {
                    topWalked = true;
                    if (solidsNear < 0) solidsNear = scene.solidNear(p, n, r * 1.4143) ? 1 : 0;
                    if (solidsNear) {
                        const Vec3 o = p + n * pathEps, top = q + n * r;
                        const double len = length(top - o);
                        topWalk = walkSegment(scene, o, (top - o) * (1.0 / len), len, pathEps);
                    }
                }
                return topWalk;
            };
            const auto side = [&]() {
                if (lastSolid == kAsk) lastSolid = walkTop().side;
                return lastSolid;
            };
            const auto cross = [&](const Hit& c, const Material* cm) {
                if (cm && passesLight(scene, *cm)) return;
                if (c.solid) {
                    lastSolid = ((c.solid > 0) == c.back) ? -1 : 1;   // leaving : entering
                    if (c.solid == 1 || c.solid == -1) probeNet1 += lastSolid;
                } else {
                    ++crossed;
                }
            };
            // The low leg, from just over the gather point to the column's foot (walkSegment): asked
            // only once what lies above puts the column inside, and then once.
            bool legAsked = false;
            SegmentCrossings leg;
            const auto lowLeg = [&]() -> const SegmentCrossings& {
                if (!legAsked) {
                    legAsked = true;
                    if (rr > pathEps) leg = walkSegment(scene, p + n * (kLowLegFrac * r), (q - p) * (1.0 / rr), rr, pathEps);
                }
                return leg;
            };
            // Whether the column lies inside something the gather point does not. An odd parity above
            // it where the low leg crosses an odd count of such faces too (a cup's wall, not a sheet
            // hovering over the surface). Or a solid's faces above saying it lies in a solid, where the
            // column lies deeper in solids than the gather point: the leg ending in more than it began
            // in (`net`: a box resting on the surface, or one touching it, whose coincident pair the
            // leg counts both of), or leaving one it began in (`low` below 0: a crate past a low
            // room's wall, the gather point inside); or else the chain over the top -- the segment from
            // the gather point to the probe's top, and the probe's descent -- ending deeper in closed
            // solids (`net1`, not a cap's faces, which it never leaves by): a slab on the surface too
            // thin for the leg to cross. Not a room the gather point lies in itself (the probe enters
            // its lid, the leg crosses nothing), nor a pillar the leg passes, nor an open quad or cup
            // there, nor a box clear of the surface at the path's time (a mover's).
            const auto inside = [&]() {
                if ((crossed & 1) && (!(rr > pathEps) || (lowLeg().open & 1))) return true;
                if (!(side() > 0)) return false;
                const SegmentCrossings& L = lowLeg();
                return L.net > 0 || L.low < 0 || walkTop().net1 + probeNet1 > 0;
            };
            bool found = false;                  // the probe ended on what it counts
            bool ended = false;                  // ... or on what it does not: either way, it ended
            // A closed solid's face in the band, met from inside, the probe passes (below), and
            // ends there only if it then meets nothing of this surface in the band.
            Hit under;
            bool underFound = false;
            // A strand the probe passes through costs none of the sixteen faces, up to 48 of them:
            // under a fur coat it may cross a few on its way to the skin.
            for (int pass = 0, skip = 0; pass < 16 + skip; ++pass) {
                h = scene.closestHit(pr);
                if (!h.valid) break;
                h.t += tAcc;
                if (h.t > 2.0 * r) break;                               // past the window: a miss
                const Material* pm = (h.matId >= 0 && h.matId < (int)scene.mats.size()) ? &scene.mats[h.matId] : nullptr;
                // A strand that holds no photon, whatever its lobes, and the surface the gather's
                // photons lie on is beyond it: through the tube, for none of the sixteen faces.
                if (pm && h.fiberRadius > 0.0 && pm->type != MatType::Hair &&
                    !holds(scene, *pm, h.fiberRadius)) {
                    if (skip < 48) ++skip;
                    tAcc = h.t;
                    pr = Ray{h.p + pr.d * 1e-9, pr.d, 2.5 * h.fiberRadius + 1e-9};   // through the tube
                    continue;
                }
                // On through a face that holds no photon -- but only one above the plane: a face ON
                // it (a glass box standing on the floor, coplanar with it) is the surface's own
                // height, and passing it would skip the surface beside it. An opaque one is a
                // crossing all the same: a mirror box shadows the floor under it.
                if (pm && probePassesThrough(scene, *pm, holds) && r - h.t > 1e-3 * r) {
                    cross(h, pm);
                    tAcc = h.t;
                    pr = Ray{h.p + pr.d * 1e-9, pr.d};
                    continue;
                }
                if (pm && pm->type == MatType::Hair && h.fiberRadius > 0.0) {
                    // A strand holds no photon either way; its opacity coin says whether it covers
                    // the disc.
                    const double o = hairOpacityAt(scene, *pm, h);
                    if (o >= 1.0 || rng.uniform() < o) { found = ended = true; break; }   // the strand IS here
                    tAcc = h.t;
                    pr = Ray{h.p + pr.d * 1e-9, pr.d, 2.5 * h.fiberRadius + 1e-9};   // through the tube
                    continue;
                }
                const double dzH = r - h.t;
                if (withinGatherBand(std::fabs(dzH), rr * rr + dzH * dzH, dot(orientedGeoN(h), n), slack)) {
                    // A closed solid's face in the band but above the plane, met from inside (a
                    // ball's tilted underside, which the band widens to take in), is crossed and
                    // passed: the probe leaves the ball over the floor, and the floor below is
                    // what it ends on. No photon on that face can face the gather's way, light
                    // reaching it from outside the solid, below. But a closed room is a solid seen
                    // from inside too, and where its surface curves up into the band nothing of it
                    // lies below: there the probe ends on the face it passed (`under`). Met from
                    // outside it is the solid's top in the band (a coaster's), a surface the
                    // gather's photons lie on, which the faces above it decide. One on the plane
                    // (a box's own bottom) is the surface's height, and says nothing: "on" to within
                    // rounding (1e-7 r), not the 1e-3 r a face passed through above keeps, or the
                    // underside of a ball resting on the floor would be "on" it for 5 mm round the
                    // point it touches, which a probe from inside the ball must leave by.
                    if (h.solid && dzH > 1e-7 * r && (h.solid > 0) == h.back) {
                        cross(h, pm);
                        if (!under.valid) { under = h; underFound = !inside(); }
                        tAcc = h.t;
                        pr = Ray{h.p + pr.d * 1e-9, pr.d};
                        continue;
                    }
                    found = !inside();
                    ended = true;
                    break;
                }
                if (!(dzH > 0.0)) break;                                // below the band: a miss
                cross(h, pm);
                tAcc = h.t;
                pr = Ray{h.p + pr.d * 1e-9, pr.d};
            }
            if (!ended && under.valid) { h = under; found = underFound; }
            if (!found) h.valid = false;         // a miss, or sixteen faces without an answer
        }
        if (gaDiagOn() && matId >= 0 && matId < (int)gaDiag().size()) {
            GaDiagMat& g = gaDiag()[matId];
            if (!(h.valid && h.t <= 2.0 * r))          g.miss.fetch_add(1, std::memory_order_relaxed);
            else {
                if (dot(orientedGeoN(h), n) < kGatherFacingCos) g.reject.fetch_add(1, std::memory_order_relaxed);
                else                   g.accept.fetch_add(1, std::memory_order_relaxed);
                // Depth below the tangent plane, over ANY hit (accepted or rejected) -- the
                // question is where the geometry is, not which way it faces.
                const double depth = (h.t - r) / r;
                g.depthMilli.fetch_add((long long)(depth * 1000.0), std::memory_order_relaxed);
                if (depth > 0.25) g.deep.fetch_add(1, std::memory_order_relaxed);
            }
        }
        // Same 60-degree acceptance the photon query uses (onGatherSurfaceSided: kGatherFacingCos, both
        // the true normal on the side the light arrives from, after the band above), so the
        // footprint and the estimator agree on what surface is "here".
        if (h.valid && h.t <= 2.0 * r) {
            const double c = dot(orientedGeoN(h), n);
            // THE PROJECTION JACOBIAN, and it is not a refinement -- without it the correction
            // overshoots badly on exactly the geometry it is for. The probe samples uniformly in
            // the TANGENT PLANE, so it measures PROJECTED area; the estimator needs SURFACE
            // area, and dA = dq / cos(tilt). A patch tilted 60 degrees carries twice the surface
            // its shadow suggests. Measured on gallery_rain without this term: alice_hair went
            // from -68.0 % to +31.1 % -- past zero, because hair is nearly all steeply-tilted
            // surface and every bit of it was counted at its projected size. Flat ground has
            // cos = 1 and is untouched either way, which is why the null control could not have
            // caught this and the truncated elements could.
            // Only `area` is gated on the ball (see gaBallOn); `nHit`/`nRej`/`depthSum` keep
            // their cylinder basis so the tangle gate is held fixed by construction and the two
            // arms differ in exactly one quantity.
            const double dz = r - h.t;      // signed height of the hit above the tangent plane
            const bool inBall = !gaBallOn() || (rr * rr + dz * dz <= r * r);
            if (c >= kGatherFacingCos) { if (inBall) area += 1.0 / c; }
            else          ++nRej;
            depthSum += (h.t - r) / r;   // < 0 when the geometry sits ABOVE the tangent plane
            ++nHit;
        }
    }
    // THE TANGLE GATE. A high reject share means the disc is full of geometry pointing every
    // which way, not hanging over empty space -- and there the correction is not merely weaker,
    // it points the wrong way. Doing nothing is the measured-correct action for fur.
    if (gaRejectPct() > 0 && nRej * 100 >= gaRejectPct() * M) {
        // With the depth condition on, a tangle whose geometry lies BELOW the plane is sparse
        // strands rather than a packed coat, and those need the correction rather than losing it.
        const bool overfilled = !gaDepthGateOn() || (nHit > 0 && depthSum < 0.0);
        if (overfilled) return 1.0;
    }
    // A REJECT IS EVIDENCE OF SURFACE, NOT OF EMPTY SPACE. Adding its area back is the
    // continuous form of the same fix the gate approximates, and on geometry that rejects
    // nothing -- which is what truncation measures -- it changes exactly nothing.
    if (gaRejWeightPct() > 0)
        area += (double)nRej * (double)gaRejWeightPct() * 0.01;
    // PROTOTYPE (see gaFiberSkipOn): decided HERE, after the probes have run and consumed their
    // rng draws, so the arms differ only on fur. Returning early would skip those draws, and the
    // caller's rng is shared across gather points, so every later point would shift too -- the
    // four ROIs that must not move would then move for an unrelated reason.
    if (fiberR > 0.0 && gaFiberSkipOn()) return 1.0;
    // The pseudo-count (see gaBiasOn). Applied HERE and not at the early returns above, because
    // those all mean "do not correct" and must stay exactly 1.0 -- which this form also gives at
    // `area == M`, so the two agree by construction rather than by a special case.
    if (gaBiasOn()) return (area + 1.0) / (double)(M + 1);
    return area / (double)M;
}
// Never divide by a coverage so small that one stray probe inflates a pixel into a firefly. A
// gather that finds under a twentieth of its disc is not a measurement worth rescaling.
// Report the split, most-probed material first. Names come from MeshGroup, which is the only
// place an authored name survives the flatten into Scene::tris.
inline const char* nmOf(const Scene& sc, int matId, char* buf) {
    // Mesh group FIRST: where one exists its name is the OBJECT (`alice_dress`), which is more
    // useful in a diagnostic than the material, and putting it first keeps existing output
    // byte-identical. The material name only fills in the cases that printed `matN`.
    if (const char* n = sc.meshNameForMat(matId)) return n;
    if (const char* n = sc.matNameFor(matId))     return n;
    std::snprintf(buf, 24, "mat%d", matId);
    return buf;
}
inline void gaDiagReport(const Scene& scene) {
    if (!gaDiagOn()) return;
    std::vector<std::string> nm(gaDiag().size());
    for (const auto& mg : scene.meshGroups)
        if (mg.matId >= 0 && mg.matId < (int)nm.size() && nm[mg.matId].empty())
            nm[mg.matId] = mg.name;
    struct Row { int id; long long mi, rj, ac, tot, dp, dm, fb, pt; };
    std::vector<Row> rows;
    for (int i = 0; i < (int)gaDiag().size(); ++i) {
        const long long mi = gaDiag()[i].miss.load(), rj = gaDiag()[i].reject.load(),
                        ac = gaDiag()[i].accept.load();
        if (mi + rj + ac > 0)
            rows.push_back({i, mi, rj, ac, mi + rj + ac,
                            gaDiag()[i].deep.load(), gaDiag()[i].depthMilli.load(),
                            gaDiag()[i].fiber.load(), gaDiag()[i].points.load()});
    }
    if (rows.empty()) return;
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.tot > b.tot; });
    std::fprintf(stderr,
        "\n[gadiag] why a footprint probe contributed nothing, per material.\n"
        "[gadiag] MISS = the disc overhangs empty space (truncation). REJECT = geometry is there\n"
        "[gadiag] but faces the wrong way (a tangle). Same coverage, opposite causes.\n"
        "[gadiag] %-22s %10s %8s %8s %8s %8s %8s %8s\n", "material", "probes",
        "miss%", "rej%", "acc%", "depth/r", "deep%", "fiber%");
    char buf[24];
    for (size_t k = 0; k < rows.size() && k < 24; ++k) {
        const Row& r = rows[k];
        const long long hits = r.rj + r.ac;
        std::fprintf(stderr,
                     "[gadiag] %-22s %10lld %7.1f%% %7.1f%% %7.1f%% %8.3f %7.1f%% %7.1f%%\n",
                     nmOf(scene, r.id, buf),
                     r.tot, 100.0 * (double)r.mi / (double)r.tot,
                     100.0 * (double)r.rj / (double)r.tot,
                     100.0 * (double)r.ac / (double)r.tot,
                     hits ? (double)r.dm / 1000.0 / (double)hits : 0.0,
                     hits ? 100.0 * (double)r.dp / (double)hits : 0.0,
                     r.pt ? 100.0 * (double)r.fb / (double)r.pt : 0.0);
    }
    // The COVERAGE DISTRIBUTION, which the means above cannot substitute for: a material spans
    // flat interior (coverage ~1, correction inert) and truncated edge (what an edge-strip ROI
    // scores), and a single mean over both is diluted by exactly the population that is not the
    // effect. Read the low bins against the coverage an ROI's error implies. See M-GATHERAREA.
    std::fprintf(stderr, "[gadiag] coverage distribution, %% of gather points per bin "
                         "(bin 0 = [0,0.125) ... bin 7 = >=0.875):\n");
    std::fprintf(stderr, "[gadiag] %-22s %7s %7s %7s %7s %7s %7s %7s %7s\n", "material",
                 "0", "1", "2", "3", "4", "5", "6", "7");
    for (size_t k = 0; k < rows.size() && k < 24; ++k) {
        const Row& r = rows[k];
        long long h[8], tot = 0;
        for (int b = 0; b < 8; ++b) { h[b] = gaDiag()[r.id].covHist[b].load(); tot += h[b]; }
        if (!tot) continue;
        std::fprintf(stderr, "[gadiag] %-22s", nmOf(scene, r.id, buf));
        for (int b = 0; b < 8; ++b)
            std::fprintf(stderr, " %6.1f%%", 100.0 * (double)h[b] / (double)tot);
        std::fprintf(stderr, "   (%lld pts)\n", tot);
    }
}

inline double gatherCoverage(const Scene& scene, const Vec3& p, const Vec3& n,
                             double r, double slack, Pcg32& rng, int M, int matId = -1,
                             double fiberR = 0.0, GatherHolds holds = holdsPhotons) {
    const double cov = gatherCoverageRaw(scene, p, n, r, slack, rng, M, matId, fiberR, holds);
    if (gaDiagOn() && matId >= 0 && matId < (int)gaDiag().size()) {
        int b = (int)(cov * 8.0);
        if (b < 0) b = 0;
        if (b > 7) b = 7;                       // coverage >= 1 (a tilted patch can exceed it)
        gaDiag()[matId].covHist[b].fetch_add(1, std::memory_order_relaxed);
    }
    return cov;
}
inline double gatherAreaScale(double cov) {
    // With the pseudo-count on, `cov` is already bounded below by 1/(M+1) and the cliff would do
    // nothing but misfire at large M -- at M = 32 a zero-coverage gather lands at 0.030, below
    // the threshold, and would have its correction thrown away entirely.
    if (gaBiasOn()) return (cov > 0.0) ? 1.0 / cov : 1.0;
    return (cov >= 0.05) ? 1.0 / cov : 1.0;
}

// THE MERGE FOOTPRINT of modes J and U (J-THIN-GAP-LEAK), at a camera vertex `p` whose true normal,
// turned to the side it was reached from, is `n`. A merge sums only the photons of its own surface,
// in the band about its tangent plane and facing its way (`keeps`, onGatherSurfaceSided), and divides
// by the share of the disc that surface covers (`scale`, gatherCoverage over the faces merge vertices
// lie on: holdsMergeVertex), the band's slack one value for both, so the light trapped in a
// millimetre gap under a slab stays off the slab. `-gatherarea 0` turns both off: every photon in the
// disc, over the whole disc. Host J (bdpt.h) and U (vcm.h); device twin DMergeFootprint.
struct MergeFootprint {
    Vec3 p, n;
    double r = 0.0, slack = 0.0, fiberR = 0.0;   // the merge radius; the band's; the vertex's strand
    int M = 0, matId = -1;                       // the probes (gatherAreaSamples), 0 off; for -gadiag
    bool twoSided = false;                       // a two-sided lobe keeps its far side's photons
    MergeFootprint(const Vec3& p_, const Vec3& n_, double radius, bool twoSided_, int matId_, double fiberR_)
        : p(p_), n(n_), r(radius), slack(gatherBandSlack(radius, fiberR_)), fiberR(fiberR_),
          M(gatherAreaSamples()), matId(matId_), twoSided(twoSided_) {}
    bool on() const { return M > 0; }
    // Whether the photon at `pp`, whose surface's true normal turned to the side its light came
    // from is `pn`, lies on this surface. Every photon, the footprint off.
    bool keeps(const Vec3& pp, const Vec3& pn) const {
        if (!on()) return true;
        const Vec3 d = pp - p;
        bool farSide;
        return onGatherSurfaceSided(p, n, pp, pn, dot(d, d), slack, twoSided, farSide);
    }
    // What a nonzero merge sum is multiplied by: 1 over the share of the disc this surface covers,
    // its probes drawn from `rng`; 1, drawing nothing, the footprint off.
    double scale(const Scene& scene, Pcg32& rng) const {
        return on() ? gatherAreaScale(gatherCoverage(scene, p, n, r, slack, rng, M, matId, fiberR,
                                                      bdpt::holdsMergeVertex))
                    : 1.0;
    }
};
