// Photon-mapped rendering (ROADMAP item 1, mode M).
//
// Two passes:
//   1. tracePhotonPass(): forward light-trace N photons with the camera splat OFF and
//      the deposit path ON, filling a PhotonMap (view-independent radiance cache).
//   2. renderPhotonCamera(): a backward camera pass that, at the first diffuse hit of
//      each camera ray, estimates reflected radiance either by a DIRECT radius density
//      query into the map (default) or, when final gather is enabled (fgRays > 0), by an
//      indirect Jensen final gather. Direct + specular reach the diffuse surface normally;
//      the map supplies the (direct + indirect) diffuse illumination.
//
// Final gather (fgRays > 0): instead of reading the density estimate AT the visible point
// x — which inherits the estimate's low-frequency blur right at the surface, softening
// contact shadows and small-scale detail — we shoot K cosine-weighted hemisphere sub-rays
// from x, trace one bounce to y_k, and query the map THERE. This decouples the visible-
// surface sharpness from the gather radius (the blur now lives one bounce away, at y),
// exactly the standard Jensen photon-map final gather. Direct light at x is recovered by
// gather rays that strike an emitter/environment directly; indirect by gather rays that
// strike another diffuse surface and read its (converged) outgoing radiance from the map.
// Costs ~K density queries per camera sample, so pair a larger fgRays with fewer spp.
//
// The map is built ONCE and can be reused for many cameras of a static scene — the
// flythrough win (build once, gather per frame): the multi-camera driver in main.cpp
// (runSharedPhotonMap) traces/builds one map, then calls renderPhotonCamera below for
// each frame's camera.
#pragma once
#include "gafootprint.h"
#include <vector>
#include <thread>
#include <cstdint>
#include "render.h"
#include "photonmap.h"
#include "photonbeams.h"   // volume single-scatter cache (mode M with -beams)
#include "beamgather.h"    // gatherPhotonBeams — the Beam x Ray estimator (shared with mode J)
#include "causticaim.h"    // Jensen projection map: aimed emission for the caustic pass
#include "allocreport.h"   // OOM that names the buffer, its size and the flag that sizes it
#include "backward.h"      // BackwardRenderer::neeLight / neeEnv for final-gather direct lighting
#include "scene_film.h"
#include "camera.h"
#include "color.h"
#include "geometry.h"
#include "parallel.h"      // ft::stopRequested — cooperative `-stop` inside the pixel loop
#include "render_progress.h"   // StageProgress — deposit progress for the live window/title
#include <algorithm>
#include <atomic>

#include <chrono>

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
// FTRACE_PHOTONBOUNCE=<n> (`-photon-bounce <n>`): cap the LIGHT path's bounce count in
// tracePhotonPass. It exists because nothing else could reach it. `-max-bounce` governs the
// CAMERA path, and in mode M that path stops at the first diffuse hit, so `-max-bounce 2`, `32`
// and `64` produce byte-identical mode-M images -- a perfectly clean, perfectly meaningless null
// for any question about how far LIGHT travels before it is deposited. The photon side sat at
// Renderer's struct default of 32 with no flag able to move it.
// HOST ONLY. The device photon pass in render_cuda.cu has its own cap and does not read this;
// use -device cpu when sweeping it, and see MAXBOUNCE-IGNORED in known-issues.md.
inline int photonMaxBounce() {
    static const int m = [] {
        const char* e = std::getenv("FTRACE_PHOTONBOUNCE");
        const int v = e ? std::atoi(e) : 0;
        return v >= 1 ? v : 32;
    }();
    return m;
}

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

// HAIR-NEE (0.334.0): mode M's camera walk connects to the lights at every fiber vertex it
// scatters through (see the Hair case of photonGather). FTRACE_HAIR_NEE=0 turns it off, for
// paired measurements only.
inline bool hairNeeOn() {
    static const bool on = [] { const char* e = std::getenv("FTRACE_HAIR_NEE"); return !(e && e[0] == '0'); }();
    return on;
}
// SPECTRAL NEE (0.341.0): evaluate a glossy vertex's light connection over the SpecThr grid
// rather than at the camera's single wavelength. `-no-spec-nee` restores the scalar term;
// the switch itself lives in lighttree.h so the CUDA upload reads the same object.
inline bool specNeeOn() { return lt::gSpecNee; }
inline int gatherAreaSamples() {
    static const int m = [] {
        const char* e = std::getenv("FTRACE_GATHERAREA");
        return e ? std::atoi(e) : 8;
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
inline double gatherCoverageRaw(const Scene& scene, const Vec3& p, const Vec3& n,
                             double r, Pcg32& rng, int M, int matId = -1,
                             double fiberR = 0.0) {
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
                                               &incomplete);
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
        const Hit h = scene.closestHit(Ray{q + n * r, n * -1.0});
        if (gaDiagOn() && matId >= 0 && matId < (int)gaDiag().size()) {
            GaDiagMat& g = gaDiag()[matId];
            if (!(h.valid && h.t <= 2.0 * r))          g.miss.fetch_add(1, std::memory_order_relaxed);
            else {
                if (dot(h.n, n) < 0.5) g.reject.fetch_add(1, std::memory_order_relaxed);
                else                   g.accept.fetch_add(1, std::memory_order_relaxed);
                // Depth below the tangent plane, over ANY hit (accepted or rejected) -- the
                // question is where the geometry is, not which way it faces.
                const double depth = (h.t - r) / r;
                g.depthMilli.fetch_add((long long)(depth * 1000.0), std::memory_order_relaxed);
                if (depth > 0.25) g.deep.fetch_add(1, std::memory_order_relaxed);
            }
        }
        // Same 60-degree acceptance the photon query uses (dot(ph.n, h.n) < 0.5 rejects), so the
        // footprint and the estimator agree on what surface is "here".
        if (h.valid && h.t <= 2.0 * r) {
            const double c = dot(h.n, n);
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
            if (c >= 0.5) { if (inBall) area += 1.0 / c; }
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
                             double r, Pcg32& rng, int M, int matId = -1,
                             double fiberR = 0.0) {
    const double cov = gatherCoverageRaw(scene, p, n, r, rng, M, matId, fiberR);
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

// ---- MODE-M PHASE PROFILE (`-mstats`) -------------------------------------------------------
// VOLCACHE asks for the split inside a mode-M frame's camera gather: how much is the SURFACE
// density estimate and how much is the BEAM gather, since only the latter is what a volumetric
// radiance cache would remove. Per-thread accumulators, summed and printed once.
struct MStats {
    std::atomic<long long> surfNs{0}, beamNs{0};
    std::atomic<long long> surfN{0}, beamN{0};
    void report(double wallSec) const {
        const double s = (double)surfNs.load() * 1e-9, b = (double)beamNs.load() * 1e-9;
        if (s <= 0.0 && b <= 0.0) return;
        std::fprintf(stderr,
            "[mstats] camera gather: surface estimate %.2f s over %lld calls | beam gather %.2f s "
            "over %lld probes | %.0f%% of the gather is beams | wall %.1f s (thread-seconds, so "
            "the two sum to more than the wall on %d threads)\n",
            s, surfN.load(), b, beamN.load(), (s + b) > 0.0 ? 100.0 * b / (s + b) : 0.0,
            wallSec, (int)std::thread::hardware_concurrency());
    }
};
inline MStats& mStats() { static MStats m; return m; }
inline bool mStatsOn() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_MSTATS");
        return e && std::atoi(e) != 0;
    }();
    return on;
}
struct MStatTimer {          // RAII: adds its lifetime to one accumulator, only when enabled
    std::atomic<long long>* ns; std::atomic<long long>* n;
    std::chrono::steady_clock::time_point t0;
    MStatTimer(std::atomic<long long>* nsAcc, std::atomic<long long>* nAcc)
        : ns(mStatsOn() ? nsAcc : nullptr), n(nAcc) {
        if (ns) t0 = std::chrono::steady_clock::now();
    }
    ~MStatTimer() {
        if (!ns) return;
        ns->fetch_add((long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t0).count(),
                      std::memory_order_relaxed);
        n->fetch_add(1, std::memory_order_relaxed);
    }
};

// ---- Forward photon pass: deposit into the map, no camera splat ---------------------
// Traces N photons across nThreads, each depositing into a private bank, then
// concatenates into pm.photons and records pm.nEmitted (= N). Does NOT build the grid;
// the caller picks the gather radius and calls pm.build(radius).
//
// `seedBase` is the ABSOLUTE index of the first photon of this pass: photon i draws
// from its own stream seeded by seedBase+i (seedUnit), so the deposited set is
// thread-count independent — and, crucially, a repeated-pass caller (SPPM) that
// passes its cumulative emitted count gets FRESH photons every pass. (Before this
// parameter existed every SPPM pass re-traced the identical photon set — a real
// correctness bug: progressive photon mapping's convergence needs independent
// passes, so mode S was re-averaging the same deposits at shrinking radii.)
//
// `bm` (optional) turns on the PHOTON-BEAM deposit: the same photons additionally store
// every medium crossing as a segment, giving mode M a view-independent volume cache (see
// photonbeams.h). Passing it changes how photons traverse media — straight, attenuated by
// the crossing, instead of the analog scatter-or-absorb free flight — so the SURFACE map
// changes too (it loses multiply-scattered volume light and gains correct single-scatter
// transmission). That is the documented `-beams` trade, now available to mode M.
// `beamTarget` is a budget on the stored beam count, applied as Russian roulette per beam;
// <= 0 keeps every crossing.
// `stage` (optional) reports deposit progress — how many of the N photons have been
// traced — so the caller can keep the live window's title bar moving through what is
// otherwise the longest silent phase of a mode-M render. Purely informational.
//
// `pmCaustic` (optional) turns on Jensen's TWO-MAP split: a deposit whose path reads
// L·S⁺·D — at least one focusing vertex, no scattering one (see photonVertexKind in
// render.h) — goes to *pmCaustic INSTEAD OF pm. The split is strict, so the two maps
// partition the same deposits: nothing is duplicated, nothing is lost, and a gather that
// sums the two estimates is exactly the one-map estimate would have been IF one radius
// suited both. It does not, which is the whole point — a caustic is a thin high-contrast
// concentration whose photons are orders of magnitude denser than the diffuse background,
// so buildAuto picks each map its own radius and the caustic stops being smeared away by
// a kernel sized for the ambient illumination. Null keeps every deposit in `pm` (mode S
// and every pre-0.199.7 caller).
//
// `aim` + `nAimed` (optional) add Jensen's PROJECTION-MAP half of the two-map scheme: a
// SECOND pass of nAimed photons whose emission is importance-sampled towards the scene's
// focusing geometry (causticaim.h), depositing into *pmCaustic only. Without it the caustic
// map is sharp and nearly empty — 0.12 % of deposits on gallery_rain — because a uniformly
// emitting sky almost never happens to hit a gem. The two passes are combined with the
// balance heuristic: both passes' caustic deposits are scaled by the SAME per-photon weight
// w = 1/(1 + (N_c/N_m)·rho) computed at emission (Renderer::applyCausticAim), and the caustic
// map's nEmitted stays at the main pass's count. So the aimed pass is a pure variance
// reduction — an incomplete or over-eager target set costs efficiency, never correctness —
// and nAimed = 0 leaves every deposit bit-for-bit what it was.
// `depositSurfaces` = false traces the pass for its BEAMS ALONE and leaves `pm` empty. This
// is mode J (UPBP), where surface transport is BDPT's job and a surface photon map would be
// both unused and, at the photon counts a beam map wants, the largest allocation in the
// process. Every other aspect of the pass — emission, media crossing, Russian roulette, the
// RNG stream — is untouched, so the beams a beams-only pass deposits are bit-identical to the
// ones a full mode-M pass would have deposited at the same seed. (Nothing branches on
// `photonDeposit` except `Renderer::depositPhoton`, which is a no-op when it is null.)
inline void tracePhotonPass(const Scene& scene, long long N, int nThreads,
                            bool diffraction, PhotonMap& pm, int heroC = hero::kHeroC,
                            uint64_t seedBase = 0, BeamMap* bm = nullptr,
                            long long beamTarget = 0,
                            const StageProgress* stage = nullptr,
                            PhotonMap* pmCaustic = nullptr,
                            const caim::AimMap* aim = nullptr,
                            long long nAimed = 0,
                            bool depositSurfaces = true) {
    if (nThreads < 1) nThreads = 1;
    // The aimed pass needs somewhere caustic to deposit and something to aim at.
    const bool doAimed = pmCaustic && aim && !aim->empty() && nAimed > 0 && N > 0;
    if (!doAimed) { aim = nullptr; nAimed = 0; }
    const double aimRatio = doAimed ? (double)nAimed / (double)N : 0.0;
    std::vector<PhotonBank> banks(nThreads);
    std::vector<PhotonBank> cbanks(pmCaustic ? nThreads : 0);
    std::vector<BeamBank>   bbanks(nThreads);
    std::vector<long long> emitted(nThreads, 0);
    // Beam budget: give each thread its share of the target as a self-thinning CAP and let
    // it keep everything until it gets there (BeamBank halves itself past the cap). Do NOT
    // precompute a survival rate from N — the beams-per-photon ratio is a property of the
    // scene's media volume fraction, not of the photon count, and guessing it undershot by
    // 131x on gallery_rain and overshot by 2.3x on _fog_cornell. See photonbeams.h.
    //
    // The 2x headroom lets a thread overshoot its exact share before halving, so the banks
    // still sum to at least the target when the final decimateTo trims to it; without it,
    // one halving per thread would land the total at ~half the budget.
    for (int t = 0; t < nThreads; ++t) {
        if (beamTarget > 0)
            bbanks[t].cap = std::max<size_t>(1024, (size_t)(2 * beamTarget / nThreads));
        // Private, per-thread stream for the self-thinning draws, so which beams get dropped
        // never depends on — or perturbs — the photon tracer's own RNG sequence. Salted by
        // `-seed` like every other stream: WHICH beams the roulette drops is part of the
        // realization, so leaving it fixed would have left mode M's beam map partly frozen
        // across seeds, which is the opposite of what the flag is for.
        bbanks[t].rng.seed(seedBase + 0x9E3779B97F4A7C15ULL * (uint64_t)(t + 1),
                           0xBF58476D1CE4E5B9ULL ^ g_rngSalt);
    }

    // Hero-wavelength deposit (modes M/S): each traced path deposits its live wavelengths
    // as per-λ photon records via tracePhotonHero (render.h). nEmitted still counts PATHS
    // (below), so the density estimate is energy-identical to single-λ but with far lower
    // chroma noise. Same gate as the forward tracers: no media / no GRIN (those stay C=1).
    const bool heroOn = (heroC > 1) && scene.media.empty() && !grin::sceneHasGrin(scene);

    // SPECTRAL BEAMS. Where `heroOn` above is gated OFF by the presence of media, this one is
    // gated on exactly the opposite thing — it is the media cache's own spectral widening, and
    // it applies precisely when there ARE media. `-beamspec` (pbeams::gSpecC) asks for it; the
    // scene has to be able to honour it (beamSpectralOK, above); and there must be a beam map
    // to deposit into at all.
    const int beamSpecC = (bm && pbeams::gSpecC > 1 && beamSpectralOK(scene))
                              ? std::min(pbeams::gSpecC, kBeamSpecMax) : 1;
    // ACHROMATIC-PATH BEAMS. Same scene-wide extinction test, asked without the `-beamspec`
    // condition — the mean-CIE fold stores no extra wavelengths, so `-beamspec 1` gets it too
    // (photonbeams.h, ACHROMATIC-PATH BEAMS).
    const bool beamAchroOK = bm && pbeams::gAchro && beamSpectralOK(scene);

    // Published photon count for `stage`. Written by the workers on the SAME 4096-photon
    // cadence as the stop poll (one relaxed fetch_add per 4096 photons is unmeasurable next
    // to 4096 path traces) and read by the monitor thread below. Relaxed ordering is right:
    // nothing is synchronised through it, it only feeds a title bar.
    std::atomic<long long> tracedTotal{0};

    auto worker = [&](int tid, bool aimed) {
        Renderer r; r.diffraction = diffraction; r.maxBounce = photonMaxBounce();
        if (aimed) {
            // Caustic-only pass: no global deposits (the global map is the main pass's and
            // is normalised by ITS nEmitted), and no beam deposits for the same reason — but
            // media must still be crossed straight, or this pass would be transporting by
            // different rules than the pass it is being combined with.
            r.causticDeposit  = &cbanks[tid];
            r.aimEmission     = true;
            r.beamStraightOnly = (bm != nullptr);
        } else {
            if (depositSurfaces) r.photonDeposit = &banks[tid];
            if (pmCaustic) r.causticDeposit = &cbanks[tid];
            if (bm) r.beamDeposit = &bbanks[tid];
        }
        // Bound on BOTH passes: the main pass does not aim, but it still has to MEASURE its
        // own samples' aimed density to weight its caustic deposits. That measurement draws
        // no randomness, so the main pass's photon set is untouched.
        r.aimMap = aim; r.aimMisRatio = aimRatio;
        r.useHero = heroOn; r.heroC = heroC;
        r.beamSpecC = aimed ? 1 : beamSpecC;
        r.beamAchroOK = !aimed && beamAchroOK;
        // Mode M never consults this one — its photon is born at the chosen emitter's own
        // spectral density, so it needs no conversion (see bdpt.h, BeamSpectral). It is set
        // anyway so the field never reads as "this scene's extinction is chromatic" in a
        // scene where it is not.
        r.beamSpecOK = !aimed && bm && beamSpectralOK(scene);
        Pcg32 rng;
        const long long Np = aimed ? nAimed : N;
        const uint64_t salt = aimed ? 0x94D049BB133111EBULL : 0xEB44ACCAB455D165ULL;
        long long lo = Np * tid / nThreads, hi = Np * (tid + 1) / nThreads;
        EnergyReport e;
        long long done = 0;
        for (long long i = lo; i < hi; ++i) {
            // Cooperative `-stop` / Ctrl-C. Until this poll existed the photon DEPOSIT was
            // completely uninterruptible: v0.194.0 taught the mode-M camera *gather* to stop,
            // but nothing polled here, so `ftrace -stop` on a deposit had to wait out the
            // entire `-n` before the flag was even looked at. On a large `-n` that is many
            // minutes of a process that ignores every stop request while its photon map keeps
            // growing (a 2e9-photon CPU pass was sitting on 10 GB and climbing) — i.e. exactly
            // the situation that tempts a `taskkill /F`, which is what `-stop` exists to
            // prevent. Checked every 4096 photons rather than every photon: a photon is
            // microseconds, so a per-iteration atomic load would be measurable in the hottest
            // loop of a mode-M/S build, while 4096 of them still lands the stop in well under
            // a tenth of a second.
            if ((done & 0xFFF) == 0) {
                if (done) tracedTotal.fetch_add(0x1000, std::memory_order_relaxed);
                if (ft::stopRequested()) break;
            }
            seedUnit(rng, seedBase + (uint64_t)i, salt);
            r.tracePhoton(scene, (const Camera*)nullptr, (Film*)nullptr, (Film*)nullptr, rng, e);
            ++done;
        }
        tracedTotal.fetch_add(done & 0xFFF, std::memory_order_relaxed);   // the tail
        // Count what was ACTUALLY emitted, not what was asked for. pm.nEmitted normalises the
        // density estimate, so reporting the full share after an early break would scale a
        // truncated pass down by the fraction it never traced and darken the image.
        // The aimed pass deliberately does NOT count: it emits no light of its own, it
        // re-estimates the main pass's caustic term with a different sampler.
        if (!aimed) emitted[tid] = done;
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t, false);
    // The deposit is a join-and-wait, so progress has to be sampled from OUTSIDE it: a
    // monitor thread polls the published count while the workers run. It is only started
    // when someone asked for progress, so a headless render spawns nothing extra.
    std::atomic<bool> monitorStop{false};
    std::thread monitor;
    if (stage && stage->report) {
        const long long nTotal = N + nAimed;
        monitor = std::thread([&, nTotal] {
            while (!monitorStop.load(std::memory_order_relaxed)) {
                stage->report("tracing photons",
                              tracedTotal.load(std::memory_order_relaxed), nTotal,
                              nullptr, 0.0);
                for (int i = 0; i < 20 && !monitorStop.load(std::memory_order_relaxed); ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    for (auto& th : pool) th.join();
    // --- Aimed caustic pass (causticaim.h) --------------------------------------------
    // Runs after the main pass rather than alongside it so both can use every core, and so
    // an `ftrace -stop` during the main pass skips it entirely (the caustic map is then
    // simply the un-aimed one, which is still correct — just noisier).
    if (doAimed && !ft::stopRequested()) {
        std::vector<std::thread> apool;
        for (int t = 0; t < nThreads; ++t) apool.emplace_back(worker, t, true);
        for (auto& th : apool) th.join();
    }
    if (monitor.joinable()) { monitorStop.store(true, std::memory_order_relaxed); monitor.join(); }

    size_t total = 0;
    for (auto& b : banks) total += b.size();
    pm.photons.clear();  ftalloc::reserve(pm.photons, total, "the photon map payloads", "-n");
    pm.pos.clear();      ftalloc::reserve(pm.pos, total, "the photon map positions", "-n");
    pm.nEmitted = 0;
    for (int t = 0; t < nThreads; ++t) {
        // Append both halves in the same thread order, so pos[k] stays the position of
        // photons[k] (PhotonMap's split layout — see photonmap.h).
        pm.photons.insert(pm.photons.end(), banks[t].payload.begin(), banks[t].payload.end());
        pm.pos.insert(pm.pos.end(), banks[t].pos.begin(), banks[t].pos.end());
        pm.nEmitted += emitted[t];
    }
    if (pmCaustic) {
        size_t ctotal = 0;
        for (auto& b : cbanks) ctotal += b.size();
        pmCaustic->photons.clear();
        ftalloc::reserve(pmCaustic->photons, ctotal, "the caustic map payloads", "-n");
        pmCaustic->pos.clear();
        ftalloc::reserve(pmCaustic->pos, ctotal, "the caustic map positions", "-n");
        for (int t = 0; t < nThreads; ++t) {
            pmCaustic->photons.insert(pmCaustic->photons.end(),
                                      cbanks[t].payload.begin(), cbanks[t].payload.end());
            pmCaustic->pos.insert(pmCaustic->pos.end(),
                                  cbanks[t].pos.begin(), cbanks[t].pos.end());
        }
        // SAME normalisation as the global map: nEmitted counts PATHS EMITTED, not photons
        // stored, and both maps were filled by the one pass. Using the caustic map's own
        // stored count here would be the classic two-map bug — it would rescale a rare
        // caustic up to the brightness of the whole light source.
        pmCaustic->nEmitted = pm.nEmitted;
    }
    if (bm) {
        size_t nb = 0;
        for (auto& b : bbanks) nb += b.size();
        bm->beams.clear();
        ftalloc::reserve(bm->beams, nb, "the photon-beam map",
                         "-beamcount (or -n, which feeds it)");
        for (int t = 0; t < nThreads; ++t)
            bm->beams.insert(bm->beams.end(), bbanks[t].beams.begin(), bbanks[t].beams.end());
        bm->nEmitted   = pm.nEmitted;   // same pass, same normalisation
        bm->nDeposited = bm->beams.size();
        // Exact trim. The per-thread banks each self-thinned to their own cap, so the total
        // lands somewhere in [target, 2*target] rather than on the number the user asked
        // for; this is the one unbiased cut that makes -beamcount mean what it says.
        if (beamTarget > 0) bm->decimateTo((size_t)beamTarget);
    }
}

// The Beam x Ray 1D volume gather (`gatherPhotonBeams`) used to be spelled out here. It moved
// to **beamgather.h** in 0.215.0 so mode J (UPBP, bdpt.h) could call it too without bdpt.h
// having to include this whole header to reach one function. Same name, same signature, same
// output — the call sites below are untouched — and it gained a template hook for a per-hit
// MIS weight that mode M instantiates as the constant 1.

// ---- Final-gather sub-ray: one INDIRECT bounce from a visible point into the map -----
// A gather ray shot from a diffuse visible point (visHit/visMat). It follows specular
// surfaces (monochromatic at `lambda`) exactly like photonGather, and terminates at:
//   * the first diffuse/translucent hit y -> a radius density query at y, where EACH
//     photon is reflected off BOTH y (its own material) AND the visible point (visMat),
//     evaluated at the photon's wavelength so the two-bounce colour bleed stays spectral
//     (the same per-photon-wavelength XYZ trick mode M already uses at the visible point).
//     The map at y already holds direct+indirect at y, so this one gather bounce captures
//     the full indirect illumination of the visible point.
//   * a finite EMITTER reached AFTER a specular bounce -> a monochromatic (camera-lambda)
//     sample reflected off the visible point. Reached WITHOUT any specular bounce (a
//     straight hemisphere ray onto a light) it returns 0: that direct term is supplied
//     instead by low-variance next-event estimation at the visible point (see
//     photonGather), so counting it here too would double-count. This specular-arrival
//     gate mirrors backward.h's MIS `specularArrival`.
//   * the ENVIRONMENT on ANY escape -> a monochromatic sample reflected off the visible
//     point (env has no finite-light NEE in mode M, so gather rays carry env's direct
//     term; its indirect bounces come from the map query at a diffuse hit above).
// Returns the XYZ radiance leaving the visible point toward the gather-ray origin for this
// single sampled direction (the caller averages over K samples). Because the sub-ray is
// cosine-weighted (pdf = cos/pi) and the visible BRDF is Lambertian (f_r = rho/pi), the
// cosine and 1/pi cancel: the visible-point weight reduces to rho(vis), folded per photon
// (diffuse hit) or applied once (specular-arrival emitter/env). `norm` = 1/(pi r^2
// nEmitted) as in the caller. Mirrors photonGather's specular walk; keep the two in sync.
//
// `pmC`/`normC` are the optional CAUSTIC map and its own normalisation (see tracePhotonPass).
// When present the density estimate is the SUM of the two maps' estimates — the maps hold
// disjoint deposits, so this is the same estimator with each population read at the radius
// that suits it.
// ---- SPECGATHER: the camera walk's spectral throughput --------------------------------------
// Mode M's camera walk is MONOCHROMATIC at the camera sample's wavelength while the photon map is
// POLYCHROMATIC -- every photon carries its own. The density estimate gets this right at the gather
// point (it evaluates the BRDF at `ph.lambda`, per photon), but every reflectance the ray collected
// on the way there used to be folded into the SCALAR `thr` at the camera's lambda, and that scalar
// then multiplied a photon sum spanning every wavelength. So the estimator computed
//     r(lambda_c) * sum_p rho(lambda_p) P_p CIE(lambda_p)
// where it owes
//     sum_p r(lambda_p) rho(lambda_p) P_p CIE(lambda_p).
// Identical when r is flat; otherwise biased by however r correlates with the photon spectrum, in
// EITHER direction -- measured against mode D on one Cornell sphere: flat 1.005, JH grey 0.985,
// JH white 0.880, JH red 1.142 (see known-issues). gallery_rain's gold gyroid and chrome ring are
// exactly this case.
//
// So the walk carries the same product sampled over a coarse wavelength grid beside the scalar, and
// the estimate reweights each photon by the ratio at ITS wavelength. Two properties make this cheap
// and safe: the ratio is identically 1 when every factor is flat (so an uncoloured scene renders
// BIT-IDENTICALLY, which is the regression test), and a factor is probed at three wavelengths first
// so a flat one costs three evaluations rather than K.
struct SpecThr {
    static constexpr int K = 24;                 // bins across [LAMBDA_MIN, LAMBDA_MAX]
    double v[K];                                 // the product over the grid (the numerator)
    // The SAME product at the camera's wavelength, accumulated from the very scalars the walk
    // multiplied into `thr`. It is NOT recomputed from the grid, and that is the whole point:
    // `thr` already contains this factor, so `thr * ratio` collapses to (everything else) *
    // v[lambda_p] exactly. The first version divided by the nearest BIN CENTRE instead, which
    // agrees for a smooth spectrum and does not where one plunges near the band edge -- there
    // the cancellation failed and the estimate detonated (a JH white measured 6e5 x mode D).
    double cam = 1.0;
    bool   any = false;                          // anything non-flat folded in? (else ratio == 1)
    SpecThr() { for (int k = 0; k < K; ++k) v[k] = 1.0; }
    static double lamOf(int k) {
        return LAMBDA_MIN + (k + 0.5) * (LAMBDA_MAX - LAMBDA_MIN) / (double)K;
    }
    static int binOf(double lam) {
        int b = (int)((lam - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN) * K);
        return b < 0 ? 0 : (b >= K ? K - 1 : b);
    }
    // Fold in one spectral factor, `f(lambda)`. Probes three wavelengths first: a factor that
    // agrees at all three is taken as flat and folded as a constant, which cancels in the ratio
    // anyway -- so even a spectrum that fools the probe can only be mis-taken where it is already
    // nearly flat, and the common case (every uncoloured material in the scene) pays three calls.
    // `f` evaluates the factor at a wavelength; `atCam` is the value the scalar throughput just
    // used at the camera's wavelength, passed in rather than recomputed so the two can never
    // disagree (see `cam`).
    template <class F> void mul(const F& f, double atCam) {
        cam *= atCam;
        const double a = f(lamOf(0)), b = f(lamOf(K / 2)), c = f(lamOf(K - 1));
        const double tol = 1e-12 * (1.0 + std::fabs(a));
        if (std::fabs(a - b) <= tol && std::fabs(a - c) <= tol) {
            for (int k = 0; k < K; ++k) v[k] *= a;   // flat: cancels in the ratio regardless
            return;
        }
        any = true;
        for (int k = 0; k < K; ++k) v[k] *= f(lamOf(k));
    }
    // Fold in a factor already evaluated ON the grid, plus its value at the camera's wavelength.
    // The media term comes this way: its per-wavelength values fall out of one shared
    // ratio-tracking walk rather than from a function that can be called per bin.
    void mulVec(const double* vk, double atCam) {
        cam *= atCam;
        for (int k = 0; k < K; ++k) {
            if (!any && std::fabs(vk[k] - vk[0]) > 1e-12 * (1.0 + std::fabs(vk[0]))) any = true;
            v[k] *= vk[k];
        }
    }
    // The correction a photon of wavelength `lamP` needs. Denominator is `cam`, so this
    // multiplied by `thr` leaves the walk's spectral factors evaluated at lamP and everything
    // else untouched. A dead path (cam == 0) contributes nothing either way.
    double ratio(double lamP) const {
        if (!any) return 1.0;
        return (cam > 1e-300) ? v[binOf(lamP)] / cam : 0.0;
    }
};

// The spectral twin of camMediaTr: the segment's transmittance at every SpecThr grid wavelength
// plus (in the last slot) the camera's own, all from one shared walk. The camera's slot is what the
// scalar `thr` takes, so the scalar and the carrier's denominator are the same number by
// construction -- the lesson from SPECGATHER's first attempt, which recomputed it and detonated.
inline void camMediaTrSpec(const Scene& scene, const Renderer& mats, const Vec3& o, const Vec3& d,
                           double len, double lambda, Pcg32& rng, double* Tv) {
    double lams[SpecThr::K + 1];
    for (int k = 0; k < SpecThr::K; ++k) lams[k] = SpecThr::lamOf(k);
    lams[SpecThr::K] = lambda;
    for (int k = 0; k <= SpecThr::K; ++k) Tv[k] = 1.0;
    // Same sun-disc multi-sampling the scalar version does (see pbeams::kSunDiscTrSamples): a
    // directly-viewed disc is 10^4 times its surroundings, so one sample of the cloud in front of it
    // twinkles. Averaging whole VECTORS keeps the wavelengths correlated within each sample.
    const int n = (scene.sunCount > 0 && scene.sunRadiance(d, lambda) > 0.0)
                      ? pbeams::kSunDiscTrSamples : 1;
    if (n == 1) {
        mats.mediaTransmittanceSpec(scene, o, d, len, lams, SpecThr::K + 1, Tv, rng);
        return;
    }
    double acc[SpecThr::K + 1] = {0.0};
    for (int i = 0; i < n; ++i) {
        double one[SpecThr::K + 1];
        for (int k = 0; k <= SpecThr::K; ++k) one[k] = 1.0;
        mats.mediaTransmittanceSpec(scene, o, d, len, lams, SpecThr::K + 1, one, rng);
        for (int k = 0; k <= SpecThr::K; ++k) acc[k] += one[k];
    }
    for (int k = 0; k <= SpecThr::K; ++k) Tv[k] = acc[k] / (double)n;
}

inline Vec3 photonGatherSub(const Scene& scene, const PhotonMap& pm, Ray ray, Pcg32& rng,
                            bool diffraction, int maxBounce, double lambda, double invPdfL,
                            double norm, const Hit& visHit, const Material& visMat,
                            const PhotonMap* pmC = nullptr, double normC = 0.0) {
    Vec3 L{0, 0, 0};
    double thr = 1.0;
    bool specularSeen = false;                           // any specular bounce so far?
    Renderer mats; mats.diffraction = diffraction;
    MediumStack stk;                                     // nested-dielectric medium stack
    const bool grinAny = grin::sceneHasGrin(scene);      // final-gather rays bend too

    // GLOSSY-NEE. `bwNee` is the shared direct-lighting estimator (its `neeLight` is what the
    // final gather already uses); `gmis` carries the lobe density of a glossy continuation to
    // whichever emitter site it reaches, and is cleared at the top of every bounce so a mirror
    // or a dielectric can never inherit one and halve the emission behind it.
    BackwardRenderer bwNee; bwNee.diffraction = diffraction;
    const bool gneeOn = BackwardRenderer::glossyNeeOn();
    BackwardRenderer::GlossyMis gmis;
    bool hairArrival = false;   // HAIR-NEE: the previous vertex was a fiber that already took its direct light

    for (int b = 0; b < maxBounce; ++b) {
        if (grinAny) {
            double arc = 0.0;
            grin::marchSegments(scene, ray,
                [&](const Vec3&, const Vec3&, double slen, double&) { arc += slen; return false; });
            int cm = stk.topMat();                       // Beer-Lambert over the marched arc
            double a = (cm >= 0) ? scene.mats[cm].absorb(lambda) : 0.0;
            if (a > 0.0 && arc > 0.0) thr *= std::exp(-a * arc);
        }
        Hit h = scene.closestHit(ray);
        if (h.valid) {                                   // Beer-Lambert in current medium
            int cm = stk.topMat();
            double a = (cm >= 0) ? scene.mats[cm].absorb(lambda) : 0.0;
            if (a > 0.0) thr *= std::exp(-a * h.t);
        }
        if (!h.valid) {                                  // escaped -> environment
            // Env is collected by the gather rays directly (mode M has no separate env
            // NEE / MIS at the visible point), so add it on ANY escape — the map at a
            // diffuse hit already supplies env's INDIRECT bounces, this supplies direct.
            if (scene.envIndex >= 0) {
                double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, lambda));
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * rhoV * scene.envRadiance(ray.d, lambda) * invPdfL);
            }
            return L;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return L;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            // Resolve to a LOBE and let the material switch shade it -- the coat with
            // probability R (the Fresnel reflectance at this angle), else a body lobe. The
            // coat is a material of its own (Material::coatChild), so it goes through the
            // switch's Glossy case and gets that case's glossy-NEE, throughput and MIS. An
            // earlier version reflected the ray inline here and skipped all three, which
            // broke the identity that a coat of reflectance 1 IS a glossy material: it
            // rendered 30 % bright against one (scraps/ident_*.ftsl). Same convention as
            // bdpt.h's randomWalk and the device's dResolveCompound.
            const double R = layeredCoatReflectance(scene, *mp, h, ray.d, lambda);
            int c = (rng.uniform() < R) ? mp->coatChild
                                        : mixResolveChild(scene, *mp, h, rng.uniform());  // body lobe: honours a bound weight map (device twin: dResolveCompound)
            if (c < 0) return L;
            mp = &scene.mats[c];
        }
        const Material& m = *mp;

        // Self-emission on a SPECULAR arrival (a diffuse arrival's direct term comes from
        // NEE at the visible point, so adding it here too would double-count). One-sided by
        // the geometric normal, matching Vertex::Le / bkRadiance.
        //
        // This no longer RETURNS: an emissive material still has a BSDF, so a glowing
        // diffuse surface both emits and reflects, and the walk has to go on to the density
        // estimate below. See photonGather for the measurement.
        if (m.isLight && specularSeen && dot(ray.d, h.ng) < 0.0 && !hairArrival) {
            double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, lambda));
            // GLOSSY-NEE's lobe-sampling half; 1 (and bit-identical) unless the last bounce was
            // a glossy one that already connected to this emitter.
            const double wMis = (gmis.pdf > 0.0)
                ? bwNee.glossyHitWeight(scene, gmis,
                        BackwardRenderer::emitterIndexOfResolved(scene, m), ray.d, &h.p, &h.n)
                : 1.0;
            L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                 * (thr * rhoV * emitSlot(scene, m, h, lambda) * invPdfL * wMis);
        }

        // GLOSSY-NEE: cleared HERE and not at the top of the loop. `gmis` is written by the
        // PREVIOUS bounce's glossy branch and read by THIS bounce's emitter/sun sites above, so
        // a clear at the loop top erases it a few lines before the only code that wants it --
        // which leaves the connection in place with no compensating weight on the lobe-sampling
        // side, i.e. double counting wherever both strategies reach the same light. See the
        // twin note in backward.h.
        gmis.clear();
        hairArrival = false;

        switch (m.type) {
            case MatType::Diffuse:
            case MatType::DiffuseTransmit:
            case MatType::Fluorescent: {
                // Density estimate at y, folding the visible-point reflectance per photon
                // wavelength: L_o(vis) += rho(vis,l_p) * [rho(y,l_p)/pi] * Phi_p / (pi r^2 N).
                // `nrmOut` comes back as the map's fixed normalisation unless the map gathers
                // at a PER-QUERY radius (PhotonMap::adaptiveRadius — the caustic map does),
                // in which case 1/(pi r_q^2 N) is recomputed for this query's own radius.
                auto est = [&](const PhotonMap& M, double normFixed, double& nrmOut) {
                    MStatTimer _t(&mStats().surfNs, &mStats().surfN);
                    const double rq = M.adaptiveRadius(h.p, h.n);
                    nrmOut = normFixed;
                    if (rq != M.radius) {
                        const double a = PI * rq * rq;
                        nrmOut = (M.nEmitted > 0 && a > 0.0)
                                     ? 1.0 / (a * (double)M.nEmitted) : 0.0;
                    }
                    // M-GATHERAREA: divide by the area actually gathered from, not by the whole
                    // disc. `gatherAreaSamples() == 0` (the default) returns coverage 1 and
                    // leaves nrmOut untouched, so every existing render is bit-identical.
                    if (const int gaM = gatherAreaSamples())
                        nrmOut *= gatherAreaScale(
                            gatherCoverage(scene, h.p, h.n, rq, rng, gaM, h.matId,
                                           h.fiberRadius));
                    Vec3 g{0, 0, 0};
                    M.queryR(h.p, rq, [&](const Photon& ph, double, int k) {
                        if (dot(ph.n, h.n) < 0.5) return;    // reject cross-surface leakage
                        double rhoY = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                        double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, ph.lambda));
                        double f = rhoY * (1.0 / PI);
                        g += M.cie[k] * (f * rhoV * (double)ph.power);  // == cie(lambda_p), precomputed
                    });
                    return g;
                };
                double nA = norm;
                L += est(pm, norm, nA) * (nA * thr);
                if (pmC && !pmC->photons.empty()) {
                    double nB = normC;
                    L += est(*pmC, normC, nB) * (nB * thr);
                }
                return L;
            }
            case MatType::Mirror: {
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
            case MatType::Glossy: {
                // NEXT-EVENT ESTIMATION AT A GLOSSY VERTEX (GLOSSY-NEE). The gather ray folds
                // the VISIBLE point's reflectance into everything it reports, so `rhoV`
                // multiplies the connection exactly as it multiplies the emission above.
                // Taken before `thr *= r`: the connection carries `r` inside bsdfF.
                if (gneeOn) {
                    const double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, lambda));
                    const BackwardRenderer::NeeBsdf nb{&m, ray.d * -1.0};
                    L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                         * (thr * rhoV * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng,
                                                        nullptr, BackwardRenderer::GiCtx{}, nullptr, nullptr, &nb));
                }
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0.0) return L;
                if (gneeOn) {
                    gmis.pdf = bdpt::bsdfPdf(m, h.n, ray.d * -1.0, o, lambda, scene, &h);
                    gmis.from = h.p;
                    gmis.n = h.n;
                }
                ray = Ray{h.p + h.n * 1e-6, o};
                break;
            }
            case MatType::Dielectric: {
                // Nested-dielectric PRIORITY resolution (Schmidt & Budge 2002): exterior
                // IOR = the medium the photon is currently inside (highest-priority stack
                // entry). Overlapping dielectrics ranked by `priority` (higher wins; lower
                // suppressed -> straight pass-through). SAFE FALLBACK to flat air<->glass
                // unless BOTH sides carry an explicit priority (priority-free scenes stay
                // bit-identical).
                bool entering = dot(ray.d, h.ng) < 0.0;
                const int mi = (int)(&m - scene.mats.data());   // true index (Mix/Layered aware)
                const int pr = m.priority;
                if (entering) {
                    const int outMat = stk.topMat();
                    const int outPri = stk.topPri();
                    const bool ranked = m.hasPriority() &&
                        (stk.empty() || (outMat >= 0 && scene.mats[outMat].hasPriority()));
                    if (ranked && !stk.empty() && pr <= outPri) {   // suppressed inner surface
                        stk.push(mi, pr);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    } else {
                        const double extIor = (ranked && outMat >= 0)
                            ? scene.mats[outMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                        if (transmitted) stk.push(mi, pr);
                    }
                } else {
                    MediumStack after = stk; after.popMat(mi);
                    const int newMat = after.topMat();
                    const int newPri = after.topPri();
                    const bool ranked = m.hasPriority() &&
                        (after.empty() || (newMat >= 0 && scene.mats[newMat].hasPriority()));
                    if (ranked && newMat >= 0 && pr <= newPri) {    // suppressed: still enclosed
                        stk.popMat(mi);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    } else {
                        const double extIor = (ranked && newMat >= 0)
                            ? scene.mats[newMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                        if (transmitted) stk.popMat(mi);            // TIR stays inside mi
                    }
                }
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() < r) ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                else                   ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Filter: {
                thr *= clamp01(transmitSlot(scene, m, h, lambda));
                ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Hair: {
                // Fiber BCSDF — scatter through it, but do NOT gather here: the photon
                // payload carries no incident direction, so a directional BCSDF density
                // estimate is impossible (see sppm_render.h's Hair case and
                // known-issues.md). Treated like the glossy/specular cases around it.
                const Vec3 wPrev{-ray.d.x, -ray.d.y, -ray.d.z};
                const HairShade hs = hairShadeAt(scene, m, h, lambda, wPrev);
                if (hairNeeOn()) {                          // HAIR-NEE: see photonGather's Hair case
                    // every term this sub-walk reports carries the visible point's albedo
                    const double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, lambda));
                    L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                         * (thr * rhoV * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng, nullptr,
                                                 BackwardRenderer::GiCtx{}, &hs, nullptr, nullptr));
                    hairArrival = true;
                }
                double pdfH = 0.0, fv = 0.0;
                const Vec3 wl = hair::sample(hs.b, hs.woLocal, rng.uniform(), rng.uniform(),
                                             rng.uniform(), rng.uniform(), pdfH, fv);
                if (!(pdfH > 0.0) || !(fv > 0.0)) return L;
                const double cosLong =
                    hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
                thr *= clamp01(fv * cosLong / pdfH);       // == T = sum_p A_p
                const Vec3 wo = hair::toWorld(hs.fr, wl);
                ray = Ray{h.p + wo * hairExitOffset(hs, h.n, wo), wo};
                break;
            }
            default: {                                   // ThinFilm/Multilayer/Grating: approx reflect
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
        }
        specularSeen = true;   // only specular cases reach here (diffuse/light returned above)
        if (thr <= 0.0) return L;
    }
    return L;
}

// ---- Camera gather: radiance along one camera ray via the photon map ----------------
// Returns the XYZ contribution. The ray is followed through specular surfaces
// (monochromatic at a sampled lambda); at the first diffuse/translucent hit the reflected
// radiance is estimated by a radius query, with each photon reflected at ITS OWN
// wavelength (density estimate built directly in XYZ). Directly-viewed emitters and the
// environment are added as a monochromatic estimate at the sampled lambda.
//
// PARTICIPATING MEDIA: when `bm` is non-null the walk also (a) gathers volume single scatter
// from the beam map along every segment it travels and (b) attenuates its throughput by the
// media transmittance of that segment, so fog correctly dims the surfaces and sky behind it.
// With `bm` null the walk is media-blind — which is what mode M has always been, and why a
// fog / rain / cloud scene renders its volume as nothing without -beams.
//
// `pmC` (optional) is the CAUSTIC map (see tracePhotonPass): a disjoint half of the same
// deposits, gathered at its own much finer radius and simply added.
// Media transmittance of one straight CAMERA segment: one ratio-tracking sample, or
// pbeams::kSunDiscTrSamples of them when the segment points into a sun's disc (see the
// constant's note). Device twin: dCamMediaTr.
inline double camMediaTr(const Scene& scene, const Renderer& mats, const Vec3& o, const Vec3& d,
                         double len, double lambda, Pcg32& rng) {
    const int n = (scene.sunCount > 0 && scene.sunRadiance(d, lambda) > 0.0)
                      ? pbeams::kSunDiscTrSamples : 1;
    if (n == 1) return mats.mediaTransmittance(scene, o, d, len, lambda, rng);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += mats.mediaTransmittance(scene, o, d, len, lambda, rng);
    return s / (double)n;
}

// `lambdaU` >= 0 is a stratified uniform for the hero wavelength (renderPhotonCamera); < 0 draws
// it from `rng` as before.
inline Vec3 photonGather(const Scene& scene, const PhotonMap& pm, Ray ray,
                         Pcg32& rng, bool diffraction, int maxBounce, int fgRays = 0,
                         const BeamMap* bm = nullptr, const PhotonMap* pmC = nullptr,
                         double lambdaU = -1.0) {
    Vec3 L{0, 0, 0};
    double thr = 1.0;
    double pdfL = 0.0;
    double lambda = (lambdaU >= 0.0) ? scene.emitSampler.sampleAt(lambdaU, pdfL)
                                     : scene.emitSampler.sample(rng, pdfL);
    if (pdfL <= 0.0) return L;
    const double invPdfL = scene.invPdfLambda(lambda);

    Renderer mats; mats.diffraction = diffraction;
    MediumStack stk;                                     // nested-dielectric medium stack
    const double area = PI * pm.radius * pm.radius;
    const double norm = (pm.nEmitted > 0 && area > 0.0)
                            ? 1.0 / (area * (double)pm.nEmitted) : 0.0;
    // The caustic map carries its OWN radius (chosen by its own buildAuto over its own,
    // far denser, population) and therefore its own 1/(pi r^2 N). When it also gathers
    // PER QUERY (PhotonMap::kGather > 0) this is only the fallback: each gather recomputes
    // the normalisation for the radius it actually used. See PhotonMap::adaptiveRadius.
    const bool causOn = (pmC != nullptr) && !pmC->photons.empty();
    const double areaC = causOn ? PI * pmC->radius * pmC->radius : 0.0;
    const double normC = (causOn && pmC->nEmitted > 0 && areaC > 0.0)
                            ? 1.0 / (areaC * (double)pmC->nEmitted) : 0.0;

    SpecThr sthr;    // SPECGATHER: the walk's spectral factors, beside the scalar `thr`
    const bool volOn = (bm != nullptr) && !bm->empty() && !scene.media.empty();
    // -sunnee: the sun's single scatter is marched, not gathered (beamgather.h sunNeeMarch).
    // Not gated on the map being non-empty: erasing the direct-sun chords may have emptied it.
    const bool sunOn = (bm != nullptr) && pbeams::gSunNee && scene.sunCount > 0 && !scene.media.empty();
    // GRADIENT-INDEX: the CAMERA ray has to bend too. Mode M's forward deposit has marched
    // since GRIN landed, but this gather called closestHit directly, so a GRIN lens bent the
    // photons and not the view: the lens rendered dead flat while mode R lensed the same
    // scene into a radial disc, and nothing warned. Marching here is what makes mode M see
    // its own geometry.
    const bool grinAny = grin::sceneHasGrin(scene);


    // GLOSSY-NEE. `bwNee` is the shared direct-lighting estimator (its `neeLight` is what the
    // final gather already uses); `gmis` carries the lobe density of a glossy continuation to
    // whichever emitter site it reaches, and is cleared at the top of every bounce so a mirror
    // or a dielectric can never inherit one and halve the emission behind it.
    BackwardRenderer bwNee; bwNee.diffraction = diffraction;
    const bool gneeOn = BackwardRenderer::glossyNeeOn();
    BackwardRenderer::GlossyMis gmis;
    bool hairArrival = false;   // HAIR-NEE: the previous vertex was a fiber that already took its direct light
    for (int b = 0; b < maxBounce; ++b) {
        const int    cmIdx  = stk.topMat();
        const double aGlass = (cmIdx >= 0) ? scene.mats[cmIdx].absorb(lambda) : 0.0;
        // Curved pre-pass. The volume estimator runs PER STRAIGHT SUB-SEGMENT of the curve:
        // Beam x Ray is a closest-approach between two straight lines, so a curved camera
        // ray has to be fed to it one Eikonal step at a time. That is exact rather than an
        // approximation — the radiance integral along a path is additive over its pieces,
        // and `thr` carries each piece's transmittance forward, which is precisely what
        // gatherPhotonBeams' own per-beam Tr_cam(0 -> tCam) expects (it measures from the
        // sub-segment start; the accumulated `thr` supplies everything before it).
        if (grinAny) {
            grin::marchSegments(scene, ray,
                [&](const Vec3& so, const Vec3& sd, double slen, double&) -> bool {
                    if (volOn || sunOn) {
                        if (volOn) { MStatTimer _t(&mStats().beamNs, &mStats().beamN);
                          L += gatherPhotonBeams(scene, mats, *bm, so, sd, slen, aGlass, rng) * thr; }
                        if (sunOn) L += sunNeeMarch(scene, mats, so, sd, slen, aGlass, lambda, invPdfL, rng) * thr;
                        {   double Tv[SpecThr::K + 1];
                            camMediaTrSpec(scene, mats, so, sd, slen, lambda, rng, Tv);
                            thr *= Tv[SpecThr::K];
                            sthr.mulVec(Tv, Tv[SpecThr::K]);
                        }
                    }
                    if (aGlass > 0.0) {
                        thr *= std::exp(-aGlass * slen);
                        const int cmA = cmIdx; const double sl = slen;
                        sthr.mul([&](double L) { return std::exp(-scene.mats[cmA].absorb(L) * sl); },
                                 std::exp(-aGlass * slen));
                    }
                    return false;   // a camera ray never terminates in the volume here:
                                    // mode M's volume answer IS the beam gather above
                },
                // b == 0 is the camera ray; see Material::hideCamera. (photonGatherSub's
                // march above is a final-gather sub-ray and keeps the default `false`.)
                /*camHide=*/(b == 0));
            if (thr <= 0.0) return L;
        }

        // b == 0 is the camera ray photonGather was handed (mode M's eye pass); see
        // Material::hideCamera. photonGatherSub's walk is NOT given this: a final-gather
        // sub-ray leaves a visible point, so it is an indirect ray and must see the flat.
        Hit h = scene.closestHit(ray, 1e-6, nullptr, /*skipHair=*/false,
                                 /*skipCamHidden=*/(b == 0));
        // --- Participating media along this segment (mode M with -beams) ---------------
        // Done BEFORE `thr` takes the segment's attenuation, because each gathered beam
        // needs the transmittance to ITS OWN closest-approach point, not to the segment end.
        if (volOn || sunOn) {
            const double dSeg = h.valid ? h.t : 1e30;
            if (volOn) { MStatTimer _t(&mStats().beamNs, &mStats().beamN);
              L += gatherPhotonBeams(scene, mats, *bm, ray.o, ray.d, dSeg, aGlass, rng)
                   * thr;
              }
            if (sunOn) L += sunNeeMarch(scene, mats, ray.o, ray.d, dSeg, aGlass, lambda, invPdfL, rng) * thr;
            // Extinction along the camera segment: what is behind the fog gets dimmed.
            {   // SPECGATHER: extinction is spectral too, and with a coloured medium badly so.
                double Tv[SpecThr::K + 1];
                camMediaTrSpec(scene, mats, ray.o, ray.d, dSeg, lambda, rng, Tv);
                thr *= Tv[SpecThr::K];
                sthr.mulVec(Tv, Tv[SpecThr::K]);
            }
            if (thr <= 0.0) return L;
        }
        if (h.valid) {                                   // Beer-Lambert in current medium
            if (aGlass > 0.0) {
                thr *= std::exp(-aGlass * h.t);      // coloured glass is spectral: SPECGATHER too
                const int cmA = cmIdx; const double tt = h.t;
                sthr.mul([&](double L) { return std::exp(-scene.mats[cmA].absorb(L) * tt); },
                         std::exp(-aGlass * h.t));
            }
        }
        if (!h.valid) {                                  // escaped -> environment
            if (scene.envIndex >= 0)
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * scene.envRadiance(ray.d, lambda) * invPdfL);
            // Directly-viewed solar disc. This walk terminates at the first diffuse
            // vertex (the density estimate returns there), so any escape reaching here
            // is a camera ray or a specular chain — never a diffuse continuation that
            // the map / NEE already credited with the sun.
            if (scene.sunCount > 0)
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * bwNee.sunRadianceMis(scene, gmis, ray.d, lambda) * invPdfL);
            return L;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return L;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            // Resolve to a LOBE and let the material switch shade it -- the coat with
            // probability R (the Fresnel reflectance at this angle), else a body lobe. The
            // coat is a material of its own (Material::coatChild), so it goes through the
            // switch's Glossy case and gets that case's glossy-NEE, throughput and MIS. An
            // earlier version reflected the ray inline here and skipped all three, which
            // broke the identity that a coat of reflectance 1 IS a glossy material: it
            // rendered 30 % bright against one (scraps/ident_*.ftsl). Same convention as
            // bdpt.h's randomWalk and the device's dResolveCompound.
            const double R = layeredCoatReflectance(scene, *mp, h, ray.d, lambda);
            int c = (rng.uniform() < R) ? mp->coatChild
                                        : mixResolveChild(scene, *mp, h, rng.uniform());  // body lobe: honours a bound weight map (device twin: dResolveCompound)
            if (c < 0) return L;
            mp = &scene.mats[c];
        }
        const Material& m = *mp;

        // Self-emission of a directly-viewed (or specularly-seen) emitter, one-sided by the
        // geometric normal to match Vertex::Le / bkRadiance — the surface glows only from
        // the face cross(u,v) points out of.
        //
        // NOT a `return`. An emissive material still has a BSDF: a glowing DIFFUSE surface
        // both emits and reflects, so the walk falls through to the density estimate below.
        // Returning here is what made gallery_rain's grid floor render as GRID-ONLY on the
        // CPU (host `m.isLight` hit, emission returned, body dropped) and as BODY-ONLY on the
        // GPU (dEmitterForMat missed the unregistered quad, so the emitter branch was never
        // taken and the grid vanished) — two mode-M paths disagreeing with each other and
        // both disagreeing with modes R and D. Measured on scraps/mini_grid.ftsl.
        if (m.isLight && dot(ray.d, h.ng) < 0.0 && !hairArrival) {
            const double wMis = (gmis.pdf > 0.0)          // GLOSSY-NEE, as in the sub-walk
                ? bwNee.glossyHitWeight(scene, gmis,
                        BackwardRenderer::emitterIndexOfResolved(scene, m), ray.d, &h.p, &h.n)
                : 1.0;
            L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                 * (thr * emitSlot(scene, m, h, lambda) * invPdfL * wMis);
        }

        // GLOSSY-NEE: cleared HERE and not at the top of the loop. `gmis` is written by the
        // PREVIOUS bounce's glossy branch and read by THIS bounce's emitter/sun sites above, so
        // a clear at the loop top erases it a few lines before the only code that wants it --
        // which leaves the connection in place with no compensating weight on the lobe-sampling
        // side, i.e. double counting wherever both strategies reach the same light. See the
        // twin note in backward.h.
        gmis.clear();
        hairArrival = false;

        switch (m.type) {
            case MatType::Diffuse:
            case MatType::DiffuseTransmit:
            case MatType::Fluorescent: {
                if (fgRays > 0 && m.type == MatType::Diffuse) {
                    // --- Jensen final gather (decouples visible-surface sharpness from
                    // the gather radius: the density estimate's blur now lives one bounce
                    // away, at y, not on this directly-seen surface). ---
                    // (a) DIRECT lighting from finite emitters via low-variance next-event
                    //     estimation (shadow rays), so we avoid the high variance of gather
                    //     rays randomly striking a small area light.
                    //     `neeLight` carries the shadow leg's media transmittance over the
                    //     WHOLE superposed `scene.media` vector. Until 0.254.0 it applied one
                    //     unbounded homogeneous haze built from media.front() instead, which
                    //     in a scene whose first medium is a dense bounded cloud (gallery_rain:
                    //     sigma_t 2.78, a 3 m box) multiplied every 10-30 m shadow ray by
                    //     exp(-28)..exp(-83) and deleted mode M's ENTIRE direct term (M-FGDARK).
                    BackwardRenderer bw; bw.diffraction = diffraction;
                    double rhoVis = clamp01(diffuseReflectance(scene, m, h, lambda));
                    double direct = bw.neeLight(scene, h, rhoVis, invPdfL, lambda, rng);
                    L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * (thr * direct);
                    // (b) INDIRECT (+ env + specular-direct) via K cosine-weighted
                    //     hemisphere sub-rays, each querying the map ONE bounce away. The
                    //     cosine/pdf and Lambertian 1/pi cancel to rho(x), folded inside
                    //     photonGatherSub; those rays skip non-specular emitter hits so the
                    //     NEE direct term above is not double-counted.
                    Vec3 fg{0, 0, 0};
                    for (int k = 0; k < fgRays; ++k) {
                        Ray gr{h.p + h.n * 1e-6, cosineHemisphere(h.n, rng)};
                        fg += photonGatherSub(scene, pm, gr, rng, diffraction, maxBounce,
                                              lambda, invPdfL, norm, h, m,
                                              causOn ? pmC : nullptr, normC);
                    }
                    L += fg * (thr * (1.0 / (double)fgRays));
                    return L;
                }
                // Direct radius density estimate (default; also DiffuseTransmit/Fluorescent
                // visible points, which fall back here rather than final-gathering):
                //   L_r(x) = (1/N) sum_p f_r * Phi_p / (pi r^2), f_r = rho/pi (Lambertian),
                // accumulated in XYZ per photon wavelength.
                // Per-query adaptive radius on any map that asks for one (the caustic map);
                // see the twin in photonGatherSub and PhotonMap::adaptiveRadius.
                auto est = [&](const PhotonMap& M, double normFixed, double& nrmOut) {
                    MStatTimer _t(&mStats().surfNs, &mStats().surfN);
                    const double rq = M.adaptiveRadius(h.p, h.n);
                    nrmOut = normFixed;
                    if (rq != M.radius) {
                        const double a = PI * rq * rq;
                        nrmOut = (M.nEmitted > 0 && a > 0.0)
                                     ? 1.0 / (a * (double)M.nEmitted) : 0.0;
                    }
                    // M-GATHERAREA: divide by the area actually gathered from, not by the whole
                    // disc. `gatherAreaSamples() == 0` (the default) returns coverage 1 and
                    // leaves nrmOut untouched, so every existing render is bit-identical.
                    if (const int gaM = gatherAreaSamples())
                        nrmOut *= gatherAreaScale(
                            gatherCoverage(scene, h.p, h.n, rq, rng, gaM, h.matId,
                                           h.fiberRadius));
                    Vec3 g{0, 0, 0};
                    M.queryR(h.p, rq, [&](const Photon& ph, double, int k) {
                        if (dot(ph.n, h.n) < 0.5) return;    // reject cross-surface leakage
                        double rho = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                        double f = rho * (1.0 / PI);
                        // SPECGATHER: the walk's reflectances belong at THIS photon's wavelength,
                        // not the camera's. Exactly 1 when they were all flat.
                        f *= sthr.ratio((double)ph.lambda);
                        g += M.cie[k] * (f * (double)ph.power);       // == cie(lambda_p), precomputed
                    });
                    return g;
                };
                double nA = norm;
                L += est(pm, norm, nA) * (nA * thr);
                if (causOn) { double nB = normC; L += est(*pmC, normC, nB) * (nB * thr); }
                return L;
            }
            case MatType::Mirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
                sthr.mul([&](double L) { return clamp01(reflectSlot(scene, m, h, L)); }, r);
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
            case MatType::Glossy: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                // NEXT-EVENT ESTIMATION AT A GLOSSY VERTEX (GLOSSY-NEE). See backward.h's twin
                // for why this is MIS and not the single-estimator split the rest of mode M's
                // walk uses: a glossy lobe can be narrower than the light as easily as wider.
                if (gneeOn) {
                    const BackwardRenderer::NeeBsdf nb{&m, ray.d * -1.0};
                    if (specNeeOn()) {
                        // SPECTRAL (0.341.0): one shadow ray, evaluated over the SpecThr grid,
                        // so a gold lobe under a warm light stops painting each sample one
                        // colour. The walk's own spectral throughput rides in through `ratio`.
                        double lamG[SpecThr::K], ratG[SpecThr::K], xyz[3] = {0, 0, 0};
                        for (int k = 0; k < SpecThr::K; ++k) {
                            lamG[k] = SpecThr::lamOf(k);
                            ratG[k] = sthr.ratio(lamG[k]);
                        }
                        const double dLam = (LAMBDA_MAX - LAMBDA_MIN) / (double)SpecThr::K;
                        bwNee.neeLightSpecGlossy(scene, h, nb, lamG, ratG, SpecThr::K, dLam,
                                                 lambda, thr, rng, xyz);
                        L += Vec3(xyz[0], xyz[1], xyz[2]);
                    } else {
                        L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                             * (thr * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng,
                                                     nullptr, BackwardRenderer::GiCtx{}, nullptr, nullptr, &nb));
                    }
                }
                thr *= r;
                sthr.mul([&](double Lw) { return clamp01(reflectSlot(scene, m, h, Lw)); }, r);
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0.0) return L;
                if (gneeOn) {
                    gmis.pdf = bdpt::bsdfPdf(m, h.n, ray.d * -1.0, o, lambda, scene, &h);
                    gmis.from = h.p;
                    gmis.n = h.n;
                }
                ray = Ray{h.p + h.n * 1e-6, o};
                break;
            }
            case MatType::Dielectric: {
                // Nested-dielectric PRIORITY resolution (Schmidt & Budge 2002): exterior
                // IOR = the medium the photon is currently inside (highest-priority stack
                // entry). Overlapping dielectrics ranked by `priority` (higher wins; lower
                // suppressed -> straight pass-through). SAFE FALLBACK to flat air<->glass
                // unless BOTH sides carry an explicit priority (priority-free scenes stay
                // bit-identical).
                bool entering = dot(ray.d, h.ng) < 0.0;
                const int mi = (int)(&m - scene.mats.data());   // true index (Mix/Layered aware)
                const int pr = m.priority;
                if (entering) {
                    const int outMat = stk.topMat();
                    const int outPri = stk.topPri();
                    const bool ranked = m.hasPriority() &&
                        (stk.empty() || (outMat >= 0 && scene.mats[outMat].hasPriority()));
                    if (ranked && !stk.empty() && pr <= outPri) {   // suppressed inner surface
                        stk.push(mi, pr);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    } else {
                        const double extIor = (ranked && outMat >= 0)
                            ? scene.mats[outMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                        if (transmitted) stk.push(mi, pr);
                    }
                } else {
                    MediumStack after = stk; after.popMat(mi);
                    const int newMat = after.topMat();
                    const int newPri = after.topPri();
                    const bool ranked = m.hasPriority() &&
                        (after.empty() || (newMat >= 0 && scene.mats[newMat].hasPriority()));
                    if (ranked && newMat >= 0 && pr <= newPri) {    // suppressed: still enclosed
                        stk.popMat(mi);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    } else {
                        const double extIor = (ranked && newMat >= 0)
                            ? scene.mats[newMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                        if (transmitted) stk.popMat(mi);            // TIR stays inside mi
                    }
                }
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() < r) ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                else                   ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Filter: {
                double t = clamp01(transmitSlot(scene, m, h, lambda));
                thr *= t;
                sthr.mul([&](double L) { return clamp01(transmitSlot(scene, m, h, L)); }, t);
                ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Hair: {
                // Fiber BCSDF — scattered through, never a gather site (the photon
                // payload has no incident direction; see photonGatherSub's Hair case).
                const Vec3 wPrev{-ray.d.x, -ray.d.y, -ray.d.z};
                const HairShade hs = hairShadeAt(scene, m, h, lambda, wPrev);
                // HAIR-NEE (0.334.0). This walk collects radiance only where it finally gathers
                // -- a diffuse surface -- and a chain that scatters strand to strand through a
                // hair mass lands on the shadowed scalp or dress beneath it, so the mass's own
                // lit glow (direct light scattered by the fibers toward the eye) was never
                // counted: Alice's strands rendered 30 % darker than mode D's at 20 M photons
                // while the molded base matched. Mode R's Hair case connects to the lights at
                // every fiber vertex with the BCSDF (`neeLight` with `hs`, rho == 1: the colour
                // lives in sigma_a) and does not count an emitter its continuation then hits;
                // the same split here. The photon map is untouched (fibers are never deposited
                // on), and light that scatters off fibers onto a surface is in the map already.
                if (hairNeeOn()) {
                    L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                         * (thr * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng, nullptr,
                                                 BackwardRenderer::GiCtx{}, &hs, nullptr, nullptr));
                    hairArrival = true;
                }
                double pdfH = 0.0, fv = 0.0;
                hair::LobeAngular la;                              // filled by the sample's own f()
                const Vec3 wl = hair::sample(hs.b, hs.woLocal, rng.uniform(), rng.uniform(),
                                             rng.uniform(), rng.uniform(), pdfH, fv, &la);
                if (!(pdfH > 0.0) || !(fv > 0.0)) return L;
                const double cosLong =
                    hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
                const double wCam = clamp01(fv * cosLong / pdfH);   // == T = sum_p A_p
                thr *= wCam;
                // SPECGATHER (0.333.0): the fiber's transmission is the most coloured factor a
                // camera walk meets -- a blonde fiber absorbs blue and green -- and until now the
                // walk folded it in at the camera's wavelength ALONE, so every photon of every
                // wavelength was scaled by one wavelength's attenuation: coloured speckle per
                // sample, and in expectation a flat AVERAGE transmission that greys the hair's
                // multiple-scatter colour. The sampled direction and its pdf are fixed by the
                // sample; only the BCSDF value moves with the absorption, so the same sample's
                // weight is evaluated at every grid wavelength (hairShadeAt inverts the
                // authored reflectance at that wavelength) and each photon is reweighted at ITS
                // wavelength, exactly as coloured glass and media already are (see SpecThr).
                // FTRACE_HAIR_SPECGATHER=0 turns the fold off (the scalar path of 0.332.0), for
                // paired A/B measurements only -- read once per process.
                static const bool hairSpecOn = [] { const char* e = std::getenv("FTRACE_HAIR_SPECGATHER"); return !(e && e[0] == '0'); }();
                if (hairSpecOn) {
                    // Per-lobe form (0.336.0): the angular products once, then per grid wavelength
                    // one absorption inversion, one exp and one Ap() -- a solid fiber's f() at any
                    // wavelength is exactly that (hair.h lobeAngular / fFromLobes). A fiber with a
                    // medulla keeps the full rebuild.
                    const bool perLobe = la.valid;
                    // the absorption per bin: a per-material table when the colour is constant
                    // (hairSigmaBins), else inverted at each wavelength
                    const double* bins = hairSigmaBins<SpecThr::K, &SpecThr::lamOf>(scene, m, h);
                    sthr.mul([&](double lamK) {
                        if (perLobe)
                            return clamp01(hair::fFromLobes(la, bins ? bins[SpecThr::binOf(lamK)] : hairSigmaAAt(scene, m, h, lamK)) * cosLong / pdfH);
                        const HairShade hk = hairShadeAt(scene, m, h, lamK, wPrev);
                        return clamp01(hair::f(hk.b, hk.woLocal, wl) * cosLong / pdfH);
                    }, wCam);
                }
                const Vec3 wo = hair::toWorld(hs.fr, wl);
                ray = Ray{h.p + wo * hairExitOffset(hs, h.n, wo), wo};
                break;
            }
            default: {                                   // ThinFilm/Multilayer/Grating: approx reflect
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
                sthr.mul([&](double Lw) { return clamp01(reflectSlot(scene, m, h, Lw)); }, r);
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
        }
        if (thr <= 0.0) return L;
    }
    return L;
}

// ---- Camera pass driver (single camera) ---------------------------------------------
// Accumulates a SUM over spp (display divides by spp via writeFilm), matching the
// backward/BDPT convention so a chunked/progressive render sums batches.
// `sampleBase` = absolute index of the first sample rendered here; each
// (pixel, absolute sample) seeds its own stream via seedUnit(), so the
// realization is chunk-split / banding / thread-count independent (see
// BackwardRenderer::renderRows).
inline Film renderPhotonCamera(const Scene& scene, const Camera& cam, int resX, int resY,
                               const PhotonMap& pm, long long spp, int nThreads,
                               bool diffraction, int maxBounce = 32,
                               unsigned long long sampleBase = 0, int fgRays = 0,
                               const BeamMap* bm = nullptr,
                               const PhotonMap* pmC = nullptr) {
    if (nThreads < 1) nThreads = 1;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    std::vector<Film> bands(nThreads);
    const uint64_t nPix = (uint64_t)resX * (uint64_t)resY;
    auto worker = [&](int tid) {
        Film& f = bands[tid]; f.resX = resX; f.resY = resY; f.alloc();
        int y0 = resY * tid / nThreads, y1 = resY * (tid + 1) / nThreads;
        for (int py = y0; py < y1; ++py) {
            for (int px = 0; px < resX; ++px) {
                // Cooperative `-stop`, polled once per PIXEL. Without this the gather is
                // uninterruptible: the callers only test the stop flag between whole frames,
                // so a single-camera mode-M render could not be stopped at all, and a
                // pathologically slow gather had to be force-killed — precisely what this
                // project forbids, because tearing down a live CUDA context that way can
                // wedge the display driver. Per PIXEL rather than per scanline because the
                // thing that makes a gather slow enough to want stopping is a slow gather:
                // with an oversized `-beams` kernel radius one scanline can be half a minute,
                // and the poll (one relaxed atomic load) is free against even a fast one.
                // A partially filled band is fine — every caller discards the film on a stop.
                if (ft::stopRequested()) return;
                const uint64_t pixIdx = (uint64_t)py * (uint64_t)resX + (uint64_t)px;
                for (long long s = 0; s < spp; ++s) {
                    Pcg32 rng;
                    seedUnit(rng, (sampleBase + (uint64_t)s) * nPix + pixIdx,
                             0xA24BAED4963EE407ULL);
                    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
                    // STRATIFIED hero wavelength over this chunk's samples (0.314.0; device twin
                    // and the reasoning: kGather in render_cuda.cu). Per chunk rather than over
                    // the whole render because the chunk is all this call can see, and a chunk
                    // stratified over [0,1) is unbiased on its own; the jitter draw stands in for
                    // the draw photonGather would otherwise have made.
                    double lambdaU = -1.0;
                    if (spp > 1) {
                        Pcg32 rrot; seedUnit(rrot, pixIdx, 0x9E3779B97F4A7C15ULL);
                        double u = rrot.uniform() + ((double)s + rng.uniform()) / (double)spp;
                        u -= std::floor(u);
                        lambdaU = (u < 1.0) ? u : 0.999999999;
                    }
                    f.add(px, py, photonGather(scene, pm, ray, rng, diffraction, maxBounce,
                                               fgRays, bm, pmC, lambdaU));
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    for (int t = 0; t < nThreads; ++t) out.merge(bands[t]);
    return out;
}
