// View-independent PHOTON BEAMS for the photon map (mode M).
//
// WHY THIS EXISTS
// ---------------
// The photon map (photonmap.h) is a SURFACE cache: every record is a point deposit at a
// diffuse vertex, and the camera gather is a radius density estimate on a surface. That
// makes mode M blind to participating media — a fog / rain / cloud / rainbow scene renders
// its volume as *nothing* under mode M, because no photon ever deposits inside a medium.
// Modes A/B do see media, and `-beams` (render.h) already gives them a decorrelated
// single-scatter volume gather for flybys — but that one is a camera SPLAT: it needs the
// camera list at photon-trace time and writes straight to each film, which is exactly the
// view-DEPENDENCE mode M exists to avoid. It cannot be reused here.
//
// So the volume needs its own view-independent cache, and a point cache is the wrong shape
// for it: a photon crossing a fog bank interacts *everywhere along its path*, and storing
// only the sampled collision points throws away the whole segment. This file stores the
// SEGMENT — the photon beam (Jarosz et al., "A Comprehensive Theory of Volumetric Radiance
// Estimation", 2011) — so one traced photon lights the entire chord it crossed, for every
// camera, forever after.
//
// THE ESTIMATOR: BEAM x RAY, 1D BLUR
// ----------------------------------
// A camera ray and a photon beam are two segments in space; they (almost) never intersect,
// so the estimate blurs across their mutual perpendicular. For the pair at closest approach
// (camera param t_c, beam param s_b, perpendicular separation d_perp) the single-scatter
// contribution to the camera ray's radiance is
//
//   L += K1(d_perp) / sin(theta) * Phi_b * Tr_beam(0 -> s_b) * sigma_s(x) * f_p(cos theta)
//        * Tr_cam(0 -> t_c)   / nEmitted
//
//   K1        1D kernel, normalised so integral over [-r, r] is 1 (Epanechnikov here)
//   sin theta |cross(d_cam, d_beam)| — the Jacobian of the 1D blur; parallel pairs are a
//             measure-zero singularity and are rejected
//   Phi_b     the beam's carried power (flux) at its stored origin
//   sigma_s   the medium's scattering coefficient AT the gather point (density field and
//             all), so a heterogeneous cloud shapes the bow correctly
//   f_p       the medium's phase function at the scattering angle. THIS is the whole point:
//             a rainbow is a function of the angle between the photon direction and the eye
//             direction, and the beam is the only record in the engine that keeps a photon
//             direction. `struct Photon` deliberately has none (see photonmap.h) — which is
//             correct for a Lambertian surface estimate and useless for a bow.
//
// Dimensionally: [W] * [1/m] * [1/m] * [1/sr] = W/(m^2 sr) = radiance. The 1/nEmitted is the
// same photon-pass normalisation the surface estimate uses; there is no 1/(pi r^2) because
// the 1D kernel already carries its own normalisation.
//
// SINGLE SCATTER ONLY, ON PURPOSE
// -------------------------------
// A beam-depositing photon crosses the medium STRAIGHT (the analog free-flight redirect is
// skipped) and is attenuated by the medium's transmittance over the crossing. So the volume
// gather sees single scattering only; the multiple-scatter wash is omitted. That is the same
// trade `-beams` already makes in modes A/B, and it is the right one for the scenes this
// serves: multiple scattering in a rain volume is a desaturating grey veil that washes the
// bow out, and it is the bow we are here for. Surfaces beyond the fog still get correctly
// dimmed direct light, because the photon's power is reduced by the crossing.
//
// THE RADIUS IS NOT THE PHOTON MAP'S RADIUS — IT IS SMALLER BY ORDERS OF MAGNITUDE
// -------------------------------------------------------------------------------
// This is the one number that will surprise anyone reading this file after working on the
// surface map, and getting it wrong makes the gather unusable rather than merely blurry. A
// beam is a 1D object being blurred in 1D, so the number of beams a camera ray collects
// grows LINEARLY in r and linearly in the total stored beam length:
//
//   E[beams gathered] = (pi/2) * r * L_ray * S / V     (S = total beam length, V = volume)
//
// With a million beams in a one-metre box that is tens of thousands of hits per ray at the
// photon map's radius — the first working version of this file did exactly that, and a
// 200x200 render at 16 spp did not finish. Solving the same expression for r at a target
// count K, and substituting the mean chord of a convex body L_ray = 4V/A, gives
//
//   r = K * A / (2 * pi * S)                            (A = surface area of the beam AABB)
//
// THE RADIUS IS PER MEDIUM, AND IT IS SET BY PHYSICS — NOT BY A TARGET SAMPLE COUNT (0.201.0)
// ------------------------------------------------------------------------------------------
// The expression above is still the right *diagnostic*, but until 0.201.0 it was also the
// RULE: buildAuto solved it for r at a fixed target population K (`-beamk`, default 32) and
// used that one radius for the whole map. That is a trap, and it is worth spelling out because
// it is the sort of self-consistent design that looks correct until you measure it.
//
// S (total stored beam length) is proportional to the photon count n, so pinning K pins
//
//     r  =  K * A / (2 pi S)   ~   1/n
//
// i.e. every extra photon the user pays for is spent SHRINKING THE KERNEL, and the gather
// still averages exactly K beams per camera segment no matter what the budget is. The
// estimator therefore has a hard variance FLOOR that neither `-n` nor `-spp` can move: each
// stored beam is monochromatic (`PhotonBeam::lambda`), a single monochromatic sample is far
// outside sRGB, and averaging ~32 of them per pixel leaves full-saturation colour speckle.
// That is exactly what the gallery_rain cloud looked like ("iridescent"), and the measurements
// that pinned it were:
//
//     -spp 16 -> 64        cloud luminance sd  0.1733 -> 0.1674   (i.e. nothing)
//     -n 40M -> 160M       cloud saturation    0.1157 -> 0.1078   (i.e. nothing), and the log
//                          showed the radius shrinking 2.83e-4 -> 1.65e-4 to hold K at 32
//     -beamk 32 -> 2048    cloud saturation    0.1157 -> 0.0487   (the only thing that worked)
//
// The trade is not merely unhelpful, it is strictly bad, because the bias it buys DOES NOT
// EXIST at these radii. The `_beams_ms` invariant scene (mode M + -beams must converge to the
// mode D reference) gives, at a fixed -n 40000000:
//
//     -beamk       8       32      128      512     2048     8192
//     radius   6.2e-6   2.7e-5   1.0e-4   4.0e-4   1.6e-3   6.5e-3
//     ball/ref  1.0563   1.0816   1.0831   1.0884   1.0841   1.0876
//
// A THOUSAND-fold radius increase moves the invariant by under one point — the residual few
// percent is a constant offset, not kernel bias. The engine was spending its entire photon
// budget on a bias reduction that does not happen, and getting no variance reduction for it.
//
// So the radius is now derived from the medium's own transport scale instead. Each stored beam
// IS a sampled free-flight chord (clipped to the medium's bound), so the medium's mean free
// path is measured directly off the map with no scene access, no wavelength choice, and exact
// handling of heterogeneity:
//
//     mfp_m = S_m / N_m        (total pre-split beam length / beam count, for medium m)
//     r_m   = beamBlur * mfp_m         (`-beamblur`, default 0.01)
//
// r_m is a fixed fraction of the distance over which the medium's own radiance field varies,
// so the blur it introduces is bounded by physics and is INDEPENDENT of n — which is the whole
// point: the gathered count now grows linearly in n, and `-n` finally buys noise reduction.
// It is per medium because a scene can hold a dense cloud and a sparse rain curtain at once,
// and one global radius is simultaneously too blurry for one and too noisy for the other.
// Sanity, MEASURED after the fact on both test scenes (the constant is fitted to neither):
// gallery_rain's two media report mfp 1.29 m / 1.73 m -> r 12.9 mm / 17.3 mm, which sits just
// under the -beamk 2048 radius of 18.6 mm that the saturation sweep above found good;
// `_beams_ms` reports mfp 0.281 m -> r 2.81 mm, between its bias-flat -beamk 2048 (1.61 mm) and
// -beamk 8192 (6.5 mm). And the measured mfp is STABLE — 1.292 m / 1.727 m at both -n 40M and
// -n 160M, to four digits — which is the check that it is a property of the medium rather than
// of the sampling.
//
// MEASURED RESULT (gallery_rain, 320x180, -spp 16, -n 40M, same photons as the old default):
// cloud chroma saturation 0.1157 -> 0.0533, at unchanged mean luminance. And -n now WORKS: at
// -n 160M it falls further to 0.0419 (-21%), against -6.8% under the old rule. The residual gap
// to the ideal 1/sqrt(n) is `-beamcount 1e6` capping the stored beams at 1.87x rather than 4x,
// which is that knob doing exactly what it says. The `_beams_ms` invariant is unmoved: ball/ref
// 1.0899 against 1.0816..1.0884 across the whole -beamk sweep — no new bias.
//
// `-beamk` survives as a FLOOR, not a target: a map so sparse that a camera segment gathers
// almost nothing degenerates into visible individual streaks, so buildAuto probes the gathered
// count at the mfp radii and scales them UP (never down) if it is under `-beamk`. And
// `-beamareaslack` is the ceiling: the radius inflates every sub-beam's AABB and so costs
// traversal time, so the radii are capped at the point where the total box area has grown by
// that fraction over the tight (r = 0) area. Bias bounded by physics, cost bounded by area,
// and nothing in between pinned to a sample count.
//
// The closed form is still used, by buildAuto's PROBE (brute force over a strided subsample,
// no BVH needed) which measures the gathered count directly — it has to, because the closed
// form assumes isotropic relative orientation and beams spread evenly through their bounding
// box, and neither holds in a sunlit rain volume where every beam is near-parallel to the sun.
//
// BEAM SPLITTING
// --------------
// A long diagonal beam has a terrible AABB — mostly empty space that every passing ray must
// test. build() therefore splits long beams into sub-segments. A sub-segment keeps the
// ORIGINAL origin `o` and records its own [s0, s0+len] parameter range, so the gather can
// still evaluate transmittance from the true beam start (which is where the stored power
// applies) without the splitter needing a Scene or an RNG to re-integrate it.
//
// WHERE to split is chosen by minimising surface area, NOT by scene scale — see sahSplitLen()
// below for the derivation. The short version: cost is the number of AABBs a ray must enter,
// that count is proportional to total box area (Cauchy), the area of a split beam is
// 2L(Qp + 4rE + 12r²/p), and it bottoms out at p* = 2r√(3/Q). The consequence that matters is
// that p* is O(r) with no scene-scale term at all, and the resulting total area is linear in
// S·r — which buildAuto holds CONSTANT, so traversal cost stops growing with the stored beam
// count. The rule this replaced, max(diag/64, S/3N), was two scene-scale terms that measurably
// did not move across a 10× beam-count sweep, which is precisely why cost was linear in beams.
//
// MEMORY / BEAM SUBSAMPLING
// -------------------------
// A beam record is ~72 B (plus a 24 B CIE triple) and a dense photon pass emits tens of
// millions of photons, so storing a beam per crossing would run to gigabytes. It is also
// unnecessary: a beam lights a whole chord, so far fewer beams than photons are needed for
// the same volume variance. The deposit therefore keeps a beam with probability `p` and
// scales its power by 1/p (Russian roulette — unbiased).
//
// `p` is MEASURED, not predicted: the bank keeps everything until it hits a memory cap, then
// halves itself and its own deposit rate. Picking `p` up front from the photon count is the
// obvious approach and it is wrong — see the long note on BeamBank for the two scenes that
// disproved it in opposite directions.
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "linalg.h"
#include "rng.h"
#include "color.h"
#include "bvh.h"
#include "allocreport.h"   // OOM that names the buffer, its size and the flag that sizes it

// MULTIPLE-SCATTERING ORDER CAP (CLI -beams-order), as a header-inline global for the same
// reason `hero::gSplit` is one: it has to be visible to BOTH the host renderer (render.h) and
// the CUDA translation unit, which do not share main.cpp's statics.
//
//   0 = unlimited (the default; bounded only by -bounces)
//   1 = single scatter, the pre-0.199.0 behaviour, bit-identical
//   n = orders 1..n
//
// See the `beamMS` block in Renderer::tracePhoton for what the number means and why LONG beams
// make multiple scattering a small change to the transport.
namespace pbeams { inline int gOrderMax = 0; }

// One stored photon beam: (a sub-segment of) the path a photon travelled through one medium.
//
// Unlike `Photon`, this record DOES carry a direction — it has a reader (the phase
// function), which is the test photonmap.h asks of any field it stores.
struct PhotonBeam {
    Vec3  o;         // TRUE segment start (world), clipped to the medium's bound. `power`
                     // is the flux here, and beam-side transmittance is measured from here
                     // — which is why splitting keeps `o` and moves `s0` instead.
    Vec3  d;         // unit direction of travel
    float s0;        // this sub-segment's start, as a distance along the beam from `o`
    float len;       // this sub-segment's length (world units)
    float power;     // carried power (flux) at `o`, after Russian-roulette rescaling
    float lambda;    // wavelength (nm) — monochromatic, like a photon
    float absorb;    // sigma_a of the enclosing DIELECTRIC (glass) the beam runs inside,
                     // so the gather can Beer-Lambert to its own closest-approach point
                     // instead of only to the segment end. 0 in air.
    int   med;       // index into Scene::media — which medium this beam lights. One beam
                     // per (segment, medium) pair: overlapping media superpose (each is an
                     // independent Poisson process; see Renderer::sampleMediaCollision), so
                     // each contributes its own sigma_s and its own phase function.

    Vec3 begin() const { return o + d * (double)s0; }
    Vec3 end()   const { return o + d * ((double)s0 + (double)len); }
};

// A per-thread beam bank, appended to during the photon pass and concatenated into the
// BeamMap afterwards. Mirrors PhotonBank's role for surface deposits.
//
// THE BANK THINS ITSELF; THE BUDGET IS NOT GUESSED UP FRONT. This is the second design here
// and the first one was wrong in a way worth recording, because the obvious fix is the wrong
// one. Originally `keepProb` was computed once from the PHOTON count as `target/nPhotons`,
// on the assumption that a photon deposits about one beam. Measured, that assumption is off
// by two orders of magnitude in BOTH directions:
//
//   * `_fog_cornell` — one UNBOUNDED medium, so every photon crosses it and a bouncing one
//     crosses repeatedly: 2.25 M crossings against a 1 M budget, a 2.3x OVERSHOOT. Left
//     unchecked this is what allocates gigabytes.
//   * `gallery_rain` — two small BOUNDED media (a rain box and a voxelized cloud) in a large
//     hall, so only ~0.8% of photons ever enter one: 7.6 k beams against that same 1 M
//     budget, a 131x UNDERSHOOT. That is not merely wasteful, it is VISIBLE — with 7.6 k
//     beams the rain curtain renders as individual streaks rather than as a volume, i.e. the
//     estimator is starved of exactly the samples the user asked for.
//
// No closed form fixes this, because the beams-per-photon ratio depends on the media's
// volume fraction and the photons' bounce depth, neither of which is knowable before the
// pass. So don't predict it — MEASURE it, by keeping everything until memory actually says
// stop:
//
//   push() appends unconditionally; when the bank reaches `cap` it HALVES itself in place
//   (each beam survives with p = 1/2, survivors' power doubled) and halves `keepProb`, which
//   the depositor reads to Russian-roulette its future pushes at the new rate.
//
// Every beam is scaled by the reciprocal of its own inclusion probability, so this is
// unbiased at every stage, and it is the standard adaptive-reservoir trick rather than
// anything exotic. The result: a scene that deposits little keeps all of it (gallery_rain
// now stores what it actually generates instead of 0.8% of a guess), and a scene that
// deposits torrentially is bounded by `cap` no matter how badly it overshoots.
//
// `cap` is per-thread (the banks concatenate), so the caller sets it to its global budget
// divided across threads and applies one final BeamMap::decimateTo for the exact trim.
struct BeamBank {
    std::vector<PhotonBeam> beams;
    double   keepProb = 1.0;    // current RR survival rate, read by the depositor
    size_t   cap      = 0;      // per-thread ceiling; 0 = unbounded (no self-thinning)
    Pcg32    rng;               // private stream: thinning must not perturb the photon RNG

    size_t size() const { return beams.size(); }

    void push(const Vec3& o, const Vec3& d, double len, double power,
              double lambda, double absorb, int med) {
        beams.push_back(PhotonBeam{o, d, 0.0f, (float)len, (float)power,
                                   (float)lambda, (float)absorb, med});
        if (cap && beams.size() >= cap) halve();
    }

    // Drop half the bank at random, double the survivors, and halve the future deposit rate.
    // Amortised O(1) per push: each halving frees cap/2 slots, so the copy cost is paid once
    // per cap/2 deposits however long the pass runs.
    void halve() {
        size_t w = 0;
        for (size_t i = 0; i < beams.size(); ++i) {
            if (rng.uniform() < 0.5) {
                beams[w] = beams[i];
                beams[w].power *= 2.0f;
                ++w;
            }
        }
        beams.resize(w);
        keepProb *= 0.5;
    }
};

// Result of one beam/ray closest-approach test that passed the radius check.
struct BeamHit {
    int    idx;      // index into BeamMap::beams
    double tCam;     // distance along the camera ray to the closest-approach point
    double sBeam;    // distance along the beam FROM ITS TRUE ORIGIN to that point
    double dPerp;    // separation at closest approach
    double sinT;     // |cross(d_cam, d_beam)| — the 1D-blur Jacobian denominator
    double cosT;     // dot(d_cam, d_beam)
};

// The gather structure: a BVH over per-beam AABBs, each inflated by the kernel radius so a
// camera ray that passes *within* the radius of a beam still enters its box.
//
// The BVH is the generic one from bvh.h, built straight over the inflated boxes; the gather
// uses traverseAny() with a leaf callback that always returns false, which turns the
// occlusion traversal into "visit every intersected leaf primitive" — exactly the all-hits
// traversal a density estimate needs, with no new traversal code to keep in sync.
struct BeamMap {
    std::vector<PhotonBeam> beams;
    std::vector<Vec3>       cie;      // per-beam CIE XYZ at beams[i].lambda, precomputed by
                                      // build() for the same reason PhotonMap::cie exists:
                                      // the analytic CIE fit is several exp() and would
                                      // otherwise be re-evaluated per beam PER GATHER.
    Bvh       bvh;
    // PER-MEDIUM 1D kernel half-width (world units), indexed by PhotonBeam::med. See the
    // header note: the radius is derived from each medium's own measured mean free path, so a
    // dense cloud and a sparse rain curtain in the same scene get their own blur scale.
    std::vector<float> radMed;
    double    radius   = 0.02;        // fallback for a beam whose med is out of range, and
                                      // (after buildAuto) the MAX over media — the one number
                                      // worth printing when a scene has a single medium.
    long long nEmitted = 0;           // photons emitted by the pass (normalisation)
    size_t    nDeposited = 0;         // crossings actually deposited, BEFORE decimateTo —
                                      // reported so the overshoot over `-beamcount` (i.e. the
                                      // mean media-crossing multiplicity of a photon) is
                                      // visible rather than inferred.

    bool empty() const { return beams.empty(); }

    // The kernel half-width for a beam in medium `med`. Out-of-range (including med < 0, and
    // a map loaded from a v3 cache before radMed was filled) falls back to the scalar.
    double radOf(int med) const {
        return (med >= 0 && (size_t)med < radMed.size()) ? (double)radMed[(size_t)med] : radius;
    }

    // --- geometry summaries used by the radius heuristic ------------------------------
    Aabb bounds() const {
        Aabb a;
        for (const PhotonBeam& b : beams) { a.expand(b.begin()); a.expand(b.end()); }
        return a;
    }
    double totalLength() const {
        double s = 0.0;
        for (const PhotonBeam& b : beams) s += (double)b.len;
        return s;
    }

    // Second-stage Russian roulette: thin an already-collected set down to at most `target`
    // beams, rescaling survivors by 1/q. Unbiased.
    //
    // This is the EXACT trim, not the memory guard — BeamBank's self-halving is what bounds
    // memory during the pass. Each per-thread bank stops at its own cap, so the concatenated
    // total lands somewhere in [target, 2*target]; one cut here is what makes `-beamcount`
    // mean the number the user actually typed.
    //
    // Verified unbiased in practice as well as on paper: on _fog_cornell, thinning 2.25 M
    // deposits to 1 M changed the frame's auto-exposure from 7.43e-13 to 7.42e-13.
    void decimateTo(size_t target, uint64_t seed = 0x243F6A8885A308D3ULL) {
        if (target == 0 || beams.size() <= target) return;
        const double q = (double)target / (double)beams.size();
        const float  boost = (float)(1.0 / q);
        Pcg32 rng; rng.seed(seed, 0x13198A2E03707344ULL);
        std::vector<PhotonBeam> out;
        out.reserve(target + target / 8 + 16);
        for (const PhotonBeam& b : beams) {
            if (rng.uniform() >= q) continue;
            out.push_back(b);
            out.back().power *= boost;
        }
        beams.swap(out);
    }

    // The area-optimal sub-segment length for ONE beam at kernel radius `r`. Returns 0 for
    // "never split this beam".
    //
    // WHY AREA. Traversal cost is not "how many beams did the ray gather" — measured, a 16×
    // change in the gathered count moved the time 3%. It is how many inflated AABBs the ray
    // had to ENTER, and by Cauchy's formula the probability a random ray enters a convex box
    // is proportional to that box's surface area. So the quantity to minimise over the choice
    // of split is the TOTAL AREA of all sub-beam boxes — the same quantity a SAH BVH builder
    // minimises, applied one level earlier, to the primitives themselves.
    //
    // THE DERIVATION. A sub-segment of length p travelling in unit direction d, inflated by
    // the kernel radius r, is an AABB with extents (|dx|p + 2r, |dy|p + 2r, |dz|p + 2r).
    // Summing 2(ab + bc + ca) over the L/p pieces of a beam of length L:
    //
    //     A_total(p) = 2L [ Q·p  +  4rE  +  12r²/p ]
    //         Q = |dx||dy| + |dy||dz| + |dz||dx|        E = |dx| + |dy| + |dz|
    //
    // The two ends of that expression are the whole tradeoff: Q·p is the empty space a long
    // diagonal box encloses (too few pieces), 12r²/p is the kernel inflation paid once per
    // piece (too many). dA/dp = 0 gives
    //
    //     p* = 2r·sqrt(3/Q)        and        A_total(p*) = 8Lr [ sqrt(3Q) + E ]
    //
    // THREE CONSEQUENCES, each of which the old rule got wrong:
    //   * p* is O(r) and contains NO scene-scale term. It tightens automatically as the radius
    //     shrinks. `max(diag/64, S/3N)` was two scene-scale quantities with no dependence on
    //     beam density, and returned 0.4999 / 0.5014 / 0.5012 across a 10× beam-count sweep —
    //     i.e. it was not adapting at all.
    //   * A_total(p*) is linear in r and in total beam length S. buildAuto chooses
    //     r = K·A_box/(2πS) to hold the gathered count at K, so the product S·r = K·A_box/(2π)
    //     is CONSTANT and the traversal cost stops depending on how many beams are stored.
    //     Cost being linear in stored beams was the measured pathology; it is not a tuning
    //     failure but a direct consequence of splitting at a length that ignored r.
    //   * An axis-aligned beam has Q = 0, hence p* = infinity: its AABB is already tight and
    //     splitting it is pure loss. A uniform rule splits it anyway.
    //
    // BUT p* IS THE INFINITE-WORK LIMIT, AND THE BUILD IS NOT FREE. Splitting also costs a BVH
    // build that is LINEAR in the sub-beam count — measured at ~1.6 us per sub-beam, which at
    // the unconstrained optimum for a small scene is minutes. That cost is paid ONCE and then
    // amortised over every camera that shares the map, so the right split is not a property of
    // the beams alone: it depends on how much gathering the map is about to do. Minimising
    //
    //     T(p) = c_build·(S/p)  +  W·k·A_total(p)
    //
    // over p (W = total pixel-samples across the sharing cameras, k = gather seconds per
    // pixel-sample per unit area) gives, after the same differentiation,
    //
    //     p_opt = sqrt(3/Q) · sqrt( 4r² + kappa/W ),      kappa = c_build / (6k)
    //
    // i.e. EXACTLY the area optimum with 4r² lifted by a work-dependent floor. W -> infinity
    // recovers p* (a 600-frame flyby should split as finely as memory allows, because the build
    // is amortised to nothing); small W backs off (do not spend three minutes of BVH build to
    // save one minute of gather on a single frame). This is why the same rule must serve both
    // a one-off still and a flythrough, and why a rule with no work term cannot.
    //
    // MEASURED, on _fog_cornell at 128x128, sweeping a pinned uniform split (-beamsplit):
    //
    //     split   sub-beams   box area   BVH build   gather
    //     0.5      3.46 M     9.44e5      5.7 s      47.9 s/spp   <- what the old rule chose
    //     0.2      8.05 M     4.08e5     12.6 s      18.4 s/spp
    //     0.05    30.6  M     1.26e5     53.2 s       9.4 s/spp
    //     0.012  126     M    5.57e4    180.0 s       4.5 s/spp
    //
    // Gather tracks box area closely (it falls 10.6x as area falls 17x), which is the evidence
    // that area is the right cost metric; build is linear in count, giving c_build ~ 1.6 us and
    // k ~ 3.7e-9 s per (pixel-sample · m²), hence kappa ~ 72 m². Substituting the single-camera
    // 128x128x4spp work W = 65536 predicts p_opt = 0.058 — and the measured single-camera
    // optimum of that sweep is between 0.05 (87 s) and 0.2 (96 s). The model is not fitted to
    // the answer; it predicts it from two independently measured constants.
    static constexpr double kSplitKappa = 72.0;   // m², = c_build / (6k); see the table above

    // `kappaOverW` is kappa/W — the work-dependent floor, 0 for "infinite work" (pure area
    // optimum). Returns 0 for "never split this beam".
    static double sahSplitLen(const PhotonBeam& b, double r, double kappaOverW) {
        const double ax = std::fabs(b.d.x), ay = std::fabs(b.d.y), az = std::fabs(b.d.z);
        const double Q  = ax * ay + ay * az + az * ax;
        if (!(Q > 1e-12) || !(r > 0.0)) return 0.0;      // axis-aligned / no radius: no split
        return std::sqrt((12.0 * r * r + 3.0 * std::max(0.0, kappaOverW)) / Q);
    }

    // Total inflated-AABB area of the current beam set, with every beam inflated by `scale`
    // times ITS OWN medium's radius — the cost metric the split minimises, reported so a change
    // to the rule is measurable rather than asserted. `scale = 0` gives the TIGHT area (no
    // kernel inflation at all), which is the baseline the area-slack ceiling is a fraction of.
    double totalBoxArea(double scale = 1.0) const {
        double a = 0.0;
        for (const PhotonBeam& b : beams) {
            const double r  = radOf(b.med) * scale;
            const double ex = std::fabs(b.d.x) * (double)b.len + 2.0 * r;
            const double ey = std::fabs(b.d.y) * (double)b.len + 2.0 * r;
            const double ez = std::fabs(b.d.z) * (double)b.len + 2.0 * r;
            a += 2.0 * (ex * ey + ey * ez + ez * ex);
        }
        return a;
    }

    // Solve for the largest uniform scale s on the per-medium radii such that the total box
    // area has grown by at most `slack` over the tight (s = 0) area. Closed form, because the
    // area is exactly quadratic in s:
    //
    //     A(s) = A0 + s * SUM 8 r_m (|dx|+|dy|+|dz|) L  +  s^2 * SUM 24 r_m^2
    //          = A0 + B s + C s^2                      (expand 2(exey+eyez+ezex), q = 2 r s)
    //
    // Returns >= 1 for "the cap does not bind". This is the CEILING half of the radius rule;
    // `-beamk` is the floor. It exists because the mean-free-path radius is a physical scale
    // with no cost term in it, and a scene can legitimately have a medium whose mfp is a large
    // fraction of its own extent (a thin haze), where 1% of the mfp still inflates every box
    // enough to matter.
    double areaSlackScale(double slack) const {
        if (beams.empty() || !(slack > 0.0)) return (slack > 0.0) ? 1.0 : 0.0;
        double A0 = 0.0, B = 0.0, C = 0.0;
        for (const PhotonBeam& b : beams) {
            const double r  = radOf(b.med);
            const double L  = (double)b.len;
            const double u  = std::fabs(b.d.x) * L, v = std::fabs(b.d.y) * L, w = std::fabs(b.d.z) * L;
            A0 += 2.0 * (u * v + v * w + w * u);
            B  += 8.0 * r * (u + v + w);
            C  += 24.0 * r * r;
        }
        const double budget = slack * A0;
        if (!(C > 0.0)) return (B > 0.0) ? std::min(1.0, budget / B) : 1.0;
        const double s = (-B + std::sqrt(std::max(0.0, B * B + 4.0 * C * budget))) / (2.0 * C);
        return (std::isfinite(s) && s > 0.0) ? std::min(1.0, s) : 1.0;
    }

    // Split every beam at its OWN area-optimal length (sahSplitLen), backing off uniformly if
    // the unconstrained optimum would exceed `budget` sub-beams. Returns the mean sub-segment
    // length actually used, for reporting.
    //
    // WHY A BUDGET AT ALL. p_opt is O(r) and buildAuto shrinks r as ~1/S, so the optimal piece
    // COUNT grows as ~S² — the cost is bounded but the memory is not, and post-split records
    // are what a GPU port has to fit in VRAM (~96 B per sub-beam plus its BVH node). Measured
    // above: at 126 M sub-beams a 1 M-beam map is ~12 GB. So the budget is a memory ceiling,
    // not a cost knob: below it we take the optimum, above it we take the finest split that
    // fits, which is still strictly better than a scene-scale length because it is at least
    // proportional to the optimum beam-by-beam.
    double splitSah(size_t budget, double kappaOverW) {
        if (beams.empty()) return 0.0;
        // Σ L/p scales as 1/f, so the closed-form factor lands close; ceil() and the
        // never-split (axis-aligned) beams only make the true curve shallower, so tighten
        // from below rather than trusting one step.
        auto pieces = [&](double f) {
            double n = 0.0;
            for (const PhotonBeam& b : beams) {
                const double p = sahSplitLen(b, radOf(b.med), kappaOverW) * f;
                n += (p > 0.0 && (double)b.len > p) ? std::ceil((double)b.len / p) : 1.0;
            }
            return n;
        };
        double f = 1.0;
        if (budget) {
            double n = pieces(1.0);
            for (int it = 0; it < 32 && n > (double)budget; ++it) {
                f *= std::max(1.05, n / (double)budget);
                n = pieces(f);
            }
        }
        // Reserve the EXACT post-split count rather than a 2x guess. `pieces(f)` is the
        // same sum the loop below is about to perform, so this is exact, and it turns the
        // geometric regrowth of a buffer that can reach many gigabytes into ONE allocation
        // — which is also the only place the split can plausibly run out of memory, and so
        // the place to name `-beamsplitmax` when it does.
        std::vector<PhotonBeam> out;
        {
            const double nOut = pieces(f);
            const size_t want = (nOut > 0.0 && nOut < 4.0e18) ? (size_t)nOut : beams.size();
            ftalloc::reserve(out, want, "the split photon-beam array",
                             "-beamsplitmax (or -beamcount, which feeds it)");
        }
        double lenSum = 0.0; size_t nSeg = 0;
        for (const PhotonBeam& b : beams) {
            const double len = (double)b.len;
            const double p   = sahSplitLen(b, radOf(b.med), kappaOverW) * f;
            if (!(p > 0.0) || len <= p) { out.push_back(b); lenSum += len; ++nSeg; continue; }
            int k = (int)std::ceil(len / p);
            if (k > 65536) k = 65536;                  // pathological guard
            const double seg = len / (double)k;
            for (int j = 0; j < k; ++j) {
                PhotonBeam s = b;
                s.s0  = (float)((double)b.s0 + (double)j * seg);
                s.len = (float)seg;
                out.push_back(s);
            }
            lenSum += len; nSeg += (size_t)k;
        }
        beams.swap(out);
        return nSeg ? lenSum / (double)nSeg : 0.0;
    }

    // Split beams longer than `maxLen` into equal sub-segments (see the header note on beam
    // splitting). Sub-segments share the parent's origin and power and only carry their own
    // [s0, s0+len] range, so nothing has to be re-integrated.
    //
    // Retained for the `-beamsplit <len>` expert override, which pins a uniform length so the
    // rule above can be measured against a fixed baseline. splitSah() is the default path.
    void splitLong(double maxLen) {
        if (!(maxLen > 0.0)) return;
        size_t extra = 0;
        for (const PhotonBeam& b : beams)
            if ((double)b.len > maxLen) extra += (size_t)std::ceil((double)b.len / maxLen) - 1;
        if (extra == 0) return;
        std::vector<PhotonBeam> out;
        ftalloc::reserve(out, beams.size() + extra, "the split photon-beam array",
                         "-beamsplit (or -beamcount, which feeds it)");
        for (const PhotonBeam& b : beams) {
            const double len = (double)b.len;
            if (len <= maxLen) { out.push_back(b); continue; }
            int k = (int)std::ceil(len / maxLen);
            if (k > 65536) k = 65536;                  // pathological guard
            const double seg = len / (double)k;
            for (int j = 0; j < k; ++j) {
                PhotonBeam s = b;
                s.s0  = (float)((double)b.s0 + (double)j * seg);
                s.len = (float)seg;
                out.push_back(s);
            }
        }
        beams.swap(out);
    }

    // Set every medium's kernel half-width to the same value — the `-beamradius` expert
    // override, and the fallback for a map with no usable per-medium statistics.
    void setUniformRadius(double r) {
        radius = (r > 0.0) ? r : 1e-6;
        int nMed = 0;
        for (const PhotonBeam& b : beams) if (b.med >= 0) nMed = std::max(nMed, b.med + 1);
        radMed.assign((size_t)nMed, (float)radius);
    }

    // Bin the beams at the CURRENT per-medium radii (radMed, set by buildAuto or
    // setUniformRadius): split, then build the BVH over per-sub-beam AABBs each inflated by its
    // OWN medium's radius, then precompute the CIE triples.
    //
    // `explicitSplitLen > 0` pins a uniform split length (the `-beamsplit` expert override);
    // otherwise every beam is split at its own cost-optimal length, bounded by `splitBudget`
    // sub-beams (0 = unbounded). `work` is the total pixel-samples every camera sharing this
    // map will gather — the term that decides how much BVH build is worth buying (0 = treat
    // the build as free, i.e. the pure area optimum). `meanSplitOut` gets the mean length used.
    void build(double explicitSplitLen = 0.0, size_t splitBudget = 0,
               double work = 0.0, double* meanSplitOut = nullptr) {
        if (radMed.empty()) setUniformRadius(radius);
        const double kappaOverW = (work > 0.0) ? kSplitKappa / work : 0.0;
        double meanSplit;
        if (explicitSplitLen > 0.0) { splitLong(explicitSplitLen); meanSplit = explicitSplitLen; }
        else                        { meanSplit = splitSah(splitBudget, kappaOverW); }
        if (meanSplitOut) *meanSplitOut = meanSplit;
        ftalloc::resize(cie, beams.size(), "the beam CIE table", "-beamcount / -beamsplitmax");
        std::vector<Aabb> boxes;
        ftalloc::resize(boxes, beams.size(), "the beam BVH bounding boxes",
                        "-beamcount / -beamsplitmax");
        for (size_t i = 0; i < beams.size(); ++i) {
            const PhotonBeam& b = beams[i];
            const double r = radOf(b.med);
            Aabb a;
            a.expand(b.begin());
            a.expand(b.end());
            a.lo = a.lo - Vec3(r, r, r);
            a.hi = a.hi + Vec3(r, r, r);
            boxes[i] = a;
            const double lam = (double)b.lambda;
            cie[i] = Vec3(cieX(lam), cieY(lam), cieZ(lam));
        }
        bvh.build(boxes);
    }

    // Per-medium statistics of the RAW (pre-split) beam set: how many chords each medium
    // stored, and their total length. mfp_m = S_m / N_m is the medium's measured mean free
    // path — bounded above by the medium's own extent, since a chord is clipped to the bound.
    struct MedStat { size_t n = 0; double len = 0.0; double mfp = 0.0; double r = 0.0; };
    std::vector<MedStat> mediumStats() const {
        int nMed = 0;
        for (const PhotonBeam& b : beams) if (b.med >= 0) nMed = std::max(nMed, b.med + 1);
        std::vector<MedStat> st((size_t)nMed);
        for (const PhotonBeam& b : beams)
            if (b.med >= 0) { MedStat& s = st[(size_t)b.med]; ++s.n; s.len += (double)b.len; }
        for (MedStat& s : st) s.mfp = s.n ? s.len / (double)s.n : 0.0;
        return st;
    }

    // Diagnostics filled in by buildAuto(), so the driver can report what it settled on
    // instead of silently choosing a radius that changes the image.
    struct AutoInfo {
        std::vector<MedStat> med;   // per medium: chord count, total length, mfp, FINAL radius
        double blur       = 0;  // -beamblur: the fraction of the mfp the radius was set to
        double rMin       = 0;  // range of the final per-medium radii, for a one-line summary
        double rMax       = 0;
        double probeK0    = 0;  // beams a probe ray gathered at the raw mfp radii
        double probeK     = 0;  // ... and after the floor/ceiling corrections
        double targetK    = 0;  // -beamk: the FLOOR on the gathered count (not a target)
        double floorScale = 1;  // >1 if -beamk had to scale the radii up
        double slackScale = 1;  // <1 if -beamareaslack had to scale them back down
        double splitLen   = 0;  // MEAN sub-segment length (the rule is per-beam, not uniform)
        size_t rawBeams   = 0;  // before splitting
        size_t outBeams   = 0;  // after splitting
        double areaTight  = 0;  // total AABB area with NO kernel inflation — the slack baseline
        double areaBefore = 0;  // total inflated-AABB area unsplit  — the cost metric, so a
        double areaAfter  = 0;  // total inflated-AABB area after split   split is measurable
        bool   budgetBit  = false;  // true if the sub-beam budget forced a coarser-than-optimal
                                    // split (i.e. cost is memory-limited, not rule-limited)
    };

    // Brute-force mean gathered count along random chords of the beam bbox, at the CURRENT
    // per-medium radii. Strided subsample so the cost is bounded regardless of beam count; the
    // measured count is scaled back up by the stride. No BVH needed (and none exists yet at
    // the point buildAuto calls this).
    //
    // This is a diagnostic AND the floor test. It exists because the closed form
    // r = K A / (2 pi S) assumes isotropic relative orientation and beams spread evenly through
    // their bounding box, and neither holds in a sunlit rain volume where every beam is
    // near-parallel to the sun — measured, it is off by more than an order of magnitude there.
    double probeGatherCount(const Aabb& bb) const {
        if (beams.empty()) return 0.0;
        const Vec3 ext = bb.hi - bb.lo;
        const size_t kProbeRays = 96;
        const size_t maxTests   = 120000;
        const size_t stride     = std::max<size_t>(1, beams.size() / maxTests);
        Pcg32 prng; prng.seed(0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL);
        auto randPt = [&]() {
            return Vec3(bb.lo.x + ext.x * prng.uniform(),
                        bb.lo.y + ext.y * prng.uniform(),
                        bb.lo.z + ext.z * prng.uniform());
        };
        double hits = 0.0;
        for (size_t q = 0; q < kProbeRays; ++q) {
            Vec3 a = randPt(), c = randPt();
            Vec3 dv = c - a;
            double L = std::sqrt(dot(dv, dv));
            if (!(L > 1e-9)) continue;
            dv = dv * (1.0 / L);
            BeamHit bh;
            for (size_t i = 0; i < beams.size(); i += stride)
                if (closestApproach((int)i, a, dv, L, bh)) hits += 1.0;
        }
        return hits * (double)stride / (double)kProbeRays;
    }

    // Choose each medium's kernel radius from its own measured mean free path, apply the
    // `-beamk` floor and the `-beamareaslack` ceiling, then build.
    //
    // See the long note at the top of this file for why the radius is NOT chosen to hit a
    // target gathered count: doing that makes r ~ 1/n, so the estimator has a variance floor
    // that no amount of photons or samples can move, in exchange for a bias reduction that
    // measurement shows does not happen.
    //
    //   blur       fraction of the medium's mfp to use as the half-width (`-beamblur`, 0.01)
    //   targetK    FLOOR on the gathered count (`-beamk`). Scales the radii UP, never down.
    //   areaSlack  ceiling: the fraction by which the kernel may inflate the total box area
    //              over the tight (r = 0) area (`-beamareaslack`). Wins over the floor.
    AutoInfo buildAuto(double blur, double targetK, double areaSlack, size_t splitBudget = 0,
                       double explicitSplitLen = 0.0, double work = 0.0) {
        AutoInfo info;
        info.blur     = blur;
        info.targetK  = targetK;
        info.rawBeams = beams.size();
        if (beams.empty()) { build(explicitSplitLen, splitBudget, work); return info; }

        const Aabb bb = bounds();
        const Vec3 ext = bb.hi - bb.lo;
        const double diag = std::max(std::sqrt(dot(ext, ext)), 1e-9);

        // --- r_m = blur * mfp_m ------------------------------------------------------
        std::vector<MedStat> st = mediumStats();
        double rSeen = 0.0;
        for (MedStat& s : st) { s.r = blur * s.mfp; rSeen = std::max(rSeen, s.r); }
        // A medium with no stored chords has no measured scale. It also has no beams to
        // gather, so the value is unobservable — give it the largest seen radius (or a
        // scene-scale sliver if no medium has any) purely so nothing downstream divides by 0.
        if (!(rSeen > 0.0)) rSeen = 1e-4 * diag;
        for (MedStat& s : st) if (!(s.r > 0.0)) s.r = rSeen;
        radMed.resize(st.size());
        auto push = [&]() { for (size_t m = 0; m < st.size(); ++m) radMed[m] = (float)st[m].r; };
        push();

        // --- FLOOR: -beamk, so a sparse map does not degenerate into single streaks ----
        // Iterated, because one probe cannot correct an arbitrarily large shortfall: the
        // gathered count is linear in r, but the probe measures 0 when the radii are far too
        // small and a ratio against 0 is meaningless. Each round scales by at most 16x and
        // re-measures, so three rounds cover 4096x — far beyond any real shortfall.
        info.probeK0 = probeGatherCount(bb);
        info.probeK  = info.probeK0;
        if (targetK > 0.0) {
            for (int round = 0; round < 3 && info.probeK < targetK; ++round) {
                const double meas = std::max(info.probeK, 0.25);
                const double f = std::clamp(targetK / meas, 1.0, 16.0);
                if (f <= 1.0000001) break;
                info.floorScale *= f;
                for (MedStat& s : st) s.r *= f;
                push();
                info.probeK = probeGatherCount(bb);
            }
        }

        // --- CEILING: -beamareaslack, the cost bound. Applied last, so it wins. --------
        info.areaTight = totalBoxArea(0.0);
        if (areaSlack > 0.0) {
            const double s = areaSlackScale(areaSlack);
            if (s < 1.0) {
                info.slackScale = s;
                for (MedStat& m : st) m.r *= s;
                push();
                info.probeK = probeGatherCount(bb);
            }
        }

        info.med  = st;
        info.rMin = info.rMax = st.empty() ? 0.0 : st[0].r;
        for (const MedStat& s : st) { info.rMin = std::min(info.rMin, s.r); info.rMax = std::max(info.rMax, s.r); }
        radius = info.rMax;      // the scalar fallback / the number worth printing

        // Split at each beam's own area-optimal length (sahSplitLen), bounded by the sub-beam
        // budget. The radii are final at this point, which is the whole reason the rule can
        // key off them — the old scene-scale rule could have been computed before any of this
        // ran, and that is exactly what was wrong with it.
        info.areaBefore = totalBoxArea();
        // Did memory, rather than the cost model, pick the split? Answer it BEFORE building,
        // by asking what the unconstrained rule would have produced — comparing the final count
        // against the budget cannot tell you, because the back-off lands strictly under it.
        if (splitBudget && explicitSplitLen <= 0.0) {
            const double kOverW = (work > 0.0) ? kSplitKappa / work : 0.0;
            double want = 0.0;
            for (const PhotonBeam& b : beams) {
                const double p = sahSplitLen(b, radOf(b.med), kOverW);
                want += (p > 0.0 && (double)b.len > p) ? std::ceil((double)b.len / p) : 1.0;
            }
            info.budgetBit = want > (double)splitBudget;
        }
        double meanSplit = 0.0;
        build(explicitSplitLen, splitBudget, work, &meanSplit);
        info.splitLen  = meanSplit;
        info.outBeams  = beams.size();
        info.areaAfter = totalBoxArea();
        return info;
    }

    // The 1D Epanechnikov kernel for a beam in medium `med`, normalised so its integral over
    // [-r_med, r_med] is 1. Smooth-topped and compactly supported: a box kernel of the same
    // width would give the same mean with visible slab edges along every beam.
    double kernel1D(double u, int med) const {
        const double r = radOf(med);
        if (!(r > 0.0)) return 0.0;
        const double x = u / r;
        const double k = 1.0 - x * x;
        return (k > 0.0) ? (0.75 / r) * k : 0.0;
    }

    // Closest approach between the camera ray (o_c + t*d_c, t in [0, tMax]) and beam i.
    // Both directions must be unit. Returns false when the pair is (near-)parallel — the
    // 1/sin(theta) Jacobian diverges there and the configuration has measure zero — or when
    // the closest approach falls outside either segment, or beyond the kernel radius.
    bool closestApproach(int i, const Vec3& oc, const Vec3& dc, double tMax, BeamHit& out) const {
        const PhotonBeam& b = beams[i];
        const double cosT = dot(dc, b.d);
        const double den  = 1.0 - cosT * cosT;              // == sin^2(theta)
        if (den < 1e-9) return false;                       // parallel: reject
        const Vec3 w0 = oc - b.o;
        const double dd = dot(dc, w0), ee = dot(b.d, w0);
        const double t = (cosT * ee - dd) / den;
        const double s = (ee - cosT * dd) / den;
        if (t < 0.0 || t > tMax) return false;
        if (s < (double)b.s0 || s > (double)b.s0 + (double)b.len) return false;
        const Vec3 diff = (oc + dc * t) - (b.o + b.d * s);
        const double d2 = dot(diff, diff);
        const double r  = radOf(b.med);           // per medium — see the header note
        if (d2 >= r * r) return false;
        out.idx = i; out.tCam = t; out.sBeam = s;
        out.dPerp = std::sqrt(d2);
        out.sinT = std::sqrt(den); out.cosT = cosT;
        return true;
    }

    // Visit every beam whose kernel cylinder the camera segment [0, tMax] passes through.
    // `fn(const BeamHit&)` is called once per surviving candidate, in traversal order.
    template <class Fn>
    void gather(const Vec3& oc, const Vec3& dc, double tMax, Fn&& fn) const {
        if (beams.empty() || bvh.nodes.empty()) return;
        Ray r{oc, dc};
        BeamHit bh;
        bvh.traverseAny(r, 0.0, tMax, [&](int i) {
            if (closestApproach(i, oc, dc, tMax, bh)) fn(bh);
            return false;            // never "blocked" -> traverseAny visits every leaf
        });
    }
};

// (There used to be a `beamKeepProb(targetBeams, nPhotons)` here that picked the deposit
// survival rate up front as target/nPhotons. It is gone rather than deprecated: the ratio it
// assumed — about one beam per photon — is set by the media's volume fraction and the bounce
// depth, so it was wrong by 131x on one test scene and 2.3x the other way on another. The
// bank measures the rate instead of predicting it; see BeamBank in this file.)
