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
// which is what buildAuto() starts from — V drops out entirely. It then PROBES (brute force
// over a strided subsample, no BVH needed) to correct for the assumptions the closed form
// makes: isotropic relative orientation, and beams spread evenly through their bounding box.
// Neither holds in a sunlit rain volume, where every beam is near-parallel to the sun.
//
// BEAM SPLITTING
// --------------
// A long diagonal beam has a terrible AABB — mostly empty space that every passing ray must
// test. build() therefore splits long beams into sub-segments. A sub-segment keeps the
// ORIGINAL origin `o` and records its own [s0, s0+len] parameter range, so the gather can
// still evaluate transmittance from the true beam start (which is where the stored power
// applies) without the splitter needing a Scene or an RNG to re-integrate it.
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
    double    radius   = 0.02;        // 1D kernel half-width (world units)
    long long nEmitted = 0;           // photons emitted by the pass (normalisation)
    size_t    nDeposited = 0;         // crossings actually deposited, BEFORE decimateTo —
                                      // reported so the overshoot over `-beamcount` (i.e. the
                                      // mean media-crossing multiplicity of a photon) is
                                      // visible rather than inferred.

    bool empty() const { return beams.empty(); }

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

    // Split beams longer than `maxLen` into equal sub-segments (see the header note on beam
    // splitting). Sub-segments share the parent's origin and power and only carry their own
    // [s0, s0+len] range, so nothing has to be re-integrated.
    void splitLong(double maxLen) {
        if (!(maxLen > 0.0)) return;
        size_t extra = 0;
        for (const PhotonBeam& b : beams)
            if ((double)b.len > maxLen) extra += (size_t)std::ceil((double)b.len / maxLen) - 1;
        if (extra == 0) return;
        std::vector<PhotonBeam> out;
        out.reserve(beams.size() + extra);
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

    // Bin the beams at an explicit kernel radius. Splits first (see splitLong), then builds
    // the BVH over per-sub-beam AABBs inflated by the radius, then precomputes the CIE
    // triples. `splitFrac`/`splitMean` shape the split length; see buildAuto for the values
    // the driver uses and why.
    void build(double r, double maxSplitLen = 0.0) {
        radius = (r > 0.0) ? r : 1e-6;
        if (maxSplitLen > 0.0) splitLong(maxSplitLen);
        cie.resize(beams.size());
        std::vector<Aabb> boxes(beams.size());
        for (size_t i = 0; i < beams.size(); ++i) {
            const PhotonBeam& b = beams[i];
            Aabb a;
            a.expand(b.begin());
            a.expand(b.end());
            a.lo = a.lo - Vec3(radius, radius, radius);
            a.hi = a.hi + Vec3(radius, radius, radius);
            boxes[i] = a;
            const double lam = (double)b.lambda;
            cie[i] = Vec3(cieX(lam), cieY(lam), cieZ(lam));
        }
        bvh.build(boxes);
    }

    // Diagnostics filled in by buildAuto(), so the driver can report what it settled on
    // instead of silently choosing a radius that changes the image.
    struct AutoInfo {
        double rAnalytic = 0;   // closed-form starting radius
        double rFinal    = 0;   // after the probe correction
        double probeK    = 0;   // beams a probe ray actually gathered at rAnalytic
        double targetK   = 0;   // beams per gather we were aiming for
        double splitLen  = 0;   // BVH split length
        size_t rawBeams  = 0;   // before splitting
        size_t outBeams  = 0;   // after splitting
    };

    // Choose the kernel radius for a target of `targetK` beams gathered per camera segment,
    // then build. See the header: the closed form assumes isotropic orientation and a
    // uniformly filled bounding box, so we correct it with a cheap brute-force probe over a
    // strided subsample of the beams (no BVH needed, and the sample is large enough that the
    // ratio is far more stable than the absolute count).
    AutoInfo buildAuto(double targetK) {
        AutoInfo info;
        info.targetK  = targetK;
        info.rawBeams = beams.size();
        if (beams.empty() || !(targetK > 0.0)) { build(radius); info.rFinal = radius; return info; }

        const Aabb bb = bounds();
        const Vec3 ext = bb.hi - bb.lo;
        const double area = std::max(bb.area(), 1e-12);
        const double S    = std::max(totalLength(), 1e-12);
        const double diag = std::sqrt(dot(ext, ext));

        // Closed form: r = K * A / (2 pi S).
        double r = targetK * area / (6.283185307179586 * S);
        if (!(r > 0.0) || !std::isfinite(r)) r = 1e-6 * std::max(diag, 1e-6);
        info.rAnalytic = r;

        // --- probe: brute-force mean gathered count along random chords of the box -------
        // Strided subsample so the cost is bounded regardless of beam count; the measured
        // count is scaled back up by the stride.
        {
            const size_t kProbeRays = 96;
            const size_t maxTests   = 120000;
            const size_t stride     = std::max<size_t>(1, beams.size() / maxTests);
            Pcg32 prng; prng.seed(0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL);
            auto randPt = [&]() {
                return Vec3(bb.lo.x + ext.x * prng.uniform(),
                            bb.lo.y + ext.y * prng.uniform(),
                            bb.lo.z + ext.z * prng.uniform());
            };
            const double rSave = radius;
            radius = r;
            double hits = 0.0;
            for (size_t q = 0; q < kProbeRays; ++q) {
                Vec3 a = randPt(), c = randPt();
                Vec3 dv = c - a;
                double L = std::sqrt(dot(dv, dv));
                if (!(L > 1e-9)) { continue; }
                dv = dv * (1.0 / L);
                BeamHit bh;
                for (size_t i = 0; i < beams.size(); i += stride)
                    if (closestApproach((int)i, a, dv, L, bh)) hits += 1.0;
            }
            radius = rSave;
            // Scale: (a) back up for the stride, (b) from the mean CHORD of the box to the
            // mean chord 4V/A the closed form assumed. Random point-to-point chords of a box
            // are a bit shorter than 4V/A, but both are O(diag) and the probe only has to
            // correct an order-of-magnitude error, not a 10% one.
            info.probeK = hits * (double)stride / (double)kProbeRays;
            if (info.probeK > 0.25) {
                double scale = targetK / info.probeK;
                scale = std::clamp(scale, 1.0 / 64.0, 64.0);
                r *= scale;
            }
        }

        // Split length: bound the post-split count at ~4x the raw count (each beam adds at
        // most len/splitLen pieces, and splitLen >= S/(3N) caps the total added at 3N), with
        // a floor at diag/64 so a scene of already-short beams is not split pointlessly.
        const double splitLen = std::max(diag / 64.0, S / (3.0 * (double)beams.size()));
        info.splitLen = splitLen;
        build(r, splitLen);
        info.rFinal   = radius;
        info.outBeams = beams.size();
        return info;
    }

    // The 1D Epanechnikov kernel, normalised so its integral over [-radius, radius] is 1.
    // Smooth-topped and compactly supported: a box kernel of the same width would give the
    // same mean with visible slab edges along every beam.
    double kernel1D(double u) const {
        const double x = u / radius;
        const double k = 1.0 - x * x;
        return (k > 0.0) ? (0.75 / radius) * k : 0.0;
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
        if (d2 >= radius * radius) return false;
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
