// The Beam x Ray 1D estimator (Jarosz et al. 2011), shared by mode M and mode J.
//
// WHY THIS IS ITS OWN HEADER. The estimator lived in photonmap_render.h until 0.215.0,
// which was the right place while mode M was its only caller. Mode J (UPBP) is the second
// caller, and it lives in bdpt.h — a header that has no business including the photon-map
// renderer, and could not do so without dragging in causticaim.h, photonmap_io.h and the
// deposit machinery to reach one function. Going the other way (bdpt.h's merge code moved
// into photonmap_render.h) is worse still: a UPBP merge weight is built from the *BDPT
// subpath densities*, so it belongs with them.
//
// So the primitive moves down to the level both callers already share. This header needs
// exactly what the estimator needs — `render.h` for Renderer::mediaTransmittance and
// `photonbeams.h` for the map — and nothing else.
//
// WHAT THE WEIGHT HOOK IS FOR. Mode M sums the estimator raw; mode J must scale EACH beam
// hit by its own multiple-importance-sampling weight, because the balance-heuristic ratio
// between "merge this beam" and "connect this vertex" depends on the merge geometry
// (sin(theta) above all) and so differs from hit to hit. A per-hit weight cannot be applied
// by the caller after the fact — by then the hits have been summed.
//
// The hook is a TEMPLATE parameter, not a std::function: this is the innermost loop of the
// mode-M gather, the two ratio-tracking transmittance marches inside it are already the
// dominant cost of a heterogeneous render, and an indirect call per hit would be paid on
// mode M's behalf for a feature mode M does not use. As a template with a
// return-1.0 default functor the constant folds away and mode M's generated code is what it
// was; `gatherPhotonBeams` below is that instantiation, kept under its old name and
// signature so every existing call site is untouched.
#pragma once
#include <cmath>
#include "render.h"
#include "photonbeams.h"

// The default weight: every beam hit counts once, in full. This is mode M, where the beam
// map IS the estimator and there is no second technique to share with.
struct BeamWeightOne {
    double operator()(const BeamHit&, const PhotonBeam&, double, double) const { return 1.0; }
};

// --- Deterministic transmittance, for MIS WEIGHTS ONLY --------------------------------
//
// Renderer::mediaTransmittance is an unbiased ESTIMATOR: for a heterogeneous medium it
// ratio-tracks, so two calls on the same segment return two different numbers. That is
// exactly right inside a contribution and exactly wrong inside a weight.
//
// A balance heuristic is unbiased only if, for one fixed path, the weights of the competing
// techniques sum to 1. Every technique's weight is built from the same pool of per-edge
// densities, and an edge's transmittance appears in several of them; if each appearance is
// an independent random draw the sum is no longer 1 and the image is biased. So the weight
// needs a FUNCTION of the path, not an estimate of one.
//
// Note what that requirement does NOT demand: accuracy. Any deterministic function keeps
// the partition of unity — Tr == 1 everywhere would be unbiased too, merely a poor weight
// with more variance. So the rule here is "deterministic first, accurate second": exact for
// a homogeneous medium (a closed-form exponential, which is also the overwhelmingly common
// case), and a fixed midpoint quadrature of the optical depth for a heterogeneous one,
// where the only cost of the approximation is weight quality.
inline double trDetMedium(const Medium& med, const Vec3& o, const Vec3& dir, double dist,
                          double lambda, const PatTables* tabs) {
    const double stBase = med.sigmaT(lambda);
    if (stBase <= 0.0) return 1.0;
    double ta, tb;
    if (!med.clipToBounds(o, dir, 0.0, dist, ta, tb)) return 1.0;
    const double L = tb - ta;
    if (!(L > 0.0)) return 1.0;
    if (!med.heterogeneous()) return std::exp(-stBase * L);
    // Midpoint rule. Four samples is enough to track the large-scale shape of a cloud,
    // which is all a weight needs; it is not integrating radiance.
    constexpr int kN = 4;
    const double dt = L / (double)kN;
    double tau = 0.0;
    for (int i = 0; i < kN; ++i)
        tau += med.densityAt(o + dir * (ta + (i + 0.5) * dt), tabs);
    return std::exp(-stBase * tau * dt);
}

// Same, over every medium in the scene (extinction adds, so transmittance multiplies).
inline double trDet(const Scene& scene, const Vec3& o, const Vec3& dir, double dist,
                    double lambda, const PatTables& tabs) {
    double Tr = 1.0;
    for (const Medium& m : scene.media) {
        Tr *= trDetMedium(m, o, dir, dist, lambda, &tabs);
        if (Tr <= 0.0) return 0.0;
    }
    return Tr;
}
inline double trDet(const Scene& scene, const Vec3& o, const Vec3& dir, double dist,
                    double lambda) {
    const PatTables tabs = scene.patTables();
    return trDet(scene, o, dir, dist, lambda, tabs);
}

// ---- Volume gather: single-scatter radiance along one camera SEGMENT from the beam map ---
// The Beam x Ray 1D estimator (photonbeams.h). For every stored beam whose kernel cylinder
// the segment [oc, oc + dc*tMax] passes through, add
//
//   Phi_b * K1(d_perp)/sin(theta) * sigma_s(x) * f_p(cos theta)
//         * Tr_beam(0 -> s_b) * Tr_cam(0 -> t_c) / nEmitted
//
// weighted by the CIE response at the BEAM's wavelength — the same "estimate built directly
// in XYZ at the photon's own lambda" trick the surface density estimate uses, so a spectral
// rainbow comes out spectral without any monochromatic reconstruction.
//
// A beam may carry a stratified BUNDLE of wavelengths rather than one (`-beamspec`; see
// PhotonBeam in photonbeams.h for why a monochromatic *line* is so much worse than a
// monochromatic *point*). The bundle shares the chord, the kernel weight, the density
// evaluation and both transmittance marches — i.e. everything this estimator actually spends
// its time on — and differs only in sigma_s, the phase function and the CIE triple.
//
// `aGlassCam` is the absorption of the dielectric the CAMERA ray is currently inside; the
// caller applies it over the whole segment afterwards, so here it is applied only as far as
// each beam's own closest-approach point. `thr` is NOT applied here — the caller multiplies
// the returned XYZ by its specular-chain throughput.
//
// `w1` is the per-hit MIS weight (see the header comment). It is applied to the whole hit,
// primary wavelength and spectral bundle alike, which is correct because the weight is a
// property of the path GEOMETRY — the balance heuristic compares the densities with which
// competing techniques would have generated this vertex, and every wavelength in a beam
// bundle rode the same chord to the same place. It is applied BEFORE the `w > 0` rejection
// so that a technique the weight kills costs no transmittance marches.
//
// Note the two transmittance marches per surviving beam (one along the beam, one back along
// the camera ray). For a heterogeneous medium those are ratio-tracking walks, and they are
// the dominant cost of this estimator; see known-issues.md.
template <class WeightFn>
inline Vec3 gatherPhotonBeamsW(const Scene& scene, const Renderer& mats, const BeamMap& bm,
                               const Vec3& oc, const Vec3& dc, double tMax,
                               double aGlassCam, Pcg32& rng, const WeightFn& w1) {
    Vec3 acc{0, 0, 0};
    if (bm.empty() || bm.nEmitted <= 0) return acc;
    const double invN = 1.0 / (double)bm.nEmitted;
    const PatTables tabs = scene.patTables();
    bm.gather(oc, dc, tMax, [&](const BeamHit& bh) {
        const PhotonBeam& b = bm.beams[bh.idx];
        if (b.med < 0 || b.med >= (int)scene.media.size()) return;
        const double lam = (double)b.lambda;
        const Medium& md = scene.media[b.med];
        const Vec3 xc = oc + dc * bh.tCam;
        // sigma_s AT the gather point — density field / imported volume included, so a
        // heterogeneous cloud shapes the bow instead of a uniform slab of it. The density
        // is wavelength-INDEPENDENT, so one evaluation serves the whole spectral bundle.
        const double dens = md.densityAt(xc, &tabs);
        const double ss = md.sigma_s(lam) * dens;
        if (!(ss > 0.0)) return;
        // Scattering angle. connectVolume's convention: phaseValue(dot(wIn, wToCamera)),
        // wIn = the photon's propagation direction (b.d), wToCamera = -dc.
        const double phase = md.phaseValue(-bh.cosT, lam);
        if (!(phase > 0.0)) return;
        // `invC` splits the chord's flux evenly over its spectral bundle (PhotonBeam: every
        // wavelength in a bundle carries the same power). It is exactly 1.0 for a
        // monochromatic beam, so the whole expression — and thus every pre-0.202.0 render —
        // is bit-for-bit unchanged.
        const double invC = 1.0 / (double)b.nLam();
        double w = (double)b.power * bm.kernel1D(bh.dPerp, b.med) / bh.sinT * ss * phase * invN * invC;
        // MIS (mode J); exactly 1.0 and folded away in mode M. `dens` and `phase` are handed
        // over rather than recomputed: the merge weight needs sigma_t(x) and the phase value
        // at the merge point, and both are one multiply away from what this line just built.
        w *= w1(bh, b, dens, phase);
        if (!(w > 0.0)) return;
        if (b.absorb > 0.0f) w *= std::exp(-(double)b.absorb * bh.sBeam);   // glass, beam side
        if (aGlassCam > 0.0) w *= std::exp(-aGlassCam * bh.tCam);           // glass, camera side
        if (bh.sBeam > 0.0)  w *= mats.mediaTransmittance(scene, b.o, b.d, bh.sBeam, lam, rng);
        if (bh.tCam  > 0.0)  w *= mats.mediaTransmittance(scene, oc, dc, bh.tCam, lam, rng);
        if (!(w > 0.0)) return;
        acc += bm.cie[bh.idx] * w;
        // SECONDARY wavelengths of the bundle. They share this beam's geometry, its kernel
        // weight and BOTH transmittance marches — the bundle only exists on a path whose
        // extinction is achromatic (Renderer::beamSpecC), which is precisely the condition
        // that makes those marches wavelength-independent — so all that differs is
        // sigma_s * phase * CIE. `w / (ss * phase)` recovers the shared factor with one
        // division instead of rebuilding the chain, and both terms are known positive.
        if (b.nSec > 0) {
            const double wShared = w / (ss * phase);
            for (int i = 0; i < b.nSec; ++i) {
                const double li = (double)b.lamS[i];
                const double ssi = md.sigma_s(li) * dens;
                if (!(ssi > 0.0)) continue;
                const double phi = md.phaseValue(-bh.cosT, li);
                if (!(phi > 0.0)) continue;
                acc += Vec3(cieX(li), cieY(li), cieZ(li)) * (wShared * ssi * phi);
            }
        }
    });
    return acc;
}

// Mode M's instantiation: the estimator with no MIS partner, under its historical name and
// signature. Bit-identical to the pre-0.215.0 function it replaces.
inline Vec3 gatherPhotonBeams(const Scene& scene, const Renderer& mats, const BeamMap& bm,
                              const Vec3& oc, const Vec3& dc, double tMax,
                              double aGlassCam, Pcg32& rng) {
    return gatherPhotonBeamsW(scene, mats, bm, oc, dc, tMax, aGlassCam, rng, BeamWeightOne{});
}
