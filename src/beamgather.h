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
#include <mutex>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "render.h"
#include "photonbeams.h"
#include "volcache.h"   // VOLCACHE prototype (flag-gated; unused unless FTRACE_VOLCACHE)

// The default weight: every beam hit counts once, in full. This is mode M, where the beam
// map IS the estimator and there is no second technique to share with.
//
// `kFoldGatherTime` says whether this weight lets the GATHER-TIME SPECTRAL FOLD (0.256.0,
// Scene::BowLut, PhotonBeam::achro == 2) be taken. Folding integrates λ out of the
// contribution, which is an exact substitution only if NOTHING ELSE in the term depends on λ.
// A weight of exactly 1 satisfies that trivially, so mode M folds. A MIS weight does not — see
// the long note at the fold's use site below — so mode J does not. In mode J the practical
// reason is even simpler than the theoretical one, and it is measured: a folded beam cannot
// also be a `-beamspec` bundle, and in mode J the bundle is worth far more.
struct BeamWeightOne {
    static constexpr bool kFoldGatherTime = true;
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

// --- The same transmittance, evaluated at MANY distances along ONE fixed ray -----------
//
// `trDet` re-derives, on every call, two things that depend only on the ray and not on the
// distance: which media it crosses (a ray/bounds clip per medium) and each one's sigma_t at
// the wavelength (a spectral curve evaluation). Mode `J`'s merge weight calls it once per
// beam HIT along a single camera segment, and a dense medium hands that segment hundreds of
// hits — 252 at the auto-tuned radius on `_fog_thick.ftsl` — so essentially all of that is
// per-ray work being paid per-hit. `TrRay` hoists it: build once, then each evaluation is
// one `exp` per crossed medium, which is all `trDet`'s own arithmetic reduces to.
//
// It is BIT-IDENTICAL to `trDet` rather than merely close, which matters because these are
// MIS weights: the clip is `[ta, min(tb, t)]` either way (`clipToBounds` intersects the
// medium's own interval with `[0, dist]`, so clipping to `tMax` and then to `t <= tMax` is
// the same interval), the surviving media are multiplied in scene order, and a medium that
// `trDet` would skip contributes exactly 1.0. Gate 1 checks this — mode `J` with an empty
// map must stay `cmp`-identical to mode `D`.
//
// A HETEROGENEOUS medium keeps the slow path. Its 4-point midpoint quadrature samples the
// density at positions that depend on the interval, so nothing about it can be hoisted;
// `slow` then routes every evaluation on this ray straight back to `trDet`.
struct TrRay {
    static constexpr int kMax = 8;
    struct Crossed { double sigT, ta, tb; };
    Crossed xs[kMax];
    int n = 0;
    bool slow = false;                 // heterogeneous medium, or more than kMax media
    const Scene* scene = nullptr;
    const PatTables* tabs = nullptr;
    Vec3 o{0, 0, 0}, d{0, 0, 0};
    double lambda = 0.0;

    void build(const Scene& sc, const Vec3& oo, const Vec3& dd, double tMax, double lam,
               const PatTables& tb) {
        scene = &sc; tabs = &tb; o = oo; d = dd; lambda = lam;
        n = 0; slow = false;
        if ((int)sc.media.size() > kMax) { slow = true; return; }
        for (const Medium& m : sc.media) {
            if (m.heterogeneous()) { slow = true; n = 0; return; }
            const double st = m.sigmaT(lam);
            if (st <= 0.0) continue;                       // trDet's 1.0 factor
            double ta, tb2;
            if (!m.clipToBounds(oo, dd, 0.0, tMax, ta, tb2)) continue;
            if (!(tb2 - ta > 0.0)) continue;
            xs[n].sigT = st; xs[n].ta = ta; xs[n].tb = tb2; ++n;
        }
    }

    double at(double t) const {
        if (slow) return trDet(*scene, o, d, t, lambda, *tabs);
        double Tr = 1.0;
        for (int i = 0; i < n; ++i) {
            const double hi = t < xs[i].tb ? t : xs[i].tb;
            const double L = hi - xs[i].ta;
            if (!(L > 0.0)) continue;                      // trDet's 1.0 factor
            Tr *= std::exp(-xs[i].sigT * L);
            if (Tr <= 0.0) return 0.0;
        }
        return Tr;
    }
};

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
// Read once; a gather runs millions of times and getenv is not free in a loop.
// VOLCACHE PROTOTYPE, gather side. One cache per BeamMap, built on first use. The prototype
// is exact only where a scalar fluence cache can be: an ISOTROPIC (g == 0) and HOMOGENEOUS
// medium. `volCacheUsable` checks both and the cache stays off otherwise, so a scene it cannot
// serve renders normally rather than wrongly.
namespace vcstate {
inline VolCache&      cache()    { static VolCache c;                 return c; }
inline std::mutex&    mtx()      { static std::mutex m;               return m; }
// What the live cache was built from. Keyed on the map POINTER **and** its contents, because a
// pointer alone is not a key here: mode M rebuilds the light side into the SAME BeamMap object
// every epoch, so a pointer-only guard serves epoch 0's cache forever while the beams under it
// change. The v0.299.0 prototype had exactly that bug; it never showed because every measurement
// was a single-epoch plain render, which is precisely the regime where the bug is invisible.
struct Key { const BeamMap* p = nullptr; size_t n = 0; long long emitted = -1;
             // Set when the DEPOSIT SPLIT built this cache. The split owns the cache outright
             // and the content key CANNOT arbitrate afterwards: the split runs before
             // BeamMap::build, which then splits the surviving chords into sub-beams, so
             // `beams.size()` changes underneath the key (28 786 chords -> 556 033 sub-beams on
             // the measured scene). A content-keyed guard therefore sees a "different" map at
             // gather time and rebuilds from the order-1 remainder -- producing an EMPTY cache
             // and silently discarding the order >= 2 energy the split had just routed into it.
             // That is not hypothetical: it is what v0.300.0's first build did, and it read as a
             // 62 % speedup because the march was contributing nothing at all.
             bool fromSplit = false; };
inline Key& builtFor() { static Key k; return k; }
inline Key keyOf(const BeamMap& bm) { return Key{&bm, bm.beams.size(), (long long)bm.nEmitted}; }
inline bool same(const Key& a, const Key& b) {
    return a.p == b.p && a.n == b.n && a.emitted == b.emitted;
}
}  // namespace vcstate

// The media gate, in ONE place so the query path and the split path cannot come to different
// conclusions about which scenes this cache may serve. A scalar fluence is only exact for an
// isotropic phase function (see volcache.h), and the attenuation term assumes homogeneity.
inline bool volCacheMediaOk(const Scene& sc, double& sigmaT, double& gOut) {
    bool ok = false, first = true;
    sigmaT = 0.0; gOut = 0.0;
    for (const auto& m : sc.media) {
        if (!m.enabled) continue;
        // A RAINBOW phase is a wavelength-dependent Airy table, not Henyey-Greenstein, so its
        // Legendre moments are neither g^l nor achromatic and the SH convolution below does
        // not describe it. Refuse rather than serve it an HG reconstruction.
        if (m.rainbow() || m.heterogeneous()) return false;
        const double st = m.sigma_a(550.0) + m.sigma_s(550.0);
        if (first) { sigmaT = st; gOut = m.g; first = false; ok = true; }
        // ONE grid carries ONE (sigma_t, g). Two media that disagree cannot both be served by
        // it, and the previous version silently applied the LAST enabled medium's sigma_t to
        // every beam -- an error that reads as a soft bias, not as a failure.
        else if (st != sigmaT || m.g != gOut) return false;
    }
    return ok;
}

inline VolCache& volCacheFor(const Scene& sc, const BeamMap& bm) {
    std::lock_guard<std::mutex> lk(vcstate::mtx());
    const vcstate::Key k = vcstate::keyOf(bm);
    // The split is authoritative when it ran: never second-guess it from the map's contents,
    // because after BeamMap::build those contents no longer describe what the cache holds.
    if (!vcstate::builtFor().fromSplit && !vcstate::same(vcstate::builtFor(), k)) {
        double sigmaT = 0.0, gHG = 0.0;
        const bool ok = volCacheMediaOk(sc, sigmaT, gHG);
        vcstate::cache().build(bm, volCacheRes(), 2, ok, sigmaT, gHG);
        vcstate::builtFor() = k;
    }
    return vcstate::cache();
}

// THE DEPOSIT SPLIT. Build the cache from the raw chords, then ERASE the ones it now carries.
//
// Must run BEFORE BeamMap::build, and that ordering is the whole feature: everything expensive
// about a beam -- the SAH split, the CIE fold, the inflated box, the BVH node it occupies and the
// traversal that later walks it -- happens in or after that call. A chord removed here costs
// nothing anywhere downstream, whereas the v0.299.0 query-side skip removed only the last of
// those. Returns the number of chords routed into the cache (0 if the split did not run).
//
// `-beams-minorder 2` is NOT the cost floor for this path, and the first version of this comment
// claimed it was. The two filter at different points in the pipeline and that changes the
// POPULATION, not just the timing: `-beams-minorder` drops at `push`, so the `-beamcount` budget
// and the per-thread bank self-halving are spent entirely on order-1 chords, while the split runs
// after the budget has already been apportioned across all orders and keeps only order-1's share.
// Measured on this scene at `-beamcount 100000`: the split leaves 28 786 chords, `-beams-minorder
// 2` leaves 99 909 -- 3.5x more -- so it renders a cleaner order-1 estimate more slowly and is a
// different experiment, not a bound on this one. The honest control is the baseline, which holds
// the order-1 population fixed by construction.
inline size_t volCacheSplit(const Scene& sc, BeamMap& bm) {
    if (volCacheRes() <= 0 || !volCacheSplitEnabled()) return 0;
    std::lock_guard<std::mutex> lk(vcstate::mtx());
    double sigmaT = 0.0, gHG = 0.0;
    const bool ok = volCacheMediaOk(sc, sigmaT, gHG);
    VolCache& c = vcstate::cache();
    c.build(bm, volCacheRes(), 2, ok, sigmaT, gHG);
    vcstate::builtFor() = vcstate::keyOf(bm);
    if (!c.ready) return 0;   // gate refused: leave fromSplit false so the query path still works     // gate refused (anisotropic / heterogeneous): keep every chord
    const size_t before = bm.beams.size();
    size_t w = 0;
    for (size_t i = 0; i < before; ++i) {
        const PhotonBeam& b = bm.beams[i];
        // The cache took exactly the chords VolCache::build splatted: known order, >= 2.
        if (b.order != kBeamOrderUnknown && (int)b.order >= 2) continue;
        bm.beams[w++] = bm.beams[i];
    }
    bm.beams.resize(w);
    // The key must be re-taken AFTER the erase, or the gather's content-keyed guard sees a
    // different beam count and rebuilds the cache from the remainder -- which is the order < 2
    // set, i.e. it would quietly replace the cache with an empty one.
    vcstate::builtFor() = vcstate::keyOf(bm);
    vcstate::builtFor().fromSplit = true;
    return before - w;
}

inline int volCacheStubSteps() {
    static const int n = [] {
        const char* e = std::getenv("FTRACE_VOLCACHE_STUB");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    return n;
}
inline double& volCacheSink() { static thread_local double s = 0.0; return s; }

template <class WeightFn>
inline Vec3 gatherPhotonBeamsW(const Scene& scene, const Renderer& mats, const BeamMap& bm,
                               const Vec3& oc, const Vec3& dc, double tMax,
                               double aGlassCam, Pcg32& rng, const WeightFn& w1) {
    Vec3 acc{0, 0, 0};
    if (bm.empty() || bm.nEmitted <= 0) return acc;
    // VOLCACHE: replace the order >= 2 beam queries with a march of the cached fluence.
    // `phase` is the ISOTROPIC 1/(4 pi) -- see volcache.h for why a scalar cache admits no
    // other, and why the build refuses anisotropic media rather than pretending.
    const bool vcLive = volCacheRes() > 0 && volCacheFor(scene, bm).ready;
    if (vcLive) {
        const VolCache& vc = volCacheFor(scene, bm);
        const PatTables vcTabs = scene.patTables();
        const int steps = 64;
        const double dt = tMax / (double)steps;
        // The direction the scattered light travels TOWARD THE CAMERA. The gather's phase
        // argument is dot(b.d, -dc) (see the note at the phaseValue call), so -dc is the
        // direction the SH convolution must be evaluated at. Getting this sign wrong would
        // mirror the phase lobe -- forward scattering would read as backward -- and on an
        // isotropic medium it would be invisible, because l = 0 has no direction to mirror.
        const Vec3 wOut = dc * -1.0;
        for (int i = 0; i < steps; ++i) {
            const double t = dt * (i + 0.5);
            const Vec3 p = oc + dc * t;
            Vec3 fl;
            if (!vc.inScatter(p, wOut, fl)) continue;
            for (size_t mi = 0; mi < scene.media.size(); ++mi) {
                const auto& md = scene.media[mi];
                if (!md.enabled) continue;
                const double ss = md.sigma_s(550.0) * md.densityAt(p, &vcTabs);
                if (!(ss > 0.0)) continue;
                const double T = mats.mediaTransmittance(scene, oc, dc, t, 550.0, rng);
                acc += fl * (ss * T * dt);   // phase already folded in by inScatter
            }
        }
    }
    // VOLCACHE COST STUB (FTRACE_VOLCACHE_STUB=<steps>, 0 = off). Marches the camera ray with
    // `steps` uniform samples, each doing one hashed 3D grid read, and throws the result away.
    // It renders nothing and answers one question: what would a cached-field march COST in place
    // of the beam queries VOLCACHE would remove? That term cannot come from instrumentation --
    // every counter in this file measures work that already happens -- and it is charged at every
    // optical depth while the saving shrinks with depth, so it decides the thin-media case.
    // A stub is enough because the cost is the memory traffic and the step count, not the values.
    if (const int vcSteps = volCacheStubSteps()) {
        static const std::vector<float> grid = [] {           // ~1 MB, sized like a real cache
            std::vector<float> g((size_t)64 * 64 * 64 * 4);
            for (size_t i = 0; i < g.size(); ++i) g[i] = (float)(i & 255) * (1.0f / 255.0f);
            return g;
        }();
        double sink = 0.0;
        const double dt = tMax / (double)vcSteps;
        for (int i = 0; i < vcSteps; ++i) {
            const Vec3 p = oc + dc * (dt * (i + 0.5));
            const int ix = (int)((p.x * 7.3 + 4096.0)) & 63;
            const int iy = (int)((p.y * 7.3 + 4096.0)) & 63;
            const int iz = (int)((p.z * 7.3 + 4096.0)) & 63;
            const size_t k = (((size_t)iz * 64 + iy) * 64 + ix) * 4;
            sink += grid[k] + grid[k + 1] + grid[k + 2] + grid[k + 3];
        }
        volCacheSink() += sink;      // keeps the loop from being optimised away
    }
    const double invN = 1.0 / (double)bm.nEmitted;
    const PatTables tabs = scene.patTables();
    bm.gather(oc, dc, tMax, [&](const BeamHit& bh) {
        const PhotonBeam& b = bm.beams[bh.idx];
        if (vcLive && b.order >= 2 && b.order != kBeamOrderUnknown) return;  // in the cache
        beamDiag().bump(beamDiag().pass);
        if (beamDiag().on && b.order >= 2 && b.order != kBeamOrderUnknown)
            beamDiag().passMS.fetch_add(1, std::memory_order_relaxed);
        if (b.med < 0 || b.med >= (int)scene.media.size()) {
            beamDiag().bump(beamDiag().rejMed); return;
        }
        const double lam = (double)b.lambda;
        const Medium& md = scene.media[b.med];
        const Vec3 xc = oc + dc * bh.tCam;
        // sigma_s AT the gather point — density field / imported volume included, so a
        // heterogeneous cloud shapes the bow instead of a uniform slab of it. The density
        // is wavelength-INDEPENDENT, so one evaluation serves the whole spectral bundle.
        const double dens = md.densityAt(xc, &tabs);
        const double ss = md.sigma_s(lam) * dens;
        if (!(ss > 0.0)) { beamDiag().bump(beamDiag().rejSS); return; }
        // Scattering angle. connectVolume's convention: phaseValue(dot(wIn, wToCamera)),
        // wIn = the photon's propagation direction (b.d), wToCamera = -dc.
        // THE GATHER-TIME SPECTRAL FOLD (scene.h, Scene::BowLut; PhotonBeam::achro == 2).
        // This beam's path was wavelength-independent, but its medium's phase is a rainbow
        // table, so the fold could not be taken at deposit time — the colour is not decidable
        // until the scattering angle is known, and it is known right here. Substitute the
        // whole spectral integral for the single sample: `bowCie * bowPhase` IS
        // integral of spd*CIE*p(cos,lambda) dlambda, so multiplying `bowCie` by a `w` built
        // from `bowPhase` reproduces it exactly while every other factor stays where it was.
        // sigma_s and both transmittance marches are evaluated at `lam` as before, which is
        // sound precisely because BowLut is only built for media with flat coefficients.
        //
        // WHY THIS IS GATED ON THE WEIGHT (WeightFn::kFoldGatherTime). Folding is a
        // Rao-Blackwellisation: it replaces CIE(lambda)*p(cos,lambda) by its conditional
        // expectation over lambda. That substitution is EXACT — unbiased and variance-reducing
        // — only if lambda appears NOWHERE ELSE in the term. Mode M satisfies that exactly: its
        // weight is the constant 1. Mode J does not: the merge's MIS weight is a ratio of path
        // densities and one of them carries the phase function, so w1 is itself a function of
        // lambda and E[w1(l)*CIE(l)*p(l)] != E[w1]*E[CIE*p]. The residual is Cov_lambda(w1,
        // CIE*p), and folding the weight too would not remove it: the connection techniques
        // evaluate their densities at the camera's hero wavelength, so a band-averaged merge
        // weight would stop summing to 1.
        //
        // HONEST ABOUT THE MAGNITUDE: that covariance term is NOT measurable here. Isolating it
        // (bundle suppressed in both arms, `-beamspec 1`) on gallery_rain's rain against the
        // 6212-spp mode-D reference, relative RMSE was 0.4157 folded vs 0.4263 unfolded at seed
        // 7, and 0.4875 vs 0.5033 at seed 11 — the fold is slightly BETTER on both seeds, so
        // whatever bias it carries is below the variance it removes. The gate is therefore
        // justified by a different fact, and that one IS decisive: BeamBank::push cannot store a
        // fold and a `-beamspec` bundle in the same record, and in mode J the bundle wins by a
        // mile (0.2587 / 0.2788 vs the numbers above). So mode J declines the fold at DEPOSIT
        // time and keeps its bundle; this gate is the belt to that braces, and it is what makes
        // the gather bit-identical to pre-0.256.0 should a folded bank reach it anyway (a
        // `-loadmap` of a map some other mode wrote). See known-issues.md, UPBP-BOWFOLD.
        const Scene::BowLut* bow =
            (WeightFn::kFoldGatherTime && b.achro == 2)
                ? scene.bowLut((int)b.emIdx, b.med) : nullptr;
        Vec3   bowCie{0, 0, 0};
        double phase;
        if (bow) bow->eval(-bh.cosT, bowCie, phase);
        else     phase = md.phaseValue(-bh.cosT, lam);
        if (!(phase > 0.0)) { beamDiag().bump(beamDiag().rejPh); return; }
        // Belt and braces: `bow` is null under any MIS weight by the gate above, so this is
        // just `phase`. It stays because the invariant it encodes is the important one — a
        // band-averaged phase belongs in the ESTIMATOR and nowhere else; every density a MIS
        // weight compares is the hero wavelength's. (Mode M's BeamWeightOne ignores the
        // argument entirely, so this costs nothing there and is bit-identical.)
        const double phaseMis = bow ? md.phaseValue(-bh.cosT, lam) : phase;
        // `invC` splits the chord's flux evenly over its spectral bundle (PhotonBeam: every
        // wavelength in a bundle carries the same power). It is exactly 1.0 for a
        // monochromatic beam, so the whole expression — and thus every pre-0.202.0 render —
        // is bit-for-bit unchanged.
        const double invC = 1.0 / (double)b.nLam();
        double w = (double)b.power * bm.kernel1D(bh.dPerp, b.med) / bh.sinT * ss * phase * invN * invC;
        // MIS (mode J); exactly 1.0 and folded away in mode M. `dens` and `phase` are handed
        // over rather than recomputed: the merge weight needs sigma_t(x) and the phase value
        // at the merge point, and both are one multiply away from what this line just built.
        w *= w1(bh, b, dens, phaseMis);
        if (!(w > 0.0)) { beamDiag().bump(beamDiag().rejW); return; }
        if (b.absorb > 0.0f) w *= std::exp(-(double)b.absorb * bh.sBeam);   // glass, beam side
        if (aGlassCam > 0.0) w *= std::exp(-aGlassCam * bh.tCam);           // glass, camera side
        if (bh.sBeam > 0.0)  w *= mats.mediaTransmittance(scene, b.o, b.d, bh.sBeam, lam, rng);
        if (bh.tCam  > 0.0)  w *= mats.mediaTransmittance(scene, oc, dc, bh.tCam, lam, rng);
        if (!(w > 0.0)) { beamDiag().bump(beamDiag().rejTr); return; }
        if (beamDiag().on) {
            beamDiag().wAll.fetch_add(w, std::memory_order_relaxed);
            if (b.order >= 2 && b.order != kBeamOrderUnknown)
                beamDiag().wMS.fetch_add(w, std::memory_order_relaxed);
        }
        acc += (bow ? bowCie : bm.cie[bh.idx]) * w;
        // SECONDARY wavelengths of the bundle. They share this beam's geometry, its kernel
        // weight and BOTH transmittance marches — the bundle only exists on a path whose
        // extinction is achromatic (Renderer::beamSpecC), which is precisely the condition
        // that makes those marches wavelength-independent — so all that differs is
        // sigma_s * phase * CIE, and each member's own accumulated spectral weight `wS`.
        // `w / (ss * phase)` recovers the shared factor with one division instead of
        // rebuilding the chain, and both terms are known positive.
        //
        // `wS[i]` is T(lamS[i])/T(lambda) — what the path's spectral factors did to member i
        // RELATIVE to the hero (photonbeams.h; the tracer-side accumulator is render.h's
        // `specW`). It is exactly 1 on a path that never met such a factor, which is every
        // bundle a pre-0.257.0 build could deposit, so those renders are bit-identical.
        if (b.nSec > 0) {
            const double wShared = w / (ss * phase);
            for (int i = 0; i < b.nSec; ++i) {
                const double li = (double)b.lamS[i];
                const double ssi = md.sigma_s(li) * dens;
                if (!(ssi > 0.0)) continue;
                const double phi = md.phaseValue(-bh.cosT, li);
                if (!(phi > 0.0)) continue;
                acc += Vec3(cieX(li), cieY(li), cieZ(li)) *
                       (wShared * (double)b.wS[i] * ssi * phi);
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
