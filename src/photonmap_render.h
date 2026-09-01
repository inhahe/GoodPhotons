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
#include <vector>
#include <thread>
#include <cstdint>
#include "render.h"
#include "photonmap.h"
#include "photonbeams.h"   // volume single-scatter cache (mode M with -beams)
#include "backward.h"      // BackwardRenderer::neeLight / neeEnv for final-gather direct lighting
#include "scene_film.h"
#include "camera.h"
#include "color.h"
#include "geometry.h"
#include "parallel.h"      // ft::stopRequested — cooperative `-stop` inside the pixel loop

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
inline void tracePhotonPass(const Scene& scene, long long N, int nThreads,
                            bool diffraction, PhotonMap& pm, int heroC = hero::kHeroC,
                            uint64_t seedBase = 0, BeamMap* bm = nullptr,
                            long long beamTarget = 0) {
    if (nThreads < 1) nThreads = 1;
    std::vector<PhotonBank> banks(nThreads);
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
        // never depends on — or perturbs — the photon tracer's own RNG sequence.
        bbanks[t].rng.seed(seedBase + 0x9E3779B97F4A7C15ULL * (uint64_t)(t + 1),
                           0xBF58476D1CE4E5B9ULL);
    }

    // Hero-wavelength deposit (modes M/S): each traced path deposits its live wavelengths
    // as per-λ photon records via tracePhotonHero (render.h). nEmitted still counts PATHS
    // (below), so the density estimate is energy-identical to single-λ but with far lower
    // chroma noise. Same gate as the forward tracers: no media / no GRIN (those stay C=1).
    const bool heroOn = (heroC > 1) && scene.media.empty() && !grin::sceneHasGrin(scene);

    auto worker = [&](int tid) {
        Renderer r; r.diffraction = diffraction; r.photonDeposit = &banks[tid];
        if (bm) r.beamDeposit = &bbanks[tid];
        r.useHero = heroOn; r.heroC = heroC;
        Pcg32 rng;
        long long lo = N * tid / nThreads, hi = N * (tid + 1) / nThreads;
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
            if ((done & 0xFFF) == 0 && ft::stopRequested()) break;
            seedUnit(rng, seedBase + (uint64_t)i, 0xEB44ACCAB455D165ULL);
            r.tracePhoton(scene, (const Camera*)nullptr, (Film*)nullptr, (Film*)nullptr, rng, e);
            ++done;
        }
        // Count what was ACTUALLY emitted, not what was asked for. pm.nEmitted normalises the
        // density estimate, so reporting the full share after an early break would scale a
        // truncated pass down by the fraction it never traced and darken the image.
        emitted[tid] = done;
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    size_t total = 0;
    for (auto& b : banks) total += b.size();
    pm.photons.clear();  pm.photons.reserve(total);
    pm.pos.clear();      pm.pos.reserve(total);
    pm.nEmitted = 0;
    for (int t = 0; t < nThreads; ++t) {
        // Append both halves in the same thread order, so pos[k] stays the position of
        // photons[k] (PhotonMap's split layout — see photonmap.h).
        pm.photons.insert(pm.photons.end(), banks[t].payload.begin(), banks[t].payload.end());
        pm.pos.insert(pm.pos.end(), banks[t].pos.begin(), banks[t].pos.end());
        pm.nEmitted += emitted[t];
    }
    if (bm) {
        size_t nb = 0;
        for (auto& b : bbanks) nb += b.size();
        bm->beams.clear(); bm->beams.reserve(nb);
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
// `aGlassCam` is the absorption of the dielectric the CAMERA ray is currently inside; the
// caller applies it over the whole segment afterwards, so here it is applied only as far as
// each beam's own closest-approach point. `thr` is NOT applied here — the caller multiplies
// the returned XYZ by its specular-chain throughput.
//
// Note the two transmittance marches per surviving beam (one along the beam, one back along
// the camera ray). For a heterogeneous medium those are ratio-tracking walks, and they are
// the dominant cost of this estimator; see known-issues.md.
inline Vec3 gatherPhotonBeams(const Scene& scene, const Renderer& mats, const BeamMap& bm,
                              const Vec3& oc, const Vec3& dc, double tMax,
                              double aGlassCam, Pcg32& rng) {
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
        // heterogeneous cloud shapes the bow instead of a uniform slab of it.
        const double ss = md.sigma_s(lam) * md.densityAt(xc, &tabs);
        if (!(ss > 0.0)) return;
        // Scattering angle. connectVolume's convention: phaseValue(dot(wIn, wToCamera)),
        // wIn = the photon's propagation direction (b.d), wToCamera = -dc.
        const double phase = md.phaseValue(-bh.cosT, lam);
        if (!(phase > 0.0)) return;
        double w = (double)b.power * bm.kernel1D(bh.dPerp) / bh.sinT * ss * phase * invN;
        if (!(w > 0.0)) return;
        if (b.absorb > 0.0f) w *= std::exp(-(double)b.absorb * bh.sBeam);   // glass, beam side
        if (aGlassCam > 0.0) w *= std::exp(-aGlassCam * bh.tCam);           // glass, camera side
        if (bh.sBeam > 0.0)  w *= mats.mediaTransmittance(scene, b.o, b.d, bh.sBeam, lam, rng);
        if (bh.tCam  > 0.0)  w *= mats.mediaTransmittance(scene, oc, dc, bh.tCam, lam, rng);
        if (!(w > 0.0)) return;
        acc += bm.cie[bh.idx] * w;
    });
    return acc;
}

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
inline Vec3 photonGatherSub(const Scene& scene, const PhotonMap& pm, Ray ray, Pcg32& rng,
                            bool diffraction, int maxBounce, double lambda, double invPdfL,
                            double norm, const Hit& visHit, const Material& visMat) {
    Vec3 L{0, 0, 0};
    double thr = 1.0;
    bool specularSeen = false;                           // any specular bounce so far?
    Renderer mats; mats.diffraction = diffraction;
    MediumStack stk;                                     // nested-dielectric medium stack

    for (int b = 0; b < maxBounce; ++b) {
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
            int c = mixPickChild(*mp, rng.uniform());
            if (c < 0) return L;
            mp = &scene.mats[c];
        }
        const Material& m = *mp;

        if (m.isLight) {                                 // hit an emitter
            if (specularSeen) {                          // specular-direct: NEE can't reach it
                double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, lambda));
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * rhoV * emitSlot(scene, m, h, lambda) * invPdfL);
            }
            return L;                                     // else: direct handled by NEE at vis
        }

        switch (m.type) {
            case MatType::Diffuse:
            case MatType::DiffuseTransmit:
            case MatType::Fluorescent: {
                // Density estimate at y, folding the visible-point reflectance per photon
                // wavelength: L_o(vis) += rho(vis,l_p) * [rho(y,l_p)/pi] * Phi_p / (pi r^2 N).
                Vec3 g{0, 0, 0};
                pm.query(h.p, [&](const Photon& ph, double, int k) {
                    if (dot(ph.n, h.n) < 0.5) return;    // reject cross-surface leakage
                    double rhoY = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                    double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, ph.lambda));
                    double f = rhoY * (1.0 / PI);
                    g += pm.cie[k] * (f * rhoV * (double)ph.power);   // == cie(lambda_p), precomputed
                });
                L += g * (norm * thr);
                return L;
            }
            case MatType::Mirror: {
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
            case MatType::Glossy: {
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0.0) return L;
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
inline Vec3 photonGather(const Scene& scene, const PhotonMap& pm, Ray ray,
                         Pcg32& rng, bool diffraction, int maxBounce, int fgRays = 0,
                         const BeamMap* bm = nullptr) {
    Vec3 L{0, 0, 0};
    double thr = 1.0;
    double pdfL = 0.0;
    double lambda = scene.emitSampler.sample(rng, pdfL);
    if (pdfL <= 0.0) return L;
    const double invPdfL = scene.invPdfLambda(lambda);

    Renderer mats; mats.diffraction = diffraction;
    MediumStack stk;                                     // nested-dielectric medium stack
    const double area = PI * pm.radius * pm.radius;
    const double norm = (pm.nEmitted > 0 && area > 0.0)
                            ? 1.0 / (area * (double)pm.nEmitted) : 0.0;

    const bool volOn = (bm != nullptr) && !bm->empty() && !scene.media.empty();

    for (int b = 0; b < maxBounce; ++b) {
        Hit h = scene.closestHit(ray);
        const int    cmIdx  = stk.topMat();
        const double aGlass = (cmIdx >= 0) ? scene.mats[cmIdx].absorb(lambda) : 0.0;
        // --- Participating media along this segment (mode M with -beams) ---------------
        // Done BEFORE `thr` takes the segment's attenuation, because each gathered beam
        // needs the transmittance to ITS OWN closest-approach point, not to the segment end.
        if (volOn) {
            const double dSeg = h.valid ? h.t : 1e30;
            L += gatherPhotonBeams(scene, mats, *bm, ray.o, ray.d, dSeg, aGlass, rng)
                 * thr;
            // Extinction along the camera segment: what is behind the fog gets dimmed.
            thr *= mats.mediaTransmittance(scene, ray.o, ray.d, dSeg, lambda, rng);
            if (thr <= 0.0) return L;
        }
        if (h.valid) {                                   // Beer-Lambert in current medium
            if (aGlass > 0.0) thr *= std::exp(-aGlass * h.t);
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
                     * (thr * scene.sunRadiance(ray.d, lambda) * invPdfL);
            return L;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return L;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            int c = mixPickChild(*mp, rng.uniform());    // approximate: gather the body lobe
            if (c < 0) return L;
            mp = &scene.mats[c];
        }
        const Material& m = *mp;

        if (m.isLight) {                                 // directly-viewed emitter
            L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                 * (thr * emitSlot(scene, m, h, lambda) * invPdfL);
            return L;
        }

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
                                              lambda, invPdfL, norm, h, m);
                    }
                    L += fg * (thr * (1.0 / (double)fgRays));
                    return L;
                }
                // Direct radius density estimate (default; also DiffuseTransmit/Fluorescent
                // visible points, which fall back here rather than final-gathering):
                //   L_r(x) = (1/N) sum_p f_r * Phi_p / (pi r^2), f_r = rho/pi (Lambertian),
                // accumulated in XYZ per photon wavelength.
                Vec3 g{0, 0, 0};
                pm.query(h.p, [&](const Photon& ph, double, int k) {
                    if (dot(ph.n, h.n) < 0.5) return;    // reject cross-surface leakage
                    double rho = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                    double f = rho * (1.0 / PI);
                    g += pm.cie[k] * (f * (double)ph.power);          // == cie(lambda_p), precomputed
                });
                L += g * (norm * thr);
                return L;
            }
            case MatType::Mirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
            case MatType::Glossy: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0.0) return L;
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
                ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Hair: {
                // Fiber BCSDF — scattered through, never a gather site (the photon
                // payload has no incident direction; see photonGatherSub's Hair case).
                const Vec3 wPrev{-ray.d.x, -ray.d.y, -ray.d.z};
                const HairShade hs = hairShadeAt(scene, m, h, lambda, wPrev);
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
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
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
                               const BeamMap* bm = nullptr) {
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
                    f.add(px, py, photonGather(scene, pm, ray, rng, diffraction, maxBounce,
                                               fgRays, bm));
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
