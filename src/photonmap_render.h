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
#include "backward.h"      // BackwardRenderer::neeLight / neeEnv for final-gather direct lighting
#include "scene_film.h"
#include "camera.h"
#include "color.h"
#include "geometry.h"

// ---- Forward photon pass: deposit into the map, no camera splat ---------------------
// Traces N photons across nThreads, each depositing into a private bank, then
// concatenates into pm.photons and records pm.nEmitted (= N). Does NOT build the grid;
// the caller picks the gather radius and calls pm.build(radius).
inline void tracePhotonPass(const Scene& scene, long long N, int nThreads,
                            bool diffraction, PhotonMap& pm) {
    if (nThreads < 1) nThreads = 1;
    std::vector<std::vector<Photon>> banks(nThreads);
    std::vector<long long> emitted(nThreads, 0);

    auto worker = [&](int tid) {
        Renderer r; r.diffraction = diffraction; r.photonDeposit = &banks[tid];
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 31,
                            0xD1B54A32D192ED03ULL ^ (uint64_t)tid);
        long long lo = N * tid / nThreads, hi = N * (tid + 1) / nThreads;
        EnergyReport e;
        for (long long i = lo; i < hi; ++i)
            r.tracePhoton(scene, (const Camera*)nullptr, (Film*)nullptr, (Film*)nullptr, rng, e);
        emitted[tid] = hi - lo;
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    size_t total = 0;
    for (auto& b : banks) total += b.size();
    pm.photons.clear();
    pm.photons.reserve(total);
    pm.nEmitted = 0;
    for (int t = 0; t < nThreads; ++t) {
        pm.photons.insert(pm.photons.end(), banks[t].begin(), banks[t].end());
        pm.nEmitted += emitted[t];
    }
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
                     * (thr * rhoV * m.emit(lambda) * invPdfL);
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
                pm.query(h.p, [&](const Photon& ph, double) {
                    if (dot(ph.n, h.n) < 0.5) return;    // reject cross-surface leakage
                    double rhoY = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                    double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, ph.lambda));
                    double f = rhoY * (1.0 / PI);
                    g += Vec3(cieX(ph.lambda), cieY(ph.lambda), cieZ(ph.lambda))
                         * (f * rhoV * (double)ph.power);
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
                thr *= clamp01(m.transmit(lambda));
                ray = Ray{h.p + ray.d * 1e-6, ray.d};
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
inline Vec3 photonGather(const Scene& scene, const PhotonMap& pm, Ray ray,
                         Pcg32& rng, bool diffraction, int maxBounce, int fgRays = 0) {
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

    for (int b = 0; b < maxBounce; ++b) {
        Hit h = scene.closestHit(ray);
        if (h.valid) {                                   // Beer-Lambert in current medium
            int cm = stk.topMat();
            double a = (cm >= 0) ? scene.mats[cm].absorb(lambda) : 0.0;
            if (a > 0.0) thr *= std::exp(-a * h.t);
        }
        if (!h.valid) {                                  // escaped -> environment
            if (scene.envIndex >= 0)
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * scene.envRadiance(ray.d, lambda) * invPdfL);
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
                 * (thr * m.emit(lambda) * invPdfL);
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
                pm.query(h.p, [&](const Photon& ph, double) {
                    if (dot(ph.n, h.n) < 0.5) return;    // reject cross-surface leakage
                    double rho = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                    double f = rho * (1.0 / PI);
                    g += Vec3(cieX(ph.lambda), cieY(ph.lambda), cieZ(ph.lambda))
                         * (f * (double)ph.power);
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
                double t = clamp01(m.transmit(lambda));
                thr *= t;
                ray = Ray{h.p + ray.d * 1e-6, ray.d};
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
inline Film renderPhotonCamera(const Scene& scene, const Camera& cam, int resX, int resY,
                               const PhotonMap& pm, long long spp, int nThreads,
                               bool diffraction, int maxBounce = 32,
                               unsigned long long seedOffset = 0, int fgRays = 0) {
    if (nThreads < 1) nThreads = 1;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    std::vector<Film> bands(nThreads);
    auto worker = [&](int tid) {
        Film& f = bands[tid]; f.resX = resX; f.resY = resY; f.alloc();
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 23,
                            0xA24BAED4963EE407ULL ^ (uint64_t)tid
                              ^ (seedOffset * 0x9E3779B97F4A7C15ULL));
        int y0 = resY * tid / nThreads, y1 = resY * (tid + 1) / nThreads;
        for (int py = y0; py < y1; ++py)
            for (int px = 0; px < resX; ++px)
                for (long long s = 0; s < spp; ++s) {
                    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
                    f.add(px, py, photonGather(scene, pm, ray, rng, diffraction, maxBounce, fgRays));
                }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    for (int t = 0; t < nThreads; ++t) out.merge(bands[t]);
    return out;
}
