// Backward path tracer — an INDEPENDENT reference renderer for validating the
// forward light tracer (model B). It shoots rays from the camera and estimates
// incident spectral radiance per pixel using next-event estimation (NEE) to the
// area light. The forward and backward estimators sample the same scene with
// opposite transport; at convergence they must produce the same image (up to a
// single global scale set by each side's measurement convention). A structured
// residual after best-fit scaling therefore flags a transport/camera bug — which
// energy conservation alone cannot catch.
//
// It deliberately REUSES the material primitives (Fresnel via Renderer::
// refractOrReflect, the glossy lobe via sampleGlossy) so the two renderers agree
// on materials by construction. That isolates the validation to the part that
// actually differs — the transport and camera math (connect / G / We).
//
// Material treatment mirrors the forward tracer exactly:
//   Diffuse    : NEE to the light + cosine-sampled continuation, Russian roulette
//                on the albedo (throughput unchanged on survival).
//   Mirror     : specular reflect, RR on reflectance; emission added on hit.
//   Glossy     : power-cosine lobe around the mirror dir, RR on reflectance.
//   HalfMirror : stochastic reflect/transmit, lossless.
//   Dielectric : Fresnel-weighted reflect/refract, lossless.
//   Fluorescent: bispectral reradiation (Stokes shift). The elastic base reflects
//                at the output wavelength; the fluorescent channel excites at a
//                separately-sampled input wavelength lambdaIn and reradiates at the
//                output wavelength (colour ~ M). Direct excitation is NEE'd; indirect
//                excitation is carried by a stochastic wavelength-switching
//                continuation. This is the unbiased backward adjoint of the forward
//                tracer's fluoroInteract(), so -scene fluoro now validates with modes
//                R/V (previously fluoro was forward-only).
// Participating media (scene.backwardMedium() / -fog) IS supported here: camera and
// scattered rays sample volume free-flight, and volume vertices do phase-function
// NEE to the light (neeVolume). So -fog CAN be combined with modes R/V, which is
// how the forward fog transport is cross-validated.
// Emission is added only when a light is reached via the camera ray or a
// specular/near-specular bounce; diffuse arrivals are covered by NEE (no double
// counting).
// An environment light (scene.envIndex >= 0) is treated as an infinitely-distant
// light: diffuse and fog-scatter vertices do env NEE (neeEnv / neeEnvVolume) by
// importance-sampling the sky's luminance CDF, MIS-combined (balance heuristic) with
// the BSDF-sampled continuation that reaches the sky on a ray miss — the miss term
// is added at full weight only on a camera/specular arrival and MIS-weighted after a
// diffuse/volume bounce. All of this is skipped when the scene has no env light, so
// non-env scenes keep a bit-identical RNG stream / backward image.
#pragma once
#include "scene.h"
#include "camera.h"
#include "render.h"   // sampleGlossy, Renderer::refractOrReflect, clamp01, PI
#include "medium_stack.h" // nested-dielectric priority stack
#include "grin.h"     // shared gradient-index (GRIN) Eikonal marcher
#include "hero.h"     // hero-wavelength spectral sampling (kHeroC)

struct BackwardRenderer {
    int maxBounce = 32;
    bool diffraction = true;   // mirrors Renderer::diffraction for MatType::Grating
    int  heroC = hero::kHeroC;  // wavelengths bundled per camera path when hero is on
                                // (runtime -heroc N, clamped to [1, kHeroMax]; 1 = single-λ)

    // Per-sample cache of emitter-SPD evaluations at the path's wavelengths. The
    // wavelengths are fixed for the whole camera path, but every NEE connection
    // used to re-evaluate em.spdFn(λ) — a std::function typically wrapping the
    // full Planck formula (exp + pow per call) — per emitter per bounce, and
    // Scene::invPdfLambda re-evaluated ALL emitters' SPDs per wavelength again at
    // sample setup. Profiling mode R showed the spdFn dispatch + blackbody math
    // at ~40% of CPU time. renderRows now evaluates spdFn once per (emitter,
    // wavelength) per sample; NEE and the invPdf setup read the table. `lam`
    // points at the pristine per-sample wavelength array; a fluorescent bounce
    // can REWRITE the path's local wavelength (Stokes shift), so readers must
    // check matches() and fall back to a live spdFn evaluation when the path's
    // wavelengths no longer equal the cached ones. Cached values are the exact
    // doubles spdFn returned and the consumers keep the same iteration order and
    // arithmetic shape, so images stay bit-identical.
    struct SpdCache {
        const double* lam = nullptr;  // wavelengths the table was built for
        const double* spd = nullptr;  // emitter-major: spd[e*C + i] = emitters[e].spdFn(lam[i])
        int C = 0;
        bool matches(const double* l, int nUp) const {
            for (int i = 0; i < nUp; ++i) if (l[i] != lam[i]) return false;
            return true;
        }
    };

    // Next-event estimation: connect a surface vertex to each area emitter (the
    // integral splits by light, summed unbiasedly). `invPdfLambda` = emitG/g(lambda)
    // is the reciprocal of the sampled-wavelength pdf; multiplied by an emitter's
    // SPD(lambda) it yields that emitter's Le/pdf weight (= its SPD integral for a
    // single light, matching the forward tracer's photon-weight convention).
    //
    // Shared per-emitter geometry sample for a SURFACE next-event connection. Draws
    // rng identically to the old inline neeLight body (0 draws for a collimated beam or
    // point-spot, 2 for an area light), so the scalar and hero NEE can share one
    // visibility sample. On success returns the λ-INDEPENDENT geometry weight `w` and
    // the connection distance `dist`; the diffuse contribution at wavelength λ is then
    //   (rho(λ)/PI) * SPD(λ)*invPdfλ * w      (× medium transmittance, added by caller).
    // Returns false to skip this emitter (back-facing, shadowed, or behind geometry).
    bool emitterGeom(const Scene& scene, const Hit& h, const Vec3& ngo,
                     const Emitter& em, Pcg32& rng, double& dist, double& w) const {
        if (em.collimated) return false;                  // beams aren't area-samplable
        if (em.shape == EmitterShape::Spot) {
            // Point spot: deterministic connect to the light point, weighted by the
            // cone falloff toward the surface (peak intensity/SPD = 1). No rng draw.
            Vec3 toL = em.origin - h.p;
            double dist2 = dot(toL, toL);
            dist = std::sqrt(dist2);
            Vec3 wi = toL / dist;
            double cosSurf = dot(h.n, wi);
            if (cosSurf <= 0) return false;
            double stG = shadowTerminatorG(wi, h.n, ngo);   // Chiang soft terminator (1 if flat)
            if (stG <= 0.0) return false;                    // behind true geometry: hard shadow
            double fall = spotFalloff(dot(-wi, em.beamDir), em.spotCosInner, em.spotCosOuter);
            if (fall <= 0) return false;
            if (scene.occluded(h.p + ngo * 1e-6, wi, dist - 2e-6)) return false;
            w = fall * cosSurf / dist2 * stG;                // I(w)/dist^2 (× BRDF & SPD by caller)
            return true;
        }
        double u1 = rng.uniform(), u2 = rng.uniform();
        Vec3 y, nLight, wi;
        double pdfW = 0.0;
        // Sphere: cone/solid-angle importance sampling of only the visible cap toward
        // `h.p` (low variance, no wasted back-facing draws). The estimator is in
        // solid-angle measure, so the cosLight/dist^2/area area-measure Jacobian is
        // replaced by 1/pdfW. Quads (and a receiver inside a sphere) keep the uniform
        // area-measure estimator.
        bool coneSampled = (em.shape == EmitterShape::Sphere) &&
                           em.sampleSphereCone(h.p, u1, u2, y, nLight, wi, dist, pdfW);
        // Cylinder: importance-sample only the front-facing lateral arc toward `h.p`
        // (area measure). Every draw is front-facing, so effArea = 1/pdfArea (the
        // visible area) replaces em.area and no samples land on the back side.
        double effArea = em.area, pdfAreaCyl = 0.0;
        bool cylVisible = !coneSampled && em.shape == EmitterShape::Cylinder &&
                          !em.caps &&   // capped tubes: uniform samplePoint covers the caps too
                          em.sampleCylinderVisible(h.p, u1, u2, y, nLight, pdfAreaCyl);
        if (cylVisible) effArea = 1.0 / pdfAreaCyl;
        if (coneSampled) {
            double cosSurf = dot(h.n, wi);
            if (cosSurf <= 0) return false;
            double stG = shadowTerminatorG(wi, h.n, ngo);   // Chiang soft terminator (1 if flat)
            if (stG <= 0.0) return false;                    // behind true geometry: hard shadow
            if (scene.occluded(h.p + ngo * 1e-6, wi, dist - 2e-6)) return false;
            w = cosSurf / pdfW * stG;                        // solid-angle measure
            return true;
        }
        if (!cylVisible) em.samplePoint(u1, u2, y, nLight);   // quad / interior-sphere / cylinder fallback
        Vec3 toL = y - h.p;
        double dist2 = dot(toL, toL);
        dist = std::sqrt(dist2);
        wi = toL / dist;
        double cosSurf = dot(h.n, wi);
        if (cosSurf <= 0) return false;
        double stG = shadowTerminatorG(wi, h.n, ngo);   // Chiang soft terminator (1 if flat)
        if (stG <= 0.0) return false;                    // behind true geometry: hard shadow
        double cosLight = dot(nLight, -wi);              // light is one-sided
        if (cosLight <= 0) return false;
        if (scene.occluded(h.p + ngo * 1e-6, wi, dist - 2e-6)) return false;
        double G = cosSurf * cosLight / dist2;           // geometry term
        w = G * effArea * stG;                           // pdf_area = 1/effArea (visible area for cylinder)
        return true;
    }

    double neeLight(const Scene& scene, const Hit& h, double rho, double invPdfLambda,
                    double lambda, Pcg32& rng, const SpdCache* spdCache = nullptr) const {
        double total = 0.0;
        // Geometric normal on the shading-normal side: every light connection must lie
        // in this hemisphere too, else a smoothed shading normal would leak light in
        // through the geometric back face (shading-normal problem). No-op for flat
        // tris / analytic spheres (ngo == h.n there). The shadow ray is also offset
        // along ngo so it clears the true surface rather than the shading normal.
        const Vec3 ngo = orientedGeoN(h);
        const bool med = scene.backwardMedium().enabled;
        // The cache is keyed on wavelength slot 0, so a scalar caller inside a hero
        // path (post-de-hero interactMaterial) reuses the hero table's i==0 column;
        // a fluorescent λ-switch fails matches() and falls back to a live spdFn call.
        const bool cached = spdCache && spdCache->matches(&lambda, 1);
        const int nEm = (int)scene.emitters.size();
        for (int e = 0; e < nEm; ++e) {
            const Emitter& em = scene.emitters[e];
            double dist = 0.0, w = 0.0;
            if (!emitterGeom(scene, h, ngo, em, rng, dist, w)) continue;
            double spdV = cached ? spdCache->spd[(size_t)e * (size_t)spdCache->C]
                                 : em.spdFn(lambda);
            double contrib = (rho / PI) * (spdV * invPdfLambda) * w;
            if (med)                                              // Beer-Lambert on the shadow ray
                contrib *= std::exp(-scene.backwardMedium().sigmaT(lambda) * dist);
            total += contrib;
        }
        return total;
    }

    // Hero-wavelength surface NEE: one shared visibility sample per emitter (identical
    // rng to neeLight), evaluated for all `nUp` active wavelengths. Accumulates
    // thr[i]·(rho[i]/PI)·SPD(λ_i)·invPdf[i]·w into L[i]. Only called on the fog-free
    // hero fast path, so there is no medium transmittance term.
    void neeLightHero(const Scene& scene, const Hit& h, const double* rho, double* L,
                      const double* thr, const double* lam, const double* invPdf,
                      int nUp, Pcg32& rng, const SpdCache* spdCache) const {
        const Vec3 ngo = orientedGeoN(h);
        const bool cached = spdCache && spdCache->matches(lam, nUp);
        const int nEm = (int)scene.emitters.size();
        for (int e = 0; e < nEm; ++e) {
            const Emitter& em = scene.emitters[e];
            double dist = 0.0, w = 0.0;
            if (!emitterGeom(scene, h, ngo, em, rng, dist, w)) continue;
            if (cached) {
                const double* spdE = spdCache->spd + (size_t)e * (size_t)spdCache->C;
                for (int i = 0; i < nUp; ++i)
                    L[i] += thr[i] * (rho[i] / PI) * (spdE[i] * invPdf[i]) * w;
            } else {
                for (int i = 0; i < nUp; ++i)
                    L[i] += thr[i] * (rho[i] / PI) * (em.spdFn(lam[i]) * invPdf[i]) * w;
            }
        }
    }

    // Volume next-event estimation: connect a fog scattering vertex `p` (photon
    // arriving along `wIn`) to a uniformly-sampled light point. The surface BRDF
    // and cosine are replaced by the single-scattering albedo and the Henyey-
    // Greenstein phase function; the shadow ray carries fog transmittance. This is
    // the backward mirror of the forward tracer's connectVolume().
    double neeVolume(const Scene& scene, const Vec3& p, const Vec3& wIn,
                     double lambda, double invPdfLambda, Pcg32& rng,
                     const SpdCache* spdCache = nullptr) const {
        double total = 0.0;
        const bool cached = spdCache && spdCache->matches(&lambda, 1);
        const int nEmV = (int)scene.emitters.size();
        for (int e = 0; e < nEmV; ++e) {
            const Emitter& em = scene.emitters[e];
            const double spdV = cached ? spdCache->spd[(size_t)e * (size_t)spdCache->C]
                                       : 0.0;   // live spdFn below when not cached
            if (em.collimated) continue;
            if (em.shape == EmitterShape::Spot) {
                // Point spot at a volume vertex: no surface cosine, cone falloff only.
                Vec3 toL = em.origin - p;
                double dist2 = dot(toL, toL);
                double dist = std::sqrt(dist2);
                Vec3 wi = toL / dist;
                double fall = spotFalloff(dot(-wi, em.beamDir), em.spotCosInner, em.spotCosOuter);
                if (fall <= 0) continue;
                if (scene.occluded(p + wi * 1e-6, wi, dist - 2e-6)) continue;
                double phase  = scene.backwardMedium().phaseValue(dot(wIn, wi), lambda);
                double albedo = scene.backwardMedium().albedo(lambda);
                double T = std::exp(-scene.backwardMedium().sigmaT(lambda) * dist);
                double emitW = (cached ? spdV : em.spdFn(lambda)) * invPdfLambda;
                total += albedo * phase * emitW * fall / dist2 * T;
                continue;
            }
            double u1 = rng.uniform(), u2 = rng.uniform();
            Vec3 y, nLight, wi;
            double dist = 0.0, pdfW = 0.0;
            bool coneSampled = (em.shape == EmitterShape::Sphere) &&
                               em.sampleSphereCone(p, u1, u2, y, nLight, wi, dist, pdfW);
            // Cylinder: front-facing lateral-arc sampling (area measure) toward `p`.
            double effArea = em.area, pdfAreaCyl = 0.0;
            bool cylVisible = !coneSampled && em.shape == EmitterShape::Cylinder &&
                              !em.caps &&   // capped tubes: uniform samplePoint covers the caps too
                              em.sampleCylinderVisible(p, u1, u2, y, nLight, pdfAreaCyl);
            if (cylVisible) effArea = 1.0 / pdfAreaCyl;
            double albedo = scene.backwardMedium().albedo(lambda);
            double emitW = (cached ? spdV : em.spdFn(lambda)) * invPdfLambda;
            double contrib;
            if (coneSampled) {
                if (scene.occluded(p + wi * 1e-6, wi, dist - 2e-6)) continue;
                double phase = scene.backwardMedium().phaseValue(dot(wIn, wi), lambda);
                contrib = albedo * phase * emitW / pdfW;   // solid-angle measure
            } else {
                if (!cylVisible) em.samplePoint(u1, u2, y, nLight);   // quad / interior-sphere / cylinder fallback
                Vec3 toL = y - p;
                double dist2 = dot(toL, toL);
                dist = std::sqrt(dist2);
                wi = toL / dist;
                double cosLight = dot(nLight, -wi);        // light is one-sided
                if (cosLight <= 0) continue;
                if (scene.occluded(p + wi * 1e-6, wi, dist - 2e-6)) continue;
                double phase = scene.backwardMedium().phaseValue(dot(wIn, wi), lambda);
                double G = cosLight / dist2;               // no surface cosine at a volume vertex
                contrib = albedo * phase * emitW * G * effArea;
            }
            contrib *= std::exp(-scene.backwardMedium().sigmaT(lambda) * dist);
            total += contrib;
        }
        return total;
    }

    // Environment next-event estimation at a diffuse surface vertex. Importance-
    // samples a direction from the env map's luminance CDF (a uniform sphere
    // direction for a constant env), connects with a shadow ray out past the scene
    // bounds, and MIS-weights (balance heuristic) against the BSDF-sampled
    // continuation that also reaches the env on a ray miss — so bright, concentrated
    // skies (a sun disk) are low-variance without being double-counted. Only invoked
    // when the scene actually has an env light (guarded by the caller so non-env
    // scenes keep a bit-identical RNG stream). Mirrors neeLight's transmittance /
    // shadow-bias conventions.
    // Shared geometry for environment NEE at a surface vertex: one env-direction
    // importance sample (draws rng once, identical to the old neeEnv), the shading /
    // shadow-terminator gate, the shadow ray, and the balance-heuristic MIS weight
    // against the cosine-sampled continuation. All λ-independent. Returns false to
    // skip; on success fills `wi`, `cosSurf`, `stG`, `pdfW`, `wMis`, `farDist`.
    bool envGeom(const Scene& scene, const Hit& h, Pcg32& rng, Vec3& wi,
                 double& cosSurf, double& stG, double& pdfW, double& wMis,
                 double& farDist) const {
        wi = scene.sampleEnvDir(rng, pdfW);
        if (pdfW <= 0.0) return false;
        cosSurf = dot(h.n, wi);
        const Vec3 ngo = orientedGeoN(h);
        if (cosSurf <= 0.0) return false;                       // below the shading horizon
        stG = shadowTerminatorG(wi, h.n, ngo);                  // Chiang soft terminator (1 if flat)
        if (stG <= 0.0) return false;                           // behind true geometry: hard shadow
        farDist = length(scene.sceneCenter - h.p) + scene.sceneRadius;
        if (scene.occluded(h.p + ngo * 1e-6, wi, farDist)) return false;
        double pdfBsdf = cosSurf / PI;                          // cosine-hemisphere pdf for wi
        wMis = pdfW / (pdfW + pdfBsdf);                         // balance heuristic
        return true;
    }

    double neeEnv(const Scene& scene, const Hit& h, double rho, double invPdfLambda,
                  double lambda, Pcg32& rng) const {
        Vec3 wi; double cosSurf, stG, pdfW, wMis, farDist;
        if (!envGeom(scene, h, rng, wi, cosSurf, stG, pdfW, wMis, farDist)) return 0.0;
        double Lenv = scene.envRadiance(wi, lambda);
        if (Lenv <= 0.0) return 0.0;
        double contrib = (rho / PI) * Lenv * cosSurf * invPdfLambda / pdfW * wMis * stG;
        if (scene.backwardMedium().enabled)                       // Beer-Lambert to the scene exit
            contrib *= std::exp(-scene.backwardMedium().sigmaT(lambda) * farDist);
        return contrib;
    }

    // Hero-wavelength environment NEE: one shared env-direction sample, evaluated for
    // all `nUp` active wavelengths (fog-free hero fast path, no transmittance term).
    void neeEnvHero(const Scene& scene, const Hit& h, const double* rho, double* L,
                    const double* thr, const double* lam, const double* invPdf,
                    int nUp, Pcg32& rng) const {
        Vec3 wi; double cosSurf, stG, pdfW, wMis, farDist;
        if (!envGeom(scene, h, rng, wi, cosSurf, stG, pdfW, wMis, farDist)) return;
        for (int i = 0; i < nUp; ++i) {
            double Lenv = scene.envRadiance(wi, lam[i]);
            if (Lenv <= 0.0) continue;
            L[i] += thr[i] * (rho[i] / PI) * Lenv * cosSurf * invPdf[i] / pdfW * wMis * stG;
        }
    }

    // Environment NEE at a fog scattering vertex: same as neeEnv but the surface
    // BRDF/cosine is replaced by the single-scattering albedo and the HG phase
    // function (which is also the pdf used for the MIS weight against the phase-
    // sampled continuation). Only invoked when the scene has an env light.
    double neeEnvVolume(const Scene& scene, const Vec3& p, const Vec3& wIn,
                        double lambda, double invPdfLambda, Pcg32& rng) const {
        double pdfW;
        Vec3 wi = scene.sampleEnvDir(rng, pdfW);
        if (pdfW <= 0.0) return 0.0;
        double farDist = length(scene.sceneCenter - p) + scene.sceneRadius;
        if (scene.occluded(p + wi * 1e-6, wi, farDist)) return 0.0;
        double Lenv = scene.envRadiance(wi, lambda);
        if (Lenv <= 0.0) return 0.0;
        double phase  = scene.backwardMedium().phaseValue(dot(wIn, wi), lambda);  // == BSDF pdf here
        double albedo = scene.backwardMedium().albedo(lambda);
        double wMis   = pdfW / (pdfW + phase);          // balance heuristic
        double T = std::exp(-scene.backwardMedium().sigmaT(lambda) * farDist);
        return albedo * phase * Lenv * invPdfLambda / pdfW * wMis * T;
    }

    // Handle ONE surface material interaction on a single wavelength — the whole
    // material switch, factored out of radiance() so the scalar tracer and the hero
    // tracer (which de-heros before calling this) share one copy. `m` is the resolved
    // leaf material (Mix/Layered already peeled by the caller). All path state is
    // in/out; the surface's own emission is handled by the caller BEFORE this call.
    // Returns true if the path continues (ray + state updated), false if it terminated
    // (L already holds this path's final value). A `break` in the old switch maps to
    // `return true`, a `return L` maps to `return false`.
    bool interactMaterial(const Scene& scene, const Material& m, const Hit& h, Renderer& mats,
                          Ray& ray, double& lambda, double& invPdfLambda, double& thr, double& L,
                          bool& specularArrival, double& contBsdfPdf, MediumStack& stk,
                          Pcg32& rng, const SpdCache* spdCache = nullptr) const {
        switch (m.type) {
            case MatType::Dielectric: {
                // Nested-dielectric PRIORITY resolution (Schmidt & Budge 2002). The
                // exterior IOR at this interface is the medium the ray is currently
                // travelling through (the highest-priority stack entry), not a hardcoded
                // 1.0 -- so a glass surface inside water refracts 1.33<->1.52. Where two
                // dielectrics overlap, the higher priority wins and the lower one's
                // boundary is suppressed (the ray passes straight through, unrefracted).
                // SAFE FALLBACK: the priority rule applies only when BOTH sides carry an
                // explicit priority; otherwise the interface degrades to the flat
                // air<->glass model (extIor 1.0), so priority-free scenes are unchanged.
                bool entering = dot(ray.d, h.ng) < 0.0;
                const int mi = (int)(&m - scene.mats.data());   // true index (Mix/Layered aware)
                const int pr = m.priority;               // INT_MIN if unset
                specularArrival = true;
                if (entering) {
                    const int outMat = stk.topMat();     // -1 == air
                    const int outPri = stk.topPri();     // INT_MIN == air
                    const bool ranked = m.hasPriority() &&
                        (stk.empty() || (outMat >= 0 && scene.mats[outMat].hasPriority()));
                    if (ranked && !stk.empty() && pr <= outPri) {   // suppressed entry
                        stk.push(mi, pr);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                        return true;
                    }
                    const double extIor = (ranked && outMat >= 0)
                        ? scene.mats[outMat].ior(lambda) : 1.0;
                    bool transmitted = false;
                    ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                    if (transmitted) stk.push(mi, pr);
                    return true;
                } else {
                    MediumStack after = stk;             // exiting solid mi
                    after.popMat(mi);
                    const int newMat = after.topMat();   // -1 == air underneath
                    const int newPri = after.topPri();
                    const bool ranked = m.hasPriority() &&
                        (after.empty() || (newMat >= 0 && scene.mats[newMat].hasPriority()));
                    if (ranked && newMat >= 0 && pr <= newPri) {    // suppressed exit
                        stk.popMat(mi);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                        return true;
                    }
                    const double extIor = (ranked && newMat >= 0)
                        ? scene.mats[newMat].ior(lambda) : 1.0;
                    bool transmitted = false;
                    ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                    if (transmitted) stk.popMat(mi);      // TIR stays inside mi
                    return true;
                }
            }
            case MatType::ThinFilm: {
                Ray nr;
                if (!mats.thinFilmInterface(scene, m, h, ray.d, lambda, rng, nr)) return false;
                ray = nr; specularArrival = true; return true;
            }
            case MatType::Multilayer: {
                Ray nr;
                if (!mats.multilayerInterface(m, h, ray.d, lambda, rng, nr)) return false;
                ray = nr; specularArrival = true; return true;
            }
            case MatType::Mirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() >= r) return false;      // RR absorb
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                specularArrival = true; return true;
            }
            case MatType::Grating: {
                // The grating equation is reciprocal, so backward tracing reuses the
                // same diffraction (m <-> -m symmetric). Specular per order.
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() >= r) return false;      // RR absorb
                bool absorbedG;
                Ray nr = mats.gratingDiffract(m, h, ray.d, lambda, rng, absorbedG);
                if (absorbedG) return false;
                ray = nr; specularArrival = true; return true;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() < r) ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                else                   ray = Ray{h.p + ray.d * 1e-6, ray.d};
                specularArrival = true; return true;
            }
            case MatType::Filter: {
                // Colored gel filter: pass straight through, survive with prob T(lambda).
                double t = clamp01(m.transmit(lambda));
                if (rng.uniform() >= t) return false;      // absorbed
                ray = Ray{h.p + ray.d * 1e-6, ray.d};      // direction unchanged
                specularArrival = true; return true;
            }
            case MatType::Glossy: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() >= r) return false;
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0) return false;
                ray = Ray{h.p + h.n * 1e-6, o};
                specularArrival = true; return true;
            }
            case MatType::Fluorescent: {
                // Bispectral reradiation — backward adjoint of the forward tracer's
                // fluoroInteract(). Elastic base reflects at the output wavelength; the
                // fluorescent channel excites at a separately-sampled lambdaIn (Stokes
                // shift). Both channels NEE; one stochastic continuation carries indirect.
                double rhoEl = clamp01(m.reflect(lambda));   // elastic base at lambda(out)
                L += thr * neeLight(scene, h, rhoEl, invPdfLambda, lambda, rng, spdCache);
                if (scene.envIndex >= 0)
                    L += thr * neeEnv(scene, h, rhoEl, invPdfLambda, lambda, rng);
                double Mint = m.fluoEmitSampler.integral;
                bool haveFluoro = (Mint > 0.0 && m.fluoYield > 0.0);
                double gOut = 0.0, rhoFluo = 0.0, lambdaIn = 0.0, invPdfIn = 0.0;
                if (haveFluoro) {
                    gOut = (m.fluoEmit(lambda) / Mint) * invPdfLambda;
                    double pin = 0.0;
                    lambdaIn = scene.emitSampler.sample(rng, pin);
                    if (pin > 0.0) {
                        invPdfIn = scene.invPdfLambda(lambdaIn);
                        double rhoIn, aEffIn;
                        fluoroWeights(m, lambdaIn, rhoIn, aEffIn);   // shared with forward
                        rhoFluo = aEffIn * m.fluoYield;              // reradiation albedo @lambdaIn
                        if (rhoFluo > 0.0) {                          // fluoro DIRECT NEE
                            // (lambdaIn ≠ the cached wavelengths → matches() fails and
                            // this evaluates spdFn live, exactly as before.)
                            L += thr * gOut * neeLight(scene, h, rhoFluo, invPdfIn, lambdaIn, rng, spdCache);
                            if (scene.envIndex >= 0)
                                L += thr * gOut * neeEnv(scene, h, rhoFluo, invPdfIn, lambdaIn, rng);
                        }
                    }
                }
                double wFluo = gOut * rhoFluo;               // natural indirect-fluoro weight
                double pF = (wFluo > 0.0) ? std::min(std::max(0.0, 1.0 - rhoEl), wFluo) : 0.0;
                double u = rng.uniform();
                if (u < rhoEl) {                             // elastic continuation
                    Vec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                    ray = Ray{h.p + h.n * 1e-6, wOut};
                    specularArrival = false; return true;
                } else if (u < rhoEl + pF) {                 // fluoro (wavelength-switched)
                    thr *= wFluo / pF;
                    lambda = lambdaIn;                       // Stokes shift (to the input wl)
                    invPdfLambda = invPdfIn;
                    Vec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                    ray = Ray{h.p + h.n * 1e-6, wOut};
                    specularArrival = false; return true;
                }
                return false;                                // absorbed / terminated
            }
            case MatType::DiffuseTransmit: {
                // Two-lobe Lambertian: NEE the reflect lobe in the front hemisphere and
                // the transmit lobe in the back (a normal-flipped Hit reuses neeLight/env).
                double rhoR = clamp01(diffuseReflectance(scene, m, h, lambda));
                double rhoT = clamp01(m.transmit(lambda));
                double sum = rhoR + rhoT;
                if (sum > 1.0) { rhoR /= sum; rhoT /= sum; sum = 1.0; }   // energy guard
                L += thr * neeLight(scene, h, rhoR, invPdfLambda, lambda, rng, spdCache);
                if (scene.envIndex >= 0)
                    L += thr * neeEnv(scene, h, rhoR, invPdfLambda, lambda, rng);
                Hit hb = h; hb.n = -h.n;                 // back hemisphere for the transmit lobe
                L += thr * neeLight(scene, hb, rhoT, invPdfLambda, lambda, rng, spdCache);
                if (scene.envIndex >= 0)
                    L += thr * neeEnv(scene, hb, rhoT, invPdfLambda, lambda, rng);
                double u = rng.uniform();
                if (u < rhoR) {                          // reflect continuation (front)
                    Vec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                    ray = Ray{h.p + h.n * 1e-6, wOut};
                    specularArrival = false; return true;
                } else if (u < sum) {                    // transmit continuation (back)
                    Vec3 wOut = cosineHemisphere(-h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, -h.n)) / PI;
                    ray = Ray{h.p - h.n * 1e-6, wOut};
                    specularArrival = false; return true;
                }
                return false;                            // absorbed / terminated
            }
            case MatType::Diffuse:
            default: {
                double rho = clamp01(diffuseReflectance(scene, m, h, lambda));
                L += thr * neeLight(scene, h, rho, invPdfLambda, lambda, rng, spdCache);
                if (scene.envIndex >= 0)   // env-NEE toward the sky (MIS'd on miss)
                    L += thr * neeEnv(scene, h, rho, invPdfLambda, lambda, rng);
                // Russian roulette on the albedo (throughput unchanged on survival).
                if (rng.uniform() >= rho) return false;
                Vec3 wOut = cosineHemisphere(h.n, rng);
                contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                ray = Ray{h.p + h.n * 1e-6, wOut};
                specularArrival = false; return true;
            }
        }
        return true;   // unreachable (Diffuse/default covers every type)
    }

    // Estimate spectral-weighted radiance for a single wavelength along `ray`.
    // `invPdfLambda` = emitG/g(lambda), the reciprocal of the sampled-wavelength
    // pdf; an emitter's Le/pdf weight is its SPD(lambda) * invPdfLambda.
    double radiance(const Scene& scene, Ray ray, double lambda, double invPdfLambda,
                    Pcg32& rng, const SpdCache* spdCache = nullptr) const {
        double L = 0.0, thr = 1.0;
        bool specularArrival = true;   // camera ray may see the light directly
        double contBsdfPdf = 0.0;      // solid-angle pdf of the current continuation
                                       // ray (for env-miss MIS after a diffuse/volume
                                       // bounce; unused while specularArrival)
        Renderer mats;                 // shared material sampling (stateless)
        mats.diffraction = diffraction; // grating order count follows the CLI toggle

        // Nested-dielectric medium stack: the solids the ray is currently inside. The
        // current medium (for Beer-Lambert absorption + the exterior IOR at the next
        // interface) is the highest-priority entry. Replaces the old single `interior`
        // pointer; behaves identically for a lone dielectric.
        MediumStack stk;
        auto curAbsorb = [&](double lam) -> double {           // sigma_a of the current medium
            int mi = stk.topMat();
            return (mi >= 0) ? scene.mats[mi].absorb(lam) : 0.0;
        };

        // GRADIENT-INDEX (GRIN) support. Any medium carrying an `ior` field bends
        // rays that pass through its bound. `grinAny` gates the shared marcher off so
        // `ior`-free scenes stay bit-identical. The marcher itself now lives in grin.h
        // and is shared verbatim by the forward and bidirectional tracers.
        bool grinAny = grin::sceneHasGrin(scene);

        for (int b = 0; b < maxBounce; ++b) {
            // GRIN curved marching pre-pass: advance the ray through any gradient-index
            // region it enters, integrating the Eikonal equation d/ds(n·dr/ds)=∇n in
            // small steps so the path bends. Pure marching does NOT consume a bounce;
            // when the ray reaches a surface (within one step) or leaves all GRIN regions
            // it stops and we fall through to the straight-ray body.
            if (grinAny) grin::march(scene, ray);

            Hit h = scene.closestHit(ray);
            double dSurf = h.valid ? h.t : 1e30;

            // Homogeneous fog: sample a free-flight collision that competes with
            // the surface. On a volume collision, estimate direct light via phase-
            // function NEE, then scatter (HG) or absorb — analog, throughput
            // unchanged. Mirrors the forward tracer exactly, so the two agree.
            if (scene.backwardMedium().enabled) {
                double st = scene.backwardMedium().sigmaT(lambda);
                if (st > 0.0) {
                    double tMed = -std::log(1.0 - rng.uniform()) / st;
                    if (tMed < dSurf) {
                        Vec3 p = ray.o + ray.d * tMed;
                        // Beer-Lambert attenuation over the in-glass free-flight leg.
                        {
                            double a = curAbsorb(lambda);
                            if (a > 0.0) thr *= std::exp(-a * tMed);
                        }
                        L += thr * neeVolume(scene, p, ray.d, lambda, invPdfLambda, rng, spdCache);
                        if (scene.envIndex >= 0)   // env-NEE at the volume vertex
                            L += thr * neeEnvVolume(scene, p, ray.d, lambda, invPdfLambda, rng);
                        if (rng.uniform() >= scene.backwardMedium().albedo(lambda)) return L; // absorbed
                        Vec3 wOut = scene.backwardMedium().phaseSample(ray.d, lambda, rng, contBsdfPdf);
                        ray = Ray{p, wOut};
                        specularArrival = false;   // phase-NEE covered the direct light
                        continue;
                    }
                }
            }

            // Ray escaped the scene: pick up the environment radiance from the escape
            // direction (0 if no env light; constant env ignores the direction, an
            // image env samples the lat-long map). On a camera/specular arrival there
            // is no competing env-NEE, so it is added at full weight (the directly-
            // viewed background and specular-chain sky). On a diffuse/volume arrival
            // the env is also sampled by neeEnv/neeEnvVolume at the previous vertex,
            // so this BSDF-sampled hit is MIS-weighted (balance heuristic) against
            // that NEE to avoid double-counting. Same spdFn*invPdfLambda form as
            // surface emission, so forward and backward agree on env illumination.
            // Beer-Lambert attenuation over the in-glass segment up to the surface
            // (only when the ray actually reached a surface inside a dielectric).
            if (h.valid) {
                double a = curAbsorb(lambda);
                if (a > 0.0) thr *= std::exp(-a * dSurf);
            }

            if (!h.valid) {
                if (scene.envIndex >= 0) {
                    double Lenv = scene.envRadiance(ray.d, lambda) * invPdfLambda;
                    if (specularArrival) {
                        L += thr * Lenv;
                    } else {
                        double pdfEnv = scene.envPdfDir(ray.d);
                        double wMis = (contBsdfPdf + pdfEnv > 0.0)
                                          ? contBsdfPdf / (contBsdfPdf + pdfEnv) : 0.0;
                        L += thr * Lenv * wMis;
                    }
                }
                return L;
            }
            const Material* mp = &scene.mats[h.matId];
            // Stochastic mix: resolve to a child material (or terminate on the
            // leftover absorption slice) before the switch, mirroring the forward
            // tracer so the two agree on the blended surface by construction.
            if (mp->type == MatType::Mix) {
                int child = mixResolveChild(scene, *mp, h, rng.uniform());
                if (child < 0) return L;   // absorbed
                mp = &scene.mats[child];
            }
            // Physical layered stack: reflect off the coat interface with prob R
            // (a glossy lobe about the mirror direction), else enter and pick one
            // body lobe. Mirrors the forward tracer so both split the photon budget
            // identically; the coat reflection is lossless (throughput unchanged).
            if (mp->type == MatType::Layered) {
                const Material& cm = *mp;
                double R = layeredCoatReflectance(scene, cm, h, ray.d, lambda);
                if (rng.uniform() < R) {
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, cm, h), rng);
                    if (dot(o, h.n) <= 0) return L;
                    ray = Ray{h.p + h.n * 1e-6, o};
                    specularArrival = true;
                    continue;
                }
                int child = mixPickChild(cm, rng.uniform());   // body lobe
                if (child < 0) return L;                        // leftover absorbs
                mp = &scene.mats[child];
            }
            const Material& m = *mp;

            // Emission (add only on specular/camera arrival; NEE covers diffuse).
            // The surface's own emitted radiance Le=m.emit(lambda), weighted by the
            // reciprocal wavelength pdf (= its SPD integral for a single light).
            if (m.isLight && specularArrival && dot(ray.d, h.ng) < 0.0)
                L += thr * m.emit(lambda) * invPdfLambda;

            if (!interactMaterial(scene, m, h, mats, ray, lambda, invPdfLambda, thr, L,
                                  specularArrival, contBsdfPdf, stk, rng, spdCache))
                return L;                                 // path terminated in the interaction
        }
        return L;
    }

    // Hero-wavelength variant of radiance(): carries C wavelengths (hero + C-1
    // stratified secondaries) down ONE camera path. Index 0 is the hero; it drives
    // every sampling decision with the same rng stream a single-wavelength path would,
    // while the secondaries ride along and are reweighted per-λ. At a dispersive /
    // wavelength-switching material (anything but Diffuse/DiffuseTransmit) the
    // secondaries de-hero (terminate) and the hero is boosted ×C so it alone carries an
    // unbiased single-λ estimate onward (PBRT-v4's TerminateSecondary convention). The
    // caller restricts this to scenes WITHOUT participating media / GRIN / a physical
    // lens, so those branches are absent here. Fills Lout[0..C).
    void radianceHero(const Scene& scene, Ray ray, const double* lamIn,
                      const double* invPdfIn, int C, double* Lout, Pcg32& rng,
                      const SpdCache* spdCache = nullptr) const {
        double lam[hero::kHeroMax], invPdf[hero::kHeroMax], thr[hero::kHeroMax], L[hero::kHeroMax];
        for (int i = 0; i < C; ++i) { lam[i] = lamIn[i]; invPdf[i] = invPdfIn[i]; thr[i] = 1.0; L[i] = 0.0; }
        bool secAlive = (C > 1);
        bool specularArrival = true;
        double contBsdfPdf = 0.0;
        Renderer mats; mats.diffraction = diffraction;
        MediumStack stk;                 // dielectric priority (Beer-Lambert uses hero λ)

        auto finish = [&]() { for (int i = 0; i < C; ++i) Lout[i] = L[i]; };
        auto deHero = [&]() {            // terminate secondaries, boost hero ×C
            if (!secAlive) return;
            thr[0] *= (double)C;
            secAlive = false;
        };

        for (int b = 0; b < maxBounce; ++b) {
            int nUp = secAlive ? C : 1;   // wavelengths still being propagated
            Hit h = scene.closestHit(ray);
            double dSurf = h.valid ? h.t : 1e30;

            // Beer-Lambert over the in-glass segment. A non-empty stack implies we've
            // already de-hero'd (dielectric entry de-heros), so nUp == 1 whenever
            // absorption is non-zero; the loop still handles the general case.
            if (h.valid) {
                int mi = stk.topMat();
                if (mi >= 0) {
                    for (int i = 0; i < nUp; ++i) {
                        double a = scene.mats[mi].absorb(lam[i]);
                        if (a > 0.0) thr[i] *= std::exp(-a * dSurf);
                    }
                }
            }

            if (!h.valid) {              // env-miss (full weight on specular arrival, else MIS)
                if (scene.envIndex >= 0) {
                    if (specularArrival) {
                        for (int i = 0; i < nUp; ++i)
                            L[i] += thr[i] * scene.envRadiance(ray.d, lam[i]) * invPdf[i];
                    } else {
                        double pdfEnv = scene.envPdfDir(ray.d);
                        double wMis = (contBsdfPdf + pdfEnv > 0.0)
                                          ? contBsdfPdf / (contBsdfPdf + pdfEnv) : 0.0;
                        for (int i = 0; i < nUp; ++i)
                            L[i] += thr[i] * scene.envRadiance(ray.d, lam[i]) * invPdf[i] * wMis;
                    }
                }
                finish(); return;
            }

            const Material* mp = &scene.mats[h.matId];
            if (mp->type == MatType::Mix) {
                int child = mixResolveChild(scene, *mp, h, rng.uniform());
                if (child < 0) { finish(); return; }
                mp = &scene.mats[child];
            }
            if (mp->type == MatType::Layered) {
                // Wavelength-dependent Fresnel coat: de-hero and run the scalar layered
                // handling on the hero channel.
                deHero(); nUp = 1;
                const Material& cm = *mp;
                double R = layeredCoatReflectance(scene, cm, h, ray.d, lam[0]);
                if (rng.uniform() < R) {
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, cm, h), rng);
                    if (dot(o, h.n) <= 0) { finish(); return; }
                    ray = Ray{h.p + h.n * 1e-6, o};
                    specularArrival = true;
                    continue;
                }
                int child = mixPickChild(cm, rng.uniform());
                if (child < 0) { finish(); return; }
                mp = &scene.mats[child];
            }
            const Material& m = *mp;

            // Surface emission on a specular/camera arrival (NEE covers diffuse).
            if (m.isLight && specularArrival && dot(ray.d, h.ng) < 0.0)
                for (int i = 0; i < nUp; ++i)
                    L[i] += thr[i] * m.emit(lam[i]) * invPdf[i];

            switch (m.type) {
                case MatType::DiffuseTransmit: {
                    double rhoR[hero::kHeroMax], rhoT[hero::kHeroMax];
                    for (int i = 0; i < nUp; ++i) {
                        double rr = clamp01(diffuseReflectance(scene, m, h, lam[i]));
                        double rt = clamp01(m.transmit(lam[i]));
                        double s = rr + rt;
                        if (s > 1.0) { rr /= s; rt /= s; }       // per-λ energy guard
                        rhoR[i] = rr; rhoT[i] = rt;
                    }
                    neeLightHero(scene, h, rhoR, L, thr, lam, invPdf, nUp, rng, spdCache);
                    if (scene.envIndex >= 0)
                        neeEnvHero(scene, h, rhoR, L, thr, lam, invPdf, nUp, rng);
                    Hit hb = h; hb.n = -h.n;                     // back hemisphere (transmit lobe)
                    neeLightHero(scene, hb, rhoT, L, thr, lam, invPdf, nUp, rng, spdCache);
                    if (scene.envIndex >= 0)
                        neeEnvHero(scene, hb, rhoT, L, thr, lam, invPdf, nUp, rng);
                    double sumHero = rhoR[0] + rhoT[0];
                    double u = rng.uniform();
                    if (u < rhoR[0]) {                           // reflect (front)
                        for (int i = 1; i < nUp; ++i) thr[i] *= rhoR[i] / rhoR[0];
                        Vec3 wOut = cosineHemisphere(h.n, rng);
                        contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                        ray = Ray{h.p + h.n * 1e-6, wOut};
                        specularArrival = false; break;
                    } else if (u < sumHero) {                    // transmit (back)
                        for (int i = 1; i < nUp; ++i) thr[i] *= rhoT[i] / rhoT[0];
                        Vec3 wOut = cosineHemisphere(-h.n, rng);
                        contBsdfPdf = std::max(0.0, dot(wOut, -h.n)) / PI;
                        ray = Ray{h.p - h.n * 1e-6, wOut};
                        specularArrival = false; break;
                    }
                    finish(); return;                            // absorbed
                }
                case MatType::Dielectric:
                case MatType::ThinFilm:
                case MatType::Multilayer:
                case MatType::Mirror:
                case MatType::Grating:
                case MatType::HalfMirror:
                case MatType::Filter:
                case MatType::Glossy:
                case MatType::Fluorescent: {
                    // Dispersive / wavelength-switching: terminate secondaries, then run
                    // the shared scalar interaction on the (boosted) hero channel.
                    deHero();
                    if (!interactMaterial(scene, m, h, mats, ray, lam[0], invPdf[0], thr[0], L[0],
                                          specularArrival, contBsdfPdf, stk, rng, spdCache)) { finish(); return; }
                    break;
                }
                case MatType::Diffuse:
                default: {
                    double rho[hero::kHeroMax];
                    for (int i = 0; i < nUp; ++i)
                        rho[i] = clamp01(diffuseReflectance(scene, m, h, lam[i]));
                    neeLightHero(scene, h, rho, L, thr, lam, invPdf, nUp, rng, spdCache);
                    if (scene.envIndex >= 0)
                        neeEnvHero(scene, h, rho, L, thr, lam, invPdf, nUp, rng);
                    double rhoHero = rho[0];
                    if (rng.uniform() >= rhoHero) { finish(); return; }   // hero RR absorb
                    for (int i = 1; i < nUp; ++i) thr[i] *= rho[i] / rhoHero;  // secondary reweight
                    Vec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                    ray = Ray{h.p + h.n * 1e-6, wOut};
                    specularArrival = false; break;
                }
            }
        }
        finish();
    }

    // Render `spp` samples per pixel into `film` (accumulates cieXYZ * radiance,
    // exactly like the forward film, so writeFilm with N=spp displays it). Renders
    // the pixel rows [y0, y1) — the caller partitions rows across threads. On a
    // pinhole camera in a scene without fog / GRIN, hero-wavelength sampling is used
    // (C wavelengths per camera path); otherwise the single-wavelength radiance().
    //
    // `sampleBase` is the ABSOLUTE index of the first sample this call renders (a
    // chunked/progressive render passes its running spp count; a resume passes the
    // checkpointed count). Each (pixel, absolute sample) pair seeds its own RNG
    // stream via seedUnit(), so the rendered realization is independent of the
    // chunk split, the row banding, and the thread count.
    void renderRows(const Scene& scene, const Camera& cam, Film& film,
                    int y0, int y1, long long spp, unsigned long long sampleBase) const {
        const int C = heroC;
        const bool useHero = (C > 1) && !scene.backwardMedium().enabled &&
                             !grin::sceneHasGrin(scene) && !cam.hasLens();
        const uint64_t nPix = (uint64_t)film.resX * (uint64_t)film.resY;
        // Per-sample emitter-SPD table (see SpdCache): nEm×C (nEm×1 on the scalar
        // path), allocated once per renderRows call (i.e. per thread) and refilled
        // for every sample.
        const int nEm = (int)scene.emitters.size();
        std::vector<double> spdBuf((size_t)nEm * (size_t)(useHero ? C : 1));
        for (int py = y0; py < y1; ++py) {
            for (int px = 0; px < film.resX; ++px) {
                const uint64_t pixIdx = (uint64_t)py * (uint64_t)film.resX + (uint64_t)px;
                for (long long s = 0; s < spp; ++s) {
                    Pcg32 rng;
                    seedUnit(rng, (sampleBase + (uint64_t)s) * nPix + pixIdx,
                             0xD1B54A32D192ED03ULL);
                    if (useHero) {
                        // One stratified base draw → hero + C-1 secondary wavelengths,
                        // all from the emission CDF. The hero (index 0) must have a
                        // valid pdf; dead secondaries (pdf 0) carry invPdf 0 and splat 0.
                        double u = rng.uniform();
                        double lamA[hero::kHeroMax], invA[hero::kHeroMax];
                        double pdfA[hero::kHeroMax];
                        pdfA[0] = 0.0;
                        lamA[0] = scene.emitSampler.sampleAt(u, pdfA[0]);
                        if (pdfA[0] <= 0) continue;
                        for (int i = 1; i < C; ++i) {
                            double uu = u + (double)i / C;
                            if (uu >= 1.0) uu -= 1.0;            // wrap into [0,1)
                            pdfA[i] = 0.0;
                            lamA[i] = scene.emitSampler.sampleAt(uu, pdfA[i]);
                        }
                        // Fill the per-sample SPD table, then derive invA from it by
                        // replicating Scene::invPdfLambda on the cached values (same
                        // emitter order, same zero guard — bit-identical). The NEE
                        // connections down the path then reuse the table instead of
                        // re-dispatching spdFn per emitter per bounce.
                        for (int e = 0; e < nEm; ++e) {
                            const Emitter& em = scene.emitters[e];
                            for (int i = 0; i < C; ++i)
                                spdBuf[(size_t)e * C + i] = em.spdFn(lamA[i]);
                        }
                        for (int i = 0; i < C; ++i) {
                            if (i > 0 && pdfA[i] <= 0.0) { invA[i] = 0.0; continue; }
                            double g = 0.0;
                            for (int e = 0; e < nEm; ++e)
                                g += scene.emitters[e].geomWeight() * spdBuf[(size_t)e * C + i];
                            invA[i] = (g > 0.0) ? scene.emitG / g : 0.0;
                        }
                        SpdCache spdCache{lamA, spdBuf.data(), C};
                        Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
                        double Lh[hero::kHeroMax];
                        radianceHero(scene, ray, lamA, invA, C, Lh, rng, &spdCache);
                        for (int i = 0; i < C; ++i)
                            film.add(px, py, Vec3(cieX(lamA[i]), cieY(lamA[i]), cieZ(lamA[i]))
                                             * (Lh[i] / C));
                        continue;
                    }
                    // Sample lambda from the combined emission distribution g(lambda).
                    double pdf = 0.0;
                    double lambda = scene.emitSampler.sample(rng, pdf);
                    if (pdf <= 0) continue;
                    // Fill the per-sample SPD table (C=1) and derive invPdfLambda from
                    // it, replicating Scene::invPdfLambda on the cached values (same
                    // emitter order, same zero guard — bit-identical, = emitG/g(λ)).
                    for (int e = 0; e < nEm; ++e)
                        spdBuf[e] = scene.emitters[e].spdFn(lambda);
                    double gSum = 0.0;
                    for (int e = 0; e < nEm; ++e)
                        gSum += scene.emitters[e].geomWeight() * spdBuf[e];
                    double invPdfLambda = (gSum > 0.0) ? scene.emitG / gSum : 0.0;
                    SpdCache spdCache{&lambda, spdBuf.data(), 1};
                    if (cam.hasLens()) {
                        // Physical multi-element lens: trace the camera ray from the
                        // film out through the real glass interfaces at this wavelength
                        // (chromatic aberration + DoF + vignetting emerge). A vignetted
                        // ray contributes nothing; survivors carry a radiometric weight.
                        double jx = rng.uniform(), jy = rng.uniform();
                        double u1 = rng.uniform(), u2 = rng.uniform();
                        Ray ray; double wLens = 0.0;
                        if (!cam.genLensRay(px, py, jx, jy, u1, u2, lambda, ray, wLens))
                            continue;                       // clipped by an element / the stop
                        double L = radiance(scene, ray, lambda, invPdfLambda, rng, &spdCache);
                        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * (L * wLens));
                        continue;
                    }
                    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
                    double L = radiance(scene, ray, lambda, invPdfLambda, rng, &spdCache);
                    film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * L);
                }
            }
        }
    }
};
