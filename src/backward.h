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
// NOTE: Fluorescence is intentionally NOT supported here — backward tracing a
// wavelength-shifting material needs the full bispectral reradiation matrix,
// whereas forward single-wavelength tracing handles it trivially. A Fluorescent
// material falls through to the Diffuse case below, so fluoro scenes must not be
// used with modes R/V (the forward-tracer's -scene fluoro is model A/B/C only).
// Participating media (scene.medium / -fog) IS supported here: camera and
// scattered rays sample volume free-flight, and volume vertices do phase-function
// NEE to the light (neeVolume). So -fog CAN be combined with modes R/V, which is
// how the forward fog transport is cross-validated.
// Emission is added only when a light is reached via the camera ray or a
// specular/near-specular bounce; diffuse arrivals are covered by NEE (no double
// counting).
#pragma once
#include "scene.h"
#include "camera.h"
#include "render.h"   // sampleGlossy, Renderer::refractOrReflect, clamp01, PI

struct BackwardRenderer {
    int maxBounce = 32;
    bool diffraction = true;   // mirrors Renderer::diffraction for MatType::Grating

    // Next-event estimation: connect a surface vertex to each area emitter (the
    // integral splits by light, summed unbiasedly). `invPdfLambda` = emitG/g(lambda)
    // is the reciprocal of the sampled-wavelength pdf; multiplied by an emitter's
    // SPD(lambda) it yields that emitter's Le/pdf weight (= its SPD integral for a
    // single light, matching the forward tracer's photon-weight convention).
    double neeLight(const Scene& scene, const Hit& h, double rho, double invPdfLambda,
                    double lambda, Pcg32& rng) const {
        double total = 0.0;
        for (const auto& em : scene.emitters) {
            if (em.collimated) continue;                  // beams aren't area-samplable
            if (em.shape == EmitterShape::Spot) {
                // Point spot: deterministic connect to the light point, weighted by
                // the cone falloff toward the surface (peak intensity/SPD = 1).
                Vec3 toL = em.origin - h.p;
                double dist2 = dot(toL, toL);
                double dist = std::sqrt(dist2);
                Vec3 wi = toL / dist;
                double cosSurf = dot(h.n, wi);
                if (cosSurf <= 0) continue;
                double fall = spotFalloff(dot(-wi, em.beamDir), em.spotCosInner, em.spotCosOuter);
                if (fall <= 0) continue;
                if (scene.occluded(h.p + h.n * 1e-6, wi, dist - 2e-6)) continue;
                double f = rho / PI;
                double emitW = em.spdFn(lambda) * invPdfLambda;
                double contrib = f * emitW * fall * cosSurf / dist2;  // I(w)/dist^2
                if (scene.medium.enabled)
                    contrib *= std::exp(-scene.medium.sigmaT(lambda) * dist);
                total += contrib;
                continue;
            }
            double u1 = rng.uniform(), u2 = rng.uniform();
            Vec3 y, nLight;
            em.samplePoint(u1, u2, y, nLight);            // quad or sphere surface point
            Vec3 toL = y - h.p;
            double dist2 = dot(toL, toL);
            double dist = std::sqrt(dist2);
            Vec3 wi = toL / dist;
            double cosSurf = dot(h.n, wi);
            if (cosSurf <= 0) continue;
            double cosLight = dot(nLight, -wi);           // light is one-sided
            if (cosLight <= 0) continue;
            if (scene.occluded(h.p + h.n * 1e-6, wi, dist - 2e-6)) continue;
            double f = rho / PI;                          // Lambertian BRDF
            double G = cosSurf * cosLight / dist2;        // geometry term
            double emitW = em.spdFn(lambda) * invPdfLambda;   // Le(lambda)/pdf_lambda
            double contrib = f * emitW * G * em.area;     // pdf_area = 1/area
            if (scene.medium.enabled)                     // Beer-Lambert on the shadow ray
                contrib *= std::exp(-scene.medium.sigmaT(lambda) * dist);
            total += contrib;
        }
        return total;
    }

    // Volume next-event estimation: connect a fog scattering vertex `p` (photon
    // arriving along `wIn`) to a uniformly-sampled light point. The surface BRDF
    // and cosine are replaced by the single-scattering albedo and the Henyey-
    // Greenstein phase function; the shadow ray carries fog transmittance. This is
    // the backward mirror of the forward tracer's connectVolume().
    double neeVolume(const Scene& scene, const Vec3& p, const Vec3& wIn,
                     double lambda, double invPdfLambda, Pcg32& rng) const {
        double total = 0.0;
        for (const auto& em : scene.emitters) {
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
                double phase  = hgPhase(dot(wIn, wi), scene.medium.g);
                double albedo = scene.medium.albedo(lambda);
                double T = std::exp(-scene.medium.sigmaT(lambda) * dist);
                double emitW = em.spdFn(lambda) * invPdfLambda;
                total += albedo * phase * emitW * fall / dist2 * T;
                continue;
            }
            double u1 = rng.uniform(), u2 = rng.uniform();
            Vec3 y, nLight;
            em.samplePoint(u1, u2, y, nLight);            // quad or sphere surface point
            Vec3 toL = y - p;
            double dist2 = dot(toL, toL);
            double dist = std::sqrt(dist2);
            Vec3 wi = toL / dist;
            double cosLight = dot(nLight, -wi);           // light is one-sided
            if (cosLight <= 0) continue;
            if (scene.occluded(p + wi * 1e-6, wi, dist - 2e-6)) continue;
            double phase  = hgPhase(dot(wIn, wi), scene.medium.g);
            double albedo = scene.medium.albedo(lambda);
            double G = cosLight / dist2;                   // no surface cosine at a volume vertex
            double T = std::exp(-scene.medium.sigmaT(lambda) * dist);
            double emitW = em.spdFn(lambda) * invPdfLambda;
            total += albedo * phase * emitW * G * em.area * T;
        }
        return total;
    }

    // Estimate spectral-weighted radiance for a single wavelength along `ray`.
    // `invPdfLambda` = emitG/g(lambda), the reciprocal of the sampled-wavelength
    // pdf; an emitter's Le/pdf weight is its SPD(lambda) * invPdfLambda.
    double radiance(const Scene& scene, Ray ray, double lambda, double invPdfLambda,
                    Pcg32& rng) const {
        double L = 0.0, thr = 1.0;
        bool specularArrival = true;   // camera ray may see the light directly
        Renderer mats;                 // shared material sampling (stateless)
        mats.diffraction = diffraction; // grating order count follows the CLI toggle

        for (int b = 0; b < maxBounce; ++b) {
            Hit h = scene.closestHit(ray);
            double dSurf = h.valid ? h.t : 1e30;

            // Homogeneous fog: sample a free-flight collision that competes with
            // the surface. On a volume collision, estimate direct light via phase-
            // function NEE, then scatter (HG) or absorb — analog, throughput
            // unchanged. Mirrors the forward tracer exactly, so the two agree.
            if (scene.medium.enabled) {
                double st = scene.medium.sigmaT(lambda);
                if (st > 0.0) {
                    double tMed = -std::log(1.0 - rng.uniform()) / st;
                    if (tMed < dSurf) {
                        Vec3 p = ray.o + ray.d * tMed;
                        L += thr * neeVolume(scene, p, ray.d, lambda, invPdfLambda, rng);
                        if (rng.uniform() >= scene.medium.albedo(lambda)) return L; // absorbed
                        ray = Ray{p, sampleHG(ray.d, scene.medium.g, rng)};
                        specularArrival = false;   // phase-NEE covered the direct light
                        continue;
                    }
                }
            }

            // Ray escaped the scene: pick up the environment radiance from the escape
            // direction (0 if no env light; constant env ignores the direction, an
            // image env samples the lat-long map). Added unconditionally (no env NEE
            // yet), the same spdFn*invPdfLambda form as surface emission, so forward
            // and backward agree on env illumination and directly-viewed background.
            if (!h.valid) {
                L += thr * scene.envRadiance(ray.d, lambda) * invPdfLambda;
                return L;
            }
            const Material* mp = &scene.mats[h.matId];
            // Stochastic mix: resolve to a child material (or terminate on the
            // leftover absorption slice) before the switch, mirroring the forward
            // tracer so the two agree on the blended surface by construction.
            if (mp->type == MatType::Mix) {
                int child = mixPickChild(*mp, rng.uniform());
                if (child < 0) return L;   // absorbed
                mp = &scene.mats[child];
            }
            const Material& m = *mp;

            // Emission (add only on specular/camera arrival; NEE covers diffuse).
            // The surface's own emitted radiance Le=m.emit(lambda), weighted by the
            // reciprocal wavelength pdf (= its SPD integral for a single light).
            if (m.isLight && specularArrival && dot(ray.d, h.ng) < 0.0)
                L += thr * m.emit(lambda) * invPdfLambda;

            switch (m.type) {
                case MatType::Dielectric: {
                    ray = mats.refractOrReflect(m, h, ray.d, lambda, rng);
                    specularArrival = true;
                    break;
                }
                case MatType::ThinFilm: {
                    // Iridescent coated dielectric: specular reflect-or-refract,
                    // same delta-BSDF handling as Dielectric (reflectance carries
                    // the thin-film interference colour).
                    ray = mats.thinFilmInterface(m, h, ray.d, lambda, rng);
                    specularArrival = true;
                    break;
                }
                case MatType::Mirror: {
                    double r = clamp01(m.reflect(lambda));
                    if (rng.uniform() >= r) return L;      // RR absorb
                    ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                    specularArrival = true;
                    break;
                }
                case MatType::Grating: {
                    // The grating equation is reciprocal, so backward tracing reuses
                    // the same diffraction (m <-> -m symmetric). Specular per order.
                    double r = clamp01(m.reflect(lambda));
                    if (rng.uniform() >= r) return L;      // RR absorb
                    bool absorbedG;
                    Ray nr = mats.gratingDiffract(m, h, ray.d, lambda, rng, absorbedG);
                    if (absorbedG) return L;
                    ray = nr;
                    specularArrival = true;
                    break;
                }
                case MatType::HalfMirror: {
                    double r = clamp01(m.reflect(lambda));
                    if (rng.uniform() < r) ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                    else                   ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    specularArrival = true;
                    break;
                }
                case MatType::Glossy: {
                    double r = clamp01(m.reflect(lambda));
                    if (rng.uniform() >= r) return L;
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), m.roughness, rng);
                    if (dot(o, h.n) <= 0) return L;
                    ray = Ray{h.p + h.n * 1e-6, o};
                    specularArrival = true;
                    break;
                }
                case MatType::Diffuse:
                default: {
                    double rho = clamp01(diffuseReflectance(scene, m, h, lambda));
                    L += thr * neeLight(scene, h, rho, invPdfLambda, lambda, rng);
                    // Russian roulette on the albedo (throughput unchanged on
                    // survival) — matches the forward tracer's diffuse handling.
                    if (rng.uniform() >= rho) return L;
                    ray = Ray{h.p + h.n * 1e-6, cosineHemisphere(h.n, rng)};
                    specularArrival = false;
                    break;
                }
            }
        }
        return L;
    }

    // Render `spp` samples per pixel into `film` (accumulates cieXYZ * radiance,
    // exactly like the forward film, so writePPM with N=spp displays it). Renders
    // the pixel rows [y0, y1) — the caller partitions rows across threads.
    void renderRows(const Scene& scene, const Camera& cam, Film& film,
                    int y0, int y1, long long spp, Pcg32& rng) const {
        for (int py = y0; py < y1; ++py) {
            for (int px = 0; px < film.resX; ++px) {
                for (long long s = 0; s < spp; ++s) {
                    // Sample lambda from the combined emission distribution g(lambda).
                    double pdf = 0.0;
                    double lambda = scene.emitSampler.sample(rng, pdf);
                    if (pdf <= 0) continue;
                    double invPdfLambda = scene.invPdfLambda(lambda); // exact emitG/g(lambda)
                    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
                    double L = radiance(scene, ray, lambda, invPdfLambda, rng);
                    film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * L);
                }
            }
        }
    }
};
