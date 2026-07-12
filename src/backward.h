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
                if (scene.backwardMedium().enabled)
                    contrib *= std::exp(-scene.backwardMedium().sigmaT(lambda) * dist);
                total += contrib;
                continue;
            }
            double u1 = rng.uniform(), u2 = rng.uniform();
            Vec3 y, nLight, wi;
            double dist = 0.0, pdfW = 0.0;
            // Sphere: cone/solid-angle importance sampling of only the visible cap
            // toward `h.p` (low variance, no wasted back-facing draws). The estimator
            // is in solid-angle measure, so the cosLight/dist^2/area area-measure
            // Jacobian is replaced by 1/pdfW. Quads (and a receiver inside a sphere)
            // keep the uniform area-measure estimator.
            bool coneSampled = (em.shape == EmitterShape::Sphere) &&
                               em.sampleSphereCone(h.p, u1, u2, y, nLight, wi, dist, pdfW);
            // Cylinder: importance-sample only the front-facing lateral arc toward
            // `h.p` (area measure). Every draw is front-facing, so effArea = 1/pdfArea
            // (the visible area) replaces em.area and no samples land on the back side.
            double effArea = em.area, pdfAreaCyl = 0.0;
            bool cylVisible = !coneSampled && em.shape == EmitterShape::Cylinder &&
                              !em.caps &&   // capped tubes: uniform samplePoint covers the caps too
                              em.sampleCylinderVisible(h.p, u1, u2, y, nLight, pdfAreaCyl);
            if (cylVisible) effArea = 1.0 / pdfAreaCyl;
            double cosSurf, contrib;
            double f = rho / PI;                          // Lambertian BRDF
            double emitW = em.spdFn(lambda) * invPdfLambda;   // Le(lambda)/pdf_lambda
            if (coneSampled) {
                cosSurf = dot(h.n, wi);
                if (cosSurf <= 0) continue;
                if (scene.occluded(h.p + h.n * 1e-6, wi, dist - 2e-6)) continue;
                contrib = f * emitW * cosSurf / pdfW;     // solid-angle measure
            } else {
                if (!cylVisible) em.samplePoint(u1, u2, y, nLight);   // quad / interior-sphere / cylinder fallback
                Vec3 toL = y - h.p;
                double dist2 = dot(toL, toL);
                dist = std::sqrt(dist2);
                wi = toL / dist;
                cosSurf = dot(h.n, wi);
                if (cosSurf <= 0) continue;
                double cosLight = dot(nLight, -wi);       // light is one-sided
                if (cosLight <= 0) continue;
                if (scene.occluded(h.p + h.n * 1e-6, wi, dist - 2e-6)) continue;
                double G = cosSurf * cosLight / dist2;    // geometry term
                contrib = f * emitW * G * effArea;        // pdf_area = 1/effArea (visible area for cylinder)
            }
            if (scene.backwardMedium().enabled)                     // Beer-Lambert on the shadow ray
                contrib *= std::exp(-scene.backwardMedium().sigmaT(lambda) * dist);
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
                double phase  = hgPhase(dot(wIn, wi), scene.backwardMedium().g);
                double albedo = scene.backwardMedium().albedo(lambda);
                double T = std::exp(-scene.backwardMedium().sigmaT(lambda) * dist);
                double emitW = em.spdFn(lambda) * invPdfLambda;
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
            double emitW = em.spdFn(lambda) * invPdfLambda;
            double contrib;
            if (coneSampled) {
                if (scene.occluded(p + wi * 1e-6, wi, dist - 2e-6)) continue;
                double phase = hgPhase(dot(wIn, wi), scene.backwardMedium().g);
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
                double phase = hgPhase(dot(wIn, wi), scene.backwardMedium().g);
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
    double neeEnv(const Scene& scene, const Hit& h, double rho, double invPdfLambda,
                  double lambda, Pcg32& rng) const {
        double pdfW;
        Vec3 wi = scene.sampleEnvDir(rng, pdfW);
        if (pdfW <= 0.0) return 0.0;
        double cosSurf = dot(h.n, wi);
        if (cosSurf <= 0.0) return 0.0;                 // below the horizon
        double farDist = length(scene.sceneCenter - h.p) + scene.sceneRadius;
        if (scene.occluded(h.p + h.n * 1e-6, wi, farDist)) return 0.0;
        double Lenv = scene.envRadiance(wi, lambda);
        if (Lenv <= 0.0) return 0.0;
        double f = rho / PI;                            // Lambertian BRDF
        double pdfBsdf = cosSurf / PI;                  // cosine-hemisphere pdf for wi
        double wMis = pdfW / (pdfW + pdfBsdf);          // balance heuristic
        double contrib = f * Lenv * cosSurf * invPdfLambda / pdfW * wMis;
        if (scene.backwardMedium().enabled)                       // Beer-Lambert to the scene exit
            contrib *= std::exp(-scene.backwardMedium().sigmaT(lambda) * farDist);
        return contrib;
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
        double phase  = hgPhase(dot(wIn, wi), scene.backwardMedium().g);  // == BSDF pdf here
        double albedo = scene.backwardMedium().albedo(lambda);
        double wMis   = pdfW / (pdfW + phase);          // balance heuristic
        double T = std::exp(-scene.backwardMedium().sigmaT(lambda) * farDist);
        return albedo * phase * Lenv * invPdfLambda / pdfW * wMis * T;
    }

    // Estimate spectral-weighted radiance for a single wavelength along `ray`.
    // `invPdfLambda` = emitG/g(lambda), the reciprocal of the sampled-wavelength
    // pdf; an emitter's Le/pdf weight is its SPD(lambda) * invPdfLambda.
    double radiance(const Scene& scene, Ray ray, double lambda, double invPdfLambda,
                    Pcg32& rng) const {
        double L = 0.0, thr = 1.0;
        bool specularArrival = true;   // camera ray may see the light directly
        double contBsdfPdf = 0.0;      // solid-angle pdf of the current continuation
                                       // ray (for env-miss MIS after a diffuse/volume
                                       // bounce; unused while specularArrival)
        Renderer mats;                 // shared material sampling (stateless)
        mats.diffraction = diffraction; // grating order count follows the CLI toggle

        const Material* interior = nullptr;   // dielectric the ray is inside (colored glass)
        for (int b = 0; b < maxBounce; ++b) {
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
                        if (interior) {
                            double a = interior->absorb(lambda);
                            if (a > 0.0) thr *= std::exp(-a * tMed);
                        }
                        L += thr * neeVolume(scene, p, ray.d, lambda, invPdfLambda, rng);
                        if (scene.envIndex >= 0)   // env-NEE at the volume vertex
                            L += thr * neeEnvVolume(scene, p, ray.d, lambda, invPdfLambda, rng);
                        if (rng.uniform() >= scene.backwardMedium().albedo(lambda)) return L; // absorbed
                        Vec3 wOut = sampleHG(ray.d, scene.backwardMedium().g, rng);
                        contBsdfPdf = hgPhase(dot(ray.d, wOut), scene.backwardMedium().g);
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
            if (interior && h.valid) {
                double a = interior->absorb(lambda);
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

            switch (m.type) {
                case MatType::Dielectric: {
                    bool entering = dot(ray.d, h.ng) < 0.0;
                    bool transmitted = false;
                    ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted);
                    if (transmitted) interior = entering ? &m : nullptr;
                    specularArrival = true;
                    break;
                }
                case MatType::ThinFilm: {
                    // Iridescent coated interface: specular reflect-or-refract, same
                    // delta-BSDF handling as Dielectric (reflectance carries the
                    // thin-film interference colour). An absorbing substrate absorbs
                    // the transmitted fraction -> the path terminates here.
                    Ray nr;
                    if (!mats.thinFilmInterface(scene, m, h, ray.d, lambda, rng, nr)) return L;
                    ray = nr;
                    specularArrival = true;
                    break;
                }
                case MatType::Multilayer: {
                    // Multilayer stack: specular reflect-or-refract via the Abeles
                    // full-stack reflectance, same delta-BSDF handling as Dielectric.
                    // An absorbing stack/substrate terminates the path.
                    Ray nr;
                    if (!mats.multilayerInterface(m, h, ray.d, lambda, rng, nr)) return L;
                    ray = nr;
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
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                    if (dot(o, h.n) <= 0) return L;
                    ray = Ray{h.p + h.n * 1e-6, o};
                    specularArrival = true;
                    break;
                }
                case MatType::Fluorescent: {
                    // Bispectral reradiation — the backward adjoint of the forward
                    // tracer's fluoroInteract(). The surface reflects ELASTICALLY at
                    // the current output wavelength (albedo rho(lambda)) AND re-radiates
                    // at lambda from excitation absorbed at a DIFFERENT input wavelength
                    // lambdaIn (Stokes shift). Both channels do NEE; a single stochastic
                    // continuation carries the indirect term (elastic at lambda, or
                    // wavelength-switched to lambdaIn for indirect excitation), so the
                    // estimator is unbiased and validates forward-vs-backward (mode V).
                    double rhoEl = clamp01(m.reflect(lambda));   // elastic base at lambda(out)
                    // Elastic diffuse NEE at the output wavelength.
                    L += thr * neeLight(scene, h, rhoEl, invPdfLambda, lambda, rng);
                    if (scene.envIndex >= 0)
                        L += thr * neeEnv(scene, h, rhoEl, invPdfLambda, lambda, rng);

                    // Fluorescent channel: draw an excitation wavelength lambdaIn.
                    double Mint = m.fluoEmitSampler.integral;
                    bool haveFluoro = (Mint > 0.0 && m.fluoYield > 0.0);
                    double gOut = 0.0, rhoFluo = 0.0, lambdaIn = 0.0, invPdfIn = 0.0;
                    if (haveFluoro) {
                        // Emission colour at lambda(out), deconvolved from the camera-
                        // path wavelength-sampling density (invPdfLambda) so the
                        // reradiated colour follows M(lambda), not the light SPD used
                        // to sample lambda.
                        gOut = (m.fluoEmit(lambda) / Mint) * invPdfLambda;
                        double pin = 0.0;
                        lambdaIn = scene.emitSampler.sample(rng, pin);
                        if (pin > 0.0) {
                            invPdfIn = scene.invPdfLambda(lambdaIn);
                            double rhoIn, aEffIn;
                            fluoroWeights(m, lambdaIn, rhoIn, aEffIn);   // shared with forward
                            rhoFluo = aEffIn * m.fluoYield;              // reradiation albedo @lambdaIn
                            if (rhoFluo > 0.0) {                          // fluoro DIRECT NEE
                                L += thr * gOut * neeLight(scene, h, rhoFluo, invPdfIn, lambdaIn, rng);
                                if (scene.envIndex >= 0)
                                    L += thr * gOut * neeEnv(scene, h, rhoFluo, invPdfIn, lambdaIn, rng);
                            }
                        }
                    }

                    // Single stochastic continuation for INDIRECT illumination:
                    //   [0, rhoEl)          -> elastic bounce at lambda (throughput kept)
                    //   [rhoEl, rhoEl+pF)   -> fluoro bounce, switch to lambdaIn, throughput
                    //                          *= wFluo/pF (indirect excitation)
                    //   else                -> terminate. pF ~ wFluo keeps the surviving
                    //   weight ~O(1); f*cos/pdf folds into the cosine-sampled direction.
                    double wFluo = gOut * rhoFluo;               // natural indirect-fluoro weight
                    double pF = (wFluo > 0.0) ? std::min(std::max(0.0, 1.0 - rhoEl), wFluo) : 0.0;
                    double u = rng.uniform();
                    if (u < rhoEl) {                             // elastic continuation
                        Vec3 wOut = cosineHemisphere(h.n, rng);
                        contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                        ray = Ray{h.p + h.n * 1e-6, wOut};
                        specularArrival = false;
                        break;
                    } else if (u < rhoEl + pF) {                 // fluoro (wavelength-switched)
                        thr *= wFluo / pF;
                        lambda = lambdaIn;                       // Stokes shift (to the input wl)
                        invPdfLambda = invPdfIn;
                        Vec3 wOut = cosineHemisphere(h.n, rng);
                        contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                        ray = Ray{h.p + h.n * 1e-6, wOut};
                        specularArrival = false;
                        break;
                    }
                    return L;                                    // absorbed / terminated
                }
                case MatType::DiffuseTransmit: {
                    // Two-lobe Lambertian (backward adjoint of render.h DiffuseTransmit):
                    // NEE the reflect lobe against lights in the front (+h.n) hemisphere
                    // and the transmit lobe against lights in the back (-h.n) hemisphere
                    // (a normal-flipped Hit copy reuses neeLight/neeEnv for the back side).
                    // The continuation picks reflect / transmit / absorb analogously to a
                    // Russian-roulette diffuse bounce, throughput unchanged on survival.
                    double rhoR = clamp01(diffuseReflectance(scene, m, h, lambda));
                    double rhoT = clamp01(m.transmit(lambda));
                    double sum = rhoR + rhoT;
                    if (sum > 1.0) { rhoR /= sum; rhoT /= sum; sum = 1.0; }   // energy guard
                    L += thr * neeLight(scene, h, rhoR, invPdfLambda, lambda, rng);
                    if (scene.envIndex >= 0)
                        L += thr * neeEnv(scene, h, rhoR, invPdfLambda, lambda, rng);
                    Hit hb = h; hb.n = -h.n;                 // back hemisphere for the transmit lobe
                    L += thr * neeLight(scene, hb, rhoT, invPdfLambda, lambda, rng);
                    if (scene.envIndex >= 0)
                        L += thr * neeEnv(scene, hb, rhoT, invPdfLambda, lambda, rng);
                    double u = rng.uniform();
                    if (u < rhoR) {                          // reflect continuation (front)
                        Vec3 wOut = cosineHemisphere(h.n, rng);
                        contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                        ray = Ray{h.p + h.n * 1e-6, wOut};
                        specularArrival = false;
                        break;
                    } else if (u < sum) {                    // transmit continuation (back)
                        Vec3 wOut = cosineHemisphere(-h.n, rng);
                        contBsdfPdf = std::max(0.0, dot(wOut, -h.n)) / PI;
                        ray = Ray{h.p - h.n * 1e-6, wOut};
                        specularArrival = false;
                        break;
                    }
                    return L;                                // absorbed / terminated
                }
                case MatType::Diffuse:
                default: {
                    double rho = clamp01(diffuseReflectance(scene, m, h, lambda));
                    L += thr * neeLight(scene, h, rho, invPdfLambda, lambda, rng);
                    if (scene.envIndex >= 0)   // env-NEE toward the sky (MIS'd on miss)
                        L += thr * neeEnv(scene, h, rho, invPdfLambda, lambda, rng);
                    // Russian roulette on the albedo (throughput unchanged on
                    // survival) — matches the forward tracer's diffuse handling.
                    if (rng.uniform() >= rho) return L;
                    Vec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                    ray = Ray{h.p + h.n * 1e-6, wOut};
                    specularArrival = false;
                    break;
                }
            }
        }
        return L;
    }

    // Render `spp` samples per pixel into `film` (accumulates cieXYZ * radiance,
    // exactly like the forward film, so writeFilm with N=spp displays it). Renders
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
                        double L = radiance(scene, ray, lambda, invPdfLambda, rng);
                        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * (L * wLens));
                        continue;
                    }
                    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
                    double L = radiance(scene, ray, lambda, invPdfLambda, rng);
                    film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * L);
                }
            }
        }
    }
};
