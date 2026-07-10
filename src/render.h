// Forward photon tracing.
//   Model A: photon physically lands on a contact sensor -> deposit, terminate.
//   Model B: at every surface vertex, connect to the camera pinhole and splat.
// Both can share a scene; typically A uses a sensor wall, B leaves it open.
//
// Energy bookkeeping (absorbed/escaped/residual) tracks the PHOTON's own energy
// only. Model-B splats are side-channel measurements and are intentionally NOT
// counted as energy sinks, so the conservation test stays valid in both modes.
#pragma once
#include <cstdint>
#include <algorithm>
#include "scene.h"
#include "camera.h"

constexpr double PI = 3.141592653589793;

struct EnergyReport {
    double emitted = 0, absorbed = 0, sensor = 0, escaped = 0, residual = 0;
};

inline double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

// Power-cosine lobe around a mirror direction (rough specular). roughness in
// [0,1]: 0 -> sharp mirror, 1 -> broad. Returns a sampled reflection direction.
inline Vec3 sampleGlossy(const Vec3& mdir, double roughness, Pcg32& rng) {
    double rr = roughness < 1e-3 ? 1e-3 : roughness;
    double e = 2.0 / (rr * rr) - 2.0; if (e < 0) e = 0;
    double u1 = rng.uniform(), u2 = rng.uniform();
    double cosT = std::pow(u1, 1.0 / (e + 1.0));
    double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
    double phi = 2.0 * PI * u2;
    Vec3 t, b; onb(mdir, t, b);
    return normalize(t * (sinT * std::cos(phi)) + b * (sinT * std::sin(phi)) + mdir * cosT);
}

// --- Fluorescence interaction (shared by the forward tracer and -checkfluoro) --
// A fluorescent surface has two competing channels: elastic diffuse reflection
// (albedo rho, wavelength preserved) and dye excitation (prob aEff = min(eps,
// 1-rho) so the channels never exceed unity, preserving energy). Excited photons
// re-radiate with probability Q at a Stokes-shifted wavelength drawn from M.
inline void fluoroWeights(const Material& m, double lambda, double& rho, double& aEff) {
    rho = clamp01(m.reflect(lambda));
    double eps = clamp01(m.fluoAbsorb(lambda));
    aEff = std::min(eps, std::max(0.0, 1.0 - rho));
}

enum class FluoroEvent { Elastic, Reemit, Absorb };
struct FluoroResult { FluoroEvent event; double lambdaOut; };

// Stochastically resolve a fluorescent interaction for an incoming photon at
// lambdaIn. Elastic -> reflect at lambdaIn; Reemit -> re-radiate at lambdaOut~M;
// Absorb -> photon lost (dye heat / non-excited fraction). Throughput weight is
// unchanged in all surviving branches (the branch probabilities carry the
// reradiation efficiency, and M/pdf cancels for the sampled lambdaOut).
inline FluoroResult fluoroInteract(const Material& m, double lambdaIn, Pcg32& rng) {
    double rho, aEff; fluoroWeights(m, lambdaIn, rho, aEff);
    double u = rng.uniform();
    if (u < rho)          return {FluoroEvent::Elastic, lambdaIn};
    if (u < rho + aEff) {
        if (rng.uniform() >= m.fluoYield) return {FluoroEvent::Absorb, 0.0};
        double pf; double lp = m.fluoEmitSampler.sample(rng, pf);
        return {FluoroEvent::Reemit, lp};
    }
    return {FluoroEvent::Absorb, 0.0};
}

// --- Henyey-Greenstein phase function (participating media) ------------------
// p(cosTheta) normalized so its integral over the sphere is 1. cosTheta is the
// cosine between the photon's propagation direction and the scattered direction;
// g in (-1,1): g>0 forward-peaked, g<0 back-scattering, g=0 isotropic.
inline double hgPhase(double cosTheta, double g) {
    double d = 1.0 + g * g - 2.0 * g * cosTheta;
    if (d < 1e-9) d = 1e-9;
    return (1.0 - g * g) / (4.0 * PI * d * std::sqrt(d));
}

// Sample a scattered direction around the propagation direction `wi` from the HG
// distribution. The sampled cosTheta has mean value g (forward for g>0), so the
// returned direction is importance-sampled proportional to the phase function.
inline Vec3 sampleHG(const Vec3& wi, double g, Pcg32& rng) {
    double u1 = rng.uniform(), u2 = rng.uniform();
    double cosT;
    if (std::fabs(g) < 1e-3) {
        cosT = 1.0 - 2.0 * u1;                          // isotropic
    } else {
        double sq = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1);
        cosT = (1.0 + g * g - sq * sq) / (2.0 * g);
    }
    double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
    double phi = 2.0 * PI * u2;
    Vec3 t, b; onb(wi, t, b);
    return normalize(t * (sinT * std::cos(phi)) + b * (sinT * std::sin(phi)) + wi * cosT);
}

// --- Thin-film interference reflectance (iridescence) ------------------------
// A thin dielectric film (index n1, thickness d nanometres) coats a substrate of
// index n2, with incident medium n0. The beams reflected off the top (n0|n1) and
// bottom (n1|n2) interfaces interfere; the round-trip optical-path phase
//   phi = 4*pi*n1*d*cos(theta1) / lambda
// makes the reflectance oscillate with wavelength AND angle -> structural colour
// (soap bubbles, oil slicks, beetle shells, anodised metal). Returns the
// unpolarised power reflectance R(lambda, theta) in [0,1]; the transmitted
// fraction is 1-R, so the film is lossless. cosI is cos of the incidence angle in
// n0; d and lambda must share units (nanometres here).
//
// STAGE 1 (iridescence) uses the two-beam (first-order) interference of the two
// front reflections, polarisation-averaged. This reproduces the colour shift
// correctly. The exact Airy multiple-beam summation is a drop-in refinement (the
// thin-film-interference milestone): divide each per-polarisation reflectance by
// (1 + r01^2 r12^2 + 2 r01 r12 cos phi), which sharpens the higher-order fringes
// and keeps R physically bounded without clamping.
inline double thinFilmReflectance(double n0, double n1, double n2, double d,
                                  double cosI, double lambda) {
    cosI = clamp01(std::fabs(cosI));
    double sin0_2 = std::max(0.0, 1.0 - cosI * cosI);
    // Snell into the film: sin(theta1) = (n0/n1) sin(theta0).
    double sin1_2 = (n0 * n0) / (n1 * n1) * sin0_2;
    if (sin1_2 >= 1.0) return 1.0;                       // (n1>=n0 so this won't fire)
    double cos1 = std::sqrt(1.0 - sin1_2);
    // Snell into the substrate: sin(theta2) = (n0/n2) sin(theta0).
    double sin2_2 = (n0 * n0) / (n2 * n2) * sin0_2;
    bool tir = sin2_2 >= 1.0;                            // TIR at the n1|n2 interface
    double cos2 = tir ? 0.0 : std::sqrt(1.0 - sin2_2);
    // Fresnel amplitude reflection coefficients (s- and p-polarised) at each face.
    auto rS = [](double na, double ca, double nb, double cb) {
        return (na * ca - nb * cb) / (na * ca + nb * cb);
    };
    auto rP = [](double na, double ca, double nb, double cb) {
        return (nb * ca - na * cb) / (nb * ca + na * cb);
    };
    double r01s = rS(n0, cosI, n1, cos1), r01p = rP(n0, cosI, n1, cos1);
    double r12s = tir ? 1.0 : rS(n1, cos1, n2, cos2);   // |r|=1 amplitude on TIR
    double r12p = tir ? 1.0 : rP(n1, cos1, n2, cos2);
    double phi  = (4.0 * PI * n1 * d * cos1) / lambda;   // interference phase
    double cphi = std::cos(phi);
    // Two-beam interference reflectance per polarisation (STAGE 1). STAGE 2 will
    // divide by (1 + r01*r01*r12*r12 + 2*r01*r12*cphi) for the exact Airy result.
    auto Rpol = [&](double r01, double r12) {
        return clamp01(r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * cphi);
    };
    return 0.5 * (Rpol(r01s, r12s) + Rpol(r01p, r12p));
}

struct Renderer {
    int maxBounce = 32;          // hard safety cap; Russian roulette normally
                                 // terminates paths well before this.
    bool forwardCatch = false;   // model A perspective: catch photons at the aperture,
                                 // no connect/splat (photons must physically fly in).

    // Model A: map a contact-sensor hit to a pixel and deposit.
    void deposit(const Sensor& s, Film& film, const Vec3& p, double lambda, double beta) const {
        Vec3 rel = p - s.origin;
        double uu = dot(rel, s.uAxis) / dot(s.uAxis, s.uAxis);
        double vv = dot(rel, s.vAxis) / dot(s.vAxis, s.vAxis);
        if (uu < 0 || uu >= 1 || vv < 0 || vv >= 1) return;
        int px = (int)(uu * film.resX), py = (int)(vv * film.resY);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * beta);
    }

    // Model B: connect a surface vertex to the pinhole and splat onto the film.
    // f = rho/pi (Lambertian). Contribution = beta * f * G * We, with
    //   G  = cosSurf * cosCam / dist^2   (geometry term)
    //   We = 1 / (A * cosCam^4)          (pinhole importance, A = image-plane area)
    void connect(const Scene& scene, const Camera& cam, Film& film,
                 const Vec3& p, const Vec3& n, double lambda, double beta, double rho) const {
        Vec3 toCam = cam.eye - p;
        double dist = length(toCam);
        Vec3 wdir = toCam / dist;
        double cosSurf = dot(n, wdir);
        if (cosSurf <= 0) return;                       // camera behind surface
        int px, py; double cosCam, dist2;
        if (!cam.project(p, px, py, cosCam, dist2)) return;
        if (scene.occluded(p + n * 1e-6, wdir, dist - 2e-6)) return;

        double f = rho / PI;
        double G = cosSurf * cosCam / dist2;
        double We = 1.0 / (cam.imagePlaneArea() * cosCam * cosCam * cosCam * cosCam);
        double contrib = beta * f * G * We;
        // Beer-Lambert attenuation of the shadow ray through a global fog.
        if (scene.medium.enabled)
            contrib *= std::exp(-scene.medium.sigmaT(lambda) * dist);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Model B for a VOLUME scattering vertex: connect the collision point to the
    // pinhole. The surface BRDF/cosine is replaced by the phase function and the
    // single-scattering albedo; there is no surface normal. wIn is the photon's
    // propagation direction into the collision.
    //   contrib = beta * albedo * p_HG(cos) * (cosCam/dist^2) * We * T_fog
    void connectVolume(const Scene& scene, const Camera& cam, Film& film,
                       const Vec3& p, const Vec3& wIn, double lambda, double beta) const {
        Vec3 toCam = cam.eye - p;
        double dist = length(toCam);
        Vec3 wdir = toCam / dist;
        int px, py; double cosCam, dist2;
        if (!cam.project(p, px, py, cosCam, dist2)) return;
        if (scene.occluded(p + wdir * 1e-6, wdir, dist - 2e-6)) return;

        double ph = hgPhase(dot(wIn, wdir), scene.medium.g);
        double Lambda = scene.medium.albedo(lambda);
        double G = cosCam / dist2;
        double We = 1.0 / (cam.imagePlaneArea() * cosCam * cosCam * cosCam * cosCam);
        double contrib = beta * Lambda * ph * G * We;
        contrib *= std::exp(-scene.medium.sigmaT(lambda) * dist);   // fog transmittance
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Trace a single photon. sensorFilm (model A) and/or cam+camFilm (model B)
    // may be null; the caller supplies per-thread films for parallel runs.
    void tracePhoton(const Scene& scene, const Camera* cam, Film* sensorFilm,
                     Film* camFilm, Pcg32& rng, EnergyReport& e) const {
        // --- Emission ---
        double u1 = rng.uniform(), u2 = rng.uniform();
        Vec3 origin = scene.lightOrigin + scene.lightU * u1 + scene.lightV * u2;
        Vec3 dir = scene.collimated ? scene.beamDir
                                    : cosineHemisphere(scene.lightNormal, rng);
        double pdfL = 0.0;
        double lambda = scene.lightSpd.sample(rng, pdfL);
        if (pdfL <= 0) return;
        // Constant weight because we importance-sample p(lambda)=Le/integral.
        double beta = scene.lightEmitIntegral * scene.lightArea * PI;
        e.emitted += beta;

        // Direct light -> camera: makes the source itself visible. The Lambertian
        // emitter term is 1/pi, i.e. connect() with rho=1 using the light normal.
        // (Skipped in forward-catch mode; there the aperture test below handles it.)
        if (cam && camFilm && !forwardCatch)
            connect(scene, *cam, *camFilm, origin, scene.lightNormal, lambda, beta, 1.0);

        Ray ray{origin + dir * 1e-6, dir};

        for (int bounce = 0; bounce < maxBounce; ++bounce) {
            Hit h = scene.closestHit(ray);
            double dSurf = h.valid ? h.t : 1e30;

            // Homogeneous fog: sample a free-flight collision. If it precedes the
            // surface, the photon interacts in the volume (in-scatter connect,
            // then scatter-or-absorb). Beer-Lambert transmittance is implicit in
            // the exponential free-flight, so beta is unchanged (analog MC).
            double dEvent = dSurf;
            bool mediumEvent = false;
            Vec3 mp;
            if (scene.medium.enabled) {
                double st = scene.medium.sigmaT(lambda);
                if (st > 0.0) {
                    double tMed = -std::log(1.0 - rng.uniform()) / st;
                    if (tMed < dSurf) { dEvent = tMed; mediumEvent = true; mp = ray.o + ray.d * tMed; }
                }
            }

            // Model A perspective catch: if the photon flies through the aperture
            // (nearer than the surface AND any fog collision), it lands on the film.
            if (forwardCatch && cam && camFilm) {
                int px, py;
                if (cam->catchPhoton(ray, dEvent, px, py)) {
                    camFilm->add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * beta);
                    e.sensor += beta;
                    return;
                }
            }

            if (mediumEvent) {
                if (cam && camFilm && !forwardCatch)
                    connectVolume(scene, *cam, *camFilm, mp, ray.d, lambda, beta);
                // Scatter (prob albedo) or absorb; throughput unchanged on scatter.
                if (rng.uniform() >= scene.medium.albedo(lambda)) { e.absorbed += beta; return; }
                ray = Ray{mp, sampleHG(ray.d, scene.medium.g, rng)};
                continue;
            }

            if (!h.valid) { e.escaped += beta; return; }

            if (h.sensorId >= 0) {
                if (sensorFilm) deposit(scene.sensor, *sensorFilm, h.p, lambda, beta);
                e.sensor += beta;
                return;
            }

            const Material& m = scene.mats[h.matId];
            // Specular/glossy vertices skip the camera connection (delta or
            // near-delta BSDF -> ~zero connection pdf; the SDS limitation).
            switch (m.type) {
                case MatType::Dielectric: {
                    ray = refractOrReflect(m, h, ray.d, lambda, rng);
                    continue;                       // lossless; beta unchanged
                }
                case MatType::ThinFilm: {
                    // Iridescent coated dielectric: specular reflect-or-refract
                    // with a thin-film interference reflectance (structural colour).
                    ray = thinFilmInterface(m, h, ray.d, lambda, rng);
                    continue;                       // lossless; beta unchanged
                }
                case MatType::Mirror: {
                    double r = clamp01(m.reflect(lambda));
                    // Russian roulette: absorb with prob (1-r), else reflect with
                    // beta unchanged. Unbiased and caps path length naturally.
                    if (rng.uniform() >= r) { e.absorbed += beta; return; }
                    Vec3 o = reflect(ray.d, h.n);
                    ray = Ray{h.p + h.n * 1e-6, o};
                    continue;
                }
                case MatType::HalfMirror: {
                    double r = clamp01(m.reflect(lambda)); // reflect probability
                    if (rng.uniform() < r) {
                        Vec3 o = reflect(ray.d, h.n);
                        ray = Ray{h.p + h.n * 1e-6, o};
                    } else {
                        ray = Ray{h.p + ray.d * 1e-6, ray.d}; // transmit straight
                    }
                    continue;                       // lossless split
                }
                case MatType::Glossy: {
                    double r = clamp01(m.reflect(lambda));
                    // Russian roulette on reflectance (see Mirror).
                    if (rng.uniform() >= r) { e.absorbed += beta; return; }
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), m.roughness, rng);
                    if (dot(o, h.n) <= 0) { e.absorbed += beta; return; } // below surface
                    ray = Ray{h.p + h.n * 1e-6, o};
                    continue;
                }
                case MatType::Fluorescent: {
                    double rho, aEff; fluoroWeights(m, lambda, rho, aEff);
                    // Model-B connections: the elastic channel splats at the
                    // incoming lambda; the fluorescent channel samples one
                    // lambda' ~ M and splats the glow (albedo aEff*Q) with the
                    // camera-response evaluated at lambda' (Stokes-shifted colour).
                    if (cam && camFilm && !forwardCatch) {
                        connect(scene, *cam, *camFilm, h.p, h.n, lambda, beta, rho);
                        if (aEff > 0.0 && m.fluoYield > 0.0 && m.fluoEmitSampler.integral > 0.0) {
                            double pf; double lp = m.fluoEmitSampler.sample(rng, pf);
                            connect(scene, *cam, *camFilm, h.p, h.n, lp, beta, aEff * m.fluoYield);
                        }
                    }
                    FluoroResult fr = fluoroInteract(m, lambda, rng);
                    if (fr.event == FluoroEvent::Absorb) { e.absorbed += beta; return; }
                    lambda = fr.lambdaOut;              // Stokes-shifted on Reemit
                    ray = Ray{h.p + h.n * 1e-6, cosineHemisphere(h.n, rng)};
                    continue;                           // beta unchanged (see above)
                }
                case MatType::Diffuse:
                default: {
                    double rho = clamp01(m.reflect(lambda));
                    if (cam && camFilm && !forwardCatch) connect(scene, *cam, *camFilm, h.p, h.n, lambda, beta, rho);
                    // Russian roulette: absorb with prob (1-rho), else scatter
                    // with beta unchanged. Unbiased; average path length ~1/(1-rho)
                    // bounces instead of running to the maxBounce cap.
                    if (rng.uniform() >= rho) { e.absorbed += beta; return; }
                    ray = Ray{h.p + h.n * 1e-6, cosineHemisphere(h.n, rng)};
                    continue;
                }
            }
        }
        e.residual += beta;
    }

    // Dielectric interface: Fresnel-weighted stochastic choice of specular
    // reflection or refraction (Snell), with wavelength-dependent index -> dispersion.
    Ray refractOrReflect(const Material& m, const Hit& h, const Vec3& d,
                         double lambda, Pcg32& rng) const {
        double ng = m.ior(lambda);
        bool entering = dot(d, h.ng) < 0.0;
        Vec3 nl = entering ? h.ng : -h.ng;      // normal on the incidence side
        double n1 = entering ? 1.0 : ng;
        double n2 = entering ? ng : 1.0;
        double eta = n1 / n2;
        double cosI = -dot(d, nl);              // > 0
        double sin2t = eta * eta * (1.0 - cosI * cosI);

        Vec3 outDir;
        if (sin2t > 1.0) {
            outDir = reflect(d, nl);            // total internal reflection
        } else {
            double cosT = std::sqrt(1.0 - sin2t);
            double rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
            double rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
            double R = 0.5 * (rs * rs + rp * rp);
            if (rng.uniform() < R) outDir = reflect(d, nl);
            else outDir = eta * d + nl * (eta * cosI - cosT); // Snell refraction
        }
        outDir = normalize(outDir);
        return Ray{h.p + outDir * 1e-6, outDir};
    }

    // Thin-film-coated dielectric interface (iridescence). Structurally identical
    // to refractOrReflect (specular reflect-or-refract into the substrate index),
    // but the reflection probability is the thin-film interference reflectance
    // R(lambda, theta) rather than the single-interface Fresnel R. Lossless:
    // reflect with prob R, else refract into the substrate. Because it is purely
    // specular it needs no camera connection and the backward tracer handles it
    // exactly like Dielectric (so modes R/V remain valid).
    Ray thinFilmInterface(const Material& m, const Hit& h, const Vec3& d,
                          double lambda, Pcg32& rng) const {
        double ns = m.ior(lambda);              // substrate index (spectral -> dispersion)
        double nf = m.filmIor;                  // coating film index
        bool entering = dot(d, h.ng) < 0.0;
        Vec3 nl = entering ? h.ng : -h.ng;      // normal on the incidence side
        double nA = entering ? 1.0 : ns;        // incidence-side medium
        double nB = entering ? ns : 1.0;        // transmission-side medium
        double eta = nA / nB;
        double cosI = -dot(d, nl);              // > 0
        double sin2t = eta * eta * (1.0 - cosI * cosI);

        Vec3 outDir;
        if (sin2t > 1.0) {
            outDir = reflect(d, nl);            // total internal reflection
        } else {
            double cosT = std::sqrt(1.0 - sin2t);
            // Interference reflectance for the actual stack traversed this hit:
            // incidence medium nA, coating nf, transmission medium nB. Reciprocal,
            // so entering and exiting rays see the same R (energy consistent).
            double R = thinFilmReflectance(nA, nf, nB, m.filmThickness, cosI, lambda);
            if (rng.uniform() < R) outDir = reflect(d, nl);
            else outDir = eta * d + nl * (eta * cosI - cosT); // Snell refraction
        }
        outDir = normalize(outDir);
        return Ray{h.p + outDir * 1e-6, outDir};
    }
};
