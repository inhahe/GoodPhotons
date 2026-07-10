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

            // Model A perspective catch: if the photon flies through the aperture
            // (nearer than any surface), it lands on the film. Supports mirrors,
            // glass, everything — pure forward physics, no connection.
            if (forwardCatch && cam && camFilm) {
                double hitDist = h.valid ? h.t : 1e30;
                int px, py;
                if (cam->catchPhoton(ray, hitDist, px, py)) {
                    camFilm->add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * beta);
                    e.sensor += beta;
                    return;
                }
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
};
