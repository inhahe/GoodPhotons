// Forward photon tracing: emit from the light, bounce diffusely, deposit on the
// contact sensor (camera model A, no lens). Deterministic energy bookkeeping
// (no Russian roulette yet) so we can verify conservation.
#pragma once
#include <cstdint>
#include "scene.h"

struct EnergyReport {
    double emitted = 0, absorbed = 0, sensor = 0, escaped = 0, residual = 0;
};

struct Renderer {
    int maxBounce = 32;
    double betaCutoff = 1e-6;

    // Map a sensor-plane hit point to a pixel; deposit XYZ contribution.
    void deposit(Sensor& s, const Vec3& p, double lambda, double beta) const {
        Vec3 rel = p - s.origin;
        double uu = dot(rel, s.uAxis) / dot(s.uAxis, s.uAxis);
        double vv = dot(rel, s.vAxis) / dot(s.vAxis, s.vAxis);
        if (uu < 0 || uu >= 1 || vv < 0 || vv >= 1) return;
        int px = (int)(uu * s.resX), py = (int)(vv * s.resY);
        size_t idx = (size_t)py * s.resX + px;
        s.xyz[idx] += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * beta;
        s.hits[idx] += 1.0;
    }

    // Trace a single photon. Accumulates into scene.sensor and the energy report.
    void tracePhoton(Scene& scene, Pcg32& rng, EnergyReport& e) const {
        // --- Emission ---
        double u1 = rng.uniform(), u2 = rng.uniform();
        Vec3 origin = scene.lightOrigin + scene.lightU * u1 + scene.lightV * u2;
        Vec3 dir = cosineHemisphere(scene.lightNormal, rng);
        double pdfL = 0.0;
        double lambda = scene.lightSpd.sample(rng, pdfL);
        if (pdfL <= 0) return;
        // Photon weight so the ensemble reconstructs emitted flux:
        //   w = Le(lambda) * A * pi / p(lambda).
        // We importance-sample p(lambda) = Le(lambda)/integral, so Le/p = integral,
        // giving a constant weight independent of the sampled wavelength.
        double beta = scene.lightEmitIntegral * scene.lightArea * 3.141592653589793;
        e.emitted += beta;

        Ray ray{origin + dir * 1e-6, dir};

        for (int bounce = 0; bounce < maxBounce; ++bounce) {
            Hit h = scene.closestHit(ray);
            if (h.tri < 0) { e.escaped += beta; return; } // left the scene

            const Tri& tri = scene.tris[h.tri];
            if (tri.sensorId >= 0) {
                deposit(scene.sensor, h.p, lambda, beta);
                e.sensor += beta;
                return;
            }

            double rho = scene.mats[tri.matId].reflect(lambda);
            rho = std::min(std::max(rho, 0.0), 1.0);
            e.absorbed += beta * (1.0 - rho);   // absorbed fraction stays here
            beta *= rho;
            if (beta < betaCutoff) { return; }

            Vec3 ns = h.n;
            Vec3 newDir = cosineHemisphere(ns, rng);
            ray = Ray{h.p + ns * 1e-6, newDir};
        }
        e.residual += beta; // hit bounce cap still carrying energy
    }
};
