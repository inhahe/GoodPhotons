// Volume phase functions shared by every tracer (forward A/B/C, backward R, BDPT D).
// ---------------------------------------------------------------------------
// Split out of render.h so the low-level Medium (scene.h) can dispatch between the
// smooth single-parameter Henyey-Greenstein (HG) lobe and the wavelength-dependent
// Airy droplet phase (rainbow.h) without a circular include. The convention is that
// a phase function integrates to 1 over the sphere, so its value at a direction IS
// the pdf over solid angle when the scatter direction is importance-sampled from it.
#pragma once
#include <cmath>
#include "linalg.h"   // Vec3, onb, normalize, PI
#include "rng.h"      // Pcg32
#include "rainbow.h"  // rainbow::RainbowPhase (Airy droplet phase)

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
