// Bridge between the renderer's Scene / Material / Hit and the standalone fiber BCSDF.
//
// `hair.h` deliberately knows nothing about this renderer — that is what lets
// `-checkhair` run nine sections of physics with no scene, no camera and no BVH. Every
// piece that needs a Scene therefore lives here instead: reading the material's spectral
// slots, inverting an authored colour into an absorption, building the local fiber frame,
// and turning a BCSDF value into the quantity the light transport actually wants.
//
// THE ONE THING TO UNDERSTAND HERE is the projection factor, because it is not the
// surface one and getting it wrong is invisible until you check energy.
//
// A surface vertex splats to the camera with `beta * f * dot(n, w)`: the cosine is the
// foreshortening of the surface patch. A fiber has no such patch. What is foreshortened
// is the STRAND, and a round strand's projected width is the same from every azimuth, so
// the only foreshortening is longitudinal: `cos theta = sqrt(1 - w_x^2)` in the fiber
// frame. That is also exactly the measure the BCSDF is normalised against — the furnace
// identity `-checkhair` §1 asserts is `integral f cos(theta) dw = 1` with THAT cosine —
// so the two agree by construction. Multiplying by `dot(n, w)` instead would both
// double-count the tube's curvature and break energy conservation, and (worse) it would
// look almost right: the error is a smooth function of angle, so it reads as "my hair
// shader is a bit dark at grazing angles" rather than as a bug.
//
// `hairFCos()` therefore returns `f * cos(theta_long)`, which is the complete
// "BSDF times projection" factor a connection needs. PBRT reaches the same number from
// the other side — it divides `f` by the renderer's `AbsCosTheta` so the integrator's
// own cosine cancels — but doing it explicitly here means the fiber path never quietly
// depends on a cancellation happening somewhere else.
#pragma once
#include "hair.h"
#include "scene.h"

// Everything needed to evaluate or sample the fiber BCSDF at one hit, for one wavelength.
struct HairShade {
    hair::Bcsdf b;
    hair::Frame fr;
    Vec3   woLocal{1, 0, 0};   // the direction the path ARRIVED from, in the fiber frame
    double radius = 0.0;       // world fiber radius; 0 = this hit is not on a strand
};

// Build the BCSDF at a hit. `wPrev` points away from the surface, back along the path
// that got here: toward the light for the forward tracer, toward the eye for the backward
// one. It is the model's reference direction — the impact parameter `h` is measured
// relative to it (as it is in PBRT, where `h` comes from the ribbon that was turned to
// face the incident ray), so it must be the arrival direction and not the sampled one.
inline HairShade hairShadeAt(const Scene& scene, const Material& m, const Hit& hit,
                             double lambda, const Vec3& wPrev) {
    hair::Params pr;
    pr.eta   = m.hairEta;
    pr.betaM = m.hairBetaM;
    pr.betaN = m.hairBetaN;
    pr.alpha = m.hairAlpha;

    double sigmaA;
    if (m.hairSigmaAFromReflect) {
        // Chiang eq. 9 run per-wavelength: the authored `reflect` spectrum is read as the
        // reflectance curve the fiber should END UP with, and inverted into the absorption
        // that produces it. Going through diffuseReflectance() (rather than the constant
        // spectrum) is what lets a groom carry a texture, a record or a pattern on its
        // colour — root-to-tip darkening is just `reflect [0.1 0.6](u)`.
        double c = diffuseReflectance(scene, m, hit, lambda);
        c = c < 0.0 ? 0.0 : (c > 1.0 ? 1.0 : c);
        sigmaA = hair::sigmaAFromReflectance(c, m.hairBetaN);
    } else {
        sigmaA = std::max(0.0, m.hairSigmaA(lambda));
    }

    HairShade s;
    // hit.n is oriented against the arriving ray, so it is the normal on the side the path
    // came from — which is the side `h` and `gamma_o` are defined from.
    s.fr      = hair::frameFromHit(hit.n, hit.tangent);
    s.woLocal = hair::toLocal(s.fr, wPrev);
    s.b       = hair::make(pr, hair::hFromHit(hit.n, hit.tangent, wPrev), sigmaA);
    s.radius  = hit.fiberRadius;
    return s;
}

// The complete "BCSDF times projection" factor toward a world direction — see the header
// comment. This is what multiplies the throughput in a connection, in place of the
// surface path's `f * dot(n, w)`.
inline double hairFCos(const HairShade& s, const Vec3& wWorld) {
    const Vec3 wl = hair::toLocal(s.fr, wWorld);
    const double cosLong = hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
    return hair::f(s.b, s.woLocal, wl) * cosLong;
}

// How far a connection or a scattered ray must step to clear the strand's own body.
//
// The near-field model puts the TT and TRT exits at the ENTRY point, but the geometry is a
// real solid tube: a ray leaving toward the far side would immediately re-hit the strand
// it just came from and be reported as occluded, deleting the forward glow that dominates
// a light-coloured coat. Stepping a couple of diameters clears the tube (strand radii are
// microns, so this displaces nothing visible), and on a non-fiber hit — where fiberRadius
// is 0 — it degrades to the ordinary 1e-6 surface offset.
inline double hairExitOffset(const HairShade& s, const Vec3& n, const Vec3& w) {
    if (dot(n, w) >= 0.0) return 1e-6;              // leaving on the arrival side: normal case
    return (s.radius > 0.0) ? 2.5 * s.radius + 1e-9 : 1e-6;
}
