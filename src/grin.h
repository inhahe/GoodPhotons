// Gradient-index (GRIN) ray marching — shared by every CPU transport path.
//
// A medium carrying an `ior` field n(x,y,z) is a GRADIENT-INDEX region: rays do NOT
// travel straight through it, they bend continuously obeying the Eikonal ray equation
//   d/ds( n · dr/ds ) = ∇n.
// This header holds the one canonical marcher so the forward light tracer (modes A/B/C),
// the backward path tracer (mode D reference) and the bidirectional tracer all bend rays
// through GRIN regions IDENTICALLY — a single source of truth instead of three copies.
//
// Semantics (Phase-2, matching the original backward-tracer prototype exactly):
//   * The marcher only BENDS the ray. Absorption / scattering *along the curved path*
//     inside a GRIN region is not integrated here — the classic use is a clear bending
//     field (sigma_a = sigma_s = 0), and any residual fog on the final straight segment
//     to the surface is still handled by each tracer's normal medium sampling. So a GRIN
//     region that also scatters is approximated, exactly as in Phase 1.
//   * Marching does NOT consume a bounce: it is a pre-pass run before each bounce's
//     closestHit. When the ray reaches a surface (within one step) or leaves every GRIN
//     region, we stop and let the straight-ray body take over on the short remaining leg.
//   * `sceneHasGrin` gates the whole thing: `ior`-free scenes never call the marcher, so
//     they stay BIT-IDENTICAL to the pre-GRIN renderer (zero added cost).
//
// The refractive index is wavelength-independent (n is a pattern program over x y z r),
// so the marcher takes no lambda — one bent geometry serves every spectral sample.
#pragma once
#include "scene.h"

namespace grin {

// True iff any enabled medium carries an `ior` field (i.e. is a GRIN region).
inline bool sceneHasGrin(const Scene& scene) {
    for (const auto& md : scene.media)
        if (md.enabled && md.grin()) return true;
    return false;
}

// Advance `ray` through any GRIN region(s) it enters via symplectic Eikonal marching,
// stopping when a surface is within one step or the ray has left all GRIN regions. A
// no-op when the ray is nowhere near a GRIN region. Callers gate on sceneHasGrin() so
// this is never entered for ordinary scenes.
inline void march(const Scene& scene, Ray& ray) {
    constexpr int GRIN_MAX_STEPS = 200000;   // safety cap on marching steps
    // The GRIN region containing a point (highest-priority membership), or null.
    auto grinAt = [&](const Vec3& p) -> const Medium* {
        for (const auto& md : scene.media)
            if (md.enabled && md.grin() && md.insideBound(p)) return &md;
        return nullptr;
    };
    for (int gstep = 0; gstep < GRIN_MAX_STEPS; ++gstep) {
        const Medium* gm = grinAt(ray.o);
        Hit hs = scene.closestHit(ray);
        double dS = hs.valid ? hs.t : 1e30;
        if (!gm) {
            // Outside any GRIN region: jump straight to the nearest GRIN entry lying
            // before the next surface, else stop marching (straight body takes over).
            double bestTa = 1e30; const Medium* bestM = nullptr;
            for (const auto& md : scene.media) {
                if (!(md.enabled && md.grin())) continue;
                double ta, tb;
                if (md.clipToBounds(ray.o, ray.d, 1e-4, dS, ta, tb) && ta < bestTa) {
                    bestTa = ta; bestM = &md;
                }
            }
            if (!bestM) break;                                   // no GRIN ahead
            ray = Ray{ray.o + ray.d * (bestTa + 1e-4), ray.d};   // nudge inside
            continue;
        }
        double ds = gm->iorStep;
        if (hs.valid && hs.t <= ds) break;                       // surface within a step
        // Symplectic Eikonal step with optical direction T = n·d (|T| = n):
        //   T += ∇n · ds ;  x += (T/n)·ds ;  d = T/|T|.
        double n0 = gm->nAt(ray.o);
        Vec3 T = ray.d * n0 + gm->gradNAt(ray.o, 0.5 * ds) * ds;
        Vec3 newPos = ray.o + (T / n0) * ds;
        double tl = std::sqrt(dot(T, T));
        Vec3 newDir = (tl > 1e-12) ? T * (1.0 / tl) : ray.d;
        ray = Ray{newPos, newDir};
    }
}

}  // namespace grin
