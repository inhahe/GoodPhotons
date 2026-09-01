// Gradient-index (GRIN) ray marching — shared by every CPU transport path.
//
// A medium carrying an `ior` field n(x,y,z) is a GRADIENT-INDEX region: rays do NOT
// travel straight through it, they bend continuously obeying the Eikonal ray equation
//   d/ds( n · dr/ds ) = ∇n.
// This header holds the one canonical marcher so the forward light tracer (modes A/B/C),
// the backward path tracer (mode D reference) and the bidirectional tracer all bend rays
// through GRIN regions IDENTICALLY — a single source of truth instead of three copies.
//
// Semantics:
//   * Marching does NOT consume a bounce: it is a pre-pass run before each bounce's
//     closestHit. When the ray reaches a surface (within one step) or leaves every GRIN
//     region, we stop and let the straight-ray body take over on the short remaining leg.
//   * `sceneHasGrin` gates the whole thing: `ior`-free scenes never call the marcher, so
//     they stay BIT-IDENTICAL to the pre-GRIN renderer (zero added cost).
//   * PARTICIPATING MEDIA ALONG THE CURVE are integrated by the caller, one straight
//     sub-segment at a time, through `marchSegments`'s hook (see below). Until 0.198.0
//     they were not integrated at all: `march` advanced the ray geometrically over the
//     whole curved span and each tracer then sampled media only on the SHORT STRAIGHT
//     REMAINDER, so a medium that both scatters and carries `ior` lost essentially all of
//     its scattering. That was documented as "approximated"; measurement showed it was
//     closer to "deleted" (a dense haze rendered 22% dark through its own centre and
//     vanished visually the moment a *constant* `ior "1.0"` — which bends nothing —
//     merely routed it through the marcher). The hook is the fix.
//
// Why a per-sub-segment hook instead of integrating media in here: the media samplers are
// Renderer members (render.h) and the tracers differ in what they want along the curve
// (a collision + phase scatter, or a transmittance product, or both). Handing back each
// straight piece keeps grin.h dependent only on scene.h while letting every tracer reuse
// its own, already-correct straight-segment media code. The decomposition is EXACT, not an
// approximation: a spatial Poisson process with rate sigma_t(x) is Markov in arc length, so
// the first-collision distribution along a polyline is exactly "sample within segment 1;
// failing that, resample afresh within segment 2; ...", and transmittance along a polyline
// is exactly the product of the per-segment transmittances.
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
// stopping when a surface is within one step or the ray has left all GRIN regions.
//
// `onSeg(o, d, len, tStop) -> bool` is called for EVERY straight sub-segment the marched
// path traverses, in travel order — both the Eikonal steps inside a region and the straight
// jumps between regions (which are equally part of the span the caller would otherwise
// never integrate). It returns true to TERMINATE the march at distance `tStop` along that
// sub-segment, which is how a caller reports "my medium scattered here"; `ray` is then left
// at that point with the local travel direction and `marchSegments` returns true.
//
// The sub-segments are exactly the polyline the ray actually walks: `len` is the true
// geometric displacement of the step (|(T/n)·ds|, which is `ds` only to first order), and
// `d` is the direction of that displacement. Sum of `len` over the call sequence is the
// arc length of the curve, which is what a caller needs for e.g. Beer-Lambert inside glass.
//
// A no-op when the ray is nowhere near a GRIN region. Callers gate on sceneHasGrin() so
// this is never entered for ordinary scenes.
template <class SegFn>
inline bool marchSegments(const Scene& scene, Ray& ray, SegFn&& onSeg) {
    constexpr int GRIN_MAX_STEPS = 200000;   // safety cap on marching steps
    // Sampled tables, so an `ior` field (or the bounding field selecting the region) can
    // read a MEASURED index volume — `ior "grin:n(x, y, z)"` — not just a formula.
    const PatTables tabs = scene.patTables();
    // The GRIN region containing a point (highest-priority membership), or null.
    auto grinAt = [&](const Vec3& p) -> const Medium* {
        for (const auto& md : scene.media)
            if (md.enabled && md.grin() && md.insideBound(p, &tabs)) return &md;
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
            if (!bestM) return false;                            // no GRIN ahead
            const double adv = bestTa + 1e-4;                    // nudge inside
            double tStop = 0.0;
            if (onSeg(ray.o, ray.d, adv, tStop)) {
                ray = Ray{ray.o + ray.d * tStop, ray.d};
                return true;
            }
            ray = Ray{ray.o + ray.d * adv, ray.d};
            continue;
        }
        double ds = gm->iorStep;
        if (hs.valid && hs.t <= ds) return false;                // surface within a step
        // Symplectic Eikonal step with optical direction T = n·d (|T| = n):
        //   T += ∇n · ds ;  x += (T/n)·ds ;  d = T/|T|.
        double n0 = gm->nAt(ray.o, &tabs);
        Vec3 T = ray.d * n0 + gm->gradNAt(ray.o, 0.5 * ds, &tabs) * ds;
        Vec3 disp = (T / n0) * ds;                    // the displacement, formed exactly as
        Vec3 newPos = ray.o + disp;                   // before so the geometry is unchanged
        double tl = std::sqrt(dot(T, T));
        Vec3 newDir = (tl > 1e-12) ? T * (1.0 / tl) : ray.d;
        // `disp` is parallel to T (n0 > 0, ds > 0), so newDir IS the travel direction of
        // this sub-segment — no separate normalisation needed.
        double segLen = std::sqrt(dot(disp, disp));
        double tStop = 0.0;
        if (segLen > 0.0 && onSeg(ray.o, newDir, segLen, tStop)) {
            ray = Ray{ray.o + newDir * tStop, newDir};
            return true;
        }
        ray = Ray{newPos, newDir};
    }
    return false;
}

// Geometry-only march: bend, integrate nothing. BIT-IDENTICAL to the pre-hook marcher — the
// hook never fires, so every float operation is the one above; that is the invariant this
// wrapper exists to state. It has no in-tree caller at the moment (every current caller wants
// at least the arc length back, so they pass a hook), but it is the honest spelling of "bend
// only" and the device side keeps the same shape in `dGrinMarch`'s defaulted `med` parameter.
inline void march(const Scene& scene, Ray& ray) {
    marchSegments(scene, ray,
                  [](const Vec3&, const Vec3&, double, double&) { return false; });
}

}  // namespace grin
