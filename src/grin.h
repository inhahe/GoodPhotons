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
//
// `camHide` marks this march as a CAMERA (primary) ray, and is threaded straight into the
// internal closestHit that bounds each step: a `hide_camera` surface must not truncate a
// camera march any more than it may stop a straight camera ray, or a hidden flat sitting
// inside/behind a GRIN region would freeze the bending at its own depth and the caller's
// own skipCamHidden trace would then resume from the wrong point with the wrong direction.
// See Material::hideCamera (scene.h).
//
// `info`, when given, reports what the march did for a CAMERA path's bookkeeping:
//   * `bent`: it stepped through a region that bends rays (Medium::bends). Light sampling's straight
//     connections never cross such a region (Scene::crossesGrin), so whatever the walk meets past
//     it was reachable by no light sample: the caller counts it as it counts an arrival through a
//     specular interface (GRIN-STRAIGHT-CONNECT).
//   * `radScale`: (n(x_k) / n(x_k+1))^2 over every Eikonal step, the field's own index at both ends,
//     so (n at entry / n at exit)^2 per region crossed. Radiance over n^2 is conserved along a ray
//     bending in a continuous field, and a region's boundary refracts nothing here (a ray crosses
//     it straight, keeping its geometric etendue, so its radiance), so a camera path that marched
//     owes this factor on the radiance it brings back (GRIN-RADIANCE-SCALE). A light path carries
//     flux, which the bend conserves, and takes none.
struct MarchInfo { bool bent = false; double radScale = 1.0; };

template <class SegFn>
inline bool marchSegments(const Scene& scene, Ray& ray, SegFn&& onSeg, bool camHide = false,
                          MarchInfo* info = nullptr) {
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
    // The field's index at the next step's start when the last step computed it there (`info`'s
    // factor), so a camera march evaluates each point's index once.
    const Medium* nNextM = nullptr;
    double nNext = 0.0;
    for (int gstep = 0; gstep < GRIN_MAX_STEPS; ++gstep) {
        const Medium* gm = grinAt(ray.o);
        Hit hs = scene.closestHit(ray, Scene::kTminEps, nullptr, /*skipHair=*/false, /*skipCamHidden=*/camHide);
        double dS = hs.valid ? hs.t : 1e30;
        if (!gm) {
            nNextM = nullptr;   // a jump: the carried index belongs to a point behind
            // Outside any GRIN region: jump straight to the nearest GRIN entry lying
            // before the next surface, else stop marching (straight body takes over).
            double bestTa = 1e30; const Medium* bestM = nullptr;
            for (const auto& md : scene.media) {
                if (!(md.enabled && md.grin())) continue;
                double ta, tb;
                if (md.clipToBounds(ray.o, ray.d, 1e-4 * scene.lenScale, dS, ta, tb) && ta < bestTa) {
                    bestTa = ta; bestM = &md;
                }
            }
            if (!bestM) return false;                            // no GRIN ahead
            const double adv = bestTa + 1e-4 * scene.lenScale;   // nudge inside
            double tStop = 0.0;
            if (onSeg(ray.o, ray.d, adv, tStop)) {
                ray = Ray{ray.o + ray.d * tStop, ray.d};
                return true;
            }
            ray = Ray{ray.o + ray.d * adv, ray.d};
            continue;
        }
        double ds = gm->iorStep;
        if (hs.valid && hs.t <= ds) {                            // surface within a step
            // The straight remainder to it lies inside the region: what the walk meets there no light
            // sample reached either (a light sample from inside counts blocked), and the field's index
            // changes along it.
            if (info) {
                const Vec3 pe = ray.o + ray.d * hs.t;
                if (gm->insideBound(pe, &tabs)) {
                    const double n0 = gm->nAt(ray.o, &tabs), n1 = gm->nAt(pe, &tabs);
                    if (n0 > 0.0 && n1 > 0.0) info->radScale *= (n0 / n1) * (n0 / n1);
                }
                info->bent = info->bent || gm->bends();
            }
            return false;
        }
        // Symplectic Eikonal step with optical direction T = n·d (|T| = n):
        //   T += ∇n · ds ;  x += (T/n)·ds ;  d = T/|T|.
        double n0 = (nNextM == gm) ? nNext : gm->nAt(ray.o, &tabs);
        nNextM = nullptr;
        Vec3 T = ray.d * n0 + gm->gradNAt(ray.o, 0.5 * ds, &tabs) * ds;
        Vec3 disp = (T / n0) * ds;                    // the displacement, formed exactly as
        Vec3 newPos = ray.o + disp;                   // before so the geometry is unchanged
        double tl = std::sqrt(dot(T, T));
        Vec3 newDir = (tl > 1e-12) ? T * (1.0 / tl) : ray.d;
        // `disp` is parallel to T (n0 > 0, ds > 0), so newDir IS the travel direction of
        // this sub-segment — no separate normalisation needed.
        double segLen = std::sqrt(dot(disp, disp));
        // The step's factor to its end is in `info` BEFORE the caller sees the step, which it may gather
        // along (mode M's beams); one ended short by the caller takes the factor to where it ended.
        // The factor is taken only to a point inside the region's bound: past it the field need not be
        // defined (an index that is no index there would blow the factor up), and the boundary itself
        // refracts nothing, so the step that leaves the region takes none.
        const double scaleIn = info ? info->radScale : 1.0;
        if (info) {
            if (gm->insideBound(newPos, &tabs)) {
                const double n1 = gm->nAt(newPos, &tabs);
                if (n1 > 0.0) info->radScale *= (n0 / n1) * (n0 / n1);
                nNextM = gm; nNext = n1;
            }
            info->bent = info->bent || gm->bends();
        }
        double tStop = 0.0;
        if (segLen > 0.0 && onSeg(ray.o, newDir, segLen, tStop)) {
            if (info) {
                const Vec3 pe = ray.o + newDir * tStop;
                info->radScale = scaleIn;
                if (gm->insideBound(pe, &tabs)) {
                    const double n1 = gm->nAt(pe, &tabs);
                    if (n1 > 0.0) info->radScale *= (n0 / n1) * (n0 / n1);
                }
            }
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
inline void march(const Scene& scene, Ray& ray, bool camHide = false) {
    marchSegments(scene, ray,
                  [](const Vec3&, const Vec3&, double, double&) { return false; }, camHide);
}

}  // namespace grin
