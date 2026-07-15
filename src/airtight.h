// -check-airtight: Monte-Carlo ray-parity audit of an isosurface's *marched* field.
//
// Unlike -check-watertight (watertight.h), which polygonises the field with marching
// tetrahedra and edge-manifold-checks that mesh, this audit probes the EXACT field the
// renderer sphere-traces at render time — the same intersectImplicit() the camera rays
// use. That matters because an isosurface is never meshed for rendering; its real
// airtightness is a property of the field's zero level-set { f = 0 } clipped to its
// container, not of any polygonisation. Marching cubes/tetrahedra can miss a leak, a
// thin wall, or a spike narrower than its grid cell, so a mesh proxy is only an
// indicator. This tool has no such resolution blind spot at the marcher's scale.
//
// Method (the ray-parity test): fire many random chords that start and END OUTSIDE the
// container (so both endpoints are unambiguously outside the solid, since the solid is
// contained). For a CLOSED, airtight solid every such chord crosses the boundary an
// EVEN number of times — each entry into the solid has a matching exit. We count the
// boundary crossings the renderer's own marcher reports along the chord:
//   * ODD count  => the chord went from outside to inside and never came back out
//                   (or vice-versa) => the boundary is not closed => LEAK. For an
//                   `open` (uncapped) surface this is the solid poking through a
//                   container wall (an open cap); for a `capped` surface it means the
//                   marcher SKIPPED a crossing (Lipschitz/max_gradient overshoot or a
//                   feature thinner than the march step) — a real light leak at render.
//   * EVEN count => consistent.
// A parallel dense reference sampling of f (finer than the marcher's step) counts the
// true field crossings; where the marcher finds FEWER, it is overshooting/missing thin
// features even if parity happens to stay even (two missed crossings). And for open
// surfaces we directly sample the container boundary for interior (f<0) area, which is
// the definitive open-cap signature.
//
// Read-only and non-destructive. Reuses implicit.h's evaluator + intersectImplicit.
#pragma once
#include "implicit.h"
#include <random>
#include <cmath>
#include <cstdint>

namespace airtight {

// Write component `a` (0=x,1=y,2=z) of a Vec3 (the assignment twin of bvh.h's vget).
inline void vset(Vec3& v, int a, double val) { (a == 0 ? v.x : (a == 1 ? v.y : v.z)) = val; }

struct Report {
    long long chords          = 0;   // exterior->exterior chords tested
    long long oddParity       = 0;   // chords with an ODD marcher-hit count (leak signal)
    long long overshoot       = 0;   // chords where the marcher found fewer crossings than the dense reference
    bool      open            = false; // capped == false (an open surface can have open caps)
    long long boundarySamples = 0;   // container-boundary points sampled (open surfaces only)
    long long boundaryInside  = 0;   // ...of those, how many sit inside the solid (f < 0) = open-cap area
    double    worstMinF       = 0.0; // most-negative f seen on the container boundary
    bool      degenerate      = false; // container had zero extent / no valid chords

    // Airtight iff no odd-parity chords and (for open surfaces) no interior boundary area.
    bool airtight() const {
        return !degenerate && oddParity == 0 && (!open || boundaryInside == 0);
    }
    double oddFrac()   const { return chords ? (double)oddParity / chords : 0.0; }
    double overFrac()  const { return chords ? (double)overshoot / chords : 0.0; }
    double capFrac()   const { return boundarySamples ? (double)boundaryInside / boundarySamples : 0.0; }
};

// [tIn,tOut] where ray o + t*d is inside the container (box `bounds` or a sphere).
// Mirrors the clip in intersectImplicit. Returns false if the chord misses the container.
inline bool containerSpan(const Implicit& im, const Vec3& o, const Vec3& d,
                          double& tIn, double& tOut) {
    if (im.container == Container::Sphere) {
        Vec3   oc = o - im.sphereCenter;
        double A  = dot(d, d);
        double B  = dot(oc, d);
        double C  = dot(oc, oc) - im.sphereRadius * im.sphereRadius;
        double disc = B * B - A * C;
        if (disc < 0.0) return false;
        double sq = std::sqrt(disc);
        tIn  = (-B - sq) / A;
        tOut = (-B + sq) / A;
        return tOut > tIn;
    }
    Vec3 invD{1.0 / d.x, 1.0 / d.y, 1.0 / d.z};
    tIn = -1e300; tOut = 1e300;
    for (int a = 0; a < 3; ++a) {
        double oo = vget(o, a), id = vget(invD, a);
        double tLo = (vget(im.bounds.lo, a) - oo) * id;
        double tHi = (vget(im.bounds.hi, a) - oo) * id;
        double tnear = id >= 0.0 ? tLo : tHi;
        double tfar  = id >= 0.0 ? tHi : tLo;
        if (tnear > tIn)  tIn  = tnear;
        if (tfar  < tOut) tOut = tfar;
        if (tOut < tIn) return false;
    }
    return tOut > tIn;
}

// Count the boundary crossings the renderer's marcher reports along the chord o+t*d,
// t in [0,1]. Re-invokes intersectImplicit, advancing past each hit, exactly as a
// path tracer would when marching through the object. Caps count (they are real
// boundary faces of the marched solid).
inline int marchHitCount(const Implicit& im, const Vec3& o, const Vec3& d, int maxHits) {
    Ray r{o, d};
    double tmin = 0.0;
    int hits = 0;
    for (int i = 0; i <= maxHits; ++i) {
        Hit h;             // h.t = DBL_MAX; clip to the chord end (t = 1)
        h.t = 1.0; h.valid = false;
        if (!intersectImplicit(r, im, tmin, h)) break;
        ++hits;
        double nt = h.t + 1e-7;          // nudge past the refined root (parametric)
        if (nt <= tmin) nt = tmin + 1e-7;
        tmin = nt;
        if (tmin >= 1.0) break;
    }
    return hits;
}

// Dense reference: sign-change count of f within [tIn,tOut], sampled finer than the
// marcher's step so it catches thin features the marcher may overshoot.
inline int refFieldCrossings(const Implicit& im, const Vec3& o, const Vec3& d,
                             double tIn, double tOut, int samples) {
    if (samples < 2) samples = 2;
    double prev = im.eval(o + d * tIn);
    int cross = 0;
    for (int i = 1; i <= samples; ++i) {
        double t = tIn + (tOut - tIn) * ((double)i / samples);
        double f = im.eval(o + d * t);
        if ((prev > 0.0 && f <= 0.0) || (prev < 0.0 && f >= 0.0) ||
            (prev == 0.0 && f != 0.0))
            ++cross;
        prev = f;
    }
    return cross;
}

// Sample the container BOUNDARY for interior (f<0) area. For an `open` surface any
// interior point on the wall is an open cap (a hole); this is the definitive signature.
inline void sampleBoundary(const Implicit& im, Report& rep, int grid) {
    auto probe = [&](const Vec3& p) {
        double f = im.eval(p);
        ++rep.boundarySamples;
        if (f < 0.0) { ++rep.boundaryInside; if (f < rep.worstMinF) rep.worstMinF = f; }
    };
    if (im.container == Container::Sphere) {
        // Fibonacci-sphere points on the container surface.
        int N = std::max(64, grid * grid * 6);
        const double GA = 3.399186938691976; // golden angle
        for (int i = 0; i < N; ++i) {
            double y = 1.0 - 2.0 * (i + 0.5) / N;
            double r = std::sqrt(std::max(0.0, 1.0 - y * y));
            double ph = GA * i;
            Vec3 n{r * std::cos(ph), y, r * std::sin(ph)};
            probe(im.sphereCenter + n * im.sphereRadius);
        }
        return;
    }
    // Box: a grid on each of the six faces.
    Vec3 lo = im.bounds.lo, hi = im.bounds.hi, ext = hi - lo;
    for (int axis = 0; axis < 3; ++axis) {
        int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
        for (int side = 0; side < 2; ++side) {
            double face = side ? vget(hi, axis) : vget(lo, axis);
            for (int i = 0; i <= grid; ++i)
            for (int j = 0; j <= grid; ++j) {
                double u = vget(lo, a1) + vget(ext, a1) * (i / (double)grid);
                double v = vget(lo, a2) + vget(ext, a2) * (j / (double)grid);
                Vec3 p{0, 0, 0};
                vset(p, axis, face); vset(p, a1, u); vset(p, a2, v);
                probe(p);
            }
        }
    }
}

// Run the audit on one Implicit. nChords random exterior->exterior chords through the
// container's bounding sphere; parity + overshoot + open-cap stats.
inline Report check(const Implicit& im, long long nChords, unsigned seed) {
    Report rep;
    rep.open = !im.capped;

    // Bounding sphere that encloses the container (chords launch from its surface, so
    // both endpoints are guaranteed outside the container -> outside the solid).
    Vec3 center; double encR;
    if (im.container == Container::Sphere) {
        center = im.sphereCenter; encR = im.sphereRadius * 1.05 + 1e-4;
    } else {
        center = (im.bounds.lo + im.bounds.hi) * 0.5;
        encR = 0.5 * length(im.bounds.hi - im.bounds.lo) * 1.05 + 1e-4;
    }
    if (!(encR > 0.0) || !std::isfinite(encR) || encR > 1e17) { rep.degenerate = true; return rep; }

    // Reference-sampling resolution: finer than the marcher's step so it can catch a
    // feature the marcher would skip. Cap for cost.
    double span = 2.0 * encR;
    double refStep = (im.minStep > 0.0 ? im.minStep : 1e-4) * 0.125;
    int refSamples = (int)std::min(8000.0, std::max(400.0, span / refStep));
    int maxHits = 4096;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto onSphere = [&]() {
        double z = 1.0 - 2.0 * U(rng);
        double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        double ph = 6.283185307179586 * U(rng);
        return Vec3{r * std::cos(ph), z, r * std::sin(ph)};
    };

    for (long long c = 0; c < nChords; ++c) {
        Vec3 p0 = center + onSphere() * encR;
        Vec3 p1 = center + onSphere() * encR;
        Vec3 d  = p1 - p0;
        if (dot(d, d) < 1e-20) continue;
        double tIn, tOut;
        if (!containerSpan(im, p0, d, tIn, tOut)) continue;  // chord misses the container
        tIn  = std::max(0.0, tIn);
        tOut = std::min(1.0, tOut);
        if (tOut <= tIn) continue;
        ++rep.chords;

        int hits = marchHitCount(im, p0, d, maxHits);
        if (hits & 1) ++rep.oddParity;

        int refCross = refFieldCrossings(im, p0, d, tIn, tOut, refSamples);
        // The marcher's field-crossing tally excludes caps; compare like-for-like by
        // only flagging overshoot when it finds strictly fewer than the reference.
        if (hits < refCross) ++rep.overshoot;
    }

    if (rep.open) sampleBoundary(im, rep, std::max(8, 32));
    return rep;
}

}  // namespace airtight
