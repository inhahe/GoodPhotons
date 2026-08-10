// Curve / fiber primitive — hair, fur, grass, wire, thread.  (TODO.md §P1)
//
// WHY THIS EXISTS.  Until this file, ftrace's primitive set was triangles + analytic
// spheres + implicit fields + instanced meshes, so a strand of hair had to be a
// triangle ribbon: ~8-32 segments x 2 tris x 2 (a ribbon has two sides) per hair, i.e.
// 10^8-10^9 triangles for one furred animal.  That is not a memory problem you optimise
// your way out of — it is the wrong representation.  A fiber is a *curve with a radius*,
// and the whole point of a native primitive is that one strand costs a handful of
// segments regardless of how it is shaded.
//
// THE REPRESENTATION.  A `Curve` is authored as control points + radii under one of four
// bases (linear / Catmull-Rom / Bezier / B-spline), and is FLATTENED AT LOAD TIME into a
// chain of `CurveSeg` — each one a *round cone*, the convex hull of a sphere at each end
// (a capsule with linearly varying radius).  The BVH leaf is one round cone.
//
// Flattening at load rather than evaluating the basis per ray is deliberate:
//   * the leaf bound is then EXACT for the thing actually intersected (bound what you
//     test, not a superset of it) — which matters enormously for a BVH over millions of
//     long, thin, wildly anisotropic prims, the failure mode TODO §P1 called out;
//   * adjacent round cones share the end sphere, so the union of a chain is watertight
//     and smooth at the joints with no cap/mitre logic at all;
//   * the per-ray inner loop becomes a closed-form quadratic with no basis evaluation,
//     no recursive subdivision and no iteration — which is what the sub-pixel variance
//     of §P2 will be paying for millions of times;
//   * a `CurveSeg` is a flat POD, i.e. exactly the shape a future GPU upload wants.
// The cost is memory (see the note on CurveSeg) and that authored curvature is
// discretised — invisible at fiber widths, and `segments` buys back as much as wanted.
//
// ROUND cones, not camera-facing ribbons: a ribbon needs a camera, which the *forward*
// modes structurally do not have, and a round cross-section is what a Marschner/Yan
// fiber BCSDF (§P3) is derived against.
#pragma once
#include <vector>
#include <string>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include "linalg.h"
#include "geometry.h"
#include "bvh.h"

// ---------------------------------------------------------------------------
// Authoring bases
// ---------------------------------------------------------------------------
// How the authored control points are read.  All four are flattened to the same
// round-cone chain; they differ only in what curve the points describe.
enum class CurveBasis {
    Linear,       // the points ARE the polyline (exact; `segments` is forced to 1)
    CatmullRom,   // uniform Catmull-Rom: INTERPOLATES every point (the hair-guide default)
    Bezier,       // cubic Bezier chain; needs 3k+1 points (P0 C0 C1 P1 C2 C3 P2 ...)
    BSpline       // uniform cubic B-spline: approximating, C2, needs >= 4 points
};

inline const char* curveBasisName(CurveBasis b) {
    switch (b) {
        case CurveBasis::Linear:     return "linear";
        case CurveBasis::CatmullRom: return "catmull_rom";
        case CurveBasis::Bezier:     return "bezier";
        default:                     return "bspline";
    }
}

// Number of cubic spans an n-point control polygon yields under `b`, or 0 if the
// point count is too small / not admissible for the basis.
inline int curveSpanCount(CurveBasis b, int n) {
    switch (b) {
        case CurveBasis::Linear:     return n >= 2 ? n - 1 : 0;
        case CurveBasis::CatmullRom: return n >= 2 ? n - 1 : 0;   // ends are clamp-duplicated
        case CurveBasis::Bezier:     return (n >= 4 && (n - 1) % 3 == 0) ? (n - 1) / 3 : 0;
        case CurveBasis::BSpline:    return n >= 4 ? n - 3 : 0;
        default:                     return 0;
    }
}

// ---------------------------------------------------------------------------
// The primitive.
// ---------------------------------------------------------------------------
// One round cone = the convex hull of sphere(p0,r0) and sphere(p1,r1).  This is the
// BVH leaf and the unit of intersection.
//
// MEMORY (relevant once §P2's fur counts arrive): 80 bytes each, so 1M strands x 16
// segments = 16M segs = ~1.28 GB.  Fine for the hundreds-to-thousands of strands this
// primitive is authored with today, and explicitly a thing to shrink (float positions,
// per-curve matId, a 16-bit u) when the aggregate-BSDF LOD work makes those counts real.
struct CurveSeg {
    Vec3   p0, p1;                  // world-space segment endpoints
    double r0 = 0.001, r1 = 0.001;  // world radius at each end
    int    matId = 0;
    int    curveId = -1;            // owning strand (diagnostics; future per-strand data)
    float  u0 = 0.0f, u1 = 1.0f;    // curve parameter at p0/p1 — the hit's texture `u`
};

// One authored strand: a contiguous run of the scene's flat CurveSeg pool.
struct Curve {
    int         firstSeg = 0;
    int         segCount = 0;
    int         matId = 0;
    std::string name;               // authored ftsl block name (diagnostics)
    CurveBasis  basis = CurveBasis::CatmullRom;
};

// Exact world bound of one round cone: the union of its two end spheres' boxes.  The
// hull of two balls is contained in the union of their bounding boxes' hull, and every
// point tested by intersectCurveSeg lies in that hull — so this is tight, not merely
// conservative, up to the box-vs-sphere slack the BVH has for spheres anyway.
inline Aabb curveSegBounds(const CurveSeg& s) {
    Aabb b;
    b.expand(s.p0 - Vec3{s.r0, s.r0, s.r0}); b.expand(s.p0 + Vec3{s.r0, s.r0, s.r0});
    b.expand(s.p1 - Vec3{s.r1, s.r1, s.r1}); b.expand(s.p1 + Vec3{s.r1, s.r1, s.r1});
    return b;
}

// ---------------------------------------------------------------------------
// Per-ray setup, hoisted out of the BVH leaf loop (cf. TriShear for triangles).
// ---------------------------------------------------------------------------
// The round-cone algebra below is written for a UNIT direction, but a Scene ray's `d`
// need not be unit (intersectSphere / intersectImplicit both say so).  Normalising once
// per ray instead of once per segment is the whole reason this struct exists: a fur
// render tests thousands of segments per ray.
//
// `invLen` converts a unit-space parameter back to the ray's own parameterisation:
// p = o + t_unit*(d/|d|) = o + (t_unit/|d|)*d, so t = t_unit * invLen.
struct CurveRay {
    Vec3   dn{0, 0, 1};    // unit ray direction
    double invLen = 1.0;   // 1/|d|
};

inline CurveRay makeCurveRay(const Vec3& d) {
    CurveRay c;
    double l2 = dot(d, d);
    if (l2 > 0.0) {
        double l = std::sqrt(l2);
        c.invLen = 1.0 / l;
        c.dn = d * c.invLen;
    }
    return c;
}

// ---------------------------------------------------------------------------
// Ray / round-cone intersection.
// ---------------------------------------------------------------------------
// The hull of two balls has a boundary made of three pieces: the tangent LATERAL cone,
// and a spherical cap at each end.  Which piece a boundary point belongs to is decided
// by one axial coordinate,
//
//     y = dot(ba, p - p0) - r0*(r0 - r1),      band = (0, d2),  d2 = |ba|^2 - (r0-r1)^2
//
// (Inigo Quilez's round-cone parameterisation): y <= 0 is the p0 cap, y >= d2 the p1 cap,
// and in between the lateral surface.  Testing every root of all three quadrics and
// keeping only those that satisfy their own piece's y range therefore enumerates exactly
// the hull's boundary crossings — no root is missed and none is spurious, including for
// a ray whose origin is INSIDE the fiber (which is why both roots of each quadric are
// offered, not just the near one: a shadow ray leaving a strand must find the exit).
//
// Degenerate case: when one ball contains the other (d2 <= 0) the hull IS the larger
// ball and the lateral surface does not exist, so only that sphere is tested.
//
// ---------------------------------------------------------------------------
// ORIGIN RECENTERING — why the ray origin is slid before any quadric is formed.
// ---------------------------------------------------------------------------
// `curveSegCrossings` first slides the ray origin along itself to its closest approach
// to p0, and reports roots in that shifted parameterisation (the caller undoes the
// shift).  That is exact in exact arithmetic — it only re-parameterises the same ray —
// and is purely a CONDITIONING fix: the round-cone analogue of the perpendicular-offset
// trick Ray Tracing Gems ch.7 uses for spheres, which `intersectSphere` already uses for
// exactly the same reason.
//
// Without it, `k0 = d2*m5 - m1*m1 + ...` catastrophically cancels at FIBER scales: with a
// 1 mm radius, 1 cm segments and an origin 2 m away, `d2*m5` and `m1*m1` are both ~4e-4
// while their difference is O(m0*r0^2) ~ 1e-10 — a six-decade cancellation.  Double has
// the headroom to absorb that (which is why it was invisible at first), but the CUDA
// megakernel is fp32 by default (`using Real = float`), where six decades is the entire
// mantissa.  Measured on 109 k rays aimed at exactly that configuration, the naive form
// loses **16 016 hits outright (15%)** and errs by up to **11.8 radii** — it renders a
// strand as speckled holes and misplaced surface, not as a slightly noisy strand.
// Recentered: 4 grazing misses (0.004%), max error 0.008 radii.  At 0.5 mm radius / 2 cm
// segments the naive form is worse still (31 837 misses, 42 radii); recentered stays at
// 0.036 radii.  After recentering every term is O(r^2) with no large common part to
// cancel away.
//
// The host uses the same formulation as the device deliberately: it costs two dot
// products, it buys the CPU the same robustness at large world coordinates (double merely
// postpones the identical failure to scenes ~1e9x bigger), and it means `-checkcurve`
// validates the exact algebra the GPU runs rather than a cousin of it.
//
// ---------------------------------------------------------------------------
// Root enumeration, generic over the scalar type.
// ---------------------------------------------------------------------------
// Templated on `S` so the SAME algebra can be instantiated at double (the renderer) and
// at float (`-checkcurve` §6, which measures the conditioning the fp32 device twin in
// render_cuda.cu depends on).  One copy means the guard cannot drift away from the code
// it guards — the failure above was found precisely because nothing tested the float
// conditioning of the double-only original.
//
// Geometry comes in as plain scalar triples rather than Vec3 so a float instantiation
// does its dot products in float too; that is the thing being measured.  `offer(tn, piece,
// y)` is called for every boundary crossing, in the SHIFTED parameterisation; the caller
// applies tmin / running-closest filtering because only it knows those units.  `shift`
// receives the amount the origin moved.  Returns false for a degenerate (zero-length)
// segment, which has nothing to sweep.
template <class S, class OfferFn>
inline bool curveSegCrossings(const S ro[3], const S rd[3],
                              const S p0[3], const S p1[3], S r0, S r1,
                              S& shift, OfferFn&& offer) {
    const S ba[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    auto d3 = [](const S a[3], const S b[3]) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; };

    const S m0 = d3(ba, ba);
    if (!(m0 > S(0))) return false;          // zero-length segment: nothing to sweep

    // Slide the origin to its closest approach to p0 (rd is unit, so this is a dot).
    const S toP0[3] = {p0[0] - ro[0], p0[1] - ro[1], p0[2] - ro[2]};
    shift = d3(toP0, rd);
    const S o[3] = {ro[0] + rd[0]*shift, ro[1] + rd[1]*shift, ro[2] + rd[2]*shift};
    const S oa[3] = {o[0] - p0[0], o[1] - p0[1], o[2] - p0[2]};
    const S ob[3] = {o[0] - p1[0], o[1] - p1[1], o[2] - p1[2]};

    const S rr = r0 - r1;
    const S m1 = d3(ba, oa);
    const S m2 = d3(ba, rd);
    const S m3 = d3(rd, oa);
    const S m5 = d3(oa, oa);
    const S m6 = d3(ob, rd);
    const S m7 = d3(ob, ob);
    const S d2 = m0 - rr * rr;

    if (d2 > S(0)) {
        // --- lateral (tangent) cone -----------------------------------------------
        const S k2 = d2 - m2 * m2;
        const S k1 = d2 * m3 - m1 * m2 + m2 * rr * r0;
        const S k0 = d2 * m5 - m1 * m1 + m1 * rr * r0 * S(2) - m0 * r0 * r0;
        const S h  = k1 * k1 - k0 * k2;
        // k2 == 0 is the ray running exactly along the axis; the quadratic degenerates
        // to a line and the crossing it would report is a cap crossing, which the two
        // sphere tests below find anyway.
        if (h >= S(0) && k2 != S(0)) {
            const S sq = std::sqrt(h);
            S ta = (-sq - k1) / k2, tb = (sq - k1) / k2;
            if (ta > tb) { S tmp = ta; ta = tb; tb = tmp; }
            const S base = m1 - r0 * rr;
            const S ya = base + ta * m2, yb = base + tb * m2;
            if (ya > S(0) && ya < d2) offer(ta, 0, ya);
            if (yb > S(0) && yb < d2) offer(tb, 0, yb);
        }
        // --- cap at p0 (the hull boundary only where y <= 0) ------------------------
        const S h1 = m3 * m3 - m5 + r0 * r0;
        if (h1 > S(0)) {
            const S q = std::sqrt(h1), base = m1 - r0 * rr;
            const S ta = -m3 - q, tb = -m3 + q;
            if (base + ta * m2 <= S(0)) offer(ta, 1, S(0));
            if (base + tb * m2 <= S(0)) offer(tb, 1, S(0));
        }
        // --- cap at p1 (the hull boundary only where y >= d2) -----------------------
        const S h2 = m6 * m6 - m7 + r1 * r1;
        if (h2 > S(0)) {
            const S q = std::sqrt(h2), base = m1 - r0 * rr;
            const S ta = -m6 - q, tb = -m6 + q;
            if (base + ta * m2 >= d2) offer(ta, 2, d2);
            if (base + tb * m2 >= d2) offer(tb, 2, d2);
        }
    } else {
        // One ball swallows the other: the hull is just the bigger sphere.
        const bool useP0 = (r0 >= r1);
        const S    rc    = useP0 ? r0 : r1;
        const S    mm    = useP0 ? m3 : m6;
        const S    mq    = useP0 ? m5 : m7;
        const S    hh    = mm * mm - mq + rc * rc;
        if (hh > S(0)) {
            const S q = std::sqrt(hh);
            offer(-mm - q, useP0 ? 1 : 2, useP0 ? S(0) : d2);
            offer(-mm + q, useP0 ? 1 : 2, useP0 ? S(0) : d2);
        }
    }
    return true;
}

// Writes into `hit` respecting hit.t as the running closest, and returns true on a
// nearer hit — the same contract as intersectTri / intersectSphere / intersectImplicit.
// `anyHit` (occlusion queries) skips the normal / UV / tangent work, which cannot change
// the boolean, so the two paths agree by construction.
inline bool intersectCurveSeg(const CurveRay& cr, const Ray& r, const CurveSeg& s,
                              double tmin, Hit& hit, bool anyHit = false) {
    const Vec3&  rd = cr.dn;
    const Vec3   ba = s.p1 - s.p0;
    const double m0 = dot(ba, ba);

    double bestTn = DBL_MAX;      // best root, in RECENTERED unit-direction parameter
    int    piece  = -1;           // 0 = lateral, 1 = cap at p0, 2 = cap at p1
    double bestY  = 0.0;          // axial coordinate at the winner (lateral -> u)
    double shift  = 0.0;

    // Roots arrive in the RECENTERED parameterisation, where the near root of a strand the
    // ray actually hits is typically NEGATIVE (the origin now sits beside the fiber, not
    // metres in front of it). So the shift must be undone BEFORE the tmin / running-closest
    // tests — testing `tn` itself against tmin would discard exactly the root wanted.
    const double roA[3] = {r.o.x, r.o.y, r.o.z}, rdA[3] = {rd.x, rd.y, rd.z};
    const double p0A[3] = {s.p0.x, s.p0.y, s.p0.z}, p1A[3] = {s.p1.x, s.p1.y, s.p1.z};
    if (!curveSegCrossings<double>(roA, rdA, p0A, p1A, s.r0, s.r1, shift,
            [&](double tn, int w, double y) {
                if (tn >= bestTn) return;
                const double t = (tn + shift) * cr.invLen;
                if (t < tmin || t >= hit.t) return;
                bestTn = tn; piece = w; bestY = y;
            }))
        return false;
    if (piece < 0) return false;

    const double t = (bestTn + shift) * cr.invLen;
    hit.t = t;
    hit.valid = true;
    if (anyHit) return true;

    // Rebuild the recentered offset vectors for the normal. Only reached on an accepted
    // non-anyHit hit, so the occlusion path (the majority of a fur render's rays) never
    // pays for them.
    const double rr = s.r0 - s.r1;
    const double d2 = m0 - rr * rr;
    const Vec3 ro = r.o + rd * shift;
    const Vec3 oa = ro - s.p0;
    const Vec3 ob = ro - s.p1;

    const Vec3 p = r.o + r.d * t;
    hit.p = p;
    hit.matId = s.matId;
    hit.sensorId = -1;
    hit.fieldVal = 0.0;

    // Outward geometric normal of the piece that was hit.
    Vec3 ng;
    if (piece == 0)      ng = (oa + rd * bestTn) * d2 - ba * bestY;   // lateral (iq)
    else if (piece == 1) ng = oa + rd * bestTn;                       // sphere at p0
    else                 ng = ob + rd * bestTn;                       // sphere at p1
    double nl = std::sqrt(dot(ng, ng));
    ng = (nl > 1e-18) ? ng * (1.0 / nl) : normalize(ba);
    hit.ng = ng;
    hit.n  = (dot(r.d, ng) < 0.0) ? ng : -ng;

    // `u` runs along the strand (0 at the root, 1 at the tip) and `v` around the
    // circumference — the parameterisation a strand texture and a fiber BCSDF both want.
    // The axial fraction is the tangent-point parameter y/d2, clamped so the two caps
    // read exactly the segment's endpoint values.
    double f = (d2 > 0.0) ? (bestY / d2) : 0.0;
    f = f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f);
    hit.u = (double)s.u0 + ((double)s.u1 - (double)s.u0) * f;

    const Vec3 axis = ba * (1.0 / std::sqrt(m0));
    Vec3 radial = (p - s.p0) - axis * dot(p - s.p0, axis);
    Vec3 T, B;
    onb(axis, T, B);
    // NOTE (v1 limitation, documented in REFERENCE.md): the azimuthal reference frame
    // comes from onb(axis) per SEGMENT rather than being parallel-transported along the
    // strand, so `v` can step at a joint where the tangent turns sharply. Invisible on a
    // fiber whose diameter is a fraction of a pixel; it would matter for a wide, visibly
    // textured tube, and the fix is a per-curve transported frame stored on CurveSeg.
    hit.v = 0.5 + std::atan2(dot(radial, B), dot(radial, T)) * (1.0 / (2.0 * PI));

    // Surface tangent = the fiber axis, Gram-Schmidt'd against the shading normal (for a
    // TAPERED cone the axis is not already perpendicular to it). This is both the
    // tangent-space-normal-map frame and the natural fiber direction for §P3.
    Vec3 tg = axis - hit.n * dot(hit.n, axis);
    double tl = std::sqrt(dot(tg, tg));
    hit.tangent = (tl > 1e-12) ? tg * (1.0 / tl) : T;
    hit.bitangentSign = 1.0;
    // Mean curvature (O3). A round cone is a surface of revolution whose principal
    // curvatures are 1/R around the axis and ~0 along it, so H = 1/(2R) at the local
    // (linearly interpolated) radius. Strand radii are tiny, so this is a LARGE number —
    // the honest answer, and exactly what lets `curv` tell fiber from body in a mixed
    // scene. Negated when we are looking at the inside wall, like every other primitive.
    double rLoc = (double)s.r0 + ((double)s.r1 - (double)s.r0) * f;
    double hSign = (dot(r.d, ng) < 0.0) ? 1.0 : -1.0;
    hit.curv = (rLoc > 1e-12) ? hSign * 0.5 / rLoc : 0.0;
    return true;
}

// ---------------------------------------------------------------------------
// Flattening: control points + basis -> a chain of round cones.
// ---------------------------------------------------------------------------
// `pts` / `radii` are the authored control points and per-point radii, ALREADY in world
// space (an affine map commutes with every basis here, so transforming the control
// points and then flattening is identical to flattening and then transforming).
// `subdiv` is the number of round cones per span; Linear ignores it (the polyline is
// already exact). Appends to `out` and returns the number of segments added, or 0 if the
// point count is not admissible for the basis.
//
// `u` is assigned uniformly in span index: a point at local parameter `lu` of span `sp`
// (of `S`) gets u = (sp + lu)/S. That is the right thing for the evenly-spaced control
// points a groom generator emits, and is documented as the parameterisation.
//
// Radius is interpolated LINEARLY between the span's two endpoint control points rather
// than through the basis: a Catmull-Rom radius can overshoot, and a negative radius is
// not a taper, it is a bug.
inline int tessellateCurve(const std::vector<Vec3>& pts,
                           const std::vector<double>& radii,
                           CurveBasis basis, int subdiv, int matId, int curveId,
                           std::vector<CurveSeg>& out) {
    const int n = (int)pts.size();
    const int S = curveSpanCount(basis, n);
    if (S <= 0) return 0;
    if (basis == CurveBasis::Linear) subdiv = 1;
    if (subdiv < 1) subdiv = 1;

    // Control points of span `sp` (P0..P3) and its two radius endpoints.
    auto spanPoints = [&](int sp, Vec3& P0, Vec3& P1, Vec3& P2, Vec3& P3,
                          double& ra, double& rb) {
        auto at = [&](int i) { return pts[i < 0 ? 0 : (i >= n ? n - 1 : i)]; };
        auto rat = [&](int i) { return radii[i < 0 ? 0 : (i >= n ? n - 1 : i)]; };
        switch (basis) {
            case CurveBasis::Linear:
            case CurveBasis::CatmullRom:
                P0 = at(sp - 1); P1 = at(sp); P2 = at(sp + 1); P3 = at(sp + 2);
                ra = rat(sp);    rb = rat(sp + 1);
                break;
            case CurveBasis::Bezier:
                P0 = at(3 * sp); P1 = at(3 * sp + 1); P2 = at(3 * sp + 2); P3 = at(3 * sp + 3);
                ra = rat(3 * sp); rb = rat(3 * sp + 3);
                break;
            default:  // BSpline
                P0 = at(sp); P1 = at(sp + 1); P2 = at(sp + 2); P3 = at(sp + 3);
                ra = rat(sp + 1); rb = rat(sp + 2);
                break;
        }
    };
    auto evalSpan = [&](const Vec3& P0, const Vec3& P1, const Vec3& P2, const Vec3& P3,
                        double u) -> Vec3 {
        const double u2 = u * u, u3 = u2 * u;
        switch (basis) {
            case CurveBasis::Linear:
                return P1 + (P2 - P1) * u;
            case CurveBasis::CatmullRom:
                // Uniform Catmull-Rom (tension 1/2): interpolates P1 at u=0, P2 at u=1.
                return (P1 * 2.0
                        + (P2 - P0) * u
                        + (P0 * 2.0 - P1 * 5.0 + P2 * 4.0 - P3) * u2
                        + (P1 * 3.0 - P0 - P2 * 3.0 + P3) * u3) * 0.5;
            case CurveBasis::Bezier: {
                const double w = 1.0 - u, w2 = w * w, w3 = w2 * w;
                return P0 * w3 + P1 * (3.0 * w2 * u) + P2 * (3.0 * w * u2) + P3 * u3;
            }
            default:  // uniform cubic B-spline
                return ((P0 * -1.0 + P1 * 3.0 - P2 * 3.0 + P3) * u3
                        + (P0 * 3.0 - P1 * 6.0 + P2 * 3.0) * u2
                        + (P2 - P0) * (3.0 * u)
                        + (P0 + P1 * 4.0 + P2)) * (1.0 / 6.0);
        }
    };

    const double invS = 1.0 / (double)S;
    int added = 0;
    for (int sp = 0; sp < S; ++sp) {
        Vec3 P0, P1, P2, P3; double ra = 0.0, rb = 0.0;
        spanPoints(sp, P0, P1, P2, P3, ra, rb);
        Vec3   prevP = evalSpan(P0, P1, P2, P3, 0.0);
        double prevR = ra;
        double prevU = (double)sp * invS;
        for (int k = 1; k <= subdiv; ++k) {
            const double lu = (double)k / (double)subdiv;
            Vec3   curP = evalSpan(P0, P1, P2, P3, lu);
            double curR = ra + (rb - ra) * lu;
            double curU = ((double)sp + lu) * invS;
            // Drop exactly-coincident samples: a zero-length round cone is degenerate
            // (m0 == 0) and its sphere is already contributed by its neighbours.
            Vec3 dp = curP - prevP;
            if (dot(dp, dp) > 0.0) {
                CurveSeg s;
                s.p0 = prevP; s.p1 = curP;
                s.r0 = prevR; s.r1 = curR;
                s.matId = matId; s.curveId = curveId;
                s.u0 = (float)prevU; s.u1 = (float)curU;
                out.push_back(s);
                ++added;
            }
            prevP = curP; prevR = curR; prevU = curU;
        }
    }
    return added;
}
