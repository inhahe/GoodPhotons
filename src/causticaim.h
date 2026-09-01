// Caustic AIM MAP — Jensen's projection map, as a continuous mixture importance sampler.
//
// THE PROBLEM. Mode `M`'s two-map split (photonmap_render.h) gives caustics their own
// population and their own gather radius, which is the *storage* half of Jensen's scheme.
// It does nothing about the *sampling* half, and without that the sharp caustic map is
// sharp and empty: on `scenes/gallery_rain.ftsl` at `-n 40000000` the forward pass makes
// 17 573 528 global deposits against 21 808 caustic ones — 0.12 %. The gallery's twelve
// gems, its gyroid, its Klein bottle and its axicon together subtend a tiny solid angle of
// a sky-lit 65 m scene, and emission is uniform over the emitter, so almost no photon ever
// reaches a dielectric. Raising `-n` fixes it only linearly, at 800x the cost.
//
// THE CLASSIC FIX is a projection map: a coarse spherical grid per emitter, marking the
// cells whose directions can reach specular geometry, sampled instead of the full sphere.
// This is the same idea with the grid taken out. A grid needs a resolution (one more knob
// with no principled value), it is conservative by a whole cell in every direction, and its
// pdf is a piecewise constant that has to be stored and normalised. What we actually want
// to importance-sample is the *union of the focusing objects' footprints* as seen from the
// emitter, and that union is available in closed form if we bound the focusing geometry
// with a handful of spheres — a cone per sphere for a local emitter, a disc per sphere for
// a distant one. So:
//
//   * `AimMap` is a small set of bounding spheres (`Target`) covering every primitive whose
//     material can focus (`materialMayFocus` below, the Hit-free conservative twin of
//     `photonVertexKind`), clustered to at most `-causticaimk` of them by repeatedly
//     splitting whichever cluster has the largest r^2 — which is exactly the quantity that
//     sets how much of the aimed sample budget lands on empty space.
//
//   * The aimed sampler picks a target proportional to its own footprint measure, then
//     samples uniformly inside it. Choosing the selection probability that way is what
//     makes the resulting mixture density collapse to something free to evaluate:
//
//         P_j = m_j / T,  q_j = 1/m_j  (uniform inside footprint j)
//         =>  p_a(x) = SUM_j P_j q_j [x in j] = (number of footprints containing x) / T
//
//     — one count and one divide, whatever the geometry is. `T` is the total footprint
//     measure (solid angle summed over targets, or projected area summed over targets),
//     and overlapping footprints simply cost a little efficiency, never correctness.
//
// UNBIASEDNESS. The aimed pass does NOT replace the ordinary one; the two are combined with
// the balance heuristic, which is what makes an incomplete or over-eager target set a
// question of efficiency rather than of correctness. Writing N_m for the main pass's photon
// count and N_c for the caustic pass's, the balance-heuristic estimator gives BOTH passes'
// caustic deposits the same weight,
//
//     w(x) = N_m p_u(x) / (N_m p_u(x) + N_c p_a(x))
//          = 1 / (1 + (N_c/N_m) * rho(x)),      rho(x) = p_a(x)/p_u(x)
//
// so the whole scheme reduces to: compute `rho` at emission, scale every CAUSTIC deposit by
// `w`, and leave the caustic map's `nEmitted` at the main pass's count exactly as it already
// was. A main-pass photon that lands nowhere near a gem has rho = 0 and w = 1, i.e. is
// completely unchanged — which is 99.88 % of them. An emitter that cannot be aimed at all
// (a collimated one: its direction is a delta) falls back to the ordinary sampler in the
// caustic pass too, where rho = 1 makes the two strategies identical and the balance
// heuristic degenerates to splitting the photons between two equal passes. Still exact.
//
// WHAT IS AIMED, PER EMITTER SHAPE. A photon's emission sample is (origin, direction), and
// which half is worth aiming depends on which one the tiny target actually constrains:
//
//   * Sun / Env — the direction is fixed by the solar cone or by the sky's own importance
//     map, and the ORIGIN is drawn on a disc of radius `sceneRadius` upstream of the whole
//     scene. Aiming the direction would be pointless (the sun's cone is 0.53 deg wide);
//     aiming the origin is everything. Footprint = the target sphere's projection onto that
//     disc, a circle of radius r_j, measure pi*r_j^2.
//   * Quad / Sphere / Cylinder / Mesh / Spot, and a volumetric (fire) birth — the origin is
//     on the emitter and the DIRECTION is drawn over a hemisphere / cone / sphere. Footprint
//     = the cone subtending the target sphere from the origin, measure 2*pi*(1-cosMax).
//
// Both cases are the same two calls, `aimFootprintTotal` + `aimSample`/`aimPdfRatio`, and
// both reduce to the counting formula above.
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "scene.h"
#include "curve.h"
#include "photonmap.h"   // kCausticGlossRoughness — one definition shared with photonVertexKind

namespace caim {

// One bounding sphere over a cluster of focusing primitives.
struct Target {
    Vec3   c{0, 0, 0};
    double r = 0.0;
};

struct AimMap {
    std::vector<Target> targets;
    // SUM_j r_j^2 — the distant-emitter footprint measure, which depends on nothing but the
    // targets themselves. Cached at build time because the alternative is recomputing it once
    // per emitted photon.
    double sumR2 = 0.0;
    // Diagnostics for the build log: how many primitives went in, and how big the union
    // ended up relative to the scene. Nothing reads these during sampling.
    long long nPrims = 0;
    bool empty() const { return targets.empty(); }
};

// ---- Which materials can start a caustic ------------------------------------------------
// The Hit-free twin of render.h's photonVertexKind. It has to be CONSERVATIVE in one
// direction only: a material wrongly called focusing costs a slightly larger target set
// (efficiency), while one wrongly called non-focusing costs coverage, and the MIS weights
// then leave those caustics to the main pass alone — correct, but back to 0.12 %.
//
// The one genuinely undecidable case is a glossy material whose roughness is driven by a
// texture / pattern / record, since the deciding value is per hit. Those are called
// focusing, which is the safe direction.
inline bool materialMayFocus(const Scene& scene, int matId, int depth = 0) {
    if (matId < 0 || matId >= (int)scene.mats.size()) return false;
    if (depth > 8) return true;                   // pathological material cycle: be safe
    const Material& m = scene.mats[matId];
    switch (m.type) {
        case MatType::Dielectric:
        case MatType::Mirror:
        case MatType::ThinFilm:
        case MatType::Multilayer:
        case MatType::Grating:
        case MatType::HalfMirror:
            return true;
        case MatType::Glossy:
            // Driven roughness: undecidable without a hit, so assume it can be tight.
            if (m.roughnessPat >= 0 || m.roughnessTex >= 0 ||
                m.recBindingFor(REC_SLOT_ROUGHNESS) != nullptr) return true;
            return m.roughness <= kCausticGlossRoughness;
        case MatType::Mix:
            for (int ch : m.mixChildren)
                if (materialMayFocus(scene, ch, depth + 1)) return true;
            return false;
        case MatType::Layered: {
            // A clearcoat reflection focuses when the coat is tight (the highlight a
            // polished coat throws IS a caustic) — see the Layered branch of tracePhoton.
            if (m.roughnessPat >= 0 || m.roughnessTex >= 0 ||
                m.recBindingFor(REC_SLOT_ROUGHNESS) != nullptr) return true;
            if (m.roughness <= kCausticGlossRoughness) return true;
            for (int ch : m.mixChildren)
                if (materialMayFocus(scene, ch, depth + 1)) return true;
            return false;
        }
        case MatType::Filter:
            return false;                          // passes straight through, direction untouched
        default:
            return false;                          // Diffuse family, Fluorescent, Hair
    }
}

// ---- Build -------------------------------------------------------------------------------
// Collect the world AABB of every focusing primitive, then cluster to at most `maxTargets`
// bounding spheres.
//
// The clustering objective is SUM r_j^2, not the usual SAH: the aimed sampler spends its
// budget uniformly over each footprint, and a footprint's measure is proportional to r_j^2
// (projected area) or, for a distant target, close to it (solid angle ~ pi r_j^2/d^2). So
// sum-of-r^2 IS the expected fraction of aimed photons that miss, and driving it down is
// the only thing splitting buys. Hence: repeatedly split whichever cluster currently has
// the largest r^2, by the median of the centroids along its longest axis.
inline AimMap build(const Scene& scene, int maxTargets = 64) {
    AimMap am;
    if (maxTargets < 1) maxTargets = 1;

    std::vector<Aabb> boxes;
    auto add = [&](const Aabb& b) { if (b.hi.x >= b.lo.x) boxes.push_back(b); };

    for (const Tri& t : scene.tris) {
        if (!materialMayFocus(scene, t.matId)) continue;
        Aabb b; b.expand(t.v0); b.expand(t.v1); b.expand(t.v2); add(b);
    }
    for (const Sphere& s : scene.spheres) {
        if (!materialMayFocus(scene, s.matId)) continue;
        Aabb b; b.expand(s.c - Vec3{s.r, s.r, s.r}); b.expand(s.c + Vec3{s.r, s.r, s.r}); add(b);
    }
    for (const Implicit& im : scene.implicits) {
        if (!materialMayFocus(scene, im.matId)) continue;
        add(im.bounds);
    }
    for (const CurveSeg& cs : scene.curveSegs) {
        if (!materialMayFocus(scene, cs.matId)) continue;
        add(curveSegBounds(cs));
    }
    // Instanced meshes. `matOverride` decides for the whole instance when it is set;
    // otherwise the BLAS's own triangle materials do, and a single focusing triangle makes
    // the WHOLE instance a target. That is coarse on purpose — the alternative is
    // transforming every triangle of every instance into world space at build time, which
    // for a mesh asset placed a hundred times is a hundred full copies of it. A too-large
    // target only costs aimed photons that miss.
    for (const MeshInstance& inst : scene.instances) {
        if (inst.blasId < 0 || inst.blasId >= (int)scene.blasList.size()) continue;
        bool focus = false;
        if (inst.matOverride >= 0) {
            focus = materialMayFocus(scene, inst.matOverride);
        } else {
            for (const Tri& t : scene.blasList[inst.blasId].tris)
                if (materialMayFocus(scene, t.matId)) { focus = true; break; }
        }
        if (!focus) continue;
        const Aabb& lb = scene.blasList[inst.blasId].localBounds;
        Aabb wb;
        for (int c = 0; c < 8; ++c) {
            Vec3 corner{ (c & 1) ? lb.hi.x : lb.lo.x,
                         (c & 2) ? lb.hi.y : lb.lo.y,
                         (c & 4) ? lb.hi.z : lb.lo.z };
            wb.expand(inst.toWorld.apply(corner));
        }
        add(wb);
    }

    am.nPrims = (long long)boxes.size();
    if (boxes.empty()) return am;

    // --- cluster ---------------------------------------------------------------------
    struct Cluster { int lo = 0, hi = 0; Vec3 c; double r = 0.0; bool frozen = false; };
    std::vector<int> idx(boxes.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;

    // Bounding sphere of a range: centre = middle of the union box, radius = the farthest
    // box corner from it. Tight enough (a hair looser than the true minimal sphere) and
    // linear, which matters when the input is a million triangles.
    auto bound = [&](Cluster& cl) {
        Aabb u;
        for (int i = cl.lo; i < cl.hi; ++i) u.expand(boxes[idx[i]]);
        cl.c = u.center();
        double r2 = 0.0;
        for (int i = cl.lo; i < cl.hi; ++i) {
            const Aabb& b = boxes[idx[i]];
            for (int k = 0; k < 8; ++k) {
                Vec3 p{ (k & 1) ? b.hi.x : b.lo.x, (k & 2) ? b.hi.y : b.lo.y,
                        (k & 4) ? b.hi.z : b.lo.z };
                Vec3 d = p - cl.c;
                r2 = std::max(r2, dot(d, d));
            }
        }
        cl.r = std::sqrt(r2);
    };

    std::vector<Cluster> cls;
    cls.push_back(Cluster{0, (int)idx.size(), Vec3{0, 0, 0}, 0.0, false});
    bound(cls[0]);

    while ((int)cls.size() < maxTargets) {
        // Split the worst cluster — the one contributing most wasted aimed samples.
        int worst = -1; double worstR = 0.0;
        for (size_t i = 0; i < cls.size(); ++i)
            if (cls[i].hi - cls[i].lo > 1 && !cls[i].frozen && cls[i].r > worstR) {
                worstR = cls[i].r; worst = (int)i;
            }
        if (worst < 0) break;                     // every cluster is a single primitive
        Cluster c = cls[worst];
        Aabb cb;                                   // centroid bounds pick the split axis
        for (int i = c.lo; i < c.hi; ++i) cb.expand(boxes[idx[i]].center());
        const int ax = cb.largestAxis();
        const int mid = c.lo + (c.hi - c.lo) / 2;
        std::nth_element(idx.begin() + c.lo, idx.begin() + mid, idx.begin() + c.hi,
                         [&](int a, int b) {
                             const Vec3 ca = boxes[a].center(), cbn = boxes[b].center();
                             const double va = ax == 0 ? ca.x : (ax == 1 ? ca.y : ca.z);
                             const double vb = ax == 0 ? cbn.x : (ax == 1 ? cbn.y : cbn.z);
                             return va < vb;
                         });
        Cluster l{c.lo, mid, {}, 0.0, false}, r{mid, c.hi, {}, 0.0, false};
        // A degenerate split (all centroids coincident, so one side is empty) would loop
        // forever picking the same cluster; freeze it instead. `frozen` rather than r = 0
        // because r is what the FOOTPRINT is computed from — zeroing it would silently
        // delete the cluster from the finished map.
        if (l.hi <= l.lo || r.hi <= r.lo) { cls[worst].frozen = true; continue; }
        bound(l); bound(r);
        cls[worst] = l;
        cls.push_back(r);
    }

    am.targets.reserve(cls.size());
    for (const Cluster& c : cls)
        if (c.r > 0.0) am.targets.push_back(Target{c.c, c.r});
    for (const Target& t : am.targets) am.sumR2 += t.r * t.r;
    return am;
}

// ---- Distant emitters (Sun / Env): aim the upstream ORIGIN DISC --------------------------
//
// The ordinary sampler draws the entry point uniformly on a disc of radius `sceneRadius`
// perpendicular to the travel direction, i.e. a density 1/(pi R^2). Target j projects onto
// that same plane as a circle of radius r_j about the projection of its centre, so
//
//     p_a(p) = (number of target circles containing p) / (pi * SUM_j r_j^2)
//     rho(p) = p_a/p_u = R^2 * count(p) / SUM_j r_j^2
//
// `t`/`b` are the plane's orthonormal basis (the same pair the emitter code builds with
// `onb(dir, t, b)`), and offsets are expressed in those 2-D coordinates.
struct DiscAim {
    double sumR2 = 0.0;                 // SUM_j r_j^2 — the footprint measure T/pi
};

inline DiscAim discAim(const AimMap& am) {
    DiscAim d;
    d.sumR2 = am.sumR2;                 // cached at build: independent of direction and origin
    return d;
}

// 2-D coordinates of target j's projected centre, in the (t, b) basis of the plane through
// `sceneCenter` perpendicular to `dir`.
inline void discProject(const Target& tg, const Vec3& sceneCenter, const Vec3& t, const Vec3& b,
                        double& x, double& y) {
    const Vec3 w = tg.c - sceneCenter;
    x = dot(w, t);
    y = dot(w, b);
}

// How many target circles contain the plane point (x, y).
inline int discCount(const AimMap& am, const Vec3& sceneCenter, const Vec3& t, const Vec3& b,
                     double x, double y) {
    int n = 0;
    for (const Target& tg : am.targets) {
        double cx, cy; discProject(tg, sceneCenter, t, b, cx, cy);
        const double dx = x - cx, dy = y - cy;
        if (dx * dx + dy * dy < tg.r * tg.r) ++n;
    }
    return n;
}

// Draw a plane point from the aimed mixture. `u0` picks the target (proportional to r^2),
// `u1`/`u2` place the point inside it. Returns false only for an empty map.
inline bool discSample(const AimMap& am, const DiscAim& da, const Vec3& sceneCenter,
                       const Vec3& t, const Vec3& b, double u0, double u1, double u2,
                       double& x, double& y) {
    if (am.targets.empty() || !(da.sumR2 > 0.0)) return false;
    double pick = u0 * da.sumR2, acc = 0.0;
    size_t j = 0;
    for (; j + 1 < am.targets.size(); ++j) {
        acc += am.targets[j].r * am.targets[j].r;
        if (pick < acc) break;
    }
    const Target& tg = am.targets[j];
    double cx, cy; discProject(tg, sceneCenter, t, b, cx, cy);
    const double rr = tg.r * std::sqrt(u1);
    const double ph = 2.0 * PI * u2;
    x = cx + rr * std::cos(ph);
    y = cy + rr * std::sin(ph);
    return true;
}

// ---- Local emitters (area / spot / volumetric birth): aim the DIRECTION -------------------
//
// Target j subtends a cone of half-angle asin(r_j/d_j) from the emission point, of solid
// angle Omega_j = 2*pi*(1 - cosMax_j). Selecting proportional to Omega_j gives
//
//     p_a(w) = (number of cones containing w) / SUM_j Omega_j
//
// A point INSIDE a target sphere subtends the whole sphere (Omega = 4*pi, every direction
// inside), which is the correct limit and keeps an emitter embedded in a dielectric from
// producing a division by zero.
struct ConeAim {
    double sumOmega = 0.0;
};

// cos of the cone half-angle for target j seen from `o`; -1 when `o` is inside it.
inline double coneCos(const Target& tg, const Vec3& o) {
    const Vec3 w = tg.c - o;
    const double d2 = dot(w, w), r2 = tg.r * tg.r;
    if (d2 <= r2) return -1.0;                    // inside: the whole sphere of directions
    return std::sqrt(std::max(0.0, 1.0 - r2 / d2));
}

inline ConeAim coneAim(const AimMap& am, const Vec3& o) {
    ConeAim c;
    for (const Target& tg : am.targets) c.sumOmega += 2.0 * PI * (1.0 - coneCos(tg, o));
    return c;
}

inline int coneCount(const AimMap& am, const Vec3& o, const Vec3& w) {
    int n = 0;
    for (const Target& tg : am.targets) {
        const Vec3 v = tg.c - o;
        const double d2 = dot(v, v), r2 = tg.r * tg.r;
        if (d2 <= r2) { ++n; continue; }
        const double dl = std::sqrt(d2);
        if (dot(w, v) / dl >= std::sqrt(std::max(0.0, 1.0 - r2 / d2))) ++n;
    }
    return n;
}

// Draw a direction from the aimed cone mixture.
inline bool coneSample(const AimMap& am, const ConeAim& ca, const Vec3& o,
                       double u0, double u1, double u2, Vec3& w) {
    if (am.targets.empty() || !(ca.sumOmega > 0.0)) return false;
    double pick = u0 * ca.sumOmega, acc = 0.0;
    size_t j = 0;
    for (; j + 1 < am.targets.size(); ++j) {
        acc += 2.0 * PI * (1.0 - coneCos(am.targets[j], o));
        if (pick < acc) break;
    }
    const Target& tg = am.targets[j];
    const double cosMax = coneCos(tg, o);
    Vec3 axis = tg.c - o;
    const double al = length(axis);
    axis = (al > 1e-12) ? axis / al : Vec3{0, 0, 1};
    const double ct = 1.0 - u1 * (1.0 - cosMax);
    const double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
    const double ph = 2.0 * PI * u2;
    Vec3 tt, bb; onb(axis, tt, bb);
    w = tt * (st * std::cos(ph)) + bb * (st * std::sin(ph)) + axis * ct;
    return true;
}

}  // namespace caim
