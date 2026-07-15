// Rays, triangles, spheres, intersection. Brute-force closest-hit for now;
// a SAH BVH replaces the linear scan later (it must not change the image).
#pragma once
#include <vector>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <string>
#include "linalg.h"

constexpr double PI = 3.141592653589793;

struct Ray { Vec3 o, d; };

struct Tri {
    Vec3 v0, v1, v2;
    int matId = 0;
    int sensorId = -1;      // >=0 if this triangle is part of a sensor
    Vec3 gn;               // geometric normal (unit)
    // Per-vertex texture coordinates (u in .x, v in .y; .z unused). Defaults give
    // a sensible mapping for an untextured tri; quads and OBJ `vt` fill real values.
    Vec3 uv0{0, 0, 0}, uv1{1, 0, 0}, uv2{1, 1, 0};
    // Per-vertex SHADING normals (barycentric-interpolated at a hit for smooth
    // shading). Zero-length => "not supplied": finalize() falls them back to the
    // geometric normal, so any tri without OBJ `vn` data stays exactly flat-shaded.
    Vec3 n0{0, 0, 0}, n1{0, 0, 0}, n2{0, 0, 0};
    void finalize() {
        gn = normalize(cross(v1 - v0, v2 - v0));
        if (dot(n0, n0) < 1e-12) n0 = gn;
        if (dot(n1, n1) < 1e-12) n1 = gn;
        if (dot(n2, n2) < 1e-12) n2 = gn;
    }
};

struct Sphere {
    Vec3 c; double r = 1.0;
    int matId = 0;
};

struct Hit {
    double t = DBL_MAX;
    bool valid = false;
    Vec3 p, n, ng;         // n = oriented against ray; ng = raw geometric normal
    int matId = 0;
    int sensorId = -1;
    double u = 0, v = 0;   // interpolated surface texture coordinates
    double fieldVal = 0;   // implicit field value at the hit (~0 on a surface; 0 for
                           // non-implicit hits). Exposed to procedural patterns as `f`.
};

// Geometric surface normal oriented onto the SAME side as the (ray-oriented) shading
// normal h.n. For a flat triangle or an analytic sphere the shading and geometric
// normals coincide, so this returns exactly h.n's direction (the helper is a no-op
// there). It only differs when a smooth/interpolated shading normal (authored `vn`
// or crease-smoothing) diverges from the true geometry — the case where next-event
// estimation and BSDF continuations must be clamped to the geometric hemisphere to
// stop light leaking through the back of the surface (the shading-normal problem).
inline Vec3 orientedGeoN(const Hit& h) {
    return (dot(h.ng, h.n) >= 0.0) ? h.ng : Vec3{-h.ng.x, -h.ng.y, -h.ng.z};
}

// Veach shading-normal ADJOINT correction factor (Veach §5.3; PBRT
// `CorrectShadingNormal`, importance/light-transport mode). A BSDF evaluated with
// an interpolated *shading* normal `ns` instead of the true *geometric* normal `ng`
// is non-symmetric: a backward path tracer (radiance transport) gets smooth shading
// for free, but a forward/particle tracer (light transport) deposits irradiance per
// GEOMETRIC area and would leave the surface faceted. Multiplying the particle
// throughput by this factor at every scattering vertex (for the sampled continuation
// direction `wo`) and at every camera connection (for `wo` = toward the camera)
// restores agreement, so smooth-normal meshes shade smoothly in the forward modes too.
//
//   corr = |cos(wi,Ns)·cos(wo,Ng)| / |cos(wi,Ng)·cos(wo,Ns)|
//
// with wi = direction toward the PREVIOUS (light-side) vertex (= -ray.d) and
// wo = the outgoing direction. It is **exactly 1 when Ns == Ng** (flat triangles,
// analytic spheres): num and denom are the identical products, so every non-smooth
// scene is bit-identical and the whole existing validation suite is untouched.
//
// The grazing `cos(wo,Ns)` denominator is guarded: at a camera connection the caller
// multiplies an existing `cosSurf = cos(wo,Ns)` term, so `cosSurf·corr` cancels that
// factor analytically (→ cos(wo,Ng)·cos(wi,Ns)/cos(wi,Ng)) and stays bounded; the
// explicit denom guard here only trips on a genuinely degenerate (measure-zero)
// grazing sample, where returning 1 (no correction) is the safe, low-bias fallback.
inline double shadingAdjointCorr(const Vec3& wi, const Vec3& wo,
                                 const Vec3& ns, const Vec3& ng) {
    double denom = std::fabs(dot(wi, ng)) * std::fabs(dot(wo, ns));
    if (denom <= 1e-8) return 1.0;                 // degenerate grazing -> no correction
    double num = std::fabs(dot(wi, ns)) * std::fabs(dot(wo, ng));
    return num / denom;
}

// Shadow-terminator softening (Chiang, Li, Burley & Hovhannisyan 2019, "Taming the Shadow
// Terminator"; the same cubic used by Blender Cycles' `bump_shadowing_term`). Returns a
// factor in [0,1] to multiply a light connection / NEE contribution by, REPLACING the old
// hard geometric-hemisphere cutoff (`dot(ng,wi) <= 0 ? reject`). It still returns exactly 0
// when `wi` is behind the true geometry (no light leaks through the geometric back face), but
// instead of a hard step it ramps up SMOOTHLY as `wi` climbs off the geometric horizon — so a
// low-poly smooth-normal mesh under grazing light shows a smooth terminator instead of hard
// facet slivers (the classic shading-normal / terminator artifact).
//
//   g = cos(Ng,wi) / (cos(Ns,wi) · cos(Ng,Ns)),   softened by  -g^3 + g^2 + g  on (0,1)
//
// `ns` = shading normal, `ng` = geometric normal ORIENTED onto the shading side (orientedGeoN),
// `wi` = direction toward the light/connection. Callers gate on the shading cosine first, so
// cos(Ns,wi) > 0 in practice. **Exactly 1 when Ns == Ng** (flat tris, analytic spheres): then
// cos(Ng,Ns)=1 and cos(Ng,wi)=cos(Ns,wi) ⇒ g=1 ⇒ factor 1, so every non-smooth scene is
// bit-identical and the whole validation suite is untouched. The cubic is C1 at g=1 (its
// derivative there is 0), so the well-lit region blends in without a crease.
inline double shadowTerminatorG(const Vec3& wi, const Vec3& ns, const Vec3& ng) {
    double cosNgNs = dot(ng, ns);
    // Exact no-op when the shading and geometric normals coincide (flat tris, analytic
    // spheres): the softening cubic would otherwise return ~1 (not bit-exactly 1) because
    // an interpolated `ns` is re-normalized and can differ from `ng` in the last bit, so
    // every flat/analytic scene would drift by ~1e-7. Short-circuit to a plain leak-free
    // step there — identical to the old hard geometric-hemisphere clamp — so the whole
    // flat-scene validation suite stays bit-identical. Softening only engages once ns and
    // ng genuinely diverge (a real smooth-normal / crease-smoothed mesh).
    if (cosNgNs >= 1.0 - 1e-7) return (dot(ng, wi) > 0.0) ? 1.0 : 0.0;
    double cosNgWi = dot(ng, wi);
    if (cosNgWi <= 0.0) return 0.0;                // behind the true geometry -> hard shadow (no leak)
    double denom = dot(ns, wi) * cosNgNs;
    if (denom <= 1e-8) return 1.0;                 // degenerate (grazing / near-perpendicular): no softening
    double g = cosNgWi / denom;
    if (g >= 1.0) return 1.0;                      // fully lit -> no darkening
    return g * g * (1.0 - g) + g;                  // Chiang cubic: -g^3 + g^2 + g
}

inline bool intersectTri(const Ray& r, const Tri& tri, double tmin, Hit& hit) {
    const double EPS = 1e-9;
    Vec3 e1 = tri.v1 - tri.v0, e2 = tri.v2 - tri.v0;
    Vec3 pv = cross(r.d, e2);
    double det = dot(e1, pv);
    if (std::fabs(det) < EPS) return false;
    double inv = 1.0 / det;
    Vec3 tv = r.o - tri.v0;
    double u = dot(tv, pv) * inv;
    if (u < 0.0 || u > 1.0) return false;
    Vec3 qv = cross(tv, e1);
    double v = dot(r.d, qv) * inv;
    if (v < 0.0 || u + v > 1.0) return false;
    double t = dot(e2, qv) * inv;
    if (t < tmin || t >= hit.t) return false;
    hit.t = t; hit.p = r.o + r.d * t; hit.valid = true;
    hit.ng = tri.gn;
    hit.matId = tri.matId; hit.sensorId = tri.sensorId;
    // Barycentric weights: u,v here are the Moller-Trumbore weights of v1,v2; the
    // v0 weight is 1-u-v. Reused for both UVs and the shading normal.
    double w0 = 1.0 - u - v;
    hit.u = w0 * tri.uv0.x + u * tri.uv1.x + v * tri.uv2.x;
    hit.v = w0 * tri.uv0.y + u * tri.uv1.y + v * tri.uv2.y;
    // Smooth shading normal: interpolate the per-vertex normals (equal to gn for a
    // flat tri, so this reduces to the geometric normal there). Orient against the
    // ray to match the geometric-normal convention.
    Vec3 ns = w0 * tri.n0 + u * tri.n1 + v * tri.n2;
    double nl = dot(ns, ns);
    ns = (nl > 1e-18) ? ns * (1.0 / std::sqrt(nl)) : tri.gn;
    hit.n = (dot(r.d, ns) < 0.0) ? ns : -ns;
    return true;
}

inline bool intersectSphere(const Ray& r, const Sphere& s, double tmin, Hit& hit) {
    Vec3 oc = r.o - s.c;
    double a = dot(r.d, r.d);
    double b = 2.0 * dot(oc, r.d);
    double c = dot(oc, oc) - s.r * s.r;
    double disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return false;
    double sq = std::sqrt(disc);
    double t = (-b - sq) / (2.0 * a);
    if (t < tmin) t = (-b + sq) / (2.0 * a);
    if (t < tmin || t >= hit.t) return false;
    hit.t = t; hit.p = r.o + r.d * t; hit.valid = true;
    Vec3 ng = normalize(hit.p - s.c);
    hit.ng = ng;
    hit.n = (dot(r.d, ng) < 0.0) ? ng : -ng;
    hit.matId = s.matId; hit.sensorId = -1;
    // Equirectangular (lat/long) UV so spheres can be textured (globes, eyeballs).
    hit.u = 0.5 + std::atan2(ng.z, ng.x) / (2.0 * PI);
    hit.v = 0.5 - std::asin(std::clamp(ng.y, -1.0, 1.0)) / PI;
    return true;
}

// ---------------------------------------------------------------------------
// Procedural UV projection — the same wrap used for un-`vt`'d meshes (spec §9.2),
// factored here so native primitives (isosurfaces) can reuse it. `axis` (0=x,1=y,
// 2=z) is the projection/up axis; coordinates normalise to [0,1] across the given
// AABB (lo..hi) so the map wraps once over the object by default.
// ---------------------------------------------------------------------------
enum class UvProjection { None = 0, Planar, Spherical, Cylindrical };

inline UvProjection parseUvProjection(const std::string& s) {
    if (s == "planar")      return UvProjection::Planar;
    if (s == "spherical")   return UvProjection::Spherical;
    if (s == "cylindrical") return UvProjection::Cylindrical;
    return UvProjection::None;
}

// Project one world-space point to (u,v) given an AABB (lo..hi), its centre, the
// projection kind and the up/projection axis (0/1/2). Returns {u,v,0}.
inline Vec3 projectUV(const Vec3& p, const Vec3& lo, const Vec3& hi,
                      const Vec3& ctr, UvProjection proj, int axis) {
    auto comp = [](const Vec3& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); };
    int a0 = (axis + 1) % 3, a1 = (axis + 2) % 3;
    auto norm01 = [&](double val, int i) {
        double l = comp(lo, i), h = comp(hi, i);
        double d = h - l;
        return d > 1e-12 ? (val - l) / d : 0.5;
    };
    if (proj == UvProjection::Planar) {
        return Vec3{norm01(comp(p, a0), a0), norm01(comp(p, a1), a1), 0};
    }
    Vec3 d = p - ctr;
    double dz = comp(d, axis);
    double dx = comp(d, a0), dy = comp(d, a1);
    double azim = 0.5 + std::atan2(dy, dx) / (2.0 * PI);   // [0,1)
    if (proj == UvProjection::Cylindrical) {
        return Vec3{azim, norm01(comp(p, axis), axis), 0};
    }
    double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    double v = (r > 1e-12) ? std::acos(std::max(-1.0, std::min(1.0, dz / r))) / PI : 0.5;
    return Vec3{azim, v, 0};
}
