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

// Watertight ray-triangle intersection (Woop, Benthin, Wald & Áfra, "Watertight
// Ray/Triangle Intersection", JCGT 2013). Unlike Möller-Trumbore, each edge is tested
// via a scaled barycentric edge function built from the *same* two shared vertices that
// the neighbouring triangle sees (in opposite winding), so the sign of the test is
// consistent across a shared edge: a ray passing exactly through the edge is claimed by
// exactly one triangle — never zero (a crack: background leaking through a closed mesh)
// and never both in a way that drops the hit. Cracks are most visible on the float GPU
// path; this double-precision CPU port keeps the two backends in lockstep.
//
// The per-ray part (axis permutation + shear) depends only on the ray direction, so it
// is factored into TriShear and computed ONCE per ray (hoisted out of the BVH leaf loop),
// then reused for every triangle.
struct TriShear {
    int    kx, ky, kz;    // permuted axes; kz = ray's dominant (largest |d|) axis
    double Sx, Sy, Sz;    // shear constants that map the ray direction onto +z
};

inline TriShear makeTriShear(const Vec3& d) {
    TriShear s;
    double ax = std::fabs(d.x), ay = std::fabs(d.y), az = std::fabs(d.z);
    if (ax >= ay && ax >= az)      s.kz = 0;   // dominant axis of the ray direction
    else if (ay >= az)             s.kz = 1;
    else                           s.kz = 2;
    s.kx = s.kz + 1; if (s.kx == 3) s.kx = 0;
    s.ky = s.kx + 1; if (s.ky == 3) s.ky = 0;
    // Swap kx,ky when the dominant component is negative so the winding (and thus the
    // edge-function sign convention) is preserved.
    if (d[s.kz] < 0.0) { int tmp = s.kx; s.kx = s.ky; s.ky = tmp; }
    s.Sx = d[s.kx] / d[s.kz];
    s.Sy = d[s.ky] / d[s.kz];
    s.Sz = 1.0     / d[s.kz];
    return s;
}

// Watertight test using a precomputed per-ray shear. Callers in a BVH leaf loop should
// build the shear once (makeTriShear(lr.d)) and pass it here for every triangle.
inline bool intersectTri(const TriShear& sh, const Ray& r, const Tri& tri,
                         double tmin, Hit& hit) {
    const int kx = sh.kx, ky = sh.ky, kz = sh.kz;
    // Triangle vertices relative to the ray origin.
    Vec3 A = tri.v0 - r.o, B = tri.v1 - r.o, C = tri.v2 - r.o;
    // Shear + scale so the ray direction becomes the +z axis; keep the xy of each vertex.
    double Ax = A[kx] - sh.Sx * A[kz], Ay = A[ky] - sh.Sy * A[kz];
    double Bx = B[kx] - sh.Sx * B[kz], By = B[ky] - sh.Sy * B[kz];
    double Cx = C[kx] - sh.Sx * C[kz], Cy = C[ky] - sh.Sy * C[kz];
    // Scaled barycentric edge functions (U,V,W weight v0,v1,v2 respectively).
    double U = Cx * By - Cy * Bx;
    double V = Ax * Cy - Ay * Cx;
    double W = Bx * Ay - By * Ax;
    // Exact-zero fallback in higher precision so a grazing edge lands deterministically
    // (a no-op precision-wise where long double == double, e.g. MSVC — the watertight
    // guarantee comes from the consistent edge ordering above, not from this).
    if (U == 0.0 || V == 0.0 || W == 0.0) {
        auto ld = [](double a, double b, double c, double d) {
            return (double)((long double)a * (long double)b - (long double)c * (long double)d);
        };
        if (U == 0.0) U = ld(Cx, By, Cy, Bx);
        if (V == 0.0) V = ld(Ax, Cy, Ay, Cx);
        if (W == 0.0) W = ld(Bx, Ay, By, Ax);
    }
    // Two-sided: reject only when the signs are mixed (point outside the triangle).
    // All-nonneg or all-nonpos both accept, so front and back faces both hit.
    if ((U < 0.0 || V < 0.0 || W < 0.0) && (U > 0.0 || V > 0.0 || W > 0.0)) return false;
    double det = U + V + W;
    if (det == 0.0) return false;
    // Interpolated (scaled) hit distance along the ray.
    double T = U * (sh.Sz * A[kz]) + V * (sh.Sz * B[kz]) + W * (sh.Sz * C[kz]);
    double invDet = 1.0 / det;
    double t = T * invDet;
    if (t < tmin || t >= hit.t) return false;
    double b0 = U * invDet, b1 = V * invDet, b2 = W * invDet;   // barycentric of v0,v1,v2
    hit.t = t; hit.p = r.o + r.d * t; hit.valid = true;
    hit.ng = tri.gn;
    hit.matId = tri.matId; hit.sensorId = tri.sensorId;
    // Barycentric-interpolated UV and shading normal (matches the old M-T convention:
    // b0,b1,b2 == w0,u,v). Orient the shading normal against the ray like the geo normal.
    hit.u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
    hit.v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
    Vec3 ns = tri.n0 * b0 + tri.n1 * b1 + tri.n2 * b2;
    double nl = dot(ns, ns);
    ns = (nl > 1e-18) ? ns * (1.0 / std::sqrt(nl)) : tri.gn;
    hit.n = (dot(r.d, ns) < 0.0) ? ns : -ns;
    return true;
}

// Interface-preserving wrapper: builds the per-ray shear inline. Fine for one-off calls;
// BVH leaf loops hoist makeTriShear(r.d) and use the overload above instead.
inline bool intersectTri(const Ray& r, const Tri& tri, double tmin, Hit& hit) {
    return intersectTri(makeTriShear(r.d), r, tri, tmin, hit);
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
