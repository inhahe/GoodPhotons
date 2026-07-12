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
    void finalize() { gn = normalize(cross(v1 - v0, v2 - v0)); }
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
    hit.n = (dot(r.d, tri.gn) < 0.0) ? tri.gn : -tri.gn;
    hit.matId = tri.matId; hit.sensorId = tri.sensorId;
    // Barycentric-interpolate the per-vertex UVs (u,v here are the Moller-Trumbore
    // weights of v1,v2; the v0 weight is 1-u-v).
    double w0 = 1.0 - u - v;
    hit.u = w0 * tri.uv0.x + u * tri.uv1.x + v * tri.uv2.x;
    hit.v = w0 * tri.uv0.y + u * tri.uv1.y + v * tri.uv2.y;
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
