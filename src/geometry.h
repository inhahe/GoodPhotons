// Rays, triangles, spheres, intersection. Brute-force closest-hit for now;
// a SAH BVH replaces the linear scan later (it must not change the image).
#pragma once
#include <vector>
#include <cfloat>
#include <cmath>
#include <algorithm>
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
