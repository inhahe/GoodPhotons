// Rays, triangles, Moller-Trumbore intersection. Brute-force closest-hit for now;
// a SAH BVH replaces the linear scan next (it must not change the image).
#pragma once
#include <vector>
#include <cfloat>
#include "linalg.h"

struct Ray { Vec3 o, d; };

struct Tri {
    Vec3 v0, v1, v2;
    int matId = 0;
    int sensorId = -1;      // >=0 if this triangle is part of a sensor
    Vec3 gn;               // geometric normal (unit)
    void finalize() { gn = normalize(cross(v1 - v0, v2 - v0)); }
};

struct Hit {
    double t = DBL_MAX;
    int tri = -1;
    Vec3 p, n;
    double u = 0, v = 0;   // barycentric-ish (for sensor mapping)
};

// Returns true on hit closer than hit.t; fills hit.
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
    hit.t = t; hit.p = r.o + r.d * t; hit.u = u; hit.v = v;
    // Two-sided: normal faces against the incoming ray.
    hit.n = (dot(r.d, tri.gn) < 0.0) ? tri.gn : -tri.gn;
    return true;
}
