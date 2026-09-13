#pragma once
// The geometric footprint for M-GATHERAREA, in its own header so both the diagnostic
// (roiboxes.h) and the estimator (photonmap_render.h) can reach it without either
// including the other.
#include <vector>
#include <cmath>

// The geometric footprint: same-facing surface area inside the gather ball, measured from
// the GEOMETRY rather than inferred from a photon statistic.
//
// TWO PRIMITIVE CASES, because the census showed spheres average 0-1.7 per ball and
// implicits and instances are exactly 0.0 in every scene this entry uses. They keep the
// present behaviour and cost nothing.
//
// TRIANGLES ARE SAMPLED THROUGH THE DISC, NOT OVER THEMSELVES. Sampling a primitive's own
// surface collapses when the primitive dwarfs the ball: `_ga_null`'s floor is a 12 m quad
// and a 0.05 m ball covers 1e-4 of it, so a few dozen samples over the triangle would
// return zero almost always. Casting through stratified points of the tangent disc instead
// makes the cost per primitive independent of its size.
//
// AND IT DELIBERATELY IGNORES OCCLUSION, which is the whole point. The shipped probe takes
// the NEAREST hit along each ray, so on a coat it measures the first layer while the query
// gathers from the entire ball — the failure this entry documents, and which the census put
// at ~600 curve segments per ball against ~3 on flat ground. Testing every candidate
// primitive against every disc ray counts all layers, so the two finally measure the same
// region.
//
// `1/|cos|` is the projection Jacobian: a disc ray meeting a surface at a slant subtends
// more surface area than the disc cell it came from.
inline double gatherFootprintArea(const Scene& sc, const Vec3& p, const Vec3& n,
                                  double r, int kDisc, int kCurve) {
    const size_t nT = sc.tris.size(), nS = sc.spheres.size(),
                 nI = sc.implicits.size(), nC = sc.curveSegs.size();
    const double r2 = r * r, kPi = 3.14159265358979323846;
    if (kDisc < 1) kDisc = 1;
    if (kCurve < 1) kCurve = 1;
    // Tangent frame and the stratified disc samples, built once per gather.
    Vec3 t1 = (std::fabs(n.x) < 0.9) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    t1 = normalize(cross(t1, n));
    const Vec3 t2 = cross(n, t1);
    std::vector<Vec3> disc;
    disc.reserve((size_t)kDisc);
    for (int i = 0; i < kDisc; ++i) {
        // Sunflower placement: equal-area radii, golden-angle azimuth. Deterministic, so
        // two runs of the same scene give the same footprint and an A/B is not comparing
        // two different random draws.
        const double rr = r * std::sqrt((i + 0.5) / (double)kDisc);
        const double th = (double)i * 2.39996322972865332;
        disc.push_back(p + (t1 * std::cos(th) + t2 * std::sin(th)) * rr);
    }
    const double cellArea = (kPi * r2) / (double)kDisc;
    double area = 0.0;
    sc.bvh.traverseSphere(p, r, [&](int pi) {
        size_t u = (size_t)pi;
        if (u < nT) {
            const Tri& t = sc.tris[u];
            const double cs = dot(t.gn, n);
            if (std::fabs(cs) < 0.5) return;        // not same-facing: the photon query
            const Vec3 e1 = t.v1 - t.v0, e2 = t.v2 - t.v0;   // rejects these too
            for (const Vec3& q : disc) {
                // Ray (q, n) against the triangle, unbounded in both directions.
                const Vec3 pv = cross(n, e2);
                const double det = dot(e1, pv);
                if (std::fabs(det) < 1e-18) continue;
                const double inv = 1.0 / det;
                const Vec3 tv = q - t.v0;
                const double bu = dot(tv, pv) * inv;
                if (bu < 0.0 || bu > 1.0) continue;
                const Vec3 qv = cross(tv, e1);
                const double bvv = dot(n, qv) * inv;
                if (bvv < 0.0 || bu + bvv > 1.0) continue;
                const double tt = dot(e2, qv) * inv;
                const Vec3 hitp = q + n * tt;
                if (dot(hitp - p, hitp - p) <= r2) area += cellArea / std::fabs(cs);
            }
            return;
        }
        u -= nT; if (u < nS) return;    // spheres:   0-1.7 per ball (census)
        u -= nS; if (u < nI) return;    // implicits: 0.0 everywhere
        u -= nI; if (u >= nC) return;   // instances: 0.0 everywhere
        // A curve segment is a thin cylinder whose size is COMPARABLE to the ball, so here
        // sampling the primitive itself is the cheap and accurate way round — the opposite
        // of the triangle case, and for the opposite reason.
        const CurveSeg& s = sc.curveSegs[u];
        const Vec3 ax = s.p1 - s.p0;
        const double L = std::sqrt(dot(ax, ax));
        if (!(L > 0.0)) return;
        const Vec3 az = ax * (1.0 / L);
        Vec3 c1 = (std::fabs(az.x) < 0.9) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
        c1 = normalize(cross(c1, az));
        const Vec3 c2 = cross(az, c1);
        const double lat = 2.0 * kPi * (0.5 * (s.r0 + s.r1)) * L;
        int acc = 0, tot = 0;
        for (int a = 0; a < kCurve; ++a) {
            const double fa = (a + 0.5) / (double)kCurve;
            const Vec3 cc = s.p0 + ax * fa;
            const double rr = s.r0 + (s.r1 - s.r0) * fa;
            for (int b = 0; b < kCurve; ++b) {
                const double th = 2.0 * kPi * ((b + 0.5) / (double)kCurve);
                const Vec3 nn = c1 * std::cos(th) + c2 * std::sin(th);
                const Vec3 q = cc + nn * rr;
                ++tot;
                // Facing is tested PER SAMPLE here: a cylinder's normal turns right around
                // its circumference, so one verdict for the whole segment would be meaningless.
                if (dot(q - p, q - p) <= r2 && std::fabs(dot(nn, n)) >= 0.5) ++acc;
            }
        }
        if (tot) area += lat * (double)acc / (double)tot;
    });
    return area;
}

