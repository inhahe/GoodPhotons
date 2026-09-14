#pragma once
// The geometric footprint for M-GATHERAREA, in its own header so both the diagnostic
// (roiboxes.h) and the estimator (photonmap_render.h) can reach it without either
// including the other.
#include <vector>
#include <cmath>

// The geometric footprint: same-facing surface area inside the gather ball, measured from
// the GEOMETRY rather than inferred from a photon statistic.
//
// WHY IT MARCHES RAYS INSTEAD OF CLASSIFYING PRIMITIVES. The first version walked the BVH's
// candidate primitives and computed each one's area analytically, which needed a routine per
// primitive class. It had two routines — triangles and curve segments — chosen from a census
// taken on `fur_creature`, which happens to contain nothing else. `gallery_rain`, the scene
// the queue actually names, is built from implicit isosurfaces: that version reported
// footprint 0.0000 over 38 gathers on a solid marble cap, because a skipped class does not
// drop out of the answer, it silently subtracts its area.
//
// Marching the ray removes the whole category of error. Every surface the renderer can
// intersect is counted, by the intersector the renderer already trusts, with no per-class
// code to be complete or incomplete. Triangles, spheres, implicits, curve segments and
// instances all work, and a class added later works without touching this file.
//
// IT IGNORES OCCLUSION ON PURPOSE — that is the entire point. The shipped probe takes the
// NEAREST hit along each ray, so on a coat it measures the first layer while the query
// gathers from the whole ball; the census put that at ~600 curve segments per ball against
// ~3 on flat ground. Marching past each hit and continuing counts every layer, so the
// footprint and the photon query finally describe the same region.
//
// `1/cos` is the projection Jacobian: a ray meeting a surface at a slant subtends more
// surface area than the disc cell it came from. The facing test is SIGNED and calibrated
// per gather (see below), so `cos` is positive wherever a layer is accepted.
//
// `incomplete` (optional) is set when a ray hit the layer cap, i.e. the chord held more
// surfaces than were counted, so the area is an under-count and the caller should not
// divide by it.
inline double gatherFootprintArea(const Scene& sc, const Vec3& p, const Vec3& n,
                                  double r, int kDisc, int maxLayers,
                                  bool* incomplete = nullptr) {
    if (incomplete) *incomplete = false;
    if (!(r > 0.0) || kDisc < 1) return 0.0;
    if (maxLayers < 1) maxLayers = 1;
    const double kPi = 3.14159265358979323846;
    const double r2 = r * r;
    const double eps = (r * 1e-4 > 1e-9) ? r * 1e-4 : 1e-9;   // scale-relative, not absolute

    Vec3 t1 = (std::fabs(n.x) < 0.9) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    t1 = normalize(cross(t1, n));
    const Vec3 t2 = cross(n, t1);

    // CALIBRATE THE AUTHORED ORIENTATION, once per gather, against the surface the gather
    // point is actually on. `Hit::ng` is raw geometric orientation, which is dependable for
    // a closed mesh (outward) and arbitrary for a hand-authored quad — `_ga_null`'s floor has
    // `u x v` pointing DOWN, so a signed test against it rejects the very surface being
    // gathered from and the footprint collapses to zero. One extra trace fixes that: find the
    // surface at `p` and, if its `ng` opposes the gather normal, flip the whole comparison for
    // this gather. Then a signed test means "faces the same way as the surface I am standing
    // on", which is exactly what the photon query's `dot(ph.n, h.n) >= 0.5` means and is
    // independent of how the scene was authored.
    //
    // Signed (rather than |dot|) is what stops a CLOSED SOLID being counted twice: its far
    // face opposes and drops out, matching the query, which gathers no photons from it. And
    // unlike parity-by-crossing-order this still counts every layer of STACKED THIN SHELLS,
    // whose normals all agree — cloth folds and coat layers are real same-facing surface and
    // the query does take photons from all of them.
    double orient = 1.0;
    {
        const Ray cal{p - n * (eps * 4.0), n};
        const Hit hc = sc.closestHit(cal, eps, nullptr, /*skipHair=*/false,
                                     /*skipCamHidden=*/true);
        if (hc.valid && dot(hc.ng, n) < 0.0) orient = -1.0;
    }
    const double cellArea = (kPi * r2) / (double)kDisc;
    double area = 0.0;
    for (int i = 0; i < kDisc; ++i) {
        // Sunflower placement: equal-area radii, golden-angle azimuth. Deterministic, so two
        // runs of the same scene give the same footprint and an A/B compares one thing.
        const double rr = r * std::sqrt((i + 0.5) / (double)kDisc);
        const double th = (double)i * 2.39996322972865332;
        const Vec3 q = p + (t1 * std::cos(th) + t2 * std::sin(th)) * rr;
        // The chord of the ball along n through q. Outside it there is no ball to be in.
        const double half = std::sqrt((r2 - rr * rr > 0.0) ? (r2 - rr * rr) : 0.0);
        if (!(half > 0.0)) continue;
        const Vec3 start = q - n * half;
        const double span = 2.0 * half;
        double t0 = eps;
        int layers = 0;
        for (; layers < maxLayers; ++layers) {
            Ray ray{start, n};
            Hit h = sc.closestHit(ray, t0, nullptr, /*skipHair=*/false,
                                  /*skipCamHidden=*/true);
            if (!h.valid || h.t > span) break;
            // THE FACING TEST MUST BE THE PHOTON QUERY'S, EXACTLY. That query keeps a photon
            // when `dot(ph.n, h.n) >= 0.5` — SIGNED, against consistently-oriented normals —
            // so a back face is rejected and contributes no photons. The first version of
            // this loop used `fabs(dot(h.n, n))`, which is wrong twice over: `Hit::n` is
            // oriented AGAINST the ray, so its sign carries no information here, and the
            // absolute value then accepts back faces the query throws away. On a closed
            // solid the march counts entry AND exit, roughly doubling the footprint, and
            // dividing by it darkens everything — which is exactly what the gallery_rain A/B
            // showed, including a 3.5-point move on a null whose footprint is 1.008 and
            // where the correction is algebraically inert.
            //
            // `Hit::ng` is the RAW geometric normal, independent of which way the ray came,
            // so unlike `Hit::n` it can be compared with the gather point's normal at all.
            //
            // The comparison is SIGNED and uses `orient` from the calibration above, so it
            // reproduces the query on all three shapes that matter: a closed solid's far face
            // opposes and drops out, stacked thin shells all agree and are all counted, and an
            // authored quad works whichever way its `u x v` happens to point.
            const double c = dot(h.ng, n) * orient;
            if (c >= 0.5) area += cellArea / c;   // same-facing: what the photon query keeps
            t0 = h.t + eps;
        }
        if (layers >= maxLayers && incomplete) *incomplete = true;
    }
    return area;
}
