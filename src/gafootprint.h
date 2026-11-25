#pragma once
// The geometric footprint for M-GATHERAREA, in its own header so both the diagnostic
// (roiboxes.h) and the estimator (photonmap_render.h) can reach it without either
// including the other.
#include <algorithm>
#include <vector>
#include <cmath>
#include "gatherplane.h"

// Whether a face of material `m` (unresolved; a strand's when `fiberRadius` > 0) holds what a gather
// sums, which its footprint must measure the surface of: holdsPhotons for the photon maps of modes M
// and S, holdsMergeVertex for the merges of modes J and U.
using GatherHolds = bool (*)(const Scene& scene, const Material& m, double fiberRadius);

// Whether a footprint (the probe, or the march below) passes through a face of material `m` off a
// curve, rather than stopping on it: one that holds none of the gather's photons (`holds`), a hair
// fiber aside, whose opacity coin the probe tosses. The photons near such a face lie on the surface beyond it,
// and a face within the radius above the tangent plane would otherwise be counted and then
// ball-rejected for its height, though that surface is there (known-issues
// GATHERAREA-TRANSPARENT). A strand is passed whatever its height (the callers). Device twin
// dProbePassesThrough.
inline bool probePassesThrough(const Scene& scene, const Material& m, GatherHolds holds = holdsPhotons) {
    return m.type != MatType::Hair && !holds(scene, m, 0.0);
}

// What a segment from `o` along unit `d`, `len` long, crosses of the faces light does not pass
// (passesLight), curves unseen: `side`, that of the last closed solid's face (Hit::solid) -- +1
// it ends inside that solid, -1 outside, 0 it crosses none -- `net`, how many more solids, closed or
// capped, it enters than leaves, `low` the least that count reaches on the way (below 0: it leaves
// a solid it began in), `net1` the same as `net` over the closed solids alone (Hit::solid +-1, not a
// cap's +-2, whose cap is no face to cross), and `open`, how many faces that cannot say their side
// (up to 15).
// Two segments of the footprint (PROBE-PATH):
//   - from the gather point to a column's top, r above the tangent plane, when the probe crosses no
//     closed solid's face on its way down (its `side`): a column inside a solid taller than the
//     radius standing on the surface is then known to be inside;
//   - the low leg, from just over the gather point (kLowLegFrac) to the column's foot, when what
//     lies above says the column is inside: the parity of the open faces above the plane (the leg's
//     `open` odd to agree), or a closed solid's side where the leg ends deeper in solids than it
//     began (`net`) or leaves one it began in (`low`), or else the chain over the top -- this first
//     segment and the probe's descent -- ends deeper in closed solids (`net1`: a slab too thin for
//     the leg to cross). A cup, a bottomless box of quads or a box standing on the surface walls the
//     column off, one touching another too; a sheet hovering over it -- a table top, a mirror, a
//     lamp's quad -- leaves the way open, the floor under it lit from the sides, and so does a room
//     the gather point lies in itself, a pillar in it or an open quad or cup there no wall.
// The counts are exact because the walk sees every face: it goes window by window from a fixed
// origin, each from the last's end to just past its nearest face -- by the rounding of a distance
// (`eps` * 1e-3), so faces any farther apart keep their order -- and takes every face in it
// (Scene::forEachCrossing), so of two lying on each other -- boxes touching face to face, A's wall
// left and B's entered -- both count. Faces of one kind in one window are one face (a ray through the
// edge two triangles share); a window's faces left count before those entered, so `low` sees the dip
// between two solids touching. `eps`: the gather's scale (r * 1e-4). Device twin dWalkSegment.
struct SegmentCrossings { int side = 0; int open = 0; int net = 0, low = 0, net1 = 0; };
// Hit h's FaceKind along the ray that met it, or -1 for a face light passes (passesLight). Device twin
// dFaceKind.
inline int faceKind(const Scene& sc, const Hit& h) {
    const Material* m = (h.matId >= 0 && h.matId < (int)sc.mats.size()) ? &sc.mats[(size_t)h.matId] : nullptr;
    if (m && passesLight(sc, *m)) return -1;
    if (!h.solid) return kFaceOpen;
    return (h.solid == 1 || h.solid == -1 ? kFaceLeft1 : kFaceLeft2) + (((h.solid > 0) == h.back) ? 0 : 1);
}
// The kinds of the faces lying where a query along `ray` from tmin `t0` met its nearest face, at `t`:
// those with t in [t0, t + tie), bit k one of FaceKind k -- of two faces lying on each other both
// (Scene::forEachCrossing), where the query returns one. `tie`: how near two faces lie to be one place,
// the rounding of their distances (1e-7 of the radius). Device twin dFaceKindsAt.
inline unsigned faceKindsAt(const Scene& sc, const Ray& ray, double t0, double t, double tie) {
    unsigned k = 0;
    double t1 = t + tie;
    if (!(t1 > t0)) return k;
    sc.forEachCrossing(ray, t0, t1, [&](const Hit& h, double&) {
        const int f = faceKind(sc, h);
        if (f >= 0) k |= 1u << f;
    });
    return k;
}
inline SegmentCrossings walkSegment(const Scene& sc, const Vec3& o, const Vec3& d, double len, double eps) {
    SegmentCrossings c;
    const Ray ray{o, d};
    const double tie = eps * 1e-3;   // faces this near lie at one place (faceKindsAt)
    double from = 0.0;
    for (int k = 0; k < 32 && from < len && c.open < 15; ++k) {
        double cut = len, nearest[kFaceKinds] = {len, len, len, len, len};   // a window's nearest of each kind
        sc.forEachCrossing(ray, from, cut, [&](const Hit& h, double& tm) {
            const int f = faceKind(sc, h);
            if (f < 0) return;
            nearest[f] = std::min(nearest[f], h.t);
            tm = std::min(tm, h.t + tie);
        });
        bool in[kFaceKinds];
        for (int j = 0; j < kFaceKinds; ++j) in[j] = nearest[j] < cut;
        const int left = in[kFaceLeft1] + in[kFaceLeft2], entered = in[kFaceIn1] + in[kFaceIn2];
        if (!(left || entered || in[kFaceOpen])) break;
        c.net -= left;
        c.low = std::min(c.low, c.net);
        c.net += entered;
        c.net1 += in[kFaceIn1] - in[kFaceLeft1];
        if (left || entered) c.side = entered ? 1 : -1;
        c.open += in[kFaceOpen];
        from = cut;
    }
    return c;
}

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
// `slack` is the gather's plane-test slack (gatherBandSlack), so a layer counts where the
// photon test would keep a photon on it.
//
// `incomplete` (optional) is set when a ray hit the layer cap, i.e. the chord held more
// surfaces than were counted, so the area is an under-count and the caller should not
// divide by it.
inline double gatherFootprintArea(const Scene& sc, const Vec3& p, const Vec3& n,
                                  double r, int kDisc, int maxLayers, double slack,
                                  bool* incomplete = nullptr, GatherHolds holds = holdsPhotons) {
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
    // on", which is exactly what the photon query's `dot(ph.n, n) >= 0.5` means and is
    // independent of how the scene was authored.
    //
    // Signed (rather than |dot|) is what stops a CLOSED SOLID being counted twice: its far
    // face opposes and drops out, matching the query, which gathers no photons from it. And
    // unlike parity-by-crossing-order this still counts every layer of STACKED THIN SHELLS,
    // whose normals all agree — cloth folds and coat layers are real same-facing surface and
    // the query does take photons from all of them.
    double orient = 1.0;
    {
        // From below and from above, the surface as each ray meets it: against n only where every face
        // met says so -- of two coincident faces facing apart (a room's floor on the ground) either may
        // be the one a ray returns, and the gather's surface is the one facing its way.
        const Hit hc = sc.closestHit(Ray{p - n * (eps * 4.0), n}, eps, nullptr, /*skipHair=*/false,
                                     /*skipCamHidden=*/true);
        const Hit hd = sc.closestHit(Ray{p + n * (eps * 4.0), n * -1.0}, eps, nullptr, /*skipHair=*/false,
                                     /*skipCamHidden=*/true, eps * 8.0);
        const bool against = (hc.valid || hd.valid) && (!hc.valid || dot(hc.ng, n) < 0.0) &&
                             (!hd.valid || dot(hd.ng, n) < 0.0);
        if (against) orient = -1.0;
    }
    const double cellArea = (kPi * r2) / (double)kDisc;
    double area = 0.0;
    // THE PROBE'S CROSSING RULE, per column (gathercoverage.h gatherCoverageRaw, where the why
    // is): faces off the band above the plane are another surface's, and the opaque ones
    // (passesLight) say whether an in-band layer lies inside a solid resting on the surface (the
    // floor under a box), where no photon can be -- so that layer counts nothing. The nearest
    // closed solid's face above a layer (Hit::solid) says whether it lies inside, met from inside
    // meaning inside; photon-holding faces that cannot say count toward a parity, and open sheets
    // that hold none for nothing. A layer no solid face within the window decides lies on the side
    // the column's top, r above the plane, is on (PROBE-PATH: along the segment from the gather
    // point, which stays within r * sqrt(2) of it). Either way a layer is out only where the low leg
    // agrees a wall stands between it and the gather point, as the probe's inside() asks.
    int solidsNear = -1;
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
        // A face counts as area only inside the ball, but as a crossing up to r above the plane,
        // the probe's window: near the disc's edge a box top lies beyond the column's chord and
        // still puts the floor under it inside the box.
        const double top = half + r;
        // The ray relaunches from `tBase` along the chord: at tmin `tl` past it, or, out of a
        // strand, with the curve-only tmin `ct` that clears the tube and still finds a surface
        // just behind it (closestHit then drops the blanket tmin, as every curve-tmin caller does).
        double tBase = 0.0, tl = eps, ct = 0.0;
        int layers = 0;
        // Each in-band layer is decided by the faces above it, as the probe's is: the nearest closed
        // solid's face above it, and the parity of those that cannot say. Climbing, a layer waits
        // in `pend` until the next solid face decides it (`dec` if outside, `drop` if inside),
        // binned by the parity of the crossings below it, so the column's total says which bin has
        // an even count above. A `drop`ped layer stays out only where the low leg agrees a wall
        // stands between it and the gather point (the probe's inside()).
        double pend[2] = {0.0, 0.0}, dec[2] = {0.0, 0.0}, drop[2] = {0.0, 0.0};
        int crossed = 0;
        int climbNet1 = 0;   // closed solids (+-1) the climb enters above the plane, less those it leaves
        int strandSkips = 0;
        for (; layers < maxLayers; ++layers) {
            Ray ray{start + n * tBase, n, ct};
            Hit h = sc.closestHit(ray, tl, nullptr, /*skipHair=*/false,
                                  /*skipCamHidden=*/true);
            if (!h.valid) break;
            h.t += tBase;                            // depth along the chord from `start`
            if (h.t > top) break;
            // THE FACING TEST MUST BE THE PHOTON QUERY'S, EXACTLY. That query (onGatherSurfaceSided)
            // keeps a photon when `dot(ph.n, n) >= 0.5` — SIGNED, against consistently-oriented
            // normals — and it lies in the tangent-plane band, which the test below adds —
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
            const Material* mm = (h.matId >= 0 && h.matId < (int)sc.mats.size()) ? &sc.mats[h.matId] : nullptr;
            // A strand that holds no photon is neither area nor a crossing: the probe passes it, and
            // so does this march, through the tube and, like the probe, for none of the column's
            // layers up to 48 strands, so a coat deeper than the layers still reaches the skin.
            if (mm && h.fiberRadius > 0.0 && mm->type != MatType::Hair && !holds(sc, *mm, h.fiberRadius)) {
                tBase = h.t; tl = 0.0; ct = 2.5 * h.fiberRadius + eps;
                if (strandSkips < 48) { ++strandSkips; --layers; }
                continue;
            }
            const double c = dot(h.ng, n) * orient;
            // Same-facing and in the tangent-plane band: what the photon query keeps.
            const Vec3 dh = h.p - p;
            const double off = dot(dh, n);           // signed height above the tangent plane
            // An opaque face above the plane (no hair strand: the probe's coin decides those). A
            // closed solid's decides the layers below it by its side, in the band or out of it --
            // a small ball's tilted underside lies in the band the tilt widens, and a thin solid's
            // top may too (a coaster's), itself a layer the faces above it decide; one on the
            // plane (a box's own bottom) says nothing, "on" it to within rounding, as in the probe
            // (a resting ball's underside near its contact is above it). One that cannot say counts
            // toward the parity where the probe would cross it: off the band, or a face that holds
            // no photon above the plane, which the probe passes through.
            // A solid's face lying on another decides with it, of two boxes stacked B's bottom with A's top:
            // the faces lying at h (faceKindsAt; none on the plane) say it, a face left there putting the
            // layers below inside, else one entered outside -- and where both lie there, h's own layer, if
            // any, lies outside the solid entered (`pairIn`, below).
            const bool strand = mm && mm->type == MatType::Hair && h.fiberRadius > 0.0;
            const bool opaque = !(mm && passesLight(sc, *mm)) && !strand;
            bool pairIn = false;
            if (opaque && off > 1e-7 * r) {
                const double t0 = std::max(ct > 0.0 ? 0.0 : tl, half + 1e-7 * r - tBase);
                const unsigned k = (h.solid ? 1u << faceKind(sc, h) : 0u) |
                                   faceKindsAt(sc, ray, t0, h.t - tBase, 1e-7 * r);
                if (k & (kFacesLeft | kFacesIn)) {
                    climbNet1 += (int)((k >> kFaceIn1) & 1u) - (int)((k >> kFaceLeft1) & 1u);
                    double* to = (k & kFacesLeft) ? drop : dec;   // left: below is inside; entered: outside
                    to[0] += pend[0]; to[1] += pend[1];
                    pend[0] = pend[1] = 0.0;
                    pairIn = (k & kFacesLeft) && (k & kFacesIn);
                }
            }
            // A face that holds no photon is no area above the plane, where the probe passes it (a
            // glossy ball's steep side, which the tilt-widened band takes in); on the plane it is
            // the surface's height, where the probe ends on it.
            const bool passed = mm && probePassesThrough(sc, *mm, holds) && off > 1e-3 * r;
            if (passed) {
                if (opaque && !h.solid) ++crossed;
            } else if (h.t <= span && withinGatherBand(std::fabs(off), dot(dh, dh), std::fabs(c), slack)) {
                // On the plane, to within rounding, either way up: a face lying on another facing apart
                // (a room's floor on the ground) is one surface to the photon query, whose photons face
                // their light whichever of the two they landed on, and which of the two a ray meets is
                // the traversal's choice (GAGEOM-COINCIDENT). Off it the sign holds.
                const double cf = std::fabs(off) <= 1e-7 * r ? std::fabs(c) : c;
                if (cf >= kGatherFacingCos) pend[crossed & 1] += cellArea / cf;
            } else if (off > 0.0 && opaque && !h.solid) {
                ++crossed;
            }
            if (pairIn) { dec[0] += pend[0]; dec[1] += pend[1]; pend[0] = pend[1] = 0.0; }
            tl = h.t - tBase + eps; ct = 0.0;
        }
        // Layers no solid face above decided lie on the column top's side: along the segment from the
        // gather point to the column's top (PROBE-PATH), walked once a column when asked.
        const int ev = crossed & 1;                  // the bin of layers with an even count above
        double restEv = pend[ev], restOdd = pend[ev ^ 1];
        bool topWalked = false;
        SegmentCrossings topWalk;
        const auto walkTop = [&]() -> const SegmentCrossings& {
            if (!topWalked) {
                topWalked = true;
                if (solidsNear < 0) solidsNear = sc.solidNear(p, n, r * 1.4143) ? 1 : 0;
                if (solidsNear) {
                    const Vec3 o = p + n * eps, colTop = q + n * r;
                    const double len = length(colTop - o);
                    topWalk = walkSegment(sc, o, (colTop - o) * (1.0 / len), len, eps);
                }
            }
            return topWalk;
        };
        if ((restEv > 0.0 || restOdd > 0.0) && walkTop().side > 0) {
            drop[ev] += restEv; drop[ev ^ 1] += restOdd;
            restEv = restOdd = 0.0;
        }
        // The low leg, from just over the gather point to the column's foot, where anything is to be
        // decided by it: a solid's inside stays out where a wall stands between the column and the
        // gather point (the leg ends deeper in solids, or leaves one it began in: a box resting on the
        // surface, not a room the gather point lies in), or the chain over the top ends deeper in
        // closed solids; an odd count above puts a layer inside only where the leg's count of faces
        // that cannot say is odd too (a cup's floor, not the floor under a hovering sheet).
        SegmentCrossings leg;
        if ((drop[0] > 0.0 || drop[1] > 0.0 || dec[ev ^ 1] + restOdd > 0.0) && rr > eps)
            leg = walkSegment(sc, p + n * (kLowLegFrac * r), (q - p) * (1.0 / rr), rr, eps);
        // (the probe's inside(), where the rule's why is)
        const bool walled = (drop[0] > 0.0 || drop[1] > 0.0) &&
                            (leg.net > 0 || leg.low < 0 || walkTop().net1 - climbNet1 > 0);
        double col = dec[ev] + restEv + (walled ? 0.0 : drop[ev]);
        const double odd = dec[ev ^ 1] + restOdd + (walled ? 0.0 : drop[ev ^ 1]);
        if (odd > 0.0 && rr > eps && (leg.open & 1) == 0)
            col += odd;
        area += col;
        if (layers >= maxLayers && incomplete) *incomplete = true;
    }
    return area;
}
