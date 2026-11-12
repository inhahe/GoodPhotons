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

// `curveTmin`: ignore CURVE primitives closer than this along the ray, and nothing else.
// A ray leaving a hair fiber on its far side has to clear the strand's own body (the
// near-field BCSDF places the TT/TRT exit at the entry point), but pushing the ORIGIN
// forward to do that also hides any non-fiber surface in the gap -- notably the skin the
// fiber grows out of. Carrying the distance as a curve-only tmin skips exactly the fibers
// it was meant to skip. Zero on every ordinary ray, and a fresh Ray is built per bounce,
// so it resets itself.
struct Ray { Vec3 o, d; double curveTmin = 0.0; };

inline double curveMin(const Ray& r, double tmin) {
    return tmin > r.curveTmin ? tmin : r.curveTmin;
}

// Slab clip of the ray o + t d, t in [t0, t1], to the box [lo, hi]: the overlap [ta, tb], or
// false when it is empty or a single point. An axis the ray runs parallel to (|d| < 1e-12)
// either contains the origin or misses.
inline bool clipRayBox(const Vec3& lo, const Vec3& hi, const Vec3& o, const Vec3& d,
                       double t0, double t1, double& ta, double& tb) {
    for (int a = 0; a < 3; ++a) {
        const double oa = (&o.x)[a], da = (&d.x)[a];
        const double mn = (&lo.x)[a], mx = (&hi.x)[a];
        if (std::fabs(da) < 1e-12) { if (oa < mn || oa > mx) return false; continue; }
        const double inv = 1.0 / da;
        double s0 = (mn - oa) * inv, s1 = (mx - oa) * inv;
        if (s0 > s1) std::swap(s0, s1);
        t0 = std::max(t0, s0); t1 = std::min(t1, s1);
        if (t0 > t1) return false;
    }
    ta = t0; tb = t1;
    return tb > ta;
}

// The distance from the origin of a triangle's farthest vertex: its Hit::pscale.
inline double triReach(const Vec3& a, const Vec3& b, const Vec3& c) {
    return std::sqrt(std::max(dot(a, a), std::max(dot(b, b), dot(c, c))));
}

// A triangle's unit face normal turned to the side of nsum, the sum of its vertex normals --
// the side intersectTri turns a hit's ng to (it turns it to the interpolated normal's, which is
// the same side wherever the vertex normals agree); 0 for a degenerate face.
inline Vec3 faceNormalToward(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& nsum) {
    const Vec3 fc = cross(b - a, c - a);
    const double l = std::sqrt(dot(fc, fc));
    return (l > 0.0) ? fc * ((dot(fc, nsum) < 0.0 ? -1.0 : 1.0) / l) : Vec3{0, 0, 0};
}

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
    // Per-triangle TANGENT frame for tangent-space normal mapping (C6). `tangent`
    // is the surface direction of increasing texture u (unit, orthogonal to gn);
    // `bitangentSign` (+1/-1) encodes the handedness so the bitangent is
    // cross(N, tangent)*bitangentSign (handles mirrored UVs). Derived once in
    // finalize() from the UV gradient (Lengyel's method); a degenerate/zero-area UV
    // parameterization falls back to an arbitrary basis around gn.
    Vec3   tangent{1, 0, 0};
    double bitangentSign = 1.0;
    // MEAN CURVATURE of the interpolated shading-normal field over this face (O3),
    // in 1/length units, with the OUTWARD (gn-side) convention: a convex bulge is
    // positive, a concave pit negative, a plane exactly 0. A unit sphere reads +1.
    //
    // Derived once here rather than per hit because it is genuinely CONSTANT over the
    // face: barycentric interpolation makes n(p) linear in p, so its differential dn
    // — and hence the shape operator and its trace — does not vary across the triangle.
    // Per-hit cost is therefore a single copy, not a computation.
    //
    // Piecewise-constant across the mesh, but NOT visibly faceted where it matters: on a
    // smoothly-normalled mesh neighbouring faces see nearly the same dn, so curvature
    // varies smoothly too. It jumps only across a crease — where the surface really does
    // have discontinuous curvature. A flat-shaded face (n0==n1==n2==gn) reads exactly 0,
    // which is the honest answer: a facet is flat, and a faceted mesh carries its
    // curvature on the edges as a distribution the field cannot represent.
    double curvature = 0.0;
    // Per-vertex COLOUR, as an index of the first of three consecutive linear-RGB
    // entries in Scene::vertColors (v0, v1, v2), or -1 for the overwhelming majority of
    // triangles that have none. Fed by the mesh formats that carry vertex colour: PLY's
    // red/green/blue, glTF's COLOR_0, OBJ's extended `v x y z r g b`, an FBX colour layer.
    //
    // Stored as an INDEX rather than inline because Tri is already ~200 bytes and sits in
    // the BVH's hot leaf data; nine floats on every triangle to serve the few meshes that
    // have any would be a poor trade. Stored as RGB rather than as fitted spectral
    // coefficients because the two consumers want different things and RGB is what both
    // can start from: the rasterizer IS rgb and uses it directly, while the spectral
    // tracer interpolates the colour and then looks the coefficients up per hit
    // (upsample::coeffLut / stochJhCoeff — the same table stochastic tiling uses for the
    // same reason, a colour that exists at no vertex needing a spectrum at a hit).
    // Interpolating the colour and looking up is also strictly better than interpolating
    // pre-fitted coefficients would be: it is the colour that varies linearly across a
    // face, not the sigmoid coefficients that reproduce it.
    int vcol = -1;
    // Motion keys, as an index into Scene::motionSets, or -1 for a triangle that does not move.
    // The vertices and normals above are the triangle at the frame's own instant; the keys
    // carry it across the scene's motion interval (Scene::movedTri).
    int motion = -1;
    // The pattern frame its patterns are read in (Scene::patFrames), or -1: world space.
    int patFrame = -1;
    // A deforming triangle (ROADMAP 11 stage E): an offset into Scene::vertexKeys, where its shape
    // at each of its `vkeyCount` keys -- spread evenly over the motion interval -- is kept, its
    // three corners and then its three normals, six entries a key; or -1. The fields above are
    // its shape at the frame's instant, which is one of the keys.
    int vkeys = -1;
    int vkeyCount = 0;
    // On the boundary of a closed solid (main.cpp markSolidFaces, over watertight::markSolidFaces):
    // 1 with its normal out, so a ray meeting it from behind (Hit::back) is leaving the solid; 2 with
    // it in. 0 on any face that cannot say. Set before a mode-M or mode-S batch, and read only by
    // its footprint probes.
    unsigned char solidFace = 0;
    // triReach of the vertices, baked here for the hit to copy (Hit::pscale).
    double reach = 0.0;
    void finalize() {
        gn = normalize(cross(v1 - v0, v2 - v0));
        reach = triReach(v0, v1, v2);
        if (dot(n0, n0) < 1e-12) n0 = gn;
        if (dot(n1, n1) < 1e-12) n1 = gn;
        if (dot(n2, n2) < 1e-12) n2 = gn;
        // Tangent from the UV gradient (Lengyel). Solve for the world direction along
        // which texture-u increases, then Gram-Schmidt it against gn and record the
        // bitangent handedness about the winding normal gn. A hit's frame is built about the
        // normal on its own side, which intersectTri turns ng to per hit, so it turns the
        // sign with it (Hit::bitangentSign): a mesh wound against its `vn`, or a face whose
        // vertex normals straddle it, keeps dP/dv. Falls back to onb(gn) when UVs are
        // degenerate.
        Vec3 e1 = v1 - v0, e2 = v2 - v0;
        double du1 = uv1.x - uv0.x, dv1 = uv1.y - uv0.y;
        double du2 = uv2.x - uv0.x, dv2 = uv2.y - uv0.y;
        double det = du1 * dv2 - du2 * dv1;
        if (std::fabs(det) > 1e-20) {
            double r = 1.0 / det;
            Vec3 T = (e1 * dv2 - e2 * dv1) * r;
            Vec3 B = (e2 * du1 - e1 * du2) * r;
            T = T - gn * dot(gn, T);               // orthogonalize against the normal
            double tl = std::sqrt(dot(T, T));
            if (tl > 1e-12) {
                tangent = T * (1.0 / tl);
                bitangentSign = (dot(cross(gn, tangent), B) < 0.0) ? -1.0 : 1.0;
            } else { Vec3 bb; onb(gn, tangent, bb); bitangentSign = 1.0; }
        } else { Vec3 bb; onb(gn, tangent, bb); bitangentSign = 1.0; }
        // Mean curvature H = 1/2 * trace(dn restricted to the tangent plane).
        // dn maps e1 -> n1-n0 and e2 -> n2-n0 (barycentric interpolation is linear, so
        // this determines dn on the whole tangent plane). The trace of a linear map is
        // basis-independent but must be taken in a DUAL basis, since {e1,e2} is not
        // orthonormal: trace = e1*·dn(e1) + e2*·dn(e2) where {e1*,e2*} is dual to
        // {e1,e2}. Inverting the 2x2 Gram matrix gives the closed form below, which
        // also drops dn's out-of-plane component automatically (the dual vectors are
        // in-plane), so no explicit projection is needed.
        //
        // Sanity anchors: a unit sphere with exact outward vertex normals has
        // n = p, so dn = identity, trace = 2, H = 1. A plane has n0==n1==n2, so
        // dn = 0 and H = 0. Scaling the mesh by s scales H by 1/s, as curvature must.
        {
            Vec3 dn1 = n1 - n0, dn2 = n2 - n0;
            double gA = dot(e1, e1), gB = dot(e1, e2), gC = dot(e2, e2);
            double gdet = gA * gC - gB * gB;                 // = 4 * area^2 > 0
            if (gdet > 1e-24) {
                double trace = (gC * dot(e1, dn1) - gB * dot(e2, dn1)
                              - gB * dot(e1, dn2) + gA * dot(e2, dn2)) / gdet;
                curvature = 0.5 * trace;
            } else {
                curvature = 0.0;                             // degenerate sliver
            }
        }
    }
};

struct Sphere {
    Vec3 c; double r = 1.0;
    int matId = 0;
    int motion = -1;   // Scene::motionSets index, or -1; c and r are the frame's instant (Scene::intersectMovingSphere)
    int patFrame = -1;   // the pattern frame its patterns are read in (Scene::patFrames), or -1
    // Its own axes, as an index of three consecutive world-space unit vectors in
    // Scene::sphereAxes, or -1: the world's. Its lat-long UV and tangent are read in them, so a
    // sphere in a rotated group -- or turned by its motion keys -- wears its texture turned with
    // it. A side table, not inline: the record sits in every BVH leaf test, and nearly every
    // sphere is unturned.
    int frame = -1;
};


// Where the current path is in time, as a fraction of the scene's motion interval (0 at its
// start, 1 at its end), or negative for "the frame's instant", which is what every primitive's
// own fields hold. Set per path by MotionTimeScope; read by Scene's queries (closestHit,
// occluded, shadowTransmittance) whenever a primitive carries motion keys. Thread-local because
// a CPU thread traces one path at a time and every query of that path must see its time, while
// the ~200 query sites stay as they are.
inline thread_local double g_motionU = -1.0;

// A whole pass's time, for the estimators that share light data across paths (photon maps,
// SPPM, VCM's merges): under a blur of a moving scene each pass -- an epoch of mode M, a pass of
// modes S and U -- is rendered at one time, and averaging the passes integrates the shutter. The
// caller sets it before the pass's workers start and clears it after (PassTimeScope); each
// worker opens a MotionTimeScope at it. Negative: no pass time.
inline double g_passMotionU = -1.0;

// Sets g_motionU for the lifetime of the scope (one path), restoring the previous value after.
struct MotionTimeScope {
    double prev;
    explicit MotionTimeScope(double u) : prev(g_motionU) { g_motionU = u; }
    ~MotionTimeScope() { g_motionU = prev; }
    MotionTimeScope(const MotionTimeScope&) = delete;
    MotionTimeScope& operator=(const MotionTimeScope&) = delete;
};

// Sets g_passMotionU for a pass (on the thread that starts its workers), clearing it after.
struct PassTimeScope {
    double prev;
    explicit PassTimeScope(double u) : prev(g_passMotionU) { g_passMotionU = u; }
    ~PassTimeScope() { g_passMotionU = prev; }
    PassTimeScope(const PassTimeScope&) = delete;
    PassTimeScope& operator=(const PassTimeScope&) = delete;
};

// A moving group's keys: `keys` (at least 2) transforms at increasing times through the scene's
// motion interval (fractions, the first 0 and the last 1), starting at `first` in
// Scene::motionXf / motionT. `shape` is what the keys do to lengths and angles, which the lights
// they move read (Scene::lightAreaScale): 0 every key rigid, 1 every key a similarity (a rigid
// motion with one uniform scale), 2 any affine.
struct MotionKeys { int keys = 0; int first = 0; int shape = 0; };
enum MotionShape { kMotionRigid = 0, kMotionSimilar = 1, kMotionAffine = 2 };

// The transform at the fraction u of the motion interval, of `n` (>= 2) keys at the increasing
// times `t` (t[0] = 0, t[n-1] = 1): the two keys bracketing u blended entry by entry (Affine::
// lerp), so a point moves in a straight line from key to key; u is clamped to [0, 1]. A u
// exactly on a key's time gives that key exactly.
inline Affine keyedAffineAt(const Affine* keys, const double* t, int n, double u) {
    if (!(u > 0.0)) return keys[0];
    if (u >= 1.0) return keys[n - 1];
    int lo = 0, hi = n - 1;                 // t[lo] <= u < t[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (t[mid] <= u) lo = mid; else hi = mid;
    }
    return keys[lo].lerp(keys[hi], (u - t[lo]) / (t[hi] - t[lo]));
}

// The pair of keys bracketing the fraction u of the motion interval, and how far between them u
// lies: keys i and i + 1, fraction f in [0, 1]. u is clamped to [0, 1].
inline void motionBracket(int keys, double u, int& i, double& f) {
    const double s = (u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u)) * (keys - 1);
    i = (int)s;
    if (i > keys - 2) i = keys - 2;
    f = s - i;
}

struct Hit {
    double t = DBL_MAX;
    bool valid = false;
    // The medium the ray met this surface from: the top of its walk's MediumStack (Scene::mats'
    // index), -1 for air or where no walk set it. A rough interference coating interferes
    // against its index (iridescenceR), as the smooth one does against the stack's.
    int extMat = -1;
    Vec3 p, n, ng;         // n = oriented against ray; ng = raw geometric normal
    int matId = 0;
    int sensorId = -1;
    double u = 0, v = 0;   // interpolated surface texture coordinates
    double fieldVal = 0;   // implicit field value at the hit (~0 on a surface; 0 for
                           // non-implicit hits). Exposed to procedural patterns as `f`.
    // Surface tangent frame at the hit, for tangent-space normal mapping (C6).
    // `tangent` is the world direction of increasing texture-u; `bitangentSign`
    // (+1/-1) gives the bitangent handedness (B = cross(n, tangent)*bitangentSign).
    Vec3   tangent{1, 0, 0};
    double bitangentSign = 1.0;
    // Local fiber radius, in world units, when the hit is on a `curve` / `fur` strand;
    // 0 on every other primitive ("not a fiber"). Needed by the hair BCSDF (§P3), whose
    // TT lobe legitimately exits the FAR side of the strand: the near-field model
    // approximates that exit as happening at the entry point, so a shadow / camera
    // connection through the fiber has to start past the strand's own body instead of
    // being occluded by it. Nothing else reads this, and the intersector already has the
    // interpolated radius in hand, so it costs one store.
    double fiberRadius = 0.0;
    // The reach of the primitive the hit lies on: the distance from the origin of its
    // farthest point (a triangle's farthest vertex, Tri::reach; a sphere's |c| + |r|; a curve
    // segment's farther end plus its radius; through an instance, a bound), or 0 where the
    // point's own distance serves (an isosurface, whose rounding is the point's). A pattern
    // reads a hit a hair off the surface (pattern.h patNudgedPoint), sized by this: the hit
    // point's rounding and the float noise of an exported mesh's vertices scale with the
    // coordinates the point was computed from, not with the point's own distance, which can
    // be far smaller on a big triangle; and over one primitive the reach is constant, so the
    // hair is too. Every intersector writes it (the record is reused across candidates).
    double pscale = 0.0;
    // The ray arrived from behind ng, so the intersector turned n (and curv) to face it. The
    // intersector's own decision, which every intersector writes (the record is reused): read
    // back off n and ng it would be wrong wherever a normal map has tipped n across the
    // surface's plane, or a non-uniform instance scale has turned an interpolated normal more
    // than 90 degrees from the face normal (the inverse transpose keeps the side of a
    // direction, dot(d, n), but not of one normal against another). An instance keeps its
    // local one, which that same property makes the world one too.
    bool back = false;
    // The face bounds a closed solid, so `back` says which way the ray crosses it: +1 with the
    // solid's normals out (from behind is leaving: an analytic sphere, a triangle Tri::solidFace
    // marks 1), -1 with them in (from behind is entering), 0 where the side a ray meets a face from
    // says nothing of inside and out (an open quad, a curve, an isosurface). Every intersector
    // writes it (the record is reused).
    signed char solid = 0;
    // The surface's material lets light cross it (Material::patOwnSide), so its patterns see
    // its own side, ng's: their normal and curvature turned there when `back`, its cavity
    // probed about ng (patCtxFromHit). Decided once, by Scene::markPatSide at the closest-hit
    // choke points, with the surface's own material -- the rule is the surface's, so a mix
    // that later resolves to an opaque part keeps it for that part's patterns too.
    bool patOwnSide = false;
    // Where the hit's patterns are read (Scene::patCtxFromHit): the motion set of the moving
    // primitive it is on, or -1 -- its patterns read the point carried back to the frame's
    // instant, so a texture travels with its object -- and the pattern frame of the primitive,
    // or -1 for world space. Every intersector writes both (the record is reused).
    int motionSet = -1;
    // The deforming triangle a hit with a time is on (Tri::vkeys), or -1: its patterns read the
    // point carried back to where it sat at the instant by its barycentrics (patCtxFromHit),
    // the intersector's own of v1 and v2 (deformB1, deformB2), so the instant's point is the
    // exact edge form intersectTri takes a hit by.
    int deformTri = -1;
    double deformB1 = 0.0, deformB2 = 0.0;
    int patFrame = -1;
    // Barycentric-interpolated per-vertex colour at the hit (linear RGB), when the
    // triangle carried one. Resolved by the intersector, which is the only place that
    // has both the triangle and the barycentrics; everything downstream reads it off the
    // Hit exactly as it reads u/v.
    bool   hasVcol = false;
    double vcolR = 1.0, vcolG = 1.0, vcolB = 1.0;
    // Mean curvature at the hit, 1/length, exposed to procedural patterns as `curv` (O3).
    // Signed RELATIVE TO THE SIDE BEING SHADED: the intersector negates it whenever it
    // flips the normal to face the ray, so a surface bulging toward the viewer is always
    // positive and a pit is always negative — the same sphere reads +1/R from outside and
    // -1/R from inside, which is what "how concave is it here" means locally.
    // Analytic where the shape knows its own curvature (sphere 1/R, curve/round-cone
    // 1/(2R)), per-face from the interpolated shading normals on a mesh that carries
    // `vn`, and honestly 0 where there is no curvature to report: a quad or other flat
    // facet, a FLAT-shaded mesh (no normal field to differentiate), and an implicit
    // isosurface (which would need the field's Hessian — see known-issues.md).
    double curv = 0.0;
    // Cavity (O3 stage 2) — the blocked fraction of a short hemispherical probe at this
    // hit, in [0,1], exposed to patterns as `cavity`. Unlike `curv` this is NOT filled
    // by the intersector: it is non-local, so computing it needs the whole scene, which
    // an intersector has no business traversing. It is filled LAZILY at shading time by
    // patCtxFromHit(), and cached here because a single shading point builds several
    // PatCtxs (a mix weight, a roughness map, a reflect map each ask), and an N-ray
    // probe per ask would be paid over and over for one identical answer.
    //
    // `mutable` so the lazy fill works through the `const Hit&` every shading helper
    // takes. Only ever written by cavityAt(); nothing else may touch it.
    mutable double cavity = 0.0;
    mutable bool   cavityDone = false;
    // Shading footprint (O8 stage 2) — the world-space DIAMETER of the surface patch this
    // one shading sample stands for, exposed to patterns as `fw` and meant to be handed
    // straight to `fnoise`. 0 means "unknown", which patterns must read as "do not filter".
    //
    // Filled by the RENDERER, not the intersector, and that is the whole reason it lives
    // here rather than being derived at shading time: it is not a property of the surface
    // at all but of the ray that arrived — of the camera's pixel cone, the distance it
    // travelled and the obliquity it landed at (cameraFootprint in camera.h). The
    // intersector has no idea which of those it is serving. So the deterministic samplers
    // set it on their PRIMARY hits and leave it 0 everywhere else: 0 on every secondary
    // bounce (no ray differentials are propagated through a scatter), and 0 throughout the
    // forward photon modes, whose pixels already area-average over the footprint by
    // scattering millions of hit points across it.
    double fw = 0.0;
};

// Geometric surface normal on the side being shaded: the side the ray came from, as the
// intersector decided it (Hit::back). For a flat triangle or an analytic sphere the shading
// and geometric normals coincide, so this returns exactly h.n's direction. It differs when
// a smooth/interpolated shading normal (authored `vn` or crease-smoothing) diverges from the
// true geometry -- the case where next-event estimation and BSDF continuations must be
// clamped to the geometric hemisphere to stop light leaking through the back of the surface
// (the shading-normal problem). Not read off h.n: where a normal map or a non-uniform
// instance scale has turned n across the surface's plane, n points to the other side, and
// spawn offsets, the terminator and the cavity probe would disagree about one hit. A hit
// made by hand sets `back` as an intersector would (and a copy whose n is flipped, flips it).
inline Vec3 orientedGeoN(const Hit& h) {
    return h.back ? Vec3{-h.ng.x, -h.ng.y, -h.ng.z} : h.ng;
}

// The same hit seen from its other side -- a two-sided transmit lobe's back hemisphere: n,
// curv and back turned together, as an intersector turns them for a ray from there.
inline Hit otherSide(const Hit& h) {
    Hit o = h;
    o.n = Vec3{-h.n.x, -h.n.y, -h.n.z};
    o.curv = -h.curv;
    o.back = !h.back;
    return o;
}

// What a sampled continuation is, which decides the side of the true surface it must leave on: a
// reflection goes back to the side the ray arrived from, a transmission through to the other.
// Only the sampler knows which it drew. The shading normal cannot tell: a mirror reflection about
// a tilted n of a ray that arrived below n's plane lies below that plane as well. `Free` is a
// direction no lobe drew about n -- a pass-through, a dielectric's or a grating's (built about the
// true normal), a fiber's -- which leavesWrongSide never rejects.
enum class Lobe { Reflect, Transmit, Free };

// Does `w`, drawn as a `lobe` at a hit whose true normal on the arrival side is `ngo`
// (orientedGeoN), leave on the wrong side of the true surface? A smoothed or normal-mapped normal
// puts a thin wedge of each lobe there, and a normal tipped across the plane (Hit::back says why
// that happens) most of it. Such a continuation would travel inside the geometry and bring back
// light from behind it, so every walk absorbs it, as Cycles discards such samples: it contributes
// nothing, where letting it through leaks. Absorbing without renormalizing leaves every other
// direction's density, and so MIS, as it was. Never true of a `Free` direction.
inline bool leavesWrongSide(const Vec3& w, const Vec3& ngo, Lobe lobe) {
    if (lobe == Lobe::Free) return false;
    const double c = dot(w, ngo);
    return (lobe == Lobe::Reflect) ? !(c > 0.0) : !(c < 0.0);
}

// The continuation from hit h along `w`, drawn as a `lobe`: false where it leaves on the wrong
// side (leavesWrongSide) -- the caller absorbs it -- else `ray`, its origin nudged 1e-6 off the
// surface toward the side w leaves on: along n where n and the true normal agree about that side,
// else along the true normal (an origin along n would start behind the surface).
inline bool continueAlong(const Hit& h, const Vec3& w, Lobe lobe, Ray& ray) {
    const Vec3 ngo = orientedGeoN(h);
    if (leavesWrongSide(w, ngo, lobe)) return false;
    const bool nAgrees = dot(h.n, ngo) > 0.0 && (dot(w, h.n) > 0.0) == (dot(w, ngo) > 0.0);
    const Vec3 off = nAgrees ? h.n : ngo;
    ray = Ray{h.p + off * (dot(w, off) > 0.0 ? 1e-6 : -1e-6), w};
    return true;
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

// The ARRIVAL half of the same adjoint, |cos(w,ns)| / |cos(w,ng)|, for light arriving along -`w`
// (`w` pointing back toward its source) at a surface whose shading normal is `ns` and geometric
// normal `ng`. A density estimate of photons reads flux per GEOMETRIC area, while mode R shades with
// the shading cosine; this turns the one into the other. Every estimator that reads arrivals takes
// it: a photon deposit of modes M and S (and the continuation drawn about ng where ns tilts,
// Renderer::arrivalAdjoint), and the point merges of modes U and J. 1 where ns == ng up to rounding;
// a grazing denominator (measure zero, ~0 flux) falls back to 1, as shadingAdjointCorr's does,
// rather than lend a near-tangent photon a weight of 1e9. Device twin dArrivalAdjointCorr.
inline double arrivalAdjointCorr(const Vec3& w, const Vec3& ns, const Vec3& ng) {
    const double denom = std::fabs(dot(w, ng));
    if (denom <= 1e-8) return 1.0;
    return std::fabs(dot(w, ns)) / denom;
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
// `ns` = shading normal, `ng` = geometric normal on the side being shaded -- the side the ray
// came from (orientedGeoN) -- `wi` = direction toward the light/connection. Callers gate on the
// shading cosine first, so cos(Ns,wi) > 0 in practice. Where ns has been tipped across the
// surface's plane (cos(Ng,Ns) <= 0: a normal map, a non-uniform instance scale), ng and ns lie
// on opposite sides and the softening means nothing: the plain geometric step applies, light
// from the shaded side passing and light from behind blocked. **Exactly 1 when Ns == Ng** (flat tris, analytic spheres): then
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
    if (cosNgNs <= 0.0) return 1.0;                // ns tipped across the plane: the geometric step
    double denom = dot(ns, wi) * cosNgNs;
    if (denom <= 1e-8) return 1.0;                 // degenerate (grazing / near-perpendicular): no softening
    double g = cosNgWi / denom;
    if (g >= 1.0) return 1.0;                      // fully lit -> no darkening
    return g * g * (1.0 - g) + g;                  // Chiang cubic: -g^3 + g^2 + g
}

// Does a merge refuse a photon arriving along `wi` at a camera vertex whose true normal on the
// camera's side is `ngo`? A one-sided surface takes no light from behind itself -- a connection
// there takes none either (shadowTerminatorG is 0 behind the true surface) -- while the photons of
// a zero-thickness sheet's other face lie within the merge radius, and a tilted or tipped shading
// normal would let the BSDF take them for reflections. A two-sided one (`twoSided`: isTwoSidedMat)
// takes light from both sides, as its connections do, so it refuses none. Only the support: the
// terminator's softening belongs to a connection, which a merge, like a sampled continuation, is
// not.
inline bool mergeFromBehind(bool twoSided, const Vec3& wi, const Vec3& ngo) {
    return !twoSided && !(dot(wi, ngo) > 0.0);
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

// Inlining control for the hot intersection path: FT_FORCEINLINE for what a BVH leaf must not
// call, FT_NOINLINE for a rare branch kept out of it so the common one stays small.
#if defined(_MSC_VER)
#define FT_FORCEINLINE __forceinline
#define FT_NOINLINE __declspec(noinline)
#else
#define FT_FORCEINLINE inline __attribute__((always_inline))
#define FT_NOINLINE __attribute__((noinline))
#endif

// Where a ray crosses a triangle: the distance along it, and the barycentric weights of the
// triangle's v0, v1, v2 there.
struct TriCrossing { double t, b0, b1, b2; };

// The watertight test's geometric half: whether the ray meets the triangle (v0, v1, v2) at a
// distance in [tmin, tmax), and if so where. intersectTri builds its hit on this; a moving
// triangle is tested by it before being rebuilt at the path's time (Scene::intersectTriNow).
FT_FORCEINLINE bool triCross(const TriShear& sh, const Ray& r, const Vec3& v0, const Vec3& v1,
                     const Vec3& v2, double tmin, double tmax, TriCrossing& x) {
    const int kx = sh.kx, ky = sh.ky, kz = sh.kz;
    // Triangle vertices relative to the ray origin.
    Vec3 A = v0 - r.o, B = v1 - r.o, C = v2 - r.o;
    // Shear + scale so the ray direction becomes the +z axis; keep the xy of each vertex.
    double Ax = A[kx] - sh.Sx * A[kz], Ay = A[ky] - sh.Sy * A[kz];
    double Bx = B[kx] - sh.Sx * B[kz], By = B[ky] - sh.Sy * B[kz];
    double Cx = C[kx] - sh.Sx * C[kz], Cy = C[ky] - sh.Sy * C[kz];
    // Scaled barycentric edge functions (U,V,W weight v0,v1,v2 respectively).
    //
    // THESE THREE LINES MUST NOT BE FMA-CONTRACTED. The watertight guarantee is that the two
    // triangles sharing an edge evaluate it from bitwise identical operands in opposite order,
    // so their edge functions are exact negatives and a ray dead-on the edge is claimed by
    // exactly one of them. Contracting `a*b - c*d` into `fma(a, b, -(c*d))` keeps one product
    // exact and rounds the other, so on an exact tie both sharers get the SAME small residual
    // and, if its sign is the minority one, both reject -> the surface cracks along the edge.
    // Safe as built: MSVC's default /fp:precise does not contract and the build sets no /fp:
    // or /arch: flag. If that ever changes, switch to explicitly-rounded products the way the
    // CUDA twin does (render_cuda.cu dCrossRn) -- it had to, because nvcc defaults to
    // -fmad=true and this exact bug cracked the cornell box's quad diagonals on the GPU.
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
    if (t < tmin || t >= tmax) return false;
    x.t = t;
    x.b0 = U * invDet; x.b1 = V * invDet; x.b2 = W * invDet;   // barycentric of v0,v1,v2
    return true;
}

// Watertight test using a precomputed per-ray shear. Callers in a BVH leaf loop should
// build the shear once (makeTriShear(lr.d)) and pass it here for every triangle.
inline bool intersectTri(const TriShear& sh, const Ray& r, const Tri& tri,
                         double tmin, Hit& hit, const float* vertColorTable = nullptr) {
    TriCrossing x;
    if (!triCross(sh, r, tri.v0, tri.v1, tri.v2, tmin, hit.t, x)) return false;
    const double t = x.t, b0 = x.b0, b1 = x.b1, b2 = x.b2;
    // The hit point from the triangle's own vertices, v0 + b1 (v1 - v0) + b2 (v2 - v0), not
    // from the ray (o + d t): on a surface lying at x, y or z = c every hit then has exactly
    // c there -- the edge differences are exactly 0 -- so a pattern that steps at c (a
    // checker floor at y = 0) reads one side on every ray, on both devices, where o + d t
    // rounds to either side from ray to ray and a pixel averages the two values. The error
    // also scales with the vertices rather than with the ray's travel. (Not
    // b0 v0 + b1 v1 + b2 v2: the b's need not sum to exactly 1, which breaks the exactness.)
    hit.t = t; hit.p = tri.v0 + (tri.v1 - tri.v0) * b1 + (tri.v2 - tri.v0) * b2; hit.valid = true;
    hit.pscale = tri.reach;
    hit.motionSet = -1; hit.deformTri = -1; hit.patFrame = tri.patFrame;
    hit.matId = tri.matId; hit.sensorId = tri.sensorId;
    // Barycentric-interpolated UV and shading normal (matches the old M-T convention:
    // b0,b1,b2 == w0,u,v).
    hit.u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
    hit.v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
    hit.fiberRadius = 0.0;                 // `hit` is a reused record: not a strand
    hit.hasVcol = (tri.vcol >= 0);
    if (hit.hasVcol && vertColorTable) {
        const float* c = vertColorTable + (size_t)tri.vcol * 3;
        hit.vcolR = b0 * c[0] + b1 * c[3] + b2 * c[6];
        hit.vcolG = b0 * c[1] + b1 * c[4] + b2 * c[7];
        hit.vcolB = b0 * c[2] + b1 * c[5] + b2 * c[8];
    } else {
        hit.hasVcol = false;
    }
    Vec3 ns = tri.n0 * b0 + tri.n1 * b1 + tri.n2 * b2;
    double nl = dot(ns, ns);
    ns = (nl > 1e-18) ? ns * (1.0 / std::sqrt(nl)) : tri.gn;
    // --- Which side of this triangle did the ray hit? (PBRT's convention.) -------
    //
    // This used to ask `dot(r.d, ns) < 0` — the SHADING normal. That is the one
    // quantity that cannot answer the question. A smooth mesh's interpolated normal
    // keeps rotating past where the flat facet stopped (that is the entire point of
    // smooth shading), so it grazes through zero in a thin band all around every
    // silhouette WHILE THE FACET IS STILL GEOMETRICALLY FRONT-FACING. In that band the
    // flag called "backface" was really reporting "the smooth normal happened to tilt
    // away", and everything derived from it inverted: `hit.n` pointed into the surface,
    // and `hit.curv` reported a bulge as a pit (O3).
    //
    // Facing is a geometric question, so the flat winding normal answers it. The
    // shading normal's only job here is to fix a *global* winding-vs-`vn` disagreement
    // (an OBJ whose faces are wound opposite to its authored normals), which is what
    // the faceforward below is for — and unlike `dot(r.d, ns)`, `dot(tri.gn, ns)` sits
    // near +/-1 for any sane mesh instead of passing through zero at silhouettes, so it
    // is a stable test. This mirrors PBRT's `Triangle::Intersect`, which likewise
    // faceforwards the geometric normal onto the shading normal and then decides
    // front/back from the geometric one.
    const bool againstWinding = dot(tri.gn, ns) < 0.0;
    Vec3 gn = againstWinding ? Vec3{-tri.gn.x, -tri.gn.y, -tri.gn.z} : tri.gn;
    bool flipped = !(dot(r.d, gn) < 0.0);
    // NB `hit.ng` is deliberately NOT flipped toward the ray, here or in any other
    // primitive: downstream `bool entering = dot(d, h.ng) < 0` (refraction, medium
    // crossing) and `dot(rd, h.ng) < 0` (one-sided emitter tests) read the side OFF it,
    // and a pre-flipped `ng` would make both unconditionally true. Callers that want the
    // geometric normal on the *shaded* side use orientedGeoN(h), which — now that both
    // fields derive from the same `gn` — provably equals `flipped ? -gn : gn`.
    hit.ng = gn;
    hit.n = flipped ? -ns : ns;
    hit.back = flipped;
    hit.solid = tri.solidFace == 1 ? 1 : tri.solidFace == 2 ? -1 : 0;
    hit.tangent = tri.tangent;             // per-triangle tangent (constant across the face)
    // For tangent-space normal mapping (C6): the frame is built about the normal on ng's side,
    // and the triangle's sign is taken about its winding (Tri::finalize).
    hit.bitangentSign = againstWinding ? -tri.bitangentSign : tri.bitangentSign;
    // Curvature follows the shaded side: seen from the back, a bulge IS a pit (O3).
    hit.curv = flipped ? -tri.curvature : tri.curvature;
    return true;
}

// Interface-preserving wrapper: builds the per-ray shear inline. Fine for one-off calls;
// BVH leaf loops hoist makeTriShear(r.d) and use the overload above instead.
inline bool intersectTri(const Ray& r, const Tri& tri, double tmin, Hit& hit,
                         const float* vertColorTable = nullptr) {
    return intersectTri(makeTriShear(r.d), r, tri, tmin, hit, vertColorTable);
}

// A hit's lat-long UV, tangent and bitangent handedness on a sphere, from its outward unit
// normal `ng`, read in the sphere's axes (three consecutive unit vectors) or the world's
// (`axes` null). A mirrored frame (left-handed axes) turns the bitangent, as a mirrored
// instance does (MeshInstance::handSign), so a normal map keeps its v slope.
inline void sphereSurfaceUV(Hit& hit, const Vec3& ng, const Vec3* axes) {
    // Equirectangular (lat/long) UV so spheres can be textured (globes, eyeballs).
    const Vec3 nl = axes ? Vec3{dot(ng, axes[0]), dot(ng, axes[1]), dot(ng, axes[2])} : ng;
    hit.u = 0.5 + std::atan2(nl.z, nl.x) / (2.0 * PI);
    hit.v = 0.5 - std::asin(std::clamp(nl.y, -1.0, 1.0)) / PI;
    // Tangent = the east-pointing longitude direction d/du (C6). d/du of the
    // equirectangular map is proportional to (-sin, 0, cos) of the longitude, i.e.
    // perpendicular to both the sphere's up and the normal; degenerates to its +x at the poles.
    Vec3 T{-nl.z, 0.0, nl.x};
    if (axes) T = axes[0] * T.x + axes[1] * T.y + axes[2] * T.z;
    double tl = std::sqrt(dot(T, T));
    hit.tangent = (tl > 1e-9) ? T * (1.0 / tl) : (axes ? axes[0] : Vec3{1, 0, 0});
    hit.bitangentSign = (axes && dot(cross(axes[0], axes[1]), axes[2]) < 0.0) ? -1.0 : 1.0;
}

// `axes`: the sphere's own (Scene::sphereAxesOf), or null for the world's.
inline bool intersectSphere(const Ray& r, const Sphere& s, double tmin, Hit& hit,
                            const Vec3* axes = nullptr) {
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
    hit.t = t; hit.valid = true;
    Vec3 ng = normalize((r.o + r.d * t) - s.c);
    // On the sphere itself: the textbook discriminant's t carries rounding that grows as the
    // square of the ray origin's distance over the radius (from 100 km, past the hair a pattern
    // reads the hit off it at, patNudgedPoint); the point's direction from the centre does not.
    // |r|: the quadratic only sees r^2, so a negative radius traces as its absolute value.
    hit.p = s.c + ng * std::fabs(s.r);
    hit.pscale = std::sqrt(dot(s.c, s.c)) + std::fabs(s.r);
    hit.motionSet = -1; hit.deformTri = -1; hit.patFrame = s.patFrame;
    hit.ng = ng;
    bool sFlipped = !(dot(r.d, ng) < 0.0);
    hit.n = sFlipped ? -ng : ng;
    hit.back = sFlipped;
    hit.solid = 1;
    // Analytic mean curvature of a sphere: 1/R everywhere, negated when seen from
    // inside (a room-sized sphere is concave to anything standing in it).
    hit.curv = (s.r > 1e-12) ? ((sFlipped ? -1.0 : 1.0) / s.r) : 0.0;
    hit.matId = s.matId; hit.sensorId = -1;
    sphereSurfaceUV(hit, ng, axes);
    hit.fiberRadius = 0.0; hit.hasVcol = false;   // `hit` is a reused record: not a strand, no colours
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
