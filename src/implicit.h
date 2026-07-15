// Implicit surfaces: isosurfaces, metaballs, and (smooth) CSG.
//
// A single primitive class — Implicit — owns a small "field expression" that maps a
// world-space point p to a signed distance f(p): f < 0 inside the solid, f > 0
// outside, |f| bounded by the true distance to the surface (a signed distance
// function, SDF). The surface is the zero level set { p : f(p) = 0 }.
//
// The field is stored as a FLAT ARRAY of FieldNodes in POSTFIX (reverse-Polish)
// order and evaluated with a tiny scalar stack — no pointers, no recursion. That
// keeps it trivially GPU-portable (a device evaluator can iterate the same array)
// and cheap to transform/upload. Leaf nodes are analytic primitives (sphere, box,
// torus, plane, cylinder); combinator nodes fold two operands into one
// (union/intersection/difference and their SMOOTH variants, which fillet the seam
// with a smooth-minimum — this is exactly what makes metaballs merge and gives
// POV-Ray-style rounded booleans).
//
// A ray is intersected by SPHERE TRACING: from the entry point, repeatedly step
// forward by |f| (the guaranteed-empty distance to the nearest surface for a true
// SDF), stopping when |f| falls below a surface epsilon. Surface normals come from
// the field gradient ∇f (central differences, tetrahedron stencil).
#pragma once
#include <vector>
#include <cmath>
#include <cfloat>
#include "linalg.h"
#include "geometry.h"
#include "bvh.h"
#include "pattern.h"   // PatNode / patternEval: arbitrary f(x,y,z) expression leaves

// ---------------------------------------------------------------------------
// Field expression: nodes in postfix order.
// ---------------------------------------------------------------------------
enum class FieldOp : int {
    // Leaves — evaluate an analytic SDF at the (leaf-local) query point. p[] holds
    // op-specific parameters (all lengths in the leaf's LOCAL space; the leaf's
    // affine transform maps world<->local and a uniform scale rescales distance).
    Sphere = 0,   // p[0] = radius
    Box,          // p[0..2] = half-extents (hx,hy,hz), p[3] = corner rounding radius (0 = sharp)
    Torus,        // p[0] = major radius R (ring), p[1] = minor radius r (tube), in xz-plane
    Plane,        // p[0..2] = unit normal, p[3] = signed offset (f = dot(pl,n) + off)
    Cylinder,     // p[0] = radius, p[1] = half-height (capped, axis = local y)
    Cone,         // p[0] = bottom radius, p[1] = top radius, p[2] = half-height (axis = local y)
    Expr,         // arbitrary f(x,y,z)=0 isosurface: exprOff/exprN index a shared PatNode
                  // pool (the compiled infix expression). NOT a true SDF, so the marcher
                  // relies on the primitive's Lipschitz bound (max_gradient) to step safely.
    // Combinators — pop two SDF values (a below b on the stack), push one.
    Union,             // min(a,b)
    Intersect,         // max(a,b)
    Difference,        // max(a,-b)  (a minus b)
    SmoothUnion,       // smin(a,b,k)      p[0] = k (blend radius)
    SmoothIntersect,   // smax(a,b,k)      p[0] = k
    SmoothDifference,  // smax(a,-b,k)     p[0] = k
};

inline bool fieldOpIsLeaf(FieldOp op) { return op <= FieldOp::Expr; }

// One node of the field expression. POD (no std:: members) so it uploads to the
// GPU verbatim. Leaf nodes carry their own world->local affine `inv` and the
// uniform scale `scale` (world = scale * local) so a local SDF distance becomes a
// world distance by multiplying by `scale`.
struct FieldNode {
    FieldOp op = FieldOp::Sphere;
    double  p[4] = {1, 0, 0, 0};
    Affine  inv;          // world -> local (leaf only)
    double  scale = 1.0;  // world = scale * local; d_world = d_local * scale (leaf only)
    int     exprOff = -1; // Expr leaf: start index into the owning Implicit's exprNodes pool
    int     exprN   = 0;  // Expr leaf: number of PatNodes in the compiled program
};

// ---- smooth-min / smooth-max (quadratic polynomial blend, Inigo Quilez) -------
// smin(a,b,k) equals min(a,b) when |a-b| >= k, and dips below by up to k/4 in the
// transition band, filleting the seam with radius ~k. k <= 0 falls back to a hard
// min, so the smooth ops degrade gracefully to plain CSG.
inline double sdfSmin(double a, double b, double k) {
    if (k <= 0.0) return a < b ? a : b;
    double h = std::fmax(k - std::fabs(a - b), 0.0) / k;
    return (a < b ? a : b) - h * h * k * 0.25;
}
inline double sdfSmax(double a, double b, double k) {
    return -sdfSmin(-a, -b, k);
}

// ---- leaf SDF evaluation (query point already mapped to the leaf's local space) ----
// `exprPool` backs FieldOp::Expr leaves (the shared PatNode program pool); it may be
// null when the field has no expression leaves.
inline double fieldLeafSDF(const FieldNode& nd, const Vec3& pl, const PatNode* exprPool) {
    switch (nd.op) {
        case FieldOp::Sphere:
            return length(pl) - nd.p[0];
        case FieldOp::Expr: {
            // Evaluate the compiled infix expression f(x,y,z) at the leaf-local point.
            // r = |p_local| is exposed too (a sphere is simply "r - 1"); f/normals are
            // meaningless when DEFINING a surface, so they read 0. The value is a raw
            // field, not a distance — the marcher's Lipschitz bound keeps steps safe.
            PatCtx c;
            c.x = pl.x; c.y = pl.y; c.z = pl.z;
            c.r = std::sqrt(pl.x * pl.x + pl.y * pl.y + pl.z * pl.z);
            return exprPool ? patternEval(exprPool + nd.exprOff, nd.exprN, c) : DBL_MAX;
        }
        case FieldOp::Box: {
            // Rounded box (Inigo Quilez): inset the half-extents by the rounding
            // radius r, take the plain box distance, then subtract r. r = 0 gives a
            // sharp box exactly.
            double r  = nd.p[3];
            double qx = std::fabs(pl.x) - nd.p[0] + r;
            double qy = std::fabs(pl.y) - nd.p[1] + r;
            double qz = std::fabs(pl.z) - nd.p[2] + r;
            double ox = std::fmax(qx, 0.0), oy = std::fmax(qy, 0.0), oz = std::fmax(qz, 0.0);
            double outside = std::sqrt(ox * ox + oy * oy + oz * oz);
            double inside  = std::fmin(std::fmax(qx, std::fmax(qy, qz)), 0.0);
            return outside + inside - r;
        }
        case FieldOp::Torus: {
            double qx = std::sqrt(pl.x * pl.x + pl.z * pl.z) - nd.p[0];
            return std::sqrt(qx * qx + pl.y * pl.y) - nd.p[1];
        }
        case FieldOp::Plane:
            return dot(pl, Vec3{nd.p[0], nd.p[1], nd.p[2]}) + nd.p[3];
        case FieldOp::Cylinder: {
            double dxz = std::sqrt(pl.x * pl.x + pl.z * pl.z) - nd.p[0];
            double dy  = std::fabs(pl.y) - nd.p[1];
            double a   = std::fmin(std::fmax(dxz, dy), 0.0);
            double bx  = std::fmax(dxz, 0.0), by = std::fmax(dy, 0.0);
            return a + std::sqrt(bx * bx + by * by);
        }
        case FieldOp::Cone: {
            // Capped/truncated cone along local y (Inigo Quilez sdCappedCone):
            // radius p[0] at y=-h, radius p[1] at y=+h, half-height h = p[2].
            double rb = nd.p[0], rt = nd.p[1], h = nd.p[2];
            double qx = std::sqrt(pl.x * pl.x + pl.z * pl.z), qy = pl.y;
            double k1x = rt,      k1y = h;
            double k2x = rt - rb, k2y = 2.0 * h;
            double cax = qx - std::fmin(qx, (qy < 0.0) ? rb : rt);
            double cay = std::fabs(qy) - h;
            double k2dot = k2x * k2x + k2y * k2y;
            double tt = (k2dot > 0.0) ? ((k1x - qx) * k2x + (k1y - qy) * k2y) / k2dot : 0.0;
            tt = tt < 0.0 ? 0.0 : (tt > 1.0 ? 1.0 : tt);
            double cbx = qx - k1x + k2x * tt;
            double cby = qy - k1y + k2y * tt;
            double s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
            double da = cax * cax + cay * cay;
            double db = cbx * cbx + cby * cby;
            return s * std::sqrt(std::fmin(da, db));
        }
        default:
            return DBL_MAX;
    }
}

// Evaluate the whole field at a world point via the postfix scalar stack.
// `exprPool` backs FieldOp::Expr leaves (may be null when the field has none).
inline double fieldEval(const FieldNode* nodes, int n, const Vec3& pw, const PatNode* exprPool) {
    double st[64];
    int sp = 0;
    for (int i = 0; i < n; ++i) {
        const FieldNode& nd = nodes[i];
        switch (nd.op) {
            case FieldOp::Union:            { double b = st[--sp], a = st[--sp]; st[sp++] = a < b ? a : b; break; }
            case FieldOp::Intersect:        { double b = st[--sp], a = st[--sp]; st[sp++] = a > b ? a : b; break; }
            case FieldOp::Difference:       { double b = st[--sp], a = st[--sp]; st[sp++] = a > -b ? a : -b; break; }
            case FieldOp::SmoothUnion:      { double b = st[--sp], a = st[--sp]; st[sp++] = sdfSmin(a,  b, nd.p[0]); break; }
            case FieldOp::SmoothIntersect:  { double b = st[--sp], a = st[--sp]; st[sp++] = sdfSmax(a,  b, nd.p[0]); break; }
            case FieldOp::SmoothDifference: { double b = st[--sp], a = st[--sp]; st[sp++] = sdfSmax(a, -b, nd.p[0]); break; }
            default: {  // leaf
                Vec3 pl = nd.inv.apply(pw);
                st[sp++] = fieldLeafSDF(nd, pl, exprPool) * nd.scale;
            }
        }
    }
    return sp > 0 ? st[0] : DBL_MAX;
}

// Surface normal from the field gradient (tetrahedron central differences). eps
// scales with the hit distance so the stencil stays well-conditioned at any range.
inline Vec3 fieldGradient(const FieldNode* nodes, int n, const Vec3& p, double eps,
                          const PatNode* exprPool) {
    const Vec3 k1{ 1, -1, -1}, k2{-1, -1,  1}, k3{-1,  1, -1}, k4{ 1,  1,  1};
    Vec3 g = k1 * fieldEval(nodes, n, p + k1 * eps, exprPool)
           + k2 * fieldEval(nodes, n, p + k2 * eps, exprPool)
           + k3 * fieldEval(nodes, n, p + k3 * eps, exprPool)
           + k4 * fieldEval(nodes, n, p + k4 * eps, exprPool);
    double len = length(g);
    return len > 0.0 ? g / len : Vec3{0, 0, 1};
}

// Ray-march strategy for finding the first zero crossing of the field along a ray.
//   Adaptive — step by max(|f|/lipschitz, minStep). For a true SDF (lipschitz=1) this
//     is sphere-tracing; for a Lipschitz-bounded arbitrary field it is a provably
//     no-overshoot bracketing march (with a correct bound it cannot skip the first
//     crossing). The default; relies on `lipschitz` being a true global bound.
//   Sample — step by a FIXED world distance `sampleStep`, ignoring |f|. Doesn't rely
//     on any Lipschitz bound (good when max_gradient can't be trusted), but a feature
//     thinner than sampleStep between two samples can be missed. POV-Ray's fixed-step
//     `evaluate`/sampling mode.
enum class MarchMethod : int { Adaptive = 0, Sample = 1 };

// Root refinement once a sign change brackets [ta, tb].
//   Bisect — halve the bracket; unconditionally robust, linear convergence.
//   RegulaFalsi — linear (secant) interpolation to the zero, with the Illinois
//     safeguard so the bracket always shrinks; faster on smooth brackets.
enum class RootRefine : int { Bisect = 0, RegulaFalsi = 1 };

// Container shape an expression isosurface is clipped to (POV `contained_by`).
//   Box    — the world AABB `bounds` (default; flat clip planes).
//   Sphere — a world sphere (center/radius); clips along a smooth curved boundary,
//            so an unbounded surface's unavoidable cut reads as a rounded edge
//            instead of hard facets. `bounds` is still the sphere's AABB for BVH.
enum class Container : int { Box = 0, Sphere = 1 };

// ---------------------------------------------------------------------------
// The primitive.
// ---------------------------------------------------------------------------
struct Implicit {
    std::vector<FieldNode> nodes;   // postfix field expression
    std::vector<PatNode> exprNodes; // shared pool backing FieldOp::Expr leaves (f(x,y,z))
    std::string name;               // authored ftsl block name (used to name -export-mesh groups)
    int    matId = 0;
    Aabb   bounds;                  // conservative world AABB (BVH leaf box + ray clip)
    // Container shape + cap policy (expression isosurfaces only; SDF/CSG leaves are
    // closed and never touch their computed bounds, so they leave these at defaults).
    Container container = Container::Box;
    Vec3      sphereCenter{0, 0, 0};   // world center  (Container::Sphere)
    double    sphereRadius = 0.0;      // world radius  (Container::Sphere)
    // Where the solid interior (f<0) is sliced by the container wall, `capped` draws
    // that slice as a face of the isosurface material (a cleanly "sawn off" solid);
    // `open` (capped=false) omits it, leaving the surface's cut edge / a see-through
    // opening. Only affects surfaces that actually reach the container.
    bool      capped = false;
    // Lipschitz bound of the field: for a true SDF this is 1, and a sphere-trace
    // step of |f| never overshoots. Fields that aren't unit-Lipschitz (e.g. summed
    // metaball densities) set this > 1 so the step is scaled down to |f|/lipschitz,
    // trading speed for correctness. Set by the builder. (Adaptive method only.)
    double lipschitz = 1.0;
    // Minimum march step (world units): the floor on the sign-change ray march and
    // the thinnest resolvable feature. Sized from the bounds by the builder.
    double minStep = 1e-4;
    // Ray-march strategy (see MarchMethod / RootRefine above). `sampleStep` is the
    // fixed world-space march step used by MarchMethod::Sample (sized by the builder
    // from `samples`/`accuracy`); unused by Adaptive.
    MarchMethod method = MarchMethod::Adaptive;
    RootRefine  refine = RootRefine::Bisect;
    double sampleStep = 0.0;
    // Procedural UV wrap for pattern/expression materials (spec §9.2, native-object
    // adaptation). When uvProj != None the hit's (u,v) are projected from the world
    // hit point across `uvBounds` (defaults to `bounds`) using `uvAxis` as the
    // up/projection axis — the same algorithm meshes use. None leaves (u,v)=(0,0).
    UvProjection uvProj = UvProjection::None;
    int  uvAxis = 1;             // 0=x, 1=y, 2=z
    Aabb uvBounds;               // reference box for the [0,1] wrap (set by builder)
    bool uvBoundsSet = false;

    double eval(const Vec3& pw) const {
        return fieldEval(nodes.data(), (int)nodes.size(), pw, exprNodes.data());
    }
    Vec3   gradient(const Vec3& pw, double eps) const {
        return fieldGradient(nodes.data(), (int)nodes.size(), pw, eps, exprNodes.data());
    }
};

// Sphere-trace a ray against one Implicit. `r.d` need not be unit length: the SDF
// is in world distance, so the parametric step is (|f|/lipschitz)/|d|. Writes into
// `hit` (respecting hit.t as the current closest) and returns true on a nearer hit.
inline bool intersectImplicit(const Ray& r, const Implicit& im, double tmin, Hit& hit) {
    // ---- Container clip: entry/exit params [tEnter, tExit] and the container's
    // OUTWARD normals at those crossings (needed to shade caps). The container is the
    // world AABB `bounds` (box) or a world sphere. `bounds` is always the BVH/broad box.
    double tEnter, tExit;
    Vec3   nEnter{0, 0, 0}, nExit{0, 0, 0};
    if (im.container == Container::Sphere) {
        Vec3   oc = r.o - im.sphereCenter;
        double A  = dot(r.d, r.d);
        double B  = dot(oc, r.d);
        double C  = dot(oc, oc) - im.sphereRadius * im.sphereRadius;
        double disc = B * B - A * C;
        if (disc < 0.0) return false;
        double sq = std::sqrt(disc);
        tEnter = (-B - sq) / A;
        tExit  = (-B + sq) / A;
        Vec3 pe = r.o + r.d * tEnter, px = r.o + r.d * tExit;
        Vec3 ge = pe - im.sphereCenter, gx = px - im.sphereCenter;
        double le = length(ge), lx = length(gx);
        nEnter = (le > 0.0) ? ge / le : Vec3{0, 0, 1};
        nExit  = (lx > 0.0) ? gx / lx : Vec3{0, 0, 1};
    } else {
        Vec3 invD{1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z};
        tEnter = -1e300; tExit = 1e300;
        int eAx = 0; double eSgn = -1.0;   // entry face axis + outward-normal sign
        int xAx = 0; double xSgn = 1.0;    // exit  face axis + outward-normal sign
        for (int a = 0; a < 3; ++a) {
            double o = vget(r.o, a), id = vget(invD, a);
            double tLo = (vget(im.bounds.lo, a) - o) * id;
            double tHi = (vget(im.bounds.hi, a) - o) * id;
            double tnear, tfar, nearSgn, farSgn;
            if (id >= 0.0) { tnear = tLo; tfar = tHi; nearSgn = -1.0; farSgn = +1.0; }
            else           { tnear = tHi; tfar = tLo; nearSgn = +1.0; farSgn = -1.0; }
            if (tnear > tEnter) { tEnter = tnear; eAx = a; eSgn = nearSgn; }
            if (tfar  < tExit)  { tExit  = tfar;  xAx = a; xSgn = farSgn; }
            if (tExit < tEnter) return false;
        }
        (eAx == 0 ? nEnter.x : eAx == 1 ? nEnter.y : nEnter.z) = eSgn;
        (xAx == 0 ? nExit.x  : xAx == 1 ? nExit.y  : nExit.z)  = xSgn;
    }
    double t0 = tmin, t1 = hit.t;
    if (tEnter > t0) t0 = tEnter;
    if (tExit  < t1) t1 = tExit;
    if (t1 < t0) return false;
    // `capped` draws the container face where the ray is inside the solid at a
    // container crossing. `exitIsContainer` = the far clip ended at the container wall
    // (not at a nearer surface via hit.t), so a far cap there is genuinely visible.
    const bool capped          = im.capped;
    const bool exitIsContainer = (tExit <= hit.t);

    const double dlen    = length(r.d);
    const int    N       = (int)im.nodes.size();
    const FieldNode* nd  = im.nodes.data();
    const PatNode* pool  = im.exprNodes.data();   // backs FieldOp::Expr leaves (may be null)
    const int    MAX_STEP = 2048;
    const double invLip  = 1.0 / (im.lipschitz > 0.0 ? im.lipschitz : 1.0);
    // Minimum world-space march step: a floor on the otherwise |f|-adaptive step so
    // the ray actually STEPS ACROSS the zero level set (rather than asymptotically
    // creeping up to it), which is what lets us detect a sign CHANGE. It also caps
    // the thinnest resolvable feature. Sized to the primitive so unit-scale scenes
    // and metre-scale scenes both behave.
    const double minStep = im.minStep > 0.0 ? im.minStep : 1e-4;

    // March by min(|f|, ...) with a floor, watching for f to change sign. A sign
    // change brackets a genuine surface crossing; merely grazing the surface (f
    // dips but stays one sign) never triggers a hit — so a shadow/bounce ray
    // spawned just off a surface can leave it without self-intersecting. Transmission
    // rays are spawned just INSIDE (f<0) by the tracers and correctly cross to the
    // far side. This is far more robust than proximity (|f|<eps) detection.
    // Sample method marches by a fixed world step (ignoring |f|); Adaptive marches by
    // the |f|/lipschitz safe distance with a minStep floor.
    const bool  sampleMode  = (im.method == MarchMethod::Sample);
    const double fixedStep  = (im.sampleStep > 0.0 ? im.sampleStep : minStep) / dlen;
    const bool  regulaFalsi = (im.refine == RootRefine::RegulaFalsi);

    // Commit a hit at parametric `th`, world point `p`, geometric normal `ng`
    // (field gradient for the surface; container outward normal for a cap).
    auto writeHit = [&](double th, const Vec3& p, const Vec3& ng) -> bool {
        hit.t = th; hit.p = p; hit.valid = true;
        hit.ng = ng;
        hit.n  = (dot(r.d, ng) < 0.0) ? ng : -ng;
        hit.matId = im.matId; hit.sensorId = -1;
        if (im.uvProj != UvProjection::None) {
            const Aabb& b = im.uvBoundsSet ? im.uvBounds : im.bounds;
            Vec3 ctr = (b.lo + b.hi) * 0.5;
            Vec3 uv = projectUV(p, b.lo, b.hi, ctr, im.uvProj, im.uvAxis);
            hit.u = uv.x; hit.v = uv.y;
        } else { hit.u = 0.0; hit.v = 0.0; }
        return true;
    };

    double t = t0;
    double f = fieldEval(nd, N, r.o + r.d * t, pool);
    // NEAR CAP: the ray enters the container from outside (tEnter >= tmin) already
    // inside the solid (f<0). The container face is then the first surface along the
    // ray — the nearest possible hit — so cap it and return. `open` surfaces skip this
    // and let the march reveal the field (a see-through opening / cut edge).
    if (capped && tEnter >= tmin && tEnter < hit.t && f < 0.0)
        return writeHit(tEnter, r.o + r.d * tEnter, nEnter);
    for (int i = 0; i < MAX_STEP; ++i) {
        double step = sampleMode ? fixedStep
                                 : std::fmax(std::fabs(f) * invLip, minStep) / dlen;
        double tn = t + step;
        // Clamp the last step to the AABB exit and still test for a crossing there:
        // a ray starting inside the surface can have its first |f|-sized step land
        // just past t1, but the genuine zero crossing lies within [t, t1]. Bailing
        // before evaluating the boundary would drop that hit.
        bool last = false;
        if (tn >= t1) { tn = t1; last = true; }
        double fn = fieldEval(nd, N, r.o + r.d * tn, pool);
        bool crossed = (f > 0.0 && fn <= 0.0) || (f < 0.0 && fn >= 0.0) ||
                       (f == 0.0 && fn != 0.0);
        if (crossed) {
            // Refine the bracket [t, tn] to a precise root (residual ~1e-12) so the
            // shared 1e-6 ray-spawn offset lands safely on the correct side. Bisection
            // is the robust default; regula-falsi (Illinois-safeguarded) interpolates.
            double ta = t, tb = tn, fa = f, fb = fn;
            int side = 0;
            for (int b = 0; b < 80; ++b) {
                double tm;
                if (regulaFalsi && (fb - fa) != 0.0) {
                    tm = (ta * fb - tb * fa) / (fb - fa);
                    if (tm <= ta || tm >= tb) tm = 0.5 * (ta + tb);  // guard out-of-bracket
                } else {
                    tm = 0.5 * (ta + tb);
                }
                double fm = fieldEval(nd, N, r.o + r.d * tm, pool);
                if ((fa > 0.0) == (fm > 0.0)) {
                    ta = tm; fa = fm;
                    if (regulaFalsi && side == +1) fb *= 0.5;   // Illinois: shrink stuck side
                    side = +1;
                } else {
                    tb = tm; fb = fm;
                    if (regulaFalsi && side == -1) fa *= 0.5;
                    side = -1;
                }
                if ((tb - ta) * dlen < 1e-12) break;
            }
            double th = 0.5 * (ta + tb);
            if (th < tmin || th >= hit.t) return false;
            Vec3 p = r.o + r.d * th;
            double eps = std::fmax(1e-6, 1e-4 * th);
            Vec3 g = fieldGradient(nd, N, p, eps, pool);
            return writeHit(th, p, g);   // field surface: geometric normal = gradient
        }
        if (last) {
            // FAR CAP: reached the container exit still inside the solid (fn<0) — the
            // exit face seals the sawn-off solid. Only when the far clip is the
            // container itself (not a nearer surface via hit.t). Handles rays that
            // originate inside the solid (bounce/transmission/shadow) and exit a cap.
            if (capped && exitIsContainer && fn < 0.0 && tExit >= tmin && tExit < hit.t)
                return writeHit(tExit, r.o + r.d * tExit, nExit);
            return false;
        }
        t = tn; f = fn;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Conservative world AABB of a field expression, computed by running the same
// postfix machine over per-node boxes. Leaf boxes are the primitive's local AABB
// transformed to world; combinators fold operand boxes the way the SDF folds
// values (union -> box union, intersect -> box intersection, difference -> first
// operand's box). Smooth variants can bulge the surface out by ~k, so those pad by
// the blend radius. Planes are unbounded and yield a huge box (must be bounded by
// another operand via intersection/difference).
// ---------------------------------------------------------------------------
namespace implicit_detail {

inline Aabb transformedLocalBox(const FieldNode& nd, const Vec3& lo, const Vec3& hi) {
    // Map the 8 local corners to world via the leaf's local->world = inverse of nd.inv.
    // We only stored world->local (nd.inv); invert its linear part + translation.
    const Affine& W2L = nd.inv;
    double a = W2L.m[0], b = W2L.m[1], c = W2L.m[2];
    double d = W2L.m[3], e = W2L.m[4], f = W2L.m[5];
    double g = W2L.m[6], h = W2L.m[7], i = W2L.m[8];
    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    Aabb box;
    if (std::fabs(det) < 1e-30) { box.lo = lo; box.hi = hi; return box; }  // degenerate
    double invDet = 1.0 / det;
    double L2W[9] = {
        (e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet,
        (f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet,
        (d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet,
    };
    Vec3 tt = W2L.t;
    // local->world: pw = L2W * (pl - ... ) ; since pl = W2L.m*pw + W2L.t, invert:
    // pw = L2W * (pl - W2L.t).
    Vec3 off{-(L2W[0] * tt.x + L2W[1] * tt.y + L2W[2] * tt.z),
             -(L2W[3] * tt.x + L2W[4] * tt.y + L2W[5] * tt.z),
             -(L2W[6] * tt.x + L2W[7] * tt.y + L2W[8] * tt.z)};
    for (int cx = 0; cx < 8; ++cx) {
        Vec3 pl{(cx & 1) ? hi.x : lo.x, (cx & 2) ? hi.y : lo.y, (cx & 4) ? hi.z : lo.z};
        Vec3 pw{L2W[0] * pl.x + L2W[1] * pl.y + L2W[2] * pl.z + off.x,
                L2W[3] * pl.x + L2W[4] * pl.y + L2W[5] * pl.z + off.y,
                L2W[6] * pl.x + L2W[7] * pl.y + L2W[8] * pl.z + off.z};
        box.expand(pw);
    }
    return box;
}

inline Aabb leafBox(const FieldNode& nd) {
    const double BIG = 1e18;
    switch (nd.op) {
        case FieldOp::Sphere:
            return transformedLocalBox(nd, Vec3{-nd.p[0], -nd.p[0], -nd.p[0]},
                                           Vec3{ nd.p[0],  nd.p[0],  nd.p[0]});
        case FieldOp::Box:
            return transformedLocalBox(nd, Vec3{-nd.p[0], -nd.p[1], -nd.p[2]},
                                           Vec3{ nd.p[0],  nd.p[1],  nd.p[2]});
        case FieldOp::Torus: {
            double R = nd.p[0] + nd.p[1];
            return transformedLocalBox(nd, Vec3{-R, -nd.p[1], -R}, Vec3{R, nd.p[1], R});
        }
        case FieldOp::Cylinder:
            return transformedLocalBox(nd, Vec3{-nd.p[0], -nd.p[1], -nd.p[0]},
                                           Vec3{ nd.p[0],  nd.p[1],  nd.p[0]});
        case FieldOp::Cone: {
            double rr = std::fmax(nd.p[0], nd.p[1]);
            return transformedLocalBox(nd, Vec3{-rr, -nd.p[2], -rr}, Vec3{rr, nd.p[2], rr});
        }
        case FieldOp::Plane:
        case FieldOp::Expr:      // an arbitrary field has no analytic box: the isosurface's
        default: {               // required `contained_by` clamps it (see addIsosurface).
            Aabb box; box.lo = Vec3{-BIG, -BIG, -BIG}; box.hi = Vec3{BIG, BIG, BIG};
            return box;
        }
    }
}

inline Aabb boxIntersect(const Aabb& a, const Aabb& b) {
    Aabb r;
    r.lo = Vec3{std::fmax(a.lo.x, b.lo.x), std::fmax(a.lo.y, b.lo.y), std::fmax(a.lo.z, b.lo.z)};
    r.hi = Vec3{std::fmin(a.hi.x, b.hi.x), std::fmin(a.hi.y, b.hi.y), std::fmin(a.hi.z, b.hi.z)};
    return r;
}
inline void boxPad(Aabb& a, double k) {
    a.lo = a.lo - Vec3{k, k, k}; a.hi = a.hi + Vec3{k, k, k};
}

}  // namespace implicit_detail

inline Aabb implicitBounds(const std::vector<FieldNode>& nodes);   // fwd decl

// March-step floor for a primitive of the given world bounds: a small fraction of
// the bounds diagonal, clamped so it neither creeps (too small) nor skips features
// (too large). Bounded planes/huge boxes clamp to the ceiling.
inline double implicitMinStep(const Aabb& b) {
    double diag = length(b.hi - b.lo);
    double s = 1e-3 * diag;
    if (s < 1e-5) s = 1e-5;
    if (s > 1e-3) s = 1e-3;
    return s;
}

// True if the field expression contains an arbitrary-expression (Expr) leaf, which is
// NOT a signed-distance function — such a field needs an explicit container box and a
// Lipschitz (max-gradient) bound instead of the SDF assumptions.
inline bool fieldHasExpr(const std::vector<FieldNode>& nodes) {
    for (const auto& nd : nodes) if (nd.op == FieldOp::Expr) return true;
    return false;
}

// Estimate a Lipschitz bound L >= max|grad f| for the WORLD field over the box, by
// sampling |grad f| on a grid (central differences). Sphere-tracing by |f|/L then
// provably never overshoots the first zero crossing (|f(p)| <= L*dist => the nearest
// surface is at least |f|/L away). The caller pads the result for safety. Returns at
// least 1 (a true SDF has L = 1) so plain SDF leaves in the same field aren't slowed.
inline double estimateFieldLipschitz(const std::vector<FieldNode>& nodes,
                                     const std::vector<PatNode>& exprNodes,
                                     const Aabb& box, int grid = 24) {
    const FieldNode* nd = nodes.data();
    const int N = (int)nodes.size();
    const PatNode* pool = exprNodes.data();
    Vec3 ext = box.hi - box.lo;
    double diag = length(ext);
    if (diag <= 0) return 1.0;
    double eps = 1e-4 * diag;
    double maxg = 1.0;
    for (int iz = 0; iz <= grid; ++iz)
    for (int iy = 0; iy <= grid; ++iy)
    for (int ix = 0; ix <= grid; ++ix) {
        Vec3 p{box.lo.x + ext.x * (ix / (double)grid),
               box.lo.y + ext.y * (iy / (double)grid),
               box.lo.z + ext.z * (iz / (double)grid)};
        double gx = fieldEval(nd, N, p + Vec3{eps,0,0}, pool) - fieldEval(nd, N, p - Vec3{eps,0,0}, pool);
        double gy = fieldEval(nd, N, p + Vec3{0,eps,0}, pool) - fieldEval(nd, N, p - Vec3{0,eps,0}, pool);
        double gz = fieldEval(nd, N, p + Vec3{0,0,eps}, pool) - fieldEval(nd, N, p - Vec3{0,0,eps}, pool);
        double g = std::sqrt(gx*gx + gy*gy + gz*gz) / (2.0 * eps);
        if (g > maxg) maxg = g;
    }
    return maxg;
}

// Build a single-sphere implicit at world center `c`, world radius `r`. The leaf
// is a unit-Lipschitz SDF (a true distance function), so a sphere trace against it
// reproduces the analytic sphere intersection to the surface epsilon — the step-1
// validation target.
inline Implicit makeSphereImplicit(const Vec3& c, double r, int matId) {
    FieldNode nd;
    nd.op = FieldOp::Sphere;
    nd.p[0] = r;                       // local radius = world radius (translation-only leaf)
    nd.inv = Affine::identity();       // world->local linear part = I
    nd.inv.t = c * -1.0;               // pl = pw - c
    nd.scale = 1.0;                    // no distance rescale
    Implicit im;
    im.nodes.push_back(nd);
    im.matId = matId;
    im.lipschitz = 1.0;
    im.bounds = implicitBounds(im.nodes);
    im.minStep = implicitMinStep(im.bounds);
    return im;
}

inline Aabb implicitBounds(const std::vector<FieldNode>& nodes) {
    using namespace implicit_detail;
    Aabb st[64]; int sp = 0;
    for (const auto& nd : nodes) {
        switch (nd.op) {
            case FieldOp::Union: {
                Aabb b = st[--sp], a = st[--sp]; a.expand(b); st[sp++] = a; break;
            }
            case FieldOp::SmoothUnion: {
                Aabb b = st[--sp], a = st[--sp]; a.expand(b); boxPad(a, nd.p[0]); st[sp++] = a; break;
            }
            case FieldOp::Intersect: {
                Aabb b = st[--sp], a = st[--sp]; st[sp++] = boxIntersect(a, b); break;
            }
            case FieldOp::SmoothIntersect: {
                Aabb b = st[--sp], a = st[--sp]; Aabb r = boxIntersect(a, b); boxPad(r, nd.p[0]); st[sp++] = r; break;
            }
            case FieldOp::Difference: {
                st[--sp]; /* discard b */ break;   // a stays on the stack
            }
            case FieldOp::SmoothDifference: {
                st[--sp]; boxPad(st[sp - 1], nd.p[0]); break;
            }
            default:
                st[sp++] = leafBox(nd);
                break;
        }
    }
    if (sp <= 0) { Aabb e; e.lo = Vec3{0, 0, 0}; e.hi = Vec3{0, 0, 0}; return e; }
    // Small pad so the sphere-trace entry point isn't exactly on the surface.
    Aabb b = st[0];
    boxPad(b, 1e-4);
    return b;
}
