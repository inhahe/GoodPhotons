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
    void finalize() {
        gn = normalize(cross(v1 - v0, v2 - v0));
        if (dot(n0, n0) < 1e-12) n0 = gn;
        if (dot(n1, n1) < 1e-12) n1 = gn;
        if (dot(n2, n2) < 1e-12) n2 = gn;
        // Tangent from the UV gradient (Lengyel). Solve for the world direction along
        // which texture-u increases, then Gram-Schmidt it against gn and record the
        // bitangent handedness. Falls back to onb(gn) when UVs are degenerate.
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
    bool flipped = !(dot(r.d, ns) < 0.0);
    hit.n = flipped ? -ns : ns;
    hit.tangent = tri.tangent;             // per-triangle tangent (constant across the face)
    hit.bitangentSign = tri.bitangentSign; // for tangent-space normal mapping (C6)
    // Curvature follows the shaded side: seen from the back, a bulge IS a pit (O3).
    hit.curv = flipped ? -tri.curvature : tri.curvature;
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
    bool sFlipped = !(dot(r.d, ng) < 0.0);
    hit.n = sFlipped ? -ng : ng;
    // Analytic mean curvature of a sphere: 1/R everywhere, negated when seen from
    // inside (a room-sized sphere is concave to anything standing in it).
    hit.curv = (s.r > 1e-12) ? ((sFlipped ? -1.0 : 1.0) / s.r) : 0.0;
    hit.matId = s.matId; hit.sensorId = -1;
    // Equirectangular (lat/long) UV so spheres can be textured (globes, eyeballs).
    hit.u = 0.5 + std::atan2(ng.z, ng.x) / (2.0 * PI);
    hit.v = 0.5 - std::asin(std::clamp(ng.y, -1.0, 1.0)) / PI;
    // Tangent = the east-pointing longitude direction d/du (C6). d/du of the
    // equirectangular map is proportional to (-sin, 0, cos) of the longitude, i.e.
    // perpendicular to both world-up and the normal; degenerates to +x at the poles.
    Vec3 T{-ng.z, 0.0, ng.x};
    double tl = std::sqrt(dot(T, T));
    hit.tangent = (tl > 1e-9) ? T * (1.0 / tl) : Vec3{1, 0, 0};
    hit.bitangentSign = 1.0;
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
