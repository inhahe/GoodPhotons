// CUDA backend for the forward light tracer (model B). See render_cuda.h.
//
// The HOST side (compiled by the host compiler under nvcc) includes the project
// headers, reads the std::function-based Scene, and bakes it into POD device
// structs: material reflectances/indices sampled into fixed spectral tables, the
// flat BVH copied verbatim, the light emission CDF copied, and the camera reduced
// to its projection frame. The DEVICE side is fully self-contained (its own vector
// math, RNG, intersection and BVH traversal) so it never depends on a host header
// being __device__-annotated — keeping all GPU concerns isolated to this file.
//
// The megakernel kTrace mirrors Renderer::tracePhoton exactly (same emission,
// same per-material interaction, same fog free-flight, same model-B connect), so
// at convergence the GPU image matches the CPU image up to Monte-Carlo noise.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <math.h>
#include <float.h>

#include "render_cuda.h"

// ============================ device-side scene ============================

#define HD __host__ __device__
static constexpr int    SPEC_N   = 96;        // spectral table resolution
static constexpr double DLMIN    = 360.0;     // mirrors color.h LAMBDA_MIN/MAX
static constexpr double DLMAX    = 830.0;
static constexpr double DPI      = 3.141592653589793;

// All device code lives in namespace gpu so its helpers (clamp01, cieX, hgPhase,
// thinFilmReflectance, ...) don't collide with the identically-named host inline
// functions pulled in via render_cuda.h -> render.h / color.h.
namespace gpu {

struct DVec3 {
    double x, y, z;
    HD DVec3() : x(0), y(0), z(0) {}
    HD DVec3(double a, double b, double c) : x(a), y(b), z(c) {}
    HD DVec3 operator+(const DVec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD DVec3 operator-(const DVec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD DVec3 operator*(double s)       const { return {x * s, y * s, z * s}; }
    HD DVec3 operator/(double s)       const { return {x / s, y / s, z / s}; }
    HD DVec3 operator-()               const { return {-x, -y, -z}; }
};
HD static inline double dot(const DVec3& a, const DVec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
HD static inline DVec3 cross(const DVec3& a, const DVec3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
HD static inline double length(const DVec3& a) { return sqrt(dot(a, a)); }
HD static inline DVec3 normalize(const DVec3& a) { return a / length(a); }
HD static inline DVec3 reflectv(const DVec3& d, const DVec3& n) { return d - n * (2.0 * dot(d, n)); }
HD static inline void onb(const DVec3& n, DVec3& t, DVec3& b) {
    double sign = copysign(1.0, n.z);
    double a = -1.0 / (sign + n.z);
    double d = n.x * n.y * a;
    t = DVec3(1.0 + sign * n.x * n.x * a, sign * d, -sign * n.x);
    b = DVec3(d, sign + n.y * n.y * a, -n.y);
}
HD static inline double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

// Material type tags (must match MatType order in scene.h).
enum { D_DIFFUSE=0, D_DIELECTRIC, D_MIRROR, D_HALFMIRROR, D_GLOSSY, D_FLUORESCENT, D_THINFILM, D_GRATING };

struct DMaterial {
    int    type;
    double reflect[SPEC_N];     // baked reflect spectrum
    double ior[SPEC_N];         // baked index spectrum
    double roughness;
    double filmIor, filmThickness;
    double grooveSpacing;
    DVec3  grooveDir;
    int    gratingMaxOrder;
};

struct DTri    { DVec3 v0, v1, v2, gn; int matId, sensorId; };
struct DSphere { DVec3 c; double r; int matId; };
struct DNode   { DVec3 lo, hi; int left, right, first, count; };

struct DMedium {
    int    enabled;
    double sigma_a[SPEC_N];
    double sigma_s[SPEC_N];
    double g;
};

struct DScene {
    const DTri*      tris;  int nTris;
    const DSphere*   sph;   int nSph;
    const DMaterial* mats;
    const DNode*     nodes; const int* primIdx; int nNodes;
    DVec3  lightOrigin, lightU, lightV, lightNormal;
    double lightArea, lightEmitIntegral;
    int    collimated; DVec3 beamDir;
    const double* lightCdf; int lightCdfN; double lightStep;
    DMedium medium;
};

struct DCamera {
    DVec3  eye, u, v, w;
    double tanHalfX, tanHalfY;
    int    resX, resY;
    HD double imagePlaneArea() const { return 4.0 * tanHalfX * tanHalfY; }
    HD bool project(const DVec3& p, int& px, int& py, double& cosCam, double& dist2) const {
        DVec3 d = p - eye;
        double cz = dot(d, w);
        if (cz <= 1e-9) return false;
        double cx = dot(d, u), cy = dot(d, v);
        double ix = (cx / cz) / tanHalfX, iy = (cy / cz) / tanHalfY;
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
        px = (int)((ix * 0.5 + 0.5) * resX);
        py = (int)((iy * 0.5 + 0.5) * resY);
        dist2 = dot(d, d);
        cosCam = cz / sqrt(dist2);
        return true;
    }
};

// ============================ device helpers ============================

struct DRng {
    unsigned long long state, inc;
    __device__ void seed(unsigned long long seq, unsigned long long s) {
        state = 0; inc = (seq << 1u) | 1u;
        next(); state += s; next();
    }
    __device__ unsigned int next() {
        unsigned long long old = state;
        state = old * 6364136223846793005ULL + inc;
        unsigned int xorshifted = (unsigned int)(((old >> 18u) ^ old) >> 27u);
        unsigned int rot = (unsigned int)(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }
    __device__ double uniform() { return (next() >> 8) * (1.0 / 16777216.0); }
};

__device__ static DVec3 cosineHemisphere(const DVec3& n, DRng& rng) {
    double u1 = rng.uniform(), u2 = rng.uniform();
    double r = sqrt(u1), phi = 6.283185307179586 * u2;
    double lx = r * cos(phi), ly = r * sin(phi), lz = sqrt(1.0 - u1);
    DVec3 t, b; onb(n, t, b);
    return normalize(t * lx + b * ly + n * lz);
}
__device__ static DVec3 sampleGlossy(const DVec3& mdir, double roughness, DRng& rng) {
    double rr = roughness < 1e-3 ? 1e-3 : roughness;
    double e = 2.0 / (rr * rr) - 2.0; if (e < 0) e = 0;
    double u1 = rng.uniform(), u2 = rng.uniform();
    double cosT = pow(u1, 1.0 / (e + 1.0));
    double sinT = sqrt(fmax(0.0, 1.0 - cosT * cosT));
    double phi = 2.0 * DPI * u2;
    DVec3 t, b; onb(mdir, t, b);
    return normalize(t * (sinT * cos(phi)) + b * (sinT * sin(phi)) + mdir * cosT);
}
__device__ static double hgPhase(double cosTheta, double g) {
    double d = 1.0 + g * g - 2.0 * g * cosTheta;
    if (d < 1e-9) d = 1e-9;
    return (1.0 - g * g) / (4.0 * DPI * d * sqrt(d));
}
__device__ static DVec3 sampleHG(const DVec3& wi, double g, DRng& rng) {
    double u1 = rng.uniform(), u2 = rng.uniform(), cosT;
    if (fabs(g) < 1e-3) cosT = 1.0 - 2.0 * u1;
    else { double sq = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1); cosT = (1.0 + g * g - sq * sq) / (2.0 * g); }
    double sinT = sqrt(fmax(0.0, 1.0 - cosT * cosT));
    double phi = 2.0 * DPI * u2;
    DVec3 t, b; onb(wi, t, b);
    return normalize(t * (sinT * cos(phi)) + b * (sinT * sin(phi)) + wi * cosT);
}

// CIE 1931 CMF (analytic multi-Gaussian fit — same as color.h).
__device__ static double gaussPiece(double x, double mu, double s1, double s2) {
    double t = (x - mu) * ((x < mu) ? s1 : s2);
    return exp(-0.5 * t * t);
}
__device__ static double cieX(double w) {
    return 0.362 * gaussPiece(w, 442.0, 0.0624, 0.0374)
         + 1.056 * gaussPiece(w, 599.8, 0.0264, 0.0323)
         - 0.065 * gaussPiece(w, 501.1, 0.0490, 0.0382);
}
__device__ static double cieY(double w) {
    return 0.821 * gaussPiece(w, 568.8, 0.0213, 0.0247)
         + 0.286 * gaussPiece(w, 530.9, 0.0613, 0.0322);
}
__device__ static double cieZ(double w) {
    return 1.217 * gaussPiece(w, 437.0, 0.0845, 0.0278)
         + 0.681 * gaussPiece(w, 459.0, 0.0385, 0.0725);
}

// Spectral table lookup with linear interpolation over [DLMIN, DLMAX].
__device__ static double specLookup(const double* tab, double lambda) {
    double f = (lambda - DLMIN) / (DLMAX - DLMIN) * (SPEC_N - 1);
    if (f <= 0) return tab[0];
    if (f >= SPEC_N - 1) return tab[SPEC_N - 1];
    int i = (int)f; double frac = f - i;
    return tab[i] * (1.0 - frac) + tab[i + 1] * frac;
}
__device__ static double medSigmaT(const DMedium& m, double lambda) {
    double a = specLookup(m.sigma_a, lambda), s = specLookup(m.sigma_s, lambda);
    double v = fmax(0.0, a) + fmax(0.0, s);
    return v;
}
__device__ static double medAlbedo(const DMedium& m, double lambda) {
    double s = fmax(0.0, specLookup(m.sigma_s, lambda));
    double t = s + fmax(0.0, specLookup(m.sigma_a, lambda));
    return t > 0.0 ? s / t : 0.0;
}

// Thin-film Airy reflectance (port of render.h thinFilmReflectance).
__device__ static double thinFilmReflectance(double n0, double n1, double n2, double d,
                                              double cosI, double lambda) {
    cosI = clamp01(fabs(cosI));
    double sin0_2 = fmax(0.0, 1.0 - cosI * cosI);
    double sin1_2 = (n0 * n0) / (n1 * n1) * sin0_2;
    if (sin1_2 >= 1.0) return 1.0;
    double cos1 = sqrt(1.0 - sin1_2);
    double sin2_2 = (n0 * n0) / (n2 * n2) * sin0_2;
    bool tir = sin2_2 >= 1.0;
    double cos2 = tir ? 0.0 : sqrt(1.0 - sin2_2);
    double r01s = (n0 * cosI - n1 * cos1) / (n0 * cosI + n1 * cos1);
    double r01p = (n1 * cosI - n0 * cos1) / (n1 * cosI + n0 * cos1);
    double r12s = tir ? 1.0 : (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2);
    double r12p = tir ? 1.0 : (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2);
    double phi  = (4.0 * DPI * n1 * d * cos1) / lambda;
    double cphi = cos(phi);
    double numS = r01s*r01s + r12s*r12s + 2.0*r01s*r12s*cphi;
    double denS = 1.0 + r01s*r01s*r12s*r12s + 2.0*r01s*r12s*cphi;
    double numP = r01p*r01p + r12p*r12p + 2.0*r01p*r12p*cphi;
    double denP = 1.0 + r01p*r01p*r12p*r12p + 2.0*r01p*r12p*cphi;
    double Rs = clamp01(denS > 1e-12 ? numS / denS : numS);
    double Rp = clamp01(denP > 1e-12 ? numP / denP : numP);
    return 0.5 * (Rs + Rp);
}

// ============================ intersection / BVH ============================

struct DHit {
    double t; bool valid;
    DVec3 p, n, ng;
    int matId, sensorId;
};

__device__ static bool intersectTri(const DVec3& ro, const DVec3& rd, const DTri& tri,
                                     double tmin, DHit& hit) {
    const double EPS = 1e-9;
    DVec3 e1 = tri.v1 - tri.v0, e2 = tri.v2 - tri.v0;
    DVec3 pv = cross(rd, e2);
    double det = dot(e1, pv);
    if (fabs(det) < EPS) return false;
    double inv = 1.0 / det;
    DVec3 tv = ro - tri.v0;
    double u = dot(tv, pv) * inv;
    if (u < 0.0 || u > 1.0) return false;
    DVec3 qv = cross(tv, e1);
    double vv = dot(rd, qv) * inv;
    if (vv < 0.0 || u + vv > 1.0) return false;
    double t = dot(e2, qv) * inv;
    if (t < tmin || t >= hit.t) return false;
    hit.t = t; hit.p = ro + rd * t; hit.valid = true;
    hit.ng = tri.gn;
    hit.n = (dot(rd, tri.gn) < 0.0) ? tri.gn : -tri.gn;
    hit.matId = tri.matId; hit.sensorId = tri.sensorId;
    return true;
}
__device__ static bool intersectSphere(const DVec3& ro, const DVec3& rd, const DSphere& s,
                                        double tmin, DHit& hit) {
    DVec3 oc = ro - s.c;
    double a = dot(rd, rd), b = 2.0 * dot(oc, rd), c = dot(oc, oc) - s.r * s.r;
    double disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return false;
    double sq = sqrt(disc);
    double t = (-b - sq) / (2.0 * a);
    if (t < tmin) t = (-b + sq) / (2.0 * a);
    if (t < tmin || t >= hit.t) return false;
    hit.t = t; hit.p = ro + rd * t; hit.valid = true;
    DVec3 ng = normalize(hit.p - s.c);
    hit.ng = ng;
    hit.n = (dot(rd, ng) < 0.0) ? ng : -ng;
    hit.matId = s.matId; hit.sensorId = -1;
    return true;
}
__device__ static bool boxHit(const DNode& nd, const DVec3& ro, const DVec3& invD,
                               double tmin, double tmax, double& tEnter) {
    double te = tmin, tx = tmax;
    double lo[3] = {nd.lo.x, nd.lo.y, nd.lo.z}, hi[3] = {nd.hi.x, nd.hi.y, nd.hi.z};
    double o[3] = {ro.x, ro.y, ro.z}, id[3] = {invD.x, invD.y, invD.z};
    for (int a = 0; a < 3; ++a) {
        double t0 = (lo[a] - o[a]) * id[a], t1 = (hi[a] - o[a]) * id[a];
        if (t0 > t1) { double tmp = t0; t0 = t1; t1 = tmp; }
        te = t0 > te ? t0 : te;
        tx = t1 < tx ? t1 : tx;
        if (tx < te) return false;
    }
    tEnter = te;
    return true;
}

__device__ static DHit closestHit(const DScene& sc, const DVec3& ro, const DVec3& rd,
                                   double tmin = 1e-6) {
    DHit h; h.t = DBL_MAX; h.valid = false; h.matId = 0; h.sensorId = -1;
    if (sc.nNodes == 0) return h;
    DVec3 invD{1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z};
    double tMax = DBL_MAX;
    int stack[64]; int sp = 0; stack[sp++] = 0;
    while (sp) {
        const DNode& n = sc.nodes[stack[--sp]];
        double tE;
        if (!boxHit(n, ro, invD, tmin, tMax, tE)) continue;
        if (n.count > 0) {
            for (int i = 0; i < n.count; ++i) {
                int prim = sc.primIdx[n.first + i];
                if (prim < sc.nTris) { if (intersectTri(ro, rd, sc.tris[prim], tmin, h)) tMax = h.t; }
                else                 { if (intersectSphere(ro, rd, sc.sph[prim - sc.nTris], tmin, h)) tMax = h.t; }
            }
        } else {
            double tL, tR;
            bool hL = boxHit(sc.nodes[n.left], ro, invD, tmin, tMax, tL);
            bool hR = boxHit(sc.nodes[n.right], ro, invD, tmin, tMax, tR);
            if (hL && hR) {
                if (tL <= tR) { stack[sp++] = n.right; stack[sp++] = n.left; }
                else          { stack[sp++] = n.left;  stack[sp++] = n.right; }
            } else if (hL) stack[sp++] = n.left;
            else if (hR)   stack[sp++] = n.right;
        }
    }
    return h;
}
__device__ static bool occluded(const DScene& sc, const DVec3& o, const DVec3& dir,
                                 double maxDist, double tmin = 1e-6) {
    if (sc.nNodes == 0) return false;
    DVec3 invD{1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z};
    double tMax = maxDist - tmin;
    int stack[64]; int sp = 0; stack[sp++] = 0;
    while (sp) {
        const DNode& n = sc.nodes[stack[--sp]];
        double tE;
        if (!boxHit(n, o, invD, tmin, tMax, tE)) continue;
        if (n.count > 0) {
            for (int i = 0; i < n.count; ++i) {
                int prim = sc.primIdx[n.first + i];
                DHit h; h.t = tMax; h.valid = false;
                bool blocked = (prim < sc.nTris) ? intersectTri(o, dir, sc.tris[prim], tmin, h)
                                                 : intersectSphere(o, dir, sc.sph[prim - sc.nTris], tmin, h);
                if (blocked) return true;
            }
        } else {
            double tc;
            if (boxHit(sc.nodes[n.left],  o, invD, tmin, tMax, tc)) stack[sp++] = n.left;
            if (boxHit(sc.nodes[n.right], o, invD, tmin, tMax, tc)) stack[sp++] = n.right;
        }
    }
    return false;
}

// ============================ material interactions ============================

__device__ static void refractOrReflect(const DMaterial& m, const DHit& h, const DVec3& d,
                                         double lambda, DRng& rng, DVec3& ro, DVec3& rd) {
    double ng = specLookup(m.ior, lambda);
    bool entering = dot(d, h.ng) < 0.0;
    DVec3 nl = entering ? h.ng : -h.ng;
    double n1 = entering ? 1.0 : ng, n2 = entering ? ng : 1.0;
    double eta = n1 / n2;
    double cosI = -dot(d, nl);
    double sin2t = eta * eta * (1.0 - cosI * cosI);
    DVec3 outDir;
    if (sin2t > 1.0) outDir = reflectv(d, nl);
    else {
        double cosT = sqrt(1.0 - sin2t);
        double rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
        double rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
        double R = 0.5 * (rs * rs + rp * rp);
        if (rng.uniform() < R) outDir = reflectv(d, nl);
        else outDir = d * eta + nl * (eta * cosI - cosT);
    }
    outDir = normalize(outDir);
    ro = h.p + outDir * 1e-6; rd = outDir;
}
__device__ static void thinFilmInterface(const DMaterial& m, const DHit& h, const DVec3& d,
                                          double lambda, DRng& rng, DVec3& ro, DVec3& rd) {
    double ns = specLookup(m.ior, lambda), nf = m.filmIor;
    bool entering = dot(d, h.ng) < 0.0;
    DVec3 nl = entering ? h.ng : -h.ng;
    double nA = entering ? 1.0 : ns, nB = entering ? ns : 1.0;
    double eta = nA / nB;
    double cosI = -dot(d, nl);
    double sin2t = eta * eta * (1.0 - cosI * cosI);
    DVec3 outDir;
    if (sin2t > 1.0) outDir = reflectv(d, nl);
    else {
        double cosT = sqrt(1.0 - sin2t);
        double R = thinFilmReflectance(nA, nf, nB, m.filmThickness, cosI, lambda);
        if (rng.uniform() < R) outDir = reflectv(d, nl);
        else outDir = d * eta + nl * (eta * cosI - cosT);
    }
    outDir = normalize(outDir);
    ro = h.p + outDir * 1e-6; rd = outDir;
}
// Grating diffraction (port of render.h gratingDiffract). Returns false if absorbed.
__device__ static bool gratingDiffract(const DMaterial& m, const DHit& h, const DVec3& din,
                                        double lambda, int diffraction, DRng& rng,
                                        DVec3& ro, DVec3& rd) {
    DVec3 nl = dot(din, h.ng) < 0.0 ? h.ng : -h.ng;
    DVec3 g = m.grooveDir - nl * dot(m.grooveDir, nl);
    if (dot(g, g) < 1e-12)
        g = fabs(nl.x) < 0.9 ? cross(nl, DVec3{1,0,0}) : cross(nl, DVec3{0,1,0});
    g = normalize(g);
    DVec3 t = normalize(cross(nl, g));
    DVec3 ut = din - nl * dot(din, nl);
    int M = diffraction ? (m.gratingMaxOrder < 0 ? 0 : (m.gratingMaxOrder > 32 ? 32 : m.gratingMaxOrder)) : 0;
    double lod = lambda / m.grooveSpacing;
    int ord[65]; double wgt[65]; int cnt = 0; double wsum = 0.0;
    for (int mm = -M; mm <= M; ++mm) {
        DVec3 a = ut + t * ((double)mm * lod);
        if (dot(a, a) >= 1.0) continue;
        double w = 1.0 / (1.0 + (mm < 0 ? -mm : mm));
        ord[cnt] = mm; wgt[cnt] = w; wsum += w; ++cnt;
    }
    if (cnt == 0 || wsum <= 0.0) return false;
    double xi = rng.uniform() * wsum, acc = 0.0; int pick = ord[cnt - 1];
    for (int i = 0; i < cnt; ++i) { acc += wgt[i]; if (xi < acc) { pick = ord[i]; break; } }
    DVec3 a = ut + t * ((double)pick * lod);
    DVec3 v = a + nl * sqrt(fmax(0.0, 1.0 - dot(a, a)));
    v = normalize(v);
    ro = h.p + nl * 1e-6; rd = v;
    return true;
}

// ============================ model-B connect / splat ============================

__device__ static void filmAdd(double* film, int resX, int px, int py, double lambda, double w) {
    size_t idx = ((size_t)py * resX + px) * 3;
    atomicAdd(&film[idx + 0], cieX(lambda) * w);
    atomicAdd(&film[idx + 1], cieY(lambda) * w);
    atomicAdd(&film[idx + 2], cieZ(lambda) * w);
}
__device__ static void connect(const DScene& sc, const DCamera& cam, double* film,
                               const DVec3& p, const DVec3& n, double lambda, double beta, double rho) {
    DVec3 toCam = cam.eye - p;
    double dist = length(toCam);
    DVec3 wdir = toCam / dist;
    double cosSurf = dot(n, wdir);
    if (cosSurf <= 0) return;
    int px, py; double cosCam, dist2;
    if (!cam.project(p, px, py, cosCam, dist2)) return;
    if (occluded(sc, p + n * 1e-6, wdir, dist - 2e-6)) return;
    double f = rho / DPI;
    double G = cosSurf * cosCam / dist2;
    double We = 1.0 / (cam.imagePlaneArea() * cosCam * cosCam * cosCam * cosCam);
    double contrib = beta * f * G * We;
    if (sc.medium.enabled) contrib *= exp(-medSigmaT(sc.medium, lambda) * dist);
    filmAdd(film, cam.resX, px, py, lambda, contrib);
}
__device__ static void connectVolume(const DScene& sc, const DCamera& cam, double* film,
                                      const DVec3& p, const DVec3& wIn, double lambda, double beta) {
    DVec3 toCam = cam.eye - p;
    double dist = length(toCam);
    DVec3 wdir = toCam / dist;
    int px, py; double cosCam, dist2;
    if (!cam.project(p, px, py, cosCam, dist2)) return;
    if (occluded(sc, p + wdir * 1e-6, wdir, dist - 2e-6)) return;
    double ph = hgPhase(dot(wIn, wdir), sc.medium.g);
    double Lambda = medAlbedo(sc.medium, lambda);
    double G = cosCam / dist2;
    double We = 1.0 / (cam.imagePlaneArea() * cosCam * cosCam * cosCam * cosCam);
    double contrib = beta * Lambda * ph * G * We;
    contrib *= exp(-medSigmaT(sc.medium, lambda) * dist);
    filmAdd(film, cam.resX, px, py, lambda, contrib);
}

// ============================ megakernel ============================

__device__ static double sampleLambda(const DScene& sc, DRng& rng, double& pdf) {
    double u = rng.uniform();
    int lo = 0, hi = sc.lightCdfN - 1;
    while (lo + 1 < hi) { int mid = (lo + hi) / 2; if (sc.lightCdf[mid] <= u) lo = mid; else hi = mid; }
    double c0 = sc.lightCdf[lo], c1 = sc.lightCdf[lo + 1];
    double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
    double w = DLMIN + (lo + frac) * sc.lightStep;
    pdf = (c1 - c0) / sc.lightStep;
    return w;
}

__global__ void kTrace(DScene sc, DCamera cam, double* film, double* energy,
                       long long N, int diffraction, unsigned long long seedBase, int maxBounce) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    DRng rng; rng.seed((unsigned long long)(g * 2 + 1), seedBase ^ (unsigned long long)g);

    double eEmitted = 0, eAbsorbed = 0, eSensor = 0, eEscaped = 0, eResidual = 0;

    for (long long i = g; i < N; i += G) {
        double u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 origin = sc.lightOrigin + sc.lightU * u1 + sc.lightV * u2;
        DVec3 dir = sc.collimated ? sc.beamDir : cosineHemisphere(sc.lightNormal, rng);
        double pdfL = 0.0;
        double lambda = sampleLambda(sc, rng, pdfL);
        if (pdfL <= 0) continue;
        double beta = sc.lightEmitIntegral * sc.lightArea * DPI;
        eEmitted += beta;

        connect(sc, cam, film, origin, sc.lightNormal, lambda, beta, 1.0);

        DVec3 ro = origin + dir * 1e-6, rd = dir;
        bool done = false;
        for (int bounce = 0; bounce < maxBounce && !done; ++bounce) {
            DHit h = closestHit(sc, ro, rd);
            double dSurf = h.valid ? h.t : 1e30;

            // fog free-flight
            bool mediumEvent = false; DVec3 mp;
            if (sc.medium.enabled) {
                double st = medSigmaT(sc.medium, lambda);
                if (st > 0.0) {
                    double tMed = -log(1.0 - rng.uniform()) / st;
                    if (tMed < dSurf) { mediumEvent = true; mp = ro + rd * tMed; }
                }
            }
            if (mediumEvent) {
                connectVolume(sc, cam, film, mp, rd, lambda, beta);
                if (rng.uniform() >= medAlbedo(sc.medium, lambda)) { eAbsorbed += beta; done = true; break; }
                DVec3 nd = sampleHG(rd, sc.medium.g, rng);
                ro = mp; rd = nd;
                continue;
            }

            if (!h.valid) { eEscaped += beta; done = true; break; }
            if (h.sensorId >= 0) { eSensor += beta; done = true; break; }

            const DMaterial& m = sc.mats[h.matId];
            if (m.type == D_DIELECTRIC) {
                DVec3 nro, nrd; refractOrReflect(m, h, rd, lambda, rng, nro, nrd); ro = nro; rd = nrd; continue;
            } else if (m.type == D_THINFILM) {
                DVec3 nro, nrd; thinFilmInterface(m, h, rd, lambda, rng, nro, nrd); ro = nro; rd = nrd; continue;
            } else if (m.type == D_MIRROR) {
                double r = clamp01(specLookup(m.reflect, lambda));
                if (rng.uniform() >= r) { eAbsorbed += beta; done = true; break; }
                DVec3 o = reflectv(rd, h.n); ro = h.p + h.n * 1e-6; rd = o; continue;
            } else if (m.type == D_GRATING) {
                double r = clamp01(specLookup(m.reflect, lambda));
                if (rng.uniform() >= r) { eAbsorbed += beta; done = true; break; }
                DVec3 nro, nrd;
                if (!gratingDiffract(m, h, rd, lambda, diffraction, rng, nro, nrd)) { eAbsorbed += beta; done = true; break; }
                ro = nro; rd = nrd; continue;
            } else if (m.type == D_HALFMIRROR) {
                double r = clamp01(specLookup(m.reflect, lambda));
                if (rng.uniform() < r) { DVec3 o = reflectv(rd, h.n); ro = h.p + h.n * 1e-6; rd = o; }
                else { ro = h.p + rd * 1e-6; }
                continue;
            } else if (m.type == D_GLOSSY) {
                double r = clamp01(specLookup(m.reflect, lambda));
                if (rng.uniform() >= r) { eAbsorbed += beta; done = true; break; }
                DVec3 o = sampleGlossy(reflectv(rd, h.n), m.roughness, rng);
                if (dot(o, h.n) <= 0) { eAbsorbed += beta; done = true; break; }
                ro = h.p + h.n * 1e-6; rd = o; continue;
            } else {
                // Diffuse (Fluorescent scenes are rejected on the host; never reached).
                double rho = clamp01(specLookup(m.reflect, lambda));
                connect(sc, cam, film, h.p, h.n, lambda, beta, rho);
                if (rng.uniform() >= rho) { eAbsorbed += beta; done = true; break; }
                ro = h.p + h.n * 1e-6; rd = cosineHemisphere(h.n, rng); continue;
            }
        }
        if (!done) eResidual += beta;
    }

    atomicAdd(&energy[0], eEmitted);
    atomicAdd(&energy[1], eAbsorbed);
    atomicAdd(&energy[2], eSensor);
    atomicAdd(&energy[3], eEscaped);
    atomicAdd(&energy[4], eResidual);
}

} // namespace gpu

// ============================ host: bake + launch ============================

static bool g_queried = false, g_available = false;
static char g_devName[256] = "none";

bool cudaAvailable() {
    if (g_queried) return g_available;
    g_queried = true;
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n <= 0) { g_available = false; return false; }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::strncpy(g_devName, prop.name, sizeof(g_devName) - 1);
        g_devName[sizeof(g_devName) - 1] = '\0';
    }
    g_available = true;
    return true;
}
const char* cudaDeviceName() { cudaAvailable(); return g_devName; }

bool cudaForwardSupported(const Scene& scene) {
    // Only reject if geometry actually USES a fluorescent material — buildCornell
    // keeps a fluorescent entry in the material palette even when nothing points at
    // it, so scanning scene.mats alone would spuriously disable the GPU path.
    auto usesFluoro = [&](int matId) {
        return matId >= 0 && matId < (int)scene.mats.size() &&
               scene.mats[matId].type == MatType::Fluorescent;
    };
    for (const auto& t : scene.tris)    if (usesFluoro(t.matId)) return false;
    for (const auto& s : scene.spheres) if (usesFluoro(s.matId)) return false;
    return true;
}

// Bake a Spectrum into a SPEC_N table over [DLMIN, DLMAX].
static void bakeSpec(const Spectrum& s, double* tab) {
    for (int i = 0; i < SPEC_N; ++i) {
        double w = DLMIN + (double)i / (SPEC_N - 1) * (DLMAX - DLMIN);
        tab[i] = s ? s(w) : 0.0;
    }
}

template <class T>
static T* uploadVec(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

Film renderForwardCudaMB(const Scene& scene, const Camera& cam, int res,
                         long long N, EnergyReport& eOut, bool diffraction) {
    using namespace gpu;
    Film out; out.resX = res; out.resY = res; out.alloc();
    if (!cudaAvailable() || !cudaForwardSupported(scene)) return out;

    // --- bake geometry ---
    std::vector<DTri> tris(scene.tris.size());
    for (size_t i = 0; i < scene.tris.size(); ++i) {
        const Tri& t = scene.tris[i]; DTri& d = tris[i];
        d.v0 = {t.v0.x, t.v0.y, t.v0.z}; d.v1 = {t.v1.x, t.v1.y, t.v1.z};
        d.v2 = {t.v2.x, t.v2.y, t.v2.z}; d.gn = {t.gn.x, t.gn.y, t.gn.z};
        d.matId = t.matId; d.sensorId = t.sensorId;
    }
    std::vector<DSphere> sph(scene.spheres.size());
    for (size_t i = 0; i < scene.spheres.size(); ++i) {
        const Sphere& s = scene.spheres[i]; DSphere& d = sph[i];
        d.c = {s.c.x, s.c.y, s.c.z}; d.r = s.r; d.matId = s.matId;
    }
    std::vector<DNode> nodes(scene.bvh.nodes.size());
    for (size_t i = 0; i < scene.bvh.nodes.size(); ++i) {
        const BvhNode& b = scene.bvh.nodes[i]; DNode& d = nodes[i];
        d.lo = {b.box.lo.x, b.box.lo.y, b.box.lo.z};
        d.hi = {b.box.hi.x, b.box.hi.y, b.box.hi.z};
        d.left = b.left; d.right = b.right; d.first = b.first; d.count = b.count;
    }
    std::vector<int> primIdx = scene.bvh.primIdx;

    // --- bake materials ---
    std::vector<DMaterial> mats(scene.mats.size());
    for (size_t i = 0; i < scene.mats.size(); ++i) {
        const Material& m = scene.mats[i]; DMaterial& d = mats[i];
        d.type = (int)m.type;
        bakeSpec(m.reflect, d.reflect);
        bakeSpec(m.ior, d.ior);
        d.roughness = m.roughness;
        d.filmIor = m.filmIor; d.filmThickness = m.filmThickness;
        d.grooveSpacing = m.grooveSpacing;
        d.grooveDir = {m.grooveDir.x, m.grooveDir.y, m.grooveDir.z};
        d.gratingMaxOrder = m.gratingMaxOrder;
    }

    // --- upload ---
    DTri*      d_tris = tris.empty() ? nullptr : uploadVec(tris);
    DSphere*   d_sph  = sph.empty()  ? nullptr : uploadVec(sph);
    DNode*     d_nodes = nodes.empty() ? nullptr : uploadVec(nodes);
    int*       d_prim = primIdx.empty() ? nullptr : uploadVec(primIdx);
    DMaterial* d_mats = mats.empty() ? nullptr : uploadVec(mats);
    std::vector<double> cdf = scene.lightSpd.cdf;
    double* d_cdf = cdf.empty() ? nullptr : uploadVec(cdf);

    DScene sc;
    sc.tris = d_tris; sc.nTris = (int)tris.size();
    sc.sph = d_sph;   sc.nSph = (int)sph.size();
    sc.mats = d_mats;
    sc.nodes = d_nodes; sc.primIdx = d_prim; sc.nNodes = (int)nodes.size();
    sc.lightOrigin = {scene.lightOrigin.x, scene.lightOrigin.y, scene.lightOrigin.z};
    sc.lightU = {scene.lightU.x, scene.lightU.y, scene.lightU.z};
    sc.lightV = {scene.lightV.x, scene.lightV.y, scene.lightV.z};
    sc.lightNormal = {scene.lightNormal.x, scene.lightNormal.y, scene.lightNormal.z};
    sc.lightArea = scene.lightArea;
    sc.lightEmitIntegral = scene.lightEmitIntegral;
    sc.collimated = scene.collimated ? 1 : 0;
    sc.beamDir = {scene.beamDir.x, scene.beamDir.y, scene.beamDir.z};
    sc.lightCdf = d_cdf; sc.lightCdfN = (int)cdf.size(); sc.lightStep = scene.lightSpd.step;
    sc.medium.enabled = scene.medium.enabled ? 1 : 0;
    sc.medium.g = scene.medium.g;
    bakeSpec(scene.medium.sigma_a, sc.medium.sigma_a);
    bakeSpec(scene.medium.sigma_s, sc.medium.sigma_s);

    DCamera dc;
    dc.eye = {cam.eye.x, cam.eye.y, cam.eye.z};
    dc.u = {cam.u.x, cam.u.y, cam.u.z};
    dc.v = {cam.v.x, cam.v.y, cam.v.z};
    dc.w = {cam.w.x, cam.w.y, cam.w.z};
    dc.tanHalfX = cam.tanHalfX; dc.tanHalfY = cam.tanHalfY;
    dc.resX = res; dc.resY = res;

    double* d_film = nullptr;   cudaMalloc(&d_film, (size_t)res * res * 3 * sizeof(double));
    cudaMemset(d_film, 0, (size_t)res * res * 3 * sizeof(double));
    double* d_energy = nullptr; cudaMalloc(&d_energy, 5 * sizeof(double));
    cudaMemset(d_energy, 0, 5 * sizeof(double));

    int blockSize = 128;
    int numBlocks = 2048;          // ~262k threads, grid-stride over N photons
    kTrace<<<numBlocks, blockSize>>>(sc, dc, d_film, d_energy, N, diffraction ? 1 : 0,
                                     0x9e3779b97f4a7c15ULL, 32);
    cudaError_t kerr = cudaGetLastError();
    if (kerr == cudaSuccess) kerr = cudaDeviceSynchronize();
    if (kerr != cudaSuccess) {
        std::fprintf(stderr, "[cuda] kernel error: %s\n", cudaGetErrorString(kerr));
    }

    // --- download ---
    std::vector<double> film((size_t)res * res * 3);
    cudaMemcpy(film.data(), d_film, film.size() * sizeof(double), cudaMemcpyDeviceToHost);
    double energy[5] = {0,0,0,0,0};
    cudaMemcpy(energy, d_energy, 5 * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < (size_t)res * res; ++i)
        out.xyz[i] = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);
    eOut.emitted  += energy[0];
    eOut.absorbed += energy[1];
    eOut.sensor   += energy[2];
    eOut.escaped  += energy[3];
    eOut.residual += energy[4];

    cudaFree(d_tris); cudaFree(d_sph); cudaFree(d_nodes); cudaFree(d_prim);
    cudaFree(d_mats); cudaFree(d_cdf); cudaFree(d_film); cudaFree(d_energy);
    return out;
}
