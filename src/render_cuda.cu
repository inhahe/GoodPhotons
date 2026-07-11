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

// ---------------------------------------------------------------------------
// GPU runtime abstraction (CUDA today, HIP-ready for AMD).
//
// Everything below the launch site is written in the portable subset of the
// CUDA/HIP device language: __global__/__device__ kernels, grid-stride loops,
// double atomicAdd, and triple-chevron <<<>>> launches all exist verbatim in
// HIP. The ONLY vendor-specific surface is the host RUNTIME API (device query,
// malloc/memcpy/memset/free, error strings, synchronize). We isolate that here:
// building with -DFTRACE_USE_HIP (or under hipcc, which defines
// __HIP_PLATFORM_AMD__) includes the HIP runtime and maps the cuda* symbols we
// use onto their hip* equivalents, which are 1:1 in name and signature. Under
// nvcc nothing changes. Porting to ROCm is therefore a build-system change
// (compile this file with hipcc, define FTRACE_USE_HIP) — not a code rewrite.
#if defined(FTRACE_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define cudaError_t             hipError_t
  #define cudaSuccess             hipSuccess
  #define cudaGetDeviceCount      hipGetDeviceCount
  #define cudaGetDeviceProperties hipGetDeviceProperties
  #define cudaDeviceProp          hipDeviceProp_t
  #define cudaMalloc              hipMalloc
  #define cudaMemcpy              hipMemcpy
  #define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
  #define cudaMemset              hipMemset
  #define cudaFree                hipFree
  #define cudaGetLastError        hipGetLastError
  #define cudaDeviceSynchronize   hipDeviceSynchronize
  #define cudaGetErrorString      hipGetErrorString
#else
  #include <cuda_runtime.h>
#endif

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

// Device transport scalar. Consumer GeForce GPUs run FP64 at ~1/64 of FP32, so the
// megakernel does its geometry/BRDF/spectral math in Real (float by default) while
// the FILM and ENERGY accumulators stay double (mixed precision: compute in float,
// accumulate in double). Configure -DFTRACE_GPU_FP32=OFF to build the exact-FP64
// device path (matches the CPU reference bit-for-bit closer, but far slower on
// GeForce). The CPU renderer is always double and remains the ground-truth.
#ifndef FTRACE_GPU_FP32
#define FTRACE_GPU_FP32 1
#endif
#if FTRACE_GPU_FP32
using Real = float;
static constexpr Real RAY_EPS = 1e-4f;   // self-intersection offset (float-safe at unit scale)
static constexpr Real DET_EPS = 1e-6f;   // triangle determinant reject (float-safe)
static constexpr Real BIG     = 1e30f;   // "no hit" sentinel distance
#else
using Real = double;
static constexpr Real RAY_EPS = 1e-6;
static constexpr Real DET_EPS = 1e-9;
static constexpr Real BIG     = 1e30;
#endif

// DVec3 stores Real and does Real arithmetic (the hot path), but its 3-arg
// constructor keeps DOUBLE parameters so the host baking code's brace-init from
// double Scene coordinates ({v.x, v.y, v.z}) is a widening conversion (legal),
// never a narrowing one. The float<->double round-trip at construction is exact.
struct DVec3 {
    Real x, y, z;
    HD DVec3() : x(0), y(0), z(0) {}
    HD DVec3(double a, double b, double c) : x((Real)a), y((Real)b), z((Real)c) {}
    HD DVec3 operator+(const DVec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD DVec3 operator-(const DVec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD DVec3 operator*(Real s)         const { return {x * s, y * s, z * s}; }
    HD DVec3 operator/(Real s)         const { return {x / s, y / s, z / s}; }
    HD DVec3 operator-()               const { return {-x, -y, -z}; }
};
HD static inline Real dot(const DVec3& a, const DVec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
HD static inline DVec3 cross(const DVec3& a, const DVec3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
HD static inline Real length(const DVec3& a) { return sqrt(dot(a, a)); }
HD static inline DVec3 normalize(const DVec3& a) { return a / length(a); }
HD static inline DVec3 reflectv(const DVec3& d, const DVec3& n) { return d - n * (2 * dot(d, n)); }
HD static inline void onb(const DVec3& n, DVec3& t, DVec3& b) {
    Real sign = copysign((Real)1, n.z);
    Real a = (Real)-1 / (sign + n.z);
    Real d = n.x * n.y * a;
    t = DVec3(1 + sign * n.x * n.x * a, sign * d, -sign * n.x);
    b = DVec3(d, sign + n.y * n.y * a, -n.y);
}
HD static inline Real clamp01(Real x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

// Material type tags (must match MatType order in scene.h).
enum { D_DIFFUSE=0, D_DIELECTRIC, D_MIRROR, D_HALFMIRROR, D_GLOSSY, D_FLUORESCENT, D_THINFILM, D_GRATING, D_MIX, D_MULTILAYER };

// Maximum child lobes in a Mix material on the GPU. Scenes whose mix materials
// exceed this fall back to the CPU forward tracer (cudaForwardSupported).
#define D_MIXMAX 8

// Maximum layers in a Multilayer stack on the GPU. Deeper stacks fall back to CPU.
#define D_MAXLAYERS 16

// Camera measurement model (mirrors -mode A/B/C).
enum { CAM_A = 0, CAM_B = 1, CAM_C = 2 };

struct DMaterial {
    int    type;
    double reflect[SPEC_N];     // baked reflect spectrum
    double ior[SPEC_N];         // baked index spectrum
    double substrateK[SPEC_N];  // baked thin-film substrate extinction kappa (0 = transparent)
    double roughness;
    double filmIor, filmThickness;
    // Multilayer stack (D_MULTILAYER): per-layer index/extinction/thickness; the
    // substrate is ior + substrateK (spectral). layer 0 is outermost.
    int    layerCount;
    double layerN[D_MAXLAYERS], layerK[D_MAXLAYERS], layerThick[D_MAXLAYERS];
    double grooveSpacing;
    DVec3  grooveDir;
    int    gratingMaxOrder;
    // Stochastic mix (D_MIX): pick child mixChild[k] with prob mixWeight[k];
    // leftover (1 - sum) absorbs. Resolved before the material switch.
    int    mixCount;
    int    mixChild[D_MIXMAX];
    double mixWeight[D_MIXMAX];
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

// One emitter (mirrors host Emitter). `cdfOffset`/`cdfN` index this emitter's
// wavelength CDF slice inside the flattened lightCdfAll buffer.
struct DEmitter {
    DVec3  origin, u, v, normal, beamDir;
    double area, power;
    int    collimated;
    int    shape;              // 0 = quad, 1 = sphere, 2 = spot, 3 = env (mirrors EmitterShape)
    double radius;             // sphere radius (shape==1)
    double spotCosInner, spotCosOuter, spotOmega;   // spot cone (shape==2)
    int    cdfOffset, cdfN;
    double cdfStep;
    // BDPT (mode D) extras. matId links this emitter to its emissive surface material
    // (for the s=0 direct-hit strategy); emitSpd is the baked emission SPD so the
    // device can evaluate Le(lambda) directly (DMaterial carries no emit spectrum).
    int    matId;
    double emitSpd[SPEC_N];
};

// Smoothstep spot falloff (mirrors host scene.h spotFalloff).
__device__ static double spotFalloff(double ct, double cosInner, double cosOuter) {
    if (ct >= cosInner) return 1.0;
    if (ct <= cosOuter) return 0.0;
    double t = (ct - cosOuter) / (cosInner - cosOuter);
    return t * t * (3.0 - 2.0 * t);
}

// Sample a surface point + outward normal on an emitter (mirrors host
// Emitter::samplePoint). Quad draws are unchanged, so quad scenes stay parity.
__device__ static void emitterSamplePoint(const DEmitter& em, double u1, double u2,
                                          DVec3& y, DVec3& nOut) {
    if (em.shape == 1) {
        double z = 1.0 - 2.0 * u1;
        double r = sqrt(fmax(0.0, 1.0 - z * z));
        double phi = 2.0 * 3.14159265358979323846 * u2;
        DVec3 d{(Real)(r * cos(phi)), (Real)(r * sin(phi)), (Real)z};
        nOut = d;
        y = em.origin + d * (Real)em.radius;
    } else {
        y = em.origin + em.u * (Real)u1 + em.v * (Real)u2;
        nOut = em.normal;
    }
}

// Image-based (lat-long) environment tables (mirrors host EnvMap). Its pointers are
// non-null only when scene.envMap is set; the constant-env path never touches these.
// The 2D luminance sampler is flattened: a marginal Distribution1D over rows (h bins)
// plus one conditional Distribution1D per row (w bins), row v's slice at the v-th
// offset. The forward reweight needs radiance(dir,lambda)/avgSpd(lambda), in which
// the shared illuminant factor cancels, so only the per-texel JH coeff/scale and the
// mean coeff/scale are uploaded — no illuminant table on the device.
struct DEnvMap {
    int    w = 0, h = 0;                  // == nu, nv of the 2D distribution
    double rot = 0.0;                     // horizontal rotation in [0,1) turns
    const double* coeff = nullptr;        // 3*w*h : texel i -> coeff[3i .. 3i+2]
    const double* scale = nullptr;        // w*h   : per-texel brightness (non-null => image env)
    double avgCoeff[3] = {0, 0, 0};
    double avgScale = 0.0;
    const double* margCdf     = nullptr;  // h+1
    const double* margFunc    = nullptr;  // h
    double        margFuncInt = 0.0;
    const double* condCdf     = nullptr;  // h*(w+1) : row v at v*(w+1)
    const double* condFunc    = nullptr;  // h*w     : row v at v*w
    const double* condFuncInt = nullptr;  // h
};

struct DScene {
    const DTri*      tris;  int nTris;
    const DSphere*   sph;   int nSph;
    const DMaterial* mats;
    const DNode*     nodes; const int* primIdx; int nNodes;
    const DEmitter*  emitters; int nEmitters;
    const double*    emitCdf;       // size nEmitters, cumulative power, normalised
    double           totalPower;
    const double*    lightCdfAll;   // flattened per-emitter wavelength CDFs
    // BDPT shared wavelength sampler (mirrors Scene::emitSampler): the combined
    // g(lambda)=sum_k geomWeight_k*SPD_k CDF, its bin step, and emitG = its integral.
    // BDPT samples one shared lambda per sample from this and sets invPdfLambda=1/pdf.
    const double*    emitSamplerCdf; int emitSamplerN; double emitSamplerStep;
    double           emitG;
    DMedium medium;
    DVec3  sensorOrigin, sensorUAxis, sensorVAxis;   // model A contact sensor plane
    DVec3  sceneCenter;              // env (shape==3): bounding-sphere center
    double sceneRadius;              // env (shape==3): bounding-sphere radius
    DEnvMap env;                     // image env tables (env.scale null => constant env)
};

struct DCamera {
    DVec3  eye, u, v, w;
    double tanHalfX, tanHalfY;
    int    resX, resY;
    double apertureR, filmDist, lensF;   // model C finite aperture / thin lens
    HD double imagePlaneArea() const { return 4.0 * tanHalfX * tanHalfY; }
    // Per-pixel image-plane area: connect() splats one photon into one pixel, so the
    // pinhole importance normalises by a single pixel's area (see camera.h). This
    // makes the GPU forward tracer measure absolute radiance, matching the CPU path.
    HD double pixelPlaneArea() const {
        return imagePlaneArea() / ((double)resX * (double)resY);
    }
    HD bool project(const DVec3& p, int& px, int& py, Real& cosCam, Real& dist2) const {
        DVec3 d = p - eye;
        Real cz = dot(d, w);
        if (cz <= (Real)1e-9) return false;
        Real cx = dot(d, u), cy = dot(d, v);
        Real ix = (cx / cz) / (Real)tanHalfX, iy = (cy / cz) / (Real)tanHalfY;
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
        px = (int)((ix * (Real)0.5 + (Real)0.5) * resX);
        py = (int)((iy * (Real)0.5 + (Real)0.5) * resY);
        dist2 = dot(d, d);
        cosCam = cz / sqrt(dist2);
        return true;
    }
    // Model A/C perspective catch: does this photon fly through the finite aperture
    // disc (before hitting the scene, within hitDist) and land on the film? Port of
    // Camera::catchPhoton, including the thin-lens paraxial refraction u' = u - rho/f.
    HD bool catchPhoton(const DVec3& ro, const DVec3& rd, Real hitDist, int& px, int& py) const {
        Real dw = dot(rd, w);
        if (dw >= (Real)-1e-9) return false;
        Real tAp = dot(eye - ro, w) / dw;
        if (tAp <= RAY_EPS || tAp >= hitDist) return false;
        DVec3 P = ro + rd * tAp;
        DVec3 rho = P - eye;
        if (dot(rho, rho) > (Real)(apertureR * apertureR)) return false;
        DVec3 nAxis = w * (Real)-1;
        DVec3 dir = rd;
        if (lensF > 0.0) {
            Real dax = dot(dir, nAxis);
            DVec3 slope = (dir - nAxis * dax) / dax;
            DVec3 slopeP = slope - rho * (Real)(1.0 / lensF);
            dir = normalize(nAxis + slopeP);
        }
        Real ddax = dot(dir, nAxis);
        if (ddax <= (Real)1e-9) return false;
        Real s = (Real)filmDist / ddax;
        DVec3 Fcenter = eye + nAxis * (Real)filmDist;
        DVec3 Q = P + dir * s;
        DVec3 rel = Q - Fcenter;
        Real ix = -dot(rel, u) / (Real)(filmDist * tanHalfX);
        Real iy = -dot(rel, v) / (Real)(filmDist * tanHalfY);
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
        px = (int)((ix * (Real)0.5 + (Real)0.5) * resX);
        py = (int)((iy * (Real)0.5 + (Real)0.5) * resY);
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
    __device__ Real uniform() { return (next() >> 8) * (Real)(1.0 / 16777216.0); }
};

__device__ static DVec3 cosineHemisphere(const DVec3& n, DRng& rng) {
    Real u1 = rng.uniform(), u2 = rng.uniform();
    Real r = sqrt(u1), phi = (Real)6.283185307179586 * u2;
    Real lx = r * cos(phi), ly = r * sin(phi), lz = sqrt((Real)1 - u1);
    DVec3 t, b; onb(n, t, b);
    return normalize(t * lx + b * ly + n * lz);
}
__device__ static DVec3 sampleGlossy(const DVec3& mdir, Real roughness, DRng& rng) {
    Real rr = roughness < (Real)1e-3 ? (Real)1e-3 : roughness;
    Real e = (Real)2 / (rr * rr) - (Real)2; if (e < 0) e = 0;
    Real u1 = rng.uniform(), u2 = rng.uniform();
    Real cosT = pow(u1, (Real)1 / (e + (Real)1));
    Real sinT = sqrt(fmax((Real)0, (Real)1 - cosT * cosT));
    Real phi = (Real)2 * (Real)DPI * u2;
    DVec3 t, b; onb(mdir, t, b);
    return normalize(t * (sinT * cos(phi)) + b * (sinT * sin(phi)) + mdir * cosT);
}
__device__ static Real hgPhase(Real cosTheta, Real g) {
    Real d = (Real)1 + g * g - (Real)2 * g * cosTheta;
    if (d < (Real)1e-9) d = (Real)1e-9;
    return ((Real)1 - g * g) / ((Real)4 * (Real)DPI * d * sqrt(d));
}
__device__ static DVec3 sampleHG(const DVec3& wi, Real g, DRng& rng) {
    Real u1 = rng.uniform(), u2 = rng.uniform(), cosT;
    if (fabs(g) < (Real)1e-3) cosT = (Real)1 - (Real)2 * u1;
    else { Real sq = ((Real)1 - g * g) / ((Real)1 + g - (Real)2 * g * u1); cosT = ((Real)1 + g * g - sq * sq) / ((Real)2 * g); }
    Real sinT = sqrt(fmax((Real)0, (Real)1 - cosT * cosT));
    Real phi = (Real)2 * (Real)DPI * u2;
    DVec3 t, b; onb(wi, t, b);
    return normalize(t * (sinT * cos(phi)) + b * (sinT * sin(phi)) + wi * cosT);
}

// CIE 1931 CMF (analytic multi-Gaussian fit — same as color.h).
__device__ static Real gaussPiece(Real x, Real mu, Real s1, Real s2) {
    Real t = (x - mu) * ((x < mu) ? s1 : s2);
    return exp((Real)-0.5 * t * t);
}
__device__ static Real cieX(Real w) {
    return (Real)0.362 * gaussPiece(w, 442.0, 0.0624, 0.0374)
         + (Real)1.056 * gaussPiece(w, 599.8, 0.0264, 0.0323)
         - (Real)0.065 * gaussPiece(w, 501.1, 0.0490, 0.0382);
}
__device__ static Real cieY(Real w) {
    return (Real)0.821 * gaussPiece(w, 568.8, 0.0213, 0.0247)
         + (Real)0.286 * gaussPiece(w, 530.9, 0.0613, 0.0322);
}
__device__ static Real cieZ(Real w) {
    return (Real)1.217 * gaussPiece(w, 437.0, 0.0845, 0.0278)
         + (Real)0.681 * gaussPiece(w, 459.0, 0.0385, 0.0725);
}

// Spectral table lookup with linear interpolation over [DLMIN, DLMAX]. Tables stay
// double (host-baked, tiny + cached); the interpolated result is returned as Real.
__device__ static Real specLookup(const double* tab, Real lambda) {
    Real f = (lambda - (Real)DLMIN) / (Real)(DLMAX - DLMIN) * (SPEC_N - 1);
    if (f <= 0) return (Real)tab[0];
    if (f >= SPEC_N - 1) return (Real)tab[SPEC_N - 1];
    int i = (int)f; Real frac = f - i;
    return (Real)tab[i] * ((Real)1 - frac) + (Real)tab[i + 1] * frac;
}
__device__ static Real medSigmaT(const DMedium& m, Real lambda) {
    Real a = specLookup(m.sigma_a, lambda), s = specLookup(m.sigma_s, lambda);
    Real v = fmax((Real)0, a) + fmax((Real)0, s);
    return v;
}
__device__ static Real medAlbedo(const DMedium& m, Real lambda) {
    Real s = fmax((Real)0, specLookup(m.sigma_s, lambda));
    Real t = s + fmax((Real)0, specLookup(m.sigma_a, lambda));
    return t > 0 ? s / t : 0;
}

// Minimal device complex (host uses std::complex; not available in device code).
struct DCplx {
    Real re, im;
    __device__ DCplx() : re(0), im(0) {}
    __device__ DCplx(Real r, Real i) : re(r), im(i) {}
};
__device__ static inline DCplx cadd(DCplx a, DCplx b) { return DCplx(a.re + b.re, a.im + b.im); }
__device__ static inline DCplx csub(DCplx a, DCplx b) { return DCplx(a.re - b.re, a.im - b.im); }
__device__ static inline DCplx cmul(DCplx a, DCplx b) {
    return DCplx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
__device__ static inline DCplx cdiv(DCplx a, DCplx b) {
    Real den = b.re * b.re + b.im * b.im;
    return DCplx((a.re * b.re + a.im * b.im) / den, (a.im * b.re - a.re * b.im) / den);
}
__device__ static inline Real cnorm(DCplx a) { return a.re * a.re + a.im * a.im; }
__device__ static inline DCplx csqrt_(DCplx z) {  // principal root, then force Im>=0
    Real r = sqrt(z.re * z.re + z.im * z.im);
    Real re = sqrt(fmax((Real)0, (Real)0.5 * (r + z.re)));
    Real im = sqrt(fmax((Real)0, (Real)0.5 * (r - z.re)));
    if (z.im < 0) im = -im;
    DCplx s(re, im);
    return s.im < 0 ? DCplx(-s.re, -s.im) : s;
}
__device__ static inline DCplx crmul(Real a, DCplx b) { return DCplx(a * b.re, a * b.im); }
__device__ static inline DCplx cimul(DCplx a) { return DCplx(-a.im, a.re); }  // i*a
__device__ static inline DCplx ccos_(DCplx z) {  // cos(a+bi) = cos a cosh b - i sin a sinh b
    return DCplx(cos(z.re) * cosh(z.im), -sin(z.re) * sinh(z.im));
}
__device__ static inline DCplx csin_(DCplx z) {  // sin(a+bi) = sin a cosh b + i cos a sinh b
    return DCplx(sin(z.re) * cosh(z.im), cos(z.re) * sinh(z.im));
}

// Multilayer stack reflectance (Abeles characteristic matrix; port of render.h
// multilayerReflectance). nL/kL/dL are the per-layer index/extinction/thickness,
// nLayers entries; substrate ns + i*ks. Returns R in [0,1].
__device__ static Real multilayerReflectance(Real n0, Real cosI, Real lambda,
                                             const double* nL, const double* kL,
                                             const double* dL, int nLayers,
                                             Real ns, Real ks) {
    cosI = clamp01(fabs(cosI));
    Real sin0_2 = fmax((Real)0, (Real)1 - cosI * cosI);
    Real n0s = n0 * n0 * sin0_2;
    Real q0 = n0 * cosI;
    Real Racc = 0;
    for (int pol = 0; pol < 2; ++pol) {
        bool pPol = (pol == 1);
        DCplx M00(1, 0), M01(0, 0), M10(0, 0), M11(1, 0);
        for (int j = 0; j < nLayers; ++j) {
            DCplx nj((Real)nL[j], (Real)kL[j]);
            DCplx qj = csqrt_(csub(cmul(nj, nj), DCplx(n0s, 0)));
            DCplx eta = pPol ? cdiv(cmul(nj, nj), qj) : qj;
            DCplx delta = crmul((Real)(2.0 * DPI * dL[j] / (double)lambda), qj);
            DCplx c = ccos_(delta), s = csin_(delta);
            DCplx L00 = c, L01 = cdiv(cimul(s), eta), L10 = cimul(cmul(eta, s)), L11 = c;
            DCplx n00 = cadd(cmul(M00, L00), cmul(M01, L10));
            DCplx n01 = cadd(cmul(M00, L01), cmul(M01, L11));
            DCplx n10 = cadd(cmul(M10, L00), cmul(M11, L10));
            DCplx n11 = cadd(cmul(M10, L01), cmul(M11, L11));
            M00 = n00; M01 = n01; M10 = n10; M11 = n11;
        }
        DCplx nsub(ns, ks);
        DCplx qs = csqrt_(csub(cmul(nsub, nsub), DCplx(n0s, 0)));
        DCplx etaS = pPol ? cdiv(cmul(nsub, nsub), qs) : qs;
        DCplx eta0 = pPol ? DCplx(n0 * n0 / q0, 0) : DCplx(q0, 0);
        DCplx B = cadd(M00, cmul(M01, etaS));
        DCplx C = cadd(M10, cmul(M11, etaS));
        DCplx r = cdiv(csub(cmul(eta0, B), C), cadd(cmul(eta0, B), C));
        Racc += clamp01(cnorm(r));
    }
    return (Real)0.5 * Racc;
}

// Thin-film Airy reflectance (port of render.h thinFilmReflectance). k2 is the
// substrate extinction coefficient: k2==0 uses the exact real-valued path (matches
// the transparent-substrate host result); k2>0 uses the complex bottom-interface
// Fresnel of an absorbing/metallic substrate (opaque structural colour).
__device__ static Real thinFilmReflectance(Real n0, Real n1, Real n2, Real k2, Real d,
                                           Real cosI, Real lambda) {
    cosI = clamp01(fabs(cosI));
    Real sin0_2 = fmax((Real)0, (Real)1 - cosI * cosI);
    Real sin1_2 = (n0 * n0) / (n1 * n1) * sin0_2;
    if (sin1_2 >= 1) return 1;
    Real cos1 = sqrt((Real)1 - sin1_2);
    if (k2 != (Real)0) {
        DCplx n2c(n2, k2);
        Real q0 = n0 * cosI, q1 = n1 * cos1;
        DCplx q2 = csqrt_(csub(cmul(n2c, n2c), DCplx(n0 * n0 * sin0_2, 0)));
        Real r01s = (q0 - q1) / (q0 + q1);
        Real r01p = (n1 * n1 * q0 - n0 * n0 * q1) / (n1 * n1 * q0 + n0 * n0 * q1);
        DCplx q1c(q1, 0);
        DCplx r12s = cdiv(csub(q1c, q2), cadd(q1c, q2));
        DCplx n2c2 = cmul(n2c, n2c);
        DCplx r12p = cdiv(csub(cmul(n2c2, q1c), DCplx(n1 * n1 * q2.re, n1 * n1 * q2.im)),
                          cadd(cmul(n2c2, q1c), DCplx(n1 * n1 * q2.re, n1 * n1 * q2.im)));
        Real phi = ((Real)4 * (Real)DPI * n1 * d * cos1) / lambda;
        DCplx p(cos(phi), sin(phi));
        // R_pol = |r01 + r12 p|^2 / |1 + r01 r12 p|^2
        DCplx numS = cadd(DCplx(r01s, 0), cmul(r12s, p));
        DCplx denS = cadd(DCplx(1, 0), cmul(DCplx(r01s, 0), cmul(r12s, p)));
        DCplx numP = cadd(DCplx(r01p, 0), cmul(r12p, p));
        DCplx denP = cadd(DCplx(1, 0), cmul(DCplx(r01p, 0), cmul(r12p, p)));
        Real Rs = clamp01(cnorm(numS) / cnorm(denS));
        Real Rp = clamp01(cnorm(numP) / cnorm(denP));
        return (Real)0.5 * (Rs + Rp);
    }
    Real sin2_2 = (n0 * n0) / (n2 * n2) * sin0_2;
    bool tir = sin2_2 >= 1;
    Real cos2 = tir ? (Real)0 : sqrt((Real)1 - sin2_2);
    Real r01s = (n0 * cosI - n1 * cos1) / (n0 * cosI + n1 * cos1);
    Real r01p = (n1 * cosI - n0 * cos1) / (n1 * cosI + n0 * cos1);
    Real r12s = tir ? (Real)1 : (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2);
    Real r12p = tir ? (Real)1 : (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2);
    Real phi  = ((Real)4 * (Real)DPI * n1 * d * cos1) / lambda;
    Real cphi = cos(phi);
    Real numS = r01s*r01s + r12s*r12s + (Real)2*r01s*r12s*cphi;
    Real denS = (Real)1 + r01s*r01s*r12s*r12s + (Real)2*r01s*r12s*cphi;
    Real numP = r01p*r01p + r12p*r12p + (Real)2*r01p*r12p*cphi;
    Real denP = (Real)1 + r01p*r01p*r12p*r12p + (Real)2*r01p*r12p*cphi;
    Real Rs = clamp01(denS > (Real)1e-12 ? numS / denS : numS);
    Real Rp = clamp01(denP > (Real)1e-12 ? numP / denP : numP);
    return (Real)0.5 * (Rs + Rp);
}

// ============================ intersection / BVH ============================

struct DHit {
    Real t; bool valid;
    DVec3 p, n, ng;
    int matId, sensorId;
};

__device__ static bool intersectTri(const DVec3& ro, const DVec3& rd, const DTri& tri,
                                     Real tmin, DHit& hit) {
    DVec3 e1 = tri.v1 - tri.v0, e2 = tri.v2 - tri.v0;
    DVec3 pv = cross(rd, e2);
    Real det = dot(e1, pv);
    if (fabs(det) < DET_EPS) return false;
    Real inv = (Real)1 / det;
    DVec3 tv = ro - tri.v0;
    Real u = dot(tv, pv) * inv;
    if (u < 0 || u > 1) return false;
    DVec3 qv = cross(tv, e1);
    Real vv = dot(rd, qv) * inv;
    if (vv < 0 || u + vv > 1) return false;
    Real t = dot(e2, qv) * inv;
    if (t < tmin || t >= hit.t) return false;
    hit.t = t; hit.p = ro + rd * t; hit.valid = true;
    hit.ng = tri.gn;
    hit.n = (dot(rd, tri.gn) < 0) ? tri.gn : -tri.gn;
    hit.matId = tri.matId; hit.sensorId = tri.sensorId;
    return true;
}
__device__ static bool intersectSphere(const DVec3& ro, const DVec3& rd, const DSphere& s,
                                        Real tmin, DHit& hit) {
    DVec3 oc = ro - s.c;
    Real a = dot(rd, rd), b = (Real)2 * dot(oc, rd), c = dot(oc, oc) - (Real)(s.r * s.r);
    Real disc = b * b - (Real)4 * a * c;
    if (disc < 0) return false;
    Real sq = sqrt(disc);
    Real t = (-b - sq) / ((Real)2 * a);
    if (t < tmin) t = (-b + sq) / ((Real)2 * a);
    if (t < tmin || t >= hit.t) return false;
    hit.t = t; hit.p = ro + rd * t; hit.valid = true;
    DVec3 ng = normalize(hit.p - s.c);
    hit.ng = ng;
    hit.n = (dot(rd, ng) < 0) ? ng : -ng;
    hit.matId = s.matId; hit.sensorId = -1;
    return true;
}
__device__ static bool boxHit(const DNode& nd, const DVec3& ro, const DVec3& invD,
                               Real tmin, Real tmax, Real& tEnter) {
    Real te = tmin, tx = tmax;
    Real lo[3] = {nd.lo.x, nd.lo.y, nd.lo.z}, hi[3] = {nd.hi.x, nd.hi.y, nd.hi.z};
    Real o[3] = {ro.x, ro.y, ro.z}, id[3] = {invD.x, invD.y, invD.z};
    for (int a = 0; a < 3; ++a) {
        Real t0 = (lo[a] - o[a]) * id[a], t1 = (hi[a] - o[a]) * id[a];
        if (t0 > t1) { Real tmp = t0; t0 = t1; t1 = tmp; }
        te = t0 > te ? t0 : te;
        tx = t1 < tx ? t1 : tx;
        if (tx < te) return false;
    }
    tEnter = te;
    return true;
}

__device__ static DHit closestHit(const DScene& sc, const DVec3& ro, const DVec3& rd,
                                   Real tmin = RAY_EPS) {
    DHit h; h.t = BIG; h.valid = false; h.matId = 0; h.sensorId = -1;
    if (sc.nNodes == 0) return h;
    DVec3 invD{(Real)1 / rd.x, (Real)1 / rd.y, (Real)1 / rd.z};
    Real tMax = BIG;
    int stack[64]; int sp = 0; stack[sp++] = 0;
    while (sp) {
        const DNode& n = sc.nodes[stack[--sp]];
        Real tE;
        if (!boxHit(n, ro, invD, tmin, tMax, tE)) continue;
        if (n.count > 0) {
            for (int i = 0; i < n.count; ++i) {
                int prim = sc.primIdx[n.first + i];
                if (prim < sc.nTris) { if (intersectTri(ro, rd, sc.tris[prim], tmin, h)) tMax = h.t; }
                else                 { if (intersectSphere(ro, rd, sc.sph[prim - sc.nTris], tmin, h)) tMax = h.t; }
            }
        } else {
            Real tL, tR;
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
                                 Real maxDist, Real tmin = RAY_EPS) {
    if (sc.nNodes == 0) return false;
    DVec3 invD{(Real)1 / dir.x, (Real)1 / dir.y, (Real)1 / dir.z};
    Real tMax = maxDist - tmin;
    int stack[64]; int sp = 0; stack[sp++] = 0;
    while (sp) {
        const DNode& n = sc.nodes[stack[--sp]];
        Real tE;
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
            Real tc;
            if (boxHit(sc.nodes[n.left],  o, invD, tmin, tMax, tc)) stack[sp++] = n.left;
            if (boxHit(sc.nodes[n.right], o, invD, tmin, tMax, tc)) stack[sp++] = n.right;
        }
    }
    return false;
}

// ============================ material interactions ============================

__device__ static void refractOrReflect(const DMaterial& m, const DHit& h, const DVec3& d,
                                         Real lambda, DRng& rng, DVec3& ro, DVec3& rd) {
    Real ng = specLookup(m.ior, lambda);
    bool entering = dot(d, h.ng) < 0;
    DVec3 nl = entering ? h.ng : -h.ng;
    Real n1 = entering ? (Real)1 : ng, n2 = entering ? ng : (Real)1;
    Real eta = n1 / n2;
    Real cosI = -dot(d, nl);
    Real sin2t = eta * eta * ((Real)1 - cosI * cosI);
    DVec3 outDir;
    if (sin2t > 1) outDir = reflectv(d, nl);
    else {
        Real cosT = sqrt((Real)1 - sin2t);
        Real rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
        Real rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
        Real R = (Real)0.5 * (rs * rs + rp * rp);
        if (rng.uniform() < R) outDir = reflectv(d, nl);
        else outDir = d * eta + nl * (eta * cosI - cosT);
    }
    outDir = normalize(outDir);
    ro = h.p + outDir * RAY_EPS; rd = outDir;
}
// Returns false if the photon is absorbed by an opaque (absorbing) substrate.
__device__ static bool thinFilmInterface(const DMaterial& m, const DHit& h, const DVec3& d,
                                          Real lambda, DRng& rng, DVec3& ro, DVec3& rd) {
    Real ns = specLookup(m.ior, lambda), nf = (Real)m.filmIor;
    Real ks = specLookup(m.substrateK, lambda);
    bool entering = dot(d, h.ng) < 0;
    DVec3 nl = entering ? h.ng : -h.ng;
    Real cosI = -dot(d, nl);
    if (ks > 0) {                                // opaque metal-backed film
        if (!entering) return false;             // inside absorbing substrate: absorbed
        Real R = thinFilmReflectance((Real)1, nf, ns, ks, (Real)m.filmThickness, cosI, lambda);
        if (rng.uniform() >= R) return false;    // transmitted -> absorbed
        DVec3 o = normalize(reflectv(d, nl));
        ro = h.p + o * RAY_EPS; rd = o;
        return true;
    }
    Real nA = entering ? (Real)1 : ns, nB = entering ? ns : (Real)1;
    Real eta = nA / nB;
    Real sin2t = eta * eta * ((Real)1 - cosI * cosI);
    DVec3 outDir;
    if (sin2t > 1) outDir = reflectv(d, nl);
    else {
        Real cosT = sqrt((Real)1 - sin2t);
        Real R = thinFilmReflectance(nA, nf, nB, (Real)0, (Real)m.filmThickness, cosI, lambda);
        if (rng.uniform() < R) outDir = reflectv(d, nl);
        else outDir = d * eta + nl * (eta * cosI - cosT);
    }
    outDir = normalize(outDir);
    ro = h.p + outDir * RAY_EPS; rd = outDir;
    return true;
}
// Multilayer stack interface (port of render.h multilayerInterface). Returns false
// if the photon is absorbed by an absorbing stack/substrate.
__device__ static bool multilayerInterface(const DMaterial& m, const DHit& h, const DVec3& d,
                                            Real lambda, DRng& rng, DVec3& ro, DVec3& rd) {
    Real ns = specLookup(m.ior, lambda);
    Real ks = specLookup(m.substrateK, lambda);
    int nL = m.layerCount;
    bool entering = dot(d, h.ng) < 0;
    DVec3 nl = entering ? h.ng : -h.ng;
    Real cosI = -dot(d, nl);
    bool anyAbs = ks > 0;
    for (int j = 0; j < nL; ++j) if (m.layerK[j] != 0) { anyAbs = true; break; }
    if (anyAbs) {                                // opaque: reflect-or-absorb
        if (!entering) return false;
        Real R = multilayerReflectance((Real)1, cosI, lambda, m.layerN, m.layerK, m.layerThick, nL, ns, ks);
        if (rng.uniform() >= R) return false;
        DVec3 o = normalize(reflectv(d, nl)); ro = h.p + o * RAY_EPS; rd = o; return true;
    }
    Real nA = entering ? (Real)1 : ns, nB = entering ? ns : (Real)1;
    Real eta = nA / nB;
    Real sin2t = eta * eta * ((Real)1 - cosI * cosI);
    DVec3 outDir;
    if (sin2t > 1) outDir = reflectv(d, nl);
    else {
        Real cosT = sqrt((Real)1 - sin2t);
        Real R;
        if (entering) R = multilayerReflectance((Real)1, cosI, lambda, m.layerN, m.layerK, m.layerThick, nL, ns, (Real)0);
        else {
            double rn[D_MAXLAYERS], rk[D_MAXLAYERS], rt[D_MAXLAYERS];
            for (int j = 0; j < nL; ++j) { rn[j] = m.layerN[nL-1-j]; rk[j] = m.layerK[nL-1-j]; rt[j] = m.layerThick[nL-1-j]; }
            R = multilayerReflectance(ns, cosI, lambda, rn, rk, rt, nL, (Real)1, (Real)0);
        }
        if (rng.uniform() < R) outDir = reflectv(d, nl);
        else outDir = d * eta + nl * (eta * cosI - cosT);
    }
    outDir = normalize(outDir); ro = h.p + outDir * RAY_EPS; rd = outDir; return true;
}
// Grating diffraction (port of render.h gratingDiffract). Returns false if absorbed.
__device__ static bool gratingDiffract(const DMaterial& m, const DHit& h, const DVec3& din,
                                        Real lambda, int diffraction, DRng& rng,
                                        DVec3& ro, DVec3& rd) {
    DVec3 nl = dot(din, h.ng) < 0 ? h.ng : -h.ng;
    DVec3 g = m.grooveDir - nl * dot(m.grooveDir, nl);
    if (dot(g, g) < (Real)1e-12)
        g = fabs(nl.x) < (Real)0.9 ? cross(nl, DVec3{1,0,0}) : cross(nl, DVec3{0,1,0});
    g = normalize(g);
    DVec3 t = normalize(cross(nl, g));
    DVec3 ut = din - nl * dot(din, nl);
    int M = diffraction ? (m.gratingMaxOrder < 0 ? 0 : (m.gratingMaxOrder > 32 ? 32 : m.gratingMaxOrder)) : 0;
    Real lod = lambda / (Real)m.grooveSpacing;
    int ord[65]; Real wgt[65]; int cnt = 0; Real wsum = 0;
    for (int mm = -M; mm <= M; ++mm) {
        DVec3 a = ut + t * ((Real)mm * lod);
        if (dot(a, a) >= 1) continue;
        Real w = (Real)1 / ((Real)1 + (mm < 0 ? -mm : mm));
        ord[cnt] = mm; wgt[cnt] = w; wsum += w; ++cnt;
    }
    if (cnt == 0 || wsum <= 0) return false;
    Real xi = rng.uniform() * wsum, acc = 0; int pick = ord[cnt - 1];
    for (int i = 0; i < cnt; ++i) { acc += wgt[i]; if (xi < acc) { pick = ord[i]; break; } }
    DVec3 a = ut + t * ((Real)pick * lod);
    DVec3 v = a + nl * sqrt(fmax((Real)0, (Real)1 - dot(a, a)));
    v = normalize(v);
    ro = h.p + nl * RAY_EPS; rd = v;
    return true;
}

// ============================ model-B connect / splat ============================

__device__ static void filmAdd(double* film, int resX, int px, int py, Real lambda, Real w) {
    size_t idx = ((size_t)py * resX + px) * 3;
    atomicAdd(&film[idx + 0], (double)(cieX(lambda) * w));
    atomicAdd(&film[idx + 1], (double)(cieY(lambda) * w));
    atomicAdd(&film[idx + 2], (double)(cieZ(lambda) * w));
}
// Model A: map a contact-sensor hit to a pixel on the output film and deposit.
__device__ static void deposit(const DScene& sc, double* film, int resX, int resY,
                               const DVec3& p, Real lambda, Real beta) {
    DVec3 rel = p - sc.sensorOrigin;
    Real uu = dot(rel, sc.sensorUAxis) / dot(sc.sensorUAxis, sc.sensorUAxis);
    Real vv = dot(rel, sc.sensorVAxis) / dot(sc.sensorVAxis, sc.sensorVAxis);
    if (uu < 0 || uu >= 1 || vv < 0 || vv >= 1) return;
    int px = (int)(uu * resX), py = (int)(vv * resY);
    filmAdd(film, resX, px, py, lambda, beta);
}
__device__ static void connect(const DScene& sc, const DCamera& cam, double* film,
                               const DVec3& p, const DVec3& n, Real lambda, Real beta, Real rho) {
    DVec3 toCam = cam.eye - p;
    Real dist = length(toCam);
    DVec3 wdir = toCam / dist;
    Real cosSurf = dot(n, wdir);
    if (cosSurf <= 0) return;
    int px, py; Real cosCam, dist2;
    if (!cam.project(p, px, py, cosCam, dist2)) return;
    if (occluded(sc, p + n * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    Real f = rho / (Real)DPI;
    Real G = cosSurf * cosCam / dist2;
    Real We = (Real)1 / ((Real)cam.pixelPlaneArea() * cosCam * cosCam * cosCam * cosCam);
    Real contrib = beta * f * G * We;
    if (sc.medium.enabled) contrib *= exp(-medSigmaT(sc.medium, lambda) * dist);
    filmAdd(film, cam.resX, px, py, lambda, contrib);
}
__device__ static void connectVolume(const DScene& sc, const DCamera& cam, double* film,
                                      const DVec3& p, const DVec3& wIn, Real lambda, Real beta) {
    DVec3 toCam = cam.eye - p;
    Real dist = length(toCam);
    DVec3 wdir = toCam / dist;
    int px, py; Real cosCam, dist2;
    if (!cam.project(p, px, py, cosCam, dist2)) return;
    if (occluded(sc, p + wdir * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    Real ph = hgPhase(dot(wIn, wdir), (Real)sc.medium.g);
    Real Lambda = medAlbedo(sc.medium, lambda);
    Real G = cosCam / dist2;
    Real We = (Real)1 / ((Real)cam.pixelPlaneArea() * cosCam * cosCam * cosCam * cosCam);
    Real contrib = beta * Lambda * ph * G * We;
    contrib *= exp(-medSigmaT(sc.medium, lambda) * dist);
    filmAdd(film, cam.resX, px, py, lambda, contrib);
}

// ============================ megakernel ============================

__device__ static Real sampleLambda(const DScene& sc, const DEmitter& em, DRng& rng, Real& pdf) {
    // CDF search stays in double (host-baked table); the returned wavelength/pdf are Real.
    double u = (double)rng.uniform();
    const double* cdf = sc.lightCdfAll + em.cdfOffset;
    int lo = 0, hi = em.cdfN - 1;
    while (lo + 1 < hi) { int mid = (lo + hi) / 2; if (cdf[mid] <= u) lo = mid; else hi = mid; }
    double c0 = cdf[lo], c1 = cdf[lo + 1];
    double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
    pdf = (Real)((c1 - c0) / em.cdfStep);
    return (Real)(DLMIN + (lo + frac) * em.cdfStep);
}

// Power-weighted emitter selection (mirrors Scene::selectEmitter). Single
// emitter consumes no randomness, preserving the RNG stream for parity with CPU.
__device__ static int selectEmitter(const DScene& sc, double u) {
    int lo = 0, hi = sc.nEmitters - 1;
    while (lo < hi) { int mid = (lo + hi) / 2; if (sc.emitCdf[mid] < u) lo = mid + 1; else hi = mid; }
    return lo;
}

// --- image-environment device sampling / evaluation (mirror src/envmap.h) --------
// Jakob-Hanika sigmoid reflectance at lambda (mirrors upsample::reflAt).
__device__ static Real dReflAt(const double* c, Real lambda) {
    double t = ((double)lambda - 595.0) / 235.0;
    double p = c[0] * t * t + c[1] * t + c[2];
    return (Real)(0.5 + 0.5 * p / sqrt(1.0 + p * p));
}

// Continuous 1D CDF sample (mirrors Distribution1D::sampleContinuous). Returns the
// sample in [0,1); pdf is the density relative to funcInt, off the chosen bin.
__device__ static double dSample1D(const double* cdf, const double* func,
                                   double funcInt, int n, double u,
                                   double& pdf, int& off) {
    int lo = 0, hi = n;
    while (lo + 1 < hi) { int m = (lo + hi) / 2; if (cdf[m] <= u) lo = m; else hi = m; }
    off = lo;
    double du = u - cdf[lo];
    double d = cdf[lo + 1] - cdf[lo];
    if (d > 0) du /= d;
    pdf = (funcInt > 0) ? func[lo] / funcInt : 0.0;
    return (lo + du) / (double)n;
}

// Importance-sample an env emission direction from the luminance CDF (mirrors
// EnvMap::sample); fills dir and the solid-angle pdf pdfW.
__device__ static void dEnvSample(const DEnvMap& e, double u0, double u1,
                                  DVec3& dir, double& pdfW) {
    const double PI = 3.14159265358979323846;
    int vo = 0, uo = 0; double dv = 0, du = 0;
    double v = dSample1D(e.margCdf, e.margFunc, e.margFuncInt, e.h, u1, dv, vo);
    const double* cCdf  = e.condCdf  + (size_t)vo * (e.w + 1);
    const double* cFunc = e.condFunc + (size_t)vo * e.w;
    double uu = dSample1D(cCdf, cFunc, e.condFuncInt[vo], e.w, u0, du, uo);
    double mapPdf = du * dv;
    double theta = v * PI;
    double sinT = sin(theta);
    pdfW = (sinT > 0.0) ? mapPdf / (2.0 * PI * PI * sinT) : 0.0;
    double phi = (uu - 0.5 + e.rot) * 2.0 * PI;          // uvToDir
    dir = DVec3{(Real)(sinT * cos(phi)), (Real)cos(theta), (Real)(sinT * sin(phi))};
}

// Nearest texel index for a direction (mirrors EnvMap::texelOf / dirToUV).
__device__ static int dEnvTexel(const DEnvMap& e, const DVec3& d) {
    const double PI = 3.14159265358979323846;
    double y = fmin(fmax((double)d.y, -1.0), 1.0);
    double theta = acos(y);
    double phi = atan2((double)d.z, (double)d.x);
    double v = theta / PI;
    double u = phi / (2.0 * PI) + 0.5 - e.rot;
    u -= floor(u);
    int col = (int)(u * e.w); if (col < 0) col = 0; if (col >= e.w) col = e.w - 1;
    int row = (int)(v * e.h); if (row < 0) row = 0; if (row >= e.h) row = e.h - 1;
    return row * e.w + col;
}

__global__ void kTrace(DScene sc, DCamera cam, double* film, double* energy,
                       long long N, int diffraction, unsigned long long seedBase, int maxBounce,
                       int camMode) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    DRng rng; rng.seed((unsigned long long)(g * 2 + 1), seedBase ^ (unsigned long long)g);

    double eEmitted = 0, eAbsorbed = 0, eSensor = 0, eEscaped = 0, eResidual = 0;

    for (long long i = g; i < N; i += G) {
        // Power-weighted emitter selection (single emitter draws no randomness).
        int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
        const DEmitter em = sc.emitters[ei];
        Real u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 origin, emitN, dir;
        Real spotW = (Real)1;                            // spot direction reweight (else 1)
        bool envImage = false; double envPdfW = 0.0;     // image env: reweight below
        if (em.shape == 2) {
            // Point spot: uniform direction in the outer cone; reweight beta by
            // falloff*(Omega_outer/Omega_eff) to match the smoothstep profile.
            origin = em.origin;
            double ct = em.spotCosOuter + (double)u1 * (1.0 - em.spotCosOuter);
            double st = sqrt(fmax(0.0, 1.0 - ct * ct));
            double phi = 2.0 * 3.14159265358979323846 * (double)u2;
            DVec3 t, b; onb(em.beamDir, t, b);
            dir = t * (Real)(st * cos(phi)) + b * (Real)(st * sin(phi)) + em.beamDir * (Real)ct;
            emitN = em.beamDir;
            double omegaOuter = 2.0 * 3.14159265358979323846 * (1.0 - em.spotCosOuter);
            spotW = (Real)(spotFalloff(ct, em.spotCosInner, em.spotCosOuter) * omegaOuter / em.spotOmega);
        } else if (em.shape == 3) {
            // Infinite environment (mirrors CPU render.h). Sample the photon direction
            // — for a constant env uniformly on the sphere (pdf 1/4pi); for an image
            // env importance-sampled from the luminance CDF (pdf envPdfW) — then its
            // entry point on a disk of radius R perpendicular to `dir`, centered on the
            // scene and pushed upstream so it starts just outside the bounding sphere
            // (disk pdf 1/(pi R^2)). For the constant case the joint pdf 1/(4pi^2 R^2)
            // = 1/envGeom makes beta = emitIntegral*envGeom exactly analog; the image
            // case reweights beta below by L(dir,lambda)/(4pi*envPdfW*avgSpd(lambda)).
            if (sc.env.scale != nullptr) {
                dEnvSample(sc.env, (double)u1, (double)u2, dir, envPdfW);
                envImage = true;
            } else {
                double z = 1.0 - 2.0 * (double)u1;
                double sr = sqrt(fmax(0.0, 1.0 - z * z));
                double phi = 2.0 * 3.14159265358979323846 * (double)u2;
                dir = DVec3{(Real)(sr * cos(phi)), (Real)(sr * sin(phi)), (Real)z};
            }
            DVec3 t, b; onb(dir, t, b);
            double rd = sc.sceneRadius * sqrt((double)rng.uniform());
            double pd = 2.0 * 3.14159265358979323846 * (double)rng.uniform();
            DVec3 disk = t * (Real)(rd * cos(pd)) + b * (Real)(rd * sin(pd));
            origin = sc.sceneCenter - dir * (Real)sc.sceneRadius + disk;
            emitN = dir;
        } else {
            emitterSamplePoint(em, u1, u2, origin, emitN);   // quad: constant normal; sphere: surface point
            dir = em.collimated ? em.beamDir : cosineHemisphere(emitN, rng);
        }
        Real pdfL = 0;
        Real lambda = sampleLambda(sc, em, rng, pdfL);
        if (pdfL <= 0) continue;
        Real beta = (Real)((sc.nEmitters == 1) ? em.power : sc.totalPower);
        beta *= spotW;                                   // exactly 1 for non-spot
        // Image env: reweight so the photon carries the radiance actually arriving
        // from `dir`, = L(dir,lambda)/(4pi*envPdfW*avgSpd(lambda)). The shared
        // illuminant in L and avgSpd cancels, leaving the per-texel JH ratio.
        if (envImage) {
            int ti = dEnvTexel(sc.env, dir);
            double rad = sc.env.scale[ti] * (double)dReflAt(&sc.env.coeff[3 * ti], lambda);
            double avg = sc.env.avgScale * (double)dReflAt(sc.env.avgCoeff, lambda);
            double denom = 4.0 * 3.14159265358979323846 * envPdfW * avg;
            beta = (denom > 0.0) ? (Real)((double)beta * rad / denom) : (Real)0;
        }
        eEmitted += beta;

        // Model B: connect the emitter itself to the pinhole (makes the source
        // visible). Modes A/C instead catch photons that physically arrive. A spot
        // is a point light with no projected area, so it has no direct term.
        if (camMode == CAM_B && em.shape != 2 && em.shape != 3)
            connect(sc, cam, film, origin, emitN, lambda, beta, (Real)1);

        DVec3 ro = origin + dir * RAY_EPS, rd = dir;
        bool done = false;
        for (int bounce = 0; bounce < maxBounce && !done; ++bounce) {
            DHit h = closestHit(sc, ro, rd);
            Real dSurf = h.valid ? h.t : BIG;

            // fog free-flight; dEvent is the nearer of surface hit / volume collision.
            bool mediumEvent = false; DVec3 mp; Real dEvent = dSurf;
            if (sc.medium.enabled) {
                Real st = medSigmaT(sc.medium, lambda);
                if (st > 0) {
                    Real tMed = -log((Real)1 - rng.uniform()) / st;
                    if (tMed < dSurf) { mediumEvent = true; mp = ro + rd * tMed; dEvent = tMed; }
                }
            }

            // Model C perspective catch: if the photon flies through the aperture
            // nearer than the surface/fog event, it lands on the film. Analog physics.
            if (camMode == CAM_C) {
                int px, py;
                if (cam.catchPhoton(ro, rd, dEvent, px, py)) {
                    filmAdd(film, cam.resX, px, py, lambda, beta);
                    eSensor += beta; done = true; break;
                }
            }

            if (mediumEvent) {
                if (camMode == CAM_B) connectVolume(sc, cam, film, mp, rd, lambda, beta);
                if (rng.uniform() >= medAlbedo(sc.medium, lambda)) { eAbsorbed += beta; done = true; break; }
                DVec3 nd = sampleHG(rd, (Real)sc.medium.g, rng);
                ro = mp; rd = nd;
                continue;
            }

            if (!h.valid) { eEscaped += beta; done = true; break; }
            if (h.sensorId >= 0) {
                if (camMode == CAM_A) deposit(sc, film, cam.resX, cam.resY, h.p, lambda, beta);
                eSensor += beta; done = true; break;
            }

            const DMaterial* mptr = &sc.mats[h.matId];
            // Stochastic mix: resolve to a child lobe (or absorb) before dispatch.
            if (mptr->type == D_MIX) {
                Real u = rng.uniform(), acc = 0; int child = -1;
                for (int k = 0; k < mptr->mixCount; ++k) {
                    acc += (Real)mptr->mixWeight[k];
                    if (u < acc) { child = mptr->mixChild[k]; break; }
                }
                if (child < 0) { eAbsorbed += beta; done = true; break; }
                mptr = &sc.mats[child];
            }
            const DMaterial& m = *mptr;
            if (m.type == D_DIELECTRIC) {
                DVec3 nro, nrd; refractOrReflect(m, h, rd, lambda, rng, nro, nrd); ro = nro; rd = nrd; continue;
            } else if (m.type == D_THINFILM) {
                DVec3 nro, nrd;
                if (!thinFilmInterface(m, h, rd, lambda, rng, nro, nrd)) { eAbsorbed += beta; done = true; break; }
                ro = nro; rd = nrd; continue;
            } else if (m.type == D_MULTILAYER) {
                DVec3 nro, nrd;
                if (!multilayerInterface(m, h, rd, lambda, rng, nro, nrd)) { eAbsorbed += beta; done = true; break; }
                ro = nro; rd = nrd; continue;
            } else if (m.type == D_MIRROR) {
                Real r = clamp01(specLookup(m.reflect, lambda));
                if (rng.uniform() >= r) { eAbsorbed += beta; done = true; break; }
                DVec3 o = reflectv(rd, h.n); ro = h.p + h.n * RAY_EPS; rd = o; continue;
            } else if (m.type == D_GRATING) {
                Real r = clamp01(specLookup(m.reflect, lambda));
                if (rng.uniform() >= r) { eAbsorbed += beta; done = true; break; }
                DVec3 nro, nrd;
                if (!gratingDiffract(m, h, rd, lambda, diffraction, rng, nro, nrd)) { eAbsorbed += beta; done = true; break; }
                ro = nro; rd = nrd; continue;
            } else if (m.type == D_HALFMIRROR) {
                Real r = clamp01(specLookup(m.reflect, lambda));
                if (rng.uniform() < r) { DVec3 o = reflectv(rd, h.n); ro = h.p + h.n * RAY_EPS; rd = o; }
                else { ro = h.p + rd * RAY_EPS; }
                continue;
            } else if (m.type == D_GLOSSY) {
                Real r = clamp01(specLookup(m.reflect, lambda));
                if (rng.uniform() >= r) { eAbsorbed += beta; done = true; break; }
                DVec3 o = sampleGlossy(reflectv(rd, h.n), (Real)m.roughness, rng);
                if (dot(o, h.n) <= 0) { eAbsorbed += beta; done = true; break; }
                ro = h.p + h.n * RAY_EPS; rd = o; continue;
            } else {
                // Diffuse (Fluorescent scenes are rejected on the host; never reached).
                Real rho = clamp01(specLookup(m.reflect, lambda));
                if (camMode == CAM_B) connect(sc, cam, film, h.p, h.n, lambda, beta, rho);
                if (rng.uniform() >= rho) { eAbsorbed += beta; done = true; break; }
                ro = h.p + h.n * RAY_EPS; rd = cosineHemisphere(h.n, rng); continue;
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
    auto isFluoro = [&](int matId) {
        return matId >= 0 && matId < (int)scene.mats.size() &&
               scene.mats[matId].type == MatType::Fluorescent;
    };
    // A used material is unsupported if it is fluorescent, uses a spatially-varying
    // (textured) albedo — the GPU kernel bakes only a single reflect spectrum, so
    // textured scenes fall back to the CPU tracer — or is a mix that either has too
    // many child lobes for the GPU or references an unsupported child.
    auto textured = [&](int matId) {
        return matId >= 0 && matId < (int)scene.mats.size() &&
               scene.mats[matId].reflectTex >= 0;
    };
    // The device multilayer stack has a fixed cap (D_MAXLAYERS); scenes with a
    // deeper stack fall back to the CPU tracer, which has no layer limit.
    auto oversizedMultilayer = [&](int matId) {
        return matId >= 0 && matId < (int)scene.mats.size() &&
               scene.mats[matId].type == MatType::Multilayer &&
               (int)scene.mats[matId].layerN.size() > D_MAXLAYERS;
    };
    auto unsupported = [&](int matId) {
        if (isFluoro(matId) || textured(matId) || oversizedMultilayer(matId)) return true;
        if (matId >= 0 && matId < (int)scene.mats.size() &&
            scene.mats[matId].type == MatType::Mix) {
            const Material& mx = scene.mats[matId];
            if ((int)mx.mixChildren.size() > D_MIXMAX) return true;
            for (int c : mx.mixChildren) if (isFluoro(c) || textured(c)) return true;
        }
        return false;
    };
    for (const auto& t : scene.tris)    if (unsupported(t.matId)) return false;
    for (const auto& s : scene.spheres) if (unsupported(s.matId)) return false;
    // Environment lighting runs on-device: the kernel emits env photons from the scene
    // bounding sphere (shape==3) and the directly-viewed background is added by the
    // backend-agnostic addEnvBackground() pass. Both a constant env and an IMAGE-based
    // env (lat-long map: the 2D luminance CDF, per-texel JH coeff/scale, and mean
    // coeff/scale are uploaded, and the sampler/reweight are ported to the device) are
    // supported (increments 1b and 2c).
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

// Baked device scene + camera, plus the list of device allocations to free. Shared
// by the forward megakernel (renderForwardCuda) and the BDPT kernel (renderBdptCuda)
// so the ~150-line Scene->POD baking lives in exactly one place.
struct DUpload {
    gpu::DScene  sc{};
    gpu::DCamera dc{};
    std::vector<void*> frees;
};
static void freeUpload(DUpload& up) {
    for (void* p : up.frees) cudaFree(p);
    up.frees.clear();
}

// Bake the std::function Scene + Camera into POD device tables and upload them.
// Every cudaMalloc'd pointer is recorded in up.frees; call freeUpload(up) when done.
static void buildUpload(const Scene& scene, const Camera& cam, int res, DUpload& up) {
    using namespace gpu;
    auto keep = [&](void* p) { if (p) up.frees.push_back(p); return p; };

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
        bakeSpec(m.substrateK, d.substrateK);
        d.roughness = m.roughness;
        d.filmIor = m.filmIor; d.filmThickness = m.filmThickness;
        d.layerCount = (int)m.layerN.size();
        if (d.layerCount > D_MAXLAYERS) d.layerCount = D_MAXLAYERS;
        for (int k = 0; k < d.layerCount; ++k) {
            d.layerN[k] = m.layerN[k]; d.layerK[k] = m.layerK[k]; d.layerThick[k] = m.layerThick[k];
        }
        d.grooveSpacing = m.grooveSpacing;
        d.grooveDir = {m.grooveDir.x, m.grooveDir.y, m.grooveDir.z};
        d.gratingMaxOrder = m.gratingMaxOrder;
        d.mixCount = (int)m.mixChildren.size();
        if (d.mixCount > D_MIXMAX) d.mixCount = D_MIXMAX;
        for (int k = 0; k < d.mixCount; ++k) { d.mixChild[k] = m.mixChildren[k]; d.mixWeight[k] = m.mixWeights[k]; }
    }

    // --- upload geometry/materials ---
    DTri*      d_tris  = tris.empty()    ? nullptr : (DTri*)keep(uploadVec(tris));
    DSphere*   d_sph   = sph.empty()     ? nullptr : (DSphere*)keep(uploadVec(sph));
    DNode*     d_nodes = nodes.empty()   ? nullptr : (DNode*)keep(uploadVec(nodes));
    int*       d_prim  = primIdx.empty() ? nullptr : (int*)keep(uploadVec(primIdx));
    DMaterial* d_mats  = mats.empty()    ? nullptr : (DMaterial*)keep(uploadVec(mats));

    // Emitters: DEmitter array + flattened wavelength-CDF buffer + power selection CDF.
    std::vector<DEmitter> dems;
    std::vector<double> cdfAll;
    for (const auto& e : scene.emitters) {
        DEmitter de;
        de.origin  = {e.origin.x, e.origin.y, e.origin.z};
        de.u       = {e.u.x, e.u.y, e.u.z};
        de.v       = {e.v.x, e.v.y, e.v.z};
        de.normal  = {e.normal.x, e.normal.y, e.normal.z};
        de.beamDir = {e.beamDir.x, e.beamDir.y, e.beamDir.z};
        de.area = e.area; de.power = e.power;
        de.collimated = e.collimated ? 1 : 0;
        de.shape = (e.shape == EmitterShape::Sphere) ? 1
                 : (e.shape == EmitterShape::Spot)   ? 2
                 : (e.shape == EmitterShape::Env)    ? 3 : 0;
        de.radius = e.radius;
        de.spotCosInner = e.spotCosInner; de.spotCosOuter = e.spotCosOuter;
        de.spotOmega = e.spotOmega;
        de.cdfOffset = (int)cdfAll.size();
        de.cdfN = (int)e.spd.cdf.size();
        de.cdfStep = e.spd.step;
        de.matId = e.matId;                 // BDPT: link to emissive surface material
        bakeSpec(e.spdFn, de.emitSpd);       // BDPT: baked emission SPD for Le(lambda)
        cdfAll.insert(cdfAll.end(), e.spd.cdf.begin(), e.spd.cdf.end());
        dems.push_back(de);
    }
    std::vector<double> emitCdf = scene.emitterCdf;
    std::vector<double> emitSampCdf = scene.emitSampler.cdf;   // BDPT shared lambda CDF
    DEmitter* d_ems     = dems.empty()       ? nullptr : (DEmitter*)keep(uploadVec(dems));
    double*   d_cdfAll  = cdfAll.empty()     ? nullptr : (double*)keep(uploadVec(cdfAll));
    double*   d_emitCdf = emitCdf.empty()    ? nullptr : (double*)keep(uploadVec(emitCdf));
    double*   d_emitSamp = emitSampCdf.empty() ? nullptr : (double*)keep(uploadVec(emitSampCdf));

    // --- image environment tables (lat-long map) ---
    DEnvMap denv;
    if (scene.envMap) {
        const EnvMap& em = *scene.envMap;
        const int w = em.w, h = em.h; const size_t nT = (size_t)w * h;
        std::vector<double> coeffFlat(nT * 3);
        for (size_t i = 0; i < nT; ++i) {
            coeffFlat[3 * i + 0] = em.coeff[i][0];
            coeffFlat[3 * i + 1] = em.coeff[i][1];
            coeffFlat[3 * i + 2] = em.coeff[i][2];
        }
        std::vector<double> condCdf, condFunc, condFuncInt(h);
        condCdf.reserve((size_t)h * (w + 1));
        condFunc.reserve(nT);
        for (int v = 0; v < h; ++v) {
            const Distribution1D& c = em.dist.cond[v];
            condCdf.insert(condCdf.end(), c.cdf.begin(), c.cdf.end());
            condFunc.insert(condFunc.end(), c.func.begin(), c.func.end());
            condFuncInt[v] = c.funcInt;
        }
        denv.w = w; denv.h = h; denv.rot = em.rotOffset;
        denv.coeff = (double*)keep(uploadVec(coeffFlat));
        denv.scale = (double*)keep(uploadVec(em.scaleT));
        denv.avgCoeff[0] = em.avgCoeff[0];
        denv.avgCoeff[1] = em.avgCoeff[1];
        denv.avgCoeff[2] = em.avgCoeff[2];
        denv.avgScale = em.avgScale;
        denv.margCdf = (double*)keep(uploadVec(em.dist.marg.cdf));
        denv.margFunc = (double*)keep(uploadVec(em.dist.marg.func));
        denv.margFuncInt = em.dist.marg.funcInt;
        denv.condCdf = (double*)keep(uploadVec(condCdf));
        denv.condFunc = (double*)keep(uploadVec(condFunc));
        denv.condFuncInt = (double*)keep(uploadVec(condFuncInt));
    }

    DScene& sc = up.sc;
    sc.tris = d_tris; sc.nTris = (int)tris.size();
    sc.sph = d_sph;   sc.nSph = (int)sph.size();
    sc.mats = d_mats;
    sc.nodes = d_nodes; sc.primIdx = d_prim; sc.nNodes = (int)nodes.size();
    sc.emitters = d_ems; sc.nEmitters = (int)dems.size();
    sc.emitCdf = d_emitCdf; sc.totalPower = scene.totalPower;
    sc.lightCdfAll = d_cdfAll;
    sc.emitSamplerCdf = d_emitSamp;
    sc.emitSamplerN = (int)(emitSampCdf.empty() ? 0 : emitSampCdf.size() - 1);
    sc.emitSamplerStep = scene.emitSampler.step;
    sc.emitG = scene.emitG;
    sc.medium.enabled = scene.medium.enabled ? 1 : 0;
    sc.medium.g = scene.medium.g;
    bakeSpec(scene.medium.sigma_a, sc.medium.sigma_a);
    bakeSpec(scene.medium.sigma_s, sc.medium.sigma_s);
    sc.sensorOrigin = {scene.sensor.origin.x, scene.sensor.origin.y, scene.sensor.origin.z};
    sc.sensorUAxis  = {scene.sensor.uAxis.x,  scene.sensor.uAxis.y,  scene.sensor.uAxis.z};
    sc.sensorVAxis  = {scene.sensor.vAxis.x,  scene.sensor.vAxis.y,  scene.sensor.vAxis.z};
    sc.sceneCenter = {scene.sceneCenter.x, scene.sceneCenter.y, scene.sceneCenter.z};
    sc.sceneRadius = scene.sceneRadius;
    sc.env = denv;

    DCamera& dc = up.dc;
    dc.eye = {cam.eye.x, cam.eye.y, cam.eye.z};
    dc.u = {cam.u.x, cam.u.y, cam.u.z};
    dc.v = {cam.v.x, cam.v.y, cam.v.z};
    dc.w = {cam.w.x, cam.w.y, cam.w.z};
    dc.tanHalfX = cam.tanHalfX; dc.tanHalfY = cam.tanHalfY;
    dc.resX = res; dc.resY = res;
    dc.apertureR = cam.apertureR; dc.filmDist = cam.filmDist; dc.lensF = cam.lensF;
}

Film renderForwardCuda(const Scene& scene, const Camera& cam, int res,
                       long long N, EnergyReport& eOut, bool diffraction,
                       char camMode) {
    using namespace gpu;
    Film out; out.resX = res; out.resY = res; out.alloc();
    if (!cudaAvailable() || !cudaForwardSupported(scene)) return out;

    DUpload up;
    buildUpload(scene, cam, res, up);

    double* d_film = nullptr;   cudaMalloc(&d_film, (size_t)res * res * 3 * sizeof(double));
    cudaMemset(d_film, 0, (size_t)res * res * 3 * sizeof(double));
    double* d_energy = nullptr; cudaMalloc(&d_energy, 5 * sizeof(double));
    cudaMemset(d_energy, 0, 5 * sizeof(double));

    int camModeInt = (camMode == 'A') ? CAM_A : (camMode == 'C') ? CAM_C : CAM_B;

    int blockSize = 128;
    int numBlocks = 2048;          // ~262k threads, grid-stride over N photons
    kTrace<<<numBlocks, blockSize>>>(up.sc, up.dc, d_film, d_energy, N, diffraction ? 1 : 0,
                                     0x9e3779b97f4a7c15ULL, 32, camModeInt);
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

    freeUpload(up);
    cudaFree(d_film); cudaFree(d_energy);
    return out;
}
