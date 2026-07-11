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

// A spatially-varying reflectance texture (mirrors host Texture). `coeff` is the
// flattened per-texel Jakob-Hanika sigmoid coefficients (3 doubles per texel,
// row-major top-left origin), so a bound albedo becomes a physical reflectance at
// any wavelength via dReflAt — the exact device twin of Texture::reflectanceAt.
struct DTexture {
    int w, h;
    int wrap;    // TexWrap:   0 Repeat, 1 Clamp, 2 Mirror
    int filter;  // TexFilter: 0 Nearest, 1 Bilinear
    const double* coeff;   // 3*w*h Jakob-Hanika coefficients (albedo maps)
    const double* gray;    // w*h per-texel grayscale (mean linear RGB) for scalar maps
                           // (roughness/film-thickness, §9.4) — dTexScalarAt twin
};

struct DMaterial {
    int    type;
    double reflect[SPEC_N];     // baked reflect spectrum
    double ior[SPEC_N];         // baked index spectrum
    double substrateK[SPEC_N];  // baked thin-film substrate extinction kappa (0 = transparent)
    // Beer-Lambert interior absorption sigma_a(lambda) per metre travelled INSIDE a
    // dielectric (colored/attenuating glass). All-zero = colorless (default). Consulted
    // only for D_DIELECTRIC, via the `interior` material tracked through the transport
    // loop (device twin of Material::absorb).
    double absorb[SPEC_N];
    double roughness;
    double filmIor, filmThickness;
    // Spatially-varying diffuse albedo: index into DScene::textures (-1 = use the
    // constant `reflect` spectrum). When >=0 the diffuse/fluoro elastic reflectance
    // is sampled from the texture at the hit (u,v) instead of specLookup(reflect).
    int    reflectTex;
    // Triplanar (box) projection: > 0 => sample reflectTex by world-space triplanar
    // projection at this world-to-texture scale instead of the per-vertex (u,v).
    // Device twin of Material::triplanarScale (dTexReflTriplanar). 0 => use (u,v).
    double triplanarScale;
    // Spatially-varying NON-albedo scalar params (spec §9.4), device twins of
    // Material::roughnessTex / filmThicknessTex. >=0 => sample the texture's grayscale
    // value (dTexScalarAt) at the hit (u,v); -1 => use the constant roughness /
    // filmThickness. Honoured by the forward paths (megakernel + wavefront); the GPU
    // BDPT kernel rejects such scenes (cudaBdptSupported) so they fall back to CPU.
    int    roughnessTex;
    int    filmThicknessTex;
    // Fluorescence (D_FLUORESCENT): fluoAbsorb is the baked excitation probability
    // epsilon(lambda); the dye re-radiates (quantum yield fluoYield) at a Stokes-
    // shifted lambda' drawn from the emission-SPD CDF slice [fluoCdfOffset,
    // fluoCdfOffset+fluoCdfN) inside DScene::fluoCdfAll (fluoCdfStep = bin width nm).
    double fluoAbsorb[SPEC_N];
    double fluoYield;
    int    fluoCdfOffset, fluoCdfN;
    double fluoCdfStep;
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
    int    mixWeightTex;   // >=0: per-hit blend mask (2-child mix); -1: constant weights
    // Procedural (math-driven) scalar drives (§4): index into DScene::patterns, or -1.
    // roughnessPat / filmThicknessPat override the constant/texture value at the hit;
    // mixWeightPat drives child-0 selection of a 2-child D_MIX. Device twins of
    // Material::roughnessPat / filmThicknessPat / mixWeightPat.
    int    roughnessPat;
    int    filmThicknessPat;
    int    mixWeightPat;
};

struct DTri    { DVec3 v0, v1, v2, gn; DVec3 uv0, uv1, uv2; int matId, sensorId; };
struct DSphere { DVec3 c; double r; int matId; };
struct DNode   { DVec3 lo, hi; int left, right, first, count; };

// Implicit surfaces (isosurface / CSG / metaballs) — device twins of implicit.h.
// The field is a flat postfix array evaluated with a scalar stack, sphere-traced for
// intersection. All math is done in DOUBLE (independent of the FP32 transport `Real`):
// the sign-change bisection converges to ~1e-12, which needs the precision, and
// implicits are already the expensive path, so the FP64 cost is acceptable. POD twins
// of FieldOp / FieldNode / Implicit (see src/implicit.h).
// NOTE: this order MUST match FieldOp in implicit.h (dn.op = (int)fn.op on upload).
// DF_EXPR is the arbitrary-formula isosurface leaf: its value is f(x,y,z) evaluated
// by the pattern VM, NOT a signed distance, so the enclosing DImplicit carries a
// container AABB + Lipschitz bound (see intersectImplicit / cudaForwardSupported).
enum { DF_SPHERE = 0, DF_BOX, DF_TORUS, DF_PLANE, DF_CYLINDER, DF_CONE, DF_EXPR,
       DF_UNION, DF_INTERSECT, DF_DIFFERENCE,
       DF_SMOOTH_UNION, DF_SMOOTH_INTERSECT, DF_SMOOTH_DIFFERENCE };
struct DFieldNode {
    int    op;
    double p[4];
    double inv[9];        // world->local linear part (row-major, matches Affine::m)
    double tx, ty, tz;    // world->local translation (Affine::t)
    double scale;         // world = scale * local; d_world = d_local * scale (leaf only)
    int    exprOff, exprN;// DF_EXPR: slice into DScene::fieldExprNodes (postfix PatNode program)
};
struct DImplicit {
    int    nodeOff, nodeN;   // slice [nodeOff, nodeOff+nodeN) into DScene::fieldNodes
    int    matId;
    double lo[3], hi[3];     // world AABB (ray clip)
    double lipschitz, minStep;
    int    method;           // 0 = adaptive (|f|/lipschitz), 1 = fixed-step sample
    int    refine;           // 0 = bisect, 1 = regula-falsi (Illinois)
    double sampleStep;       // fixed world march step for method==1
};

// Procedural pattern (math-driven scalar field, §4) — device twin of pattern.h.
// One flat postfix PatNode pool (DScene::patNodes) holds every pattern back-to-back;
// each DPattern slices it by [off, off+n). A material's roughnessPat/filmThicknessPat/
// mixWeightPat index a DPattern (or -1). The postfix VM (dPatternEval) runs the same
// opcode/hash-noise math as the host so CPU and GPU agree. PatNode/PatOp come from
// pattern.h (POD, uploaded verbatim) — no device-specific node type is needed.
struct DPattern { int off, n; };   // slice into DScene::patNodes

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
    int    shape;              // 0 quad, 1 sphere, 2 spot, 3 env, 4 cylinder (EmitterShape)
    double radius;             // sphere radius (shape==1) / tube radius (shape==4)
    int    caps;               // cylinder (shape==4): also emit from the two end discs
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
    } else if (em.shape == 4) {
        // Cylinder (tube) lateral surface: u1 along the axis (v), u2 around it. u and
        // normal are the precomputed radial basis (mirrors host Emitter::samplePoint).
        double phi = 2.0 * 3.14159265358979323846 * u2;
        DVec3 rad = em.u * (Real)cos(phi) + em.normal * (Real)sin(phi);
        if (em.caps) {
            // Closed capsule: pick lateral wall or one end disc proportional to area,
            // then reuse u1 (remapped) within the region (mirrors host samplePoint).
            double len = length(em.v);
            DVec3 a = (len > 0.0) ? em.v / (Real)len : DVec3{(Real)0,(Real)1,(Real)0};
            double latA = 2.0 * 3.14159265358979323846 * em.radius * len;
            double capA = 3.14159265358979323846 * em.radius * em.radius;
            double total = latA + 2.0 * capA;
            double pLat = latA / total, pCap = capA / total;
            if (u1 < pLat) {
                double uu = u1 / pLat;
                y = em.origin + em.v * (Real)uu + rad * (Real)em.radius;
                nOut = rad;
            } else if (u1 < pLat + pCap) {
                double rr = em.radius * sqrt((u1 - pLat) / pCap);
                y = em.origin + rad * (Real)rr;
                nOut = a * (Real)(-1.0);
            } else {
                double rr = em.radius * sqrt((u1 - pLat - pCap) / pCap);
                y = em.origin + em.v + rad * (Real)rr;
                nOut = a;
            }
        } else {
            y = em.origin + em.v * (Real)u1 + rad * (Real)em.radius;
            nOut = rad;
        }
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
    // Implicit surfaces (isosurface/CSG/metaballs). BVH prims with index
    // >= nTris+nSph map to implicits[prim - nTris - nSph]; fieldNodes is the flat
    // postfix node pool the DImplicit slices index into.
    const DFieldNode* fieldNodes;
    // Flat postfix PatNode pool for DF_EXPR leaves (arbitrary-formula isosurfaces);
    // each Expr FieldNode slices it by [exprOff, exprOff+exprN). Separate from
    // patNodes so material patterns and field formulas don't share offsets.
    const PatNode*    fieldExprNodes;
    const DImplicit*  implicits; int nImplicits;
    // Procedural patterns (§4): flat postfix PatNode pool + per-pattern slices.
    // A material's roughnessPat/filmThicknessPat/mixWeightPat index `patterns`.
    const PatNode*   patNodes;
    const DPattern*  patterns; int nPatterns;
    const DEmitter*  emitters; int nEmitters;
    const double*    emitCdf;       // size nEmitters, cumulative power, normalised
    double           totalPower;
    const double*    lightCdfAll;   // flattened per-emitter wavelength CDFs
    const DTexture*  textures; int nTex;   // reflectance textures (mat.reflectTex)
    const double*    fluoCdfAll;    // flattened per-material fluorescence emission CDFs
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

// Lens-projection radius maps (device twins of camera.h projRadius/Inv/Deriv). The
// projection tag matches CameraProjection (camera.h): 0 rectilinear, 1 equidistant,
// 2 equisolid, 3 stereographic, 4 orthographic. Kept as free HD helpers so DCamera's
// project()/pixelSolidAngle() can share them.
HD static inline double dProjRadius(int proj, double th) {
    switch (proj) {
        case CAM_EQUIDISTANT:   return th;
        case CAM_EQUISOLID:     return 2.0 * sin(0.5 * th);
        case CAM_STEREOGRAPHIC: return 2.0 * tan(0.5 * th);
        case CAM_ORTHOGRAPHIC:  return sin(th);
        default:                return tan(th);              // CAM_RECTILINEAR
    }
}
// (dProjRadiusInv — the r->theta inverse used by Camera::genRay — is intentionally
// omitted: the GPU forward path only SPLATS to the camera (project), it never
// generates camera rays, so the inverse map has no device caller.)
HD static inline double dProjRadiusDeriv(int proj, double th) {
    switch (proj) {
        case CAM_EQUIDISTANT:   return 1.0;
        case CAM_EQUISOLID:     return cos(0.5 * th);
        case CAM_STEREOGRAPHIC: { double c = cos(0.5 * th); return 1.0 / (c * c); }
        case CAM_ORTHOGRAPHIC:  return cos(th);
        default:              { double c = cos(th);        return 1.0 / (c * c); }  // sec^2
    }
}
// r->theta inverse (device twin of camera.h projRadiusInv). Needed by the backward
// tracer's dGenRay: unlike the forward path (which only SPLATS to the camera), mode R
// GENERATES camera rays, so it must invert the projection map to place a film sample.
HD static inline double dProjRadiusInv(int proj, double r) {
    double x;
    switch (proj) {
        case CAM_EQUIDISTANT:   return r;
        case CAM_EQUISOLID:     x = 0.5 * r; x = x < -1 ? -1 : (x > 1 ? 1 : x); return 2.0 * asin(x);
        case CAM_STEREOGRAPHIC: return 2.0 * atan(0.5 * r);
        case CAM_ORTHOGRAPHIC:  x = r; x = x < -1 ? -1 : (x > 1 ? 1 : x); return asin(x);
        default:                return atan(r);              // CAM_RECTILINEAR
    }
}

// Maximum refracting interfaces in a physical (mesh-lens) camera on the GPU. Lenses
// with more surfaces fall back to the CPU backward tracer (cudaBackwardSupported).
#define D_MAXLENS 16

// One refracting interface of the physical lens (device twin of LensSurface). The
// per-surface sensor-side index is baked into DLensSystem::iorAll (SPEC_N entries per
// surface) so the std::function Spectrum never crosses the device barrier — the same
// bake-to-table trick DMaterial uses. `zpos` is the cached vertex z (mm).
struct DLensSurface {
    double radius;      // signed radius of curvature (mm); 0 => planar
    double thickness;   // axial gap to the next surface toward the sensor (mm)
    double aperture;    // clear semi-diameter / stop radius (mm)
    double zpos;        // cached vertex z (mm), sensor nominally at 0
    int    isStop;
};
// Physical multi-element lens (device twin of LensSystem). Embedded by value in
// DCamera; `iorAll` points at nSurf*SPEC_N baked sensor-side index tables uploaded by
// buildUpload. Surfaces are stored front (scene, idx 0) -> rear (sensor, idx nSurf-1).
struct DLensSystem {
    int    nSurf;
    double filmW_mm, filmH_mm;   // sensor size (mm)
    double T, filmZ;             // total track (front vertex z) and sensor plane z (mm)
    DLensSurface  surf[D_MAXLENS];
    const double* iorAll;        // nSurf*SPEC_N sensor-side index tables (air baked as 1)
};

struct DCamera {
    DVec3  eye, u, v, w;
    double tanHalfX, tanHalfY;
    int    resX, resY;
    double apertureR, filmDist, lensF;   // models A/C finite aperture + thin lens
    // Lens projection (mirrors Camera): rectilinear (default) keeps the pinhole math
    // byte-for-byte; fisheye/panoramic modes remap the ray angle in project()/
    // pixelSolidAngle(). halfFovY is the vertical half-field (rad) and rEdge the
    // image radius at the vertical film edge (= dProjRadius(projection, halfFovY)).
    int    projection;
    double halfFovY, rEdge;
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
        if (projection == CAM_RECTILINEAR) {
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
        // Fisheye/panoramic: map the direction's angle-from-axis theta to a normalised
        // image radius rho, then place it along the (u,v) azimuth. A wide lens sees
        // theta > 90 deg (cz <= 0), so do NOT reject on cz (mirrors Camera::project).
        Real len = length(d);
        if (len < (Real)1e-12) return false;
        Real costh = cz / len;
        costh = costh < (Real)-1 ? (Real)-1 : (costh > (Real)1 ? (Real)1 : costh);
        Real th = acos(costh);
        Real rho = (Real)(dProjRadius(projection, (double)th) / rEdge);
        Real ru = dot(d, u), rv = dot(d, v);
        Real rhoDir = sqrt(ru * ru + rv * rv);
        Real ix, iy;
        if (rhoDir < (Real)1e-12) { ix = 0; iy = 0; }
        else { ix = rho * ru / rhoDir; iy = rho * rv / rhoDir; }
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
        px = (int)((ix * (Real)0.5 + (Real)0.5) * resX);
        py = (int)((iy * (Real)0.5 + (Real)0.5) * resY);
        dist2 = len * len;
        cosCam = costh;
        return true;
    }
    // Solid angle subtended by one pixel for a connection at cosine cosCam from the
    // axis: the projection-general splat normaliser (port of Camera::pixelSolidAngle).
    // For rectilinear this is pixelPlaneArea()*cosCam^3, recovering the classic
    // 1/(A_pix cos^4) importance once the geometry's cosCam is folded in.
    HD double pixelSolidAngle(Real cosCam) const {
        if (projection == CAM_RECTILINEAR)
            return pixelPlaneArea() * (double)cosCam * (double)cosCam * (double)cosCam;
        double c = cosCam < -1 ? -1 : (cosCam > 1 ? 1 : (double)cosCam);
        double th = acos(c);
        double dr = dProjRadiusDeriv(projection, th);
        double r  = dProjRadius(projection, th);
        double denom = dr * r;
        if (denom < 1e-12) denom = 1e-12;
        double aNorm = 4.0 / ((double)resX * (double)resY);   // pixel area in [-1,1]^2 view
        return aNorm * sin(th) * rEdge * rEdge / denom;
    }
    // Image a pupil point A along direction dir onto a film cell (port of
    // Camera::lensImage). With a thin lens the direction is refracted by the paraxial
    // ray transfer u' = u - rho/f. Shared by the model-C brute-force catch and the
    // model-A next-event splat.
    HD bool lensImage(const DVec3& A, const DVec3& dir, int& px, int& py) const {
        DVec3 nAxis = w * (Real)-1;
        DVec3 rho = A - eye;
        DVec3 d = dir;
        if (lensF > 0.0) {
            Real dax = dot(d, nAxis);
            if (dax <= (Real)1e-9) return false;
            DVec3 slope = (d - nAxis * dax) / dax;
            DVec3 slopeP = slope - rho * (Real)(1.0 / lensF);
            d = normalize(nAxis + slopeP);
        }
        Real ddax = dot(d, nAxis);
        if (ddax <= (Real)1e-9) return false;
        Real s = (Real)filmDist / ddax;
        DVec3 Fcenter = eye + nAxis * (Real)filmDist;
        DVec3 Q = A + d * s;
        DVec3 rel = Q - Fcenter;
        Real ix = -dot(rel, u) / (Real)(filmDist * tanHalfX);
        Real iy = -dot(rel, v) / (Real)(filmDist * tanHalfY);
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
        px = (int)((ix * (Real)0.5 + (Real)0.5) * resX);
        py = (int)((iy * (Real)0.5 + (Real)0.5) * resY);
        return true;
    }
    // Model C perspective catch: does this photon fly through the finite aperture
    // disc (before hitting the scene, within hitDist) and land on the film? Port of
    // Camera::catchPhoton.
    HD bool catchPhoton(const DVec3& ro, const DVec3& rd, Real hitDist, int& px, int& py) const {
        Real dw = dot(rd, w);
        if (dw >= (Real)-1e-9) return false;
        Real tAp = dot(eye - ro, w) / dw;
        if (tAp <= RAY_EPS || tAp >= hitDist) return false;
        DVec3 P = ro + rd * tAp;
        DVec3 rho = P - eye;
        if (dot(rho, rho) > (Real)(apertureR * apertureR)) return false;
        return lensImage(P, rd, px, py);
    }
    // Physical multi-element (mesh-lens) camera. When hasLens is set the backward
    // tracer (mode R) generates rays by refracting them from the film out through the
    // real glass interfaces (dGenLensRay), superseding the pinhole/thin-lens model.
    int         hasLens;
    DLensSystem lens;
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
    Real u, v;   // interpolated surface texture coordinates
};

// ---- implicit field evaluation (device twin of implicit.h) ----------------
// smin/smax (Inigo Quilez quadratic blend) — filleted CSG / metaball merge.
__device__ static inline double dSmin(double a, double b, double k) {
    if (k <= 0.0) return a < b ? a : b;
    double h = fmax(k - fabs(a - b), 0.0) / k;
    return (a < b ? a : b) - h * h * k * 0.25;
}
__device__ static inline double dSmax(double a, double b, double k) { return -dSmin(-a, -b, k); }

// Leaf SDF at the leaf-LOCAL query point (px,py,pz). Mirrors fieldLeafSDF exactly.
// Forward decl: DF_EXPR leaves evaluate their formula with the pattern VM, which is
// defined further down (dPatternEval). The field VM only needs it for the Expr case.
__device__ static double dPatternEval(const PatNode* nodes, int n,
                                      double x, double y, double z, double f,
                                      double nx, double ny, double nz, double r);

__device__ static double dFieldLeafSDF(const DFieldNode& nd, double px, double py, double pz,
                                       const PatNode* exprPool) {
    switch (nd.op) {
        case DF_SPHERE:
            return sqrt(px*px + py*py + pz*pz) - nd.p[0];
        case DF_EXPR: {   // arbitrary formula f(x,y,z); r=|p|, other vars (f/normals) are 0
            if (!exprPool) return BIG;
            double r = sqrt(px*px + py*py + pz*pz);
            return dPatternEval(exprPool + nd.exprOff, nd.exprN, px, py, pz, 0.0,
                                0.0, 0.0, 0.0, r);
        }
        case DF_BOX: {
            double r = nd.p[3];
            double qx = fabs(px) - nd.p[0] + r, qy = fabs(py) - nd.p[1] + r, qz = fabs(pz) - nd.p[2] + r;
            double ox = fmax(qx, 0.0), oy = fmax(qy, 0.0), oz = fmax(qz, 0.0);
            double outside = sqrt(ox*ox + oy*oy + oz*oz);
            double inside  = fmin(fmax(qx, fmax(qy, qz)), 0.0);
            return outside + inside - r;
        }
        case DF_TORUS: {
            double qx = sqrt(px*px + pz*pz) - nd.p[0];
            return sqrt(qx*qx + py*py) - nd.p[1];
        }
        case DF_PLANE:
            return px*nd.p[0] + py*nd.p[1] + pz*nd.p[2] + nd.p[3];
        case DF_CYLINDER: {
            double dxz = sqrt(px*px + pz*pz) - nd.p[0];
            double dy  = fabs(py) - nd.p[1];
            double a   = fmin(fmax(dxz, dy), 0.0);
            double bx  = fmax(dxz, 0.0), by = fmax(dy, 0.0);
            return a + sqrt(bx*bx + by*by);
        }
        case DF_CONE: {
            double rb = nd.p[0], rt = nd.p[1], h = nd.p[2];
            double qx = sqrt(px*px + pz*pz), qy = py;
            double k1x = rt, k1y = h, k2x = rt - rb, k2y = 2.0*h;
            double cax = qx - fmin(qx, (qy < 0.0) ? rb : rt);
            double cay = fabs(qy) - h;
            double k2dot = k2x*k2x + k2y*k2y;
            double tt = (k2dot > 0.0) ? ((k1x - qx)*k2x + (k1y - qy)*k2y) / k2dot : 0.0;
            tt = tt < 0.0 ? 0.0 : (tt > 1.0 ? 1.0 : tt);
            double cbx = qx - k1x + k2x*tt, cby = qy - k1y + k2y*tt;
            double s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
            double da = cax*cax + cay*cay, db = cbx*cbx + cby*cby;
            return s * sqrt(fmin(da, db));
        }
        default: return BIG;
    }
}
// Whole-field SDF at world point (pw) via the postfix scalar stack. Mirrors fieldEval.
__device__ static double dFieldEval(const DFieldNode* nodes, int n,
                                    double pwx, double pwy, double pwz,
                                    const PatNode* exprPool) {
    double st[64]; int sp = 0;
    for (int i = 0; i < n; ++i) {
        const DFieldNode& nd = nodes[i];
        switch (nd.op) {
            case DF_UNION:            { double b = st[--sp], a = st[--sp]; st[sp++] = a < b ? a : b; break; }
            case DF_INTERSECT:        { double b = st[--sp], a = st[--sp]; st[sp++] = a > b ? a : b; break; }
            case DF_DIFFERENCE:       { double b = st[--sp], a = st[--sp]; st[sp++] = a > -b ? a : -b; break; }
            case DF_SMOOTH_UNION:     { double b = st[--sp], a = st[--sp]; st[sp++] = dSmin(a,  b, nd.p[0]); break; }
            case DF_SMOOTH_INTERSECT: { double b = st[--sp], a = st[--sp]; st[sp++] = dSmax(a,  b, nd.p[0]); break; }
            case DF_SMOOTH_DIFFERENCE:{ double b = st[--sp], a = st[--sp]; st[sp++] = dSmax(a, -b, nd.p[0]); break; }
            default: {   // leaf: world -> local via inv, then local SDF * scale
                double plx = nd.inv[0]*pwx + nd.inv[1]*pwy + nd.inv[2]*pwz + nd.tx;
                double ply = nd.inv[3]*pwx + nd.inv[4]*pwy + nd.inv[5]*pwz + nd.ty;
                double plz = nd.inv[6]*pwx + nd.inv[7]*pwy + nd.inv[8]*pwz + nd.tz;
                st[sp++] = dFieldLeafSDF(nd, plx, ply, plz, exprPool) * nd.scale;
            }
        }
    }
    return sp > 0 ? st[0] : BIG;
}
// Field gradient (tetrahedron central differences) -> unit normal. Mirrors fieldGradient.
__device__ static void dFieldGradient(const DFieldNode* nodes, int n,
                                      double px, double py, double pz, double eps,
                                      double& gx, double& gy, double& gz,
                                      const PatNode* exprPool) {
    // stencil offsets k1(1,-1,-1) k2(-1,-1,1) k3(-1,1,-1) k4(1,1,1)
    double f1 = dFieldEval(nodes, n, px + eps, py - eps, pz - eps, exprPool);
    double f2 = dFieldEval(nodes, n, px - eps, py - eps, pz + eps, exprPool);
    double f3 = dFieldEval(nodes, n, px - eps, py + eps, pz - eps, exprPool);
    double f4 = dFieldEval(nodes, n, px + eps, py + eps, pz + eps, exprPool);
    gx =  f1 - f2 - f3 + f4;
    gy = -f1 - f2 + f3 + f4;
    gz = -f1 + f2 - f3 + f4;
    double len = sqrt(gx*gx + gy*gy + gz*gz);
    if (len > 0.0) { gx /= len; gy /= len; gz /= len; }
    else           { gx = 0.0; gy = 0.0; gz = 1.0; }
}
// Sphere-trace one implicit; writes into `hit` (respecting hit.t). Mirrors intersectImplicit.
__device__ static bool intersectImplicit(const DScene& sc, const DImplicit& im,
                                          const DVec3& roR, const DVec3& rdR, Real tmin, DHit& hit) {
    double ox = roR.x, oy = roR.y, oz = roR.z, dx = rdR.x, dy = rdR.y, dz = rdR.z;
    double idx = 1.0/dx, idy = 1.0/dy, idz = 1.0/dz;
    double t0 = tmin, t1 = hit.t;
    // clip to world AABB
    { double ta = (im.lo[0]-ox)*idx, tb = (im.hi[0]-ox)*idx; if (ta>tb){double s=ta;ta=tb;tb=s;} if(ta>t0)t0=ta; if(tb<t1)t1=tb; if(t1<t0)return false; }
    { double ta = (im.lo[1]-oy)*idy, tb = (im.hi[1]-oy)*idy; if (ta>tb){double s=ta;ta=tb;tb=s;} if(ta>t0)t0=ta; if(tb<t1)t1=tb; if(t1<t0)return false; }
    { double ta = (im.lo[2]-oz)*idz, tb = (im.hi[2]-oz)*idz; if (ta>tb){double s=ta;ta=tb;tb=s;} if(ta>t0)t0=ta; if(tb<t1)t1=tb; if(t1<t0)return false; }

    const DFieldNode* nd = sc.fieldNodes + im.nodeOff;
    const PatNode* exprPool = sc.fieldExprNodes;
    const int N = im.nodeN;
    const double dlen = sqrt(dx*dx + dy*dy + dz*dz);
    const int MAX_STEP = 2048;
    const double invLip = 1.0 / (im.lipschitz > 0.0 ? im.lipschitz : 1.0);
    const double minStep = im.minStep > 0.0 ? im.minStep : 1e-4;

    const bool   sampleMode  = (im.method == 1);
    const double fixedStep   = (im.sampleStep > 0.0 ? im.sampleStep : minStep) / dlen;
    const bool   regulaFalsi = (im.refine == 1);

    double t = t0;
    double f = dFieldEval(nd, N, ox + dx*t, oy + dy*t, oz + dz*t, exprPool);
    for (int i = 0; i < MAX_STEP; ++i) {
        double step = sampleMode ? fixedStep : fmax(fabs(f) * invLip, minStep) / dlen;
        double tn = t + step;
        bool last = false;
        if (tn >= t1) { tn = t1; last = true; }
        double fn = dFieldEval(nd, N, ox + dx*tn, oy + dy*tn, oz + dz*tn, exprPool);
        bool crossed = (f > 0.0 && fn <= 0.0) || (f < 0.0 && fn >= 0.0) || (f == 0.0 && fn != 0.0);
        if (crossed) {
            double ta = t, tb = tn, fa = f, fb = fn;
            int rfSide = 0;
            for (int b = 0; b < 80; ++b) {
                double tm;
                if (regulaFalsi && (fb - fa) != 0.0) {
                    tm = (ta * fb - tb * fa) / (fb - fa);
                    if (tm <= ta || tm >= tb) tm = 0.5*(ta + tb);
                } else {
                    tm = 0.5*(ta + tb);
                }
                double fm = dFieldEval(nd, N, ox + dx*tm, oy + dy*tm, oz + dz*tm, exprPool);
                if ((fa > 0.0) == (fm > 0.0)) {
                    ta = tm; fa = fm;
                    if (regulaFalsi && rfSide == +1) fb *= 0.5;
                    rfSide = +1;
                } else {
                    tb = tm; fb = fm;
                    if (regulaFalsi && rfSide == -1) fa *= 0.5;
                    rfSide = -1;
                }
                if ((tb - ta) * dlen < 1e-12) break;
            }
            double th = 0.5*(ta + tb);
            if (th < tmin || th >= (double)hit.t) return false;
            double px = ox + dx*th, py = oy + dy*th, pz = oz + dz*th;
            double eps = fmax(1e-6, 1e-4*th);
            double gx, gy, gz; dFieldGradient(nd, N, px, py, pz, eps, gx, gy, gz, exprPool);
            hit.t = (Real)th; hit.p = DVec3(px, py, pz); hit.valid = true;
            hit.ng = DVec3(gx, gy, gz);
            double side = dx*gx + dy*gy + dz*gz;
            hit.n = (side < 0.0) ? DVec3(gx, gy, gz) : DVec3(-gx, -gy, -gz);
            hit.matId = im.matId; hit.sensorId = -1;
            hit.u = 0; hit.v = 0;
            return true;
        }
        if (last) return false;
        t = tn; f = fn;
    }
    return false;
}

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
    // Barycentric-interpolate the per-vertex UVs (u,vv are the Moller-Trumbore
    // weights of v1,v2; the v0 weight is 1-u-vv). Mirrors host intersectTri.
    Real w0 = (Real)1 - u - vv;
    hit.u = w0 * tri.uv0.x + u * tri.uv1.x + vv * tri.uv2.x;
    hit.v = w0 * tri.uv0.y + u * tri.uv1.y + vv * tri.uv2.y;
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
    // Equirectangular (lat/long) UV so spheres can be textured (mirrors host).
    Real ny = ng.y < (Real)-1 ? (Real)-1 : (ng.y > (Real)1 ? (Real)1 : ng.y);
    hit.u = (Real)0.5 + atan2(ng.z, ng.x) / (Real)(2.0 * DPI);
    hit.v = (Real)0.5 - asin(ny) / (Real)DPI;
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
                if (prim < sc.nTris)              { if (intersectTri(ro, rd, sc.tris[prim], tmin, h)) tMax = h.t; }
                else if (prim < sc.nTris + sc.nSph){ if (intersectSphere(ro, rd, sc.sph[prim - sc.nTris], tmin, h)) tMax = h.t; }
                else                              { if (intersectImplicit(sc, sc.implicits[prim - sc.nTris - sc.nSph], ro, rd, tmin, h)) tMax = h.t; }
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
                             : (prim < sc.nTris + sc.nSph) ? intersectSphere(o, dir, sc.sph[prim - sc.nTris], tmin, h)
                             : intersectImplicit(sc, sc.implicits[prim - sc.nTris - sc.nSph], o, dir, tmin, h);
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

// Per-hit roughness helper (defined below, after the texture/pattern samplers) — used
// here for frosted glass before its point of definition.
__device__ static Real dMatRoughness(const DScene& sc, const DMaterial& m, const DHit& h);

// Dielectric interface: Fresnel-weighted specular reflect-or-refract (Snell, spectral
// index -> dispersion). A non-zero per-hit roughness frosts BOTH lobes (rough glass):
// the chosen direction is jittered by a power-cosine lobe, rejecting jitters that would
// cross to the wrong side so no light leaks through. `transmitted` (optional) reports
// whether the ray refracted vs. reflected/TIR — the caller uses it to track which
// medium it is now inside (interior absorption). Mirrors host refractOrReflect.
__device__ static void refractOrReflect(const DScene& sc, const DMaterial& m, const DHit& h,
                                         const DVec3& d, Real lambda, DRng& rng,
                                         DVec3& ro, DVec3& rd, bool* transmitted = nullptr) {
    Real ng = specLookup(m.ior, lambda);
    bool entering = dot(d, h.ng) < 0;
    DVec3 nl = entering ? h.ng : -h.ng;
    Real n1 = entering ? (Real)1 : ng, n2 = entering ? ng : (Real)1;
    Real eta = n1 / n2;
    Real cosI = -dot(d, nl);
    Real sin2t = eta * eta * ((Real)1 - cosI * cosI);
    DVec3 outDir;
    bool refracted = false;
    if (sin2t > 1) outDir = reflectv(d, nl);
    else {
        Real cosT = sqrt((Real)1 - sin2t);
        Real rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
        Real rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
        Real R = (Real)0.5 * (rs * rs + rp * rp);
        if (rng.uniform() < R) outDir = reflectv(d, nl);
        else { outDir = d * eta + nl * (eta * cosI - cosT); refracted = true; }
    }
    outDir = normalize(outDir);
    // Frosted glass: jitter the chosen lobe, keeping it on the intended side.
    Real rough = dMatRoughness(sc, m, h);
    if (rough > (Real)1e-3) {
        DVec3 pert = sampleGlossy(outDir, rough, rng);
        bool ok = refracted ? (dot(pert, nl) < 0) : (dot(pert, nl) > 0);
        if (ok) outDir = pert;
    }
    if (transmitted) *transmitted = refracted;
    ro = h.p + outDir * RAY_EPS; rd = outDir;
}
// Returns false if the photon is absorbed by an opaque (absorbing) substrate.
// Forward decl: the per-hit thin-film thickness helper (definition with the other
// texture samplers, below dTexScalarAt) is used here before its point of definition.
__device__ static Real dMatFilmThickness(const DScene& sc, const DMaterial& m, const DHit& h);

__device__ static bool thinFilmInterface(const DScene& sc, const DMaterial& m, const DHit& h,
                                          const DVec3& d,
                                          Real lambda, DRng& rng, DVec3& ro, DVec3& rd) {
    Real ns = specLookup(m.ior, lambda), nf = (Real)m.filmIor;
    Real ks = specLookup(m.substrateK, lambda);
    Real thickness = dMatFilmThickness(sc, m, h);   // per-hit (map or constant)
    bool entering = dot(d, h.ng) < 0;
    DVec3 nl = entering ? h.ng : -h.ng;
    Real cosI = -dot(d, nl);
    if (ks > 0) {                                // opaque metal-backed film
        if (!entering) return false;             // inside absorbing substrate: absorbed
        Real R = thinFilmReflectance((Real)1, nf, ns, ks, thickness, cosI, lambda);
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
        Real R = thinFilmReflectance(nA, nf, nB, (Real)0, thickness, cosI, lambda);
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

// Accumulate one photon contribution into the film cell, and count it in `hits` (the
// per-pixel contribution count that drives the CPU-side graininess/noise estimate;
// mirrors Film::add incrementing hits by 1). `hits` may be null for callers that
// don't track it.
__device__ static void filmAdd(double* film, double* hits, int resX, int px, int py,
                               Real lambda, Real w) {
    size_t pix = (size_t)py * resX + px;
    size_t idx = pix * 3;
    atomicAdd(&film[idx + 0], (double)(cieX(lambda) * w));
    atomicAdd(&film[idx + 1], (double)(cieY(lambda) * w));
    atomicAdd(&film[idx + 2], (double)(cieZ(lambda) * w));
    if (hits) atomicAdd(&hits[pix], 1.0);
}
__device__ static void connect(const DScene& sc, const DCamera& cam, double* film, double* hits,
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
    // Projection-general splat: contrib = beta*f*cosSurf / (dist^2 * pixelSolidAngle).
    // For a rectilinear lens pixelSolidAngle = pixelPlaneArea*cosCam^3, recovering the
    // classic 1/(A_pix cos^4) form; fisheye/panoramic uses the remapped solid angle.
    double solidAngle = cam.pixelSolidAngle(cosCam);
    Real contrib = beta * f * cosSurf / (Real)((double)dist2 * solidAngle);
    if (sc.medium.enabled) contrib *= exp(-medSigmaT(sc.medium, lambda) * dist);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}
__device__ static void connectVolume(const DScene& sc, const DCamera& cam, double* film, double* hits,
                                      const DVec3& p, const DVec3& wIn, Real lambda, Real beta) {
    DVec3 toCam = cam.eye - p;
    Real dist = length(toCam);
    DVec3 wdir = toCam / dist;
    int px, py; Real cosCam, dist2;
    if (!cam.project(p, px, py, cosCam, dist2)) return;
    if (occluded(sc, p + wdir * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    Real ph = hgPhase(dot(wIn, wdir), (Real)sc.medium.g);
    Real Lambda = medAlbedo(sc.medium, lambda);
    // Projection-general splat (no cosSurf for a volume vertex): the phase carries the
    // scattering; normalise by dist^2 * pixelSolidAngle (rectilinear or fisheye).
    double solidAngle = cam.pixelSolidAngle(cosCam);
    Real contrib = beta * Lambda * ph / (Real)((double)dist2 * solidAngle);
    contrib *= exp(-medSigmaT(sc.medium, lambda) * dist);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}
// Model A (physical camera): next-event splat through the finite lens pupil. Sample a
// point A uniformly on the aperture disc, connect the surface vertex to A, refract
// through the thin lens and splat onto the film cell A images to. Port of
// Renderer::connectLens — the importance-sampled form of model C's brute-force catch,
// so A and C share both scale and shape. Weight beta*rho*cosSurf*cosLens*R^2/dist^2
// (the BRDF's 1/pi cancels the pupil pdf's pi R^2).
__device__ static void connectLens(const DScene& sc, const DCamera& cam, double* film, double* hits,
                                   const DVec3& p, const DVec3& n, Real lambda, Real beta,
                                   Real rho, DRng& rng) {
    Real R  = (Real)cam.apertureR;
    Real rr = R * sqrt(rng.uniform());
    Real a  = (Real)(2.0 * DPI) * rng.uniform();
    DVec3 A = cam.eye + cam.u * (rr * cos(a)) + cam.v * (rr * sin(a));
    DVec3 toA = A - p;
    Real dist = length(toA);
    if (dist < (Real)1e-9) return;
    DVec3 wdir = toA / dist;
    Real cosSurf = dot(n, wdir);
    if (cosSurf <= 0) return;                        // pupil behind the surface
    Real cosLens = -dot(wdir, cam.w);                // cosine at the lens (w faces scene)
    if (cosLens <= (Real)1e-6) return;               // not heading toward the film
    int px, py;
    if (!cam.lensImage(A, wdir, px, py)) return;
    if (occluded(sc, p + n * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    Real contrib = beta * rho * cosSurf * cosLens * (R * R) / (dist * dist);
    if (sc.medium.enabled) contrib *= exp(-medSigmaT(sc.medium, lambda) * dist);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}
// Model A lens splat for a VOLUME scattering vertex (fog). As connectLens but the
// surface BRDF*cosSurf is replaced by albedo*phase; the phase carries no 1/pi, so the
// pupil pdf's pi R^2 stays. Port of Renderer::connectLensVolume.
__device__ static void connectLensVolume(const DScene& sc, const DCamera& cam, double* film, double* hits,
                                         const DVec3& p, const DVec3& wIn, Real lambda,
                                         Real beta, DRng& rng) {
    Real R  = (Real)cam.apertureR;
    Real rr = R * sqrt(rng.uniform());
    Real a  = (Real)(2.0 * DPI) * rng.uniform();
    DVec3 A = cam.eye + cam.u * (rr * cos(a)) + cam.v * (rr * sin(a));
    DVec3 toA = A - p;
    Real dist = length(toA);
    if (dist < (Real)1e-9) return;
    DVec3 wdir = toA / dist;
    Real cosLens = -dot(wdir, cam.w);
    if (cosLens <= (Real)1e-6) return;
    int px, py;
    if (!cam.lensImage(A, wdir, px, py)) return;
    if (occluded(sc, p + wdir * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    Real ph = hgPhase(dot(wIn, wdir), (Real)sc.medium.g);
    Real Lambda = medAlbedo(sc.medium, lambda);
    Real contrib = beta * Lambda * ph * cosLens * (Real)DPI * (R * R) / (dist * dist);
    contrib *= exp(-medSigmaT(sc.medium, lambda) * dist);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
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

// Wrap a texel index into [0,n) per the texture's wrap mode (mirrors Texture::wrapIndex).
__device__ static int dWrapIndex(int i, int n, int wrap) {
    if (wrap == 1) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }       // Clamp
    if (wrap == 2) {                                                  // Mirror
        int period = 2 * n;
        int m = ((i % period) + period) % period;
        return (m < n) ? m : (period - 1 - m);
    }
    int m = i % n; return (m < 0) ? m + n : m;                        // Repeat
}

// Spatially-varying reflectance at (u,v,lambda): bilerp the per-texel JH coeffs
// (v flipped so v=0 is the image bottom) then evaluate the sigmoid. The exact
// device twin of Texture::reflectanceAt (nearest + bilinear filtering).
__device__ static Real dTexReflAt(const DTexture& tx, Real u, Real v, Real lambda) {
    if (tx.filter == 0) {   // Nearest
        int x = dWrapIndex((int)floor((double)u * tx.w), tx.w, tx.wrap);
        int y = dWrapIndex((int)floor((1.0 - (double)v) * tx.h), tx.h, tx.wrap);
        return dReflAt(&tx.coeff[3 * ((size_t)y * tx.w + x)], lambda);
    }
    double tu = (double)u * tx.w - 0.5, tv = (1.0 - (double)v) * tx.h - 0.5;
    double flx = floor(tu), fly = floor(tv);
    double fx = tu - flx, fy = tv - fly;
    int x0 = dWrapIndex((int)flx, tx.w, tx.wrap), x1 = dWrapIndex((int)flx + 1, tx.w, tx.wrap);
    int y0 = dWrapIndex((int)fly, tx.h, tx.wrap), y1 = dWrapIndex((int)fly + 1, tx.h, tx.wrap);
    const double* c00 = &tx.coeff[3 * ((size_t)y0 * tx.w + x0)];
    const double* c10 = &tx.coeff[3 * ((size_t)y0 * tx.w + x1)];
    const double* c01 = &tx.coeff[3 * ((size_t)y1 * tx.w + x0)];
    const double* c11 = &tx.coeff[3 * ((size_t)y1 * tx.w + x1)];
    double c[3];
    for (int k = 0; k < 3; ++k) {
        double a = c00[k] * (1 - fx) + c10[k] * fx;
        double b = c01[k] * (1 - fx) + c11[k] * fx;
        c[k] = a * (1 - fy) + b * fy;
    }
    return dReflAt(c, lambda);
}

// Triplanar (box) projection reflectance at a world hit: sample the texture from
// the three world axes (plane ⊥X at (z,y), ⊥Y at (x,z), ⊥Z at (x,y), each scaled)
// and blend by |n|^4 componentwise. Exact device twin of Texture::reflectanceTriplanar.
__device__ static Real dTexReflTriplanar(const DTexture& tx, const DVec3& p, const DVec3& n,
                                         double scale, Real lambda) {
    double ax = fabs((double)n.x), ay = fabs((double)n.y), az = fabs((double)n.z);
    double wx = ax * ax * ax * ax, wy = ay * ay * ay * ay, wz = az * az * az * az;
    double s = wx + wy + wz;
    if (s <= 0.0) return dTexReflAt(tx, (Real)(p.x * scale), (Real)(p.y * scale), lambda);
    wx /= s; wy /= s; wz /= s;
    double r = 0.0;
    if (wx > 0.0) r += wx * (double)dTexReflAt(tx, (Real)(p.z * scale), (Real)(p.y * scale), lambda);
    if (wy > 0.0) r += wy * (double)dTexReflAt(tx, (Real)(p.x * scale), (Real)(p.z * scale), lambda);
    if (wz > 0.0) r += wz * (double)dTexReflAt(tx, (Real)(p.x * scale), (Real)(p.y * scale), lambda);
    return (Real)r;
}

// Scalar (grayscale) texture sample at (u,v) — device twin of Texture::scalarAt.
// Bilerps the per-texel `gray` array (mean linear RGB); used for non-albedo scalar
// maps (roughness, film thickness, §9.4). v flipped so v=0 is the image bottom.
__device__ static double dTexScalarAt(const DTexture& tx, Real u, Real v) {
    if (!tx.gray) return 0.5;
    if (tx.filter == 0) {   // Nearest
        int x = dWrapIndex((int)floor((double)u * tx.w), tx.w, tx.wrap);
        int y = dWrapIndex((int)floor((1.0 - (double)v) * tx.h), tx.h, tx.wrap);
        return tx.gray[(size_t)y * tx.w + x];
    }
    double tu = (double)u * tx.w - 0.5, tv = (1.0 - (double)v) * tx.h - 0.5;
    double flx = floor(tu), fly = floor(tv);
    double fx = tu - flx, fy = tv - fly;
    int x0 = dWrapIndex((int)flx, tx.w, tx.wrap), x1 = dWrapIndex((int)flx + 1, tx.w, tx.wrap);
    int y0 = dWrapIndex((int)fly, tx.h, tx.wrap), y1 = dWrapIndex((int)fly + 1, tx.h, tx.wrap);
    double a = tx.gray[(size_t)y0 * tx.w + x0] * (1 - fx) + tx.gray[(size_t)y0 * tx.w + x1] * fx;
    double b = tx.gray[(size_t)y1 * tx.w + x0] * (1 - fx) + tx.gray[(size_t)y1 * tx.w + x1] * fx;
    return a * (1 - fy) + b * fy;
}

// ---- procedural pattern VM (device twin of pattern.h) ----------------------
// Deterministic integer-hash 3-D value noise; matches patHash3/patValueNoise so the
// GPU and CPU produce the same noise field. Output in [0,1].
__device__ static double dPatHash3(int ix, int iy, int iz) {
    unsigned int h = (unsigned int)ix * 374761393u + (unsigned int)iy * 668265263u
                   + (unsigned int)iz * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (double)h / 4294967295.0;
}
__device__ static double dPatValueNoise(double x, double y, double z) {
    double fx = floor(x), fy = floor(y), fz = floor(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    double tx = x - fx, ty = y - fy, tz = z - fz;
    double ux = tx * tx * (3.0 - 2.0 * tx);
    double uy = ty * ty * (3.0 - 2.0 * ty);
    double uz = tz * tz * (3.0 - 2.0 * tz);
    double c000 = dPatHash3(ix,     iy,     iz);
    double c100 = dPatHash3(ix + 1, iy,     iz);
    double c010 = dPatHash3(ix,     iy + 1, iz);
    double c110 = dPatHash3(ix + 1, iy + 1, iz);
    double c001 = dPatHash3(ix,     iy,     iz + 1);
    double c101 = dPatHash3(ix + 1, iy,     iz + 1);
    double c011 = dPatHash3(ix,     iy + 1, iz + 1);
    double c111 = dPatHash3(ix + 1, iy + 1, iz + 1);
    double x00 = c000 + (c100 - c000) * ux;
    double x10 = c010 + (c110 - c010) * ux;
    double x01 = c001 + (c101 - c001) * ux;
    double x11 = c011 + (c111 - c011) * ux;
    double y0  = x00 + (x10 - x00) * uy;
    double y1  = x01 + (x11 - x01) * uy;
    return y0 + (y1 - y0) * uz;
}
// Postfix scalar-stack evaluator (exact port of patternEval). PatNode/PatOp are the
// POD host types (pattern.h), uploaded verbatim; variables come in as scalar args.
__device__ static double dPatternEval(const PatNode* nodes, int n,
                                      double x, double y, double z, double f,
                                      double nx, double ny, double nz, double r) {
    double st[64]; int sp = 0;
    for (int i = 0; i < n; ++i) {
        const PatNode& nd = nodes[i];
        switch (nd.op) {
            case PatOp::Const:    st[sp++] = nd.a; break;
            case PatOp::VarX:     st[sp++] = x;  break;
            case PatOp::VarY:     st[sp++] = y;  break;
            case PatOp::VarZ:     st[sp++] = z;  break;
            case PatOp::VarF:     st[sp++] = f;  break;
            case PatOp::VarNx:    st[sp++] = nx; break;
            case PatOp::VarNy:    st[sp++] = ny; break;
            case PatOp::VarNz:    st[sp++] = nz; break;
            case PatOp::VarR:     st[sp++] = r;  break;
            case PatOp::Neg:      st[sp-1] = -st[sp-1]; break;
            case PatOp::Abs:      st[sp-1] = fabs(st[sp-1]); break;
            case PatOp::Sqrt:     st[sp-1] = sqrt(fmax(0.0, st[sp-1])); break;
            case PatOp::Sin:      st[sp-1] = sin(st[sp-1]); break;
            case PatOp::Cos:      st[sp-1] = cos(st[sp-1]); break;
            case PatOp::Tan:      st[sp-1] = tan(st[sp-1]); break;
            case PatOp::Exp:      st[sp-1] = exp(st[sp-1]); break;
            case PatOp::Log:      st[sp-1] = log(fmax(1e-300, st[sp-1])); break;
            case PatOp::Floor:    st[sp-1] = floor(st[sp-1]); break;
            case PatOp::Fract:    st[sp-1] = st[sp-1] - floor(st[sp-1]); break;
            case PatOp::Sign:     st[sp-1] = (st[sp-1] > 0.0) - (st[sp-1] < 0.0); break;
            case PatOp::Saturate: st[sp-1] = fmin(1.0, fmax(0.0, st[sp-1])); break;
            case PatOp::Add:      { double b = st[--sp]; st[sp-1] += b; break; }
            case PatOp::Sub:      { double b = st[--sp]; st[sp-1] -= b; break; }
            case PatOp::Mul:      { double b = st[--sp]; st[sp-1] *= b; break; }
            case PatOp::Div:      { double b = st[--sp]; st[sp-1] = (b != 0.0) ? st[sp-1] / b : 0.0; break; }
            case PatOp::Mod:      { double b = st[--sp]; st[sp-1] = (b != 0.0) ? st[sp-1] - b * floor(st[sp-1] / b) : 0.0; break; }
            case PatOp::Pow:      { double b = st[--sp]; st[sp-1] = pow(st[sp-1], b); break; }
            case PatOp::Min:      { double b = st[--sp]; st[sp-1] = fmin(st[sp-1], b); break; }
            case PatOp::Max:      { double b = st[--sp]; st[sp-1] = fmax(st[sp-1], b); break; }
            case PatOp::Atan2:    { double b = st[--sp]; st[sp-1] = atan2(st[sp-1], b); break; }
            case PatOp::Step:     { double b = st[--sp]; st[sp-1] = (b >= st[sp-1]) ? 1.0 : 0.0; break; }
            case PatOp::Clamp:    { double hi = st[--sp], lo = st[--sp]; st[sp-1] = fmin(hi, fmax(lo, st[sp-1])); break; }
            case PatOp::Mix:      { double t = st[--sp], b = st[--sp]; st[sp-1] = st[sp-1] + (b - st[sp-1]) * t; break; }
            case PatOp::Smoothstep: {
                double xx = st[--sp], e1 = st[--sp], e0 = st[sp-1];
                double tt = (e1 != e0) ? (xx - e0) / (e1 - e0) : 0.0;
                tt = fmin(1.0, fmax(0.0, tt));
                st[sp-1] = tt * tt * (3.0 - 2.0 * tt);
                break;
            }
            case PatOp::Noise:    { double zz = st[--sp], yy = st[--sp]; st[sp-1] = dPatValueNoise(st[sp-1], yy, zz); break; }
        }
    }
    return sp > 0 ? st[0] : 0.0;
}
// Evaluate a bound pattern at a hit (device twin of patternScalarAt/patCtxFromHit).
// The implicit field value f is 0 (like the CPU: intersectImplicit never sets it), so
// the `f` variable is 0 at surfaces on both backends.
__device__ static double dPatternScalarAt(const DScene& sc, int pat, const DHit& h) {
    const DPattern& p = sc.patterns[pat];
    double px = h.p.x, py = h.p.y, pz = h.p.z;
    double r = sqrt(px * px + py * py + pz * pz);
    return dPatternEval(sc.patNodes + p.off, p.n, px, py, pz, 0.0,
                        h.n.x, h.n.y, h.n.z, r);
}

// Per-hit glossy roughness / thin-film thickness (device twins of materialRoughness
// / materialFilmThickness): a bound pattern (highest priority) or scalar map's value
// at the hit, else the constant.
__device__ static Real dMatRoughness(const DScene& sc, const DMaterial& m, const DHit& h) {
    if (m.roughnessPat >= 0) {
        double r = dPatternScalarAt(sc, m.roughnessPat, h);
        return (Real)(r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r));
    }
    if (m.roughnessTex >= 0) return (Real)dTexScalarAt(sc.textures[m.roughnessTex], h.u, h.v);
    return (Real)m.roughness;
}
__device__ static Real dMatFilmThickness(const DScene& sc, const DMaterial& m, const DHit& h) {
    if (m.filmThicknessPat >= 0)
        return (Real)(dPatternScalarAt(sc, m.filmThicknessPat, h) * m.filmThickness);
    if (m.filmThicknessTex >= 0)
        return (Real)(dTexScalarAt(sc.textures[m.filmThicknessTex], h.u, h.v) * m.filmThickness);
    return (Real)m.filmThickness;
}

// Device twin of scene.h mixResolveChild: resolve a D_MIX to a child index, honouring
// an optional per-hit blend mask (2-child mix: map value t = prob of child 0). Returns
// -1 for the leftover absorption slice (constant-weight path only). u is one uniform.
__device__ static int dMixResolveChild(const DScene& sc, const DMaterial& m, const DHit& h, Real u) {
    if ((m.mixWeightPat >= 0 || m.mixWeightTex >= 0) && m.mixCount == 2) {
        Real t = (m.mixWeightPat >= 0)
               ? (Real)dPatternScalarAt(sc, m.mixWeightPat, h)
               : (Real)dTexScalarAt(sc.textures[m.mixWeightTex], h.u, h.v);
        if (t < 0) t = 0; else if (t > 1) t = 1;
        return (u < t) ? m.mixChild[0] : m.mixChild[1];
    }
    Real acc = 0;
    for (int k = 0; k < m.mixCount; ++k) { acc += (Real)m.mixWeight[k]; if (u < acc) return m.mixChild[k]; }
    return -1;
}

// Diffuse reflectance at a hit: texture-sampled when the material binds one, else
// the constant baked reflect spectrum (mirrors host diffuseReflectance).
__device__ static Real dDiffuseRho(const DScene& sc, const DMaterial& m, const DHit& h, Real lambda) {
    if (m.reflectTex >= 0) {
        const DTexture& tx = sc.textures[m.reflectTex];
        if (m.triplanarScale > 0.0)
            return clamp01(dTexReflTriplanar(tx, h.p, h.ng, m.triplanarScale, lambda));
        return clamp01(dTexReflAt(tx, h.u, h.v, lambda));
    }
    return clamp01(specLookup(m.reflect, lambda));
}

// Sample a Stokes-shifted emission wavelength lambda' ~ M for a fluorescent
// material (mirrors EmissionSampler::sample over [DLMIN, DLMAX]).
__device__ static Real sampleFluoEmit(const DScene& sc, const DMaterial& m, DRng& rng) {
    const double* cdf = sc.fluoCdfAll + m.fluoCdfOffset;
    double u = (double)rng.uniform();
    int lo = 0, hi = m.fluoCdfN - 1;
    while (lo + 1 < hi) { int mid = (lo + hi) / 2; if (cdf[mid] <= u) lo = mid; else hi = mid; }
    double c0 = cdf[lo], c1 = cdf[lo + 1];
    double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
    return (Real)(DLMIN + (lo + frac) * m.fluoCdfStep);
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

// ---- shared photon physics (megakernel + wavefront share these exactly) ----
// Both backends run identical physics; only the *scheduling* of the two stages
// differs (megakernel: an inner per-thread loop; wavefront: separate coherent
// launches over a persistent state pool). Because rng is threaded by reference
// through both stages in the same call order, the megakernel's RNG stream — and
// thus its image and energy report — is bit-for-bit unchanged by this refactor.

enum { WF_CONTINUE = 0, WF_TERMINATE = 1 };

// Sample one photon from the emitters: fills ro/rd/beta/lambda, accumulates the
// emitted energy, and performs the direct emitter->camera connection (models A/B).
// Returns false when the wavelength draw yields a zero pdf (skip this photon).
__device__ static bool genPhoton(const DScene& sc, const DCamera& cam, double* film, double* hits,
                                 int camMode, DRng& rng,
                                 DVec3& ro, DVec3& rd, Real& beta, Real& lambda, double& eEmitted) {
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
        double rdd = sc.sceneRadius * sqrt((double)rng.uniform());
        double pd = 2.0 * 3.14159265358979323846 * (double)rng.uniform();
        DVec3 disk = t * (Real)(rdd * cos(pd)) + b * (Real)(rdd * sin(pd));
        origin = sc.sceneCenter - dir * (Real)sc.sceneRadius + disk;
        emitN = dir;
    } else {
        emitterSamplePoint(em, u1, u2, origin, emitN);   // quad: constant normal; sphere: surface point
        dir = em.collimated ? em.beamDir : cosineHemisphere(emitN, rng);
    }
    Real pdfL = 0;
    lambda = sampleLambda(sc, em, rng, pdfL);
    if (pdfL <= 0) return false;
    beta = (Real)((sc.nEmitters == 1) ? em.power : sc.totalPower);
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

    // Connect the emitter itself to the camera (makes the source visible): model
    // B splats to the pinhole, model A splats through the finite lens pupil. Model
    // C instead catches photons that physically arrive. A spot is a point light
    // with no projected area, so it has no direct term.
    if (em.shape != 2 && em.shape != 3) {
        if (camMode == CAM_B) connect(sc, cam, film, hits, origin, emitN, lambda, beta, (Real)1);
        else if (camMode == CAM_A) connectLens(sc, cam, film, hits, origin, emitN, lambda, beta, (Real)1, rng);
    }

    ro = origin + dir * RAY_EPS; rd = dir;
    return true;
}

// Advance a photon by one bounce given its precomputed intersection `h`. Mutates
// ro/rd/beta and accumulates absorbed/sensor/escaped energy. Returns WF_TERMINATE
// when the path ends (absorbed / escaped / landed on the sensor), else WF_CONTINUE
// with ro/rd set for the next segment. `h` is the closestHit(sc, ro, rd) result.
__device__ static int shadeStep(const DScene& sc, const DCamera& cam, double* film, double* hits,
                                int camMode, int diffraction, const DHit& h,
                                DVec3& ro, DVec3& rd, Real& beta, Real& lambda, DRng& rng,
                                double& eAbsorbed, double& eSensor, double& eEscaped,
                                int& interior) {
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
            filmAdd(film, hits, cam.resX, px, py, lambda, beta);
            eSensor += beta; return WF_TERMINATE;
        }
    }

    // Beer-Lambert attenuation over the free path just travelled inside a dielectric
    // (colored/attenuating glass), applied before the event is processed (matches the
    // host: attenuate over dEvent using the medium carried from the previous vertex).
    if (interior >= 0) {
        Real a = (Real)specLookup(sc.mats[interior].absorb, lambda);
        if (a > 0) beta *= exp(-a * dEvent);
    }

    if (mediumEvent) {
        if (camMode == CAM_B) connectVolume(sc, cam, film, hits, mp, rd, lambda, beta);
        else if (camMode == CAM_A) connectLensVolume(sc, cam, film, hits, mp, rd, lambda, beta, rng);
        if (rng.uniform() >= medAlbedo(sc.medium, lambda)) { eAbsorbed += beta; return WF_TERMINATE; }
        DVec3 nd = sampleHG(rd, (Real)sc.medium.g, rng);
        ro = mp; rd = nd;
        return WF_CONTINUE;
    }

    if (!h.valid) { eEscaped += beta; return WF_TERMINATE; }
    if (h.sensorId >= 0) {
        // Legacy contact sensor: no geometry carries a sensorId in the current
        // camera modes, so this is inert (kept for absorption bookkeeping).
        eSensor += beta; return WF_TERMINATE;
    }

    const DMaterial* mptr = &sc.mats[h.matId];
    int matIndex = h.matId;
    // Stochastic mix: resolve to a child lobe (or absorb) before dispatch.
    if (mptr->type == D_MIX) {
        int child = dMixResolveChild(sc, *mptr, h, rng.uniform());
        if (child < 0) { eAbsorbed += beta; return WF_TERMINATE; }
        mptr = &sc.mats[child]; matIndex = child;
    }
    const DMaterial& m = *mptr;
    if (m.type == D_DIELECTRIC) {
        bool entering = dot(rd, h.ng) < 0;
        bool transmitted = false;
        DVec3 nro, nrd; refractOrReflect(sc, m, h, rd, lambda, rng, nro, nrd, &transmitted);
        if (transmitted) interior = entering ? matIndex : -1;   // track medium for Beer-Lambert
        ro = nro; rd = nrd; return WF_CONTINUE;
    } else if (m.type == D_THINFILM) {
        DVec3 nro, nrd;
        if (!thinFilmInterface(sc, m, h, rd, lambda, rng, nro, nrd)) { eAbsorbed += beta; return WF_TERMINATE; }
        ro = nro; rd = nrd; return WF_CONTINUE;
    } else if (m.type == D_MULTILAYER) {
        DVec3 nro, nrd;
        if (!multilayerInterface(m, h, rd, lambda, rng, nro, nrd)) { eAbsorbed += beta; return WF_TERMINATE; }
        ro = nro; rd = nrd; return WF_CONTINUE;
    } else if (m.type == D_MIRROR) {
        Real r = clamp01(specLookup(m.reflect, lambda));
        if (rng.uniform() >= r) { eAbsorbed += beta; return WF_TERMINATE; }
        DVec3 o = reflectv(rd, h.n); ro = h.p + h.n * RAY_EPS; rd = o; return WF_CONTINUE;
    } else if (m.type == D_GRATING) {
        Real r = clamp01(specLookup(m.reflect, lambda));
        if (rng.uniform() >= r) { eAbsorbed += beta; return WF_TERMINATE; }
        DVec3 nro, nrd;
        if (!gratingDiffract(m, h, rd, lambda, diffraction, rng, nro, nrd)) { eAbsorbed += beta; return WF_TERMINATE; }
        ro = nro; rd = nrd; return WF_CONTINUE;
    } else if (m.type == D_HALFMIRROR) {
        Real r = clamp01(specLookup(m.reflect, lambda));
        if (rng.uniform() < r) { DVec3 o = reflectv(rd, h.n); ro = h.p + h.n * RAY_EPS; rd = o; }
        else { ro = h.p + rd * RAY_EPS; }
        return WF_CONTINUE;
    } else if (m.type == D_GLOSSY) {
        Real r = clamp01(specLookup(m.reflect, lambda));
        if (rng.uniform() >= r) { eAbsorbed += beta; return WF_TERMINATE; }
        DVec3 o = sampleGlossy(reflectv(rd, h.n), dMatRoughness(sc, m, h), rng);
        if (dot(o, h.n) <= 0) { eAbsorbed += beta; return WF_TERMINATE; }
        ro = h.p + h.n * RAY_EPS; rd = o; return WF_CONTINUE;
    } else if (m.type == D_FLUORESCENT) {
        // Two competing channels: elastic diffuse reflection (albedo rho, wavelength
        // preserved) and dye excitation (prob aEff = min(eps, 1-rho) so the channels
        // never exceed unity). Excited photons re-radiate (prob fluoYield) at a
        // Stokes-shifted lambda' ~ M. The camera sees both: an elastic splat at
        // lambda, and a glow splat at lambda' with albedo aEff*fluoYield. Mirrors the
        // host MatType::Fluorescent branch (render.h) + fluoroInteract.
        Real rho = dDiffuseRho(sc, m, h, lambda);
        Real eps = clamp01(specLookup(m.fluoAbsorb, lambda));
        Real oneMinusRho = (Real)1 - rho; if (oneMinusRho < 0) oneMinusRho = 0;
        Real aEff = eps < oneMinusRho ? eps : oneMinusRho;
        bool canGlow = (aEff > 0 && m.fluoYield > 0 && m.fluoCdfN > 0);
        if (camMode == CAM_B) {
            connect(sc, cam, film, hits, h.p, h.n, lambda, beta, rho);
            if (canGlow) {
                Real lp = sampleFluoEmit(sc, m, rng);
                connect(sc, cam, film, hits, h.p, h.n, lp, beta, (Real)(aEff * m.fluoYield));
            }
        } else if (camMode == CAM_A) {
            connectLens(sc, cam, film, hits, h.p, h.n, lambda, beta, rho, rng);
            if (canGlow) {
                Real lp = sampleFluoEmit(sc, m, rng);
                connectLens(sc, cam, film, hits, h.p, h.n, lp, beta, (Real)(aEff * m.fluoYield), rng);
            }
        }
        // Stochastic interaction (fluoroInteract): elastic / reemit / absorb. Beta is
        // unchanged in both surviving branches (M/pdf cancels for the sampled lambda').
        Real u = rng.uniform();
        if (u < rho) {
            /* elastic: lambda unchanged */
        } else if (u < rho + aEff) {
            if (rng.uniform() >= m.fluoYield) { eAbsorbed += beta; return WF_TERMINATE; }
            lambda = sampleFluoEmit(sc, m, rng);   // Stokes-shifted re-radiation
        } else {
            eAbsorbed += beta; return WF_TERMINATE;
        }
        ro = h.p + h.n * RAY_EPS; rd = cosineHemisphere(h.n, rng); return WF_CONTINUE;
    } else {
        // Diffuse (texture-sampled reflectance when the material binds a texture).
        Real rho = dDiffuseRho(sc, m, h, lambda);
        if (camMode == CAM_B) connect(sc, cam, film, hits, h.p, h.n, lambda, beta, rho);
        else if (camMode == CAM_A) connectLens(sc, cam, film, hits, h.p, h.n, lambda, beta, rho, rng);
        if (rng.uniform() >= rho) { eAbsorbed += beta; return WF_TERMINATE; }
        ro = h.p + h.n * RAY_EPS; rd = cosineHemisphere(h.n, rng); return WF_CONTINUE;
    }
}

__global__ void kTrace(DScene sc, DCamera cam, double* film, double* hits, double* energy,
                       long long N, int diffraction, unsigned long long seedBase, int maxBounce,
                       int camMode) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    DRng rng; rng.seed((unsigned long long)(g * 2 + 1), seedBase ^ (unsigned long long)g);

    double eEmitted = 0, eAbsorbed = 0, eSensor = 0, eEscaped = 0, eResidual = 0;

    for (long long i = g; i < N; i += G) {
        DVec3 ro, rd; Real beta, lambda;
        if (!genPhoton(sc, cam, film, hits, camMode, rng, ro, rd, beta, lambda, eEmitted)) continue;
        bool done = false;
        int interior = -1;   // dielectric the photon is currently inside (-1 = vacuum)
        for (int bounce = 0; bounce < maxBounce && !done; ++bounce) {
            DHit h = closestHit(sc, ro, rd);
            if (shadeStep(sc, cam, film, hits, camMode, diffraction, h, ro, rd, beta, lambda, rng,
                          eAbsorbed, eSensor, eEscaped, interior) == WF_TERMINATE) done = true;
        }
        if (!done) eResidual += beta;
    }

    atomicAdd(&energy[0], eEmitted);
    atomicAdd(&energy[1], eAbsorbed);
    atomicAdd(&energy[2], eSensor);
    atomicAdd(&energy[3], eEscaped);
    atomicAdd(&energy[4], eResidual);
}

// ============================ wavefront (streaming) backend ==================
// Same physics as the megakernel (genPhoton + shadeStep, identical device code),
// but scheduled as separate coherent kernel launches over a *persistent* pool of
// photon slots instead of one long-running per-thread loop. Each pass runs the two
// stages — extend (one closestHit per live slot) then shade (one shadeStep) — across
// the whole pool, so a warp's threads execute the same stage together rather than
// diverging on per-photon path length. When a path terminates, its slot immediately
// regenerates a fresh photon (path compaction by regeneration), keeping SIMD lanes
// full until the N-photon budget is spent. This wins on divergent / deep-path scenes
// and small GPUs; the megakernel wins on shallow, uniform scenes on big GPUs, so the
// backend is selectable (default = megakernel). See known-issues.md "GPU scaling path".
//
// The RNG stream differs from the megakernel (each slot, not each grid-stride thread,
// carries a stream), so images are NOT bit-identical — but the physics is the same, so
// energy conserves exactly and the two agree to within Monte-Carlo noise.

// SoA photon-state pool. One entry per slot; hit[] is filled by the extend stage and
// consumed by the shade stage of the same pass.
struct WFState {
    DVec3* ro;
    DVec3* rd;
    Real*  beta;
    Real*  lambda;
    DRng*  rng;
    int*   bounce;   // bounces already shaded for the photon currently in this slot
    int*   alive;    // 1 = slot holds a live photon, 0 = drained (budget spent)
    int*   interior; // dielectric material index the photon is inside (-1 = vacuum)
    DHit*  hit;      // extend-stage intersection, consumed by shade
};

// Claim photon budget and emit fresh photons into `slot` until one is successfully
// launched or the N-photon budget is exhausted. Mirrors the megakernel's per-iteration
// genPhoton: a zero-pdf wavelength draw is skipped but still consumes its budget index,
// so the total genPhoton count is exactly N across the whole render. Returns true and
// fills the slot (alive=1, bounce=0) on success; false when the budget is spent (the
// caller marks the slot dead). Emitted energy accrues into energy[0].
__device__ static bool wfSpawn(const DScene& sc, const DCamera& cam, double* film, double* hits,
                               double* energy, int camMode, long long N,
                               unsigned long long* dispatched, WFState st, int slot, DRng& rng) {
    for (;;) {
        unsigned long long idx = atomicAdd(dispatched, 1ULL);
        if (idx >= (unsigned long long)N) return false;
        DVec3 ro, rd; Real beta, lambda; double eEm = 0;
        if (genPhoton(sc, cam, film, hits, camMode, rng, ro, rd, beta, lambda, eEm)) {
            st.ro[slot] = ro; st.rd[slot] = rd;
            st.beta[slot] = beta; st.lambda[slot] = lambda;
            st.bounce[slot] = 0; st.alive[slot] = 1; st.interior[slot] = -1;
            atomicAdd(&energy[0], eEm);
            return true;
        }
        // zero-pdf photon: its budget index is consumed, loop and try the next
    }
}

// Seed each slot's RNG and fill it with a first photon.
__global__ void kWfInit(DScene sc, DCamera cam, double* film, double* hits, double* energy,
                        WFState st, long long N, int W, unsigned long long* dispatched,
                        int* liveCount, unsigned long long seedBase, int camMode) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= W) return;
    DRng rng; rng.seed((unsigned long long)(slot * 2 + 1), seedBase ^ (unsigned long long)slot);
    bool live = wfSpawn(sc, cam, film, hits, energy, camMode, N, dispatched, st, slot, rng);
    st.rng[slot] = rng;
    if (live) atomicAdd(liveCount, 1);
    else st.alive[slot] = 0;
}

// Extend: one closestHit per live slot.
__global__ void kWfExtend(DScene sc, WFState st, int W) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= W || !st.alive[slot]) return;
    st.hit[slot] = closestHit(sc, st.ro[slot], st.rd[slot]);
}

// Shade: advance each live slot by one bounce; regenerate on termination / bounce cap.
__global__ void kWfShade(DScene sc, DCamera cam, double* film, double* hits, double* energy,
                         WFState st, int W, long long N, int diffraction, int maxBounce,
                         unsigned long long* dispatched, int* liveCount, int camMode) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= W || !st.alive[slot]) return;
    DRng rng = st.rng[slot];
    DVec3 ro = st.ro[slot], rd = st.rd[slot];
    Real beta = st.beta[slot], lambda = st.lambda[slot];
    DHit h = st.hit[slot];
    int interior = st.interior[slot];
    double eAbs = 0, eSen = 0, eEsc = 0;
    int res = shadeStep(sc, cam, film, hits, camMode, diffraction, h, ro, rd, beta, lambda, rng,
                        eAbs, eSen, eEsc, interior);
    int bounce = st.bounce[slot] + 1;
    bool pathDone = (res == WF_TERMINATE);
    // Bounce cap: the photon survived maxBounce shadeStep calls without terminating —
    // count its carried energy as residual, exactly as the megakernel's !done branch.
    if (!pathDone && bounce >= maxBounce) { atomicAdd(&energy[4], (double)beta); pathDone = true; }
    if (eAbs != 0.0) atomicAdd(&energy[1], eAbs);
    if (eSen != 0.0) atomicAdd(&energy[2], eSen);
    if (eEsc != 0.0) atomicAdd(&energy[3], eEsc);
    if (!pathDone) {
        st.ro[slot] = ro; st.rd[slot] = rd; st.beta[slot] = beta;
        st.bounce[slot] = bounce; st.rng[slot] = rng;
        st.lambda[slot] = lambda;   // fluorescence may Stokes-shift lambda mid-path
        st.interior[slot] = interior;   // carry the dielectric medium to the next segment
        return;
    }
    // Path finished: regenerate this slot from the remaining budget (compaction).
    bool live = wfSpawn(sc, cam, film, hits, energy, camMode, N, dispatched, st, slot, rng);
    st.rng[slot] = rng;
    if (!live) { st.alive[slot] = 0; atomicSub(liveCount, 1); }
}

// ============================ bidirectional path tracing (mode D) ============
// Device port of bdpt.h (Veach / PBRT-v3). One thread renders one (pixel,sample):
// it builds a camera subpath and a light subpath at a single shared wavelength,
// then MIS-connects every vertex pair (balance heuristic). Geometry stays in Real;
// all pdf/MIS arithmetic runs in double (ddot) to keep the balance-heuristic ratios
// stable, matching the CPU reference to within Monte-Carlo noise. Emissive surfaces
// and area/sphere lights only (spot/env/collimated/fog scenes fall back to the CPU
// via cudaBdptSupported). See bdpt.h for the derivation of every quantity below.

#define BDPT_MAXDEPTH 8
#define BDPT_MAXV     (BDPT_MAXDEPTH + 3)   // path[0] endpoint + up to MAXDEPTH surfaces + slack
enum { BV_CAMERA = 0, BV_LIGHT = 1, BV_SURFACE = 2 };

// A path vertex. Mirrors bdpt.h Vertex, but stores INDICES (matId into sc.mats,
// lightIdx into sc.emitters) instead of pointers, and drops the Hit field (the GPU
// rejects textured scenes, so albedo needs no surface-local (u,v)). pdfFwd/pdfRev/
// beta stay double for MIS stability; geometry (p/ns/ng) is Real.
struct DVertex {
    int   type;                 // BV_CAMERA / BV_LIGHT / BV_SURFACE
    DVec3 p, ns, ng;            // position, shading normal, geometric normal
    double beta;                // throughput carried to this vertex
    double pdfFwd, pdfRev;      // area-measure densities (0 for delta vertices)
    int   delta;                // 1 => specular (skipped in connections/MIS)
    int   matId;                // sc.mats index (-1 for camera)
    int   lightIdx;             // sc.emitters index if emissive, else -1
};

__device__ static inline double ddot(const DVec3& a, const DVec3& b) {
    return (double)a.x * b.x + (double)a.y * b.y + (double)a.z * b.z;
}
__device__ static inline bool dOnSurface(const DVertex& v) { return v.type != BV_CAMERA; }
__device__ static inline bool dConnectibleType(int tp) {
    return tp == D_DIFFUSE || tp == D_GLOSSY || tp == D_FLUORESCENT;
}
__device__ static bool dVertConnectible(const DScene& sc, const DVertex& v) {
    if (v.type == BV_CAMERA) return true;
    if (v.type == BV_LIGHT)  return v.lightIdx >= 0 && sc.emitters[v.lightIdx].collimated == 0;
    if (v.delta) return false;
    return dConnectibleType(sc.mats[v.matId].type);
}
__device__ static inline bool dIsLightVertex(const DVertex& v) {
    return v.type == BV_LIGHT || (v.type == BV_SURFACE && v.lightIdx >= 0);
}
__device__ static inline double dGlossyExp(double roughness) {
    double rr = roughness < 1e-3 ? 1e-3 : roughness;
    double e = 2.0 / (rr * rr) - 2.0;
    return e < 0 ? 0 : e;
}

// BSDF value f(wo->wi) at a surface vertex (double, for the connection radiance L).
__device__ static double dBsdfF(const DScene& sc, int matId, const DVec3& ns,
                                const DVec3& wo, const DVec3& wi, Real lambda) {
    const DMaterial& m = sc.mats[matId];
    double cosWi = ddot(wi, ns), cosWo = ddot(wo, ns);
    if (m.type == D_DIFFUSE || m.type == D_FLUORESCENT) {
        if (cosWi <= 0 || cosWo <= 0) return 0.0;
        double rho = clamp01(specLookup(m.reflect, lambda));
        return rho / DPI;
    } else if (m.type == D_GLOSSY) {
        if (cosWi <= 0 || cosWo <= 0) return 0.0;
        double r = clamp01(specLookup(m.reflect, lambda));
        double e = dGlossyExp(m.roughness);
        DVec3 mdir = reflectv(wo * (Real)-1, ns);
        double cosLobe = ddot(wi, mdir);
        if (cosLobe <= 0) return 0.0;
        double lobe = (e + 1.0) / (2.0 * DPI) * pow(cosLobe, e);
        return r * lobe / cosWi;
    }
    return 0.0;
}
// Directional pdf (solid angle) of sampling wi at a surface vertex (double, for MIS).
__device__ static double dBsdfPdf(const DScene& sc, int matId, const DVec3& ns,
                                  const DVec3& wo, const DVec3& wi) {
    const DMaterial& m = sc.mats[matId];
    double cosWi = ddot(wi, ns), cosWo = ddot(wo, ns);
    if (m.type == D_DIFFUSE || m.type == D_FLUORESCENT) {
        if (cosWi <= 0 || cosWo <= 0) return 0.0;
        return cosWi / DPI;
    } else if (m.type == D_GLOSSY) {
        if (cosWi <= 0 || cosWo <= 0) return 0.0;
        double e = dGlossyExp(m.roughness);
        DVec3 mdir = reflectv(wo * (Real)-1, ns);
        double cosLobe = ddot(wi, mdir);
        if (cosLobe <= 0) return 0.0;
        return (e + 1.0) / (2.0 * DPI) * pow(cosLobe, e);
    }
    return 0.0;
}

// Camera importance (PBRT imagePlaneArea convention — see bdpt.h cameraWe/PdfDir).
__device__ static double dCamCos(const DCamera& cam, const DVec3& p) {
    DVec3 d = p - cam.eye;
    double len = sqrt(ddot(d, d));
    return len > 0 ? ddot(d, cam.w) / len : 0.0;
}
__device__ static double dCameraPdfDir(const DCamera& cam, double cosCam) {
    if (cosCam <= 0) return 0.0;
    return 1.0 / (cam.imagePlaneArea() * cosCam * cosCam * cosCam);
}
__device__ static double dCameraWe(const DCamera& cam, double cosCam) {
    if (cosCam <= 0) return 0.0;
    double c2 = cosCam * cosCam;
    return 1.0 / (cam.imagePlaneArea() * c2 * c2);
}

// Convert a solid-angle pdf of leaving `from` toward `to` into an area density at `to`.
__device__ static double dConvertDensity(double pdfW, const DVertex& from, const DVertex& to) {
    DVec3 w = to.p - from.p;
    double d2 = ddot(w, w);
    if (d2 == 0.0) return 0.0;
    double invD2 = 1.0 / d2;
    if (dOnSurface(to)) pdfW *= fabs(ddot(to.ns, w * (Real)sqrt(invD2)));
    return pdfW * invD2;
}
// Emission directional density at a light vertex toward `next` (area measure).
__device__ static double dVertexPdfLight(const DVertex& cur, const DVertex& next) {
    DVec3 w = next.p - cur.p;
    double d2 = ddot(w, w);
    if (d2 == 0.0) return 0.0;
    double invD2 = 1.0 / d2;
    DVec3 wn = w * (Real)sqrt(invD2);
    double cosLight = ddot(cur.ng, wn);
    if (cosLight <= 0.0) return 0.0;
    double pdf = (cosLight / DPI) * invD2;
    if (dOnSurface(next)) pdf *= fabs(ddot(next.ns, wn));
    return pdf;
}
__device__ static double dVertexPdf(const DScene& sc, const DCamera& cam,
                                    const DVertex* prev, const DVertex& cur, const DVertex& next) {
    if (cur.type == BV_LIGHT) return dVertexPdfLight(cur, next);
    DVec3 wn = next.p - cur.p;
    if (ddot(wn, wn) == 0.0) return 0.0;
    wn = normalize(wn);
    double pdfW = 0.0;
    if (cur.type == BV_CAMERA) {
        pdfW = dCameraPdfDir(cam, dCamCos(cam, next.p));
    } else {
        if (!prev) return 0.0;
        DVec3 wp = prev->p - cur.p;
        if (ddot(wp, wp) == 0.0) return 0.0;
        wp = normalize(wp);
        pdfW = dBsdfPdf(sc, cur.matId, cur.ns, wp, wn);
    }
    return dConvertDensity(pdfW, cur, next);
}
__device__ static double dVertexPdfLightOrigin(const DScene& sc, const DVertex& cur) {
    if (cur.lightIdx < 0) return 0.0;
    const DEmitter& em = sc.emitters[cur.lightIdx];
    if (sc.totalPower <= 0.0 || em.area <= 0.0) return 0.0;
    double pdfChoice = em.power / sc.totalPower;
    return pdfChoice / em.area;
}
// Emitted radiance (single wavelength) leaving a light vertex toward w.
__device__ static double dVertexLe(const DScene& sc, const DVertex& v, const DVec3& w,
                                   Real lambda, double invPdfLambda) {
    if (v.lightIdx < 0) return 0.0;
    if (ddot(v.ng, w) <= 0.0) return 0.0;
    return (double)specLookup(sc.emitters[v.lightIdx].emitSpd, lambda) * invPdfLambda;
}
// Emitter that owns an emissive surface material (mirrors Scene::emitterForMat).
__device__ static int dEmitterForMat(const DScene& sc, int matId) {
    for (int i = 0; i < sc.nEmitters; ++i)
        if (sc.emitters[i].matId == matId) return i;
    return -1;
}

// Sample the shared wavelength from the scene emission sampler (mirrors
// EmissionSampler::sample). Sets pdf (per nm, for the >0 guard); the BDPT weight
// uses the continuous invPdfLambda below (exactly as the CPU path does).
__device__ static Real dSampleSceneLambda(const DScene& sc, DRng& rng, double& pdf) {
    double u = (double)rng.uniform();
    const double* cdf = sc.emitSamplerCdf;
    int lo = 0, hi = sc.emitSamplerN;
    while (lo + 1 < hi) { int m = (lo + hi) / 2; if (cdf[m] <= u) lo = m; else hi = m; }
    double c0 = cdf[lo], c1 = cdf[lo + 1];
    double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
    pdf = (c1 - c0) / sc.emitSamplerStep;
    return (Real)(DLMIN + (lo + frac) * sc.emitSamplerStep);
}
// invPdfLambda(lambda) = emitG / g(lambda), g(lambda) = sum_k geomWeight_k*SPD_k.
// In BDPT scope every emitter is an area/sphere light, so geomWeight = area*PI.
__device__ static double dInvPdfLambda(const DScene& sc, Real lambda) {
    double g = 0.0;
    for (int k = 0; k < sc.nEmitters; ++k) {
        const DEmitter& e = sc.emitters[k];
        g += (double)e.area * DPI * (double)specLookup(e.emitSpd, lambda);
    }
    return (g > 0.0) ? sc.emitG / g : 0.0;
}

// ======================= backward reference (GPU mode R) =====================
// Device port of backward.h — the unidirectional reference tracer, now with the
// physical (mesh-lens) camera as a ray-generation front-end. Reuses the shared BVH
// (closestHit/occluded), the specular BSDFs (refractOrReflect / thinFilmInterface /
// multilayerInterface / gratingDiffract), the diffuse reflectance (dDiffuseRho) and
// the emitter sampler exactly, so materials agree with the CPU path by construction.
// v1 scope (gated by cudaBackwardSupported): area/sphere/cylinder Lambertian lights
// only, no fog/env/fluorescence/spot — which makes dInvPdfLambda exact and matches
// the CPU reference up to Monte-Carlo noise (independent RNG realization).

// Sensor-side index of surface j (air baked as 1); scene-side = sensor side of j-1.
__device__ static Real dLensIorSensor(const DLensSystem& L, int j, Real lambda) {
    return specLookup(L.iorAll + (size_t)j * SPEC_N, lambda);
}
__device__ static Real dLensIorScene(const DLensSystem& L, int j, Real lambda) {
    return (j == 0) ? (Real)1 : dLensIorSensor(L, j - 1, lambda);
}
// Refract `d` at `n` (faced against d) with eta = n_in/n_out (port of refractDir).
__device__ static bool dLensRefract(const DVec3& d, const DVec3& n, Real eta, DVec3& out) {
    Real cosi = -dot(d, n);
    Real k = (Real)1 - eta * eta * ((Real)1 - cosi * cosi);
    if (k < 0) return false;                          // TIR -> blocked
    out = normalize(d * eta + n * (eta * cosi - sqrt(k)));
    return true;
}
// Intersect a lens-local ray with surface j; hitP + normal (against d). Clipped by the
// clear aperture => false (vignetting). Port of LensSystem::hitSurface.
__device__ static bool dLensHitSurface(const DLensSystem& L, int j, const DVec3& o,
                                       const DVec3& d, DVec3& hitP, DVec3& nrm) {
    double zv = L.surf[j].zpos, R = L.surf[j].radius, ap = L.surf[j].aperture;
    if (R == 0.0) {                                   // planar (stop / flat)
        if (fabs((double)d.z) < 1e-12) return false;
        double t = (zv - (double)o.z) / (double)d.z;
        if (t < 1e-9) return false;
        hitP = o + d * (Real)t;
        nrm = DVec3{0, 0, d.z > 0 ? -1.0 : 1.0};
    } else {
        DVec3 C{0, 0, zv + R};
        DVec3 op = o - C;
        Real b = dot(op, d);
        Real c = dot(op, op) - (Real)(R * R);
        Real disc = b * b - c;
        if (disc < 0) return false;
        Real sq = sqrt(disc);
        Real t0 = -b - sq, t1 = -b + sq;
        bool closer = (d.z > 0) ^ (R < 0);
        Real t = closer ? fmin(t0, t1) : fmax(t0, t1);
        if (t < (Real)1e-9) t = closer ? fmax(t0, t1) : fmin(t0, t1);
        if (t < (Real)1e-9) return false;
        hitP = o + d * t;
        nrm = normalize(hitP - C);
        if (dot(nrm, d) > 0) nrm = -nrm;
    }
    if ((double)hitP.x * (double)hitP.x + (double)hitP.y * (double)hitP.y > ap * ap)
        return false;                                 // clipped by the clear aperture
    return true;
}
// Trace a lens-local ray from the film out through every interface (sensor->scene;
// the only order the camera-ray generator needs). Port of LensSystem::trace.
__device__ static bool dLensTrace(const DLensSystem& L, const DVec3& o0, const DVec3& d0,
                                  Real lambda, DVec3& outO, DVec3& outD) {
    DVec3 o = o0, d = normalize(d0);
    for (int j = L.nSurf - 1; j >= 0; --j) {
        DVec3 hp, n;
        if (!dLensHitSurface(L, j, o, d, hp, n)) return false;
        o = hp;
        if (L.surf[j].radius != 0.0) {
            Real eta = dLensIorSensor(L, j, lambda) / dLensIorScene(L, j, lambda);
            DVec3 nd;
            if (!dLensRefract(d, n, eta, nd)) return false;
            d = nd;
        }
    }
    outO = o; outD = d;
    return true;
}
// Generate a world-space camera ray through the physical lens (port of
// Camera::genLensRay). Returns false on vignetting (element/stop clip or TIR); on
// success `weight` is the radiometric importance cos^4*A_rear/Z^2.
__device__ static bool dGenLensRay(const DCamera& cam, int px, int py, Real jx, Real jy,
                                   Real u1, Real u2, Real lambda,
                                   DVec3& oW, DVec3& dW, Real& weight) {
    weight = 0;
    const DLensSystem& L = cam.lens;
    double sx = 2.0 * ((px + jx) / (double)cam.resX) - 1.0;
    double sy = 2.0 * ((py + jy) / (double)cam.resY) - 1.0;
    double halfW = 0.5 * L.filmW_mm;
    double halfH = halfW * ((double)cam.resY / (double)cam.resX);
    DVec3 pFilm{-sx * halfW, -sy * halfH, L.filmZ};
    double rearAp = L.surf[L.nSurf - 1].aperture;
    double rearZ  = L.surf[L.nSurf - 1].zpos;
    double rr  = sqrt(u1 > 0 ? (double)u1 : 0.0) * rearAp;
    double phi = 2.0 * DPI * (double)u2;
    DVec3 pRear{rr * cos(phi), rr * sin(phi), rearZ};
    DVec3 d0 = normalize(pRear - pFilm);
    DVec3 oL, dL;
    if (!dLensTrace(L, pFilm, d0, lambda, oL, dL)) return false;
    Real cosT = d0.z;                                 // d0 unit; z = cos to axis
    if (cosT <= 0) return false;
    Real cos4 = (cosT * cosT) * (cosT * cosT);
    double A = DPI * rearAp * rearAp;
    double Z = rearZ - L.filmZ;
    if (Z <= 1e-9) return false;
    weight = (Real)((double)cos4 * A / (Z * Z));
    // Lens-local (mm) -> world: front vertex plane pinned at eye, mm -> scene metres.
    oW = cam.eye + (cam.u * oL.x + cam.v * oL.y) * (Real)1e-3
                 + cam.w * (Real)(((double)oL.z - L.T) * 1e-3);
    dW = normalize(cam.u * dL.x + cam.v * dL.y + cam.w * dL.z);
    return true;
}
// Pinhole/fisheye camera ray for pixel (px,py) (port of Camera::genRay). Used by the
// backward tracer when the camera has no physical lens.
__device__ static void dGenRay(const DCamera& cam, int px, int py, Real jx, Real jy,
                               DVec3& ro, DVec3& rd) {
    double sx = 2.0 * ((px + jx) / (double)cam.resX) - 1.0;
    double sy = 2.0 * ((py + jy) / (double)cam.resY) - 1.0;
    ro = cam.eye;
    if (cam.projection == CAM_RECTILINEAR) {
        rd = normalize(cam.w + cam.u * (Real)(sx * cam.tanHalfX) + cam.v * (Real)(sy * cam.tanHalfY));
        return;
    }
    double rho = sqrt(sx * sx + sy * sy);
    if (rho < 1e-12) { rd = cam.w; return; }
    double th = dProjRadiusInv(cam.projection, rho * cam.rEdge);
    if (th > DPI) th = DPI;
    DVec3 radial = (cam.u * (Real)sx + cam.v * (Real)sy) * (Real)(1.0 / rho);
    rd = normalize(cam.w * (Real)cos(th) + radial * (Real)sin(th));
}

// Surface next-event estimation (port of backward.h neeLight, v1 scope). Uniform
// area-measure connection to each area/sphere/cylinder emitter (device emitterSample-
// Point matches the BDPT device path; unbiased, an independent noise realization vs
// the CPU's sphere-cone / cylinder-arc importance sampling). spot/env/collimated are
// gated to the CPU, so they're skipped here.
__device__ static double bkNeeLight(const DScene& sc, const DHit& h, Real rho,
                                    double invPdfLambda, Real lambda, DRng& rng) {
    double total = 0.0;
    Real f = rho / (Real)DPI;                         // Lambertian BRDF
    for (int k = 0; k < sc.nEmitters; ++k) {
        const DEmitter& em = sc.emitters[k];
        if (em.collimated || em.shape == 2 || em.shape == 3) continue;   // beams/spot/env
        Real u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 y, nL;
        emitterSamplePoint(em, (double)u1, (double)u2, y, nL);
        DVec3 toL = y - h.p;
        Real dist2 = dot(toL, toL);
        Real dist = sqrt(dist2);
        DVec3 wi = toL / dist;
        Real cosSurf = dot(h.n, wi);
        if (cosSurf <= 0) continue;
        Real cosLight = dot(nL, -wi);                 // light is one-sided
        if (cosLight <= 0) continue;
        if (occluded(sc, h.p + h.n * RAY_EPS, wi, dist - (Real)2 * RAY_EPS)) continue;
        Real G = cosSurf * cosLight / dist2;
        double emitW = (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
        double contrib = (double)(f * G) * emitW * (double)em.area;
        if (sc.medium.enabled) contrib *= exp(-(double)medSigmaT(sc.medium, lambda) * (double)dist);
        total += contrib;
    }
    return total;
}

// Estimate spectral-weighted radiance for one wavelength along a camera ray (port of
// backward.h radiance, v1 scope: no fog/env). Emission added only on specular/camera
// arrival; diffuse arrivals are covered by NEE (no double counting).
__device__ static double bkRadiance(const DScene& sc, int diffraction, DVec3 ro, DVec3 rd,
                                    Real lambda, double invPdfLambda, DRng& rng) {
    double L = 0.0, thr = 1.0;
    bool specularArrival = true;                       // camera ray may see a light directly
    int interior = -1;                                 // dielectric the ray is inside (-1 = vacuum)
    const int maxBounce = 32;
    for (int b = 0; b < maxBounce; ++b) {
        DHit h = closestHit(sc, ro, rd);
        if (!h.valid) return L;                        // escaped (no env in v1)
        // Beer-Lambert attenuation over the in-glass segment up to this surface
        // (colored/attenuating glass carried from the previous vertex).
        if (interior >= 0) {
            Real a = (Real)specLookup(sc.mats[interior].absorb, lambda);
            if (a > 0) thr *= exp(-(double)a * (double)h.t);
        }
        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (mp->type == D_MIX) {                       // resolve stochastic mix
            int child = dMixResolveChild(sc, *mp, h, rng.uniform());
            if (child < 0) return L;                    // absorbed
            mp = &sc.mats[child]; matId = child;
        }
        // Emission on specular/camera arrival (NEE covers diffuse arrivals).
        int li = dEmitterForMat(sc, matId);
        if (li >= 0 && specularArrival && dot(rd, h.ng) < 0)
            L += thr * (double)specLookup(sc.emitters[li].emitSpd, lambda) * invPdfLambda;

        switch (mp->type) {
            case D_DIELECTRIC: {
                bool entering = dot(rd, h.ng) < 0;
                bool transmitted = false;
                DVec3 nro, nrd; refractOrReflect(sc, *mp, h, rd, lambda, rng, nro, nrd, &transmitted);
                if (transmitted) interior = entering ? matId : -1;
                ro = nro; rd = nrd; specularArrival = true; break;
            }
            case D_THINFILM: {
                DVec3 nro, nrd;
                if (!thinFilmInterface(sc, *mp, h, rd, lambda, rng, nro, nrd)) return L;
                ro = nro; rd = nrd; specularArrival = true; break;
            }
            case D_MULTILAYER: {
                DVec3 nro, nrd;
                if (!multilayerInterface(*mp, h, rd, lambda, rng, nro, nrd)) return L;
                ro = nro; rd = nrd; specularArrival = true; break;
            }
            case D_MIRROR: {
                Real r = clamp01(specLookup(mp->reflect, lambda));
                if (rng.uniform() >= r) return L;       // RR absorb
                ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); specularArrival = true; break;
            }
            case D_GRATING: {
                Real r = clamp01(specLookup(mp->reflect, lambda));
                if (rng.uniform() >= r) return L;
                DVec3 nro, nrd;
                if (!gratingDiffract(*mp, h, rd, lambda, diffraction, rng, nro, nrd)) return L;
                ro = nro; rd = nrd; specularArrival = true; break;
            }
            case D_HALFMIRROR: {
                Real r = clamp01(specLookup(mp->reflect, lambda));
                if (rng.uniform() < r) { ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); }
                else                   { ro = h.p + rd * RAY_EPS; }
                specularArrival = true; break;
            }
            case D_GLOSSY: {
                Real r = clamp01(specLookup(mp->reflect, lambda));
                if (rng.uniform() >= r) return L;
                DVec3 o = sampleGlossy(reflectv(rd, h.n), dMatRoughness(sc, *mp, h), rng);
                if (dot(o, h.n) <= 0) return L;
                ro = h.p + h.n * RAY_EPS; rd = o; specularArrival = true; break;
            }
            case D_DIFFUSE:
            case D_FLUORESCENT:
            default: {
                Real rho = clamp01(dDiffuseRho(sc, *mp, h, lambda));
                L += thr * bkNeeLight(sc, h, rho, invPdfLambda, lambda, rng);
                if (rng.uniform() >= rho) return L;     // RR on albedo
                DVec3 wOut = cosineHemisphere(h.n, rng);
                ro = h.p + h.n * RAY_EPS; rd = wOut; specularArrival = false; break;
            }
        }
    }
    return L;
}

// Backward reference megakernel (GPU mode R). Grid-strides over res*res*spp samples;
// each samples a wavelength, generates a camera ray (physical lens or pinhole/fisheye),
// estimates radiance, and accumulates cieXYZ * (L * lensWeight) into the film. The film
// holds the SUM over spp (writeFilm divides by spp), matching renderForwardCuda.
__global__ void kBackward(DScene sc, DCamera cam, double* film, double* hits,
                          long long totalSamples, long long spp, int resX,
                          int diffraction, unsigned long long seedBase) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    for (long long idx = g; idx < totalSamples; idx += G) {
        DRng rng; rng.seed((unsigned long long)(idx * 2 + 1), seedBase ^ (unsigned long long)idx);
        long long pix = idx / spp;
        int px = (int)(pix % resX);
        int py = (int)(pix / resX);

        double pdf = 0.0;
        Real lambda = dSampleSceneLambda(sc, rng, pdf);
        if (pdf <= 0.0) continue;
        double invPdfLambda = dInvPdfLambda(sc, lambda);

        DVec3 ro, rd;
        double wLens = 1.0;
        if (cam.hasLens) {
            Real jx = rng.uniform(), jy = rng.uniform();
            Real u1 = rng.uniform(), u2 = rng.uniform();
            Real wl = 0;
            if (!dGenLensRay(cam, px, py, jx, jy, u1, u2, lambda, ro, rd, wl)) continue;  // vignetted
            wLens = (double)wl;
        } else {
            Real jx = rng.uniform(), jy = rng.uniform();
            dGenRay(cam, px, py, jx, jy, ro, rd);
        }
        double Lval = bkRadiance(sc, diffraction, ro, rd, lambda, invPdfLambda, rng);
        double w = Lval * wLens;
        size_t o = ((size_t)py * resX + px) * 3;
        atomicAdd(&film[o + 0], (double)cieX(lambda) * w);
        atomicAdd(&film[o + 1], (double)cieY(lambda) * w);
        atomicAdd(&film[o + 2], (double)cieZ(lambda) * w);
        if (hits) atomicAdd(&hits[(size_t)py * resX + px], 1.0);
    }
}

// Continue a subpath whose endpoint is already path[0]; append surface vertices
// until a miss/absorption/maxDepth. Direct port of bdpt.h randomWalk.
__device__ static void dRandomWalk(const DScene& sc, const DCamera& cam, int diffraction,
                                   DVec3 ro, DVec3 rd, double beta, double pdfDir, Real lambda,
                                   int maxDepth, DRng& rng, DVertex* path, int& n) {
    if (maxDepth == 0) return;
    double pdfFwd = pdfDir;
    for (int bounces = 0;;) {
        DHit h = closestHit(sc, ro, rd);
        if (!h.valid) return;
        if (h.sensorId >= 0) return;

        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (mp->type == D_MIX) {
            Real u = rng.uniform(), acc = 0; int child = -1;
            for (int k = 0; k < mp->mixCount; ++k) { acc += (Real)mp->mixWeight[k]; if (u < acc) { child = mp->mixChild[k]; break; } }
            if (child < 0) return;
            mp = &sc.mats[child]; matId = child;
        }
        if (n >= BDPT_MAXV) return;
        DVertex v;
        v.type = BV_SURFACE; v.p = h.p; v.ns = h.n; v.ng = h.ng;
        v.beta = beta; v.pdfFwd = 0; v.pdfRev = 0; v.delta = 0;
        v.matId = matId; v.lightIdx = dEmitterForMat(sc, matId);
        v.pdfFwd = dConvertDensity(pdfFwd, path[n - 1], v);
        path[n] = v; int cur = n; n++;
        if (++bounces >= maxDepth) return;

        DVec3 wo = normalize(path[cur - 1].p - path[cur].p);
        DVec3 wi; double pdfW = 0, pdfRevW = 0, betaFactor = 0; int delta = 0; bool terminate = false;
        switch (mp->type) {
            case D_DIFFUSE:
            case D_FLUORESCENT: {
                wi = cosineHemisphere(path[cur].ns, rng);
                if (dot(wi, path[cur].ns) <= 0) { terminate = true; break; }
                double rho = clamp01(specLookup(mp->reflect, lambda));
                pdfW = dBsdfPdf(sc, matId, path[cur].ns, wo, wi);
                pdfRevW = dBsdfPdf(sc, matId, path[cur].ns, wi, wo);
                betaFactor = rho;
                if (rho <= 0) terminate = true;
                break;
            }
            case D_GLOSSY: {
                DVec3 mdir = reflectv(rd, path[cur].ns);   // rd == -wo (incoming dir)
                wi = sampleGlossy(mdir, (Real)mp->roughness, rng);
                if (dot(wi, path[cur].ns) <= 0) { terminate = true; break; }
                double r = clamp01(specLookup(mp->reflect, lambda));
                pdfW = dBsdfPdf(sc, matId, path[cur].ns, wo, wi);
                pdfRevW = dBsdfPdf(sc, matId, path[cur].ns, wi, wo);
                betaFactor = r;
                if (r <= 0 || pdfW <= 0) terminate = true;
                break;
            }
            case D_MIRROR: {
                double r = clamp01(specLookup(mp->reflect, lambda));
                wi = reflectv(rd, path[cur].ns); betaFactor = r; delta = 1;
                if (r <= 0) terminate = true;
                break;
            }
            case D_DIELECTRIC: {
                DVec3 nro, nrd; refractOrReflect(sc, *mp, h, rd, lambda, rng, nro, nrd);
                wi = nrd; betaFactor = 1.0; delta = 1;
                break;
            }
            case D_HALFMIRROR: {
                double r = clamp01(specLookup(mp->reflect, lambda));
                if (rng.uniform() < r) wi = reflectv(rd, path[cur].ns); else wi = rd;
                betaFactor = 1.0; delta = 1;
                break;
            }
            case D_THINFILM: {
                DVec3 nro, nrd;
                if (!thinFilmInterface(sc, *mp, h, rd, lambda, rng, nro, nrd)) { terminate = true; break; }
                wi = nrd; betaFactor = 1.0; delta = 1;
                break;
            }
            case D_MULTILAYER: {
                DVec3 nro, nrd;
                if (!multilayerInterface(*mp, h, rd, lambda, rng, nro, nrd)) { terminate = true; break; }
                wi = nrd; betaFactor = 1.0; delta = 1;
                break;
            }
            case D_GRATING: {
                double r = clamp01(specLookup(mp->reflect, lambda));
                if (r <= 0) { terminate = true; break; }
                DVec3 nro, nrd;
                if (!gratingDiffract(*mp, h, rd, lambda, diffraction, rng, nro, nrd)) { terminate = true; break; }
                wi = nrd; betaFactor = r; delta = 1;
                break;
            }
            default: terminate = true; break;
        }
        if (terminate || betaFactor <= 0.0) return;

        path[cur].delta = delta;
        if (delta) { pdfW = 0.0; pdfRevW = 0.0; }
        path[cur - 1].pdfRev = dConvertDensity(pdfRevW, path[cur], path[cur - 1]);

        beta *= betaFactor;
        double sgn = dot(wi, path[cur].ng) >= 0.0 ? 1.0 : -1.0;
        ro = path[cur].p + path[cur].ng * (Real)(sgn * 1e-6);
        rd = normalize(wi);
        pdfFwd = delta ? 0.0 : pdfW;
    }
}

// Trace an eye subpath through pixel (px,py). path[0] is the camera vertex (beta=1).
// Realistic-lens cameras (cam.hasLens, Plan B): the first ray is traced through the real
// glass (dGenLensRay, as GPU mode R does), the camera vertex sits at the ray's scene-entry
// point with beta = the lens weight wLens, and is flagged delta. The multi-element lens map
// has no closed-form inverse, so the light-image splat (t=1) is disabled in dConnectBDPT and
// the delta flag makes dMisWeight omit that strategy too — the surviving strategies (s>=0,
// t>=2) still partition unity, so the estimator stays unbiased. The camera vertex being
// delta means its direction pdf only enters the excluded t=1 term, so the placeholder
// dCameraPdfDir seed is never used in a retained MIS ratio. Mirrors bdpt.h.
__device__ static int dGenCameraSubpath(const DScene& sc, const DCamera& cam, int diffraction,
                                        int px, int py, Real lambda, int maxDepth,
                                        DRng& rng, DVertex* path) {
    DVertex c;
    c.type = BV_CAMERA; c.ns = cam.w; c.ng = cam.w;
    c.beta = 1.0; c.pdfFwd = 0; c.pdfRev = 0; c.delta = 0; c.matId = -1; c.lightIdx = -1;
    if (cam.hasLens) {
        Real jx = rng.uniform(), jy = rng.uniform();
        Real u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 ro, rd; Real wl = 0;
        if (!dGenLensRay(cam, px, py, jx, jy, u1, u2, lambda, ro, rd, wl) || wl <= 0) {
            // Vignetted: lone delta camera vertex (nE=1) contributes 0 (t=1 off, t>=2 needs a
            // scene vertex we never added).
            c.p = cam.eye; c.delta = 1;
            path[0] = c; return 1;
        }
        c.p = ro;                // scene-entry point (front element plane): correct wo / dist
        c.beta = (double)wl;     // radiometric lens weight -> per-pixel measurement
        c.delta = 1;             // no closed-form lens inverse: not connectible (t=1 off)
        path[0] = c; int n = 1;
        double pdfDir = dCameraPdfDir(cam, ddot(rd, cam.w));   // MIS-irrelevant placeholder
        dRandomWalk(sc, cam, diffraction, ro, rd, (double)wl, pdfDir, lambda, maxDepth - 1, rng, path, n);
        return n;
    }
    c.p = cam.eye;
    path[0] = c; int n = 1;
    Real jx = rng.uniform(), jy = rng.uniform();
    Real sx = (Real)2 * (((Real)px + jx) / (Real)cam.resX) - (Real)1;
    Real sy = (Real)2 * (((Real)py + jy) / (Real)cam.resY) - (Real)1;
    DVec3 rd = normalize(cam.w + cam.u * (sx * (Real)cam.tanHalfX) + cam.v * (sy * (Real)cam.tanHalfY));
    double cosCam = ddot(rd, cam.w);
    double pdfDir = dCameraPdfDir(cam, cosCam);
    dRandomWalk(sc, cam, diffraction, cam.eye, rd, 1.0, pdfDir, lambda, maxDepth - 1, rng, path, n);
    return n;
}
// Sample a light subpath. path[0] is the light endpoint (beta = Le).
__device__ static int dGenLightSubpath(const DScene& sc, const DCamera& cam, int diffraction,
                                       Real lambda, double invPdfLambda, int maxDepth,
                                       DRng& rng, DVertex* path) {
    if (sc.nEmitters == 0 || sc.totalPower <= 0.0) return 0;
    int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
    const DEmitter& em = sc.emitters[ei];
    if (em.shape == 2 || em.shape == 3 || em.collimated) return 0;
    Real u1 = rng.uniform(), u2 = rng.uniform();
    DVec3 y, nOut; emitterSamplePoint(em, u1, u2, y, nOut);
    double Le = (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
    if (Le <= 0.0) return 0;
    double pdfChoice = em.power / sc.totalPower;
    double pdfPos = (em.area > 0.0) ? 1.0 / em.area : 0.0;
    if (pdfPos <= 0.0) return 0;
    DVertex L0;
    L0.type = BV_LIGHT; L0.p = y; L0.ns = nOut; L0.ng = nOut;
    L0.beta = Le; L0.pdfFwd = pdfChoice * pdfPos; L0.pdfRev = 0; L0.delta = 0;
    L0.matId = em.matId; L0.lightIdx = ei;
    path[0] = L0; int n = 1;
    DVec3 dir = cosineHemisphere(nOut, rng);
    double cosLight = ddot(nOut, dir);
    if (cosLight <= 0.0) return 1;
    double pdfDir = cosLight / DPI;
    double betaWalk = Le * cosLight / (pdfChoice * pdfPos * pdfDir);
    DVec3 ro = y + nOut * (Real)1e-6;
    dRandomWalk(sc, cam, diffraction, ro, dir, betaWalk, pdfDir, lambda, maxDepth - 1, rng, path, n);
    return n;
}

// Balance-heuristic MIS weight for strategy (s,t). Direct port of bdpt.h misWeight:
// PBRT temporarily rewrites the connection vertices' reverse densities / delta flags
// and (for s==1/t==1) installs the resampled endpoint, sums the density ratios of all
// other strategies, then rolls the mutations back. Here the vertices live in the
// per-thread local arrays, so we save whole vertices before ANY mutation and restore
// them at the end (whole-vertex restore subsumes PBRT's field-wise ScopedAssignments).
__device__ static double dMisWeight(const DScene& sc, const DCamera& cam,
                                    DVertex* light, DVertex* eye, const DVertex& sampled,
                                    int s, int t) {
    if (s + t == 2) return 1.0;
    int si = s - 1, ti = t - 1, sMi = s - 2, tMi = t - 2;
    bool hasQs = s > 0, hasPt = t > 0, hasQsM = s > 1, hasPtM = t > 1;

    DVertex sQs, sPt, sQsM, sPtM;
    if (hasQs)  sQs  = light[si];
    if (hasPt)  sPt  = eye[ti];
    if (hasQsM) sQsM = light[sMi];
    if (hasPtM) sPtM = eye[tMi];

    // a1: install the resampled endpoint for s==1 / t==1.
    if (s == 1)      light[si] = sampled;
    else if (t == 1) eye[ti]   = sampled;
    // a2/a3: connection endpoints act as non-delta while probing hypotheticals.
    if (hasPt) eye[ti].delta   = 0;
    if (hasQs) light[si].delta = 0;
    // a4: reverse density of the eye connection vertex pt.
    if (hasPt) {
        double val = (s > 0) ? dVertexPdf(sc, cam, hasQsM ? &light[sMi] : nullptr, light[si], eye[ti])
                             : dVertexPdfLightOrigin(sc, eye[ti]);
        eye[ti].pdfRev = val;
    }
    // a5: reverse density of pt's predecessor.
    if (hasPtM) {
        double val = (s > 0) ? dVertexPdf(sc, cam, hasQs ? &light[si] : nullptr, eye[ti], eye[tMi])
                             : dVertexPdfLight(eye[ti], eye[tMi]);
        eye[tMi].pdfRev = val;
    }
    // a6/a7: reverse density of the light connection vertex qs and its predecessor.
    if (hasQs)  light[si].pdfRev  = dVertexPdf(sc, cam, hasPtM ? &eye[tMi] : nullptr, eye[ti], light[si]);
    if (hasQsM) light[sMi].pdfRev = dVertexPdf(sc, cam, hasPt ? &eye[ti] : nullptr, light[si], light[sMi]);

    double sumRi = 0.0, ri = 1.0;
    for (int i = t - 1; i > 0; --i) {
        double num = eye[i].pdfRev != 0.0 ? eye[i].pdfRev : 1.0;
        double den = eye[i].pdfFwd != 0.0 ? eye[i].pdfFwd : 1.0;
        ri *= num / den;
        if (!eye[i].delta && !eye[i - 1].delta) sumRi += ri;
    }
    ri = 1.0;
    for (int i = s - 1; i >= 0; --i) {
        double num = light[i].pdfRev != 0.0 ? light[i].pdfRev : 1.0;
        double den = light[i].pdfFwd != 0.0 ? light[i].pdfFwd : 1.0;
        ri *= num / den;
        bool deltaPrev = (i > 0) ? (light[i - 1].delta != 0) : false;
        if (!light[i].delta && !deltaPrev) sumRi += ri;
    }

    if (hasQsM) light[sMi] = sQsM;
    if (hasPtM) eye[tMi]   = sPtM;
    if (hasQs)  light[si]  = sQs;
    if (hasPt)  eye[ti]    = sPt;
    return 1.0 / (1.0 + sumRi);
}

// Connect strategy (s,t); returns the MIS-weighted radiance. For t==1 the result is a
// light-image splat to (outPx,outPy) with isSplat=1. Direct port of bdpt.h connectBDPT.
__device__ static double dConnectBDPT(const DScene& sc, const DCamera& cam,
                                      DVertex* light, DVertex* eye, int s, int t,
                                      Real lambda, double invPdfLambda, DRng& rng,
                                      int& outPx, int& outPy, int& isSplat) {
    isSplat = 0;
    if (t > 1 && s != 0 && dIsLightVertex(eye[t - 1])) return 0.0;

    double L = 0.0;
    DVertex sampled;
    sampled.type = BV_SURFACE; sampled.beta = 0; sampled.pdfFwd = 0; sampled.pdfRev = 0;
    sampled.delta = 0; sampled.matId = -1; sampled.lightIdx = -1;

    if (s == 0) {
        if (t < 2) return 0.0;
        const DVertex& pt = eye[t - 1];
        if (!dIsLightVertex(pt)) return 0.0;
        DVec3 wo = normalize(eye[t - 2].p - pt.p);
        double Le = dVertexLe(sc, pt, wo, lambda, invPdfLambda);
        if (Le <= 0.0) return 0.0;
        L = pt.beta * Le;
    } else if (t == 1) {
        // Realistic lens (Plan B): the light-image splat needs a world->sensor projection
        // the multi-element lens map can't provide (no closed-form inverse), so it's
        // disabled. dMisWeight omits this strategy too (camera vertex is delta), so the
        // retained strategies still partition unity.
        if (cam.hasLens) return 0.0;
        const DVertex& qs = light[s - 1];
        if (!dVertConnectible(sc, qs)) return 0.0;
        int px, py; Real cc, d2f;
        if (!cam.project(qs.p, px, py, cc, d2f)) return 0.0;
        double dist2 = ddot(cam.eye - qs.p, cam.eye - qs.p);
        double dist = sqrt(dist2);
        DVec3 wcam = (cam.eye - qs.p) * (Real)(1.0 / dist);
        double cosSurf = ddot(qs.ns, wcam);
        if (cosSurf <= 0.0) return 0.0;
        DVec3 wo = normalize(light[s - 2].p - qs.p);
        double f = dBsdfF(sc, qs.matId, qs.ns, wo, wcam, lambda);
        if (f <= 0.0) return 0.0;
        double sgn = ddot(qs.ng, wcam) >= 0.0 ? 1.0 : -1.0;
        DVec3 o = qs.p + qs.ng * (Real)(sgn * 1e-6);
        if (occluded(sc, o, wcam, (Real)(dist - 2e-6))) return 0.0;
        double cosCam = ddot(qs.p - cam.eye, cam.w) / dist;   // positive (point in front)
        double G = cosSurf * cosCam / dist2;
        L = qs.beta * f * G * dCameraWe(cam, cosCam);
        if (L <= 0.0) return 0.0;
        sampled.type = BV_CAMERA; sampled.p = cam.eye; sampled.ns = cam.w; sampled.ng = cam.w;
        sampled.beta = 1.0;
        outPx = px; outPy = py; isSplat = 1;
    } else if (s == 1) {
        const DVertex& pt = eye[t - 1];
        if (!dVertConnectible(sc, pt)) return 0.0;
        int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
        const DEmitter& em = sc.emitters[ei];
        if (em.shape == 2 || em.shape == 3 || em.collimated) return 0.0;
        Real u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 y, nOut; emitterSamplePoint(em, u1, u2, y, nOut);
        DVec3 toL = y - pt.p; double dist2 = ddot(toL, toL);
        if (dist2 <= 0.0) return 0.0;
        double dist = sqrt(dist2); DVec3 wi = toL * (Real)(1.0 / dist);
        double cosLight = ddot(nOut, wi * (Real)-1);
        double cosSurf = ddot(pt.ns, wi);
        if (cosLight <= 0.0 || cosSurf <= 0.0) return 0.0;
        double Le = (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
        if (Le <= 0.0) return 0.0;
        double sgn = ddot(pt.ng, wi) >= 0.0 ? 1.0 : -1.0;
        DVec3 o = pt.p + pt.ng * (Real)(sgn * 1e-6);
        if (occluded(sc, o, wi, (Real)(dist - 2e-6))) return 0.0;
        DVec3 wo = normalize(eye[t - 2].p - pt.p);
        double f = dBsdfF(sc, pt.matId, pt.ns, wo, wi, lambda);
        if (f <= 0.0) return 0.0;
        double pdfChoice = em.power / sc.totalPower;
        double pdfA = pdfChoice / em.area;
        if (pdfA <= 0.0) return 0.0;
        double G = cosSurf * cosLight / dist2;
        L = pt.beta * f * Le * G / pdfA;
        if (L <= 0.0) return 0.0;
        sampled.type = BV_LIGHT; sampled.p = y; sampled.ns = nOut; sampled.ng = nOut;
        sampled.lightIdx = ei; sampled.matId = em.matId; sampled.beta = Le / pdfA; sampled.pdfFwd = pdfA;
    } else {
        const DVertex& qs = light[s - 1];
        const DVertex& pt = eye[t - 1];
        if (!dVertConnectible(sc, qs) || !dVertConnectible(sc, pt)) return 0.0;
        DVec3 d = qs.p - pt.p; double dist2 = ddot(d, d);
        if (dist2 <= 0.0) return 0.0;
        double dist = sqrt(dist2); DVec3 w = d * (Real)(1.0 / dist);
        double cosE = ddot(pt.ns, w), cosL = ddot(qs.ns, w * (Real)-1);
        if (cosE <= 0.0 || cosL <= 0.0) return 0.0;
        DVec3 woE = normalize(eye[t - 2].p - pt.p);
        DVec3 woL = normalize(light[s - 2].p - qs.p);
        double fE = dBsdfF(sc, pt.matId, pt.ns, woE, w, lambda);
        double fL = dBsdfF(sc, qs.matId, qs.ns, woL, w * (Real)-1, lambda);
        if (fE <= 0.0 || fL <= 0.0) return 0.0;
        double sgn = ddot(pt.ng, w) >= 0.0 ? 1.0 : -1.0;
        DVec3 o = pt.p + pt.ng * (Real)(sgn * 1e-6);
        if (occluded(sc, o, w, (Real)(dist - 2e-6))) return 0.0;
        double G = cosE * cosL / dist2;
        L = pt.beta * fE * fL * qs.beta * G;
    }
    if (L <= 0.0) return 0.0;
    return L * dMisWeight(sc, cam, light, eye, sampled, s, t);
}

// BDPT megakernel: one thread renders one (pixel,sample), grid-stride over all
// res*res*spp samples. t>=2 connections land on the sample's own pixel (camFilm);
// t==1 splats land on the projected raster pixel (splatFilm). Both are normalised by
// 1/spp on the host (bdpt.h renderBdpt convention).
__global__ void kBdpt(DScene sc, DCamera cam, double* camFilm, double* splatFilm,
                      long long totalSamples, long long spp, int resX, int maxDepth,
                      int diffraction, unsigned long long seedBase) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    for (long long idx = g; idx < totalSamples; idx += G) {
        DRng rng; rng.seed((unsigned long long)(idx * 2 + 1), seedBase ^ (unsigned long long)idx);
        long long pix = idx / spp;
        int px = (int)(pix % resX);
        int py = (int)(pix / resX);

        double pdfLam = 0.0;
        Real lambda = dSampleSceneLambda(sc, rng, pdfLam);
        if (pdfLam <= 0.0) continue;
        double invPdfLambda = dInvPdfLambda(sc, lambda);

        DVertex eye[BDPT_MAXV], light[BDPT_MAXV];
        int nE = dGenCameraSubpath(sc, cam, diffraction, px, py, lambda, maxDepth + 1, rng, eye);
        int nL = dGenLightSubpath(sc, cam, diffraction, lambda, invPdfLambda, maxDepth + 1, rng, light);

        Real cx = cieX(lambda), cy = cieY(lambda), cz = cieZ(lambda);
        for (int t = 1; t <= nE; ++t)
            for (int s = 0; s <= nL; ++s) {
                int depth = t + s - 2;
                if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth) continue;
                int spx = 0, spy = 0, isSplat = 0;
                double c = dConnectBDPT(sc, cam, light, eye, s, t, lambda, invPdfLambda, rng, spx, spy, isSplat);
                if (c <= 0.0) continue;
                if (isSplat) {
                    size_t o = ((size_t)spy * resX + spx) * 3;
                    atomicAdd(&splatFilm[o + 0], (double)(cx * c));
                    atomicAdd(&splatFilm[o + 1], (double)(cy * c));
                    atomicAdd(&splatFilm[o + 2], (double)(cz * c));
                } else {
                    size_t o = ((size_t)py * resX + px) * 3;
                    atomicAdd(&camFilm[o + 0], (double)(cx * c));
                    atomicAdd(&camFilm[o + 1], (double)(cy * c));
                    atomicAdd(&camFilm[o + 2], (double)(cz * c));
                }
            }
    }
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
    // Implicit surfaces (isosurface / CSG / metaballs) are now sphere-traced on the
    // device too (DImplicit + intersectImplicit); their materials are checked by the
    // same `unsupported()` gate as tri/sphere materials below.
    // The device multilayer stack has a fixed cap (D_MAXLAYERS); scenes with a
    // deeper stack fall back to the CPU tracer, which has no layer limit. (Textured
    // albedo and fluorescence are now BOTH ported to the device: per-texel Jakob-
    // Hanika coeff tables + per-tri UVs feed dTexReflAt, and fluorescent materials
    // carry a baked excitation spectrum + emission-SPD CDF the shadeStep fluoro
    // branch samples for the Stokes shift — so neither forces a CPU fallback here.)
    auto oversizedMultilayer = [&](int matId) {
        return matId >= 0 && matId < (int)scene.mats.size() &&
               scene.mats[matId].type == MatType::Multilayer &&
               (int)scene.mats[matId].layerN.size() > D_MAXLAYERS;
    };
    // Indexed-spectral palette maps (§9.3) resolve per-texel to an arbitrary named
    // reflectance spectrum; the device only bakes the JH-upsampled coeff path, so a
    // palette-bound albedo forces the CPU tracer (which evaluates the palette exactly).
    auto paletteTex = [&](int t) {
        return t >= 0 && t < (int)scene.textures.size() && scene.textures[t].hasPalette();
    };
    auto usesPaletteTex = [&](int matId) {
        if (matId < 0 || matId >= (int)scene.mats.size()) return false;
        const Material& m = scene.mats[matId];
        if (paletteTex(m.reflectTex)) return true;
        if (m.type == MatType::Mix)
            for (int c : m.mixChildren)
                if (c >= 0 && c < (int)scene.mats.size() && paletteTex(scene.mats[c].reflectTex)) return true;
        return false;
    };
    // Dielectric translucency now runs on the device forward + backward tracers:
    // frosting (a roughness lobe on both dielectric lobes, from constant/tex/pattern
    // roughness) and Beer-Lambert interior absorption (colored glass) are both threaded
    // through shadeStep / bkRadiance via the `interior` medium index. Procedural patterns
    // (§4) also run on-device (dPatternEval / dMatRoughness / dMixResolveChild). So neither
    // frosted/colored glass nor a roughness/film/mix-weight pattern forces a CPU forward
    // fallback here. (The GPU BDPT kernel still can't MIS either — cudaBdptSupported gates
    // both.) Implicit surfaces (isosurface) are gated separately below.
    auto unsupported = [&](int matId) {
        if (oversizedMultilayer(matId)) return true;
        if (usesPaletteTex(matId)) return true;
        // The physical layered stack (coat interface over a weighted body) is CPU-only;
        // the device shadeStep has no Layered branch, so any Layered material forces a
        // CPU forward/backward fallback (like indexed palettes).
        if (matId >= 0 && matId < (int)scene.mats.size() &&
            scene.mats[matId].type == MatType::Layered) return true;
        if (matId >= 0 && matId < (int)scene.mats.size() &&
            scene.mats[matId].type == MatType::Mix) {
            const Material& mx = scene.mats[matId];
            if ((int)mx.mixChildren.size() > D_MIXMAX) return true;
            for (int c : mx.mixChildren) if (oversizedMultilayer(c)) return true;
        }
        return false;
    };
    for (const auto& t : scene.tris)      if (unsupported(t.matId)) return false;
    for (const auto& s : scene.spheres)   if (unsupported(s.matId)) return false;
    for (const auto& im : scene.implicits) if (unsupported(im.matId)) return false;
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
static void buildUpload(const Scene& scene, const Camera& cam, int resX, int resY, DUpload& up) {
    using namespace gpu;
    auto keep = [&](void* p) { if (p) up.frees.push_back(p); return p; };

    // --- bake geometry ---
    std::vector<DTri> tris(scene.tris.size());
    for (size_t i = 0; i < scene.tris.size(); ++i) {
        const Tri& t = scene.tris[i]; DTri& d = tris[i];
        d.v0 = {t.v0.x, t.v0.y, t.v0.z}; d.v1 = {t.v1.x, t.v1.y, t.v1.z};
        d.v2 = {t.v2.x, t.v2.y, t.v2.z}; d.gn = {t.gn.x, t.gn.y, t.gn.z};
        d.uv0 = {t.uv0.x, t.uv0.y, t.uv0.z};
        d.uv1 = {t.uv1.x, t.uv1.y, t.uv1.z};
        d.uv2 = {t.uv2.x, t.uv2.y, t.uv2.z};
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

    // --- bake implicit surfaces (isosurface / CSG / metaballs) ---
    // Flatten every Implicit's postfix FieldNode array into one pool; each DImplicit
    // slices it by [nodeOff, nodeOff+nodeN). BVH prims >= nTris+nSph index these.
    std::vector<DFieldNode> fieldNodes;
    std::vector<PatNode>    fieldExprNodes;   // flat pool for DF_EXPR formulas (all implicits)
    std::vector<DImplicit>  dimpl(scene.implicits.size());
    for (size_t i = 0; i < scene.implicits.size(); ++i) {
        const Implicit& im = scene.implicits[i]; DImplicit& d = dimpl[i];
        d.nodeOff = (int)fieldNodes.size();
        d.nodeN   = (int)im.nodes.size();
        d.matId   = im.matId;
        d.lo[0] = im.bounds.lo.x; d.lo[1] = im.bounds.lo.y; d.lo[2] = im.bounds.lo.z;
        d.hi[0] = im.bounds.hi.x; d.hi[1] = im.bounds.hi.y; d.hi[2] = im.bounds.hi.z;
        d.lipschitz = im.lipschitz; d.minStep = im.minStep;
        d.method = (int)im.method; d.refine = (int)im.refine; d.sampleStep = im.sampleStep;
        // Rebase this implicit's expr programs into the shared device pool. Each Implicit
        // owns a private exprNodes vector on the host (FieldNode.exprOff indexes it), so we
        // add the running base and copy the programs into fieldExprNodes.
        int exprBase = (int)fieldExprNodes.size();
        fieldExprNodes.insert(fieldExprNodes.end(), im.exprNodes.begin(), im.exprNodes.end());
        for (const FieldNode& fn : im.nodes) {
            DFieldNode dn;
            dn.op = (int)fn.op;
            dn.p[0] = fn.p[0]; dn.p[1] = fn.p[1]; dn.p[2] = fn.p[2]; dn.p[3] = fn.p[3];
            for (int k = 0; k < 9; ++k) dn.inv[k] = fn.inv.m[k];
            dn.tx = fn.inv.t.x; dn.ty = fn.inv.t.y; dn.tz = fn.inv.t.z;
            dn.scale = fn.scale;
            dn.exprOff = (fn.op == FieldOp::Expr) ? exprBase + fn.exprOff : -1;
            dn.exprN   = (fn.op == FieldOp::Expr) ? fn.exprN : 0;
            fieldNodes.push_back(dn);
        }
    }

    // --- bake procedural patterns (§4) ---
    // Flatten every Pattern's postfix PatNode program into one pool; each DPattern
    // slices it by [off, off+n). Materials index these via roughnessPat/etc.
    std::vector<PatNode> patNodes;
    std::vector<DPattern> dpat(scene.patterns.size());
    for (size_t i = 0; i < scene.patterns.size(); ++i) {
        const Pattern& p = scene.patterns[i]; DPattern& d = dpat[i];
        d.off = (int)patNodes.size();
        d.n   = (int)p.nodes.size();
        patNodes.insert(patNodes.end(), p.nodes.begin(), p.nodes.end());
    }

    // --- bake materials ---
    // Fluorescent materials append their emission-SPD CDF to one flat buffer
    // (fluoCdfAll), sliced per material by fluoCdfOffset/fluoCdfN (like lightCdfAll).
    std::vector<DMaterial> mats(scene.mats.size());
    std::vector<double> fluoCdfAll;
    for (size_t i = 0; i < scene.mats.size(); ++i) {
        const Material& m = scene.mats[i]; DMaterial& d = mats[i];
        d.type = (int)m.type;
        bakeSpec(m.reflect, d.reflect);
        bakeSpec(m.ior, d.ior);
        bakeSpec(m.substrateK, d.substrateK);
        bakeSpec(m.absorb, d.absorb);   // Beer-Lambert interior tint (colored glass)
        d.reflectTex = m.reflectTex;
        d.triplanarScale = m.triplanarScale;
        // Fluorescence tables (zero/inert for every non-fluorescent material).
        bakeSpec(m.fluoAbsorb, d.fluoAbsorb);
        d.fluoYield = m.fluoYield;
        if (m.type == MatType::Fluorescent && !m.fluoEmitSampler.cdf.empty() &&
            m.fluoEmitSampler.integral > 0.0) {
            d.fluoCdfOffset = (int)fluoCdfAll.size();
            d.fluoCdfN = (int)m.fluoEmitSampler.cdf.size();
            d.fluoCdfStep = m.fluoEmitSampler.step;
            fluoCdfAll.insert(fluoCdfAll.end(), m.fluoEmitSampler.cdf.begin(),
                              m.fluoEmitSampler.cdf.end());
        } else {
            d.fluoCdfOffset = 0; d.fluoCdfN = 0; d.fluoCdfStep = 1.0;
        }
        d.roughness = m.roughness;
        d.filmIor = m.filmIor; d.filmThickness = m.filmThickness;
        d.roughnessTex = m.roughnessTex;
        d.filmThicknessTex = m.filmThicknessTex;
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
        d.mixWeightTex = m.mixWeightTex;
        d.roughnessPat = m.roughnessPat;
        d.filmThicknessPat = m.filmThicknessPat;
        d.mixWeightPat = m.mixWeightPat;
    }

    // --- upload geometry/materials ---
    DTri*      d_tris  = tris.empty()    ? nullptr : (DTri*)keep(uploadVec(tris));
    DSphere*   d_sph   = sph.empty()     ? nullptr : (DSphere*)keep(uploadVec(sph));
    DNode*     d_nodes = nodes.empty()   ? nullptr : (DNode*)keep(uploadVec(nodes));
    int*       d_prim  = primIdx.empty() ? nullptr : (int*)keep(uploadVec(primIdx));
    DMaterial* d_mats  = mats.empty()    ? nullptr : (DMaterial*)keep(uploadVec(mats));
    DFieldNode* d_fnodes = fieldNodes.empty() ? nullptr : (DFieldNode*)keep(uploadVec(fieldNodes));
    PatNode*    d_fexpr  = fieldExprNodes.empty() ? nullptr : (PatNode*)keep(uploadVec(fieldExprNodes));
    DImplicit*  d_impl   = dimpl.empty()      ? nullptr : (DImplicit*)keep(uploadVec(dimpl));
    PatNode*    d_pnodes = patNodes.empty()   ? nullptr : (PatNode*)keep(uploadVec(patNodes));
    DPattern*   d_pat    = dpat.empty()       ? nullptr : (DPattern*)keep(uploadVec(dpat));

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
        de.shape = (e.shape == EmitterShape::Sphere)   ? 1
                 : (e.shape == EmitterShape::Spot)     ? 2
                 : (e.shape == EmitterShape::Env)      ? 3
                 : (e.shape == EmitterShape::Cylinder) ? 4 : 0;
        de.radius = e.radius;
        de.caps = e.caps ? 1 : 0;
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

    // --- reflectance textures (per-texel Jakob-Hanika coefficients) ---
    std::vector<DTexture> dtex;
    for (const auto& tx : scene.textures) {
        DTexture dt;
        dt.w = tx.w; dt.h = tx.h;
        dt.wrap   = (tx.wrap   == TexWrap::Clamp)   ? 1 : (tx.wrap == TexWrap::Mirror) ? 2 : 0;
        dt.filter = (tx.filter == TexFilter::Nearest) ? 0 : 1;
        if (!tx.coeff.empty()) {
            std::vector<double> flat(tx.coeff.size() * 3);
            for (size_t i = 0; i < tx.coeff.size(); ++i) {
                flat[3 * i + 0] = tx.coeff[i][0];
                flat[3 * i + 1] = tx.coeff[i][1];
                flat[3 * i + 2] = tx.coeff[i][2];
            }
            dt.coeff = (double*)keep(uploadVec(flat));
        } else {
            dt.coeff = nullptr;
        }
        // Per-texel grayscale (mean of the linear RGB) for scalar (non-albedo) maps.
        // Bilerp of the means equals the mean of the bilerp, so this matches the host
        // Texture::scalarAt exactly. Always uploaded (rgb is populated after load).
        if (!tx.rgb.empty()) {
            std::vector<double> g(tx.rgb.size());
            for (size_t i = 0; i < tx.rgb.size(); ++i)
                g[i] = (tx.rgb[i].x + tx.rgb[i].y + tx.rgb[i].z) * (1.0 / 3.0);
            dt.gray = (double*)keep(uploadVec(g));
        } else {
            dt.gray = nullptr;
        }
        dtex.push_back(dt);
    }
    DTexture* d_tex     = dtex.empty()       ? nullptr : (DTexture*)keep(uploadVec(dtex));
    double*   d_fluoCdf = fluoCdfAll.empty() ? nullptr : (double*)keep(uploadVec(fluoCdfAll));

    DScene& sc = up.sc;
    sc.tris = d_tris; sc.nTris = (int)tris.size();
    sc.sph = d_sph;   sc.nSph = (int)sph.size();
    sc.mats = d_mats;
    sc.nodes = d_nodes; sc.primIdx = d_prim; sc.nNodes = (int)nodes.size();
    sc.fieldNodes = d_fnodes; sc.fieldExprNodes = d_fexpr;
    sc.implicits = d_impl; sc.nImplicits = (int)dimpl.size();
    sc.patNodes = d_pnodes; sc.patterns = d_pat; sc.nPatterns = (int)dpat.size();
    sc.emitters = d_ems; sc.nEmitters = (int)dems.size();
    sc.emitCdf = d_emitCdf; sc.totalPower = scene.totalPower;
    sc.lightCdfAll = d_cdfAll;
    sc.textures = d_tex; sc.nTex = (int)dtex.size();
    sc.fluoCdfAll = d_fluoCdf;
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
    dc.resX = resX; dc.resY = resY;
    dc.apertureR = cam.apertureR; dc.filmDist = cam.filmDist; dc.lensF = cam.lensF;
    dc.projection = cam.projection; dc.halfFovY = cam.halfFovY; dc.rEdge = cam.rEdge;

    // Physical multi-element lens (mesh-lens camera). Bake each surface's sensor-side
    // index into an SPEC_N table (air => 1) so the std::function Spectrum stays host-
    // side. Used by the backward tracer (GPU mode R) and the BDPT camera subpath (mode D,
    // Plan B); the pinhole forward kernels ignore it.
    dc.hasLens = 0;
    dc.lens.iorAll = nullptr;
    if (cam.hasLens()) {
        const LensSystem& L = *cam.lens;
        int M = (int)L.surf.size();
        if (M > 0 && M <= D_MAXLENS) {
            dc.hasLens = 1;
            dc.lens.nSurf = M;
            dc.lens.filmW_mm = L.filmW_mm; dc.lens.filmH_mm = L.filmH_mm;
            dc.lens.T = L.T; dc.lens.filmZ = L.filmZ;
            std::vector<double> iorAll((size_t)M * SPEC_N);
            for (int j = 0; j < M; ++j) {
                dc.lens.surf[j].radius    = L.surf[j].radius;
                dc.lens.surf[j].thickness = L.surf[j].thickness;
                dc.lens.surf[j].aperture  = L.surf[j].aperture;
                dc.lens.surf[j].zpos      = L.zpos[j];
                dc.lens.surf[j].isStop    = L.surf[j].isStop ? 1 : 0;
                for (int i = 0; i < SPEC_N; ++i) {
                    double w = DLMIN + (double)i / (SPEC_N - 1) * (DLMAX - DLMIN);
                    iorAll[(size_t)j * SPEC_N + i] = L.surf[j].ior ? L.surf[j].ior(w) : 1.0;
                }
            }
            dc.lens.iorAll = (const double*)keep(uploadVec(iorAll));
        }
    }
}

// Host driver for the wavefront backend. Allocates the SoA photon pool, seeds it, then
// runs extend/shade passes until every slot has drained the N-photon budget. Writes into
// the same d_film / d_hits / d_energy buffers as the megakernel path.
static void wavefrontTrace(DUpload& up, double* d_film, double* d_hits, double* d_energy,
                           long long N, int diffraction, unsigned long long kseed,
                           int maxBounce, int camModeInt) {
    using namespace gpu;
    if (N <= 0) return;
    int W = (int)((N < (1 << 20)) ? N : (1 << 20));   // persistent slot count
    if (W < 1) W = 1;

    WFState st;
    cudaMalloc(&st.ro,     (size_t)W * sizeof(DVec3));
    cudaMalloc(&st.rd,     (size_t)W * sizeof(DVec3));
    cudaMalloc(&st.beta,   (size_t)W * sizeof(Real));
    cudaMalloc(&st.lambda, (size_t)W * sizeof(Real));
    cudaMalloc(&st.rng,    (size_t)W * sizeof(DRng));
    cudaMalloc(&st.bounce, (size_t)W * sizeof(int));
    cudaMalloc(&st.alive,  (size_t)W * sizeof(int));
    cudaMalloc(&st.interior, (size_t)W * sizeof(int));
    cudaMalloc(&st.hit,    (size_t)W * sizeof(DHit));
    cudaMemset(st.alive, 0, (size_t)W * sizeof(int));

    unsigned long long* d_dispatched = nullptr;
    cudaMalloc(&d_dispatched, sizeof(unsigned long long));
    cudaMemset(d_dispatched, 0, sizeof(unsigned long long));
    int* d_live = nullptr;
    cudaMalloc(&d_live, sizeof(int));
    cudaMemset(d_live, 0, sizeof(int));

    int bs = 128;
    int gb = (W + bs - 1) / bs;

    kWfInit<<<gb, bs>>>(up.sc, up.dc, d_film, d_hits, d_energy, st, N, W,
                        d_dispatched, d_live, kseed, camModeInt);

    // Guard the pass loop against an unexpected non-terminating condition: the longest a
    // slot can stay busy is one path (<= maxBounce shades) before it must regenerate or
    // die, so once dispatched >= N every slot drains within maxBounce passes. This cap is
    // generous slack over that bound and never triggers in normal operation.
    long long maxPasses = (N / W + 2) * (long long)(maxBounce + 1) + 16;
    for (long long pass = 0; pass < maxPasses; ++pass) {
        kWfExtend<<<gb, bs>>>(up.sc, st, W);
        kWfShade<<<gb, bs>>>(up.sc, up.dc, d_film, d_hits, d_energy, st, W, N,
                             diffraction, maxBounce, d_dispatched, d_live, camModeInt);
        int live = 0;
        cudaMemcpy(&live, d_live, sizeof(int), cudaMemcpyDeviceToHost);
        if (live <= 0) break;
    }

    cudaFree(st.ro); cudaFree(st.rd); cudaFree(st.beta); cudaFree(st.lambda);
    cudaFree(st.rng); cudaFree(st.bounce); cudaFree(st.alive); cudaFree(st.interior); cudaFree(st.hit);
    cudaFree(d_dispatched); cudaFree(d_live);
}

Film renderForwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                       long long N, EnergyReport& eOut, bool diffraction,
                       char camMode, unsigned long long seedBase, bool wavefront) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    if (!cudaAvailable() || !cudaForwardSupported(scene)) return out;

    DUpload up;
    buildUpload(scene, cam, resX, resY, up);

    const size_t npix = (size_t)resX * resY;
    double* d_film = nullptr;   cudaMalloc(&d_film, npix * 3 * sizeof(double));
    cudaMemset(d_film, 0, npix * 3 * sizeof(double));
    double* d_hits = nullptr;   cudaMalloc(&d_hits, npix * sizeof(double));
    cudaMemset(d_hits, 0, npix * sizeof(double));
    double* d_energy = nullptr; cudaMalloc(&d_energy, 5 * sizeof(double));
    cudaMemset(d_energy, 0, 5 * sizeof(double));

    int camModeInt = (camMode == 'A') ? CAM_A : (camMode == 'C') ? CAM_C : CAM_B;

    int blockSize = 128;
    int numBlocks = 2048;          // ~262k threads, grid-stride over N photons
    // seedBase==0 keeps the original single-shot seed exactly; each accumulation
    // chunk passes a distinct cumulative-photon offset for an independent stream.
    unsigned long long kseed = 0x9e3779b97f4a7c15ULL + seedBase * 0x9e3779b97f4a7c15ULL;
    if (wavefront) {
        // Streaming backend: identical physics, path-regeneration scheduling (see the
        // wavefront section above). Same maxBounce (32) and camera mode as the megakernel.
        wavefrontTrace(up, d_film, d_hits, d_energy, N, diffraction ? 1 : 0, kseed, 32, camModeInt);
    } else {
        kTrace<<<numBlocks, blockSize>>>(up.sc, up.dc, d_film, d_hits, d_energy, N, diffraction ? 1 : 0,
                                         kseed, 32, camModeInt);
    }
    cudaError_t kerr = cudaGetLastError();
    if (kerr == cudaSuccess) kerr = cudaDeviceSynchronize();
    if (kerr != cudaSuccess) {
        std::fprintf(stderr, "[cuda] kernel error: %s\n", cudaGetErrorString(kerr));
    }

    // --- download ---
    std::vector<double> film(npix * 3);
    cudaMemcpy(film.data(), d_film, film.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.hits.data(), d_hits, npix * sizeof(double), cudaMemcpyDeviceToHost);
    double energy[5] = {0,0,0,0,0};
    cudaMemcpy(energy, d_energy, 5 * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < npix; ++i)
        out.xyz[i] = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);
    eOut.emitted  += energy[0];
    eOut.absorbed += energy[1];
    eOut.sensor   += energy[2];
    eOut.escaped  += energy[3];
    eOut.residual += energy[4];

    freeUpload(up);
    cudaFree(d_film); cudaFree(d_hits); cudaFree(d_energy);
    return out;
}

// ------------------------------ BDPT (mode D) host ---------------------------

bool cudaBdptSupported(const Scene& scene) {
    // BDPT-GPU needs the same POD-bakeable materials as the forward path, PLUS the
    // BDPT scope restrictions (bdpt.h / mode-D guard in main.cpp): no participating
    // media, and only area/sphere/cylinder Lambertian emitters (no spot/env/collimated).
    if (!cudaForwardSupported(scene)) return false;
    if (scene.medium.enabled) return false;
    // Dielectric translucency (frosting + Beer-Lambert interior absorption) runs on the
    // device forward/backward tracers, but the BDPT kernel (kBdpt) treats every dielectric
    // as smooth & non-absorbing and its pdf/eval use constant params — a frosted or colored
    // glass would bias MIS. Fall back to the CPU BDPT for such scenes.
    auto frostedOrColoredGlass = [&](const Material& m) {
        if (m.type != MatType::Dielectric) return false;
        if (m.roughness > 1e-3 || m.roughnessTex >= 0 || m.roughnessPat >= 0) return true; // frosted
        if (m.absorb)
            for (int i = 0; i <= 8; ++i)
                if (m.absorb(DLMIN + (DLMAX - DLMIN) * i / 8.0) > 0.0) return true;  // colored
        return false;
    };
    // The forward path now supports textured albedo + fluorescence on the device, but
    // the BDPT kernel (kBdpt) does not implement either — its diffuse vertices sample
    // the constant reflect spectrum and it has no fluorescent-vertex strategy — so
    // scenes using them fall back to the CPU BDPT (which does handle both).
    auto usesTexOrFluoro = [&](int matId) {
        if (matId < 0 || matId >= (int)scene.mats.size()) return false;
        const Material& m = scene.mats[matId];
        if (frostedOrColoredGlass(m)) return true;
        if (m.reflectTex >= 0 || m.type == MatType::Fluorescent) return true;
        // Non-albedo texture maps (roughness / film-thickness) drive the glossy/thin-film
        // BSDF sampling per-hit; the GPU BDPT kernel's pdf/eval use the constant params,
        // which would bias MIS. Fall back to CPU BDPT (which threads the Hit through).
        if (m.roughnessTex >= 0 || m.filmThicknessTex >= 0) return true;
        // Procedural patterns drive the same per-hit params; the GPU BDPT kernel's
        // pdf/eval use the constants, so a pattern would bias MIS — use CPU BDPT.
        if (m.roughnessPat >= 0 || m.filmThicknessPat >= 0 || m.mixWeightPat >= 0) return true;
        // Mix blend mask: the GPU BDPT mix-pick uses constant weights; use CPU BDPT.
        if (m.mixWeightTex >= 0) return true;
        if (m.type == MatType::Mix)
            for (int c : m.mixChildren)
                if (c >= 0 && c < (int)scene.mats.size() &&
                    (frostedOrColoredGlass(scene.mats[c]) ||
                     scene.mats[c].reflectTex >= 0 || scene.mats[c].type == MatType::Fluorescent ||
                     scene.mats[c].roughnessTex >= 0 || scene.mats[c].filmThicknessTex >= 0 ||
                     scene.mats[c].roughnessPat >= 0 || scene.mats[c].filmThicknessPat >= 0 ||
                     scene.mats[c].mixWeightPat >= 0))
                    return true;
        return false;
    };
    for (const auto& t : scene.tris)      if (usesTexOrFluoro(t.matId)) return false;
    for (const auto& s : scene.spheres)   if (usesTexOrFluoro(s.matId)) return false;
    for (const auto& im : scene.implicits) if (usesTexOrFluoro(im.matId)) return false;
    for (const auto& em : scene.emitters)
        if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env || em.collimated)
            return false;
    return true;
}

Film renderBdptCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                    long long spp, int maxDepth, bool diffraction) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    if (!cudaAvailable() || !cudaBdptSupported(scene)) return out;
    if (maxDepth > BDPT_MAXDEPTH) maxDepth = BDPT_MAXDEPTH;   // device array bound

    DUpload up;
    buildUpload(scene, cam, resX, resY, up);

    const size_t npix = (size_t)resX * resY;
    double* d_cam   = nullptr; cudaMalloc(&d_cam,   npix * 3 * sizeof(double));
    double* d_splat = nullptr; cudaMalloc(&d_splat, npix * 3 * sizeof(double));
    cudaMemset(d_cam,   0, npix * 3 * sizeof(double));
    cudaMemset(d_splat, 0, npix * 3 * sizeof(double));

    long long totalSamples = (long long)npix * spp;
    kBdpt<<<2048, 128>>>(up.sc, up.dc, d_cam, d_splat, totalSamples, spp, resX,
                         maxDepth, diffraction ? 1 : 0, 0x9e3779b97f4a7c15ULL);
    cudaError_t kerr = cudaGetLastError();
    if (kerr == cudaSuccess) kerr = cudaDeviceSynchronize();
    if (kerr != cudaSuccess)
        std::fprintf(stderr, "[cuda] bdpt kernel error: %s\n", cudaGetErrorString(kerr));

    std::vector<double> camH(npix * 3), splatH(npix * 3);
    cudaMemcpy(camH.data(),   d_cam,   npix * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(splatH.data(), d_splat, npix * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    const double inv = 1.0 / (double)spp;   // camera + light images share the 1/spp scale
    for (size_t i = 0; i < npix; ++i)
        out.xyz[i] = Vec3((camH[3 * i + 0] + splatH[3 * i + 0]) * inv,
                          (camH[3 * i + 1] + splatH[3 * i + 1]) * inv,
                          (camH[3 * i + 2] + splatH[3 * i + 2]) * inv);

    freeUpload(up);
    cudaFree(d_cam); cudaFree(d_splat);
    return out;
}

// --------------------- backward reference (mode R) host ----------------------

bool cudaBackwardSupported(const Scene& scene, const Camera& cam) {
    // GPU mode R needs the same POD-bakeable materials as the forward path, plus the
    // v1 backward scope: no participating media, no environment light, only area/
    // sphere/cylinder Lambertian emitters (spot/env/collimated fall back to the CPU),
    // and no fluorescence. Textured albedo IS supported (dDiffuseRho ports it). This
    // keeps dInvPdfLambda exact (geomWeight = area*PI for every emitter).
    if (!cudaForwardSupported(scene)) return false;
    if (scene.medium.enabled) return false;
    if (scene.envIndex >= 0)   return false;
    auto usesFluoro = [&](int matId) {
        if (matId < 0 || matId >= (int)scene.mats.size()) return false;
        const Material& m = scene.mats[matId];
        if (m.type == MatType::Fluorescent) return true;
        if (m.type == MatType::Mix)
            for (int c : m.mixChildren)
                if (c >= 0 && c < (int)scene.mats.size() &&
                    scene.mats[c].type == MatType::Fluorescent) return true;
        return false;
    };
    for (const auto& t : scene.tris)    if (usesFluoro(t.matId)) return false;
    for (const auto& s : scene.spheres) if (usesFluoro(s.matId)) return false;
    for (const auto& em : scene.emitters)
        if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env || em.collimated)
            return false;
    // A physical lens deeper than the device cap falls back to the CPU tracer.
    if (cam.hasLens() && (int)cam.lens->surf.size() > D_MAXLENS) return false;
    return true;
}

Film renderBackwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                        long long spp, bool diffraction) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    if (!cudaAvailable() || !cudaBackwardSupported(scene, cam)) return out;

    DUpload up;
    buildUpload(scene, cam, resX, resY, up);

    const size_t npix = (size_t)resX * resY;
    double* d_film = nullptr; cudaMalloc(&d_film, npix * 3 * sizeof(double));
    double* d_hits = nullptr; cudaMalloc(&d_hits, npix * sizeof(double));
    cudaMemset(d_film, 0, npix * 3 * sizeof(double));
    cudaMemset(d_hits, 0, npix * sizeof(double));

    long long totalSamples = (long long)npix * spp;
    kBackward<<<2048, 128>>>(up.sc, up.dc, d_film, d_hits, totalSamples, spp, resX,
                             diffraction ? 1 : 0, 0x9e3779b97f4a7c15ULL);
    cudaError_t kerr = cudaGetLastError();
    if (kerr == cudaSuccess) kerr = cudaDeviceSynchronize();
    if (kerr != cudaSuccess)
        std::fprintf(stderr, "[cuda] backward kernel error: %s\n", cudaGetErrorString(kerr));

    std::vector<double> film(npix * 3);
    cudaMemcpy(film.data(), d_film, film.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.hits.data(), d_hits, npix * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < npix; ++i)   // SUM over spp; writeFilm divides by spp
        out.xyz[i] = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);

    freeUpload(up);
    cudaFree(d_film); cudaFree(d_hits);
    return out;
}
