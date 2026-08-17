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
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>                // gpuSppChunks' stall watchdog
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <math.h>
#include <float.h>

// Thrust (CCCL under CUDA, rocThrust under HIP — same headers/namespaces) provides the
// device scan/sort/search primitives for the ON-DEVICE VCM/SPPM grid builds: exclusive_scan
// (per-path vertex offsets), stable_sort_by_key (cell-id counting-sort twin — stable radix
// sort reproduces the host counting sort's exact order), lower_bound (cellStart offsets),
// and transform_reduce (bbox / max-radius reductions). FT_THRUST_PAR(alloc) is the
// backend-parallel execution policy carrying our arena allocator so thrust's temporary
// storage stops churning cudaMalloc/cudaFree every pass.
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#if defined(FTRACE_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
  #define FT_THRUST_PAR(alloc) thrust::hip::par(alloc)
#else
  #define FT_THRUST_PAR(alloc) thrust::cuda::par(alloc)
#endif

#include "render_cuda.h"
#include "render_progress.h"
#include "pattern_device.cuh"   // DPattern / DPatEnvT / dPatternEval — shared with raster_cuda.cu
#include "stochtile.h"    // O7: the host/device-shared histogram-preserving tiling operator
#include "grin.h"         // grin::sceneHasGrin (host gate mirrored into DScene::hasGrin)
#include "photonmap.h"    // host PhotonMap::build reused for the mode-M grid (GPU gather)
#include "raster.h"       // G2 iso preview: shared deriveLight/materialColor/exposeAndEncode (host)
#include "lighttree.h"    // Conty-Kulla light BVH: the SAME traversal the CPU runs, not a copy
                          // (the header is dependency-free and __host__ __device__ for this)

// Abort-loud wrapper for CUDA API calls. Every cudaMalloc/cudaMemcpy/cudaMemset and
// every kernel launch/sync return code MUST be checked: under GPU contention (a second
// process pressuring device memory or scheduling) an allocation or copy can fail, and
// silently ignoring that leaves a zero-initialised host buffer that gets written out as
// a black image with no error. Checking every return turns "silently black" into a
// precise, diagnosable failure (which call, where, and the CUDA error string), then
// exits non-zero so no garbage image is produced. This is the root-cause fix for the
// concurrent-GPU black-render bug (see known-issues.md).
#define CUDA_CHECK(call) do {                                                    \
        cudaError_t _cudaCheckErr = (call);                                      \
        if (_cudaCheckErr != cudaSuccess) {                                      \
            std::fprintf(stderr, "[cuda] %s failed at %s:%d: %s\n",              \
                         #call, __FILE__, __LINE__,                              \
                         cudaGetErrorString(_cudaCheckErr));                     \
            std::fflush(stderr);                                                 \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// After a kernel launch, check both the launch (cudaGetLastError) and the execution
// (cudaDeviceSynchronize) status; abort loudly on either. `what` names the kernel for
// the diagnostic. A display-driver TDR or an out-of-resources launch surfaces here.
static void cudaCheckKernel(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[cuda] %s kernel failed: %s\n", what, cudaGetErrorString(e));
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}

// ============================ device-side scene ============================

#define HD __host__ __device__
static constexpr int    SPEC_N   = 96;        // spectral table resolution
static constexpr double DLMIN    = 360.0;     // mirrors color.h LAMBDA_MIN/MAX
static constexpr double DLMAX    = 830.0;
static constexpr double DPI      = 3.141592653589793;

// ---- host-side spectral->RGB bakes for the fast RGB backward (mode R -rgb) ----
// These run at scene-build time (buildUploadScene) to precompute the Option-B RGB
// throughput tables. They live at global scope (outside namespace gpu) so `cieX/Y/Z`
// and `xyzToLinearSrgb` resolve to the HOST color.h functions, not the device twins.
// A reflectance R(lambda) bakes to its linear-sRGB colour under an equal-energy white
// (a spectrally flat white reflector -> (1,1,1)); an emission SPD bakes to the exact
// wavelength-integrated XYZ->linear-sRGB radiance the spectral estimator converges to.
namespace rgbbake {
// integral over [LAMBDA_MIN,LAMBDA_MAX] of CIE(lambda)*s(lambda) dlambda (1 nm Riemann).
inline Vec3 specToXyz(const Spectrum& s) {
    Vec3 xyz(0, 0, 0);
    if (!s) return xyz;
    for (double w = LAMBDA_MIN; w <= LAMBDA_MAX; w += 1.0) {
        double v = s(w);
        xyz.x += cieX(w) * v; xyz.y += cieY(w) * v; xyz.z += cieZ(w) * v;
    }
    return xyz;   // dlambda = 1 nm
}
// Emission SPD -> linear-sRGB radiance (matches the spectral film's absolute scale).
inline Vec3 emitToRgb(const Spectrum& s) { return xyzToLinearSrgb(specToXyz(s)); }
// Reflectance SPD -> linear-sRGB albedo under an equal-energy white (white -> 1,1,1),
// clamped to [0,1] per channel.
inline Vec3 reflToRgb(const Spectrum& s) {
    static const Vec3 whiteRgb = [] {
        Vec3 x(0, 0, 0);
        for (double w = LAMBDA_MIN; w <= LAMBDA_MAX; w += 1.0) { x.x += cieX(w); x.y += cieY(w); x.z += cieZ(w); }
        return xyzToLinearSrgb(x);
    }();
    Vec3 numRgb = xyzToLinearSrgb(specToXyz(s));
    auto c01 = [](double a, double b) { double r = (b != 0.0) ? a / b : 0.0; return r < 0 ? 0 : (r > 1 ? 1 : r); };
    return Vec3(c01(numRgb.x, whiteRgb.x), c01(numRgb.y, whiteRgb.y), c01(numRgb.z, whiteRgb.z));
}
}  // namespace rgbbake

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

// Max-t for a CONNECTION ray — a shadow ray whose far end lands ON a sampled point of
// another surface (BDPT/VCM's s=1 light connection, its t=1 camera connection, an s,t>=2
// vertex-to-vertex connection, a caustic chain's connection to the eye). Such a ray must
// stop just SHORT of that point or it re-hits the very surface the point was sampled from
// and the sample is thrown away as occluded.
//
// The CPU reference shortens by a fixed 2e-6 and gets away with it because it is double
// throughout. The fp32 device build cannot: `dist` is rebuilt from fp32 coordinates, so the
// endpoint jitters by ~|coord|*2^-23 around the true surface (half of that jitter lands
// INSIDE the emitter), and worse, 2e-6 is below one ulp of `dist` past ~17 m, so the cast
// rounds it straight back to `dist` and the shortening vanishes entirely. The visible
// symptom is a DISTANT light losing most of its energy: measured on a 1.85 m sphere 400 m
// from the shading point, mode D on the GPU came out 2.7x darker than the CPU reference
// (1.5x at 40 m, clean by ~4 m). Any scene modelling the sun as a far-away sphere hit this.
//
// So shorten by a RELATIVE amount, which tracks the fp32 jitter at every scene scale, and
// keep the caller's absolute epsilon as a floor for near geometry. 1e-5 is ~84 ulp of a
// float — two orders above the jitter — and in absolute terms 10 um at 1 m or 4 mm at
// 400 m, far below any occluder these rays could legitimately need to see. The fp64 build
// sets the relative term to zero so it stays bit-identical to the CPU reference.
#if FTRACE_GPU_FP32
static constexpr double CONN_REL_EPS = 1e-5;
#else
static constexpr double CONN_REL_EPS = 0.0;
#endif
// `absEps == 0` means "do not shorten at all": the distant-sun connection's far end is the
// scene EXIT, not a sampled surface point, so it must keep the whole segment (see the
// shape==6 branch of the s=1 connection, which sets occlEps = 0 for exactly that reason).
__device__ __host__ inline Real connMaxT(double dist, double absEps = 2e-6) {
    if (absEps <= 0.0) return (Real)dist;
    double e = dist * CONN_REL_EPS;
    return (Real)(dist - (e > absEps ? e : absEps));
}

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
    // Indexed component access (0=x,1=y,2=z) for the watertight tri test's axis
    // permutation. No bounds check (hot path); callers pass 0..2.
    HD Real  operator[](int i) const { return (&x)[i]; }
    HD Real& operator[](int i)       { return (&x)[i]; }
};
HD static inline Real dot(const DVec3& a, const DVec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
// Componentwise (Hadamard) product — RGB throughput * albedo in the fast RGB backward.
HD static inline DVec3 hadamard(const DVec3& a, const DVec3& b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
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
// Values MUST match MatType (scene.h) 1:1 — the upload does d.type = (int)m.type. D_LAYERED
// (MatType::Layered) has no device branch (Layered scenes fall back to the CPU tracer via
// cudaForwardSupported), but the placeholder keeps D_DIFFUSETRANSMIT aligned at index 11.
enum { D_DIFFUSE=0, D_DIELECTRIC, D_MIRROR, D_HALFMIRROR, D_GLOSSY, D_FLUORESCENT, D_THINFILM,
       D_GRATING, D_MIX, D_MULTILAYER, D_LAYERED, D_DIFFUSETRANSMIT, D_FILTER, D_HAIR };

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
    const double* rgb;     // 3*w*h linear RGB, uploaded only for NORMAL-MAP textures
                           // (C6) — dTexNormalAt twin (needs true vector direction)
    // O7 stochastic tiling: the parameters plus the two Gaussianized plane sets and
    // their inverse LUTs, mirroring Texture::gauss*/lut* one for one. All null (and
    // stoch.on == 0) unless the texture asked for `tiling stochastic`; the samplers below
    // then take exactly the same branch the host does, over the same float data, so the
    // two backends land on the same crop of the image with the same weights.
    // `jhLut` is the ONE shared, texture-independent RGB -> Jakob-Hanika coefficient
    // table (upsample::coeffLut(), sqrt-warped STOCH_JH_N^3); the spectral path blends
    // in linear RGB and then converts through it, because blending JH coefficients per
    // channel leaves the image's colour manifold (see stochtile.h).
    StochTile    stoch;
    const float* gaussRgb;   const float* lutRgb;
    const float* gaussGray;  const float* lutGray;
    const float* jhLut;
    // The pattern VM's `tex:` hook (pattern_device.cuh). Defined out of line below,
    // next to dTexScalarAt, which it forwards to.
    __device__ double patScalarAt(double u, double v) const;
};

struct DMaterial {
    int    type;
    // Twin of Material::readsCavity — the per-hit gate on the cavity probe. See the
    // note there and on ftsl::setupCavity: without it, one cavity-driven material would
    // charge cavitySamples occlusion rays to every OTHER patterned surface in the scene.
    int    readsCavity;
    double reflect[SPEC_N];     // baked reflect spectrum
    double ior[SPEC_N];         // baked index spectrum
    double substrateK[SPEC_N];  // baked thin-film substrate extinction kappa (0 = transparent)
    // Beer-Lambert interior absorption sigma_a(lambda) per metre travelled INSIDE a
    // dielectric (colored/attenuating glass). All-zero = colorless (default). Consulted
    // only for D_DIELECTRIC, via the `interior` material tracked through the transport
    // loop (device twin of Material::absorb).
    double absorb[SPEC_N];
    // Diffuse-transmission albedo (D_DIFFUSETRANSMIT only): the back-hemisphere (-n)
    // Lambertian lobe. `reflect` is the front (+n) lobe; reflect+transmit is energy-
    // clamped to <= 1 per wavelength at shade time. Device twin of Material::transmit.
    double transmit[SPEC_N];
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
    // filmThickness. Honoured by the forward paths (megakernel + wavefront) AND the GPU
    // BDPT kernel (M9: the per-hit point is threaded into dMatRoughness/dMatFilmThickness).
    int    roughnessTex;
    int    filmThicknessTex;
    // Tangent-space NORMAL MAP (C6), device twin of Material::normalTex/normalStrength.
    // >=0 => at a hit, perturb the shading normal by the texel's tangent-space normal
    // (dTexNormalAt) rotated through the surface TBN frame; -1 => geometry normal.
    // normalStrength scales the tangential perturbation. Applied in the device
    // closestHit (dApplyNormalMap) so every GPU path sees it consistently.
    int    normalTex;
    double normalStrength;
    // Fluorescence (D_FLUORESCENT): fluoAbsorb is the baked excitation probability
    // epsilon(lambda); the dye re-radiates (quantum yield fluoYield) at a Stokes-
    // shifted lambda' drawn from the emission-SPD CDF slice [fluoCdfOffset,
    // fluoCdfOffset+fluoCdfN) inside DScene::fluoCdfAll (fluoCdfStep = bin width nm).
    double fluoAbsorb[SPEC_N];
    double fluoYield;
    int    fluoCdfOffset, fluoCdfN;
    double fluoCdfStep;
    // Baked emission SPD M(lambda) and its integral, so the backward adjoint can
    // evaluate the continuous emission density gOut = (M(lambda)/fluoMint)*invPdf
    // at a FIXED output wavelength (the forward path samples lambda' from the CDF
    // instead, where M/pdf cancels and these aren't needed).
    double fluoEmitSpec[SPEC_N];
    double fluoMint;
    // EXCITATION-wavelength CDF slice (device twin of Material::fluoInSampler), also
    // living inside DScene::fluoCdfAll. Built from absorb(lambda)*g(lambda), so the
    // backward adjoint's lambda_in always lands inside the dye's absorption band
    // instead of being thrown at the whole illuminant (a variance reduction in the
    // stochastic modes, and 1-spp correctness in mode W). fluoInCdfN == 0 => the
    // product had no mass; fall back to the scene illuminant sampler.
    int    fluoInCdfOffset, fluoInCdfN;
    double fluoInCdfStep;
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
    // Material::roughnessPat / filmThicknessPat / mixWeightPat. reflectPat instead
    // MULTIPLIES the reflect slot per hit (device twin of Material::reflectPat) — the
    // greyscale-albedo half of `reflect pattern:<n>` / `reflect_map pattern:<n>`.
    // transmitPat does the same on the transmit slot (device twin of Material::transmitPat).
    int    roughnessPat;
    int    filmThicknessPat;
    int    mixWeightPat;
    int    reflectPat;
    int    transmitPat;
    // emitPat does the same on the emission slot, for emission-on-hit (device twin of
    // Material::emitPat). The other half of the slot — Le at a point the emitter SAMPLER
    // drew — goes through DEmitter::emitPat; the two are constructed to agree pointwise
    // because MIS combines them.
    int    emitPat;
    // Self-emission carried by the MATERIAL itself (device twin of Material::isLight /
    // Material::emit). A `light` block registers a DEmitter and emission-on-hit is read
    // from that (see dEmitterForMat below), but a material can also carry a bare `emit`
    // spectrum on geometry that has no registered emitter at all — an isosurface, a CSG
    // sphere, a quadric. Those are marched/analytic rather than tessellated, so there is
    // nothing to area-sample and no emitter to register; they still have to LOOK lit when
    // the camera sees them. matEmit is that fallback, consulted only when
    // dEmitterForMat() < 0 so a mesh/quad light never double-counts.
    int    matIsLight;
    double matEmit[SPEC_N];
    DVec3  rgbMatEmit;          // matEmit baked to linear sRGB, for the fast RGB backward
    // Nested-dielectric priority (Schmidt & Budge 2002): higher wins where dielectrics
    // overlap; INT_MIN (D_NO_PRIORITY) means "unset" -> flat air<->glass fallback. Device
    // twin of Material::priority.
    int    priority;
    // Parametric-record REFLECT binding (§records stage 6a — device twin of the CPU
    // recordReflectBound path). recReflDriven==0 means no record drives reflect per-hit
    // (a *constant* selStop binding is instead baked straight into reflect[] at upload,
    // needing no device branch). recReflDriven==1 means per-hit driven: sample the coeff
    // LUT recCoeff[recReflOff .. +3*REC_LUT_N) at driver position
    // d = dPatternEval(recDrivers[recReflDrvOff .. +recReflDrvN)) over the
    // [recReflLo,recReflHi] domain, then evaluate the JH sigmoid (dRecReflAt). Consulted
    // by dDiffuseRho / dReflectSlot; BDPT (no per-hit Hit in dBsdfF) rejects driven-record
    // materials to CPU.
    int    recReflDriven;
    int    recReflOff;
    int    recReflDrvOff, recReflDrvN;
    float  recReflLo, recReflHi;
    // Parametric-record ROUGHNESS binding (scalar slot — §records stage 6b, device twin of
    // the CPU materialRoughness record path). recRoughMode: -1 none; 0 direct scalar
    // expression (recordIndex<0; program recDrivers[recRoughDrvOff .. +recRoughDrvN));
    // 1 constant selStop (one stop expr recScalarStops[recRoughStopOff], evaluated per-hit);
    // 2 per-hit driven (driver recDrivers[recRoughDrvOff..) picks a position over the stop
    // range, then recScalarStops[recRoughStopOff .. +recRoughStopN) are evaluated per-hit
    // and interpolated by recRoughInterp — exactly recSampleScalar). Result clamped [0,1];
    // consulted first by dMatRoughness. BDPT rejects record materials to CPU.
    int    recRoughMode;
    int    recRoughDrvOff, recRoughDrvN;
    int    recRoughStopOff, recRoughStopN;
    int    recRoughInterp;
    float  recRoughLo, recRoughHi;
    // Fast RGB backward (Option B — mode R -rgb). Precomputed linear-sRGB reflectance /
    // transmission albedo (the surface colour under an equal-energy white, so a spectrally
    // flat white reflector bakes to (1,1,1)); a single representative achromatic index
    // rgbIor = ior(550 nm) (the RGB path drops dispersion); and a 3-tap Beer-Lambert
    // absorption rgbAbsorb = sigma_a at the R/G/B pivots (610/550/465 nm) for coloured
    // glass. Baked once in buildUploadScene; consumed only by bkRadianceRGB.
    DVec3  rgbAlbedo;
    DVec3  rgbTransmit;
    DVec3  rgbAbsorb;
    double rgbIor;
    // Hair fiber BCSDF (MatType::Hair — device twin of Material::hair* fields, hair.h /
    // hair_shade.h). Single-fiber Marschner/Chiang R/TT/TRT + residual, plus the Yan 2017
    // medulla scattered lobes when hairKappa > 0. hairSigmaAFromReflect != 0 means derive
    // the cortex absorption from the material's diffuse reflectance colour per-wavelength
    // (Chiang eq. 9, via dDiffuseRho so textures/patterns tint the fiber); otherwise
    // hairSigmaA[] is the explicit absorption spectrum. Dual scattering and the fur-volume
    // aggregate stay CPU-only (main.cpp gates them off the GPU path).
    double hairEta, hairBetaM, hairBetaN, hairAlpha, hairKappa, hairMedullaG;
    double hairSigmaA[SPEC_N];
    double hairMedullaSigmaS[SPEC_N];
    double hairMedullaSigmaA[SPEC_N];
    int    hairSigmaAFromReflect;
};
// Sentinel for an unset dielectric priority (device twin of host INT_MIN).
#define D_NO_PRIORITY (-2147483647 - 1)
__device__ __host__ static inline bool dHasPriority(const DMaterial& m) { return m.priority != D_NO_PRIORITY; }

// Per-path nested-dielectric medium stack (device twin of host MediumStack). The current
// optical medium (for Beer-Lambert absorption + exterior IOR at the next interface) is the
// highest-priority entry. A smaller CAP than the host (deep dielectric nesting is rare;
// overflow degrades gracefully) keeps per-slot wavefront memory and megakernel local
// footprint modest.
struct DMediumStack {
    static const int CAP = 8;
    int matIdx[CAP];
    int pri[CAP];
    int n;
    __device__ __host__ void clear() { n = 0; }
    __device__ __host__ bool empty() const { return n == 0; }
    __device__ __host__ int topPri() const {
        int bp = D_NO_PRIORITY;
        for (int i = 0; i < n; ++i) if (pri[i] > bp) bp = pri[i];
        return n ? bp : D_NO_PRIORITY;
    }
    __device__ __host__ int topMat() const {
        int bp = D_NO_PRIORITY, bm = -1;
        for (int i = 0; i < n; ++i) if (pri[i] >= bp) { bp = pri[i]; bm = matIdx[i]; }
        return bm;
    }
    __device__ __host__ void push(int mi, int p) { if (n < CAP) { matIdx[n] = mi; pri[n] = p; ++n; } }
    __device__ __host__ void popMat(int mi) {
        for (int i = n - 1; i >= 0; --i)
            if (matIdx[i] == mi) {
                for (int j = i; j < n - 1; ++j) { matIdx[j] = matIdx[j + 1]; pri[j] = pri[j + 1]; }
                --n; return;
            }
    }
};

struct DTri    { DVec3 v0, v1, v2, gn; DVec3 uv0, uv1, uv2; DVec3 n0, n1, n2; int matId, sensorId;
                 DVec3 tangent; double bitangentSign;    // C6 tangent frame for normal mapping
                 double curvature; };                    // O3 per-face mean curvature (Tri::finalize)
struct DSphere { DVec3 c; double r; int matId; };
// One round cone of a curve/fiber strand — the device twin of CurveSeg (curve.h). The host
// record is already a POD in exactly this shape; only the scalar type narrows to Real.
struct DCurveSeg { DVec3 p0, p1; Real r0, r1; int matId; float u0, u1; };
struct DNode   { DVec3 lo, hi; int left, right, first, count; };

// One flat mirror SURFACE (device twin of Scene::MirrorPlane), i.e. the set of coplanar
// mirror-material triangles sharing a plane, collapsed to that plane plus the AABB of its
// members. The forward camera connection reflects a photon vertex in this plane
// analytically, so the geometry is solved in DOUBLE (like every other specular connector
// here) regardless of the render Real — hence plain doubles rather than DVec3.
struct DMirrorPlane {
    double nx, ny, nz;      // unit plane normal (canonically oriented)
    double d;               // plane equation: dot(n, x) == d
    double lox, loy, loz;   // AABB of the member faces — a cheap reject before the
    double hix, hiy, hiz;   // closestHit that confirms the mirror really is there
};

// Two-level BVH for instancing (device twin of scene.h Blas / MeshInstance). A DBlas
// is a shared mesh asset held ONCE in local (authored) space as a slice of the flat
// per-BLAS pools (blasNodes/blasTris/blasPrim); a DInstance places it into the world
// via an affine WITHOUT baking a private triangle copy. The TLAS (DScene::nodes) gets
// one leaf per instance; the leaf transforms the ray into BLAS-local space and walks
// the shared sub-BVH. This is the device memory win over expanding instances to world
// tris at upload (N copies cost N affines, not N triangle sets). See known-issues.md.
struct DBlas     { int nodeOff, triOff, primOff; };   // offsets into the flat BLAS pools
struct DInstance {
    // world -> local affine (toLocal): p_local = Lm*p + Lt, dir_local = Lm*dir.
    double Lm[9], Lt[3];
    // shading/geometric normal local -> world = (toWorld linear)^-T = transpose of
    // toWorld.inverse().m — precomputed on the host so the device does no inverse.
    double Nm[9];
    // toWorld linear part (local -> world for plain DIRECTIONS): transforms the surface
    // tangent for normal mapping on instanced meshes (C6). affDir(Wm, tangent).
    double Wm[9];
    // O3: local curvature (1/length) -> world curvature multiplier, = 1/|det(linear)|^(1/3).
    // Host-precomputed (MeshInstance::curvScale) so the device does no cbrt per hit.
    double curvScale;
    int    blasId;
    int    matOverride;   // >=0 replaces the BLAS triangles' matId (mirrors host)
};

// Implicit surfaces (isosurface / CSG / metaballs) — device twins of implicit.h.
// The field is a flat postfix array evaluated with a scalar stack, sphere-traced for
// intersection. The MARCH + root REFINE run in FP32 on pre-converted mirror pools
// (DFieldNodeF/PatNodeF): the committed hit is float anyway, and FP64 VM ops
// serialize on consumer GPUs' 1/64-rate FP64 pipe. Normals (dFieldGradient) and
// media bound-field evals stay in DOUBLE on the original pools. POD twins
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
// FP32 mirrors of the field/pattern node pools, pre-converted on upload. The sphere-
// trace MARCH + root REFINE in intersectImplicit run entirely on these: the committed
// hit is stored in float anyway (DHit::t/p are Real), so double stepping buys nothing
// there, while the FP64 VM ops serialize on the 1/64-rate FP64 pipe of consumer GPUs
// (measured: ~90% of BDPT subpath generation on the gallery scene). Pre-converting
// whole pools (rather than casting per-op) keeps F64<->F32 cvt instructions — which
// also issue on the FP64 pipe — out of the inner loop. Normals (dFieldGradient) and
// media bound-fields still use the DOUBLE pools.
struct DFieldNodeF {
    int   op;
    float p[4];
    float inv[9];
    float tx, ty, tz;
    float scale;
    int   exprOff, exprN;  // same slice indices as the double twin (pools are parallel)
};
struct PatNodeF { int op; float a; };   // FP32 twin of pattern.h PatNode (8B vs 16B)
struct DImplicit {
    int    nodeOff, nodeN;   // slice [nodeOff, nodeOff+nodeN) into DScene::fieldNodes
    int    matId;
    double lo[3], hi[3];     // world AABB (ray clip)
    double lipschitz, minStep;
    int    method;           // 0 = adaptive (|f|/lipschitz), 1 = fixed-step sample
    int    refine;           // 0 = bisect, 1 = regula-falsi (Illinois)
    double sampleStep;       // fixed world march step for method==1
    int    uvProj;           // 0 none, 1 planar, 2 spherical, 3 cylindrical (UvProjection)
    int    uvAxis;           // 0=x, 1=y, 2=z (projection/up axis)
    double uvLo[3], uvHi[3]; // reference box for the [0,1] UV wrap
    int    container;        // 0 = box (lo/hi), 1 = sphere (sphereCenter/sphereRadius)
    double sphereCenter[3];  // world center for Container::Sphere
    double sphereRadius;     // world radius for Container::Sphere
    int    capped;           // 1 = draw container caps (closed); 0 = `open`
    // Oriented container box — device twin of Implicit::boxOriented/boxInv/boxLo/boxHi.
    // When set, the ray is mapped by boxInv and slab-tested against boxLo/boxHi there
    // instead of against the world AABB, which for a rotated box is strictly larger
    // than the authored container and outside the reach of its `max_gradient` bound.
    int    boxOriented;      // 1 = clip in the box's own frame
    double boxInvM[9], boxInvT[3];   // world -> container-local
    double boxLo[3], boxHi[3];       // authored box in that frame
};

// Procedural pattern (math-driven scalar field, §4) — device twin of pattern.h.
// One flat postfix PatNode pool (DScene::patNodes) holds every pattern back-to-back;
// each DPattern slices it by [off, off+n). A material's roughnessPat/filmThicknessPat/
// mixWeightPat index a DPattern (or -1). The postfix VM (dPatternEval) runs the same
// opcode/hash-noise math as the host so CPU and GPU agree. PatNode/PatOp come from
// pattern.h (POD, uploaded verbatim) — no device-specific node type is needed.
// `DPattern` itself lives in pattern_device.cuh, shared with the preview rasterizer.

// A NATIVE SPARSE brick grid: the device twin of host VdbGrid, uploaded as a bricked
// sparse lattice (ROADMAP C2). The dense lattice is partitioned into B^3 bricks; only
// bricks with a nonzero voxel are uploaded, so VRAM scales with occupied volume, not
// the bounding box. `brickIndex[(k>>sh)*by*bx + (j>>sh)*bx + (i>>sh)]` gives the brick's
// slot (or -1 => value 0); the voxel lives at `brickData[slot*B^3 + ((k&mask)*B +
// (j&mask))*B + (i&mask)]`. `dVdbSample` trilinearly samples it, bit-for-bit like the
// host VdbGrid::sample (the stencil is clamped to [0,n-1] before any lookup). Used for
// BOTH the density multiplier and the emissive-volume temperature field.
struct DVdbGrid {
    const int32_t*   brickIndex;   // bx*by*bz brick slots (or -1); null => grid absent
    const uint16_t*  brickData;    // active*B^3 fp16 voxels
    int              bx, by, bz;   // brick-grid dimensions
    int              brickB;       // brick edge length (power of two)
    int              brickShift;   // log2(B): brick = idx>>shift, local = idx&(B-1)
    int              nx, ny, nz;   // dense lattice dims
    double           ainv[9];      // world->index linear map (row-major 3x3)
    DVec3            w0;            // world position of index origin (0,0,0)
    DVec3            imin;          // integer min-corner of the baked lattice
};

struct DMedium {
    int    enabled;
    double sigma_a[SPEC_N];
    double sigma_s[SPEC_N];
    double g;
    // --- Optional heterogeneous density field + spatial bound (mirrors host Medium) ---
    // When `heterogeneous`, sigma_a/sigma_s are multiplied per point by a dimensionless
    // density(x,y,z) >= 0 evaluated by the postfix pattern VM over `density`[0..densityN).
    // Sampling then switches to delta (Woodcock) tracking for collisions and ratio
    // tracking for transmittance, with majorant sigma_max = sigmaT * densityMax. When
    // `bounded`, the medium exists only inside the AABB [bmin,bmax]. A homogeneous
    // unbounded medium keeps the exact analytic behaviour (bit-identical to before).
    int              heterogeneous;   // 1 => density program present
    const PatNode*   density;         // device pool for the density formula (or null)
    int              densityN;        // node count of the density program
    double           densityMax;      // majorant (sup of density over the bound)
    // --- Optional imported .nvdb/.vdb volume, uploaded as a NATIVE SPARSE brick grid ---
    // When `densGrid.brickData` is non-null the density multiplier is TRILINEARLY
    // sampled from the sparse lattice (ROADMAP C2) instead of the pattern VM; takes
    // precedence. See DVdbGrid.
    DVdbGrid         densGrid;        // density field (brickData null => none)
    // --- Optional volumetric blackbody EMISSION ("fire", ROADMAP C3) ---------------
    // When `emissive` a temperature field (tempGrid) drives self-illuminated blackbody
    // emission: T(x) = emitKelvin * tempGrid(x)/tempPeak (peak-normalised, robust to
    // whatever units the grid was authored in), and the emission source radiance is
    // emissionScale * blackbodyEmissionRadiance(T, lambda). Mirrors host Medium.
    int              emissive;        // 1 => temperature-driven blackbody emission
    DVdbGrid         tempGrid;        // raw relative temperature field
    double           emitKelvin;      // Kelvin of the hottest voxel
    double           tempPeak;        // raw temperature-grid peak (for peak-normalisation)
    double           emissionScale;   // brightness multiplier on the Planck term
    int              bounded;         // 1 => clip to the bound region
    int              boundShape;      // 0 => box [bmin,bmax], 1 => sphere, 2 => implicit field,
                                      // 3 => MESH containment (boundGrid occupancy >= 0.5)
    DVec3            bmin, bmax;
    DVec3            bcenter;
    double           bradius;
    // --- Optional implicit/isosurface bound (boundShape==2). The fog fills the field's
    // interior: a point is inside when dFieldEval < 0 (boundInsideNeg) or > 0. The field
    // program lives in its own device slice; bmin/bmax hold the field AABB (box clip). ---
    const DFieldNode* boundField;     // implicit bound field nodes (or null)
    int               boundFieldN;    // node count
    const PatNode*    boundFieldExpr; // expr pool backing DF_EXPR leaves (or null)
    int               boundInsideNeg; // 1 => inside when field < 0, else inside when > 0
    // --- Optional MESH containment lattice (boundShape==3). Occupancy baked from the
    // named mesh's triangles on the host (meshvoxel.h) and uploaded as a sparse brick
    // grid like any other volume; inside == trilinear sample >= 0.5. Separate from
    // densGrid because it decides MEMBERSHIP only — a `density` field still multiplies
    // on top, shaping the fog within the mesh silhouette. ---
    DVdbGrid          boundGrid;
    // --- Optional gradient-index (GRIN) refractive field n(x,y,z) (mirrors host Medium) ---
    // When `iorN > 0` this region bends rays along the Eikonal ray equation; the forward
    // megakernel/wavefront march through it (dGrinMarch) before each bounce's closestHit.
    const PatNode*    ior;            // compiled n(x,y,z) program (device pool) or null
    int               iorN;           // node count of the ior program (0 => not GRIN)
    double            iorStep;        // Eikonal march step in world units (>0 for GRIN)
    // --- Optional spectral rainbow (Airy droplet) phase table (mirrors host RainbowPhase) ---
    // When `rbPdf` is non-null the angular phase is the tabulated (lambda x mu) Airy
    // droplet function instead of the analytic HG lobe (`g` above), reproducing the
    // primary/secondary bows, supernumeraries and fogbow. rbPdf/rbCdf are nLam*nMu
    // row-major [li*nMu + mi]; mu = -1 + mi*dMu, dMu = 2/(nMu-1); lambda = rbLam0 + li*rbDLam.
    const double*     rbPdf;          // phase table p(lambda, mu) (device pool) or null => HG
    const double*     rbCdf;          // per-lambda CDF over mu for importance sampling
    int               rbNLam, rbNMu;  // table dimensions
    double            rbLam0, rbDLam; // wavelength axis origin/step (nm)
};

// One triangle of a Mesh emitter (mirrors host EmitTri): v0 + two edge vectors, the
// unit normal, and the inclusive cumulative-area CDF value used for area sampling.
// uv0/uvE1/uvE2 mirror the same-named host fields so a sampled point can report the
// SAME (u,v) the ray-hit path interpolates — only read when an emission pattern is
// bound, but uploaded unconditionally (they are part of the host EmitTri).
struct DEmitTri { DVec3 v0, e1, e2, nrm; double cumArea; DVec3 uv0, uvE1, uvE2; };

// One emitter (mirrors host Emitter). `cdfOffset`/`cdfN` index this emitter's
// wavelength CDF slice inside the flattened lightCdfAll buffer.
struct DEmitter {
    DVec3  origin, u, v, normal, beamDir;
    double area, power;
    int    collimated;
    int    shape;              // 0 quad, 1 sphere, 2 spot, 3 env, 4 cylinder, 5 mesh, 6 sun
    double radius;             // sphere radius (shape==1) / tube radius (shape==4)
    int    caps;               // cylinder (shape==4): also emit from the two end discs
    // Cone cosines / solid angle. Spot (shape==2): the smoothstep penumbra. Distant sun
    // (shape==6): inner == outer == cos(halfAngle), so spotOmega = PI*(2-ci-co) is
    // exactly the solar cone's solid angle 2*PI*(1-cos theta) — the same field reuse the
    // host Emitter makes, so no extra members are needed on either side.
    double spotCosInner, spotCosOuter, spotOmega;
    // Mesh area light (shape==5): device pointer to this emitter's triangle CDF and its
    // count. nullptr/0 for every other shape. area == sum of the triangle areas.
    const DEmitTri* meshTris; int meshTriN;
    int    cdfOffset, cdfN;
    double cdfStep;
    // BDPT (mode D) extras. matId links this emitter to its emissive surface material
    // (for the s=0 direct-hit strategy); emitSpd is the baked emission SPD so the
    // device can evaluate Le(lambda) directly (DMaterial carries no emit spectrum).
    int    matId;
    double emitSpd[SPEC_N];
    // Index into DScene::patterns of this emitter's emission profile (device twin of
    // Emitter::emitPat, itself adopted from the emissive material at registration);
    // -1 = uniform. Deliberately absent from `power` — the pattern modulates the
    // radiance at a point, not the emitter's selection weight, exactly as on the host.
    int    emitPat;
    // Fast RGB backward (mode R -rgb): the emitter's linear-sRGB radiance, baked as
    // xyzToLinearSrgb(integral over lambda of CIE(lambda)*emitSpd(lambda)) — the exact
    // wavelength-integrated radiance the spectral estimator converges to (the
    // p(lambda)*invPdfLambda cancellation), so NEE folds it in with no per-wavelength term.
    DVec3  rgbEmit;
};

// PBRT's IsDeltaLight (device twin of bdpt.h isDeltaEmitter): this emitter's emission
// carries a Dirac delta, so some BDPT strategies cannot produce it and must be DROPPED
// from the balance heuristic rather than weighted:
//   spot (2)       delta POSITION (a mathematical point) - no eye ray can hit it, so the
//                  s=0 strategy is impossible; NEE (s=1) works (deterministic point).
//   sun (6)        delta DIRECTION and infinitely distant - again no geometry to hit;
//                  NEE samples inside the solar cone with pdf 1/Omega.
//   collimated     delta direction from a FINITE surface - neither s=0 nor NEE works, so
//                  these stay out of BDPT scope entirely (the mode-D guard refuses them).
__device__ static inline bool dIsDeltaEmitter(const DEmitter& em) {
    return em.shape == 2 || em.shape == 6 || em.collimated != 0;
}
// PBRT's IsInfiniteLight (device twin of bdpt.h isInfiniteEmitter): no finite emission
// point, so the light subpath's origin is a fictitious point on a disc outside the scene
// bounds and the first scene vertex's density is PLANAR (1/(pi R^2)) rather than a
// solid-angle density converted by 1/dist^2. Sun is one; env is the other, but env is
// still outside BDPT scope (it also needs escaped-ray radiance the walk doesn't do).
__device__ static inline bool dIsInfiniteEmitter(const DEmitter& em) {
    return em.shape == 6 || em.shape == 3;
}

// Smoothstep spot falloff (mirrors host scene.h spotFalloff).
__device__ static double spotFalloff(double ct, double cosInner, double cosOuter) {
    if (ct >= cosInner) return 1.0;
    if (ct <= cosOuter) return 0.0;
    double t = (ct - cosOuter) / (cosInner - cosOuter);
    return t * t * (3.0 - 2.0 * t);
}

// ---- cone-emitter helpers, device twins of host Emitter::sampleCone/inCone -----------
// Uniform direction inside the cone of half-angle acos(spotCosOuter) about `axis`
// (solid-angle pdf 1/(2 PI (1 - spotCosOuter)), which for a SUN — whose inner and outer
// cosines coincide — is exactly 1/spotOmega). Same closed form and same u1/u2 roles as the
// host, so CPU and GPU agree on the shape of the penumbra. Named for the sun because that
// was its first user; GPU BDPT also draws a SPOT's emission direction through it, where the
// sampling cone is the OUTER one and the smoothstep penumbra is throughput, not density
// (note spotOmega = PI*(2-ci-co) is the FALLOFF-WEIGHTED solid angle there, NOT the
// sampling cone — they coincide only when ci == co, i.e. for a sun).
__device__ static inline DVec3 dSunSampleCone(const DEmitter& em, const DVec3& axis,
                                              double u1, double u2) {
    double ct = em.spotCosOuter + u1 * (1.0 - em.spotCosOuter);
    double st = sqrt(fmax(0.0, 1.0 - ct * ct));
    double phi = 2.0 * 3.14159265358979323846 * u2;
    DVec3 t, b; onb(axis, t, b);
    return t * (Real)(st * cos(phi)) + b * (Real)(st * sin(phi)) + axis * (Real)ct;
}
// Does viewing direction `d` land on this sun's disc? `beamDir` is the TRAVEL direction,
// so a ray looking AT the sun runs opposite it.
__device__ static inline bool dInSunCone(const DEmitter& em, const DVec3& d) {
    return (double)dot(d, em.beamDir) <= -em.spotCosOuter;
}
// (dSunRadiance — the summed radiance of every sun whose disc contains a direction — is
// defined further down, once DScene exists.)

// Sample a surface point + outward normal on an emitter (mirrors host
// Emitter::samplePoint). Quad draws are unchanged, so quad scenes stay parity.
//
// `uuOut`/`vvOut` optionally report the sampled point's TEXTURE coordinates, which an
// emission pattern needs. As on the host they are filled only for the two shapes that
// can carry one — Quad (the bilinear parameters) and Mesh (the chosen triangle's
// barycentric UV, the same interpolation the ray-hit path uses) — and left at 0
// elsewhere, since sphere / tube / spot / env emitters reject `emit pattern:` at load.
// Passing null (the default) keeps every existing caller's arithmetic untouched.
__device__ static void emitterSamplePoint(const DEmitter& em, double u1, double u2,
                                          DVec3& y, DVec3& nOut,
                                          double* uuOut = nullptr, double* vvOut = nullptr) {
    if (uuOut) *uuOut = 0.0;
    if (vvOut) *vvOut = 0.0;
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
    } else if (em.shape == 5) {
        // Mesh area light: pick a triangle with probability proportional to its area
        // (binary-search u1*area over the cumulative-area CDF), remap the leftover to a
        // fresh uniform, then sample the chosen triangle barycentrically (mirrors host
        // Emitter::samplePoint's Mesh branch). pdf = 1/area over the whole surface.
        double target = u1 * em.area;
        int lo = 0, hi = em.meshTriN;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (em.meshTris[mid].cumArea < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo >= em.meshTriN) lo = em.meshTriN - 1;
        const DEmitTri& t = em.meshTris[lo];
        double prev = (lo == 0) ? 0.0 : em.meshTris[lo - 1].cumArea;
        double span = t.cumArea - prev;
        double uu = (span > 0.0) ? (target - prev) / span : u1;
        double su = sqrt(fmax(0.0, uu));
        double b1 = 1.0 - su;
        double b2 = u2 * su;
        y = t.v0 + t.e1 * (Real)b1 + t.e2 * (Real)b2;
        nOut = t.nrm;
        // Same barycentric weights the ray-hit path uses, so a bound emission pattern
        // reads identically from either side of the transport.
        if (uuOut) *uuOut = t.uv0.x + t.uvE1.x * (Real)b1 + t.uvE2.x * (Real)b2;
        if (vvOut) *vvOut = t.uv0.y + t.uvE1.y * (Real)b1 + t.uvE2.y * (Real)b2;
    } else {
        y = em.origin + em.u * (Real)u1 + em.v * (Real)u2;
        nOut = em.normal;
        if (uuOut) *uuOut = u1;
        if (vvOut) *vvOut = u2;
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
    // Normalised illuminant baked over [DLMIN,DLMAX] (SPEC_N). The forward reweight
    // cancels the illuminant, but image-env NEE (bkNeeEnv) needs the ABSOLUTE radiance
    // L(dir,lambda) = scale*reflAt(coeff,lambda)*illum(lambda), so upload it.
    const double* illum = nullptr;        // SPEC_N : illumAt(lambda), non-null iff image env
    double avgCoeff[3] = {0, 0, 0};
    double avgScale = 0.0;
    const double* margCdf     = nullptr;  // h+1
    const double* margFunc    = nullptr;  // h
    double        margFuncInt = 0.0;
    const double* condCdf     = nullptr;  // h*(w+1) : row v at v*(w+1)
    const double* condFunc    = nullptr;  // h*w     : row v at v*w
    const double* condFuncInt = nullptr;  // h
};

// One scalar-record stop on the device (§records stage 6b): its domain position + a slice
// of the shared recDrivers PatNode pool holding the stop's per-hit expression program.
struct DRecScalarStop { double pos; int exprOff; int exprN; };

// One emissive ("fire") volume (device twin of Scene::EmissiveVolume, ROADMAP C3). A
// photon born inside carries beta = grandTotal*emissionAt(x,lambda)/(meanKe*dLam*p(lambda)),
// with the position uniform in [bmin,bmax] and lambda importance-sampled from `lamCdf`
// (a Planck-at-emitKelvin per-nm CDF, `lamN`+1 entries, step `lamStep`; pdf per nm =
// binMass/step). meanKe/power are MC-estimated on the host (finalizeEmissiveVolumes).
struct DEmissiveVolume {
    int    mediumIndex;
    DVec3  bmin, bmax;
    double meanKe;
    double power;
    const double* lamCdf;   // lamN+1 normalised CDF over [LAMBDA_MIN,LAMBDA_MAX]
    int           lamN;     // number of bins
    double        lamStep;  // bin width in nm
};

struct DScene {
    const DTri*      tris;  int nTris;
    const DSphere*   sph;   int nSph;
    // Flat mirror surfaces (Scene::mirrorPlanes, uploaded verbatim): one entry per
    // coplanar group of mirror-material triangles, capped at Scene::kMaxMirrorPlanes.
    // Read ONLY by the forward specular camera connection (dConnectSpecularPlane), which
    // is what stops mode B rendering every mirror black. Null/0 in a mirror-free scene,
    // so such scenes pay one integer compare per connected photon vertex.
    const DMirrorPlane* mirrorPlanes; int nMirrorPlanes;
    // Sphere index lists (Scene::dielSphereIdx / mirrorSphereIdx, ascending): the spheres
    // whose material is Dielectric / a planar-mirror metal. Read ONLY by the forward
    // specular camera connection (dCamSpecularSplatAllVtxN) so it can skip the
    // per-vertex-per-lambda scan of ALL spheres in scenes where few (or none) qualify.
    // Null/0 when the scene has no such spheres.
    const int* dielSph;   int nDielSph;
    const int* mirrorSph; int nMirrorSph;
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
    // FP32 mirrors of fieldNodes/fieldExprNodes (same order/offsets) for the sphere-
    // trace march hot path. Null iff the double pools are null.
    const DFieldNodeF* fieldNodesF;
    const PatNodeF*    fieldExprNodesF;
    const DImplicit*  implicits; int nImplicits;
    // Curve / fiber round cones (curve.h). BVH prims with index >= nTris+nSph+nImplicits
    // map to curveSegs[prim - nTris - nSph - nImplicits]. This range sits BEFORE instances
    // in exactly the order Scene::buildBvh pushes them (tris | spheres | implicits |
    // curveSegs | instances) — the host and device leaf dispatch must agree index for
    // index, so any new primitive range has to be inserted in all three places at once.
    const DCurveSeg*  curveSegs; int nCurveSegs;
    // Instancing (two-level BVH). BVH prims with index >= nTris+nSph+nImplicits+nCurveSegs
    // map to instances[prim - nTris - nSph - nImplicits - nCurveSegs]; each references a DBlas
    // (offsets into the shared blasNodes/blasTris/blasPrim pools). Null/0 when the scene
    // has no instances (the common path uploads Scene::bvh verbatim, bit-identical).
    const DInstance*  instances; int nInstances;
    const DBlas*      blas;                 // per-BLAS pool offsets, indexed by blasId
    const DNode*      blasNodes;            // concatenated per-BLAS BVH nodes (0-based)
    const int*        blasPrim;             // concatenated per-BLAS primIdx (0-based)
    const DTri*       blasTris;             // concatenated per-BLAS local-space tris
    // Procedural patterns (§4): flat postfix PatNode pool + per-pattern slices.
    // A material's roughnessPat/filmThicknessPat/mixWeightPat index `patterns`.
    const PatNode*   patNodes;
    const DPattern*  patterns; int nPatterns;
    // Parametric-record reflect binding (§records stage 6a). recCoeff: each driven reflect
    // channel's baked JH coeff LUT (REC_LUT_N*3 doubles per channel, sliced by
    // DMaterial::recReflOff); recDrivers: flat driver-program PatNode pool (sliced by
    // recReflDrvOff/recReflDrvN). Null when no material drives reflect per-hit off a record.
    // (Constant selStop bindings are baked into DMaterial::reflect[] and use neither pool.)
    const double*    recCoeff;
    const PatNode*   recDrivers;
    // Parametric-record ROUGHNESS scalar-stop table (§records stage 6b): each driven /
    // constant-selStop roughness binding's stops (pos + expr slice into recDrivers),
    // sliced by DMaterial::recRoughStopOff/recRoughStopN. Null when no scalar record is bound.
    const DRecScalarStop* recScalarStops;
    const DEmitter*  emitters; int nEmitters;
    const double*    emitCdf;       // size nEmitters, cumulative power, normalised
    double           totalPower;
    // Conty-Kulla light BVH over the emitters (lighttree.h), uploaded VERBATIM from
    // Scene::lightTree — LightTreeNode is POD holding only doubles and child INDICES, and
    // `dems` is built 1:1 with scene.emitters, so a leaf's `emitter` field indexes this
    // DEmitter array directly with no remap. lightTreeRoot < 0 => no tree (fewer than two
    // tree-eligible emitters, or the host build declined), and the device then falls back
    // to the exact all-emitters loop. `lightTreeAlways` lists emitters with no usable
    // spatial bound (distant suns) that are connected unconditionally.
    const LightTreeNode* lightTree; int lightTreeRoot;
    const int*       lightTreeAlways; int nLightTreeAlways;
    int              bkLightTree;    // 0 = -no-lighttree: exact all-emitters splitting
    double           bkLightSplit;   // -light-split
    int              bkLightSamples; // -light-samples
    const double*    lightCdfAll;   // flattened per-emitter wavelength CDFs
    const DTexture*  textures; int nTex;   // reflectance textures (mat.reflectTex)
    // N-D data tables (§grids), reached from a pattern as `grid:<name>(c0, …)` (regular
    // lattice) or `scatter:<name>(c0, …)` (ragged). Uploaded VERBATIM: the host headers
    // are already POD that refer to their numbers by OFFSET (never a pointer), and
    // dataPool is the same flat float run the host reads, so patGridSample /
    // patScatterSample — the shared __host__ __device__ samplers in pattern.h — run here
    // unchanged and the two backends agree bit-for-bit.
    const PatGrid*    grids;    int nGrids;
    const PatScatter* scatters; int nScatters;
    const float*      dataPool; int dataPoolN;
    const double*    fluoCdfAll;    // flattened per-material fluorescence emission CDFs
    // BDPT shared wavelength sampler (mirrors Scene::emitSampler): the combined
    // g(lambda)=sum_k geomWeight_k*SPD_k CDF, its bin step, and emitG = its integral.
    // BDPT samples one shared lambda per sample from this and sets invPdfLambda=1/pdf.
    const double*    emitSamplerCdf; int emitSamplerN; double emitSamplerStep;
    double           emitG;
    const DMedium*   media;    // participating media array (superposed); null if none
    int              mediaN;   // number of media (0 => vacuum)
    // Volumetric blackbody emission ("fire", ROADMAP C3). Photon birth splits emitter-
    // vs-fire by power: grandTotal = totalPower + totalEmissionPower. Null/0 => no fire.
    const DEmissiveVolume* emissiveVolumes; int emissiveVolN;
    double           totalEmissionPower;
    int              hasGrin;  // 1 => some enabled medium carries an `ior` (GRIN) field.
                               // Gates the per-bounce Eikonal march (grin::sceneHasGrin twin);
                               // 0 keeps ordinary scenes bit-identical (march never entered).
    DVec3  sensorOrigin, sensorUAxis, sensorVAxis;   // model A contact sensor plane
    DVec3  sceneCenter;              // env (shape==3): bounding-sphere center
    double sceneRadius;              // env (shape==3): bounding-sphere radius
    // `cavity` pattern variable (O3 stage 2) — twins of Scene::needsCavity /
    // cavityRadius / cavitySamples. needsCavity is the load-time "does any material
    // actually read `cavity`?" gate: it is 0 for essentially every scene, and every
    // probe site tests it first, so a scene that never says `cavity` pays one integer
    // compare per shading point and fires no probe rays at all.
    int    needsCavity;
    double cavityRadius;
    int    cavitySamples;
    DEnvMap env;                     // image env tables (env.scale null => constant env)
    int    envIndex;                 // index of the env emitter in `emitters`, or -1 (mirrors Scene::envIndex)
    int    sunCount;                 // number of shape==6 (distant sun) emitters (mirrors Scene::sunCount);
                                     // every sun-aware hot path tests this first, so a scene
                                     // without a sun pays one integer compare
    DVec3  rgbEnv;                    // fast RGB backward: constant-env radiance in linear sRGB (0 if no env)
    // Scene-ignore render params (Stage 3), set by renderBackward[RGB]Cuda from the CLI
    // flags. bkMaxBounce caps the backward path-depth loop (default 32). bkDirectOnly=1
    // renders direct lighting + specular recursion only (no diffuse indirect) — a
    // Whitted-style near-1-spp preview; 0 = full path tracing.
    int    bkMaxBounce;
    int    bkDirectOnly;
    // Mode W (deterministic Whitted/POV-Ray preview) knobs — the device twin of
    // BackwardRenderer's fields in src/backward.h. bkWhitted=1 replaces every stochastic
    // estimator on the path with fixed quadrature; the rest are only read when it is set.
    // bkGrid / bkGiGrid: the N of the N*N area-light NEE lattice at a primary / gather
    // vertex. bkGiDirs: deterministic one-bounce gather ray count (0 = off, use bkAmbient
    // only). bkGiBounce: path-depth cap on a gather ray. bkGiClamp: per-wavelength firefly
    // ceiling on one gather ray's returned radiance, 0 = off (twin of
    // BackwardRenderer::giClamp — the caustic-through-the-gather aliasing fix; see the long
    // comment there). bkAmbient: flat fill added at each diffuse vertex. bkGiClamp and
    // bkAmbient are both already pre-scaled by Scene::ambientRef() on the host.
    // bkHeroSplit is the ONE knob here that is NOT mode-W-only: it fans the hero bundle into
    // monochromatic sub-paths at a dispersive vertex (bkRadianceHeroLoop<true>), and plain
    // mode R takes it from `-herosplit` while mode W forces it on — see design.md.
    int    bkWhitted;
    int    bkGrid;
    int    bkGiDirs;
    int    bkGiGrid;
    int    bkGiBounce;
    int    bkHeroSplit;
    double bkAmbient;
    double bkGiClamp;
    // O8 stage 2: the camera's footprint coefficient — the world-space patch diameter one
    // shading sample stands for AT UNIT DISTANCE, head-on (Camera::footprintPerDist(spp)).
    // The kernel turns it into a per-hit `fw` at the camera segment via
    // patShadingFootprint(). Device twin of BackwardRenderer::fwPerDist, and set for the
    // same narrow case: mode W only. 0 = unknown, so `fw` stays 0 and fnoise() runs
    // unfiltered — which is what stochastic mode R wants, since jittering over the pixel
    // already averages the footprint.
    double bkFwPerDist;
};

// Everything the pattern VM (dPatternEval) needs beyond the scalar variables: the
// SCENE-OWNED sample tables a pattern expression can reach into. The struct itself is
// DPatEnvT<TexT> in pattern_device.cuh (shared with the preview rasterizer, which stores
// its textures differently); this backend's instantiation samples spectral DTextures.
//
// dPatEnvNone() is the OUT-OF-SCOPE environment used at value sites the host compiler
// already refuses `tex:`/`grid:` at (implicit field formulas, medium density/ior), so
// such a node can never actually appear there; the null tables just make the VM
// total instead of undefined if one ever did.
using DPatEnv = DPatEnvT<DTexture>;
__host__ __device__ static inline DPatEnv dPatEnvNone() { return dPatEnvNoneT<DTexture>(); }
__host__ __device__ static inline DPatEnv dPatEnvOf(const DScene& sc) {
    DPatEnv e; e.tex = sc.textures; e.nTex = sc.nTex;
    e.grids = sc.grids; e.nGrids = sc.nGrids;
    e.scatters = sc.scatters; e.nScatters = sc.nScatters;
    e.dataPool = sc.dataPool; e.dataPoolN = sc.dataPoolN;
    return e;
}

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
    double frustumShiftX;   // off-axis stereo shear (normalised view units); 0 = on-axis
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
            Real ix = (cx / cz) / (Real)tanHalfX - (Real)frustumShiftX, iy = (cy / cz) / (Real)tanHalfY;
            if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
            px = (int)((ix * (Real)0.5 + (Real)0.5) * resX);
            py = (int)((iy * (Real)0.5 + (Real)0.5) * resY);
            // FP32 rounding at the film edge can push (ix*0.5+0.5)*res up to exactly
            // res, yielding px==resX/py==resY and an out-of-bounds film write. The
            // ix/iy<1 rejection above guarantees the point is on-film, so clamp the
            // boundary case back to the last valid pixel. (CPU project uses double and
            // never rounds up this way, so it needs no clamp — behaviour still matches.)
            px = px < 0 ? 0 : (px >= resX ? resX - 1 : px);
            py = py < 0 ? 0 : (py >= resY ? resY - 1 : py);
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
        px = px < 0 ? 0 : (px >= resX ? resX - 1 : px);   // clamp FP32 edge roundup
        py = py < 0 ? 0 : (py >= resY ? resY - 1 : py);
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
        Real ix = -dot(rel, u) / (Real)(filmDist * tanHalfX) - (Real)frustumShiftX;
        Real iy = -dot(rel, v) / (Real)(filmDist * tanHalfY);
        if (ix < -1 || ix >= 1 || iy < -1 || iy >= 1) return false;
        px = (int)((ix * (Real)0.5 + (Real)0.5) * resX);
        py = (int)((iy * (Real)0.5 + (Real)0.5) * resY);
        px = px < 0 ? 0 : (px >= resX ? resX - 1 : px);   // clamp FP32 edge roundup
        py = py < 0 ? 0 : (py >= resY ? resY - 1 : py);
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
// Power-cosine lobe around a mirror direction, from two CANONICAL uniforms rather than an
// rng — the device twin of glossyDirUV (src/render.h), split out of sampleGlossy for the
// same reason: mode W has no rng to draw from without reintroducing noise, so it drives the
// identical lobe off a low-discrepancy lattice. `cosT = u1^(1/(e+1))` makes **u1 == 1
// exactly the mirror direction**, which is why the deterministic caller complements its
// sequence instead of rotating it, and why u1 >= 1 returns mdir verbatim (skipping the
// normalize(), whose last-bit rescale would spoil that identity).
__device__ static DVec3 glossyDirUV(const DVec3& mdir, Real roughness, Real u1, Real u2) {
    if (u1 >= (Real)1) return mdir;          // exact mirror (only a deterministic caller
                                             // reaches this; rng.uniform() is [0,1))
    Real rr = roughness < (Real)1e-3 ? (Real)1e-3 : roughness;
    Real e = (Real)2 / (rr * rr) - (Real)2; if (e < 0) e = 0;
    Real cosT = pow(u1, (Real)1 / (e + (Real)1));
    Real sinT = sqrt(fmax((Real)0, (Real)1 - cosT * cosT));
    Real phi = (Real)2 * (Real)DPI * u2;
    DVec3 t, b; onb(mdir, t, b);
    return normalize(t * (sinT * cos(phi)) + b * (sinT * sin(phi)) + mdir * cosT);
}
__device__ static DVec3 sampleGlossy(const DVec3& mdir, Real roughness, DRng& rng) {
    // Sequenced into locals deliberately: passing rng.uniform() twice as arguments would
    // leave the draw order unspecified and desynchronise the stream.
    Real u1 = rng.uniform(), u2 = rng.uniform();
    return glossyDirUV(mdir, roughness, u1, u2);
}

// ---------------- mode W: deterministic sample placement (device twin) ----------------
// Exact ports of BackwardRenderer's statics in src/backward.h. All of these run in
// `double` regardless of FTRACE_GPU_FP32, and on pure integer / double arithmetic, so they
// are bit-identical to the host versions — which is what N4's part (a) tests directly.
// See backward.h for the full rationale; the load-bearing invariants are (1) every pixel
// uses the SAME offsets (that is what makes the mode noise-free) and (2) the sequences are
// indexed by the ABSOLUTE sample index, so the image is chunk-split-independent.
__device__ static double dRadicalInverse2(unsigned long long i) {
    i = (i << 32) | (i >> 32);
    i = ((i & 0x0000ffff0000ffffULL) << 16) | ((i & 0xffff0000ffff0000ULL) >> 16);
    i = ((i & 0x00ff00ff00ff00ffULL) <<  8) | ((i & 0xff00ff00ff00ff00ULL) >>  8);
    i = ((i & 0x0f0f0f0f0f0f0f0fULL) <<  4) | ((i & 0xf0f0f0f0f0f0f0f0ULL) >>  4);
    i = ((i & 0x3333333333333333ULL) <<  2) | ((i & 0xccccccccccccccccULL) >>  2);
    i = ((i & 0x5555555555555555ULL) <<  1) | ((i & 0xaaaaaaaaaaaaaaaaULL) >>  1);
    return (double)i * (1.0 / 18446744073709551616.0);
}
// DIGIT-SCRAMBLED radical inverse (Faure's fix for high-dimensional Halton). A plain radical
// inverse in base b returns exactly i/b for i < b, so its first N points cover only the prefix
// [0, N/b) -- and every base below is larger than a typical preview's `-spp`. Permuting the
// digits, r = Σ π(dₖ) b^-(k+1), keeps the same b-point grid (so the same discrepancy) but visits
// it scattered instead of monotone. π(d) = (d·m) mod b with m ≈ b/φ needs no tables, is a
// bijection for prime b, and has π(0) = 0 -- so sIdx 0 still maps to exactly 0 in every base and
// every "sample 0 is the canonical outcome" contract below is untouched.
// (Host twin: BackwardRenderer::radicalInverseScr / goldenDigitMul. Must stay bit-identical.)
//
// BOTH multiply-adds below are spelled with __dmul_rn / __dadd_rn rather than `a * b + c`,
// and that is load-bearing, not style. nvcc contracts a multiply feeding an add into a
// single FMA by default (-fmad=true); MSVC does not. FMA is the *more* accurate of the two
// — it drops the intermediate rounding — but "more accurate" is not the contract here:
// mode W has no Monte-Carlo noise to absorb a difference, so the host is the reference and
// the device must reproduce its rounding exactly. Contracted, the digit loop disagreed with
// the CPU by 1 ULP on ~1.6% of indices (caught by `-checklattice`, N4a); the rn intrinsics
// are individually-rounded operations the compiler is not permitted to fuse, which pins the
// evaluation order to the host's without touching any other kernel's codegen.
__device__ static unsigned dGoldenDigitMul(unsigned base) {
    unsigned m = (unsigned)__dadd_rn(__dmul_rn((double)base, 0.6180339887498949), 0.5);
    return (m == 0u || m >= base) ? 1u : m;
}
__device__ static double dRadicalInverseScr(unsigned base, unsigned long long i) {
    const unsigned mul = dGoldenDigitMul(base);
    const double invB = 1.0 / (double)base;
    double f = invB, r = 0.0;
    while (i) {
        r = __dadd_rn(r, __dmul_rn((double)((unsigned)(i % base) * mul % base), f));
        i /= base; f *= invB;
    }
    return r;
}
__device__ static double dRot05(double x) { x += 0.5; return (x >= 1.0) ? x - 1.0 : x; }
__device__ static void dWhittedSample(unsigned long long idx, double& u, double& v) {
    u = dRot05(dRadicalInverse2(idx));
    v = dRot05(dRadicalInverseScr(3, idx));
}
__device__ static double dWhittedLambdaU(unsigned long long idx) {
    return dRot05(dRadicalInverseScr(5, idx));
}
// One direction of the deterministic one-bounce-gather lattice: point `j` of an `n`-point
// Fibonacci spiral on the WHOLE sphere, Cranley-Patterson-rotated by (p1, p2).
// (Host twin: BackwardRenderer::giDir. Must stay bit-identical, so every intermediate is
// `double` whatever `Real` is, and the golden angle is spelled to the same 16 digits.)
//
// The lattice is built in WORLD space and the caller keeps the ~half of it with cos > 0,
// weighting by cos and normalising by the realised sum. That beats a cosine-weighted lattice
// in a local frame for two reasons: no tangent frame is needed, so there is no
// orthonormal-basis discontinuity to appear as a seam; and a direction entering or leaving
// the hemisphere does so at cos == 0, i.e. with zero weight, so the estimate is continuous
// in the normal — which is what makes a rotating object's shading slide instead of pop.
__device__ static DVec3 dGiDir(int j, int n, double p1, double p2) {
    double t = ((double)j + 0.5) / (double)n + p2;
    t -= floor(t);                                   // CP rotation of the z-strata
    const double z = 1.0 - 2.0 * t;
    const double r = sqrt(fmax(0.0, 1.0 - z * z));
    const double kGolden = 2.399963229728653;        // pi * (3 - sqrt 5)
    const double a = kGolden * (double)j + 2.0 * DPI * p1;
    return DVec3(r * cos(a), r * sin(a), z);
}
// The two Cranley-Patterson phases of the gather lattice, from the ABSOLUTE sample index, on
// two decorrelated scrambled radical inverses. Bases 7 and 11 collide with neither the
// subpixel lattice (2, 3), the wavelength lattice (5), nor the glossy/discrete lattices
// (>= 13). Every pixel shares them — the invariant that makes this mode noise-free — so
// raising -spp rotates the whole frame's lattice coherently and the banding averages out.
__device__ static void dGiPhases(unsigned long long sIdx, double& p1, double& p2) {
    p1 = dRot05(dRadicalInverseScr(7, sIdx));
    p2 = dRot05(dRadicalInverseScr(11, sIdx));
}
// Deterministic rough-specular direction: point `sIdx` of a fixed 2-D lattice on the
// power-cosine lobe. The polar coordinate is COMPLEMENTED (not rot05'd) so sample 0 is
// exactly the mirror direction, since dRadicalInverseScr(b, 0) == 0 in every base. Each
// bounce depth takes its own prime pair so two glossy vertices on one path are not driven
// by the same 1-D sequence. Bases 2/3 are the subpixel lattice, 5 the wavelength, 7/11 the
// gather, so these start at 13.
// The two lattice COORDINATES are factored out of the direction (host twin:
// BackwardRenderer::whittedGlossyUV) so `-checklattice` can probe exactly the numbers the
// kernel uses. They are pure integer-and-`double` arithmetic and so bit-comparable with the
// host; the direction itself goes through glossyDirUV's trig in `Real` and is not.
__device__ static void dWhittedGlossyUV(unsigned long long sIdx, int bounce,
                                        double& u1, double& u2) {
    const unsigned kBases[4][2] = {{13, 17}, {19, 23}, {29, 31}, {37, 41}};
    const unsigned b0 = kBases[bounce & 3][0], b1 = kBases[bounce & 3][1];
    u1 = 1.0 - dRadicalInverseScr(b0, sIdx);   // 1 at sIdx 0 => mirror
    u2 = dRadicalInverseScr(b1, sIdx);
}
__device__ static DVec3 dWhittedGlossyDir(const DVec3& mdir, Real roughness,
                                          unsigned long long sIdx, int bounce) {
    double u1, u2;
    dWhittedGlossyUV(sIdx, bounce, u1, u2);
    return glossyDirUV(mdir, roughness, (Real)u1, (Real)u2);
}
// Deterministic DISCRETE-CHOICE coordinate: one scalar off the (sIdx, bounce) lattice for a
// pick out of a finite weighted set, as opposed to a direction on a lobe. Not rot05'd, so
// u == 0 at sIdx 0 selects the FIRST candidate of the caller's traversal -- which
// gratingDiffract orders by descending efficiency, making sample 0 the specular order.
// (Host twin: BackwardRenderer::whittedOrderU. Must stay bit-identical.)
__device__ static double dWhittedOrderU(unsigned long long sIdx, int bounce) {
    const unsigned kBases[4] = {43, 47, 53, 59};
    return dRadicalInverseScr(kBases[bounce & 3], sIdx);
}
// Deterministic Stokes-shift excitation-wavelength coordinate. Rot05'd like the other
// wavelength lattices -- there is no "specular" outcome to prefer, so sample 0 should land on
// the MEDIAN of the excitation CDF rather than its short-λ extreme.
// (Host twin: BackwardRenderer::whittedFluoroU.)
__device__ static double dWhittedFluoroU(unsigned long long sIdx, int bounce) {
    const unsigned kBases[4] = {61, 67, 71, 73};
    return dRot05(dRadicalInverseScr(kBases[bounce & 3], sIdx));
}
// Replace a Russian-roulette survival test with a throughput WEIGHT: same expected value,
// zero variance. False once the path is too dim to matter — POV-Ray's `adc_bailout`.
static constexpr double kWhittedCutoff = 1.0 / 512.0;   // POV-Ray's default adc_bailout is
                                                        // 1/255; backward.h uses 1/512
__device__ static bool dWhittedAttenuate(double& thr, double w) {
    if (w <= 0.0) return false;
    thr *= w;
    return thr > kWhittedCutoff;
}
// Centre of cell (g%G, g/G) of a G x G lattice over an emitter's [0,1)^2 sample domain —
// POV-Ray's `area_light`: a fixed set of shadow rays whose average is a soft shadow with
// NO variance, rather than one random point whose average only smooths out over many spp.
__device__ static void dGridUV(int g, int G, Real& u1, Real& u2) {
    u1 = (Real)(((double)(g % G) + 0.5) / (double)G);
    u2 = (Real)(((double)(g / G) + 0.5) / (double)G);
}
// ---- N4a: the device half of the lattice bit-exactness probe ---------------
// Evaluates every helper above at one index per thread and writes a fixed-width row of raw
// doubles. See render_cuda.h for the column layout; the host compares the bits directly.
// Deliberately calls the SAME functions the megakernel does — a re-implementation here
// would test the copy, not the kernel.
// `bases` is kLatticeProbeBases uploaded as a plain device buffer rather than a
// `__constant__` symbol, so the probe needs no API beyond the handful the HIP shim at the
// top of this file already covers — and the base list stays single-sourced in the header.
__global__ void kLatticeProbe(const unsigned long long* idx, int n,
                              const unsigned* bases, double* out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const unsigned long long i = idx[t];
    const int b = (int)(i & 3ull);
    double* r = out + (size_t)t * kLatticeProbeCols;
    const double ri2 = dRadicalInverse2(i);
    r[0] = ri2;
    r[1] = dRot05(ri2);
    dWhittedSample(i, r[2], r[3]);
    r[4] = dWhittedLambdaU(i);
    r[5] = dWhittedOrderU(i, b);
    r[6] = dWhittedFluoroU(i, b);
    dWhittedGlossyUV(i, b, r[7], r[8]);
    dGiPhases(i, r[9], r[10]);
    // gridUV takes an `int` cell index and writes `Real`; widen back so the host can compare
    // against its own (Real)-narrowed value. G varies with the index so the probe covers
    // several lattice sizes rather than one.
    const int G = 4 + (int)(i % 5ull);
    Real gu1, gu2;
    dGridUV((int)(i % (unsigned long long)(G * G)), G, gu1, gu2);
    r[11] = (double)gu1;
    r[12] = (double)gu2;
    for (int k = 0; k < kLatticeProbeNBases; ++k)
        r[13 + k] = dRadicalInverseScr(bases[k], i);
}
// `sizeof(Real)` for the host half of the probe, which has no view of this file's typedef.
// (The public `cudaRealBytes` wrapper is with the rest of the host API, past the namespace.)
inline int realBytes() { return (int)sizeof(Real); }
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

// --- Medium phase dispatch: analytic HG lobe vs. tabulated Airy rainbow ---------------
// A rainbow medium (rbPdf != null) carries a per-medium (lambda x mu) phase table plus a
// per-lambda CDF (mirrors host rainbow::RainbowPhase); HG media fall through to the
// analytic lobe. dMedPhase returns the solid-angle phase value p(cos) at wavelength
// lambda, which equals the sampling pdf of dMedPhaseSample for BOTH models. Table math is
// done in double to track the CPU tracer bit-closely.
__device__ static Real dRbEval(const DMedium& m, Real cosTheta, Real lambda) {
    double fl = ((double)lambda - m.rbLam0) / m.rbDLam;
    int li = (int)floor(fl); double tl = fl - li;
    if (li < 0) { li = 0; tl = 0.0; }
    if (li > m.rbNLam - 2) { li = m.rbNLam - 2; tl = 1.0; }
    if (tl < 0.0) tl = 0.0; else if (tl > 1.0) tl = 1.0;
    double dMu = 2.0 / (m.rbNMu - 1);
    double fm = ((double)cosTheta + 1.0) / dMu;
    int mi = (int)floor(fm); double tm = fm - mi;
    if (mi < 0) { mi = 0; tm = 0.0; }
    if (mi > m.rbNMu - 2) { mi = m.rbNMu - 2; tm = 1.0; }
    if (tm < 0.0) tm = 0.0; else if (tm > 1.0) tm = 1.0;
    const double* P = m.rbPdf; int nMu = m.rbNMu;
    double a = P[(size_t)li * nMu + mi] * (1 - tm) + P[(size_t)li * nMu + mi + 1] * tm;
    double b = P[(size_t)(li + 1) * nMu + mi] * (1 - tm) + P[(size_t)(li + 1) * nMu + mi + 1] * tm;
    return (Real)(a * (1 - tl) + b * tl);
}
__device__ static Real dMedPhase(const DMedium& m, Real cosTheta, Real lambda) {
    if (m.rbPdf) return dRbEval(m, cosTheta, lambda);
    return hgPhase(cosTheta, (Real)m.g);
}
// Importance-sample a scattered direction about propagation `wi` at wavelength lambda;
// sets pdfOut = p(cos) of the chosen direction. Mirrors host Medium::phaseSample.
__device__ static DVec3 dMedPhaseSample(const DMedium& m, const DVec3& wi, Real lambda, DRng& rng, Real& pdfOut) {
    if (m.rbPdf) {
        int li = (int)lround(((double)lambda - m.rbLam0) / m.rbDLam);
        if (li < 0) li = 0;
        if (li > m.rbNLam - 1) li = m.rbNLam - 1;
        const double* cdf = &m.rbCdf[(size_t)li * m.rbNMu];
        double u = (double)rng.uniform();
        int lo = 0, hi = m.rbNMu - 1;
        while (lo + 1 < hi) { int mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid; else hi = mid; }
        double c0 = cdf[lo], c1 = cdf[hi];
        double t = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.0;
        double dMu = 2.0 / (m.rbNMu - 1);
        double mu = -1.0 + (lo + t) * dMu;
        if (mu < -1.0) mu = -1.0; else if (mu > 1.0) mu = 1.0;
        Real sinT = sqrt(fmax((Real)0, (Real)1 - (Real)(mu * mu)));
        Real phi = (Real)2 * (Real)DPI * rng.uniform();
        DVec3 tb, bb; onb(wi, tb, bb);
        DVec3 dir = normalize(tb * (sinT * cos(phi)) + bb * (sinT * sin(phi)) + wi * (Real)mu);
        pdfOut = dRbEval(m, (Real)mu, lambda);
        return dir;
    }
    DVec3 d = sampleHG(wi, (Real)m.g, rng);
    pdfOut = hgPhase(dot(wi, d), (Real)m.g);
    return d;
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

// Linear-sRGB (D65) -> CIE XYZ. Inverse of color.h xyzToLinearSrgb; used by the fast
// RGB backward path (bkRadianceRGB) to deposit its accumulated linear-RGB radiance into
// the XYZ film (round-trips an unclamped emitter colour exactly, so a neutral scene
// matches the spectral estimator's absolute luminance).
__device__ static inline DVec3 dRgbToXyz(const DVec3& c) {
    return DVec3(
        (Real)(0.4124 * c.x + 0.3576 * c.y + 0.1805 * c.z),
        (Real)(0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z),
        (Real)(0.0193 * c.x + 0.1192 * c.y + 0.9505 * c.z));
}
// Componentwise clamp of an RGB triple to [0,1] (baked reflectance albedos).
HD static inline DVec3 clampRgb01(const DVec3& c) {
    return DVec3(clamp01((Real)c.x), clamp01((Real)c.y), clamp01((Real)c.z));
}
// Luminance of a linear-sRGB triple (Rec.709 / sRGB Y). Used as the diffuse
// continuation survival probability (RGB Russian roulette) in bkRadianceRGB.
HD static inline Real rgbLuma(const DVec3& c) {
    return (Real)(0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z);
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
__device__ static inline double dSunRadiance(const DScene& sc, const DVec3& d, Real lambda) {
    if (sc.sunCount == 0) return 0.0;
    double L = 0.0;
    for (int k = 0; k < sc.nEmitters; ++k) {
        const DEmitter& e = sc.emitters[k];
        if (e.shape == 6 && dInSunCone(e, d)) L += (double)specLookup(e.emitSpd, lambda);
    }
    return L;
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

// dFieldEval is defined further down; forward-declare for the density evaluator (the
// implicit-bound membership test needs the field VM). dPatternEval needs no forward
// declaration: it is the template from pattern_device.cuh, included at the top.
__device__ static double dFieldEval(const DFieldNode* nodes, int n,
                                    double pwx, double pwy, double pwz,
                                    const PatNode* exprPool, const DPatEnv& env);

// IEEE-754 binary16 -> binary32 (device twin of halfBitsToFloat in vdbgrid.h).
// The uploaded VDB lattice is fp16; this decodes it in the hot density sampler.
// Portable bit math (no cuda_fp16 dependency, HIP-safe).
__device__ static inline float dHalfBitsToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {
            exp = 1;
            while ((man & 0x400u) == 0) { man <<= 1; --exp; }
            man &= 0x3FFu;
            bits = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (man << 13);
    }
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}

// Trilinearly sample a sparse brick grid at a world point (>= 0; 0 outside the lattice).
// Device twin of VdbGrid::sample — bit-for-bit with the host (stencil clamped to
// [0,n-1]). Shared by the density multiplier and the emissive temperature field.
__device__ static double dVdbSample(const DVdbGrid& g, const DVec3& p) {
    double rx = (double)p.x - g.w0.x, ry = (double)p.y - g.w0.y, rz = (double)p.z - g.w0.z;
    double fi = g.ainv[0]*rx + g.ainv[1]*ry + g.ainv[2]*rz - g.imin.x;
    double fj = g.ainv[3]*rx + g.ainv[4]*ry + g.ainv[5]*rz - g.imin.y;
    double fk = g.ainv[6]*rx + g.ainv[7]*ry + g.ainv[8]*rz - g.imin.z;
    int nx = g.nx, ny = g.ny, nz = g.nz;
    if (fi < -0.5 || fj < -0.5 || fk < -0.5 ||
        fi > nx - 0.5 || fj > ny - 0.5 || fk > nz - 0.5) return 0.0;
    // Clamp the COORDINATE (not just the stencil indices) — see VdbGrid::sample.
    auto cld = [](double v, double hi) { return v < 0.0 ? 0.0 : (v > hi ? hi : v); };
    double ci = cld(fi, nx-1), cj = cld(fj, ny-1), ck = cld(fk, nz-1);
    int i0c = (int)ci, j0c = (int)cj, k0c = (int)ck;   // ci >= 0, so trunc == floor
    int i1c = i0c + 1 < nx ? i0c + 1 : nx - 1;
    int j1c = j0c + 1 < ny ? j0c + 1 : ny - 1;
    int k1c = k0c + 1 < nz ? k0c + 1 : nz - 1;
    double tx = ci - i0c, ty = cj - j0c, tz = ck - k0c;
    const int32_t*  BI = g.brickIndex;
    const uint16_t* BD = g.brickData;
    const int  sh = g.brickShift, mask = g.brickB - 1;
    const int  bxN = g.bx, byN = g.by;
    const size_t B3 = (size_t)g.brickB * g.brickB * g.brickB;
    auto AT = [&](int i, int j, int k) -> double {
        int slot = BI[(((size_t)(k>>sh))*byN + (j>>sh))*bxN + (i>>sh)];
        if (slot < 0) return 0.0;
        int li = i & mask, lj = j & mask, lk = k & mask;
        return (double)dHalfBitsToFloat(BD[(size_t)slot*B3 + ((size_t)lk*g.brickB + lj)*g.brickB + li]);
    };
    double c00 = AT(i0c,j0c,k0c)*(1-tx) + AT(i1c,j0c,k0c)*tx;
    double c10 = AT(i0c,j1c,k0c)*(1-tx) + AT(i1c,j1c,k0c)*tx;
    double c01 = AT(i0c,j0c,k1c)*(1-tx) + AT(i1c,j0c,k1c)*tx;
    double c11 = AT(i0c,j1c,k1c)*(1-tx) + AT(i1c,j1c,k1c)*tx;
    double c0 = c00*(1-ty) + c10*ty, c1 = c01*(1-ty) + c11*ty;
    double v = c0*(1-tz) + c1*tz;
    return v > 0.0 ? v : 0.0;
}

// Dimensionless density multiplier at a world point (>= 0). Device twin of
// Medium::densityAt: the shared pattern VM with x y z r live (f/normal/uv read 0).
// For an implicit bound the multiplier is 0 outside the field (medium absent there).
__device__ static double dMedDensityAt(const DMedium& m, const DVec3& p, const DPatEnv& env) {
    if (m.boundShape == 2 && m.boundField) {   // implicit-field membership carve-out
        double f = dFieldEval(m.boundField, m.boundFieldN, p.x, p.y, p.z, m.boundFieldExpr, env);
        bool inside = m.boundInsideNeg ? (f < 0.0) : (f > 0.0);
        if (!inside) return 0.0;
    }
    if (m.boundShape == 3 && m.boundGrid.brickData) {   // mesh-containment carve-out
        if (dVdbSample(m.boundGrid, p) < 0.5) return 0.0;
    }
    // Imported .nvdb/.vdb volume: trilinearly sample the uploaded sparse brick grid.
    // Takes precedence over the pattern-VM density.
    if (m.densGrid.brickData) return dVdbSample(m.densGrid, p);
    if (!m.heterogeneous || !m.density) return 1.0;
    double r = sqrt((double)p.x * p.x + (double)p.y * p.y + (double)p.z * p.z);
    double d = dPatternEval(m.density, m.densityN, p.x, p.y, p.z, 0.0,
                            0.0, 0.0, 0.0, r, 0.0, 0.0, 0.0, 0.0, 0.0, env);
    return d > 0.0 ? d : 0.0;
}

// Planck spectral radiance at temperature `kelvin`, wavelength `lambdaNm` (device twin
// of blackbodyRadiance, spectrum.h). Kept in double so the exp stays accurate.
__device__ static double dBlackbodyRadiance(double kelvin, double lambdaNm) {
    const double h = 6.62607015e-34, c = 2.99792458e8, kb = 1.380649e-23;
    double l = lambdaNm * 1e-9;
    double e = exp((h * c) / (l * kb * kelvin)) - 1.0;
    return (2.0 * h * c * c) / (pow(l, 5.0) * e);
}
// Planck radiance normalised against the fixed 6500 K / 560 nm reference (device twin
// of blackbodyEmissionRadiance). kRef is a compile-time-derivable constant here.
__device__ static double dBlackbodyEmissionRadiance(double kelvin, double lambdaNm) {
    const double kRef = dBlackbodyRadiance(6500.0, 560.0);   // matches host kRef exactly
    return dBlackbodyRadiance(kelvin, lambdaNm) / kRef;
}
// Physical temperature (Kelvin) at a world point; 0 outside the grid / cold. Device twin
// of Medium::temperatureAt (peak-normalised T = emitKelvin * raw/tempPeak).
__device__ static double dMedTemperatureAt(const DMedium& m, const DVec3& p) {
    if (!m.emissive || !m.tempGrid.brickData) return 0.0;
    double raw = dVdbSample(m.tempGrid, p);
    if (raw <= 0.0) return 0.0;
    return m.emitKelvin * (raw / (m.tempPeak > 0.0 ? m.tempPeak : 1.0));
}
// Volumetric emission SOURCE radiance L_e(x,lambda) (>= 0). Device twin of
// Medium::emissionAt: emissionScale * blackbodyEmissionRadiance(T(x), lambda).
__device__ static double dMedEmissionAt(const DMedium& m, const DVec3& p, double lambda) {
    double T = dMedTemperatureAt(m, p);
    if (T <= 0.0) return 0.0;
    return m.emissionScale * dBlackbodyEmissionRadiance(T, lambda);
}

// Clip ray (o + t*dir, t in [t0,t1]) to the medium bound. Device twin of
// Medium::clipToBounds: returns the sub-interval [ta,tb] inside the box, or false on a
// miss. Unbounded media pass the interval through unchanged.
__device__ static bool dMedClip(const DMedium& m, const DVec3& o, const DVec3& dir,
                                 double t0, double t1, double& ta, double& tb) {
    if (!m.bounded) { ta = t0; tb = t1; return t1 > t0; }
    if (m.boundShape == 1) {   // sphere region: ray∩sphere chord ∩ [t0,t1]
        double ocx = (double)o.x - (double)m.bcenter.x;
        double ocy = (double)o.y - (double)m.bcenter.y;
        double ocz = (double)o.z - (double)m.bcenter.z;
        double dx = (double)dir.x, dy = (double)dir.y, dz = (double)dir.z;
        double A = dx * dx + dy * dy + dz * dz;
        double B = 2.0 * (ocx * dx + ocy * dy + ocz * dz);
        double C = ocx * ocx + ocy * ocy + ocz * ocz - m.bradius * m.bradius;
        double disc = B * B - 4.0 * A * C;
        if (disc <= 0.0 || A <= 0.0) return false;
        double sd = sqrt(disc);
        double s0 = (-B - sd) / (2.0 * A), s1 = (-B + sd) / (2.0 * A);
        double lo = fmax(t0, s0), hi = fmin(t1, s1);
        if (lo > hi) return false;
        ta = lo; tb = hi; return tb > ta;
    }
    double lo = t0, hi = t1;
    const double oo[3] = { (double)o.x, (double)o.y, (double)o.z };
    const double dd[3] = { (double)dir.x, (double)dir.y, (double)dir.z };
    const double mn[3] = { (double)m.bmin.x, (double)m.bmin.y, (double)m.bmin.z };
    const double mx[3] = { (double)m.bmax.x, (double)m.bmax.y, (double)m.bmax.z };
    for (int a = 0; a < 3; ++a) {
        double oa = oo[a], da = dd[a];
        if (fabs(da) < 1e-12) { if (oa < mn[a] || oa > mx[a]) return false; continue; }
        double inv = 1.0 / da;
        double s0 = (mn[a] - oa) * inv, s1 = (mx[a] - oa) * inv;
        if (s0 > s1) { double tmp = s0; s0 = s1; s1 = tmp; }
        lo = fmax(lo, s0); hi = fmin(hi, s1);
        if (lo > hi) return false;
    }
    ta = lo; tb = hi; return tb > ta;
}

// ============================ GRIN (gradient-index) marching ==================
// Device twins of scene.h's Medium GRIN helpers + grin::march. A GRIN medium carries an
// `ior` field n(x,y,z); rays bend along the Eikonal ray equation d/ds(n·dr/ds)=∇n. The
// forward megakernel/wavefront call dGrinMarch before each bounce's closestHit, gated by
// sc.hasGrin so ordinary scenes never enter it (bit-identical). Kept byte-for-byte in step
// with the CPU marcher (grin.h) so CPU and GPU bend rays identically.

// Point-in-bound membership (device twin of Medium::insideBound).
__device__ static bool dMedInside(const DMedium& m, const DVec3& p, const DPatEnv& env) {
    if (!m.bounded) return true;
    if (m.boundShape == 1) {   // sphere
        double dx = (double)p.x - m.bcenter.x, dy = (double)p.y - m.bcenter.y,
               dz = (double)p.z - m.bcenter.z;
        return dx * dx + dy * dy + dz * dz <= m.bradius * m.bradius;
    }
    if (m.boundShape == 2 && m.boundField) {   // implicit field
        double f = dFieldEval(m.boundField, m.boundFieldN, p.x, p.y, p.z, m.boundFieldExpr, env);
        return m.boundInsideNeg ? (f < 0.0) : (f > 0.0);
    }
    if (m.boundShape == 3 && m.boundGrid.brickData)   // mesh containment
        return dVdbSample(m.boundGrid, p) >= 0.5;
    return p.x >= m.bmin.x && p.x <= m.bmax.x && p.y >= m.bmin.y &&
           p.y <= m.bmax.y && p.z >= m.bmin.z && p.z <= m.bmax.z;
}

// Local refractive index n at a world point (device twin of Medium::nAt): the shared
// pattern VM with x y z r live, floored at 1e-3. 1.0 when this medium is not GRIN.
__device__ static double dMedNAt(const DMedium& m, const DVec3& p, const DPatEnv& env) {
    if (m.iorN <= 0 || !m.ior) return 1.0;
    double r = sqrt((double)p.x * p.x + (double)p.y * p.y + (double)p.z * p.z);
    double n = dPatternEval(m.ior, m.iorN, p.x, p.y, p.z, 0.0,
                            0.0, 0.0, 0.0, r, 0.0, 0.0, 0.0, 0.0, 0.0, env);
    return n > 1e-3 ? n : 1e-3;
}

// ∇n at a world point via central differences with step h (device twin of Medium::gradNAt).
__device__ static DVec3 dMedGradN(const DMedium& m, const DVec3& p, double h,
                                  const DPatEnv& env) {
    double inv = 0.5 / h;
    DVec3 xp = p, xm = p; xp.x = (Real)(p.x + h); xm.x = (Real)(p.x - h);
    DVec3 yp = p, ym = p; yp.y = (Real)(p.y + h); ym.y = (Real)(p.y - h);
    DVec3 zp = p, zm = p; zp.z = (Real)(p.z + h); zm.z = (Real)(p.z - h);
    double gx = dMedNAt(m, xp, env) - dMedNAt(m, xm, env);
    double gy = dMedNAt(m, yp, env) - dMedNAt(m, ym, env);
    double gz = dMedNAt(m, zp, env) - dMedNAt(m, zm, env);
    return DVec3{ (Real)(gx * inv), (Real)(gy * inv), (Real)(gz * inv) };
}

// (dGrinMarch — the Eikonal ray marcher — needs DHit + closestHit, so it is defined just
// after the closestHit definition below.)

// Sample the next real collision along (o,dir) within [0,dMax]. Device twin of
// Renderer::sampleMediumCollision: exact analytic free-flight (one draw) for a
// homogeneous medium (bit-identical to before), else delta (Woodcock) tracking.
__device__ static bool dMedSampleCollision(const DMedium& m, const DVec3& o, const DVec3& dir,
                                           Real dMax, Real lambda, DRng& rng, Real& tHit,
                                           const DPatEnv& env) {
    double stBase = (double)medSigmaT(m, lambda);
    if (stBase <= 0.0) return false;
    double ta, tb;
    if (!dMedClip(m, o, dir, 0.0, (double)dMax, ta, tb)) return false;
    if (!m.heterogeneous) {
        double t = ta - log(1.0 - (double)rng.uniform()) / stBase;
        if (t < tb) { tHit = (Real)t; return true; }
        return false;
    }
    double sigMax = stBase * m.densityMax;
    if (sigMax <= 0.0) return false;
    double t = ta;
    for (;;) {
        t += -log(1.0 - (double)rng.uniform()) / sigMax;
        if (t >= tb) return false;
        DVec3 pp = o + dir * (Real)t;
        double sigT = stBase * dMedDensityAt(m, pp, env);
        if ((double)rng.uniform() * sigMax < sigT) { tHit = (Real)t; return true; }
    }
}

// Unbiased transmittance along [o, o+dir*dist]. Device twin of
// Renderer::mediumTransmittance: exact exp for a homogeneous medium (no RNG draw), else
// ratio tracking. Homogeneous scenes therefore keep the exact analytic transmittance.
__device__ static Real dMedTransmittance(const DMedium& m, const DVec3& o, const DVec3& dir,
                                         Real dist, Real lambda, DRng& rng,
                                         const DPatEnv& env) {
    double stBase = (double)medSigmaT(m, lambda);
    if (stBase <= 0.0) return (Real)1;
    double ta, tb;
    if (!dMedClip(m, o, dir, 0.0, (double)dist, ta, tb)) return (Real)1;
    if (!m.heterogeneous) return (Real)exp(-stBase * (tb - ta));
    double sigMax = stBase * m.densityMax;
    if (sigMax <= 0.0) return (Real)1;
    double Tr = 1.0, t = ta;
    for (;;) {
        t += -log(1.0 - (double)rng.uniform()) / sigMax;
        if (t >= tb) break;
        DVec3 pp = o + dir * (Real)t;
        double sigT = stBase * dMedDensityAt(m, pp, env);
        Tr *= 1.0 - sigT / sigMax;
    }
    return (Real)Tr;
}

// --- Multi-medium (superposition) device twins of Renderer::sampleMediaCollision /
// mediaTransmittance. The scene may hold several independent, possibly overlapping
// media (sc.media[0..mediaN)). Extinction adds, so total transmittance = product of
// per-medium transmittances, and the first collision across all media is the EARLIEST
// of their independent free-flight samples (Poisson superposition). With one medium
// these reduce to the exact single-medium paths above.
// Take the whole DScene (not just media[0..n)) because a density program may sample the
// scene's grid:/scatter: tables, which live beside the media — same reasoning as the
// host's Renderer::sampleMediaCollision, and it means no call site can forget them.
__device__ static bool dMediaSampleCollision(const DScene& sc, const DVec3& o,
                                             const DVec3& dir, Real dMax, Real lambda,
                                             DRng& rng, Real& tHit, int& whichMed) {
    const DMedium* media = sc.media; const int n = sc.mediaN;
    const DPatEnv env = dPatEnvOf(sc);
    Real best = dMax; int which = -1;
    for (int i = 0; i < n; ++i) {
        Real t;
        if (dMedSampleCollision(media[i], o, dir, dMax, lambda, rng, t, env) && t < best) {
            best = t; which = i;
        }
    }
    if (which < 0) return false;
    tHit = best; whichMed = which; return true;
}

__device__ static Real dMediaTransmittance(const DScene& sc, const DVec3& o,
                                           const DVec3& dir, Real dist, Real lambda, DRng& rng) {
    const DPatEnv env = dPatEnvOf(sc);       // see dMediaSampleCollision
    Real Tr = (Real)1;
    for (int i = 0; i < sc.mediaN; ++i) {
        Tr *= dMedTransmittance(sc.media[i], o, dir, dist, lambda, rng, env);
        if (Tr <= (Real)0) break;
    }
    return Tr;
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
    DVec3 tangent; Real bitangentSign;  // C6 surface tangent frame for normal mapping
    Real curv;   // O3 mean curvature, signed toward the shaded side (see Hit::curv)
    // O3 stage 2 cavity, filled LAZILY by dCavityOf() and cached, exactly like the
    // host's Hit::cavity: it needs whole-scene traversal so no intersector fills it,
    // and one shading point asks for it several times (mix weight, roughness, reflect).
    // These two carry DEFAULT INITIALISERS, unlike every other field, because they are
    // the only ones no intersector writes: a `DHit h;` that some path leaves partly
    // unwritten (dVertHit, the any-hit scratch hits in occluded()) would otherwise
    // start with cavityDone = garbage and could return an uninitialised cavity.
    mutable Real cavity = (Real)0;
    mutable bool cavityDone = false;
    // O8 stage 2 shading footprint (device twin of Hit::fw): the world-space diameter of
    // the surface patch this shading sample stands for. Like cavity it is NOT written by
    // any intersector — it is a property of the ARRIVING RAY, not of the geometry — so it
    // is stamped by the kernel at the camera segment and left 0 (= unknown = unfiltered)
    // everywhere else. Default-initialised for the same reason cavity is.
    Real fw = (Real)0;
    // Hair fiber cross-section radius at the hit (device twin of Hit::fiberRadius).
    // Written ONLY by the curve-segment intersector; every other primitive leaves the
    // default 0. Consumed by the hair BCSDF exit-offset (dHairExitOffset) so far-side
    // (TT) exit rays clear the fiber body. Default-initialised for the same reason
    // cavity/fw are: most intersectors never touch it.
    Real fiberRadius = (Real)0;
};

// ---- implicit field evaluation (device twin of implicit.h) ----------------
// smin/smax (Inigo Quilez quadratic blend) — filleted CSG / metaball merge.
__device__ static inline double dSmin(double a, double b, double k) {
    if (k <= 0.0) return a < b ? a : b;
    double h = fmax(k - fabs(a - b), 0.0) / k;
    return (a < b ? a : b) - h * h * k * 0.25;
}
__device__ static inline double dSmax(double a, double b, double k) { return -dSmin(-a, -b, k); }
__device__ static inline float dSminF(float a, float b, float k) {
    if (k <= 0.0f) return a < b ? a : b;
    float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
    return (a < b ? a : b) - h * h * k * 0.25f;
}
__device__ static inline float dSmaxF(float a, float b, float k) { return -dSminF(-a, -b, k); }

// Leaf SDF at the leaf-LOCAL query point (px,py,pz). Mirrors fieldLeafSDF exactly.
// Forward decl: DF_EXPR leaves evaluate their formula with the FP32 pattern VM, defined
// further down. (The double VM, dPatternEval, is already visible: it is the template from
// pattern_device.cuh.)
__device__ static float dPatternEvalF(const PatNodeF* nodes, int n,
                                      float x, float y, float z, float f,
                                      float nx, float ny, float nz, float r,
                                      float u, float v, const DPatEnv& env);

// `env` publishes the scene's texture/grid/scatter tables so a DF_EXPR leaf can BE a
// sampled volume or height field (`function { expr "grid:terrain(x, z) - y" }`), exactly
// as the host's fieldLeafSDF takes a PatTables. Pass dPatEnvOf(sc), never dPatEnvNone(),
// at any site reachable from a real scene: a `grid:` node evaluated without its table
// makes patternEval bail to 0, i.e. an empty field.
__device__ static double dFieldLeafSDF(const DFieldNode& nd, double px, double py, double pz,
                                       const PatNode* exprPool, const DPatEnv& env) {
    switch (nd.op) {
        case DF_SPHERE:
            return sqrt(px*px + py*py + pz*pz) - nd.p[0];
        case DF_EXPR: {   // arbitrary formula f(x,y,z); r=|p|, other vars (f/normals) are 0
            if (!exprPool) return BIG;
            double r = sqrt(px*px + py*py + pz*pz);
            return dPatternEval(exprPool + nd.exprOff, nd.exprN, px, py, pz, 0.0,
                                0.0, 0.0, 0.0, r, 0.0, 0.0, 0.0, 0.0, 0.0, env);
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
                                    const PatNode* exprPool, const DPatEnv& env) {
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
                st[sp++] = dFieldLeafSDF(nd, plx, ply, plz, exprPool, env) * nd.scale;
            }
        }
    }
    return sp > 0 ? st[0] : BIG;
}
// ---- FP32 twins of the field VM (see DFieldNodeF): used ONLY by the sphere-trace
// march/refine in intersectImplicit, where the result is stored in float anyway.
__device__ static float dFieldLeafSDFF(const DFieldNodeF& nd, float px, float py, float pz,
                                       const PatNodeF* exprPool, const DPatEnv& env) {
    switch (nd.op) {
        case DF_SPHERE:
            return sqrtf(px*px + py*py + pz*pz) - nd.p[0];
        case DF_EXPR: {   // arbitrary formula f(x,y,z); r=|p|, other vars (f/normals) are 0
            if (!exprPool) return (float)BIG;
            float r = sqrtf(px*px + py*py + pz*pz);
            return dPatternEvalF(exprPool + nd.exprOff, nd.exprN, px, py, pz, 0.0f,
                                 0.0f, 0.0f, 0.0f, r, 0.0f, 0.0f, env);
        }
        case DF_BOX: {
            float r = nd.p[3];
            float qx = fabsf(px) - nd.p[0] + r, qy = fabsf(py) - nd.p[1] + r, qz = fabsf(pz) - nd.p[2] + r;
            float ox = fmaxf(qx, 0.0f), oy = fmaxf(qy, 0.0f), oz = fmaxf(qz, 0.0f);
            float outside = sqrtf(ox*ox + oy*oy + oz*oz);
            float inside  = fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f);
            return outside + inside - r;
        }
        case DF_TORUS: {
            float qx = sqrtf(px*px + pz*pz) - nd.p[0];
            return sqrtf(qx*qx + py*py) - nd.p[1];
        }
        case DF_PLANE:
            return px*nd.p[0] + py*nd.p[1] + pz*nd.p[2] + nd.p[3];
        case DF_CYLINDER: {
            float dxz = sqrtf(px*px + pz*pz) - nd.p[0];
            float dy  = fabsf(py) - nd.p[1];
            float a   = fminf(fmaxf(dxz, dy), 0.0f);
            float bx  = fmaxf(dxz, 0.0f), by = fmaxf(dy, 0.0f);
            return a + sqrtf(bx*bx + by*by);
        }
        case DF_CONE: {
            float rb = nd.p[0], rt = nd.p[1], h = nd.p[2];
            float qx = sqrtf(px*px + pz*pz), qy = py;
            float k1x = rt, k1y = h, k2x = rt - rb, k2y = 2.0f*h;
            float cax = qx - fminf(qx, (qy < 0.0f) ? rb : rt);
            float cay = fabsf(qy) - h;
            float k2dot = k2x*k2x + k2y*k2y;
            float tt = (k2dot > 0.0f) ? ((k1x - qx)*k2x + (k1y - qy)*k2y) / k2dot : 0.0f;
            tt = tt < 0.0f ? 0.0f : (tt > 1.0f ? 1.0f : tt);
            float cbx = qx - k1x + k2x*tt, cby = qy - k1y + k2y*tt;
            float s = (cbx < 0.0f && cay < 0.0f) ? -1.0f : 1.0f;
            float da = cax*cax + cay*cay, db = cbx*cbx + cby*cby;
            return s * sqrtf(fminf(da, db));
        }
        default: return (float)BIG;
    }
}
__device__ static float dFieldEvalF(const DFieldNodeF* nodes, int n,
                                    float pwx, float pwy, float pwz,
                                    const PatNodeF* exprPool, const DPatEnv& env) {
    float st[64]; int sp = 0;
    for (int i = 0; i < n; ++i) {
        const DFieldNodeF& nd = nodes[i];
        switch (nd.op) {
            case DF_UNION:            { float b = st[--sp], a = st[--sp]; st[sp++] = a < b ? a : b; break; }
            case DF_INTERSECT:        { float b = st[--sp], a = st[--sp]; st[sp++] = a > b ? a : b; break; }
            case DF_DIFFERENCE:       { float b = st[--sp], a = st[--sp]; st[sp++] = a > -b ? a : -b; break; }
            case DF_SMOOTH_UNION:     { float b = st[--sp], a = st[--sp]; st[sp++] = dSminF(a,  b, nd.p[0]); break; }
            case DF_SMOOTH_INTERSECT: { float b = st[--sp], a = st[--sp]; st[sp++] = dSmaxF(a,  b, nd.p[0]); break; }
            case DF_SMOOTH_DIFFERENCE:{ float b = st[--sp], a = st[--sp]; st[sp++] = dSmaxF(a, -b, nd.p[0]); break; }
            default: {   // leaf: world -> local via inv, then local SDF * scale
                float plx = nd.inv[0]*pwx + nd.inv[1]*pwy + nd.inv[2]*pwz + nd.tx;
                float ply = nd.inv[3]*pwx + nd.inv[4]*pwy + nd.inv[5]*pwz + nd.ty;
                float plz = nd.inv[6]*pwx + nd.inv[7]*pwy + nd.inv[8]*pwz + nd.tz;
                st[sp++] = dFieldLeafSDFF(nd, plx, ply, plz, exprPool, env) * nd.scale;
            }
        }
    }
    return sp > 0 ? st[0] : (float)BIG;
}
// Field gradient (tetrahedron central differences) -> unit normal. Mirrors fieldGradient.
__device__ static void dFieldGradient(const DFieldNode* nodes, int n,
                                      double px, double py, double pz, double eps,
                                      double& gx, double& gy, double& gz,
                                      const PatNode* exprPool, const DPatEnv& env) {
    // stencil offsets k1(1,-1,-1) k2(-1,-1,1) k3(-1,1,-1) k4(1,1,1)
    double f1 = dFieldEval(nodes, n, px + eps, py - eps, pz - eps, exprPool, env);
    double f2 = dFieldEval(nodes, n, px - eps, py - eps, pz + eps, exprPool, env);
    double f3 = dFieldEval(nodes, n, px - eps, py + eps, pz - eps, exprPool, env);
    double f4 = dFieldEval(nodes, n, px + eps, py + eps, pz + eps, exprPool, env);
    gx =  f1 - f2 - f3 + f4;
    gy = -f1 - f2 + f3 + f4;
    gz = -f1 + f2 - f3 + f4;
    double len = sqrt(gx*gx + gy*gy + gz*gz);
    if (len > 0.0) { gx /= len; gy /= len; gz /= len; }
    else           { gx = 0.0; gy = 0.0; gz = 1.0; }
}
// Device twin of geometry.h projectUV: wrap a world point to (u,v) over box lo..hi.
// proj: 1 planar, 2 spherical, 3 cylindrical. axis 0/1/2 = up/projection axis.
__device__ static inline double dNorm01(double val, double lo, double hi) {
    double d = hi - lo;
    return d > 1e-12 ? (val - lo) / d : 0.5;
}
__device__ static void dProjectUV(double px, double py, double pz,
                                  const double* lo, const double* hi,
                                  int proj, int axis, double& outU, double& outV) {
    double p[3] = {px, py, pz};
    int a0 = (axis + 1) % 3, a1 = (axis + 2) % 3;
    if (proj == 1) {   // planar
        outU = dNorm01(p[a0], lo[a0], hi[a0]);
        outV = dNorm01(p[a1], lo[a1], hi[a1]);
        return;
    }
    double ctr[3] = {0.5*(lo[0]+hi[0]), 0.5*(lo[1]+hi[1]), 0.5*(lo[2]+hi[2])};
    double dvec[3] = {p[0]-ctr[0], p[1]-ctr[1], p[2]-ctr[2]};
    double dz = dvec[axis], dx = dvec[a0], dy = dvec[a1];
    double azim = 0.5 + atan2(dy, dx) / (2.0 * DPI);
    if (proj == 3) {   // cylindrical
        outU = azim; outV = dNorm01(p[axis], lo[axis], hi[axis]); return;
    }
    double r = sqrt(dx*dx + dy*dy + dz*dz);   // spherical
    outU = azim;
    outV = (r > 1e-12) ? acos(fmax(-1.0, fmin(1.0, dz / r))) / DPI : 0.5;
}

// Sphere-trace one implicit; writes into `hit` (respecting hit.t). Mirrors intersectImplicit.
// `anyHit=true` (occlusion): boolean-only — skip root refine + gradient + writeHit when the
// answer is already determined (bit-identical return value; see the host twin in implicit.h).
__device__ static bool intersectImplicit(const DScene& sc, const DImplicit& im,
                                          const DVec3& roR, const DVec3& rdR, Real tmin, DHit& hit,
                                          bool anyHit = false) {
    double ox = roR.x, oy = roR.y, oz = roR.z, dx = rdR.x, dy = rdR.y, dz = rdR.z;

    const DFieldNode* nd = sc.fieldNodes + im.nodeOff;   // double pool: gradient/normal only
    const PatNode* exprPool = sc.fieldExprNodes;
    // FP32 mirror pools: the march + root refine run entirely in float (the committed
    // hit is float anyway; FP64 VM ops serialize on the 1/64-rate FP64 pipe).
    const DFieldNodeF* ndF = sc.fieldNodesF + im.nodeOff;
    const PatNodeF* exprPoolF = sc.fieldExprNodesF;
    const int N = im.nodeN;
    // The scene's texture/grid/scatter tables, so a `function` leaf that samples a
    // measured volume marches the real field (dPatEnvNone() would read 0 everywhere).
    const DPatEnv env = dPatEnvOf(sc);

    // ---- Container clip: entry/exit params [tEnter, tExit] and the container's OUTWARD
    // normals at those crossings (needed to shade caps). Box (lo/hi) or world sphere.
    double tEnter, tExit;
    double neX = 0, neY = 0, neZ = 0, nxX = 0, nxY = 0, nxZ = 0;
    if (im.container == 1) {
        double ocx = ox - im.sphereCenter[0], ocy = oy - im.sphereCenter[1], ocz = oz - im.sphereCenter[2];
        double A = dx*dx + dy*dy + dz*dz;
        double B = ocx*dx + ocy*dy + ocz*dz;
        double C = ocx*ocx + ocy*ocy + ocz*ocz - im.sphereRadius*im.sphereRadius;
        double disc = B*B - A*C;
        if (disc < 0.0) return false;
        double sq = sqrt(disc);
        tEnter = (-B - sq) / A;
        tExit  = (-B + sq) / A;
        if (!anyHit) {   // cap normals are only needed when a hit gets written
            double pex = ox + dx*tEnter, pey = oy + dy*tEnter, pez = oz + dz*tEnter;
            double pxx = ox + dx*tExit,  pxy = oy + dy*tExit,  pxz = oz + dz*tExit;
            double gex = pex - im.sphereCenter[0], gey = pey - im.sphereCenter[1], gez = pez - im.sphereCenter[2];
            double gxx = pxx - im.sphereCenter[0], gxy = pxy - im.sphereCenter[1], gxz = pxz - im.sphereCenter[2];
            double le = sqrt(gex*gex + gey*gey + gez*gez), lx = sqrt(gxx*gxx + gxy*gxy + gxz*gxz);
            if (le > 0.0) { neX = gex/le; neY = gey/le; neZ = gez/le; } else neZ = 1.0;
            if (lx > 0.0) { nxX = gxx/lx; nxY = gxy/lx; nxZ = gxz/lx; } else nxZ = 1.0;
        }
    } else if (im.boxOriented != 0) {
        // Slab test in the container's own frame (host twin: the boxOriented branch of
        // intersectImplicit). An affine map preserves the ray parameter, so these t
        // values stay comparable with tmin/hit.t; only the face normals must be pushed
        // back to world, by boxInv's linear part TRANSPOSED (the local->world normal
        // map). Split from the AABB test below for the same reason as on the host: the
        // unrotated path then runs exactly the code it always did, with no ray transform.
        const double* M = im.boxInvM; const double* T = im.boxInvT;
        double bx = M[0]*ox + M[1]*oy + M[2]*oz + T[0];
        double by = M[3]*ox + M[4]*oy + M[5]*oz + T[1];
        double bz = M[6]*ox + M[7]*oy + M[8]*oz + T[2];
        double bdx = M[0]*dx + M[1]*dy + M[2]*dz;
        double bdy = M[3]*dx + M[4]*dy + M[5]*dz;
        double bdz = M[6]*dx + M[7]*dy + M[8]*dz;
        tEnter = -1e300; tExit = 1e300;
        int eAx = 0; double eSgn = -1.0;
        int xAx = 0; double xSgn = 1.0;
        double o3[3] = {bx, by, bz}, id3[3] = {1.0/bdx, 1.0/bdy, 1.0/bdz};
        for (int a = 0; a < 3; ++a) {
            double tLo = (im.boxLo[a] - o3[a]) * id3[a];
            double tHi = (im.boxHi[a] - o3[a]) * id3[a];
            double tnear, tfar, nearSgn, farSgn;
            if (id3[a] >= 0.0) { tnear = tLo; tfar = tHi; nearSgn = -1.0; farSgn = +1.0; }
            else               { tnear = tHi; tfar = tLo; nearSgn = +1.0; farSgn = -1.0; }
            if (tnear > tEnter) { tEnter = tnear; eAx = a; eSgn = nearSgn; }
            if (tfar  < tExit)  { tExit  = tfar;  xAx = a; xSgn = farSgn; }
            if (tExit < tEnter) return false;
        }
        if (!anyHit) {
            double ne[3] = {0, 0, 0}, nx[3] = {0, 0, 0};
            ne[eAx] = eSgn; nx[xAx] = xSgn;
            double ex = M[0]*ne[0] + M[3]*ne[1] + M[6]*ne[2];
            double ey = M[1]*ne[0] + M[4]*ne[1] + M[7]*ne[2];
            double ez = M[2]*ne[0] + M[5]*ne[1] + M[8]*ne[2];
            double xx = M[0]*nx[0] + M[3]*nx[1] + M[6]*nx[2];
            double xy = M[1]*nx[0] + M[4]*nx[1] + M[7]*nx[2];
            double xz = M[2]*nx[0] + M[5]*nx[1] + M[8]*nx[2];
            double le = sqrt(ex*ex + ey*ey + ez*ez), lx = sqrt(xx*xx + xy*xy + xz*xz);
            if (le > 0.0) { neX = ex/le; neY = ey/le; neZ = ez/le; }
            if (lx > 0.0) { nxX = xx/lx; nxY = xy/lx; nxZ = xz/lx; }
        }
    } else {
        double idx = 1.0/dx, idy = 1.0/dy, idz = 1.0/dz;
        tEnter = -1e300; tExit = 1e300;
        int eAx = 0; double eSgn = -1.0;
        int xAx = 0; double xSgn = 1.0;
        double o3[3] = {ox, oy, oz}, id3[3] = {idx, idy, idz};
        double lo3[3] = {im.lo[0], im.lo[1], im.lo[2]}, hi3[3] = {im.hi[0], im.hi[1], im.hi[2]};
        for (int a = 0; a < 3; ++a) {
            double tLo = (lo3[a] - o3[a]) * id3[a];
            double tHi = (hi3[a] - o3[a]) * id3[a];
            double tnear, tfar, nearSgn, farSgn;
            if (id3[a] >= 0.0) { tnear = tLo; tfar = tHi; nearSgn = -1.0; farSgn = +1.0; }
            else               { tnear = tHi; tfar = tLo; nearSgn = +1.0; farSgn = -1.0; }
            if (tnear > tEnter) { tEnter = tnear; eAx = a; eSgn = nearSgn; }
            if (tfar  < tExit)  { tExit  = tfar;  xAx = a; xSgn = farSgn; }
            if (tExit < tEnter) return false;
        }
        if (eAx == 0) neX = eSgn; else if (eAx == 1) neY = eSgn; else neZ = eSgn;
        if (xAx == 0) nxX = xSgn; else if (xAx == 1) nxY = xSgn; else nxZ = xSgn;
    }
    double t0 = tmin, t1 = hit.t;
    if (tEnter > t0) t0 = tEnter;
    if (tExit  < t1) t1 = tExit;
    if (t1 < t0) return false;
    const bool capped          = (im.capped != 0);
    const bool exitIsContainer = (tExit <= (double)hit.t);

    const double dlen = sqrt(dx*dx + dy*dy + dz*dz);
    const int MAX_STEP = 2048;
    const double invLip = 1.0 / (im.lipschitz > 0.0 ? im.lipschitz : 1.0);
    const double minStep = im.minStep > 0.0 ? im.minStep : 1e-4;

    const bool   sampleMode  = (im.method == 1);
    const double fixedStep   = (im.sampleStep > 0.0 ? im.sampleStep : minStep) / dlen;
    const bool   regulaFalsi = (im.refine == 1);

    // FP32 march state (double ray kept for the hit commit + gradient).
    const float oxF = (float)ox, oyF = (float)oy, ozF = (float)oz;
    const float dxF = (float)dx, dyF = (float)dy, dzF = (float)dz;
    const float dlenF    = (float)dlen;
    const float invLipF  = (float)invLip, minStepF = (float)minStep;
    const float fixedStepF = (float)fixedStep;
    const float t1F = (float)t1;

    // Commit a hit at parametric `th`, world point (px,py,pz), geometric normal (gx,gy,gz).
    auto writeHit = [&](double th, double px, double py, double pz,
                        double gx, double gy, double gz) -> bool {
        hit.t = (Real)th; hit.p = DVec3(px, py, pz); hit.valid = true;
        hit.ng = DVec3(gx, gy, gz);
        double side = dx*gx + dy*gy + dz*gz;
        hit.n = (side < 0.0) ? DVec3(gx, gy, gz) : DVec3(-gx, -gy, -gz);
        hit.matId = im.matId; hit.sensorId = -1;
        if (im.uvProj != 0) {
            double uu, vv;
            dProjectUV(px, py, pz, im.uvLo, im.uvHi, im.uvProj, im.uvAxis, uu, vv);
            hit.u = (Real)uu; hit.v = (Real)vv;
        } else { hit.u = 0; hit.v = 0; }
        hit.curv = (Real)0;   // O3: isosurface curvature needs the Hessian — see implicit.h
        return true;
    };

    float t = (float)t0;
    float f = dFieldEvalF(ndF, N, oxF + dxF*t, oyF + dyF*t, ozF + dzF*t, exprPoolF, env);
    // NEAR CAP: ray enters the container already inside the solid (f<0); the container
    // face is the nearest surface. `open` skips this to reveal the cut edge.
    if (capped && tEnter >= tmin && tEnter < (double)hit.t && f < 0.0f)
        return anyHit ? true : writeHit(tEnter, ox + dx*tEnter, oy + dy*tEnter, oz + dz*tEnter, neX, neY, neZ);
    for (int i = 0; i < MAX_STEP; ++i) {
        float step = sampleMode ? fixedStepF : fmaxf(fabsf(f) * invLipF, minStepF) / dlenF;
        float tn = t + step;
        bool last = false;
        if (tn >= t1F) { tn = t1F; last = true; }
        float fn = dFieldEvalF(ndF, N, oxF + dxF*tn, oyF + dyF*tn, ozF + dzF*tn, exprPoolF, env);
        bool crossed = (f > 0.0f && fn <= 0.0f) || (f < 0.0f && fn >= 0.0f) || (f == 0.0f && fn != 0.0f);
        if (crossed) {
            // Any-hit fast path: refinement keeps [ta,tb] inside [t,tn] (tm is guarded
            // strictly interior), and th = 0.5*((double)ta+(double)tb) lies in
            // [(double)ta,(double)tb]. So when (double)t >= tmin and (double)tn <
            // (double)hit.t, the acceptance test below is guaranteed to pass — return
            // the boolean without refining. (The guard is conservative: t = (float)t0
            // can round below tmin, and tn == t1F can sit at/above hit.t after the
            // float clamp; those rare corners take the full path and match baseline.)
            if (anyHit && (double)t >= (double)tmin && (double)tn < (double)hit.t) return true;
            float ta = t, tb = tn, fa = f, fb = fn;
            int rfSide = 0;
            for (int b = 0; b < 48; ++b) {
                float tm;
                if (regulaFalsi && (fb - fa) != 0.0f) {
                    tm = (ta * fb - tb * fa) / (fb - fa);
                    if (tm <= ta || tm >= tb) tm = 0.5f*(ta + tb);
                } else {
                    tm = 0.5f*(ta + tb);
                }
                if (tm <= ta || tm >= tb) break;   // float interval exhausted: converged
                float fm = dFieldEvalF(ndF, N, oxF + dxF*tm, oyF + dyF*tm, ozF + dzF*tm, exprPoolF, env);
                if ((fa > 0.0f) == (fm > 0.0f)) {
                    ta = tm; fa = fm;
                    if (regulaFalsi && rfSide == +1) fb *= 0.5f;
                    rfSide = +1;
                } else {
                    tb = tm; fb = fm;
                    if (regulaFalsi && rfSide == -1) fa *= 0.5f;
                    rfSide = -1;
                }
            }
            double th = 0.5*((double)ta + (double)tb);
            if (th < tmin || th >= (double)hit.t) return false;
            if (anyHit) return true;   // boolean only: skip gradient + writeHit
            double px = ox + dx*th, py = oy + dy*th, pz = oz + dz*th;
            double eps = fmax(1e-6, 1e-4*th);
            double gx, gy, gz; dFieldGradient(nd, N, px, py, pz, eps, gx, gy, gz, exprPool, env);
            return writeHit(th, px, py, pz, gx, gy, gz);
        }
        if (last) {
            // FAR CAP: reached the container exit still inside the solid (fn<0), and the
            // far clip is the container itself — seal the sawn-off solid.
            if (capped && exitIsContainer && fn < 0.0f && tExit >= tmin && tExit < (double)hit.t)
                return anyHit ? true : writeHit(tExit, ox + dx*tExit, oy + dy*tExit, oz + dz*tExit, nxX, nxY, nxZ);
            return false;
        }
        t = tn; f = fn;
    }
    return false;
}

// Watertight ray-triangle intersection (Woop et al., JCGT 2013) — device twin of the
// host intersectTri in geometry.h. The consistent per-edge sign test means a ray through
// a shared edge is claimed by exactly one triangle: no cracks (background leaking through
// a closed mesh) and no dropped hits. This matters most HERE, on the float (Real) path,
// where the old Moller-Trumbore's independent per-triangle edge signs cracked at grazing
// angles. The per-ray axis permutation + shear (DTriShear) is hoisted out of the BVH leaf
// loop and reused for every triangle, exactly like the host.
struct DTriShear {
    int  kx, ky, kz;
    Real Sx, Sy, Sz;
};
// `a*b - c*d` with NO FMA contraction, which the watertight guarantee actually depends on.
//
// The guarantee is: the two triangles sharing an edge evaluate that edge from bitwise
// identical operands in opposite order, so their edge functions are exact negatives and a
// ray dead-on the edge is claimed by exactly one of them (both `>= 0` chains accept a zero).
// Negation is exact in IEEE and round-to-nearest is symmetric under it -- but ONLY if both
// products are rounded. nvcc defaults to `-fmad=true` and contracts `a*b - c*d` into
// `fma(a, b, -(c*d))`, which keeps `a*b` exact and rounds only `c*d`. The two sharers then
// compute `exact(pq) - rounded(qp)` and `exact(qp) - rounded(pq)`; on an exact tie the two
// exact products are equal, so BOTH come out as the same small residual with the same sign
// -- and if that sign is the minority one, BOTH triangles reject and the surface cracks.
//
// Measured on `scraps/cor_gi.ftsl` at 240x240 (v0.115.1, before this fix): the cornell box's
// back-wall quad diagonal and its four ceiling/floor-to-side-wall corner seams project onto
// the frame diagonals `x == y` / `x + y == 239`, i.e. dead through hundreds of consecutive
// pixel centres. 134 of those pixels came back pure black on the GPU (escaped ray, no hit)
// against a lit ~(247,234,236) wall on the CPU. That is the exact same failure -- and the
// exact same fix -- as the rasterizer's `edgeRow`/`edgeAt` (known-issues.md, v0.98.2).
__device__ static inline float  dMulRn(float  a, float  b) { return __fmul_rn(a, b); }
__device__ static inline double dMulRn(double a, double b) { return __dmul_rn(a, b); }
__device__ static inline float  dSubRn(float  a, float  b) { return __fsub_rn(a, b); }
__device__ static inline double dSubRn(double a, double b) { return __dsub_rn(a, b); }
__device__ static inline Real dCrossRn(Real a, Real b, Real c, Real d) {
    return dSubRn(dMulRn(a, b), dMulRn(c, d));
}
__device__ static inline DTriShear makeTriShear(const DVec3& d) {
    DTriShear s;
    Real ax = fabs(d.x), ay = fabs(d.y), az = fabs(d.z);
    if (ax >= ay && ax >= az)      s.kz = 0;
    else if (ay >= az)             s.kz = 1;
    else                           s.kz = 2;
    s.kx = s.kz + 1; if (s.kx == 3) s.kx = 0;
    s.ky = s.kx + 1; if (s.ky == 3) s.ky = 0;
    if (d[s.kz] < 0) { int tmp = s.kx; s.kx = s.ky; s.ky = tmp; }
    s.Sx = d[s.kx] / d[s.kz];
    s.Sy = d[s.ky] / d[s.kz];
    s.Sz = (Real)1  / d[s.kz];
    return s;
}
__device__ static bool intersectTri(const DTriShear& sh, const DVec3& ro, const DVec3& rd,
                                     const DTri& tri, Real tmin, DHit& hit) {
    const int kx = sh.kx, ky = sh.ky, kz = sh.kz;
    DVec3 A = tri.v0 - ro, B = tri.v1 - ro, C = tri.v2 - ro;
    Real Ax = A[kx] - sh.Sx * A[kz], Ay = A[ky] - sh.Sy * A[kz];
    Real Bx = B[kx] - sh.Sx * B[kz], By = B[ky] - sh.Sy * B[kz];
    Real Cx = C[kx] - sh.Sx * C[kz], Cy = C[ky] - sh.Sy * C[kz];
    // Non-contracted (see dCrossRn): an FMA here breaks the shared-edge antisymmetry and
    // cracks the surface along any edge that lands on exact pixel centres.
    Real U = dCrossRn(Cx, By, Cy, Bx);
    Real V = dCrossRn(Ax, Cy, Ay, Cx);
    Real W = dCrossRn(Bx, Ay, By, Ax);
    // Exact-zero fallback in double (helps the float path land a grazing edge on one side).
    // Non-contracted for the same reason as above: an exact zero is precisely the tie case,
    // so this is the code that MUST stay antisymmetric across the two sharers of an edge.
    if (U == 0 || V == 0 || W == 0) {
        auto xd = [](Real a, Real b, Real c, Real d) {
            return (Real)dSubRn(dMulRn((double)a, (double)b), dMulRn((double)c, (double)d));
        };
        if (U == 0) U = xd(Cx, By, Cy, Bx);
        if (V == 0) V = xd(Ax, Cy, Ay, Cx);
        if (W == 0) W = xd(Bx, Ay, By, Ax);
    }
    // Two-sided: reject only when the edge signs are mixed (point outside the triangle).
    if ((U < 0 || V < 0 || W < 0) && (U > 0 || V > 0 || W > 0)) return false;
    Real det = U + V + W;
    if (det == 0) return false;
    Real T = U * (sh.Sz * A[kz]) + V * (sh.Sz * B[kz]) + W * (sh.Sz * C[kz]);
    Real invDet = (Real)1 / det;
    Real t = T * invDet;
    if (t < tmin || t >= hit.t) return false;
    Real b0 = U * invDet, b1 = V * invDet, b2 = W * invDet;   // barycentric of v0,v1,v2
    hit.t = t; hit.p = ro + rd * t; hit.valid = true;
    hit.matId = tri.matId; hit.sensorId = tri.sensorId;
    hit.u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
    hit.v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
    DVec3 ns = tri.n0 * b0 + tri.n1 * b1 + tri.n2 * b2;
    Real nl = dot(ns, ns);
    ns = (nl > (Real)1e-18) ? ns * ((Real)1 / sqrt(nl)) : tri.gn;
    // Facing is decided from the GEOMETRIC normal, exactly as the host intersectTri
    // does — see the long comment there for why the shading normal cannot answer it
    // (it grazes through zero at every silhouette while the facet is still
    // front-facing). The faceforward of gn onto ns handles a mesh whose winding
    // disagrees with its authored `vn`; hit.ng stays un-ray-flipped because
    // `entering` / one-sided-emitter tests read the side off it. The two copies MUST
    // agree: a CPU/GPU split here shows up as one device shading silhouettes
    // differently from the other.
    DVec3 gn = (dot(tri.gn, ns) < 0) ? -tri.gn : tri.gn;
    bool flipped = !(dot(rd, gn) < 0);
    hit.ng = gn;
    hit.n = flipped ? -ns : ns;
    hit.tangent = tri.tangent;                 // per-triangle tangent (C6 normal mapping)
    hit.bitangentSign = (Real)tri.bitangentSign;
    // O3: curvature follows the shaded side, exactly as the host intersectTri does.
    hit.curv = (Real)(flipped ? -tri.curvature : tri.curvature);
    return true;
}
// Interface-preserving wrapper (builds the shear inline) for any one-off caller.
__device__ static inline bool intersectTri(const DVec3& ro, const DVec3& rd, const DTri& tri,
                                            Real tmin, DHit& hit) {
    return intersectTri(makeTriShear(rd), ro, rd, tri, tmin, hit);
}
__device__ static bool intersectSphere(const DVec3& ro, const DVec3& rd, const DSphere& s,
                                        Real tmin, DHit& hit) {
    // NUMERICALLY STABLE ray/sphere (Ray Tracing Gems ch. 7). The textbook
    // disc = b^2 - 4ac catastrophically cancels once the sphere is far away relative to its
    // radius: both terms are O(dist^2) and the difference is O(radius^2), so in fp32 the
    // hit distance of a sphere `k` radii away carries ~k^2 ulp of error. A 1.85 m sphere at
    // 400 m (a scene modelling the sun that way) landed +-1 cm off, which is enough for a
    // BDPT/VCM connection ray aimed AT that sphere to re-hit it inside its own endpoint
    // epsilon and be thrown away as occluded — the light then rendered ~2.7x too dim.
    // Forming the discriminant from the ray's PERPENDICULAR offset instead keeps every
    // term O(radius^2), and computing the near root by Vieta (c/q) keeps it stable too.
    DVec3 oc = ro - s.c;
    Real a = dot(rd, rd);
    Real bP = -dot(oc, rd);                       // = -b/2
    DVec3 fd = oc + rd * (bP / a);                // ray's closest approach, relative to c
    Real disc = a * ((Real)(s.r * s.r) - dot(fd, fd));   // == bP*bP - a*c, without the cancellation
    if (disc < 0) return false;
    Real c = dot(oc, oc) - (Real)(s.r * s.r);
    Real sq = sqrt(disc);
    Real q = bP + (bP >= (Real)0 ? sq : -sq);
    if (q == (Real)0) return false;               // exactly tangential; no usable root
    Real t = c / q, tFar = q / a;                 // near, far (Vieta) — order by value below
    if (t > tFar) { Real tmp = t; t = tFar; tFar = tmp; }
    if (t < tmin) t = tFar;
    if (t < tmin || t >= hit.t) return false;
    hit.t = t; hit.p = ro + rd * t; hit.valid = true;
    DVec3 ng = normalize(hit.p - s.c);
    hit.ng = ng;
    bool sFlipped = !(dot(rd, ng) < 0);
    hit.n = sFlipped ? -ng : ng;
    // O3: analytic 1/R, negated from inside — mirrors the host intersectSphere.
    hit.curv = (s.r > 1e-12) ? (Real)((sFlipped ? -1.0 : 1.0) / s.r) : (Real)0;
    hit.matId = s.matId; hit.sensorId = -1;
    // Equirectangular (lat/long) UV so spheres can be textured (mirrors host).
    Real ny = ng.y < (Real)-1 ? (Real)-1 : (ng.y > (Real)1 ? (Real)1 : ng.y);
    hit.u = (Real)0.5 + atan2(ng.z, ng.x) / (Real)(2.0 * DPI);
    hit.v = (Real)0.5 - asin(ny) / (Real)DPI;
    // Longitude (east) tangent d/du, mirroring the host sphere path (C6).
    DVec3 tg{-ng.z, (Real)0, ng.x};
    Real tgl = sqrt(dot(tg, tg));
    hit.tangent = (tgl > (Real)1e-9) ? tg * ((Real)1 / tgl) : DVec3{(Real)1, (Real)0, (Real)0};
    hit.bitangentSign = (Real)1;
    return true;
}
// ---- curve / fiber segments (device twin of curve.h) ----------------------
// A round cone: the convex hull of sphere(p0,r0) and sphere(p1,r1). See curve.h for the
// three-piece decomposition (lateral tangent band + a spherical cap at each end) and the
// axial band test that selects between them; this is a transliteration of
// `curveSegCrossings` + the hit fill of `intersectCurveSeg`.
//
// THE ONE THING THAT IS NOT OPTIONAL HERE is the origin recentering. `Real` is float by
// default, and the naive quadric cancels over six decades at fiber scale — measured, it
// loses 12-36% of hits and misplaces the rest by 11-42 fiber radii, i.e. the GPU would
// draw a strand as speckled holes while the CPU drew it perfectly. `ftrace -checkcurve`
// §6 exists specifically to guard this invariant and instantiates the host's
// `curveSegCrossings` at `float` to do it; if that section fails, THIS function is wrong
// too. Do not "simplify" the shift away.
struct DCurveRay { DVec3 dn; Real invLen; };   // unit direction + 1/|d|, hoisted per ray

__device__ static inline DCurveRay makeCurveRay(const DVec3& d) {
    DCurveRay c; c.dn = d; c.invLen = (Real)1;
    Real l2 = dot(d, d);
    if (l2 > (Real)0) { Real l = sqrt(l2); c.invLen = (Real)1 / l; c.dn = d * c.invLen; }
    return c;
}

__device__ static bool intersectCurveSeg(const DCurveRay& cr, const DVec3& ro, const DVec3& rd,
                                          const DCurveSeg& s, Real tmin, DHit& hit,
                                          bool anyHit = false) {
    const DVec3 dn = cr.dn;
    const DVec3 ba = s.p1 - s.p0;
    const Real  m0 = dot(ba, ba);
    if (!(m0 > (Real)0)) return false;           // zero-length segment: nothing to sweep

    // Slide the origin to its closest approach to p0 (dn is unit) — the conditioning fix.
    const Real  shift = dot(s.p0 - ro, dn);
    const DVec3 o  = ro + dn * shift;
    const DVec3 oa = o - s.p0;
    const DVec3 ob = o - s.p1;

    const Real rr = s.r0 - s.r1;
    const Real m1 = dot(ba, oa);
    const Real m2 = dot(ba, dn);
    const Real m3 = dot(dn, oa);
    const Real m5 = dot(oa, oa);
    const Real m6 = dot(ob, dn);
    const Real m7 = dot(ob, ob);
    const Real d2 = m0 - rr * rr;

    Real bestTn = BIG;      // best root, in the RECENTERED unit-direction parameter
    int  piece   = -1;      // 0 = lateral, 1 = cap at p0, 2 = cap at p1
    Real bestY   = (Real)0;

    // Roots are in the shifted parameterisation and the near one is typically NEGATIVE,
    // so undo the shift before the tmin / running-closest tests (host comment applies).
    #define CURVE_OFFER(TN, W, Y) do {                                        \
        const Real _tn = (TN);                                                \
        if (_tn < bestTn) {                                                   \
            const Real _t = (_tn + shift) * cr.invLen;                        \
            if (_t >= tmin && _t < hit.t) { bestTn = _tn; piece = (W); bestY = (Y); } \
        }                                                                     \
    } while (0)

    if (d2 > (Real)0) {
        const Real k2 = d2 - m2 * m2;
        const Real k1 = d2 * m3 - m1 * m2 + m2 * rr * s.r0;
        const Real k0 = d2 * m5 - m1 * m1 + m1 * rr * s.r0 * (Real)2 - m0 * s.r0 * s.r0;
        const Real h  = k1 * k1 - k0 * k2;
        if (h >= (Real)0 && k2 != (Real)0) {
            const Real sq = sqrt(h);
            Real ta = (-sq - k1) / k2, tb = (sq - k1) / k2;
            if (ta > tb) { Real tmp = ta; ta = tb; tb = tmp; }
            const Real base = m1 - s.r0 * rr;
            const Real ya = base + ta * m2, yb = base + tb * m2;
            if (ya > (Real)0 && ya < d2) CURVE_OFFER(ta, 0, ya);
            if (yb > (Real)0 && yb < d2) CURVE_OFFER(tb, 0, yb);
        }
        const Real h1 = m3 * m3 - m5 + s.r0 * s.r0;
        if (h1 > (Real)0) {
            const Real q = sqrt(h1), base = m1 - s.r0 * rr;
            const Real ta = -m3 - q, tb = -m3 + q;
            if (base + ta * m2 <= (Real)0) CURVE_OFFER(ta, 1, (Real)0);
            if (base + tb * m2 <= (Real)0) CURVE_OFFER(tb, 1, (Real)0);
        }
        const Real h2 = m6 * m6 - m7 + s.r1 * s.r1;
        if (h2 > (Real)0) {
            const Real q = sqrt(h2), base = m1 - s.r0 * rr;
            const Real ta = -m6 - q, tb = -m6 + q;
            if (base + ta * m2 >= d2) CURVE_OFFER(ta, 2, d2);
            if (base + tb * m2 >= d2) CURVE_OFFER(tb, 2, d2);
        }
    } else {
        // One ball swallows the other: the hull is just the bigger sphere.
        const bool useP0 = (s.r0 >= s.r1);
        const Real rc    = useP0 ? s.r0 : s.r1;
        const Real mm    = useP0 ? m3 : m6;
        const Real mq    = useP0 ? m5 : m7;
        const Real hh    = mm * mm - mq + rc * rc;
        if (hh > (Real)0) {
            const Real q = sqrt(hh);
            CURVE_OFFER(-mm - q, useP0 ? 1 : 2, useP0 ? (Real)0 : d2);
            CURVE_OFFER(-mm + q, useP0 ? 1 : 2, useP0 ? (Real)0 : d2);
        }
    }
    #undef CURVE_OFFER

    if (piece < 0) return false;

    const Real t = (bestTn + shift) * cr.invLen;
    hit.t = t;
    hit.valid = true;
    if (anyHit) return true;

    const DVec3 p = ro + rd * t;
    hit.p = p;
    hit.matId = s.matId;
    hit.sensorId = -1;

    DVec3 ng;
    if (piece == 0)      ng = (oa + dn * bestTn) * d2 - ba * bestY;   // lateral (iq)
    else if (piece == 1) ng = oa + dn * bestTn;                       // sphere at p0
    else                 ng = ob + dn * bestTn;                       // sphere at p1
    const Real nl = sqrt(dot(ng, ng));
    ng = (nl > (Real)1e-18) ? ng * ((Real)1 / nl) : normalize(ba);
    hit.ng = ng;
    hit.n  = (dot(rd, ng) < (Real)0) ? ng : -ng;

    // u along the strand (root -> tip), v around the circumference.
    Real f = (d2 > (Real)0) ? (bestY / d2) : (Real)0;
    f = f < (Real)0 ? (Real)0 : (f > (Real)1 ? (Real)1 : f);
    hit.u = (Real)s.u0 + ((Real)s.u1 - (Real)s.u0) * f;

    const DVec3 axis = ba * ((Real)1 / sqrt(m0));
    const DVec3 dp   = p - s.p0;
    const DVec3 radial = dp - axis * dot(dp, axis);
    // Same onb() basis as the host, so u/v patterns land identically on both devices.
    const Real sgn = (axis.z >= (Real)0) ? (Real)1 : (Real)-1;
    const Real ia  = (Real)-1 / (sgn + axis.z);
    const Real dd  = axis.x * axis.y * ia;
    const DVec3 T{(Real)1 + sgn * axis.x * axis.x * ia, sgn * dd, -sgn * axis.x};
    const DVec3 B{dd, sgn + axis.y * axis.y * ia, -axis.y};
    hit.v = (Real)0.5 + atan2(dot(radial, B), dot(radial, T)) * (Real)(1.0 / (2.0 * DPI));

    // Surface tangent = the fiber axis, Gram-Schmidt'd against the shading normal.
    DVec3 tg = axis - hit.n * dot(hit.n, axis);
    const Real tl = sqrt(dot(tg, tg));
    hit.tangent = (tl > (Real)1e-12) ? tg * ((Real)1 / tl) : T;
    hit.bitangentSign = (Real)1;
    // O3 mean curvature H = 1/(2R) at the local radius — mirrors host intersectCurveSeg.
    const Real rLoc = s.r0 + (s.r1 - s.r0) * f;
    const Real hSign = (dot(rd, ng) < (Real)0) ? (Real)1 : (Real)-1;
    hit.curv = (rLoc > (Real)1e-12) ? hSign * (Real)0.5 / rLoc : (Real)0;
    hit.fiberRadius = rLoc;   // hair BCSDF exit offset (dHairExitOffset)
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

// Apply an affine's linear part + translation to a point (device twin of Affine::apply).
HD static inline DVec3 affPoint(const double* M, const double* T, const DVec3& p) {
    return DVec3(M[0]*p.x + M[1]*p.y + M[2]*p.z + T[0],
                 M[3]*p.x + M[4]*p.y + M[5]*p.z + T[1],
                 M[6]*p.x + M[7]*p.y + M[8]*p.z + T[2]);
}
// Apply an affine's linear part only, to a direction (device twin of Affine::applyDir).
// NOTE (mirrors host Blas): the direction is NOT renormalized, so the local parametric
// t equals the world t and the shared tMax needs no rescaling.
HD static inline DVec3 affDir(const double* M, const DVec3& v) {
    return DVec3(M[0]*v.x + M[1]*v.y + M[2]*v.z,
                 M[3]*v.x + M[4]*v.y + M[5]*v.z,
                 M[6]*v.x + M[7]*v.y + M[8]*v.z);
}

// Closest hit inside one BLAS, in its LOCAL space. `h.t` carries the running
// world(==local) tMax on entry; a closer local hit updates `h` (local normals/UVs).
// Mirrors Blas::intersectLocal. Returns true if `h` was updated.
__device__ static bool blasClosest(const DScene& sc, const DInstance& inst,
                                    const DVec3& lro, const DVec3& lrd, Real tmin, DHit& h) {
    const DBlas& bl = sc.blas[inst.blasId];
    const DNode* N = sc.blasNodes + bl.nodeOff;
    const int*   P = sc.blasPrim  + bl.primOff;
    const DTri*  T = sc.blasTris   + bl.triOff;
    DVec3 invD{(Real)1 / lrd.x, (Real)1 / lrd.y, (Real)1 / lrd.z};
    const DTriShear sh = makeTriShear(lrd);   // watertight shear: once per ray
    Real tMax = h.t;
    bool found = false;
    // Children are slab-tested once at push time with tEnter recorded; the pop-time
    // 6-plane retest is the exactly-equivalent scalar prune tEnter > tMax (boxHit
    // seeds te with tmin only — see Bvh::traverseClosest for the derivation). A
    // pruned pop also skips loading the node entirely.
    int stack[48]; Real tStack[48]; int sp = 0;
    Real tRoot;
    if (!boxHit(N[0], lro, invD, tmin, tMax, tRoot)) return false;
    stack[0] = 0; tStack[0] = tRoot; sp = 1;
    while (sp) {
        --sp;
        if (tStack[sp] > tMax) continue;
        const DNode& n = N[stack[sp]];
        if (n.count > 0) {
            for (int i = 0; i < n.count; ++i) {
                int prim = P[n.first + i];
                if (intersectTri(sh, lro, lrd, T[prim], tmin, h)) { tMax = h.t; found = true; }
            }
        } else {
            Real tL, tR;
            bool hL = boxHit(N[n.left],  lro, invD, tmin, tMax, tL);
            bool hR = boxHit(N[n.right], lro, invD, tmin, tMax, tR);
            if (hL && hR) {
                if (tL <= tR) { stack[sp] = n.right; tStack[sp] = tR; ++sp;
                                stack[sp] = n.left;  tStack[sp] = tL; ++sp; }
                else          { stack[sp] = n.left;  tStack[sp] = tL; ++sp;
                                stack[sp] = n.right; tStack[sp] = tR; ++sp; }
            } else if (hL) { stack[sp] = n.left;  tStack[sp] = tL; ++sp; }
            else if (hR)   { stack[sp] = n.right; tStack[sp] = tR; ++sp; }
        }
    }
    return found;
}

// Any hit inside one BLAS (local space), before `maxDist`. Mirrors Blas::occludedLocal.
__device__ static bool blasOccluded(const DScene& sc, const DInstance& inst,
                                     const DVec3& lro, const DVec3& lrd, Real tmin, Real maxDist) {
    const DBlas& bl = sc.blas[inst.blasId];
    const DNode* N = sc.blasNodes + bl.nodeOff;
    const int*   P = sc.blasPrim  + bl.primOff;
    const DTri*  T = sc.blasTris   + bl.triOff;
    DVec3 invD{(Real)1 / lrd.x, (Real)1 / lrd.y, (Real)1 / lrd.z};
    const DTriShear sh = makeTriShear(lrd);
    // maxDist never shrinks in any-hit traversal, so a child that passed its
    // push-time slab test cannot fail the identical pop-time retest — test the
    // root once and drop the per-pop retest entirely.
    Real tRoot;
    if (!boxHit(N[0], lro, invD, tmin, maxDist, tRoot)) return false;
    int stack[48]; int sp = 0; stack[sp++] = 0;
    while (sp) {
        const DNode& n = N[stack[--sp]];
        if (n.count > 0) {
            for (int i = 0; i < n.count; ++i) {
                int prim = P[n.first + i];
                DHit h; h.t = maxDist; h.valid = false;
                if (intersectTri(sh, lro, lrd, T[prim], tmin, h)) return true;
            }
        } else {
            Real tc;
            if (boxHit(N[n.left],  lro, invD, tmin, maxDist, tc)) stack[sp++] = n.left;
            if (boxHit(N[n.right], lro, invD, tmin, maxDist, tc)) stack[sp++] = n.right;
        }
    }
    return false;
}

// Transform a BLAS-local hit `lh` back into world space for instance `inst` under the
// world ray (ro,rd). Positions map by the world t (== local t); normals by (toWorld)^-T
// (Nm); the shading normal is re-oriented against the world ray. Mirrors the host
// Scene::instanceHitToWorld exactly.
__device__ static void instanceHitToWorld(const DInstance& inst, const DVec3& ro,
                                          const DVec3& rd, DHit& lh) {
    lh.p = ro + rd * lh.t;
    DVec3 wn  = normalize(affDir(inst.Nm, lh.n));
    DVec3 wng = normalize(affDir(inst.Nm, lh.ng));
    lh.ng = wng;
    lh.n  = (dot(rd, wn) < 0) ? wn : -wn;
    // Map the surface tangent through the instance's toWorld linear part (C6).
    DVec3 wt = affDir(inst.Wm, lh.tangent);
    Real wtl = sqrt(dot(wt, wt));
    if (wtl > (Real)1e-12) lh.tangent = wt * ((Real)1 / wtl);
    // O3: rescale curvature into world units and re-flip with the normal — mirrors host.
    lh.curv *= (Real)inst.curvScale;
    if (dot(rd, wn) >= 0) lh.curv = -lh.curv;
    if (inst.matOverride >= 0) lh.matId = inst.matOverride;
}

// `tCap` bounds the search: only hits with t < tCap are found (h.t and the traversal
// tMax both start there, so every primitive/box test prunes against it — a caller that
// only cares about "anything within d?" skips nearly the whole tree). Default BIG keeps
// every existing call site's behaviour bit-identical. Used by dGrinMarch, whose per-step
// query only consumes hits within one Eikonal step length.
// Forward decl: the tangent-space normal-map perturbation (C6) is defined below with the
// texture samplers (which depend on dWrapIndex), but is called from closestHit's tail.
__device__ static inline void dApplyNormalMap(const DScene& sc, DHit& h);

__device__ static DHit closestHit(const DScene& sc, const DVec3& ro, const DVec3& rd,
                                   Real tmin = RAY_EPS, Real tCap = BIG) {
    DHit h; h.t = tCap; h.valid = false; h.matId = 0; h.sensorId = -1;
    if (sc.nNodes == 0) return h;
    DVec3 invD{(Real)1 / rd.x, (Real)1 / rd.y, (Real)1 / rd.z};
    const DTriShear sh = makeTriShear(rd);
    // Hoisted per ray, not per segment (a fur render tests thousands of segments per ray).
    // Guarded so a curve-free scene never pays the normalising sqrt — the host does the
    // same in Scene::closestHit.
    const DCurveRay cray = sc.nCurveSegs ? makeCurveRay(rd) : DCurveRay{DVec3{(Real)0,(Real)0,(Real)1}, (Real)1};
    Real tMax = tCap;
    // Push-time slab tests + pop-time scalar prune (see blasClosest).
    int stack[64]; Real tStack[64]; int sp = 0;
    Real tRoot;
    if (!boxHit(sc.nodes[0], ro, invD, tmin, tMax, tRoot)) return h;
    stack[0] = 0; tStack[0] = tRoot; sp = 1;
    while (sp) {
        --sp;
        if (tStack[sp] > tMax) continue;
        const DNode& n = sc.nodes[stack[sp]];
        if (n.count > 0) {
            for (int i = 0; i < n.count; ++i) {
                int prim = sc.primIdx[n.first + i];
                if (prim < sc.nTris)              { if (intersectTri(sh, ro, rd, sc.tris[prim], tmin, h)) tMax = h.t; }
                else if (prim < sc.nTris + sc.nSph){ if (intersectSphere(ro, rd, sc.sph[prim - sc.nTris], tmin, h)) tMax = h.t; }
                else if (prim < sc.nTris + sc.nSph + sc.nImplicits) { if (intersectImplicit(sc, sc.implicits[prim - sc.nTris - sc.nSph], ro, rd, tmin, h)) tMax = h.t; }
                else if (prim < sc.nTris + sc.nSph + sc.nImplicits + sc.nCurveSegs) { if (intersectCurveSeg(cray, ro, rd, sc.curveSegs[prim - sc.nTris - sc.nSph - sc.nImplicits], tmin, h)) tMax = h.t; }
                else {
                    // Instance leaf: transform the ray into BLAS-local space, walk the
                    // shared sub-BVH, and map any closer hit back to world space.
                    const DInstance& inst = sc.instances[prim - sc.nTris - sc.nSph - sc.nImplicits - sc.nCurveSegs];
                    DVec3 lro = affPoint(inst.Lm, inst.Lt, ro);
                    DVec3 lrd = affDir(inst.Lm, rd);
                    DHit lh; lh.t = h.t; lh.valid = false;
                    if (blasClosest(sc, inst, lro, lrd, tmin, lh)) {
                        instanceHitToWorld(inst, ro, rd, lh);
                        h = lh; tMax = h.t;
                    }
                }
            }
        } else {
            Real tL, tR;
            bool hL = boxHit(sc.nodes[n.left], ro, invD, tmin, tMax, tL);
            bool hR = boxHit(sc.nodes[n.right], ro, invD, tmin, tMax, tR);
            if (hL && hR) {
                if (tL <= tR) { stack[sp] = n.right; tStack[sp] = tR; ++sp;
                                stack[sp] = n.left;  tStack[sp] = tL; ++sp; }
                else          { stack[sp] = n.left;  tStack[sp] = tL; ++sp;
                                stack[sp] = n.right; tStack[sp] = tR; ++sp; }
            } else if (hL) { stack[sp] = n.left;  tStack[sp] = tL; ++sp; }
            else if (hR)   { stack[sp] = n.right; tStack[sp] = tR; ++sp; }
        }
    }
    dApplyNormalMap(sc, h);   // C6: perturb shading normal by a bound normal map
    return h;
}

// Advance (ro,rd) through any GRIN region(s) via symplectic Eikonal marching, stopping when
// a surface is within one step or the ray has left all GRIN regions. Device twin of
// grin::march — the forward megakernel/wavefront call it before each bounce's closestHit,
// gated by sc.hasGrin so ordinary scenes never enter it (bit-identical). Kept byte-for-byte
// in step with the CPU marcher (grin.h) so CPU and GPU bend rays identically.
__device__ static void dGrinMarch(const DScene& sc, DVec3& ro, DVec3& rd) {
    const int GRIN_MAX_STEPS = 200000;
    // Accumulate position/direction in DOUBLE (not Real=float) so the running Eikonal
    // state mirrors the double-precision CPU marcher (grin.h). The symplectic update
    // compounds over up to hundreds of steps; carrying the running (ro,rd) in double is
    // the precision-correct choice for a pre-pass and costs essentially nothing here (it
    // runs once per bounce, not in the per-photon hot loop). We snapshot to a float DVec3
    // only for the geometry/pattern queries (closestHit / dMedInside / dMedClip / dMedNAt
    // / dMedGradN — which return their scalars in double anyway).
    //
    // NOTE (measured): this double accumulation does NOT close a residual GPU-vs-CPU
    // disagreement seen on a strong *radial* gradient lens (~17% mean rel-error inside the
    // caustic-heavy lens disc). That gap survives it unchanged — it is dominated by (a) a
    // pre-existing global ~1.2x mode-R float-GPU vs double-CPU exposure difference present
    // even with NO medium, and (b) the extreme ray->image magnification of a radial caustic
    // where the two backends' geometry/BLAS float paths diverge. A smooth *linear* gradient
    // lens matches CPU to ~the noise floor (see known-issues.md "mode-R GRIN radial caustic").
    double px = ro.x, py = ro.y, pz = ro.z;
    double dx = rd.x, dy = rd.y, dz = rd.z;
    // Hoisted out of the (up to 10^5-iteration) march: three pointer copies, so an `ior`
    // field can read a MEASURED index volume (`ior "1 + grid:n(x, y, z)"`).
    const DPatEnv env = dPatEnvOf(sc);
    for (int gstep = 0; gstep < GRIN_MAX_STEPS; ++gstep) {
        DVec3 cro{px, py, pz}, crd{dx, dy, dz};   // float snapshot for the geometry queries
        // GRIN region containing ro (first enabled GRIN membership), or -1.
        int gm = -1;
        for (int mi = 0; mi < sc.mediaN; ++mi) {
            const DMedium& md = sc.media[mi];
            if (md.enabled && md.iorN > 0 && dMedInside(md, cro, env)) { gm = mi; break; }
        }
        if (gm < 0) {
            // Outside any GRIN region: jump to the nearest GRIN entry before the next
            // surface, else stop (straight-ray body takes over). This branch needs the
            // full-range closest hit (dS bounds the entry search) but runs only at
            // region entries, not per Eikonal step.
            DHit hs = closestHit(sc, cro, crd);
            double dS = hs.valid ? (double)hs.t : 1e30;
            double bestTa = 1e30; int bestM = -1;
            for (int mi = 0; mi < sc.mediaN; ++mi) {
                const DMedium& md = sc.media[mi];
                if (!(md.enabled && md.iorN > 0)) continue;
                double ta, tb;
                if (dMedClip(md, cro, crd, 1e-4, dS, ta, tb) && ta < bestTa) { bestTa = ta; bestM = mi; }
            }
            if (bestM < 0) break;
            double adv = bestTa + 1e-4;
            px += dx * adv; py += dy * adv; pz += dz * adv;   // nudge inside
            continue;
        }
        const DMedium& g = sc.media[gm];
        double ds = g.iorStep;
        // Inside a GRIN region the ONLY question is "surface within one step?", so cap
        // the whole BVH walk at one step length: tcap is the smallest Real STRICTLY
        // greater than ds, making "found under the cap" ⟺ "(double)t <= ds" — exactly
        // the uncapped acceptance below, while pruning essentially the entire tree on
        // each of the up-to-10^5 Eikonal steps a ray spends inside the lens.
        Real tcap = (Real)ds;
#if FTRACE_GPU_FP32
        if ((double)tcap <= ds) tcap = nextafterf(tcap, FLT_MAX);
#else
        tcap = nextafter(tcap, DBL_MAX);
#endif
        DHit hs = closestHit(sc, cro, crd, RAY_EPS, tcap);
        if (hs.valid && (double)hs.t <= ds) break;   // surface within a step
        // Symplectic Eikonal step with optical direction T = n·d (|T| = n):
        //   T += ∇n·ds ;  x += (T/n)·ds ;  d = T/|T|.
        double n0 = dMedNAt(g, cro, env);
        DVec3 grad = dMedGradN(g, cro, 0.5 * ds, env);
        double Tx = dx * n0 + (double)grad.x * ds;
        double Ty = dy * n0 + (double)grad.y * ds;
        double Tz = dz * n0 + (double)grad.z * ds;
        double inv = ds / n0;
        px += Tx * inv; py += Ty * inv; pz += Tz * inv;
        double tl = sqrt(Tx * Tx + Ty * Ty + Tz * Tz);
        if (tl > 1e-12) { double s = 1.0 / tl; dx = Tx * s; dy = Ty * s; dz = Tz * s; }
    }
    ro = DVec3{px, py, pz};
    rd = DVec3{dx, dy, dz};
}

__device__ static bool occluded(const DScene& sc, const DVec3& o, const DVec3& dir,
                                 Real maxDist, Real tmin = RAY_EPS) {
    if (sc.nNodes == 0) return false;
    DVec3 invD{(Real)1 / dir.x, (Real)1 / dir.y, (Real)1 / dir.z};
    const DTriShear sh = makeTriShear(dir);
    const DCurveRay cray = sc.nCurveSegs ? makeCurveRay(dir) : DCurveRay{DVec3{(Real)0,(Real)0,(Real)1}, (Real)1};
    Real tMax = maxDist - tmin;
    // tMax is fixed for the whole walk: push-time tests suffice (see blasOccluded).
    Real tRoot;
    if (!boxHit(sc.nodes[0], o, invD, tmin, tMax, tRoot)) return false;
    int stack[64]; int sp = 0; stack[sp++] = 0;
    while (sp) {
        const DNode& n = sc.nodes[stack[--sp]];
        if (n.count > 0) {
            for (int i = 0; i < n.count; ++i) {
                int prim = sc.primIdx[n.first + i];
                DHit h; h.t = tMax; h.valid = false;
                bool blocked;
                if (prim < sc.nTris)                              blocked = intersectTri(sh, o, dir, sc.tris[prim], tmin, h);
                else if (prim < sc.nTris + sc.nSph)               blocked = intersectSphere(o, dir, sc.sph[prim - sc.nTris], tmin, h);
                else if (prim < sc.nTris + sc.nSph + sc.nImplicits) blocked = intersectImplicit(sc, sc.implicits[prim - sc.nTris - sc.nSph], o, dir, tmin, h, /*anyHit=*/true);
                else if (prim < sc.nTris + sc.nSph + sc.nImplicits + sc.nCurveSegs) blocked = intersectCurveSeg(cray, o, dir, sc.curveSegs[prim - sc.nTris - sc.nSph - sc.nImplicits], tmin, h, /*anyHit=*/true);
                else {
                    // Instance leaf: any-hit inside the shared BLAS in local space.
                    const DInstance& inst = sc.instances[prim - sc.nTris - sc.nSph - sc.nImplicits - sc.nCurveSegs];
                    DVec3 lo = affPoint(inst.Lm, inst.Lt, o);
                    DVec3 ld = affDir(inst.Lm, dir);
                    blocked = blasOccluded(sc, inst, lo, ld, tmin, tMax);
                }
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

// ---- `cavity` pattern variable (O3 stage 2) — device twin of scene.h cavityAt ----
// Fires cavitySamples short occlusion rays into the cosine-weighted hemisphere about
// the shaded-side GEOMETRIC normal and returns the blocked fraction: 0 on a lone
// plane, ~0.5 in a right-angled interior corner, ->1 deep in a crevice.
//
// This must stay BIT-IDENTICAL to the host, because `cavity` is a material input, not
// a transport estimator: a CPU and a GPU render of the same scene have to agree on the
// mix weight at every point, and a mixed forward/backward pipeline reads it from both.
// So the direction set is spelled out here the long way — the same golden-angle
// Fibonacci spiral, the same (i+0.5)/N stratification, and the same Duff ONB inlined
// verbatim from linalg.h's onb() rather than calling d3onb (which takes D3, not DVec3,
// and is free to differ). Everything is double even where Real is float, for the same
// reason: the direction set is scene data, not a sample.
//
// Deliberately NOT a pattern-VM function: DPatEnvT carries only sampler tables, and
// giving the VM a scene/BVH pointer to call through would cost far more than a
// variable slot. See the design note in todo.md O3 stage 2.
__device__ static double dCavityAt(const DScene& sc, const DHit& h) {
    const int N = sc.cavitySamples;
    if (N <= 0 || sc.cavityRadius <= 0.0) return 0.0;
    // Shaded-side geometric normal (orientedGeoN twin): the interpolated shading
    // normal could tilt the hemisphere into a coarse mesh and self-report occlusion.
    double nx = (double)h.ng.x, ny = (double)h.ng.y, nz = (double)h.ng.z;
    if (nx * (double)h.n.x + ny * (double)h.n.y + nz * (double)h.n.z < 0.0) {
        nx = -nx; ny = -ny; nz = -nz;
    }
    // Duff et al. branchless ONB — inlined from linalg.h onb() for bit-parity.
    const double sign = copysign(1.0, nz);
    const double a = -1.0 / (sign + nz);
    const double dd = nx * ny * a;
    const double tx = 1.0 + sign * nx * nx * a, ty = sign * dd,        tz = -sign * nx;
    const double bx = dd,                       by = sign + ny * ny * a, bz = -ny;
    // Offset along the normal by a hair so the probe does not re-hit its own surface.
    const DVec3 o{(Real)((double)h.p.x + nx * 1e-6),
                  (Real)((double)h.p.y + ny * 1e-6),
                  (Real)((double)h.p.z + nz * 1e-6)};
    const double R = sc.cavityRadius;
    const double golden = 3.14159265358979323846 * (3.0 - sqrt(5.0));
    int blocked = 0;
    for (int i = 0; i < N; ++i) {
        const double u  = (i + 0.5) / (double)N;
        const double sr = sqrt(u);          // radius in the projected disc
        const double cz = sqrt(1.0 - u);    // cos(theta) — the cosine weight
        const double ph = golden * i;
        const double cx = sr * cos(ph), cy = sr * sin(ph);
        const DVec3 d{(Real)(tx * cx + bx * cy + nx * cz),
                      (Real)(ty * cx + by * cy + ny * cz),
                      (Real)(tz * cx + bz * cy + nz * cz)};
        if (occluded(sc, o, d, (Real)R)) ++blocked;
    }
    return (double)blocked / (double)N;
}

// Lazy-fill accessor used at every pattern-eval site that has a surface hit. Cached on
// the DHit because one shading point builds several pattern contexts (mix weight,
// roughness, film thickness, a driven reflect record) and each probe costs N rays.
__device__ static inline double dCavityOf(const DScene& sc, const DHit& h) {
    // Doubly gated, exactly as on the host (see patCtxFromHit): scene-wide first —
    // one integer compare for the overwhelming majority of scenes, which never say
    // `cavity` — then per-material, so a patterned surface that does not read it pays
    // nothing even in a scene that does.
    if (!sc.needsCavity) return 0.0;
    if (h.matId < 0 || !sc.mats[h.matId].readsCavity) return 0.0;
    if (!h.cavityDone) { h.cavity = (Real)dCavityAt(sc, h); h.cavityDone = true; }
    return (double)h.cavity;
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
//
// `whittedWeight` (non-null only in mode W) switches the Fresnel coin flip for the DOMINANT
// branch (reflect iff R >= 0.5) and reports that branch's weight for the caller to fold into
// the throughput — same expected value, zero variance. It also suppresses the frosting
// perturbation, the other rng draw at this interface: at 1 spp a coin flip per pixel is not
// noise but salt-and-pepper, and glass rendered as a speckled blob. TIR reports weight 1.
__device__ static void refractOrReflect(const DScene& sc, const DMaterial& m, const DHit& h,
                                         const DVec3& d, Real lambda, DRng& rng,
                                         DVec3& ro, DVec3& rd, bool* transmitted = nullptr,
                                         Real extIor = (Real)1, double* whittedWeight = nullptr) {
    Real ng = specLookup(m.ior, lambda);
    bool entering = dot(d, h.ng) < 0;
    DVec3 nl = entering ? h.ng : -h.ng;
    Real n1 = entering ? extIor : ng, n2 = entering ? ng : extIor;
    Real eta = n1 / n2;
    Real cosI = -dot(d, nl);
    Real sin2t = eta * eta * ((Real)1 - cosI * cosI);
    DVec3 outDir;
    bool refracted = false;
    if (sin2t > 1) { outDir = reflectv(d, nl); if (whittedWeight) *whittedWeight = 1.0; }
    else {
        Real cosT = sqrt((Real)1 - sin2t);
        Real rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
        Real rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
        Real R = (Real)0.5 * (rs * rs + rp * rp);
        const bool doReflect = whittedWeight ? (R >= (Real)0.5) : (rng.uniform() < R);
        if (whittedWeight) *whittedWeight = doReflect ? (double)R : 1.0 - (double)R;
        if (doReflect) outDir = reflectv(d, nl);
        else { outDir = d * eta + nl * (eta * cosI - cosT); refracted = true; }
    }
    outDir = normalize(outDir);
    // Frosted glass: jitter the chosen lobe, keeping it on the intended side. Skipped in
    // mode W (see whittedWeight) — that draw is the other source of 1-spp salt-and-pepper.
    Real rough = whittedWeight ? (Real)0 : dMatRoughness(sc, m, h);
    if (rough > (Real)1e-3) {
        DVec3 pert = sampleGlossy(outDir, rough, rng);
        bool ok = refracted ? (dot(pert, nl) < 0) : (dot(pert, nl) > 0);
        if (ok) outDir = pert;
    }
    if (transmitted) *transmitted = refracted;
    ro = h.p + outDir * RAY_EPS; rd = outDir;
}

// Nested-dielectric PRIORITY step (Schmidt & Budge 2002), shared by every device transport
// loop. Resolves the exterior IOR at a dielectric interface from the enclosing medium (the
// highest-priority stack entry), suppresses lower-priority overlapping boundaries (straight
// pass-through), and maintains `stk`. `mi` is the resolved material index (Mix/Layered
// aware). SAFE FALLBACK: the priority rule applies only when BOTH sides carry an explicit
// priority (air always counts, IOR 1.0); otherwise this degrades to the old flat
// air<->glass model, so priority-free scenes render bit-identically. Mirrors the host.
__device__ static void dDielectricStep(const DScene& sc, const DMaterial& m, const DHit& h,
                                        const DVec3& d, Real lambda, DRng& rng,
                                        int mi, DMediumStack& stk, DVec3& outO, DVec3& outD,
                                        double* whittedWeight = nullptr) {
    // Mode W: the suppressed-boundary pass-throughs below take no Fresnel branch, so their
    // weight is 1; the two real interfaces overwrite this from refractOrReflect.
    if (whittedWeight) *whittedWeight = 1.0;
    bool entering = dot(d, h.ng) < 0;
    int pr = m.priority;
    if (entering) {
        int outMat = stk.topMat();
        int outPri = stk.topPri();
        bool ranked = dHasPriority(m) &&
            (stk.empty() || (outMat >= 0 && dHasPriority(sc.mats[outMat])));
        if (ranked && !stk.empty() && pr <= outPri) {   // suppressed inner surface
            stk.push(mi, pr);
            outO = h.p + d * RAY_EPS; outD = d; return;
        }
        Real extIor = (ranked && outMat >= 0) ? specLookup(sc.mats[outMat].ior, lambda) : (Real)1;
        bool transmitted = false; DVec3 nro, nrd;
        refractOrReflect(sc, m, h, d, lambda, rng, nro, nrd, &transmitted, extIor, whittedWeight);
        if (transmitted) stk.push(mi, pr);
        outO = nro; outD = nrd;
    } else {
        DMediumStack after = stk; after.popMat(mi);
        int newMat = after.topMat();
        int newPri = after.topPri();
        bool ranked = dHasPriority(m) &&
            (after.empty() || (newMat >= 0 && dHasPriority(sc.mats[newMat])));
        if (ranked && newMat >= 0 && pr <= newPri) {    // suppressed: still enclosed
            stk.popMat(mi);
            outO = h.p + d * RAY_EPS; outD = d; return;
        }
        Real extIor = (ranked && newMat >= 0) ? specLookup(sc.mats[newMat].ior, lambda) : (Real)1;
        bool transmitted = false; DVec3 nro, nrd;
        refractOrReflect(sc, m, h, d, lambda, rng, nro, nrd, &transmitted, extIor, whittedWeight);
        if (transmitted) stk.popMat(mi);                // TIR stays inside mi
        outO = nro; outD = nrd;
    }
}
// Returns false if the photon is absorbed by an opaque (absorbing) substrate.
// Forward decl: the per-hit thin-film thickness helper (definition with the other
// texture samplers, below dTexScalarAt) is used here before its point of definition.
__device__ static Real dMatFilmThickness(const DScene& sc, const DMaterial& m, const DHit& h);

// `whittedWeight` (non-null only in mode W) is the same deterministic contract
// refractOrReflect has: take the DOMINANT interference branch and report its weight for the
// caller to fold into the throughput, instead of tossing a coin against R. Opaque substrate
// has only one surviving branch, so it always reflects and reports R (reflectance as a weight
// rather than a survival probability, as Mirror/Filter already do in mode W).
__device__ static bool thinFilmInterface(const DScene& sc, const DMaterial& m, const DHit& h,
                                          const DVec3& d,
                                          Real lambda, DRng& rng, DVec3& ro, DVec3& rd,
                                          double* whittedWeight = nullptr) {
    Real ns = specLookup(m.ior, lambda), nf = (Real)m.filmIor;
    Real ks = specLookup(m.substrateK, lambda);
    Real thickness = dMatFilmThickness(sc, m, h);   // per-hit (map or constant)
    bool entering = dot(d, h.ng) < 0;
    DVec3 nl = entering ? h.ng : -h.ng;
    Real cosI = -dot(d, nl);
    if (ks > 0) {                                // opaque metal-backed film
        if (!entering) return false;             // inside absorbing substrate: absorbed
        Real R = thinFilmReflectance((Real)1, nf, ns, ks, thickness, cosI, lambda);
        if (whittedWeight) *whittedWeight = (double)R;   // weight, not a survival roll
        else if (rng.uniform() >= R) return false;       // transmitted -> absorbed
        DVec3 o = normalize(reflectv(d, nl));
        ro = h.p + o * RAY_EPS; rd = o;
        return true;
    }
    Real nA = entering ? (Real)1 : ns, nB = entering ? ns : (Real)1;
    Real eta = nA / nB;
    Real sin2t = eta * eta * ((Real)1 - cosI * cosI);
    DVec3 outDir;
    if (sin2t > 1) { outDir = reflectv(d, nl); if (whittedWeight) *whittedWeight = 1.0; }
    else {
        Real cosT = sqrt((Real)1 - sin2t);
        Real R = thinFilmReflectance(nA, nf, nB, (Real)0, thickness, cosI, lambda);
        const bool doReflect = whittedWeight ? (R >= (Real)0.5) : (rng.uniform() < R);
        if (whittedWeight) *whittedWeight = doReflect ? (double)R : 1.0 - (double)R;
        if (doReflect) outDir = reflectv(d, nl);
        else outDir = d * eta + nl * (eta * cosI - cosT);
    }
    outDir = normalize(outDir);
    ro = h.p + outDir * RAY_EPS; rd = outDir;
    return true;
}
// Multilayer stack interface (port of render.h multilayerInterface). Returns false
// if the photon is absorbed by an absorbing stack/substrate.
// `whittedWeight`: same deterministic dominant-branch contract as thinFilmInterface.
__device__ static bool multilayerInterface(const DMaterial& m, const DHit& h, const DVec3& d,
                                            Real lambda, DRng& rng, DVec3& ro, DVec3& rd,
                                            double* whittedWeight = nullptr) {
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
        if (whittedWeight) *whittedWeight = (double)R;   // weight, not a survival roll
        else if (rng.uniform() >= R) return false;
        DVec3 o = normalize(reflectv(d, nl)); ro = h.p + o * RAY_EPS; rd = o; return true;
    }
    Real nA = entering ? (Real)1 : ns, nB = entering ? ns : (Real)1;
    Real eta = nA / nB;
    Real sin2t = eta * eta * ((Real)1 - cosI * cosI);
    DVec3 outDir;
    if (sin2t > 1) { outDir = reflectv(d, nl); if (whittedWeight) *whittedWeight = 1.0; }
    else {
        Real cosT = sqrt((Real)1 - sin2t);
        Real R;
        if (entering) R = multilayerReflectance((Real)1, cosI, lambda, m.layerN, m.layerK, m.layerThick, nL, ns, (Real)0);
        else {
            double rn[D_MAXLAYERS], rk[D_MAXLAYERS], rt[D_MAXLAYERS];
            for (int j = 0; j < nL; ++j) { rn[j] = m.layerN[nL-1-j]; rk[j] = m.layerK[nL-1-j]; rt[j] = m.layerThick[nL-1-j]; }
            R = multilayerReflectance(ns, cosI, lambda, rn, rk, rt, nL, (Real)1, (Real)0);
        }
        const bool doReflect = whittedWeight ? (R >= (Real)0.5) : (rng.uniform() < R);
        if (whittedWeight) *whittedWeight = doReflect ? (double)R : 1.0 - (double)R;
        if (doReflect) outDir = reflectv(d, nl);
        else outDir = d * eta + nl * (eta * cosI - cosT);
    }
    outDir = normalize(outDir); ro = h.p + outDir * RAY_EPS; rd = outDir; return true;
}
// Grating diffraction (port of render.h gratingDiffract). Returns false if absorbed.
// `whittedU` (non-null only in mode W) replaces the rng draw with a coordinate off the
// deterministic (sIdx, bounce) lattice, and switches the candidate walk to DESCENDING
// efficiency (0, -1, +1, -2, +2, ...) so u = 0 selects the specular order m = 0 -- see the host
// gratingDiffract for the full rationale. The pick is analog either way, so this is a variance
// change only, not an estimator change.
__device__ static bool gratingDiffract(const DMaterial& m, const DHit& h, const DVec3& din,
                                        Real lambda, int diffraction, DRng& rng,
                                        DVec3& ro, DVec3& rd,
                                        const double* whittedU = nullptr) {
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
    int slot[65];                                   // mm+M -> index in ord[], -1 evanescent
    if (whittedU) for (int i = 0; i <= 2 * M; ++i) slot[i] = -1;
    for (int mm = -M; mm <= M; ++mm) {
        DVec3 a = ut + t * ((Real)mm * lod);
        if (dot(a, a) >= 1) continue;
        Real w = (Real)1 / ((Real)1 + (mm < 0 ? -mm : mm));
        if (whittedU) slot[mm + M] = cnt;
        ord[cnt] = mm; wgt[cnt] = w; wsum += w; ++cnt;
    }
    if (cnt == 0 || wsum <= 0) return false;
    int pick;
    if (whittedU) {
        // Deterministic order pick. Done in DOUBLE, unlike the stochastic path's Real, and
        // recomputing each 1/(1+|m|) from the order rather than reading the Real wgt[] above:
        // the weights are small exact rationals, so accumulating them in double in the same
        // sequence the host uses makes the selection BIT-IDENTICAL to the CPU. That matters far
        // more here than anywhere else in the port -- picking a neighbouring order sends the ray
        // in a visibly different direction, so an fp32 tie-break near a cumulative boundary
        // would be a structural CPU/GPU difference rather than the usual silhouette sliver.
        double wsumD = 0.0;                         // host's ascending-mm accumulation order
        for (int i = 0; i < cnt; ++i) wsumD += 1.0 / (1.0 + (double)(ord[i] < 0 ? -ord[i] : ord[i]));
        double xi = *whittedU * wsumD, acc = 0.0;
        pick = ord[cnt - 1];                        // guard against fp round-off at u->1
        bool done = false;
        for (int k = 0; k <= M && !done; ++k) {      // descending efficiency: 0, -1, +1, -2, ...
            for (int s = 0; s < (k == 0 ? 1 : 2); ++s) {
                int idx = slot[(s == 0 ? -k : k) + M];
                if (idx < 0) continue;              // that order is evanescent
                acc += 1.0 / (1.0 + (double)k); pick = ord[idx];
                if (xi < acc) { done = true; break; }
            }
        }
    } else {
        Real xi = rng.uniform() * wsum, acc = 0; pick = ord[cnt - 1];
        for (int i = 0; i < cnt; ++i) { acc += wgt[i]; if (xi < acc) { pick = ord[i]; break; } }
    }
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
// Device twin of geometry.h's shadingAdjointCorr (Veach §5.3 adjoint shading-normal
// correction). Reweights a forward/particle vertex so an interpolated shading normal
// `ns` shades smoothly instead of faceting; EXACTLY 1 when ns==ng (flat tris, analytic
// spheres), so every flat/analytic GPU scene stays bit-identical. `wi` = toward the
// previous (light-side) vertex, `wo` = the outgoing direction.
__device__ static Real dShadingAdjointCorr(const DVec3& wi, const DVec3& wo,
                                           const DVec3& ns, const DVec3& ng) {
    Real denom = (Real)fabs((double)dot(wi, ng)) * (Real)fabs((double)dot(wo, ns));
    if (denom <= (Real)1e-8) return (Real)1;          // degenerate grazing -> no correction
    Real num = (Real)fabs((double)dot(wi, ns)) * (Real)fabs((double)dot(wo, ng));
    return num / denom;
}
// Device twin of geometry.h's shadowTerminatorG (Chiang, Li, Burley & Hovhannisyan 2019
// "Taming the Shadow Terminator"; Cycles' bump_shadowing_term). Returns a [0,1] factor
// that REPLACES the old hard geometric-hemisphere cutoff (dot(ng,wi)<=0 ? reject): still
// exactly 0 when `wi` is behind the true geometry (no back-face light leak), but ramps up
// smoothly off the geometric horizon so a low-poly smooth-normal mesh shows a smooth
// terminator instead of facet slivers. EXACTLY 1 when ns==ng (flat tris, analytic spheres),
// so every flat/analytic GPU scene stays bit-identical. `wi` = direction toward the
// light/connection, `ns` = shading normal, `ng` = geo normal oriented onto the shading side.
__device__ static Real dShadowTerminatorG(const DVec3& wi, const DVec3& ns, const DVec3& ng) {
    Real cosNgNs = dot(ng, ns);
    // Exact no-op when ns==ng (flat tris, analytic spheres): the cubic would otherwise drift
    // by ~1e-7 since a re-normalized `ns` differs from `ng` in the last bit. Short-circuit to
    // a plain leak-free step (identical to the old hard clamp) so flat/analytic GPU scenes
    // stay bit-identical; softening engages only once ns and ng genuinely diverge.
    if (cosNgNs >= (Real)1 - (Real)1e-7) return (dot(ng, wi) > (Real)0) ? (Real)1 : (Real)0;
    Real cosNgWi = dot(ng, wi);
    if (cosNgWi <= (Real)0) return (Real)0;            // behind the true geometry -> hard shadow
    Real denom = dot(ns, wi) * cosNgNs;
    if (denom <= (Real)1e-8) return (Real)1;           // degenerate grazing -> no softening
    Real g = cosNgWi / denom;
    if (g >= (Real)1) return (Real)1;                  // fully lit -> no darkening
    return g * g * ((Real)1 - g) + g;                  // Chiang cubic: -g^3 + g^2 + g
}
__device__ static void connect(const DScene& sc, const DCamera& cam, double* film, double* hits,
                               const DVec3& p, const DVec3& n, const DVec3& ng, const DVec3& wi,
                               Real lambda, Real beta, Real rho, DRng& rng) {
    DVec3 toCam = cam.eye - p;
    Real dist = length(toCam);
    DVec3 wdir = toCam / dist;
    Real cosSurf = dot(n, wdir);
    // Reject below the shading horizon; soften across the GEOMETRIC horizon (ng = geo normal
    // on the shading side): a smoothed shading normal must not splat a vertex whose true
    // geometry faces away from the camera, but a hard cutoff facets the terminator, so ramp
    // it smoothly (Chiang 2019). No-op for flat tris / analytic spheres (ng == n, stG == 1).
    // Matches CPU render.h.
    if (cosSurf <= 0) return;
    Real stG = dShadowTerminatorG(wdir, n, ng);
    if (stG <= (Real)0) return;
    int px, py; Real cosCam, dist2;
    if (!cam.project(p, px, py, cosCam, dist2)) return;
    if (occluded(sc, p + ng * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    Real f = rho / (Real)DPI;
    // Projection-general splat: contrib = beta*f*cosSurf*corr / (dist^2 * pixelSolidAngle).
    // For a rectilinear lens pixelSolidAngle = pixelPlaneArea*cosCam^3, recovering the
    // classic 1/(A_pix cos^4) form; fisheye/panoramic uses the remapped solid angle.
    // corr is the Veach adjoint shading-normal factor (1 when ns==ng).
    Real corr = dShadingAdjointCorr(wi, wdir, n, ng);
    double solidAngle = cam.pixelSolidAngle(cosCam);
    Real contrib = beta * f * cosSurf * corr / (Real)((double)dist2 * solidAngle) * stG;
    if (sc.mediaN > 0) contrib *= dMediaTransmittance(sc, p, wdir, dist, lambda, rng);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}
// `med` is the medium that scattered the photon (its phase/albedo); transmittance is
// over ALL media (product).
__device__ static void connectVolume(const DScene& sc, const DMedium& med, const DCamera& cam,
                                      double* film, double* hits,
                                      const DVec3& p, const DVec3& wIn, Real lambda, Real beta,
                                      DRng& rng) {
    DVec3 toCam = cam.eye - p;
    Real dist = length(toCam);
    DVec3 wdir = toCam / dist;
    int px, py; Real cosCam, dist2;
    if (!cam.project(p, px, py, cosCam, dist2)) return;
    if (occluded(sc, p + wdir * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    Real ph = dMedPhase(med, dot(wIn, wdir), lambda);
    Real Lambda = medAlbedo(med, lambda);
    // Projection-general splat (no cosSurf for a volume vertex): the phase carries the
    // scattering; normalise by dist^2 * pixelSolidAngle (rectilinear or fisheye).
    double solidAngle = cam.pixelSolidAngle(cosCam);
    Real contrib = beta * Lambda * ph / (Real)((double)dist2 * solidAngle);
    contrib *= dMediaTransmittance(sc, p, wdir, dist, lambda, rng);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}
// Model A (physical camera): next-event splat through the finite lens pupil. Sample a
// point A uniformly on the aperture disc, connect the surface vertex to A, refract
// through the thin lens and splat onto the film cell A images to. Port of
// Renderer::connectLens — the importance-sampled form of model C's brute-force catch,
// so A and C share both scale and shape. Weight beta*rho*cosSurf*cosLens*R^2/dist^2
// (the BRDF's 1/pi cancels the pupil pdf's pi R^2).
__device__ static void connectLens(const DScene& sc, const DCamera& cam, double* film, double* hits,
                                   const DVec3& p, const DVec3& n, const DVec3& ng, const DVec3& wi,
                                   Real lambda, Real beta, Real rho, DRng& rng) {
    Real R  = (Real)cam.apertureR;
    Real rr = R * sqrt(rng.uniform());
    Real a  = (Real)(2.0 * DPI) * rng.uniform();
    DVec3 A = cam.eye + cam.u * (rr * cos(a)) + cam.v * (rr * sin(a));
    DVec3 toA = A - p;
    Real dist = length(toA);
    if (dist < (Real)1e-9) return;
    DVec3 wdir = toA / dist;
    Real cosSurf = dot(n, wdir);
    // Below the shading horizon reject; soften across the geometric horizon (see connect()):
    // no-op for flat/sphere (stG == 1).
    if (cosSurf <= 0) return;                        // pupil behind the shading surface
    Real stG = dShadowTerminatorG(wdir, n, ng);
    if (stG <= (Real)0) return;                      // pupil behind true geometry: hard cutoff
    Real cosLens = -dot(wdir, cam.w);                // cosine at the lens (w faces scene)
    if (cosLens <= (Real)1e-6) return;               // not heading toward the film
    int px, py;
    if (!cam.lensImage(A, wdir, px, py)) return;
    if (occluded(sc, p + ng * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    Real corr = dShadingAdjointCorr(wi, wdir, n, ng);   // Veach adjoint (1 when ns==ng)
    Real contrib = beta * rho * cosSurf * corr * cosLens * (R * R) / (dist * dist) * stG;
    // ABSOLUTE-SCALE NORMALISER (A/C <-> B unification) — CPU twin: render.h connectLens.
    // Divide the pupil FLUX deposited in the cell by the physical cell area
    // A_cell = pixelPlaneArea()*filmDist^2 to turn it into film IRRADIANCE, matching
    // mode B's radiance*camEq absolute scale. Per-camera constant; auto-exposed scenes
    // stay byte-identical, only absolute-EV A/C are re-seated to mid-tone at gain 6.
    contrib *= (Real)1 / (Real)(cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
    if (sc.mediaN > 0) contrib *= dMediaTransmittance(sc, p, wdir, dist, lambda, rng);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}
// Model A lens splat for a VOLUME scattering vertex (fog). As connectLens but the
// surface BRDF*cosSurf is replaced by albedo*phase; the phase carries no 1/pi, so the
// pupil pdf's pi R^2 stays. Port of Renderer::connectLensVolume. `med` is the scattering
// medium; transmittance is over all media.
__device__ static void connectLensVolume(const DScene& sc, const DMedium& med, const DCamera& cam,
                                         double* film, double* hits,
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
    Real ph = dMedPhase(med, dot(wIn, wdir), lambda);
    Real Lambda = medAlbedo(med, lambda);
    Real contrib = beta * Lambda * ph * cosLens * (Real)DPI * (R * R) / (dist * dist);
    // Same flux->film-irradiance normaliser as connectLens (see there); per-camera
    // constant, so auto-exposed scenes are unaffected.
    contrib *= (Real)1 / (Real)(cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
    contrib *= dMediaTransmittance(sc, p, wdir, dist, lambda, rng);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}

// ==================== hair fiber BCSDF (device twin of hair.h) ====================
// Port of src/hair.h's single-fiber Marschner/Chiang BCSDF — lobes p = 0 (R), 1 (TT),
// 2 (TRT) and a p >= 3 residual — plus the Yan 2017 medulla scattered lobes TT^s / TRT^s
// (kappa > 0). See hair.h for the full physics commentary; this is a line-for-line
// transliteration, kept in the same order so the two files can be diffed.
//
// NOT ported: dual scattering (hair.h lines 754+ / hair_shade.h's HairDualCtx) and the
// fur-volume aggregate — both stay CPU-only (main.cpp's backwardOnGpuOk gates them off
// the GPU path), exactly like the CPU-only Layered material.
//
// PRECISION. The internal math is deliberately DOUBLE even when Real is float. The
// longitudinal M_p works in log space with exp/log cancellations at magnitude ~1/v
// (tens of thousands for a smooth fiber), where float's 24-bit mantissa corrupts the
// exponent by whole units; and the residual-lobe energy identities (furnace closure,
// -checkhair §1) were established in double. The BCSDF runs a handful of times per hair
// VERTEX while the BVH traverses hundreds of thousands of curve segments per RAY, so
// FP64 here is not the bottleneck. Only the frame/impact-parameter geometry (built from
// Real-precision hit data, which is already float-noisy) stays in Real.
namespace dhair {

constexpr double kPi = 3.14159265358979323846;
constexpr int kPMax  = 3;                    // R, TT, TRT + residual index
constexpr int kSTT   = kPMax + 1;            // 4: medulla-scattered TT^s
constexpr int kSTRT  = kPMax + 2;            // 5: medulla-scattered TRT^s
constexpr int kNLobes = kPMax + 3;           // 6

__device__ static inline double sqr(double x) { return x * x; }
__device__ static inline double safeSqrt(double x) { return sqrt(fmax(0.0, x)); }
__device__ static inline double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

// Minimal double-precision direction triple for the LOCAL FIBER FRAME (+x = tangent).
// Not DVec3: the BCSDF math stays double regardless of Real (see precision note above).
struct V3 { double x, y, z; };

// Bessel I0 in two pure halves, deliberately NOT the CPU's mutually-recursive
// besselI0 <-> logBesselI0 shape: on the device that static cycle (each one's
// other-range branch calling the other) made nvlink declare the stack of EVERY
// kernel that can reach dhair "cannot be statically determined" (kTrace, kBackward,
// kWfShade). Undetermined kernels lose their MIN_STACK attribute, so the driver
// stops auto-reserving their real stack need (~7-12 KB measured on 0.180.x) and
// falls back to the 1 KB cudaLimitStackSize default -- and every render, hair or
// not, died of device stack overflow. Two cross-call-free cores keep the call graph
// acyclic; each public wrapper picks the branch with the same expressions as
// before, so values are bit-identical.
//
// Small-x core: ascending series sum_k (x/2)^{2k} / (k!)^2, iterated to double
// convergence (used for |x| <= 12).
__device__ static double besselI0Series(double x) {
    const double q = 0.25 * x * x;
    double t = 1.0, sum = 1.0;
    for (int k = 1; k < 64; ++k) {
        t *= q / (double(k) * double(k));
        sum += t;
        if (t < 1e-18 * sum) break;
    }
    return sum;
}
// Large-x core (|x| > 12), in log space so it cannot overflow:
// A&S 9.7.1: I0(x) ~ e^x / sqrt(2 pi x) * sum_k prod_j (2j-1)^2 / (k! (8x)^k).
__device__ static double logBesselI0Asym(double x) {
    double t = 1.0, sum = 1.0;
    for (int k = 1; k <= 12; ++k) {
        t *= sqr(2.0 * k - 1.0) / (8.0 * x * double(k));
        sum += t;
    }
    return x - 0.5 * log(2.0 * kPi * x) + log(sum);
}
__device__ static double besselI0(double x) {
    x = fabs(x);
    return (x > 12.0) ? exp(logBesselI0Asym(x)) : besselI0Series(x);
}
__device__ static double logBesselI0(double x) {
    x = fabs(x);
    return (x > 12.0) ? logBesselI0Asym(x) : log(besselI0Series(x));
}

// Longitudinal lobe M_p: normalised Gaussian on the sphere (d'Eon); log branch for
// small v where the Bessel I0 normalisation overflows.
__device__ static double Mp(double cosThetaI, double cosThetaO,
                            double sinThetaI, double sinThetaO, double v) {
    const double a = cosThetaI * cosThetaO / v;
    const double b = sinThetaI * sinThetaO / v;
    if (v <= 0.1)
        return exp(logBesselI0(a) - b - 1.0 / v + 0.6931471805599453 + log(0.5 / v));
    return (exp(-b) * besselI0(a)) / (sinh(1.0 / v) * 2.0 * v);
}

// Azimuthal machinery: exact specular exit angle Phi + trimmed-logistic smear N_p.
__device__ static inline double Phi(int p, double gammaO, double gammaT) {
    return 2.0 * p * gammaT - 2.0 * gammaO + p * kPi;
}
__device__ static inline double logistic(double x, double s) {
    x = fabs(x);
    const double e = exp(-x / s);
    return e / (s * sqr(1.0 + e));
}
__device__ static inline double logisticCdf(double x, double s) { return 1.0 / (1.0 + exp(-x / s)); }
__device__ static inline double trimmedLogistic(double x, double s, double a, double b) {
    return logistic(x, s) / (logisticCdf(b, s) - logisticCdf(a, s));
}
__device__ static inline double sampleTrimmedLogistic(double u, double s, double a, double b) {
    const double k = logisticCdf(b, s) - logisticCdf(a, s);
    double x = -s * log(1.0 / (u * k + logisticCdf(a, s)) - 1.0);
    return clampd(x, a, b);
}
__device__ static inline double wrapAngle(double phi) {
    while (phi > kPi)  phi -= 2.0 * kPi;
    while (phi < -kPi) phi += 2.0 * kPi;
    return phi;
}
__device__ static inline double Np(double phi, int p, double s, double gammaO, double gammaT) {
    return trimmedLogistic(wrapAngle(phi - Phi(p, gammaO, gammaT)), s, -kPi, kPi);
}

__device__ static double frDielectric(double cosThetaI, double etaI, double etaT) {
    cosThetaI = clampd(cosThetaI, -1.0, 1.0);
    if (cosThetaI < 0.0) { double tmp = etaI; etaI = etaT; etaT = tmp; cosThetaI = -cosThetaI; }
    const double sinThetaI = safeSqrt(1.0 - sqr(cosThetaI));
    const double sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.0) return 1.0;                        // total internal reflection
    const double cosThetaT = safeSqrt(1.0 - sqr(sinThetaT));
    const double rParl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                         ((etaT * cosThetaI) + (etaI * cosThetaT));
    const double rPerp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                         ((etaI * cosThetaI) + (etaT * cosThetaT));
    return 0.5 * (rParl * rParl + rPerp * rPerp);
}

// Attenuation A_p: Fresnel at each crossing + Beer-Lambert through the interior; the
// p = kPMax entry is the closed-form geometric tail (energy conservation).
__device__ static void Ap(double cosThetaO, double eta, double h, double T, double ap[kPMax + 1]) {
    const double cosGammaO = safeSqrt(1.0 - h * h);
    const double f = frDielectric(cosThetaO * cosGammaO, 1.0, eta);
    ap[0] = f;                                   // R
    ap[1] = sqr(1.0 - f) * T;                    // TT
    for (int p = 2; p < kPMax; ++p) ap[p] = ap[p - 1] * T * f;
    const double denom = 1.0 - T * f;
    ap[kPMax] = (denom > 1e-12) ? ap[kPMax - 1] * f * T / denom : 0.0;
}

// Authored parameters (device twin of hair::Params; see hair.h for semantics).
struct Params {
    double eta   = 1.55;
    double betaM = 0.3;
    double betaN = 0.3;
    double alpha = 2.0;
    double kappa   = 0.0;   // medullary index, [0, 1)
    double mSigmaS = 0.0;   // medulla scattering, per fiber radius
    double mSigmaA = 0.0;   // medulla absorption, per fiber radius
    double mG      = 0.0;   // medulla HG anisotropy
};

// Hit-and-parameter constants hoisted out of f()/pdf()/sample().
struct Bcsdf {
    double h = 0.0, gammaO = 0.0;
    double eta = 1.55, sigmaA = 0.0;
    double v[kPMax + 1] = {0, 0, 0, 0};   // longitudinal variances per lobe
    double s = 0.0;                       // azimuthal logistic scale
    double sin2kAlpha[3] = {0, 0, 0}, cos2kAlpha[3] = {1, 1, 1};
    bool   hasMedulla = false;
    double kappa = 0.0, mSigmaS = 0.0, mSigmaA = 0.0, mG = 0.0;
};

// One interior traversal: refracted azimuth + the five chord-derived survival terms.
struct Chord {
    double gammaT   = 0.0;
    double T        = 1.0;   // survival of the unscattered path across one traversal
    double Tsolid   = 1.0;   // same with the medulla replaced by cortex
    double albedoM  = 0.0;   // medulla single-scattering albedo
    double Tb       = 1.0;   // Yan eq. 20 post-scatter escape
    double tauP     = 0.0;   // reduced optical depth (1-g) sigma_s chord
};

__device__ static Bcsdf make(const Params& pr, double h, double sigmaA) {
    Bcsdf b;
    b.h      = clampd(h, -1.0, 1.0);
    b.gammaO = asin(b.h);
    b.eta    = pr.eta;
    b.sigmaA = fmax(0.0, sigmaA);
    // Chiang's perceptual-roughness fits (hair.h::make).
    const double bm = clampd(pr.betaM, 1e-4, 1.0);
    const double bn = clampd(pr.betaN, 1e-4, 1.0);
    b.v[0] = sqr(0.726 * bm + 0.812 * sqr(bm) + 3.7 * pow(bm, 20.0));
    b.v[1] = 0.25 * b.v[0];
    b.v[2] = 4.0 * b.v[0];
    for (int p = 3; p <= kPMax; ++p) b.v[p] = b.v[2];
    b.s = 0.626657069 *      // sqrt(pi/8)
          (0.265 * bn + 1.194 * sqr(bn) + 5.372 * pow(bn, 22.0));
    // Cuticle tilt double-angle recurrence.
    b.sin2kAlpha[0] = sin(pr.alpha * kPi / 180.0);
    b.cos2kAlpha[0] = safeSqrt(1.0 - sqr(b.sin2kAlpha[0]));
    for (int i = 1; i < 3; ++i) {
        b.sin2kAlpha[i] = 2.0 * b.cos2kAlpha[i - 1] * b.sin2kAlpha[i - 1];
        b.cos2kAlpha[i] = sqr(b.cos2kAlpha[i - 1]) - sqr(b.sin2kAlpha[i - 1]);
    }
    b.kappa   = clampd(pr.kappa, 0.0, 0.999);
    b.mSigmaS = fmax(0.0, pr.mSigmaS);
    b.mSigmaA = fmax(0.0, pr.mSigmaA);
    b.mG      = clampd(pr.mG, -0.999, 0.999);
    b.hasMedulla = (b.kappa > 0.0) && (b.mSigmaS + b.mSigmaA > 0.0);
    return b;
}

// Apply the lobe-p cuticle tilt to the outgoing longitudinal angle.
__device__ static void tiltO(const Bcsdf& b, int p, double sinThetaO, double cosThetaO,
                             double& sinThetaOp, double& cosThetaOp) {
    if (p == 0) {
        sinThetaOp = sinThetaO * b.cos2kAlpha[1] - cosThetaO * b.sin2kAlpha[1];
        cosThetaOp = cosThetaO * b.cos2kAlpha[1] + sinThetaO * b.sin2kAlpha[1];
    } else if (p == 1) {
        sinThetaOp = sinThetaO * b.cos2kAlpha[0] + cosThetaO * b.sin2kAlpha[0];
        cosThetaOp = cosThetaO * b.cos2kAlpha[0] - sinThetaO * b.sin2kAlpha[0];
    } else if (p == 2) {
        sinThetaOp = sinThetaO * b.cos2kAlpha[2] + cosThetaO * b.sin2kAlpha[2];
        cosThetaOp = cosThetaO * b.cos2kAlpha[2] - sinThetaO * b.sin2kAlpha[2];
    } else {
        sinThetaOp = sinThetaO;
        cosThetaOp = cosThetaO;
    }
    cosThetaOp = fabs(cosThetaOp);
}

// Refraction geometry + interior chords (Bravais virtual index; Yan's double cylinder).
__device__ static Chord refractGeom(const Bcsdf& b, double sinThetaO, double cosThetaO) {
    Chord ch;
    const double etap      = safeSqrt(sqr(b.eta) - sqr(sinThetaO)) / fmax(cosThetaO, 1e-9);
    const double sinGammaT = clampd(b.h / etap, -1.0, 1.0);
    const double cosGammaT = safeSqrt(1.0 - sqr(sinGammaT));
    ch.gammaT = asin(sinGammaT);
    const double sinThetaT = sinThetaO / b.eta;
    const double cosThetaT = safeSqrt(1.0 - sqr(sinThetaT));
    const double invCosT = 1.0 / fmax(cosThetaT, 1e-9);

    if (!b.hasMedulla) {
        ch.T = exp(-b.sigmaA * (2.0 * cosGammaT * invCosT));
        ch.Tsolid = ch.T;
        return ch;
    }

    const double sm = safeSqrt(sqr(b.kappa) - sqr(sinGammaT));   // half-chord in medulla
    const double sc = fmax(0.0, cosGammaT - sm);                 // half-chord in cortex
    const double mExt = b.mSigmaS + b.mSigmaA;                   // medulla extinction
    ch.T      = exp(-(2.0 * sc * b.sigmaA + 2.0 * sm * mExt) * invCosT);
    ch.Tsolid = exp(-(2.0 * cosGammaT * b.sigmaA) * invCosT);
    ch.albedoM = (mExt > 1e-12) ? b.mSigmaS / mExt : 0.0;
    ch.Tb = exp(-(b.kappa * b.mSigmaA + (1.0 - b.kappa) * b.sigmaA) * invCosT);
    ch.tauP = (1.0 - clampd(b.mG, -0.999, 0.999)) * b.mSigmaS * (2.0 * sm) * invCosT;
    return ch;
}

// Similarity-theory smear of the scattered lobes' exit direction, in [0, 1).
__device__ static inline double scatteredSpread(const Chord& ch) {
    return 1.0 - exp(-fmax(0.0, ch.tauP));
}

// Attenuation of the scattered lobes A^s_p, computed as a DIFFERENCE so the white
// furnace closes exactly (see hair.h::ApScattered).
__device__ static void ApScattered(const Chord& ch, double cosThetaO, double eta, double h,
                                   double aps[2]) {
    aps[0] = aps[1] = 0.0;
    if (ch.albedoM <= 0.0) return;
    double apMed[kPMax + 1], apSolid[kPMax + 1];
    Ap(cosThetaO, eta, h, ch.T, apMed);
    Ap(cosThetaO, eta, h, ch.Tsolid, apSolid);
    double missing = 0.0;
    for (int p = 0; p <= kPMax; ++p) missing += apSolid[p] - apMed[p];
    if (!(missing > 0.0)) return;                 // Tsolid >= T always, but be defensive
    const double total = missing * ch.albedoM * ch.Tb;
    const double cosGammaO = safeSqrt(1.0 - h * h);
    const double F = frDielectric(cosThetaO * cosGammaO, 1.0, eta);
    const double norm = 1.0 + F;
    aps[0] = total * (1.0 / norm);
    aps[1] = total * (F / norm);
}

// Exit azimuth of a scattered lobe (Yan eq. 21).
__device__ static inline double PhiS(int p, double gammaO, double gammaT) {
    return (gammaT - gammaO) + double(p - 1) * (kPi + 2.0 * gammaT);
}
__device__ static inline double scatteredV(const Bcsdf& b, int p, double spread) {
    return b.v[p] + spread;
}
__device__ static inline double scatteredS(const Bcsdf& b, double spread) {
    return b.s + 2.0 * spread;
}

// The BCSDF value for wo -> wi (local fiber frame). Already divided by |cos theta_i|
// (surface-BRDF convention; see hair.h's cosine note).
//
// f / pdf / sample / dHairShadeAt are all __noinline__ ON PURPOSE: they carry big
// double-precision frames (per-lobe attenuation + Mp/Np locals), and several SHARED
// functions (bkEmitterGeom, bkEnvGeom, interactSpecular's dispatch) contain guarded
// calls to them. Inlined, those locals would merge permanently into the shared tracer
// frames, inflating every kernel's statically-computed MIN_STACK (the per-thread
// stack the driver auto-reserves at launch) — and the driver multiplies that by
// ~200k resident threads of VRAM, hair scene or not. As calls, the frames only go
// live on actual hair evaluations, and MIN_STACK grows by the one deepest hair
// chain instead of leaking hair bytes into every frame on the path.
__device__ __noinline__ static double f(const Bcsdf& b, const V3& wo, const V3& wi) {
    const double sinThetaO = clampd(wo.x, -1.0, 1.0), cosThetaO = safeSqrt(1.0 - sqr(sinThetaO));
    const double sinThetaI = clampd(wi.x, -1.0, 1.0), cosThetaI = safeSqrt(1.0 - sqr(sinThetaI));
    const double phiO = atan2(wo.z, wo.y);
    const double phiI = atan2(wi.z, wi.y);
    const double phi  = phiI - phiO;

    const Chord ch = refractGeom(b, sinThetaO, cosThetaO);
    const double gammaT = ch.gammaT;

    double ap[kPMax + 1];
    Ap(cosThetaO, b.eta, b.h, ch.T, ap);

    double sum = 0.0;
    for (int p = 0; p < kPMax; ++p) {
        double sinThetaOp, cosThetaOp;
        tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);
        sum += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, b.v[p]) * ap[p] *
               Np(phi, p, b.s, b.gammaO, gammaT);
    }
    sum += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, b.v[kPMax]) * ap[kPMax] *
           (0.5 / kPi);

    if (b.hasMedulla) {
        double aps[2];
        ApScattered(ch, cosThetaO, b.eta, b.h, aps);
        const double spread = scatteredSpread(ch);
        const double ss = scatteredS(b, spread);
        for (int k = 0; k < 2; ++k) {
            if (aps[k] <= 0.0) continue;
            const int p = k + 1;                       // TT^s -> 1, TRT^s -> 2
            double sinThetaOp, cosThetaOp;
            tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);
            sum += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp,
                      scatteredV(b, p, spread)) * aps[k] *
                   trimmedLogistic(wrapAngle(phi - PhiS(p, b.gammaO, gammaT)), ss, -kPi, kPi);
        }
    }

    const double absCosI = fabs(cosThetaI);
    if (absCosI > 1e-9) sum /= absCosI;
    return isfinite(sum) ? fmax(0.0, sum) : 0.0;
}

// Discrete lobe-choice probabilities, proportional to the energy each carries.
__device__ static void apPdf(const Bcsdf& b, double sinThetaO, double cosThetaO, double pdf[kNLobes]) {
    const Chord ch = refractGeom(b, sinThetaO, cosThetaO);
    double ap[kPMax + 1];
    Ap(cosThetaO, b.eta, b.h, ch.T, ap);
    double aps[2] = {0.0, 0.0};
    if (b.hasMedulla) ApScattered(ch, cosThetaO, b.eta, b.h, aps);
    double total = 0.0;
    for (int p = 0; p <= kPMax; ++p) total += ap[p];
    total += aps[0] + aps[1];
    if (total <= 1e-12) {                     // degenerate: fall back to uniform
        const int n = b.hasMedulla ? kNLobes : (kPMax + 1);
        for (int p = 0; p < kNLobes; ++p) pdf[p] = (p < n) ? 1.0 / double(n) : 0.0;
        return;
    }
    for (int p = 0; p <= kPMax; ++p) pdf[p] = ap[p] / total;
    pdf[kSTT]  = aps[0] / total;
    pdf[kSTRT] = aps[1] / total;
}

__device__ __noinline__ static double pdf(const Bcsdf& b, const V3& wo, const V3& wi) {   // __noinline__: see f above
    const double sinThetaO = clampd(wo.x, -1.0, 1.0), cosThetaO = safeSqrt(1.0 - sqr(sinThetaO));
    const double sinThetaI = clampd(wi.x, -1.0, 1.0), cosThetaI = safeSqrt(1.0 - sqr(sinThetaI));
    const double phi = atan2(wi.z, wi.y) - atan2(wo.z, wo.y);

    const Chord ch = refractGeom(b, sinThetaO, cosThetaO);
    const double gammaT = ch.gammaT;
    double ap[kNLobes];
    apPdf(b, sinThetaO, cosThetaO, ap);

    double sum = 0.0;
    for (int p = 0; p < kPMax; ++p) {
        double sinThetaOp, cosThetaOp;
        tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);
        sum += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, b.v[p]) * ap[p] *
               Np(phi, p, b.s, b.gammaO, gammaT);
    }
    sum += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, b.v[kPMax]) * ap[kPMax] *
           (0.5 / kPi);
    if (b.hasMedulla) {
        const double spread = scatteredSpread(ch);
        const double ss = scatteredS(b, spread);
        for (int k = 0; k < 2; ++k) {
            const double w = ap[kSTT + k];
            if (w <= 0.0) continue;
            const int p = k + 1;
            double sinThetaOp, cosThetaOp;
            tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);
            sum += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp,
                      scatteredV(b, p, spread)) * w *
                   trimmedLogistic(wrapAngle(phi - PhiS(p, b.gammaO, gammaT)), ss, -kPi, kPi);
        }
    }
    return isfinite(sum) ? fmax(0.0, sum) : 0.0;
}

// Importance-sample an outgoing direction; pdfOut/fOut are filled to match, so a
// caller's weight is fOut * |cos theta_i| / pdfOut (see hair.h::sample).
__device__ __noinline__ static V3 sample(const Bcsdf& b, const V3& wo, double u0, double u1, double u2,
                            double u3, double& pdfOut, double& fOut) {   // __noinline__: see f above
    const double sinThetaO = clampd(wo.x, -1.0, 1.0), cosThetaO = safeSqrt(1.0 - sqr(sinThetaO));
    const double phiO = atan2(wo.z, wo.y);

    const Chord ch = refractGeom(b, sinThetaO, cosThetaO);
    const double gammaT = ch.gammaT;
    double ap[kNLobes];
    apPdf(b, sinThetaO, cosThetaO, ap);

    const int nLobes = b.hasMedulla ? kNLobes : (kPMax + 1);
    int lobe = 0;
    { double c = 0.0;
      for (; lobe < nLobes - 1; ++lobe) { c += ap[lobe]; if (u0 < c) break; } }

    const bool  scattered = (lobe >= kSTT);
    const int   p  = scattered ? (lobe - kSTT + 1) : lobe;
    const double spread = scattered ? scatteredSpread(ch) : 0.0;
    const double vUse = scattered ? scatteredV(b, p, spread) : b.v[p];

    double sinThetaOp, cosThetaOp;
    tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);

    // Longitudinal: exact inversion of M_p (Ou & Pellacini). The 1e-5 floor prevents a
    // log(0) NaN direction.
    u1 = fmax(u1, 1e-5);
    const double cosTheta =
        1.0 + vUse * log(u1 + (1.0 - u1) * exp(-2.0 / vUse));
    const double sinTheta = safeSqrt(1.0 - sqr(cosTheta));
    const double cosPhi   = cos(2.0 * kPi * u2);
    const double sinThetaI = clampd(-cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp,
                                    -1.0, 1.0);
    const double cosThetaI = safeSqrt(1.0 - sqr(sinThetaI));

    // Azimuthal: specular exit + trimmed-logistic offset; uniform for the residual.
    double dphi;
    if (scattered)
        dphi = PhiS(p, b.gammaO, gammaT) +
               sampleTrimmedLogistic(u3, scatteredS(b, spread), -kPi, kPi);
    else if (p < kPMax)
        dphi = Phi(p, b.gammaO, gammaT) + sampleTrimmedLogistic(u3, b.s, -kPi, kPi);
    else
        dphi = 2.0 * kPi * u3;
    const double phiI = phiO + dphi;

    const V3 wi{sinThetaI, cosThetaI * cos(phiI), cosThetaI * sin(phiI)};
    pdfOut = pdf(b, wo, wi);
    fOut   = f(b, wo, wi);
    return wi;
}

// Absorption from a target multiply-scattered reflectance (Chiang eq. 9).
__device__ static double sigmaAFromReflectance(double c, double betaN) {
    c = clampd(c, 1e-4, 1.0 - 1e-6);
    const double bn = clampd(betaN, 1e-4, 1.0);
    const double d = 5.969 - 0.215 * bn + 2.532 * sqr(bn) - 10.73 * pow(bn, 3.0) +
                     5.574 * pow(bn, 4.0) + 0.245 * pow(bn, 5.0);
    return sqr(log(c) / d);
}

// --- Hooking the model to a hit (device twins of hair.h's frame helpers) -------
// Impact parameter h = sin(gamma_o), signed about the fiber axis, recovered from the
// outward normal + tangent the curve intersector wrote. Guard threshold is 1e-7 (not
// the CPU's 1e-12): the hit data is Real precision, so below ~1e-7 the perpendicular
// components are float noise and "exactly end-on" (h = 0) is the honest answer.
__device__ static double hFromHit(const DVec3& n, const DVec3& tangent, const DVec3& wo) {
    const DVec3 ax = normalize(tangent);
    DVec3 nPerp = n  - ax * dot(ax, n);
    DVec3 oPerp = wo - ax * dot(ax, wo);
    const Real nl = length(nPerp), ol = length(oPerp);
    if (nl < (Real)1e-7 || ol < (Real)1e-7) return 0.0;   // end-on: axis degenerate
    nPerp = nPerp * ((Real)1 / nl);
    oPerp = oPerp * ((Real)1 / ol);
    const double cosGamma = clampd((double)dot(nPerp, oPerp), -1.0, 1.0);
    const double sign = ((double)dot(cross(nPerp, oPerp), ax) < 0.0) ? -1.0 : 1.0;
    return sign * safeSqrt(1.0 - sqr(cosGamma));
}

// World <-> local fiber frame; x = tangent, y built from the surface normal so the
// frame is continuous around the fiber.
struct Frame { DVec3 x, y, z; };

__device__ static Frame frameFromHit(const DVec3& n, const DVec3& tangent) {
    Frame fr;
    fr.x = normalize(tangent);
    DVec3 yy = n - fr.x * dot(fr.x, n);
    const Real yl = length(yy);
    if (yl > (Real)1e-7) {
        fr.y = yy * ((Real)1 / yl);
    } else {                                  // degenerate: any perpendicular will do
        DVec3 t, bt; onb(fr.x, t, bt); fr.y = t;
    }
    fr.z = cross(fr.x, fr.y);
    return fr;
}

__device__ static inline V3 toLocal(const Frame& fr, const DVec3& w) {
    return V3{(double)dot(w, fr.x), (double)dot(w, fr.y), (double)dot(w, fr.z)};
}
__device__ static inline DVec3 toWorld(const Frame& fr, const V3& w) {
    return fr.x * (Real)w.x + fr.y * (Real)w.y + fr.z * (Real)w.z;
}

}  // namespace dhair

// Everything needed to evaluate or sample the fiber BCSDF at one hit, for one
// wavelength — device twin of hair_shade.h's HairShade. No `aggregate` flag: the
// fur-volume tier is CPU-only, so a device fiber is always a real BVH strand.
struct DHairShade {
    dhair::Bcsdf b;
    dhair::Frame fr;
    dhair::V3    woLocal;   // the direction the path ARRIVED from, in the fiber frame
    Real         radius;    // world fiber radius at the hit (0 = not on a strand)
};

// Forward declaration: dHairShadeAt's from-reflectance path reads the material's diffuse
// reflectance (textures/patterns/records included); dDiffuseRho is defined further down
// with the texture helpers.
__device__ static Real dDiffuseRho(const DScene& sc, const DMaterial& m, const DHit& h, Real lambda);

// Build the BCSDF at a hit — device twin of hairShadeAt. `wPrev` points away from the
// surface, back along the path that got here (toward the light for the forward tracer,
// toward the eye for the backward one); it is the reference direction h is measured from.
__device__ __noinline__ static DHairShade dHairShadeAt(const DScene& sc, const DMaterial& m,
                                          const DHit& h,   // __noinline__: see dhair::f
                                          Real lambda, const DVec3& wPrev) {
    dhair::Params pr;
    pr.eta   = m.hairEta;
    pr.betaM = m.hairBetaM;
    pr.betaN = m.hairBetaN;
    pr.alpha = m.hairAlpha;
    pr.kappa = m.hairKappa;
    pr.mG    = m.hairMedullaG;
    pr.mSigmaS = fmax(0.0, (double)specLookup(m.hairMedullaSigmaS, lambda));
    pr.mSigmaA = fmax(0.0, (double)specLookup(m.hairMedullaSigmaA, lambda));

    double sigmaA;
    if (m.hairSigmaAFromReflect) {
        // Chiang eq. 9 per-wavelength: invert the authored reflectance (via dDiffuseRho,
        // so textures/patterns/records tint the fiber) into the absorption producing it.
        double c = (double)dDiffuseRho(sc, m, h, lambda);
        c = c < 0.0 ? 0.0 : (c > 1.0 ? 1.0 : c);
        sigmaA = dhair::sigmaAFromReflectance(c, m.hairBetaN);
    } else {
        sigmaA = fmax(0.0, (double)specLookup(m.hairSigmaA, lambda));
    }

    DHairShade s;
    // h.n is oriented against the arriving ray, so it is the normal on the side the path
    // came from — the side h and gamma_o are defined from.
    s.fr      = dhair::frameFromHit(h.n, h.tangent);
    s.woLocal = dhair::toLocal(s.fr, wPrev);
    s.b       = dhair::make(pr, dhair::hFromHit(h.n, h.tangent, wPrev), sigmaA);
    s.radius  = h.fiberRadius;
    return s;
}

// The complete "BCSDF times projection" factor toward a world direction: for a fiber the
// projection is the LONGITUDINAL cosine sqrt(1 - w_x^2), not dot(n, w) — a round strand's
// projected width is the same from every azimuth (see hair_shade.h's header comment).
__device__ static inline Real dHairFCos(const DHairShade& s, const DVec3& wWorld) {
    const dhair::V3 wl = dhair::toLocal(s.fr, wWorld);
    const double cosLong = dhair::safeSqrt(1.0 - dhair::sqr(dhair::clampd(wl.x, -1.0, 1.0)));
    return (Real)(dhair::f(s.b, s.woLocal, wl) * cosLong);
}

// How far a connection or scattered ray must step to clear the strand's own body: TT/TRT
// exit the FAR side of a solid tube, so leaving against the arrival-side normal steps
// 2.5 radii (or the ordinary RAY_EPS when that is larger — float-safe floor).
__device__ static inline Real dHairExitOffset(const DHairShade& s, const DVec3& n, const DVec3& w) {
    if (dot(n, w) >= (Real)0) return RAY_EPS;        // leaving on the arrival side
    return fmax(RAY_EPS, (Real)2.5 * s.radius);
}

// Fiber (D_HAIR) twin of connect(): mode-B pinhole splat from a hair vertex. Mirrors the
// CPU connectGeom/connect `hs` branches (render.h): the projection is folded into
// dHairFCos (cosSurf, the Veach corr and the terminator stG are all exactly 1 — the fiber
// does not use its normal as a projection axis); TT transmits, so a camera BEHIND the
// shading normal is a legitimate, often bright connection (no horizon reject); and the
// shadow ray starts past the strand's own body (dHairExitOffset), or the tube would
// occlude its own transmitted lobe.
__device__ static void connectHair(const DScene& sc, const DCamera& cam, double* film, double* hits,
                                   const DVec3& p, const DVec3& n, const DHairShade& hs,
                                   Real lambda, Real beta, DRng& rng) {
    DVec3 toCam = cam.eye - p;
    Real dist = length(toCam);
    DVec3 wdir = toCam / dist;
    int px, py; Real cosCam, dist2;
    if (!cam.project(p, px, py, cosCam, dist2)) return;
    const Real off = dHairExitOffset(hs, n, wdir);
    if (off >= dist) return;
    if (occluded(sc, p + wdir * off, wdir, dist - off - RAY_EPS)) return;
    Real f = dHairFCos(hs, wdir);
    double solidAngle = cam.pixelSolidAngle(cosCam);
    Real contrib = beta * f / (Real)((double)dist2 * solidAngle);
    if (sc.mediaN > 0) contrib *= dMediaTransmittance(sc, p, wdir, dist, lambda, rng);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}
// Fiber (D_HAIR) twin of connectLens(): model-A finite-lens next-event splat from a hair
// vertex. fcos = PI * dHairFCos — the BCSDF's 1/pi convention meets the pupil pdf's
// pi R^2 (see CPU connectLens); corr and stG are 1, and there is no horizon reject.
__device__ static void connectLensHair(const DScene& sc, const DCamera& cam, double* film, double* hits,
                                       const DVec3& p, const DVec3& n, const DHairShade& hs,
                                       Real lambda, Real beta, DRng& rng) {
    Real R  = (Real)cam.apertureR;
    Real rr = R * sqrt(rng.uniform());
    Real a  = (Real)(2.0 * DPI) * rng.uniform();
    DVec3 A = cam.eye + cam.u * (rr * cos(a)) + cam.v * (rr * sin(a));
    DVec3 toA = A - p;
    Real dist = length(toA);
    if (dist < (Real)1e-9) return;
    DVec3 wdir = toA / dist;
    Real cosLens = -dot(wdir, cam.w);                // cosine at the lens (w faces scene)
    if (cosLens <= (Real)1e-6) return;               // not heading toward the film
    int px, py;
    if (!cam.lensImage(A, wdir, px, py)) return;
    const Real off = dHairExitOffset(hs, n, wdir);
    if (off >= dist) return;
    if (occluded(sc, p + wdir * off, wdir, dist - off - RAY_EPS)) return;
    Real fcos = (Real)DPI * dHairFCos(hs, wdir);
    Real contrib = beta * fcos * cosLens * (R * R) / (dist * dist);
    // Same flux -> film-irradiance normaliser as connectLens (see there).
    contrib *= (Real)1 / (Real)(cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
    if (sc.mediaN > 0) contrib *= dMediaTransmittance(sc, p, wdir, dist, lambda, rng);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}

// A set of cameras sharing ONE photon trace (the multi-camera forward pass). The
// forward tracer only ever SPLATS to a camera (project/connect); it never generates
// camera rays, so a single photon path can deposit into every camera at once — the
// "many cameras for the price of one photon set" win, the device twin of the CPU
// renderForwardShared. films[c] / hits[c] are camera c's own device buffers (each
// camera keeps its resolution/projection/exposure). nCam==1 is the ordinary single-
// camera render. Model B's connect() draws no RNG in a homogeneous medium, so a multi-
// camera model-B pass is bit-identical to per-camera renders there; model A's
// connectLens() samples each pupil (it draws RNG), so its per-camera images match a
// standalone render in distribution only. (A HETEROGENEOUS medium adds a ratio-tracking
// transmittance draw per connect, so multi-cam model-B also matches only in distribution
// then — inherent to the estimator, and consistent with the CPU tracer.)
// One deposited photon (device twin of Photon + PhotonMap::pos in photonmap.h): where
// light landed, the shading normal (for cross-surface leak rejection), and the
// monochromatic power / wavelength it carried. Laid out to round-trip through the host
// PhotonMap (float here <-> double on the host build).
//
// There is NO incident direction: nothing reads one. The density estimate is Lambertian,
// so neither the device gather (kSppmGatherConvert -> DGatherPhoton) nor the host gather
// ever looked at it — it was pure freight. This buffer is capacity-limited (depCap is
// sized from FREE VRAM), so every byte per record is photons the GPU can't hold: dropping
// it shrinks the record from 44 to 32 bytes, ~27% more photons in the same VRAM. Don't add
// a field here without a reader.
struct DPhoton {
    DVec3 pos, n;
    float power, lambda;
};

struct DCamSet {
    const DCamera* cams;      // nCam cameras
    double* const* films;     // nCam film buffers  (XYZ*3 doubles each)
    double* const* hits;      // nCam per-pixel hit-count buffers
    int nCam;
    // Photon-map deposit (mode M forward pass). When depCount != nullptr the forward
    // tracer appends a photon record at every diffuse / diffuse-transmit vertex (device
    // twin of Renderer::depositPhoton) instead of / in addition to splatting: the pass
    // runs with nCam == 0 (all camera splats become no-ops) so the trace is pure deposit.
    // depCount is an atomic write cursor; depPhotons may be null (count-only sizing pass),
    // else records with index < depCap are stored (excess is counted but dropped).
    DPhoton*            depPhotons = nullptr;
    unsigned long long* depCount   = nullptr;
    unsigned long long  depCap     = 0;
    // PHOTON-BEAMS gather (CLI -beams, shared multi-camera pass). When set (and nCam>1,
    // a medium exists, camMode!=C) the photon crosses each medium in a STRAIGHT beam and
    // every camera independently resamples its own single-scatter in-scatter point with a
    // per-photon RNG stream, decoupling the deposit (shared flight) from the gather so a
    // volumetric flyby gets independent per-frame noise instead of one frozen speckle.
    bool beamGather = false;
};

// Gather-tuned photon record: what the mode-M density-estimate kernel actually reads.
// The deposit-side DPhoton carries (pos, n, power, lambda); the gather used to weight
// every visited photon by cie{X,Y,Z}(lambda_p) * power *
// norm / pi — all per-photon CONSTANTS of the estimate (norm = 1/(pi r^2 nEmitted)).
// That triple is folded into pX/pY/pZ at upload time (host doubles from PhotonMap::cie
// times power*norm/pi, rounded to float once), so the per-photon inner loop is just
// g? += rho(lambda_p) * p? — no CIE multi-Gaussian evaluation (7 exp()s) per visited
// photon. Device analogue of the CPU PhotonMap::cie precompute (photonmap.h), which
// was ~74% of profiled mode-M render time when it was added there.
struct DGatherPhoton {
    DVec3 pos;              // deposit position (world)
    DVec3 n;                // shading normal (cross-surface leak rejection)
    float pX, pY, pZ;       // cie{X,Y,Z}(lambda) * power * norm / pi (see above)
    float lambda;           // wavelength (nm) — rho(lambda_p) still varies per photon
};

// A view-independent photon-map query structure on the device (device twin of PhotonMap
// in photonmap.h): a uniform hash grid (cell size == gather radius) over cell-contiguous
// photon records, so a radius-r query touches only the 3x3x3 neighbourhood. Built on the
// host (PhotonMap::build) from the deposited photons, then uploaded for the gather kernel.
// (No nEmitted here: the normalization is folded into each record's pX/pY/pZ.)
struct DPhotonMap {
    const DGatherPhoton* photons; // reordered into cell-contiguous runs
    const int*     cellStart; // size nCells+1; cell c occupies [cellStart[c], cellStart[c+1])
    DVec3  lo;                // grid origin (world)
    Real   cellSize;          // == gather radius
    Real   radius;            // gather radius (world units)
    int    nx, ny, nz;
};

// Splat a surface vertex to every camera (model B pinhole connect, or model A finite-
// lens next-event splat). Device twin of Renderer::camSplatAll. Model C never shares
// (it consumes the photon per camera), so this is a no-op for CAM_C. For model A each
// camera draws its own aperture sample, so the loop consumes RNG proportional to nCam
// (deterministic given the camera order).
__device__ static void splatSurfaceAll(const DScene& sc, const DCamSet& cs, int camMode,
                                        const DVec3& p, const DVec3& n, const DVec3& ng,
                                        const DVec3& wi, Real lambda,
                                        Real beta, Real rho, DRng& rng) {
    for (int c = 0; c < cs.nCam; ++c) {
        if (camMode == CAM_B) connect(sc, cs.cams[c], cs.films[c], cs.hits[c], p, n, ng, wi, lambda, beta, rho, rng);
        else if (camMode == CAM_A) connectLens(sc, cs.cams[c], cs.films[c], cs.hits[c], p, n, ng, wi, lambda, beta, rho, rng);
    }
}
// Fiber (D_HAIR) analogue of splatSurfaceAll — device twin of Renderer::camSplatAllHair.
// The BCSDF and its projection arrive together in `hs`, so there is no rho/wi/ng: the
// fiber takes neither the Lambertian rho nor the Veach shading-normal correction.
__device__ static void splatSurfaceAllHair(const DScene& sc, const DCamSet& cs, int camMode,
                                           const DVec3& p, const DVec3& n, const DHairShade& hs,
                                           Real lambda, Real beta, DRng& rng) {
    for (int c = 0; c < cs.nCam; ++c) {
        if (camMode == CAM_B) connectHair(sc, cs.cams[c], cs.films[c], cs.hits[c], p, n, hs, lambda, beta, rng);
        else if (camMode == CAM_A) connectLensHair(sc, cs.cams[c], cs.films[c], cs.hits[c], p, n, hs, lambda, beta, rng);
    }
}
// Append a photon record at a diffuse / translucent vertex (device twin of
// Renderer::depositPhoton). No-op unless the camera set carries a deposit buffer
// (normal renders leave cs.depCount == nullptr, so this compiles away to nothing).
// Always increments the atomic count (so a null-buffer sizing pass measures the exact
// deposit total); stores only when a buffer is bound and the slot is within capacity.
// The photon's travel/incident direction is deliberately not a parameter: no gather reads
// it (see DPhoton), so it isn't stored (matches Renderer::depositPhoton on the host).
__device__ static void depositPhoton(const DCamSet& cs, const DVec3& p,
                                     const DVec3& n, Real beta, Real lambda) {
    if (!cs.depCount) return;
    unsigned long long i = atomicAdd(cs.depCount, 1ULL);
    if (cs.depPhotons && i < cs.depCap) {
        DPhoton ph;
        ph.pos = p; ph.n = n;
        ph.power = (float)beta; ph.lambda = (float)lambda;
        cs.depPhotons[i] = ph;
    }
}
// Volume (fog) analogue of splatSurfaceAll.
__device__ static void splatVolumeAll(const DScene& sc, const DMedium& med, const DCamSet& cs,
                                       int camMode, const DVec3& p, const DVec3& wIn, Real lambda,
                                       Real beta, DRng& rng) {
    for (int c = 0; c < cs.nCam; ++c) {
        if (camMode == CAM_B) connectVolume(sc, med, cs.cams[c], cs.films[c], cs.hits[c], p, wIn, lambda, beta, rng);
        else if (camMode == CAM_A) connectLensVolume(sc, med, cs.cams[c], cs.films[c], cs.hits[c], p, wIn, lambda, beta, rng);
    }
}

// Isotropic volumetric-EMISSION splat (fire) — device twin of Renderer::connectEmission-
// Volume (render.h). Like connectVolume but the albedo*phase is replaced by the isotropic
// 1/(4pi): a fire voxel radiates equally in all directions, so the direct term is
// beta*(1/4pi)/(dist^2*pixelSolidAngle)*transmittance. No incoming direction is needed.
__device__ static void connectEmissionVolume(const DScene& sc, const DCamera& cam,
                                             double* film, double* hits,
                                             const DVec3& p, Real lambda, Real beta, DRng& rng) {
    DVec3 toCam = cam.eye - p;
    Real dist = length(toCam);
    DVec3 wdir = toCam / dist;
    int px, py; Real cosCam, dist2;
    if (!cam.project(p, px, py, cosCam, dist2)) return;
    if (occluded(sc, p + wdir * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    double solidAngle = cam.pixelSolidAngle(cosCam);
    Real contrib = beta * (Real)(1.0 / (4.0 * DPI)) / (Real)((double)dist2 * solidAngle);
    contrib *= dMediaTransmittance(sc, p, wdir, dist, lambda, rng);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}
// Model A (finite-lens) isotropic emission splat — device twin of connectEmissionLensVolume.
// As connectLensVolume but albedo*phase -> 1/(4pi).
__device__ static void connectEmissionLensVolume(const DScene& sc, const DCamera& cam,
                                                 double* film, double* hits,
                                                 const DVec3& p, Real lambda, Real beta, DRng& rng) {
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
    Real contrib = beta * (Real)(1.0 / (4.0 * DPI)) * cosLens * (Real)DPI * (R * R) / (dist * dist);
    contrib *= (Real)1 / (Real)(cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
    contrib *= dMediaTransmittance(sc, p, wdir, dist, lambda, rng);
    filmAdd(film, hits, cam.resX, px, py, lambda, contrib);
}
__device__ static void camSplatEmissionAll(const DScene& sc, const DCamSet& cs, int camMode,
                                            const DVec3& p, Real lambda, Real beta, DRng& rng) {
    for (int c = 0; c < cs.nCam; ++c) {
        if (camMode == CAM_B) connectEmissionVolume(sc, cs.cams[c], cs.films[c], cs.hits[c], p, lambda, beta, rng);
        else if (camMode == CAM_A) connectEmissionLensVolume(sc, cs.cams[c], cs.films[c], cs.hits[c], p, lambda, beta, rng);
    }
}

// ==================== analytic specular sphere connection ====================
// Device twin of Renderer::connectSpecularSphere / connectSpecularSphereInside
// (render.h). Restores the paths the SDS limitation makes black: mode B can
// directly image a smooth glass sphere (and fly the camera THROUGH one). All the
// precision-critical math (planar root solve, ray-differential Jacobian) runs in
// DOUBLE regardless of the render Real, so the fp32 GPU build stays robust; only
// the project()/occluded()/transmittance boundary casts back to Real.

struct D3 {
    double x, y, z;
    __device__ D3() : x(0), y(0), z(0) {}
    __device__ D3(double a, double b, double c) : x(a), y(b), z(c) {}
    __device__ D3(const DVec3& v) : x((double)v.x), y((double)v.y), z((double)v.z) {}
    __device__ D3 operator+(const D3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __device__ D3 operator-(const D3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __device__ D3 operator*(double s)   const { return {x * s, y * s, z * s}; }
    __device__ DVec3 toR() const { return DVec3(x, y, z); }
};
__device__ static inline double d3dot(const D3& a, const D3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ static inline double d3len(const D3& a) { return sqrt(d3dot(a, a)); }
__device__ static inline D3 d3norm(const D3& a) { double l = 1.0 / d3len(a); return {a.x*l, a.y*l, a.z*l}; }
__device__ static inline void d3onb(const D3& n, D3& t, D3& b) {
    double sign = copysign(1.0, n.z);
    double a = -1.0 / (sign + n.z);
    double d = n.x * n.y * a;
    t = D3(1 + sign * n.x * n.x * a, sign * d, -sign * n.x);
    b = D3(d, sign + n.y * n.y * a, -n.y);
}

// Describes the photon vertex being connected through the glass: a diffuse
// surface (Lambertian weight=rho, normal np) or a volume in-scatter (weight=albedo,
// HG phase g, incoming dir wIn). Device twin of Renderer::SpecVtx.
struct DSpecVtx {
    bool   volume;
    D3     np;                 // surface normal (surface vertices)
    D3     wIn;                // incoming photon direction (volume vertices)
    double g;                  // HG asymmetry (volume vertices)
    double weight;             // surface: Lambertian rho ; volume: single-scatter albedo
    const DMedium* med;        // scattering medium (volume vertices) — for rainbow phase
    Real   lambda;             // wavelength (volume vertices) — for rainbow phase
    // Throughput at the vertex for a connection leaving toward `wP` (unit, toward the
    // sphere). Returns <0 to signal "reject" (camera-side behind a surface). A rainbow
    // medium uses its tabulated Airy phase; HG media use the analytic lobe.
    __device__ double term(const D3& wP) const {
        if (volume) return weight * (double)dMedPhase(*med, (Real)d3dot(wIn, wP), lambda);
        double cosSurf = d3dot(np, wP);
        return cosSurf <= 0.0 ? -1.0 : (weight / DPI) * cosSurf;
    }
    // The WAVELENGTH-INDEPENDENT half of term(): everything but `weight` (the per-λ
    // albedo). The mirror connectors solve ONE shared geometry for a whole hero bundle —
    // a flat/metallic reflection does not disperse — so they evaluate this once and
    // multiply by each λ's weight. term() keeps its own arithmetic rather than being
    // expressed through this, so the dielectric-sphere path's float ordering (and
    // therefore its images) stay bit-identical to before this existed. Device twin of
    // Renderer::SpecVtx::shape. (A volume vertex only ever reaches the mirrors through
    // the scalar wrapper, where vt.lambda IS the single live λ, so a rainbow medium's
    // λ-dependent phase is still evaluated at the right wavelength.)
    __device__ double shape(const D3& wP) const {
        if (volume) return (double)dMedPhase(*med, (Real)d3dot(wIn, wP), lambda);
        double cosSurf = d3dot(np, wP);
        return cosSurf <= 0.0 ? -1.0 : cosSurf / DPI;
    }
};

struct DSphereRefr { D3 P1, P2, exitDir; double Tf, innerLen; };

// Trace a ray from `o` (outside sphere S) that ENTERS S, crosses the glass, and
// EXITS. False on miss / TIR. (Port of traceThroughSphere2.)
__device__ static bool dTraceThroughSphere(const D3& o, const D3& d, const DSphere& S,
                                           double n, DSphereRefr& out) {
    D3 O(S.c); double r = S.r;
    D3 oc = o - O;
    double b = d3dot(oc, d), c = d3dot(oc, oc) - r * r;
    double disc = b * b - c;
    if (disc < 0.0) return false;
    double sq = sqrt(disc);
    double t1 = -b - sq;
    if (t1 < 1e-7) return false;
    D3 P1 = o + d * t1;
    D3 N1 = (P1 - O) * (1.0 / r);
    double cosI = -d3dot(d, N1);
    if (cosI <= 1e-6) return false;
    double eta = 1.0 / n;
    double sin2t = eta * eta * (1.0 - cosI * cosI);
    if (sin2t >= 1.0) return false;
    double cosT = sqrt(1.0 - sin2t);
    D3 tin = d3norm(d * eta + N1 * (eta * cosI - cosT));
    double rs = (cosI - n * cosT) / (cosI + n * cosT);
    double rp = (cosT - n * cosI) / (cosT + n * cosI);
    double Fe = 0.5 * (rs * rs + rp * rp);
    double sInner = -2.0 * d3dot(P1 - O, tin);
    if (sInner <= 1e-9) return false;
    D3 P2 = P1 + tin * sInner;
    D3 N2 = (P2 - O) * (1.0 / r);
    double cosI2 = d3dot(tin, N2);
    if (cosI2 <= 1e-6) return false;
    double sin2t2 = n * n * (1.0 - cosI2 * cosI2);
    if (sin2t2 >= 1.0) return false;
    double cosT2 = sqrt(1.0 - sin2t2);
    D3 exitDir = d3norm(tin * n - N2 * (n * cosI2 - cosT2));
    double rs2 = (n * cosI2 - cosT2) / (n * cosI2 + cosT2);
    double rp2 = (n * cosT2 - cosI2) / (n * cosT2 + cosI2);
    double Fx = 0.5 * (rs2 * rs2 + rp2 * rp2);
    out.P1 = P1; out.P2 = P2; out.exitDir = exitDir;
    out.Tf = (1.0 - Fe) * (1.0 - Fx); out.innerLen = sInner;
    return true;
}

struct DSphereRefr1 { D3 P1, exitDir; double Tf, innerLen; };

// Trace a ray from `o` INSIDE sphere S to its exit, refracting glass->vacuum.
// False on TIR / degenerate. (Port of traceOutOfSphere.)
__device__ static bool dTraceOutOfSphere(const D3& o, const D3& d, const DSphere& S,
                                         double n, DSphereRefr1& out) {
    D3 O(S.c); double r = S.r;
    D3 oc = o - O;
    double b = d3dot(oc, d), c = d3dot(oc, oc) - r * r;
    double disc = b * b - c;
    if (disc <= 0.0) return false;
    double sq = sqrt(disc);
    double t1 = -b + sq;
    if (t1 < 1e-7) return false;
    D3 P1 = o + d * t1;
    D3 N1 = (P1 - O) * (1.0 / r);
    double cosI = d3dot(d, N1);
    if (cosI <= 1e-6) return false;
    double sin2t = n * n * (1.0 - cosI * cosI);
    if (sin2t >= 1.0) return false;
    double cosT = sqrt(1.0 - sin2t);
    D3 exitDir = d3norm(d * n - N1 * (n * cosI - cosT));
    double rs = (n * cosI - cosT) / (n * cosI + cosT);
    double rp = (n * cosT - cosI) / (n * cosT + cosI);
    double F = 0.5 * (rs * rs + rp * rp);
    out.P1 = P1; out.exitDir = exitDir; out.Tf = 1.0 - F; out.innerLen = t1;
    return true;
}

// --- specular-sphere scan-angle tables (Opt 3) -------------------------------
// Both sphere connectors scan the SAME SPH_SCAN_N+1 fixed entry angles on every
// call, and each scan step used to pay a software-fp64 cos+sin pair — the
// dominant transcendental cost of the glass-sphere splat on GPU. The angles
// never change, so a one-time init kernel fills these tables with the SAME
// device cos/sin the macros called (host-computed values could differ by 1 ulp),
// making table reads bit-identical to the per-step evaluation they replace.
// Bisection refinement still computes live cos/sin (midpoints are data-dependent).
static constexpr int SPH_SCAN_N = 96;
__device__ static double g_sphScanC[SPH_SCAN_N + 1];
__device__ static double g_sphScanS[SPH_SCAN_N + 1];

__global__ void kSphScanInit() {
    int i = threadIdx.x;
    if (i > SPH_SCAN_N) return;
    const int NS = SPH_SCAN_N;
    double phi = -DPI + (2.0 * DPI) * i / NS;
    g_sphScanC[i] = cos(phi);
    g_sphScanS[i] = sin(phi);
}

// Connect EXTERIOR vertex p to a pinhole INSIDE dielectric sphere S (single
// refraction) — the path the camera sees flying THROUGH the glass. Port of
// Renderer::connectSpecularSphereInside.
__device__ static void dConnectSpecularSphereInside(const DScene& sc, const DCamera& cam,
        double* film, double* hits, const DSphere& S, const DMaterial& glass, double n,
        const D3& p, const DSpecVtx& vt, Real lambda, double beta, DRng& rng) {
    D3 O(S.c); double r = S.r; D3 eye(cam.eye);
    double dEyeO = d3len(eye - O);

    D3 ex, ey;
    if (dEyeO < 1e-9) { D3 tb; d3onb(d3norm(p - O), ex, tb); }
    else              ex = (eye - O) * (1.0 / dEyeO);
    D3 ap = p - O;
    D3 perp = ap - ex * d3dot(ap, ex);
    double perpLen = d3len(perp);
    if (perpLen < 1e-9) { D3 tb; d3onb(ex, ey, tb); }
    else                ey = perp * (1.0 / perpLen);
    double ex_e = d3dot(eye - O, ex), ey_e = d3dot(eye - O, ey);
    double px2 = d3dot(ap, ex), py2 = d3dot(ap, ey);

    // In-plane once-refracted exit-ray miss of p. Encoded as a helper via lambda-free
    // repeated code (no std::function on device): returns miss, sets valid.
    // Takes the entry angle as its (cos, sin) pair.
    #define D_TRACE2D_INSIDE(C1, S1, MISS, VALID) do {                                \
        VALID = false; MISS = 0.0;                                                    \
        double c1 = (C1), s1 = (S1);                                                  \
        double P1x = r * c1, P1y = r * s1;                                            \
        double dinx = P1x - ex_e, diny = P1y - ey_e;                                  \
        double dl = sqrt(dinx*dinx + diny*diny);                                      \
        if (dl >= 1e-12) {                                                            \
            dinx /= dl; diny /= dl;                                                   \
            double cosI = dinx*c1 + diny*s1;                                          \
            if (cosI > 1e-6) {                                                        \
                double sin2t = n*n*(1.0 - cosI*cosI);                                 \
                if (sin2t < 1.0) {                                                    \
                    double cosT = sqrt(1.0 - sin2t);                                  \
                    double kk = n*cosI - cosT;                                        \
                    double doutx = n*dinx - kk*c1, douty = n*diny - kk*s1;            \
                    double dl2 = sqrt(doutx*doutx + douty*douty);                     \
                    doutx /= dl2; douty /= dl2;                                       \
                    double fw = (px2-P1x)*doutx + (py2-P1y)*douty;                    \
                    if (fw > 0.0) {                                                   \
                        VALID = true;                                                 \
                        MISS = doutx*(py2-P1y) - douty*(px2-P1x);                     \
                    }                                                                 \
                }                                                                     \
            }                                                                         \
        }                                                                            \
    } while (0)

    const int NS = SPH_SCAN_N; double roots[4]; int nroot = 0;
    double prevMiss = 0.0, prevPhi = 0.0; bool prevValid = false;
    for (int i = 0; i <= NS && nroot < 4; ++i) {
        double phi = -DPI + (2.0 * DPI) * i / NS;
        bool v; double mss; D_TRACE2D_INSIDE(g_sphScanC[i], g_sphScanS[i], mss, v);
        if (v && prevValid && ((mss < 0.0) != (prevMiss < 0.0))) {
            double a = prevPhi, b = phi, fa = prevMiss;
            for (int k = 0; k < 40; ++k) {
                double mid = 0.5 * (a + b); bool vm; double fm; D_TRACE2D_INSIDE(cos(mid), sin(mid), fm, vm);
                if (!vm) break;
                if ((fm < 0.0) != (fa < 0.0)) b = mid; else { a = mid; fa = fm; }
            }
            roots[nroot++] = 0.5 * (a + b);
        }
        prevMiss = mss; prevValid = v; prevPhi = phi;
    }
    #undef D_TRACE2D_INSIDE

    for (int ri = 0; ri < nroot; ++ri) {
        double phi = roots[ri];
        D3 P1chief = O + ex * (r * cos(phi)) + ey * (r * sin(phi));
        D3 d0 = d3norm(P1chief - eye);
        DSphereRefr1 ch;
        if (!dTraceOutOfSphere(eye, d0, S, n, ch)) continue;

        D3 a1, a2; d3onb(d0, a1, a2);
        const double eps = 2e-4;
        DSphereRefr1 rA, rB;
        if (!dTraceOutOfSphere(eye, d3norm(d0 + a1 * eps), S, n, rA)) continue;
        if (!dTraceOutOfSphere(eye, d3norm(d0 + a2 * eps), S, n, rB)) continue;
        D3 e1, e2; d3onb(ch.exitDir, e1, e2);
        double ax, ay, bx, by;
        {   double denom = d3dot(rA.exitDir, ch.exitDir);
            if (fabs(denom) < 1e-9) denom = (denom < 0 ? -1e-9 : 1e-9);
            double s = d3dot(p - rA.P1, ch.exitDir) / denom;
            D3 off = (rA.P1 + rA.exitDir * s) - p;
            ax = d3dot(off, e1); ay = d3dot(off, e2); }
        {   double denom = d3dot(rB.exitDir, ch.exitDir);
            if (fabs(denom) < 1e-9) denom = (denom < 0 ? -1e-9 : 1e-9);
            double s = d3dot(p - rB.P1, ch.exitDir) / denom;
            D3 off = (rB.P1 + rB.exitDir * s) - p;
            bx = d3dot(off, e1); by = d3dot(off, e2); }
        double jac = fabs(ax * by - ay * bx);
        if (jac < 1e-24) continue;
        double G = (eps * eps) / jac;

        int px, py; Real cosCam, dist2e;
        if (!cam.project(P1chief.toR(), px, py, cosCam, dist2e)) continue;
        double omega = cam.pixelSolidAngle(cosCam);
        if (omega <= 0.0) continue;

        D3 wP = ch.P1 - p; double dP = d3len(wP);
        if (dP < 1e-9) continue;
        wP = wP * (1.0 / dP);
        double term = vt.term(wP);
        if (term < 0.0) continue;

        double contrib = beta * term * G * ch.Tf / omega;
        if (contrib <= 0.0) continue;
        double aGlass = (double)specLookup(glass.absorb, lambda);
        if (aGlass > 0.0) contrib *= exp(-aGlass * ch.innerLen);

        DVec3 wPR = wP.toR();
        if (occluded(sc, (p + wP * 1e-6).toR(), wPR, connMaxT(dP))) continue;
        if (sc.mediaN > 0)
            contrib *= (double)dMediaTransmittance(sc, p.toR(), wPR, (Real)dP, lambda, rng);

        filmAdd(film, hits, cam.resX, px, py, lambda, (Real)contrib);
    }
}

// Connect vertex p to a pinhole OUTSIDE dielectric sphere S, THROUGH the glass
// (two refractions). Port of Renderer::connectSpecularSphere. Dispatches to the
// single-refraction path when the eye is inside the glass.
__device__ static void dConnectSpecularSphere(const DScene& sc, const DCamera& cam,
        double* film, double* hits, const DSphere& S, const DMaterial& glass, double n,
        const D3& p, const DSpecVtx& vt, Real lambda, double beta, DRng& rng) {
    D3 O(S.c); double r = S.r; D3 eye(cam.eye);
    double dEyeO = d3len(eye - O);
    double dPO   = d3len(p - O);
    if (dPO   <  r * 0.9999) return;                 // vertex inside glass -> skip
    if (dEyeO <= r * 0.9999) {                        // eye inside -> single refraction
        dConnectSpecularSphereInside(sc, cam, film, hits, S, glass, n, p, vt, lambda, beta, rng);
        return;
    }
    if (dEyeO <= r * 1.0001) return;                  // eye ~on surface -> degenerate

    D3 ex = (eye - O) * (1.0 / dEyeO);
    D3 ap = p - O;
    D3 perp = ap - ex * d3dot(ap, ex);
    double perpLen = d3len(perp);
    D3 ey;
    if (perpLen < 1e-9) { D3 tb; d3onb(ex, ey, tb); }
    else                ey = perp * (1.0 / perpLen);
    double ex_e = dEyeO;
    double px2 = d3dot(ap, ex), py2 = d3dot(ap, ey);

    // Takes the entry angle as its (cos, sin) pair.
    #define D_TRACE2D_THRU(C1, S1, MISS, VALID) do {                                 \
        VALID = false; MISS = 0.0;                                                   \
        double c1 = (C1), s1 = (S1);                                                 \
        double P1x = r * c1, P1y = r * s1;                                           \
        double dinx = P1x - ex_e, diny = P1y;                                        \
        double dl = sqrt(dinx*dinx + diny*diny);                                     \
        if (dl >= 1e-12) {                                                           \
            dinx /= dl; diny /= dl;                                                  \
            double cosI = -(dinx*c1 + diny*s1);                                      \
            if (cosI > 1e-6) {                                                       \
                double eta = 1.0/n, sin2t = eta*eta*(1.0 - cosI*cosI);              \
                if (sin2t < 1.0) {                                                   \
                    double cosT = sqrt(1.0 - sin2t);                                 \
                    double tinx = eta*dinx + (eta*cosI - cosT)*c1;                   \
                    double tiny = eta*diny + (eta*cosI - cosT)*s1;                   \
                    double tl = sqrt(tinx*tinx + tiny*tiny); tinx/=tl; tiny/=tl;     \
                    double sInner = -2.0*(P1x*tinx + P1y*tiny);                      \
                    if (sInner > 1e-9) {                                             \
                        double P2x = P1x + tinx*sInner, P2y = P1y + tiny*sInner;     \
                        double n2x = P2x/r, n2y = P2y/r;                             \
                        double cosI2 = tinx*n2x + tiny*n2y;                          \
                        if (cosI2 > 1e-6) {                                          \
                            double sin2t2 = n*n*(1.0 - cosI2*cosI2);                 \
                            if (sin2t2 < 1.0) {                                      \
                                double cosT2 = sqrt(1.0 - sin2t2);                   \
                                double doutx = n*tinx - (n*cosI2 - cosT2)*n2x;       \
                                double douty = n*tiny - (n*cosI2 - cosT2)*n2y;       \
                                double dl2 = sqrt(doutx*doutx + douty*douty);        \
                                doutx/=dl2; douty/=dl2;                              \
                                double fw = (px2-P2x)*doutx + (py2-P2y)*douty;       \
                                if (fw > 0.0) {                                      \
                                    VALID = true;                                    \
                                    MISS = doutx*(py2-P2y) - douty*(px2-P2x);        \
                                }                                                    \
                            }                                                        \
                        }                                                            \
                    }                                                                \
                }                                                                    \
            }                                                                        \
        }                                                                           \
    } while (0)

    const int NS = SPH_SCAN_N; double roots[4]; int nroot = 0;
    double prevMiss = 0.0, prevPhi = 0.0; bool prevValid = false;
    for (int i = 0; i <= NS && nroot < 4; ++i) {
        double phi = -DPI + (2.0 * DPI) * i / NS;
        bool v; double mss; D_TRACE2D_THRU(g_sphScanC[i], g_sphScanS[i], mss, v);
        if (v && prevValid && ((mss < 0.0) != (prevMiss < 0.0))) {
            double a = prevPhi, b = phi, fa = prevMiss;
            for (int k = 0; k < 40; ++k) {
                double mid = 0.5 * (a + b); bool vm; double fm; D_TRACE2D_THRU(cos(mid), sin(mid), fm, vm);
                if (!vm) break;
                if ((fm < 0.0) != (fa < 0.0)) b = mid; else { a = mid; fa = fm; }
            }
            roots[nroot++] = 0.5 * (a + b);
        }
        prevMiss = mss; prevValid = v; prevPhi = phi;
    }
    #undef D_TRACE2D_THRU

    for (int ri = 0; ri < nroot; ++ri) {
        double phi = roots[ri];
        D3 P1chief = O + ex * (r * cos(phi)) + ey * (r * sin(phi));
        D3 d0 = d3norm(P1chief - eye);
        DSphereRefr ch;
        if (!dTraceThroughSphere(eye, d0, S, n, ch)) continue;

        D3 a1, a2; d3onb(d0, a1, a2);
        const double eps = 2e-4;
        DSphereRefr rA, rB;
        if (!dTraceThroughSphere(eye, d3norm(d0 + a1 * eps), S, n, rA)) continue;
        if (!dTraceThroughSphere(eye, d3norm(d0 + a2 * eps), S, n, rB)) continue;
        D3 e1, e2; d3onb(ch.exitDir, e1, e2);
        double ax, ay, bx, by;
        {   double denom = d3dot(rA.exitDir, ch.exitDir);
            if (fabs(denom) < 1e-9) denom = (denom < 0 ? -1e-9 : 1e-9);
            double s = d3dot(p - rA.P2, ch.exitDir) / denom;
            D3 off = (rA.P2 + rA.exitDir * s) - p;
            ax = d3dot(off, e1); ay = d3dot(off, e2); }
        {   double denom = d3dot(rB.exitDir, ch.exitDir);
            if (fabs(denom) < 1e-9) denom = (denom < 0 ? -1e-9 : 1e-9);
            double s = d3dot(p - rB.P2, ch.exitDir) / denom;
            D3 off = (rB.P2 + rB.exitDir * s) - p;
            bx = d3dot(off, e1); by = d3dot(off, e2); }
        double jac = fabs(ax * by - ay * bx);
        if (jac < 1e-24) continue;
        double G = (eps * eps) / jac;

        int px, py; Real cosCam, dist2e;
        if (!cam.project(P1chief.toR(), px, py, cosCam, dist2e)) continue;
        double omega = cam.pixelSolidAngle(cosCam);
        if (omega <= 0.0) continue;

        D3 wP = ch.P2 - p; double dP2 = d3len(wP);
        if (dP2 < 1e-9) continue;
        wP = wP * (1.0 / dP2);
        double term = vt.term(wP);
        if (term < 0.0) continue;

        double contrib = beta * term * G * ch.Tf / omega;
        if (contrib <= 0.0) continue;
        double aGlass = (double)specLookup(glass.absorb, lambda);
        if (aGlass > 0.0) contrib *= exp(-aGlass * ch.innerLen);

        DVec3 wPR = wP.toR();
        if (occluded(sc, (p + wP * 1e-6).toR(), wPR, connMaxT(dP2))) continue;
        D3 wE = eye - ch.P1; double dE = d3len(wE); wE = wE * (1.0 / dE);
        DVec3 wER = wE.toR();
        if (occluded(sc, (ch.P1 + wE * 1e-6).toR(), wER, connMaxT(dE))) continue;

        if (sc.mediaN > 0) {
            contrib *= (double)dMediaTransmittance(sc, p.toR(),   wPR, (Real)dP2, lambda, rng);
            contrib *= (double)dMediaTransmittance(sc, ch.P1.toR(), wER, (Real)dE, lambda, rng);
        }
        filmAdd(film, hits, cam.resX, px, py, lambda, (Real)contrib);
    }
}

// ================== analytic specular connection by REFLECTION ==================
// Device twin of the mirror connectors in render.h (see the long rationale there).
// A photon vertex that can only reach the pinhole by bouncing off a PERFECT MIRROR has
// no connection under the plain estimator — a delta reflection has ~zero pdf toward a
// point — so mode B used to draw every mirror surface pure BLACK while mode R drew it
// correctly. The fix is to solve the reflection analytically instead of sampling it:
//
//   PLANE: unfold. Mirror the eye across the plane (eye' = eye - 2*(dot(n,eye)-d)*n) and
//   the bent chain p -> R -> eye becomes the straight segment p -> eye'. The reflection
//   point R is where that segment crosses the plane (no root solve), and the specular
//   Jacobian is exactly G = 1/D^2 with D = |p - eye'| — connect()'s own inverse-square
//   law measured along the folded-out path.
//
//   SPHERE: no closed form, so it reuses the glass-sphere machinery — a 1-D root scan in
//   plane(eye, p, centre) and a ray-differential Jacobian, with ONE reflection in place
//   of the two refractions.
//
// Same limits as the host: one specular vertex per connection, pinhole only, and flat
// mirrors must be authored as world triangles (see Scene::buildMirrorPlanes).

// Defined further down (it needs the pattern/record evaluators); declared here so the
// mirror connectors can read a textured/driven mirror's reflectance at the exact hit.
__device__ static Real dReflectSlot(const DScene& sc, const DMaterial& m, const DHit& h, Real lambda);

// Materials whose reflection is a perfect delta the connection can solve for. Device twin
// of isPlanarMirrorMat (scene.h): halfmirror's reflect slot IS its specular reflectance in
// expectation, and glossy degenerates to a mirror only when nothing can roughen it.
__device__ static inline bool dIsPlanarMirrorMat(const DMaterial& m) {
    if (m.type == D_MIRROR || m.type == D_HALFMIRROR) return true;
    return m.type == D_GLOSSY && m.roughness <= 0.0 &&
           m.roughnessTex < 0 && m.roughnessPat < 0;
}

// Reflect a ray off sphere S at its first intersection ahead of `o`. Works from either
// side (a convex mirror ball seen from outside, a mirrored room seen from within), since
// the reflecting normal is taken against the ray. Twin of Renderer::reflectOffSphere.
struct DSphereRefl { D3 P1, exitDir; };
__device__ static bool dReflectOffSphere(const D3& o, const D3& d, const DSphere& S,
                                         DSphereRefl& out) {
    D3 O(S.c); double r = S.r;
    D3 oc = o - O;
    double b = d3dot(oc, d), c = d3dot(oc, oc) - r * r;
    double disc = b * b - c;
    if (disc <= 0.0) return false;
    double sq = sqrt(disc);
    double t = -b - sq;                              // near root (exterior origin)
    if (t < 1e-7) t = -b + sq;                       // interior origin: the far one
    if (t < 1e-7) return false;
    D3 P1 = o + d * t;
    D3 N = (P1 - O) * (1.0 / r);                     // outward normal
    double cn = d3dot(d, N);
    if (fabs(cn) <= 1e-6) return false;              // grazing: reflection is degenerate
    out.P1 = P1;
    out.exitDir = d3norm(d - N * (2.0 * cn));        // reflect(d, N); sign of N is moot
    return true;
}

// Is the surface the eye actually sees along the connection's eye-side leg the mirror we
// solved the geometry for? One traversal settles BOTH questions the connection needs —
// "is there really mirror material at that spot" (containment inside the authored panel,
// which the plane itself does not know) and "is the leg from the eye clear" — and hands
// back the DHit, so the reflectance is read at the exact texel/pattern value there.
// Twin of Renderer::mirrorSeenAt.
__device__ static bool dMirrorSeenAt(const DScene& sc, const D3& eye, const D3& wE,
                                     double dE, DHit& hm) {
    hm = closestHit(sc, eye.toR(), wE.toR());
    if (!hm.valid) return false;
    if (fabs((double)hm.t - dE) > 1e-4 * (1.0 + dE)) return false;   // something in front
    return dIsPlanarMirrorMat(sc.mats[hm.matId]);
}

// Splat the `nUp` live wavelengths of a photon vertex through one connection whose
// geometry is already solved: the light leaves p toward wP over dP to the mirror point R,
// the mirror hands it to the eye over dE, and the whole unfolded path is D long. Shared by
// the planar and spherical mirror connectors — below this point the only λ-dependence is
// the mirror's own reflectance and the fog. Twin of Renderer::splatMirrorLegs.
__device__ static void dSplatMirrorLegs(const DScene& sc, const DCamera& cam, double* film,
        double* hits, const DMaterial& mm, const DHit& hm, int px, int py, double omega,
        const D3& p, const D3& wP, double dP, const D3& R, const D3& wRE, double dE,
        double D, double shape, const Real* lam, const Real* beta, const Real* w,
        int nUp, DRng& rng) {
    const double invD2Omega = 1.0 / (D * D * omega);
    const DVec3 wPR = wP.toR(), wRER = wRE.toR();
    const DVec3 pR = p.toR(), RR = R.toR();
    for (int i = 0; i < nUp; ++i) {
        double refl = clamp01((double)dReflectSlot(sc, mm, hm, lam[i]));
        double contrib = (double)beta[i] * ((double)w[i] * shape) * refl * invD2Omega;
        if (contrib <= 0.0) continue;
        if (sc.mediaN > 0) {
            contrib *= (double)dMediaTransmittance(sc, pR, wPR,  (Real)dP, lam[i], rng);
            contrib *= (double)dMediaTransmittance(sc, RR, wRER, (Real)dE, lam[i], rng);
        }
        filmAdd(film, hits, cam.resX, px, py, lam[i], (Real)contrib);
    }
}

// Connect vertex p to the pinhole `cam` by reflection in one PLANAR mirror.
// Twin of Renderer::connectSpecularPlane.
__device__ static void dConnectSpecularPlane(const DScene& sc, const DCamera& cam,
        double* film, double* hits, const DMirrorPlane& mp, const D3& p, const DSpecVtx& vt,
        const Real* lam, const Real* beta, const Real* w, int nUp, DRng& rng) {
    const D3 eye(cam.eye);
    const D3 mn(mp.nx, mp.ny, mp.nz);
    const double se = d3dot(mn, eye) - mp.d;
    const double sp = d3dot(mn, p)   - mp.d;
    // A mirror reflects whichever face you look at, but the eye and the vertex must be on
    // the SAME face — otherwise the "reflection" is through the mirror's back.
    if (!(se * sp > 0.0)) return;
    const D3 eyeM = eye - mn * (2.0 * se);            // the eye's mirror image
    D3 wP = eyeM - p;
    const double D = d3len(wP);                       // unfolded path length
    if (D < 1e-9) return;
    wP = wP * (1.0 / D);
    const double shape = vt.shape(wP);
    if (shape < 0.0) return;                          // camera side behind the surface
    const double denom = d3dot(mn, wP);
    if (fabs(denom) < 1e-12) return;
    const double dP = (mp.d - d3dot(mn, p)) / denom;  // p + wP*dP lands on the plane
    if (dP <= 1e-7 || dP >= D - 1e-7) return;
    const D3 R = p + wP * dP;
    // Cheap extent reject before paying for the traversal that confirms the mirror.
    if (R.x < mp.lox || R.x > mp.hix || R.y < mp.loy || R.y > mp.hiy ||
        R.z < mp.loz || R.z > mp.hiz) return;

    int px, py; Real cosCam, dist2e;
    if (!cam.project(R.toR(), px, py, cosCam, dist2e)) return;
    const double omega = cam.pixelSolidAngle(cosCam);
    if (omega <= 0.0) return;

    const double dE = D - dP;                         // |R - eye| (R lies on the plane)
    const D3 wRE = (eye - R) * (1.0 / dE);            // mirror -> eye
    // Light-side leg first (stopping just short of the mirror's own surface): the
    // any-hit occlusion walk is cheaper than the closest-hit that confirms the mirror,
    // so a shadowed connection never pays for the expensive query. Both tests must
    // pass and neither draws RNG, so the order is unobservable.
    if (occluded(sc, (p + wP * 1e-6).toR(), wP.toR(), connMaxT(dP))) return;
    DHit hm;
    if (!dMirrorSeenAt(sc, eye, D3(0,0,0) - wRE, dE, hm)) return;

    dSplatMirrorLegs(sc, cam, film, hits, sc.mats[hm.matId], hm, px, py, omega,
                     p, wP, dP, R, wRE, dE, D, shape, lam, beta, w, nUp, rng);
}

// Connect vertex p to the pinhole `cam` by reflection in one SPHERICAL mirror. Same 1-D
// root solve + ray-differential Jacobian as the glass sphere, with a single reflection in
// place of the two refractions. Twin of Renderer::connectSpecularSphereMirror.
__device__ static void dConnectSpecularSphereMirror(const DScene& sc, const DCamera& cam,
        double* film, double* hits, const DSphere& S, const D3& p, const DSpecVtx& vt,
        const Real* lam, const Real* beta, const Real* w, int nUp, DRng& rng) {
    const D3 O(S.c); const double r = S.r; const D3 eye(cam.eye);
    const double dEyeO = d3len(eye - O);
    const double dPO   = d3len(p - O);
    if (fabs(dEyeO - r) < 1e-4 * r) return;           // eye ~on the surface: degenerate
    const bool outside = dEyeO > r;
    // The vertex has to be on the same side of the silvering as the eye, or there is no
    // reflection joining them (a mirror is opaque).
    if (outside ? (dPO <= r * 1.0001) : (dPO >= r * 0.9999)) return;

    // Plane(eye, p, O) — the reflection path off a sphere is planar by symmetry.
    D3 ex, ey;
    if (dEyeO < 1e-9) { D3 tb; d3onb(d3norm(p - O), ex, tb); }
    else              ex = (eye - O) * (1.0 / dEyeO);
    const D3 ap = p - O;
    const D3 perp = ap - ex * d3dot(ap, ex);
    const double perpLen = d3len(perp);
    if (perpLen < 1e-9) { D3 tb; d3onb(ex, ey, tb); }
    else                ey = perp * (1.0 / perpLen);
    const double ex_e = d3dot(eye - O, ex), ey_e = d3dot(eye - O, ey);
    const double px2 = d3dot(ap, ex), py2 = d3dot(ap, ey);

    // Signed perpendicular distance of p from the ray reflected at surface angle phi.
    // Macro rather than a lambda for the same reason the refraction traces are: no
    // std::function on device, and this must inline into the scan loop.
    #define D_TRACE2D_MIRROR(C1, S1, MISS, VALID) do {                               \
        VALID = false; MISS = 0.0;                                                   \
        double c1 = (C1), s1 = (S1);                                                 \
        double P1x = r * c1, P1y = r * s1;                                           \
        double dinx = P1x - ex_e, diny = P1y - ey_e;                                 \
        double dl = sqrt(dinx*dinx + diny*diny);                                     \
        if (dl >= 1e-12) {                                                           \
            dinx /= dl; diny /= dl;                                                  \
            double cn = dinx*c1 + diny*s1;          /* dot(din, outward normal) */   \
            /* Reachability: from OUTSIDE only the front face is hit first (cn<0);   \
               from inside the ray always leaves through the far face (cn>0). */     \
            if (fabs(cn) > 1e-6 && (outside == (cn < 0.0))) {                        \
                double doutx = dinx - 2.0*cn*c1, douty = diny - 2.0*cn*s1;           \
                double fw = (px2-P1x)*doutx + (py2-P1y)*douty;                       \
                if (fw > 0.0) {                     /* p on the forward side */      \
                    VALID = true;                                                    \
                    MISS = doutx*(py2-P1y) - douty*(px2-P1x);                        \
                }                                                                    \
            }                                                                        \
        }                                                                            \
    } while (0)

    // Scan the circle; bisect sign changes into chief reflection angles.
    const int NS = SPH_SCAN_N; double roots[4]; int nroot = 0;
    double prevMiss = 0.0, prevPhi = 0.0; bool prevValid = false;
    for (int i = 0; i <= NS && nroot < 4; ++i) {
        double phi = -DPI + (2.0 * DPI) * i / NS;
        bool v; double mss; D_TRACE2D_MIRROR(g_sphScanC[i], g_sphScanS[i], mss, v);
        if (v && prevValid && ((mss < 0.0) != (prevMiss < 0.0))) {
            double a = prevPhi, b = phi, fa = prevMiss;
            for (int k = 0; k < 40; ++k) {
                double mid = 0.5 * (a + b); bool vm; double fm;
                D_TRACE2D_MIRROR(cos(mid), sin(mid), fm, vm);
                if (!vm) break;
                if ((fm < 0.0) != (fa < 0.0)) b = mid; else { a = mid; fa = fm; }
            }
            roots[nroot++] = 0.5 * (a + b);
        }
        prevMiss = mss; prevValid = v; prevPhi = phi;
    }
    #undef D_TRACE2D_MIRROR

    for (int ri = 0; ri < nroot; ++ri) {
        const double phi = roots[ri];
        const D3 P1chief = O + ex * (r * cos(phi)) + ey * (r * sin(phi));
        // Everything below is pure and RNG-free, so the tests are ordered cheapest
        // first — an off-frame or backfacing connection rejects before paying for the
        // two extra differential reflections, and the traversals come last, any-hit
        // before closest-hit. Every value is computed by the same expression as
        // before, so the accepted set and the splats are unchanged.
        int px, py; Real cosCam, dist2e;
        if (!cam.project(P1chief.toR(), px, py, cosCam, dist2e)) continue;
        const double omega = cam.pixelSolidAngle(cosCam);
        if (omega <= 0.0) continue;

        const D3 d0 = d3norm(P1chief - eye);
        DSphereRefl ch;
        if (!dReflectOffSphere(eye, d0, S, ch)) continue;

        D3 wP = ch.P1 - p; const double dP = d3len(wP);
        if (dP < 1e-9) continue;
        wP = wP * (1.0 / dP);
        const double shape = vt.shape(wP);
        if (shape < 0.0) continue;

        D3 toEye = eye - ch.P1; const double dE = d3len(toEye);
        if (dE < 1e-9) continue;
        const D3 wRE = toEye * (1.0 / dE);

        // Ray-differential geometry factor G = dOmega_eye / dA_perp at p.
        D3 a1, a2; d3onb(d0, a1, a2);
        const double eps = 2e-4;
        DSphereRefl rA, rB;
        if (!dReflectOffSphere(eye, d3norm(d0 + a1 * eps), S, rA)) continue;
        if (!dReflectOffSphere(eye, d3norm(d0 + a2 * eps), S, rB)) continue;
        D3 e1, e2; d3onb(ch.exitDir, e1, e2);
        double ax, ay, bx, by;
        {   double den = d3dot(rA.exitDir, ch.exitDir);
            if (fabs(den) < 1e-9) den = (den < 0 ? -1e-9 : 1e-9);
            double s = d3dot(p - rA.P1, ch.exitDir) / den;
            D3 off = (rA.P1 + rA.exitDir * s) - p;
            ax = d3dot(off, e1); ay = d3dot(off, e2); }
        {   double den = d3dot(rB.exitDir, ch.exitDir);
            if (fabs(den) < 1e-9) den = (den < 0 ? -1e-9 : 1e-9);
            double s = d3dot(p - rB.P1, ch.exitDir) / den;
            D3 off = (rB.P1 + rB.exitDir * s) - p;
            bx = d3dot(off, e1); by = d3dot(off, e2); }
        const double jac = fabs(ax * by - ay * bx);
        if (jac < 1e-24) continue;                    // caustic singularity guard
        // The same G the flat mirror gets in closed form as 1/D^2; expressed here as an
        // equivalent unfolded distance so both connectors share one splat.
        const double G = (eps * eps) / jac;
        const double D = 1.0 / sqrt(G);

        if (occluded(sc, (p + wP * 1e-6).toR(), wP.toR(), connMaxT(dP))) continue;
        DHit hm;
        if (!dMirrorSeenAt(sc, eye, D3(0,0,0) - wRE, dE, hm)) continue;

        dSplatMirrorLegs(sc, cam, film, hits, sc.mats[hm.matId], hm, px, py, omega,
                         p, wP, dP, ch.P1, wRE, dE, D, shape, lam, beta, w, nUp, rng);
    }
}

// Splat vertex p to every camera through every smooth dielectric sphere (the refracted
// image of p) and off every perfect mirror (the reflected image of p). Device twin of
// Renderer::camSpecularSplatAllVtxN. Mode B only; draws no aperture RNG.
//
// `w[i]` is the vertex's per-λ weight (Lambertian rho / single-scatter albedo); vt.weight
// is ignored, and vt is copied per λ for the dielectric path. The loop nesting of the
// dielectric-sphere connection (λ, then sphere, then camera) is the one the scalar and
// hero wrappers used before this was factored, so its film accumulation order — and
// therefore its images — are unchanged bit for bit.
__device__ static void camSpecularSplatAllVtxN(const DScene& sc, const DCamSet& cs, int camMode,
        const D3& pd, const DSpecVtx& vt, const Real* lam, const Real* beta, const Real* w,
        int nUp, DRng& rng) {
    if (camMode != CAM_B) return;
    // sc.dielSph / sc.mirrorSph are the ascending indices of the qualifying spheres
    // (host-precomputed with the same predicates the loops below used to test inline),
    // so iteration — and therefore film accumulation — order is unchanged bit for bit,
    // and a scene with many plain spheres no longer scans them all per vertex per λ.
    if (sc.nDielSph > 0) {
        for (int i = 0; i < nUp; ++i) {
            DSpecVtx vi = vt; vi.weight = (double)w[i]; vi.lambda = lam[i];
            for (int k = 0; k < sc.nDielSph; ++k) {
                const DSphere& S = sc.sph[sc.dielSph[k]];
                const DMaterial& gm = sc.mats[S.matId];
                double ng = (double)specLookup(gm.ior, lam[i]);
                for (int c = 0; c < cs.nCam; ++c)
                    dConnectSpecularSphere(sc, cs.cams[c], cs.films[c], cs.hits[c], S, gm, ng,
                                           pd, vi, lam[i], (double)beta[i], rng);
            }
        }
    }
    // Mirrors. Achromatic geometry, so one solve serves the whole hero bundle.
    for (int k = 0; k < sc.nMirrorSph; ++k) {
        const DSphere& S = sc.sph[sc.mirrorSph[k]];
        for (int c = 0; c < cs.nCam; ++c)
            dConnectSpecularSphereMirror(sc, cs.cams[c], cs.films[c], cs.hits[c], S,
                                         pd, vt, lam, beta, w, nUp, rng);
    }
    for (int mi = 0; mi < sc.nMirrorPlanes; ++mi)
        for (int c = 0; c < cs.nCam; ++c)
            dConnectSpecularPlane(sc, cs.cams[c], cs.films[c], cs.hits[c],
                                  sc.mirrorPlanes[mi], pd, vt, lam, beta, w, nUp, rng);
}
__device__ static void camSpecularSplatAllVtx(const DScene& sc, const DCamSet& cs, int camMode,
                                              const D3& pd, const DSpecVtx& vt, Real lambda,
                                              Real beta, DRng& rng) {
    const Real w = (Real)vt.weight;
    camSpecularSplatAllVtxN(sc, cs, camMode, pd, vt, &lambda, &beta, &w, 1, rng);
}
// Surface vertex: refract the Lambertian reflection of p through every glass sphere, and
// reflect it in every mirror.
__device__ static void camSpecularSplatAll(const DScene& sc, const DCamSet& cs, int camMode,
                                           const DVec3& p, const DVec3& n, Real lambda,
                                           Real beta, Real rho, DRng& rng) {
    DSpecVtx vt; vt.volume = false; vt.np = D3(n); vt.weight = (double)rho; vt.g = 0;
    vt.med = nullptr; vt.lambda = lambda;
    camSpecularSplatAllVtx(sc, cs, camMode, D3(p), vt, lambda, beta, rng);
}
// Volume vertex: refract the fog in-scatter at p through every glass sphere, so the
// glowing haze itself bends through the glass the camera flies through.
__device__ static void camSpecularSplatVolumeAll(const DScene& sc, const DMedium& med,
                                                 const DCamSet& cs, int camMode, const DVec3& p,
                                                 const DVec3& wIn, Real lambda, Real beta, DRng& rng) {
    DSpecVtx vt; vt.volume = true; vt.wIn = D3(wIn); vt.g = med.g;
    vt.weight = (double)medAlbedo(med, lambda);
    vt.med = &med; vt.lambda = lambda;
    camSpecularSplatAllVtx(sc, cs, camMode, D3(p), vt, lambda, beta, rng);
}

// ============================ megakernel ============================

// Wavelength from an emitter's SPD CDF given an explicit uniform `u` in [0,1). Split out
// so the hero-wavelength sampler can pass stratified strata (u + i/C wrapped) that share
// the emitter's CDF, while the scalar sampleLambda below just draws its own u. The CDF
// search stays in double (host-baked table); the returned wavelength/pdf are Real.
__device__ static Real sampleLambdaU(const DScene& sc, const DEmitter& em, double u, Real& pdf) {
    const double* cdf = sc.lightCdfAll + em.cdfOffset;
    int lo = 0, hi = em.cdfN - 1;
    while (lo + 1 < hi) { int mid = (lo + hi) / 2; if (cdf[mid] <= u) lo = mid; else hi = mid; }
    double c0 = cdf[lo], c1 = cdf[lo + 1];
    double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
    pdf = (Real)((c1 - c0) / em.cdfStep);
    return (Real)(DLMIN + (lo + frac) * em.cdfStep);
}
__device__ static Real sampleLambda(const DScene& sc, const DEmitter& em, DRng& rng, Real& pdf) {
    return sampleLambdaU(sc, em, (double)rng.uniform(), pdf);
}

// Power-weighted emitter selection (mirrors Scene::selectEmitter). Single
// emitter consumes no randomness, preserving the RNG stream for parity with CPU.
__device__ static int selectEmitter(const DScene& sc, double u) {
    int lo = 0, hi = sc.nEmitters - 1;
    while (lo < hi) { int mid = (lo + hi) / 2; if (sc.emitCdf[mid] < u) lo = mid + 1; else hi = mid; }
    return lo;
}

// Invert an emissive volume's Planck-shaped wavelength CDF at uniform u in [0,1).
// Device twin of EmissionSampler::sampleAt (spectrum.h): returns lambda (nm) and sets
// pdf = per-nm density = binMass/step. Mirrors the host bit-for-bit.
__device__ static double dEmissionSampleLambda(const DEmissiveVolume& ev, double u, double& pdf) {
    const double* cdf = ev.lamCdf;
    int lo = 0, hi = ev.lamN;   // cdf has lamN+1 entries
    while (lo + 1 < hi) { int mid = (lo + hi) / 2; (cdf[mid] <= u ? lo : hi) = mid; }
    double c0 = cdf[lo], c1 = cdf[lo + 1];
    double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
    double w = LAMBDA_MIN + (lo + frac) * ev.lamStep;
    pdf = (c1 - c0) / ev.lamStep;
    return w;
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
    if (tx.stoch.on && tx.gaussRgb && tx.jhLut) {   // O7 — twin of Texture::reflectanceAt
        // Blend the three crops in linear RGB — the same planes, lattice and weights the
        // raster preview uses — then convert that colour to a spectrum through the shared
        // LUT. See stochtile.h for why this is not done in coefficient space.
        double c[3];
        stochSample(tx.stoch, tx.gaussRgb, tx.lutRgb, 3, tx.w, tx.h, tx.filter,
                    (double)u, (double)v, c);
        double cs[3];
        stochJhCoeff(tx.jhLut, c[0], c[1], c[2], cs);
        return dReflAt(cs, lambda);
    }
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
    if (tx.stoch.on && tx.gaussGray) {   // O7 — twin of Texture::scalarAt's branch
        double o[1];
        stochSample(tx.stoch, tx.gaussGray, tx.lutGray, 1, tx.w, tx.h, tx.filter,
                    (double)u, (double)v, o);
        return o[0];
    }
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
// The pattern VM's `tex:` hook: pattern_device.cuh reaches a texture only through this
// member, so `tex:` inside a pattern samples exactly what Texture::scalarAt does on the
// host. (Declared with DTexture; defined here because it forwards to dTexScalarAt.)
__device__ inline double DTexture::patScalarAt(double u, double v) const {
    return dTexScalarAt(*this, (Real)u, (Real)v);
}

// Tangent-space normal at (u,v) — device twin of Texture::sampleNormalTS (C6). Bilerps
// the linear RGB, remaps [0,1]->[-1,1], normalizes. v flipped so v=0 is image bottom.
__device__ static DVec3 dTexNormalAt(const DTexture& tx, Real u, Real v) {
    if (!tx.rgb) return DVec3{(Real)0, (Real)0, (Real)1};
    if (tx.stoch.on && tx.gaussRgb) {   // O7 — the host reaches this via sampleRgb
        double o[3];
        stochSample(tx.stoch, tx.gaussRgb, tx.lutRgb, 3, tx.w, tx.h, tx.filter,
                    (double)u, (double)v, o);
        DVec3 n{(Real)(2.0 * o[0] - 1.0), (Real)(2.0 * o[1] - 1.0), (Real)(2.0 * o[2] - 1.0)};
        Real l = (Real)sqrt((double)(n.x * n.x + n.y * n.y + n.z * n.z));
        return (l > (Real)1e-12) ? DVec3{n.x / l, n.y / l, n.z / l}
                                 : DVec3{(Real)0, (Real)0, (Real)1};
    }
    auto texel = [&](int x, int y) -> DVec3 {
        size_t o = ((size_t)y * tx.w + x) * 3;
        return DVec3{(Real)tx.rgb[o], (Real)tx.rgb[o + 1], (Real)tx.rgb[o + 2]};
    };
    DVec3 c;
    if (tx.filter == 0) {   // Nearest
        int x = dWrapIndex((int)floor((double)u * tx.w), tx.w, tx.wrap);
        int y = dWrapIndex((int)floor((1.0 - (double)v) * tx.h), tx.h, tx.wrap);
        c = texel(x, y);
    } else {
        double tu = (double)u * tx.w - 0.5, tv = (1.0 - (double)v) * tx.h - 0.5;
        double flx = floor(tu), fly = floor(tv);
        double fx = tu - flx, fy = tv - fly;
        int x0 = dWrapIndex((int)flx, tx.w, tx.wrap), x1 = dWrapIndex((int)flx + 1, tx.w, tx.wrap);
        int y0 = dWrapIndex((int)fly, tx.h, tx.wrap), y1 = dWrapIndex((int)fly + 1, tx.h, tx.wrap);
        DVec3 a = texel(x0, y0) * (Real)(1 - fx) + texel(x1, y0) * (Real)fx;
        DVec3 b = texel(x0, y1) * (Real)(1 - fx) + texel(x1, y1) * (Real)fx;
        c = a * (Real)(1 - fy) + b * (Real)fy;
    }
    DVec3 n{(Real)2 * c.x - (Real)1, (Real)2 * c.y - (Real)1, (Real)2 * c.z - (Real)1};
    Real l = sqrt(dot(n, n));
    return (l > (Real)1e-12) ? n * ((Real)1 / l) : DVec3{(Real)0, (Real)0, (Real)1};
}

// Perturb a hit's shading normal by a bound tangent-space normal map (C6), device twin
// of Scene::applyNormalMap. Builds a TBN from the (ray-oriented) shading normal + the
// hit tangent, rotates the sampled tangent-space normal into world, replaces h.n. Called
// from the device closestHit choke point so every GPU path is consistent.
__device__ static inline void dApplyNormalMap(const DScene& sc, DHit& h) {
    if (!h.valid || h.matId < 0) return;
    const DMaterial& m = sc.mats[h.matId];
    if (m.normalTex < 0 || m.normalTex >= sc.nTex) return;
    const DTexture& tx = sc.textures[m.normalTex];
    if (!tx.rgb) return;
    DVec3 tn = dTexNormalAt(tx, h.u, h.v);
    DVec3 N = h.n;
    DVec3 T = h.tangent - N * dot(N, h.tangent);
    Real tl = sqrt(dot(T, T));
    if (tl < (Real)1e-9) return;
    T = T * ((Real)1 / tl);
    DVec3 B = cross(N, T) * h.bitangentSign;
    Real s = (Real)m.normalStrength;
    DVec3 pert = T * (tn.x * s) + B * (tn.y * s) + N * tn.z;
    Real pl = sqrt(dot(pert, pert));
    if (pl > (Real)1e-12) h.n = pert * ((Real)1 / pl);
}

// ---- procedural pattern VM ------------------------------------------------
// The double-precision VM (dPatHash3 / dPatValueNoise / dPatternEval) now lives in
// pattern_device.cuh, shared verbatim with the preview rasterizer (raster_cuda.cu).
// It reaches this backend's textures through DTexture::patScalarAt, above.
// ---- FP32 twin of the pattern VM: used ONLY for DF_EXPR field formulas inside the
// sphere-trace march (dFieldLeafSDFF). Same integer hash lattice, float blend.
__device__ static float dPatHash3F(int ix, int iy, int iz) {
    unsigned int h = (unsigned int)ix * 374761393u + (unsigned int)iy * 668265263u
                   + (unsigned int)iz * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (float)h * (1.0f / 4294967295.0f);   // [0,1]
}
__device__ static float dPatValueNoiseF(float x, float y, float z) {
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float tx = x - fx, ty = y - fy, tz = z - fz;
    float ux = tx * tx * (3.0f - 2.0f * tx);
    float uy = ty * ty * (3.0f - 2.0f * ty);
    float uz = tz * tz * (3.0f - 2.0f * tz);
    float c000 = dPatHash3F(ix,     iy,     iz);
    float c100 = dPatHash3F(ix + 1, iy,     iz);
    float c010 = dPatHash3F(ix,     iy + 1, iz);
    float c110 = dPatHash3F(ix + 1, iy + 1, iz);
    float c001 = dPatHash3F(ix,     iy,     iz + 1);
    float c101 = dPatHash3F(ix + 1, iy,     iz + 1);
    float c011 = dPatHash3F(ix,     iy + 1, iz + 1);
    float c111 = dPatHash3F(ix + 1, iy + 1, iz + 1);
    float x00 = c000 + (c100 - c000) * ux;
    float x10 = c010 + (c110 - c010) * ux;
    float x01 = c001 + (c101 - c001) * ux;
    float x11 = c011 + (c111 - c011) * ux;
    float y0  = x00 + (x10 - x00) * uy;
    float y1  = x01 + (x11 - x01) * uy;
    return y0 + (y1 - y0) * uz;
}
__device__ static float dPatternEvalF(const PatNodeF* nodes, int n,
                                      float x, float y, float z, float f,
                                      float nx, float ny, float nz, float r,
                                      float u, float v, const DPatEnv& env) {
    float st[64]; int sp = 0;
    float reg[PAT_CSE_REGS];    // CSE registers (float: bit-identical to re-running the
                                // stored subtree in this evaluator's own precision)
    for (int i = 0; i < n; ++i) {
        const PatNodeF& nd = nodes[i];
        switch ((PatOp)nd.op) {
            case PatOp::Const:    st[sp++] = nd.a; break;
            case PatOp::VarX:     st[sp++] = x;  break;
            case PatOp::VarY:     st[sp++] = y;  break;
            case PatOp::VarZ:     st[sp++] = z;  break;
            case PatOp::VarF:     st[sp++] = f;  break;
            case PatOp::VarNx:    st[sp++] = nx; break;
            case PatOp::VarNy:    st[sp++] = ny; break;
            case PatOp::VarNz:    st[sp++] = nz; break;
            case PatOp::VarR:     st[sp++] = r;  break;
            case PatOp::VarU:     st[sp++] = u;  break;
            case PatOp::VarV:     st[sp++] = v;  break;
            // This fp32 VM only ever evaluates DF_EXPR implicit FIELDS, which are
            // functions of a point in space with no surface at all — so `curv` is 0
            // here for the same reason `f`/`nx`/`u` are, and the host agrees (the CPU
            // field path builds its PatCtx with curv = 0 too).
            case PatOp::VarCurv:  st[sp++] = 0.0f; break;
            case PatOp::VarCavity: st[sp++] = 0.0f; break;
            // `fw` (O8 stage 2) is 0 here for a second, stronger reason: a field formula
            // is evaluated at march samples along a ray, not at a shading point, so there
            // is no footprint to speak of — and 0 means "unknown", i.e. fnoise() runs
            // unfiltered, which is what the sphere-tracer needs (a filtered SDF would
            // round off the very detail the march is trying to find, and would make the
            // distance bound non-conservative at grazing angles).
            case PatOp::VarFootprint: st[sp++] = 0.0f; break;
            case PatOp::VarT:     st[sp++] = 0.0f; break;
            case PatOp::Neg:      st[sp-1] = -st[sp-1]; break;
            case PatOp::Abs:      st[sp-1] = fabsf(st[sp-1]); break;
            case PatOp::Sqrt:     st[sp-1] = sqrtf(fmaxf(0.0f, st[sp-1])); break;
            case PatOp::Sin:      st[sp-1] = sinf(st[sp-1]); break;
            case PatOp::Cos:      st[sp-1] = cosf(st[sp-1]); break;
            case PatOp::Tan:      st[sp-1] = tanf(st[sp-1]); break;
            case PatOp::Exp:      st[sp-1] = expf(st[sp-1]); break;
            case PatOp::Log:      st[sp-1] = logf(fmaxf(1e-30f, st[sp-1])); break;
            case PatOp::Floor:    st[sp-1] = floorf(st[sp-1]); break;
            case PatOp::Fract:    st[sp-1] = st[sp-1] - floorf(st[sp-1]); break;
            case PatOp::Sign:     st[sp-1] = (float)((st[sp-1] > 0.0f) - (st[sp-1] < 0.0f)); break;
            case PatOp::Saturate: st[sp-1] = fminf(1.0f, fmaxf(0.0f, st[sp-1])); break;
            case PatOp::Add:      { float b = st[--sp]; st[sp-1] += b; break; }
            case PatOp::Sub:      { float b = st[--sp]; st[sp-1] -= b; break; }
            case PatOp::Mul:      { float b = st[--sp]; st[sp-1] *= b; break; }
            case PatOp::Div:      { float b = st[--sp]; st[sp-1] = (b != 0.0f) ? st[sp-1] / b : 0.0f; break; }
            case PatOp::Mod:      { float b = st[--sp]; st[sp-1] = (b != 0.0f) ? st[sp-1] - b * floorf(st[sp-1] / b) : 0.0f; break; }
            case PatOp::Pow:      { float b = st[--sp]; st[sp-1] = powf(st[sp-1], b); break; }
            case PatOp::Min:      { float b = st[--sp]; st[sp-1] = fminf(st[sp-1], b); break; }
            case PatOp::Max:      { float b = st[--sp]; st[sp-1] = fmaxf(st[sp-1], b); break; }
            case PatOp::Atan2:    { float b = st[--sp]; st[sp-1] = atan2f(st[sp-1], b); break; }
            case PatOp::Step:     { float b = st[--sp]; st[sp-1] = (b >= st[sp-1]) ? 1.0f : 0.0f; break; }
            case PatOp::Clamp:    { float hi = st[--sp], lo = st[--sp]; st[sp-1] = fminf(hi, fmaxf(lo, st[sp-1])); break; }
            case PatOp::Mix:      { float t = st[--sp], b = st[--sp]; st[sp-1] = st[sp-1] + (b - st[sp-1]) * t; break; }
            case PatOp::Smoothstep: {
                float xx = st[--sp], e1 = st[--sp], e0 = st[sp-1];
                float tt = (e1 != e0) ? (xx - e0) / (e1 - e0) : 0.0f;
                tt = fminf(1.0f, fmaxf(0.0f, tt));
                st[sp-1] = tt * tt * (3.0f - 2.0f * tt);
                break;
            }
            case PatOp::Noise:    { float zz = st[--sp], yy = st[--sp]; st[sp-1] = dPatValueNoiseF(st[sp-1], yy, zz); break; }
            case PatOp::DNoise: {  // POV vector noise is double-only: promote / demote like PovFn
                float zz = st[--sp], yy = st[--sp];
                double vout[3];
                povDNoise((double)st[sp-1], (double)yy, (double)zz, vout);
                st[sp-1] = (float)vout[(int)nd.a];
                break;
            }
            case PatOp::DTurb: {   // ditto (octave fBm of DNoise)
                float om = st[--sp], la = st[--sp], oc = st[--sp];
                float zz = st[--sp], yy = st[--sp];
                double vout[3];
                povDTurbulence((double)st[sp-1], (double)yy, (double)zz,
                               (double)oc, (double)la, (double)om, vout);
                st[sp-1] = (float)vout[(int)nd.a];
                break;
            }
            case PatOp::Worley: {  // cellular noise (double core): promote / demote like PovFn
                float m = st[--sp], zz = st[--sp], yy = st[--sp];
                double w[3];
                int mi = (int)floorf(m + 0.5f);
                if (mi < 0) mi = 0; if (mi > 2) mi = 2;
                patWorley((double)st[sp-1], (double)yy, (double)zz, mi, w);
                int sel = (int)nd.a;
                st[sp-1] = (float)((sel == 3) ? w[2] : (sel == 2) ? (w[1] - w[0]) : w[sel]);
                break;
            }
            case PatOp::Gabor: {   // Gabor noise (double core): promote / demote like PovFn
                float dz = st[--sp], dy = st[--sp], dx = st[--sp];
                float ff = st[--sp], zz = st[--sp], yy = st[--sp];
                st[sp-1] = (float)patGabor((double)st[sp-1], (double)yy, (double)zz,
                                           (double)ff, (double)dx, (double)dy, (double)dz);
                break;
            }
            case PatOp::FNoise: {  // filtered fBm (double core): promote / demote like PovFn
                float oc = st[--sp], ww = st[--sp], zz = st[--sp], yy = st[--sp];
                st[sp-1] = (float)patFilteredNoise((double)st[sp-1], (double)yy, (double)zz,
                                                   (double)ww, (double)oc);
                break;
            }
            case PatOp::BlueNoise: {  // Poisson-disk placement (double core): promote / demote
                float rr = st[--sp], zz = st[--sp], yy = st[--sp];
                double b[3];
                patBlueNoise((double)st[sp-1], (double)yy, (double)zz, (double)rr, b);
                int sel = (int)nd.a;
                st[sp-1] = (float)((sel == 3) ? b[2] : (sel == 2) ? (b[1] - b[0]) : b[sel]);
                break;
            }
            case PatOp::PovFn: {   // POV internals are double-only: promote args, demote result
                int id = (int)nd.a;
                int na = povFnArity(id);
                double args[POV_FN_MAX_ARGS];
                for (int k = na - 1; k >= 0; --k) args[k] = (double)st[--sp];
                st[sp++] = (float)povFnEval(id, args);
                break;
            }
            case PatOp::Tex: {   // args pushed as (u, v); the sampler is double-only, so
                float vv = st[--sp];                       // promote / demote like PovFn
                int   ti = (int)nd.a;
                st[sp-1] = (env.tex && ti >= 0 && ti < env.nTex)
                             ? (float)dTexScalarAt(env.tex[ti], (double)st[sp-1], (double)vv)
                             : 0.0f;
                break;
            }
            case PatOp::Grid: {
                // Arity is the GRID's own dimensionality, so the operands can only be
                // popped once the header is in hand — which is why the not-found case must
                // abandon the program rather than push a placeholder (see dPatternEval).
                // patGridSample is the SHARED __host__ __device__ sampler from pattern.h;
                // it is double-only, so coordinates are promoted and the result demoted,
                // exactly as PatOp::PovFn does above.
                int gi = (int)nd.a;
                if (gi < 0 || gi >= env.nGrids || !env.grids) return 0.0f;
                const PatGrid& g = env.grids[gi];
                int gnd = g.ndim < 1 ? 1 : (g.ndim > PAT_ND_MAX_DIM ? PAT_ND_MAX_DIM : g.ndim);
                double co[PAT_ND_MAX_DIM];
                for (int k = gnd - 1; k >= 0; --k) co[k] = (double)st[--sp];
                st[sp++] = (float)patGridSample(g, env.dataPool, env.dataPoolN, co);
                break;
            }
            case PatOp::Scatter: {   // same contract as Grid, inverse-distance blended
                int si = (int)nd.a;
                if (si < 0 || si >= env.nScatters || !env.scatters) return 0.0f;
                const PatScatter& sc = env.scatters[si];
                int snd = sc.ndim < 1 ? 1 : (sc.ndim > PAT_ND_MAX_DIM ? PAT_ND_MAX_DIM : sc.ndim);
                double co[PAT_ND_MAX_DIM];
                for (int k = snd - 1; k >= 0; --k) co[k] = (double)st[--sp];
                st[sp++] = (float)patScatterSample(sc, env.dataPool, env.dataPoolN, co);
                break;
            }
            case PatOp::StReg:    reg[(int)nd.a] = st[sp-1]; break;   // save, keep on stack
            case PatOp::LdReg:    st[sp++] = reg[(int)nd.a]; break;   // reuse saved value
        }
    }
    return sp > 0 ? st[0] : 0.0f;
}
// Evaluate a bound pattern at a hit (device twin of patternScalarAt/patCtxFromHit).
// The implicit field value f is 0 (like the CPU: intersectImplicit never sets it), so
// the `f` variable is 0 at surfaces on both backends.
__device__ static double dPatternScalarAt(const DScene& sc, int pat, const DHit& h) {
    const DPattern& p = sc.patterns[pat];
    double px = h.p.x, py = h.p.y, pz = h.p.z;
    double r = sqrt(px * px + py * py + pz * pz);
    return dPatternEval(sc.patNodes + p.off, p.n, px, py, pz, 0.0,
                        h.n.x, h.n.y, h.n.z, r, h.u, h.v, h.curv, dCavityOf(sc, h), h.fw, dPatEnvOf(sc));
}

// Fritsch-Carlson monotone-cubic tangent at node k (device twin of recFCTangent).
__device__ static double dRecFCTangent(const double* pos, const double* sec, int n, int k) {
    if (k == 0)     return sec[0];
    if (k == n - 1) return sec[n - 2];
    double s0 = sec[k - 1], s1 = sec[k];
    if (s0 * s1 <= 0.0) return 0.0;
    double h0 = pos[k]     - pos[k - 1];
    double h1 = pos[k + 1] - pos[k];
    double w0 = 2.0 * h1 + h0, w1 = h1 + 2.0 * h0;
    return (w0 + w1) / (w0 / s0 + w1 / s1);
}
// Evaluate one scalar-record stop's per-hit expression program at the hit.
__device__ static double dRecStopVal(const DScene& sc, const DRecScalarStop& s, const DHit& h) {
    if (s.exprN <= 0) return 0.0;
    double px = h.p.x, py = h.p.y, pz = h.p.z;
    double r = sqrt(px * px + py * py + pz * pz);
    return dPatternEval(sc.recDrivers + s.exprOff, s.exprN, px, py, pz, 0.0,
                        h.n.x, h.n.y, h.n.z, r, h.u, h.v, h.curv, dCavityOf(sc, h), h.fw, dPatEnvOf(sc));
}
// Sample a scalar record channel at driver position `d` (device twin of recSampleScalar):
// evaluate each stop's per-hit expression, then interpolate by the record's interp mode.
__device__ static double dRecSampleScalar(const DScene& sc, const DRecScalarStop* stops,
                                          int n, int interp, const DHit& h, double d) {
    if (n <= 0) return 0.0;
    if (n == 1) return dRecStopVal(sc, stops[0], h);
    double lo = stops[0].pos, hi = stops[n - 1].pos;
    if (d < lo) d = lo; else if (d > hi) d = hi;
    int i = 0;                                             // recLocate
    while (i < n - 2 && d > stops[i + 1].pos) ++i;
    double p0 = stops[i].pos, p1 = stops[i + 1].pos;
    double span = p1 - p0;
    double t = (span > 1e-12) ? (d - p0) / span : 0.0;
    if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
    if (interp == (int)RecInterp::Nearest)
        return dRecStopVal(sc, stops[t < 0.5 ? i : i + 1], h);
    double v0 = dRecStopVal(sc, stops[i], h), v1 = dRecStopVal(sc, stops[i + 1], h);
    if (interp == (int)RecInterp::Linear) return v0 + (v1 - v0) * t;
    // Smooth: monotone cubic Hermite (evaluate all stop values + secants once).
    double vs[64], ps[64], sec[64];
    int m = n < 64 ? n : 64;
    for (int k = 0; k < m; ++k) { vs[k] = dRecStopVal(sc, stops[k], h); ps[k] = stops[k].pos; }
    for (int k = 0; k < m - 1; ++k) {
        double hh = ps[k + 1] - ps[k];
        sec[k] = (hh > 1e-12) ? (vs[k + 1] - vs[k]) / hh : 0.0;
    }
    if (i > m - 2) i = m - 2;
    double mk  = dRecFCTangent(ps, sec, m, i);
    double mk1 = dRecFCTangent(ps, sec, m, i + 1);
    double hh  = ps[i + 1] - ps[i];
    double t2 = t * t, t3 = t2 * t;
    double h00 =  2 * t3 - 3 * t2 + 1;
    double h10 =      t3 - 2 * t2 + t;
    double h01 = -2 * t3 + 3 * t2;
    double h11 =      t3 -     t2;
    return h00 * vs[i] + h10 * hh * mk + h01 * vs[i + 1] + h11 * hh * mk1;
}
// Per-hit roughness from a bound scalar record, if the material binds one (device twin of
// the record branch of materialRoughness). Returns true + sets `out` (clamped [0,1]).
__device__ static bool dRecordRoughness(const DScene& sc, const DMaterial& m, const DHit& h, Real& out) {
    if (m.recRoughMode < 0) return false;
    double v;
    if (m.recRoughMode == 0) {                             // direct scalar expression
        double px = h.p.x, py = h.p.y, pz = h.p.z, r = sqrt(px * px + py * py + pz * pz);
        v = dPatternEval(sc.recDrivers + m.recRoughDrvOff, m.recRoughDrvN,
                         px, py, pz, 0.0, h.n.x, h.n.y, h.n.z, r, h.u, h.v, h.curv, dCavityOf(sc, h), h.fw,
                         dPatEnvOf(sc));
    } else if (m.recRoughMode == 1) {                      // constant selStop (one stop, per-hit)
        v = dRecStopVal(sc, sc.recScalarStops[m.recRoughStopOff], h);
    } else {                                               // per-hit driven
        double px = h.p.x, py = h.p.y, pz = h.p.z, r = sqrt(px * px + py * py + pz * pz);
        double d = dPatternEval(sc.recDrivers + m.recRoughDrvOff, m.recRoughDrvN,
                                px, py, pz, 0.0, h.n.x, h.n.y, h.n.z, r, h.u, h.v, h.curv, dCavityOf(sc, h), h.fw,
                                dPatEnvOf(sc));
        v = dRecSampleScalar(sc, sc.recScalarStops + m.recRoughStopOff, m.recRoughStopN,
                             m.recRoughInterp, h, d);
    }
    out = (Real)(v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));
    return true;
}

// Per-hit glossy roughness / thin-film thickness (device twins of materialRoughness
// / materialFilmThickness): a bound record (highest priority), then a bound pattern or
// scalar map's value at the hit, else the constant.
__device__ static Real dMatRoughness(const DScene& sc, const DMaterial& m, const DHit& h) {
    Real rr;
    if (dRecordRoughness(sc, m, h, rr)) return rr;
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

// Deterministic dMixResolveChild for mode W (device twin of scene.h mixResolveDominant +
// mixDominantChild). A pattern/texture-driven two-way mix picks whichever child dominates AT
// THIS POINT, so the blend becomes a hard threshold at t == 0.5 rather than a stochastic
// dither: the preview shows a crisp boundary where the render shows a smooth gradient. That
// is the honest cost of one deterministic sample per pixel, and it stays put frame to frame.
// A constant-weight mix picks the heaviest lobe, unless the leftover absorption slice
// outweighs every single lobe (then -1, i.e. absorbed).
__device__ static int dMixResolveDominant(const DScene& sc, const DMaterial& m, const DHit& h) {
    if ((m.mixWeightPat >= 0 || m.mixWeightTex >= 0) && m.mixCount == 2) {
        Real t = (m.mixWeightPat >= 0)
               ? (Real)dPatternScalarAt(sc, m.mixWeightPat, h)
               : (Real)dTexScalarAt(sc.textures[m.mixWeightTex], h.u, h.v);
        if (t < 0) t = 0; else if (t > 1) t = 1;
        return (t >= (Real)0.5) ? m.mixChild[0] : m.mixChild[1];
    }
    int best = -1; double bestW = 0.0, sum = 0.0;
    for (int k = 0; k < m.mixCount; ++k) {
        if (m.mixWeight[k] > bestW) { bestW = m.mixWeight[k]; best = m.mixChild[k]; }
        sum += m.mixWeight[k];
    }
    if (1.0 - sum > bestW) return -1;   // leftover absorbs more than any single lobe
    return best;
}

// Per-hit reflectance of a baked driven record channel at driver `d` and wavelength
// `lambda` (device twin of recReflectanceAt): map d -> LUT position over [lo,hi], lerp
// the neighbouring bins' JH sigmoid coeffs, evaluate the sigmoid. `coeff` points at the
// channel's REC_LUT_N*3-double slice (recCoeff + recReflOff).
__device__ static Real dRecReflAt(const double* coeff, int N, double lo, double hi,
                                  double d, Real lambda) {
    if (N <= 0) return (Real)0;
    if (N == 1) return dReflAt(coeff, lambda);
    double t = (hi > lo) ? (d - lo) / (hi - lo) : 0.0;
    if (t < 0) t = 0; else if (t > 1) t = 1;
    double fidx = t * (N - 1);
    int i = (int)fidx; if (i > N - 2) i = N - 2;
    double f = fidx - i;
    const double* a = coeff + 3 * i;
    const double* b = coeff + 3 * (i + 1);
    double c[3] = { a[0] + (b[0] - a[0]) * f,
                    a[1] + (b[1] - a[1]) * f,
                    a[2] + (b[2] - a[2]) * f };
    return dReflAt(c, lambda);
}

// Reflect-slot reflectance from a per-hit driven parametric record, if the material
// binds one (device twin of the driven branch of recordReflectBound). Returns true and
// sets `out`; false when no record drives reflect per-hit (constant selStop bindings are
// pre-baked into m.reflect, so they take the ordinary specLookup path). The single point
// of truth so diffuse albedo AND specular tint see identical driven reflectance.
__device__ static bool dRecordReflect(const DScene& sc, const DMaterial& m,
                                      const DHit& h, Real lambda, Real& out) {
    if (m.recReflDriven != 1) return false;
    double px = h.p.x, py = h.p.y, pz = h.p.z;
    double r = sqrt(px * px + py * py + pz * pz);
    double d = dPatternEval(sc.recDrivers + m.recReflDrvOff, m.recReflDrvN,
                            px, py, pz, 0.0, h.n.x, h.n.y, h.n.z, r, h.u, h.v, h.curv, dCavityOf(sc, h), h.fw,
                            dPatEnvOf(sc));
    out = dRecReflAt(sc.recCoeff + m.recReflOff, REC_LUT_N,
                     (double)m.recReflLo, (double)m.recReflHi, d, lambda);
    return true;
}

// Per-hit multiplier from a bound reflectPat, clamped to [0,1] (device twin of host
// reflectPatMul). 1 when unbound, so both reflect accessors apply it unconditionally.
__device__ static Real dReflectPatMul(const DScene& sc, const DMaterial& m, const DHit& h) {
    if (m.reflectPat < 0) return (Real)1;
    return (Real)clamp01(dPatternScalarAt(sc, m.reflectPat, h));
}

// Reflect-slot reflectance for the SPECULAR families (mirror / glossy / grating /
// halfmirror): a driven record if present, else the constant baked reflect spectrum,
// scaled by a bound reflect pattern (device twin of host reflectSlot; these types never
// bind a reflect texture).
__device__ static Real dReflectSlot(const DScene& sc, const DMaterial& m, const DHit& h, Real lambda) {
    Real v;
    if (!dRecordReflect(sc, m, h, lambda, v)) v = specLookup(m.reflect, lambda);
    return m.reflectPat < 0 ? v : v * dReflectPatMul(sc, m, h);
}

// Diffuse reflectance at a hit: a driven parametric record (highest priority), else a
// bound texture, else the constant baked reflect spectrum — then scaled by a bound
// reflect pattern (mirrors host diffuseReflectance).
__device__ static Real dDiffuseRho(const DScene& sc, const DMaterial& m, const DHit& h, Real lambda) {
    Real rv;
    if (!dRecordReflect(sc, m, h, lambda, rv)) {
        if (m.reflectTex >= 0) {
            const DTexture& tx = sc.textures[m.reflectTex];
            rv = (m.triplanarScale > 0.0)
                     ? dTexReflTriplanar(tx, h.p, h.ng, m.triplanarScale, lambda)
                     : dTexReflAt(tx, h.u, h.v, lambda);
        } else {
            rv = specLookup(m.reflect, lambda);
        }
    }
    return clamp01(m.reflectPat < 0 ? rv : rv * dReflectPatMul(sc, m, h));
}

// Transmit-slot value at a hit (device twin of host transmitSlot): the constant baked
// transmit spectrum scaled by a bound transmit pattern. Serves BOTH readings of the slot
// — a filter's gel transmittance T(lambda) and a translucent's back-lobe albedo rhoT —
// so every device transmit read goes through here, exactly as the host does.
__device__ static Real dTransmitSlot(const DScene& sc, const DMaterial& m, const DHit& h, Real lambda) {
    Real v = specLookup(m.transmit, lambda);
    if (m.transmitPat < 0) return v;
    return v * (Real)clamp01(dPatternScalarAt(sc, m.transmitPat, h));
}

// ---- emission slot: the two halves of `emit pattern:` on the device -----------------
// Emission is read from BOTH sides of transport — at a hit ON an emissive surface (the
// s=0 / direct-hit strategy) and at a point the emitter SAMPLER drew (NEE and light
// subpaths) — and MIS combines them, so the two must agree pointwise. dEmitPatMul covers
// the first (device twin of host emitSlot's slotPatMul, identical in form to
// dReflectPatMul) and dEmitterSamplePointPat the second (device twin of host
// emitterSamplePoint + emitterPatMulAt). Every device emission read goes through one of
// them; a missed site would bias the image rather than drop a visible effect.

// Per-hit multiplier from a bound emitPat, clamped to [0,1]. 1 when unbound, so callers
// can multiply unconditionally.
__device__ static double dEmitPatMul(const DScene& sc, int emitPat, const DHit& h) {
    if (emitPat < 0 || emitPat >= sc.nPatterns) return 1.0;
    return clamp01(dPatternScalarAt(sc, emitPat, h));
}

// The emission-pattern multiplier at a point on `em`, given the point's own texture
// coordinates (device twin of host emitterPatMulAt). Clamped to [0,1] like the
// reflect/transmit slot patterns, so a runaway formula can never manufacture light.
// The implicit field value f is 0, matching dPatternScalarAt and the host's makePatCtx.
__device__ static double dEmitterPatMulAt(const DScene& sc, const DEmitter& em,
                                          const DVec3& y, const DVec3& nOut,
                                          double uu, double vv) {
    if (em.emitPat < 0 || em.emitPat >= sc.nPatterns) return 1.0;
    const DPattern& p = sc.patterns[em.emitPat];
    double px = y.x, py = y.y, pz = y.z;
    double r = sqrt(px * px + py * py + pz * pz);
    return clamp01(dPatternEval(sc.patNodes + p.off, p.n, px, py, pz, 0.0,
                                nOut.x, nOut.y, nOut.z, r, uu, vv, 0.0, 0.0, 0.0, dPatEnvOf(sc)));
}

// Draw a point on `em` and return the emission-pattern multiplier there, so a caller can
// write `Le = specLookup(em.emitSpd, lambda) * invPdfLambda * pmul` (device twin of host
// emitterSamplePoint(scene, ...)). 1.0 whenever no pattern is bound, which is every scene
// that does not use the feature — those keep bit-identical draws, since the uv outputs are
// the only extra work and they do not touch the RNG.
__device__ static double dEmitterSamplePointPat(const DScene& sc, const DEmitter& em,
                                                double u1, double u2,
                                                DVec3& y, DVec3& nOut) {
    if (em.emitPat < 0) { emitterSamplePoint(em, u1, u2, y, nOut); return 1.0; }
    double uu = 0.0, vv = 0.0;
    emitterSamplePoint(em, u1, u2, y, nOut, &uu, &vv);
    return dEmitterPatMulAt(sc, em, y, nOut, uu, vv);
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

// Absolute env radiance in a direction (mirrors EnvMap::radiance): the per-texel JH
// reflectance times the baked normalised illuminant. Image env only (e.scale non-null).
__device__ static double dEnvRadiance(const DEnvMap& e, const DVec3& d, Real lambda) {
    int ti = dEnvTexel(e, d);
    double refl = (double)dReflAt(&e.coeff[3 * ti], lambda) * e.scale[ti];
    return refl * (double)specLookup(e.illum, lambda);
}

// Solid-angle pdf of the env importance sampler for a direction (mirrors EnvMap::pdf /
// Distribution2D::pdf): condFunc[iv*w+iu]/margFuncInt divided by the lat-long Jacobian
// 2*PI^2*sin(theta). Image env only.
__device__ static double dEnvPdf(const DEnvMap& e, const DVec3& d) {
    const double PI = 3.14159265358979323846;
    double y = fmin(fmax((double)d.y, -1.0), 1.0);
    double theta = acos(y);
    double phi = atan2((double)d.z, (double)d.x);
    double v = theta / PI;
    double u = phi / (2.0 * PI) + 0.5 - e.rot;
    u -= floor(u);
    int iu = (int)(u * e.w); if (iu < 0) iu = 0; if (iu >= e.w) iu = e.w - 1;
    int iv = (int)(v * e.h); if (iv < 0) iv = 0; if (iv >= e.h) iv = e.h - 1;
    if (e.margFuncInt == 0.0) return 0.0;
    double sinT = sin(theta);
    if (sinT <= 0.0) return 0.0;
    double distPdf = e.condFunc[(size_t)iv * e.w + iu] / e.margFuncInt;
    return distPdf / (2.0 * PI * PI * sinT);
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
__device__ static bool genPhoton(const DScene& sc, const DCamSet& cs,
                                 int camMode, DRng& rng,
                                 DVec3& ro, DVec3& rd, Real& beta, Real& lambda, double& eEmitted) {
    // ROADMAP C3: split birth between surface/point/env emitters and volumetric "fire"
    // emission by power. grandTotal = totalPower + totalEmissionPower; the volumeBirth
    // test short-circuits (drawing NO extra RNG) when there are no emissive volumes, so
    // every non-fire scene stays bit-identical to before.
    const double grandTotal = sc.totalPower + sc.totalEmissionPower;
    if (grandTotal <= 0.0) return false;
    const bool volumeBirth = (sc.emissiveVolN > 0) &&
                             ((double)rng.uniform() * grandTotal < sc.totalEmissionPower);
    if (volumeBirth) {
        // Power-weighted emissive-volume selection.
        double r = (double)rng.uniform() * sc.totalEmissionPower;
        int vi = 0;
        for (; vi + 1 < sc.emissiveVolN; ++vi) { r -= sc.emissiveVolumes[vi].power; if (r <= 0.0) break; }
        const DEmissiveVolume ev = sc.emissiveVolumes[vi];
        const DMedium& fm = sc.media[ev.mediumIndex];
        // Uniform position in the grid AABB; lambda importance-sampled from the volume's
        // Planck-at-emitKelvin CDF. beta = grandTotal*ke/(meanKe*dLam*p(lambda)) so the
        // isotropic 1/(4pi)/(dist^2*Omega) splat reproduces the emission line-integral
        // (host twin: render.h tracePhoton volume-birth branch).
        DVec3 origin = DVec3{ (Real)(ev.bmin.x + (ev.bmax.x - ev.bmin.x) * (double)rng.uniform()),
                              (Real)(ev.bmin.y + (ev.bmax.y - ev.bmin.y) * (double)rng.uniform()),
                              (Real)(ev.bmin.z + (ev.bmax.z - ev.bmin.z) * (double)rng.uniform()) };
        double pdfLam = 0.0;
        double lam = dEmissionSampleLambda(ev, (double)rng.uniform(), pdfLam);
        lambda = (Real)lam;
        double ke = dMedEmissionAt(fm, origin, lam);
        const double dLamE = LAMBDA_MAX - LAMBDA_MIN;
        beta = (Real)((ev.meanKe > 0.0 && pdfLam > 0.0)
                      ? grandTotal * ke / (ev.meanKe * dLamE * pdfLam) : 0.0);
        eEmitted += beta;
        if (beta <= (Real)0) return false;   // cold voxel: nothing to emit or transport
        // Isotropic emission direction.
        double z = 1.0 - 2.0 * (double)rng.uniform();
        double sr = sqrt(fmax(0.0, 1.0 - z * z));
        double phi = 2.0 * DPI * (double)rng.uniform();
        DVec3 dir = DVec3{ (Real)(sr * cos(phi)), (Real)(sr * sin(phi)), (Real)z };
        // Direct-visibility emission splat (the flame seen directly by the camera).
        camSplatEmissionAll(sc, cs, camMode, origin, lambda, beta, rng);
        ro = origin + dir * RAY_EPS; rd = dir;
        return true;
    }
    // Power-weighted emitter selection (single emitter draws no randomness).
    int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
    const DEmitter em = sc.emitters[ei];
    Real u1 = rng.uniform(), u2 = rng.uniform();
    DVec3 origin, emitN, dir;
    Real spotW = (Real)1;                            // spot direction reweight (else 1)
    bool envImage = false; double envPdfW = 0.0;     // image env: reweight below
    double emitPatW = 1.0;                           // `emit pattern:` factor at the point
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
    } else if (em.shape == 6) {
        // Distant directional sun (device twin of render.h). Sample the travel direction
        // inside the solar cone (pdf 1/Omega), then the entry point on a disk of radius R
        // perpendicular to it (pdf 1/(pi R^2)) — the same upstream-disk trick the env
        // uses, but aimed instead of isotropic, so EVERY photon crosses the scene rather
        // than one in ~10^5. The joint pdf 1/(Omega*pi*R^2) is exactly 1/envGeom, so
        // beta = emitIntegral*envGeom is analog with no reweight.
        dir = dSunSampleCone(em, em.beamDir, (double)u1, (double)u2);
        DVec3 t, b; onb(dir, t, b);
        double rdd = sc.sceneRadius * sqrt((double)rng.uniform());
        double pd = 2.0 * 3.14159265358979323846 * (double)rng.uniform();
        origin = sc.sceneCenter - dir * (Real)sc.sceneRadius
               + t * (Real)(rdd * cos(pd)) + b * (Real)(rdd * sin(pd));
        emitN = dir;
    } else {
        // quad: constant normal; sphere: surface point. Also returns this point's
        // `emit pattern:` factor — 1.0 (and a bit-identical draw) when unpatterned.
        emitPatW = dEmitterSamplePointPat(sc, em, u1, u2, origin, emitN);
        dir = em.collimated ? em.beamDir : cosineHemisphere(emitN, rng);
    }
    Real pdfL = 0;
    lambda = sampleLambda(sc, em, rng, pdfL);
    if (pdfL <= 0) return false;
    // When emissive volumes exist the emitter-vs-fire split already consumed the
    // totalPower/grandTotal factor, so a chosen emitter photon carries the full
    // grandTotal (host twin: render.h). Otherwise the ordinary emitter scaling.
    beta = (Real)((sc.emissiveVolN > 0) ? grandTotal
                  : ((sc.nEmitters == 1) ? em.power : sc.totalPower));
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
    // An emission pattern is a pure post-multiplier on the photon's carried power: the
    // emitter is still SELECTED by its unpatterned power and the point still drawn
    // uniformly over its area, so no pdf changes and the estimator stays unbiased
    // (host twin: render.h). eEmitted is credited the patterned value so the energy
    // report matches what actually leaves the surface.
    if (emitPatW != 1.0) beta = (Real)((double)beta * emitPatW);
    eEmitted += beta;

    // Connect the emitter itself to the camera (makes the source visible): model
    // B splats to the pinhole, model A splats through the finite lens pupil. Model
    // C instead catches photons that physically arrive. A spot is a point light
    // with no projected area, so it has no direct term; a distant sun's disc is at
    // infinity, so its direct view is the backend-agnostic addEnvBackground pass.
    if (em.shape != 2 && em.shape != 3 && em.shape != 6) {
        // Emitter vertex: ns==ng==emitN, so the adjoint correction is identically 1
        // (wi is irrelevant here — pass emitN).
        splatSurfaceAll(sc, cs, camMode, origin, emitN, emitN, emitN, lambda, beta, (Real)1, rng);
        camSpecularSplatAll(sc, cs, camMode, origin, emitN, lambda, beta, (Real)1, rng);
    }

    ro = origin + dir * RAY_EPS; rd = dir;
    return true;
}

// Fiber (D_HAIR) interaction — the body of interactSpecular's hair branch, split out
// and explicitly __noinline__. NOT a style choice: the BCSDF works in double precision
// through a fat frame (DHairShade = fiber frame + Bcsdf + the dhair::sample locals),
// and inlined into interactSpecular those locals would merge into the shared
// shadeStep/traceHeroPhoton frames, permanently inflating every kernel's
// statically-computed MIN_STACK — which the driver reserves per thread × ~200k
// resident threads of VRAM on every launch, hair scene or not. As a separate call
// frame the bytes are only on the stack during an actual hair hit, and MIN_STACK
// grows by max(hair chain, other chains) instead of their sum.
//
// Fiber BCSDF (Marschner/Chiang + Yan medulla; dhair above). It routes through
// interactSpecular rather than beside Diffuse because, like the dielectric family, it
// is wavelength-COUPLED: sigma_a is per-λ and an authored reflectance is inverted
// per-λ, so the hero tracer de-heroes onto this path. The interaction is a
// connect-then-scatter, like Fluorescent: the lobes are narrow but finite, so the
// vertex IS visible to a mode-A/B camera and must splat before it scatters.
// Device twin of the CPU MatType::Hair case (render.h).
__device__ __noinline__ static int interactHair(const DScene& sc, const DCamSet& cs,
        int camMode, const DMaterial& m, const DHit& h,
        DVec3& ro, DVec3& rd, Real beta, Real lambda, DRng& rng, double& eAbsorbed) {
    const DVec3 wPrev = -rd;                       // toward the light-side vertex
    const DHairShade hs = dHairShadeAt(sc, m, h, lambda, wPrev);
    if (camMode == CAM_A || camMode == CAM_B)
        splatSurfaceAllHair(sc, cs, camMode, h.p, h.n, hs, lambda, beta, rng);

    const double u0 = rng.uniform(), u1 = rng.uniform(), u2 = rng.uniform(), u3 = rng.uniform();
    double pdf = 0.0, fv = 0.0;
    const dhair::V3 wl = dhair::sample(hs.b, hs.woLocal, u0, u1, u2, u3, pdf, fv);
    if (!(pdf > 0.0) || !(fv > 0.0)) { eAbsorbed += beta; return WF_TERMINATE; }
    // fv*cosLong/pdf is EXACTLY T = sum_p A_p <= 1 (the total Fresnel/Beer
    // attenuation over the lobes — per-lobe pdf weights are A_p/T and everything
    // else cancels), so Russian-roulette on it and leave beta untouched: the same
    // trick Mirror plays with its reflectance, but the number is the physics.
    const double cosLong = dhair::safeSqrt(1.0 - dhair::sqr(dhair::clampd(wl.x, -1.0, 1.0)));
    double T = fv * cosLong / pdf; T = T < 0.0 ? 0.0 : (T > 1.0 ? 1.0 : T);
    if ((double)rng.uniform() >= T) { eAbsorbed += beta; return WF_TERMINATE; }
    const DVec3 wo = dhair::toWorld(hs.fr, wl);
    // TT and TRT leave through the FAR side of a real solid strand, so step clear
    // of the tube's own body (dHairExitOffset); near side is the ordinary RAY_EPS.
    ro = h.p + wo * dHairExitOffset(hs, h.n, wo); rd = wo;
    return WF_CONTINUE;   // beta unchanged (RR carried the weight)
}

// Specular / wavelength-switching material interaction (the nine families that are NOT
// Diffuse / DiffuseTransmit): Dielectric, ThinFilm, Multilayer, Mirror, Grating,
// HalfMirror, Filter, Glossy, Fluorescent. Split out of shadeStep so BOTH the scalar
// forward tracer AND the hero-wavelength tracer (after de-hero) share one source of truth
// for these lobes — the device twin of Renderer::interactPhotonSpecular (render.h). `m` is
// the already-resolved material (post-Mix); `matIndex` drives the dielectric priority stack.
// Mutates ro/rd/beta/lambda (Fluorescent Stokes shift) and returns WF_CONTINUE/WF_TERMINATE.
__device__ static int interactSpecular(const DScene& sc, const DCamSet& cs, int camMode,
        int diffraction, const DMaterial& m, int matIndex, const DHit& h,
        DVec3& ro, DVec3& rd, Real& beta, Real& lambda, DRng& rng, double& eAbsorbed,
        DMediumStack& stk) {
    if (m.type == D_DIELECTRIC) {
        DVec3 nro, nrd; dDielectricStep(sc, m, h, rd, lambda, rng, matIndex, stk, nro, nrd);
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
        Real r = clamp01(dReflectSlot(sc, m, h, lambda));
        if (rng.uniform() >= r) { eAbsorbed += beta; return WF_TERMINATE; }
        DVec3 o = reflectv(rd, h.n); ro = h.p + h.n * RAY_EPS; rd = o; return WF_CONTINUE;
    } else if (m.type == D_GRATING) {
        Real r = clamp01(dReflectSlot(sc, m, h, lambda));
        if (rng.uniform() >= r) { eAbsorbed += beta; return WF_TERMINATE; }
        DVec3 nro, nrd;
        if (!gratingDiffract(m, h, rd, lambda, diffraction, rng, nro, nrd)) { eAbsorbed += beta; return WF_TERMINATE; }
        ro = nro; rd = nrd; return WF_CONTINUE;
    } else if (m.type == D_HALFMIRROR) {
        Real r = clamp01(dReflectSlot(sc, m, h, lambda));
        if (rng.uniform() < r) { DVec3 o = reflectv(rd, h.n); ro = h.p + h.n * RAY_EPS; rd = o; }
        else { ro = h.p + rd * RAY_EPS; }
        return WF_CONTINUE;
    } else if (m.type == D_FILTER) {
        // Colored gel / Wratten filter (device twin of render.h MatType::Filter): a thin
        // non-scattering absorber. Pass straight through; survive with prob T(lambda),
        // else absorb. RR on the transmittance keeps beta unchanged and unbiased.
        Real t = clamp01(dTransmitSlot(sc, m, h, lambda));
        if (rng.uniform() >= t) { eAbsorbed += beta; return WF_TERMINATE; }
        ro = h.p + rd * RAY_EPS;   // straight through, direction unchanged
        return WF_CONTINUE;
    } else if (m.type == D_GLOSSY) {
        Real r = clamp01(dReflectSlot(sc, m, h, lambda));
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
        // Elastic splat at the incoming lambda, then a glow splat at a Stokes-shifted
        // lambda' drawn ONCE (camera-independent) — matching the CPU camSplatAll order,
        // so a multi-camera model-B pass stays bit-identical. Skipped for model C.
        DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);   // geo normal on shading side
        DVec3 wiPrev = -rd;                                             // toward previous (light-side)
        if (camMode == CAM_A || camMode == CAM_B) {
            splatSurfaceAll(sc, cs, camMode, h.p, h.n, ngo, wiPrev, lambda, beta, rho, rng);
            if (canGlow) {
                Real lp = sampleFluoEmit(sc, m, rng);
                splatSurfaceAll(sc, cs, camMode, h.p, h.n, ngo, wiPrev, lp, beta, (Real)(aEff * m.fluoYield), rng);
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
        { DVec3 wo = cosineHemisphere(h.n, rng);
          beta *= dShadingAdjointCorr(wiPrev, wo, h.n, ngo);   // Veach adjoint (1 when ns==ng)
          ro = h.p + h.n * RAY_EPS; rd = wo; return WF_CONTINUE; }
    } else if (m.type == D_HAIR) {
        // Split out (__noinline__) so its fat double-precision frame is only paid on an
        // actual hair hit — see the comment on interactHair.
        return interactHair(sc, cs, camMode, m, h, ro, rd, beta, lambda, rng, eAbsorbed);
    }
    return WF_CONTINUE;   // unreachable: caller dispatches only the specular types
}

// Advance a photon by one bounce given its precomputed intersection `h`. Mutates
// ro/rd/beta and accumulates absorbed/sensor/escaped energy. Returns WF_TERMINATE
// when the path ends (absorbed / escaped / landed on the sensor), else WF_CONTINUE
// with ro/rd set for the next segment. `h` is the closestHit(sc, ro, rd) result.
__device__ static int shadeStep(const DScene& sc, const DCamSet& cs,
                                int camMode, int diffraction, const DHit& h,
                                DVec3& ro, DVec3& rd, Real& beta, Real& lambda, DRng& rng,
                                double& eAbsorbed, double& eSensor, double& eEscaped,
                                DMediumStack& stk, DRng* crng = nullptr) {
    Real dSurf = h.valid ? h.t : BIG;

    // PHOTON-BEAMS gather active for THIS step: shared multi-camera pass, a medium exists,
    // and the caller handed us an independent RNG stream. When on, the photon does NOT
    // redirect in the medium (it crosses straight) and each camera resamples its own
    // in-scatter point below — so skip the analog medium-collision sampling here.
    const bool doBeam = cs.beamGather && crng && cs.nCam > 1 && camMode != CAM_C && sc.mediaN > 0;

    // fog free-flight; dEvent is the nearer of surface hit / volume collision.
    bool mediumEvent = false; int scatterMed = -1; DVec3 mp; Real dEvent = dSurf;
    if (sc.mediaN > 0 && !doBeam) {
        // Superposition of all media: each does its own delta (Woodcock) tracking (or
        // exact analytic free-flight if homogeneous); the earliest collision wins and
        // its medium (scatterMed) drives the scatter. Device twin of sampleMediaCollision.
        Real tMed; int which;
        if (dMediaSampleCollision(sc, ro, rd, dSurf, lambda, rng, tMed, which)) {
            mediumEvent = true; scatterMed = which; mp = ro + rd * tMed; dEvent = tMed;
        }
    }

    // Model C perspective catch: if the photon flies through the aperture
    // nearer than the surface/fog event, it lands on the film. Analog physics.
    if (camMode == CAM_C) {
        int px, py;
        // Model C never shares a trace (it consumes the photon), so nCam==1 here.
        if (cs.cams[0].catchPhoton(ro, rd, dEvent, px, py)) {
            // Flux->film-irradiance normaliser (see host render.h): keep brute-force C
            // on the SAME absolute scale as A/B (per-camera constant; auto-exposed
            // scenes unaffected).
            Real cCell = (Real)1 / (Real)(cs.cams[0].pixelPlaneArea() * cs.cams[0].filmDist * cs.cams[0].filmDist);
            filmAdd(cs.films[0], cs.hits[0], cs.cams[0].resX, px, py, lambda, beta * cCell);
            eSensor += beta; return WF_TERMINATE;
        }
    }

    // Beer-Lambert attenuation over the free path just travelled inside a dielectric
    // (colored/attenuating glass), applied before the event is processed (matches the
    // host: attenuate over dEvent using the medium carried from the previous vertex).
    // `betaPre` is throughput BEFORE this attenuation, so a beam-gather camera re-applies
    // glass absorption to ITS own resampled collision distance tC (not the photon's).
    Real betaPre = beta;
    {
        int cm = stk.topMat();
        Real a = (cm >= 0) ? (Real)specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
        if (a > 0) beta *= exp(-a * dEvent);
    }

    // PHOTON-BEAMS single-scatter gather (device twin of the CPU -beams block in render.h).
    // The photon crosses the medium in a STRAIGHT beam (analog redirect skipped above), and
    // each camera independently samples ONE in-scatter point along [ro, dSurf] with its OWN
    // stream `crng`, then splats it. Decoupling the shared deposit from the per-camera gather
    // gives a volumetric flyby independent per-frame noise instead of one frozen speckle.
    // Unbiased for SINGLE scattering (each resample is a free-flight collision pdf sigma_t*Tr,
    // and connectVolume's albedo*phase*T_cam*beta cancels that Tr). Multiple scattering is
    // intentionally omitted — the right trade for a crisp view-dependent bow / glory / rays.
    if (doBeam) {
        int cm = stk.topMat();
        Real aC = (cm >= 0) ? (Real)specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
        for (int c = 0; c < cs.nCam; ++c) {
            Real tC; int whichC;
            if (!dMediaSampleCollision(sc, ro, rd, dSurf, lambda, *crng, tC, whichC))
                continue;   // this camera saw no in-scatter along this beam
            DVec3 xc = ro + rd * tC;
            Real betaC = (aC > 0) ? betaPre * exp(-aC * tC) : betaPre;
            const DMedium& smc = sc.media[whichC];
            if (camMode == CAM_A) connectLensVolume(sc, smc, cs.cams[c], cs.films[c], cs.hits[c], xc, rd, lambda, betaC, *crng);
            else                  connectVolume(sc, smc, cs.cams[c], cs.films[c], cs.hits[c], xc, rd, lambda, betaC, *crng);
            // Per-camera specular volume splat (fog seen through a smooth sphere caustic):
            // a 1-camera slice of the set so the "All" helper targets only camera c.
            DCamSet cs1 = cs; cs1.nCam = 1; cs1.cams = &cs.cams[c]; cs1.films = &cs.films[c]; cs1.hits = &cs.hits[c];
            camSpecularSplatVolumeAll(sc, smc, cs1, camMode, xc, rd, lambda, betaC, *crng);
        }
        // Attenuate the photon by the medium extinction over the whole crossing (single-
        // scatter transmission) so surfaces behind the fog are correctly dimmed; the removed
        // energy (out-scattered + absorbed) is booked as absorbed. Then continue STRAIGHT.
        Real before = beta;
        beta *= dMediaTransmittance(sc, ro, rd, dSurf, lambda, *crng);
        eAbsorbed += (double)(before - beta);
    }

    if (mediumEvent) {
        const DMedium& sm = sc.media[scatterMed];
        splatVolumeAll(sc, sm, cs, camMode, mp, rd, lambda, beta, rng);
        camSpecularSplatVolumeAll(sc, sm, cs, camMode, mp, rd, lambda, beta, rng);
        if (rng.uniform() >= medAlbedo(sm, lambda)) { eAbsorbed += beta; return WF_TERMINATE; }
        Real phPdf;   // scatter dir from HG or the rainbow droplet phase (pdf unused: p/pdf==1)
        DVec3 nd = dMedPhaseSample(sm, rd, lambda, rng, phPdf);
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
    if (m.type == D_DIELECTRIC || m.type == D_THINFILM || m.type == D_MULTILAYER ||
        m.type == D_MIRROR || m.type == D_GRATING || m.type == D_HALFMIRROR ||
        m.type == D_FILTER || m.type == D_GLOSSY || m.type == D_FLUORESCENT ||
        m.type == D_HAIR) {
        // The specular / wavelength-switching lobes (+ the fiber BCSDF, which is
        // wavelength-coupled like them) — shared with the hero tracer.
        return interactSpecular(sc, cs, camMode, diffraction, m, matIndex, h,
                                ro, rd, beta, lambda, rng, eAbsorbed, stk);
    } else if (m.type == D_DIFFUSETRANSMIT) {
        // Two-lobe Lambertian (device twin of render.h DiffuseTransmit): `reflect` into
        // the front (+n) hemisphere, `transmit` into the back (-n) hemisphere. Splat BOTH
        // lobes — connect()/connectLens() self-reject the wrong-side lobe (cosSurf<=0), so
        // passing the flipped normal images whichever side the camera is on. Non-specular,
        // so a directly-viewed translucent solid is VISIBLE in model B (unlike dielectric).
        Real rhoR = dDiffuseRho(sc, m, h, lambda);
        Real rhoT = clamp01(dTransmitSlot(sc, m, h, lambda));
        Real sum = rhoR + rhoT;
        if (sum > (Real)1) { rhoR /= sum; rhoT /= sum; sum = (Real)1; }   // energy guard
        DVec3 nb = h.n * (Real)(-1);
        DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);   // geo normal on shading side
        DVec3 wiPrev = -rd;                                             // toward previous (light-side)
        depositPhoton(cs, h.p, h.n, beta, lambda);   // photon-map deposit (mode M)
        // Both lobes get the adjoint correction; |cos| in the factor makes it lobe-agnostic,
        // so h.n / ngo serve the transmit lobe too (nb = -h.n is used only for the splat side).
        splatSurfaceAll(sc, cs, camMode, h.p, h.n, ngo, wiPrev, lambda, beta, rhoR, rng);
        splatSurfaceAll(sc, cs, camMode, h.p, nb,  ngo * (Real)(-1), wiPrev, lambda, beta, rhoT, rng);
        camSpecularSplatAll(sc, cs, camMode, h.p, h.n, lambda, beta, rhoR, rng);
        camSpecularSplatAll(sc, cs, camMode, h.p, nb,  lambda, beta, rhoT, rng);
        // Analog scatter: reflect (prob rhoR), transmit (prob rhoT), else absorb — beta
        // unchanged on a scatter (like the diffuse case), plus the adjoint correction.
        Real u = rng.uniform();
        if (u < rhoR)      { DVec3 wo = cosineHemisphere(h.n, rng); beta *= dShadingAdjointCorr(wiPrev, wo, h.n, ngo); ro = h.p + h.n * RAY_EPS; rd = wo; return WF_CONTINUE; }
        else if (u < sum)  { DVec3 wo = cosineHemisphere(nb,  rng); beta *= dShadingAdjointCorr(wiPrev, wo, h.n, ngo); ro = h.p + nb  * RAY_EPS; rd = wo; return WF_CONTINUE; }
        eAbsorbed += beta; return WF_TERMINATE;
    } else {
        // Diffuse (texture-sampled reflectance when the material binds a texture).
        Real rho = dDiffuseRho(sc, m, h, lambda);
        DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);   // geo normal on shading side
        DVec3 wiPrev = -rd;                                             // toward previous (light-side)
        depositPhoton(cs, h.p, h.n, beta, lambda);   // photon-map deposit (mode M)
        splatSurfaceAll(sc, cs, camMode, h.p, h.n, ngo, wiPrev, lambda, beta, rho, rng);
        camSpecularSplatAll(sc, cs, camMode, h.p, h.n, lambda, beta, rho, rng);
        if (rng.uniform() >= rho) { eAbsorbed += beta; return WF_TERMINATE; }
        { DVec3 wo = cosineHemisphere(h.n, rng);
          beta *= dShadingAdjointCorr(wiPrev, wo, h.n, ngo);   // Veach adjoint (1 when ns==ng)
          ro = h.p + h.n * RAY_EPS; rd = wo; return WF_CONTINUE; }
    }
}

// ==================== hero-wavelength forward tracer (device) ================
// Device twin of Renderer::tracePhotonHero (render.h): one path carries a HERO wavelength
// (index 0) plus C-1 stratified SECONDARY wavelengths that SHARE a single BVH walk. The
// geometry is driven by the hero λ; each λ carries its own throughput beta[i]. At any
// dispersive / wavelength-switching interface the secondaries "de-hero" (terminate,
// beta[0] *= C) and the path continues as an ordinary single-λ photon — energy is
// preserved exactly and the estimator is unbiased. Only CHROMATIC noise is reduced (all C
// share one geometric sample). Gated OFF when the scene has participating media or a GRIN
// region (see launchForward), so the fog/GRIN code paths never appear here.

// Mode-B pinhole connect for all `nUp` live wavelengths through ONE shared geometry
// (draws no RNG). Per-λ ordering matches connect().
__device__ static void connectHero(const DScene& sc, const DCamera& cam, double* film, double* hits,
        const DVec3& p, const DVec3& n, const DVec3& ng, const DVec3& wi,
        const Real* lam, const Real* beta, const Real* rho, int nUp, DRng& rng) {
    DVec3 toCam = cam.eye - p;
    Real dist = length(toCam);
    DVec3 wdir = toCam / dist;
    Real cosSurf = dot(n, wdir);
    if (cosSurf <= 0) return;
    Real stG = dShadowTerminatorG(wdir, n, ng);
    if (stG <= (Real)0) return;
    int px, py; Real cosCam, dist2;
    if (!cam.project(p, px, py, cosCam, dist2)) return;
    if (occluded(sc, p + ng * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    Real corr = dShadingAdjointCorr(wi, wdir, n, ng);
    double solidAngle = cam.pixelSolidAngle(cosCam);
    Real geo = cosSurf * corr / (Real)((double)dist2 * solidAngle) * stG;
    for (int i = 0; i < nUp; ++i) {
        Real contrib = beta[i] * (rho[i] / (Real)DPI) * geo;
        if (sc.mediaN > 0) contrib *= dMediaTransmittance(sc, p, wdir, dist, lam[i], rng);
        filmAdd(film, hits, cam.resX, px, py, lam[i], contrib);
    }
}
// Model-A finite-lens splat for all `nUp` live wavelengths through ONE shared aperture
// sample (drawn once — the thin-lens pupil is achromatic, so C wavelengths legitimately
// share the connection, exactly like connectLensHero on the CPU). Per-λ ordering matches
// connectLens().
__device__ static void connectLensHero(const DScene& sc, const DCamera& cam, double* film, double* hits,
        const DVec3& p, const DVec3& n, const DVec3& ng, const DVec3& wi,
        const Real* lam, const Real* beta, const Real* rho, int nUp, DRng& rng) {
    Real R  = (Real)cam.apertureR;
    Real rr = R * sqrt(rng.uniform());
    Real a  = (Real)(2.0 * DPI) * rng.uniform();
    DVec3 A = cam.eye + cam.u * (rr * cos(a)) + cam.v * (rr * sin(a));
    DVec3 toA = A - p;
    Real dist = length(toA);
    if (dist < (Real)1e-9) return;
    DVec3 wdir = toA / dist;
    Real cosSurf = dot(n, wdir);
    if (cosSurf <= 0) return;
    Real stG = dShadowTerminatorG(wdir, n, ng);
    if (stG <= (Real)0) return;
    Real cosLens = -dot(wdir, cam.w);
    if (cosLens <= (Real)1e-6) return;
    int px, py;
    if (!cam.lensImage(A, wdir, px, py)) return;
    if (occluded(sc, p + ng * RAY_EPS, wdir, dist - (Real)2 * RAY_EPS)) return;
    Real corr = dShadingAdjointCorr(wi, wdir, n, ng);
    Real cellNorm = (Real)1 / (Real)(cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
    Real geo = cosSurf * corr * cosLens * (R * R) / (dist * dist) * stG * cellNorm;
    for (int i = 0; i < nUp; ++i) {
        Real contrib = beta[i] * rho[i] * geo;
        if (sc.mediaN > 0) contrib *= dMediaTransmittance(sc, p, wdir, dist, lam[i], rng);
        filmAdd(film, hits, cam.resX, px, py, lam[i], contrib);
    }
}
// Splat a surface vertex to every camera, all `nUp` live wavelengths (mode A/B). No-op for
// mode C (which consumes the photon) and for the mode-M deposit pass (nCam == 0).
__device__ static void splatSurfaceAllHero(const DScene& sc, const DCamSet& cs, int camMode,
        const DVec3& p, const DVec3& n, const DVec3& ng, const DVec3& wi,
        const Real* lam, const Real* beta, const Real* rho, int nUp, DRng& rng) {
    for (int c = 0; c < cs.nCam; ++c) {
        if (camMode == CAM_B) connectHero(sc, cs.cams[c], cs.films[c], cs.hits[c], p, n, ng, wi, lam, beta, rho, nUp, rng);
        else if (camMode == CAM_A) connectLensHero(sc, cs.cams[c], cs.films[c], cs.hits[c], p, n, ng, wi, lam, beta, rho, nUp, rng);
    }
}
// Refract each live wavelength's Lambertian reflection through every glass sphere, and
// reflect the bundle in every mirror. The sphere REFRACTION cannot share geometry: the
// glass IOR is dispersive (per-λ), so each wavelength traces its own refracted image —
// exactly what makes a glass-sphere caustic chromatically dispersed. A MIRROR is
// achromatic, so its geometry IS shared across the bundle and only the reflectance varies
// per λ. Draws no RNG (mode B only inside).
__device__ static void camSpecularSplatAllHero(const DScene& sc, const DCamSet& cs, int camMode,
        const DVec3& p, const DVec3& n, const Real* lam, const Real* beta, const Real* rho,
        int nUp, DRng& rng) {
    DSpecVtx vt; vt.volume = false; vt.np = D3(n); vt.weight = (double)rho[0]; vt.g = 0;
    vt.med = nullptr; vt.lambda = lam[0];
    camSpecularSplatAllVtxN(sc, cs, camMode, D3(p), vt, lam, beta, rho, nUp, rng);
}

// Emit one hero photon: fills ro/rd and the per-λ lam[]/beta[] bundle, sets secAlive, does
// the direct emitter->camera splat, and accrues emitted energy (sum of live betas). Returns
// false when the hero wavelength draws a zero pdf (skip this photon). Emission geometry is
// byte-identical to genPhoton (λ-independent); only the wavelength/throughput bundle differs.
__device__ static bool genPhotonHero(const DScene& sc, const DCamSet& cs, int camMode, int C,
        DRng& rng, DVec3& ro, DVec3& rd, Real* lam, Real* beta, bool& secAlive, double& eEmitted) {
    int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
    const DEmitter em = sc.emitters[ei];
    Real u1 = rng.uniform(), u2 = rng.uniform();
    DVec3 origin, emitN, dir;
    Real spotW = (Real)1;
    bool envImage = false; double envPdfW = 0.0;
    double emitPatW = 1.0;                           // `emit pattern:` factor at the point
    if (em.shape == 2) {
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
    } else if (em.shape == 6) {                       // distant sun — see genPhoton
        dir = dSunSampleCone(em, em.beamDir, (double)u1, (double)u2);
        DVec3 t, b; onb(dir, t, b);
        double rdd = sc.sceneRadius * sqrt((double)rng.uniform());
        double pd = 2.0 * 3.14159265358979323846 * (double)rng.uniform();
        origin = sc.sceneCenter - dir * (Real)sc.sceneRadius
               + t * (Real)(rdd * cos(pd)) + b * (Real)(rdd * sin(pd));
        emitN = dir;
    } else {
        emitPatW = dEmitterSamplePointPat(sc, em, u1, u2, origin, emitN);
        dir = em.collimated ? em.beamDir : cosineHemisphere(emitN, rng);
    }

    // Hero + stratified secondaries from this emitter's SPD (one base draw, C-1 wrapped
    // strata). The hero must have a valid pdf; a dead secondary simply carries beta 0.
    double u = (double)rng.uniform();
    Real pdf0 = 0;
    lam[0] = sampleLambdaU(sc, em, u, pdf0);
    if (pdf0 <= 0) return false;
    for (int i = 1; i < C; ++i) {
        double uu = u + (double)i / C;
        if (uu >= 1.0) uu -= 1.0;                     // wrap into [0,1)
        Real pdfi;
        lam[i] = sampleLambdaU(sc, em, uu, pdfi);
    }
    Real base = (Real)((sc.nEmitters == 1) ? em.power : sc.totalPower);
    base *= spotW;
    for (int i = 0; i < C; ++i) beta[i] = base / (Real)C;
    if (envImage) {
        int ti = dEnvTexel(sc.env, dir);
        for (int i = 0; i < C; ++i) {
            double rad = sc.env.scale[ti] * (double)dReflAt(&sc.env.coeff[3 * ti], lam[i]);
            double avg = sc.env.avgScale * (double)dReflAt(sc.env.avgCoeff, lam[i]);
            double denom = 4.0 * 3.14159265358979323846 * envPdfW * avg;
            beta[i] = (denom > 0.0) ? (Real)((double)beta[i] * rad / denom) : (Real)0;
        }
    }
    // Achromatic post-multiplier — see genPhoton for why this changes no pdf.
    if (emitPatW != 1.0)
        for (int i = 0; i < C; ++i) beta[i] = (Real)((double)beta[i] * emitPatW);
    secAlive = (C > 1);
    int nUp = secAlive ? C : 1;
    for (int i = 0; i < nUp; ++i) eEmitted += (double)beta[i];

    // Direct emitter->camera connection (area/quad emitters only; spot/env/sun have no
    // direct term). No-op for mode C (splat helpers skip it) and the mode-M deposit pass (nCam==0).
    if (em.shape != 2 && em.shape != 3 && em.shape != 6) {
        Real rhoOne[hero::kHeroMax]; for (int i = 0; i < nUp; ++i) rhoOne[i] = (Real)1;
        splatSurfaceAllHero(sc, cs, camMode, origin, emitN, emitN, emitN, lam, beta, rhoOne, nUp, rng);
        camSpecularSplatAllHero(sc, cs, camMode, origin, emitN, lam, beta, rhoOne, nUp, rng);
    }

    ro = origin + dir * RAY_EPS; rd = dir;
    return true;
}

// One hero bounce (called only while secAlive, so nUp == C). Handles the model-C catch,
// escape/sensor bookkeeping, and the diffuse / diffuse-transmit lobes with per-λ deposit +
// splat. EVERY Russian roulette here survives on the MAX over live λ (q = max_i c_i) and
// reweights survivors by c_i/q <= 1, so no secondary is ever amplified; at nUp == 1 that is
// exactly the scalar analog RR with a *= 1.0 reweight. Mirror/Filter/Glossy are delta lobes
// but ACHROMATIC (λ-independent outgoing direction), so the bundle keeps riding through them.
// At the six dispersive / wavelength-switching materials it DE-HEROS (beta[0] *= C, secAlive =
// false) and delegates that same hit to the shared scalar interactSpecular — from then on
// the caller runs the ordinary single-λ shadeStep. No fog/GRIN here (gated out upstream).
__device__ static int shadeStepHero(const DScene& sc, const DCamSet& cs, int camMode,
        int diffraction, int C, const DHit& h, DVec3& ro, DVec3& rd, Real* lam, Real* beta,
        bool& secAlive, DRng& rng, double& eAbsorbed, double& eSensor, double& eEscaped,
        DMediumStack& stk) {
    const int nUp = C;
    Real dEvent = h.valid ? h.t : BIG;

    // Model C aperture catch: deposit every live wavelength that threads the pupil.
    if (camMode == CAM_C) {
        int px, py;
        if (cs.cams[0].catchPhoton(ro, rd, dEvent, px, py)) {
            Real cCell = (Real)1 / (Real)(cs.cams[0].pixelPlaneArea() * cs.cams[0].filmDist * cs.cams[0].filmDist);
            for (int i = 0; i < nUp; ++i) {
                filmAdd(cs.films[0], cs.hits[0], cs.cams[0].resX, px, py, lam[i], beta[i] * cCell);
                eSensor += (double)beta[i];
            }
            return WF_TERMINATE;
        }
    }
    // (No Beer-Lambert while secAlive: entering a dielectric de-heros, so the stack is empty
    // here and topMat() would be -1.)
    if (!h.valid) { for (int i = 0; i < nUp; ++i) eEscaped += (double)beta[i]; return WF_TERMINATE; }
    if (h.sensorId >= 0) { for (int i = 0; i < nUp; ++i) eSensor += (double)beta[i]; return WF_TERMINATE; }

    const DMaterial* mptr = &sc.mats[h.matId];
    int matIndex = h.matId;
    if (mptr->type == D_MIX) {
        int child = dMixResolveChild(sc, *mptr, h, rng.uniform());
        if (child < 0) { for (int i = 0; i < nUp; ++i) eAbsorbed += (double)beta[i]; return WF_TERMINATE; }
        mptr = &sc.mats[child]; matIndex = child;
    }
    const DMaterial& m = *mptr;

    if (m.type == D_DIFFUSETRANSMIT) {
        Real rhoR[hero::kHeroMax], rhoT[hero::kHeroMax];
        for (int i = 0; i < nUp; ++i) {
            Real rr = clamp01(dDiffuseRho(sc, m, h, lam[i]));
            Real rt = clamp01(dTransmitSlot(sc, m, h, lam[i]));
            Real s = rr + rt; if (s > (Real)1) { rr /= s; rt /= s; }   // per-λ energy guard
            rhoR[i] = rr; rhoT[i] = rt;
        }
        DVec3 nb = h.n * (Real)(-1);
        DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
        DVec3 wiPrev = -rd;
        for (int i = 0; i < nUp; ++i) depositPhoton(cs, h.p, h.n, beta[i], lam[i]);
        if (camMode == CAM_A || camMode == CAM_B) {
            splatSurfaceAllHero(sc, cs, camMode, h.p, h.n, ngo, wiPrev, lam, beta, rhoR, nUp, rng);
            splatSurfaceAllHero(sc, cs, camMode, h.p, nb, ngo * (Real)(-1), wiPrev, lam, beta, rhoT, nUp, rng);
            camSpecularSplatAllHero(sc, cs, camMode, h.p, h.n, lam, beta, rhoR, nUp, rng);
            camSpecularSplatAllHero(sc, cs, camMode, h.p, nb, lam, beta, rhoT, nUp, rng);
        }
        // Lobe pick + RR over the whole bundle (see the diffuse tail): the reflect/transmit
        // probabilities are the per-lobe MAX over live λ, so no secondary is ever amplified.
        // The maxima can sum past 1 (each λ alone is guarded), in which case both shrink
        // proportionally — guarded by nUp > 1 so the scalar path can never take that branch.
        // At nUp == 1 the two maxima are rhoR[0]/rhoT[0] and every reweight is *= 1.0.
        Real qR = rhoR[0], qT = rhoT[0];
        for (int i = 1; i < nUp; ++i) {
            if (rhoR[i] > qR) qR = rhoR[i];
            if (rhoT[i] > qT) qT = rhoT[i];
        }
        Real sumHero = qR + qT;
        if (nUp > 1 && sumHero > (Real)1) { qR /= sumHero; qT /= sumHero; sumHero = qR + qT; }
        Real uu = rng.uniform();
        if (uu < qR) {
            // The reweight is deterministic absorption — book it, or the energy ledger loses
            // the difference (sum/emitted would drop well below 1).
            for (int i = 0; i < nUp; ++i) {
                Real w = rhoR[i] / qR;
                eAbsorbed += (double)beta[i] * (double)((Real)1 - w);
                beta[i] *= w;
            }
            DVec3 wo = cosineHemisphere(h.n, rng);
            Real corr = dShadingAdjointCorr(wiPrev, wo, h.n, ngo);
            for (int i = 0; i < nUp; ++i) beta[i] *= corr;
            ro = h.p + h.n * RAY_EPS; rd = wo; return WF_CONTINUE;
        } else if (uu < sumHero) {
            for (int i = 0; i < nUp; ++i) {
                Real w = rhoT[i] / qT;
                eAbsorbed += (double)beta[i] * (double)((Real)1 - w);
                beta[i] *= w;
            }
            DVec3 wo = cosineHemisphere(nb, rng);
            Real corr = dShadingAdjointCorr(wiPrev, wo, h.n, ngo);
            for (int i = 0; i < nUp; ++i) beta[i] *= corr;
            ro = h.p + nb * RAY_EPS; rd = wo; return WF_CONTINUE;
        }
        for (int i = 0; i < nUp; ++i) eAbsorbed += (double)beta[i];
        return WF_TERMINATE;
    }

    if (m.type == D_MIRROR || m.type == D_FILTER || m.type == D_GLOSSY) {
        // ACHROMATIC delta lobes (device twin of render.h's Mirror/Filter/Glossy hero case):
        // specular — so no camera connect, exactly like the scalar path — but the outgoing
        // DIRECTION does not depend on λ, so the bundle keeps riding and only the per-λ
        // coefficient differs. The scalar lobe survives by ANALOG Russian roulette on its
        // coefficient; rolling that coin on the hero alone would kill live secondaries
        // whenever c_hero == 0 (a Wratten gel is 0 over most of the spectrum) AND amplify by
        // c_i/c_hero, so the survival probability is the MAX over live λ and survivors
        // reweight by c_i/q <= 1.
        Real c[hero::kHeroMax];
        Real q = (Real)0;
        for (int i = 0; i < nUp; ++i) {
            c[i] = (m.type == D_FILTER) ? clamp01(dTransmitSlot(sc, m, h, lam[i]))
                                        : clamp01(dReflectSlot(sc, m, h, lam[i]));
            if (c[i] > q) q = c[i];
        }
        if (rng.uniform() >= q) { for (int i = 0; i < nUp; ++i) eAbsorbed += (double)beta[i]; return WF_TERMINATE; }
        for (int i = 0; i < nUp; ++i) {                     // bounded reweight
            Real w = c[i] / q;
            eAbsorbed += (double)beta[i] * (double)((Real)1 - w);   // deterministic absorption
            beta[i] *= w;
        }
        if (m.type == D_MIRROR) {
            DVec3 o = reflectv(rd, h.n); ro = h.p + h.n * RAY_EPS; rd = o;
        } else if (m.type == D_FILTER) {
            ro = h.p + rd * RAY_EPS;   // straight through, direction unchanged
        } else {
            DVec3 o = sampleGlossy(reflectv(rd, h.n), dMatRoughness(sc, m, h), rng);
            if (dot(o, h.n) <= 0) { for (int i = 0; i < nUp; ++i) eAbsorbed += (double)beta[i]; return WF_TERMINATE; }
            ro = h.p + h.n * RAY_EPS; rd = o;
        }
        return WF_CONTINUE;
    }

    if (m.type == D_DIELECTRIC || m.type == D_THINFILM || m.type == D_MULTILAYER ||
        m.type == D_GRATING || m.type == D_HALFMIRROR || m.type == D_FLUORESCENT ||
        m.type == D_HAIR) {
        // Dispersive / wavelength-switching: terminate secondaries, boost the hero ×C, then
        // run the shared scalar interaction on the (now single-λ) hero channel. Hair rides
        // here because its sigma_a (and the reflectance→absorption inversion) is per-λ, so
        // one fiber interaction cannot be shared across C wavelengths — matching the CPU
        // tracePhotonHero, which de-heroes onto the scalar MatType::Hair path.
        beta[0] *= (Real)C; secAlive = false;
        return interactSpecular(sc, cs, camMode, diffraction, m, matIndex, h,
                                ro, rd, beta[0], lam[0], rng, eAbsorbed, stk);
    }

    // Diffuse (texture-sampled reflectance when the material binds a texture).
    Real rho[hero::kHeroMax];
    for (int i = 0; i < nUp; ++i) rho[i] = clamp01(dDiffuseRho(sc, m, h, lam[i]));
    DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
    DVec3 wiPrev = -rd;
    for (int i = 0; i < nUp; ++i) depositPhoton(cs, h.p, h.n, beta[i], lam[i]);
    if (camMode == CAM_A || camMode == CAM_B) {
        splatSurfaceAllHero(sc, cs, camMode, h.p, h.n, ngo, wiPrev, lam, beta, rho, nUp, rng);
        camSpecularSplatAllHero(sc, cs, camMode, h.p, h.n, lam, beta, rho, nUp, rng);
    }
    // Continuation RR over the WHOLE bundle: the survival probability is max_i rho_i, not the
    // hero's own albedo, and every live λ reweights by rho_i/q <= 1. Rolling the coin on the
    // hero alone (beta[i] *= rho_i/rho_0) amplifies a secondary by up to rho_max/rho_hero — on
    // a saturated wall (redWall spans 0.05..0.75) a 15x weight spike per bounce, which cancels
    // the whole stratification win. At nUp == 1, q == rho[0] and beta[0] *= 1.0.
    Real q = rho[0];
    for (int i = 1; i < nUp; ++i) if (rho[i] > q) q = rho[i];
    if (rng.uniform() >= q) { for (int i = 0; i < nUp; ++i) eAbsorbed += (double)beta[i]; return WF_TERMINATE; }
    for (int i = 0; i < nUp; ++i) {                              // bounded reweight
        Real w = rho[i] / q;
        eAbsorbed += (double)beta[i] * (double)((Real)1 - w);    // deterministic absorption
        beta[i] *= w;
    }
    DVec3 wo = cosineHemisphere(h.n, rng);
    Real corr = dShadingAdjointCorr(wiPrev, wo, h.n, ngo);
    for (int i = 0; i < nUp; ++i) beta[i] *= corr;
    ro = h.p + h.n * RAY_EPS; rd = wo; return WF_CONTINUE;
}

// Full hero photon: emit, then bounce until termination. While the secondaries are alive
// each bounce runs shadeStepHero; once a dispersive interface de-heros the path, it falls
// through to the ordinary single-λ shadeStep on beta[0]/lam[0].
__device__ static void traceHeroPhoton(const DScene& sc, const DCamSet& cs, int camMode,
        int diffraction, int maxBounce, int C, DRng& rng,
        double& eEmitted, double& eAbsorbed, double& eSensor, double& eEscaped, double& eResidual) {
    Real lam[hero::kHeroMax], beta[hero::kHeroMax];
    bool secAlive = false;
    DVec3 ro, rd;
    if (!genPhotonHero(sc, cs, camMode, C, rng, ro, rd, lam, beta, secAlive, eEmitted)) return;
    DMediumStack stk; stk.clear();
    bool done = false;
    for (int bounce = 0; bounce < maxBounce && !done; ++bounce) {
        if (sc.hasGrin) dGrinMarch(sc, ro, rd);   // gate excludes GRIN; kept for symmetry
        DHit h = closestHit(sc, ro, rd);
        int r;
        if (secAlive)
            r = shadeStepHero(sc, cs, camMode, diffraction, C, h, ro, rd, lam, beta, secAlive, rng,
                              eAbsorbed, eSensor, eEscaped, stk);
        else
            r = shadeStep(sc, cs, camMode, diffraction, h, ro, rd, beta[0], lam[0], rng,
                          eAbsorbed, eSensor, eEscaped, stk);
        if (r == WF_TERMINATE) done = true;
    }
    if (!done) {
        int n = secAlive ? C : 1;
        for (int i = 0; i < n; ++i) eResidual += (double)beta[i];
    }
}

__global__ void kTrace(DScene sc, DCamSet cs, double* energy,
                       long long N, int diffraction, unsigned long long seedBase, int maxBounce,
                       int camMode, int heroC) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    DRng rng; rng.seed((unsigned long long)(g * 2 + 1), seedBase ^ (unsigned long long)g);

    double eEmitted = 0, eAbsorbed = 0, eSensor = 0, eEscaped = 0, eResidual = 0;

    for (long long i = g; i < N; i += G) {
        if (heroC > 1) {
            // Hero-wavelength path: one BVH walk carries C stratified wavelengths, halving
            // chromatic noise. De-heros to the single-λ shadeStep at a dispersive interface.
            traceHeroPhoton(sc, cs, camMode, diffraction, maxBounce, heroC, rng,
                            eEmitted, eAbsorbed, eSensor, eEscaped, eResidual);
            continue;
        }
        DVec3 ro, rd; Real beta, lambda;
        if (!genPhoton(sc, cs, camMode, rng, ro, rd, beta, lambda, eEmitted)) continue;
        bool done = false;
        DMediumStack stk; stk.clear();   // nested-dielectric medium stack (empty = vacuum)
        // PHOTON-BEAMS: an INDEPENDENT per-photon stream used ONLY to resample each camera's
        // in-scatter point (seeded from the main stream so it is unique per photon/thread).
        // Only drawn from inside shadeStep's doBeam branch, so non-beam renders are unchanged.
        DRng crng;
        if (cs.beamGather) crng.seed(((unsigned long long)rng.next() << 32) ^ rng.next(),
                                     ((unsigned long long)rng.next() << 32) ^ rng.next());
        for (int bounce = 0; bounce < maxBounce && !done; ++bounce) {
            if (sc.hasGrin) dGrinMarch(sc, ro, rd);   // bend through any GRIN region first
            DHit h = closestHit(sc, ro, rd);
            if (shadeStep(sc, cs, camMode, diffraction, h, ro, rd, beta, lambda, rng,
                          eAbsorbed, eSensor, eEscaped, stk, cs.beamGather ? &crng : nullptr) == WF_TERMINATE) done = true;
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
    // Nested-dielectric medium stack per slot (SoA): stkMat/stkPri are CAP-strided
    // (slot*CAP + i), stkN is the entry count. Empty = vacuum. Replaces the old single
    // `interior` material index so overlapping dielectrics resolve by priority.
    int*   stkMat;
    int*   stkPri;
    int*   stkN;
    DHit*  hit;      // extend-stage intersection, consumed by shade
};

// Claim photon budget and emit fresh photons into `slot` until one is successfully
// launched or the N-photon budget is exhausted. Mirrors the megakernel's per-iteration
// genPhoton: a zero-pdf wavelength draw is skipped but still consumes its budget index,
// so the total genPhoton count is exactly N across the whole render. Returns true and
// fills the slot (alive=1, bounce=0) on success; false when the budget is spent (the
// caller marks the slot dead). Emitted energy accrues into energy[0].
__device__ static bool wfSpawn(const DScene& sc, const DCamSet& cs,
                               double* energy, int camMode, long long N,
                               unsigned long long* dispatched, WFState st, int slot, DRng& rng) {
    for (;;) {
        unsigned long long idx = atomicAdd(dispatched, 1ULL);
        if (idx >= (unsigned long long)N) return false;
        DVec3 ro, rd; Real beta, lambda; double eEm = 0;
        if (genPhoton(sc, cs, camMode, rng, ro, rd, beta, lambda, eEm)) {
            st.ro[slot] = ro; st.rd[slot] = rd;
            st.beta[slot] = beta; st.lambda[slot] = lambda;
            st.bounce[slot] = 0; st.alive[slot] = 1; st.stkN[slot] = 0;
            atomicAdd(&energy[0], eEm);
            return true;
        }
        // zero-pdf photon: its budget index is consumed, loop and try the next
    }
}

// Seed each slot's RNG and fill it with a first photon.
__global__ void kWfInit(DScene sc, DCamSet cs, double* energy,
                        WFState st, long long N, int W, unsigned long long* dispatched,
                        int* liveCount, unsigned long long seedBase, int camMode) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= W) return;
    DRng rng; rng.seed((unsigned long long)(slot * 2 + 1), seedBase ^ (unsigned long long)slot);
    bool live = wfSpawn(sc, cs, energy, camMode, N, dispatched, st, slot, rng);
    st.rng[slot] = rng;
    if (live) atomicAdd(liveCount, 1);
    else st.alive[slot] = 0;
}

// Extend: one closestHit per live slot. For GRIN scenes, bend the slot's ray through any
// gradient-index region first (writing the bent ro/rd back so kWfShade sees them) — exactly
// as the megakernel marches before closestHit. Non-GRIN scenes skip it (bit-identical).
__global__ void kWfExtend(DScene sc, WFState st, int W) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= W || !st.alive[slot]) return;
    if (sc.hasGrin) {
        DVec3 ro = st.ro[slot], rd = st.rd[slot];
        dGrinMarch(sc, ro, rd);
        st.ro[slot] = ro; st.rd[slot] = rd;
    }
    st.hit[slot] = closestHit(sc, st.ro[slot], st.rd[slot]);
}

// Shade: advance each live slot by one bounce; regenerate on termination / bounce cap.
__global__ void kWfShade(DScene sc, DCamSet cs, double* energy,
                         WFState st, int W, long long N, int diffraction, int maxBounce,
                         unsigned long long* dispatched, int* liveCount, int camMode) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= W || !st.alive[slot]) return;
    DRng rng = st.rng[slot];
    DVec3 ro = st.ro[slot], rd = st.rd[slot];
    Real beta = st.beta[slot], lambda = st.lambda[slot];
    DHit h = st.hit[slot];
    // Load the per-slot medium stack from SoA into a local (CAP-strided).
    DMediumStack stk; stk.n = st.stkN[slot];
    for (int i = 0; i < stk.n; ++i) {
        stk.matIdx[i] = st.stkMat[slot * DMediumStack::CAP + i];
        stk.pri[i]    = st.stkPri[slot * DMediumStack::CAP + i];
    }
    double eAbs = 0, eSen = 0, eEsc = 0;
    int res = shadeStep(sc, cs, camMode, diffraction, h, ro, rd, beta, lambda, rng,
                        eAbs, eSen, eEsc, stk);
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
        // Store the medium stack back to SoA (carry to the next segment).
        st.stkN[slot] = stk.n;
        for (int i = 0; i < stk.n; ++i) {
            st.stkMat[slot * DMediumStack::CAP + i] = stk.matIdx[i];
            st.stkPri[slot * DMediumStack::CAP + i] = stk.pri[i];
        }
        return;
    }
    // Path finished: regenerate this slot from the remaining budget (compaction).
    bool live = wfSpawn(sc, cs, energy, camMode, N, dispatched, st, slot, rng);
    st.rng[slot] = rng;
    if (!live) { st.alive[slot] = 0; atomicSub(liveCount, 1); }
}

// ============================ bidirectional path tracing (mode D) ============
// Device port of bdpt.h (Veach / PBRT-v3). One thread renders one (pixel,sample):
// it builds a camera subpath and a light subpath at a single shared wavelength,
// then MIS-connects every vertex pair (balance heuristic). Geometry stays in Real;
// all pdf/MIS arithmetic runs in double (ddot) to keep the balance-heuristic ratios
// stable, matching the CPU reference to within Monte-Carlo noise. Emissive surfaces,
// area/sphere/tube/mesh lights AND the two DELTA emitters (`light spot`, `light sun`);
// env / collimated scenes are outside BDPT's transport scope on both backends and are
// refused by the mode-D guard. See bdpt.h for the derivation of every quantity below,
// and dIsDeltaEmitter / dGenLightSubpath / dMisWeight for the delta-light treatment.

// Per-thread vertex-stack bound. `kBdptT` is templated on it (MAXD) because the two
// subpath arrays are THREAD-LOCAL: doubling the depth doubles ~100 B/vertex of local
// memory, so the default launch must not pay for a depth it will not use. BDPT_MAXDEPTH
// is the DEFAULT instantiation; BDPT_DEEPDEPTH is the opt-in one that `-max-bounce N`
// selects for N > 8. A specular cavity (mirror-lined sphere, kaleidoscope, deeply nested
// dielectrics) needs the deep variant: at 8 edges its recursive images truncate to black.
#define BDPT_MAXDEPTH  8
#define BDPT_DEEPDEPTH 64
#define BDPT_MAXV     (BDPT_MAXDEPTH + 3)   // path[0] endpoint + up to MAXDEPTH surfaces + slack
#define BDPT_MAXV_OF(d) ((d) + 3)
enum { BV_CAMERA = 0, BV_LIGHT = 1, BV_SURFACE = 2, BV_MEDIUM = 3 };

// A path vertex. Mirrors bdpt.h Vertex, but stores INDICES (matId into sc.mats,
// lightIdx into sc.emitters) instead of pointers, and drops the Hit field (the GPU
// rejects textured scenes, so albedo needs no surface-local (u,v)). pdfFwd/pdfRev/
// beta stay double for MIS stability; geometry (p/ns/ng) is Real.
struct DVertex {
    int   type;                 // BV_CAMERA / BV_LIGHT / BV_SURFACE / BV_MEDIUM
    DVec3 p, ns, ng;            // position, shading normal, geometric normal
    double beta;                // throughput carried to this vertex
    double pdfFwd, pdfRev;      // area-measure densities (0 for delta vertices)
    int   delta;                // 1 => specular (skipped in connections/MIS)
    int   matId;                // sc.mats index (-1 for camera)
    int   lightIdx;             // sc.emitters index if emissive, else -1
    double mediumG;             // HG asymmetry g at a BV_MEDIUM vertex
    int   mediumId;             // sc.media index at a BV_MEDIUM vertex (-1 otherwise)
    Real  u, v;                 // interpolated surface texcoords (per-hit BSDF eval, M9)
    // Mean curvature at this vertex (O3). Carried for the same reason u/v are: the
    // connection BSDF is re-evaluated here from a reconstructed DHit, and a material
    // whose roughness/reflectance is driven by `curv` would otherwise read a DIFFERENT
    // value on the GPU than on the CPU — where bdpt.h's Vertex keeps the whole Hit and
    // so gets curvature for free. Matching it here is what keeps GPU BDPT consistent
    // with every other integrator on a curvature-driven material.
    Real  curv;
    // This vertex's `emit pattern:` factor (device twin of bdpt.h Vertex::emitPatW),
    // cached because dVertexLe is called from several MIS strategies and has no DHit to
    // re-evaluate the pattern from. 1 for a non-emissive vertex or an unpatterned light,
    // so every existing scene multiplies by exactly one. A BV_LIGHT vertex gets it from
    // dEmitterSamplePointPat (the sampled point); a BV_SURFACE vertex from dEmitPatMul at
    // the hit — the two agree pointwise, which is what keeps s=0 / s=1 MIS unbiased.
    Real  emitPatW;
    int   nUp;                  // live hero wavelengths here (1 = single-λ walk / de-hero'd)
};

// Number of SECONDARY wavelength slots a hero bundle can carry (the hero itself rides in
// the scalar `beta`). The per-vertex secondary throughputs live in a PARALLEL array
// (`pathSec`, stride `secStride`) rather than inside DVertex, so the scalar kernel — which
// declares that array at size 1 — pays no local-memory cost for a feature it never uses
// (DVertex is already ~100B and there are 2*BDPT_MAXV of them in every thread's frame).
#define BDPT_NSEC (hero::kHeroMax - 1)

// Where, and with what weight, a subpath's LAST ray left the scene (device twin of bdpt.h
// Escape). BDPT has no environment, but a `light sun` is an infinitely distant delta-
// DIRECTION emitter that an eye ray can still look straight into. Because a delta light is
// excluded from the s=0 strategy (see dMisWeight), nothing else in BDPT can deliver the
// sun's own disc — nor any mirror/water glint of it, which NEE cannot produce either (a
// specular vertex is not connectible). Capturing the escaping ray lets kBdptT add that one
// strategy back with MIS weight exactly 1, since no other strategy can generate the same
// path (every vertex on it is delta, so neither NEE nor a light-subpath connection reaches
// it). `betaSec` is 3 doubles regardless of the kernel's NS template arg — it is ONE
// per-thread struct, not per-vertex state, so the scalar kernel's footprint is unaffected.
struct DEscape {
    int    escaped;                      // 0 = the subpath never left the scene
    DVec3  dir;                          // unit direction the ray left along
    double beta;                         // hero throughput carried out of the scene
    double betaSec[BDPT_NSEC];
    int    nUp;
};

// Hero-wavelength bundle (device twin of bdpt.h HeroBundle). lam[0] is the HERO: it alone
// drives geometry, every sampling decision, every pdf and therefore every MIS weight — so a
// connection's MIS weight is shared by the whole bundle and is computed once. lam[1..C-1]
// are stratified secondaries riding the same BVH walk, carrying only their own throughput.
// No ×C de-hero boost is folded into any throughput: the two subpaths de-hero
// independently, so the normalisation is applied once at splat time as
// 1/min(nUp_light, nUp_eye) — see bdpt.h Vertex::nUp for the derivation.
struct DHeroBundle {
    Real   lam[hero::kHeroMax];
    double invPdf[hero::kHeroMax];
    int    C;
};

// Reconstruct a minimal DHit at a surface vertex so the per-hit material helpers
// (dDiffuseRho / dReflectSlot / dMatRoughness / dRecordReflect / dMixResolveChild) can
// evaluate textured / patterned / record-driven params on the connection BSDF exactly as
// the sampler did — the enabler for per-hit BSDFs in GPU BDPT (M9). Only p/n/ng/u/v are
// read by those helpers (t/valid/sensorId are unused there).
__device__ static inline DHit dVertHit(const DVertex& vt) {
    DHit h;
    h.t = (Real)0; h.valid = true;
    h.p = vt.p; h.n = vt.ns; h.ng = vt.ng;
    h.matId = vt.matId; h.sensorId = -1;
    h.u = vt.u; h.v = vt.v; h.curv = vt.curv;
    return h;
}

__device__ static inline double ddot(const DVec3& a, const DVec3& b) {
    return (double)a.x * b.x + (double)a.y * b.y + (double)a.z * b.z;
}
// Medium (volume) vertices carry no surface, so onSurface() is false — ConvertDensity
// then omits the cosine Jacobian, giving the correct cosine-free volume area density.
__device__ static inline bool dOnSurface(const DVertex& v) {
    return v.type == BV_SURFACE || v.type == BV_LIGHT;
}
__device__ static inline bool dConnectibleType(int tp) {
    return tp == D_DIFFUSE || tp == D_GLOSSY || tp == D_FLUORESCENT || tp == D_DIFFUSETRANSMIT;
}
__device__ static bool dVertConnectible(const DScene& sc, const DVertex& v) {
    if (v.type == BV_CAMERA) return true;
    if (v.type == BV_MEDIUM) return true;   // volume in-scatter always connects
    if (v.type == BV_LIGHT)  return v.lightIdx >= 0 && sc.emitters[v.lightIdx].collimated == 0;
    if (v.delta) return false;
    return dConnectibleType(sc.mats[v.matId].type);
}
// Medium phase as the "BSDF": propagation INTO the vertex is -wo, scattered dir is wi
// (both point away from v), so cosTheta = -dot(wo,wi). The phase is its own pdf, so
// dPhaseF == dPhasePdf. A rainbow medium uses its tabulated Airy phase (wavelength-
// dependent), so both take the scene + lambda to look up sc.media[v.mediumId].
// mediumScatterF = albedo * phase is the CONNECTION response (albedo corrects the
// sigma_t-rate collision to the sigma_s scatter rate; it enters the throughput once,
// never the MIS density). Mirrors bdpt.h phaseF / mediumScatterF.
__device__ static inline double dPhaseF(const DScene& sc, const DVertex& v,
                                        const DVec3& wo, const DVec3& wi, Real lambda) {
    return (double)dMedPhase(sc.media[v.mediumId], (Real)(-ddot(wo, wi)), lambda);
}
__device__ static inline double dPhasePdf(const DScene& sc, const DVertex& v,
                                          const DVec3& wo, const DVec3& wi, Real lambda) {
    return (double)dMedPhase(sc.media[v.mediumId], (Real)(-ddot(wo, wi)), lambda);
}
__device__ static inline double dMediumScatterF(const DScene& sc, const DVertex& v,
                                                const DVec3& wo, const DVec3& wi, Real lambda) {
    return (double)medAlbedo(sc.media[v.mediumId], lambda) * dPhaseF(sc, v, wo, wi, lambda);
}
// Device twin of bdpt.h Vertex::isLightVertex. The `matIsLight` half is NOT redundant with
// `lightIdx >= 0`: only tessellated geometry registers an Emitter, so an emissive `quad`,
// `isosurface` or CSG solid has an emissive MATERIAL but no emitter at all, and testing
// lightIdx alone silently dropped its self-emission from the s=0 strategy — a GPU-only
// blackout of every non-mesh glowing surface, where the CPU (which has always tested
// mat->isLight) rendered it. See dVertexLe for the radiance half of the same fix.
__device__ static inline bool dIsLightVertex(const DScene& sc, const DVertex& v) {
    if (v.type == BV_LIGHT) return true;
    if (v.type != BV_SURFACE) return false;
    return v.lightIdx >= 0 || (v.matId >= 0 && sc.mats[v.matId].matIsLight);
}
// PBRT's Vertex::IsDeltaLight (device twin of bdpt.h Vertex::isDeltaLight): this vertex IS
// a light whose emission carries a Dirac delta, so the "eye path lands on the light" (s=0)
// strategy cannot produce it. dMisWeight uses it to drop that strategy from the sum.
__device__ static inline bool dIsDeltaLightVertex(const DScene& sc, const DVertex& v) {
    return v.type == BV_LIGHT && v.lightIdx >= 0 && dIsDeltaEmitter(sc.emitters[v.lightIdx]);
}
__device__ static inline double dGlossyExp(double roughness) {
    double rr = roughness < 1e-3 ? 1e-3 : roughness;
    double e = 2.0 / (rr * rr) - 2.0;
    return e < 0 ? 0 : e;
}
// A two-sided (transmissive) connectible material scatters into BOTH hemispheres, so a
// connection edge on the side opposite the shading normal is legal (transmit lobe). Only
// DiffuseTransmit qualifies (device twin of bdpt.h isTwoSidedMat).
__device__ static inline bool dTwoSidedType(int tp) { return tp == D_DIFFUSETRANSMIT; }
// Clamped reflect/transmit albedos of a DiffuseTransmit vertex (energy guard shared by
// dBsdfF / dBsdfPdf / the scatter switch so MIS densities stay consistent; twin of
// bdpt.h diffuseTransmitAlbedos). rhoR is the per-hit front lobe (texture-aware).
__device__ static inline void dDiffuseTransmitAlbedos(const DScene& sc, const DMaterial& m,
                                                      const DHit& h, Real lambda,
                                                      double& rhoR, double& rhoT) {
    rhoR = clamp01(dDiffuseRho(sc, m, h, lambda));
    rhoT = clamp01(dTransmitSlot(sc, m, h, lambda));
    double sum = rhoR + rhoT;
    if (sum > 1.0) { rhoR /= sum; rhoT /= sum; }
}

// BSDF value f(wo->wi) at a surface vertex (double, for the connection radiance L).
// Takes the full DVertex so textured/patterned/record-driven albedo & roughness are
// evaluated per-hit (M9) — the reconstructed DHit feeds dDiffuseRho / dReflectSlot /
// dMatRoughness exactly as the sampler used them, so MIS densities stay consistent.
__device__ static double dBsdfF(const DScene& sc, const DVertex& vt,
                                const DVec3& wo, const DVec3& wi, Real lambda) {
    const DMaterial& m = sc.mats[vt.matId];
    const DVec3& ns = vt.ns;
    double cosWi = ddot(wi, ns), cosWo = ddot(wo, ns);
    if (m.type == D_DIFFUSE || m.type == D_FLUORESCENT) {
        if (cosWi <= 0 || cosWo <= 0) return 0.0;
        DHit h = dVertHit(vt);
        double rho = clamp01(dDiffuseRho(sc, m, h, lambda));
        return rho / DPI;
    } else if (m.type == D_GLOSSY) {
        if (cosWi <= 0 || cosWo <= 0) return 0.0;
        DHit h = dVertHit(vt);
        double r = clamp01(dReflectSlot(sc, m, h, lambda));
        double e = dGlossyExp((double)dMatRoughness(sc, m, h));
        DVec3 mdir = reflectv(wo * (Real)-1, ns);
        double cosLobe = ddot(wi, mdir);
        if (cosLobe <= 0) return 0.0;
        double lobe = (e + 1.0) / (2.0 * DPI) * pow(cosLobe, e);
        return r * lobe / cosWi;
    } else if (m.type == D_DIFFUSETRANSMIT) {
        // Two-sided Lambertian: same-hemisphere pair -> reflect albedo, opposite -> transmit.
        DHit h = dVertHit(vt);
        double rhoR, rhoT; dDiffuseTransmitAlbedos(sc, m, h, lambda, rhoR, rhoT);
        bool sameSide = (cosWi * cosWo) > 0.0;
        return (sameSide ? rhoR : rhoT) / DPI;
    }
    return 0.0;
}
// Directional pdf (solid angle) of sampling wi at a surface vertex (double, for MIS).
// Per-hit roughness (glossy) is read from the vertex's texcoords so the density matches
// the sampling that used the same textured/patterned roughness (M9).
__device__ static double dBsdfPdf(const DScene& sc, const DVertex& vt,
                                  const DVec3& wo, const DVec3& wi, Real lambda) {
    const DMaterial& m = sc.mats[vt.matId];
    const DVec3& ns = vt.ns;
    double cosWi = ddot(wi, ns), cosWo = ddot(wo, ns);
    if (m.type == D_DIFFUSE || m.type == D_FLUORESCENT) {
        if (cosWi <= 0 || cosWo <= 0) return 0.0;
        return cosWi / DPI;
    } else if (m.type == D_GLOSSY) {
        if (cosWi <= 0 || cosWo <= 0) return 0.0;
        DHit h = dVertHit(vt);
        double e = dGlossyExp((double)dMatRoughness(sc, m, h));
        DVec3 mdir = reflectv(wo * (Real)-1, ns);
        double cosLobe = ddot(wi, mdir);
        if (cosLobe <= 0) return 0.0;
        return (e + 1.0) / (2.0 * DPI) * pow(cosLobe, e);
    } else if (m.type == D_DIFFUSETRANSMIT) {
        // The reflect lobe is chosen with prob rhoR/(rhoR+rhoT) and cosine-samples wo's
        // hemisphere; the transmit lobe (prob rhoT/(rhoR+rhoT)) cosine-samples the opposite
        // hemisphere. For a given wi only one lobe applies (by its sign vs wo).
        DHit h = dVertHit(vt);
        double rhoR, rhoT; dDiffuseTransmitAlbedos(sc, m, h, lambda, rhoR, rhoT);
        double tot = rhoR + rhoT;
        if (tot <= 0.0) return 0.0;
        bool sameSide = (cosWi * cosWo) > 0.0;
        double pSel = sameSide ? rhoR / tot : rhoT / tot;
        return pSel * fabs(cosWi) / DPI;
    }
    return 0.0;
}

// Camera importance (PBRT imagePlaneArea convention — see bdpt.h cameraWe/PdfDir).
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
// ---- FP32 MIS pdf probes ----------------------------------------------------
// These are used ONLY by dMisWeight. The MIS weight is a bounded ratio sum
// (1/(1+sumRi), sumRi >= 0): any weight partition of unity keeps the estimator
// unbiased, so last-ulp precision buys nothing -- and on GeForce parts FP64
// issues at 1/64 rate, which makes the per-connection reverse-density probes a
// direct FP64-pipe cost. FP32 is ample for these like-magnitude area-density
// ratios, and overflow is graceful (an inf ratio drives the weight to 0, same
// place the double path was headed). BSDF/phase backends keep their double
// internals; results fold to float at the boundary. The transport quantities
// themselves (beta/pdfFwd along the walk, connect radiance) stay double.
// Emission directional density at a light vertex toward `next` (area measure). Three
// emission models, mirroring the CPU vertexPdfLight (bdpt.h):
//   area/sphere/tube/mesh - cosine-weighted about the surface normal (cos/PI), converted
//                           to area measure by the usual cos(next)/dist^2.
//   spot (shape 2)        - uniform inside the OUTER cone (the smoothstep falloff is
//                           throughput, not density), pdfW = 1/(2 PI (1 - cosOuter)).
//   sun (shape 6)         - INFINITE: no finite emission point, so the density of `next`
//                           is the PLANAR density 1/(pi R^2) over the fictitious origin
//                           disc, with NO 1/dist^2 (the disc-point -> hit-point map is a
//                           shear along the beam whose Jacobian is exactly the cos(next)
//                           applied at the bottom).
__device__ static float dVertexPdfLightF(const DScene& sc, const DVertex& cur, const DVertex& next) {
    DVec3 w = next.p - cur.p;
    float d2 = dot(w, w);
    if (d2 == 0.f) return 0.f;
    float invD2 = 1.0f / d2;
    DVec3 wn = w * (Real)sqrtf(invD2);
    // Only a BV_LIGHT endpoint can be a delta emitter: spot/sun carry no geometry, so an
    // emissive SURFACE vertex — whether or not it registered an emitter — always takes the
    // cosine branch, which is exactly what an unregistered glowing quad/isosurface wants.
    const DEmitter* em = (cur.type == BV_LIGHT && cur.lightIdx >= 0) ? &sc.emitters[cur.lightIdx]
                                                                     : nullptr;
    float pdf;
    if (em && dIsInfiniteEmitter(*em)) {
        float R = (float)sc.sceneRadius;
        if (R <= 0.f) return 0.f;
        pdf = 1.0f / ((float)DPI * R * R);            // planar density over the sun disc
    } else if (em && em->shape == 2) {
        float solid = 2.0f * (float)DPI * (1.0f - (float)em->spotCosOuter);
        if (solid <= 0.f) return 0.f;
        pdf = (1.0f / solid) * invD2;                 // uniform in the outer cone
    } else {
        float cosLight = dot(cur.ng, wn);             // one-sided Lambertian emitter
        if (cosLight <= 0.f) return 0.f;
        pdf = (cosLight * (float)(1.0 / DPI)) * invD2;
    }
    if (dOnSurface(next)) pdf *= fabsf(dot(next.ns, wn));
    return pdf;
}
__device__ static float dVertexPdfF(const DScene& sc, const DCamera& cam,
                                    const DVertex* prev, const DVertex& cur, const DVertex& next,
                                    Real lambda) {
    if (cur.type == BV_LIGHT) return dVertexPdfLightF(sc, cur, next);
    DVec3 wn = next.p - cur.p;
    if (dot(wn, wn) == 0.f) return 0.f;
    wn = normalize(wn);
    float pdfW = 0.f;
    if (cur.type == BV_CAMERA) {
        DVec3 d = next.p - cam.eye;                       // dCameraPdfDir in FP32
        float len = sqrtf(dot(d, d));
        float cosCam = len > 0.f ? dot(d, cam.w) / len : 0.f;
        if (cosCam <= 0.f) return 0.f;
        pdfW = 1.0f / ((float)cam.imagePlaneArea() * cosCam * cosCam * cosCam);
    } else if (cur.type == BV_MEDIUM) {          // volume in-scatter: HG phase pdf
        if (!prev) return 0.f;
        DVec3 wp = prev->p - cur.p;
        if (dot(wp, wp) == 0.f) return 0.f;
        wp = normalize(wp);
        pdfW = (float)dPhasePdf(sc, cur, wp, wn, lambda);
    } else {
        if (!prev) return 0.f;
        DVec3 wp = prev->p - cur.p;
        if (dot(wp, wp) == 0.f) return 0.f;
        wp = normalize(wp);
        pdfW = (float)dBsdfPdf(sc, cur, wp, wn, lambda);
    }
    // dConvertDensity in FP32: solid angle -> area density at `next`.
    DVec3 wv = next.p - cur.p;
    float d2 = dot(wv, wv);
    if (d2 == 0.f) return 0.f;
    float invD2 = 1.0f / d2;
    if (dOnSurface(next)) pdfW *= fabsf(dot(next.ns, wv) * sqrtf(invD2));
    return pdfW * invD2;
}
// Positional density (area measure) of sampling this light vertex's ORIGIN via light
// sampling = P(choose this emitter) * (1/area). PBRT's PdfLightOrigin. A DELTA light
// (spot: a point; sun: infinitely distant) has no area density at all — PBRT returns 0 for
// exactly these. The 0 is consistent on BOTH sides of every MIS ratio (the light subpath
// stores the same 0 in path[0].pdfFwd) and dMisWeight's remap-0 turns both into 1, so the
// ratios stay finite and unbiased. Mirrors bdpt.h vertexPdfLightOrigin.
__device__ static float dVertexPdfLightOriginF(const DScene& sc, const DVertex& cur) {
    if (cur.lightIdx < 0) return 0.f;
    const DEmitter& em = sc.emitters[cur.lightIdx];
    if (sc.totalPower <= 0.0) return 0.f;
    if (dIsDeltaEmitter(em)) return 0.f;            // delta position / direction
    if (em.area <= 0.0) return 0.f;
    return (float)((em.power / sc.totalPower) / em.area);
}
// Emitted radiance (single wavelength) leaving a light vertex toward w.
//
// Prefer the registered emitter's BAKED spd — it may carry a `power`/`lumens` flux
// normalisation the raw material spectrum does not — and fall back to the material's own
// `emit` when this geometry registered no emitter. Only tessellated geometry registers one,
// so that fallback is what makes an emissive `quad` / `isosurface` / CSG solid visible in
// mode D at all; without it the GPU returned 0 here and the surface rendered black while the
// CPU (bdpt.h Vertex::Le, which reads mat->emit unconditionally) rendered it correctly. Same
// emitter-then-material precedence the backward tracer uses in bkRadiance.
__device__ static double dVertexLe(const DScene& sc, const DVertex& v, const DVec3& w,
                                   Real lambda, double invPdfLambda) {
    const double* eSpd = nullptr;
    if (v.lightIdx >= 0)                                  eSpd = sc.emitters[v.lightIdx].emitSpd;
    else if (v.matId >= 0 && sc.mats[v.matId].matIsLight) eSpd = sc.mats[v.matId].matEmit;
    if (!eSpd) return 0.0;
    if (ddot(v.ng, w) <= 0.0) return 0.0;
    return (double)specLookup(eSpd, lambda) * invPdfLambda * (double)v.emitPatW;
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
// Inverse-CDF core, split out so the hero-wavelength bundle can push its own stratified
// u values (base draw + C-1 wrapped strata) through the same sampler the scalar path
// uses — the device twin of EmissionSampler::sampleAt.
__device__ static Real dSampleSceneLambdaU(const DScene& sc, double u, double& pdf) {
    const double* cdf = sc.emitSamplerCdf;
    int lo = 0, hi = sc.emitSamplerN;
    while (lo + 1 < hi) { int m = (lo + hi) / 2; if (cdf[m] <= u) lo = m; else hi = m; }
    double c0 = cdf[lo], c1 = cdf[lo + 1];
    double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
    pdf = (c1 - c0) / sc.emitSamplerStep;
    return (Real)(DLMIN + (lo + frac) * sc.emitSamplerStep);
}
__device__ static Real dSampleSceneLambda(const DScene& sc, DRng& rng, double& pdf) {
    return dSampleSceneLambdaU(sc, (double)rng.uniform(), pdf);
}
// Sample a fluorophore's EXCITATION wavelength from its own absorb*illuminant CDF
// (device twin of Material::fluoInSampler.sampleAt, same inverse-CDF core as above).
// Returns lambda_in and sets pdf per nm; the caller's weight is exactly 1/pdf. Falls
// back to the scene illuminant when the material has no excitation table, so a dye
// that this illuminant cannot excite still terminates the branch (rhoFluo == 0).
__device__ static Real dSampleFluoInU(const DScene& sc, const DMaterial& m,
                                      double u, double& pdf) {
    if (m.fluoInCdfN <= 1) return dSampleSceneLambdaU(sc, u, pdf);
    const double* cdf = sc.fluoCdfAll + m.fluoInCdfOffset;
    int lo = 0, hi = m.fluoInCdfN - 1;
    while (lo + 1 < hi) { int mid = (lo + hi) / 2; if (cdf[mid] <= u) lo = mid; else hi = mid; }
    double c0 = cdf[lo], c1 = cdf[lo + 1];
    double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
    pdf = (c1 - c0) / m.fluoInCdfStep;
    return (Real)(DLMIN + (lo + frac) * m.fluoInCdfStep);
}
// invPdfLambda(lambda) = emitG / g(lambda), g(lambda) = sum_k geomWeight_k*SPD_k.
// In BDPT scope every emitter is an area/sphere light, so geomWeight = area*PI.
__device__ static double dInvPdfLambda(const DScene& sc, Real lambda) {
    double g = 0.0;
    for (int k = 0; k < sc.nEmitters; ++k) {
        const DEmitter& e = sc.emitters[k];
        // geomWeight (mirrors Scene::Emitter::geomWeight): area/sphere/cylinder = area*PI;
        // point-spot (shape 2) = spotOmega (falloff-weighted solid angle); env (shape 3) =
        // envGeom = 4*PI^2*R^2; distant sun (shape 6) = envGeom = Omega*PI*R^2 (the solar
        // cone times the scene's projected disc). Collimated beams are gated to the CPU.
        double gw = (e.shape == 2) ? e.spotOmega
                  : (e.shape == 3) ? (4.0 * DPI * DPI * sc.sceneRadius * sc.sceneRadius)
                  : (e.shape == 6) ? (e.spotOmega * DPI * sc.sceneRadius * sc.sceneRadius)
                                   : ((double)e.area * DPI);
        g += gw * (double)specLookup(e.emitSpd, lambda);
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
        rd = normalize(cam.w + cam.u * (Real)((sx + cam.frustumShiftX) * cam.tanHalfX) + cam.v * (Real)(sy * cam.tanHalfY));
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
// One emitter connection's SAMPLING + VISIBILITY, factored out of bkNeeLight so the
// scalar NEE and the hero-wavelength NEE below run the identical geometry off the
// identical rng stream. Everything here is wavelength-INDEPENDENT; the caller supplies
// rho/PI and the emitter SPD. The final product is deliberately left to the caller
// rather than fused into one weight here, so the scalar path's float rounding is
// unchanged by this refactor (device `Real` is fp32 by default — see FTRACE_GPU_FP32).
struct BkNeeGeom {
    DVec3 wi;        // unit direction surface -> sampled light point
    Real  dist;      // shadow-ray length
    Real  dist2;     // dist*dist (spot: inverse-square falloff)
    Real  cosSurf;   // cosine at the shading surface
    Real  stG;       // Chiang shadow-terminator gate (1 on flat geometry)
    Real  fall;      // spot cone falloff (spot emitters only)
    Real  G;         // area-measure geometry term cosSurf*cosLight/dist2 (non-spot)
    bool  spot;      // point-spot emitter (deterministic connect, draws no rng)
    bool  sun;       // distant-sun emitter (cone NEE in solid-angle measure)
    Real  wSun;      // sun only: the complete λ-independent weight cosSurf*Omega*stG
};
// Does this emitter consume its two sample coordinates? A collimated beam and a point-spot
// are deterministic connections and draw nothing; every area shape (and the sun's cone)
// draws two. Device twin of backward.h's emitterNeedsUV, and the reason (u1,u2) are
// PARAMETERS rather than drawn inside bkEmitterGeom: mode W has to feed them from the G x G
// lattice, while the stochastic callers must keep drawing at exactly the same point in the
// stream (a scene that merely *added* a spot light would otherwise reshuffle every other
// emitter's rng and change an unrelated image).
__device__ static bool dEmitterNeedsUV(const DEmitter& em) {
    return !em.collimated && em.shape != 2;
}
// Hair-fiber NEE response + shadow ray (device twins of the `hs` branches inside
// backward.h emitterGeom's `response` / `blocked` lambdas). A fiber has no hemisphere:
// the response is the full-sphere PI * f_hair * cosLong (the PI cancels neeLight's
// Lambertian 1/PI; callers pass rho = 1) with the shadow-terminator gate forced to 1,
// and the shadow ray starts from the BCSDF exit offset along wi ITSELF — a normal-offset
// origin would still sit inside the fiber's own thickness for far-side (TT) exits.
__device__ static bool bkHairResponse(const DHairShade& hsv, const DVec3& wi,
                                      Real& cosSurf, Real& stG) {
    cosSurf = (Real)DPI * dHairFCos(hsv, wi);
    stG = (Real)1;
    return cosSurf > (Real)0;
}
__device__ static bool bkHairBlocked(const DScene& sc, const DHit& h, const DHairShade& hsv,
                                     const DVec3& wi, Real d) {
    const Real off = dHairExitOffset(hsv, h.n, wi);
    const Real len = d - off - RAY_EPS;
    if (len <= (Real)0) return true;
    return occluded(sc, h.p + wi * off, wi, len);
}
// `hs` non-null = the shading vertex is a hair fiber: swap the cosine/terminator response
// and the surface-offset shadow ray for the hair versions above (all remaining geometry —
// point sampling, falloff, cosLight, 1/dist^2 — is shared). No rng draw moves either way.
__device__ static bool bkEmitterGeom(const DScene& sc, const DHit& h, const DVec3& ngo,
                                     const DEmitter& em, Real su1, Real su2, BkNeeGeom& g,
                                     const DHairShade* hs = nullptr) {
    if (em.shape == 2) {
        // Point spot (device twin of emitterGeom's spot branch): deterministic connect
        // to the light point, cone falloff toward the surface, no rng draw. Peak
        // intensity/SPD = 1; the falloff scales it toward the cone edge.
        DVec3 toL = em.origin - h.p;
        g.dist2 = dot(toL, toL);
        g.dist  = sqrt(g.dist2);
        g.wi = toL / g.dist;
        if (hs) {
            if (!bkHairResponse(*hs, g.wi, g.cosSurf, g.stG)) return false;
        } else {
            g.cosSurf = dot(h.n, g.wi);
            if (g.cosSurf <= (Real)0) return false;
            g.stG = dShadowTerminatorG(g.wi, h.n, ngo);
            if (g.stG <= (Real)0) return false;
        }
        g.fall = (Real)spotFalloff(dot(g.wi * (Real)(-1), em.beamDir), em.spotCosInner, em.spotCosOuter);
        if (g.fall <= (Real)0) return false;
        if (hs ? bkHairBlocked(sc, h, *hs, g.wi, g.dist)
               : occluded(sc, h.p + ngo * RAY_EPS, g.wi, g.dist - (Real)2 * RAY_EPS)) return false;
        g.G = (Real)0; g.spot = true; g.sun = false;
        return true;
    }
    if (em.shape == 6) {
        // Distant sun (device twin of emitterGeom's Sun branch): sample wi uniformly in
        // the solar cone about -beamDir (pdf 1/Omega) and shadow-ray it to the scene exit.
        // No finite light distance, so no 1/dist^2 and no cosLight: in solid-angle measure
        // the whole λ-independent weight is cosSurf/pdfW = cosSurf*Omega. Two sample
        // coordinates, matching the area path (see dEmitterNeedsUV).
        g.wi = dSunSampleCone(em, em.beamDir * (Real)(-1), (double)su1, (double)su2);
        if (hs) {
            if (!bkHairResponse(*hs, g.wi, g.cosSurf, g.stG)) return false;
        } else {
            g.cosSurf = dot(h.n, g.wi);
            if (g.cosSurf <= (Real)0) return false;
            g.stG = dShadowTerminatorG(g.wi, h.n, ngo);
            if (g.stG <= (Real)0) return false;
        }
        g.dist = (Real)((double)length(sc.sceneCenter - h.p) + sc.sceneRadius);
        g.dist2 = g.dist * g.dist;
        if (hs ? bkHairBlocked(sc, h, *hs, g.wi, g.dist)
               : occluded(sc, h.p + ngo * RAY_EPS, g.wi, g.dist)) return false;
        g.wSun = (Real)((double)g.cosSurf * em.spotOmega * (double)g.stG);
        g.G = (Real)0; g.fall = (Real)1; g.spot = false; g.sun = true;
        return true;
    }
    Real u1 = su1, u2 = su2;
    DVec3 y, nL;
    // Also returns this point's `emit pattern:` multiplier (1.0, and a bit-identical
    // draw, when there is none). Folding it into the λ-independent geometry weight G
    // below makes the scalar AND hero NEE pick it up at once, and matches the
    // emission-on-hit factor at the same surface point — which keeps the MIS pair
    // consistent (host twin: backward.h emitterGeom).
    double epat = dEmitterSamplePointPat(sc, em, (double)u1, (double)u2, y, nL);
    DVec3 toL = y - h.p;
    g.dist2 = dot(toL, toL);
    g.dist = sqrt(g.dist2);
    g.wi = toL / g.dist;
    if (hs) {
        if (!bkHairResponse(*hs, g.wi, g.cosSurf, g.stG)) return false;
    } else {
        g.cosSurf = dot(h.n, g.wi);
        if (g.cosSurf <= 0) return false;
        // Geometric-hemisphere softening (matches CPU backward.h neeLight): the light must lie
        // on the geometric front side too, ramped smoothly instead of a hard cutoff (Chiang
        // 2019). No-op when h.n==h.ng (flat tris / analytic spheres, stG==1); shadow ray offset
        // along the geometric normal so it clears the true surface.
        g.stG = dShadowTerminatorG(g.wi, h.n, ngo);
        if (g.stG <= (Real)0) return false;
    }
    Real cosLight = dot(nL, -g.wi);               // light is one-sided
    if (cosLight <= 0) return false;
    if (hs ? bkHairBlocked(sc, h, *hs, g.wi, g.dist)
           : occluded(sc, h.p + ngo * RAY_EPS, g.wi, g.dist - (Real)2 * RAY_EPS)) return false;
    g.G = g.cosSurf * cosLight / g.dist2;
    if (epat != 1.0) g.G = (Real)((double)g.G * epat);   // no-op without a pattern
    g.fall = (Real)1; g.spot = false; g.sun = false;
    return true;
}

// ---- many-lights selection on the device (device twin of BackwardRenderer::pickEmitters)
//
// Every NEE routine below used to loop `k = 0 .. nEmitters`, connecting a shadow ray to
// every light at every non-specular vertex. Unbiased, but O(N) per bounce for nothing once
// the lights are redundant — measured on THIS backend: same room, same total flux split
// across N ceiling panels, mode R 256 spp 256^2, 1 light 0.4 s vs 256 lights 75.9 s at an
// identical 6.25 % noise. The draw below replaces the loop bound: it returns the emitters
// to connect to plus 1/p(e) for each, turning `sum_e w_e` into `sum_selected w_e / p(e)`.
//
// kDMaxLightPick is deliberately half the CPU's 32: the LtSample array and ltSample's
// traversal stack are per-thread local-memory frames in a megakernel that is already
// register-starved, so every slot is paid by every thread. 16 connections per vertex is
// far past the point where more splitting improves the image.
#define kDMaxLightPick 16

struct DEmitterDraw {
    int  n;             // number of connections to make
    bool all;           // true: entries are emitters 0..n-1, each with weight 1 (the old loop)
    LtSample s[kDMaxLightPick];
    __device__ int emitter(int i) const { return all ? i : s[i].emitter; }
    __device__ double weight(int i) const {
        return all ? 1.0 : (s[i].pdf > 0.0 ? 1.0 / s[i].pdf : 0.0);
    }
};

// `nrm` is the receiver normal, or null at a volume vertex (which has none to bound with).
// Mode W keeps the exact path: its deterministic G x G lattice has no variance to trade.
__device__ static void dPickEmitters(const DScene& sc, const DVec3& p, const DVec3* nrm,
                                     DRng& rng, DEmitterDraw& d) {
    d.n = 0; d.all = false;
    if (!sc.bkLightTree || sc.bkWhitted || sc.lightTreeRoot < 0 || !sc.lightTree) {
        d.all = true; d.n = sc.nEmitters; return;
    }
    for (int i = 0; i < sc.nLightTreeAlways && d.n < kDMaxLightPick; ++i) {
        d.s[d.n].emitter = sc.lightTreeAlways[i];
        d.s[d.n].pdf = 1.0;
        ++d.n;
    }
    int budget = sc.bkLightSamples;
    if (budget > kDMaxLightPick - d.n) budget = kDMaxLightPick - d.n;
    if (budget > 0) {
        const double pp[3] = {(double)p.x, (double)p.y, (double)p.z};
        double nn[3] = {0.0, 0.0, 0.0};
        if (nrm) { nn[0] = (double)nrm->x; nn[1] = (double)nrm->y; nn[2] = (double)nrm->z; }
        d.n += ltSample<DRng, kDMaxLightPick + 2>(
                   sc.lightTree, sc.lightTreeRoot, pp, nn, nrm != nullptr,
                   sc.bkLightSplit, budget, d.s + d.n, rng);
    }
}

// `giDepth` selects mode W's shadow-ray grid: 0 = a primary vertex (bkGrid), 1 = a gather
// vertex (the coarser bkGiGrid — its soft-shadow detail is about to be averaged over giDirs
// directions anyway, so paying bkGrid^2 there multiplies the gather's cost for no return).
// Ignored unless sc.bkWhitted.
// `hs` non-null = hair-fiber vertex: the caller passes rho = 1 and bkEmitterGeom swaps in
// the PI*hairFCos response, so f * w collapses to f_hair * cosLong * (light geometry) —
// the exact CPU neeLight convention (backward.h).
__device__ static double bkNeeLight(const DScene& sc, const DHit& h, Real rho,
                                    double invPdfLambda, Real lambda, DRng& rng,
                                    int giDepth = 0, const DHairShade* hs = nullptr) {
    double total = 0.0;
    Real f = rho / (Real)DPI;                         // Lambertian BRDF
    DVec3 ngo0 = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
    const bool whitted = (sc.bkWhitted != 0);
    DEmitterDraw draw; dPickEmitters(sc, h.p, &h.n, rng, draw);
    for (int di = 0; di < draw.n; ++di) {
        const int k = draw.emitter(di);
        const double selW = draw.weight(di);            // 1/p(e); exactly 1 on the all path
        if (selW <= 0.0) continue;
        const DEmitter& em = sc.emitters[k];
        if (em.collimated || em.shape == 3) continue;   // collimated beams / env (env: bkNeeEnv)
        const bool uv = dEmitterNeedsUV(em);
        // Whitted: G x G deterministic shadow rays per area light, averaged. A
        // deterministic emitter (spot/beam) has nothing to stratify, so it stays at 1.
        const int G = (whitted && uv) ? (giDepth ? sc.bkGiGrid : sc.bkGrid) : 1;
        const int nS = G * G;
        // Loop-invariant across the lattice (a pure table lookup), so hoisted rather than
        // repeated per shadow ray — value-identical either way.
        const double emitW = (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
        double acc = 0.0;
        for (int s = 0; s < nS; ++s) {
            Real u1 = (Real)0, u2 = (Real)0;
            if (whitted) { if (uv) dGridUV(s, G, u1, u2); }
            else if (uv) { u1 = rng.uniform(); u2 = rng.uniform(); }
            BkNeeGeom g;
            if (!bkEmitterGeom(sc, h, ngo0, em, u1, u2, g, hs)) continue;
            double contrib = g.sun
                ? (double)(f * g.wSun) * emitW
                : g.spot
                ? (double)(f * g.fall * g.cosSurf / g.dist2 * g.stG) * emitW
                : (double)(f * g.G) * emitW * (double)em.area * (double)g.stG;
            // Shadow-ray transmittance through any participating media (superposition;
            // homogeneous = exact exp with no rng draw, heterogeneous = ratio tracking).
            // Matches the forward connectVolume / device volume-NEE transmittance so
            // surface direct light agrees between the forward and backward estimators.
            if (sc.mediaN > 0)
                contrib *= (double)dMediaTransmittance(sc, h.p, g.wi, g.dist, lambda, rng);
            acc += contrib;
        }
        total += ((nS > 1) ? acc / (double)nS : acc) * selW;
    }
    return total;
}

// Hero-wavelength surface NEE (device twin of backward.h neeLightHero): ONE shared
// visibility sample per emitter — the very rng stream bkNeeLight would draw — evaluated
// for all `nUp` live wavelengths, accumulating thr[i]*(rho[i]/PI)*SPD(lam[i])*invPdf[i]*w
// into L[i]. Only reached on the media-free hero fast path, so there is no shadow-ray
// transmittance term (bkRadianceHero is gated on mediaN == 0).
__device__ static void bkNeeLightHero(const DScene& sc, const DHit& h, const Real* rho,
                                      double* L, const double* thr, const Real* lam,
                                      const double* invPdf, int nUp, DRng& rng,
                                      int giDepth = 0) {
    DVec3 ngo0 = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
    const bool whitted = (sc.bkWhitted != 0);
    DEmitterDraw draw; dPickEmitters(sc, h.p, &h.n, rng, draw);
    for (int di = 0; di < draw.n; ++di) {
        const int k = draw.emitter(di);
        const double selW = draw.weight(di);            // 1/p(e); exactly 1 on the all path
        if (selW <= 0.0) continue;
        const DEmitter& em = sc.emitters[k];
        if (em.collimated || em.shape == 3) continue;
        const bool uv = dEmitterNeedsUV(em);
        const int G = (whitted && uv) ? (giDepth ? sc.bkGiGrid : sc.bkGrid) : 1;
        const int nS = G * G;
        // Folding the selection weight into the per-sample scale keeps the all-emitters
        // path bit-identical (selW is exactly 1.0 there, and x*1.0 == x).
        const double invS = ((nS > 1) ? 1.0 / (double)nS : 1.0) * selW;
        for (int s = 0; s < nS; ++s) {
            Real su1 = (Real)0, su2 = (Real)0;
            if (whitted) { if (uv) dGridUV(s, G, su1, su2); }
            else if (uv) { su1 = rng.uniform(); su2 = rng.uniform(); }
            BkNeeGeom g;
            if (!bkEmitterGeom(sc, h, ngo0, em, su1, su2, g)) continue;
            for (int i = 0; i < nUp; ++i) {
                Real f = rho[i] / (Real)DPI;
                double emitW = (double)specLookup(em.emitSpd, lam[i]) * invPdf[i];
                double contrib = g.sun
                    ? (double)(f * g.wSun) * emitW
                    : g.spot
                    ? (double)(f * g.fall * g.cosSurf / g.dist2 * g.stG) * emitW
                    : (double)(f * g.G) * emitW * (double)em.area * (double)g.stG;
                L[i] += thr[i] * contrib * invS;
            }
        }
    }
}

// Volume next-event estimation (device twin of backward.h neeVolume): connect a fog
// scattering vertex `p` (photon arriving along `wIn`) to each area/sphere/cylinder
// emitter. The surface BRDF+cosine are replaced by the single-scattering albedo and the
// HG phase function; the shadow ray carries media transmittance (superposition over all
// media). Spot/env/collimated emitters are gated to the CPU, so they're skipped here.
// Uniform area sampling (an independent noise realization vs the CPU's cone/arc
// importance sampling, same expectation) — matches bkNeeLight's convention.
__device__ static double bkNeeVolume(const DScene& sc, const DVec3& p, const DVec3& wIn,
                                     const DMedium& med, double invPdfLambda, Real lambda,
                                     DRng& rng) {
    double total = 0.0;
    Real alb = medAlbedo(med, lambda);
    if (alb <= (Real)0) return 0.0;
    // No shading normal at a fog vertex, so the tree bounds on the receiver side only.
    DEmitterDraw draw; dPickEmitters(sc, p, nullptr, rng, draw);
    for (int di = 0; di < draw.n; ++di) {
        const int k = draw.emitter(di);
        const double selW = draw.weight(di);            // 1/p(e); exactly 1 on the all path
        if (selW <= 0.0) continue;
        const DEmitter& em = sc.emitters[k];
        if (em.collimated || em.shape == 3) continue;   // collimated beams / env (env: bkNeeEnvVolume)
        if (em.shape == 2) {
            // Point spot at a volume vertex (device twin of neeVolume's spot branch): no
            // surface cosine, cone falloff only; HG phase toward the light supplies the pdf.
            DVec3 toL = em.origin - p;
            Real dist2 = dot(toL, toL);
            Real dist  = sqrt(dist2);
            DVec3 wi = toL / dist;
            Real fall = (Real)spotFalloff(dot(wi * (Real)(-1), em.beamDir), em.spotCosInner, em.spotCosOuter);
            if (fall <= (Real)0) continue;
            if (occluded(sc, p + wi * RAY_EPS, wi, dist - (Real)2 * RAY_EPS)) continue;
            Real phase = dMedPhase(med, dot(wIn, wi), lambda);
            double emitW = (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
            double contrib = (double)(alb * phase * fall / dist2) * emitW;
            contrib *= (double)dMediaTransmittance(sc, p, wi, dist, lambda, rng);
            total += contrib * selW;
            continue;
        }
        if (em.shape == 6) {
            // Distant sun at a volume vertex (device twin of neeVolume's Sun branch):
            // cone-sampled direction (pdf 1/Omega, so 1/pdfW = Omega), no surface cosine,
            // transmittance out to the scene exit.
            double s1 = (double)rng.uniform(), s2 = (double)rng.uniform();
            DVec3 wi = dSunSampleCone(em, em.beamDir * (Real)(-1), s1, s2);
            Real dist = (Real)((double)length(sc.sceneCenter - p) + sc.sceneRadius);
            if (occluded(sc, p + wi * RAY_EPS, wi, dist)) continue;
            Real phase = dMedPhase(med, dot(wIn, wi), lambda);
            double emitW = (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
            double contrib = (double)(alb * phase) * emitW * em.spotOmega;
            contrib *= (double)dMediaTransmittance(sc, p, wi, dist, lambda, rng);
            total += contrib * selW;
            continue;
        }
        Real u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 y, nL;
        // Also returns the sampled point's emission-pattern factor (1.0 when unpatterned).
        double epat = dEmitterSamplePointPat(sc, em, (double)u1, (double)u2, y, nL);
        DVec3 toL = y - p;
        Real dist2 = dot(toL, toL);
        Real dist  = sqrt(dist2);
        DVec3 wi = toL / dist;
        Real cosLight = dot(nL, wi * (Real)(-1));         // light is one-sided
        if (cosLight <= 0) continue;
        if (occluded(sc, p + wi * RAY_EPS, wi, dist - (Real)2 * RAY_EPS)) continue;
        Real phase = dMedPhase(med, dot(wIn, wi), lambda); // phase == its own pdf (HG or rainbow)
        Real G = cosLight / dist2;                        // no surface cosine at a volume vertex
        double emitW = (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
        double contrib = (double)(alb * phase * G) * emitW * (double)em.area;
        if (epat != 1.0) contrib *= epat;                 // no-op without a pattern
        contrib *= (double)dMediaTransmittance(sc, p, wi, dist, lambda, rng);
        total += contrib * selW;
    }
    return total;
}

// Environment next-event estimation at a surface vertex (device twin of backward.h
// neeEnv / envGeom, CONSTANT-env scope: an image env stays on the CPU). One uniform-
// sphere env-direction sample (pdf 1/4pi), the shading / shadow-terminator gate, a
// shadow ray to the scene exit carrying media transmittance, and a balance-heuristic
// MIS weight against the cosine-sampled continuation (MIS'd again on the BSDF-sampled
// escape in bkRadiance). Returns the contribution (0 if occluded / below the horizon).
// The env connection's SAMPLING + VISIBILITY + MIS weight, all wavelength-independent,
// factored out of bkNeeEnv (device twin of backward.h envGeom) so the scalar and hero
// env NEE share one direction sample and one shadow ray.
struct BkEnvGeom {
    DVec3  wi;        // sampled incoming env direction
    double pdfW;      // its solid-angle pdf
    Real   cosSurf;   // cosine at the shading surface
    Real   stG;       // Chiang shadow-terminator gate
    double wMis;      // balance heuristic vs. the cosine-sampled continuation
    double farDist;   // shadow-ray length to the scene exit
};
__device__ static bool bkEnvGeom(const DScene& sc, const DHit& h, DRng& rng, BkEnvGeom& g,
                                 const DHairShade* hs = nullptr) {
    // Sample an incoming env direction: image env importance-samples the luminance CDF
    // (dEnvSample gives dir + solid-angle pdfW), constant env is uniform on the sphere
    // (pdf 1/4pi). Both draw exactly two uniforms in the same order as the CPU
    // scene.sampleEnvDir, so the estimator matches.
    if (sc.env.scale != nullptr) {
        dEnvSample(sc.env, (double)rng.uniform(), (double)rng.uniform(), g.wi, g.pdfW);
        if (g.pdfW <= 0.0) return false;
    } else {
        double z = 1.0 - 2.0 * (double)rng.uniform();
        double sr = sqrt(fmax(0.0, 1.0 - z * z));
        double phi = 2.0 * DPI * (double)rng.uniform();
        g.wi = DVec3{(Real)(sr * cos(phi)), (Real)(sr * sin(phi)), (Real)z};
        g.pdfW = 1.0 / (4.0 * DPI);
    }
    if (hs) {
        // Hair fiber (device twin of backward.h envGeom's hs branch): full-sphere response
        // PI*hairFCos with stG = 1 (a fiber has no hemisphere or terminator), shadow ray
        // from the fiber exit offset with the FULL farDist length, and the MIS counterpart
        // is the hair BCSDF's own pdf for wi rather than the cosine hemisphere — the
        // continuation this NEE sample competes against is dhair::sample.
        g.cosSurf = (Real)DPI * dHairFCos(*hs, g.wi);
        if (!(g.cosSurf > (Real)0)) return false;
        g.stG = (Real)1;
        g.farDist = (double)length(sc.sceneCenter - h.p) + sc.sceneRadius;
        const Real off = dHairExitOffset(*hs, h.n, g.wi);
        if (occluded(sc, h.p + g.wi * off, g.wi, (Real)g.farDist)) return false;
        const double pdfBsdf = dhair::pdf(hs->b, hs->woLocal, dhair::toLocal(hs->fr, g.wi));
        g.wMis = g.pdfW / (g.pdfW + pdfBsdf);
        return true;
    }
    g.cosSurf = dot(h.n, g.wi);
    if (g.cosSurf <= (Real)0) return false;                 // below the shading horizon
    DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
    g.stG = dShadowTerminatorG(g.wi, h.n, ngo);             // Chiang soft terminator (1 if flat)
    if (g.stG <= (Real)0) return false;                     // behind true geometry: hard shadow
    g.farDist = (double)length(sc.sceneCenter - h.p) + sc.sceneRadius;
    if (occluded(sc, h.p + ngo * RAY_EPS, g.wi, (Real)g.farDist)) return false;
    double pdfBsdf = (double)g.cosSurf / DPI;               // cosine-hemisphere pdf for wi
    g.wMis = g.pdfW / (g.pdfW + pdfBsdf);                   // balance heuristic
    return true;
}

__device__ static double bkNeeEnv(const DScene& sc, const DHit& h, Real rho,
                                  double invPdfLambda, Real lambda, DRng& rng,
                                  const DHairShade* hs = nullptr) {
    if (sc.envIndex < 0) return 0.0;
    BkEnvGeom g;
    if (!bkEnvGeom(sc, h, rng, g, hs)) return 0.0;
    double Lenv = (sc.env.scale != nullptr) ? dEnvRadiance(sc.env, g.wi, lambda)
                                            : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lambda);
    if (Lenv <= 0.0) return 0.0;
    double contrib = ((double)rho / DPI) * Lenv * (double)g.cosSurf * invPdfLambda / g.pdfW
                     * g.wMis * (double)g.stG;
    if (sc.mediaN > 0)                                      // Beer-Lambert to the scene exit
        contrib *= (double)dMediaTransmittance(sc, h.p, g.wi, (Real)g.farDist, lambda, rng);
    return contrib;
}

// Hero-wavelength environment NEE (device twin of backward.h neeEnvHero): one shared env
// direction + shadow ray, evaluated for all `nUp` live wavelengths. Media-free hero fast
// path, so no transmittance term.
__device__ static void bkNeeEnvHero(const DScene& sc, const DHit& h, const Real* rho,
                                    double* L, const double* thr, const Real* lam,
                                    const double* invPdf, int nUp, DRng& rng) {
    if (sc.envIndex < 0) return;
    BkEnvGeom g;
    if (!bkEnvGeom(sc, h, rng, g)) return;
    const bool imageEnv = (sc.env.scale != nullptr);
    for (int i = 0; i < nUp; ++i) {
        double Lenv = imageEnv ? dEnvRadiance(sc.env, g.wi, lam[i])
                               : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lam[i]);
        if (Lenv <= 0.0) continue;
        L[i] += thr[i] * (((double)rho[i] / DPI) * Lenv * (double)g.cosSurf * invPdf[i]
                          / g.pdfW * g.wMis * (double)g.stG);
    }
}

// Environment NEE at a fog scattering vertex (device twin of backward.h neeEnvVolume,
// constant-env scope). The surface BRDF/cosine is replaced by the single-scattering
// albedo and the HG phase (which is also the pdf for the MIS weight against the phase-
// sampled continuation). Only invoked when the scene has an env light.
__device__ static double bkNeeEnvVolume(const DScene& sc, const DVec3& p, const DVec3& wIn,
                                        const DMedium& med, double invPdfLambda, Real lambda,
                                        DRng& rng) {
    if (sc.envIndex < 0) return 0.0;
    Real alb = medAlbedo(med, lambda);
    if (alb <= (Real)0) return 0.0;
    DVec3 wi; double pdfW;
    const bool imageEnv = (sc.env.scale != nullptr);
    if (imageEnv) {
        dEnvSample(sc.env, (double)rng.uniform(), (double)rng.uniform(), wi, pdfW);
        if (pdfW <= 0.0) return 0.0;
    } else {
        double z = 1.0 - 2.0 * (double)rng.uniform();
        double sr = sqrt(fmax(0.0, 1.0 - z * z));
        double phi = 2.0 * DPI * (double)rng.uniform();
        wi = DVec3{(Real)(sr * cos(phi)), (Real)(sr * sin(phi)), (Real)z};
        pdfW = 1.0 / (4.0 * DPI);
    }
    double farDist = (double)length(sc.sceneCenter - p) + sc.sceneRadius;
    if (occluded(sc, p + wi * RAY_EPS, wi, (Real)farDist)) return 0.0;
    double Lenv = imageEnv ? dEnvRadiance(sc.env, wi, lambda)
                           : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lambda);
    if (Lenv <= 0.0) return 0.0;
    Real phase = dMedPhase(med, dot(wIn, wi), lambda);      // phase == its own pdf (HG or rainbow)
    double wMis = pdfW / (pdfW + (double)phase);            // balance heuristic
    double contrib = (double)alb * (double)phase * Lenv * invPdfLambda / pdfW * wMis;
    contrib *= (double)dMediaTransmittance(sc, p, wi, (Real)farDist, lambda, rng);
    return contrib;
}

// Mode W path context (device twin of BackwardRenderer::GiCtx). `depth == 0` is the primary
// camera path, `depth == 1` a gather ray (does not recurse, uses bkGiGrid, caps at
// bkGiBounce). `sIdx` is the ABSOLUTE sample index, which rotates every deterministic lattice
// so -spp progressively refines instead of re-rendering the identical image. `bounce` is the
// bounce index along the path, so a per-vertex deterministic decision (the glossy lobe — see
// dWhittedGlossyDir) picks a different sequence at each vertex rather than driving every
// glossy bounce off the same 1-D lattice.
struct DGiCtx {
    int depth = 0;
    unsigned long long sIdx = 0;
    int bounce = 0;
};

// ---- the deterministic one-bounce gather (N3c) ----------------------------------------
// A gather is MUTUAL recursion between the tracers and the gather helpers, so both halves
// are declared here and defined below. The recursion is bounded at COMPILE time by the
// `GiDepth` template parameter: only a `GiDepth == 0` instantiation contains a gather call
// at all, and the rays it spawns are `GiDepth == 1`, which contains none. Exactly the same
// trick `AllowSplit` already uses for split-at-dispersion, and for the same reason — no
// device stack sizing and no -rdc / relocatable-device-code requirement.
//
// Depth therefore has to be a template parameter rather than a runtime `gi.depth` test even
// though `gi.depth` still carries it for the RUNTIME behaviours it also selects (the coarser
// bkGiGrid shadow lattice, the bkGiBounce depth cap, the non-specular start, and the escaped
// ray's far-field bkAmbient tail — all of which landed with N3a).
template<int GiDepth>
__device__ static double bkRadiance(const DScene& sc, int diffraction, DVec3 ro, DVec3 rd,
                                    Real lambda, double invPdfLambda, DRng& rng, DGiCtx gi);
template<int GiDepth>
__device__ static void bkRadianceHero(const DScene& sc, int diffraction, DVec3 ro, DVec3 rd,
                                      const Real* lamIn, const double* invPdfIn, int C,
                                      double* Lout, DRng& rng, DGiCtx gi);
// Neither gather is templated: a gather only ever happens at depth 0, so its rays are always
// depth 1 and it can name that instantiation directly.
__device__ static double bkGiGather(const DScene& sc, int diffraction, const DHit& h, Real rho,
                                    Real lambda, double invPdfLambda, DRng& rng, DGiCtx gi);
__device__ static void bkGiGatherHero(const DScene& sc, int diffraction, const DHit& h,
                                      const Real* rho, double* L, const double* thr,
                                      const Real* lam, const double* invPdf, int nUp,
                                      DRng& rng, DGiCtx gi);

// Backward hair-fiber interaction — the body of bkInteract's D_HAIR case, split out and
// explicitly __noinline__ for the same reason as the forward interactHair (see its
// comment): inlined, the double-precision DHairShade/dhair locals would merge into the
// shared bkInteract/bkRadiance frames and permanently inflate kBackward's MIN_STACK
// (driver-reserved per thread × ~200k resident threads, hair scene or not). As a
// separate frame the bytes are live only during a hair hit. Plain function (not a
// template): hair never gathers, so one instantiation serves bkInteract<false>/<true>.
//
// Hair fiber BCSDF (device twin of backward.h MatType::Hair). NEE runs with rho = 1:
// bkEmitterGeom / bkEnvGeom swap the surface cosine for the full-sphere PI*hairFCos
// response, so neeLight's f = rho/PI times that is exactly f_hair * cosLong * (light
// geometry). (-dual-scatter hair scenes are gated to the CPU by backwardOnGpuOk, so
// there is no dual-scatter path here.) Returns the bkInteract contract: true = path
// continues (ro/rd set), false = terminated.
__device__ __noinline__ static bool bkInteractHair(const DScene& sc, const DMaterial& m,
        const DHit& h, bool directOnly, bool whitted,
        DVec3& ro, DVec3& rd, Real lambda, double invPdfLambda,
        double& thr, double& L, bool& specularArrival,
        double& contBsdfPdf, DRng& rng, int giDepth) {
    const DVec3 wPrev = rd * (Real)(-1);
    const DHairShade hsv = dHairShadeAt(sc, m, h, lambda, wPrev);
    L += thr * bkNeeLight(sc, h, (Real)1, invPdfLambda, lambda, rng, giDepth, &hsv);
    if (sc.envIndex >= 0)
        L += thr * bkNeeEnv(sc, h, (Real)1, invPdfLambda, lambda, rng, &hsv);
    if (directOnly) return false;             // mode W -direct: direct fiber light only
    // Continuation: importance-sample the BCSDF. Named u's — C++ leaves argument
    // evaluation order unspecified, and these four MUST come off the stream in order.
    const double u0 = rng.uniform(), u1 = rng.uniform(),
                 u2 = rng.uniform(), u3 = rng.uniform();
    double pdfH = 0.0, fv = 0.0;
    const dhair::V3 wl = dhair::sample(hsv.b, hsv.woLocal, u0, u1, u2, u3, pdfH, fv);
    if (!(pdfH > 0.0) || !(fv > 0.0)) return false;
    // Survive with T = f*cosLong/pdf (= sum of lobe albedos when the sample is
    // exact), throughput otherwise unchanged — the same analog RR as the CPU and
    // the forward tracer. Mode W carries T as weight instead of flipping the coin.
    const double cosLong = dhair::safeSqrt(1.0 - dhair::sqr(dhair::clampd(wl.x, -1.0, 1.0)));
    double T = fv * cosLong / pdfH; T = T < 0.0 ? 0.0 : (T > 1.0 ? 1.0 : T);
    if (whitted) { if (!dWhittedAttenuate(thr, T)) return false; }
    else if ((double)rng.uniform() >= T) return false;
    const DVec3 wOut = dhair::toWorld(hsv.fr, wl);
    contBsdfPdf = pdfH;                       // env-escape MIS vs the hair pdf
    ro = h.p + wOut * dHairExitOffset(hsv, h.n, wOut);
    rd = wOut;
    specularArrival = false;                  // NEE covered direct light here
    return true;
}

// Handle ONE surface material interaction on a single wavelength — the whole material
// switch, factored out of bkRadiance (device twin of backward.h interactMaterial) so the
// scalar tracer and the hero tracer (which de-heros before calling this) share one copy.
// `mp` is the resolved leaf material (Mix already peeled by the caller); the surface's own
// emission is handled by the caller BEFORE this call. All path state is in/out. Returns
// true if the path continues (ray + state updated), false if it terminated (L already
// holds this path's final value): a `break` in the old switch maps to `return true`, a
// `return L` to `return false`.
//
// `AllowGather` is the compile-time half of the -gi gather test (see above): true only for a
// depth-0 scalar camera path, so that this is the one instantiation carrying the recursive
// call. The hero tracer passes FALSE, not its own depth — it handles Diffuse and
// DiffuseTransmit (the only two materials that gather) inline with the whole bundle, and
// routes only the dispersive materials here, so a gathering copy for it would be dead code.
template<bool AllowGather>
__device__ static bool bkInteract(const DScene& sc, const DMaterial* mp, const DHit& h,
                                  int matId, int diffraction, bool directOnly,
                                  DVec3& ro, DVec3& rd, Real& lambda, double& invPdfLambda,
                                  double& thr, double& L, bool& specularArrival,
                                  double& contBsdfPdf, DMediumStack& stk, DRng& rng,
                                  DGiCtx gi) {
    const bool whitted = (sc.bkWhitted != 0);
    switch (mp->type) {
        case D_DIELECTRIC: {
            // Mode W: dominant Fresnel branch weighted into the throughput instead of a coin
            // flip (see refractOrReflect's whittedWeight). Attenuating AFTER the call is safe
            // because the ray has not been traced yet — returning false here just ends the
            // path at this vertex, as elsewhere in mode W.
            double wW = 1.0;
            DVec3 nro, nrd;
            dDielectricStep(sc, *mp, h, rd, lambda, rng, matId, stk, nro, nrd,
                            whitted ? &wW : nullptr);
            if (whitted && !dWhittedAttenuate(thr, wW)) return false;
            ro = nro; rd = nrd; specularArrival = true; return true;
        }
        case D_THINFILM: {
            // Mode W: dominant interference branch weighted in, exactly as D_DIELECTRIC above
            // (see thinFilmInterface's whittedWeight).
            DVec3 nro, nrd; double wW = 1.0;
            if (!thinFilmInterface(sc, *mp, h, rd, lambda, rng, nro, nrd,
                                   whitted ? &wW : nullptr)) return false;
            if (whitted && !dWhittedAttenuate(thr, wW)) return false;
            ro = nro; rd = nrd; specularArrival = true; return true;
        }
        case D_MULTILAYER: {
            DVec3 nro, nrd; double wW = 1.0;          // mode W: see D_THINFILM above
            if (!multilayerInterface(*mp, h, rd, lambda, rng, nro, nrd,
                                     whitted ? &wW : nullptr)) return false;
            if (whitted && !dWhittedAttenuate(thr, wW)) return false;
            ro = nro; rd = nrd; specularArrival = true; return true;
        }
        case D_MIRROR: {
            Real r = clamp01(dReflectSlot(sc, *mp, h, lambda));
            // Mode W: carry the reflectance as WEIGHT instead of rolling for survival. Same
            // expected value, zero variance — the whole reason a deterministic preview
            // converges at 1 spp where Russian roulette needs tens.
            if (whitted) { if (!dWhittedAttenuate(thr, (double)r)) return false; }
            else if (rng.uniform() >= r) return false;   // RR absorb
            ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); specularArrival = true; return true;
        }
        case D_GRATING: {
            Real r = clamp01(dReflectSlot(sc, *mp, h, lambda));
            if (whitted) { if (!dWhittedAttenuate(thr, (double)r)) return false; }
            else if (rng.uniform() >= r) return false;
            DVec3 nro, nrd;
            // Mode W: the diffraction ORDER comes off the (sIdx, bounce) lattice instead of the
            // rng, so every pixel picks the same order and the preview is noise-free.
            const double uOrd = whitted ? dWhittedOrderU(gi.sIdx, gi.bounce) : 0.0;
            if (!gratingDiffract(*mp, h, rd, lambda, diffraction, rng, nro, nrd,
                                 whitted ? &uOrd : nullptr)) return false;
            ro = nro; rd = nrd; specularArrival = true; return true;
        }
        case D_HALFMIRROR: {
            Real r = clamp01(dReflectSlot(sc, *mp, h, lambda));
            // Mode W: a true beam splitter needs the path to FORK, which this iterative loop
            // cannot do. Take the dominant branch and weight it, so a preview is stable rather
            // than a 50/50 coin flipped per pixel. (The minority branch is dropped, not just
            // dimmed — a half mirror previews as whichever lobe is stronger.)
            if (whitted) {
                const bool refl = (r >= (Real)0.5);
                if (!dWhittedAttenuate(thr, refl ? (double)r : 1.0 - (double)r)) return false;
                if (refl) { ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); }
                else      { ro = h.p + rd * RAY_EPS; }
            } else if (rng.uniform() < r) { ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); }
            else                          { ro = h.p + rd * RAY_EPS; }
            specularArrival = true; return true;
        }
        case D_FILTER: {
            // Colored gel filter: pass straight through, survive with prob T(lambda).
            Real t = clamp01(dTransmitSlot(sc, *mp, h, lambda));
            if (whitted) { if (!dWhittedAttenuate(thr, (double)t)) return false; }
            else if (rng.uniform() >= t) return false;   // absorbed
            ro = h.p + rd * RAY_EPS;                // direction unchanged
            specularArrival = true; return true;
        }
        case D_GLOSSY: {
            Real r = clamp01(dReflectSlot(sc, *mp, h, lambda));
            // Mode W: the lobe off a deterministic lattice rather than the rng, so the
            // direction is the same for every pixel (noise-free) but varies with the sample
            // index (so -spp actually resolves the lobe). At -spp 1 this IS the mirror
            // direction, which is exact for a near-mirror and over-sharpens as roughness
            // grows; the fix for that is more spp, which now works.
            if (whitted) {
                if (!dWhittedAttenuate(thr, (double)r)) return false;
                DVec3 o = dWhittedGlossyDir(reflectv(rd, h.n), dMatRoughness(sc, *mp, h),
                                            gi.sIdx, gi.bounce);
                if (dot(o, h.n) <= 0) return false;
                ro = h.p + h.n * RAY_EPS; rd = o; specularArrival = true; return true;
            }
            if (rng.uniform() >= r) return false;
            DVec3 o = sampleGlossy(reflectv(rd, h.n), dMatRoughness(sc, *mp, h), rng);
            if (dot(o, h.n) <= 0) return false;
            ro = h.p + h.n * RAY_EPS; rd = o; specularArrival = true; return true;
        }
        case D_DIFFUSETRANSMIT: {
            // Two-lobe Lambertian (device twin of backward.h DiffuseTransmit): NEE the
            // reflect lobe against lights in the front (+n) hemisphere and the transmit
            // lobe in the back (-n) hemisphere (a normal-flipped Hit reuses bkNeeLight),
            // then continue reflect / transmit / absorb (throughput unchanged on survival).
            Real rhoR = clamp01(dDiffuseRho(sc, *mp, h, lambda));
            Real rhoT = clamp01(dTransmitSlot(sc, *mp, h, lambda));
            Real sum = rhoR + rhoT;
            if (sum > (Real)1) { rhoR /= sum; rhoT /= sum; sum = (Real)1; }   // energy guard
            DVec3 nb = h.n * (Real)(-1);
            L += thr * bkNeeLight(sc, h, rhoR, invPdfLambda, lambda, rng, gi.depth);  // front
            if (sc.envIndex >= 0)
                L += thr * bkNeeEnv(sc, h, rhoR, invPdfLambda, lambda, rng);
            DHit hb = h; hb.n = nb;
            L += thr * bkNeeLight(sc, hb, rhoT, invPdfLambda, lambda, rng, gi.depth); // back
            if (sc.envIndex >= 0)
                L += thr * bkNeeEnv(sc, hb, rhoT, invPdfLambda, lambda, rng);
            // Mode W indirect diffuse. A translucent surface receives from the FULL sphere, so
            // both lobes take the fill — before v0.106.0 a DiffuseTransmit vertex got none.
            // With -gi each lobe runs a real gather into its OWN hemisphere (the back lobe off
            // the normal-flipped `hb`), which is why the flat fill is the `else`.
            if (whitted) {
                bool gathered = false;
                if constexpr (AllowGather) {
                    if (sc.bkGiDirs > 0) {
                        L += thr * bkGiGather(sc, diffraction, h,  rhoR, lambda, invPdfLambda, rng, gi);
                        L += thr * bkGiGather(sc, diffraction, hb, rhoT, lambda, invPdfLambda, rng, gi);
                        gathered = true;
                    }
                }
                if (!gathered && sc.bkAmbient > 0.0)
                    L += thr * (double)(rhoR + rhoT) * sc.bkAmbient;
            }
            if (directOnly) return false;            // Whitted: no diffuse indirect
            Real u = rng.uniform();
            if (u < rhoR)     { DVec3 wOut = cosineHemisphere(h.n, rng); contBsdfPdf = fmax(0.0, (double)dot(wOut, h.n)) / DPI; ro = h.p + h.n * RAY_EPS; rd = wOut; specularArrival = false; return true; }
            else if (u < sum) { DVec3 wOut = cosineHemisphere(nb,  rng); contBsdfPdf = fmax(0.0, (double)dot(wOut, nb )) / DPI; ro = h.p + nb  * RAY_EPS; rd = wOut; specularArrival = false; return true; }
            return false;                            // absorbed
        }
        case D_FLUORESCENT: {
            // Bispectral reradiation — device adjoint of backward.h MatType::Fluorescent.
            // Elastic base reflects at the output wavelength; the fluorescent channel
            // excites at a separately-sampled lambdaIn (Stokes shift). Both channels NEE;
            // one stochastic continuation carries the indirect term.
            double rhoEl = clamp01((double)specLookup(mp->reflect, lambda));   // elastic base @lambda(out)
            // `gi.depth` matters: it selects bkGiGrid over bkGrid at a gather vertex, exactly
            // as the Diffuse case does. Omitting it made a fluorescent surface pay bkGrid^2
            // shadow rays inside a -gi gather (host twin: backward.h MatType::Fluorescent).
            L += thr * bkNeeLight(sc, h, (Real)rhoEl, invPdfLambda, lambda, rng, gi.depth);
            if (sc.envIndex >= 0)
                L += thr * bkNeeEnv(sc, h, (Real)rhoEl, invPdfLambda, lambda, rng);
            double Mint = mp->fluoMint;
            bool haveFluoro = (Mint > 0.0 && mp->fluoYield > (Real)0);
            double gOut = 0.0, rhoFluo = 0.0, invPdfIn = 0.0;
            Real lambdaIn = 0;
            if (haveFluoro) {
                gOut = ((double)specLookup(mp->fluoEmitSpec, lambda) / Mint) * invPdfLambda;
                double pin = 0.0;
                // Mode W: the Stokes-shift EXCITATION wavelength comes off the (sIdx, bounce)
                // lattice rather than the rng -- the same CDF inversion, just a stratified u,
                // so the estimator is untouched and only the per-pixel luck goes away. This is
                // mode W's last rng draw here; the continuation coin below is unreachable
                // because mode W implies directOnly, which returns first.
                //
                // The CDF is the material's own excitation table (absorb x illuminant), so
                // every draw lands inside the dye's absorption band -- see
                // DMaterial::fluoInCdfOffset / Material::fluoInSampler.
                lambdaIn = whitted
                    ? dSampleFluoInU(sc, *mp, dWhittedFluoroU(gi.sIdx, gi.bounce), pin)
                    : dSampleFluoInU(sc, *mp, (double)rng.uniform(), pin);
                if (pin > 0.0) {
                    // 1/pdf of the sampler we actually drew from (pre-0.115.0: the
                    // analytic dInvPdfLambda, correct only while that sampler was the
                    // illuminant).
                    invPdfIn = 1.0 / pin;
                    double rhoIn = clamp01((double)specLookup(mp->reflect, lambdaIn));
                    double eps   = clamp01((double)specLookup(mp->fluoAbsorb, lambdaIn));
                    double aEffIn = fmin(eps, fmax(0.0, 1.0 - rhoIn));
                    rhoFluo = aEffIn * (double)mp->fluoYield;                 // reradiation albedo @lambdaIn
                    if (rhoFluo > 0.0) {                                      // fluoro DIRECT NEE
                        L += thr * gOut * bkNeeLight(sc, h, (Real)rhoFluo, invPdfIn, lambdaIn,
                                                     rng, gi.depth);
                        if (sc.envIndex >= 0)
                            L += thr * gOut * bkNeeEnv(sc, h, (Real)rhoFluo, invPdfIn, lambdaIn, rng);
                    }
                }
            }
            if (directOnly) return false;                                    // Whitted: no indirect (elastic or fluoro)
            double wFluo = gOut * rhoFluo;                                    // natural indirect-fluoro weight
            double pF = (wFluo > 0.0) ? fmin(fmax(0.0, 1.0 - rhoEl), wFluo) : 0.0;
            double u = rng.uniform();
            if (u < rhoEl) {                                                  // elastic continuation
                DVec3 wOut = cosineHemisphere(h.n, rng);
                contBsdfPdf = fmax(0.0, (double)dot(wOut, h.n)) / DPI;
                ro = h.p + h.n * RAY_EPS; rd = wOut;
                specularArrival = false; return true;
            } else if (u < rhoEl + pF) {                                      // fluoro (wavelength-switched)
                thr *= wFluo / pF;
                lambda = lambdaIn;                                            // Stokes shift (to the input wl)
                invPdfLambda = invPdfIn;
                DVec3 wOut = cosineHemisphere(h.n, rng);
                contBsdfPdf = fmax(0.0, (double)dot(wOut, h.n)) / DPI;
                ro = h.p + h.n * RAY_EPS; rd = wOut;
                specularArrival = false; return true;
            }
            return false;                                                     // absorbed / terminated
        }
        case D_HAIR:
            // Split out (__noinline__) so its fat double-precision frame is only paid
            // on an actual hair hit — see the comment on bkInteractHair.
            return bkInteractHair(sc, *mp, h, directOnly, whitted, ro, rd, lambda,
                                  invPdfLambda, thr, L, specularArrival, contBsdfPdf,
                                  rng, gi.depth);
        case D_DIFFUSE:
        default: {
            Real rho = clamp01(dDiffuseRho(sc, *mp, h, lambda));
            L += thr * bkNeeLight(sc, h, rho, invPdfLambda, lambda, rng, gi.depth);
            if (sc.envIndex >= 0)                   // env-NEE toward the sky (MIS'd on miss)
                L += thr * bkNeeEnv(sc, h, rho, invPdfLambda, lambda, rng);
            // Indirect diffuse. With -gi this is a real single-bounce hemisphere gather
            // (occlusion-aware and spectral); with -gi 0 it falls back to POV-Ray's flat
            // `ambient`, physically a lie but without it a CLOSED room previews with black
            // shadows, since every non-key-lit surface there is lit purely by bounce.
            if (whitted) {
                bool gathered = false;
                if constexpr (AllowGather) {
                    if (sc.bkGiDirs > 0) {
                        L += thr * bkGiGather(sc, diffraction, h, rho, lambda, invPdfLambda,
                                              rng, gi);
                        gathered = true;
                    }
                }
                if (!gathered && sc.bkAmbient > 0.0)
                    L += thr * (double)rho * sc.bkAmbient;
            }
            if (directOnly) return false;           // Whitted: no diffuse indirect
            if (rng.uniform() >= rho) return false; // RR on albedo
            DVec3 wOut = cosineHemisphere(h.n, rng);
            contBsdfPdf = fmax(0.0, (double)dot(wOut, h.n)) / DPI;
            ro = h.p + h.n * RAY_EPS; rd = wOut; specularArrival = false; return true;
        }
    }
}

// Estimate spectral-weighted radiance for one wavelength along a camera ray (port of
// backward.h radiance, v1 scope: participating media + constant environment light).
// Emission added only on specular/camera arrival; diffuse arrivals are covered by NEE.
//
// `GiDepth` is 0 for a camera path and 1 for a -gi gather ray; it decides at COMPILE time
// whether this instantiation's diffuse vertices gather (see the declarations above), which is
// what bounds the recursion to one level.
template<int GiDepth>
__device__ static double bkRadiance(const DScene& sc, int diffraction, DVec3 ro, DVec3 rd,
                                    Real lambda, double invPdfLambda, DRng& rng,
                                    DGiCtx gi) {
    double L = 0.0, thr = 1.0;
    bool specularArrival = (gi.depth == 0);            // camera ray may see a light directly; a
                                                       // gather ray must NOT (the vertex's own
                                                       // NEE already counted that emitter)
    double contBsdfPdf = 0.0;                           // solid-angle pdf of the current continuation (env MIS)
    DMediumStack stk; stk.clear();                     // nested-dielectric medium stack (empty = vacuum)
    const bool whitted = (sc.bkWhitted != 0);
    // Gather rays are bounce-capped (see bkGiBounce). The `min` matters: the host is
    // std::min(maxBounce, giBounce) (backward.h), so without it a `-gi-bounce` larger than
    // `-max-bounce` would let the device trace a gather ray DEEPER than the camera path.
    const int maxBounce = (whitted && gi.depth)
                        ? (sc.bkGiBounce < sc.bkMaxBounce ? sc.bkGiBounce : sc.bkMaxBounce)
                        : sc.bkMaxBounce;
    const bool directOnly = (sc.bkDirectOnly != 0);
    for (int b = 0; b < maxBounce; ++b) {
        // Publish the bounce index so a deterministic per-vertex choice (mode W's glossy
        // lobe) can pick a decorrelated sequence at each depth. Costs nothing otherwise.
        gi.bounce = b;
        // GRIN curved-marching pre-pass (M11): bend the ray through any gradient-index
        // region it enters (symplectic Eikonal integration) BEFORE the surface query —
        // the exact device twin of the forward megakernel's pre-closestHit march and of
        // the CPU backward's grin::march (backward.h). Pure marching does NOT consume a
        // bounce; it stops within one step of a surface or on leaving all GRIN regions.
        // Gated by sc.hasGrin so ordinary scenes are bit-identical. Media free-flight
        // below then samples along the post-bend straight segment, matching the CPU order.
        if (sc.hasGrin) dGrinMarch(sc, ro, rd);
        DHit h = closestHit(sc, ro, rd);
        // O8 stage 2: stamp the shading footprint on the CAMERA SEGMENT only (host twin:
        // backward.h radiance()). A secondary bounce would need ray differentials /
        // cones to know how much its own footprint spread, so it keeps fw = 0
        // (= unknown = unfiltered) rather than reusing the primary's number.
        if (b == 0 && gi.depth == 0 && h.valid)
            h.fw = (Real)patShadingFootprint(sc.bkFwPerDist, (double)h.t, (double)dot(rd, h.n));
        Real dSurf = h.valid ? h.t : (Real)1e30;
        // Participating media: sample a free-flight collision (superposition over all
        // media — homogeneous = exact free-flight, heterogeneous = Woodcock/delta
        // tracking) that competes with the surface hit. On a volume collision, add
        // phase-function NEE, then scatter (HG) or absorb — analog, throughput unchanged.
        // Mirrors backward.h radiance() so the two estimators agree.
        if (sc.mediaN > 0) {
            Real tMed; int whichMed;
            if (dMediaSampleCollision(sc, ro, rd, dSurf, lambda, rng, tMed, whichMed)) {
                DVec3 p = ro + rd * tMed;
                int cm = stk.topMat();     // Beer-Lambert over the in-glass free-flight leg
                Real a = (cm >= 0) ? (Real)specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
                if (a > 0) thr *= exp(-(double)a * (double)tMed);
                const DMedium& med = sc.media[whichMed];
                L += thr * bkNeeVolume(sc, p, rd, med, invPdfLambda, lambda, rng);
                if (sc.envIndex >= 0)                              // env-NEE at the volume vertex
                    L += thr * bkNeeEnvVolume(sc, p, rd, med, invPdfLambda, lambda, rng);
                if (directOnly) return L;                          // Whitted: single-scatter only
                Real alb = medAlbedo(med, lambda);
                if (rng.uniform() >= (double)alb) return L;        // absorbed
                DVec3 wIn = rd;
                Real phPdf;
                ro = p; rd = dMedPhaseSample(med, rd, lambda, rng, phPdf); specularArrival = false;
                contBsdfPdf = (double)phPdf;  // phase pdf of the scatter (env MIS)
                continue;
            }
        }
        if (!h.valid) {                                // escaped the scene
            // Environment radiance from the escape direction (constant env ignores the
            // direction). Full weight on a camera/specular arrival (directly-viewed sky);
            // MIS-weighted (balance heuristic) on a diffuse/volume arrival against the
            // env-NEE already done at the previous vertex, to avoid double-counting.
            if (sc.envIndex >= 0) {
                const bool imageEnv = (sc.env.scale != nullptr);
                double Lenv = (imageEnv ? dEnvRadiance(sc.env, rd, lambda)
                                        : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lambda))
                              * invPdfLambda;
                if (specularArrival) {
                    L += thr * Lenv;
                } else {
                    double pdfEnv = imageEnv ? dEnvPdf(sc.env, rd) : 1.0 / (4.0 * DPI);
                    double wMis = (contBsdfPdf + pdfEnv > 0.0)
                                      ? contBsdfPdf / (contBsdfPdf + pdfEnv) : 0.0;
                    L += thr * Lenv * wMis;
                }
            }
            // Directly-viewed solar disc: camera / specular arrivals only. A diffuse or
            // volume vertex already spent its one estimator on the sun via NEE
            // (bkEmitterGeom / bkNeeVolume) and sets specularArrival = false, so this is
            // a clean single-strategy split, not a missing MIS weight (host twin: backward.h).
            if (sc.sunCount > 0 && specularArrival)
                L += thr * dSunRadiance(sc, rd, lambda) * invPdfLambda;
            // Escaped gather ray -> the far-field `ambient` fill. This is what makes -gi and
            // -ambient compose: in an empty scene every direction escapes and the normalised
            // gather collapses exactly back to rho * ambient, so switching -gi on never
            // steps the exposure.
            if (whitted && gi.depth && sc.bkAmbient > 0.0) L += thr * sc.bkAmbient;
            return L;
        }
        // Beer-Lambert attenuation over the in-glass segment up to this surface
        // (current medium = highest-priority stack entry).
        {
            int cm = stk.topMat();
            Real a = (cm >= 0) ? (Real)specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
            if (a > 0) thr *= exp(-(double)a * (double)h.t);
        }
        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (mp->type == D_MIX) {                       // resolve the mix to a child material
            int child = whitted ? dMixResolveDominant(sc, *mp, h)
                                : dMixResolveChild(sc, *mp, h, rng.uniform());
            if (child < 0) return L;                    // absorbed
            mp = &sc.mats[child]; matId = child;
        }
        // Emission on specular/camera arrival (NEE covers diffuse arrivals), scaled by
        // this hit's `emit pattern:` factor — the same value the NEE side gets from the
        // sampler at this point (device twin of host emitSlot).
        int li = dEmitterForMat(sc, matId);
        if (specularArrival && dot(rd, h.ng) < 0) {
            // Prefer the registered emitter's baked SPD (it may carry a `power`/`lumens`
            // flux normalisation the raw material spectrum does not); fall back to the
            // material's own `emit` when this geometry has no emitter — an isosurface or
            // CSG solid is marched, never tessellated, so nothing registers for it.
            const double* eSpd = (li >= 0)          ? sc.emitters[li].emitSpd
                               : (mp->matIsLight)   ? mp->matEmit
                                                    : nullptr;
            if (eSpd)
                L += thr * (double)specLookup(eSpd, lambda) * invPdfLambda
                         * dEmitPatMul(sc, mp->emitPat, h);
        }

        if (!bkInteract<GiDepth == 0>(sc, mp, h, matId, diffraction, directOnly, ro, rd, lambda,
                                      invPdfLambda, thr, L, specularArrival, contBsdfPdf, stk,
                                      rng, gi))
            return L;                                   // path terminated in the interaction
    }
    return L;
}

// Hero-wavelength variant of bkRadiance — the device twin of backward.h radianceHeroLoop.
// Carries C wavelengths (hero + C-1 stratified secondaries) down ONE camera path: index 0
// is the hero and drives every sampling decision off the same rng stream a single-λ path
// would, while the secondaries ride the identical vertices and are reweighted per-λ. The
// caller gates this to scenes WITHOUT participating media / GRIN / a physical lens, so those
// branches are absent here. Fills Lout[0..C).
//
// At a dispersive / wavelength-switching material (anything but Diffuse/DiffuseTransmit) the
// bundle cannot keep riding one shared direction, and there are two policies:
//
//   `AllowSplit == false` — DE-HERO: terminate the secondaries and boost the hero ×C so it
//     alone carries an unbiased single-λ estimate onward (PBRT-v4's TerminateSecondary).
//     Cheap, unbiased, but collapses the path onto ONE wavelength.
//   `AllowSplit == true`  — SPLIT-AT-DISPERSION (`-herosplit`, forced on by mode W): fan out
//     into C monochromatic sub-paths, each refracting along its own Snell direction. Mode W
//     REQUIRES this: its λ lattice is shared by every pixel, so a de-hero would collapse the
//     whole FRAME onto one λ and mistint every dispersive surface — 36.7 pp of chroma error,
//     measured (see cudaBackwardWhittedSupported and TODO.md §N).
//
// This is a compile-time switch rather than a runtime one so that nvcc instantiates two
// separate bodies from one source and the `AllowSplit == false` body carries NO recursive
// call at all — no device stack sizing, no -rdc / relocatable-device-code requirement. A
// sub-path is spawned with `secAlive == false` and the split is guarded on `secAlive`, so
// `bkRadianceHeroLoop<false, GiDepth>` is the only re-entry and the split recursion is
// exactly one level deep. (The direct twin of the CPU's radianceHeroLoop, which relies on
// runtime recursion for the same effect.)
//
// `GiDepth` is the second, independent compile-time depth (N3c): 0 = a camera path, whose
// diffuse vertices run the -gi gather, 1 = a gather ray, whose diffuse vertices do NOT (they
// terminate on the flat `bkAmbient` tail instead — that constant is what closes the single
// bounce). So the four instantiations form a DAG, not a cycle:
//
//   <true, 0>  --gather-->  <*, 1>          <true, 0>  --split-->  <false, 0>
//   <false, 0> --gather-->  <*, 1>          <true, 1>  --split-->  <false, 1>
//   <*, 1>     --gather-->  (none)          <false, *> --split-->  (none)
//
// A split sub-path keeps its parent's GiDepth, so a monochromatic sub-path of a camera path
// still gathers — matching the CPU, where a sub-path re-enters radianceHeroLoop with the same
// GiCtx. Deepest chain is <true,0> -> gather -> <true,1> -> split -> <false,1>: three nested
// tracer frames, still fully resolved at compile time and still no -rdc.
//
// `Lout` is ASSIGNED, not accumulated — a split parent therefore adds each sub-path's
// returned radiance into its OWN L[i] slot, which is what keeps wavelength i's radiance
// attributed to wavelength i in the caller's per-λ cieXYZ splat.
template<bool AllowSplit, int GiDepth>
__device__ static void bkRadianceHeroLoop(const DScene& sc, int diffraction,
                                          DVec3 ro, DVec3 rd, DMediumStack stk,
                                          const Real* lamIn, const double* invPdfIn,
                                          const double* thrIn, int C, bool secAlive,
                                          bool specularArrival, double contBsdfPdf,
                                          int bounce0, double* Lout, DRng& rng, DGiCtx gi) {
    Real   lam[hero::kHeroMax];
    double invPdf[hero::kHeroMax], thr[hero::kHeroMax];
    // Copy only the LIVE entries: a monochromatic sub-path spawned by the split fills only
    // slot 0 of its lamIn/invPdfIn/thrIn, so reading all C would read indeterminate values
    // (harmless while nUp == 1 ignores them, but still UB).
    const int nLive = secAlive ? C : 1;
    for (int i = 0; i < nLive; ++i) { lam[i] = lamIn[i]; invPdf[i] = invPdfIn[i]; thr[i] = thrIn[i]; }
    for (int i = nLive; i < C; ++i) { lam[i] = 0;        invPdf[i] = 0.0;         thr[i] = 0.0; }
    for (int i = 0; i < C; ++i) Lout[i] = 0.0;
    double* L = Lout;                                  // accumulate straight into the output
    const bool whitted = (sc.bkWhitted != 0);
    // Gather rays are bounce-capped (see bkGiBounce) so a highly reflective lattice cannot
    // turn one gather direction into a 60-deep ricochet.
    const int maxBounce = (whitted && gi.depth)
                        ? (sc.bkGiBounce < sc.bkMaxBounce ? sc.bkGiBounce : sc.bkMaxBounce)
                        : sc.bkMaxBounce;
    const bool directOnly = (sc.bkDirectOnly != 0);

    for (int b = bounce0; b < maxBounce; ++b) {
        int nUp = secAlive ? C : 1;                    // wavelengths still being propagated
        gi.bounce = b;                                 // see the scalar twin: mode W's per-vertex lattice
        DHit h = closestHit(sc, ro, rd);
        // O8 stage 2 footprint, camera segment only — see the scalar twin. The test is
        // `b == 0`, NOT `b == bounce0`: a heroSplit re-entry resumes at a DEEPER bounce,
        // and that sub-path's first vertex is not a camera vertex.
        if (b == 0 && gi.depth == 0 && h.valid)
            h.fw = (Real)patShadingFootprint(sc.bkFwPerDist, (double)h.t, (double)dot(rd, h.n));

        // Beer-Lambert over the in-glass segment. A non-empty stack implies the bundle already
        // collapsed to one λ here (a dielectric entry either de-heros or splits, and the split
        // clears secAlive too), so nUp == 1 whenever absorption is non-zero; the loop still
        // handles the general case.
        if (h.valid) {
            int cm = stk.topMat();
            if (cm >= 0)
                for (int i = 0; i < nUp; ++i) {
                    Real a = (Real)specLookup(sc.mats[cm].absorb, lam[i]);
                    if (a > 0) thr[i] *= exp(-(double)a * (double)h.t);
                }
        }

        if (!h.valid) {                                // escaped: env radiance per λ
            if (sc.envIndex >= 0) {
                const bool imageEnv = (sc.env.scale != nullptr);
                double wMis = 1.0;
                if (!specularArrival) {                // MIS against the env-NEE at the last vertex
                    double pdfEnv = imageEnv ? dEnvPdf(sc.env, rd) : 1.0 / (4.0 * DPI);
                    wMis = (contBsdfPdf + pdfEnv > 0.0) ? contBsdfPdf / (contBsdfPdf + pdfEnv) : 0.0;
                }
                for (int i = 0; i < nUp; ++i) {
                    double Lenv = (imageEnv ? dEnvRadiance(sc.env, rd, lam[i])
                                            : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lam[i]))
                                  * invPdf[i];
                    L[i] += thr[i] * Lenv * wMis;
                }
            }
            if (sc.sunCount > 0 && specularArrival)     // directly-viewed solar disc
                for (int i = 0; i < nUp; ++i)
                    L[i] += thr[i] * dSunRadiance(sc, rd, lam[i]) * invPdf[i];
            // Escaped GATHER ray -> the far-field `ambient` fill, which is what makes -gi and
            // -ambient compose instead of compete (see the scalar twin bkRadiance).
            if (whitted && gi.depth && sc.bkAmbient > 0.0)
                for (int i = 0; i < nUp; ++i) L[i] += thr[i] * sc.bkAmbient;
            return;
        }

        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (mp->type == D_MIX) {                       // resolve the mix to a child material
            int child = whitted ? dMixResolveDominant(sc, *mp, h)
                                : dMixResolveChild(sc, *mp, h, rng.uniform());
            if (child < 0) return;                      // absorbed
            mp = &sc.mats[child]; matId = child;
        }
        // Surface emission on a specular/camera arrival (NEE covers diffuse arrivals).
        // The `emit pattern:` factor is achromatic, so one eval serves the whole bundle.
        int li = dEmitterForMat(sc, matId);
        if (specularArrival && dot(rd, h.ng) < 0) {
            // Same emitter-then-material fallback as the single-wavelength path above.
            const double* eSpd = (li >= 0)        ? sc.emitters[li].emitSpd
                               : (mp->matIsLight) ? mp->matEmit
                                                  : nullptr;
            if (eSpd) {
                double ep = dEmitPatMul(sc, mp->emitPat, h);
                for (int i = 0; i < nUp; ++i)
                    L[i] += thr[i] * (double)specLookup(eSpd, lam[i]) * invPdf[i] * ep;
            }
        }

        switch (mp->type) {
            case D_DIFFUSETRANSMIT: {
                Real rhoR[hero::kHeroMax], rhoT[hero::kHeroMax];
                for (int i = 0; i < nUp; ++i) {
                    Real rr = clamp01(dDiffuseRho(sc, *mp, h, lam[i]));
                    Real rt = clamp01(dTransmitSlot(sc, *mp, h, lam[i]));
                    Real s = rr + rt;
                    if (s > (Real)1) { rr /= s; rt /= s; }        // per-λ energy guard
                    rhoR[i] = rr; rhoT[i] = rt;
                }
                DVec3 nb = h.n * (Real)(-1);
                bkNeeLightHero(sc, h, rhoR, L, thr, lam, invPdf, nUp, rng, gi.depth);   // front
                if (sc.envIndex >= 0) bkNeeEnvHero(sc, h, rhoR, L, thr, lam, invPdf, nUp, rng);
                DHit hb = h; hb.n = nb;                            // back hemisphere (transmit lobe)
                bkNeeLightHero(sc, hb, rhoT, L, thr, lam, invPdf, nUp, rng, gi.depth);
                if (sc.envIndex >= 0) bkNeeEnvHero(sc, hb, rhoT, L, thr, lam, invPdf, nUp, rng);
                // Mode W indirect diffuse: a translucent surface receives from the FULL
                // sphere, so both lobes take the fill — or, with -gi, each lobe runs its own
                // gather into its own hemisphere (the back lobe off normal-flipped `hb`).
                if (whitted) {
                    bool gathered = false;
                    if constexpr (GiDepth == 0) {
                        if (sc.bkGiDirs > 0) {
                            bkGiGatherHero(sc, diffraction, h,  rhoR, L, thr, lam, invPdf, nUp,
                                           rng, gi);
                            bkGiGatherHero(sc, diffraction, hb, rhoT, L, thr, lam, invPdf, nUp,
                                           rng, gi);
                            gathered = true;
                        }
                    }
                    if (!gathered && sc.bkAmbient > 0.0)
                        for (int i = 0; i < nUp; ++i)
                            L[i] += thr[i] * (double)(rhoR[i] + rhoT[i]) * sc.bkAmbient;
                }
                if (directOnly) return;                            // Whitted: no diffuse indirect
                // Lobe pick + RR over the whole bundle (see D_DIFFUSE): the reflect/transmit
                // probabilities are the per-lobe MAX over live λ, so no secondary is ever
                // amplified. The maxima can sum past 1 (each λ alone is guarded), in which
                // case both shrink proportionally. At nUp == 1 this is the scalar code.
                Real qR = rhoR[0], qT = rhoT[0];
                for (int i = 1; i < nUp; ++i) {
                    if (rhoR[i] > qR) qR = rhoR[i];
                    if (rhoT[i] > qT) qT = rhoT[i];
                }
                Real sumHero = qR + qT;
                if (nUp > 1 && sumHero > (Real)1) { qR /= sumHero; qT /= sumHero; sumHero = qR + qT; }
                Real u = rng.uniform();
                if (u < qR) {                                      // reflect (front)
                    for (int i = 0; i < nUp; ++i) thr[i] *= (double)rhoR[i] / (double)qR;
                    DVec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = fmax(0.0, (double)dot(wOut, h.n)) / DPI;
                    ro = h.p + h.n * RAY_EPS; rd = wOut; specularArrival = false; break;
                } else if (u < sumHero) {                          // transmit (back)
                    for (int i = 0; i < nUp; ++i) thr[i] *= (double)rhoT[i] / (double)qT;
                    DVec3 wOut = cosineHemisphere(nb, rng);
                    contBsdfPdf = fmax(0.0, (double)dot(wOut, nb)) / DPI;
                    ro = h.p + nb * RAY_EPS; rd = wOut; specularArrival = false; break;
                }
                return;                                            // absorbed
            }
            case D_MIRROR: case D_FILTER: case D_GLOSSY: {
                // ACHROMATIC delta lobes (device twin of backward.h): specular, so no NEE
                // and specularArrival stays true, but the outgoing DIRECTION is the same
                // for every λ — a mirror reflects, a gel passes straight through, a glossy
                // lobe is the mirror direction blurred by a λ-independent roughness. So the
                // bundle keeps riding and only the per-λ coefficient differs.
                // The scalar path survives by ANALOG Russian roulette on the hero's own
                // coefficient; rolling that coin on the hero alone would kill live
                // secondaries whenever c_hero == 0 (a Wratten gel is 0 over most of the
                // spectrum), so the survival probability is the MAX over live λ and the
                // survivors reweight by c_i/q. At nUp == 1, q == c[0] and thr[0] *= 1.0,
                // i.e. the scalar code verbatim (same rng draws, same order).
                Real c[hero::kHeroMax];
                double q = 0.0;
                for (int i = 0; i < nUp; ++i) {
                    c[i] = (mp->type == D_FILTER) ? clamp01(dTransmitSlot(sc, *mp, h, lam[i]))
                                                  : clamp01(dReflectSlot(sc, *mp, h, lam[i]));
                    if ((double)c[i] > q) q = (double)c[i];
                }
                if (whitted) {
                    // Deterministic: carry every live λ's coefficient as weight (no coin, no
                    // c_i/q reweight) and stop only once the WHOLE bundle has fallen under the
                    // bailout — a per-λ cutoff would silently de-hero at a gel.
                    double thrMax = 0.0;
                    for (int i = 0; i < nUp; ++i) {
                        thr[i] *= (double)c[i];
                        if (thr[i] > thrMax) thrMax = thr[i];
                    }
                    if (thrMax <= kWhittedCutoff) return;
                } else {
                    if (rng.uniform() >= q) return;                // RR absorb (q == 0 -> always)
                    for (int i = 0; i < nUp; ++i) thr[i] *= (double)c[i] / q;
                }
                if (mp->type == D_MIRROR) {
                    ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n);
                } else if (mp->type == D_FILTER) {
                    ro = h.p + rd * RAY_EPS;                       // direction unchanged
                } else if (whitted) {
                    // Glossy: the lobe off the deterministic lattice (mirror at sample 0).
                    DVec3 o = dWhittedGlossyDir(reflectv(rd, h.n), dMatRoughness(sc, *mp, h),
                                                gi.sIdx, b);
                    if (dot(o, h.n) <= 0) return;
                    ro = h.p + h.n * RAY_EPS; rd = o;
                } else {
                    DVec3 o = sampleGlossy(reflectv(rd, h.n), dMatRoughness(sc, *mp, h), rng);
                    if (dot(o, h.n) <= 0) return;
                    ro = h.p + h.n * RAY_EPS; rd = o;
                }
                specularArrival = true;
                break;
            }
            case D_DIELECTRIC: case D_THINFILM: case D_MULTILAYER:
            case D_GRATING:    case D_HALFMIRROR:
            case D_FLUORESCENT: case D_HAIR: {
                // Dispersive / wavelength-switching: the outgoing direction (and, for a
                // grating/fluorophore, the wavelength itself) depends on λ, so the bundle
                // cannot keep riding one shared direction past this interface. Hair is
                // λ-coupled too: σ_a(λ) steers the lobe-selection pdf, so its sampled
                // direction cannot be shared across the bundle either.
                if constexpr (AllowSplit) {
                if (secAlive && nUp > 1) {
                    // SPLIT-AT-DISPERSION: fan out instead of de-hero'ing. Each secondary
                    // runs the SAME interaction with its OWN λ — refracting along its own
                    // Snell direction / diffracting into its own grating order — and then
                    // continues as an independent monochromatic sub-path from this vertex.
                    // Its radiance lands in L[i], the slot for ITS wavelength, so the
                    // caller's per-λ cieXYZ splat stays correctly attributed.
                    //
                    // No ×C boost anywhere: each of the C wavelengths now carries its own
                    // unboosted estimate and the caller averages them (Lh[i]/C), whereas
                    // de-hero boosts the lone survivor to stand in for all C. Both are
                    // unbiased; this one is simply not collapsed.
                    for (int i = 1; i < nUp; ++i) {
                        if (invPdf[i] == 0.0) continue;   // dead secondary (zero-mass λ bin)
                        // Sub-path state: mutable per-λ copies (bkInteract takes
                        // lambda/invPdf by reference — a fluorescent Stokes shift rewrites
                        // them) and a private medium stack, since sub-paths diverge here.
                        Real   sLam = lam[i];
                        double sInv = invPdf[i], sThr = thr[i], sL = 0.0;
                        bool   sSpec = specularArrival;
                        double sPdf = contBsdfPdf;
                        DMediumStack sStk = stk;
                        DVec3 sRo = ro, sRd = rd;
                        // <false>: the hero tracer handles its own diffuse vertices inline, so
                        // the shared scalar interaction never needs the -gi gather (see the
                        // bkInteract declaration).
                        if (bkInteract<false>(sc, mp, h, matId, diffraction, directOnly, sRo, sRd,
                                              sLam, sInv, sThr, sL, sSpec, sPdf, sStk, rng, gi)) {
                            double sub[hero::kHeroMax];
                            // <false, ...>: a sub-path can never split again, which is what
                            // bounds the split re-entry at one level (see the header comment).
                            // GiDepth is INHERITED, so a sub-path of a camera path still
                            // gathers and a sub-path of a gather ray still does not.
                            bkRadianceHeroLoop<false, GiDepth>(sc, diffraction, sRo, sRd, sStk,
                                                               &sLam, &sInv, &sThr, /*C=*/1,
                                                               /*secAlive=*/false, sSpec, sPdf,
                                                               b + 1, sub, rng, gi);
                            sL += sub[0];
                        }
                        L[i] += sL;      // this wavelength's own estimate, own slot
                        thr[i] = 0.0;    // it is now that sub-path's business, not ours
                    }
                    secAlive = false;    // hero carries on alone, UNBOOSTED
                    if (!bkInteract<false>(sc, mp, h, matId, diffraction, directOnly, ro, rd,
                                           lam[0], invPdf[0], thr[0], L[0], specularArrival,
                                           contBsdfPdf, stk, rng, gi))
                        return;
                    break;
                }
                }
                // Default policy: terminate the secondaries (boosting the hero ×C so the
                // estimate stays unbiased), then run the shared scalar interaction on the
                // hero channel alone. Note bkInteract may itself switch lam[0]/invPdf[0]
                // (a fluorescent Stokes shift) — legal now that index 0 is the only live
                // wavelength.
                if (secAlive) { thr[0] *= (double)C; secAlive = false; }
                if (!bkInteract<false>(sc, mp, h, matId, diffraction, directOnly, ro, rd, lam[0],
                                       invPdf[0], thr[0], L[0], specularArrival, contBsdfPdf, stk,
                                       rng, gi))
                    return;
                break;
            }
            case D_DIFFUSE:
            default: {
                Real rho[hero::kHeroMax];
                for (int i = 0; i < nUp; ++i) rho[i] = clamp01(dDiffuseRho(sc, *mp, h, lam[i]));
                bkNeeLightHero(sc, h, rho, L, thr, lam, invPdf, nUp, rng, gi.depth);
                if (sc.envIndex >= 0) bkNeeEnvHero(sc, h, rho, L, thr, lam, invPdf, nUp, rng);
                // The mode-W indirect-diffuse term. With -gi it is a real single-bounce
                // hemisphere gather (occlusion-aware and spectral); with -gi 0 it falls back to
                // POV-Ray's flat `ambient`, without which a CLOSED room previews with black
                // shadows, since every non-key-lit surface there is lit purely by bounce.
                if (whitted) {
                    bool gathered = false;
                    if constexpr (GiDepth == 0) {
                        if (sc.bkGiDirs > 0) {
                            bkGiGatherHero(sc, diffraction, h, rho, L, thr, lam, invPdf, nUp,
                                           rng, gi);
                            gathered = true;
                        }
                    }
                    if (!gathered && sc.bkAmbient > 0.0)
                        for (int i = 0; i < nUp; ++i)
                            L[i] += thr[i] * (double)rho[i] * sc.bkAmbient;
                }
                if (directOnly) return;                            // Whitted: no diffuse indirect
                // Continuation RR over the WHOLE bundle: survival probability is max_i rho_i,
                // not the hero's own albedo, and every live λ reweights by rho_i/q <= 1.
                // Rolling the coin on the hero alone (thr[i] *= rho_i/rho_0) amplifies a
                // secondary by up to rho_max/rho_hero — on a saturated wall (redWall spans
                // 0.05..0.75) a 15x weight spike. At nUp == 1, q == rho[0] and thr[0] *= 1.0.
                Real q = rho[0];
                for (int i = 1; i < nUp; ++i) if (rho[i] > q) q = rho[i];
                if (rng.uniform() >= q) return;                    // RR absorb
                for (int i = 0; i < nUp; ++i) thr[i] *= (double)rho[i] / (double)q;
                DVec3 wOut = cosineHemisphere(h.n, rng);
                contBsdfPdf = fmax(0.0, (double)dot(wOut, h.n)) / DPI;
                ro = h.p + h.n * RAY_EPS; rd = wOut; specularArrival = false; break;
            }
        }
    }
}

// Entry point for a fresh camera/gather hero bundle: unit throughput, empty medium stack, at
// bounce 0. A camera ray may see a light directly; a GATHER ray may not — the vertex it left
// already NEE'd the direct light, so counting the emitter again here would double it. A
// specular bounce re-arms this, so gold-bounced light still lands.
//
// `sc.bkHeroSplit` picks the dispersive policy, and because it is a warp-uniform scene flag
// the branch costs one predictable jump per path, not a divergent one. Device twin of
// backward.h radianceHero.
template<int GiDepth>
__device__ static void bkRadianceHero(const DScene& sc, int diffraction, DVec3 ro, DVec3 rd,
                                      const Real* lamIn, const double* invPdfIn, int C,
                                      double* Lout, DRng& rng, DGiCtx gi) {
    double thr[hero::kHeroMax];
    for (int i = 0; i < C; ++i) thr[i] = 1.0;
    DMediumStack stk; stk.clear();                     // dielectric priority (Beer-Lambert per λ)
    if (sc.bkHeroSplit)
        bkRadianceHeroLoop<true, GiDepth>(sc, diffraction, ro, rd, stk, lamIn, invPdfIn, thr, C,
                                          /*secAlive=*/(C > 1), /*specularArrival=*/(gi.depth == 0),
                                          /*contBsdfPdf=*/0.0, /*bounce0=*/0, Lout, rng, gi);
    else
        bkRadianceHeroLoop<false, GiDepth>(sc, diffraction, ro, rd, stk, lamIn, invPdfIn, thr, C,
                                           /*secAlive=*/(C > 1), /*specularArrival=*/(gi.depth == 0),
                                           /*contBsdfPdf=*/0.0, /*bounce0=*/0, Lout, rng, gi);
}

// ---- the gather itself (declared above bkInteract) -------------------------------------
// Estimate the cosine-weighted mean INCIDENT radiance over the hemisphere above a diffuse
// vertex by tracing the fixed dGiDir lattice, then add the Lambertian response rho * that.
// This is the term the flat `bkAmbient` was standing in for, computed instead of assumed.
//
// Each gather ray runs the same deterministic Whitted radiance the camera ray does, one
// GiCtx depth further along, which (a) stops it gathering again — single bounce — (b) drops it
// to bkGiGrid shadow rays, (c) makes it terminate its own diffuse vertices on the flat
// bkAmbient tail, and (d) starts it NON-specular so a ray landing straight on a light adds
// nothing (this vertex's own NEE already counted that; adding it here would double the direct
// light). A ray that reaches a light *via* a mirror still counts, because a specular bounce
// re-arms specularArrival — so gold-bounced light, the whole point of this, rides at full
// weight. Device twin of BackwardRenderer::giGatherHero.
//
// Normalising by the REALISED sum of cosines makes the estimator exact for constant incident
// radiance, so in an empty scene every direction escapes, each gather ray returns bkAmbient
// (see the escaped-ray tail in the tracers) and the whole thing collapses back to
// rho * ambient — switching -gi on therefore never steps the exposure.
__device__ static void bkGiGatherHero(const DScene& sc, int diffraction, const DHit& h,
                                      const Real* rho, double* L, const double* thr,
                                      const Real* lam, const double* invPdf, int nUp,
                                      DRng& rng, DGiCtx gi) {
    const DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
    const int n = sc.bkGiDirs * 2;                 // full-sphere lattice; ~half faces outward
    double p1, p2; dGiPhases(gi.sIdx, p1, p2);
    double acc[hero::kHeroMax];
    for (int i = 0; i < nUp; ++i) acc[i] = 0.0;
    double wSum = 0.0;
    const DGiCtx sub{gi.depth + 1, gi.sIdx, 0};
    for (int j = 0; j < n; ++j) {
        const DVec3 d = dGiDir(j, n, p1, p2);
        const double c = (double)dot(h.n, d);
        if (c <= 0.0) continue;
        // Also require the GEOMETRIC hemisphere, or a smoothed shading normal would gather
        // through the true back face (the shading-normal problem again).
        if (dot(ngo, d) <= 0) continue;
        wSum += c;
        double Lg[hero::kHeroMax];
        bkRadianceHero<1>(sc, diffraction, h.p + ngo * RAY_EPS, d, lam, invPdf, nUp, Lg,
                          rng, sub);
        // Firefly clamp (see bkGiClamp). NOT applied to wSum: a clamped direction keeps its
        // weight c, so the estimator still normalises by the realised sum of cosines and an
        // unclamped gather is untouched bit-for-bit.
        if (sc.bkGiClamp > 0.0)
            for (int i = 0; i < nUp; ++i) if (Lg[i] > sc.bkGiClamp) Lg[i] = sc.bkGiClamp;
        for (int i = 0; i < nUp; ++i) acc[i] += c * Lg[i];
    }
    if (wSum <= 0.0) return;
    const double inv = 1.0 / wSum;
    for (int i = 0; i < nUp; ++i) L[i] += thr[i] * (double)rho[i] * (acc[i] * inv);
}

// Scalar twin of bkGiGatherHero, for the paths that cannot use the hero bundle (media, GRIN, a
// physical lens, -heroc 1). Device twin of BackwardRenderer::giGather.
__device__ static double bkGiGather(const DScene& sc, int diffraction, const DHit& h, Real rho,
                                    Real lambda, double invPdfLambda, DRng& rng, DGiCtx gi) {
    const DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
    const int n = sc.bkGiDirs * 2;
    double p1, p2; dGiPhases(gi.sIdx, p1, p2);
    double acc = 0.0, wSum = 0.0;
    const DGiCtx sub{gi.depth + 1, gi.sIdx, 0};
    for (int j = 0; j < n; ++j) {
        const DVec3 d = dGiDir(j, n, p1, p2);
        const double c = (double)dot(h.n, d);
        if (c <= 0.0) continue;
        if (dot(ngo, d) <= 0) continue;
        wSum += c;
        double Lg = bkRadiance<1>(sc, diffraction, h.p + ngo * RAY_EPS, d, lambda, invPdfLambda,
                                  rng, sub);
        if (sc.bkGiClamp > 0.0 && Lg > sc.bkGiClamp) Lg = sc.bkGiClamp;   // see bkGiClamp
        acc += c * Lg;
    }
    return (wSum > 0.0) ? (double)rho * (acc / wSum) : 0.0;
}

// Backward reference megakernel (GPU mode R). Grid-strides over res*res*spp samples;
// each samples a wavelength, generates a camera ray (physical lens or pinhole/fisheye),
// estimates radiance, and accumulates cieXYZ * (L * lensWeight) into the film. The film
// holds the SUM over spp (writeFilm divides by spp), matching renderForwardCuda.
// Renders `chunkSpp` samples-per-pixel for the chunk starting at sample `sampleBase`,
// accumulating (atomicAdd) into `film`/`hits`. The RNG is seeded on the GLOBAL sample
// index (pixel * sppTotal + sampleBase + localSample) so a render split into any number
// of chunks draws exactly the same union of streams as one single-shot pass of sppTotal
// samples — chunked progress is therefore bit-identical to the monolithic render.
// `heroC` > 1 selects the hero-wavelength bundle (bkRadianceHero): one stratified base
// draw yields C wavelengths that share a single BVH walk, each splatting L/C. heroC == 1
// runs the classic single-λ estimator bit-for-bit (the host gates heroC to 1 whenever the
// scene has media / GRIN / a physical lens, which bkRadianceHero does not cover).
__global__ void kBackward(DScene sc, DCamera cam, double* film, double* hits,
                          long long totalSamples, long long chunkSpp, long long sppTotal,
                          long long sampleBase, int resX,
                          int diffraction, unsigned long long seedBase, int heroC) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    for (long long idx = g; idx < totalSamples; idx += G) {
        long long pix = idx / chunkSpp;
        long long gidx = pix * sppTotal + sampleBase + (idx - pix * chunkSpp);
        DRng rng; rng.seed((unsigned long long)(gidx * 2 + 1), seedBase ^ (unsigned long long)gidx);
        int px = (int)(pix % resX);
        int py = (int)(pix / resX);
        size_t o = ((size_t)py * resX + px) * 3;
        // Mode W's ABSOLUTE sample index — the same quantity the host calls sIdx, i.e. WITHOUT
        // the pixel term that `gidx` carries. Every pixel must share it, since sharing the
        // sample offsets is exactly what makes the mode noise-free; and it must be absolute,
        // so the image is independent of how the budget was chunked (a per-chunk index would
        // collapse to "sample 0 forever" under -window, which chunks into 1-spp batches).
        const unsigned long long sIdx =
            (unsigned long long)(sampleBase + (idx - pix * chunkSpp));
        const bool whitted = (sc.bkWhitted != 0);

        if (heroC > 1) {
            // One stratified base draw -> hero + C-1 secondary wavelengths, all from the
            // scene emission CDF (device twin of BackwardRenderer::renderRows). The hero
            // (index 0) must have a valid pdf; a dead secondary carries invPdf 0 and
            // splats nothing. Hero is gated off for a physical lens, so no lens weight.
            Real   lam[hero::kHeroMax];
            double invPdf[hero::kHeroMax];
            // Mode W: the bundle's base coordinate comes off the progressive deterministic
            // sequence instead of the rng, so the C wavelengths land on a FIXED lattice of the
            // emission CDF. Without this the mode would still be noise-free in geometry and
            // shading and yet visibly speckled in COLOUR, because λ was the last random draw.
            double u = whitted ? dWhittedLambdaU(sIdx) : (double)rng.uniform();
            double pdf0 = 0.0;
            lam[0] = dSampleSceneLambdaU(sc, u, pdf0);
            if (pdf0 <= 0.0) continue;
            invPdf[0] = dInvPdfLambda(sc, lam[0]);
            for (int i = 1; i < heroC; ++i) {
                double uu = u + (double)i / heroC;
                if (uu >= 1.0) uu -= 1.0;              // wrap into [0,1)
                double pdfi = 0.0;
                lam[i] = dSampleSceneLambdaU(sc, uu, pdfi);
                invPdf[i] = (pdfi > 0.0) ? dInvPdfLambda(sc, lam[i]) : 0.0;
            }
            DVec3 hro, hrd;
            Real jx, jy;
            if (whitted) { double u1, u2; dWhittedSample(sIdx, u1, u2); jx = (Real)u1; jy = (Real)u2; }
            else         { jx = rng.uniform(); jy = rng.uniform(); }
            dGenRay(cam, px, py, jx, jy, hro, hrd);
            double Lh[hero::kHeroMax];
            // DGiCtx carries the ABSOLUTE sample index down the path, which rotates every
            // deterministic lattice (glossy lobe, gather directions) by it — so, exactly like
            // the subpixel and wavelength lattices, they are progressive and chunk-independent.
            bkRadianceHero<0>(sc, diffraction, hro, hrd, lam, invPdf, heroC, Lh, rng,
                              DGiCtx{0, sIdx, 0});
            for (int i = 0; i < heroC; ++i) {
                double w = Lh[i] / (double)heroC;
                atomicAdd(&film[o + 0], (double)cieX(lam[i]) * w);
                atomicAdd(&film[o + 1], (double)cieY(lam[i]) * w);
                atomicAdd(&film[o + 2], (double)cieZ(lam[i]) * w);
            }
            if (hits) atomicAdd(&hits[(size_t)py * resX + px], 1.0);
            continue;
        }

        double pdf = 0.0;
        // Mode W: stratified λ, as in the hero path above. Note this scalar path carries ONE
        // wavelength per sample, so a deterministic spectral preview here needs spp raised to
        // cover the spectrum (the hero path, which is the usual one, gets heroC per sample).
        Real lambda = whitted ? dSampleSceneLambdaU(sc, dWhittedLambdaU(sIdx), pdf)
                              : dSampleSceneLambda(sc, rng, pdf);
        if (pdf <= 0.0) continue;
        double invPdfLambda = dInvPdfLambda(sc, lambda);

        DVec3 ro, rd;
        double wLens = 1.0;
        if (cam.hasLens) {
            // The physical-lens path stays stochastic even in mode W (host twin: renderRows
            // takes jx/jy/u1/u2 off the rng here), so a lens render is not noise-free.
            Real jx = rng.uniform(), jy = rng.uniform();
            Real u1 = rng.uniform(), u2 = rng.uniform();
            Real wl = 0;
            if (!dGenLensRay(cam, px, py, jx, jy, u1, u2, lambda, ro, rd, wl)) continue;  // vignetted
            wLens = (double)wl;
        } else {
            Real jx, jy;
            if (whitted) { double s1, s2; dWhittedSample(sIdx, s1, s2); jx = (Real)s1; jy = (Real)s2; }
            else         { jx = rng.uniform(); jy = rng.uniform(); }
            dGenRay(cam, px, py, jx, jy, ro, rd);
        }
        double Lval = bkRadiance<0>(sc, diffraction, ro, rd, lambda, invPdfLambda, rng,
                                    DGiCtx{0, sIdx, 0});
        double w = Lval * wLens;
        atomicAdd(&film[o + 0], (double)cieX(lambda) * w);
        atomicAdd(&film[o + 1], (double)cieY(lambda) * w);
        atomicAdd(&film[o + 2], (double)cieZ(lambda) * w);
        if (hits) atomicAdd(&hits[(size_t)py * resX + px], 1.0);
    }
}

// ======================= fast RGB backward (mode R -rgb) =====================
// Option B (gpu-backward-fast.md): a non-spectral backward tracer that carries a
// linear-sRGB throughput triple `beta` (baked per-material RGB albedo) and produces a
// full-colour result in ONE intersection walk per sample — no wavelength dimension, so
// a clean colour image converges far faster than the spectral estimator (mode R). The
// per-emitter/env radiance is baked as the exact wavelength-integrated XYZ->RGB radiance
// (the p(lambda)*invPdfLambda cancellation), so a neutral (spectrally flat) scene matches
// the spectral estimator's absolute luminance; colour picks up the Option-B metamerism
// approximation (no dispersion / thin-film / fluorescence — those stay on mode R / D).
// Representative wavelength for the achromatic specular interfaces (Fresnel, refraction).
#define LREP_RGB ((Real)550)

// Surface NEE, RGB twin of bkNeeLight: connect to every area/sphere/cylinder/spot emitter
// and accumulate the linear-RGB direct contribution (Lambertian BRDF rhoRGB/pi times the
// baked emitter radiance). No wavelength / invPdfLambda term (baked into rgbEmit).
__device__ static DVec3 bkNeeLightRGB(const DScene& sc, const DHit& h, const DVec3& rhoRGB,
                                      DRng& rng) {
    DVec3 total(0, 0, 0);
    DVec3 f = rhoRGB / (Real)DPI;                     // Lambertian BRDF (per channel)
    DVec3 ngo0 = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
    DEmitterDraw draw; dPickEmitters(sc, h.p, &h.n, rng, draw);
    for (int di = 0; di < draw.n; ++di) {
        const int k = draw.emitter(di);
        const double selW = draw.weight(di);           // 1/p(e); exactly 1 on the all path
        if (selW <= 0.0) continue;
        const Real selWr = (Real)selW;
        const DEmitter& em = sc.emitters[k];
        if (em.collimated || em.shape == 3) continue;  // collimated beams / env (env: bkNeeEnvRGB)
        if (em.shape == 2) {                           // point spot
            DVec3 toL = em.origin - h.p;
            Real dist2 = dot(toL, toL);
            Real dist  = sqrt(dist2);
            DVec3 wi = toL / dist;
            Real cosSurf = dot(h.n, wi);
            if (cosSurf <= (Real)0) continue;
            Real stG = dShadowTerminatorG(wi, h.n, ngo0);
            if (stG <= (Real)0) continue;
            Real fall = (Real)spotFalloff(dot(wi * (Real)(-1), em.beamDir), em.spotCosInner, em.spotCosOuter);
            if (fall <= (Real)0) continue;
            if (occluded(sc, h.p + ngo0 * RAY_EPS, wi, dist - (Real)2 * RAY_EPS)) continue;
            total = total + hadamard(f * (fall * cosSurf / dist2 * stG * selWr), em.rgbEmit);
            continue;
        }
        if (em.shape == 6) {                           // distant sun: cone NEE, 1/pdfW = Omega
            double s1 = (double)rng.uniform(), s2 = (double)rng.uniform();
            DVec3 wi = dSunSampleCone(em, em.beamDir * (Real)(-1), s1, s2);
            Real cosSurf = dot(h.n, wi);
            if (cosSurf <= (Real)0) continue;
            Real stG = dShadowTerminatorG(wi, h.n, ngo0);
            if (stG <= (Real)0) continue;
            Real dist = (Real)((double)length(sc.sceneCenter - h.p) + sc.sceneRadius);
            if (occluded(sc, h.p + ngo0 * RAY_EPS, wi, dist)) continue;
            total = total + hadamard(f * (Real)((double)(cosSurf * stG) * em.spotOmega * selW), em.rgbEmit);
            continue;
        }
        Real u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 y, nL;
        // Also returns the sampled point's emission-pattern factor (1.0 when unpatterned).
        // The pattern is achromatic, so it scales the baked RGB radiance directly.
        double epat = dEmitterSamplePointPat(sc, em, (double)u1, (double)u2, y, nL);
        DVec3 toL = y - h.p;
        Real dist2 = dot(toL, toL);
        Real dist = sqrt(dist2);
        DVec3 wi = toL / dist;
        Real cosSurf = dot(h.n, wi);
        if (cosSurf <= 0) continue;
        Real stG = dShadowTerminatorG(wi, h.n, ngo0);
        if (stG <= (Real)0) continue;
        Real cosLight = dot(nL, -wi);
        if (cosLight <= 0) continue;
        if (occluded(sc, h.p + ngo0 * RAY_EPS, wi, dist - (Real)2 * RAY_EPS)) continue;
        Real G = cosSurf * cosLight / dist2;
        if (epat != 1.0) G = (Real)((double)G * epat);   // no-op without a pattern
        total = total + hadamard(f * (G * em.area * stG * selWr), em.rgbEmit);
    }
    return total;
}

// Constant-env NEE, RGB twin of bkNeeEnv: one uniform-sphere sample (pdf 1/4pi), MIS'd
// (balance heuristic) against the cosine continuation, returns the linear-RGB contribution.
__device__ static DVec3 bkNeeEnvRGB(const DScene& sc, const DHit& h, const DVec3& rhoRGB,
                                    DRng& rng) {
    if (sc.envIndex < 0) return DVec3(0, 0, 0);
    double z = 1.0 - 2.0 * (double)rng.uniform();
    double sr = sqrt(fmax(0.0, 1.0 - z * z));
    double phi = 2.0 * DPI * (double)rng.uniform();
    DVec3 wi{(Real)(sr * cos(phi)), (Real)(sr * sin(phi)), (Real)z};
    double pdfW = 1.0 / (4.0 * DPI);
    Real cosSurf = dot(h.n, wi);
    if (cosSurf <= (Real)0) return DVec3(0, 0, 0);
    DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
    Real stG = dShadowTerminatorG(wi, h.n, ngo);
    if (stG <= (Real)0) return DVec3(0, 0, 0);
    double farDist = (double)length(sc.sceneCenter - h.p) + sc.sceneRadius;
    if (occluded(sc, h.p + ngo * RAY_EPS, wi, (Real)farDist)) return DVec3(0, 0, 0);
    double pdfBsdf = (double)cosSurf / DPI;
    double wMis = pdfW / (pdfW + pdfBsdf);
    Real k = (Real)((double)cosSurf / pdfW * wMis * (double)stG);
    return hadamard(rhoRGB / (Real)DPI, sc.rgbEnv) * k;
}

// RGB backward radiance along a camera ray (Option B). Carries a linear-sRGB throughput
// and accumulates linear-sRGB radiance; returns L (the kernel converts to XYZ for the
// film). Scope (cudaBackwardRGBSupported): Lambertian (constant-albedo) + diffuse-transmit
// + mirror/glossy/half-mirror/filter + non-dispersive dielectric + stochastic mix, area/
// sphere/cylinder/spot lights and a constant env. Media / fluorescence / thin-film /
// multilayer / grating / textured albedo / image-env fall back to the spectral tracer.
__device__ static DVec3 bkRadianceRGB(const DScene& sc, int diffraction, DVec3 ro, DVec3 rd,
                                      DRng& rng) {
    DVec3 L(0, 0, 0), beta(1, 1, 1);
    bool specularArrival = true;
    double contBsdfPdf = 0.0;
    DMediumStack stk; stk.clear();
    const int maxBounce = sc.bkMaxBounce;
    const bool directOnly = (sc.bkDirectOnly != 0);
    for (int b = 0; b < maxBounce; ++b) {
        DHit h = closestHit(sc, ro, rd);
        if (!h.valid) {                                // escaped -> constant env
            if (sc.envIndex >= 0) {
                if (specularArrival) {
                    L = L + hadamard(beta, sc.rgbEnv);
                } else {
                    double pdfEnv = 1.0 / (4.0 * DPI);
                    double wMis = (contBsdfPdf + pdfEnv > 0.0)
                                      ? contBsdfPdf / (contBsdfPdf + pdfEnv) : 0.0;
                    L = L + hadamard(beta, sc.rgbEnv) * (Real)wMis;
                }
            }
            // Directly-viewed solar disc (camera / specular arrivals only, as in the
            // spectral walk). rgbEmit already carries the sun's wavelength-integrated
            // radiance, so no per-λ term is needed here.
            if (sc.sunCount > 0 && specularArrival)
                for (int k = 0; k < sc.nEmitters; ++k) {
                    const DEmitter& e = sc.emitters[k];
                    if (e.shape == 6 && dInSunCone(e, rd)) L = L + hadamard(beta, e.rgbEmit);
                }
            return L;
        }
        // Beer-Lambert attenuation over the in-glass segment (3-tap RGB sigma_a).
        {
            int cm = stk.topMat();
            if (cm >= 0) {
                DVec3 a = sc.mats[cm].rgbAbsorb;
                if (a.x > 0 || a.y > 0 || a.z > 0)
                    beta = hadamard(beta, DVec3(exp(-(double)a.x * (double)h.t),
                                                exp(-(double)a.y * (double)h.t),
                                                exp(-(double)a.z * (double)h.t)));
            }
        }
        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (mp->type == D_MIX) {
            int child = dMixResolveChild(sc, *mp, h, rng.uniform());
            if (child < 0) return L;
            mp = &sc.mats[child]; matId = child;
        }
        int li = dEmitterForMat(sc, matId);
        if (specularArrival && dot(rd, h.ng) < 0 && (li >= 0 || mp->matIsLight)) {
            // The emission pattern is achromatic, so it scales the baked RGB radiance.
            // Emitter-less emissive geometry (isosurface / CSG solid) falls back to the
            // material's own baked emission, exactly as the spectral paths do.
            double ep = dEmitPatMul(sc, mp->emitPat, h);
            L = L + hadamard(beta * (Real)ep,
                             (li >= 0) ? sc.emitters[li].rgbEmit : mp->rgbMatEmit);
        }

        switch (mp->type) {
            case D_DIELECTRIC: {
                DVec3 nro, nrd; dDielectricStep(sc, *mp, h, rd, LREP_RGB, rng, matId, stk, nro, nrd);
                ro = nro; rd = nrd; specularArrival = true; break;
            }
            case D_MIRROR: {
                Real q = rgbLuma(mp->rgbAlbedo);
                if (q <= (Real)0 || rng.uniform() >= (double)q) return L;
                beta = hadamard(beta, mp->rgbAlbedo) / q;
                ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); specularArrival = true; break;
            }
            case D_HALFMIRROR: {
                Real r = clamp01(rgbLuma(mp->rgbAlbedo));
                if (rng.uniform() < (double)r) { ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); }
                else                           { ro = h.p + rd * RAY_EPS; }
                specularArrival = true; break;
            }
            case D_FILTER: {
                Real q = rgbLuma(mp->rgbTransmit);
                if (q <= (Real)0 || rng.uniform() >= (double)q) return L;
                beta = hadamard(beta, mp->rgbTransmit) / q;
                ro = h.p + rd * RAY_EPS; specularArrival = true; break;
            }
            case D_GLOSSY: {
                Real q = rgbLuma(mp->rgbAlbedo);
                if (q <= (Real)0 || rng.uniform() >= (double)q) return L;
                DVec3 o = sampleGlossy(reflectv(rd, h.n), dMatRoughness(sc, *mp, h), rng);
                if (dot(o, h.n) <= 0) return L;
                beta = hadamard(beta, mp->rgbAlbedo) / q;
                ro = h.p + h.n * RAY_EPS; rd = o; specularArrival = true; break;
            }
            case D_DIFFUSETRANSMIT: {
                DVec3 rhoR = clampRgb01(mp->rgbAlbedo);
                DVec3 rhoT = clampRgb01(mp->rgbTransmit);
                Real pR = rgbLuma(rhoR), pT = rgbLuma(rhoT);
                Real s = pR + pT;
                if (s > (Real)1) { pR /= s; pT /= s; }     // energy guard on the selection probs
                DVec3 nb = h.n * (Real)(-1);
                L = L + hadamard(beta, bkNeeLightRGB(sc, h, rhoR, rng));
                if (sc.envIndex >= 0) L = L + hadamard(beta, bkNeeEnvRGB(sc, h, rhoR, rng));
                DHit hb = h; hb.n = nb;
                L = L + hadamard(beta, bkNeeLightRGB(sc, hb, rhoT, rng));
                if (sc.envIndex >= 0) L = L + hadamard(beta, bkNeeEnvRGB(sc, hb, rhoT, rng));
                if (directOnly) return L;                   // Whitted: no diffuse indirect
                Real u = rng.uniform();
                if (u < pR) {
                    beta = hadamard(beta, rhoR) / pR;
                    DVec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = fmax(0.0, (double)dot(wOut, h.n)) / DPI;
                    ro = h.p + h.n * RAY_EPS; rd = wOut; specularArrival = false; break;
                } else if (u < pR + pT) {
                    beta = hadamard(beta, rhoT) / pT;
                    DVec3 wOut = cosineHemisphere(nb, rng);
                    contBsdfPdf = fmax(0.0, (double)dot(wOut, nb)) / DPI;
                    ro = h.p + nb * RAY_EPS; rd = wOut; specularArrival = false; break;
                }
                return L;                                   // absorbed
            }
            case D_DIFFUSE:
            default: {
                DVec3 rho = clampRgb01(mp->rgbAlbedo);
                L = L + hadamard(beta, bkNeeLightRGB(sc, h, rho, rng));
                if (sc.envIndex >= 0)
                    L = L + hadamard(beta, bkNeeEnvRGB(sc, h, rho, rng));
                if (directOnly) return L;                                   // Whitted: no diffuse indirect
                Real q = rgbLuma(rho);
                if (q <= (Real)0 || rng.uniform() >= (double)q) return L;   // RR on luminance
                beta = hadamard(beta, rho) / q;
                DVec3 wOut = cosineHemisphere(h.n, rng);
                contBsdfPdf = fmax(0.0, (double)dot(wOut, h.n)) / DPI;
                ro = h.p + h.n * RAY_EPS; rd = wOut; specularArrival = false; break;
            }
        }
    }
    return L;
}

// Fast RGB backward megakernel (mode R -rgb). Same grid-stride / seeding scheme as
// kBackward, but each sample does ONE colour walk and deposits the linear-RGB radiance
// (converted to XYZ) into the film. The specular interfaces use a fixed representative
// wavelength (LREP_RGB); the camera lens ray uses it too (RGB drops lens dispersion).
__global__ void kBackwardRGB(DScene sc, DCamera cam, double* film, double* hits,
                             long long totalSamples, long long chunkSpp, long long sppTotal,
                             long long sampleBase, int resX,
                             int diffraction, unsigned long long seedBase) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    for (long long idx = g; idx < totalSamples; idx += G) {
        long long pix = idx / chunkSpp;
        long long gidx = pix * sppTotal + sampleBase + (idx - pix * chunkSpp);
        DRng rng; rng.seed((unsigned long long)(gidx * 2 + 1), seedBase ^ (unsigned long long)gidx);
        int px = (int)(pix % resX);
        int py = (int)(pix / resX);

        DVec3 ro, rd;
        double wLens = 1.0;
        if (cam.hasLens) {
            Real jx = rng.uniform(), jy = rng.uniform();
            Real u1 = rng.uniform(), u2 = rng.uniform();
            Real wl = 0;
            if (!dGenLensRay(cam, px, py, jx, jy, u1, u2, LREP_RGB, ro, rd, wl)) continue;  // vignetted
            wLens = (double)wl;
        } else {
            Real jx = rng.uniform(), jy = rng.uniform();
            dGenRay(cam, px, py, jx, jy, ro, rd);
        }
        DVec3 Lrgb = bkRadianceRGB(sc, diffraction, ro, rd, rng);
        DVec3 xyz = dRgbToXyz(Lrgb * (Real)wLens);
        size_t o = ((size_t)py * resX + px) * 3;
        atomicAdd(&film[o + 0], (double)xyz.x);
        atomicAdd(&film[o + 1], (double)xyz.y);
        atomicAdd(&film[o + 2], (double)xyz.z);
        if (hits) atomicAdd(&hits[(size_t)py * resX + px], 1.0);
    }
}

// ---- G2: deterministic primary-ray isosurface PREVIEW kernel -----------------
// A GPU sibling of the CPU solid rasterizer (raster.h) that renders implicit
// isosurfaces (and any other bakeable geometry) WITHOUT tessellation: it casts one
// deterministic pixel-centre primary ray per pixel through the pinhole/fisheye
// camera, finds the nearest surface with the shared closestHit (which sphere-traces
// implicits via intersectImplicit), and shades it once with the SAME preview model
// the CPU raster uses — flat per-material albedo lit by `ambient + keyScale·Σ(w·N·L·
// atten·cone) + fill·N·V`. The linear-RGB result + depth/emitter masks are downloaded
// and run through raster::exposeAndEncode on the host, so the tone map and a
// camera_path's shared auto-exposure anchor are bit-identical to the CPU preview.
struct DPLight {
    DVec3  pos, dir;
    int    spot;
    double cosInner, cosOuter, weight, falloff2;
};
struct DPreviewLight {
    const DPLight* lights; int nLights;
    double ambient, keyScale, fill;
};
// Device twin of scene.h spotFalloff (smoothstep penumbra between the cone cosines).
__device__ static inline double dSpotFalloff(double ct, double cosInner, double cosOuter) {
    if (ct >= cosInner) return 1.0;
    if (ct <= cosOuter) return 0.0;
    double t = (ct - cosOuter) / (cosInner - cosOuter);
    return t * t * (3.0 - 2.0 * t);
}
// ---- image-skin (linear-RGB) sampling for the iso preview -----------------
// Device twins of raster.h's Texture::sampleRgb / sampleRgbTriplanar (the CPU
// rasterizer's textured-preview path). All bound skins' texels are flattened into
// one shared linear-RGB array; each texture's DPTex gives dims/filter/wrap and its
// first-texel offset. Procedural (formula) skins bake to `rgb` at load (E1), so this
// one path covers both image and formula skins. Kept a private twin of raster_cuda.cu's
// dSampleRgb because that sampler lives in a separate translation unit.
struct DPTex { int w, h, filter, wrap, offset, valid; };

__device__ static inline int dSkinWrap(int i, int n, int wrap) {
    if (wrap == 1) return (i < 0) ? 0 : (i >= n ? n - 1 : i);        // clamp
    if (wrap == 2) {                                                 // mirror
        int period = 2 * n;
        int m = ((i % period) + period) % period;
        return (m < n) ? m : (period - 1 - m);
    }
    int m = i % n; return (m < 0) ? m + n : m;                       // repeat
}

__device__ static DVec3 dSkinRgb(const DPTex* meta, const DVec3* texels, int ti,
                                 double u, double v) {
    const DPTex& t = meta[ti];
    if (!t.valid) return DVec3(0.5, 0.5, 0.5);
    const DVec3* px = texels + t.offset;
    if (t.filter == 0) {   // nearest
        int x = dSkinWrap((int)floor(u * t.w), t.w, t.wrap);
        int y = dSkinWrap((int)floor((1.0 - v) * t.h), t.h, t.wrap);
        return px[(size_t)y * t.w + x];
    }
    double tu = u * t.w - 0.5, tv = (1.0 - v) * t.h - 0.5;
    double flx = floor(tu), fly = floor(tv);
    double fx = tu - flx, fy = tv - fly;
    int x0 = dSkinWrap((int)flx, t.w, t.wrap), x1 = dSkinWrap((int)flx + 1, t.w, t.wrap);
    int y0 = dSkinWrap((int)fly, t.h, t.wrap), y1 = dSkinWrap((int)fly + 1, t.h, t.wrap);
    DVec3 c00 = px[(size_t)y0 * t.w + x0], c10 = px[(size_t)y0 * t.w + x1];
    DVec3 c01 = px[(size_t)y1 * t.w + x0], c11 = px[(size_t)y1 * t.w + x1];
    DVec3 a = c00 * (1.0 - fx) + c10 * fx;
    DVec3 b = c01 * (1.0 - fx) + c11 * fx;
    return a * (1.0 - fy) + b * fy;
}

__device__ static DVec3 dSkinTri(const DPTex* meta, const DVec3* texels, int ti,
                                 const DVec3& p, const DVec3& n, double scale) {
    double ax = fabs(n.x), ay = fabs(n.y), az = fabs(n.z);
    double wx = ax*ax*ax*ax, wy = ay*ay*ay*ay, wz = az*az*az*az;
    double s = wx + wy + wz;
    if (s <= 0.0) return dSkinRgb(meta, texels, ti, p.x * scale, p.y * scale);
    wx /= s; wy /= s; wz /= s;
    DVec3 c(0, 0, 0);
    if (wx > 0.0) c = c + dSkinRgb(meta, texels, ti, p.z * scale, p.y * scale) * wx;
    if (wy > 0.0) c = c + dSkinRgb(meta, texels, ti, p.x * scale, p.z * scale) * wy;
    if (wz > 0.0) c = c + dSkinRgb(meta, texels, ti, p.x * scale, p.y * scale) * wz;
    return c;
}

__global__ void kIsoPreview(DScene sc, DCamera cam, DPreviewLight pl,
                            const DVec3* matCol, const int* matEmit, int nMats,
                            const DPTex* texMeta, const DVec3* texels,
                            const int* matTex, const double* matTri,
                            double* accum, float* zbuf, unsigned char* emis,
                            int W, int H, DVec3 bg, double emisBoost) {
    int total = W * H;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        int px = idx % W, py = idx / W;
        // accum row 0 is the image TOP (raster::exposeAndEncode convention), but dGenRay
        // maps py=0 to sy=-1 (the image BOTTOM), so flip the camera row to match.
        int camPy = H - 1 - py;
        DVec3 ro, rd;
        dGenRay(cam, px, camPy, (Real)0.5, (Real)0.5, ro, rd);   // pixel-centre primary ray
        DHit h = closestHit(sc, ro, rd);
        size_t o = (size_t)idx * 3;
        if (!h.valid) {
            accum[o + 0] = bg.x; accum[o + 1] = bg.y; accum[o + 2] = bg.z;
            zbuf[idx] = 0.0f; emis[idx] = 0;
            continue;
        }
        DVec3 col = (h.matId >= 0 && h.matId < nMats) ? matCol[h.matId] : DVec3(0.6, 0.6, 0.6);
        int   em  = (h.matId >= 0 && h.matId < nMats) ? matEmit[h.matId] : 0;
        zbuf[idx] = (float)h.t;
        if (em) {   // emitter: raw tinted glow, boosted so it clips to white (matches CPU)
            accum[o + 0] = col.x * emisBoost;
            accum[o + 1] = col.y * emisBoost;
            accum[o + 2] = col.z * emisBoost;
            emis[idx] = 1;
            continue;
        }
        emis[idx] = 0;
        // Image/formula skin: replace the flat material albedo with the texture's linear
        // RGB, sampled at the interpolated (u,v) or by world triplanar (device twin of
        // raster.h's textured-preview path; matTex encodes raster.h's binding rule).
        int ti = (matTex && h.matId >= 0 && h.matId < nMats) ? matTex[h.matId] : -1;
        if (ti >= 0) {
            double tri = matTri[h.matId];
            col = (tri > 0.0) ? dSkinTri(texMeta, texels, ti, h.p, h.n, tri)
                              : dSkinRgb(texMeta, texels, ti, h.u, h.v);
        }
        // Two-sided preview, oriented from the GEOMETRIC normal.
        //
        // `intersectTri` orients hit.n by the *interpolated* normal
        // (`flipped = !(dot(rd, ns) < 0)`), and on a smooth-shaded mesh that grazes
        // through zero in a ~1-px band at every silhouette while the triangle is still
        // front-facing. Flipping there inverts N·L with it, so one pixel drops to
        // ambient while its neighbour lights up from the far side — a stipple of light
        // and dark specks tracing every outline. Re-testing dot(N,V) here only repeated
        // the same mistake. raster.h hit this exact bug and fixed it the same way (see
        // the note in its shade loop, which decides the flip once per triangle at
        // projection time); the path tracer's `ngo` likewise derives the geometric
        // normal for the shading side rather than trusting a grazing shading normal.
        //
        // Invisible until a mesh actually carries smooth normals across a silhouette,
        // which is why loom's analytic sweep normals surfaced it: flat/creased facets
        // step across the graze band instead of sliding through it.
        DVec3 N  = normalize(h.n);
        DVec3 Ng = normalize(h.ng);
        DVec3 V  = normalize(cam.eye - h.p);
        if (dot(N, Ng) < 0) N = -N;        // undo a silhouette-graze flip
        if (dot(Ng, V) < 0) N = -N;        // genuine backface: shade the side we see
        double lit = 0.0;
        for (int k = 0; k < pl.nLights; ++k) {
            const DPLight& lp = pl.lights[k];
            DVec3 d = lp.pos - h.p;
            double dist2 = dot(d, d);
            DVec3 Ld = (dist2 > 1e-12) ? d * (Real)(1.0 / sqrt(dist2)) : V;
            double ndl = fmax(0.0, (double)dot(N, Ld));
            if (ndl <= 0.0) continue;
            double atten = 1.0;
            if (lp.falloff2 > 0.0) atten = lp.falloff2 / (lp.falloff2 + dist2);
            double cone = 1.0;
            if (lp.spot) cone = dSpotFalloff((double)dot(lp.dir, -Ld), lp.cosInner, lp.cosOuter);
            lit += lp.weight * ndl * atten * cone;
        }
        double head = fmax(0.0, (double)dot(N, V));           // camera headlight fill
        double kk = pl.ambient + pl.keyScale * lit + pl.fill * head;
        accum[o + 0] = col.x * kk; accum[o + 1] = col.y * kk; accum[o + 2] = col.z * kk;
    }
}

// Continue a subpath whose endpoint is already path[0]; append surface vertices
// until a miss/absorption/maxDepth. Direct port of bdpt.h randomWalk.
// `importance` marks the LIGHT (particle) subpath: only then is the Veach adjoint
// shading-normal correction applied at each non-specular vertex (mode==Importance in
// bdpt.h). The eye (Radiance) subpath smooth-shades for free and passes false.
//
// Hero-wavelength bundle: `hb` supplies the C wavelengths, `betaSecIn`/`nUpIn` the incoming
// secondary throughputs (nUpIn == 1 for a plain single-λ walk, which takes exactly the
// original code path — every added loop has an empty trip count). Only the THROUGHPUT is
// per-λ; every direction, pdf, Russian-roulette draw and MIS density comes from the hero,
// so each secondary reweights by the ratio of its own scattering albedo to the hero's. At a
// delta (dispersive / wavelength-switching) interface the secondaries can no longer follow
// the hero's refracted direction, so the bundle DE-HEROS: nUp drops to 1 for this and every
// later vertex. Secondary throughputs are written into pathSec[vertexIndex*secStride + i].
__device__ static void dRandomWalk(const DScene& sc, const DCamera& cam, int diffraction,
                                   DVec3 ro, DVec3 rd, double beta, double pdfDir,
                                   const DHeroBundle& hb, int maxDepth, DRng& rng,
                                   DVertex* path, double* pathSec, int secStride, int maxV, int& n,
                                   bool importance, const double* betaSecIn, int nUpIn,
                                   DEscape* esc = nullptr) {
    const Real lambda = hb.lam[0];   // the hero drives geometry, sampling and every pdf
    if (maxDepth == 0) return;
    double pdfFwd = pdfDir;
    // Live secondary throughputs. betaSec[i] tracks wavelength hb.lam[i+1].
    double betaSec[BDPT_NSEC];
    int nUp = nUpIn < 1 ? 1 : nUpIn;
    for (int i = 0; i + 1 < nUp; ++i) betaSec[i] = betaSecIn[i];
    DMediumStack stk; stk.clear();   // nested-dielectric medium stack for exterior-IOR resolution
    for (int bounces = 0;;) {
        DHit h = closestHit(sc, ro, rd);
        if (h.valid && h.sensorId >= 0) return;
        double dSurf = h.valid ? (double)h.t : 1e30;

        // Participating media: sample the earliest real collision up to the surface
        // (or 1e30 in open space). Homogeneous free-flight — its transmittance is
        // implicit in the exponential so beta is unchanged (analog MC). No media => no
        // RNG draw, so vacuum walks stay bit-identical. Mirrors bdpt.h randomWalk.
        double tMed = 0.0; bool mediumEvent = false; int scatterMed = -1;
        if (sc.mediaN > 0) {
            Real tm; int which;
            if (dMediaSampleCollision(sc, ro, rd, (Real)dSurf, lambda, rng, tm, which)) {
                tMed = (double)tm; mediumEvent = true; scatterMed = which;
            }
        }

        // Beer-Lambert attenuation over the in-glass segment just traversed, up to the
        // event (surface hit OR medium collision, whichever is nearer). Mirrors CPU
        // bdpt.h randomWalk: attenuates only the subpath walk; connection edges that
        // cross glass are NOT absorption-weighted (see known-issues.md).
        {
            int cm = stk.topMat();
            double a = (cm >= 0) ? (double)specLookup(sc.mats[cm].absorb, lambda) : 0.0;
            if (a > 0.0) beta *= exp(-a * (mediumEvent ? tMed : dSurf));
            // Per-λ absorption for the bundle. A non-empty stack means we are inside a
            // dielectric, and entering one de-heros — so nUp is always 1 whenever `a` can be
            // non-zero and this loop never actually runs. Kept for generality.
            if (cm >= 0) for (int i = 0; i + 1 < nUp; ++i) {
                double ai = (double)specLookup(sc.mats[cm].absorb, hb.lam[i + 1]);
                if (ai > 0.0) betaSec[i] *= exp(-ai * (mediumEvent ? tMed : dSurf));
            }
        }

        // Medium collision precedes the surface: append a volume in-scatter vertex, then
        // scatter (prob = albedo) or absorb. Throughput unchanged on scatter (HG sampling
        // pdf == phase value, analog MC). Stored area densities are cosine-free and carry
        // only the phase direction density; the free-flight distance pdf and transmittance
        // are omitted here AND in dVertexPdf, so they cancel pairwise in every MIS ratio.
        if (mediumEvent) {
            if (n >= maxV) return;
            const DMedium& sm = sc.media[scatterMed];
            DVec3 mpos = ro + rd * (Real)tMed;
            int prevIdx = n - 1;
            DVertex v;
            v.type = BV_MEDIUM; v.p = mpos; v.ns = rd; v.ng = rd;
            v.beta = beta; v.pdfFwd = 0; v.pdfRev = 0; v.delta = 0;
            v.matId = -1; v.lightIdx = -1; v.emitPatW = (Real)1;
            v.mediumG = sm.g; v.mediumId = scatterMed;
            // Hero is gated off for scenes with media, so nUp is 1 here in practice.
            v.nUp = nUp;
            v.pdfFwd = dConvertDensity(pdfFwd, path[prevIdx], v);
            path[n] = v; int cur = n; n++;
            for (int i = 0; i + 1 < nUp; ++i) pathSec[cur * secStride + i] = betaSec[i];
            if (++bounces >= maxDepth) return;
            if (rng.uniform() >= (double)medAlbedo(sm, lambda)) return;   // absorbed (vertex retained)
            DVec3 wo = normalize(path[prevIdx].p - path[cur].p);          // toward previous vertex
            Real phPdf;
            DVec3 wi = dMedPhaseSample(sm, rd, lambda, rng, phPdf);       // scattered dir (HG or rainbow)
            double pdfW    = dPhasePdf(sc, path[cur], wo, wi, lambda);
            double pdfRevW = dPhasePdf(sc, path[cur], wi, wo, lambda);
            path[prevIdx].pdfRev = dConvertDensity(pdfRevW, path[cur], path[prevIdx]);
            ro = mpos; rd = normalize(wi);
            pdfFwd = pdfW;
            continue;
        }

        if (!h.valid) {                              // escaped (no env in BDPT scope)
            if (esc) {
                esc->escaped = 1; esc->dir = rd; esc->beta = beta; esc->nUp = nUp;
                for (int i = 0; i + 1 < nUp; ++i) esc->betaSec[i] = betaSec[i];
            }
            return;
        }

        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (mp->type == D_MIX) {
            int child = dMixResolveChild(sc, *mp, h, rng.uniform());   // honours per-hit blend mask
            if (child < 0) return;
            mp = &sc.mats[child]; matId = child;
        }
        if (n >= maxV) return;
        DVertex v;
        v.type = BV_SURFACE; v.p = h.p; v.ns = h.n; v.ng = h.ng;
        v.beta = beta; v.pdfFwd = 0; v.pdfRev = 0; v.delta = 0;
        v.matId = matId; v.lightIdx = dEmitterForMat(sc, matId);
        // Emission-on-hit half of the slot, evaluated once here where the DHit is in hand
        // (dVertexLe is called later from several MIS strategies with no hit available).
        // Host twin: bdpt.h's slotPatMul at the same hit.
        v.emitPatW = (mp->emitPat >= 0) ? (Real)dEmitPatMul(sc, mp->emitPat, h) : (Real)1;
        v.mediumG = 0.0; v.mediumId = -1;
        v.u = h.u; v.v = h.v; v.curv = h.curv;   // per-hit texcoords + curvature for textured/patterned/record BSDF eval (M9, O3)
        v.nUp = nUp;
        v.pdfFwd = dConvertDensity(pdfFwd, path[n - 1], v);
        path[n] = v; int cur = n; n++;
        for (int i = 0; i + 1 < nUp; ++i) pathSec[cur * secStride + i] = betaSec[i];
        if (++bounces >= maxDepth) return;

        DVec3 wo = normalize(path[cur - 1].p - path[cur].p);
        DVec3 wi; double pdfW = 0, pdfRevW = 0, betaFactor = 0; int delta = 0; bool terminate = false;
        // Mirror / Filter are delta but choose their continuation WITHOUT consulting λ,
        // so the secondaries can ride through them; they set keepBundle to opt out of
        // the `if (delta) nUp = 1` collapse below (device twin of bdpt.h).
        bool keepBundle = false;
        // Hero bundle: per-secondary throughput factor secF[i] = f_{i+1}·cos/pdf for the
        // lobe the hero actually sampled (pdf is always the hero's). ABSOLUTE, not a ratio
        // to the hero's — a ratio is undefined exactly where it matters most, a chromatic
        // lobe whose hero value is 0 while a secondary's is not. Wavelength-INDEPENDENT
        // cases leave `secChromatic` false and reuse `betaFactor`. Ignored when nUp == 1.
        double secF[BDPT_NSEC];
        bool secChromatic = false;
        switch (mp->type) {
            case D_DIFFUSE:
            case D_FLUORESCENT: {
                wi = cosineHemisphere(path[cur].ns, rng);
                if (dot(wi, path[cur].ns) <= 0) { terminate = true; break; }
                double rho = clamp01(dDiffuseRho(sc, *mp, h, lambda));   // per-hit (tex/pat/record)
                pdfW = dBsdfPdf(sc, path[cur], wo, wi, lambda);
                pdfRevW = dBsdfPdf(sc, path[cur], wi, wo, lambda);
                betaFactor = rho;
                secChromatic = true;                  // rho <= 0 is caught by the max test
                for (int i = 0; i + 1 < nUp; ++i)
                    secF[i] = clamp01(dDiffuseRho(sc, *mp, h, hb.lam[i + 1]));
                break;
            }
            case D_GLOSSY: {
                DVec3 mdir = reflectv(rd, path[cur].ns);   // rd == -wo (incoming dir)
                wi = sampleGlossy(mdir, dMatRoughness(sc, *mp, h), rng);   // per-hit roughness
                if (dot(wi, path[cur].ns) <= 0) { terminate = true; break; }
                double r = clamp01(dReflectSlot(sc, *mp, h, lambda));      // per-hit reflect
                pdfW = dBsdfPdf(sc, path[cur], wo, wi, lambda);
                pdfRevW = dBsdfPdf(sc, path[cur], wi, wo, lambda);
                betaFactor = r;
                if (pdfW <= 0) terminate = true;      // r <= 0 is caught by the max test
                // The glossy LOBE (mirror direction + roughness exponent) carries no
                // wavelength dependence, so the whole bundle follows the sampled direction
                // and only the reflectance differs per λ. (The unidirectional hero tracers
                // de-hero here instead — see known-issues.md.)
                secChromatic = true;
                for (int i = 0; i + 1 < nUp; ++i)
                    secF[i] = clamp01(dReflectSlot(sc, *mp, h, hb.lam[i + 1]));
                break;
            }
            case D_DIFFUSETRANSMIT: {
                // Two-lobe Lambertian: pick reflect (+ns) with prob rhoR/(rhoR+rhoT) or
                // transmit (-ns) with prob rhoT/(rhoR+rhoT); cosine-sample the chosen
                // hemisphere. Analog: on a scatter beta is unchanged (f*|cos|/pdf == 1 with
                // the lobe-selection pdf), like the diffuse case. Device twin of render.h.
                double rhoR, rhoT; dDiffuseTransmitAlbedos(sc, *mp, h, lambda, rhoR, rhoT);
                double tot = rhoR + rhoT;
                if (tot <= 0) { terminate = true; break; }
                DVec3 nb = path[cur].ns * (Real)(-1);
                const bool reflLobe = (rng.uniform() < rhoR / tot);
                if (reflLobe) wi = cosineHemisphere(path[cur].ns, rng);
                else          wi = cosineHemisphere(nb, rng);
                pdfW = dBsdfPdf(sc, path[cur], wo, wi, lambda);
                pdfRevW = dBsdfPdf(sc, path[cur], wi, wo, lambda);
                if (pdfW <= 0) { terminate = true; break; }
                // f*|cos|/pdf = rho_lobe/PI * |cos| / (pSel*|cos|/PI) = rho_lobe/pSel = tot.
                betaFactor = tot;
                // The lobe was CHOSEN by the hero's albedo split, so each secondary divides
                // by the HERO's albedo for that lobe, not its own:
                // f_i·cos/pdf_hero = rho_i(lobe) · tot_hero / rho_hero(lobe).
                secChromatic = true;
                for (int i = 0; i + 1 < nUp; ++i) {
                    double rR, rT; dDiffuseTransmitAlbedos(sc, *mp, h, hb.lam[i + 1], rR, rT);
                    double num = reflLobe ? rR   : rT;
                    double den = reflLobe ? rhoR : rhoT;
                    secF[i] = (den > 0.0) ? num * tot / den : 0.0;
                }
                break;
            }
            case D_MIRROR: {
                double r = clamp01(specLookup(mp->reflect, lambda));
                wi = reflectv(rd, path[cur].ns); betaFactor = r; delta = 1;
                // Mirror direction is λ-independent: keep the bundle, reweight per-λ.
                keepBundle = true; secChromatic = true;
                for (int i = 0; i + 1 < nUp; ++i)
                    secF[i] = clamp01(specLookup(mp->reflect, hb.lam[i + 1]));
                break;
            }
            case D_DIELECTRIC: {
                // Nested-dielectric priority: exterior IOR from the medium stack; overlapping
                // dielectrics ranked by priority (SAFE FALLBACK to flat air<->glass otherwise).
                DVec3 nro, nrd; dDielectricStep(sc, *mp, h, rd, lambda, rng, matId, stk, nro, nrd);
                wi = nrd; betaFactor = 1.0; delta = 1;
                break;
            }
            case D_HALFMIRROR: {
                double r = clamp01(specLookup(mp->reflect, lambda));
                if (rng.uniform() < r) wi = reflectv(rd, path[cur].ns); else wi = rd;
                betaFactor = 1.0; delta = 1;
                break;
            }
            case D_FILTER: {
                // Colored gel filter: straight-through delta, throughput ×= T(lambda).
                double t = clamp01(dTransmitSlot(sc, *mp, h, lambda));
                wi = rd; betaFactor = t; delta = 1;
                // Straight-through for every λ: keep the bundle, reweight per-λ. This is
                // the case with the widest per-λ spread (a Wratten gel is 0 over most of
                // the spectrum), and the reason secF is absolute rather than a ratio.
                keepBundle = true; secChromatic = true;
                for (int i = 0; i + 1 < nUp; ++i)
                    secF[i] = clamp01(dTransmitSlot(sc, *mp, h, hb.lam[i + 1]));
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
        // Kill the walk only when EVERY live wavelength is dead: the hero's own factor can
        // legitimately be 0 while a secondary's is not (gel filter, saturated spectral
        // reflectance), and dropping the bundle there biases low. nUp == 1 -> empty loop ->
        // mxF == betaFactor, i.e. exactly the old scalar test.
        double mxF = betaFactor;
        if (secChromatic) for (int i = 0; i + 1 < nUp; ++i) if (secF[i] > mxF) mxF = secF[i];
        if (terminate || mxF <= 0.0) return;

        path[cur].delta = delta;
        if (delta) { pdfW = 0.0; pdfRevW = 0.0; }
        path[cur - 1].pdfRev = dConvertDensity(pdfRevW, path[cur], path[cur - 1]);

        beta *= betaFactor;
        for (int i = 0; i + 1 < nUp; ++i) betaSec[i] *= secChromatic ? secF[i] : betaFactor;
        // Veach adjoint shading-normal correction on the LIGHT subpath only (1 when
        // ns==ng). wo = toward previous (light-side) vertex; wi = sampled continuation.
        if (importance && !delta) {
            DVec3 ngo = (dot(path[cur].ng, path[cur].ns) >= 0.0) ? path[cur].ng : path[cur].ng * (Real)(-1);
            const double adj = (double)dShadingAdjointCorr(wo, normalize(wi), path[cur].ns, ngo);
            beta *= adj;                                        // purely geometric: same for all λ
            for (int i = 0; i + 1 < nUp; ++i) betaSec[i] *= adj;
        }
        // DE-HERO. Every delta interface (dielectric / thin-film / grating / multilayer /
        // half-mirror ...) picks a direction the secondaries cannot follow, so the bundle
        // collapses to the hero from here on. The vertex JUST pushed keeps its full nUp (it
        // really was reached by all C wavelengths); only its continuation is single-λ.
        // EXCEPTION: Mirror and Filter are delta but wavelength-INDEPENDENT in direction,
        // so they set keepBundle and carry the secondaries on a per-λ secF instead.
        if (delta && !keepBundle) nUp = 1;
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
                                        int px, int py, const DHeroBundle& hb, int maxDepth,
                                        DRng& rng, DVertex* path, double* pathSec, int secStride,
                                        int maxV, DEscape* esc = nullptr) {
    const Real lambda = hb.lam[0];
    // The camera vertex sees every wavelength at unit throughput: the bundle starts at full
    // width with all secondary throughputs 1 (importance leaves the camera achromatic).
    double betaSec0[BDPT_NSEC];
    for (int i = 0; i + 1 < hb.C; ++i) betaSec0[i] = 1.0;
    DVertex c;
    c.type = BV_CAMERA; c.ns = cam.w; c.ng = cam.w;
    c.beta = 1.0; c.pdfFwd = 0; c.pdfRev = 0; c.delta = 0; c.matId = -1; c.lightIdx = -1;
    c.emitPatW = (Real)1;
    c.mediumG = 0.0; c.mediumId = -1; c.nUp = hb.C;
    for (int i = 0; i + 1 < hb.C; ++i) pathSec[i] = 1.0;
    if (cam.hasLens) {
        Real jx = rng.uniform(), jy = rng.uniform();
        Real u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 ro, rd; Real wl = 0;
        c.nUp = 1;               // lensed cameras are single-λ (the hero gate excludes them)
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
        dRandomWalk(sc, cam, diffraction, ro, rd, (double)wl, pdfDir, hb, maxDepth - 1, rng,
                    path, pathSec, secStride, maxV, n, false, betaSec0, 1, esc);
        return n;
    }
    c.p = cam.eye;
    path[0] = c; int n = 1;
    Real jx = rng.uniform(), jy = rng.uniform();
    Real sx = (Real)2 * (((Real)px + jx) / (Real)cam.resX) - (Real)1;
    Real sy = (Real)2 * (((Real)py + jy) / (Real)cam.resY) - (Real)1;
    DVec3 rd = normalize(cam.w + cam.u * ((sx + (Real)cam.frustumShiftX) * (Real)cam.tanHalfX) + cam.v * (sy * (Real)cam.tanHalfY));
    double cosCam = ddot(rd, cam.w);
    double pdfDir = dCameraPdfDir(cam, cosCam);
    dRandomWalk(sc, cam, diffraction, cam.eye, rd, 1.0, pdfDir, hb, maxDepth - 1, rng,
                path, pathSec, secStride, maxV, n, false, betaSec0, hb.C, esc);
    return n;
}
// Sample a light subpath. path[0] is the light endpoint (beta = Le).
__device__ static int dGenLightSubpath(const DScene& sc, const DCamera& cam, int diffraction,
                                       const DHeroBundle& hb, int maxDepth,
                                       DRng& rng, DVertex* path, double* pathSec, int secStride,
                                       int maxV) {
    const Real lambda = hb.lam[0];
    const double invPdfLambda = hb.invPdf[0];
    if (sc.nEmitters == 0 || sc.totalPower <= 0.0) return 0;
    int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
    const DEmitter& em = sc.emitters[ei];
    if (em.shape == 3 || em.collimated) return 0;   // env / collimated: outside BDPT scope
    if (dIsDeltaEmitter(em)) {
        // ---- delta emitter (spot / distant sun), device twin of bdpt.h deltaLightSubpath.
        //  spot  a POINT at em.origin radiating into the outer cone: the direction is
        //        uniform there (pdfW = 1/(2 PI (1 - cosOuter))) and the smoothstep penumbra
        //        scales the emitted INTENSITY (emitSpd is W/sr for a spot, since its
        //        geomWeight is spotOmega). pdfPos is 1 — the position is a Dirac delta.
        //  sun   an infinitely distant disc: the origin is a point on a disc of radius
        //        sceneRadius, centred one radius UPSTREAM of the scene centre and
        //        perpendicular to beamDir, so every emitted ray enters the scene's
        //        cross-section (pdfPos = 1/(pi R^2)); the direction is drawn inside the
        //        solar cone (pdfW = 1/Omega) and emitSpd is a RADIANCE (addSunLight already
        //        divided the irradiance by Omega).
        // Both store path[0].pdfFwd = 0 (PBRT's delta convention — see
        // dVertexPdfLightOriginF) and dMisWeight drops the s=0 strategy for them.
        const bool isSun = (em.shape == 6);
        double pdfChoiceD = em.power / sc.totalPower;
        if (pdfChoiceD <= 0.0) return 0;
        Real u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 dir = dSunSampleCone(em, em.beamDir, (double)u1, (double)u2);
        double coneSolid = 2.0 * DPI * (1.0 - em.spotCosOuter);
        if (coneSolid <= 0.0) return 0;
        double pdfDirD = 1.0 / coneSolid;
        DVec3 org; double pdfPosD = 1.0;             // delta (spot) unless the sun's disc
        if (isSun) {
            double R = sc.sceneRadius;
            if (R <= 0.0) return 0;
            DVec3 tt, bb; onb(em.beamDir, tt, bb);
            double rr = R * sqrt((double)rng.uniform()), phi = 2.0 * DPI * (double)rng.uniform();
            org = sc.sceneCenter + tt * (Real)(rr * cos(phi)) + bb * (Real)(rr * sin(phi))
                - em.beamDir * (Real)R;
            pdfPosD = 1.0 / (DPI * R * R);
        } else {
            org = em.origin;
        }
        // Spot: the smoothstep penumbra scales the emitted intensity. Sun: no falloff.
        double fall = isSun ? 1.0
                            : spotFalloff(ddot(dir, em.beamDir), em.spotCosInner, em.spotCosOuter);
        if (fall <= 0.0) return 0;
        double LeD = (double)specLookup(em.emitSpd, lambda) * invPdfLambda * fall;
        double LeSecD[BDPT_NSEC], mxLeD = LeD;
        for (int i = 0; i + 1 < hb.C; ++i) {
            LeSecD[i] = (double)specLookup(em.emitSpd, hb.lam[i + 1]) * hb.invPdf[i + 1] * fall;
            if (LeSecD[i] > mxLeD) mxLeD = LeSecD[i];
        }
        if (mxLeD <= 0.0) return 0;

        DVertex D0;
        D0.type = BV_LIGHT; D0.p = org; D0.ns = dir; D0.ng = dir;
        D0.beta = LeD;
        D0.pdfFwd = 0.0;                             // delta origin (dVertexPdfLightOriginF)
        D0.pdfRev = 0; D0.delta = 0;                 // NEE (s=1) to a spot/sun IS possible
        D0.matId = -1; D0.lightIdx = ei;             // no emissive geometry to land on
        D0.emitPatW = (Real)1;                       // spot/sun reject `emit pattern:` at load
        D0.mediumG = 0.0; D0.mediumId = -1;
        D0.u = 0; D0.v = 0;
        D0.nUp = hb.C;
        for (int i = 0; i + 1 < hb.C; ++i) pathSec[i] = LeSecD[i];
        path[0] = D0; int nD = 1;
        // Walk throughput = Le / (pdfChoice * pdfPos * pdfDir); no cosine, because the
        // emission normal IS the emission direction for both of these (|cos| == 1).
        const double invP = 1.0 / (pdfChoiceD * pdfPosD * pdfDirD);
        double betaWalkD = LeD * invP;
        double betaWalkSecD[BDPT_NSEC];
        for (int i = 0; i + 1 < hb.C; ++i) betaWalkSecD[i] = LeSecD[i] * invP;
        dRandomWalk(sc, cam, diffraction, org, dir, betaWalkD, pdfDirD, hb, maxDepth - 1, rng,
                    path, pathSec, secStride, maxV, nD, true, betaWalkSecD, hb.C);
        // Infinite-light density patch (PBRT): the first scene vertex was given a
        // solid-angle density converted with 1/dist^2 from the fictitious disc point, but
        // the reverse direction (dVertexPdfLightF) reports the planar 1/(pi R^2). Rewrite
        // it to match, or the two sides of every MIS ratio disagree.
        if (isSun && nD > 1) {
            double pdf = pdfPosD;
            if (dOnSurface(path[1])) pdf *= fabs(ddot(path[1].ns, dir));
            path[1].pdfFwd = pdf;
        }
        return nD;
    }
    Real u1 = rng.uniform(), u2 = rng.uniform();
    DVec3 y, nOut;
    // `emitPatW` is this point's `emit pattern:` factor (1.0, and a bit-identical draw,
    // when the emitter is unpatterned). It scales the emitted radiance the subpath starts
    // with, exactly as the emission-on-hit side scales the s=0 strategy's radiance.
    double emitPatW = dEmitterSamplePointPat(sc, em, u1, u2, y, nOut);
    double Le = (double)specLookup(em.emitSpd, lambda) * invPdfLambda * emitPatW;
    if (Le <= 0.0) return 0;
    double pdfChoice = em.power / sc.totalPower;
    double pdfPos = (em.area > 0.0) ? 1.0 / em.area : 0.0;
    if (pdfPos <= 0.0) return 0;
    DVertex L0;
    L0.type = BV_LIGHT; L0.p = y; L0.ns = nOut; L0.ng = nOut;
    L0.beta = Le; L0.pdfFwd = pdfChoice * pdfPos; L0.pdfRev = 0; L0.delta = 0;
    L0.matId = em.matId; L0.lightIdx = ei;
    L0.emitPatW = (Real)emitPatW;
    L0.mediumG = 0.0; L0.mediumId = -1;
    // The light endpoint carries each wavelength's own emitted radiance Le(λ)/pdf(λ).
    L0.nUp = hb.C;
    for (int i = 0; i + 1 < hb.C; ++i)
        pathSec[i] = (double)specLookup(em.emitSpd, hb.lam[i + 1]) * hb.invPdf[i + 1] * emitPatW;
    path[0] = L0; int n = 1;
    DVec3 dir = cosineHemisphere(nOut, rng);
    double cosLight = ddot(nOut, dir);
    if (cosLight <= 0.0) return 1;
    double pdfDir = cosLight / DPI;
    double betaWalk = Le * cosLight / (pdfChoice * pdfPos * pdfDir);
    double betaWalkSec[BDPT_NSEC];
    for (int i = 0; i + 1 < hb.C; ++i)
        betaWalkSec[i] = pathSec[i] * cosLight / (pdfChoice * pdfPos * pdfDir);
    DVec3 ro = y + nOut * (Real)1e-6;
    dRandomWalk(sc, cam, diffraction, ro, dir, betaWalk, pdfDir, hb, maxDepth - 1, rng,
                path, pathSec, secStride, maxV, n, true, betaWalkSec, hb.C);
    return n;
}

// Balance-heuristic MIS weight for strategy (s,t). Direct port of bdpt.h misWeight:
// PBRT temporarily rewrites the connection vertices' reverse densities / delta flags
// and (for s==1/t==1) installs the resampled endpoint, sums the density ratios of all
// other strategies, then rolls the mutations back. Here the vertices live in the
// per-thread local arrays, so we save whole vertices before ANY mutation and restore
// them at the end (whole-vertex restore subsumes PBRT's field-wise ScopedAssignments).
__device__ static double dMisWeight(const DScene& sc, const DCamera& cam,
                                    const DVertex* light, const DVertex* eye,
                                    const DVertex& sampled, int s, int t, Real lambda) {
    if (s + t == 2) return 1.0;
    const int si = s - 1, ti = t - 1, sMi = s - 2, tMi = t - 2;

    // Non-mutating rewrite of the PBRT ScopedAssignment dance: the old code copied the
    // four vertices adjacent to the connection edge to locals, wrote hypotheticals into
    // the arrays, walked, and restored (8 x 96B local-memory copies per call). Instead,
    // resolve the a1 endpoint substitution (s==1/t==1 use the resampled endpoint) through
    // pointers and apply the a2/a3 delta-clears + a4..a7 pdfRev overrides inline in the
    // walks by index compare. Arithmetic per PBRT 16.3; FP32 per the note above.
    const DVertex* QsP = (s > 0) ? ((s == 1) ? &sampled : &light[si]) : nullptr;
    const DVertex* PtP = (t == 1) ? &sampled : &eye[ti];   // t >= 1 always

    // a4..a7: reverse densities of the connection-adjacent vertices. Each is only read
    // by a walk index >= 1, so skip the provably-dead ones (the old code computed those
    // too, wrote them into the arrays, and restored over them unread).
    float ptPdfRev = 0.f, ptMPdfRev = 0.f, qsPdfRev = 0.f, qsMPdfRev = 0.f;
    if (t >= 2)
        ptPdfRev = (s > 0) ? dVertexPdfF(sc, cam, (s > 1) ? &light[sMi] : nullptr, *QsP, *PtP, lambda)
                           : dVertexPdfLightOriginF(sc, *PtP);
    if (t >= 3)
        ptMPdfRev = (s > 0) ? dVertexPdfF(sc, cam, QsP, *PtP, eye[tMi], lambda)
                            : dVertexPdfLightF(sc, *PtP, eye[tMi]);
    if (s >= 1) qsPdfRev  = dVertexPdfF(sc, cam, (t > 1) ? &eye[tMi] : nullptr, *PtP, *QsP, lambda);
    if (s >= 2) qsMPdfRev = dVertexPdfF(sc, cam, PtP, *QsP, light[sMi], lambda);

    float sumRi = 0.f, ri = 1.f;
    for (int i = t - 1; i > 0; --i) {
        float num, den; int dl;
        if (i == ti)       { num = ptPdfRev;             den = (float)PtP->pdfFwd;   dl = 0; }
        else if (i == tMi) { num = ptMPdfRev;            den = (float)eye[i].pdfFwd; dl = eye[i].delta; }
        else               { num = (float)eye[i].pdfRev; den = (float)eye[i].pdfFwd; dl = eye[i].delta; }
        ri *= (num != 0.f ? num : 1.f) / (den != 0.f ? den : 1.f);
        if (!dl && !eye[i - 1].delta) sumRi += ri;
    }
    ri = 1.f;
    for (int i = s - 1; i >= 0; --i) {
        float num, den; int dl;
        if (i == si)       { num = qsPdfRev;               den = (float)QsP->pdfFwd;     dl = 0; }
        else if (i == sMi) { num = qsMPdfRev;              den = (float)light[i].pdfFwd; dl = light[i].delta; }
        else               { num = (float)light[i].pdfRev; den = (float)light[i].pdfFwd; dl = light[i].delta; }
        ri *= (num != 0.f ? num : 1.f) / (den != 0.f ? den : 1.f);
        // The hypothetical strategy at index i connects light[i-1] to the eye side, so it is
        // impossible if either end of that new edge is delta. At i == 0 the "edge" is instead
        // the eye path LANDING on the emitter, which a DELTA light (spot: a point; sun:
        // infinitely far, no geometry) can never be hit by — PBRT's IsDeltaLight(). The
        // light endpoint is light[0], except for s==1 where the substituted `sampled`
        // endpoint (index si == 0) is the one carrying the emitter.
        bool deltaPrev = (i > 0) ? (light[i - 1].delta != 0)
                                 : dIsDeltaLightVertex(sc, (si == 0) ? *QsP : light[0]);
        if (!dl && !deltaPrev) sumRi += ri;
    }
    return 1.0 / (1.0 + (double)sumRi);
}

// Connect strategy (s,t); returns the MIS-weighted radiance of the HERO wavelength. For
// t==1 the result is a light-image splat to (outPx,outPy) with isSplat=1. Direct port of
// bdpt.h connectBDPT.
//
// Hero bundle: `Lsec[i]` receives the C-1 secondaries' MIS-weighted radiance and `nUpConn`
// how many wavelengths this connection actually carries — min(nUp of the two endpoints),
// since either subpath may have de-hero'd independently. `nUpConn == 0` means "no
// contribution" (every early-out leaves it 0), which is how the caller detects a reject.
// Because every SAMPLING decision was hero-driven, the MIS weight is identical for all
// wavelengths, so dMisWeight runs ONCE and multiplies the whole bundle.
__device__ static double dConnectBDPT(const DScene& sc, const DCamera& cam,
                                      const DVertex* light, const DVertex* eye,
                                      const double* lightSec, const double* eyeSec, int secStride,
                                      int s, int t, const DHeroBundle& hb, DRng& rng,
                                      int& outPx, int& outPy, int& isSplat,
                                      double* Lsec, int& nUpConn) {
    const Real lambda = hb.lam[0];
    const double invPdfLambda = hb.invPdf[0];
    isSplat = 0;
    nUpConn = 0;                    // set to the real width only once a contribution exists
    if (t > 1 && s != 0 && dIsLightVertex(sc, eye[t - 1])) return 0.0;

    double L = 0.0;
    int nUp = 1;                    // live wavelengths for THIS connection (set per branch)
    DVertex sampled;
    sampled.type = BV_SURFACE; sampled.beta = 0; sampled.pdfFwd = 0; sampled.pdfRev = 0;
    sampled.delta = 0; sampled.matId = -1; sampled.lightIdx = -1;
    sampled.emitPatW = (Real)1;
    sampled.mediumG = 0.0; sampled.mediumId = -1; sampled.u = 0; sampled.v = 0;

    if (s == 0) {
        if (t < 2) return 0.0;
        const DVertex& pt = eye[t - 1];
        if (!dIsLightVertex(sc, pt)) return 0.0;
        DVec3 wo = normalize(eye[t - 2].p - pt.p);
        nUp = pt.nUp;
        double Le = dVertexLe(sc, pt, wo, lambda, invPdfLambda);
        // The reject tests the max over the live wavelengths — identical to `Le <= 0` when
        // nUp == 1, so the single-λ path is unchanged bit-for-bit.
        double LeSec[BDPT_NSEC], mxLe = Le;
        for (int i = 0; i + 1 < nUp; ++i) {
            LeSec[i] = dVertexLe(sc, pt, wo, hb.lam[i + 1], hb.invPdf[i + 1]);
            if (LeSec[i] > mxLe) mxLe = LeSec[i];
        }
        if (mxLe <= 0.0) return 0.0;
        L = pt.beta * Le;
        for (int i = 0; i + 1 < nUp; ++i) Lsec[i] = eyeSec[(t - 1) * secStride + i] * LeSec[i];
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
        DVec3 wo = normalize(light[s - 2].p - qs.p);
        // Medium endpoint: phase*albedo, cosine 1, occlusion from the exact point.
        nUp = qs.nUp;
        double fSec[BDPT_NSEC];
        double cosSurf, f; DVec3 o;
        if (qs.type == BV_MEDIUM) {
            cosSurf = 1.0; f = dMediumScatterF(sc, qs, wo, wcam, lambda); o = qs.p;
            for (int i = 0; i + 1 < nUp; ++i)
                fSec[i] = dMediumScatterF(sc, qs, wo, wcam, hb.lam[i + 1]);
        } else {
            cosSurf = ddot(qs.ns, wcam);
            // Reflect-only vertices require the +ns side; a two-sided (DiffuseTransmit)
            // vertex may connect on either side (transmit lobe), so gate on dBsdfF and use
            // |cosSurf| in G (mirrors CPU bdpt.h connectBDPT).
            bool twoSided = dTwoSidedType(sc.mats[qs.matId].type);
            if (cosSurf == 0.0 || (!twoSided && cosSurf < 0.0)) return 0.0;
            // Geometric-hemisphere softening (matches CPU bdpt.h): the camera must lie on the
            // geometric front side too, else a smoothed shading normal leaks light through the
            // back face; ramp smoothly instead of a hard cutoff (Chiang 2019). No-op when ns==ng
            // (stG==1); skipped for two-sided (transmissive) materials.
            DVec3 ngoQ = (ddot(qs.ng, qs.ns) >= 0.0) ? qs.ng : qs.ng * (Real)(-1);
            double stG = twoSided ? 1.0 : (double)dShadowTerminatorG(wcam, qs.ns, ngoQ);
            if (stG <= 0.0) return 0.0;
            f = dBsdfF(sc, qs, wo, wcam, lambda);
            // Adjoint correction on the LIGHT-subpath vertex qs (particle side, outgoing
            // toward camera). 1 when ns==ng. wo = toward previous (light-side) vertex.
            // |cos| inside dShadingAdjointCorr makes it lobe-agnostic (serves the transmit lobe).
            // Purely geometric, so the same factor serves every wavelength.
            const double adj = (double)dShadingAdjointCorr(wo, wcam, qs.ns, ngoQ) * stG;
            f *= adj;
            for (int i = 0; i + 1 < nUp; ++i)
                fSec[i] = dBsdfF(sc, qs, wo, wcam, hb.lam[i + 1]) * adj;
            double sgn = ddot(qs.ng, wcam) >= 0.0 ? 1.0 : -1.0;
            o = qs.p + qs.ng * (Real)(sgn * 1e-6);
        }
        {   // max over live wavelengths (identical to `f <= 0` when nUp == 1)
            double mxF = f;
            for (int i = 0; i + 1 < nUp; ++i) if (fSec[i] > mxF) mxF = fSec[i];
            if (mxF <= 0.0) return 0.0;
        }
        if (occluded(sc, o, wcam, connMaxT(dist))) return 0.0;
        double cosCam = ddot(qs.p - cam.eye, cam.w) / dist;   // positive (point in front)
        // Hero-only transmittance: the hero gate excludes any medium, so Tr is exactly 1
        // whenever nUp > 1.
        double Tr = (sc.mediaN > 0) ? (double)dMediaTransmittance(sc, qs.p, wcam, (Real)dist, lambda, rng) : 1.0;
        double G = fabs(cosSurf) * cosCam / dist2;
        L = qs.beta * f * G * dCameraWe(cam, cosCam) * Tr;
        for (int i = 0; i + 1 < nUp; ++i)
            Lsec[i] = lightSec[(s - 1) * secStride + i] * fSec[i] * G * dCameraWe(cam, cosCam) * Tr;
        {   double mxL = L;
            for (int i = 0; i + 1 < nUp; ++i) if (Lsec[i] > mxL) mxL = Lsec[i];
            if (mxL <= 0.0) return 0.0;
        }
        sampled.type = BV_CAMERA; sampled.p = cam.eye; sampled.ns = cam.w; sampled.ng = cam.w;
        sampled.beta = 1.0;
        outPx = px; outPy = py; isSplat = 1;
    } else if (s == 1) {
        const DVertex& pt = eye[t - 1];
        if (!dVertConnectible(sc, pt)) return 0.0;
        int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
        const DEmitter& em = sc.emitters[ei];
        if (em.shape == 3 || em.collimated) return 0.0;   // outside BDPT scope (mode-D guard)
        double pdfChoice = em.power / sc.totalPower;
        if (pdfChoice <= 0.0) return 0.0;
        Real u1 = rng.uniform(), u2 = rng.uniform();
        // Per-shape connection geometry. `Wgeom` is the whole λ-independent weight that
        // multiplies |cosSurf| in the estimator, i.e.
        //     L = beta * f * Le * |cosSurf| * Wgeom * Tr * stG
        // which for an area light is the familiar cosLight/(dist^2 * pdfA). Collecting it
        // into one scalar is what lets the three emission models (Lambertian area, spot
        // cone, distant sun) share the BSDF / occlusion / transmittance code below.
        // Mirrors bdpt.h connectBDPT.
        DVec3 y, nOut, wi;
        double dist = 0.0, Wgeom = 0.0, emitPatW = 1.0;
        const bool deltaLight = dIsDeltaEmitter(em);
        // A distant sun has no finite light point: the shadow ray runs all the way to the
        // scene exit, so it must NOT be shortened by the usual endpoint epsilon.
        double occlEps = 2e-6;
        if (em.shape == 2) {
            // Point spot: the connection point is deterministic (delta position); the
            // smoothstep penumbra weights the intensity toward this receiver.
            DVec3 toL = em.origin - pt.p; double dist2 = ddot(toL, toL);
            if (dist2 <= 0.0) return 0.0;
            dist = sqrt(dist2); wi = toL * (Real)(1.0 / dist);
            double fall = spotFalloff(ddot(wi * (Real)-1, em.beamDir),
                                      em.spotCosInner, em.spotCosOuter);
            if (fall <= 0.0) return 0.0;               // outside the cone
            y = em.origin; nOut = wi * (Real)-1;
            Wgeom = fall / (dist2 * pdfChoice);        // emitSpd is an INTENSITY (W/sr)
        } else if (em.shape == 6) {
            // Distant sun: sample a direction inside the solar cone (pdfW = 1/Omega) and
            // shadow-ray it out of the scene. No 1/dist^2 and no cosLight — the source is
            // at infinity — so Wgeom is just Omega/pdfChoice and emitSpd is a radiance.
            wi = dSunSampleCone(em, em.beamDir * (Real)-1, (double)u1, (double)u2);
            dist = (double)length(sc.sceneCenter - pt.p) + sc.sceneRadius;
            occlEps = 0.0;
            y = pt.p + wi * (Real)dist; nOut = wi * (Real)-1;
            Wgeom = em.spotOmega / pdfChoice;
        } else {
            // The sampled point's `emit pattern:` factor scales the radiance this strategy
            // sees; s=0 applies the pointwise-equal emission-on-hit factor, which is what
            // keeps the two MIS-combined strategies consistent.
            emitPatW = dEmitterSamplePointPat(sc, em, u1, u2, y, nOut);
            DVec3 toL = y - pt.p; double dist2 = ddot(toL, toL);
            if (dist2 <= 0.0) return 0.0;
            dist = sqrt(dist2); wi = toL * (Real)(1.0 / dist);
            double cosLight = ddot(nOut, wi * (Real)-1);
            if (cosLight <= 0.0) return 0.0;           // emitter stays one-sided
            if (em.area <= 0.0) return 0.0;
            Wgeom = cosLight * em.area / (dist2 * pdfChoice);   // == cosLight/(d^2 * pdfA)
        }
        nUp = pt.nUp;
        double Le = (double)specLookup(em.emitSpd, lambda) * invPdfLambda * emitPatW;
        double LeSec[BDPT_NSEC];
        {   double mxLe = Le;
            for (int i = 0; i + 1 < nUp; ++i) {
                LeSec[i] = (double)specLookup(em.emitSpd, hb.lam[i + 1]) * hb.invPdf[i + 1] * emitPatW;
                if (LeSec[i] > mxLe) mxLe = LeSec[i];
            }
            if (mxLe <= 0.0) return 0.0;
        }
        DVec3 wo = normalize(eye[t - 2].p - pt.p);
        // Cheap sidedness/terminator rejects and the shadow ray run FIRST; the BSDF/phase
        // eval (texture fetches, lobe math) is deferred until the connection is known
        // unoccluded. occluded() consumes no RNG and the eval is deterministic, so the
        // reorder is bit-identical — it only skips work for shadowed connections.
        double cosSurf, stG = 1.0; DVec3 o;
        if (pt.type == BV_MEDIUM) {
            cosSurf = 1.0; o = pt.p;
        } else {
            cosSurf = ddot(pt.ns, wi);
            // Two-sided eye vertex may connect to the light through its transmit lobe (back
            // hemisphere); gate on dBsdfF, use |cosSurf| in G (mirrors CPU bdpt.h).
            bool twoSided = dTwoSidedType(sc.mats[pt.matId].type);
            if (cosSurf == 0.0 || (!twoSided && cosSurf < 0.0)) return 0.0;
            // Geometric-hemisphere softening on the eye/radiance vertex (matches CPU bdpt.h):
            // ramp smoothly instead of a hard cutoff (Chiang 2019). No-op when ns==ng (stG==1);
            // skipped for two-sided (transmissive) materials.
            if (!twoSided) {
                DVec3 ngoP = (ddot(pt.ng, pt.ns) >= 0.0) ? pt.ng : pt.ng * (Real)(-1);
                stG = (double)dShadowTerminatorG(wi, pt.ns, ngoP);
                if (stG <= 0.0) return 0.0;
            }
            double sgn = ddot(pt.ng, wi) >= 0.0 ? 1.0 : -1.0;
            o = pt.p + pt.ng * (Real)(sgn * 1e-6);
        }
        if (occluded(sc, o, wi, connMaxT(dist, occlEps))) return 0.0;
        double f = (pt.type == BV_MEDIUM) ? dMediumScatterF(sc, pt, wo, wi, lambda)
                                          : dBsdfF(sc, pt, wo, wi, lambda) * stG;
        double fSec[BDPT_NSEC];
        for (int i = 0; i + 1 < nUp; ++i)
            fSec[i] = (pt.type == BV_MEDIUM) ? dMediumScatterF(sc, pt, wo, wi, hb.lam[i + 1])
                                             : dBsdfF(sc, pt, wo, wi, hb.lam[i + 1]) * stG;
        {   // max over live wavelengths (identical to `f <= 0` when nUp == 1)
            double mxF = f;
            for (int i = 0; i + 1 < nUp; ++i) if (fSec[i] > mxF) mxF = fSec[i];
            if (mxF <= 0.0) return 0.0;
        }
        // Hero-only transmittance (exactly 1 whenever nUp > 1; see the t==1 branch).
        double Tr = (sc.mediaN > 0) ? (double)dMediaTransmittance(sc, pt.p, wi, (Real)dist, lambda, rng) : 1.0;
        double G = fabs(cosSurf) * Wgeom;
        L = pt.beta * f * Le * G * Tr;
        for (int i = 0; i + 1 < nUp; ++i)
            Lsec[i] = eyeSec[(t - 1) * secStride + i] * fSec[i] * LeSec[i] * G * Tr;
        {   double mxL = L;
            for (int i = 0; i + 1 < nUp; ++i) if (Lsec[i] > mxL) mxL = Lsec[i];
            if (mxL <= 0.0) return 0.0;
        }
        sampled.type = BV_LIGHT; sampled.p = y; sampled.ns = nOut; sampled.ng = nOut;
        sampled.lightIdx = ei;
        // A delta light has no emissive geometry and no area density: matId is -1 and
        // pdfFwd is 0 on BOTH sides of every MIS ratio (the light subpath stores the same
        // 0), matching PBRT — see dVertexPdfLightOriginF.
        sampled.matId = deltaLight ? -1 : em.matId;
        sampled.beta = deltaLight ? Le * Wgeom : Le * em.area / pdfChoice;   // == Le/pdfA
        sampled.pdfFwd = deltaLight ? 0.0 : (pdfChoice / em.area);
        sampled.emitPatW = (Real)emitPatW;   // so dVertexLe on this sampled vertex agrees
    } else {
        const DVertex& qs = light[s - 1];
        const DVertex& pt = eye[t - 1];
        if (!dVertConnectible(sc, qs) || !dVertConnectible(sc, pt)) return 0.0;
        // The two subpaths de-hero independently; the connection carries the narrower bundle.
        nUp = (qs.nUp < pt.nUp) ? qs.nUp : pt.nUp;
        DVec3 d = qs.p - pt.p; double dist2 = ddot(d, d);
        if (dist2 <= 0.0) return 0.0;
        double dist = sqrt(dist2); DVec3 w = d * (Real)(1.0 / dist);   // pt -> qs
        DVec3 woE = normalize(eye[t - 2].p - pt.p);
        DVec3 woL = normalize(light[s - 2].p - qs.p);
        // Each endpoint is a surface (BSDF, cosine) or a medium (phase*albedo, cos=1).
        // Cheap sidedness/terminator rejects and the shadow ray run FIRST; both endpoint
        // BSDF/phase evals (texture fetches, lobe math) are deferred until the connection
        // is known unoccluded. occluded() consumes no RNG and the evals are deterministic,
        // so the reorder is bit-identical — it only skips work for shadowed connections.
        double cosE, cosL, stGE = 1.0, stGL = 1.0; DVec3 o, ngoQ;
        if (pt.type == BV_MEDIUM) {
            cosE = 1.0; o = pt.p;
        } else {
            cosE = ddot(pt.ns, w);
            // Two-sided eye endpoint may connect on its back hemisphere (transmit lobe);
            // gate on dBsdfF, use |cosE| in G (mirrors CPU bdpt.h).
            bool twoSidedE = dTwoSidedType(sc.mats[pt.matId].type);
            if (cosE == 0.0 || (!twoSidedE && cosE < 0.0)) return 0.0;
            // Geometric-hemisphere softening on the eye endpoint (connection dir w): ramp
            // smoothly instead of a hard cutoff (Chiang 2019). No-op ns==ng (stGE==1);
            // skipped for two-sided (transmissive) materials.
            if (!twoSidedE) {
                DVec3 ngoE = (ddot(pt.ng, pt.ns) >= 0.0) ? pt.ng : pt.ng * (Real)(-1);
                stGE = (double)dShadowTerminatorG(w, pt.ns, ngoE);
                if (stGE <= 0.0) return 0.0;
            }
            double sgn = ddot(pt.ng, w) >= 0.0 ? 1.0 : -1.0;
            o = pt.p + pt.ng * (Real)(sgn * 1e-6);
        }
        if (qs.type != BV_MEDIUM) {
            cosL = ddot(qs.ns, w * (Real)-1);
            // Two-sided light endpoint may connect on its back hemisphere (transmit lobe).
            bool twoSidedL = dTwoSidedType(sc.mats[qs.matId].type);
            if (cosL == 0.0 || (!twoSidedL && cosL < 0.0)) return 0.0;
            // Geometric-hemisphere softening on the light endpoint (connection dir -w): ramp
            // smoothly instead of a hard cutoff (Chiang 2019). No-op ns==ng (stGL==1);
            // skipped for two-sided (transmissive) materials.
            ngoQ = (ddot(qs.ng, qs.ns) >= 0.0) ? qs.ng : qs.ng * (Real)(-1);
            if (!twoSidedL) {
                stGL = (double)dShadowTerminatorG(w * (Real)-1, qs.ns, ngoQ);
                if (stGL <= 0.0) return 0.0;
            }
        } else {
            cosL = 1.0;
        }
        if (occluded(sc, o, w, connMaxT(dist))) return 0.0;
        double fE, fL;
        double fESec[BDPT_NSEC], fLSec[BDPT_NSEC];
        if (pt.type == BV_MEDIUM) {
            fE = dMediumScatterF(sc, pt, woE, w, lambda);
            for (int i = 0; i + 1 < nUp; ++i)
                fESec[i] = dMediumScatterF(sc, pt, woE, w, hb.lam[i + 1]);
        } else {
            fE = dBsdfF(sc, pt, woE, w, lambda) * stGE;
            for (int i = 0; i + 1 < nUp; ++i)
                fESec[i] = dBsdfF(sc, pt, woE, w, hb.lam[i + 1]) * stGE;
        }
        if (qs.type == BV_MEDIUM) {
            fL = dMediumScatterF(sc, qs, woL, w * (Real)-1, lambda);
            for (int i = 0; i + 1 < nUp; ++i)
                fLSec[i] = dMediumScatterF(sc, qs, woL, w * (Real)-1, hb.lam[i + 1]);
        } else {
            fL = dBsdfF(sc, qs, woL, w * (Real)-1, lambda) * stGL;
            // Adjoint correction on the LIGHT-subpath endpoint qs only (particle side,
            // outgoing = -w toward the eye vertex). fE is the Radiance side — no correction.
            // |cos| inside dShadingAdjointCorr makes it lobe-agnostic (serves the transmit lobe).
            // Purely geometric, so the same factor serves every wavelength.
            const double adjL = (double)dShadingAdjointCorr(woL, w * (Real)-1, qs.ns, ngoQ);
            fL *= adjL;
            for (int i = 0; i + 1 < nUp; ++i)
                fLSec[i] = dBsdfF(sc, qs, woL, w * (Real)-1, hb.lam[i + 1]) * stGL * adjL;
        }
        {   // max over live wavelengths on each side (identical to the scalar tests at nUp==1)
            double mxE = fE, mxL = fL;
            for (int i = 0; i + 1 < nUp; ++i) {
                if (fESec[i] > mxE) mxE = fESec[i];
                if (fLSec[i] > mxL) mxL = fLSec[i];
            }
            if (mxE <= 0.0 || mxL <= 0.0) return 0.0;
        }
        // Hero-only transmittance (exactly 1 whenever nUp > 1; see the t==1 branch).
        double Tr = (sc.mediaN > 0) ? (double)dMediaTransmittance(sc, pt.p, w, (Real)dist, lambda, rng) : 1.0;
        double G = fabs(cosE) * fabs(cosL) / dist2;
        L = pt.beta * fE * fL * qs.beta * G * Tr;
        for (int i = 0; i + 1 < nUp; ++i)
            Lsec[i] = eyeSec[(t - 1) * secStride + i] * fESec[i] * fLSec[i]
                    * lightSec[(s - 1) * secStride + i] * G * Tr;
    }
    // One shared reject and ONE shared MIS weight for the whole bundle: every sampling
    // decision was hero-driven, so the balance-heuristic ratios do not depend on λ.
    double mx = L;
    for (int i = 0; i + 1 < nUp; ++i) if (Lsec[i] > mx) mx = Lsec[i];
    if (mx <= 0.0) return 0.0;
    const double mis = dMisWeight(sc, cam, light, eye, sampled, s, t, lambda);
    for (int i = 0; i + 1 < nUp; ++i) Lsec[i] *= mis;
    nUpConn = nUp;
    return L * mis;
}

// BDPT megakernel: one thread renders one (pixel,sample), grid-stride over all
// res*res*spp samples. t>=2 connections land on the sample's own pixel (camFilm);
// t==1 splats land on the projected raster pixel (splatFilm). Both are normalised by
// 1/spp on the host (bdpt.h renderBdpt convention).
// Chunked exactly like kBackward: renders `chunkSpp` samples-per-pixel starting at
// `sampleBase`, seeding on the global sample index (pixel*sppTotal + sampleBase + local)
// so any chunking is bit-identical to a single sppTotal pass.
// __launch_bounds__(128, 3): unconstrained the kernel compiles to 254 regs -> only 2
// blocks (256 threads) resident per SM. Capping at 3 blocks/SM (<=170 regs) trades a few
// extra spills for +50% latency hiding; the kernel is latency-bound on spilled/local
// state (8KB stack/thread), so occupancy wins.
//
// Templated on NS = the number of SECONDARY hero wavelength slots, so the scalar
// instantiation (kBdptT<0, MAXD>) allocates the per-vertex secondary-throughput arrays at one
// element and is bit-for-bit the original single-λ kernel: every hero loop it contains has
// an empty trip count, and every added reject is a max over one value. The hero
// instantiation (kBdptT<BDPT_NSEC, MAXD>) pays 2*MAXV*BDPT_NSEC doubles (~1.2 KB) of extra
// per-thread local state for the bundle.
//
// Also templated on MAXD = the per-thread vertex-stack depth bound. `eye`/`light` are
// thread-local arrays of ~100 B DVertex, so this is the one knob that sets the kernel's
// local-memory footprint: the MAXD = BDPT_MAXDEPTH instantiation is bit-for-bit the
// original kernel, and the MAXD = BDPT_DEEPDEPTH one is only instantiated — and only
// launched — when `-max-bounce N` asks for N > BDPT_MAXDEPTH. The connection double-loop
// below is O(MAXD^2), so the deep variant is genuinely slower per sample; that, plus the
// local-memory cost, is why it is opt-in rather than the default.
template <int NS, int MAXD>
__global__ void __launch_bounds__(128, 3)
kBdptT(DScene sc, DCamera cam, double* camFilm, double* splatFilm,
                      long long totalSamples, long long chunkSpp, long long sppTotal,
                      long long sampleBase, int resX, int maxDepth,
                      int diffraction, unsigned long long seedBase, int heroC) {
    enum { SECN = (NS > 0 ? NS : 1), MAXV = BDPT_MAXV_OF(MAXD) };
    if (maxDepth > MAXD) maxDepth = MAXD;   // device array bound (host picks the variant)
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    const int C = (NS > 0) ? heroC : 1;
    for (long long idx = g; idx < totalSamples; idx += G) {
        long long pix = idx / chunkSpp;
        long long gidx = pix * sppTotal + sampleBase + (idx - pix * chunkSpp);
        DRng rng; rng.seed((unsigned long long)(gidx * 2 + 1), seedBase ^ (unsigned long long)gidx);
        int px = (int)(pix % resX);
        int py = (int)(pix / resX);

        // Hero + C-1 stratified secondaries from ONE base uniform (u + i/C wrapped into
        // [0,1)), pushed through the same inverse-CDF the scalar path uses. The hero must
        // have a valid pdf; a dead secondary simply carries invPdf 0 (contributes nothing).
        DHeroBundle hb;
        if (C > 1) {
            double u = (double)rng.uniform(), pdf0 = 0.0;
            hb.lam[0] = dSampleSceneLambdaU(sc, u, pdf0);
            if (pdf0 <= 0.0) continue;
            hb.invPdf[0] = dInvPdfLambda(sc, hb.lam[0]);
            hb.C = C;
            for (int i = 1; i < C; ++i) {
                double uu = u + (double)i / C;
                if (uu >= 1.0) uu -= 1.0;
                double pdfI = 0.0;
                hb.lam[i] = dSampleSceneLambdaU(sc, uu, pdfI);
                hb.invPdf[i] = (pdfI > 0.0) ? dInvPdfLambda(sc, hb.lam[i]) : 0.0;
            }
        } else {
            double pdfLam = 0.0;
            hb.lam[0] = dSampleSceneLambda(sc, rng, pdfLam);
            if (pdfLam <= 0.0) continue;
            hb.invPdf[0] = dInvPdfLambda(sc, hb.lam[0]);
            hb.C = 1;
        }
        const Real lambda = hb.lam[0];

        DVertex eye[MAXV], light[MAXV];
        double eyeSec[MAXV * SECN], lightSec[MAXV * SECN];
        // Only pay for escape tracking when the scene actually has a distant sun.
        DEscape esc; esc.escaped = 0; esc.beta = 0.0; esc.nUp = 1;
        int nE = dGenCameraSubpath(sc, cam, diffraction, px, py, hb, maxDepth + 1, rng, eye, eyeSec, NS, MAXV,
                                   (sc.sunCount > 0) ? &esc : nullptr);
        int nL = dGenLightSubpath(sc, cam, diffraction, hb, maxDepth + 1, rng, light, lightSec, NS, MAXV);

        Real cx = cieX(lambda), cy = cieY(lambda), cz = cieZ(lambda);
        Real cxS[SECN], cyS[SECN], czS[SECN];
        for (int i = 0; i + 1 < hb.C; ++i) {
            cxS[i] = cieX(hb.lam[i + 1]); cyS[i] = cieY(hb.lam[i + 1]); czS[i] = cieZ(hb.lam[i + 1]);
        }
        // Direct view of a distant `light sun`. A sun is a delta-DIRECTION emitter with no
        // geometry, so dMisWeight drops the s=0 strategy for it and no connection strategy
        // can reach it either — without this the solar disc itself, and every mirror/water
        // glint of it, would simply be missing from mode D. It is added only when the
        // escaping ray came through camera + delta (specular) vertices ONLY, which is
        // exactly the case where no other strategy competes: NEE needs a connectible
        // (non-delta) vertex, and so does every s>=2 connection or t==1 splat. The MIS
        // weight is therefore exactly 1. Mirrors BdptRenderer::renderRows.
        if (esc.escaped) {
            bool allDelta = true;
            for (int i = 1; i < nE; ++i) if (!eye[i].delta) { allDelta = false; break; }
            if (allDelta) {
                for (int k = 0; k < sc.nEmitters; ++k) {
                    const DEmitter& em = sc.emitters[k];
                    if (em.shape != 6 || !dInSunCone(em, esc.dir)) continue;
                    double e0 = esc.beta * (double)specLookup(em.emitSpd, lambda) * hb.invPdf[0];
                    double ax = cx * e0, ay = cy * e0, az = cz * e0;
                    for (int i = 0; i + 1 < esc.nUp; ++i) {
                        double ek = esc.betaSec[i] *
                                    (double)specLookup(em.emitSpd, hb.lam[i + 1]) * hb.invPdf[i + 1];
                        ax += cxS[i] * ek; ay += cyS[i] * ek; az += czS[i] * ek;
                    }
                    if (esc.nUp > 1) { double inv = 1.0 / esc.nUp; ax *= inv; ay *= inv; az *= inv; }
                    size_t o = ((size_t)py * resX + px) * 3;
                    atomicAdd(&camFilm[o + 0], ax);
                    atomicAdd(&camFilm[o + 1], ay);
                    atomicAdd(&camFilm[o + 2], az);
                }
            }
        }
        for (int t = 1; t <= nE; ++t)
            for (int s = 0; s <= nL; ++s) {
                int depth = t + s - 2;
                if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth) continue;
                int spx = 0, spy = 0, isSplat = 0, nUpConn = 0;
                double Lsec[SECN];
                double c = dConnectBDPT(sc, cam, light, eye, lightSec, eyeSec, NS,
                                        s, t, hb, rng, spx, spy, isSplat, Lsec, nUpConn);
                if (nUpConn <= 0) continue;
                // The ×C de-hero boost is applied ONCE here, as 1/min(nUp_light, nUp_eye):
                // folding it into either subpath's throughput would square it whenever both
                // sides stayed multi-λ. nUpConn == 1 reproduces the scalar accumulation exactly.
                double ax = cx * c, ay = cy * c, az = cz * c;
                for (int i = 0; i + 1 < nUpConn; ++i) {
                    ax += cxS[i] * Lsec[i]; ay += cyS[i] * Lsec[i]; az += czS[i] * Lsec[i];
                }
                if (nUpConn > 1) {
                    double inv = 1.0 / nUpConn; ax *= inv; ay *= inv; az *= inv;
                }
                if (isSplat) {
                    size_t o = ((size_t)spy * resX + spx) * 3;
                    atomicAdd(&splatFilm[o + 0], ax);
                    atomicAdd(&splatFilm[o + 1], ay);
                    atomicAdd(&splatFilm[o + 2], az);
                } else {
                    size_t o = ((size_t)py * resX + px) * 3;
                    atomicAdd(&camFilm[o + 0], ax);
                    atomicAdd(&camFilm[o + 1], ay);
                    atomicAdd(&camFilm[o + 2], az);
                }
            }
    }
}

// ----------------------- final-gather sub-ray (mode M, -pmfg) ----------------
// Device twin of photonGatherSub (photonmap_render.h). One INDIRECT gather sub-ray shot from
// a diffuse visible point (visHit/visMat): it follows specular surfaces (monochromatic at
// `lambda`) exactly like dPhotonGather and terminates at
//   * the first diffuse/translucent hit y -> a radius density query at y, each photon
//     reflected off BOTH y (its material) AND the visible point (visMat) at the photon's
//     wavelength (spectral two-bounce colour bleed). pX/pY/pZ already fold norm/pi, so the
//     query needs no extra norm — just rho(y)*rho(vis) per photon;
//   * a finite EMITTER reached AFTER a specular bounce -> a monochromatic (camera-lambda)
//     sample reflected off the visible point (a straight hemisphere ray onto a light returns
//     0: that direct term is supplied by NEE at the visible point, so counting it here would
//     double-count — the specular-arrival gate mirrors backward.h);
//   * the ENVIRONMENT on ANY escape -> a monochromatic sample reflected off the visible point.
// Returns the XYZ radiance leaving the visible point toward the sub-ray origin for this one
// sampled direction (the caller averages K). Because the sub-ray is cosine-weighted and the
// visible BRDF is Lambertian, the cosine and 1/pi cancel to rho(vis), folded per photon
// (diffuse hit) or applied once (specular-arrival emitter/env). Keep in sync with photonGatherSub.
__device__ static void dPhotonGatherSub(const DScene& sc, const DPhotonMap& pm, int diffraction,
                                        DVec3 ro, DVec3 rd, Real lambda, double invPdfL,
                                        const DHit& visHit, const DMaterial& visMat, DRng& rng,
                                        double& oX, double& oY, double& oZ) {
    oX = oY = oZ = 0.0;
    double thr = 1.0;
    bool specularSeen = false;                           // any specular bounce so far?
    DMediumStack stk; stk.clear();
    const Real r2 = (Real)((double)pm.radius * (double)pm.radius);
    const int maxBounce = 32;
    for (int b = 0; b < maxBounce; ++b) {
        DHit h = closestHit(sc, ro, rd);
        if (h.valid) {                                   // Beer-Lambert in current medium
            int cm = stk.topMat();
            Real a = (cm >= 0) ? specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
            if (a > 0) thr *= exp(-(double)a * (double)h.t);
        }
        if (!h.valid) {                                  // escaped -> environment (direct off vis)
            if (sc.envIndex >= 0) {
                double envRad = (sc.env.scale != nullptr)
                                    ? dEnvRadiance(sc.env, rd, lambda)
                                    : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lambda);
                double rhoV = (double)clamp01(dDiffuseRho(sc, visMat, visHit, lambda));
                double e = thr * rhoV * envRad * invPdfL;
                oX += (double)cieX(lambda) * e; oY += (double)cieY(lambda) * e; oZ += (double)cieZ(lambda) * e;
            }
            return;
        }
        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (mp->type == D_MIX) {
            int child = dMixResolveChild(sc, *mp, h, rng.uniform());
            if (child < 0) return;
            mp = &sc.mats[child]; matId = child;
        }
        const DMaterial& m = *mp;

        int li = dEmitterForMat(sc, matId);
        if (li >= 0) {                                   // emitter
            if (specularSeen) {                          // specular-direct: NEE can't reach it
                double rhoV = (double)clamp01(dDiffuseRho(sc, visMat, visHit, lambda));
                double e = (double)specLookup(sc.emitters[li].emitSpd, lambda) * thr * rhoV * invPdfL
                         * dEmitPatMul(sc, m.emitPat, h);   // `emit pattern:` at this hit
                oX += (double)cieX(lambda) * e; oY += (double)cieY(lambda) * e; oZ += (double)cieZ(lambda) * e;
            }
            return;                                       // else: direct handled by NEE at vis
        }

        if (m.type == D_DIFFUSE || m.type == D_DIFFUSETRANSMIT || m.type == D_FLUORESCENT) {
            // Density estimate at y, folding the visible-point reflectance per photon wavelength.
            float gx = 0.f, gy = 0.f, gz = 0.f;
            int ix = (int)floor(((double)h.p.x - (double)pm.lo.x) / (double)pm.cellSize);
            int iy = (int)floor(((double)h.p.y - (double)pm.lo.y) / (double)pm.cellSize);
            int iz = (int)floor(((double)h.p.z - (double)pm.lo.z) / (double)pm.cellSize);
            ix = min(max(ix, 0), pm.nx - 1);
            iy = min(max(iy, 0), pm.ny - 1);
            iz = min(max(iz, 0), pm.nz - 1);
            for (int dz = -1; dz <= 1; ++dz) { int cz = iz + dz; if (cz < 0 || cz >= pm.nz) continue;
              for (int dy = -1; dy <= 1; ++dy) { int cy = iy + dy; if (cy < 0 || cy >= pm.ny) continue;
                for (int dx = -1; dx <= 1; ++dx) { int cx = ix + dx; if (cx < 0 || cx >= pm.nx) continue;
                  int c = (cz * pm.ny + cy) * pm.nx + cx;
                  for (int k = pm.cellStart[c]; k < pm.cellStart[c + 1]; ++k) {
                      const DGatherPhoton& ph = pm.photons[k];
                      DVec3 d = h.p - ph.pos;
                      if (dot(d, d) > r2) continue;
                      if (dot(ph.n, h.n) < (Real)0.5) continue;
                      float rhoY = (float)dDiffuseRho(sc, m, h, (Real)ph.lambda);
                      float rhoV = (float)dDiffuseRho(sc, visMat, visHit, (Real)ph.lambda);
                      float w = rhoY * rhoV;
                      gx += w * ph.pX; gy += w * ph.pY; gz += w * ph.pZ;
                  }
                }}}
            oX += (double)gx * thr; oY += (double)gy * thr; oZ += (double)gz * thr;
            return;
        }

        switch (m.type) {                                // specular walk (monochromatic)
            case D_MIRROR: {
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); break;
            }
            case D_GLOSSY: {
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                DVec3 o = sampleGlossy(reflectv(rd, h.n), dMatRoughness(sc, m, h), rng);
                if (dot(o, h.n) <= 0) return;
                ro = h.p + h.n * RAY_EPS; rd = o; break;
            }
            case D_DIELECTRIC: {
                DVec3 nro, nrd; dDielectricStep(sc, m, h, rd, lambda, rng, matId, stk, nro, nrd);
                ro = nro; rd = nrd; break;
            }
            case D_HALFMIRROR: {
                Real r = clamp01(dReflectSlot(sc, m, h, lambda));
                if (rng.uniform() < r) { ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); }
                else                   { ro = h.p + rd * RAY_EPS; }
                break;
            }
            case D_FILTER: {
                thr *= (double)clamp01(dTransmitSlot(sc, m, h, lambda));
                ro = h.p + rd * RAY_EPS; break;
            }
            default: {                                   // ThinFilm/Multilayer/Grating: approx reflect
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); break;
            }
        }
        specularSeen = true;                             // only specular cases reach here
        if (thr <= 0.0) return;
    }
}

// ----------------------- photon-map camera gather (mode M) -------------------
// Device twin of photonGather (photonmap_render.h). Follows a camera ray through specular
// surfaces (monochromatic at the sampled `lambda`); at the first diffuse / translucent hit it
// estimates reflected radiance either by a DIRECT radius density query into the uploaded
// photon map (fgRays == 0; each photon reflected at ITS OWN wavelength so the estimate is
// built directly in XYZ) or, when final gather is enabled (fgRays > 0 on a Diffuse visible
// point), by NEE direct lighting (bkNeeLight) plus K cosine-hemisphere gather sub-rays one
// bounce into the map (dPhotonGatherSub). Directly-viewed emitters and the environment (on
// gather-ray escape) are added as a monochromatic estimate at the sampled lambda. Beer-Lambert
// interior absorption is tracked exactly like bkRadiance. Keep in sync with photonGather.
__device__ static void dPhotonGather(const DScene& sc, const DPhotonMap& pm, int diffraction,
                                     DVec3 ro, DVec3 rd, Real lambda, double invPdfL, DRng& rng,
                                     int fgRays, double& oX, double& oY, double& oZ) {
    oX = oY = oZ = 0.0;
    double thr = 1.0;
    DMediumStack stk; stk.clear();   // nested-dielectric medium stack (empty = vacuum)
    // norm folded into pX/pY/pZ; radius^2 kept in Real — the distance test runs once per
    // VISITED photon (~85% of visits fail it), and on GeForce parts a double compare +
    // f2d convert issue at 1/64 rate, so keeping the test in FP32 matters.
    const Real r2 = (Real)((double)pm.radius * (double)pm.radius);
    const int maxBounce = 32;

    for (int b = 0; b < maxBounce; ++b) {
        DHit h = closestHit(sc, ro, rd);
        if (h.valid) {                                   // Beer-Lambert in current medium
            int cm = stk.topMat();
            Real a = (cm >= 0) ? specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
            if (a > 0) thr *= exp(-(double)a * (double)h.t);
        }
        if (!h.valid) {                                  // escaped -> environment
            // Direct env term on gather-ray escape (mirrors photonGather, fgRays==0). The
            // map already carries env's INDIRECT bounces (the deposit emits env photons);
            // this adds env's DIRECT contribution, monochromatic at the sampled lambda.
            if (sc.envIndex >= 0) {
                double envRad = (sc.env.scale != nullptr)
                                    ? dEnvRadiance(sc.env, rd, lambda)
                                    : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lambda);
                double e = thr * envRad * invPdfL;
                oX += (double)cieX(lambda) * e;
                oY += (double)cieY(lambda) * e;
                oZ += (double)cieZ(lambda) * e;
            }
            // Directly-viewed solar disc. This walk terminates at the first diffuse
            // vertex (the density estimate returns there), so any escape reaching here
            // is a camera ray or a specular chain — never a diffuse continuation that
            // the map / NEE already credited with the sun. (Host twin: photonmap_render.h.)
            if (sc.sunCount > 0) {
                double e = thr * dSunRadiance(sc, rd, lambda) * invPdfL;
                oX += (double)cieX(lambda) * e;
                oY += (double)cieY(lambda) * e;
                oZ += (double)cieZ(lambda) * e;
            }
            return;
        }

        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (mp->type == D_MIX) {
            int child = dMixResolveChild(sc, *mp, h, rng.uniform());
            if (child < 0) return;                       // absorbed by the mix
            mp = &sc.mats[child]; matId = child;
        }
        const DMaterial& m = *mp;

        int li = dEmitterForMat(sc, matId);
        if (li >= 0) {                                   // directly-viewed / specular-seen emitter
            double e = (double)specLookup(sc.emitters[li].emitSpd, lambda) * thr * invPdfL
                     * dEmitPatMul(sc, m.emitPat, h);    // `emit pattern:` at this hit
            oX += (double)cieX(lambda) * e;
            oY += (double)cieY(lambda) * e;
            oZ += (double)cieZ(lambda) * e;
            return;
        }

        if (m.type == D_DIFFUSE || m.type == D_DIFFUSETRANSMIT || m.type == D_FLUORESCENT) {
            if (fgRays > 0 && m.type == D_DIFFUSE) {
                // Jensen final gather (device twin of photonGather's fgRays branch): decouples
                // the directly-seen surface's sharpness from the gather radius by moving the
                // density-estimate blur one bounce away (to y, inside dPhotonGatherSub).
                //   (a) DIRECT lighting from finite emitters via low-variance NEE (bkNeeLight).
                double rhoVis = (double)clamp01(dDiffuseRho(sc, m, h, lambda));
                double direct = bkNeeLight(sc, h, (Real)rhoVis, invPdfL, lambda, rng);
                double dcie = thr * direct;
                oX += (double)cieX(lambda) * dcie;
                oY += (double)cieY(lambda) * dcie;
                oZ += (double)cieZ(lambda) * dcie;
                //   (b) INDIRECT (+ env + specular-direct) via K cosine-weighted hemisphere
                //       sub-rays, each querying the map ONE bounce away; the cosine/pdf and
                //       Lambertian 1/pi cancel to rho(vis), folded inside dPhotonGatherSub.
                double fx = 0.0, fy = 0.0, fz = 0.0;
                for (int k = 0; k < fgRays; ++k) {
                    DVec3 gro = h.p + h.n * RAY_EPS;
                    DVec3 grd = cosineHemisphere(h.n, rng);
                    double sx, sy, sz;
                    dPhotonGatherSub(sc, pm, diffraction, gro, grd, lambda, invPdfL, h, m, rng, sx, sy, sz);
                    fx += sx; fy += sy; fz += sz;
                }
                double inv = thr / (double)fgRays;
                oX += fx * inv; oY += fy * inv; oZ += fz * inv;
                return;
            }
            // Radius density estimate at the visible point, accumulated in XYZ (each photon
            // folded at its own wavelength): L_r = (1/N) sum_p rho(l_p)/pi * Phi_p / (pi r^2).
            // Everything but rho(l_p) is baked into the record's pX/pY/pZ (see DGatherPhoton),
            // so the per-photon work is the two rejection tests + one rho + three FMAs — all
            // FP32: a float sum of <= (cell occupancy) same-sign terms errs ~n*2^-24 relative,
            // far below the 8-bit output quantum, and FP64 FMAs would issue at 1/64 rate. The
            // per-sample total is promoted to double once at the end (film math stays double).
            float gx = 0.f, gy = 0.f, gz = 0.f;
            int ix = (int)floor(((double)h.p.x - (double)pm.lo.x) / (double)pm.cellSize);
            int iy = (int)floor(((double)h.p.y - (double)pm.lo.y) / (double)pm.cellSize);
            int iz = (int)floor(((double)h.p.z - (double)pm.lo.z) / (double)pm.cellSize);
            ix = min(max(ix, 0), pm.nx - 1);
            iy = min(max(iy, 0), pm.ny - 1);
            iz = min(max(iz, 0), pm.nz - 1);
            for (int dz = -1; dz <= 1; ++dz) { int cz = iz + dz; if (cz < 0 || cz >= pm.nz) continue;
              for (int dy = -1; dy <= 1; ++dy) { int cy = iy + dy; if (cy < 0 || cy >= pm.ny) continue;
                for (int dx = -1; dx <= 1; ++dx) { int cx = ix + dx; if (cx < 0 || cx >= pm.nx) continue;
                  int c = (cz * pm.ny + cy) * pm.nx + cx;
                  for (int k = pm.cellStart[c]; k < pm.cellStart[c + 1]; ++k) {
                      const DGatherPhoton& ph = pm.photons[k];
                      DVec3 d = h.p - ph.pos;
                      if (dot(d, d) > r2) continue;
                      if (dot(ph.n, h.n) < (Real)0.5) continue;   // reject cross-surface leakage
                      float rho = (float)dDiffuseRho(sc, m, h, (Real)ph.lambda);
                      gx += rho * ph.pX;
                      gy += rho * ph.pY;
                      gz += rho * ph.pZ;
                  }
                }}}
            oX += (double)gx * thr; oY += (double)gy * thr; oZ += (double)gz * thr;
            return;
        }

        switch (m.type) {                                // specular walk (monochromatic)
            case D_MIRROR: {
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); break;
            }
            case D_GLOSSY: {
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                DVec3 o = sampleGlossy(reflectv(rd, h.n), dMatRoughness(sc, m, h), rng);
                if (dot(o, h.n) <= 0) return;
                ro = h.p + h.n * RAY_EPS; rd = o; break;
            }
            case D_DIELECTRIC: {
                DVec3 nro, nrd; dDielectricStep(sc, m, h, rd, lambda, rng, matId, stk, nro, nrd);
                ro = nro; rd = nrd; break;
            }
            case D_HALFMIRROR: {
                Real r = clamp01(dReflectSlot(sc, m, h, lambda));
                if (rng.uniform() < r) { ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); }
                else                   { ro = h.p + rd * RAY_EPS; }
                break;
            }
            case D_FILTER: {
                thr *= (double)clamp01(dTransmitSlot(sc, m, h, lambda));
                ro = h.p + rd * RAY_EPS; break;
            }
            default: {                                   // ThinFilm/Multilayer/Grating: approx reflect
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); break;
            }
        }
        if (thr <= 0.0) return;
    }
}

// One thread per (pixel, sample); grid-strides over totalSamples. Mirrors kBackward's
// seeding (global sample index) so a chunked gather is decorrelated across chunks. The
// gather already returns XYZ, so (unlike kBackward) no cie(lambda) multiply is applied.
__global__ void kGather(DScene sc, DPhotonMap pm, DCamera cam, double* film, double* hits,
                        long long totalSamples, long long chunkSpp, long long sppTotal,
                        long long sampleBase, int resX, int diffraction, int fgRays,
                        unsigned long long seedBase) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    for (long long idx = g; idx < totalSamples; idx += G) {
        long long pix  = idx / chunkSpp;
        long long gidx = pix * sppTotal + sampleBase + (idx - pix * chunkSpp);
        DRng rng; rng.seed((unsigned long long)(gidx * 2 + 1), seedBase ^ (unsigned long long)gidx);
        int px = (int)(pix % resX);
        int py = (int)(pix / resX);

        double pdf = 0.0;
        Real lambda = dSampleSceneLambda(sc, rng, pdf);
        if (pdf <= 0.0) continue;
        double invPdfL = dInvPdfLambda(sc, lambda);

        DVec3 ro, rd;
        Real jx = rng.uniform(), jy = rng.uniform();
        dGenRay(cam, px, py, jx, jy, ro, rd);            // pinhole only (lens cams gated to CPU)

        double oX, oY, oZ;
        dPhotonGather(sc, pm, diffraction, ro, rd, lambda, invPdfL, rng, fgRays, oX, oY, oZ);
        size_t o = ((size_t)py * resX + px) * 3;
        atomicAdd(&film[o + 0], oX);
        atomicAdd(&film[o + 1], oY);
        atomicAdd(&film[o + 2], oZ);
        if (hits) atomicAdd(&hits[(size_t)py * resX + px], 1.0);
    }
}

// ============================ device: SPPM (mode S) ============================
// Device twin of sppm_render.h. SPPM runs REPEATED bounded photon passes with a per-pixel
// shrinking gather radius; per-pixel progressive state (tau/radius/nAcc/directSum + the
// current pass's visible point) lives resident on the device across passes. Each of the
// three phases below is one kernel; the deposit + host grid build reuse the mode-M path.

// Trace one camera ray to its first diffuse/translucent hit (the "visible point"), following
// specular surfaces exactly like dPhotonGather / CPU sppmVisiblePoint. Returns the specular
// throughput in `thrOut` and the diffuse hit in `vp` (with the ORIGINAL matId, matching CPU);
// emitter/env radiance reached directly or through specular is added to directL (XYZ, a
// monochromatic MC estimate at the sampled lambda). `vpValid` is false when the ray
// terminated (light, env, or absorption) without reaching a diffuse surface.
__device__ static void dSppmVisiblePoint(const DScene& sc, DVec3 ro, DVec3 rd, Real lambda,
                                         double invPdfL, DRng& rng, int maxBounce,
                                         DHit& vp, double& thrOut, bool& vpValid,
                                         double& dX, double& dY, double& dZ) {
    thrOut = 0.0; vpValid = false; dX = dY = dZ = 0.0;
    double thr = 1.0;
    DMediumStack stk; stk.clear();
    for (int b = 0; b < maxBounce; ++b) {
        DHit h = closestHit(sc, ro, rd);
        if (h.valid) {                                   // Beer-Lambert in current medium
            int cm = stk.topMat();
            Real a = (cm >= 0) ? specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
            if (a > 0) thr *= exp(-(double)a * (double)h.t);
        }
        if (!h.valid) {                                  // escaped -> environment (direct)
            if (sc.envIndex >= 0) {
                double envRad = (sc.env.scale != nullptr)
                                    ? dEnvRadiance(sc.env, rd, lambda)
                                    : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lambda);
                double e = thr * envRad * invPdfL;
                dX += (double)cieX(lambda) * e;
                dY += (double)cieY(lambda) * e;
                dZ += (double)cieZ(lambda) * e;
            }
            // Directly-viewed solar disc (camera / specular escapes only — a diffuse
            // vertex stores a hit point and returns before it can reach here).
            // Host twin: sppm_render.h.
            if (sc.sunCount > 0) {
                double e = thr * dSunRadiance(sc, rd, lambda) * invPdfL;
                dX += (double)cieX(lambda) * e;
                dY += (double)cieY(lambda) * e;
                dZ += (double)cieZ(lambda) * e;
            }
            return;
        }
        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (mp->type == D_MIX) {
            int child = dMixResolveChild(sc, *mp, h, rng.uniform());
            if (child < 0) return;                       // absorbed by the mix
            mp = &sc.mats[child]; matId = child;
        }
        const DMaterial& m = *mp;

        int li = dEmitterForMat(sc, matId);
        if (li >= 0) {                                   // directly-viewed / specular-seen emitter
            double e = (double)specLookup(sc.emitters[li].emitSpd, lambda) * thr * invPdfL
                     * dEmitPatMul(sc, m.emitPat, h);    // `emit pattern:` at this hit
            dX += (double)cieX(lambda) * e;
            dY += (double)cieY(lambda) * e;
            dZ += (double)cieZ(lambda) * e;
            return;
        }

        if (m.type == D_DIFFUSE || m.type == D_DIFFUSETRANSMIT || m.type == D_FLUORESCENT) {
            vp = h; thrOut = thr; vpValid = true;        // record the visible point (parent matId)
            return;
        }

        switch (m.type) {                                // specular walk (monochromatic)
            case D_MIRROR: {
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); break;
            }
            case D_GLOSSY: {
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                DVec3 o = sampleGlossy(reflectv(rd, h.n), dMatRoughness(sc, m, h), rng);
                if (dot(o, h.n) <= 0) return;
                ro = h.p + h.n * RAY_EPS; rd = o; break;
            }
            case D_DIELECTRIC: {
                DVec3 nro, nrd; dDielectricStep(sc, m, h, rd, lambda, rng, matId, stk, nro, nrd);
                ro = nro; rd = nrd; break;
            }
            case D_HALFMIRROR: {
                Real r = clamp01(dReflectSlot(sc, m, h, lambda));
                if (rng.uniform() < r) { ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); }
                else                   { ro = h.p + rd * RAY_EPS; }
                break;
            }
            case D_FILTER: {
                thr *= (double)clamp01(dTransmitSlot(sc, m, h, lambda));
                ro = h.p + rd * RAY_EPS; break;
            }
            default: {                                   // ThinFilm/Multilayer/Grating: approx reflect
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); break;
            }
        }
        if (thr <= 0.0) return;
    }
}

// SPPM per-pixel state (structure-of-arrays on the device). One entry per pixel.
struct DSppmState {
    double* tau;       // npix*3  accumulated radius-rescaled flux (XYZ)
    double* radius;    // npix    current gather radius R_i
    double* nAcc;      // npix    accumulated photon count N_i
    double* directSum; // npix*3  direct/specular-viewed emitter + env, summed over passes
    DHit*   vpHit;     // npix    this pass's visible point (diffuse hit)
    double* vpThr;     // npix    specular throughput camera -> visible point
    unsigned char* vpValid; // npix
};

// Phase 1: resample each pixel's camera visible point for this pass and accumulate its
// direct term. One thread per pixel. Seeds mirror kBackward/kGather (global sample index).
__global__ void kSppmVisiblePoint(DScene sc, DCamera cam, DSppmState st, int resX, int resY,
                                  int maxBounce, unsigned long long seedBase, long long passIdx) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    long long npix = (long long)resX * resY;
    for (long long pix = g; pix < npix; pix += G) {
        int px = (int)(pix % resX), py = (int)(pix / resX);
        // Distinct stream per (pixel, pass).
        unsigned long long s = (unsigned long long)(pix) * 0x9E3779B97F4A7C15ULL
                             + (unsigned long long)passIdx * 0xD1B54A32D192ED03ULL;
        DRng rng; rng.seed(s * 2 + 23, seedBase ^ s);

        double pdf = 0.0;
        Real lambda = dSampleSceneLambda(sc, rng, pdf);
        st.vpValid[pix] = 0;
        if (pdf <= 0.0) return;
        double invPdfL = dInvPdfLambda(sc, lambda);

        DVec3 ro, rd;
        Real jx = rng.uniform(), jy = rng.uniform();
        dGenRay(cam, px, py, jx, jy, ro, rd);            // pinhole only (lens cams gated to CPU)

        DHit vp; double thr = 0.0; bool valid = false; double dX, dY, dZ;
        dSppmVisiblePoint(sc, ro, rd, lambda, invPdfL, rng, maxBounce, vp, thr, valid, dX, dY, dZ);
        st.vpHit[pix] = vp; st.vpThr[pix] = thr; st.vpValid[pix] = valid ? 1 : 0;
        st.directSum[pix * 3 + 0] += dX;
        st.directSum[pix * 3 + 1] += dY;
        st.directSum[pix * 3 + 2] += dZ;
    }
}

// Phase 3: gather each valid pixel's visible point at its current radius, then apply the
// shared-statistics progressive radius/flux update. One thread per pixel. The photon records
// carry pX/pY/pZ = cie(lambda)*power/pi (NO area/nEmitted fold — those depend on the current
// per-pixel radius and are applied at resolve), so phi? += rho(lambda_p) * p?.
__global__ void kSppmGather(DScene sc, DPhotonMap pm, DSppmState st, int resX, int resY,
                            double alpha) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    long long npix = (long long)resX * resY;
    for (long long pix = g; pix < npix; pix += G) {
        if (!st.vpValid[pix]) continue;
        const DHit h = st.vpHit[pix];
        const DMaterial& m = sc.mats[h.matId];           // parent matId, matching CPU SPPM
        double R = st.radius[pix];
        Real r2 = (Real)(R * R);
        float gx = 0.f, gy = 0.f, gz = 0.f;
        double M = 0.0;
        int ix = (int)floor(((double)h.p.x - (double)pm.lo.x) / (double)pm.cellSize);
        int iy = (int)floor(((double)h.p.y - (double)pm.lo.y) / (double)pm.cellSize);
        int iz = (int)floor(((double)h.p.z - (double)pm.lo.z) / (double)pm.cellSize);
        ix = min(max(ix, 0), pm.nx - 1);
        iy = min(max(iy, 0), pm.ny - 1);
        iz = min(max(iz, 0), pm.nz - 1);
        for (int dz = -1; dz <= 1; ++dz) { int cz = iz + dz; if (cz < 0 || cz >= pm.nz) continue;
          for (int dy = -1; dy <= 1; ++dy) { int cy = iy + dy; if (cy < 0 || cy >= pm.ny) continue;
            for (int dx = -1; dx <= 1; ++dx) { int cx = ix + dx; if (cx < 0 || cx >= pm.nx) continue;
              int c = (cz * pm.ny + cy) * pm.nx + cx;
              for (int k = pm.cellStart[c]; k < pm.cellStart[c + 1]; ++k) {
                  const DGatherPhoton& ph = pm.photons[k];
                  DVec3 d = h.p - ph.pos;
                  if (dot(d, d) > r2) continue;
                  if (dot(ph.n, h.n) < (Real)0.5) continue;   // reject cross-surface leakage
                  float rho = (float)dDiffuseRho(sc, m, h, (Real)ph.lambda);
                  gx += rho * ph.pX;
                  gy += rho * ph.pY;
                  gz += rho * ph.pZ;
                  M += 1.0;
              }
            }}}
        // Shared-statistics PPM update (Hachisuka 2008).
        double nAcc = st.nAcc[pix];
        double Nnew = nAcc + alpha * M;
        double denom = nAcc + M;
        double ratio2 = (denom > 0.0) ? (Nnew / denom) : 1.0;   // (R'/R)^2
        double thr = st.vpThr[pix];
        st.tau[pix * 3 + 0] = (st.tau[pix * 3 + 0] + (double)gx * thr) * ratio2;
        st.tau[pix * 3 + 1] = (st.tau[pix * 3 + 1] + (double)gy * thr) * ratio2;
        st.tau[pix * 3 + 2] = (st.tau[pix * 3 + 2] + (double)gz * thr) * ratio2;
        st.radius[pix] = R * sqrt(ratio2);
        st.nAcc[pix] = Nnew;
    }
}

// Resolve the accumulated state into a film: L = directSum/passes + tau/(pi R^2 Nemit).
__global__ void kSppmResolve(DSppmState st, double* film, int resX, int resY,
                             long long passes, double Nemit) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    long long npix = (long long)resX * resY;
    double invPasses = (passes > 0) ? 1.0 / (double)passes : 0.0;
    for (long long pix = g; pix < npix; pix += G) {
        double lx = st.directSum[pix * 3 + 0] * invPasses;
        double ly = st.directSum[pix * 3 + 1] * invPasses;
        double lz = st.directSum[pix * 3 + 2] * invPasses;
        double R = st.radius[pix];
        double area = DPI * R * R;
        if (area > 0.0 && Nemit > 0.0) {
            double inv = 1.0 / (area * Nemit);
            lx += st.tau[pix * 3 + 0] * inv;
            ly += st.tau[pix * 3 + 1] * inv;
            lz += st.tau[pix * 3 + 2] * inv;
        }
        film[pix * 3 + 0] = lx;
        film[pix * 3 + 1] = ly;
        film[pix * 3 + 2] = lz;
    }
}

// ============================ device: VCM / UPS (mode U) ============================
// Device twin of vcm.h. VCM combines BDPT vertex CONNECTIONS with photon-map vertex
// MERGING under one balance-heuristic MIS. Per pass: (1) kVcmLight traces one light
// subpath per pixel, stores its connectible vertices into a per-path slab and splats its
// connect-to-camera (t=1) contributions; the host compacts the slab in path order and
// builds a uniform hash grid over ALL light vertices (cell = merge radius). (2) kVcmCamera
// traces one camera subpath per pixel doing emission (s=0), NEE (s=1), vertex connection to
// the PAIRED light subpath, and merging from the grid; it accumulates the pass image into
// the persistent `accum` sum. The resolve divides by the pass count. Mirrors SmallVCM's
// dVCM/dVC/dVM bookkeeping (misArrival / misScatter inlined; Mis(x)=x, so the wrappers are
// dropped). Scope (gated in cudaVcmSupported): surfaces only, area/sphere/tube/mesh Lambertian
// lights PLUS the two delta emitters (`spot`, `sun` — since 0.127.0, with dVC starting at 0,
// the sun's planar first-edge density and its escaped-ray disc), rectilinear pinhole camera,
// NO participating media (media.empty()).

// One stored light-subpath vertex (device twin of vcm.h LightVertex). Stores INDICES + the
// per-hit texcoords (u,v) so textured/patterned/record BSDFs evaluate per-hit like M9's BDPT.
struct DVcmLV {
    DVec3  p, ns, ng, wo;
    double beta;
    double dVCM, dVC, dVM;
    double cx, cy, cz;     // cie(lambda) cached at store time (bit-identical per gather)
    float  lambda;
    int    matId, edges;
    Real   u, v;
    Real   curv;           // O3 mean curvature — here for the same reason u/v are: the
                           // connection BSDF is re-evaluated at this vertex, so a
                           // curvature-driven material must not read a different value
                           // in VCM than in every other integrator. Costs 8 B/slot after
                           // alignment (the slab goes 128 -> 136 B), which is the price
                           // of that consistency on the largest allocation in a session.
    int    nUp;            // hero wavelengths still live here (1 == de-hero'd / single-λ)
};

// The SECONDARY hero wavelengths of a stored light vertex, one slot per secondary. Kept in a
// PARALLEL slab (`lvSec[(i*vcmCap + k)*secStride + j]`, secStride == C-1) rather than inline in
// DVcmLV, exactly like GPU BDPT's `pathSec`: the light-vertex slab is `npix * vcmCap * 128 B`
// and is by far the largest allocation in a VCM session, so a single-λ run (`-heroc 1`) must
// allocate the sec slab at zero and pay nothing. Only `beta` and `lam` are stored — cie(λ) is
// recomputed at gather time, which is bit-identical to caching it (DVcmLV::cx is likewise just
// `(double)cieX(lambda)`) and halves the slot to 16 B.
struct DVcmSec {
    double beta;
    float  lam;
};

// Uniform hash grid over the compacted light vertices (device twin of vcm.h VcmGrid). `lv`
// is the compact array in path order; `order` holds indices into it in cell-contiguous order.
struct DVcmGrid {
    const DVcmLV* lv;       // compact light-vertex array (path order)
    int    nLV;
    const int* cellStart;   // nCells+1
    const int* order;       // nLV vertex indices, cell-contiguous
    DVec3  lo;
    double cell;            // == merge radius
    int    nx, ny, nz;
};

// Per-pass constants (device twin of vcm.h PassCtx; media/diffraction handled elsewhere).
struct DVcmCtx {
    double radius, nLightPaths, misVcWeight, misVmWeight, vmNorm, imagePlaneDist;
    int    maxDepth;
};

// Reconstruct a minimal DVertex for the per-hit BSDF helpers (dBsdfF / dBsdfPdf reconstruct a
// DHit from it — they read p/ns/ng/matId/u/v).
__device__ static inline DVertex dVertFromHit(const DHit& h, int matId) {
    DVertex v; v.type = BV_SURFACE; v.p = h.p; v.ns = h.n; v.ng = h.ng;
    v.beta = 0; v.pdfFwd = 0; v.pdfRev = 0; v.delta = 0; v.matId = matId; v.lightIdx = -1;
    v.emitPatW = (Real)1;   // BSDF-only helper vertex; never read for emission
    v.mediumG = 0; v.mediumId = -1; v.u = h.u; v.v = h.v; v.curv = h.curv; return v;
}
__device__ static inline DVertex dVertFromLV(const DVcmLV& lv) {
    DVertex v; v.type = BV_SURFACE; v.p = lv.p; v.ns = lv.ns; v.ng = lv.ng;
    v.beta = 0; v.pdfFwd = 0; v.pdfRev = 0; v.delta = 0; v.matId = lv.matId; v.lightIdx = -1;
    v.emitPatW = (Real)1;   // BSDF-only helper vertex; never read for emission
    v.mediumG = 0; v.mediumId = -1; v.u = lv.u; v.v = lv.v; v.curv = lv.curv; return v;
}

// Sample a scattering continuation at a surface vertex (device twin of vcm.h scatterSample).
// Returns wi/betaFactor/pdfW/pdfRevW/cosThetaOut/delta/terminate. Uses per-hit slots
// (dReflectSlot / dMatRoughness / dDiffuseRho) so it matches the CPU VCM exactly. Media are
// out of scope; `stk` still resolves the nested-dielectric exterior IOR (dDielectricStep).
//
// HERO BUNDLE (Wilkie 2014; the four shared policies live in hero.h). When `nUp > 1` the caller
// carries nUp wavelengths on this one ray: `lamAll` is the bundle (lamAll[0] == the hero ==
// `lambda`) and on return `secF[i]` is the ABSOLUTE throughput factor for secondary i — NOT a
// ratio to the hero's, which is undefined exactly where it matters most (a gel whose T(λ_hero)
// is 0 while a secondary is wide open). `secChromatic` says secF was filled at all; the
// λ-independent lobes leave it false and the caller reuses `betaFactor` for every λ.
// `keepBundle` marks the delta lobes that nevertheless pick their continuation WITHOUT
// consulting λ (Mirror reflects, Filter passes straight through), so they opt out of the
// caller's `if (delta) nUp = 1` collapse.
//
// NOTE the `<= 0` early terminations the scalar version did for a zero reflectance /
// transmittance are GONE from the chromatic lobes: the caller applies a max-over-live-λ test
// instead, which at nUp == 1 is exactly the old scalar test (`mxF == betaFactor`). Grating
// keeps its `r <= 0` bail because it gates an RNG consumer (gratingDiffract).
__device__ static void dVcmScatter(const DScene& sc, const DMaterial& m, const DHit& h,
                                   const DVec3& rd, Real lambda, DRng& rng, int matId,
                                   DMediumStack& stk, int diffraction,
                                   DVec3& wi, double& betaFactor, double& pdfW, double& pdfRevW,
                                   double& cosThetaOut, bool& delta, bool& terminate,
                                   const Real* lamAll = nullptr, int nUp = 1,
                                   double* secF = nullptr, bool* secChromatic = nullptr,
                                   bool* keepBundle = nullptr) {
    DVertex vt = dVertFromHit(h, matId);
    const DVec3& ns = h.n;
    DVec3 wo = normalize(rd * (Real)-1);
    wi = DVec3(0, 0, 0); betaFactor = 0; pdfW = 0; pdfRevW = 0; cosThetaOut = 0;
    delta = false; terminate = false;
    const int nSec = (secF && lamAll && nUp > 1) ? nUp - 1 : 0;   // secondaries to fill
    if (secChromatic) *secChromatic = false;
    if (keepBundle)   *keepBundle = false;
    switch (m.type) {
        case D_DIFFUSE:
        case D_FLUORESCENT: {
            wi = cosineHemisphere(ns, rng);
            if (dot(wi, ns) <= 0) { terminate = true; break; }
            double rho = clamp01(dDiffuseRho(sc, m, h, lambda));
            pdfW = dBsdfPdf(sc, vt, wo, wi, lambda);
            pdfRevW = dBsdfPdf(sc, vt, wi, wo, lambda);
            betaFactor = rho;                     // rho <= 0 is caught by the caller's max test
            if (nSec) {
                *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(dDiffuseRho(sc, m, h, lamAll[i + 1]));
            }
            break;
        }
        case D_GLOSSY: {
            DVec3 mdir = reflectv(rd, ns);
            wi = sampleGlossy(mdir, dMatRoughness(sc, m, h), rng);
            if (dot(wi, ns) <= 0) { terminate = true; break; }
            double r = clamp01(dReflectSlot(sc, m, h, lambda));
            pdfW = dBsdfPdf(sc, vt, wo, wi, lambda);
            pdfRevW = dBsdfPdf(sc, vt, wi, wo, lambda);
            betaFactor = r;
            if (pdfW <= 0) terminate = true;      // r <= 0 is caught by the caller's max test
            // The glossy LOBE (mirror direction + roughness exponent) carries no wavelength
            // dependence, so the whole bundle follows the sampled direction and only the
            // reflectance differs per λ.
            if (nSec) {
                *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(dReflectSlot(sc, m, h, lamAll[i + 1]));
            }
            break;
        }
        case D_DIFFUSETRANSMIT: {
            double rhoR, rhoT; dDiffuseTransmitAlbedos(sc, m, h, lambda, rhoR, rhoT);
            double tot = rhoR + rhoT;
            if (tot <= 0.0) { terminate = true; break; }
            const bool reflLobe = (rng.uniform() * tot < rhoR);
            if (reflLobe) wi = cosineHemisphere(ns, rng);
            else          wi = cosineHemisphere(ns * (Real)-1, rng);
            pdfW = dBsdfPdf(sc, vt, wo, wi, lambda);
            pdfRevW = dBsdfPdf(sc, vt, wi, wo, lambda);
            betaFactor = tot;
            if (pdfW <= 0) terminate = true;
            // The lobe was CHOSEN by the hero's albedo split, so each secondary divides by the
            // HERO's albedo for that lobe: f_i·cos/pdf_hero = rho_i(lobe)·tot_hero/rho_hero(lobe).
            if (nSec) {
                *secChromatic = true;
                for (int i = 0; i < nSec; ++i) {
                    double rR, rT; dDiffuseTransmitAlbedos(sc, m, h, lamAll[i + 1], rR, rT);
                    double num = reflLobe ? rR   : rT;
                    double den = reflLobe ? rhoR : rhoT;
                    secF[i] = (den > 0.0) ? num * tot / den : 0.0;
                }
            }
            break;
        }
        case D_MIRROR: {
            double r = clamp01(dReflectSlot(sc, m, h, lambda));
            wi = reflectv(rd, ns); betaFactor = r; delta = true;
            // The mirror direction is the same for every λ, so the bundle survives; only the
            // reflectance is per-λ (cf. Glossy, the rough version of this).
            if (nSec) {
                *keepBundle = true; *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(dReflectSlot(sc, m, h, lamAll[i + 1]));
            }
            break;
        }
        case D_DIELECTRIC: {
            DVec3 nro, nrd; dDielectricStep(sc, m, h, rd, lambda, rng, matId, stk, nro, nrd);
            wi = nrd; betaFactor = 1.0; delta = true;
            break;
        }
        case D_HALFMIRROR: {
            double r = clamp01(dReflectSlot(sc, m, h, lambda));
            if (rng.uniform() < r) wi = reflectv(rd, ns); else wi = rd;
            betaFactor = 1.0; delta = true;
            break;
        }
        case D_FILTER: {
            double t = clamp01(dTransmitSlot(sc, m, h, lambda));
            wi = rd; betaFactor = t; delta = true;   // t <= 0 -> caller's max test
            // Straight-through for every λ, so the bundle survives — and a gel filter is
            // exactly where the per-λ transmittance spread is largest, i.e. the case that
            // benefits most from NOT de-heroing, AND the case that forces the absolute
            // (rather than ratio) secF, since T(λ_hero) is legitimately 0 across most of a
            // Wratten passband.
            if (nSec) {
                *keepBundle = true; *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(dTransmitSlot(sc, m, h, lamAll[i + 1]));
            }
            break;
        }
        case D_THINFILM: {
            DVec3 nro, nrd;
            if (!thinFilmInterface(sc, m, h, rd, lambda, rng, nro, nrd)) { terminate = true; break; }
            wi = nrd; betaFactor = 1.0; delta = true;
            break;
        }
        case D_MULTILAYER: {
            DVec3 nro, nrd;
            if (!multilayerInterface(m, h, rd, lambda, rng, nro, nrd)) { terminate = true; break; }
            wi = nrd; betaFactor = 1.0; delta = true;
            break;
        }
        case D_GRATING: {
            double r = clamp01(dReflectSlot(sc, m, h, lambda));
            if (r <= 0) { terminate = true; break; }
            DVec3 nro, nrd;
            if (!gratingDiffract(m, h, rd, lambda, diffraction, rng, nro, nrd)) { terminate = true; break; }
            wi = nrd; betaFactor = r; delta = true;
            break;
        }
        default: terminate = true; break;
    }
    if (!terminate) cosThetaOut = fabs(ddot(wi, ns));
}

// Phase 1: one light subpath per pixel. Stores connectible vertices into the per-path slab
// `lvSlab[i*vcmCap + k]` (count in `lvCount[i]`), splats connect-to-camera contributions into
// `splat` (atomic, W*H*3 XYZ), and records this path index's wavelength BUNDLE (shared with the
// camera path of the SAME index) into lamBuf/invLamBuf at stride C. Device twin of vcm.h
// traceLightSubpath.
//
// Templated on the number of SECONDARY hero slots exactly like kBdptT: the scalar
// instantiation kVcmLightT<0> sizes every per-λ array at 1, so a `-heroc 1` run pays zero extra
// registers/local memory and stays bit-identical to the pre-hero kernel.
template <int NS>
__global__ void kVcmLightT(DScene sc, DCamera cam, int diffraction, DVcmCtx ctx,
                           DVcmLV* lvSlab, DVcmSec* lvSec, int secStride, int heroC,
                           int* lvCount, double* splat,
                           Real* lamBuf, double* invLamBuf, int resX, int resY, int vcmCap,
                           unsigned long long seedBase, long long passIdx) {
    constexpr int SECN = (NS > 0) ? NS : 1;
    const int C = (NS > 0) ? heroC : 1;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    long long npix = (long long)resX * resY;
    for (long long i = g; i < npix; i += G) {
        lvCount[i] = 0;
        unsigned long long s = (unsigned long long)i * 0x9E3779B97F4A7C15ULL
                             + (unsigned long long)passIdx * 0xD1B54A32D192ED03ULL;
        DRng rng; rng.seed(s * 2 + 55, seedBase ^ s);

        // Hero + C-1 stratified secondaries from ONE base uniform (u + k/C wrapped into [0,1)),
        // pushed through the same inverse CDF as the scalar draw — so C == 1 consumes the
        // identical single rng variate the pre-hero kernel did.
        Real lamAll[SECN + 1];
        double invAll[SECN + 1];
        {
            double pdfL = 0.0;
            if (C > 1) {
                double u = (double)rng.uniform();
                lamAll[0] = dSampleSceneLambdaU(sc, u, pdfL);
                invAll[0] = (pdfL > 0.0) ? dInvPdfLambda(sc, lamAll[0]) : 0.0;
                for (int k = 1; k < C; ++k) {
                    double uu = u + (double)k / C;
                    if (uu >= 1.0) uu -= 1.0;
                    double pdfK = 0.0;
                    lamAll[k] = dSampleSceneLambdaU(sc, uu, pdfK);
                    invAll[k] = (pdfK > 0.0) ? dInvPdfLambda(sc, lamAll[k]) : 0.0;
                }
            } else {
                lamAll[0] = dSampleSceneLambda(sc, rng, pdfL);
                invAll[0] = (pdfL > 0.0) ? dInvPdfLambda(sc, lamAll[0]) : 0.0;
            }
        }
        const Real lambda = lamAll[0];
        const double invPdfLambda = invAll[0];
        for (int k = 0; k < C; ++k) { lamBuf[i * C + k] = lamAll[k]; invLamBuf[i * C + k] = invAll[k]; }
        if (invPdfLambda <= 0.0) continue;
        if (sc.nEmitters == 0 || sc.totalPower <= 0.0) continue;
        int nUp = C;                                  // wavelengths still riding this ray

        int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
        const DEmitter& em = sc.emitters[ei];
        if (em.shape == 3 || em.collimated) continue;      // env / collimated: out of VCM scope
        const bool isDelta    = dIsDeltaEmitter(em);       // spot / sun: no s=0 strategy
        const bool isInfinite = dIsInfiniteEmitter(em);    // sun: planar first-edge density
        Real u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 y, nOut, dir;
        // Per-shape emission sampling, the device twin of `vcm.h`'s traceLightSubpath.
        // `emitScale` is the achromatic factor that multiplies the emitted SPECTRUM only and
        // never a density (the `emit pattern:` weight for an area light, the smoothstep
        // penumbra for a spot, 1 for a sun); `emitCos` is |cos| at the light, which is
        // exactly 1 for both delta lights because their emission "normal" IS the direction.
        // Keeping emitScale out of every pdf is what lets directPdf / emissionPdfW stay
        // exactly what the camera kernel's s=0 / s=1 terms assume.
        double emitScale = 1.0, emitCos = 1.0, pdfPos = 1.0, pdfDirW = 0.0;
        if (isDelta) {
            // Both delta lights emit uniformly inside a cone about beamDir (pdfW = 1/Omega).
            // NOTE spotOmega is the FALLOFF-weighted solid angle for a spot, so the sampling
            // cone has to be recomputed here (the two coincide only for a sun).
            double coneSolid = 2.0 * DPI * (1.0 - em.spotCosOuter);
            if (coneSolid <= 0.0) continue;
            pdfDirW = 1.0 / coneSolid;
            dir = dSunSampleCone(em, em.beamDir, (double)u1, (double)u2);
            if (isInfinite) {
                // Origin on a disc of radius sceneRadius, one radius upstream of the scene
                // centre and perpendicular to beamDir: a PLANAR density 1/(pi R^2).
                double R = sc.sceneRadius;
                if (R <= 0.0) continue;
                DVec3 tt, bb; onb(em.beamDir, tt, bb);
                double rr = R * sqrt((double)rng.uniform()), phi = 2.0 * DPI * (double)rng.uniform();
                y = sc.sceneCenter + tt * (Real)(rr * cos(phi)) + bb * (Real)(rr * sin(phi))
                  - em.beamDir * (Real)R;
                pdfPos = 1.0 / (DPI * R * R);
            } else {
                y = em.origin;                             // delta position: pdfPos == 1
                emitScale = spotFalloff(ddot(dir, em.beamDir), em.spotCosInner, em.spotCosOuter);
                if (emitScale <= 0.0) continue;
            }
            nOut = dir;
        } else {
            // The sampled point's `emit pattern:` factor (1.0, and a bit-identical draw, when
            // unpatterned) scales the radiance the light subpath starts with — the MIS pdfs
            // are untouched, exactly as in dGenLightSubpath.
            emitScale = dEmitterSamplePointPat(sc, em, u1, u2, y, nOut);
            if (em.area <= 0.0) continue;
            pdfPos = 1.0 / em.area;
            dir = cosineHemisphere(nOut, rng);
            emitCos = ddot(nOut, dir);
            if (emitCos <= 0.0) continue;
            pdfDirW = emitCos / DPI;
        }
        double Le = (double)specLookup(em.emitSpd, lambda) * invPdfLambda * emitScale;
        // Max-over-live-λ: the hero can legitimately sit in a gap of the emission spectrum
        // while a secondary is on it. nUp == 1 -> empty loop -> mxLe == Le, the old test.
        double LeSec[SECN], mxLe = Le;
        for (int k = 0; k + 1 < nUp; ++k) {
            LeSec[k] = (double)specLookup(em.emitSpd, lamAll[k + 1]) * invAll[k + 1] * emitScale;
            if (LeSec[k] > mxLe) mxLe = LeSec[k];
        }
        if (mxLe <= 0.0) continue;
        double pdfChoice = em.power / sc.totalPower;
        if (pdfPos <= 0.0 || pdfDirW <= 0.0 || pdfChoice <= 0.0) continue;

        double emissionPdfW = pdfPos * pdfDirW * pdfChoice;
        if (emissionPdfW <= 0.0) continue;
        // The density NEE (s=1) would have used for THIS sample, in the measure that strategy
        // works in: an area density for a finite surface light, the solid-angle 1/Omega for
        // the infinitely distant sun, and 1 for a delta position (SmallVCM's PointLight::Emit
        // reports oDirectPdfA = 1). dVCM then comes out as Omega for a spot and pi R^2 for a
        // sun — exactly SmallVCM's PointLight / DirectionalLight constants.
        double directPdfW = pdfChoice * (isInfinite ? pdfDirW : (isDelta ? 1.0 : pdfPos));

        double beta = Le * emitCos / emissionPdfW;
        // MIS bookkeeping is the HERO's for the whole bundle: every sampling density in this
        // renderer (cosine / glossy-lobe / emitter pdfs) is wavelength-INDEPENDENT, so one
        // dVCM/dVC/dVM triple serves every λ — only the throughput VALUES differ.
        // A delta light cannot be hit by BSDF sampling, so dVC (and the dVM built from it)
        // start at ZERO: that is what DROPS the unsamplable strategies from the balance
        // heuristic instead of under-weighting the ones that do work.
        double dVCM = directPdfW / emissionPdfW;
        double dVC  = isDelta ? 0.0 : (emitCos / emissionPdfW);
        double dVM  = dVC * ctx.misVcWeight;

        double betaSec[SECN];
        double cieSx[SECN], cieSy[SECN], cieSz[SECN];
        for (int k = 0; k + 1 < nUp; ++k) {
            betaSec[k] = LeSec[k] * emitCos / emissionPdfW;
            cieSx[k] = (double)cieX(lamAll[k + 1]);
            cieSy[k] = (double)cieY(lamAll[k + 1]);
            cieSz[k] = (double)cieZ(lamAll[k + 1]);
        }
        double cieLx = (double)cieX(lambda), cieLy = (double)cieY(lambda), cieLz = (double)cieZ(lambda);
        DMediumStack stk; stk.clear();
        DVec3 prevP = y;
        DVec3 ro = y + nOut * (Real)1e-6;
        DVec3 rd = dir;
        int stored = 0;

        for (int edges = 1; edges <= ctx.maxDepth; ++edges) {
            DHit h = closestHit(sc, ro, rd);
            if (!h.valid) break;                          // escaped (no env in scope)
            {                                             // colored-glass Beer-Lambert (delta chains)
                int cm = stk.topMat();
                double a = (cm >= 0) ? (double)specLookup(sc.mats[cm].absorb, lambda) : 0.0;
                if (a > 0.0) beta *= exp(-a * (double)h.t);
                // Per-λ absorption: this IS the colour of coloured glass, so it must not be
                // evaluated at the hero alone.
                for (int k = 0; k + 1 < nUp; ++k) {
                    double ak = (cm >= 0) ? (double)specLookup(sc.mats[cm].absorb, lamAll[k + 1]) : 0.0;
                    if (ak > 0.0) betaSec[k] *= exp(-ak * (double)h.t);
                }
            }
            double dist = h.t;
            DVec3 rdCur = rd;
            double cosThetaIn = fabs(ddot(h.n, rdCur * (Real)-1));
            if (cosThetaIn <= 1e-9) break;

            const DMaterial* mp = &sc.mats[h.matId];
            int matId = h.matId;
            if (mp->type == D_MIX) {
                int c = dMixResolveChild(sc, *mp, h, rng.uniform());
                if (c < 0) break;
                mp = &sc.mats[c]; matId = c;
            }
            if (dEmitterForMat(sc, matId) >= 0) break;    // light subpath doesn't scatter off emitters

            // misArrival(dist, cosThetaIn, foldDist2 = !(isInfinite && edges == 1)). The one
            // edge in the whole renderer that skips the dist^2 area conversion is the FIRST
            // edge of a sun's subpath: that vertex's forward density is the planar 1/(pi R^2)
            // over the fictitious disc, not a solid-angle density (SmallVCM's
            // `if (pathLength > 1 || isFiniteLight) dVCM *= dist^2`; pbrt's "correct subpath
            // sampling densities for infinite area lights" patch).
            if (!(isInfinite && edges == 1)) dVCM *= dist * dist;
            dVCM /= cosThetaIn; dVC /= cosThetaIn; dVM /= cosThetaIn;

            DVec3 wo = normalize(prevP - h.p);
            DVec3 ngo = (ddot(h.ng, h.n) >= 0.0) ? h.ng : h.ng * (Real)-1;

            if (dConnectibleType(mp->type)) {
                if (stored < vcmCap) {
                    DVcmLV lv;
                    lv.p = h.p; lv.ns = h.n; lv.ng = h.ng; lv.wo = wo;
                    lv.beta = beta; lv.lambda = (float)lambda;
                    lv.cx = cieLx; lv.cy = cieLy; lv.cz = cieLz;
                    lv.dVCM = dVCM; lv.dVC = dVC; lv.dVM = dVM;
                    lv.matId = matId; lv.edges = edges; lv.u = h.u; lv.v = h.v; lv.curv = h.curv;
                    lv.nUp = nUp;
                    lvSlab[i * vcmCap + stored] = lv;
                    if (NS > 0 && lvSec) {
                        DVcmSec* row = lvSec + (size_t)(i * vcmCap + stored) * secStride;
                        for (int k = 0; k + 1 < nUp; ++k) { row[k].beta = betaSec[k]; row[k].lam = (float)lamAll[k + 1]; }
                    }
                    stored++;
                }
                // Connect this vertex to the pinhole camera (t=1 light-image splat).
                if (!cam.hasLens && edges + 1 <= ctx.maxDepth) {
                    DVec3 toCam = cam.eye - h.p;
                    double dist2c = ddot(toCam, toCam);
                    if (dist2c > 1e-12) {
                        double distc = sqrt(dist2c);
                        DVec3 wcam = toCam * (Real)(1.0 / distc);
                        double cosToCamera = ddot(h.n, wcam);
                        bool twoSided = dTwoSidedType(mp->type);
                        double stG = twoSided ? 1.0 : (double)dShadowTerminatorG(wcam, h.n, ngo);
                        bool sideOk = twoSided ? (cosToCamera != 0.0) : (cosToCamera > 0.0 && stG > 0.0);
                        double cosAtCamera = ddot(cam.w, wcam * (Real)-1);
                        if (sideOk && cosAtCamera > 1e-9) {
                            int px, py; Real cc, d2c;
                            if (cam.project(h.p, px, py, cc, d2c)) {
                                // Shadow ray first, BSDF eval after (bit-identical: no RNG
                                // in either; skips the eval for occluded splats).
                                double sgn = ddot(h.ng, wcam) >= 0.0 ? 1.0 : -1.0;
                                DVec3 oo = h.p + h.ng * (Real)(sgn * 1e-6);
                                if (!occluded(sc, oo, wcam, connMaxT(distc))) {
                                    DVertex vt = dVertFromHit(h, matId);
                                    // The adjoint correction and shadow-terminator G are purely
                                    // geometric, so they scale every λ the same way.
                                    double geo = (double)dShadingAdjointCorr(wo, wcam, h.n, ngo) * stG;
                                    double f = dBsdfF(sc, vt, wo, wcam, lambda) * geo;
                                    double fSec[SECN], mxf = f;
                                    for (int k = 0; k + 1 < nUp; ++k) {
                                        fSec[k] = dBsdfF(sc, vt, wo, wcam, lamAll[k + 1]) * geo;
                                        if (fSec[k] > mxf) mxf = fSec[k];
                                    }
                                    if (mxf > 0.0) {
                                        double bsdfRevPdfW = dBsdfPdf(sc, vt, wcam, wo, lambda);
                                        double imgPtDist = ctx.imagePlaneDist / cosAtCamera;
                                        double imgToSolid = imgPtDist * imgPtDist / cosAtCamera;
                                        double imgToSurf = imgToSolid * fabs(cosToCamera) / dist2c;
                                        double wLight = (imgToSurf / ctx.nLightPaths) *
                                                        (ctx.misVmWeight + dVCM + dVC * bsdfRevPdfW);
                                        double misW = 1.0 / (wLight + 1.0);
                                        double ax = 0, ay = 0, az = 0;
                                        double contrib = misW * beta * f * imgToSurf / ctx.nLightPaths;
                                        if (contrib > 0.0) {
                                            ax = cieLx * contrib; ay = cieLy * contrib; az = cieLz * contrib;
                                        }
                                        for (int k = 0; k + 1 < nUp; ++k) {
                                            double cs = misW * betaSec[k] * fSec[k] * imgToSurf / ctx.nLightPaths;
                                            if (cs > 0.0) { ax += cieSx[k] * cs; ay += cieSy[k] * cs; az += cieSz[k] * cs; }
                                        }
                                        // The C wavelengths are C samples of ONE spectral
                                        // estimate, so the bundle averages (see hero.h).
                                        if (nUp > 1) { double inv = 1.0 / nUp; ax *= inv; ay *= inv; az *= inv; }
                                        if (ax != 0.0 || ay != 0.0 || az != 0.0) {
                                            size_t o2 = ((size_t)py * resX + px) * 3;
                                            atomicAdd(&splat[o2 + 0], ax);
                                            atomicAdd(&splat[o2 + 1], ay);
                                            atomicAdd(&splat[o2 + 2], az);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (edges == ctx.maxDepth) break;

            DVec3 wi; double betaFactor, pdfW, pdfRevW, cosThetaOut; bool delta, terminate;
            double secF[SECN]; bool secChromatic = false, keepBundle = false;
            dVcmScatter(sc, *mp, h, rdCur, lambda, rng, matId, stk, diffraction,
                        wi, betaFactor, pdfW, pdfRevW, cosThetaOut, delta, terminate,
                        lamAll, nUp, secF, &secChromatic, &keepBundle);
            // Kill the walk only when EVERY live λ is dead (nUp == 1 -> mxF == betaFactor).
            double mxF = betaFactor;
            if (secChromatic) for (int k = 0; k + 1 < nUp; ++k) if (secF[k] > mxF) mxF = secF[k];
            if (terminate || mxF <= 0.0) break;
            if (!delta && (pdfW <= 0.0 || cosThetaOut <= 0.0)) break;

            // misScatter(delta, cosThetaOut, pdfW, pdfRevW)
            if (delta) { dVCM = 0.0; dVC *= cosThetaOut; dVM *= cosThetaOut; }
            else {
                double t = cosThetaOut / pdfW;
                dVC = t * (dVC * pdfRevW + dVCM + ctx.misVmWeight);
                dVM = t * (dVM * pdfRevW + dVCM * ctx.misVcWeight + 1.0);
                dVCM = 1.0 / pdfW;
            }
            beta *= betaFactor;
            for (int k = 0; k + 1 < nUp; ++k) betaSec[k] *= secChromatic ? secF[k] : betaFactor;
            if (!delta) {
                double adj = (double)dShadingAdjointCorr(wo, normalize(wi), h.n, ngo);
                beta *= adj;
                for (int k = 0; k + 1 < nUp; ++k) betaSec[k] *= adj;
            }
            // De-hero at a λ-DEPENDENT direction change (dielectric / thin-film / multilayer /
            // grating / half-mirror). Mirror and Filter set keepBundle: their outgoing direction
            // does not consult λ at all, so the secondaries ride on.
            if (delta && !keepBundle) nUp = 1;
            prevP = h.p;
            double sgn2 = ddot(wi, h.ng) >= 0.0 ? 1.0 : -1.0;
            ro = h.p + h.ng * (Real)(sgn2 * 1e-6);
            rd = normalize(wi);
        }
        lvCount[i] = stored;
    }
}

// Phase 3: one camera subpath per pixel. Does emission (s=0), NEE (s=1), vertex connection to
// the PAIRED light subpath [pathBegin[i],pathEnd[i]) and merging over the grid, then adds this
// pass's per-pixel radiance (camera result + the light splat) into the persistent `accum` sum.
// Device twin of vcm.h traceCameraSubpath. `grid.lv` is the compact light-vertex array.
template <int NS>
__global__ void kVcmCameraT(DScene sc, DCamera cam, int diffraction, DVcmCtx ctx, DVcmGrid grid,
                            const DVcmSec* lvSec, int secStride, int heroC,
                            const int* pathBegin, const int* pathEnd, const double* splat,
                            double* accum, const Real* lamBuf, const double* invLamBuf,
                            int resX, int resY, unsigned long long seedBase, long long passIdx) {
    constexpr int SECN = (NS > 0) ? NS : 1;
    const int C = (NS > 0) ? heroC : 1;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    long long npix = (long long)resX * resY;
    for (long long i = g; i < npix; i += G) {
        double sxl = splat[i * 3 + 0], syl = splat[i * 3 + 1], szl = splat[i * 3 + 2];
        double invPdfLambda = invLamBuf[i * C];
        if (invPdfLambda <= 0.0) {                         // no valid wavelength: only the splat
            accum[i * 3 + 0] += sxl; accum[i * 3 + 1] += syl; accum[i * 3 + 2] += szl;
            continue;
        }
        // THE bundle of path index i — the SAME C wavelengths the light kernel walked for this
        // index. That pairing is what makes strategy (c) (connection to the paired light
        // subpath) exact per-λ rather than an approximation: both ends share the λ set.
        Real lamAll[SECN + 1];
        double invAll[SECN + 1];
        for (int k = 0; k < C; ++k) { lamAll[k] = lamBuf[i * C + k]; invAll[k] = invLamBuf[i * C + k]; }
        Real lambda = lamAll[0];
        int nUp = C;
        int px = (int)(i % resX), py = (int)(i / resX);
        unsigned long long s = (unsigned long long)i * 0xC2B2AE3D27D4EB4FULL
                             + (unsigned long long)passIdx * 0xA24BAED4963EE407ULL;
        DRng rng; rng.seed(s * 2 + 77, seedBase ^ s);

        double rx = 0, ry = 0, rz = 0;
        double cieCx = (double)cieX(lambda), cieCy = (double)cieY(lambda), cieCz = (double)cieZ(lambda);
        double betaSec[SECN];
        double cieSx[SECN], cieSy[SECN], cieSz[SECN];
        for (int k = 0; k + 1 < C; ++k) {
            betaSec[k] = 1.0;                              // camera vertex beta == 1 for every λ
            cieSx[k] = (double)cieX(lamAll[k + 1]);
            cieSy[k] = (double)cieY(lamAll[k + 1]);
            cieSz[k] = (double)cieZ(lamAll[k + 1]);
        }

        DVec3 ro, rd;
        Real jx = rng.uniform(), jy = rng.uniform();
        dGenRay(cam, px, py, jx, jy, ro, rd);
        double cosAtCamera = ddot(rd, cam.w);
        if (cosAtCamera <= 1e-9) {
            accum[i * 3 + 0] += sxl; accum[i * 3 + 1] += syl; accum[i * 3 + 2] += szl;
            continue;
        }
        double cameraPdfW = ctx.imagePlaneDist * ctx.imagePlaneDist /
                            (cosAtCamera * cosAtCamera * cosAtCamera);
        double beta = 1.0;
        double dVCM = ctx.nLightPaths / cameraPdfW;
        double dVC = 0.0, dVM = 0.0;
        DMediumStack stk; stk.clear();
        DVec3 prevP = cam.eye;
        // Has this eye chain passed through ONLY delta (specular) vertices so far? That is
        // the precondition for the escaped-sun strategy below; tracked explicitly rather than
        // inferred from dVCM == 0, which a delta vertex sets but which says nothing about the
        // vertices before it.
        bool camAllDelta = true;
        const bool hasSun = sc.sunCount > 0;

        for (int edges = 1; edges <= ctx.maxDepth; ++edges) {
            DHit h = closestHit(sc, ro, rd);
            if (!h.valid) {
                // The ray left the scene. No env map in VCM scope, but a `light sun` is a
                // delta-DIRECTION emitter with no geometry: the s=0 term never fires for it
                // and no connection or merge can reach it either, so without this the solar
                // disc — and every mirror glint of it — would simply be missing. Add it only
                // when the escaping ray came through camera + delta vertices ONLY, which is
                // exactly the case where nothing else competes (NEE, every s>=2 connection,
                // the t==1 splat and merging all need a connectible vertex). The MIS weight
                // is therefore exactly 1.
                if (hasSun && camAllDelta) {
                    for (int k = 0; k < sc.nEmitters; ++k) {
                        const DEmitter& em = sc.emitters[k];
                        if (em.shape != 6 || !dInSunCone(em, rd)) continue;
                        double e0 = beta * (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
                        double ax = cieCx * e0, ay = cieCy * e0, az = cieCz * e0;
                        for (int j = 0; j + 1 < nUp; ++j) {
                            double ej = betaSec[j] *
                                        (double)specLookup(em.emitSpd, lamAll[j + 1]) * invAll[j + 1];
                            ax += cieSx[j] * ej; ay += cieSy[j] * ej; az += cieSz[j] * ej;
                        }
                        if (nUp > 1) { double inv = 1.0 / nUp; ax *= inv; ay *= inv; az *= inv; }
                        rx += ax; ry += ay; rz += az;
                    }
                }
                break;
            }
            {
                int cm = stk.topMat();
                double a = (cm >= 0) ? (double)specLookup(sc.mats[cm].absorb, lambda) : 0.0;
                if (a > 0.0) beta *= exp(-a * (double)h.t);
                for (int k = 0; k + 1 < nUp; ++k) {       // per-λ: the colour of coloured glass
                    double ak = (cm >= 0) ? (double)specLookup(sc.mats[cm].absorb, lamAll[k + 1]) : 0.0;
                    if (ak > 0.0) betaSec[k] *= exp(-ak * (double)h.t);
                }
            }
            double dist = h.t;
            DVec3 rdCur = rd;
            double cosThetaIn = fabs(ddot(h.n, rdCur * (Real)-1));
            if (cosThetaIn <= 1e-9) break;

            const DMaterial* mp = &sc.mats[h.matId];
            int matId = h.matId;
            if (mp->type == D_MIX) {
                int c = dMixResolveChild(sc, *mp, h, rng.uniform());
                if (c < 0) break;
                mp = &sc.mats[c]; matId = c;
            }

            // misArrival(dist, cosThetaIn)
            dVCM *= dist * dist; dVCM /= cosThetaIn; dVC /= cosThetaIn; dVM /= cosThetaIn;

            DVec3 wo = normalize(prevP - h.p);

            // (a) Emission (s=0).
            int li = dEmitterForMat(sc, matId);
            if (li >= 0) {
                double cosLight = ddot(h.ng, wo);
                if (cosLight > 0.0) {
                    // Achromatic `emit pattern:` factor at this hit — the same value the
                    // light-subpath / NEE sides get from the sampler at this point.
                    double ep = dEmitPatMul(sc, mp->emitPat, h);
                    double Le = (double)specLookup(sc.emitters[li].emitSpd, lambda) * invPdfLambda * ep;
                    double LeSec[SECN], mxLe = Le;
                    for (int k = 0; k + 1 < nUp; ++k) {
                        LeSec[k] = (double)specLookup(sc.emitters[li].emitSpd, lamAll[k + 1]) * invAll[k + 1] * ep;
                        if (LeSec[k] > mxLe) mxLe = LeSec[k];
                    }
                    if (mxLe > 0.0) {
                        double misW = 1.0;
                        const DEmitter& em = sc.emitters[li];
                        if (em.area > 0.0 && sc.totalPower > 0.0 && edges >= 2) {
                            double pdfChoice = em.power / sc.totalPower;
                            double directPdfA = pdfChoice / em.area;
                            double emissionPdfW = pdfChoice * cosLight / DPI;
                            double wCamera = directPdfA * dVCM + emissionPdfW * dVC;
                            misW = 1.0 / (1.0 + wCamera);
                        }
                        double e = beta * Le * misW;
                        double ax = cieCx * e, ay = cieCy * e, az = cieCz * e;
                        for (int k = 0; k + 1 < nUp; ++k) {
                            double ek = betaSec[k] * LeSec[k] * misW;
                            ax += cieSx[k] * ek; ay += cieSy[k] * ek; az += cieSz[k] * ek;
                        }
                        if (nUp > 1) { double inv = 1.0 / nUp; ax *= inv; ay *= inv; az *= inv; }
                        rx += ax; ry += ay; rz += az;
                    }
                }
                break;                                    // can't scatter off a light
            }

            if (dConnectibleType(mp->type)) {
                DVertex vt = dVertFromHit(h, matId);
                DVec3 ngoCam = (ddot(h.ng, h.n) >= 0.0) ? h.ng : h.ng * (Real)-1;
                bool twoSidedCam = dTwoSidedType(mp->type);

                // (b) NEE (s=1) — connect to a freshly sampled light point.
                if (edges + 1 <= ctx.maxDepth && sc.nEmitters > 0 && sc.totalPower > 0.0) {
                    int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
                    const DEmitter& em = sc.emitters[ei];
                    if (!(em.shape == 3 || em.collimated)) {
                        const bool neeDelta = dIsDeltaEmitter(em);
                        Real u1 = rng.uniform(), u2 = rng.uniform();
                        // Per-shape connection geometry (device twin of vcm.h's NEE branch).
                        // All three emission models produce the same five quantities, after
                        // which the BSDF / occlusion / MIS code below is shared:
                        //   wiL          connection direction (this vertex -> the light)
                        //   distL        shadow-ray length (out of the scene for a sun)
                        //   cosAtLight   |cos| at the light (1 for both delta lights)
                        //   directPdfW   the NEE density in SOLID ANGLE at this vertex
                        //   emissionPdfW pdfPos*pdfDirW of the same sample (no pickProb: it
                        //                cancels in the wCamera ratio and is applied
                        //                explicitly in wLight / the contribution below)
                        // `epat` scales the emitted spectrum only, never a pdf.
                        DVec3 wiL; double distL = 0.0, dist2 = 0.0, cosAtLight = 1.0;
                        double neeDirectPdfW = 0.0, neeEmissionPdfW = 0.0, epat = 1.0;
                        double occlEps = 2e-6;
                        bool geomOk = false;
                        if (em.shape == 2) {
                            // Delta POSITION: the connection point is deterministic, so the
                            // positional pdf is 1 and the solid-angle density is just dist^2.
                            // The penumbra weights the INTENSITY toward this receiver (it is a
                            // radiance factor, never a density — folding it into a pdf would
                            // break the light subpath's matching dVCM).
                            DVec3 toL = em.origin - h.p; dist2 = ddot(toL, toL);
                            if (dist2 > 1e-12) {
                                distL = sqrt(dist2); wiL = toL * (Real)(1.0 / distL);
                                epat = spotFalloff(ddot(wiL * (Real)-1, em.beamDir),
                                                   em.spotCosInner, em.spotCosOuter);
                                double coneSolid = 2.0 * DPI * (1.0 - em.spotCosOuter);
                                if (epat > 0.0 && coneSolid > 0.0) {
                                    neeDirectPdfW = dist2;        // pdfA(=1)*dist^2/cosAtLight(=1)
                                    neeEmissionPdfW = 1.0 / coneSolid;
                                    geomOk = true;                // emitSpd is an INTENSITY (W/sr)
                                }
                            }
                        } else if (em.shape == 6) {
                            // Delta DIRECTION at infinity: sample inside the solar cone
                            // (pdfW = 1/Omega) and shadow-ray all the way out of the scene, so
                            // the ray must NOT be shortened by the endpoint epsilon. No 1/dist^2
                            // and no cosLight; emitSpd is already a radiance.
                            double coneSolid = 2.0 * DPI * (1.0 - em.spotCosOuter);
                            double R = sc.sceneRadius;
                            if (coneSolid > 0.0 && R > 0.0) {
                                wiL = dSunSampleCone(em, em.beamDir * (Real)-1, (double)u1, (double)u2);
                                distL = (double)length(sc.sceneCenter - h.p) + R;
                                dist2 = distL * distL;
                                occlEps = 0.0;
                                neeDirectPdfW = 1.0 / coneSolid;
                                neeEmissionPdfW = 1.0 / (DPI * R * R * coneSolid);
                                geomOk = true;
                            }
                        } else {
                            DVec3 yL, nL;
                            // Sampled point's emission-pattern factor (1.0 when unpatterned).
                            epat = dEmitterSamplePointPat(sc, em, u1, u2, yL, nL);
                            DVec3 toL = yL - h.p; dist2 = ddot(toL, toL);
                            if (dist2 > 1e-12 && em.area > 0.0) {
                                distL = sqrt(dist2); wiL = toL * (Real)(1.0 / distL);
                                cosAtLight = ddot(nL, wiL * (Real)-1);
                                if (cosAtLight > 0.0) {           // emitter stays one-sided
                                    double invArea = 1.0 / em.area;
                                    neeDirectPdfW = invArea * dist2 / cosAtLight;
                                    neeEmissionPdfW = invArea * cosAtLight / DPI;
                                    geomOk = true;
                                }
                            }
                        }
                        if (geomOk) {
                            double cosToLight = ddot(h.n, wiL);
                            double stG = twoSidedCam ? 1.0 : (double)dShadowTerminatorG(wiL, h.n, ngoCam);
                            bool sideOk = twoSidedCam ? (cosToLight != 0.0) : (cosToLight > 0.0 && stG > 0.0);
                            if (sideOk) {
                                // Cheap gates + shadow ray first, BSDF eval after (bit-
                                // identical: no RNG in either; skips the eval when shadowed).
                                double Le = (double)specLookup(em.emitSpd, lambda) * invPdfLambda * epat;
                                double LeSec[SECN], mxLe = Le;
                                for (int k = 0; k + 1 < nUp; ++k) {
                                    LeSec[k] = (double)specLookup(em.emitSpd, lamAll[k + 1]) * invAll[k + 1] * epat;
                                    if (LeSec[k] > mxLe) mxLe = LeSec[k];
                                }
                                double sgn = ddot(h.ng, wiL) >= 0.0 ? 1.0 : -1.0;
                                DVec3 oo = h.p + h.ng * (Real)(sgn * 1e-6);
                                double f = 0.0, fSec[SECN];
                                for (int k = 0; k + 1 < nUp; ++k) fSec[k] = 0.0;
                                if (mxLe > 0.0 &&
                                    !occluded(sc, oo, wiL, connMaxT(distL, occlEps))) {
                                    f = dBsdfF(sc, vt, wo, wiL, lambda) * stG;
                                    for (int k = 0; k + 1 < nUp; ++k)
                                        fSec[k] = dBsdfF(sc, vt, wo, wiL, lamAll[k + 1]) * stG;
                                }
                                // Fuse BSDF x Le per-λ before the max test: either factor may
                                // vanish at the hero while the product is alive at a secondary.
                                // At nUp == 1 this is exactly the old `f > 0` gate.
                                double mxfLe = f * Le;
                                for (int k = 0; k + 1 < nUp; ++k) {
                                    double p = fSec[k] * LeSec[k];
                                    if (p > mxfLe) mxfLe = p;
                                }
                                if (mxfLe > 0.0) {
                                    double pdfChoice = em.power / sc.totalPower;
                                    double directPdfW = neeDirectPdfW;
                                    double emissionPdfW = neeEmissionPdfW;
                                    double bsdfDirPdfW = dBsdfPdf(sc, vt, wo, wiL, lambda);
                                    double bsdfRevPdfW = dBsdfPdf(sc, vt, wiL, wo, lambda);
                                    // A delta light can never be found by BSDF sampling, so
                                    // that strategy is DROPPED from the balance heuristic
                                    // (weight 0) rather than given a density it does not have.
                                    double wLight = neeDelta ? 0.0
                                                             : (bsdfDirPdfW / (pdfChoice * directPdfW));
                                    double wCamera = (emissionPdfW * fabs(cosToLight) /
                                                      (directPdfW * cosAtLight)) *
                                                     (ctx.misVmWeight + dVCM + dVC * bsdfRevPdfW);
                                    double misW = 1.0 / (wLight + 1.0 + wCamera);
                                    double contrib = misW * fabs(cosToLight) /
                                                     (pdfChoice * directPdfW) * Le * f;
                                    double ax = 0, ay = 0, az = 0;
                                    if (contrib > 0.0) {
                                        double e = beta * contrib;
                                        ax = cieCx * e; ay = cieCy * e; az = cieCz * e;
                                    }
                                    for (int k = 0; k + 1 < nUp; ++k) {
                                        double ck = misW * fabs(cosToLight) /
                                                    (pdfChoice * directPdfW) * LeSec[k] * fSec[k];
                                        if (ck > 0.0) {
                                            double ek = betaSec[k] * ck;
                                            ax += cieSx[k] * ek; ay += cieSy[k] * ek; az += cieSz[k] * ek;
                                        }
                                    }
                                    if (nUp > 1) { double inv = 1.0 / nUp; ax *= inv; ay *= inv; az *= inv; }
                                    rx += ax; ry += ay; rz += az;
                                }
                            }
                        }
                    }
                }

                // (c) Vertex connection to the PAIRED light subpath's stored vertices.
                int pb = pathBegin[i], pe = pathEnd[i];
                for (int j = pb; j < pe; ++j) {
                    const DVcmLV& lv = grid.lv[j];
                    if (edges + lv.edges + 1 > ctx.maxDepth) continue;
                    DVec3 dv = lv.p - h.p; double dist2 = ddot(dv, dv);
                    if (dist2 <= 1e-12) continue;
                    double distc = sqrt(dist2); DVec3 w = dv * (Real)(1.0 / distc);
                    double cosCam = ddot(h.n, w);
                    double cosLit = ddot(lv.ns, w * (Real)-1);
                    DVec3 ngoLit = (ddot(lv.ng, lv.ns) >= 0.0) ? lv.ng : lv.ng * (Real)-1;
                    bool twoSidedLit = dTwoSidedType(sc.mats[lv.matId].type);
                    double stGCam = twoSidedCam ? 1.0 : (double)dShadowTerminatorG(w, h.n, ngoCam);
                    double stGLit = twoSidedLit ? 1.0 : (double)dShadowTerminatorG(w * (Real)-1, lv.ns, ngoLit);
                    bool camSide = twoSidedCam ? (cosCam != 0.0) : (cosCam > 0.0 && stGCam > 0.0);
                    bool litSide = twoSidedLit ? (cosLit != 0.0) : (cosLit > 0.0 && stGLit > 0.0);
                    if (!camSide || !litSide) continue;
                    // Occlusion FIRST: the shadow ray kills most connections in a
                    // typical scene, and it is far cheaper than the 2x dBsdfF +
                    // 4x dBsdfPdf + MIS block below. No RNG is consumed in this
                    // loop and occluded() has no side effects, so hoisting the
                    // test is bit-identical — it only skips work for connections
                    // that contributed nothing anyway.
                    double sgn = ddot(h.ng, w) >= 0.0 ? 1.0 : -1.0;
                    DVec3 oo = h.p + h.ng * (Real)(sgn * 1e-6);
                    if (occluded(sc, oo, w, connMaxT(distc))) continue;
                    DVertex lvt = dVertFromLV(lv);
                    double adjLit = (double)dShadingAdjointCorr(lv.wo, w * (Real)-1, lv.ns, ngoLit) * stGLit;
                    double fCam = dBsdfF(sc, vt, wo, w, lambda) * stGCam;
                    double fLit = dBsdfF(sc, lvt, lv.wo, w * (Real)-1, lambda);
                    fLit *= adjLit;
                    // The camera path and the stored light path share this pass's bundle (same
                    // path index i), so a connection is EXACT per-λ over the wavelengths still
                    // live at BOTH ends. `lv.nUp` is 1 whenever the light walk de-hero'd.
                    const int lvUp = (NS > 0) ? lv.nUp : 1;
                    const int nUpConn = (nUp < lvUp) ? nUp : lvUp;
                    double fProdSec[SECN], mxProd = fCam * fLit;
                    const DVcmSec* lsRow = ((NS > 0) && lvSec) ? (lvSec + (size_t)j * secStride) : nullptr;
                    for (int k = 0; k + 1 < nUpConn; ++k) {
                        double fc = dBsdfF(sc, vt, wo, w, lamAll[k + 1]) * stGCam;
                        double fl = dBsdfF(sc, lvt, lv.wo, w * (Real)-1, lamAll[k + 1]) * adjLit;
                        fProdSec[k] = fc * fl;
                        if (fProdSec[k] > mxProd) mxProd = fProdSec[k];
                    }
                    // At nUpConn == 1 this is exactly the old `fCam <= 0 || fLit <= 0` bail
                    // (both factors are non-negative, so the product is 0 iff either is).
                    if (mxProd <= 0.0) continue;
                    double camDirPdfW = dBsdfPdf(sc, vt, wo, w, lambda);
                    double camRevPdfW = dBsdfPdf(sc, vt, w, wo, lambda);
                    double litDirPdfW = dBsdfPdf(sc, lvt, lv.wo, w * (Real)-1, lambda);
                    double litRevPdfW = dBsdfPdf(sc, lvt, w * (Real)-1, lv.wo, lambda);
                    double camDirPdfA = camDirPdfW * fabs(cosLit) / dist2;
                    double litDirPdfA = litDirPdfW * fabs(cosCam) / dist2;
                    double wLight = camDirPdfA * (ctx.misVmWeight + lv.dVCM + lv.dVC * litRevPdfW);
                    double wCamera = litDirPdfA * (ctx.misVmWeight + dVCM + dVC * camRevPdfW);
                    double misW = 1.0 / (wLight + 1.0 + wCamera);
                    double Gt = fabs(cosCam) * fabs(cosLit) / dist2;
                    double contrib = misW * Gt * fCam * fLit * beta * lv.beta;
                    double ax = 0, ay = 0, az = 0;
                    if (contrib > 0.0) { ax = cieCx * contrib; ay = cieCy * contrib; az = cieCz * contrib; }
                    for (int k = 0; k + 1 < nUpConn; ++k) {
                        double ck = misW * Gt * fProdSec[k] * betaSec[k] * lsRow[k].beta;
                        if (ck > 0.0) { ax += cieSx[k] * ck; ay += cieSy[k] * ck; az += cieSz[k] * ck; }
                    }
                    if (nUpConn > 1) { double inv = 1.0 / nUpConn; ax *= inv; ay *= inv; az *= inv; }
                    rx += ax; ry += ay; rz += az;
                }

                // (d) Vertex merging — gather nearby light vertices from ALL paths (XYZ estimate).
                if (ctx.vmNorm > 0.0 && grid.nLV > 0) {
                    double mx = 0, my = 0, mz = 0;
                    double r2 = ctx.radius * ctx.radius;
                    int ix = (int)floor(((double)h.p.x - grid.lo.x) / grid.cell);
                    int iy = (int)floor(((double)h.p.y - grid.lo.y) / grid.cell);
                    int iz = (int)floor(((double)h.p.z - grid.lo.z) / grid.cell);
                    ix = min(max(ix, 0), grid.nx - 1);
                    iy = min(max(iy, 0), grid.ny - 1);
                    iz = min(max(iz, 0), grid.nz - 1);
                    for (int dz = -1; dz <= 1; ++dz) { int cz = iz + dz; if (cz < 0 || cz >= grid.nz) continue;
                      for (int dy = -1; dy <= 1; ++dy) { int cy = iy + dy; if (cy < 0 || cy >= grid.ny) continue;
                        for (int dx = -1; dx <= 1; ++dx) { int cx = ix + dx; if (cx < 0 || cx >= grid.nx) continue;
                          int c = (cz * grid.ny + cy) * grid.nx + cx;
                          for (int k = grid.cellStart[c]; k < grid.cellStart[c + 1]; ++k) {
                              int idx = grid.order[k];
                              const DVcmLV& lv = grid.lv[idx];
                              DVec3 d = h.p - lv.p;
                              if (ddot(d, d) > r2) continue;
                              if (edges + lv.edges > ctx.maxDepth) continue;
                              DVec3 wMerge = lv.wo;
                              Real lam = (Real)lv.lambda;
                              // A MERGE crosses paths, so the two ends carry DIFFERENT bundles
                              // and there is no shared λ set. The estimate therefore stays keyed
                              // on the LIGHT vertex's own wavelengths (the pre-existing spectral
                              // photon-mapping approximation, generalised from 1 λ to lv.nUp):
                              // sum over its live λ, divide by lv.nUp, and let the camera's HERO
                              // throughput/MIS weight scale the whole gather as before.
                              const int lvUp = (NS > 0) ? lv.nUp : 1;
                              const DVcmSec* row = ((NS > 0) && lvSec)
                                                 ? (lvSec + (size_t)idx * secStride) : nullptr;
                              double fCam = dBsdfF(sc, vt, wo, wMerge, lam);
                              double fSec[SECN], mxF = fCam;
                              for (int q = 0; q + 1 < lvUp; ++q) {
                                  fSec[q] = dBsdfF(sc, vt, wo, wMerge, (Real)row[q].lam);
                                  if (fSec[q] > mxF) mxF = fSec[q];
                              }
                              if (mxF <= 0.0) continue;   // == the old `fCam <= 0` at lvUp == 1
                              double denom = fabs(ddot(wMerge, ngoCam));
                              double gcorr = (denom <= 1e-8) ? 1.0 : fabs(ddot(wMerge, h.n)) / denom;
                              fCam *= gcorr;
                              double camDirPdfW = dBsdfPdf(sc, vt, wo, wMerge, lam);
                              double camRevPdfW = dBsdfPdf(sc, vt, wMerge, wo, lam);
                              double wLight = lv.dVCM * ctx.misVcWeight + lv.dVM * camDirPdfW;
                              double wCamera = dVCM * ctx.misVcWeight + dVM * camRevPdfW;
                              double misW = 1.0 / (wLight + 1.0 + wCamera);
                              double wgt = misW * fCam * lv.beta;
                              double ax = lv.cx * wgt, ay = lv.cy * wgt, az = lv.cz * wgt;
                              for (int q = 0; q + 1 < lvUp; ++q) {
                                  double wq = misW * fSec[q] * gcorr * row[q].beta;
                                  Real lq = (Real)row[q].lam;
                                  ax += (double)cieX(lq) * wq;
                                  ay += (double)cieY(lq) * wq;
                                  az += (double)cieZ(lq) * wq;
                              }
                              if (lvUp > 1) { double inv = 1.0 / lvUp; ax *= inv; ay *= inv; az *= inv; }
                              mx += ax; my += ay; mz += az;
                          }
                    }}}
                    double bn = beta * ctx.vmNorm;
                    rx += mx * bn; ry += my * bn; rz += mz * bn;
                }
            }

            if (edges == ctx.maxDepth) break;

            DVec3 wi; double betaFactor, pdfW, pdfRevW, cosThetaOut; bool delta, terminate;
            double secF[SECN]; bool secChromatic = false, keepBundle = false;
            dVcmScatter(sc, *mp, h, rdCur, lambda, rng, matId, stk, diffraction,
                        wi, betaFactor, pdfW, pdfRevW, cosThetaOut, delta, terminate,
                        lamAll, nUp, secF, &secChromatic, &keepBundle);
            double mxF = betaFactor;
            if (secChromatic) for (int k = 0; k + 1 < nUp; ++k) if (secF[k] > mxF) mxF = secF[k];
            if (terminate || mxF <= 0.0) break;
            if (!delta && (pdfW <= 0.0 || cosThetaOut <= 0.0)) break;

            if (delta) { dVCM = 0.0; dVC *= cosThetaOut; dVM *= cosThetaOut; }
            else {
                double t = cosThetaOut / pdfW;
                dVC = t * (dVC * pdfRevW + dVCM + ctx.misVmWeight);
                dVM = t * (dVM * pdfRevW + dVCM * ctx.misVcWeight + 1.0);
                dVCM = 1.0 / pdfW;
            }
            beta *= betaFactor;                           // camera side: no adjoint on continuation
            for (int k = 0; k + 1 < nUp; ++k) betaSec[k] *= secChromatic ? secF[k] : betaFactor;
            if (!delta) camAllDelta = false;               // disqualifies the escaped-sun strategy
            if (delta && !keepBundle) nUp = 1;             // λ-dependent direction change
            prevP = h.p;
            double sgn2 = ddot(wi, h.ng) >= 0.0 ? 1.0 : -1.0;
            ro = h.p + h.ng * (Real)(sgn2 * 1e-6);
            rd = normalize(wi);
        }

        accum[i * 3 + 0] += rx + sxl;
        accum[i * 3 + 1] += ry + syl;
        accum[i * 3 + 2] += rz + szl;
    }
}

// ================= device: on-device grid builds (VCM / SPPM sessions) =================
// Kernels + thrust functors that move the per-pass VCM/SPPM photon-grid construction onto
// the device. Each piece is an EXACT-arithmetic mirror of the host build it replaces
// (vcm.h VcmGrid::build twin in vcmSessionPass / photonmap.h PhotonMap::build), so the
// consuming kernels see bit-identical inputs in the identical order: min/max reductions
// are order-independent and exact; the cell-key kernels repeat the host's float/double
// mixed expressions type-for-type; and a STABLE sort by cell id equals a counting sort's
// output order (both keep original index order within a cell).

// Double-precision CIE twins (exact color.h mirror; the float `cieX` above serves the Real
// transport path). Used by kSppmGatherConvert: the host build computed cie{X,Y,Z}(lambda)
// in double and rounded cie*power/pi to float ONCE. CUDA's double exp() is within 1 ulp of
// the host's, and that last-ulp double wobble is absorbed by the final float rounding
// (~2^-28 odds per value of landing on a rounding boundary), so the stored floats match.
__device__ static double gaussPiece64(double x, double mu, double s1, double s2) {
    double t = (x - mu) * ((x < mu) ? s1 : s2);
    return exp(-0.5 * t * t);
}
__device__ static double cieX64(double w) {
    return 0.362 * gaussPiece64(w, 442.0, 0.0624, 0.0374)
         + 1.056 * gaussPiece64(w, 599.8, 0.0264, 0.0323)
         - 0.065 * gaussPiece64(w, 501.1, 0.0490, 0.0382);
}
__device__ static double cieY64(double w) {
    return 0.821 * gaussPiece64(w, 568.8, 0.0213, 0.0247)
         + 0.286 * gaussPiece64(w, 530.9, 0.0613, 0.0322);
}
__device__ static double cieZ64(double w) {
    return 1.217 * gaussPiece64(w, 437.0, 0.0845, 0.0278)
         + 0.681 * gaussPiece64(w, 459.0, 0.0385, 0.0725);
}

// Float AABB carried through thrust::transform_reduce. min/max are exact and commute with
// the host's float->double promotion, so the reduced bounds equal the host loop's bounds.
struct BboxF { float mnx, mny, mnz, mxx, mxy, mxz; };
struct LvToBboxF {
    HD BboxF operator()(const DVcmLV& v) const {
        return BboxF{(float)v.p.x, (float)v.p.y, (float)v.p.z,
                     (float)v.p.x, (float)v.p.y, (float)v.p.z};
    }
};
struct PhToBboxF {
    HD BboxF operator()(const DPhoton& p) const {
        return BboxF{(float)p.pos.x, (float)p.pos.y, (float)p.pos.z,
                     (float)p.pos.x, (float)p.pos.y, (float)p.pos.z};
    }
};
struct BboxMergeF {
    HD BboxF operator()(const BboxF& a, const BboxF& b) const {
        return BboxF{fminf(a.mnx, b.mnx), fminf(a.mny, b.mny), fminf(a.mnz, b.mnz),
                     fmaxf(a.mxx, b.mxx), fmaxf(a.mxy, b.mxy), fmaxf(a.mxz, b.mxz)};
    }
};
// SPPM rMax: radius[i] where the visible point is valid, else 0 (max-reduced; exact).
struct SppmRMaxF {
    const double* radius; const unsigned char* valid;
    HD double operator()(long long i) const { return valid[i] ? radius[i] : 0.0; }
};
// Plain max functor (thrust::maximum is deprecated in CCCL 3.x, and naming it makes nvcc
// echo cub source context that MSBuild's canonical-error regex misparses as an error).
struct MaxD { HD double operator()(double a, double b) const { return a < b ? b : a; } };

// Scatter each light path's stored slab vertices into the compact array at its scanned
// offset (device twin of the old host compaction loop; per-path order preserved). The hero
// secondary rows ride along in lockstep so `secCompact[j*secStride + q]` stays paired with
// `compact[j]`; `secSlab == nullptr` (a `-heroc 1` session) skips that entirely.
__global__ void kVcmCompactScatter(const DVcmLV* slab, const DVcmSec* secSlab, int secStride,
                                   const int* lvCount, const int* pathBegin,
                                   DVcmLV* compact, DVcmSec* secCompact, int npix, int vcmCap) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < npix; i += stride) {
        int cnt = lvCount[i], b = pathBegin[i];
        const DVcmLV* src = slab + (size_t)i * vcmCap;
        for (int k = 0; k < cnt; ++k) compact[b + k] = src[k];
        if (secSlab) {
            const DVcmSec* ssrc = secSlab + (size_t)i * vcmCap * secStride;
            for (int k = 0; k < cnt; ++k) {
                int nu = src[k].nUp;
                DVcmSec* dst = secCompact + (size_t)(b + k) * secStride;
                for (int q = 0; q + 1 < nu; ++q) dst[q] = ssrc[(size_t)k * secStride + q];
            }
        }
    }
}

// Cell id per compacted light vertex. Mirrors the former host build EXPRESSION-FOR-
// EXPRESSION: positions and gLo are Real (float), so the subtraction happens in float;
// the divide promotes to double; floor/clamp in double/int. Identical bit results.
__global__ void kVcmCellKey(const DVcmLV* lv, int n, DVec3 gLo, double cell,
                            int gnx, int gny, int gnz, int* key) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int ix = (int)floor((lv[i].p.x - gLo.x) / cell);
        int iy = (int)floor((lv[i].p.y - gLo.y) / cell);
        int iz = (int)floor((lv[i].p.z - gLo.z) / cell);
        ix = min(max(ix, 0), gnx - 1);
        iy = min(max(iy, 0), gny - 1);
        iz = min(max(iz, 0), gnz - 1);
        key[i] = (iz * gny + iy) * gnx + ix;
    }
}

// Cell id per deposited photon (PhotonMap::cellCoord twin). The host converted DPhoton's
// float position to a double Vec3 BEFORE the all-double cell math, so promote first.
__global__ void kSppmCellKey(const DPhoton* ph, long long n, double lox, double loy, double loz,
                             double cellSize, int gnx, int gny, int gnz, int* key) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += stride) {
        int ix = (int)floor(((double)ph[i].pos.x - lox) / cellSize);
        int iy = (int)floor(((double)ph[i].pos.y - loy) / cellSize);
        int iz = (int)floor(((double)ph[i].pos.z - loz) / cellSize);
        ix = min(max(ix, 0), gnx - 1);
        iy = min(max(iy, 0), gny - 1);
        iz = min(max(iz, 0), gnz - 1);
        key[i] = (iz * gny + iy) * gnx + ix;      // int math, as in PhotonMap::cellIndex
    }
}

// Deposit record -> gather record, in sorted (cell-contiguous) order: out[i] converts
// ph[order[i]] (the stable sort's permutation == the host counting sort's). pos/n copy
// float-for-float (the host round-tripped float->double->float, which is exact); the
// cie*power/pi fold repeats the host's double math (see cieX64 note).
__global__ void kSppmGatherConvert(const DPhoton* ph, const int* order, long long n,
                                   DGatherPhoton* out) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += stride) {
        const DPhoton p = ph[order[i]];
        DGatherPhoton g;
        g.pos = p.pos;
        g.n   = p.n;
        const double l = (double)p.lambda;
        const double w = (double)p.power * (1.0 / DPI);
        g.pX = (float)(cieX64(l) * w);
        g.pY = (float)(cieY64(l) * w);
        g.pZ = (float)(cieZ64(l) * w);
        g.lambda = p.lambda;
        out[i] = g;
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

// ---- N4a: host half of the mode-W lattice bit-exactness probe (-checklattice) ----
// See lattice_probe.h for the column layout and render_cuda.h for the contract. The base
// list travels as a plain device buffer rather than a `__constant__` symbol, so this needs
// no API beyond the handful the HIP shim at the top of this file already covers.
int cudaRealBytes() { return gpu::realBytes(); }
bool cudaLatticeProbe(const unsigned long long* idx, int n, double* out) {
    if (!cudaAvailable() || n <= 0) return false;
    unsigned long long* dIdx = nullptr;
    unsigned* dBases = nullptr;
    double* dOut = nullptr;
    const size_t nOut = (size_t)n * kLatticeProbeCols;
    bool ok = cudaMalloc(&dIdx, (size_t)n * sizeof(unsigned long long)) == cudaSuccess &&
              cudaMalloc(&dBases, sizeof(kLatticeProbeBases)) == cudaSuccess &&
              cudaMalloc(&dOut, nOut * sizeof(double)) == cudaSuccess;
    if (ok) ok = cudaMemcpy(dIdx, idx, (size_t)n * sizeof(unsigned long long),
                            cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(dBases, kLatticeProbeBases, sizeof(kLatticeProbeBases),
                            cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        gpu::kLatticeProbe<<<(n + 255) / 256, 256>>>(dIdx, n, dBases, dOut);
        ok = cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess;
    }
    if (ok) ok = cudaMemcpy(out, dOut, nOut * sizeof(double),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
    if (dIdx)   cudaFree(dIdx);
    if (dBases) cudaFree(dBases);
    if (dOut)   cudaFree(dOut);
    return ok;
}

// Append one flushed line to the teardown-trace file named by $FTRACE_TEARDOWN_LOG
// (no-op when the env var is unset). Each call opens/appends/flushes/closes so that if
// the machine hard-reboots mid-teardown, the LAST line on disk names the exact step that
// was in flight when it died — the "log every step so the last log is the culprit"
// diagnostic. Deliberately heavyweight-per-line (reopen + fflush) precisely so nothing is
// buffered away and lost in a crash.
static void teardownLog(const char* step) {
    const char* path = std::getenv("FTRACE_TEARDOWN_LOG");
    if (!path || !*path) return;
    FILE* f = std::fopen(path, "a");
    if (!f) return;
    std::fprintf(f, "[teardown] %s\n", step);
    std::fflush(f);
    std::fclose(f);
}

void cudaGracefulShutdown() {
    // Only touch the driver if a device was actually brought up; probing when none is
    // present (or CUDA never initialised) would needlessly spin up a context just to
    // tear it down. cudaAvailable() is cached and cheap.
    if (!g_queried) { teardownLog("cuda: never initialised, skip"); return; }  // CUDA path never taken
    if (!g_available) { teardownLog("cuda: no device, skip"); return; }        // no usable device
    // Drain outstanding work first so the reset doesn't race a still-running kernel /
    // async copy, then destroy the primary context on this thread synchronously. Ignore
    // errors: this runs at shutdown and there's nothing left to salvage — the point is
    // simply to reclaim the context HERE, in-process, rather than leaving it for the
    // driver's asynchronous DPC teardown after main() returns. Bracketing each driver call
    // with a flushed log line means a BSOD during teardown leaves the offending call as the
    // last line in $FTRACE_TEARDOWN_LOG.
    teardownLog("cuda: cudaDeviceSynchronize enter");
    cudaDeviceSynchronize();
    teardownLog("cuda: cudaDeviceSynchronize returned");
    teardownLog("cuda: cudaDeviceReset enter");
    cudaDeviceReset();
    teardownLog("cuda: cudaDeviceReset returned");
}

// --- Emitter shapes the device kernels actually implement ----------------------
// Single source of truth for the host EmitterShape -> DEmitter::shape mapping, shared
// by BOTH the support gate below and the DEmitter upload, so the two can never drift.
// Returns the device shape code, or -1 for a shape the device has no branch for.
//
// This is deliberately a CLOSED whitelist that fails SAFE: a shape the kernels don't
// implement sends the scene to the CPU tracer instead of being coerced into a
// different shape. It replaces a ternary chain in the upload that ended in `: 0` --
// "anything I don't recognise is a Quad" -- so a shape added host-side but not yet
// implemented device-side would reach the kernel as a Quad with a garbage (typically
// all-zero) origin/u/v basis. Malformed geometry inside a kernel does not fail cleanly:
// an out-of-range/degenerate sample can fault the display driver and take the whole
// machine down with it (see the teardown/BSOD logging above), so the mapping must be
// exhaustive by construction rather than by convention.
//
// The switch lists every enumerator and has NO default, so ADDING a new EmitterShape
// raises a compiler warning here (MSVC C4062) rather than silently falling through;
// the `return -1` after it is the fail-safe should one slip past anyway.
static int deviceEmitterShapeCode(EmitterShape s) {
    switch (s) {
        case EmitterShape::Quad:     return 0;
        case EmitterShape::Sphere:   return 1;
        case EmitterShape::Spot:     return 2;
        case EmitterShape::Env:      return 3;
        case EmitterShape::Cylinder: return 4;
        case EmitterShape::Mesh:     return 5;
        case EmitterShape::Sun:      return 6;
    }
    return -1;   // unknown / newly-added shape: CPU fallback, never a silent coercion
}

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
    // Parametric records (§records) drive a material's slots from a per-hit driver
    // sampling a named LUT bank. Stages 6a/6b put BOTH slots on the device forward +
    // backward-reference tracers: REFLECT (constant selStop baked into reflect[], driven
    // via recCoeff LUT + recDrivers → dRecordReflect / dReflectSlot / dDiffuseRho) and the
    // SCALAR roughness slot (direct expr / constant selStop / driven stops evaluated
    // per-hit → dRecordRoughness / dMatRoughness). The only remaining CPU-only case is a
    // per-hit *driven* scalar channel with >64 stops, which overflows the device interp
    // arrays (dRecSampleScalar vs[64]) — vanishingly rare, kept on CPU for safety.
    auto usesRecord = [&](int matId) {
        if (matId < 0 || matId >= (int)scene.mats.size()) return false;
        const RecBinding* rb = scene.mats[matId].recBindingFor(REC_SLOT_ROUGHNESS);
        if (!rb) return false;
        if (rb->recordIndex >= 0 && rb->recordIndex < (int)scene.records.size() && rb->selStop < 0) {
            const Record& rec = scene.records[rb->recordIndex];
            if (rb->channel >= 0 && rb->channel < (int)rec.channels.size() &&
                (int)rec.channels[rb->channel].stops.size() > 64) return true;
        }
        return false;
    };
    auto unsupported = [&](int matId) {
        if (oversizedMultilayer(matId)) return true;
        if (usesPaletteTex(matId)) return true;
        if (usesRecord(matId)) return true;
        // The physical layered stack (coat interface over a weighted body) is CPU-only;
        // the device shadeStep has no Layered branch, so any Layered material forces a
        // CPU forward/backward fallback (like indexed palettes).
        if (matId >= 0 && matId < (int)scene.mats.size() &&
            scene.mats[matId].type == MatType::Layered) return true;
        // The fiber BCSDF (MatType::Hair) runs on the device as of 0.181.0: the dhair
        // namespace ports hair.h's single-fiber Marschner/Chiang + Yan-medulla model, the
        // curve intersector fills DHit::fiberRadius/tangent, and shadeStep / bkRadiance
        // have D_HAIR branches. Dual scattering and the fur aggregate volume are still
        // CPU-only, but both are RENDER-OPTION gates (-dual-scatter / -fur-volume in
        // main.cpp's backwardOnGpuOk), not scene-material properties, so no reject here.
        if (matId >= 0 && matId < (int)scene.mats.size() &&
            scene.mats[matId].type == MatType::Mix) {
            const Material& mx = scene.mats[matId];
            if ((int)mx.mixChildren.size() > D_MIXMAX) return true;
            for (int c : mx.mixChildren)
                if (oversizedMultilayer(c) || usesRecord(c))
                    return true;
        }
        return false;
    };
    for (const auto& t : scene.tris)      if (unsupported(t.matId)) return false;
    for (const auto& s : scene.spheres)   if (unsupported(s.matId)) return false;
    for (const auto& im : scene.implicits) if (unsupported(im.matId)) return false;
    // Curve / fiber round cones (curve.h, TODO §P1) have a device twin as of 0.151.0, so
    // strands no longer force a whole-scene CPU fallback — but their MATERIALS still go
    // through the same support screen as every other primitive's.
    for (const auto& cs : scene.curveSegs) if (unsupported(cs.matId)) return false;
    // `emit pattern:` / `emit_map` runs on the device (0.82.0). Unlike reflect/transmit it
    // is not a one-sided throughput slot: the same profile has to be applied on BOTH sides
    // of transport — emission-on-hit AND the Le at an emitter-sampled point — because MIS
    // combines them, so every device emission read goes through dEmitPatMul (the hit side)
    // or dEmitterSamplePointPat (the sampler side) rather than reading emitSpd raw. No
    // fallback needed; a patterned scene renders on the GPU in every supported mode.
    // Spectral water-droplet (rainbow) phase is now on the device: the (lambda x mu) Airy
    // table + per-lambda CDF (rainbow.h) is uploaded per medium and dMedPhase / dMedPhaseSample
    // reproduce the bow bit-closely against the CPU tracer (M10). No fallback needed.
    // Environment lighting runs on-device: the kernel emits env photons from the scene
    // bounding sphere (shape==3) and the directly-viewed background is added by the
    // backend-agnostic addEnvBackground() pass. Both a constant env and an IMAGE-based
    // env (lat-long map: the 2D luminance CDF, per-texel JH coeff/scale, and mean
    // coeff/scale are uploaded, and the sampler/reweight are ported to the device) are
    // supported (increments 1b and 2c).
    // Volumetric blackbody emission ("fire": a medium with a `temperature` grid +
    // `emission`, ROADMAP C3) is now supported on-device: the temperature field is
    // uploaded as a sparse brick grid, DScene carries the emissive-volume table + per-
    // volume Planck-λ CDF, and genPhoton has a volume-birth branch + isotropic emission
    // splat mirroring the CPU tracer. An emissive-only scene (nEmitters==0) never indexes
    // sc.emitters because volumeBirth is always true when totalPower==0.
    // A distant `sun` emitter (shape==6) is supported too: genPhoton/genPhotonHero have a
    // parallel-beam birth branch over the scene cross-section, bkEmitterGeom/bkNeeLight/
    // bkNeeVolume/bkNeeLightRGB do the cone NEE, and the ray-miss paths add the directly-
    // viewed solar disc under the same `specularArrival` gate as the CPU tracer.
    // Emitter shapes: a shape the device kernels have no branch for must send the scene to
    // the CPU tracer rather than be coerced into a different shape at upload time. Handing
    // a kernel an emitter with a zeroed origin/u/v basis does not fail cleanly -- it can
    // fault the display driver -- so this gate is the safety net for every GPU mode (all
    // the other *Supported() gates chain to this one).
    // A Mesh emitter additionally REQUIRES a non-empty triangle CDF: the device sampler's
    // shape==5 branch indexes em.meshTris unconditionally, and with meshTriN==0 it clamps
    // to index -1 on a null pointer. Degenerate/empty mesh lights therefore go to the CPU.
    for (const auto& e : scene.emitters) {
        if (deviceEmitterShapeCode(e.shape) < 0) return false;
        if (e.shape == EmitterShape::Mesh && e.meshTris.empty()) return false;
    }
    return true;
}

// Does any primitive bind a hair material (directly or as a Mix child)? The device
// FORWARD and BACKWARD tracers shade hair (0.181.0), but the BDPT and photon-map
// machinery does not: dBsdfF / dBsdfPdf / dRandomWalk / dConnectBDPT have no fiber
// branch (a hair vertex needs the strand frame, the impact parameter and a non-cosine
// connection response in the MIS densities), and the mode-M gather treats every query
// surface as Lambertian. Those gates call this and fall back to their CPU twins.
static bool sceneUsesHairMaterial(const Scene& scene) {
    auto isHair = [&](int matId) {
        if (matId < 0 || matId >= (int)scene.mats.size()) return false;
        const Material& m = scene.mats[matId];
        if (m.type == MatType::Hair) return true;
        if (m.type == MatType::Mix)
            for (int c : m.mixChildren)
                if (c >= 0 && c < (int)scene.mats.size() &&
                    scene.mats[c].type == MatType::Hair) return true;
        return false;
    };
    for (const auto& t : scene.tris)       if (isHair(t.matId)) return true;
    for (const auto& s : scene.spheres)    if (isHair(s.matId)) return true;
    for (const auto& im : scene.implicits) if (isHair(im.matId)) return true;
    for (const auto& cs : scene.curveSegs) if (isHair(cs.matId)) return true;
    return false;
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
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// Baked device scene + camera, plus the list of device allocations to free. Shared
// by the forward megakernel (renderForwardCuda) and the BDPT kernel (renderBdptCuda)
// so the ~150-line Scene->POD baking lives in exactly one place.
struct DUpload {
    gpu::DScene  sc{};
    gpu::DCamera dc{};
    std::vector<void*> frees;
    // Record a device allocation for later freeUpload(); returns it for chaining.
    void* keep(void* p) { if (p) frees.push_back(p); return p; }
};
static void freeUpload(DUpload& up) {
    for (void* p : up.frees) cudaFree(p);
    up.frees.clear();
}

// Bake the std::function Scene into POD device tables and upload them (camera-
// independent). Every cudaMalloc'd pointer is recorded in up.frees; call freeUpload(up)
// when done. The camera is baked separately (bakeCamera) so one baked scene can serve a
// whole multi-camera shared pass.
static void buildUploadScene(const Scene& scene, DUpload& up) {
    using namespace gpu;
    auto keep = [&](void* p) { if (p) up.frees.push_back(p); return p; };

    // NOTE: no cudaLimitStackSize fiddling here, deliberately. The kernels' call
    // graphs are kept statically acyclic (see the dhair Bessel comment) so nvlink can
    // record a real per-kernel MIN_STACK_SIZE and the driver auto-reserves exactly
    // that much per thread at launch — the mechanism the tracers have always relied
    // on (they need 6–12 KB, far beyond the 1 KB cudaLimitStackSize default, which
    // only matters as the fallback when MIN_STACK is undeterminable).

    // --- bake geometry ---
    // Convert one host Tri (already in its own space) into a device DTri. Shared by the
    // base Scene::tris and the per-BLAS local-space tris so the conversion lives once.
    auto bakeTri = [](const Tri& t) {
        DTri d;
        d.v0 = {t.v0.x, t.v0.y, t.v0.z}; d.v1 = {t.v1.x, t.v1.y, t.v1.z};
        d.v2 = {t.v2.x, t.v2.y, t.v2.z}; d.gn = {t.gn.x, t.gn.y, t.gn.z};
        d.uv0 = {t.uv0.x, t.uv0.y, t.uv0.z};
        d.uv1 = {t.uv1.x, t.uv1.y, t.uv1.z};
        d.uv2 = {t.uv2.x, t.uv2.y, t.uv2.z};
        d.n0 = {t.n0.x, t.n0.y, t.n0.z};
        d.n1 = {t.n1.x, t.n1.y, t.n1.z};
        d.n2 = {t.n2.x, t.n2.y, t.n2.z};
        d.tangent = {t.tangent.x, t.tangent.y, t.tangent.z};
        d.bitangentSign = t.bitangentSign;
        d.curvature = t.curvature;      // O3 per-face mean curvature (from Tri::finalize)
        d.matId = t.matId; d.sensorId = t.sensorId;
        return d;
    };
    // Instancing now uses a true TWO-LEVEL BVH on the device (matching the CPU): base
    // Scene::tris stay as the flat tri list, and each MeshInstance becomes one TLAS leaf
    // (in Scene::bvh, uploaded verbatim below) that references a shared DBlas — no
    // world-triangle expansion, so device memory scales with UNIQUE geometry, not with
    // the instance count. See the DBlas/DInstance traversal in closestHit/occluded.
    const bool haveInstances = !scene.instances.empty();
    std::vector<DTri> tris(scene.tris.size());
    for (size_t i = 0; i < scene.tris.size(); ++i) tris[i] = bakeTri(scene.tris[i]);
    std::vector<DSphere> sph(scene.spheres.size());
    for (size_t i = 0; i < scene.spheres.size(); ++i) {
        const Sphere& s = scene.spheres[i]; DSphere& d = sph[i];
        d.c = {s.c.x, s.c.y, s.c.z}; d.r = s.r; d.matId = s.matId;
    }
    // Flat mirror surfaces, grouped per plane by Scene::buildMirrorPlanes (which the host
    // already ran inside Scene::build). A straight field-for-field copy: the device solves
    // the reflection in the same doubles the host does, so there is nothing to narrow.
    std::vector<DMirrorPlane> mirrorPlanes(scene.mirrorPlanes.size());
    for (size_t i = 0; i < scene.mirrorPlanes.size(); ++i) {
        const Scene::MirrorPlane& m = scene.mirrorPlanes[i]; DMirrorPlane& d = mirrorPlanes[i];
        d.nx = m.n.x; d.ny = m.n.y; d.nz = m.n.z; d.d = m.d;
        d.lox = m.lo.x; d.loy = m.lo.y; d.loz = m.lo.z;
        d.hix = m.hi.x; d.hiy = m.hi.y; d.hiz = m.hi.z;
    }
    // Curve / fiber round cones. The host CurveSeg is already a flat POD in this exact
    // shape (curve.h says so, and this is what it was for), so the bake is a straight
    // narrowing copy — no flattening, no per-strand indirection. curveId is host-side
    // diagnostics only and is not uploaded.
    std::vector<DCurveSeg> curveSegs(scene.curveSegs.size());
    for (size_t i = 0; i < scene.curveSegs.size(); ++i) {
        const CurveSeg& s = scene.curveSegs[i]; DCurveSeg& d = curveSegs[i];
        d.p0 = {(Real)s.p0.x, (Real)s.p0.y, (Real)s.p0.z};
        d.p1 = {(Real)s.p1.x, (Real)s.p1.y, (Real)s.p1.z};
        d.r0 = (Real)s.r0; d.r1 = (Real)s.r1;
        d.matId = s.matId; d.u0 = s.u0; d.u1 = s.u1;
    }

    // Top-level BVH: upload Scene::bvh VERBATIM in every case. Its prim-index layout is
    // [tris | spheres | implicits | curveSegs | instances] — the device leaf dispatch in
    // closestHit/occluded now understands all five ranges (an instance leaf transforms
    // the ray into BLAS-local space and walks the shared sub-BVH), so no flat rebuild /
    // instance expansion is needed. This is bit-identical to the old path for scenes
    // with no instances, and the memory win (shared BLAS) for scenes with them.
    const Bvh* srcBvh = &scene.bvh;
    std::vector<DNode> nodes(srcBvh->nodes.size());
    for (size_t i = 0; i < srcBvh->nodes.size(); ++i) {
        const BvhNode& b = srcBvh->nodes[i]; DNode& d = nodes[i];
        d.lo = {b.box.lo.x, b.box.lo.y, b.box.lo.z};
        d.hi = {b.box.hi.x, b.box.hi.y, b.box.hi.z};
        d.left = b.left; d.right = b.right; d.first = b.first; d.count = b.count;
    }
    std::vector<int> primIdx = srcBvh->primIdx;

    // --- bake the two-level BVH (shared BLAS pools + instance table) ---
    // Each Blas contributes its local-space tris, its own BVH nodes, and its primIdx to
    // three concatenated pools; a DBlas records the per-BLAS start offsets. Each
    // MeshInstance becomes a DInstance carrying the world->local affine (for the ray),
    // the (toWorld)^-T normal matrix (host-precomputed so the device does no inverse),
    // its blasId, and any material override.
    std::vector<DBlas>     dblas(scene.blasList.size());
    std::vector<DTri>      blasTris;
    std::vector<DNode>     blasNodes;
    std::vector<int>       blasPrim;
    for (size_t bi = 0; bi < scene.blasList.size(); ++bi) {
        const Blas& bl = scene.blasList[bi];
        dblas[bi].triOff  = (int)blasTris.size();
        dblas[bi].nodeOff = (int)blasNodes.size();
        dblas[bi].primOff = (int)blasPrim.size();
        for (const Tri& t : bl.tris) blasTris.push_back(bakeTri(t));
        for (const BvhNode& b : bl.bvh.nodes) {
            DNode d;
            d.lo = {b.box.lo.x, b.box.lo.y, b.box.lo.z};
            d.hi = {b.box.hi.x, b.box.hi.y, b.box.hi.z};
            d.left = b.left; d.right = b.right; d.first = b.first; d.count = b.count;
            blasNodes.push_back(d);
        }
        blasPrim.insert(blasPrim.end(), bl.bvh.primIdx.begin(), bl.bvh.primIdx.end());
    }
    std::vector<DInstance> dinst(scene.instances.size());
    for (size_t ii = 0; ii < scene.instances.size(); ++ii) {
        const MeshInstance& in = scene.instances[ii]; DInstance& d = dinst[ii];
        for (int k = 0; k < 9; ++k) d.Lm[k] = in.toLocal.m[k];
        d.Lt[0] = in.toLocal.t.x; d.Lt[1] = in.toLocal.t.y; d.Lt[2] = in.toLocal.t.z;
        // Normal local->world = (toWorld linear)^-T. toWorld.applyNormal(n) uses
        // (toWorld.inverse().m) read in COLUMN order, i.e. the transpose of inv.m. Bake
        // that transposed matrix so the device applies it as a plain row-major matvec.
        Affine inv = in.toWorld.inverse();
        d.Nm[0] = inv.m[0]; d.Nm[1] = inv.m[3]; d.Nm[2] = inv.m[6];
        d.Nm[3] = inv.m[1]; d.Nm[4] = inv.m[4]; d.Nm[5] = inv.m[7];
        d.Nm[6] = inv.m[2]; d.Nm[7] = inv.m[5]; d.Nm[8] = inv.m[8];
        for (int k = 0; k < 9; ++k) d.Wm[k] = in.toWorld.m[k];   // tangent local->world (C6)
        d.curvScale = in.curvScale;                              // curvature local->world (O3)
        d.blasId = in.blasId; d.matOverride = in.matOverride;
    }
    (void)haveInstances;

    // --- bake implicit surfaces (isosurface / CSG / metaballs) ---
    // Flatten every Implicit's postfix FieldNode array into one pool; each DImplicit
    // slices it by [nodeOff, nodeOff+nodeN). BVH prims >= nTris+nSph index these.
    std::vector<DFieldNode> fieldNodes;
    std::vector<PatNode>    fieldExprNodes;   // flat pool for DF_EXPR formulas (all implicits)
    // Append a host field program (FieldNode postfix + its private expr pool) into the
    // shared device pools, rebasing DF_EXPR leaf offsets. Writes the slice [outOff,outN).
    // Shared by isosurface geometry and implicit-shaped fog bounds so the conversion
    // (and its expr-pool rebasing) lives in exactly one place.
    auto appendFieldProgram = [&](const std::vector<FieldNode>& nodes,
                                  const std::vector<PatNode>& expr,
                                  int& outOff, int& outN) {
        outOff = (int)fieldNodes.size();
        outN   = (int)nodes.size();
        int exprBase = (int)fieldExprNodes.size();
        fieldExprNodes.insert(fieldExprNodes.end(), expr.begin(), expr.end());
        for (const FieldNode& fn : nodes) {
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
    };
    std::vector<DImplicit>  dimpl(scene.implicits.size());
    for (size_t i = 0; i < scene.implicits.size(); ++i) {
        const Implicit& im = scene.implicits[i]; DImplicit& d = dimpl[i];
        d.matId   = im.matId;
        d.lo[0] = im.bounds.lo.x; d.lo[1] = im.bounds.lo.y; d.lo[2] = im.bounds.lo.z;
        d.hi[0] = im.bounds.hi.x; d.hi[1] = im.bounds.hi.y; d.hi[2] = im.bounds.hi.z;
        d.lipschitz = im.lipschitz; d.minStep = im.minStep;
        d.method = (int)im.method; d.refine = (int)im.refine; d.sampleStep = im.sampleStep;
        d.uvProj = (int)im.uvProj; d.uvAxis = im.uvAxis;
        {
            const Aabb& ub = im.uvBoundsSet ? im.uvBounds : im.bounds;
            d.uvLo[0] = ub.lo.x; d.uvLo[1] = ub.lo.y; d.uvLo[2] = ub.lo.z;
            d.uvHi[0] = ub.hi.x; d.uvHi[1] = ub.hi.y; d.uvHi[2] = ub.hi.z;
        }
        d.container = (int)im.container;
        d.sphereCenter[0] = im.sphereCenter.x;
        d.sphereCenter[1] = im.sphereCenter.y;
        d.sphereCenter[2] = im.sphereCenter.z;
        d.sphereRadius = im.sphereRadius;
        d.capped = im.capped ? 1 : 0;
        d.boxOriented = im.boxOriented ? 1 : 0;
        for (int k = 0; k < 9; ++k) d.boxInvM[k] = im.boxInv.m[k];
        d.boxInvT[0] = im.boxInv.t.x; d.boxInvT[1] = im.boxInv.t.y; d.boxInvT[2] = im.boxInv.t.z;
        d.boxLo[0] = im.boxLo.x; d.boxLo[1] = im.boxLo.y; d.boxLo[2] = im.boxLo.z;
        d.boxHi[0] = im.boxHi.x; d.boxHi[1] = im.boxHi.y; d.boxHi[2] = im.boxHi.z;
        // Rebase this implicit's field program (and its private expr pool) into the
        // shared device pools; sets d.nodeOff/d.nodeN.
        appendFieldProgram(im.nodes, im.exprNodes, d.nodeOff, d.nodeN);
    }

    // Implicit-shaped fog bounds (Medium::boundShape == Implicit) carry a copy of a
    // named isosurface's field program. Bake each into the SAME device field pools so
    // the density evaluator can test membership on-device; record the per-medium slice.
    struct MedFieldSlice { int off = -1, n = 0; };
    std::vector<MedFieldSlice> medField(scene.media.size());
    for (size_t i = 0; i < scene.media.size(); ++i) {
        const Medium& m = scene.media[i];
        if (m.boundShape != MediumBound::Implicit || m.boundField.empty()) continue;
        appendFieldProgram(m.boundField, m.boundFieldExpr, medField[i].off, medField[i].n);
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
    // Parametric-record reflect pools (§records stage 6a): each per-hit driven reflect
    // binding flattens its channel's baked JH coeff LUT (REC_LUT_N*3 doubles) into
    // recCoeffPool and copies its driver program into recDrvPool; DMaterial slices both.
    std::vector<double>  recCoeffPool;
    std::vector<PatNode> recDrvPool;
    // Scalar (roughness) record stops: each stop's domain position + its per-hit expression
    // program (appended to recDrvPool). DMaterial::recRoughStopOff/N slice this (stage 6b).
    std::vector<DRecScalarStop> recStopPool;
    for (size_t i = 0; i < scene.mats.size(); ++i) {
        const Material& m = scene.mats[i]; DMaterial& d = mats[i];
        d.type = (int)m.type;
        bakeSpec(m.reflect, d.reflect);
        bakeSpec(m.ior, d.ior);
        bakeSpec(m.substrateK, d.substrateK);
        bakeSpec(m.absorb, d.absorb);   // Beer-Lambert interior tint (colored glass)
        bakeSpec(m.transmit, d.transmit); // diffuse-transmission back-lobe albedo (translucent)
        // Fast RGB backward (mode R -rgb) bakes: linear-sRGB reflect/transmit albedo,
        // a representative achromatic index (ior at 550 nm) and a 3-tap Beer-Lambert
        // sigma_a at the R/G/B pivots (610/550/465 nm). Inert for the spectral paths.
        { Vec3 ra = rgbbake::reflToRgb(m.reflect); d.rgbAlbedo = {ra.x, ra.y, ra.z}; }
        { Vec3 rt = rgbbake::reflToRgb(m.transmit); d.rgbTransmit = {rt.x, rt.y, rt.z}; }
        d.rgbAbsorb = { m.absorb ? m.absorb(610.0) : 0.0,
                        m.absorb ? m.absorb(550.0) : 0.0,
                        m.absorb ? m.absorb(465.0) : 0.0 };
        d.rgbIor = m.ior ? m.ior(550.0) : 1.0;
        d.priority = m.priority;        // nested-dielectric priority (INT_MIN == unset)
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
            bakeSpec(m.fluoEmit, d.fluoEmitSpec);       // continuous M(lambda) for backward gOut
            d.fluoMint = m.fluoEmitSampler.integral;
        } else {
            d.fluoCdfOffset = 0; d.fluoCdfN = 0; d.fluoCdfStep = 1.0;
            for (int s = 0; s < SPEC_N; ++s) d.fluoEmitSpec[s] = 0.0;
            d.fluoMint = 0.0;
        }
        // Excitation CDF (absorb x illuminant), appended to the SAME flat buffer with
        // its own slice. N == 0 makes the device sampler fall back to the illuminant,
        // matching the host's `fluoInSampler.integral > 0 ? ... : scene.emitSampler`.
        if (m.type == MatType::Fluorescent && !m.fluoInSampler.cdf.empty() &&
            m.fluoInSampler.integral > 0.0) {
            d.fluoInCdfOffset = (int)fluoCdfAll.size();
            d.fluoInCdfN = (int)m.fluoInSampler.cdf.size();
            d.fluoInCdfStep = m.fluoInSampler.step;
            fluoCdfAll.insert(fluoCdfAll.end(), m.fluoInSampler.cdf.begin(),
                              m.fluoInSampler.cdf.end());
        } else {
            d.fluoInCdfOffset = 0; d.fluoInCdfN = 0; d.fluoInCdfStep = 1.0;
        }
        d.roughness = m.roughness;
        d.filmIor = m.filmIor; d.filmThickness = m.filmThickness;
        d.roughnessTex = m.roughnessTex;
        d.filmThicknessTex = m.filmThicknessTex;
        d.normalTex = m.normalTex;
        d.normalStrength = m.normalStrength;
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
        d.readsCavity = m.readsCavity ? 1 : 0;
        d.roughnessPat = m.roughnessPat;
        d.filmThicknessPat = m.filmThicknessPat;
        d.mixWeightPat = m.mixWeightPat;
        d.reflectPat = m.reflectPat;
        d.transmitPat = m.transmitPat;
        d.emitPat = m.emitPat;
        // Self-emission carried by the material (the emitter-less case: isosurfaces,
        // CSG/quadric solids). bakeSpec of a null Spectrum yields all zeros, so a
        // non-emissive material costs nothing but the storage.
        d.matIsLight = m.isLight ? 1 : 0;
        bakeSpec(m.emit, d.matEmit);
        { Vec3 le = m.emit ? rgbbake::emitToRgb(m.emit) : Vec3{0, 0, 0};
          d.rgbMatEmit = {le.x, le.y, le.z}; }
        // --- parametric-record REFLECT binding (§records stage 6a) ---
        // Device twin of recordReflectBound. A constant selStop binding bakes the stop's
        // colour straight into reflect[] (so the plain specLookup path is exact, no device
        // branch). A per-hit driven binding flattens the channel's baked coeff LUT + copies
        // its driver program, and dRecordReflect samples them. Scalar (roughness) record
        // bindings are gated to CPU (stage 6b) — cudaForwardSupported rejects them.
        d.recReflDriven = 0; d.recReflOff = 0;
        d.recReflDrvOff = 0; d.recReflDrvN = 0;
        d.recReflLo = 0.0f;  d.recReflHi = 1.0f;
        if (const RecBinding* rb = m.recBindingFor(REC_SLOT_REFLECT)) {
            if (rb->recordIndex >= 0 && rb->recordIndex < (int)scene.records.size()) {
                const Record& rec = scene.records[rb->recordIndex];
                const RecChannel& ch = rec.channels[rb->channel];
                if (rb->selStop >= 0 && rb->selStop < (int)ch.stops.size()) {
                    bakeSpec(ch.stops[rb->selStop].color, d.reflect);   // constant stop → bake
                } else if ((int)ch.coeff.size() == REC_LUT_N) {
                    d.recReflDriven = 1;
                    d.recReflOff = (int)recCoeffPool.size();
                    for (const auto& c : ch.coeff) {                    // REC_LUT_N * 3 doubles
                        recCoeffPool.push_back(c[0]);
                        recCoeffPool.push_back(c[1]);
                        recCoeffPool.push_back(c[2]);
                    }
                    d.recReflDrvOff = (int)recDrvPool.size();
                    d.recReflDrvN   = (int)rb->driver.size();
                    recDrvPool.insert(recDrvPool.end(), rb->driver.begin(), rb->driver.end());
                    d.recReflLo = (float)rec.lo; d.recReflHi = (float)rec.hi;
                }
            }
        }
        // --- parametric-record ROUGHNESS binding (scalar slot — §records stage 6b) ---
        // Device twin of the record branch of materialRoughness. Scalar stops evaluate
        // per-hit (they can reference hit vars), so — unlike the reflect coeff LUT — nothing
        // bakes to a table: each stop's expression program is copied verbatim and evaluated
        // on-device. `appendProg` copies a PatNode program into the shared recDrvPool.
        auto appendProg = [&](const std::vector<PatNode>& prog, int& off, int& n) {
            off = (int)recDrvPool.size();
            n   = (int)prog.size();
            recDrvPool.insert(recDrvPool.end(), prog.begin(), prog.end());
        };
        d.recRoughMode = -1;
        d.recRoughDrvOff = 0; d.recRoughDrvN = 0;
        d.recRoughStopOff = 0; d.recRoughStopN = 0;
        d.recRoughInterp = (int)RecInterp::Linear;
        d.recRoughLo = 0.0f; d.recRoughHi = 1.0f;
        if (const RecBinding* rb = m.recBindingFor(REC_SLOT_ROUGHNESS)) {
            if (rb->recordIndex < 0) {                                   // direct scalar expr
                d.recRoughMode = 0;
                appendProg(rb->driver, d.recRoughDrvOff, d.recRoughDrvN);
            } else if (rb->recordIndex < (int)scene.records.size()) {
                const Record& rec = scene.records[rb->recordIndex];
                const RecChannel& ch = rec.channels[rb->channel];
                auto pushStop = [&](const RecStop& s) {
                    DRecScalarStop ds; ds.pos = s.pos;
                    appendProg(s.expr, ds.exprOff, ds.exprN);
                    recStopPool.push_back(ds);
                };
                if (rb->selStop >= 0 && rb->selStop < (int)ch.stops.size()) {
                    d.recRoughMode = 1;                                  // constant selStop
                    d.recRoughStopOff = (int)recStopPool.size();
                    d.recRoughStopN   = 1;
                    pushStop(ch.stops[rb->selStop]);
                } else if ((int)ch.stops.size() <= 64) {                 // per-hit driven
                    d.recRoughMode = 2;
                    appendProg(rb->driver, d.recRoughDrvOff, d.recRoughDrvN);
                    d.recRoughStopOff = (int)recStopPool.size();
                    d.recRoughStopN   = (int)ch.stops.size();
                    for (const RecStop& s : ch.stops) pushStop(s);
                    d.recRoughInterp = (int)rec.interp;
                    d.recRoughLo = (float)rec.lo; d.recRoughHi = (float)rec.hi;
                }
                // (>64 stops: recRoughMode stays -1; cudaForwardSupported gates it to CPU.)
            }
        }
        // --- hair fiber BCSDF (MatType::Hair) ---
        // Baked unconditionally (the fields are defaults/zero spectra on every other
        // type, and dHairShadeAt only reads them behind a D_HAIR dispatch). The
        // from-reflectance inversion is NOT pre-baked: it depends on the diffuse rho at
        // the hit (texture/pattern-driven), so the device inverts per-hit exactly like
        // the CPU hairShadeAt.
        d.hairEta      = m.hairEta;
        d.hairBetaM    = m.hairBetaM;
        d.hairBetaN    = m.hairBetaN;
        d.hairAlpha    = m.hairAlpha;
        d.hairKappa    = m.hairKappa;
        d.hairMedullaG = m.hairMedullaG;
        bakeSpec(m.hairSigmaA,        d.hairSigmaA);
        bakeSpec(m.hairMedullaSigmaS, d.hairMedullaSigmaS);
        bakeSpec(m.hairMedullaSigmaA, d.hairMedullaSigmaA);
        d.hairSigmaAFromReflect = m.hairSigmaAFromReflect ? 1 : 0;
    }

    // --- upload geometry/materials ---
    DTri*      d_tris  = tris.empty()    ? nullptr : (DTri*)keep(uploadVec(tris));
    DSphere*   d_sph   = sph.empty()     ? nullptr : (DSphere*)keep(uploadVec(sph));
    DMirrorPlane* d_mirp = mirrorPlanes.empty() ? nullptr
                                                : (DMirrorPlane*)keep(uploadVec(mirrorPlanes));
    int* d_dielSph   = scene.dielSphereIdx.empty()   ? nullptr
                                                     : (int*)keep(uploadVec(scene.dielSphereIdx));
    int* d_mirrorSph = scene.mirrorSphereIdx.empty() ? nullptr
                                                     : (int*)keep(uploadVec(scene.mirrorSphereIdx));
    DNode*     d_nodes = nodes.empty()   ? nullptr : (DNode*)keep(uploadVec(nodes));
    int*       d_prim  = primIdx.empty() ? nullptr : (int*)keep(uploadVec(primIdx));
    DMaterial* d_mats  = mats.empty()    ? nullptr : (DMaterial*)keep(uploadVec(mats));
    DFieldNode* d_fnodes = fieldNodes.empty() ? nullptr : (DFieldNode*)keep(uploadVec(fieldNodes));
    PatNode*    d_fexpr  = fieldExprNodes.empty() ? nullptr : (PatNode*)keep(uploadVec(fieldExprNodes));
    // FP32 mirrors of the (final, fully-appended) field pools for the sphere-trace
    // march. Same order/offsets as the double pools; converted once here so the march
    // never touches FP64 loads or F64->F32 cvts.
    std::vector<DFieldNodeF> fieldNodesF(fieldNodes.size());
    for (size_t i = 0; i < fieldNodes.size(); ++i) {
        const DFieldNode& s = fieldNodes[i]; DFieldNodeF& d = fieldNodesF[i];
        d.op = s.op;
        for (int k = 0; k < 4; ++k) d.p[k] = (float)s.p[k];
        for (int k = 0; k < 9; ++k) d.inv[k] = (float)s.inv[k];
        d.tx = (float)s.tx; d.ty = (float)s.ty; d.tz = (float)s.tz;
        d.scale = (float)s.scale;
        d.exprOff = s.exprOff; d.exprN = s.exprN;
    }
    std::vector<PatNodeF> fieldExprNodesF(fieldExprNodes.size());
    for (size_t i = 0; i < fieldExprNodes.size(); ++i) {
        fieldExprNodesF[i].op = (int)fieldExprNodes[i].op;
        fieldExprNodesF[i].a  = (float)fieldExprNodes[i].a;
    }
    DFieldNodeF* d_fnodesF = fieldNodesF.empty() ? nullptr : (DFieldNodeF*)keep(uploadVec(fieldNodesF));
    PatNodeF*    d_fexprF  = fieldExprNodesF.empty() ? nullptr : (PatNodeF*)keep(uploadVec(fieldExprNodesF));
    DImplicit*  d_impl   = dimpl.empty()      ? nullptr : (DImplicit*)keep(uploadVec(dimpl));
    DCurveSeg*  d_curves = curveSegs.empty()  ? nullptr : (DCurveSeg*)keep(uploadVec(curveSegs));
    // Two-level BVH pools (shared BLAS + instance table). Empty for scenes with no instances.
    DInstance*  d_inst   = dinst.empty()     ? nullptr : (DInstance*)keep(uploadVec(dinst));
    DBlas*      d_blas   = dblas.empty()     ? nullptr : (DBlas*)keep(uploadVec(dblas));
    DNode*      d_blasN  = blasNodes.empty() ? nullptr : (DNode*)keep(uploadVec(blasNodes));
    int*        d_blasP  = blasPrim.empty()  ? nullptr : (int*)keep(uploadVec(blasPrim));
    DTri*       d_blasT  = blasTris.empty()  ? nullptr : (DTri*)keep(uploadVec(blasTris));
    PatNode*    d_pnodes = patNodes.empty()   ? nullptr : (PatNode*)keep(uploadVec(patNodes));
    DPattern*   d_pat    = dpat.empty()       ? nullptr : (DPattern*)keep(uploadVec(dpat));
    // Parametric-record reflect pools (§records stage 6a) + scalar-stop pool (stage 6b).
    double*     d_recCoeff = recCoeffPool.empty() ? nullptr : (double*)keep(uploadVec(recCoeffPool));
    PatNode*    d_recDrv   = recDrvPool.empty()   ? nullptr : (PatNode*)keep(uploadVec(recDrvPool));
    DRecScalarStop* d_recStops = recStopPool.empty() ? nullptr : (DRecScalarStop*)keep(uploadVec(recStopPool));

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
        // Shared whitelist (see deviceEmitterShapeCode) -- NEVER coerce an unknown shape to
        // a Quad here. cudaForwardSupported() already rejects such a scene, so reaching this
        // branch means the gate and the upload have drifted apart: say so loudly and emit a
        // dead emitter (zero power/area, so it is never selected) instead of feeding the
        // kernel a zeroed-basis Quad that could fault the display driver.
        int shapeCode = deviceEmitterShapeCode(e.shape);
        if (shapeCode < 0) {
            std::fprintf(stderr,
                "[cuda] INTERNAL: emitter shape %d has no device implementation; "
                "disabling this emitter (the GPU support gate should have prevented this)\n",
                (int)e.shape);
            shapeCode = 0;
            de.area = 0.0; de.power = 0.0;
        }
        de.shape = shapeCode;
        de.radius = e.radius;
        de.caps = e.caps ? 1 : 0;
        // Mesh area light: upload this emitter's triangle CDF to the device and point the
        // DEmitter at it (device pointer inside a POD later uploaded to device memory).
        // Every non-mesh emitter keeps a null pointer so nothing dereferences garbage.
        de.meshTris = nullptr; de.meshTriN = 0;
        if (e.shape == EmitterShape::Mesh && !e.meshTris.empty()) {
            std::vector<DEmitTri> dtris(e.meshTris.size());
            for (size_t i = 0; i < e.meshTris.size(); ++i) {
                const EmitTri& s = e.meshTris[i];
                dtris[i].v0  = {(Real)s.v0.x,  (Real)s.v0.y,  (Real)s.v0.z};
                dtris[i].e1  = {(Real)s.e1.x,  (Real)s.e1.y,  (Real)s.e1.z};
                dtris[i].e2  = {(Real)s.e2.x,  (Real)s.e2.y,  (Real)s.e2.z};
                dtris[i].nrm = {(Real)s.nrm.x, (Real)s.nrm.y, (Real)s.nrm.z};
                dtris[i].cumArea = s.cumArea;
                // Source-triangle UVs, so a sampled point reports the same (u,v) the
                // ray-hit path interpolates (an emission pattern reads both sides).
                dtris[i].uv0  = {(Real)s.uv0.x,  (Real)s.uv0.y,  (Real)s.uv0.z};
                dtris[i].uvE1 = {(Real)s.uvE1.x, (Real)s.uvE1.y, (Real)s.uvE1.z};
                dtris[i].uvE2 = {(Real)s.uvE2.x, (Real)s.uvE2.y, (Real)s.uvE2.z};
            }
            de.meshTris = (const DEmitTri*)keep(uploadVec(dtris));
            de.meshTriN = (int)dtris.size();
        }
        // Same fail-safe for the other half of the mesh contract: shape 5 with no triangles
        // would have the device sampler clamp to index -1 on a null pointer. The gate above
        // rejects such a scene, so this only fires if the two ever drift apart.
        if (de.shape == 5 && de.meshTriN == 0) {
            std::fprintf(stderr,
                "[cuda] INTERNAL: mesh emitter has no triangles; disabling it "
                "(the GPU support gate should have prevented this)\n");
            de.shape = 0; de.area = 0.0; de.power = 0.0;
        }
        de.spotCosInner = e.spotCosInner; de.spotCosOuter = e.spotCosOuter;
        de.spotOmega = e.spotOmega;
        de.cdfOffset = (int)cdfAll.size();
        de.cdfN = (int)e.spd.cdf.size();
        de.cdfStep = e.spd.step;
        de.matId = e.matId;                 // BDPT: link to emissive surface material
        de.emitPat = e.emitPat;             // `emit pattern:` profile over this emitter
        bakeSpec(e.spdFn, de.emitSpd);       // BDPT: baked emission SPD for Le(lambda)
        { Vec3 le = rgbbake::emitToRgb(e.spdFn); de.rgbEmit = {le.x, le.y, le.z}; }  // fast RGB backward
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
        // Bake the normalised illuminant over [DLMIN,DLMAX] for image-env NEE.
        std::vector<double> illumTab(SPEC_N);
        for (int i = 0; i < SPEC_N; ++i) {
            double wl = DLMIN + (double)i / (SPEC_N - 1) * (DLMAX - DLMIN);
            illumTab[i] = em.illumAt(wl);
        }
        denv.w = w; denv.h = h; denv.rot = em.rotOffset;
        denv.coeff = (double*)keep(uploadVec(coeffFlat));
        denv.scale = (double*)keep(uploadVec(em.scaleT));
        denv.illum = (double*)keep(uploadVec(illumTab));
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
    // Which textures are bound as tangent-space normal maps (C6)? Only those need the
    // full RGB direction uploaded (dTexNormalAt); everything else just needs coeff/gray.
    std::vector<char> usedAsNormal(scene.textures.size(), 0);
    for (const auto& m : scene.mats)
        if (m.normalTex >= 0 && m.normalTex < (int)scene.textures.size())
            usedAsNormal[m.normalTex] = 1;
    std::vector<DTexture> dtex;
    size_t txIdx = 0;
    // O7: the RGB -> Jakob-Hanika coefficient table is texture-independent, so upload it
    // once (lazily — only if some texture is actually stochastic) and share the pointer.
    const float* d_jhLut = nullptr;
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
        // Full linear RGB, only for normal-map textures (C6): raw [0,1] vector data.
        if (usedAsNormal[txIdx] && !tx.rgb.empty()) {
            std::vector<double> rgb3(tx.rgb.size() * 3);
            for (size_t i = 0; i < tx.rgb.size(); ++i) {
                rgb3[3 * i + 0] = tx.rgb[i].x;
                rgb3[3 * i + 1] = tx.rgb[i].y;
                rgb3[3 * i + 2] = tx.rgb[i].z;
            }
            dt.rgb = (double*)keep(uploadVec(rgb3));
        } else {
            dt.rgb = nullptr;
        }
        // O7: the Gaussianized plane sets, already built (and already float) on the host —
        // uploaded verbatim so the device transform IS the host transform, not a re-derived
        // one. Empty vectors upload as null and the samplers fall through to plain tiling.
        dt.stoch = tx.stoch;
        auto up1 = [&](const std::vector<float>& v) -> const float* {
            return v.empty() ? nullptr : (const float*)keep(uploadVec(v));
        };
        dt.gaussRgb   = up1(tx.gaussRgb);    dt.lutRgb   = up1(tx.lutRgb);
        dt.gaussGray  = up1(tx.gaussGray);   dt.lutGray  = up1(tx.lutGray);
        if (dt.gaussRgb && !d_jhLut) d_jhLut = up1(upsample::coeffLut());
        dt.jhLut = dt.gaussRgb ? d_jhLut : nullptr;
        dtex.push_back(dt);
        ++txIdx;
    }
    DTexture* d_tex     = dtex.empty()       ? nullptr : (DTexture*)keep(uploadVec(dtex));
    double*   d_fluoCdf = fluoCdfAll.empty() ? nullptr : (double*)keep(uploadVec(fluoCdfAll));

    DScene& sc = up.sc;
    sc.tris = d_tris; sc.nTris = (int)tris.size();
    sc.sph = d_sph;   sc.nSph = (int)sph.size();
    sc.mirrorPlanes = d_mirp; sc.nMirrorPlanes = (int)mirrorPlanes.size();
    sc.dielSph   = d_dielSph;   sc.nDielSph   = (int)scene.dielSphereIdx.size();
    sc.mirrorSph = d_mirrorSph; sc.nMirrorSph = (int)scene.mirrorSphereIdx.size();
    sc.mats = d_mats;
    sc.nodes = d_nodes; sc.primIdx = d_prim; sc.nNodes = (int)nodes.size();
    sc.fieldNodes = d_fnodes; sc.fieldExprNodes = d_fexpr;
    sc.fieldNodesF = d_fnodesF; sc.fieldExprNodesF = d_fexprF;
    sc.implicits = d_impl; sc.nImplicits = (int)dimpl.size();
    sc.curveSegs = d_curves; sc.nCurveSegs = (int)curveSegs.size();
    sc.instances = d_inst; sc.nInstances = (int)dinst.size();
    sc.blas = d_blas; sc.blasNodes = d_blasN; sc.blasPrim = d_blasP; sc.blasTris = d_blasT;
    sc.patNodes = d_pnodes; sc.patterns = d_pat; sc.nPatterns = (int)dpat.size();
    sc.recCoeff = d_recCoeff; sc.recDrivers = d_recDrv; sc.recScalarStops = d_recStops;
    sc.emitters = d_ems; sc.nEmitters = (int)dems.size();
    sc.emitCdf = d_emitCdf; sc.totalPower = scene.totalPower;
    // Light BVH: POD nodes referring to each other by index, so a straight memcpy. The
    // knobs come from lt:: (the same globals the CPU BackwardRenderer reads) so a scene
    // rendered on either backend makes the same selection decisions.
    sc.lightTree       = scene.lightTree.empty() ? nullptr
                                                 : (const LightTreeNode*)keep(uploadVec(scene.lightTree));
    sc.lightTreeRoot   = scene.lightTree.empty() ? -1 : scene.lightTreeRoot;
    sc.lightTreeAlways = scene.lightTreeAlways.empty() ? nullptr
                                                       : (const int*)keep(uploadVec(scene.lightTreeAlways));
    sc.nLightTreeAlways = (int)scene.lightTreeAlways.size();
    sc.bkLightTree     = lt::gEnabled ? 1 : 0;
    sc.bkLightSplit    = lt::gSplit;
    sc.bkLightSamples  = lt::gSamples;
    sc.lightCdfAll = d_cdfAll;
    sc.textures = d_tex; sc.nTex = (int)dtex.size();
    // N-D data tables upload VERBATIM — the host PatGrid / PatScatter headers refer to
    // their numbers by offset into the shared pool, never by pointer, so no fix-up is
    // needed and the device samplers are literally the same functions the host runs.
    // One pool serves both kinds, so this is one allocation however many tables.
    sc.grids     = scene.grids.empty()    ? nullptr : (const PatGrid*)keep(uploadVec(scene.grids));
    sc.nGrids    = (int)scene.grids.size();
    sc.scatters  = scene.scatters.empty() ? nullptr : (const PatScatter*)keep(uploadVec(scene.scatters));
    sc.nScatters = (int)scene.scatters.size();
    sc.dataPool  = scene.dataPool.empty() ? nullptr : (const float*)keep(uploadVec(scene.dataPool));
    sc.dataPoolN = (int)scene.dataPool.size();
    sc.fluoCdfAll = d_fluoCdf;
    sc.emitSamplerCdf = d_emitSamp;
    sc.emitSamplerN = (int)(emitSampCdf.empty() ? 0 : emitSampCdf.size() - 1);
    sc.emitSamplerStep = scene.emitSampler.step;
    sc.emitG = scene.emitG;
    // Participating media array (superposed). Each medium's density program is uploaded
    // separately; then the flat DMedium array is uploaded once. Empty => media=null, mediaN=0.
    {
        // Upload a host VdbGrid as a NATIVE SPARSE brick grid (ROADMAP C2) into a DVdbGrid.
        // Shared by the density field and the emissive-volume temperature field. An empty
        // grid leaves DVdbGrid inert (brickData=null, identity transform). `label` tags the
        // once-per-grid VRAM report.
        auto uploadVdbGrid = [&](const VdbGrid& g, DVdbGrid& out, const char* label) {
            const int B = 8;
            std::vector<int32_t> bidx; std::vector<uint16_t> bdata; int bx, by, bz;
            int nActive = g.buildBricks(B, bx, by, bz, bidx, bdata);
            out.brickIndex = (const int32_t*)keep(uploadVec(bidx));
            out.brickData  = (const uint16_t*)keep(uploadVec(bdata));
            out.bx = bx; out.by = by; out.bz = bz;
            out.brickB = B; out.brickShift = 3;   // 3 == log2(8)
            out.nx = g.nx; out.ny = g.ny; out.nz = g.nz;
            for (int k = 0; k < 9; ++k) out.ainv[k] = g.ainv[k];
            out.w0   = {g.w0.x, g.w0.y, g.w0.z};
            out.imin = {g.imin.x, g.imin.y, g.imin.z};
            size_t denseB  = (size_t)g.nx * g.ny * g.nz * sizeof(uint16_t);
            size_t sparseB = bdata.size() * sizeof(uint16_t) + bidx.size() * sizeof(int32_t);
            int nBricks = bx * by * bz;
            // The scene is re-uploaded on every progressive refresh; report each distinct
            // grid's sparse footprint only once to avoid log spam.
            static std::vector<const void*> reportedVdb;
            const void* vkey = (const void*)&g;
            bool seen = false;
            for (const void* p : reportedVdb) if (p == vkey) { seen = true; break; }
            if (!seen) {
                reportedVdb.push_back(vkey);
                std::fprintf(stderr,
                    "[vdb] sparse device grid (%s): %d/%d bricks active (%.1f%%), "
                    "%.1f MB -> %.1f MB VRAM (%.1fx)\n",
                    label, nActive, nBricks, nBricks ? 100.0 * nActive / nBricks : 0.0,
                    denseB / 1048576.0, sparseB / 1048576.0,
                    sparseB ? (double)denseB / sparseB : 1.0);
            }
        };
        auto clearVdbGrid = [](DVdbGrid& out) {
            out.brickIndex = nullptr; out.brickData = nullptr;
            out.bx = out.by = out.bz = 0;
            out.brickB = 0; out.brickShift = 0;
            out.nx = out.ny = out.nz = 0;
            for (int k = 0; k < 9; ++k) out.ainv[k] = (k % 4 == 0) ? 1.0 : 0.0;
            out.w0 = {0,0,0}; out.imin = {0,0,0};
        };
        std::vector<DMedium> dmeds(scene.media.size());
        for (size_t i = 0; i < scene.media.size(); ++i) {
            const Medium& m = scene.media[i];
            DMedium& dm = dmeds[i];
            dm.enabled = m.enabled ? 1 : 0;
            dm.g = m.g;
            bakeSpec(m.sigma_a, dm.sigma_a);
            bakeSpec(m.sigma_s, dm.sigma_s);
            dm.heterogeneous = m.heterogeneous() ? 1 : 0;
            dm.density  = m.density.empty() ? nullptr : (const PatNode*)keep(uploadVec(m.density));
            dm.densityN = (int)m.density.size();
            dm.densityMax = m.densityMax;
            // Imported .nvdb/.vdb volume: upload a NATIVE SPARSE brick grid (ROADMAP C2).
            if (m.vdb && !m.vdb->empty()) uploadVdbGrid(*m.vdb, dm.densGrid, "density");
            else                          clearVdbGrid(dm.densGrid);
            // Volumetric blackbody emission ("fire", ROADMAP C3): upload the temperature
            // field + emission params so the device genPhoton can birth fire photons.
            dm.emissive = (m.emissive() && m.temperature && !m.temperature->empty()) ? 1 : 0;
            if (dm.emissive) uploadVdbGrid(*m.temperature, dm.tempGrid, "temperature");
            else             clearVdbGrid(dm.tempGrid);
            dm.emitKelvin    = m.emitKelvin;
            dm.tempPeak      = m.tempPeak;
            dm.emissionScale = m.emissionScale;
            dm.bounded  = m.bounded ? 1 : 0;
            dm.boundShape = (m.boundShape == MediumBound::Sphere)   ? 1
                          : (m.boundShape == MediumBound::Implicit) ? 2
                          : (m.boundShape == MediumBound::Mesh)     ? 3 : 0;
            // Mesh containment: upload the baked occupancy lattice the same way as any
            // other volume, so CPU and GPU membership are the same trilinear test.
            if (m.boundShape == MediumBound::Mesh && m.boundGrid && !m.boundGrid->empty())
                uploadVdbGrid(*m.boundGrid, dm.boundGrid, "mesh bound");
            else
                clearVdbGrid(dm.boundGrid);
            dm.bmin = {m.bmin.x, m.bmin.y, m.bmin.z};
            dm.bmax = {m.bmax.x, m.bmax.y, m.bmax.z};
            dm.bcenter = {m.bcenter.x, m.bcenter.y, m.bcenter.z};
            dm.bradius = m.bradius;
            // Implicit bound: point at this medium's slice of the shared field pool.
            // exprOff is baked absolute into each node, so pass the whole expr pool.
            const MedFieldSlice& fs = medField[i];
            dm.boundField     = (fs.off >= 0 && d_fnodes) ? (d_fnodes + fs.off) : nullptr;
            dm.boundFieldN    = fs.n;
            dm.boundFieldExpr = d_fexpr;
            dm.boundInsideNeg = m.boundInsideNeg ? 1 : 0;
            // Gradient-index (GRIN) field: upload the compiled n(x,y,z) program + march step.
            dm.ior     = m.ior.empty() ? nullptr : (const PatNode*)keep(uploadVec(m.ior));
            dm.iorN    = (int)m.ior.size();
            dm.iorStep = m.iorStep;
            // Spectral rainbow (Airy droplet) phase: upload the (lambda x mu) pdf table +
            // per-lambda CDF so the device reproduces the bow instead of the HG lobe.
            if (m.rainbow() && m.rainbowPhase->built()) {
                const rainbow::RainbowPhase& rp = *m.rainbowPhase;
                dm.rbPdf  = (const double*)keep(uploadVec(rp.pdfTable()));
                dm.rbCdf  = (const double*)keep(uploadVec(rp.cdfTable()));
                dm.rbNLam = rp.nLam();
                dm.rbNMu  = rp.nMu();
                dm.rbLam0 = rp.lam0();
                dm.rbDLam = rp.dLam();
            } else {
                dm.rbPdf = nullptr; dm.rbCdf = nullptr;
                dm.rbNLam = dm.rbNMu = 0; dm.rbLam0 = 0.0; dm.rbDLam = 1.0;
            }
        }
        sc.media  = dmeds.empty() ? nullptr : (const DMedium*)keep(uploadVec(dmeds));
        sc.mediaN = (int)dmeds.size();
        sc.hasGrin = grin::sceneHasGrin(scene) ? 1 : 0;   // gate for dGrinMarch (host twin)
        // Emissive "fire" volumes (ROADMAP C3): upload the AABB/meanKe/power + the per-
        // volume Planck-at-emitKelvin wavelength CDF, and the total emission power for the
        // emitter-vs-fire birth split. Empty => emissiveVolumes=null, totalEmissionPower=0.
        std::vector<DEmissiveVolume> devs(scene.emissiveVolumes.size());
        for (size_t v = 0; v < scene.emissiveVolumes.size(); ++v) {
            const Scene::EmissiveVolume& ev = scene.emissiveVolumes[v];
            DEmissiveVolume& d = devs[v];
            d.mediumIndex = ev.mediumIndex;
            d.bmin = {ev.bmin.x, ev.bmin.y, ev.bmin.z};
            d.bmax = {ev.bmax.x, ev.bmax.y, ev.bmax.z};
            d.meanKe = ev.meanKe;
            d.power  = ev.power;
            d.lamCdf  = (const double*)keep(uploadVec(ev.lamSampler.cdf));
            d.lamN    = (int)ev.lamSampler.cdf.size() - 1;   // cdf has N+1 entries
            d.lamStep = ev.lamSampler.step;
        }
        sc.emissiveVolumes    = devs.empty() ? nullptr : (const DEmissiveVolume*)keep(uploadVec(devs));
        sc.emissiveVolN       = (int)devs.size();
        sc.totalEmissionPower = scene.totalEmissionPower;
    }
    sc.sensorOrigin = {scene.sensor.origin.x, scene.sensor.origin.y, scene.sensor.origin.z};
    sc.sensorUAxis  = {scene.sensor.uAxis.x,  scene.sensor.uAxis.y,  scene.sensor.uAxis.z};
    sc.sensorVAxis  = {scene.sensor.vAxis.x,  scene.sensor.vAxis.y,  scene.sensor.vAxis.z};
    sc.sceneCenter = {scene.sceneCenter.x, scene.sceneCenter.y, scene.sceneCenter.z};
    sc.sceneRadius = scene.sceneRadius;
    // `cavity` probe settings (O3 stage 2). The host loader has already resolved the
    // radius (explicit `cavity_radius`, else 2% of the scene AABB diagonal) and the
    // gate, so the device just mirrors the resolved values — the two backends must
    // agree on all three or CPU and GPU renders of the same scene diverge.
    sc.needsCavity   = scene.needsCavity ? 1 : 0;
    sc.cavityRadius  = scene.cavityRadius;
    sc.cavitySamples = scene.cavitySamples;
    sc.env = denv;
    sc.envIndex = scene.envIndex;
    sc.sunCount = scene.sunCount;      // >0 enables the direct-view solar-disc miss term
    // Fast RGB backward: constant-env radiance in linear sRGB (0 when there's no env).
    if (scene.envIndex >= 0 && scene.envIndex < (int)scene.emitters.size()) {
        Vec3 le = rgbbake::emitToRgb(scene.emitters[scene.envIndex].spdFn);
        sc.rgbEnv = {le.x, le.y, le.z};
    } else {
        sc.rgbEnv = {0.0, 0.0, 0.0};
    }
    // Scene-ignore render params (Stage 3). Defaults = full path tracing; the backward
    // wrappers (renderBackward[RGB]Cuda) override from the CLI flags before launch.
    sc.bkMaxBounce  = 32;
    sc.bkDirectOnly = 0;
    // Mode W knobs. Defaults = mode W off, so every stochastic mode is untouched.
    sc.bkWhitted   = 0;
    sc.bkGrid      = 4;
    sc.bkGiDirs    = 0;
    sc.bkGiGrid    = 1;
    sc.bkGiBounce  = 4;
    sc.bkGiClamp   = 0.0;
    sc.bkFwPerDist = 0.0;   // O8 stage 2: unknown footprint => fnoise() unfiltered
    // Split-at-dispersion is NOT a mode-W-only knob: `-herosplit` applies to plain mode R too
    // (see BackwardRenderer::heroSplit, which takes the same default), so default it from the
    // global rather than to 0. Before v0.111.0 the device could not split at all and GPU mode
    // R silently de-hero'd through `-herosplit`, disagreeing with the CPU; the deterministic
    // preview's split (bkRadianceHeroLoop<true>) is the same machinery, so honour it here.
    sc.bkHeroSplit = hero::gSplit ? 1 : 0;
    sc.bkAmbient   = 0.0;

    // One-time fill of the specular-sphere scan-angle cos/sin tables (device-computed
    // so table entries are bit-identical to the per-step evaluation they replace).
    kSphScanInit<<<1, SPH_SCAN_N + 1>>>();
    cudaCheckKernel("sphScanInit");
}

// Bake one Camera into a POD DCamera for the given film resolution. Any device memory
// (the realistic-lens index tables) is recorded in up.frees. Split out of the scene
// bake so a multi-camera shared pass can bake N cameras against one uploaded scene.
static gpu::DCamera bakeCamera(const Scene& /*scene*/, const Camera& cam, int resX, int resY, DUpload& up) {
    using namespace gpu;
    DCamera dc{};
    dc.eye = {cam.eye.x, cam.eye.y, cam.eye.z};
    dc.u = {cam.u.x, cam.u.y, cam.u.z};
    dc.v = {cam.v.x, cam.v.y, cam.v.z};
    dc.w = {cam.w.x, cam.w.y, cam.w.z};
    dc.tanHalfX = cam.tanHalfX; dc.tanHalfY = cam.tanHalfY;
    dc.resX = resX; dc.resY = resY;
    dc.apertureR = cam.apertureR; dc.filmDist = cam.filmDist; dc.lensF = cam.lensF;
    dc.projection = cam.projection; dc.halfFovY = cam.halfFovY; dc.rEdge = cam.rEdge;
    dc.frustumShiftX = cam.frustumShiftX;   // off-axis stereo shear

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
            dc.lens.iorAll = (const double*)up.keep(uploadVec(iorAll));
        }
    }
    return dc;
}

// Bake scene + one camera and upload them (the historical single-camera entry point).
// Fills up.dc for callers (renderBdptCuda / renderBackwardCuda) that key off it.
static void buildUpload(const Scene& scene, const Camera& cam, int resX, int resY, DUpload& up) {
    buildUploadScene(scene, up);
    up.dc = bakeCamera(scene, cam, resX, resY, up);
}

// Assemble a device DCamSet: upload the DCamera array plus the arrays of per-camera
// film / hits device pointers. All three device arrays are recorded in up.frees; the
// film/hits buffers themselves stay owned by the caller.
static gpu::DCamSet makeCamSet(DUpload& up, const std::vector<gpu::DCamera>& hcams,
                               const std::vector<double*>& films,
                               const std::vector<double*>& hits) {
    using namespace gpu;
    DCamSet cs{};
    cs.nCam  = (int)hcams.size();
    cs.cams  = (const DCamera*)up.keep(uploadVec(hcams));
    cs.films = (double* const*)up.keep(uploadVec(films));
    cs.hits  = (double* const*)up.keep(uploadVec(hits));
    return cs;
}

// Host driver for the wavefront backend. Allocates the SoA photon pool, seeds it, then
// runs extend/shade passes until every slot has drained the N-photon budget. Writes into
// the same d_film / d_hits / d_energy buffers as the megakernel path.
static void wavefrontTrace(DUpload& up, const gpu::DCamSet& cs, double* d_energy,
                           long long N, int diffraction, unsigned long long kseed,
                           int maxBounce, int camModeInt) {
    using namespace gpu;
    if (N <= 0) return;
    int W = (int)((N < (1 << 20)) ? N : (1 << 20));   // persistent slot count
    if (W < 1) W = 1;

    WFState st;
    CUDA_CHECK(cudaMalloc(&st.ro,     (size_t)W * sizeof(DVec3)));
    CUDA_CHECK(cudaMalloc(&st.rd,     (size_t)W * sizeof(DVec3)));
    CUDA_CHECK(cudaMalloc(&st.beta,   (size_t)W * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&st.lambda, (size_t)W * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&st.rng,    (size_t)W * sizeof(DRng)));
    CUDA_CHECK(cudaMalloc(&st.bounce, (size_t)W * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&st.alive,  (size_t)W * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&st.stkMat, (size_t)W * DMediumStack::CAP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&st.stkPri, (size_t)W * DMediumStack::CAP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&st.stkN,   (size_t)W * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&st.hit,    (size_t)W * sizeof(DHit)));
    CUDA_CHECK(cudaMemset(st.alive, 0, (size_t)W * sizeof(int)));

    unsigned long long* d_dispatched = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dispatched, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_dispatched, 0, sizeof(unsigned long long)));
    int* d_live = nullptr;
    CUDA_CHECK(cudaMalloc(&d_live, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_live, 0, sizeof(int)));

    int bs = 128;
    int gb = (W + bs - 1) / bs;

    kWfInit<<<gb, bs>>>(up.sc, cs, d_energy, st, N, W,
                        d_dispatched, d_live, kseed, camModeInt);
    cudaCheckKernel("wavefront-init");

    // Guard the pass loop against an unexpected non-terminating condition: the longest a
    // slot can stay busy is one path (<= maxBounce shades) before it must regenerate or
    // die, so once dispatched >= N every slot drains within maxBounce passes. This cap is
    // generous slack over that bound and never triggers in normal operation.
    long long maxPasses = (N / W + 2) * (long long)(maxBounce + 1) + 16;
    for (long long pass = 0; pass < maxPasses; ++pass) {
        kWfExtend<<<gb, bs>>>(up.sc, st, W);
        kWfShade<<<gb, bs>>>(up.sc, cs, d_energy, st, W, N,
                             diffraction, maxBounce, d_dispatched, d_live, camModeInt);
        cudaCheckKernel("wavefront-pass");
        int live = 0;
        CUDA_CHECK(cudaMemcpy(&live, d_live, sizeof(int), cudaMemcpyDeviceToHost));
        if (live <= 0) break;
    }

    cudaFree(st.ro); cudaFree(st.rd); cudaFree(st.beta); cudaFree(st.lambda);
    cudaFree(st.rng); cudaFree(st.bounce); cudaFree(st.alive);
    cudaFree(st.stkMat); cudaFree(st.stkPri); cudaFree(st.stkN); cudaFree(st.hit);
    cudaFree(d_dispatched); cudaFree(d_live);
}

// Launch the forward trace (megakernel or wavefront backend) over the baked scene `up`
// and camera set `cs`, accumulating into cs.films/cs.hits and d_energy. Shared by the
// single-camera and multi-camera drivers so the launch/seeding logic lives in one place.
static void launchForward(DUpload& up, const gpu::DCamSet& cs, double* d_energy,
                          long long N, bool diffraction, unsigned long long seedBase,
                          bool wavefront, int camModeInt, int heroC) {
    using namespace gpu;
    // seedBase==0 keeps the original single-shot seed exactly; each accumulation chunk
    // passes a distinct cumulative-photon offset for an independent stream.
    unsigned long long kseed = 0x9e3779b97f4a7c15ULL + seedBase * 0x9e3779b97f4a7c15ULL;
    // Hero-wavelength sampling shares one BVH walk across C stratified wavelengths, cutting
    // chromatic noise. It is only physical without participating media / GRIN bending (the
    // geometry must be wavelength-independent between dispersive events), and it lives ONLY
    // in the megakernel — so gate on the scene and force the megakernel when hero is active.
    int effHeroC = 1;
    if (heroC > 1 && up.sc.mediaN == 0 && !up.sc.hasGrin) {
        effHeroC = (heroC > hero::kHeroMax) ? hero::kHeroMax : heroC;
    }
    if (wavefront && effHeroC == 1 && !cs.beamGather) {
        // Streaming backend: identical physics, path-regeneration scheduling. Same
        // maxBounce (32) and camera mode/set as the megakernel. (Hero AND photon-beams
        // force the megakernel — the wavefront pool carries no per-photon beam RNG stream.)
        wavefrontTrace(up, cs, d_energy, N, diffraction ? 1 : 0, kseed, 32, camModeInt);
    } else {
        int blockSize = 128;
        int numBlocks = 2048;          // ~262k threads, grid-stride over N photons
        kTrace<<<numBlocks, blockSize>>>(up.sc, cs, d_energy, N, diffraction ? 1 : 0,
                                         kseed, 32, camModeInt, effHeroC);
    }
    cudaCheckKernel("forward");
}

Film renderForwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                       long long N, EnergyReport& eOut, bool diffraction,
                       char camMode, unsigned long long seedBase, bool wavefront, int heroC) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    if (!cudaAvailable() || !cudaForwardSupported(scene)) return out;

    DUpload up;
    buildUpload(scene, cam, resX, resY, up);

    const size_t npix = (size_t)resX * resY;
    double* d_film = nullptr;   CUDA_CHECK(cudaMalloc(&d_film, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_film, 0, npix * 3 * sizeof(double)));
    double* d_hits = nullptr;   CUDA_CHECK(cudaMalloc(&d_hits, npix * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_hits, 0, npix * sizeof(double)));
    double* d_energy = nullptr; CUDA_CHECK(cudaMalloc(&d_energy, 5 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_energy, 0, 5 * sizeof(double)));

    int camModeInt = (camMode == 'A') ? CAM_A : (camMode == 'C') ? CAM_C : CAM_B;

    // One-camera DCamSet: the multi-camera code path with nCam==1 (bit-identical to the
    // old single-camera launch — connect draws no RNG and the loop runs exactly once).
    std::vector<DCamera> hc{ up.dc };
    std::vector<double*> fp{ d_film }, hp{ d_hits };
    DCamSet cs = makeCamSet(up, hc, fp, hp);
    launchForward(up, cs, d_energy, N, diffraction, seedBase, wavefront, camModeInt, heroC);

    // --- download ---
    std::vector<double> film(npix * 3);
    CUDA_CHECK(cudaMemcpy(film.data(), d_film, film.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.hits.data(), d_hits, npix * sizeof(double), cudaMemcpyDeviceToHost));
    double energy[5] = {0,0,0,0,0};
    CUDA_CHECK(cudaMemcpy(energy, d_energy, 5 * sizeof(double), cudaMemcpyDeviceToHost));
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

// ---- Resident shared forward session (see render_cuda.h) --------------------
// Everything a progressive shared pass needs to keep on the device across batches:
// the baked scene, the baked cameras, and the per-camera film/hits accumulators.
// batch() is then a bare kernel launch; the full bake/upload/alloc/download/free
// round trip is paid once per RENDER instead of once per ~2M-photon batch.
struct SharedGpuSession {
    DUpload up;
    std::vector<gpu::DCamera> hcams;
    std::vector<double*> d_films, d_hits;   // resident per-camera accumulators
    std::vector<size_t>  npix;
    double* d_energy = nullptr;             // resident 5-counter energy tally
    gpu::DCamSet cs{};
    int  camModeInt = 0;
    bool wavefront = false;
    int  heroC = 1;
    int  nc = 0;
};

SharedGpuSession* sharedForwardGpuBegin(const Scene& scene,
                                        const std::vector<Camera>& cams,
                                        const std::vector<int>& resX,
                                        const std::vector<int>& resY,
                                        char camMode, bool wavefront, int heroC,
                                        bool beamGather,
                                        const std::vector<Film>* seedFilms,
                                        const EnergyReport* seedEnergy) {
    using namespace gpu;
    int nc = (int)cams.size();
    if (nc == 0 || !cudaAvailable() || !cudaForwardSupported(scene)) return nullptr;

    auto* s = new SharedGpuSession;
    s->nc = nc; s->wavefront = wavefront; s->heroC = heroC;
    s->camModeInt = (camMode == 'A') ? CAM_A : CAM_B;   // shared pass never runs mode C

    // Bake the scene ONCE, then bake every camera against it (the win: one photon set,
    // splat to all cameras — and with the session, ONE bake for the whole render).
    buildUploadScene(scene, s->up);
    s->hcams.resize(nc);
    for (int c = 0; c < nc; ++c) s->hcams[c] = bakeCamera(scene, cams[c], resX[c], resY[c], s->up);

    // Per-camera film / hits device accumulators (each camera keeps its own resolution).
    // On -resume the checkpoint films seed them, so the device always holds the full
    // running totals and download() is a plain replace on the host side.
    s->d_films.assign(nc, nullptr); s->d_hits.assign(nc, nullptr); s->npix.resize(nc);
    for (int c = 0; c < nc; ++c) {
        s->npix[c] = (size_t)resX[c] * resY[c];
        CUDA_CHECK(cudaMalloc(&s->d_films[c], s->npix[c] * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s->d_hits[c],  s->npix[c] * sizeof(double)));
        if (seedFilms && c < (int)seedFilms->size() && (*seedFilms)[c].xyz.size() == s->npix[c]) {
            const Film& sf = (*seedFilms)[c];
            std::vector<double> f(s->npix[c] * 3);
            for (size_t i = 0; i < s->npix[c]; ++i) {
                f[i * 3 + 0] = sf.xyz[i].x; f[i * 3 + 1] = sf.xyz[i].y; f[i * 3 + 2] = sf.xyz[i].z;
            }
            CUDA_CHECK(cudaMemcpy(s->d_films[c], f.data(), f.size() * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(s->d_hits[c], sf.hits.data(), s->npix[c] * sizeof(double), cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemset(s->d_films[c], 0, s->npix[c] * 3 * sizeof(double)));
            CUDA_CHECK(cudaMemset(s->d_hits[c],  0, s->npix[c] * sizeof(double)));
        }
    }
    CUDA_CHECK(cudaMalloc(&s->d_energy, 5 * sizeof(double)));
    double e0[5] = {0, 0, 0, 0, 0};
    if (seedEnergy) {
        e0[0] = seedEnergy->emitted; e0[1] = seedEnergy->absorbed; e0[2] = seedEnergy->sensor;
        e0[3] = seedEnergy->escaped; e0[4] = seedEnergy->residual;
    }
    CUDA_CHECK(cudaMemcpy(s->d_energy, e0, sizeof e0, cudaMemcpyHostToDevice));

    s->cs = makeCamSet(s->up, s->hcams, s->d_films, s->d_hits);
    // Photon-beams gather: only meaningful with several cameras sharing one flight and a
    // participating medium present (the device gates the per-step branch on the same).
    s->cs.beamGather = beamGather && nc > 1 && !scene.media.empty();
    return s;
}

void sharedForwardGpuBatch(SharedGpuSession* s, long long N,
                           unsigned long long seedBase, bool diffraction) {
    if (!s || N <= 0) return;
    launchForward(s->up, s->cs, s->d_energy, N, diffraction, seedBase,
                  s->wavefront, s->camModeInt, s->heroC);
}

void sharedForwardGpuHits0(SharedGpuSession* s, std::vector<double>& hits) {
    if (!s || hits.size() < s->npix[0]) return;
    CUDA_CHECK(cudaMemcpy(hits.data(), s->d_hits[0], s->npix[0] * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void sharedForwardGpuDownload(SharedGpuSession* s, std::vector<Film>& films,
                              EnergyReport& e) {
    if (!s) return;
    for (int c = 0; c < s->nc && c < (int)films.size(); ++c) {
        std::vector<double> f(s->npix[c] * 3);
        CUDA_CHECK(cudaMemcpy(f.data(), s->d_films[c], f.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(films[c].hits.data(), s->d_hits[c], s->npix[c] * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < s->npix[c]; ++i)
            films[c].xyz[i] = Vec3(f[i * 3 + 0], f[i * 3 + 1], f[i * 3 + 2]);
    }
    // The photon trace is shared, so energy is counted once for the whole pass. These
    // are running TOTALS (the seed energy was uploaded at begin), so REPLACE.
    double energy[5] = {0, 0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpy(energy, s->d_energy, sizeof energy, cudaMemcpyDeviceToHost));
    e.emitted = energy[0]; e.absorbed = energy[1]; e.sensor = energy[2];
    e.escaped = energy[3]; e.residual = energy[4];
}

void sharedForwardGpuEnd(SharedGpuSession* s) {
    if (!s) return;
    freeUpload(s->up);
    for (int c = 0; c < s->nc; ++c) { cudaFree(s->d_films[c]); cudaFree(s->d_hits[c]); }
    cudaFree(s->d_energy);
    delete s;
}

std::vector<Film> renderForwardSharedCuda(const Scene& scene,
                                          const std::vector<Camera>& cams,
                                          const std::vector<int>& resX,
                                          const std::vector<int>& resY,
                                          long long N, EnergyReport& eOut, bool diffraction,
                                          char camMode, unsigned long long seedBase,
                                          bool wavefront, int heroC, bool beamGather) {
    // One-shot wrapper over the session API (bit-compatible with the historical
    // single-call behaviour: same bake, same launch, same seeding, energy ADDED to eOut).
    int nc = (int)cams.size();
    std::vector<Film> out(nc);
    for (int c = 0; c < nc; ++c) { out[c].resX = resX[c]; out[c].resY = resY[c]; out[c].alloc(); }
    SharedGpuSession* s = sharedForwardGpuBegin(scene, cams, resX, resY, camMode,
                                               wavefront, heroC, beamGather);
    if (!s) return out;
    sharedForwardGpuBatch(s, N, seedBase, diffraction);
    EnergyReport e;
    sharedForwardGpuDownload(s, out, e);
    eOut.emitted += e.emitted; eOut.absorbed += e.absorbed; eOut.sensor += e.sensor;
    eOut.escaped += e.escaped; eOut.residual += e.residual;
    sharedForwardGpuEnd(s);
    return out;
}

// ------------------------------ BDPT (mode D) host ---------------------------

bool cudaBdptSupported(const Scene& scene) {
    // BDPT-GPU needs the same POD-bakeable materials as the forward path, PLUS the
    // BDPT scope restrictions (bdpt.h / mode-D guard in main.cpp): participating media
    // — homogeneous AND heterogeneous (density-field / bounded) — are supported (device
    // random walk places medium vertices by delta tracking and weights connections by
    // ratio-tracking transmittance, matching the CPU BDPT), and only area/sphere/cylinder
    // Lambertian emitters (no spot/env/collimated).
    if (!cudaForwardSupported(scene)) return false;
    // Hair runs on the device forward/backward tracers but NOT in the BDPT vertex
    // machinery (see sceneUsesHairMaterial) — a hair scene falls back to the CPU BDPT.
    if (sceneUsesHairMaterial(scene)) return false;
    // M9: the GPU BDPT kernel now threads the per-hit texcoords (DVertex.u/v -> DHit)
    // through dBsdfF / dBsdfPdf / dRandomWalk / dConnect, so per-hit-driven throughput
    // slots evaluate consistently in the sampler AND the pdf/eval — MIS-safe. Enabled:
    // textured/patterned/record diffuse albedo & glossy reflect, per-hit glossy roughness
    // and thin-film thickness maps, mix blend masks, and Beer-Lambert colored-glass
    // interior absorption (delta vertex — throughput only).
    //
    // FROSTED (rough) dielectric is now on-device too (M9): the device refractOrReflect
    // jitters the chosen reflect/refract lobe by the per-hit roughness (dMatRoughness,
    // texture/pattern/constant), keeping it on the intended side — bit-for-bit the same
    // "stochastic-delta" model as the CPU bdpt.h (a rough dielectric stays a non-connectable
    // delta vertex; only its scattered direction is jittered). dDielectricStep in the BDPT
    // random walk already routes through it, so no separate gate is needed.
    // Diffuse-transmission (two-sided Lambertian) is likewise on-device (M9): dRandomWalk
    // samples the two lobes, dBsdfF/dBsdfPdf evaluate them, and dConnectBDPT allows
    // back-hemisphere connections (|cos| G, shadow-terminator skip).
    //
    // No per-MATERIAL reject remains here. Env and collimated lights are outside BDPT's
    // transport scope on BOTH backends: main.cpp's bdptUnsupportedFeature() flags them at
    // the mode-D guard, which REFUSES the render (or demotes D->B with -on-unsupported
    // fallback) before any BDPT dispatch, so such a scene never reaches the BDPT path (CPU
    // or GPU) at all. They are re-checked below purely as belt-and-suspenders.
    //
    // SPOT and SUN lights now render on the device too (0.126.0), closing what used to be a
    // GPU-vs-CPU gap: dGenLightSubpath grows the delta-emission branch (point origin / sun
    // disc, cone direction sampling, path[0].pdfFwd = 0 and the infinite-light path[1]
    // planar-density patch), dConnectBDPT's s=1 NEE the unified per-shape `Wgeom`,
    // dVertexPdfLight / dVertexPdfLightOrigin / dMisWeight the delta-aware densities and
    // s=0 exclusion, and kBdptT the escaped-ray solar disc — all direct ports of bdpt.h.
    for (const auto& em : scene.emitters)
        if (em.shape == EmitterShape::Env || em.collimated)
            return false;
    // Gradient-index (GRIN) media bend rays along curved paths; BDPT's connection geometry,
    // area-measure pdf conversion and MIS weights all assume STRAIGHT connecting segments, so
    // a GRIN region would bias the estimator. The mode-D guard (bdptUnsupportedFeature) already
    // refuses GRIN before dispatch; reject here too so GPU BDPT can never render it straight.
    if (grin::sceneHasGrin(scene)) return false;
    // Spectral rainbow phase now runs on the device BDPT too (M10): dPhaseF/dPhasePdf and
    // the random-walk medium scatter dispatch through dMedPhase / dMedPhaseSample, which
    // read the uploaded (lambda x mu) Airy table for rainbow media. No fallback needed.
    return true;
}

// Drives a chunked samples-per-pixel render for the GPU reference/BDPT paths. Repeatedly
// renders `chunkSpp` more samples (via `launch(chunkSpp, sampleBase)`, which does the
// kernel launch + cudaCheckKernel accumulating into the resident device buffers) and
// downloads the running SUM film (via `download(out)`), reporting to `prog` after each
// chunk. Stops when `prog.report` returns true or the requested `spp` is reached. Chunk
// size adapts toward ~0.15 s of GPU work per launch so a wall-clock budget or Ctrl-C is
// honoured promptly without paying per-launch overhead on fast scenes.
//
// The base handed to `launch` is the ABSOLUTE sample index, i.e. biased past whatever a
// `-resume` checkpoint already holds (`prog.sampleBase`) — exactly what cpuSppChunks does
// and what kBackward's `sIdx` is documented to receive. Without the bias a resumed run
// re-rendered sample indices [0, c) on top of a film that already contained them, which
// for the DETERMINISTIC mode W is not "an independent realization" but a straightforwardly
// wrong image: its lattice is indexed by absolute sample, so `3 spp + resume 5` produced
// samples {0,1,2} ∪ {0..4} instead of {0..7} — measurably different from a plain 8 spp
// (max channel difference 163/255 on the loom jumping_jack frame that found this).
template <class LaunchFn, class DownloadFn>
static void gpuSppChunks(long long spp, const SppProgress& prog, Film& out,
                         LaunchFn&& launch, DownloadFn&& download) {
    using clk = std::chrono::steady_clock;
    // Same two diagnostic levers cpuSppChunks has, and for the same reason: when a render
    // appears to hang, the first question is always "how big did the chunk get, and how long
    // is one taking?" — and the adaptive answer is invisible from outside.
    //   FTRACE_CHUNK_SPP=K   pin every chunk to K spp (0/unset = adapt)
    //   FTRACE_CHUNK_DEBUG=1 log each chunk's size and wall time
    // The GPU chunker went without these, which is exactly why a 320x180 4-spp frame that
    // would not finish in 7 minutes could not be told apart from a deadlock without a debugger.
    long long forcedChunk = 0;
    if (const char* e = std::getenv("FTRACE_CHUNK_SPP")) forcedChunk = std::atoll(e);
    const bool chunkDebug = [] { const char* e = std::getenv("FTRACE_CHUNK_DEBUG");
                                 return e && *e && std::strcmp(e, "0"); }();
    long long done = 0, chunk = (forcedChunk > 0) ? forcedChunk : 1;

    // ---- stall watchdog ------------------------------------------------------------
    // A kernel launch is synchronous from here (cudaCheckKernel synchronizes), so while one
    // chunk is in flight this thread is parked in cuCtxSynchronize and CANNOT report, write
    // the -interval image, honour -time, or answer -stop. Normally that is fine: chunks are
    // retargeted to ~0.15 s. But if ANOTHER process on the machine is saturating the GPU,
    // even the minimum 1-spp chunk can take minutes-to-hours, and the render becomes
    // indistinguishable from a deadlock — silent, unwritable and unkillable except by
    // taskkill. (Measured: a 160x90 2-spp frame did not complete a single chunk in minutes
    // while an unrelated CUDA process held ~20 GB and 100% of an RTX 4090. Note that
    // cudaMemGetInfo is NO help here — under WDDM it cheerfully reports the memory as
    // available, because the driver will page rather than fail.)
    //
    // So watch the in-flight launch from a second thread and SAY SO. It cannot make the
    // kernel come back sooner, but it converts an apparent hang into an accurate diagnosis,
    // which is the whole difference between "ftrace is broken" and "something else is using
    // your GPU".
    std::mutex wmx;
    std::condition_variable wcv;
    bool watchQuit = false;
    std::atomic<long long> launchStartMs{0};    // 0 = nothing in flight
    std::atomic<long long> launchSpp{0};
    auto nowMs = [] {
        return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                   clk::now().time_since_epoch()).count();
    };
    std::thread watchdog([&] {
        bool warned = false;
        double nextWarn = 30.0;                 // first complaint after 30 s in one chunk
        for (;;) {
            std::unique_lock<std::mutex> lk(wmx);
            wcv.wait_for(lk, std::chrono::seconds(2), [&] { return watchQuit; });
            if (watchQuit) return;
            lk.unlock();
            const long long t0 = launchStartMs.load(std::memory_order_relaxed);
            if (!t0) { warned = false; nextWarn = 30.0; continue; }   // between chunks: reset
            const double el = (nowMs() - t0) / 1000.0;
            if (el < nextWarn) continue;
            if (!warned) {
                std::fprintf(stderr,
                    "\n[gpu-stall] one %lld-spp chunk has been running on the GPU for %.0f s "
                    "(target is 0.15 s).\n"
                    "[gpu-stall] The render is NOT hung, but until this chunk returns it "
                    "cannot write the -interval image, end on -time, or answer -stop.\n"
                    "[gpu-stall] The usual cause is another process saturating the GPU: check "
                    "`nvidia-smi` for a second CUDA program. Run with -device cpu to sidestep "
                    "it, or free the card.\n",
                    launchSpp.load(std::memory_order_relaxed), el);
                warned = true;
            } else {
                std::fprintf(stderr, "[gpu-stall] still in the same chunk after %.0f s.\n", el);
            }
            std::fflush(stderr);
            nextWarn = el + 60.0;               // then once a minute, so it stays visible
        }
    });
    struct WatchdogJoin {                        // join on every exit path, including a throw
        std::mutex& m; std::condition_variable& cv; bool& q; std::thread& t;
        ~WatchdogJoin() { { std::lock_guard<std::mutex> lk(m); q = true; } cv.notify_all();
                          if (t.joinable()) t.join(); }
    } wjoin{wmx, wcv, watchQuit, watchdog};

    while (done < spp) {
        long long c = chunk; if (c > spp - done) c = spp - done;
        auto t0 = clk::now();
        launchSpp.store(c, std::memory_order_relaxed);
        launchStartMs.store(nowMs(), std::memory_order_relaxed);
        launch(c, prog.sampleBase + done);
        launchStartMs.store(0, std::memory_order_relaxed);
        done += c;
        double dt = std::chrono::duration<double>(clk::now() - t0).count();
        if (chunkDebug)
            std::fprintf(stderr, "[chunk] gpu %lld spp in %.3f s (%.1f spp/s), %lld/%lld done\n",
                         c, dt, dt > 0 ? c / dt : 0.0, done, spp);
        if (forcedChunk <= 0 && dt > 1e-4) {               // retarget ~0.15 s per chunk
            long long next = (long long)((double)c * (0.15 / dt));
            if (next < 1)          next = 1;
            if (next > c * 8 + 1)  next = c * 8 + 1;        // ramp up, but not explosively
            chunk = next;
        }
        download(out);
        if (prog.report(out, done, done >= spp)) break;
    }
}

Film renderBdptCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                    long long spp, int maxDepth, bool diffraction, const SppProgress* prog,
                    int heroC) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    if (!cudaAvailable() || !cudaBdptSupported(scene)) return out;
    // Two kernel variants: the default shallow stack, and a deep one for scenes that need
    // it. Anything past BDPT_DEEPDEPTH still clamps — going deeper would need another
    // instantiation, and the local-memory footprint is already ~13 KB/thread there.
    const bool deep = (maxDepth > BDPT_MAXDEPTH);
    if (maxDepth > BDPT_DEEPDEPTH) maxDepth = BDPT_DEEPDEPTH;   // device array bound

    DUpload up;
    buildUpload(scene, cam, resX, resY, up);

    // Hero-wavelength gate — the same one the CPU BDPT applies (bdpt.h BdptRenderer::
    // renderRows): a participating medium makes the shadow-ray transmittance wavelength-
    // dependent, GRIN bends each λ differently, and a physical lens disperses the primary
    // ray, so all three fall back to the single-λ kernel. (GRIN never reaches here at all —
    // cudaBdptSupported rejects it outright.)
    int C = heroC;
    if (C > hero::kHeroMax) C = hero::kHeroMax;
    if (C < 1) C = 1;
    const bool useHero = (C > 1) && up.sc.mediaN == 0 && !up.sc.hasGrin && !cam.hasLens();

    const size_t npix = (size_t)resX * resY;
    double* d_cam   = nullptr; CUDA_CHECK(cudaMalloc(&d_cam,   npix * 3 * sizeof(double)));
    double* d_splat = nullptr; CUDA_CHECK(cudaMalloc(&d_splat, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cam,   0, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_splat, 0, npix * 3 * sizeof(double)));
    // Resume (mode D disk resume): fresh samples carry ABSOLUTE indices
    // [sppBase, sppBase + spp), so the kernel's stride is the FINAL total (see
    // renderBackwardCuda for why). The loaded sample count is also mixed into the seed
    // base so the continued samples stay decorrelated from the ones already in the
    // checkpoint film — mode D is Monte-Carlo throughout, so there is no deterministic
    // lattice to continue exactly, only variance to keep independent.
    const long long sppBase  = prog ? (long long)prog->sampleBase : 0;
    const long long sppTotal = sppBase + spp;
    const unsigned long long seed = 0x9e3779b97f4a7c15ULL
        ^ (prog ? (unsigned long long)prog->sampleBase * 0x9E3779B97F4A7C15ULL : 0ULL);

    std::vector<double> camH(npix * 3), splatH(npix * 3);
    auto download = [&](Film& o) {
        CUDA_CHECK(cudaMemcpy(camH.data(),   d_cam,   npix * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(splatH.data(), d_splat, npix * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < npix; ++i)   // SUM (cam+splat); writeFilm divides by spp
            o.xyz[i] = Vec3(camH[3 * i + 0] + splatH[3 * i + 0],
                            camH[3 * i + 1] + splatH[3 * i + 1],
                            camH[3 * i + 2] + splatH[3 * i + 2]);
    };
    auto launch = [&](long long c, long long base) {
        long long totalSamples = (long long)npix * c;
        if (useHero && deep)
            kBdptT<BDPT_NSEC, BDPT_DEEPDEPTH><<<2048, 128>>>(up.sc, up.dc, d_cam, d_splat, totalSamples, c, sppTotal, base,
                                                             resX, maxDepth, diffraction ? 1 : 0, seed, C);
        else if (useHero)
            kBdptT<BDPT_NSEC, BDPT_MAXDEPTH><<<2048, 128>>>(up.sc, up.dc, d_cam, d_splat, totalSamples, c, sppTotal, base,
                                                            resX, maxDepth, diffraction ? 1 : 0, seed, C);
        else if (deep)
            kBdptT<0, BDPT_DEEPDEPTH><<<2048, 128>>>(up.sc, up.dc, d_cam, d_splat, totalSamples, c, sppTotal, base,
                                                     resX, maxDepth, diffraction ? 1 : 0, seed, 1);
        else
            kBdptT<0, BDPT_MAXDEPTH><<<2048, 128>>>(up.sc, up.dc, d_cam, d_splat, totalSamples, c, sppTotal, base,
                                                    resX, maxDepth, diffraction ? 1 : 0, seed, 1);
        cudaCheckKernel("bdpt");
    };

    if (!prog || !prog->report) { launch(spp, sppBase); download(out); }   // single-shot
    else gpuSppChunks(spp, *prog, out, launch, download);

    freeUpload(up);
    cudaFree(d_cam); cudaFree(d_splat);
    return out;
}

// ------------------------- device-memory preflight ---------------------------
//
// WHY THIS EXISTS: a megakernel's per-thread LOCAL memory is real device memory. The BDPT
// kernel carries its path vertices in thread-local arrays, so the fixed 2048 x 128 grid
// every megakernel here launches with (MEGAKERNEL_THREADS below) reserves
// `localSizeBytes * 262144` bytes of backing store before it traces a single sample — on
// this project that is GIGABYTES, not megabytes.
//
// When another process has filled the card, that allocation does NOT fail. The WDDM driver
// backs the spill with host memory across PCIe and the kernel simply CRAWLS — measured at
// 2-3 orders of magnitude slower, while nvidia-smi still reports 100% GPU utilisation, so
// from the outside it is indistinguishable from a hang. Worse, it disables the render's
// whole control surface: `-interval` writes, the `-time` budget and `-stop` are all polled
// BETWEEN chunks (gpuSppChunks), and a chunk that should take 0.15 s never returns.
//
// Measured case that motivated this: a 320x180 4-spp gallery_rain frame — normally ~2 s —
// had not finished after 7 minutes, with an unrelated process holding 20 GB of the card's
// 24 GB. Nothing in the renderer said a word.
//
// So: ask the driver what the kernel actually reserves, ask the card what is actually free,
// and let the caller refuse the GPU with a real explanation instead of thrashing silently.
#define MEGAKERNEL_THREADS ((size_t)2048 * 128)

bool cudaMemInfo(size_t* freeB, size_t* totalB) {
    if (!cudaAvailable()) return false;
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return false;
    if (freeB)  *freeB  = f;
    if (totalB) *totalB = t;
    return true;
}

// Bytes of device memory the megakernel for `mode` reserves purely for per-thread local
// storage, at the instantiation this render will actually launch. `maxDepth`/`heroC` pick
// the same four-way BDPT branch renderBdptCuda picks, so the number is the one that will
// really be reserved rather than a worst case. Returns 0 when unknown (no device, or a
// mode with no megakernel), which callers must treat as "no opinion", not as "free".
size_t cudaMegakernelLocalBytes(char mode, int maxDepth, int heroC) {
    using namespace gpu;                      // the megakernels live there
    if (!cudaAvailable()) return 0;
    const void* fn = nullptr;
    if (mode == 'D') {
        // Mirrors renderBdptCuda's dispatch exactly (deep = maxDepth past the default
        // instantiation; hero = the multi-wavelength bundle). The deep variant is the
        // expensive one: BDPT_DEEPDEPTH is 64 vertices of local path state per thread.
        const bool deep = (maxDepth > BDPT_MAXDEPTH);
        const bool hero = (heroC > 1);
        if      (hero && deep) fn = (const void*)kBdptT<BDPT_NSEC, BDPT_DEEPDEPTH>;
        else if (hero)         fn = (const void*)kBdptT<BDPT_NSEC, BDPT_MAXDEPTH>;
        else if (deep)         fn = (const void*)kBdptT<0, BDPT_DEEPDEPTH>;
        else                   fn = (const void*)kBdptT<0, BDPT_MAXDEPTH>;
    } else if (mode == 'R' || mode == 'W') {
        fn = (const void*)kBackward;
    } else if (mode == 'P') {
        fn = (const void*)kBackwardRGB;
    }
    if (!fn) return 0;
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, fn) != cudaSuccess) return 0;
    cudaGetLastError();                       // don't leave a sticky error for the next launch
    return (size_t)fa.localSizeBytes * MEGAKERNEL_THREADS;
}

// --------------------- backward reference (mode R) host ----------------------

bool cudaBackwardSupported(const Scene& scene, const Camera& cam) {
    // GPU mode R needs the same POD-bakeable materials as the forward path, plus the
    // current backward scope: participating media, fluorescence AND a constant
    // environment light ARE supported (media: homogeneous + heterogeneous + spectral-
    // rainbow phase, minus GRIN; fluorescence: the bispectral Stokes-shift adjoint; env: bkNeeEnv /
    // bkNeeEnvVolume + MIS'd env-miss, constant only — an IMAGE env stays on the CPU).
    // Emitters: area/sphere/cylinder Lambertian AND point-spot lights (bkNeeLight /
    // bkNeeVolume spot branch); only collimated beams (not NEE-samplable) fall back to
    // the CPU. Textured albedo IS supported (dDiffuseRho ports it). dInvPdfLambda is
    // exact (geomWeight = area*PI for area emitters, 4pi^2 R^2 for the env emitter).
    if (!cudaForwardSupported(scene)) return false;
    // Participating media now run on the device backward walk (dMediaSampleCollision /
    // dMediaTransmittance — homogeneous AND heterogeneous), INCLUDING spectral-rainbow
    // phase (M10: bkNeeVolume / bkNeeEnvVolume / the volume scatter dispatch through
    // dMedPhase / dMedPhaseSample read the uploaded Airy table) AND gradient-index (GRIN)
    // media (M11: bkRadiance runs the shared Eikonal marcher dGrinMarch as a per-bounce
    // pre-closestHit pass — the exact device twin of grin::march, so CPU and GPU mode R
    // bend rays identically). Mode-D BDPT still keeps GRIN on the CPU (area-measure MIS
    // assumes straight segments), gated separately in cudaBdptSupported.
    // Environment light IS supported on the device backward walk: a CONSTANT env
    // (bkNeeEnv / bkNeeEnvVolume / env-miss with MIS) and now an IMAGE env (lat-long
    // map — importance-sampled via dEnvSample, evaluated via dEnvRadiance/dEnvPdf, with
    // the baked illuminant table). Neither forces a CPU fallback (M1).
    for (size_t i = 0; i < scene.emitters.size(); ++i) {
        if ((int)i == scene.envIndex) continue;    // constant env: handled by bkNeeEnv, not area NEE
        const auto& em = scene.emitters[i];
        if (em.collimated) return false;                    // collimated beams: not NEE-samplable, CPU only
        if (em.shape == EmitterShape::Env) return false;    // stray (non-env) env-shape emitter: CPU
        // Spot lights ARE supported now (bkNeeLight / bkNeeVolume point-spot branch).
    }
    // A physical lens deeper than the device cap falls back to the CPU tracer.
    if (cam.hasLens() && (int)cam.lens->surf.size() > D_MAXLENS) return false;
    return true;
}

bool cudaBackwardWhittedSupported(const Scene& scene, const Camera& cam,
                                  const WhittedOpts& w) {
    // Mode W rides the SAME device megakernel as mode R (kBackward), swapping only the
    // estimators, so it inherits mode R's whole scope first.
    if (!cudaBackwardSupported(scene, cam)) return false;
    // NOTHING mode-W-specific gates any more, as of v0.116.0 (§N/N3c).
    //
    // Dispersive materials stopped gating in v0.111.0 (N3b): bkRadianceHeroLoop<true, ...> fans
    // the hero bundle into monochromatic sub-paths on the device exactly as the CPU does, which
    // is what mode W needs — its λ lattice is shared by every pixel, so a de-hero would collapse
    // the WHOLE FRAME onto one wavelength (36.7 pp of chroma error, measured).
    //
    // `-gi <n>`, the deterministic one-bounce gather, was the last one: bkGiGather /
    // bkGiGatherHero now trace the same dGiDir lattice on the device, with the gather's depth
    // carried as a TEMPLATE parameter so the one level of recursion is resolved at compile time
    // (see bkRadianceHeroLoop's header). Mode W therefore has the full mode-R device scope, and
    // the only remaining fallbacks are the ones mode R already has (`Layered`, via
    // cudaForwardSupported) — an important property, because an estimator the device is MISSING
    // in this mode would show up as a visible deterministic difference, not as extra noise.
    (void)w;
    return true;
}

Film renderBackwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                        long long spp, bool diffraction, const SppProgress* prog,
                        int maxBounce, bool directOnly, int heroC, const WhittedOpts* whitted) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    if (!cudaAvailable() || !cudaBackwardSupported(scene, cam)) return out;
    if (whitted && !cudaBackwardWhittedSupported(scene, cam, *whitted)) return out;

    DUpload up;
    buildUpload(scene, cam, resX, resY, up);
    if (maxBounce >= 1) up.sc.bkMaxBounce = maxBounce;   // Stage 3: -max-bounce cap
    up.sc.bkDirectOnly = directOnly ? 1 : 0;             // Stage 3: -direct-only (Whitted)
    if (whitted) {                                       // -mode W deterministic preview
        up.sc.bkWhitted   = 1;
        up.sc.bkGrid      = whitted->grid;
        up.sc.bkGiDirs    = whitted->giDirs;
        up.sc.bkGiGrid    = whitted->giGrid;
        up.sc.bkGiBounce  = whitted->giBounce;
        up.sc.bkGiClamp   = whitted->giClamp;
        up.sc.bkHeroSplit = whitted->heroSplit ? 1 : 0;
        up.sc.bkAmbient   = whitted->ambient;
        up.sc.bkFwPerDist = whitted->fwPerDist;   // O8 stage 2 (mode W only — see DScene)
    }
    // Hero-wavelength bundle (`-heroc N`). bkRadianceHero covers the plain surface walk
    // only, so fall back to the single-λ estimator when the scene needs a branch it does
    // not carry: participating media, gradient-index bending, or a physical lens (whose
    // per-λ refraction would give each wavelength its own camera ray). Same gate as the
    // CPU BackwardRenderer::renderRows, so CPU and GPU mode R agree on when hero applies.
    const int effHeroC = (heroC > 1 && up.sc.mediaN == 0 && !up.sc.hasGrin && !cam.hasLens())
                             ? ((heroC > hero::kHeroMax) ? hero::kHeroMax : heroC) : 1;

    const size_t npix = (size_t)resX * resY;
    double* d_film = nullptr; CUDA_CHECK(cudaMalloc(&d_film, npix * 3 * sizeof(double)));
    double* d_hits = nullptr; CUDA_CHECK(cudaMalloc(&d_hits, npix * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_film, 0, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_hits, 0, npix * sizeof(double)));
    // Resume (mode R / W disk resume). Fresh samples carry ABSOLUTE indices
    // [sppBase, sppBase + spp), so the kernel's stride must be the FINAL total, not just
    // this invocation's request — otherwise `pix * sppTotal + sampleBase` walks off the
    // end of its pixel's slot and aliases into the next pixel's seed range.
    const long long sppBase  = prog ? (long long)prog->sampleBase : 0;
    const long long sppTotal = sppBase + spp;
    // Mode W is a deterministic quadrature: its lattice is a function of the absolute
    // sample index alone, so a resumed frame must be BIT-IDENTICAL to an uninterrupted one
    // and the seed base has to stay put. The Monte-Carlo modes instead mix the loaded
    // sample count in, because there `pix * sppTotal + sampleBase` can (rarely) collide
    // with a stream the checkpointed samples already drew, and two correlated samples are
    // worse than a decorrelated stream.
    const unsigned long long seed = 0x9e3779b97f4a7c15ULL
        ^ ((prog && !whitted) ? (unsigned long long)prog->sampleBase * 0x9E3779B97F4A7C15ULL : 0ULL);

    std::vector<double> film(npix * 3);
    auto download = [&](Film& o) {
        CUDA_CHECK(cudaMemcpy(film.data(), d_film, film.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(o.hits.data(), d_hits, npix * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < npix; ++i)   // SUM over spp; writeFilm divides by spp
            o.xyz[i] = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);
    };
    auto launch = [&](long long c, long long base) {
        long long totalSamples = (long long)npix * c;
        kBackward<<<2048, 128>>>(up.sc, up.dc, d_film, d_hits, totalSamples, c, sppTotal, base, resX,
                                 diffraction ? 1 : 0, seed, effHeroC);
        cudaCheckKernel("backward");
    };

    if (!prog || !prog->report) { launch(spp, sppBase); download(out); }   // single-shot
    else gpuSppChunks(spp, *prog, out, launch, download);

    freeUpload(up);
    cudaFree(d_film); cudaFree(d_hits);
    return out;
}

// --------------------- fast RGB backward (mode R -rgb) host ------------------

bool cudaBackwardRGBSupported(const Scene& scene, const Camera& cam) {
    // The fast RGB path (bkRadianceRGB) is a deliberately-reduced Option-B tracer. It
    // needs the shared POD scene bake (cudaForwardSupported) and the same emitter/camera
    // gates as the spectral backward, PLUS: no participating media (not yet ported to the
    // RGB walk), no dispersion-dependent materials (thin-film / grating / multilayer /
    // layered / fluorescence — those effects can't survive an RGB throughput), and only
    // CONSTANT per-material reflectance (no textured / record-driven albedo, which the
    // baked rgbAlbedo doesn't capture). Scenes outside this scope fall back to the
    // spectral backward (renderBackwardCuda) or the CPU tracer.
    if (!cudaForwardSupported(scene)) return false;
    if (grin::sceneHasGrin(scene)) return false;
    for (const auto& m : scene.media) if (m.enabled) return false;   // media -> spectral path
    if (scene.envIndex >= 0 && scene.envMap) return false;           // image env -> spectral path
    for (size_t i = 0; i < scene.emitters.size(); ++i) {
        if ((int)i == scene.envIndex) continue;
        const auto& em = scene.emitters[i];
        if (em.collimated) return false;
        if (em.shape == EmitterShape::Env) return false;
    }
    for (const auto& m : scene.mats) {
        switch (m.type) {
            case MatType::Diffuse: case MatType::Dielectric: case MatType::Mirror:
            case MatType::HalfMirror: case MatType::Glossy: case MatType::Mix:
            case MatType::DiffuseTransmit: case MatType::Filter:
                break;                                              // handled by bkRadianceRGB
            default: return false;                                  // dispersion/thin-film/etc -> spectral
        }
        if (m.reflectTex >= 0) return false;                        // textured albedo not baked to RGB
        if (m.reflectPat >= 0) return false;                        // pattern-modulated albedo, ditto
        if (m.transmitPat >= 0) return false;                       // pattern-modulated transmittance, ditto
        // emitPat is fine here: an emission pattern is an ACHROMATIC scalar, so it
        // commutes with the spectral->RGB bake — ep*integral(CIE*emitSpd) is exactly
        // integral(CIE*ep*emitSpd). bkNeeLightRGB / bkRadianceRGB apply it to rgbEmit.
        if (m.recBindingFor(REC_SLOT_REFLECT)) return false;        // record-driven reflectance
    }
    if (cam.hasLens() && (int)cam.lens->surf.size() > D_MAXLENS) return false;
    return true;
}

Film renderBackwardRGBCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                           long long spp, bool diffraction, const SppProgress* prog,
                           int maxBounce, bool directOnly) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    if (!cudaAvailable() || !cudaBackwardRGBSupported(scene, cam)) return out;

    DUpload up;
    buildUpload(scene, cam, resX, resY, up);
    if (maxBounce >= 1) up.sc.bkMaxBounce = maxBounce;   // Stage 3: -max-bounce cap
    up.sc.bkDirectOnly = directOnly ? 1 : 0;             // Stage 3: -direct-only (Whitted)

    const size_t npix = (size_t)resX * resY;
    double* d_film = nullptr; CUDA_CHECK(cudaMalloc(&d_film, npix * 3 * sizeof(double)));
    double* d_hits = nullptr; CUDA_CHECK(cudaMalloc(&d_hits, npix * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_film, 0, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_hits, 0, npix * sizeof(double)));
    // Same absolute-sample-index resume contract as renderBackwardCuda; the RGB preview
    // has no deterministic mode-W path, so it keeps the decorrelating seed mix.
    const long long sppBase  = prog ? (long long)prog->sampleBase : 0;
    const long long sppTotal = sppBase + spp;
    const unsigned long long seed = 0x9e3779b97f4a7c15ULL
        ^ (prog ? (unsigned long long)prog->sampleBase * 0x9E3779B97F4A7C15ULL : 0ULL);

    std::vector<double> film(npix * 3);
    auto download = [&](Film& o) {
        CUDA_CHECK(cudaMemcpy(film.data(), d_film, film.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(o.hits.data(), d_hits, npix * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < npix; ++i)
            o.xyz[i] = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);
    };
    auto launch = [&](long long c, long long base) {
        long long totalSamples = (long long)npix * c;
        kBackwardRGB<<<2048, 128>>>(up.sc, up.dc, d_film, d_hits, totalSamples, c, sppTotal, base, resX,
                                    diffraction ? 1 : 0, seed);
        cudaCheckKernel("backwardRGB");
    };

    if (!prog || !prog->report) { launch(spp, sppBase); download(out); }
    else gpuSppChunks(spp, *prog, out, launch, download);

    freeUpload(up);
    cudaFree(d_film); cudaFree(d_hits);
    return out;
}

// ------------- Resident fast-RGB backward PREVIEW session (interactive) -------------
// See render_cuda.h for the rationale. The scene bake (buildUploadScene) is done once in
// begin() and kept in `up`; each setCamera() re-bakes only the pinhole DCamera (bakeCamera
// records nothing for a lensless camera, so there is no per-aim leak) and zeroes the SUM
// film. accumulate() advances `base` so every batch draws an independent RNG stream, with
// a fixed large `sppTotal` cap so the gidx = pix*sppTotal + base + local index stays unique
// per pixel across the whole idle convergence.
struct BackwardRGBSession {
    DUpload up;                   // resident baked scene (+ current camera in up.dc)
    const Scene* scene = nullptr; // borrowed; must outlive the session (bakeCamera arg)
    int    resX = 0, resY = 0;
    size_t npix = 0;
    double* d_film = nullptr;     // resX*resY*3 doubles (running XYZ SUM)
    double* d_hits = nullptr;     // resX*resY doubles
    long long accum = 0;          // spp accumulated since the last setCamera()
    bool   haveCam = false;
    static constexpr long long kSppCap = 1LL << 22;   // per-pixel RNG-stream capacity
};

BackwardRGBSession* backwardRGBSessionBegin(const Scene& scene, int resX, int resY,
                                            int maxBounce, bool directOnly) {
    using namespace gpu;
    if (!cudaAvailable() || resX <= 0 || resY <= 0) return nullptr;
    auto* s = new BackwardRGBSession();
    s->scene = &scene;
    s->resX = resX; s->resY = resY;
    s->npix = (size_t)resX * resY;
    buildUploadScene(scene, s->up);
    if (maxBounce >= 1) s->up.sc.bkMaxBounce = maxBounce;
    s->up.sc.bkDirectOnly = directOnly ? 1 : 0;
    CUDA_CHECK(cudaMalloc(&s->d_film, s->npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_hits, s->npix * sizeof(double)));
    CUDA_CHECK(cudaMemset(s->d_film, 0, s->npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(s->d_hits, 0, s->npix * sizeof(double)));
    return s;
}

void backwardRGBSessionSetCamera(BackwardRGBSession* s, const Camera& cam) {
    using namespace gpu;
    if (!s) return;
    s->up.dc = bakeCamera(*s->scene, cam, s->resX, s->resY, s->up);
    CUDA_CHECK(cudaMemset(s->d_film, 0, s->npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(s->d_hits, 0, s->npix * sizeof(double)));
    s->accum = 0;
    s->haveCam = true;
}

long long backwardRGBSessionAccumulate(BackwardRGBSession* s, long long spp, bool diffraction) {
    using namespace gpu;
    if (!s || !s->haveCam || spp <= 0) return s ? s->accum : 0;
    const long long base = s->accum;
    const unsigned long long seed = 0x9e3779b97f4a7c15ULL
        ^ (unsigned long long)base * 0x9E3779B97F4A7C15ULL;
    long long totalSamples = (long long)s->npix * spp;
    kBackwardRGB<<<2048, 128>>>(s->up.sc, s->up.dc, s->d_film, s->d_hits,
                                totalSamples, spp, BackwardRGBSession::kSppCap, base,
                                s->resX, diffraction ? 1 : 0, seed);
    cudaCheckKernel("backwardRGB-session");
    s->accum += spp;
    return s->accum;
}

void backwardRGBSessionDownload(BackwardRGBSession* s, Film& out) {
    using namespace gpu;
    if (!s) return;
    std::vector<double> film(s->npix * 3);
    CUDA_CHECK(cudaMemcpy(film.data(), s->d_film, film.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.hits.data(), s->d_hits, s->npix * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < s->npix; ++i)
        out.xyz[i] = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);
}

long long backwardRGBSessionSamples(const BackwardRGBSession* s) { return s ? s->accum : 0; }

void backwardRGBSessionEnd(BackwardRGBSession* s) {
    if (!s) return;
    freeUpload(s->up);
    if (s->d_film) cudaFree(s->d_film);
    if (s->d_hits) cudaFree(s->d_hits);
    delete s;
}

// --------------------- G2 iso preview (GPU raster) host ----------------------

bool cudaIsoPreviewSupported(const Scene& scene, const Camera& cam) {
    // The preview only needs closestHit (geometry traversal) + a per-material solid
    // colour, so it reuses buildUpload's POD scene bake. Gate on the forward-bake
    // support (buildUploadScene must produce a valid device scene) and a non-physical
    // camera (dGenRay covers pinhole + fisheye/panoramic; a mesh-lens camera stays on
    // the CPU raster). Fluorescence etc. never reach the shading here, but the bake path
    // is shared, so we conservatively require cudaForwardSupported.
    if (!cudaAvailable()) return false;
    if (cam.hasLens()) return false;
    return cudaForwardSupported(scene);
}

std::vector<uint8_t> renderIsoPreviewCuda(const Scene& scene, const Camera& cam,
                                          int W, int H, int nThreads, double exposure,
                                          bool autoExpose, double* lockAnchor,
                                          IsoPreviewTiming* timing) {
    using namespace gpu;
    if (!cudaIsoPreviewSupported(scene, cam)) return {};   // caller falls back to CPU raster
    if (W <= 0 || H <= 0) return {};

    // Phase clock for the optional timing split (see IsoPreviewTiming in render_cuda.h).
    // steady_clock rather than CUDA events, because host marshalling must be charged to
    // `upload` and events only see device-side work. The kernel phase syncs explicitly
    // before stopping its clock, so an async launch cannot leak its cost into `readback`.
    using PhaseClock = std::chrono::steady_clock;
    auto phaseMark = PhaseClock::now();
    auto phaseLap = [&phaseMark]() {
        auto now = PhaseClock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phaseMark).count();
        phaseMark = now;
        return ms;
    };
    if (timing) *timing = IsoPreviewTiming{};

    DUpload up;
    buildUpload(scene, cam, W, H, up);

    // Distil the scene's lights into preview keys (host deriveLight) and upload them.
    raster::PreviewLight plHost = raster::deriveLight(scene);
    std::vector<DPLight> hLights(plHost.lights.size());
    for (size_t i = 0; i < plHost.lights.size(); ++i) {
        const raster::PLight& s = plHost.lights[i];
        DPLight d;
        d.pos = DVec3(s.pos.x, s.pos.y, s.pos.z);
        d.dir = DVec3(s.dir.x, s.dir.y, s.dir.z);
        d.spot = s.spot ? 1 : 0;
        d.cosInner = s.cosInner; d.cosOuter = s.cosOuter;
        d.weight = s.weight; d.falloff2 = s.falloff2;
        hLights[i] = d;
    }
    DPreviewLight dpl;
    dpl.nLights = (int)hLights.size();
    dpl.lights  = hLights.empty() ? nullptr : (const DPLight*)up.keep(uploadVec(hLights));
    dpl.ambient = plHost.ambient; dpl.keyScale = plHost.keyScale; dpl.fill = plHost.fill;

    // One solid preview colour + emissive flag per material (host materialColor).
    std::vector<DVec3> hCol(scene.mats.size());
    std::vector<int>   hEmit(scene.mats.size());
    for (size_t i = 0; i < scene.mats.size(); ++i) {
        bool em = false;
        Vec3 c = raster::materialColor(scene.mats[i], em);
        hCol[i]  = DVec3(c.x, c.y, c.z);
        hEmit[i] = em ? 1 : 0;
    }
    const DVec3* dCol  = hCol.empty()  ? nullptr : (const DVec3*)up.keep(uploadVec(hCol));
    const int*   dEmit = hEmit.empty() ? nullptr : (const int*)up.keep(uploadVec(hEmit));

    // Image/formula-skin tables: one shared linear-RGB texel array (device twin of
    // Texture::sampleRgb) + per-material binding (reflectTex index / triplanar scale),
    // mirroring raster.h buildScene's matTex/matTri rule exactly. Procedural skins bake
    // to `rgb` at load, so image and formula skins share this one path.
    std::vector<DPTex> hTexMeta(scene.textures.size());
    std::vector<DVec3> hTexels;
    for (size_t i = 0; i < scene.textures.size(); ++i) {
        const Texture& tx = scene.textures[i];
        DPTex& m = hTexMeta[i];
        m.w = tx.w; m.h = tx.h;
        m.filter = (tx.filter == TexFilter::Nearest) ? 0 : 1;
        m.wrap   = (tx.wrap == TexWrap::Clamp) ? 1 : (tx.wrap == TexWrap::Mirror ? 2 : 0);
        m.offset = (int)hTexels.size();
        m.valid  = tx.valid() ? 1 : 0;
        if (m.valid)
            for (const Vec3& c : tx.rgb) hTexels.push_back(DVec3(c.x, c.y, c.z));
    }
    std::vector<int>    hMatTex(scene.mats.size(), -1);
    std::vector<double> hMatTri(scene.mats.size(), 0.0);
    for (size_t i = 0; i < scene.mats.size(); ++i) {
        int rt = scene.mats[i].reflectTex;
        if (!scene.mats[i].isLight && rt >= 0 && rt < (int)scene.textures.size() &&
            scene.textures[rt].valid() && !scene.textures[rt].hasPalette()) {
            hMatTex[i] = rt;
            hMatTri[i] = scene.mats[i].triplanarScale;
        }
    }
    const DPTex*  dTexMeta = hTexMeta.empty() ? nullptr : (const DPTex*)up.keep(uploadVec(hTexMeta));
    const DVec3*  dTexels  = hTexels.empty()  ? nullptr : (const DVec3*)up.keep(uploadVec(hTexels));
    const int*    dMatTex  = hMatTex.empty()  ? nullptr : (const int*)up.keep(uploadVec(hMatTex));
    const double* dMatTri  = hMatTri.empty()  ? nullptr : (const double*)up.keep(uploadVec(hMatTri));

    const size_t npix = (size_t)W * H;
    double*        d_accum = nullptr; CUDA_CHECK(cudaMalloc(&d_accum, npix * 3 * sizeof(double)));
    float*         d_z     = nullptr; CUDA_CHECK(cudaMalloc(&d_z, npix * sizeof(float)));
    unsigned char* d_emis  = nullptr; CUDA_CHECK(cudaMalloc(&d_emis, npix * sizeof(unsigned char)));

    const double EMIS_BOOST = 4.0;                       // matches raster.h renderFrame
    const DVec3  bg(0.06, 0.07, 0.09);                   // background tint (raster.h bg)
    int block = 128;
    int grid  = (int)((npix + block - 1) / block);
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;
    // Everything above marshalled the whole scene and pushed it across PCIe; charge it
    // all to `upload` before the first launch.
    if (timing) timing->msUpload = phaseLap();

    kIsoPreview<<<grid, block>>>(up.sc, up.dc, dpl, dCol, dEmit, (int)scene.mats.size(),
                                 dTexMeta, dTexels, dMatTex, dMatTri,
                                 d_accum, d_z, d_emis, W, H, bg, EMIS_BOOST);
    cudaCheckKernel("iso-preview");                  // ends with cudaDeviceSynchronize
    if (timing) timing->msKernel = phaseLap();

    std::vector<double>        accD(npix * 3);
    std::vector<float>         zf(npix);
    std::vector<unsigned char> ef(npix);
    CUDA_CHECK(cudaMemcpy(accD.data(), d_accum, accD.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(zf.data(),   d_z,     npix * sizeof(float),         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ef.data(),   d_emis,  npix * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cudaFree(d_accum); cudaFree(d_z); cudaFree(d_emis);
    freeUpload(up);

    // Shared host tail: exact same auto-exposure + sRGB tone map as the CPU rasterizer,
    // so `-raster-gpu` frames match `-raster` and a camera_path's locked anchor carries.
    std::vector<Vec3>    accum(npix);
    std::vector<uint8_t> emis(npix);
    for (size_t i = 0; i < npix; ++i) {
        accum[i] = Vec3(accD[i * 3 + 0], accD[i * 3 + 1], accD[i * 3 + 2]);
        emis[i]  = ef[i];
    }
    const double expComp = (exposure > 0.0) ? exposure : 1.0;
    const Vec3 kMilkColor{0.52, 0.55, 0.60};             // unused (seeThrough=false)
    std::vector<uint8_t> out =
        raster::exposeAndEncode(accum, zf, emis, W, H, nThreads, expComp, autoExpose,
                                lockAnchor, /*seeThrough=*/false, {}, {}, kMilkColor);
    // D2H copies + the host tone map. Named into a local first so this lap covers
    // exposeAndEncode -- returning its result directly would leave that cost untimed
    // and silently inflate whatever the caller measures around this function.
    if (timing) timing->msReadback = phaseLap();
    return out;
}

// --------------------- photon map (mode M) host ------------------------------

bool cudaPhotonMapSupported(const Scene& scene) {
    // The GPU mode-M path reuses the forward photon tracer (deposit) and a device gather
    // that mirrors photonGather's DIRECT (fgRays == 0) density estimate. Scope for v1:
    //   * same POD-bakeable materials as the forward path (cudaForwardSupported).
    // Environment lights (constant AND image-based, M2) are supported: the deposit emits
    // env photons (env's INDIRECT bounces land in the map) and dPhotonGather adds env's
    // DIRECT term on gather-ray escape, mirroring CPU photonGather.
    // Final gather (g_pmFinalGather > 0) and physical-lens cameras are gated by the caller
    // (main.cpp) since those are render-config, not scene, properties. Fluorescence is fine:
    // neither CPU nor GPU deposits at a fluorescent vertex, and both gather it as a query
    // point, so the two agree. Participating media (fog) are supported — the forward deposit
    // pass runs the same Woodcock free-flight as the CPU tracePhoton.
    if (!cudaForwardSupported(scene)) return false;
    // Hair runs on the device forward tracer but the mode-M gather shades every query
    // point as Lambertian (see sceneUsesHairMaterial) — hair scenes go to the CPU
    // photon map (and, via the chained gates, CPU SPPM).
    if (sceneUsesHairMaterial(scene)) return false;
    return true;
}

// ---- mode-M photon-map cache file (-savemap / -loadmap) -----------------------
// The deposited photon map is view-INDEPENDENT: it is the (expensive) result of the
// forward photon trace, and any camera at any gather radius can be gathered from it. So
// it is worth persisting. `-savemap <f>` writes the built map after the deposit pass;
// `-loadmap <f>` reloads it and SKIPS the deposit entirely, re-gathering new camera
// angles / a new radius for free without re-tracing a single photon. The file stores the
// raw photon set + emitted count + energy tally; the grid is rebuilt on load via
// PhotonMap::build(radius), so one file serves any gather radius. A scene-identity guard
// (magic "FTPMP01\n") refuses to blend a stale map into a different scene.
static uint64_t photonMapGuard(const Scene& scene, bool diffraction) {
    uint64_t h = 14695981039346656037ULL;                 // FNV-1a offset basis
    auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ULL; };
    mix((uint64_t)scene.tris.size());
    mix((uint64_t)scene.spheres.size());
    mix((uint64_t)scene.emitters.size());
    uint64_t tp; std::memcpy(&tp, &scene.totalPower, sizeof tp); mix(tp);
    mix(diffraction ? 0x9E37ULL : 0x1234ULL);
    return h;
}

static bool savePhotonMap(const char* path, const PhotonMap& pm,
                          const EnergyReport& e, uint64_t guard) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "[savemap] cannot open %s for writing\n", path); return false; }
    // FTPMP02: positions and payloads are stored as two separate blocks, matching
    // PhotonMap's split layout (FTPMP01 held one interleaved array that also carried a
    // never-read incident direction). Bumping the magic makes an old cache fail the
    // recognition check below rather than being misread as garbage.
    const char magic[8] = {'F','T','P','M','P','0','2','\n'};
    long long nPh = (long long)pm.photons.size();
    double en[5] = {e.emitted, e.absorbed, e.sensor, e.escaped, e.residual};
    bool ok = true;
    ok = ok && std::fwrite(magic, 1, 8, f) == 8;
    ok = ok && std::fwrite(&guard, sizeof guard, 1, f) == 1;
    ok = ok && std::fwrite(&pm.nEmitted, sizeof pm.nEmitted, 1, f) == 1;
    ok = ok && std::fwrite(en, sizeof en, 1, f) == 1;
    ok = ok && std::fwrite(&nPh, sizeof nPh, 1, f) == 1;
    if (ok && nPh > 0) {
        ok = std::fwrite(pm.pos.data(), sizeof(Vec3), (size_t)nPh, f) == (size_t)nPh;
        ok = ok && std::fwrite(pm.photons.data(), sizeof(Photon), (size_t)nPh, f) == (size_t)nPh;
    }
    std::fclose(f);
    if (!ok) std::fprintf(stderr, "[savemap] write to %s failed\n", path);
    return ok;
}

static bool loadPhotonMap(const char* path, PhotonMap& pm,
                          EnergyReport& e, uint64_t guard) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "[loadmap] cannot open %s\n", path); return false; }
    char magic[8] = {0};
    long long nEmitted = 0, nPh = 0; double en[5] = {0,0,0,0,0}; uint64_t g = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8;
    if (!ok || std::memcmp(magic, "FTPMP02\n", 8) != 0) {
        // Name the stale-version case explicitly: a user with a cache from before the
        // split layout should be told to re-deposit, not left guessing.
        if (ok && std::memcmp(magic, "FTPMP01\n", 8) == 0)
            std::fprintf(stderr, "[loadmap] %s is an old FTPMP01 map (pre split-layout); "
                                 "re-run with -savemap to rebuild it. Ignoring.\n", path);
        else
            std::fprintf(stderr, "[loadmap] %s is not a recognised photon-map file; ignoring\n", path);
        std::fclose(f); return false;
    }
    ok = ok && std::fread(&g, sizeof g, 1, f) == 1;
    ok = ok && std::fread(&nEmitted, sizeof nEmitted, 1, f) == 1;
    ok = ok && std::fread(en, sizeof en, 1, f) == 1;
    ok = ok && std::fread(&nPh, sizeof nPh, 1, f) == 1;
    if (!ok) { std::fprintf(stderr, "[loadmap] %s truncated header; ignoring\n", path); std::fclose(f); return false; }
    if (g != guard) {
        std::fprintf(stderr, "[loadmap] %s was built for a different scene; ignoring\n", path);
        std::fclose(f); return false;
    }
    if (nPh > 0) {
        pm.pos.resize((size_t)nPh);
        pm.photons.resize((size_t)nPh);
        ok = std::fread(pm.pos.data(), sizeof(Vec3), (size_t)nPh, f) == (size_t)nPh;
        ok = ok && std::fread(pm.photons.data(), sizeof(Photon), (size_t)nPh, f) == (size_t)nPh;
    }
    std::fclose(f);
    if (!ok) {
        std::fprintf(stderr, "[loadmap] %s truncated photon data; ignoring\n", path);
        pm.photons.clear(); pm.pos.clear(); return false;
    }
    pm.nEmitted = nEmitted;
    e.emitted += en[0]; e.absorbed += en[1]; e.sensor += en[2]; e.escaped += en[3]; e.residual += en[4];
    return true;
}

// Build the view-independent photon map on the GPU (forward deposit pass) and gather every
// camera from it — the flythrough win, on the device. The deposit runs the megakernel
// forward tracer TWICE: a count-only pass to size the buffer exactly, then a fill pass
// (deterministic same-seed launch). The grid is built on the host (PhotonMap::build, the
// tested counting sort) and re-uploaded, then each camera is gathered by kGather. Films are
// SUMs over spp (writeFilm divides by spp), matching renderPhotonCamera / the backward path.
std::vector<Film> renderPhotonMapSharedCuda(const Scene& scene, const std::vector<Camera>& cams,
                                            const std::vector<int>& resX, const std::vector<int>& resY,
                                            long long N, double radius, EnergyReport& eOut,
                                            bool diffraction, long long spp,
                                            const SppProgress* prog,
                                            const std::function<bool(int, const Film&)>* onFrame,
                                            const char* mapLoad, const char* mapSave, int heroC,
                                            int fgRays, double autoK) {
    using namespace gpu;
    int nc = (int)cams.size();
    std::vector<Film> out(nc);
    for (int c = 0; c < nc; ++c) { out[c].resX = resX[c]; out[c].resY = resY[c]; out[c].alloc(); }
    if (nc == 0 || !cudaAvailable() || !cudaPhotonMapSupported(scene)) return out;

    DUpload up;
    buildUploadScene(scene, up);

    // Staging-chunk size for the host<->device photon copies (deposit download AND grid
    // upload below). Streaming the deposits in blocks (instead of one giant transfer) means
    // the host never holds a second full DPhoton mirror of the map alongside pm.photons -- at
    // high photon counts (tens of millions emitted, each depositing at several bounces) those
    // two full copies together would exhaust host RAM (std::bad_alloc). A 4M-photon block is
    // ~176 MB, negligible next to the map itself.
    const size_t PM_CHUNK = 4u << 20;

    // Load a cached map (-loadmap) and skip the deposit entirely, or run the forward deposit
    // trace and optionally persist it (-savemap). The map is view-independent, so a loaded
    // one is re-gathered for any camera/radius without re-tracing a photon.
    PhotonMap pm;

    // Bin the map, honouring the density-adaptive radius (autoK > 0). Say out loud what
    // radius it settled on: the one printed before the deposit is only a starting point and
    // a silently-different one would be baffling when comparing renders. The gather below
    // reads pm.radius, so nothing else needs to know which branch ran.
    auto buildMap = [&]() {
        if (autoK <= 0.0) { pm.build(radius); return; }
        double nProbe = 0.0, kTarget = 0.0;
        const double r = pm.buildAuto(radius, autoK, &nProbe, &kTarget);
        std::printf("[gpu] adaptive gather radius: %.4g -> %.4g (a typical gather saw %.0f "
                    "photons at the starting radius; target %.0f for %zu stored)\n",
                    radius, r, nProbe, kTarget, pm.photons.size());
    };

    bool mapLoaded = false;
    if (mapLoad && *mapLoad) {
        mapLoaded = loadPhotonMap(mapLoad, pm, eOut, photonMapGuard(scene, diffraction));
        if (mapLoaded) {
            std::printf("[loadmap] %s: %zu photons from %lld emitted -- deposit skipped\n",
                        mapLoad, pm.photons.size(), (long long)pm.nEmitted);
            buildMap();                         // (re)build the grid at the requested radius
        } else {
            std::fprintf(stderr, "[loadmap] falling back to a fresh deposit\n");
        }
    }
    if (!mapLoaded) {
    // ---- forward deposit pass (count, then fill) ----
    unsigned long long* d_depCount = nullptr;
    CUDA_CHECK(cudaMalloc(&d_depCount, sizeof(unsigned long long)));
    double* d_energy = nullptr; CUDA_CHECK(cudaMalloc(&d_energy, 5 * sizeof(double)));

    auto depositLaunch = [&](DPhoton* buf, unsigned long long cap) {
        CUDA_CHECK(cudaMemset(d_depCount, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_energy, 0, 5 * sizeof(double)));
        DCamSet cs{};                       // nCam == 0: every camera splat is a no-op
        cs.cams = nullptr; cs.films = nullptr; cs.hits = nullptr; cs.nCam = 0;
        cs.depPhotons = buf; cs.depCount = d_depCount; cs.depCap = cap;
        launchForward(up, cs, d_energy, N, diffraction, /*seedBase*/0, /*wavefront*/false, CAM_B, heroC);
    };

    // Single-pass deposit: the old flow ran the WHOLE forward trace twice (a count-only
    // sizing pass, then a fill pass). Instead, stage into a generously guessed buffer
    // (2.5 deposits/photon + slack, clamped to half of free VRAM) and only rerun if the
    // guess undershot — the atomic cursor keeps counting past the cap, so an overflow
    // still yields the exact total and the rerun costs no more than the old flow did.
    // Same seed every pass => identical deposits regardless of which path executes.
    pm.nEmitted = N;
    unsigned long long cap = (unsigned long long)((double)N * 2.5) + (1ull << 20);
    { size_t freeB = 0, totalB = 0;
      if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
          unsigned long long fit = (unsigned long long)(freeB / 2 / sizeof(DPhoton));
          if (cap > fit) cap = fit; } }
    DPhoton* d_photons = nullptr;
    while (cap >= (1ull << 22) &&
           cudaMalloc(&d_photons, (size_t)cap * sizeof(DPhoton)) != cudaSuccess) {
        cudaGetLastError(); d_photons = nullptr; cap >>= 1;   // halve until it fits
    }
    unsigned long long nDep = 0;
    if (d_photons) {
        depositLaunch(d_photons, cap);      // optimistic fill against the guess
        CUDA_CHECK(cudaMemcpy(&nDep, d_depCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (nDep > cap) {                   // undershot: rerun once at the exact size
            cudaFree(d_photons); d_photons = nullptr;
            CUDA_CHECK(cudaMalloc(&d_photons, (size_t)nDep * sizeof(DPhoton)));
            depositLaunch(d_photons, nDep);
            unsigned long long nFill = 0;
            CUDA_CHECK(cudaMemcpy(&nFill, d_depCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            if (nFill < nDep) nDep = nFill;
        }
    } else {
        // Couldn't stage even a modest guess: fall back to the old two-pass flow.
        depositLaunch(nullptr, 0);          // count-only sizing pass
        CUDA_CHECK(cudaMemcpy(&nDep, d_depCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (nDep > 0) {
            CUDA_CHECK(cudaMalloc(&d_photons, (size_t)nDep * sizeof(DPhoton)));
            depositLaunch(d_photons, nDep); // fill pass (same seed => same nDep deposits)
        }
    }
    if (nDep > 0 && d_photons) {
        // Download + convert in chunks (never a full host-side DPhoton copy). Positions
        // and payloads split into PhotonMap's two parallel arrays (see Photon in
        // photonmap.h); DPhoton has the same fields, so this is a pure widen + split.
        pm.photons.resize((size_t)nDep);
        pm.pos.resize((size_t)nDep);
        std::vector<DPhoton> stage;
        for (size_t off = 0; off < (size_t)nDep; off += PM_CHUNK) {
            size_t cnt = std::min(PM_CHUNK, (size_t)nDep - off);
            stage.resize(cnt);
            CUDA_CHECK(cudaMemcpy(stage.data(), d_photons + off, cnt * sizeof(DPhoton),
                                  cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < cnt; ++i) {
                const DPhoton& d = stage[i];
                Photon& p = pm.photons[off + i];
                pm.pos[off + i] = Vec3(d.pos.x, d.pos.y, d.pos.z);
                p.n   = Vec3(d.n.x,   d.n.y,   d.n.z);
                p.power = d.power; p.lambda = d.lambda;
            }
        }
    }
    if (d_photons) cudaFree(d_photons);
    buildMap();                             // host counting sort -> cell-contiguous runs

    double energy[5] = {0,0,0,0,0};
    CUDA_CHECK(cudaMemcpy(energy, d_energy, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    eOut.emitted  += energy[0]; eOut.absorbed += energy[1]; eOut.sensor += energy[2];
    eOut.escaped  += energy[3]; eOut.residual += energy[4];
    cudaFree(d_depCount); cudaFree(d_energy);
    if (mapSave && *mapSave) {
        EnergyReport passE{energy[0], energy[1], energy[2], energy[3], energy[4]};
        if (savePhotonMap(mapSave, pm, passE, photonMapGuard(scene, diffraction)))
            std::printf("[savemap] wrote %s: %zu photons (%lld emitted)\n",
                        mapSave, pm.photons.size(), (long long)pm.nEmitted);
    }
    }   // end if (!mapLoaded): deposit + build + optional save

    // ---- upload the built grid ----
    DPhotonMap dpm{};
    dpm.lo = DVec3(pm.lo.x, pm.lo.y, pm.lo.z);
    dpm.cellSize = (Real)pm.cellSize; dpm.radius = (Real)pm.radius;
    dpm.nx = pm.nx; dpm.ny = pm.ny; dpm.nz = pm.nz;
    dpm.photons = nullptr;
    if (!pm.photons.empty()) {
        // Upload the sorted map host->device in chunks (no full mirror), folding each
        // photon's constant gather weight into the record (see DGatherPhoton):
        // p? = cie?(lambda) * power * norm / pi, with the CIE triple taken from
        // PhotonMap::cie (precomputed in double by build(), index-aligned with the
        // sorted photons). The radius is cast through Real first so the folded
        // normalization equals the old in-kernel double((Real)radius)^2 exactly.
        const double rr   = (double)(Real)pm.radius;
        const double area = DPI * rr * rr;
        const double fold = (pm.nEmitted > 0 && area > 0.0)
                          ? 1.0 / (area * (double)pm.nEmitted * DPI) : 0.0;
        size_t n = pm.photons.size();
        DGatherPhoton* d_sorted = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sorted, n * sizeof(DGatherPhoton)));
        std::vector<DGatherPhoton> stage;
        for (size_t off = 0; off < n; off += PM_CHUNK) {
            size_t cnt = std::min(PM_CHUNK, n - off);
            stage.resize(cnt);
            for (size_t i = 0; i < cnt; ++i) {
                const Photon& p = pm.photons[off + i];
                const Vec3&  pp = pm.pos[off + i];
                const Vec3&  ci = pm.cie[off + i];
                DGatherPhoton& d = stage[i];
                d.pos = DVec3(pp.x, pp.y, pp.z);
                d.n   = DVec3(p.n.x,   p.n.y,   p.n.z);
                const double w = (double)p.power * fold;
                d.pX = (float)(ci.x * w);
                d.pY = (float)(ci.y * w);
                d.pZ = (float)(ci.z * w);
                d.lambda = p.lambda;
            }
            CUDA_CHECK(cudaMemcpy(d_sorted + off, stage.data(), cnt * sizeof(DGatherPhoton),
                                  cudaMemcpyHostToDevice));
        }
        dpm.photons = (const DGatherPhoton*)up.keep(d_sorted);
    }
    // cellStart always has >= 2 entries after build() (even for an empty map, where every
    // [begin,end) slice is empty so the gather loop simply never runs) — always upload it.
    dpm.cellStart = (const int*)up.keep(uploadVec(pm.cellStart));

    // ---- gather each camera ----
    // Pull the current device accumulation for camera c into out[c] (film + hit map).
    auto downloadFilm = [&](int c, const double* d_film, const double* d_hits, size_t npix) {
        std::vector<double> film(npix * 3);
        CUDA_CHECK(cudaMemcpy(film.data(), d_film, film.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out[c].hits.data(), d_hits, npix * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < npix; ++i)
            out[c].xyz[i] = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);
    };
    const bool live = (prog && prog->report);
    auto lastReport = std::chrono::steady_clock::now();
    bool stopped = false;
    for (int c = 0; c < nc && !stopped; ++c) {
        DCamera hc = bakeCamera(scene, cams[c], resX[c], resY[c], up);
        const size_t npix = (size_t)resX[c] * resY[c];
        double* d_film = nullptr; CUDA_CHECK(cudaMalloc(&d_film, npix * 3 * sizeof(double)));
        double* d_hits = nullptr; CUDA_CHECK(cudaMalloc(&d_hits, npix * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_film, 0, npix * 3 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_hits, 0, npix * sizeof(double)));
        const unsigned long long seed = 0xA24BAED4963EE407ULL ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(c + 1));
        // Chunk spp so a single launch stays well under the Windows TDR watchdog even when a
        // caustic cell holds a dense photon cluster (heavy density query).
        long long chunk = 200000 / (long long)(npix ? npix : 1); if (chunk < 1) chunk = 1;
        for (long long base = 0; base < spp; base += chunk) {
            long long cs2 = (base + chunk <= spp) ? chunk : (spp - base);
            long long total = (long long)npix * cs2;
            kGather<<<2048, 128>>>(up.sc, dpm, hc, d_film, d_hits, total, cs2, spp, base,
                                   resX[c], diffraction ? 1 : 0, fgRays, seed);
            cudaCheckKernel("photon-gather");
            // Live view: after a chunk, hand the host the frame-so-far so it can refresh the
            // window/preview. Throttle to ~10 Hz (a high-res gather chunks one spp at a time,
            // which is far finer than the eye needs) but always report the completed frame.
            if (live) {
                long long done = base + cs2;
                bool frameDone = (done >= spp);
                auto now = std::chrono::steady_clock::now();
                if (frameDone || std::chrono::duration<double>(now - lastReport).count() >= 0.1) {
                    downloadFilm(c, d_film, d_hits, npix);
                    if (prog->report(out[c], done, frameDone)) stopped = true;
                    lastReport = now;
                    if (stopped) break;
                }
            }
        }
        downloadFilm(c, d_film, d_hits, npix);   // ensure out[c] holds the final accumulation
        cudaFree(d_film); cudaFree(d_hits);
        // Hand the finished frame to the host for IMMEDIATE crash-safe write, then release
        // its buffers so a long flythrough runs in ~one frame of host RAM instead of holding
        // all nc films to the end (mirrors the CPU mode-M path, which writes per frame). If
        // the host asks to stop (window closed / Ctrl-C), quit after this frame — everything
        // written so far is already safely on disk.
        if (onFrame) {
            bool stopReq = (*onFrame)(c, out[c]);
            Film empty; empty.resX = resX[c]; empty.resY = resY[c];   // shape kept, buffers freed
            out[c] = std::move(empty);
            if (stopReq) stopped = true;
        }
        if (nc > 1) {   // watchable per-frame progress on a multi-camera (flythrough) render
            std::printf("\r[camera] mode-M GPU gather %d/%d ...", c + 1, nc);
            std::fflush(stdout);
        }
    }
    if (nc > 1) { std::printf("\n"); std::fflush(stdout); }

    freeUpload(up);
    return out;
}

// ================= host: device-scratch reuse (VCM / SPPM sessions) =================
// thrust algorithms allocate temporary device storage per call; by default that is a
// cudaMalloc/cudaFree pair EVERY call, which (with the sessions' own per-pass buffer
// churn) profiled at ~10% of per-pass API time. This bump arena keeps grow-only blocks
// alive across passes: alloc() carves from existing blocks (first-fit) and cudaMallocs
// only on a new high-water mark; deallocate is a no-op; reset() rewinds the offsets at
// the start of each pass. Steady state: zero device malloc/free per pass.
struct ThrustArena {
    struct Block { char* p; size_t cap, off; };
    std::vector<Block> blocks;
    void reset() { for (Block& b : blocks) b.off = 0; }
    char* alloc(size_t n) {
        n = (n + 255) & ~(size_t)255;                    // 256-byte aligned carves
        for (Block& b : blocks)
            if (b.cap - b.off >= n) { char* r = b.p + b.off; b.off += n; return r; }
        Block nb{}; nb.cap = n; nb.off = n;
        CUDA_CHECK(cudaMalloc(&nb.p, nb.cap));
        blocks.push_back(nb);
        return nb.p;
    }
    void release() { for (Block& b : blocks) cudaFree(b.p); blocks.clear(); }
};
// Minimal Allocator facade over the arena for FT_THRUST_PAR(alloc).
struct ThrustArenaAlloc {
    using value_type = char;
    ThrustArena* arena;
    char* allocate(std::ptrdiff_t n) { return arena->alloc((size_t)n); }
    void deallocate(char*, size_t) {}
};

// Grow-only device buffer: (re)allocates only when `need` exceeds the current capacity
// (1.5x growth), so per-pass session buffers stop churning cudaMalloc/cudaFree.
template <class T>
static void ensureDevCap(T*& p, size_t& cap, size_t need) {
    if (need <= cap) return;
    if (p) { cudaFree(p); p = nullptr; }
    size_t newCap = cap + cap / 2;
    if (newCap < need) newCap = need;
    CUDA_CHECK(cudaMalloc(&p, newCap * sizeof(T)));
    cap = newCap;
}

// ============================ GPU SPPM (mode S) ============================
// Resident device SPPM session: keeps per-pixel progressive state (tau/radius/nAcc/directSum)
// on the device across passes, so the mode-S driver in main.cpp calls one pass per iteration
// and resolves the current radiance whenever it wants a preview/checkpoint frame. Each pass
// (1) resamples camera visible points (kSppmVisiblePoint), (2) deposits a bounded photon set
// via the SAME forward tracer as mode M into a persistent grow-only buffer, builds the photon
// grid ON DEVICE at the largest current per-pixel radius (exact PhotonMap::build mirror —
// stable sort == counting sort, so the gather sees identical photon order), and (3) gathers +
// progressively updates every pixel (kSppmGather). No photon ever round-trips to the host.
struct SppmSession {
    DUpload up;
    gpu::DCamera cam{};
    int resX = 0, resY = 0;
    size_t npix = 0;
    bool diffraction = false;
    int  maxBounce = 32, heroC = 1;
    long long emittedTotal = 0;
    long long passes = 0;
    gpu::DSppmState st{};
    double* d_film = nullptr;
    unsigned long long* d_depCount = nullptr;
    double* d_energy = nullptr;
    EnergyReport energy{};
    // Grow-only device scratch for the on-device photon grid build (no per-pass malloc):
    gpu::DPhoton* d_photons = nullptr;       size_t photonsCap = 0;   // deposit buffer (records)
    gpu::DGatherPhoton* d_gather = nullptr;  size_t gatherCap = 0;    // sorted gather records
    int* d_cellKey   = nullptr;              size_t cellKeyCap = 0;
    int* d_order     = nullptr;              size_t orderCap = 0;
    int* d_cellStart = nullptr;              size_t cellStartCap = 0; // entries (nCells+1)
    ThrustArena arena;                        // thrust temp storage (sort/scan/reduce)
};

bool cudaSppmSupported(const Scene& scene) {
    // SPPM reuses the mode-M deposit + a per-pixel visible-point/gather pass. Same
    // device-bakeable scope as the photon map (which now includes constant + image env, M2);
    // pinhole cameras only (dGenRay) — the caller gates the camera.
    return cudaPhotonMapSupported(scene);
}

SppmSession* sppmSessionBegin(const Scene& scene, const Camera& cam, int resX, int resY,
                              double R0, bool diffraction, int maxBounce, int heroC) {
    if (!cudaAvailable() || !cudaSppmSupported(scene)) return nullptr;
    SppmSession* s = new SppmSession();
    s->resX = resX; s->resY = resY; s->npix = (size_t)resX * resY;
    s->diffraction = diffraction; s->maxBounce = (maxBounce > 0) ? maxBounce : 32; s->heroC = heroC;
    buildUploadScene(scene, s->up);
    s->cam = bakeCamera(scene, cam, resX, resY, s->up);
    const size_t np = s->npix;
    CUDA_CHECK(cudaMalloc(&s->st.tau,       np * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->st.directSum, np * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->st.nAcc,      np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->st.vpThr,     np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->st.radius,    np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->st.vpValid,   np * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&s->st.vpHit,     np * sizeof(gpu::DHit)));
    CUDA_CHECK(cudaMemset(s->st.tau,       0, np * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(s->st.directSum, 0, np * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(s->st.nAcc,      0, np * sizeof(double)));
    CUDA_CHECK(cudaMemset(s->st.vpThr,     0, np * sizeof(double)));
    CUDA_CHECK(cudaMemset(s->st.vpValid,   0, np * sizeof(unsigned char)));
    { std::vector<double> r0(np, R0);
      CUDA_CHECK(cudaMemcpy(s->st.radius, r0.data(), np * sizeof(double), cudaMemcpyHostToDevice)); }
    CUDA_CHECK(cudaMalloc(&s->d_film,    np * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_depCount, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&s->d_energy,   5 * sizeof(double)));
    return s;
}

// Run one SPPM pass. photonsPerPass photons are deposited fresh (seedBase = cumulative
// emitted, so every pass is an independent photon set) into a bounded map.
void sppmSessionPass(SppmSession* s, long long photonsPerPass, double alpha) {
    using namespace gpu;
    const long long passIdx = s->passes;                 // 0-based index of THIS pass

    // (1) Camera visible-point pass: fresh visible point + direct sample per pixel.
    unsigned long long vpSeed = 0xA24BAED4963EE407ULL
                              ^ ((unsigned long long)(passIdx + 1) * 0x9E3779B97F4A7C15ULL);
    kSppmVisiblePoint<<<2048, 128>>>(s->up.sc, s->cam, s->st, s->resX, s->resY,
                                     s->maxBounce, vpSeed, passIdx + 1);
    cudaCheckKernel("sppm-visible-point");

    // (2) Forward deposit into a bounded, PERSISTENT grow-only device buffer (photonsPerPass
    // is constant across a run, so after the first pass this never reallocates). seedBase =
    // cumulative emitted (fresh set). The grid is then built ENTIRELY ON DEVICE — the old
    // path downloaded every photon record, counting-sorted + CIE-filled on the host, and
    // uploaded the sorted records + grid (~100 MB down + ~90 MB up per pass, ~45% memcpy +
    // ~10% malloc of API time under nsys, plus dominant host convert/sort wall time).
    auto depositLaunch = [&](DPhoton* buf, unsigned long long capv) {
        CUDA_CHECK(cudaMemset(s->d_depCount, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(s->d_energy, 0, 5 * sizeof(double)));
        DCamSet cs{}; cs.nCam = 0;
        cs.depPhotons = buf; cs.depCount = s->d_depCount; cs.depCap = capv;
        launchForward(s->up, cs, s->d_energy, photonsPerPass, s->diffraction,
                      (unsigned long long)s->emittedTotal, /*wavefront*/false, CAM_B, s->heroC);
    };
    unsigned long long cap = (unsigned long long)((double)photonsPerPass * 2.5) + (1ull << 20);
    if (cap > s->photonsCap) {
        { size_t freeB = 0, totalB = 0;
          if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
              // Budget half the free VRAM, counting what the current buffer already holds.
              unsigned long long fit = (unsigned long long)(
                  (freeB + s->photonsCap * sizeof(DPhoton)) / 2 / sizeof(DPhoton));
              if (cap > fit) cap = fit; } }
        if (cap > s->photonsCap) {
            if (s->d_photons) { cudaFree(s->d_photons); s->d_photons = nullptr; s->photonsCap = 0; }
            while (cap >= (1ull << 20) &&
                   cudaMalloc(&s->d_photons, (size_t)cap * sizeof(DPhoton)) != cudaSuccess) {
                cudaGetLastError(); s->d_photons = nullptr; cap >>= 1;
            }
            if (s->d_photons) s->photonsCap = cap;
        }
    }
    unsigned long long nDep = 0;
    if (s->d_photons) {
        depositLaunch(s->d_photons, s->photonsCap);
        CUDA_CHECK(cudaMemcpy(&nDep, s->d_depCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (nDep > s->photonsCap) {                      // undershot: regrow, rerun at exact size
            cudaFree(s->d_photons); s->d_photons = nullptr; s->photonsCap = 0;
            CUDA_CHECK(cudaMalloc(&s->d_photons, (size_t)nDep * sizeof(DPhoton)));
            s->photonsCap = nDep;
            depositLaunch(s->d_photons, nDep);
            unsigned long long nFill = 0;
            CUDA_CHECK(cudaMemcpy(&nFill, s->d_depCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            if (nFill < nDep) nDep = nFill;
        }
    } else {
        depositLaunch(nullptr, 0);                       // count-only sizing pass
        CUDA_CHECK(cudaMemcpy(&nDep, s->d_depCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (nDep > 0) { CUDA_CHECK(cudaMalloc(&s->d_photons, (size_t)nDep * sizeof(DPhoton)));
                        s->photonsCap = nDep;
                        depositLaunch(s->d_photons, nDep); }
    }
    double energy[5] = {0,0,0,0,0};
    CUDA_CHECK(cudaMemcpy(energy, s->d_energy, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    s->energy.emitted += energy[0]; s->energy.absorbed += energy[1]; s->energy.sensor += energy[2];
    s->energy.escaped += energy[3]; s->energy.residual += energy[4];

    s->arena.reset();
    ThrustArenaAlloc tal{&s->arena};
    auto pol = FT_THRUST_PAR(tal);

    // rMax = largest radius over VALID pixels (grid built there so every pixel's — never
    // larger — radius stays inside the 3x3x3 neighbourhood). Mirrors CPU sppmPass; a device
    // max-reduce is order-independent, so it yields the host loop's exact double.
    double rMax = thrust::transform_reduce(pol,
        thrust::counting_iterator<long long>(0),
        thrust::counting_iterator<long long>((long long)s->npix),
        SppmRMaxF{s->st.radius, s->st.vpValid}, 0.0, MaxD{});
    if (rMax <= 0.0) rMax = 1e-4;
    s->emittedTotal += photonsPerPass;
    s->passes += 1;

    // On-device grid build — exact PhotonMap::build mirror (photonmap.h): same half-cell
    // bbox padding, same promoted-double cell coords, and a stable sort by cell id that
    // reproduces the host counting sort's exact photon order. kSppmGatherConvert then
    // folds cie(lambda)*power/pi into each record (NO area/nEmitted fold; those depend on
    // the current per-pixel radius, applied at resolve). Then gather + progressive update.
    const double cellSize = (rMax > 0.0) ? rMax : 1e-6;
    double lox = 0.0, loy = 0.0, loz = 0.0;
    int gnx = 1, gny = 1, gnz = 1;
    DPhotonMap dpm{};
    dpm.photons = nullptr;
    if (nDep > 0) {
        const size_t n = (size_t)nDep;
        BboxF bb = thrust::transform_reduce(pol,
            thrust::device_pointer_cast(s->d_photons),
            thrust::device_pointer_cast(s->d_photons + n),
            PhToBboxF{}, BboxF{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX},
            BboxMergeF{});
        // The host build promoted float positions to double BEFORE the bbox/pad math;
        // min/max commute with that promotion, so these are the same doubles.
        lox = (double)bb.mnx - cellSize * 0.5;
        loy = (double)bb.mny - cellSize * 0.5;
        loz = (double)bb.mnz - cellSize * 0.5;
        const double ex = ((double)bb.mxx - lox) + cellSize * 0.5;
        const double ey = ((double)bb.mxy - loy) + cellSize * 0.5;
        const double ez = ((double)bb.mxz - loz) + cellSize * 0.5;
        gnx = std::max(1, (int)std::ceil(ex / cellSize));
        gny = std::max(1, (int)std::ceil(ey / cellSize));
        gnz = std::max(1, (int)std::ceil(ez / cellSize));
        const long long nCells = (long long)gnx * gny * gnz;
        ensureDevCap(s->d_cellKey, s->cellKeyCap, n);
        ensureDevCap(s->d_order,   s->orderCap,   n);
        kSppmCellKey<<<2048, 128>>>(s->d_photons, (long long)n, lox, loy, loz, cellSize,
                                    gnx, gny, gnz, s->d_cellKey);
        cudaCheckKernel("sppm-cellkey");
        thrust::device_ptr<int> tKey(s->d_cellKey), tOrd(s->d_order);
        thrust::sequence(pol, tOrd, tOrd + n);
        thrust::stable_sort_by_key(pol, tKey, tKey + n, tOrd);
        ensureDevCap(s->d_cellStart, s->cellStartCap, (size_t)nCells + 1);
        thrust::lower_bound(pol, tKey, tKey + n,
                            thrust::counting_iterator<int>(0),
                            thrust::counting_iterator<int>((int)(nCells + 1)),
                            thrust::device_pointer_cast(s->d_cellStart));
        ensureDevCap(s->d_gather, s->gatherCap, n);
        kSppmGatherConvert<<<2048, 128>>>(s->d_photons, s->d_order, (long long)n, s->d_gather);
        cudaCheckKernel("sppm-gather-convert");
        dpm.photons = s->d_gather;
    } else {
        ensureDevCap(s->d_cellStart, s->cellStartCap, 2);
        CUDA_CHECK(cudaMemset(s->d_cellStart, 0, 2 * sizeof(int)));
    }
    dpm.lo = DVec3(lox, loy, loz);
    dpm.cellSize = (Real)cellSize; dpm.radius = (Real)rMax;
    dpm.nx = gnx; dpm.ny = gny; dpm.nz = gnz;
    dpm.cellStart = s->d_cellStart;

    // (3) Gather + progressive update.
    kSppmGather<<<2048, 128>>>(s->up.sc, dpm, s->st, s->resX, s->resY, alpha);
    cudaCheckKernel("sppm-gather");
}

// Resolve the current accumulated state into `out` (radiance L, exactly like CPU sppmResolve).
void sppmSessionResolve(SppmSession* s, Film& out) {
    using namespace gpu;
    kSppmResolve<<<2048, 128>>>(s->st, s->d_film, s->resX, s->resY, s->passes, (double)s->emittedTotal);
    cudaCheckKernel("sppm-resolve");
    std::vector<double> film(s->npix * 3);
    CUDA_CHECK(cudaMemcpy(film.data(), s->d_film, film.size() * sizeof(double), cudaMemcpyDeviceToHost));
    if (out.resX != s->resX || out.resY != s->resY || out.xyz.empty()) {
        out.resX = s->resX; out.resY = s->resY; out.alloc();
    }
    for (size_t i = 0; i < s->npix; ++i) {
        out.xyz[i]  = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);
        out.hits[i] = 1.0;
    }
}

long long sppmSessionPasses(const SppmSession* s)  { return s ? s->passes : 0; }
long long sppmSessionEmitted(const SppmSession* s) { return s ? s->emittedTotal : 0; }

void sppmSessionEnd(SppmSession* s) {
    if (!s) return;
    cudaFree(s->st.tau); cudaFree(s->st.directSum); cudaFree(s->st.nAcc);
    cudaFree(s->st.vpThr); cudaFree(s->st.radius); cudaFree(s->st.vpValid); cudaFree(s->st.vpHit);
    cudaFree(s->d_film); cudaFree(s->d_depCount); cudaFree(s->d_energy);
    if (s->d_photons)   cudaFree(s->d_photons);
    if (s->d_gather)    cudaFree(s->d_gather);
    if (s->d_cellKey)   cudaFree(s->d_cellKey);
    if (s->d_order)     cudaFree(s->d_order);
    if (s->d_cellStart) cudaFree(s->d_cellStart);
    s->arena.release();
    freeUpload(s->up);
    delete s;
}

// ============================ host: VCM / UPS session (mode U) ============================
// Resident device VCM session mirroring vcm.h's vcmPass orchestration. Each pass: (1) launch
// kVcmLight — one light subpath per pixel stores its connectible vertices into a per-path slab
// (avoids cross-thread atomics) and splats connect-to-camera contributions; (2) compact the
// slab ON DEVICE into contiguous per-path ranges (exclusive-scan the counts -> pathBegin/End,
// scatter — so strategy (c)'s same-lambda vertex connection reads its PAIRED light path);
// (3) build the uniform hash grid over the compacted vertices ON DEVICE (bbox reduce + cell
// keys + STABLE sort by cell id + vectorized lower_bound — arithmetic-exact twin of the vcm.h
// VcmGrid::build counting sort, including its output order); (4) launch kVcmCamera — one
// camera subpath per pixel does emission/NEE/connection/merge and adds this pass's radiance
// (camera result + light splat) into the persistent `accum` sum. Resolve divides accum by the
// pass count. Only ~30 bytes (vertex total + bbox) cross the PCIe bus per pass — the original
// host round trip moved the whole slab down and the compacted grid back up every pass.
// Validated statistically against the CPU (independent MC) and byte-identically against the
// host-build implementation it replaced.
struct VcmSession {
    DUpload up;
    gpu::DCamera cam{};
    int resX = 0, resY = 0;
    size_t npix = 0;
    int  diffraction = 0;
    int  maxDepth = 8;
    int  vcmCap = 8;              // max stored connectible vertices per light subpath (== maxDepth)
    int  heroC = 1;               // hero bundle width (1 == classic single-λ session)
    int  secStride = 0;           // heroC-1 secondary slots per stored vertex (0 when heroC==1)
    long long passes = 0;
    // Persistent (allocated once in Begin):
    gpu::DVcmLV* d_lvSlab = nullptr;   // npix * vcmCap
    gpu::DVcmSec* d_lvSecSlab = nullptr;  // npix*vcmCap*secStride (NULL unless heroC>1)
    int*    d_lvCount = nullptr;       // npix
    double* d_splat   = nullptr;       // npix*3 (this pass's connect-to-camera XYZ)
    double* d_accum   = nullptr;       // npix*3 (running SUM over passes)
    gpu::Real* d_lamBuf = nullptr;     // npix*heroC (per-path BUNDLE, shared light<->camera)
    double* d_invLam  = nullptr;       // npix*heroC (invPdfLambda; [i*C]<=0 marks "no wavelength")
    int*    d_pathBegin = nullptr;     // npix (device-scanned per-pass)
    int*    d_pathEnd   = nullptr;     // npix
    // Grow-only device scratch for the on-device compaction + grid build (no per-pass malloc):
    gpu::DVcmLV* d_lvCompact = nullptr; size_t lvCompactCap = 0;
    gpu::DVcmSec* d_lvSecCompact = nullptr; size_t lvSecCompactCap = 0;
    int*    d_cellKey   = nullptr;      size_t cellKeyCap = 0;
    int*    d_order     = nullptr;      size_t orderCap = 0;
    int*    d_cellStart = nullptr;      size_t cellStartCap = 0;  // entries (nCells+1)
    ThrustArena arena;                  // thrust temp storage (scan/sort/reduce)
};

bool cudaVcmSupported(const Scene& scene) {
    // VCM reuses the BDPT device scope (per-hit BSDFs, area/sphere/spot/sun lights, no
    // fluorescence/layered/env/collimated, no GRIN) PLUS a NO-media restriction: the CPU
    // vcm.h path handles surfaces only (participating media are out of mode-U scope entirely,
    // guarded by vcmUnsupportedFeature), and the device kernels place no medium vertices, so a
    // scene with any medium must stay on the CPU. Pinhole cameras only (dGenRay / cam.project);
    // the caller gates the camera.
    // Delta lights (spot / sun) render on-device since 0.127.0 — kVcmLightT emits into the
    // cone with dVC == 0, kVcmCameraT carries the per-shape NEE geometry and the escaped-ray
    // solar disc — so mode U needs no reject of its own beyond BDPT's env/collimated one.
    return cudaBdptSupported(scene) && scene.media.empty();
}

VcmSession* vcmSessionBegin(const Scene& scene, const Camera& cam, int resX, int resY,
                            bool diffraction, int maxDepth, int heroC) {
    if (!cudaAvailable() || !cudaVcmSupported(scene)) return nullptr;
    VcmSession* s = new VcmSession();
    s->resX = resX; s->resY = resY; s->npix = (size_t)resX * resY;
    s->diffraction = diffraction ? 1 : 0;
    s->maxDepth = (maxDepth > 0) ? maxDepth : 8;
    // The light-vertex slab is npix * vcmCap * sizeof(DVcmLV) (~128 B) of DEVICE memory, so
    // vcmCap == maxDepth would ask for 6.6 GB at 1100x733 and -max-bounce 64. Bound the slab
    // by a byte budget instead and say so: a smaller cap only means the deepest light-subpath
    // vertices are not STORED for merging (paths still walk to maxDepth and still connect),
    // so the estimator stays unbiased, it just loses some of the merge strategies down there.
    {
        const size_t kSlabBudget = (size_t)768 << 20;   // 768 MB
        int capByMem = (int)(kSlabBudget / (s->npix * sizeof(gpu::DVcmLV)));
        if (capByMem < 4) capByMem = 4;
        s->vcmCap = (s->maxDepth < capByMem) ? s->maxDepth : capByMem;
        if (s->vcmCap < s->maxDepth)
            std::printf("[mode U] light-vertex store capped at %d of %d vertices/subpath "
                        "(slab budget %zu MB at %dx%d)\n",
                        s->vcmCap, s->maxDepth, kSlabBudget >> 20, resX, resY);
    }
    // Hero bundle width. The kernels are templated on the SECONDARY slot count, so a C==1
    // session instantiates kVcm*T<0>: no sec slab, no per-λ arrays, no extra registers.
    int C = (heroC < 1) ? 1 : heroC;
    if (C > BDPT_NSEC + 1) C = BDPT_NSEC + 1;
    s->heroC = C;
    s->secStride = C - 1;
    buildUploadScene(scene, s->up);
    s->cam = bakeCamera(scene, cam, resX, resY, s->up);
    const size_t np = s->npix;
    CUDA_CHECK(cudaMalloc(&s->d_lvSlab,  np * (size_t)s->vcmCap * sizeof(gpu::DVcmLV)));
    if (s->secStride > 0)
        CUDA_CHECK(cudaMalloc(&s->d_lvSecSlab,
                              np * (size_t)s->vcmCap * (size_t)s->secStride * sizeof(gpu::DVcmSec)));
    CUDA_CHECK(cudaMalloc(&s->d_lvCount, np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->d_splat,   np * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_accum,   np * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_lamBuf,  np * (size_t)C * sizeof(gpu::Real)));
    CUDA_CHECK(cudaMalloc(&s->d_invLam,  np * (size_t)C * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_pathBegin, np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->d_pathEnd,   np * sizeof(int)));
    CUDA_CHECK(cudaMemset(s->d_accum, 0, np * 3 * sizeof(double)));
    return s;
}

// Run one VCM pass at the given merge `radius`.
void vcmSessionPass(VcmSession* s, double radius) {
    using namespace gpu;
    const long long passIdx = s->passes;                 // 0-based index of THIS pass
    const int W = s->resX, H = s->resY;
    const size_t np = s->npix;
    if (radius <= 0.0) radius = 1e-6;

    // Per-pass MIS constants (device twin of vcm.h PassCtx).
    DVcmCtx ctx{};
    ctx.radius = radius;
    ctx.nLightPaths = (double)np;
    double etaVCM = DPI * radius * radius * ctx.nLightPaths;
    ctx.misVcWeight = (etaVCM > 0.0) ? 1.0 / etaVCM : 0.0;
    ctx.misVmWeight = etaVCM;
    ctx.vmNorm      = (etaVCM > 0.0) ? 1.0 / etaVCM : 0.0;
    ctx.imagePlaneDist = (double)W / (2.0 * s->cam.tanHalfX);
    ctx.maxDepth = s->maxDepth;

    // (1) Light pass — zero the per-pass splat, then trace one light subpath per pixel.
    CUDA_CHECK(cudaMemset(s->d_splat, 0, np * 3 * sizeof(double)));
    unsigned long long seedL = 0xD1B54A32D192ED03ULL
                             ^ ((unsigned long long)(passIdx + 1) * 0x9E3779B97F4A7C15ULL);
    if (s->secStride > 0)
        kVcmLightT<BDPT_NSEC><<<2048, 128>>>(s->up.sc, s->cam, s->diffraction, ctx,
                                 s->d_lvSlab, s->d_lvSecSlab, s->secStride, s->heroC,
                                 s->d_lvCount, s->d_splat,
                                 s->d_lamBuf, s->d_invLam, W, H, s->vcmCap, seedL, passIdx);
    else
        kVcmLightT<0><<<2048, 128>>>(s->up.sc, s->cam, s->diffraction, ctx,
                                 s->d_lvSlab, nullptr, 0, 1,
                                 s->d_lvCount, s->d_splat,
                                 s->d_lamBuf, s->d_invLam, W, H, s->vcmCap, seedL, passIdx);
    cudaCheckKernel("vcm-light");

    // (2) Compact the slab ON DEVICE into contiguous per-path ranges: exclusive-scan the
    // per-path vertex counts into pathBegin and inclusive-scan them into pathEnd
    // (pathEnd[i] = pathBegin[i] + count[i]; integer scans are exact in any order).
    // Only the 4-byte vertex total comes back to the host.
    s->arena.reset();
    ThrustArenaAlloc tal{&s->arena};
    auto pol = FT_THRUST_PAR(tal);
    thrust::device_ptr<int> tCount(s->d_lvCount), tBegin(s->d_pathBegin), tEnd(s->d_pathEnd);
    thrust::exclusive_scan(pol, tCount, tCount + np, tBegin);
    thrust::inclusive_scan(pol, tCount, tCount + np, tEnd);
    int nLVi = 0;
    CUDA_CHECK(cudaMemcpy(&nLVi, s->d_pathEnd + (np - 1), sizeof(int), cudaMemcpyDeviceToHost));
    const size_t nLV = (size_t)((nLVi > 0) ? nLVi : 0);
    if (nLV > 0) {
        ensureDevCap(s->d_lvCompact, s->lvCompactCap, nLV);
        if (s->secStride > 0)
            ensureDevCap(s->d_lvSecCompact, s->lvSecCompactCap, nLV * (size_t)s->secStride);
        kVcmCompactScatter<<<2048, 128>>>(s->d_lvSlab, s->d_lvSecSlab, s->secStride,
                                          s->d_lvCount, s->d_pathBegin,
                                          s->d_lvCompact, s->d_lvSecCompact, (int)np, s->vcmCap);
        cudaCheckKernel("vcm-compact");
    }

    // (3) Build the uniform hash grid over the compacted light vertices ON DEVICE (twin of
    // vcm.h VcmGrid::build). The grid geometry repeats the former host math type-for-type
    // (float mins/maxes, double pads, float-subtract/double-divide cell coords), and the
    // STABLE sort by cell id equals the counting sort's output order, so kVcmCamera visits
    // vertices in the identical sequence -> bit-identical merge sums.
    double cell = radius;
    DVec3 gLo(0, 0, 0); int gnx = 1, gny = 1, gnz = 1;
    if (nLV == 0) {
        ensureDevCap(s->d_cellStart, s->cellStartCap, 2);
        CUDA_CHECK(cudaMemset(s->d_cellStart, 0, 2 * sizeof(int)));
    } else {
        BboxF bb = thrust::transform_reduce(pol,
            thrust::device_pointer_cast(s->d_lvCompact),
            thrust::device_pointer_cast(s->d_lvCompact + nLV),
            LvToBboxF{}, BboxF{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX},
            BboxMergeF{});
        gLo = DVec3(bb.mnx - cell * 0.5, bb.mny - cell * 0.5, bb.mnz - cell * 0.5);
        DVec3 ext(bb.mxx - gLo.x + cell * 0.5, bb.mxy - gLo.y + cell * 0.5, bb.mxz - gLo.z + cell * 0.5);
        gnx = std::max(1, (int)std::ceil(ext.x / cell));
        gny = std::max(1, (int)std::ceil(ext.y / cell));
        gnz = std::max(1, (int)std::ceil(ext.z / cell));
        const long long nCells = (long long)gnx * gny * gnz;
        ensureDevCap(s->d_cellKey, s->cellKeyCap, nLV);
        ensureDevCap(s->d_order,   s->orderCap,   nLV);
        kVcmCellKey<<<2048, 128>>>(s->d_lvCompact, (int)nLV, gLo, cell, gnx, gny, gnz,
                                   s->d_cellKey);
        cudaCheckKernel("vcm-cellkey");
        thrust::device_ptr<int> tKey(s->d_cellKey), tOrd(s->d_order);
        thrust::sequence(pol, tOrd, tOrd + nLV);
        thrust::stable_sort_by_key(pol, tKey, tKey + nLV, tOrd);
        ensureDevCap(s->d_cellStart, s->cellStartCap, (size_t)nCells + 1);
        thrust::lower_bound(pol, tKey, tKey + nLV,
                            thrust::counting_iterator<int>(0),
                            thrust::counting_iterator<int>((int)(nCells + 1)),
                            thrust::device_pointer_cast(s->d_cellStart));
    }

    DVcmGrid grid{};
    grid.lv = (nLV > 0) ? s->d_lvCompact : nullptr;
    grid.nLV = (int)nLV;
    grid.cellStart = s->d_cellStart;
    grid.order = (nLV > 0) ? s->d_order : nullptr;
    grid.lo = gLo; grid.cell = cell; grid.nx = gnx; grid.ny = gny; grid.nz = gnz;

    // (5) Camera pass — one camera subpath per pixel; adds this pass's radiance into accum.
    unsigned long long seedC = 0xC2B2AE3D27D4EB4FULL
                             ^ ((unsigned long long)(passIdx + 1) * 0xA24BAED4963EE407ULL);
    if (s->secStride > 0)
        kVcmCameraT<BDPT_NSEC><<<2048, 128>>>(s->up.sc, s->cam, s->diffraction, ctx, grid,
                              (nLV > 0) ? s->d_lvSecCompact : nullptr, s->secStride, s->heroC,
                              s->d_pathBegin, s->d_pathEnd, s->d_splat, s->d_accum,
                              s->d_lamBuf, s->d_invLam, W, H, seedC, passIdx);
    else
        kVcmCameraT<0><<<2048, 128>>>(s->up.sc, s->cam, s->diffraction, ctx, grid,
                              nullptr, 0, 1,
                              s->d_pathBegin, s->d_pathEnd, s->d_splat, s->d_accum,
                              s->d_lamBuf, s->d_invLam, W, H, seedC, passIdx);
    cudaCheckKernel("vcm-camera");

    s->passes += 1;
}

// Resolve the running average image (accum / passes) into `out`, exactly like vcmResolve.
void vcmSessionResolve(VcmSession* s, Film& out) {
    const size_t np = s->npix;
    std::vector<double> accum(np * 3);
    CUDA_CHECK(cudaMemcpy(accum.data(), s->d_accum, np * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    if (out.resX != s->resX || out.resY != s->resY || out.xyz.empty()) {
        out.resX = s->resX; out.resY = s->resY; out.alloc();
    }
    double inv = (s->passes > 0) ? 1.0 / (double)s->passes : 0.0;
    for (size_t i = 0; i < np; ++i) {
        out.xyz[i]  = Vec3(accum[i * 3 + 0] * inv, accum[i * 3 + 1] * inv, accum[i * 3 + 2] * inv);
        out.hits[i] = 1.0;
    }
}

long long vcmSessionPasses(const VcmSession* s) { return s ? s->passes : 0; }

void vcmSessionEnd(VcmSession* s) {
    if (!s) return;
    cudaFree(s->d_lvSlab); cudaFree(s->d_lvCount); cudaFree(s->d_splat); cudaFree(s->d_accum);
    cudaFree(s->d_lamBuf); cudaFree(s->d_invLam); cudaFree(s->d_pathBegin); cudaFree(s->d_pathEnd);
    if (s->d_lvSecSlab)    cudaFree(s->d_lvSecSlab);
    if (s->d_lvSecCompact) cudaFree(s->d_lvSecCompact);
    if (s->d_lvCompact) cudaFree(s->d_lvCompact);
    if (s->d_cellKey)   cudaFree(s->d_cellKey);
    if (s->d_cellStart) cudaFree(s->d_cellStart);
    if (s->d_order)     cudaFree(s->d_order);
    s->arena.release();
    freeUpload(s->up);
    delete s;
}
