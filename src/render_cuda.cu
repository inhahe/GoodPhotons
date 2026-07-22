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
#include <math.h>
#include <float.h>

#include "render_cuda.h"
#include "render_progress.h"
#include "grin.h"         // grin::sceneHasGrin (host gate mirrored into DScene::hasGrin)
#include "photonmap.h"    // host PhotonMap::build reused for the mode-M grid (GPU gather)
#include "raster.h"       // G2 iso preview: shared deriveLight/materialColor/exposeAndEncode (host)

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
    // Indexed component access (0=x,1=y,2=z) for the watertight tri test's axis
    // permutation. No bounds check (hot path); callers pass 0..2.
    HD Real  operator[](int i) const { return (&x)[i]; }
    HD Real& operator[](int i)       { return (&x)[i]; }
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
// Values MUST match MatType (scene.h) 1:1 — the upload does d.type = (int)m.type. D_LAYERED
// (MatType::Layered) has no device branch (Layered scenes fall back to the CPU tracer via
// cudaForwardSupported), but the placeholder keeps D_DIFFUSETRANSMIT aligned at index 11.
enum { D_DIFFUSE=0, D_DIELECTRIC, D_MIRROR, D_HALFMIRROR, D_GLOSSY, D_FLUORESCENT, D_THINFILM,
       D_GRATING, D_MIX, D_MULTILAYER, D_LAYERED, D_DIFFUSETRANSMIT, D_FILTER };

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

struct DTri    { DVec3 v0, v1, v2, gn; DVec3 uv0, uv1, uv2; DVec3 n0, n1, n2; int matId, sensorId; };
struct DSphere { DVec3 c; double r; int matId; };
struct DNode   { DVec3 lo, hi; int left, right, first, count; };

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
    // --- Optional imported .nvdb volume baked to a dense grid (mirrors VdbGrid) ---
    // When `vdbData` is non-null the density multiplier is TRILINEARLY sampled from
    // this uploaded dense lattice instead of the pattern VM; takes precedence.
    const float*     vdbData;         // nx*ny*nz values, index [(k*ny+j)*nx+i] (or null)
    int              vdbNx, vdbNy, vdbNz;
    double           vdbAinv[9];      // world->index linear map (row-major 3x3)
    DVec3            vdbW0;            // world position of index origin (0,0,0)
    DVec3            vdbImin;          // integer min-corner of the baked lattice
    int              bounded;         // 1 => clip to the bound region
    int              boundShape;      // 0 => box [bmin,bmax], 1 => sphere, 2 => implicit field
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
    // --- Optional gradient-index (GRIN) refractive field n(x,y,z) (mirrors host Medium) ---
    // When `iorN > 0` this region bends rays along the Eikonal ray equation; the forward
    // megakernel/wavefront march through it (dGrinMarch) before each bounce's closestHit.
    const PatNode*    ior;            // compiled n(x,y,z) program (device pool) or null
    int               iorN;           // node count of the ior program (0 => not GRIN)
    double            iorStep;        // Eikonal march step in world units (>0 for GRIN)
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

// One scalar-record stop on the device (§records stage 6b): its domain position + a slice
// of the shared recDrivers PatNode pool holding the stop's per-hit expression program.
struct DRecScalarStop { double pos; int exprOff; int exprN; };

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
    // FP32 mirrors of fieldNodes/fieldExprNodes (same order/offsets) for the sphere-
    // trace march hot path. Null iff the double pools are null.
    const DFieldNodeF* fieldNodesF;
    const PatNodeF*    fieldExprNodesF;
    const DImplicit*  implicits; int nImplicits;
    // Instancing (two-level BVH). BVH prims with index >= nTris+nSph+nImplicits map to
    // instances[prim - nTris - nSph - nImplicits]; each instance references a DBlas
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
    const double*    lightCdfAll;   // flattened per-emitter wavelength CDFs
    const DTexture*  textures; int nTex;   // reflectance textures (mat.reflectTex)
    const double*    fluoCdfAll;    // flattened per-material fluorescence emission CDFs
    // BDPT shared wavelength sampler (mirrors Scene::emitSampler): the combined
    // g(lambda)=sum_k geomWeight_k*SPD_k CDF, its bin step, and emitG = its integral.
    // BDPT samples one shared lambda per sample from this and sets invPdfLambda=1/pdf.
    const double*    emitSamplerCdf; int emitSamplerN; double emitSamplerStep;
    double           emitG;
    const DMedium*   media;    // participating media array (superposed); null if none
    int              mediaN;   // number of media (0 => vacuum)
    int              hasGrin;  // 1 => some enabled medium carries an `ior` (GRIN) field.
                               // Gates the per-bounce Eikonal march (grin::sceneHasGrin twin);
                               // 0 keeps ordinary scenes bit-identical (march never entered).
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

// dPatternEval / dFieldEval are defined further down; forward-declare for the density
// evaluator (the implicit-bound membership test needs the field VM).
__device__ static double dPatternEval(const PatNode* nodes, int n,
                                      double x, double y, double z, double f,
                                      double nx, double ny, double nz, double r,
                                      double u, double v);
__device__ static double dFieldEval(const DFieldNode* nodes, int n,
                                    double pwx, double pwy, double pwz,
                                    const PatNode* exprPool);

// Dimensionless density multiplier at a world point (>= 0). Device twin of
// Medium::densityAt: the shared pattern VM with x y z r live (f/normal/uv read 0).
// For an implicit bound the multiplier is 0 outside the field (medium absent there).
__device__ static double dMedDensityAt(const DMedium& m, const DVec3& p) {
    if (m.boundShape == 2 && m.boundField) {   // implicit-field membership carve-out
        double f = dFieldEval(m.boundField, m.boundFieldN, p.x, p.y, p.z, m.boundFieldExpr);
        bool inside = m.boundInsideNeg ? (f < 0.0) : (f > 0.0);
        if (!inside) return 0.0;
    }
    // Imported .nvdb volume: trilinearly sample the uploaded dense grid (device twin
    // of VdbGrid::sample). Takes precedence over the pattern-VM density formula.
    if (m.vdbData) {
        double rx = (double)p.x - m.vdbW0.x, ry = (double)p.y - m.vdbW0.y, rz = (double)p.z - m.vdbW0.z;
        double fi = m.vdbAinv[0]*rx + m.vdbAinv[1]*ry + m.vdbAinv[2]*rz - m.vdbImin.x;
        double fj = m.vdbAinv[3]*rx + m.vdbAinv[4]*ry + m.vdbAinv[5]*rz - m.vdbImin.y;
        double fk = m.vdbAinv[6]*rx + m.vdbAinv[7]*ry + m.vdbAinv[8]*rz - m.vdbImin.z;
        int nx = m.vdbNx, ny = m.vdbNy, nz = m.vdbNz;
        if (fi < -0.5 || fj < -0.5 || fk < -0.5 ||
            fi > nx - 0.5 || fj > ny - 0.5 || fk > nz - 0.5) return 0.0;
        double ffi = floor(fi), ffj = floor(fj), ffk = floor(fk);
        int i0 = (int)ffi, j0 = (int)ffj, k0 = (int)ffk;
        auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
        int i0c = cl(i0, nx-1), i1c = cl(i0+1, nx-1);
        int j0c = cl(j0, ny-1), j1c = cl(j0+1, ny-1);
        int k0c = cl(k0, nz-1), k1c = cl(k0+1, nz-1);
        double tx = fi - ffi, ty = fj - ffj, tz = fk - ffk;
        tx = tx < 0 ? 0 : (tx > 1 ? 1 : tx);
        ty = ty < 0 ? 0 : (ty > 1 ? 1 : ty);
        tz = tz < 0 ? 0 : (tz > 1 ? 1 : tz);
        const float* D = m.vdbData;
        auto AT = [&](int i, int j, int k) -> double {
            return (double)D[((size_t)k * ny + j) * nx + i];
        };
        double c00 = AT(i0c,j0c,k0c)*(1-tx) + AT(i1c,j0c,k0c)*tx;
        double c10 = AT(i0c,j1c,k0c)*(1-tx) + AT(i1c,j1c,k0c)*tx;
        double c01 = AT(i0c,j0c,k1c)*(1-tx) + AT(i1c,j0c,k1c)*tx;
        double c11 = AT(i0c,j1c,k1c)*(1-tx) + AT(i1c,j1c,k1c)*tx;
        double c0 = c00*(1-ty) + c10*ty, c1 = c01*(1-ty) + c11*ty;
        double v = c0*(1-tz) + c1*tz;
        return v > 0.0 ? v : 0.0;
    }
    if (!m.heterogeneous || !m.density) return 1.0;
    double r = sqrt((double)p.x * p.x + (double)p.y * p.y + (double)p.z * p.z);
    double d = dPatternEval(m.density, m.densityN, p.x, p.y, p.z, 0.0,
                            0.0, 0.0, 0.0, r, 0.0, 0.0);
    return d > 0.0 ? d : 0.0;
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
__device__ static bool dMedInside(const DMedium& m, const DVec3& p) {
    if (!m.bounded) return true;
    if (m.boundShape == 1) {   // sphere
        double dx = (double)p.x - m.bcenter.x, dy = (double)p.y - m.bcenter.y,
               dz = (double)p.z - m.bcenter.z;
        return dx * dx + dy * dy + dz * dz <= m.bradius * m.bradius;
    }
    if (m.boundShape == 2 && m.boundField) {   // implicit field
        double f = dFieldEval(m.boundField, m.boundFieldN, p.x, p.y, p.z, m.boundFieldExpr);
        return m.boundInsideNeg ? (f < 0.0) : (f > 0.0);
    }
    return p.x >= m.bmin.x && p.x <= m.bmax.x && p.y >= m.bmin.y &&
           p.y <= m.bmax.y && p.z >= m.bmin.z && p.z <= m.bmax.z;
}

// Local refractive index n at a world point (device twin of Medium::nAt): the shared
// pattern VM with x y z r live, floored at 1e-3. 1.0 when this medium is not GRIN.
__device__ static double dMedNAt(const DMedium& m, const DVec3& p) {
    if (m.iorN <= 0 || !m.ior) return 1.0;
    double r = sqrt((double)p.x * p.x + (double)p.y * p.y + (double)p.z * p.z);
    double n = dPatternEval(m.ior, m.iorN, p.x, p.y, p.z, 0.0,
                            0.0, 0.0, 0.0, r, 0.0, 0.0);
    return n > 1e-3 ? n : 1e-3;
}

// ∇n at a world point via central differences with step h (device twin of Medium::gradNAt).
__device__ static DVec3 dMedGradN(const DMedium& m, const DVec3& p, double h) {
    double inv = 0.5 / h;
    DVec3 xp = p, xm = p; xp.x = (Real)(p.x + h); xm.x = (Real)(p.x - h);
    DVec3 yp = p, ym = p; yp.y = (Real)(p.y + h); ym.y = (Real)(p.y - h);
    DVec3 zp = p, zm = p; zp.z = (Real)(p.z + h); zm.z = (Real)(p.z - h);
    double gx = dMedNAt(m, xp) - dMedNAt(m, xm);
    double gy = dMedNAt(m, yp) - dMedNAt(m, ym);
    double gz = dMedNAt(m, zp) - dMedNAt(m, zm);
    return DVec3{ (Real)(gx * inv), (Real)(gy * inv), (Real)(gz * inv) };
}

// (dGrinMarch — the Eikonal ray marcher — needs DHit + closestHit, so it is defined just
// after the closestHit definition below.)

// Sample the next real collision along (o,dir) within [0,dMax]. Device twin of
// Renderer::sampleMediumCollision: exact analytic free-flight (one draw) for a
// homogeneous medium (bit-identical to before), else delta (Woodcock) tracking.
__device__ static bool dMedSampleCollision(const DMedium& m, const DVec3& o, const DVec3& dir,
                                           Real dMax, Real lambda, DRng& rng, Real& tHit) {
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
        double sigT = stBase * dMedDensityAt(m, pp);
        if ((double)rng.uniform() * sigMax < sigT) { tHit = (Real)t; return true; }
    }
}

// Unbiased transmittance along [o, o+dir*dist]. Device twin of
// Renderer::mediumTransmittance: exact exp for a homogeneous medium (no RNG draw), else
// ratio tracking. Homogeneous scenes therefore keep the exact analytic transmittance.
__device__ static Real dMedTransmittance(const DMedium& m, const DVec3& o, const DVec3& dir,
                                         Real dist, Real lambda, DRng& rng) {
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
        double sigT = stBase * dMedDensityAt(m, pp);
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
__device__ static bool dMediaSampleCollision(const DMedium* media, int n, const DVec3& o,
                                             const DVec3& dir, Real dMax, Real lambda,
                                             DRng& rng, Real& tHit, int& whichMed) {
    Real best = dMax; int which = -1;
    for (int i = 0; i < n; ++i) {
        Real t;
        if (dMedSampleCollision(media[i], o, dir, dMax, lambda, rng, t) && t < best) {
            best = t; which = i;
        }
    }
    if (which < 0) return false;
    tHit = best; whichMed = which; return true;
}

__device__ static Real dMediaTransmittance(const DMedium* media, int n, const DVec3& o,
                                           const DVec3& dir, Real dist, Real lambda, DRng& rng) {
    Real Tr = (Real)1;
    for (int i = 0; i < n; ++i) {
        Tr *= dMedTransmittance(media[i], o, dir, dist, lambda, rng);
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
// Forward decl: DF_EXPR leaves evaluate their formula with the pattern VM, which is
// defined further down (dPatternEval). The field VM only needs it for the Expr case.
__device__ static double dPatternEval(const PatNode* nodes, int n,
                                      double x, double y, double z, double f,
                                      double nx, double ny, double nz, double r,
                                      double u, double v);
__device__ static float dPatternEvalF(const PatNodeF* nodes, int n,
                                      float x, float y, float z, float f,
                                      float nx, float ny, float nz, float r,
                                      float u, float v);

__device__ static double dFieldLeafSDF(const DFieldNode& nd, double px, double py, double pz,
                                       const PatNode* exprPool) {
    switch (nd.op) {
        case DF_SPHERE:
            return sqrt(px*px + py*py + pz*pz) - nd.p[0];
        case DF_EXPR: {   // arbitrary formula f(x,y,z); r=|p|, other vars (f/normals) are 0
            if (!exprPool) return BIG;
            double r = sqrt(px*px + py*py + pz*pz);
            return dPatternEval(exprPool + nd.exprOff, nd.exprN, px, py, pz, 0.0,
                                0.0, 0.0, 0.0, r, 0.0, 0.0);
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
// ---- FP32 twins of the field VM (see DFieldNodeF): used ONLY by the sphere-trace
// march/refine in intersectImplicit, where the result is stored in float anyway.
__device__ static float dFieldLeafSDFF(const DFieldNodeF& nd, float px, float py, float pz,
                                       const PatNodeF* exprPool) {
    switch (nd.op) {
        case DF_SPHERE:
            return sqrtf(px*px + py*py + pz*pz) - nd.p[0];
        case DF_EXPR: {   // arbitrary formula f(x,y,z); r=|p|, other vars (f/normals) are 0
            if (!exprPool) return (float)BIG;
            float r = sqrtf(px*px + py*py + pz*pz);
            return dPatternEvalF(exprPool + nd.exprOff, nd.exprN, px, py, pz, 0.0f,
                                 0.0f, 0.0f, 0.0f, r, 0.0f, 0.0f);
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
                                    const PatNodeF* exprPool) {
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
                st[sp++] = dFieldLeafSDFF(nd, plx, ply, plz, exprPool) * nd.scale;
            }
        }
    }
    return sp > 0 ? st[0] : (float)BIG;
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
__device__ static bool intersectImplicit(const DScene& sc, const DImplicit& im,
                                          const DVec3& roR, const DVec3& rdR, Real tmin, DHit& hit) {
    double ox = roR.x, oy = roR.y, oz = roR.z, dx = rdR.x, dy = rdR.y, dz = rdR.z;

    const DFieldNode* nd = sc.fieldNodes + im.nodeOff;   // double pool: gradient/normal only
    const PatNode* exprPool = sc.fieldExprNodes;
    // FP32 mirror pools: the march + root refine run entirely in float (the committed
    // hit is float anyway; FP64 VM ops serialize on the 1/64-rate FP64 pipe).
    const DFieldNodeF* ndF = sc.fieldNodesF + im.nodeOff;
    const PatNodeF* exprPoolF = sc.fieldExprNodesF;
    const int N = im.nodeN;

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
        double pex = ox + dx*tEnter, pey = oy + dy*tEnter, pez = oz + dz*tEnter;
        double pxx = ox + dx*tExit,  pxy = oy + dy*tExit,  pxz = oz + dz*tExit;
        double gex = pex - im.sphereCenter[0], gey = pey - im.sphereCenter[1], gez = pez - im.sphereCenter[2];
        double gxx = pxx - im.sphereCenter[0], gxy = pxy - im.sphereCenter[1], gxz = pxz - im.sphereCenter[2];
        double le = sqrt(gex*gex + gey*gey + gez*gez), lx = sqrt(gxx*gxx + gxy*gxy + gxz*gxz);
        if (le > 0.0) { neX = gex/le; neY = gey/le; neZ = gez/le; } else neZ = 1.0;
        if (lx > 0.0) { nxX = gxx/lx; nxY = gxy/lx; nxZ = gxz/lx; } else nxZ = 1.0;
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
        return true;
    };

    float t = (float)t0;
    float f = dFieldEvalF(ndF, N, oxF + dxF*t, oyF + dyF*t, ozF + dzF*t, exprPoolF);
    // NEAR CAP: ray enters the container already inside the solid (f<0); the container
    // face is the nearest surface. `open` skips this to reveal the cut edge.
    if (capped && tEnter >= tmin && tEnter < (double)hit.t && f < 0.0f)
        return writeHit(tEnter, ox + dx*tEnter, oy + dy*tEnter, oz + dz*tEnter, neX, neY, neZ);
    for (int i = 0; i < MAX_STEP; ++i) {
        float step = sampleMode ? fixedStepF : fmaxf(fabsf(f) * invLipF, minStepF) / dlenF;
        float tn = t + step;
        bool last = false;
        if (tn >= t1F) { tn = t1F; last = true; }
        float fn = dFieldEvalF(ndF, N, oxF + dxF*tn, oyF + dyF*tn, ozF + dzF*tn, exprPoolF);
        bool crossed = (f > 0.0f && fn <= 0.0f) || (f < 0.0f && fn >= 0.0f) || (f == 0.0f && fn != 0.0f);
        if (crossed) {
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
                float fm = dFieldEvalF(ndF, N, oxF + dxF*tm, oyF + dyF*tm, ozF + dzF*tm, exprPoolF);
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
            double px = ox + dx*th, py = oy + dy*th, pz = oz + dz*th;
            double eps = fmax(1e-6, 1e-4*th);
            double gx, gy, gz; dFieldGradient(nd, N, px, py, pz, eps, gx, gy, gz, exprPool);
            return writeHit(th, px, py, pz, gx, gy, gz);
        }
        if (last) {
            // FAR CAP: reached the container exit still inside the solid (fn<0), and the
            // far clip is the container itself — seal the sawn-off solid.
            if (capped && exitIsContainer && fn < 0.0f && tExit >= tmin && tExit < (double)hit.t)
                return writeHit(tExit, ox + dx*tExit, oy + dy*tExit, oz + dz*tExit, nxX, nxY, nxZ);
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
    Real U = Cx * By - Cy * Bx;
    Real V = Ax * Cy - Ay * Cx;
    Real W = Bx * Ay - By * Ax;
    // Exact-zero fallback in double (helps the float path land a grazing edge on one side).
    if (U == 0 || V == 0 || W == 0) {
        if (U == 0) U = (Real)((double)Cx * (double)By - (double)Cy * (double)Bx);
        if (V == 0) V = (Real)((double)Ax * (double)Cy - (double)Ay * (double)Cx);
        if (W == 0) W = (Real)((double)Bx * (double)Ay - (double)By * (double)Ax);
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
    hit.ng = tri.gn;
    hit.matId = tri.matId; hit.sensorId = tri.sensorId;
    hit.u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
    hit.v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
    DVec3 ns = tri.n0 * b0 + tri.n1 * b1 + tri.n2 * b2;
    Real nl = dot(ns, ns);
    ns = (nl > (Real)1e-18) ? ns * ((Real)1 / sqrt(nl)) : tri.gn;
    hit.n = (dot(rd, ns) < 0) ? ns : -ns;
    return true;
}
// Interface-preserving wrapper (builds the shear inline) for any one-off caller.
__device__ static inline bool intersectTri(const DVec3& ro, const DVec3& rd, const DTri& tri,
                                            Real tmin, DHit& hit) {
    return intersectTri(makeTriShear(rd), ro, rd, tri, tmin, hit);
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
    if (inst.matOverride >= 0) lh.matId = inst.matOverride;
}

__device__ static DHit closestHit(const DScene& sc, const DVec3& ro, const DVec3& rd,
                                   Real tmin = RAY_EPS) {
    DHit h; h.t = BIG; h.valid = false; h.matId = 0; h.sensorId = -1;
    if (sc.nNodes == 0) return h;
    DVec3 invD{(Real)1 / rd.x, (Real)1 / rd.y, (Real)1 / rd.z};
    const DTriShear sh = makeTriShear(rd);
    Real tMax = BIG;
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
                else {
                    // Instance leaf: transform the ray into BLAS-local space, walk the
                    // shared sub-BVH, and map any closer hit back to world space.
                    const DInstance& inst = sc.instances[prim - sc.nTris - sc.nSph - sc.nImplicits];
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
    return h;
}

// Advance (ro,rd) through any GRIN region(s) via symplectic Eikonal marching, stopping when
// a surface is within one step or the ray has left all GRIN regions. Device twin of
// grin::march — the forward megakernel/wavefront call it before each bounce's closestHit,
// gated by sc.hasGrin so ordinary scenes never enter it (bit-identical). Kept byte-for-byte
// in step with the CPU marcher (grin.h) so CPU and GPU bend rays identically.
__device__ static void dGrinMarch(const DScene& sc, DVec3& ro, DVec3& rd) {
    const int GRIN_MAX_STEPS = 200000;
    for (int gstep = 0; gstep < GRIN_MAX_STEPS; ++gstep) {
        // GRIN region containing ro (first enabled GRIN membership), or -1.
        int gm = -1;
        for (int mi = 0; mi < sc.mediaN; ++mi) {
            const DMedium& md = sc.media[mi];
            if (md.enabled && md.iorN > 0 && dMedInside(md, ro)) { gm = mi; break; }
        }
        DHit hs = closestHit(sc, ro, rd);
        double dS = hs.valid ? (double)hs.t : 1e30;
        if (gm < 0) {
            // Outside any GRIN region: jump to the nearest GRIN entry before the next
            // surface, else stop (straight-ray body takes over).
            double bestTa = 1e30; int bestM = -1;
            for (int mi = 0; mi < sc.mediaN; ++mi) {
                const DMedium& md = sc.media[mi];
                if (!(md.enabled && md.iorN > 0)) continue;
                double ta, tb;
                if (dMedClip(md, ro, rd, 1e-4, dS, ta, tb) && ta < bestTa) { bestTa = ta; bestM = mi; }
            }
            if (bestM < 0) break;
            ro = ro + rd * (Real)(bestTa + 1e-4);   // nudge inside
            continue;
        }
        const DMedium& g = sc.media[gm];
        double ds = g.iorStep;
        if (hs.valid && (double)hs.t <= ds) break;   // surface within a step
        // Symplectic Eikonal step with optical direction T = n·d (|T| = n):
        //   T += ∇n·ds ;  x += (T/n)·ds ;  d = T/|T|.
        double n0 = dMedNAt(g, ro);
        DVec3 grad = dMedGradN(g, ro, 0.5 * ds);
        DVec3 T = rd * (Real)n0 + grad * (Real)ds;
        DVec3 newPos = ro + T * (Real)(ds / n0);
        double tl = sqrt((double)dot(T, T));
        DVec3 newDir = (tl > 1e-12) ? T * (Real)(1.0 / tl) : rd;
        ro = newPos; rd = newDir;
    }
}

__device__ static bool occluded(const DScene& sc, const DVec3& o, const DVec3& dir,
                                 Real maxDist, Real tmin = RAY_EPS) {
    if (sc.nNodes == 0) return false;
    DVec3 invD{(Real)1 / dir.x, (Real)1 / dir.y, (Real)1 / dir.z};
    const DTriShear sh = makeTriShear(dir);
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
                else if (prim < sc.nTris + sc.nSph + sc.nImplicits) blocked = intersectImplicit(sc, sc.implicits[prim - sc.nTris - sc.nSph], o, dir, tmin, h);
                else {
                    // Instance leaf: any-hit inside the shared BLAS in local space.
                    const DInstance& inst = sc.instances[prim - sc.nTris - sc.nSph - sc.nImplicits];
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
                                         DVec3& ro, DVec3& rd, bool* transmitted = nullptr,
                                         Real extIor = (Real)1) {
    Real ng = specLookup(m.ior, lambda);
    bool entering = dot(d, h.ng) < 0;
    DVec3 nl = entering ? h.ng : -h.ng;
    Real n1 = entering ? extIor : ng, n2 = entering ? ng : extIor;
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

// Nested-dielectric PRIORITY step (Schmidt & Budge 2002), shared by every device transport
// loop. Resolves the exterior IOR at a dielectric interface from the enclosing medium (the
// highest-priority stack entry), suppresses lower-priority overlapping boundaries (straight
// pass-through), and maintains `stk`. `mi` is the resolved material index (Mix/Layered
// aware). SAFE FALLBACK: the priority rule applies only when BOTH sides carry an explicit
// priority (air always counts, IOR 1.0); otherwise this degrades to the old flat
// air<->glass model, so priority-free scenes render bit-identically. Mirrors the host.
__device__ static void dDielectricStep(const DScene& sc, const DMaterial& m, const DHit& h,
                                        const DVec3& d, Real lambda, DRng& rng,
                                        int mi, DMediumStack& stk, DVec3& outO, DVec3& outD) {
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
        refractOrReflect(sc, m, h, d, lambda, rng, nro, nrd, &transmitted, extIor);
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
        refractOrReflect(sc, m, h, d, lambda, rng, nro, nrd, &transmitted, extIor);
        if (transmitted) stk.popMat(mi);                // TIR stays inside mi
        outO = nro; outD = nrd;
    }
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
    if (sc.mediaN > 0) contrib *= dMediaTransmittance(sc.media, sc.mediaN, p, wdir, dist, lambda, rng);
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
    Real ph = hgPhase(dot(wIn, wdir), (Real)med.g);
    Real Lambda = medAlbedo(med, lambda);
    // Projection-general splat (no cosSurf for a volume vertex): the phase carries the
    // scattering; normalise by dist^2 * pixelSolidAngle (rectilinear or fisheye).
    double solidAngle = cam.pixelSolidAngle(cosCam);
    Real contrib = beta * Lambda * ph / (Real)((double)dist2 * solidAngle);
    contrib *= dMediaTransmittance(sc.media, sc.mediaN, p, wdir, dist, lambda, rng);
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
    if (sc.mediaN > 0) contrib *= dMediaTransmittance(sc.media, sc.mediaN, p, wdir, dist, lambda, rng);
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
    Real ph = hgPhase(dot(wIn, wdir), (Real)med.g);
    Real Lambda = medAlbedo(med, lambda);
    Real contrib = beta * Lambda * ph * cosLens * (Real)DPI * (R * R) / (dist * dist);
    // Same flux->film-irradiance normaliser as connectLens (see there); per-camera
    // constant, so auto-exposed scenes are unaffected.
    contrib *= (Real)1 / (Real)(cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
    contrib *= dMediaTransmittance(sc.media, sc.mediaN, p, wdir, dist, lambda, rng);
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
// One deposited photon (device twin of Photon in photonmap.h): where light landed, the
// incident direction (= -travel), the shading normal (for cross-surface leak rejection),
// and the monochromatic power / wavelength it carried. Laid out to round-trip through the
// host PhotonMap (float here <-> double on the host build).
struct DPhoton {
    DVec3 pos, wi, n;
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
};

// Gather-tuned photon record: what the mode-M density-estimate kernel actually reads.
// The deposit-side DPhoton carries (pos, wi, n, power, lambda), but the gather never
// reads wi, and it weighted every visited photon by cie{X,Y,Z}(lambda_p) * power *
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
// Append a photon record at a diffuse / translucent vertex (device twin of
// Renderer::depositPhoton). No-op unless the camera set carries a deposit buffer
// (normal renders leave cs.depCount == nullptr, so this compiles away to nothing).
// Always increments the atomic count (so a null-buffer sizing pass measures the exact
// deposit total); stores only when a buffer is bound and the slot is within capacity.
__device__ static void depositPhoton(const DCamSet& cs, const DVec3& p, const DVec3& wtravel,
                                     const DVec3& n, Real beta, Real lambda) {
    if (!cs.depCount) return;
    unsigned long long i = atomicAdd(cs.depCount, 1ULL);
    if (cs.depPhotons && i < cs.depCap) {
        DPhoton ph;
        ph.pos = p; ph.wi = -wtravel; ph.n = n;
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
    // Throughput at the vertex for a connection leaving toward `wP` (unit, toward the
    // sphere). Returns <0 to signal "reject" (camera-side behind a surface).
    __device__ double term(const D3& wP) const {
        if (volume) return weight * (double)hgPhase((Real)d3dot(wIn, wP), (Real)g);
        double cosSurf = d3dot(np, wP);
        return cosSurf <= 0.0 ? -1.0 : (weight / DPI) * cosSurf;
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
        if (occluded(sc, (p + wP * 1e-6).toR(), wPR, (Real)(dP - 2e-6))) continue;
        if (sc.mediaN > 0)
            contrib *= (double)dMediaTransmittance(sc.media, sc.mediaN, p.toR(), wPR, (Real)dP, lambda, rng);

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
        if (occluded(sc, (p + wP * 1e-6).toR(), wPR, (Real)(dP2 - 2e-6))) continue;
        D3 wE = eye - ch.P1; double dE = d3len(wE); wE = wE * (1.0 / dE);
        DVec3 wER = wE.toR();
        if (occluded(sc, (ch.P1 + wE * 1e-6).toR(), wER, (Real)(dE - 2e-6))) continue;

        if (sc.mediaN > 0) {
            contrib *= (double)dMediaTransmittance(sc.media, sc.mediaN, p.toR(),   wPR, (Real)dP2, lambda, rng);
            contrib *= (double)dMediaTransmittance(sc.media, sc.mediaN, ch.P1.toR(), wER, (Real)dE, lambda, rng);
        }
        filmAdd(film, hits, cam.resX, px, py, lambda, (Real)contrib);
    }
}

// Splat vertex p to every camera through every smooth dielectric sphere (the
// refracted image of p). Device twin of Renderer::camSpecularSplatAllVtx. Mode B only.
__device__ static void camSpecularSplatAllVtx(const DScene& sc, const DCamSet& cs, int camMode,
                                              const D3& pd, const DSpecVtx& vt, Real lambda,
                                              Real beta, DRng& rng) {
    if (camMode != CAM_B) return;
    for (int si = 0; si < sc.nSph; ++si) {
        const DSphere& S = sc.sph[si];
        const DMaterial& gm = sc.mats[S.matId];
        if (gm.type != D_DIELECTRIC) continue;
        double ng = (double)specLookup(gm.ior, lambda);
        for (int c = 0; c < cs.nCam; ++c)
            dConnectSpecularSphere(sc, cs.cams[c], cs.films[c], cs.hits[c], S, gm, ng,
                                   pd, vt, lambda, (double)beta, rng);
    }
}
// Surface vertex: refract the Lambertian reflection of p through every glass sphere.
__device__ static void camSpecularSplatAll(const DScene& sc, const DCamSet& cs, int camMode,
                                           const DVec3& p, const DVec3& n, Real lambda,
                                           Real beta, Real rho, DRng& rng) {
    DSpecVtx vt; vt.volume = false; vt.np = D3(n); vt.weight = (double)rho; vt.g = 0;
    camSpecularSplatAllVtx(sc, cs, camMode, D3(p), vt, lambda, beta, rng);
}
// Volume vertex: refract the fog in-scatter at p through every glass sphere, so the
// glowing haze itself bends through the glass the camera flies through.
__device__ static void camSpecularSplatVolumeAll(const DScene& sc, const DMedium& med,
                                                 const DCamSet& cs, int camMode, const DVec3& p,
                                                 const DVec3& wIn, Real lambda, Real beta, DRng& rng) {
    DSpecVtx vt; vt.volume = true; vt.wIn = D3(wIn); vt.g = med.g;
    vt.weight = (double)medAlbedo(med, lambda);
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
                                      double nx, double ny, double nz, double r,
                                      double u, double v) {
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
            case PatOp::VarU:     st[sp++] = u;  break;
            case PatOp::VarV:     st[sp++] = v;  break;
            case PatOp::VarT:     st[sp++] = 0.0; break;  // flyby timeline: never in scope on-device (camera_curve exprs are consumed at load)
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
            case PatOp::PovFn: {
                int id = (int)nd.a;
                int na = povFnArity(id);
                double args[POV_FN_MAX_ARGS];
                for (int k = na - 1; k >= 0; --k) args[k] = st[--sp];
                st[sp++] = povFnEval(id, args);
                break;
            }
        }
    }
    return sp > 0 ? st[0] : 0.0;
}
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
                                      float u, float v) {
    float st[64]; int sp = 0;
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
            case PatOp::PovFn: {   // POV internals are double-only: promote args, demote result
                int id = (int)nd.a;
                int na = povFnArity(id);
                double args[POV_FN_MAX_ARGS];
                for (int k = na - 1; k >= 0; --k) args[k] = (double)st[--sp];
                st[sp++] = (float)povFnEval(id, args);
                break;
            }
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
                        h.n.x, h.n.y, h.n.z, r, h.u, h.v);
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
                        h.n.x, h.n.y, h.n.z, r, h.u, h.v);
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
                         px, py, pz, 0.0, h.n.x, h.n.y, h.n.z, r, h.u, h.v);
    } else if (m.recRoughMode == 1) {                      // constant selStop (one stop, per-hit)
        v = dRecStopVal(sc, sc.recScalarStops[m.recRoughStopOff], h);
    } else {                                               // per-hit driven
        double px = h.p.x, py = h.p.y, pz = h.p.z, r = sqrt(px * px + py * py + pz * pz);
        double d = dPatternEval(sc.recDrivers + m.recRoughDrvOff, m.recRoughDrvN,
                                px, py, pz, 0.0, h.n.x, h.n.y, h.n.z, r, h.u, h.v);
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
                            px, py, pz, 0.0, h.n.x, h.n.y, h.n.z, r, h.u, h.v);
    out = dRecReflAt(sc.recCoeff + m.recReflOff, REC_LUT_N,
                     (double)m.recReflLo, (double)m.recReflHi, d, lambda);
    return true;
}

// Reflect-slot reflectance for the SPECULAR families (mirror / glossy / grating /
// halfmirror): a driven record if present, else the constant baked reflect spectrum
// (device twin of host reflectSlot; these types never bind a reflect texture).
__device__ static Real dReflectSlot(const DScene& sc, const DMaterial& m, const DHit& h, Real lambda) {
    Real v;
    if (dRecordReflect(sc, m, h, lambda, v)) return v;
    return specLookup(m.reflect, lambda);
}

// Diffuse reflectance at a hit: a driven parametric record (highest priority), else a
// bound texture, else the constant baked reflect spectrum (mirrors host diffuseReflectance).
__device__ static Real dDiffuseRho(const DScene& sc, const DMaterial& m, const DHit& h, Real lambda) {
    Real rv;
    if (dRecordReflect(sc, m, h, lambda, rv)) return clamp01(rv);
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
__device__ static bool genPhoton(const DScene& sc, const DCamSet& cs,
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
        // Emitter vertex: ns==ng==emitN, so the adjoint correction is identically 1
        // (wi is irrelevant here — pass emitN).
        splatSurfaceAll(sc, cs, camMode, origin, emitN, emitN, emitN, lambda, beta, (Real)1, rng);
        camSpecularSplatAll(sc, cs, camMode, origin, emitN, lambda, beta, (Real)1, rng);
    }

    ro = origin + dir * RAY_EPS; rd = dir;
    return true;
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
        Real t = clamp01(specLookup(m.transmit, lambda));
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
    }
    return WF_CONTINUE;   // unreachable: caller dispatches only the nine specular types
}

// Advance a photon by one bounce given its precomputed intersection `h`. Mutates
// ro/rd/beta and accumulates absorbed/sensor/escaped energy. Returns WF_TERMINATE
// when the path ends (absorbed / escaped / landed on the sensor), else WF_CONTINUE
// with ro/rd set for the next segment. `h` is the closestHit(sc, ro, rd) result.
__device__ static int shadeStep(const DScene& sc, const DCamSet& cs,
                                int camMode, int diffraction, const DHit& h,
                                DVec3& ro, DVec3& rd, Real& beta, Real& lambda, DRng& rng,
                                double& eAbsorbed, double& eSensor, double& eEscaped,
                                DMediumStack& stk) {
    Real dSurf = h.valid ? h.t : BIG;

    // fog free-flight; dEvent is the nearer of surface hit / volume collision.
    bool mediumEvent = false; int scatterMed = -1; DVec3 mp; Real dEvent = dSurf;
    if (sc.mediaN > 0) {
        // Superposition of all media: each does its own delta (Woodcock) tracking (or
        // exact analytic free-flight if homogeneous); the earliest collision wins and
        // its medium (scatterMed) drives the scatter. Device twin of sampleMediaCollision.
        Real tMed; int which;
        if (dMediaSampleCollision(sc.media, sc.mediaN, ro, rd, dSurf, lambda, rng, tMed, which)) {
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
    {
        int cm = stk.topMat();
        Real a = (cm >= 0) ? (Real)specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
        if (a > 0) beta *= exp(-a * dEvent);
    }

    if (mediumEvent) {
        const DMedium& sm = sc.media[scatterMed];
        splatVolumeAll(sc, sm, cs, camMode, mp, rd, lambda, beta, rng);
        camSpecularSplatVolumeAll(sc, sm, cs, camMode, mp, rd, lambda, beta, rng);
        if (rng.uniform() >= medAlbedo(sm, lambda)) { eAbsorbed += beta; return WF_TERMINATE; }
        DVec3 nd = sampleHG(rd, (Real)sm.g, rng);
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
        m.type == D_FILTER || m.type == D_GLOSSY || m.type == D_FLUORESCENT) {
        // The nine specular / wavelength-switching lobes — shared with the hero tracer.
        return interactSpecular(sc, cs, camMode, diffraction, m, matIndex, h,
                                ro, rd, beta, lambda, rng, eAbsorbed, stk);
    } else if (m.type == D_DIFFUSETRANSMIT) {
        // Two-lobe Lambertian (device twin of render.h DiffuseTransmit): `reflect` into
        // the front (+n) hemisphere, `transmit` into the back (-n) hemisphere. Splat BOTH
        // lobes — connect()/connectLens() self-reject the wrong-side lobe (cosSurf<=0), so
        // passing the flipped normal images whichever side the camera is on. Non-specular,
        // so a directly-viewed translucent solid is VISIBLE in model B (unlike dielectric).
        Real rhoR = dDiffuseRho(sc, m, h, lambda);
        Real rhoT = clamp01(specLookup(m.transmit, lambda));
        Real sum = rhoR + rhoT;
        if (sum > (Real)1) { rhoR /= sum; rhoT /= sum; sum = (Real)1; }   // energy guard
        DVec3 nb = h.n * (Real)(-1);
        DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);   // geo normal on shading side
        DVec3 wiPrev = -rd;                                             // toward previous (light-side)
        depositPhoton(cs, h.p, rd, h.n, beta, lambda);   // photon-map deposit (mode M)
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
        depositPhoton(cs, h.p, rd, h.n, beta, lambda);   // photon-map deposit (mode M)
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
        if (sc.mediaN > 0) contrib *= dMediaTransmittance(sc.media, sc.mediaN, p, wdir, dist, lam[i], rng);
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
        if (sc.mediaN > 0) contrib *= dMediaTransmittance(sc.media, sc.mediaN, p, wdir, dist, lam[i], rng);
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
// Refract each live wavelength's Lambertian reflection through every glass sphere. Unlike
// the achromatic camera connection this CANNOT share geometry: the sphere IOR is dispersive
// (per-λ), so each wavelength traces its own refracted image — exactly what makes a glass-
// sphere caustic chromatically dispersed. Draws no RNG (mode B only inside).
__device__ static void camSpecularSplatAllHero(const DScene& sc, const DCamSet& cs, int camMode,
        const DVec3& p, const DVec3& n, const Real* lam, const Real* beta, const Real* rho,
        int nUp, DRng& rng) {
    for (int i = 0; i < nUp; ++i)
        camSpecularSplatAll(sc, cs, camMode, p, n, lam[i], beta[i], rho[i], rng);
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
    } else {
        emitterSamplePoint(em, u1, u2, origin, emitN);
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
    secAlive = (C > 1);
    int nUp = secAlive ? C : 1;
    for (int i = 0; i < nUp; ++i) eEmitted += (double)beta[i];

    // Direct emitter->camera connection (area/quad emitters only; spot/env have no direct
    // term). No-op for mode C (splat helpers skip it) and the mode-M deposit pass (nCam==0).
    if (em.shape != 2 && em.shape != 3) {
        Real rhoOne[hero::kHeroMax]; for (int i = 0; i < nUp; ++i) rhoOne[i] = (Real)1;
        splatSurfaceAllHero(sc, cs, camMode, origin, emitN, emitN, emitN, lam, beta, rhoOne, nUp, rng);
        camSpecularSplatAllHero(sc, cs, camMode, origin, emitN, lam, beta, rhoOne, nUp, rng);
    }

    ro = origin + dir * RAY_EPS; rd = dir;
    return true;
}

// One hero bounce (called only while secAlive, so nUp == C). Handles the model-C catch,
// escape/sensor bookkeeping, and the diffuse / diffuse-transmit lobes with per-λ deposit +
// splat + Russian-roulette on the hero (secondaries reweighted by rho[i]/rho[0]). At any of
// the nine specular / wavelength-switching materials it DE-HEROS (beta[0] *= C, secAlive =
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
            Real rt = clamp01(specLookup(m.transmit, lam[i]));
            Real s = rr + rt; if (s > (Real)1) { rr /= s; rt /= s; }   // per-λ energy guard
            rhoR[i] = rr; rhoT[i] = rt;
        }
        DVec3 nb = h.n * (Real)(-1);
        DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
        DVec3 wiPrev = -rd;
        for (int i = 0; i < nUp; ++i) depositPhoton(cs, h.p, rd, h.n, beta[i], lam[i]);
        if (camMode == CAM_A || camMode == CAM_B) {
            splatSurfaceAllHero(sc, cs, camMode, h.p, h.n, ngo, wiPrev, lam, beta, rhoR, nUp, rng);
            splatSurfaceAllHero(sc, cs, camMode, h.p, nb, ngo * (Real)(-1), wiPrev, lam, beta, rhoT, nUp, rng);
            camSpecularSplatAllHero(sc, cs, camMode, h.p, h.n, lam, beta, rhoR, nUp, rng);
            camSpecularSplatAllHero(sc, cs, camMode, h.p, nb, lam, beta, rhoT, nUp, rng);
        }
        Real sumHero = rhoR[0] + rhoT[0];
        Real uu = rng.uniform();
        if (uu < rhoR[0]) {
            for (int i = 1; i < nUp; ++i) beta[i] *= rhoR[i] / rhoR[0];
            DVec3 wo = cosineHemisphere(h.n, rng);
            Real corr = dShadingAdjointCorr(wiPrev, wo, h.n, ngo);
            for (int i = 0; i < nUp; ++i) beta[i] *= corr;
            ro = h.p + h.n * RAY_EPS; rd = wo; return WF_CONTINUE;
        } else if (uu < sumHero) {
            for (int i = 1; i < nUp; ++i) beta[i] *= rhoT[i] / rhoT[0];
            DVec3 wo = cosineHemisphere(nb, rng);
            Real corr = dShadingAdjointCorr(wiPrev, wo, h.n, ngo);
            for (int i = 0; i < nUp; ++i) beta[i] *= corr;
            ro = h.p + nb * RAY_EPS; rd = wo; return WF_CONTINUE;
        }
        for (int i = 0; i < nUp; ++i) eAbsorbed += (double)beta[i];
        return WF_TERMINATE;
    }

    if (m.type == D_DIELECTRIC || m.type == D_THINFILM || m.type == D_MULTILAYER ||
        m.type == D_MIRROR || m.type == D_GRATING || m.type == D_HALFMIRROR ||
        m.type == D_FILTER || m.type == D_GLOSSY || m.type == D_FLUORESCENT) {
        // Dispersive / wavelength-switching: terminate secondaries, boost the hero ×C, then
        // run the shared scalar interaction on the (now single-λ) hero channel.
        beta[0] *= (Real)C; secAlive = false;
        return interactSpecular(sc, cs, camMode, diffraction, m, matIndex, h,
                                ro, rd, beta[0], lam[0], rng, eAbsorbed, stk);
    }

    // Diffuse (texture-sampled reflectance when the material binds a texture).
    Real rho[hero::kHeroMax];
    for (int i = 0; i < nUp; ++i) rho[i] = clamp01(dDiffuseRho(sc, m, h, lam[i]));
    DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
    DVec3 wiPrev = -rd;
    for (int i = 0; i < nUp; ++i) depositPhoton(cs, h.p, rd, h.n, beta[i], lam[i]);
    if (camMode == CAM_A || camMode == CAM_B) {
        splatSurfaceAllHero(sc, cs, camMode, h.p, h.n, ngo, wiPrev, lam, beta, rho, nUp, rng);
        camSpecularSplatAllHero(sc, cs, camMode, h.p, h.n, lam, beta, rho, nUp, rng);
    }
    Real rhoHero = rho[0];
    if (rng.uniform() >= rhoHero) { for (int i = 0; i < nUp; ++i) eAbsorbed += (double)beta[i]; return WF_TERMINATE; }
    for (int i = 1; i < nUp; ++i) beta[i] *= rho[i] / rhoHero;   // secondary reweight
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
        for (int bounce = 0; bounce < maxBounce && !done; ++bounce) {
            if (sc.hasGrin) dGrinMarch(sc, ro, rd);   // bend through any GRIN region first
            DHit h = closestHit(sc, ro, rd);
            if (shadeStep(sc, cs, camMode, diffraction, h, ro, rd, beta, lambda, rng,
                          eAbsorbed, eSensor, eEscaped, stk) == WF_TERMINATE) done = true;
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
// stable, matching the CPU reference to within Monte-Carlo noise. Emissive surfaces
// and area/sphere lights only (spot/env/collimated/fog scenes fall back to the CPU
// via cudaBdptSupported). See bdpt.h for the derivation of every quantity below.

#define BDPT_MAXDEPTH 8
#define BDPT_MAXV     (BDPT_MAXDEPTH + 3)   // path[0] endpoint + up to MAXDEPTH surfaces + slack
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
};

__device__ static inline double ddot(const DVec3& a, const DVec3& b) {
    return (double)a.x * b.x + (double)a.y * b.y + (double)a.z * b.z;
}
// Medium (volume) vertices carry no surface, so onSurface() is false — ConvertDensity
// then omits the cosine Jacobian, giving the correct cosine-free volume area density.
__device__ static inline bool dOnSurface(const DVertex& v) {
    return v.type == BV_SURFACE || v.type == BV_LIGHT;
}
__device__ static inline bool dConnectibleType(int tp) {
    return tp == D_DIFFUSE || tp == D_GLOSSY || tp == D_FLUORESCENT;
}
__device__ static bool dVertConnectible(const DScene& sc, const DVertex& v) {
    if (v.type == BV_CAMERA) return true;
    if (v.type == BV_MEDIUM) return true;   // volume in-scatter always connects
    if (v.type == BV_LIGHT)  return v.lightIdx >= 0 && sc.emitters[v.lightIdx].collimated == 0;
    if (v.delta) return false;
    return dConnectibleType(sc.mats[v.matId].type);
}
// HG phase as the medium "BSDF": propagation INTO the vertex is -wo, scattered dir is wi
// (both point away from v), so cosTheta = -dot(wo,wi). hgPhase is its own pdf, so
// dPhaseF == dPhasePdf. mediumScatterF = albedo * phase is the CONNECTION response
// (albedo corrects the sigma_t-rate collision to the sigma_s scatter rate; it enters the
// throughput once, never the MIS density). Mirrors bdpt.h phaseF / mediumScatterF.
__device__ static inline double dPhaseF(const DVertex& v, const DVec3& wo, const DVec3& wi) {
    return (double)hgPhase((Real)(-ddot(wo, wi)), (Real)v.mediumG);
}
__device__ static inline double dPhasePdf(const DVertex& v, const DVec3& wo, const DVec3& wi) {
    return (double)hgPhase((Real)(-ddot(wo, wi)), (Real)v.mediumG);
}
__device__ static inline double dMediumScatterF(const DScene& sc, const DVertex& v,
                                                const DVec3& wo, const DVec3& wi, Real lambda) {
    return (double)medAlbedo(sc.media[v.mediumId], lambda) * dPhaseF(v, wo, wi);
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
// Emission directional density at a light vertex toward `next` (area measure).
__device__ static float dVertexPdfLightF(const DVertex& cur, const DVertex& next) {
    DVec3 w = next.p - cur.p;
    float d2 = dot(w, w);
    if (d2 == 0.f) return 0.f;
    float invD2 = 1.0f / d2;
    DVec3 wn = w * (Real)sqrtf(invD2);
    float cosLight = dot(cur.ng, wn);
    if (cosLight <= 0.f) return 0.f;
    float pdf = (cosLight * (float)(1.0 / DPI)) * invD2;
    if (dOnSurface(next)) pdf *= fabsf(dot(next.ns, wn));
    return pdf;
}
__device__ static float dVertexPdfF(const DScene& sc, const DCamera& cam,
                                    const DVertex* prev, const DVertex& cur, const DVertex& next) {
    if (cur.type == BV_LIGHT) return dVertexPdfLightF(cur, next);
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
        pdfW = (float)dPhasePdf(cur, wp, wn);
    } else {
        if (!prev) return 0.f;
        DVec3 wp = prev->p - cur.p;
        if (dot(wp, wp) == 0.f) return 0.f;
        wp = normalize(wp);
        pdfW = (float)dBsdfPdf(sc, cur.matId, cur.ns, wp, wn);
    }
    // dConvertDensity in FP32: solid angle -> area density at `next`.
    DVec3 wv = next.p - cur.p;
    float d2 = dot(wv, wv);
    if (d2 == 0.f) return 0.f;
    float invD2 = 1.0f / d2;
    if (dOnSurface(next)) pdfW *= fabsf(dot(next.ns, wv) * sqrtf(invD2));
    return pdfW * invD2;
}
__device__ static float dVertexPdfLightOriginF(const DScene& sc, const DVertex& cur) {
    if (cur.lightIdx < 0) return 0.f;
    const DEmitter& em = sc.emitters[cur.lightIdx];
    if (sc.totalPower <= 0.0 || em.area <= 0.0) return 0.f;
    return (float)((em.power / sc.totalPower) / em.area);
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
        // Geometric-hemisphere softening (matches CPU backward.h neeLight): the light must lie
        // on the geometric front side too, ramped smoothly instead of a hard cutoff (Chiang
        // 2019). No-op when h.n==h.ng (flat tris / analytic spheres, stG==1); shadow ray offset
        // along the geometric normal so it clears the true surface.
        DVec3 ngo = (dot(h.ng, h.n) >= 0) ? h.ng : h.ng * (Real)(-1);
        Real stG = dShadowTerminatorG(wi, h.n, ngo);
        if (stG <= (Real)0) continue;
        Real cosLight = dot(nL, -wi);                 // light is one-sided
        if (cosLight <= 0) continue;
        if (occluded(sc, h.p + ngo * RAY_EPS, wi, dist - (Real)2 * RAY_EPS)) continue;
        Real G = cosSurf * cosLight / dist2;
        double emitW = (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
        double contrib = (double)(f * G) * emitW * (double)em.area * (double)stG;
        // Backward (mode R) treats media as a single global HOMOGENEOUS haze (first
        // medium); GPU backward is only reached when no medium is present (cudaBackwardSupported
        // rejects any), so this is defensive parity with the host backwardMedium().
        if (sc.mediaN > 0) contrib *= exp(-(double)medSigmaT(sc.media[0], lambda) * (double)dist);
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
    DMediumStack stk; stk.clear();                     // nested-dielectric medium stack (empty = vacuum)
    const int maxBounce = 32;
    for (int b = 0; b < maxBounce; ++b) {
        DHit h = closestHit(sc, ro, rd);
        if (!h.valid) return L;                        // escaped (no env in v1)
        // Beer-Lambert attenuation over the in-glass segment up to this surface
        // (current medium = highest-priority stack entry).
        {
            int cm = stk.topMat();
            Real a = (cm >= 0) ? (Real)specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
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
                DVec3 nro, nrd; dDielectricStep(sc, *mp, h, rd, lambda, rng, matId, stk, nro, nrd);
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
                Real r = clamp01(dReflectSlot(sc, *mp, h, lambda));
                if (rng.uniform() >= r) return L;       // RR absorb
                ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); specularArrival = true; break;
            }
            case D_GRATING: {
                Real r = clamp01(dReflectSlot(sc, *mp, h, lambda));
                if (rng.uniform() >= r) return L;
                DVec3 nro, nrd;
                if (!gratingDiffract(*mp, h, rd, lambda, diffraction, rng, nro, nrd)) return L;
                ro = nro; rd = nrd; specularArrival = true; break;
            }
            case D_HALFMIRROR: {
                Real r = clamp01(dReflectSlot(sc, *mp, h, lambda));
                if (rng.uniform() < r) { ro = h.p + h.n * RAY_EPS; rd = reflectv(rd, h.n); }
                else                   { ro = h.p + rd * RAY_EPS; }
                specularArrival = true; break;
            }
            case D_FILTER: {
                // Colored gel filter: pass straight through, survive with prob T(lambda).
                Real t = clamp01(specLookup(mp->transmit, lambda));
                if (rng.uniform() >= t) return L;   // absorbed
                ro = h.p + rd * RAY_EPS;            // direction unchanged
                specularArrival = true; break;
            }
            case D_GLOSSY: {
                Real r = clamp01(dReflectSlot(sc, *mp, h, lambda));
                if (rng.uniform() >= r) return L;
                DVec3 o = sampleGlossy(reflectv(rd, h.n), dMatRoughness(sc, *mp, h), rng);
                if (dot(o, h.n) <= 0) return L;
                ro = h.p + h.n * RAY_EPS; rd = o; specularArrival = true; break;
            }
            case D_DIFFUSETRANSMIT: {
                // Two-lobe Lambertian (device twin of backward.h DiffuseTransmit): NEE the
                // reflect lobe against lights in the front (+n) hemisphere and the transmit
                // lobe in the back (-n) hemisphere (a normal-flipped Hit reuses bkNeeLight),
                // then continue reflect / transmit / absorb (throughput unchanged on survival).
                Real rhoR = clamp01(dDiffuseRho(sc, *mp, h, lambda));
                Real rhoT = clamp01(specLookup(mp->transmit, lambda));
                Real sum = rhoR + rhoT;
                if (sum > (Real)1) { rhoR /= sum; rhoT /= sum; sum = (Real)1; }   // energy guard
                DVec3 nb = h.n * (Real)(-1);
                L += thr * bkNeeLight(sc, h, rhoR, invPdfLambda, lambda, rng);   // front lobe
                DHit hb = h; hb.n = nb;
                L += thr * bkNeeLight(sc, hb, rhoT, invPdfLambda, lambda, rng);  // back lobe
                Real u = rng.uniform();
                if (u < rhoR)     { ro = h.p + h.n * RAY_EPS; rd = cosineHemisphere(h.n, rng); specularArrival = false; break; }
                else if (u < sum) { ro = h.p + nb  * RAY_EPS; rd = cosineHemisphere(nb,  rng); specularArrival = false; break; }
                return L;                                // absorbed
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
// Renders `chunkSpp` samples-per-pixel for the chunk starting at sample `sampleBase`,
// accumulating (atomicAdd) into `film`/`hits`. The RNG is seeded on the GLOBAL sample
// index (pixel * sppTotal + sampleBase + localSample) so a render split into any number
// of chunks draws exactly the same union of streams as one single-shot pass of sppTotal
// samples — chunked progress is therefore bit-identical to the monolithic render.
__global__ void kBackward(DScene sc, DCamera cam, double* film, double* hits,
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
        DVec3 N = normalize(h.n);
        DVec3 V = normalize(cam.eye - h.p);
        if (dot(N, V) < 0) N = -N;                            // two-sided preview
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
__device__ static void dRandomWalk(const DScene& sc, const DCamera& cam, int diffraction,
                                   DVec3 ro, DVec3 rd, double beta, double pdfDir, Real lambda,
                                   int maxDepth, DRng& rng, DVertex* path, int& n,
                                   bool importance) {
    if (maxDepth == 0) return;
    double pdfFwd = pdfDir;
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
            if (dMediaSampleCollision(sc.media, sc.mediaN, ro, rd, (Real)dSurf, lambda, rng, tm, which)) {
                tMed = (double)tm; mediumEvent = true; scatterMed = which;
            }
        }

        // Medium collision precedes the surface: append a volume in-scatter vertex, then
        // scatter (prob = albedo) or absorb. Throughput unchanged on scatter (HG sampling
        // pdf == phase value, analog MC). Stored area densities are cosine-free and carry
        // only the phase direction density; the free-flight distance pdf and transmittance
        // are omitted here AND in dVertexPdf, so they cancel pairwise in every MIS ratio.
        if (mediumEvent) {
            if (n >= BDPT_MAXV) return;
            const DMedium& sm = sc.media[scatterMed];
            DVec3 mpos = ro + rd * (Real)tMed;
            int prevIdx = n - 1;
            DVertex v;
            v.type = BV_MEDIUM; v.p = mpos; v.ns = rd; v.ng = rd;
            v.beta = beta; v.pdfFwd = 0; v.pdfRev = 0; v.delta = 0;
            v.matId = -1; v.lightIdx = -1;
            v.mediumG = sm.g; v.mediumId = scatterMed;
            v.pdfFwd = dConvertDensity(pdfFwd, path[prevIdx], v);
            path[n] = v; int cur = n; n++;
            if (++bounces >= maxDepth) return;
            if (rng.uniform() >= (double)medAlbedo(sm, lambda)) return;   // absorbed (vertex retained)
            DVec3 wo = normalize(path[prevIdx].p - path[cur].p);          // toward previous vertex
            DVec3 wi = sampleHG(rd, (Real)sm.g, rng);                     // scattered propagation dir
            double pdfW    = dPhasePdf(path[cur], wo, wi);
            double pdfRevW = dPhasePdf(path[cur], wi, wo);
            path[prevIdx].pdfRev = dConvertDensity(pdfRevW, path[cur], path[prevIdx]);
            ro = mpos; rd = normalize(wi);
            pdfFwd = pdfW;
            continue;
        }

        if (!h.valid) return;

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
        v.mediumG = 0.0; v.mediumId = -1;
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
                double t = clamp01(specLookup(mp->transmit, lambda));
                wi = rd; betaFactor = t; delta = 1;
                if (t <= 0) terminate = true;
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
        // Veach adjoint shading-normal correction on the LIGHT subpath only (1 when
        // ns==ng). wo = toward previous (light-side) vertex; wi = sampled continuation.
        if (importance && !delta) {
            DVec3 ngo = (dot(path[cur].ng, path[cur].ns) >= 0.0) ? path[cur].ng : path[cur].ng * (Real)(-1);
            beta *= (double)dShadingAdjointCorr(wo, normalize(wi), path[cur].ns, ngo);
        }
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
    c.mediumG = 0.0; c.mediumId = -1;
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
        dRandomWalk(sc, cam, diffraction, ro, rd, (double)wl, pdfDir, lambda, maxDepth - 1, rng, path, n, false);
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
    dRandomWalk(sc, cam, diffraction, cam.eye, rd, 1.0, pdfDir, lambda, maxDepth - 1, rng, path, n, false);
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
    L0.mediumG = 0.0; L0.mediumId = -1;
    path[0] = L0; int n = 1;
    DVec3 dir = cosineHemisphere(nOut, rng);
    double cosLight = ddot(nOut, dir);
    if (cosLight <= 0.0) return 1;
    double pdfDir = cosLight / DPI;
    double betaWalk = Le * cosLight / (pdfChoice * pdfPos * pdfDir);
    DVec3 ro = y + nOut * (Real)1e-6;
    dRandomWalk(sc, cam, diffraction, ro, dir, betaWalk, pdfDir, lambda, maxDepth - 1, rng, path, n, true);
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
                                    const DVertex& sampled, int s, int t) {
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
        ptPdfRev = (s > 0) ? dVertexPdfF(sc, cam, (s > 1) ? &light[sMi] : nullptr, *QsP, *PtP)
                           : dVertexPdfLightOriginF(sc, *PtP);
    if (t >= 3)
        ptMPdfRev = (s > 0) ? dVertexPdfF(sc, cam, QsP, *PtP, eye[tMi])
                            : dVertexPdfLightF(*PtP, eye[tMi]);
    if (s >= 1) qsPdfRev  = dVertexPdfF(sc, cam, (t > 1) ? &eye[tMi] : nullptr, *PtP, *QsP);
    if (s >= 2) qsMPdfRev = dVertexPdfF(sc, cam, PtP, *QsP, light[sMi]);

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
        bool deltaPrev = (i > 0) ? (light[i - 1].delta != 0) : false;
        if (!dl && !deltaPrev) sumRi += ri;
    }
    return 1.0 / (1.0 + (double)sumRi);
}

// Connect strategy (s,t); returns the MIS-weighted radiance. For t==1 the result is a
// light-image splat to (outPx,outPy) with isSplat=1. Direct port of bdpt.h connectBDPT.
__device__ static double dConnectBDPT(const DScene& sc, const DCamera& cam,
                                      const DVertex* light, const DVertex* eye, int s, int t,
                                      Real lambda, double invPdfLambda, DRng& rng,
                                      int& outPx, int& outPy, int& isSplat) {
    isSplat = 0;
    if (t > 1 && s != 0 && dIsLightVertex(eye[t - 1])) return 0.0;

    double L = 0.0;
    DVertex sampled;
    sampled.type = BV_SURFACE; sampled.beta = 0; sampled.pdfFwd = 0; sampled.pdfRev = 0;
    sampled.delta = 0; sampled.matId = -1; sampled.lightIdx = -1;
    sampled.mediumG = 0.0; sampled.mediumId = -1;

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
        DVec3 wo = normalize(light[s - 2].p - qs.p);
        // Medium endpoint: phase*albedo, cosine 1, occlusion from the exact point.
        double cosSurf, f; DVec3 o;
        if (qs.type == BV_MEDIUM) {
            cosSurf = 1.0; f = dMediumScatterF(sc, qs, wo, wcam, lambda); o = qs.p;
        } else {
            cosSurf = ddot(qs.ns, wcam);
            if (cosSurf <= 0.0) return 0.0;
            // Geometric-hemisphere softening (matches CPU bdpt.h): the camera must lie on the
            // geometric front side too, else a smoothed shading normal leaks light through the
            // back face; ramp smoothly instead of a hard cutoff (Chiang 2019). No-op when ns==ng
            // (stG==1). GPU BDPT is reflect-only (v1), so unconditional.
            DVec3 ngoQ = (ddot(qs.ng, qs.ns) >= 0.0) ? qs.ng : qs.ng * (Real)(-1);
            double stG = (double)dShadowTerminatorG(wcam, qs.ns, ngoQ);
            if (stG <= 0.0) return 0.0;
            f = dBsdfF(sc, qs.matId, qs.ns, wo, wcam, lambda);
            // Adjoint correction on the LIGHT-subpath vertex qs (particle side, outgoing
            // toward camera). 1 when ns==ng. wo = toward previous (light-side) vertex.
            f *= (double)dShadingAdjointCorr(wo, wcam, qs.ns, ngoQ) * stG;
            double sgn = ddot(qs.ng, wcam) >= 0.0 ? 1.0 : -1.0;
            o = qs.p + qs.ng * (Real)(sgn * 1e-6);
        }
        if (f <= 0.0) return 0.0;
        if (occluded(sc, o, wcam, (Real)(dist - 2e-6))) return 0.0;
        double cosCam = ddot(qs.p - cam.eye, cam.w) / dist;   // positive (point in front)
        double Tr = (sc.mediaN > 0) ? (double)dMediaTransmittance(sc.media, sc.mediaN, qs.p, wcam, (Real)dist, lambda, rng) : 1.0;
        double G = cosSurf * cosCam / dist2;
        L = qs.beta * f * G * dCameraWe(cam, cosCam) * Tr;
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
        if (cosLight <= 0.0) return 0.0;               // emitter stays one-sided
        double Le = (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
        if (Le <= 0.0) return 0.0;
        DVec3 wo = normalize(eye[t - 2].p - pt.p);
        double cosSurf, f; DVec3 o;
        if (pt.type == BV_MEDIUM) {
            cosSurf = 1.0; f = dMediumScatterF(sc, pt, wo, wi, lambda); o = pt.p;
        } else {
            cosSurf = ddot(pt.ns, wi);
            if (cosSurf <= 0.0) return 0.0;
            // Geometric-hemisphere softening on the eye/radiance vertex (matches CPU bdpt.h):
            // ramp smoothly instead of a hard cutoff (Chiang 2019). No-op when ns==ng (stG==1).
            DVec3 ngoP = (ddot(pt.ng, pt.ns) >= 0.0) ? pt.ng : pt.ng * (Real)(-1);
            double stG = (double)dShadowTerminatorG(wi, pt.ns, ngoP);
            if (stG <= 0.0) return 0.0;
            f = dBsdfF(sc, pt.matId, pt.ns, wo, wi, lambda) * stG;
            double sgn = ddot(pt.ng, wi) >= 0.0 ? 1.0 : -1.0;
            o = pt.p + pt.ng * (Real)(sgn * 1e-6);
        }
        if (f <= 0.0) return 0.0;
        if (occluded(sc, o, wi, (Real)(dist - 2e-6))) return 0.0;
        double pdfChoice = em.power / sc.totalPower;
        double pdfA = pdfChoice / em.area;
        if (pdfA <= 0.0) return 0.0;
        double Tr = (sc.mediaN > 0) ? (double)dMediaTransmittance(sc.media, sc.mediaN, pt.p, wi, (Real)dist, lambda, rng) : 1.0;
        double G = cosSurf * cosLight / dist2;
        L = pt.beta * f * Le * G / pdfA * Tr;
        if (L <= 0.0) return 0.0;
        sampled.type = BV_LIGHT; sampled.p = y; sampled.ns = nOut; sampled.ng = nOut;
        sampled.lightIdx = ei; sampled.matId = em.matId; sampled.beta = Le / pdfA; sampled.pdfFwd = pdfA;
    } else {
        const DVertex& qs = light[s - 1];
        const DVertex& pt = eye[t - 1];
        if (!dVertConnectible(sc, qs) || !dVertConnectible(sc, pt)) return 0.0;
        DVec3 d = qs.p - pt.p; double dist2 = ddot(d, d);
        if (dist2 <= 0.0) return 0.0;
        double dist = sqrt(dist2); DVec3 w = d * (Real)(1.0 / dist);   // pt -> qs
        DVec3 woE = normalize(eye[t - 2].p - pt.p);
        DVec3 woL = normalize(light[s - 2].p - qs.p);
        // Each endpoint is a surface (BSDF, cosine) or a medium (phase*albedo, cos=1).
        double cosE, cosL, fE, fL; DVec3 o;
        if (pt.type == BV_MEDIUM) {
            cosE = 1.0; fE = dMediumScatterF(sc, pt, woE, w, lambda); o = pt.p;
        } else {
            cosE = ddot(pt.ns, w);
            if (cosE <= 0.0) return 0.0;
            // Geometric-hemisphere softening on the eye endpoint (connection dir w): ramp
            // smoothly instead of a hard cutoff (Chiang 2019). No-op ns==ng (stGE==1).
            DVec3 ngoE = (ddot(pt.ng, pt.ns) >= 0.0) ? pt.ng : pt.ng * (Real)(-1);
            double stGE = (double)dShadowTerminatorG(w, pt.ns, ngoE);
            if (stGE <= 0.0) return 0.0;
            fE = dBsdfF(sc, pt.matId, pt.ns, woE, w, lambda) * stGE;
            double sgn = ddot(pt.ng, w) >= 0.0 ? 1.0 : -1.0;
            o = pt.p + pt.ng * (Real)(sgn * 1e-6);
        }
        if (qs.type == BV_MEDIUM) {
            cosL = 1.0; fL = dMediumScatterF(sc, qs, woL, w * (Real)-1, lambda);
        } else {
            cosL = ddot(qs.ns, w * (Real)-1);
            if (cosL <= 0.0) return 0.0;
            // Geometric-hemisphere softening on the light endpoint (connection dir -w): ramp
            // smoothly instead of a hard cutoff (Chiang 2019). No-op ns==ng (stGL==1).
            DVec3 ngoQ = (ddot(qs.ng, qs.ns) >= 0.0) ? qs.ng : qs.ng * (Real)(-1);
            double stGL = (double)dShadowTerminatorG(w * (Real)-1, qs.ns, ngoQ);
            if (stGL <= 0.0) return 0.0;
            fL = dBsdfF(sc, qs.matId, qs.ns, woL, w * (Real)-1, lambda) * stGL;
            // Adjoint correction on the LIGHT-subpath endpoint qs only (particle side,
            // outgoing = -w toward the eye vertex). fE is the Radiance side — no correction.
            fL *= (double)dShadingAdjointCorr(woL, w * (Real)-1, qs.ns, ngoQ);
        }
        if (fE <= 0.0 || fL <= 0.0) return 0.0;
        if (occluded(sc, o, w, (Real)(dist - 2e-6))) return 0.0;
        double Tr = (sc.mediaN > 0) ? (double)dMediaTransmittance(sc.media, sc.mediaN, pt.p, w, (Real)dist, lambda, rng) : 1.0;
        double G = cosE * cosL / dist2;
        L = pt.beta * fE * fL * qs.beta * G * Tr;
    }
    if (L <= 0.0) return 0.0;
    return L * dMisWeight(sc, cam, light, eye, sampled, s, t);
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
__global__ void __launch_bounds__(128, 3)
kBdpt(DScene sc, DCamera cam, double* camFilm, double* splatFilm,
                      long long totalSamples, long long chunkSpp, long long sppTotal,
                      long long sampleBase, int resX, int maxDepth,
                      int diffraction, unsigned long long seedBase) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    for (long long idx = g; idx < totalSamples; idx += G) {
        long long pix = idx / chunkSpp;
        long long gidx = pix * sppTotal + sampleBase + (idx - pix * chunkSpp);
        DRng rng; rng.seed((unsigned long long)(gidx * 2 + 1), seedBase ^ (unsigned long long)gidx);
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

// ----------------------- photon-map camera gather (mode M) -------------------
// Device twin of photonGather (photonmap_render.h, fgRays == 0 path). Follows a camera
// ray through specular surfaces (monochromatic at the sampled `lambda`); at the first
// diffuse / translucent hit it estimates reflected radiance by a radius density query
// into the uploaded photon map, each photon reflected at ITS OWN wavelength so the
// estimate is built directly in XYZ (returned via oX/oY/oZ). Directly-viewed emitters
// are added as a monochromatic estimate at the sampled lambda. No final gather, no env
// (both gated out by cudaPhotonMapSupported); Beer-Lambert interior absorption is tracked
// exactly like bkRadiance. Keep in sync with photonGather.
__device__ static void dPhotonGather(const DScene& sc, const DPhotonMap& pm, int diffraction,
                                     DVec3 ro, DVec3 rd, Real lambda, double invPdfL, DRng& rng,
                                     double& oX, double& oY, double& oZ) {
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
        if (!h.valid) return;                            // escaped (env gated out of mode-M GPU)

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
            double e = (double)specLookup(sc.emitters[li].emitSpd, lambda) * thr * invPdfL;
            oX += (double)cieX(lambda) * e;
            oY += (double)cieY(lambda) * e;
            oZ += (double)cieZ(lambda) * e;
            return;
        }

        if (m.type == D_DIFFUSE || m.type == D_DIFFUSETRANSMIT || m.type == D_FLUORESCENT) {
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
                thr *= (double)clamp01(specLookup(m.transmit, lambda));
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
                        long long sampleBase, int resX, int diffraction,
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
        dPhotonGather(sc, pm, diffraction, ro, rd, lambda, invPdfL, rng, oX, oY, oZ);
        size_t o = ((size_t)py * resX + px) * 3;
        atomicAdd(&film[o + 0], oX);
        atomicAdd(&film[o + 1], oY);
        atomicAdd(&film[o + 2], oZ);
        if (hits) atomicAdd(&hits[(size_t)py * resX + px], 1.0);
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
        if (matId >= 0 && matId < (int)scene.mats.size() &&
            scene.mats[matId].type == MatType::Mix) {
            const Material& mx = scene.mats[matId];
            if ((int)mx.mixChildren.size() > D_MIXMAX) return true;
            for (int c : mx.mixChildren) if (oversizedMultilayer(c) || usesRecord(c)) return true;
        }
        return false;
    };
    for (const auto& t : scene.tris)      if (unsupported(t.matId)) return false;
    for (const auto& s : scene.spheres)   if (unsupported(s.matId)) return false;
    for (const auto& im : scene.implicits) if (unsupported(im.matId)) return false;
    // Spectral water-droplet (rainbow) phase is a CPU-tabulated (lambda x mu) table with
    // per-lambda CDF importance sampling (rainbow.h); the device volume path only knows
    // the analytic HG lobe (hgPhase). Rather than silently drop the bow to a smooth HG
    // haze, fall back to the CPU tracer for any scene with a rainbow-phase medium.
    for (const auto& m : scene.media) if (m.enabled && m.rainbow()) return false;
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
    // Top-level BVH: upload Scene::bvh VERBATIM in every case. Its prim-index layout is
    // [tris | spheres | implicits | instances] — the device leaf dispatch in
    // closestHit/occluded now understands all four ranges (an instance leaf transforms
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
    }

    // --- upload geometry/materials ---
    DTri*      d_tris  = tris.empty()    ? nullptr : (DTri*)keep(uploadVec(tris));
    DSphere*   d_sph   = sph.empty()     ? nullptr : (DSphere*)keep(uploadVec(sph));
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
    sc.fieldNodesF = d_fnodesF; sc.fieldExprNodesF = d_fexprF;
    sc.implicits = d_impl; sc.nImplicits = (int)dimpl.size();
    sc.instances = d_inst; sc.nInstances = (int)dinst.size();
    sc.blas = d_blas; sc.blasNodes = d_blasN; sc.blasPrim = d_blasP; sc.blasTris = d_blasT;
    sc.patNodes = d_pnodes; sc.patterns = d_pat; sc.nPatterns = (int)dpat.size();
    sc.recCoeff = d_recCoeff; sc.recDrivers = d_recDrv; sc.recScalarStops = d_recStops;
    sc.emitters = d_ems; sc.nEmitters = (int)dems.size();
    sc.emitCdf = d_emitCdf; sc.totalPower = scene.totalPower;
    sc.lightCdfAll = d_cdfAll;
    sc.textures = d_tex; sc.nTex = (int)dtex.size();
    sc.fluoCdfAll = d_fluoCdf;
    sc.emitSamplerCdf = d_emitSamp;
    sc.emitSamplerN = (int)(emitSampCdf.empty() ? 0 : emitSampCdf.size() - 1);
    sc.emitSamplerStep = scene.emitSampler.step;
    sc.emitG = scene.emitG;
    // Participating media array (superposed). Each medium's density program is uploaded
    // separately; then the flat DMedium array is uploaded once. Empty => media=null, mediaN=0.
    {
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
            // Imported .nvdb volume: upload the baked dense grid + world->index affine.
            if (m.vdb && !m.vdb->empty()) {
                const VdbGrid& g = *m.vdb;
                dm.vdbData = (const float*)keep(uploadVec(g.data));
                dm.vdbNx = g.nx; dm.vdbNy = g.ny; dm.vdbNz = g.nz;
                for (int k = 0; k < 9; ++k) dm.vdbAinv[k] = g.ainv[k];
                dm.vdbW0   = {g.w0.x, g.w0.y, g.w0.z};
                dm.vdbImin = {g.imin.x, g.imin.y, g.imin.z};
            } else {
                dm.vdbData = nullptr;
                dm.vdbNx = dm.vdbNy = dm.vdbNz = 0;
                for (int k = 0; k < 9; ++k) dm.vdbAinv[k] = (k % 4 == 0) ? 1.0 : 0.0;
                dm.vdbW0 = {0,0,0}; dm.vdbImin = {0,0,0};
            }
            dm.bounded  = m.bounded ? 1 : 0;
            dm.boundShape = (m.boundShape == MediumBound::Sphere)   ? 1
                          : (m.boundShape == MediumBound::Implicit) ? 2 : 0;
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
        }
        sc.media  = dmeds.empty() ? nullptr : (const DMedium*)keep(uploadVec(dmeds));
        sc.mediaN = (int)dmeds.size();
        sc.hasGrin = grin::sceneHasGrin(scene) ? 1 : 0;   // gate for dGrinMarch (host twin)
    }
    sc.sensorOrigin = {scene.sensor.origin.x, scene.sensor.origin.y, scene.sensor.origin.z};
    sc.sensorUAxis  = {scene.sensor.uAxis.x,  scene.sensor.uAxis.y,  scene.sensor.uAxis.z};
    sc.sensorVAxis  = {scene.sensor.vAxis.x,  scene.sensor.vAxis.y,  scene.sensor.vAxis.z};
    sc.sceneCenter = {scene.sceneCenter.x, scene.sceneCenter.y, scene.sceneCenter.z};
    sc.sceneRadius = scene.sceneRadius;
    sc.env = denv;

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
    if (wavefront && effHeroC == 1) {
        // Streaming backend: identical physics, path-regeneration scheduling. Same
        // maxBounce (32) and camera mode/set as the megakernel. (Hero forces megakernel.)
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

std::vector<Film> renderForwardSharedCuda(const Scene& scene,
                                          const std::vector<Camera>& cams,
                                          const std::vector<int>& resX,
                                          const std::vector<int>& resY,
                                          long long N, EnergyReport& eOut, bool diffraction,
                                          char camMode, unsigned long long seedBase,
                                          bool wavefront, int heroC) {
    using namespace gpu;
    int nc = (int)cams.size();
    std::vector<Film> out(nc);
    for (int c = 0; c < nc; ++c) { out[c].resX = resX[c]; out[c].resY = resY[c]; out[c].alloc(); }
    if (nc == 0 || !cudaAvailable() || !cudaForwardSupported(scene)) return out;

    // Bake the scene ONCE, then bake every camera against it (the win: one photon set,
    // splat to all cameras). Shared pass is model A or B only (C consumes the photon).
    DUpload up;
    buildUploadScene(scene, up);
    std::vector<DCamera> hcams(nc);
    for (int c = 0; c < nc; ++c) hcams[c] = bakeCamera(scene, cams[c], resX[c], resY[c], up);

    // Per-camera film / hits device buffers (each camera keeps its own resolution).
    std::vector<double*> d_films(nc, nullptr), d_hits(nc, nullptr);
    std::vector<size_t>  npix(nc);
    for (int c = 0; c < nc; ++c) {
        npix[c] = (size_t)resX[c] * resY[c];
        CUDA_CHECK(cudaMalloc(&d_films[c], npix[c] * 3 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_films[c], 0, npix[c] * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_hits[c], npix[c] * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_hits[c], 0, npix[c] * sizeof(double)));
    }
    double* d_energy = nullptr; CUDA_CHECK(cudaMalloc(&d_energy, 5 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_energy, 0, 5 * sizeof(double)));

    int camModeInt = (camMode == 'A') ? CAM_A : CAM_B;   // shared pass never runs mode C
    DCamSet cs = makeCamSet(up, hcams, d_films, d_hits);
    launchForward(up, cs, d_energy, N, diffraction, seedBase, wavefront, camModeInt, heroC);

    // --- download each camera's film ---
    for (int c = 0; c < nc; ++c) {
        std::vector<double> film(npix[c] * 3);
        CUDA_CHECK(cudaMemcpy(film.data(), d_films[c], film.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out[c].hits.data(), d_hits[c], npix[c] * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < npix[c]; ++i)
            out[c].xyz[i] = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);
    }
    // The photon trace is shared, so energy is counted once for the whole pass.
    double energy[5] = {0,0,0,0,0};
    CUDA_CHECK(cudaMemcpy(energy, d_energy, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    eOut.emitted  += energy[0];
    eOut.absorbed += energy[1];
    eOut.sensor   += energy[2];
    eOut.escaped  += energy[3];
    eOut.residual += energy[4];

    freeUpload(up);
    for (int c = 0; c < nc; ++c) { cudaFree(d_films[c]); cudaFree(d_hits[c]); }
    cudaFree(d_energy);
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
        // Diffuse-transmission (two-sided Lambertian): the GPU BDPT kernel (dBsdfF /
        // dBsdfPdf / dRandomWalk / dConnect) has no two-lobe / back-hemisphere strategy,
        // so a translucent vertex would render black or bias MIS. The CPU BDPT (bdpt.h)
        // handles it fully (isConnectibleMat + isTwoSidedMat), so fall back to it.
        if (m.type == MatType::DiffuseTransmit) return true;
        // Non-albedo texture maps (roughness / film-thickness) drive the glossy/thin-film
        // BSDF sampling per-hit; the GPU BDPT kernel's pdf/eval use the constant params,
        // which would bias MIS. Fall back to CPU BDPT (which threads the Hit through).
        if (m.roughnessTex >= 0 || m.filmThicknessTex >= 0) return true;
        // Procedural patterns drive the same per-hit params; the GPU BDPT kernel's
        // pdf/eval use the constants, so a pattern would bias MIS — use CPU BDPT.
        if (m.roughnessPat >= 0 || m.filmThicknessPat >= 0 || m.mixWeightPat >= 0) return true;
        // Parametric records drive slots from a per-hit driver, but the BDPT connection
        // BSDF (dBsdfF / dBsdfPdf) has no Hit in scope — it can't sample the driver, so a
        // record-bound vertex would bias MIS. Forward records run on-device (stage 6a),
        // but any record binding forces the CPU BDPT here (both driven reflect AND the
        // constant selStop case, since even the baked reflect would MIS against a driver-
        // less pdf inconsistently for driven neighbours). Keep BDPT record scenes on CPU.
        if (m.hasRecordBinding()) return true;
        // Mix blend mask: the GPU BDPT mix-pick uses constant weights; use CPU BDPT.
        if (m.mixWeightTex >= 0) return true;
        if (m.type == MatType::Mix)
            for (int c : m.mixChildren)
                if (c >= 0 && c < (int)scene.mats.size() &&
                    (frostedOrColoredGlass(scene.mats[c]) ||
                     scene.mats[c].reflectTex >= 0 || scene.mats[c].type == MatType::Fluorescent ||
                     scene.mats[c].type == MatType::DiffuseTransmit ||
                     scene.mats[c].roughnessTex >= 0 || scene.mats[c].filmThicknessTex >= 0 ||
                     scene.mats[c].roughnessPat >= 0 || scene.mats[c].filmThicknessPat >= 0 ||
                     scene.mats[c].mixWeightPat >= 0 || scene.mats[c].hasRecordBinding()))
                    return true;
        return false;
    };
    for (const auto& t : scene.tris)      if (usesTexOrFluoro(t.matId)) return false;
    for (const auto& s : scene.spheres)   if (usesTexOrFluoro(s.matId)) return false;
    for (const auto& im : scene.implicits) if (usesTexOrFluoro(im.matId)) return false;
    for (const auto& em : scene.emitters)
        if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env || em.collimated)
            return false;
    // Gradient-index (GRIN) media bend rays along curved paths; BDPT's connection geometry,
    // area-measure pdf conversion and MIS weights all assume STRAIGHT connecting segments, so
    // a GRIN region would bias the estimator. The mode-D guard (bdptUnsupportedFeature) already
    // refuses GRIN before dispatch; reject here too so GPU BDPT can never render it straight.
    if (grin::sceneHasGrin(scene)) return false;
    // Spectral rainbow phase is CPU-tabulated (see cudaForwardSupported); the device
    // volume connect/sample only knows the analytic HG lobe, so refuse rainbow media
    // here too and let mode D run on the CPU BDPT (which evaluates the bow exactly).
    for (const auto& m : scene.media) if (m.enabled && m.rainbow()) return false;
    return true;
}

// Drives a chunked samples-per-pixel render for the GPU reference/BDPT paths. Repeatedly
// renders `chunkSpp` more samples (via `launch(chunkSpp, sampleBase)`, which does the
// kernel launch + cudaCheckKernel accumulating into the resident device buffers) and
// downloads the running SUM film (via `download(out)`), reporting to `prog` after each
// chunk. Stops when `prog.report` returns true or the requested `spp` is reached. Chunk
// size adapts toward ~0.15 s of GPU work per launch so a wall-clock budget or Ctrl-C is
// honoured promptly without paying per-launch overhead on fast scenes.
template <class LaunchFn, class DownloadFn>
static void gpuSppChunks(long long spp, const SppProgress& prog, Film& out,
                         LaunchFn&& launch, DownloadFn&& download) {
    using clk = std::chrono::steady_clock;
    long long done = 0, chunk = 1;
    while (done < spp) {
        long long c = chunk; if (c > spp - done) c = spp - done;
        auto t0 = clk::now();
        launch(c, done);
        done += c;
        double dt = std::chrono::duration<double>(clk::now() - t0).count();
        if (dt > 1e-4) {                                   // retarget ~0.15 s per chunk
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
                    long long spp, int maxDepth, bool diffraction, const SppProgress* prog) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    if (!cudaAvailable() || !cudaBdptSupported(scene)) return out;
    if (maxDepth > BDPT_MAXDEPTH) maxDepth = BDPT_MAXDEPTH;   // device array bound

    DUpload up;
    buildUpload(scene, cam, resX, resY, up);

    const size_t npix = (size_t)resX * resY;
    double* d_cam   = nullptr; CUDA_CHECK(cudaMalloc(&d_cam,   npix * 3 * sizeof(double)));
    double* d_splat = nullptr; CUDA_CHECK(cudaMalloc(&d_splat, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cam,   0, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_splat, 0, npix * 3 * sizeof(double)));
    // Resume (mode D disk resume): mix the loaded sample count into the seed base so the
    // continued samples are decorrelated from the ones already in the checkpoint film.
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
        kBdpt<<<2048, 128>>>(up.sc, up.dc, d_cam, d_splat, totalSamples, c, spp, base, resX,
                             maxDepth, diffraction ? 1 : 0, seed);
        cudaCheckKernel("bdpt");
    };

    if (!prog || !prog->report) { launch(spp, 0); download(out); }   // single-shot
    else gpuSppChunks(spp, *prog, out, launch, download);

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
    if (scene.anyMedium()) return false;
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
                        long long spp, bool diffraction, const SppProgress* prog) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    if (!cudaAvailable() || !cudaBackwardSupported(scene, cam)) return out;

    DUpload up;
    buildUpload(scene, cam, resX, resY, up);

    const size_t npix = (size_t)resX * resY;
    double* d_film = nullptr; CUDA_CHECK(cudaMalloc(&d_film, npix * 3 * sizeof(double)));
    double* d_hits = nullptr; CUDA_CHECK(cudaMalloc(&d_hits, npix * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_film, 0, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_hits, 0, npix * sizeof(double)));
    // Resume (mode R disk resume): mix the loaded sample count into the seed base so the
    // continued samples are decorrelated from the ones already in the checkpoint film.
    const unsigned long long seed = 0x9e3779b97f4a7c15ULL
        ^ (prog ? (unsigned long long)prog->sampleBase * 0x9E3779B97F4A7C15ULL : 0ULL);

    std::vector<double> film(npix * 3);
    auto download = [&](Film& o) {
        CUDA_CHECK(cudaMemcpy(film.data(), d_film, film.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(o.hits.data(), d_hits, npix * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < npix; ++i)   // SUM over spp; writeFilm divides by spp
            o.xyz[i] = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);
    };
    auto launch = [&](long long c, long long base) {
        long long totalSamples = (long long)npix * c;
        kBackward<<<2048, 128>>>(up.sc, up.dc, d_film, d_hits, totalSamples, c, spp, base, resX,
                                 diffraction ? 1 : 0, seed);
        cudaCheckKernel("backward");
    };

    if (!prog || !prog->report) { launch(spp, 0); download(out); }   // single-shot
    else gpuSppChunks(spp, *prog, out, launch, download);

    freeUpload(up);
    cudaFree(d_film); cudaFree(d_hits);
    return out;
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
                                          bool autoExpose, double* lockAnchor) {
    using namespace gpu;
    if (!cudaIsoPreviewSupported(scene, cam)) return {};   // caller falls back to CPU raster
    if (W <= 0 || H <= 0) return {};

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
    kIsoPreview<<<grid, block>>>(up.sc, up.dc, dpl, dCol, dEmit, (int)scene.mats.size(),
                                 dTexMeta, dTexels, dMatTex, dMatTri,
                                 d_accum, d_z, d_emis, W, H, bg, EMIS_BOOST);
    cudaCheckKernel("iso-preview");

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
    return raster::exposeAndEncode(accum, zf, emis, W, H, nThreads, expComp, autoExpose,
                                   lockAnchor, /*seeThrough=*/false, {}, {}, kMilkColor);
}

// --------------------- photon map (mode M) host ------------------------------

bool cudaPhotonMapSupported(const Scene& scene) {
    // The GPU mode-M path reuses the forward photon tracer (deposit) and a device gather
    // that mirrors photonGather's DIRECT (fgRays == 0) density estimate. Scope for v1:
    //   * same POD-bakeable materials as the forward path (cudaForwardSupported), and
    //   * NO environment light — the device gather has no env term (CPU photonGather adds
    //     env on escape / at diffuse hits); an env scene must stay on the CPU.
    // Final gather (g_pmFinalGather > 0) and physical-lens cameras are gated by the caller
    // (main.cpp) since those are render-config, not scene, properties. Fluorescence is fine:
    // neither CPU nor GPU deposits at a fluorescent vertex, and both gather it as a query
    // point, so the two agree. Participating media (fog) are supported — the forward deposit
    // pass runs the same Woodcock free-flight as the CPU tracePhoton.
    if (!cudaForwardSupported(scene)) return false;
    if (scene.envIndex >= 0) return false;
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
    const char magic[8] = {'F','T','P','M','P','0','1','\n'};
    long long nPh = (long long)pm.photons.size();
    double en[5] = {e.emitted, e.absorbed, e.sensor, e.escaped, e.residual};
    bool ok = true;
    ok = ok && std::fwrite(magic, 1, 8, f) == 8;
    ok = ok && std::fwrite(&guard, sizeof guard, 1, f) == 1;
    ok = ok && std::fwrite(&pm.nEmitted, sizeof pm.nEmitted, 1, f) == 1;
    ok = ok && std::fwrite(en, sizeof en, 1, f) == 1;
    ok = ok && std::fwrite(&nPh, sizeof nPh, 1, f) == 1;
    if (ok && nPh > 0)
        ok = std::fwrite(pm.photons.data(), sizeof(Photon), (size_t)nPh, f) == (size_t)nPh;
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
    if (!ok || std::memcmp(magic, "FTPMP01\n", 8) != 0) {
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
        pm.photons.resize((size_t)nPh);
        ok = std::fread(pm.photons.data(), sizeof(Photon), (size_t)nPh, f) == (size_t)nPh;
    }
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "[loadmap] %s truncated photon data; ignoring\n", path); pm.photons.clear(); return false; }
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
                                            const char* mapLoad, const char* mapSave, int heroC) {
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
    bool mapLoaded = false;
    if (mapLoad && *mapLoad) {
        mapLoaded = loadPhotonMap(mapLoad, pm, eOut, photonMapGuard(scene, diffraction));
        if (mapLoaded) {
            std::printf("[loadmap] %s: %zu photons from %lld emitted -- deposit skipped\n",
                        mapLoad, pm.photons.size(), (long long)pm.nEmitted);
            pm.build(radius);                   // (re)build the grid at the requested radius
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
        // Download + convert to Photon in chunks (never a full host-side DPhoton copy).
        pm.photons.resize((size_t)nDep);
        std::vector<DPhoton> stage;
        for (size_t off = 0; off < (size_t)nDep; off += PM_CHUNK) {
            size_t cnt = std::min(PM_CHUNK, (size_t)nDep - off);
            stage.resize(cnt);
            CUDA_CHECK(cudaMemcpy(stage.data(), d_photons + off, cnt * sizeof(DPhoton),
                                  cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < cnt; ++i) {
                const DPhoton& d = stage[i];
                Photon& p = pm.photons[off + i];
                p.pos = Vec3(d.pos.x, d.pos.y, d.pos.z);
                p.wi  = Vec3(d.wi.x,  d.wi.y,  d.wi.z);
                p.n   = Vec3(d.n.x,   d.n.y,   d.n.z);
                p.power = d.power; p.lambda = d.lambda;
            }
        }
    }
    if (d_photons) cudaFree(d_photons);
    pm.build(radius);                       // host counting sort -> cell-contiguous runs

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
                const Vec3&  ci = pm.cie[off + i];
                DGatherPhoton& d = stage[i];
                d.pos = DVec3(p.pos.x, p.pos.y, p.pos.z);
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
                                   resX[c], diffraction ? 1 : 0, seed);
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
