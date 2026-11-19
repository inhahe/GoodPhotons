// The GPU backend's host side and the kernels without launch bounds. The device code it shares
// with the register-capped units is in render_cuda_dev.cuh and render_cuda_dev2.cuh, whose
// opening comment says why it is split; the account of the backend begins there.
#include "render_cuda_dev.cuh"

namespace gpu {
// Defined in render_cuda_bdpt.cu, under its own register cap.
template <int NS, int MAXD, bool MERGE, bool STRAT>
__global__ void __launch_bounds__(128, 3)
kBdptT(DScene sc, const __grid_constant__ DCamera cam, double* camFilm, double* splatFilm,
                      long long totalSamples, long long chunkSpp, long long sppTotal,
                      long long sampleBase, int resX, int maxDepth,
                      int diffraction, unsigned long long seedBase, int heroC,
                      DBeamMap bm, DSurfMap sm, DWfQueue wq, long long idxBegin, long long idxEnd,
                      DSurfQueue sq);
}  // namespace gpu

#include "render_cuda_dev2.cuh"
#include "merge_grid.h"    // the merge maps' hashed lattice: one with the host's VcmGrid and SurfMap

namespace gpu {
// Defined in render_cuda_jsurf.cu, under its own register cap.
template <int MAXD>
__global__ void __launch_bounds__(128, 4)
kJSurfLightT(DScene sc, DCamera cam, int diffraction, int maxDepth,
             long long nPaths, unsigned long long seedBase, DJSurfOut out);

// ---- DEVICE PROBE (FTRACE_DEVICE_PROBE=1; 0.385.0) -------------------------------------------
// nvcc compiles this file twice -- the host half with the host compiler, the device half with its
// own front end -- and BOTH halves read the HOST compiler's headers. So two things can go wrong
// on the device alone after a host-toolchain change, silently: no CUDA error, a plausible image.
//   * a struct the host fills and uploads is laid out differently by the device front end;
//   * a <cmath> function compiles to something else on the device. MSVC 14.5x's float copysign
//     returned 0 for every input there (see the copysign overloads at the top of namespace gpu),
//     which NaN-ed every sampled direction -- what CMakeLists.txt had blamed on a "miscompile".
// The probe prints sizeof() of every struct that crosses the boundary as each half sees it, and
// evaluates every math function the device code calls (ftMathRow, float and double, edge inputs
// included) on both halves. RUN IT after any change to the host compiler or the CUDA toolkit
// (tools/check_device_probe.py); it found the copysign bug in one build.
#define FT_DEVICE_LAYOUT_TYPES(X) \
    X(DVec3) X(DTexture) X(DMaterial) X(DMediumStack) X(DTri) X(DSphere) \
    X(DCurveSeg) X(DNode) X(DMirrorPlane) X(DBlas) X(DInstance) X(DFieldNode) \
    X(DFieldNodeF) X(DImplicit) X(DVdbGrid) X(DMajorant) X(DMedium) X(DEmitTri) \
    X(DEmitter) X(DEnvMap) X(DRecScalarStop) X(DEmissiveVolume) X(DBowTab) X(DScene) \
    X(DLensSurface) X(DLensSystem) X(DCamera) X(DRng) X(DHit) X(DGrinMedia) \
    X(DHairShade) X(DPhoton) X(DBeamDep) X(DBeamSpec) X(DAimTarget) X(DAimMap) \
    X(DCamSet) X(DGatherPhoton) X(DPhotonMap) X(DTrRay) X(DBeamRec) X(DBeamMis) \
    X(DVolCache) X(DBeamMap) X(DBeamMergeW) X(DSurfPhoton) X(DSurfMis) X(DSurfMap) \
    X(DWfSeg) X(DWfHit) X(DWfQueue) X(DVertex) X(DEscape) X(DHeroBundle) \
    X(DPLight) X(DPreviewLight) X(DPTex) X(DPathSeg) X(DSurfMergeW) X(DSppmState) X(MergeLattice)
enum { FT_MATH_N = 25 };
template <class T> HD inline void ftMathRow(T x, T y, double* o) {
    o[0] = sqrt(x);  o[1] = fabs(x);  o[2] = cos(x);   o[3] = exp(x);   o[4] = sin(x);  o[5] = floor(x);
    o[6] = log(x);   o[7] = acos(x);  o[8] = asin(x);  o[9] = sinh(x);  o[10] = tan(x); o[11] = log2(x);
    o[12] = cosh(x); o[13] = atan(x); o[14] = cbrt(x); o[15] = ceil(x); o[16] = fmax(x, y);
    o[17] = fmin(x, y); o[18] = pow(x, y); o[19] = atan2(x, y); o[20] = copysign(x, y);
    o[21] = nextafter(x, y); o[22] = isfinite(x) ? 1.0 : 0.0; o[23] = ldexp(x, 3);
    o[24] = fma(x, y, (T)0.5);
}
__global__ void kDeviceProbe(unsigned long long* sizes, double* math, const double* xs,
                             const double* ys, int n) {
    int i = 0;
#define FT_SZ(T) sizes[i++] = sizeof(T);
    FT_DEVICE_LAYOUT_TYPES(FT_SZ)
#undef FT_SZ
    for (int r = 0; r < n; ++r) {
        ftMathRow<float>((float)xs[r], (float)ys[r], math + (2 * r) * FT_MATH_N);
        ftMathRow<double>(xs[r], ys[r], math + (2 * r + 1) * FT_MATH_N);
    }
}
// A device map's point's position, for kMergeKey: a light vertex's, a surface photon's, a photon's.
__device__ inline const DVec3& mergePosOf(const DVcmLV& v) { return v.p; }
__device__ inline const DVec3& mergePosOf(const DSurfPhoton& v) { return v.p; }
__device__ inline const DVec3& mergePosOf(const DPhoton& v) { return v.pos; }
// The bucket in `lat` of each of the n points (merge_grid.h, mergeLatticeBucket: the host's binning
// of a point, which a merge site's DGridCell agrees with).
template <class P>
__global__ void kMergeKey(const P* pts, long long n, MergeLattice lat, int* key) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += stride) {
        const DVec3& p = mergePosOf(pts[i]);
        key[i] = (int)mergeLatticeBucket(lat, p.x, p.y, p.z);
    }
}
// The tag of each of the n points the bin's sort put in bucket order (`order`), in that order: what a
// gather tells the points of its own cell from those of other blocks sharing the bucket by.
template <class P>
__global__ void kMergeTag(const P* pts, const int* order, long long n, MergeLattice lat, unsigned char* tag) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += stride) {
        const DVec3& p = mergePosOf(pts[order[i]]);
        tag[i] = mergeLatticeKey(lat, p.x, p.y, p.z).tag;
    }
}
// The merge-lattice half of the probe: the bucket a merge site at each of the n points (xyz, float,
// as a device site's position is) takes in `lat` -- its own cell's, DGridCell's centre neighbour.
__global__ void kMergeCellProbe(MergeLattice lat, const float* pts, int n, unsigned int* out) {
    for (int i = 0; i < n; ++i) {
        const MergeKey k = DGridCell(DVec3((Real)pts[3 * i], (Real)pts[3 * i + 1], (Real)pts[3 * i + 2]), lat)
                               .neighbour(13, lat.slots);
        out[i] = k.bucket * 256u + k.tag;   // the bucket and the tag (MergeKey), the bucket below 2^24 here
    }
}

} // namespace gpu

// ============================ host: bake + launch ============================

// Of `local` bytes of per-thread local memory a family of kernels reserves at its first launch, what
// is still to be reserved: all of it until a launch of ours has reserved as much, then none
// (cudaLocalReservedBytes). A buffer allocated before that launch leaves this much free beside it.
// All of it, not the growth past a smaller reservation held: nothing says the driver frees the old
// block before taking the new one rather than holding both while it swaps (vramPreflight's rule
// too). The one rule for every session budget: the store, the queues, the photon buffers, the
// cell table.
static size_t pendingLocalBytes(size_t local) {
    return (cudaLocalReservedBytes() >= local) ? 0 : local;
}

// THE DEVICE'S VIEW OF `-stop`. A kernel cannot ask ft::stopRequested(), and the thread that
// launched it is parked in a synchronize and can issue no CUDA call, so the device keeps its own
// copy of the flag: one int of MAPPED PINNED host memory, allocated once and kept for the life of
// the process (until cudaGracefulShutdown), which whoever raises or clears the stop writes too
// (ft::stopMirrorSlot; main.cpp raiseStop / clearStop). Kernels read it through the device global
// gpu::g_dGatherStop (dGatherStopped), which is published only inside a DeviceStopScope -- null
// elsewhere, so a renderer that keeps whatever a launch wrote never sees a stop cut a gather short.
// The memory being never freed, a published pointer cannot dangle (GATHER-STOP-DANGLING).
struct DeviceStopFlag { int* host; int* dev; };
static DeviceStopFlag deviceStopFlag() {
    static DeviceStopFlag f = [] {
        DeviceStopFlag r{nullptr, nullptr};
        CUDA_CHECK(cudaHostAlloc((void**)&r.host, sizeof(int), cudaHostAllocMapped));
        *r.host = 0;
        ft::setStopMirror(r.host);              // from here on every raise or clear writes it...
        if (ft::stopRequested()) *r.host = 1;   // ...and a stop raised before then is copied in
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&r.dev, r.host, 0));
        return r;
    }();
    return f;
}
// While one lives, the kernels launched honour -stop (the previous scope's pointer is put back
// when it ends, so scopes nest). Only for a renderer that drops a launch a stop cut short: one
// that kept it would keep a gather stopped part-way, dark. The device's copy starts equal to the
// host's flag: a raise racing a clear (a -serve frame's) could have left it raised over a cleared
// flag, which would cut every launch of the next render.
class DeviceStopScope {
public:
    DeviceStopScope() {
        const DeviceStopFlag f = deviceStopFlag();
        *(volatile int*)f.host = ft::stopRequested() ? 1 : 0;
        CUDA_CHECK(cudaMemcpyFromSymbol(&prev_, gpu::g_dGatherStop, sizeof(int*)));
        CUDA_CHECK(cudaMemcpyToSymbol(gpu::g_dGatherStop, &f.dev, sizeof(int*)));
    }
    ~DeviceStopScope() { cudaMemcpyToSymbol(gpu::g_dGatherStop, &prev_, sizeof(int*)); }   // no throw
    // Around a launch: clear the device's record that a kernel saw the stop, then read it once
    // the launch has completed -- true when the stop cut it short.
    static void clearSeen() {
        const int zero = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(gpu::g_dStopSeen, &zero, sizeof(int)));
    }
    static bool seen() {
        int v = 0;
        CUDA_CHECK(cudaMemcpyFromSymbol(&v, gpu::g_dStopSeen, sizeof(int)));
        return v != 0;
    }
    DeviceStopScope(const DeviceStopScope&) = delete;
    DeviceStopScope& operator=(const DeviceStopScope&) = delete;
private:
    int* prev_ = nullptr;
};

static bool g_queried = false, g_available = false;
static char g_devName[256] = "none";

// Host half of the device probe (see kDeviceProbe). Returns the number of disagreements.
static int deviceProbe() {
    using namespace gpu;
    static const char* tn[] = {
#define FT_NM(T) #T,
        FT_DEVICE_LAYOUT_TYPES(FT_NM)
#undef FT_NM
    };
    static const unsigned long long hs[] = {
#define FT_SZ(T) (unsigned long long)sizeof(T),
        FT_DEVICE_LAYOUT_TYPES(FT_SZ)
#undef FT_SZ
    };
    const int NT = (int)(sizeof(hs) / sizeof(hs[0]));
    static const char* fn[FT_MATH_N] = {"sqrt", "fabs", "cos", "exp", "sin", "floor", "log", "acos",
        "asin", "sinh", "tan", "log2", "cosh", "atan", "cbrt", "ceil", "fmax", "fmin", "pow", "atan2",
        "copysign", "nextafter", "isfinite", "ldexp", "fma"};
    const double xs[] = {0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 3.0, 1e-30, 1e30, INFINITY, -INFINITY, NAN};
    const double ys[] = {-1.5, 2.0, -0.0, 0.5, -3.0, 0.0, -1.0, 2.5, -2.0, 1.0, -0.5, 3.0};
    const int NR = (int)(sizeof(xs) / sizeof(xs[0]));
    unsigned long long* dSz = nullptr; double *dM = nullptr, *dX = nullptr, *dY = nullptr;
    cudaMalloc(&dSz, NT * sizeof(unsigned long long));
    cudaMalloc(&dM, 2 * NR * FT_MATH_N * sizeof(double));
    cudaMalloc(&dX, NR * sizeof(double)); cudaMalloc(&dY, NR * sizeof(double));
    cudaMemcpy(dX, xs, NR * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, ys, NR * sizeof(double), cudaMemcpyHostToDevice);
    kDeviceProbe<<<1, 1>>>(dSz, dM, dX, dY, NR);
    std::vector<unsigned long long> dsz(NT);
    std::vector<double> dm(2 * NR * FT_MATH_N), hm(2 * NR * FT_MATH_N);
    cudaMemcpy(dsz.data(), dSz, NT * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(dm.data(), dM, dm.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dSz); cudaFree(dM); cudaFree(dX); cudaFree(dY);
    int bad = 0;
    for (int k = 0; k < NT; ++k)
        if (dsz[k] != hs[k]) {
            ++bad;
            std::printf("[probe] LAYOUT %-16s host %llu device %llu\n", tn[k], hs[k], dsz[k]);
        }
    for (int r = 0; r < NR; ++r) {
        ftMathRow<float>((float)xs[r], (float)ys[r], hm.data() + (2 * r) * FT_MATH_N);
        ftMathRow<double>(xs[r], ys[r], hm.data() + (2 * r + 1) * FT_MATH_N);
    }
    for (size_t k = 0; k < hm.size(); ++k) {
        const double a = hm[k], b = dm[k];
        const bool same = (std::isnan(a) && std::isnan(b)) || a == b ||
                          (std::isfinite(a) && std::isfinite(b) &&
                           std::fabs(a - b) <= 1e-4 * std::fmax(std::fabs(a), 1e-300));
        if (!same) {
            ++bad;
            const int f = (int)(k % FT_MATH_N), r = (int)(k / FT_MATH_N / 2);
            const bool dbl = ((k / FT_MATH_N) % 2) != 0;
            std::printf("[probe] MATH %s<%s>(%g, %g)  host %.9g  device %.9g\n", fn[f],
                        dbl ? "double" : "float", xs[r], ys[r], a, b);
        }
    }
    // Merge lattices far from the origin and points a rounding or two either side of their cell
    // planes, as floats: the device must place each in the bucket the host bins it in (merge_grid.h,
    // mergeLatticeBucket). A lattice the device holds in another rounding than the binning's takes
    // the cell beside near a plane, and its gather misses what lies two cells off.
    int nMerge = 0;
    {
        const MergeLattice lats[2] = {mergeLatticeAt(1.0e4 + 0.123, -2.5e3 + 0.7, 750.0137, 0.0037, 61),
                                      mergeLatticeAt(-31.4, 2.71, 1.0e3 + 0.5, 0.0213, 4093)};
        for (const MergeLattice& lat : lats) {
            std::vector<float> pts;
            for (int a = 0; a < 3; ++a)
                for (long long k : {1LL, 977LL, 23456LL, 270000LL}) {
                    const float plane = (float)(lat.lo[a] + (double)k * lat.cell);
                    float x = plane;
                    for (int j = 0; j < 3; ++j) x = std::nextafter(x, -INFINITY);
                    for (int j = 0; j < 7; ++j, x = std::nextafter(x, INFINITY)) {
                        float q[3];
                        for (int b = 0; b < 3; ++b) q[b] = (float)(lat.lo[b] + 0.5 * lat.cell);
                        q[a] = x;
                        pts.insert(pts.end(), q, q + 3);
                    }
                }
            const int n = (int)pts.size() / 3;
            float* dP = nullptr; unsigned int* dO = nullptr;
            cudaMalloc(&dP, pts.size() * sizeof(float)); cudaMalloc(&dO, n * sizeof(unsigned int));
            cudaMemcpy(dP, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);
            kMergeCellProbe<<<1, 1>>>(lat, dP, n, dO);
            std::vector<unsigned int> dev(n);
            cudaMemcpy(dev.data(), dO, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaFree(dP); cudaFree(dO);
            int apart = 0;
            for (int i = 0; i < n; ++i) {
                const MergeKey h = mergeLatticeKey(lat, pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
                if (dev[i] != h.bucket * 256u + h.tag) ++apart;
            }
            if (apart) std::printf("[probe] MERGE lattice at (%g, %g, %g), cell %g: %d of %d points in another "
                                   "bucket on the device than the host bins them in\n",
                                   lat.lo[0], lat.lo[1], lat.lo[2], lat.cell, apart, n);
            nMerge += n;
            bad += apart;
        }
    }
    std::printf("[probe] %d disagreement(s) between the host and device compilers (%d structs, %d math values, "
                "%d merge cells)\n", bad, NT, (int)hm.size(), nMerge);
    return bad;
}

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
    if (const char* e = std::getenv("FTRACE_DEVICE_PROBE")) if (*e && std::strcmp(e, "0")) deviceProbe();
    return true;
}
const char* cudaDeviceName() { cudaAvailable(); return g_devName; }

// ---- N4a: host half of the mode-W lattice bit-exactness probe (-checklattice) ----
// See lattice_probe.h for the column layout and render_cuda.h for the contract. The base
// list travels as a plain device buffer rather than a `__constant__` symbol, so this needs
// no API beyond the handful the HIP shim at the top of this file already covers.
int cudaRealBytes() { return gpu::realBytes(); }
bool cudaPatOpProbe(const PatNode* prog, const int* off, const int* len, int nProg,
                    const PatOpProbeIn& in, double* outF64, double* outF32) {
    if (!cudaAvailable() || nProg <= 0) return false;
    int total = 0;
    for (int i = 0; i < nProg; ++i) total = (off[i] + len[i] > total) ? off[i] + len[i] : total;
    std::vector<gpu::PatNodeF> hF((size_t)total);
    for (int i = 0; i < total; ++i) { hF[i].op = (int)prog[i].op; hF[i].a = (float)prog[i].a; }

    PatNode*  dP  = nullptr; gpu::PatNodeF* dPF = nullptr;
    int*      dO  = nullptr; int*      dL  = nullptr;
    double*   d64 = nullptr; double*   d32 = nullptr;
    PatOpProbeIn* dIn = nullptr;
    bool ok = cudaMalloc(&dP,  sizeof(PatNode)  * total) == cudaSuccess
           && cudaMalloc(&dPF, sizeof(gpu::PatNodeF) * total) == cudaSuccess
           && cudaMalloc(&dO,  sizeof(int) * nProg) == cudaSuccess
           && cudaMalloc(&dL,  sizeof(int) * nProg) == cudaSuccess
           && cudaMalloc(&d64, sizeof(double) * nProg) == cudaSuccess
           && cudaMalloc(&d32, sizeof(double) * nProg) == cudaSuccess
           && cudaMalloc(&dIn, sizeof(PatOpProbeIn)) == cudaSuccess;
    if (ok) ok = cudaMemcpy(dP,  prog,     sizeof(PatNode)  * total, cudaMemcpyHostToDevice) == cudaSuccess
              && cudaMemcpy(dPF, hF.data(), sizeof(gpu::PatNodeF) * total, cudaMemcpyHostToDevice) == cudaSuccess
              && cudaMemcpy(dO,  off,      sizeof(int) * nProg, cudaMemcpyHostToDevice) == cudaSuccess
              && cudaMemcpy(dL,  len,      sizeof(int) * nProg, cudaMemcpyHostToDevice) == cudaSuccess
              && cudaMemcpy(dIn, &in,       sizeof(PatOpProbeIn), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        gpu::kPatOpProbe<<<(nProg + 127) / 128, 128>>>(dP, dPF, dO, dL, nProg, dIn, d64, d32);
        ok = cudaDeviceSynchronize() == cudaSuccess
          && cudaMemcpy(outF64, d64, sizeof(double) * nProg, cudaMemcpyDeviceToHost) == cudaSuccess
          && cudaMemcpy(outF32, d32, sizeof(double) * nProg, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(dP); cudaFree(dPF); cudaFree(dO); cudaFree(dL); cudaFree(d64); cudaFree(d32);
    cudaFree(dIn);
    return ok;
}

// ---- -checkjsurfw: the tap's buffers (jsurf_tap.h) ------------------------------------------
static gpu::DSurfTap g_surfTapHost = {0, 0, nullptr, nullptr, nullptr, nullptr};
static void surfTapFree() {
    cudaFree(g_surfTapHost.eye); cudaFree(g_surfTapHost.nE); cudaFree(g_surfTapHost.lam);
    cudaFree(g_surfTapHost.halves);
    g_surfTapHost = gpu::DSurfTap{0, 0, nullptr, nullptr, nullptr, nullptr};
    cudaMemcpyToSymbol(gpu::c_surfTap, &g_surfTapHost, sizeof g_surfTapHost);
}
void cudaJSurfTapArm(int n, int stride) {
    if (!cudaAvailable()) return;
    surfTapFree();
    if (n <= 0) return;
    stride = std::min(std::max(stride, 1), kJSurfTapMaxV);
    gpu::DSurfTap t{n, stride, nullptr, nullptr, nullptr, nullptr};
    const size_t nv = (size_t)n * (size_t)stride;
    bool ok = cudaMalloc(&t.eye, nv * sizeof(JSurfTapVertex)) == cudaSuccess &&
              cudaMalloc(&t.nE, (size_t)n * sizeof(int)) == cudaSuccess &&
              cudaMalloc(&t.lam, (size_t)n * sizeof(double)) == cudaSuccess &&
              cudaMalloc(&t.halves, nv * sizeof(JSurfTapHalf)) == cudaSuccess;
    ok = ok && cudaMemset(t.nE, 0, (size_t)n * sizeof(int)) == cudaSuccess &&
         cudaMemset(t.halves, 0, nv * sizeof(JSurfTapHalf)) == cudaSuccess;
    g_surfTapHost = t;
    if (!ok) { surfTapFree(); return; }
    cudaMemcpyToSymbol(gpu::c_surfTap, &t, sizeof t);
}
bool cudaJSurfTapTake(std::vector<JSurfTapVertex>& eye, std::vector<int>& nE,
                      std::vector<double>& lam, std::vector<JSurfTapHalf>& halves, int& stride) {
    const gpu::DSurfTap t = g_surfTapHost;
    if (!cudaAvailable() || t.n <= 0) return false;
    stride = t.stride;
    const size_t nv = (size_t)t.n * (size_t)t.stride;
    eye.resize(nv); nE.resize((size_t)t.n); lam.resize((size_t)t.n); halves.resize(nv);
    const bool ok =
        cudaMemcpy(eye.data(), t.eye, nv * sizeof(JSurfTapVertex), cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(nE.data(), t.nE, (size_t)t.n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(lam.data(), t.lam, (size_t)t.n * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(halves.data(), t.halves, nv * sizeof(JSurfTapHalf), cudaMemcpyDeviceToHost) == cudaSuccess;
    surfTapFree();
    return ok;
}
bool cudaJSurfWEval(const std::vector<JSurfTapHalf>& halves,
                    const std::vector<JSurfTapState>& states, std::vector<double>& out) {
    if (!cudaAvailable() || halves.empty() || states.empty()) return false;
    const int n = (int)states.size();
    JSurfTapHalf* dH = nullptr; JSurfTapState* dS = nullptr; double* dO = nullptr;
    bool ok = cudaMalloc(&dH, halves.size() * sizeof(JSurfTapHalf)) == cudaSuccess &&
              cudaMalloc(&dS, states.size() * sizeof(JSurfTapState)) == cudaSuccess &&
              cudaMalloc(&dO, states.size() * sizeof(double)) == cudaSuccess;
    if (ok) ok = cudaMemcpy(dH, halves.data(), halves.size() * sizeof(JSurfTapHalf),
                            cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(dS, states.data(), states.size() * sizeof(JSurfTapState),
                            cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        gpu::kJSurfWEval<<<(n + 127) / 128, 128>>>(dH, dS, n, dO);
        out.resize(states.size());
        // A launch's own error, taken here so the render's next kernel check does not report it.
        ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
             cudaMemcpy(out.data(), dO, states.size() * sizeof(double),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(dH); cudaFree(dS); cudaFree(dO);
    return ok;
}

bool cudaRainbowSampleHist(const rainbow::RainbowPhase& rp, double lambda, long long n,
                           std::vector<double>& counts) {
    if (!cudaAvailable() || !rp.built() || n <= 0) return false;
    const std::vector<double>& pdf = rp.pdfTable();
    const std::vector<double>& cdf = rp.cdfTable();
    const size_t nb = (size_t)rp.nMu() - 1;
    double *dPdf = nullptr, *dCdf = nullptr, *dHist = nullptr;
    bool ok = cudaMalloc(&dPdf, pdf.size() * sizeof(double)) == cudaSuccess &&
              cudaMalloc(&dCdf, cdf.size() * sizeof(double)) == cudaSuccess &&
              cudaMalloc(&dHist, nb * sizeof(double)) == cudaSuccess;
    if (ok) ok = cudaMemcpy(dPdf, pdf.data(), pdf.size() * sizeof(double),
                            cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(dCdf, cdf.data(), cdf.size() * sizeof(double),
                            cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemset(dHist, 0, nb * sizeof(double)) == cudaSuccess;
    if (ok) {
        gpu::DMedium m{};
        m.rbPdf = dPdf; m.rbCdf = dCdf;
        m.rbNLam = rp.nLam(); m.rbNMu = rp.nMu();
        m.rbLam0 = rp.lam0(); m.rbDLam = rp.dLam();
        gpu::kRainbowSampleHist<<<2048, 128>>>(m, (gpu::Real)lambda, n, 0x5EEDF00DULL, dHist);
        ok = cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        counts.assign(nb, 0.0);
        ok = cudaMemcpy(counts.data(), dHist, nb * sizeof(double),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (dPdf)  cudaFree(dPdf);
    if (dCdf)  cudaFree(dCdf);
    if (dHist) cudaFree(dHist);
    return ok;
}

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

void cudaGracefulShutdown(bool force) {
    // Only touch the driver if a device was actually brought up; probing when none is
    // present (or CUDA never initialised) would needlessly spin up a context just to
    // tear it down. cudaAvailable() is cached and cheap. `force`: another library in this
    // process (OIDN) used the GPU, so its context is there to tear down even if ours never was.
    if (!g_queried && force) cudaAvailable();
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
    ft::setStopMirror(nullptr);   // the reset frees deviceStopFlag's memory: no stop may write it
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
        case EmitterShape::Glow:     return 7;
    }
    return -1;   // unknown / newly-added shape: CPU fallback, never a silent coercion
}

bool cudaForwardSupported(const Scene& scene) {
    // Object-space patterns (`pattern_space object`) are read on the host only so far: every
    // device path renders such a scene on the CPU rather than read its patterns in world space.
    if (scene.hasObjectSpacePatterns()) return false;
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
               (scene.mats[matId].type == MatType::Multilayer ||
                scene.mats[matId].iridescence == 2) &&       // a rough multilayer too
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
        // The physical layered stack (coat interface over a weighted body) RUNS ON THE DEVICE
        // since 0.317.0: uploadScene gives each coat a synthetic glossy lobe and every
        // dispatch site resolves the stack through dResolveCompound, exactly as it resolves a
        // mix. Only the body-lobe COUNT is still bounded, by the same D_MIXMAX the mix arrays
        // impose -- a stack with more lobes than that has nowhere to put them.
        if (matId >= 0 && matId < (int)scene.mats.size() &&
            scene.mats[matId].type == MatType::Layered) {
            const Material& ly = scene.mats[matId];
            if ((int)ly.mixChildren.size() > D_MIXMAX) return true;
            for (int c : ly.mixChildren)
                if (oversizedMultilayer(c) || usesRecord(c) || usesPaletteTex(c)) return true;
        }
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
    // or dEmitterSamplePoint (the sampler side) rather than reading emitSpd raw. No
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
    // bkNeeVolumeHero/bkNeeLightRGB do the cone NEE, and the ray-miss paths add the directly-
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
    // Knot arrays already uploaded, by (table, scale) (bakeSpec): a table that many materials or
    // emitters share goes up once.
    struct Knots { const float* kx; const double* ky; const unsigned int* kb; int kn; };
    std::map<std::pair<const void*, double>, Knots> knots;
    // Record a device allocation for later freeUpload(); returns it for chaining.
    void* keep(void* p) { if (p) frees.push_back(p); return p; }
};
static void freeUpload(DUpload& up) {
    for (void* p : up.frees) cudaFree(p);
    up.frees.clear();
}

// Bake a Spectrum into a DSpec: its grid (bakeSpec above), and -- when the host can see the curve
// is piecewise linear (spectrumKnots) -- its knots, uploaded through `up`, so the device reads the
// curve itself rather than its samples 4.9 nm apart. Sets every field, so an uninitialised DSpec
// is fine to pass.
static void bakeSpec(const Spectrum& s, gpu::DSpec& d, DUpload& up) {
    bakeSpec(s, d.tab);
    d.kx = nullptr; d.ky = nullptr; d.kb = nullptr; d.kn = 0;
    std::vector<double> x, y;
    const void* id = nullptr;
    double scale = 1.0;
    if (!s || !spectrumKnots(s, x, y, &id, &scale)) return;
    const int n = (int)x.size();
    if (n < 2) return;
    const auto key = std::make_pair(id, scale);
    if (id) {
        const auto it = up.knots.find(key);
        if (it != up.knots.end()) {
            d.kx = it->second.kx; d.ky = it->second.ky; d.kb = it->second.kb; d.kn = it->second.kn;
            return;
        }
    }
    std::vector<float> kx((size_t)n);
    for (int i = 0; i < n; ++i) kx[(size_t)i] = (float)x[(size_t)i];
    std::vector<unsigned int> kb((size_t)gpu::DSPEC_KB);
    int k = 0;
    for (int b = 0; b < gpu::DSPEC_KB; ++b) {     // the last knot at or below DLMIN + b
        const double w = DLMIN + (double)b;
        while (k + 1 < n && x[(size_t)k + 1] <= w) ++k;
        kb[(size_t)b] = (unsigned int)std::min(k, n - 2);
    }
    d.kx = (const float*)up.keep(uploadVec(kx));
    d.ky = (const double*)up.keep(uploadVec(y));
    d.kb = (const unsigned int*)up.keep(uploadVec(kb));
    d.kn = n;
    if (id) up.knots.emplace(key, DUpload::Knots{d.kx, d.ky, d.kb, d.kn});
}

// Motion blur (ROADMAP 11): the scene's motion sets, its primitives' sets and vertex keys, and
// its movers' segment trees (Scene::bvhMove, each leaf naming global primitives), into
// up.sc.mo. A still scene uploads nothing (mo.sets stays null, and no query looks at time).
static void uploadMotion(const Scene& scene, DUpload& up) {
    using namespace gpu;
    DMotion& mo = up.sc.mo;
    mo = DMotion{};
    if (scene.motionSets.empty() && scene.vertexKeys.empty()) return;
    auto keep = [&](void* p) { if (p) up.frees.push_back(p); return p; };
    // A set with no keys never happens, but a deforming scene may have no set at all: one
    // identity key keeps `sets` non-null, the marker that the scene moves.
    std::vector<DAff> xf;
    std::vector<double> T;
    std::vector<DMotionSet> sets;
    for (const MotionKeys& mk : scene.motionSets) sets.push_back(DMotionSet{mk.keys, mk.first, mk.shape});
    for (size_t k = 0; k < scene.motionXf.size(); ++k) {
        DAff a;
        for (int i = 0; i < 9; ++i) a.m[i] = scene.motionXf[k].m[i];
        a.t[0] = scene.motionXf[k].t.x; a.t[1] = scene.motionXf[k].t.y; a.t[2] = scene.motionXf[k].t.z;
        xf.push_back(a);
        T.push_back(scene.motionT[k]);
    }
    if (sets.empty()) {
        DAff id{};
        id.m[0] = id.m[4] = id.m[8] = 1.0;
        xf.push_back(id); T.push_back(0.0);
        sets.push_back(DMotionSet{1, 0, kMotionRigid});
    }
    mo.xf = (const DAff*)keep(uploadVec(xf));
    mo.T = (const double*)keep(uploadVec(T));
    mo.sets = (const DMotionSet*)keep(uploadVec(sets));
    mo.nSets = (int)sets.size();
    // Each kind's sets, and the movers by global number (the BVH's [tris | spheres | implicits |
    // curve segments | instances]).
    const size_t nT = scene.tris.size(), nS = scene.spheres.size(), nI = scene.implicits.size(),
                 nC = scene.curveSegs.size(), nN = scene.instances.size();
    std::vector<unsigned char> moves(nT + nS + nI + nC + nN, 0);
    std::vector<int> tm(nT), sm(nS), im(nI), cm(nC), nm(nN);
    std::vector<int2> vk;
    if (!scene.vertexKeys.empty()) vk.assign(nT, make_int2(-1, 0));
    for (size_t i = 0; i < nT; ++i) {
        const Tri& t = scene.tris[i];
        tm[i] = t.motion;
        if (!vk.empty() && t.vkeys >= 0) vk[i] = make_int2(t.vkeys, t.vkeyCount);
        moves[i] = (t.motion >= 0 || t.vkeys >= 0) ? 1 : 0;
    }
    for (size_t i = 0; i < nS; ++i) { sm[i] = scene.spheres[i].motion; moves[nT + i] = sm[i] >= 0; }
    for (size_t i = 0; i < nI; ++i) { im[i] = scene.implicits[i].motion; moves[nT + nS + i] = im[i] >= 0; }
    for (size_t i = 0; i < nC; ++i) { cm[i] = scene.curveSegs[i].motion; moves[nT + nS + nI + i] = cm[i] >= 0; }
    for (size_t i = 0; i < nN; ++i) { nm[i] = scene.instances[i].motion; moves[nT + nS + nI + nC + i] = nm[i] >= 0; }
    mo.triMotion = tm.empty() ? nullptr : (const int*)keep(uploadVec(tm));
    mo.sphMotion = sm.empty() ? nullptr : (const int*)keep(uploadVec(sm));
    mo.impMotion = im.empty() ? nullptr : (const int*)keep(uploadVec(im));
    mo.curveMotion = cm.empty() ? nullptr : (const int*)keep(uploadVec(cm));
    mo.instMotion = nm.empty() ? nullptr : (const int*)keep(uploadVec(nm));
    mo.primMoves = moves.empty() ? nullptr : (const unsigned char*)keep(uploadVec(moves));
    if (!vk.empty()) {
        std::vector<DVec3> keys(scene.vertexKeys.size());
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i] = {(Real)scene.vertexKeys[i].x, (Real)scene.vertexKeys[i].y, (Real)scene.vertexKeys[i].z};
        mo.triVkeys = (const int2*)keep(uploadVec(vk));
        mo.vertexKeys = (const DVec3*)keep(uploadVec(keys));
    }
    // The segment trees, concatenated: children by absolute node index, leaves into segPrim.
    std::vector<int> roots;
    std::vector<DNode> nodes;
    std::vector<int> prims;
    for (size_t k = 0; k < scene.bvhMove.size(); ++k) {
        const Bvh& b = scene.bvhMove[k];
        const int nodeOff = (int)nodes.size(), primOff = (int)prims.size();
        roots.push_back(b.nodes.empty() ? -1 : nodeOff);
        for (const BvhNode& n : b.nodes) {
            DNode d;
            d.lo = {n.box.lo.x, n.box.lo.y, n.box.lo.z};
            d.hi = {n.box.hi.x, n.box.hi.y, n.box.hi.z};
            d.count = n.count;
            d.left = n.count > 0 ? n.left : n.left + nodeOff;
            d.right = n.count > 0 ? n.right : n.right + nodeOff;
            d.first = n.count > 0 ? n.first + primOff : n.first;
            nodes.push_back(d);
        }
        for (int lp : b.primIdx) prims.push_back(scene.movePrim[k][(size_t)lp]);
    }
    mo.nSeg = (int)roots.size();
    mo.segRoot = roots.empty() ? nullptr : (const int*)keep(uploadVec(roots));
    mo.segNodes = nodes.empty() ? nullptr : (const DNode*)keep(uploadVec(nodes));
    mo.segPrim = prims.empty() ? nullptr : (const int*)keep(uploadVec(prims));
    mo.motionA = scene.motionA;
    mo.motionB = scene.motionB;
    mo.sweptCenter = {(Real)scene.sweptCenter.x, (Real)scene.sweptCenter.y, (Real)scene.sweptCenter.z};
    mo.sweptRadius = scene.sweptRadius;
    mo.sweptReach = scene.sweptReach;
    const Scene::LightTreeTimed& lt = scene.lightTreeTimed;
    if (lt.built && !lt.nodes.empty()) {
        mo.ltNodes = (const LightTreeNode*)keep(uploadVec(lt.nodes));
        mo.ltRoot = lt.root;
        mo.ltAlways = lt.always.empty() ? nullptr : (const int*)keep(uploadVec(lt.always));
        mo.nLtAlways = (int)lt.always.size();
        mo.ltParent = lt.parent.empty() ? nullptr : (const int*)keep(uploadVec(lt.parent));
        mo.ltLeaf = lt.leaf.empty() ? nullptr : (const int*)keep(uploadVec(lt.leaf));
        mo.nLtLeaf = (int)lt.leaf.size();
    }
    if (!scene.mirrorPlaneParts.empty()) {
        std::vector<DMirrorPlane> parts(scene.mirrorPlaneParts.size());
        for (size_t i = 0; i < parts.size(); ++i) {
            const Scene::MirrorPlane& m = scene.mirrorPlaneParts[i]; DMirrorPlane& d = parts[i];
            d.nx = m.n.x; d.ny = m.n.y; d.nz = m.n.z; d.d = m.d;
            d.lox = m.lo.x; d.loy = m.lo.y; d.loz = m.lo.z;
            d.hix = m.hi.x; d.hiy = m.hi.y; d.hiz = m.hi.z;
            d.motionSet = m.motionSet; d.mixed = m.mixed ? 1 : 0; d.part = m.part ? 1 : 0;
            d.deformTri = m.deformTri; d.partId = m.partId; d.keyBoxes = m.keyBoxes;
        }
        mo.mirrorParts = (const DMirrorPlane*)keep(uploadVec(parts));
        mo.nMirrorParts = (int)parts.size();
    }
    if (!scene.mirrorDeformPart.empty())
        mo.mirrorDeformPart = (const int*)keep(uploadVec(scene.mirrorDeformPart));
    if (!scene.mirrorPartBoxes.empty()) {
        std::vector<double> boxes;
        for (const Aabb& b : scene.mirrorPartBoxes)
            for (double v : {b.lo.x, b.lo.y, b.lo.z, b.hi.x, b.hi.y, b.hi.z}) boxes.push_back(v);
        mo.mirrorPartBoxes = (const double*)keep(uploadVec(boxes));
    }
}

// Bake the std::function Scene into POD device tables and upload them (camera-
// independent). Every cudaMalloc'd pointer is recorded in up.frees; call freeUpload(up)
// when done. The camera is baked separately (bakeCamera) so one baked scene can serve a
// whole multi-camera shared pass.
// Mode J's estimator switches (see cudaSetJSwitches): uploaded at the top of every renderBdptCuda,
// and read by the scene upload (UPBP-W-CHROMATIC's table).
static int g_jSpecHost = 1, g_jTrSpecHost = 0, g_jDeltaMergeHost = 1, g_jHalfHost = 0;
static int g_jChromaHost = 1;

// Mode J's wavefront queues' capacities. The VRAM preflight budgets the beam queue's
// (cudaModeJBeamQueueBytes); the point merges' queue takes what room is left.
static constexpr int       kWfSegCap   = 1 << 20;    // beam segments (gpu::DWfSeg)
static constexpr int       kWfHitCap   = 1 << 25;    // beam hits (gpu::DWfHit)
static constexpr long long kSurfReqCap = 1LL << 20;  // point-merge sites (gpu::DSurfReq)
static size_t residentThreads();   // the device-memory preflight's, below
// One point-merge request's bytes: the request, and the JSPEC carrier and chromatic weights'
// bins where those are on.
static size_t surfReqBytes() {
    return sizeof(gpu::DSurfReq) + (g_jSpecHost ? sizeof(gpu::DJSpec) : 0) +
           ((g_jTrSpecHost && g_jChromaHost) ? 2 * DJSPEC_K * sizeof(double) : 0);
}
// The requests a merge queue gets (modes J and U: DSurfQueue, DVcmMergeQueue): up to `hardCap`, or
// what an eighth of `room` holds at `perReq` bytes a request; 0, the queue left out, below 65 536.
// The diagnostic `capEnv` (FTRACE_SURFWFCAP, FTRACE_VCMWFCAP) shrinks it to its value where that is
// at least 16 and below the cap, and then sets `shrunk`, which lets the queue's waves go down to 256
// paths.
static long long mergeQueueCap(size_t room, size_t perReq, long long hardCap, const char* capEnv, bool& shrunk) {
    long long cap = std::min(hardCap, (long long)(room / 8 / perReq));
    shrunk = false;
    if (const char* e = std::getenv(capEnv)) {
        const long long v = std::atoll(e);
        if (v >= 16 && v < cap) { cap = v; shrunk = true; }
    }
    return (shrunk || cap >= 65536) ? cap : 0;
}
// The threads kernel `fn` keeps resident across the device in blocks of 128; 0 where the runtime
// cannot say. A merge queue's wave of fewer paths than its camera kernel's idles the device.
static long long residentThreadsOf(const void* fn) {
    int occ = 0, dev = 0, sms = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 128, 0) == cudaSuccess && occ > 0 &&
        cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess)
        return (long long)occ * 128 * sms;
    (void)cudaGetLastError();
    return 0;
}

// gpu::c_lenScale, a process-wide constant, for the scene about to be rendered: its lenScale
// (Scene::build), so the device's floors and the host's eps scale together. Set with every upload, before
// a call renders a held scene (sceneOrHeld), and at every pass of a resident session (dSetLenScale from
// the uploaded DScene's copy), so a scene uploaded in between cannot leave its own there.
static void setSceneScale(const Scene& scene) {
    const double lenScale = scene.lenScale;
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_lenScale, &lenScale, sizeof lenScale));
}
static void dSetLenScale(const gpu::DScene& sc) {
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_lenScale, &sc.lenScale, sizeof sc.lenScale));
}

static void buildUploadScene(const Scene& scene, DUpload& up) {
    using namespace gpu;
    auto keep = [&](void* p) { if (p) up.frees.push_back(p); return p; };
    setSceneScale(scene);

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
        d.bitangentSign = (Real)t.bitangentSign;   // +-1, exact
        d.emitter = t.emitter;
        d.curvature = t.curvature;      // O3 per-face mean curvature (from Tri::finalize)
        d.solid = t.solidFace;          // PROBE-SOLID-SIDE (markSolidFaces)
        d.bounds = t.boundsSolid;        // (Tri::boundsSolid, a short already)
        d.matId = t.matId; d.sensorId = t.sensorId;
        d.vcol  = t.vcol;               // per-vertex colour (Scene::vertColors), or -1
        d.reach = (Real)t.reach;
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
        d.frame = s.frame;
    }
    std::vector<DVec3> sphAxes;
    for (const Scene::SphereAxes& a : scene.sphereAxes)
        for (const Vec3& v : {a.ax, a.ay, a.az}) sphAxes.push_back(DVec3{(Real)v.x, (Real)v.y, (Real)v.z});
    // Flat mirror surfaces, grouped per plane by Scene::buildMirrorPlanes (which the host
    // already ran inside Scene::build). A straight field-for-field copy: the device solves
    // the reflection in the same doubles the host does, so there is nothing to narrow.
    std::vector<DMirrorPlane> mirrorPlanes(scene.mirrorPlanes.size());
    for (size_t i = 0; i < scene.mirrorPlanes.size(); ++i) {
        const Scene::MirrorPlane& m = scene.mirrorPlanes[i]; DMirrorPlane& d = mirrorPlanes[i];
        d.nx = m.n.x; d.ny = m.n.y; d.nz = m.n.z; d.d = m.d;
        d.lox = m.lo.x; d.loy = m.lo.y; d.loz = m.lo.z;
        d.hix = m.hi.x; d.hiy = m.hi.y; d.hiz = m.hi.z;
        d.motionSet = m.motionSet; d.mixed = m.mixed ? 1 : 0; d.part = m.part ? 1 : 0;
        d.deformTri = m.deformTri; d.partId = m.partId; d.keyBoxes = m.keyBoxes;
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
    std::vector<int> instEmit;   // MeshInstance::emitOf, flattened (DInstance::emitOff)
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
        for (int k = 0; k < 9; ++k) d.Wm[k] = in.toWorld.m[k];   // hit point, tangent local->world
        d.Wt[0] = in.toWorld.t.x; d.Wt[1] = in.toWorld.t.y; d.Wt[2] = in.toWorld.t.z;
        d.curvScale = in.curvScale;                              // curvature local->world (O3)
        d.handSign = in.handSign;
        d.reachMul = in.reachMul; d.reachAdd = in.reachAdd;
        d.blasId = in.blasId; d.matOverride = in.matOverride; d.route = in.route;
        d.emitOff = (int)(instEmit.size() / 2); d.emitN = (int)in.emitOf.size();
        for (const auto& [m, e] : in.emitOf) { instEmit.push_back(m); instEmit.push_back(e); }
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
        d.emitter = im.emitter;
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
        d.solidSide = im.solidSide ? 1 : 0;
        d.readsCurv = im.readsCurv ? 1 : 0;
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
    // Sized to the scene's materials, then GROWN by one synthetic lobe per layered material
    // (see the loop after the bake). Nothing indexes this array by host material id beyond
    // scene.mats.size(), and a hit finds its emitter by its primitive (dEmitterForHit), so the
    // extra entries are invisible to everything that does not ask for them.
    std::vector<DMaterial> mats(scene.mats.size());
    std::vector<double> fluoCdfAll;
    std::vector<double> coatTbarAll;   // the layered materials' T-bar tables (DMaterial::coatTbarOff)
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
        bakeSpec(m.reflect, d.reflect, up);
        bakeSpec(m.ior, d.ior, up);
        bakeSpec(m.substrateK, d.substrateK, up);
        bakeSpec(m.absorb, d.absorb, up);   // Beer-Lambert interior tint (colored glass)
        bakeSpec(m.transmit, d.transmit, up); // diffuse-transmission back-lobe albedo (translucent)
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
        d.solidId  = m.solidId((int)i); // routed copies of one material are one solid
        d.encSolid = i < scene.encSolidOf.size() ? scene.encSolidOf[i] : -1;
        d.routeTo[0] = m.routeTo[0]; d.routeTo[1] = m.routeTo[1];
        d.absorbChroma = m.absorbChroma ? 1 : 0;
        d.reflectTex = m.reflectTex;
        d.triplanarScale = m.triplanarScale;
        // Fluorescence tables (zero/inert for every non-fluorescent material).
        bakeSpec(m.fluoAbsorb, d.fluoAbsorb, up);
        d.fluoYield = m.fluoYield;
        if (m.type == MatType::Fluorescent && !m.fluoEmitSampler.cdf.empty() &&
            m.fluoEmitSampler.integral > 0.0) {
            d.fluoCdfOffset = (int)fluoCdfAll.size();
            d.fluoCdfN = (int)m.fluoEmitSampler.cdf.size();
            d.fluoCdfStep = m.fluoEmitSampler.step;
            fluoCdfAll.insert(fluoCdfAll.end(), m.fluoEmitSampler.cdf.begin(),
                              m.fluoEmitSampler.cdf.end());
            bakeSpec(m.fluoEmit, d.fluoEmitSpec, up);       // continuous M(lambda) for backward gOut
            d.fluoMint = m.fluoEmitSampler.integral;
        } else {
            d.fluoCdfOffset = 0; d.fluoCdfN = 0; d.fluoCdfStep = 1.0;
            for (int s = 0; s < SPEC_N; ++s) d.fluoEmitSpec.tab[s] = 0.0;
            d.fluoMint = 0.0;
        }
        // Excitation CDF (absorb x illuminant), appended to the SAME flat buffer with
        // its own slice. N == 0 makes the device sampler fall back to the emitters' density
        // where the host's `fluoInSampler.integral > 0 ? ... : scene.camSampler` takes the
        // camera's: the product is then 0 everywhere, so the dye absorbs at no wavelength the
        // scene emits and either draw only ends the branch.
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
        bakeSpec(m.filmIor, d.filmIor, up);
        d.filmThickness = m.filmThickness;
        d.roughnessTex = m.roughnessTex;
        d.filmThicknessTex = m.filmThicknessTex;
        d.normalTex = m.normalTex;
        d.normalStrength = m.normalStrength;
        d.iridescence = m.iridescence;
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
        d.coatModel = m.coatModel;          // D_LAYERED coat (filled out below)
        d.coatSpecular = m.coatSpecular;
        d.coatChild = -1;                   // filled by the coat-synthesis loop after this bake
        d.coatLobe = 0;
        d.coatFdr   = m.coatFdr;            // >0 on a body copy under a coat (scene.h coatedAlbedo)
        bakeSpec(m.coatAbsorb, d.coatAbsorb, up);   // A2: sigma_a inside the coat layer, 1/m
        d.coatPathIo = m.coatPathIo;
        d.coatPathRt = m.coatPathRt;
        d.coatMomN = m.coatMomN;                // A3: 0 keeps the analytic series
        for (int q = 0; q < 8; ++q) d.coatMom[q] = m.coatMom[q];
        d.coatStack = m.coatStack;
        d.coatParent = m.coatParent;
        d.coatTbarOff = -1;
        d.coatTbarNT = 1;
        if (!m.coatTbar.empty()) {
            d.coatTbarOff = (int)coatTbarAll.size();
            d.coatTbarNT = m.coatTbarNT;
            coatTbarAll.insert(coatTbarAll.end(), m.coatTbar.begin(), m.coatTbar.end());
        }
        d.readsCavity = m.readsCavity ? 1 : 0;
        d.patOwnSide = m.patOwnSide ? 1 : 0;
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
        d.emitsBack = m.emitsBack ? 1 : 0;
        d.matReflectsNothing = m.reflectsNothing ? 1 : 0;
        d.hideCamera = m.hideCamera ? 1 : 0;
        d.route = m.route;
        bakeSpec(m.emit, d.matEmit, up);
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
                    bakeSpec(ch.stops[rb->selStop].color, d.reflect, up);   // constant stop → bake
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
        bakeSpec(m.hairSpecular, d.hairSpecular, up);
        d.hairSpecPat  = m.hairSpecPat;
        d.hairOpacity  = m.hairOpacity;
        d.hairOpacityPat = m.hairOpacityPat;
        d.hairBetaM    = m.hairBetaM;
        d.hairBetaN    = m.hairBetaN;
        d.hairAlpha    = m.hairAlpha;
        d.hairKappa    = m.hairKappa;
        d.hairMedullaG = m.hairMedullaG;
        bakeSpec(m.hairSigmaA, d.hairSigmaA, up);
        bakeSpec(m.hairMedullaSigmaS, d.hairMedullaSigmaS, up);
        bakeSpec(m.hairMedullaSigmaA, d.hairMedullaSigmaA, up);
        d.hairSigmaAFromReflect = m.hairSigmaAFromReflect ? 1 : 0;
    }

    // A LAYERED material's coat is a set of FIELDS on the parent, not a child material, so the
    // device would have nothing to shade with once the coat wins its Fresnel draw. Give it
    // something: one synthetic GLOSSY lobe per layered material, appended past the scene's own
    // materials. It inherits the parent's spectral and texture plumbing (so the coat's roughness
    // map, normal map and pattern bindings come along), is made WHITE because a dielectric coat's
    // reflection carries no tint of its own, and drops the body lobes and any emission. With this
    // the coat is an ordinary material and dResolveCompound can hand it to any dispatch site.
    for (size_t i = 0; i < scene.mats.size(); ++i) {
        if (scene.mats[i].type != MatType::Layered) continue;
        DMaterial coat = mats[i];
        coat.type = D_GLOSSY;
        for (int s2 = 0; s2 < SPEC_N; ++s2) coat.reflect.tab[s2] = 1.0;
        coat.reflect.kn = 0;   // the copy carried the base's knots
        coat.rgbAlbedo = {1.0, 1.0, 1.0};
        coat.reflectTex = -1;
        coat.reflectPat = -1;
        coat.mixCount = 0;
        coat.coatChild = -1;
        coat.coatLobe = 1;
        coat.coatStack = -1;
        coat.coatTbarOff = -1;
        coat.coatTbarNT = 1;
        coat.coatParent = (mats[i].coatModel == 2) ? -1 : (int)i;   // a manual coat's factor is 1
        coat.matIsLight = 0;
        coat.matReflectsNothing = 0;
        coat.emitPat = -1;
        mats[i].coatChild = (int)mats.size();
        mats.push_back(coat);
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
    int*        d_instEmit = instEmit.empty() ? nullptr : (int*)keep(uploadVec(instEmit));
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
        de.nGeom   = {e.nGeom.x, e.nGeom.y, e.nGeom.z};
        de.normalTilted = e.normalTilted ? 1 : 0;
        de.aimNorm = e.aimNorm;
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
                dtris[i].reach = s.reach;
                // Source-triangle UVs, so a sampled point reports the same (u,v) the
                // ray-hit path interpolates (an emission pattern reads both sides).
                dtris[i].uv0  = {(Real)s.uv0.x,  (Real)s.uv0.y,  (Real)s.uv0.z};
                dtris[i].uvE1 = {(Real)s.uvE1.x, (Real)s.uvE1.y, (Real)s.uvE1.z};
                dtris[i].uvE2 = {(Real)s.uvE2.x, (Real)s.uvE2.y, (Real)s.uvE2.z};
                dtris[i].tri = s.tri;
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
        de.sunDiscR = e.sunDiscR;
        de.viewXYZ = {e.viewXYZ.x, e.viewXYZ.y, e.viewXYZ.z};
        de.cdfOffset = (int)cdfAll.size();
        de.cdfN = (int)e.spd.cdf.size();
        de.cdfStep = e.spd.step;
        de.matId = e.matId;                 // BDPT: link to emissive surface material
        de.emitPat = e.emitPat;             // `emit pattern:` profile over this emitter
        bakeSpec(e.spdFn, de.emitSpd, up);       // BDPT: baked emission SPD for Le(lambda)
        { Vec3 le = rgbbake::emitToRgb(e.spdFn); de.rgbEmit = {le.x, le.y, le.z}; }  // fast RGB backward
        de.cieMean = {e.cieMean.x, e.cieMean.y, e.cieMean.z};   // achromatic-path beam fold
        de.motion = e.motion;
        de.deforms = e.deforms ? 1 : 0;
        de.selW = e.selW;
        de.glowPrim = e.glowPrim;
        de.glowSphere = e.glowSphere ? 1 : 0;
        de.glowLo = {(Real)e.glowLo.x, (Real)e.glowLo.y, (Real)e.glowLo.z};
        de.glowHi = {(Real)e.glowHi.x, (Real)e.glowHi.y, (Real)e.glowHi.z};
        cdfAll.insert(cdfAll.end(), e.spd.cdf.begin(), e.spd.cdf.end());
        dems.push_back(de);
    }
    std::vector<double> emitCdf = scene.emitterCdf;
    std::vector<double> emitSampCdf = scene.emitSampler.cdf;   // BDPT shared lambda CDF
    DEmitter* d_ems     = dems.empty()       ? nullptr : (DEmitter*)keep(uploadVec(dems));
    double*   d_cdfAll  = cdfAll.empty()     ? nullptr : (double*)keep(uploadVec(cdfAll));
    double*   d_emitCdf = emitCdf.empty()    ? nullptr : (double*)keep(uploadVec(emitCdf));
    double*   d_emitSamp = emitSampCdf.empty() ? nullptr : (double*)keep(uploadVec(emitSampCdf));
    std::vector<double> camCdf = scene.camSampler.cdf;         // the camera's illuminant
    std::vector<double> camVisCdf = scene.camVisSampler.cdf;   // mode R's camera lambda CDF
    std::vector<double> camWCdf = scene.camWSampler.cdf;       // mode W's
    double*   d_cam = camCdf.empty() ? nullptr : (double*)keep(uploadVec(camCdf));
    double*   d_camVis = camVisCdf.empty() ? nullptr : (double*)keep(uploadVec(camVisCdf));
    double*   d_camW = camWCdf.empty() ? nullptr : (double*)keep(uploadVec(camWCdf));

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
    // Per-vertex colour: upload the table, and make sure the JH LUT is resident even
    // when no stochastic texture asked for it (that was the only previous consumer).
    if (!scene.vertColors.empty()) {
        sc.vertColors  = (const float*)keep(uploadVec(scene.vertColors));
        sc.nVertColors = (int)scene.vertColors.size();
        if (!d_jhLut) d_jhLut = (const float*)keep(uploadVec(upsample::coeffLut()));
        sc.jhLut = d_jhLut;
    } else {
        sc.vertColors = nullptr; sc.nVertColors = 0; sc.jhLut = d_jhLut;
    }
    sc.ndSlice = scene.ndSlice;   // by value: 40 doubles, read by every field sample
    sc.tris = d_tris; sc.nTris = (int)tris.size();
    sc.sph = d_sph;   sc.nSph = (int)sph.size();
    sc.sphAxes = sphAxes.empty() ? nullptr : (const DVec3*)keep(uploadVec(sphAxes));
    if (!scene.sphereEmitter.empty()) {   // one per sphere: closestHit indexes it by sphere
        std::vector<int> se(scene.sphereEmitter);
        se.resize(scene.spheres.size(), -1);
        sc.sphEmitter = (const int*)keep(uploadVec(se));
    } else {
        sc.sphEmitter = nullptr;
    }
    sc.mirrorPlanes = d_mirp; sc.nMirrorPlanes = (int)mirrorPlanes.size();
    sc.dielSph   = d_dielSph;   sc.nDielSph   = (int)scene.dielSphereIdx.size();
    sc.mirrorSph = d_mirrorSph; sc.nMirrorSph = (int)scene.mirrorSphereIdx.size();
    sc.mats = d_mats;
    sc.coatTbar = coatTbarAll.empty() ? nullptr : (const double*)keep(uploadVec(coatTbarAll));
    sc.camHiddenAny = scene.camHiddenAny ? 1 : 0;   // see DScene::camHiddenAny
    sc.nodes = d_nodes; sc.primIdx = d_prim; sc.nNodes = (int)nodes.size();
    sc.fieldNodes = d_fnodes; sc.fieldExprNodes = d_fexpr;
    sc.fieldNodesF = d_fnodesF; sc.fieldExprNodesF = d_fexprF;
    sc.implicits = d_impl; sc.nImplicits = (int)dimpl.size();
    sc.curveSegs = d_curves; sc.nCurveSegs = (int)curveSegs.size();
    sc.instances = d_inst; sc.nInstances = (int)dinst.size();
    sc.instEmit = d_instEmit;
    sc.blas = d_blas; sc.blasNodes = d_blasN; sc.blasPrim = d_blasP; sc.blasTris = d_blasT;
    uploadMotion(scene, up);
    sc.patNodes = d_pnodes; sc.patterns = d_pat; sc.nPatterns = (int)dpat.size();
    sc.recCoeff = d_recCoeff; sc.recDrivers = d_recDrv; sc.recScalarStops = d_recStops;
    sc.emitters = d_ems; sc.nEmitters = (int)dems.size();
    {   // JSPEC: each emitter's fold quadrature, {lam, X, Y, Z} x DJSPEC_FOLD (dJSpecFoldColor)
        static_assert(kFoldBins == DJSPEC_FOLD, "DJSPEC_FOLD must track scene.h kFoldBins");
        std::vector<float> fold((size_t)scene.emitters.size() * DJSPEC_FOLD * 4, 0.0f);
        for (size_t e = 0; e < scene.emitters.size(); ++e) {
            const Emitter& em = scene.emitters[e];
            for (int k = 0; k < em.foldN && k < DJSPEC_FOLD; ++k) {
                float* f = fold.data() + (e * DJSPEC_FOLD + (size_t)k) * 4;
                f[0] = (float)em.foldLam[k];
                f[1] = (float)em.foldCie[k].x; f[2] = (float)em.foldCie[k].y; f[3] = (float)em.foldCie[k].z;
            }
        }
        sc.emFold = fold.empty() ? nullptr : (const float*)keep(uploadVec(fold));
        std::vector<float> spdInv(scene.emitters.size(), 0.0f);
        for (size_t e = 0; e < scene.emitters.size(); ++e)
            spdInv[e] = scene.emitters[e].spdDen > 0.0 ? (float)(1.0 / scene.emitters[e].spdDen) : 0.0f;
        sc.emSpdInv = spdInv.empty() ? nullptr : (const float*)keep(uploadVec(spdInv));
    }
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
    sc.lightTreeParent = scene.lightTreeParent.empty() ? nullptr
                                                       : (const int*)keep(uploadVec(scene.lightTreeParent));
    sc.lightTreeLeaf   = scene.lightTreeLeaf.empty() ? nullptr
                                                     : (const int*)keep(uploadVec(scene.lightTreeLeaf));
    sc.nLightTreeLeaf  = (int)scene.lightTreeLeaf.size();
    sc.bkLightTree     = lt::gEnabled ? 1 : 0;
    sc.bkGlossyNee     = lt::gGlossyNee ? 1 : 0;   // GLOSSY-NEE (known-issues.md)
    {   const char* e = std::getenv("FTRACE_FIRE_NEE");          // host twin: fireNeeOn()
        sc.bkFireNee = (e && e[0] == '0') ? 0 : 1; }
    sc.specNee         = lt::gSpecNee ? 1 : 0;     // SPECTRAL NEE; the host gather reads the same object
    {   // M-GATHERAREA footprint budget, same environment channel and same default (8) the host
        // uses -- the two MUST agree or -device gpu and -device cpu diverge on truncated geometry.
        const char* e = std::getenv("FTRACE_GATHERAREA");
        sc.gatherArea = e ? std::max(0, std::atoi(e)) : 8;   // below 0: none, as 0 (host twin)
        // The tangle gate rides the same channel for the same reason: read it anywhere else and
        // the two backends can disagree about whether it is on.
        const char* g = std::getenv("FTRACE_GAREJECT");
        sc.gatherRejPct = g ? std::atoi(g) : 30;   // default must match the host's gaRejectPct()
        const char* gf = std::getenv("FTRACE_GAFIBER");
        sc.gaFiberSkip = (gf && *gf == '0') ? 0 : 1;   // default must match host gaFiberSkipOn()
        const char* gb = std::getenv("FTRACE_GABIAS");
        sc.gaBias = (gb && *gb == '0') ? 0 : 1;        // default must match host gaBiasOn()
        const char* gg = std::getenv("FTRACE_GAGATE");
        sc.gaGate = gg ? std::atoi(gg) : -1;            // default must match gaGateProbes()
        const char* gbl = std::getenv("FTRACE_GABALL");
        sc.gaBall = (gbl && *gbl == '0') ? 0 : 1;      // default must match gaBallOn()
    }
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
    sc.glowG = scene.glowGTab.empty() ? nullptr : (const double*)keep(uploadVec(scene.glowGTab));
    sc.glowGN = (int)scene.glowGTab.size();
    sc.camCdf = d_cam;
    sc.camVisCdf = d_camVis;
    sc.camWCdf = d_camW;
    sc.camG = scene.camG;
    // dSampleCamLambdaU reads a camera draw's CDF and the camera illuminant's at the one bin it
    // drew, on the emission sampler's bins: they must share them (finalizeCameraSpectrum builds
    // them so). If one ever does not, say so and draw from the illuminant itself, whose film
    // weight is then exactly 1.
    auto sameBins = [&](const EmissionSampler& s) {
        return s.cdf.size() == emitSampCdf.size() && s.step == scene.emitSampler.step;
    };
    bool camFires = true;              // the fires' term in the camera's illuminant (dFireG)
    if (!sameBins(scene.camSampler) || !sameBins(scene.camVisSampler) ||
        !sameBins(scene.camWSampler)) {
        std::fprintf(stderr,
            "[cuda] INTERNAL: a camera wavelength sampler and the emission sampler differ in "
            "bins; drawing camera wavelengths from the camera illuminant\n");
        if (sameBins(scene.camSampler)) {
            sc.camVisCdf = sc.camCdf;
            sc.camWCdf = sc.camCdf;
        } else {   // the emission sampler, and weights to match: a fire is then reached only
                   // at the emitters' wavelengths
            sc.camCdf = sc.camVisCdf = sc.camWCdf = sc.emitSamplerCdf;
            sc.camG = scene.emitG;
            camFires = false;
        }
    }
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
            dm.route = m.route;
            dm.g = m.g;
            bakeSpec(m.sigma_a, dm.sigma_a, up);
            bakeSpec(m.sigma_s, dm.sigma_s, up);
            dm.heterogeneous = m.heterogeneous() ? 1 : 0;
            dm.density  = m.density.empty() ? nullptr : (const PatNode*)keep(uploadVec(m.density));
            dm.densityN = (int)m.density.size();
            dm.densityMax = m.densityMax;
            // Per-cell control/residual majorant grid (majorant.h). Two dense float arrays
            // — a 128^3 grid is 16 MB, and unlike the density field it is read once per
            // tracking cell rather than per sample, so a sparse brick layout would buy
            // little. Absent (unbounded density field) => ctrl null, global majorant.
            if (m.majorant && m.majorant->valid()) {
                const MajorantGrid& mg = *m.majorant;
                dm.majorant.ctrl = (const float*)keep(uploadVec(mg.ctrl));
                dm.majorant.res  = (const float*)keep(uploadVec(mg.res));
                dm.majorant.wmin    = {mg.wmin.x, mg.wmin.y, mg.wmin.z};
                dm.majorant.cell    = {mg.cell.x, mg.cell.y, mg.cell.z};
                dm.majorant.invCell = {mg.invCell.x, mg.invCell.y, mg.invCell.z};
                dm.majorant.nx = mg.nx; dm.majorant.ny = mg.ny; dm.majorant.nz = mg.nz;
            } else {
                dm.majorant.ctrl = nullptr; dm.majorant.res = nullptr;
                dm.majorant.wmin = {0,0,0}; dm.majorant.cell = {0,0,0};
                dm.majorant.invCell = {0,0,0};
                dm.majorant.nx = dm.majorant.ny = dm.majorant.nz = 0;
            }
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
            dm.boundRamp = m.boundRamp ? 1 : 0;
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
            dm.bends   = m.bends() ? 1 : 0;
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
            dm.achro = mediumAchromatic(m) ? 1 : 0;   // achromatic-path beam fold (scene.h)
        }
        sc.media  = dmeds.empty() ? nullptr : (const DMedium*)keep(uploadVec(dmeds));
        sc.mediaN = (int)dmeds.size();
        // UPBP-W-CHROMATIC's table (DChromaRun), read where c_jTrSpec says some medium's
        // extinction varies with wavelength -- the host's test for ChromaMis -- and c_jChroma;
        // only then is it uploaded (cudaSetJSwitches runs before any render's upload).
        sc.chromaSig = nullptr;
        if (!scene.media.empty() && g_jTrSpecHost && g_jChromaHost) {
            std::vector<double> sig(scene.media.size() * (size_t)DJSPEC_K);
            for (size_t m = 0; m < scene.media.size(); ++m)
                for (int b = 0; b < DJSPEC_K; ++b)
                    sig[m * DJSPEC_K + b] = scene.media[m].sigmaT(bandBinLambda(b));
            sc.chromaSig = (const double*)keep(uploadVec(sig));
        }
        // GATHER-TIME SPECTRAL FOLD tables (FOLD-GPU (2), host: Scene::bowLuts). One
        // kBowBins-entry table per eligible (emitter, medium) pair, flattened into one buffer
        // with an offset table; -1 marks a pair the host has no table for, which is the same
        // question `Scene::bowLut()` answers with a null pointer. A scene with no rainbow
        // medium uploads nothing and leaves `sc.bow.tab` null.
        sc.bow = gpu::DBowTab{nullptr, nullptr, 0, 0};
        {
            const int nEm = (int)scene.emitters.size(), nMed = (int)scene.media.size();
            if (nEm > 0 && nMed > 0 && !scene.bowLuts.empty()) {
                std::vector<int> off((size_t)nEm * nMed, -1);
                std::vector<float4> tab;
                for (int e = 0; e < nEm; ++e)
                    for (int m2 = 0; m2 < nMed; ++m2) {
                        const Scene::BowLut* L = scene.bowLut(e, m2);
                        if (!L || (int)L->cie.size() != (int)Scene::BowLut::kBins) continue;
                        off[(size_t)e * nMed + m2] = (int)tab.size();
                        for (int i = 0; i < (int)Scene::BowLut::kBins; ++i)
                            tab.push_back(make_float4((float)L->cie[i].x, (float)L->cie[i].y,
                                                      (float)L->cie[i].z, L->phaseLum[i]));
                    }
                const char* noBow = std::getenv("FTRACE_NOBOWGPU");
                if (noBow && std::atoi(noBow) != 0) tab.clear();   // A/B control: demote instead
                if (!tab.empty()) {
                    sc.bow.tab  = (const float4*)keep(uploadVec(tab));
                    sc.bow.off  = (const int*)keep(uploadVec(off));
                    sc.bow.nEm  = nEm;
                    sc.bow.nMed = nMed;
                    std::fprintf(stderr, "[gpu] gather-time spectral fold: %zu bow tables uploaded "
                                         "(%.1f MB) — a rainbow beam keeps its folded colour instead "
                                         "of being demoted to CIE(lambda)\n",
                                 tab.size() / (size_t)Scene::BowLut::kBins,
                                 tab.size() * sizeof(float4) / 1048576.0);
                }
            }
        }
        sc.hasGrin = grin::sceneHasGrin(scene) ? 1 : 0;   // gate for dGrinMarch (host twin)
        sc.bendsAny = scene.bendsAny ? 1 : 0;
        // Emissive "fire" volumes (ROADMAP C3): upload the AABB/meanKe/power + the per-
        // volume Planck-at-emitKelvin wavelength CDF, and the total emission power for the
        // emitter-vs-fire birth split. Empty => emissiveVolumes=null, totalEmissionPower=0.
        std::vector<DEmissiveVolume> devs(scene.emissiveVolumes.size());
        for (size_t v = 0; v < scene.emissiveVolumes.size(); ++v) {
            const Scene::EmissiveVolume& ev = scene.emissiveVolumes[v];
            DEmissiveVolume& d = devs[v];
            d.mediumIndex = ev.mediumIndex;
            for (int a = 0; a < 3; ++a) { d.bmin[a] = (&ev.bmin.x)[a]; d.bmax[a] = (&ev.bmax.x)[a]; }
            d.meanKe = ev.meanKe;
            d.power  = ev.power;
            d.lamCdf  = (const double*)keep(uploadVec(ev.lamSampler.cdf));
            d.lamN    = (int)ev.lamSampler.cdf.size() - 1;   // cdf has N+1 entries
            d.lamStep = ev.lamSampler.step;
            // Mode R's light sampling (dFireNee). buildFireCells leaves at least one cell.
            d.share   = scene.fireShare((int)v);
            d.voxel   = ev.voxel;
            for (int a = 0; a < 3; ++a) d.gres[a] = ev.gres[a];
            d.cellW   = ev.cellW.empty() ? nullptr : (const double*)keep(uploadVec(ev.cellW));
            d.cellCdf = ev.cellCdf.empty() ? nullptr : (const double*)keep(uploadVec(ev.cellCdf));
            d.nCells  = (int)ev.cellCdf.size();
            for (int a = 0; a < 3; ++a) { d.cellSize[a] = ev.cellSize[a]; d.invCell[a] = ev.invCell[a]; }
            d.pointNorm = ev.pointNorm;
            d.kelvin  = scene.media[(size_t)ev.mediumIndex].emitKelvin;
            d.gW      = (camFires && ev.lamSampler.integral > 0.0)
                            ? ev.power / ev.lamSampler.integral : 0.0;
        }
        sc.emissiveVolumes    = devs.empty() ? nullptr : (const DEmissiveVolume*)keep(uploadVec(devs));
        sc.emissiveVolN       = (int)devs.size();
        sc.totalEmissionPower = scene.totalEmissionPower;
    }
    {   // BIRTH-EMPTY-STACK: the lights' starting stacks (Scene::buildEnclosures)
        std::vector<int> pool, emOff, emN, fireOff, fireN;
        auto add = [&](const MediumStack& st, bool uniform, std::vector<int>& off, std::vector<int>& cnt) {
            off.push_back((int)(pool.size() / 2));
            if (!uniform) { cnt.push_back(-1); return; }
            // The innermost, which the walk is in, where the device's stack holds fewer.
            const int n = st.n < gpu::DMediumStack::CAP ? st.n : gpu::DMediumStack::CAP;
            for (int i = st.n - n; i < st.n; ++i) { pool.push_back(st.matIdx[i]); pool.push_back(st.pri[i]); }
            cnt.push_back(n);
        };
        for (size_t i = 0; i < scene.emitterEnclosure.size(); ++i)
            add(scene.emitterEnclosure[i], scene.emitterEnclosureUniform[i] != 0, emOff, emN);
        for (size_t i = 0; i < scene.fireEnclosure.size(); ++i)
            add(scene.fireEnclosure[i], scene.fireEnclosureUniform[i] != 0, fireOff, fireN);
        sc.encAny     = scene.anyEnterable ? 1 : 0;
        sc.encEmOff   = emOff.empty()   ? nullptr : (const int*)keep(uploadVec(emOff));
        sc.encEmN     = emN.empty()     ? nullptr : (const int*)keep(uploadVec(emN));
        sc.encFireOff = fireOff.empty() ? nullptr : (const int*)keep(uploadVec(fireOff));
        sc.encFireN   = fireN.empty()   ? nullptr : (const int*)keep(uploadVec(fireN));
        sc.encPool    = pool.empty()    ? nullptr : (const int*)keep(uploadVec(pool));
    }
    sc.sensorOrigin = {scene.sensor.origin.x, scene.sensor.origin.y, scene.sensor.origin.z};
    sc.sensorUAxis  = {scene.sensor.uAxis.x,  scene.sensor.uAxis.y,  scene.sensor.uAxis.z};
    sc.sensorVAxis  = {scene.sensor.vAxis.x,  scene.sensor.vAxis.y,  scene.sensor.vAxis.z};
    sc.sceneCenter = {scene.sceneCenter.x, scene.sceneCenter.y, scene.sceneCenter.z};
    sc.sceneRadius = scene.sceneRadius;
    sc.lenScale = scene.lenScale;
    sc.jBeamDomR2  = scene.jBeamDomainR2;
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

// A baked camera for a pass or an epoch that takes one time, already posed there: it keeps no
// time of its own (no shutter, no keys), so no kernel can pose it again or move the scene to a
// sample's time (dSetCameraTime, dPoseCameraAtSample).
static void pinToPass(gpu::DCamera& dc) { dc.blur = 0; dc.keys = nullptr; dc.nKeys = 0; dc.lensKeyed = 0; }
// The baked camera `dc` posed at `tFrames` frames on the host (Camera::poseAt) and pinned there,
// for a pass or an epoch that takes one time.
static gpu::DCamera posedBaked(const gpu::DCamera& dc, const Camera& cam, double tFrames) {
    gpu::DCamera out = dc;
    pinToPass(out);
    if (!cam.moves()) return out;
    Camera c = cam;
    cam.poseAt(tFrames, c);
    out.eye = {c.eye.x, c.eye.y, c.eye.z};
    out.u = {c.u.x, c.u.y, c.u.z};
    out.v = {c.v.x, c.v.y, c.v.z};
    out.w = {c.w.x, c.w.y, c.w.z};
    out.tanHalfX = c.tanHalfX; out.tanHalfY = c.tanHalfY;
    out.halfFovY = c.halfFovY; out.rEdge = c.rEdge;
    out.apertureR = c.apertureR; out.filmDist = c.filmDist; out.lensF = c.lensF;
    return out;
}
// A pass's time at `tFrames` for the device's queries (DMotion::passU): in the interval, or -1
// when the scene holds still.
static float devicePassU(const Scene& scene, double tFrames) {
    return scene.hasMotion() ? (float)scene.motionU(tFrames) : -1.0f;
}
// The host's pass time (g_passMotionU, under a PassTimeScope) for the device's queries: an epoch
// of mode J or mode M over a moving scene, its camera posed there by the host. -1 when none is set
// or the scene holds still.
static float hostPassU(const Scene& scene) {
    return (g_passMotionU >= 0.0 && scene.hasMotion()) ? (float)g_passMotionU : -1.0f;
}

// Bake one Camera into a POD DCamera for the given film resolution. Any device memory
// (the realistic-lens index tables) is recorded in up.frees. Split out of the scene
// bake so a multi-camera shared pass can bake N cameras against one uploaded scene.
// `pinned`: for a pass or an epoch that takes one time, which the host has posed the camera at --
// baked with no time of its own (pinToPass), its keys not uploaded.
static gpu::DCamera bakeCamera(const Scene& scene, const Camera& cam, int resX, int resY, DUpload& up,
                               bool pinned = false) {
    using namespace gpu;
    DCamera dc{};
    // Motion blur: the shutter, when the camera sees motion (Camera::seesMotion: the scene moves,
    // or the camera does), and a moving camera's keys, so a path is posed at its own time.
    dc.blur = cam.seesMotion(scene) ? 1 : 0;
    dc.shOpen = cam.shutterOpen; dc.shClose = cam.shutterClose;
    dc.timeSalt = 0x7F4A7C159E3779B9ULL ^ g_rngSalt;   // BackwardRenderer::renderRows' phase stream
    dc.keys = nullptr; dc.nKeys = 0; dc.keyA = dc.keyB = 0.0;
    dc.eyeShiftU = cam.eyeShiftU;
    dc.lensKeyed = dc.lensPhysical = 0; dc.lensFilmHmm = 24.0;
    if (dc.blur && cam.moves() && !pinned) {
        std::vector<DCamKey> keys;
        for (const CameraPose& p : cam.motion->keys)
            keys.push_back(DCamKey{{(Real)p.eye.x, (Real)p.eye.y, (Real)p.eye.z},
                                   {(Real)p.look.x, (Real)p.look.y, (Real)p.look.z},
                                   {(Real)p.up.x, (Real)p.up.y, (Real)p.up.z}, p.fovY,
                                   p.lens.fstop, p.lens.aperture, p.lens.focus});
        dc.keys = (const DCamKey*)up.keep(uploadVec(keys));
        dc.nKeys = (int)keys.size();
        dc.keyA = cam.motion->a; dc.keyB = cam.motion->b;
        const CameraPose& k0 = cam.motion->keys[0];
        if (!cam.motion->lensKeysAgree())   // the flags below are the first key's for every key
            throw std::logic_error("bakeCamera: a camera's motion keys do not share a lens");
        if (k0.lensKeyed) {
            dc.lensKeyed = 1;
            dc.lensPhysical = k0.lens.physical ? 1 : 0;
            dc.lensFilmHmm = k0.lens.filmHmm;
        }
    }
    dc.eye = {cam.eye.x, cam.eye.y, cam.eye.z};
    dc.enclosure.clear();   // as many of its solids as the device's stack holds, the innermost
    for (int i = cam.enclosure.n > DMediumStack::CAP ? cam.enclosure.n - DMediumStack::CAP : 0; i < cam.enclosure.n; ++i)
        dc.enclosure.push(cam.enclosure.matIdx[i], cam.enclosure.pri[i]);
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
    // side. Used by the backward tracer (GPU mode R), by BDPT (camera subpath and light
    // image), and by the forward kernels' lens splat and catch (modes A, B and C).
    dc.hasLens = 0;
    dc.lens.surf = nullptr;
    dc.lens.iorAll = nullptr;
    dc.lens.pupilCells = nullptr;
    dc.lens.pupilNT = dc.lens.pupilNU = dc.lens.pupilN = 0;
    dc.lens.pupilZRef = dc.lens.pupilZStart = dc.lens.pupilDMin = dc.lens.pupilDTheta = 0.0;
    dc.lens.exitCells = nullptr;
    dc.lens.exitN = 0;
    dc.lens.exitZ = dc.lens.exitDRho = 0.0;
    if (cam.hasLens()) {
        const LensSystem& L = *cam.lens;
        const int M = (int)L.surf.size();
        dc.hasLens = 1;
        dc.lens.nSurf = M;
        dc.lens.filmW_mm = L.filmW_mm; dc.lens.filmH_mm = L.filmH_mm;
        dc.lens.T = L.T; dc.lens.filmZ = L.filmZ;
        std::vector<DLensSurface> surf((size_t)M);
        std::vector<double> iorAll((size_t)M * SPEC_N);
        for (int j = 0; j < M; ++j) {
            surf[j].radius    = L.surf[j].radius;
            surf[j].thickness = L.surf[j].thickness;
            surf[j].aperture  = L.surf[j].aperture;
            surf[j].zpos      = L.zpos[j];
            surf[j].isStop    = L.surf[j].isStop ? 1 : 0;
            for (int i = 0; i < SPEC_N; ++i) {
                double w = DLMIN + (double)i / (SPEC_N - 1) * (DLMAX - DLMIN);
                iorAll[(size_t)j * SPEC_N + i] = L.surf[j].ior ? L.surf[j].ior(w) : 1.0;
            }
        }
        dc.lens.surf = (const DLensSurface*)up.keep(uploadVec(surf));
        dc.lens.iorAll = (const double*)up.keep(uploadVec(iorAll));
        // The pupil table, if the camera has one that fits the raster this bake images onto
        // (the host's Camera::pupil test, against resX x resY).
        const LensPupil* P = cam.lensPupil.get();
        const double halfW = 0.5 * L.filmW_mm, halfH = halfW * ((double)resY / (double)resX);
        auto flatten = [](const std::vector<LensPupilCell>& cs) {
            std::vector<double> out(cs.size() * 4);
            for (size_t i = 0; i < cs.size(); ++i) {
                out[4 * i + 0] = cs[i].cx; out[4 * i + 1] = cs[i].a;
                out[4 * i + 2] = cs[i].b;  out[4 * i + 3] = (double)cs[i].kind;
            }
            return out;
        };
        if (P && P->fits(&L, halfW * halfW + halfH * halfH) && !P->entry.cells.empty()) {
            dc.lens.pupilCells = (const double*)up.keep(uploadVec(flatten(P->entry.cells)));
            dc.lens.pupilNT = P->nTheta; dc.lens.pupilNU = P->nU; dc.lens.pupilN = (int)P->entry.cells.size();
            dc.lens.pupilZRef = P->entry.z; dc.lens.pupilZStart = P->zStart;
            dc.lens.pupilDMin = P->dMin; dc.lens.pupilDTheta = P->dTheta;
            if (!P->exit.cells.empty()) {
                dc.lens.exitCells = (const double*)up.keep(uploadVec(flatten(P->exit.cells)));
                dc.lens.exitN = (int)P->exit.cells.size();
                dc.lens.exitZ = P->exit.z; dc.lens.exitDRho = P->dRho;
            }
        } else if (P) {
            lensPupilRefused();
        }
    }
    if (pinned) pinToPass(dc);
    return dc;
}

// Bake scene + one camera and upload them (the historical single-camera entry point).
// Fills up.dc for callers (renderBdptCuda / renderBackwardCuda) that key off it.
static void buildUpload(const Scene& scene, const Camera& cam, int resX, int resY, DUpload& up) {
    buildUploadScene(scene, up);
    up.dc = bakeCamera(scene, cam, resX, resY, up);
}

// render_cuda.h. `up` holds only the scene half (up.dc is unused) and owns its allocations,
// which a renderBdptCuda call against the hold never frees: the call copies `up.sc` (a POD of
// device pointers) and keeps its own per-call allocations in a DUpload of its own.
struct CudaSceneHold {
    const Scene* scene = nullptr;
    uint64_t generation = 0;   // with the address: a scene loaded where the old one was is new
    uint64_t buildCount = 0;   // and a scene rebuilt in place (Scene::build) is new too
    size_t   bytes = 0;        // the device memory it holds (cudaSceneHoldBytes)
    DUpload up;
};
size_t cudaSceneHoldBytes(const CudaSceneHold* held) { return held ? held->bytes : 0; }
bool cudaSceneHoldFits(const CudaSceneHold* held, const Scene& scene) {
    return held && held->scene == &scene && held->generation == scene.generation &&
           held->buildCount == scene.buildCount;
}
// The scene half of a device call's upload: `held`'s copy (cudaHoldScene's, of this same scene)
// when given, else this call's own. A hold of another scene would render its geometry under this
// one's lights and materials, so `who` (the call) refuses it. The copy is a POD of device
// pointers: the call's own changes to up.sc stay its own, and freeing `up` frees none of the hold.
static void sceneOrHeld(const Scene& scene, const CudaSceneHold* held, DUpload& up, const char* who) {
    if (!held) { buildUploadScene(scene, up); return; }
    if (!cudaSceneHoldFits(held, scene))
        throw std::logic_error(std::string(who) + ": the held device scene is not the scene being rendered");
    up.sc = held->up.sc;
    setSceneScale(scene);
}
static double g_lastBdptOverheadSec = 0.0;
double cudaLastBdptOverheadSec() { return g_lastBdptOverheadSec; }
static long long g_lastBdptWork = 0, g_lastBdptSetupWork = 0;
long long cudaLastBdptWork() { return g_lastBdptWork; }
long long cudaLastBdptSetupWork() { return g_lastBdptSetupWork; }
// The running call's count (renderBdptCuda's workSoFar), set for the call's length.
static std::function<long long()> g_bdptWorkSoFar;
long long cudaBdptCallWork() { return g_bdptWorkSoFar ? g_bdptWorkSoFar() : 0; }
CudaSceneHold* cudaHoldScene(const Scene& scene) {
    if (!cudaAvailable()) return nullptr;
    // FTRACE_NOHOLD=1: hold nothing, so every call uploads the scene itself -- the A/B arm for
    // checking that a held render is bit-identical to an unheld one.
    static const bool off = std::getenv("FTRACE_NOHOLD") && std::atoi(std::getenv("FTRACE_NOHOLD")) != 0;
    if (off) return nullptr;
    CudaSceneHold* h = new CudaSceneHold;
    h->scene = &scene;
    h->generation = scene.generation;
    h->buildCount = scene.buildCount;
    // Measured as the fall in free device memory: the upload's allocations are many and sized
    // in many places, and what the VRAM preflight needs is what they took from the card.
    size_t free0 = 0, free1 = 0, total = 0;
    cudaMemGetInfo(&free0, &total);
    buildUploadScene(scene, h->up);
    cudaMemGetInfo(&free1, &total);
    h->bytes = free0 > free1 ? free0 - free1 : 0;
    return h;
}
void cudaReleaseScene(CudaSceneHold* held) {
    if (!held) return;
    freeUpload(held->up);
    delete held;
}

bool cudaHitProbe(const Scene& scene, const DevHitProbeRay* rays, int n, DevHitProbeOut* out) {
    if (!cudaAvailable() || n <= 0) return false;
    DUpload up;
    buildUploadScene(scene, up);
    DevHitProbeRay* dR = nullptr;
    DevHitProbeOut* dO = nullptr;
    bool ok = cudaMalloc(&dR, sizeof(DevHitProbeRay) * n) == cudaSuccess
           && cudaMalloc(&dO, sizeof(DevHitProbeOut) * n) == cudaSuccess
           && cudaMemcpy(dR, rays, sizeof(DevHitProbeRay) * n, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        gpu::kHitProbe<<<(n + 63) / 64, 64>>>(up.sc, dR, n, dO);
        ok = cudaGetLastError() == cudaSuccess        // a launch that never ran
          && cudaDeviceSynchronize() == cudaSuccess
          && cudaMemcpy(out, dO, sizeof(DevHitProbeOut) * n, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(dR); cudaFree(dO);
    freeUpload(up);
    return ok;
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
    // Motion blur: the shutter of the cameras that blur, which a set shares (the host gates any
    // other mix off the device), so each photon takes one time for all of them.
    cs.baseCams = cs.cams;
    for (const DCamera& c : hcams)
        if (c.blur) { cs.blur = 1; cs.shOpen = c.shOpen; cs.shClose = c.shClose; break; }
    // ...which every camera that blurs must share: a photon takes one time for all of them.
    for (const DCamera& c : hcams)
        if (c.blur && (c.shOpen != cs.shOpen || c.shClose != cs.shClose))
            throw std::logic_error("makeCamSet: a device camera set holds cameras with different shutters");
    for (const DCamera& c : hcams) cs.camMoves = cs.camMoves || (c.blur && c.nKeys > 0);
    return cs;
}

// The wavefront pool's persistent slot count for an N-photon trace.
static int wavefrontSlots(long long N) {
    const int W = (int)((N < (1 << 20)) ? N : (1 << 20));
    return W < 1 ? 1 : W;
}
// The forward megakernel's grid (launchForward's kTrace): ~262k threads.
static constexpr int kFwdBlockSize = 128, kFwdNumBlocks = 2048;
// The hero bundle a forward trace carries for `heroC` asked: none (1) through participating media
// or GRIN bending, where the geometry is not the same for every wavelength between dispersive
// events.
static int forwardHeroC(int heroC, bool media, bool grin) {
    if (heroC <= 1 || media || grin) return 1;
    return heroC > hero::kHeroMax ? hero::kHeroMax : heroC;
}
// Whether a forward trace runs on the wavefront backend: asked for, and nothing the megakernel
// alone carries -- the hero bundle, photon beams (the pool carries no per-photon beam stream), or a
// stable seed (mode P's layers: each photon seeded by its absolute index, which the pool's
// regeneration does not keep).
static bool forwardRunsWavefront(bool wavefront, int effHeroC, bool beams, bool stableSeed) {
    return wavefront && effHeroC == 1 && !beams && !stableSeed;
}
// The wavefront's SoA photon pool, an entry an array: its WFState member, its element type and the
// elements a slot holds. wavefrontTrace allocates and frees the pool by this list and
// wavefrontPoolBytes sizes it by it, so the preflight's count cannot leave an array out.
#define FT_WF_POOL(X)                                                                           \
    X(ro, DVec3, 1) X(rd, DVec3, 1) X(beta, Real, 1) X(lambda, Real, 1) X(rng, DRng, 1)         \
    X(bounce, int, 1) X(alive, int, 1) X(stkMat, int, DMediumStack::CAP)                        \
    X(stkPri, int, DMediumStack::CAP) X(stkN, int, 1) X(hit, DHit, 1)                           \
    X(gmMed, int, 1) X(gmArc, Real, 1)        /* GRIN pre-pass results, extend -> shade */      \
    X(pathC, DPathCaustic, 1)                 /* caustic path state */                          \
    X(curveTmin, Real, 1)                     /* fiber exit, shade -> extend */                 \
    X(timeX, float, 1)                        /* the photon's time, all stages */
// The device memory the wavefront's photon pool takes for an N-photon trace.
static size_t wavefrontPoolBytes(long long N) {
    using namespace gpu;
#define FT_WF_SLOT_BYTES(m, T, n) + sizeof(T) * (size_t)(n)
    const size_t slot = 0 FT_WF_POOL(FT_WF_SLOT_BYTES);
#undef FT_WF_SLOT_BYTES
    return (size_t)wavefrontSlots(N) * slot;
}

// Host driver for the wavefront backend. Allocates the SoA photon pool, seeds it, then
// runs extend/shade passes until every slot has drained the N-photon budget. Writes into
// the same d_film / d_hits / d_energy buffers as the megakernel path.
static void wavefrontTrace(DUpload& up, const gpu::DCamSet& cs, double* d_energy,
                           long long N, int diffraction, unsigned long long kseed,
                           int maxBounce, int camModeInt) {
    using namespace gpu;
    if (N <= 0) return;
    const int W = wavefrontSlots(N);   // persistent slot count

    WFState st;
#define FT_WF_ALLOC(m, T, n) CUDA_CHECK(cudaMalloc(&st.m, (size_t)W * (size_t)(n) * sizeof(T)));
    FT_WF_POOL(FT_WF_ALLOC)
#undef FT_WF_ALLOC
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
        kWfExtend<<<gb, bs>>>(up.sc, cs, st, W);
        kWfShade<<<gb, bs>>>(up.sc, cs, d_energy, st, W, N,
                             diffraction, maxBounce, d_dispatched, d_live, camModeInt, kseed);
        cudaCheckKernel("wavefront-pass");
        int live = 0;
        CUDA_CHECK(cudaMemcpy(&live, d_live, sizeof(int), cudaMemcpyDeviceToHost));
        if (live <= 0) break;
    }

#define FT_WF_FREE(m, T, n) cudaFree(st.m);
    FT_WF_POOL(FT_WF_FREE)
#undef FT_WF_FREE
    cudaFree(d_dispatched); cudaFree(d_live);
}

// Launch the forward trace (megakernel or wavefront backend) over the baked scene `up`
// and camera set `cs`, accumulating into cs.films/cs.hits and d_energy. Shared by the
// single-camera and multi-camera drivers so the launch/seeding logic lives in one place.
// `stableSeed`: each photon's stream from its absolute index (seedBase, the photons before this
// launch, on) under one seed, so a render's photons draw the same streams however they are split
// into launches (mode P's batches, mode M's clock-sized deposit chunks); on the megakernel, the
// one backend that seeds per photon.
// `posedKeep` / `posedCapKeep`: a buffer the caller keeps across launches for a moving camera set's
// posed copies (a session's), grown here as a launch needs, in DCameras; null, a launch allocates
// and frees its own.
// `maxBounce`: the most surface interactions a photon makes -- `-max-bounce` for modes A, B and C
// (main.cpp effMaxBounce, as the CPU's renderForward takes it), photonMaxBounce() (photonmap.h,
// `-photon-bounce`) for mode M's and mode S's deposits.
static void launchForward(DUpload& up, const gpu::DCamSet& cs, double* d_energy,
                          long long N, bool diffraction, unsigned long long seedBase,
                          bool wavefront, int camModeInt, int heroC, int maxBounce,
                          bool stableSeed = false,
                          gpu::DCamera** posedKeep = nullptr, size_t* posedCapKeep = nullptr) {
    using namespace gpu;
    // seedBase==0 keeps the original single-shot seed exactly; each accumulation chunk
    // passes a distinct cumulative-photon offset for an independent stream -- or, seeding
    // stably, its photons' absolute indices start there under the single-shot seed.
    unsigned long long kseed = (0x9e3779b97f4a7c15ULL + (stableSeed ? 0ULL : seedBase) * 0x9e3779b97f4a7c15ULL)
                             ^ g_rngSalt;   // `-seed`; 0 by default, so XOR is a no-op
    // Hero-wavelength sampling shares one BVH walk across C stratified wavelengths, cutting
    // chromatic noise. It lives ONLY in the megakernel, so the megakernel runs when it is on.
    const int effHeroC = forwardHeroC(heroC, up.sc.mediaN != 0, up.sc.hasGrin != 0);
    const bool wf = forwardRunsWavefront(wavefront, effHeroC, cs.beamsOn(), stableSeed);
    // A moving camera under a blur: each thread poses its own copy of the set's cameras at its
    // photon's time (DCamSet::cam), one per thread of the launch -- the wavefront's slots, or
    // the megakernel's grid (cudaForwardPosedCameraBytes counts the same).
    DCamSet csRun = cs;
    csRun.maxBounce = maxBounce;
    DCamera* d_posed = nullptr;
    if (cs.camMoves && cs.nCam > 0) {
        const long long threads = wf ? wavefrontSlots(N) : (long long)kFwdBlockSize * kFwdNumBlocks;
        const size_t need = (size_t)threads * cs.nCam;
        if (posedKeep) {
            if (*posedCapKeep < need) {
                if (*posedKeep) cudaFree(*posedKeep);
                *posedKeep = nullptr; *posedCapKeep = 0;
                CUDA_CHECK(cudaMalloc(posedKeep, need * sizeof(DCamera)));
                *posedCapKeep = need;
            }
            csRun.cams = *posedKeep; csRun.poseOut = *posedKeep;
        } else {
            CUDA_CHECK(cudaMalloc(&d_posed, need * sizeof(DCamera)));
            csRun.cams = d_posed; csRun.poseOut = d_posed;
        }
        csRun.camStride = cs.nCam;
    }
    if (wf) {
        // Streaming backend: identical physics, path-regeneration scheduling. Same
        // maxBounce and camera mode/set as the megakernel. (Hero AND photon-beams
        // force the megakernel — the wavefront pool carries no per-photon beam RNG stream.)
        wavefrontTrace(up, csRun, d_energy, N, diffraction ? 1 : 0, kseed, maxBounce, camModeInt);
    } else {
        // FTRACE_GPU_PHOTONSEED=1: per-photon seeding (GPU-VARIANCE's A/B arm) for the launches that
        // seed per thread -- modes A, B and C's own renders. Stable seeding (mode P's batches, mode
        // M's deposit) seeds per photon whatever it says.
        static const int perPhotonSeed = [] {
            const char* e = std::getenv("FTRACE_GPU_PHOTONSEED");
            return (e && *e) ? std::atoi(e) : 0;
        }();
        kTrace<<<kFwdNumBlocks, kFwdBlockSize>>>(up.sc, csRun, d_energy, N, diffraction ? 1 : 0,
                                           kseed, maxBounce, camModeInt, effHeroC,
                                           stableSeed ? 1 : perPhotonSeed,
                                           stableSeed ? (long long)seedBase : 0LL);
    }
    cudaCheckKernel("forward");
    if (d_posed) cudaFree(d_posed);
}

Film renderForwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                       long long N, EnergyReport& eOut, bool diffraction,
                       char camMode, unsigned long long seedBase, bool wavefront, int heroC,
                       bool pSplit, bool stableSeed, const CudaSceneHold* held, int maxBounce) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    if (!cudaAvailable() || !cudaForwardSupported(scene)) return out;

    DUpload up;
    sceneOrHeld(scene, held, up, "renderForwardCuda");
    up.dc = bakeCamera(scene, cam, resX, resY, up);

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
    cs.pSplit = pSplit ? 1 : 0;
    launchForward(up, cs, d_energy, N, diffraction, seedBase, wavefront, camModeInt, heroC,
                  maxBounce >= 1 ? maxBounce : 32, stableSeed);

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
    gpu::DCamera* d_posed = nullptr;        // a moving camera set's posed copies (launchForward)
    size_t posedCap = 0;                    // ...in DCameras
    gpu::DCamSet cs{};
    int  camModeInt = 0;
    bool wavefront = false;
    int  heroC = 1;
    int  maxBounce = 32;                    // a photon's surface interactions (launchForward)
    int  nc = 0;
};

// Whether a shared forward flight of `nCam` cameras gathers photon beams (`beamGather`, -beams):
// only several cameras sharing one flight through a participating medium (the device gates the
// per-step branch on the same).
static bool sharedFlightGathersBeams(const Scene& scene, bool beamGather, int nCam) {
    return beamGather && nCam > 1 && !scene.media.empty();
}

SharedGpuSession* sharedForwardGpuBegin(const Scene& scene,
                                        const std::vector<Camera>& cams,
                                        const std::vector<int>& resX,
                                        const std::vector<int>& resY,
                                        char camMode, bool wavefront, int heroC,
                                        bool beamGather, int maxBounce,
                                        const std::vector<Film>* seedFilms,
                                        const EnergyReport* seedEnergy) {
    using namespace gpu;
    int nc = (int)cams.size();
    if (nc == 0 || !cudaAvailable() || !cudaForwardSupported(scene)) return nullptr;

    auto* s = new SharedGpuSession;
    s->maxBounce = maxBounce >= 1 ? maxBounce : 32;
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
    s->cs.beamGather = sharedFlightGathersBeams(scene, beamGather, nc);
    s->cs.beamOrderMax = pbeams::gOrderMax;   // -beams-order (host twin: Renderer::beamOrderMax)
    // A moving camera set, under a blur: every thread poses its own copy of every camera at its
    // photon's time, a buffer of threads x cameras that a big flight can make larger than the card
    // has free. Made once for the session, here, where the flight can still fly on the CPU, by the
    // preflight's own rule (vramPreflight, which counted it: cudaSharedFlightPosedBytes): the walk's
    // stacks, reserved at its first launch, after this, set aside, and its headroom kept.
    if (s->cs.camMoves) {
        const size_t needB = cudaSharedFlightPosedBytes(scene, nc, wavefront, heroC, beamGather);
        size_t freeB = 0, totalB = 0;
        cudaMemGetInfo(&freeB, &totalB);
        const size_t pending = pendingLocalBytes(cudaRouteLocalBytes(
            cudaForwardRoute(scene, wavefront, heroC, beamGather, nc)));
        // ...and the wavefront's photon pool, which each batch allocates beside it.
        const size_t keep = pending + kVramHeadroomBytes +
                            cudaForwardWavefrontBytes(scene, wavefront, heroC, 1ll << 40, beamGather, nc);
        const size_t room = (freeB > keep) ? freeB - keep : 0;
        if (needB > room || cudaMalloc(&s->d_posed, needB) != cudaSuccess) {
            cudaGetLastError();
            s->d_posed = nullptr;
            std::fprintf(stderr, "[device] a blurred shared flight poses its %d cameras at each photon's time in "
                                 "%.2f GB of device memory, more than the %.2f GB free beside the walk's "
                                 "stacks, its photon pool and the headroom; using CPU\n",
                         nc, (double)needB / 1e9, (double)room / 1e9);
            sharedForwardGpuEnd(s);
            return nullptr;
        }
        s->posedCap = needB / sizeof(DCamera);
    }
    return s;
}

void sharedForwardGpuBatch(SharedGpuSession* s, long long N,
                           unsigned long long seedBase, bool diffraction) {
    if (!s || N <= 0) return;
    dSetLenScale(s->up.sc);
    launchForward(s->up, s->cs, s->d_energy, N, diffraction, seedBase,
                  s->wavefront, s->camModeInt, s->heroC, s->maxBounce, false, &s->d_posed,
                  &s->posedCap);
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
    if (s->d_posed) cudaFree(s->d_posed);
    delete s;
}

std::vector<Film> renderForwardSharedCuda(const Scene& scene,
                                          const std::vector<Camera>& cams,
                                          const std::vector<int>& resX,
                                          const std::vector<int>& resY,
                                          long long N, EnergyReport& eOut, bool diffraction,
                                          char camMode, unsigned long long seedBase,
                                          bool wavefront, int heroC, bool beamGather,
                                          int maxBounce) {
    // One-shot wrapper over the session API (bit-compatible with the historical
    // single-call behaviour: same bake, same launch, same seeding, energy ADDED to eOut).
    int nc = (int)cams.size();
    std::vector<Film> out(nc);
    for (int c = 0; c < nc; ++c) { out[c].resX = resX[c]; out[c].resY = resY[c]; out[c].alloc(); }
    SharedGpuSession* s = sharedForwardGpuBegin(scene, cams, resX, resY, camMode,
                                               wavefront, heroC, beamGather, maxBounce);
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

static bool cudaBidirCoreSupported(const Scene& scene) {
    // BDPT-GPU needs the same POD-bakeable materials as the forward path, PLUS the
    // BDPT scope restrictions (bdpt.h / mode-D guard in main.cpp): participating media
    // — homogeneous AND heterogeneous (density-field / bounded) — are supported (device
    // random walk places medium vertices by delta tracking and weights connections by
    // ratio-tracking transmittance, matching the CPU BDPT), and only area/sphere/cylinder
    // Lambertian emitters (no spot/env/collimated).
    if (!cudaForwardSupported(scene)) return false;
    // M9: the GPU BDPT kernel now threads the per-hit texcoords (DVertex.u/v -> DHit)
    // through dBsdfF / dBsdfPdf / dRandomWalk / dConnect, so per-hit-driven throughput
    // slots evaluate consistently in the sampler AND the pdf/eval — MIS-safe. Enabled:
    // textured/patterned/record diffuse albedo & glossy reflect, per-hit glossy roughness
    // and thin-film thickness maps, mix blend masks, and Beer-Lambert colored-glass
    // interior absorption (delta vertex — throughput only).
    //
    // FROSTED (rough) dielectric is on-device too (M9): dDielectricStep jitters the chosen
    // reflect/refract lobe by the per-hit roughness (dMatRoughness, texture/pattern/constant),
    // keeping it on the intended side — bit-for-bit the same "stochastic-delta" model as the
    // CPU bdpt.h (a rough dielectric stays a non-connectable delta vertex; only its scattered
    // direction is jittered). The BDPT random walk routes through it, so no separate gate is
    // needed.
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
// Hair runs on the device forward/backward tracers, since 0.364.0 in mode U's kernels (kVcmLightT /
// kVcmCameraT carry the fiber BCSDF, the tube clearance and the strand merge exclusion), and since
// 0.385.0 in the BDPT vertex machinery that modes D and J share -- so hair no longer splits the two
// gates. What still does is a fire: the CPU BDPT reaches it (bdpt.h, the fires note); the
// device's does not.
bool cudaBdptSupported(const Scene& scene) {
    for (const Medium& m : scene.media) if (m.emissive()) return false;
    // Hair joined the BDPT kernels in 0.385.0: dRandomWalk samples the fiber BCSDF and ends a light
    // walk at a strand, dConnectBDPT clears a strand eye vertex's tube, dMisWeight and mode J's
    // camera-side sums drop the light-side alternatives through one, and dSurfMergeSite refuses
    // strands (and every curve). sceneUsesHairMaterial is kept for the other gates that use it.
    return cudaBidirCoreSupported(scene);
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
//
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
    long long end = spp;   // the call's samples, which prog.span may cut

    // ---- the -time deadline (SppProgress::deadline) ----------------------------------
    // Only a launch that can be dropped -- one returning whether it landed whole, whose kernels
    // read the device stop flag (renderBdptCuda) -- can be cut. The watchdog below raises that
    // flag when the deadline passes with a chunk in flight and the session has a sample of its own
    // (`filmSpp`: prog.sessionSpp, then this call's chunks as they land); the launch is dropped
    // and the report after it sees the budget spent. The flag is the one -stop raises, put back
    // to the host's own on the way out (CutRestore), so a cut never reaches the next render.
    constexpr bool droppable = std::is_same_v<decltype(launch((long long)1, (long long)0)), bool>;
    const bool hasDeadline = droppable && prog.deadline != clk::time_point{};
    std::atomic<long long> filmSpp{prog.sessionSpp};
    std::atomic<bool> cut{false};
    struct CutRestore {   // declared before the watchdog, so it runs after the join
        std::atomic<bool>& cut;
        ~CutRestore() {
            if (cut.load()) *(volatile int*)deviceStopFlag().host = ft::stopRequested() ? 1 : 0;
        }
    } cutRestore{cut};
    std::atomic<bool> firstNoted{false};
    auto noteFirst = [&] {   // from whichever thread sees it first
        if (firstNoted.exchange(true)) return;
        std::printf("\n[time] the budget ran out inside this session's first sample, which runs "
                    "on so the session adds one (-stop ends it now, adding none).\n");
        std::fflush(stdout);
    };

    // ---- stall watchdog ------------------------------------------------------------
    // A kernel launch is synchronous from here (cudaCheckKernel synchronizes), so while one
    // chunk is in flight this thread is parked in cuCtxSynchronize and CANNOT report, write
    // the -interval image or honour -time; a -stop reaches the chunk only where its kernels read
    // the device stop flag and its launch drops it (renderBdptCuda: DeviceStopScope). Normally
    // that is fine: chunks are
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
    PollStop watchQuit;
    std::atomic<long long> launchStartMs{0};    // 0 = nothing in flight
    std::atomic<long long> launchSpp{0};
    auto nowMs = [] {
        return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                   clk::now().time_since_epoch()).count();
    };
    std::thread watchdog([&] {
        bool warned = false;
        double nextWarn = 30.0;                 // first complaint after 30 s in one chunk
        bool spent = false;                     // the deadline has been dealt with
        for (;;) {
            // Woken AT the deadline, not up to 2 s after it.
            const auto wake = clk::now() + std::chrono::seconds(2);
            if (watchQuit.waitUntil((hasDeadline && !spent && prog.deadline < wake) ? prog.deadline : wake))
                return;
            // The clock first, then the launch: the main thread publishes a launch before it reads
            // the clock (below), so a deadline seen here with no launch published is one the main
            // thread's own test will see before it launches, and one seen with a launch published
            // is that launch's to cut.
            const bool pastDl = hasDeadline && !spent && clk::now() >= prog.deadline;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const long long t0 = launchStartMs.load(std::memory_order_seq_cst);
            if (pastDl) {
                spent = true;
                if (t0 && filmSpp.load() > 0) {
                    cut.store(true);
                    *(volatile int*)deviceStopFlag().host = 1;
                } else if (t0) {
                    noteFirst();
                }
            }
            if (!t0) { warned = false; nextWarn = 30.0; continue; }   // between chunks: reset
            const double el = (nowMs() - t0) / 1000.0;
            if (el < nextWarn) continue;
            if (!warned) {
                std::fprintf(stderr,
                    "\n[gpu-stall] one %lld-spp chunk has been running on the GPU for %s "
                    "(target is 0.15 s).\n"
                    "[gpu-stall] The render is NOT hung, but until this chunk returns it "
                    "cannot write the -interval image (in modes D and J a -stop drops it, and "
                    "so does -time's end once the session has a sample).\n"
                    "[gpu-stall] The usual cause is another process saturating the GPU: check "
                    "`nvidia-smi` for a second CUDA program. Run with -device cpu to sidestep "
                    "it, or free the card.\n",
                    launchSpp.load(std::memory_order_relaxed), humanDur(el).c_str());
                warned = true;
            } else {
                std::fprintf(stderr, "[gpu-stall] still in the same chunk after %s.\n",
                             humanDur(el).c_str());
            }
            std::fflush(stderr);
            nextWarn = el + 60.0;               // then once a minute, so it stays visible
        }
    });
    struct WatchdogJoin {                        // join on every exit path, including a throw
        PollStop& q; std::thread& t;
        ~WatchdogJoin() { q.stop(); if (t.joinable()) t.join(); }
    } wjoin{watchQuit, watchdog};

    while (done < end) {
        long long c = (done == 0 && prog.span) ? 1 : chunk;   // a span prices one sample
        if (c > end - done) c = end - done;
        auto t0 = clk::now();
        launchSpp.store(c, std::memory_order_relaxed);
        launchStartMs.store(nowMs(), std::memory_order_seq_cst);
        // Published before this test reads the clock (the watchdog reads the clock before it
        // reads the launch), so a deadline that is not this test's is a launch's to cut.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (hasDeadline && clk::now() >= prog.deadline) {
            if (filmSpp.load() > 0) {   // the report sees the budget spent and writes the end
                launchStartMs.store(0, std::memory_order_relaxed);
                prog.report(out, done, false);
                break;
            }
            noteFirst();
        }
        // A launch that returns false was cut short by a stop and dropped (renderBdptCuda): `out`
        // still holds the `done` spp downloaded before it, which is flushed once more as final
        // (skipped at done == 0, which has no image), and the frame is `abandoned` -- the
        // contract cpuSppChunks keeps. A launch returning nothing always lands whole.
        bool whole = true;
        if constexpr (droppable)
            whole = launch(c, prog.sampleBase + done);
        else
            launch(c, prog.sampleBase + done);
        launchStartMs.store(0, std::memory_order_relaxed);
        if (!whole) {
            // The deadline's cut is the budget's end, not an abandoned frame: the film holds what
            // was rendered before it, the report writes it, and the caller's earlier epochs stand
            // in for a `done` of 0.
            if (cut.load() && !ft::stopRequested()) { prog.report(out, done, false); break; }
            if (done > 0) prog.report(out, done, /*final*/true);
            out.abandoned = true;
            break;
        }
        done += c;
        filmSpp.store(prog.sessionSpp + done);
        double dt = std::chrono::duration<double>(clk::now() - t0).count();
        if (chunkDebug)
            std::fprintf(stderr, "[chunk] gpu %lld spp in %.3f s (%.1f spp/s), %lld/%lld done\n",
                         c, dt, dt > 0 ? c / dt : 0.0, done, end);
        if (forcedChunk <= 0 && dt > 1e-4) {               // retarget ~0.15 s per chunk
            long long next = (long long)((double)c * (0.15 / dt));
            if (next < 1)          next = 1;
            if (next > c * 8 + 1)  next = c * 8 + 1;        // ramp up, but not explosively
            chunk = next;
        }
        if (prog.span) end = std::clamp(prog.span(done), done, spp);
        download(out);
        if (prog.report(out, done, done >= end)) break;
    }
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

// Grow-only device buffer of `cap` elements: reallocated only when `need` passes it, to 1.5x the old
// capacity (no less than `need`), so per-pass session buffers stop churning cudaMalloc/cudaFree;
// `need` alone where the device refuses the growth. A refusal of `need` too ends the run, saying
// why -- which need not be a shortage (a context a driver reset left broken).
template <class T>
static void ensureDevCap(T*& p, size_t& cap, size_t need) {
    if (need <= cap) return;
    const size_t grown = std::max(need, cap + cap / 2);
    if (p) { cudaFree(p); p = nullptr; }
    cap = 0;
    cudaError_t why = cudaSuccess;
    for (size_t want : {grown, need}) {
        why = cudaMalloc(&p, want * sizeof(T));
        if (why == cudaSuccess) { cap = want; return; }
        (void)cudaGetLastError();
        p = nullptr;
    }
    std::fprintf(stderr, "[cuda] a device buffer of %zu x %zu B could not be allocated: %s\n",
                 need, sizeof(T), cudaGetErrorString(why));
    std::fflush(stderr);
    std::exit(EXIT_FAILURE);
}

// The device's counting sort of `n` points into lattice `lat` (merge_grid.h; the host's mergeBin):
// each point's bucket in `key`, `order` the point indices sorted by it STABLY -- as the host's
// counting sort is, so a gather meets a cell's points in index order and sums them as the host does
// --, each bucket's start in `cellStart` (mergeBucketCount(lat.slots) + 1 entries) and, where `tag` is
// given (modes U and J), (*tag)[k] the tag of the point sorted to position k (MergeKey). `blocks`: the
// kernels' grid. The buffers grow as their capacities need (ensureDevCap).
template <class Policy, class P>
static void mergeBinDevice(Policy& pol, const P* pts, size_t n, const MergeLattice& lat, int blocks,
                           int*& key, size_t& keyCap, int*& order, size_t& orderCap,
                           int*& cellStart, size_t& cellStartCap, unsigned char** tag, size_t* tagCap) {
    const size_t nBuckets = mergeBucketCount(lat.slots);
    ensureDevCap(key, keyCap, n);
    ensureDevCap(order, orderCap, n);
    ensureDevCap(cellStart, cellStartCap, nBuckets + 1);
    if (tag) ensureDevCap(*tag, *tagCap, n);
    gpu::kMergeKey<<<blocks, 128>>>(pts, (long long)n, lat, key);
    cudaCheckKernel("merge-key");
    thrust::device_ptr<int> tKey(key), tOrd(order);
    thrust::sequence(pol, tOrd, tOrd + n);
    thrust::stable_sort_by_key(pol, tKey, tKey + n, tOrd);
    thrust::lower_bound(pol, tKey, tKey + n, thrust::counting_iterator<int>(0),
                        thrust::counting_iterator<int>((int)(nBuckets + 1)),
                        thrust::device_pointer_cast(cellStart));
    if (tag) {
        gpu::kMergeTag<<<blocks, 128>>>(pts, order, (long long)n, lat, *tag);
        cudaCheckKernel("merge-tag");
    }
}

// ---- a device LBVH over the beam map's sub-beams (FTRACE_JLBVH=1) --------------------
//
// Grow-only scratch, same discipline as JSurfDev: a per-chunk rebuild must not cudaMalloc four
// buffers every 0.15 s. Kept separate from JSurfDev because the two maps have different
// lifetimes -- the surface map is rebuilt per sub-launch, the beam map (for now) per epoch.
struct JBeamDev {
    unsigned* code  = nullptr;   size_t codeCap = 0;
    int* order      = nullptr;   size_t orderCap = 0;
    int* parent     = nullptr;   size_t parentCap = 0;
    int* visited    = nullptr;   size_t visitedCap = 0;
    gpu::DNode* nodes = nullptr; size_t nodesCap = 0;
    ThrustArena arena;
    void release() {
        cudaFree(code); cudaFree(order); cudaFree(parent); cudaFree(visited); cudaFree(nodes);
        code = nullptr; order = nullptr; parent = nullptr; visited = nullptr; nodes = nullptr;
        codeCap = orderCap = parentCap = visitedCap = nodesCap = 0;
        arena.release();
    }
    ~JBeamDev() { release(); }
    JBeamDev() = default;
    JBeamDev(const JBeamDev&) = delete;
    JBeamDev& operator=(const JBeamDev&) = delete;
};

// Build an LBVH over `n` >= 1 sub-beams already resident as `DBeamRec`s, and point `dbm` at it.
// One sub-beam makes a one-node tree, its leaf the root (no internal node to build or refit).
// `lo`/`hi` are the map's world bounds -- taken from the host BVH's root box, which is exactly
// the same set of primitives, so no device reduction is needed for them.
static void buildBeamLbvhDevice(JBeamDev& d, const gpu::DBeamRec* beams, int n,
                                const Aabb& bounds, double radMax, gpu::DBeamMap& dbm) {
    const size_t nNodes = (size_t)(2 * n - 1);
    ensureDevCap(d.code, d.codeCap, (size_t)n);
    ensureDevCap(d.order, d.orderCap, (size_t)n);
    ensureDevCap(d.parent, d.parentCap, nNodes);
    ensureDevCap(d.visited, d.visitedCap, (size_t)(n - 1));
    ensureDevCap(d.nodes, d.nodesCap, nNodes);
    if (n > 1) CUDA_CHECK(cudaMemset(d.visited, 0, (size_t)(n - 1) * sizeof(int)));

    const gpu::DVec3 lo((gpu::Real)bounds.lo.x, (gpu::Real)bounds.lo.y, (gpu::Real)bounds.lo.z);
    auto inv = [](double e) { return (e > 0.0) ? 1.0 / e : 0.0; };
    const gpu::DVec3 invExt((gpu::Real)inv(bounds.hi.x - bounds.lo.x),
                            (gpu::Real)inv(bounds.hi.y - bounds.lo.y),
                            (gpu::Real)inv(bounds.hi.z - bounds.lo.z));
    int blocks = (n + 127) / 128; if (blocks > 2048) blocks = 2048; if (blocks < 1) blocks = 1;
    gpu::kBeamMorton<<<blocks, 128>>>(beams, n, lo, invExt, d.code, d.order);
    cudaCheckKernel("lbvh-morton");

    d.arena.reset();
    ThrustArenaAlloc tal{&d.arena};
    auto pol = FT_THRUST_PAR(tal);
    thrust::device_ptr<unsigned> tk(d.code);
    thrust::device_ptr<int> to(d.order);
    // STABLE, so that equal Morton codes keep sub-beam order. `dLbvhDelta`'s index tie-break
    // then makes the hierarchy deterministic, which is what lets this be A/B'd at all.
    thrust::stable_sort_by_key(pol, tk, tk + n, to);

    gpu::kLbvhInternal<<<blocks, 128>>>(d.code, n, d.nodes, d.parent);
    cudaCheckKernel("lbvh-internal");
    gpu::kLbvhLeaves<<<blocks, 128>>>(beams, d.order, n, (float)radMax, d.nodes);
    cudaCheckKernel("lbvh-leaves");
    gpu::kLbvhRefit<<<blocks, 128>>>(d.parent, n, d.visited, d.nodes);
    cudaCheckKernel("lbvh-refit");

    dbm.nodes = d.nodes;
    dbm.primIdx = d.order;
    dbm.nNodes = (int)nNodes;
}

// ---- mode J's surface photon map, TRACED AND GRIDDED ON THE DEVICE -------------------
//
// The point is FREQUENCY, not speed. Measured ([jstats], known-issues U-vs-J): mode J's whole
// light side is 4 % of a render, so moving it buys 4 % of the wall clock and nothing else --
// but it costs 27 ms per realization on the host, which is 230 s for the ~8500 realizations
// mode U gets in 90 s. On the device the same walk is sub-millisecond, so the realization
// count stops being budget-bound. `FTRACE_JDEVLIGHT=1` selects this path.
//
// Grow-only scratch, exactly as the VCM/SPPM sessions do it: a per-chunk rebuild (step 3) must
// not cudaMalloc/cudaFree seven buffers every 0.15 s.
struct JSurfDev {
    gpu::DSurfPhoton* pts = nullptr;   size_t ptsCap = 0;
    gpu::DSurfMis*    mis = nullptr;   size_t misCap = 0;
    int* count     = nullptr;
    int* cellKey   = nullptr;          size_t cellKeyCap = 0;
    int* order     = nullptr;          size_t orderCap = 0;
    int* cellStart = nullptr;          size_t cellStartCap = 0;
    unsigned char* tag = nullptr;      size_t tagCap = 0;
    ThrustArena arena;
    long long dropped = 0;             // photons the cap refused, cumulative
    long long stored  = 0;             // photons in the CURRENT map
    void release() {
        cudaFree(pts); cudaFree(mis); cudaFree(count);
        cudaFree(cellKey); cudaFree(order); cudaFree(cellStart); cudaFree(tag);
        pts = nullptr; mis = nullptr; count = nullptr;
        cellKey = nullptr; order = nullptr; cellStart = nullptr; tag = nullptr;
        ptsCap = misCap = cellKeyCap = orderCap = cellStartCap = tagCap = 0;
        arena.release();
    }
    // A local of renderBdptCuda, so the destructor is the release: seven device buffers, one of
    // them the photon slab, must not survive the call that made them.
    ~JSurfDev() { release(); }
    JSurfDev() = default;
    JSurfDev(const JSurfDev&) = delete;
    JSurfDev& operator=(const JSurfDev&) = delete;
};

// The device light pass's salt. "JSURFDEL" as a base, XORed with `-seed` and with the
// ABSOLUTE SAMPLE INDEX this call starts at -- which is what makes consecutive light-side
// epochs draw INDEPENDENT realizations rather than the same one over and over. Without the
// sampleBase term a refreshing render would freeze the device map at epoch 0's realization
// while the host map beside it kept redrawing: the two arms would then not be comparable, and
// the device one would silently be the -beamfreeze estimator wearing the refresh's name.
// `-jhostlight` (see render_cuda.h). Set once from argv, read per render.
static bool g_jHostLight = false;
void cudaSetJHostLight(bool on) { g_jHostLight = on; }
// renderBdptCuda's light-pass arm (see the level table there): 0 host, else the device's.
static int jDevLightLevel() {
    int level = g_jHostLight ? 0 : 3;
    if (const char* e = std::getenv("FTRACE_JDEVLIGHT")) level = std::atoi(e);
    return level;
}
bool cudaJSurfOnDevice() { return jDevLightLevel() != 0; }
bool cudaJSurfHostless() { const int l = jDevLightLevel(); return l == 1 || l == 3; }
// Per-object render modes (0.386.0): see gpu::c_routeOn. Set by runRender before a D/J render.
static int g_routeOn = 0, g_routeDefault = 1, g_routeAnyMedJ = 0;
static double g_routeSppD = 1.0, g_routeSppJ = 1.0;
void cudaSetRouting(int on, int routeDefault, int anyMedJ, double sppD, double sppJ) {
    g_routeOn = on; g_routeDefault = routeDefault; g_routeAnyMedJ = anyMedJ;
    g_routeSppD = sppD; g_routeSppJ = sppJ;
}
// JSPEC's dispersive window (see gpu::c_jDispWin). Set once from argv.
static double g_jDispWinHost = 10.0;
static int    g_jDispAllHost = 0;
void cudaSetJDisp(double winNm, int all) { g_jDispWinHost = winNm; g_jDispAllHost = all; }
void cudaSetJSwitches(int jSpec, int trSpec, int deltaMerge, int half, int chroma) {
    g_jSpecHost = jSpec; g_jTrSpecHost = trSpec; g_jDeltaMergeHost = deltaMerge; g_jHalfHost = half;
    g_jChromaHost = chroma;
}
// gpu::c_jBinQ from the sampler the camera walk's wavelength is drawn from: on the device both modes
// draw from the emission sampler (dSampleSceneLambda), where the host's mode J draws from camSampler.
static void uploadJBinQ(const EmissionSampler& s) {
    static_assert(sizeof(s.binMass) == sizeof(double) * DJSPEC_K, "c_jBinQ mirrors EmissionSampler::binMass");
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jBinQ, s.binMass, sizeof(s.binMass)));
}
// The aimed sun (bdpt.h SunAim): host copies, uploaded at the top of every renderBdptCuda.
static std::vector<gpu::Real> g_sunAimWHost;
static std::vector<double> g_sunAimRHost;
static double g_sunAimSumR2Host = 0.0, g_sunAimFHost = 0.0;
void cudaSetSunAim(const double* wxyz, const double* r, int n, double sumR2, double f) {
    if (n > SUNAIM_MAX) {
        // main.cpp's prepareSunAim clusters the targets to at most this many, so host and device
        // aim at the same set; a larger set reaching here is a bug, not a configuration.
        throw std::logic_error("cudaSetSunAim: " + std::to_string(n) + " targets exceed the "
                               "device's " + std::to_string(SUNAIM_MAX));
    }
    g_sunAimWHost.assign(wxyz, wxyz + 3 * (size_t)(n > 0 ? n : 0));
    g_sunAimRHost.assign(r, r + (size_t)(n > 0 ? n : 0));
    g_sunAimSumR2Host = (n > 0) ? sumR2 : 0.0;
    g_sunAimFHost = (n > 0) ? f : 0.0;
}
static void uploadSunAim() {
    const int n = (int)g_sunAimRHost.size();
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_sunAimN, &n, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_sunAimF, &g_sunAimFHost, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_sunAimSumR2, &g_sunAimSumR2Host, sizeof(double)));
    if (n > 0) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_sunAimW, g_sunAimWHost.data(), sizeof(gpu::Real) * 3 * n));
        CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_sunAimR, g_sunAimRHost.data(), sizeof(double) * n));
    }
}

static unsigned long long jSurfSalt(const SppProgress* prog) {
    const unsigned long long base = (unsigned long long)(prog ? prog->sampleBase : 0);
    return 0x4A5355524644454CULL ^ g_rngSalt ^ (base * 0x9E3779B97F4A7C15ULL);
}
// The realization of a block of samples (renderBdptCuda's surfBlock), `first` the block's first
// sample's absolute index: that index alone (and -seed), not where its call started, so a block
// starting at a sample draws the same map in every call, frame and resumed session.
static unsigned long long jSurfBlockSalt(long long first) {
    return 0x4A5355524644454CULL ^ g_rngSalt ^ ((unsigned long long)first * 0xD1B54A32D192ED03ULL);
}

static void buildJSurfMapDevice(JSurfDev& d, const gpu::DScene& sc, const gpu::DCamera& cam,
                                int diffraction, int maxDepth, long long nPaths, double radius,
                                long long maxPhotons, unsigned long long salt,
                                gpu::DSurfMap& dsm) {
    dsm = gpu::DSurfMap{};
    d.stored = 0;
    if (nPaths <= 0 || !(radius > 0.0)) return;
    size_t cap = (maxPhotons > 0) ? (size_t)maxPhotons : (size_t)4000000;
    if (cap > (size_t)2000000000) cap = (size_t)2000000000;   // the slot cursor is an int
    ensureDevCap(d.pts, d.ptsCap, cap);
    ensureDevCap(d.mis, d.misCap, cap);
    if (!d.count) CUDA_CHECK(cudaMalloc(&d.count, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d.count, 0, 2 * sizeof(int)));
    gpu::DJSurfOut out{};
    out.pts = d.pts; out.mis = d.mis; out.count = d.count; out.cap = (int)cap;
    const bool deep = (maxDepth > BDPT_MAXDEPTH);
    // GRID SIZED TO THE WORK, not copied from the megakernel. Each thread reserves
    // `DVertex path[MAXV]` in local memory, so a fixed 2048x128 launch reserves hundreds of MB
    // of local store to run one subpath per thread -- and a per-CHUNK rebuild pays that setup
    // hundreds of times a render rather than once. The grid-stride loop is unchanged and every
    // subpath is seeded by its ABSOLUTE index, so the launch geometry cannot alter the map:
    // this is a pure setup-cost change, bit-identical by construction.
    int blocks = (int)((nPaths + 127) / 128);
    if (blocks < 1)    blocks = 1;
    if (blocks > 2048) blocks = 2048;      // still saturate the card on a large -n
    if (deep)
        gpu::kJSurfLightT<BDPT_DEEPDEPTH><<<blocks, 128>>>(sc, cam, diffraction, maxDepth,
                                                           nPaths, salt, out);
    else
        gpu::kJSurfLightT<BDPT_MAXDEPTH><<<blocks, 128>>>(sc, cam, diffraction, maxDepth,
                                                          nPaths, salt, out);
    cudaCheckKernel("jsurf-light");
    int n = 0;
    CUDA_CHECK(cudaMemcpy(&n, d.count, sizeof(int), cudaMemcpyDeviceToHost));
    // The cursor counts every ATTEMPT, so it overshoots when the cap binds; the writes past it
    // were skipped, not wrapped. Clamp and remember, so the ceiling is reportable rather than
    // a silent thinning of the map (which would darken the merges by an unknown factor).
    if (n > (int)cap) { d.dropped += (long long)n - (long long)cap; n = (int)cap; }
    if (n <= 0) {   // drew empty: the merge keeps its constant (kBdptT's kappaSurf), no points
        dsm.radius = (gpu::Real)radius;
        dsm.kappaS = (double)nPaths * PI * radius * radius;
        return;
    }
    d.stored = n;

    // --- the grid. surfmerge.h SurfMap::build's geometry, vcmSessionPass's machinery. ---
    d.arena.reset();
    ThrustArenaAlloc tal{&d.arena};
    auto pol = FT_THRUST_PAR(tal);
    const gpu::BboxF bb = thrust::transform_reduce(
        pol, thrust::device_pointer_cast(d.pts), thrust::device_pointer_cast(d.pts + n),
        gpu::JSurfToBboxF{}, gpu::BboxF{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX},
        gpu::BboxMergeF{});
    // The lattice over the photons as SurfMap::build's (mergeLatticeOver).
    const MergeLattice lat = mergeLatticeOver(bb.mnx, bb.mny, bb.mnz, radius, (size_t)n);
    int ckBlocks = (n + 127) / 128;
    if (ckBlocks < 1)    ckBlocks = 1;
    if (ckBlocks > 2048) ckBlocks = 2048;
    mergeBinDevice(pol, d.pts, (size_t)n, lat, ckBlocks, d.cellKey, d.cellKeyCap, d.order, d.orderCap,
                   d.cellStart, d.cellStartCap, &d.tag, &d.tagCap);

    dsm.pts = d.pts; dsm.mis = d.mis;
    dsm.cellStart = d.cellStart; dsm.order = d.order; dsm.tag = d.tag;
    dsm.lat = lat;
    dsm.radius = (gpu::Real)radius;
    // n_m counts subpaths ASKED FOR, not photons stored -- the host pass counts the same way
    // (it increments before its `pdfLam <= 0` continue), and getting it wrong scales every
    // merge by a constant.
    dsm.kappaS = (double)nPaths * PI * radius * radius;
    dsm.nPts   = (long long)n;
}

// ---- FTRACE_JDEVLIGHT=2: compare the host-traced and device-traced maps AS DATA ------
//
// The image A/B of these two arms is weak where it matters: the caustic is merge-dominated and
// high-variance, so a seed-spread error bar from a handful of renders cannot separate "the
// deposit is wrong" from "not enough seeds". The maps are the better instrument -- ~49 000
// independent samples each of the same distribution, so their summary statistics agree to
// ~1/sqrt(N), while every plausible transcription error is GROSS here: an off-by-one in the
// accumulator index moves the whole sumC distribution by a step of the recurrence, a wrong
// `vert` moves the depth histogram by a bin, a wrong site predicate moves the count and the
// gateC1 fraction, a wrong cosPrev/rho2 moves rCoef's median by a factor.
//
// Medians as well as means, because pdfFwdA and rCoef are densities: their means are dominated
// by a tail that two 49 000-sample draws will not agree on, and a mean that disagrees while the
// median agrees is a statement about the tail, not about the transcription.
static void compareJSurfMaps(const bdpt::SurfMap& host, const JSurfDev& dev) {
    if (dev.stored <= 0 || host.pts.empty()) { std::printf("[jdevcmp] nothing to compare\n"); return; }
    const size_t n = (size_t)dev.stored;
    std::vector<gpu::DSurfPhoton> dp(n);
    std::vector<gpu::DSurfMis>    dm(n);
    CUDA_CHECK(cudaMemcpy(dp.data(), dev.pts, n * sizeof(gpu::DSurfPhoton), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dm.data(), dev.mis, n * sizeof(gpu::DSurfMis), cudaMemcpyDeviceToHost));
    auto med = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto mean = [](const std::vector<double>& v) {
        double s = 0.0; for (double x : v) s += x; return v.empty() ? 0.0 : s / (double)v.size();
    };
    struct Col { const char* name; std::vector<double> h, d; };
    Col cols[] = {
        {"beta",    {}, {}}, {"pdfFwdA", {}, {}}, {"rCoef",  {}, {}},
        {"sumC",    {}, {}}, {"sumMb",   {}, {}}, {"sumMs",  {}, {}},
    };
    for (size_t i = 0; i < host.pts.size(); ++i) {
        const bdpt::SurfMis* m = host.misOf(i);
        if (!m) continue;
        cols[0].h.push_back((double)host.pts[i].beta);
        cols[1].h.push_back(m->pdfFwdA);
        cols[2].h.push_back((double)m->rCoef);
        cols[3].h.push_back(m->sumC);
        cols[4].h.push_back(m->sumMb);
        cols[5].h.push_back(m->sumMs);
    }
    for (size_t i = 0; i < n; ++i) {
        const gpu::DSurfMis& m = dm[dp[i].misIdx < n ? dp[i].misIdx : i];
        cols[0].d.push_back((double)dp[i].beta);
        cols[1].d.push_back(m.pdfFwdA);
        cols[2].d.push_back((double)m.rCoef);
        cols[3].d.push_back(m.sumC);
        cols[4].d.push_back(m.sumMb);
        cols[5].d.push_back(m.sumMs);
    }
    std::printf("\n[jdevcmp] host %zu photons vs device %zu (%.2f%%)\n",
                host.pts.size(), n,
                100.0 * ((double)n / (double)host.pts.size() - 1.0));
    // MAX AND HIGH QUANTILES, not just mean and median. The first run of this comparator gave
    // mean ratios of 4e-4 beside median ratios of 0.99, which says the draws differ in the
    // TAIL and nothing about whether the transcription is right -- a density's mean over
    // 12 000 samples is a statement about its largest one or two entries. These columns answer
    // the question that one could not: is the device MISSING the tail, or has it merely not
    // drawn it? It matters, because `pdfFwdA` is the merge technique's own density (eta =
    // kappaS * pl_j), so a missing tail under-weights exactly the near-specular merges a
    // caustic is made of -- which is the sign the reference comparison actually measured.
    auto quant = [](std::vector<double> v, double q) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[(size_t)(q * (double)(v.size() - 1))];
    };
    std::printf("[jdevcmp] %-9s %11s %11s %8s | %11s %11s %8s | %10s %10s %7s\n",
                "field", "host mean", "dev mean", "d/h", "host p99.9", "dev p99.9", "d/h",
                "host max", "dev max", "d/h");
    for (const Col& c : cols) {
        const double hm = mean(c.h), dmn = mean(c.d);
        const double hq = quant(c.h, 0.999), dq = quant(c.d, 0.999);
        const double hx = quant(c.h, 1.0),   dx = quant(c.d, 1.0);
        std::printf("[jdevcmp] %-9s %11.4g %11.4g %7.4fx | %11.4g %11.4g %7.4fx | %10.4g %10.4g %6.4fx\n",
                    c.name, hm, dmn, hm != 0.0 ? dmn / hm : 0.0,
                    hq, dq, hq != 0.0 ? dq / hq : 0.0,
                    hx, dx, hx != 0.0 ? dx / hx : 0.0);
    }
    std::printf("[jdevcmp] %-9s %11s %11s %8s | %11s %11s %8s\n",
                "(median)", "", "", "", "host med", "dev med", "d/h");
    for (const Col& c : cols) {
        const double hM = med(c.h), dM = med(c.d);
        std::printf("[jdevcmp] %-9s %11s %11s %8s | %11.4g %11.4g %7.4fx\n",
                    c.name, "", "", "", hM, dM, hM != 0.0 ? dM / hM : 0.0);
    }
    // gateC1 and the depth histogram: pure structure, no tail, so these are the columns that
    // catch a predicate or an index error outright rather than statistically.
    long long hg = 0, dg = 0;
    long long hv[16] = {0}, dv[16] = {0};
    for (size_t i = 0; i < host.pts.size(); ++i) {
        const bdpt::SurfMis* m = host.misOf(i);
        if (!m) continue;
        hg += (m->gateC1 != 0);
        if (m->vert < 16) ++hv[m->vert];
    }
    for (size_t i = 0; i < n; ++i) {
        const gpu::DSurfMis& m = dm[dp[i].misIdx < n ? dp[i].misIdx : i];
        dg += (m.gateC1 != 0);
        if (m.vert < 16) ++dv[m.vert];
    }
    std::printf("[jdevcmp] gateC1 on: host %.4f  dev %.4f\n",
                (double)hg / (double)host.pts.size(), (double)dg / (double)n);
    std::printf("[jdevcmp] vert histogram (fraction), host | dev:\n");
    for (int v = 0; v < 12; ++v) {
        if (!hv[v] && !dv[v]) continue;
        std::printf("[jdevcmp]   vert %2d: %.5f | %.5f\n", v,
                    (double)hv[v] / (double)host.pts.size(), (double)dv[v] / (double)n);
    }
    std::printf("\n");
}

// ---- upload a built photon-beam map to the device ------------------------------------
//
// Shared by mode M's volume gather and mode J's BDPT merges, which need the SAME device
// layout and differ only in whether the MIS partials come with it. Same fold-at-upload trick
// as DGatherPhoton: the per-beam constants (carried flux, the 1/nEmitted pass normalisation,
// and the CIE triple at the beam's own wavelength — the host precomputed the last one in
// BeamMap::build for exactly this reason) collapse into one float3, so the inner loop
// multiplies geometry and medium terms alone. The BVH goes over verbatim; nNodes == 0 is the
// "no volume gather" sentinel every entry point tests.
//
// `withMis` is mode J. It additionally uploads BeamMis + misIdx and sets `mergeKappa` —
// n_m * 2r, the merge technique's sample count times the 1D kernel's full width — which is
// what turns the device gather from mode M's whole estimator into one half of a two-technique
// MIS pair. The three travel together on purpose: a null `mis` makes DBeamMap::misOf return
// null, which the gather reads as "weight 1", and a zero `mergeKappa` deletes every merge
// term inside dMisWeight. Uploading one without the other would double-count or under-count.
// Upload of the surface photon map (`-jsurf`). Mirrors uploadBeamMapCuda's shape: convert the
// host vectors into device layouts, hand the pointers over in a DSurfMap, and let `up` own the
// allocations. An absent map leaves `dsm.kappaS == 0`: no surface merges, the two-technique
// estimator unchanged. A map that drew empty keeps its kappaS with `nPts == 0`: the technique
// stays in every weight and gathers nothing, as on the host.
static void uploadSurfMapCuda(const bdpt::SurfMap* smap, DUpload& up, gpu::DSurfMap& dsm) {
    if (!smap || smap->nEmitted <= 0 || !(smap->radius > 0.0)) return;
    if (smap->pts.empty()) {   // drew empty: the constant, no points
        dsm.radius = (gpu::Real)smap->radius;
        dsm.kappaS = (double)smap->nEmitted * PI * (double)smap->radius * (double)smap->radius;
        return;
    }
    if (smap->cellStart.empty()) return;   // points but no grid (radiusOnly): nothing to gather
    const size_t n = smap->pts.size();
    std::vector<gpu::DSurfPhoton> pts(n);
    for (size_t i = 0; i < n; ++i) {
        const bdpt::SurfPhoton& q = smap->pts[i];
        gpu::DSurfPhoton& d = pts[i];
        d.p  = gpu::DVec3{(gpu::Real)q.p.x,  (gpu::Real)q.p.y,  (gpu::Real)q.p.z};
        d.wo = gpu::DVec3{(gpu::Real)q.wo.x, (gpu::Real)q.wo.y, (gpu::Real)q.wo.z};
        d.lambda = q.lambda; d.beta = q.beta;
        d.cx = q.cx; d.cy = q.cy; d.cz = q.cz;
        d.misIdx = q.misIdx;
        d.nx = q.nx; d.ny = q.ny; d.nz = q.nz;
    }
    // Every SurfMis field is copied below, and a field a copy misses reaches the device as 0. Both
    // structs carry their padding explicitly, so any field added to either changes its size and
    // stops the build here until the copy has been revisited.
    static_assert(sizeof(bdpt::SurfMis) == 40, "SurfMis changed: copy the new field into DSurfMis "
                                               "here, then update this size");
    static_assert(sizeof(gpu::DSurfMis) == 40, "DSurfMis changed: fill the new field from SurfMis "
                                               "here, then update this size");
    std::vector<gpu::DSurfMis> mis(smap->mis.size());
    for (size_t i = 0; i < smap->mis.size(); ++i) {
        const bdpt::SurfMis& m = smap->mis[i];
        gpu::DSurfMis& d = mis[i];
        d.sumC = m.sumC; d.sumMb = m.sumMb; d.sumMs = m.sumMs; d.pdfFwdA = m.pdfFwdA;
        d.rCoef = m.rCoef; d.gateC1 = m.gateC1; d.vert = m.vert;
    }
    dsm.pts       = (const gpu::DSurfPhoton*)up.keep(uploadVec(pts));
    dsm.mis       = (const gpu::DSurfMis*)up.keep(uploadVec(mis));
    dsm.cellStart = (const int*)up.keep(uploadVec(smap->cellStart));
    dsm.order     = (const int*)up.keep(uploadVec(smap->order));
    dsm.tag       = (const unsigned char*)up.keep(uploadVec(smap->tag));
    dsm.lat    = smap->lat;
    dsm.radius = (gpu::Real)smap->radius;
    // n_m * pi r_s^2, the constant the estimator divides by and the weight scales its own
    // technique with -- the exact twin of the beam map's mergeKappa = n_m * 2 r_b.
    dsm.kappaS = (double)smap->nEmitted * PI * (double)smap->radius * (double)smap->radius;
    dsm.nPts   = (long long)n;
}

// Mode J: draw every sub-beam's DBeamRec::tr0 (kBeamTr0), one walk per parent beam. The host
// split pushes a parent's sub-beams contiguously and in order of s0; a map that is not laid out
// that way is left at tr0 = -1, where every hit walks from the beam's origin as before.
static void drawBeamTr0(const BeamMap* bmap, DUpload& up, gpu::DBeamMap& dbm) {
    if (!bmap || !dbm.beams || !dbm.mis) return;
    {   // FTRACE_BEAMTR0=0: leave tr0 at -1 (every hit walks from the beam's origin) -- the A/B arm
        static const char* e = std::getenv("FTRACE_BEAMTR0");
        if (e && std::atoi(e) == 0) return;
    }
    const auto tr0T0 = std::chrono::steady_clock::now();
    const std::vector<int>& mi = bmap->misIdx;
    const size_t nb = bmap->beams.size();
    if (mi.size() != nb || nb == 0) return;
    std::vector<int> start; start.reserve(bmap->mis.size() + 1);
    std::vector<char> seen(bmap->mis.size(), 0);
    for (size_t i = 0; i < nb; ++i) {
        if (i == 0 || mi[i] != mi[i - 1]) {
            const int p = mi[i];
            if (p < 0 || (size_t)p >= seen.size() || seen[p]) return;
            seen[p] = 1;
            start.push_back((int)i);
        } else if (bmap->beams[i].s0 < bmap->beams[i - 1].s0) {
            return;
        }
    }
    start.push_back((int)nb);
    const int nP = (int)start.size() - 1;
    const int* dStart = (const int*)up.keep(uploadVec(start));
    float* dTr0 = (float*)up.keep(uploadVec(std::vector<float>(nb, -1.0f)));
    // Keyed to the realization (BeamMap::salt), not to the call: a flyby renders the same
    // realizations in every frame (-beamepochs, -beamfreeze), and a draw that changed per call
    // would make the beams' transmittance through a heterogeneous medium flicker between them.
    gpu::kBeamTr0<<<(nP + 127) / 128, 128>>>(up.sc, dbm.beams, dTr0, dStart, nP,
                                             mix64(bmap->salt ^ 0xD6E8FEB86659FD93ull));
    dbm.tr0 = dTr0;
    cudaCheckKernel("beam-tr0");
    if (std::getenv("FTRACE_BEAMTR0_TIME")) {
        cudaDeviceSynchronize();
        std::fprintf(stderr, "[beam-tr0] %d parents, %zu sub-beams: %.1f ms\n", nP, nb,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0T0).count());
    }
}

// `scene` names each deposit-folded beam's emitter for the gather only where the beam's colour is
// that emitter's mean (gatherPhotonBeamsW's flatFoldEmitter, where the why is).
static void uploadBeamMapCuda(const Scene& scene, const BeamMap* bmap, DUpload& up, gpu::DBeamMap& dbm,
                              const StageProgress* stage, bool withMis, bool lbvh = false) {
    // `lbvh` means the host deliberately skipped its tree because a device one is coming, so an
    // empty `bvh.nodes` is expected rather than a malformed map. Without that distinction the
    // guard below would silently drop a perfectly good beam set and the volume would vanish.
    if (!bmap || bmap->nEmitted <= 0) return;
    // A map that drew empty: no beams to upload, but the merge technique stays the render's
    // (kBdptT's mergeOn), with its constant.
    if (bmap->beams.empty()) {
        if (withMis) dbm.mergeKappa = (double)bmap->nEmitted * 2.0 * bmap->radRef();
        return;
    }
    if (!lbvh && bmap->bvh.nodes.empty()) return;
    const double invN = 1.0 / (double)bmap->nEmitted;
    const size_t nb = bmap->beams.size();
    // The fourth and last silent stretch: converting millions of sub-beams and millions of
    // BVH nodes into their device layouts, on one host thread, before the line at the bottom
    // of this block finally says anything. Name it and make it interruptible.
    if (stage && stage->reset) stage->reset();
    if (stage && stage->report) stage->report("uploading photon beams", 0, (long long)nb,
                                              nullptr, 0.0);
    std::vector<gpu::DBeamRec> recs(nb);
    for (size_t i = 0; i < nb; ++i) {
        if ((i & 0xFFFFF) == 0) {
            if (ft::stopRequested()) break;
            if (stage && stage->report)
                stage->report("uploading photon beams", (long long)i, (long long)nb,
                              nullptr, 0.0);
        }
        const PhotonBeam& b = bmap->beams[i];
        // GATHER-TIME SPECTRAL FOLD (achro == 2, scene.h Scene::BowLut). `bmap->cie[i]` holds
        // the emitter's mean CIE for such a beam, but the whole point of the gather-time fold
        // is that the colour is only decidable once the scattering angle is known — and the
        // device gather has no bow table to consult. Using the mean here would pair a
        // band-averaged colour with a SINGLE-wavelength phase value, which is neither the
        // folded estimator nor the monochromatic one. So demote to the monochromatic record:
        // CIE(lambda) is exactly what this beam meant before 0.256.0, so a device gather of a
        // CPU-traced map stays unbiased — it is simply as noisy as it was, and re-gathering on
        // the CPU is what buys the fold. (Device parity is tracked in known-issues.md.)
        const Vec3 ci = (b.achro == 2)
                        ? Vec3(cieX((double)b.lambda), cieY((double)b.lambda), cieZ((double)b.lambda))
                        : bmap->cie[i];
        gpu::DBeamRec& r = recs[i];
        r.o = gpu::DVec3(b.o.x, b.o.y, b.o.z);
        r.d = gpu::DVec3(b.d.x, b.d.y, b.d.z);
        r.s0 = b.s0; r.len = b.len;
        // `invC` splits the chord's flux evenly over its spectral bundle — every wavelength
        // in it carries the same power (photonbeams.h) — and is exactly 1.0 for a
        // monochromatic beam, so a `-beamspec 1` render is bit-identical.
        const double invC = 1.0 / (double)b.nLam();
        const double w = (double)b.power * invN * invC;
        r.pX = (float)(ci.x * w); r.pY = (float)(ci.y * w); r.pZ = (float)(ci.z * w);
        // ...and the colourless weight beside it, for the gather-time fold (FOLD-GPU (2)): a
        // folded beam takes its colour from the bow table at the crossing angle, so it needs
        // the power without a colour baked in. `pX/pY/pZ` stay exactly as they were, and are
        // still what a beam with no table uses -- the demotion above is now the FALLBACK
        // rather than the only choice.
        r.pw = (float)w;
        // A gather-time fold names its emitter in `emIdx`; a DEPOSIT fold (achro == 1) carries it
        // in `srcEm`, and JSPEC needs it to re-colour the beam for a chromatic camera path.
        bool flatFold = false;
        if (b.achro == 1 && b.srcEm >= 0 && b.srcEm < (int)scene.emitters.size()) {
            const Vec3& cm = scene.emitters[(size_t)b.srcEm].cieMean;
            flatFold = b.cieA[0] == (float)cm.x && b.cieA[1] == (float)cm.y && b.cieA[2] == (float)cm.z;
        }
        r.emIdx = (b.achro == 2) ? (short)b.emIdx : (flatFold ? (short)b.srcEm : (short)-1);
        r.achro = (short)b.achro;
        r.lambda = b.lambda; r.absorb = b.absorb; r.med = b.med;
        r.nSec = (unsigned char)b.nSec; r.pad = 0;
        for (int k = 0; k < 3; ++k) {
            // The same shared constant, minus the CIE fold, times this member's relative
            // spectral throughput. `wS` is 1 for every member of a bundle whose path never
            // diverged spectrally, so a pre-0.257.0 map uploads to exactly the old numbers.
            r.pwSec[k] = (k < b.nSec) ? (float)(w * (double)b.wS[k]) : 0.0f;
            r.lamS[k] = (k < b.nSec) ? gpu::DBeamRec::packLam((double)b.lamS[k]) : (unsigned short)0;
        }
        const double rm = bmap->radOf(b.med);
        r.invRad = (float)(rm > 0.0 ? 1.0 / rm : 0.0);
    }
    // With `lbvh` there is no host tree to convert; the device builds one straight after.
    std::vector<gpu::DNode> bnodes(lbvh ? 0 : bmap->bvh.nodes.size());
    for (size_t i = 0; i < bnodes.size(); ++i) {
        if ((i & 0xFFFFF) == 0 && ft::stopRequested()) break;
        const BvhNode& s = bmap->bvh.nodes[i]; gpu::DNode& d = bnodes[i];
        d.lo = {s.box.lo.x, s.box.lo.y, s.box.lo.z};
        d.hi = {s.box.hi.x, s.box.hi.y, s.box.hi.z};
        d.left = s.left; d.right = s.right; d.first = s.first; d.count = s.count;
    }
    // A stop during either conversion leaves `recs`/`bnodes` half-built. Don't pay the PCIe
    // transfer for data nothing will read, and above all don't print the "uploaded" line,
    // which would be a plain lie in the log about what the device holds. `dbm` stays zeroed,
    // and `nNodes == 0` is already the "no volume gather" sentinel.
    if (ft::stopRequested()) return;
    dbm.beams     = (const gpu::DBeamRec*)up.keep(uploadVec(recs));
    dbm.nodes     = (const gpu::DNode*)up.keep(uploadVec(bnodes));
    dbm.primIdx   = (const int*)up.keep(uploadVec(bmap->bvh.primIdx));
    dbm.radiusMax = (gpu::Real)bmap->radius;
    dbm.nNodes    = (int)bnodes.size();
    // VOLCACHE grids, if the host built (and split against) them for this map. The host sets
    // g_volCachesForUpload right before the device render and clears it after; a null means
    // "no cache", and the march is then a no-op. See DVolCache.
    dbm.vc = nullptr; dbm.nVc = 0;
    dbm.sunNee = pbeams::gSunNee ? 1 : 0;   // after sunNeeSplit (buildBeamMap) has had its say
    if (g_volCachesForUpload && !g_volCachesForUpload->empty()) {
        std::vector<gpu::DVolCache> dv(g_volCachesForUpload->size());
        size_t nReady = 0;
        for (size_t m = 0; m < dv.size(); ++m) {
            const VolCache& h = (*g_volCachesForUpload)[m];
            gpu::DVolCache& d = dv[m];
            d.ready = h.ready ? 1 : 0;
            if (!h.ready) continue;
            d.nx = h.nx; d.ny = h.ny; d.nz = h.nz; d.nSH = h.nSH; d.nCh = h.nCh;
            d.bowMode = h.bowMode ? 1 : 0; d.gHG = h.gHG;
            d.lo = {h.lo.x, h.lo.y, h.lo.z}; d.hi = {h.hi.x, h.hi.y, h.hi.z};
            for (int k = 0; k < 3; ++k) d.kMom[k] = {h.kMom[k].x, h.kMom[k].y, h.kMom[k].z};
            d.sh = (const float*)up.keep(uploadVec(h.sh));
            ++nReady;
        }
        if (nReady) {
            dbm.vc  = (const gpu::DVolCache*)up.keep(uploadVec(dv));
            dbm.nVc = (int)dv.size();
            std::printf("[gpu] volcache: %zu medium grid(s) uploaded for the device march\n", nReady);
        }
    }
    size_t nMis = 0;
    if (withMis && !bmap->mis.empty()) {
        // Every BeamMis field is copied below; see the SurfMis upload for why these trip.
        static_assert(sizeof(BeamMis) == 56, "BeamMis changed: copy the new field into DBeamMis "
                                             "here, then update this size");
        static_assert(sizeof(gpu::DBeamMis) == 56, "DBeamMis changed: fill the new field from "
                                                   "BeamMis here, then update this size");
        std::vector<gpu::DBeamMis> dm(bmap->mis.size());
        for (size_t i = 0; i < dm.size(); ++i) {
            const BeamMis& s = bmap->mis[i]; gpu::DBeamMis& d = dm[i];
            // BOTH merge kinds since 0.263.0: `sumMs` and `etaPrevS` are the point x point
            // half, zero unless `-jsurf` filled a surface map, and every term they feed is
            // multiplied by kappaS -- which is itself zero without one. So an unused surface
            // map costs exactly the two loads and nothing else.
            d.sumC = s.sumC; d.sumM = s.sumMb; d.sumMs = s.sumMs;
            d.pdfDir = s.pdfDir; d.pdfPlanar = s.pdfPlanar; d.rCoef = s.rCoef;
            d.etaPrev = s.etaPrev; d.etaPrevS = s.etaPrevS; d.leadIn = s.leadIn;
            d.sunLeg = s.sunLeg;
            d.gateC1 = (int)s.gateC1; d.vert = (int)s.vert;
        }
        nMis = dm.size();
        dbm.mis    = (const gpu::DBeamMis*)up.keep(uploadVec(dm));
        dbm.nMis   = (int)nMis;
        if (!bmap->misIdx.empty())
            dbm.misIdx = (const int*)up.keep(uploadVec(bmap->misIdx));
        dbm.mergeKappa = (double)bmap->nEmitted * 2.0 * bmap->radRef();
        dbm.sinMin = bmap->sinMin;
    }
    double rlo = bmap->radius;
    for (float rm : bmap->radMed) if (rm > 0.f) rlo = std::min(rlo, (double)rm);
    char rtxt[64];
    if (rlo < bmap->radius) std::snprintf(rtxt, sizeof rtxt, "%.4g .. %.4g", rlo, bmap->radius);
    else                    std::snprintf(rtxt, sizeof rtxt, "%.4g", bmap->radius);
    // Silent on a light-side refresh epoch: the sub-beam / BVH / radius figures describe the
    // map's SHAPE, which every epoch of a run shares — see g_gpuQuietRebuild.
    if (g_gpuQuietRebuild) return;
    if (nMis)
        std::printf("[gpu] photon beams: %zu sub-beams, %zu BVH nodes, kernel radius %s, "
                    "%zu MIS entries (kappa %.4g) uploaded for the mode-J merges\n",
                    nb, bnodes.size(), rtxt, nMis, dbm.mergeKappa);
    else
        std::printf("[gpu] photon beams: %zu sub-beams, %zu BVH nodes, kernel radius %s "
                    "uploaded for the volume gather\n", nb, bnodes.size(), rtxt);
}

// renderBdptCuda's instantiation, besides its merges: the hero-wavelength bundle where asked and the
// render keeps it -- the CPU BDPT's gate (bdpt.h BdptRenderer::renderRows): a participating medium
// makes the shadow-ray transmittance wavelength-dependent, GRIN bends each lambda differently, and a
// physical lens disperses the primary ray, so all three fall back to the single-lambda kernel -- and
// the deep stack past BDPT_MAXDEPTH. The launcher picks by it, and the VRAM preflight weighs by it
// (cudaBdptRoute).
struct BdptPick { bool hero, deep; };
static bool bdptDeep(int maxDepth) { return maxDepth > BDPT_MAXDEPTH; }
static BdptPick bdptPick(int heroC, bool media, bool grin, bool lens, int maxDepth) {
    return BdptPick{heroC > 1 && !media && !grin && !lens, bdptDeep(maxDepth)};
}
// The kBdptT instantiation a pick runs: renderBdptCuda launches through it and cudaBdptRoute weighs
// it, so the two name one kernel. `merge`: mode J's merges are on (the scalar merge instantiation);
// `strat`: the beam queue's stratified wave order (wfStratOn).
using BdptKernelFn = decltype(&gpu::kBdptT<0, BDPT_MAXDEPTH, false, false>);
static BdptKernelFn bdptKernelFor(const BdptPick& p, bool merge, bool strat) {
    using namespace gpu;
    if (merge)
        return p.deep ? (strat ? kBdptT<0, BDPT_DEEPDEPTH, true, true> : kBdptT<0, BDPT_DEEPDEPTH, true, false>)
                      : (strat ? kBdptT<0, BDPT_MAXDEPTH, true, true> : kBdptT<0, BDPT_MAXDEPTH, true, false>);
    if (p.hero)
        return p.deep ? kBdptT<BDPT_NSEC, BDPT_DEEPDEPTH, false, false> : kBdptT<BDPT_NSEC, BDPT_MAXDEPTH, false, false>;
    return p.deep ? kBdptT<0, BDPT_DEEPDEPTH, false, false> : kBdptT<0, BDPT_MAXDEPTH, false, false>;
}
// Whether mode J's beam queue takes its waves in stratified order (FTRACE_WFSTRAT=1, a diagnostic;
// bands by default), which the STRAT instantiation runs.
static bool wfStratOn() {
    const char* e = std::getenv("FTRACE_WFSTRAT");
    return e && std::atoi(e) != 0;
}
// Whether -jsurf's point merges queue for kWfSurfGather (the default) rather than all merging inline
// (FTRACE_NOSURFWF=1, a diagnostic).
static bool surfGatherQueued() {
    const char* e = std::getenv("FTRACE_NOSURFWF");
    return !(e && std::atoi(e) != 0);
}

Film renderBdptCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                    long long spp, int maxDepth, bool diffraction, const SppProgress* prog,
                    int heroC, const BeamMap* bmap, const StageProgress* stage,
                    const bdpt::SurfMap* smap, const CudaSceneHold* held) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    g_lastBdptWork = 0;
    g_lastBdptSetupWork = 0;
    if (!cudaAvailable() || !cudaBdptSupported(scene)) return out;
    // Everything but the chunks is this call's fixed cost (cudaLastBdptOverheadSec): what a
    // mode-J epoch pays on top of its samples.
    const auto tCall0 = std::chrono::steady_clock::now();
    // The call's camera work (cudaLastBdptWork): the kernels' count, and the beam hits the
    // waves evaluate.
    {
        const unsigned long long zero[kDWorkSlots] = {};
        CUDA_CHECK(cudaMemcpyToSymbol(g_dWork, zero, sizeof zero));
    }
    long long wfHitsCall = 0;
    // The call's camera work so far: the kernels' count, and the beam hits the waves evaluated.
    const auto workSoFar = [&]() -> long long {
        unsigned long long w[kDWorkSlots] = {};
        CUDA_CHECK(cudaMemcpyFromSymbol(w, g_dWork, sizeof w));
        long long sum = wfHitsCall;
        for (unsigned long long x : w) sum += (long long)x;
        return sum;
    };
    struct WorkSoFarScope {   // cudaBdptCallWork, for this call's length
        explicit WorkSoFarScope(std::function<long long()> f) { g_bdptWorkSoFar = std::move(f); }
        ~WorkSoFarScope() { g_bdptWorkSoFar = nullptr; }
    } workScope{workSoFar};
    // Two kernel variants: the default shallow stack, and a deep one for scenes that need
    // it. Anything past BDPT_DEEPDEPTH still clamps — going deeper would need another
    // instantiation, and the local-memory footprint is already ~13 KB/thread there.
    const bool deep = bdptDeep(maxDepth);   // pick.deep, below, by the same rule
    if (maxDepth > BDPT_DEEPDEPTH) maxDepth = BDPT_DEEPDEPTH;   // device array bound

    DUpload up;
    sceneOrHeld(scene, held, up, "renderBdptCuda");
    up.dc = bakeCamera(scene, cam, resX, resY, up);
    // Mode J under a blur: its epoch is one time (the host's PassTimeScope, atEpochJ), which the
    // light subpaths and the camera pass take; mode D's paths take their own (g_passMotionU < 0).
    up.sc.mo.passU = hostPassU(scene);
    // The aimed sun (MODE-J-SDS, 0.387.0): before the first light subpath of this render, which
    // in mode J is the device light pass below, ahead of the per-render J constants.
    uploadSunAim();

    // The hero-wavelength gate (bdptPick). (GRIN never reaches here at all -- cudaBdptSupported
    // rejects it outright.)
    int C = heroC;
    if (C > hero::kHeroMax) C = hero::kHeroMax;
    if (C < 1) C = 1;
    const BdptPick pick = bdptPick(C, up.sc.mediaN != 0, up.sc.hasGrin != 0, cam.hasLens(), maxDepth);

    const size_t npix = (size_t)resX * resY;
    double* d_cam   = nullptr; CUDA_CHECK(cudaMalloc(&d_cam,   npix * 3 * sizeof(double)));
    double* d_splat = nullptr; CUDA_CHECK(cudaMalloc(&d_splat, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cam,   0, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_splat, 0, npix * 3 * sizeof(double)));
    // A CHUNKED render (gpuSppChunks) stops part-way: its kernels honour the device stop flag
    // (DeviceStopScope), and a chunk the stop cut short is dropped. Its partial sums in d_cam /
    // d_splat need no undoing: gpuSppChunks downloads nothing after a dropped chunk, and the films
    // are freed unread. A single-shot render honours no stop and runs whole.
    const bool abortable = prog && prog->report;
    std::unique_ptr<DeviceStopScope> stopScope;
    if (abortable) stopScope = std::make_unique<DeviceStopScope>();
    // A stop the host has seen, or -time's end cutting the chunk (gpuSppChunks raises the device's
    // flag alone): no more waves or passes are launched.
    auto stopped = [&] {
        return abortable && (ft::stopRequested() || *(volatile int*)deviceStopFlag().host != 0);
    };
    // Resume (mode D disk resume): fresh samples carry ABSOLUTE indices
    // [sppBase, sppBase + spp), so the kernel's stride is the FINAL total (see
    // renderBackwardCuda for why). The loaded sample count is also mixed into the seed
    // base so the continued samples stay decorrelated from the ones already in the
    // checkpoint film — mode D is Monte-Carlo throughout, so there is no deterministic
    // lattice to continue exactly, only variance to keep independent.
    const long long sppBase  = prog ? (long long)prog->sampleBase : 0;
    const long long sppTotal = sppBase + spp;
    const unsigned long long seed = 0x9e3779b97f4a7c15ULL
        ^ (prog ? (unsigned long long)prog->sampleBase * 0x9E3779B97F4A7C15ULL : 0ULL)
        ^ g_rngSalt;   // `-seed`; 0 by default, so XOR is a no-op

    std::vector<double> camH(npix * 3), splatH(npix * 3);
    auto download = [&](Film& o) {
        CUDA_CHECK(cudaMemcpy(camH.data(),   d_cam,   npix * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(splatH.data(), d_splat, npix * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < npix; ++i)   // SUM (cam+splat); writeFilm divides by spp
            o.xyz[i] = Vec3(camH[3 * i + 0] + splatH[3 * i + 0],
                            camH[3 * i + 1] + splatH[3 * i + 1],
                            camH[3 * i + 2] + splatH[3 * i + 2]);
    };
    // Mode J: the beam map, and with it the merge half of the estimator. Uploaded here and
    // freed with the rest of `up`. A null map leaves `mergeKappa == 0` and `nNodes == 0`, which
    // the kernel reads as "no merges" — so `-nobeams` degenerates to mode D exactly. A refresh
    // that drew empty keeps its `mergeKappa` and gathers nothing (uploadBeamMapCuda).
    DBeamMap dbm{};
    // Default on (0.272.3). Must agree EXACTLY with main.cpp's jSkipHostBvh(): if the host
    // skipped its tree and this said no, the map would arrive with no tree at all and the
    // volume would silently vanish. Two predicates, one condition -- so the condition is
    // written the same way in both, and `g_jHostLight` is the single switch behind both.
    const bool envOff = std::getenv("FTRACE_JLBVH") &&
                        std::atoi(std::getenv("FTRACE_JLBVH")) == 0;
    const bool wantLbvh = !g_jHostLight && !envOff;
    if (bmap) uploadBeamMapCuda(scene, bmap, up, dbm, stage, /*withMis=*/true, wantLbvh);
    if (bmap) drawBeamTr0(bmap, up, dbm);
    // FTRACE_JLBVH=1: rebuild the NODES on the device over the very same sub-beams. Same beams,
    // same gather, a different tree -- so an image difference is a traversal-acceptance
    // difference and a time difference is tree quality, with no third variable to blame.
    JBeamDev jbvh;
    if (dbm.beams && bmap && !bmap->beams.empty() && wantLbvh) {
        const int nb = (int)bmap->beams.size();
        const auto t0 = std::chrono::steady_clock::now();
        buildBeamLbvhDevice(jbvh, dbm.beams, nb, bmap->worldBounds, bmap->radius, dbm);
        CUDA_CHECK(cudaDeviceSynchronize());
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        static bool said = false;
        if (!said) {
            said = true;
            std::printf("[jlbvh] device LBVH over %d sub-beams in %.1f ms (host SAH tree had %zu "
                        "nodes; LBVH has %d)\n", nb, ms, bmap->bvh.nodes.size(), dbm.nNodes);
        }
    }
    // A map with beams always reaches the device with a tree: without one its merges would keep
    // their weight (mergeKappa) and gather nothing -- the image dark by the merges' share, silently.
    // An internal error, so it ends the process as CUDA_CHECK does: thrown, it would leak this
    // call's device allocations into a -serve loop that carries on.
    if (bmap && dbm.beams && dbm.nNodes == 0 && !ft::stopRequested()) {
        std::fprintf(stderr, "[gpu] internal error: a beam map of %zu sub-beam(s) reached the "
                             "device without a tree\n", bmap->beams.size());
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
    // `-jsurf`'s map. (The "increment 1: ... an uploaded map changes nothing yet" this comment
    // used to carry was true for one version. The gather landed in 0.263.1 -- see `mergeAny`
    // below, which launches the MERGE kernel for a surface map with no beams, and
    // `dSurfMergeAt`.)
    // FTRACE_JDEVLIGHT=1: re-trace and re-grid that map ON THE DEVICE instead. Deliberately
    // the same nPaths / radius / single realization as the host map it replaces, so this is a
    // clean A/B of WHERE the light side runs. It cannot be bit-identical -- the two walks draw
    // from different RNG streams -- so the acceptance test is statistical, per ROI.
    JSurfDev jdev;
    // DEFAULT 3 -- the device light pass, redrawn every block of samples (surfBlock). `-jhostlight` turns it off;
    // FTRACE_JDEVLIGHT still overrides both, because level 1 (device map, once per epoch) and
    // level 2 (that plus the host-vs-device map dump) are the two arms every measurement in
    // known-issues was taken with and they have to stay reachable.
    //
    // THE LEVELS ARE NAMED ARMS, NOT A VERBOSITY LADDER. Read them as a set, never as an
    // ordering where a higher number implies everything below it:
    //     0  host light pass (`-jhostlight`)
    //     1  device map, rebuilt once per epoch
    //     2  = 1 plus the host-vs-device map dump
    //     3  device map, redrawn every few samples  <- the default
    //     4  = 3 plus the dump  (for timing the dump against 3 inside one binary)
    // Getting this wrong is what JDEVCMP-DEFAULT was: the dump's gate read `>= 2`, which was
    // right while the default was 1 and silently swept the default into the diagnostic arm the
    // moment it became 3.
    //
    // DECIDED BEFORE THE UPLOAD BELOW, because the answer changes whether the upload is worth
    // doing at all: `buildJSurfMapDevice` opens with `dsm = gpu::DSurfMap{}`, so on the device
    // path every byte `uploadSurfMapCuda` converts and ships is overwritten a few lines later.
    const int jdevLevel = jDevLightLevel();
    // The subpaths the device draws its surface map from, and the photon slots it reserves: the
    // host map's own count and twice its size, or what the host sized for a device-drawn map
    // whose host pass traced only the beams' subpaths (SurfMap::devPaths, mode J's fixed map).
    const long long jDevPaths = smap ? (smap->devPaths > 0 ? smap->devPaths : smap->nEmitted) : 0;
    const long long jDevCap   = smap ? (smap->devCap > 0 ? smap->devCap
                                                         : (long long)smap->pts.size() * 2 + 1024) : 0;
    const bool jDevLight = smap && jDevPaths > 0 && jdevLevel != 0;

    DSurfMap dsm{};
    // Skip the host map's upload when the device is about to build its own over the top. The
    // cost is not the PCIe transfer but the two staging vectors built ahead of it, which walk
    // every surface photon on one core to widen it into the device layout.
    if (smap && !jDevLight) uploadSurfMapCuda(smap, up, dsm);
    // Level 3 redraws per block of samples (surfBlock, below): remember what the rebuild needs,
    // since the launch lambda has no `smap` of its own and must not reach for one that may be freed.
    const bool jPerChunk   = jDevLight && jdevLevel >= 3;
    long long startLight = 0;   // the map built here, in device units (the count is zero before it)
    if (jDevLight) {
        // Per block, this is block 0's realization (its salt, below), which the first chunk keeps.
        buildJSurfMapDevice(jdev, up.sc, up.dc, diffraction, maxDepth, jDevPaths,
                            smap->radius, jDevCap,
                            jPerChunk ? jSurfBlockSalt(sppBase) : jSurfSalt(prog),
                            dsm);
        startLight = workSoFar();
        g_lastBdptSetupWork = startLight;
        // ONCE. A refreshing render rebuilds this every epoch -- 126 of them in 90 s -- and a
        // line each would bury the -interval status lines that carry the actual progress.
        static bool jdevSaid = false;
        if (!jdevSaid || jdev.dropped) {
            jdevSaid = true;
            std::printf("[jdevlight] device light pass: %lld subpaths -> %lld surface photons "
                        "(host traced %zu)%s\n", jDevPaths, jdev.stored,
                        smap->pts.size(),
                        jdev.dropped ? "  [CAP BOUND: some photons dropped]" : "");
        }
        // `== 2`, NOT `>= 2`. The levels are not a severity ladder where each one implies the
        // one below -- they are three named arms, and level 2 is specifically "device map plus
        // the host-vs-device dump". When the default moved from 1 to 3 in 0.272.x, `>= 2`
        // silently swept the DEFAULT into the diagnostic arm: every shipped mode-J render with
        // `-jsurf` downloaded the device map and ran the whole median/histogram comparison once
        // per light-side epoch. On a 256-spp `_fog_cornell` that is 20 dumps against 9 `[spp]`
        // status lines -- the diagnostic output outnumbering the progress output 2:1, which is
        // the exact failure the `jdevSaid` latch a few lines above exists to prevent.
        //
        // LEVEL 4 = LEVEL 3 PLUS THE DUMP, and it exists so the cost of the dump can be measured
        // WITHIN ONE BINARY. Timing the bug by comparing a pre-fix build against a post-fix one
        // is not a controlled experiment: the two runs are minutes apart on a card whose clocks
        // depend on how hot it already is, and a first attempt that way came out "2.2x slower
        // after the fix" -- the beam-hit kernel alone drifting 7.5 s -> 15.7 s on an identical
        // 6.8 M sub-beam map that neither arm touches. Arms that can be interleaved, seed by
        // seed, in one process image are the only ones that answer a throughput question here.
        if (jdevLevel == 2 || jdevLevel == 4) compareJSurfMaps(*smap, jdev);
    }
    // ONE SURFACE REALIZATION A BLOCK OF SAMPLES (jPerChunk, the device drawing the surface map).
    // Realization k covers the call's samples [callBase + kB, callBase + (k+1)B) and is salted by
    // its first, so which samples meet which map is set by the device's counts, never by the
    // chunks, whose sizes the clock sets (GPU-J-CLOCK). B (`surfBlock`, spp) is decided from the
    // counts of the call's first sample, launched alone whatever its chunk (FTRACE_CHUNK_SPP's
    // too): as many samples as keep a realization's light
    // pass at kSurfShare of their camera work -- about where four realizations a chunk measured
    // best (0.390x whole-frame variance against one, for 8.6 % of the samples; sixteen cost 53 %
    // and came out at 0.503x). A chunk is split at the blocks' edges, and a block's map is drawn
    // once, however many chunks it spans.
    // Under -beamepochs (SppProgress::surfPerSample) one realization a sample: the layout every
    // frame of a flight shares, which counts taken from each frame would not.
    constexpr double kSurfShare = 0.1;
    long long surfBlock = (prog && prog->surfPerSample) ? 1 : 0;   // 0: until the first sample prices one
    long long surfDrawn = jPerChunk ? 0 : -1;  // the block whose map dsm holds
    long long surfLightUnits = startLight;     // its light pass, in device units
    if (jPerChunk) {
        static bool said = false;
        if (!said) {
            said = true;
            std::printf("mode J: light side on the DEVICE, its surface photons redrawn every few "
                        "samples (-jhostlight for the pre-0.272.0 host pass)\n");
        }
    }
    // FTRACE_JBAND=1: redraw per WAVEFRONT BAND as well as per block. The band loop
    // subdivides a chunk by PIXEL, which is the only axis left once the chunk is down to 1 spp
    // -- the case an expensive media scene is always in. A band is a wave, which the beam queue
    // and the point merges' queue both cut (so FTRACE_NOSURFWF=1 changes the bands too); without
    // either queue there is one band per chunk. Its bands follow the chunks, which the clock
    // sizes: a diagnostic, it does not repeat run to run.
    bool jPerBand = false;
    if (jPerChunk) {
        const char* e = std::getenv("FTRACE_JBAND");
        jPerBand = e && std::atoi(e) != 0;
        if (jPerBand)
            std::printf("[jdevlight] one light-side realization per wavefront band "
                        "(FTRACE_JBAND) — pixels in one frame may gather from different "
                        "realizations; unbiased per pixel, but the realization is no longer "
                        "shared across the image\n");
    }
    const long long jPaths = jDevLight ? jDevPaths : 0;
    const double    jRad   = jDevLight ? smap->radius : 0.0;
    const long long jCap   = jDevLight ? jDevCap : 0;
    // Mode J's switches, as the host settled them for this render (cudaSetJSwitches).
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jHalf, &g_jHalfHost, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jSpecOn, &g_jSpecHost, sizeof(int)));
    uploadJBinQ(scene.emitSampler);   // the camera walk's wavelength density (DJSpecRun::hole):
                                      // the device's draw (dSampleSceneLambda), not camSampler
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jTrSpec, &g_jTrSpecHost, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jChroma, &g_jChromaHost, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jDeltaMerge, &g_jDeltaMergeHost, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jDispWin, &g_jDispWinHost, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jDispAll, &g_jDispAllHost, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_routeOn, &g_routeOn, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_routeDefault, &g_routeDefault, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_routeAnyMedJ, &g_routeAnyMedJ, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_routeSppD, &g_routeSppD, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_routeSppJ, &g_routeSppJ, sizeof(double)));
    {   // FTRACE_ONLY_ST="s,t" (see c_dbgOnlyS)
        int os = -1, ot = -1;
        if (const char* e = std::getenv("FTRACE_ONLY_ST")) std::sscanf(e, "%d,%d", &os, &ot);
        CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_dbgOnlyS, &os, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_dbgOnlyT, &ot, sizeof(int)));
        if (os >= 0 || ot >= 0) std::printf("[diag] FTRACE_ONLY_ST: strategy s=%d t=%d only\n", os, ot);
    }
    const bool mergeOn = dbm.mergeOn();
    // WAVEFRONT gather queue (UPBP-CONV): sized for one wave of camera paths; the chunk is run
    // as consecutive waves. FTRACE_NOWAVEFRONT=1 forces the inline gather (the A/B control).
    DWfQueue wq{};
    int* d_wfCounters = nullptr;
    long long waveSamples = 0;
    if (dbm.gathers() &&
        !(std::getenv("FTRACE_NOWAVEFRONT") && std::atoi(std::getenv("FTRACE_NOWAVEFRONT")) != 0)) {
        const int segCap = kWfSegCap;        // 424 B each (~445 MB), 128 of them JSPEC's
        int hitCap = kWfHitCap;              // 16 B each  (512 MB): a dense map yields thousands of
                                             // candidates per segment (_fog_cornell: ~2000), and what
                                             // does not fit costs a resume pass (kWfBeamHits)
        // FTRACE_WFHITCAP=<n> (diagnostic) shrinks the hit queue, so an ordinary scene takes
        // resume passes: tools/check_wf_resume.py renders with and without it and requires the
        // same image.
        if (const char* e = std::getenv("FTRACE_WFHITCAP")) {
            const int v = std::atoi(e);
            if (v >= 256 && v < hitCap) hitCap = v;
        }
        CUDA_CHECK(cudaMalloc(&wq.segs, (size_t)segCap * sizeof(DWfSeg)));
        CUDA_CHECK(cudaMalloc(&wq.hits, (size_t)hitCap * sizeof(DWfHit)));
        CUDA_CHECK(cudaMalloc(&wq.resume, (size_t)segCap * sizeof(int)));
        // ...and, where the merge weights read their camera half at the beam's wavelength
        // (UPBP-W-CHROMATIC), that half's two bin tables per slot: 384 B each (~400 MB). Only
        // with room to spare: without the table a chromatic segment is gathered inline in the
        // camera kernel (the queue test there), which is slower but the same estimator, where a
        // failed or paged-out allocation would abort the render or crawl it.
        if (g_jTrSpecHost && g_jChromaHost) {
            const size_t want = (size_t)segCap * 2 * DJSPEC_K * sizeof(double);
            size_t freeB = 0, totalB = 0;
            const bool room = cudaMemGetInfo(&freeB, &totalB) == cudaSuccess &&
                              freeB > want + ((size_t)512 << 20);
            if (!room || cudaMalloc(&wq.chroma, want) != cudaSuccess) {
                (void)cudaGetLastError();   // a failed cudaMalloc leaves its error behind
                wq.chroma = nullptr;
                static bool said = false;
                if (!said) {
                    said = true;
                    std::printf("[jchroma] no room on the device for the wavefront queue's chroma "
                                "table (%.0f MB): chromatic segments are gathered inline -- "
                                "slower, the same estimator\n", want / 1048576.0);
                }
            }
        }
        CUDA_CHECK(cudaMalloc(&d_wfCounters, 4 * sizeof(int)));
        wq.nSegs = d_wfCounters; wq.nHits = d_wfCounters + 1; wq.overflow = d_wfCounters + 2;
        wq.segCap = segCap; wq.hitCap = hitCap;
        // The wave CEILING is one queued segment per path, not the worst case. It was
        // segCap / (maxDepth + 3) -- room for every path of the wave to queue its deepest walk --
        // which at `max_bounce 32` is 29 959 paths: fewer than the GPU holds resident, and each
        // wave then waited on its single longest path (a 32-bounce walk through a cloud) with the
        // rest of the device idle. gallery_rain's paths queue ~4 segments in a band with media
        // and exactly 1 in a plain one, so that ceiling ran its camera walk 3x slower than the
        // same kernel launched once per chunk. The waves are sized from the MEASURED segments
        // and hits per path (sizeFor, and the previous chunk's per-band profile), and a wave
        // that outruns its estimate loses nothing: segments that do not fit are gathered inline,
        // hits that do not fit wait for a resume pass.
        waveSamples = (long long)segCap;
        (void)maxDepth;
        // wave order (see DWfQueue): bands by default; FTRACE_WFSTRAT=1 = stratified, 32-pixel
        // runs on a power-of-two domain permuted by an odd golden-ratio multiplier under the mask
        wq.runsPerRow = ((unsigned)resX + 31u) / 32u;
        wq.nRuns = wq.runsPerRow * (unsigned)resY;
        {
            unsigned dom = 1u;
            while (dom < wq.nRuns) dom <<= 1;
            wq.runMask = dom - 1u;
            wq.runMul = wfStratOn() ? ((unsigned)(0.6180339887498949 * (double)dom) | 1u) : 0u;
        }
    }
    // THE POINT MERGES' QUEUE (DSurfQueue, SURF-WAVEFRONT): wherever a surface map gathers.
    // FTRACE_NOSURFWF=1 gathers them inline in the camera kernel (the A/B control). Up to
    // kSurfReqCap requests (surfReqBytes each), or what an eighth of the memory left beside the
    // camera kernel's local memory holds. Its waves are sized by surfWave, below; a site past the
    // requests is gathered inline.
    DSurfQueue sq{};
    int* d_sqCounters = nullptr;
    cudaEvent_t sqEv[2] = {nullptr, nullptr};
    bool sqShrunk = false;   // FTRACE_SURFWFCAP took effect
    if (dsm.kappaS > 0.0 && surfGatherQueued()) {
        size_t freeB = 0, totalB = 0;
        (void)cudaMemGetInfo(&freeB, &totalB);
        // The camera kernel's first launch reserves its local memory after this, unless an
        // earlier launch's reservation already holds it.
        const size_t local = cudaRouteLocalBytes(cudaBdptRoute(scene, cam.hasLens(), maxDepth, 1, true,
                                                               wq.segs != nullptr, true));
        const size_t pending = pendingLocalBytes(local);
        const size_t room = (freeB > pending) ? freeB - pending : 0;
        // FTRACE_SURFWFCAP=<n> (diagnostic) shrinks the queue to n requests (at least 16), so a
        // small frame takes several waves, or with a tiny n turns sites away every wave:
        // tools/check_surf_wavefront.py requires the waves' sizes and the same image.
        const long long cap = mergeQueueCap(room, surfReqBytes(), kSurfReqCap, "FTRACE_SURFWFCAP", sqShrunk);
        if (cap > 0) {
            sq.cap = (int)cap;
            CUDA_CHECK(cudaMalloc(&sq.reqs, (size_t)cap * sizeof(DSurfReq)));
            if (g_jSpecHost) CUDA_CHECK(cudaMalloc(&sq.spec, (size_t)cap * sizeof(DJSpec)));
            if (g_jTrSpecHost && g_jChromaHost)
                CUDA_CHECK(cudaMalloc(&sq.chroma, (size_t)cap * 2 * DJSPEC_K * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_sqCounters, 2 * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_sqCounters, 0, 2 * sizeof(int)));
            sq.nReqs = d_sqCounters; sq.overflow = d_sqCounters + 1;
            for (auto& e : sqEv) CUDA_CHECK(cudaEventCreate(&e));
        } else {
            static bool said = false;
            if (!said) {
                said = true;
                std::printf("[gpu] no room on the device for mode J's point-merge queue (%.0f MB "
                            "free beside the camera kernel): the surface merges are gathered "
                            "inline -- slower, the same estimator\n", room / 1048576.0);
            }
        }
    }
    long long sqSites = 0, sqPaths = 0;
    double sqMsGather = 0.0;
    long long wfWaves = 0, wfInlineSegs = 0, wfResumes = 0, wfDeferred = 0, wfLastHits = 0;
    int wfLastSegs = 0;
    // Where a mode-J render's seconds go: CUDA-event time of the three wave kernels and the wall
    // time around each wave (the wave already syncs on the counter copy, so the elapsed-time
    // queries are free). Reported with the queue use at the end of the render.
    cudaEvent_t wfEv[4] = {nullptr, nullptr, nullptr, nullptr};
    double wfMsWalk = 0.0, wfMsHits = 0.0, wfMsEval = 0.0, wfMsWall = 0.0;
    if (wq.segs) for (auto& e : wfEv) CUDA_CHECK(cudaEventCreate(&e));
    // The PROFILE: what every band of the previous chunk produced, in hits, segments and
    // point-merge sites per path, keyed by pixel slot so chunks of different spp line up. A stop
    // that cut the chunk short leaves it as it was. Stationary across chunks
    // (same map, same camera), so it is the right estimate for a wave about to cover those
    // bands -- including the top band a new chunk starts on, and the doubling at the light's
    // edge that the last wave cannot see coming (_fog_cornell spilled 94.6 M hits at 256^2 /
    // 60 s sized from the last wave alone, with the chunk's densest wave carried across).
    // It lives ACROSS calls: the progressive loop re-enters this function once per epoch
    // (~17 times a minute on _fog_thick), and an epoch whose first chunk had to relearn from a
    // 1024-path wave both spilled on a dense map and, through the epoch rule that scales the
    // epoch by its own setup time, ran shorter epochs -- 3 % fewer paths a minute against the
    // 0.261.0 schedule, all of it in the extra map rebuilds. Reset when the frame changes shape.
    if (g_wfProfileResX != resX || g_wfProfileResY != resY || g_wfProfileDepth != maxDepth) {
        g_wfProfile.clear();
        g_wfProfileResX = resX; g_wfProfileResY = resY; g_wfProfileDepth = maxDepth;
    }
    const bool mergeAny = mergeOn || dsm.kappaS > 0.0;   // either kind wants MERGE=true
    std::vector<WfBand> wfProfileCur;
    // Whether the PROFILE measured a queue's field: a chunk's bands share their call's queues,
    // so the first band speaks for all of them.
    auto beamProfiled = [&] { return !g_wfProfile.empty() && g_wfProfile.front().segsPerPath >= 0.0; };
    auto surfProfiled = [&] { return !g_wfProfile.empty() && g_wfProfile.front().sitesPerPath >= 0.0; };
    static const bool waveDebug = std::getenv("FTRACE_WAVE_DEBUG") != nullptr;   // a line a wave
    // The fewest paths a point-merge wave holds: the camera kernel's resident threads, below
    // which a wave idles it, and that costs more than gathering the excess inline. Under
    // FTRACE_SURFWFCAP (diagnostic) 256, so a small frame's waves are sized by the bands.
    long long surfLeast = 1024;
    if (sq.reqs && sqShrunk) {
        surfLeast = 256;
    } else if (sq.reqs) {
        static long long resident[2] = {-1, -1};   // kBdptT's, at BDPT_MAXDEPTH and DEEPDEPTH
        long long& t = resident[deep ? 1 : 0];
        if (t < 0)
            t = residentThreadsOf((const void*)bdptKernelFor(pick, true, false));
        surfLeast = std::max<long long>(surfLeast, t);
    }
    // A point-merge wave's paths from path `from` on, at most `cand` (DSurfQueue): the largest
    // n whose sites fit the requests, n * S(n) <= cap, where S(n) is the most sites a path, a
    // quarter high, of the last wave (`sqLast`) and of the PROFILE's bands within
    // [from, from + n) -- found by walking the bands in order, so a sparse wave is sized by its
    // own bands and not by a dense one further on, and the last wave cannot hide the next band
    // (sky rows above a room). Without a profile, at least 2 sites a path: a wave of sky must not
    // size the room below it. Never fewer than surfLeast paths, and where `cand` takes several
    // waves, waves of one size.
    double sqLast = -1.0;   // the last wave's sites a path, a quarter high; < 0 before one
    auto surfWave = [&](long long from, long long cand, long long c) -> long long {
        if (cand <= surfLeast) return cand;
        const double cap = (double)sq.cap;
        double s = std::max(sqLast, 0.0);
        long long n = cand;
        bool walked = false;
        if (!surfProfiled()) {
            s = std::max(s, 2.5);
        } else {
            for (const WfBand& w : g_wfProfile) {   // in path order
                const long long o0 = w.px0 * c - from, o1 = w.px1 * c - from;   // from `from` on
                if (o1 <= 0) continue;
                if (o0 >= n) break;
                walked = true;
                const double s2 = std::max(s, 1.25 * w.sitesPerPath);
                const long long lim = (long long)(cap / std::max(s2, 0.05));
                // The wave ends in this band, or at its start if it cannot enter it.
                if (lim < o1) { n = std::min(n, std::max(o0, lim)); break; }
                s = s2;
            }
        }
        if (!walked) n = std::min(n, (long long)(cap / std::max(s, 0.05)));
        n = std::max(n, surfLeast);
        if (n >= cand) return cand;
        // Waves of equal size: a split that left a remainder smaller than the rest would run it
        // on a part of the device, through the longest path's tail. Smaller, so it still fits.
        const long long k = (cand + n - 1) / n;
        return std::max(surfLeast, (cand + k - 1) / k);
    };
    // Gathers the sites a wave of `paths` paths queued, after kBdptT. Returns the sites a path
    // queued, those the full queue turned away included (`demand` of them in all); `gathered`
    // gets the sites gathered.
    auto surfGather = [&](long long paths, long long& gathered, long long& demand) -> double {
        int nq = 0;
        CUDA_CHECK(cudaMemcpy(&nq, sq.nReqs, sizeof(int), cudaMemcpyDeviceToHost));
        // Counts the sites turned away too, so it can wrap past 2^31 of them.
        demand = (nq < 0) ? (1LL << 31) : (long long)nq;
        gathered = std::min<long long>(demand, sq.cap);
        if (gathered > 0) {
            const int gBlocks = (int)std::min(4096LL, (gathered * 32 + 127) / 128);   // a warp a site
            CUDA_CHECK(cudaEventRecord(sqEv[0]));
            kWfSurfGather<<<gBlocks, 128>>>(up.sc, dsm, sq, (int)gathered, d_cam);
            CUDA_CHECK(cudaEventRecord(sqEv[1]));
            cudaCheckKernel("wf-surf-gather");
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, sqEv[0], sqEv[1]));
            sqMsGather += ms;
        }
        sqSites += gathered; sqPaths += paths;
        const double perPath = (paths > 0) ? (double)demand / (double)paths : 0.0;
        sqLast = 1.25 * perPath;
        return perPath;
    };
    // Returns whether the chunk landed whole (it is folded into the totals) or a stop cut it
    // short (dropped).
    const long long callBase = sppBase;   // the call's first sample, where its blocks count from
    auto launch = [&](long long cAll, long long baseAll) -> bool {
      if (abortable) DeviceStopScope::clearSeen();
      // Whether a stop made this chunk skip work: a loop below that left with some to do. With
      // the kernels' own record (seen) it is all that drops a chunk -- not the flag itself, which
      // -time's cut can raise after the last wave has already finished.
      bool leftEarly = false;
      // ONE SURFACE REALIZATION A BLOCK OF SAMPLES (jPerChunk; surfBlock above): the chunk is
      // split at the blocks' edges, each sub-launch against its block's map, salted by the
      // block's first ABSOLUTE sample -- which advances block by block and never repeats across
      // a resume or an epoch, as the camera side's samples do not.
      //
      // WHY THE SPLIT IS INSIDE `launch` RATHER THAN A SMALLER CHUNK. Measured: at
      // FTRACE_CHUNK_SPP=1 the render manages 665 / 713 / 672 spp in 20 s at JDEVLIGHT 0 / 1 / 3
      // -- indistinguishable -- so the rebuild is free and the ~30 ms per chunk is the generic
      // chunk machinery: launch, film download, progress report, film merge. Shrinking the chunk
      // pays all four per realization; splitting it pays only the launch and the rebuild.
      //
      // Each sub-launch is an independent, correctly-normalised estimate against its own valid
      // map, accumulating into the same device film exactly as consecutive chunks already do.
      // `kappaS` is unchanged across them (same nPaths, same radius), so the MIS weights stay
      // consistent within and between sub-launches.
      // The call's first sample, a sub-launch of its own, prices the block (surfBlock above):
      // `priceW0`, the work before it, until the next sub-launch or the chunk's end settles it.
      long long priceW0 = -1;
      const auto settleBlock = [&] {
          if (priceW0 < 0) return;
          const double camPerSpp = (double)(workSoFar() - priceW0);
          const double b = camPerSpp > 0.0
                               ? std::ceil((double)surfLightUnits / (kSurfShare * camPerSpp)) : 1.0;
          surfBlock = std::max(1LL, (long long)std::min(b, 1e12));
          priceW0 = -1;
      };
      for (long long pos = baseAll, endAll = baseAll + cAll; pos < endAll; ) {
        if (stopped()) { leftEarly = true; break; }
        settleBlock();
        long long subEnd = endAll;
        if (jPerChunk && surfBlock == 0) {   // the call's first sample, on block 0's map
            subEnd = pos + 1;
            priceW0 = workSoFar();
        } else if (jPerChunk) {
            const long long blk = (pos - callBase) / surfBlock;
            subEnd = std::min(endAll, callBase + (blk + 1) * surfBlock);
            if (blk != surfDrawn) {
                const long long w0 = workSoFar();
                buildJSurfMapDevice(jdev, up.sc, up.dc, diffraction, maxDepth, jPaths, jRad, jCap,
                                    jSurfBlockSalt(callBase + blk * surfBlock), dsm);
                surfLightUnits = workSoFar() - w0;
                surfDrawn = blk;
            }
        }
        const long long c = subEnd - pos;
        const long long base = pos;
        pos = subEnd;

        long long totalSamples = (long long)npix * c;
        // Mode J's merges take the scalar (NS == 0) instantiation; where its maps come out empty
        // it renders as mode D, through the plain instantiation below, the hero bundle and all.
        // EITHER merge kind needs the MERGE=true instantiation, not just the beams: SEGN --
        // the size of the camera-side MIS sum arrays -- is `MERGE ? MAXV : 1`, so launching the
        // MERGE=false kernel for a media-free `-jsurf` render left those sums structurally
        // absent (always index 0, always zero). The surface merges then ran with a denominator
        // missing its whole camera-side term, i.e. weights too large: +15 % on _cornell_diffuse.
        if (mergeAny) {
            // ADAPTIVE WAVES: the first wave is sized by segments alone; every later wave is sized
            // so that the previous wave's observed hits-per-segment would fill ~70% of the hit
            // queue, and never more than the segment queue allows. One synchronous 4-int copy per
            // wave buys a schedule that rarely needs a resume pass on a stationary scene.
            // The FIRST wave is small (1024 paths): a dense map yields thousands of candidates
            // per segment, and a segment-sized first wave would take dozens of resume passes
            // before any measurement exists. The adaptive rule then grows the wave, up to the
            // segment-sized ceiling. Each wave advances by the range it actually
            // ran (b0 = b1) -- NOT by the wave size, which changes inside the loop.
            // A wave's size is an estimate of its hits from what is known: the last wave (bands
            // are neighbours), and from the second chunk on the previous chunk's PROFILE -- the
            // hits per path every band produced. The size is the largest that keeps the densest
            // band the wave would cover at ~70% of the hit queue; that is a fixed point (a
            // smaller wave reaches fewer bands), reached by shrinking from the last-wave size.
            // So a band gets the size its own density allows, and the two jumps the last wave
            // cannot see -- the chunk boundary (bottom band -> top band) and the light's edge
            // inside every chunk -- are simply entries in the profile. Under FTRACE_WFSTRAT=1 the index space is padded to whole runs of the power-of-two
            // domain, and the profile, keyed by slot, sees stationary waves -- harmless.
            const long long waveTotal = (wq.runMul != 0u) ? ((long long)wq.runMask + 1) * 32 * c : totalSamples;
            wfProfileCur.clear();
            auto sizeFor = [&](double hitsPerPath, double segsPerPath) -> long long {
                long long n = waveSamples;
                if (segsPerPath > 0.0) {
                    const double byHits = (hitsPerPath > 0.0) ? 0.7 * (double)wq.hitCap / hitsPerPath : (double)waveSamples;
                    const double bySegs = 0.9 * (double)wq.segCap / segsPerPath;
                    n = (long long)((byHits < bySegs) ? byHits : bySegs);
                }
                // Below a few hundred paths a wave leaves the device idle while its longest walk
                // finishes; a map dense enough to fill the queue with fewer (an unbounded density
                // field's ~10^5 hits a path, J-FIELD-SLOW) gets its remainder by resume passes.
                if (n < 256) n = 256;
                if (n > waveSamples) n = waveSamples;   // the segment-sized wave is the ceiling
                return n;
            };
            auto profiled = [&](long long from, long long cand, double hpp, double spp) -> long long {
                if (!beamProfiled()) return cand;
                for (int it = 0; it < 8; ++it) {
                    const long long p0 = from / c, p1 = (from + cand + c - 1) / c;   // slots the wave would cover
                    double h = hpp, sg = spp;
                    for (const WfBand& w : g_wfProfile)
                        if (w.px1 > p0 && w.px0 < p1) {
                            if (w.hitsPerPath > h) h = w.hitsPerPath;
                            if (w.segsPerPath > sg) sg = w.segsPerPath;
                        }
                    const long long n = sizeFor(h, sg);
                    if (n >= cand) return cand;
                    cand = n;
                }
                return cand;
            };
            long long wave = !wq.segs ? waveTotal
                           : profiled(0, !beamProfiled() ? ((waveSamples < 1024) ? waveSamples : 1024) : waveSamples, 0.0, 0.0);
            for (long long b0 = 0; b0 < waveTotal; ) {
                if (stopped()) { leftEarly = true; break; }
                // No more paths than the point merges' queue holds (surfWave). A clamp of this
                // wave only: `wave` stays the beam queue's size.
                const long long waveNow =
                    sq.reqs ? surfWave(b0, std::min(wave, waveTotal - b0), c) : wave;
                const long long b1 = (b0 + waveNow < waveTotal) ? b0 + waveNow : waveTotal;
                // A REALIZATION PER BAND (FTRACE_JBAND). Salted by the band's first path slot as
                // well as by `base`, so two bands of one chunk draw different maps and two
                // chunks never repeat a band's map. `dsm` is passed BY VALUE to the kernel
                // below, so rebuilding here is all that is needed for the launch to see it.
                if (jPerBand)
                    buildJSurfMapDevice(jdev, up.sc, up.dc, diffraction, maxDepth, jPaths, jRad,
                                        jCap,
                                        jSurfSalt(prog)
                                            ^ ((unsigned long long)base * 0xD1B54A32D192ED03ULL)
                                            ^ ((unsigned long long)(b0 + 1) *
                                               0x9E3779B97F4A7C15ULL),
                                        dsm);
                const auto wfT0 = std::chrono::steady_clock::now();
                if (wq.segs) {
                    CUDA_CHECK(cudaMemsetAsync(d_wfCounters, 0, 4 * sizeof(int)));
                    CUDA_CHECK(cudaEventRecord(wfEv[0]));
                }
                if (sq.reqs) CUDA_CHECK(cudaMemsetAsync(sq.nReqs, 0, sizeof(int)));
                bdptKernelFor(pick, true, wq.runMul != 0u)<<<2048, 128>>>(
                    up.sc, up.dc, d_cam, d_splat, totalSamples, c, sppTotal, base, resX, maxDepth,
                    diffraction ? 1 : 0, seed, 1, dbm, dsm, wq, b0, b1, sq);
                if (wq.segs) {
                    cudaCheckKernel("bdpt");
                    CUDA_CHECK(cudaEventRecord(wfEv[1]));
                    // Queue and evaluate the wave's hits, pass after pass while the hit queue turns
                    // segments away (kWfBeamHits). Every pass queues at least one hit, so the
                    // passes end; a stop ends them sooner, the chunk being dropped.
                    int cnt[4] = {0, 0, 0, 0};
                    long long waveHits = 0;
                    int passes = 0;
                    float msWalk = 0.f, msHits = 0.f, msEval = 0.f;
                    for (;;) {
                        kWfBeamHits<<<2048, 128>>>(up.sc, dbm, wq, passes);
                        cudaCheckKernel("wf-hits");
                        CUDA_CHECK(cudaEventRecord(wfEv[2]));
                        kWfBeamEval<<<4096, 128>>>(up.sc, dbm, wq, d_cam, resX, up.dc);
                        cudaCheckKernel("wf-eval");
                        CUDA_CHECK(cudaEventRecord(wfEv[3]));
                        CUDA_CHECK(cudaMemcpy(cnt, d_wfCounters, sizeof cnt, cudaMemcpyDeviceToHost));
                        float ms = 0.f;
                        if (passes == 0) { CUDA_CHECK(cudaEventElapsedTime(&ms, wfEv[0], wfEv[1])); msWalk = ms; }
                        CUDA_CHECK(cudaEventElapsedTime(&ms, wfEv[1], wfEv[2])); msHits += ms;
                        CUDA_CHECK(cudaEventElapsedTime(&ms, wfEv[2], wfEv[3])); msEval += ms;
                        waveHits += (cnt[1] < wq.hitCap) ? cnt[1] : wq.hitCap;
                        ++passes;
                        if (cnt[3] == 0) break;
                        if (stopped()) { leftEarly = true; break; }
                        wfDeferred += cnt[3];
                        ++wfResumes;
                        CUDA_CHECK(cudaMemsetAsync(wq.nHits, 0, sizeof(int)));       // an empty hit queue
                        CUDA_CHECK(cudaMemsetAsync(wq.overflow + 1, 0, sizeof(int))); // and a fresh count
                        CUDA_CHECK(cudaEventRecord(wfEv[1]));
                    }
                    // The wave's point merges, while its time is still the wave's.
                    const double sqMs0 = sqMsGather;
                    long long sites = 0, demand = 0;
                    const double sitesPerPath =
                        (sq.reqs && !leftEarly) ? surfGather(b1 - b0, sites, demand) : -1.0;
                    ++wfWaves;
                    wfMsWalk += msWalk; wfMsHits += msHits; wfMsEval += msEval;
                    const double wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wfT0).count();
                    wfMsWall += wallMs;
                    if (waveDebug)
                        std::fprintf(stderr, "[wave] c=%lld paths %lld..%lld (%lld) segs %d (%d inline) hits %lld "
                                     "in %d pass%s, %lld point-merge sites (%lld turned away); walk %.2f "
                                     "hits %.2f eval %.2f point merges %.2f wall %.2f ms\n",
                                     c, b0, b1, b1 - b0, cnt[0], cnt[2], waveHits, passes,
                                     passes == 1 ? "" : "es", sites, demand - sites, msWalk, msHits,
                                     msEval, sqMsGather - sqMs0, wallMs);
                    wfInlineSegs += cnt[2];
                    wfLastSegs = (cnt[0] < wq.segCap) ? cnt[0] : wq.segCap;
                    wfLastHits = waveHits;
                    wfHitsCall += waveHits;   // each hit is a beam weighed (g_dWork's count)
                    const long long paths = b1 - b0;
                    const double segsPerPath = (paths > 0) ? (double)cnt[0] / (double)paths : (double)(maxDepth + 3);
                    // per QUEUED segment: those gathered inline (cnt[2]) left no hits in the queue
                    const double hitsPerSeg  = (wfLastSegs > 0) ? (double)waveHits / (double)wfLastSegs : 0.0;
                    const double hitsPerPath = hitsPerSeg * segsPerPath;
                    wfProfileCur.push_back(WfBand{b0 / c, (b1 + c - 1) / c, hitsPerPath, segsPerPath, sitesPerPath});
                    wave = profiled(b1, sizeFor(hitsPerPath, segsPerPath), hitsPerPath, segsPerPath);
                } else if (sq.reqs) {
                    cudaCheckKernel("bdpt");
                    long long sites = 0, demand = 0;
                    const double sitesPerPath = surfGather(b1 - b0, sites, demand);
                    wfProfileCur.push_back(WfBand{b0 / c, (b1 + c - 1) / c, -1.0, -1.0, sitesPerPath});
                    if (waveDebug)
                        std::fprintf(stderr, "[wave] c=%lld paths %lld..%lld (%lld), %lld point-merge sites "
                                     "(%lld turned away)\n", c, b0, b1, b1 - b0, sites, demand - sites);
                }
                b0 = b1;
            }
            // This chunk's bands size the next -- unless a stop cut its waves short, by leaving
            // some unlaunched or by ending a launched one's kernels early (the chunk is dropped
            // either way, and its bands are an under-count).
            if ((wq.segs || sq.reqs) && !wfProfileCur.empty() && !leftEarly &&
                !(abortable && DeviceStopScope::seen()))
                g_wfProfile.swap(wfProfileCur);
        }
        else
            bdptKernelFor(pick, false, false)<<<2048, 128>>>(
                up.sc, up.dc, d_cam, d_splat, totalSamples, c, sppTotal, base, resX, maxDepth,
                diffraction ? 1 : 0, seed, pick.hero ? C : 1, dbm, dsm, DWfQueue{}, 0, totalSamples, DSurfQueue{});
        cudaCheckKernel("bdpt");
      }
      if (!leftEarly) settleBlock();   // a chunk of one sample
      // Dropped if a kernel saw the stop (its own record, DeviceStopScope::seen) or the host
      // stopped launching waves.
      return !(abortable && (DeviceStopScope::seen() || leftEarly));
    };

    const auto tCall1 = std::chrono::steady_clock::now();
    if (!prog || !prog->report) { launch(spp, sppBase); download(out); }   // single-shot
    else gpuSppChunks(spp, *prog, out, launch, download);
    const auto tCall2 = std::chrono::steady_clock::now();

    freeUpload(up);
    if (wq.segs) {
        // Once per frame, not once per light-side epoch (see g_gpuQuietRebuild): one realization
        // per sample would otherwise print it 64 times a frame.
        if (!g_gpuQuietRebuild) std::fprintf(stderr, "[gpu] mode J wavefront gather: %lld waves (at most %lld paths, adaptive); "
                             "last wave %d segments / %lld hits; over the whole render: %lld segments "
                             "gathered inline (segment queue full), %lld resume passes for %lld segments "
                             "the hit queue turned away (FTRACE_NOWAVEFRONT=1 forces the inline path)\n"
                             "[gpu] mode J wave loop: GPU time walk %.1f s, beam hits %.1f s, hit eval %.1f s, "
                             "point merges %.1f s; wall %.1f s (host gaps %.1f s)\n",
                     wfWaves, waveSamples, wfLastSegs, wfLastHits, wfInlineSegs, wfResumes, wfDeferred,
                     wfMsWalk / 1e3, wfMsHits / 1e3, wfMsEval / 1e3, sqMsGather / 1e3, wfMsWall / 1e3,
                     (wfMsWall - wfMsWalk - wfMsHits - wfMsEval - sqMsGather) / 1e3);
        for (auto& e : wfEv) cudaEventDestroy(e);
        cudaFree(wq.segs); cudaFree(wq.hits); cudaFree(wq.resume); cudaFree(d_wfCounters);
        if (wq.chroma) cudaFree(wq.chroma);
    }
    if (sq.reqs) {
        int inl = 0;
        (void)cudaMemcpy(&inl, sq.overflow, sizeof(int), cudaMemcpyDeviceToHost);
        if (!g_gpuQuietRebuild && sqPaths > 0)
            std::fprintf(stderr, "[gpu] mode J point merges: %lld sites gathered a warp each "
                                 "(%.2f a path) in %.1f s of GPU time, %d inline with the queue "
                                 "full (FTRACE_NOSURFWF=1 gathers every one inline)\n",
                         sqSites, (double)sqSites / (double)sqPaths, sqMsGather / 1e3, inl);
        for (auto& e : sqEv) cudaEventDestroy(e);
        cudaFree(sq.reqs); cudaFree(d_sqCounters);
        if (sq.spec) cudaFree(sq.spec);
        if (sq.chroma) cudaFree(sq.chroma);
    }
    cudaFree(d_cam); cudaFree(d_splat);
    g_lastBdptWork = workSoFar();
    g_lastBdptOverheadSec =
        std::chrono::duration<double>((tCall1 - tCall0) +
                                      (std::chrono::steady_clock::now() - tCall2)).count();
    return out;
}

// ------------------------- device-memory preflight ---------------------------
//
// WHY THIS EXISTS: a megakernel's per-thread LOCAL memory is real device memory. The BDPT
// kernel carries its path vertices in thread-local arrays, and the driver backs them for
// every thread the card can hold RESIDENT at once (residentThreads below) before the kernel
// traces a single sample — on this project that is GIGABYTES, not megabytes. Resident, not
// the 2048 x 128 launch grid: the grid's 262144 threads run in waves of the card's resident
// capacity (196608 on an RTX 4090: 128 SMs x 1536), and the measured reservation is exactly
// localSizeBytes x resident threads (11.43 GB for gallery_rain's mode-J kernel at 62.4 KB a
// thread). The context then KEEPS that reservation after the kernel returns, which the
// caller's preflight has to know (main.cpp, runRender): a later render finds the card that
// much emptier, and the memory is its own.
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

// Threads the current device holds resident at once: what the driver sizes a kernel's local
// memory for. 0 if the device cannot be queried.
static size_t residentThreads() {
    static const size_t n = [] {
        int dev = 0;
        cudaDeviceProp p{};
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
            cudaGetLastError();
            return (size_t)0;
        }
        return (size_t)p.multiProcessorCount * (size_t)p.maxThreadsPerMultiProcessor;
    }();
    return n;
}

size_t cudaLocalReservedBytes() {
    if (!cudaAvailable()) return 0;
    size_t perThread = 0;
    if (cudaDeviceGetLimit(&perThread, cudaLimitStackSize) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return perThread * residentThreads();
}

bool cudaMemInfo(size_t* freeB, size_t* totalB) {
    if (!cudaAvailable()) return false;
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return false;
    if (freeB)  *freeB  = f;
    if (totalB) *totalB = t;
    return true;
}

// The local memory a launch of the most demanding of kernels `ks` reserves: its per-thread local size
// on every resident thread (0 for a kernel the runtime cannot describe; a null entry is skipped).
static size_t mostLocalOf(const std::vector<const void*>& ks) {
    if (!cudaAvailable()) return 0;
    size_t most = 0;
    for (const void* k : ks) {
        if (!k) continue;
        cudaFuncAttributes fa{};
        if (cudaFuncGetAttributes(&fa, k) == cudaSuccess) most = std::max(most, (size_t)fa.localSizeBytes);
        cudaGetLastError();
    }
    return most * residentThreads();
}

size_t cudaRouteLocalBytes(const GpuRoute& r) {
    return mostLocalOf(r.kernels);
}

bool cudaForwardRunsWavefront(const Scene& scene, bool wavefront, int heroC, bool beamGather, int nCam,
                              bool stableSeed) {
    const int effHeroC = forwardHeroC(heroC, !scene.media.empty(), grin::sceneHasGrin(scene));
    return forwardRunsWavefront(wavefront, effHeroC, sharedFlightGathersBeams(scene, beamGather, nCam),
                                stableSeed);
}

GpuRoute cudaForwardRoute(const Scene& scene, bool wavefront, int heroC, bool beamGather, int nCam,
                          bool stableSeed) {
    using namespace gpu;
    if (cudaForwardRunsWavefront(scene, wavefront, heroC, beamGather, nCam, stableSeed))
        return GpuRoute{{(const void*)kWfInit, (const void*)kWfExtend, (const void*)kWfShade}};
    return GpuRoute{{(const void*)kTrace}};
}

GpuRoute cudaBackwardRoute(bool rgb) {
    using namespace gpu;
    return GpuRoute{{rgb ? (const void*)kBackwardRGB : (const void*)kBackward}};
}

GpuRoute cudaBdptRoute(const Scene& scene, bool lens, int maxDepth, int heroC, bool merging, bool beamQueue,
                       bool surfMerge) {
    using namespace gpu;
    const BdptPick p = bdptPick(heroC, !scene.media.empty(), grin::sceneHasGrin(scene), lens, maxDepth);
    // The plain instantiation: mode D's, and mode J's where its maps come out empty, which
    // renderBdptCuda then renders as mode D.
    GpuRoute r{{(const void*)bdptKernelFor(p, false, false)}};
    if (!merging) return r;
    // Mode J's merges: the merge instantiation -- and its stratified variant too where the beam queue
    // may take that order (the launcher falls back to the first wherever the queue is not built) --
    // with the queue's two kernels where the beams queue; -jsurf's light pass where the device runs
    // it, and its surface gather where it queues.
    r.kernels.push_back((const void*)bdptKernelFor(p, true, false));
    if (beamQueue && wfStratOn()) r.kernels.push_back((const void*)bdptKernelFor(p, true, true));
    if (beamQueue) {
        r.kernels.push_back((const void*)kWfBeamHits);
        r.kernels.push_back((const void*)kWfBeamEval);
    }
    if (surfMerge && cudaJSurfOnDevice())
        r.kernels.push_back(p.deep ? (const void*)kJSurfLightT<BDPT_DEEPDEPTH> : (const void*)kJSurfLightT<BDPT_MAXDEPTH>);
    if (surfMerge && surfGatherQueued()) r.kernels.push_back((const void*)kWfSurfGather);
    return r;
}

GpuRoute cudaPhotonMapRoute() {
    using namespace gpu;
    return GpuRoute{{(const void*)kTrace, (const void*)kGather}};
}

GpuRoute cudaSppmRoute() {
    using namespace gpu;
    return GpuRoute{{(const void*)kTrace, (const void*)kSppmVisiblePoint, (const void*)kSppmGather}};
}

// The hero bundle mode U's session carries for `heroC` asked: 1 to BDPT_NSEC + 1 wavelengths (its
// kernels' NS instantiation holds the secondaries).
static int vcmSessionHeroC(int heroC) {
    const int C = (heroC < 1) ? 1 : heroC;
    return (C > BDPT_NSEC + 1) ? BDPT_NSEC + 1 : C;
}

GpuRoute cudaVcmRoute(int heroC) {
    using namespace gpu;
    const bool hero = vcmSessionHeroC(heroC) > 1;
    return GpuRoute{{hero ? (const void*)kVcmLightT<BDPT_NSEC> : (const void*)kVcmLightT<0>,
                     hero ? (const void*)kVcmCameraT<BDPT_NSEC> : (const void*)kVcmCameraT<0>,
                     hero ? (const void*)kVcmMergeGather<BDPT_NSEC> : (const void*)kVcmMergeGather<0>}};
}

size_t cudaModeJBeamQueueBytes() {
    using namespace gpu;
    // Its chromatic table is left out: it is allocated only with room to spare.
    return (size_t)kWfSegCap * (sizeof(DWfSeg) + sizeof(int)) + (size_t)kWfHitCap * sizeof(DWfHit);
}

// --------------------- backward reference (mode R) host ----------------------

// The posed copies a blurred forward launch keeps of a moving set of `nCam` cameras (DCamSet::cam):
// one of each camera a thread, on the backend launchForward picks -- the wavefront's slots for an
// N-photon trace (`runsWavefront`, cudaForwardRunsWavefront's answer), or the megakernel's grid.
static size_t forwardPosedBytes(bool runsWavefront, long long N, int nCam) {
    const long long threads = runsWavefront ? (long long)wavefrontSlots(N) : (long long)kFwdBlockSize * kFwdNumBlocks;
    return (size_t)threads * (size_t)nCam * sizeof(gpu::DCamera);
}

size_t cudaForwardPosedCameraBytes(const Scene& scene, bool wavefront, int heroC, long long N, bool stableSeed) {
    return forwardPosedBytes(cudaForwardRunsWavefront(scene, wavefront, heroC, false, 1, stableSeed), N, 1);
}

size_t cudaForwardWavefrontBytes(const Scene& scene, bool wavefront, int heroC, long long N, bool beamGather,
                                 int nCam, bool stableSeed) {
    return cudaForwardRunsWavefront(scene, wavefront, heroC, beamGather, nCam, stableSeed) ? wavefrontPoolBytes(N) : 0;
}

size_t cudaSharedFlightPosedBytes(const Scene& scene, int nCam, bool wavefront, int heroC, bool beamGather) {
    // made once for the session, so for its largest batch: the wavefront's pool at its fullest
    return forwardPosedBytes(cudaForwardRunsWavefront(scene, wavefront, heroC, beamGather, nCam), 1ll << 40, nCam);
}

const char* cudaBlurUnsupported(const Scene& /*scene*/, const Camera& /*cam*/, char mode) {
    if (mode == 0 || !std::strchr("RWDABCSUJMP", mode))
        return "motion blur in this mode renders on the CPU so far (the device blurs every mode but V)";
    return nullptr;
}

bool cudaBackwardSupported(const Scene& scene, const Camera& cam) {
    // GPU mode R renders what the forward path can bake (cudaForwardSupported) with the
    // backward walk's own scope on top: participating media (homogeneous and heterogeneous,
    // a rainbow phase, GRIN through dGrinMarch, fires through dFireNee and the emission along
    // every segment), fluorescence (the bispectral adjoint), a constant or image environment
    // (bkNeeEnv / bkNeeEnvVolumeHero and the MIS'd escape), and every emitter NEE can sample:
    // area, sphere, cylinder, mesh, spot and sun. Collimated beams, which NEE cannot sample,
    // keep a scene on the CPU.
    if (!cudaForwardSupported(scene)) return false;
    for (size_t i = 0; i < scene.emitters.size(); ++i) {
        if ((int)i == scene.envIndex) continue;    // constant env: handled by bkNeeEnv, not area NEE
        const auto& em = scene.emitters[i];
        if (em.collimated) return false;                    // collimated beams: not NEE-samplable, CPU only
        if (em.shape == EmitterShape::Env) return false;    // stray (non-env) env-shape emitter: CPU
    }
    (void)cam;
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
                        int maxBounce, bool directOnly, int heroC, const WhittedOpts* whitted,
                        Film* specOut, const CudaSceneHold* held) {
    using namespace gpu;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    // The split's second film, allocated before any early return, so a caller always gets both
    // films at the render's size (empty of light when nothing rendered).
    if (specOut) { *specOut = Film{}; specOut->resX = resX; specOut->resY = resY; specOut->alloc(); }
    if (!cudaAvailable() || !cudaBackwardSupported(scene, cam)) return out;
    if (whitted && !cudaBackwardWhittedSupported(scene, cam, *whitted)) return out;

    DUpload up;
    sceneOrHeld(scene, held, up, "renderBackwardCuda");
    up.dc = bakeCamera(scene, cam, resX, resY, up);
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
    // Hero-wavelength bundle (`-heroc N`). Fall back to the single-λ estimator when the scene
    // needs a branch bkRadianceHero does not carry: a medium whose extinction varies with λ
    // (Scene::mediaKeepBundle), gradient-index bending, or a physical lens (whose per-λ
    // refraction would give each wavelength its own camera ray). Same gate as the CPU
    // BackwardRenderer::renderRows, so CPU and GPU mode R agree on when hero applies.
    const int effHeroC = (heroC > 1 && scene.mediaKeepBundle() && !up.sc.hasGrin && !cam.hasLens())
                             ? ((heroC > hero::kHeroMax) ? hero::kHeroMax : heroC) : 1;

    const size_t npix = (size_t)resX * resY;
    double* d_film = nullptr; CUDA_CHECK(cudaMalloc(&d_film, npix * 3 * sizeof(double)));
    double* d_hits = nullptr; CUDA_CHECK(cudaMalloc(&d_hits, npix * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_film, 0, npix * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_hits, 0, npix * sizeof(double)));
    // Mode P's blurred split: the specular-first samples' film (kBackward's filmSpec).
    double* d_filmSpec = nullptr;
    if (specOut) {
        CUDA_CHECK(cudaMalloc(&d_filmSpec, npix * 3 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_filmSpec, 0, npix * 3 * sizeof(double)));
    }
    // Resume (mode R / W disk resume). Fresh samples carry ABSOLUTE indices
    // [sppBase, sppBase + spp), so the kernel's stride must be the FINAL total, not just
    // this invocation's request — otherwise `pix * sppTotal + sampleBase` walks off the
    // end of its pixel's slot and aliases into the next pixel's seed range.
    const long long sppBase  = prog ? (long long)prog->sampleBase : 0;
    // A caller's fixed stride (SppProgress::sampleStride: mode P's batches) indexes every sample
    // absolutely, so its stream is the same however the render is split into calls.
    const bool fixedStride = prog && prog->sampleStride > 0;
    const long long sppTotal = fixedStride ? prog->sampleStride : sppBase + spp;
    // Mode W is a deterministic quadrature: its lattice is a function of the absolute
    // sample index alone, so a resumed frame must be BIT-IDENTICAL to an uninterrupted one
    // and the seed base has to stay put. The Monte-Carlo modes instead mix the loaded
    // sample count in, because there `pix * sppTotal + sampleBase` can (rarely) collide
    // with a stream the checkpointed samples already drew, and two correlated samples are
    // worse than a decorrelated stream -- but a fixed stride never collides, so it mixes nothing.
    const unsigned long long seed = 0x9e3779b97f4a7c15ULL
        ^ ((prog && !whitted && !fixedStride) ? (unsigned long long)prog->sampleBase * 0x9E3779B97F4A7C15ULL : 0ULL)
        ^ g_rngSalt;   // `-seed`; 0 by default, so XOR is a no-op

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
                                 diffraction ? 1 : 0, seed, effHeroC, d_filmSpec);
        cudaCheckKernel("backward");
    };

    if (!prog || !prog->report) { launch(spp, sppBase); download(out); }   // single-shot
    else gpuSppChunks(spp, *prog, out, launch, download);
    if (specOut) {
        CUDA_CHECK(cudaMemcpy(film.data(), d_filmSpec, film.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < npix; ++i)
            specOut->xyz[i] = Vec3(film[i * 3 + 0], film[i * 3 + 1], film[i * 3 + 2]);
        cudaFree(d_filmSpec);
    }

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
        if (m.iridescence) return false;                            // a rough thin film / stack, ditto
        if (m.reflectTex >= 0) return false;                        // textured albedo not baked to RGB
        if (m.reflectPat >= 0) return false;                        // pattern-modulated albedo, ditto
        if (m.transmitPat >= 0) return false;                       // pattern-modulated transmittance, ditto
        // emitPat is fine here: an emission pattern is an ACHROMATIC scalar, so it
        // commutes with the spectral->RGB bake — ep*integral(CIE*emitSpd) is exactly
        // integral(CIE*ep*emitSpd). bkNeeLightRGB / bkRadianceRGB apply it to rgbEmit.
        if (m.recBindingFor(REC_SLOT_REFLECT)) return false;        // record-driven reflectance
    }
    (void)cam;
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
        ^ (prog ? (unsigned long long)prog->sampleBase * 0x9E3779B97F4A7C15ULL : 0ULL)
        ^ g_rngSalt;   // `-seed`; 0 by default, so XOR is a no-op

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
    dSetLenScale(s->up.sc);
    const long long base = s->accum;
    const unsigned long long seed = 0x9e3779b97f4a7c15ULL
        ^ (unsigned long long)base * 0x9E3779B97F4A7C15ULL
        ^ g_rngSalt;   // `-seed`; 0 by default, so XOR is a no-op
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
                                          IsoPreviewTiming* timing,
                                          const raster::PreviewLight* light, bool showHidden) {
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

    // The scene's lights distilled for the preview. The caller passes the one it derived once
    // (deriveLight bakes an environment map, and in a scene a light probe -- far too much work
    // for a function called per frame); without one, derive a probe-less one here.
    raster::PreviewLight plLocal;
    if (!light) plLocal = raster::deriveLight(scene, /*withSpecular*/ false);
    const raster::PreviewLight& plHost = light ? *light : plLocal;
    std::vector<DPLight> hLights(plHost.lights.size());
    for (size_t i = 0; i < plHost.lights.size(); ++i) {
        const raster::PLight& s = plHost.lights[i];
        DPLight d;
        d.pos = DVec3(s.pos.x, s.pos.y, s.pos.z);
        d.dir = DVec3(s.dir.x, s.dir.y, s.dir.z);
        d.kind = s.kind;
        d.cosInner = s.cosInner; d.cosOuter = s.cosOuter;
        d.rgb = DVec3(s.rgb.x, s.rgb.y, s.rgb.z); d.size = s.size;
        hLights[i] = d;
    }
    DPreviewLight dpl;
    dpl.nLights = (int)hLights.size();
    dpl.lights  = hLights.empty() ? nullptr : (const DPLight*)up.keep(uploadVec(hLights));
    for (int k = 0; k < 9; ++k) dpl.sh[k] = DVec3(plHost.sh[k].x, plHost.sh[k].y, plHost.sh[k].z);
    dpl.fill = plHost.fill;

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
                                 d_accum, d_z, d_emis, W, H, bg, EMIS_BOOST, showHidden ? 0 : 1);
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
    // Hair: the device forward tracer deposits through it and, since 0.335.0, the gather
    // scatters through it too (dPhotonGather's D_HAIR case, the twin of photonGather's), so
    // a hair scene stays on the GPU. SPPM's gather has no hair case; its gate refuses hair.
    return true;
}

// ---- mode-M photon-map cache file (-savemap / -loadmap) -----------------------
// Moved to src/photonmap_io.h so the CPU mode-M path can use it too: none of it was
// ever CUDA, and living here made -savemap / -loadmap GPU-only by accident of
// placement rather than by design (and therefore silently unavailable exactly when
// -beams forced mode M onto the CPU). The header also persists the beam map.

// Build the view-independent photon map on the GPU (forward deposit pass) and gather every
// camera from it — the flythrough win, on the device. The deposit runs the megakernel
// forward tracer TWICE: a count-only pass to size the buffer exactly, then a fill pass
// (deterministic same-seed launch). The grid is built on the host (PhotonMap::build, the
// tested counting sort) and re-uploaded, then each camera is gathered by kGather. Films are
// SUMs over spp (writeFilm divides by spp), matching renderPhotonCamera / the backward path.
//
// With `beams` (the -beams volume pass) the SAME forward pass also deposits photon BEAMS —
// the chords photons cut through each medium — because -beams changes the transport and the
// two halves of the cache must therefore come from one trace. The crossings come back to the
// host, which owns the trim / radius / split / BVH (BeamPass::build), and the built map is
// uploaded once so every camera's gather picks up single scatter from it.

// How many of the staged deposits are CAUSTIC (an L.S+.D path; the kernel set DPhoton::caustic
// at deposit time). Counted on the DEVICE, in one cheap pass over a buffer that is already
// there, purely so the host can size its two output vectors EXACTLY before the download. The
// alternatives are both bad: sizing both to nDep doubles the host map's peak footprint at
// exactly the photon counts where it is already the binding constraint, and growing them
// chunk-by-chunk makes std::vector::resize recopy the whole (multi-GB) array once per chunk.
__global__ void kCountCaustic(const gpu::DPhoton* p, unsigned long long n,
                              unsigned long long* out) {
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long G = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long c = 0;
    for (unsigned long long i = g; i < n; i += G) if (p[i].caustic) ++c;
    if (c) atomicAdd(out, c);
}

std::vector<Film> renderPhotonMapSharedCuda(const Scene& scene, const std::vector<Camera>& cams,
                                            const std::vector<int>& resX, const std::vector<int>& resY,
                                            long long N, double radius, EnergyReport& eOut,
                                            bool diffraction, long long spp,
                                            const SppProgress* prog,
                                            const std::function<bool(int, const Film&, long long)>* onFrame,
                                            const char* mapLoad, const char* mapSave, int heroC,
                                            int fgRays, double autoK, BeamPass* beams,
                                            const StageProgress* stage, double causticK,
                                            const caim::AimMap* aim, long long nAimed,
                                            double causticAdaptK, PmRadiiPin* pin,
                                            const CudaSceneHold* held) {
    using namespace gpu;
    const auto tCall = std::chrono::steady_clock::now();
    int nc = (int)cams.size();
    std::vector<Film> out(nc);
    for (int c = 0; c < nc; ++c) { out[c].resX = resX[c]; out[c].resY = resY[c]; out[c].alloc(); }
    if (nc == 0 || !cudaAvailable() || !cudaPhotonMapSupported(scene)) return out;

    DUpload up;
    sceneOrHeld(scene, held, up, "renderPhotonMapSharedCuda");
    // A blurred moving scene's epoch (the host's refresh loop, under its PassTimeScope): the
    // deposit, the maps and the gather all at its one time.
    up.sc.mo.passU = hostPassU(scene);

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
    // The caustic half of the split (empty, and never uploaded, when causticK <= 0). Its
    // photons are a DISJOINT subset of the same deposit, so it shares pm's nEmitted and gets
    // its own — much smaller — adaptive radius. See the header comment on `causticK`.
    PhotonMap pmC;
    const bool causticsOn = (causticK > 0.0);

    // Bin the map, honouring the density-adaptive radius (autoK > 0). Say out loud what
    // radius it settled on: the one printed before the deposit is only a starting point and
    // a silently-different one would be baffling when comparing renders. The gather below
    // reads pm.radius, so nothing else needs to know which branch ran.
    auto buildMap = [&]() {
        // A refresh epoch re-bins at the radius the first pass settled on — see PmRadiiPin for
        // why re-adapting per epoch would not be variance reduction.
        if (pin && pin->radius > 0.0) { pm.build(pin->radius); return; }
        if (autoK <= 0.0) { pm.build(radius); if (pin) pin->radius = pm.radius; return; }
        double nProbe = 0.0, kTarget = 0.0;
        const double r = pm.buildAuto(radius, autoK, &nProbe, &kTarget);
        std::printf("[gpu] adaptive gather radius: %.4g -> %.4g (a typical gather saw %.0f "
                    "photons at the starting radius; target %.0f for %zu stored)\n",
                    radius, r, nProbe, kTarget, pm.photons.size());
        if (pin) pin->radius = pm.radius;
    };
    // Host twin: buildCausticMap in main.cpp. Always adaptive — a caustic map built at the
    // GLOBAL radius is the very thing the split exists to avoid — and short-circuited on an
    // empty map, because buildAuto's density probe on zero photons returns a meaningless
    // radius rather than an error.
    auto buildCaustic = [&]() {
        if (!causticsOn) return;
        pmC.nEmitted = pm.nEmitted;          // counts PATHS EMITTED, not photons stored
        // Refresh epoch: re-bin at the first pass's radius and carry its per-query target.
        // `kGather` is NOT derived from the radius by PhotonMap, so dropping it here would
        // silently return the caustic gather to a fixed radius on every epoch but the first.
        if (pin && pin->radiusC > 0.0) {
            pmC.build(pin->radiusC);
            pmC.kGather = pin->kGather;
            return;
        }
        if (pmC.photons.empty()) {
            if (!g_gpuQuietRebuild)
                std::printf("[gpu] caustic map: 0 photons (no L-S+-D path in this scene)\n");
            pmC.build(pm.radius);            // still bin it, so the empty gather is well-formed
            if (pin) pin->radiusC = pmC.radius;
            return;
        }
        // Probe from the radius the GLOBAL map settled on, not the command-line starting
        // point: buildAuto's answer is scale-free but its two-octave clamp is not, and the
        // caustic radius is only ever interesting BELOW the global one anyway.
        double nProbe = 0.0, kTarget = 0.0;
        const double r0 = pm.radius;
        const double r = pmC.buildAuto(r0, causticK, &nProbe, &kTarget, r0);
        std::printf("[gpu] caustic map: %zu photons, gather radius %.4g -> %.4g (probe saw "
                    "%.0f; target %.0f)\n", pmC.photons.size(), r0, r, nProbe, kTarget);
        // ...and then the radius just chosen becomes a MAXIMUM, with each gather tightening
        // to whatever holds `k` photons locally. See dPmAdaptiveRadius for why one radius per
        // map is not enough even after the split. Host twin: buildCausticMap in main.cpp.
        pmC.kGather = 0.0;
        if (causticAdaptK != 0.0) {
            const double k = (causticAdaptK > 0.0) ? causticAdaptK : kTarget;
            if (k > 0.0) {
                pmC.kGather = k;
                std::printf("[gpu] caustic map: per-query adaptive gather ON — target %.0f "
                            "photons, radius %.4g down to %.4g as density allows\n",
                            k, pmC.radius, pmC.radius / 256.0);
            }
        }
        // Stored flux per emitted path — the invariant the aimed pass (`-causticn`) must leave
        // alone while changing only the variance. Host twin: buildCausticMap in main.cpp, and
        // the two numbers are directly comparable, which is how CPU/GPU agreement is checked.
        long double flux = 0.0L;
        for (const Photon& p : pmC.photons) flux += (long double)p.power;
        if (pmC.nEmitted > 0)
            std::printf("[gpu] caustic map: stored flux/emitted = %.6Lg (%zu photons)\n",
                        flux / (long double)pmC.nEmitted, pmC.photons.size());
        if (pin) { pin->radiusC = pmC.radius; pin->kGather = pmC.kGather; }
    };

    // The photon-BEAM volume pass (-beams). Only live when the caller supplied a BeamPass AND
    // the scene actually has a medium to store crossings of: with no media every beam gate is
    // dead code, and switching it on would needlessly force the megakernel (the wavefront pool
    // carries no per-photon side RNG stream — see launchForward).
    BeamMap* bmap = (beams && beams->map && !scene.media.empty()) ? beams->map : nullptr;

    bool mapLoaded = false;
    if (mapLoad && *mapLoad) {
        bool beamsMissing = false;
        mapLoaded = loadPhotonMap(mapLoad, pm, eOut, photonMapGuard(scene, diffraction),
                                  bmap, &beamsMissing, causticsOn ? &pmC : nullptr);
        if (mapLoaded) {
            std::printf("[loadmap] %s: %zu photons from %lld emitted", mapLoad,
                        pm.photons.size(), (long long)pm.nEmitted);
            if (causticsOn) std::printf(" (+%zu caustic)", pmC.photons.size());
            if (bmap) std::printf(", %zu beams", bmap->beams.size());
            std::printf(" -- deposit skipped\n");
            if (bmap && beamsMissing && beams) beams->loadedMissing = true;
            buildMap();                         // (re)build the grid at the requested radius
            buildCaustic();
            if (bmap && beams->build) beams->build(*bmap);
        } else {
            std::fprintf(stderr, "[loadmap] falling back to a fresh deposit\n");
            if (bmap) { bmap->beams.clear(); bmap->nDeposited = 0; bmap->nEmitted = 0; }
        }
    }
    if (!mapLoaded) {
    // ---- forward deposit pass (count, then fill) ----
    unsigned long long* d_depCount = nullptr;
    CUDA_CHECK(cudaMalloc(&d_depCount, sizeof(unsigned long long)));
    double* d_energy = nullptr; CUDA_CHECK(cudaMalloc(&d_energy, 5 * sizeof(double)));

    // ---- aimed caustic emission (causticaim.h): upload the target spheres ----
    // Same gate as the CPU pass (tracePhotonPass): the aimed pass needs somewhere caustic to
    // deposit and something to aim at. Note the map is bound to the MAIN launch too — that is
    // not an oversight but the balance heuristic: the main pass has to measure its own samples'
    // aimed density to weight the caustic deposits the aimed pass is also making. Binding it
    // draws no extra randomness, so a main pass with nAimed == 0 (misRatio 0 => weight 1) is
    // bit-for-bit what it was before the aimed pass existed.
    const bool doAimed = causticsOn && aim && !aim->empty() && nAimed > 0 && N > 0;
    DAimTarget* d_aim = nullptr;
    int         nAimT = 0;
    if (doAimed) {
        std::vector<DAimTarget> ht(aim->targets.size());
        for (size_t i = 0; i < aim->targets.size(); ++i) {
            const caim::Target& t = aim->targets[i];
            ht[i].c = DVec3(t.c.x, t.c.y, t.c.z);
            ht[i].r = (Real)t.r;
        }
        if (cudaMalloc(&d_aim, ht.size() * sizeof(DAimTarget)) == cudaSuccess) {
            CUDA_CHECK(cudaMemcpy(d_aim, ht.data(), ht.size() * sizeof(DAimTarget),
                                  cudaMemcpyHostToDevice));
            nAimT = (int)ht.size();
        } else {
            cudaGetLastError(); d_aim = nullptr;
            std::fprintf(stderr, "[caustic aim] could not upload %zu target spheres; "
                                 "falling back to the un-aimed caustic map.\n", ht.size());
        }
    }
    // One place decides whether the aimed pass happens, so the main launch's MIS ratio and the
    // second launch can never disagree (a nonzero ratio with no second pass would darken the
    // caustics by exactly the weight it applied).
    const bool aimOn      = doAimed && (d_aim != nullptr);
    const double aimRatio = aimOn ? (double)nAimed / (double)N : 0.0;

    // Beam staging buffer, sized like the CPU's per-thread banks: 2x the -beamcount target so
    // the pass can overshoot before the one exact (unbiased) host-side trim to the target,
    // which is what makes -beamcount mean what it says. Clamped to a quarter of free VRAM so
    // it never competes with the photon buffer for the card.
    unsigned long long beamCap = 0;
    DBeamDep* d_beams = nullptr;
    unsigned long long* d_beamCount = nullptr;
    if (bmap) {
        CUDA_CHECK(cudaMalloc(&d_beamCount, sizeof(unsigned long long)));
        beamCap = (beams->target > 0) ? (unsigned long long)(2 * beams->target)
                                      : ((unsigned long long)N * 2 + (1ull << 20));
        { size_t freeB = 0, totalB = 0;
          if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
              unsigned long long fit = (unsigned long long)(freeB / 4 / sizeof(DBeamDep));
              if (beamCap > fit) beamCap = fit; } }
        if (beamCap < (1ull << 16)) beamCap = (1ull << 16);
        while (beamCap >= (1ull << 14) &&
               cudaMalloc(&d_beams, (size_t)beamCap * sizeof(DBeamDep)) != cudaSuccess) {
            cudaGetLastError(); d_beams = nullptr; beamCap >>= 1;   // halve until it fits
        }
        if (!d_beams) { beamCap = 0; cudaFree(d_beamCount); d_beamCount = nullptr; bmap = nullptr;
            std::fprintf(stderr, "[beams] could not allocate a device beam buffer; "
                                 "the volume will be invisible.\n"); }
    }

    // The deposit trace. Split into launches when — and only when — someone is listening.
    //
    // Why splitting is safe. (a) Photon power is ABSOLUTE: genPhoton hands each photon its
    // own beta and the density estimate normalises by pm.nEmitted, so the kernel's `N`
    // argument is a loop bound, not a divisor, and 8 launches of N/8 emit exactly the energy
    // one launch of N does. (b) The deposit cursor is an ATOMIC on the device that we zero
    // once, here, outside the loop — so successive launches append rather than overwrite.
    // (c) Each launch takes `seedBase` = the cumulative photon offset, which is 0 for the
    // first one, so an unsplit deposit is bit-identical to the historical single launch and a
    // split one merely draws a different (equally valid) realization per chunk. (d) The split
    // is a pure function of N, so the count/fill rerun above reproduces it deposit for
    // deposit — which is the invariant the rerun depends on.
    //
    // Why splitting at all. A showcase `-n` is billions of photons, i.e. one kernel launch
    // running for many minutes: nothing can be read back from it, so the window title froze
    // on "tracing photons…" for most of the render and the console said nothing at all. A
    // monolithic launch of that length is also exactly what the Windows TDR watchdog exists
    // to shoot at. Chunks are sized ADAPTIVELY from the measured rate of the previous one
    // (target ~1 s), because photons-per-second spans orders of magnitude across scenes and
    // any fixed chunk count would be far too coarse on one and pure launch overhead on
    // another. The first chunk is a deliberately small probe for the same reason.
    const bool splitDeposit = (stage && stage->report);
    // How many photons the last depositLaunch actually EMITTED. Normally N, but a `-stop`
    // between chunks ends the trace early, and pm.nEmitted normalises the density estimate —
    // so reporting the full N after a truncated pass would scale the map down by the fraction
    // never traced and render a darkened image. (Exactly the accounting tracePhotonPass does
    // with its per-thread `emitted[tid] = done`.)
    long long depEmitted = N;
    // The aimed pass's own emitted count. It deliberately never reaches pm.nEmitted (it emits
    // no light of its own — it re-estimates the main pass's caustic term with a different
    // sampler), but a `-stop` mid-way still has to be visible to the accounting below.
    long long aimEmitted = 0;
    // Energy for the aimed pass, allocated only if one runs. Kept SEPARATE from d_energy and
    // then thrown away: the aimed photons are not additional emitted light, so folding their
    // joules into eOut would double-count the emission and break the energy audit.
    double* d_energyAim = nullptr;
    auto depositLaunch = [&](DPhoton* buf, unsigned long long cap, double beamKeep,
                             bool aimedPass) {
        const long long Nq = aimedPass ? nAimed : N;
        double* const eBuf = aimedPass ? d_energyAim : d_energy;
        // Disjoint RNG stream. The aimed pass must not replay the main pass's photons under a
        // different sampler — that would correlate the two estimators the balance heuristic
        // assumes are independent. (Host twin: the per-pass `salt` in tracePhotonPass.)
        const unsigned long long seed0 = aimedPass ? 0x94D049BB13311000ULL : 0ULL;
        const char* const label = aimedPass ? "tracing aimed photons" : "tracing photons";
        CUDA_CHECK(cudaMemset(d_depCount, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(eBuf, 0, 5 * sizeof(double)));
        DCamSet cs{};                       // nCam == 0: every camera splat is a no-op
        cs.cams = nullptr; cs.films = nullptr; cs.hits = nullptr; cs.nCam = 0;
        cs.depPhotons = buf; cs.depCount = d_depCount; cs.depCap = cap;
        cs.beamOrderMax = pbeams::gOrderMax;   // -beams-order (host twin: Renderer::beamOrderMax)
        // Bound on BOTH passes; only `aimed` differs. See the upload above for why the main
        // pass needs it at all.
        cs.aim.targets = d_aim; cs.aim.n = nAimT;
        cs.aim.sumR2 = aim ? aim->sumR2 : 0.0;
        cs.aim.aimed = aimedPass; cs.aim.misRatio = aimRatio;
        if (aimedPass) {
            // Caustic-only: no beam records (the beam map is the main pass's and is normalised
            // by ITS nEmitted), but media must still be crossed straight or this pass would be
            // transporting by different rules than the pass it is MIS-combined with.
            cs.beamStraightOnly = (d_beamCount != nullptr);
        } else if (d_beamCount) {
            CUDA_CHECK(cudaMemset(d_beamCount, 0, sizeof(unsigned long long)));
            cs.beamOut = d_beams; cs.beamCount = d_beamCount;
            cs.beamCap = beamCap; cs.beamKeep = beamKeep;
            // SPECTRAL BEAMS. Same gate the CPU deposit uses (photonmap_render.h), asked here
            // rather than on the device so the device never has to walk the media spectra.
            cs.beamSpecC = (pbeams::gSpecC > 1 && beamSpectralOK(scene))
                               ? std::min(pbeams::gSpecC, kBeamSpecMax) : 1;
            // ACHROMATIC-PATH BEAMS: the same scene-wide extinction test, asked without the
            // `-beamspec` condition (see DCamSet::beamAchroOK).
            cs.beamAchroOK = pbeams::gAchro && beamSpectralOK(scene);
        }
        (aimedPass ? aimEmitted : depEmitted) = Nq;
        // Each photon's stream from its absolute index (stableSeed), so the chunks the clock cuts
        // below draw the photons one launch draws: one seed, one map, and a count-only pass and the
        // fill after it deposit the same photons.
        if (!splitDeposit) {
            launchForward(up, cs, eBuf, Nq, diffraction, seed0, /*wavefront*/false,
                          CAM_B, heroC, photonMaxBounce(), /*stableSeed*/true);
            return;
        }
        stage->report(label, 0, Nq, nullptr, 0.0);
        // 1 M is small enough to be a sub-second probe on any card that can run this at all,
        // and large enough to keep ~262 k threads busy rather than measuring launch latency.
        // FTRACE_DEPOSIT_CHUNK=K pins every chunk to K photons instead (determinism triage: one
        // seed must deposit one map whatever the chunks).
        static const long long forcedChunk = [] {
            const char* e = std::getenv("FTRACE_DEPOSIT_CHUNK");
            return (e && *e) ? std::atoll(e) : 0LL;
        }();
        long long chunk = forcedChunk > 0 ? forcedChunk : ((Nq < (1ll << 20)) ? Nq : (1ll << 20));
        for (long long off = 0; off < Nq; ) {
            const long long cs2 = (off + chunk <= Nq) ? chunk : (Nq - off);
            const auto t0 = std::chrono::steady_clock::now();
            launchForward(up, cs, eBuf, cs2, diffraction, seed0 + (unsigned long long)off,
                          /*wavefront*/false, CAM_B, heroC, photonMaxBounce(), /*stableSeed*/true);
            CUDA_CHECK(cudaDeviceSynchronize());   // the launch is async; time the WORK
            const double sec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            off += cs2;
            stage->report(label, off, Nq, nullptr, 0.0);
            // Cooperative stop: the deposit is the phase a `-stop` most often lands in, and
            // before the split there was no seam to honour it at.
            if (ft::stopRequested()) { (aimedPass ? aimEmitted : depEmitted) = off; break; }
            // Retarget ~1 s of work, but never shrink below the probe (a chunk that keeps
            // halving turns the deposit into launch overhead) and never grow more than 4x at
            // a step (one anomalously fast chunk must not produce a 10-minute next one).
            if (forcedChunk <= 0 && sec > 1e-4) {
                double want = (double)cs2 / sec;              // photons/s -> photons per 1 s
                if (want > 4.0 * (double)chunk) want = 4.0 * (double)chunk;
                chunk = (long long)want;
                if (chunk < (1ll << 20)) chunk = (1ll << 20);
            }
        }
    };
    auto readBeamCount = [&]() -> unsigned long long {
        if (!d_beamCount) return 0;
        unsigned long long n = 0;
        CUDA_CHECK(cudaMemcpy(&n, d_beamCount, sizeof n, cudaMemcpyDeviceToHost));
        return n;
    };

    // Single-pass deposit: the old flow ran the WHOLE forward trace twice (a count-only
    // sizing pass, then a fill pass). Instead, stage into a generously guessed buffer
    // (2.5 deposits/photon + slack, clamped to half of free VRAM) and only rerun if the
    // guess undershot — the atomic cursor keeps counting past the cap, so an overflow
    // still yields the exact total and the rerun costs no more than the old flow did.
    // Same seed every pass => identical deposits regardless of which path executes.
    pm.nEmitted = N;
    unsigned long long cap = (unsigned long long)((double)N * 2.5) + (1ull << 20);
    // Half the memory free once the walks' stacks are set aside: kTrace reserves them at its first
    // launch, after this, unless an earlier launch holds them (pendingLocalBytes).
    { size_t freeB = 0, totalB = 0;
      if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
          const size_t pending = pendingLocalBytes(cudaRouteLocalBytes(cudaPhotonMapRoute()));
          unsigned long long fit = (unsigned long long)((freeB > pending ? freeB - pending : 0) / 2 / sizeof(DPhoton));
          if (cap > fit) cap = fit; } }
    DPhoton* d_photons = nullptr;
    while (cap >= (1ull << 22) &&
           cudaMalloc(&d_photons, (size_t)cap * sizeof(DPhoton)) != cudaSuccess) {
        cudaGetLastError(); d_photons = nullptr; cap >>= 1;   // halve until it fits
    }
    unsigned long long nDep = 0, nBeam = 0;
    if (d_photons) {
        depositLaunch(d_photons, cap, 1.0, /*aimed*/false); // optimistic fill against the guess
        CUDA_CHECK(cudaMemcpy(&nDep, d_depCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        nBeam = readBeamCount();
        // ONE rerun settles both overflows, because the two are independent knobs on the same
        // trace: the photon buffer is resized to the now-exact count, and the beams get a
        // survival rate that fits them into theirs. Neither perturbs the transport RNG (the
        // beam roulette draws from the per-photon SIDE stream, see dEmitBeams), so the rerun
        // reproduces the surface deposit photon-for-photon whatever the beam rate is.
        const bool phOver = (nDep > cap);
        const bool bmOver = (d_beams && nBeam > beamCap);
        if (phOver || bmOver) {
            if (phOver) {
                cudaFree(d_photons); d_photons = nullptr;
                CUDA_CHECK(cudaMalloc(&d_photons, (size_t)nDep * sizeof(DPhoton)));
                cap = nDep;
            }
            // 0.95, not 1.0: the survivor count is Binomial(nBeam, keep), so aiming exactly at
            // the cap would overflow half the time — and an overflow past the cap is DROPPED
            // (unlike the photon count, which the atomic still reports exactly), which would
            // bias the estimate. 5% headroom is ~50 standard deviations at a 1 M cap.
            const double keep = bmOver ? 0.95 * (double)beamCap / (double)nBeam : 1.0;
            depositLaunch(d_photons, cap, keep, /*aimed*/false);
            unsigned long long nFill = 0;
            CUDA_CHECK(cudaMemcpy(&nFill, d_depCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            if (nFill < nDep) nDep = nFill;
            nBeam = readBeamCount();
        }
    } else {
        // Couldn't stage even a modest guess: fall back to the old two-pass flow.
        depositLaunch(nullptr, 0, 1.0, /*aimed*/false);   // count-only sizing pass
        CUDA_CHECK(cudaMemcpy(&nDep, d_depCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        nBeam = readBeamCount();
        if (nDep > 0) {
            CUDA_CHECK(cudaMalloc(&d_photons, (size_t)nDep * sizeof(DPhoton)));
            const double keep = (d_beams && nBeam > beamCap)
                              ? 0.95 * (double)beamCap / (double)nBeam : 1.0;
            depositLaunch(d_photons, nDep, keep, /*aimed*/false); // fill pass (same seed => same nDep deposits)
            nBeam = readBeamCount();
        }
    }
    // What the trace actually emitted, which is N unless a `-stop` cut a chunked deposit
    // short. Set AFTER the deposit (the optimistic `pm.nEmitted = N` above only pre-fills it
    // for the paths that never reach here), because the density estimate divides by it.
    pm.nEmitted = depEmitted;
    if (nDep > 0 && d_photons) {
        // Caustic count first, on the device, so both host vectors can be sized exactly (see
        // kCountCaustic). Zero when the split is off, which collapses the loop below to the
        // historical single-destination copy.
        unsigned long long nCau = 0;
        if (causticsOn) {
            unsigned long long* d_cc = nullptr;
            CUDA_CHECK(cudaMalloc(&d_cc, sizeof(unsigned long long)));
            CUDA_CHECK(cudaMemset(d_cc, 0, sizeof(unsigned long long)));
            kCountCaustic<<<1024, 256>>>(d_photons, nDep, d_cc);
            cudaCheckKernel("caustic-count");
            CUDA_CHECK(cudaMemcpy(&nCau, d_cc, sizeof nCau, cudaMemcpyDeviceToHost));
            cudaFree(d_cc);
            if (nCau > nDep) nCau = nDep;
        }
        // Download + convert in chunks (never a full host-side DPhoton copy). Positions
        // and payloads split into PhotonMap's two parallel arrays (see Photon in
        // photonmap.h); DPhoton has the same fields, so this is a pure widen + split.
        // The caustic flag additionally routes each record to one of TWO destination maps —
        // a strict partition, so the two counts sum to nDep and no photon is duplicated.
        ftalloc::resize(pm.photons, (size_t)(nDep - nCau), "the photon map payloads", "-n");
        ftalloc::resize(pm.pos, (size_t)(nDep - nCau), "the photon map positions", "-n");
        ftalloc::resize(pmC.photons, (size_t)nCau, "the caustic map payloads", "-n");
        ftalloc::resize(pmC.pos, (size_t)nCau, "the caustic map positions", "-n");
        size_t wg = 0, wc = 0;                  // independent write cursors
        std::vector<DPhoton> stage;
        for (size_t off = 0; off < (size_t)nDep; off += PM_CHUNK) {
            size_t cnt = std::min(PM_CHUNK, (size_t)nDep - off);
            stage.resize(cnt);
            CUDA_CHECK(cudaMemcpy(stage.data(), d_photons + off, cnt * sizeof(DPhoton),
                                  cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < cnt; ++i) {
                const DPhoton& d = stage[i];
                const bool caus = (nCau > 0) && d.caustic;
                PhotonMap& M = caus ? pmC : pm;
                size_t&    w = caus ? wc  : wg;
                if (w >= M.photons.size()) continue;   // can't happen; never write past the end
                Photon& p = M.photons[w];
                M.pos[w] = Vec3(d.pos.x, d.pos.y, d.pos.z);
                p.n   = Vec3(d.n.x,   d.n.y,   d.n.z);
                p.power = d.power; p.lambda = d.lambda;
                ++w;
            }
        }
    }
    if (d_photons) cudaFree(d_photons);
    d_photons = nullptr;

    // ---- aimed caustic pass (Jensen's projection map; causticaim.h) --------------------
    // A SECOND deposit launch whose emission is importance-sampled towards the focusing
    // geometry, appended to the caustic map only. Deliberately AFTER the main pass and its
    // download, for three reasons: the main pass's photon buffer is already freed so this one
    // gets the whole card; the two RNG streams stay disjoint (see `seed0`); and an
    // `ftrace -stop` during the main pass skips it entirely, leaving the un-aimed caustic map,
    // which is still correct — just noisier.
    //
    // Every record it produces is caustic by construction: device depositPhoton drops a
    // non-caustic deposit outright when cs.aim.aimed, so no partition is needed here.
    // pm.nEmitted / pmC.nEmitted stay at the MAIN pass's count — the aimed photons carry no
    // light of their own, they re-estimate the same caustic term with a different sampler, and
    // the balance-heuristic weight applied at emission is what makes the sum unbiased.
    if (aimOn && !ft::stopRequested()) {
        CUDA_CHECK(cudaMalloc(&d_energyAim, 5 * sizeof(double)));
        // Caustic deposits are a small fraction of all deposits, so the guess that sizes the
        // main buffer (2.5/photon) would be wildly over here. Start at the observed main-pass
        // caustic rate with generous slack, and let the same overflow rerun settle it exactly.
        double perPhoton = 0.25;
        if (depEmitted > 0 && pmC.photons.size() > 0)
            perPhoton = 4.0 * (double)pmC.photons.size() / (double)depEmitted;
        if (perPhoton < 0.05) perPhoton = 0.05;
        if (perPhoton > 2.5)  perPhoton = 2.5;
        unsigned long long acap =
            (unsigned long long)((double)nAimed * perPhoton) + (1ull << 20);
        { size_t freeB = 0, totalB = 0;
          if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
              unsigned long long fit = (unsigned long long)(freeB / 2 / sizeof(DPhoton));
              if (acap > fit) acap = fit; } }
        DPhoton* d_aphotons = nullptr;
        while (acap >= (1ull << 20) &&
               cudaMalloc(&d_aphotons, (size_t)acap * sizeof(DPhoton)) != cudaSuccess) {
            cudaGetLastError(); d_aphotons = nullptr; acap >>= 1;
        }
        unsigned long long nAimDep = 0;
        if (d_aphotons) {
            depositLaunch(d_aphotons, acap, 1.0, /*aimed*/true);
            CUDA_CHECK(cudaMemcpy(&nAimDep, d_depCount, sizeof nAimDep, cudaMemcpyDeviceToHost));
            if (nAimDep > acap) {                       // overflow: resize to the exact count
                cudaFree(d_aphotons); d_aphotons = nullptr;
                if (cudaMalloc(&d_aphotons, (size_t)nAimDep * sizeof(DPhoton)) == cudaSuccess) {
                    acap = nAimDep;
                    depositLaunch(d_aphotons, acap, 1.0, /*aimed*/true);
                    unsigned long long nFill = 0;
                    CUDA_CHECK(cudaMemcpy(&nFill, d_depCount, sizeof nFill,
                                          cudaMemcpyDeviceToHost));
                    if (nFill < nAimDep) nAimDep = nFill;
                } else {
                    // Couldn't hold them all: keep the prefix the first launch did store. That
                    // is a biased subset (it is whichever deposits won the atomic race), so
                    // say so rather than silently shipping it.
                    cudaGetLastError(); d_aphotons = nullptr; nAimDep = 0;
                    std::fprintf(stderr, "[caustic aim] the aimed pass overflowed its device "
                                         "buffer and could not be resized; dropping it. Lower "
                                         "-causticn.\n");
                }
            }
        } else {
            std::fprintf(stderr, "[caustic aim] could not allocate a device photon buffer for "
                                 "the aimed pass; falling back to the un-aimed caustic map.\n");
        }
        if (d_aphotons && nAimDep > 0) {
            const size_t base = pmC.photons.size();
            ftalloc::resize(pmC.photons, base + (size_t)nAimDep, "the caustic map payloads",
                            "-causticn");
            ftalloc::resize(pmC.pos, base + (size_t)nAimDep, "the caustic map positions",
                            "-causticn");
            std::vector<DPhoton> astage;
            size_t w = base;
            for (size_t off = 0; off < (size_t)nAimDep; off += PM_CHUNK) {
                size_t cnt = std::min(PM_CHUNK, (size_t)nAimDep - off);
                astage.resize(cnt);
                CUDA_CHECK(cudaMemcpy(astage.data(), d_aphotons + off, cnt * sizeof(DPhoton),
                                      cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < cnt; ++i) {
                    const DPhoton& d = astage[i];
                    Photon& p = pmC.photons[w];
                    pmC.pos[w] = Vec3(d.pos.x, d.pos.y, d.pos.z);
                    p.n = Vec3(d.n.x, d.n.y, d.n.z);
                    p.power = d.power; p.lambda = d.lambda;
                    ++w;
                }
            }
            if (!g_gpuQuietRebuild)
                std::printf("[gpu] aimed caustic pass: %lld photons -> %llu caustic deposits "
                            "(main pass: %zu)\n",
                            (long long)aimEmitted, (unsigned long long)nAimDep, base);
        }
        if (d_aphotons) cudaFree(d_aphotons);
        cudaFree(d_energyAim); d_energyAim = nullptr;   // discarded: not additional emission
    }
    if (d_aim) { cudaFree(d_aim); d_aim = nullptr; }

    // ---- download the deposited beams ----
    // Straight widen into PhotonBeam: `s0` is 0 for every downloaded record because a DEPOSIT
    // is always a WHOLE crossing — sub-segments are a product of the host-side split, which
    // needs the final kernel radius and therefore cannot run until after this (same reason
    // .pmap stores raw crossings; see photonbeams.h).
    if (bmap) {
        const unsigned long long stored = (nBeam < beamCap) ? nBeam : beamCap;
        if (nBeam > beamCap)
            std::fprintf(stderr, "[beams] warning: %llu crossings overflowed the %llu-record "
                                 "device buffer even after rescaling; keeping %llu. Lower "
                                 "-beamcount or free VRAM.\n",
                         (unsigned long long)nBeam, (unsigned long long)beamCap,
                         (unsigned long long)stored);
        bmap->beams.clear();
        ftalloc::resize(bmap->beams, (size_t)stored, "the photon-beam map",
                        "-beamcount (or -n, which feeds it)");
        std::vector<DBeamDep> bstage;
        for (size_t off = 0; off < (size_t)stored; off += PM_CHUNK) {
            size_t cnt = std::min(PM_CHUNK, (size_t)stored - off);
            bstage.resize(cnt);
            CUDA_CHECK(cudaMemcpy(bstage.data(), d_beams + off, cnt * sizeof(DBeamDep),
                                  cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < cnt; ++i) {
                const DBeamDep& d = bstage[i];
                PhotonBeam& b = bmap->beams[off + i];
                b.o = Vec3(d.o.x, d.o.y, d.o.z);
                b.d = Vec3(d.d.x, d.d.y, d.d.z);
                b.s0 = 0.0f; b.len = d.len; b.power = d.power;
                b.lambda = d.lambda; b.absorb = d.absorb; b.med = d.med;
                // Spectral bundle (`-beamspec`): 0 secondaries is the classic monochromatic
                // beam, which is what every non-qualifying scene deposits.
                b.nSec = (d.nSec < 0) ? 0 : (d.nSec > kBeamSecMax ? kBeamSecMax : d.nSec);
                for (int k = 0; k < kBeamSecMax; ++k) b.lamS[k] = (k < b.nSec) ? d.lamS[k] : 0.0f;
                // Per-member weights (0.257.0). The device tracer still retires the bundle at
                // the first wavelength-DEPENDENT event rather than reweighting it, so every
                // bundle it deposits is an equal-weight one and 1 is exact, not a placeholder.
                // (Weighted deposit on the device is FOLD-GPU's other half; see known-issues.md.)
                for (int k = 0; k < kBeamSecMax; ++k) b.wS[k] = (k < b.nSec) ? 1.0f : 0.0f;
                // Achromatic-path fold (`-beamachro`): the emitter's mean CIE, used by
                // BeamMap::build in place of CIE(lambda). Mutually exclusive with the bundle.
                b.achro = d.achro ? 1 : 0;
                // Scattering order (BEAMORDER-GPU). Clamped exactly as BeamBank::push does, so
                // a device-traced map and a host-traced one report the same histogram.
                b.order = (d.order < 0 || d.order >= (int)kBeamOrderUnknown)
                              ? kBeamOrderUnknown : (unsigned char)d.order;
                // `cieA` is the fallback colour for the GATHER-time fold as well as the payload
                // of the achromatic-path one, so it must survive a beam whose `achro` is about
                // to become 2 below -- otherwise BeamMap::build would give it a black cieMean.
                const bool wantCieA = d.achro || d.bowEm >= 0;
                for (int k = 0; k < 3; ++k) b.cieA[k] = wantCieA ? d.cieA[k] : 0.0f;
                // GATHER-time fold (achro == 2, scene.h Scene::BowLut). Since 0.264.1 the device
                // tracer decides this exactly as render.h does, so a GPU-traced map carries it
                // too -- which is what makes the device bow tables worth having on the common
                // path rather than only for a `-loadmap`'d CPU map.
                if (d.bowEm >= 0) { b.achro = 2; b.emIdx = (short)d.bowEm; }
                else                b.emIdx = -1;
                // Chord provenance (0.312.0): clamped exactly as BeamBank::push does.
                b.srcEm = (d.srcEm >= 0 && d.srcEm <= 32767) ? (short)d.srcEm : (short)-1;
                b.surf  = (d.surf < 0) ? (unsigned char)255
                                       : (d.surf > 254 ? (unsigned char)254 : (unsigned char)d.surf);
                b.pad2  = 0;
            }
        }
        bmap->nEmitted   = pm.nEmitted;      // same pass, same normalisation
        bmap->nDeposited = bmap->beams.size();
        // The one exact, unbiased trim that makes -beamcount mean what it says (host twin:
        // tracePhotonPass). Set nDeposited FIRST so the overshoot stays reportable.
        if (beams->target > 0) bmap->decimateTo((size_t)beams->target);
    }
    if (d_beams)     cudaFree(d_beams);
    if (d_beamCount) cudaFree(d_beamCount);

    // Downloading tens of millions of deposits and counting-sorting them into cells is tens
    // of seconds on a showcase `-n` — another silent phase between the deposit's last chunk
    // and the first gathered pixel, so it names itself too.
    if (stage && stage->reset)  stage->reset();
    if (stage && stage->report) stage->report("building photon map", 0, 0, nullptr, 0.0);
    buildMap();                             // host counting sort -> cell-contiguous runs
    // Named separately from the global map: on a caustics run this is a second full sort of its
    // own partition, and lumping the two under one caption made a stall in the second look like
    // a stall in the first.
    if (stage && stage->report) stage->report("building caustic map", 0, 0, nullptr, 0.0);
    buildCaustic();                         // ... and again for the caustic partition

    double energy[5] = {0,0,0,0,0};
    CUDA_CHECK(cudaMemcpy(energy, d_energy, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    eOut.emitted  += energy[0]; eOut.absorbed += energy[1]; eOut.sensor += energy[2];
    eOut.escaped  += energy[3]; eOut.residual += energy[4];
    cudaFree(d_depCount); cudaFree(d_energy);
    if (mapSave && *mapSave) {
        EnergyReport passE{energy[0], energy[1], energy[2], energy[3], energy[4]};
        if (savePhotonMap(mapSave, pm, passE, photonMapGuard(scene, diffraction), bmap,
                          causticsOn ? &pmC : nullptr))
            std::printf("[savemap] wrote %s: %zu photons (+%zu caustic) + %zu beams "
                        "(%lld emitted)\n",
                        mapSave, pm.photons.size(), pmC.photons.size(),
                        bmap ? bmap->beams.size() : (size_t)0, (long long)pm.nEmitted);
    }
    // Built AFTER the save, deliberately: the file must hold the RAW crossings so one cache
    // serves any later -beamradius / -beamk (the split is radius-dependent).
    //
    // This names itself for the same reason `buildMap` above does: the split subdivides every
    // stored chord and then builds a BVH over the result — on a showcase `-n` that is millions
    // of sub-beams and it reports nothing until it is finished, so an uninstrumented run sits
    // on the previous caption for the whole of it. Measured 7.6 s at `-beamcount 6M` and far
    // more at showcase counts.
    if (bmap && beams->build) {
        if (stage && stage->report) stage->report("splitting photon beams", 0, 0, nullptr, 0.0);
        beams->build(*bmap);
    }
    }   // end if (!mapLoaded): deposit + build + optional save

    // ---- upload the built grid ----
    // Shared by the global map and the caustic map: identical layout, identical fold, and the
    // per-map constants (radius, nEmitted) are read from the map itself — which is exactly what
    // makes the caustic map's smaller radius normalise correctly with no second code path.
    // `what` names the phase in the live window / status line. Tens of millions of photons get
    // converted on the host and streamed over PCIe here, which is tens of seconds on a showcase
    // `-n` and reported nothing at all before — the third silent stretch between the last deposit
    // chunk and the first gathered pixel.
    auto uploadPhotonMap = [&](const PhotonMap& M, DPhotonMap& D, const char* what) {
        D = DPhotonMap{};
        D.lat = M.lat;
        D.radius = (Real)M.radius;
        D.kGather = (Real)M.kGather;   // > 0 only on the caustic map (see -pmadaptive)
        D.photons = nullptr;
        if (!M.photons.empty()) {
            // Upload the sorted map host->device in chunks (no full mirror), folding each
            // photon's constant gather weight into the record (see DGatherPhoton):
            // p? = cie?(lambda) * power * norm / pi, with the CIE triple taken from
            // PhotonMap::cie (precomputed in double by build(), index-aligned with the
            // sorted photons). The radius is cast through Real first so the folded
            // normalization equals the old in-kernel double((Real)radius)^2 exactly.
            const double rr   = (double)(Real)M.radius;
            const double area = DPI * rr * rr;
            const double fold = (M.nEmitted > 0 && area > 0.0)
                              ? 1.0 / (area * (double)M.nEmitted * DPI) : 0.0;
            size_t n = M.photons.size();
            DGatherPhoton* d_sorted = nullptr;
            CUDA_CHECK(cudaMalloc(&d_sorted, n * sizeof(DGatherPhoton)));
            std::vector<DGatherPhoton> stg;
            if (stage && stage->reset)  stage->reset();   // rate/ETA measured from THIS phase
            if (stage && stage->report) stage->report(what, 0, (long long)n, nullptr, 0.0);
            for (size_t off = 0; off < n; off += PM_CHUNK) {
                // The chunk boundary is already here for host-RAM reasons, so it is also a free
                // stop seam. Bailing leaves `d_sorted` partly filled, which is harmless: the
                // camera loop below sees the same flag and gathers nothing, and the pointer is
                // still handed to `up.keep` after the loop so it is freed on the way out.
                if (ft::stopRequested()) break;
                size_t cnt = std::min(PM_CHUNK, n - off);
                stg.resize(cnt);
                for (size_t i = 0; i < cnt; ++i) {
                    const Photon& p = M.photons[off + i];
                    const Vec3&  pp = M.pos[off + i];
                    const Vec3&  ci = M.cie[off + i];
                    DGatherPhoton& d = stg[i];
                    d.pos = DVec3(pp.x, pp.y, pp.z);
                    d.n   = DVec3(p.n.x,   p.n.y,   p.n.z);
                    const double w = (double)p.power * fold;
                    d.pX = (float)(ci.x * w);
                    d.pY = (float)(ci.y * w);
                    d.pZ = (float)(ci.z * w);
                    d.lambda = p.lambda;
                }
                CUDA_CHECK(cudaMemcpy(d_sorted + off, stg.data(), cnt * sizeof(DGatherPhoton),
                                      cudaMemcpyHostToDevice));
                if (stage && stage->report)
                    stage->report(what, (long long)(off + cnt), (long long)n, nullptr, 0.0);
            }
            D.photons = (const DGatherPhoton*)up.keep(d_sorted);
        }
        // cellStart always has its buckets after build() (an empty map the 27 residue buckets, every
        // [begin,end) slice empty, so the gather loop simply never runs) — always upload it.
        D.cellStart = (const int*)up.keep(uploadVec(M.cellStart));
    };
    DPhotonMap dpm{}, dpmC{};
    uploadPhotonMap(pm, dpm, "uploading photon map");
    // dpmC.photons stays null when the split is off or produced nothing — which is the very
    // flag the two device gathers test to skip their second estimate entirely.
    if (causticsOn && !pmC.photons.empty()) uploadPhotonMap(pmC, dpmC, "uploading caustic map");
    else                                    dpmC.cellStart = dpm.cellStart;  // never dereferenced

    // ---- upload the built beam map ----
    // Mode M's volume gather: the same device layout mode J's merges use, minus the MIS
    // partials (see uploadBeamMapCuda). nNodes == 0 is the "no volume gather" sentinel.
    DBeamMap dbm{};
    uploadBeamMapCuda(scene, bmap, up, dbm, stage, /*withMis=*/false);
    if (pin) pin->lightSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tCall).count();

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
    // Diagnostic twin of gpuSppChunks' FTRACE_CHUNK_DEBUG, and for the same reason: this
    // gather reports only at whole-spp boundaries AND only on the ~30 s log cadence, so an
    // spp whose line is throttled away merges into its neighbour and two very differently
    // priced samples read as one. That is not hypothetical — it is how a claimed 10x
    // per-spp cost difference on gallery_rain came to be argued about from differenced log
    // timestamps rather than measured. Set FTRACE_CHUNK_DEBUG=1 for every spp's wall time,
    // =2 for every SLICE's as well (sample range + rate), which is what exposes the
    // per-scanline-band cost structure inside one spp.
    const int chunkDebug = [] { const char* e = std::getenv("FTRACE_CHUNK_DEBUG");
                                return (e && *e) ? std::atoi(e) : 0; }();
    auto lastReport = std::chrono::steady_clock::now();
    bool stopped = false;
    // ---- the gather's cancellation flag (see kGather's poll for why it must exist) ----
    // A chunk it cuts short is dropped, not folded (kFilmFold), as DeviceStopScope requires.
    DeviceStopScope stopScope;
    // ---- and the gather's PROGRESS counter, the same mechanism pointed the other way ----
    //
    // kGather counts retired samples into this; the poller below reads it while the launching
    // thread is blocked. Without it the caption was correct but frozen at 0% for a whole spp,
    // which at 960x540 is the ENTIRE launch (see g_dGatherDone).
    unsigned long long* h_gatherDone = nullptr;
    unsigned long long* d_gatherDone = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&h_gatherDone, sizeof(unsigned long long),
                             cudaHostAllocMapped));
    *h_gatherDone = 0;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_gatherDone, h_gatherDone, 0));
    CUDA_CHECK(cudaMemcpyToSymbol(g_dGatherDone, &d_gatherDone, sizeof(unsigned long long*)));
    // Armed by the launch site for exactly the span of one launch. `progOffset < 0` means
    // disarmed, which is the state in every phase that is NOT a gather launch — the poller must
    // not paint over another phase's caption, and must not touch `stage` once the main thread is
    // running again and reporting for itself.
    std::atomic<long long> progOffset{-1};
    std::atomic<long long> progTotal{0};
    char progText[96] = {0};
    // `stage->report` mutates a shared Ticker (rate window, repaint throttles) and is not
    // reentrant. Disarming happens UNDER this lock, so the main thread's own reports after a
    // launch cannot overlap a report the poller is still inside.
    std::mutex reportMu;
    PollStop pollQuit;
    std::thread pollThread([&, h_gatherDone] {
        do {
            if (ft::stopRequested()) return;   // the kernels see the stop itself (DeviceStopScope)
            // reportLive, not report: this is progress from INSIDE a launch that has not
            // returned, where the retirement curve is convex and no honest rate exists — and
            // feeding those samples to the trailing window would wreck the rate for the
            // per-launch reports too. See StageProgress::reportLive.
            if (stage && stage->reportLive && progOffset.load(std::memory_order_acquire) >= 0) {
                const long long d = (long long)*(volatile unsigned long long*)h_gatherDone;
                std::lock_guard<std::mutex> lk(reportMu);
                const long long off = progOffset.load(std::memory_order_acquire);
                if (off >= 0)
                    stage->reportLive(progText, off + d,
                                      progTotal.load(std::memory_order_relaxed));
            }
        } while (!pollQuit.wait(std::chrono::milliseconds(50)));
    });
    // Joined and freed however this function leaves — including the CUDA_CHECK throw paths,
    // which are numerous below and would otherwise leak a detached thread into a dead context.
    // The device global is reset to null BEFORE the memory goes: dGatherRetire runs in other
    // modes' kernels too, and a pointer left at a freed mapping is an illegal-address fault in
    // the next of them this process launches.
    struct GatherProgressGuard {
        PollStop& quit; std::thread& th; unsigned long long* mem;
        ~GatherProgressGuard() {
            quit.stop();
            if (th.joinable()) th.join();
            unsigned long long* noDone = nullptr;
            cudaMemcpyToSymbol(g_dGatherDone, &noDone, sizeof noDone);
            if (mem) cudaFreeHost(mem);
        }
    } gatherProgressGuard{pollQuit, pollThread, h_gatherDone};
    // Learned sub-chunk slice (flat (pixel, sample) units), carried ACROSS cameras: on a
    // flythrough consecutive frames cost almost the same, so re-probing 600 times would be
    // 600 needlessly small launches for nothing. Re-clamped to the camera's own npix below.
    long long slice = 0;
    for (int c = 0; c < nc && !stopped; ++c) {
        // A stop pending before the first camera must not buy four device allocations and a
        // camera bake it will only throw away. `stopped` is seeded false and is only ever set
        // from inside the loop, so without this the flag is invisible on the first iteration.
        if (ft::stopRequested()) { stopped = true; break; }
        // An epoch of a moving scene: the host posed the camera at the epoch's time, which the
        // gather takes too (a camera moving over a still scene keeps its own, kGather).
        DCamera hc = bakeCamera(scene, cams[c], resX[c], resY[c], up, up.sc.mo.passU >= 0.0f);
        const size_t npix = (size_t)resX[c] * resY[c];
        double* d_film = nullptr; CUDA_CHECK(cudaMalloc(&d_film, npix * 3 * sizeof(double)));
        double* d_hits = nullptr; CUDA_CHECK(cudaMalloc(&d_hits, npix * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_film, 0, npix * 3 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_hits, 0, npix * sizeof(double)));
        // Scratch accumulator for the chunk in flight. kGather writes here, and kFilmFold
        // folds it into d_film only once the WHOLE chunk has landed — see kFilmFold for why
        // a half-committed chunk would band the image. ~16 MB at 960x540, i.e. nothing next
        // to the photon map this mode is already holding.
        double* s_film = nullptr; CUDA_CHECK(cudaMalloc(&s_film, npix * 3 * sizeof(double)));
        double* s_hits = nullptr; CUDA_CHECK(cudaMalloc(&s_hits, npix * sizeof(double)));
        const unsigned long long seed = 0xA24BAED4963EE407ULL
                                      ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(c + 1))
                                      ^ g_rngSalt;   // `-seed`; 0 by default, so XOR is a no-op
        // Chunk spp so a single launch stays well under the Windows TDR watchdog even when a
        // caustic cell holds a dense photon cluster (heavy density query).
        long long chunk = 200000 / (long long)(npix ? npix : 1); if (chunk < 1) chunk = 1;
        // Sub-chunk slicing, retargeted to ~0.25 s of work per launch. This is the ONLY seam
        // an external `ftrace -stop` can land on inside a gather: at these resolutions
        // `chunk` has already clamped to a single spp, so before slicing the shortest
        // possible stop latency was one entire frame of gathering — minutes with -beams,
        // well past the -stop wait, which is exactly how a healthy render came to be
        // reported as a failed stop. Seeds depend only on the global sample index, so
        // slicing changes no sample: the image is the one an unsliced gather produces.
        //
        // THE FLOOR IS AN OCCUPANCY FLOOR, NOT A FRACTION OF THE FRAME, and getting that
        // wrong cost an order of magnitude. kGather launches a PERSISTENT grid of
        // kGatherGrid x kGatherBlock threads and grid-strides over [i, hi): a slice holding
        // fewer samples than that leaves the surplus threads with literally nothing to do,
        // so throughput falls off a cliff and the 0.25 s target becomes unreachable — at
        // which point the controller shrinks the slice again, which starves the device
        // further. The loop drives its own input. Measured on gallery_rain (960x540,
        // -beams, RTX 4090), one frame, identical work:
        //
        //     slice 503904 (whole frame, one launch)   16927 samples/s
        //     slice 129616                             40165 samples/s
        //     slice   8101 (the old npix>>6 floor)       631-1224 samples/s   <- 3% of the grid
        //
        // i.e. 334 s for a frame sliced at the old floor against 29.8 s for the same frame
        // in one launch — an 11x tax for nothing, and it alternated spp to spp depending on
        // whether the previous chunk's tiny tail slice happened to time as "too fast to
        // measure" and let the size grow back. Measured end to end on that frame at -spp 4:
        // 713 s before, 119 s after (6.0x), and flat spp to spp instead of 334/30/321/29.
        //
        // The stop seam it buys back is small and worth stating honestly: the worst single
        // launch goes from 12.8 s (a floor slice in the cloud band) to 32 s (a whole spp),
        // still comfortably inside the 120 s `-stop` wait, and the frame it is a seam
        // *within* now finishes 6x sooner — so a stop lands sooner in wall clock either way.
        //
        // FOUR samples per thread, not one. Filling the grid once is necessary but not
        // sufficient: kGather is wildly divergent (a probe ray through the cloud gathers
        // ~840 beams, one aimed at the floor gathers none), and the grid-stride loop is the
        // only load balancing there is — with one sample per thread it has nothing to hand
        // a thread that finished early, and the launch costs its slowest ray. Measured on
        // the same frame: 262144 samples (1.0/thread) ran the top half at 16085/s and the
        // bottom half at 8428/s, 46.7 s for the spp; one 503904-sample launch (1.9/thread)
        // covering BOTH did it at 16927/s, 29.8 s — 1.57x for nothing but a bigger slice.
        const long long sliceOcc = 4ll * kGatherGrid * kGatherBlock;
        const long long sliceFrac = ((long long)npix >> 6) + 1;
        const long long sliceLo = (sliceFrac > sliceOcc) ? sliceFrac : sliceOcc;
        if (slice < sliceLo) slice = ((long long)npix >> 4) + 1;          // first probe: 1/16 spp
        if (slice < sliceLo) slice = sliceLo;
        long long sppDone = 0;
        bool abandoned = false;
        // Name the gather in the window title / log, with a measure.
        //
        // WHY THIS IS NOT REDUNDANT WITH `prog` BELOW. The live SppProgress report only fires
        // after a COMPLETE chunk, and `chunk` has already clamped to one spp at any real
        // resolution — so on a heavy scene the first title update is one whole spp away. On
        // gallery_rain that is upwards of fifteen minutes during which the window still read
        // `building beam map…`, the caption the previous phase left behind: the beam map had
        // in fact finished in 23.7 s and the render was gathering the whole time, but nothing
        // said so, and the only visible evidence — one CPU core pegged by the default
        // spin-waiting cudaDeviceSynchronize, with the GPU at 100% — reads exactly like a
        // single-threaded host build that will not end. A stage line fires every 0.25 s from
        // inside the slice loop instead, so the gather names itself from its first quarter
        // second, and (on the 30 s log cadence) a headless run's log says so too.
        //
        // THAT LAST SENTENCE WAS WRONG AND THE BUG SURVIVED IT — see the report added just
        // before the `base` loop below. "Every 0.25 s from inside the slice loop" assumes the
        // slice loop ITERATES; where `sliceOcc >= total` it does not, and the first report is
        // one whole spp away, which is exactly the symptom described above. The report before
        // the first launch is what actually fixes it; this one refines the caption afterwards.
        const long long sampTotal = (long long)npix * spp;   // pixel-samples in this frame
        char stageText[96];
        std::snprintf(stageText, sizeof stageText, "gathering frame %d/%d", c + 1, nc);
        // The poller's copy of the caption. Written here, with the poller disarmed, so it is
        // never read while being written.
        std::snprintf(progText, sizeof progText, "%s", stageText);
        if (stage && stage->reset) stage->reset();   // rate/ETA measured from THIS phase
        // Name the gather BEFORE the first launch, not after it.
        //
        // The report below fires from inside the slice loop, i.e. only once a launch has
        // already RETURNED — and wherever `sliceOcc >= total` the loop runs exactly once and
        // covers the whole frame, so "after the first launch" is one entire spp away. That is
        // the case at every ordinary resolution: at 960x540, total = 518400 while sliceOcc =
        // 4*2048*128 = 1048576, twice the frame. So the caption the previous phase left
        // behind (`building beam map…`) stayed up for the whole first sample — measured at
        // 45+ minutes on gallery_rain at -n 800M, with a render in perfect health behind it.
        // Fixing the 17109 report was never going to help; it is on the wrong side of the
        // launch. This one costs nothing: done/total = 0 is the documented "no meaningful
        // measure yet" encoding, and passing a null `partial` means no film download.
        if (stage && stage->report) stage->report(stageText, 0, sampTotal, nullptr, 0.0);
        for (long long base = 0; base < spp; base += chunk) {
            long long cs2 = (base + chunk <= spp) ? chunk : (spp - base);
            long long total = (long long)npix * cs2;
            const auto ct0 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaMemset(s_film, 0, npix * 3 * sizeof(double)));
            CUDA_CHECK(cudaMemset(s_hits, 0, npix * sizeof(double)));
            bool chunkDone = true;
            for (long long i = 0; i < total; ) {
                // Honour a stop that arrived BEFORE or BETWEEN launches, without paying a
                // launch first. The post-launch poll further down is guarded by `i < total`,
                // which is never true wherever `sliceOcc >= total` — the loop runs once and
                // covers the frame — so a stop that was already pending when the gather
                // STARTED still cost a full spp before anything looked at the flag. That is
                // not hypothetical: on gallery_rain at -n 800M the `[stop] external stop
                // requested` line sits in the log immediately after the beam-BVH upload, i.e.
                // the flag was set before this loop was entered, and the process still ran
                // 45+ more minutes. Polling at the top makes such a stop free.
                if (ft::stopRequested()) { chunkDone = false; break; }
                const long long hi = (i + slice < total) ? (i + slice) : total;
                const auto t0 = std::chrono::steady_clock::now();
                // Arm the intra-launch progress counter. Zeroed per launch and offset by the
                // samples this frame has already retired, so the bar reads against the whole
                // frame rather than restarting at each slice. Both stores are ordinary host
                // writes to memory the device has not been told to touch yet — the kernel is
                // launched on the next line, so there is no race to order against.
                *(volatile unsigned long long*)h_gatherDone = 0;
                progTotal.store(sampTotal, std::memory_order_relaxed);
                progOffset.store(base * (long long)npix + i, std::memory_order_release);
                DeviceStopScope::clearSeen();
                kGather<<<kGatherGrid, kGatherBlock>>>(
                                       up.sc, dpm, dpmC, dbm, hc, s_film, s_hits, i, hi, cs2, spp,
                                       base, resX[c], diffraction ? 1 : 0, fgRays, seed);
                cudaCheckKernel("photon-gather");
                CUDA_CHECK(cudaDeviceSynchronize());   // the launch is async; time the WORK
                // Disarm UNDER the lock: the poller may be inside stage->report right now, and
                // the main thread is about to call it itself a few lines below.
                { std::lock_guard<std::mutex> lk(reportMu);
                  progOffset.store(-1, std::memory_order_release); }
                const double sec = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                const long long did = hi - i;
                i = hi;
                if (chunkDebug >= 2)
                    std::fprintf(stderr, "[chunk] gather c%d spp %lld..%lld slice [%lld,%lld) "
                                 "%lld samples in %s (%.0f/s)\n", c + 1, base, base + cs2,
                                 hi - did, hi, did, humanDur(sec).c_str(),
                                 sec > 0 ? did / sec : 0.0);
                // Only until the frame's first complete chunk exists: from there on `prog`
                // below owns the caption and says strictly more (spp, photons, noise %).
                //
                // The sub-chunk accumulator is also a perfectly good IMAGE while it fills.
                // kGather maps sample index -> pixel as idx/chunkSpp, so a partial chunk is a
                // scanline-order prefix: the covered pixels each hold exactly `cs2` samples
                // and the rest are still zero, which draws as the frame arriving top to
                // bottom. That is the difference between a dark placeholder for the whole
                // first spp — half an hour on gallery_rain with -beams — and watching it
                // land. Only assembled when the host says it would actually be drawn, since
                // it costs a device->host copy of the film.
                if (sppDone == 0 && stage && stage->report) {
                    const bool wantImg = stage->wantFilm && stage->wantFilm();
                    if (wantImg) downloadFilm(c, s_film, s_hits, npix);
                    stage->report(stageText, base * (long long)npix + i, sampTotal,
                                  wantImg ? &out[c] : nullptr, (double)cs2);
                }
                // Abandon the chunk in flight, first chunk included.
                //
                // It used to require `base > 0`, so that a stop could never leave the film
                // with zero complete samples in it — the frame would then have to be written
                // black, which is a worse answer to a stop than the one extra chunk it costs.
                // The premise is right and the conclusion was wrong: the cost is not "one
                // extra chunk", it is one entire spp of a gather whose spp can run to a
                // quarter of an hour, and measured on gallery_rain a `-stop` sat unanswered
                // for the full 900 s wait and was reported FAILED on a render that was
                // winding down correctly. The frame with nothing in it is not written at all
                // instead (see the `sppDone > 0` guard on `onFrame` below), which is both
                // honest and what every earlier frame of a flythrough — already safely on
                // disk — makes harmless.
                // The `i < total` guard this used to carry is now not merely useless but
                // WRONG. It meant "only abandon if slices remain", which assumed a launch
                // that returned had run to completion. With device-side cancellation a launch
                // can return having stopped half way, leaving the scratch film holding a
                // sample for some pixels and not others — and `i` still reaches `total`,
                // because the slice was issued whole. Folding that would put exactly the
                // ~1/spp brightness band across the frame that kFilmFold's comment describes.
                // So: if a stop is pending, the chunk in flight is partial by definition and
                // must be discarded, wherever `i` happens to be -- as it must if a kernel saw
                // the stop, whatever the host's flag says now (DeviceStopScope::seen).
                if (ft::stopRequested() || DeviceStopScope::seen()) { chunkDone = false; break; }
                // Never below 1/64 spp (a slice that keeps halving turns the gather into
                // launch latency), never more than 4x up in one step (one anomalously fast
                // slice must not produce a minutes-long next one). A slice too fast to time
                // takes the 4x growth outright, so a cheap scene climbs back out of the
                // probe in two or three launches instead of paying per-launch overhead for
                // the whole frame.
                double want = (sec > 1e-6) ? (double)did * (0.25 / sec) : 4.0 * (double)slice;
                if (want > 4.0 * (double)slice) want = 4.0 * (double)slice;
                slice = (long long)want;
                if (slice < sliceLo) slice = sliceLo;
            }
            // Partial chunk discarded, not folded: the film still holds exactly `sppDone`
            // complete samples per pixel, so it needs one last report to say so.
            if (!chunkDone) { stopped = true; abandoned = true; break; }
            kFilmFold<<<256, 128>>>(d_film, d_hits, s_film, s_hits, (long long)npix);
            cudaCheckKernel("photon-gather-fold");
            sppDone = base + cs2;
            if (chunkDebug >= 1) {
                const double cdt = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - ct0).count();
                std::fprintf(stderr, "[chunk] gather cam %d/%d spp %lld..%lld (%lld spp, "
                             "%lld samples) in %s (%.0f samples/s)\n", c + 1, nc, base,
                             sppDone, cs2, total, humanDur(cdt).c_str(),
                             cdt > 0 ? total / cdt : 0.0);
                std::fflush(stderr);
            }
            const bool frameDone = (sppDone >= spp);
            const bool stopNow = ft::stopRequested();
            // Live view: after a chunk, hand the host the frame-so-far so it can refresh the
            // window/preview. Throttle to ~10 Hz (a high-res gather chunks one spp at a time,
            // which is far finer than the eye needs) but always report the completed frame —
            // and always report the last one before a stop, so the host normalises by the spp
            // that actually landed rather than by a stale count.
            if (live) {
                auto now = std::chrono::steady_clock::now();
                if (frameDone || stopNow ||
                    std::chrono::duration<double>(now - lastReport).count() >= 0.1) {
                    downloadFilm(c, d_film, d_hits, npix);
                    if (prog->report(out[c], sppDone, frameDone)) stopped = true;
                    lastReport = now;
                }
            }
            if (stopped || stopNow) { stopped = true; break; }
        }
        if (abandoned && live && sppDone > 0) {
            downloadFilm(c, d_film, d_hits, npix);
            prog->report(out[c], sppDone, false);
        }
        downloadFilm(c, d_film, d_hits, npix);   // ensure out[c] holds the final accumulation
        cudaFree(d_film); cudaFree(d_hits);
        cudaFree(s_film); cudaFree(s_hits);
        // Hand the finished frame to the host for IMMEDIATE crash-safe write, then release
        // its buffers so a long flythrough runs in ~one frame of host RAM instead of holding
        // all nc films to the end (mirrors the CPU mode-M path, which writes per frame). If
        // the host asks to stop (window closed / Ctrl-C), quit after this frame — everything
        // written so far is already safely on disk.
        // A frame the stop caught before its first complete sample holds nothing but zeros;
        // writing it would replace a good previous render of the same path with a black PNG.
        // Say so and skip it — the frames before it are already on disk.
        if (onFrame && sppDone > 0) {
            bool stopReq = (*onFrame)(c, out[c], sppDone);
            Film empty; empty.resX = resX[c]; empty.resY = resY[c];   // shape kept, buffers freed
            out[c] = std::move(empty);
            if (stopReq) stopped = true;
        } else if (onFrame) {
            std::printf("\n[camera] frame %d/%d stopped before its first complete sample — "
                        "nothing written for it.\n", c + 1, nc);
            std::fflush(stdout);
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
    const Scene* scene = nullptr;   // for a blurred pass's time and pose (sppmSessionPass)
    Camera hostCam;
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
    // SPPM's gather shades every query point as Lambertian (no hair case) -- a hair scene
    // goes to the CPU SPPM even though the mode-M gather takes hair since 0.335.0.
    if (sceneUsesHairMaterial(scene)) return false;
    return cudaPhotonMapSupported(scene);
}

// The device memory a mode-S session keeps a pixel (sppmSessionBegin's state and film): what the VRAM
// preflight charges a mode-S frame (main.cpp, vramAffords).
size_t cudaSppmPixelBytes() {
    // tau, directSum and the film (3 doubles each); nAcc, vpThr and radius; vpValid and vpSpec;
    // vpHit; vpRatio's spectral bins.
    return 12 * sizeof(double) + 2 * sizeof(unsigned char) + sizeof(gpu::DHit) + gpu::DSpecThr::K * sizeof(float);
}
SppmSession* sppmSessionBegin(const Scene& scene, const Camera& cam, int resX, int resY,
                              double R0, bool diffraction, int maxBounce, int heroC) {
    if (!cudaAvailable() || !cudaSppmSupported(scene)) return nullptr;
    SppmSession* s = new SppmSession();
    s->resX = resX; s->resY = resY; s->npix = (size_t)resX * resY;
    s->diffraction = diffraction; s->maxBounce = (maxBounce > 0) ? maxBounce : 32; s->heroC = heroC;
    buildUploadScene(scene, s->up);
    s->cam = bakeCamera(scene, cam, resX, resY, s->up);
    s->scene = &scene; s->hostCam = cam;
    const size_t np = s->npix;
    CUDA_CHECK(cudaMalloc(&s->st.tau,       np * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->st.directSum, np * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->st.nAcc,      np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->st.vpThr,     np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->st.radius,    np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->st.vpValid,   np * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&s->st.vpHit,     np * sizeof(gpu::DHit)));
    CUDA_CHECK(cudaMalloc(&s->st.vpRatio,   np * gpu::DSpecThr::K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->st.vpSpec,    np * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(s->st.vpSpec,    0, np * sizeof(unsigned char)));
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
    dSetLenScale(s->up.sc);
    const long long passIdx = s->passes;                 // 0-based index of THIS pass

    // (1) Camera visible-point pass: fresh visible point + direct sample per pixel.
    unsigned long long vpSeed = 0xA24BAED4963EE407ULL
                              ^ ((unsigned long long)(passIdx + 1) * 0x9E3779B97F4A7C15ULL)
                              ^ g_rngSalt;   // `-seed`; 0 by default, so XOR is a no-op
    // Motion blur (host twin sppmPass): a moving scene's pass -- visible points, photons and
    // gather -- at one time, stratified over the passes, the camera posed there; a moving camera
    // over a still scene leaves the photons timeless and poses each pixel's ray at its own time.
    const Camera& hc = s->hostCam;
    const bool passTimed = hc.timed() && s->scene->hasMotion();
    const double tPass = passTimed ? hc.shutterTime(passTime((uint64_t)passIdx, 0x2545F4914F6CDD1DULL)) : 0.0;
    s->up.sc.mo.passU = passTimed ? devicePassU(*s->scene, tPass) : -1.0f;
    const DCamera camPass = passTimed ? posedBaked(s->cam, hc, tPass) : s->cam;
    const int perPixel = (!passTimed && hc.timed() && hc.moves()) ? 1 : 0;
    kSppmVisiblePoint<<<2048, 128>>>(s->up.sc, camPass, s->st, s->resX, s->resY,
                                     s->maxBounce, vpSeed, passIdx + 1, perPixel);
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
                      (unsigned long long)s->emittedTotal, /*wavefront*/false, CAM_B, s->heroC, photonMaxBounce());
    };
    unsigned long long cap = (unsigned long long)((double)photonsPerPass * 2.5) + (1ull << 20);
    if (cap > s->photonsCap) {
        { size_t freeB = 0, totalB = 0;
          if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
              // Budget half the free VRAM, counting what the current buffer already holds, once the
              // walk's and the gather's stacks are set aside: the first pass allocates this before
              // kTrace and kSppmGather have launched (pendingLocalBytes).
              const size_t pending = pendingLocalBytes(cudaRouteLocalBytes(cudaSppmRoute()));
              const size_t avail = freeB + s->photonsCap * sizeof(DPhoton);
              unsigned long long fit = (unsigned long long)((avail > pending ? avail - pending : 0) / 2 / sizeof(DPhoton));
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
    MergeLattice lat = mergeLatticeAt(0.0, 0.0, 0.0, cellSize, 1);
    DPhotonMap dpm{};
    dpm.photons = nullptr;
    if (nDep > 0) {
        const size_t n = (size_t)nDep;
        BboxF bb = thrust::transform_reduce(pol,
            thrust::device_pointer_cast(s->d_photons),
            thrust::device_pointer_cast(s->d_photons + n),
            PhToBboxF{}, BboxF{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX},
            BboxMergeF{});
        // The host build promoted float positions to double BEFORE the bbox/pad math; min/max
        // commute with that promotion, so these are the same doubles (mergeLatticeOver, as the
        // host's PhotonMap::buildGrid).
        lat = mergeLatticeOver(bb.mnx, bb.mny, bb.mnz, cellSize, n);
        mergeBinDevice(pol, s->d_photons, n, lat, 2048, s->d_cellKey, s->cellKeyCap, s->d_order, s->orderCap,
                       s->d_cellStart, s->cellStartCap, nullptr, nullptr);
        ensureDevCap(s->d_gather, s->gatherCap, n);
        kSppmGatherConvert<<<2048, 128>>>(s->d_photons, s->d_order, (long long)n, s->d_gather);
        cudaCheckKernel("sppm-gather-convert");
        dpm.photons = s->d_gather;
    } else {   // the residue buckets alone, empty: a gather reads 27 of them
        ensureDevCap(s->d_cellStart, s->cellStartCap, mergeBucketCount(1) + 1);
        CUDA_CHECK(cudaMemset(s->d_cellStart, 0, (mergeBucketCount(1) + 1) * sizeof(int)));
    }
    dpm.lat = lat;
    dpm.radius = (Real)rMax;
    dpm.cellStart = s->d_cellStart;

    // (3) Gather + progressive update.
    // `vpSeed` / `passIdx + 1` are the same pair kSppmVisiblePoint was given, so the footprint
    // probe's stream is reproducible for a fixed pass sequence; the kernel offsets it so the two
    // do not overlap.
    kSppmGather<<<2048, 128>>>(s->up.sc, dpm, s->st, s->resX, s->resY, alpha,
                               vpSeed, passIdx + 1);
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
    cudaFree(s->st.vpRatio); cudaFree(s->st.vpSpec);
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
// kVcmLight — nLight light subpaths (one per pixel, or fewer when the store would not fit:
// vcmSessionBegin) store their connectible vertices into a per-path slab (avoids cross-thread
// atomics) and splat connect-to-camera contributions; (2) compact the slab ON DEVICE into
// contiguous per-path ranges (exclusive-scan the counts -> pathBegin/End, scatter — so strategy
// (c)'s same-lambda vertex connection reads its PAIRED light path); (3) build the uniform hash
// grid over the compacted vertices ON DEVICE (bbox reduce + cell keys + STABLE sort by cell id +
// vectorized lower_bound — arithmetic-exact twin of the vcm.h VcmGrid::build counting sort,
// including its output order); (4) launch kVcmCamera — one camera subpath per pixel does
// emission/NEE/connection/merge and adds this pass's radiance (camera result + light splat) into
// the persistent `accum` sum. Resolve divides accum by the pass count. Only ~30 bytes (vertex
// total + bbox) cross the PCIe bus per pass. Validated statistically against the CPU
// (independent MC).
struct VcmSession {
    DUpload up;
    gpu::DCamera cam{};
    const Scene* scene = nullptr;   // for a blurred pass's time and pose (vcmSessionPass)
    Camera hostCam;
    int resX = 0, resY = 0;
    size_t npix = 0;
    int  diffraction = 0;
    int  maxDepth = 8;
    int  vcmCap = 8;              // max stored connectible vertices per light subpath (maxDepth - 1)
    size_t nLight = 0;            // light subpaths a pass: npix, or fewer if the store would not fit
                                  // (the arrays marked nLight below are indexed by light subpath)
    int  heroC = 1;               // hero bundle width (1 == classic single-λ session)
    int  secStride = 0;           // heroC-1 secondary slots per stored vertex (0 when heroC==1)
    long long passes = 0;
    // Persistent (allocated once in Begin):
    // nLight * vcmCap: the light walks' stores, and after the compaction the grid's cell-ordered
    // copy (DVcmGrid::byCell); its secondary-λ rows beside it (nLight*vcmCap*secStride, NULL unless
    // heroC>1).
    gpu::DVcmLV* d_lvSlab = nullptr;
    gpu::DVcmSec* d_lvSecSlab = nullptr;
    int*    d_lvCount = nullptr;       // nLight
    double* d_splat   = nullptr;       // npix*3 (this pass's connect-to-camera XYZ)
    double* d_accum   = nullptr;       // npix*3 (running SUM over passes)
    gpu::Real* d_lamBuf = nullptr;     // nLight*heroC (per-path BUNDLE, shared light<->camera)
    double* d_invLam  = nullptr;       // nLight*heroC (invPdfLambda; [i*C]<=0 marks "no wavelength")
    int*    d_pathBegin = nullptr;     // nLight (device-scanned per-pass)
    int*    d_pathEnd   = nullptr;     // nLight
    // Device scratch for the on-device compaction + grid build. The vertex-sized ones and the bucket
    // table are allocated in Begin at the most a pass can need (nLight * vcmCap vertices), so
    // ensureDevCap never grows them; the thrust arena grows with the sort:
    gpu::DVcmLV* d_lvCompact = nullptr; size_t lvCompactCap = 0;
    gpu::DVcmSec* d_lvSecCompact = nullptr; size_t lvSecCompactCap = 0;
    // The stored vertices' hit states (gpu::DVertHitState), slab and compact copy, nLight * vcmCap
    // each, kept where a connection's patterns need them -- a scene that moves or reads `cavity`
    // (null otherwise, so the common scene pays nothing).
    gpu::DVertHitState* d_lvHsSlab = nullptr;
    gpu::DVertHitState* d_lvHsCompact = nullptr;
    int*    d_cellKey   = nullptr;      size_t cellKeyCap = 0;
    unsigned char* d_tag = nullptr;     size_t tagCap = 0;   // byCell[k]'s tag (MergeKey)
    int*    d_order     = nullptr;      size_t orderCap = 0;
    int*    d_cellStart = nullptr;      size_t cellStartCap = 0;  // entries (buckets + 1)
    ThrustArena arena;                  // thrust temp storage (scan/sort/reduce)
    // What the passes must leave free beside the bucket table: the sort's scratch at its worst (the
    // arena grows to it through allocations that end the render if refused) and the kernels' local
    // memory (cudaVcmRoute's kernels), reserved at their first launch.
    size_t sortWorst = 0, localNeed = 0;
    // The merges' queue (gpu::DVcmMergeQueue, VCM-WAVEFRONT; null reqs: merged inline; its two
    // counters, sites asked for and sites merged inline, in one allocation at nReqs) and its profile (mq.profile, and the last
    // pass's on the host: mqProfile, empty before the first pass has run), and how its waves went:
    // the last wave's sites a camera path (< 0 before the first), the fewest paths a wave holds
    // (the camera kernel's resident threads), and the totals vcmSessionEnd reports.
    gpu::DVcmMergeQueue mq{};
    std::vector<int> mqProfile;
    cudaEvent_t mqEv[2] = {nullptr, nullptr};
    double  mqRate = -1.0;
    long long mqLeast = 1;
    long long mqSites = 0, mqInline = 0, mqPaths = 0, mqWaves = 0;
    double  mqMsGather = 0.0;
};

// Mode U's merge queue's capacity (gpu::DVcmMergeReq), or what an eighth of the memory left after
// the light-vertex store holds.
static constexpr long long kVcmReqCap = 1LL << 20;
// The most light vertices a merge site holds and is still merged where it was found
// (DVcmMergeQueue::inlineMax).
static constexpr int kVcmInlineMax = 32;

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
    return cudaBidirCoreSupported(scene) && scene.media.empty();   // hair included (0.364.0)
}

VcmSession* vcmSessionBegin(const Scene& scene, const Camera& cam, int resX, int resY,
                            bool diffraction, int maxDepth, int heroC) {
    if (!cudaAvailable() || !cudaVcmSupported(scene)) return nullptr;
    // The camera walk's spectral carrier (VCM-SPEC) takes mode J's switches, as the host settled
    // them for this render: bdpt::jSpecEnabled (cudaSetJSwitches) and -jdispwin / -jdispall
    // (cudaSetJDisp).
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jSpecOn, &g_jSpecHost, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jDispWin, &g_jDispWinHost, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu::c_jDispAll, &g_jDispAllHost, sizeof(int)));
    uploadJBinQ(scene.emitSampler);   // mode U's bundles are drawn from it (DJSpecRun::hole)
    VcmSession* s = new VcmSession();
    s->resX = resX; s->resY = resY; s->npix = (size_t)resX * resY;
    s->diffraction = diffraction ? 1 : 0;
    s->maxDepth = (maxDepth > 1) ? maxDepth : 9;     // EDGES: -max-bounce + 1
    // Hero bundle width. The kernels are templated on the SECONDARY slot count, so a C==1
    // session instantiates kVcm*T<0>: no sec slab, no per-λ arrays, no extra registers.
    const int C = vcmSessionHeroC(heroC);
    s->heroC = C;
    s->secStride = C - 1;
    // A light subpath stores at most one vertex per edge up to maxDepth - 1 (kVcmLightT's
    // lastEdge), and a pass holds every stored vertex twice -- the per-path slab and its compact
    // copy -- with the secondary-λ rows beside each, plus a cell key, an order index, a tag and two
    // buckets of the grid's table. One light subpath per pixel can ask for gigabytes (6.6 GB at 1100x733 and
    // -max-bounce 64), so the store gets a budget: an eighth of the card, at least 768 MB, and
    // never more than half the memory free once the scene is up and the kernels' stacks are set
    // aside (reserved at their first launch, after this, unless an earlier launch holds them). A pass that would not fit traces FEWER light
    // subpaths, each shared by several camera paths (kVcmCameraT's pairing). Every vertex of
    // every subpath is stored, so no strategy the weights count goes missing: only the variance
    // moves.
    s->vcmCap = s->maxDepth - 1;
    buildUploadScene(scene, s->up);
    s->cam = bakeCamera(scene, cam, resX, resY, s->up);
    s->scene = &scene; s->hostCam = cam;
    const size_t np = s->npix;
    size_t budget = (size_t)768 << 20;
    s->localNeed = cudaRouteLocalBytes(cudaVcmRoute(C));
    {
        const size_t pending = pendingLocalBytes(s->localNeed);
        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
            const size_t floorB = budget;
            if (freeB > pending) {
                budget = std::min(std::max(totalB / 8, budget), (freeB - pending) / 2);
                // Said where the stacks are why: some still to reserve, and the free memory's half
                // alone would have met the floor (other programs' shortage is not theirs).
                if (budget < floorB && pending > 0 && freeB / 2 >= floorB)
                    std::printf("[mode U] the kernels' stacks (%zu MB, reserved at their first launch) leave "
                                "the light-vertex store %zu MB of the %zu MB free\n",
                                pending >> 20, budget >> 20, freeB >> 20);
            } else {
                // The stacks do not fit whatever the store takes (-device gpu past the preflight, or
                // memory another program took since): they spill to host memory either way, so the
                // store keeps half the free memory rather than shrink to one subpath a pass.
                budget = std::min(std::max(totalB / 8, budget), freeB / 2);
                std::fprintf(stderr, "[mode U] the kernels' stacks (%zu MB) exceed the %zu MB free: they "
                                     "will spill to host memory, and the render crawl\n",
                             pending >> 20, freeB >> 20);
            }
        } else {
            cudaGetLastError();
        }
        // FTRACE_VCM_BUDGET_MB caps it, so a check can make a frame overflow on any card
        // (tools/check_vcm_store.py).
        if (const char* e = std::getenv("FTRACE_VCM_BUDGET_MB"))
            if (std::atoll(e) > 0) budget = std::min(budget, (size_t)std::atoll(e) << 20);
    }
    const bool keepHs = !scene.motionSets.empty() || !scene.vertexKeys.empty() || scene.needsCavity;
    // A subpath's share of everything the loop below allocates: per vertex, the slab and its compact
    // copy, their hit states, the cell key, the order index, the tag and two buckets of the table
    // (mergeSlotCount); per subpath, its vertex count, wavelengths and span.
    const size_t perPath = (size_t)s->vcmCap *
        (2 * sizeof(gpu::DVcmLV) + 2 * (size_t)s->secStride * sizeof(gpu::DVcmSec) + 4 * sizeof(int) + 1 +
         (keepHs ? 2 * sizeof(gpu::DVertHitState) : 0)) +
        3 * sizeof(int) + (size_t)C * (sizeof(gpu::Real) + sizeof(double));
    size_t nl = std::max<size_t>(1, std::min(np, budget / perPath));
    // Allocation failures cost light subpaths (variance), not the run: the budget is an estimate.
    auto tryAlloc = [](auto** p, size_t bytes) {
        if (cudaMalloc((void**)p, bytes) == cudaSuccess) return true;
        *p = nullptr;
        cudaGetLastError();
        return false;
    };
    if (!tryAlloc(&s->d_splat, np * 3 * sizeof(double)) ||
        !tryAlloc(&s->d_accum, np * 3 * sizeof(double))) {
        std::fprintf(stderr, "[mode U] no room on the GPU for a %dx%d film\n", resX, resY);
        vcmSessionEnd(s);
        return nullptr;
    }
    for (;;) {
        const size_t lv = nl * (size_t)s->vcmCap;
        if (tryAlloc(&s->d_lvSlab, lv * sizeof(gpu::DVcmLV)) &&
            (s->secStride == 0 ||
             tryAlloc(&s->d_lvSecSlab, lv * (size_t)s->secStride * sizeof(gpu::DVcmSec))) &&
            tryAlloc(&s->d_lvCount, nl * sizeof(int)) &&
            tryAlloc(&s->d_lamBuf, nl * (size_t)C * sizeof(gpu::Real)) &&
            tryAlloc(&s->d_invLam, nl * (size_t)C * sizeof(double)) &&
            tryAlloc(&s->d_pathBegin, nl * sizeof(int)) &&
            tryAlloc(&s->d_pathEnd, nl * sizeof(int)) &&
            // ...and the pass's compact copy, grid keys and bucket table at the most a pass can need
            // (every subpath storing every vertex), so no pass allocates them and none can fail to.
            tryAlloc(&s->d_lvCompact, lv * sizeof(gpu::DVcmLV)) &&
            (s->secStride == 0 ||
             tryAlloc(&s->d_lvSecCompact, lv * (size_t)s->secStride * sizeof(gpu::DVcmSec))) &&
            tryAlloc(&s->d_cellKey, lv * sizeof(int)) &&
            tryAlloc(&s->d_order, lv * sizeof(int)) &&
            tryAlloc(&s->d_cellStart, (mergeBucketCount(mergeSlotCount(lv)) + 1) * sizeof(int)) &&
            tryAlloc(&s->d_tag, lv) &&
            (!keepHs || (tryAlloc(&s->d_lvHsSlab, lv * sizeof(gpu::DVertHitState)) &&
                         tryAlloc(&s->d_lvHsCompact, lv * sizeof(gpu::DVertHitState))))) {
            s->lvCompactCap = lv;
            s->lvSecCompactCap = lv * (size_t)s->secStride;
            s->cellKeyCap = s->orderCap = lv;
            s->cellStartCap = mergeBucketCount(mergeSlotCount(lv)) + 1;
            s->tagCap = lv;
            break;
        }
        for (void* q : {(void*)s->d_lvSlab, (void*)s->d_lvSecSlab, (void*)s->d_lvCount,
                        (void*)s->d_lamBuf, (void*)s->d_invLam, (void*)s->d_pathBegin,
                        (void*)s->d_pathEnd, (void*)s->d_lvCompact, (void*)s->d_lvSecCompact,
                        (void*)s->d_cellKey, (void*)s->d_order, (void*)s->d_cellStart, (void*)s->d_tag,
                        (void*)s->d_lvHsSlab, (void*)s->d_lvHsCompact})
            cudaFree(q);
        s->d_lvSlab = nullptr; s->d_lvSecSlab = nullptr; s->d_lvCount = nullptr;
        s->d_lamBuf = nullptr; s->d_invLam = nullptr; s->d_pathBegin = nullptr; s->d_pathEnd = nullptr;
        s->d_lvCompact = nullptr; s->d_lvSecCompact = nullptr; s->d_cellKey = nullptr;
        s->d_order = nullptr; s->d_cellStart = nullptr; s->d_tag = nullptr; s->d_lvHsSlab = nullptr;
        s->d_lvHsCompact = nullptr;
        if (nl == 1) {
            std::fprintf(stderr, "[mode U] no room on the GPU for the light-vertex store\n");
            vcmSessionEnd(s);
            return nullptr;
        }
        nl /= 2;
    }
    s->nLight = nl;
    s->sortWorst = nl * (size_t)s->vcmCap * 24;
    if (nl < np)
        std::printf("[mode U] %zu light subpaths a pass for %zu pixels (light-vertex store "
                    "%zu MB of a %zu MB budget at %dx%d, %d vertices a subpath)\n",
                    nl, np, (nl * perPath) >> 20, budget >> 20, resX, resY, s->vcmCap);
    // THE MERGES' QUEUE (DVcmMergeQueue, VCM-WAVEFRONT). FTRACE_NOVCMWF=1 merges inline in the
    // camera kernel (the A/B control). Up to kVcmReqCap requests, or what an eighth of the memory
    // left beside the light-vertex store and the kernels' stacks holds; vcmSessionPass sizes its
    // waves to it, and a site past the requests is merged inline.
    if (!(std::getenv("FTRACE_NOVCMWF") && std::atoi(std::getenv("FTRACE_NOVCMWF")) != 0)) {
        const void* camFn = (C > 1) ? (const void*)gpu::kVcmCameraT<BDPT_NSEC> : (const void*)gpu::kVcmCameraT<0>;
        // The stacks are reserved at the kernels' first launch, after this, unless an earlier
        // launch's reservation already holds them; the sort's worst stays free too. (The grid's bucket
        // table was allocated above, at the most a pass can need.)
        const size_t pending = pendingLocalBytes(s->localNeed);
        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) != cudaSuccess) { cudaGetLastError(); freeB = 0; }
        const size_t room = (freeB > pending + s->sortWorst) ? freeB - pending - s->sortWorst : 0;
        // A wave holds at least the camera kernel's resident paths (no fewer than its launch where the
        // runtime cannot say), so the queue holds at least what they may ask for -- a site at each of
        // their vertices -- where the memory allows, past kVcmReqCap.
        const long long resident = residentThreadsOf(camFn);
        const long long least = (resident > 0) ? std::min<long long>(resident, 2048LL * 128) : 2048LL * 128;
        // FTRACE_VCMWFCAP=<n> (diagnostic) shrinks the queue to n requests (at least 16), so a small
        // frame takes several waves, or with a tiny n turns sites away every wave:
        // tools/check_vcm_wavefront.py requires the same image.
        bool shrunk = false;
        const long long cap = mergeQueueCap(room, sizeof(gpu::DVcmMergeReq) + (g_jSpecHost ? sizeof(gpu::DJSpec) : 0),
                                            std::max(kVcmReqCap, least * (long long)s->maxDepth), "FTRACE_VCMWFCAP",
                                            shrunk);
        const size_t nProf = (np + gpu::kVcmProfBlock - 1) / gpu::kVcmProfBlock;
        if (cap > 0 &&
            tryAlloc(&s->mq.reqs, (size_t)cap * sizeof(gpu::DVcmMergeReq)) &&
            (!g_jSpecHost || tryAlloc(&s->mq.spec, (size_t)cap * sizeof(gpu::DJSpec))) &&
            tryAlloc(&s->mq.nReqs, 2 * sizeof(int)) &&
            tryAlloc(&s->mq.profile, nProf * sizeof(int))) {
            s->mq.cap = (int)cap;
            s->mq.overflow = s->mq.nReqs + 1;
            s->mq.inlineMax = kVcmInlineMax;
            for (auto& e : s->mqEv) CUDA_CHECK(cudaEventCreate(&e));
            s->mqLeast = shrunk ? 256 : least;
        } else {
            for (void* q : {(void*)s->mq.reqs, (void*)s->mq.spec, (void*)s->mq.nReqs, (void*)s->mq.profile})
                cudaFree(q);
            s->mq = gpu::DVcmMergeQueue{};
            std::printf("[gpu] no room on the device for mode U's merge queue (%.0f MB free beside "
                        "the light-vertex store): the merges are gathered inline -- slower, the "
                        "same estimator\n", room / 1048576.0);
        }
    }
    CUDA_CHECK(cudaMemset(s->d_accum, 0, np * 3 * sizeof(double)));
    return s;
}

// Run one VCM pass at the given merge `radius`.
void vcmSessionPass(VcmSession* s, double radius) {
    using namespace gpu;
    dSetLenScale(s->up.sc);
    const long long passIdx = s->passes;                 // 0-based index of THIS pass
    // Motion blur (host twin vcmPass): the whole pass -- both subpaths, their connections and
    // merges -- at one time, stratified over the passes, the camera posed there, even for a camera
    // moving over a still scene, since the light pass connects to the camera too.
    const Camera& hc = s->hostCam;
    const bool passTimed = hc.seesMotion(*s->scene);
    const double tPass = passTimed ? hc.shutterTime(passTime((uint64_t)passIdx, 0x1B873593CC9E2D51ULL)) : 0.0;
    s->up.sc.mo.passU = passTimed ? devicePassU(*s->scene, tPass) : -1.0f;
    const DCamera camPass = passTimed ? posedBaked(s->cam, hc, tPass) : s->cam;
    const int W = s->resX, H = s->resY;
    const size_t np = s->npix, nl = s->nLight;
    if (radius <= 0.0) radius = 1e-6;

    // Per-pass MIS constants (device twin of vcm.h PassCtx). The light-path count is what the
    // merges and the camera splat are normalised by, so it is the pass's own (nLight).
    DVcmCtx ctx{};
    ctx.radius = radius;
    ctx.nLightPaths = (double)nl;
    double etaVCM = DPI * radius * radius * ctx.nLightPaths;
    ctx.misVcWeight = (etaVCM > 0.0) ? 1.0 / etaVCM : 0.0;
    ctx.misVmWeight = etaVCM;
    ctx.vmNorm      = (etaVCM > 0.0) ? 1.0 / etaVCM : 0.0;
    ctx.imagePlaneDist = (double)W / (2.0 * camPass.tanHalfX);
    ctx.maxDepth = s->maxDepth;

    // (1) Light pass — zero the per-pass splat, then trace nLight light subpaths.
    uploadSunAim();                       // the aimed sun (0.387.0), before any light subpath
    CUDA_CHECK(cudaMemset(s->d_splat, 0, np * 3 * sizeof(double)));
    unsigned long long seedL = 0xD1B54A32D192ED03ULL
                             ^ ((unsigned long long)(passIdx + 1) * 0x9E3779B97F4A7C15ULL)
                             ^ g_rngSalt;   // `-seed`; 0 by default, so XOR is a no-op
    if (s->secStride > 0)
        kVcmLightT<BDPT_NSEC><<<2048, 128>>>(s->up.sc, camPass, s->diffraction, ctx,
                                 s->d_lvSlab, s->d_lvSecSlab, s->secStride, s->heroC,
                                 s->d_lvCount, s->d_splat,
                                 s->d_lamBuf, s->d_invLam, W, H, s->vcmCap, (long long)nl,
                                 seedL, passIdx, s->d_lvHsSlab);
    else
        kVcmLightT<0><<<2048, 128>>>(s->up.sc, camPass, s->diffraction, ctx,
                                 s->d_lvSlab, nullptr, 0, 1,
                                 s->d_lvCount, s->d_splat,
                                 s->d_lamBuf, s->d_invLam, W, H, s->vcmCap, (long long)nl,
                                 seedL, passIdx, s->d_lvHsSlab);
    cudaCheckKernel("vcm-light");

    // (2) Compact the slab ON DEVICE into contiguous per-path ranges: exclusive-scan the
    // per-path vertex counts into pathBegin and inclusive-scan them into pathEnd
    // (pathEnd[i] = pathBegin[i] + count[i]; integer scans are exact in any order).
    // Only the 4-byte vertex total comes back to the host.
    s->arena.reset();
    ThrustArenaAlloc tal{&s->arena};
    auto pol = FT_THRUST_PAR(tal);
    thrust::device_ptr<int> tCount(s->d_lvCount), tBegin(s->d_pathBegin), tEnd(s->d_pathEnd);
    thrust::exclusive_scan(pol, tCount, tCount + nl, tBegin);
    thrust::inclusive_scan(pol, tCount, tCount + nl, tEnd);
    int nLVi = 0;
    CUDA_CHECK(cudaMemcpy(&nLVi, s->d_pathEnd + (nl - 1), sizeof(int), cudaMemcpyDeviceToHost));
    const size_t nLV = (size_t)((nLVi > 0) ? nLVi : 0);
    if (nLV > 0) {
        ensureDevCap(s->d_lvCompact, s->lvCompactCap, nLV);
        if (s->secStride > 0)
            ensureDevCap(s->d_lvSecCompact, s->lvSecCompactCap, nLV * (size_t)s->secStride);
        kVcmCompactScatter<<<2048, 128>>>(s->d_lvSlab, s->d_lvSecSlab, s->secStride,
                                          s->d_lvCount, s->d_pathBegin,
                                          s->d_lvCompact, s->d_lvSecCompact, (int)nl, s->vcmCap,
                                          s->d_lvHsSlab, s->d_lvHsCompact);
        cudaCheckKernel("vcm-compact");
    }

    // (3) Build the hashed lattice over the compacted light vertices ON DEVICE (merge_grid.h;
    // twin of vcm.h VcmGrid::build): the corner half a cell below the vertices, buckets from the
    // vertex count, and the STABLE sort by bucket equals the counting sort's output order, so the
    // merges meet each cell's vertices in the compact array's order.
    const double cell = radius;
    MergeLattice lat = mergeLatticeAt(0.0, 0.0, 0.0, cell, 1);
    if (nLV == 0) {   // the residue buckets alone, empty: a site reads 27 of them
        ensureDevCap(s->d_cellStart, s->cellStartCap, mergeBucketCount(1) + 1);
        CUDA_CHECK(cudaMemset(s->d_cellStart, 0, (mergeBucketCount(1) + 1) * sizeof(int)));
    } else {
        BboxF bb = thrust::transform_reduce(pol,
            thrust::device_pointer_cast(s->d_lvCompact),
            thrust::device_pointer_cast(s->d_lvCompact + nLV),
            LvToBboxF{}, BboxF{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX},
            BboxMergeF{});
        lat = mergeLatticeOver(bb.mnx, bb.mny, bb.mnz, cell, nLV);
        mergeBinDevice(pol, s->d_lvCompact, nLV, lat, 2048, s->d_cellKey, s->cellKeyCap, s->d_order, s->orderCap,
                       s->d_cellStart, s->cellStartCap, &s->d_tag, &s->tagCap);
        // The merges read a cell's vertices one after another (DVcmGrid::byCell), so they get them
        // in cell order, contiguous, where a lookup through `order` would scatter a site's
        // vertices over the whole store. The copy goes into the light slab, which holds at least
        // nLV vertices and which nothing reads again before the next pass's light walks refill it.
        if (nLV > nl * (size_t)s->vcmCap) {
            std::fprintf(stderr, "[mode U] %zu light vertices stored, past the slab's %zu\n",
                         nLV, nl * (size_t)s->vcmCap);
            std::exit(EXIT_FAILURE);
        }
        kVcmCellGather<<<2048, 128>>>(s->d_lvCompact, (s->secStride > 0) ? s->d_lvSecCompact : nullptr,
                                      s->secStride, s->d_order, (int)nLV, s->d_lvSlab, s->d_lvSecSlab);
        cudaCheckKernel("vcm-cellgather");
    }

    DVcmGrid grid{};
    grid.lv = (nLV > 0) ? s->d_lvCompact : nullptr;
    grid.nLV = (int)nLV;
    grid.cellStart = s->d_cellStart;
    grid.byCell = (nLV > 0) ? s->d_lvSlab : nullptr;
    grid.byCellSec = (nLV > 0 && s->secStride > 0) ? s->d_lvSecSlab : nullptr;
    grid.tag = s->d_tag;
    grid.lat = lat;

    // (5) Camera pass — one camera subpath per pixel, each paired with light subpath
    // (pixel % nLight); adds this pass's radiance into accum.
    unsigned long long seedC = 0xC2B2AE3D27D4EB4FULL
                             ^ ((unsigned long long)(passIdx + 1) * 0xA24BAED4963EE407ULL)
                             ^ g_rngSalt;   // `-seed`; 0 by default, so XOR is a no-op
    const DVcmSec* lvSec = (s->secStride > 0 && nLV > 0) ? s->d_lvSecCompact : nullptr;
    // A wave: the camera paths of pixels [p0, p1), their merges queued in `mq` (null reqs: inline).
    auto camWave = [&](long long p0, long long p1, const DVcmMergeQueue& mq) {
        if (s->secStride > 0)
            kVcmCameraT<BDPT_NSEC><<<2048, 128>>>(s->up.sc, camPass, s->diffraction, ctx, grid,
                                  lvSec, s->secStride, s->heroC,
                                  s->d_pathBegin, s->d_pathEnd, s->d_splat, s->d_accum,
                                  s->d_lamBuf, s->d_invLam, W, H, (long long)nl, seedC, passIdx,
                                  mq, p0, p1, (nLV > 0) ? s->d_lvHsCompact : nullptr);
        else
            kVcmCameraT<0><<<2048, 128>>>(s->up.sc, camPass, s->diffraction, ctx, grid,
                                  nullptr, 0, 1,
                                  s->d_pathBegin, s->d_pathEnd, s->d_splat, s->d_accum,
                                  s->d_lamBuf, s->d_invLam, W, H, (long long)nl, seedC, passIdx,
                                  mq, p0, p1, (nLV > 0) ? s->d_lvHsCompact : nullptr);
        cudaCheckKernel("vcm-camera");
    };
    // With the merges' queue, the pixels in waves whose merge sites it holds, each wave's sites
    // gathered a warp each after it (kVcmMergeGather). The last pass's profile (the sites each block
    // of kVcmProfBlock pixels asked the queue for: DVcmMergeQueue::profile) predicts this one's --
    // the same camera and scene, the radius a little smaller -- so the rest of a pass is split into
    // the fewest waves whose predicted sites, a quarter high, the queue holds, of like predicted
    // sites each: a band of sky takes no share of a wave, and the room below it is not sized by it.
    // A session's first pass has no profile: its first wave is mqLeast paths, and each next one as
    // many as the queue holds at the last wave's sites a path, a quarter high, and at least 2.5. No
    // wave is fewer than mqLeast paths, nor leaves fewer than that behind it. A pass's camera paths
    // are seeded by their pixel, so the waves leave every path as one launch traces it.
    const bool merging = ctx.vmNorm > 0.0 && nLV > 0;
    if (!s->mq.reqs || !merging) {
        camWave(0, (long long)np, DVcmMergeQueue{});
    } else {
        static const bool waveDebug = std::getenv("FTRACE_WAVE_DEBUG") != nullptr;   // a line a wave
        const double cap = (double)s->mq.cap;
        const long long B = kVcmProfBlock;
        const size_t nProf = (np + (size_t)B - 1) / (size_t)B;
        const bool profiled = s->mqProfile.size() == nProf;
        // The predicted sites of pixels [0, x): the last pass's profile summed, a block's count
        // spread evenly over its pixels.
        std::vector<double> sum(profiled ? nProf + 1 : 0, 0.0);
        for (size_t b = 0; profiled && b < nProf; ++b) sum[b + 1] = sum[b] + (double)s->mqProfile[b];
        auto sitesBefore = [&](long long x) -> double {
            const size_t b = (size_t)(x / B);
            if (b >= nProf) return sum[nProf];
            const long long len = std::min<long long>(B, (long long)np - (long long)b * B);
            return sum[b] + (double)s->mqProfile[b] * (double)(x - (long long)b * B) / (double)len;
        };
        auto waveEnd = [&](long long p0) -> long long {
            const long long left = (long long)np - p0;
            long long n;
            if (profiled) {
                const double at = sitesBefore(p0), rest = sum[nProf] - at;
                const double k = std::ceil(1.25 * rest / cap);
                if (k <= 1.0) return (long long)np;
                // The first pixel past which the wave holds its share of the rest.
                const double target = at + rest / k;
                long long lo = p0, hi = (long long)np;
                while (lo < hi) {
                    const long long mid = lo + (hi - lo) / 2;
                    if (sitesBefore(mid) >= target) hi = mid; else lo = mid + 1;
                }
                n = lo - p0;
            } else {
                n = (s->mqRate < 0.0) ? s->mqLeast : (long long)(cap / (1.25 * std::max(s->mqRate, 2.5)));
            }
            n = std::min(left, std::max(n, s->mqLeast));
            // A remainder too small to fill the device joins this wave (its excess sites merged
            // inline) rather than run as a wave of its own through its paths' longest tail.
            if (left - n < s->mqLeast) n = left;
            return p0 + n;
        };
        CUDA_CHECK(cudaMemset(s->mq.profile, 0, nProf * sizeof(int)));
        for (long long p0 = 0; p0 < (long long)np;) {
            const long long n = waveEnd(p0) - p0;
            CUDA_CHECK(cudaMemset(s->mq.nReqs, 0, 2 * sizeof(int)));
            camWave(p0, p0 + n, s->mq);
            int cnt[2] = {0, 0};
            CUDA_CHECK(cudaMemcpy(cnt, s->mq.nReqs, sizeof cnt, cudaMemcpyDeviceToHost));
            // nReqs counts the sites turned away too, so it can wrap past 2^31 of them.
            const long long demand = (cnt[0] < 0) ? (1LL << 31) : (long long)cnt[0];
            const long long gathered = std::min<long long>(demand, s->mq.cap);
            if (gathered > 0) {
                const int gBlocks = (int)std::min(4096LL, (gathered * 32 + 127) / 128);   // a warp a site
                CUDA_CHECK(cudaEventRecord(s->mqEv[0]));
                if (s->secStride > 0)
                    kVcmMergeGather<BDPT_NSEC><<<gBlocks, 128>>>(s->up.sc, ctx, grid, s->secStride,
                                                                 s->mq, (int)gathered, s->d_accum);
                else
                    kVcmMergeGather<0><<<gBlocks, 128>>>(s->up.sc, ctx, grid, 0,
                                                         s->mq, (int)gathered, s->d_accum);
                CUDA_CHECK(cudaEventRecord(s->mqEv[1]));
                cudaCheckKernel("vcm-merge-gather");
                float ms = 0.f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, s->mqEv[0], s->mqEv[1]));
                s->mqMsGather += ms;
            }
            if (waveDebug)
                std::printf("[wave] U pass %lld paths %lld..%lld (%lld), %lld merge sites (%d turned away)\n",
                            passIdx, p0, p0 + n, n, demand, cnt[1]);
            s->mqRate = (double)demand / (double)n;
            s->mqSites += gathered; s->mqInline += cnt[1]; s->mqPaths += n; s->mqWaves += 1;
            p0 += n;
        }
        s->mqProfile.resize(nProf);
        CUDA_CHECK(cudaMemcpy(s->mqProfile.data(), s->mq.profile, nProf * sizeof(int), cudaMemcpyDeviceToHost));
    }

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
    if (s->mqWaves > 0)
        std::printf("mode U merges: %lld sites gathered a warp each (%.2f a path) in %.2f s of GPU "
                    "time, %lld inline\n", s->mqSites,
                    s->mqPaths > 0 ? (double)(s->mqSites + s->mqInline) / (double)s->mqPaths : 0.0,
                    s->mqMsGather / 1000.0, s->mqInline);
    if (s->mq.reqs) cudaFree(s->mq.reqs);
    if (s->mq.spec) cudaFree(s->mq.spec);
    if (s->mq.nReqs) cudaFree(s->mq.nReqs);
    if (s->mq.profile) cudaFree(s->mq.profile);
    for (auto& e : s->mqEv) if (e) cudaEventDestroy(e);
    cudaFree(s->d_lvSlab); cudaFree(s->d_lvCount); cudaFree(s->d_splat); cudaFree(s->d_accum);
    cudaFree(s->d_lamBuf); cudaFree(s->d_invLam); cudaFree(s->d_pathBegin); cudaFree(s->d_pathEnd);
    if (s->d_lvSecSlab)    cudaFree(s->d_lvSecSlab);
    if (s->d_lvSecCompact) cudaFree(s->d_lvSecCompact);
    if (s->d_lvCompact) cudaFree(s->d_lvCompact);
    if (s->d_cellKey)   cudaFree(s->d_cellKey);
    if (s->d_cellStart) cudaFree(s->d_cellStart);
    if (s->d_tag)       cudaFree(s->d_tag);
    if (s->d_lvHsSlab)    cudaFree(s->d_lvHsSlab);
    if (s->d_lvHsCompact) cudaFree(s->d_lvHsCompact);
    if (s->d_order)     cudaFree(s->d_order);
    s->arena.release();
    freeUpload(s->up);
    delete s;
}
