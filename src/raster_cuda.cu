// raster_cuda.cu — GPU (CUDA) backend for the solid-shaded PREVIEW rasterizer. See
// raster_cuda.h for the scope and rationale.
//
// Pipeline (device twin of raster.h's deferred-visibility CPU rasterizer):
//
//   HOST  upload():  bake the world-space raster::PTri list + preview lights into POD
//                    device arrays (float3), once. Reused for every camera of a flyby.
//
//   Pass A  kProject  (1 thread / input triangle): transform to camera space, then either
//                     (rectilinear) near-plane clip (Sutherland-Hodgman against z=zn) and
//                     project the resulting fan into up to TWO screen sub-triangles, or
//                     (fisheye/panoramic) apply the same angular projRadius() lens map the
//                     real camera uses, reject-culling any triangle touching the rear pole
//                     and emitting ONE sub-triangle. Written to fixed slots [2*i], [2*i+1].
//                     The clip runs entirely in registers (the 8 in/out cases enumerate to
//                     at most 4 named vertices — no dynamically indexed local arrays, so
//                     nothing spills to local memory), sub-triangles that would be culled
//                     by rasterization anyway (empty clamped bbox / degenerate area — the
//                     exact setupSlot predicates) are dropped before storing anything,
//                     and validity/clear/clipped bits live in a DENSE per-slot `flags`
//                     array so invalidation is a coalesced 4-byte store per slot. Output
//                     is SPLIT per consumer: a 36B DGeo (screen geometry) always, plus a
//                     120B DAttr (shading attributes) ONLY for near-clipped slots — for
//                     unclipped slots the attributes are bit-verbatim DPTri fields, so
//                     the shade/clear passes read the source triangle instead and the
//                     projector skips those stores entirely.
//
//   Pass B  kClassify + kRasterSmall/Med/Large: bin every valid slot by clamped bbox
//                     pixel count, then rasterize each bin at a matching parallel width
//                     (small: 1 thread walks the bbox; medium: a warp strides the rows;
//                     large: a whole block strides the rows), packing (1/depth, slotIdx)
//                     into a 64-bit visibility buffer with a single atomicMax. Nearest
//                     surface (largest 1/depth) wins per pixel, in any scheduling order,
//                     so the binning is bit-identical to the old 1-thread-per-slot kernel
//                     while a screen-filling quad no longer serializes on one thread.
//                     Dead slots are skipped via the dense 4-byte `flags` probe (8 slots
//                     per 32-byte sector); live ones read only the dense 36B DGeo.
//                     The raster kernels read their bin count from device memory
//                     (small: upper-bound 1:1 grid; med/large: warp-/block-level
//                     ticket queues), so the host never reads the counts back and
//                     the whole frame enqueues without a mid-frame wait.
//
//   Pass C  kShade    (1 thread / pixel): decode the winning slot, recompute barycentrics
//                     at the pixel centre, interpolate world pos/normal, and shade once
//                     (the same ambient + Σ weighted N·L + headlight model as the CPU).
//                     Writes an HDR accum buffer + a 1/depth hit key + an emitter mask.
//
//   Pass D  kLumHist1/2 + kToneMap: the exposure/tonemap tail, ON the device (twin of
//                     raster::exposeAndEncodeT). The p99 auto-exposure anchor is found
//                     EXACTLY without sorting: qualifying luminances are non-negative
//                     floats, and non-negative IEEE floats order identically to their
//                     bit patterns, so two 65536-bin histogram rounds (top 16 bits, then
//                     low 16 bits within the winning bin) locate the same k-th order
//                     statistic the host's nth_element selects. kToneMap then applies
//                     the anchor in DOUBLE precision using explicit round-to-nearest
//                     intrinsics (__dmul_rn/__dadd_rn/__dsub_rn — no FMA contraction)
//                     and encodes through the shared raster::srgbLut8() table, so the
//                     IEEE operation sequence — and therefore every output byte — is
//                     the host tail's. Only the finished W*H*3 RGB8 image is downloaded
//                     (~4MB) instead of the ~26MB of HDR buffers the host tail needed.
//
// Everything geometric on the device runs in single precision (float): this is a
// solid-shaded preview, float is amply accurate for the geometry/shading, and it halves
// the per-slot and buffer memory. The exposure/tonemap that must match the host tail
// byte-for-byte runs in double precision on-device (Pass D above).
//
// Portable CUDA/HIP host runtime surface (mirrors render_cuda.cu): only the host runtime
// API is vendor-specific; the device language used here (grid-stride loops, atomicMax on
// unsigned long long, __float_as_uint) is identical under nvcc and hipcc.
#if defined(FTRACE_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define cudaError_t             hipError_t
  #define cudaSuccess             hipSuccess
  #define cudaGetDeviceCount      hipGetDeviceCount
  #define cudaMalloc              hipMalloc
  #define cudaMemcpy              hipMemcpy
  #define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
  #define cudaMemset              hipMemset
  #define cudaFree                hipFree
  #define cudaMallocHost          hipHostMalloc
  #define cudaFreeHost            hipHostFree
  #define cudaGetLastError        hipGetLastError
  #define cudaDeviceSynchronize   hipDeviceSynchronize
  #define cudaGetErrorString      hipGetErrorString
  #define cudaEvent_t             hipEvent_t
  #define cudaEventCreate         hipEventCreate
  #define cudaEventRecord         hipEventRecord
  #define cudaEventSynchronize    hipEventSynchronize
  #define cudaEventElapsedTime    hipEventElapsedTime
  #define cudaEventDestroy        hipEventDestroy
  // NOTE: kRasterMed's ticket broadcast uses __shfl_sync (32-wide). No alias on
  // purpose: a HIP build fails loudly there instead of silently mis-broadcasting on
  // wave64 GPUs — port it to a wave-size-aware __shfl if HIP is ever actually built.
#else
  #include <cuda_runtime.h>
#endif

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>

#include "raster_cuda.h"
#include "camera.h"    // Camera, CAM_RECTILINEAR

namespace raster_cuda {

// ---------------------------------------------------------------------------
// POD device structs (float). One preview triangle, one distilled light, the camera
// frame, and a projected/clipped screen sub-triangle.
struct DPTri {
    float3 p0, p1, p2;
    float3 n0, n1, n2;
    float3 color;
    float2 uv0, uv1, uv2;   // per-vertex texture coords (only meaningful when tex >= 0)
    int    tex;             // bound skin texture index, or -1 (use flat `color`)
    float  triplanarScale;  // >0: sample the skin by world triplanar instead of UV
    int    emissive;
    int    clear;           // see-through transmissive surface (handled by the clear pass)
};

// A device image texture (linear RGB), mirroring raster.h's use of Texture::sampleRgb /
// sampleRgbTriplanar for the solid preview. All textures' texels are flattened into one
// shared device array; `offset` is this texture's first texel. `valid`=0 => sample gray.
struct DTex {
    int w, h;
    int filter;   // 0 = nearest, 1 = bilinear
    int wrap;     // 0 = repeat, 1 = clamp, 2 = mirror
    int offset;   // start index into the shared texel array
    int valid;    // w > 0 && h > 0 && texels present
};

struct DLight {
    float3 pos, dir;
    int    spot;
    float  cosInner, cosOuter, weight, falloff2;
};

struct DCam {
    float3 eye, u, v, w;   // right, up, forward (orthonormal)
    float  tanHalfX, tanHalfY;
    int    projection;     // CAM_RECTILINEAR (0) or a fisheye/panoramic lens map
    float  rEdge;          // image radius at the vertical film edge (angular projections)
};

// A projected screen sub-triangle, SPLIT into what each pass actually reads:
//
//   DGeo  (36B, one per slot, always written): the screen-space geometry — the ONLY
//         thing the classify/raster passes touch, kept dense so their per-slot reads
//         cover 2 cache sectors instead of scattering across a fat combined record.
//   DAttr (120B, one per slot, written ONLY for near-plane-CLIPPED slots): the shading
//         attributes. For every unclipped slot these are bit-verbatim copies of the
//         source DPTri's fields (the projection chain never does arithmetic on them),
//         so kShade/kClear read them straight from tris[slot >> 1] instead — which
//         lets kProject skip ~120B of stores per slot for the overwhelming majority
//         of triangles. Only slots holding lerped (clipped) vertices store a DAttr,
//         marked by kSlotClipped in the flags array.
//
// Validity/clear/clipped bits live in a dense per-slot `flags` int array: the
// classify/clear passes probe every slot each frame, and a dense 4-byte probe touches
// 8 slots per 32B sector — and kProject's per-frame slot invalidation is coalesced.
struct DGeo {
    float sx0, sy0, invd0;
    float sx1, sy1, invd1;
    float sx2, sy2, invd2;
};
struct DAttr {
    float3 wp0, wn0; float2 uv0;
    float3 wp1, wn1; float2 uv1;
    float3 wp2, wn2; float2 uv2;
    float3 color;
    int    tex;             // skin texture index, or -1
    float  triplanarScale;  // >0: sample by world triplanar instead of UV
    int    emissive;
};
// Per-slot flags array values (kProject writes, classify/raster/shade/clear probe).
constexpr int kSlotValid   = 1;   // bit0: slot holds a projected sub-triangle
constexpr int kSlotClear   = 2;   // bit1: see-through transmissive surface
constexpr int kSlotClipped = 4;   // bit2: verts were lerped by the near clip -> attrs in DAttr

// A camera-space vertex carrying the interpolated attributes (mirrors raster::VtxCS).
struct DVtxCS { float x, y, z; float3 wpos, wn; float2 uv; };

// ---------------------------------------------------------------------------
// Device vector helpers (self-contained; no host header is __device__-annotated).
__host__ __device__ inline float3 mk(float x, float y, float z) { return make_float3(x, y, z); }
__device__ inline float3 operator+(float3 a, float3 b) { return mk(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ inline float3 operator-(float3 a, float3 b) { return mk(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ inline float3 operator*(float3 a, float s)  { return mk(a.x*s, a.y*s, a.z*s); }
__device__ inline float  dot3(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline float3 normalize3(float3 a) {
    float L = sqrtf(dot3(a, a));
    return (L > 1e-12f) ? a * (1.0f / L) : a;
}

// spotFalloff (smoothstep between the inner/outer cone cosines) — mirrors scene.h.
__device__ inline float spotFalloffD(float ct, float cosInner, float cosOuter) {
    if (ct >= cosInner) return 1.0f;
    if (ct <= cosOuter) return 0.0f;
    float t = (ct - cosOuter) / (cosInner - cosOuter);
    return t * t * (3.0f - 2.0f * t);
}

// World -> camera space for one vertex (mirrors raster::projectRange's toCS).
__device__ inline DVtxCS toCS(const DCam& cam, float3 P, float3 Nn, float2 UV) {
    float3 d = P - cam.eye;
    DVtxCS c; c.x = dot3(d, cam.u); c.y = dot3(d, cam.v); c.z = dot3(d, cam.w);
    c.wpos = P; c.wn = Nn; c.uv = UV; return c;
}

// Linear interpolation of two camera-space vertices (near-plane clip).
__device__ inline DVtxCS lerpVtx(const DVtxCS& a, const DVtxCS& b, float s) {
    DVtxCS r;
    r.x = a.x + (b.x - a.x)*s; r.y = a.y + (b.y - a.y)*s; r.z = a.z + (b.z - a.z)*s;
    r.wpos = a.wpos + (b.wpos - a.wpos)*s; r.wn = a.wn + (b.wn - a.wn)*s;
    r.uv = make_float2(a.uv.x + (b.uv.x - a.uv.x)*s, a.uv.y + (b.uv.y - a.uv.y)*s);
    return r;
}

// Atomic multiplicative update on a float (order-independent product accumulation for the
// see-through clear pass, where several clear fragments may cover the same pixel). CAS loop.
__device__ inline void atomicMulF(float* addr, float val) {
    unsigned int* ua = (unsigned int*)addr;
    unsigned int old = *ua, assumed;
    do {
        assumed = old;
        float f = __uint_as_float(assumed) * val;
        old = atomicCAS(ua, assumed, __float_as_uint(f));
    } while (old != assumed);
}

// Wrap a texel index into [0,n) per the texture's wrap mode (mirrors Texture::wrapIndex).
__device__ inline int wrapIdxD(int i, int n, int wrap) {
    if (wrap == 1) return (i < 0) ? 0 : (i >= n ? n - 1 : i);        // clamp
    if (wrap == 2) {                                                 // mirror
        int period = 2 * n;
        int m = ((i % period) + period) % period;
        return (m < n) ? m : (period - 1 - m);
    }
    int m = i % n; return (m < 0) ? m + n : m;                       // repeat
}

// Sample a device texture's LINEAR RGB at (u,v). Exact device twin of Texture::sampleRgb:
// nearest or bilinear, v flipped (v=0 => bottom row), with the texture's wrap mode.
__device__ inline float3 dSampleRgb(const DTex* meta, const float3* texels, int ti,
                                    float u, float v) {
    const DTex& t = meta[ti];
    if (!t.valid) return mk(0.5f, 0.5f, 0.5f);
    const float3* px = texels + t.offset;
    if (t.filter == 0) {   // nearest
        int x = wrapIdxD((int)floorf(u * t.w), t.w, t.wrap);
        int y = wrapIdxD((int)floorf((1.0f - v) * t.h), t.h, t.wrap);
        return px[(size_t)y * t.w + x];
    }
    float tu = u * t.w - 0.5f;
    float tv = (1.0f - v) * t.h - 0.5f;
    float flx = floorf(tu), fly = floorf(tv);
    float fx = tu - flx, fy = tv - fly;
    int x0 = wrapIdxD((int)flx, t.w, t.wrap), x1 = wrapIdxD((int)flx + 1, t.w, t.wrap);
    int y0 = wrapIdxD((int)fly, t.h, t.wrap), y1 = wrapIdxD((int)fly + 1, t.h, t.wrap);
    float3 c00 = px[(size_t)y0 * t.w + x0], c10 = px[(size_t)y0 * t.w + x1];
    float3 c01 = px[(size_t)y1 * t.w + x0], c11 = px[(size_t)y1 * t.w + x1];
    float3 a = c00 * (1.0f - fx) + c10 * fx;
    float3 b = c01 * (1.0f - fx) + c11 * fx;
    return a * (1.0f - fy) + b * fy;
}

// Triplanar (box) projection of the LINEAR RGB albedo at a world hit — device twin of
// Texture::sampleRgbTriplanar. |n|^4 componentwise axis blend of the three world planes.
__device__ inline float3 dSampleRgbTri(const DTex* meta, const float3* texels, int ti,
                                       float3 p, float3 n, float scale) {
    float ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z);
    float wx = ax*ax*ax*ax, wy = ay*ay*ay*ay, wz = az*az*az*az;
    float s = wx + wy + wz;
    if (s <= 0.0f) return dSampleRgb(meta, texels, ti, p.x * scale, p.y * scale);
    wx /= s; wy /= s; wz /= s;
    float3 c = mk(0, 0, 0);
    if (wx > 0.0f) c = c + dSampleRgb(meta, texels, ti, p.z * scale, p.y * scale) * wx;
    if (wy > 0.0f) c = c + dSampleRgb(meta, texels, ti, p.x * scale, p.z * scale) * wy;
    if (wz > 0.0f) c = c + dSampleRgb(meta, texels, ti, p.x * scale, p.y * scale) * wz;
    return c;
}

// Image radius for a ray at angle `th` from the optical axis (mirrors camera.h projRadius).
// The enum values (CAM_EQUIDISTANT..CAM_ORTHOGRAPHIC) come from camera.h, included above.
__device__ inline float projRadiusD(int proj, float th) {
    switch (proj) {
        case CAM_EQUIDISTANT:   return th;
        case CAM_EQUISOLID:     return 2.0f * sinf(0.5f * th);
        case CAM_STEREOGRAPHIC: return 2.0f * tanf(0.5f * th);
        case CAM_ORTHOGRAPHIC:  return sinf(th);
        default:                return tanf(th);            // CAM_RECTILINEAR
    }
}

// Project a camera-space vertex to the raster. Mirrors raster::projectVtx exactly: the
// rectilinear pinhole is the inverse of Camera::genRay; a fisheye/panoramic lens applies the
// same angular projRadius() map (so off-axis stretch matches). sx in [0,W], sy=0 at image
// top, invd=1/depth (camera-forward distance for rectilinear, ray length for angular).
__device__ inline void projectVtxG(const DCam& cam, const DVtxCS& v, int W, int H,
                                    float& sx, float& sy, float& invd,
                                    float3& wp, float3& wn, float2& uv) {
    float ndcx, ndcy, depth;
    if (cam.projection == CAM_RECTILINEAR) {
        ndcx = (v.x / v.z) / cam.tanHalfX;
        ndcy = (v.y / v.z) / cam.tanHalfY;
        depth = v.z;                                        // camera-forward distance
    } else {
        float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
        float costh = (len > 1e-12f) ? v.z / len : 1.0f;
        costh = fminf(1.0f, fmaxf(-1.0f, costh));
        float th = acosf(costh);
        float rho = projRadiusD(cam.projection, th) / fmaxf(cam.rEdge, 1e-12f);
        float rhoDir = sqrtf(v.x*v.x + v.y*v.y);
        if (rhoDir < 1e-12f) { ndcx = 0.0f; ndcy = 0.0f; }
        else { ndcx = rho * v.x / rhoDir; ndcy = rho * v.y / rhoDir; }
        depth = len;                                        // ray length (angular lens)
    }
    sx = (ndcx * 0.5f + 0.5f) * W;
    sy = (0.5f - 0.5f * ndcy) * H;                          // +y (up) -> top of image
    invd = 1.0f / fmaxf(depth, 1e-9f);
    wp = v.wpos; wn = v.wn; uv = v.uv;
}

// ---------------------------------------------------------------------------
// Pass A: transform + project each input triangle into up to two slots. Rectilinear does a
// Sutherland-Hodgman near-plane clip + fan (up to 2 sub-triangles); a fisheye/panoramic lens
// instead rejects any triangle that reaches (nearly) behind the camera and projects the
// three vertices straight through the angular map into a single sub-triangle. This mirrors
// raster::projectRange's two branches exactly, so the GPU and CPU rasters agree per camera.
//
// The whole clip lives in REGISTERS: clipping a triangle against ONE plane yields at most
// four vertices, so the edge walk's eight in/out cases are enumerated explicitly into
// named locals (q0..q3) instead of dynamically-indexed staging arrays, which nvcc would
// spill to per-thread local memory — that spill traffic, times millions of triangles, was
// the bulk of this pass's cost. Each case emits the EXACT vertex sequence the array walk
// produced, so every stored float is unchanged.

// One projected vertex (screen x/y, 1/depth, world pos/normal, uv) — projectVtxG's
// outputs bundled so the fan pieces can be assembled from named registers.
struct PV { float sx, sy, invd; float3 wp, wn; float2 uv; };

__device__ inline PV projectPV(const DCam& cam, const DVtxCS& v, int W, int H) {
    PV p;
    projectVtxG(cam, v, W, H, p.sx, p.sy, p.invd, p.wp, p.wn, p.uv);
    return p;
}

// Near-plane crossing point of edge A->B (the old inline lerp, verbatim).
__device__ inline DVtxCS clipNear(const DVtxCS& A, const DVtxCS& B, float zn) {
    float s = (zn - A.z) / (B.z - A.z);
    return lerpVtx(A, B, s);
}

// Store one fan piece into slot `idx` — unless it could never touch a pixel. The bbox
// and degenerate-area predicates are setupSlot's EXACT arithmetic (kClear applies the
// same two rejections), so skipping the stores for off-screen / degenerate pieces is
// invisible to every downstream pass; the slot's flags simply stay 0. The shading
// attributes (DAttr) are stored ONLY when `clipped` — otherwise every attribute is a
// bit-verbatim copy of tris[idx >> 1]'s fields and the shade/clear passes read the
// source triangle directly.
__device__ inline void emitSlot(DGeo* geos, DAttr* attrs, int* flags, int idx,
                                const DPTri& t, int W, int H, bool clipped,
                                const PV& A, const PV& B, const PV& C) {
    float minx = floorf(fminf(A.sx, fminf(B.sx, C.sx)));
    float maxx = ceilf (fmaxf(A.sx, fmaxf(B.sx, C.sx)));
    float miny = floorf(fminf(A.sy, fminf(B.sy, C.sy)));
    float maxy = ceilf (fmaxf(A.sy, fmaxf(B.sy, C.sy)));
    int xlo = max(0, (int)minx), xhi = min(W - 1, (int)maxx);
    int ylo = max(0, (int)miny), yhi = min(H - 1, (int)maxy);
    if (xlo > xhi || ylo > yhi) return;
    float area = (B.sx - A.sx) * (C.sy - A.sy) - (B.sy - A.sy) * (C.sx - A.sx);
    if (fabsf(area) < 1e-9f) return;
    DGeo g;
    g.sx0 = A.sx; g.sy0 = A.sy; g.invd0 = A.invd;
    g.sx1 = B.sx; g.sy1 = B.sy; g.invd1 = B.invd;
    g.sx2 = C.sx; g.sy2 = C.sy; g.invd2 = C.invd;
    geos[idx] = g;
    if (clipped) {
        DAttr a;
        a.wp0 = A.wp; a.wn0 = A.wn; a.uv0 = A.uv;
        a.wp1 = B.wp; a.wn1 = B.wn; a.uv1 = B.uv;
        a.wp2 = C.wp; a.wn2 = C.wn; a.uv2 = C.uv;
        a.color = t.color; a.tex = t.tex; a.triplanarScale = t.triplanarScale;
        a.emissive = t.emissive;
        attrs[idx] = a;
    }
    flags[idx] = kSlotValid | (t.clear ? kSlotClear : 0) | (clipped ? kSlotClipped : 0);
}

__global__ void kProject(const DPTri* tris, int nTris, DCam cam, int W, int H,
                         DGeo* geos, DAttr* attrs, int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nTris) return;
    const DPTri& t = tris[i];
    // Invalidate both output slots up front (dense, coalesced — adjacent threads write
    // adjacent flag pairs).
    flags[2*i] = 0; flags[2*i+1] = 0;

    // World -> camera space.
    DVtxCS c0 = toCS(cam, t.p0, t.n0, t.uv0);
    DVtxCS c1 = toCS(cam, t.p1, t.n1, t.uv1);
    DVtxCS c2 = toCS(cam, t.p2, t.n2, t.uv2);

    if (cam.projection != CAM_RECTILINEAR) {
        // Fisheye/panoramic: no near-plane clip. Reject the triangle if any vertex points
        // (nearly) backward (z <= -0.999*len), else project all three straight through the
        // angular lens map into ONE sub-triangle (slot [2*i]; [2*i+1] stays invalid).
        float l0 = sqrtf(c0.x*c0.x + c0.y*c0.y + c0.z*c0.z);
        if (c0.z <= -0.999f * l0) return;
        float l1 = sqrtf(c1.x*c1.x + c1.y*c1.y + c1.z*c1.z);
        if (c1.z <= -0.999f * l1) return;
        float l2 = sqrtf(c2.x*c2.x + c2.y*c2.y + c2.z*c2.z);
        if (c2.z <= -0.999f * l2) return;
        PV A = projectPV(cam, c0, W, H);
        PV B = projectPV(cam, c1, W, H);
        PV C = projectPV(cam, c2, W, H);
        emitSlot(geos, attrs, flags, 2*i, t, W, H, /*clipped=*/false, A, B, C);
        return;
    }

    // Rectilinear: Sutherland-Hodgman clip against the near plane z=zn, in registers.
    // The edge walk (e=0,1,2: emit A if inside, emit the crossing if the edge straddles)
    // emits, per in/out mask, exactly the sequences enumerated below.
    const float zn = 1e-3f;
    int mask = (c0.z > zn ? 1 : 0) | (c1.z > zn ? 2 : 0) | (c2.z > zn ? 4 : 0);
    if (mask == 0) return;                       // fully behind the near plane
    DVtxCS q0 = c0, q1 = c1, q2 = c2, q3 = c2;   // defaults; q3 only read when np == 4
    int np = 3;
    switch (mask) {
        case 7:                                                                  break; // [A0 A1 A2]
        case 1: q1 = clipNear(c0, c1, zn); q2 = clipNear(c2, c0, zn);            break; // [A0 X01 X20]
        case 2: q0 = clipNear(c0, c1, zn); q2 = clipNear(c1, c2, zn);            break; // [X01 A1 X12]
        case 4: q0 = clipNear(c1, c2, zn); q1 = c2; q2 = clipNear(c2, c0, zn);   break; // [X12 A2 X20]
        case 3: q2 = clipNear(c1, c2, zn); q3 = clipNear(c2, c0, zn); np = 4;    break; // [A0 A1 X12 X20]
        case 5: q1 = clipNear(c0, c1, zn); q2 = clipNear(c1, c2, zn); np = 4;    break; // [A0 X01 X12 A2]
        case 6: q0 = clipNear(c0, c1, zn); q3 = clipNear(c2, c0, zn); np = 4;    break; // [X01 A1 A2 X20]
    }

    // Fan: (q0,q1,q2) then, for a quad, (q0,q2,q3) — the array walk's k=1,2 pieces.
    // mask==7 means no vertex was lerped, so the slot's attributes are verbatim DPTri
    // fields and no DAttr store is needed; any other mask mixes in clip crossings.
    bool clipped = (mask != 7);
    PV A = projectPV(cam, q0, W, H);
    PV B = projectPV(cam, q1, W, H);
    PV C = projectPV(cam, q2, W, H);
    emitSlot(geos, attrs, flags, 2*i + 0, t, W, H, clipped, A, B, C);
    if (np == 4) {
        PV D = projectPV(cam, q3, W, H);
        emitSlot(geos, attrs, flags, 2*i + 1, t, W, H, clipped, A, C, D);
    }
}

// ---------------------------------------------------------------------------
// Pass B: rasterize each valid slot into the 64-bit visibility buffer. Each covered pixel
// packs (1/depth as float bits) << 32 | slotIdx; atomicMax keeps the nearest (largest
// 1/depth) surface. Mirrors fillTriangleG's barycentric coverage + perspective 1/depth.
// Per-slot rasterization setup: cull checks, clamped pixel bbox, area/derivatives.
// This is the exact preamble of the old monolithic kRaster, factored out so the
// classifier and all three binned kernels compute identical values.
struct SlotSetup { int xlo, xhi, ylo, yhi; float inv, dw0dx, dw1dx; };

__device__ inline bool setupSlot(const DGeo& t, int flg, int W, int H, int seeThrough, SlotSetup& s) {
    if (!(flg & kSlotValid)) return false;
    if (seeThrough && (flg & kSlotClear)) return false;   // clear surfaces handled by the clear-accumulation pass
    float minx = floorf(fminf(t.sx0, fminf(t.sx1, t.sx2)));
    float maxx = ceilf (fmaxf(t.sx0, fmaxf(t.sx1, t.sx2)));
    float miny = floorf(fminf(t.sy0, fminf(t.sy1, t.sy2)));
    float maxy = ceilf (fmaxf(t.sy0, fmaxf(t.sy1, t.sy2)));
    s.xlo = max(0, (int)minx); s.xhi = min(W - 1, (int)maxx);
    s.ylo = max(0, (int)miny); s.yhi = min(H - 1, (int)maxy);
    if (s.xlo > s.xhi || s.ylo > s.yhi) return false;
    float area = (t.sx1 - t.sx0) * (t.sy2 - t.sy0) - (t.sy1 - t.sy0) * (t.sx2 - t.sx0);
    if (fabsf(area) < 1e-9f) return false;
    s.inv = 1.0f / area;
    s.dw0dx = (t.sy1 - t.sy2) * s.inv;
    s.dw1dx = (t.sy2 - t.sy0) * s.inv;
    return true;
}

// Rasterize ONE bbox row of one sub-triangle: seed the barycentrics at the row's left
// edge by direct evaluation (exactly as the old kernel did per row) and step
// incrementally along x. The float arithmetic per (slot,row) is identical no matter
// which thread executes it, and the atomicMax visibility merge is order-independent,
// so any distribution of rows across threads yields bit-identical output.
__device__ inline void rasterRow(const DGeo& t, int slot, int y, const SlotSetup& s,
                                 int W, unsigned long long* vis) {
    float py = y + 0.5f, pxL = s.xlo + 0.5f;
    float w0 = ((t.sx1 - pxL) * (t.sy2 - py) - (t.sy1 - py) * (t.sx2 - pxL)) * s.inv;
    float w1 = ((t.sx2 - pxL) * (t.sy0 - py) - (t.sy2 - py) * (t.sx0 - pxL)) * s.inv;
    unsigned long long row = (unsigned long long)y * W + s.xlo;
    for (int x = s.xlo; x <= s.xhi; ++x, ++row, w0 += s.dw0dx, w1 += s.dw1dx) {
        float w2 = 1.0f - w0 - w1;
        if (w0 < 0 || w1 < 0 || w2 < 0) continue;
        float invd = w0 * t.invd0 + w1 * t.invd1 + w2 * t.invd2;   // 1/depth
        if (invd <= 0.0f) continue;
        unsigned long long packed =
            ((unsigned long long)__float_as_uint(invd) << 32) | (unsigned int)slot;
        atomicMax(&vis[row], packed);
    }
}

// Bin thresholds (clamped bbox pixel count). ≤ kSmallMaxPx: one thread walks the whole
// bbox (the common case for finely tessellated scenes). ≤ kMedMaxPx: a 32-lane warp
// strides the bbox rows. Larger: a whole block strides the rows — a full-screen wall
// quad's bbox is walked by 256 threads instead of serializing on one.
constexpr long long kSmallMaxPx = 128;
constexpr long long kMedMaxPx   = 16384;

__global__ void kClassify(const DGeo* geos, const int* flags, int nSlots, int W, int H,
                          int seeThrough, int* binSmall, int* binMed, int* binLarge, int* cnt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nSlots) return;
    int flg = flags[idx];                       // dense 4B probe: skip dead slots without
    if (!(flg & kSlotValid)) return;            // touching the DGeo record at all
    SlotSetup s;
    if (!setupSlot(geos[idx], flg, W, H, seeThrough, s)) return;   // culled: nothing to raster
    long long px = (long long)(s.xhi - s.xlo + 1) * (s.yhi - s.ylo + 1);
    if      (px <= kSmallMaxPx) binSmall[atomicAdd(&cnt[0], 1)] = idx;
    else if (px <= kMedMaxPx)   binMed  [atomicAdd(&cnt[1], 1)] = idx;
    else                        binLarge[atomicAdd(&cnt[2], 1)] = idx;
}

// The three binned kernels read their bin's count from device memory (cnt[0..2],
// written by kClassify), so the host never has to read the counts back to size the
// grids — on WDDM that readback was the frame's only mid-frame flush+wait. The small
// bin (millions of items in tessellated scenes) keeps its 1:1 thread↔item mapping
// under an upper-bound grid (excess threads exit after one cached load) because the
// hardware block scheduler load-balances the variable per-item cost far better than
// a grid-stride loop; med warps and large blocks take tickets from work queues
// (cnt[3]/cnt[4]) so their variable-cost items stay dynamically balanced. In every
// case each (slot,row) executes the exact old arithmetic and atomicMax merges
// order-independently, so the output stays bit-identical no matter which thread
// runs it.
__global__ void kRasterSmall(const DGeo* geos, const int* flags, const int* list, const int* cnt,
                             int W, int H, int seeThrough, unsigned long long* vis) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= cnt[0]) return;
    int slot = list[li];
    const DGeo& t = geos[slot];
    SlotSetup s;
    if (!setupSlot(t, flags[slot], W, H, seeThrough, s)) return;
    for (int y = s.ylo; y <= s.yhi; ++y)
        rasterRow(t, slot, y, s, W, vis);
}

// Med items are heavy (128..16K px) and can number far beyond the fixed grid's warps,
// so a static stride would leave straggler warps serially walking several of them.
// Instead the warps take tickets from a work queue (cnt[3], zeroed with the counts):
// one cheap atomicAdd per ITEM keeps every warp busy until the list is drained —
// the same dynamic balancing the hardware gave the old exact-sized 1:1 launch.
__global__ void kRasterMed(const DGeo* geos, const int* flags, const int* list, int* cnt,
                           int W, int H, int seeThrough, unsigned long long* vis) {
    const int n = cnt[1];
    int lane = threadIdx.x & 31;
    for (;;) {
        int li;
        if (lane == 0) li = atomicAdd(&cnt[3], 1);
        li = __shfl_sync(0xffffffffu, li, 0);
        if (li >= n) return;
        int slot = list[li];
        const DGeo& t = geos[slot];
        SlotSetup s;
        if (!setupSlot(t, flags[slot], W, H, seeThrough, s)) continue;
        for (int y = s.ylo + lane; y <= s.yhi; y += 32)
            rasterRow(t, slot, y, s, W, vis);
    }
}

// Large items are the heaviest of all (16K..screen-sized bboxes), so like med they
// self-balance through a ticket queue (cnt[4]) instead of a static stride: thread 0
// takes the block's next ticket and broadcasts it through shared memory. Every branch
// below is uniform across the block (all threads see the same li/slot), so the
// barriers can't diverge.
__global__ void kRasterLarge(const DGeo* geos, const int* flags, const int* list, int* cnt,
                             int W, int H, int seeThrough, unsigned long long* vis) {
    const int n = cnt[2];
    __shared__ int sLi;
    for (;;) {
        if (threadIdx.x == 0) sLi = atomicAdd(&cnt[4], 1);
        __syncthreads();                 // ticket visible to the whole block
        int li = sLi;
        __syncthreads();                 // everyone has copied it before the next write
        if (li >= n) return;
        int slot = list[li];
        const DGeo& t = geos[slot];
        SlotSetup s;
        if (!setupSlot(t, flags[slot], W, H, seeThrough, s)) continue;
        for (int y = s.ylo + threadIdx.x; y <= s.yhi; y += blockDim.x)
            rasterRow(t, slot, y, s, W, vis);
    }
}

// ---------------------------------------------------------------------------
// Pass C: resolve + shade each pixel once. Decode the winning slot, recompute barycentrics
// at the pixel centre (same float math as kRaster, so the winner's 1/depth reproduces),
// interpolate world pos/normal, and shade with the same model as raster.h Pass 3.
__global__ void kShade(const DPTri* tris, const DGeo* geos, const DAttr* attrs,
                       const int* flags, const unsigned long long* vis,
                       const DLight* lights, int nLights,
                       float ambient, float keyScale, float fill,
                       DCam cam, int W, int H, float3 bg, float emisBoost,
                       const DTex* texMeta, const float3* texels, int nTex,
                       float3* accum, float* zbuf, unsigned char* emis) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)W * H) return;
    unsigned long long v = vis[i];
    if (v == 0ULL) { accum[i] = bg; zbuf[i] = 0.0f; emis[i] = 0; return; }
    int slot = (int)(unsigned int)(v & 0xffffffffULL);
    const DGeo& t = geos[slot];

    // Recompute barycentrics at this pixel's centre.
    int px = (int)(i % (unsigned long long)W);
    int py = (int)(i / (unsigned long long)W);
    float fx = px + 0.5f, fy = py + 0.5f;
    float area = (t.sx1 - t.sx0) * (t.sy2 - t.sy0) - (t.sy1 - t.sy0) * (t.sx2 - t.sx0);
    float inv = (fabsf(area) > 1e-12f) ? 1.0f / area : 0.0f;
    float w0 = ((t.sx1 - fx) * (t.sy2 - fy) - (t.sy1 - fy) * (t.sx2 - fx)) * inv;
    float w1 = ((t.sx2 - fx) * (t.sy0 - fy) - (t.sy2 - fy) * (t.sx0 - fx)) * inv;
    float w2 = 1.0f - w0 - w1;
    float invd = w0 * t.invd0 + w1 * t.invd1 + w2 * t.invd2;
    zbuf[i] = invd;

    // Shading attributes: bit-verbatim from the source triangle unless this slot's
    // vertices were lerped by the near clip (then the DAttr store holds them).
    float3 wp0, wp1, wp2, wn0, wn1, wn2, color;
    float2 uv0, uv1, uv2;
    int tex, emissive; float tps;
    if (flags[slot] & kSlotClipped) {
        const DAttr& a = attrs[slot];
        wp0 = a.wp0; wn0 = a.wn0; uv0 = a.uv0;
        wp1 = a.wp1; wn1 = a.wn1; uv1 = a.uv1;
        wp2 = a.wp2; wn2 = a.wn2; uv2 = a.uv2;
        color = a.color; tex = a.tex; tps = a.triplanarScale; emissive = a.emissive;
    } else {
        const DPTri& s = tris[slot >> 1];
        wp0 = s.p0; wn0 = s.n0; uv0 = s.uv0;
        wp1 = s.p1; wn1 = s.n1; uv1 = s.uv1;
        wp2 = s.p2; wn2 = s.n2; uv2 = s.uv2;
        color = s.color; tex = s.tex; tps = s.triplanarScale; emissive = s.emissive;
    }
    emis[i] = emissive ? 1 : 0;
    if (emissive) { accum[i] = color * emisBoost; return; }   // raw emitter radiance

    // Perspective-correct world pos / normal.
    float d = 1.0f / fmaxf(invd, 1e-12f);
    float3 wpos = (wp0 * (w0 * t.invd0) + wp1 * (w1 * t.invd1) + wp2 * (w2 * t.invd2)) * d;
    float3 wn   = (wn0 * (w0 * t.invd0) + wn1 * (w1 * t.invd1) + wn2 * (w2 * t.invd2)) * d;

    // Image skin: replace the flat albedo with the texture's linear RGB, sampled either at
    // the interpolated per-vertex UV or by world triplanar projection (mirrors raster.h P3).
    float3 col = color;
    if (tex >= 0 && tex < nTex) {
        if (tps > 0.0f) {
            col = dSampleRgbTri(texMeta, texels, tex, wpos, wn, tps);
        } else {
            float u = (uv0.x * (w0 * t.invd0) + uv1.x * (w1 * t.invd1) + uv2.x * (w2 * t.invd2)) * d;
            float v2 = (uv0.y * (w0 * t.invd0) + uv1.y * (w1 * t.invd1) + uv2.y * (w2 * t.invd2)) * d;
            col = dSampleRgb(texMeta, texels, tex, u, v2);
        }
    }

    float3 N3 = normalize3(wn);
    float3 V  = normalize3(cam.eye - wpos);
    if (dot3(N3, V) < 0.0f) N3 = N3 * -1.0f;             // two-sided
    float lit = 0.0f;
    for (int li = 0; li < nLights; ++li) {
        const DLight& lp = lights[li];
        float3 dd = lp.pos - wpos;
        float dist2 = dot3(dd, dd);
        float3 Ld = (dist2 > 1e-12f) ? dd * (1.0f / sqrtf(dist2)) : V;
        float ndl = fmaxf(0.0f, dot3(N3, Ld));
        if (ndl <= 0.0f) continue;
        float atten = 1.0f;
        if (lp.falloff2 > 0.0f) atten = lp.falloff2 / (lp.falloff2 + dist2);
        float cone = 1.0f;
        if (lp.spot) cone = spotFalloffD(dot3(lp.dir, Ld * -1.0f), lp.cosInner, lp.cosOuter);
        lit += lp.weight * ndl * atten * cone;
    }
    float head = fmaxf(0.0f, dot3(N3, V));
    float k = ambient + keyScale * lit + fill * head;
    accum[i] = col * k;
}

// ---------------------------------------------------------------------------
// See-through clear pass: for each clear (transmissive) slot, every covered pixel whose
// clear fragment lies IN FRONT of the opaque depth (invd > zbuf) multiplies that pixel's
// running transmittance `clearT` by the per-surface transmittance and its milk product
// `milkT` by (1 - per-surface milk). Order-independent (commutative product), so no sort;
// atomicMulF makes concurrent fragments on one pixel safe. Device twin of fillTriangleClear.
__global__ void kClear(const DPTri* tris, const DGeo* geos, const DAttr* attrs,
                       const int* flags, int nSlots, const float* zbuf,
                       DCam cam, int W, int H, float clarity, float milkPerSurface,
                       float rimStrength, float* clearT, float* milkT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nSlots) return;
    int flg = flags[idx];                       // dense probe before touching the record
    if ((flg & (kSlotValid | kSlotClear)) != (kSlotValid | kSlotClear)) return;
    const DGeo& t = geos[idx];
    // World pos/normal sources (verbatim DPTri fields unless the slot was clipped).
    float3 wp0, wp1, wp2, wn0, wn1, wn2;
    if (flg & kSlotClipped) {
        const DAttr& a = attrs[idx];
        wp0 = a.wp0; wn0 = a.wn0; wp1 = a.wp1; wn1 = a.wn1; wp2 = a.wp2; wn2 = a.wn2;
    } else {
        const DPTri& s = tris[idx >> 1];
        wp0 = s.p0; wn0 = s.n0; wp1 = s.p1; wn1 = s.n1; wp2 = s.p2; wn2 = s.n2;
    }

    float minx = floorf(fminf(t.sx0, fminf(t.sx1, t.sx2)));
    float maxx = ceilf (fmaxf(t.sx0, fmaxf(t.sx1, t.sx2)));
    float miny = floorf(fminf(t.sy0, fminf(t.sy1, t.sy2)));
    float maxy = ceilf (fmaxf(t.sy0, fmaxf(t.sy1, t.sy2)));
    int xlo = max(0, (int)minx), xhi = min(W - 1, (int)maxx);
    int ylo = max(0, (int)miny), yhi = min(H - 1, (int)maxy);
    if (xlo > xhi || ylo > yhi) return;

    float area = (t.sx1 - t.sx0) * (t.sy2 - t.sy0) - (t.sy1 - t.sy0) * (t.sx2 - t.sx0);
    if (fabsf(area) < 1e-9f) return;
    float inv = 1.0f / area;
    const float dw0dx = (t.sy1 - t.sy2) * inv;
    const float dw1dx = (t.sy2 - t.sy0) * inv;
    const float tau = clarity;

    for (int y = ylo; y <= yhi; ++y) {
        float py = y + 0.5f, pxL = xlo + 0.5f;
        float w0 = ((t.sx1 - pxL) * (t.sy2 - py) - (t.sy1 - py) * (t.sx2 - pxL)) * inv;
        float w1 = ((t.sx2 - pxL) * (t.sy0 - py) - (t.sy2 - py) * (t.sx0 - pxL)) * inv;
        int row = y * W + xlo;
        for (int x = xlo; x <= xhi; ++x, ++row, w0 += dw0dx, w1 += dw1dx) {
            float w2 = 1.0f - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            float invd = w0 * t.invd0 + w1 * t.invd1 + w2 * t.invd2;   // 1/depth
            if (invd <= zbuf[row]) continue;   // behind (or at) the opaque surface: occluded
            // Grazing term from the interpolated normal for a silhouette milk rim.
            float d = 1.0f / fmaxf(invd, 1e-12f);
            float3 wpos = (wp0 * (w0 * t.invd0) + wp1 * (w1 * t.invd1) + wp2 * (w2 * t.invd2)) * d;
            float3 wn   = (wn0 * (w0 * t.invd0) + wn1 * (w1 * t.invd1) + wn2 * (w2 * t.invd2)) * d;
            float3 Nn = normalize3(wn);
            float3 Vv = normalize3(cam.eye - wpos);
            float ndv = fabsf(dot3(Nn, Vv));
            float graze = 1.0f - ndv;
            float perMilk = milkPerSurface + rimStrength * graze * graze * graze;
            if (perMilk > 0.95f) perMilk = 0.95f;
            atomicMulF(&clearT[row], tau);
            atomicMulF(&milkT[row], 1.0f - perMilk);
        }
    }
}

// Fill a device float array with a constant (used to reset clearT/milkT to 1.0 each frame;
// cudaMemset can only set byte patterns, not an arbitrary float).
__global__ void kFillF(float* p, float val, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = val;
}

// ---------------------------------------------------------------------------
// Pass D: device exposure + tonemap (twin of raster::exposeAndEncodeT).
//
// Auto-exposure luminance of pixel i, exactly as the host computes it: hit, non-emitter
// pixels only; v = max(r, g, b, 0). The host takes the max over the DOUBLE widenings,
// but double(float) is exact and order-preserving, so the float max widens to the same
// value. All qualifying v are >= 0 by the max-with-0, whose bit patterns (as unsigned)
// order identically to the float values — the basis of the exact histogram selection.
// The one wrinkle is -0.0f (possible only via fmaxf zero-sign ambiguity): it is the
// SAME VALUE as +0.0f but has bit pattern 0x80000000, so it is remapped to +0's bits;
// the selected order statistic is unchanged (equal values), matching the host, where
// nth_element treats +-0 as equal and eAuto degenerates to 1 either way.
__device__ inline bool lumBits(const float3* accum, const float* zbuf,
                               const unsigned char* emis, size_t i, unsigned int& ub) {
    if (zbuf[i] <= 0.0f || emis[i]) return false;   // skip background + emitters
    float3 a = accum[i];
    float v = fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, 0.0f));
    ub = __float_as_uint(v);
    if (ub == 0x80000000u) ub = 0u;                 // -0 == +0; bin as +0
    return true;
}

// Round 1: histogram of the TOP 16 bits of each qualifying luminance's bit pattern.
__global__ void kLumHist1(const float3* accum, const float* zbuf, const unsigned char* emis,
                          size_t n, unsigned int* hist) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int ub;
    if (lumBits(accum, zbuf, emis, i, ub)) atomicAdd(&hist[ub >> 16], 1u);
}

// Round 2: among pixels whose top 16 bits equal `hi` (the bin the k-th value fell in),
// histogram the LOW 16 bits — pinning down the exact k-th smallest bit pattern.
__global__ void kLumHist2(const float3* accum, const float* zbuf, const unsigned char* emis,
                          size_t n, unsigned int hi, unsigned int* hist) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int ub;
    if (lumBits(accum, zbuf, emis, i, ub) && (ub >> 16) == hi)
        atomicAdd(&hist[ub & 0xFFFFu], 1u);
}

// sRGB encode through the shared raster::srgbLut8() bytes — the host's encode lambda
// verbatim (same clamps, same index rounding via explicit RN double ops).
__device__ inline unsigned char encodeSrgb(double c, const unsigned char* lut) {
    if (c <= 0.0) return lut[0];
    if (c >= 1.0) return 255u;
    return lut[(int)__dadd_rn(__dmul_rn(c, 4096.0), 0.5)];
}

// Tonemap + encode, one thread per pixel — the host tonemap loop operation-for-operation.
// Every arithmetic op is an explicit round-to-nearest DOUBLE intrinsic so nvcc cannot
// contract mul+add into FMA: the host build (MSVC /fp:precise, no AVX2 codegen) performs
// plain IEEE mul/add there, and matching that sequence exactly is what keeps the output
// bytes identical. Background pixels (zbuf<=0) keep the unexposed bg tint; the
// see-through composite applies to ALL pixels (background included), as on the host.
__global__ void kToneMap(const float3* accum, const float* zbuf, size_t n,
                         double finalExp, int seeThrough,
                         const float* clearT, const float* milkT,
                         double milkX, double milkY, double milkZ,
                         const unsigned char* lut, unsigned char* img) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 a = accum[i];
    double cx = (double)a.x, cy = (double)a.y, cz = (double)a.z;
    if (zbuf[i] > 0.0f) {                          // hit pixels get the exposure
        cx = __dmul_rn(cx, finalExp);
        cy = __dmul_rn(cy, finalExp);
        cz = __dmul_rn(cz, finalExp);
    }
    if (seeThrough) {                              // composite clear glass (display-linear)
        float T = clearT[i], mt = milkT[i];
        if (T < 1.0f || mt < 1.0f) {
            double m = __dsub_rn(1.0, (double)mt); // (1 - mt) evaluated once, as on host
            cx = __dadd_rn(__dmul_rn(cx, (double)T), __dmul_rn(milkX, m));
            cy = __dadd_rn(__dmul_rn(cy, (double)T), __dmul_rn(milkY, m));
            cz = __dadd_rn(__dmul_rn(cz, (double)T), __dmul_rn(milkZ, m));
        }
    }
    img[i * 3 + 0] = encodeSrgb(cx, lut);
    img[i * 3 + 1] = encodeSrgb(cy, lut);
    img[i * 3 + 2] = encodeSrgb(cz, lut);
}

// ---------------------------------------------------------------------------
// Host side.

// True if a usable CUDA device is present (cached).
bool available() {
    static int cached = -1;
    if (cached < 0) {
        int n = 0;
        cudaError_t e = cudaGetDeviceCount(&n);
        cached = (e == cudaSuccess && n > 0) ? 1 : 0;
    }
    return cached == 1;
}

// Opaque uploaded scene: persistent device triangle + light arrays, plus cached per-pixel
// scratch (grown as needed) so a camera_path re-renders without re-uploading geometry.
struct Scene {
    DPTri*   dtris   = nullptr;
    int      nTris   = 0;
    DGeo*    dgeos   = nullptr;   // 2*nTris slots: screen geometry (classify/raster read this)
    DAttr*   dattrs  = nullptr;   // 2*nTris slots: shading attrs, written only for clipped slots
    int*     dflags  = nullptr;   // per-slot kSlotValid|kSlotClear|kSlotClipped bits (dense probe)
    DLight*  dlights = nullptr;
    int      nLights = 0;
    float    ambient = 0.12f, keyScale = 1.15f, fill = 0.08f;
    // Image-skin textures (flattened): per-texture metadata + one shared texel array.
    DTex*    dtexMeta = nullptr;
    float3*  dtexels  = nullptr;
    int      nTex     = 0;
    // Per-pixel scratch (sized to the largest W*H seen).
    unsigned long long* vis    = nullptr;
    float3*             accum  = nullptr;
    float*              zbuf   = nullptr;
    unsigned char*      emis   = nullptr;
    float*              clearT = nullptr;   // see-through cumulative transmittance
    float*              milkT  = nullptr;   // see-through milk (haze) product
    size_t              pixCap = 0;
    // Raster bin lists (slot indices by bbox size, rebuilt per frame) + 3 counters.
    int* dbinSmall = nullptr;
    int* dbinMed   = nullptr;
    int* dbinLarge = nullptr;
    int* dbinCnt   = nullptr;
    // Device exposure/tonemap (Pass D): 65536-bin luminance histogram + its pinned host
    // copy (the two exact-p99 rounds), the shared sRGB LUT bytes (uploaded once from
    // raster::srgbLut8()), and the finished RGB8 frame + its pinned download staging
    // (pinned = full PCIe rate; only these ~4MB cross the bus per frame).
    unsigned int*  dhist  = nullptr;   // 65536 bins (device)
    unsigned int*  h_hist = nullptr;   // pinned host copy of dhist
    unsigned char* dlut   = nullptr;   // raster::srgbLut8() bytes (4097)
    unsigned char* dimg   = nullptr;   // final W*H*3 RGB8 (device, pixCap-sized)
    unsigned char* h_img  = nullptr;   // pinned download staging for dimg
    // Per-pass profiling events (stream marks; see renderFrame). [0]=frame start,
    // then after: clearvis, project, raster, shade, clear, tonemap, download.
    cudaEvent_t ev[8] = {};
};

static float3 toF3(const Vec3& v) { return make_float3((float)v.x, (float)v.y, (float)v.z); }

// Non-aborting device alloc: returns false on failure (the rasterizer falls back to CPU
// rather than killing the process, unlike the transport CUDA path's CUDA_CHECK).
static bool tryMalloc(void** p, size_t bytes) {
    cudaError_t e = cudaMalloc(p, bytes);
    if (e != cudaSuccess) { *p = nullptr; return false; }
    return true;
}

// Same, for page-locked (pinned) host staging memory.
static bool tryMallocHost(void** p, size_t bytes) {
    cudaError_t e = cudaMallocHost(p, bytes);
    if (e != cudaSuccess) { *p = nullptr; return false; }
    return true;
}

void destroy(Scene* sc) {
    if (!sc) return;
    if (sc->dtris)    cudaFree(sc->dtris);
    if (sc->dgeos)    cudaFree(sc->dgeos);
    if (sc->dattrs)   cudaFree(sc->dattrs);
    if (sc->dflags)   cudaFree(sc->dflags);
    if (sc->dlights)  cudaFree(sc->dlights);
    if (sc->dtexMeta) cudaFree(sc->dtexMeta);
    if (sc->dtexels)  cudaFree(sc->dtexels);
    if (sc->vis)      cudaFree(sc->vis);
    if (sc->accum)    cudaFree(sc->accum);
    if (sc->zbuf)     cudaFree(sc->zbuf);
    if (sc->emis)     cudaFree(sc->emis);
    if (sc->clearT)   cudaFree(sc->clearT);
    if (sc->milkT)    cudaFree(sc->milkT);
    if (sc->dbinSmall) cudaFree(sc->dbinSmall);
    if (sc->dbinMed)   cudaFree(sc->dbinMed);
    if (sc->dbinLarge) cudaFree(sc->dbinLarge);
    if (sc->dbinCnt)   cudaFree(sc->dbinCnt);
    if (sc->dhist)  cudaFree(sc->dhist);
    if (sc->dlut)   cudaFree(sc->dlut);
    if (sc->dimg)   cudaFree(sc->dimg);
    if (sc->h_hist) cudaFreeHost(sc->h_hist);
    if (sc->h_img)  cudaFreeHost(sc->h_img);
    for (int i = 0; i < 8; ++i)
        if (sc->ev[i]) cudaEventDestroy(sc->ev[i]);
    delete sc;
}

Scene* upload(const std::vector<raster::PTri>& tris, const raster::PreviewLight& light,
              const std::vector<Texture>* textures) {
    if (!available() || tris.empty()) return nullptr;
    Scene* sc = new Scene();
    sc->nTris = (int)tris.size();

    // Bake triangles.
    std::vector<DPTri> h(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
        const raster::PTri& t = tris[i];
        DPTri& d = h[i];
        d.p0 = toF3(t.p0); d.p1 = toF3(t.p1); d.p2 = toF3(t.p2);
        d.n0 = toF3(t.n0); d.n1 = toF3(t.n1); d.n2 = toF3(t.n2);
        d.color = toF3(t.color);
        d.uv0 = make_float2((float)t.uv0.x, (float)t.uv0.y);
        d.uv1 = make_float2((float)t.uv1.x, (float)t.uv1.y);
        d.uv2 = make_float2((float)t.uv2.x, (float)t.uv2.y);
        d.tex = t.tex;
        d.triplanarScale = (float)t.triplanarScale;
        d.emissive = t.emissive ? 1 : 0;
        d.clear = t.clear ? 1 : 0;
    }
    // Bake image-skin textures: flatten every texture's LINEAR rgb into one shared texel
    // array, plus per-texture metadata (dims/filter/wrap/offset). Sampling on the device
    // mirrors Texture::sampleRgb / sampleRgbTriplanar exactly. Palette (indexed) textures
    // sample their raw index-map rgb here, identical to the CPU preview's sampleRgb path.
    std::vector<DTex>   htexMeta;
    std::vector<float3> htexels;
    if (textures && !textures->empty()) {
        htexMeta.resize(textures->size());
        for (size_t i = 0; i < textures->size(); ++i) {
            const Texture& tx = (*textures)[i];
            DTex& m = htexMeta[i];
            m.w = tx.w; m.h = tx.h;
            m.filter = (tx.filter == TexFilter::Nearest) ? 0 : 1;
            m.wrap   = (tx.wrap == TexWrap::Clamp) ? 1 : (tx.wrap == TexWrap::Mirror ? 2 : 0);
            m.offset = (int)htexels.size();
            m.valid  = tx.valid() ? 1 : 0;
            if (m.valid) {
                for (const Vec3& c : tx.rgb)
                    htexels.push_back(make_float3((float)c.x, (float)c.y, (float)c.z));
            }
        }
    }
    sc->nTex = (int)htexMeta.size();
    // Bake lights.
    std::vector<DLight> hl(light.lights.size());
    for (size_t i = 0; i < light.lights.size(); ++i) {
        const raster::PLight& p = light.lights[i];
        DLight& d = hl[i];
        d.pos = toF3(p.pos); d.dir = toF3(p.dir); d.spot = p.spot ? 1 : 0;
        d.cosInner = (float)p.cosInner; d.cosOuter = (float)p.cosOuter;
        d.weight = (float)p.weight; d.falloff2 = (float)p.falloff2;
    }
    sc->nLights  = (int)hl.size();
    sc->ambient  = (float)light.ambient;
    sc->keyScale = (float)light.keyScale;
    sc->fill     = (float)light.fill;

    bool ok = tryMalloc((void**)&sc->dtris,  sizeof(DPTri) * tris.size())
           && tryMalloc((void**)&sc->dgeos,  sizeof(DGeo)  * 2 * tris.size())
           && tryMalloc((void**)&sc->dattrs, sizeof(DAttr) * 2 * tris.size())
           && tryMalloc((void**)&sc->dflags, sizeof(int) * 2 * tris.size())
           && tryMalloc((void**)&sc->dbinSmall, sizeof(int) * 2 * tris.size())
           && tryMalloc((void**)&sc->dbinMed,   sizeof(int) * 2 * tris.size())
           && tryMalloc((void**)&sc->dbinLarge, sizeof(int) * 2 * tris.size())
           && tryMalloc((void**)&sc->dbinCnt,   sizeof(int) * 5)   // 3 counts + med/large tickets
           && tryMalloc((void**)&sc->dhist, 65536 * sizeof(unsigned int))
           && tryMalloc((void**)&sc->dlut,  raster::srgbLut8().size())
           && tryMallocHost((void**)&sc->h_hist, 65536 * sizeof(unsigned int));
    for (int i = 0; ok && i < 8; ++i)
        ok = (cudaEventCreate(&sc->ev[i]) == cudaSuccess);
    if (ok && !hl.empty())
        ok = tryMalloc((void**)&sc->dlights, sizeof(DLight) * hl.size());
    if (ok && !htexMeta.empty())
        ok = tryMalloc((void**)&sc->dtexMeta, sizeof(DTex) * htexMeta.size());
    if (ok && !htexels.empty())
        ok = tryMalloc((void**)&sc->dtexels, sizeof(float3) * htexels.size());
    if (!ok) { destroy(sc); return nullptr; }

    if (cudaMemcpy(sc->dtris, h.data(), sizeof(DPTri) * tris.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) { destroy(sc); return nullptr; }
    if (!hl.empty() &&
        cudaMemcpy(sc->dlights, hl.data(), sizeof(DLight) * hl.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) { destroy(sc); return nullptr; }
    if (!htexMeta.empty() &&
        cudaMemcpy(sc->dtexMeta, htexMeta.data(), sizeof(DTex) * htexMeta.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) { destroy(sc); return nullptr; }
    if (!htexels.empty() &&
        cudaMemcpy(sc->dtexels, htexels.data(), sizeof(float3) * htexels.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) { destroy(sc); return nullptr; }
    // The EXACT sRGB table bytes the host tonemap indexes — sharing it is part of the
    // byte-identity guarantee of the device tonemap (see kToneMap / encodeSrgb).
    if (cudaMemcpy(sc->dlut, raster::srgbLut8().data(), raster::srgbLut8().size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) { destroy(sc); return nullptr; }
    return sc;
}

// Grow the cached per-pixel scratch to hold N pixels (freeing + reallocating on growth).
static bool ensurePix(Scene* sc, size_t N) {
    if (N <= sc->pixCap && sc->vis) return true;
    if (sc->vis)    { cudaFree(sc->vis);    sc->vis = nullptr; }
    if (sc->accum)  { cudaFree(sc->accum);  sc->accum = nullptr; }
    if (sc->zbuf)   { cudaFree(sc->zbuf);   sc->zbuf = nullptr; }
    if (sc->emis)   { cudaFree(sc->emis);   sc->emis = nullptr; }
    if (sc->clearT) { cudaFree(sc->clearT); sc->clearT = nullptr; }
    if (sc->milkT)  { cudaFree(sc->milkT);  sc->milkT = nullptr; }
    if (sc->dimg)   { cudaFree(sc->dimg);   sc->dimg = nullptr; }
    if (sc->h_img)  { cudaFreeHost(sc->h_img); sc->h_img = nullptr; }
    sc->pixCap = 0;
    bool ok = tryMalloc((void**)&sc->vis,    sizeof(unsigned long long) * N)
           && tryMalloc((void**)&sc->accum,  sizeof(float3) * N)
           && tryMalloc((void**)&sc->zbuf,   sizeof(float) * N)
           && tryMalloc((void**)&sc->emis,   sizeof(unsigned char) * N)
           && tryMalloc((void**)&sc->clearT, sizeof(float) * N)
           && tryMalloc((void**)&sc->milkT,  sizeof(float) * N)
           && tryMalloc((void**)&sc->dimg,   N * 3)
           && tryMallocHost((void**)&sc->h_img, N * 3);
    if (!ok) return false;
    sc->pixCap = N;
    return true;
}

// --- optional per-pass profiling (see raster_cuda.h). Passes are timed with CUDA
// events recorded into the stream between passes: the GPU timestamps each mark as it
// reaches it, so the windows have exactly the same composition as the old synced wall
// clocks (any host-readback bubbles inside a pass are included) WITHOUT renderFrame
// having to synchronize after every pass. The frame runs sync-free except for its real
// data dependencies (first-frame histogram readbacks, final image download); event
// pairs are resolved once per frame after that download has fenced everything.
static bool g_prof = false;
static Prof g_profAcc;
void profEnable(bool on) { g_prof = on; }
Prof profTake() { Prof p = g_profAcc; g_profAcc = Prof(); return p; }
static inline void paddEv(double& acc, cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, a, b) == cudaSuccess) acc += ms;
}

std::vector<uint8_t> renderFrame(Scene* sc, const Camera& cam, int W, int H, int nThreads,
                                 double exposure, bool autoExpose, double* lockAnchor,
                                 bool seeThrough, double glassClarity) {
    std::vector<uint8_t> empty;
    (void)nThreads;   // whole frame (incl. expose/tonemap) runs on the device now
    if (!sc || sc->nTris == 0 || W <= 0 || H <= 0) return empty;
    const size_t N = (size_t)W * H;
    if (!ensurePix(sc, N)) return empty;

    DCam dc;
    dc.eye = toF3(cam.eye); dc.u = toF3(cam.u); dc.v = toF3(cam.v); dc.w = toF3(cam.w);
    dc.tanHalfX = (float)cam.tanHalfX; dc.tanHalfY = (float)cam.tanHalfY;
    dc.projection = cam.projection;
    dc.rEdge = (float)cam.rEdge;

    const float3 bg = make_float3(0.06f, 0.07f, 0.09f);
    const float  EMIS_BOOST = 4.0f;

    // See-through (clear-glass) preview parameters — identical to raster::renderFrame's.
    const double kMilkPerSurface = std::max(0.0, (1.0 - glassClarity)) * 0.55;
    const double kRimStrength    = 0.55;
    const Vec3   kMilkColor{0.52, 0.55, 0.60};

    // Per-pass stream marks (no-ops unless profiling; resolved after the download).
    auto rec = [&](int i) { if (g_prof) cudaEventRecord(sc->ev[i], 0); };

    rec(0);
    if (cudaMemset(sc->vis, 0, sizeof(unsigned long long) * N) != cudaSuccess) return empty;
    rec(1);

    int TPB = 256;
    int gTris  = (sc->nTris + TPB - 1) / TPB;
    int gSlots = (2 * sc->nTris + TPB - 1) / TPB;
    int gPix   = (int)((N + TPB - 1) / TPB);

    kProject<<<gTris, TPB>>>(sc->dtris, sc->nTris, dc, W, H, sc->dgeos, sc->dattrs, sc->dflags);
    rec(2);
    // Bin the slots by bbox size, then rasterize each bin at a matching parallel width
    // (thread / warp / block per sub-triangle). Every (slot,row) runs the exact row maths
    // of the old single kernel and atomicMax merges order-independently, so the result is
    // bit-identical while a screen-filling quad no longer serializes on one thread.
    if (cudaMemset(sc->dbinCnt, 0, 5 * sizeof(int)) != cudaSuccess) return empty;   // counts + tickets
    kClassify<<<gSlots, TPB>>>(sc->dgeos, sc->dflags, 2 * sc->nTris, W, H, seeThrough ? 1 : 0,
                               sc->dbinSmall, sc->dbinMed, sc->dbinLarge, sc->dbinCnt);
    // The raster kernels read their bin count from dbinCnt, so nothing here waits on
    // the classify results. Small gets a 1-thread-per-item upper-bound grid (all slots
    // could be small); med/large get fixed device-filling grids and self-balance via
    // their ticket queues. Excess threads/blocks exit after one cached 4-byte load.
    kRasterSmall<<<gSlots, TPB>>>(sc->dgeos, sc->dflags, sc->dbinSmall, sc->dbinCnt,
                                  W, H, seeThrough ? 1 : 0, sc->vis);
    kRasterMed<<<2048, TPB>>>(sc->dgeos, sc->dflags, sc->dbinMed, sc->dbinCnt,
                              W, H, seeThrough ? 1 : 0, sc->vis);
    kRasterLarge<<<768, TPB>>>(sc->dgeos, sc->dflags, sc->dbinLarge, sc->dbinCnt,
                               W, H, seeThrough ? 1 : 0, sc->vis);
    rec(3);
    kShade<<<gPix, TPB>>>(sc->dtris, sc->dgeos, sc->dattrs, sc->dflags, sc->vis,
                          sc->dlights, sc->nLights,
                          sc->ambient, sc->keyScale, sc->fill, dc, W, H, bg, EMIS_BOOST,
                          sc->dtexMeta, sc->dtexels, sc->nTex,
                          sc->accum, sc->zbuf, sc->emis);
    rec(4);

    // See-through clear pass: reset clearT/milkT to 1 and accumulate each clear surface's
    // transmittance/haze against the now-complete opaque depth (sc->zbuf, written by kShade).
    if (seeThrough) {
        kFillF<<<gPix, TPB>>>(sc->clearT, 1.0f, N);
        kFillF<<<gPix, TPB>>>(sc->milkT,  1.0f, N);
        // Stream order already runs kClear after both fills complete.
        kClear<<<gSlots, TPB>>>(sc->dtris, sc->dgeos, sc->dattrs, sc->dflags, 2 * sc->nTris,
                                sc->zbuf, dc, W, H,
                                (float)glassClarity, (float)kMilkPerSurface, (float)kRimStrength,
                                sc->clearT, sc->milkT);
    }
    rec(5);   // recorded either way; the clear window is simply ~0 when see-through is off

    // Pass D: exposure + tonemap on the DEVICE (twin of raster::exposeAndEncodeT — see
    // the kernels above). The p99 auto-exposure anchor needs the exact k-th smallest
    // qualifying luminance; non-negative floats order like their bit patterns, so two
    // 65536-bin histogram rounds (top 16 bits, then low 16 within the winning bin) find
    // that exact value with no sort and only a 256KB readback per round. eAuto / the
    // lockAnchor handshake stay on the host, mirroring the host tail line for line.
    const double expComp = (exposure > 0.0) ? exposure : 1.0;
    double eAuto = 1.0;
    if (autoExpose) {
        if (lockAnchor && *lockAnchor > 0.0) {
            eAuto = *lockAnchor;                    // reuse the path's locked anchor
        } else {
            unsigned int cut[2] = {0, 0};           // winning top-16 / low-16 bin
            size_t total = 0, rank = 0;
            for (int round = 0; round < 2; ++round) {
                if (cudaMemset(sc->dhist, 0, 65536 * sizeof(unsigned int)) != cudaSuccess) return empty;
                if (round == 0)
                    kLumHist1<<<gPix, TPB>>>(sc->accum, sc->zbuf, sc->emis, N, sc->dhist);
                else
                    kLumHist2<<<gPix, TPB>>>(sc->accum, sc->zbuf, sc->emis, N, cut[0], sc->dhist);
                if (cudaMemcpy(sc->h_hist, sc->dhist, 65536 * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost) != cudaSuccess) return empty;   // syncs
                if (round == 0) {
                    for (int b = 0; b < 65536; ++b) total += sc->h_hist[b];
                    if (total == 0) break;          // no lit surfaces: eAuto stays 1
                    rank = (size_t)(0.99 * (total - 1));   // the host tail's exact k
                }
                size_t cum = 0;                     // descend to the bin holding rank #k
                for (int b = 0; b < 65536; ++b) {
                    size_t c = sc->h_hist[b];
                    if (rank < cum + c) { cut[round] = (unsigned int)b; rank -= cum; break; }
                    cum += c;
                }
            }
            if (total > 0) {
                unsigned int bits = (cut[0] << 16) | cut[1];
                float p99f;                          // exact k-th smallest luminance
                std::memcpy(&p99f, &bits, sizeof(float));
                const double p99 = (double)p99f;     // widening is exact, as on the host
                eAuto = (p99 > 0.0) ? 0.9 / p99 : 1.0;
            }
            if (lockAnchor) *lockAnchor = eAuto;     // first frame sets the anchor
        }
    }
    const double finalExp = eAuto * expComp;
    kToneMap<<<gPix, TPB>>>(sc->accum, sc->zbuf, N, finalExp, seeThrough ? 1 : 0,
                            sc->clearT, sc->milkT, kMilkColor.x, kMilkColor.y, kMilkColor.z,
                            sc->dlut, sc->dimg);
    rec(6);

    // Download ONLY the finished RGB8 frame through the pinned staging buffer. This
    // blocking copy fences every pass enqueued above; a poisoned context surfaces in
    // its return code (or the earlier readbacks'), and the sticky-error sweep below
    // catches kernel-launch failures that never poisoned a blocking call.
    if (cudaMemcpy(sc->h_img, sc->dimg, N * 3, cudaMemcpyDeviceToHost) != cudaSuccess) return empty;
    rec(7);
    if (cudaGetLastError() != cudaSuccess) return empty;
    if (g_prof) {
        if (cudaEventSynchronize(sc->ev[7]) == cudaSuccess) {
            paddEv(g_profAcc.clearvis_ms, sc->ev[0], sc->ev[1]);
            paddEv(g_profAcc.project_ms,  sc->ev[1], sc->ev[2]);
            paddEv(g_profAcc.raster_ms,   sc->ev[2], sc->ev[3]);
            paddEv(g_profAcc.shade_ms,    sc->ev[3], sc->ev[4]);
            paddEv(g_profAcc.clear_ms,    sc->ev[4], sc->ev[5]);
            paddEv(g_profAcc.expose_ms,   sc->ev[5], sc->ev[6]);
            paddEv(g_profAcc.download_ms, sc->ev[6], sc->ev[7]);
        }
        ++g_profAcc.frames;
    }
    return std::vector<uint8_t>(sc->h_img, sc->h_img + N * 3);
}

}  // namespace raster_cuda
