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
//
//   Pass B  kRaster   (1 thread / DSTri slot): rasterize the sub-triangle's pixel bbox,
//                     packing (1/depth, slotIdx) into a 64-bit visibility buffer with a
//                     single atomicMax. Nearest surface (largest 1/depth) wins per pixel.
//
//   Pass C  kShade    (1 thread / pixel): decode the winning slot, recompute barycentrics
//                     at the pixel centre, interpolate world pos/normal, and shade once
//                     (the same ambient + Σ weighted N·L + headlight model as the CPU).
//                     Writes an HDR accum buffer + a 1/depth hit key + an emitter mask.
//
//   HOST  download accum/zbuf/emis, then raster::exposeAndEncode() applies the p99 auto-
//         exposure + sRGB tone map on the HOST — the SAME code the CPU path uses, so
//         exposure (incl. a camera_path's shared lockAnchor) and encoding are identical.
//
// Everything on the device runs in single precision (float): this is a solid-shaded
// preview, float is amply accurate for the geometry/shading, and it halves the DSTri and
// buffer memory. The exposure/tonemap tail that must match the real render frame-to-frame
// stays in host double precision (exposeAndEncode).
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
  #define cudaGetLastError        hipGetLastError
  #define cudaDeviceSynchronize   hipDeviceSynchronize
  #define cudaGetErrorString      hipGetErrorString
#else
  #include <cuda_runtime.h>
#endif

#include <cstdio>
#include <cstdint>
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

// A projected screen sub-triangle produced by the clip/project pass and consumed by the
// raster + shade passes. `valid`=0 marks an unused slot (a triangle that clipped away, or
// the second slot of a triangle that produced only one sub-triangle).
struct DSTri {
    float  sx0, sy0, invd0; float3 wp0, wn0; float2 uv0;
    float  sx1, sy1, invd1; float3 wp1, wn1; float2 uv1;
    float  sx2, sy2, invd2; float3 wp2, wn2; float2 uv2;
    float3 color;
    int    tex;             // skin texture index, or -1
    float  triplanarScale;  // >0: sample by world triplanar instead of UV
    int    emissive;
    int    clear;           // see-through transmissive surface
    int    valid;
};

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
// Pass A: transform + project each input triangle into up to two DSTri. Rectilinear does a
// Sutherland-Hodgman near-plane clip + fan (up to 2 sub-triangles); a fisheye/panoramic lens
// instead rejects any triangle that reaches (nearly) behind the camera and projects the
// three vertices straight through the angular map into a single sub-triangle. This mirrors
// raster::projectRange's two branches exactly, so the GPU and CPU rasters agree per camera.
__global__ void kProject(const DPTri* tris, int nTris, DCam cam, int W, int H,
                         DSTri* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nTris) return;
    const DPTri& t = tris[i];
    // Invalidate both output slots up front.
    out[2*i].valid = 0; out[2*i+1].valid = 0;

    // World -> camera space.
    DVtxCS cs[3] = { toCS(cam, t.p0, t.n0, t.uv0), toCS(cam, t.p1, t.n1, t.uv1),
                     toCS(cam, t.p2, t.n2, t.uv2) };

    if (cam.projection != CAM_RECTILINEAR) {
        // Fisheye/panoramic: no near-plane clip. Reject the triangle if any vertex points
        // (nearly) backward (z <= -0.999*len), else project all three straight through the
        // angular lens map into ONE sub-triangle (slot [2*i]; [2*i+1] stays invalid).
        for (int e = 0; e < 3; ++e) {
            float len = sqrtf(cs[e].x*cs[e].x + cs[e].y*cs[e].y + cs[e].z*cs[e].z);
            if (cs[e].z <= -0.999f * len) return;
        }
        DSTri s;
        projectVtxG(cam, cs[0], W, H, s.sx0, s.sy0, s.invd0, s.wp0, s.wn0, s.uv0);
        projectVtxG(cam, cs[1], W, H, s.sx1, s.sy1, s.invd1, s.wp1, s.wn1, s.uv1);
        projectVtxG(cam, cs[2], W, H, s.sx2, s.sy2, s.invd2, s.wp2, s.wn2, s.uv2);
        s.color = t.color; s.tex = t.tex; s.triplanarScale = t.triplanarScale;
        s.emissive = t.emissive; s.clear = t.clear; s.valid = 1;
        out[2*i] = s;
        return;
    }

    // Rectilinear: Sutherland-Hodgman clip against the near plane z=zn.
    const float zn = 1e-3f;
    DVtxCS poly[8]; int np = 0;
    for (int e = 0; e < 3; ++e) {
        const DVtxCS& A = cs[e]; const DVtxCS& B = cs[(e+1)%3];
        bool inA = A.z > zn, inB = B.z > zn;
        if (inA && np < 8) poly[np++] = A;
        if (inA != inB && np < 8) { float s = (zn - A.z) / (B.z - A.z); poly[np++] = lerpVtx(A, B, s); }
    }
    if (np < 3) return;

    // Fan the clipped polygon into (np-2) triangles; store up to the first two.
    float sx[8], sy[8], invd[8]; float3 wp[8], wn[8]; float2 uv[8];
    for (int k = 0; k < np; ++k)
        projectVtxG(cam, poly[k], W, H, sx[k], sy[k], invd[k], wp[k], wn[k], uv[k]);

    int slot = 0;
    for (int k = 1; k + 1 < np && slot < 2; ++k, ++slot) {
        DSTri s;
        s.sx0 = sx[0];   s.sy0 = sy[0];   s.invd0 = invd[0];   s.wp0 = wp[0];   s.wn0 = wn[0];   s.uv0 = uv[0];
        s.sx1 = sx[k];   s.sy1 = sy[k];   s.invd1 = invd[k];   s.wp1 = wp[k];   s.wn1 = wn[k];   s.uv1 = uv[k];
        s.sx2 = sx[k+1]; s.sy2 = sy[k+1]; s.invd2 = invd[k+1]; s.wp2 = wp[k+1]; s.wn2 = wn[k+1]; s.uv2 = uv[k+1];
        s.color = t.color; s.tex = t.tex; s.triplanarScale = t.triplanarScale;
        s.emissive = t.emissive; s.clear = t.clear; s.valid = 1;
        out[2*i + slot] = s;
    }
}

// ---------------------------------------------------------------------------
// Pass B: rasterize each valid DSTri into the 64-bit visibility buffer. Each covered pixel
// packs (1/depth as float bits) << 32 | slotIdx; atomicMax keeps the nearest (largest
// 1/depth) surface. Mirrors fillTriangleG's barycentric coverage + perspective 1/depth.
__global__ void kRaster(const DSTri* stris, int nSlots, int W, int H, int seeThrough,
                        unsigned long long* vis) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nSlots) return;
    const DSTri& t = stris[idx];
    if (!t.valid) return;
    if (seeThrough && t.clear) return;   // clear surfaces handled by the clear-accumulation pass

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

    for (int y = ylo; y <= yhi; ++y) {
        float py = y + 0.5f, pxL = xlo + 0.5f;
        float w0 = ((t.sx1 - pxL) * (t.sy2 - py) - (t.sy1 - py) * (t.sx2 - pxL)) * inv;
        float w1 = ((t.sx2 - pxL) * (t.sy0 - py) - (t.sy2 - py) * (t.sx0 - pxL)) * inv;
        unsigned long long row = (unsigned long long)y * W + xlo;
        for (int x = xlo; x <= xhi; ++x, ++row, w0 += dw0dx, w1 += dw1dx) {
            float w2 = 1.0f - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            float invd = w0 * t.invd0 + w1 * t.invd1 + w2 * t.invd2;   // 1/depth
            if (invd <= 0.0f) continue;
            unsigned long long packed =
                ((unsigned long long)__float_as_uint(invd) << 32) | (unsigned int)idx;
            atomicMax(&vis[row], packed);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass C: resolve + shade each pixel once. Decode the winning DSTri, recompute barycentrics
// at the pixel centre (same float math as kRaster, so the winner's 1/depth reproduces),
// interpolate world pos/normal, and shade with the same model as raster.h Pass 3.
__global__ void kShade(const DSTri* stris, const unsigned long long* vis,
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
    const DSTri& t = stris[slot];

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
    emis[i] = t.emissive ? 1 : 0;
    if (t.emissive) { accum[i] = t.color * emisBoost; return; }   // raw emitter radiance

    // Perspective-correct world pos / normal.
    float d = 1.0f / fmaxf(invd, 1e-12f);
    float3 wpos = (t.wp0 * (w0 * t.invd0) + t.wp1 * (w1 * t.invd1) + t.wp2 * (w2 * t.invd2)) * d;
    float3 wn   = (t.wn0 * (w0 * t.invd0) + t.wn1 * (w1 * t.invd1) + t.wn2 * (w2 * t.invd2)) * d;

    // Image skin: replace the flat albedo with the texture's linear RGB, sampled either at
    // the interpolated per-vertex UV or by world triplanar projection (mirrors raster.h P3).
    float3 col = t.color;
    if (t.tex >= 0 && t.tex < nTex) {
        if (t.triplanarScale > 0.0f) {
            col = dSampleRgbTri(texMeta, texels, t.tex, wpos, wn, t.triplanarScale);
        } else {
            float u = (t.uv0.x * (w0 * t.invd0) + t.uv1.x * (w1 * t.invd1) + t.uv2.x * (w2 * t.invd2)) * d;
            float v = (t.uv0.y * (w0 * t.invd0) + t.uv1.y * (w1 * t.invd1) + t.uv2.y * (w2 * t.invd2)) * d;
            col = dSampleRgb(texMeta, texels, t.tex, u, v);
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
// See-through clear pass: for each clear (transmissive) DSTri, every covered pixel whose
// clear fragment lies IN FRONT of the opaque depth (invd > zbuf) multiplies that pixel's
// running transmittance `clearT` by the per-surface transmittance and its milk product
// `milkT` by (1 - per-surface milk). Order-independent (commutative product), so no sort;
// atomicMulF makes concurrent fragments on one pixel safe. Device twin of fillTriangleClear.
__global__ void kClear(const DSTri* stris, int nSlots, const float* zbuf, DCam cam,
                       int W, int H, float clarity, float milkPerSurface, float rimStrength,
                       float* clearT, float* milkT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nSlots) return;
    const DSTri& t = stris[idx];
    if (!t.valid || !t.clear) return;

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
            float3 wpos = (t.wp0 * (w0 * t.invd0) + t.wp1 * (w1 * t.invd1) + t.wp2 * (w2 * t.invd2)) * d;
            float3 wn   = (t.wn0 * (w0 * t.invd0) + t.wn1 * (w1 * t.invd1) + t.wn2 * (w2 * t.invd2)) * d;
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
    DSTri*   dstris  = nullptr;   // 2*nTris slots
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
};

static float3 toF3(const Vec3& v) { return make_float3((float)v.x, (float)v.y, (float)v.z); }

// Non-aborting device alloc: returns false on failure (the rasterizer falls back to CPU
// rather than killing the process, unlike the transport CUDA path's CUDA_CHECK).
static bool tryMalloc(void** p, size_t bytes) {
    cudaError_t e = cudaMalloc(p, bytes);
    if (e != cudaSuccess) { *p = nullptr; return false; }
    return true;
}

void destroy(Scene* sc) {
    if (!sc) return;
    if (sc->dtris)    cudaFree(sc->dtris);
    if (sc->dstris)   cudaFree(sc->dstris);
    if (sc->dlights)  cudaFree(sc->dlights);
    if (sc->dtexMeta) cudaFree(sc->dtexMeta);
    if (sc->dtexels)  cudaFree(sc->dtexels);
    if (sc->vis)      cudaFree(sc->vis);
    if (sc->accum)    cudaFree(sc->accum);
    if (sc->zbuf)     cudaFree(sc->zbuf);
    if (sc->emis)     cudaFree(sc->emis);
    if (sc->clearT)   cudaFree(sc->clearT);
    if (sc->milkT)    cudaFree(sc->milkT);
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
           && tryMalloc((void**)&sc->dstris, sizeof(DSTri) * 2 * tris.size());
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
    sc->pixCap = 0;
    bool ok = tryMalloc((void**)&sc->vis,    sizeof(unsigned long long) * N)
           && tryMalloc((void**)&sc->accum,  sizeof(float3) * N)
           && tryMalloc((void**)&sc->zbuf,   sizeof(float) * N)
           && tryMalloc((void**)&sc->emis,   sizeof(unsigned char) * N)
           && tryMalloc((void**)&sc->clearT, sizeof(float) * N)
           && tryMalloc((void**)&sc->milkT,  sizeof(float) * N);
    if (!ok) return false;
    sc->pixCap = N;
    return true;
}

static bool sync() {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    return e == cudaSuccess;
}

std::vector<uint8_t> renderFrame(Scene* sc, const Camera& cam, int W, int H, int nThreads,
                                 double exposure, bool autoExpose, double* lockAnchor,
                                 bool seeThrough, double glassClarity) {
    std::vector<uint8_t> empty;
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

    if (cudaMemset(sc->vis, 0, sizeof(unsigned long long) * N) != cudaSuccess) return empty;

    int TPB = 256;
    int gTris  = (sc->nTris + TPB - 1) / TPB;
    int gSlots = (2 * sc->nTris + TPB - 1) / TPB;
    int gPix   = (int)((N + TPB - 1) / TPB);

    kProject<<<gTris, TPB>>>(sc->dtris, sc->nTris, dc, W, H, sc->dstris);
    if (!sync()) return empty;
    kRaster<<<gSlots, TPB>>>(sc->dstris, 2 * sc->nTris, W, H, seeThrough ? 1 : 0, sc->vis);
    if (!sync()) return empty;
    kShade<<<gPix, TPB>>>(sc->dstris, sc->vis, sc->dlights, sc->nLights,
                          sc->ambient, sc->keyScale, sc->fill, dc, W, H, bg, EMIS_BOOST,
                          sc->dtexMeta, sc->dtexels, sc->nTex,
                          sc->accum, sc->zbuf, sc->emis);
    if (!sync()) return empty;

    // See-through clear pass: reset clearT/milkT to 1 and accumulate each clear surface's
    // transmittance/haze against the now-complete opaque depth (sc->zbuf, written by kShade).
    if (seeThrough) {
        kFillF<<<gPix, TPB>>>(sc->clearT, 1.0f, N);
        kFillF<<<gPix, TPB>>>(sc->milkT,  1.0f, N);
        if (!sync()) return empty;
        kClear<<<gSlots, TPB>>>(sc->dstris, 2 * sc->nTris, sc->zbuf, dc, W, H,
                                (float)glassClarity, (float)kMilkPerSurface, (float)kRimStrength,
                                sc->clearT, sc->milkT);
        if (!sync()) return empty;
    }

    // Download the HDR accum + hit key + emitter mask; run the SHARED host exposure/tonemap
    // tail so exposure (incl. lockAnchor) and encoding match the CPU path exactly.
    std::vector<float3>  haccum(N);
    std::vector<float>   hzbuf(N);
    std::vector<uint8_t> hemis(N);
    if (cudaMemcpy(haccum.data(), sc->accum, sizeof(float3) * N, cudaMemcpyDeviceToHost) != cudaSuccess) return empty;
    if (cudaMemcpy(hzbuf.data(),  sc->zbuf,  sizeof(float)  * N, cudaMemcpyDeviceToHost) != cudaSuccess) return empty;
    if (cudaMemcpy(hemis.data(),  sc->emis,  sizeof(unsigned char) * N, cudaMemcpyDeviceToHost) != cudaSuccess) return empty;

    std::vector<float> hclear, hmilk;
    if (seeThrough) {
        hclear.resize(N); hmilk.resize(N);
        if (cudaMemcpy(hclear.data(), sc->clearT, sizeof(float) * N, cudaMemcpyDeviceToHost) != cudaSuccess) return empty;
        if (cudaMemcpy(hmilk.data(),  sc->milkT,  sizeof(float) * N, cudaMemcpyDeviceToHost) != cudaSuccess) return empty;
    }

    std::vector<Vec3> accum(N);
    for (size_t i = 0; i < N; ++i)
        accum[i] = Vec3{ (double)haccum[i].x, (double)haccum[i].y, (double)haccum[i].z };

    const double expComp = (exposure > 0.0) ? exposure : 1.0;
    static const std::vector<float> emptyClear;   // unused when !seeThrough
    if (nThreads < 1) nThreads = 1;
    return raster::exposeAndEncode(accum, hzbuf, hemis, W, H, nThreads,
                                   expComp, autoExpose, lockAnchor,
                                   seeThrough, seeThrough ? hclear : emptyClear,
                                   seeThrough ? hmilk : emptyClear,
                                   kMilkColor);
}

}  // namespace raster_cuda
