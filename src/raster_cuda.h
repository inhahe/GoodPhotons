// raster_cuda.h — GPU (CUDA) backend for the solid-shaded PREVIEW rasterizer.
//
// This is the device twin of the CPU rasterizer in raster.h. It runs the same
// deferred-visibility pipeline — project+clip each world triangle, resolve the nearest
// surface per pixel, then shade it once — on the GPU, and returns a byte-identical-format
// RGB8 frame. The WHOLE frame stays on the device: geometry + shading, then a device
// twin of raster::exposeAndEncodeT's exposure/tonemap (exact p99 luminance selection via
// float-bit histograms, double-precision tonemap, the shared raster::srgbLut8() table),
// so only the final W*H*3 RGB8 image is downloaded. The device maths mirrors the host
// tail operation-for-operation, so a camera_path's shared auto-exposure anchor and the
// sRGB encoding stay bit-identical regardless of backend.
//
// Scope: all camera projections (RECTILINEAR pinhole plus the fisheye/panoramic lens
// maps — equidistant, equisolid, stereographic, orthographic), OPAQUE geometry, and
// see-through (clear-glass) compositing (a device clear-accumulation pass mirrors the
// CPU one). The shade pass has FULL parity with raster.h's: image skins (per-vertex UV,
// world triplanar, or a marched implicit's own `uv` projection), palette (indexed)
// maps, normal maps, and procedural `pattern` drives on the albedo and the emission —
// the last running the shared device pattern VM (pattern_device.cuh), the same one the
// path tracer uses. A render() call on any config still returns an empty vector on
// device failure so the caller can fall back to raster::renderFrame.
//
// This is a plain-C++ interface (no __device__ symbols leak out) so main.cpp (MSVC) can
// call into the nvcc-compiled translation unit raster_cuda.cu. When the project is built
// WITHOUT CUDA this header is not used (main.cpp guards on HAVE_CUDA).
#pragma once
#include <vector>
#include <cstdint>
#include "raster.h"    // raster::PTri, raster::PreviewLight
#include "texture.h"   // Texture (image skins baked to the device)
#include "camera.h"

namespace raster_cuda {

// True if a usable CUDA device is present (cached after first query). Cheap to call.
bool available();

// Opaque uploaded preview scene: the world-space triangle set baked to a device array
// (built ONCE and reused for every camera of a flyby), the distilled preview lights, the
// image-skin textures, and cached per-pixel device scratch buffers. Create with upload(),
// free with destroy().
struct Scene;

// Bake `geom` (the world-space preview triangles plus the side tables they index) +
// `light` + the source scene to the device. `scene` supplies everything the shade pass
// reaches beyond the triangle itself: the image skins bound by PTri::tex /
// PTri::normalTex, and the procedural `pattern` programs (plus their grid/scatter sample
// tables) bound by PTri::reflectPat / PTri::emitPat / PMix::weightPat. May be null, in
// which case textured and pattern-driven surfaces fall back to their flat colour.
// Returns nullptr if CUDA is unavailable or a device allocation fails (caller must then use
// the CPU path).
Scene* upload(const raster::PreviewGeom& geom, const raster::PreviewLight& light,
              const ::Scene* scene = nullptr);

// Free a scene created by upload() (safe on nullptr).
void destroy(Scene* sc);

// Render one camera to W*H*3 RGB8 (row 0 = image top), matching raster::renderFrame's
// format and exposure. Any projection (rectilinear or fisheye/panoramic) is supported, as
// are image skins and (when `seeThrough`) clear-glass compositing. `exposure`, `autoExpose`,
// `lockAnchor`, `seeThrough` and `glassClarity` have the same meaning as in
// raster::renderFrame (exposure/tonemap runs on the device, byte-identical to the host
// tail). Returns an EMPTY vector on any device failure so the caller can fall back to the
// CPU rasterizer.
std::vector<uint8_t> renderFrame(Scene* sc, const Camera& cam, int W, int H, int nThreads,
                                 double exposure = 1.0, bool autoExpose = true,
                                 double* lockAnchor = nullptr,
                                 bool seeThrough = false, double glassClarity = 0.85);

// ---- Zero-copy present (CUDA <-> Direct3D 11 interop) ----------------------------
// When the live preview window is presenting with D3D11 (see LiveWindow::renderShared),
// the finished frame does not have to come back to the host at all: CUDA can be given
// the very texture D3D samples and the tonemap can write its bytes into it in place.
// That removes the device->host image copy, the host->device re-upload, and every host
// touch of the pixels between them.
//
// bindPresentTarget() registers `d3d11Texture` (an ID3D11Texture2D*, RGBA8, W x H, from
// `d3d11Device`) with this scene, and is cheap to call every frame — it re-registers only
// when handed a different texture or size. It returns false when interop is unavailable:
// a non-Windows/HIP build, or (the common real case) D3D chose a different adapter than
// the CUDA device, as on hybrid iGPU/dGPU laptops.
//
// renderFrameToTarget() then renders exactly as renderFrame does — same geometry, same
// shading, same auto-exposure handshake, same tonemap maths, byte-for-byte — but stores
// the result into the registered texture instead of a downloadable buffer. It returns
// false on any failure, so the caller can fall back to renderFrame + update().
//
// Both must be called with the presenter's device lock held (LiveWindow::renderShared
// does this): the interop drives D3D's immediate context, which is not thread-safe, and
// D3D must not touch the texture between map and unmap.
bool bindPresentTarget(Scene* sc, void* d3d11Device, void* d3d11Texture, int W, int H);
bool renderFrameToTarget(Scene* sc, const Camera& cam, int W, int H, int nThreads,
                         double exposure = 1.0, bool autoExpose = true,
                         double* lockAnchor = nullptr,
                         bool seeThrough = false, double glassClarity = 0.85);

// Optional per-pass profiling (used by -raster-bench). While enabled, renderFrame
// records CUDA events into the stream between passes and accumulates each pass's
// GPU-timeline milliseconds into an internal tally (resolved once per frame after
// the final image download; the frame itself stays sync-free). Zero overhead when
// disabled. profTake() returns the tally accumulated since the last take and
// resets it; frames==0 means no GPU frames ran while enabled.
struct Prof {
    double clearvis_ms = 0;   // vis-buffer clear (cudaMemset)
    double project_ms  = 0;   // kProject (clip + project)
    double raster_ms   = 0;   // kRaster (visibility)
    double shade_ms    = 0;   // kShade (resolve + shade)
    double clear_ms    = 0;   // see-through clear pass (0 unless -see-through)
    double expose_ms   = 0;   // device expose: p99 histogram rounds + tonemap kernel
    double download_ms = 0;   // device->host copy of the final RGB8 image
    int    frames      = 0;
};
void profEnable(bool on);
Prof profTake();

}  // namespace raster_cuda
