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
// maps — equidistant, equisolid, stereographic, orthographic), OPAQUE geometry, image
// skins (per-vertex UV + world-triplanar textures, sampled on-device), and see-through
// (clear-glass) compositing (a device clear-accumulation pass mirrors the CPU one). A
// render() call on any config still returns an empty vector on device failure so the
// caller can fall back to raster::renderFrame.
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

// Bake `tris` (world-space preview triangles) + `light` + `textures` (image skins bound by
// PTri::tex; may be null/empty) to the device. Returns nullptr if CUDA is unavailable or a
// device allocation fails (caller must then use the CPU path).
Scene* upload(const std::vector<raster::PTri>& tris, const raster::PreviewLight& light,
              const std::vector<Texture>* textures = nullptr);

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
