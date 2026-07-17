// raster_cuda.h — GPU (CUDA) backend for the solid-shaded PREVIEW rasterizer.
//
// This is the device twin of the CPU rasterizer in raster.h. It runs the same
// deferred-visibility pipeline — project+clip each world triangle, resolve the nearest
// surface per pixel, then shade it once — on the GPU, and returns a byte-identical-format
// RGB8 frame. Only the GEOMETRY + SHADING (the embarrassingly parallel part) runs on the
// device; the downloaded HDR buffer is exposed and tone-mapped by the SAME host code the
// CPU path uses (raster::exposeAndEncode), so a camera_path's shared auto-exposure anchor
// and the sRGB encoding are bit-identical regardless of backend.
//
// Scope: all camera projections (RECTILINEAR pinhole plus the fisheye/panoramic lens
// maps — equidistant, equisolid, stereographic, orthographic) and OPAQUE geometry. Only
// the see-through (clear-glass) compositing still falls back to the CPU. The caller is
// responsible for that gate; a render() call on an unsupported config simply returns an
// empty vector so the caller can fall back to raster::renderFrame.
//
// This is a plain-C++ interface (no __device__ symbols leak out) so main.cpp (MSVC) can
// call into the nvcc-compiled translation unit raster_cuda.cu. When the project is built
// WITHOUT CUDA this header is not used (main.cpp guards on HAVE_CUDA).
#pragma once
#include <vector>
#include <cstdint>
#include "raster.h"    // raster::PTri, raster::PreviewLight
#include "camera.h"

namespace raster_cuda {

// True if a usable CUDA device is present (cached after first query). Cheap to call.
bool available();

// Opaque uploaded preview scene: the world-space triangle set baked to a device array
// (built ONCE and reused for every camera of a flyby), the distilled preview lights, and
// cached per-pixel device scratch buffers. Create with upload(), free with destroy().
struct Scene;

// Bake `tris` (world-space preview triangles) + `light` to the device. Returns nullptr if
// CUDA is unavailable or a device allocation fails (caller must then use the CPU path).
Scene* upload(const std::vector<raster::PTri>& tris, const raster::PreviewLight& light);

// Free a scene created by upload() (safe on nullptr).
void destroy(Scene* sc);

// Render one camera to W*H*3 RGB8 (row 0 = image top), matching raster::renderFrame's
// format and exposure. Any projection (rectilinear or fisheye/panoramic) is supported.
// `exposure`, `autoExpose` and `lockAnchor` have the same meaning as in
// raster::renderFrame (the shared host tail applies them). Returns an EMPTY vector on any
// device failure so the caller can fall back to the CPU rasterizer.
std::vector<uint8_t> renderFrame(Scene* sc, const Camera& cam, int W, int H, int nThreads,
                                 double exposure = 1.0, bool autoExpose = true,
                                 double* lockAnchor = nullptr);

}  // namespace raster_cuda
