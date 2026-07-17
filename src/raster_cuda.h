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
// raster::renderFrame (the shared host tail applies exposure). Returns an EMPTY vector on
// any device failure so the caller can fall back to the CPU rasterizer.
std::vector<uint8_t> renderFrame(Scene* sc, const Camera& cam, int W, int H, int nThreads,
                                 double exposure = 1.0, bool autoExpose = true,
                                 double* lockAnchor = nullptr,
                                 bool seeThrough = false, double glassClarity = 0.85);

}  // namespace raster_cuda
