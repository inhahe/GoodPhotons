// OptiX's temporal denoiser (`-denoise-shot optix`, ROADMAP 11 stage D).
//
// OptiX lives in the NVIDIA display driver; building against it needs only NVIDIA's headers
// (github.com/NVIDIA/optix-dev), which CMake finds in external/optix-dev and which are not
// vendored -- their licence allows redistributing binaries built with them, not the headers.
// Without them ftrace builds without this file and `-denoise-shot optix` says why. Kept in its
// own translation unit so the OptiX headers stay out of main.cpp.
#pragma once
#include <string>

namespace optixdn {

// One shot's denoiser (OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV): frames are handed to it in
// order, and each is denoised against the previous frame's result.
class Temporal {
public:
    Temporal();
    ~Temporal();
    Temporal(const Temporal&) = delete;
    Temporal& operator=(const Temporal&) = delete;
    // Denoise one frame in place.
    //   color     W*H RGB float triples, scene-linear; overwritten with the result
    //   albedo    the guide albedo, same layout
    //   normal    the guide normal, world space, same layout
    //   motionBack  W*H float PAIRS: each pixel's offset to where its surface sat a frame
    //             earlier, in pixels along the image's x and along its rows (dguide's .motion)
    // Row order is the caller's, the same for every image. The first call, or one after a change
    // of size, starts a new sequence. Returns false with `err` set if OptiX cannot be loaded or a
    // call fails; a failed load is remembered.
    bool denoise(float* color, const float* albedo, const float* normal, const float* motionBack,
                 int W, int H, std::string& err);
private:
    struct Impl;
    Impl* impl_;
};

// Whether denoise() made a CUDA context (main's teardown must then run).
bool usedCuda();

}  // namespace optixdn
