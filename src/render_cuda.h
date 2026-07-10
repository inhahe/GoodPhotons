// CUDA backend for the forward light tracer (model B).
//
// This is a plain-C++ interface so main.cpp (compiled by MSVC) can call into the
// CUDA translation unit (render_cuda.cu, compiled by nvcc) without ever seeing a
// __device__ symbol. The .cu file's HOST side includes the project headers, bakes
// the std::function-based Scene into POD device tables (sampled spectra, flat BVH,
// POD materials), uploads them, launches a megakernel that mirrors
// Renderer::tracePhoton, then downloads the accumulated film.
//
// The GPU covers the three forward camera models: A (contact sensor deposit),
// B (connect/splat to the pinhole — the default, and the one mode V validates), and
// C (finite-aperture thin-lens forward catch). Fluorescence is NOT supported on the
// device (it needs the emission-sampler reradiation path); a scene containing a
// Fluorescent material must fall back to the CPU. The caller is responsible for that
// check via cudaForwardSupported().
#pragma once
#include "scene.h"
#include "camera.h"
#include "render.h"   // EnergyReport, Film

// True if a usable CUDA device is present (driver + at least one device). Cheap to
// call; result is cached after the first query.
bool cudaAvailable();

// Human-readable name of the primary CUDA device (or "none").
const char* cudaDeviceName();

// True if this scene can be rendered on the GPU (no unsupported material such as
// Fluorescent). When false, the caller must use the CPU renderer.
bool cudaForwardSupported(const Scene& scene);

// GPU forward light trace. camMode selects the camera model: 'A' (contact-sensor
// deposit), 'B' (connect/splat to the pinhole), or 'C' (finite-aperture forward
// catch). Traces N photons and returns the accumulated camera film (same
// units/convention as the CPU renderForward for the matching mode), so
// writePPM(film, N) and the mode-V comparison work unchanged. Fills eOut with the
// same energy report. Requires cudaAvailable() && cudaForwardSupported(scene);
// otherwise returns an empty film.
Film renderForwardCuda(const Scene& scene, const Camera& cam, int res,
                       long long N, EnergyReport& eOut, bool diffraction,
                       char camMode);
