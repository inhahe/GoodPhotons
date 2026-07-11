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
// writeFilm(film, N) and the mode-V comparison work unchanged. Fills eOut with the
// same energy report. Requires cudaAvailable() && cudaForwardSupported(scene);
// otherwise returns an empty film.
// `seedBase` offsets the RNG stream so successive calls (render-in-chunks for a time
// budget, or resuming an accumulated film) draw statistically-independent photons;
// pass the cumulative photon count already traced. seedBase==0 reproduces the
// original single-shot stream bit-for-bit.
// `wavefront` selects the streaming (path-regeneration) GPU backend instead of the
// default megakernel. Both run identical physics and conserve energy exactly; the
// wavefront scheduler keeps SIMD lanes full on divergent / deep-path scenes and small
// GPUs, at the cost of extra memory traffic (the RNG stream — and thus the exact image
// noise — differs, but the two agree to within Monte-Carlo noise).
Film renderForwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                       long long N, EnergyReport& eOut, bool diffraction,
                       char camMode, unsigned long long seedBase = 0,
                       bool wavefront = false);

// True if this scene can be rendered by the GPU BDPT megakernel (mode D). Stricter
// than cudaForwardSupported: also requires no participating media and only area/sphere/
// cylinder Lambertian emitters (no spot/env/collimated) — the BDPT scope. When false, the
// caller must use the CPU BDPT renderer.
bool cudaBdptSupported(const Scene& scene);

// GPU bidirectional path trace (mode D). Renders `spp` samples per pixel at the given
// resolution and returns the final absolute-radiance film (same units/convention as
// the CPU renderBdpt, i.e. writeFilm(film, 1.0) for display). maxDepth is the maximum
// path length in edges (clamped to the device capacity). Requires cudaAvailable() &&
// cudaBdptSupported(scene); otherwise returns an empty film.
Film renderBdptCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                    long long spp, int maxDepth, bool diffraction);

// True if this scene + camera can be rendered by the GPU backward reference megakernel
// (mode R), including the physical (mesh-lens) camera as a ray-generation front-end.
// v1 scope: no participating media, no environment light, only area/sphere/cylinder
// Lambertian emitters (no spot/env/collimated), no fluorescence, and a lens no deeper
// than the device cap (D_MAXLENS). Textured albedo IS supported. When false, the
// caller must use the CPU backward tracer (which has no such restrictions).
bool cudaBackwardSupported(const Scene& scene, const Camera& cam);

// GPU backward reference trace (mode R). Renders `spp` samples per pixel at the given
// resolution and returns the film accumulated over spp (same convention as the CPU
// renderBackward: writeFilm(film, spp) for display). Handles the physical lens exactly
// as the CPU path (film + rear-pupil sample refracted through the glass stack, with the
// cos^4*A/Z^2 radiometric weight). Requires cudaAvailable() && cudaBackwardSupported();
// otherwise returns an empty film. The device RNG differs from the CPU, so the image is
// an independent noise realization that agrees to within Monte-Carlo noise.
Film renderBackwardCuda(const Scene& scene, const Camera& cam, int resX, int resY,
                        long long spp, bool diffraction);
