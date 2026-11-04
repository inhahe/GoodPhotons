// OptiX's temporal denoiser: see optix_denoise.h.
// The normal guide is world space: optix_types.h says so for every model but the legacy LDR/HDR
// ones, and NVIDIA recommends it for them all on R535+ drivers (optix_host.h's older note asking
// camera space of the temporal models predates that).
#ifndef NOMINMAX
#define NOMINMAX   // optix_stubs.h brings <windows.h>, whose min/max macros would eat std::max
#endif
#include "optix_denoise.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace optixdn {

namespace {
bool g_usedCuda = false;

const char* name(OptixResult r) { return optixGetErrorName(r); }

// optixInit loads the driver's OptiX; tried once, its verdict kept.
bool ensureInit(std::string& err) {
    static bool tried = false, ok = false;
    static std::string why;
    if (!tried) {
        tried = true;
        if (cudaFree(nullptr) != cudaSuccess) {
            why = "no CUDA device";
        } else {
            g_usedCuda = true;
            const OptixResult r = optixInit();
            if (r != OPTIX_SUCCESS)
                why = std::string("optixInit failed (") + std::to_string((int)r) +
                      "): the display driver's OptiX did not load";
            else
                ok = true;
        }
    }
    if (!ok) err = why;
    return ok;
}

OptixImage2D image(CUdeviceptr p, int W, int H, OptixPixelFormat f, unsigned pixelBytes) {
    OptixImage2D im{};
    im.data = p;
    im.width = (unsigned)W;
    im.height = (unsigned)H;
    im.pixelStrideInBytes = pixelBytes;
    im.rowStrideInBytes = pixelBytes * (unsigned)W;
    im.format = f;
    return im;
}
}  // namespace

bool usedCuda() { return g_usedCuda; }

struct Temporal::Impl {
    OptixDeviceContext ctx = nullptr;
    OptixDenoiser den = nullptr;
    int W = 0, H = 0;
    OptixDenoiserSizes sizes{};
    CUdeviceptr state = 0, scratch = 0, intensity = 0;
    size_t scratchBytes = 0;
    CUdeviceptr color = 0, albedo = 0, normal = 0, flow = 0, out = 0, prevOut = 0, guideA = 0, guideB = 0;
    bool havePrev = false;

    ~Impl() { release(); if (ctx) optixDeviceContextDestroy(ctx); }

    void release() {
        if (den) { optixDenoiserDestroy(den); den = nullptr; }
        for (CUdeviceptr* p : {&state, &scratch, &intensity, &color, &albedo, &normal, &flow, &out,
                               &prevOut, &guideA, &guideB})
            if (*p) { cudaFree(reinterpret_cast<void*>(*p)); *p = 0; }
        W = H = 0;
        havePrev = false;
    }

    bool alloc(CUdeviceptr& p, size_t bytes, std::string& err) {
        void* q = nullptr;
        if (cudaMalloc(&q, bytes) != cudaSuccess) { err = "cudaMalloc failed"; return false; }
        p = reinterpret_cast<CUdeviceptr>(q);
        return true;
    }

    bool setup(int w, int h, std::string& err) {
        release();
        if (!ctx) {
            OptixDeviceContextOptions o{};
            const OptixResult r = optixDeviceContextCreate(nullptr, &o, &ctx);   // the current context
            if (r != OPTIX_SUCCESS) { err = std::string("optixDeviceContextCreate: ") + name(r); return false; }
        }
        OptixDenoiserOptions o{};
        o.guideAlbedo = 1;
        o.guideNormal = 1;
        o.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
        OptixResult r = optixDenoiserCreate(ctx, OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV, &o, &den);
        if (r != OPTIX_SUCCESS) { err = std::string("optixDenoiserCreate: ") + name(r); return false; }
        r = optixDenoiserComputeMemoryResources(den, (unsigned)w, (unsigned)h, &sizes);
        if (r != OPTIX_SUCCESS) { err = std::string("optixDenoiserComputeMemoryResources: ") + name(r); return false; }
        scratchBytes = std::max(sizes.withoutOverlapScratchSizeInBytes, sizes.computeIntensitySizeInBytes);
        const size_t px = (size_t)w * (size_t)h;
        if (!alloc(state, sizes.stateSizeInBytes, err) || !alloc(scratch, scratchBytes, err) ||
            !alloc(intensity, sizeof(float), err) || !alloc(color, px * 12, err) ||
            !alloc(albedo, px * 12, err) || !alloc(normal, px * 12, err) || !alloc(flow, px * 8, err) ||
            !alloc(out, px * 12, err) || !alloc(prevOut, px * 12, err) ||
            !alloc(guideA, px * sizes.internalGuideLayerPixelSizeInBytes, err) ||
            !alloc(guideB, px * sizes.internalGuideLayerPixelSizeInBytes, err))
            return false;
        r = optixDenoiserSetup(den, nullptr, (unsigned)w, (unsigned)h, state, sizes.stateSizeInBytes,
                               scratch, scratchBytes);
        if (r != OPTIX_SUCCESS) { err = std::string("optixDenoiserSetup: ") + name(r); return false; }
        W = w; H = h;
        return true;
    }
};

Temporal::Temporal() : impl_(new Impl) {}
Temporal::~Temporal() { delete impl_; }

bool Temporal::denoise(float* color, const float* albedo, const float* normal, const float* motionBack,
                       int W, int H, std::string& err) {
    if (!ensureInit(err)) return false;
    Impl& m = *impl_;
    if (W != m.W || H != m.H || !m.den)
        if (!m.setup(W, H, err)) return false;
    const size_t px = (size_t)W * (size_t)H;
    const auto up = [&](CUdeviceptr d, const float* s, size_t bytes) {
        return cudaMemcpy(reinterpret_cast<void*>(d), s, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    };
    if (!up(m.color, color, px * 12) || !up(m.albedo, albedo, px * 12) || !up(m.normal, normal, px * 12) ||
        !up(m.flow, motionBack, px * 8)) {
        err = "cudaMemcpy to the device failed";
        return false;
    }
    const OptixImage2D in = image(m.color, W, H, OPTIX_PIXEL_FORMAT_FLOAT3, 12);
    OptixResult r = optixDenoiserComputeIntensity(m.den, nullptr, &in, m.intensity, m.scratch, m.scratchBytes);
    if (r != OPTIX_SUCCESS) { err = std::string("optixDenoiserComputeIntensity: ") + name(r); return false; }
    OptixDenoiserParams p{};
    p.hdrIntensity = m.intensity;
    p.blendFactor = 0.0f;
    p.temporalModeUsePreviousLayers = m.havePrev ? 1u : 0u;
    // OptiX's flow runs from the previous frame to this one; ours points back, so it is negated.
    // FTRACE_OPTIX_FLOWMUL="x,y" (diagnostic) overrides the two factors, to measure a convention.
    p.flowMulX = -1.0f;
    p.flowMulY = -1.0f;
    if (const char* fm = std::getenv("FTRACE_OPTIX_FLOWMUL")) {
        float x = 0.0f, y = 0.0f;
        if (std::sscanf(fm, "%f,%f", &x, &y) == 2) { p.flowMulX = x; p.flowMulY = y; }
    }
    OptixDenoiserGuideLayer g{};
    g.albedo = image(m.albedo, W, H, OPTIX_PIXEL_FORMAT_FLOAT3, 12);
    g.normal = image(m.normal, W, H, OPTIX_PIXEL_FORMAT_FLOAT3, 12);
    g.flow = image(m.flow, W, H, OPTIX_PIXEL_FORMAT_FLOAT2, 8);
    const unsigned gp = (unsigned)m.sizes.internalGuideLayerPixelSizeInBytes;
    g.previousOutputInternalGuideLayer = image(m.guideA, W, H, OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER, gp);
    g.outputInternalGuideLayer = image(m.guideB, W, H, OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER, gp);
    OptixDenoiserLayer L{};
    L.input = in;
    L.previousOutput = m.havePrev ? image(m.prevOut, W, H, OPTIX_PIXEL_FORMAT_FLOAT3, 12) : in;
    L.output = image(m.out, W, H, OPTIX_PIXEL_FORMAT_FLOAT3, 12);
    L.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;
    r = optixDenoiserInvoke(m.den, nullptr, &p, m.state, m.sizes.stateSizeInBytes, &g, &L, 1, 0, 0,
                            m.scratch, m.scratchBytes);
    if (r != OPTIX_SUCCESS) { err = std::string("optixDenoiserInvoke: ") + name(r); return false; }
    if (cudaMemcpy(color, reinterpret_cast<void*>(m.out), px * 12, cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "cudaMemcpy from the device failed";
        return false;
    }
    std::swap(m.out, m.prevOut);      // this result is the next frame's previous output
    std::swap(m.guideA, m.guideB);    // and its internal guide, the next frame's previous one
    m.havePrev = true;
    return true;
}

}  // namespace optixdn
