// Stochastic Progressive Photon Mapping (ROADMAP item 2, mode S).
//
// SPPM (Hachisuka & Jensen 2009) removes the two weaknesses of the single-pass photon
// map (mode M): the fixed-radius bias and the unbounded memory of one giant map. Instead
// it runs REPEATED photon passes with a bounded per-pass map and SHRINKS a per-pixel
// gather radius over iterations, so the estimate converges to the unbiased result while
// memory stays flat. Each pass also re-samples the camera subpaths (the "stochastic" in
// SPPM), so it is robust for distributed effects (anti-aliasing, DoF, glossy) and, above
// all, resolves caustics / SDS paths that a backward path tracer (R) and even BDPT (D)
// find slowly.
//
// Per pixel we keep (Hachisuka 2008 shared-statistics form):
//   R   — current gather radius            (starts at R0, only shrinks)
//   nAcc— accumulated photon count         (real-valued, grows by alpha*M each pass)
//   tau — accumulated, radius-rescaled flux (XYZ)
//   directSum/passes — the emitter seen directly or through specular (a plain MC average)
// After a pass that finds M photons in radius R with local flux Phi:
//   N' = nAcc + alpha*M
//   R' = R * sqrt(N' / (nAcc + M))
//   tau' = (tau + thr*Phi) * (R'^2 / R^2)
// and the final radiance is  L = tau / (pi*R^2 * Nemitted_total) + directSum/passes,
// which for a single pass reduces EXACTLY to mode M's estimate (validated equivalence).
#pragma once
#include <vector>
#include <thread>
#include <cstdint>
#include <cmath>
#include "render.h"
#include "photonmap.h"
#include "photonmap_render.h"   // tracePhotonPass, shared specular-walk conventions
#include "scene_film.h"
#include "camera.h"
#include "color.h"
#include "geometry.h"

// Per-pixel progressive state.
struct SPPMPixel {
    Vec3   tau{0, 0, 0};      // accumulated (radius-rescaled) flux, XYZ
    double radius = 0.0;      // current gather radius R_i
    double nAcc   = 0.0;      // accumulated photon count N_i (real-valued)
    Vec3   directSum{0, 0, 0};// direct/specular-viewed emitter + env, summed over passes
    // A per-pass "visible point": the first diffuse hit of this pass's camera ray.
    Hit    vpHit;             // diffuse hit record (pos/normal/uv/matId for BSDF eval)
    double vpThr = 0.0;       // specular throughput camera -> visible point (0 = none)
    bool   vpValid = false;
};

struct SPPMState {
    int resX = 0, resY = 0;
    std::vector<SPPMPixel> px;
    long long emittedTotal = 0;   // photons emitted across ALL passes (normalization)
    long long passes = 0;
    void init(int w, int h, double R0) {
        resX = w; resY = h;
        px.assign((size_t)w * h, {});
        for (auto& p : px) p.radius = R0;
        emittedTotal = 0; passes = 0;
    }
};

// Trace one camera ray to its first diffuse/translucent hit (the "visible point"),
// following specular surfaces exactly like photonGather. Returns the specular throughput
// in `thr` and the diffuse hit in `vp`; any emitter/environment radiance reached through
// specular is added to `directL` (a monochromatic MC estimate at the sampled lambda).
// `vpValid` is false when the ray terminated (on a light, env, or absorption) without
// reaching a diffuse surface.
inline void sppmVisiblePoint(const Scene& scene, Ray ray, Pcg32& rng, bool diffraction,
                             int maxBounce, Hit& vp, double& thrOut, bool& vpValid,
                             Vec3& directL) {
    directL = Vec3{0, 0, 0};
    thrOut = 0.0; vpValid = false;

    double thr = 1.0, pdfL = 0.0;
    double lambda = scene.emitSampler.sample(rng, pdfL);
    if (pdfL <= 0.0) return;
    const double invPdfL = scene.invPdfLambda(lambda);

    Renderer mats; mats.diffraction = diffraction;
    const Material* interior = nullptr;

    for (int b = 0; b < maxBounce; ++b) {
        Hit h = scene.closestHit(ray);
        if (interior && h.valid) {
            double a = interior->absorb(lambda);
            if (a > 0.0) thr *= std::exp(-a * h.t);
        }
        if (!h.valid) {
            if (scene.envIndex >= 0)
                directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                           * (thr * scene.envRadiance(ray.d, lambda) * invPdfL);
            return;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            int c = mixPickChild(*mp, rng.uniform());
            if (c < 0) return;
            mp = &scene.mats[c];
        }
        const Material& m = *mp;

        if (m.isLight) {
            directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                       * (thr * m.emit(lambda) * invPdfL);
            return;
        }

        switch (m.type) {
            case MatType::Diffuse:
            case MatType::DiffuseTransmit:
            case MatType::Fluorescent:
                vp = h; thrOut = thr; vpValid = true;   // record the visible point
                return;
            case MatType::Mirror: {
                thr *= clamp01(m.reflect(lambda));
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
            case MatType::Glossy: {
                thr *= clamp01(m.reflect(lambda));
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0.0) return;
                ray = Ray{h.p + h.n * 1e-6, o};
                break;
            }
            case MatType::Dielectric: {
                bool entering = dot(ray.d, h.ng) < 0.0;
                bool transmitted = false;
                ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted);
                if (transmitted) interior = entering ? &m : nullptr;
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(m.reflect(lambda));
                if (rng.uniform() < r) ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                else                   ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Filter: {
                thr *= clamp01(m.transmit(lambda));
                ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            default: {
                thr *= clamp01(m.reflect(lambda));
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
        }
        if (thr <= 0.0) return;
    }
}

// Run ONE SPPM pass: (1) re-sample the camera visible points, (2) trace M photons into a
// bounded map, (3) gather each visible point at its current radius and apply the
// progressive radius/flux update. Updates `st` in place.
inline void sppmPass(const Scene& scene, const Camera& cam, SPPMState& st,
                     long long photonsPerPass, int nThreads, bool diffraction,
                     double alpha, int maxBounce, uint64_t passSeed) {
    if (nThreads < 1) nThreads = 1;
    const int W = st.resX, H = st.resY;

    // (1) Camera pass — fresh visible point + direct sample per pixel this pass.
    {
        auto camWorker = [&](int tid) {
            Pcg32 rng; rng.seed(passSeed * 0x9E3779B97F4A7C15ULL + (uint64_t)tid * 2 + 23,
                                0xA24BAED4963EE407ULL ^ (uint64_t)tid ^ (passSeed << 1));
            int y0 = H * tid / nThreads, y1 = H * (tid + 1) / nThreads;
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x) {
                    SPPMPixel& P = st.px[(size_t)y * W + x];
                    Ray ray = cam.genRay(x, y, rng.uniform(), rng.uniform());
                    Vec3 directL;
                    sppmVisiblePoint(scene, ray, rng, diffraction, maxBounce,
                                     P.vpHit, P.vpThr, P.vpValid, directL);
                    P.directSum += directL;
                }
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < nThreads; ++t) pool.emplace_back(camWorker, t);
        for (auto& th : pool) th.join();
    }

    // (2) Photon pass into a bounded map. Build the grid at the LARGEST current per-pixel
    // radius so every pixel's (never-larger) radius stays within the 3x3x3 neighbourhood.
    double rMax = 0.0;
    for (const auto& P : st.px) if (P.vpValid) rMax = std::max(rMax, P.radius);
    if (rMax <= 0.0) rMax = 1e-4;
    PhotonMap pm;
    tracePhotonPass(scene, photonsPerPass, nThreads, diffraction, pm);
    pm.build(rMax);
    st.emittedTotal += pm.nEmitted;
    st.passes += 1;

    // (3) Gather + progressive update, parallel over pixels (each pixel is independent).
    auto gatherWorker = [&](int tid) {
        int y0 = H * tid / nThreads, y1 = H * (tid + 1) / nThreads;
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x) {
                SPPMPixel& P = st.px[(size_t)y * W + x];
                if (!P.vpValid) continue;
                const Hit& h = P.vpHit;
                const Material& m = scene.mats[h.matId];
                double M = 0.0;          // photons found this pass
                Vec3   phi{0, 0, 0};     // local flux sum (XYZ, per-photon wavelength)
                pm.queryR(h.p, P.radius, [&](const Photon& ph, double) {
                    if (dot(ph.n, h.n) < 0.5) return;   // reject cross-surface leakage
                    double rho = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                    double f = rho * (1.0 / PI);
                    phi += Vec3(cieX(ph.lambda), cieY(ph.lambda), cieZ(ph.lambda))
                           * (f * (double)ph.power);
                    M += 1.0;
                });
                // Progressive radius / flux update (shared-statistics PPM).
                double Nnew = P.nAcc + alpha * M;
                double denom = P.nAcc + M;
                double ratio2 = (denom > 0.0) ? (Nnew / denom) : 1.0;   // (R'/R)^2
                P.tau = (P.tau + phi * P.vpThr) * ratio2;
                P.radius *= std::sqrt(ratio2);
                P.nAcc = Nnew;
            }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(gatherWorker, t);
    for (auto& th : pool) th.join();
}

// Resolve the current radiance image from the accumulated SPPM state:
//   L = tau / (pi*R^2 * Nemitted_total)   [indirect, from the density estimate]
//     + directSum / passes                [direct/specular-viewed emitter, MC average]
inline Film sppmResolve(const SPPMState& st) {
    Film f; f.resX = st.resX; f.resY = st.resY; f.alloc();
    const double invPasses = (st.passes > 0) ? 1.0 / (double)st.passes : 0.0;
    const double Nemit = (double)st.emittedTotal;
    for (int y = 0; y < st.resY; ++y)
        for (int x = 0; x < st.resX; ++x) {
            const SPPMPixel& P = st.px[(size_t)y * st.resX + x];
            Vec3 L = P.directSum * invPasses;
            if (P.vpValid || P.nAcc > 0.0) {
                double area = PI * P.radius * P.radius;
                if (area > 0.0 && Nemit > 0.0)
                    L += P.tau * (1.0 / (area * Nemit));
            }
            f.xyz[(size_t)y * st.resX + x] = L;
            f.hits[(size_t)y * st.resX + x] = 1.0;
        }
    return f;
}
