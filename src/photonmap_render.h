// Photon-mapped rendering (ROADMAP item 1, mode M).
//
// Two passes:
//   1. tracePhotonPass(): forward light-trace N photons with the camera splat OFF and
//      the deposit path ON, filling a PhotonMap (view-independent radiance cache).
//   2. renderPhotonCamera(): a backward camera pass that, at the first diffuse hit of
//      each camera ray, estimates reflected radiance by a radius density query into the
//      map. Direct + specular reach the diffuse surface normally; the map supplies the
//      (direct + indirect) diffuse illumination.
//
// The map is built ONCE and can be reused for many cameras of a static scene — the
// flythrough win (build once, gather per frame). See renderPhotonCameraMulti().
#pragma once
#include <vector>
#include <thread>
#include <cstdint>
#include "render.h"
#include "photonmap.h"
#include "scene_film.h"
#include "camera.h"
#include "color.h"
#include "geometry.h"

// ---- Forward photon pass: deposit into the map, no camera splat ---------------------
// Traces N photons across nThreads, each depositing into a private bank, then
// concatenates into pm.photons and records pm.nEmitted (= N). Does NOT build the grid;
// the caller picks the gather radius and calls pm.build(radius).
inline void tracePhotonPass(const Scene& scene, long long N, int nThreads,
                            bool diffraction, PhotonMap& pm) {
    if (nThreads < 1) nThreads = 1;
    std::vector<std::vector<Photon>> banks(nThreads);
    std::vector<long long> emitted(nThreads, 0);

    auto worker = [&](int tid) {
        Renderer r; r.diffraction = diffraction; r.photonDeposit = &banks[tid];
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 31,
                            0xD1B54A32D192ED03ULL ^ (uint64_t)tid);
        long long lo = N * tid / nThreads, hi = N * (tid + 1) / nThreads;
        EnergyReport e;
        for (long long i = lo; i < hi; ++i)
            r.tracePhoton(scene, (const Camera*)nullptr, (Film*)nullptr, (Film*)nullptr, rng, e);
        emitted[tid] = hi - lo;
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    size_t total = 0;
    for (auto& b : banks) total += b.size();
    pm.photons.clear();
    pm.photons.reserve(total);
    pm.nEmitted = 0;
    for (int t = 0; t < nThreads; ++t) {
        pm.photons.insert(pm.photons.end(), banks[t].begin(), banks[t].end());
        pm.nEmitted += emitted[t];
    }
}

// ---- Camera gather: radiance along one camera ray via the photon map ----------------
// Returns the XYZ contribution. The ray is followed through specular surfaces
// (monochromatic at a sampled lambda); at the first diffuse/translucent hit the reflected
// radiance is estimated by a radius query, with each photon reflected at ITS OWN
// wavelength (density estimate built directly in XYZ). Directly-viewed emitters and the
// environment are added as a monochromatic estimate at the sampled lambda.
inline Vec3 photonGather(const Scene& scene, const PhotonMap& pm, Ray ray,
                         Pcg32& rng, bool diffraction, int maxBounce) {
    Vec3 L{0, 0, 0};
    double thr = 1.0;
    double pdfL = 0.0;
    double lambda = scene.emitSampler.sample(rng, pdfL);
    if (pdfL <= 0.0) return L;
    const double invPdfL = scene.invPdfLambda(lambda);

    Renderer mats; mats.diffraction = diffraction;
    const Material* interior = nullptr;
    const double area = PI * pm.radius * pm.radius;
    const double norm = (pm.nEmitted > 0 && area > 0.0)
                            ? 1.0 / (area * (double)pm.nEmitted) : 0.0;

    for (int b = 0; b < maxBounce; ++b) {
        Hit h = scene.closestHit(ray);
        if (interior && h.valid) {                       // Beer-Lambert in glass
            double a = interior->absorb(lambda);
            if (a > 0.0) thr *= std::exp(-a * h.t);
        }
        if (!h.valid) {                                  // escaped -> environment
            if (scene.envIndex >= 0)
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * scene.envRadiance(ray.d, lambda) * invPdfL);
            return L;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return L;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            int c = mixPickChild(*mp, rng.uniform());    // approximate: gather the body lobe
            if (c < 0) return L;
            mp = &scene.mats[c];
        }
        const Material& m = *mp;

        if (m.isLight) {                                 // directly-viewed emitter
            L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                 * (thr * m.emit(lambda) * invPdfL);
            return L;
        }

        switch (m.type) {
            case MatType::Diffuse:
            case MatType::DiffuseTransmit:
            case MatType::Fluorescent: {
                // Radius density estimate: L_r(x) = (1/N) sum_p f_r * Phi_p / (pi r^2),
                // f_r = rho/pi (Lambertian), accumulated in XYZ per photon wavelength.
                Vec3 g{0, 0, 0};
                pm.query(h.p, [&](const Photon& ph, double) {
                    if (dot(ph.n, h.n) < 0.5) return;    // reject cross-surface leakage
                    double rho = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                    double f = rho * (1.0 / PI);
                    g += Vec3(cieX(ph.lambda), cieY(ph.lambda), cieZ(ph.lambda))
                         * (f * (double)ph.power);
                });
                L += g * (norm * thr);
                return L;
            }
            case MatType::Mirror: {
                double r = clamp01(m.reflect(lambda));
                thr *= r;
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
            case MatType::Glossy: {
                double r = clamp01(m.reflect(lambda));
                thr *= r;
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0.0) return L;
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
                double t = clamp01(m.transmit(lambda));
                thr *= t;
                ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            default: {                                   // ThinFilm/Multilayer/Grating: approx reflect
                double r = clamp01(m.reflect(lambda));
                thr *= r;
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
        }
        if (thr <= 0.0) return L;
    }
    return L;
}

// ---- Camera pass driver (single camera) ---------------------------------------------
// Accumulates a SUM over spp (display divides by spp via writeFilm), matching the
// backward/BDPT convention so a chunked/progressive render sums batches.
inline Film renderPhotonCamera(const Scene& scene, const Camera& cam, int resX, int resY,
                               const PhotonMap& pm, long long spp, int nThreads,
                               bool diffraction, int maxBounce = 32,
                               unsigned long long seedOffset = 0) {
    if (nThreads < 1) nThreads = 1;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    std::vector<Film> bands(nThreads);
    auto worker = [&](int tid) {
        Film& f = bands[tid]; f.resX = resX; f.resY = resY; f.alloc();
        Pcg32 rng; rng.seed((uint64_t)tid * 2 + 23,
                            0xA24BAED4963EE407ULL ^ (uint64_t)tid
                              ^ (seedOffset * 0x9E3779B97F4A7C15ULL));
        int y0 = resY * tid / nThreads, y1 = resY * (tid + 1) / nThreads;
        for (int py = y0; py < y1; ++py)
            for (int px = 0; px < resX; ++px)
                for (long long s = 0; s < spp; ++s) {
                    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
                    f.add(px, py, photonGather(scene, pm, ray, rng, diffraction, maxBounce));
                }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    for (int t = 0; t < nThreads; ++t) out.merge(bands[t]);
    return out;
}
