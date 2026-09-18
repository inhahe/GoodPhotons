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
    MediumStack stk;                                     // nested-dielectric medium stack
    // GRADIENT-INDEX: the camera ray bends, exactly as mode M's gather does. Without this
    // the SPPM deposit marched its photons through a GRIN lens while the view walked
    // straight past it, and the lens rendered flat.
    const bool grinAny = grin::sceneHasGrin(scene);

    // GLOSSY-NEE (known-issues.md): mode S's camera walk had the same hole as mode R's and mode
    // M's -- a Glossy vertex multiplied by the reflectance and continued, so its light was found
    // only when a lobe sample happened to land on the emitter. `bwNee` is the shared estimator;
    // `gmis` carries the continuation's lobe density to the two sites that can reach a light.
    BackwardRenderer bwNee; bwNee.diffraction = diffraction;
    const bool gneeOn = BackwardRenderer::glossyNeeOn();
    BackwardRenderer::GlossyMis gmis;

    for (int b = 0; b < maxBounce; ++b) {
        if (grinAny) {
            double arc = 0.0;
            grin::marchSegments(scene, ray,
                [&](const Vec3&, const Vec3&, double slen, double&) { arc += slen; return false; },
                // b == 0 is the camera ray; see Material::hideCamera.
                /*camHide=*/(b == 0));
            int cm = stk.topMat();                       // Beer-Lambert over the marched arc
            double a = (cm >= 0) ? scene.mats[cm].absorb(lambda) : 0.0;
            if (a > 0.0 && arc > 0.0) thr *= std::exp(-a * arc);
        }
        // b == 0 is the camera ray this function was handed; see Material::hideCamera.
        Hit h = scene.closestHit(ray, 1e-6, nullptr, /*skipHair=*/false,
                                 /*skipCamHidden=*/(b == 0));
        if (h.valid) {
            int cm = stk.topMat();
            double a = (cm >= 0) ? scene.mats[cm].absorb(lambda) : 0.0;
            if (a > 0.0) thr *= std::exp(-a * h.t);
        }
        if (!h.valid) {
            if (scene.envIndex >= 0)
                directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                           * (thr * scene.envRadiance(ray.d, lambda) * invPdfL);
            // Directly-viewed solar disc (camera / specular escapes only — a diffuse
            // vertex stores a hit point and returns before it can reach here).
            if (scene.sunCount > 0)
                directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                           * (thr * bwNee.sunRadianceMis(scene, gmis, ray.d, lambda) * invPdfL);
            return;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());  // body lobe: honours a bound weight map (device twin: dResolveCompound)
            if (c < 0) return;
            mp = &scene.mats[c];
        }
        const Material& m = *mp;

        if (m.isLight) {
            // GLOSSY-NEE's lobe-sampling half; 1, and this expression bit-identical to its
            // pre-0.266 form, unless the previous bounce was a MIS'd glossy one.
            const double wMis = (gmis.pdf > 0.0)
                ? bwNee.glossyHitWeight(scene, gmis, BackwardRenderer::emitterIndexOfResolved(scene, m),
                                        ray.d, &h.p, &h.n)
                : 1.0;
            directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                       * (thr * emitSlot(scene, m, h, lambda) * invPdfL * wMis);
            return;
        }

        // GLOSSY-NEE: cleared HERE, after the two sites that read it and before the branch that
        // writes it -- a clear at the loop top would erase the previous bounce's value a few
        // lines before its only consumer, leaving the connection with no compensating weight.
        // See the twin note in backward.h.
        gmis.clear();

        switch (m.type) {
            case MatType::Diffuse:
            case MatType::DiffuseTransmit:
            case MatType::Fluorescent:
                vp = h; thrOut = thr; vpValid = true;   // record the visible point
                return;
            case MatType::Mirror: {
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
            case MatType::Glossy: {
                // NEXT-EVENT ESTIMATION AT A GLOSSY VERTEX. Taken before `thr *= r`: the
                // connection carries `r` inside bsdfF. See backward.h's twin for why this is
                // MIS rather than the single-estimator split used elsewhere on this walk.
                if (gneeOn) {
                    const BackwardRenderer::NeeBsdf nb{&m, ray.d * -1.0};
                    directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                             * (thr * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng, nullptr,
                                                     BackwardRenderer::GiCtx{}, nullptr, nullptr,
                                                     &nb));
                }
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0.0) return;
                if (gneeOn) {
                    gmis.pdf = bdpt::bsdfPdf(m, h.n, ray.d * -1.0, o, lambda, scene, &h);
                    gmis.from = h.p;
                    gmis.n = h.n;
                }
                ray = Ray{h.p + h.n * 1e-6, o};
                break;
            }
            case MatType::Dielectric: {
                // Nested-dielectric PRIORITY resolution (Schmidt & Budge 2002): exterior
                // IOR = the medium the photon is currently inside (highest-priority stack
                // entry). Overlapping dielectrics ranked by `priority` (higher wins; lower
                // suppressed -> straight pass-through). SAFE FALLBACK to flat air<->glass
                // unless BOTH sides carry an explicit priority (priority-free scenes stay
                // bit-identical).
                bool entering = dot(ray.d, h.ng) < 0.0;
                const int mi = (int)(&m - scene.mats.data());   // true index (Mix/Layered aware)
                const int pr = m.priority;
                if (entering) {
                    const int outMat = stk.topMat();
                    const int outPri = stk.topPri();
                    const bool ranked = m.hasPriority() &&
                        (stk.empty() || (outMat >= 0 && scene.mats[outMat].hasPriority()));
                    if (ranked && !stk.empty() && pr <= outPri) {   // suppressed inner surface
                        stk.push(mi, pr);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    } else {
                        const double extIor = (ranked && outMat >= 0)
                            ? scene.mats[outMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                        if (transmitted) stk.push(mi, pr);
                    }
                } else {
                    MediumStack after = stk; after.popMat(mi);
                    const int newMat = after.topMat();
                    const int newPri = after.topPri();
                    const bool ranked = m.hasPriority() &&
                        (after.empty() || (newMat >= 0 && scene.mats[newMat].hasPriority()));
                    if (ranked && newMat >= 0 && pr <= newPri) {    // suppressed: still enclosed
                        stk.popMat(mi);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    } else {
                        const double extIor = (ranked && newMat >= 0)
                            ? scene.mats[newMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                        if (transmitted) stk.popMat(mi);            // TIR stays inside mi
                    }
                }
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() < r) ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                else                   ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Filter: {
                thr *= clamp01(transmitSlot(scene, m, h, lambda));
                ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Hair: {
                // Fiber BCSDF (src/hair.h via hair_shade.h). A strand is deliberately NOT
                // recorded as a visible point: the photon payload (`struct Photon` in
                // photonmap.h) stores only position/normal/power/lambda and carries NO
                // incident direction, so the density estimate cannot evaluate a
                // DIRECTIONAL BCSDF at the gather — there is nothing to plug in for wi.
                // So hair is treated the way these modes already treat glossy and
                // specular surfaces: the camera walk SCATTERS through it correctly and
                // records its visible point on whatever diffuse surface lies beyond.
                // (Logged in known-issues.md; modes R/D/V shade strands fully.)
                const Vec3 wPrev{-ray.d.x, -ray.d.y, -ray.d.z};
                const HairShade hs = hairShadeAt(scene, m, h, lambda, wPrev);
                double pdfH = 0.0, fv = 0.0;
                const Vec3 wl = hair::sample(hs.b, hs.woLocal, rng.uniform(), rng.uniform(),
                                             rng.uniform(), rng.uniform(), pdfH, fv);
                if (!(pdfH > 0.0) || !(fv > 0.0)) return;
                // Exactly T = sum_p A_p, the total lobe attenuation (see render.h's Hair
                // case for why the ratio collapses to a deterministic number).
                const double cosLong =
                    hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
                thr *= clamp01(fv * cosLong / pdfH);
                const Vec3 wo = hair::toWorld(hs.fr, wl);
                // TT/TRT leave through the FAR side of a real solid tube.
                ray = Ray{h.p + wo * hairExitOffset(hs, h.n, wo), wo};
                break;
            }
            default: {
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
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
                     double alpha, int maxBounce, uint64_t passSeed,
                     int heroC = hero::kHeroC) {
    if (nThreads < 1) nThreads = 1;
    // `-seed`: this pass's streams all descend from `passSeed`, so salting it once here is
    // the whole of the flag for mode `S` (0 by default, so XOR is the identity).
    passSeed ^= g_rngSalt;
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
    // seedBase = cumulative photons emitted before this pass, so every pass traces a
    // FRESH, independent photon set (SPPM's convergence requires it) while staying
    // deterministic for a fixed pass sequence.
    tracePhotonPass(scene, photonsPerPass, nThreads, diffraction, pm, heroC,
                    (uint64_t)st.emittedTotal);
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
                pm.queryR(h.p, P.radius, [&](const Photon& ph, double, int k) {
                    if (dot(ph.n, h.n) < 0.5) return;   // reject cross-surface leakage
                    double rho = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                    double f = rho * (1.0 / PI);
                    phi += pm.cie[k] * (f * (double)ph.power);        // == cie(lambda_p), precomputed
                    M += 1.0;
                });
                // M-GATHERAREA, mode `S`'s twin of the mode-`M` correction. The query above
                // rejects photons whose normal disagrees with the hit's, and nothing clips the
                // disc to the surface, yet `sppmResolve` divides by the area of a FULL disc --
                // so a gather on thin or truncated geometry is normalised by an area it never
                // collected from. Measured on `scenes/_ga_strip.ftsl` (a 0.4 m strip under a
                // 0.5 m pinned radius) mode `S` read -21.9 % against a mode-`D` anchor, where
                // uncorrected mode `M` read -52.5 % and corrected mode `M` read +8.3 %.
                //
                // APPLIED HERE, AT ACCUMULATION, AND NOT AT RESOLVE. `sppmResolve` divides the
                // accumulated `tau` by `pi R^2` at the FINAL radius, which is correct only
                // because every pass's contribution has been rescaled by the `ratio2` chain --
                // the product of later ratios is exactly `R_final^2 / R_i^2`, so the sum
                // telescopes into `sum_i phi_i / (pi R_i^2)`. Coverage is a property of the
                // radius that was actually gathered at, and SPPM's radius shrinks every pass,
                // so a single coverage measured at `R_final` would misprice every earlier pass.
                // Scaling `phi` before it enters `tau` puts each pass's flux over its own
                // footprint, which is the quantity the telescoping sum then carries.
                //
                // The RNG is seeded per PIXEL and per PASS rather than per thread, so the
                // probe pattern -- and hence the image -- does not depend on `-t`.
                if (const int gaM = gatherAreaSamples()) {
                    Pcg32 grng;
                    grng.seed(((uint64_t)y << 20) ^ (uint64_t)x,
                              0x9e3779b97f4a7c15ULL ^ (uint64_t)st.passes);
                    phi = phi * gatherAreaScale(
                        gatherCoverage(scene, h.p, h.n, P.radius, grng, gaM, h.matId,
                                       h.fiberRadius));
                }
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
