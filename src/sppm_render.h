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
    // SPPM-SPECTRAL: whether the walk folded a factor that is not flat, and the gather then reads
    // the pixel's ratios (SPPMState::vpRatio).
    bool   vpSpec = false;
};

struct SPPMState {
    int resX = 0, resY = 0;
    std::vector<SPPMPixel> px;
    // SPPM-SPECTRAL: the walk's spectral factors as the gather needs them, `cells` ratios a pixel, one
    // per cell of the scene's carrier (SpecThr::ratio at the cell, over its peak, which vpThr carries:
    // a ratio can pass a float), so a photon of wavelength L is weighted by the camera prefix's
    // factors at L rather than at the walk's own wavelength. A pixel's are read only where vpSpec.
    int cells = kBandProbeBins;
    std::vector<float> vpRatio;
    float* ratios(size_t pix) { return vpRatio.data() + pix * (size_t)cells; }
    long long emittedTotal = 0;   // photons emitted across ALL passes (normalization)
    long long passes = 0;
    // `nCells`: the scene's carrier cells (Scene::specCells.n).
    void init(int w, int h, double R0, int nCells) {
        resX = w; resY = h;
        cells = nCells;
        vpRatio.assign((size_t)w * h * (size_t)nCells, 0.0f);
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
// reaching a diffuse surface. `sthr` (constructed over scene.emitSampler, the walk's wavelength
// density) takes every spectral factor folded into `thr`, as mode M's gather walk folds them
// (SPECGATHER): the photons the visible point gathers come in every wavelength, and each owes the
// camera prefix's factors at its own (SPPM-SPECTRAL).
inline void sppmVisiblePoint(const Scene& scene, Ray ray, Pcg32& rng, bool diffraction,
                             int maxBounce, Hit& vp, double& thrOut, bool& vpValid,
                             Vec3& directL, SpecThr& sthr, const MediumStack& stk0) {
    directL = Vec3{0, 0, 0};
    thrOut = 0.0; vpValid = false;

    double thr = 1.0, pdfL = 0.0;
    double lambda = scene.emitSampler.sample(rng, pdfL);
    if (pdfL <= 0.0) return;
    const double invPdfL = scene.invPdfLambda(lambda);

    Renderer mats; mats.diffraction = diffraction;
    // nested-dielectric medium stack, from the camera's (`stk0`, Camera::enclosure)
    MediumStack stk = stk0;
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
    // CURVE-GATHER: the previous vertex was a diffuse strand, which took its direct light by NEE.
    bool strandArrival = false;
    // The walk has gone on past a fibre's or a diffuse strand's scattering: a glow no emitter
    // samples is seen only before (BackwardRenderer::GiCtx::pastDiffuse).
    bool pastDiffuse = false;
    double contSoft = 1.0;   // the softening the last vertex lends what it reaches (contSoftening)

    // A coverage pass-through is no vertex: its bounce is refunded (capped: 256 is far above
    // anything real fur hands back), as backward.h's walks refund it.
    int nullSteps = 0;
    // The tail (b == maxBounce): the ray the last vertex sampled, for the light it reaches and
    // nothing past it -- no visible point either (backward.h radianceLoop, GLOSSY-DEPTH-CAP).
    for (int b = 0; b <= maxBounce; ++b) {
        const bool tail = (b == maxBounce);
        if (grinAny) {
            double arc = 0.0;
            grin::MarchInfo mInfo;
            grin::marchSegments(scene, ray,
                [&](const Vec3&, const Vec3&, double slen, double&) { arc += slen; return false; },
                // b == 0 is the camera ray; see Material::hideCamera.
                /*camHide=*/(b == 0), &mInfo);
            // Radiance over n^2 along the bend (flat in wavelength: the spectral carrier is
            // unchanged); past a bend the light the walk meets was reached by no light sample (grin.h
            // MarchInfo).
            thr *= mInfo.radScale;
            if (mInfo.bent) { strandArrival = false; gmis.clear(); }
            const int cm = stk.topMat();                 // Beer-Lambert over the marched arc
            if (cm >= 0 && arc > 0.0) {
                const double T = std::exp(-scene.mats[cm].absorb(lambda) * arc);
                thr *= T;
                sthr.mul([&](double L) { return std::exp(-scene.mats[cm].absorb(L) * arc); }, T);
            }
        }
        // b == 0 is the camera ray this function was handed; see Material::hideCamera.
        Hit h = closestHitIn(scene, stk, ray, Scene::kTminEps, nullptr, /*skipHair=*/false,
                                         /*skipCamHidden=*/(b == 0));
        if (h.valid) {
            // Inside a dielectric whatever it absorbs at the walk's own wavelength: a glass clear
            // there may absorb at a photon's (clear glass folds exactly 1).
            const int cm = h.extMat;
            if (cm >= 0) {
                const double T = std::exp(-scene.mats[cm].absorb(lambda) * h.t), tt = h.t;
                thr *= T;
                sthr.mul([&](double L) { return std::exp(-scene.mats[cm].absorb(L) * tt); }, T);
            }
        }
        if (!h.valid) {
            // camera / specular escape: the env as SEEN, bilinear (Scene::envRadianceSeen)
            if (scene.envIndex >= 0)
                directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                           * (thr * scene.envRadianceSeen(ray.d, lambda) * invPdfL * contSoft);
            // Directly-viewed solar disc (camera / specular escapes only — a diffuse
            // vertex stores a hit point and returns before it can reach here).
            if (scene.sunCount > 0 && !strandArrival)
                directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                           * (thr * bwNee.sunRadianceMis(scene, gmis, ray.d, lambda) * invPdfL * contSoft);
            return;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            // The coat with probability R (its Fresnel reflectance here), else a body lobe, as
            // mode M's walk and the device's dResolveCompound resolve it: the coat is a glossy
            // material of its own (Material::coatChild), shaded by the switch below.
            const double R = layeredCoatReflectance(scene, *mp, h, ray.d, lambda);
            const bool coatPick = rng.uniform() < R;
            int c = coatPick ? mp->coatChild
                             : mixResolveChild(scene, *mp, h, rng.uniform());  // body lobe: honours a bound weight map
            if (c < 0) return;
            foldCoatCoin(sthr, scene, *mp, h, ray.d, coatPick, R);   // SPECGATHER
            mp = &scene.mats[c];
        }
        const Material& m = *mp;

        if (m.isLight) {
            // A light emits from the face its triangles are wound to (the side its `normal`
            // picks, as backward.h tests it); not after a strand's NEE took it.
            const int eIdx = BackwardRenderer::emitterIndexOfResolved(scene, h, m);
            if (dot(ray.d, h.ng) < 0.0 && !strandArrival && (eIdx >= 0 || !pastDiffuse)) {
                // GLOSSY-NEE's lobe-sampling half; 1, and this expression bit-identical to its
                // pre-0.266 form, unless the previous bounce was a MIS'd glossy one.
                const double wMis = (gmis.pdf > 0.0) ? bwNee.glossyHitWeight(scene, gmis, eIdx, ray.d, &h)
                                                     : 1.0;
                directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                           * (thr * emitSlot(scene, m, h, lambda, ray.d * -1.0) * invPdfL * wMis * contSoft);
            }
            // A light block's own surface ends the path; any other emissive material also has
            // a body, which is shaded below like any surface (mode M's gather does the same).
            if (m.reflectsNothing) return;
        }
        if (tail) {                                      // the light it reached, and no more --
            // past a strand it passes, which is no vertex (BackwardRenderer::tailPassesStrand)
            if (m.type == MatType::Hair && nullSteps < 256 &&
                bwNee.tailPassesStrand(scene, m, h, lambda, ray, rng)) { --b; ++nullSteps; continue; }
            return;
        }

        // GLOSSY-NEE: cleared HERE, after the two sites that read it and before the branch that
        // writes it -- a clear at the loop top would erase the previous bounce's value a few
        // lines before its only consumer, leaving the connection with no compensating weight.
        // See the twin note in backward.h.
        // ...each put back after a coverage pass-through, which is no vertex (the Hair case).
        const BackwardRenderer::GlossyMis gmisIn = gmis;
        const bool strandArrivalIn = strandArrival;
        const double contSoftIn = contSoft;
        gmis.clear();
        strandArrival = false;
        contSoft = 1.0;   // a specular vertex lends nothing; the glossy case sets its own

        // CURVE-GATHER: no visible point on a `type diffuse` strand (strandStep, where the why
        // is): lit by NEE, the walk goes on along its diffuse lobe to the surface that gathers.
        if (gatherlessStrand(m, h.fiberRadius)) {
            bool cont;
            directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                       * strandStep(scene, bwNee, m, h, lambda, invPdfL, rng, thr, &sthr, ray, cont);
            strandArrival = true;
            pastDiffuse = true;
            if (!cont) return;
            continue;
        }

        const Vec3 wArr = ray.d * -1.0;   // the arrival, for the coat (coatWalkOut)
        switch (m.type) {
            case MatType::Diffuse:
            case MatType::DiffuseTransmit:
            case MatType::Fluorescent:
                // Record the visible point under the lobe the walk resolved -- a layered body's
                // base, a mix's child -- whose reflectance the gather folds per photon, as mode M's
                // gather does: the compound's own slot is not that surface's colour (a red car
                // paint's green and blue read 7x mode R gathered through it).
                vp = h; vp.matId = (int)(mp - scene.mats.data());
                thrOut = thr; vpValid = true;
                return;
            case MatType::Mirror: {
                const double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
                sthr.mul([&](double L) { return clamp01(reflectSlot(scene, m, h, L)); }, r);
                if (!continueAlong(h, reflect(ray.d, h.n), Lobe::Reflect, ray, scene.hitEps(h))) return;   // across the surface
                break;
            }
            case MatType::Glossy: {
                if (!glossyCameraSide(ray.d, h.n)) return;     // met from below ns (glossyCameraSide)
                // NEXT-EVENT ESTIMATION AT A GLOSSY VERTEX. Taken before `thr *= r`: the
                // connection carries `r` inside bsdfF. See backward.h's twin for why this is
                // MIS rather than the single-estimator split used elsewhere on this walk.
                if (gneeOn) {
                    const BackwardRenderer::NeeBsdf nb{&m, ray.d * -1.0};
                    directL += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                             * (thr * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng, nullptr,
                                                     coatGi(m, ray.d * -1.0), nullptr, nullptr,
                                                     &nb));
                }
                const double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
                sthr.mul([&](double L) { return clamp01(reflectSlot(scene, m, h, L)); }, r);
                const GlossyDraw gd = drawGlossy(scene, m, h, h.n, ray.d * -1.0, rng);
                const Vec3& o = gd.o;
                if (dot(o, h.n) <= 0.0) return;
                thr *= gd.w;   // the draw's weight: the same at every wavelength, so not sthr's
                // An iridescent lobe's interference reflectance at the pair's half-vector. Where it
                // is 0 at the walk's own wavelength the path carries nothing (and the ratio would
                // divide by it).
                if (gd.irid) {
                    const double Rh = gd.R(lambda);
                    if (!(Rh > 0.0)) return;
                    thr *= Rh;
                    sthr.mul([&](double L) { return gd.R(L); }, Rh);
                }
                if (gneeOn) {
                    gmis.pdf = bdpt::bsdfPdf(m, h.n, ray.d * -1.0, o, lambda, scene, &h);
                    gmis.from = h.p;
                    gmis.n = h.n;
                }
                if (!continueAlong(h, o, Lobe::Reflect, ray, scene.hitEps(h))) return;
                contSoft = contSoftening(m, h, ray.d);
                break;
            }
            case MatType::Dielectric:
            case MatType::ThinFilm:
            case MatType::Multilayer: {
                // A dielectric body's surface, plain or coated (Renderer::interfaceStep); this
                // camera-side walk carries radiance across a refraction.
                // The step at each photon's wavelength (stepFactorAt) against its value here
                // (st.factor): the branch the coin took and the radiance factor, as mode M's walk.
                const Vec3 d0 = ray.d;
                const Renderer::InterfaceStep st =
                    mats.interfaceStep(scene, m, h, d0, lambda, rng, stk,
                                       Renderer::Transport::Radiance);
                if (st.absorbed) return;
                ray = st.ray;
                thr *= st.scale;
                sthr.mulCoin([&](double L, double& pr) {
                    return mats.stepFactorAt(scene, m, st, h, d0, L, &pr);
                }, st.factor, st.prob, m.type != MatType::Dielectric);
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                const bool refl = rng.uniform() < r;
                if (refl) { if (!continueAlong(h, reflect(ray.d, h.n), Lobe::Reflect, ray, scene.hitEps(h))) return; }
                else      ray = passThrough(h, ray.d, scene.hitEps(h));
                // The coin at each photon's wavelength: a coloured half mirror reflects each
                // wavelength by its own reflectance.
                const double pc = refl ? r : 1.0 - r;
                sthr.mulCoin([&](double L, double& pr) {
                    const double rL = clamp01(reflectSlot(scene, m, h, L));
                    return pr = refl ? rL : 1.0 - rL;
                }, pc, pc, true);
                break;
            }
            case MatType::Filter: {
                const double t = clamp01(transmitSlot(scene, m, h, lambda));
                thr *= t;
                sthr.mul([&](double L) { return clamp01(transmitSlot(scene, m, h, L)); }, t);
                ray = passThrough(h, ray.d, scene.hitEps(h));
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
                hair::LobeAngular la;                              // filled by the sample's own f()
                const Vec3 wl = hair::sampleWith(hs.b, hs.woLocal, rng, pdfH, fv, &la);
                if (!(pdfH > 0.0) || !(fv > 0.0)) return;
                // Exactly T = sum_p A_p, the total lobe attenuation (see render.h's Hair
                // case for why the ratio collapses to a deterministic number).
                const double cosLong =
                    hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
                const double wCam = clamp01(fv * cosLong / pdfH);
                thr *= wCam;
                // The fiber's transmission at each photon's wavelength, as mode M's walk folds it:
                // the sampled direction and pdf stay, the BCSDF value moves with the absorption. A
                // coverage pass-through (exactly -wo) is achromatic and folds nothing.
                const bool passedThru = (hs.b.opacity < 1.0) && wl.x == -hs.woLocal.x &&
                                        wl.y == -hs.woLocal.y && wl.z == -hs.woLocal.z;
                if (passedThru) {   // no vertex: the last real one's record stands, its bounce refunded
                    gmis = gmisIn; strandArrival = strandArrivalIn; contSoft = contSoftIn;
                    if (nullSteps < 256) { --b; ++nullSteps; }
                }
                if (!passedThru) {
                    pastDiffuse = true;   // a fibre's scattering: no glow is seen past it
                    const double* bins = hairSigmaCells(scene, *sthr.cells, m, h);
                    sthr.mul([&](double lamK) {
                        const HairShade hk = hairShadeRetint(scene, m, h, lamK, hs,
                                                             bins ? bins[sthr.binOf(lamK)] : -1.0);
                        return clamp01(hairFoldFCos(la, hk, wl, cosLong, nullptr) / pdfH);
                    }, wCam);
                }
                const Vec3 wo = hair::toWorld(hs.fr, wl);
                // TT/TRT leave through the FAR side of a real solid tube.
                ray = Ray{h.p + wo * scene.hitEps(h), wo, hairExitOffset(hs, h.n, wo, scene.hitEps(h))};
                break;
            }
            default: {
                const double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
                sthr.mul([&](double L) { return clamp01(reflectSlot(scene, m, h, L)); }, r);
                if (!continueAlong(h, reflect(ray.d, h.n), Lobe::Reflect, ray, scene.hitEps(h))) return;   // across the surface
                break;
            }
        }
        // A coated vertex's continuation takes the coat, at each photon's wavelength too
        // (photonmap_render.h coatWalkOut, as mode M's walk). A fibre has no coat, so its
        // pass-through never reaches this.
        if (!coatWalkOut(scene, m, h, wArr, ray.d, lambda, thr, &sthr)) return;
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
    // Motion blur (ROADMAP 11). A moving scene's pass -- visible points, photons and gather -- is
    // at one time, stratified over the passes (passTime), the camera posed there: the passes'
    // average integrates the shutter, as SPPM averages every other dimension. A moving camera
    // over a still scene leaves the photons timeless and poses each pixel's ray at its own time.
    const bool passTimed = cam.timed() && scene.hasMotion();
    const double tPass = passTimed ? cam.shutterTime(passTime((uint64_t)st.passes, 0x2545F4914F6CDD1DULL)) : 0.0;
    const PassTimeScope passAt(passTimed ? scene.motionU(tPass) : -1.0);
    Camera camAt = cam;
    if (passTimed && cam.moves()) cam.poseAt(tPass, camAt);
    const bool perPixel = !passTimed && cam.timed() && cam.moves();

    // (1) Camera pass — fresh visible point + direct sample per pixel this pass.
    {
        auto camWorker = [&](int tid) {
            const MotionTimeScope at(g_passMotionU);
            PosedCamera posed(camAt, perPixel);
            Pcg32 rng; rng.seed(passSeed * 0x9E3779B97F4A7C15ULL + (uint64_t)tid * 2 + 23,
                                0xA24BAED4963EE407ULL ^ (uint64_t)tid ^ (passSeed << 1));
            int y0 = H * tid / nThreads, y1 = H * (tid + 1) / nThreads;
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x) {
                    SPPMPixel& P = st.px[(size_t)y * W + x];
                    const Camera& c = perPixel ? posed.at(cam.shutterTime(unitTime(
                                          ((uint64_t)st.passes << 32) ^ ((uint64_t)y * W + x), 0x6C62272E07BB0142ULL)))
                                               : camAt;
                    const double pjx = rng.uniform(), pjy = rng.uniform();   // in order
                    Ray ray = c.genRay(x, y, pjx, pjy);
                    Vec3 directL;
                    SpecThr sthr(scene.specCells);
                    sppmVisiblePoint(scene, ray, rng, diffraction, maxBounce,
                                     P.vpHit, P.vpThr, P.vpValid, directL, sthr, c.enclosure);
                    P.vpSpec = P.vpValid && sthr.any;
                    if (P.vpSpec) {   // in float over its peak, the peak in vpThr (SpecThr::peak)
                        const double pk = sthr.peak();
                        if (st.cells != sthr.n)   // init() was handed another scene's cells
                            throw std::logic_error("sppm: the pixels' ratio stride is not the carrier's cells");
                        float* vr = st.ratios((size_t)y * W + x);
                        for (int k = 0; k < sthr.n; ++k) vr[k] = (float)(sthr.ratioAt(k) / pk);
                        P.vpThr *= pk;
                    }
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
        const MotionTimeScope at(g_passMotionU);   // the visible points' patterns, at their time
        int y0 = H * tid / nThreads, y1 = H * (tid + 1) / nThreads;
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x) {
                SPPMPixel& P = st.px[(size_t)y * W + x];
                if (!P.vpValid) continue;
                const Hit& h = P.vpHit;
                const Material& m = scene.mats[h.matId];
                const Vec3 gn = orientedGeoN(h);   // the true normal, as the photons carry it
                const float* vr = P.vpSpec ? st.ratios((size_t)y * W + x) : nullptr;
                double M = 0.0;          // photons found this pass
                Vec3   phi{0, 0, 0};     // local flux sum (XYZ, per-photon wavelength)
                const double slack = gatherBandSlack(P.radius, h.fiberRadius);
                pm.queryR(h.p, P.radius, [&](const Photon& ph, double d2, int k) {
                    bool farSide;
                    if (!onGatherSurfaceSided(h.p, gn, pm.pos[k], ph.n, d2, slack,
                                              m.type == MatType::DiffuseTransmit, farSide)) return;   // another surface's
                    double rho = gatherAlbedo(scene, m, h, ph.lambda, farSide);   // a translucent's far side: rho_T
                    double f = rho * (1.0 / PI);
                    // the camera prefix's factors at this photon's wavelength (SPPM-SPECTRAL)
                    if (vr) f *= (double)vr[scene.specCells.cellOf(ph.lambda)];
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
                        gatherCoverage(scene, h.p, gn, P.radius, slack, grng, gaM, h.matId,
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
