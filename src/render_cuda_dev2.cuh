// THE DEVICE CODE SHARED BY render_cuda.cu AND render_cuda_jsurf.cu, between kBdptT and
// kJSurfLightT: see render_cuda_dev.cuh, whose constraints it keeps, for why it is a header.
#pragma once
#include "render_cuda_dev.cuh"

namespace gpu {

// ----------------------- final-gather sub-ray (mode M, -pmfg) ----------------
// Device twin of photonGatherSub (photonmap_render.h). One INDIRECT gather sub-ray shot from
// a diffuse visible point (visHit/visMat): it follows specular surfaces (monochromatic at
// `lambda`) exactly like dPhotonGather and terminates at
//   * the first diffuse/translucent hit y -> a radius density query at y, each photon
//     reflected off BOTH y (its material) AND the visible point (visMat) at the photon's
//     wavelength (spectral two-bounce colour bleed). pX/pY/pZ already fold norm/pi, so the
//     query needs no extra norm — just rho(y)*rho(vis) per photon;
//   * a finite EMITTER reached AFTER a specular bounce -> a monochromatic (camera-lambda)
//     sample reflected off the visible point (a straight hemisphere ray onto a light returns
//     0: that direct term is supplied by NEE at the visible point, so counting it here would
//     double-count — the specular-arrival gate mirrors backward.h);
//   * the ENVIRONMENT on ANY escape -> a monochromatic sample reflected off the visible point.
// Returns the XYZ radiance leaving the visible point toward the sub-ray origin for this one
// sampled direction (the caller averages K). Because the sub-ray is cosine-weighted and the
// visible BRDF is Lambertian, the cosine and 1/pi cancel to rho(vis), folded per photon
// (diffuse hit) or applied once (specular-arrival emitter/env). Keep in sync with photonGatherSub.
//
// `pmC` is the CAUSTIC map (the L.S+.D partition of the same deposit); its photons are a
// disjoint set from `pm`'s and it carries its own, much smaller, radius — so the two density
// estimates are simply summed. `pmC.photons == nullptr` when the split is off or the scene
// produced no caustic path, and then this costs one predictable branch per gather.
struct DSpecThr;
__device__ static double dStrandStep(const DScene& sc, const DMaterial& m, const DHit& h, Real lambda,
                                     double invPdfL, DRng& rng, double& thr, DSpecThr* sthr,
                                     DVec3& ro, DVec3& rd, bool& cont);

__device__ static void dPhotonGatherSub(const DScene& sc, const DPhotonMap& pm,
                                        const DPhotonMap& pmC, int diffraction,
                                        DVec3 ro, DVec3 rd, Real lambda, double invPdfL,
                                        const DHit& visHit, const DMaterial& visMat, const DVec3& visWa,
                                        DRng& rng, double& oX, double& oY, double& oZ,
                                        const DMediumStack& stk0) {
    oX = oY = oZ = 0.0;
    double thr = 1.0;
    bool specularSeen = false;                           // any specular bounce so far?
    bool hairArrival = false;                            // HAIR-NEE (see dPhotonGather)
    double contSoft = dContSoftening(visMat, visHit, rd); // first the visible point's (dContSoftening)
    // A coated visible point's lobe takes the coat on the gather ray's way in, at each wavelength a
    // term reads the visible point's reflectance at: it rides in rhoV (host twin photonGatherSub).
    // Prepared once (DCoatEval), so a photon pays only its wavelength's Fresnel or film term.
    const DCoatEval visCe = dCoatAny(visMat) ? dCoatEval(sc, visMat, visHit, visWa, rd) : DCoatEval{};
    const auto sVis = [&](Real Lw) { return visCe.at(Lw); };
    DMediumStack stk = stk0;   // the visible point's (host twin photonGatherSub)
    const Real r2 = (Real)((double)pm.radius * (double)pm.radius);
    const bool causOn = (pmC.photons != nullptr);
    const Real r2C = (Real)((double)pmC.radius * (double)pmC.radius);
    const int maxBounce = 32;
    Real curveTmin = (Real)0;   // a fiber's exit clearance for the next hop, curve prims only
    int nullSteps = 0;          // coverage pass-throughs refunded (host twin)
    for (int b = 0; b <= maxBounce; ++b) {                     // the tail (host twin)
        const bool tail = (b == maxBounce);
        if (sc.hasGrin) {                                // final-gather rays bend too
            DGrinMedia gm;                               // rng == nullptr: bend only
            gm.camera = true;
            dGrinMarch(sc, ro, rd, &gm);
            int cm = stk.topMat();                       // Beer-Lambert over the marched arc
            Real a = (cm >= 0) ? specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
            if (a > 0 && gm.arc > 0) thr *= exp(-(double)a * (double)gm.arc);
            // Radiance over n^2 along the bend; past a bend the light the walk meets was reached
            // by no light sample (host twin photonGatherSub).
            thr *= gm.radScale;
            if (gm.bent) { specularSeen = true; hairArrival = false; }
        }
        DHit h = closestHitIn(stk, sc, ro, rd, curveTmin > (Real)0 ? (Real)0 : RAY_TMIN, BIG, false,
                                   curveTmin);
        curveTmin = (Real)0;
        if (h.valid) {                                   // Beer-Lambert in current medium
            int cm = stk.topMat();
            Real a = (cm >= 0) ? specLookup(sc.mats[cm].absorb, lambda) : (Real)0;
            if (a > 0) thr *= exp(-(double)a * (double)h.t);
        }
        if (!h.valid) {                                  // escaped -> environment (direct off vis)
            if (sc.envIndex >= 0) {
                double envRad = (sc.env.scale != nullptr)
                                    ? dEnvRadiance(sc.env, rd, lambda)
                                    : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lambda);
                double rhoV = ((double)clamp01(dDiffuseRho(sc, visMat, visHit, lambda)) * sVis(lambda));
                double e = thr * rhoV * envRad * invPdfL * contSoft;
                oX += (double)cieX(lambda) * e; oY += (double)cieY(lambda) * e; oZ += (double)cieZ(lambda) * e;
            }
            return;
        }
        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (dIsCompound(mp->type)) {
            int child = dResolveCompound(sc, *mp, h, rd, lambda, rng);
            if (child < 0) return;
            mp = &sc.mats[child]; matId = child;
        }
        const DMaterial& m = *mp;

        // Self-emission on a SPECULAR arrival (a diffuse arrival's direct term comes from NEE
        // at the visible point), the registered emitter's baked SPD. A glow no emitter samples
        // lights nothing, so what this ray -- one gathering light for the visible point -- sees of
        // one is not counted (DGiCtx::pastDiffuse). Does NOT return: an emissive material still
        // has a BSDF. Host twin: photonGatherSub.
        {
            const int li = dEmitterForHit(sc, h, matId);
            const DSpec* eSpd = (li >= 0) ? &sc.emitters[li].emitSpd : nullptr;
            if (eSpd && specularSeen && dot(rd, h.ng) < 0 && !hairArrival) {
                double rhoV = ((double)clamp01(dDiffuseRho(sc, visMat, visHit, lambda)) * sVis(lambda));
                double e = (double)specLookup(*eSpd, lambda) * thr * rhoV * invPdfL * contSoft
                         * dEmitHitScale(sc, m.emitPat, li, h, rd * (Real)(-1));   // pattern and aim at this hit
                oX += (double)cieX(lambda) * e; oY += (double)cieY(lambda) * e; oZ += (double)cieZ(lambda) * e;
            }
        }
        if (tail) {                                     // the light it reached, and no more --
            // past a strand it passes, which is no vertex (host twin; bkTailPassesStrand)
            if (m.type == D_HAIR && nullSteps < 256 &&
                bkTailPassesStrand(sc, m, h, lambda, ro, rd, rng, &curveTmin)) { --b; ++nullSteps; continue; }
            return;
        }

        const bool hairArrivalIn = hairArrival, specularSeenIn = specularSeen;   // a hair pass-through puts these back
        const double contSoftIn = contSoft;
        contSoft = 1.0;   // a specular vertex lends nothing; the glossy case sets its own
        bool nullHere = false;
        hairArrival = false;
        // CURVE-GATHER: no density estimate on a strand that holds no photon (host twin).
        if (dGatherlessStrand(m, h)) {
            const double rhoV = ((double)clamp01(dDiffuseRho(sc, visMat, visHit, lambda)) * sVis(lambda));
            bool cont;
            const double e = dStrandStep(sc, m, h, lambda, invPdfL, rng, thr, nullptr, ro, rd, cont) * rhoV;
            oX += (double)cieX(lambda) * e; oY += (double)cieY(lambda) * e; oZ += (double)cieZ(lambda) * e;
            hairArrival = true;
            if (!cont) return;
            continue;
        }
        if (m.type == D_DIFFUSE || m.type == D_DIFFUSETRANSMIT || m.type == D_FLUORESCENT) {
            const DVec3 gn = dShadedNg(h);   // the true normal, as the photons carry it
            // Density estimate at y, folding the visible-point reflectance per photon wavelength.
            float gx = 0.f, gy = 0.f, gz = 0.f;
            const Real slkM = gatherBandSlack<Real>(pm.radius, h.fiberRadius, dRoundScale(h));
            dPmNeighborhood(pm, h.p, [&](int k) {
                const DGatherPhoton& ph = pm.photons[k];
                DVec3 d = h.p - ph.pos;
                const Real d2 = dot(d, d);
                if (d2 > r2) return;
                bool farSide;
                if (!dOnGatherSurfaceSided(h.p, gn, ph.pos, ph.n, d2, slkM, m.type == D_DIFFUSETRANSMIT, farSide)) return;
                float rhoY = (float)dGatherAlbedo(sc, m, h, (Real)ph.lambda, farSide);
                float rhoV = (float)((double)dDiffuseRho(sc, visMat, visHit, (Real)ph.lambda) * sVis((Real)ph.lambda));
                float w = rhoY * rhoV;
                gx += w * ph.pX; gy += w * ph.pY; gz += w * ph.pZ;
            });
            // M-GATHERAREA, the MAIN map's coverage, on the main map's sum alone -- before the
            // caustic map's is added, which takes its own coverage below (host twin: each `est`
            // scales its own map). The device folds norm/pi into every photon record at upload
            // (DGatherPhoton), so the correction multiplies the sum rather than a normalisation.
            if (sc.gatherArea) {
                const double ms = dGatherAreaScale(sc.gaBias,
                    dGatherCoverage(sc, h.p, gn, (Real)sqrt((double)r2), slkM, rng, sc.gatherArea, h.pscale,
                                    h.fiberRadius));
                gx = (float)((double)gx * ms);
                gy = (float)((double)gy * ms);
                gz = (float)((double)gz * ms);
            }
            if (causOn) {
                // The caustic map gathers at its own PER-QUERY radius (dPmAdaptiveRadius).
                // pX/pY/pZ carry the fold for the map's fixed radius, so rescale the sum by
                // the area ratio r^2/r_q^2 rather than re-folding every record.
                const Real slkC = gatherBandSlack<Real>(pmC.radius, h.fiberRadius, dRoundScale(h));
                const Real rq  = dPmAdaptiveRadius(pmC, h.p, gn, h.fiberRadius, m.type == D_DIFFUSETRANSMIT, dRoundScale(h));
                const Real r2q = rq * rq;
                float cx = 0.f, cy = 0.f, cz = 0.f;
                dPmNeighborhood(pmC, h.p, [&](int k) {
                    const DGatherPhoton& ph = pmC.photons[k];
                    DVec3 d = h.p - ph.pos;
                    const Real d2 = dot(d, d);
                    if (d2 > r2q) return;
                    bool farSide;
                    if (!dOnGatherSurfaceSided(h.p, gn, ph.pos, ph.n, d2, slkC, m.type == D_DIFFUSETRANSMIT, farSide)) return;
                    float rhoY = (float)dGatherAlbedo(sc, m, h, (Real)ph.lambda, farSide);
                    float rhoV = (float)((double)dDiffuseRho(sc, visMat, visHit, (Real)ph.lambda) * sVis((Real)ph.lambda));
                    float w = rhoY * rhoV;
                    cx += w * ph.pX; cy += w * ph.pY; cz += w * ph.pZ;
                });
                const float aw = (r2q > (Real)0) ? (float)((double)r2C / (double)r2q) : 0.f;
                // M-GATHERAREA: the caustic map has its OWN radius, so its own coverage --
                // matching the host, where the correction lives inside each `est` call keyed
                // on that map's rq. One shared coverage would be cheaper and wrong.
                double cs = 1.0;
                if (sc.gatherArea)
                    cs = dGatherAreaScale(sc.gaBias, dGatherCoverage(sc, h.p, gn, rq, slkC, rng, sc.gatherArea, h.pscale,
                                                          h.fiberRadius));
                gx += cx * (float)(aw * cs);
                gy += cy * (float)(aw * cs);
                gz += cz * (float)(aw * cs);
            }
            oX += (double)gx * thr; oY += (double)gy * thr; oZ += (double)gz * thr;
            return;
        }

        const DVec3 wArr = -rd;                          // the arrival, for the coat (dCoatF)
        switch (m.type) {                                // specular walk (monochromatic)
            case D_MIRROR: {
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                if (!dContinueAlong(h, reflectv(rd, h.n), DL_REFLECT, ro, rd)) return; break;
            }
            case D_GLOSSY: {
                if (!dGlossyCameraSide(rd, h.n)) return;        // met from below ns (host twin)
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                const DGlossyDraw gd = dDrawGlossy(sc, m, h, h.n, -rd, rng);
                const DVec3& o = gd.o;
                if (dot(o, h.n) <= 0) return;
                // The draw's weight, and an iridescent lobe's interference reflectance at the
                // pair's half-vector.
                thr *= (double)gd.w * (double)gd.R(lambda);
                if (!dContinueAlong(h, o, DL_REFLECT, ro, rd)) return;
                contSoft = dContSoftening(m, h, rd); break;
            }
            case D_DIELECTRIC:
            case D_THINFILM:
            case D_MULTILAYER: {                         // a dielectric body, plain or coated
                DVec3 nro, nrd; double rs = 1.0;
                if (!dDielectricStep(sc, m, h, rd, lambda, rng, stk, nro, nrd, nullptr, &rs))
                    return;
                thr *= rs;                               // radiance across the interface
                ro = nro; rd = nrd; break;
            }
            case D_HALFMIRROR: {
                Real r = clamp01(dReflectSlot(sc, m, h, lambda));
                if (rng.uniform() < r) { if (!dContinueAlong(h, reflectv(rd, h.n), DL_REFLECT, ro, rd)) return; }
                else                   { ro = dOffsetAlong(h.p, h.ng, rd, h.pscale); }
                break;
            }
            case D_FILTER: {
                thr *= (double)clamp01(dTransmitSlot(sc, m, h, lambda));
                ro = dOffsetAlong(h.p, h.ng, rd, h.pscale); break;
            }
            case D_HAIR: {
                // HAIR (0.335.0): see dPhotonGather's case. This sub-walk carries no spectral
                // carrier (host twin: photonGatherSub), so the weight stays scalar; every term it
                // reports carries the visible point's albedo, the NEE term included.
                const DVec3 wPrev = rd * (Real)(-1);
                const DHairShade hsv = dHairShadeAt(sc, m, h, lambda, wPrev);
                {   const double rhoV = ((double)clamp01(dDiffuseRho(sc, visMat, visHit, lambda)) * sVis(lambda));
                    const double direct = bkNeeLight(sc, h, (Real)1, invPdfL, lambda, rng, 1, &hsv);
                    const double dc = thr * rhoV * direct;
                    oX += (double)cieX(lambda) * dc; oY += (double)cieY(lambda) * dc; oZ += (double)cieZ(lambda) * dc;
                    hairArrival = true;
                }
                const double u0 = rng.uniform(), u1 = rng.uniform(), u2 = rng.uniform(), u3 = rng.uniform();
                double pdfH = 0.0, fv = 0.0;
                const dhair::V3 wl = dhair::sample(hsv.b, hsv.woLocal, u0, u1, u2, u3, pdfH, fv);
                if (!(pdfH > 0.0) || !(fv > 0.0)) return;
                const double cosLong = dhair::safeSqrt(1.0 - dhair::sqr(dhair::clampd(wl.x, -1.0, 1.0)));
                double wCam = fv * cosLong / pdfH; wCam = wCam < 0.0 ? 0.0 : (wCam > 1.0 ? 1.0 : wCam);
                thr *= wCam;
                // A coverage pass-through (exactly -wo) is no vertex: the last real one's
                // emitter-hit bookkeeping stands (host twin photonGatherSub).
                if (hsv.b.opacity < 1.0 && wl.x == -hsv.woLocal.x && wl.y == -hsv.woLocal.y &&
                    wl.z == -hsv.woLocal.z) {
                    nullHere = true; hairArrival = hairArrivalIn; contSoft = contSoftIn;
                    if (nullSteps < 256) { --b; ++nullSteps; }   // ...and its bounce refunded
                }
                const DVec3 wOut = dhair::toWorld(hsv.fr, wl);
                // Clear the strand by skipping curves within its exit distance on the next
                // trace, from a ULP-moved origin (see interactHair): moving the origin out by
                // that distance carries a ray leaving near a root through the skin below.
                curveTmin = dHairExitOffset(hsv, h.n, wOut, h.p);
                ro = dOffsetPoint(h.p, wOut, h.pscale);
                rd = wOut;
                break;
            }
            default: {                                   // Grating: approx reflect
                thr *= (double)clamp01(dReflectSlot(sc, m, h, lambda));
                if (!dContinueAlong(h, reflectv(rd, h.n), DL_REFLECT, ro, rd)) return; break;
            }
        }
        // A coated vertex's continuation takes the coat (host twin coatWalkOut); a pass-through
        // is no vertex.
        if (!nullHere && dCoatAny(m)) thr *= dCoatF(sc, m, h, wArr, rd, lambda);
        specularSeen = nullHere ? specularSeenIn : true;   // only specular cases reach here; a pass-through inherits
        if (thr <= 0.0) return;
    }
}

// -sunnee (0.312.0): the sun's SINGLE-scatter term along a camera segment, by next-event
// estimation instead of stored beams. Device twin of beamgather.h sunNeeMarch -- same jittered
// steps, same fold, and the same shadow-ray conventions as the volume-vertex sun connection in
// the BDPT NEE above (cone-sampled direction, pdf 1/Omega, transmittance out to the scene
// sphere). The direct-sun chords this replaces were erased from the map by sunNeeSplit, so
// nothing is counted twice; DBeamMap::sunNee carries that decision with the map it was made on.
//
//   L_1 = integral_0^tMax  Tr_cam(t) * sigma_s(p_t) * phase(cos) * Tr_sun(p_t) * L_sun * Omega  dt
//
// kSunNeeSteps JITTERED steps per medium span (one offset per segment: an unbiased estimate,
// not a fixed quadrature), Tr_cam carried incrementally step to step, Tr_sun by ratio tracking
// to the scene exit plus a shadow ray. An unbounded medium is cut at the deposit's own far limit
// so the two estimators integrate one volume.
//
// COLOUR is folded where the beams' would have been: a fully achromatic medium takes the sun's
// CIE integral `viewXYZ`; a bow medium with a table for this (emitter, medium) pair takes
// Bow(cos) * (integral SPD) -- deterministic in colour, a rainbow with no chromatic grain;
// anything else evaluates at the hero wavelength times invPdfL like every other spectral term.
// Transmittances are at the hero wavelength in every branch, exactly as the folded beams' are.
__device__ static void dSunNeeMarch(const DScene& sc, const DVec3& oc, const DVec3& dc, Real tMax,
                                    double aGlassCam, Real lambda, double invPdfL, DRng& rng,
                                    double& oX, double& oY, double& oZ) {
    if (sc.sunCount <= 0 || sc.mediaN <= 0 || !(tMax > (Real)0)) return;
    const DPatEnv env = dPatEnvOf(sc);
    const double cX = (double)cieX(lambda), cY = (double)cieY(lambda), cZ = (double)cieZ(lambda);
    const double farLimit = 8.0 * fmax(sc.sceneRadius, 1e-3);   // dEmitBeams' own far limit
    for (int mi = 0; mi < sc.mediaN; ++mi) {
        const DMedium& md = sc.media[mi];
        if (!md.enabled) continue;
        const double ssL = fmax(0.0, (double)specLookup(md.sigma_s, lambda));
        if (!(ssL > 0.0)) continue;
        double ta = 0.0, tb = 0.0;
        if (!dMedClip(md, oc, dc, 0.0, (double)tMax, ta, tb) || !(tb > ta)) continue;
        if (tb - ta > farLimit) tb = ta + farLimit;
        const int steps = pbeams::kSunNeeSteps;
        const double dt = (tb - ta) / (double)steps;
        const double jit = (double)rng.uniform();
        double Tcam = (double)dMediaTransmittance(sc, oc, dc, (Real)ta, lambda, rng);
        if (aGlassCam > 0.0) Tcam *= exp(-aGlassCam * ta);
        double tPrev = ta;
        for (int i = 0; i < steps && Tcam > 1e-6; ++i) {
            const double t = ta + dt * ((double)i + jit);
            const DVec3 p = oc + dc * (Real)t;
            Tcam *= (double)dMediaTransmittance(sc, oc + dc * (Real)tPrev, dc, (Real)(t - tPrev), lambda, rng);
            if (aGlassCam > 0.0) Tcam *= exp(-aGlassCam * (t - tPrev));
            tPrev = t;
            const double dens = dMedDensityAt(md, p, env);   // 1 when homogeneous; carves the bound
            if (!(dens > 0.0)) continue;
            for (int k = 0; k < sc.nEmitters; ++k) {
                const DEmitter& em = sc.emitters[k];
                if (em.shape != 6 || !(em.spotOmega > 0.0)) continue;
                const double s1 = (double)rng.uniform(), s2 = (double)rng.uniform();
                const DVec3 wi = dSunSampleCone(em, em.beamDir * (Real)(-1), s1, s2);   // toward the sun
                const Real dist = (Real)(dExitDistance(sc, p));
                if (occluded(sc, p + wi * dEpsAt(p), wi, dist)) continue;
                const double Tsun = (double)dMediaTransmittance(sc, p, wi, (Real)dMediaExitDistance(sc, p),
                                                                lambda, rng);
                if (!(Tsun > 0.0)) continue;
                // Light travels along -wi and leaves toward the camera along -dc: the same argument
                // the beam gather hands the phase (dot(b.d, -dc)).
                const double cosT = (double)dot(wi, dc);
                const double w = Tcam * Tsun * ssL * dens * dt * em.spotOmega;
                double fx, fy, fz, ph;
                if (md.achro) {
                    const double phase = (double)dMedPhase(md, (Real)cosT, lambda);
                    oX += w * phase * (double)em.viewXYZ.x;
                    oY += w * phase * (double)em.viewXYZ.y;
                    oZ += w * phase * (double)em.viewXYZ.z;
                } else if (dBowEval(sc.bow, k, mi, cosT, fx, fy, fz, ph)) {
                    // Bow(cos) = cie * phaseLum = integral CIE*SPD*phase / integral SPD, and
                    // integral SPD = viewXYZ.y / cieMean.y (both per-nm sums, scene.h).
                    const double den = ((double)em.cieMean.y > 0.0)
                                           ? (double)em.viewXYZ.y / (double)em.cieMean.y : 0.0;
                    const double sF = w * ph * den;
                    oX += sF * fx; oY += sF * fy; oZ += sF * fz;
                } else {
                    const double phase = (double)dMedPhase(md, (Real)cosT, lambda);
                    const double e = w * phase * (double)specLookup(em.emitSpd, lambda) * invPdfL;
                    oX += cX * e; oY += cY * e; oZ += cZ * e;
                }
            }
        }
    }
}

// ---- SPECGATHER: the camera walk's spectral throughput (host twin: photonmap_render.h) ------
// The walk is MONOCHROMATIC at the camera sample's wavelength; the photon map is POLYCHROMATIC.
// The density estimate below gets this right per photon (`dDiffuseRho(..., ph.lambda)`), but every
// reflectance collected on the way there used to be folded into the scalar `thr` at the CAMERA's
// wavelength, and that scalar then multiplied a photon sum spanning all of them. Exact for a flat
// spectrum, wrong in either direction otherwise -- measured against mode D on one Cornell glossy
// sphere: JH white 0.880x, JH red 1.142x, both ~1.00 after the fix.
// The emission sampler's CDF at `lam` (host twin EmissionSampler::cdfAt): the probability that a
// wavelength drawn by dSampleSceneLambda falls below it.
__device__ static double dEmitCdfAt(const DScene& sc, double lam) {
    const int n = sc.emitSamplerN;
    if (n <= 0 || !sc.emitSamplerCdf) return 0.0;
    const double t = (lam - DLMIN) / sc.emitSamplerStep;
    if (!(t > 0.0)) return 0.0;
    if (t >= (double)n) return 1.0;
    const int b = (int)t;
    return sc.emitSamplerCdf[b] + (t - b) * (sc.emitSamplerCdf[b + 1] - sc.emitSamplerCdf[b]);
}

struct DSpecThr {
    static const int K = kBandProbeBins;   // SpecThr::K, the host twin's grid
    double v[K];
    // The same product at the camera's wavelength, accumulated from the very scalars that went into
    // `thr` -- NOT re-sampled from the grid. `thr` already contains this factor, so `thr * ratio`
    // collapses to (everything else) * v[lambda_p] with no cancellation to get wrong. (The host's
    // first attempt divided by the nearest bin centre instead; where a spectrum plunges near the
    // band edge the two disagree by orders of magnitude and the estimate detonates.)
    double cam;
    int    any;
    // The coins the walk tossed, apart from their weights (host twin SpecThr::p, where the reasoning
    // lives): per bin the product of the branch probabilities, the same at the walk's wavelength,
    // whether some bin cannot take the branches, and then pcam over the probability averaged over
    // the emission sampler the walk's wavelength came from (MODE-M-FOLD-SUPPORT). Probabilities, so
    // float: they enter only that average.
    float  p[K];
    double pcam;
    int    partial;
    double pscale;
    __device__ void init() {
        for (int k = 0; k < K; ++k) { v[k] = 1.0; p[k] = 1.0f; }
        cam = 1.0; any = 0; pcam = 1.0; partial = 0; pscale = 1.0;
    }
    __device__ static Real lamOf(int k) {
        return (Real)(DLMIN + (k + 0.5) * (DLMAX - DLMIN) / (double)K);
    }
    __device__ static int binOf(Real lam) {
        int b = (int)(((double)lam - DLMIN) / (DLMAX - DLMIN) * K);
        return b < 0 ? 0 : (b >= K ? K - 1 : b);
    }
    // Fold in a factor already evaluated ON the grid plus its value at the camera's wavelength --
    // the media term, whose per-wavelength values come from one shared walk rather than from a
    // function that can be called per bin.
    // `support`: whether a zero in `vk` is a hole in the support, as a zero weight in mul is (a
    // fiber's deterministic per-bin weight, host twin SpecThr::mul) -- not for the media term, a
    // ratio-tracking estimate whose zeros are one realization's (host twin SpecThr::mulVec).
    __device__ void mulVec(const DScene& sc, const double* vk, double atCam, bool support) {
        cam *= atCam;
        bool zero = false;
        for (int k = 0; k < K; ++k) {
            if (!any && fabs(vk[k] - vk[0]) > 1e-12 * (1.0 + fabs(vk[0]))) any = 1;
            v[k] *= vk[k];
            if (support && !(vk[k] > 0.0)) { p[k] = 0.0f; zero = true; }
        }
        if (zero) { partial = 1; rescale(sc); }
    }
    // Fold in one spectral factor, f(lambda), at every bin (host twin SpecThr::mul, where the
    // reasoning lives), with `atCam` its value at the camera's wavelength -- passed in rather than
    // recomputed, so the ratio's denominator is exactly the factor sitting inside `thr`.
    template <class F> __device__ void mul(const DScene& sc, const F& f, double atCam) {
        cam *= atCam;
        double a = 0.0;
        bool flat = true, zero = false;
        for (int k = 0; k < K; ++k) {
            const double fk = f(lamOf(k));
            if (k == 0) a = fk;
            else if (fabs(fk - a) > 1e-12 * (1.0 + fabs(a))) flat = false;
            v[k] *= fk;
            if (!(fk > 0.0)) { p[k] = 0.0f; zero = true; }
        }
        if (!flat) any = 1;
        if (zero) { partial = 1; rescale(sc); }     // a weight zero in some bins: a support hole
    }
    // Fold in a coin (host twin SpecThr::mulCoin): f(L, prob) the branch's factor at L, its
    // probability alone in `prob`; the same at the walk's wavelength passed in. Without
    // `exhaustive`, mul's probe; with it every bin, once.
    template <class F> __device__ void mulCoin(const DScene& sc, const F& f, double factorAtCam,
                                               double probAtCam, bool exhaustive) {
        double pr;
        if (!exhaustive) {
            const double a = f(lamOf(0), pr), b = f(lamOf(K / 2), pr), c = f(lamOf(K - 1), pr);
            const double tol = 1e-12 * (1.0 + fabs(a));
            if (fabs(a - b) <= tol && fabs(a - c) <= tol) {
                cam *= factorAtCam;
                for (int k = 0; k < K; ++k) v[k] *= a;
                return;
            }
        }
        // Every bin, once; a coin flat over them and at the walk's wavelength leaves `any` alone.
        cam *= factorAtCam;
        pcam *= probAtCam;
        double a = 0.0;
        bool flat = true;
        for (int k = 0; k < K; ++k) {
            const double fk = f(lamOf(k), pr);
            if (k == 0) a = fk;
            else if (fabs(fk - a) > 1e-12 * (1.0 + fabs(a))) flat = false;
            v[k] *= fk;
            p[k] *= (float)pr;
            if (!(pr > 0.0)) partial = 1;
        }
        if (!flat || fabs(factorAtCam - a) > 1e-12 * (1.0 + fabs(a))) any = 1;
        if (partial) rescale(sc);
    }
    // pscale = pcam / sum_k Q_k p[k] (host twin SpecThr::rescale).
    __device__ void rescale(const DScene& sc) {
        double D = 0.0;
        for (int k = 0; k < K; ++k) {
            const double lo = DLMIN + k * (DLMAX - DLMIN) / (double)K;
            const double hi = DLMIN + (k + 1) * (DLMAX - DLMIN) / (double)K;
            D += (dEmitCdfAt(sc, hi) - dEmitCdfAt(sc, lo)) * (double)p[k];
        }
        pscale = (D > 0.0) ? pcam / D : 1.0;
    }
    __device__ double ratio(Real lamP) const {
        if (!any) return 1.0;
        return (cam > 1e-300) ? v[binOf(lamP)] / cam * pscale : 0.0;
    }
    // The largest ratio over the bins, 1 when nothing chromatic was folded (host twin
    // SpecThr::peak, where the why is): a float sum takes ratio / peak, the double throughput the
    // peak -- the ratio itself passes FLT_MAX behind dark glass (MODEM-COMPOTE-NAN).
    __device__ double peak() const {
        if (!any || !(cam > 1e-300)) return 1.0;
        double m = 0.0;
        for (int k = 0; k < K; ++k) m = fmax(m, v[k] / cam * pscale);
        return (m > 0.0) ? m : 1.0;
    }
    // The carrier as the beam gather reads it (host twin SpecThr::toJSpec). False, and `js`
    // untouched, while nothing chromatic was folded.
    __device__ bool toJSpec(DJSpec& js) const {
        if (!any) return false;
        const double pk = peak();
        const bool live = cam > 1e-300;
        js.any = 1; js.disp = 0; js.lamC = 0.0f; js.invPdfC = 0.0f;
        js.logS = log(pk);
        for (int k = 0; k < K; ++k) js.r[k] = live ? (float)(v[k] / cam * pscale / pk) : 0.0f;
        return true;
    }
};
static_assert(DSpecThr::K == DJSPEC_K, "mode M's carrier and the beam gather's share a grid");

// SPECGATHER: a layered coat's coin at every bin, for mode M's walk and mode S's (host twin
// foldCoatCoin): `coatPick` and `coatR` as dResolveCompound returned them, `coatCos` and `coatTh` the
// incidence and film it captured.
__device__ static void dFoldCoatCoin(DSpecThr& sthr, const DScene& sc, const DMaterial& lm, int coatPick,
                                     double coatR, double coatCos, Real coatTh) {
    const bool cp = coatPick == 1;
    const double pc = cp ? coatR : 1.0 - coatR;
    sthr.mulCoin(sc, [&](Real L, double& pr) {
        const double r = dCoatRAtJ(lm, coatCos, coatTh, L);
        return pr = cp ? r : 1.0 - r;
    }, pc, pc, false);
}

// CURVE-GATHER: a camera walk's step at a strand that holds no photon (host twin strandStep, where the
// why is): its direct light by NEE times `thr` as it arrived, returned; its reflectance into `thr`
// and, if given, the carrier; the ray turned along its diffuse lobe; `cont` false where the walk has
// nothing left to carry.
__device__ static double dStrandStep(const DScene& sc, const DMaterial& m, const DHit& h, Real lambda,
                                     double invPdfL, DRng& rng, double& thr, DSpecThr* sthr,
                                     DVec3& ro, DVec3& rd, bool& cont) {
    const double rho = (double)clamp01(dDiffuseRho(sc, m, h, lambda));
    const DVec3 waS = -rd;   // the arrival, for the coat (dCoatF)
    const DCoatCtx coatC = dCoatCtxOf(m, waS);
    const double d = thr * bkNeeLight(sc, h, (Real)rho, invPdfL, lambda, rng, 0, nullptr, nullptr,
                                      coatC.m ? &coatC : nullptr);
    thr *= rho;
    if (sthr) sthr->mul(sc, [&](Real L) { return (double)clamp01(dDiffuseRho(sc, m, h, L)); }, rho);
    cont = dContinueAlong(h, cosineHemisphere(h.n, rng), DL_REFLECT, ro, rd) && thr > 0.0 &&
           dCoatContinueWalk(sc, m, h, waS, rd, lambda, thr, nullptr, nullptr, 0,   // host twin coatWalkOut
                             [&](const DCoatEval& ce, double s) {
                                 if (sthr) sthr->mul(sc, [&](Real L) { return ce.at(L); }, s); });
    return d;
}

// Mode M's beam gather over one straight camera segment, ADDED to (oX, oY, oZ) times the walk's
// throughput `thr`, with its spectral factors (`sthr`) and the glass the segment runs through
// (`glass`, -1 if none) read at each beam's own wavelength (host twin: photonGather's
// gatherBeams, MODEM-BEAM-CARRIER).
__device__ static void dGatherBeamsM(const DScene& sc, const DBeamMap& bm, const DSpecThr& sthr,
                                     double thr, const DVec3& o, const DVec3& d, Real len,
                                     double aGlass, int glass, DRng& rng,
                                     double& oX, double& oY, double& oZ) {
    DJSpec js;
    const bool jsOn = sthr.toJSpec(js);
    double bX, bY, bZ;
    dGatherPhotonBeams(sc, bm, o, d, len, aGlass, rng, bX, bY, bZ, nullptr, nullptr, 0,
                       jsOn ? &js : nullptr, glass);
    const double t = jsOn ? js.beta(thr) : thr;
    oX += bX * t; oY += bY * t; oZ += bZ * t;
}

// A material's reflect or transmit slot, folded into the carrier (DSpecThr::mul).
__device__ static void dSpecFold(DSpecThr& st, const DScene& sc, const DMaterial& m, const DHit& h,
                                 double atCam, bool transmit) {
    st.mul(sc, [&](Real L) {
        return transmit ? (double)clamp01(dTransmitSlot(sc, m, h, L))
                        : (double)clamp01(dReflectSlot(sc, m, h, L));
    }, atCam);
}


// The device twin of camMediaTrSpec (photonmap_render.h): one straight camera segment's media
// transmittance -- one ratio-tracking sample, or pbeams::kSunDiscTrSamples of them when the
// segment points into a sun's disc -- at every DSpecThr grid wavelength plus, in the last slot,
// the camera's own, all from ONE shared walk, so the scalar `thr` and the
// carrier's denominator are the same number by construction rather than by coincidence.
__device__ static void dCamMediaTrSpec(const DScene& sc, const DVec3& o, const DVec3& d, Real len,
                                       Real lambda, DRng& rng, double* Tv) {
    Real lams[DSpecThr::K + 1];
    for (int k = 0; k < DSpecThr::K; ++k) lams[k] = DSpecThr::lamOf(k);
    lams[DSpecThr::K] = lambda;
    for (int k = 0; k <= DSpecThr::K; ++k) Tv[k] = 1.0;
    // The same sun-disc multi-sampling the scalar version does: a directly-viewed disc is ~10^4x
    // its surroundings, so one sample of the medium in front of it twinkles. Averaging whole
    // VECTORS keeps the wavelengths correlated within each sample.
    int n = 1;
    if (sc.sunCount > 0)
        for (int k = 0; k < sc.nEmitters; ++k)
            if (sc.emitters[k].shape == 6 && dInSunCone(sc.emitters[k], d)) { n = pbeams::kSunDiscTrSamples; break; }
    if (n == 1) {
        dMediaTransmittanceSpec(sc, o, d, len, lams, DSpecThr::K + 1, Tv, rng);
        return;
    }
    double acc[DSpecThr::K + 1];
    for (int k = 0; k <= DSpecThr::K; ++k) acc[k] = 0.0;
    for (int i = 0; i < n; ++i) {
        double one[DSpecThr::K + 1];
        for (int k = 0; k <= DSpecThr::K; ++k) one[k] = 1.0;
        dMediaTransmittanceSpec(sc, o, d, len, lams, DSpecThr::K + 1, one, rng);
        for (int k = 0; k <= DSpecThr::K; ++k) acc[k] += one[k];
    }
    for (int k = 0; k <= DSpecThr::K; ++k) Tv[k] = acc[k] / (double)n;
}

// ----------------------- photon-map camera gather (mode M) -------------------
// Device twin of photonGather (photonmap_render.h). Follows a camera ray through specular
// surfaces (monochromatic at the sampled `lambda`); at the first diffuse / translucent hit it
// estimates reflected radiance either by a DIRECT radius density query into the uploaded
// photon map (fgRays == 0; each photon reflected at ITS OWN wavelength so the estimate is
// built directly in XYZ) or, when final gather is enabled (fgRays > 0 on a Diffuse visible
// point), by NEE direct lighting (bkNeeLight) plus K cosine-hemisphere gather sub-rays one
// bounce into the map (dPhotonGatherSub). Directly-viewed emitters and the environment (on
// gather-ray escape) are added as a monochromatic estimate at the sampled lambda. Beer-Lambert
// interior absorption is tracked exactly like bkRadiance. Keep in sync with photonGather.
//
// `bm` is the photon-BEAM map (mode M with -beams), or null when the scene has no media / the
// beam pass was not run. When present, every segment of this walk — camera ray and each
// specular leg alike — first collects single scatter from the beams it passes near, then takes
// the medium's own attenuation. Beams are the only cache record carrying a photon DIRECTION,
// so they are the only thing that can evaluate a phase function; see photonbeams.h.
//
// `pmC` is the CAUSTIC map: the L.S+.D subset of the same deposit, held apart precisely so it
// can carry its OWN (much smaller) gather radius. One radius cannot serve both populations —
// buildAuto sizes it for the majority, which is the broad ambient wash, and a caustic is a thin
// high-contrast concentration that such a radius convolves flat. The two sets are disjoint and
// share nEmitted, so the estimates just add: nothing is double-counted and nothing is dropped.
__device__ static void dPhotonGather(const DScene& sc, const DPhotonMap& pm,
                                     const DPhotonMap& pmC, int diffraction,
                                     DVec3 ro, DVec3 rd, Real lambda, double invPdfL, DRng& rng,
                                     int fgRays, const DBeamMap* bm,
                                     double& oX, double& oY, double& oZ, const DMediumStack& stk0) {
    oX = oY = oZ = 0.0;
    double thr = 1.0;
    DSpecThr sthr; sthr.init();      // SPECGATHER: the walk's spectral factors (see above)
    bool hairArrival = false;        // HAIR-NEE: the previous vertex was a fiber that already took its direct light
    bool pastDiffuse = false;        // past a fibre's scattering: no glow is seen (host twin photonGather)
    DGlossyMis gmis; gmis.clear();   // GLOSSY-NEE: the lobe density of a glossy continuation, for the MIS weight
    DMediumStack stk = stk0;         // nested-dielectric medium stack, from the camera's
    // norm folded into pX/pY/pZ; radius^2 kept in Real — the distance test runs once per
    // VISITED photon (~85% of visits fail it), and on GeForce parts a double compare +
    // f2d convert issue at 1/64 rate, so keeping the test in FP32 matters.
    const Real r2 = (Real)((double)pm.radius * (double)pm.radius);
    const bool causOn = (pmC.photons != nullptr);
    const Real r2C = (Real)((double)pmC.radius * (double)pmC.radius);
    const int maxBounce = 32;
    const bool volOn = (bm != nullptr) && bm->nNodes > 0 && sc.mediaN > 0;
    // -sunnee: the sun's single scatter is marched, not gathered. Not gated on nNodes: erasing
    // the direct-sun chords may have emptied the map.
    const bool sunOn = (bm != nullptr) && bm->sunNee && sc.sunCount > 0 && sc.mediaN > 0;

    Real curveTmin = (Real)0;   // a fiber's exit clearance for the next hop, curve prims only
    double contSoft = 1.0;   // what the last vertex lends what it reaches (dContSoftening)
    int nullSteps = 0;       // coverage pass-throughs refunded (host twin)
    // The tail (b == maxBounce), in-scattering along it skipped: host twin photonGather.
    for (int b = 0; b <= maxBounce; ++b) {
        const bool tail = (b == maxBounce);
        // Second cancellation point, covering the rays that spend their time BOUNCING rather
        // than in one big beam walk (a scene with no media never enters dGatherPhotonBeams).
        if (dGatherStopped()) return;
        // Glass absorption for THIS segment, hoisted above the escape test because the beam
        // gather below needs it: a beam seen through a dielectric is attenuated to its own
        // closest-approach point, not to the segment end (host twin: photonmap_render.h).
        const int  cmIdx  = stk.topMat();
        const double aGlass = (cmIdx >= 0)
                                  ? (double)specLookup(sc.mats[cmIdx].absorb, lambda) : 0.0;
        // GRADIENT-INDEX: the CAMERA ray bends too. Mode M's forward deposit has marched
        // since GRIN landed, but this gather went straight to closestHit — so a GRIN lens
        // bent the photons and not the view, and the lens rendered dead flat while mode R
        // lensed the same scene into a radial disc. Nothing warned.
        //
        // The volume estimator runs PER STRAIGHT SUB-SEGMENT (dGrinMarch driven one step at
        // a time): Beam x Ray is a closest approach between two straight lines, so a curved
        // camera ray must be fed to it piecewise. Exact, not approximate — radiance along a
        // path is additive over its pieces, and `thr` carries each piece's transmittance
        // forward, which is what dGatherPhotonBeams' own Tr_cam(0 -> tCam) assumes (it
        // measures from the sub-segment start).
        if (sc.hasGrin) {
            bool bent = false;
            int carryM = -1; double carryN = 0.0;  // the marcher's index carry, between driven steps
            for (int gs = 0; gs < 200000; ++gs) {  // same safety cap the marcher applies
                DVec3 pro = ro;
                DGrinMedia gm;                     // rng == nullptr: bend only, no collisions
                gm.camera = true;
                gm.carryM = carryM; gm.carryN = carryN;
                dGrinMarch(sc, ro, rd, &gm, 1, /*camHide=*/(b == 0));  // — a camera ray's volume answer IS the
                carryM = gm.carryM; carryN = gm.carryN;
                // This step's radiance factor (or the straight remainder's, when the march stopped short
                // of a surface), before what the walk gathers along it (host twin photonGather, grin.h
                // MarchInfo); a factor flat in wavelength leaves the spectral carrier as it is.
                thr *= gm.radScale;
                bent = bent || gm.bent;
                if (!gm.stepped) break;            //   beam gather below
                const Real slen = (Real)gm.arc;
                if ((volOn || sunOn) && slen > 0) {
                    double bX = 0.0, bY = 0.0, bZ = 0.0;
                    if (volOn && !tail) dGatherBeamsM(sc, *bm, sthr, thr, pro, rd, slen, aGlass, cmIdx, rng, oX, oY, oZ);
                    if (sunOn && !tail) dSunNeeMarch(sc, pro, rd, slen, aGlass, lambda, invPdfL, rng, bX, bY, bZ);
                    oX += bX * thr; oY += bY * thr; oZ += bZ * thr;
                    {   // SPECGATHER: extinction is spectral too (badly so in a coloured medium)
                        double Tv[DSpecThr::K + 1];
                        dCamMediaTrSpec(sc, pro, rd, slen, lambda, rng, Tv);
                        thr *= Tv[DSpecThr::K];
                        sthr.mulVec(sc, Tv, Tv[DSpecThr::K], false);
                    }
                }
                // Coloured glass is spectral (SPECGATHER), whatever it absorbs at the walk's own
                // wavelength: a glass clear there may absorb at a photon's (host twin).
                if (cmIdx >= 0) {
                    const double arc = (double)gm.arc;
                    thr *= exp(-aGlass * arc);
                    sthr.mul(sc, [&](Real L) {
                        return exp(-(double)specLookup(sc.mats[cmIdx].absorb, L) * arc);
                    }, exp(-aGlass * arc));
                }
            }
            // Past a bend the light the walk meets was reached by no light sample.
            if (bent) { hairArrival = false; gmis.clear(); }
            if (thr <= 0.0) return;
        }
        // b == 0 is the camera ray dPhotonGather was handed (mode M's eye pass); see
        // DMaterial::hideCamera. dPhotonGatherSub's walk is NOT given this: a final-gather
        // sub-ray leaves a visible point, so it is an indirect ray and must see the flat.
        DHit h = closestHitIn(stk, sc, ro, rd, curveTmin > (Real)0 ? (Real)0 : RAY_TMIN, BIG,
                                   /*camHide=*/(b == 0), curveTmin);
        curveTmin = (Real)0;
        // --- Participating media along this segment (mode M with -beams) ------------------
        // Done BEFORE `thr` takes the segment's attenuation, for the same reason: each
        // gathered beam carries the transmittance to ITS OWN closest approach.
        if (volOn || sunOn) {
            const Real dSeg = h.valid ? h.t : (Real)1e30;
            double bX = 0.0, bY = 0.0, bZ = 0.0;
            if (volOn && !tail) dGatherBeamsM(sc, *bm, sthr, thr, ro, rd, dSeg, aGlass, cmIdx, rng, oX, oY, oZ);
            if (sunOn && !tail) dSunNeeMarch(sc, ro, rd, dSeg, aGlass, lambda, invPdfL, rng, bX, bY, bZ);
            oX += bX * thr; oY += bY * thr; oZ += bZ * thr;
            {   // SPECGATHER: extinction is spectral too (badly so in a coloured medium)
                double Tv[DSpecThr::K + 1];
                dCamMediaTrSpec(sc, ro, rd, dSeg, lambda, rng, Tv);
                thr *= Tv[DSpecThr::K];
                sthr.mulVec(sc, Tv, Tv[DSpecThr::K], false);
            }
            if (thr <= 0.0) return;
        }
        if (h.valid && cmIdx >= 0) {                     // Beer-Lambert in current medium
            const double tt = (double)h.t;               // (spectral: see the march above)
            thr *= exp(-aGlass * tt);
            sthr.mul(sc, [&](Real L) {
                return exp(-(double)specLookup(sc.mats[cmIdx].absorb, L) * tt);
            }, exp(-aGlass * tt));
        }
        if (!h.valid) {                                  // escaped -> environment
            // Direct env term on gather-ray escape (mirrors photonGather, fgRays==0). The
            // map already carries env's INDIRECT bounces (the deposit emits env photons);
            // this adds env's DIRECT contribution, monochromatic at the sampled lambda.
            if (sc.envIndex >= 0) {
                // camera / specular escape, weight 1: the env as SEEN (bilinear), host photonGather
                double envRad = (sc.env.scale != nullptr)
                                    ? dEnvRadianceSeen(sc.env, rd, lambda)
                                    : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lambda);
                double e = thr * envRad * invPdfL * contSoft;
                oX += (double)cieX(lambda) * e;
                oY += (double)cieY(lambda) * e;
                oZ += (double)cieZ(lambda) * e;
            }
            // Directly-viewed solar disc. This walk terminates at the first diffuse vertex, so an
            // escape reaching here is a camera ray, a specular chain, or a fiber's continuation --
            // and a fiber's own light sampling (HAIR-NEE) already took the sun (`hairArrival`).
            // The sky is not among what it took. (Host twin: photonmap_render.h.)
            if (sc.sunCount > 0 && !hairArrival) {
                double e = thr * dSunRadianceMis(sc, gmis, rd, lambda) * invPdfL * contSoft;
                oX += (double)cieX(lambda) * e;
                oY += (double)cieY(lambda) * e;
                oZ += (double)cieZ(lambda) * e;
            }
            return;
        }

        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (dIsCompound(mp->type)) {
            int coatPick = -1; double coatR = 0.0, coatCos = 1.0; Real coatTh = (Real)0;
            int child = dResolveCompound(sc, *mp, h, rd, lambda, rng, &coatPick, &coatR, &coatCos, &coatTh);
            if (child < 0) return;                       // absorbed by the mix
            if (coatPick >= 0) dFoldCoatCoin(sthr, sc, *mp, coatPick, coatR, coatCos, coatTh);   // SPECGATHER
            mp = &sc.mats[child]; matId = child;
        }
        const DMaterial& m = *mp;

        // Self-emission of a directly-viewed / specularly-seen emitter, one-sided by the
        // geometric normal to match bkRadiance. Prefers the registered emitter's baked SPD
        // (it may carry a `power`/`lumens` flux normalisation the raw material spectrum does
        // not) and falls back to the material's own `emit` when this geometry registered no
        // emitter at all — only tessellated geometry does, so a glowing `quad`, isosurface or
        // CSG solid has an emissive MATERIAL and no DEmitter. Testing `li >= 0` alone is what
        // made gallery_rain's grid floor lose its green lines entirely under mode M on GPU.
        //
        // NOT a `return`: an emissive material still has a BSDF, so a glowing DIFFUSE surface
        // both emits and reflects and the walk falls through to the density estimate. Host
        // twin: photonGather — see its comment for the four-mode measurement.
        {
            const int li = dEmitterForHit(sc, h, matId);
            const DSpec* eSpd = (li >= 0)                        ? &sc.emitters[li].emitSpd
                               : (m.matIsLight && !pastDiffuse)   ? &m.matEmit   // a glow: seen only before
                                                                  : nullptr;
            if (eSpd && dot(rd, h.ng) < 0 && !hairArrival) {
                // GLOSSY-NEE's other half: if the previous bounce was a glossy lobe that was
                // already connected to this light, the two strategies are balance-weighted so
                // the emitter is not counted twice (host twin: photonGather).
                const double wMis = (gmis.pdf > 0.0)
                                  ? dGlossyHitWeight(sc, gmis, li, rd, &h) : 1.0;
                double e = (double)specLookup(*eSpd, lambda) * thr * invPdfL * wMis * contSoft
                         * dEmitHitScale(sc, m.emitPat, li, h, rd * (Real)(-1));    // pattern and aim at this hit
                oX += (double)cieX(lambda) * e;
                oY += (double)cieY(lambda) * e;
                oZ += (double)cieZ(lambda) * e;
            }
        }
        if (tail) {                                     // the light it reached, and no more --
            // past a strand it passes, which is no vertex (host twin; bkTailPassesStrand)
            if (m.type == D_HAIR && nullSteps < 256 &&
                bkTailPassesStrand(sc, m, h, lambda, ro, rd, rng, &curveTmin)) { --b; ++nullSteps; continue; }
            return;
        }

        const bool hairArrivalIn = hairArrival;   // ...and this (host twin)
        hairArrival = false;
        // Cleared HERE, not at the loop top: `gmis` is written by the PREVIOUS bounce's
        // glossy branch and read by THIS bounce's emitter/sun sites above, so clearing it
        // earlier would erase it a few lines before its only reader -- leaving the NEE
        // connection in place with no compensating weight, i.e. double counting.
        const DGlossyMis gmisIn = gmis;   // a hair pass-through puts it back
        gmis.clear();
        const double contSoftIn = contSoft;
        contSoft = 1.0;   // a specular vertex lends nothing; the glossy case sets its own
        // CURVE-GATHER (host twin photonGather, where the why is): no gather on a `type diffuse`
        // strand; it is lit by NEE and the walk goes on along its diffuse lobe (dStrandStep).
        if (dGatherlessStrand(m, h)) {
            bool cont;
            const double dc = dStrandStep(sc, m, h, lambda, invPdfL, rng, thr, &sthr, ro, rd, cont);
            oX += (double)cieX(lambda) * dc; oY += (double)cieY(lambda) * dc; oZ += (double)cieZ(lambda) * dc;
            hairArrival = true;
            pastDiffuse = true;
            if (!cont) return;
            continue;
        }
        if (m.type == D_DIFFUSE || m.type == D_DIFFUSETRANSMIT || m.type == D_FLUORESCENT) {
            const DVec3 gn = dShadedNg(h);   // the true normal, as the photons carry it
            if (fgRays > 0 && m.type == D_DIFFUSE) {
                // Jensen final gather (device twin of photonGather's fgRays branch): decouples
                // the directly-seen surface's sharpness from the gather radius by moving the
                // density-estimate blur one bounce away (to y, inside dPhotonGatherSub).
                //   (a) DIRECT lighting from finite emitters via low-variance NEE (bkNeeLight).
                double rhoVis = (double)clamp01(dDiffuseRho(sc, m, h, lambda));
                const DCoatCtx coatC = dCoatCtxOf(m, -rd);   // a coated body's
                double direct = bkNeeLight(sc, h, (Real)rhoVis, invPdfL, lambda, rng, 0, nullptr, nullptr,
                                           coatC.m ? &coatC : nullptr);
                double dcie = thr * direct;
                oX += (double)cieX(lambda) * dcie;
                oY += (double)cieY(lambda) * dcie;
                oZ += (double)cieZ(lambda) * dcie;
                //   (b) INDIRECT (+ env + specular-direct) via K cosine-weighted hemisphere
                //       sub-rays, each querying the map ONE bounce away; the cosine/pdf and
                //       Lambertian 1/pi cancel to rho(vis), folded inside dPhotonGatherSub.
                double fx = 0.0, fy = 0.0, fz = 0.0;
                for (int k = 0; k < fgRays; ++k) {
                    DVec3 grd = cosineHemisphere(h.n, rng);
                    if (dLeavesWrongSide(h, grd, DL_REFLECT)) continue;   // across the surface: gathers nothing
                    DVec3 gro = dOffsetAlong(h.p, h.ng, grd, h.pscale);
                    double sx, sy, sz;
                    dPhotonGatherSub(sc, pm, pmC, diffraction, gro, grd, lambda, invPdfL, h, m, -rd, rng,
                                     sx, sy, sz, stk);
                    fx += sx; fy += sy; fz += sz;
                }
                double inv = thr / (double)fgRays;
                oX += fx * inv; oY += fy * inv; oZ += fz * inv;
                return;
            }
            // Radius density estimate at the visible point, accumulated in XYZ (each photon
            // folded at its own wavelength): L_r = (1/N) sum_p rho(l_p)/pi * Phi_p / (pi r^2).
            // Everything but rho(l_p) is baked into the record's pX/pY/pZ (see DGatherPhoton),
            // so the per-photon work is the two rejection tests + one rho + three FMAs — all
            // FP32: a float sum of <= (cell occupancy) same-sign terms errs ~n*2^-24 relative,
            // far below the 8-bit output quantum, and FP64 FMAs would issue at 1/64 rate. The
            // per-sample total is promoted to double once at the end (film math stays double).
            float gx = 0.f, gy = 0.f, gz = 0.f;
            const Real slkM = gatherBandSlack<Real>(pm.radius, h.fiberRadius, dRoundScale(h));
            // The carrier's peak goes into the double throughput, ratio / peak into the float sum
            // (DSpecThr::peak): behind dark glass the ratio alone overflows a float.
            const double pk = sthr.peak(), ipk = 1.0 / pk;
            dPmNeighborhood(pm, h.p, [&](int k) {
                const DGatherPhoton& ph = pm.photons[k];
                DVec3 d = h.p - ph.pos;
                const Real d2 = dot(d, d);
                if (d2 > r2) return;
                bool farSide;
                if (!dOnGatherSurfaceSided(h.p, gn, ph.pos, ph.n, d2, slkM, m.type == D_DIFFUSETRANSMIT, farSide)) return;   // another surface's
                // SPECGATHER: the walk's reflectances belong at THIS photon's wavelength,
                // not the camera's. Exactly 1.0 when every one of them was flat.
                float rho = (float)(dGatherAlbedo(sc, m, h, (Real)ph.lambda, farSide)
                                    * (sthr.ratio((Real)ph.lambda) * ipk));
                gx += rho * ph.pX;
                gy += rho * ph.pY;
                gz += rho * ph.pZ;
            });
            // M-GATHERAREA, the MAIN map's coverage, on the main map's sum alone -- before the
            // caustic map's is added, which takes its own coverage below (host twin: each `est`
            // scales its own map). The device folds norm/pi into every photon record at upload
            // (DGatherPhoton), so the correction multiplies the sum rather than a normalisation.
            if (sc.gatherArea) {
                const double ms = dGatherAreaScale(sc.gaBias,
                    dGatherCoverage(sc, h.p, gn, (Real)sqrt((double)r2), slkM, rng, sc.gatherArea, h.pscale,
                                    h.fiberRadius));
                gx = (float)((double)gx * ms);
                gy = (float)((double)gy * ms);
                gz = (float)((double)gz * ms);
            }
            // Caustic map: same estimator, its own PER-QUERY radius, summed in (see the
            // header comment and dPmAdaptiveRadius). The area ratio r^2/r_q^2 corrects the
            // fixed-radius normalisation already folded into pX/pY/pZ.
            if (causOn) {
                const Real slkC = gatherBandSlack<Real>(pmC.radius, h.fiberRadius, dRoundScale(h));
                const Real rq  = dPmAdaptiveRadius(pmC, h.p, gn, h.fiberRadius, m.type == D_DIFFUSETRANSMIT, dRoundScale(h));
                const Real r2q = rq * rq;
                float cx = 0.f, cy = 0.f, cz = 0.f;
                dPmNeighborhood(pmC, h.p, [&](int k) {
                    const DGatherPhoton& ph = pmC.photons[k];
                    DVec3 d = h.p - ph.pos;
                    const Real d2 = dot(d, d);
                    if (d2 > r2q) return;
                    bool farSide;
                    if (!dOnGatherSurfaceSided(h.p, gn, ph.pos, ph.n, d2, slkC, m.type == D_DIFFUSETRANSMIT, farSide)) return;
                    // SPECGATHER: the walk's reflectances belong at THIS photon's wavelength,
                    // not the camera's. Exactly 1.0 when every one of them was flat.
                    float rho = (float)(dGatherAlbedo(sc, m, h, (Real)ph.lambda, farSide)
                                        * (sthr.ratio((Real)ph.lambda) * ipk));
                    cx += rho * ph.pX;
                    cy += rho * ph.pY;
                    cz += rho * ph.pZ;
                });
                const float aw = (r2q > (Real)0) ? (float)((double)r2C / (double)r2q) : 0.f;
                // M-GATHERAREA: the caustic map has its OWN radius, so its own coverage --
                // matching the host, where the correction lives inside each `est` call keyed
                // on that map's rq. One shared coverage would be cheaper and wrong.
                double cs = 1.0;
                if (sc.gatherArea)
                    cs = dGatherAreaScale(sc.gaBias, dGatherCoverage(sc, h.p, gn, rq, slkC, rng, sc.gatherArea, h.pscale,
                                                          h.fiberRadius));
                gx += cx * (float)(aw * cs);
                gy += cy * (float)(aw * cs);
                gz += cz * (float)(aw * cs);
            }
            const double tp = thr * pk;
            oX += (double)gx * tp; oY += (double)gy * tp; oZ += (double)gz * tp;
            return;
        }

        const DVec3 wArr = -rd;                          // the arrival, for the coat (dCoatF)
        switch (m.type) {                                // specular walk (monochromatic)
            case D_MIRROR: {
                {   const double rC = (double)clamp01(dReflectSlot(sc, m, h, lambda));
                    thr *= rC;
                    dSpecFold(sthr, sc, m, h, rC, false);   // SPECGATHER
                }
                if (!dContinueAlong(h, reflectv(rd, h.n), DL_REFLECT, ro, rd)) return; break;
            }
            case D_GLOSSY: {
                if (!dGlossyCameraSide(rd, h.n)) return;        // met from below ns (host twin)
                // NEXT-EVENT ESTIMATION AT A GLOSSY VERTEX (GLOSSY-NEE, 0.340.0). The host
                // gather has always done this; the device gather did not, so the only route
                // from a metal to a light was a lobe sample landing on it -- gallery_rain's
                // gold gyroid rendered 44 % below the host while every diffuse surface agreed
                // to 1 %. Taken BEFORE `thr *= rC`: the connection carries rC inside dBsdfF.
                if (sc.bkGlossyNee) {
                    const DNeeBsdf nb{&m, rd * (Real)(-1)};
                    if (sc.specNee) {
                        // SPECTRAL (0.342.0, host twin photonmap_render.h): one shadow ray,
                        // evaluated over the DSpecThr grid, so a gold lobe under a warm light
                        // stops painting each sample one colour.
                        Real   lamG[DSpecThr::K];
                        double ratG[DSpecThr::K], xyz[3] = {0.0, 0.0, 0.0};
                        for (int i = 0; i < DSpecThr::K; ++i) {
                            lamG[i] = DSpecThr::lamOf(i);
                            ratG[i] = sthr.ratio(lamG[i]);
                        }
                        const double dLam = (DLMAX - DLMIN) / (double)DSpecThr::K;
                        bkNeeLightSpecGlossy(sc, h, nb, lamG, ratG, DSpecThr::K, dLam,
                                             lambda, thr, rng, xyz);
                        oX += xyz[0]; oY += xyz[1]; oZ += xyz[2];
                    } else {
                        const DCoatCtx coatC = dCoatCtxOf(m, -rd);
                        const double dc = thr * bkNeeLight(sc, h, (Real)1, invPdfL, lambda, rng, 0, nullptr, &nb,
                                                           coatC.m ? &coatC : nullptr);
                        oX += (double)cieX(lambda) * dc;
                        oY += (double)cieY(lambda) * dc;
                        oZ += (double)cieZ(lambda) * dc;
                    }
                }
                {   const double rC = (double)clamp01(dReflectSlot(sc, m, h, lambda));
                    thr *= rC;
                    dSpecFold(sthr, sc, m, h, rC, false);   // SPECGATHER
                }
                const DGlossyDraw gd = dDrawGlossy(sc, m, h, h.n, -rd, rng);
                const DVec3& o = gd.o;
                if (dot(o, h.n) <= 0) return;
                thr *= (double)gd.w;   // the draw's weight: the same at every wavelength, so not sthr's
                // An iridescent lobe's interference reflectance at the pair's half-vector, into the
                // scalar and the spectral throughput alike; where it is 0 at the walk's own
                // wavelength the path carries nothing (and sthr's ratio would divide by it).
                // Host twin photonmap_render.h.
                if (gd.irid) {
                    const double Rh = (double)gd.R(lambda);
                    if (!(Rh > 0.0)) return;
                    thr *= Rh;
                    sthr.mul(sc, [&](double Lw) { return (double)gd.R((Real)Lw); }, Rh);
                }
                if (sc.bkGlossyNee) {                       // the lobe's own density, for the weight above
                    gmis.pdf  = dGlossyPdfHit(sc, m, h, rd * (Real)(-1), o);
                    gmis.from = h.p;
                    gmis.n    = h.n;
                }
                if (!dContinueAlong(h, o, DL_REFLECT, ro, rd)) return;
                contSoft = dContSoftening(m, h, rd); break;
            }
            case D_DIELECTRIC:
            case D_THINFILM:
            case D_MULTILAYER: {
                // A dielectric body's surface, plain or coated: radiance across a refraction, and
                // SPECGATHER re-expresses the step at each photon's wavelength (dStepFactorAt).
                DVec3 nro, nrd; double rs = 1.0;
                DStepRecord st;
                const DVec3 d0 = rd;
                if (!dDielectricStep(sc, m, h, rd, lambda, rng, stk, nro, nrd, nullptr, &rs,
                                     &st)) return;
                thr *= rs;                               // radiance across the interface
                sthr.mulCoin(sc, [&](Real L, double& pr) {
                    return dStepFactorAt(sc, m, h, d0, st, L, &pr);
                }, st.factor, st.prob, m.type != D_DIELECTRIC);
                ro = nro; rd = nrd; break;
            }
            case D_HALFMIRROR: {
                Real r = clamp01(dReflectSlot(sc, m, h, lambda));
                const bool refl = rng.uniform() < r;
                if (refl) { if (!dContinueAlong(h, reflectv(rd, h.n), DL_REFLECT, ro, rd)) return; }
                else      { ro = dOffsetAlong(h.p, h.ng, rd, h.pscale); }
                // SPECGATHER: the coin at each photon's wavelength (host twin).
                const double pc = refl ? (double)r : 1.0 - (double)r;
                sthr.mulCoin(sc, [&](Real L, double& pr) {
                    const double rL = (double)clamp01(dReflectSlot(sc, m, h, L));
                    return pr = refl ? rL : 1.0 - rL;
                }, pc, pc, true);
                break;
            }
            case D_FILTER: {
                {   const double tC = (double)clamp01(dTransmitSlot(sc, m, h, lambda));
                    thr *= tC;
                    dSpecFold(sthr, sc, m, h, tC, true);    // SPECGATHER
                }
                ro = dOffsetAlong(h.p, h.ng, rd, h.pscale); break;
            }
            case D_HAIR: {
                // HAIR (0.335.0): the device twin of photonGather's Hair case. Scatter through the
                // fiber BCSDF and continue -- fibers are never gathered on; direct light at the
                // fiber by NEE (HAIR-NEE, mode R's split: an emitter the continuation then hits
                // is not counted again); and the sample's weight folded into the spectral carrier
                // at every grid wavelength (SPECGATHER for hair: the sampled direction and its pdf
                // are fixed, only the BCSDF value moves with the absorption).
                const DVec3 wPrev = rd * (Real)(-1);
                const DHairShade hsv = dHairShadeAt(sc, m, h, lambda, wPrev);
                {   const double direct = bkNeeLight(sc, h, (Real)1, invPdfL, lambda, rng, 0, &hsv);
                    const double dc = thr * direct;
                    oX += (double)cieX(lambda) * dc; oY += (double)cieY(lambda) * dc; oZ += (double)cieZ(lambda) * dc;
                    hairArrival = true;
                }
                const double u0 = rng.uniform(), u1 = rng.uniform(), u2 = rng.uniform(), u3 = rng.uniform();
                double pdfH = 0.0, fv = 0.0;
                dhair::LobeAngular la;                    // filled by the sample's own f()
                const dhair::V3 wl = dhair::sample(hsv.b, hsv.woLocal, u0, u1, u2, u3, pdfH, fv, &la);
                if (!(pdfH > 0.0) || !(fv > 0.0)) return;
                const double cosLong = dhair::safeSqrt(1.0 - dhair::sqr(dhair::clampd(wl.x, -1.0, 1.0)));
                // A coverage pass-through returns EXACTLY -wo (negation is exact in IEEE), carries
                // weight 1 and is achromatic, so it must not be spectrally reweighted: its pdf is
                // the branch probability, not a scatter density, and dividing by it is meaningless.
                const bool passedThru = (hsv.b.opacity < 1.0) && wl.x == -hsv.woLocal.x &&
                                        wl.y == -hsv.woLocal.y && wl.z == -hsv.woLocal.z;
                double wCam = fv * cosLong / pdfH; wCam = wCam < 0.0 ? 0.0 : (wCam > 1.0 ? 1.0 : wCam);
                thr *= wCam;
                // ...and no vertex: the last real one's emitter-hit bookkeeping stands (host twin).
                if (passedThru) {   // ...and its bounce refunded (host twin)
                    hairArrival = hairArrivalIn; gmis = gmisIn; contSoft = contSoftIn;
                    if (nullSteps < 256) { --b; ++nullSteps; }
                }
                if (!passedThru) {   // each bin folded from the sample's angular half (dHairFoldFCos)
                    pastDiffuse = true;   // a fibre's scattering: no glow is seen past it
                    const double den = m.hairSigmaAFromReflect ? dhair::sigmaADenominator(m.hairBetaN) : -1.0;   // hoisted: three pow() per bin otherwise
                    const double specPat = dHairSpecPatAt(sc, m, h);   // hoisted: the pattern is the same at every bin
                    double vk[DSpecThr::K];
                    for (int k = 0; k < DSpecThr::K; ++k) {
                        const DHairSpec spk = dHairSpecAt(sc, m, h, DSpecThr::lamOf(k), specPat,
                                                          hsv.b.kappa, den);
                        const double w = dHairFoldFCos(la, hsv, spk, wl, cosLong, nullptr) / pdfH;
                        vk[k] = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
                    }
                    sthr.mulVec(sc, vk, wCam, true);
                }
                const DVec3 wOut = dhair::toWorld(hsv.fr, wl);
                // Clear the strand by skipping curves within its exit distance on the next
                // trace, from a ULP-moved origin (see interactHair): moving the origin out by
                // that distance carries a ray leaving near a root through the skin below.
                curveTmin = dHairExitOffset(hsv, h.n, wOut, h.p);
                ro = dOffsetPoint(h.p, wOut, h.pscale);
                rd = wOut;
                break;
            }
            default: {                                   // Grating: approx reflect
                {   const double rC = (double)clamp01(dReflectSlot(sc, m, h, lambda));
                    thr *= rC;
                    dSpecFold(sthr, sc, m, h, rC, false);   // SPECGATHER
                }
                if (!dContinueAlong(h, reflectv(rd, h.n), DL_REFLECT, ro, rd)) return; break;
            }
        }
        // A coated vertex's continuation takes the coat, at each photon's wavelength too (host twin
        // coatWalkOut). A fibre has no coat, so its pass-through never reaches this.
        if (!dCoatContinueWalk(sc, m, h, wArr, rd, lambda, thr, nullptr, nullptr, 0,
                               [&](const DCoatEval& ce, double s) {
                                   sthr.mul(sc, [&](Real Lw) { return ce.at(Lw); }, s); }))
            return;
        if (thr <= 0.0) return;
    }
}

// One thread per (pixel, sample); grid-strides over [idxBase, idxEnd) of this chunk's flat
// (pixel, sample) index space. Mirrors kBackward's seeding (global sample index) so a
// chunked gather is decorrelated across chunks. The gather already returns XYZ, so (unlike
// kBackward) no cie(lambda) multiply is applied.
//
// The [idxBase, idxEnd) window exists so the host can cut ONE chunk into several launches
// and poll the stop flag between them. At the resolutions this mode runs at, the chunk
// size clamps to a single spp (200000/npix < 1), so without the window the finest seam an
// external `ftrace -stop` could land on was a whole frame's worth of gathering — minutes,
// with -beams — which is what made a perfectly healthy stop look like a failure. Every
// seed here is a pure function of the GLOBAL index gidx, never of the launch bounds, so a
// sliced chunk samples exactly the same paths as an unsliced one.
//
// The launch geometry is named because the SLICE SIZE MUST NOT FALL BELOW IT. This is a
// persistent grid that grid-strides over its sample range, so a slice with fewer samples
// than there are threads leaves threads idle outright — see the sliceLo comment in
// renderPhotonMapSharedCuda for the 11x that cost before the floor was tied to this.
static constexpr int kGatherGrid  = 2048;
static constexpr int kGatherBlock = 128;
#ifndef FT_FAMILY_TU
__global__ void kGather(DScene sc, DPhotonMap pm, DPhotonMap pmC, DBeamMap bm, const __grid_constant__ DCamera cam,
                        double* film, double* hits,
                        long long idxBase, long long idxEnd,
                        long long chunkSpp, long long sppTotal,
                        long long sampleBase, int resX, int diffraction, int fgRays,
                        unsigned long long seedBase) {
    D_MOTION_CLEAR();   // this thread's time: the launch's (passU) until a path sets one
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    int poll = 0;
    for (long long idx = idxBase + g; idx < idxEnd; idx += G) {
        // Cooperative cancellation, and the ONLY thing that makes a gather interruptible.
        //
        // The host can only poll BETWEEN launches, and at every ordinary resolution there is
        // exactly one launch: `sliceOcc` (4 * kGatherGrid * kGatherBlock = 1048576) exceeds a
        // frame's sample count (518400 at 960x540; 172800 for a whole 3-spp render at
        // 320x180), so the slice loop never iterates and there is no "between". Measured
        // before this existed: a `-stop` issued one minute into a 320x180 gather returned
        // 879 s later having cancelled nothing — the render had simply run to completion. On
        // gallery_rain at -n 800M the same single launch runs for hours.
        //
        // Shrinking the slice is NOT the alternative: tying it below the occupancy floor left
        // most of this persistent grid idle and cost a measured 11x (see sliceLo). Polling
        // here is independent of slice size, so the floor stays exactly as tuned.
        //
        // NOTE this poll is nearly worthless ON ITS OWN and is kept only as a cheap guard for
        // the multi-sample-per-thread case: at 960x540 there are 518400 samples for 262144
        // threads, i.e. ONE iteration each, so it fires at launch time and never again. The
        // polls that actually make a stop land are the ones inside dGatherPhotonBeams' BVH
        // walk and dPhotonGather's bounce loop, where the time is really spent.
        //
        // Returning early leaves the scratch film partially written, which is exactly what
        // the host's `chunkDone = false` discard path already exists to throw away.
        if ((poll++ & 3) == 0 && dGatherStopped()) return;
        long long pix  = idx / chunkSpp;
        long long gidx = pix * sppTotal + sampleBase + (idx - pix * chunkSpp);
        DRng rng; rng.seedUnit((unsigned long long)gidx, seedBase);
        int px = (int)(pix % resX);
        int py = (int)(pix / resX);

        double pdf = 0.0;
        // STRATIFIED hero wavelength (0.314.0): sample s of this pixel takes the s-th point of the
        // golden-ratio sequence over the emission CDF, under a per-pixel random rotation so
        // neighbouring pixels do not walk the spectrum in step and every sample is uniform. One
        // monochromatic sample's CIE weight varies 0..1 across the band, which at 32 spp was
        // +-25 % of luminance on any bright spectral term -- the sun's disc through the cloud
        // twinkled frame to frame, and the rainbow's chroma grain was mostly this; spread evenly
        // the same integral converges nearly like 1/N instead of 1/sqrt(N). A sequence rather than
        // N strata because any run of it is even: a stop that keeps only the launch's first
        // chunks keeps a tilt-free spectrum. `s` counts from this call's first sample, the call
        // being one epoch under its own salt (the shared path's refresh loop). Host twin:
        // renderPhotonCamera (photonmap_render.h).
        const unsigned long long sIdx = (unsigned long long)(gidx - pix * sppTotal);
        DRng rrot;
        rrot.seedUnit((unsigned long long)pix, seedBase ^ 0x5851F42D4C957F2DULL);
        double u = (double)rrot.uniform() + goldenFrac(sIdx);
        u -= floor(u);
        if (!(u < 1.0)) u = 0.999999999;
        Real lambda = dSampleSceneLambdaU(sc, u, pdf);
        // Retire on the reject path too. It is a rare sample, but an undercount here would
        // leave the bar stalled just short of 100% at the end of a launch — which reads as
        // exactly the wedged render this counter exists to rule out.
        if (pdf <= 0.0) { dGatherRetire(); continue; }
        double invPdfL = dInvPdfLambda(sc, lambda);

        DVec3 ro, rd;
        Real jx = rng.uniform(), jy = rng.uniform();
        // A moving camera (host twin renderPhotonCamera): each sample poses the camera at its own
        // time, as mode R's do, and the scene stays at the map's -- timeless over a still scene,
        // the instant under a meter's map of a moving one. An epoch of a moving scene arrives with
        // its camera posed and no time of its own (renderPhotonMapSharedCuda).
        DCamera camT;
        const DCamera& camS = dPoseCameraAtSample(cam, (unsigned long long)pix, (unsigned long long)sIdx, camT)
                            ? camT : cam;
        dGenRay(camS, px, py, jx, jy, ro, rd);           // pinhole only (lens cams gated to CPU)

        double oX, oY, oZ;
        dPhotonGather(sc, pm, pmC, diffraction, ro, rd, lambda, invPdfL, rng, fgRays,
                      bm.nNodes > 0 ? &bm : nullptr, oX, oY, oZ, camS.enclosure);
        size_t o = ((size_t)py * resX + px) * 3;
        atomicAdd(&film[o + 0], oX);
        atomicAdd(&film[o + 1], oY);
        atomicAdd(&film[o + 2], oZ);
        if (hits) atomicAdd(&hits[(size_t)py * resX + px], 1.0);
        dGatherRetire();   // one retired sample — the host's only view inside this launch
    }
}
#endif

// Fold a completed chunk's scratch accumulation into the camera's film, then leave the
// scratch for the host to clear. kGather writes into scratch rather than straight into the
// film so that a chunk abandoned part-way (external `ftrace -stop`, closed window) can be
// DISCARDED whole. That matters because the film is normalised by a single global spp
// count, not per pixel: committing half a chunk would leave the pixels that got their
// sample ~1/spp brighter than the ones that did not, i.e. a visible band across the last
// frame of every stopped render. Dropping the partial chunk makes a stop anywhere inside a
// chunk produce exactly the image a stop at the preceding chunk boundary would have.
#ifndef FT_FAMILY_TU
__global__ void kFilmFold(double* film, double* hits, const double* sfilm, const double* shits,
                          long long npix) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long S = (long long)gridDim.x * blockDim.x;
    for (; i < npix; i += S) {
        film[i * 3 + 0] += sfilm[i * 3 + 0];
        film[i * 3 + 1] += sfilm[i * 3 + 1];
        film[i * 3 + 2] += sfilm[i * 3 + 2];
        if (hits && shits) hits[i] += shits[i];
    }
}
#endif

// ============================ device: SPPM (mode S) ============================
// Device twin of sppm_render.h. SPPM runs REPEATED bounded photon passes with a per-pixel
// shrinking gather radius; per-pixel progressive state (tau/radius/nAcc/directSum + the
// current pass's visible point) lives resident on the device across passes. Each of the
// three phases below is one kernel; the deposit + host grid build reuse the mode-M path.

// Trace one camera ray to its first diffuse/translucent hit (the "visible point"), following
// specular surfaces exactly like dPhotonGather / CPU sppmVisiblePoint. Returns the specular
// throughput in `thrOut` and the diffuse hit in `vp`, its matId the lobe a compound resolved to
// (as the host's), so the gather folds that lobe's reflectance and not the compound's;
// emitter/env radiance reached directly or through specular is added to directL (XYZ, a
// monochromatic MC estimate at the sampled lambda). `vpValid` is false when the ray
// terminated (light, env, or absorption) without reaching a diffuse surface.
// `sthr` takes every spectral factor folded into `thr` (host twin sppm_render.h, SPPM-SPECTRAL).
__device__ static void dSppmVisiblePoint(const DScene& sc, DVec3 ro, DVec3 rd, Real lambda,
                                         double invPdfL, DRng& rng, int maxBounce,
                                         DHit& vp, double& thrOut, bool& vpValid,
                                         double& dX, double& dY, double& dZ, DSpecThr& sthr,
                                         const DMediumStack& stk0) {
    thrOut = 0.0; vpValid = false; dX = dY = dZ = 0.0;
    double thr = 1.0;
    DMediumStack stk = stk0;   // the camera's (DCamera::enclosure)
    bool strandArrival = false;   // CURVE-GATHER: the previous vertex, a strand, took its direct light
    bool pastDiffuse = false;     // past a strand's scattering: no glow is seen (host twin sppm_render.h)
    double contSoft = 1.0;        // what the last vertex lends what it reaches (dContSoftening)
    for (int b = 0; b <= maxBounce; ++b) {                     // the tail (host twin)
        const bool tail = (b == maxBounce);
        if (sc.hasGrin) {                                // GRADIENT-INDEX: the camera ray bends,
            DGrinMedia gm;                               // exactly as the deposit's photons do.
            gm.camera = true;
            dGrinMarch(sc, ro, rd, &gm, 200000, /*camHide=*/(b == 0));  // rng == nullptr: bend only
            // Radiance over n^2 along the bend, at every wavelength alike; past a bend the light the
            // walk meets was reached by no light sample (host twin sppm_render.h).
            thr *= gm.radScale;   // flat in wavelength: the spectral carrier is unchanged
            if (gm.bent) strandArrival = false;
            const int cm = stk.topMat();                 // Beer-Lambert over the marched arc
            if (cm >= 0 && gm.arc > 0) {
                const double arc = (double)gm.arc;
                const double T = exp(-(double)specLookup(sc.mats[cm].absorb, lambda) * arc);
                thr *= T;
                sthr.mul(sc, [&](Real L) { return exp(-(double)specLookup(sc.mats[cm].absorb, L) * arc); }, T);
            }
        }
        // b == 0 is the camera ray this visible-point walk was handed; see DMaterial::hideCamera.
        DHit h = closestHitIn(stk, sc, ro, rd, RAY_TMIN, BIG, /*camHide=*/(b == 0));
        if (h.valid) {                                   // Beer-Lambert in current medium, spectral
            const int cm = stk.topMat();
            if (cm >= 0) {
                const double tt = (double)h.t;
                const double T = exp(-(double)specLookup(sc.mats[cm].absorb, lambda) * tt);
                thr *= T;
                sthr.mul(sc, [&](Real L) { return exp(-(double)specLookup(sc.mats[cm].absorb, L) * tt); }, T);
            }
        }
        if (!h.valid) {                                  // escaped -> environment (direct)
            if (sc.envIndex >= 0) {
                // camera / specular escape, weight 1: the env as SEEN (bilinear), host sppm_render.h
                double envRad = (sc.env.scale != nullptr)
                                    ? dEnvRadianceSeen(sc.env, rd, lambda)
                                    : (double)specLookup(sc.emitters[sc.envIndex].emitSpd, lambda);
                double e = thr * envRad * invPdfL * contSoft;
                dX += (double)cieX(lambda) * e;
                dY += (double)cieY(lambda) * e;
                dZ += (double)cieZ(lambda) * e;
            }
            // Directly-viewed solar disc (camera / specular escapes only — a diffuse
            // vertex stores a hit point and returns before it can reach here; a strand's NEE
            // took it already). Host twin: sppm_render.h.
            if (sc.sunCount > 0 && !strandArrival) {
                double e = thr * dSunRadiance(sc, rd, lambda) * invPdfL * contSoft;
                dX += (double)cieX(lambda) * e;
                dY += (double)cieY(lambda) * e;
                dZ += (double)cieZ(lambda) * e;
            }
            return;
        }
        const DMaterial* mp = &sc.mats[h.matId];
        int matId = h.matId;
        if (dIsCompound(mp->type)) {
            int coatPick = -1; double coatR = 0.0, coatCos = 1.0; Real coatTh = (Real)0;
            int child = dResolveCompound(sc, *mp, h, rd, lambda, rng, &coatPick, &coatR, &coatCos, &coatTh);
            if (child < 0) return;                       // absorbed by the mix
            if (coatPick >= 0) dFoldCoatCoin(sthr, sc, *mp, coatPick, coatR, coatCos, coatTh);   // SPECGATHER
            mp = &sc.mats[child]; matId = child;
        }
        const DMaterial& m = *mp;

        // A directly-viewed / specular-seen light, as on the host (sppm_render.h's `m.isLight`):
        // an emitter's own geometry, or a surface whose material glows with no emitter
        // registered for it (a marched isosurface or CSG solid; bkRadiance falls back to the
        // material's `emit` the same way). It adds its emission from its front face only; a
        // light block's own surface then ends the path, and any other glowing material has a
        // body that is shaded below like any surface.
        int li = dEmitterForHit(sc, h, matId);
        const DSpec* eSpd = (li >= 0)        ? &sc.emitters[li].emitSpd
                           : (m.matIsLight)   ? &m.matEmit
                                              : nullptr;
        if (eSpd) {
            if (dot(rd, h.ng) < (Real)0 && !strandArrival && (li >= 0 || !pastDiffuse)) {
                double e = (double)specLookup(*eSpd, lambda) * thr * invPdfL * contSoft
                         * dEmitHitScale(sc, m.emitPat, li, h, rd * (Real)(-1));    // pattern and aim at this hit
                dX += (double)cieX(lambda) * e;
                dY += (double)cieY(lambda) * e;
                dZ += (double)cieZ(lambda) * e;
            }
            if (m.matReflectsNothing) return;
        }
        if (tail) return;                               // the light it reached; no visible point
        strandArrival = false;
        contSoft = 1.0;   // a specular vertex lends nothing; the glossy case sets its own

        // CURVE-GATHER: no visible point on a `type diffuse` strand (host twin).
        if (dGatherlessStrand(m, h)) {
            bool cont;
            const double dc = dStrandStep(sc, m, h, lambda, invPdfL, rng, thr, &sthr, ro, rd, cont);
            dX += (double)cieX(lambda) * dc; dY += (double)cieY(lambda) * dc; dZ += (double)cieZ(lambda) * dc;
            strandArrival = true;
            pastDiffuse = true;
            if (!cont) return;
            continue;
        }

        if (m.type == D_DIFFUSE || m.type == D_DIFFUSETRANSMIT || m.type == D_FLUORESCENT) {
            // under the lobe the walk resolved, whose reflectance the gather folds (host twin)
            vp = h; vp.matId = matId; thrOut = thr; vpValid = true;
            return;
        }

        const DVec3 wArr = -rd;                          // the arrival, for the coat (dCoatF)
        switch (m.type) {                                // specular walk, its factors into sthr
            case D_MIRROR: {
                const double r = (double)clamp01(dReflectSlot(sc, m, h, lambda));
                thr *= r;
                dSpecFold(sthr, sc, m, h, r, false);
                if (!dContinueAlong(h, reflectv(rd, h.n), DL_REFLECT, ro, rd)) return; break;
            }
            case D_GLOSSY: {
                if (!dGlossyCameraSide(rd, h.n)) return;        // met from below ns (host twin)
                const double r = (double)clamp01(dReflectSlot(sc, m, h, lambda));
                thr *= r;
                dSpecFold(sthr, sc, m, h, r, false);
                const DGlossyDraw gd = dDrawGlossy(sc, m, h, h.n, -rd, rng);
                const DVec3& o = gd.o;
                if (dot(o, h.n) <= 0) return;
                thr *= (double)gd.w;   // the draw's weight: the same at every wavelength, so not sthr's
                // An iridescent lobe's interference reflectance at the pair's half-vector; where it
                // is 0 at the walk's own wavelength the path carries nothing.
                if (gd.irid) {
                    const double Rh = (double)gd.R(lambda);
                    if (!(Rh > 0.0)) return;
                    thr *= Rh;
                    sthr.mul(sc, [&](double Lw) { return (double)gd.R((Real)Lw); }, Rh);
                }
                if (!dContinueAlong(h, o, DL_REFLECT, ro, rd)) return;
                contSoft = dContSoftening(m, h, rd); break;
            }
            case D_DIELECTRIC:
            case D_THINFILM:
            case D_MULTILAYER: {                         // a dielectric body, plain or coated
                DVec3 nro, nrd; double rs = 1.0;
                DStepRecord st;
                const DVec3 d0 = rd;
                if (!dDielectricStep(sc, m, h, rd, lambda, rng, stk, nro, nrd, nullptr, &rs, &st))
                    return;
                thr *= rs;                               // radiance across the interface
                sthr.mulCoin(sc, [&](Real L, double& pr) {
                    return dStepFactorAt(sc, m, h, d0, st, L, &pr);
                }, st.factor, st.prob, m.type != D_DIELECTRIC);
                ro = nro; rd = nrd; break;
            }
            case D_HALFMIRROR: {
                Real r = clamp01(dReflectSlot(sc, m, h, lambda));
                const bool refl = rng.uniform() < r;
                if (refl) { if (!dContinueAlong(h, reflectv(rd, h.n), DL_REFLECT, ro, rd)) return; }
                else      { ro = dOffsetAlong(h.p, h.ng, rd, h.pscale); }
                const double pc = refl ? (double)r : 1.0 - (double)r;
                sthr.mulCoin(sc, [&](Real L, double& pr) {
                    const double rL = (double)clamp01(dReflectSlot(sc, m, h, L));
                    return pr = refl ? rL : 1.0 - rL;
                }, pc, pc, true);
                break;
            }
            case D_FILTER: {
                const double t = (double)clamp01(dTransmitSlot(sc, m, h, lambda));
                thr *= t;
                dSpecFold(sthr, sc, m, h, t, true);
                ro = dOffsetAlong(h.p, h.ng, rd, h.pscale); break;
            }
            default: {                                   // Grating: approx reflect
                const double r = (double)clamp01(dReflectSlot(sc, m, h, lambda));
                thr *= r;
                dSpecFold(sthr, sc, m, h, r, false);
                if (!dContinueAlong(h, reflectv(rd, h.n), DL_REFLECT, ro, rd)) return; break;
            }
        }
        // A coated vertex's continuation takes the coat, at each photon's wavelength too (as mode
        // M's walk).
        if (!dCoatContinueWalk(sc, m, h, wArr, rd, lambda, thr, nullptr, nullptr, 0,
                               [&](const DCoatEval& ce, double s) {
                                   sthr.mul(sc, [&](Real Lw) { return ce.at(Lw); }, s); }))
            return;
        if (thr <= 0.0) return;
    }
}

// SPPM per-pixel state (structure-of-arrays on the device). One entry per pixel.
struct DSppmState {
    double* tau;       // npix*3  accumulated radius-rescaled flux (XYZ)
    double* radius;    // npix    current gather radius R_i
    double* nAcc;      // npix    accumulated photon count N_i
    double* directSum; // npix*3  direct/specular-viewed emitter + env, summed over passes
    DHit*   vpHit;     // npix    this pass's visible point (diffuse hit)
    double* vpThr;     // npix    specular throughput camera -> visible point
    unsigned char* vpValid; // npix
    // SPPM-SPECTRAL (host twin SPPMPixel::vpRatio): the camera prefix's spectral ratio per
    // DSpecThr bin, read where vpSpec is set.
    float*  vpRatio;   // npix*K
    unsigned char* vpSpec;  // npix
};

// Phase 1: resample each pixel's camera visible point for this pass and accumulate its
// direct term. One thread per pixel. Seeds mirror kBackward/kGather (global sample index).
#ifndef FT_FAMILY_TU
__global__ void kSppmVisiblePoint(DScene sc, const __grid_constant__ DCamera cam, DSppmState st, int resX,
                                  int resY, int maxBounce, unsigned long long seedBase, long long passIdx,
                                  int perPixel) {
    D_MOTION_CLEAR();   // this thread's time: the launch's (passU) until a path sets one
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    long long npix = (long long)resX * resY;
    for (long long pix = g; pix < npix; pix += G) {
        int px = (int)(pix % resX), py = (int)(pix / resX);
        // Distinct stream per (pixel, pass).
        unsigned long long s = (unsigned long long)(pix) * 0x9E3779B97F4A7C15ULL
                             + (unsigned long long)passIdx * 0xD1B54A32D192ED03ULL;
        DRng rng; rng.seedUnit(s, seedBase + 23);

        double pdf = 0.0;
        Real lambda = dSampleSceneLambda(sc, rng, pdf);
        st.vpValid[pix] = 0;
        st.vpSpec[pix] = 0;
        if (pdf <= 0.0) continue;   // this pixel only: the thread owns more of the grid
        double invPdfL = dInvPdfLambda(sc, lambda);

        DVec3 ro, rd;
        Real jx = rng.uniform(), jy = rng.uniform();
        // A moving camera over a still scene (`perPixel`; host twin sppmPass): each pixel's ray at
        // a time of its own, off a stream keyed by the pass and the pixel; else the pass's camera.
        DCamera posed;
        const DCamera* cp = &cam;
        if (perPixel) {
            DRng tr;
            tr.seedUnit(((unsigned long long)passIdx << 32) ^ (unsigned long long)pix,
                        seedBase ^ 0x6C62272E07BB0142ULL);
            dPoseCamera(cam, cam.shOpen + (cam.shClose - cam.shOpen) * (double)tr.uniform(), posed);
            cp = &posed;
        }
        dGenRay(*cp, px, py, jx, jy, ro, rd);            // pinhole only (lens cams gated to CPU)

        DHit vp; double thr = 0.0; bool valid = false; double dX, dY, dZ;
        DSpecThr sthr; sthr.init();
        dSppmVisiblePoint(sc, ro, rd, lambda, invPdfL, rng, maxBounce, vp, thr, valid, dX, dY, dZ, sthr,
                          cp->enclosure);
        st.vpHit[pix] = vp; st.vpThr[pix] = thr; st.vpValid[pix] = valid ? 1 : 0;
        const bool spec = valid && sthr.any;
        st.vpSpec[pix] = spec ? 1 : 0;
        if (spec) {   // in float over its peak, the peak in vpThr (host twin)
            const double pk = sthr.peak();
            for (int k = 0; k < DSpecThr::K; ++k)
                st.vpRatio[(size_t)pix * DSpecThr::K + k] = (float)(sthr.ratio(DSpecThr::lamOf(k)) / pk);
            st.vpThr[pix] = thr * pk;
        }
        st.directSum[pix * 3 + 0] += dX;
        st.directSum[pix * 3 + 1] += dY;
        st.directSum[pix * 3 + 2] += dZ;
    }
}
#endif

// Phase 3: gather each valid pixel's visible point at its current radius, then apply the
// shared-statistics progressive radius/flux update. One thread per pixel. The photon records
// carry pX/pY/pZ = cie(lambda)*power/pi (NO area/nEmitted fold — those depend on the current
// per-pixel radius and are applied at resolve), so phi? += rho(lambda_p) * p?.
#ifndef FT_FAMILY_TU
__global__ void kSppmGather(DScene sc, DPhotonMap pm, DSppmState st, int resX, int resY,
                            double alpha, unsigned long long seedBase, long long passIdx) {
    D_MOTION_CLEAR();   // this thread's time: the launch's (passU) until a path sets one
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    long long npix = (long long)resX * resY;
    for (long long pix = g; pix < npix; pix += G) {
        if (!st.vpValid[pix]) continue;
        const DHit h = st.vpHit[pix];
        const DMaterial& m = sc.mats[h.matId];           // the resolved lobe's (host twin)
        const DVec3 gn = dShadedNg(h);                  // the true normal, as the photons carry it
        double R = st.radius[pix];
        Real r2 = (Real)(R * R);
        float gx = 0.f, gy = 0.f, gz = 0.f;
        double M = 0.0;
        // the camera prefix's factors at each photon's wavelength (SPPM-SPECTRAL)
        const float* ratio = st.vpSpec[pix] ? st.vpRatio + (size_t)pix * DSpecThr::K : nullptr;
        const Real slkS = gatherBandSlack<Real>((Real)R, h.fiberRadius, dRoundScale(h));
        dPmNeighborhood(pm, h.p, [&](int k) {
            const DGatherPhoton& ph = pm.photons[k];
            DVec3 d = h.p - ph.pos;
            const Real d2 = dot(d, d);
            if (d2 > r2) return;
            bool farSide;
            if (!dOnGatherSurfaceSided(h.p, gn, ph.pos, ph.n, d2, slkS, m.type == D_DIFFUSETRANSMIT, farSide)) return;   // another surface's photon
            float rho = (float)dGatherAlbedo(sc, m, h, (Real)ph.lambda, farSide);
            if (ratio) rho *= ratio[DSpecThr::binOf((Real)ph.lambda)];
            gx += rho * ph.pX;
            gy += rho * ph.pY;
            gz += rho * ph.pZ;
            M += 1.0;
        });
        // M-GATHERAREA, the DEVICE twin of sppm_render.h's correction. Mode `S` runs on the GPU
        // by DEFAULT, so the host edit alone changed nothing a default render executes: the
        // positive control `_ga_strip` read -19.78 % with the correction off and -19.76 % with
        // it on, i.e. bit-identical, while the CPU path moved -25.4 % -> -3.3 %. Both halves,
        // or neither.
        //
        // Applied to the flux BEFORE it enters `tau`, for the reason spelled out in the host
        // twin: `kSppmResolve` divides by `pi R^2` at the FINAL radius and is correct only
        // because the `ratio2` chain telescopes each pass's contribution to `R_final^2/R_i^2`.
        // Coverage belongs to the radius actually gathered at, and SPPM shrinks R every pass.
        if (sc.gatherArea > 0) {
            // Seeded per (pixel, pass) exactly like kSppmVisiblePoint above, and NOT per thread
            // or per launch index, so the probe pattern -- and the image -- is independent of
            // the grid geometry. `+ 0x5851F42D4C957F2DULL` keeps this stream disjoint from the
            // visible-point stream that shares the same (pix, pass).
            unsigned long long s = (unsigned long long)(pix) * 0x9E3779B97F4A7C15ULL
                                 + (unsigned long long)passIdx * 0xD1B54A32D192ED03ULL
                                 + 0x5851F42D4C957F2DULL;
            DRng grng; grng.seedUnit(s, seedBase + 23);
            const double cs = dGatherAreaScale(sc.gaBias,
                dGatherCoverage(sc, h.p, gn, (Real)R, slkS, grng, sc.gatherArea, h.pscale, h.fiberRadius));
            gx = (float)((double)gx * cs);
            gy = (float)((double)gy * cs);
            gz = (float)((double)gz * cs);
        }
        // Shared-statistics PPM update (Hachisuka 2008).
        double nAcc = st.nAcc[pix];
        double Nnew = nAcc + alpha * M;
        double denom = nAcc + M;
        double ratio2 = (denom > 0.0) ? (Nnew / denom) : 1.0;   // (R'/R)^2
        double thr = st.vpThr[pix];
        st.tau[pix * 3 + 0] = (st.tau[pix * 3 + 0] + (double)gx * thr) * ratio2;
        st.tau[pix * 3 + 1] = (st.tau[pix * 3 + 1] + (double)gy * thr) * ratio2;
        st.tau[pix * 3 + 2] = (st.tau[pix * 3 + 2] + (double)gz * thr) * ratio2;
        st.radius[pix] = R * sqrt(ratio2);
        st.nAcc[pix] = Nnew;
    }
}
#endif

// Resolve the accumulated state into a film: L = directSum/passes + tau/(pi R^2 Nemit).
#ifndef FT_FAMILY_TU
__global__ void kSppmResolve(DSppmState st, double* film, int resX, int resY,
                             long long passes, double Nemit) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    long long npix = (long long)resX * resY;
    double invPasses = (passes > 0) ? 1.0 / (double)passes : 0.0;
    for (long long pix = g; pix < npix; pix += G) {
        double lx = st.directSum[pix * 3 + 0] * invPasses;
        double ly = st.directSum[pix * 3 + 1] * invPasses;
        double lz = st.directSum[pix * 3 + 2] * invPasses;
        double R = st.radius[pix];
        double area = DPI * R * R;
        if (area > 0.0 && Nemit > 0.0) {
            double inv = 1.0 / (area * Nemit);
            lx += st.tau[pix * 3 + 0] * inv;
            ly += st.tau[pix * 3 + 1] * inv;
            lz += st.tau[pix * 3 + 2] * inv;
        }
        film[pix * 3 + 0] = lx;
        film[pix * 3 + 1] = ly;
        film[pix * 3 + 2] = lz;
    }
}
#endif

// ============================ device: VCM / UPS (mode U) ============================
// Device twin of vcm.h. VCM combines BDPT vertex CONNECTIONS with photon-map vertex
// MERGING under one balance-heuristic MIS. Per pass: (1) kVcmLight traces nLight light
// subpaths (one per pixel unless the store would not fit, vcmSessionBegin), stores their
// connectible vertices into a per-path slab and splats their connect-to-camera (t=1)
// contributions; the slab is compacted in path order on the device and a uniform hash grid
// built over ALL light vertices (cell = merge radius). (2) kVcmCamera traces one camera
// subpath per pixel doing emission (s=0), NEE (s=1), vertex connection to the PAIRED light
// subpath (pixel % nLight), and merging from the grid; it accumulates the pass image into
// the persistent `accum` sum. The resolve divides by the pass count. Mirrors SmallVCM's
// dVCM/dVC/dVM bookkeeping (misArrival / misScatter inlined; Mis(x)=x, so the wrappers are
// dropped). Scope (gated in cudaVcmSupported): surfaces only, area/sphere/tube/mesh Lambertian
// lights PLUS the two delta emitters (`spot`, `sun` — since 0.127.0, with dVC starting at 0,
// the sun's planar first-edge density and its escaped-ray disc), rectilinear pinhole camera,
// NO participating media (media.empty()).

// One stored light-subpath vertex (device twin of vcm.h LightVertex). Stores INDICES + the
// per-hit texcoords (u,v) so textured/patterned/record BSDFs evaluate per-hit like M9's BDPT.
struct DVcmLV {
    DVec3  p, ns, ng, wo;
    double beta;
    double dVCM, dVC, dVM;
    float  cx, cy, cz;     // cie(lambda) cached at store time, read by every merge that gathers it
    float  lambda;
    int    matId, edges;
    Real   u, v;
    Real   curv;           // O3 mean curvature — here for the same reason u/v are: the
                           // connection BSDF is re-evaluated at this vertex, so a
                           // curvature-driven material must not read a different value
                           // in VCM than in every other integrator. Costs 8 B/slot after
                           // alignment (the slab goes 128 -> 136 B), which is the price
                           // of that consistency on the largest allocation in a session.
    // Hero wavelengths still live here (1 == de-hero'd / single-λ), the hit's DHit::back and
    // DHit::patOwnSide (DVertex's), and its DHit::extMat plus one (0: air) -- in one word, so the
    // slot does not grow. One type throughout, so no compiler splits the word.
    unsigned nUp : 5, back : 1, patOwnSide : 1, extMat1 : 25;
    DVec3  tangent;        // fiber axis at a strand vertex (its BCSDF frame for VC / splats, 0.364.0)
    Real   fiberR;         // fiber radius (far-side clearance of a connection that ends on it); 0 off a strand
    Real   pscale;         // the hit's DHit::pscale (DVertex::pscale)
    int    vcol;           // the hit's vertex colour and packed barycentrics (DVertex's)
    unsigned short vb0q, vb1q;
    float  mergeW;         // the merges' share of it (host twin LightVertex::mergeW)
};
#if FTRACE_GPU_FP32
static_assert(sizeof(DVcmLV) == 152, "DVcmLV changed size: it is the largest allocation in a session");
#endif
// dVertexGeoN's twin for a stored light vertex.
__device__ static inline DVec3 dLvGeoN(const DVcmLV& v) { return v.back ? v.ng * (Real)(-1) : v.ng; }

// The SECONDARY hero wavelengths of a stored light vertex, one slot per secondary. Kept in a
// PARALLEL slab (`lvSec[(i*vcmCap + k)*secStride + j]`, secStride == C-1) rather than inline in
// DVcmLV, exactly like GPU BDPT's `pathSec`: the light-vertex slab is `nLight * vcmCap *
// sizeof(DVcmLV)` and is by far the largest allocation in a VCM session, so a single-λ run
// (`-heroc 1`) must allocate the sec slab at zero and pay nothing. Only `beta` and `lam` are
// stored, and cie(λ) is recomputed at gather time, which halves the slot to 16 B. (The hero's
// DVcmLV::cx caches a float of it, so the two differ in the last bits of a float.)
struct DVcmSec {
    double beta;
    float  lam;
};

// The hashed lattice over the compacted light vertices (merge_grid.h; device twin of vcm.h VcmGrid).
// `lv` is the compact array in path order, which the connections read a path at a time; `byCell`
// holds the same vertices in bucket-contiguous order (bucket b's are [cellStart[b], cellStart[b + 1])),
// which the merges read a bucket at a time, and `byCellSec` their secondary-λ rows (DVcmSec,
// secStride a vertex; null without the hero bundle). The order within a bucket is the compact
// array's.
struct DVcmGrid {
    const DVcmLV* lv;       // compact light-vertex array (path order)
    int    nLV;
    const int* cellStart;   // mergeBucketCount(lat.slots) + 1
    const DVcmLV* byCell;   // nLV vertices, bucket-contiguous
    const unsigned char* tag;   // byCell[k]'s block's tag (MergeKey)
    const DVcmSec* byCellSec;
    MergeLattice lat;       // the lattice, its cells the merge radius wide (merge_grid.h)
};

// Per-pass constants (device twin of vcm.h PassCtx; media/diffraction handled elsewhere).
struct DVcmCtx {
    double radius, nLightPaths, misVcWeight, misVmWeight, vmNorm, imagePlaneDist;
    int    maxDepth;
};

// Reconstruct a minimal DVertex for the per-hit BSDF helpers (dBsdfF / dBsdfPdf reconstruct a
// DHit from it — they read p/ns/ng/matId/u/v and the hit's pattern state, DVertHitState).
__device__ static inline DVertex dVertFromHit(const DScene& sc, const DHit& h, int matId) {
    DVertex v; v.type = BV_SURFACE; v.p = h.p; v.ns = h.n; v.ng = h.ng;
    v.beta = 0; v.pdfFwd = 0; v.pdfRev = 0; v.delta = 0; v.matId = matId; v.lightIdx = -1;
    v.emitPatW = (Real)1;   // BSDF-only helper vertex; never read for emission
    v.areaScale = 1.0; v.mediumId = -1; v.u = h.u; v.v = h.v; v.curv = h.curv;
    v.vcol = h.vcol; v.vb0q = dPackBary(h.vb0); v.vb1q = dPackBary(h.vb1);
    v.extMat = h.extMat;
    v.tangent = h.tangent; v.fiberR = h.fiberRadius; v.pscale = h.pscale;
    v.back = h.back ? 1 : 0; v.patOwnSide = h.patOwnSide ? 1 : 0;
    v.hs = dHitStateOf(sc, h);   // where a connection or a merge reads its patterns
    return v;
}
// A light vertex as a DVertex for a connection's BSDF reads; `hs` its hit state (DVcmLV keeps none:
// the session's slab beside it, d_lvHsCompact, null where the scene reads no pattern state).
__device__ static inline DVertex dVertFromLV(const DVcmLV& lv, const DVertHitState* hs) {
    DVertex v; v.type = BV_SURFACE; v.p = lv.p; v.ns = lv.ns; v.ng = lv.ng;
    v.beta = 0; v.pdfFwd = 0; v.pdfRev = 0; v.delta = 0; v.matId = lv.matId; v.lightIdx = -1;
    v.emitPatW = (Real)1;   // BSDF-only helper vertex; never read for emission
    v.areaScale = 1.0; v.mediumId = -1; v.u = lv.u; v.v = lv.v; v.curv = lv.curv;
    v.vcol = lv.vcol; v.vb0q = lv.vb0q; v.vb1q = lv.vb1q;
    v.tangent = lv.tangent; v.fiberR = lv.fiberR; v.pscale = lv.pscale;
    v.back = lv.back; v.patOwnSide = lv.patOwnSide;
    v.extMat = (int)lv.extMat1 - 1;
    if (hs) v.hs = *hs;
    return v;
}

// Sample a scattering continuation at a surface vertex (device twin of vcm.h scatterSample).
// Returns wi/betaFactor/pdfW/pdfRevW/cosThetaOut/delta/terminate. Uses per-hit slots
// (dReflectSlot / dMatRoughness / dDiffuseRho) so it matches the CPU VCM exactly. Media are
// out of scope; `stk` still resolves the nested-dielectric exterior IOR (dDielectricStep).
//
// HERO BUNDLE (Wilkie 2014; the four shared policies live in hero.h). When `nUp > 1` the caller
// carries nUp wavelengths on this one ray: `lamAll` is the bundle (lamAll[0] == the hero ==
// `lambda`) and on return `secF[i]` is the ABSOLUTE throughput factor for secondary i — NOT a
// ratio to the hero's, which is undefined exactly where it matters most (a gel whose T(λ_hero)
// is 0 while a secondary is wide open). `secChromatic` says secF was filled at all; the
// λ-independent lobes leave it false and the caller reuses `betaFactor` for every λ.
// `keepBundle` marks the delta lobes that nevertheless pick their continuation WITHOUT
// consulting λ (Mirror reflects, Filter passes straight through), so they opt out of the
// caller's `if (delta) nUp = 1` collapse.
//
// NOTE the `<= 0` early terminations the scalar version did for a zero reflectance /
// transmittance are GONE from the chromatic lobes: the caller applies a max-over-live-λ test
// instead, which at nUp == 1 is exactly the old scalar test (`mxF == betaFactor`). Grating
// keeps its `r <= 0` bail because it gates an RNG consumer (gratingDiffract).
__device__ static void dVcmScatter(const DScene& sc, const DMaterial& m, const DHit& h,
                                   const DVec3& rd, Real lambda, DRng& rng, int matId,
                                   DMediumStack& stk, double* radianceScale, int diffraction,
                                   DVec3& wi, double& betaFactor, double& pdfW, double& pdfRevW,
                                   double& cosThetaOut, bool& delta, bool& terminate,
                                   const Real* lamAll = nullptr, int nUp = 1,
                                   double* secF = nullptr, bool* secChromatic = nullptr,
                                   bool* keepBundle = nullptr, DJSpecRun* jr = nullptr,
                                   double jInvPdf = 0.0) {
    // (n_from / n_to)^2 at a refraction, 1 otherwise: the camera walk multiplies its throughput
    // by it, the light walk does not (host twin: vcm.h scatterSample).
    // `jr` (the camera walk's carrier, VCM-SPEC; null on the light walk): this scatter's factor at
    // every bin over its value at lambda, as mode J's device walk folds it -- the slots through the
    // shape their own read captured (DJBase), the step through its record (host twin scatterSample).
    if (radianceScale) *radianceScale = 1.0;
    // The light walk is the one that passes no radianceScale (host twin: Transport::Flux).
    const bool importance = radianceScale == nullptr;
    DVertex vt = dVertFromHit(sc, h, matId);
    const DVec3& ns = h.n;
    DVec3 wo = normalize(rd * (Real)(-1));
    wi = DVec3(0, 0, 0); betaFactor = 0; pdfW = 0; pdfRevW = 0; cosThetaOut = 0;
    delta = false; terminate = false;
    // What wi is, for dLeavesWrongSide: a reflection unless the case says otherwise (dRandomWalk).
    DLobe lobe = DL_REFLECT;
    const int nSec = (secF && lamAll && nUp > 1) ? nUp - 1 : 0;   // secondaries to fill
    if (secChromatic) *secChromatic = false;
    if (keepBundle)   *keepBundle = false;
    DJBase jbS; jbS.mode = 0; jbS.vc = 0; jbS.pm = (Real)1;   // the slot's shape, for the carrier
    Real tpS = (Real)1;                                       // the transmit pattern's multiplier
    DStepRecord jSt;
    bool jHmRefl = false, jReflLobe = false;
    DGlossyDraw gd;   // a D_GLOSSY scatter's draw, which its JSPEC fold re-weighs too
    gd.irid = false; gd.cH = (Real)1;
    switch (m.type) {
        case D_DIFFUSE:
        case D_FLUORESCENT: {
            // Light arriving below the shading normal lights nothing (dLambertLit).
            if (importance && !dLambertLit(ddot(wo, ns))) { terminate = true; break; }
            wi = cosineHemisphere(ns, rng);
            if (dot(wi, ns) <= 0) { terminate = true; break; }
            double rho = clamp01(dDiffuseRho(sc, m, h, lambda, jr ? &jbS : nullptr));
            pdfW = dBsdfPdf(sc, vt, wo, wi, lambda);
            pdfRevW = dBsdfPdf(sc, vt, wi, wo, lambda);
            betaFactor = rho;                     // rho <= 0 is caught by the caller's max test
            if (nSec) {
                *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(dDiffuseRho(sc, m, h, lamAll[i + 1]));
            }
            break;
        }
        case D_GLOSSY: {
            gd = dDrawGlossy(sc, m, h, ns, wo, rng);
            wi = gd.o;
            if (dot(wi, ns) <= 0) { terminate = true; break; }
            // times an iridescent lobe's interference reflectance at the sampled pair's
            // half-vector: dBsdfF's r for this pair. Host twin vcm.h.
            double r = clamp01(dReflectSlot(sc, m, h, lambda, jr ? &jbS : nullptr)) * (double)gd.R(lambda);
            pdfW = dBsdfPdf(sc, vt, wo, wi, lambda);
            pdfRevW = dBsdfPdf(sc, vt, wi, wo, lambda);
            // r times the draw's weight, from either end: the lobe is reciprocal (host twin vcm.h).
            const double wG = (double)gd.w;
            betaFactor = r * wG;
            if (pdfW <= 0) terminate = true;      // r <= 0 is caught by the caller's max test
            // The glossy LOBE (its microfacet draw reads the roughness alone) carries no wavelength
            // dependence, so the whole bundle follows the sampled direction and only the
            // reflectance differs per λ.
            if (nSec) {
                *secChromatic = true;
                for (int i = 0; i < nSec; ++i) {
                    secF[i] = clamp01(dReflectSlot(sc, m, h, lamAll[i + 1])) * wG;
                    if (gd.irid) secF[i] *= gd.R(lamAll[i + 1]);
                }
            }
            break;
        }
        case D_DIFFUSETRANSMIT: {
            if (importance && !dLambertLit(ddot(wo, ns))) { terminate = true; break; }   // host twin
            double rhoR, rhoT; dDiffuseTransmitAlbedos(sc, m, h, lambda, rhoR, rhoT,
                                                       jr ? &jbS : nullptr, jr ? &tpS : nullptr);
            double tot = rhoR + rhoT;
            if (tot <= 0.0) { terminate = true; break; }
            const bool reflLobe = (rng.uniform() * tot < rhoR);
            jReflLobe = reflLobe;
            if (reflLobe) wi = cosineHemisphere(ns, rng);
            else          wi = cosineHemisphere(ns * (Real)-1, rng);
            if (!reflLobe) lobe = DL_TRANSMIT;
            pdfW = dBsdfPdf(sc, vt, wo, wi, lambda);
            pdfRevW = dBsdfPdf(sc, vt, wi, wo, lambda);
            betaFactor = tot;
            if (pdfW <= 0) terminate = true;
            // The lobe was CHOSEN by the hero's albedo split, so each secondary divides by the
            // HERO's albedo for that lobe: f_i·cos/pdf_hero = rho_i(lobe)·tot_hero/rho_hero(lobe).
            if (nSec) {
                *secChromatic = true;
                for (int i = 0; i < nSec; ++i) {
                    double rR, rT; dDiffuseTransmitAlbedos(sc, m, h, lamAll[i + 1], rR, rT);
                    double num = reflLobe ? rR   : rT;
                    double den = reflLobe ? rhoR : rhoT;
                    secF[i] = (den > 0.0) ? num * tot / den : 0.0;
                }
            }
            break;
        }
        case D_HAIR: {
            // Fiber BCSDF -- device twin of vcm.h scatterSample's Hair case. The lobes are narrow
            // but finite, so a strand is connectible on the CAMERA side (dConnectibleType); a light
            // walk stops at one (kVcmLightT) and nothing merges at one. The COVERAGE
            // pass-through (hair `opacity` < 1) returns EXACTLY -wo and is a delta lobe whose
            // direction is the same for every λ, so it is flagged like a Mirror: delta AND keepBundle.
            const DHairShade hs = dHairShadeAt(sc, m, h, lambda, wo);
            double pdfH = 0.0, fv = 0.0;
            const double u0 = rng.uniform(), u1 = rng.uniform(), u2 = rng.uniform(), u3 = rng.uniform();
            const dhair::V3 wl = dhair::sample(hs.b, hs.woLocal, u0, u1, u2, u3, pdfH, fv);
            if (!(pdfH > 0.0) || !(fv > 0.0)) { terminate = true; break; }
            wi = dhair::toWorld(hs.fr, wl); lobe = DL_FREE;
            const bool passThru = (wl.x == -hs.woLocal.x && wl.y == -hs.woLocal.y &&
                                   wl.z == -hs.woLocal.z);
            pdfW    = pdfH;
            pdfRevW = passThru ? pdfH : dBsdfPdf(sc, vt, wi, wo, lambda);
            delta   = passThru;
            if (passThru && keepBundle) *keepBundle = true;
            // f*cos/pdf collapses to the total lobe attenuation (hair.h); exactly 1 for a pass-through.
            // The camera walk's factor; a light walk stops at a strand before it matters (kVcmLightT).
            const double cosLong = dhair::safeSqrt(1.0 - dhair::sqr(dhair::clampd(wl.x, -1.0, 1.0)));
            betaFactor = clamp01(fv * cosLong / pdfH);
            // sigma_a is per-λ: each secondary re-evaluates its own fiber along the hero's direction.
            // The secondaries and, for the carrier, the fiber at lambda and at every bin, in ONE
            // loop: a second call site of the BCSDF readers tips nvcc into emitting them out of
            // line (mode J's walk, and DJBase).
            if (nSec) *secChromatic = true;
            const int nJ = (jr && !passThru) ? DJSPEC_K + 1 : 0;
            double hv[DJSPEC_K + 1];
            for (int i = 0; i < nSec + nJ; ++i) {
                const bool sec = i < nSec;
                if (sec && passThru) { secF[i] = 1.0; continue; }   // the ray missed the fiber at every λ
                const int jk = i - nSec - 1;                        // -1: lambda itself
                const Real Li = sec ? lamAll[i + 1] : (jk < 0 ? lambda : (Real)DJSpec::lamOf(jk));
                const DHairShade hsi = dHairShadeAt(sc, m, h, Li, wo);
                const double fc = (double)dHairFCos(hsi, wi);
                if (sec) secF[i] = fc / pdfH; else hv[jk + 1] = (double)clamp01((Real)(fc / pdfH));
            }
            if (nJ) jr->mulVec(hv + 1, hv[0], true);   // a fiber's weight: its zeros are holes
            break;
        }
        case D_MIRROR: {
            double r = clamp01(dReflectSlot(sc, m, h, lambda, jr ? &jbS : nullptr));
            wi = reflectv(rd, ns); betaFactor = r; delta = true;
            // The mirror direction is the same for every λ, so the bundle survives; only the
            // reflectance is per-λ (cf. Glossy, the rough version of this).
            if (nSec) {
                *keepBundle = true; *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(dReflectSlot(sc, m, h, lamAll[i + 1]));
            }
            break;
        }
        case D_DIELECTRIC:
        case D_THINFILM:
        case D_MULTILAYER: {
            DVec3 nro, nrd;
            if (!dDielectricStep(sc, m, h, rd, lambda, rng, stk, nro, nrd, nullptr,
                                 radianceScale, jr ? &jSt : nullptr)) { terminate = true; break; }
            wi = nrd; betaFactor = 1.0; delta = true; lobe = DL_FREE;   // about ng
            // -jdispall: a dispersive refraction opens the window (host twin scatterSample).
            if (jr && c_jDispAll && jSt.refracted &&
                fabs((double)specLookup(m.ior, (Real)DLMIN) - (double)specLookup(m.ior, (Real)DLMAX)) > 1e-9)
                jr->markDisp((double)lambda, jInvPdf);
            break;
        }
        case D_HALFMIRROR: {
            double r = clamp01(dReflectSlot(sc, m, h, lambda, jr ? &jbS : nullptr));
            jHmRefl = rng.uniform() < r;
            if (jHmRefl) wi = reflectv(rd, ns); else { wi = rd; lobe = DL_FREE; }
            betaFactor = 1.0; delta = true;
            break;
        }
        case D_FILTER: {
            double t = clamp01(dTransmitSlot(sc, m, h, lambda, jr ? &tpS : nullptr));
            wi = rd; betaFactor = t; delta = true; lobe = DL_FREE;   // t <= 0 -> caller's max test
            // Straight-through for every λ, so the bundle survives — and a gel filter is
            // exactly where the per-λ transmittance spread is largest, i.e. the case that
            // benefits most from NOT de-heroing, AND the case that forces the absolute
            // (rather than ratio) secF, since T(λ_hero) is legitimately 0 across most of a
            // Wratten passband.
            if (nSec) {
                *keepBundle = true; *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(dTransmitSlot(sc, m, h, lamAll[i + 1]));
            }
            break;
        }
        case D_GRATING: {
            double r = clamp01(dReflectSlot(sc, m, h, lambda, jr ? &jbS : nullptr));
            if (r <= 0) { terminate = true; break; }
            DVec3 nro, nrd; int pickG = 0;
            if (!gratingDiffract(m, h, rd, lambda, diffraction, rng, nro, nrd, nullptr,
                                 jr ? &pickG : nullptr)) { terminate = true; break; }
            wi = nrd; betaFactor = r; delta = true; lobe = DL_FREE;   // orders about ng
            // The order at every bin, r(L) / wsum(L), 0 where it is evanescent; any order but 0
            // opens the dispersive window (mode J's device fold).
            if (jr) {
                bool liveC;
                const double wsC = dGratingWsum(m, h, rd, lambda, diffraction, pickG, liveC);
                if (wsC > 0.0)
                    jr->mul([&](double L) {
                                bool live;
                                const double ws = dGratingWsum(m, h, rd, (Real)L, diffraction, pickG, live);
                                return (live && ws > 0.0) ? dJSlotAt(sc, m, jbS, L, false) / ws : 0.0; },
                            r / wsC, true);
                if (pickG != 0) jr->markDisp((double)lambda, jInvPdf);
            }
            break;
        }
        default: terminate = true; break;
    }
    // A lobe that leaves on the wrong side of the true surface ends the walk (host twin).
    if (!terminate && dLeavesWrongSide(h, wi, lobe)) terminate = true;
    if (!terminate) cosThetaOut = fabs(ddot(wi, ns));
    // VCM-SPEC: the other lobes' factors at every bin (mode J's fold; hair and the grating fold in
    // their cases).
    if (jr && !terminate)
        dJFoldLobe(*jr, sc, m, h, rd, lambda, jbS, tpS, jSt, gd, jHmRefl, jReflLobe);
}

// Phase 1: `nPaths` light subpaths. Stores connectible vertices into the per-path slab
// `lvSlab[i*vcmCap + k]` (count in `lvCount[i]`), splats connect-to-camera contributions into
// `splat` (atomic, W*H*3 XYZ), and records this path index's wavelength BUNDLE (shared with the
// camera paths it is paired with) into lamBuf/invLamBuf at stride C. `nPaths` light subpaths a
// pass -- the pixel count, unless the store would not fit (vcmSessionBegin). Device twin of vcm.h
// traceLightSubpath.
//
// Templated on the number of SECONDARY hero slots exactly like kBdptT: the scalar
// instantiation kVcmLightT<0> sizes every per-λ array at 1, so a `-heroc 1` run pays zero extra
// registers/local memory and stays bit-identical to the pre-hero kernel.
#ifndef FT_FAMILY_TU
template <int NS>
__global__ void kVcmLightT(DScene sc, DCamera cam, int diffraction, DVcmCtx ctx,
                           DVcmLV* lvSlab, DVcmSec* lvSec, int secStride, int heroC,
                           int* lvCount, double* splat,
                           Real* lamBuf, double* invLamBuf, int resX, int resY, int vcmCap,
                           long long nPaths, unsigned long long seedBase, long long passIdx,
                           DVertHitState* lvHs) {
    D_MOTION_CLEAR();   // this thread's time: the launch's (passU) until a path sets one
    constexpr int SECN = (NS > 0) ? NS : 1;
    const int C = (NS > 0) ? heroC : 1;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    for (long long i = g; i < nPaths; i += G) {
        lvCount[i] = 0;
        unsigned long long s = (unsigned long long)i * 0x9E3779B97F4A7C15ULL
                             + (unsigned long long)passIdx * 0xD1B54A32D192ED03ULL;
        DRng rng; rng.seedUnit(s, seedBase + 55);

        // Hero + C-1 stratified secondaries from ONE base uniform (u + k/C wrapped into [0,1)),
        // pushed through the same inverse CDF as the scalar draw — so C == 1 consumes the
        // identical single rng variate the pre-hero kernel did.
        Real lamAll[SECN + 1];
        double invAll[SECN + 1];
        {
            double pdfL = 0.0;
            if (C > 1) {
                double u = (double)rng.uniform();
                lamAll[0] = dSampleSceneLambdaU(sc, u, pdfL);
                invAll[0] = (pdfL > 0.0) ? dInvPdfLambda(sc, lamAll[0]) : 0.0;
                for (int k = 1; k < C; ++k) {
                    double uu = u + (double)k / C;
                    if (uu >= 1.0) uu -= 1.0;
                    double pdfK = 0.0;
                    lamAll[k] = dSampleSceneLambdaU(sc, uu, pdfK);
                    invAll[k] = (pdfK > 0.0) ? dInvPdfLambda(sc, lamAll[k]) : 0.0;
                }
            } else {
                lamAll[0] = dSampleSceneLambda(sc, rng, pdfL);
                invAll[0] = (pdfL > 0.0) ? dInvPdfLambda(sc, lamAll[0]) : 0.0;
            }
        }
        const Real lambda = lamAll[0];
        const double invPdfLambda = invAll[0];
        for (int k = 0; k < C; ++k) { lamBuf[i * C + k] = lamAll[k]; invLamBuf[i * C + k] = invAll[k]; }
        if (invPdfLambda <= 0.0) continue;
        if (sc.nEmitters == 0 || sc.totalPower <= 0.0) continue;
        int nUp = C;                                  // wavelengths still riding this ray

        int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
        const DEmitter& em = sc.emitters[ei];
        if (em.shape == 3 || em.collimated) continue;      // env / collimated: out of VCM scope
        const bool isDelta    = dIsDeltaEmitter(em);       // spot / sun: no s=0 strategy
        const bool isInfinite = dIsInfiniteEmitter(em);    // sun: planar first-edge density
        Real u1 = rng.uniform(), u2 = rng.uniform();
        DVec3 y, nOut, dir;
        // Per-shape emission sampling, the device twin of `vcm.h`'s traceLightSubpath.
        // `emitScale` is the achromatic factor that multiplies the emitted SPECTRUM only and
        // never a density (the `emit pattern:` weight for an area light, the smoothstep
        // penumbra for a spot, 1 for a sun); `emitCos` is |cos| at the light, which is
        // exactly 1 for both delta lights because their emission "normal" IS the direction.
        // Keeping emitScale out of every pdf is what lets directPdf / emissionPdfW stay
        // exactly what the camera kernel's s=0 / s=1 terms assume.
        // `emitLeCos` is Le(dir) * cos / Le, what the throughput carries: emitCos, except for an
        // aimed quad, whose emitCos is its face's and emitLeCos its aimCos (host twin vcm.h).
        double emitScale = 1.0, emitCos = 1.0, emitLeCos = 1.0, pdfPos = 1.0, pdfDirW = 0.0;
        if (isDelta) {
            // Both delta lights emit uniformly inside a cone about beamDir (pdfW = 1/Omega).
            // NOTE spotOmega is the FALLOFF-weighted solid angle for a spot, so the sampling
            // cone has to be recomputed here (the two coincide only for a sun).
            double coneSolid = 2.0 * DPI * (1.0 - em.spotCosOuter);
            if (coneSolid <= 0.0) continue;
            pdfDirW = 1.0 / coneSolid;
            dir = dSunSampleCone(em, em.beamDir, (double)u1, (double)u2);
            if (isInfinite) {
                // Origin on the sun's disc upstream of the scene: a PLANAR density (host twin
                // vcm.h, reasoning bdpt.h sampleSunOrigin).
                if (sc.sceneRadius <= 0.0) continue;
                double x, yy;
                y = dSampleSunOrigin(sc, em, rng, x, yy);
                pdfPos = dSunOriginPdf(sc, em, x, yy) / (double)ddot(dir, em.beamDir);   // across the ray
            } else {
                // Delta position (pdfPos == 1): where the spot is at the pass's time, the cone
                // drawn about its beam at the instant and turned with it (host twin vcm.h).
                y = em.origin;
                emitScale = spotFalloff(ddot(dir, em.beamDir), em.spotCosInner, em.spotCosOuter);
                if (emitScale <= 0.0) continue;
                dLightCarry(sc, em, y, dir);
            }
            nOut = dir;
        } else {
            // The sampled point's `emit pattern:` factor (1.0, and a bit-identical draw, when
            // unpatterned) scales the radiance the light subpath starts with — the MIS pdfs
            // are untouched, exactly as in dGenLightSubpath.
            double areaScale = 1.0;                  // its area at the pass's time about the point
            emitScale = dEmitterSamplePoint(sc, em, u1, u2, y, nOut, &areaScale, &rng);
            if (em.area <= 0.0) continue;
            pdfPos = 1.0 / (em.area * areaScale);    // over its area now
            if (em.normalTilted) {                   // an aimed quad: intensity-proportional
                dir = dSampleAimDirNow(sc, em, rng, pdfDirW);
                nOut = dAimFaceNow(sc, em);          // the face, for the offset and the measure
                emitCos = ddot(nOut, dir);
                emitLeCos = (double)dAimCosNow(sc, em, dir);
                if (!(pdfDirW > 0.0) || emitCos <= 0.0 || emitLeCos <= 0.0) continue;
            } else {
                dir = cosineHemisphere(nOut, rng);
                emitCos = ddot(nOut, dir);
                if (emitCos <= 0.0) continue;
                pdfDirW = emitCos / DPI;
                emitLeCos = emitCos;
            }
        }
        double Le = (double)specLookup(em.emitSpd, lambda) * invPdfLambda * emitScale;
        // Max-over-live-λ: the hero can legitimately sit in a gap of the emission spectrum
        // while a secondary is on it. nUp == 1 -> empty loop -> mxLe == Le, the old test.
        double LeSec[SECN], mxLe = Le;
        for (int k = 0; k + 1 < nUp; ++k) {
            LeSec[k] = (double)specLookup(em.emitSpd, lamAll[k + 1]) * invAll[k + 1] * emitScale;
            if (LeSec[k] > mxLe) mxLe = LeSec[k];
        }
        if (!(mxLe > 0.0)) continue;
        double pdfChoice = em.power / sc.totalPower;
        if (pdfPos <= 0.0 || pdfDirW <= 0.0 || pdfChoice <= 0.0) continue;

        double emissionPdfW = pdfPos * pdfDirW * pdfChoice;
        if (emissionPdfW <= 0.0) continue;
        // The density NEE (s=1) would have used for THIS sample, in the measure that strategy
        // works in: an area density for a finite surface light, the solid-angle 1/Omega for
        // the infinitely distant sun, and 1 for a delta position (SmallVCM's PointLight::Emit
        // reports oDirectPdfA = 1). dVCM then comes out as Omega for a spot and pi R^2 for a
        // sun — exactly SmallVCM's PointLight / DirectionalLight constants.
        double directPdfW = pdfChoice * (isInfinite ? pdfDirW : (isDelta ? 1.0 : pdfPos));

        double beta = Le * emitLeCos / emissionPdfW;
        // MIS bookkeeping is the HERO's for the whole bundle: every sampling density in this
        // renderer (cosine / glossy-lobe / emitter pdfs) is wavelength-INDEPENDENT, so one
        // dVCM/dVC/dVM triple serves every λ — only the throughput VALUES differ.
        // A delta light cannot be hit by BSDF sampling, so dVC (and the dVM built from it)
        // start at ZERO: that is what DROPS the unsamplable strategies from the balance
        // heuristic instead of under-weighting the ones that do work.
        double dVCM = directPdfW / emissionPdfW;
        double dVC  = isDelta ? 0.0 : (emitCos / emissionPdfW);
        double dVM  = dVC * ctx.misVcWeight;

        double betaSec[SECN];
        double cieSx[SECN], cieSy[SECN], cieSz[SECN];
        for (int k = 0; k + 1 < nUp; ++k) {
            betaSec[k] = LeSec[k] * emitLeCos / emissionPdfW;
            cieSx[k] = (double)cieX(lamAll[k + 1]);
            cieSy[k] = (double)cieY(lamAll[k + 1]);
            cieSz[k] = (double)cieZ(lamAll[k + 1]);
        }
        double cieLx = (double)cieX(lambda), cieLy = (double)cieY(lambda), cieLz = (double)cieZ(lambda);
        DVec3 prevP = y;
        DVec3 ro = dOffsetAlong(y, isDelta ? DVec3{(Real)0, (Real)0, (Real)0} : nOut, dir);
        DVec3 rd = dir;
        DMediumStack stk; dBirthStack(sc, ei, -1, ro, stk);   // born inside its enclosing solids
        int stored = 0;
        Real curveTmin = (Real)0;   // far-side fiber clearance for the NEXT hop (see the continuation)
        // A light vertex reached over more than maxDepth - 1 edges joins no path short enough
        // (host twin: traceLightSubpath).
        const int lastEdge = ctx.maxDepth - 1;

        for (int edges = 1; edges <= lastEdge; ++edges) {
            DHit h = closestHitIn(stk, sc, ro, rd, (curveTmin > (Real)0) ? (Real)0 : RAY_TMIN, BIG, false, curveTmin);
            if (!h.valid) break;                          // escaped (no env in scope)
            {                                             // colored-glass Beer-Lambert (delta chains)
                int cm = stk.topMat();
                double a = (cm >= 0) ? (double)specLookup(sc.mats[cm].absorb, lambda) : 0.0;
                if (a > 0.0) beta *= exp(-a * (double)h.t);
                // Per-λ absorption: this IS the colour of coloured glass, so it must not be
                // evaluated at the hero alone.
                for (int k = 0; k + 1 < nUp; ++k) {
                    double ak = (cm >= 0) ? (double)specLookup(sc.mats[cm].absorb, lamAll[k + 1]) : 0.0;
                    if (ak > 0.0) betaSec[k] *= exp(-ak * (double)h.t);
                }
            }
            double dist = h.t;
            DVec3 rdCur = rd;
            double cosThetaIn = fabs(ddot(h.n, rdCur * (Real)-1));
            if (cosThetaIn <= 1e-9) break;

            const DMaterial* mp = &sc.mats[h.matId];
            int matId = h.matId;
            double storeP = 0.0;      // COMPOUND-MERGE-PICK (DVcmLV::mergeW)
            if (dIsCompound(mp->type)) {
                int coatPick = -1; double coatR = 0.0, coatCos = 1.0, coatU = 0.0; Real coatTh = (Real)0;
                int c = dResolveCompound(sc, *mp, h, rdCur, lamAll[0], rng,
                                         &coatPick, &coatR, &coatCos, &coatTh, &coatU);
                storeP = dCompoundMergeStoreProb(sc, *mp, coatR);
                if (c < 0) break;
                // One coat coin for the bundle, compared per wavelength; disagreeing wavelengths
                // collapse it to the hero here (vcm.h, bdpt.h HERO-COAT-COIN).
                if (coatPick >= 0)
                    for (int k = 0; nUp > 1 && k + 1 < nUp; ++k)
                        if ((coatU < dCoatRAtJ(*mp, coatCos, coatTh, lamAll[k + 1])) != (coatPick == 1))
                            nUp = 1;
                mp = &sc.mats[c]; matId = c;
            }
            // A light block's own surface absorbs; any other glowing material also reflects, and
            // the subpath scatters off it as off any surface (vcm.h, bdpt.h's walk).
            if (mp->matReflectsNothing) break;

            // misArrival(dist, cosThetaIn, foldDist2 = !(isInfinite && edges == 1)). The one
            // edge in the whole renderer that skips the dist^2 area conversion is the FIRST
            // edge of a sun's subpath: that vertex's forward density is the planar 1/(pi R^2)
            // over the fictitious disc, not a solid-angle density (SmallVCM's
            // `if (pathLength > 1 || isFiniteLight) dVCM *= dist^2`; pbrt's "correct subpath
            // sampling densities for infinite area lights" patch).
            if (!(isInfinite && edges == 1)) dVCM *= dist * dist;
            dVCM /= cosThetaIn; dVC /= cosThetaIn; dVM /= cosThetaIn;

            DVec3 wo = normalize(prevP - h.p);
            DVec3 ngo = dShadedNg(h);

            // A strand is never a light vertex (vcm.h traceLightSubpath, 0.365.0): neither stored
            // nor splat, and the walk ends after its scatter below. A compound's merge vertex is
            // stored whatever lobe the pick took (storeP), for the merges alone where that lobe
            // does not connect (host twin).
            const bool connectible = dLvConnects(mp->type);
            // The first bounce softens toward the emitter it came from, in the throughput every
            // later vertex inherits (host twin vcm.h; dEmitterSoftening).
            const double sL = (edges == 1 && connectible) ? (double)dShadowTerminatorG(wo, h.n, ngo) : 1.0;
            if (edges == 1 && connectible) {   // a translucent one too, on its arrival side
                beta *= sL;
                for (int k = 0; k + 1 < nUp; ++k) betaSec[k] *= sL;
            }
            // ...and a compound's merge copy: its gathering lobes' softening in its merge weight,
            // over the vertex's own, sL (host twin vcm.h, dCompoundMergeSoftening).
            const double mergeSoft = (edges == 1 && storeP > 0.0)
                ? dCompoundMergeSoftening(wo, h.n, ngo, sL) : 1.0;
            if (connectible || storeP > 0.0) {
                if (stored < vcmCap) {
                    DVcmLV lv;
                    lv.p = h.p; lv.ns = h.n; lv.ng = h.ng; lv.wo = wo;
                    lv.beta = beta; lv.lambda = (float)lambda;
                    lv.cx = (float)cieLx; lv.cy = (float)cieLy; lv.cz = (float)cieLz;
                    lv.dVCM = dVCM; lv.dVC = dVC; lv.dVM = dVM;
                    lv.matId = matId; lv.edges = edges; lv.u = h.u; lv.v = h.v; lv.curv = h.curv;
                    lv.vcol = h.vcol; lv.vb0q = dPackBary(h.vb0); lv.vb1q = dPackBary(h.vb1);
                    lv.extMat1 = (unsigned)(h.extMat + 1);
                    lv.mergeW = (float)(((storeP > 0.0) ? 1.0 / storeP : 1.0) * mergeSoft);
                    lv.tangent = h.tangent; lv.fiberR = h.fiberRadius; lv.pscale = h.pscale;
                    lv.back = h.back ? 1u : 0u; lv.patOwnSide = h.patOwnSide ? 1u : 0u;
                    lv.nUp = (unsigned)nUp;
                    lvSlab[i * vcmCap + stored] = lv;
                    if (lvHs) lvHs[i * vcmCap + stored] = dHitStateOf(sc, h);   // a connection's patterns
                    if (NS > 0 && lvSec) {
                        DVcmSec* row = lvSec + (size_t)(i * vcmCap + stored) * secStride;
                        for (int k = 0; k + 1 < nUp; ++k) { row[k].beta = betaSec[k]; row[k].lam = (float)lamAll[k + 1]; }
                    }
                    stored++;
                }
                // Connect this vertex to the pinhole camera (t=1 light-image splat).
                if (connectible && !cam.hasLens && edges + 1 <= ctx.maxDepth) {
                    DVec3 toCam = cam.eye - h.p;
                    double dist2c = ddot(toCam, toCam);
                    if (dist2c > 1e-12) {
                        double distc = sqrt(dist2c);
                        DVec3 wcam = toCam * (Real)(1.0 / distc);
                        double cosToCamera = ddot(h.n, wcam);
                        bool twoSided = dTwoSidedType(mp->type);
                        // The geometric front, either side of ns, unsoftened (host twin vcm.h).
                        bool sideOk = (cosToCamera != 0.0) && (twoSided || ddot(ngo, wcam) > 0.0);
                        double cosAtCamera = ddot(cam.w, wcam * (Real)-1);
                        if (sideOk && cosAtCamera > 1e-9) {
                            int px, py; Real cc, d2c;
                            if (cam.project(h.p, px, py, cc, d2c)) {
                                // Shadow ray first, BSDF eval after (bit-identical: no RNG
                                // in either; skips the eval for occluded splats).
                                                                DVec3 oo = dFiberConnOrigin(h, mp->type == D_HAIR, wcam);
                                if (!occludedTo(sc, oo, h.p + wcam * (Real)distc, 2e-6, RAY_TMIN, /*camLeg=*/true,
                                                dFiberStep(h, mp->type == D_HAIR, wcam))) {  // camera leg: hide_camera applies (Scene::occluded / DMaterial::hideCamera)
                                    DVertex vt = dVertFromHit(sc, h, matId);
                                    // The adjoint correction and shadow-terminator G are purely
                                    // geometric, so they scale every λ the same way.
                                    double geo = (double)dShadingAdjointCorr(wo, wcam, h.n, ngo);
                                    // ...times a coated vertex's coat toward the camera (dCoatF).
                                    const DCoatEval ce = dCoatEval(sc, *mp, h, wo, wcam);
                                    double f = dBsdfFAdjoint(sc, vt, wo, wcam, lambda) * geo * ce.at(lambda);
                                    double fSec[SECN], mxf = f;
                                    for (int k = 0; k + 1 < nUp; ++k) {
                                        fSec[k] = dBsdfFAdjoint(sc, vt, wo, wcam,
                                                                lamAll[k + 1]) * geo * ce.at(lamAll[k + 1]);
                                        if (fSec[k] > mxf) mxf = fSec[k];
                                    }
                                    if (mxf > 0.0) {
                                        double bsdfRevPdfW = dBsdfPdf(sc, vt, wcam, wo, lambda);
                                        double imgPtDist = ctx.imagePlaneDist / cosAtCamera;
                                        double imgToSolid = imgPtDist * imgPtDist / cosAtCamera;
                                        double imgToSurf = imgToSolid * fabs(cosToCamera) / dist2c;
                                        double wLight = (imgToSurf / ctx.nLightPaths) *
                                                        (mp->type == D_HAIR ? 0.0   // no light-side alternative through a strand
                                                         : ((dVcmMergeAt(*mp, h.fiberRadius) ? ctx.misVmWeight : 0.0) +
                                                            dVCM + dVC * bsdfRevPdfW));
                                        double misW = 1.0 / (wLight + 1.0);
                                        double ax = 0, ay = 0, az = 0;
                                        const int cmL = cam.enclosure.topMat();   // the leg in the camera's solid
                                        double contrib = misW * beta * f * imgToSurf / ctx.nLightPaths *
                                                         dLegAbsorption(sc, cmL, distc, lamAll[0]);
                                        if (contrib > 0.0) {
                                            ax = cieLx * contrib; ay = cieLy * contrib; az = cieLz * contrib;
                                        }
                                        for (int k = 0; k + 1 < nUp; ++k) {
                                            double cs = misW * betaSec[k] * fSec[k] * imgToSurf / ctx.nLightPaths *
                                                        dLegAbsorption(sc, cmL, distc, lamAll[k + 1]);
                                            if (cs > 0.0) { ax += cieSx[k] * cs; ay += cieSy[k] * cs; az += cieSz[k] * cs; }
                                        }
                                        // The C wavelengths are C samples of ONE spectral
                                        // estimate, so the bundle averages (see hero.h).
                                        if (nUp > 1) { double inv = 1.0 / nUp; ax *= inv; ay *= inv; az *= inv; }
                                        if (ax != 0.0 || ay != 0.0 || az != 0.0) {
                                            size_t o2 = ((size_t)py * resX + px) * 3;
                                            atomicAdd(&splat[o2 + 0], ax);
                                            atomicAdd(&splat[o2 + 1], ay);
                                            atomicAdd(&splat[o2 + 2], az);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (edges == lastEdge) break;

            DVec3 wi; double betaFactor, pdfW, pdfRevW, cosThetaOut; bool delta, terminate;
            double secF[SECN]; bool secChromatic = false, keepBundle = false;
            dVcmScatter(sc, *mp, h, rdCur, lambda, rng, matId, stk, /*flux*/ nullptr, diffraction,
                        wi, betaFactor, pdfW, pdfRevW, cosThetaOut, delta, terminate,
                        lamAll, nUp, secF, &secChromatic, &keepBundle);
            // Kill the walk only when EVERY live λ is dead (nUp == 1 -> mxF == betaFactor).
            double mxF = betaFactor;
            if (secChromatic) for (int k = 0; k + 1 < nUp; ++k) if (secF[k] > mxF) mxF = secF[k];
            if (terminate || !(mxF > 0.0)) break;
            if (!delta && (pdfW <= 0.0 || cosThetaOut <= 0.0)) break;

            // misScatter(delta, cosThetaOut, pdfW, pdfRevW)
            if (delta) { dVCM = 0.0; dVC *= cosThetaOut; dVM *= cosThetaOut; }
            else {
                double t = cosThetaOut / pdfW;
                // No merge AT a strand (vcm.h misScatter `mergeable`): the merge-at-this-vertex
                // alternative leaves both accumulators (dVM is dVC in units of 1/etaVCM).
                const bool mergeable = dVcmMergeAt(*mp, h.fiberRadius);
                dVC = t * (dVC * pdfRevW + dVCM + (mergeable ? ctx.misVmWeight : 0.0));
                dVM = t * (dVM * pdfRevW + dVCM * ctx.misVcWeight + (mergeable ? 1.0 : 0.0));
                dVCM = 1.0 / pdfW;
            }
            // LIGHT SUBPATHS STOP AT STRANDS (bdpt.h randomWalk, 0.365.0); a coverage pass-through is
            // a delta continuation and rides on.
            if (mp->type == D_HAIR && !delta) break;
            beta *= betaFactor;
            for (int k = 0; k + 1 < nUp; ++k) betaSec[k] *= secChromatic ? secF[k] : betaFactor;
            // a coated vertex's coat on the way out (dCoatF; host twin vcm.h)
            if (!dCoatContinueWalk(sc, *mp, h, wo, normalize(wi), lambda, beta, betaSec, lamAll + 1, nUp - 1,
                                   [](const DCoatEval&, double) {}))
                break;
            if (!delta) {
                double adj = (double)dShadingAdjointCorr(wo, normalize(wi), h.n, ngo);
                beta *= adj;
                for (int k = 0; k + 1 < nUp; ++k) betaSec[k] *= adj;
            }
            // De-hero at a λ-DEPENDENT direction change (dielectric / thin-film / multilayer /
            // grating / half-mirror). Mirror and Filter set keepBundle: their outgoing direction
            // does not consult λ at all, so the secondaries ride on.
            if (delta && !keepBundle) nUp = 1;
            prevP = h.p;
            // A fiber scattering out its far side steps clear of its own tube (vcm.h fiberStep): the
            // clearance is closestHit's curve-only tmin and the origin an ULP nudge along the ray --
            // RAY_EPS is 1e-4 in FP32, a whole strand, and would skip the neighbour too.
            curveTmin = dFiberStep(h, mp->type == D_HAIR, wi);
            rd = normalize(wi);
            ro = (curveTmin > (Real)0) ? dOffsetPoint(h.p, rd, h.pscale)
               : delta ? dOffsetSpecular(h.p, h.ng, rd, h.pscale)                      // its line kept (host twin vcm.h)
                       : dOffsetAlong(h.p, h.ng, wi, h.pscale);
        }
        lvCount[i] = stored;
    }
}
#endif

// MODE U'S MERGES, QUEUED (VCM-WAVEFRONT). A merge weighs every light vertex within the radius,
// and a warp's camera paths reach their merge sites at different depths, so a thread merging at its
// own site keeps few lanes busy: at a large radius, a site some hundreds of vertices, the merges
// ran at single-lane speed. kVcmCameraT therefore queues the site (DVcmMergeReq), and after the
// camera wave kVcmMergeGather gathers it with a whole warp, its vertices across the lanes, and adds
// it to the accumulator by atomics: the same terms as the inline merge's, summed in another order.
// The camera kernel merges where it is a site with few light vertices (inlineMax), and one the full
// queue turns away, which it counts (`overflow`); a site with none it skips.
struct DVcmMergeReq {
    DVertex vt;            // the camera vertex merged at (dVertFromHit), its lobe's (DMergeLobe)
    DVec3   wo;            // toward the camera walk's previous vertex
    double  dVCM, dVM;     // the camera walk's MIS partials on arrival there
    double  f;             // the site's scale: its throughput (the VCM-SPEC carrier's) * vmNorm
    long long pix;         // the accumulator offset (pixel * 3) the merge adds to
    unsigned long long gaSeed;   // its footprint probe's stream (dSurfFootprint)
    int     edges;         // the camera walk's edges to it
    int     flags;         // 1: a VCM-SPEC carrier in the queue's `spec`
    Real    coatTh;        // its coat's film thickness, read at the walk's hit (dCoatThicknessAt)
};
struct DVcmMergeQueue {
    DVcmMergeReq* reqs; int* nReqs; int cap;   // null reqs: merge inline
    DJSpec* spec;          // per slot when the carrier is on (c_jSpecOn), else null
    int*    overflow;      // sites turned away and merged inline: the queue full, or no `spec` for a carrier
    // The sites that asked for the queue, a count per kVcmProfBlock pixels over the pass, by which
    // vcmSessionPass sizes the next pass's waves.
    int*    profile;
    // A site whose 27 cells hold at most this many light vertices is merged where it was found: a
    // warp would idle on it, and its request cost more than its merge.
    int     inlineMax;
};
// The pixels a count of DVcmMergeQueue::profile covers.
constexpr int kVcmProfBlock = 256;

// A merge site's place in the pass's grid (DVcmGrid): the cell holding camera vertex `p`, as the build
// bins a light vertex's.
namespace {
struct DVcmSite {
    DGridCell g;
    __device__ DVcmSite(const DVcmGrid& grid, const DVec3& p)
        : g(p, grid.lat) {}
    // The light vertices [b, e) of its neighbour cell n's bucket and the cell's tag t (dSiteListWalk's
    // `cell`).
    __device__ void cell(const DVcmGrid& grid, int n, int& b, int& e, unsigned char& t) const {
        const MergeKey c = g.neighbour(n, grid.lat.slots);
        b = grid.cellStart[c.bucket]; e = grid.cellStart[c.bucket + 1]; t = c.tag;
    }
    // The light vertices of its 27 cells, theirs alone (their tags) -- what a merge there weighs --
    // counted no further than `upTo`: the camera kernel asks only whether there are any, and whether
    // more than the queue's inline bound (DVcmMergeQueue::inlineMax).
    __device__ int count(const DVcmGrid& grid, int upTo) const {
        int n = 0;
        for (int c = 0; c < 27; ++c) {
            int b, e; unsigned char t;
            cell(grid, c, b, e, t);
            for (int k = b; k < e; ++k)
                if (grid.tag[k] == t && ++n >= upTo) return n;
        }
        return n;
    }
};
}  // namespace

// Vertex merging, strategy (d), at camera vertex `vt` (host twin vcm.h), whose cell in the grid is
// `site`: every light vertex of the pass within the radius, its XYZ weighted and added to mx, my,
// mz -- before the site's scale and its footprint's (fp, dVertexMergeFootprint's at the pass's
// radius). `wo` points back along the camera walk, which reached `vt` over `edges` edges with MIS
// partials dVCM and dVM; `coatTh` is its coat's film there (dCoatThicknessAt at the walk's hit);
// `js` is the VCM-SPEC carrier arriving there, null when off. The site's vertices are one list: the
// inline merge (WARP false) takes all of it on its own thread (dSiteListWalk), a warp's lanes (`lane`,
// its 64 ints of shared memory `warpBuf`) take it together (dSiteListWarp), each weighing the
// vertices the warp's cheap tests passed; WARP a template parameter, so each caller's copy holds its
// own walk alone. Forced inline, into the gather (whose copy ran 14 % faster inlined) and into the camera
// kernel's out-of-line dVcmMergeSumInline.
template <int NS, bool WARP>
__device__ __forceinline__ static void dVcmMergeSum(const DScene& sc, const DVcmCtx& ctx, const DVcmGrid& grid,
                                    const DVcmSite& site, int secStride, const DVertex& vt, Real coatTh,
                                    const DMergeFootprint& fp, const DVec3& wo, double dVCM, double dVM,
                                    int edges, const DJSpec* js, int lane, int* warpBuf,
                                    double& mx, double& my, double& mz) {
    constexpr int SECN = (NS > 0) ? NS : 1;
    const DMaterial& m = sc.mats[vt.matId];
    const DVec3 ngoCam = fp.n;
    const double r2 = ctx.radius * ctx.radius;
    // Its own lobe, prepared now: a site here holds vertices (the camera kernel counts its cells' own
    // first), and a warp's lanes then read the hit once, together, rather than each at its own first
    // candidate. Never a caller's lobe by reference: a lobe whose address leaves its kernel for a
    // call ptxas keeps out of line lives in local memory there, every query of it a load.
    DMergeLobe lobe(sc, vt, wo, true);
    // Entry k's refusals before its lobe: another block's (its cell's tag t), or one the merge does not
    // weigh (host twin vcm.h); k if it passes, else -1.
    const auto pass = [&](int k, unsigned char t) -> int {
        if (grid.tag[k] != t) return -1;   // another block's
        const DVcmLV& lv = grid.byCell[k];
        // Never merge with a vertex stored ON a strand (vcm.h, 0.362.0): the merge evaluates the
        // CAMERA BSDF against a photon and never looks at where the photon sits, so a strand lying
        // on the skin would bleed into it.
        if (lv.fiberR > (Real)0) return -1;
        const DVec3 d = vt.p - lv.p;
        if (ddot(d, d) > r2) return -1;
        if (edges + lv.edges > ctx.maxDepth) return -1;
        if (dMergeFromBehind(fp.twoSided, lv.wo, ngoCam)) return -1;   // host twin mergeFromBehind
        // its surface's true normal on its light's side (the hit's, host twin)
        return fp.keeps(lv.p, lv.back ? lv.ng * (Real)-1 : lv.ng, (Real)ddot(d, d)) ? k : -1;
    };
    // Entry k, passed, weighed into mx, my, mz.
    const auto weigh = [&](int k) {
        const DVcmLV& lv = grid.byCell[k];
        const DVcmSec* row = ((NS > 0) && grid.byCellSec) ? (grid.byCellSec + (size_t)k * secStride) : nullptr;
        const DVec3 wMerge = lv.wo;
        const Real lam = (Real)lv.lambda;
        // A MERGE crosses paths, so the two ends carry DIFFERENT bundles and there is no shared λ
        // set. The estimate therefore stays keyed on the LIGHT vertex's own wavelengths (the
        // spectral photon-mapping approximation, generalised from 1 λ to lv.nUp): sum over its
        // live λ, divide by lv.nUp, and let the camera's HERO throughput/MIS weight scale the
        // whole gather.
        const int lvUp = (NS > 0) ? lv.nUp : 1;
        // ...times a coated body's coat on the light's way in: the light vertex was stored before
        // its own pick.
        const DCoatEval ce = dCoatEvalAt(sc, m, vt.ns, coatTh, wo, wMerge);
        lobe.at(wMerge);
        double fCam = lobe.f(lam) * ce.at(lam);
        // The secondaries' loops run to SECN, a constant, so fSec stays in registers.
        double fSec[SECN], mxF = fCam;
#pragma unroll
        for (int q = 0; q < SECN; ++q) {
            if (q + 1 >= lvUp) break;
            fSec[q] = lobe.f((Real)row[q].lam) * ce.at((Real)row[q].lam);
            if (fSec[q] > mxF) mxF = fSec[q];
        }
        if (!(mxF > 0.0)) return;   // == `fCam <= 0` at lvUp == 1
        const double gcorr = dArrivalAdjointCorr(wMerge, vt.ns, ngoCam);
        fCam *= gcorr;
        const double camDirPdfW = lobe.pdf(false, lam);
        const double camRevPdfW = lobe.pdf(true, lam);
        const double wLight = lv.dVCM * ctx.misVcWeight + lv.dVM * camDirPdfW;
        const double wCamera = dVCM * ctx.misVcWeight + dVM * camRevPdfW;
        const double misW = 1.0 / (wLight + 1.0 + wCamera);
        // The camera's factors at each member's wavelength, or past a dispersive event the window
        // around lambda coloured there (host twin).
        const bool win = js && js->disp;
        double wgt = misW * fCam * lv.beta;
        if (js) wgt *= win ? js->winW((double)lam) : js->at((double)lam);
        double ax = lv.cx * wgt, ay = lv.cy * wgt, az = lv.cz * wgt;
        if (win) ax = ay = az = 0.0;
        double wWin = win ? wgt : 0.0;
#pragma unroll
        for (int q = 0; q < SECN; ++q) {
            if (q + 1 >= lvUp) break;
            double wq = misW * fSec[q] * gcorr * row[q].beta;
            const Real lq = (Real)row[q].lam;
            if (js) wq *= win ? js->winW((double)lq) : js->at((double)lq);
            if (win) { wWin += wq; continue; }
            ax += (double)cieX(lq) * wq;
            ay += (double)cieY(lq) * wq;
            az += (double)cieZ(lq) * wq;
        }
        if (win) {
            const Real lc = (Real)js->lamC;
            const double wc = wWin * js->atC();
            ax = (double)cieX(lc) * wc; ay = (double)cieY(lc) * wc; az = (double)cieZ(lc) * wc;
        }
        if (lvUp > 1) { const double inv = 1.0 / lvUp; ax *= inv; ay *= inv; az *= inv; }
        const double mw = (double)lv.mergeW;   // a compound's (COMPOUND-MERGE-PICK)
        mx += ax * mw; my += ay * mw; mz += az * mw;
    };
    const auto cell = [&](int n, int& b, int& e, unsigned char& t) { site.cell(grid, n, b, e, t); };
    if constexpr (WARP) dSiteListWarp(cell, lane, warpBuf, pass, weigh);
    else dSiteListWalk(cell, [&](int k, unsigned char t) { if (pass(k, t) >= 0) weigh(k); });
}
// The camera kernel's inline merge (dVcmMergeSum, its own thread's walk), kept out of line: inlined into
// the camera kernel, which sits at the register cap, the merge ran mode U's renders a fifth slower.
template <int NS>
__device__ __noinline__ static void dVcmMergeSumInline(const DScene& sc, const DVcmCtx& ctx, const DVcmGrid& grid,
                                                       const DVcmSite& site, int secStride, const DVertex& vt,
                                                       Real coatTh, const DMergeFootprint& fp, const DVec3& wo,
                                                       double dVCM, double dVM, int edges, const DJSpec* js,
                                                       double& mx, double& my, double& mz) {
    dVcmMergeSum<NS, false>(sc, ctx, grid, site, secStride, vt, coatTh, fp, wo, dVCM, dVM, edges, js, 0, nullptr,
                            mx, my, mz);
}

// The first `n` queued merge sites (DVcmMergeQueue), a warp each: the lanes take the site's light
// vertices together (dVcmMergeSum), the footprint's probes are the warp's, and lane 0 adds the
// site's sum to the accumulator. Launched in blocks of kSiteWarpBlock threads (dSiteWarpBuf).
#ifndef FT_FAMILY_TU
template <int NS>
__global__ void kVcmMergeGather(DScene sc, DVcmCtx ctx, DVcmGrid grid, int secStride,
                                DVcmMergeQueue mq, int n, double* accum) {
    int* const buf = dSiteWarpBuf();
    D_MOTION_CLEAR();   // this thread's time: the launch's (passU), as the camera walk's
    const int lane = threadIdx.x & 31;
    const int warps = (int)((gridDim.x * blockDim.x) >> 5);
    for (int r = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5); r < n; r += warps) {
        const DVcmMergeReq& rq = mq.reqs[r];
        const DJSpec* js = (rq.flags & 1) ? &mq.spec[r] : nullptr;
        const DMergeFootprint fp = dVertexMergeFootprint(sc, rq.vt, (Real)ctx.radius);
        double mx = 0.0, my = 0.0, mz = 0.0;
        dVcmMergeSum<NS, true>(sc, ctx, grid, DVcmSite(grid, rq.vt.p), secStride, rq.vt, rq.coatTh, fp, rq.wo,
                               rq.dVCM, rq.dVM, rq.edges, js, lane, buf, mx, my, mz);
        dWarpSiteAdd(sc, fp, rq.gaSeed, lane, mx, my, mz, rq.f, accum, rq.pix);
    }
}
#endif

// Phase 3: one camera subpath per pixel of [pixBegin, pixEnd). Does emission (s=0), NEE (s=1),
// vertex connection to the PAIRED light subpath lp = i % nLight, [pathBegin[lp],pathEnd[lp]), and
// merging over the grid -- queued in `mq` for kVcmMergeGather where it holds them -- then adds this
// pass's per-pixel radiance (camera result + the light splat) into the persistent `accum` sum.
// Device twin of vcm.h traceCameraSubpath. `grid.lv` is the compact light-vertex array, `lvHs` its
// vertices' hit states (null where the session keeps none: no motion, no cavity).
#ifndef FT_FAMILY_TU
template <int NS>
__global__ void kVcmCameraT(DScene sc, DCamera cam, int diffraction, DVcmCtx ctx, DVcmGrid grid,
                            const DVcmSec* lvSec, int secStride, int heroC,
                            const int* pathBegin, const int* pathEnd, const double* splat,
                            double* accum, const Real* lamBuf, const double* invLamBuf,
                            int resX, int resY, long long nLight, unsigned long long seedBase,
                            long long passIdx, DVcmMergeQueue mq, long long pixBegin, long long pixEnd,
                            const DVertHitState* lvHs) {
    D_MOTION_CLEAR();   // this thread's time: the launch's (passU) until a path sets one
    constexpr int SECN = (NS > 0) ? NS : 1;
    const int C = (NS > 0) ? heroC : 1;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    for (long long i = pixBegin + g; i < pixEnd; i += G) {
        // The light subpath this camera path is paired with -- its own index, unless a pass
        // traces fewer light subpaths than pixels (vcmSessionBegin), when several camera paths
        // share one. Each still connects to an independent draw of the light subpath, and the
        // weights count nLight light subpaths for merging and the camera splat (ctx.nLightPaths),
        // so the estimator is the same; only its variance changes.
        const long long lp = i % nLight;
        double sxl = splat[i * 3 + 0], syl = splat[i * 3 + 1], szl = splat[i * 3 + 2];
        double invPdfLambda = invLamBuf[lp * C];
        if (invPdfLambda <= 0.0) {                         // no valid wavelength: only the splat
            accum[i * 3 + 0] += sxl; accum[i * 3 + 1] += syl; accum[i * 3 + 2] += szl;
            continue;
        }
        // THE bundle of the paired light subpath lp — the SAME C wavelengths the light kernel
        // walked. That pairing is what makes strategy (c) (connection to the paired light
        // subpath) exact per-λ rather than an approximation: both ends share the λ set.
        Real lamAll[SECN + 1];
        double invAll[SECN + 1];
        for (int k = 0; k < C; ++k) { lamAll[k] = lamBuf[lp * C + k]; invAll[k] = invLamBuf[lp * C + k]; }
        Real lambda = lamAll[0];
        int nUp = cam.enclosure.n ? 1 : C;   // the hero alone from a camera inside a solid (host twin)
        int px = (int)(i % resX), py = (int)(i / resX);
        unsigned long long s = (unsigned long long)i * 0xC2B2AE3D27D4EB4FULL
                             + (unsigned long long)passIdx * 0xA24BAED4963EE407ULL;
        DRng rng; rng.seedUnit(s, seedBase + 77);

        double rx = 0, ry = 0, rz = 0;
        double cieCx = (double)cieX(lambda), cieCy = (double)cieY(lambda), cieCz = (double)cieZ(lambda);
        double betaSec[SECN];
        double cieSx[SECN], cieSy[SECN], cieSz[SECN];
        for (int k = 0; k + 1 < C; ++k) {
            betaSec[k] = 1.0;                              // camera vertex beta == 1 for every λ
            cieSx[k] = (double)cieX(lamAll[k + 1]);
            cieSy[k] = (double)cieY(lamAll[k + 1]);
            cieSz[k] = (double)cieZ(lamAll[k + 1]);
        }

        DVec3 ro, rd;
        Real jx = rng.uniform(), jy = rng.uniform();
        dGenRay(cam, px, py, jx, jy, ro, rd);
        double cosAtCamera = ddot(rd, cam.w);
        if (cosAtCamera <= 1e-9) {
            accum[i * 3 + 0] += sxl; accum[i * 3 + 1] += syl; accum[i * 3 + 2] += szl;
            continue;
        }
        double cameraPdfW = ctx.imagePlaneDist * ctx.imagePlaneDist /
                            (cosAtCamera * cosAtCamera * cosAtCamera);
        double beta = 1.0;
        // VCM-SPEC: the walk's spectral factors on mode J's carrier (host twin vcm.h).
        const bool jOn = c_jSpecOn != 0;
        DJSpecRun jrun; jrun.init();
        double dVCM = ctx.nLightPaths / cameraPdfW;
        double dVC = 0.0, dVM = 0.0;
        DMediumStack stk = cam.enclosure;   // from the eye's solids
        DVec3 prevP = cam.eye;
        // Has this eye chain passed through ONLY delta (specular) vertices so far? That is
        // the precondition for the escaped-sun strategy below; tracked explicitly rather than
        // inferred from dVCM == 0, which a delta vertex sets but which says nothing about the
        // vertices before it.
        bool camAllDelta = true;
        bool camGlow = true;   // ...and through specular surfaces only: a glow is seen (host twin vcm.h)
        const bool hasSun = sc.sunCount > 0;
        // The vertex just left, should the walk land on an emitter next (host twin vcm.h).
        DVec3 softN, softNg;
        bool softOn = false, softTwo = false;
        Real curveTmin = (Real)0;   // far-side fiber clearance for the NEXT hop (see the continuation)
        Real nextTmin = (Real)-1;   // >= 0: the next hop leaves a strand (dStrandStart) -- its tmin

        for (int edges = 1; edges <= ctx.maxDepth; ++edges) {
            // edges == 1 is the camera-to-first-vertex edge — the primary ray; see
            // DMaterial::hideCamera. (The light subpath walk never gets this.)
            DHit h = closestHitIn(stk, sc, ro, rd,
                                  (nextTmin >= (Real)0) ? nextTmin : ((curveTmin > (Real)0) ? (Real)0 : RAY_TMIN), BIG,
                                  /*camHide=*/(edges == 1), curveTmin);
            nextTmin = (Real)-1;
            if (!h.valid) {
                // The ray left the scene. No env map in VCM scope, but a `light sun` is a
                // delta-DIRECTION emitter with no geometry: the s=0 term never fires for it
                // and no connection or merge can reach it either, so without this the solar
                // disc — and every mirror glint of it — would simply be missing. Add it only
                // when the escaping ray came through camera + delta vertices ONLY, which is
                // exactly the case where nothing else competes (NEE, every s>=2 connection,
                // the t==1 splat and merging all need a connectible vertex). The MIS weight
                // is therefore exactly 1.
                if (hasSun && camAllDelta) {
                    for (int k = 0; k < sc.nEmitters; ++k) {
                        const DEmitter& em = sc.emitters[k];
                        if (em.shape != 6 || !dInSunCone(em, rd)) continue;
                        double e0 = beta * (double)specLookup(em.emitSpd, lambda) * invPdfLambda;
                        double ax = cieCx * e0, ay = cieCy * e0, az = cieCz * e0;
                        for (int j = 0; j + 1 < nUp; ++j) {
                            double ej = betaSec[j] *
                                        (double)specLookup(em.emitSpd, lamAll[j + 1]) * invAll[j + 1];
                            ax += cieSx[j] * ej; ay += cieSy[j] * ej; az += cieSz[j] * ej;
                        }
                        if (nUp > 1) { double inv = 1.0 / nUp; ax *= inv; ay *= inv; az *= inv; }
                        rx += ax; ry += ay; rz += az;
                    }
                }
                break;
            }
            {
                int cm = stk.topMat();
                double a = (cm >= 0) ? (double)specLookup(sc.mats[cm].absorb, lambda) : 0.0;
                if (a > 0.0) beta *= exp(-a * (double)h.t);
                for (int k = 0; k + 1 < nUp; ++k) {       // per-λ: the colour of coloured glass
                    double ak = (cm >= 0) ? (double)specLookup(sc.mats[cm].absorb, lamAll[k + 1]) : 0.0;
                    if (ak > 0.0) betaSec[k] *= exp(-ak * (double)h.t);
                }
                // VCM-SPEC: coloured glass, whatever it absorbs at lambda (host twin vcm.h)
                if (jOn && cm >= 0) {
                    const double tt = (double)h.t;
                    jrun.mul([&](double L) { return exp(-(double)specLookup(sc.mats[cm].absorb, (Real)L) * tt); },
                             exp(-a * tt), false);
                }
            }
            double dist = h.t;
            DVec3 rdCur = rd;
            double cosThetaIn = fabs(ddot(h.n, rdCur * (Real)-1));
            if (cosThetaIn <= 1e-9) break;

            const DMaterial* mp = &sc.mats[h.matId];
            int matId = h.matId;
            if (dIsCompound(mp->type)) {
                int coatPick = -1; double coatR = 0.0, coatCos = 1.0, coatU = 0.0; Real coatTh = (Real)0;
                int c = dResolveCompound(sc, *mp, h, rdCur, lamAll[0], rng,
                                         &coatPick, &coatR, &coatCos, &coatTh, &coatU);
                if (c < 0) break;
                // One coat coin for the bundle, compared per wavelength (as the light kernel).
                if (coatPick >= 0)
                    for (int k = 0; nUp > 1 && k + 1 < nUp; ++k)
                        if ((coatU < dCoatRAtJ(*mp, coatCos, coatTh, lamAll[k + 1])) != (coatPick == 1))
                            nUp = 1;
                // VCM-SPEC: the coat-or-body coin at every bin (host twin vcm.h)
                if (jOn && coatPick >= 0) {
                    const DMaterial& lm = *mp;
                    const bool cp = coatPick == 1;
                    jrun.mul([&](double L) { const double r = dCoatRAtJ(lm, coatCos, coatTh, (Real)L);
                                             return cp ? r : 1.0 - r; },
                             cp ? coatR : 1.0 - coatR, false);
                }
                mp = &sc.mats[c]; matId = c;
            }

            // misArrival(dist, cosThetaIn)
            dVCM *= dist * dist; dVCM /= cosThetaIn; dVC /= cosThetaIn; dVM /= cosThetaIn;

            DVec3 wo = normalize(prevP - h.p);

            // (a) Emission (s=0): an emitter's own geometry, or a surface that glows with no
            // emitter registered (a marched isosurface or CSG solid: vcm.h's `isLight`), whose
            // spectrum falls back to the material's `emit` as bkRadiance's does. With no emitter
            // nothing else can reach the path, so it keeps the whole weight -- where the glow is
            // seen at all (camGlow: through specular surfaces only).
            int li = dEmitterForHit(sc, h, matId);
            const DSpec* eSpd = (li >= 0)                     ? &sc.emitters[li].emitSpd
                               : (mp->matIsLight && camGlow)   ? &mp->matEmit
                                                               : nullptr;
            if (eSpd) {
                double cosLight = ddot(h.ng, wo);
                if (cosLight > 0.0) {
                    // Achromatic `emit pattern:` factor at this hit — the same value the
                    // light-subpath / NEE sides get from the sampler at this point.
                    double ep = dEmitHitScale(sc, mp->emitPat, li, h, wo);
                    double Le = (double)specLookup(*eSpd, lambda) * invPdfLambda * ep;
                    double LeSec[SECN], mxLe = Le;
                    for (int k = 0; k + 1 < nUp; ++k) {
                        LeSec[k] = (double)specLookup(*eSpd, lamAll[k + 1]) * invAll[k + 1] * ep;
                        if (LeSec[k] > mxLe) mxLe = LeSec[k];
                    }
                    if (mxLe > 0.0) {
                        double misW = 1.0;
                        if (li >= 0 && sc.emitters[li].area > 0.0 && sc.totalPower > 0.0 && edges >= 2) {
                            const DEmitter& em = sc.emitters[li];
                            double pdfChoice = em.power / sc.totalPower;
                            const double areaNow = em.area * dLightAreaScaleAt(sc, em, h.ng, h.deformTri);   // its area now
                            double directPdfA = pdfChoice / areaNow;
                            double emissionPdfW = em.normalTilted       // joint density (see vcm.h)
                                ? pdfChoice * dAimDirPdfNow(sc, em, wo) / areaNow
                                : pdfChoice * cosLight / (DPI * areaNow);
                            double wCamera = directPdfA * dVCM + emissionPdfW * dVC;
                            misW = 1.0 / (1.0 + wCamera);
                        }
                        // ...softened at the vertex before, toward the emitter (dEmitterSoftening).
                        const double sE = softOn ? dSideSoftening(rd, softN, softNg, softTwo) : 1.0;
                        double e = beta * Le * misW * sE;
                        double ax = cieCx * e, ay = cieCy * e, az = cieCz * e;
                        for (int k = 0; k + 1 < nUp; ++k) {
                            double ek = betaSec[k] * LeSec[k] * misW * sE;
                            ax += cieSx[k] * ek; ay += cieSy[k] * ek; az += cieSz[k] * ek;
                        }
                        if (nUp > 1) { double inv = 1.0 / nUp; ax *= inv; ay *= inv; az *= inv; }
                        rx += ax; ry += ay; rz += az;
                    }
                }
                // A light block's own surface ends the path; any other glowing material has a
                // body, which the strategies below shade as any surface's.
                if (mp->matReflectsNothing) break;
            }

            if (dConnectibleType(mp->type)) {
                DVertex vt = dVertFromHit(sc, h, matId);
                // vt's BSDF toward the walk's side, for NEE's direction and every connection's (at()).
                DVertexLobe camLobe(sc, vt, wo);
                DVec3 ngoCam = dShadedNg(h);
                bool twoSidedCam = dTwoSidedType(mp->type);
                // Its coat's film (dCoatThicknessAt) at the walk's hit: read once, for NEE, every
                // connection and the merge, and only if one of them weighs a direction.
                bool camCoatRead = false;
                Real camCoatThV = (Real)0;
                const auto camCoatTh = [&]() {
                    if (!camCoatRead) { camCoatThV = dCoatThicknessAt(sc, *mp, h); camCoatRead = true; }
                    return camCoatThV;
                };

                // (b) NEE (s=1) — connect to a freshly sampled light point.
                if (edges + 1 <= ctx.maxDepth && sc.nEmitters > 0 && sc.totalPower > 0.0) {
                    int ei = (sc.nEmitters > 1) ? selectEmitter(sc, (double)rng.uniform()) : 0;
                    const DEmitter& em = sc.emitters[ei];
                    if (!(em.shape == 3 || em.collimated)) {
                        const bool neeDelta = dIsDeltaEmitter(em);
                        Real u1 = rng.uniform(), u2 = rng.uniform();
                        // Per-shape connection geometry (device twin of vcm.h's NEE branch).
                        // All three emission models produce the same five quantities, after
                        // which the BSDF / occlusion / MIS code below is shared:
                        //   wiL          connection direction (this vertex -> the light)
                        //   distL        shadow-ray length (out of the scene for a sun)
                        //   cosAtLight   |cos| at the light (1 for both delta lights)
                        //   directPdfW   the NEE density in SOLID ANGLE at this vertex
                        //   emissionPdfW pdfPos*pdfDirW of the same sample (no pickProb: it
                        //                cancels in the wCamera ratio and is applied
                        //                explicitly in wLight / the contribution below)
                        // `epat` scales the emitted spectrum only, never a pdf.
                        DVec3 wiL; double distL = 0.0, dist2 = 0.0, cosAtLight = 1.0;
                        double neeDirectPdfW = 0.0, neeEmissionPdfW = 0.0, epat = 1.0;
                        double occlEps = 2e-6;
                        bool geomOk = false;
                        if (em.shape == 2) {
                            // Delta POSITION: the connection point is deterministic, so the
                            // positional pdf is 1 and the solid-angle density is just dist^2.
                            // The penumbra weights the INTENSITY toward this receiver (it is a
                            // radiance factor, never a density — folding it into a pdf would
                            // break the light subpath's matching dVCM).
                            // where the spot is at the pass's time, its beam turned with it
                            DVec3 org = em.origin, beam = em.beamDir;
                            dLightCarry(sc, em, org, beam);
                            DVec3 toL = org - h.p; dist2 = ddot(toL, toL);
                            if (dist2 > 1e-12) {
                                distL = sqrt(dist2); wiL = toL * (Real)(1.0 / distL);
                                epat = spotFalloff(ddot(wiL * (Real)-1, beam),
                                                   em.spotCosInner, em.spotCosOuter);
                                double coneSolid = 2.0 * DPI * (1.0 - em.spotCosOuter);
                                if (epat > 0.0 && coneSolid > 0.0) {
                                    neeDirectPdfW = dist2;        // pdfA(=1)*dist^2/cosAtLight(=1)
                                    neeEmissionPdfW = 1.0 / coneSolid;
                                    geomOk = true;                // emitSpd is an INTENSITY (W/sr)
                                }
                            }
                        } else if (em.shape == 6) {
                            // Delta DIRECTION at infinity: sample inside the solar cone
                            // (pdfW = 1/Omega) and shadow-ray all the way out of the scene, so
                            // the ray must NOT be shortened by the endpoint epsilon. No 1/dist^2
                            // and no cosLight; emitSpd is already a radiance.
                            double coneSolid = 2.0 * DPI * (1.0 - em.spotCosOuter);
                            double R = sc.sceneRadius;
                            if (coneSolid > 0.0 && R > 0.0) {
                                wiL = dSunSampleCone(em, em.beamDir * (Real)-1, (double)u1, (double)u2);
                                distL = (double)length(sc.sceneCenter - h.p) + R;
                                dist2 = distL * distL;
                                occlEps = 0.0;
                                neeDirectPdfW = 1.0 / coneSolid;
                                // The origin density a light subpath would have needed
                                // (dSampleSunOrigin, the aimed mixture included).
                                neeEmissionPdfW = dSunOriginPdfAlong(sc, em, h.p + wiL * (Real)distL, h.p)
                                                / coneSolid;
                                geomOk = true;
                            }
                        } else {
                            DVec3 yL, nL;
                            // Sampled point's emission-pattern factor (1.0 when unpatterned).
                            double areaScale = 1.0;   // its area at the pass's time about the point
                            epat = dEmitterSamplePoint(sc, em, u1, u2, yL, nL, &areaScale, &rng);
                            DVec3 toL = yL - h.p; dist2 = ddot(toL, toL);
                            if (dist2 > 1e-12 && em.area > 0.0) {
                                distL = sqrt(dist2); wiL = toL * (Real)(1.0 / distL);
                                // An aimed quad (host twin vcm.h): the face's cosine is the
                                // measure, the aimed density the emission's, and its radiance
                                // factor scales Le alone, as `epat` does.
                                // an aimed quad's readings in the instant's frame, the direction
                                // carried back once (dLightDirThen)
                                const DVec3 wInst = em.normalTilted ? dLightDirThen(sc, em, wiL * (Real)-1)
                                                                    : wiL * (Real)-1;
                                cosAtLight = em.normalTilted ? ddot(em.nGeom, wInst)
                                                             : ddot(nL, wiL * (Real)-1);
                                const double aimR = em.normalTilted ? (double)dAimRadiance(em, wInst) : 1.0;
                                if (cosAtLight > 0.0 && aimR > 0.0) {   // emitter stays one-sided
                                    double invArea = 1.0 / (em.area * areaScale);   // at the pass's time
                                    neeDirectPdfW = invArea * dist2 / cosAtLight;
                                    neeEmissionPdfW = em.normalTilted
                                        ? invArea * dAimDirPdf(em, wInst)
                                        : invArea * cosAtLight / DPI;
                                    if (em.normalTilted) epat *= aimR;
                                    geomOk = true;
                                }
                            }
                        }
                        if (geomOk) {
                            double cosToLight = ddot(h.n, wiL);
                            double stG = (mp->type == D_HAIR) ? 1.0
                                       : dSideSoftening(wiL, h.n, ngoCam, twoSidedCam);
                            bool sideOk = twoSidedCam ? (cosToLight != 0.0) : (cosToLight > 0.0 && stG > 0.0);
                            if (sideOk) {
                                // Cheap gates + shadow ray first, BSDF eval after (bit-
                                // identical: no RNG in either; skips the eval when shadowed).
                                double Le = (double)specLookup(em.emitSpd, lambda) * invPdfLambda * epat;
                                double LeSec[SECN], mxLe = Le;
                                for (int k = 0; k + 1 < nUp; ++k) {
                                    LeSec[k] = (double)specLookup(em.emitSpd, lamAll[k + 1]) * invAll[k + 1] * epat;
                                    if (LeSec[k] > mxLe) mxLe = LeSec[k];
                                }
                                                                DVec3 oo = dFiberConnOrigin(h, mp->type == D_HAIR, wiL);
                                double f = 0.0, fSec[SECN];
                                for (int k = 0; k + 1 < nUp; ++k) fSec[k] = 0.0;
#if FTRACE_GPU_FP32
                                // A strand camera vertex starts its shadow ray per dStrandStart (0.385.0).
                                DStrandStart nst; nst.o = oo; nst.tmin = RAY_TMIN;
                                nst.curveTmin = dFiberStep(h, mp->type == D_HAIR, wiL);
                                if (mp->type == D_HAIR && h.fiberRadius > (Real)0)
                                    nst = dStrandStart(h.p, h.n, h.tangent, h.fiberRadius, wiL);
#else
                                const DStrandStart nst{oo, RAY_TMIN, dFiberStep(h, mp->type == D_HAIR, wiL)};
#endif
                                if (mxLe > 0.0 &&
                                    !occludedTo(sc, nst.o, h.p + wiL * (Real)distL, occlEps, nst.tmin, false,
                                                nst.curveTmin)) {
                                    // ...times a coated vertex's coat toward the light (dCoatF).
                                    const DCoatEval ce = dCoatEvalAt(sc, *mp, h.n, camCoatTh(), wo, wiL);
                                    camLobe.at(wiL);
                                    f = camLobe.f(lambda) * stG * ce.at(lambda);
                                    for (int k = 0; k + 1 < nUp; ++k)
                                        fSec[k] = camLobe.f(lamAll[k + 1]) * stG * ce.at(lamAll[k + 1]);
                                }
                                // Fuse BSDF x Le per-λ before the max test: either factor may
                                // vanish at the hero while the product is alive at a secondary.
                                // At nUp == 1 this is exactly the old `f > 0` gate.
                                double mxfLe = f * Le;
                                for (int k = 0; k + 1 < nUp; ++k) {
                                    double p = fSec[k] * LeSec[k];
                                    if (p > mxfLe) mxfLe = p;
                                }
                                if (mxfLe > 0.0) {
                                    double pdfChoice = em.power / sc.totalPower;
                                    double directPdfW = neeDirectPdfW;
                                    double emissionPdfW = neeEmissionPdfW;
                                    // camLobe is at wiL: the shadow test passed, or f would be 0.
                                    double bsdfDirPdfW = camLobe.pdf(false, lambda);
                                    double bsdfRevPdfW = camLobe.pdf(true, lambda);
                                    // A delta light can never be found by BSDF sampling, so
                                    // that strategy is DROPPED from the balance heuristic
                                    // (weight 0) rather than given a density it does not have.
                                    double wLight = neeDelta ? 0.0
                                                             : (bsdfDirPdfW / (pdfChoice * directPdfW));
                                    double wCamera = (emissionPdfW * fabs(cosToLight) /
                                                      (directPdfW * cosAtLight)) *
                                                     (mp->type == D_HAIR ? 0.0   // no light-side alternative through a strand
                                                      : ((dVcmMergeAt(*mp, h.fiberRadius) ? ctx.misVmWeight : 0.0) +
                                                         dVCM + dVC * bsdfRevPdfW));
                                    double misW = 1.0 / (wLight + 1.0 + wCamera);
                                    double contrib = misW * fabs(cosToLight) /
                                                     (pdfChoice * directPdfW) * Le * f *
                                                     dLegAbsorption(sc, h.extMat, distL, lamAll[0]);   // the vertex's solid
                                    double ax = 0, ay = 0, az = 0;
                                    if (contrib > 0.0) {
                                        double e = beta * contrib;
                                        ax = cieCx * e; ay = cieCy * e; az = cieCz * e;
                                    }
                                    for (int k = 0; k + 1 < nUp; ++k) {
                                        double ck = misW * fabs(cosToLight) /
                                                    (pdfChoice * directPdfW) * LeSec[k] * fSec[k] *
                                                    dLegAbsorption(sc, h.extMat, distL, lamAll[k + 1]);
                                        if (ck > 0.0) {
                                            double ek = betaSec[k] * ck;
                                            ax += cieSx[k] * ek; ay += cieSy[k] * ek; az += cieSz[k] * ek;
                                        }
                                    }
                                    if (nUp > 1) { double inv = 1.0 / nUp; ax *= inv; ay *= inv; az *= inv; }
                                    rx += ax; ry += ay; rz += az;
                                }
                            }
                        }
                    }
                }

                // (c) Vertex connection to the PAIRED light subpath's stored vertices.
                int pb = pathBegin[lp], pe = pathEnd[lp];
                for (int j = pb; j < pe; ++j) {
                    const DVcmLV& lv = grid.lv[j];
                    if (edges + lv.edges + 1 > ctx.maxDepth) continue;
                    // A compound's merge vertex stored under a lobe that does not connect.
                    if (!dLvConnects(sc.mats[lv.matId].type)) continue;
                    DVec3 dv = lv.p - h.p; double dist2 = ddot(dv, dv);
                    if (dist2 <= 1e-12) continue;
                    double distc = sqrt(dist2); DVec3 w = dv * (Real)(1.0 / distc);
                    double cosCam = ddot(h.n, w);
                    double cosLit = ddot(lv.ns, w * (Real)-1);
                    DVec3 ngoLit = dLvGeoN(lv);
                    bool twoSidedLit = dTwoSidedType(sc.mats[lv.matId].type);
                    // Unsoftened at both ends, each on the other's geometric front; the light end's
                    // camera side on either side of its shading normal (host twin vcm.h).
                    bool camSide = twoSidedCam ? (cosCam != 0.0) : (cosCam > 0.0 && ddot(ngoCam, w) > 0.0);
                    bool litSide = (cosLit != 0.0) && (twoSidedLit || ddot(ngoLit, w * (Real)-1) > 0.0);
                    if (!camSide || !litSide) continue;
                    // Occlusion FIRST: the shadow ray kills most connections in a
                    // typical scene, and it is far cheaper than the BSDF and MIS block
                    // below. No RNG is consumed in this
                    // loop and occluded() has no side effects, so hoisting the
                    // test is bit-identical — it only skips work for connections
                    // that contributed nothing anyway.
                    // A strand camera vertex clears its tube with the curve-only tmin (vcm.h fiberStep);
                    // the light end is never a strand (dLvConnects).
                    DVec3 oo = dFiberConnOrigin(h, mp->type == D_HAIR, w);
#if FTRACE_GPU_FP32
                    // A strand camera vertex starts its shadow ray per dStrandStart (0.385.0).
                    DStrandStart cst; cst.o = oo; cst.tmin = RAY_TMIN;
                    cst.curveTmin = dFiberStep(h, mp->type == D_HAIR, w);
                    if (mp->type == D_HAIR && h.fiberRadius > (Real)0)
                        cst = dStrandStart(h.p, h.n, h.tangent, h.fiberRadius, w);
#else
                    const DStrandStart cst{oo, RAY_TMIN, dFiberStep(h, mp->type == D_HAIR, w)};
#endif
                    if (occludedTo(sc, cst.o, h.p + w * (Real)distc, 2e-6, cst.tmin, false,
                                   cst.curveTmin, -1, lv.pscale)) continue;
                    const DVertex lvt = dVertFromLV(lv, lvHs ? &lvHs[j] : nullptr);
                    double adjLit = (double)dShadingAdjointCorr(lv.wo, w * (Real)-1, lv.ns, ngoLit);
                    // Each end a coated vertex's coat along the connection (dCoatF).
                    const DCoatEval ceCam = dCoatEvalAt(sc, *mp, h.n, camCoatTh(), wo, w);
                    // The light vertex's lobe from the connection's side, at its walk's direction: its
                    // value there is dBsdfFAdjoint's at every wavelength, its two pdfs the connection's
                    // reversed (dPairSwapped), its hit the coat's.
                    DVertexLobe litLobe(sc, lvt, w * (Real)-1, true);
                    litLobe.at(lv.wo);
                    const DCoatEval ceLit = dCoatEval(sc, litLobe.m, litLobe.h, lv.wo, w * (Real)-1);
                    camLobe.at(w);
                    double fCam = camLobe.f(lambda) * ceCam.at(lambda);
                    double fLit = litLobe.f(lambda) * ceLit.at(lambda);
                    fLit *= adjLit;
                    // The camera path and the stored light path share this pass's bundle (same
                    // path index i), so a connection is EXACT per-λ over the wavelengths still
                    // live at BOTH ends. `lv.nUp` is 1 whenever the light walk de-hero'd.
                    const int lvUp = (NS > 0) ? lv.nUp : 1;
                    const int nUpConn = (nUp < lvUp) ? nUp : lvUp;
                    double fProdSec[SECN], mxProd = fCam * fLit;
                    const DVcmSec* lsRow = ((NS > 0) && lvSec) ? (lvSec + (size_t)j * secStride) : nullptr;
                    for (int k = 0; k + 1 < nUpConn; ++k) {
                        double fc = camLobe.f(lamAll[k + 1]) * ceCam.at(lamAll[k + 1]);
                        double fl = litLobe.f(lamAll[k + 1]) * adjLit * ceLit.at(lamAll[k + 1]);
                        fProdSec[k] = fc * fl;
                        if (fProdSec[k] > mxProd) mxProd = fProdSec[k];
                    }
                    // At nUpConn == 1 this is exactly the old `fCam <= 0 || fLit <= 0` bail
                    // (both factors are non-negative, so the product is 0 iff either is).
                    if (mxProd <= 0.0) continue;
                    double camDirPdfW = camLobe.pdf(false, lambda);
                    double camRevPdfW = camLobe.pdf(true, lambda);
                    double litDirPdfW = litLobe.pdf(true, lambda);    // -w drawn from lv.wo
                    double litRevPdfW = litLobe.pdf(false, lambda);   // lv.wo drawn from -w
                    double camDirPdfA = camDirPdfW * fabs(cosLit) / dist2;
                    double litDirPdfA = litDirPdfW * fabs(cosCam) / dist2;
                    // No merge AT a curve, at either end.
                    const double vmLit = dVcmMergeAt(sc.mats[lv.matId], lv.fiberR) ? ctx.misVmWeight : 0.0;
                    const double vmCam = dVcmMergeAt(*mp, h.fiberRadius) ? ctx.misVmWeight : 0.0;
                    double wLight = camDirPdfA * (vmLit + lv.dVCM + lv.dVC * litRevPdfW);
                    // No light-side alternative runs THROUGH a strand (a light walk stops at one).
                    double wCamera = (mp->type == D_HAIR) ? 0.0
                                     : litDirPdfA * (vmCam + dVCM + dVC * camRevPdfW);
                    double misW = 1.0 / (wLight + 1.0 + wCamera);
                    double Gt = fabs(cosCam) * fabs(cosLit) / dist2;
                    double contrib = misW * Gt * fCam * fLit * beta * lv.beta *
                                     dLegAbsorption(sc, h.extMat, distc, lamAll[0]);   // the solid both ends lie in
                    double ax = 0, ay = 0, az = 0;
                    if (contrib > 0.0) { ax = cieCx * contrib; ay = cieCy * contrib; az = cieCz * contrib; }
                    for (int k = 0; k + 1 < nUpConn; ++k) {
                        double ck = misW * Gt * fProdSec[k] * betaSec[k] * lsRow[k].beta *
                                    dLegAbsorption(sc, h.extMat, distc, lamAll[k + 1]);
                        if (ck > 0.0) { ax += cieSx[k] * ck; ay += cieSy[k] * ck; az += cieSz[k] * ck; }
                    }
                    if (nUpConn > 1) { double inv = 1.0 / nUpConn; ax *= inv; ay *= inv; az *= inv; }
                    rx += ax; ry += ay; rz += az;
                }

                // (d) Vertex merging — gather nearby light vertices from ALL paths (XYZ estimate):
                // queued for kVcmMergeGather (DVcmMergeQueue), or here where the site holds few
                // vertices (inlineMax) or the queue is full.
                if (ctx.vmNorm > 0.0 && grid.nLV > 0 && dVcmMergeAt(*mp, h.fiberRadius)) {   // not AT a curve (vcm.h)
                    // The footprint probe's stream, off the walk's (dSurfFootprint): one draw at
                    // every site, so the stream past it is the same whether the site is queued,
                    // merged here or empty.
                    unsigned long long gaSeed = 0;
                    if (sc.gatherArea) gaSeed = rng.next64();
                    const DVcmSite site(grid, vt.p);
                    // The light vertices it weighs, as far as the choice below needs them counted.
                    const int pop = site.count(grid, mq.reqs ? mq.inlineMax + 1 : 1);
                    if (pop > 0) {
                        DJSpec js;
                        if (jOn) jrun.snap(js);
                        const DJSpec* jsp = (jOn && js.on()) ? &js : nullptr;   // the carrier arriving here (VCM-SPEC)
                        const double f = (jsp ? js.beta(beta) : beta) * ctx.vmNorm;
                        const Real coatTh = camCoatTh();
                        bool queued = false;
                        // nReqs counts the sites turned away as well (they size the next wave), so
                        // past 2^31 of them it wraps negative: slot >= 0. A carrier with no table to
                        // ride in is turned away and counted, which tools/check_vcm_wavefront.py fails.
                        if (mq.reqs && pop > mq.inlineMax) {
                            atomicAdd(&mq.profile[i / kVcmProfBlock], 1);
                            const int slot = (!jsp || mq.spec) ? atomicAdd(mq.nReqs, 1) : -1;
                            if (slot >= 0 && slot < mq.cap) {
                                DVcmMergeReq& rq = mq.reqs[slot];
                                rq.vt = vt; rq.wo = wo; rq.dVCM = dVCM; rq.dVM = dVM; rq.f = f;
                                rq.pix = i * 3; rq.gaSeed = gaSeed; rq.edges = edges; rq.flags = 0;
                                rq.coatTh = coatTh;
                                if (jsp) { mq.spec[slot] = js; rq.flags = 1; }
                                queued = true;
                            } else {
                                atomicAdd(mq.overflow, 1);
                            }
                        }
                        if (!queued) {
                            const DMergeFootprint fp = dVertexMergeFootprint(sc, vt, (Real)ctx.radius);
                            double mx = 0.0, my = 0.0, mz = 0.0;
                            dVcmMergeSumInline<NS>(sc, ctx, grid, site, secStride, vt, coatTh, fp, wo, dVCM, dVM,
                                                   edges, jsp, mx, my, mz);
                            if (mx != 0.0 || my != 0.0 || mz != 0.0) {
                                const double fa = f * (sc.gatherArea ? dSurfFootprint(sc, fp, gaSeed) : 1.0);
                                rx += mx * fa; ry += my * fa; rz += mz * fa;
                            }
                        }
                    }
                }
            }

            if (edges == ctx.maxDepth) break;

            DVec3 wi; double betaFactor, pdfW, pdfRevW, cosThetaOut; bool delta, terminate;
            double secF[SECN]; bool secChromatic = false, keepBundle = false;
            double radianceScale = 1.0;
            dVcmScatter(sc, *mp, h, rdCur, lambda, rng, matId, stk, &radianceScale, diffraction,
                        wi, betaFactor, pdfW, pdfRevW, cosThetaOut, delta, terminate,
                        lamAll, nUp, secF, &secChromatic, &keepBundle, jOn ? &jrun : nullptr,
                        invPdfLambda);
            betaFactor *= radianceScale;   // the camera walk carries radiance across a refraction
            double mxF = betaFactor;
            if (secChromatic) for (int k = 0; k + 1 < nUp; ++k) if (secF[k] > mxF) mxF = secF[k];
            if (terminate || !(mxF > 0.0)) break;
            if (!delta && (pdfW <= 0.0 || cosThetaOut <= 0.0)) break;

            if (delta) { dVCM = 0.0; dVC *= cosThetaOut; dVM *= cosThetaOut; }
            else {
                double t = cosThetaOut / pdfW;
                // No merge AT a strand (vcm.h misScatter `mergeable`): the merge-at-this-vertex
                // alternative leaves both accumulators (dVM is dVC in units of 1/etaVCM).
                const bool mergeable = dVcmMergeAt(*mp, h.fiberRadius);
                dVC = t * (dVC * pdfRevW + dVCM + (mergeable ? ctx.misVmWeight : 0.0));
                dVM = t * (dVM * pdfRevW + dVCM * ctx.misVcWeight + (mergeable ? 1.0 : 0.0));
                dVCM = 1.0 / pdfW;
            }
            // A light walk never scatters at a strand (kVcmLightT, 0.365.0): no alternative from
            // here inward puts one on the light side. dVCM stays -- the light subpath may still
            // END at the next vertex and connect back to this one.
            if (mp->type == D_HAIR && !delta) { dVC = 0.0; dVM = 0.0; }
            beta *= betaFactor;                           // camera side: no adjoint on continuation
            for (int k = 0; k + 1 < nUp; ++k) betaSec[k] *= secChromatic ? secF[k] : betaFactor;
            // a coated vertex's coat on the way out, and on mode J's carrier
            if (!dCoatContinueWalk(sc, *mp, h, wo, normalize(wi), lambda, beta, betaSec, lamAll + 1, nUp - 1,
                                   [&](const DCoatEval& ce, double s) {
                                       if (jOn) jrun.mul([&](double L) { return ce.at((Real)L); }, s, false); }))
                break;
            if (!delta) camAllDelta = false;               // disqualifies the escaped-sun strategy
            if (!delta && !dIsSpecularType(mp->type)) camGlow = false;
            if (delta && !keepBundle) nUp = 1;             // λ-dependent direction change
            prevP = h.p;
            softOn = !delta && dLvConnects(mp->type);
            softTwo = mp->type == D_DIFFUSETRANSMIT;
            softN = h.n;
            softNg = dShadedNg(h);
            // A fiber scattering out its far side steps clear of its own tube (vcm.h fiberStep): the
            // clearance is closestHit's curve-only tmin and the origin an ULP nudge along the ray --
            // RAY_EPS is 1e-4 in FP32, a whole strand, and would skip the neighbour too.
            curveTmin = dFiberStep(h, mp->type == D_HAIR, wi);
            rd = normalize(wi);
            ro = (curveTmin > (Real)0) ? dOffsetPoint(h.p, rd, h.pscale)
               : delta ? dOffsetSpecular(h.p, h.ng, rd, h.pscale)                      // its line kept (host twin vcm.h)
                       : dOffsetAlong(h.p, h.ng, wi, h.pscale);
#if FTRACE_GPU_FP32
            if (mp->type == D_HAIR && h.fiberRadius > (Real)0) {   // see dStrandStart (0.385.0)
                const DStrandStart st = dStrandStart(h.p, h.n, h.tangent, h.fiberRadius, rd);
                ro = st.o; nextTmin = st.tmin; curveTmin = st.curveTmin;
            }
#endif
        }

        accum[i * 3 + 0] += rx + sxl;
        accum[i * 3 + 1] += ry + syl;
        accum[i * 3 + 2] += rz + szl;
    }
}
#endif

// ================= device: on-device grid builds (VCM / SPPM sessions) =================
// Kernels + thrust functors that move the per-pass VCM/SPPM photon-grid construction onto
// the device. Each piece is an EXACT-arithmetic mirror of the host build it replaces
// (vcm.h VcmGrid::build twin in vcmSessionPass / photonmap.h PhotonMap::build), so the
// consuming kernels see bit-identical inputs in the identical order: min/max reductions
// are order-independent and exact; the key kernel (kMergeKey, render_cuda.cu's mergeBinDevice)
// computes a point's bucket with the host's merge_grid.h functions; and a STABLE sort by bucket
// equals a counting sort's output order (both keep original index order within a bucket).

// Double-precision CIE twins (exact color.h mirror; the float `cieX` above serves the Real
// transport path). Used by kSppmGatherConvert: the host build computed cie{X,Y,Z}(lambda)
// in double and rounded cie*power/pi to float ONCE. CUDA's double exp() is within 1 ulp of
// the host's, and that last-ulp double wobble is absorbed by the final float rounding
// (~2^-28 odds per value of landing on a rounding boundary), so the stored floats match.
__device__ static double gaussPiece64(double x, double mu, double s1, double s2) {
    double t = (x - mu) * ((x < mu) ? s1 : s2);
    return exp(-0.5 * t * t);
}
__device__ static double cieX64(double w) {
    return 0.362 * gaussPiece64(w, 442.0, 0.0624, 0.0374)
         + 1.056 * gaussPiece64(w, 599.8, 0.0264, 0.0323)
         - 0.065 * gaussPiece64(w, 501.1, 0.0490, 0.0382);
}
__device__ static double cieY64(double w) {
    return 0.821 * gaussPiece64(w, 568.8, 0.0213, 0.0247)
         + 0.286 * gaussPiece64(w, 530.9, 0.0613, 0.0322);
}
__device__ static double cieZ64(double w) {
    return 1.217 * gaussPiece64(w, 437.0, 0.0845, 0.0278)
         + 0.681 * gaussPiece64(w, 459.0, 0.0385, 0.0725);
}

// Float AABB carried through thrust::transform_reduce. min/max are exact and commute with
// the host's float->double promotion, so the reduced bounds equal the host loop's bounds.
struct BboxF { float mnx, mny, mnz, mxx, mxy, mxz; };
struct LvToBboxF {
    HD BboxF operator()(const DVcmLV& v) const {
        return BboxF{(float)v.p.x, (float)v.p.y, (float)v.p.z,
                     (float)v.p.x, (float)v.p.y, (float)v.p.z};
    }
};
struct PhToBboxF {
    HD BboxF operator()(const DPhoton& p) const {
        return BboxF{(float)p.pos.x, (float)p.pos.y, (float)p.pos.z,
                     (float)p.pos.x, (float)p.pos.y, (float)p.pos.z};
    }
};
struct JSurfToBboxF {
    HD BboxF operator()(const DSurfPhoton& p) const {
        return BboxF{(float)p.p.x, (float)p.p.y, (float)p.p.z,
                     (float)p.p.x, (float)p.p.y, (float)p.p.z};
    }
};
struct BboxMergeF {
    HD BboxF operator()(const BboxF& a, const BboxF& b) const {
        return BboxF{fminf(a.mnx, b.mnx), fminf(a.mny, b.mny), fminf(a.mnz, b.mnz),
                     fmaxf(a.mxx, b.mxx), fmaxf(a.mxy, b.mxy), fmaxf(a.mxz, b.mxz)};
    }
};
// SPPM rMax: radius[i] where the visible point is valid, else 0 (max-reduced; exact).
struct SppmRMaxF {
    const double* radius; const unsigned char* valid;
    HD double operator()(long long i) const { return valid[i] ? radius[i] : 0.0; }
};
// Plain max functor (thrust::maximum is deprecated in CCCL 3.x, and naming it makes nvcc
// echo cub source context that MSBuild's canonical-error regex misparses as an error).
struct MaxD { HD double operator()(double a, double b) const { return a < b ? b : a; } };

// Scatter each light path's stored slab vertices into the compact array at its scanned
// offset (device twin of the old host compaction loop; per-path order preserved). The hero
// secondary rows ride along in lockstep so `secCompact[j*secStride + q]` stays paired with
// `compact[j]`; `secSlab == nullptr` (a `-heroc 1` session) skips that entirely. So do the
// vertices' hit states (`hsSlab`, null where the session keeps none).
#ifndef FT_FAMILY_TU
__global__ void kVcmCompactScatter(const DVcmLV* slab, const DVcmSec* secSlab, int secStride,
                                   const int* lvCount, const int* pathBegin,
                                   DVcmLV* compact, DVcmSec* secCompact, int npix, int vcmCap,
                                   const DVertHitState* hsSlab, DVertHitState* hsCompact) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < npix; i += stride) {
        int cnt = lvCount[i], b = pathBegin[i];
        const DVcmLV* src = slab + (size_t)i * vcmCap;
        for (int k = 0; k < cnt; ++k) compact[b + k] = src[k];
        if (hsSlab)
            for (int k = 0; k < cnt; ++k) hsCompact[b + k] = hsSlab[(size_t)i * vcmCap + k];
        if (secSlab) {
            const DVcmSec* ssrc = secSlab + (size_t)i * vcmCap * secStride;
            for (int k = 0; k < cnt; ++k) {
                int nu = src[k].nUp;
                DVcmSec* dst = secCompact + (size_t)(b + k) * secStride;
                for (int q = 0; q + 1 < nu; ++q) dst[q] = ssrc[(size_t)k * secStride + q];
            }
        }
    }
}
#endif

// The grid's cell-ordered copy (DVcmGrid::byCell): out[k] = lv[order[k]] for the n vertices, and
// each one's `stride` secondary-λ rows beside it (sec null: none).
#ifndef FT_FAMILY_TU
__global__ void kVcmCellGather(const DVcmLV* lv, const DVcmSec* sec, int stride, const int* order, int n,
                               DVcmLV* out, DVcmSec* secOut) {
    const int step = gridDim.x * blockDim.x;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += step) {
        const int i = order[k];
        out[k] = lv[i];
        if (sec)
            for (int q = 0; q < stride; ++q) secOut[(size_t)k * stride + q] = sec[(size_t)i * stride + q];
    }
}
#endif

// ================== A DEVICE LBVH OVER PHOTON BEAMS (mode J, beam half) =================
//
// Karras 2012, emitted into the SAME `DNode` layout the host's SAH builder produces, so
// `dGatherPhotonBeams` traverses either without knowing which built it. See the scraps note for
// why the BVH (not the trace) is the thing worth porting: a host realization is 72 % BVH build.
//
// 30-bit Morton code from the sub-beam's AABB CENTROID, normalised into the map's bounds. The
// centroid, not an endpoint: a beam is a segment and its two ends can be far apart, so keying on
// an end would sort two halves of one chord into different branches.
__device__ static inline unsigned dExpandBits10(unsigned v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
__device__ static inline unsigned dMorton3D(float x, float y, float z) {
    // Clamp before scaling: a centroid exactly on the upper bound would otherwise index 1024.
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    return (dExpandBits10((unsigned)x) << 2) | (dExpandBits10((unsigned)y) << 1)
         | dExpandBits10((unsigned)z);
}

// One Morton code per sub-beam. `lo`/`invExt` normalise the centroid into [0,1]^3.
#ifndef FT_FAMILY_TU
__global__ void kBeamMorton(const DBeamRec* beams, int n, DVec3 lo, DVec3 invExt,
                            unsigned* code, int* idx) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const DBeamRec& b = beams[i];
        // The sub-beam's own extent: origin + s0*d to origin + (s0+len)*d.
        const DVec3 a{b.o.x + b.d.x * b.s0, b.o.y + b.d.y * b.s0, b.o.z + b.d.z * b.s0};
        const DVec3 e{a.x + b.d.x * b.len,  a.y + b.d.y * b.len,  a.z + b.d.z * b.len};
        const float cx = (float)(((double)a.x + (double)e.x) * 0.5 - (double)lo.x) * (float)invExt.x;
        const float cy = (float)(((double)a.y + (double)e.y) * 0.5 - (double)lo.y) * (float)invExt.y;
        const float cz = (float)(((double)a.z + (double)e.z) * 0.5 - (double)lo.z) * (float)invExt.z;
        code[i] = dMorton3D(cx, cy, cz);
        idx[i]  = i;
    }
}
#endif

// Length of the common prefix of codes i and j, with the INDEX appended as a tie-break so that
// duplicate Morton codes (common: many beams inside one medium cell) still yield a strict
// ordering. Without the tie-break Karras's range search does not terminate on duplicates -- the
// single most common way this algorithm is got wrong.
__device__ static inline int dLbvhDelta(const unsigned* code, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    const unsigned ci = code[i], cj = code[j];
    if (ci == cj) return 32 + __clz((unsigned)i ^ (unsigned)j);
    return __clz(ci ^ cj);
}

// One thread per INTERNAL node (there are n-1). Determines the node's range by walking out from
// its own index, finds the split, and wires children: an internal child is its own index, a leaf
// child is offset by (n-1) into the shared node array.
#ifndef FT_FAMILY_TU
__global__ void kLbvhInternal(const unsigned* code, int n, DNode* nodes, int* parent) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n - 1; i += stride) {
        // Direction of the range this node covers.
        const int d = (dLbvhDelta(code, n, i, i + 1) - dLbvhDelta(code, n, i, i - 1)) >= 0 ? 1 : -1;
        const int dMin = dLbvhDelta(code, n, i, i - d);
        int lMax = 2;
        while (dLbvhDelta(code, n, i, i + lMax * d) > dMin) lMax <<= 1;
        int l = 0;
        for (int t = lMax >> 1; t >= 1; t >>= 1)
            if (dLbvhDelta(code, n, i, i + (l + t) * d) > dMin) l += t;
        const int j = i + l * d;
        const int dNode = dLbvhDelta(code, n, i, j);
        int sp = 0;
        for (int t = (l + 1) >> 1; ; t = (t + 1) >> 1) {
            if (dLbvhDelta(code, n, i, i + (sp + t) * d) > dNode) sp += t;
            if (t <= 1) break;
        }
        const int split = i + sp * d + (d < 0 ? -1 : 0);
        const int left  = (min(i, j) == split)     ? (split + n - 1)     : split;
        const int right = (max(i, j) == split + 1) ? (split + 1 + n - 1) : (split + 1);
        nodes[i].left = left;
        nodes[i].right = right;
        nodes[i].first = 0;
        nodes[i].count = 0;          // internal, per BvhNode::isLeaf()
        parent[left] = i;
        parent[right] = i;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) parent[0] = -1;
}
#endif

// Leaves: one per sub-beam, its box set from the beam's own extent inflated by the kernel radius.
#ifndef FT_FAMILY_TU
__global__ void kLbvhLeaves(const DBeamRec* beams, const int* order, int n, float radMax,
                            DNode* nodes) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const DBeamRec& b = beams[order[i]];
        const DVec3 a{b.o.x + b.d.x * b.s0, b.o.y + b.d.y * b.s0, b.o.z + b.d.z * b.s0};
        const DVec3 e{a.x + b.d.x * b.len,  a.y + b.d.y * b.len,  a.z + b.d.z * b.len};
        // The gather accepts a beam within its own medium's kernel radius, so the box has to be
        // inflated by that radius or the traversal will reject hits the estimator would keep --
        // a silent darkening rather than a crash, which is why it is stated here.
        const float r = (b.invRad > 0.0f) ? (1.0f / b.invRad) : radMax;
        DNode& nd = nodes[n - 1 + i];
        nd.lo = {fminf(a.x, e.x) - r, fminf(a.y, e.y) - r, fminf(a.z, e.z) - r};
        nd.hi = {fmaxf(a.x, e.x) + r, fmaxf(a.y, e.y) + r, fmaxf(a.z, e.z) + r};
        nd.left = -1; nd.right = -1;
        nd.first = i; nd.count = 1;
    }
}
#endif

// Bottom-up refit. One thread per leaf walks to the root; an atomic counter per internal node
// lets only the SECOND arriving child proceed, so each internal box is computed exactly once
// and only after both children are final.
#ifndef FT_FAMILY_TU
__global__ void kLbvhRefit(const int* parent, int n, int* visited, DNode* nodes) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int node = parent[n - 1 + i];
        while (node >= 0) {
            if (atomicAdd(&visited[node], 1) == 0) break;   // first child: leave it to the second
            const DNode& L = nodes[nodes[node].left];
            const DNode& R = nodes[nodes[node].right];
            nodes[node].lo = {fminf(L.lo.x, R.lo.x), fminf(L.lo.y, R.lo.y), fminf(L.lo.z, R.lo.z)};
            nodes[node].hi = {fmaxf(L.hi.x, R.hi.x), fmaxf(L.hi.y, R.hi.y), fmaxf(L.hi.z, R.hi.z)};
            __threadfence();
            if (node == 0) break;
            node = parent[node];
        }
    }
}
#endif

// Deposit record -> gather record, in sorted (cell-contiguous) order: out[i] converts
// ph[order[i]] (the stable sort's permutation == the host counting sort's). pos/n copy
// float-for-float (the host round-tripped float->double->float, which is exact); the
// cie*power/pi fold repeats the host's double math (see cieX64 note).
#ifndef FT_FAMILY_TU
__global__ void kSppmGatherConvert(const DPhoton* ph, const int* order, long long n,
                                   DGatherPhoton* out) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += stride) {
        const DPhoton p = ph[order[i]];
        DGatherPhoton g;
        g.pos = p.pos;
        g.n   = p.n;
        const double l = (double)p.lambda;
        const double w = (double)p.power * (1.0 / DPI);
        g.pX = (float)(cieX64(l) * w);
        g.pY = (float)(cieY64(l) * w);
        g.pZ = (float)(cieZ64(l) * w);
        g.lambda = p.lambda;
        out[i] = g;
    }
}
#endif

// One thread per program: evaluate it through both device VMs. Deliberately dumb — this
// runs once from a self-test, and the point is that it exercises the SAME two functions the
// renderer calls, not a simplified copy of them.
#ifndef FT_FAMILY_TU
__global__ static void kPatOpProbe(const PatNode* prog, const PatNodeF* progF,
                                   const int* off, const int* len, int nProg,
                                   const PatOpProbeIn* in, double* outF64, double* outF32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nProg) return;
    DPatEnv env = dPatEnvNoneT<DTexture>();
    // `dext` must be REAL here, not null. It is the extra-dimension bank d4..d12, and it is
    // the whole reason this check exists: leaving it null would make d4..d12 read 0 on the
    // device AND 0 on the host, so the family that shipped broken in 0.230.0 would agree
    // with itself and pass. The check has to supply the inputs that make a missing opcode
    // observable.
    float dF[9];
    for (int k = 0; k < 9; ++k) dF[k] = (float)in->d[k];
    outF64[i] = dPatternEval(prog + off[i], len[i], in->x, in->y, in->z, in->f,
                             in->nx, in->ny, in->nz, in->r, in->u, in->v,
                             in->curv, in->cavity, in->fw, env, in->d);
    outF32[i] = (double)dPatternEvalF(progF + off[i], len[i],
                                      (float)in->x, (float)in->y, (float)in->z, (float)in->f,
                                      (float)in->nx, (float)in->ny, (float)in->nz,
                                      (float)in->r, (float)in->u, (float)in->v, env, dF);
}
#endif

// -checkdevhits (cudaHitProbe): one ray per thread, through closestHit or one intersector alone.
// The intersectors are the renderer's own; the record handed to one alone is full of another
// primitive's values, which every field it owns must overwrite.
#ifndef FT_FAMILY_TU
__global__ static void kHitProbe(DScene sc, const DevHitProbeRay* rays, int n, DevHitProbeOut* out) {
    D_MOTION_CLEAR();   // this thread's time: the launch's (passU) until a path sets one
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const DevHitProbeRay& q = rays[i];
    const DVec3 ro{(Real)q.o[0], (Real)q.o[1], (Real)q.o[2]};
    const DVec3 rd{(Real)q.d[0], (Real)q.d[1], (Real)q.d[2]};
    DHit h;
    if (q.kind < 0) {
        h = closestHit(sc, ro, rd);
    } else {
        h.t = (Real)BIG; h.valid = false; h.matId = -7; h.sensorId = -1; h.emitter = -1; h.back = true;
        h.pscale = (Real)777; h.fiberRadius = (Real)0.5; h.vcol = 3; h.vb0 = h.vb1 = (Real)0.3;
        h.tangent = DVec3{(Real)0, (Real)0, (Real)9}; h.bitangentSign = (Real)-1; h.curv = (Real)99;
        h.solid = 7;
        const Real tmin = (Real)1e-6;
        switch (q.kind) {
            case 0: { const DSphere& s = sc.sph[q.index];
                      intersectSphere(ro, rd, s, tmin, h, s.frame >= 0 ? sc.sphAxes + 3 * s.frame : nullptr); break; }
            case 1: intersectTri(makeTriShear(rd), ro, rd, sc.tris[q.index], tmin, h); break;
            case 2: intersectCurveSeg(makeCurveRay(rd), ro, rd, sc.curveSegs[q.index], tmin, h); break;
            case 3: intersectImplicit(sc, sc.implicits[q.index], ro, rd, tmin, h); break;
            default: break;
        }
    }
    DevHitProbeOut o{};
    o.valid = h.valid ? 1 : 0; o.matId = h.matId; o.back = h.back ? 1 : 0; o.patOwnSide = h.patOwnSide ? 1 : 0;
    o.vcol = h.vcol; o.solid = h.solid;
    o.t = (double)h.t;
    o.p[0] = h.p.x; o.p[1] = h.p.y; o.p[2] = h.p.z;
    o.n[0] = h.n.x; o.n[1] = h.n.y; o.n[2] = h.n.z;
    o.ng[0] = h.ng.x; o.ng[1] = h.ng.y; o.ng[2] = h.ng.z;
    o.pscale = (double)h.pscale; o.fiberRadius = (double)h.fiberRadius; o.curv = (double)h.curv;
    o.tangentZ = (double)h.tangent.z;
    const DVec3 sng = dShadedNg(h);
    o.sng[0] = sng.x; o.sng[1] = sng.y; o.sng[2] = sng.z;
    if (h.valid && h.matId >= 0) {
        const DPatIn in = dPatIn(sc, h);
        o.q[0] = in.x; o.q[1] = in.y; o.q[2] = in.z;
        o.qn[0] = in.nx; o.qn[1] = in.ny; o.qn[2] = in.nz; o.qcurv = in.curv;
        o.cavity = dCavityOf(sc, h);
    }
    out[i] = o;
}
#endif

// ================ MODE J'S SURFACE-PHOTON LIGHT PASS, ON THE DEVICE ====================
//
// Device twin of the deposit inside bdpt.h's traceLightBeamPass (~line 2080) -- the SURFACE
// half only; the beam half needs a BVH rather than a grid and comes second. One thread per
// light subpath: sample a wavelength, walk the subpath with the SAME dGenLightSubpath the
// connection half already uses, and append a DSurfPhoton + its DSurfMis partials at every
// vertex the host would have stored one at.
//
// THE THREE THINGS THAT MUST MATCH THE HOST EXACTLY, because each failure is silent:
//
//  * THE SITE PREDICATE. `dSurfMergeSite` is the device twin of `surfMergeSite`, and the
//    gather and the MIS weight both ask it too. A site the weight counts but the map never
//    fills under-weights every competing technique; one the map fills but the weight ignores
//    double-counts. Three call sites, one predicate.
//  * THE ACCUMULATOR INDEX. `sumC/sumMb/sumMs` are the recurrence's value at u-1, NOT at u: a
//    point merge's reference connection splits after y_{u-1}. The host keeps three arrays and
//    indexes back; here they are three scalars read before they are advanced, which is the
//    same thing and cheaper. Getting this off by one produces a plausible image with a wrong
//    denominator.
//  * `nEmitted` COUNTS ATTEMPTS, NOT SUCCESSES. The host increments `done` before the
//    `pdfLam <= 0` continue, so its n_m is the subpath count it was asked for. Counting only
//    the subpaths that emitted would inflate kappaS and darken every merge.
//
// Vertices beyond `cap` are DROPPED rather than wrapped, and the overflow is reported so the
// caller can raise the cap: a wrapped write would corrupt a photon another thread is about to
// gather from, which is a data race that reads as fireflies.
struct DJSurfOut {
    DSurfPhoton* pts;
    DSurfMis*    mis;
    int*         count;      // [0] photons appended (may exceed cap -- see overflow), [1] unused
    int          cap;
};

}  // namespace gpu
