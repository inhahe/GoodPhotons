// kBdptT, the BDPT megakernel of modes D and J, in a translation unit of its own: compiled under
// -maxrregcount at its launch bounds (CMakeLists.txt), so no helper it calls out of line can take
// more registers than they allow (render_cuda_dev.cuh says why). render_cuda.cu launches it.
#define FT_FAMILY_TU 1
#include "render_cuda_dev.cuh"

namespace gpu {

// BDPT megakernel: one thread renders one (pixel,sample), grid-stride over all
// res*res*spp samples. t>=2 connections land on the sample's own pixel (camFilm);
// t==1 splats land on the projected raster pixel (splatFilm). Both are normalised by
// 1/spp on the host (bdpt.h renderBdpt convention).
// Chunked exactly like kBackward: renders `chunkSpp` samples-per-pixel starting at
// `sampleBase`, seeding on the global sample index (pixel*sppTotal + sampleBase + local)
// so any chunking is bit-identical to a single sppTotal pass.
// __launch_bounds__(128, 3): unconstrained the kernel compiles to 254 regs -> only 2
// blocks (256 threads) resident per SM. Capping at 3 blocks/SM (<=170 regs) trades a few
// extra spills for +50% latency hiding; the kernel is latency-bound on spilled/local
// state (8KB stack/thread), so occupancy wins.
//
// Templated on NS = the number of SECONDARY hero wavelength slots, so the scalar
// instantiation (kBdptT<0, MAXD>) allocates the per-vertex secondary-throughput arrays at one
// element and is bit-for-bit the original single-λ kernel: every hero loop it contains has
// an empty trip count, and every added reject is a max over one value. The hero
// instantiation (kBdptT<BDPT_NSEC, MAXD>) pays 2*MAXV*BDPT_NSEC doubles (~1.2 KB) of extra
// per-thread local state for the bundle.
//
// Also templated on MAXD = the per-thread vertex-stack depth bound. `eye`/`light` are
// thread-local arrays of ~100 B DVertex, so this is the one knob that sets the kernel's
// local-memory footprint: the MAXD = BDPT_MAXDEPTH instantiation is bit-for-bit the
// original kernel, and the MAXD = BDPT_DEEPDEPTH one is only instantiated — and only
// launched — when `-max-bounce N` asks for N > BDPT_MAXDEPTH. The connection double-loop
// below is O(MAXD^2), so the deep variant is genuinely slower per sample; that, plus the
// local-memory cost, is why it is opt-in rather than the default.
//
// Templated on MERGE = mode J (UPBP). False is mode D and the kernel is bit-for-bit what it
// was: no DPathSeg array, no segSum arrays, mergeKappa == 0 (so every merge term inside
// dMisWeight vanishes) and no merge block. True adds the beam×ray half of the estimator.
// A template flag rather than a runtime one because the per-thread segment array is the
// kernel's largest single local allocation after the vertex stacks — mode D must not pay
// occupancy for storage it never writes. Mode J only ever instantiates NS == 0: the hero
// bundle is gated off for any scene with a participating medium, and mode J without one has
// nothing to merge.
template <int NS, int MAXD, bool MERGE, bool STRAT>
__global__ void __launch_bounds__(128, 3)
kBdptT(DScene sc, const __grid_constant__ DCamera cam, double* camFilm, double* splatFilm,
                      long long totalSamples, long long chunkSpp, long long sppTotal,
                      long long sampleBase, int resX, int maxDepth,
                      int diffraction, unsigned long long seedBase, int heroC,
                      DBeamMap bm, DSurfMap sm, DWfQueue wq, long long idxBegin, long long idxEnd,
                      DSurfQueue sq) {
    D_MOTION_CLEAR();   // this thread's time: the launch's (passU) until a path sets one
    enum { SECN = (NS > 0 ? NS : 1), MAXV = BDPT_MAXV_OF(MAXD),
           SEGN = (MERGE ? BDPT_MAXV_OF(MAXD) : 1) };
    // DBeamMap::mergeOn is the whole gate: with it false the camera walk is not even asked to
    // record its segments and the render is mode D down to the RNG stream (gate 1).
    const bool beamsHere = MERGE && bm.gathers();
    const bool mergeOn = MERGE && bm.mergeOn();
    // The merge technique's constant, n_m * 2r, travelling with the map so no call site can
    // forget it. Zero unless the map actually carries MIS partials, and that zero is what
    // makes every merge term inside dMisWeight disappear. Weighted and unweighted merges
    // must move together: weighting the connections down while the merges are still raw
    // would darken the volume as surely as the reverse brightens it, so ONE value gates both.
    const double mergeKappa = mergeOn ? bm.mergeKappa : 0.0;
    // The point x point kind's constant. Zero unless `-jsurf` filled a surface map, and that
    // zero is what makes every surface term in the weight vanish -- the same gate mergeKappa is
    // for the beams.
    // Its map may have drawn empty (nPts 0), which keeps the constant and gathers nothing.
    const double kappaSurf = sm.kappaS;
    // JSPEC: the camera walk carries its spectral factors on a grid so each merge can read them
    // at its beam's or photon's own wavelength (DJSpec). A beam merge reads it off the recorded
    // segment it lies on (DPathSeg::spec), a point merge off the eye vertex it merges at (vspec).
    const bool jSpecOn = MERGE && c_jSpecOn && (mergeOn || kappaSurf > 0.0);
    const bool wantSegs = mergeOn;
    if (maxDepth > MAXD) maxDepth = MAXD;   // device array bound (host picks the variant)
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long G = (long long)gridDim.x * blockDim.x;
    const int C = (NS > 0) ? heroC : 1;
    // [idxBegin, idxEnd) is the wave the host is running (the whole chunk when the
    // wavefront gather is off); idxEnd <= totalSamples.
    // A stop (or -time's cut) ends the launch; its chunk is dropped (DeviceStopScope). Asked before
    // every sample the thread starts -- at a frame's worth of threads each runs only two or three,
    // and they can take seconds -- but the flag, mapped host memory a bus round trip away, is read
    // only on the first and then once ~1 ms of clock (2^21 cycles) has passed.
    long long stopNext = 0;
    unsigned long long work = 0;   // mode J's camera work count (g_dWork), added as the thread ends
    for (long long idx = idxBegin + g; idx < idxEnd; idx += G) {
        const long long cyc = clock64();
        if (cyc >= stopNext) {
            if (dGatherStopped()) return;
            stopNext = cyc + (1ll << 21);
        }
        const long long slot = idx / chunkSpp;   // pixel slot (padded to whole runs when stratified)
        long long pix = slot;
        if (STRAT) {                             // stratified waves: see DWfQueue (compile-time:
                                                 // the default kernel carries none of this)
            const unsigned run = ((unsigned)(slot >> 5) * wq.runMul) & wq.runMask;
            if (run >= wq.nRuns) continue;       // padding run of the power-of-two domain
            const unsigned row = run / wq.runsPerRow;
            const int px0 = (int)(run - row * wq.runsPerRow) * 32 + (int)(slot & 31);
            if (px0 >= resX) continue;           // padding slot of a partial run
            pix = (long long)row * (long long)resX + px0;
        }
        const long long gidxBase = pix * sppTotal + sampleBase + (idx - slot * chunkSpp);
        int px = (int)(pix % resX);
        int py = (int)(pix / resX);
        // PER-MODE SAMPLE COUNTS (see c_routeSppD): kD / kJ sub-samples of this base sample. Sub-sample
        // 0 is the base sample itself (its index, its stream), so both counts at 1 change nothing.
        // (The loop body below keeps its old indentation.)
        int kD = 1, kJ = 1;
        if (MERGE && c_routeOn) {
            kD = dRouteSubCount(c_routeSppD, (unsigned long long)gidxBase, seedBase, 0x5851F42D4C957F2Dull);
            kJ = dRouteSubCount(c_routeSppJ, (unsigned long long)gidxBase, seedBase, 0x14057B7EF767814Full);
        }
        const int nSub = (kD > kJ) ? kD : kJ;
        for (int sub = 0; sub < nSub; ++sub) {
        const long long gidx = (sub == 0) ? gidxBase
            : (long long)(dMix64((unsigned long long)gidxBase * 0xD1B54A32D192ED03ull + (unsigned long long)sub) >> 1);
        const bool allowD = sub < kD, allowJ = sub < kJ;
        DRng rng; rng.seedUnit((unsigned long long)gidx, seedBase);
        // Motion blur (mode D; host twin BdptRenderer::renderRows' blurD): the sample's subpaths and
        // every connection between them at one time in the shutter, off a stream of its own, with
        // the camera posed there when it moves (camT). A camera that holds still is read where it
        // is, in the launch's constant parameters (__grid_constant__ lets its address be taken
        // without a per-thread copy).
        DCamera camT;
        const DCamera* camP = &cam;
        if (!MERGE && cam.blur) {
            DRng tr;
            tr.seedUnit((unsigned long long)gidx, cam.timeSalt ^ 0x3C6EF372FE94F82BULL);
            const bool moves = cam.nKeys > 0;
            dSetPathTime(sc, cam, (double)tr.uniform(), moves ? &camT : nullptr);
            if (moves) camP = &camT;
        }
        const DCamera& camS = *camP;

        // Hero + C-1 stratified secondaries from ONE base uniform (u + i/C wrapped into
        // [0,1)), pushed through the same inverse-CDF the scalar path uses. The hero must
        // have a valid pdf; a dead secondary simply carries invPdf 0 (contributes nothing).
        DHeroBundle hb;
        if (C > 1) {
            double u = (double)rng.uniform(), pdf0 = 0.0;
            hb.lam[0] = dSampleSceneLambdaU(sc, u, pdf0);
            if (pdf0 <= 0.0) continue;
            hb.invPdf[0] = dInvPdfLambda(sc, hb.lam[0]);
            hb.C = C;
            for (int i = 1; i < C; ++i) {
                double uu = u + (double)i / C;
                if (uu >= 1.0) uu -= 1.0;
                double pdfI = 0.0;
                hb.lam[i] = dSampleSceneLambdaU(sc, uu, pdfI);
                hb.invPdf[i] = (pdfI > 0.0) ? dInvPdfLambda(sc, hb.lam[i]) : 0.0;
            }
        } else {
            double pdfLam = 0.0;
            hb.lam[0] = dSampleSceneLambda(sc, rng, pdfLam);
            if (pdfLam <= 0.0) continue;
            hb.invPdf[0] = dInvPdfLambda(sc, hb.lam[0]);
            hb.C = 1;
        }
        const Real lambda = hb.lam[0];

        DVertex eye[MAXV], light[MAXV];
        double eyeSec[MAXV * SECN], lightSec[MAXV * SECN];
        DPathSeg segs[SEGN];
        int nSegs = 0;
        DJSpecRun jrun;
        DJSpec vspec[MERGE ? MAXV : 1];   // JSPEC: the carrier arriving at each eye vertex (point merges)
        // Only pay for escape tracking when the scene actually has a distant sun.
        DEscape esc; esc.escaped = 0; esc.beta = 0.0; esc.nUp = 1;
        // maxDepth + 2 eye vertices, so s = 0 can make a path of every depth it is weighted for
        // (host twin: BdptRenderer::renderRows).
        int nE = dGenCameraSubpath(sc, camS, diffraction, px, py, hb, maxDepth + 2, rng, eye, eyeSec, NS, MAXV,
                                   (sc.sunCount > 0) ? &esc : nullptr,
                                   wantSegs ? segs : nullptr, wantSegs ? &nSegs : nullptr,
                                   wantSegs ? (int)SEGN : 0, jSpecOn ? &jrun : nullptr,
                                   (jSpecOn && kappaSurf > 0.0) ? vspec : nullptr);
        int nL = dGenLightSubpath(sc, camS, diffraction, hb, maxDepth + 1, rng, light, lightSec, NS, MAXV);
        // Its vertices and the connections the loop below can evaluate.
        if (MERGE) work += (unsigned long long)(nE + nL + nE * ((nL > 0 ? nL : 1) + 1));
        // The beam merge's factors at every interior vertex, between its subpath neighbours: each
        // connection's weight reads them (dMisWeight), as do the camera-side sums below -- only
        // where this sub-sample draws mode J's paths (a D-routed connection weighs no merge).
        DMergeGeo geoE[MERGE ? MAXV : 1], geoL[MERGE ? MAXV : 1];
        const bool geoNow = MERGE && mergeKappa > 0.0 && allowJ;   // else the weights get null
        if (geoNow) {
            const DPatEnv envG = dPatEnvOf(sc);
            for (int k = 1; k + 1 < nE; ++k)
                geoE[k] = dMergeGeo(sc, eye[k + 1].p, eye[k], eye[k - 1].p, lambda, envG);
            for (int i = 1; i + 1 < nL; ++i)
                geoL[i] = dMergeGeo(sc, light[i - 1].p, light[i], light[i + 1].p, lambda, envG);
        }

        // PER-OBJECT ROUTE (0.386.0; see c_routeOn). routeR: the route of every path this camera
        // subpath makes with t >= 2, which all share its leading chain eye[1..fIdx]. prefJ bit k
        // (0 < k < fIdx): the chain PREFIX eye[1..k] is already J, so a beam merge on segment k,
        // which extends that prefix by a medium point, is J whatever the medium; otherwise such a
        // merge is J only in a J medium (DWfSeg::routeMedJ).
        int routeR = 1, fIdx = nE;
        // 64 bits, as the host's routeOfEye: the deep kernel's chains run past 32 vertices.
        unsigned long long prefJ = 0ull;
        const bool routeOn = MERGE && c_routeOn;
        if (routeOn) {
            int r = -1;
            for (int k = 1; k < nE; ++k) {
                const int m = dRouteOfVert(sc, eye[k]);
                if (m > r) r = m;
                if (!eye[k].delta) { fIdx = k; break; }
                if (r == 1 && k < 64) prefJ |= (1ull << k);
            }
            routeR = (r < 0) ? c_routeDefault : r;
        }
        const double wD = routeOn ? 1.0 / c_routeSppD : 1.0;   // per-mode sample weights
        const double wJ = routeOn ? 1.0 / c_routeSppJ : 1.0;

        Real cx = cieX(lambda), cy = cieY(lambda), cz = cieZ(lambda);
        Real cxS[SECN], cyS[SECN], czS[SECN];
        for (int i = 0; i + 1 < hb.C; ++i) {
            cxS[i] = cieX(hb.lam[i + 1]); cyS[i] = cieY(hb.lam[i + 1]); czS[i] = cieZ(hb.lam[i + 1]);
        }
        // Direct view of a distant `light sun`. A sun is a delta-DIRECTION emitter with no
        // geometry, so dMisWeight drops the s=0 strategy for it and no connection strategy
        // can reach it either — without this the solar disc itself, and every mirror/water
        // glint of it, would simply be missing from mode D. It is added only when the
        // escaping ray came through camera + delta (specular) vertices ONLY, which is
        // exactly the case where no other strategy competes: NEE needs a connectible
        // (non-delta) vertex, and so does every s>=2 connection or t==1 splat. The MIS
        // weight is therefore exactly 1. Mirrors BdptRenderer::renderRows.
        if (esc.escaped) {
            bool allDelta = true;
            for (int i = 1; i < nE; ++i) if (!eye[i].delta) { allDelta = false; break; }
            const bool escOk = !routeOn || ((routeR == 1) ? allowJ : allowD);
            const double wEsc = (routeOn && routeR != 1) ? wD : wJ;
            if (allDelta && escOk) {
                for (int k = 0; k < sc.nEmitters; ++k) {
                    const DEmitter& em = sc.emitters[k];
                    if (em.shape != 6 || !dInSunCone(em, esc.dir)) continue;
                    double e0 = esc.beta * (double)specLookup(em.emitSpd, lambda) * hb.invPdf[0];
                    double ax = cx * e0, ay = cy * e0, az = cz * e0;
                    for (int i = 0; i + 1 < esc.nUp; ++i) {
                        double ek = esc.betaSec[i] *
                                    (double)specLookup(em.emitSpd, hb.lam[i + 1]) * hb.invPdf[i + 1];
                        ax += cxS[i] * ek; ay += cyS[i] * ek; az += czS[i] * ek;
                    }
                    if (esc.nUp > 1) { double inv = 1.0 / esc.nUp; ax *= inv; ay *= inv; az *= inv; }
                    ax *= wEsc; ay *= wEsc; az *= wEsc;
                    size_t o = ((size_t)py * resX + px) * 3;
                    atomicAdd(&camFilm[o + 0], ax);
                    atomicAdd(&camFilm[o + 1], ay);
                    atomicAdd(&camFilm[o + 2], az);
                }
            }
        }
        // s = 1 draws a light point of its own (dConnectBDPT reads no light[] for it), so it runs
        // whatever the light subpath drew: an empty one must not take the s = 1 sample with it.
        // Host twin: BdptRenderer::renderRows.
        const int sMax = (nL > 0) ? nL : 1;
        for (int t = 1; t <= nE; ++t)
            for (int s = 0; s <= sMax; ++s) {
                int depth = t + s - 2;
                if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth) continue;
                if ((c_dbgOnlyS >= 0 && s != c_dbgOnlyS) || (c_dbgOnlyT >= 0 && t != c_dbgOnlyT)) continue;
                int spx = 0, spy = 0, isSplat = 0, nUpConn = 0;
                double Lsec[SECN];
                // A D-routed path takes mode D's weight: both merge constants zero. A t = 1 splat's
                // chain is the light vertex it connects to the lens; every t >= 2 path shares routeR.
                double mkST = mergeKappa, ksST = kappaSurf, wST = 1.0;
                if (routeOn) {
                    const int rt = (t >= 2) ? routeR : ((s >= 1) ? dRouteOfVert(sc, light[s - 1]) : routeR);
                    if (rt != 1) { if (!allowD) continue; mkST = 0.0; ksST = 0.0; wST = wD; }
                    else         { if (!allowJ) continue; wST = wJ; }
                }
                double c = (c_jHalf == 2) ? 0.0 : dConnectBDPT(sc, camS, light, eye, lightSec, eyeSec, NS,
                                        s, t, hb, rng, spx, spy, isSplat, Lsec, nUpConn,
                                        mkST, ksST, geoNow ? geoE : nullptr, geoNow ? geoL : nullptr);
                if (wST != 1.0) { c *= wST; for (int i = 0; i + 1 < nUpConn; ++i) Lsec[i] *= wST; }
                if (nUpConn <= 0) continue;
                // Reject a non-positive — and, critically, a NON-FINITE — contribution before
                // it reaches the film, exactly as BdptRenderer::renderRows does. Negated form
                // on purpose: NaN compares false against everything, so `mx <= 0.0` would let
                // an undefined connection through to atomicAdd and permanently poison the
                // pixel. Bit-identical for every finite value (adding 0 is a no-op).
                {
                    double mx = c;
                    for (int i = 0; i + 1 < nUpConn; ++i) if (Lsec[i] > mx) mx = Lsec[i];
                    if (!(mx > 0.0)) continue;
                }
                // The ×C de-hero boost is applied ONCE here, as 1/min(nUp_light, nUp_eye):
                // folding it into either subpath's throughput would square it whenever both
                // sides stayed multi-λ. nUpConn == 1 reproduces the scalar accumulation exactly.
                double ax = cx * c, ay = cy * c, az = cz * c;
                for (int i = 0; i + 1 < nUpConn; ++i) {
                    ax += cxS[i] * Lsec[i]; ay += cyS[i] * Lsec[i]; az += czS[i] * Lsec[i];
                }
                if (nUpConn > 1) {
                    double inv = 1.0 / nUpConn; ax *= inv; ay *= inv; az *= inv;
                }
                if (isSplat) {
                    size_t o = ((size_t)spy * resX + spx) * 3;
                    atomicAdd(&splatFilm[o + 0], ax);
                    atomicAdd(&splatFilm[o + 1], ay);
                    atomicAdd(&splatFilm[o + 2], az);
                } else {
                    size_t o = ((size_t)py * resX + px) * 3;
                    atomicAdd(&camFilm[o + 0], ax);
                    atomicAdd(&camFilm[o + 1], ay);
                    atomicAdd(&camFilm[o + 2], az);
                }
            }

        // ---- MERGES (mode J / UPBP) -------------------------------------------------
        // The other half of the estimator: every photon beam whose kernel this camera
        // subpath passed through. Unlike a connection this needs no light subpath at all —
        // the beam map IS a cache of light subpaths, traced once for the whole frame —
        // which is exactly why it reaches where a connection cannot: deep inside a thick
        // medium the camera's own free-flight sampling essentially never lands on the
        // scattering point a connection would have to be made from, whereas a beam is a
        // whole LINE of deposited power and is hit by merely passing near it.
        //
        // The gather returns XYZ built at the BEAMS' wavelengths, not at this sample's hero
        // λ (the standard spectral-photon-mapping estimate); `sg.beta` is this sample's own
        // hero throughput to the segment origin, a scalar multiplying the XYZ triple.
        //
        // Placed after every connection deliberately: the gather draws from `rng`
        // (ratio-tracked transmittance in a heterogeneous medium), and running it here means
        // those draws cannot shift the connection half of the SAME sample. They cannot shift
        // the next sample either, because the stream is re-seeded per (pixel,sample) at the
        // top of this loop — so "turn the merges off and mode J is mode D bit-for-bit"
        // survives even though the two halves share a generator. Host twin: renderRows.
        // Either merge kind opens this region. `mergeOn` needs a BEAM map, so gating the
        // whole block on it silently dropped the point merges on any scene without media --
        // exactly the scene the point merges exist for.
        // Every merge makes a J-routed path (see c_routeOn), so a sub-sample that does not draw
        // mode J does none of them; one that does weights them wJ.
        if ((mergeOn || kappaSurf > 0.0) && c_jHalf != 1 && allowJ) {
            // The camera half of every merge weight, replayed ONCE for the whole subpath.
            // dMisWeight's camera loop telescopes inward from the merge point; everything it
            // accumulates strictly camera-side of eye[k] is independent of WHERE along the
            // segment a beam is hit, so it is summed here and read off per segment.
            double segSumC[SEGN], segSumM[SEGN];
            // UPBP-W-CHROMATIC (DChromaRun): where some medium's extinction varies with
            // wavelength, every merge weight's camera half is read at the merged light's.
            // Only a merging kernel holds the tables (DChromaRunNone); with them, the
            // camera-wavelength segSumM below goes unread and is not summed.
            const bool chromaOn = MERGE && c_jTrSpec && c_jChroma && sc.chromaSig &&
                                  (mergeKappa > 0.0 || kappaSurf > 0.0);
            const DPatEnv envC = dPatEnvOf(sc);
            typename std::conditional<MERGE, DChromaRun, DChromaRunNone>::type crun;
            if (chromaOn) crun.reset();
            double cbuf[MERGE ? 2 * DCHROMA_K : 1];   // this segment's segSumM and etaKCoef bins
            if (mergeKappa > 0.0 || kappaSurf > 0.0) {
                for (int k = 0; k < nE && k < (int)SEGN; ++k) segSumC[k] = segSumM[k] = 0.0;
                for (int k = 1; k < nE && k < (int)SEGN; ++k) {
                    const double gate = (!eye[k].delta && !eye[k - 1].delta) ? 1.0 : 0.0;
                    // eta' of the merge AT eye[k-1]; needs both its neighbours, so it starts
                    // at k = 2. Its pdfRev is the RECORDED one: moving the light-side
                    // neighbour of eye[k] along the same ray does not change the direction
                    // arriving at eye[k-1].
                    // Both kinds, each scaled by its OWN kappa here rather than factored
                    // out of the sum (bdpt.h: mergeEtaPrime(...).scale(mk)) -- with two kinds
                    // there is no single kappa to factor. A vertex is a medium point or a
                    // surface, never both, so exactly one of these is non-zero.
                    double eK = 0.0;
                    if (k >= 2 && !chromaOn) {
                        const double plK = dMergeLightDensity((double)eye[k - 1].pdfRev,
                                                              eye[k].delta != 0);
                        if (mergeKappa > 0.0) eK = mergeKappa * dMergeEtaAt(geoE[k - 1], plK);
                        if (kappaSurf > 0.0)
                            eK += kappaSurf * dMergeEtaSurf(dSurfMergeSite(sc, eye[k - 1]), plK);
                    }
                    double carry = 0.0, carryM = 0.0;
                    if (k >= 2) {
                        // No ratio exists at the camera vertex itself, which is why the
                        // recurrence starts one step in.
                        const double rC = dMisRevDensity(eye[k - 1].pdfRev, eye[k].delta != 0) /
                                          dMisRemap0(eye[k - 1].pdfFwd);
                        if (rC != 0.0) {              // 0 carries nothing (no inf * 0)
                            carry  = rC * segSumC[k - 1];
                            carryM = rC * segSumM[k - 1];
                        }
                    }
                    // A strand is never a light vertex (dRandomWalk, 0.365.0): every camera-side
                    // alternative from eye[k] inward puts it there (bdpt.h cameraSideSums' fiberK).
                    const bool fiberK = !eye[k].delta && eye[k].type == BV_SURFACE &&
                                        eye[k].matId >= 0 && sc.mats[eye[k].matId].type == D_HAIR;
                    segSumC[k] = fiberK ? 0.0 : gate + carry;
                    segSumM[k] = fiberK ? 0.0 : eK + carryM;
                }
            }
            for (int i = 0; beamsHere && i < nSegs; ++i) {
                const DPathSeg& sg = segs[i];
                if (!(sg.beta > 0.0)) continue;
                if (sg.vert >= maxDepth) continue;   // every merge past the depth cap (host twin)
                // The merged path's route (see routeR above): past the chain it is this subpath's;
                // on a chain segment it is the prefix's, or the hit medium's if the prefix is D.
                int routeMedJ = 0;
                if (routeOn) {
                    if (sg.vert >= fIdx) { if (routeR != 1) continue; }
                    else if (!(sg.vert >= 1 && sg.vert < 64 && ((prefJ >> sg.vert) & 1ull))) {
                        if (!c_routeAnyMedJ) continue;
                        routeMedJ = 1;
                    }
                }
                // Hoisted out of the per-hit weight: the ray/bounds clip and the spectral
                // sigma_t lookup are constants of the SEGMENT, and a dense medium hands one
                // segment hundreds of hits. Built unconditionally so the gather's `camTr` is
                // never dangling — cheap (one clip per medium).
                DTrRay camTr;
                camTr.build(sc, sg.o, sg.d, sg.tMax, lambda, chromaOn);
                DBeamMergeW mw;
                mw.kappa = mergeKappa; mw.lamCam = (double)lambda; mw.pdfDirCam = sg.pdfDir;
                if (mergeKappa > 0.0) {
                    const int k = (sg.vert < nE) ? sg.vert : nE - 1;
                    const DVertex& vk = eye[k < 0 ? 0 : k];
                    mw.gateS1     = vk.delta ? 0.0 : 1.0;
                    mw.cosFacK    = (vk.type == BV_SURFACE || vk.type == BV_LIGHT)
                                  ? fabs((double)ddot(vk.ns, sg.d)) : 1.0;
                    mw.invPdfFwdK = 1.0 / dMisRemap0(vk.pdfFwd);
                    mw.segSumC    = (k >= 0 && k < (int)SEGN) ? segSumC[k] : 0.0;
                    mw.segSumM    = (k >= 0 && k < (int)SEGN) ? segSumM[k] : 0.0;
                    mw.kappaS     = kappaSurf;
                    // The POINT merge AT eye[k]: unlike its beam twin there is no geometry to
                    // gather, so the whole merge-point-independent coefficient IS kappaS.
                    mw.etaKSurf   = (kappaSurf > 0.0 && dSurfMergeSite(sc, vk)) ? kappaSurf : 0.0;
                    // The depth cap, the same one the connection loop above applies: a merge
                    // at light vertex j on this segment makes a path of depth j + k + 1.
                    mw.camVert    = k;
                    mw.maxDepth   = maxDepth;
                    if (chromaOn) for (int b = 0; b < DCHROMA_K; ++b) cbuf[DCHROMA_K + b] = 0.0;
                    // The merge AT eye[k]. Its light-side incoming direction is -sg.d
                    // whatever x turns out to be, so sin(theta) and the whole coefficient are
                    // merge-point INDEPENDENT and belong here, not in the per-hit weight.
                    if (k >= 1 && vk.type == BV_MEDIUM &&
                        vk.mediumId >= 0 && vk.mediumId < sc.mediaN &&
                        dJBeamMerges(sc, sc.media[vk.mediumId], vk.p)) {
                        DVec3 dp = eye[k - 1].p - vk.p;
                        const double dn = sqrt(ddot(dp, dp));
                        if (dn > 0.0) {
                            dp = dp * (Real)(1.0 / dn);
                            const double cc = ddot(sg.d, dp);
                            const double s2 = 1.0 - cc * cc;
                            const DMedium& md = sc.media[vk.mediumId];
                            const DPatEnv env = dPatEnvOf(sc);
                            const double sT = (double)medSigmaT(md, lambda) *
                                              dMedDensityAt(md, vk.p, env);
                            const double tr = dTrDet(sc, vk.p, dp, dn, lambda, env);
                            if (s2 > 0.0 && sT > 0.0 && tr > 0.0)
                                mw.etaKCoef = sqrt(s2) / (sT * tr);
                            if (chromaOn)
                                dChromaBeamEta1(sc, vk, sg.d, dp, dn, env, cbuf + DCHROMA_K);
                        }
                    }
                    if (chromaOn) {
                        crun.advanceTo(k < 0 ? 0 : k, sc, eye, mergeKappa, kappaSurf, envC);
                        for (int b = 0; b < DCHROMA_K; ++b) cbuf[b] = crun.m[b];
                        mw.chroma = cbuf;
                    }
                }
                // WAVEFRONT (UPBP-CONV): hand the segment -- and the weight state just built for
                // it -- to the queue; kWfBeamHits / kWfBeamEval do the gather after this kernel.
                // A full queue means this segment is gathered inline, right here, and so does a
                // chromatic one when the queue has no table for its bins -- each hit from the
                // stream the queue would give it (`segSeed`), so which segments the queue took,
                // a race, moves only the order of the film's sums.
                // The segment's own stream: its sample and its index along the walk, under the
                // kernel's seed (which carries -seed), so it is the same however the queue's slots
                // were handed out.
                const unsigned long long segSeed =
                    ((unsigned long long)gidx * 1315423911ull + (unsigned long long)i * 2654435761ull) ^ seedBase;
                bool queued = false;
                if (wq.segs && (!mw.chroma || wq.chroma)) {
                    const int si = atomicAdd(wq.nSegs, 1);
                    if (si < wq.segCap) {
                        DWfSeg& ws = wq.segs[si];
                        ws.o = sg.o; ws.d = sg.d; ws.tMax = sg.tMax; ws.aGlass = sg.aGlass;
                        ws.beta = sg.spec.beta(sg.beta) * wJ;   // JSPEC: the carrier's scale folded in
                        ws.routeMedJ = routeMedJ;
                        ws.mw = mw; ws.tr = camTr; ws.px = px; ws.py = py; ws.lambda = (float)lambda;
                        if (mw.chroma) {   // the bins move into the queue's table with the segment
                            double* q = wq.chroma + (size_t)si * 2 * DCHROMA_K;
                            for (int b = 0; b < 2 * DCHROMA_K; ++b) q[b] = cbuf[b];
                            ws.mw.chroma = q;
                        }
                        ws.spec = sg.spec;   // JSPEC (any == 0 when the carrier is off)
                        ws.glassMat = sg.glassMat;
                        ws.seed = segSeed;
                        queued = true;
                    } else {
                        atomicAdd(&wq.overflow[0], 1);
                    }
                }
                if (!queued) {
                    double mX = 0.0, mY = 0.0, mZ = 0.0;
                    dGatherPhotonBeams(sc, bm, sg.o, sg.d, (Real)sg.tMax, sg.aGlass, rng,
                                       mX, mY, mZ, &mw, &camTr, routeMedJ, &sg.spec, sg.glassMat, &cam,
                                       &segSeed, &work);
                    if (mX != 0.0 || mY != 0.0 || mZ != 0.0) {
                        size_t o = ((size_t)py * resX + px) * 3;
                        const double bJ = sg.spec.beta(sg.beta) * wJ;   // JSPEC scale
                        atomicAdd(&camFilm[o + 0], mX * bJ);
                        atomicAdd(&camFilm[o + 1], mY * bJ);
                        atomicAdd(&camFilm[o + 2], mZ * bJ);
                    }
                }
            }

            // ---- POINT MERGES (`-jsurf`): the half folded in from mode U ------------------
            // Per VERTEX, not per segment -- a point merge happens where the camera walk
            // actually landed. Its one ray-traced part is the footprint's probe, whose seed it draws
            // from the walk's stream at every site with the footprint on (dSurfFootprint).
            if (kappaSurf > 0.0 && sm.nPts > 0 && (!routeOn || routeR == 1)) {
                const double vmNorm = 1.0 / kappaSurf;
                const int kEnd = (nE < maxDepth + 1) ? nE : maxDepth + 1;   // past it every merge is capped
                // -checkjsurfw: this sample's subpath and each half below, as built here.
                const bool tapped = MERGE && c_surfTap.n > 0 && sub == 0 && gidx < c_surfTap.n;
                if (tapped) dSurfTapPath(gidx, eye, nE, (double)lambda);
                if (chromaOn) crun.reset();
                for (int k = 1; k < kEnd; ++k) {
                    if (chromaOn) crun.advanceTo(k, sc, eye, mergeKappa, kappaSurf, envC);
                    const DVertex& vk = eye[k];
                    if (!dSurfMergeSite(sc, vk) || !(vk.beta > 0.0)) continue;
                    const DVertex& vp = eye[k - 1];
                    DVec3 woCam{vp.p.x - vk.p.x, vp.p.y - vk.p.y, vp.p.z - vk.p.z};
                    const double d2 = ddot(woCam, woCam);
                    if (!(d2 > 0.0)) continue;
                    const double invD = 1.0 / sqrt(d2);
                    woCam = woCam * (Real)invD;
                    DSurfMergeW sw;
                    sw.kappaB       = mergeKappa;
                    sw.kappaS       = kappaSurf;
                    sw.invPdfFwdK   = 1.0 / dMisRemap0(vk.pdfFwd);
                    sw.gateD1       = vp.delta ? 0.0 : 1.0;
                    sw.cosPrev      = dOnSurface(vp) ? fabs(ddot(vp.ns, woCam)) : 1.0;
                    sw.invDistPrev2 = 1.0 / d2;
                    sw.invPdfFwdKm1 = 1.0 / dMisRemap0(vp.pdfFwd);
                    // The merge AT eye[k-1], less its pdfRev. pLight = 1 turns the primed eta
                    // into the bare coefficient, which is legitimate HERE (and not in the beam
                    // gather) because both of eye[k-1]'s neighbours are known: the merge site
                    // is eye[k] itself, so there is no per-photon geometry.
                    sw.etaKm1Coef   = 0.0;
                    if (k >= 2) {
                        if (mergeKappa > 0.0) sw.etaKm1Coef = mergeKappa * dMergeEtaAt(geoE[k - 1], 1.0);
                        if (dSurfMergeSite(sc, vp)) sw.etaKm1Coef += kappaSurf;
                    }
                    sw.segSumC      = (k - 1 < (int)SEGN) ? segSumC[k - 1] : 0.0;
                    sw.segSumM      = (k - 1 < (int)SEGN) ? segSumM[k - 1] : 0.0;
                    if (chromaOn) { sw.chromaM = crun.mPrev; sw.chromaKm1 = crun.e1; }
                    sw.camVert      = k;
                    sw.maxDepth     = maxDepth;
                    if (tapped) dSurfTapHalf(gidx, k, sw);
                    // JSPEC: the carrier arriving at eye[k] (the walk's snapshot), coin included;
                    // eye[k]'s own BSDF is evaluated at the photon's wavelength by the merge.
                    const DJSpec* vks = (jSpecOn && vspec[k].on()) ? &vspec[k] : nullptr;
                    const double f = (vks ? vks->beta(vk.beta) : vk.beta) * vmNorm * wJ;   // JSPEC scale
                    const size_t o = ((size_t)py * resX + px) * 3;
                    // The footprint probe's stream, off the walk's (dSurfFootprint).
                    unsigned long long gaSeed = 0;
                    if (sc.gatherArea) gaSeed = rng.next64();
                    // Queued for kWfSurfGather (DSurfQueue); gathered inline, and counted in
                    // `overflow`, where the queue is full. renderBdptCuda allocates the JSPEC
                    // carrier and chromatic tables exactly when c_jSpecOn and chromaOn's switches
                    // call for them: `tables` holds that out loud, a site it turned away being
                    // counted, which check_surf_wavefront fails. nReqs counts the sites turned
                    // away as well (they size the next wave), so past 2^31 of them it wraps
                    // negative: qslot >= 0.
                    if (MERGE && sq.reqs) {
                        const bool tables = (!chromaOn || sq.chroma) && (!vks || sq.spec);
                        const int qslot = tables ? atomicAdd(sq.nReqs, 1) : -1;
                        if (qslot >= 0 && qslot < sq.cap) {
                            DSurfReq& rq = sq.reqs[qslot];
                            rq.vk = vk;
                            rq.sw = sw;
                            rq.sw.chromaM = nullptr; rq.sw.chromaKm1 = nullptr;
                            rq.woCam = woCam;
                            rq.f = f;
                            rq.pix = (long long)o;
                            rq.gaSeed = gaSeed;
                            rq.flags = 0;
                            if (vks) { sq.spec[qslot] = *vks; rq.flags |= 1; }
                            if (chromaOn) {
                                double* ch = sq.chroma + (size_t)qslot * 2 * DCHROMA_K;
                                for (int b = 0; b < DCHROMA_K; ++b) {
                                    ch[b] = crun.mPrev[b];
                                    ch[DCHROMA_K + b] = crun.e1[b];
                                }
                                rq.flags |= 2;
                            }
                            continue;
                        }
                        atomicAdd(sq.overflow, 1);
                    }
                    double gX = 0.0, gY = 0.0, gZ = 0.0;
                    work += (unsigned long long)dSurfMergeAt(sc, sm, vk, woCam, sw, gX, gY, gZ, vks);
                    if (gX != 0.0 || gY != 0.0 || gZ != 0.0) {
                        const double fa = f * (sc.gatherArea ? dSurfFootprint(sc, DSurfSite(sc, sm, vk).fp, gaSeed)
                                                             : 1.0);
                        atomicAdd(&camFilm[o + 0], gX * fa);
                        atomicAdd(&camFilm[o + 1], gY * fa);
                        atomicAdd(&camFilm[o + 2], gZ * fa);
                    }
                }
            }
        }
        }   // sub-samples (per-mode sample counts)
    }
    if (MERGE) dAddWork(work);
}

// Every variant render_cuda.cu launches or asks the attributes of (its declaration there).
template __global__ void kBdptT<0, BDPT_MAXDEPTH, false, false>(
    DScene, const __grid_constant__ DCamera, double*, double*, long long, long long, long long, long long, int, int, int,
    unsigned long long, int, DBeamMap, DSurfMap, DWfQueue, long long, long long, DSurfQueue);
template __global__ void kBdptT<0, BDPT_DEEPDEPTH, false, false>(
    DScene, const __grid_constant__ DCamera, double*, double*, long long, long long, long long, long long, int, int, int,
    unsigned long long, int, DBeamMap, DSurfMap, DWfQueue, long long, long long, DSurfQueue);
template __global__ void kBdptT<0, BDPT_MAXDEPTH, true, false>(
    DScene, const __grid_constant__ DCamera, double*, double*, long long, long long, long long, long long, int, int, int,
    unsigned long long, int, DBeamMap, DSurfMap, DWfQueue, long long, long long, DSurfQueue);
template __global__ void kBdptT<0, BDPT_DEEPDEPTH, true, false>(
    DScene, const __grid_constant__ DCamera, double*, double*, long long, long long, long long, long long, int, int, int,
    unsigned long long, int, DBeamMap, DSurfMap, DWfQueue, long long, long long, DSurfQueue);
template __global__ void kBdptT<0, BDPT_MAXDEPTH, true, true>(
    DScene, const __grid_constant__ DCamera, double*, double*, long long, long long, long long, long long, int, int, int,
    unsigned long long, int, DBeamMap, DSurfMap, DWfQueue, long long, long long, DSurfQueue);
template __global__ void kBdptT<0, BDPT_DEEPDEPTH, true, true>(
    DScene, const __grid_constant__ DCamera, double*, double*, long long, long long, long long, long long, int, int, int,
    unsigned long long, int, DBeamMap, DSurfMap, DWfQueue, long long, long long, DSurfQueue);
template __global__ void kBdptT<BDPT_NSEC, BDPT_MAXDEPTH, false, false>(
    DScene, const __grid_constant__ DCamera, double*, double*, long long, long long, long long, long long, int, int, int,
    unsigned long long, int, DBeamMap, DSurfMap, DWfQueue, long long, long long, DSurfQueue);
template __global__ void kBdptT<BDPT_NSEC, BDPT_DEEPDEPTH, false, false>(
    DScene, const __grid_constant__ DCamera, double*, double*, long long, long long, long long, long long, int, int, int,
    unsigned long long, int, DBeamMap, DSurfMap, DWfQueue, long long, long long, DSurfQueue);

}  // namespace gpu
