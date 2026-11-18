// kJSurfLightT, mode J's surface light pass, in a translation unit of its own: compiled under
// -maxrregcount at its launch bounds (CMakeLists.txt), as render_cuda_bdpt.cu is kBdptT's
// (render_cuda_dev.cuh says why). render_cuda.cu launches it.
#define FT_FAMILY_TU 1
#include "render_cuda_dev2.cuh"

namespace gpu {

template <int MAXD>
__global__ void __launch_bounds__(128, 4)
kJSurfLightT(DScene sc, DCamera cam, int diffraction, int maxDepth,
             long long nPaths, unsigned long long seedBase, DJSurfOut out) {
    D_MOTION_CLEAR();   // this thread's time: the launch's (passU) until a path sets one
    enum { MAXV = BDPT_MAXV_OF(MAXD) };
    if (maxDepth > MAXD) maxDepth = MAXD;
    const DPatEnv env = dPatEnvOf(sc);
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long G = (long long)gridDim.x * blockDim.x;
    unsigned long long work = 0;   // the vertices traced, mode J's camera-side count (g_dWork)
    for (long long i = g; i < nPaths; i += G) {
        // Seeded by ABSOLUTE subpath index, exactly as the host pass is ("so the map is
        // identical for any thread count"), so the realization depends on (index, salt) and
        // not on the launch geometry.
        DRng rng; rng.seedUnit((unsigned long long)i, seedBase);
        DHeroBundle hb;
        double pdfLam = 0.0;
        hb.lam[0] = dSampleSceneLambda(sc, rng, pdfLam);
        if (pdfLam <= 0.0) continue;
        hb.invPdf[0] = dInvPdfLambda(sc, hb.lam[0]);
        hb.C = 1;                       // single-lambda always: the host pass is too
        const Real lambda = hb.lam[0];
        DVertex path[MAXV];
        double   pathSec[1];            // secStride 0: never written, never read
        float    compoundMergeW[MAXV];  // a compound's merge vertex's share, 0 elsewhere (COMPOUND-MERGE-PICK)
        for (int k = 0; k < MAXV; ++k) compoundMergeW[k] = 0.0f;
        const int np = dGenLightSubpath(sc, cam, diffraction, hb, maxDepth + 1, rng,
                                        path, pathSec, 0, MAXV, compoundMergeW);
        if (np <= 0) continue;
        work += (unsigned long long)np;
        const Real cx = cieX(lambda), cy = cieY(lambda), cz = cieZ(lambda);
        // The recurrence, carried as scalars at index u-1 (see the note above).
        double accC = 0.0, accMb = 0.0, accMs = 0.0;
        for (int u = 0; u < np; ++u) {
            const bool dPrev = (u > 0) ? (path[u - 1].delta != 0)
                                       : (dIsDeltaLightVertex(sc, path[0]) != 0);
            const double gate = (!path[u].delta && !dPrev) ? 1.0 : 0.0;
            if (u == 0) { accC = gate; continue; }
            const double rL = dMisRevDensity(path[u - 1].pdfRev, path[u].delta != 0) /
                              dMisRemap0(path[u - 1].pdfFwd);
            // The merge AT y_{u-1}, which the recurrence would not fold in until the next
            // iteration but which this vertex's stored partials need now. Both kinds, because
            // a denominator that carries one is not a partition of unity; at most one is
            // non-zero (a vertex is in a medium or on a surface, never both).
            double eBeam = 0.0, eSurf = 0.0;
            if (u >= 2) {
                const double plP = dMergeLightDensity((double)path[u - 1].pdfFwd, path[u - 2].delta != 0);
                eBeam = dMergeEtaPrime(sc, path[u - 2].p, path[u - 1], path[u].p, plP, lambda, env);
                eSurf = dMergeEtaSurf(dSurfMergeSite(sc, path[u - 1]), plP);
            }
            // A photon that arrived through glass (pdfFwd 0 after a delta y_{u-1}) is stored too
            // since 0.387.0, with that 0 read as 1 (host: traceLightBeamPass's store; MODE-J-SDS).
            const double plU = dMergeLightDensity((double)path[u].pdfFwd, path[u - 1].delta != 0);
            // A compound's merge vertex is a photon whatever lobe the pick took (host twin).
            const bool compoundSite = compoundMergeW[u] > 0.0f && path[u].type == BV_SURFACE &&
                                      !(path[u].fiberR > (Real)0);
            if ((dSurfMergeSite(sc, path[u]) || compoundSite) && plU > 0.0 && path[u].beta > 0.0) {
                DVec3 dp = path[u - 1].p - path[u].p;
                const double rho2 = ddot(dp, dp);
                if (rho2 > 0.0) {
                    const double rho = sqrt(rho2);
                    dp = dp * (Real)(1.0 / rho);
                    // The cosine of pdfRev*(y_{u-1}), at y_{u-1}, facing back along this same
                    // edge; 1 at a medium vertex, which has no normal.
                    const bool prevOnSurf = (path[u - 1].type == BV_SURFACE ||
                                             path[u - 1].type == BV_LIGHT);
                    const double cosPrev = prevOnSurf ? fabs(ddot(path[u - 1].ns, dp)) : 1.0;
                    const int slot = atomicAdd(out.count, 1);
                    if (slot < out.cap) {
                        DSurfPhoton& ph = out.pts[slot];
                        ph.p      = path[u].p;
                        ph.wo     = dp;                 // unit, toward the previous vertex
                        ph.lambda = (float)lambda;
                        ph.beta   = (float)(path[u].beta * (compoundSite ? (double)compoundMergeW[u] : 1.0));
                        ph.cx = (float)cx; ph.cy = (float)cy; ph.cz = (float)cz;
                        ph.misIdx = (unsigned)slot;     // appended 1:1, so index == slot
                        {   // its surface's true normal, to its light's side (host twin vertexGeoN)
                            const DVec3 pn = path[u].back ? path[u].ng * (Real)-1 : path[u].ng;
                            ph.nx = (float)pn.x; ph.ny = (float)pn.y; ph.nz = (float)pn.z;
                        }
                        DSurfMis& sm = out.mis[slot];
                        sm.sumC    = accC;
                        sm.sumMb   = accMb + eBeam;
                        sm.sumMs   = accMs + eSurf;
                        sm.pdfFwdA = plU;                 // a delta y_{u-1}'s 0 reads 1
                        sm.rCoef   = (float)(cosPrev /
                                             (rho2 * dMisRemap0(path[u - 1].pdfFwd)));
                        // The merged path scatters at y_u through the camera's lobe (host twin).
                        sm.gateC1  = (unsigned char)(dPrev ? 0 : 1);
                        sm.vert    = (unsigned short)u;
                    }
                }
            }
            const double cPrev = accC, mbPrev = accMb, msPrev = accMs;
            // A ratio of 0 (dMisRevDensity) carries nothing: tested, not multiplied (no inf * 0).
            accC  = gate + (rL != 0.0 ? rL * cPrev : 0.0);
            accMb = rL != 0.0 ? rL * (mbPrev + eBeam) : 0.0;
            accMs = rL != 0.0 ? rL * (msPrev + eSurf) : 0.0;
        }
    }
    dAddWork(work);
}

// Forced instantiation of both depth variants (BDPT_MAXDEPTH / BDPT_DEEPDEPTH, the same pair
// kBdptT ships). Without this an unreferenced template is not compiled at all, so the step
// that "adds the kernel only" would compile trivially and prove nothing -- which is the exact
// opposite of the point.
template __global__ void kJSurfLightT<BDPT_MAXDEPTH>(
    DScene, DCamera, int, int, long long, unsigned long long, DJSurfOut);
template __global__ void kJSurfLightT<BDPT_DEEPDEPTH>(
    DScene, DCamera, int, int, long long, unsigned long long, DJSurfOut);

}  // namespace gpu
