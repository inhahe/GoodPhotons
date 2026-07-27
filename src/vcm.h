// Vertex Connection and Merging (VCM / UPS) — mode 'U'. ROADMAP item 3.
//
// VCM (Georgiev et al. 2012) is the "have it all" unbiased estimator: it runs BOTH
// bidirectional vertex CONNECTIONS (the BDPT strategies — mode D) AND photon-map
// vertex MERGING (density estimation reinterpreted as an extra sampling technique),
// and MIS-combines every strategy under one balance heuristic. Connections resolve
// what merging is bad at (diffuse/glossy interreflection connected directly) and
// merging resolves what connections are bad at (caustics / SDS focusing), so the
// combined estimator is robust across diffuse GI, glossy, and specular-caustic light
// transport all at once. This is item (1)'s photon map + item (2)'s progressive radius
// glued onto item (D)'s connections.
//
// Implementation follows SmallVCM's compact balance-heuristic bookkeeping: instead of
// PBRT/bdpt.h's explicit per-vertex pdfFwd/pdfRev ratio loop, each subpath carries
// three running partial-MIS quantities updated recursively at every vertex:
//   dVCM — the connect-to-this-vertex weight accumulator
//   dVC  — the vertex-connection weight accumulator
//   dVM  — the vertex-merging weight accumulator
// A connection or merge then combines the stored light-vertex quantities with the live
// camera-vertex quantities into a single balance-heuristic weight. See Georgiev 2012
// §4 / the SmallVCM `VertexCM` reference for the derivation.
//
// PER ITERATION (one "pass"):
//   1. Trace nLightPaths light subpaths (one per pixel). Store every connectible
//      (non-delta) surface vertex into a flat array + a uniform hash grid, keyed by
//      path index so a camera path can connect to its PAIRED light path. Each stored
//      vertex also carries dVCM/dVC/dVM. During the walk, connect each light vertex to
//      the camera (the light-image / t=1 splat).
//   2. Build the hash grid over the stored light vertices (cell = merge radius).
//   3. Trace nLightPaths camera subpaths (one per pixel). At every camera vertex do:
//        (a) hit emitter directly            (s=0 emission, MIS-weighted)
//        (b) NEE — connect to a light source (s=1, MIS-weighted)
//        (c) connect to the PAIRED light subpath's stored vertices (vertex connection)
//        (d) merge — gather nearby light vertices from ALL paths (vertex merging)
//   Accumulate; shrink the radius across passes (r_i = r0 * i^((alpha-1)/2)); average
//   over passes. A single pass with merging disabled reduces to BDPT (mode D).
//
// SPECTRAL NOTE. This renderer is single-wavelength spectral MC: each sample carries one
// lambda. VCM's CONNECTION strategies pair a camera subpath with its OWN light subpath,
// so both share one lambda per path index — connections are EXACT (like BDPT). The MERGE
// strategy, however, gathers light vertices from OTHER paths (each at its own lambda), so
// like the photon map (modes M/S) it builds the density estimate directly in XYZ, using
// cie(lambda_lightvertex) per merged vertex and the camera material's reflectance at that
// lambda. This is the standard spectral-photon-mapping approximation (the camera-side
// throughput is carried at the camera lambda); crucially the MIS pdfs in this renderer
// are WAVELENGTH-INDEPENDENT (diffuse cosine / glossy lobe densities don't depend on
// lambda — only the throughput/BSDF VALUES do), so the balance-heuristic weights stay a
// consistent partition of unity across the mixed wavelengths.
//
// SCOPE. Surfaces only: Diffuse / Glossy / DiffuseTransmit connect and merge; the
// specular family (Mirror/Dielectric/HalfMirror/Filter/ThinFilm/Multilayer/Grating)
// forms delta chains. Area/sphere (Lambertian) lights. Rectilinear pinhole camera.
// NOT handled (guarded in main.cpp): participating media, fluorescence, layered
// materials, spot/env/collimated lights, fisheye projections, realistic lenses.
#pragma once
#include <vector>
#include <thread>
#include <cstdint>
#include <cmath>
#include "scene.h"
#include "camera.h"
#include "render.h"
#include "bdpt.h"        // bsdfF, bsdfPdf, isConnectibleMat, isTwoSidedMat, offsetOrigin, glossyExponent
#include "color.h"

namespace vcm {

using bdpt::bsdfF;
using bdpt::bsdfPdf;
using bdpt::isConnectibleMat;
using bdpt::isTwoSidedMat;

// Offset a connection/shadow-ray origin off a surface along the geometric normal, flipped
// to the same side as `dir` (PBRT's OffsetRayOrigin; ng is the raw winding normal).
inline Vec3 offsetOrigin(const Vec3& p, const Vec3& ng, const Vec3& dir) {
    double sgn = dot(ng, dir) >= 0.0 ? 1.0 : -1.0;
    return p + ng * (sgn * 1e-6);
}

// Gather-side shading-normal correction for the VERTEX-MERGE (VM / photon-density) strategy.
// The VM contribution is a Jensen density estimate MIS-combined with the vertex-connection
// (VC) strategies; on a smooth (interpolated-normal) mesh the merge reads incident flux per
// GEOMETRIC area and shades per facet, while the reference backward path tracer (mode R) and
// the VC strategies integrate against the SHADING cosine. Reweighting each merged light
// vertex (incident direction `wp`, back toward its source) by cos_s/cos_g rebalances the
// merge onto the shading cosine so mode U smooth-shades like mode R:
//
//   gcorr = |cos(wp, Ns)| / |cos(wp, Ng)|
//
// EXACTLY 1 when Ns==Ng (flat tris, analytic spheres), so every non-smooth scene — and the
// whole existing mode-U validation suite — is bit-identical. The grazing `cos(wp,Ng)`
// denominator is guarded (measure-zero, ~0 flux) so a degenerate sample falls back to no
// correction. NOTE: the standalone photon-map / SPPM gathers (modes M/S) already smooth-shade
// in this renderer and must NOT use this — only the MIS-coupled VM merge needs it. Why the
// coupling makes the difference is logged as tech debt in known-issues.md.
inline double vmGatherCorr(const Vec3& wp, const Vec3& ns, const Vec3& ng) {
    double denom = std::fabs(dot(wp, ng));
    if (denom <= 1e-8) return 1.0;
    return std::fabs(dot(wp, ns)) / denom;
}

// A stored light-subpath vertex (only connectible/non-delta surface vertices are kept).
struct LightVertex {
    Vec3   p;             // world position
    Vec3   ns, ng;        // shading / geometric normal
    Vec3   wo;            // unit dir toward the PREVIOUS light vertex (the fixed BSDF dir /
                          // reversed incident photon direction — used by connect AND merge)
    double beta = 0.0;    // geometric throughput carried here (single wavelength, no cie)
    float  lambda = 0.0f; // this subpath's wavelength (for the XYZ merge estimate)
    // cieX/Y/Z(lambda) cached at store time: the merge loop reads a vertex once per
    // NEARBY CAMERA VERTEX, so evaluating the three CIE Gaussian sums there repaid
    // the cost per gather instead of once per vertex. Same call on the same stored
    // float lambda -> bit-identical values.
    double cx = 0.0, cy = 0.0, cz = 0.0;
    double dVCM = 0.0, dVC = 0.0, dVM = 0.0;
    int    matId = -1;
    const Material* mat = nullptr;
    Hit    hit;           // for textured albedo / roughness at merge & connect
    int    edges = 0;     // number of edges from the light emitter to this vertex
};

// The hero bundle's SECONDARY wavelengths for a stored light vertex. Deliberately a
// SEPARATE array indexed in lockstep with `lightVerts` rather than extra fields inside
// LightVertex: stored light vertices are the dominant memory cost of a VCM pass (and of
// the GPU slab), so a plain single-wavelength run (`-heroc 1`, and every scene the hero
// gate rejects) must allocate exactly nothing extra and stay byte-identical. The whole
// array is left empty in that case.
//
// `lam[i]` mirrors the path's bundle, so a light vertex and the camera vertex it is
// CONNECTED to always speak about the same wavelengths (both subpaths of path index p are
// drawn from one bundle — see vcmPass). The MERGE strategy gathers vertices from other
// paths, so it reads these wavelengths on their own terms, exactly as it already does for
// the single stored `LightVertex::lambda`.
struct LightVertexSec {
    double beta[hero::kHeroMax - 1] = {0};   // per-secondary geometric throughput
    double cx[hero::kHeroMax - 1] = {0};     // cieX/Y/Z(lam[i]) cached at store time, for
    double cy[hero::kHeroMax - 1] = {0};     // the same reason LightVertex caches the
    double cz[hero::kHeroMax - 1] = {0};     // hero's: the merge reads it per gather.
    float  lam[hero::kHeroMax - 1] = {0};    // the secondary wavelengths
    int    nUp = 1;                          // wavelengths still live at this vertex
};

// Uniform hash grid over the stored light vertices (indices into a flat array), mirroring
// photonmap.h's counting-sort layout. Cell size == merge radius, so a radius query only
// touches the 3x3x3 neighbourhood.
struct VcmGrid {
    Vec3 lo{0, 0, 0};
    double cell = 1e-4;
    int nx = 1, ny = 1, nz = 1;
    std::vector<int> cellStart;   // size nCells+1
    std::vector<int> order;       // vertex indices in cell-contiguous order

    long long cellCount() const { return (long long)nx * ny * nz; }
    int cellIndex(int ix, int iy, int iz) const { return (iz * ny + iy) * nx + ix; }
    void cellCoord(const Vec3& p, int& ix, int& iy, int& iz) const {
        ix = (int)std::floor((p.x - lo.x) / cell);
        iy = (int)std::floor((p.y - lo.y) / cell);
        iz = (int)std::floor((p.z - lo.z) / cell);
        ix = std::min(std::max(ix, 0), nx - 1);
        iy = std::min(std::max(iy, 0), ny - 1);
        iz = std::min(std::max(iz, 0), nz - 1);
    }

    void build(const std::vector<LightVertex>& v, double radius) {
        cell = radius > 0.0 ? radius : 1e-6;
        const size_t n = v.size();
        if (n == 0) { nx = ny = nz = 1; cellStart.assign(2, 0); order.clear(); return; }
        Vec3 mn = v[0].p, mx = v[0].p;
        for (const auto& lv : v) {
            mn.x = std::min(mn.x, lv.p.x); mn.y = std::min(mn.y, lv.p.y); mn.z = std::min(mn.z, lv.p.z);
            mx.x = std::max(mx.x, lv.p.x); mx.y = std::max(mx.y, lv.p.y); mx.z = std::max(mx.z, lv.p.z);
        }
        lo = mn - Vec3{cell, cell, cell} * 0.5;
        Vec3 ext = (mx - lo) + Vec3{cell, cell, cell} * 0.5;
        nx = std::max(1, (int)std::ceil(ext.x / cell));
        ny = std::max(1, (int)std::ceil(ext.y / cell));
        nz = std::max(1, (int)std::ceil(ext.z / cell));
        const long long nCells = cellCount();
        std::vector<int> cellOf(n);
        cellStart.assign((size_t)nCells + 1, 0);
        for (size_t i = 0; i < n; ++i) {
            int ix, iy, iz; cellCoord(v[i].p, ix, iy, iz);
            int c = cellIndex(ix, iy, iz);
            cellOf[i] = c; ++cellStart[c + 1];
        }
        for (long long c = 0; c < nCells; ++c) cellStart[c + 1] += cellStart[c];
        order.assign(n, 0);
        std::vector<int> cursor(cellStart.begin(), cellStart.end() - 1);
        for (size_t i = 0; i < n; ++i) order[cursor[cellOf[i]]++] = (int)i;
    }

    // Invoke fn(vertexIndex) for every stored vertex within radius r of p (r <= cell).
    template <class F>
    void query(const std::vector<LightVertex>& v, const Vec3& p, double r, F&& fn) const {
        if (order.empty()) return;
        int ix, iy, iz; cellCoord(p, ix, iy, iz);
        const double r2 = r * r;
        for (int dz = -1; dz <= 1; ++dz) { int cz = iz + dz; if (cz < 0 || cz >= nz) continue;
        for (int dy = -1; dy <= 1; ++dy) { int cy = iy + dy; if (cy < 0 || cy >= ny) continue;
        for (int dx = -1; dx <= 1; ++dx) { int cx = ix + dx; if (cx < 0 || cx >= nx) continue;
            int c = cellIndex(cx, cy, cz);
            for (int k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                int idx = order[k];
                Vec3 d = p - v[idx].p;
                if (dot(d, d) <= r2) fn(idx);
            }
        }}}
    }
};

// Persistent accumulator across passes. `accum` holds the running SUM over passes of the
// full per-pixel radiance (camera-path contributions + connect-to-camera splats); the
// resolve divides by the pass count. `pathStart/pathCount` and `lightVerts` are rebuilt
// each pass (not persistent), so only accum + passes live here.
struct VcmState {
    int resX = 0, resY = 0;
    std::vector<Vec3> accum;   // XYZ sum over passes
    long long passes = 0;
    void init(int w, int h) { resX = w; resY = h; accum.assign((size_t)w * h, Vec3{0, 0, 0}); passes = 0; }
};

// Balance heuristic: Mis(x) = x (power heuristic would square).
inline double Mis(double x) { return x; }

// MIS update on ARRIVING at a vertex after travelling `dist` with incident cosine
// `cosThetaIn` (= |dot(ns, -rayDir)|): fold the edge's area-conversion Jacobian into the
// running quantities before this vertex is used/stored.
inline void misArrival(double dist, double cosThetaIn, double& dVCM, double& dVC, double& dVM) {
    dVCM *= Mis(dist * dist);
    dVCM /= Mis(cosThetaIn);
    dVC  /= Mis(cosThetaIn);
    dVM  /= Mis(cosThetaIn);
}

// MIS update after SCATTERING (continuation) with the sampled direction's cosine
// `cosThetaOut = |dot(wi, ns)|`, forward/reverse solid-angle pdfs, and the per-pass VC/VM
// weight factors. `specular` selects the delta branch (no connection/merge through it).
inline void misScatter(bool specular, double cosThetaOut, double bsdfDirPdfW, double bsdfRevPdfW,
                       double misVcWeight, double misVmWeight,
                       double& dVCM, double& dVC, double& dVM) {
    if (specular) {
        dVCM = 0.0;
        dVC *= Mis(cosThetaOut);
        dVM *= Mis(cosThetaOut);
    } else {
        dVC = Mis(cosThetaOut / bsdfDirPdfW) * (dVC * Mis(bsdfRevPdfW) + dVCM + misVmWeight);
        dVM = Mis(cosThetaOut / bsdfDirPdfW) * (dVM * Mis(bsdfRevPdfW) + dVCM * misVcWeight + 1.0);
        dVCM = Mis(1.0 / bsdfDirPdfW);
    }
}

// Sample a scattering continuation at a surface vertex (mirrors bdpt.h's randomWalk switch
// but returned as a standalone step). On return: `wi` continuation dir, `betaFactor` the
// throughput multiplier (f*|cos|/pdf for non-delta, reflectance for delta), `pdfW`/`pdfRevW`
// the forward/reverse solid-angle densities (0 for delta), `cosThetaOut = |dot(wi,ns)|`,
// `delta` the specular flag. `terminate` requests ending the walk. `stk` is the nested-
// dielectric medium stack the path carries (colored-glass Beer-Lambert + exterior IOR at
// each interface; Schmidt & Budge 2002). Media are out of VCM scope.
//
// HERO BUNDLE (Wilkie et al. 2014; the four shared policies live in hero.h). When `nUp > 1`
// the caller is carrying nUp wavelengths on one ray: `lamAll` is the bundle (lamAll[0] is
// the hero == `lambda`), and on return `secF[i]` is the ABSOLUTE throughput factor for
// secondary i — f_{i+1}·cos/pdf for the lobe the HERO sampled — whenever `secChromatic` is
// set. Absolute, never a ratio to the hero's: a ratio is undefined exactly where it matters
// most (a gel filter whose hero value is 0 while a secondary's is not). Wavelength-
// INDEPENDENT outcomes (all the geometry, the specular interfaces) leave `secChromatic`
// false, and the caller reuses `betaFactor` for every λ. `keepBundle` marks the delta lobes
// that nevertheless pick their continuation WITHOUT consulting λ (Mirror reflects, Filter
// passes straight through), so the secondaries may keep riding the hero's ray past them;
// every other delta lobe de-heroes (the caller collapses nUp to 1). All five outputs are
// ignored when nUp == 1, so a single-wavelength run is bit-identical to before.
//
// NOTE the `<= 0` early terminations that the scalar version used to do for a zero
// reflectance/transmittance are GONE: with a bundle the hero can be dark while a secondary
// is not, so killing the walk on the hero alone biases low. The caller applies the
// max-over-live-λ test instead, which for nUp == 1 is exactly the old scalar test.
inline void scatterSample(const Scene& scene, const Renderer& mats, const Material* mp,
                          const Hit& h, const Vec3& rayDir, double lambda, Pcg32& rng,
                          Vec3& wi, double& betaFactor, double& pdfW, double& pdfRevW,
                          double& cosThetaOut, bool& delta, bool& terminate,
                          MediumStack& stk,
                          const double* lamAll = nullptr, int nUp = 1,
                          double* secF = nullptr, bool* secChromatic = nullptr,
                          bool* keepBundle = nullptr) {
    const Vec3 ns = h.n;
    const Vec3 wo = normalize(rayDir * -1.0);
    wi = Vec3{0, 0, 0}; betaFactor = 0.0; pdfW = 0.0; pdfRevW = 0.0; cosThetaOut = 0.0;
    delta = false; terminate = false;
    const int nSec = (secF && lamAll && nUp > 1) ? nUp - 1 : 0;   // secondaries to fill
    if (secChromatic) *secChromatic = false;
    if (keepBundle)   *keepBundle = false;
    switch (mp->type) {
        case MatType::Diffuse:
        case MatType::Fluorescent: {
            wi = cosineHemisphere(ns, rng);
            if (dot(wi, ns) <= 0) { terminate = true; break; }
            double rho = clamp01(diffuseReflectance(scene, *mp, h, lambda));
            pdfW = bsdfPdf(*mp, ns, wo, wi, lambda, scene, &h);
            pdfRevW = bsdfPdf(*mp, ns, wi, wo, lambda, scene, &h);
            betaFactor = rho;                     // rho <= 0 is caught by the caller's max test
            if (nSec) {
                *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(diffuseReflectance(scene, *mp, h, lamAll[i + 1]));
            }
            break;
        }
        case MatType::Glossy: {
            Vec3 mdir = reflect(rayDir, ns);
            wi = sampleGlossy(mdir, materialRoughness(scene, *mp, h), rng);
            if (dot(wi, ns) <= 0) { terminate = true; break; }
            double r = clamp01(reflectSlot(scene, *mp, h, lambda));
            pdfW = bsdfPdf(*mp, ns, wo, wi, lambda, scene, &h);
            pdfRevW = bsdfPdf(*mp, ns, wi, wo, lambda, scene, &h);
            betaFactor = r;
            if (pdfW <= 0) terminate = true;      // r <= 0 is caught by the caller's max test
            // The glossy LOBE (mirror direction + roughness exponent) carries no wavelength
            // dependence, so the whole bundle follows the sampled direction and only the
            // reflectance differs per λ.
            if (nSec) {
                *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(reflectSlot(scene, *mp, h, lamAll[i + 1]));
            }
            break;
        }
        case MatType::DiffuseTransmit: {
            double rhoR, rhoT; bdpt::diffuseTransmitAlbedos(*mp, lambda, scene, &h, rhoR, rhoT);
            double tot = rhoR + rhoT;
            if (tot <= 0.0) { terminate = true; break; }
            const bool reflLobe = (rng.uniform() * tot < rhoR);
            if (reflLobe) wi = cosineHemisphere(ns, rng);
            else          wi = cosineHemisphere(ns * -1.0, rng);
            pdfW = bsdfPdf(*mp, ns, wo, wi, lambda, scene, &h);
            pdfRevW = bsdfPdf(*mp, ns, wi, wo, lambda, scene, &h);
            betaFactor = tot;
            if (pdfW <= 0) terminate = true;
            // The lobe was CHOSEN by the hero's albedo split, so each secondary divides by
            // the HERO's albedo for that lobe, not its own:
            //   f_i·cos/pdf_hero = rho_i(lobe) · tot_hero / rho_hero(lobe).
            if (nSec) {
                *secChromatic = true;
                for (int i = 0; i < nSec; ++i) {
                    double rR, rT;
                    bdpt::diffuseTransmitAlbedos(*mp, lamAll[i + 1], scene, &h, rR, rT);
                    double num = reflLobe ? rR   : rT;
                    double den = reflLobe ? rhoR : rhoT;
                    secF[i] = (den > 0.0) ? num * tot / den : 0.0;
                }
            }
            break;
        }
        case MatType::Mirror: {
            double r = clamp01(reflectSlot(scene, *mp, h, lambda));
            wi = reflect(rayDir, ns); betaFactor = r; delta = true;
            // The mirror direction is the same for every λ, so the bundle survives; only
            // the reflectance is per-λ (cf. Glossy, the rough version of this).
            if (nSec) {
                *keepBundle = true; *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(reflectSlot(scene, *mp, h, lamAll[i + 1]));
            }
            break;
        }
        case MatType::Dielectric: {
            // Nested-dielectric PRIORITY resolution: exterior IOR = the medium the path is
            // currently inside (highest-priority stack entry). Overlapping dielectrics are
            // ranked by `priority` (higher wins; lower is suppressed -> straight pass-
            // through). SAFE FALLBACK to flat air<->glass (extIor 1.0) unless BOTH sides
            // carry an explicit priority, keeping priority-free scenes bit-identical.
            bool entering = dot(rayDir, h.ng) < 0.0;
            const int mi = (int)(mp - scene.mats.data());   // true index (Mix/Layered aware)
            const int pr = mp->priority;
            delta = true; betaFactor = 1.0;
            if (entering) {
                const int outMat = stk.topMat();
                const int outPri = stk.topPri();
                const bool ranked = mp->hasPriority() &&
                    (stk.empty() || (outMat >= 0 && scene.mats[outMat].hasPriority()));
                if (ranked && !stk.empty() && pr <= outPri) {   // suppressed inner surface
                    wi = rayDir; stk.push(mi, pr);
                } else {
                    const double extIor = (ranked && outMat >= 0)
                        ? scene.mats[outMat].ior(lambda) : 1.0;
                    bool transmitted = false;
                    Ray nr = mats.refractOrReflect(scene, *mp, h, rayDir, lambda, rng, &transmitted, extIor);
                    wi = nr.d;
                    if (transmitted) stk.push(mi, pr);
                }
            } else {
                MediumStack after = stk; after.popMat(mi);
                const int newMat = after.topMat();
                const int newPri = after.topPri();
                const bool ranked = mp->hasPriority() &&
                    (after.empty() || (newMat >= 0 && scene.mats[newMat].hasPriority()));
                if (ranked && newMat >= 0 && pr <= newPri) {    // suppressed: still enclosed
                    wi = rayDir; stk.popMat(mi);
                } else {
                    const double extIor = (ranked && newMat >= 0)
                        ? scene.mats[newMat].ior(lambda) : 1.0;
                    bool transmitted = false;
                    Ray nr = mats.refractOrReflect(scene, *mp, h, rayDir, lambda, rng, &transmitted, extIor);
                    wi = nr.d;
                    if (transmitted) stk.popMat(mi);            // TIR stays inside mi
                }
            }
            break;
        }
        case MatType::HalfMirror: {
            double r = clamp01(reflectSlot(scene, *mp, h, lambda));
            if (rng.uniform() < r) wi = reflect(rayDir, ns);
            else                   wi = rayDir;
            betaFactor = 1.0; delta = true;
            break;
        }
        case MatType::Filter: {
            double t = clamp01(transmitSlot(scene, *mp, h, lambda));
            wi = rayDir; betaFactor = t; delta = true;   // t <= 0 -> caller's max test
            // Straight-through for every λ, so the bundle survives — and a gel filter is
            // exactly where the per-λ transmittance spread is largest, i.e. the case that
            // benefits most from NOT de-heroing, AND the case that forces the absolute
            // (rather than ratio) secF, since T(λ_hero) is legitimately 0 across most of
            // a Wratten passband.
            if (nSec) {
                *keepBundle = true; *secChromatic = true;
                for (int i = 0; i < nSec; ++i)
                    secF[i] = clamp01(transmitSlot(scene, *mp, h, lamAll[i + 1]));
            }
            break;
        }
        case MatType::ThinFilm: {
            Ray nr;
            if (!mats.thinFilmInterface(scene, *mp, h, rayDir, lambda, rng, nr)) { terminate = true; break; }
            wi = nr.d; betaFactor = 1.0; delta = true;
            break;
        }
        case MatType::Multilayer: {
            Ray nr;
            if (!mats.multilayerInterface(*mp, h, rayDir, lambda, rng, nr)) { terminate = true; break; }
            wi = nr.d; betaFactor = 1.0; delta = true;
            break;
        }
        case MatType::Grating: {
            double r = clamp01(reflectSlot(scene, *mp, h, lambda));
            if (r <= 0) { terminate = true; break; }
            bool absorbedG; Ray nr = mats.gratingDiffract(*mp, h, rayDir, lambda, rng, absorbedG);
            if (absorbedG) { terminate = true; break; }
            wi = nr.d; betaFactor = r; delta = true;
            break;
        }
        default: terminate = true; break;
    }
    if (!terminate) cosThetaOut = std::fabs(dot(wi, ns));
}

// Per-pass constants bundled for the walks.
struct PassCtx {
    double radius = 0.0;
    double nLightPaths = 0.0;   // light subpaths this pass (== resX*resY)
    double misVcWeight = 0.0;   // 1 / etaVCM
    double misVmWeight = 0.0;   // etaVCM = pi r^2 nLightPaths
    double vmNorm = 0.0;        // 1 / etaVCM (merge density normaliser)
    double imagePlaneDist = 0.0;// resX / (2 tanHalfX), for camera importance
    int    maxDepth = 8;        // max path length in edges
    bool   diffraction = true;
};

// --- Light subpath: trace, store connectible vertices, connect-to-camera splats --------
// Appends this path's stored vertices to `out` and its connect-to-camera contributions to
// `splat` (a local W*H XYZ buffer merged by the caller). `pathEdges` is unused externally;
// vertices carry their own edge counts. Returns nothing (the caller records out.size()
// before/after to slice this path's range).
inline void traceLightSubpath(const Scene& scene, const Camera& cam, const Renderer& mats,
                              const PassCtx& ctx, const bdpt::HeroBundle& hb,
                              Pcg32& rng, std::vector<LightVertex>& out,
                              std::vector<LightVertexSec>& outSec,
                              std::vector<Vec3>& splat, int W) {
    const double lambda = hb.lam[0], invPdfLambda = hb.invPdf[0];
    const int C = hb.C;                 // bundle width (1 == plain single-λ)
    int nUp = C;                        // wavelengths still riding this ray
    double betaSec[hero::kHeroMax - 1] = {0};
    Vec3   cieSec[hero::kHeroMax - 1];
    for (int i = 0; i + 1 < C; ++i)
        cieSec[i] = Vec3(cieX(hb.lam[i + 1]), cieY(hb.lam[i + 1]), cieZ(hb.lam[i + 1]));

    if (scene.emitters.empty() || scene.totalPower <= 0.0) return;
    int ei = scene.selectEmitter(rng);
    const Emitter& em = scene.emitters[ei];
    if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env ||
        em.shape == EmitterShape::Sun || em.collimated) return;

    double u1 = rng.uniform(), u2 = rng.uniform();
    Vec3 y, nOut;
    // `emit pattern:` factor at the sampled point (1.0 without one). It scales the
    // emitted radiance ONLY: directPdfW / emissionPdfW below — and therefore every dVCM
    // / dVC / dVM MIS quantity derived from them — stay exactly as the camera side's
    // s=0/s=1 terms assume, which is what keeps VCM's weights consistent.
    double emitPatW = emitterSamplePoint(scene, em, u1, u2, y, nOut);
    double Le = em.spdFn(lambda) * invPdfLambda * emitPatW;
    // Max over the LIVE wavelengths, never the hero alone (hero.h policy 3): a spiky
    // emission spectrum can be dark at the hero and bright at a secondary.
    double LeSec[hero::kHeroMax - 1] = {0}, mxLe = Le;
    for (int i = 0; i + 1 < C; ++i) {
        LeSec[i] = em.spdFn(hb.lam[i + 1]) * hb.invPdf[i + 1] * emitPatW;
        if (LeSec[i] > mxLe) mxLe = LeSec[i];
    }
    if (mxLe <= 0.0) return;
    double pdfChoice = em.power / scene.totalPower;
    double pdfPos = (em.area > 0.0) ? 1.0 / em.area : 0.0;
    if (pdfPos <= 0.0 || pdfChoice <= 0.0) return;

    Vec3 dir = cosineHemisphere(nOut, rng);
    double cosLight = dot(nOut, dir);
    if (cosLight <= 0.0) return;
    double pdfDirW = cosLight / PI;
    double emissionPdfW = pdfPos * pdfDirW * pdfChoice;   // full emission density (solid angle)
    if (emissionPdfW <= 0.0) return;
    double directPdfW = pdfChoice * pdfPos;               // area density of the emitter point

    // Geometric throughput leaving the light (Le carries the spectral radiance). Every
    // factor but Le is purely geometric, so the secondaries share it exactly.
    double beta = Le * cosLight / emissionPdfW;
    for (int i = 0; i + 1 < C; ++i) betaSec[i] = LeSec[i] * cosLight / emissionPdfW;

    // Running MIS quantities (SmallVCM GenerateLightSample).
    double dVCM = Mis(directPdfW / emissionPdfW);
    double dVC  = Mis(cosLight / emissionPdfW);
    double dVM  = dVC * ctx.misVcWeight;

    const Vec3 cie(cieX(lambda), cieY(lambda), cieZ(lambda));
    MediumStack stk;                              // nested-dielectric medium stack
    Vec3 prevP = y;
    Ray ray{y + nOut * 1e-6, dir};

    for (int edges = 1; edges <= ctx.maxDepth; ++edges) {
        Hit h = scene.closestHit(ray);
        if (!h.valid) return;                    // escaped (no env in scope)
        {
            int mi = stk.topMat();
            if (mi >= 0) {
                double a = scene.mats[mi].absorb(lambda);
                if (a > 0.0) beta *= std::exp(-a * h.t);
                // Beer-Lambert is per-λ (that IS the colour of coloured glass), so each
                // live secondary attenuates by its own absorption coefficient.
                for (int i = 0; i + 1 < nUp; ++i) {
                    double ai = scene.mats[mi].absorb(hb.lam[i + 1]);
                    if (ai > 0.0) betaSec[i] *= std::exp(-ai * h.t);
                }
            }
        }
        double dist = h.t;
        const Vec3 rd = ray.d;
        double cosThetaIn = std::fabs(dot(h.n, rd * -1.0));
        if (cosThetaIn <= 1e-9) return;

        // Resolve material (Mix -> child).
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return;
            mp = &scene.mats[c];
        }
        if (mp->isLight) return;                 // light subpath doesn't scatter off emitters

        misArrival(dist, cosThetaIn, dVCM, dVC, dVM);

        Vec3 wo = normalize(prevP - h.p);        // toward the previous light vertex
        // Geometric normal oriented onto the shading-normal side, for the Veach adjoint
        // shading-normal correction on this LIGHT (particle) subpath (§5.3; identical role
        // to render.h/bdpt.h). Exactly a no-op when h.n==h.ng (flat tris, analytic spheres).
        Vec3 ngo = (dot(h.ng, h.n) >= 0.0) ? h.ng : h.ng * -1.0;

        // Store the vertex + connect to camera (only for connectible, non-delta surfaces).
        bool connectible = isConnectibleMat(*mp);
        if (connectible) {
            LightVertex lv;
            lv.p = h.p; lv.ns = h.n; lv.ng = h.ng; lv.wo = wo;
            lv.beta = beta; lv.lambda = (float)lambda;
            lv.cx = cieX(lv.lambda); lv.cy = cieY(lv.lambda); lv.cz = cieZ(lv.lambda);
            lv.dVCM = dVCM; lv.dVC = dVC; lv.dVM = dVM;
            lv.matId = h.matId; lv.mat = mp; lv.hit = h; lv.edges = edges;
            out.push_back(lv);
            if (C > 1) {                              // lockstep secondary payload
                LightVertexSec ls; ls.nUp = nUp;
                for (int i = 0; i + 1 < nUp; ++i) {
                    ls.lam[i] = (float)hb.lam[i + 1];
                    ls.beta[i] = betaSec[i];
                    ls.cx[i] = cieX(ls.lam[i]); ls.cy[i] = cieY(ls.lam[i]); ls.cz[i] = cieZ(ls.lam[i]);
                }
                outSec.push_back(ls);
            }

            // Connect this vertex to the pinhole camera (light-image splat, t=1). The
            // path here has (edges + 1) edges: skip if it would exceed maxDepth.
            if (!cam.hasLens() && edges + 1 <= ctx.maxDepth) {
                Vec3 toCam = cam.eye - h.p;
                double dist2c = dot(toCam, toCam);
                if (dist2c > 1e-12) {
                    double distc = std::sqrt(dist2c);
                    Vec3 wcam = toCam / distc;
                    double cosToCamera = dot(h.n, wcam);
                    // Geometric-hemisphere softening for a reflect-only vertex: the camera must
                    // lie on the geometric front side too (no back-face light leak), but ramp
                    // that boundary smoothly instead of a hard cutoff (Chiang 2019). `ngo` is the
                    // geo normal oriented to h.n (line above). No-op when h.n==h.ng (stG==1).
                    double stG = isTwoSidedMat(*mp) ? 1.0 : shadowTerminatorG(wcam, h.n, ngo);
                    bool sideOk = isTwoSidedMat(*mp) ? (cosToCamera != 0.0)
                                                     : (cosToCamera > 0.0 && stG > 0.0);
                    double cosAtCamera = dot(cam.w, wcam * -1.0);   // forward vs camera->point
                    if (sideOk && cosAtCamera > 1e-9) {
                        int px, py; double cc, d2c;
                        if (cam.project(h.p, px, py, cc, d2c)) {
                            // The adjoint + soft-terminator corrections are purely
                            // geometric, so the whole bundle shares them; only the BSDF
                            // value is per-λ (hero.h policy 4: absolute, not a ratio).
                            const double geo = shadingAdjointCorr(wo, wcam, h.n, ngo) * stG;
                            double f = bsdfF(*mp, h.n, wo, wcam, lambda, scene, &h) * geo;
                            double fSec[hero::kHeroMax - 1] = {0}, mxF = f;
                            for (int i = 0; i + 1 < nUp; ++i) {
                                fSec[i] = bsdfF(*mp, h.n, wo, wcam, hb.lam[i + 1], scene, &h) * geo;
                                if (fSec[i] > mxF) mxF = fSec[i];
                            }
                            if (mxF > 0.0 &&
                                !scene.occluded(offsetOrigin(h.p, h.ng, wcam), wcam, distc - 2e-6)) {
                                double bsdfRevPdfW = bsdfPdf(*mp, h.n, wcam, wo, lambda, scene, &h);
                                double imgPtDist = ctx.imagePlaneDist / cosAtCamera;
                                double imgToSolid = imgPtDist * imgPtDist / cosAtCamera;
                                double imgToSurf = imgToSolid * std::fabs(cosToCamera) / dist2c;
                                double wLight = Mis(imgToSurf / ctx.nLightPaths) *
                                                (ctx.misVmWeight + dVCM + dVC * Mis(bsdfRevPdfW));
                                double misW = 1.0 / (wLight + 1.0);
                                // contrib = misW * beta * f * (cameraAreaPdf / nLightPaths).
                                // The MIS weight and every density above are the HERO's
                                // (all sampling densities here are λ-independent), so the
                                // bundle differs only in beta*f. De-hero by averaging the
                                // nUp live wavelengths — the same 1/nUp normalisation
                                // bdpt.h applies at its splat, NOT a ×C boost.
                                double k = misW * imgToSurf / ctx.nLightPaths;
                                Vec3 add = cie * (k * beta * f);
                                for (int i = 0; i + 1 < nUp; ++i)
                                    add = add + cieSec[i] * (k * betaSec[i] * fSec[i]);
                                if (nUp > 1) add = add * (1.0 / (double)nUp);
                                splat[(size_t)py * W + px] += add;
                            }
                        }
                    }
                }
            }
        }

        if (edges == ctx.maxDepth) return;       // no room to continue

        // Sample a continuation direction.
        Vec3 wi; double betaFactor, pdfW, pdfRevW, cosThetaOut; bool delta, terminate;
        double secF[hero::kHeroMax - 1]; bool secChromatic = false, keepBundle = false;
        scatterSample(scene, mats, mp, h, rd, lambda, rng, wi, betaFactor, pdfW, pdfRevW,
                      cosThetaOut, delta, terminate, stk,
                      hb.lam, nUp, secF, &secChromatic, &keepBundle);
        // Kill the walk only when EVERY live wavelength is dead (hero.h policy 3). With
        // nUp == 1 the loop is empty and mxF == betaFactor: the old scalar test exactly.
        double mxF = betaFactor;
        if (secChromatic) for (int i = 0; i + 1 < nUp; ++i) if (secF[i] > mxF) mxF = secF[i];
        if (terminate || mxF <= 0.0) return;
        if (!delta && (pdfW <= 0.0 || cosThetaOut <= 0.0)) return;

        misScatter(delta, cosThetaOut, pdfW, pdfRevW, ctx.misVcWeight, ctx.misVmWeight,
                   dVCM, dVC, dVM);
        beta *= betaFactor;
        for (int i = 0; i + 1 < nUp; ++i) betaSec[i] *= secChromatic ? secF[i] : betaFactor;
        if (!delta) {
            const double adj = shadingAdjointCorr(wo, normalize(wi), h.n, ngo);  // adjoint (continuation)
            beta *= adj;                                     // purely geometric: same for every λ
            for (int i = 0; i + 1 < nUp; ++i) betaSec[i] *= adj;
        }
        // DE-HERO (hero.h policy 2). A delta vertex that picked its continuation by a
        // λ-DEPENDENT specular process (dielectric refraction, grating order, thin-film /
        // multilayer interface, the half-mirror's r(λ) coin) cannot carry the secondaries
        // along the hero's outgoing ray any further. Mirror and Filter are delta but
        // λ-independent in direction, so they set keepBundle and ride on. The vertex JUST
        // stored keeps its full nUp — it really was reached by all C wavelengths.
        if (delta && !keepBundle) nUp = 1;
        prevP = h.p;
        double sgn = dot(wi, h.ng) >= 0.0 ? 1.0 : -1.0;
        ray = Ray{h.p + h.ng * (sgn * 1e-6), normalize(wi)};
    }
}

// --- Camera subpath: walk and accumulate all camera-side strategies --------------------
// `lightVerts` is the full stored light-vertex array; [pathBegin,pathEnd) is THIS pixel's
// paired light subpath's slice (for exact same-lambda vertex connections); `grid` gathers
// vertices from ALL paths for merging. Returns the pixel's accumulated XYZ radiance.
inline Vec3 traceCameraSubpath(const Scene& scene, const Camera& cam, const Renderer& mats,
                               const PassCtx& ctx, int px, int py, const bdpt::HeroBundle& hb,
                               Pcg32& rng, const std::vector<LightVertex>& lightVerts,
                               const std::vector<LightVertexSec>& lightSec,
                               int pathBegin, int pathEnd, const VcmGrid& grid) {
    Vec3 result{0, 0, 0};
    const double lambda = hb.lam[0], invPdfLambda = hb.invPdf[0];
    const int C = hb.C;                 // bundle width (1 == plain single-λ)
    const bool heroOn = (C > 1);
    int nUp = C;                        // wavelengths still riding this ray
    double betaSec[hero::kHeroMax - 1] = {0};
    Vec3   cieSec[hero::kHeroMax - 1];
    for (int i = 0; i + 1 < C; ++i) {
        betaSec[i] = 1.0;               // camera vertex beta == 1 for every λ
        cieSec[i] = Vec3(cieX(hb.lam[i + 1]), cieY(hb.lam[i + 1]), cieZ(hb.lam[i + 1]));
    }
    const Vec3 cie(cieX(lambda), cieY(lambda), cieZ(lambda));

    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
    double cosAtCamera = dot(ray.d, cam.w);
    if (cosAtCamera <= 1e-9) return result;
    double cameraPdfW = ctx.imagePlaneDist * ctx.imagePlaneDist / (cosAtCamera * cosAtCamera * cosAtCamera);

    double beta = 1.0;
    double dVCM = Mis(ctx.nLightPaths / cameraPdfW);
    double dVC = 0.0, dVM = 0.0;
    MediumStack stk;                              // nested-dielectric medium stack
    Vec3 prevP = cam.eye;

    for (int edges = 1; edges <= ctx.maxDepth; ++edges) {
        Hit h = scene.closestHit(ray);
        if (!h.valid) {
            // Directly-viewed environment (no env in scope; ignore).
            return result;
        }
        {
            int mi = stk.topMat();
            if (mi >= 0) {
                double a = scene.mats[mi].absorb(lambda);
                if (a > 0.0) beta *= std::exp(-a * h.t);
                for (int i = 0; i + 1 < nUp; ++i) {     // per-λ Beer-Lambert (coloured glass)
                    double ai = scene.mats[mi].absorb(hb.lam[i + 1]);
                    if (ai > 0.0) betaSec[i] *= std::exp(-ai * h.t);
                }
            }
        }
        double dist = h.t;
        const Vec3 rd = ray.d;
        double cosThetaIn = std::fabs(dot(h.n, rd * -1.0));
        if (cosThetaIn <= 1e-9) return result;

        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return result;
            mp = &scene.mats[c];
        }

        misArrival(dist, cosThetaIn, dVCM, dVC, dVM);

        Vec3 wo = normalize(prevP - h.p);        // toward the previous camera vertex (= -rd)

        // (a) Emission: the camera path hit an emitter (s=0 strategy).
        if (mp->isLight) {
            double cosLight = dot(h.ng, wo);
            if (cosLight > 0.0) {
                // Achromatic `emit pattern:` factor at this hit — the same value the
                // light-subpath / NEE sides get from emitterSamplePoint at this point.
                double ep = (mp->emitPat < 0) ? 1.0 : slotPatMul(scene, mp->emitPat, h);
                double Le = mp->emit(lambda) * invPdfLambda * ep;
                double LeSec[hero::kHeroMax - 1] = {0}, mxLe = Le;
                for (int i = 0; i + 1 < nUp; ++i) {
                    LeSec[i] = mp->emit(hb.lam[i + 1]) * hb.invPdf[i + 1] * ep;
                    if (LeSec[i] > mxLe) mxLe = LeSec[i];
                }
                if (mxLe > 0.0) {
                    double misW = 1.0;
                    const Emitter* em = scene.emitterForMat(h.matId);
                    if (em && em->area > 0.0 && scene.totalPower > 0.0 && edges >= 2) {
                        double pdfChoice = em->power / scene.totalPower;
                        double directPdfA = pdfChoice / em->area;         // area pdf of the point
                        double emissionPdfW = pdfChoice * cosLight / PI;  // directional emission pdf
                        double wCamera = Mis(directPdfA) * dVCM + Mis(emissionPdfW) * dVC;
                        misW = 1.0 / (1.0 + wCamera);
                    }
                    // Densities are λ-independent, so the bundle shares misW; only the
                    // emitted radiance and the carried throughput are per-λ. De-hero by
                    // averaging the live wavelengths (1/nUp, not a ×C boost).
                    Vec3 add = cie * (beta * Le * misW);
                    for (int i = 0; i + 1 < nUp; ++i)
                        add = add + cieSec[i] * (betaSec[i] * LeSec[i] * misW);
                    if (nUp > 1) add = add * (1.0 / (double)nUp);
                    result += add;
                }
            }
            return result;   // can't scatter off a light
        }

        bool connectible = isConnectibleMat(*mp);
        if (connectible) {
            // (b) NEE — connect this camera vertex to a freshly sampled light point (s=1).
            if (edges + 1 <= ctx.maxDepth && !scene.emitters.empty() && scene.totalPower > 0.0) {
                int ei = scene.selectEmitter(rng);
                const Emitter& em = scene.emitters[ei];
                if (!(em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env ||
                      em.shape == EmitterShape::Sun || em.collimated)) {
                    double u1 = rng.uniform(), u2 = rng.uniform();
                    Vec3 yL, nL;
                    double epat = emitterSamplePoint(scene, em, u1, u2, yL, nL);
                    Vec3 toL = yL - h.p; double dist2 = dot(toL, toL);
                    if (dist2 > 1e-12) {
                        double distL = std::sqrt(dist2); Vec3 wi = toL / distL;
                        double cosAtLight = dot(nL, wi * -1.0);
                        double cosToLight = dot(h.n, wi);
                        // Geometric-hemisphere softening on this eye/radiance vertex (matches
                        // backward.h neeLight): the sampled light must be on h's geometric front
                        // side too, ramped smoothly instead of a hard cutoff (Chiang 2019). No-op
                        // when h.n==h.ng (stG==1); skipped for two-sided.
                        double stG = isTwoSidedMat(*mp) ? 1.0 : shadowTerminatorG(wi, h.n, orientedGeoN(h));
                        bool sideOk = isTwoSidedMat(*mp)
                                          ? (cosToLight != 0.0)
                                          : (cosToLight > 0.0 && stG > 0.0);
                        if (cosAtLight > 0.0 && sideOk) {
                            double f = bsdfF(*mp, h.n, wo, wi, lambda, scene, &h) * stG;
                            double Le = em.spdFn(lambda) * invPdfLambda * epat;
                            // Per-λ BSDF × emitted radiance; the emitter-sampling densities
                            // below are λ-independent and stay the hero's (the pattern
                            // scales radiance only, never a pdf).
                            double fLeSec[hero::kHeroMax - 1] = {0}, mxfLe = f * Le;
                            for (int i = 0; i + 1 < nUp; ++i) {
                                fLeSec[i] = bsdfF(*mp, h.n, wo, wi, hb.lam[i + 1], scene, &h) * stG *
                                            em.spdFn(hb.lam[i + 1]) * hb.invPdf[i + 1] * epat;
                                if (fLeSec[i] > mxfLe) mxfLe = fLeSec[i];
                            }
                            if (mxfLe > 0.0 && em.area > 0.0 &&
                                !scene.occluded(offsetOrigin(h.p, h.ng, wi), wi, distL - 2e-6)) {
                                double pdfChoice = em.power / scene.totalPower;
                                double invArea = 1.0 / em.area;
                                double directPdfW = invArea * dist2 / cosAtLight;   // (no pickProb)
                                double emissionPdfW = invArea * cosAtLight / PI;    // (no pickProb)
                                double bsdfDirPdfW = bsdfPdf(*mp, h.n, wo, wi, lambda, scene, &h);
                                double bsdfRevPdfW = bsdfPdf(*mp, h.n, wi, wo, lambda, scene, &h);
                                double wLight = Mis(bsdfDirPdfW / (pdfChoice * directPdfW));
                                double wCamera = Mis(emissionPdfW * std::fabs(cosToLight) /
                                                     (directPdfW * cosAtLight)) *
                                                 (ctx.misVmWeight + dVCM + dVC * Mis(bsdfRevPdfW));
                                double misW = 1.0 / (wLight + 1.0 + wCamera);
                                double k = misW * std::fabs(cosToLight) / (pdfChoice * directPdfW);
                                Vec3 add = cie * (beta * k * Le * f);
                                for (int i = 0; i + 1 < nUp; ++i)
                                    add = add + cieSec[i] * (betaSec[i] * k * fLeSec[i]);
                                if (nUp > 1) add = add * (1.0 / (double)nUp);
                                result += add;
                            }
                        }
                    }
                }
            }

            // (c) Vertex connection: connect to the PAIRED light subpath's vertices.
            for (int j = pathBegin; j < pathEnd; ++j) {
                const LightVertex& lv = lightVerts[j];
                if (edges + lv.edges + 1 > ctx.maxDepth) continue;   // full path too long
                Vec3 d = lv.p - h.p; double dist2 = dot(d, d);
                if (dist2 <= 1e-12) continue;
                double distc = std::sqrt(dist2); Vec3 w = d / distc;   // camera -> light vertex
                double cosCam = dot(h.n, w);
                double cosLit = dot(lv.ns, w * -1.0);
                // Geometric-hemisphere softening on BOTH reflect-only endpoints (connection dir w
                // at the camera vertex, -w at the light vertex): each must see the other on its
                // geometric front side, ramped smoothly instead of a hard cutoff (Chiang 2019).
                // No-op when ns==ng (stG==1); skipped for two-sided materials.
                Vec3 ngoCam = orientedGeoN(h);
                Vec3 ngoLit = (dot(lv.ng, lv.ns) >= 0.0) ? lv.ng : lv.ng * -1.0;
                double stGCam = isTwoSidedMat(*mp)     ? 1.0 : shadowTerminatorG(w, h.n, ngoCam);
                double stGLit = isTwoSidedMat(*lv.mat) ? 1.0 : shadowTerminatorG(w * -1.0, lv.ns, ngoLit);
                bool camSide = isTwoSidedMat(*mp)
                                   ? (cosCam != 0.0) : (cosCam > 0.0 && stGCam > 0.0);
                bool litSide = isTwoSidedMat(*lv.mat)
                                   ? (cosLit != 0.0) : (cosLit > 0.0 && stGLit > 0.0);
                if (!camSide || !litSide) continue;
                double fCam = bsdfF(*mp, h.n, wo, w, lambda, scene, &h) * stGCam;
                double fLit = bsdfF(*lv.mat, lv.ns, lv.wo, w * -1.0, lambda, scene, &lv.hit);
                // Adjoint correction on the LIGHT-subpath endpoint lv only (particle side;
                // outgoing = -w toward the camera vertex). fCam is the Radiance side — none.
                // Uses lv.ng oriented to lv.ns (ngoLit above); a no-op when the mesh is flat.
                const double adjLit = shadingAdjointCorr(lv.wo, w * -1.0, lv.ns, ngoLit) * stGLit;
                fLit *= adjLit;
                // HERO CONNECTION. Both subpaths of this path index were drawn from ONE
                // bundle (see vcmPass), so lv's secondaries are hb.lam[i+1] as well and the
                // connection is exact per-λ — no cross-wavelength approximation, exactly as
                // in bdpt.h. Only the wavelengths live on BOTH ends can be connected, so the
                // number of usable λ is min(camera nUp, light-vertex nUp); the estimator
                // averages them (1/nUpConn) rather than boosting by C.
                const int lvUp = heroOn ? lightSec[(size_t)j].nUp : 1;
                const int nUpConn = (nUp < lvUp) ? nUp : lvUp;
                double fProdSec[hero::kHeroMax - 1] = {0}, mxProd = fCam * fLit;
                for (int i = 0; i + 1 < nUpConn; ++i) {
                    const double li = hb.lam[i + 1];
                    double fc = bsdfF(*mp, h.n, wo, w, li, scene, &h) * stGCam;
                    double fl = bsdfF(*lv.mat, lv.ns, lv.wo, w * -1.0, li, scene, &lv.hit) * adjLit;
                    fProdSec[i] = fc * fl;
                    if (fProdSec[i] > mxProd) mxProd = fProdSec[i];
                }
                if (mxProd <= 0.0) continue;
                double camDirPdfW = bsdfPdf(*mp, h.n, wo, w, lambda, scene, &h);
                double camRevPdfW = bsdfPdf(*mp, h.n, w, wo, lambda, scene, &h);
                double litDirPdfW = bsdfPdf(*lv.mat, lv.ns, lv.wo, w * -1.0, lambda, scene, &lv.hit);
                double litRevPdfW = bsdfPdf(*lv.mat, lv.ns, w * -1.0, lv.wo, lambda, scene, &lv.hit);
                double camDirPdfA = camDirPdfW * std::fabs(cosLit) / dist2;
                double litDirPdfA = litDirPdfW * std::fabs(cosCam) / dist2;
                double wLight = Mis(camDirPdfA) * (ctx.misVmWeight + lv.dVCM + lv.dVC * Mis(litRevPdfW));
                double wCamera = Mis(litDirPdfA) * (ctx.misVmWeight + dVCM + dVC * Mis(camRevPdfW));
                double misW = 1.0 / (wLight + 1.0 + wCamera);
                if (scene.occluded(offsetOrigin(h.p, h.ng, w), w, distc - 2e-6)) continue;
                double G = std::fabs(cosCam) * std::fabs(cosLit) / dist2;
                double k = misW * G;
                Vec3 add = cie * (k * fCam * fLit * beta * lv.beta);
                if (nUpConn > 1) {
                    const LightVertexSec& ls = lightSec[(size_t)j];
                    for (int i = 0; i + 1 < nUpConn; ++i)
                        add = add + cieSec[i] * (k * fProdSec[i] * betaSec[i] * ls.beta[i]);
                    add = add * (1.0 / (double)nUpConn);
                }
                result += add;
            }

            // (d) Vertex merging: gather nearby light vertices from ALL paths (XYZ estimate).
            if (ctx.vmNorm > 0.0) {
                Vec3 mergeXYZ{0, 0, 0};
                grid.query(lightVerts, h.p, ctx.radius, [&](int idx) {
                    const LightVertex& lv = lightVerts[idx];
                    if (edges + lv.edges > ctx.maxDepth) return;        // merged path too long
                    Vec3 wMerge = lv.wo;                                // incident illumination dir
                    // camera BSDF evaluated at the light vertex's wavelength (XYZ estimate).
                    double lam = (double)lv.lambda;
                    double fCam = bsdfF(*mp, h.n, wo, wMerge, lam, scene, &h);
                    // HERO MERGE. Unlike the connection above, a merge gathers vertices from
                    // OTHER paths, i.e. other bundles — so it is keyed on the LIGHT VERTEX's
                    // wavelengths (its hero plus its live secondaries), exactly as the
                    // single-λ version already keyed on lv.lambda. The camera-side throughput
                    // stays the hero's (the standard spectral-photon-mapping approximation,
                    // see the SPECTRAL NOTE at the top of this file); the per-λ part is the
                    // camera BSDF and the light vertex's own throughput. Averaged over the
                    // light vertex's live λ, so the merge is normalised like every other
                    // strategy here.
                    const int lvUp = heroOn ? lightSec[(size_t)idx].nUp : 1;
                    double fSec[hero::kHeroMax - 1] = {0}, mxF = fCam;
                    for (int k = 0; k + 1 < lvUp; ++k) {
                        fSec[k] = bsdfF(*mp, h.n, wo, wMerge,
                                        (double)lightSec[(size_t)idx].lam[k], scene, &h);
                        if (fSec[k] > mxF) mxF = fSec[k];
                    }
                    if (mxF <= 0.0) return;
                    // Gather-side shading-normal correction (cos_s/cos_g at the merge point):
                    // rebalances the geometric-cosine density estimate onto the shading cosine
                    // so a smooth mesh merges smoothly like mode R. 1 when h.n==h.ng (flat).
                    Vec3 ngoCam = (dot(h.ng, h.n) >= 0.0) ? h.ng : h.ng * -1.0;
                    const double gcorr = vmGatherCorr(wMerge, h.n, ngoCam);
                    fCam *= gcorr;
                    // The merge MIS weight is built from λ-independent densities, evaluated
                    // once at the light vertex's hero and shared by its secondaries.
                    double camDirPdfW = bsdfPdf(*mp, h.n, wo, wMerge, lam, scene, &h);
                    double camRevPdfW = bsdfPdf(*mp, h.n, wMerge, wo, lam, scene, &h);
                    double wLight = lv.dVCM * ctx.misVcWeight + lv.dVM * Mis(camDirPdfW);
                    double wCamera = dVCM * ctx.misVcWeight + dVM * Mis(camRevPdfW);
                    double misW = 1.0 / (wLight + 1.0 + wCamera);
                    Vec3 cieL(lv.cx, lv.cy, lv.cz);   // cached at store time (bit-identical)
                    Vec3 add = cieL * (misW * fCam * lv.beta);
                    if (lvUp > 1) {
                        const LightVertexSec& ls = lightSec[(size_t)idx];
                        for (int k = 0; k + 1 < lvUp; ++k)
                            add = add + Vec3(ls.cx[k], ls.cy[k], ls.cz[k]) *
                                        (misW * fSec[k] * gcorr * ls.beta[k]);
                        add = add * (1.0 / (double)lvUp);
                    }
                    mergeXYZ += add;
                });
                result += mergeXYZ * (beta * ctx.vmNorm);
            }
        }

        if (edges == ctx.maxDepth) return result;

        // Sample a continuation direction.
        Vec3 wi; double betaFactor, pdfW, pdfRevW, cosThetaOut; bool delta, terminate;
        double secF[hero::kHeroMax - 1]; bool secChromatic = false, keepBundle = false;
        scatterSample(scene, mats, mp, h, rd, lambda, rng, wi, betaFactor, pdfW, pdfRevW,
                      cosThetaOut, delta, terminate, stk,
                      hb.lam, nUp, secF, &secChromatic, &keepBundle);
        double mxF = betaFactor;      // max over live λ, never the hero alone (policy 3)
        if (secChromatic) for (int i = 0; i + 1 < nUp; ++i) if (secF[i] > mxF) mxF = secF[i];
        if (terminate || mxF <= 0.0) return result;
        if (!delta && (pdfW <= 0.0 || cosThetaOut <= 0.0)) return result;

        misScatter(delta, cosThetaOut, pdfW, pdfRevW, ctx.misVcWeight, ctx.misVmWeight,
                   dVCM, dVC, dVM);
        beta *= betaFactor;
        for (int i = 0; i + 1 < nUp; ++i) betaSec[i] *= secChromatic ? secF[i] : betaFactor;
        if (delta && !keepBundle) nUp = 1;      // de-hero (see traceLightSubpath)
        prevP = h.p;
        double sgn = dot(wi, h.ng) >= 0.0 ? 1.0 : -1.0;
        ray = Ray{h.p + h.ng * (sgn * 1e-6), normalize(wi)};
    }
    return result;
}

// Run ONE VCM pass at the given merge `radius`. Traces nLightPaths light subpaths (storing
// vertices + connect-to-camera splats), builds the grid, then traces nLightPaths camera
// subpaths (one per pixel) accumulating all strategies. Adds this pass's full per-pixel
// XYZ into st.accum and increments st.passes.
// `heroC` bundles that many wavelengths onto every traced ray (Wilkie et al. 2014; 1 == the
// plain single-λ estimator, bit-identical to before). The bundle is drawn ONCE PER PATH
// INDEX and shared by light path p and camera path p, so the vertex-connection strategies
// stay exact per-λ; merging, which crosses paths, is keyed on the light vertex's own
// wavelengths (see traceCameraSubpath).
inline void vcmPass(const Scene& scene, const Camera& cam, VcmState& st, double radius,
                    int nThreads, bool diffraction, int maxDepth, uint64_t passSeed,
                    int heroC = 1) {
    if (nThreads < 1) nThreads = 1;
    const int W = st.resX, H = st.resY;
    const long long nPix = (long long)W * H;

    PassCtx ctx;
    ctx.radius = radius;
    ctx.nLightPaths = (double)nPix;
    double etaVCM = PI * radius * radius * ctx.nLightPaths;
    ctx.misVcWeight = (etaVCM > 0.0) ? 1.0 / etaVCM : 0.0;
    ctx.misVmWeight = etaVCM;
    ctx.vmNorm = (etaVCM > 0.0) ? 1.0 / etaVCM : 0.0;
    ctx.imagePlaneDist = (double)W / (2.0 * cam.tanHalfX);
    ctx.maxDepth = maxDepth;
    ctx.diffraction = diffraction;

    // One wavelength BUNDLE per PATH INDEX (shared by light path p and camera path p so
    // their vertex connections are exact). Sampled up front from the emission importance.
    // heroC == 1 draws the single hero exactly as before (hero::sampleBundle with C == 1
    // is the plain emitSampler draw), so this path is bit-identical.
    const int C = (heroC < 1) ? 1 : (heroC > hero::kHeroMax ? hero::kHeroMax : heroC);
    std::vector<bdpt::HeroBundle> bundles((size_t)nPix);
    {
        Pcg32 rng; rng.seed(passSeed * 0x2545F4914F6CDD1DULL + 7, 0x9E3779B97F4A7C15ULL ^ passSeed);
        for (long long i = 0; i < nPix; ++i) {
            bdpt::HeroBundle& hb = bundles[(size_t)i];
            hb.C = C;
            double pdfA[hero::kHeroMax] = {0};
            if (!hero::sampleBundle(scene.emitSampler, rng.uniform(), C, hb.lam, pdfA)) {
                hb.C = 0;                      // dead bundle: skipped by both passes
                continue;
            }
            for (int k = 0; k < C; ++k)
                hb.invPdf[k] = (pdfA[k] > 0.0) ? scene.invPdfLambda(hb.lam[k]) : 0.0;
        }
    }

    // (1) Light pass — per-thread local vertex + splat buffers, concatenated in path order.
    std::vector<std::vector<LightVertex>> tVerts(nThreads);
    std::vector<std::vector<LightVertexSec>> tSec(nThreads);   // empty unless C > 1
    std::vector<std::vector<int>> tPathCount(nThreads);  // per-path stored-vertex count
    std::vector<std::vector<Vec3>> tSplat(nThreads);
    {
        auto lightWorker = [&](int tid) {
            Renderer mats; mats.diffraction = diffraction;
            Pcg32 rng; rng.seed(passSeed * 0x9E3779B97F4A7C15ULL + (uint64_t)tid * 2 + 101,
                                0xD1B54A32D192ED03ULL ^ ((uint64_t)tid << 7) ^ (passSeed << 1));
            long long i0 = nPix * tid / nThreads, i1 = nPix * (tid + 1) / nThreads;
            auto& verts = tVerts[tid]; auto& counts = tPathCount[tid];
            auto& secs = tSec[tid];
            auto& splat = tSplat[tid]; splat.assign((size_t)nPix, Vec3{0, 0, 0});
            counts.assign((size_t)(i1 - i0), 0);
            for (long long i = i0; i < i1; ++i) {
                const bdpt::HeroBundle& hb = bundles[(size_t)i];
                if (hb.C <= 0 || hb.invPdf[0] <= 0.0) { continue; }
                size_t before = verts.size();
                traceLightSubpath(scene, cam, mats, ctx, hb, rng, verts, secs, splat, W);
                counts[(size_t)(i - i0)] = (int)(verts.size() - before);
            }
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < nThreads; ++t) pool.emplace_back(lightWorker, t);
        for (auto& th : pool) th.join();
    }

    // Concatenate per-thread vertex buffers in path (pixel) order; build per-path ranges.
    std::vector<LightVertex> lightVerts;
    std::vector<LightVertexSec> lightSec;      // parallel to lightVerts; empty when C == 1
    {
        size_t total = 0; for (auto& v : tVerts) total += v.size();
        lightVerts.reserve(total);
        if (C > 1) lightSec.reserve(total);
    }
    std::vector<int> pathBegin((size_t)nPix, 0), pathEnd((size_t)nPix, 0);
    for (int t = 0; t < nThreads; ++t) {
        long long i0 = nPix * t / nThreads, i1 = nPix * (t + 1) / nThreads;
        int cursor = (int)lightVerts.size();
        for (long long i = i0; i < i1; ++i) {
            int cnt = tPathCount[t][(size_t)(i - i0)];
            pathBegin[(size_t)i] = cursor;
            pathEnd[(size_t)i] = cursor + cnt;
            cursor += cnt;
        }
        for (auto& lv : tVerts[t]) lightVerts.push_back(lv);
        for (auto& ls : tSec[t]) lightSec.push_back(ls);
    }

    // (2) Build the merge grid over all stored light vertices.
    VcmGrid grid; grid.build(lightVerts, radius);

    // Merge the per-thread splat buffers (connect-to-camera) into this pass's image.
    std::vector<Vec3> passImg((size_t)nPix, Vec3{0, 0, 0});
    for (int t = 0; t < nThreads; ++t)
        for (long long i = 0; i < nPix; ++i) passImg[(size_t)i] += tSplat[t][(size_t)i];

    // (3) Camera pass — one camera subpath per pixel; accumulate all camera-side strategies.
    {
        auto camWorker = [&](int tid) {
            Renderer mats; mats.diffraction = diffraction;
            Pcg32 rng; rng.seed(passSeed * 0xA24BAED4963EE407ULL + (uint64_t)tid * 2 + 211,
                                0xC2B2AE3D27D4EB4FULL ^ ((uint64_t)tid << 9) ^ (passSeed << 2));
            int y0 = H * tid / nThreads, y1 = H * (tid + 1) / nThreads;
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x) {
                    long long i = (long long)y * W + x;
                    const bdpt::HeroBundle& hb = bundles[(size_t)i];
                    if (hb.C <= 0 || hb.invPdf[0] <= 0.0) continue;
                    Vec3 c = traceCameraSubpath(scene, cam, mats, ctx, x, y, hb,
                                                rng, lightVerts, lightSec, pathBegin[(size_t)i],
                                                pathEnd[(size_t)i], grid);
                    passImg[(size_t)i] += c;
                }
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < nThreads; ++t) pool.emplace_back(camWorker, t);
        for (auto& th : pool) th.join();
    }

    // Accumulate this pass's image into the persistent sum.
    for (long long i = 0; i < nPix; ++i) st.accum[(size_t)i] += passImg[(size_t)i];
    st.passes += 1;
}

// Resolve the running average image (accum / passes).
inline Film vcmResolve(const VcmState& st) {
    Film f; f.resX = st.resX; f.resY = st.resY; f.alloc();
    double inv = (st.passes > 0) ? 1.0 / (double)st.passes : 0.0;
    for (long long i = 0; i < (long long)st.accum.size(); ++i) {
        f.xyz[(size_t)i] = st.accum[(size_t)i] * inv;
        f.hits[(size_t)i] = 1.0;
    }
    return f;
}

} // namespace vcm
