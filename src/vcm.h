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

// A stored light-subpath vertex (only connectible/non-delta surface vertices are kept).
struct LightVertex {
    Vec3   p;             // world position
    Vec3   ns, ng;        // shading / geometric normal
    Vec3   wo;            // unit dir toward the PREVIOUS light vertex (the fixed BSDF dir /
                          // reversed incident photon direction — used by connect AND merge)
    double beta = 0.0;    // geometric throughput carried here (single wavelength, no cie)
    float  lambda = 0.0f; // this subpath's wavelength (for the XYZ merge estimate)
    double dVCM = 0.0, dVC = 0.0, dVM = 0.0;
    int    matId = -1;
    const Material* mat = nullptr;
    Hit    hit;           // for textured albedo / roughness at merge & connect
    int    edges = 0;     // number of edges from the light emitter to this vertex
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
// `delta` the specular flag. `terminate` requests ending the walk. `interior` tracks the
// dielectric the path is inside (colored-glass Beer-Lambert). Media are out of VCM scope.
inline void scatterSample(const Scene& scene, const Renderer& mats, const Material* mp,
                          const Hit& h, const Vec3& rayDir, double lambda, Pcg32& rng,
                          Vec3& wi, double& betaFactor, double& pdfW, double& pdfRevW,
                          double& cosThetaOut, bool& delta, bool& terminate,
                          const Material*& interior) {
    const Vec3 ns = h.n;
    const Vec3 wo = normalize(rayDir * -1.0);
    wi = Vec3{0, 0, 0}; betaFactor = 0.0; pdfW = 0.0; pdfRevW = 0.0; cosThetaOut = 0.0;
    delta = false; terminate = false;
    switch (mp->type) {
        case MatType::Diffuse:
        case MatType::Fluorescent: {
            wi = cosineHemisphere(ns, rng);
            if (dot(wi, ns) <= 0) { terminate = true; break; }
            double rho = clamp01(diffuseReflectance(scene, *mp, h, lambda));
            pdfW = bsdfPdf(*mp, ns, wo, wi, lambda, scene, &h);
            pdfRevW = bsdfPdf(*mp, ns, wi, wo, lambda, scene, &h);
            betaFactor = rho;
            if (rho <= 0) terminate = true;
            break;
        }
        case MatType::Glossy: {
            Vec3 mdir = reflect(rayDir, ns);
            wi = sampleGlossy(mdir, materialRoughness(scene, *mp, h), rng);
            if (dot(wi, ns) <= 0) { terminate = true; break; }
            double r = clamp01(mp->reflect(lambda));
            pdfW = bsdfPdf(*mp, ns, wo, wi, lambda, scene, &h);
            pdfRevW = bsdfPdf(*mp, ns, wi, wo, lambda, scene, &h);
            betaFactor = r;
            if (r <= 0 || pdfW <= 0) terminate = true;
            break;
        }
        case MatType::DiffuseTransmit: {
            double rhoR, rhoT; bdpt::diffuseTransmitAlbedos(*mp, lambda, scene, &h, rhoR, rhoT);
            double tot = rhoR + rhoT;
            if (tot <= 0.0) { terminate = true; break; }
            if (rng.uniform() * tot < rhoR) wi = cosineHemisphere(ns, rng);
            else                            wi = cosineHemisphere(ns * -1.0, rng);
            pdfW = bsdfPdf(*mp, ns, wo, wi, lambda, scene, &h);
            pdfRevW = bsdfPdf(*mp, ns, wi, wo, lambda, scene, &h);
            betaFactor = tot;
            if (pdfW <= 0) terminate = true;
            break;
        }
        case MatType::Mirror: {
            double r = clamp01(mp->reflect(lambda));
            wi = reflect(rayDir, ns); betaFactor = r; delta = true;
            if (r <= 0) terminate = true;
            break;
        }
        case MatType::Dielectric: {
            bool entering = dot(rayDir, h.ng) < 0.0;
            bool transmitted = false;
            Ray nr = mats.refractOrReflect(scene, *mp, h, rayDir, lambda, rng, &transmitted);
            wi = nr.d; betaFactor = 1.0; delta = true;
            if (transmitted) interior = entering ? mp : nullptr;
            break;
        }
        case MatType::HalfMirror: {
            double r = clamp01(mp->reflect(lambda));
            if (rng.uniform() < r) wi = reflect(rayDir, ns);
            else                   wi = rayDir;
            betaFactor = 1.0; delta = true;
            break;
        }
        case MatType::Filter: {
            double t = clamp01(mp->transmit(lambda));
            wi = rayDir; betaFactor = t; delta = true;
            if (t <= 0) terminate = true;
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
            double r = clamp01(mp->reflect(lambda));
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
                              const PassCtx& ctx, double lambda, double invPdfLambda,
                              Pcg32& rng, std::vector<LightVertex>& out,
                              std::vector<Vec3>& splat, int W) {
    if (scene.emitters.empty() || scene.totalPower <= 0.0) return;
    int ei = scene.selectEmitter(rng);
    const Emitter& em = scene.emitters[ei];
    if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env || em.collimated) return;

    double u1 = rng.uniform(), u2 = rng.uniform();
    Vec3 y, nOut; em.samplePoint(u1, u2, y, nOut);
    double Le = em.spdFn(lambda) * invPdfLambda;
    if (Le <= 0.0) return;
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

    // Geometric throughput leaving the light (Le carries the spectral radiance).
    double beta = Le * cosLight / emissionPdfW;

    // Running MIS quantities (SmallVCM GenerateLightSample).
    double dVCM = Mis(directPdfW / emissionPdfW);
    double dVC  = Mis(cosLight / emissionPdfW);
    double dVM  = dVC * ctx.misVcWeight;

    const Vec3 cie(cieX(lambda), cieY(lambda), cieZ(lambda));
    const Material* interior = nullptr;
    Vec3 prevP = y;
    Ray ray{y + nOut * 1e-6, dir};

    for (int edges = 1; edges <= ctx.maxDepth; ++edges) {
        Hit h = scene.closestHit(ray);
        if (!h.valid) return;                    // escaped (no env in scope)
        if (interior) {
            double a = interior->absorb(lambda);
            if (a > 0.0) beta *= std::exp(-a * h.t);
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

        // Store the vertex + connect to camera (only for connectible, non-delta surfaces).
        bool connectible = isConnectibleMat(*mp);
        if (connectible) {
            LightVertex lv;
            lv.p = h.p; lv.ns = h.n; lv.ng = h.ng; lv.wo = wo;
            lv.beta = beta; lv.lambda = (float)lambda;
            lv.dVCM = dVCM; lv.dVC = dVC; lv.dVM = dVM;
            lv.matId = h.matId; lv.mat = mp; lv.hit = h; lv.edges = edges;
            out.push_back(lv);

            // Connect this vertex to the pinhole camera (light-image splat, t=1). The
            // path here has (edges + 1) edges: skip if it would exceed maxDepth.
            if (!cam.hasLens() && edges + 1 <= ctx.maxDepth) {
                Vec3 toCam = cam.eye - h.p;
                double dist2c = dot(toCam, toCam);
                if (dist2c > 1e-12) {
                    double distc = std::sqrt(dist2c);
                    Vec3 wcam = toCam / distc;
                    double cosToCamera = dot(h.n, wcam);
                    bool sideOk = isTwoSidedMat(*mp) ? (cosToCamera != 0.0) : (cosToCamera > 0.0);
                    double cosAtCamera = dot(cam.w, wcam * -1.0);   // forward vs camera->point
                    if (sideOk && cosAtCamera > 1e-9) {
                        int px, py; double cc, d2c;
                        if (cam.project(h.p, px, py, cc, d2c)) {
                            double f = bsdfF(*mp, h.n, wo, wcam, lambda, scene, &h);
                            if (f > 0.0 &&
                                !scene.occluded(offsetOrigin(h.p, h.ng, wcam), wcam, distc - 2e-6)) {
                                double bsdfRevPdfW = bsdfPdf(*mp, h.n, wcam, wo, lambda, scene, &h);
                                double imgPtDist = ctx.imagePlaneDist / cosAtCamera;
                                double imgToSolid = imgPtDist * imgPtDist / cosAtCamera;
                                double imgToSurf = imgToSolid * std::fabs(cosToCamera) / dist2c;
                                double wLight = Mis(imgToSurf / ctx.nLightPaths) *
                                                (ctx.misVmWeight + dVCM + dVC * Mis(bsdfRevPdfW));
                                double misW = 1.0 / (wLight + 1.0);
                                // contrib = misW * beta * f * (cameraAreaPdf / nLightPaths)
                                double contrib = misW * beta * f * imgToSurf / ctx.nLightPaths;
                                if (contrib > 0.0)
                                    splat[(size_t)py * W + px] += cie * contrib;
                            }
                        }
                    }
                }
            }
        }

        if (edges == ctx.maxDepth) return;       // no room to continue

        // Sample a continuation direction.
        Vec3 wi; double betaFactor, pdfW, pdfRevW, cosThetaOut; bool delta, terminate;
        scatterSample(scene, mats, mp, h, rd, lambda, rng, wi, betaFactor, pdfW, pdfRevW,
                      cosThetaOut, delta, terminate, interior);
        if (terminate || betaFactor <= 0.0) return;
        if (!delta && (pdfW <= 0.0 || cosThetaOut <= 0.0)) return;

        misScatter(delta, cosThetaOut, pdfW, pdfRevW, ctx.misVcWeight, ctx.misVmWeight,
                   dVCM, dVC, dVM);
        beta *= betaFactor;
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
                               const PassCtx& ctx, int px, int py, double lambda, double invPdfLambda,
                               Pcg32& rng, const std::vector<LightVertex>& lightVerts,
                               int pathBegin, int pathEnd, const VcmGrid& grid) {
    Vec3 result{0, 0, 0};
    const Vec3 cie(cieX(lambda), cieY(lambda), cieZ(lambda));

    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
    double cosAtCamera = dot(ray.d, cam.w);
    if (cosAtCamera <= 1e-9) return result;
    double cameraPdfW = ctx.imagePlaneDist * ctx.imagePlaneDist / (cosAtCamera * cosAtCamera * cosAtCamera);

    double beta = 1.0;
    double dVCM = Mis(ctx.nLightPaths / cameraPdfW);
    double dVC = 0.0, dVM = 0.0;
    const Material* interior = nullptr;
    Vec3 prevP = cam.eye;

    for (int edges = 1; edges <= ctx.maxDepth; ++edges) {
        Hit h = scene.closestHit(ray);
        if (!h.valid) {
            // Directly-viewed environment (no env in scope; ignore).
            return result;
        }
        if (interior) {
            double a = interior->absorb(lambda);
            if (a > 0.0) beta *= std::exp(-a * h.t);
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
                double Le = mp->emit(lambda) * invPdfLambda;
                if (Le > 0.0) {
                    double misW = 1.0;
                    const Emitter* em = scene.emitterForMat(h.matId);
                    if (em && em->area > 0.0 && scene.totalPower > 0.0 && edges >= 2) {
                        double pdfChoice = em->power / scene.totalPower;
                        double directPdfA = pdfChoice / em->area;         // area pdf of the point
                        double emissionPdfW = pdfChoice * cosLight / PI;  // directional emission pdf
                        double wCamera = Mis(directPdfA) * dVCM + Mis(emissionPdfW) * dVC;
                        misW = 1.0 / (1.0 + wCamera);
                    }
                    result += cie * (beta * Le * misW);
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
                if (!(em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env || em.collimated)) {
                    double u1 = rng.uniform(), u2 = rng.uniform();
                    Vec3 yL, nL; em.samplePoint(u1, u2, yL, nL);
                    Vec3 toL = yL - h.p; double dist2 = dot(toL, toL);
                    if (dist2 > 1e-12) {
                        double distL = std::sqrt(dist2); Vec3 wi = toL / distL;
                        double cosAtLight = dot(nL, wi * -1.0);
                        double cosToLight = dot(h.n, wi);
                        bool sideOk = isTwoSidedMat(*mp) ? (cosToLight != 0.0) : (cosToLight > 0.0);
                        if (cosAtLight > 0.0 && sideOk) {
                            double f = bsdfF(*mp, h.n, wo, wi, lambda, scene, &h);
                            double Le = em.spdFn(lambda) * invPdfLambda;
                            if (f > 0.0 && Le > 0.0 && em.area > 0.0 &&
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
                                double contrib = misW * std::fabs(cosToLight) /
                                                 (pdfChoice * directPdfW) * Le * f;
                                if (contrib > 0.0) result += cie * (beta * contrib);
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
                bool camSide = isTwoSidedMat(*mp) ? (cosCam != 0.0) : (cosCam > 0.0);
                bool litSide = isTwoSidedMat(*lv.mat) ? (cosLit != 0.0) : (cosLit > 0.0);
                if (!camSide || !litSide) continue;
                double fCam = bsdfF(*mp, h.n, wo, w, lambda, scene, &h);
                double fLit = bsdfF(*lv.mat, lv.ns, lv.wo, w * -1.0, lambda, scene, &lv.hit);
                if (fCam <= 0.0 || fLit <= 0.0) continue;
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
                double contrib = misW * G * fCam * fLit * beta * lv.beta;
                if (contrib > 0.0) result += cie * contrib;
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
                    if (fCam <= 0.0) return;
                    double camDirPdfW = bsdfPdf(*mp, h.n, wo, wMerge, lam, scene, &h);
                    double camRevPdfW = bsdfPdf(*mp, h.n, wMerge, wo, lam, scene, &h);
                    double wLight = lv.dVCM * ctx.misVcWeight + lv.dVM * Mis(camDirPdfW);
                    double wCamera = dVCM * ctx.misVcWeight + dVM * Mis(camRevPdfW);
                    double misW = 1.0 / (wLight + 1.0 + wCamera);
                    Vec3 cieL(cieX(lv.lambda), cieY(lv.lambda), cieZ(lv.lambda));
                    mergeXYZ += cieL * (misW * fCam * lv.beta);
                });
                result += mergeXYZ * (beta * ctx.vmNorm);
            }
        }

        if (edges == ctx.maxDepth) return result;

        // Sample a continuation direction.
        Vec3 wi; double betaFactor, pdfW, pdfRevW, cosThetaOut; bool delta, terminate;
        scatterSample(scene, mats, mp, h, rd, lambda, rng, wi, betaFactor, pdfW, pdfRevW,
                      cosThetaOut, delta, terminate, interior);
        if (terminate || betaFactor <= 0.0) return result;
        if (!delta && (pdfW <= 0.0 || cosThetaOut <= 0.0)) return result;

        misScatter(delta, cosThetaOut, pdfW, pdfRevW, ctx.misVcWeight, ctx.misVmWeight,
                   dVCM, dVC, dVM);
        beta *= betaFactor;
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
inline void vcmPass(const Scene& scene, const Camera& cam, VcmState& st, double radius,
                    int nThreads, bool diffraction, int maxDepth, uint64_t passSeed) {
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

    // One wavelength per PATH INDEX (shared by light path p and camera path p so their
    // vertex connections are exact). Sampled up front from the emission importance.
    std::vector<double> lam((size_t)nPix), invLam((size_t)nPix);
    {
        Pcg32 rng; rng.seed(passSeed * 0x2545F4914F6CDD1DULL + 7, 0x9E3779B97F4A7C15ULL ^ passSeed);
        for (long long i = 0; i < nPix; ++i) {
            double pdfL = 0.0; double l = scene.emitSampler.sample(rng, pdfL);
            lam[i] = l; invLam[i] = (pdfL > 0.0) ? scene.invPdfLambda(l) : 0.0;
        }
    }

    // (1) Light pass — per-thread local vertex + splat buffers, concatenated in path order.
    std::vector<std::vector<LightVertex>> tVerts(nThreads);
    std::vector<std::vector<int>> tPathCount(nThreads);  // per-path stored-vertex count
    std::vector<std::vector<Vec3>> tSplat(nThreads);
    {
        auto lightWorker = [&](int tid) {
            Renderer mats; mats.diffraction = diffraction;
            Pcg32 rng; rng.seed(passSeed * 0x9E3779B97F4A7C15ULL + (uint64_t)tid * 2 + 101,
                                0xD1B54A32D192ED03ULL ^ ((uint64_t)tid << 7) ^ (passSeed << 1));
            long long i0 = nPix * tid / nThreads, i1 = nPix * (tid + 1) / nThreads;
            auto& verts = tVerts[tid]; auto& counts = tPathCount[tid];
            auto& splat = tSplat[tid]; splat.assign((size_t)nPix, Vec3{0, 0, 0});
            counts.assign((size_t)(i1 - i0), 0);
            for (long long i = i0; i < i1; ++i) {
                if (invLam[i] <= 0.0) { continue; }
                size_t before = verts.size();
                traceLightSubpath(scene, cam, mats, ctx, lam[i], invLam[i], rng, verts, splat, W);
                counts[(size_t)(i - i0)] = (int)(verts.size() - before);
            }
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < nThreads; ++t) pool.emplace_back(lightWorker, t);
        for (auto& th : pool) th.join();
    }

    // Concatenate per-thread vertex buffers in path (pixel) order; build per-path ranges.
    std::vector<LightVertex> lightVerts;
    {
        size_t total = 0; for (auto& v : tVerts) total += v.size();
        lightVerts.reserve(total);
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
                    if (invLam[i] <= 0.0) continue;
                    Vec3 c = traceCameraSubpath(scene, cam, mats, ctx, x, y, lam[i], invLam[i],
                                                rng, lightVerts, pathBegin[(size_t)i],
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
