// Bidirectional path tracing (BDPT) — mode 'D'. An additive, fully unbiased
// estimator that traces a subpath from the camera AND a subpath from a light,
// then connects every pair of vertices, MIS-combining all strategies (balance
// heuristic). Unlike mode B (forward light tracing, which renders specular/glossy
// surfaces black — the SDS limitation) and mode P (a calibrated forward+backward
// composite with a seam), BDPT produces one absolute-radiance image in a single
// estimator: it is light-tracing where that wins (caustics on diffuse walls) and
// path-tracing where that wins (directly-viewed specular), with no calibration.
//
// Structure follows Veach / PBRT-v3 (GenerateCameraSubpath, GenerateLightSubpath,
// ConnectBDPT, MISWeight) adapted to this renderer's single-wavelength spectral
// Monte Carlo: every quantity that PBRT carries as a Spectrum is a scalar radiance
// at the one sampled wavelength lambda; both subpaths share that lambda.
//
// Material scope: Diffuse and Glossy are CONNECTIBLE (non-delta) vertices; the
// specular family (Dielectric, Mirror, HalfMirror, ThinFilm, Multilayer, Grating)
// are delta pass-through vertices that carry a chain but never connect (their
// connection pdf is zero). Emission comes from area/sphere quad lights. Fluorescence,
// participating media, spot and environment lights are NOT handled here (a scene
// using them should be rendered with mode B/P/R instead); see the guard in main.cpp.
#pragma once
#include <vector>
#include <algorithm>
#include "scene.h"
#include "camera.h"
#include "render.h"   // sampleGlossy, Renderer material primitives, clamp01, PI

namespace bdpt {

// Which direction a subpath transports. Radiance = from the camera (eye subpath);
// Importance = from a light (light subpath). Only affects the (non-reciprocal)
// glossy lobe's cosine-denominator choice so both subpaths stay consistent with how
// the forward/backward tracers sample that lobe.
enum class Mode { Radiance, Importance };

// Power-cosine glossy exponent, matching render.h's sampleGlossy exactly.
inline double glossyExponent(double roughness) {
    double rr = roughness < 1e-3 ? 1e-3 : roughness;
    double e = 2.0 / (rr * rr) - 2.0;
    return e < 0 ? 0 : e;
}

// Is this material a connectible (non-delta) surface for BDPT? Diffuse and Glossy
// have a finite BSDF value we can evaluate on an arbitrary connection direction;
// the specular family is delta (zero connection pdf) and only forms chains.
inline bool isConnectibleMat(const Material& m) {
    return m.type == MatType::Diffuse || m.type == MatType::Glossy ||
           m.type == MatType::Fluorescent;   // fluoro's elastic base is diffuse-like
}

// Evaluate the BSDF value f at a surface vertex for the pair (wo, wi), both unit
// world directions pointing AWAY from the surface. `wo` is toward where the subpath
// came from; `wi` is the connection/continuation direction. Returns 0 for a delta
// material (no finite value) or for directions on opposite sides than the lobe
// supports. `ns` is the shading normal. Consistent with render.h sampling: on the
// sampled direction, f*|cos(wi)|/pdf(wi) equals the throughput factor (rho or r).
inline double bsdfF(const Material& m, const Vec3& ns, const Vec3& wo, const Vec3& wi,
                    double lambda, const Scene& scene, const Hit* hitForTex) {
    double cosWi = dot(wi, ns), cosWo = dot(wo, ns);
    switch (m.type) {
        case MatType::Diffuse: {
            if (cosWi <= 0 || cosWo <= 0) return 0.0;   // same-hemisphere reflection only
            double rho = hitForTex ? clamp01(diffuseReflectance(scene, m, *hitForTex, lambda))
                                   : clamp01(m.reflect(lambda));
            return rho / PI;
        }
        case MatType::Fluorescent: {
            // Only the elastic (wavelength-preserving) diffuse base connects; the
            // Stokes-shifted re-emission is a wavelength change we don't connect.
            if (cosWi <= 0 || cosWo <= 0) return 0.0;
            double rho = clamp01(m.reflect(lambda));
            return rho / PI;
        }
        case MatType::Glossy: {
            if (cosWi <= 0 || cosWo <= 0) return 0.0;
            double r = clamp01(m.reflect(lambda));
            double e = glossyExponent(hitForTex ? materialRoughness(scene, m, *hitForTex)
                                                : m.roughness);
            // Mirror direction of the outgoing ray about ns, as render.h forms it:
            // sampleGlossy lobes around reflect(rayDir, n) with rayDir = -wo.
            Vec3 mdir = reflect(wo * -1.0, ns);
            double cosLobe = dot(wi, mdir);
            if (cosLobe <= 0) return 0.0;
            double lobe = (e + 1.0) / (2.0 * PI) * std::pow(cosLobe, e);
            return r * lobe / cosWi;   // denom = sampled-direction cosine (see header)
        }
        default: return 0.0;   // delta materials have no finite BSDF value
    }
}

// Directional pdf (solid angle) of sampling `wi` at a surface vertex given the
// subpath arrived along `wo` (incoming ray dir = -wo). Matches render.h's sampling
// densities. 0 for delta materials (handled separately) or unsupported hemispheres.
// `hitForTex` (with `scene`) supplies the per-hit roughness when a roughness map is
// bound, so the density matches the sampling that used the same textured roughness —
// essential for unbiased MIS. Pass nullptr where no hit UV is available (constant).
inline double bsdfPdf(const Material& m, const Vec3& ns, const Vec3& wo, const Vec3& wi,
                      const Scene& scene, const Hit* hitForTex) {
    double cosWi = dot(wi, ns), cosWo = dot(wo, ns);
    switch (m.type) {
        case MatType::Diffuse:
        case MatType::Fluorescent: {
            if (cosWi <= 0 || cosWo <= 0) return 0.0;
            return cosWi / PI;                       // cosine-weighted hemisphere
        }
        case MatType::Glossy: {
            if (cosWi <= 0 || cosWo <= 0) return 0.0;
            double e = glossyExponent(hitForTex ? materialRoughness(scene, m, *hitForTex)
                                                : m.roughness);
            Vec3 mdir = reflect(wo * -1.0, ns);
            double cosLobe = dot(wi, mdir);
            if (cosLobe <= 0) return 0.0;
            return (e + 1.0) / (2.0 * PI) * std::pow(cosLobe, e);
        }
        default: return 0.0;
    }
}

// --- Camera (pinhole) importance, PBRT-v3 convention -----------------------------
// Uses the FULL image-plane area A (imagePlaneArea), NOT the per-pixel area that mode
// B's connect() uses. This is essential for correct MIS: the camera-subpath sampling
// density must be expressed over the whole image plane so the t=1 light-tracing
// strategy gets a FAIR balance-heuristic weight (with a per-pixel area A the camera
// pdf is W*H too large, which crushes the light-tracing weight to ~0 and collapses
// BDPT into a plain path tracer — losing its caustic/SDS advantage). The absolute
// radiance scale is preserved because the light image is normalised by the per-pixel
// sample count spp (not W*H*spp): (1/spp)*We(A_full) == (1/(W*H*spp))*We(A_pixel).
// cosCam = cosine between the camera forward axis w and the ray from eye to point.
//   We(cosCam)     = 1 / (A * cosCam^4)   (importance value)
//   pdfDir(cosCam) = 1 / (A * cosCam^3)   (importance-sampling density)
inline double cameraWe(const Camera& cam, double cosCam) {
    if (cosCam <= 0) return 0.0;
    double c2 = cosCam * cosCam;
    return 1.0 / (cam.imagePlaneArea() * c2 * c2);
}
inline double cameraPdfDir(const Camera& cam, double cosCam) {
    if (cosCam <= 0) return 0.0;
    return 1.0 / (cam.imagePlaneArea() * cosCam * cosCam * cosCam);
}

// --- Path vertex -----------------------------------------------------------------
enum class VType { Camera, Light, Surface };

struct Vertex {
    VType type = VType::Surface;
    Vec3 p{0, 0, 0};      // world position
    Vec3 ns{0, 0, 0};     // shading normal (used for all cosines/BSDF)
    Vec3 ng{0, 0, 0};     // geometric normal (orientation / emission side)
    double beta = 0.0;    // throughput carried to this vertex (single-wavelength)
    // Area-measure pdfs of sampling THIS vertex from the previous / next vertex
    // along the two transport directions. Delta vertices store 0 (skipped in MIS).
    double pdfFwd = 0.0;
    double pdfRev = 0.0;
    bool delta = false;   // specular vertex (no connection, delta pdf)

    // Surface data
    int matId = -1;
    const Material* mat = nullptr;   // resolved material (Mix already resolved)
    Hit hit;                         // full hit (for textured albedo / u,v)

    // Light data (type == Light, or a Surface that is emissive)
    const Emitter* light = nullptr;

    bool onSurface() const { return type == VType::Surface || type == VType::Light; }
    bool isConnectible() const {
        if (type == VType::Camera) return true;      // pinhole connects (delta pos handled)
        if (type == VType::Light)  return light && !light->collimated;
        return mat && !delta && isConnectibleMat(*mat);
    }
    // Emitted radiance (single wavelength) leaving this vertex toward direction w,
    // if it is (or sits on) a light. 0 otherwise or if w is on the unlit side.
    double Le(const Vec3& w, double lambda, double invPdfLambda) const {
        if (!mat || !mat->isLight) return 0.0;
        if (dot(ng, w) <= 0.0) return 0.0;           // one-sided emitter
        return mat->emit(lambda) * invPdfLambda;
    }
    bool isLightVertex() const {
        return type == VType::Light || (type == VType::Surface && mat && mat->isLight);
    }
};

// Convert a solid-angle pdf `pdfW` of leaving `from` toward `to` into an area-
// measure density at `to` (PBRT's Vertex::ConvertDensity). Surfaces pick up the
// projected-cosine Jacobian; the 1/dist^2 is always applied.
inline double convertDensity(double pdfW, const Vertex& from, const Vertex& to) {
    Vec3 w = to.p - from.p;
    double d2 = dot(w, w);
    if (d2 == 0.0) return 0.0;
    double invD2 = 1.0 / d2;
    if (to.onSurface()) pdfW *= std::abs(dot(to.ns, w * std::sqrt(invD2)));
    return pdfW * invD2;
}

// RAII temporary field mutation, restored on scope exit — PBRT's ScopedAssignment.
// MISWeight temporarily rewrites a few vertices' reverse pdfs / delta flags to
// evaluate hypothetical strategies, then rolls them back.
template <typename T>
struct ScopedAssign {
    T* target = nullptr;
    T backup{};
    ScopedAssign() = default;
    ScopedAssign(T* t, T v) : target(t), backup(*t) { *t = v; }
    ScopedAssign(const ScopedAssign&) = delete;
    ScopedAssign& operator=(const ScopedAssign&) = delete;
    ScopedAssign& operator=(ScopedAssign&& o) noexcept {
        if (target) *target = backup;
        target = o.target; backup = o.backup; o.target = nullptr;
        return *this;
    }
    ~ScopedAssign() { if (target) *target = backup; }
};

// cos between the camera forward axis and the ray eye->p (for camera importance).
inline double camCos(const Camera& cam, const Vec3& p) {
    Vec3 d = p - cam.eye;
    double len = length(d);
    return len > 0 ? dot(d, cam.w) / len : 0.0;
}

// Area-measure pdf of sampling `next` by scattering at `cur` (arriving from `prev`),
// PBRT's Vertex::Pdf. For a Light `cur` this is the emission density (pdfLight).
inline double vertexPdf(const Scene& scene, const Camera& cam,
                        const Vertex* prev, const Vertex& cur, const Vertex& next);
inline double vertexPdfLight(const Camera& cam, const Vertex& cur, const Vertex& next);

// Emission directional density at a light vertex `cur` toward `next`, area measure.
inline double vertexPdfLight(const Camera& /*cam*/, const Vertex& cur, const Vertex& next) {
    Vec3 w = next.p - cur.p;
    double d2 = dot(w, w);
    if (d2 == 0.0) return 0.0;
    double invD2 = 1.0 / d2;
    w = w * std::sqrt(invD2);
    double cosLight = dot(cur.ng, w);               // one-sided Lambertian emitter
    if (cosLight <= 0.0) return 0.0;
    double pdfW = cosLight / PI;                     // cosine-weighted emission
    double pdf = pdfW * invD2;
    if (next.onSurface()) pdf *= std::abs(dot(next.ns, w));
    return pdf;
}

inline double vertexPdf(const Scene& scene, const Camera& cam,
                        const Vertex* prev, const Vertex& cur, const Vertex& next) {
    if (cur.type == VType::Light) return vertexPdfLight(cam, cur, next);
    Vec3 wn = next.p - cur.p;
    if (dot(wn, wn) == 0.0) return 0.0;
    wn = normalize(wn);
    double pdfW = 0.0;
    if (cur.type == VType::Camera) {
        pdfW = cameraPdfDir(cam, camCos(cam, next.p));
    } else {                                         // Surface
        if (!prev || !cur.mat) return 0.0;
        Vec3 wp = prev->p - cur.p;
        if (dot(wp, wp) == 0.0) return 0.0;
        wp = normalize(wp);
        pdfW = bsdfPdf(*cur.mat, cur.ns, wp, wn, scene, &cur.hit);
    }
    return convertDensity(pdfW, cur, next);
}

// Positional density (area measure) of sampling this light vertex's ORIGIN via
// light sampling = P(choose this emitter) * (1/area). PBRT's PdfLightOrigin, minus
// the infinite-light branch (unsupported in BDPT scope).
inline double vertexPdfLightOrigin(const Scene& scene, const Vertex& cur) {
    if (!cur.light || scene.totalPower <= 0.0 || cur.light->area <= 0.0) return 0.0;
    double pdfChoice = cur.light->power / scene.totalPower;
    return pdfChoice / cur.light->area;
}

// --- Random walk -----------------------------------------------------------------
// Continue a subpath whose endpoint is already path[0] (Camera or Light). `ray` is
// the first ray leaving that endpoint; `beta` the throughput carried along it;
// `pdfDir` the solid-angle density of that first direction; `mode` the transport
// direction. Appends surface vertices until a miss, absorption, or maxDepth. No
// environment/medium handling (BDPT scope). Uses `mats` for specular primitives.
inline void randomWalk(const Scene& scene, const Camera& cam, const Renderer& mats,
                       Ray ray, double beta, double pdfDir, double lambda,
                       int maxDepth, Mode mode, Pcg32& rng, std::vector<Vertex>& path) {
    (void)cam; (void)mode;   // cam/mode reserved for future NEE-to-camera & adjoint use
    if (maxDepth == 0) return;
    double pdfFwd = pdfDir;   // solid-angle density of the current ray direction
    const Material* interior = nullptr;   // dielectric the subpath is inside (colored glass)
    for (int bounces = 0;;) {
        Hit h = scene.closestHit(ray);
        if (!h.valid) return;                        // escaped (no env in BDPT scope)
        if (h.sensorId >= 0) return;                 // model-A sensor: not used in BDPT

        // Beer-Lambert attenuation over the in-glass segment just traversed. NOTE:
        // this attenuates only the *subpath walk*; connection edges (connectBDPT)
        // that cross glass are NOT absorption-weighted (see known-issues.md).
        if (interior) {
            double a = interior->absorb(lambda);
            if (a > 0.0) beta *= std::exp(-a * h.t);
        }

        // Resolve material (Mix -> child, or absorbed on the leftover slice).
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return;                       // absorbed
            mp = &scene.mats[c];
        }

        Vertex v;
        v.type = VType::Surface;
        v.p = h.p; v.ns = h.n; v.ng = h.ng; v.hit = h;
        v.matId = h.matId; v.mat = mp; v.beta = beta;
        if (mp->isLight) v.light = scene.emitterForMat(h.matId);
        Vertex& prev = path.back();
        v.pdfFwd = convertDensity(pdfFwd, prev, v);
        path.push_back(v);
        Vertex& cur = path.back();
        if (++bounces >= maxDepth) return;

        // Sample a continuation direction wi, its forward solid-angle pdf pdfW, the
        // reverse pdf pdfRevW (wi<->wo swapped), the throughput factor, and whether
        // this vertex is a delta (specular) scatter.
        Vec3 wo = normalize(prev.p - cur.p);         // toward the previous vertex
        Vec3 wi; double pdfW = 0.0, pdfRevW = 0.0, betaFactor = 0.0;
        bool delta = false, terminate = false;
        switch (mp->type) {
            case MatType::Diffuse:
            case MatType::Fluorescent: {              // elastic base only (see header)
                wi = cosineHemisphere(cur.ns, rng);
                if (dot(wi, cur.ns) <= 0) { terminate = true; break; }
                double rho = clamp01(diffuseReflectance(scene, *mp, h, lambda));
                pdfW = bsdfPdf(*mp, cur.ns, wo, wi, scene, &h);
                pdfRevW = bsdfPdf(*mp, cur.ns, wi, wo, scene, &h);
                betaFactor = rho;                     // f*cos/pdf = rho
                if (rho <= 0) terminate = true;
                break;
            }
            case MatType::Glossy: {
                Vec3 mdir = reflect(ray.d, cur.ns);   // ray.d == -wo (incoming dir)
                wi = sampleGlossy(mdir, materialRoughness(scene, *mp, h), rng);
                if (dot(wi, cur.ns) <= 0) { terminate = true; break; }
                double r = clamp01(mp->reflect(lambda));
                pdfW = bsdfPdf(*mp, cur.ns, wo, wi, scene, &h);
                pdfRevW = bsdfPdf(*mp, cur.ns, wi, wo, scene, &h);
                betaFactor = r;                       // f*cos/pdf = r
                if (r <= 0 || pdfW <= 0) terminate = true;
                break;
            }
            case MatType::Mirror: {
                double r = clamp01(mp->reflect(lambda));
                wi = reflect(ray.d, cur.ns);
                betaFactor = r; delta = true;
                if (r <= 0) terminate = true;
                break;
            }
            case MatType::Dielectric: {
                bool entering = dot(ray.d, h.ng) < 0.0;
                bool transmitted = false;
                Ray nr = mats.refractOrReflect(scene, *mp, h, ray.d, lambda, rng, &transmitted);
                wi = nr.d; betaFactor = 1.0; delta = true;
                if (transmitted) interior = entering ? mp : nullptr;
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(mp->reflect(lambda));
                if (rng.uniform() < r) wi = reflect(ray.d, cur.ns);
                else                   wi = ray.d;    // transmit straight
                betaFactor = 1.0; delta = true;
                break;
            }
            case MatType::ThinFilm: {
                Ray nr;
                if (!mats.thinFilmInterface(scene, *mp, h, ray.d, lambda, rng, nr)) { terminate = true; break; }
                wi = nr.d; betaFactor = 1.0; delta = true;
                break;
            }
            case MatType::Multilayer: {
                Ray nr;
                if (!mats.multilayerInterface(*mp, h, ray.d, lambda, rng, nr)) { terminate = true; break; }
                wi = nr.d; betaFactor = 1.0; delta = true;
                break;
            }
            case MatType::Grating: {
                double r = clamp01(mp->reflect(lambda));
                if (r <= 0) { terminate = true; break; }
                bool absorbedG; Ray nr = mats.gratingDiffract(*mp, h, ray.d, lambda, rng, absorbedG);
                if (absorbedG) { terminate = true; break; }
                wi = nr.d; betaFactor = r; delta = true;
                break;
            }
            default: terminate = true; break;
        }
        if (terminate || betaFactor <= 0.0) return;

        // Specular vertices carry a delta density: PBRT stores 0 for both the forward
        // and reverse area densities so MIS skips connections through them.
        cur.delta = delta;
        if (delta) { pdfW = 0.0; pdfRevW = 0.0; }

        // The reverse density flows back to the previous vertex (area measure).
        prev.pdfRev = convertDensity(pdfRevW, cur, prev);

        beta *= betaFactor;
        // Spawn the continuation from the correct side of the geometric normal.
        double sgn = dot(wi, cur.ng) >= 0.0 ? 1.0 : -1.0;
        ray = Ray{cur.p + cur.ng * (sgn * 1e-6), normalize(wi)};
        pdfFwd = delta ? 0.0 : pdfW;
    }
}

// Trace an eye subpath through pixel (px,py) with sub-pixel jitter. path[0] is the
// camera vertex (beta=1: the per-pixel radiance convention, matching the backward
// reference). Returns the number of vertices.
//
// Realistic-lens cameras (cam.hasLens(), Plan B): instead of the pinhole genRay, the
// first ray is generated by tracing a sampled film point + rear-pupil point out
// through the real glass interfaces (genLensRay, exactly as mode R does). The camera
// vertex then sits at the ray's scene-entry point with beta = the lens radiometric
// weight wLens (so a pure eye path measures L*wLens, matching mode R's film add), and
// it is flagged `delta`. The delta flag matters for MIS: the multi-element lens map
// has NO closed-form inverse, so a scene point can't be projected back onto a sensor
// pixel -> the light-image splat (t=1) strategy is disabled (see connectBDPT). Marking
// the camera vertex delta makes misWeight's balance heuristic omit the t=1 strategy
// too, so the surviving strategies (s>=0, t>=2: pure path trace, NEE, and scene-side
// light<->eye connections) still form a partition of unity and the estimator stays
// unbiased. Because the camera vertex is delta, its own direction pdf never enters any
// *retained* MIS ratio (only the excluded t=1 term), so the lens ray's exact direction
// density need not be computed — the pinhole cameraPdfDir seeds eye[1].pdfFwd purely as
// an unused placeholder.
inline int generateCameraSubpath(const Scene& scene, const Camera& cam, const Renderer& mats,
                                 int px, int py, double lambda, int maxDepth,
                                 Pcg32& rng, std::vector<Vertex>& path) {
    path.clear();
    Vertex c;
    c.type = VType::Camera; c.ns = cam.w; c.ng = cam.w; c.beta = 1.0;
    if (cam.hasLens()) {
        Ray ray; double wLens = 0.0;
        if (!cam.genLensRay(px, py, rng.uniform(), rng.uniform(),
                            rng.uniform(), rng.uniform(), lambda, ray, wLens)
            || wLens <= 0.0) {
            // Vignetted (clipped by an element / the stop, or TIR): no camera path this
            // sample. Push a lone delta camera vertex (nE=1); every retained strategy
            // needs a scene vertex (t>=2) and t=1 is disabled, so this contributes 0.
            c.p = cam.eye; c.delta = true;
            path.push_back(c);
            return (int)path.size();
        }
        c.p = ray.o;             // scene-entry point (front element plane), for correct
                                 // wo = -ray.d and camera<->eye[1] distance
        c.beta = wLens;          // radiometric lens weight -> per-pixel measurement
        c.delta = true;          // no closed-form lens inverse: not connectible (t=1 off)
        path.push_back(c);
        double pdfDir = cameraPdfDir(cam, dot(ray.d, cam.w));   // MIS-irrelevant placeholder
        randomWalk(scene, cam, mats, ray, wLens, pdfDir, lambda, maxDepth - 1,
                   Mode::Radiance, rng, path);
        return (int)path.size();
    }
    c.p = cam.eye;
    path.push_back(c);
    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
    double cosCam = dot(ray.d, cam.w);
    double pdfDir = cameraPdfDir(cam, cosCam);
    randomWalk(scene, cam, mats, ray, /*beta*/1.0, pdfDir, lambda, maxDepth - 1,
               Mode::Radiance, rng, path);
    return (int)path.size();
}

// Sample a light subpath: choose an emitter (power-weighted), a surface point and a
// cosine-distributed emission direction, at the shared wavelength `lambda`
// (invPdfLambda folds the wavelength importance so Le is radiance, as in backward.h).
// path[0] is the light endpoint (beta = Le, its pdfFwd the positional area density).
// Only area/sphere (Lambertian) lights participate; spot/env/collimated are skipped.
inline int generateLightSubpath(const Scene& scene, const Camera& cam, const Renderer& mats,
                                double lambda, double invPdfLambda, int maxDepth,
                                Pcg32& rng, std::vector<Vertex>& path) {
    path.clear();
    if (scene.emitters.empty() || scene.totalPower <= 0.0) return 0;
    int ei = scene.selectEmitter(rng);
    const Emitter& em = scene.emitters[ei];
    if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env || em.collimated)
        return 0;                                    // unsupported in BDPT scope

    double u1 = rng.uniform(), u2 = rng.uniform();
    Vec3 y, nOut;
    em.samplePoint(u1, u2, y, nOut);
    double Le = em.spdFn(lambda) * invPdfLambda;      // emitted radiance at lambda
    if (Le <= 0.0) return 0;

    double pdfChoice = em.power / scene.totalPower;
    double pdfPos = (em.area > 0.0) ? 1.0 / em.area : 0.0;
    if (pdfPos <= 0.0) return 0;

    Vertex L0;
    L0.type = VType::Light; L0.p = y; L0.ns = nOut; L0.ng = nOut;
    L0.light = &em; L0.matId = em.matId;
    L0.mat = (em.matId >= 0) ? &scene.mats[em.matId] : nullptr;
    L0.beta = Le;                                    // radiance (see header / MIS notes)
    L0.pdfFwd = pdfChoice * pdfPos;                  // positional area density
    path.push_back(L0);

    Vec3 dir = cosineHemisphere(nOut, rng);
    double cosLight = dot(nOut, dir);
    if (cosLight <= 0.0) return 1;
    double pdfDir = cosLight / PI;                    // cosine-weighted emission
    // Walk throughput = Le * cosLight / (pdfChoice * pdfPos * pdfDir)
    //                 = Le * area / pdfChoice  (= emitter power for a single light).
    double betaWalk = Le * cosLight / (pdfChoice * pdfPos * pdfDir);
    Ray ray{y + nOut * 1e-6, dir};
    randomWalk(scene, cam, mats, ray, betaWalk, pdfDir, lambda, maxDepth - 1,
               Mode::Importance, rng, path);
    return (int)path.size();
}

// --- MIS weight (balance heuristic) ----------------------------------------------
// PBRT's MISWeight: temporarily rewrite the connection vertices' reverse densities
// and delta flags for the current strategy (s,t), then sum the density ratios of all
// other strategies that could have produced the same path. `sampled` is the
// resampled endpoint used when s==1 (light NEE) or t==1 (camera splat). `light`/`eye`
// are mutated in place but restored by the ScopedAssigns before returning.
inline double misWeight(const Scene& scene, const Camera& cam,
                        std::vector<Vertex>& light, std::vector<Vertex>& eye,
                        Vertex& sampled, int s, int t) {
    if (s + t == 2) return 1.0;
    auto remap0 = [](double f) { return f != 0.0 ? f : 1.0; };
    Vertex* qs  = s > 0 ? &light[s - 1] : nullptr;
    Vertex* pt  = t > 0 ? &eye[t - 1]   : nullptr;
    Vertex* qsM = s > 1 ? &light[s - 2] : nullptr;
    Vertex* ptM = t > 1 ? &eye[t - 2]   : nullptr;

    // Install the resampled endpoint for s==1 / t==1.
    ScopedAssign<Vertex> a1;
    if (s == 1)      a1 = ScopedAssign<Vertex>(qs, sampled);
    else if (t == 1) a1 = ScopedAssign<Vertex>(pt, sampled);

    // Connection endpoints act as non-delta while evaluating hypothetical strategies.
    ScopedAssign<bool> a2, a3;
    if (pt) a2 = ScopedAssign<bool>(&pt->delta, false);
    if (qs) a3 = ScopedAssign<bool>(&qs->delta, false);

    // Reverse density of the eye connection vertex pt.
    ScopedAssign<double> a4;
    if (pt) {
        double val = (s > 0) ? vertexPdf(scene, cam, qsM, *qs, *pt)
                             : vertexPdfLightOrigin(scene, *pt);
        a4 = ScopedAssign<double>(&pt->pdfRev, val);
    }
    // Reverse density of pt's predecessor.
    ScopedAssign<double> a5;
    if (ptM) {
        double val = (s > 0) ? vertexPdf(scene, cam, qs, *pt, *ptM)
                             : vertexPdfLight(cam, *pt, *ptM);
        a5 = ScopedAssign<double>(&ptM->pdfRev, val);
    }
    // Reverse density of the light connection vertex qs and its predecessor.
    ScopedAssign<double> a6;
    if (qs) a6 = ScopedAssign<double>(&qs->pdfRev, vertexPdf(scene, cam, ptM, *pt, *qs));
    ScopedAssign<double> a7;
    if (qsM) a7 = ScopedAssign<double>(&qsM->pdfRev, vertexPdf(scene, cam, pt, *qs, *qsM));

    double sumRi = 0.0, ri = 1.0;
    for (int i = t - 1; i > 0; --i) {                // hypothetical camera strategies
        ri *= remap0(eye[i].pdfRev) / remap0(eye[i].pdfFwd);
        if (!eye[i].delta && !eye[i - 1].delta) sumRi += ri;
    }
    ri = 1.0;
    for (int i = s - 1; i >= 0; --i) {               // hypothetical light strategies
        ri *= remap0(light[i].pdfRev) / remap0(light[i].pdfFwd);
        bool deltaPrev = (i > 0) ? light[i - 1].delta : false; // area lights aren't delta
        if (!light[i].delta && !deltaPrev) sumRi += ri;
    }
    return 1.0 / (1.0 + sumRi);
}

// Offset a shadow-ray origin off a surface along the geometric normal, flipped to the
// SAME side as the connection direction `dir`. Vertex::ng is the RAW geometric normal
// (winding-defined, not oriented to any ray — see geometry.h), so a fixed +ng offset
// pushes the origin *behind* the surface whenever ng faces away from `dir`, which
// makes the connection self-occlude. Orienting the offset by dir fixes that (this is
// PBRT's OffsetRayOrigin convention).
inline Vec3 offsetOrigin(const Vertex& v, const Vec3& dir) {
    double sgn = dot(v.ng, dir) >= 0.0 ? 1.0 : -1.0;
    return v.p + v.ng * (sgn * 1e-6);
}

// --- Connect one strategy (s,t) --------------------------------------------------
// Returns the MIS-weighted radiance contribution of connecting the s-vertex light
// subpath with the t-vertex eye subpath. For t==1 the contribution is a light-image
// splat to raster (outPx,outPy) with isSplat=true; otherwise it belongs to the
// current pixel. `light`/`eye` are non-const because misWeight mutates them.
inline double connectBDPT(const Scene& scene, const Camera& cam, const Renderer& mats,
                          std::vector<Vertex>& light, std::vector<Vertex>& eye,
                          int s, int t, double lambda, double invPdfLambda,
                          Pcg32& rng, int& outPx, int& outPy, bool& isSplat) {
    (void)mats;
    isSplat = false;
    // Can't connect ONTO a vertex that already sits on a light (PBRT guard).
    if (t > 1 && s != 0 && eye[t - 1].isLightVertex()) return 0.0;

    double L = 0.0;
    Vertex sampled;

    if (s == 0) {
        // Pure eye path: contributes iff its last vertex is emissive.
        if (t < 2) return 0.0;
        const Vertex& pt = eye[t - 1];
        if (!pt.isLightVertex()) return 0.0;
        Vec3 wo = normalize(eye[t - 2].p - pt.p);
        double Le = pt.Le(wo, lambda, invPdfLambda);
        if (Le <= 0.0) return 0.0;
        L = pt.beta * Le;
    } else if (t == 1) {
        // Splat a light-subpath vertex onto the camera (light image). Requires
        // projecting a world point onto the sensor. A realistic multi-element lens has
        // no closed-form inverse (Plan B), so this strategy is disabled for a lensed
        // camera; the camera vertex is flagged delta (generateCameraSubpath) so
        // misWeight omits it consistently and the retained strategies still partition
        // unity. Scene-side connections (s>=1, t>=2) keep the forward-transport caustic
        // efficiency through the physical lens.
        if (cam.hasLens()) return 0.0;
        // s>=2 here (s==1&&t==1 is skipped by the caller), so qs is an interior vertex.
        const Vertex& qs = light[s - 1];
        if (!qs.isConnectible()) return 0.0;
        int px, py; double cosCam, dist2;
        if (!cam.project(qs.p, px, py, cosCam, dist2)) return 0.0;
        double dist = std::sqrt(dist2);
        Vec3 wcam = (cam.eye - qs.p) / dist;
        double cosSurf = dot(qs.ns, wcam);
        if (cosSurf <= 0.0) return 0.0;
        Vec3 wo = normalize(light[s - 2].p - qs.p);
        double f = bsdfF(*qs.mat, qs.ns, wo, wcam, lambda, scene, &qs.hit);
        if (f <= 0.0) return 0.0;
        if (scene.occluded(offsetOrigin(qs, wcam), wcam, dist - 2e-6)) return 0.0;
        double G = cosSurf * cosCam / dist2;
        L = qs.beta * f * G * cameraWe(cam, cosCam);
        if (L <= 0.0) return 0.0;
        sampled.type = VType::Camera; sampled.p = cam.eye; sampled.ns = cam.w; sampled.ng = cam.w;
        sampled.beta = 1.0; sampled.delta = false;
        outPx = px; outPy = py; isSplat = true;
    } else if (s == 1) {
        // NEE: connect the eye vertex to a freshly sampled point on a light.
        const Vertex& pt = eye[t - 1];
        if (!pt.isConnectible()) return 0.0;
        int ei = scene.selectEmitter(rng);
        const Emitter& em = scene.emitters[ei];
        if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Env || em.collimated)
            return 0.0;
        double u1 = rng.uniform(), u2 = rng.uniform();
        Vec3 y, nOut; em.samplePoint(u1, u2, y, nOut);
        Vec3 toL = y - pt.p; double dist2 = dot(toL, toL);
        if (dist2 <= 0.0) return 0.0;
        double dist = std::sqrt(dist2); Vec3 wi = toL / dist;
        double cosLight = dot(nOut, wi * -1.0);
        double cosSurf  = dot(pt.ns, wi);
        if (cosLight <= 0.0 || cosSurf <= 0.0) return 0.0;
        double Le = em.spdFn(lambda) * invPdfLambda;
        if (Le <= 0.0) return 0.0;
        if (scene.occluded(offsetOrigin(pt, wi), wi, dist - 2e-6)) return 0.0;
        Vec3 wo = normalize(eye[t - 2].p - pt.p);
        double f = bsdfF(*pt.mat, pt.ns, wo, wi, lambda, scene, &pt.hit);
        if (f <= 0.0) return 0.0;
        double pdfChoice = em.power / scene.totalPower;
        double pdfA = pdfChoice / em.area;             // area-measure light pdf
        if (pdfA <= 0.0) return 0.0;
        double G = cosSurf * cosLight / dist2;
        L = pt.beta * f * Le * G / pdfA;
        if (L <= 0.0) return 0.0;
        sampled.type = VType::Light; sampled.p = y; sampled.ns = nOut; sampled.ng = nOut;
        sampled.light = &em; sampled.matId = em.matId;
        sampled.mat = (em.matId >= 0) ? &scene.mats[em.matId] : nullptr;
        sampled.beta = Le / pdfA; sampled.delta = false; sampled.pdfFwd = pdfA;
    } else {
        // Interior connection light[s-1] <-> eye[t-1].
        const Vertex& qs = light[s - 1];
        const Vertex& pt = eye[t - 1];
        if (!qs.isConnectible() || !pt.isConnectible()) return 0.0;
        Vec3 d = qs.p - pt.p; double dist2 = dot(d, d);
        if (dist2 <= 0.0) return 0.0;
        double dist = std::sqrt(dist2); Vec3 w = d / dist;   // pt -> qs
        double cosE = dot(pt.ns, w), cosL = dot(qs.ns, w * -1.0);
        if (cosE <= 0.0 || cosL <= 0.0) return 0.0;
        Vec3 woE = normalize(eye[t - 2].p - pt.p);
        Vec3 woL = normalize(light[s - 2].p - qs.p);
        double fE = bsdfF(*pt.mat, pt.ns, woE, w, lambda, scene, &pt.hit);
        double fL = bsdfF(*qs.mat, qs.ns, woL, w * -1.0, lambda, scene, &qs.hit);
        if (fE <= 0.0 || fL <= 0.0) return 0.0;
        if (scene.occluded(offsetOrigin(pt, w), w, dist - 2e-6)) return 0.0;
        double G = cosE * cosL / dist2;
        L = pt.beta * fE * fL * qs.beta * G;
    }
    if (L <= 0.0) return 0.0;
    return L * misWeight(scene, cam, light, eye, sampled, s, t);
}

// --- Renderer --------------------------------------------------------------------
// Renders pixel rows [y0,y1). t>=2 (camera-image) contributions land on the current
// pixel in `camFilm`; t==1 (light-image) splats land on `splatFilm` at the projected
// raster position. The caller normalises camFilm by spp and splatFilm by the total
// light-subpath count (W*H*spp), matching mode B's absolute-radiance convention.
struct BdptRenderer {
    int maxDepth = 8;          // maximum path length in edges (connection cost ~ depth^2)
    bool diffraction = true;   // mirrors Renderer::diffraction for MatType::Grating

    void renderRows(const Scene& scene, const Camera& cam, Film& camFilm, Film& splatFilm,
                    int y0, int y1, long long spp, Pcg32& rng) const {
        Renderer mats; mats.diffraction = diffraction;
        std::vector<Vertex> eye, light;
        for (int py = y0; py < y1; ++py)
            for (int px = 0; px < camFilm.resX; ++px)
                for (long long si = 0; si < spp; ++si) {
                    double pdfLam = 0.0;
                    double lambda = scene.emitSampler.sample(rng, pdfLam);
                    if (pdfLam <= 0.0) continue;
                    double invPdfLambda = scene.invPdfLambda(lambda);
                    int nE = generateCameraSubpath(scene, cam, mats, px, py, lambda,
                                                   maxDepth + 1, rng, eye);
                    int nL = generateLightSubpath(scene, cam, mats, lambda, invPdfLambda,
                                                  maxDepth + 1, rng, light);
                    Vec3 cie(cieX(lambda), cieY(lambda), cieZ(lambda));
                    for (int t = 1; t <= nE; ++t)
                        for (int s = 0; s <= nL; ++s) {
                            int depth = t + s - 2;
                            if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth) continue;
                            int spx = 0, spy = 0; bool isSplat = false;
                            double c = connectBDPT(scene, cam, mats, light, eye, s, t,
                                                   lambda, invPdfLambda, rng, spx, spy, isSplat);
                            if (c <= 0.0) continue;
                            if (isSplat) splatFilm.add(spx, spy, cie * c);
                            else         camFilm.add(px, py, cie * c);
                        }
                }
    }
};

} // namespace bdpt
