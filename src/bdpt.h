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
// connection pdf is zero). Emission comes from area/sphere quad lights. HOMOGENEOUS
// participating media ARE handled: a subpath can scatter at a volume (Medium) vertex
// via the HG phase function, connections carry a transmittance factor, and the
// balance-heuristic MIS uses cosine-free phase densities (the σt·exp free-flight and
// transmittance terms cancel pairwise for homogeneous media, so this is exactly
// unbiased). Heterogeneous / density-field / implicit-bounded media, fluorescence,
// spot and environment lights are NOT handled here (use mode B/P/R instead); see the
// guard in main.cpp.
#pragma once
#include <vector>
#include <algorithm>
#include "scene.h"
#include "camera.h"
#include "hero.h"     // kHeroC / kHeroMax — hero-wavelength bundle sizes
#include "render.h"   // sampleGlossy, Renderer material primitives, clamp01, PI

namespace bdpt {

// Which direction a subpath transports. Radiance = from the camera (eye subpath);
// Importance = from a light (light subpath). Only affects the (non-reciprocal)
// glossy lobe's cosine-denominator choice so both subpaths stay consistent with how
// the forward/backward tracers sample that lobe.
enum class Mode { Radiance, Importance };

// The wavelength bundle a sample carries: the hero λ (index 0) plus C-1 stratified
// secondaries, each with its own 1/pdf(λ) importance weight. C == 1 is the classic
// single-wavelength sample, and every hero code path below then collapses to exactly
// the scalar arithmetic it replaced (bit-identical). Geometry, sampling decisions and
// MIS densities always use `lam[0]`; the secondaries only ever affect throughput.
struct HeroBundle {
    double lam[hero::kHeroMax]    = {0};
    double invPdf[hero::kHeroMax] = {0};
    int    C = 1;
    double hero() const { return lam[0]; }
    double heroInvPdf() const { return invPdf[0]; }
};

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
           m.type == MatType::Fluorescent ||   // fluoro's elastic base is diffuse-like
           m.type == MatType::DiffuseTransmit;  // two-sided Lambertian (finite BSDF both sides)
}

// A two-sided (transmissive) connectible material scatters into BOTH hemispheres, so a
// connection edge on the side OPPOSITE the shading normal is legal (transmit lobe).
// Reflect-only materials require the edge on the +ns side; the surface-cosine sign guards
// in connectBDPT must not reject the back hemisphere for these materials — bsdfF (which
// returns 0 for unsupported directions) is the real validity gate, and the geometry term
// uses |cos| accordingly.
inline bool isTwoSidedMat(const Material& m) {
    return m.type == MatType::DiffuseTransmit;
}

// Clamped reflect/transmit albedos of a DiffuseTransmit vertex (energy guard shared by
// bsdfF / bsdfPdf / the scatter switch so MIS densities stay consistent).
inline void diffuseTransmitAlbedos(const Material& m, double lambda, const Scene& scene,
                                   const Hit* hitForTex, double& rhoR, double& rhoT) {
    rhoR = hitForTex ? clamp01(diffuseReflectance(scene, m, *hitForTex, lambda))
                     : clamp01(m.reflect(lambda));
    rhoT = hitForTex ? clamp01(transmitSlot(scene, m, *hitForTex, lambda))
                     : clamp01(m.transmit(lambda));
    double sum = rhoR + rhoT;
    if (sum > 1.0) { rhoR /= sum; rhoT /= sum; }
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
            double r = hitForTex ? clamp01(reflectSlot(scene, m, *hitForTex, lambda))
                                 : clamp01(m.reflect(lambda));
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
        case MatType::DiffuseTransmit: {
            // Two-sided Lambertian: same-hemisphere pair (wo,wi) -> reflect albedo,
            // opposite hemispheres -> transmit albedo. Symmetric in wo<->wi.
            double rhoR, rhoT; diffuseTransmitAlbedos(m, lambda, scene, hitForTex, rhoR, rhoT);
            bool sameSide = (cosWi * cosWo) > 0.0;
            double rho = sameSide ? rhoR : rhoT;
            return rho / PI;
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
                      double lambda, const Scene& scene, const Hit* hitForTex) {
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
        case MatType::DiffuseTransmit: {
            // Directional pdf of the lobe-selected cosine sampling: the reflect lobe is
            // chosen with prob rhoR/(rhoR+rhoT) and cosine-samples the same hemisphere as
            // wo; the transmit lobe (prob rhoT/(rhoR+rhoT)) cosine-samples the opposite
            // hemisphere. For a given wi only one lobe applies (by its sign vs wo).
            double rhoR, rhoT; diffuseTransmitAlbedos(m, lambda, scene, hitForTex, rhoR, rhoT);
            double tot = rhoR + rhoT;
            if (tot <= 0.0) return 0.0;
            bool sameSide = (cosWi * cosWo) > 0.0;
            double pSel = sameSide ? rhoR / tot : rhoT / tot;
            return pSel * std::fabs(cosWi) / PI;
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
// A Medium vertex is a volume in-scatter point (participating media): it has no
// surface (no ns/ng, no material), scatters via the HG phase function, and is always
// connectible. onSurface() is false for it, so ConvertDensity omits the cosine (the
// area density at a medium interaction is cosine-free) — see the MIS notes below.
enum class VType { Camera, Light, Surface, Medium };

// --- Delta / infinite light classification (PBRT's LightFlags, specialised) --------
// A DELTA light's emission involves a Dirac delta in position and/or direction, so some
// BDPT strategies are impossible and must be dropped from the MIS balance heuristic (a
// strategy that cannot be sampled must not appear in the denominator, or the retained
// ones are under-weighted and the image loses energy):
//   Spot       - delta POSITION (a mathematical point). No eye ray can ever hit it, so
//                the s=0 strategy (eye path lands on emissive geometry) is impossible.
//                NEE (s=1) works: the connection point is deterministic.
//   Sun        - delta DIRECTION and infinitely distant. It has no geometry in the scene
//                either, so again s=0 is impossible; NEE samples a direction inside the
//                0.53-degree solar cone with pdf 1/Omega.
//   collimated - delta DIRECTION from a FINITE surface. Both s=0 and NEE are impossible
//                (a shading point sees the beam only if it happens to lie exactly on it),
//                so these stay out of BDPT scope entirely (mode-D guard refuses them).
inline bool isDeltaEmitter(const Emitter& em) {
    return em.shape == EmitterShape::Spot || em.shape == EmitterShape::Sun || em.collimated;
}
// INFINITE lights have no finite emission point: their light-subpath origin is a fictitious
// point on a disc outside the scene bounds, so the density of the first scene vertex is the
// PLANAR density 1/(pi R^2) over that disc rather than a solid-angle density over 1/dist^2
// (PBRT's Vertex::PdfLight infinite branch). Sun is one; Env would be the other, but Env is
// still outside BDPT scope (it also needs escaped-ray radiance, which the walk doesn't do).
inline bool isInfiniteEmitter(const Emitter& em) {
    return em.shape == EmitterShape::Sun || em.shape == EmitterShape::Env;
}
// Can a shading point next-event-estimate this emitter (the s=1 strategy)? Everything
// except a collimated beam, whose emitted direction is a Dirac delta about beamDir.
inline bool isNeeEmitter(const Emitter& em) { return !em.collimated; }

struct Vertex {
    VType type = VType::Surface;
    Vec3 p{0, 0, 0};      // world position
    Vec3 ns{0, 0, 0};     // shading normal (used for all cosines/BSDF)
    Vec3 ng{0, 0, 0};     // geometric normal (orientation / emission side)
    double beta = 0.0;    // throughput carried to this vertex at the HERO wavelength
    // Hero-wavelength bundle (`-heroc N`). The path GEOMETRY and every MIS density are
    // decided by the hero wavelength alone, so `pdfFwd`/`pdfRev`/`delta` are unchanged;
    // only the throughput is per-wavelength. `betaSec[i]` is the throughput of secondary
    // wavelength i+1 and is live only while `nUp > i + 1`. `nUp` is the number of
    // wavelengths still riding this subpath when the vertex was created: it starts at C
    // and drops to 1 at the first dispersive / wavelength-switching (delta) interface
    // ("de-hero"), after which `beta` alone carries the path. It is therefore monotone
    // non-increasing along a subpath and always either C or 1.
    //
    // NOTE the ×C de-hero boost is deliberately NOT folded into `beta` here (unlike the
    // unidirectional tracers): a BDPT contribution joins TWO subpaths that may have
    // de-hero'd independently, and boosting each side would square the factor. Instead
    // the normalisation is applied once at splat time as 1/min(nUp_light, nUp_eye) —
    // see BdptRenderer::renderRows. Both spellings are the same estimator.
    double betaSec[hero::kHeroMax - 1] = {0};
    int nUp = 1;          // live wavelengths (1 = hero off, or already de-hero'd)
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
    // This vertex's `emit pattern:` factor (Material::emitPat evaluated here), cached
    // because Le() below has no Scene to evaluate it from and is called from several
    // MIS strategies. 1.0 for a non-emissive vertex or an unpatterned light, so every
    // existing scene multiplies by exactly one. A Light vertex gets it from
    // emitterSamplePoint (the sampled point); a Surface vertex from slotPatMul at the
    // hit — the two agree pointwise, which is what keeps s=0 / s=1 MIS unbiased.
    double emitPatW = 1.0;

    // Medium data (type == Medium): HG anisotropy g and index into scene.media of the
    // medium that scattered here (for the phase function value and pdf).
    double mediumG = 0.0;
    int mediumId = -1;

    bool onSurface() const { return type == VType::Surface || type == VType::Light; }
    bool isConnectible() const {
        if (type == VType::Camera) return true;      // pinhole connects (delta pos handled)
        if (type == VType::Light)  return light && !light->collimated;
        if (type == VType::Medium) return true;      // volume in-scatter always connects
        return mat && !delta && isConnectibleMat(*mat);
    }
    // Emitted radiance (single wavelength) leaving this vertex toward direction w,
    // if it is (or sits on) a light. 0 otherwise or if w is on the unlit side.
    double Le(const Vec3& w, double lambda, double invPdfLambda) const {
        if (!mat || !mat->isLight) return 0.0;
        if (dot(ng, w) <= 0.0) return 0.0;           // one-sided emitter
        return mat->emit(lambda) * invPdfLambda * emitPatW;
    }
    bool isLightVertex() const {
        return type == VType::Light || (type == VType::Surface && mat && mat->isLight);
    }
    // PBRT's Vertex::IsDeltaLight: this vertex IS a light whose emission carries a Dirac
    // delta, so the "eye path hits the light" (s=0) strategy cannot produce it. Used by
    // misWeight to drop that strategy from the balance heuristic.
    bool isDeltaLight() const {
        return type == VType::Light && light && isDeltaEmitter(*light);
    }
    // PBRT's Vertex::IsInfiniteLight: an infinitely-distant light whose subpath origin is
    // a fictitious point outside the scene (planar emission density; see isInfiniteEmitter).
    bool isInfiniteLight() const {
        return type == VType::Light && light && isInfiniteEmitter(*light);
    }
};

// Henyey-Greenstein phase function value AND sampling pdf at a medium vertex `v`, for
// an incoming subpath direction `wo` (toward the previous vertex) and an outgoing /
// connection direction `wi` (both unit, both pointing AWAY from v). The propagation
// direction INTO v is -wo; the scattered direction is wi; the phase cosine is therefore
//   cosTheta = dot(-wo, wi) = -dot(wo, wi).
// The phase is normalized over the sphere, so it is its own pdf (the scatter dir is
// importance-sampled from it): phaseF == phasePdf. Two names are kept for readability
// (BSDF value vs BSDF pdf on the surface side map to these on the medium side). The
// medium dispatches HG vs the wavelength-dependent rainbow droplet phase, so both take
// lambda + scene; the phase depends only on the scattering angle, so it is symmetric in
// (wo,wi) and the forward/reverse pdfs are equal.
inline double phaseF(const Vertex& v, const Vec3& wo, const Vec3& wi,
                     double lambda, const Scene& scene) {
    return scene.media[v.mediumId].phaseValue(-dot(wo, wi), lambda);
}
inline double phasePdf(const Vertex& v, const Vec3& wo, const Vec3& wi,
                       double lambda, const Scene& scene) {
    return scene.media[v.mediumId].phaseValue(-dot(wo, wi), lambda);
}

// In-scatter CONNECTION response at a medium vertex = single-scattering albedo (σs/σt)
// times the phase function. The albedo factor is the fraction of the σt-sampled
// collision that scatters (rather than absorbs); the subpath continuation applies the
// same albedo implicitly via Russian-roulette survival, so a medium vertex carries the
// albedo exactly once — through RR when the path passes through it, or through this
// factor when the path connects at it. This mirrors the forward connectVolume and the
// backward volume-NEE (render.h / backward.h), which both use albedo*phase. Note this
// is a THROUGHPUT factor, not a density: MIS pdfs use the phase pdf alone (no albedo).
inline double mediumScatterF(const Vertex& v, const Vec3& wo, const Vec3& wi,
                             double lambda, const Scene& scene) {
    return scene.media[v.mediumId].albedo(lambda) * phaseF(v, wo, wi, lambda, scene);
}

// Convert a solid-angle pdf `pdfW` of leaving `from` toward `to` into an area-
// measure density at `to` (PBRT's Vertex::ConvertDensity). Surfaces pick up the
// projected-cosine Jacobian; the 1/dist^2 is always applied. A Medium `to` is NOT
// onSurface(), so it stays cosine-free (correct volume area density).
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
                        const Vertex* prev, const Vertex& cur, const Vertex& next,
                        double lambda);
inline double vertexPdfLight(const Scene& scene, const Vertex& cur, const Vertex& next);

// Emission directional density at a light vertex `cur` toward `next`, area measure
// (PBRT's Vertex::PdfLight). Three emission models:
//   area/sphere/tube/mesh - cosine-weighted about the surface normal, cos/PI, converted
//                           to area measure by the usual 1/dist^2 * cos(next).
//   Spot                  - uniform inside the OUTER cone (the falloff is throughput, not
//                           density), pdfW = 1/(2 PI (1 - cosOuter)), same conversion.
//   Sun (infinite)        - no finite emission point: the subpath origin is a point on a
//                           disc of radius sceneRadius outside the scene, so the density
//                           of `next` is the PLANAR density 1/(pi R^2) with NO 1/dist^2
//                           (the disc-point -> hit-point map is a shear along the beam,
//                           whose Jacobian is exactly the cos(next) below).
inline double vertexPdfLight(const Scene& scene, const Vertex& cur, const Vertex& next) {
    Vec3 w = next.p - cur.p;
    double d2 = dot(w, w);
    if (d2 == 0.0) return 0.0;
    double invD2 = 1.0 / d2;
    w = w * std::sqrt(invD2);
    double pdf;
    if (cur.light && isInfiniteEmitter(*cur.light)) {
        double R = scene.sceneRadius;
        if (R <= 0.0) return 0.0;
        pdf = 1.0 / (PI * R * R);                    // planar density over the sun disc
    } else if (cur.light && cur.light->shape == EmitterShape::Spot) {
        double solid = 2.0 * PI * (1.0 - cur.light->spotCosOuter);
        if (solid <= 0.0) return 0.0;
        pdf = (1.0 / solid) * invD2;                 // uniform in the outer cone
    } else {
        double cosLight = dot(cur.ng, w);            // one-sided Lambertian emitter
        if (cosLight <= 0.0) return 0.0;
        pdf = (cosLight / PI) * invD2;               // cosine-weighted emission
    }
    if (next.onSurface()) pdf *= std::abs(dot(next.ns, w));
    return pdf;
}

inline double vertexPdf(const Scene& scene, const Camera& cam,
                        const Vertex* prev, const Vertex& cur, const Vertex& next,
                        double lambda) {
    if (cur.type == VType::Light) return vertexPdfLight(scene, cur, next);
    Vec3 wn = next.p - cur.p;
    if (dot(wn, wn) == 0.0) return 0.0;
    wn = normalize(wn);
    double pdfW = 0.0;
    if (cur.type == VType::Camera) {
        pdfW = cameraPdfDir(cam, camCos(cam, next.p));
    } else if (cur.type == VType::Medium) {          // volume in-scatter: HG phase pdf
        if (!prev) return 0.0;
        Vec3 wp = prev->p - cur.p;
        if (dot(wp, wp) == 0.0) return 0.0;
        wp = normalize(wp);
        pdfW = phasePdf(cur, wp, wn, lambda, scene);
    } else {                                         // Surface
        if (!prev || !cur.mat) return 0.0;
        Vec3 wp = prev->p - cur.p;
        if (dot(wp, wp) == 0.0) return 0.0;
        wp = normalize(wp);
        pdfW = bsdfPdf(*cur.mat, cur.ns, wp, wn, lambda, scene, &cur.hit);
    }
    return convertDensity(pdfW, cur, next);
}

// Positional density (area measure) of sampling this light vertex's ORIGIN via
// light sampling = P(choose this emitter) * (1/area). PBRT's PdfLightOrigin.
// A DELTA light (spot: a point; sun: infinitely distant) has no area density at all —
// PBRT returns 0 for exactly these (SpotLight/DistantLight::Pdf_Le set pdfPos = 0, and
// InfiniteLightDensity is 0 for a delta-direction light). The 0 is consistent on BOTH
// sides of every MIS ratio (the light subpath stores the same 0 in path[0].pdfFwd), and
// misWeight's remap0 turns both into 1, so the ratios stay finite and unbiased.
inline double vertexPdfLightOrigin(const Scene& scene, const Vertex& cur) {
    if (!cur.light || scene.totalPower <= 0.0) return 0.0;
    if (isDeltaEmitter(*cur.light)) return 0.0;      // delta position / direction
    if (cur.light->area <= 0.0) return 0.0;
    double pdfChoice = cur.light->power / scene.totalPower;
    return pdfChoice / cur.light->area;
}

// --- Random walk -----------------------------------------------------------------
// Where, and with what weight, a subpath's LAST ray left the scene. BDPT has no
// environment, but a `light sun` is an infinitely distant delta-DIRECTION emitter that an
// eye ray can still look straight into. Because a delta light is excluded from the s=0
// strategy (see misWeight), nothing else in BDPT can deliver the sun's own disc — nor any
// mirror/water glint of it, which NEE cannot produce either (a specular vertex is not
// connectible). Capturing the escaping ray lets the renderer add that one strategy back
// with MIS weight exactly 1, since no other strategy can generate the same path (every
// vertex on such a path is delta, so neither NEE nor a light-subpath connection reaches
// it). See BdptRenderer::renderRows.
struct Escape {
    bool escaped = false;
    Vec3 dir{0, 0, 0};                        // unit direction the ray left along
    double beta = 0.0;                        // hero throughput carried out of the scene
    double betaSec[hero::kHeroMax - 1] = {0};
    int nUp = 1;
};

// Continue a subpath whose endpoint is already path[0] (Camera or Light). `ray` is
// the first ray leaving that endpoint; `beta` the throughput carried along it;
// `pdfDir` the solid-angle density of that first direction; `mode` the transport
// direction. Appends surface AND medium (volume in-scatter) vertices until a miss,
// absorption, or maxDepth. No environment handling (BDPT scope). Uses `mats` for
// specular primitives and participating-media collision sampling.
//
// Hero-wavelength bundle: `hb` supplies the C wavelengths, `betaSec`/`nUp` the incoming
// secondary throughputs (nUp == 1 for a plain single-λ walk, which takes exactly the
// original code path). Only the THROUGHPUT is per-λ — every direction, pdf, Russian
// roulette draw and MIS density comes from the hero λ, so the secondaries reweight by
// the ratio of their own scattering albedo to the hero's. At a delta (dispersive /
// wavelength-switching) interface the secondaries can no longer follow the hero's
// refracted direction, so the bundle de-heros: nUp drops to 1 for this and every
// later vertex. See Vertex::nUp for why no ×C boost is applied here.
inline void randomWalk(const Scene& scene, const Camera& cam, const Renderer& mats,
                       Ray ray, double beta, double pdfDir, const HeroBundle& hb,
                       int maxDepth, Mode mode, Pcg32& rng, std::vector<Vertex>& path,
                       const double* betaSecIn, int nUpIn, Escape* esc = nullptr) {
    (void)cam;   // cam reserved for future NEE-to-camera use; mode now drives adjoint corr
    const double lambda = hb.lam[0];   // the hero drives geometry, sampling and every pdf
    if (maxDepth == 0) return;
    double pdfFwd = pdfDir;   // solid-angle density of the current ray direction
    // Live secondary throughputs. betaSec[i] tracks wavelength hb.lam[i+1].
    double betaSec[hero::kHeroMax - 1] = {0};
    int nUp = nUpIn < 1 ? 1 : nUpIn;
    for (int i = 0; i + 1 < nUp; ++i) betaSec[i] = betaSecIn[i];
    // Nested-dielectric medium stack (Schmidt & Budge 2002): the solids the subpath is
    // currently inside. Current medium (Beer-Lambert absorption + exterior IOR at the
    // next interface) = the highest-priority entry. Behaves like the old single-pointer
    // `interior` for a lone dielectric.
    MediumStack stk;
    auto curAbsorb = [&](double lam) -> double {
        int mi = stk.topMat();
        return (mi >= 0) ? scene.mats[mi].absorb(lam) : 0.0;
    };
    for (int bounces = 0;;) {
        Hit h = scene.closestHit(ray);
        if (h.valid && h.sensorId >= 0) return;      // model-A sensor: not used in BDPT
        double dSurf = h.valid ? h.t : 1e30;

        // Participating media: sample the earliest real collision along the ray up to
        // the surface (or 1e30 in open space). A homogeneous medium draws one exact
        // free-flight; its transmittance is implicit in that exponential so beta is
        // unchanged (analog MC), exactly matching modes B/C. With no media this call
        // draws no RNG and returns false, so vacuum walks are bit-identical.
        double dEvent = dSurf;
        bool mediumEvent = false;
        int scatterMed = -1;
        double tMed = 0.0;
        if (!scene.media.empty()) {
            int which;
            if (mats.sampleMediaCollision(scene, ray.o, ray.d, dSurf, lambda,
                                          rng, tMed, which)) {
                dEvent = tMed; mediumEvent = true; scatterMed = which;
            }
        }

        // Beer-Lambert attenuation over the in-glass segment just traversed, up to the
        // event (surface hit OR medium collision, whichever is nearer). NOTE: this
        // attenuates only the *subpath walk*; connection edges (connectBDPT) that cross
        // glass are NOT absorption-weighted (see known-issues.md).
        {
            double a = curAbsorb(lambda);
            if (a > 0.0) beta *= std::exp(-a * dEvent);
            // Per-λ absorption for the bundle. A non-empty stack means we are inside a
            // dielectric, and entering one de-heros — so nUp is always 1 whenever `a`
            // can be non-zero and this loop never actually runs. Kept for generality.
            for (int i = 0; i + 1 < nUp; ++i) {
                double ai = curAbsorb(hb.lam[i + 1]);
                if (ai > 0.0) betaSec[i] *= std::exp(-ai * dEvent);
            }
        }

        // A medium collision precedes the surface: append a volume in-scatter vertex,
        // then scatter (prob = single-scattering albedo) or absorb. Throughput is
        // unchanged on scatter; the HG sampling pdf equals the phase value, so the
        // f*cos/pdf factor collapses to 1 (analog MC), matching the forward tracer.
        // The stored area densities are cosine-free (Medium is not onSurface()) and
        // carry ONLY the phase direction density — the free-flight distance pdf and
        // transmittance are omitted here AND in vertexPdf, so they cancel pairwise in
        // every balance-heuristic ratio (exact for homogeneous media).
        if (mediumEvent) {
            const Medium& sm = scene.media[scatterMed];
            Vec3 mpos = ray.o + ray.d * tMed;
            size_t prevIdx = path.size() - 1;
            Vertex v;
            v.type = VType::Medium;
            v.p = mpos; v.beta = beta;
            // Hero is gated off for scenes with media, so nUp is 1 here in practice.
            v.nUp = nUp;
            for (int i = 0; i + 1 < nUp; ++i) v.betaSec[i] = betaSec[i];
            v.mediumG = sm.g; v.mediumId = scatterMed;
            v.pdfFwd = convertDensity(pdfFwd, path[prevIdx], v);
            path.push_back(v);
            if (++bounces >= maxDepth) return;
            if (rng.uniform() >= sm.albedo(lambda)) return;   // absorbed (vertex retained)
            Vertex& cur = path.back();
            Vec3 wo = normalize(path[prevIdx].p - cur.p);     // toward the previous vertex
            double pdfW;
            Vec3 wi = sm.phaseSample(ray.d, lambda, rng, pdfW); // scattered dir (HG or rainbow)
            double pdfRevW = pdfW;                             // phase symmetric in (wo,wi)
            path[prevIdx].pdfRev = convertDensity(pdfRevW, cur, path[prevIdx]);
            ray = Ray{mpos, wi};
            pdfFwd = pdfW;
            continue;
        }

        if (!h.valid) {                              // escaped (no env in BDPT scope)
            if (esc) {
                esc->escaped = true; esc->dir = ray.d; esc->beta = beta; esc->nUp = nUp;
                for (int i = 0; i + 1 < nUp; ++i) esc->betaSec[i] = betaSec[i];
            }
            return;
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
        v.nUp = nUp;
        for (int i = 0; i + 1 < nUp; ++i) v.betaSec[i] = betaSec[i];
        if (mp->isLight) {
            v.light = scene.emitterForMat(h.matId);
            // Evaluate the emission pattern once, here, where the Hit is in hand — Le()
            // is called later from several MIS strategies with no Scene available.
            if (mp->emitPat >= 0) v.emitPatW = slotPatMul(scene, mp->emitPat, h);
        }
        // Index, not a reference: push_back below may reallocate the vector, and a
        // Vertex& taken before it would dangle (stale reads corrupted mode-D MIS pdfs
        // and the pdfRev write below scribbled on freed heap memory — ASan-verified).
        size_t prevSurfIdx = path.size() - 1;
        v.pdfFwd = convertDensity(pdfFwd, path[prevSurfIdx], v);
        path.push_back(v);
        Vertex& cur = path.back();
        if (++bounces >= maxDepth) return;

        // Sample a continuation direction wi, its forward solid-angle pdf pdfW, the
        // reverse pdf pdfRevW (wi<->wo swapped), the throughput factor, and whether
        // this vertex is a delta (specular) scatter.
        Vec3 wo = normalize(path[prevSurfIdx].p - cur.p);   // toward the previous vertex
        Vec3 wi; double pdfW = 0.0, pdfRevW = 0.0, betaFactor = 0.0;
        // Hero bundle: per-secondary throughput factor, i.e. secF[i] = f_{i+1}·cos/pdf
        // for the lobe the hero actually sampled (pdf is always the hero's). This is the
        // ABSOLUTE factor, not a ratio to the hero's: a ratio would be undefined exactly
        // where it matters most — a strongly chromatic lobe whose hero value is 0 while a
        // secondary's is not (a Wratten gel is 0 over most of the spectrum). Cases that
        // are wavelength-INDEPENDENT (all the geometry, the specular interfaces, the
        // adjoint correction) leave `secChromatic` false and reuse `betaFactor` for every
        // λ. Ignored entirely when nUp == 1.
        double secF[hero::kHeroMax - 1];
        bool secChromatic = false;
        // A few DELTA lobes are nevertheless wavelength-INDEPENDENT in direction
        // (Mirror reflects, Filter passes straight through — neither consults λ to
        // pick the continuation), so the secondaries CAN keep riding the hero's ray
        // past them; only their per-λ reflectance/transmittance differs. Those set
        // `keepBundle` to opt out of the `if (delta) nUp = 1` collapse below.
        bool delta = false, terminate = false, keepBundle = false;
        switch (mp->type) {
            case MatType::Diffuse:
            case MatType::Fluorescent: {              // elastic base only (see header)
                wi = cosineHemisphere(cur.ns, rng);
                if (dot(wi, cur.ns) <= 0) { terminate = true; break; }
                double rho = clamp01(diffuseReflectance(scene, *mp, h, lambda));
                pdfW = bsdfPdf(*mp, cur.ns, wo, wi, lambda, scene, &h);
                pdfRevW = bsdfPdf(*mp, cur.ns, wi, wo, lambda, scene, &h);
                betaFactor = rho;                     // f*cos/pdf = rho
                secChromatic = true;                  // rho <= 0 is caught by the max test
                for (int i = 0; i + 1 < nUp; ++i)
                    secF[i] = clamp01(diffuseReflectance(scene, *mp, h, hb.lam[i + 1]));
                break;
            }
            case MatType::Glossy: {
                Vec3 mdir = reflect(ray.d, cur.ns);   // ray.d == -wo (incoming dir)
                wi = sampleGlossy(mdir, materialRoughness(scene, *mp, h), rng);
                if (dot(wi, cur.ns) <= 0) { terminate = true; break; }
                double r = clamp01(reflectSlot(scene, *mp, h, lambda));
                pdfW = bsdfPdf(*mp, cur.ns, wo, wi, lambda, scene, &h);
                pdfRevW = bsdfPdf(*mp, cur.ns, wi, wo, lambda, scene, &h);
                betaFactor = r;                       // f*cos/pdf = r
                if (pdfW <= 0) terminate = true;      // r <= 0 is caught by the max test
                // The glossy LOBE (mirror direction + roughness exponent) carries no
                // wavelength dependence, so the whole bundle can follow the sampled
                // direction and only the reflectance differs per λ. (The unidirectional
                // hero tracers de-hero here instead — see known-issues.md.)
                secChromatic = true;
                for (int i = 0; i + 1 < nUp; ++i)
                    secF[i] = clamp01(reflectSlot(scene, *mp, h, hb.lam[i + 1]));
                break;
            }
            case MatType::DiffuseTransmit: {
                // Pick the reflect or transmit lobe in proportion to their albedos, then
                // cosine-sample the corresponding hemisphere (front = +ns, back = -ns).
                // f*cos/pdf collapses to the TOTAL albedo (rhoR+rhoT) either way, so the
                // throughput darkens by the total albedo per bounce (expected-value, like
                // the Diffuse case). MIS densities come from bsdfPdf (lobe-select * cos/PI).
                double rhoR, rhoT; diffuseTransmitAlbedos(*mp, lambda, scene, &h, rhoR, rhoT);
                double tot = rhoR + rhoT;
                if (tot <= 0.0) { terminate = true; break; }
                const bool reflLobe = (rng.uniform() * tot < rhoR);
                if (reflLobe) wi = cosineHemisphere(cur.ns, rng);          // reflect
                else          wi = cosineHemisphere(cur.ns * -1.0, rng);   // transmit
                pdfW    = bsdfPdf(*mp, cur.ns, wo, wi, lambda, scene, &h);
                pdfRevW = bsdfPdf(*mp, cur.ns, wi, wo, lambda, scene, &h);
                betaFactor = tot;                     // f*cos/pdf = rhoR+rhoT
                if (pdfW <= 0) terminate = true;
                // The lobe was CHOSEN by the hero's albedo split, so each secondary
                // divides by the hero's albedo for that lobe, not its own:
                // f_i·cos/pdf_hero = rho_i(lobe) · tot_hero / rho_hero(lobe).
                secChromatic = true;
                for (int i = 0; i + 1 < nUp; ++i) {
                    double rR, rT; diffuseTransmitAlbedos(*mp, hb.lam[i + 1], scene, &h, rR, rT);
                    double num = reflLobe ? rR   : rT;
                    double den = reflLobe ? rhoR : rhoT;
                    secF[i] = (den > 0.0) ? num * tot / den : 0.0;
                }
                break;
            }
            case MatType::Mirror: {
                double r = clamp01(reflectSlot(scene, *mp, h, lambda));
                wi = reflect(ray.d, cur.ns);
                betaFactor = r; delta = true;
                // The mirror direction is the same for every λ, so the bundle survives;
                // only the reflectance is per-λ (cf. Glossy, the rough version of this).
                keepBundle = true; secChromatic = true;
                for (int i = 0; i + 1 < nUp; ++i)
                    secF[i] = clamp01(reflectSlot(scene, *mp, h, hb.lam[i + 1]));
                break;
            }
            case MatType::Dielectric: {
                // Nested-dielectric PRIORITY resolution: exterior IOR = the medium the
                // subpath is currently inside (highest-priority stack entry). Overlapping
                // dielectrics are ranked by `priority` (higher wins; lower is suppressed
                // -> straight pass-through). SAFE FALLBACK to flat air<->glass (extIor 1.0)
                // unless BOTH sides carry an explicit priority, keeping priority-free
                // scenes bit-identical.
                bool entering = dot(ray.d, h.ng) < 0.0;
                const int mi = (int)(mp - scene.mats.data());   // true index (Mix/Layered aware)
                const int pr = mp->priority;
                delta = true; betaFactor = 1.0;
                if (entering) {
                    const int outMat = stk.topMat();
                    const int outPri = stk.topPri();
                    const bool ranked = mp->hasPriority() &&
                        (stk.empty() || (outMat >= 0 && scene.mats[outMat].hasPriority()));
                    if (ranked && !stk.empty() && pr <= outPri) {   // suppressed inner surface
                        wi = ray.d; stk.push(mi, pr);
                    } else {
                        const double extIor = (ranked && outMat >= 0)
                            ? scene.mats[outMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        Ray nr = mats.refractOrReflect(scene, *mp, h, ray.d, lambda, rng, &transmitted, extIor);
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
                        wi = ray.d; stk.popMat(mi);
                    } else {
                        const double extIor = (ranked && newMat >= 0)
                            ? scene.mats[newMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        Ray nr = mats.refractOrReflect(scene, *mp, h, ray.d, lambda, rng, &transmitted, extIor);
                        wi = nr.d;
                        if (transmitted) stk.popMat(mi);            // TIR stays inside mi
                    }
                }
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, *mp, h, lambda));
                if (rng.uniform() < r) wi = reflect(ray.d, cur.ns);
                else                   wi = ray.d;    // transmit straight
                betaFactor = 1.0; delta = true;
                break;
            }
            case MatType::Filter: {
                // Colored gel filter: straight-through delta, throughput ×= T(lambda).
                double t = clamp01(transmitSlot(scene, *mp, h, lambda));
                wi = ray.d; betaFactor = t; delta = true;   // direction unchanged
                // Straight-through for every λ, so the bundle survives; a gel filter is
                // exactly where the per-λ transmittance spread is largest, so this is
                // the case that benefits most from NOT de-heroing — AND the case that
                // forces the absolute (rather than ratio) formulation of secF, since
                // T(λ_hero) is legitimately 0 across most of a Wratten passband.
                keepBundle = true; secChromatic = true;
                for (int i = 0; i + 1 < nUp; ++i)
                    secF[i] = clamp01(transmitSlot(scene, *mp, h, hb.lam[i + 1]));
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
                double r = clamp01(reflectSlot(scene, *mp, h, lambda));
                if (r <= 0) { terminate = true; break; }
                bool absorbedG; Ray nr = mats.gratingDiffract(*mp, h, ray.d, lambda, rng, absorbedG);
                if (absorbedG) { terminate = true; break; }
                wi = nr.d; betaFactor = r; delta = true;
                break;
            }
            default: terminate = true; break;
        }
        // Kill the walk only when EVERY live wavelength is dead. The hero's own factor can
        // legitimately be 0 while a secondary's is not (a gel filter, a saturated spectral
        // reflectance), and dropping the whole bundle there biases the estimate low — it
        // measured -4.9 % on a Wratten-58 test scene. With nUp == 1 the loop is empty and
        // mxF == betaFactor, so this is exactly the old scalar test.
        double mxF = betaFactor;
        if (secChromatic) for (int i = 0; i + 1 < nUp; ++i) if (secF[i] > mxF) mxF = secF[i];
        if (terminate || mxF <= 0.0) return;

        // Specular vertices carry a delta density: PBRT stores 0 for both the forward
        // and reverse area densities so MIS skips connections through them.
        cur.delta = delta;
        if (delta) { pdfW = 0.0; pdfRevW = 0.0; }

        // The reverse density flows back to the previous vertex (area measure).
        path[prevSurfIdx].pdfRev = convertDensity(pdfRevW, cur, path[prevSurfIdx]);

        beta *= betaFactor;
        for (int i = 0; i + 1 < nUp; ++i) betaSec[i] *= secChromatic ? secF[i] : betaFactor;
        // Veach shading-normal ADJOINT correction (§5.3) for the LIGHT (Importance)
        // subpath only: a particle tracer deposits irradiance per GEOMETRIC area, so an
        // interpolated shading normal must be reweighted at each non-specular vertex or
        // the mesh facets in mode D (exactly as in modes A/B/C, render.h). `wo` points
        // toward the previous (light-side) vertex (= Veach's wi); the sampled
        // continuation `wi` is the outgoing direction (= Veach's wo). Exactly 1 when
        // ns==ng, so flat triangles / analytic spheres stay bit-identical, and the eye
        // (Radiance) subpath — which smooth-shades for free — is untouched.
        if (mode == Mode::Importance && !delta) {
            Vec3 ngo = (dot(cur.ng, cur.ns) >= 0.0) ? cur.ng : cur.ng * -1.0;
            const double adj = shadingAdjointCorr(wo, normalize(wi), cur.ns, ngo);
            beta *= adj;                                     // purely geometric: same for every λ
            for (int i = 0; i + 1 < nUp; ++i) betaSec[i] *= adj;
        }
        // DE-HERO. A delta vertex picked its continuation by a wavelength-dependent
        // specular process (dielectric refraction, grating order, thin-film/multilayer
        // interface, the half-mirror's r(λ) coin), so the secondaries cannot ride the
        // hero's outgoing direction any further and stop here. The vertex JUST pushed
        // keeps its full nUp (it really was reached by all C wavelengths); only the
        // continuation collapses. Delta vertices are exactly the non-connectible ones,
        // so this also means every vertex that can take part in a connection has a
        // meaningful nUp. EXCEPTION: Mirror and Filter are delta but pick their
        // continuation without consulting λ, so they set `keepBundle` and carry the
        // secondaries through on a per-λ `secF` instead (see those cases above).
        if (delta && !keepBundle) nUp = 1;
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
                                 int px, int py, const HeroBundle& hb, int maxDepth,
                                 Pcg32& rng, std::vector<Vertex>& path,
                                 Escape* esc = nullptr) {
    const double lambda = hb.lam[0];
    path.clear();
    // The camera vertex is wavelength-neutral: every λ in the bundle leaves it with
    // throughput 1 (the per-pixel radiance convention), so the bundle starts C wide.
    // (Hero is gated off for a lensed camera, whose ray IS wavelength-dependent.)
    double betaSec0[hero::kHeroMax - 1];
    for (int i = 0; i + 1 < hb.C; ++i) betaSec0[i] = 1.0;
    Vertex c;
    c.type = VType::Camera; c.ns = cam.w; c.ng = cam.w; c.beta = 1.0;
    c.nUp = hb.C;
    for (int i = 0; i + 1 < hb.C; ++i) c.betaSec[i] = 1.0;
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
        c.nUp = 1;               // lensed cameras are single-λ (hero gate excludes them)
        path.push_back(c);
        double pdfDir = cameraPdfDir(cam, dot(ray.d, cam.w));   // MIS-irrelevant placeholder
        randomWalk(scene, cam, mats, ray, wLens, pdfDir, hb, maxDepth - 1,
                   Mode::Radiance, rng, path, betaSec0, 1, esc);
        return (int)path.size();
    }
    c.p = cam.eye;
    path.push_back(c);
    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
    double cosCam = dot(ray.d, cam.w);
    double pdfDir = cameraPdfDir(cam, cosCam);
    randomWalk(scene, cam, mats, ray, /*beta*/1.0, pdfDir, hb, maxDepth - 1,
               Mode::Radiance, rng, path, betaSec0, hb.C, esc);
    return (int)path.size();
}

// Emit a light subpath from a DELTA light (spot or sun) — the two emitters with no
// finite emissive surface. Shares the tail (random walk + throughput bookkeeping) with
// the area-light path below, but the endpoint sampling is entirely different:
//
//   Spot  a point at `origin` radiating uniformly into the OUTER cone (pdfW =
//         1/(2 PI (1-cosOuter))); the smoothstep falloff is carried as THROUGHPUT, not
//         density, so the mean walk weight is I*spotOmega = the emitter's power. `spdFn`
//         is the peak intensity per unit SPD (geomWeight == spotOmega), so the emitted
//         quantity is an intensity (W/sr), not a radiance.
//   Sun   an infinitely distant disc: the origin is a point on a disc of radius
//         sceneRadius, centred one radius UPSTREAM of the scene centre and perpendicular
//         to `beamDir`, so every emitted ray enters the scene's cross-section (pdfPos =
//         1/(pi R^2)); the direction is drawn inside the solar cone (pdfW = 1/Omega).
//         `spdFn` is the sun's radiance (addSunLight already divided by Omega).
//
// Both are delta lights: path[0].pdfFwd is 0 (PBRT's convention — see
// vertexPdfLightOrigin) and misWeight drops the s=0 strategy for them. For the INFINITE
// sun the first scene vertex's forward density must also be rewritten to the planar
// 1/(pi R^2) form that vertexPdfLight reports in reverse (PBRT's "correct subpath
// sampling densities for infinite area lights" patch), or the two sides disagree and the
// MIS weights are wrong. Returns the subpath length.
inline int deltaLightSubpath(const Scene& scene, const Camera& cam, const Renderer& mats,
                             const Emitter& em, double pdfChoice, const HeroBundle& hb,
                             int maxDepth, Pcg32& rng, std::vector<Vertex>& path) {
    const double lambda = hb.lam[0], invPdfLambda = hb.invPdf[0];
    const bool isSun = (em.shape == EmitterShape::Sun);
    double u1 = rng.uniform(), u2 = rng.uniform();
    Vec3 dir = em.sampleCone(em.beamDir, u1, u2);    // uniform in the (outer / solar) cone
    double coneSolid = 2.0 * PI * (1.0 - em.spotCosOuter);
    if (coneSolid <= 0.0) return 0;
    double pdfDir = 1.0 / coneSolid;

    Vec3 org;
    double pdfPos = 1.0;                             // delta (spot) unless the sun's disc
    if (isSun) {
        double R = scene.sceneRadius;
        if (R <= 0.0) return 0;
        Vec3 t, b; onb(em.beamDir, t, b);
        double rr = R * std::sqrt(rng.uniform()), phi = 2.0 * PI * rng.uniform();
        org = scene.sceneCenter + t * (rr * std::cos(phi)) + b * (rr * std::sin(phi))
            - em.beamDir * R;
        pdfPos = 1.0 / (PI * R * R);
    } else {
        org = em.origin;
    }
    // Spot: the smoothstep penumbra scales the emitted intensity. Sun: no falloff.
    double fall = isSun ? 1.0
                        : spotFalloff(dot(dir, em.beamDir), em.spotCosInner, em.spotCosOuter);
    if (fall <= 0.0) return 0;
    double Le = em.spdFn(lambda) * invPdfLambda * fall;
    double LeSec[hero::kHeroMax - 1] = {0};
    double mxLe = Le;
    for (int i = 0; i + 1 < hb.C; ++i) {
        LeSec[i] = em.spdFn(hb.lam[i + 1]) * hb.invPdf[i + 1] * fall;
        if (LeSec[i] > mxLe) mxLe = LeSec[i];
    }
    if (mxLe <= 0.0) return 0;

    Vertex L0;
    L0.type = VType::Light; L0.p = org; L0.ns = dir; L0.ng = dir;
    L0.light = &em; L0.matId = -1; L0.mat = nullptr;
    L0.beta = Le;
    L0.pdfFwd = 0.0;                                 // delta origin (see vertexPdfLightOrigin)
    L0.delta = false;                                // NEE (s=1) to a spot/sun IS possible
    L0.nUp = hb.C;
    for (int i = 0; i + 1 < hb.C; ++i) L0.betaSec[i] = LeSec[i];
    path.push_back(L0);

    // Walk throughput = Le / (pdfChoice * pdfPos * pdfDir); no cosine, because the
    // emission normal IS the emission direction for both of these (|cos| == 1).
    const double invP = 1.0 / (pdfChoice * pdfPos * pdfDir);
    double betaWalk = Le * invP;
    double betaWalkSec[hero::kHeroMax - 1];
    for (int i = 0; i + 1 < hb.C; ++i) betaWalkSec[i] = LeSec[i] * invP;
    Ray ray{org, dir};
    randomWalk(scene, cam, mats, ray, betaWalk, pdfDir, hb, maxDepth - 1,
               Mode::Importance, rng, path, betaWalkSec, hb.C);
    // Infinite-light density patch (PBRT): the first scene vertex was given a solid-angle
    // density converted with 1/dist^2 from the fictitious disc point, but the reverse
    // direction (vertexPdfLight) reports the planar 1/(pi R^2). Rewrite it to match.
    if (isSun && path.size() > 1) {
        double pdf = pdfPos;
        if (path[1].onSurface()) pdf *= std::abs(dot(path[1].ns, dir));
        path[1].pdfFwd = pdf;
    }
    return (int)path.size();
}

// Sample a light subpath: choose an emitter (power-weighted), a surface point and a
// cosine-distributed emission direction, at the shared wavelength `lambda`
// (invPdfLambda folds the wavelength importance so Le is radiance, as in backward.h).
// path[0] is the light endpoint (beta = Le, its pdfFwd the positional area density).
// Spot and sun lights route to deltaLightSubpath above; env/collimated are out of scope
// (the mode-D guard refuses those scenes before any of this runs).
inline int generateLightSubpath(const Scene& scene, const Camera& cam, const Renderer& mats,
                                const HeroBundle& hb, int maxDepth,
                                Pcg32& rng, std::vector<Vertex>& path) {
    const double lambda = hb.lam[0], invPdfLambda = hb.invPdf[0];
    path.clear();
    if (scene.emitters.empty() || scene.totalPower <= 0.0) return 0;
    int ei = scene.selectEmitter(rng);
    const Emitter& em = scene.emitters[ei];
    double pdfChoiceSel = em.power / scene.totalPower;
    if (pdfChoiceSel <= 0.0) return 0;
    if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Sun)
        return deltaLightSubpath(scene, cam, mats, em, pdfChoiceSel, hb, maxDepth, rng, path);
    if (em.shape == EmitterShape::Env || em.collimated)
        return 0;                                    // unsupported in BDPT scope

    double u1 = rng.uniform(), u2 = rng.uniform();
    Vec3 y, nOut;
    // `emitPatW` is this point's `emit pattern:` factor (1.0, and a bit-identical call,
    // when the emitter has none). It scales the emitted radiance only — the positional
    // pdf below stays 1/area and pdfChoice stays power-weighted, exactly as the eye
    // subpath's s=0/s=1 MIS terms assume — so the estimator is unchanged apart from the
    // radiance itself. Vertex::emitPatW carries the same factor for those MIS terms.
    double emitPatW = emitterSamplePoint(scene, em, u1, u2, y, nOut);
    double Le = em.spdFn(lambda) * invPdfLambda * emitPatW;   // emitted radiance at lambda
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
    // Hero bundle: the emitter, its sampled point and the emission direction are all
    // chosen once (by the hero), so the secondaries differ ONLY in the emitted radiance
    // Le(λ)/p(λ) they start with. The eye subpath is generated from the same bundle, so
    // both sides of every connection speak about the same C wavelengths.
    L0.nUp = hb.C;
    L0.emitPatW = emitPatW;
    for (int i = 0; i + 1 < hb.C; ++i)
        L0.betaSec[i] = em.spdFn(hb.lam[i + 1]) * hb.invPdf[i + 1] * emitPatW;
    path.push_back(L0);

    Vec3 dir = cosineHemisphere(nOut, rng);
    double cosLight = dot(nOut, dir);
    if (cosLight <= 0.0) return 1;
    double pdfDir = cosLight / PI;                    // cosine-weighted emission
    // Walk throughput = Le * cosLight / (pdfChoice * pdfPos * pdfDir)
    //                 = Le * area / pdfChoice  (= emitter power for a single light).
    double betaWalk = Le * cosLight / (pdfChoice * pdfPos * pdfDir);
    double betaWalkSec[hero::kHeroMax - 1];
    for (int i = 0; i + 1 < hb.C; ++i)
        betaWalkSec[i] = L0.betaSec[i] * cosLight / (pdfChoice * pdfPos * pdfDir);
    Ray ray{y + nOut * 1e-6, dir};
    randomWalk(scene, cam, mats, ray, betaWalk, pdfDir, hb, maxDepth - 1,
               Mode::Importance, rng, path, betaWalkSec, hb.C);
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
                        Vertex& sampled, int s, int t, double lambda) {
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
        double val = (s > 0) ? vertexPdf(scene, cam, qsM, *qs, *pt, lambda)
                             : vertexPdfLightOrigin(scene, *pt);
        a4 = ScopedAssign<double>(&pt->pdfRev, val);
    }
    // Reverse density of pt's predecessor.
    ScopedAssign<double> a5;
    if (ptM) {
        double val = (s > 0) ? vertexPdf(scene, cam, qs, *pt, *ptM, lambda)
                             : vertexPdfLight(scene, *pt, *ptM);
        a5 = ScopedAssign<double>(&ptM->pdfRev, val);
    }
    // Reverse density of the light connection vertex qs and its predecessor.
    ScopedAssign<double> a6;
    if (qs) a6 = ScopedAssign<double>(&qs->pdfRev, vertexPdf(scene, cam, ptM, *pt, *qs, lambda));
    ScopedAssign<double> a7;
    if (qsM) a7 = ScopedAssign<double>(&qsM->pdfRev, vertexPdf(scene, cam, pt, *qs, *qsM, lambda));

    double sumRi = 0.0, ri = 1.0;
    for (int i = t - 1; i > 0; --i) {                // hypothetical camera strategies
        ri *= remap0(eye[i].pdfRev) / remap0(eye[i].pdfFwd);
        if (!eye[i].delta && !eye[i - 1].delta) sumRi += ri;
    }
    ri = 1.0;
    for (int i = s - 1; i >= 0; --i) {               // hypothetical light strategies
        ri *= remap0(light[i].pdfRev) / remap0(light[i].pdfFwd);
        // The hypothetical strategy at index i connects light[i-1] to the eye side, so it
        // is impossible if either end of that new edge is delta. At i == 0 the "edge" is
        // instead the eye path LANDING on the emitter, which a delta light (spot: a point;
        // sun: infinitely far, no geometry) can never be hit by — PBRT's IsDeltaLight().
        bool deltaPrev = (i > 0) ? light[i - 1].delta : light[0].isDeltaLight();
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

// Connection-ray origin at a connectible vertex. A surface offsets off its geometric
// normal (above); a medium in-scatter point has no surface to self-occlude against, so
// the exact point is used (the transmittance factor accounts for the fog it sits in).
inline Vec3 connOrigin(const Vertex& v, const Vec3& dir) {
    if (v.type == VType::Medium) return v.p;
    return offsetOrigin(v, dir);
}

// --- Connect one strategy (s,t) --------------------------------------------------
// Returns the MIS-weighted radiance contribution of connecting the s-vertex light
// subpath with the t-vertex eye subpath. For t==1 the contribution is a light-image
// splat to raster (outPx,outPy) with isSplat=true; otherwise it belongs to the
// current pixel. `light`/`eye` are non-const because misWeight mutates them.
// Hero bundle: the return value is the HERO wavelength's contribution; `Lsec` receives
// the C-1 secondaries' and `nUpConn` how many wavelengths this connection actually
// carries — min(nUp of the two endpoints), since either subpath may have de-hero'd
// independently. `nUpConn == 0` means "no contribution" (every early-out leaves it 0),
// which is what the caller tests instead of the return value: the hero can legitimately
// evaluate to zero (a wall that is black at λ0) while a secondary does not.
// Every SAMPLING decision here (the emitter pick, the NEE point, the MIS weight) is
// still made at the hero wavelength; only the evaluated radiance is per-λ.
inline double connectBDPT(const Scene& scene, const Camera& cam, const Renderer& mats,
                          std::vector<Vertex>& light, std::vector<Vertex>& eye,
                          int s, int t, const HeroBundle& hb,
                          Pcg32& rng, int& outPx, int& outPy, bool& isSplat,
                          double* Lsec, int& nUpConn) {
    const double lambda = hb.lam[0], invPdfLambda = hb.invPdf[0];
    isSplat = false;
    nUpConn = 0;                    // set to the real width only once a contribution exists
    // Can't connect ONTO a vertex that already sits on a light (PBRT guard).
    if (t > 1 && s != 0 && eye[t - 1].isLightVertex()) return 0.0;

    double L = 0.0;
    int nUp = 1;                    // live wavelengths for THIS connection (set per branch)
    Vertex sampled;

    if (s == 0) {
        // Pure eye path: contributes iff its last vertex is emissive.
        if (t < 2) return 0.0;
        const Vertex& pt = eye[t - 1];
        if (!pt.isLightVertex()) return 0.0;
        Vec3 wo = normalize(eye[t - 2].p - pt.p);
        double Le = pt.Le(wo, lambda, invPdfLambda);
        nUp = pt.nUp;
        // The hero may legitimately be black where a secondary is not, so the early-out
        // tests the max over the live wavelengths (identical to `Le <= 0` when nUp==1).
        double LeSec[hero::kHeroMax - 1] = {0}, mxLe = Le;
        for (int i = 0; i + 1 < nUp; ++i) {
            LeSec[i] = pt.Le(wo, hb.lam[i + 1], hb.invPdf[i + 1]);
            if (LeSec[i] > mxLe) mxLe = LeSec[i];
        }
        if (mxLe <= 0.0) return 0.0;
        L = pt.beta * Le;
        for (int i = 0; i + 1 < nUp; ++i) Lsec[i] = pt.betaSec[i] * LeSec[i];
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
        Vec3 wo = normalize(light[s - 2].p - qs.p);
        // Scattering value f and the endpoint cosine. A medium vertex has no surface:
        // its phase function replaces the BSDF and the geometry cosine is 1.
        double cosSurf, f, fSec[hero::kHeroMax - 1] = {0};
        nUp = qs.nUp;
        if (qs.type == VType::Medium) {
            cosSurf = 1.0;
            f = mediumScatterF(qs, wo, wcam, lambda, scene);
            for (int i = 0; i + 1 < nUp; ++i)
                fSec[i] = mediumScatterF(qs, wo, wcam, hb.lam[i + 1], scene);
        } else {
            cosSurf = dot(qs.ns, wcam);
            // Reflect-only vertices require the +ns side; a two-sided vertex may connect on
            // either side (transmit lobe), so gate on bsdfF and use |cosSurf| in G.
            if (cosSurf == 0.0 || (!isTwoSidedMat(*qs.mat) && cosSurf < 0.0)) return 0.0;
            Vec3 ngo = (dot(qs.ng, qs.ns) >= 0.0) ? qs.ng : qs.ng * -1.0;
            // Geometric-hemisphere softening: a reflect-only vertex must see the camera on
            // its GEOMETRIC front side, else a smoothed shading normal leaks light through the
            // back face (shading-normal problem). A hard cutoff there facets the terminator,
            // so ramp smoothly (Chiang 2019; matches backward.h/render.h). No-op when ns==ng
            // (flat/analytic, stG==1); skipped for two-sided (transmissive) materials.
            double stG = isTwoSidedMat(*qs.mat) ? 1.0 : shadowTerminatorG(wcam, qs.ns, ngo);
            if (stG <= 0.0) return 0.0;
            f = bsdfF(*qs.mat, qs.ns, wo, wcam, lambda, scene, &qs.hit);
            // Adjoint shading-normal correction: qs is a LIGHT-subpath (particle) vertex
            // whose f is evaluated toward the camera (wcam = outgoing). 1 when ns==ng.
            // The correction and stG are pure geometry — shared by every wavelength.
            const double adj = shadingAdjointCorr(wo, wcam, qs.ns, ngo) * stG;
            f *= adj;
            for (int i = 0; i + 1 < nUp; ++i)
                fSec[i] = bsdfF(*qs.mat, qs.ns, wo, wcam, hb.lam[i + 1], scene, &qs.hit) * adj;
        }
        {   // max over live wavelengths (identical to `f <= 0` when nUp==1)
            double mxF = f;
            for (int i = 0; i + 1 < nUp; ++i) if (fSec[i] > mxF) mxF = fSec[i];
            if (mxF <= 0.0) return 0.0;
        }
        if (scene.occluded(connOrigin(qs, wcam), wcam, dist - 2e-6)) return 0.0;
        // Transmittance of the fog the connection ray crosses (1 in vacuum, no RNG).
        // Evaluated at the hero only: the hero gate disables bundling when the scene has
        // any medium, so Tr is exactly 1 whenever nUp > 1.
        double Tr = mats.mediaTransmittance(scene, qs.p, wcam, dist, lambda, rng);
        double G = std::fabs(cosSurf) * cosCam / dist2;
        L = qs.beta * f * G * cameraWe(cam, cosCam) * Tr;
        for (int i = 0; i + 1 < nUp; ++i)
            Lsec[i] = qs.betaSec[i] * fSec[i] * G * cameraWe(cam, cosCam) * Tr;
        sampled.type = VType::Camera; sampled.p = cam.eye; sampled.ns = cam.w; sampled.ng = cam.w;
        sampled.beta = 1.0; sampled.delta = false;
        outPx = px; outPy = py; isSplat = true;
    } else if (s == 1) {
        // NEE: connect the eye vertex to a freshly sampled point on a light.
        const Vertex& pt = eye[t - 1];
        if (!pt.isConnectible()) return 0.0;
        int ei = scene.selectEmitter(rng);
        const Emitter& em = scene.emitters[ei];
        if (em.shape == EmitterShape::Env || em.collimated)
            return 0.0;                                // out of BDPT scope (mode-D guard)
        double pdfChoice = em.power / scene.totalPower;
        if (pdfChoice <= 0.0) return 0.0;
        double u1 = rng.uniform(), u2 = rng.uniform();
        // Per-shape connection geometry. `Wgeom` is the whole lambda-independent weight
        // that multiplies |cosSurf| in the estimator, i.e.
        //     L = beta * f * Le * |cosSurf| * Wgeom * Tr * stG
        // which for an area light is the familiar cosLight/(dist^2 * pdfA). Collecting it
        // into one scalar is what lets the three emission models (Lambertian area, spot
        // cone, distant sun) share the BSDF / occlusion / transmittance code below.
        Vec3 y, nOut, wi;
        double dist, Wgeom, emitPatW = 1.0;
        const bool deltaLight = isDeltaEmitter(em);
        // A distant sun has no finite light point: the shadow ray runs all the way to the
        // scene exit, so it must NOT be shortened by the usual endpoint epsilon.
        double occlEps = 2e-6;
        if (em.shape == EmitterShape::Spot) {
            // Point spot: the connection point is deterministic (delta position); the
            // smoothstep penumbra weights the intensity toward this receiver.
            Vec3 toL = em.origin - pt.p; double dist2 = dot(toL, toL);
            if (dist2 <= 0.0) return 0.0;
            dist = std::sqrt(dist2); wi = toL / dist;
            double fall = spotFalloff(dot(wi * -1.0, em.beamDir),
                                      em.spotCosInner, em.spotCosOuter);
            if (fall <= 0.0) return 0.0;               // outside the cone
            y = em.origin; nOut = wi * -1.0;
            Wgeom = fall / (dist2 * pdfChoice);        // spdFn is an INTENSITY (W/sr)
        } else if (em.shape == EmitterShape::Sun) {
            // Distant sun: sample a direction inside the solar cone (pdfW = 1/Omega) and
            // shadow-ray it out of the scene. No 1/dist^2 and no cosLight — the source is
            // at infinity — so Wgeom is just Omega/pdfChoice and spdFn is a radiance.
            wi = em.sampleCone(em.beamDir * -1.0, u1, u2);
            dist = length(scene.sceneCenter - pt.p) + scene.sceneRadius;
            occlEps = 0.0;
            y = pt.p + wi * dist; nOut = wi * -1.0;
            Wgeom = em.spotOmega / pdfChoice;
        } else {
            // Pattern factor at the sampled point (1.0 without a pattern). It scales Le
            // below, never pdfA — see generateLightSubpath for why that stays unbiased.
            emitPatW = emitterSamplePoint(scene, em, u1, u2, y, nOut);
            Vec3 toL = y - pt.p; double dist2 = dot(toL, toL);
            if (dist2 <= 0.0) return 0.0;
            dist = std::sqrt(dist2); wi = toL / dist;
            double cosLight = dot(nOut, wi * -1.0);
            if (cosLight <= 0.0) return 0.0;           // emitter stays one-sided
            if (em.area <= 0.0) return 0.0;
            Wgeom = cosLight * em.area / (dist2 * pdfChoice);   // == cosLight/(d^2 * pdfA)
        }
        Vec3 wo = normalize(eye[t - 2].p - pt.p);
        // Scattering value f and endpoint cosine (phase / cos=1 at a medium vertex).
        double cosSurf, f, stG = 1.0, fSec[hero::kHeroMax - 1] = {0};
        nUp = pt.nUp;
        if (pt.type == VType::Medium) {
            cosSurf = 1.0;
            f = mediumScatterF(pt, wo, wi, lambda, scene);
            for (int i = 0; i + 1 < nUp; ++i)
                fSec[i] = mediumScatterF(pt, wo, wi, hb.lam[i + 1], scene);
        } else {
            cosSurf = dot(pt.ns, wi);
            if (cosSurf == 0.0 || (!isTwoSidedMat(*pt.mat) && cosSurf < 0.0)) return 0.0;
            // Geometric-hemisphere softening (see t==1 splat above): the eye/radiance vertex
            // must see the sampled light on its geometric front side; ramp smoothly instead of
            // a hard cutoff (Chiang 2019). No-op when ns==ng (stG==1).
            if (!isTwoSidedMat(*pt.mat)) {
                Vec3 ngo = (dot(pt.ng, pt.ns) >= 0.0) ? pt.ng : pt.ng * -1.0;
                stG = shadowTerminatorG(wi, pt.ns, ngo);
                if (stG <= 0.0) return 0.0;
            }
            f = bsdfF(*pt.mat, pt.ns, wo, wi, lambda, scene, &pt.hit);
            for (int i = 0; i + 1 < nUp; ++i)
                fSec[i] = bsdfF(*pt.mat, pt.ns, wo, wi, hb.lam[i + 1], scene, &pt.hit);
        }
        {   // max over live wavelengths (identical to `f <= 0` when nUp==1)
            double mxF = f;
            for (int i = 0; i + 1 < nUp; ++i) if (fSec[i] > mxF) mxF = fSec[i];
            if (mxF <= 0.0) return 0.0;
        }
        // The emitter was CHOSEN at the hero wavelength; only its emitted radiance is
        // re-evaluated per-λ (the pdf stays hero-driven, as everywhere else).
        double Le = em.spdFn(lambda) * invPdfLambda * emitPatW;
        double LeSec[hero::kHeroMax - 1] = {0};
        {
            double mxLe = Le;
            for (int i = 0; i + 1 < nUp; ++i) {
                LeSec[i] = em.spdFn(hb.lam[i + 1]) * hb.invPdf[i + 1] * emitPatW;
                if (LeSec[i] > mxLe) mxLe = LeSec[i];
            }
            if (mxLe <= 0.0) return 0.0;
        }
        if (scene.occluded(connOrigin(pt, wi), wi, dist - occlEps)) return 0.0;
        // Hero-only transmittance (exactly 1 whenever nUp > 1; see the t==1 branch).
        double Tr = mats.mediaTransmittance(scene, pt.p, wi, dist, lambda, rng);
        double G = std::fabs(cosSurf) * Wgeom;
        L = pt.beta * f * Le * G * Tr * stG;
        for (int i = 0; i + 1 < nUp; ++i)
            Lsec[i] = pt.betaSec[i] * fSec[i] * LeSec[i] * G * Tr * stG;
        sampled.type = VType::Light; sampled.p = y; sampled.ns = nOut; sampled.ng = nOut;
        sampled.light = &em; sampled.matId = deltaLight ? -1 : em.matId;
        sampled.mat = (sampled.matId >= 0) ? &scene.mats[sampled.matId] : nullptr;
        sampled.emitPatW = emitPatW;
        // A delta light has no area density: pdfFwd is 0 on BOTH sides of every MIS ratio
        // (the light subpath stores the same 0), matching PBRT — see vertexPdfLightOrigin.
        sampled.beta = deltaLight ? Le * Wgeom : Le * em.area / pdfChoice;   // == Le/pdfA
        sampled.delta = false;
        sampled.pdfFwd = deltaLight ? 0.0 : (pdfChoice / em.area);
    } else {
        // Interior connection light[s-1] <-> eye[t-1].
        const Vertex& qs = light[s - 1];
        const Vertex& pt = eye[t - 1];
        if (!qs.isConnectible() || !pt.isConnectible()) return 0.0;
        Vec3 d = qs.p - pt.p; double dist2 = dot(d, d);
        if (dist2 <= 0.0) return 0.0;
        double dist = std::sqrt(dist2); Vec3 w = d / dist;   // pt -> qs
        Vec3 woE = normalize(eye[t - 2].p - pt.p);
        Vec3 woL = normalize(light[s - 2].p - qs.p);
        // Each endpoint is a surface (BSDF, cosine) or a medium (phase, cos=1).
        double cosE, cosL, fE, fL, stGE = 1.0, stGL = 1.0;
        double fESec[hero::kHeroMax - 1] = {0}, fLSec[hero::kHeroMax - 1] = {0};
        // Either subpath may have de-hero'd independently at a delta vertex, so the
        // connection carries only the wavelengths BOTH endpoints still track.
        nUp = (qs.nUp < pt.nUp) ? qs.nUp : pt.nUp;
        if (pt.type == VType::Medium) {
            cosE = 1.0; fE = mediumScatterF(pt, woE, w, lambda, scene);
            for (int i = 0; i + 1 < nUp; ++i)
                fESec[i] = mediumScatterF(pt, woE, w, hb.lam[i + 1], scene);
        } else {
            cosE = dot(pt.ns, w);
            if (cosE == 0.0 || (!isTwoSidedMat(*pt.mat) && cosE < 0.0)) return 0.0;
            // Geometric-hemisphere softening on the eye endpoint (connection dir w): ramp
            // smoothly instead of a hard cutoff (Chiang 2019). No-op ns==ng (stGE==1).
            if (!isTwoSidedMat(*pt.mat)) {
                Vec3 ngoE = (dot(pt.ng, pt.ns) >= 0.0) ? pt.ng : pt.ng * -1.0;
                stGE = shadowTerminatorG(w, pt.ns, ngoE);
                if (stGE <= 0.0) return 0.0;
            }
            fE = bsdfF(*pt.mat, pt.ns, woE, w, lambda, scene, &pt.hit);
            for (int i = 0; i + 1 < nUp; ++i)
                fESec[i] = bsdfF(*pt.mat, pt.ns, woE, w, hb.lam[i + 1], scene, &pt.hit);
        }
        if (qs.type == VType::Medium) {
            cosL = 1.0; fL = mediumScatterF(qs, woL, w * -1.0, lambda, scene);
            for (int i = 0; i + 1 < nUp; ++i)
                fLSec[i] = mediumScatterF(qs, woL, w * -1.0, hb.lam[i + 1], scene);
        } else {
            cosL = dot(qs.ns, w * -1.0);
            if (cosL == 0.0 || (!isTwoSidedMat(*qs.mat) && cosL < 0.0)) return 0.0;
            Vec3 ngoL = (dot(qs.ng, qs.ns) >= 0.0) ? qs.ng : qs.ng * -1.0;
            // Geometric-hemisphere softening on the light endpoint (connection dir -w): ramp
            // smoothly instead of a hard cutoff (Chiang 2019). No-op ns==ng (stGL==1).
            if (!isTwoSidedMat(*qs.mat)) {
                stGL = shadowTerminatorG(w * -1.0, qs.ns, ngoL);
                if (stGL <= 0.0) return 0.0;
            }
            fL = bsdfF(*qs.mat, qs.ns, woL, w * -1.0, lambda, scene, &qs.hit);
            // Adjoint shading-normal correction on the LIGHT-subpath endpoint qs (particle
            // vertex; outgoing = w*-1 toward the eye vertex). The eye endpoint pt is a
            // Radiance vertex and gets NO correction. 1 when ns==ng (flat/analytic).
            // Pure geometry, so it applies unchanged to every wavelength.
            const double adjL = shadingAdjointCorr(woL, w * -1.0, qs.ns, ngoL);
            fL *= adjL;
            for (int i = 0; i + 1 < nUp; ++i)
                fLSec[i] = bsdfF(*qs.mat, qs.ns, woL, w * -1.0, hb.lam[i + 1], scene, &qs.hit)
                           * adjL;
        }
        {   // max over live wavelengths (identical to `fE<=0 || fL<=0` when nUp==1)
            double mxE = fE, mxL = fL;
            for (int i = 0; i + 1 < nUp; ++i) {
                if (fESec[i] > mxE) mxE = fESec[i];
                if (fLSec[i] > mxL) mxL = fLSec[i];
            }
            if (mxE <= 0.0 || mxL <= 0.0) return 0.0;
        }
        if (scene.occluded(connOrigin(pt, w), w, dist - 2e-6)) return 0.0;
        // Hero-only transmittance (exactly 1 whenever nUp > 1; see the t==1 branch).
        double Tr = mats.mediaTransmittance(scene, pt.p, w, dist, lambda, rng);
        double G = std::fabs(cosE) * std::fabs(cosL) / dist2;
        L = pt.beta * fE * fL * qs.beta * G * Tr * stGE * stGL;
        for (int i = 0; i + 1 < nUp; ++i)
            Lsec[i] = pt.betaSec[i] * fESec[i] * fLSec[i] * qs.betaSec[i] * G * Tr * stGE * stGL;
    }
    // The MIS weight is a function of the pdfs alone, and every pdf in this renderer is
    // decided by the hero wavelength — so ONE weight serves the whole bundle.
    double mx = L;
    for (int i = 0; i + 1 < nUp; ++i) if (Lsec[i] > mx) mx = Lsec[i];
    if (mx <= 0.0) return 0.0;
    const double mis = misWeight(scene, cam, light, eye, sampled, s, t, lambda);
    for (int i = 0; i + 1 < nUp; ++i) Lsec[i] *= mis;
    nUpConn = nUp;
    return L * mis;
}

// --- Renderer --------------------------------------------------------------------
// Renders pixel rows [y0,y1). t>=2 (camera-image) contributions land on the current
// pixel in `camFilm`; t==1 (light-image) splats land on `splatFilm` at the projected
// raster position. The caller normalises camFilm by spp and splatFilm by the total
// light-subpath count (W*H*spp), matching mode B's absolute-radiance convention.
struct BdptRenderer {
    int maxDepth = 8;          // maximum path length in edges (connection cost ~ depth^2)
    bool diffraction = true;   // mirrors Renderer::diffraction for MatType::Grating
    int heroC = 1;             // wavelengths bundled per path pair (1 = plain single-λ)

    // `sampleBase` = absolute index of the first sample rendered here; each
    // (pixel, absolute sample) seeds its own stream via seedUnit(), so the
    // realization is chunk-split / banding / thread-count independent (see
    // BackwardRenderer::renderRows).
    void renderRows(const Scene& scene, const Camera& cam, Film& camFilm, Film& splatFilm,
                    int y0, int y1, long long spp, unsigned long long sampleBase) const {
        Renderer mats; mats.diffraction = diffraction;
        std::vector<Vertex> eye, light;
        const uint64_t nPix = (uint64_t)camFilm.resX * (uint64_t)camFilm.resY;
        // Hero gate, matching BackwardRenderer: bundling needs a wavelength-independent
        // ray path, so any participating medium (per-λ free flight), a GRIN field
        // (per-λ curvature) or a dispersive finite lens (per-λ refraction at the
        // elements) forces the scalar single-λ path.
        const int C = (heroC > hero::kHeroMax) ? hero::kHeroMax : heroC;
        const bool useHero = (C > 1) && !scene.backwardMedium().enabled &&
                             !grin::sceneHasGrin(scene) && !cam.hasLens();
        // Only pay for escape tracking when the scene actually has a distant sun.
        const bool hasSun = scene.sunCount > 0;
        for (int py = y0; py < y1; ++py)
            for (int px = 0; px < camFilm.resX; ++px) {
                const uint64_t pixIdx = (uint64_t)py * (uint64_t)camFilm.resX + (uint64_t)px;
                for (long long si = 0; si < spp; ++si) {
                    Pcg32 rng;
                    seedUnit(rng, (sampleBase + (uint64_t)si) * nPix + pixIdx,
                             0x8CB92BA72F3D8DD7ULL);
                    HeroBundle hb;
                    if (useHero) {
                        // One stratified base draw -> hero + C-1 secondaries, all from
                        // the emission CDF (hero.h policy 1). The hero (index 0) must have
                        // a valid pdf; a dead secondary carries invPdf 0 and contributes 0.
                        double pdfA[hero::kHeroMax];
                        if (!hero::sampleBundle(scene.emitSampler, rng.uniform(), C,
                                                hb.lam, pdfA)) continue;
                        hb.C = C;
                        hb.invPdf[0] = scene.invPdfLambda(hb.lam[0]);
                        for (int i = 1; i < C; ++i)
                            hb.invPdf[i] = (pdfA[i] > 0.0) ? scene.invPdfLambda(hb.lam[i]) : 0.0;
                    } else {
                        double pdfLam = 0.0;
                        hb.lam[0] = scene.emitSampler.sample(rng, pdfLam);
                        if (pdfLam <= 0.0) continue;
                        hb.invPdf[0] = scene.invPdfLambda(hb.lam[0]);
                        hb.C = 1;
                    }
                    Escape esc;
                    int nE = generateCameraSubpath(scene, cam, mats, px, py, hb,
                                                   maxDepth + 1, rng, eye,
                                                   hasSun ? &esc : nullptr);
                    int nL = generateLightSubpath(scene, cam, mats, hb,
                                                  maxDepth + 1, rng, light);
                    Vec3 cie[hero::kHeroMax];
                    for (int i = 0; i < hb.C; ++i)
                        cie[i] = Vec3(cieX(hb.lam[i]), cieY(hb.lam[i]), cieZ(hb.lam[i]));
                    // Direct view of a distant `light sun`. A sun is a delta-DIRECTION
                    // emitter with no geometry, so misWeight drops the s=0 strategy for it
                    // and no connection strategy can reach it either — without this the
                    // solar disc itself, and every mirror/water glint of it, would simply
                    // be missing from mode D. It is added only when the escaping ray came
                    // through camera + delta (specular) vertices ONLY, which is exactly
                    // the case where no other strategy competes: NEE needs a connectible
                    // (non-delta) vertex, and so does every s>=2 connection or t==1 splat.
                    // The MIS weight is therefore exactly 1.
                    if (esc.escaped) {
                        bool allDelta = true;
                        for (int i = 1; i < nE; ++i)
                            if (!eye[i].delta) { allDelta = false; break; }
                        if (allDelta) {
                            for (const Emitter& em : scene.emitters) {
                                if (em.shape != EmitterShape::Sun) continue;
                                if (!em.inCone(esc.dir)) continue;
                                Vec3 contrib = cie[0] * (esc.beta * em.spdFn(hb.lam[0]) *
                                                         hb.invPdf[0]);
                                for (int i = 0; i + 1 < esc.nUp; ++i)
                                    contrib = contrib + cie[i + 1] *
                                              (esc.betaSec[i] * em.spdFn(hb.lam[i + 1]) *
                                               hb.invPdf[i + 1]);
                                if (esc.nUp > 1) contrib = contrib * (1.0 / esc.nUp);
                                camFilm.add(px, py, contrib);
                            }
                        }
                    }
                    for (int t = 1; t <= nE; ++t)
                        for (int s = 0; s <= nL; ++s) {
                            int depth = t + s - 2;
                            if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth) continue;
                            int spx = 0, spy = 0; bool isSplat = false;
                            double Lsec[hero::kHeroMax - 1] = {0};
                            int nUpConn = 0;
                            double c = connectBDPT(scene, cam, mats, light, eye, s, t, hb,
                                                   rng, spx, spy, isSplat, Lsec, nUpConn);
                            if (nUpConn <= 0) continue;
                            double mx = c;
                            for (int i = 0; i + 1 < nUpConn; ++i)
                                if (Lsec[i] > mx) mx = Lsec[i];
                            if (mx <= 0.0) continue;
                            // Average over the wavelengths this connection actually
                            // carries. Either subpath may have de-hero'd at a delta
                            // vertex, in which case nUpConn == 1 and this reduces
                            // EXACTLY to the scalar single-λ estimator — which is why
                            // no ×C boost is folded into the vertex throughputs (two
                            // independently de-hero'd subpaths would square it).
                            Vec3 contrib = cie[0] * c;
                            for (int i = 0; i + 1 < nUpConn; ++i)
                                contrib = contrib + cie[i + 1] * Lsec[i];
                            if (nUpConn > 1) contrib = contrib * (1.0 / nUpConn);
                            if (isSplat) splatFilm.add(spx, spy, contrib);
                            else         camFilm.add(px, py, contrib);
                        }
                }
            }
    }
};

} // namespace bdpt
