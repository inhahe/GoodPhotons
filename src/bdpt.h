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
// connection pdf is zero). Emission comes from area/sphere quad lights.
//
// PARTICIPATING MEDIA ARE HANDLED — HOMOGENEOUS *AND* HETEROGENEOUS. A subpath can
// scatter at a volume (Medium) vertex via the HG phase function, connections carry a
// transmittance factor, and the balance-heuristic MIS uses cosine-free phase densities.
// For a homogeneous medium the σt·exp free-flight and transmittance terms cancel
// pairwise, so those weights are exact. A heterogeneous (density-field / bounded)
// medium goes through the SAME `mats.sampleMediaCollision` call in traceSubpath below,
// which places the vertex by delta (Woodcock) tracking with analog throughput, while
// connection edges are weighted by ratio-tracking transmittance. Both are unbiased,
// exactly as the forward tracer does it. The MIS weights then omit the heterogeneous
// distance-pdf / transmittance — a variance-only simplification (the PBRT-v3
// convention), because the balance heuristic is a partition of unity for ANY consistent
// pdfs: the estimator stays unbiased regardless, since only the SAMPLED strategy's
// throughput has to be exact, which analog + ratio tracking guarantee.
//
// THIS COMMENT USED TO SAY heterogeneous media were "NOT handled here (use mode B/P/R
// instead); see the guard in main.cpp" — which was wrong, and wrong in the expensive
// direction, because it cited a guard that says the opposite. `bdptUnsupportedFeature`
// (main.cpp ~14278) refuses exactly one medium class, GRADIENT-INDEX (GRIN), because
// curved Eikonal paths break BDPT's straight-edge assumptions (the geometric term G,
// the area-measure pdf conversion and MIS all assume the connecting segment is a line).
// Everything else — global haze, bounded, density-field — renders here, which is why
// `gallery_rain` (two heterogeneous media, one with a wavelength-dependent droplet
// phase) has mode D as its reference path. Corrected 2026-09-02, after the comment sent
// a reader hunting for a heterogeneous-capable BDPT that was already the one they were
// reading.
//
// Fluorescence, spot and environment lights are genuinely NOT handled; see the guard.
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "scene.h"
#include "camera.h"
#include "hero.h"     // kHeroC / kHeroMax — hero-wavelength bundle sizes
#include "render.h"   // sampleGlossy, Renderer material primitives, clamp01, PI
#include "photonbeams.h"  // BeamMap — mode J (UPBP) merges camera rays against photon beams
#include "beamgather.h"   // gatherPhotonBeamsW — the Beam x Ray estimator, with a MIS hook
#include "parallel.h"          // ft::stopRequested — cooperative -stop inside mode J's beam pass
#include "render_progress.h"   // StageProgress — deposit progress for the live window/title
#include "allocreport.h"       // OOM that names the buffer and the flag that sizes it
#include <thread>
#include <atomic>
#include <chrono>

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
           m.type == MatType::DiffuseTransmit ||  // two-sided Lambertian (finite BSDF both sides)
           m.type == MatType::Hair;               // fiber BCSDF: narrow lobes, but finite
}

// A two-sided (transmissive) connectible material scatters into BOTH hemispheres, so a
// connection edge on the side OPPOSITE the shading normal is legal (transmit lobe).
// Reflect-only materials require the edge on the +ns side; the surface-cosine sign guards
// in connectBDPT must not reject the back hemisphere for these materials — bsdfF (which
// returns 0 for unsupported directions) is the real validity gate, and the geometry term
// uses |cos| accordingly.
inline bool isTwoSidedMat(const Material& m) {
    return m.type == MatType::DiffuseTransmit || m.type == MatType::Hair;
}

// A fiber takes NO shading-normal adjoint correction: that correction rescales a
// projection taken about an interpolated normal, and a strand projects longitudinally
// instead (hair_shade.h). On curve geometry ns == +-ng so it would be 1 regardless;
// the test is what keeps `hair` correct on a triangle mesh with smoothed normals.
inline bool isFiberMat(const Material& m) { return m.type == MatType::Hair; }

// --- Fiber (MatType::Hair) in a bidirectional estimator ---------------------------
//
// BDPT's whole vocabulary is "f, its pdf, and a geometry term carrying cos(ns, w) at
// each endpoint". A fiber's projection is NOT cos(ns, w) — it is the strand's
// longitudinal cosine (see hair_shade.h) — so rather than special-casing the geometry
// term at every one of the ~dozen places it is formed, `bsdfF` returns
//     hairFCos(wi) / |cos(ns, wi)|,
// i.e. it pre-divides by the cosine BDPT is about to multiply back in. The product
// f*G then carries exactly hairFCos, which is the right answer, and every MIS ratio,
// every strategy weight and every connection stays untouched. (This is also how PBRT
// puts hair inside a plain path tracer, for the same reason.) The clamp keeps the
// division finite at a grazing endpoint, where G's own cosine is heading to zero and
// the product is well-behaved regardless.
inline double hairCosGuard(double c) { return (c < 1e-7) ? 1e-7 : c; }

// The BCSDF at a fiber vertex, referenced to the direction the subpath ARRIVED from.
// `wo` is that direction (toward the previous vertex), which is what the impact
// parameter is measured against — see hairShadeAt.
inline HairShade hairAt(const Scene& scene, const Material& m, const Hit& h,
                        double lambda, const Vec3& wo) {
    return hairShadeAt(scene, m, h, lambda, wo);
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
        case MatType::Hair: {
            // Needs the hit (the fiber frame comes from the strand tangent), so a
            // texture-less caller cannot evaluate it — that only happens for vertices
            // that carry no Hit, which are never fiber vertices.
            if (!hitForTex) return 0.0;
            const HairShade hs = hairAt(scene, m, *hitForTex, lambda, wo);
            return hairFCos(hs, wi) / hairCosGuard(std::fabs(cosWi));
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
        case MatType::Hair: {
            // The BCSDF's own sampling density (hair.h), in solid angle — exactly the
            // density hair::sample draws from, so MIS is consistent by construction.
            if (!hitForTex) return 0.0;
            const HairShade hs = hairAt(scene, m, *hitForTex, lambda, wo);
            return hair::pdf(hs.b, hs.woLocal, hair::toLocal(hs.fr, wi));
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

// --- Subpath ray segments, for the mode-J (UPBP) beam merge ---------------------------
//
// A BDPT *connection* joins two sampled vertices, so it only ever needs the vertices. A
// beam *merge* is different in kind: it integrates along a whole ray, pairing it with
// every photon beam whose kernel the ray passes through — so the thing it needs is the
// SEGMENT, which the vertex list does not contain and randomWalk otherwise throws away.
//
// BOTH SIDES OF THE ESTIMATOR NEED IT, which is why this is a PathSeg and not a CamSeg.
// On the camera side a segment is a ray that collects beams (renderRows); on the light
// side it is a ray that *becomes* a beam (traceLightBeamPass) — one long photon beam per
// medium it crosses. The two are recorded by the same three lines in randomWalk because
// they are the same object: a straight span of a subpath, with the throughput that entered
// it. The symmetry is not cosmetic — it is what makes the merge weight a ratio of two
// quantities produced by one walk function from one set of densities (known-issues.md,
// "PHASE 3 ARCHITECTURE DECISION").
//
// Note that a segment is not "the edge between path[i] and path[i+1]". It runs from
// path[i] to the SURFACE that ends the ray, which is strictly further whenever a medium
// collision created path[i+1] in between. That is the point: on the camera side the merge
// is an *alternative* to the free-flight distance sample, so it must see the whole span
// that sample was drawn from, not just the part up to the sample that happened to be
// drawn; on the light side that same whole span is exactly the LONG beam the estimator
// wants (photonbeams.h), whose density carries no distance factor.
//
// For the same reason `beta` here is the throughput arriving at the segment's ORIGIN with
// no free-flight factor in it, and the segment's own transmittance is left to the gather
// (which computes Tr to each beam's own closest-approach point, not to a shared endpoint).
// Both work out because ftrace's media transport is ANALOG — a homogeneous free flight
// draws from the exact transmittance pdf, so beta is unchanged across the event and the
// throughput at the segment origin is the throughput anywhere along it.
struct PathSeg {
    Vec3 o{0, 0, 0};      // segment origin
    Vec3 d{0, 0, 0};      // unit direction
    double tMax = 0.0;    // distance to the surface that ends the segment (large if it escapes)
    double beta = 0.0;    // hero throughput arriving at `o` (see above: no free-flight factor)
    double betaSec[hero::kHeroMax - 1] = {0};
    int nUp = 1;
    double aGlass = 0.0;  // absorption of the dielectric the ray is inside (exp(-a*t) per hit)
    int vert = 0;         // index in the subpath of the vertex this segment leaves
    // Solid-angle density of `d` at the vertex this segment leaves. The merge weight needs
    // it on BOTH sides and cannot recover it from the vertex list: a merge point x is not a
    // vertex, so no Vertex ever recorded the density of the direction that reaches it. On
    // the camera side it converts to the geometric density of x (pdfDir / t_c^2, cosine-free
    // because x is a medium point); on the light side it is the beam's own transverse line
    // density. Zero for a walk that was not asked to record segments.
    double pdfDir = 0.0;
};
using PathSegs = std::vector<PathSeg>;

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
                       const double* betaSecIn, int nUpIn, Escape* esc = nullptr,
                       PathSegs* segs = nullptr) {
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
        // `hide_camera` is primary visibility only, so it applies to exactly one ray in
        // this walk: the first edge of the RADIANCE (camera) subpath. An importance walk
        // starts at a light and never has a camera ray at all.
        Hit h = scene.closestHit(ray, 1e-6, nullptr, /*skipHair=*/false,
                                 /*skipCamHidden=*/(mode == Mode::Radiance && bounces == 0));
        if (h.valid && h.sensorId >= 0) return;      // model-A sensor: not used in BDPT
        double dSurf = h.valid ? h.t : 1e30;

        // Record the segment for the mode-J (UPBP) merge, BEFORE the free-flight draw below
        // consumes it (see PathSeg). Both walks ask for this and mean different things by
        // it: a RADIANCE walk's segments are the camera rays that gather beams, an
        // IMPORTANCE walk's are the spans that get deposited AS beams. `dSurf` is 1e30 on
        // an escaping ray — which the camera side hands to a BVH gather that clips it to
        // the beams that actually exist, and the light side hands to emitBeams, which
        // clamps it to kBeamFarScale * sceneRadius. So neither needs a special case for a
        // ray that leaves the scene through a medium.
        if (segs) {
            PathSeg sg;
            sg.o = ray.o; sg.d = ray.d; sg.tMax = dSurf;
            sg.beta = beta; sg.nUp = nUp;
            for (int i = 0; i + 1 < nUp; ++i) sg.betaSec[i] = betaSec[i];
            sg.aGlass = curAbsorb(lambda);
            sg.vert = (int)path.size() - 1;
            sg.pdfDir = pdfFwd;          // solid-angle density of ray.d (see PathSeg::pdfDir)
            segs->push_back(sg);
        }

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
            double pdfW;
            // phaseSample takes the INCOMING direction (ray.d) directly, so there is no
            // need to reconstruct `wo` from the two vertex positions here — and doing so
            // would be the one place in this walk that could produce a NaN direction.
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
        // Toward the previous vertex. Degeneracy guard: a hit at zero distance from the
        // previous vertex would make this 0/0, and a NaN scatter frame poisons every
        // vertex downstream of it. End the subpath instead — the vertices already stored
        // stay valid and connectible.
        Vec3 dwo = path[prevSurfIdx].p - cur.p;
        if (dot(dwo, dwo) == 0.0) return;
        Vec3 wo = normalize(dwo);
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
            case MatType::Hair: {
                // Fiber BCSDF. NOT delta — the lobes are narrow but finite — so a strand
                // vertex is fully connectible and BDPT's light-tracing strategies can
                // splat a backlit hair directly, which is exactly the transport a
                // unidirectional tracer struggles with.
                const HairShade hs = hairAt(scene, *mp, h, lambda, wo);
                double pdfH = 0.0, fv = 0.0;
                const Vec3 wl = hair::sample(hs.b, hs.woLocal, rng.uniform(), rng.uniform(),
                                             rng.uniform(), rng.uniform(), pdfH, fv);
                if (!(pdfH > 0.0) || !(fv > 0.0)) { terminate = true; break; }
                wi = hair::toWorld(hs.fr, wl);
                pdfW    = pdfH;
                pdfRevW = bsdfPdf(*mp, cur.ns, wi, wo, lambda, scene, &h);
                // f*cos/pdf collapses exactly to T = sum_p A_p (see hair.h): the total
                // Fresnel-and-Beer attenuation of the four lobes.
                const double cosLong = hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
                betaFactor = clamp01(fv * cosLong / pdfH);
                // Per-λ absorption means each secondary gets its OWN fiber evaluated along
                // the hero's sampled direction: f_i*cos/pdf_hero. Unlike the unidirectional
                // tracers (which de-hero at a strand), the bundle survives a fiber here —
                // and a fiber is precisely where the spectral spread is interesting, since
                // sigma_a is what colours the TT/TRT lobes.
                secChromatic = true;
                for (int i = 0; i + 1 < nUp; ++i) {
                    const HairShade hsi = hairAt(scene, *mp, h, hb.lam[i + 1], wo);
                    secF[i] = hairFCos(hsi, wi) / pdfH;
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
        if (terminate || !(mxF > 0.0)) return;

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
        // (A fiber is excluded: the Veach correction rescales a projection taken about an
        // interpolated shading normal, and a strand does not project about its normal at
        // all — its factor is longitudinal. On a curve ns == ±ng so this would evaluate to
        // 1 anyway; the guard is what keeps `hair` honest on a triangle mesh.)
        if (mode == Mode::Importance && !delta && mp->type != MatType::Hair) {
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
        // Spawn the continuation from the correct side of the geometric normal. A fiber
        // leaving on the far side has to clear the strand's own tube instead, or its TT
        // and TRT lobes immediately re-hit the hair they just left (hair_shade.h).
        const bool fiberFar = (mp->type == MatType::Hair) && h.fiberRadius > 0.0 &&
                              dot(wi, cur.ns) < 0.0;
        double sgn = dot(wi, cur.ng) >= 0.0 ? 1.0 : -1.0;
        ray = fiberFar
                  ? Ray{cur.p + normalize(wi) * (2.5 * h.fiberRadius + 1e-9), normalize(wi)}
                  : Ray{cur.p + cur.ng * (sgn * 1e-6), normalize(wi)};
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
                                 Escape* esc = nullptr, PathSegs* segs = nullptr) {
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
                   Mode::Radiance, rng, path, betaSec0, 1, esc, segs);
        return (int)path.size();
    }
    c.p = cam.eye;
    path.push_back(c);
    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
    double cosCam = dot(ray.d, cam.w);
    double pdfDir = cameraPdfDir(cam, cosCam);
    randomWalk(scene, cam, mats, ray, /*beta*/1.0, pdfDir, hb, maxDepth - 1,
               Mode::Radiance, rng, path, betaSec0, hb.C, esc, segs);
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
                             int maxDepth, Pcg32& rng, std::vector<Vertex>& path,
                             PathSegs* segs = nullptr) {
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
    if (!(mxLe > 0.0)) return 0;

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
               Mode::Importance, rng, path, betaWalkSec, hb.C, nullptr, segs);
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
                                Pcg32& rng, std::vector<Vertex>& path,
                                PathSegs* segs = nullptr) {
    const double lambda = hb.lam[0], invPdfLambda = hb.invPdf[0];
    path.clear();
    if (scene.emitters.empty() || scene.totalPower <= 0.0) return 0;
    int ei = scene.selectEmitter(rng);
    const Emitter& em = scene.emitters[ei];
    double pdfChoiceSel = em.power / scene.totalPower;
    if (pdfChoiceSel <= 0.0) return 0;
    if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Sun)
        return deltaLightSubpath(scene, cam, mats, em, pdfChoiceSel, hb, maxDepth, rng, path,
                                segs);
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
               Mode::Importance, rng, path, betaWalkSec, hb.C, nullptr, segs);
    return (int)path.size();
}

// The "a delta pdf is not a density" remap misWeight applies to every FACTOR of a ratio:
// a delta vertex stores 0, and a 0 in a telescoped product would annihilate the whole
// strategy rather than merely excluding it. Free-standing because Phase 3b's accumulators
// are built outside misWeight (in the light pass) and must use the identical convention.
inline double misRemap0(double f) { return f != 0.0 ? f : 1.0; }

// --- The merge technique's density, relative to the connection it competes with -------
//
// This is the one new quantity Phase 3b adds to the balance heuristic, and the whole of
// mode J's weight is built from it. Write the path light-to-camera as x_0..x_{n-1} and let
// x_i be a medium vertex. Two techniques produce that vertex:
//
//   * CONNECTION with i light vertices ("C1"): the camera walk sampled a free flight that
//     landed on x_i, and the light subpath, which ends at x_{i-1}, connected to it.
//   * MERGE at x_i: the light subpath's beam leaves x_{i-1} and passes THROUGH x_i, and the
//     camera ray from x_{i+1} passes through it too; the 1D kernel accepts the pair.
//
// Every other factor of the two path densities is the same object sampled the same way, so
// the ratio collapses to what happens at x_i (known-issues.md has the telescoping proof —
// in particular the transmittance of the connection edge x_{i-1}--x_i cancels EXACTLY, so
// mode D's existing "omit Tr" convention needs no change):
//
//     p_merge / p_C1  =  n_m * 2r * sin(theta_i) * p_L(x_i) / ( sigma_t(x_i) * Tr(e_i) )
//
//   p_L(x_i)  the light side's geometric density of x_i — pdf_omega / dist^2, cosine-free,
//             i.e. exactly what ftrace stores (pdfFwd on the light half of a path, pdfRev on
//             the eye half, since an eye subpath's "forward" is camera-to-light).
//   sigma_t   the extinction the camera's free flight would have used — its distance pdf is
//             sigma_t * Tr, and the merge has no distance pdf at all, which is the entire
//             source of the ratio.
//   Tr(e_i)   the transmittance of the CAMERA-side edge x_i--x_{i+1}, deterministic (trDet).
//   theta_i   the angle between the beam and the camera ray at x_i; the kernel's acceptance
//             volume is 2r * dL * dS * sin(theta), which is where both factors come from.
//   n_m       how many light subpaths the beam map holds — the merge technique's sample
//             count in the multi-sample balance heuristic.
//
// `n_m * 2r` is the same constant at every site, so it is factored out (the caller supplies
// it as `kappa`) and this returns the PRIMED eta. Returning 0 means "no merge technique
// exists here", which is the correct weight contribution for a non-medium vertex.
//
// WHY 2r AND NOT THE KERNEL VALUE K1(d_perp). Same reason VCM uses 1/(pi r^2) rather than
// its own kernel: when this ratio is used to weight a CONNECTION, the merge being weighed
// against is hypothetical and has d_perp = 0, where K1 is at its maximum and nothing like
// what a real merge sees. The balance heuristic needs a consistent partition, not the
// pointwise kernel; 1/(2r) is the mean of K1 over its support, the 1D analogue of VCM's
// uniform acceptance. See known-issues.md.
//
// It sits HERE, above the light pass, because both halves of mode J need it: the light pass
// below folds it into each beam's stored accumulator (BeamMis::sumM), and misWeight further
// down calls it per hypothetical strategy.
inline double mergeEtaPrime(const Scene& scene, const Vec3& pPrev, const Vertex& v,
                            const Vec3& pNext, double pLight, double lambda) {
    if (v.type != VType::Medium) return 0.0;
    if (v.mediumId < 0 || v.mediumId >= (int)scene.media.size()) return 0.0;
    if (!(pLight > 0.0)) return 0.0;
    Vec3 din = v.p - pPrev;
    const double dl = std::sqrt(dot(din, din));
    if (!(dl > 0.0)) return 0.0;
    din = din * (1.0 / dl);
    Vec3 dout = pNext - v.p;
    const double dn = std::sqrt(dot(dout, dout));
    if (!(dn > 0.0)) return 0.0;
    dout = dout * (1.0 / dn);
    // sin of the angle between the beam and the camera ray. The camera ray runs x_{i+1}->x_i,
    // i.e. -dout, but |sin| is unchanged by the flip.
    const double c = dot(din, dout);
    const double s2 = 1.0 - c * c;
    if (!(s2 > 0.0)) return 0.0;                 // exactly collinear: no acceptance volume
    const Medium& md = scene.media[v.mediumId];
    const PatTables tabs = scene.patTables();
    const double sigT = md.sigmaT(lambda) * md.densityAt(v.p, &tabs);
    if (!(sigT > 0.0)) return 0.0;
    const double tr = trDet(scene, v.p, dout, dn, lambda, tabs);
    if (!(tr > 0.0)) return 0.0;
    return std::sqrt(s2) * pLight / (sigT * tr);
}

// --- Mode J's own light-subpath / photon-beam pass (UPBP phase 3a) -------------------
//
// WHY MODE J DOES NOT REUSE tracePhotonPass (the whole point of this function)
// ---------------------------------------------------------------------------
// Phase 1 and 2 borrowed mode M's photon pass, which was fine while the merges were
// unweighted, and is fatal the moment they are not. A balance-heuristic weight is a ratio
// of the densities with which two *competing techniques* would have produced the SAME
// path. mode J's connection half gets its densities from the per-vertex pdfFwd/pdfRev
// bookkeeping above; a beam out of Renderer::tracePhoton carries no densities at all and
// — worse — is not even drawn from the same distributions (its own Russian roulette, its
// own direction sampling, its own spectral bundle). A ratio between the two would be a
// ratio of things that are not the same kind of thing. So mode J traces its own light
// subpaths, with the SAME randomWalk that produces `light[]` in renderRows, and deposits
// its beams from the PathSegs that walk records. Both halves of the estimator then come
// out of one walk function and one set of densities. (known-issues.md, "PHASE 3
// ARCHITECTURE DECISION"; the same reason vcm.h traces its own light subpaths.)
//
// UNITS: the beams land in exactly mode M's normalisation, which is not a coincidence and
// is worth checking rather than assuming. generateLightSubpath's walk throughput is
// Le * cos / (pdfChoice * pdfPos * pdfDir) = Le * area / pdfChoice = the emitter's power
// (its comment says so) — and Renderer::tracePhoton's photon is born carrying
// scene.totalPower. Same quantity, so `nEmitted` counts PATHS here exactly as it counts
// photons there, and beamgather.h's 1/nEmitted is correct untouched.
//
// The beams are LONG: each PathSeg spans from its origin to the surface that ends it, not
// to the sampled collision, and beamgather.h applies Tr analytically to each closest
// approach. That is what makes the beam's density the transverse line density p_L-perp
// with no distance factor — the quantity the Phase-3b merge weight is derived in terms of.
// A short beam (ending at the analog scatter point) is an equally valid estimator but
// would double-count Tr against this gather; see photonbeams.h.
//
// Unlike mode M this pass does MULTIPLE scattering: the walk really scatters in the medium
// (analog free flight) and each subsequent span deposits its own beam, so the beam map
// holds the full multiple-scatter light field rather than mode M's single-scatter one. The
// earlier spans' transmittance is carried stochastically by the free-flight sampling that
// produced the scatter vertex, and the *current* span's analytically by the gather, so the
// two never overlap.
//
// NO RUSSIAN ROULETTE ON THE DEPOSIT, deliberately (see known-issues.md). Mode M thins its
// banks as they fill (BeamBank::cap -> halve()), which leaves each beam with its own
// existence probability; that is invisible to mode M, whose power rescaling makes it
// unbiased, but a merge weight has to know the technique's density, and a per-beam,
// history-dependent keepProb is not something the weight can read. `cap = 0` keeps
// keepProb at exactly 1 for every beam — and, incidentally, draws no RNG in emitBeams, so
// the deposit is deterministic in the path index alone. The beam count is therefore
// `nPaths` x (media crossings per path), and it is reported by the caller.
inline void traceLightBeamPass(const Scene& scene, const Camera& cam, long long nPaths,
                               int nThreads, int maxDepth, bool diffraction,
                               BeamMap& bm, StageProgress* stage = nullptr) {
    if (nThreads < 1) nThreads = 1;
    if (nPaths < 1) nPaths = 1;
    // "JUPBPLGT" — a salt of its own, so mode J's light pass cannot alias mode M's photon
    // stream or its own render-time streams however the counts line up.
    const uint64_t seedBase = 0x4A555042504C4754ULL;
    std::vector<BeamBank> banks((size_t)nThreads);     // cap stays 0: no self-thinning
    // The light half of every merge's MIS weight, one entry per beam, kept in lockstep with
    // `banks[t].beams`. It is NOT a member of BeamBank on purpose: mode M's photon pass
    // shares that type and must not pay 40 bytes a beam for something it never reads, and
    // BeamBank::halve() would silently desynchronise a parallel array (mode J avoids that by
    // setting cap = 0, so banks only ever grow — see the note above).
    std::vector<std::vector<BeamMis>> misBanks((size_t)nThreads);
    std::vector<long long> emitted((size_t)nThreads, 0);
    std::atomic<long long> tracedTotal{0};

    auto worker = [&](int tid) {
        Renderer mats; mats.diffraction = diffraction;
        mats.beamDeposit = &banks[(size_t)tid];
        Pcg32 rng;
        std::vector<Vertex> path;
        PathSegs segs;
        std::vector<BeamMis>& misBank = misBanks[(size_t)tid];
        std::vector<double> accC, accM;                // per-subpath, reused
        const PatTables tabs = scene.patTables();
        const long long lo = nPaths * tid / nThreads, hi = nPaths * (tid + 1) / nThreads;
        long long done = 0;
        for (long long i = lo; i < hi; ++i) {
            // Cooperative `-stop` / Ctrl-C on the same 4096-path cadence as the photon pass.
            if ((done & 0xFFF) == 0) {
                if (done) tracedTotal.fetch_add(0x1000, std::memory_order_relaxed);
                if (ft::stopRequested()) break;
            }
            // Seeded by ABSOLUTE path index, so the map is identical for any thread count.
            seedUnit(rng, seedBase + (uint64_t)i, 0xD1B54A32D192ED03ULL);
            ++done;
            // Single-λ, always. The hero bundle exists to share one ray path across C
            // wavelengths, and a scene with media has a per-λ free flight — renderRows
            // gates hero off for exactly that reason, and this pass only ever runs when
            // there are media.
            HeroBundle hb;
            double pdfLam = 0.0;
            hb.lam[0] = scene.emitSampler.sample(rng, pdfLam);
            if (pdfLam <= 0.0) continue;
            hb.invPdf[0] = scene.invPdfLambda(hb.lam[0]);
            hb.C = 1;
            segs.clear();
            generateLightSubpath(scene, cam, mats, hb, maxDepth + 1, rng, path, &segs);

            // --- The light half of every merge weight this subpath can take part in -------
            // Both accumulators telescope exactly as misWeight's light loop does, one vertex
            // at a time, so a beam leaving y_j can read off the whole prefix in O(1). The
            // recurrences (see BeamMis in photonbeams.h for what they sum):
            //   ratioL(u) = pdfRev(y_u)/pdfFwd(y_u)          [the loop's per-step factor]
            //   accC[j]   = gate(j) + ratioL(j-1) * accC[j-1]
            //   accM[j]   =           ratioL(j-1) * (accM[j-1] + eta'(y_{j-1}))
            // `eta'(y_{j-1})` needs BOTH of y_{j-1}'s neighbours, so it only exists from
            // j = 2 on; the missing j = 1 term is the merge at y_0, which is the light
            // itself and is not a medium vertex in any case.
            const size_t np = path.size();
            accC.assign(np, 0.0);
            accM.assign(np, 0.0);
            for (size_t u = 0; u < np; ++u) {
                const bool dPrev = (u > 0) ? path[u - 1].delta : path[0].isDeltaLight();
                const double gate = (!path[u].delta && !dPrev) ? 1.0 : 0.0;
                if (u == 0) { accC[0] = gate; continue; }
                const double rL = misRemap0(path[u - 1].pdfRev) / misRemap0(path[u - 1].pdfFwd);
                const double eL = (u >= 2)
                    ? mergeEtaPrime(scene, path[u - 2].p, path[u - 1], path[u].p,
                                    path[u - 1].pdfFwd, hb.lam[0])
                    : 0.0;
                accC[u] = gate + rL * accC[u - 1];
                accM[u] = rL * (accM[u - 1] + eL);
            }

            for (const PathSeg& sg : segs) {
                if (!(sg.beta > 0.0)) continue;
                const size_t before = banks[(size_t)tid].beams.size();
                // MedAll, not the emitBeams default MedStraight: nothing about this span is
                // carried stochastically (that is what LONG means), so every medium the
                // lead-in crosses must be charged. See Renderer::emitBeams.
                mats.emitBeams(scene, sg.o, sg.d, sg.tMax, hb.lam[0], sg.beta, sg.aGlass,
                               rng, Renderer::MedAll);
                const size_t after = banks[(size_t)tid].beams.size();
                if (after == before) continue;

                // One template for every beam this segment deposited: they differ only in
                // where along the segment each medium starts (`leadIn`), because everything
                // else here is a property of the vertex the segment LEAVES.
                const size_t j = (size_t)sg.vert;
                const Vertex& y = path[j < np ? j : np - 1];
                BeamMis m;
                m.sumC   = accC[j < np ? j : np - 1];
                m.sumM   = accM[j < np ? j : np - 1];
                m.pdfDir = (float)sg.pdfDir;
                // cos / pdfFwd, the merge-independent half of pdfRev*(y_{s-1})/pdfFwd(y_{s-1}):
                // the gather multiplies in the phase value at x and 1/rho_L^2. The cosine is
                // at y_{s-1} and faces along the beam, so it is 1 for a medium vertex.
                const double cosFac = y.onSurface() ? std::fabs(dot(y.ns, sg.d)) : 1.0;
                m.rCoef  = (float)(cosFac / misRemap0(y.pdfFwd));
                // Is "connect x to y_{s-1}" — the technique this whole ratio is measured
                // against — a legal strategy? x is a medium point and never delta, so only
                // y_{s-1} can veto it.
                m.gateC1 = y.delta ? 0 : 1;
                // The light vertex index, carried purely so the gather can apply the same
                // depth cap the connection loop applies: the merged path has depth j + k + 1.
                m.vert = (unsigned short)(j < np ? j : np - 1);
                // eta' of the merge AT y_{s-1}. Its outgoing direction is this segment's own
                // `d` (the merged path leaves y_{s-1} along the beam), so sin(theta) is known
                // here; only its transmittance, which needs the not-yet-known distance to x,
                // is left to the gather.
                m.etaPrev = 0.0f;
                if (y.type == VType::Medium && j >= 1 && j < np &&
                    y.mediumId >= 0 && y.mediumId < (int)scene.media.size() &&
                    y.pdfFwd > 0.0) {
                    Vec3 din = y.p - path[j - 1].p;
                    const double dl = std::sqrt(dot(din, din));
                    if (dl > 0.0) {
                        din = din * (1.0 / dl);
                        const double c = dot(din, sg.d);
                        const double s2 = 1.0 - c * c;
                        const Medium& md = scene.media[y.mediumId];
                        const double sigT = md.sigmaT(hb.lam[0]) * md.densityAt(y.p, &tabs);
                        if (s2 > 0.0 && sigT > 0.0)
                            m.etaPrev = (float)(std::sqrt(s2) * y.pdfFwd / sigT);
                    }
                }
                for (size_t bi = before; bi < after; ++bi) {
                    BeamMis mm = m;
                    // BeamHit::sBeam is measured from the beam's own clipped origin, not from
                    // y_{s-1}; this is the difference the gather adds back to recover the full
                    // y_{s-1} -> x distance the light-side density is expressed over.
                    mm.leadIn = (float)dot(banks[(size_t)tid].beams[bi].o - sg.o, sg.d);
                    misBank.push_back(mm);
                }
            }
        }
        tracedTotal.fetch_add(done & 0xFFF, std::memory_order_relaxed);
        // Count what was actually emitted, not what was asked for: an early `-stop` that
        // reported the full share would scale the whole map down and darken the merges.
        emitted[(size_t)tid] = done;
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    std::atomic<bool> monitorStop{false};
    std::thread monitor;
    if (stage && stage->report) {
        monitor = std::thread([&] {
            while (!monitorStop.load(std::memory_order_relaxed)) {
                stage->report("tracing light subpaths",
                              tracedTotal.load(std::memory_order_relaxed), nPaths,
                              nullptr, 0.0);
                for (int i = 0; i < 20 && !monitorStop.load(std::memory_order_relaxed); ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    for (auto& th : pool) th.join();
    if (monitor.joinable()) { monitorStop.store(true, std::memory_order_relaxed); monitor.join(); }

    size_t nb = 0;
    for (auto& b : banks) nb += b.size();
    bm.beams.clear();
    bm.mis.clear();
    bm.misIdx.clear();
    ftalloc::reserve(bm.beams, nb, "mode J's photon-beam map", "-n (which sizes it)");
    ftalloc::reserve(bm.mis, nb, "mode J's per-beam MIS partials", "-n (which sizes it)");
    bm.nEmitted = 0;
    for (int t = 0; t < nThreads; ++t) {
        bm.beams.insert(bm.beams.end(), banks[(size_t)t].beams.begin(),
                        banks[(size_t)t].beams.end());
        bm.mis.insert(bm.mis.end(), misBanks[(size_t)t].begin(), misBanks[(size_t)t].end());
        bm.nEmitted += emitted[(size_t)t];
    }
    // The two arrays are built one push apart and can only disagree through a bug; if they
    // ever do, drop the MIS data rather than index it wrongly — BeamMap::misOf then returns
    // null and the gather falls back to weight 1, i.e. mode M's (over-bright but not
    // garbage) estimator, which is a failure the image shows rather than hides.
    if (bm.mis.size() != bm.beams.size()) bm.mis.clear();
    bm.nDeposited = bm.beams.size();
}

// --- Gate (2): an independent, ABSOLUTE-form balance heuristic -------------------
// `misWeight` below is PBRT's RELATIVE form: it never builds a path density, only the
// ratios p_j/p_s, accumulated by telescoping one vertex at a time. That is fast and it
// is what ships — but it is also where the index arithmetic lives, and an off-by-one in
// a ratio loop produces a plausible-looking weight rather than a crash.
//
// The obvious audit — "sum the weights of every strategy and assert 1" — CANNOT FAIL and
// so proves nothing. Within a single call the returned weight is r_c/(1 + sum r) by
// construction, and summing across calls sums weights belonging to *different sampled
// paths*. See the UPBP entry in known-issues.md for the full argument.
//
// So this is the real check: a second implementation that computes each strategy's path
// density OUTRIGHT and divides, sharing no arithmetic with the ratio loops.
//
//     unified path  x[0..n-1], light-to-camera:   x[i] = light[i]        for i < s
//                                                 x[i] = eye[n-1-i]      for i >= s
//     each vertex carries BOTH directions:        pl[i] = density generating x[i]
//                                                         while walking FROM THE LIGHT
//                                                 pc[i] = ... FROM THE CAMERA
//     (which is pdfFwd/pdfRev with the roles swapped on the eye half, because "forward"
//      on an eye subpath means camera-to-light)
//
//     strategy j (j light vertices, n-j camera vertices):
//         p_j = prod_{i<j} pl[i] * prod_{i>=j} pc[i]
//         w_j = p_j / sum_k p_k
//
// Products over a dozen area densities overflow doubles in both directions, so the sum
// is done in logs. The same `remap0` (0 -> 1) is applied per FACTOR, matching what the
// ratio form does per factor, and the same strategies are excluded: strategy j needs the
// edge x[j-1]--x[j] to be a real connection, so it dies if either end is delta, and j==0
// (the eye path landing on the emitter) dies for a delta light.
//
// Called from inside `misWeight` while its ScopedAssigns are still installed, so both
// forms read exactly the same patched densities — the densities are not what is under
// test here, the combination arithmetic is.
namespace misaudit {

inline std::atomic<bool>      enabled{false};
// `-misaudit-poison`: deliberately BREAK the reference weight, so the alarm can be seen
// to fire. A cross-check that has only ever agreed proves nothing on its own — it is
// equally consistent with "both forms are right" and "the check is vacuous" (a mistyped
// tolerance, a hook that never runs, a reference that accidentally recomputes the thing
// it is auditing). This flag injects the exact class of bug the gate exists to catch —
// forgetting that a subpath walked camera-to-light has its two densities swapped
// relative to the unified light-to-camera path — and a run with it MUST report
// disagreements. Kept in the shipping binary, not reverted after one use, so the
// negative control is re-runnable by anyone later (Phase 3b will want it again).
inline std::atomic<bool>      poison{false};
inline std::atomic<long long> nChecked{0};
inline std::atomic<long long> nBad{0};       // relative discrepancy > kTol
inline std::atomic<double>    worst{0.0};
inline std::atomic<int>       worstS{0};
inline std::atomic<int>       worstT{0};
constexpr double kTol = 1e-9;

inline void note(double relErr, int s, int t) {
    if (relErr > kTol) nBad.fetch_add(1, std::memory_order_relaxed);
    // atomic<double> has no fetch_max; CAS until we are no longer the maximum.
    double cur = worst.load(std::memory_order_relaxed);
    while (relErr > cur &&
           !worst.compare_exchange_weak(cur, relErr, std::memory_order_relaxed)) {}
    if (relErr >= worst.load(std::memory_order_relaxed)) {
        worstS.store(s, std::memory_order_relaxed);
        worstT.store(t, std::memory_order_relaxed);
    }
}

inline void reset() {
    nChecked.store(0, std::memory_order_relaxed);
    nBad.store(0, std::memory_order_relaxed);
    worst.store(0.0, std::memory_order_relaxed);
}

} // namespace misaudit

// The reference weight itself. `light`/`eye` must already carry the current strategy's
// patched densities and delta flags (i.e. call from inside misWeight).
//
// Mode J adds the MERGE strategies to the same sum: one per interior medium vertex, each
// weighted by mergeEtaPrime * kappa against the connection strategy at that same vertex.
// `mergeKappa` is 0 in mode D, which deletes them and leaves this function byte-identical.
inline double misWeightReference(const Scene& scene, const std::vector<Vertex>& light,
                                 const std::vector<Vertex>& eye, int s, int t,
                                 double lambda, double mergeKappa) {
    const int n = s + t;
    if (n <= 2) return 1.0;
    auto remap0 = [](double f) { return f != 0.0 ? f : 1.0; };

    // Unified light-to-camera path, each vertex with both directional densities.
    std::vector<double> pl((size_t)n), pc((size_t)n);
    std::vector<char>   dl((size_t)n);
    const bool bad = misaudit::poison.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) {
        const Vertex& v = (i < s) ? light[(size_t)i] : eye[(size_t)(n - 1 - i)];
        // On the light half "forward" already means light-to-camera; on the eye half the
        // subpath was walked camera-to-light, so its forward density is the CAMERA one.
        // (`bad` omits that swap on purpose — see misaudit::poison.)
        const bool swap = (i >= s) && !bad;
        pl[(size_t)i] = swap ? v.pdfRev : v.pdfFwd;
        pc[(size_t)i] = swap ? v.pdfFwd : v.pdfRev;
        dl[(size_t)i] = v.delta ? 1 : 0;
    }
    // The s==0 strategy is the eye path landing on the emitter, which a delta light can
    // never be hit by. x[0] is light[0] when s>0 and the emitter-hit eye vertex when s==0;
    // the latter is a Surface, so isDeltaLight() is correctly false for it.
    const bool deltaLight = (s > 0) ? light[0].isDeltaLight()
                                    : eye[(size_t)(n - 1)].isDeltaLight();

    // log p_j = A[j] + B[j], with A the light-side prefix and B the camera-side suffix.
    std::vector<double> A((size_t)n + 1, 0.0), B((size_t)n + 1, 0.0);
    for (int i = 0; i < n; ++i)
        A[(size_t)i + 1] = A[(size_t)i] + std::log(remap0(pl[(size_t)i]));
    for (int i = n - 1; i >= 0; --i)
        B[(size_t)i] = B[(size_t)i + 1] + std::log(remap0(pc[(size_t)i]));

    // Strategies j = 0..n-1 (t' = n-j >= 1; BDPT never runs t'==0).
    auto allowed = [&](int j) {
        if (dl[(size_t)j]) return false;                     // camera-side end of the edge
        if (j == 0) return !deltaLight;                      // eye path hits the emitter
        return !dl[(size_t)(j - 1)];                         // light-side end of the edge
    };

    const double logPs = A[(size_t)s] + B[(size_t)s];
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        if (!allowed(j)) continue;
        sum += std::exp(A[(size_t)j] + B[(size_t)j] - logPs);
    }
    // Mode J: one merge strategy per interior vertex. p_merge,i = eta_i * p_i with p_i the
    // connection strategy at the same vertex, so it rides the ratio already computed above.
    // No `allowed` test: a merge connects nothing, so a delta neighbour does not kill it —
    // it only needs both subpaths to REACH x_i, which by construction they do.
    if (mergeKappa > 0.0) {
        auto vAt = [&](int i) -> const Vertex& {
            return (i < s) ? light[(size_t)i] : eye[(size_t)(n - 1 - i)];
        };
        for (int i = 1; i <= n - 2; ++i) {
            const double e = mergeEtaPrime(scene, vAt(i - 1).p, vAt(i), vAt(i + 1).p,
                                           pl[(size_t)i], lambda);
            if (e > 0.0)
                sum += e * mergeKappa * std::exp(A[(size_t)i] + B[(size_t)i] - logPs);
        }
    }
    return sum > 0.0 ? 1.0 / sum : 0.0;
}

// --- MIS weight (balance heuristic) ----------------------------------------------
// PBRT's MISWeight: temporarily rewrite the connection vertices' reverse densities
// and delta flags for the current strategy (s,t), then sum the density ratios of all
// other strategies that could have produced the same path. `sampled` is the
// resampled endpoint used when s==1 (light NEE) or t==1 (camera splat). `light`/`eye`
// are mutated in place but restored by the ScopedAssigns before returning.
//
// `mergeKappa` (= n_m * 2r, the merge technique's sample count times the kernel width; 0 in
// every mode but J) adds the beam-merge strategies to the denominator. It has to be here
// rather than applied afterwards: a merge is a technique that competes with THIS connection
// for the same path, so leaving it out would make mode J's connection weights sum with its
// merge weights to more than 1 and brighten every medium.
//
// Index mapping, which is the part worth stating explicitly because it is where an
// off-by-one hides. Writing the unified path x_0..x_{n-1} light-to-camera:
//   * the camera loop's `i` accumulates the ratio for strategy j = n - i, whose vertex is
//     x_{n-i} = eye[i-1]; so the merge site charged at step i is eye[i-1], needing i >= 2
//     for its camera-side neighbour eye[i-2] to exist.
//   * the light loop's `i` accumulates strategy j = i, vertex x_i = light[i]; the merge site
//     is light[i] itself, whose camera-side neighbour is light[i+1] — or `pt` at i == s-1,
//     where the light subpath ends.
//   * strategy j = s (this connection, ratio 1) has vertex x_s = pt, so it too carries a
//     merge site, added once outside both loops.
// Gate (2) already validated the first two mappings for the connection terms.
inline double misWeight(const Scene& scene, const Camera& cam,
                        std::vector<Vertex>& light, std::vector<Vertex>& eye,
                        Vertex& sampled, int s, int t, double lambda,
                        double mergeKappa = 0.0) {
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

    const bool merges = mergeKappa > 0.0;
    double sumRi = 0.0, ri = 1.0;
    for (int i = t - 1; i > 0; --i) {                // hypothetical camera strategies
        ri *= remap0(eye[i].pdfRev) / remap0(eye[i].pdfFwd);
        if (!eye[i].delta && !eye[i - 1].delta) sumRi += ri;
        if (merges && i >= 2) {                      // merge at eye[i-1] (see header note)
            const double e = mergeEtaPrime(scene, eye[i].p, eye[i - 1], eye[i - 2].p,
                                           eye[i - 1].pdfRev, lambda);
            if (e > 0.0) sumRi += ri * e * mergeKappa;
        }
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
        if (merges && i >= 1 && t > 0) {             // merge at light[i]
            const Vec3& pNext = (i + 1 <= s - 1) ? light[i + 1].p : eye[t - 1].p;
            const double e = mergeEtaPrime(scene, light[i - 1].p, light[i], pNext,
                                           light[i].pdfFwd, lambda);
            if (e > 0.0) sumRi += ri * e * mergeKappa;
        }
    }
    // The merge at the connection vertex pt itself. Its own connection strategy IS this
    // one, so the ratio multiplying it is exactly 1.
    if (merges && s >= 1 && t >= 2) {
        const double e = mergeEtaPrime(scene, light[s - 1].p, eye[t - 1], eye[t - 2].p,
                                       eye[t - 1].pdfRev, lambda);
        if (e > 0.0) sumRi += e * mergeKappa;
    }
    const double w = 1.0 / (1.0 + sumRi);
    // Gate (2), off unless `-misaudit`: cross-check the relative form above against the
    // absolute one, here, where the ScopedAssigns are still installed and both therefore
    // see identical densities.
    if (misaudit::enabled.load(std::memory_order_relaxed)) {
        const double ref = misWeightReference(scene, light, eye, s, t, lambda, mergeKappa);
        const double den = std::fabs(w) > std::fabs(ref) ? std::fabs(w) : std::fabs(ref);
        misaudit::nChecked.fetch_add(1, std::memory_order_relaxed);
        misaudit::note(den > 0.0 ? std::fabs(w - ref) / den : 0.0, s, t);
    }
    return w;
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
    // A fiber connecting out the FAR side has to clear the strand's own body: strand
    // radii are microns, so ng*1e-6 lands inside the tube and the connection reports
    // itself occluded, deleting exactly the TT glow that makes light hair look lit.
    if (v.mat && v.mat->type == MatType::Hair && v.hit.fiberRadius > 0.0 &&
        dot(v.ns, dir) < 0.0)
        return v.p + dir * (2.5 * v.hit.fiberRadius + 1e-9);
    return offsetOrigin(v, dir);
}

// How much shorter the shadow ray is than the full endpoint distance, given that
// connOrigin may have moved the start point forward along `dir`. Keeps the ray from
// overshooting into the light it is testing visibility to.
inline double connShorten(const Vertex& v, const Vec3& dir, double eps) {
    if (v.type != VType::Medium && v.mat && v.mat->type == MatType::Hair &&
        v.hit.fiberRadius > 0.0 && dot(v.ns, dir) < 0.0)
        return 2.5 * v.hit.fiberRadius + 1e-9 + eps;
    return eps;
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
                          double* Lsec, int& nUpConn, double mergeKappa = 0.0) {
    const double lambda = hb.lam[0], invPdfLambda = hb.invPdf[0];
    isSplat = false;
    nUpConn = 0;                    // set to the real width only once a contribution exists
    // PBRT's "ignore invalid connections related to infinite area lights" guard. It tests
    // the vertex TYPE, not `isLightVertex()`, and the difference is load-bearing: a
    // VType::Light eye vertex is a fictitious endpoint standing in for an infinite light,
    // which has no surface and nothing to connect to; an EMISSIVE SURFACE is an ordinary
    // Surface vertex that happens to carry an `emit` slot, and it still has a BSDF.
    //
    // Testing isLightVertex() here (as this did) rejected every s>=1 strategy that ended on
    // an emissive surface, i.e. deleted ALL DIRECT LIGHTING ON ANY SURFACE THAT ALSO GLOWS.
    // Measured on scraps/mini_grid.ftsl: four 5%-albedo tiles differing only in their slots
    // rendered 59 / 61 / (clipped) / 14 in mode D, where mode R — which has no such guard —
    // gave 59 / 61 / 61 / 61. The 14 is the surviving indirect; the direct sun and sky were
    // gone. gallery_rain's glowing floor grid is exactly this material, which is why its
    // 5%-albedo body read near-black in D and mid-grey in every other mode.
    if (t > 1 && s != 0 && eye[t - 1].type == VType::Light) return 0.0;

    double L = 0.0;
    int nUp = 1;                    // live wavelengths for THIS connection (set per branch)
    Vertex sampled;

    if (s == 0) {
        // Pure eye path: contributes iff its last vertex is emissive.
        if (t < 2) return 0.0;
        const Vertex& pt = eye[t - 1];
        if (!pt.isLightVertex()) return 0.0;
        Vec3 dwo = eye[t - 2].p - pt.p;          // degeneracy guard: normalize(0) is NaN
        if (dot(dwo, dwo) == 0.0) return 0.0;
        Vec3 wo = normalize(dwo);
        double Le = pt.Le(wo, lambda, invPdfLambda);
        nUp = pt.nUp;
        // The hero may legitimately be black where a secondary is not, so the early-out
        // tests the max over the live wavelengths (identical to `Le <= 0` when nUp==1).
        double LeSec[hero::kHeroMax - 1] = {0}, mxLe = Le;
        for (int i = 0; i + 1 < nUp; ++i) {
            LeSec[i] = pt.Le(wo, hb.lam[i + 1], hb.invPdf[i + 1]);
            if (LeSec[i] > mxLe) mxLe = LeSec[i];
        }
        if (!(mxLe > 0.0)) return 0.0;
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
        // Degeneracy guard on the incoming edge, for the reason given at the interior
        // connection below: coincident vertices make `normalize` 0/0 and the NaN then
        // survives every `<= 0` test on its way to the film.
        Vec3 dwo = light[s - 2].p - qs.p;
        if (dot(dwo, dwo) == 0.0) return 0.0;
        Vec3 wo = normalize(dwo);
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
            const double adj = (isFiberMat(*qs.mat) ? 1.0
                                                    : shadingAdjointCorr(wo, wcam, qs.ns, ngo)) * stG;
            f *= adj;
            for (int i = 0; i + 1 < nUp; ++i)
                fSec[i] = bsdfF(*qs.mat, qs.ns, wo, wcam, hb.lam[i + 1], scene, &qs.hit) * adj;
        }
        {   // max over live wavelengths (identical to `f <= 0` when nUp==1)
            double mxF = f;
            for (int i = 0; i + 1 < nUp; ++i) if (fSec[i] > mxF) mxF = fSec[i];
            if (!(mxF > 0.0)) return 0.0;
        }
        // The t=1 strategy: this segment runs from a LIGHT-subpath vertex to the camera, so
        // it is the camera leg and `hide_camera` applies to it (see Scene::occluded).
        if (scene.occluded(connOrigin(qs, wcam), wcam, dist - connShorten(qs, wcam, 2e-6),
                           1e-6, /*camLeg=*/true)) return 0.0;
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
        Vec3 dwo = eye[t - 2].p - pt.p;          // degeneracy guard: normalize(0) is NaN
        if (dot(dwo, dwo) == 0.0) return 0.0;
        Vec3 wo = normalize(dwo);
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
            if (!(mxF > 0.0)) return 0.0;
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
            if (!(mxLe > 0.0)) return 0.0;
        }
        if (scene.occluded(connOrigin(pt, wi), wi, dist - connShorten(pt, wi, occlEps))) return 0.0;
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
        // The two INCOMING edges need the same degeneracy guard the connection edge just
        // got, because `normalize` of a zero vector is 0/0 = NaN — and a NaN scatter
        // direction propagates straight through `mxE/mxL <= 0` (NaN compares false against
        // everything) onto the film as an undefined pixel. Coincident vertices are exactly
        // what a zero-length free flight produces (see Pcg32::uniformOpen, now fixed at the
        // source), but a grazing hit or a degenerate emitter can do it too, so the guard
        // stays. vertexPdf already rejects both edges the same way, so the pdf side and the
        // throughput side now agree: this strategy contributes nothing, not something
        // undefined.
        Vec3 dwE = eye[t - 2].p - pt.p, dwL = light[s - 2].p - qs.p;
        if (dot(dwE, dwE) == 0.0 || dot(dwL, dwL) == 0.0) return 0.0;
        Vec3 woE = normalize(dwE);
        Vec3 woL = normalize(dwL);
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
            const double adjL = isFiberMat(*qs.mat)
                                    ? 1.0
                                    : shadingAdjointCorr(woL, w * -1.0, qs.ns, ngoL);
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
            // Negated form, NOT `mxE <= 0.0`: NaN compares false against every operand, so
            // `<= 0.0` lets an undefined scatter value through while `!(mxE > 0.0)` rejects
            // it. Identical for every finite value, so nothing else changes.
            if (!(mxE > 0.0) || !(mxL > 0.0)) return 0.0;
        }
        // Both ENDS can be fibers here, and a connection arriving at a strand from its far
        // side would clip the tube just short of the vertex, so stop the shadow ray early at
        // that end too. Adds an exact 0.0 for every non-fiber pair.
        if (scene.occluded(connOrigin(pt, w), w,
                           dist - connShorten(pt, w, 2e-6) - connShorten(qs, w * -1.0, 0.0)))
            return 0.0;
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
    if (!(mx > 0.0)) return 0.0;        // negated: also rejects NaN (see the mxE/mxL note)
    const double mis = misWeight(scene, cam, light, eye, sampled, s, t, lambda, mergeKappa);
    for (int i = 0; i + 1 < nUp; ++i) Lsec[i] *= mis;
    nUpConn = nUp;
    return L * mis;
}

// --- The merge weight itself (mode J) ------------------------------------------------
//
// One of these is built per CAMERA SEGMENT and handed to gatherPhotonBeamsW, which calls it
// once per beam hit. It returns the balance-heuristic weight of "merge THIS beam here"
// against every other technique that could have produced the same path.
//
// SHAPE. The merged path is y_0..y_{s-1}, x, eye[k], eye[k-1], ..., eye[0]. Its reference
// technique is the connection "C1" that would have made x the LAST CAMERA vertex and joined
// it to y_{s-1} — strategy j = s. Every term below is a density RATIO against that one, so
// the weight is etaS / (sum of all of them), and no absolute path density is ever formed.
//
// The two halves reach it by different routes and that is the whole trick: the camera half
// is explicit (`eye[]` is right here, so misWeight's own loops can be replayed over it and
// their results cached per segment), while the light half is long gone — it was summed into
// BeamMis while the beam was deposited. Only three densities in the merged path differ from
// the recorded ones, because the beam leaves y_{s-1} along exactly the direction the light
// walk continued on: pdfRev(y_{s-1}), the pair at x, and pdfRev(eye[k]). Everything else is
// what the two walks already stored, which is why the accumulators are legal at all.
//
// SPECTRAL MISMATCH (documented approximation). The beam carries its own wavelength and the
// camera path another; a weight built from both is not a function of one path in one
// spectrum. Camera-side quantities use the camera's lambda, light-side ones the beam's, and
// the phase value at x — computed once by the gather — is used for both densities there.
// This is the standard spectral-photon-mapping fudge (mode M and mode U's merges make the
// same one); it perturbs the weight, never the estimator's support, so it costs quality and
// not unbiasedness in the achromatic case, and see known-issues.md for the chromatic one.
//
// A DELTA LIGHT-SIDE DENSITY RETURNS 0, i.e. drops the merge. If the light walk left
// y_{s-1} by a SPECULAR bounce then p_L(x) is a Dirac and `lm->pdfDir` is 0 (randomWalk
// stores 0 for a delta continuation). misWeight's own merge terms vanish in exactly the
// same case, so the partition of unity still holds and nothing is double- or under-counted
// — the path is simply left to the connection strategies. It is a variance loss, not a
// bias, and it is logged in known-issues.md.
struct BeamMergeWeight {
    const Scene*   scene = nullptr;
    const BeamMap* bm    = nullptr;
    const PathSeg* sg    = nullptr;
    double lamCam = 0.0;
    double kappa  = 0.0;      // n_m * 2r: the merge technique's sample count times the kernel
    // Per-segment camera-side constants (see BdptRenderer::renderRows, where they are built).
    double gateS1     = 0.0;  // is the reference connection C1 legal? (eye[k] not delta)
    double cosFacK    = 1.0;  // projected cosine at eye[k] along the segment; 1 off a surface
    double invPdfFwdK = 1.0;  // 1 / remap0(eye[k].pdfFwd)
    double etaKCoef   = 0.0;  // sin(theta_k) / (sigma_t(eye[k]) * Tr~(eye[k] -> eye[k-1]))
    double segSumC    = 0.0;  // camera-side connection accumulator from eye[k] inward
    double segSumM    = 0.0;  // camera-side merge accumulator, kappa factored out
    int    camVert    = 0;    // k: the camera subpath index of the vertex this segment leaves
    int    maxDepth   = 0;    // the same cap the connection loop applies (see below)
    // Per-ray transmittance data, hoisted out of the per-hit path. `camTr` is built once for
    // this camera segment (see TrRay: a dense medium hands one segment hundreds of hits, and
    // trDet would re-clip and re-evaluate sigma_t for every one of them); `tabs` is the
    // gather's own PatTables, so the light-side march stops rebuilding one per hit too.
    const TrRay*     camTr = nullptr;
    const PatTables* tabs  = nullptr;

    double operator()(const BeamHit& bh, const PhotonBeam& b, double dens, double phase) const {
        const BeamMis* lm = bm->misOf(bh.idx);
        if (!lm) return 1.0;                  // no MIS data: mode M's raw estimator
        // THE DEPTH CAP. The merged path has s = j+1 light vertices and t = k+2 camera ones,
        // hence depth = s+t-2 = j+k+1. renderRows' connection loop refuses depth > maxDepth,
        // so a merge past the cap would contribute a path length mode D never builds — energy
        // with nothing to MIS against, which is exactly what it looked like: 1.6x too bright
        // on an optically thick medium, and worse the more beams were emitted (as kappa grows
        // the weight of an uncontested technique tends to 1). Every strategy in this
        // denominator describes the SAME path with the same n, so a single gate here is the
        // whole fix: past the cap the merge does not exist, and no other term needs adjusting.
        if ((int)lm->vert + camVert + 1 > maxDepth) return 0.0;
        const double tc = bh.tCam;
        const double rhoL = (double)lm->leadIn + bh.sBeam;   // y_{s-1} -> x, not b.o -> x
        if (!(tc > 0.0) || !(rhoL > 0.0)) return 0.0;
        const double invR2 = 1.0 / (rhoL * rhoL), invT2 = 1.0 / (tc * tc);
        // The pair of geometric densities AT x. Both are cosine-free: x is a medium point,
        // so convertDensity applies no projected cosine to it.
        const double gL = (double)lm->pdfDir * invR2;        // light side  (p_L-perp)
        const double gC = sg->pdfDir * invT2;                // camera side (the free flight)
        if (!(gL > 0.0)) return 0.0;                         // delta light-side density
        const double sigT = scene->media[b.med].sigmaT(lamCam) * dens;
        if (!(sigT > 0.0)) return 0.0;
        const double trC = camTr->at(tc);                             // Tr~(x -> eye[k])
        if (!(trC > 0.0)) return 0.0;
        // eta of THIS merge against C1 — the numerator of the weight, and a term of its
        // denominator (a technique competes with itself at ratio exactly its own).
        const double etaS = kappa * bh.sinT * gL / (sigT * trC);
        if (!(etaS > 0.0)) return 0.0;

        // pdfRev(y_{s-1}): the density of the last light vertex seen from x. The phase value
        // at x is its direction density (HG samples proportionally to its own value), and
        // BeamMis::rCoef carries the cosine and the 1/pdfFwd that turn it into the ratio the
        // light loop accumulates.
        const double R  = phase * (double)lm->rCoef * invR2;
        // pdfRev(x)/pdfFwd(x): the camera loop's first step. remap0 on BOTH, exactly as
        // misWeight does, so a delta camera continuation cancels instead of annihilating.
        const double C1 = misRemap0(gL) / misRemap0(gC);
        // pdfRev(eye[k]) in the merged path, and the camera loop's second step.
        const double pdfRevK = phase * cosFacK * invT2;
        const double C2   = pdfRevK * invPdfFwdK;
        const double etaK = kappa * etaKCoef * pdfRevK;      // merge AT eye[k]
        // The merge AT y_{s-1}. Its sin(theta) and p_L were known when the beam was
        // deposited (its outgoing direction IS the beam); only the transmittance over the
        // now-known y_{s-1} -> x span is left.
        double etaPrevTerm = 0.0;
        if (lm->etaPrev > 0.0f) {
            const Vec3 yPrev = b.o - b.d * (double)lm->leadIn;
            const double trP = trDet(*scene, yPrev, b.d, rhoL, (double)b.lambda, *tabs);
            if (trP > 0.0) etaPrevTerm = kappa * (double)lm->etaPrev / trP;
        }
        const double den = (double)lm->gateC1                    // C1 itself (ratio 1)
                         + R * lm->sumC                          // light-side connections
                         + gateS1 * C1                           // the t-1 camera connection
                         + C1 * C2 * segSumC                     // camera-side connections
                         + R * (kappa * lm->sumM + etaPrevTerm)  // light-side merges
                         + etaS                                  // this merge
                         + C1 * etaK                             // merge at eye[k]
                         + C1 * C2 * kappa * segSumM;            // camera-side merges
        if (!(den > 0.0)) return 0.0;
        return etaS / den;
    }
};

// --- Renderer --------------------------------------------------------------------
// Renders pixel rows [y0,y1). t>=2 (camera-image) contributions land on the current
// pixel in `camFilm`; t==1 (light-image) splats land on `splatFilm` at the projected
// raster position. The caller normalises camFilm by spp and splatFilm by the total
// light-subpath count (W*H*spp), matching mode B's absolute-radiance convention.
struct BdptRenderer {
    int maxDepth = 8;          // maximum path length in edges (connection cost ~ depth^2)
    bool diffraction = true;   // mirrors Renderer::diffraction for MatType::Grating
    int heroC = 1;             // wavelengths bundled per path pair (1 = plain single-λ)

    // MODE J (UPBP): the view-independent photon-beam cache to merge camera rays against.
    // Null (mode D) means connections only, and every line below that touches `beams` is
    // then dead — which is the point, and is gate (1) of the UPBP validation plan in
    // known-issues.md: an absent (or empty) beam map must leave mode D BIT-IDENTICAL.
    const BeamMap* beams = nullptr;

    // `sampleBase` = absolute index of the first sample rendered here; each
    // (pixel, absolute sample) seeds its own stream via seedUnit(), so the
    // realization is chunk-split / banding / thread-count independent (see
    // BackwardRenderer::renderRows).
    void renderRows(const Scene& scene, const Camera& cam, Film& camFilm, Film& splatFilm,
                    int y0, int y1, long long spp, unsigned long long sampleBase) const {
        Renderer mats; mats.diffraction = diffraction;
        std::vector<Vertex> eye, light;
        // Mode J (UPBP) only. Hoisted out of the pixel loop so its capacity is reached once
        // per thread rather than reallocated per sample, exactly like `eye`/`light`.
        PathSegs segs;
        // Camera-side MIS accumulators, one entry per eye vertex (Phase 3b). Same hoist.
        std::vector<double> segSumC, segSumM;
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
        // Mode J. `beams` is null in mode D, and an empty map is passed as null by the
        // dispatcher rather than as an empty map, so this single test is the whole gate:
        // when it is false the camera walk is not even asked to record its segments and the
        // render is mode D down to the RNG stream (validation gate 1).
        const bool mergeOn = beams && !beams->empty() && beams->nEmitted > 0;
        // The merge technique's constant: its sample count (one beam map per frame, built
        // from `nEmitted` light subpaths) times the 1D kernel's full width. Zero unless the
        // map actually carries MIS partials, and that zero is what makes every merge term
        // below — and inside misWeight — disappear, leaving mode D's arithmetic untouched.
        // Weighted and unweighted merges must move together: weighting the connections down
        // while the merges are still raw would darken the volume as surely as the reverse
        // brightens it, so ONE flag gates both.
        const double mergeKappa = (mergeOn && !beams->mis.empty())
                                ? (double)beams->nEmitted * 2.0 * beams->radRef() : 0.0;
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
                    if (mergeOn) segs.clear();
                    int nE = generateCameraSubpath(scene, cam, mats, px, py, hb,
                                                   maxDepth + 1, rng, eye,
                                                   hasSun ? &esc : nullptr,
                                                   mergeOn ? &segs : nullptr);
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
                                                   rng, spx, spy, isSplat, Lsec, nUpConn,
                                                   mergeKappa);
                            if (nUpConn <= 0) continue;
                            double mx = c;
                            for (int i = 0; i + 1 < nUpConn; ++i)
                                if (Lsec[i] > mx) mx = Lsec[i];
                            if (!(mx > 0.0)) continue;   // negated: also drops NaN (see connectBDPT)
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

                    // ---- MERGES (mode J / UPBP) ---------------------------------------
                    // The other half of the estimator: every photon beam whose kernel this
                    // camera subpath passed through. Unlike a connection this needs no
                    // light subpath at all — the beam map IS a cache of light subpaths,
                    // traced once for the whole frame — which is exactly why it reaches
                    // where a connection cannot: deep inside a thick medium the camera's
                    // own free-flight sampling essentially never lands on the scattering
                    // point a connection would have to be made from, whereas a beam is a
                    // whole LINE of deposited power and is hit by merely passing near it.
                    //
                    // The gather returns XYZ built at the BEAMS' wavelengths, not at this
                    // sample's hero λ — the standard spectral-photon-mapping estimate, and
                    // the same choice mode U makes for its merges (a merge gathers photons
                    // from other paths, so there is no shared wavelength to be exact in).
                    // `sg.beta` is this sample's own hero throughput to the segment origin,
                    // which is a scalar and multiplies the XYZ triple directly.
                    //
                    // Placed after every connection deliberately. The gather draws from
                    // `rng` (ratio-tracked transmittance in a heterogeneous medium), and
                    // running it here means those draws cannot shift the connection half of
                    // the SAME sample. They cannot shift the next sample either, because the
                    // stream is re-seeded per (pixel, sample) at the top of this loop — so
                    // "turn the merges off and mode J is mode D bit-for-bit" survives even
                    // though the two halves share a generator.
                    if (mergeOn) {
                        // The camera half of every merge weight, replayed ONCE for the whole
                        // subpath. misWeight's camera loop telescopes inward from the merge
                        // point; everything it accumulates strictly camera-side of eye[k] is
                        // independent of where along the segment a beam is hit, so it is
                        // summed here and read off per segment. (The recurrences mirror the
                        // light-side ones in traceLightBeamPass; see BeamMergeWeight.)
                        const double lamCam = hb.lam[0];
                        const PatTables tabs = scene.patTables();
                        if (mergeKappa > 0.0) {
                            segSumC.assign((size_t)nE, 0.0);
                            segSumM.assign((size_t)nE, 0.0);
                            for (int k = 1; k < nE; ++k) {
                                const double gate = (!eye[(size_t)k].delta &&
                                                     !eye[(size_t)k - 1].delta) ? 1.0 : 0.0;
                                // eta' of the merge AT eye[k-1]; needs both its neighbours,
                                // so it starts at k = 2. Its pdfRev is the RECORDED one:
                                // moving the light-side neighbour of eye[k] along the same
                                // ray does not change the direction arriving at eye[k-1].
                                const double eK = (k >= 2)
                                    ? mergeEtaPrime(scene, eye[(size_t)k].p, eye[(size_t)k - 1],
                                                    eye[(size_t)k - 2].p,
                                                    eye[(size_t)k - 1].pdfRev, lamCam)
                                    : 0.0;
                                double carry = 0.0, carryM = 0.0;
                                if (k >= 2) {
                                    // No ratio exists at the camera vertex itself, which is
                                    // why the recurrence starts one step in.
                                    const double rC = misRemap0(eye[(size_t)k - 1].pdfRev) /
                                                      misRemap0(eye[(size_t)k - 1].pdfFwd);
                                    carry  = rC * segSumC[(size_t)k - 1];
                                    carryM = rC * segSumM[(size_t)k - 1];
                                }
                                segSumC[(size_t)k] = gate + carry;
                                segSumM[(size_t)k] = eK + carryM;
                            }
                        }
                        TrRay camTr;
                        for (const PathSeg& sg : segs) {
                            if (!(sg.beta > 0.0)) continue;
                            BeamMergeWeight w1;
                            w1.scene = &scene; w1.bm = beams; w1.sg = &sg;
                            w1.lamCam = lamCam; w1.kappa = mergeKappa;
                            w1.tabs = &tabs;
                            // Hoisted out of the per-hit weight: the ray/bounds clip and the
                            // spectral sigma_t lookup are constants of the SEGMENT, and a
                            // dense medium hands one segment hundreds of hits. Built
                            // unconditionally so `w1.camTr` is never null — cheap (one clip
                            // per medium) and the alternative is a dangling read the day the
                            // `mergeKappa == 0 => misOf() is null` coupling stops holding.
                            camTr.build(scene, sg.o, sg.d, sg.tMax, lamCam, tabs);
                            w1.camTr = &camTr;
                            if (mergeKappa > 0.0) {
                                const size_t k = (size_t)sg.vert;
                                const Vertex& vk = eye[k < (size_t)nE ? k : (size_t)nE - 1];
                                w1.gateS1     = vk.delta ? 0.0 : 1.0;
                                w1.cosFacK    = vk.onSurface()
                                              ? std::fabs(dot(vk.ns, sg.d)) : 1.0;
                                w1.invPdfFwdK = 1.0 / misRemap0(vk.pdfFwd);
                                w1.segSumC    = segSumC[k < (size_t)nE ? k : (size_t)nE - 1];
                                w1.segSumM    = segSumM[k < (size_t)nE ? k : (size_t)nE - 1];
                                // The depth cap, the same one the connection loop above
                                // applies: a merge at light vertex j on this segment makes a
                                // path of depth j + k + 1.
                                w1.camVert    = (int)(k < (size_t)nE ? k : (size_t)nE - 1);
                                w1.maxDepth   = maxDepth;
                                // The merge AT eye[k]. Its light-side incoming direction is
                                // -sg.d whatever x turns out to be, so sin(theta) and the
                                // whole coefficient are merge-point INDEPENDENT and belong
                                // here rather than in the per-hit weight.
                                if (k >= 1 && k < (size_t)nE &&
                                    vk.type == VType::Medium && vk.mediumId >= 0 &&
                                    vk.mediumId < (int)scene.media.size()) {
                                    Vec3 dp = eye[k - 1].p - vk.p;
                                    const double dn = std::sqrt(dot(dp, dp));
                                    if (dn > 0.0) {
                                        dp = dp * (1.0 / dn);
                                        const double c = dot(sg.d, dp);
                                        const double s2 = 1.0 - c * c;
                                        const Medium& md = scene.media[vk.mediumId];
                                        const double sT = md.sigmaT(lamCam) *
                                                          md.densityAt(vk.p, &tabs);
                                        const double tr = trDet(scene, vk.p, dp, dn,
                                                                lamCam, tabs);
                                        if (s2 > 0.0 && sT > 0.0 && tr > 0.0)
                                            w1.etaKCoef = std::sqrt(s2) / (sT * tr);
                                    }
                                }
                            }
                            Vec3 m = gatherPhotonBeamsW(scene, mats, *beams, sg.o, sg.d,
                                                        sg.tMax, sg.aGlass, rng, w1);
                            if (m.x != 0.0 || m.y != 0.0 || m.z != 0.0)
                                camFilm.add(px, py, m * sg.beta);
                        }
                    }
                }
            }
    }
};

} // namespace bdpt
