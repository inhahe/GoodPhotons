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
#include "bsdf_eval.h"    // bsdfF / bsdfPdf — the evaluable half, shared with backward.h
#include "photonbeams.h"  // BeamMap — mode J (UPBP) merges camera rays against photon beams
#include "surfmerge.h"    // SurfMap — mode J's OTHER merge kind, the point x point (VCM's VM)
#include "beamgather.h"   // gatherPhotonBeamsW — the Beam x Ray estimator, with a MIS hook
#include "parallel.h"          // ft::stopRequested — cooperative -stop inside mode J's beam pass
#include "render_progress.h"   // StageProgress — deposit progress for the live window/title
#include "allocreport.h"       // OOM that names the buffer and the flag that sizes it
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>

namespace bdpt {

// Diagnostic tallies for the mode-J merge path, printed at exit when FTRACE_J_HALF is set.
// Every counter is behind that flag, so a normal render pays nothing: the atomics would
// otherwise be incremented once per beam hit, and a dense medium hands one camera segment
// hundreds of those.
struct JDiag {
    std::atomic<long long> segs{0}, hits{0}, capped{0}, zeroed{0};
    ~JDiag() {
        const long long s = segs.load(), h = hits.load();
        if (s || h)
            std::fprintf(stderr, "[jdiag] camera segments gathered along: %lld | beam hits: %lld"
                                 " | depth-capped: %lld | weight==0: %lld\n",
                         s, h, capped.load(), zeroed.load());
    }
};
inline JDiag& jDiag() { static JDiag d; return d; }

// Diagnostic knob for mode J: `FTRACE_J_HALF=connections` or `=merges` renders only one
// half of the UPBP estimator, MIS weights and all. Neither half is a correct image on its
// own, but their SUM is exactly the full mode-J image (the RNG is re-seeded per
// (pixel, sample), so suppressing one half cannot perturb the other's stream), which makes
// it possible to attribute a whole-frame energy error to the connections or to the merges
// with two renders. Off (0) unless the variable is set; read once.
inline int jHalfMode() {
    static const int v = [] {
        const char* e = std::getenv("FTRACE_J_HALF");
        if (!e) return 0;
        if (!std::strcmp(e, "connections")) return 1;
        if (!std::strcmp(e, "merges")) return 2;
        if (!std::strcmp(e, "merges-raw")) return 3;
        return 0;
    }();
    return v;
}

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
    // SPECTRAL-CLAIM STATE (see BeamSpectral below). True while nothing wavelength-dependent
    // has happened to `beta` since the subpath left the emitter, so the beam this segment
    // deposits may be re-expressed at the emitter's own spectral density — folded to its
    // SPD-mean CIE (`-beamachro`), or spread over a stratified bundle (`-beamspec`), or at
    // worst moved to a single wavelength drawn from the right density instead of the wrong
    // one. Carried on the SEGMENT rather than read at the deposit site because the deposit
    // happens later, in the mode-J beam pass, long after the walk that established the claim
    // has moved on. See randomWalk for the update rule.
    bool achro = false;
    // THE SPECTRAL CLAIM'S PAYLOAD, snapshotted alongside `achro` and for the same reason:
    // the deposit happens in the beam pass, long after the walk that accumulated these.
    //
    //  * `fCie` is the `-beamachro` colour — sum_k foldCie[k] * T(foldLam[k]) over the
    //    emitter's quadrature table (scene.h, Emitter::foldCie). A claim that has picked up
    //    no spectral factor leaves it at the emitter's plain cieMean, so a neutral scene
    //    deposits bit-for-bit the beams it deposited before this existed.
    //  * `fW[i]` is T(bs.lam[i]) / T(lambda_walk) — bundle member i's accumulated spectral
    //    throughput RELATIVE TO THE WAVELENGTH THE WALK ACTUALLY USED. The record's hero is
    //    bs.lam[0], not the walk's lambda (see BeamSpectral), so unlike render.h's `specW`
    //    this covers member 0 too: an UNFOLDED deposit is beta * scale * fW[0] with
    //    per-member weights fW[i] / fW[0]. A FOLDED deposit (fCie in use) is beta * scale
    //    at the walk's own wavelength -- fCie's ratios are relative to that, not to
    //    member 0 (see depositSeg; getting this wrong was the mode-J X excess).
    //
    // Both stay at 1 / cieMean until something wavelength-dependent happens, which before
    // 0.257.0 was the only state that could reach a deposit at all.
    Vec3   fCie{0, 0, 0};
    double fW[kBeamSpecMax] = {1.0, 1.0, 1.0, 1.0};
};
using PathSegs = std::vector<PathSeg>;

// THE EMITTER-SIDE SPECTRAL CLAIM a mode-J light subpath hands to its beam deposits.
//
// Built once at the subpath's birth and read back unchanged at every deposit that subpath
// makes; `PathSeg::achro` says whether the claim was still alive where a given segment
// began. It exists because the two beam-depositing modes draw λ from different densities
// and the beam record cannot express the difference:
//
//   * render.h's photon (mode M) draws λ from the CHOSEN emitter's own SPD. Its pdf is
//     spd/∫spd, so spd(λ)/p(λ) is that emitter's SPD integral for every λ alike and the
//     photon's power comes out λ-free. THAT is what lets a beam store several wavelengths
//     with no per-wavelength weight, and what makes the mean-CIE fold exact.
//   * bdpt.h (mode J) draws λ from the SCENE-WIDE `emitSampler`, before the emitter is even
//     chosen, because the camera and light subpaths must agree on a hero wavelength. So a
//     mode-J subpath's beta carries spd_em(λ)/p_comb(λ), which genuinely varies with λ
//     whenever the scene mixes emitters of different colour — gallery_rain has five.
//
// `scale` converts the second into the first: multiplying beta by it replaces
// spd_em(λ)/p_comb(λ) with ∫spd_em, giving exactly the power render.h's photon would have
// carried. `lam[0..nLam)` are C wavelengths drawn stratified from the emitter's OWN SPD,
// replacing the hero for the deposit — the hero comes from the wrong density and so cannot
// be a bundle member (weighting it like the others would bias the estimate toward
// ∫p_comb·f instead of ∫p_em·f). Both together make mode J's deposit arithmetically
// identical to mode M's, which is what lets it use mode M's bundle AND its fold.
//
// Note this is a strict improvement even at `-beamspec 1 -beamachro off`, where it reduces
// to "deposit at one wavelength drawn from the emitter's SPD rather than from the mixture":
// the estimator loses the spd_em(λ)/p_comb(λ) weight ratio, which in a five-emitter scene
// is itself a large source of beam-to-beam variance.
struct BeamSpectral {
    Vec3   cie{0, 0, 0};              // Emitter::cieMean — the `-beamachro` fold colour
    double scale = 1.0;               // ∫spd_em / (spdFn(hero) · invPdfLambda(hero))
    double lam[kBeamSpecMax] = {0};   // C wavelengths ~ this emitter's own SPD
    int    nLam = 0;
    bool   ok = false;                // false: deposit exactly as a pre-0.251.0 build did
    // Index of the emitter above, for the GATHER-TIME spectral fold (0.256.0, Scene::BowLut).
    // A beam crossing a `phase rainbow` medium cannot be folded at deposit time — its colour
    // is a function of a scattering angle nobody knows yet — but it CAN be folded at gather
    // time if the gather is told which emitter's spectrum to integrate against. `scale` is
    // what makes that legal here: it converts the subpath's β from the scene-wide emission
    // mixture to the emitter's OWN density, which is exactly the density BowLut integrates.
    // -1 when no emitter is identifiable (then the beam stays monochromatic, as before).
    int    emIdx = -1;
};

// Fill `bs` at a light subpath's birth. `leSpectral` is the λ-DEPENDENT part of the emitted
// radiance — spdFn(hero) · invPdfLambda(hero) — and is the only thing `scale` has to undo,
// because every other factor in Le (the emission pattern, a spot's falloff, the cosine and
// the three pdfs) is λ-free and cancels between the two forms.
inline void beginBeamSpectral(const Scene& scene, const Renderer& mats, const Emitter& em,
                              double leSpectral, Pcg32& rng, BeamSpectral& bs) {
    bs = BeamSpectral{};
    // An IMAGE environment is excluded for the reason render.h excludes it from its bundle:
    // its emitted radiance carries a directional factor L(dir,λ)/spd(λ) that is genuinely
    // per-wavelength, so no single `scale` describes it. (BDPT refuses env maps today; the
    // guard costs nothing and keeps this honest if that ever changes.)
    if (!mats.beamDeposit || !mats.beamSpecOK) return;
    if (em.shape == EmitterShape::Env && scene.envMap) return;
    if (!(leSpectral > 0.0) || !(em.spd.integral > 0.0)) return;
    bs.scale = em.spd.integral / leSpectral;
    bs.cie   = em.cieMean;
    // Recover `em`'s index for the gather-time fold. `em` is always a reference INTO
    // scene.emitters (generateLightSubpath picks it from there), so pointer arithmetic is
    // well-defined; the bounds test is belt-and-braces, and the 32767 clamp is the width of
    // PhotonBeam::emIdx.
    if (!scene.emitters.empty()) {
        const ptrdiff_t k = &em - &scene.emitters[0];
        if (k >= 0 && k < (ptrdiff_t)scene.emitters.size() && k <= 32767) bs.emIdx = (int)k;
    }
    const int C = mats.beamSpecC < 1 ? 1
                : (mats.beamSpecC > kBeamSpecMax ? kBeamSpecMax : mats.beamSpecC);
    // ONE uniform variate, stratified: member i takes u + i/C wrapped into [0,1) through the
    // emission CDF, so the bundle spans the emitter's spectrum instead of clumping. Same
    // scheme render.h uses, deliberately, so the two modes' bundles stratify alike.
    const double u0 = rng.uniform();
    for (int i = 0; i < C; ++i) {
        double uu = u0 + (double)i / (double)C;
        if (uu >= 1.0) uu -= 1.0;
        double pI = 0.0;
        const double lI = em.spd.sampleAt(uu, pI);
        // A zero-density member cannot be carried: `randomWalk` accumulates every member's
        // weight as a RATIO to the hero's factor, and a member the emitter cannot emit at all
        // has no meaningful ratio. Drop the WHOLE claim rather than renormalise over the
        // survivors, which would over-count them. (Inverting a CDF at a uniform variate cannot
        // land in a zero-mass bin, so this is a guard, not a path.)
        if (!(pI > 0.0)) { bs = BeamSpectral{}; return; }
        bs.lam[i] = lI;
    }
    bs.nLam = C;
    bs.ok   = true;
}

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
//
// `bsIn` seeds the spectral claim the beam deposits rest on (see BeamSpectral and
// PathSeg::achro) — it is read, never written, and supplies both the emitter identity the
// fold quadrature belongs to and the bundle's wavelengths. A caller that does not deposit
// beams passes nullptr and the walk is bit-identical. No GRIN guard is needed here, unlike
// render.h's photon walk, which has to exclude a bent path per medium: BDPT refuses a scene
// containing ANY gradient-index medium outright (`bdptUnsupportedFeature`, main.cpp), so a
// mode-J walk can never bend.
inline void randomWalk(const Scene& scene, const Camera& cam, const Renderer& mats,
                       Ray ray, double beta, double pdfDir, const HeroBundle& hb,
                       int maxDepth, Mode mode, Pcg32& rng, std::vector<Vertex>& path,
                       const double* betaSecIn, int nUpIn, Escape* esc = nullptr,
                       PathSegs* segs = nullptr, const BeamSpectral* bsIn = nullptr) {
    (void)cam;   // cam reserved for future NEE-to-camera use; mode now drives adjoint corr
    const double lambda = hb.lam[0];   // the hero drives geometry, sampling and every pdf
    if (maxDepth == 0) return;

    // ---- THE SPECTRAL CLAIM (see PathSeg::achro / BeamSpectral / render.h's foldWorthIt) --
    //
    // Before 0.257.0 this was a bare bool that ANY surface interaction cleared, because the
    // beam record could express only "every wavelength on this chord carries equal power".
    // With `PhotonBeam::wS` it can express unequal power, so the rule becomes the same one
    // render.h's tracePhoton uses: a wavelength-DIVERGENT event (one after which the
    // wavelengths no longer share a chord) retires the claim, while a merely
    // wavelength-DEPENDENT one (a diffuse albedo) is FOLDED into `foldT` / `beamW` and the
    // claim survives. That is the difference between mode J's rain reaching 36.2 % of its
    // beams with a bundle and reaching the same ~84 % mode M's fold reaches.
    const BeamSpectral* bs = (bsIn && bsIn->ok) ? bsIn : nullptr;
    bool achroPath = bs != nullptr;
    // The emitter whose quadrature table `foldT` indexes and whose bins the verdict reads.
    // `foldN == 0` would leave foldWorthIt nothing to judge, so such an emitter simply keeps
    // the pre-0.257.0 behaviour (retire at the first surface).
    const Emitter* foldEm =
        (bs && bs->emIdx >= 0 && bs->emIdx < (int)scene.emitters.size() &&
         scene.emitters[bs->emIdx].foldN > 0) ? &scene.emitters[bs->emIdx] : nullptr;
    const int specN = bs ? bs->nLam : 0;      // bundle members, INCLUDING the record's hero
    double foldT[kFoldBins];                  // T(foldLam[k]) — lazily initialised
    bool   foldChroma = false;                // has anything been folded into foldT yet?
    double beamW[kBeamSpecMax];               // T(bs->lam[i]) / T(lambda)
    for (int i = 0; i < kBeamSpecMax; ++i) beamW[i] = 1.0;
    if (segs && mode == Mode::Importance) g_foldKill = achroPath ? FK_None : FK_NeverBorn;
    // Retire the claim, stamping the diagnostic histogram with the event that killed it (and
    // only while it was actually live, so the attribution stays with the true cause).
    auto retireSpectral = [&](FoldKill why) {
        if (achroPath) { g_foldKill = why; achroPath = false; }
    };
    // Multiply the per-bin ratios f(lam_k)/f(lambda) into the running claim. `fk` is TWO
    // grids in one array — [0, K) the emitter's quadrature bins, [K, K + specN) the bundle's
    // own wavelengths — because they share every expensive thing: one texture fetch per
    // wavelength, one energy guard, one verdict. Exactly render.h's foldApply, except that
    // the bundle half runs over ALL members here (mode J's record hero is bs->lam[0], not
    // the walk's lambda, so member 0 needs a weight too).
    auto foldApply = [&](double fHero, const double* fk, int K) {
        const double inv = 1.0 / fHero;
        if (!foldChroma) { for (int k = 0; k < K; ++k) foldT[k] = 1.0; foldChroma = true; }
        for (int k = 0; k < K; ++k) foldT[k] *= fk[k] * inv;
        for (int i = 0; i < specN; ++i) beamW[i] *= fk[K + i] * inv;
    };
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
            // Snapshot the spectral claim AS IT STANDS AT THE SEGMENT ORIGIN, which is the
            // point the deposited beam's power refers to (`beta` above is the same snapshot,
            // taken for the same reason). The per-iteration rule below decides the NEXT
            // segment, not this one.
            //
            // `aGlass` is the exception and has to be folded in here rather than left to that
            // rule: emitBeams re-applies exp(-aGlass * ta) over each beam's own lead-in, so a
            // segment travelling inside coloured glass deposits a beam whose power is
            // wavelength-dependent even though nothing has happened to `beta` yet. Same
            // condition render.h's tracePhoton reaches by clearing the flag with the
            // Beer-Lambert step that precedes its deposit.
            sg.achro = achroPath && !(sg.aGlass > 0.0);
            if (sg.achro) {
                // Collapse the running per-bin ratios into the single CIE triple this beam
                // will be stored with. With nothing folded yet (`foldChroma` false) that is
                // still the emitter's plain cieMean, so an achromatic scene deposits
                // bit-for-bit what it deposited before the fold existed.
                if (foldChroma && foldEm) {
                    Vec3 c{0, 0, 0};
                    for (int k = 0; k < foldEm->foldN; ++k) c += foldEm->foldCie[k] * foldT[k];
                    sg.fCie = c;
                } else {
                    sg.fCie = bs->cie;
                }
                for (int i = 0; i < specN; ++i) sg.fW[i] = beamW[i];
            }
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
            // Glass absorption is spectral — that is what makes coloured glass coloured — so
            // once it has been applied `beta` no longer represents the whole band at equal
            // power and the achromatic claim is dead. (Twin of render.h's tracePhoton, which
            // clears it at the same point and for the same reason.)
            //
            // This one stays FATAL even now that the claim can carry per-member weights,
            // because `PhotonBeam::absorb` is a single scalar the gather re-applies to every
            // member of the record alike: there is nowhere to put a per-wavelength Beer
            // exponent. (Same reasoning as render.h's FK_GlassAbsorb.)
            if (a > 0.0) { beta *= std::exp(-a * dEvent); retireSpectral(FK_GlassAbsorb); }
            // Per-λ absorption for the bundle. A non-empty stack means we are inside a
            // dielectric, and entering one de-heros — so nUp is always 1 whenever `a`
            // can be non-zero and this loop never actually runs. Kept for generality.
            for (int i = 0; i + 1 < nUp; ++i) {
                double ai = curAbsorb(hb.lam[i + 1]);
                if (ai > 0.0) betaSec[i] *= std::exp(-ai * dEvent);
            }
        }

        // THE SPECTRAL-CLAIM RULE, medium half. Applied BEFORE the event below is acted on —
        // the segment above already took its snapshot, so what is being decided here is
        // whether the NEXT segment may still claim.
        //
        // A scatter in an ACHROMATIC medium is wavelength-independent outright (achromatic
        // sigma_t for the free flight that reached it, a flat albedo for the roulette, and an
        // HG direction that depends only on `g`), so the claim passes through untouched. A
        // scatter in a `phase rainbow` or spectrally-varying medium is wavelength-DIVERGENT —
        // the members would leave along different directions — and retires it. There is no
        // middle case here the way there is at a surface: a medium's chromaticity changes the
        // GEOMETRY, which no per-member weight can absorb.
        //
        // That the rule survives a scatter at all is the whole point: at gallery_rain's cloud
        // albedo of 0.9964 a light subpath scatters on the order of 278 times inside the
        // cloud, so a one-event rule would reach ~1 deposit in 278 there. This reaches all of
        // them, which is what stops mode J's clouds coming out iridescent.
        //
        // Note the test is on the medium the walk is scattering IN, not on the media the
        // segment crosses: a subpath that scatters in the achromatic cloud keeps the claim
        // and its next segment can still deposit a good beam into the `phase rainbow` rain it
        // passes through — as a stratified bundle rather than a fold, which emitBeams decides
        // per medium. (Twins: render.h tracePhoton and render_cuda.cu, same position.)
        //
        // The SURFACE half of the rule lives in the material switch below, because unlike a
        // medium event a surface event can be wavelength-DEPENDENT without being divergent,
        // and telling those apart needs the material in hand.
        if (achroPath && mediumEvent && !mediumAchromatic(scene.media[scatterMed]))
            retireSpectral(FK_ChromaMedium);

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
        } else if (mp->type == MatType::Layered) {
            // The same resolve a mix gets, with the coat as the first lobe: it wins with
            // probability R (the Fresnel reflectance at THIS angle), otherwise the ray enters
            // and a body lobe shades. The vertex then holds an ordinary material, which is what
            // every MIS density downstream evaluates -- the same convention a mix already uses,
            // where the pdf stored is the resolved child's and not the mixture's.
            const double R = layeredCoatReflectance(scene, *mp, h, ray.d, lambda);
            int c = (rng.uniform() < R) ? mp->coatChild
                                        : mixResolveChild(scene, *mp, h, rng.uniform());  // body lobe: honours a bound weight map (device twin: dResolveCompound)
            if (c < 0) return;                       // absorbed on the leftover slice
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
        // THE SPECTRAL-CLAIM RULE, surface half (see the medium half above, and render.h's
        // foldWorthIt / foldApply — ONE rule, because both tracers fill the same beam bank).
        // A case that can express its wavelength dependence as a scalar ratio per bin fills
        // `fldF` (the two-grid array foldApply consumes), sets `fldHero` to the factor the
        // walk's own lambda took, and sets `fldOK`. Everything else leaves `fldOK` false and
        // retires the claim with `fldWhy`, which defaults to the divergent case: a lobe that
        // picks its continuation direction as a function of lambda (dispersive refraction, a
        // grating order, thin-film/multilayer interference, fluorescence's wavelength switch)
        // leaves the members without a shared chord at all, so there is no T(lambda) to carry
        // and no weight that could rescue it.
        bool     fldOK = false;
        int      fldK = 0;
        double   fldHero = 0.0;
        FoldKill fldWhy = FK_Specular;
        double   fldF[kFoldBins + kBeamSpecMax];
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
                // SPECTRAL FOLD. A Lambertian albedo is the archetypal wavelength-DEPENDENT
                // but non-divergent factor: every wavelength leaves along the one cosine-
                // sampled direction, and only the throughput differs, as rho(lam_k)/rho(lam).
                // FLUORESCENCE IS EXCLUDED — it shares this case for its elastic base, but a
                // re-emission moves the photon to a different wavelength entirely, which is
                // divergent in the strongest sense. (Twin: render.h's MatType::Diffuse.)
                if (achroPath && mp->type == MatType::Diffuse) {
                    fldK = foldEm ? foldEm->foldN : 0;
                    for (int k = 0; k < fldK + specN; ++k) {
                        const double lk = (k < fldK) ? foldEm->foldLam[k] : bs->lam[k - fldK];
                        fldF[k] = clamp01(diffuseReflectance(scene, *mp, h, lk));
                    }
                    // One verdict, over the QUADRATURE half only. The bundle's wavelengths
                    // are drawn from the same variate as the walk's lambda, so a test that
                    // read them would correlate the fold/no-fold split with lambda and bias
                    // the mixture; the quadrature bins depend on the emitter alone.
                    bool ok = fldK > 0 && rho > 0.0 && foldWorthIt(*foldEm, fldF, fldK);
                    for (int i = 0; i < specN && ok; ++i) ok = fldF[fldK + i] > 0.0;
                    if (ok) { fldOK = true; fldHero = rho; }
                    else fldWhy = (rho > 0.0) ? FK_DeclineDiffuse : FK_ZeroWeight;
                }
                break;
            }
            case MatType::Glossy: {
                Vec3 mdir = reflect(ray.d, cur.ns);   // ray.d == -wo (incoming dir)
                wi = sampleGlossy(mdir, materialRoughness(scene, *mp, h), rng);
                if (dot(wi, cur.ns) <= 0) { terminate = true; break; }
                double r = clamp01(reflectSlot(scene, *mp, h, lambda));
                // FOLD-GLOSSY (0.260.1): the lobe's direction is wavelength-free, so the claim
                // survives; the albedo is folded exactly as Diffuse's is above (render.h's
                // tracePhoton does the same -- one PhotonBeam bank, one rule).
                if (achroPath) {
                    fldK = foldEm ? foldEm->foldN : 0;
                    for (int k = 0; k < fldK + specN; ++k) {
                        const double lk = (k < fldK) ? foldEm->foldLam[k] : bs->lam[k - fldK];
                        fldF[k] = clamp01(reflectSlot(scene, *mp, h, lk));
                    }
                    bool ok = fldK > 0 && r > 0.0 && foldWorthIt(*foldEm, fldF, fldK);
                    for (int i = 0; i < specN && ok; ++i) ok = fldF[fldK + i] > 0.0;
                    if (ok) { fldOK = true; fldHero = r; }
                    else fldWhy = (r > 0.0) ? FK_DeclineGlossy : FK_ZeroWeight;
                }
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
                // SPECTRAL FOLD. Both lobes are cosine-sampled about the same normal, so no
                // wavelength diverges — but WHICH lobe is taken depends on the walk's lambda,
                // and a verdict that depended on the lobe would therefore depend on lambda
                // and bias the estimator. So evaluate and test BOTH lobes up front, and only
                // then keep the one that was actually sampled. The per-bin ratio works out to
                // rho_k(lobe)/rho_hero(lobe): the secondary's factor here is the expected-
                // value form rho_k(lobe) * tot / rho_hero(lobe) and the hero's is `tot`, so
                // the `tot` cancels — the same ratio render.h's analog roulette arrives at
                // from the other direction. `diffuseTransmitAlbedos` applies the shared energy
                // guard per wavelength, which matters because the guard changes the SAMPLING
                // PROBABILITY and it is the guarded value the split actually used.
                if (achroPath) {
                    double fldTr[kFoldBins + kBeamSpecMax];
                    fldK = foldEm ? foldEm->foldN : 0;
                    for (int k = 0; k < fldK + specN; ++k) {
                        const double lk = (k < fldK) ? foldEm->foldLam[k] : bs->lam[k - fldK];
                        double a, b; diffuseTransmitAlbedos(*mp, lk, scene, &h, a, b);
                        fldF[k] = a; fldTr[k] = b;
                    }
                    const double hero = reflLobe ? rhoR : rhoT;
                    bool ok = fldK > 0 && hero > 0.0 &&
                              foldWorthIt(*foldEm, fldF, fldK) && foldWorthIt(*foldEm, fldTr, fldK);
                    for (int i = 0; i < specN && ok; ++i)
                        ok = fldF[fldK + i] > 0.0 && fldTr[fldK + i] > 0.0;
                    if (ok) {
                        fldOK = true; fldHero = hero;
                        if (!reflLobe)
                            for (int k = 0; k < fldK + specN; ++k) fldF[k] = fldTr[k];
                    } else {
                        fldWhy = (hero > 0.0) ? FK_DeclineTransmit : FK_ZeroWeight;
                    }
                }
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
                // vertex is connectible on the CAMERA side (NEE, vertex connections). A LIGHT
                // walk stops at one instead (below).
                const HairShade hs = hairAt(scene, *mp, h, lambda, wo);
                double pdfH = 0.0, fv = 0.0;
                const Vec3 wl = hair::sample(hs.b, hs.woLocal, rng.uniform(), rng.uniform(),
                                             rng.uniform(), rng.uniform(), pdfH, fv);
                if (!(pdfH > 0.0) || !(fv > 0.0)) { terminate = true; break; }
                wi = hair::toWorld(hs.fr, wl);
                // A coverage pass-through returns EXACTLY -wo: the ray was never intercepted.
                // That is a DELTA vertex in this file's own sense -- it carries the chain but
                // cannot be connected to, because its outgoing direction is fixed and its pdf
                // is a Dirac, not the finite solid-angle density `pdfH` reports.
                const bool passThru = (wl.x == -hs.woLocal.x && wl.y == -hs.woLocal.y &&
                                       wl.z == -hs.woLocal.z);
                pdfW    = pdfH;
                pdfRevW = passThru ? pdfH : bsdfPdf(*mp, cur.ns, wi, wo, lambda, scene, &h);
                delta   = passThru;
                // The pass-through direction is -wo for EVERY wavelength, so the bundle
                // rides through exactly as it does off a Mirror. Without this the walk's
                // de-hero rule (`delta && !keepBundle`) collapses it to the hero alone.
                if (passThru) keepBundle = true;
                // LIGHT SUBPATHS STOP AT STRANDS (0.365.0). The near-field fiber model is not
                // reciprocal and its offset `h` belongs to the CAMERA ray, so a strand on the light
                // side of a path has no value that agrees pointwise with the camera side's for the
                // same path: the best light-side value that agrees in integral over the strand's
                // width still differs across it, and the camera-walk pdf in the MIS weights varies
                // across it the same way -- measured +22 % on isolated fiber rows and up to 2.3x on
                // dense fur (known-issues HAIR-RECIPROCITY). So a light walk that scatters at a
                // strand ends here, unstored and unconnectible: every path through a strand is built
                // from the camera side alone, exactly as mode R builds it, and the MIS weights drop
                // the light-side alternatives through it (misWeight, misWeightReference, mode J's
                // camera-side accumulators). A coverage pass-through is a delta continuation that
                // is the same from either side, so it rides on.
                if (mode == Mode::Importance && !passThru) { path.pop_back(); return; }
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
                    // A pass-through is ACHROMATIC -- the ray missed the fiber, so every
                    // wavelength continues at weight 1. Re-evaluating the BCSDF here would
                    // return the coverage-scaled scatter value (0 at `opacity 0`) and
                    // extinguish the bundle at a fiber it never touched.
                    if (passThru) { secF[i] = 1.0; continue; }
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
        // Fold this vertex's spectral factor into the beam claim, or retire it. Sits beside
        // `beta *= betaFactor` because it is the same multiplication seen per wavelength:
        // foldT and beamW carry f(lam_k)/f(lambda), the ratio beta itself cannot express.
        if (achroPath) {
            if (fldOK) foldApply(fldHero, fldF, fldK);
            else       retireSpectral(fldWhy);
        }
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
                  ? Ray{cur.p + normalize(wi) * 1e-9, normalize(wi),
                        hairChordExit(h.fiberRadius, h.n, h.tangent, normalize(wi))}   // the exact chord (0.364.0)
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
                             PathSegs* segs = nullptr, BeamSpectral* bs = nullptr) {
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
    // Spectral-claim birth (see BeamSpectral). `fall` and the cone pdf are
    // wavelength-independent, so a spot or a sun starts the walk with the claim intact for
    // exactly the same reason an area light does, and `spdFn(lambda) * invPdfLambda` is the
    // whole of the λ-dependence in `Le`.
    if (bs) beginBeamSpectral(scene, mats, em, em.spdFn(lambda) * invPdfLambda, rng, *bs);
    randomWalk(scene, cam, mats, ray, betaWalk, pdfDir, hb, maxDepth - 1,
               Mode::Importance, rng, path, betaWalkSec, hb.C, nullptr, segs, bs);
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
                                PathSegs* segs = nullptr, BeamSpectral* bs = nullptr) {
    const double lambda = hb.lam[0], invPdfLambda = hb.invPdf[0];
    path.clear();
    // No claim until an emitter is chosen and accepted below; a caller that reuses one
    // `BeamSpectral` across subpaths must not see the previous subpath's claim on a path
    // that took one of the early returns.
    if (bs) *bs = BeamSpectral{};
    if (scene.emitters.empty() || scene.totalPower <= 0.0) return 0;
    int ei = scene.selectEmitter(rng);
    const Emitter& em = scene.emitters[ei];
    double pdfChoiceSel = em.power / scene.totalPower;
    if (pdfChoiceSel <= 0.0) return 0;
    if (em.shape == EmitterShape::Spot || em.shape == EmitterShape::Sun)
        return deltaLightSubpath(scene, cam, mats, em, pdfChoiceSel, hb, maxDepth, rng, path,
                                segs, bs);
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
    // SPECTRAL-CLAIM BIRTH for the beam deposits (see BeamSpectral; twin of the bundle and
    // achro blocks in render.h's tracePhoton). `Le`'s only λ-dependence is
    // `spdFn(lambda) * invPdfLambda` — `emitPatW`, the cosine and all three pdfs are λ-free —
    // so that product is exactly what `scale` has to undo to recover the emitter-normalised
    // power a mode-M photon would have carried.
    if (bs) beginBeamSpectral(scene, mats, em, em.spdFn(lambda) * invPdfLambda, rng, *bs);
    randomWalk(scene, cam, mats, ray, betaWalk, pdfDir, hb, maxDepth - 1,
               Mode::Importance, rng, path, betaWalkSec, hb.C, nullptr, segs, bs);
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
// it as `kappa`) and this returns the PRIMED eta. Returning 0 means "no beam-merge technique
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
inline double mergeEtaPrimeBeam(const Scene& scene, const Vec3& pPrev, const Vertex& v,
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

// --- The SECOND merge kind: point x point on a surface (UPBP's P-P, VCM's VM) ----------
//
// A photon stored at a light-subpath surface vertex, gathered by a camera subpath vertex
// that lands within r_s of it. Same balance-heuristic question as above, and it turns out to
// have a far simpler answer than the beam case, for a reason worth writing down: a beam
// merge INSERTS a vertex neither walk had (so the camera side's free-flight distance density
// survives in the ratio and the medium's sigma_t*Tr appears), whereas a surface merge
// IDENTIFIES a vertex both walks already have — the camera walk had to land its own vertex
// there too — so the camera-side area density cancels outright.
//
// Merged path x_0..x_{n-1} light-to-camera, merge site x_i (the photon's own position; the
// camera vertex within r_s of it is the same vertex to O(r_s), which is VCM's one geometric
// approximation). Reference technique, as for beams, is the CONNECTION strategy at that same
// index: m = i, i.e. "x_i is the last camera vertex, joined to x_{i-1}".
//
//     p_merge  = [prod_{u<=i} pl_u] * [prod_{u>i} pc_u] * n_m * pi r_s^2 * pc_i
//     p_C1     = [prod_{u<i}  pl_u] * [prod_{u>=i} pc_u]
//     ratio    = n_m * pi r_s^2 * pl_i
//
// (the `pi r_s^2 * pc_i` factor is the probability that the camera walk's own vertex lands
// inside the acceptance disc, and n_m the number of light subpaths that could have supplied
// the photon — the merge technique's sample count in the multi-sample balance heuristic.)
//
// CROSS-CHECK against SmallVCM, since this is the one place the two formalisms can be made
// to speak: vcm.h weighs the merge against a CONNECTION at strategy m = i+1 and gets
// `etaVCM * camDirPdfA` (`vcm.h`, wLight/wCamera in the connect branch), where camDirPdfA is
// the camera-side AREA density of the light vertex. Converting the reference here from m = i
// to m = i+1 multiplies by p_{m=i}/p_{m=i+1} = pc_i/pl_i, giving n_m*pi r_s^2*pc_i. Same
// expression. The two derivations share no arithmetic, so this is a real check.
//
// So the whole per-site part is `pLight` — the light-side AREA density of the merge site,
// which is exactly what the vertex already stores (pdfFwd on the light half, pdfRev on the
// eye half). The gate is only "could a photon have been STORED here", and it must agree
// EXACTLY with what the light pass stores or the partition of unity breaks: a site the
// weight counts but the map never fills under-weights every competing technique, and a site
// the map fills but the weight ignores double-counts. Hence one predicate, used by both.
inline bool surfMergeSite(const Vertex& v) {
    // NOT a fiber (0.365.0). A surface merge treats a stored vertex purely as a photon and
    // evaluates the CAMERA vertex's BSDF against it, so a photon that landed on a strand lying
    // on the skin bled into every skin gather point within the radius -- the defect VCM was
    // cured of in 0.362.0. This one predicate gates the SurfMap store, the gather and the
    // merge's MIS eta term alike, so excluding strands here is consistent by construction.
    // (Since 0.365.0 a light walk never stores a strand vertex at all -- randomWalk -- so the
    // store half is moot; the gather and the eta term at a strand CAMERA vertex are not.)
    // Measured before: mode J 2.3x mode R on the fur of a 3 000-strand test scene, 6.2x with
    // the beam map off (where `-nobeams` is documented as mode D bit-for-bit).
    return v.type == VType::Surface && !v.delta && v.mat && isConnectibleMat(*v.mat) &&
           !isFiberMat(*v.mat);
}
inline double mergeEtaPrimeSurf(const Vertex& v, double pLight) {
    if (!surfMergeSite(v)) return 0.0;
    return (pLight > 0.0) ? pLight : 0.0;
}

// The two merge kinds' constants, and one site's primed eta in each. They travel as a pair
// because a mode-J denominator has to carry BOTH: a camera path crossing a cloud and landing
// on a wall competes with beam merges in the medium and point merges on the wall, in the same
// sum, and dropping either one over-weights everything that is left.
//
// WHY THE CONSTANTS ARE DEFERRED (i.e. why `MergeEta` is primed rather than finished): the
// light pass accumulates each subpath's merge terms BEFORE either constant exists — n_m is
// how many subpaths that pass turns out to emit, and both radii are chosen from the finished
// map — so a stored accumulator has to keep the two kinds apart and let the camera pass scale
// them. Everything downstream of the map (misWeight, the gathers) knows both and uses
// `scale()` immediately.
struct MergeK {
    double beam = 0.0;      // n_m * 2 r_b     — beam x ray (BB1D), 1D kernel of full width 2r
    double surf = 0.0;      // n_m * pi r_s^2  — point x point (VM), acceptance disc
    bool any() const { return beam > 0.0 || surf > 0.0; }
};
struct MergeEta {
    double beam = 0.0;
    double surf = 0.0;
    double scale(const MergeK& k) const { return k.beam * beam + k.surf * surf; }
    bool any() const { return beam > 0.0 || surf > 0.0; }
};

// Both primed etas at one site. At most one is ever non-zero (a vertex is in a medium or on a
// surface, not both), but they are returned as a pair so no caller has to know which.
inline MergeEta mergeEtaPrime(const Scene& scene, const Vec3& pPrev, const Vertex& v,
                              const Vec3& pNext, double pLight, double lambda) {
    MergeEta e;
    if (v.type == VType::Medium)
        e.beam = mergeEtaPrimeBeam(scene, pPrev, v, pNext, pLight, lambda);
    else
        e.surf = mergeEtaPrimeSurf(v, pLight);
    return e;
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
//
// What the pilot below settled on, so the driver can report it. A budget that silently
// rewrote the user's `-n` would be the same kind of invisible surprise this whole entry is
// about, so mode J prints every field of this.
struct BeamBudgetInfo {
    long long pilotPaths   = 0;      // subpaths traced and thrown away to measure the rate
    double    beamsPerPath = 0.0;    // ... and the rate it measured
    long long kneeBeams    = 0;      // ... and where it put the -beamk knee (0 = not measured)
    long long pathsAsked   = 0;      // `-n`, or mode J's default
    long long pathsUsed    = 0;      // what the budget allowed
    long long budget       = 0;      // the beam ceiling actually enforced
    bool      kneeBound    = false;  // was the knee the binding half, or the resource ceiling?
    bool      applied      = false;  // did it bite at all, or was `-n` already smaller?
    // --- the surface map's half of the same decision (-jsurf) ---------------------------
    double    surfPerPath  = 0.0;    // surface photons per subpath, measured by the same pilot
    bool      surfBound    = false;  // did the SURFACE ceiling pick nPaths, rather than the beams?
};

// What the caller wants of the map. A struct rather than three more parameters because `blur`
// and `targetK` MUST be the same values the subsequent buildAuto() is given -- the knee is only
// meaningful at the radii the real build will use -- so they are forwarded from the one place
// that owns them (`-beamblur` / `-beamk`) rather than defaulted twice.
struct BeamBudgetReq {
    long long maxBeams = 0;      // resource ceiling in raw beams; 0 disables the budget entirely
    double    blur     = 0.01;   // -beamblur:  kernel half-width as a fraction of the mfp
    double    targetK  = 32.0;   // -beamk:     the FLOOR whose release marks the knee
    double    safety   = 1.0;    // scale on the measured knee; 1.0 = aim AT it (see the note below)
    // --- THE SURFACE MAP'S HALF (-jsurf, 0.258.0) ---------------------------------------
    // One subpath count feeds TWO maps, so both get a say, and they say different kinds of
    // thing. The beam side names a *beam* ceiling because its radius adapts: undershoot the
    // knee and buildAuto silently widens the kernel, which is a BIAS (see `safety`). The
    // surface side has a FIXED radius (-jsurf-radius), so more photons is only ever more
    // quality for more cost -- never a different estimator. Its two knobs reflect that:
    //
    //   `surfPaths` -- a subpath TARGET, applied only when the caller knows the beams have no
    //   opinion (a media-free scene). Mode U's convention, one light subpath per pixel per
    //   pass, is what makes "-mode J -jsurf against -mode U" a comparison of estimators; and
    //   without it a media-free mode-J render inherits the inert `-n 2000000` default and
    //   tries to store ~7 M photons for a 64x64 image.
    //
    //   `maxSurfPhotons` -- a pure MEMORY ceiling, applied always. A SurfPhoton + its SurfMis
    //   is 72 B, so an unbounded map is the one way this feature can take the process down;
    //   it lowers nPaths, never thins the map (a thinned map's survival probability is exactly
    //   what a MIS weight cannot read -- the same reason mode J refuses to trim its beams).
    long long surfPaths      = 0;   // subpath target from the surface side; 0 = no opinion
    long long maxSurfPhotons = 0;   // ceiling on STORED surface photons; 0 = unbounded
};
// WHY `safety` DEFAULTS TO 1.0, AND NOT TO SOMETHING SAFELY BELOW THE KNEE.
//
// It shipped at 0.5 on the reasoning that the knee is where extra beams stop being free, so
// sitting under it must be the cautious side. Gate 3 -- the closed-form single-scatter slab,
// the one validation gate whose ground truth is not another ftrace mode -- says otherwise.
// Sweeping `-n` across the knee on scenes/_slab_ss.ftsl (knee ~8742 beams; `-max-bounce 1`
// makes it exactly 1 beam/subpath, which is what makes this the clean scene to measure on),
// each run 90 s, `scale` relative to the first row and `Q1` the dimmest brightness quartile:
//
//     beams    vs knee    scale     Q1      noise rms
//      4371      0.5x     1.064x   1.291      1.168     <- the old default: the ONLY outlier
//      8742      1.0x     0.943x   1.150      1.121
//     17484      2.0x     0.924x   1.046      0.804
//     50000      5.7x     0.906x   1.058      0.592
//    200000     22.9x     0.940x   1.021      0.281
//
// The last four agree on the absolute level to ~4 %; the half-knee map alone sits ~6 % above
// them, with a monotonic shape error across the quartiles (1.29 in the dimmest, 0.99 in the
// brightest). Both are the same artefact: below the knee `buildAuto` INFLATES the radii to
// hold the gathered count at the `-beamk` floor, and a wider kernel smears energy from the
// bright near-light region into the dim far one. That is bias, and no amount of render time
// removes it.
//
// What undershooting actually trades, stated carefully -- because the first version of this
// comment got it wrong. Below the knee the gather still RETURNS `targetK` beams per probe (the
// floor guarantees exactly that), so shrinking the map does not reduce the estimator's
// variance-per-sample. But it does still make the BVH cheaper to traverse -- fewer split
// sub-beams, smaller total box area -- and traversal, not the returned count, is where the
// per-camera-segment time goes. So undershooting is not free: it buys spp, and it pays for
// them in kernel width. Measured on `_fog_cornell` at 120 s, 128^2: 57 k beams -> 13 spp and
// 0.450 relative RMSE, 114 k (the knee) -> 7 spp and 0.587.
//
// We decline that trade anyway, on the grounds that the two sides are not the same kind of
// error. Noise is removable by rendering longer; a widened kernel is not, and worse, it is
// silent -- below the knee the render is using a kernel wider than the `-beamblur` the user
// asked for, and nothing says so. Mode J's headline claim is that its ABSOLUTE radiance is
// right (gate 3), so the default belongs where `-beamblur` means what it says. The knee is
// exactly that point: the SMALLEST map whose radii are the requested `blur * mfp` with no
// inflation. Anyone who wants `_fog_cornell`'s extra spp can ask for it with `-beamcount`.
//
// Known limitation, deliberately not papered over: on a scene whose gather is very cheap
// (`_slab_ss` is `-max-bounce 1`, so one medium span per camera path) the table above keeps
// improving well past the knee, because the per-sample cost is dominated by everything other
// than the gather. The knee is a good target when the gather is a significant share of
// per-sample cost -- which is the case that motivated the budget at all (J-BEAMCOST: a 12 M
// beam map that made a 64x64 frame look like a hang) -- and merely a conservative one when it
// is not. Raising `-beamcount` (or setting `-n`) is the escape hatch for that case.
//
// The measured spread of the true optimum, once normalised by the knee, is narrow: about
// 0.66x the knee on `_fog_cornell` and about 1.0x on `_fog_thick` (where the knee beat every
// hand-swept point by 29 %). That 1.5x spread is what is left of a 23x raw spread in beam
// count, which is the real justification for the knee as the normaliser.
//
// Pilot sizing. `kPilotMin` is large enough that the measured rate is stable to a few percent
// on any scene where the media are reachable at all; `kPilotMax` caps it so a huge `-n` does
// not pay for a huge pilot it does not need (the rate is an average — its error falls as
// 1/sqrt(n) and 64k subpaths is already far past the point of usefulness).
inline constexpr long long kPilotMin = 2048;
inline constexpr long long kPilotMax = 65536;
// Never let the budget shrink the pass to a map too sparse to be a technique at all. A map
// this small contributes almost nothing and the MIS weights correctly give it almost nothing,
// so the render degrades to mode D rather than going wrong.
inline constexpr long long kPathsMin = 256;

// THE BEAM BUDGET, AND WHY IT IS SPENT ON *SUBPATHS* RATHER THAN ON BEAMS (0.242.0)
// --------------------------------------------------------------------------------
// Because the deposit takes no Russian roulette, the map's size is `nPaths` times a
// scene-dependent "media spans per subpath", and NOTHING used to bound it. That is not a
// tuning wart, it is a cliff: the gather's cost per camera segment is linear in the stored
// beam count (the kernel radius is `blur * mfp`, fixed, so twice the beams means twice the
// beams a probe ray sweeps), and past `-beamsplitmax` the BVH-tightening split is starved
// outright and the box area jumps ~100x. Measured on `_fog_cornell` at 128x128 against a
// converged mode-D reference, 120 s each: `-n 12500` reached 15 spp at 0.44 relative RMSE,
// `-n 200000` reached 1 spp at 1.24 and tripped the split budget, and the DEFAULT `-n
// 2000000` did not finish a single sample and wrote no image at all. See J-BEAMCOST.
//
// `req` therefore caps the map. It is spent by lowering `nPaths`, not by thinning the beams
// afterwards, for two reasons. The map is this pass's ONLY product — mode J's connection half
// traces its own light subpaths at render time — so tracing subpaths whose beams get thrown
// away is pure waste. And a uniform post-hoc thin is only equivalent to tracing fewer subpaths
// in expectation anyway, since keeping a fraction `keep` of the beams and normalising by
// `nEmitted * keep` is exactly what `nEmitted = nPaths * keep` already means. Lowering
// `nPaths` gets the same map for less work.
//
// THE PILOT MEASURES TWO THINGS, and neither is knowable up front. (1) Spans-per-subpath, which
// ranges from well under 1 (a small bounded cloud in a big room, most subpaths missing it) to
// `maxDepth` (a global haze) — this converts a beam count into a subpath count. (2) The KNEE:
// the beam count past which extra beams stop being free, which is where the `-beamk` floor
// releases. See the note at the measurement itself. Both come from one pilot of a few thousand
// subpaths, which is then DISCARDED.
//
// Discarding it is the point, not laziness about reusing it. If the pilot's beams were kept,
// `nPaths` would be a function of those same beams, the map's normalisation would be
// correlated with its contents, and the estimator would pick up an O(pilot/nPaths) bias --
// small, but this mode's whole validation story is that its absolute radiance is right (gate
// 3), and "biased by an amount we think is small" is not that. Run under an INDEPENDENT salt
// and thrown away, the pilot makes `nPaths` a random variable independent of the map's own
// randomness; the map is unbiased for every fixed `nPaths`, hence unbiased averaged over the
// pilot's choice of it. The cost of that guarantee is the pilot's own tracing, a couple of
// percent of the pass.
//
// SURFACE PHOTONS RIDE ALONG (`smOut`, 0.258.0). The point-merge map has to be built from
// these SAME subpaths, for exactly the reason this function exists at all: a merge weight is
// a ratio against the connections `renderRows` makes, and only a photon carrying this pass's
// own pdfFwd/pdfRev bookkeeping is a ratio between comparable things. It is a second output
// rather than a second pass because a second pass would be a second set of subpaths — the
// beams and the photons would then have different n_m, and every weight that mentions both
// (which is every weight in a scene with media AND surfaces) would be wrong.
inline void traceLightBeamPass(const Scene& scene, const Camera& cam, long long nPaths,
                               int nThreads, int maxDepth, bool diffraction,
                               BeamMap& bm, StageProgress* stage = nullptr,
                               BeamBudgetReq req = {}, BeamBudgetInfo* budgetOut = nullptr,
                               SurfMap* smOut = nullptr) {
    if (nThreads < 1) nThreads = 1;
    if (nPaths < 1) nPaths = 1;
    const long long nPathsAsked = nPaths;
    // "JUPBPLGT" — a salt of its own, so mode J's light pass cannot alias mode M's photon
    // stream or its own render-time streams however the counts line up.
    const uint64_t seedBase = 0x4A555042504C4754ULL;
    // "JUPBPPLT" — the pilot's own stream, disjoint from the map's. See the note above: this
    // independence is what keeps the budget from biasing the estimator.
    const uint64_t pilotSeed = 0x4A55504250504C54ULL;
    std::vector<BeamBank> banks((size_t)nThreads);     // cap stays 0: no self-thinning
    // The light half of every merge's MIS weight, one entry per beam, kept in lockstep with
    // `banks[t].beams`. It is NOT a member of BeamBank on purpose: mode M's photon pass
    // shares that type and must not pay 40 bytes a beam for something it never reads, and
    // BeamBank::halve() would silently desynchronise a parallel array (mode J avoids that by
    // setting cap = 0, so banks only ever grow — see the note above).
    std::vector<std::vector<BeamMis>> misBanks((size_t)nThreads);
    // The point-merge map's per-thread banks, filled only when the caller asked for one.
    // Same lockstep discipline as `misBanks`: SurfBank holds the photon and its MIS partials
    // in two arrays pushed one after the other, and a size mismatch at the end is treated as
    // a bug rather than indexed through (see the merge below).
    std::vector<SurfBank> sbanks((size_t)nThreads);
    const bool wantSurf = (smOut != nullptr);
    std::vector<long long> emitted((size_t)nThreads, 0);
    std::atomic<long long> tracedTotal{0};

    // `pilot` shares the deposit path with the real pass and skips only the MIS bookkeeping,
    // which is deliberate: what the pilot has to predict is how many beams `emitBeams` makes,
    // so it must run the SAME code that makes them. A separate hand-written estimator would
    // silently drift the first time the deposit rule changed.

    // The scene-wide spectral predicate, hoisted out of the worker because beamSpectralOK()
    // scans every medium's spectra and must not run per subpath. Mode M's driver sets the
    // same three fields in photonmap_render.h; without them here, mode J deposited a beam at
    // ONE saturated wavelength per subpath — drawn, worse, from the scene-wide mixture rather
    // than from the emitter that actually emitted it — and painted both media in coloured
    // streaks. A beam is a LINE, so a monochromatic deposit lays its hue down a whole chord
    // and reads as iridescent structure rather than as grain.
    const bool beamSpecOK  = beamSpectralOK(scene);
    const bool beamAchroOK = pbeams::gAchro && beamSpecOK;
    const int  beamSpecC   = beamSpecOK ? pbeams::gSpecC : 1;

    auto worker = [&](int tid, long long total, uint64_t salt, bool pilot) {
        Renderer mats; mats.diffraction = diffraction;
        mats.beamDeposit = &banks[(size_t)tid];
        mats.beamAchroOK = beamAchroOK;
        mats.beamSpecOK  = beamSpecOK;
        mats.beamSpecC   = beamSpecC;
        Pcg32 rng;
        std::vector<Vertex> path;
        PathSegs segs;
        BeamSpectral bs;                               // per-subpath, reset by every call
        std::vector<BeamMis>& misBank = misBanks[(size_t)tid];
        std::vector<double> accC, accMb, accMs;        // per-subpath, reused
        const PatTables tabs = scene.patTables();
        const long long lo = total * tid / nThreads, hi = total * (tid + 1) / nThreads;
        long long done = 0;
        for (long long i = lo; i < hi; ++i) {
            // Cooperative `-stop` / Ctrl-C on the same 4096-path cadence as the photon pass.
            if ((done & 0xFFF) == 0) {
                if (done && !pilot) tracedTotal.fetch_add(0x1000, std::memory_order_relaxed);
                if (ft::stopRequested()) break;
            }
            // Seeded by ABSOLUTE path index, so the map is identical for any thread count.
            seedUnit(rng, salt + (uint64_t)i, 0xD1B54A32D192ED03ULL);
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
            generateLightSubpath(scene, cam, mats, hb, maxDepth + 1, rng, path, &segs, &bs);
            // The deposit form this subpath's segments will use, decided once (it is a
            // property of the emitter, not of the segment; `sg.achro` supplies the per-segment
            // half). See BeamSpectral for why the hero wavelength is REPLACED rather than
            // extended: it comes from the scene-wide mixture density, and a bundle member
            // weighted 1/C has to come from the emitter's own.
            const double* dLamS = (bs.ok && bs.nLam > 1) ? bs.lam + 1 : nullptr;
            const int     dSec  = (bs.ok && bs.nLam > 1) ? bs.nLam - 1 : 0;
            const double  dLam  = bs.ok ? bs.lam[0] : hb.lam[0];
            const double  dSc   = bs.ok ? bs.scale : 1.0;
            // MODE J DECLINES THE GATHER-TIME FOLD (0.256.0), AT DEPOSIT TIME, ON PURPOSE.
            // `-1` here is the whole opt-out, and it has to be here rather than only at the
            // gather, for a reason that is easy to miss: BeamBank::push treats the fold and the
            // `-beamspec` BUNDLE as mutually exclusive (`nSec = (lamS && nSec > 0 && !cieA)`),
            // because one record cannot carry both an emitter-folded colour and a set of
            // secondary wavelengths. So asking for the fold SILENTLY DESTROYS the bundle. Mode
            // J's gather refuses to use the fold (BeamMergeWeight::kFoldGatherTime is false),
            // so requesting it here would leave the rain's beams with neither a fold nor a
            // bundle: strictly monochromatic, the worst of the three. Measured on
            // gallery_rain's rain ROI against the 6212-spp mode-D reference, relative RMSE:
            //
            //                             seed 7   seed 11
            //   monochromatic (neither)   0.4263   0.5033
            //   bow fold, no bundle       0.4157   0.4875
            //   bundle, no fold           0.2876   0.2835
            //   SHIPPED (bundle + the cloud's own deposit-time fold)
            //                             0.2587   0.2788
            //
            // The bundle is worth roughly 4x what the fold is worth here, and the two are
            // mutually exclusive, so mode `J` takes the bundle. (Mode `M` orders the middle two
            // the other way round — there the beam map IS the whole volumetric estimate, so
            // removing ALL chromatic variance from every crossing beats sampling four
            // wavelengths of it; in mode `J` the merge is one MIS-weighted technique among
            // several and the weight stays lambda-dependent whatever the colour does, which is
            // variance the fold cannot reach but a bundle can.)  See known-issues.md,
            // UPBP-BOWFOLD.
            //
            // `bs.emIdx` is still computed and still travels, because randomWalk needs it to
            // find the emitter's fold quadrature table (0.257.0); it simply is not requested
            // as a GATHER-time fold here.
            const int     dEm   = -1;

            // Deposit one segment's beams. The per-segment spectral payload (PathSeg::fCie /
            // fW) is what randomWalk accumulated up to that segment's ORIGIN, which is the
            // point the beam's stored power refers to.
            //
            //   power   = beta * scale * fW[0]      — the flux a mode-M photon drawn from
            //                                         this emitter's own SPD at bs.lam[0]
            //                                         would have carried (see BeamSpectral)
            //   wS[i-1] = fW[i] / fW[0]             — member i relative to that hero
            //
            // With no spectral factor folded, every fW is 1 and this reduces exactly to the
            // pre-0.257.0 `sg.beta * dSc` with an equal-weight bundle.
            auto depositSeg = [&](const PathSeg& sg) {
                double lam = hb.lam[0], pw = sg.beta;
                const double* lamS = nullptr; int nSec = 0;
                const Vec3* cie = nullptr;
                double wsBuf[kBeamSpecMax];
                if (sg.achro) {
                    const bool fold = beamAchroOK &&
                        (sg.fCie.x > 0.0 || sg.fCie.y > 0.0 || sg.fCie.z > 0.0);
                    if (fold) {
                        // FOLDED (-beamachro): the beam is gathered at `fCie`, whose per-bin
                        // ratios foldT[k] = T(foldLam[k]) / T(lambda_walk) are relative to the
                        // WALK's wavelength. So the power must be the walk's own throughput,
                        // beta * scale, at lambda_walk. It used to be beta * scale * fW[0] --
                        // the throughput re-expressed at bundle member 0, which in mode J is a
                        // FRESH SPD sample, not the walk's hero (see BeamSpectral) -- and the
                        // product then carried a stray T(lam0)/T(lambda_walk). Over two
                        // independent wavelengths that factor averages E[T]*E[1/T] >= 1: 1.00
                        // on a flat white wall, several-fold on a red wall (T 0.05..0.6), which
                        // is why _fog_cornell came out +40% in X with Y untouched, mode J only
                        // (mode M's member 0 IS its walk's hero), and clean with -beamachro off.
                        // The bundle is mutually exclusive with the fold (emitBeams zeroes nSec
                        // when cieA is given), so no per-member weight needs the same treatment.
                        lam = hb.lam[0];
                        pw  = sg.beta * dSc;
                        cie = &sg.fCie;
                    } else {
                        // MONOCHROMATIC at bundle member 0 (with the other members as a bundle):
                        // here the beam really is the flux at bs.lam[0], so fW[0] applies.
                        const double w0 = sg.fW[0];
                        if (!(w0 > 0.0)) return;             // T(bs.lam[0]) == 0: no flux
                        lam = dLam;
                        pw  = sg.beta * dSc * w0;
                        if (dSec > 0) {
                            for (int i = 0; i < dSec; ++i) wsBuf[i] = sg.fW[i + 1] / w0;
                            lamS = dLamS; nSec = dSec;
                        }
                    }
                }
                // MedAll, not the emitBeams default MedStraight: nothing about this span is
                // carried stochastically (that is what LONG means), so every medium the
                // lead-in crosses must be charged. See Renderer::emitBeams.
                //
                // `sg.achro` is the spectral claim as it stood where this span began
                // (randomWalk / PathSeg::achro); emitBeams then asks the further, per-medium
                // question of whether THIS medium's gather tail is flat, so a subpath crossing
                // the achromatic cloud and the `phase rainbow` rain in one step folds the
                // cloud's beam to a single colour and gives the rain's a weighted bundle
                // instead — the rain's scattering really is chromatic, so its beam has to keep
                // wavelengths, it just no longer has to keep only ONE.
                // kBeamOrderUnknown, NOT `sg.vert`. `PathSeg::vert` is the subpath VERTEX
                // index and counts surface bounces too, where PhotonBeam::order means MEDIUM
                // scattering order -- the quantity `-beams-order` caps and the one VOLCACHE
                // would key on. Passing `vert` here would read as a valid order and be wrong
                // by however many surfaces the subpath touched. Counting medium scatters in
                // randomWalk is the fix when mode `J` needs this; until then the field says
                // "unknown" out loud.
                mats.emitBeams(scene, sg.o, sg.d, sg.tMax, lam, pw, sg.aGlass, rng,
                               kBeamOrderUnknown,
                               Renderer::MedAll, lamS, nSec, cie, dEm,
                               nSec > 0 ? wsBuf : nullptr);
            };

            // --- The light half of every merge weight this subpath can take part in -------
            // Both accumulators telescope exactly as misWeight's light loop does, one vertex
            // at a time, so a beam leaving y_j can read off the whole prefix in O(1). The
            // recurrences (see BeamMis in photonbeams.h for what they sum):
            //   ratioL(u) = pdfRev(y_u)/pdfFwd(y_u)          [the loop's per-step factor]
            //   accC[j]   = gate(j) + ratioL(j-1) * accC[j-1]
            //   accM*[j]  =           ratioL(j-1) * (accM*[j-1] + eta'(y_{j-1}))
            // `eta'(y_{j-1})` needs BOTH of y_{j-1}'s neighbours, so it only exists from
            // j = 2 on; the missing j = 1 term is the merge at y_0, which is the light
            // itself and is neither a medium vertex nor a stored photon site in any case.
            //
            // TWO merge accumulators, not one, because the two kinds' constants are not the
            // same number and neither is known yet (MergeK): `accMb` sums the beam-merge
            // sites (medium vertices), `accMs` the point-merge ones (stored photons), and the
            // camera pass scales each by its own kappa. Folding them together here would be
            // an unrecoverable loss — the ratio between the kinds varies per site.
            const size_t np = path.size();
            if (pilot) {
                // Count only. The pilot's beams are thrown away with the bank it fills, so
                // the merge-weight prefixes below would be computed for nothing.
                for (const PathSeg& sg : segs)
                    if (sg.beta > 0.0) depositSeg(sg);
                continue;
            }
            accC.assign(np, 0.0);
            accMb.assign(np, 0.0);
            accMs.assign(np, 0.0);
            for (size_t u = 0; u < np; ++u) {
                const bool dPrev = (u > 0) ? path[u - 1].delta : path[0].isDeltaLight();
                const double gate = (!path[u].delta && !dPrev) ? 1.0 : 0.0;
                if (u == 0) { accC[0] = gate; continue; }
                const double rL = misRemap0(path[u - 1].pdfRev) / misRemap0(path[u - 1].pdfFwd);
                const MergeEta eL = (u >= 2)
                    ? mergeEtaPrime(scene, path[u - 2].p, path[u - 1], path[u].p,
                                    path[u - 1].pdfFwd, hb.lam[0])
                    : MergeEta{};
                // --- STORE A SURFACE PHOTON, if this vertex is one -----------------------
                // Done HERE rather than in a second loop because everything a point merge's
                // light half needs is in hand at exactly this step and nowhere else: the
                // accumulators one index BACK (a point merge's reference connection splits
                // after y_{u-1}, not after y_u — see SurfMis), plus `eL`, which is the merge
                // at y_{u-1} that the recurrence would not fold in until the next iteration.
                if (wantSurf && surfMergeSite(path[u]) && path[u].pdfFwd > 0.0 &&
                    path[u].beta > 0.0) {
                    Vec3 dPrev = path[u - 1].p - path[u].p;
                    const double rho2 = dot(dPrev, dPrev);
                    if (rho2 > 0.0) {
                        const double rho = std::sqrt(rho2);
                        dPrev = dPrev * (1.0 / rho);
                        // The cosine of pdfRev*(y_{u-1}) is at y_{u-1} and faces back along
                        // this same edge; 1 at a medium vertex, which has no normal.
                        const double cosPrev = path[u - 1].onSurface()
                                             ? std::fabs(dot(path[u - 1].ns, dPrev)) : 1.0;
                        SurfBank& sb = sbanks[(size_t)tid];
                        SurfPhoton ph;
                        ph.p      = path[u].p;
                        ph.wo     = dPrev;                 // unit, toward the previous vertex
                        ph.lambda = (float)hb.lam[0];
                        ph.beta   = (float)path[u].beta;
                        ph.cx = (float)cieX(ph.lambda);
                        ph.cy = (float)cieY(ph.lambda);
                        ph.cz = (float)cieZ(ph.lambda);
                        ph.misIdx = (unsigned)sb.mis.size();
                        SurfMis sm;
                        sm.sumC    = accC[u - 1];
                        sm.sumMb   = accMb[u - 1] + eL.beam;
                        sm.sumMs   = accMs[u - 1] + eL.surf;
                        sm.pdfFwdA = path[u].pdfFwd;
                        sm.rCoef   = (float)(cosPrev /
                                             (rho2 * misRemap0(path[u - 1].pdfFwd)));
                        sm.gateC1  = (unsigned char)(gate > 0.0 ? 1 : 0);
                        sm.vert    = (unsigned short)u;
                        sb.pts.push_back(ph);
                        sb.mis.push_back(sm);
                    }
                }
                accC[u]  = gate + rL * accC[u - 1];
                accMb[u] = rL * (accMb[u - 1] + eL.beam);
                accMs[u] = rL * (accMs[u - 1] + eL.surf);
            }

            for (const PathSeg& sg : segs) {
                if (!(sg.beta > 0.0)) continue;
                const size_t before = banks[(size_t)tid].beams.size();
                depositSeg(sg);
                const size_t after = banks[(size_t)tid].beams.size();
                if (after == before) continue;

                // One template for every beam this segment deposited: they differ only in
                // where along the segment each medium starts (`leadIn`), because everything
                // else here is a property of the vertex the segment LEAVES.
                const size_t j = (size_t)sg.vert;
                const Vertex& y = path[j < np ? j : np - 1];
                BeamMis m;
                m.sumC   = accC[j < np ? j : np - 1];
                m.sumMb  = accMb[j < np ? j : np - 1];
                m.sumMs  = accMs[j < np ? j : np - 1];
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
                // Its POINT twin, for a beam that left a SURFACE: eta = pdfFwd(y_{s-1}), and
                // there is nothing merge-point-dependent left in it at all. The j >= 1 gate
                // matches the beam one — y_0 is the emitter, which no photon is stored at.
                m.etaPrevS = (j >= 1 && j < np && surfMergeSite(y) && y.pdfFwd > 0.0)
                                 ? (float)y.pdfFwd : 0.0f;
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
        // The tail, matching the in-loop `done && !pilot` guard: the pilot's subpaths are
        // thrown away, so counting them here would report more progress than the pass will
        // ever have work for and drive the percentage past 100.
        if (!pilot) tracedTotal.fetch_add(done & 0xFFF, std::memory_order_relaxed);
        // Count what was actually emitted, not what was asked for: an early `-stop` that
        // reported the full share would scale the whole map down and darken the merges.
        emitted[(size_t)tid] = done;
    };

    // --- THE SURFACE MAP'S SUBPATH TARGET, which needs no measurement --------------------
    // Applied before the pilot so the pilot is sized against the count that will actually be
    // traced. The caller sets this only when it knows the beams have no opinion (see
    // BeamBudgetReq::surfPaths), so there is no arbitration to do here: whichever side is
    // speaking is the only side speaking.
    if (smOut && req.surfPaths > 0)
        nPaths = std::clamp(req.surfPaths, kPathsMin, nPathsAsked);

    // --- PILOT: how many beams does one subpath actually deposit in THIS scene? ----------
    // Sized as a fraction of the request with a hard ceiling, so it is a rounding error on a
    // large run and never dominates a small one. Skipped entirely when there is no budget to
    // meet, or when the request is already too small to be worth measuring.
    //
    // A surface-photon ceiling is the SECOND thing that can require a pilot, and it can do so
    // on a scene with no media at all (where `maxBeams` measures nothing) — hence the `||`
    // rather than a beams-only gate.
    if ((req.maxBeams > 0 || (smOut && req.maxSurfPhotons > 0)) && nPaths > 4 * kPilotMin) {
        // FTRACE_JPILOT overrides the size, for the sweep that decides whether `/32` is
        // buying anything. The sizing is off the REQUESTED nPaths, which on a knee-bound scene
        // is wildly larger than what the pass will actually trace: `-n 2000000` gives a 62 500
        // subpath pilot to size a pass that then traces 2 514. The entry's own note says the
        // rate is stable at kPilotMin and that 64k is "far past the point of usefulness", so
        // the `/32` term may be paying 30x for nothing -- but that is a claim to measure, not
        // to assume, and this hook is how.
        long long nPilot = std::clamp(nPaths / 32, kPilotMin, kPilotMax);
        if (const char* e = std::getenv("FTRACE_JPILOT")) {
            const long long v = std::atoll(e);
            if (v > 0) nPilot = v;
        }
        std::vector<std::thread> ppool;
        for (int t = 0; t < nThreads; ++t)
            ppool.emplace_back(worker, t, nPilot, pilotSeed, /*pilot*/true);
        for (auto& th : ppool) th.join();
        // Merge the pilot's beams into a throwaway map, because the SECOND thing the pilot
        // measures needs them as a map rather than as a count — see the knee note below.
        BeamMap pm;
        size_t pilotBeams = 0;
        for (auto& b : banks) pilotBeams += b.size();
        pm.beams.reserve(pilotBeams);
        for (auto& b : banks) {
            pm.beams.insert(pm.beams.end(), b.beams.begin(), b.beams.end());
            b.beams.clear(); b.beams.shrink_to_fit();
        }
        // THE SURFACE BANKS MUST BE EMPTIED HERE TOO, and this is not symmetry for its own
        // sake: the pilot's subpaths are thrown away, but `smOut->nEmitted` is set from the
        // REAL pass's emission count. Leaving the pilot's photons in the banks would put
        // ~3 % more flux in the map than the normalisation divides by, i.e. a silent
        // brightening of every surface merge that scales with the pilot fraction.
        size_t pilotSurf = 0;
        for (SurfBank& sb : sbanks) pilotSurf += sb.size();
        for (SurfBank& sb : sbanks) {
            sb.pts.clear(); sb.pts.shrink_to_fit();
            sb.mis.clear(); sb.mis.shrink_to_fit();
        }
        const double surfPerPath = (double)pilotSurf / (double)nPilot;
        // A pilot that deposited NOTHING says the media are hard to reach, not that they are
        // unreachable — leave `nPaths` alone rather than dividing by zero or inflating it to
        // something unbounded on the strength of a sample that measured nothing.
        const double perPath = (double)pilotBeams / (double)nPilot;

        // THE KNEE, MEASURED ON THIS SCENE (0.242.0). The caller's `beamBudget` is a resource
        // ceiling, not a physics one. The count that actually matters is where the `-beamk`
        // FLOOR stops binding: while the raw-mfp-radius gather is below the floor, buildAuto
        // inflates the radii to hold the GATHERED count at `-beamk`, so extra beams cost the
        // gather nothing and buy a tighter (less blurred) kernel. Past that point the floor
        // lets go and every extra beam is gathered and paid for.
        //
        // MEASURED, and it is the reason this is not the frame-scaled rule J-BEAMCOST first
        // proposed. On `_fog_cornell` at 120 s, the collapse happens at the same RAW BEAM COUNT
        // at both resolutions -- 128x128: 18708 beams -> 15 spp / 0.426 relRMSE, 75288 -> 15 spp
        // / 0.410, 300033 -> 4 spp / 0.887. 256x256: the same three maps give 4 spp / 0.805,
        // 4 spp / 0.822, 1 spp / 1.888. Four times the pixels, same knee. The optimum is a
        // property of the SCENE (mfp, extent, how far beams reach), not of the frame, so it is
        // measured here rather than computed from res*spp.
        //
        // probeGatherCount at the raw radii is exactly buildAuto's `probeK0`, and it is very
        // nearly linear in the beam count, so one pilot extrapolates it:
        //     beams at the knee ~= pilotBeams * (targetK / probeK0_pilot)
        // The estimate is biased LOW -- the map's bounding box grows with the beam count, which
        // damps probeK0's growth -- which is the safe direction: the RMSE curve is flat below
        // the knee (0.426 vs 0.410 over a 4x range) and doubles above it.
        // Mirrors buildAuto's radius rule exactly (r = blur * mfp per medium, with a medium
        // that stored nothing borrowing the largest radius seen so nothing divides by zero).
        // It has to: `probeK0` is only the knee's coordinate if it is measured at the radii the
        // real build will actually use.
        double kneeBeams = 0.0;
        if (pilotBeams && req.targetK > 0.0) {
            std::vector<BeamMap::MedStat> st = pm.mediumStats();
            double rSeen = 0.0;
            for (BeamMap::MedStat& s : st) { s.r = req.blur * s.mfp; rSeen = std::max(rSeen, s.r); }
            if (!(rSeen > 0.0)) {
                const Vec3 ext = pm.bounds().hi - pm.bounds().lo;
                rSeen = 1e-4 * std::max(std::sqrt(dot(ext, ext)), 1e-9);
            }
            pm.radMed.resize(st.size());
            for (size_t m = 0; m < st.size(); ++m)
                pm.radMed[m] = (float)(st[m].r > 0.0 ? st[m].r : rSeen);
            const double k0 = pm.probeGatherCount(pm.bounds());
            if (k0 > 1e-6) kneeBeams = (double)pilotBeams * (req.targetK / k0);
        }
        pm.beams.clear(); pm.beams.shrink_to_fit();

        // The budget that binds is the smaller of the two: the caller's resource ceiling and
        // the scene's own knee. `req.safety` scales the knee, and it defaults to 1.0 -- i.e.
        // aim AT the knee -- because the knee is not "where extra beams stop being free", it is
        // the SMALLEST map whose kernel is the one the user asked for. Below it `buildAuto`
        // widens the radii to hold the gathered count at the `-beamk` floor, so undershooting
        // buys no time (the gather still returns `targetK` beams either way) and pays for it in
        // kernel blur. That is a bias, not just noise, and gate 3 measures it: see the sweep in
        // the header note above `kPilotMin`.
        double effBudget = (double)req.maxBeams;
        if (kneeBeams > 0.0) effBudget = std::min(effBudget, req.safety * kneeBeams);
        if (req.maxBeams > 0 && perPath > 0.0) {
            const long long fit = (long long)(effBudget / perPath);
            nPaths = std::clamp(fit, kPathsMin, nPathsAsked);
        }
        // THE SURFACE CEILING, applied AFTER the beam knee and only ever downward. It is a
        // memory bound, not a quality target: the point-merge radius is fixed, so a bigger
        // surface map is strictly better and this must not be allowed to raise nPaths back up
        // on a scene where the beam knee already chose a smaller one.
        bool surfBound = false;
        if (smOut && req.maxSurfPhotons > 0 && surfPerPath > 0.0) {
            const long long fitS = (long long)((double)req.maxSurfPhotons / surfPerPath);
            const long long capped = std::clamp(fitS, kPathsMin, nPaths);
            surfBound = (capped < nPaths);
            nPaths = capped;
        }
        if (budgetOut) {
            budgetOut->pilotPaths   = nPilot;
            budgetOut->beamsPerPath = perPath;
            budgetOut->kneeBeams    = (long long)kneeBeams;
            budgetOut->pathsAsked   = nPathsAsked;
            budgetOut->pathsUsed    = nPaths;
            budgetOut->budget       = (long long)effBudget;
            budgetOut->kneeBound    = (kneeBeams > 0.0 &&
                                       req.safety * kneeBeams < (double)req.maxBeams);
            budgetOut->applied      = (nPaths < nPathsAsked);
            budgetOut->surfPerPath  = surfPerPath;
            budgetOut->surfBound    = surfBound;
        }
        if (ft::stopRequested()) {
            bm.beams.clear(); bm.mis.clear(); bm.nEmitted = 0;
            if (smOut) { smOut->pts.clear(); smOut->mis.clear(); smOut->nEmitted = 0; }
            return;
        }
    }

    // The subpath count is settled here whether a pilot ran or not, and the caller needs it
    // either way: a refresh epoch re-traces exactly `pathsUsed`, so leaving it 0 on the
    // no-pilot path would make every epoch after the first fall back to the raw `-n` default
    // and quietly trace hundreds of times more than epoch 0 did.
    if (budgetOut) {
        budgetOut->pathsAsked = nPathsAsked;
        budgetOut->pathsUsed  = nPaths;
    }
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t)
        pool.emplace_back(worker, t, nPaths, seedBase, /*pilot*/false);
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

    // --- The SECOND output: the surface photon map (mode J's point merges) -----------
    // Merged from the same banks the beams came from, and given the SAME nEmitted, because
    // the beams and the photons are two views of ONE set of light subpaths. Every weight
    // that mentions both kinds (SurfMis::sumMb / BeamMis::sumMs, and the camera-side
    // segSumM which is scaled by both kappas) assumes a single n_m; two passes would give
    // them two, and every cross term would be wrong by that ratio.
    if (smOut) {
        size_t np = 0;
        for (const SurfBank& sb : sbanks) np += sb.size();
        smOut->pts.clear();
        smOut->mis.clear();
        ftalloc::reserve(smOut->pts, np, "mode J's surface photon map", "-n (which sizes it)");
        ftalloc::reserve(smOut->mis, np, "mode J's per-photon MIS partials",
                         "-n (which sizes it)");
        smOut->nEmitted = bm.nEmitted;
        for (int t = 0; t < nThreads; ++t) {
            SurfBank& sb = sbanks[(size_t)t];
            // `misIdx` is BANK-LOCAL (each worker numbered from 0 into its own `sb.mis`), so
            // it MUST be rebased by the running offset as the banks are concatenated. Getting
            // this wrong does not crash and does not look wrong — it silently pairs a photon
            // with another thread's MIS partials, which is the one bug class this file cannot
            // survive, so the rebase happens in the same loop as the copy and nowhere else.
            const unsigned base = (unsigned)smOut->mis.size();
            for (const SurfPhoton& ph : sb.pts) {
                SurfPhoton q = ph;
                q.misIdx += base;
                smOut->pts.push_back(q);
            }
            smOut->mis.insert(smOut->mis.end(), sb.mis.begin(), sb.mis.end());
        }
        // Same contract as the beams': the two arrays are pushed one after the other and can
        // only disagree through a bug. Dropping the MIS array disables the point merges
        // outright (SurfMap::misOf returns null) rather than adding unweighted energy to an
        // already-complete BDPT sum.
        if (smOut->mis.size() != smOut->pts.size()) { smOut->mis.clear(); smOut->pts.clear(); }
    }
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
// Mode J adds the MERGE strategies to the same sum: one per interior vertex that a photon
// beam (medium) or a photon (surface) could have been stored at, each weighted by its primed
// eta times that kind's constant, against the connection strategy at that same vertex. `mk`
// is all-zero in mode D, which deletes them and leaves this function byte-identical.
inline double misWeightReference(const Scene& scene, const std::vector<Vertex>& light,
                                 const std::vector<Vertex>& eye, int s, int t,
                                 double lambda, const MergeK& mk) {
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
    // A strand is never a light vertex (randomWalk, 0.365.0): a strategy whose light subpath
    // would contain one does not exist, nor does a merge the light subpath could only reach
    // through one.
    int firstFiber = n;
    for (int i = 0; i < n; ++i) {
        const Vertex& v = (i < s) ? light[(size_t)i] : eye[(size_t)(n - 1 - i)];
        if (!v.delta && v.type == VType::Surface && v.mat && isFiberMat(*v.mat)) { firstFiber = i; break; }
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
        if (j > firstFiber) return false;                    // a strand on the light side
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
    if (mk.any()) {
        auto vAt = [&](int i) -> const Vertex& {
            return (i < s) ? light[(size_t)i] : eye[(size_t)(n - 1 - i)];
        };
        for (int i = 1; i <= n - 2; ++i) {
            if (i >= firstFiber) break;                      // reached only through a strand
            const double e = mergeEtaPrime(scene, vAt(i - 1).p, vAt(i), vAt(i + 1).p,
                                           pl[(size_t)i], lambda).scale(mk);
            if (e > 0.0)
                sum += e * std::exp(A[(size_t)i] + B[(size_t)i] - logPs);
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
// `mk` (the two merge kinds' constants — see MergeK; all-zero in every mode but J) adds the
// merge strategies to the denominator. It has to be here
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
                        const MergeK& mk = MergeK{}) {
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

    const bool merges = mk.any();
    double sumRi = 0.0, ri = 1.0;
    for (int i = t - 1; i > 0; --i) {                // hypothetical camera strategies
        // A strand is never a light vertex (randomWalk, 0.365.0): every strategy from here
        // inward would put eye[i] on the light side, so none of them exists.
        if (!eye[i].delta && eye[i].mat && isFiberMat(*eye[i].mat)) break;
        ri *= remap0(eye[i].pdfRev) / remap0(eye[i].pdfFwd);
        if (!eye[i].delta && !eye[i - 1].delta) sumRi += ri;
        if (merges && i >= 2) {                      // merge at eye[i-1] (see header note)
            const double e = mergeEtaPrime(scene, eye[i].p, eye[i - 1], eye[i - 2].p,
                                           eye[i - 1].pdfRev, lambda).scale(mk);
            if (e > 0.0) sumRi += ri * e;
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
                                           light[i].pdfFwd, lambda).scale(mk);
            if (e > 0.0) sumRi += ri * e;
        }
    }
    // The merge at the connection vertex pt itself. Its own connection strategy IS this
    // one, so the ratio multiplying it is exactly 1.
    if (merges && s >= 1 && t >= 2) {
        const double e = mergeEtaPrime(scene, light[s - 1].p, eye[t - 1], eye[t - 2].p,
                                       eye[t - 1].pdfRev, lambda).scale(mk);
        if (e > 0.0) sumRi += e;
    }
    const double w = 1.0 / (1.0 + sumRi);
    // Gate (2), off unless `-misaudit`: cross-check the relative form above against the
    // absolute one, here, where the ScopedAssigns are still installed and both therefore
    // see identical densities.
    if (misaudit::enabled.load(std::memory_order_relaxed)) {
        const double ref = misWeightReference(scene, light, eye, s, t, lambda, mk);
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
    // FAR-SIDE ORIGIN (0.365.0). A fiber connecting out the FAR side starts 1e-6 ALONG the
    // connection, inside its own tube, so the tube's far wall lies at chord - 1e-6: inside the
    // curve-only skip the exact chord supplies. Offsetting along the normal instead put the
    // far wall at chord + 1e-6/cos, BEYOND the skip, for directions more than 45 deg off the
    // inward normal -- the strand occluded its own grazing far-side connections, which is
    // where a wide lobe's tail goes: -2 % (beta 0.05) to -5.4 % (beta 0.8) of a strand's
    // direct light against mode R, compounding to -9 % on dense fur. Mode R's shadow rays
    // always started along the direction (backward.h emitterGeom).
    if (v.mat && v.mat->type == MatType::Hair && v.hit.fiberRadius > 0.0 && dot(v.ns, dir) < 0.0)
        return v.p + dir * 1e-6;
    // A fiber connecting out the FAR side has to clear the strand's own body: strand
    // radii are microns, so ng*1e-6 lands inside the tube and the connection reports
    // itself occluded, deleting exactly the TT glow that makes light hair look lit.
    // The far-side fiber clearance is no longer a displacement of the origin (which hid
    // any non-fiber surface within 2.5r -- the 0.360.2 defect); it travels as the
    // curve-only tmin from connFiberStep() instead.
    return offsetOrigin(v, dir);
}

// The far-side clearance for a connection leaving a fiber, as a CURVE-ONLY tmin.
inline double connFiberStep(const Vertex& v, const Vec3& dir) {
    if (v.type != VType::Medium && v.mat && v.mat->type == MatType::Hair &&
        v.hit.fiberRadius > 0.0 && dot(v.ns, dir) < 0.0)
        return hairChordExit(v.hit.fiberRadius, v.ns, v.hit.tangent, dir);   // the exact chord (0.364.0)
    return 0.0;
}

// How much shorter the shadow ray is than the full endpoint distance, given that
// connOrigin may have moved the start point forward along `dir`. Keeps the ray from
// overshooting into the light it is testing visibility to.
inline double connShorten(const Vertex& v, const Vec3& dir, double eps) {
    if (v.type != VType::Medium && v.mat && v.mat->type == MatType::Hair &&
        v.hit.fiberRadius > 0.0 && dot(v.ns, dir) < 0.0)
        return hairChordExit(v.hit.fiberRadius, v.ns, v.hit.tangent, dir) + eps;
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
                          double* Lsec, int& nUpConn, const MergeK& mk = MergeK{}) {
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
            // ADJOINT: qs is a particle vertex, so the BSDF's own Veach correction applies as
            // well as the shading-normal one below. See bsdfFAdjoint.
            f = bsdfFAdjoint(*qs.mat, qs.ns, wo, wcam, lambda, scene, &qs.hit);
            // Adjoint shading-normal correction: qs is a LIGHT-subpath (particle) vertex
            // whose f is evaluated toward the camera (wcam = outgoing). 1 when ns==ng.
            // The correction and stG are pure geometry — shared by every wavelength.
            const double adj = (isFiberMat(*qs.mat) ? 1.0
                                                    : shadingAdjointCorr(wo, wcam, qs.ns, ngo)) * stG;
            f *= adj;
            for (int i = 0; i + 1 < nUp; ++i)
                fSec[i] = bsdfFAdjoint(*qs.mat, qs.ns, wo, wcam, hb.lam[i + 1], scene,
                                       &qs.hit) * adj;
        }
        {   // max over live wavelengths (identical to `f <= 0` when nUp==1)
            double mxF = f;
            for (int i = 0; i + 1 < nUp; ++i) if (fSec[i] > mxF) mxF = fSec[i];
            if (!(mxF > 0.0)) return 0.0;
        }
        // The t=1 strategy: this segment runs from a LIGHT-subpath vertex to the camera, so
        // it is the camera leg and `hide_camera` applies to it (see Scene::occluded).
        // Partial, not yes/no: fur below opacity 1 attenuates the connection rather than
        // killing it. The origin stays put; the strand's own body is excluded by tmin.
        // HARD block, deliberately: in a bidirectional integrator a coverage pass-through is a
        // DELTA VERTEX of the sampled path, so a connection that passed THROUGH a fiber would
        // be the same transport as a sampled path one vertex longer, and MIS cannot pair paths
        // of different lengths -- letting connections through double-counted it (D 0.9450 ->
        // 1.4123 on the invisibility null). Fur-crossing transport is carried by sampling.
        if (scene.occluded(connOrigin(qs, wcam), wcam, dist - 2e-6, 1e-6, /*camLeg=*/true,
                           connFiberStep(qs, wcam))) return 0.0;
        const double vis = 1.0;
        // Transmittance of the fog the connection ray crosses (1 in vacuum, no RNG).
        // Evaluated at the hero only: the hero gate disables bundling when the scene has
        // any medium, so Tr is exactly 1 whenever nUp > 1.
        double Tr = mats.mediaTransmittance(scene, qs.p, wcam, dist, lambda, rng) * vis;
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
        if (scene.occluded(connOrigin(pt, wi), wi, dist - occlEps, 1e-6, false,
                           connFiberStep(pt, wi))) return 0.0;   // hard block: see t=1
        const double vis = 1.0;
        // Hero-only transmittance (exactly 1 whenever nUp > 1; see the t==1 branch).
        double Tr = mats.mediaTransmittance(scene, pt.p, wi, dist, lambda, rng) * vis;
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
            // ADJOINT: qs is the LIGHT endpoint, a particle vertex (see bsdfFAdjoint). pt, the
            // eye endpoint above, is a radiance vertex and correctly uses plain bsdfF.
            fL = bsdfFAdjoint(*qs.mat, qs.ns, woL, w * -1.0, lambda, scene, &qs.hit);
            // Adjoint shading-normal correction on the LIGHT-subpath endpoint qs (particle
            // vertex; outgoing = w*-1 toward the eye vertex). The eye endpoint pt is a
            // Radiance vertex and gets NO correction. 1 when ns==ng (flat/analytic).
            // Pure geometry, so it applies unchanged to every wavelength.
            const double adjL = isFiberMat(*qs.mat)
                                    ? 1.0
                                    : shadingAdjointCorr(woL, w * -1.0, qs.ns, ngoL);
            fL *= adjL;
            for (int i = 0; i + 1 < nUp; ++i)
                fLSec[i] = bsdfFAdjoint(*qs.mat, qs.ns, woL, w * -1.0, hb.lam[i + 1], scene,
                                        &qs.hit) * adjL;
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
        if (scene.occluded(connOrigin(pt, w), w, dist - 2e-6 - connShorten(qs, w * -1.0, 0.0),
                           1e-6, false, connFiberStep(pt, w))) return 0.0;   // hard block: see t=1
        const double vis = 1.0;
        // Hero-only transmittance (exactly 1 whenever nUp > 1; see the t==1 branch).
        double Tr = mats.mediaTransmittance(scene, pt.p, w, dist, lambda, rng) * vis;
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
    const double mis = misWeight(scene, cam, light, eye, sampled, s, t, lambda, mk);
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
    // NO GATHER-TIME SPECTRAL FOLD HERE (beamgather.h, WeightFn::kFoldGatherTime). The fold
    // substitutes E_lambda[CIE(lambda)*p(cos,lambda)] for one sample of it — an exact
    // substitution only when nothing else in the term depends on lambda. `operator()` below is
    // a ratio of path densities and one of them carries the phase function, so the weight IS a
    // function of lambda and E[w*CIE*p] != E[w]*E[CIE*p]; the residual is Cov(w, CIE*p).
    // Measured, that residual is below the noise floor (see the note at the fold site), so this
    // flag is not what makes mode J correct. What it does is keep the gather bit-identical to
    // pre-0.256.0 if a folded bank ever reaches it. The DECISIVE reason mode J has no bow fold
    // is upstream, at the deposit: a record cannot hold both a fold and a `-beamspec` bundle,
    // and mode J's answer to a rainbow medium is emphatically the bundle — rain relative RMSE
    // 0.2788 with it against 0.4875 folded, at seed 11. See known-issues.md, UPBP-BOWFOLD.
    static constexpr bool kFoldGatherTime = false;
    const Scene*   scene = nullptr;
    const BeamMap* bm    = nullptr;
    const PathSeg* sg    = nullptr;
    double lamCam = 0.0;
    double kappa  = 0.0;      // n_m * 2r: THIS technique's sample count times its kernel
    double kappaS = 0.0;      // n_m * pi r_s^2: the OTHER merge kind's, for the denominator.
                              // A camera ray that crosses a medium and lands on a wall is
                              // competed for by point merges as well as beam ones, and a
                              // denominator missing them over-weights every beam merge.
    // Per-segment camera-side constants (see BdptRenderer::renderRows, where they are built).
    double gateS1     = 0.0;  // is the reference connection C1 legal? (eye[k] not delta)
    double cosFacK    = 1.0;  // projected cosine at eye[k] along the segment; 1 off a surface
    double invPdfFwdK = 1.0;  // 1 / remap0(eye[k].pdfFwd)
    double etaKCoef   = 0.0;  // sin(theta_k) / (sigma_t(eye[k]) * Tr~(eye[k] -> eye[k-1]))
    double etaKSurf   = 0.0;  // kappaS if eye[k] is a stored-photon site, else 0 — the POINT
                              // merge at eye[k], whose eta is just kappaS * pdfRev(eye[k])
    double segSumC    = 0.0;  // camera-side connection accumulator from eye[k] inward
    double segSumM    = 0.0;  // camera-side merge accumulator, BOTH kinds, already scaled by
                              // their kappas (unlike the light side, the camera pass knows
                              // them, so nothing is deferred here)
    int    camVert    = 0;    // k: the camera subpath index of the vertex this segment leaves
    int    maxDepth   = 0;    // the same cap the connection loop applies (see below)
    // Per-ray transmittance data, hoisted out of the per-hit path. `camTr` is built once for
    // this camera segment (see TrRay: a dense medium hands one segment hundreds of hits, and
    // trDet would re-clip and re-evaluate sigma_t for every one of them); `tabs` is the
    // gather's own PatTables, so the light-side march stops rebuilding one per hit too.
    const TrRay*     camTr = nullptr;
    const PatTables* tabs  = nullptr;

    bool   raw = false;       // FTRACE_J_HALF=merges-raw (diagnostic): skip the weight entirely
    JDiag* diag = nullptr;    // non-null only under FTRACE_J_HALF (see JDiag)

    double operator()(const BeamHit& bh, const PhotonBeam& b, double dens, double phase) const {
        if (diag) diag->hits.fetch_add(1, std::memory_order_relaxed);
        if (raw) return 1.0;                  // diagnostic: unweighted BB1D, i.e. mode M's
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
        if ((int)lm->vert + camVert + 1 > maxDepth) {
            if (diag) diag->capped.fetch_add(1, std::memory_order_relaxed);
            return 0.0;
        }
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
        // Merge AT eye[k], of whichever kind eye[k] admits: the beam form when it is a medium
        // vertex (etaKCoef carries its sin/sigma_t/Tr), the point form when it is a stored-
        // photon site (eta = kappaS * pdfRev, so the coefficient IS kappaS). Both read the
        // same PATCHED pdfRev — the density of eye[k] seen from the merge point x.
        const double etaK = (kappa * etaKCoef + etaKSurf) * pdfRevK;
        // The merge AT y_{s-1}. Its sin(theta) and p_L were known when the beam was
        // deposited (its outgoing direction IS the beam); only the transmittance over the
        // now-known y_{s-1} -> x span is left.
        double etaPrevTerm = 0.0;
        if (lm->etaPrev > 0.0f) {
            const Vec3 yPrev = b.o - b.d * (double)lm->leadIn;
            const double trP = trDet(*scene, yPrev, b.d, rhoL, (double)b.lambda, *tabs);
            if (trP > 0.0) etaPrevTerm = kappa * (double)lm->etaPrev / trP;
        }
        // ...and the POINT merge at y_{s-1}, for the case the beam left a SURFACE into the
        // medium. Nothing about it depends on x (its eta is kappaS * pdfFwd(y_{s-1}) — see
        // mergeEtaPrimeSurf), so unlike its beam twin above it needs no transmittance and was
        // finished at deposit time.
        const double etaPrevS = kappaS * (double)lm->etaPrevS;
        const double den = (double)lm->gateC1                    // C1 itself (ratio 1)
                         + R * lm->sumC                          // light-side connections
                         + gateS1 * C1                           // the t-1 camera connection
                         + C1 * C2 * segSumC                     // camera-side connections
                         + R * (kappa * lm->sumMb + kappaS * lm->sumMs
                                + etaPrevTerm + etaPrevS)        // light-side merges
                         + etaS                                  // this merge
                         + C1 * etaK                             // merge at eye[k]
                         + C1 * C2 * segSumM;                    // camera-side merges
        if (!(den > 0.0)) return 0.0;
        return etaS / den;
    }
};

// --- The OTHER merge weight (mode J): point x point on a surface ----------------------
//
// One of these is built per CAMERA VERTEX (not per segment — a point merge happens AT a
// vertex, not along a ray) and applied once per gathered photon.
//
// SHAPE. The merged path is y_0 .. y_{j-1}, [ y_j == eye[k] ], eye[k-1], .., eye[0]: j+k+1
// vertices, depth j+k — one SHORTER than a beam merge with the same indices, because this
// merge identifies a vertex both walks already have instead of inserting a new one. For the
// same reason its reference technique C1 is the connection at split s = j (join y_{j-1} to
// the merge site, which is the last camera vertex), one strategy earlier than the beam case.
// EVERY light-side accumulator is therefore read at j-1 where BeamMis reads at j; that offset
// is baked into SurfMis at store time, so nothing here has to know about it.
//
// THE PATCHED DENSITIES. The light walk left y_j along its own continuation and the camera
// walk arrived along another, so two recorded densities are wrong in the merged path and are
// recomputed from the CAMERA vertex's BSDF (the photon's own material is never consulted —
// a merge shades with the camera vertex's BSDF, and its density has to match):
//
//   pdfRev(y_{j-1})  <- pdfDirRev = pdf(eye[k]: wo_cam -> photon.wo)   [VCM's camDirPdfW]
//   pdfRev(eye[k-1]) <- pdfDirFwd = pdf(eye[k]: photon.wo -> wo_cam)   [VCM's camRevPdfW]
//
// The pair AT the site does not appear at all, unlike the beam case: with no inserted vertex
// the camera-side area density of the site cancels between p_merge and p_C1, leaving
// p_merge/p_C1 = n_m * pi r_s^2 * pl_j — the light-side area density alone. See
// mergeEtaPrimeSurf for that derivation and its SmallVCM cross-check.
//
// SPECTRAL MISMATCH: same documented approximation as BeamMergeWeight — the photon carries
// its own wavelength, the camera path another, and the two BSDF pdfs above are evaluated at
// the PHOTON's (they pair with light-side densities). See that struct's note.
struct SurfMergeWeight {
    double kappaB = 0.0;      // n_m * 2 r_b     — the OTHER merge kind, for the denominator
    double kappaS = 0.0;      // n_m * pi r_s^2  — THIS technique's constant
    // Per-camera-vertex constants (built in BdptRenderer::renderRows).
    double invPdfFwdK   = 1.0;  // 1 / remap0(eye[k].pdfFwd)
    double gateD1       = 0.0;  // is connecting eye[k] to eye[k-1] legal? (eye[k-1] not delta;
                                // eye[k] cannot be, surfMergeSite already refused a delta)
    double cosPrev      = 1.0;  // |cos| at eye[k-1] along the eye[k]<->eye[k-1] edge; 1 in a
                                // medium and at the camera vertex, neither of which has one
    double invDistPrev2 = 0.0;  // 1 / |eye[k] - eye[k-1]|^2
    double invPdfFwdKm1 = 1.0;  // 1 / remap0(eye[k-1].pdfFwd)
    double etaKm1Coef   = 0.0;  // the merge AT eye[k-1], less its pdfRev: kappaB*sin/(sigT*Tr)
                                // in a medium, kappaS at a stored-photon site, 0 otherwise.
                                // Merge-site independent because both of eye[k-1]'s
                                // neighbours are known (the site IS eye[k]).
    double segSumC      = 0.0;  // camera connections from eye[k-1] inward  (segSumC[k-1])
    double segSumM      = 0.0;  // camera merges from eye[k-2] inward, both kinds already
                                // scaled by their kappas       (segSumM[k-1])
    int    camVert      = 0;    // k
    int    maxDepth     = 0;

    double operator()(const SurfMis& lm, double pdfDirRev, double pdfDirFwd) const {
        // THE DEPTH CAP, exactly as BeamMergeWeight's and for the same reason — but at j+k,
        // not j+k+1: this merge adds no vertex. Past the cap the merged path is one no
        // connection strategy builds, so its energy would have nothing to MIS against.
        if ((int)lm.vert + camVert > maxDepth) return 0.0;
        const double etaS = kappaS * lm.pdfFwdA;      // THIS merge, as a ratio against C1
        if (!(etaS > 0.0)) return 0.0;
        // pdfRev(y_{j-1})/pdfFwd(y_{j-1}) — the light loop's first (and only patched) step.
        // rCoef carries the cosine, the inverse-square and the remapped 1/pdfFwd, all fixed
        // when the photon was stored; only the directional density had to wait for a gather.
        const double R = pdfDirRev * (double)lm.rCoef;
        // pl_j/pc_k: the camera loop's first step. remap0 on both, as misWeight does.
        const double D1 = misRemap0(lm.pdfFwdA) * invPdfFwdK;
        // pdfRev(eye[k-1]) in the MERGED path, and with it the camera loop's second step.
        const double pdfRevKm1 = pdfDirFwd * cosPrev * invDistPrev2;
        const double D2 = pdfRevKm1 * invPdfFwdKm1;
        const double den = (double)lm.gateC1                     // C1 itself (ratio 1)
                         + R * (lm.sumC                          // light-side connections
                                + kappaB * lm.sumMb              // light-side merges, both
                                + kappaS * lm.sumMs)             //   kinds, scaled here
                         + etaS                                  // this merge
                         + D1 * (gateD1                          // connect eye[k]<->eye[k-1]
                                 + etaKm1Coef * pdfRevKm1        // merge at eye[k-1]
                                 + D2 * (segSumC + segSumM));    // everything further in
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

    // MODE J, the SECOND merge kind: the view-independent SURFACE photon map (surfmerge.h)
    // to merge camera surface vertices against — VCM's vertex merging, which mode J did not
    // have and mode U existed to provide. Null means beams only, exactly as `beams` null
    // means mode D, and the same gate applies: with it null the arithmetic below must be
    // bit-identical to the beams-only mode J that preceded it.
    //
    // The two maps are INDEPENDENT: a scene with no media gets `photons` and no `beams`
    // (that is mode U's job, done here), a scene of pure fog gets `beams` and an empty
    // `photons`, and a scene with both gets both — which is the whole point of folding U in.
    const SurfMap* photons = nullptr;

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
        const bool useHero = (C > 1) && scene.media.empty() &&
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
        MergeK mk;
        if (mergeOn && !beams->mis.empty())
            mk.beam = (double)beams->nEmitted * 2.0 * beams->radRef();
        // ...and the SURFACE merge kind's, when a photon map was built alongside the beams.
        // Both constants are shared by every weight in the frame, and both must be visible to
        // every weight: the two techniques compete for the same paths.
        // The MIS array is part of the gate here, unlike the beam case: a beam map without
        // partials still renders (weight 1, i.e. mode M's estimator, over-bright but a
        // picture), while an unweighted point merge would be added on top of a COMPLETE BDPT
        // sum and double-count every path it touches. No weights, no point merges — and then
        // `mk.surf` must stay 0 too, or every other technique's denominator would carry terms
        // for a strategy that is not running.
        const bool surfOn = photons && !photons->empty() && !photons->mis.empty() &&
                            photons->nEmitted > 0 && photons->radius > 0.0;
        if (surfOn)
            mk.surf = (double)photons->nEmitted * PI * photons->radius * photons->radius;
        const double mergeKappa = mk.beam;
        // FTRACE_J_HALF (diagnostic): 1 = connections only, 2 = merges only. Deliberately
        // does NOT touch the merge constants — both halves keep the SAME MIS weights they
        // have in a full render, so the two images sum to the full one.
        const int jHalf = jHalfMode();
        for (int py = y0; py < y1; ++py)
            for (int px = 0; px < camFilm.resX; ++px) {
                // Cooperative `-stop` / Ctrl-C, polled per pixel. WITHOUT this the camera
                // pass is uninterruptible: the light/beam pass at the top of this file polls
                // every 4096 paths, but nothing here did, so a `-stop` aimed at a mode-D/J
                // render could only ever land BETWEEN spp chunks. That is not a theoretical
                // gap — mode J on `scenes/_fog_cornell.ftsl` spent over half an hour inside
                // its very first 1-spp chunk (the beam gather is far dearer per sample than a
                // mode-D connection), ignored `-stop` throughout, and could not be ended
                // without the force-kill this project forbids, while holding ftrace.exe open
                // against a rebuild. One relaxed atomic load per pixel is free next to a full
                // BDPT path.
                //
                // Returning leaves the rest of this thread's band at whatever it has so far;
                // the caller drops a stopped chunk rather than merging a partial one.
                if (ft::stopRequested()) return;
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
                                                   mk);
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
                            if (jHalf >= 2) continue;   // FTRACE_J_HALF=merges*: drop connections
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
                    if ((mergeOn || surfOn) && jHalf != 1) {
                        // The camera half of every merge weight, replayed ONCE for the whole
                        // subpath. misWeight's camera loop telescopes inward from the merge
                        // point; everything it accumulates strictly camera-side of eye[k] is
                        // independent of where along the segment a beam is hit, so it is
                        // summed here and read off per segment. (The recurrences mirror the
                        // light-side ones in traceLightBeamPass; see BeamMergeWeight.)
                        //
                        // ONE pair of accumulators serves BOTH merge kinds, and must: a beam
                        // merge and a point merge on the same camera subpath compete for the
                        // same paths, so each one's denominator has to see the other's
                        // camera-side terms. That is also why `segSumM` is pre-scaled here by
                        // both kappas rather than left primed the way the light side leaves
                        // its two sums — the camera pass knows both constants, the light pass
                        // knows neither.
                        const double lamCam = hb.lam[0];
                        const PatTables tabs = scene.patTables();
                        if (mk.any()) {
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
                                                    eye[(size_t)k - 1].pdfRev, lamCam).scale(mk)
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
                                // A strand is never a light vertex (randomWalk, 0.365.0): every
                                // camera-side alternative from eye[k] inward puts it there.
                                const bool fiberK = !eye[(size_t)k].delta && eye[(size_t)k].mat &&
                                                    isFiberMat(*eye[(size_t)k].mat);
                                segSumC[(size_t)k] = fiberK ? 0.0 : gate + carry;
                                segSumM[(size_t)k] = fiberK ? 0.0 : eK + carryM;
                            }
                        }
                        TrRay camTr;
                        for (const PathSeg& sg : segs) {
                            if (!mergeOn) break;
                            if (!(sg.beta > 0.0)) continue;
                            BeamMergeWeight w1;
                            w1.scene = &scene; w1.bm = beams; w1.sg = &sg;
                            w1.lamCam = lamCam; w1.kappa = mergeKappa;
                            w1.kappaS = mk.surf;
                            w1.tabs = &tabs;
                            w1.raw = (jHalf == 3);
                            if (jHalf) { w1.diag = &jDiag(); jDiag().segs.fetch_add(1, std::memory_order_relaxed); }
                            // Hoisted out of the per-hit weight: the ray/bounds clip and the
                            // spectral sigma_t lookup are constants of the SEGMENT, and a
                            // dense medium hands one segment hundreds of hits. Built
                            // unconditionally so `w1.camTr` is never null — cheap (one clip
                            // per medium) and the alternative is a dangling read the day the
                            // `mergeKappa == 0 => misOf() is null` coupling stops holding.
                            camTr.build(scene, sg.o, sg.d, sg.tMax, lamCam, tabs);
                            w1.camTr = &camTr;
                            if (mk.any()) {
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
                                // ...and the POINT merge at eye[k], for the case the segment
                                // leaves a SURFACE into the medium. Its eta is kappaS *
                                // pdfRev(eye[k]) outright (mergeEtaPrimeSurf), so unlike its
                                // beam twin above there is no geometry to gather: the whole
                                // merge-point-independent coefficient IS kappaS. Mutually
                                // exclusive with etaKCoef — a vertex is a medium point or a
                                // surface, never both — but summed rather than branched so
                                // the day a third kind appears nothing here has to change.
                                if (k >= 1 && k < (size_t)nE && surfMergeSite(vk))
                                    w1.etaKSurf = mk.surf;
                            }
                            Vec3 m = gatherPhotonBeamsW(scene, mats, *beams, sg.o, sg.d,
                                                        sg.tMax, sg.aGlass, rng, w1);
                            if (m.x != 0.0 || m.y != 0.0 || m.z != 0.0)
                                camFilm.add(px, py, m * sg.beta);
                        }

                        // ---- POINT MERGES (the half folded in from mode U) --------------
                        // The same estimator mode U calls vertex merging, now MIS-combined
                        // with mode J's connections AND its beam merges in one denominator —
                        // which is the whole reason for folding U in here rather than running
                        // the two modes side by side. A camera path that crosses a cloud and
                        // lands on a wall is competed for by all three, and only a single
                        // weight can partition that unity.
                        //
                        // Per VERTEX, not per segment: a point merge happens where the camera
                        // walk actually landed, so there is no ray to march and no rng draw —
                        // which also keeps the "merges off == mode D bit-for-bit" gate intact
                        // for free.
                        if (surfOn) {
                            const double vmNorm = 1.0 / mk.surf;
                            for (int k = 1; k < nE; ++k) {
                                const Vertex& vk = eye[(size_t)k];
                                // The site predicate is `surfMergeSite` and MUST be, because
                                // the light pass stored photons by exactly it: a site gathered
                                // here but never stored under-weights every competing
                                // technique, and one stored but not gathered double-counts.
                                if (!surfMergeSite(vk) || !(vk.beta > 0.0) || !vk.mat) continue;
                                const Vertex& vp = eye[(size_t)k - 1];
                                Vec3 woCam = vp.p - vk.p;          // toward the camera side
                                const double d2 = dot(woCam, woCam);
                                if (!(d2 > 0.0)) continue;
                                woCam = woCam * (1.0 / std::sqrt(d2));
                                SurfMergeWeight sw;
                                sw.kappaB       = mk.beam;
                                sw.kappaS       = mk.surf;
                                sw.invPdfFwdK   = 1.0 / misRemap0(vk.pdfFwd);
                                sw.gateD1       = vp.delta ? 0.0 : 1.0;
                                sw.cosPrev      = vp.onSurface()
                                                ? std::fabs(dot(vp.ns, woCam)) : 1.0;
                                sw.invDistPrev2 = 1.0 / d2;
                                sw.invPdfFwdKm1 = 1.0 / misRemap0(vp.pdfFwd);
                                // The merge AT eye[k-1], less its pdfRev. Passing pLight = 1
                                // is what turns `mergeEtaPrime`'s primed eta into the bare
                                // coefficient — legitimate here (and not in the beam gather)
                                // because BOTH of eye[k-1]'s neighbours are known: the merge
                                // site is eye[k] itself, so there is no per-photon geometry.
                                sw.etaKm1Coef   = (k >= 2)
                                    ? mergeEtaPrime(scene, vk.p, vp, eye[(size_t)k - 2].p,
                                                    1.0, lamCam).scale(mk)
                                    : 0.0;
                                sw.segSumC      = segSumC[(size_t)k - 1];
                                sw.segSumM      = segSumM[(size_t)k - 1];
                                sw.camVert      = k;
                                sw.maxDepth     = maxDepth;
                                const Vec3 ngo = (dot(vk.ng, vk.ns) >= 0.0) ? vk.ng
                                                                            : vk.ng * -1.0;
                                Vec3 mergeXYZ{0, 0, 0};
                                photons->query(vk.p, [&](int idx) {
                                    const SurfPhoton& ph = photons->pts[(size_t)idx];
                                    const SurfMis* lm = photons->misOf((size_t)idx);
                                    if (!lm) return;      // no weight: refuse (see misOf)
                                    const double lam = (double)ph.lambda;
                                    double fCam = bsdfF(*vk.mat, vk.ns, woCam, ph.wo, lam,
                                                        scene, &vk.hit);
                                    if (!(fCam > 0.0)) return;
                                    // Gather-side shading-normal correction: the density
                                    // estimate reads flux per GEOMETRIC area while every
                                    // strategy it is MIS-combined with integrates against the
                                    // shading cosine. Exactly 1 on flat geometry.
                                    fCam *= vmGatherCorr(ph.wo, vk.ns, ngo);
                                    const double pdfDirRev = bsdfPdf(*vk.mat, vk.ns, woCam,
                                                                     ph.wo, lam, scene,
                                                                     &vk.hit);
                                    const double pdfDirFwd = bsdfPdf(*vk.mat, vk.ns, ph.wo,
                                                                     woCam, lam, scene,
                                                                     &vk.hit);
                                    const double w = sw(*lm, pdfDirRev, pdfDirFwd);
                                    if (!(w > 0.0)) return;
                                    mergeXYZ = mergeXYZ + Vec3(ph.cx, ph.cy, ph.cz) *
                                                          (w * fCam * (double)ph.beta);
                                });
                                if (mergeXYZ.x != 0.0 || mergeXYZ.y != 0.0 ||
                                    mergeXYZ.z != 0.0)
                                    camFilm.add(px, py, mergeXYZ * (vk.beta * vmNorm));
                            }
                        }
                    }
                }
            }
    }
};

} // namespace bdpt
