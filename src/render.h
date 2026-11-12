// Forward photon tracing.
//   Model A: physical camera. At every surface vertex, connect through a sampled
//            point on the finite lens pupil, refract through the thin lens onto the
//            film cell, and splat (next-event estimation of the lens). Gives real
//            depth of field / bokeh and converges (unlike waiting for photons to
//            physically thread the aperture, which is model C).
//   Model B: the apertureR -> 0 pinhole limit of A: connect to a single point and
//            splat. Infinitely sharp (no DOF), fastest.
//   Model C: brute-force oracle. Only photons that physically fly through the lens
//            pupil are caught (camera.h catchPhoton). Unbiased but very slow.
// (A legacy flat "contact sensor" wall — deposit()/Scene::sensor — still exists for
// irradiance-map diagnostics but is no longer wired to any camera model.)
//
// Energy bookkeeping (absorbed/escaped/residual) tracks the PHOTON's own energy
// only. Model-B splats are side-channel measurements and are intentionally NOT
// counted as energy sinks, so the conservation test stays valid in both modes.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <algorithm>
#include <complex>
#include "scene.h"
#include "camera.h"
#include "photonmap.h"
#include "photonbeams.h"   // view-independent volume cache for mode M (photon beams)
#include "causticaim.h"    // Jensen projection map: aimed emission for the caustic pass
#include "medium_stack.h"
#include "grin.h"     // shared gradient-index (GRIN) Eikonal marcher
#include "hero.h"     // hero-wavelength spectral sampling (kHeroC)
#include "hair_shade.h" // fiber BCSDF bridge (MatType::Hair, TODO §P3)
#include "interference.h" // thin-film / multilayer reflectance, clamp01

struct EnergyReport {
    double emitted = 0, absorbed = 0, sensor = 0, escaped = 0, residual = 0;
};

// ---- SPECTRAL-FOLD DIAGNOSTICS (temporary; FTRACE_FOLDDIAG=1) --------------------------
// Attributes every beam that COULD have folded (its medium is achromatic, so the gather-time
// tail is flat) but did not, to the exact event that retired the path's fold. The question
// these counters answer is which of two things makes up the residual after 0.255.0: paths
// that lost the fold to something genuinely chromatic (correct, and only a weighted bundle
// can help), or surface folds that `foldWorthIt` DECLINED (a tunable guard). Those two land
// in different rows, so one render separates them.
enum FoldKill {
    FK_None = 0,        // still folded when it reached the deposit
    FK_NeverBorn,       // the path never had a fold to lose
    FK_ChromaMedium,    // scattered in a medium whose sigma_t/phase is wavelength-dependent
    FK_GlassAbsorb,     // Beer-Lambert inside a dielectric: beta itself became spectral
    FK_Grin,            // gradient-index arc: the GEOMETRY is a function of lambda
    FK_Layered,         // coat: peaked Airy/Fresnel, iridescence is the point
    FK_Specular,        // dispersive refraction / grating / thin film / fluorescence / hair
    FK_DeclineTransmit, // DiffuseTransmit: foldWorthIt said the fold would cost variance
    FK_DeclineDiffuse,  // Diffuse albedo:  foldWorthIt said the fold would cost variance
    FK_DeclineGlossy,   // Glossy albedo:   foldWorthIt said the fold would cost variance
    FK_ZeroWeight,      // rho == 0 at the surviving lobe: T would divide by zero
    FK_COUNT
};
inline const char* foldKillName(int k) {
    static const char* n[FK_COUNT] = {"folded", "never-born", "chroma-medium", "glass-absorb",
                                      "grin", "layered", "specular", "decline-transmit",
                                      "decline-diffuse", "decline-glossy", "zero-weight"};
    return (k >= 0 && k < FK_COUNT) ? n[k] : "?";
}
// Set by tracePhoton at each retirement, read by emitBeams at the deposit. Thread-local, so
// the histogram is the only shared state.
inline thread_local int g_foldKill = FK_None;
inline std::atomic<uint64_t> g_foldKillHist[16][FK_COUNT];
inline bool foldDiagOn() {
    static const bool on = [] {
        const char* s = std::getenv("FTRACE_FOLDDIAG");
        return s && *s && *s != '0';
    }();
    return on;
}
// FTRACE_FOLDFORCE=1 makes `foldWorthIt` always say yes, so a single render measures how much
// of the residual the variance guard is responsible for. Diagnostic only — the guard exists
// because an unguarded fold fireflies on a peaked albedo.
inline bool foldForceOn() {
    static const bool on = [] {
        const char* s = std::getenv("FTRACE_FOLDFORCE");
        return s && *s && *s != '0';
    }();
    return on;
}
// FTRACE_NOSURFFOLD=1 makes `foldWorthIt` always say NO, which retires the spectral claim at
// every surface and so restores the pre-0.255.0 (mode `M`) / pre-0.257.0 (mode `J`) rule
// exactly. It is the A/B switch the surface fold is measured with: one binary, one seed, one
// `-spp`, two runs, so the only thing that differs between the two images is the fold itself.
// Without it the arms have to be two binaries or two `-time` budgets, and a `-time` budget
// varies the sample count run to run by more than the effect being measured.
inline bool foldNoSurfOn() {
    static const bool on = [] {
        const char* s = std::getenv("FTRACE_NOSURFFOLD");
        return s && *s && *s != '0';
    }();
    return on;
}

// ---- IS FOLDING THIS FACTOR WORTH IT? --------------------------------------------------
//
// Folding removes the chromatic variance of CIE(lambda_h) but introduces a 1/f(lambda_h)
// in its place, and for a strongly PEAKED factor — a saturated red wall sampled at a green
// hero wavelength — that trade is a loss: the surviving photon carries a huge T and turns
// into a firefly. Both second moments are computable in closed form from the quadrature we
// already have, so decide by comparing them rather than by a hand-tuned threshold:
//
//   unfolded  E[X^2] = E_lam[ f(lam) |CIE(lam)|^2 ]      ~  K * sum_k f_k |F_k|^2
//   folded    E[Y^2] = E_lam[1/f] * |E_lam[CIE f]|^2     ~ (1/K sum_k 1/f_k)
//                                                          * | sum_k f_k F_k |^2
//
// with F_k = em.foldCie[k] (which already carries the bin's 1/K of the emission mass).
// For a CONSTANT f, Cauchy-Schwarz makes E[Y^2] <= E[X^2] unconditionally, so a neutral
// surface always folds — which is the case the artifact lives in.
//
// THE VERDICT MUST NOT DEPEND ON lambda_hero. If it did, the fold/no-fold split would
// correlate with the wavelength and the mixture would stop being unbiased (P(fold) *
// E[CIE f] != integral over the folded subset). Everything the test reads — the per-bin
// factors and the emitter's table — is a function of the surface and the emitter alone,
// never of lambda_h, which is what keeps the estimator exact.
//
// A FREE FUNCTION rather than a lambda inside one tracer, because BOTH forward tracers
// need the identical verdict: `tracePhoton` below (mode M) and `randomWalk` in bdpt.h
// (mode J's beam pass) deposit into the same `PhotonBeam` records, read back by the same
// gather. If their fold/no-fold rules could drift apart, two beams in one bank would
// disagree about what `power`, `cieA` and `wS` mean. One definition is the only way that
// stays true.
inline bool foldWorthIt(const Emitter& em, const double* fk, int K) {
    if (foldNoSurfOn()) return false;      // diagnostics: restore the retire-at-any-surface rule
    double A = 0.0, invF = 0.0;
    Vec3 M{0, 0, 0};
    for (int k = 0; k < K; ++k) {
        if (!(fk[k] > 0.0)) return false;     // a zero bin makes E[1/f] infinite
        const Vec3& F = em.foldCie[k];
        A += fk[k] * (F.x * F.x + F.y * F.y + F.z * F.z);
        invF += 1.0 / fk[k];
        M += F * fk[k];
    }
    A *= (double)K;
    const double B = (invF / (double)K) * (M.x * M.x + M.y * M.y + M.z * M.z);
    // FTRACE_FOLDFORCE (diagnostics): say yes unless the fold is UNDEFINED — the zero bin
    // above still returns false — so a single render measures how much of the residual
    // "coloured bars" the variance guard itself is responsible for.
    if (foldForceOn()) return true;
    return B <= A;
}

// Power-cosine lobe around a mirror direction (rough specular), from two CANONICAL
// uniforms rather than an rng. roughness in [0,1]: 0 -> sharp mirror, 1 -> broad.
//
// Split out of sampleGlossy so a DETERMINISTIC caller (mode W, which has no rng to draw
// from without reintroducing noise) can drive exactly the same lobe off a low-discrepancy
// lattice. The polar coordinate is `cosT = u1^(1/(e+1))`, so **u1 == 1 is exactly the
// mirror direction** — that is what lets mode W's sample 0 reproduce the old
// mirror-direction-only behaviour bit-for-bit, and it is why a deterministic caller should
// *complement* its sequence (1 - radicalInverse) instead of Cranley-Patterson rotating it.
// The u1 == 1 case returns `mdir` verbatim, skipping the normalize() below, whose last-bit
// rescale would otherwise spoil that bit-identity.
inline Vec3 glossyDirUV(const Vec3& mdir, double roughness, double u1, double u2) {
    if (u1 >= 1.0) return mdir;              // exact mirror (rng.uniform() is [0,1), so
                                             // only a deterministic caller reaches this)
    double rr = roughness < 1e-3 ? 1e-3 : roughness;
    double e = 2.0 / (rr * rr) - 2.0; if (e < 0) e = 0;
    double cosT = std::pow(u1, 1.0 / (e + 1.0));
    double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
    double phi = 2.0 * PI * u2;
    Vec3 t, b; onb(mdir, t, b);
    return normalize(t * (sinT * std::cos(phi)) + b * (sinT * std::sin(phi)) + mdir * cosT);
}

// Stochastic form: two draws off the caller's stream, in that order.
inline Vec3 sampleGlossy(const Vec3& mdir, double roughness, Pcg32& rng) {
    // Sequenced into locals deliberately: passing rng.uniform() twice as arguments would
    // leave the draw order unspecified and desynchronise the stream.
    double u1 = rng.uniform(), u2 = rng.uniform();
    return glossyDirUV(mdir, roughness, u1, u2);
}

// --- Caustic classification: what a photon vertex does to a beam of light -------------
//
// Mode M splits its deposit into a GLOBAL map and a CAUSTIC map (Jensen's two-map scheme),
// because the two populations want completely different gather radii: diffuse illumination
// is smooth and low-density and wants a wide kernel; a caustic is a thin, high-contrast
// concentration and a kernel sized for the former erases it. Which map a deposit lands in
// is decided by the path that reached it — the classic L·S⁺·D regular expression, i.e. "at
// least one FOCUSING vertex and no SCATTERING one since the light".
//
// So every non-diffuse interaction is classified into three kinds:
//
//   FOCUS    — a deterministic (or near-deterministic) deflection that PRESERVES the beam's
//              coherence and can therefore concentrate it: refraction through a gem,
//              a mirror, a thin-film/multilayer interface, a grating order, a beam
//              splitter, and a glossy lobe tight enough to still focus.
//   SCATTER  — a wide, memory-destroying redirect. Any focus the beam had is gone, so a
//              subsequent deposit is ordinary indirect light, not a caustic: a rough glossy
//              lobe, a fluorescent re-emission (cosine-distributed, and wavelength-shifted
//              on top), a hair BCSDF, an analog medium collision, and of course any diffuse
//              vertex.
//   NEUTRAL  — no deflection at all, so the classification is unchanged either way: a
//              `filter` gel is a coloured absorber the photon passes straight through.
//
// The glossy threshold is the one judgement call; it is kCausticGlossRoughness, and it lives
// in photonmap.h (with the argument for its value) so that the CUDA translation unit — which
// does not include this header — reads the SAME number. Two threshold constants would mean a
// CPU and a GPU render of one scene classify glossy vertices differently and disagree on the
// image, with no symptom beyond "the GPU looks wrong".
enum PhotonVertexKind { PV_NEUTRAL = 0, PV_FOCUS = 1, PV_SCATTER = 2 };

inline int photonVertexKind(const Scene& scene, const Material& m, const Hit& h) {
    switch (m.type) {
        case MatType::Dielectric:
        case MatType::Mirror:
        case MatType::ThinFilm:
        case MatType::Multilayer:
        case MatType::Grating:
        case MatType::HalfMirror:
            return PV_FOCUS;
        case MatType::Glossy:
            return (materialRoughness(scene, m, h) <= kCausticGlossRoughness) ? PV_FOCUS
                                                                              : PV_SCATTER;
        case MatType::Filter:
            return PV_NEUTRAL;                  // straight through: direction untouched
        default:
            return PV_SCATTER;                  // Fluorescent, Hair, and the diffuse family
    }
}

// --- Fluorescence interaction (shared by the forward tracer and -checkfluoro) --
// A fluorescent surface has two competing channels: elastic diffuse reflection
// (albedo rho, wavelength preserved) and dye excitation (prob aEff = min(eps,
// 1-rho) so the channels never exceed unity, preserving energy). Excited photons
// re-radiate with probability Q at a Stokes-shifted wavelength drawn from M.
inline void fluoroWeights(const Material& m, double lambda, double& rho, double& aEff) {
    rho = clamp01(m.reflect(lambda));
    double eps = clamp01(m.fluoAbsorb(lambda));
    aEff = std::min(eps, std::max(0.0, 1.0 - rho));
}

enum class FluoroEvent { Elastic, Reemit, Absorb };
struct FluoroResult { FluoroEvent event; double lambdaOut; };

// Stochastically resolve a fluorescent interaction for an incoming photon at
// lambdaIn. Elastic -> reflect at lambdaIn; Reemit -> re-radiate at lambdaOut~M;
// Absorb -> photon lost (dye heat / non-excited fraction). Throughput weight is
// unchanged in all surviving branches (the branch probabilities carry the
// reradiation efficiency, and M/pdf cancels for the sampled lambdaOut).
inline FluoroResult fluoroInteract(const Material& m, double lambdaIn, Pcg32& rng) {
    double rho, aEff; fluoroWeights(m, lambdaIn, rho, aEff);
    double u = rng.uniform();
    if (u < rho)          return {FluoroEvent::Elastic, lambdaIn};
    if (u < rho + aEff) {
        if (rng.uniform() >= m.fluoYield) return {FluoroEvent::Absorb, 0.0};
        double pf; double lp = m.fluoEmitSampler.sample(rng, pf);
        return {FluoroEvent::Reemit, lp};
    }
    return {FluoroEvent::Absorb, 0.0};
}

// Interface (coat) reflectance for a MatType::Layered surface at a hit. Returns the
// unpolarised power reflectance R in [0,1] used as the reflect-vs-enter probability:
//   coatModel 2 (manual): the constant coatSpecular (angle/wavelength independent).
//   coatModel 1 (thinfilm): thin-film Airy R with film index filmIor(lambda) over the body's
//     effective index m.ior(lambda) — iridescent coat (transparent, so 1-R enters).
//   coatModel 0 (fresnel): plain dielectric Fresnel from m.ior(lambda) at cosI.
// Shared by the forward and backward tracers so both split the photon identically.
inline double layeredCoatReflectance(const Scene& scene, const Material& m, const Hit& h,
                                     const Vec3& d, double lambda) {
    if (m.coatModel == 2) return clamp01(m.coatSpecular);
    bool entering = dot(d, h.ng) < 0.0;
    Vec3 nl = entering ? h.ng : -h.ng;
    double cosI = clamp01(-dot(d, nl));
    if (m.coatModel == 1) {                          // thin-film Airy coat
        double thickness = materialFilmThickness(scene, m, h);
        double ns = m.ior(lambda);                   // effective index below the film
        return clamp01(thinFilmReflectance(1.0, m.filmIor(lambda), ns, 0.0, thickness, cosI, lambda));
    }
    double n1 = 1.0, n2 = m.ior(lambda);             // Fresnel dielectric
    double eta = n1 / n2;
    double sin2t = eta * eta * (1.0 - cosI * cosI);
    if (sin2t >= 1.0) return 1.0;                     // TIR (only from inside)
    double cosT = std::sqrt(1.0 - sin2t);
    double rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
    double rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
    return clamp01(0.5 * (rs * rs + rp * rp));
}

// --- The glossy lobe, and a rough interference reflector ----------------------------------
// Power-cosine glossy exponent for a roughness, the lobe sampleGlossy draws from.
inline double glossyExponent(double roughness) {
    double rr = roughness < 1e-3 ? 1e-3 : roughness;
    double e = 2.0 / (rr * rr) - 2.0;
    return e < 0 ? 0 : e;
}

// The interference reflectance an iridescent Glossy (Material::iridescence) weights its lobe
// by, at `lambda`, for light meeting the coating at the cosine `cosH`: the angle each direction
// of the pair makes with their half-vector, i.e. at the microfacet that reflects one into the
// other. Symmetric in the pair, so the factor is the same whichever side the light is on; as
// roughness goes to 0 the half-vector becomes the normal and this becomes the specular
// coating's own reflectance. Its incident medium is the specular coating's: the one the ray met
// it from (Hit::extMat) where both carry an explicit `priority`, else air (exteriorIor's rule).
// One difference remains: the rough lobe is two-sided, like every Glossy (the specular one,
// being opaque, is black from inside its body). 1 on a plain glossy. `h` (may be null, read as
// air) supplies a mapped thickness and the incident medium.
inline double iridescenceR(const Scene& scene, const Material& m, const Hit* h, double lambda,
                           double cosH) {
    if (m.iridescence == 0) return 1.0;
    const int e = h ? h->extMat : -1;
    const double n0 = (e >= 0 && m.hasPriority() && scene.mats[(size_t)e].hasPriority())
                          ? scene.mats[(size_t)e].ior(lambda) : 1.0;
    return coatingReflectance(m, (h && m.iridescence == 1) ? materialFilmThickness(scene, m, *h)
                                                          : m.filmThickness, lambda, cosH, n0);
}

// The cosine iridescenceR takes for the pair (a, b), both unit: each one's cosine with their
// half-vector, which is |a + b| / 2.
inline double halfVectorCos(const Vec3& a, const Vec3& b) {
    const double c = 0.5 * length(a + b);
    return c < 1.0 ? c : 1.0;
}

// A direction drawn from a glossy lobe, and the factor the draw leaves beyond the lobe's `reflect`
// slot: f * cos / pdf = reflect(lambda) * R(lambda), R being an iridescent lobe's interference
// reflectance at the pair's half-vector and exactly 1 on a plain lobe. Every sampler of a Glossy
// material's lobe draws through drawGlossy (or glossyAt, for a direction a lattice chose), so the
// factor comes with the direction, and so does every fold that re-weighs it (mode J's spectral
// carrier). How to take R -- multiplied in, a roulette, a roulette on a bundle's largest -- is the
// caller's; `irid` says whether R can differ from 1, for a roulette or a per-wavelength loop
// that would otherwise run on a plain lobe. A layered coat's sheen lobe draws sampleGlossy
// itself: a coat never interferes. The draw keeps the hit's address, so it takes no temporary.
struct GlossyDraw {
    Vec3 o;              // the direction; the caller ends the path where it is not above the surface
    bool irid = false;   // the lobe interferes
    double cH = 1.0;     // the half-vector cosine R reads (1 on a plain lobe)
    const Scene* scene = nullptr;
    const Material* m = nullptr;
    const Hit* h = nullptr;
    double R(double lambda) const { return irid ? iridescenceR(*scene, *m, h, lambda, cH) : 1.0; }
};
// `o` drawn from m's lobe at h some other way; `wo` the direction back along the incoming ray.
inline GlossyDraw glossyAt(const Scene& scene, const Material& m, const Hit& h, const Vec3& wo,
                           const Vec3& o) {
    GlossyDraw g;
    g.o = o;
    g.irid = m.iridescence != 0;
    if (g.irid) g.cH = halfVectorCos(wo, o);
    g.scene = &scene; g.m = &m; g.h = &h;
    return g;
}
// A draw from m's lobe about the mirror of the incoming direction `dIn` in the normal `n`; `wo`
// the direction back along it, as the caller holds it.
inline GlossyDraw drawGlossy(const Scene& scene, const Material& m, const Hit& h, const Vec3& dIn,
                             const Vec3& n, const Vec3& wo, Pcg32& rng) {
    return glossyAt(scene, m, h, wo, sampleGlossy(reflect(dIn, n), materialRoughness(scene, m, h), rng));
}
GlossyDraw glossyAt(const Scene&, const Material&, const Hit&&, const Vec3&, const Vec3&) = delete;
GlossyDraw drawGlossy(const Scene&, const Material&, const Hit&&, const Vec3&, const Vec3&,
                      const Vec3&, Pcg32&) = delete;

// The glossy BSDF for the pair (wo, wi), both unit and pointing away from the surface, the lobe
// mirrored about `wo`: r * lobe / cos(wi), in bsdf_eval.h's convention (see its header), r the
// `reflect` slot -- times the interference reflectance on an iridescent glossy. 0 unless both
// lie above the shading normal `ns` and wi inside the lobe. `h` (may be null) supplies mapped
// slots; without it the material's constants are read.
inline double glossyF(const Scene& scene, const Material& m, const Hit* h, const Vec3& ns,
                      const Vec3& wo, const Vec3& wi, double lambda) {
    const double cosWi = dot(wi, ns), cosWo = dot(wo, ns);
    if (cosWi <= 0 || cosWo <= 0) return 0.0;
    double r = h ? clamp01(reflectSlot(scene, m, *h, lambda)) : clamp01(m.reflect(lambda));
    if (m.iridescence) r *= iridescenceR(scene, m, h, lambda, halfVectorCos(wo, wi));
    const double e = glossyExponent(h ? materialRoughness(scene, m, *h) : m.roughness);
    // The mirror direction of the outgoing ray about ns, as sampleGlossy is given it.
    const Vec3 mdir = reflect(wo * -1.0, ns);
    const double cosLobe = dot(wi, mdir);
    if (cosLobe <= 0) return 0.0;
    const double lobe = (e + 1.0) / (2.0 * PI) * std::pow(cosLobe, e);
    return r * lobe / cosWi;
}

// THE GLOSSY LOBE ON A WALK FROM THE LIGHT. glossyF divides by the cosine on the light's side, so
// the lobe is not reciprocal, and a walk from the light carries its adjoint, glossyF with the pair
// swapped (bsdf_eval.h bsdfFAdjoint). Drawing the outgoing `wOut` from the lobe mirrored about the
// arrival `wIn` (pointing back toward the light), that walk's factor f*cos/pdf is therefore r times
// this ratio, cos(wOut, ns) / cos(wIn, ns), where a camera walk's is r alone. 1 at the mirror
// direction; 0 for an arrival from below the shading normal `ns`, whose adjoint is 0. It can exceed
// 1: the lobe conserves energy toward the camera, not toward the light. Device twin
// dGlossyAdjointRatio.
inline double glossyAdjointRatio(const Vec3& ns, const Vec3& wIn, const Vec3& wOut) {
    const double ci = dot(wIn, ns), co = dot(wOut, ns);
    return (ci > 0.0 && co > 0.0) ? co / ci : 0.0;
}

// Whether a glossy lobe met along `rayDir` reflects toward where the ray came from: only a camera
// side above the shading normal `ns`. glossyF -- and so bsdfF, every connection's value -- is 0 for
// wo below ns, the light walks end there (glossyAdjointRatio's 0) and BDPT's camera walk has no
// density there (bsdfPdf), so a unidirectional camera walk (modes R, W, M, S) ends there too. Its
// lobe sampler would otherwise go on and light a point no other strategy can: on a normal-mapped
// glossy room mode R read itself 1.044 with glossy NEE off, the lobe alone. Where ns == ng a ray
// meeting the surface from its front always passes. Device twin dGlossyCameraSide.
inline bool glossyCameraSide(const Vec3& rayDir, const Vec3& ns) { return dot(rayDir, ns) < 0.0; }

// THE SOFTENING A VERTEX LENDS THE LIGHT ITS CONTINUATION REACHES. The terminator's softening
// (shadowTerminatorG) belongs to the vertex next to a light (bdpt.h emitterSoftening), and it is the
// INTEGRAND's, not one strategy's: a camera walk's light sampling takes it toward the light (neeLight,
// neeEnv, fireNee), so the same light reached by the walk's own continuation -- an emitter hit after a
// glossy bounce, the sky after a diffuse or glossy one, the sun, a fire along the segment -- must
// take it too, or the two halves of the MIS weight sum different integrands on a tilted shading normal
// (mode R read itself 1.044 on a normal-mapped glossy room with its lobe alone, 1.022 on a sky-lit
// diffuse one with its escape alone). This is that factor for a vertex `h` (material `m`, its lobe
// resolved) whose continuation left along `d`: shadowTerminatorG toward it at a Lambertian or glossy
// vertex, 1 at a specular, two-sided or fiber one. Exactly 1 where ns == ng. Device twin
// dContSoftening.
inline double contSoftening(const Material& m, const Hit& h, const Vec3& d) {
    if (m.type != MatType::Diffuse && m.type != MatType::Fluorescent && m.type != MatType::Glossy)
        return 1.0;
    return shadowTerminatorG(d, h.n, orientedGeoN(h));
}

// A glossy photon vertex's camera splat (Renderer::camSplatAll and kin). A glossy lobe is not
// Lambertian, so its value is known only once the connection's direction is: the splats call
// f(wc) for the direction wc they found toward the camera point. `wl` points back along the
// photon's arrival; a particle vertex takes the ADJOINT BSDF (bsdf_eval.h bsdfFAdjoint), which
// for the glossy lobe is glossyF with the pair swapped -- divided by the cosine on the light's
// side, as energy conservation requires.
struct GlossySplat {
    const Scene* scene;
    const Material* m;
    const Hit* h;
    Vec3 ns;   // shading normal
    Vec3 wl;   // toward the previous, light-side vertex
    // A layered stack whose coat `m` is (its coatChild), or null. The coat's lobe then carries
    // the stack's coat reflectance toward `wc`: a camera-side walk takes the coat with R at its
    // own incidence, so that is the R its sheen is seen with (FWD-GLOSSY-INDIRECT).
    const Material* stack = nullptr;
    double f(const Vec3& wc, double lambda) const {
        const double v = glossyF(*scene, *m, h, ns, wc, wl, lambda);
        return stack ? v * layeredCoatReflectance(*scene, *stack, *h, -wc, lambda) : v;
    }
};

// Does a glossy vertex splat to the camera in the forward modes? Every lobe that has a finite
// value does. The one that has none, roughness pinned at 0, is mode B's analytic planar-mirror
// connection's to image (isPlanarMirrorMat); splatting it too would count it twice.
inline bool glossySplats(const Material& m) {
    return m.type == MatType::Glossy && !isPlanarMirrorMat(m);
}

// One camera + its film for the forward light-tracer's next-event splats. A photon
// path is camera-independent until the connect() splat, so `tracePhoton` takes a list
// of these and splats each diffuse/emitter/volume vertex to every target at once — one
// shared photon pass feeding N images instead of re-tracing per camera. In model B
// (pinhole splat) connect() draws no RNG, so adding cameras never perturbs the photon's
// RNG stream: a single-target trace is bit-identical to the old one-camera path, and an
// N-camera shared pass reproduces N independent single-camera renders exactly. Models A
// (finite-lens aperture sample) and C (forward catch) draw RNG / consume the photon per
// camera, so they stay single-target (nCam==1); the shared pass is model-B only.
struct CamTarget {
    const Camera* cam = nullptr;
    Film*         film = nullptr;
};

// The specular-sphere connectors (connectSpecularSphere / ...Inside) scan the SAME
// kSphScanN+1 fixed entry angles on every call, and each scan step used to pay a
// fresh cos+sin pair — the dominant transcendental cost of the mode-B glass-sphere
// splat. The angles never change, so evaluate them once at first use, with the same
// runtime std::cos/std::sin the scan itself called (NOT constant-folded — the loop
// variable keeps the compiler honest), making table reads bit-identical to the
// per-step evaluation they replace. Bisection refinement still computes live
// cos/sin (its midpoints are data-dependent).
inline constexpr int kSphScanN = 96;
struct SphScanTab { double c[kSphScanN + 1], s[kSphScanN + 1]; };
inline const SphScanTab& sphScanTab() {
    static const SphScanTab tab = [] {
        SphScanTab t;
        for (int i = 0; i <= kSphScanN; ++i) {
            double phi = -PI + (2.0 * PI) * i / kSphScanN;
            t.c[i] = std::cos(phi);
            t.s[i] = std::sin(phi);
        }
        return t;
    }();
    return tab;
}

struct Renderer {
    int maxBounce = 32;          // hard safety cap; Russian roulette normally
                                 // terminates paths well before this.
    // Off for mode P's blurred split (runCompositeProgressive): no camera splat at a vertex the
    // camera side draws -- a glossy lobe on a surface whose own material is specular-side
    // (isSpecularType of the hit's material, not of the child a mix resolves to: a mix's
    // glossy child is the forward layer's, as the backward layer files its samples), and the
    // analytic connections through a mirror or glass sphere or off a mirror plane
    // (camSpecularSplatAllVtxN) -- since the backward layer takes every sample whose first
    // event is a specular-side surface or the sky (BackwardRenderer::specFirstFilm), and the two
    // would count that light twice. A medium vertex's in-scatter stays: a sample scattered in a
    // medium before any surface is the forward layer's. The still composite needs no such
    // switch: it drops the forward film on a specular-side pixel wholesale.
    bool specularSideSplats = true;
    bool forwardCatch = false;   // model C: catch photons that physically fly through
                                 // the aperture (brute-force oracle, no connect/splat).
    bool lensMode     = false;   // model A: next-event splat through the finite lens
                                 // pupil (physical camera with depth of field).
    // Does this camera take the lens splat? Mode A always does; a camera with a PHYSICAL
    // multi-element lens does in every forward mode, since it has no pinhole limit to splat
    // to (its aperture is part of the glass): in mode B it images through its lens as in A.
    bool lensCam(const Camera& cam) const { return lensMode || cam.hasLens(); }
    bool diffraction = true;     // when false, MatType::Grating collapses to its m=0
                                 // (specular) order — a plain mirror (CLI -diffraction).
    bool useHero      = false;   // hero-wavelength sampling: each photon carries C
                                 // wavelengths (hero + C-1 stratified secondaries) down
                                 // one shared BVH walk, cutting chromatic noise. Gated ON
                                 // by the driver only when heroC>1 and the scene has no
                                 // media / GRIN (dispersive events de-hero mid-path).
    int  heroC        = hero::kHeroC; // number of wavelengths bundled per path when
                                 // useHero is on (hero + heroC-1 secondaries). Runtime-
                                 // configurable via -heroc N, clamped to [1, kHeroMax];
                                 // defaults to kHeroC. C==1 collapses to single-λ.
    bool heroSplit    = hero::gSplit; // SPLIT-AT-DISPERSION policy (-herosplit): at a
                                 // dispersive interface, fan the bundle out into C
                                 // monochromatic sub-paths (each refracting along its own
                                 // per-λ direction) instead of de-hero'ing to the hero
                                 // alone. Off by default; costs C× traversal past the
                                 // split but resolves chromatic spread geometrically.
    bool beamGather   = false;   // PHOTON-BEAMS gather for the shared multi-camera pass
                                 // (CLI -beams). When on and nCam>1, each camera samples
                                 // its OWN collision point along every medium beam segment
                                 // (its own RNG) instead of all cameras splatting the one
                                 // shared collision point. The expensive photon flight is
                                 // still traced once (shared, 1× cost), but the per-camera
                                 // splats are now decorrelated — so a volumetric flyby
                                 // (rainbow/fogbow/fog) gets INDEPENDENT per-frame noise
                                 // instead of one frozen speckle pattern baked into every
                                 // frame. Unbiased: each camera's resampled splat has the
                                 // same expectation as the shared point splat (the free-
                                 // flight collision pdf's transmittance cancels either way).

    // PHOTON-BEAM MULTIPLE SCATTERING (CLI -beams-order). Maximum scattering order the beam
    // paths carry: 1 = single scatter (the pre-0.199.0 behaviour), 0 = unlimited (bounded only
    // by -bounces). See the long note above the `-beams` block in tracePhoton for why LONG
    // beams make this a small change, and why truncating the beam at the sampled collision
    // instead would double-count transmittance.
    //
    // The counting: the photon's FIRST chord is already scattering order 1 (the gather point
    // on that beam IS one scattering event), so a photon that has scattered `nDone` times in a
    // beam medium is depositing order nDone+1. Allowing one more scatter buys order nDone+2,
    // so the cap admits it only while nDone + 2 <= beamOrderMax. `beamOrderMax == 1` therefore
    // permits no scatter at all and takes the pre-0.199.0 straight-crossing path verbatim —
    // which is what makes `-beams-order 1` bit-identical to 0.198.0.
    int beamOrderMax = pbeams::gOrderMax;   // 0 = unlimited (bounded only by -bounces)
    bool beamMSAllowed(int nDone) const {
        return beamOrderMax == 0 || (nDone + 2) <= beamOrderMax;
    }

    // Photon-map deposit (ROADMAP item 1 / mode M). When non-null, every diffuse-family
    // surface vertex ALSO appends a Photon record here (view-independent radiance cache).
    // A photon pass runs with nCam==0 (no camera splat) + photonDeposit set: paths bounce
    // and deposit, but energy goes into the map instead of onto a sensor. Null in modes
    // A/B/C, so their splat behaviour is byte-for-byte unchanged.
    PhotonBank* photonDeposit = nullptr;

    // CAUSTIC map deposit (mode M, Jensen's two-map scheme). When non-null, a deposit whose
    // path matches L·S⁺·D — at least one PV_FOCUS vertex and no PV_SCATTER vertex since the
    // light, see photonVertexKind above — goes HERE INSTEAD OF `photonDeposit`. The split is
    // strict, so the two maps partition the deposits and the gather is their plain sum: no
    // photon is counted twice and none is dropped. Null (the pre-0.199.7 behaviour) puts
    // everything in the global map, which is what mode S and any caller that does not want
    // the split get.
    PhotonBank* causticDeposit = nullptr;

    // ---- DEDICATED CAUSTIC PASS (Jensen's projection map; see causticaim.h) --------------
    // Non-null enables aimed emission and the balance-heuristic weighting that pairs with it.
    // Bound on BOTH passes: the main pass needs it to down-weight the caustic deposits the
    // aimed pass is also making, and it changes nothing else about the main pass (a photon
    // that lands nowhere near focusing geometry has rho = 0 and weight exactly 1).
    const caim::AimMap* aimMap = nullptr;
    // True only in the dedicated caustic pass: emission is drawn from the aim map instead of
    // from the emitter's own distribution. Costs the main pass no RNG draws at all, so a
    // render with the feature bound but the pass switched off stays bit-identical.
    bool aimEmission = false;
    // N_c / N_m — the caustic pass's photon count over the main pass's. The balance
    // heuristic's only free parameter, and it is not free: it is exactly the ratio the two
    // passes were actually run at.
    double aimMisRatio = 0.0;
    // Caustic-deposit weight for the photon currently being traced, w(x) in causticaim.h.
    // Mutable because every tracer here is const and this is per-photon scratch, exactly
    // like the RNG the callers thread through.
    mutable double causticW = 1.0;
    // Cross media STRAIGHT without storing beams. The caustic pass must transport photons by
    // the same rules as the main pass or the two are not estimating the same integrand and
    // the MIS combination is meaningless — and with `-beams` the main pass crosses media
    // straight. It must not store beams though: the beam map is the main pass's, normalised
    // by the main pass's nEmitted.
    bool beamStraightOnly = false;

    // Append a photon record at a diffuse/translucent vertex (no-op when the map is off). `n`
    // is the true normal on the side the photon arrived from (orientedGeoN), which a gather
    // compares with its own (see Photon in photonmap.h). The incident direction is deliberately
    // NOT stored: the density estimate is Lambertian, so no gather has ever read it. Callers
    // store nothing on a light block's own surface (reflectsNothing): no gather there sees
    // anything, and every other surface within the radius would take it (known-issues
    // J-LAMP-PHOTONS): the photon walks end there before any deposit.
    // `caustic` routes the record to the caustic bank when one is bound (see above), and
    // applies this photon's caustic MIS weight — 1.0 unless a dedicated caustic pass is
    // running alongside, in which case the two passes share the deposit between them.
    void depositPhoton(const Vec3& p, const Vec3& n, double lambda, double beta,
                       bool caustic = false) const {
        PhotonBank* bank = (caustic && causticDeposit) ? causticDeposit : photonDeposit;
        if (!bank) return;
        if (bank == causticDeposit && causticW != 1.0) {
            beta *= causticW;
            if (!(beta > 0.0)) return;
        }
        bank->push(p, n, (float)beta, (float)lambda);
    }

    // ---- Aimed emission + caustic MIS weight -------------------------------------------
    // Called once per photon, straight after the ordinary emission sample has been drawn.
    // In the MAIN pass it only *measures* that sample — computing rho = p_a/p_u and storing
    // the balance-heuristic weight in `causticW` — and draws no randomness, so the main
    // pass's photon set is untouched. In the AIMED pass it additionally RESAMPLES the half
    // of the emission the target actually constrains (the upstream disc point for a distant
    // emitter, the direction for a local one), overwriting `origin` / `dir` / `spotW`.
    //
    // `em == nullptr` means a volumetric blackbody birth: an isotropic direction from a point
    // inside the fire, which is the cone case with p_u = 1/(4*pi).
    //
    // Returns false when the sample carries no light — an aimed direction outside a spot's
    // outer cone, below an area emitter's horizon, or outside the upstream disc that IS a
    // distant emitter's entire phase space. Those are not rejections that need compensating:
    // p_u is genuinely zero there, so the contribution being discarded is zero.
    bool applyCausticAim(const Scene& scene, const Emitter* em, Vec3& origin, Vec3& dir,
                         const Vec3& emitN, double& spotW, Pcg32& rng) const {
        causticW = 1.0;
        if (!aimMap || aimMap->empty()) return true;
        // rho = p_a/p_u. The default is 1, not 0: an emitter that CANNOT be aimed (a
        // collimated one — its direction is a delta) is emitted by the caustic pass with the
        // ordinary sampler, so there the two strategies are identical and rho is exactly 1.
        // The balance heuristic then degenerates to splitting the deposit between two equal
        // passes, which is still exact.
        double rho = 1.0;
        const bool distant = em && (em->shape == EmitterShape::Env ||
                                    em->shape == EmitterShape::Sun);
        const bool collimated = em && em->collimated && !distant;
        if (collimated) {
            // nothing to aim: rho stays 1
        } else if (distant) {
            // --- upstream origin disc -------------------------------------------------
            Vec3 t, b; onb(dir, t, b);
            const Vec3 base = scene.sceneCenter - dir * scene.lightRadius();   // the photon's disc
            const caim::DiscAim da = caim::discAim(*aimMap);
            if (!(da.sumR2 > 0.0)) return true;
            double x, y;
            if (aimEmission) {
                if (!caim::discSample(*aimMap, da, scene.sceneCenter, t, b,
                                      rng.uniform(), rng.uniform(), rng.uniform(), x, y))
                    return true;
                origin = base + t * x + b * y;
            } else {
                const Vec3 off = origin - base;
                x = dot(off, t); y = dot(off, b);
            }
            const double R = scene.lightRadius();
            if (x * x + y * y > R * R) {
                // Outside the disc the emitter delivers nothing at all, so p_u = 0 and the
                // whole contribution is zero — not a lost sample, a zero one. (Reachable
                // only from the aimed pass, and only when a target's bounding sphere pokes
                // marginally past the scene's own.)
                return !aimEmission;
            }
            int n = caim::discCount(*aimMap, scene.sceneCenter, t, b, x, y);
            if (aimEmission && n < 1) n = 1;   // we drew it from a disc, so it is in one
            rho = (double)n * R * R / da.sumR2;
        } else {
            // --- direction cone -------------------------------------------------------
            const caim::ConeAim ca = caim::coneAim(*aimMap, origin);
            if (!(ca.sumOmega > 0.0)) return true;
            if (aimEmission) {
                Vec3 w;
                if (!caim::coneSample(*aimMap, ca, origin,
                                      rng.uniform(), rng.uniform(), rng.uniform(), w))
                    return true;
                dir = w;
                if (em && em->shape == EmitterShape::Spot) {
                    const double ct = dot(dir, emitN);   // a spot's emitN: its axis at the photon's time
                    if (ct <= em->spotCosOuter) return false;     // outside the cone: p_u = 0
                    const double omegaOuter = 2.0 * PI * (1.0 - em->spotCosOuter);
                    spotW = spotFalloff(ct, em->spotCosInner, em->spotCosOuter)
                          * omegaOuter / em->spotOmega;
                } else if (em && (em->normalTilted ? scene.aimCosNow(*em, dir) <= 0.0
                                                   : dot(dir, emitN) <= 0.0)) {
                    return false;                                 // below the horizon: p_u = 0
                }
            }
            double pu;
            if (em && em->shape == EmitterShape::Spot) {
                const double ct = dot(dir, emitN);
                pu = (ct > em->spotCosOuter) ? 1.0 / (2.0 * PI * (1.0 - em->spotCosOuter)) : 0.0;
            } else if (em && em->normalTilted) {
                pu = scene.aimDirPdfNow(*em, dir);                // an aimed quad (Emitter::nGeom)
            } else if (em) {
                const double c = dot(dir, emitN);
                pu = (c > 0.0) ? c / PI : 0.0;                    // cosine hemisphere
            } else {
                pu = 1.0 / (4.0 * PI);                            // isotropic volumetric birth
            }
            if (!(pu > 0.0)) return !aimEmission;
            int n = caim::coneCount(*aimMap, origin, dir);
            if (aimEmission && n < 1) n = 1;
            rho = ((double)n / ca.sumOmega) / pu;
        }
        // Balance heuristic: w = N_m p_u / (N_m p_u + N_c p_a). See causticaim.h for why the
        // SAME weight is right for a photon from either pass.
        causticW = 1.0 / (1.0 + aimMisRatio * rho);
        return true;
    }

    // Photon-BEAM deposit (mode M with -beams). The surface map above cannot represent a
    // participating medium at all — no photon ever deposits inside one — so mode M renders
    // fog / rain / cloud / rainbow as nothing. When this is non-null the photon crosses
    // every medium in a STRAIGHT beam (the analog free-flight redirect is skipped, exactly
    // as in the mode-A/B `-beams` path) and appends the whole crossed SEGMENT here, which
    // any camera can later gather from. See photonbeams.h for the estimator and for why the
    // segment — not a point — is the right record. Null in every other mode.
    BeamBank* beamDeposit = nullptr;

    // SPECTRAL BEAMS (CLI -beamspec N). How many stratified wavelengths one deposited beam
    // carries. 1 = the classic monochromatic beam, bit-for-bit. See the long note above
    // PhotonBeam (photonbeams.h) for why a monochromatic LINE is so much worse than a
    // monochromatic point, and Renderer::tracePhoton for the (deliberately conservative)
    // rule that keeps a bundle alive only while the path is provably wavelength-independent.
    // The driver sets this to 1 for any scene whose media have chromatic extinction, since a
    // bundle's shared transmittance would then be wrong for its secondaries.
    int beamSpecC = 1;

    // ACHROMATIC-PATH BEAMS (CLI -beamachro, photonbeams.h). Scene-wide permission for the
    // mean-CIE fold: every medium's extinction is wavelength-independent, so a free flight
    // samples the same distance for every wavelength and the path's GEOMETRY — not merely its
    // colour — is lambda-invariant. Same predicate the driver applies to beamSpecC above, but
    // kept separately because this fold needs no `-beamspec > 1`: it stores no extra
    // wavelengths, so a `-beamspec 1` render gets it too. Set by the driver, not per photon
    // (beamSpectralOK scans every medium's spectra and must not run inside the trace).
    bool beamAchroOK = false;

    // The SAME scene-wide predicate, stated positively and independently of either CLI flag,
    // for the one caller that needs it when both are off: bdpt.h's light-beam pass. Mode J
    // draws lambda from the scene-wide emitSampler rather than from the chosen emitter's own
    // SPD, so before it can deposit a beam at all it has to convert its beta into the one
    // render.h's photon would have carried (bdpt.h, BeamSpectral) -- and that conversion rests
    // on the same claim these two flags rest on, that no free flight in this scene depends on
    // lambda. Mode M has no use for it: its photon is already born at the right density, so it
    // reads the predicate through `beamSpecC > 1` / `beamAchroOK` and never needs it alone.
    bool beamSpecOK = false;

    // Longest beam we will store when the photon escapes to infinity through an UNBOUNDED
    // medium, as a multiple of the scene radius. An unbounded medium clips to [0, 1e30], and
    // a 1e30-long AABB would swallow the whole BVH; transmittance has long since killed the
    // beam by a few scene radii anyway.
    static constexpr double kBeamFarScale = 8.0;

    // Model A: map a contact-sensor hit to a pixel and deposit.
    void deposit(const Sensor& s, Film& film, const Vec3& p, double lambda, double beta) const {
        Vec3 rel = p - s.origin;
        double uu = dot(rel, s.uAxis) / dot(s.uAxis, s.uAxis);
        double vv = dot(rel, s.vAxis) / dot(s.vAxis, s.vAxis);
        if (uu < 0 || uu >= 1 || vv < 0 || vv >= 1) return;
        int px = (int)(uu * film.resX), py = (int)(vv * film.resY);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * beta);
    }

    // --- Participating-media sampling helpers --------------------------------
    // These cover the homogeneous, bounded-homogeneous, and heterogeneous (density-
    // field) media in one place. A homogeneous medium keeps the exact analytic
    // behaviour and draws exactly the same RNG as before (bit-identical to the
    // pre-heterogeneous engine); a density field switches to delta / residual-ratio
    // tracking against the per-cell majorant grid (majorant.h) when the medium has one,
    // and to the single global majorant sigma_max = sigmaT(lambda) * densityMax when it
    // does not (an unbounded density field).
    //
    // WALKING THE MAJORANT GRID. Both estimators below step the ray cell by cell with a
    // 3D DDA and run their per-cell tracking loop over [tEnter, tExit] with THAT cell's
    // coefficients. The two share `majorantWalk`, which calls `body(cellIndex, t0, t1)`
    // for each cell the segment crosses and stops early when the body returns false. The
    // walk is in the grid's own AABB, which is the medium bound's AABB, so `clipToBounds`
    // has already trimmed the segment to it — but a ray can still start on/outside a face
    // by an epsilon, so the entry cell is clamped rather than assumed in range.
    template <class Body>
    static void majorantWalk(const MajorantGrid& g, const Vec3& o, const Vec3& dir,
                             double ta, double tb, Body&& body) {
        auto cellOf = [&](double t, int a[3]) {
            Vec3 p = o + dir * t;
            double f[3] = {(p.x - g.wmin.x) * g.invCell.x,
                           (p.y - g.wmin.y) * g.invCell.y,
                           (p.z - g.wmin.z) * g.invCell.z};
            int n[3] = {g.nx, g.ny, g.nz};
            for (int k = 0; k < 3; ++k) {
                int i = (int)std::floor(f[k]);
                a[k] = i < 0 ? 0 : (i >= n[k] ? n[k] - 1 : i);
            }
        };
        int c[3]; cellOf(ta, c);
        const double d[3] = {dir.x, dir.y, dir.z};
        const double cs[3] = {g.cell.x, g.cell.y, g.cell.z};
        const double lo[3] = {g.wmin.x, g.wmin.y, g.wmin.z};
        const int    nn[3] = {g.nx, g.ny, g.nz};
        int step[3]; double tNext[3], tDelta[3];
        for (int k = 0; k < 3; ++k) {
            if (d[k] > 1e-12) {
                step[k] = 1;
                tNext[k] = (lo[k] + (c[k] + 1) * cs[k] - (k == 0 ? o.x : k == 1 ? o.y : o.z)) / d[k];
                tDelta[k] = cs[k] / d[k];
            } else if (d[k] < -1e-12) {
                step[k] = -1;
                tNext[k] = (lo[k] + c[k] * cs[k] - (k == 0 ? o.x : k == 1 ? o.y : o.z)) / d[k];
                tDelta[k] = -cs[k] / d[k];
            } else {
                step[k] = 0; tNext[k] = 1e300; tDelta[k] = 1e300;
            }
        }
        double t0 = ta;
        for (;;) {
            int axis = (tNext[0] < tNext[1]) ? ((tNext[0] < tNext[2]) ? 0 : 2)
                                             : ((tNext[1] < tNext[2]) ? 1 : 2);
            double t1 = std::min(tNext[axis], tb);
            if (t1 > t0 && !body(g.idx(c[0], c[1], c[2]), t0, t1)) return;
            if (t1 >= tb) return;
            t0 = t1;
            c[axis] += step[axis];
            if (c[axis] < 0 || c[axis] >= nn[axis]) return;   // left the grid
            tNext[axis] += tDelta[axis];
        }
    }

    // Sample the next real collision along (o,dir) within [0,dMax]. Returns true and
    // sets tHit at a real scattering/absorption event; false if the photon reaches
    // dMax first. Delta (Woodcock) tracking for a heterogeneous medium: candidate
    // collisions at rate sigma_max, accepted as real with prob sigmaT(x)/sigma_max
    // (a rejected "null collision" just continues) — unbiased, throughput unchanged.
    // The majorant may be ANY upper bound on the local extinction, so the per-cell sup
    // (ctrl + res) is used where a grid exists: same answer, far fewer null collisions,
    // and a vacuum cell is skipped outright with no RNG draw at all.
    // `tabs` are the scene's grid:/scatter: tables, required (not defaulted) so a
    // density program that samples a measured volume can never be silently evaluated
    // without them — the two wrappers below are the only callers and both have a Scene.
    static bool sampleMediumCollision(const Medium& med, const Vec3& o, const Vec3& dir,
                                      double dMax, double lambda, Pcg32& rng, double& tHit,
                                      const PatTables* tabs) {
        double stBase = med.sigmaT(lambda);
        if (stBase <= 0.0) return false;
        double ta, tb;
        if (!med.clipToBounds(o, dir, 0.0, dMax, ta, tb)) return false;
        if (!med.heterogeneous()) {                       // exact free-flight (one draw)
            double t = ta - std::log(1.0 - rng.uniformOpen()) / stBase;
            if (t < tb) { tHit = t; return true; }
            return false;
        }
        if (med.majorant && med.majorant->valid()) {
            const MajorantGrid& g = *med.majorant;
            bool hit = false;
            majorantWalk(g, o, dir, ta, tb, [&](size_t ci, double t0, double t1) {
                double sigMax = stBase * ((double)g.ctrl[ci] + (double)g.res[ci]);
                if (sigMax <= 0.0) return true;           // vacuum cell: skip, no RNG draw
                double t = t0;
                for (;;) {
                    t += -std::log(1.0 - rng.uniformOpen()) / sigMax;
                    if (t >= t1) return true;
                    double sigT = stBase * med.densityAt(o + dir * t, tabs);
                    if (rng.uniform() * sigMax < sigT) { tHit = t; hit = true; return false; }
                }
            });
            return hit;
        }
        double sigMax = stBase * med.densityMax;
        if (sigMax <= 0.0) return false;
        double t = ta;
        for (;;) {
            t += -std::log(1.0 - rng.uniformOpen()) / sigMax;
            if (t >= tb) return false;
            double sigT = stBase * med.densityAt(o + dir * t, tabs);
            if (rng.uniform() * sigMax < sigT) { tHit = t; return true; }  // real collision
        }                                                                 // else null collision
    }

    // Unbiased transmittance along [o, o+dir*dist] through the medium. Exact exp for a
    // homogeneous medium (clipped to its bound); RESIDUAL ratio tracking against the
    // per-cell majorant grid otherwise, falling back to plain ratio tracking against the
    // global majorant when the medium has no grid.
    //
    // Residual ratio tracking (Novak et al. 2014) splits each cell's extinction into a
    // constant CONTROL term integrated analytically and a residual tracked stochastically:
    //     Tr_cell = exp(-sigma_c * L) * PROD (1 - (sigma(x_i) - sigma_c)/sigma_r)
    // with candidates at rate sigma_r >= sup|sigma - sigma_c| over the cell. The residual
    // factors sit near 1 instead of near 0, which is the whole difference between an
    // optically thick volume converging and not — see majorant.h for the measured
    // variance reduction (~1100x at tau = 8). Note the factors may exceed 1 (the residual
    // is signed); that is correct and is what keeps the estimator unbiased.
    // Upper bound on the wavelength count mediumTransmittanceSpec accepts (a stack array in a
    // hot path, so it is fixed rather than allocated). SpecThr::K + 1 -- the grid plus the
    // camera's own wavelength, which rides along so the scalar and the vector cannot disagree.
    static constexpr int kSpecTrMax = 32;
    static double mediumTransmittance(const Medium& med, const Vec3& o, const Vec3& dir,
                                      double dist, double lambda, Pcg32& rng,
                                      const PatTables* tabs) {
        double stBase = med.sigmaT(lambda);
        if (stBase <= 0.0) return 1.0;
        double ta, tb;
        if (!med.clipToBounds(o, dir, 0.0, dist, ta, tb)) return 1.0;   // ray never enters fog
        if (!med.heterogeneous())
            return std::exp(-stBase * (tb - ta));
        if (med.majorant && med.majorant->valid()) {
            const MajorantGrid& g = *med.majorant;
            double Tr = 1.0;
            majorantWalk(g, o, dir, ta, tb, [&](size_t ci, double t0, double t1) {
                const double sigC = stBase * (double)g.ctrl[ci];
                const double sigR = stBase * (double)g.res[ci];
                if (sigC > 0.0) Tr *= std::exp(-sigC * (t1 - t0));       // control, analytic
                if (sigR <= 0.0) return Tr > 0.0;                        // uniform cell: exact
                double t = t0;
                for (;;) {
                    t += -std::log(1.0 - rng.uniformOpen()) / sigR;
                    if (t >= t1) return true;
                    double sigT = stBase * med.densityAt(o + dir * t, tabs);
                    Tr *= 1.0 - (sigT - sigC) / sigR;
                    if (Tr == 0.0) return false;
                }
            });
            return Tr > 0.0 ? Tr : 0.0;
        }
        double sigMax = stBase * med.densityMax;
        if (sigMax <= 0.0) return 1.0;
        double Tr = 1.0, t = ta;
        for (;;) {
            t += -std::log(1.0 - rng.uniformOpen()) / sigMax;
            if (t >= tb) break;
            double sigT = stBase * med.densityAt(o + dir * t, tabs);
            Tr *= 1.0 - sigT / sigMax;
        }
        return Tr;
    }

    // Transmittance at SEVERAL wavelengths along one segment through one medium, from a SINGLE
    // shared walk. Mode M's camera walk is monochromatic while its photon map is not, so the
    // extinction it applies has to be known at the photons' wavelengths too, not just the camera's
    // -- with a coloured medium the difference is not subtle (measured: a 113 % channel spread, see
    // known-issues). SPECGATHER's ratio trick cannot be reused here because a transmittance is a
    // stochastic ESTIMATE and E[A/B] != E[A]/E[B]; the vector has to be carried.
    //
    // The wavelength enters only through one scalar per medium, `sigmaT(lambda)` -- the density
    // field is achromatic -- which is what makes both tiers cheap:
    //   * HOMOGENEOUS: exp(-sigma_t(lambda) * len) per wavelength. Analytic, exact, no sampling.
    //   * HETEROGENEOUS: ratio tracking driven by a majorant that bounds EVERY wavelength (not just
    //     the hero's), with one weight update per wavelength at each collision. One march, wider
    //     arithmetic, and the estimates are CORRELATED -- which matters, because the consumer takes
    //     their ratio and independent estimates would make that ratio noisy as well as biased.
    // `Tr[i]` is MULTIPLIED into, so a caller can chain media.
    // FTRACE_SPECMEDIA=0 forces the scalar tier even where sigma_t VARIES with wavelength -- i.e.
    // the pre-0.322.0 behaviour, one wavelength's transmittance applied to all of them. HOST ONLY,
    // and it exists for exactly one reason: without it there is no way to ask whether a rig can SEE
    // the error the three-tier fix removes, and a clean "after" number from an instrument that
    // cannot see the "before" proves nothing. tools/specmedia_rig.py uses it as its control.
    static bool specMediaOn() {
        static const bool on = [] {
            const char* e = std::getenv("FTRACE_SPECMEDIA");
            return !(e && e[0] == '0');
        }();
        return on;
    }

    static void mediumTransmittanceSpec(const Medium& med, const Vec3& o, const Vec3& dir,
                                        double dist, const double* lams, int K, double* Tr,
                                        Pcg32& rng, const PatTables* tabs) {
        double stB[kSpecTrMax];
        double stMax = 0.0;
        for (int i = 0; i < K; ++i) {
            stB[i] = med.sigmaT(lams[i]);
            if (stB[i] < 0.0) stB[i] = 0.0;
            if (stB[i] > stMax) stMax = stB[i];
        }
        if (stMax <= 0.0) return;                                   // transparent at every lambda
        // A medium that is FLAT in sigma_t has one transmittance, not K of them -- so take the
        // ordinary scalar walk once and repeat it, rather than paying K correlated weight
        // updates at every collision. This is exact, not an approximation: a flat medium's
        // transmittance genuinely is wavelength-independent. It matters because MOST media are
        // flat (both of gallery_rain's are), and the stochastic tier is expensive in a thick
        // noisy density field -- this keeps them paying nothing for a feature they cannot use.
        bool flatMed = true;
        for (int i = 1; i < K; ++i)
            if (std::fabs(stB[i] - stB[0]) > 1e-12 * (1.0 + stB[0])) { flatMed = false; break; }
        const bool forceScalar = !specMediaOn();
        if (flatMed || forceScalar) {
            // A genuinely flat medium: any wavelength gives the same answer, so lams[0] keeps this
            // bit-identical. FORCED: use the LAST entry, which the gather's caller sets to the
            // camera's own wavelength (camMediaTrSpec appends it after the grid) -- that is the
            // historical bug being reproduced, not an arbitrary bin.
            const double lamUse = flatMed ? lams[0] : lams[K - 1];
            const double T = mediumTransmittance(med, o, dir, dist, lamUse, rng, tabs);
            for (int i = 0; i < K; ++i) Tr[i] *= T;
            return;
        }
        double ta, tb;
        if (!med.clipToBounds(o, dir, 0.0, dist, ta, tb)) return;   // never enters
        if (!med.heterogeneous()) {                                  // ---- analytic tier
            const double len = tb - ta;
            for (int i = 0; i < K; ++i) Tr[i] *= std::exp(-stB[i] * len);
            return;
        }
        // ---- stochastic tier: one collision sequence, K correlated weights ------------------
        // The majorant must bound every wavelength, or a wavelength whose sigma_t exceeds it would
        // get a negative weight update and a biased (indeed, possibly negative) estimate.
        if (med.majorant && med.majorant->valid()) {
            const MajorantGrid& g = *med.majorant;
            majorantWalk(g, o, dir, ta, tb, [&](size_t ci, double t0, double t1) {
                const double cC = (double)g.ctrl[ci], cR = (double)g.res[ci];
                for (int i = 0; i < K; ++i)                          // control part: analytic
                    if (stB[i] * cC > 0.0) Tr[i] *= std::exp(-stB[i] * cC * (t1 - t0));
                const double sigR = stMax * cR;                      // residual: bounds all lambdas
                if (sigR <= 0.0) return true;
                double t = t0;
                for (;;) {
                    t += -std::log(1.0 - rng.uniformOpen()) / sigR;
                    if (t >= t1) return true;
                    const double dens = med.densityAt(o + dir * t, tabs);
                    for (int i = 0; i < K; ++i) {
                        const double sigT = stB[i] * dens, sigC = stB[i] * cC;
                        Tr[i] *= 1.0 - (sigT - sigC) / sigR;
                        if (Tr[i] < 0.0) Tr[i] = 0.0;
                    }
                }
            });
            return;
        }
        const double sigMax = stMax * med.densityMax;                // global majorant over all lambdas
        if (sigMax <= 0.0) return;
        double t = ta;
        for (;;) {
            t += -std::log(1.0 - rng.uniformOpen()) / sigMax;
            if (t >= tb) break;
            const double dens = med.densityAt(o + dir * t, tabs);
            for (int i = 0; i < K; ++i) {
                Tr[i] *= 1.0 - (stB[i] * dens) / sigMax;
                if (Tr[i] < 0.0) Tr[i] = 0.0;
            }
        }
    }


    // --- Multi-medium (superposition) forward helpers ------------------------
    // The scene may hold several independent media (Scene::media) that overlap. Two
    // facts make combining them exact and per-medium bit-identity-preserving:
    //   * Extinction adds:  T_total = exp(-INT (sig1+sig2+..)) = PROD exp(-INT sig_i),
    //     so the total transmittance is the PRODUCT of the per-medium transmittances,
    //     each estimated independently (product of independent unbiased estimators is
    //     unbiased for the product).
    //   * Collisions superpose:  the union of independent Poisson collision processes
    //     with rates sig_i(x) is a Poisson process with rate SUM sig_i(x), whose first
    //     event is the EARLIEST of the components' first events, and the component that
    //     produced it (the scattering medium) is picked with the correct probability.
    // So we sample each medium's first collision independently and take the minimum.
    // With a single medium these reduce to the exact single-medium paths above (same
    // RNG draws), so existing scenes are unchanged.

    // WHICH media a transport call should consider. Everything defaults to MedAll, so an
    // ordinary scene and every pre-existing call site behave exactly as before.
    //
    // The split exists for `-beams`. Under `-beams` a photon does not scatter analog in a
    // medium: it crosses STRAIGHT, the extinction is booked as absorbed, and the stored
    // beam pays that light back at gather time. That bargain needs a straight chord to
    // store, which a GRIN medium — one that bends the photon through itself — cannot
    // provide. So the two halves are transported by different rules and must therefore be
    // sampled separately, or a GRIN medium would be charged extinction by the straight
    // crossing AND scattered analog (double counting), or charged and never paid back
    // (which is what it did before 0.198.0: a scattering GRIN medium acted purely
    // absorbing under `-beams`).
    //
    //   MedStraight — non-GRIN media: crossed straight, deposited/gathered as beams.
    //   MedCurved   — GRIN media: keep full analog transport, exactly as without `-beams`.
    enum MedFilter { MedAll = 0, MedStraight = 1, MedCurved = 2 };
    static inline bool medPasses(const Medium& m, MedFilter f) {
        return f == MedAll || ((f == MedCurved) == m.grin());
    }

    // Earliest real collision across all media within [0,dMax]. On a hit, `tHit` is the
    // distance and `whichMed` the index of the scattering medium. false if none.
    //
    // Takes the whole Scene rather than just `scene.media` because a density program may
    // sample the scene's `grid:`/`scatter:` tables (`density "grid:rho(x, y, z)"`), which
    // live beside the media in the Scene. Deriving the tables here means no caller can
    // forget to pass them — and every caller already had the Scene in hand.
    static bool sampleMediaCollision(const Scene& scene, const Vec3& o,
                                     const Vec3& dir, double dMax, double lambda, Pcg32& rng,
                                     double& tHit, int& whichMed, MedFilter filt = MedAll) {
        const std::vector<Medium>& media = scene.media;
        const PatTables tabs = scene.patTables();
        double best = dMax; int which = -1;
        for (int i = 0; i < (int)media.size(); ++i) {
            if (!medPasses(media[i], filt)) continue;
            double t;
            if (sampleMediumCollision(media[i], o, dir, dMax, lambda, rng, t, &tabs) && t < best) {
                best = t; which = i;
            }
        }
        if (which < 0) return false;
        tHit = best; whichMed = which; return true;
    }

    // Combined transmittance through all media = product of per-medium transmittances.
    static double mediaTransmittance(const Scene& scene, const Vec3& o,
                                     const Vec3& dir, double dist, double lambda, Pcg32& rng,
                                     MedFilter filt = MedAll) {
        const PatTables tabs = scene.patTables();   // see sampleMediaCollision
        double Tr = 1.0;
        for (const Medium& m : scene.media) {
            if (!medPasses(m, filt)) continue;
            Tr *= mediumTransmittance(m, o, dir, dist, lambda, rng, &tabs);
            if (Tr <= 0.0) break;
        }
        return Tr;
    }

    // Whole-scene version: media superpose, so the transmittances multiply (see the note below).
    static void mediaTransmittanceSpec(const Scene& scene, const Vec3& o, const Vec3& dir,
                                       double dist, const double* lams, int K, double* Tr,
                                       Pcg32& rng, MedFilter filt = MedAll) {
        const PatTables tabs = scene.patTables();
        for (const Medium& m : scene.media) {
            if (!medPasses(m, filt)) continue;
            mediumTransmittanceSpec(m, o, dir, dist, lams, K, Tr, rng, &tabs);
        }
    }

    // Append the photon beams for one straight crossing of the media, from `o` along `dir`
    // for `dLen` world units carrying power `beta` at the start (mode M, -beams).
    //
    // ONE BEAM PER MEDIUM, each clipped to that medium's own bound. Media superpose (see
    // sampleMediaCollision: independent Poisson processes, minimum of their sampled
    // collisions), so the in-scatter at a point is the SUM over the media containing it —
    // which means a per-medium record is not just an optimisation but the correct
    // decomposition: each beam then carries its own sigma_s and its own phase function, and
    // a rain volume overlapping a haze contributes a bow and a glow independently. Clipping
    // also keeps each beam's AABB tight, which is what makes the BVH worth having.
    //
    // `aGlass` is the absorption of the dielectric the photon is currently inside, carried
    // on the beam so the gather can Beer-Lambert to its own closest-approach point.
    //
    // `offFilt` is which media pre-attenuate the power over the lead-in `ta` (from the ray
    // origin to this medium's bound). Under SINGLE scatter that is MedStraight — a GRIN
    // medium's extinction is carried stochastically, by the caller clipping the crossing at
    // the analog collision, so applying it again here would double-count. Under MULTIPLE
    // scatter nothing is carried stochastically (the beam runs the whole chord to the surface
    // and the gather integrates Tr analytically), so every medium must be charged: MedAll.
    //
    // `lamS`/`nSec` are the photon's live SPECTRAL BUNDLE — extra stratified wavelengths that
    // this same chord also carries (photonbeams.h), and `specW` their relative throughputs
    // T(lamS[i])/T(lambda). They ride along untouched: the chord's GEOMETRY and its
    // transmittance are wavelength-independent wherever the bundle is still alive, which is
    // exactly the condition tracePhoton maintains; only the members' accumulated spectral
    // weights may differ, and that is what `specW` carries.
    // `order` = medium scattering order of this chord, 1 = single scatter. REQUIRED, not
    // defaulted -- see BeamBank::push. Pass `kBeamOrderUnknown` if the caller genuinely does
    // not track it, so the gap is explicit at the call site instead of invisible.
    void emitBeams(const Scene& scene, const Vec3& o, const Vec3& dir, double dLen,
                   double lambda, double beta, double aGlass, Pcg32& rng, int order,
                   MedFilter offFilt = MedStraight,
                   const double* lamS = nullptr, int nSec = 0,
                   const Vec3* achroCie = nullptr, int foldEmIdx = -1,
                   const double* specW = nullptr, int srcEm = -1, int surf = -1,
                   bool jDomain = false) const {
        if (!beamDeposit || !(beta > 0.0)) return;
        // Bound an escape-to-infinity crossing so an unbounded medium cannot produce a
        // 1e30-long box (see kBeamFarScale).
        // (named `farLimit`, not `far`: `far` is a legacy Windows SDK keyword macro)
        //
        // Applied PER MEDIUM, and only to the unbounded ones. A bounded medium already hands
        // `clipToBounds` a finite exit, so the clamp can only ever cut a beam SHORT of the
        // region it is supposed to fill — which is what it did on `scenes/_slab_ss.ftsl`: the
        // clamp is a multiple of `sceneRadius`, and until this was fixed alongside it
        // sceneRadius did not count media at all, so a big fog box lit by a small emitter
        // truncated every beam to a stub. Clamping the whole call up front also meant one
        // unbounded haze could shorten the beams of an unrelated bounded cloud it happened to
        // overlap. (Device twin: dEmitBeams in render_cuda.cu.)
        // `jDomain` (mode J): an unbounded medium's beam is clipped to the scene's beam domain
        // instead (Scene::jBeamMerges), the region the MIS weights count its merges in -- under a
        // blur one that holds the movers' sweep, so a mover far outside the instant's sphere is
        // still merged on. A length limit here would end beams where the weights still count them.
        const double farLimit = kBeamFarScale * std::max(scene.sceneRadius, 1e-3);
        if (!(dLen > 0.0)) return;
        const double keep = beamDeposit->keepProb;
        for (int i = 0; i < (int)scene.media.size(); ++i) {
            const Medium& md = scene.media[i];
            if (md.sigma_s(lambda) <= 0.0) continue;      // absorbing-only: nothing to gather
            // GRIN media are refused HERE, per medium, rather than scene-wide: a photon bends
            // only INSIDE a gradient-index region (grin::march jumps straight between them), so
            // its chord through an ordinary fog is a perfectly good straight beam even when some
            // other object in the scene is a GRIN lens. Only a medium that itself bends the
            // photon has no segment to store.
            if (md.grin()) continue;
            double ta, tb;
            const double dBeam = (md.bounded || jDomain) ? dLen : std::min(dLen, farLimit);
            if (!md.clipToBounds(o, dir, 0.0, dBeam, ta, tb)) continue;
            if (jDomain && !md.bounded && !scene.jBeamDomainClip(o, dir, ta, tb)) continue;
            if (!(tb > ta)) continue;
            // Russian roulette on the beam count (photonbeams.h): a beam lights a whole
            // chord, so far fewer beams than photons are needed — and a beam is ~72 B, so
            // one per crossing would run to gigabytes on a dense pass.
            if (keep < 1.0) { if (rng.uniform() >= keep) continue; }
            // Power at the beam's stored origin: the photon's throughput carried forward
            // through the glass absorption and the media extinction it crossed to get there.
            double p = beta / ((keep < 1.0) ? keep : 1.0);
            if (aGlass > 0.0 && ta > 0.0) p *= std::exp(-aGlass * ta);
            // MedStraight: charge only the media that were themselves crossed straight. A
            // GRIN medium overlapping this fog is transported ANALOG, so its extinction is
            // already carried by the deposition probability — the caller clips this crossing
            // at the analog collision, so the chance a beam covers depth s is exactly the
            // GRIN transmittance to s. Applying it here as well would count it twice.
            if (ta > 0.0) p *= mediaTransmittance(scene, o, dir, ta, lambda, rng, offFilt);
            if (!(p > 0.0)) continue;
            // ACHROMATIC-PATH FOLD, decided per DEPOSITED BEAM, because its two halves live in
            // different places: the PATH being wavelength-independent is a property of the
            // photon (the caller's `achroPath`), while the gather-time tail being flat is a
            // property of THIS medium. A photon crossing gallery_rain's achromatic cloud and
            // its `phase rainbow` rain curtain in one step deposits one beam of each kind, and
            // only the cloud's may fold. (Device twin: dEmitBeams in render_cuda.cu.)
            const double ca[3] = {achroCie ? achroCie->x : 0.0,
                                  achroCie ? achroCie->y : 0.0,
                                  achroCie ? achroCie->z : 0.0};
            const bool useAchro = achroCie && mediumAchromatic(md);
            // GATHER-TIME FOLD (scene.h, Scene::BowLut). The medium's coefficients are flat but
            // its phase is a rainbow table, so `useAchro` above correctly refused the fold — the
            // colour is not decidable without the scattering angle. It IS decidable at gather
            // time, from a per-(emitter, medium) table, provided the path carried no spectral
            // weight of its own (`foldEmIdx >= 0` is the caller's assertion that T == 1). This
            // is what stops a rain curtain being drawn as saturated single-wavelength streaks.
            const int bowEm = (!useAchro && achroCie && foldEmIdx >= 0 &&
                               scene.bowLut(foldEmIdx, i)) ? foldEmIdx : -1;
            // Fold diagnostics. Row FK_None counts the beams whose PATH was still
            // wavelength-independent at the deposit; the rest attribute the ones that were not
            // to the event that retired them. Deliberately keyed off `achroCie`, NOT
            // `useAchro`: in a CHROMATIC medium the path can be perfectly foldable and still
            // be refused, because it is the medium's gather-time tail that is not flat. That
            // gap is exactly the population a gather-time fold would serve, so it has to be
            // visible rather than hidden behind the same zero as a genuinely divergent path.
            if (foldDiagOn() && i < 16)
                g_foldKillHist[i][achroCie ? FK_None : g_foldKill].fetch_add(
                    1, std::memory_order_relaxed);
            beamDeposit->push(o + dir * ta, dir, tb - ta, p, lambda, aGlass, i, order,
                              lamS, nSec, (useAchro || bowEm >= 0) ? ca : nullptr, bowEm,
                              specW, srcEm, surf);
        }
    }

    // Wavelength-INDEPENDENT part of the mode-B pinhole connection: project the surface
    // vertex to the film, test its camera side (camSideOk), and occlusion-test the
    // shadow ray. On success returns true and fills `g` with the pixel and the geometry
    // factors, from which the caller forms the per-λ contribution as
    //   beta * (rho/PI) * cosSurf * corr / denom  (× media transmittance).
    // Sharing this across the C hero wavelengths (one BVH occlusion test, one projection)
    // is the whole point of hero splatting; the scalar connect() below reuses it too so
    // its float ordering — and thus mode B — stays bit-identical.
    struct ConnGeom {
        int px = 0, py = 0;
        double cosSurf = 0, corr = 0, denom = 0, dist = 0;
        Vec3 wdir;
    };
    // A FIBER vertex connects differently, in three ways that all live down here rather
    // than in the caller (see hair_shade.h for why the projection factor changes):
    //   (1) the projection is the strand's longitudinal cosine, folded into `hairFCos`
    //       together with the BCSDF, so `cosSurf` is forced to 1 and plays no part;
    //   (2) TT transmits, so the camera sitting BEHIND the shading normal is a legitimate
    //       and often bright connection — the surface path's `cosSurf <= 0` rejection
    //       would delete the forward glow that a pale coat is mostly made of;
    //   (3) the shadow ray has to start past the strand's own body, or the tube occludes
    //       its own transmitted lobe.
    // `shadowTerminatorG` and the Veach shading-normal adjoint are both corrections for
    // an interpolated surface normal used as a projection axis; the fiber does not use its
    // normal that way, so both are exactly 1 here rather than approximately so.
    // The camera side of a forward camera connection from a surface vertex (shading normal `n`,
    // geometric `ng` on its shaded side) along `wdir`: false where it may not connect, else true and
    // `cosSurf` its projected cosine. As mode R's integrand has it (bdpt::lambertLit): a Lambertian
    // vertex lights a camera anywhere on its geometric front, on either side of n; a glossy one
    // (`glossy`) only above n, as its lobe asks (bsdfF). Never softened: the terminator's softening
    // belongs to the light side, at the emitter (bdpt.h emitterSoftening). Where n == ng this is the
    // plain n.wdir > 0.
    static bool camSideOk(const Vec3& n, const Vec3& ng, const Vec3& wdir, bool glossy,
                          double& cosSurf) {
        const double c = dot(n, wdir);
        if (c == 0.0 || !(dot(ng, wdir) > 0.0) || (glossy && c < 0.0)) return false;
        cosSurf = std::fabs(c);
        return true;
    }
    // A photon's continuation from a Lambertian vertex it reached along -`wi` (lit: wi above the
    // shading normal), into `wo`, returning the throughput factor beyond the albedo. Where the
    // shading normal tilts it is drawn about the GEOMETRIC one, so every direction mode R lights is
    // reachable -- one below the shading normal too (bdpt::lambertLit) -- weighted by Veach's
    // adjoint for that density, |cos(wi,ns)| / |cos(wi,ng)|. Where ns == ng it is the plain cosine
    // draw about ns, with the adjoint factor that is then exactly 1 (bit-identical).
    static double diffuseContinuation(const Hit& h, const Vec3& ngo, const Vec3& wi, Pcg32& rng,
                                      Vec3& wo) {
        if (dot(h.n, ngo) >= 1.0 - 1e-7) {
            wo = cosineHemisphere(h.n, rng);
            return shadingAdjointCorr(wi, wo, h.n, ngo);
        }
        wo = cosineHemisphere(ngo, rng);
        return arrivalAdjoint(h, ngo, wi);
    }
    // The arrival half of Veach's adjoint at a Lambertian vertex reached along -`wi`:
    // |cos(wi,ns)| / |cos(wi,ng)|, the shading normal's cosine where the photon's flux per unit of
    // GEOMETRIC area carries the geometric one -- what turns a deposit's density estimate into the
    // irradiance mode R shades with, and the continuation's weight above. Exactly 1 where ns == ng.
    static double arrivalAdjoint(const Hit& h, const Vec3& ngo, const Vec3& wi) {
        if (dot(h.n, ngo) >= 1.0 - 1e-7) return 1.0;
        return arrivalAdjointCorr(wi, h.n, ngo);
    }
    // A photon reaching a Lambertian lobe (Diffuse, a Fluorescent's elastic base, a compound's
    // deposit for its diffuse lobes) along -`wi`: whether it lights it -- only from above the
    // shading normal (bdpt::lambertLit) -- and in `soften` what its arrival is worth there, the
    // terminator's softening toward the emitter on its first bounce (`firstBounce`; bdpt.h
    // emitterSoftening), 1 after. The one rule every photon tracer applies; device twin
    // dLambertArrival.
    static bool lambertArrival(const Hit& h, const Vec3& ngo, const Vec3& wi, bool firstBounce,
                               double& soften) {
        if (!(dot(wi, h.n) > 0.0)) return false;
        soften = firstBounce ? shadowTerminatorG(wi, h.n, ngo) : 1.0;
        return true;
    }
    // A glossy lobe's first bounce (`firstBounce`, arriving along -`wi`) softens toward the
    // emitter, as mode R's light sampling softens it and every bidirectional mode's first bounce
    // (bdpt.h emitterSoftening); 1 after, and for a lobe with no finite value (isPlanarMirrorMat:
    // a delta, which R never connects). Device twin dGlossyArrival.
    static double glossyArrival(const Material& lobe, const Hit& h, const Vec3& wi, bool firstBounce) {
        if (!firstBounce || isPlanarMirrorMat(lobe)) return 1.0;
        return shadowTerminatorG(wi, h.n, orientedGeoN(h));
    }
    // A photon's glossy continuation from its arrival (`wIn`, toward the light) into `o`: the lobe's
    // adjoint (glossyAdjointRatio) times the shading normal's half of Veach's (shadingAdjointCorr), as
    // bdpt.h's light walk takes them. 0 for an arrival from below ns. Device twin dGlossyPhotonAdjoint.
    static double glossyPhotonAdjoint(const Hit& h, const Vec3& wIn, const Vec3& o) {
        return glossyAdjointRatio(h.n, wIn, o) * shadingAdjointCorr(wIn, o, h.n, orientedGeoN(h));
    }
    // A compound's own deposit (compoundGathers), made before its pick, of a photon arriving along
    // -`wi` with throughput `beta`: what it stores. A compound whose gathering lobes are all
    // Lambertian takes their arrival rule (lambertArrival, then the arrival adjoint) -- the lobe the
    // pick then takes applies its own rule to the photon going on; one with a translucent lobe stores
    // the arrival as a translucent's own case does (BUMP-MODES stage 3, known-issues). 0: nothing.
    static double compoundDepositWeight(const Scene& scene, const Material& cm, const Hit& h,
                                        const Vec3& wi, bool firstBounce, double beta) {
        if (compoundTransmits(scene, cm)) return beta;
        const Vec3 ngo = orientedGeoN(h);
        double s;
        if (!lambertArrival(h, ngo, wi, firstBounce, s)) return 0.0;
        return beta * s * arrivalAdjoint(h, ngo, wi);
    }
    bool connectGeom(const Scene& scene, const Camera& cam, const Vec3& p, const Vec3& n,
                     const Vec3& ng, const Vec3& wi, ConnGeom& g,
                     const HairShade* hs = nullptr, bool glossy = false) const {
        Vec3 toCam = cam.eye - p;
        g.dist = length(toCam);
        g.wdir = toCam / g.dist;
        if (hs) {
            g.cosSurf = 1.0; g.corr = 1.0;
            double cosCamH, dist2H;
            if (!cam.project(p, g.px, g.py, cosCamH, dist2H)) return false;
            const double off = hairExitOffset(*hs, n, g.wdir);
            if (off >= g.dist) return false;
            if (scene.occluded(p + g.wdir * 1e-6, g.wdir, g.dist - off - 1e-6, 1e-6,
                               /*camLeg=*/true, off)) return false;
            g.denom = dist2H * cam.pixelSolidAngle(cosCamH);
            return true;
        }
        // The camera side (camSideOk): never behind the true geometry, never softened.
        if (!camSideOk(n, ng, g.wdir, glossy, g.cosSurf)) return false;
        double cosCam, dist2;
        if (!cam.project(p, g.px, g.py, cosCam, dist2)) return false;
        if (scene.occluded(p + ng * 1e-6, g.wdir, g.dist - 2e-6, 1e-6, /*camLeg=*/true)) return false;
        double omega = cam.pixelSolidAngle(cosCam);
        // Veach shading-normal adjoint correction for this particle connection
        // (wi = toward the previous/light-side vertex, wo = wdir toward the camera).
        // cosSurf * corr = cos(wo,Ng)*cos(wi,Ns)/cos(wi,Ng), so the grazing cosSurf
        // cancels analytically and this stays bounded. Exactly 1 when Ns == Ng.
        g.corr = shadingAdjointCorr(wi, g.wdir, n, ng);
        g.denom = dist2 * omega;
        return true;
    }

    // Model B: connect a surface vertex to the pinhole and splat onto the film.
    // f = rho/pi (Lambertian). The measurement contribution of a surface patch into
    // one pixel is  beta * f * cosSurf / (dist^2 * Omega_pix), where Omega_pix is the
    // solid angle that pixel subtends. This form is projection-general (fisheye and
    // rectilinear alike): for a rectilinear lens Omega_pix = A_pix*cosCam^3, which
    // reproduces the classic G * We = cosSurf*cosCam/dist^2 * 1/(A_pix cosCam^4).
    // `gs` (optional): a glossy vertex, whose f replaces rho/pi toward the camera (GlossySplat).
    void connect(const Scene& scene, const Camera& cam, Film& film,
                 const Vec3& p, const Vec3& n, const Vec3& ng, const Vec3& wi,
                 double lambda, double beta, double rho, Pcg32& rng,
                 const HairShade* hs = nullptr, const GlossySplat* gs = nullptr) const {
        ConnGeom g;
        if (!connectGeom(scene, cam, p, n, ng, wi, g, hs, gs != nullptr)) return;
        // For a fiber, `hairFCos` IS f*projection (hair_shade.h); cosSurf is 1 so the
        // arithmetic below stays one shared expression instead of two near-copies.
        double f = hs ? hairFCos(*hs, g.wdir) : gs ? gs->f(g.wdir, lambda) : rho / PI;
        double contrib = beta * f * g.cosSurf * g.corr / g.denom;
        // Attenuation of the shadow ray through the fog (Beer-Lambert; ratio tracking
        // for a heterogeneous medium, exact exp for a homogeneous one; product over media).
        if (!scene.media.empty())
            contrib *= mediaTransmittance(scene, p, g.wdir, g.dist, lambda, rng);
        film.add(g.px, g.py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Hero-wavelength mode-B connection: one shared connectGeom, then splat each of the
    // `nUp` live wavelengths with its own throughput beta[i] and reflectance rho[i]. The
    // per-λ contribution uses the SAME float ordering as connect() above.
    void connectHero(const Scene& scene, const Camera& cam, Film& film,
                     const Vec3& p, const Vec3& n, const Vec3& ng, const Vec3& wi,
                     const double* lam, const double* beta, const double* rho, int nUp,
                     Pcg32& rng, const GlossySplat* gs = nullptr) const {   // see connect
        ConnGeom g;
        if (!connectGeom(scene, cam, p, n, ng, wi, g, nullptr, gs != nullptr)) return;
        for (int i = 0; i < nUp; ++i) {
            double f = gs ? gs->f(g.wdir, lam[i]) : rho[i] / PI;
            double contrib = beta[i] * f * g.cosSurf * g.corr / g.denom;
            if (!scene.media.empty())
                contrib *= mediaTransmittance(scene, p, g.wdir, g.dist, lam[i], rng);
            film.add(g.px, g.py, Vec3(cieX(lam[i]), cieY(lam[i]), cieZ(lam[i])) * contrib);
        }
    }

    // Model B for a VOLUME scattering vertex: connect the collision point to the
    // pinhole. The surface BRDF/cosine is replaced by the phase function and the
    // single-scattering albedo; there is no surface normal. wIn is the photon's
    // propagation direction into the collision.
    //   contrib = beta * albedo * p_HG(cos) / (dist^2 * Omega_pix) * T_fog
    void connectVolume(const Scene& scene, const Medium& med, const Camera& cam, Film& film,
                       const Vec3& p, const Vec3& wIn, double lambda, double beta,
                       Pcg32& rng) const {
        Vec3 toCam = cam.eye - p;
        double dist = length(toCam);
        Vec3 wdir = toCam / dist;
        int px, py; double cosCam, dist2;
        if (!cam.project(p, px, py, cosCam, dist2)) return;
        if (scene.occluded(p + wdir * 1e-6, wdir, dist - 2e-6, 1e-6, /*camLeg=*/true)) return;

        double ph = med.phaseValue(dot(wIn, wdir), lambda); // scattering medium's phase (HG or rainbow)
        double Lambda = med.albedo(lambda);
        double omega = cam.pixelSolidAngle(cosCam);         // projection-general pixel solid angle
        double contrib = beta * Lambda * ph / (dist2 * omega);
        contrib *= mediaTransmittance(scene, p, wdir, dist, lambda, rng);   // fog transmittance (all media)
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // ---- next-event splats through a PHYSICAL multi-element lens (modes A and B) -------
    // The thin-lens connections below sample a point on the aperture disc and image it with
    // the paraxial lens law. A physical lens samples a point on its front element instead
    // and traces the ray through the real glass onto the sensor (Camera::lensConnect, which
    // also derives the weight), so distortion, vignetting, aberrations and dispersion
    // emerge exactly as in the backward camera (genLensRay), and the image lands on the
    // same absolute scale: the flux through the lens divided by the film cell's area, the
    // irradiance the backward camera's cos^4*A/Z^2 weight measures.
    //
    // Surface (or fiber) vertex: beta * f * cosSurf * w, f the BRDF (rho/pi for this
    // Lambertian splat; the fiber's hairFCos carries its own projection).
    void connectLensPhysical(const Scene& scene, const Camera& cam, Film& film,
                             const Vec3& p, const Vec3& n, const Vec3& ng, const Vec3& wi,
                             double lambda, double beta, double rho, Pcg32& rng,
                             const HairShade* hs, const Vec3* plane,
                             const GlossySplat* gs = nullptr) const {        // see connect
        const double u1 = rng.uniform(), u2 = rng.uniform();
        Camera::LensEntry le;
        if (!cam.lensEntry(p, u1, u2, le)) return;
        // The tests that need only the entry come before the trace through the glass.
        if (plane && dot(*plane, le.Y - p) <= 0.0) return;
        const Vec3& wdir = le.wdir;
        double cosSurf = dot(n, wdir);
        if (!hs && !camSideOk(n, ng, wdir, gs != nullptr, cosSurf)) return;   // see connectLens
        const double off = hs ? hairExitOffset(*hs, n, wdir) : 1e-6;
        if (off >= le.dist) return;
        int px, py; double wgt;
        if (!cam.lensEntryImage(le, lambda, px, py, wgt)) return;
        if (scene.occluded(p + (hs ? wdir : ng) * (hs ? 1e-6 : off), wdir, le.dist - off - 1e-6, 1e-6,
                           /*camLeg=*/true, hs ? off : 0.0)) return;
        const double corr = hs ? 1.0 : shadingAdjointCorr(wi, wdir, n, ng);
        const double fcos = hs ? hairFCos(*hs, wdir)
                               : (gs ? gs->f(wdir, lambda) : rho / PI) * cosSurf;
        double contrib = beta * fcos * corr * wgt;
        if (!scene.media.empty())
            contrib *= mediaTransmittance(scene, p, wdir, le.dist, lambda, rng);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Hero bundle through a physical lens. The wavelengths share the front-element sample (one
    // draw, so the RNG stream does not depend on C), the entry and the scene leg from p to it;
    // the glass disperses, so each is traced through it on its own and lands in its own cell,
    // or is clipped on its own.
    void connectLensHeroPhysical(const Scene& scene, const Camera& cam, Film& film,
                                 const Vec3& p, const Vec3& n, const Vec3& ng, const Vec3& wi,
                                 const double* lam, const double* beta, const double* rho,
                                 int nUp, Pcg32& rng, const Vec3* plane,
                                 const GlossySplat* gs = nullptr) const {    // see connect
        const double u1 = rng.uniform(), u2 = rng.uniform();
        Camera::LensEntry le;
        if (!cam.lensEntry(p, u1, u2, le)) return;
        if (plane && dot(*plane, le.Y - p) <= 0.0) return;
        const Vec3& wdir = le.wdir;
        double cosSurf;
        if (!camSideOk(n, ng, wdir, gs != nullptr, cosSurf)) return;
        int px[hero::kHeroMax], py[hero::kHeroMax];
        double wgt[hero::kHeroMax];
        bool ok[hero::kHeroMax] = {}, any = false;
        for (int i = 0; i < nUp; ++i) {
            ok[i] = cam.lensEntryImage(le, lam[i], px[i], py[i], wgt[i]);
            any = any || ok[i];
        }
        if (!any) return;
        if (scene.occluded(p + ng * 1e-6, wdir, le.dist - 2e-6, 1e-6, /*camLeg=*/true)) return;
        const double corr = shadingAdjointCorr(wi, wdir, n, ng);
        for (int i = 0; i < nUp; ++i) {
            if (!ok[i]) continue;
            double contrib = beta[i] * (gs ? gs->f(wdir, lam[i]) : rho[i] / PI) * cosSurf * corr *
                             wgt[i];
            if (!scene.media.empty())
                contrib *= mediaTransmittance(scene, p, wdir, le.dist, lam[i], rng);
            film.add(px[i], py[i], Vec3(cieX(lam[i]), cieY(lam[i]), cieZ(lam[i])) * contrib);
        }
    }

    // A volume vertex through a physical lens: `phaseW` is what replaces f * cosSurf, the
    // in-scatter albedo * phase for a fog vertex or the isotropic 1/(4 pi) for a fire.
    void connectVolumeLensPhysical(const Scene& scene, const Camera& cam, Film& film,
                                   const Vec3& p, double lambda, double beta, double phaseW,
                                   const Vec3* wIn, const Medium* med, Pcg32& rng) const {
        const double u1 = rng.uniform(), u2 = rng.uniform();
        Vec3 Y, wdir; double dist, wgt; int px, py;
        if (!cam.lensConnect(p, lambda, u1, u2, Y, wdir, dist, px, py, wgt)) return;
        if (scene.occluded(p + wdir * 1e-6, wdir, dist - 2e-6, 1e-6, /*camLeg=*/true)) return;
        double f = phaseW;
        if (med && wIn) f = med->albedo(lambda) * med->phaseValue(dot(*wIn, wdir), lambda);
        double contrib = beta * f * wgt;
        contrib *= mediaTransmittance(scene, p, wdir, dist, lambda, rng);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Model A (physical camera): next-event splat through the finite lens pupil.
    // Sample a point A uniformly on the aperture disc, connect the surface vertex to
    // A, refract through the thin lens and splat onto the film cell A images to.
    // This is the importance-sampled form of model C's brute-force catch: the same
    // flux-per-cell estimator, so A and C share both scale and shape (validated), but
    // A converges because it never waits for a photon to randomly thread the pupil.
    //
    // Deriving the weight. In C a diffuse vertex scatters cosine-distributed and, if
    // the ray happens to pass through pupil area dA around A, deposits beta into
    // cell(A). The expected deposit is  beta * rho * INT (cosSurf/pi)(cosLens/dist^2) dA
    // over the pupil. Importance-sampling A ~ uniform(1/(pi R^2)) gives the single
    // sample estimator  beta * rho * cosSurf * cosLens * R^2 / dist^2  (the BRDF's 1/pi
    // cancels the pupil pdf's pi R^2). cosLens is the cosine at the pupil (natural
    // vignetting); the film-cell mapping supplies the rest of the angular falloff and
    // the depth-of-field spread automatically. Rectilinear film mapping only — a real
    // fisheye needs a wide-angle lens element, so author fisheye with model B instead.
    // `plane` (optional): an aimed quad's face (Emitter::nGeom); a pupil point behind it gets
    // nothing, whatever `n` (the aim) says. Tested at the sampled point, not the eye: a pupil
    // can straddle the plane.
    void connectLens(const Scene& scene, const Camera& cam, Film& film,
                     const Vec3& p, const Vec3& n, const Vec3& ng, const Vec3& wi,
                     double lambda, double beta, double rho, Pcg32& rng,
                     const HairShade* hs = nullptr, const Vec3* plane = nullptr,
                     const GlossySplat* gs = nullptr) const {                // see connect
        if (cam.hasLens()) {
            connectLensPhysical(scene, cam, film, p, n, ng, wi, lambda, beta, rho, rng, hs, plane,
                                gs);
            return;
        }
        double R = cam.apertureR;
        double rr = R * std::sqrt(rng.uniform());
        double a  = 2.0 * PI * rng.uniform();
        Vec3 A = cam.eye + cam.u * (rr * std::cos(a)) + cam.v * (rr * std::sin(a));
        if (plane && dot(*plane, A - p) <= 0.0) return;
        Vec3 toA = A - p;
        double dist = length(toA);
        if (dist < 1e-9) return;
        Vec3 wdir = toA / dist;
        double cosSurf = dot(n, wdir);
        // The pupil on the camera side (camSideOk; see connect()). A FIBER skips it -- its
        // transmitted lobe legitimately exits the far side, and its projection is longitudinal
        // rather than normal-relative (see hair_shade.h / connectGeom).
        if (!hs && !camSideOk(n, ng, wdir, gs != nullptr, cosSurf)) return;
        double cosLens = -dot(wdir, cam.w);              // cosine at the lens (w faces the scene)
        if (cosLens <= 1e-6) return;                     // not heading toward the film
        int px, py;
        if (!cam.lensImage(A, wdir, px, py)) return;
        const double off = hs ? hairExitOffset(*hs, n, wdir) : 1e-6;
        if (off >= dist) return;
        if (scene.occluded(p + (hs ? wdir : ng) * (hs ? 1e-6 : off), wdir, dist - off - 1e-6, 1e-6,
                           /*camLeg=*/true, hs ? off : 0.0)) return;

        // beta * (rho/pi BRDF) * cosSurf * cosLens / dist^2 * (pi R^2 = 1/pdf_A).
        // cosSurf carries the Veach shading-normal adjoint correction (see connect()).
        // For a fiber the BRDF-times-projection is hairFCos, and the pi it would be
        // divided by is the same pi that pi*R^2 supplies — hence the explicit factor.
        double corr = hs ? 1.0 : shadingAdjointCorr(wi, wdir, n, ng);
        double fcos = hs ? PI * hairFCos(*hs, wdir)
                         : (gs ? PI * gs->f(wdir, lambda) : rho) * cosSurf;
        double contrib = beta * fcos * corr * cosLens * (R * R) / (dist * dist);
        // ABSOLUTE-SCALE NORMALISER (A/C <-> B unification). The line above deposits
        // radiant FLUX through the pupil into the film CELL (it carries the pupil area
        // R^2 but no 1/cell-area), whereas mode B's connect() deposits RADIANCE (it
        // divides by the pixel solid angle). Dividing the flux by the physical cell
        // area A_cell = pixelPlaneArea()*filmDist^2 turns it into film-plane IRRADIANCE
        // E, so the finite lens now records exactly E = L * (pi/4)/N^2 (N = filmDist/2R)
        // -- the same absolute scale as B*camEq. Equivalent derivation: current splat =
        // B * (pi R^2 * A_pix); target = B * (pi R^2 / filmDist^2); ratio = 1/A_cell.
        // A_cell depends only on the camera (fov/res/filmDist), so this is a per-camera
        // constant: auto-exposed scenes stay byte-identical (the p99 anchor divides it
        // out) and A stays consistent with C; only ABSOLUTE-EV A/C are corrected to
        // land mid-tone at ABS_EXPOSURE_GAIN, matching B.
        contrib *= 1.0 / (cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
        if (!scene.media.empty())
            contrib *= mediaTransmittance(scene, p, wdir, dist, lambda, rng);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Hero-wavelength model-A splat: one shared aperture sample (drawn once, so the RNG
    // stream is identical whatever C is) and a single lens/occlusion geometry, then splat
    // each of the `nUp` live wavelengths with its own beta[i]/rho[i]. The finite-lens
    // pupil is achromatic (thin-lens geometry, no per-λ dispersion), so C wavelengths
    // legitimately share the connection — same as mode B. Per-λ ordering matches
    // connectLens() above.
    void connectLensHero(const Scene& scene, const Camera& cam, Film& film,
                         const Vec3& p, const Vec3& n, const Vec3& ng, const Vec3& wi,
                         const double* lam, const double* beta, const double* rho, int nUp,
                         Pcg32& rng, const Vec3* plane = nullptr,
                         const GlossySplat* gs = nullptr) const {   // see connectLens / connect
        if (cam.hasLens()) {
            connectLensHeroPhysical(scene, cam, film, p, n, ng, wi, lam, beta, rho, nUp, rng, plane,
                                    gs);
            return;
        }
        double R = cam.apertureR;
        double rr = R * std::sqrt(rng.uniform());
        double a  = 2.0 * PI * rng.uniform();
        Vec3 A = cam.eye + cam.u * (rr * std::cos(a)) + cam.v * (rr * std::sin(a));
        if (plane && dot(*plane, A - p) <= 0.0) return;
        Vec3 toA = A - p;
        double dist = length(toA);
        if (dist < 1e-9) return;
        Vec3 wdir = toA / dist;
        double cosSurf;
        if (!camSideOk(n, ng, wdir, gs != nullptr, cosSurf)) return;   // the pupil (connect())
        double cosLens = -dot(wdir, cam.w);
        if (cosLens <= 1e-6) return;                     // not heading toward the film
        int px, py;
        if (!cam.lensImage(A, wdir, px, py)) return;
        if (scene.occluded(p + ng * 1e-6, wdir, dist - 2e-6, 1e-6, /*camLeg=*/true)) return;
        double corr = shadingAdjointCorr(wi, wdir, n, ng);
        double cellNorm = 1.0 / (cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
        for (int i = 0; i < nUp; ++i) {
            double contrib = beta[i] * (gs ? PI * gs->f(wdir, lam[i]) : rho[i]) * cosSurf * corr *
                             cosLens * (R * R) / (dist * dist);
            contrib *= cellNorm;
            if (!scene.media.empty())
                contrib *= mediaTransmittance(scene, p, wdir, dist, lam[i], rng);
            film.add(px, py, Vec3(cieX(lam[i]), cieY(lam[i]), cieZ(lam[i])) * contrib);
        }
    }

    // Model A lens splat for a VOLUME scattering vertex (fog). As connectLens but the
    // surface BRDF*cosSurf is replaced by albedo*phase; the phase function carries no
    // 1/pi, so the pupil pdf's pi R^2 stays. wIn is the photon's incoming direction.
    void connectLensVolume(const Scene& scene, const Medium& med, const Camera& cam, Film& film,
                           const Vec3& p, const Vec3& wIn, double lambda, double beta,
                           Pcg32& rng) const {
        if (cam.hasLens()) {
            connectVolumeLensPhysical(scene, cam, film, p, lambda, beta, 0.0, &wIn, &med, rng);
            return;
        }
        double R = cam.apertureR;
        double rr = R * std::sqrt(rng.uniform());
        double a  = 2.0 * PI * rng.uniform();
        Vec3 A = cam.eye + cam.u * (rr * std::cos(a)) + cam.v * (rr * std::sin(a));
        Vec3 toA = A - p;
        double dist = length(toA);
        if (dist < 1e-9) return;
        Vec3 wdir = toA / dist;
        double cosLens = -dot(wdir, cam.w);
        if (cosLens <= 1e-6) return;
        int px, py;
        if (!cam.lensImage(A, wdir, px, py)) return;
        if (scene.occluded(p + wdir * 1e-6, wdir, dist - 2e-6, 1e-6, /*camLeg=*/true)) return;

        double ph = med.phaseValue(dot(wIn, wdir), lambda); // scattering medium's phase (HG or rainbow)
        double Lambda = med.albedo(lambda);
        double contrib = beta * Lambda * ph * cosLens * (PI * R * R) / (dist * dist);
        // Same flux->film-irradiance normaliser as connectLens (see there): divide the
        // pupil flux deposited in the cell by the physical cell area so a fog vertex
        // matches B's absolute scale in absolute-EV modes (per-camera constant; auto-
        // exposed scenes unaffected).
        contrib *= 1.0 / (cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
        contrib *= mediaTransmittance(scene, p, wdir, dist, lambda, rng);   // all media
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Model B camera splat for a VOLUME EMISSION vertex (blackbody "fire"). Identical
    // geometry to connectVolume, but the in-scatter term albedo*phase is replaced by
    // isotropic emission 1/(4π): the hot voxel radiates equally in all directions, so
    // there is no incoming direction and no phase. `beta` already carries the emission
    // strength (grandTotal·κ_e(x,λ)/meanKe); the /(dist²·Ω) converts the volume-birth
    // integral into the emission line-integral seen by the pixel. Fog transmittance
    // still applies (the flame's own soot self-absorbs its glow).
    void connectEmissionVolume(const Scene& scene, const Camera& cam, Film& film,
                               const Vec3& p, double lambda, double beta, Pcg32& rng) const {
        Vec3 toCam = cam.eye - p;
        double dist = length(toCam);
        Vec3 wdir = toCam / dist;
        int px, py; double cosCam, dist2;
        if (!cam.project(p, px, py, cosCam, dist2)) return;
        if (scene.occluded(p + wdir * 1e-6, wdir, dist - 2e-6, 1e-6, /*camLeg=*/true)) return;
        double omega = cam.pixelSolidAngle(cosCam);
        double contrib = beta * (1.0 / (4.0 * PI)) / (dist2 * omega);
        contrib *= mediaTransmittance(scene, p, wdir, dist, lambda, rng);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Model A (finite-lens) camera splat for a volume emission vertex. As
    // connectLensVolume, with the albedo*phase in-scatter term replaced by the isotropic
    // 1/(4π) emission term.
    void connectEmissionLensVolume(const Scene& scene, const Camera& cam, Film& film,
                                   const Vec3& p, double lambda, double beta, Pcg32& rng) const {
        if (cam.hasLens()) {
            connectVolumeLensPhysical(scene, cam, film, p, lambda, beta, 1.0 / (4.0 * PI),
                                      nullptr, nullptr, rng);
            return;
        }
        double R = cam.apertureR;
        double rr = R * std::sqrt(rng.uniform());
        double a  = 2.0 * PI * rng.uniform();
        Vec3 A = cam.eye + cam.u * (rr * std::cos(a)) + cam.v * (rr * std::sin(a));
        Vec3 toA = A - p;
        double dist = length(toA);
        if (dist < 1e-9) return;
        Vec3 wdir = toA / dist;
        double cosLens = -dot(wdir, cam.w);
        if (cosLens <= 1e-6) return;
        int px, py;
        if (!cam.lensImage(A, wdir, px, py)) return;
        if (scene.occluded(p + wdir * 1e-6, wdir, dist - 2e-6, 1e-6, /*camLeg=*/true)) return;
        double contrib = beta * (1.0 / (4.0 * PI)) * cosLens * (PI * R * R) / (dist * dist);
        contrib *= 1.0 / (cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
        contrib *= mediaTransmittance(scene, p, wdir, dist, lambda, rng);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Splat a volume-emission vertex to every camera (pinhole B or finite lens A).
    void camSplatEmissionAll(const Scene& scene, const CamTarget* cams, int nCam,
                             const Vec3& p, double lambda, double beta, Pcg32& rng) const {
        for (int c = 0; c < nCam; ++c)
            if (cams[c].cam && cams[c].film) {
                if (lensCam(*cams[c].cam)) connectEmissionLensVolume(scene, *cams[c].cam, *cams[c].film, p, lambda, beta, rng);
                else          connectEmissionVolume(scene, *cams[c].cam, *cams[c].film, p, lambda, beta, rng);
            }
    }

    // Route a camera connection to the pinhole (model B) or the finite lens (model A).
    // `plane` (optional): an aimed quad's face (Emitter::nGeom); a camera point behind it gets
    // nothing -- the eye for the pinhole, the sampled pupil point for the lens.
    // `gs` (optional): a glossy vertex (GlossySplat), whose lobe replaces the Lambertian rho/pi.
    void camSplat(const Scene& scene, const Camera& cam, Film& film, const Vec3& p,
                  const Vec3& n, const Vec3& ng, const Vec3& wi, double lambda, double beta,
                  double rho, Pcg32& rng, const Vec3* plane = nullptr,
                  const GlossySplat* gs = nullptr) const {
        if (lensCam(cam)) connectLens(scene, cam, film, p, n, ng, wi, lambda, beta, rho, rng, nullptr,
                                  plane, gs);
        else if (!(plane && dot(*plane, cam.eye - p) <= 0.0))
            connect(scene, cam, film, p, n, ng, wi, lambda, beta, rho, rng, nullptr, gs);
    }

    // Splat a surface vertex to every camera target. In model B (the shared-pass case)
    // camSplat -> connect draws no RNG, so the loop is RNG-neutral; with nCam==1 this is
    // exactly the old single-camera call (model A draws its aperture sample once here).
    // `plane` (optional): an aimed quad's face, passed to camSplat.
    void camSplatAll(const Scene& scene, const CamTarget* cams, int nCam, const Vec3& p,
                     const Vec3& n, const Vec3& ng, const Vec3& wi, double lambda, double beta,
                     double rho, Pcg32& rng, const Vec3* plane = nullptr,
                     const GlossySplat* gs = nullptr) const {
        for (int c = 0; c < nCam; ++c)
            if (cams[c].cam && cams[c].film)
                camSplat(scene, *cams[c].cam, *cams[c].film, p, n, ng, wi, lambda, beta, rho, rng,
                         plane, gs);
    }

    // Fiber (MatType::Hair) analogue of camSplat/camSplatAll. The BCSDF and its projection
    // arrive together in `hs` (hair_shade.h), so there is no `rho` to pass: the 0.0 below is
    // the ignored Lambertian slot. `wi` is likewise unused — the fiber does not take the
    // Veach shading-normal correction — but it is kept in the signature so the call sites
    // read the same as the surface ones.
    void camSplatHair(const Scene& scene, const Camera& cam, Film& film, const Vec3& p,
                      const Vec3& n, const Vec3& ng, const Vec3& wi, double lambda,
                      double beta, const HairShade& hs, Pcg32& rng) const {
        if (lensCam(cam)) connectLens(scene, cam, film, p, n, ng, wi, lambda, beta, 0.0, rng, &hs);
        else          connect(scene, cam, film, p, n, ng, wi, lambda, beta, 0.0, rng, &hs);
    }
    void camSplatAllHair(const Scene& scene, const CamTarget* cams, int nCam, const Vec3& p,
                         const Vec3& n, const Vec3& ng, const Vec3& wi, double lambda,
                         double beta, const HairShade& hs, Pcg32& rng) const {
        for (int c = 0; c < nCam; ++c)
            if (cams[c].cam && cams[c].film)
                camSplatHair(scene, *cams[c].cam, *cams[c].film, p, n, ng, wi, lambda, beta, hs, rng);
    }

    // Hero-wavelength routing (mode A finite lens or mode B pinhole), splatting all `nUp`
    // live wavelengths through one shared connection. Mode B draws no RNG (RNG-neutral
    // over the camera loop); mode A draws its single aperture sample once per camera.
    void camSplatHero(const Scene& scene, const Camera& cam, Film& film, const Vec3& p,
                      const Vec3& n, const Vec3& ng, const Vec3& wi, const double* lam,
                      const double* beta, const double* rho, int nUp, Pcg32& rng,
                      const Vec3* plane = nullptr,
                      const GlossySplat* gs = nullptr) const {       // see camSplat
        if (lensCam(cam))
            connectLensHero(scene, cam, film, p, n, ng, wi, lam, beta, rho, nUp, rng, plane, gs);
        else if (!(plane && dot(*plane, cam.eye - p) <= 0.0))
            connectHero(scene, cam, film, p, n, ng, wi, lam, beta, rho, nUp, rng, gs);
    }
    void camSplatAllHero(const Scene& scene, const CamTarget* cams, int nCam, const Vec3& p,
                         const Vec3& n, const Vec3& ng, const Vec3& wi, const double* lam,
                         const double* beta, const double* rho, int nUp, Pcg32& rng,
                         const Vec3* plane = nullptr,
                         const GlossySplat* gs = nullptr) const {    // see camSplatAll
        for (int c = 0; c < nCam; ++c)
            if (cams[c].cam && cams[c].film)
                camSplatHero(scene, *cams[c].cam, *cams[c].film, p, n, ng, wi, lam, beta, rho, nUp,
                             rng, plane, gs);
    }

    // ===================================================================
    //  Analytic specular connection through a smooth dielectric SPHERE.
    //  (Manifold next-event estimation, specialised to an analytic sphere.)
    //
    //  Mode B normally skips specular vertices for the camera connection, so a
    //  directly-viewed clear sphere is black (the SDS limitation). This routine
    //  restores the missing paths: a diffuse/emissive vertex p that lies behind
    //  the sphere is connected to the pinhole along the refracted chain
    //      eye -> P1 -> (glass) -> P2 -> p
    //  obeying Snell at both interfaces. The chain is found by a 1-D root solve in
    //  the plane(eye, p, centre) — a sphere's refraction path is planar by
    //  symmetry — and there may be several roots (multiple refracted images).
    //  The radiometric weight uses a ray-differential geometry factor (the
    //  footprint the pixel's beam covers at p), which is the specular Jacobian and
    //  reduces EXACTLY to connect()'s cosSurf/dist^2 as n->1 (verified in code).
    //  Smooth spheres only (a rough sphere reopens the lobe -> not a point path).
    // ===================================================================
    struct SphereRefr { Vec3 P1, P2, exitDir; double Tf = 0, innerLen = 0; };

    // Describes the photon vertex being connected through the glass: a diffuse
    // surface (Lambertian rho, normal np) or a volume in-scatter (medium albedo,
    // HG phase g, incoming dir wIn). The connection geometry is identical for both;
    // only the throughput term at the vertex differs (rho/pi*cosSurf vs albedo*phase),
    // exactly mirroring connect() vs connectVolume().
    struct SpecVtx {
        bool  volume = false;
        Vec3  np;                 // surface normal (surface vertices)
        Vec3  wIn;                // incoming photon direction (volume vertices)
        double g = 0;             // HG asymmetry (volume vertices)
        double weight = 0;        // surface: Lambertian rho ; volume: single-scatter albedo
        // An aimed quad (Emitter::nGeom) sends nothing behind its own face, whatever `np`
        // (its aim) says: `gated` turns on that second test, against `plane`.
        bool gated = false;
        Vec3 plane;
        // A rough glossy vertex (FWD-GLOSSY-INDIRECT): its lobe toward the first specular point,
        // at `lambda` here and at each wavelength in the mirror legs, replaces the Lambertian
        // rho/pi, and `weight` is unused. Null for a diffuse or volume vertex.
        const GlossySplat* glossy = nullptr;
        double lambda = 0.0;
        // Throughput at the vertex for a connection leaving toward `wP` (unit, toward
        // the sphere). Returns <0 to signal "reject" (camera-side behind a surface).
        double term(const Vec3& wP) const {
            if (volume) return weight * hgPhase(dot(wIn, wP), g);
            if (glossy) {
                const double c = dot(np, wP);
                return c <= 0.0 ? -1.0 : glossy->f(wP, lambda) * c;
            }
            if (gated && dot(plane, wP) <= 0.0) return -1.0;
            double cosSurf = dot(np, wP);
            return cosSurf <= 0.0 ? -1.0 : (weight / PI) * cosSurf;
        }
        // The WAVELENGTH-INDEPENDENT half of term(): everything but `weight` (which is
        // the per-λ albedo). The mirror connectors solve one shared geometry for the
        // whole hero bundle — a flat/metallic reflection does not disperse — so they
        // evaluate this once and multiply by each λ's weight. term() keeps its own
        // arithmetic rather than being expressed through this, so the dielectric-sphere
        // path's float ordering (and therefore its images) stay bit-identical.
        double shape(const Vec3& wP) const {
            if (volume) return hgPhase(dot(wIn, wP), g);
            if (gated && dot(plane, wP) <= 0.0) return -1.0;
            double cosSurf = dot(np, wP);
            return cosSurf <= 0.0 ? -1.0 : cosSurf / PI;
        }
    };

    // Trace a ray from `o` (outside sphere S) that ENTERS S, crosses the glass, and
    // EXITS. Fills P1 (entry), P2 (exit), exitDir (outward), the product of the two
    // Fresnel transmittances Tf, and the internal path length. False on miss / TIR.
    static bool traceThroughSphere2(const Vec3& o, const Vec3& d, const Sphere& S,
                                    double n, SphereRefr& out) {
        Vec3 oc = o - S.c;
        double b = dot(oc, d), c = dot(oc, oc) - S.r * S.r;
        double disc = b * b - c;
        if (disc < 0.0) return false;
        double sq = std::sqrt(disc);
        double t1 = -b - sq;
        if (t1 < 1e-7) return false;                    // entry must be ahead & outside
        Vec3 P1 = o + d * t1;
        Vec3 N1 = (P1 - S.c) * (1.0 / S.r);             // outward normal
        double cosI = -dot(d, N1);
        if (cosI <= 1e-6) return false;                 // must hit the front face
        double eta = 1.0 / n;
        double sin2t = eta * eta * (1.0 - cosI * cosI);
        if (sin2t >= 1.0) return false;                 // (cannot TIR entering a denser medium)
        double cosT = std::sqrt(1.0 - sin2t);
        Vec3 tin = normalize(d * eta + N1 * (eta * cosI - cosT));
        double rs = (cosI - n * cosT) / (cosI + n * cosT);
        double rp = (cosT - n * cosI) / (cosT + n * cosI);
        double Fe = 0.5 * (rs * rs + rp * rp);          // Fresnel reflectance, entry
        double sInner = -2.0 * dot(P1 - S.c, tin);      // second intersection param
        if (sInner <= 1e-9) return false;
        Vec3 P2 = P1 + tin * sInner;
        Vec3 N2 = (P2 - S.c) * (1.0 / S.r);             // outward normal at exit
        double cosI2 = dot(tin, N2);                    // incidence cosine inside (>0)
        if (cosI2 <= 1e-6) return false;
        double sin2t2 = n * n * (1.0 - cosI2 * cosI2);
        if (sin2t2 >= 1.0) return false;                // total internal reflection -> no exit
        double cosT2 = std::sqrt(1.0 - sin2t2);
        Vec3 exitDir = normalize(tin * n + N2 * (-(n * cosI2 - cosT2)));
        double rs2 = (n * cosI2 - cosT2) / (n * cosI2 + cosT2);
        double rp2 = (n * cosT2 - cosI2) / (n * cosT2 + cosI2);
        double Fx = 0.5 * (rs2 * rs2 + rp2 * rp2);      // Fresnel reflectance, exit
        out.P1 = P1; out.P2 = P2; out.exitDir = exitDir;
        out.Tf = (1.0 - Fe) * (1.0 - Fx);
        out.innerLen = sInner;
        return true;
    }

    // Connect vertex p (normal np, Lambertian weight rho) to the pinhole `cam`
    // THROUGH one smooth dielectric sphere S (glass index n). Adds the refracted
    // image of p seen in the sphere. Pinhole (mode B) only.
    void connectSpecularSphere(const Scene& scene, const Camera& cam, Film& film,
                               const Sphere& S, const Material& glass, double n,
                               const Vec3& p, const SpecVtx& vt, double lambda,
                               double beta, Pcg32& rng) const {
        const Vec3 O = S.c; const double r = S.r; const Vec3 eye = cam.eye;
        double dEyeO = length(eye - O);
        double dPO   = length(p - O);
        if (dPO   <  r * 0.9999) return;   // vertex inside the glass -> skip (MVP)
        if (dEyeO <= r * 0.9999) {         // eye inside the glass -> single-refraction path
            connectSpecularSphereInside(scene, cam, film, S, glass, n, p, vt, lambda,
                                        beta, rng);
            return;
        }
        if (dEyeO <= r * 1.0001) return;   // eye ~on the surface -> degenerate, skip

        // Plane(eye, p, O) with ex toward the eye; p has 2-D coords (px2,py2).
        Vec3 ex = (eye - O) * (1.0 / dEyeO);
        Vec3 ap = p - O;
        Vec3 perp = ap - ex * dot(ap, ex);
        double perpLen = length(perp);
        Vec3 ey;
        if (perpLen < 1e-9) { Vec3 tb; onb(ex, ey, tb); }   // axial: pick any perpendicular
        else                ey = perp * (1.0 / perpLen);
        double ex_e = dEyeO;                                // eye 2-D = (ex_e, 0)
        double px2 = dot(ap, ex), py2 = dot(ap, ey);        // p   2-D

        // In-plane trace: signed perpendicular distance of p from the exit ray, for
        // entry angle phi (measured in (ex,ey), passed as its cos/sin pair). Sets
        // valid on a real forward exit.
        auto trace2D = [&](double c1, double s1, bool& valid) -> double {
            valid = false;
            double P1x = r * c1, P1y = r * s1;
            double dinx = P1x - ex_e, diny = P1y;
            double dl = std::sqrt(dinx * dinx + diny * diny);
            if (dl < 1e-12) return 0.0;
            dinx /= dl; diny /= dl;
            double cosI = -(dinx * c1 + diny * s1);
            if (cosI <= 1e-6) return 0.0;                   // front-facing entry only
            double eta = 1.0 / n, sin2t = eta * eta * (1.0 - cosI * cosI);
            if (sin2t >= 1.0) return 0.0;
            double cosT = std::sqrt(1.0 - sin2t);
            double tinx = eta * dinx + (eta * cosI - cosT) * c1;
            double tiny = eta * diny + (eta * cosI - cosT) * s1;
            double tl = std::sqrt(tinx * tinx + tiny * tiny); tinx /= tl; tiny /= tl;
            double sInner = -2.0 * (P1x * tinx + P1y * tiny);
            if (sInner <= 1e-9) return 0.0;
            double P2x = P1x + tinx * sInner, P2y = P1y + tiny * sInner;
            double n2x = P2x / r, n2y = P2y / r;
            double cosI2 = tinx * n2x + tiny * n2y;
            if (cosI2 <= 1e-6) return 0.0;
            double sin2t2 = n * n * (1.0 - cosI2 * cosI2);
            if (sin2t2 >= 1.0) return 0.0;                  // TIR
            double cosT2 = std::sqrt(1.0 - sin2t2);
            double doutx = n * tinx - (n * cosI2 - cosT2) * n2x;
            double douty = n * tiny - (n * cosI2 - cosT2) * n2y;
            double dl2 = std::sqrt(doutx * doutx + douty * douty); doutx /= dl2; douty /= dl2;
            double fw = (px2 - P2x) * doutx + (py2 - P2y) * douty;
            if (fw <= 0.0) return 0.0;                      // p must be on the forward side
            valid = true;
            return doutx * (py2 - P2y) - douty * (px2 - P2x);
        };

        // Scan the front arc; bisect sign changes into chief entry angles (<=4 roots).
        const int NS = kSphScanN; double roots[4]; int nroot = 0;
        const SphScanTab& T = sphScanTab();
        double prevMiss = 0.0, prevPhi = 0.0; bool prevValid = false;
        for (int i = 0; i <= NS && nroot < 4; ++i) {
            double phi = -PI + (2.0 * PI) * i / NS;
            bool v; double mss = trace2D(T.c[i], T.s[i], v);
            if (v && prevValid && ((mss < 0.0) != (prevMiss < 0.0))) {
                double a = prevPhi, b = phi, fa = prevMiss;
                for (int k = 0; k < 40; ++k) {
                    double mid = 0.5 * (a + b); bool vm; double fm = trace2D(std::cos(mid), std::sin(mid), vm);
                    if (!vm) break;
                    if ((fm < 0.0) != (fa < 0.0)) b = mid; else { a = mid; fa = fm; }
                }
                roots[nroot++] = 0.5 * (a + b);
            }
            prevMiss = mss; prevValid = v; prevPhi = phi;
        }

        for (int ri = 0; ri < nroot; ++ri) {
            double phi = roots[ri];
            Vec3 P1chief = O + ex * (r * std::cos(phi)) + ey * (r * std::sin(phi));
            Vec3 d0 = normalize(P1chief - eye);
            SphereRefr ch;
            if (!traceThroughSphere2(eye, d0, S, n, ch)) continue;

            // Ray-differential geometry factor: perturb the eye direction by eps in
            // two orthogonal directions, trace both through the same interfaces, and
            // measure the footprint they cover on the plane through p (normal =
            // chief exit dir). G = dOmega_eye / dA_p = eps^2 / |dA x dB|.
            Vec3 a1, a2; onb(d0, a1, a2);
            const double eps = 2e-4;
            SphereRefr rA, rB;
            if (!traceThroughSphere2(eye, normalize(d0 + a1 * eps), S, n, rA)) continue;
            if (!traceThroughSphere2(eye, normalize(d0 + a2 * eps), S, n, rB)) continue;
            Vec3 e1, e2; onb(ch.exitDir, e1, e2);
            auto planeOff = [&](const SphereRefr& R, double& ox, double& oy) {
                double denom = dot(R.exitDir, ch.exitDir);
                if (std::fabs(denom) < 1e-9) denom = (denom < 0 ? -1e-9 : 1e-9);
                double s = dot(p - R.P2, ch.exitDir) / denom;
                Vec3 off = (R.P2 + R.exitDir * s) - p;
                ox = dot(off, e1); oy = dot(off, e2);
            };
            double ax, ay, bx, by;
            planeOff(rA, ax, ay); planeOff(rB, bx, by);
            double jac = std::fabs(ax * by - ay * bx);
            if (jac < 1e-24) continue;                      // caustic singularity guard
            double G = (eps * eps) / jac;

            int px, py; double cosCam, dist2e;
            if (!cam.project(P1chief, px, py, cosCam, dist2e)) continue;
            double omega = cam.pixelSolidAngle(cosCam);
            if (omega <= 0.0) continue;

            Vec3 wP = ch.P2 - p; double dP2 = length(wP);
            if (dP2 < 1e-9) continue;
            wP = wP * (1.0 / dP2);
            double term = vt.term(wP);
            if (term < 0.0) continue;                       // camera side behind the surface

            double contrib = beta * term * G * ch.Tf / omega;
            if (contrib <= 0.0) continue;
            double aGlass = glass.absorb(lambda);           // Beer-Lambert inside the glass
            if (aGlass > 0.0) contrib *= std::exp(-aGlass * ch.innerLen);

            // Visibility on the two outer segments (the connecting sphere's own
            // surface is excluded by shortening maxDist just short of the endpoint).
            if (scene.occluded(p + wP * 1e-6, wP, dP2 - 2e-6)) continue;
            Vec3 wE = eye - ch.P1; double dE = length(wE); wE = wE * (1.0 / dE);
            if (scene.occluded(ch.P1 + wE * 1e-6, wE, dE - 2e-6, 1e-6, /*camLeg=*/true)) continue;

            // Fog transmittance on the two outer (vacuum-side) segments only; the
            // interior segment is solid glass (its absorption is the Beer-Lambert above).
            if (!scene.media.empty()) {
                contrib *= mediaTransmittance(scene, p,     wP, dP2, lambda, rng);
                contrib *= mediaTransmittance(scene, ch.P1, wE, dE,  lambda, rng);
            }
            film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
        }
    }

    // Single-interface refraction datum: exit surface point, refracted (outward)
    // direction, Fresnel transmittance, and the interior path length eye->P1.
    struct SphereRefr1 { Vec3 P1, exitDir; double Tf = 0, innerLen = 0; };

    // Trace a ray from `o` (INSIDE sphere S) to its exit through the surface, refract
    // glass->vacuum. False on TIR / degenerate. (Sign convention matches the exit
    // interface of traceThroughSphere2.)
    static bool traceOutOfSphere(const Vec3& o, const Vec3& d, const Sphere& S,
                                 double n, SphereRefr1& out) {
        Vec3 oc = o - S.c;
        double b = dot(oc, d), c = dot(oc, oc) - S.r * S.r;
        double disc = b * b - c;
        if (disc <= 0.0) return false;                  // (o inside => c<0 => disc>0)
        double sq = std::sqrt(disc);
        double t1 = -b + sq;                            // far root: the exit ahead
        if (t1 < 1e-7) return false;
        Vec3 P1 = o + d * t1;
        Vec3 N1 = (P1 - S.c) * (1.0 / S.r);             // outward normal
        double cosI = dot(d, N1);                       // outgoing (>0)
        if (cosI <= 1e-6) return false;
        double sin2t = n * n * (1.0 - cosI * cosI);     // glass -> vacuum
        if (sin2t >= 1.0) return false;                 // total internal reflection
        double cosT = std::sqrt(1.0 - sin2t);
        Vec3 exitDir = normalize(d * n - N1 * (n * cosI - cosT));
        double rs = (n * cosI - cosT) / (n * cosI + cosT);
        double rp = (n * cosT - cosI) / (n * cosT + cosI);
        double F = 0.5 * (rs * rs + rp * rp);
        out.P1 = P1; out.exitDir = exitDir; out.Tf = 1.0 - F; out.innerLen = t1;
        return true;
    }

    // Connect an EXTERIOR vertex p to a pinhole whose eye sits INSIDE dielectric
    // sphere S: light travels p -> P1 (surface) -> refracts once -> eye. This is the
    // path the camera sees while flying THROUGH the glass. Planar in plane(eye,p,O).
    void connectSpecularSphereInside(const Scene& scene, const Camera& cam, Film& film,
                                     const Sphere& S, const Material& glass, double n,
                                     const Vec3& p, const SpecVtx& vt, double lambda,
                                     double beta, Pcg32& rng) const {
        const Vec3 O = S.c; const double r = S.r; const Vec3 eye = cam.eye;
        double dEyeO = length(eye - O);

        // Plane(eye, p, O): ex toward the eye (or any axis if eye ~at center).
        Vec3 ex, ey;
        if (dEyeO < 1e-9) { Vec3 tb; onb(normalize(p - O), ex, tb); }
        else              ex = (eye - O) * (1.0 / dEyeO);
        Vec3 ap = p - O;
        Vec3 perp = ap - ex * dot(ap, ex);
        double perpLen = length(perp);
        if (perpLen < 1e-9) { Vec3 tb; onb(ex, ey, tb); }
        else                ey = perp * (1.0 / perpLen);
        double ex_e = dot(eye - O, ex), ey_e = dot(eye - O, ey);   // eye 2-D
        double px2 = dot(ap, ex), py2 = dot(ap, ey);               // p   2-D

        // In-plane trace: signed perp distance of p from the once-refracted exit ray
        // leaving the surface point at angle phi (passed as its cos/sin pair). valid
        // on a real forward exit.
        auto trace2D = [&](double c1, double s1, bool& valid) -> double {
            valid = false;
            double P1x = r * c1, P1y = r * s1;
            double dinx = P1x - ex_e, diny = P1y - ey_e;
            double dl = std::sqrt(dinx * dinx + diny * diny);
            if (dl < 1e-12) return 0.0;
            dinx /= dl; diny /= dl;
            double cosI = dinx * c1 + diny * s1;            // outgoing across surface (>0)
            if (cosI <= 1e-6) return 0.0;
            double sin2t = n * n * (1.0 - cosI * cosI);
            if (sin2t >= 1.0) return 0.0;                   // TIR
            double cosT = std::sqrt(1.0 - sin2t);
            double k = n * cosI - cosT;
            double doutx = n * dinx - k * c1, douty = n * diny - k * s1;
            double dl2 = std::sqrt(doutx * doutx + douty * douty); doutx /= dl2; douty /= dl2;
            double fw = (px2 - P1x) * doutx + (py2 - P1y) * douty;
            if (fw <= 0.0) return 0.0;
            valid = true;
            return doutx * (py2 - P1y) - douty * (px2 - P1x);
        };

        // Scan the full circle; bisect sign changes into chief exit angles (<=2 roots).
        const int NS = kSphScanN; double roots[4]; int nroot = 0;
        const SphScanTab& T = sphScanTab();
        double prevMiss = 0.0, prevPhi = 0.0; bool prevValid = false;
        for (int i = 0; i <= NS && nroot < 4; ++i) {
            double phi = -PI + (2.0 * PI) * i / NS;
            bool v; double mss = trace2D(T.c[i], T.s[i], v);
            if (v && prevValid && ((mss < 0.0) != (prevMiss < 0.0))) {
                double a = prevPhi, b = phi, fa = prevMiss;
                for (int k = 0; k < 40; ++k) {
                    double mid = 0.5 * (a + b); bool vm; double fm = trace2D(std::cos(mid), std::sin(mid), vm);
                    if (!vm) break;
                    if ((fm < 0.0) != (fa < 0.0)) b = mid; else { a = mid; fa = fm; }
                }
                roots[nroot++] = 0.5 * (a + b);
            }
            prevMiss = mss; prevValid = v; prevPhi = phi;
        }

        for (int ri = 0; ri < nroot; ++ri) {
            double phi = roots[ri];
            Vec3 P1chief = O + ex * (r * std::cos(phi)) + ey * (r * std::sin(phi));
            Vec3 d0 = normalize(P1chief - eye);
            SphereRefr1 ch;
            if (!traceOutOfSphere(eye, d0, S, n, ch)) continue;

            // Ray-differential geometry factor G = dOmega_eye / dA_p.
            Vec3 a1, a2; onb(d0, a1, a2);
            const double eps = 2e-4;
            SphereRefr1 rA, rB;
            if (!traceOutOfSphere(eye, normalize(d0 + a1 * eps), S, n, rA)) continue;
            if (!traceOutOfSphere(eye, normalize(d0 + a2 * eps), S, n, rB)) continue;
            Vec3 e1, e2; onb(ch.exitDir, e1, e2);
            auto planeOff = [&](const SphereRefr1& R, double& ox, double& oy) {
                double denom = dot(R.exitDir, ch.exitDir);
                if (std::fabs(denom) < 1e-9) denom = (denom < 0 ? -1e-9 : 1e-9);
                double s = dot(p - R.P1, ch.exitDir) / denom;
                Vec3 off = (R.P1 + R.exitDir * s) - p;
                ox = dot(off, e1); oy = dot(off, e2);
            };
            double ax, ay, bx, by;
            planeOff(rA, ax, ay); planeOff(rB, bx, by);
            double jac = std::fabs(ax * by - ay * bx);
            if (jac < 1e-24) continue;                      // caustic singularity guard
            double G = (eps * eps) / jac;

            int px, py; double cosCam, dist2e;
            if (!cam.project(P1chief, px, py, cosCam, dist2e)) continue;
            double omega = cam.pixelSolidAngle(cosCam);
            if (omega <= 0.0) continue;

            Vec3 wP = ch.P1 - p; double dP = length(wP);
            if (dP < 1e-9) continue;
            wP = wP * (1.0 / dP);
            double term = vt.term(wP);
            if (term < 0.0) continue;

            double contrib = beta * term * G * ch.Tf / omega;
            if (contrib <= 0.0) continue;
            double aGlass = glass.absorb(lambda);           // Beer-Lambert eye->P1 (in glass)
            if (aGlass > 0.0) contrib *= std::exp(-aGlass * ch.innerLen);

            // Visibility on the exterior segment p -> P1 only (eye -> P1 is inside glass).
            if (scene.occluded(p + wP * 1e-6, wP, dP - 2e-6)) continue;

            // Fog transmittance on the exterior segment only (interior is solid glass).
            if (!scene.media.empty())
                contrib *= mediaTransmittance(scene, p, wP, dP, lambda, rng);

            film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
        }
    }

    // ===================================================================
    //  Analytic specular connection by REFLECTION in a perfect mirror.
    //
    //  The same missing-path problem connectSpecularSphere solves for glass, solved
    //  for mirrors — and this is the case that actually shows: mode B renders every
    //  mirror surface pure BLACK, because a delta reflection has no pinhole
    //  connection, so a mirrored wall (and every emitter reflected in it) simply is
    //  not drawn, while mode R draws it correctly.
    //
    //  Reflection is far kinder than refraction: UNFOLDING the eye across the mirror
    //  (eye' = the eye's mirror image) turns the bent chain  p -> R -> eye  into the
    //  straight segment  p -> eye'. So there is no root solve for a plane — R is just
    //  where that segment crosses the plane — and the specular Jacobian, which the
    //  sphere code has to estimate with ray differentials, is exactly
    //      G = dOmega_eye / dA_perp = 1 / D^2,   D = |p - eye'| = |p - R| + |R - eye|
    //  i.e. connect()'s own 1/dist^2 measured along the folded-out path. The vertex
    //  term and the pixel solid angle are shared with every other connector, so the
    //  estimator is connect() with a longer, bent distance and one reflectance factor.
    //
    //  A curved (spherical) mirror keeps the unfolding idea but loses the closed form,
    //  so it reuses the sphere machinery: the reflection point is a 1-D root in
    //  plane(eye, p, centre) and G comes from ray differentials, exactly as for glass.
    //
    //  Limits: ONE specular vertex per connection (a mirror seen in a mirror is still
    //  missing), pinhole only, and flat mirrors must be authored as triangles (an
    //  instanced/BLAS mirror is not collected — see Scene::buildMirrorPlanes).
    // ===================================================================

    // Reflect a ray off sphere S at its first intersection ahead of `o`. Works from
    // either side (a convex mirror ball seen from outside, a mirrored room seen from
    // within), since the reflecting normal is taken against the ray.
    struct SphereRefl { Vec3 P1, exitDir; };
    static bool reflectOffSphere(const Vec3& o, const Vec3& d, const Sphere& S,
                                 SphereRefl& out) {
        Vec3 oc = o - S.c;
        double b = dot(oc, d), c = dot(oc, oc) - S.r * S.r;
        double disc = b * b - c;
        if (disc <= 0.0) return false;
        double sq = std::sqrt(disc);
        double t = -b - sq;                             // near root (exterior origin)
        if (t < 1e-7) t = -b + sq;                      // interior origin: the far one
        if (t < 1e-7) return false;
        Vec3 P1 = o + d * t;
        Vec3 N = (P1 - S.c) * (1.0 / S.r);              // outward normal
        double cn = dot(d, N);
        if (std::fabs(cn) <= 1e-6) return false;        // grazing: reflection is degenerate
        out.P1 = P1;
        out.exitDir = normalize(d - N * (2.0 * cn));    // reflect(d, N); sign of N is moot
        return true;
    }

    // Is the surface the eye actually sees at `hitPoint` (found by a closest-hit along
    // the connection's eye-side leg) the mirror we solved the geometry for? One BVH
    // query settles BOTH questions the connection needs — "is there really mirror
    // material at that spot" (containment within the authored panel) and "is the leg
    // from the eye clear" — and hands back the Hit, so the reflectance can be read at
    // the exact texel/pattern value the surface has there.
    static bool mirrorSeenAt(const Scene& scene, const Vec3& eye, const Vec3& wE,
                             double dE, Hit& hm) {
        // This leg starts AT THE EYE, so it is a camera ray and a `hide_camera` flat must
        // not answer either of the two questions it asks — it is neither the mirror nor a
        // legitimate blocker of the view. See Material::hideCamera.
        hm = scene.closestHit(Ray{eye, wE}, 1e-6, nullptr, /*skipHair=*/false,
                              /*skipCamHidden=*/true);
        if (!hm.valid) return false;
        if (std::fabs(hm.t - dE) > 1e-4 * (1.0 + dE)) return false;   // something in front
        if (hm.matId < 0 || hm.matId >= (int)scene.mats.size()) return false;
        return isPlanarMirrorMat(scene.mats[hm.matId]);
    }

    // Splat the `nUp` live wavelengths of a photon vertex through one connection whose
    // geometry has already been solved: the vertex is at p, the light leaves it toward
    // wP over a distance dP to the mirror point R, the mirror hands it to the eye over
    // dE, and the whole unfolded path is D long. Shared by the planar and spherical
    // mirror connectors — everything below this point is wavelength-dependent only
    // through the mirror's reflectance and the fog.
    void splatMirrorLegs(const Scene& scene, Film& film, const Material& mm, const Hit& hm,
                         int px, int py, double omega, const Vec3& p, const Vec3& wP,
                         double dP, const Vec3& R, const Vec3& wRE, double dE, double D,
                         double shape, const double* lam, const double* beta,
                         const double* w, int nUp, Pcg32& rng,
                         const GlossySplat* gs = nullptr) const {
        const double invD2Omega = 1.0 / (D * D * omega);
        for (int i = 0; i < nUp; ++i) {
            double refl = clamp01(reflectSlot(scene, mm, hm, lam[i]));
            // A glossy vertex's lobe toward wP, at this wavelength (shape is cos/pi there).
            const double wi = gs ? gs->f(wP, lam[i]) * PI : w[i];
            double contrib = beta[i] * (wi * shape) * refl * invD2Omega;
            if (contrib <= 0.0) continue;
            if (!scene.media.empty()) {
                contrib *= mediaTransmittance(scene, p, wP,  dP, lam[i], rng);
                contrib *= mediaTransmittance(scene, R, wRE, dE, lam[i], rng);
            }
            film.add(px, py, Vec3(cieX(lam[i]), cieY(lam[i]), cieZ(lam[i])) * contrib);
        }
    }

    // Connect vertex p to the pinhole `cam` by reflection in one PLANAR mirror.
    void connectSpecularPlane(const Scene& scene, const Camera& cam, Film& film,
                              const Scene::MirrorPlane& mp, const Vec3& p, const SpecVtx& vt,
                              const double* lam, const double* beta, const double* w,
                              int nUp, Pcg32& rng) const {
        const Vec3 eye = cam.eye;
        const double se = dot(mp.n, eye) - mp.d;
        const double sp = dot(mp.n, p)   - mp.d;
        // A mirror reflects whichever face you look at, but the eye and the vertex must
        // be on the SAME face — otherwise the "reflection" is through the mirror's back.
        if (!(se * sp > 0.0)) return;
        const Vec3 eyeM = eye - mp.n * (2.0 * se);        // the eye's mirror image
        Vec3 wP = eyeM - p;
        const double D = length(wP);                      // unfolded path length
        if (D < 1e-9) return;
        wP = wP * (1.0 / D);
        const double shape = vt.shape(wP);
        if (shape < 0.0) return;                          // camera side behind the surface
        const double denom = dot(mp.n, wP);
        if (std::fabs(denom) < 1e-12) return;
        const double dP = (mp.d - dot(mp.n, p)) / denom;  // p + wP*dP lands on the plane
        if (dP <= 1e-7 || dP >= D - 1e-7) return;
        const Vec3 R = p + wP * dP;
        // Cheap extent reject before paying for the BVH query that confirms the mirror.
        if (R.x < mp.lo.x || R.x > mp.hi.x || R.y < mp.lo.y || R.y > mp.hi.y ||
            R.z < mp.lo.z || R.z > mp.hi.z) return;

        int px, py; double cosCam, dist2e;
        if (!cam.project(R, px, py, cosCam, dist2e)) return;
        const double omega = cam.pixelSolidAngle(cosCam);
        if (omega <= 0.0) return;

        const double dE = D - dP;                         // |R - eye| (R lies on the plane)
        Vec3 wRE = (eye - R) * (1.0 / dE);                // mirror -> eye
        // Light-side leg first (stopping just short of the mirror's own surface): the
        // any-hit occlusion walk is cheaper than the closest-hit that confirms the
        // mirror, so a shadowed connection never pays for the expensive query. Both
        // tests must pass and neither draws RNG, so the order is unobservable.
        if (scene.occluded(p + wP * 1e-6, wP, dP - 2e-6)) return;
        Hit hm;
        if (!mirrorSeenAt(scene, eye, -wRE, dE, hm) || !scene.mirrorClaims(mp, hm)) return;

        splatMirrorLegs(scene, film, scene.mats[hm.matId], hm, px, py, omega,
                        p, wP, dP, R, wRE, dE, D, shape, lam, beta, w, nUp, rng, vt.glossy);
    }

    // Connect vertex p to the pinhole `cam` by reflection in one SPHERICAL mirror.
    // Same 1-D root solve + ray-differential Jacobian as the glass sphere, with a
    // single reflection in place of the two refractions.
    void connectSpecularSphereMirror(const Scene& scene, const Camera& cam, Film& film,
                                     const Sphere& S, const Vec3& p, const SpecVtx& vt,
                                     const double* lam, const double* beta, const double* w,
                                     int nUp, Pcg32& rng) const {
        const Vec3 O = S.c; const double r = S.r; const Vec3 eye = cam.eye;
        const double dEyeO = length(eye - O);
        const double dPO   = length(p - O);
        if (std::fabs(dEyeO - r) < 1e-4 * r) return;      // eye ~on the surface: degenerate
        const bool outside = dEyeO > r;
        // The vertex has to be on the same side of the silvering as the eye, or there is
        // no reflection joining them (a mirror is opaque).
        if (outside ? (dPO <= r * 1.0001) : (dPO >= r * 0.9999)) return;

        // Plane(eye, p, O) — the reflection path off a sphere is planar by symmetry.
        Vec3 ex, ey;
        if (dEyeO < 1e-9) { Vec3 tb; onb(normalize(p - O), ex, tb); }
        else              ex = (eye - O) * (1.0 / dEyeO);
        const Vec3 ap = p - O;
        const Vec3 perp = ap - ex * dot(ap, ex);
        const double perpLen = length(perp);
        if (perpLen < 1e-9) { Vec3 tb; onb(ex, ey, tb); }
        else                ey = perp * (1.0 / perpLen);
        const double ex_e = dot(eye - O, ex), ey_e = dot(eye - O, ey);
        const double px2 = dot(ap, ex), py2 = dot(ap, ey);

        // Signed perpendicular distance of p from the ray reflected at surface angle phi.
        auto trace2D = [&](double c1, double s1, bool& valid) -> double {
            valid = false;
            const double P1x = r * c1, P1y = r * s1;
            double dinx = P1x - ex_e, diny = P1y - ey_e;
            const double dl = std::sqrt(dinx * dinx + diny * diny);
            if (dl < 1e-12) return 0.0;
            dinx /= dl; diny /= dl;
            const double cn = dinx * c1 + diny * s1;      // dot(din, outward normal)
            if (std::fabs(cn) <= 1e-6) return 0.0;
            // Reachability: from OUTSIDE only the front face is hit first (cn < 0);
            // from inside the ray always leaves through the far face (cn > 0).
            if (outside != (cn < 0.0)) return 0.0;
            const double doutx = dinx - 2.0 * cn * c1, douty = diny - 2.0 * cn * s1;
            const double fw = (px2 - P1x) * doutx + (py2 - P1y) * douty;
            if (fw <= 0.0) return 0.0;                    // p must be on the forward side
            valid = true;
            return doutx * (py2 - P1y) - douty * (px2 - P1x);
        };

        // Scan the circle; bisect sign changes into chief reflection angles.
        const int NS = kSphScanN; double roots[4]; int nroot = 0;
        const SphScanTab& T = sphScanTab();
        double prevMiss = 0.0, prevPhi = 0.0; bool prevValid = false;
        for (int i = 0; i <= NS && nroot < 4; ++i) {
            const double phi = -PI + (2.0 * PI) * i / NS;
            bool v; const double mss = trace2D(T.c[i], T.s[i], v);
            if (v && prevValid && ((mss < 0.0) != (prevMiss < 0.0))) {
                double a = prevPhi, b = phi, fa = prevMiss;
                for (int k = 0; k < 40; ++k) {
                    const double mid = 0.5 * (a + b);
                    bool vm; const double fm = trace2D(std::cos(mid), std::sin(mid), vm);
                    if (!vm) break;
                    if ((fm < 0.0) != (fa < 0.0)) b = mid; else { a = mid; fa = fm; }
                }
                roots[nroot++] = 0.5 * (a + b);
            }
            prevMiss = mss; prevValid = v; prevPhi = phi;
        }

        for (int ri = 0; ri < nroot; ++ri) {
            const double phi = roots[ri];
            const Vec3 P1chief = O + ex * (r * std::cos(phi)) + ey * (r * std::sin(phi));
            // Everything below is pure and RNG-free, so the tests are ordered cheapest
            // first — an off-frame or backfacing connection rejects before paying for
            // the two extra differential reflections, and the BVH queries come last,
            // any-hit before closest-hit. Every value is computed by the same
            // expression as before, so the accepted set and the splats are unchanged.
            int px, py; double cosCam, dist2e;
            if (!cam.project(P1chief, px, py, cosCam, dist2e)) continue;
            const double omega = cam.pixelSolidAngle(cosCam);
            if (omega <= 0.0) continue;

            const Vec3 d0 = normalize(P1chief - eye);
            SphereRefl ch;
            if (!reflectOffSphere(eye, d0, S, ch)) continue;

            Vec3 wP = ch.P1 - p; const double dP = length(wP);
            if (dP < 1e-9) continue;
            wP = wP * (1.0 / dP);
            const double shape = vt.shape(wP);
            if (shape < 0.0) continue;

            Vec3 toEye = eye - ch.P1; const double dE = length(toEye);
            if (dE < 1e-9) continue;
            const Vec3 wRE = toEye * (1.0 / dE);

            // Ray-differential geometry factor G = dOmega_eye / dA_perp at p.
            Vec3 a1, a2; onb(d0, a1, a2);
            const double eps = 2e-4;
            SphereRefl rA, rB;
            if (!reflectOffSphere(eye, normalize(d0 + a1 * eps), S, rA)) continue;
            if (!reflectOffSphere(eye, normalize(d0 + a2 * eps), S, rB)) continue;
            Vec3 e1, e2; onb(ch.exitDir, e1, e2);
            auto planeOff = [&](const SphereRefl& Rf, double& ox, double& oy) {
                double den = dot(Rf.exitDir, ch.exitDir);
                if (std::fabs(den) < 1e-9) den = (den < 0 ? -1e-9 : 1e-9);
                const double s = dot(p - Rf.P1, ch.exitDir) / den;
                const Vec3 off = (Rf.P1 + Rf.exitDir * s) - p;
                ox = dot(off, e1); oy = dot(off, e2);
            };
            double ax, ay, bx, by;
            planeOff(rA, ax, ay); planeOff(rB, bx, by);
            const double jac = std::fabs(ax * by - ay * bx);
            if (jac < 1e-24) continue;                    // caustic singularity guard
            // The same G the flat mirror gets in closed form as 1/D^2; expressed here as
            // an equivalent unfolded distance so both connectors share one splat.
            const double G = (eps * eps) / jac;
            const double D = 1.0 / std::sqrt(G);

            if (scene.occluded(p + wP * 1e-6, wP, dP - 2e-6)) continue;
            Hit hm;
            if (!mirrorSeenAt(scene, eye, -wRE, dE, hm)) continue;

            splatMirrorLegs(scene, film, scene.mats[hm.matId], hm, px, py, omega,
                            p, wP, dP, ch.P1, wRE, dE, D, shape, lam, beta, w, nUp, rng, vt.glossy);
        }
    }

    // Splat vertex p to every camera through every smooth dielectric sphere (the
    // refracted image of p) and off every perfect mirror (the reflected image of p).
    // Mode B (pinhole) only; draws no aperture RNG. A camera with a physical lens is
    // skipped: these closed-form connections aim at the pinhole, which it does not have.
    //
    // `w[i]` is the vertex's per-λ weight (Lambertian rho / single-scatter albedo);
    // vt.weight is ignored, and vt is copied per λ for the dielectric path. The loop
    // nesting of the dielectric-sphere connection (λ, then sphere, then camera) is the
    // one the scalar and hero wrappers used before this was factored, so its film
    // accumulation order — and therefore its images — are unchanged bit for bit.
    void camSpecularSplatAllVtxN(const Scene& scene, const CamTarget* cams, int nCam,
                                 const Vec3& p, const SpecVtx& vt, const double* lam,
                                 const double* beta, const double* w, int nUp,
                                 Pcg32& rng) const {
        if (lensMode || forwardCatch) return;               // finite-lens/catch not supported
        if (!specularSideSplats) return;                    // mode P's blurred split
        // Scene::dielSphereIdx / mirrorSphereIdx replace the old scan of EVERY sphere's
        // material per vertex (per λ, for the dielectrics). The lists are ascending, so
        // the visit order — and the film accumulation order — is exactly the old scan's.
        if (!scene.dielSphereIdx.empty()) {
            for (int i = 0; i < nUp; ++i) {
                SpecVtx vi = vt; vi.weight = w[i]; vi.lambda = lam[i];
                for (int si : scene.dielSphereIdx) {
                    Sphere moved;
                    const Sphere& S = scene.sphereNow(scene.spheres[(size_t)si], moved);
                    const Material& gm = scene.mats[S.matId];
                    const double ng = gm.ior(lam[i]);
                    for (int c = 0; c < nCam; ++c)
                        if (cams[c].cam && cams[c].film && !cams[c].cam->hasLens())
                            connectSpecularSphere(scene, *cams[c].cam, *cams[c].film, S, gm, ng,
                                                  p, vi, lam[i], beta[i], rng);
                }
            }
        }
        // Mirrors. Achromatic geometry, so one solve serves the whole hero bundle.
        for (int si : scene.mirrorSphereIdx) {
            Sphere moved;
            const Sphere& S = scene.sphereNow(scene.spheres[(size_t)si], moved);
            for (int c = 0; c < nCam; ++c)
                if (cams[c].cam && cams[c].film && !cams[c].cam->hasLens())
                    connectSpecularSphereMirror(scene, *cams[c].cam, *cams[c].film, S,
                                                p, vt, lam, beta, w, nUp, rng);
        }
        // A path with a time takes a mixed plane (faces moving differently, or deforming) by
        // its parts, each moving as one.
        for (const auto* list : {&scene.mirrorPlanes, &scene.mirrorPlaneParts}) {
            if (list == &scene.mirrorPlaneParts && g_motionU < 0.0) break;
            for (const Scene::MirrorPlane& mp0 : *list) {
                Scene::MirrorPlane moved;
                const Scene::MirrorPlane* mp = scene.mirrorPlaneNow(mp0, moved);
                if (!mp) continue;                          // a mixed plane: its parts serve
                for (int c = 0; c < nCam; ++c)
                    if (cams[c].cam && cams[c].film && !cams[c].cam->hasLens())
                        connectSpecularPlane(scene, *cams[c].cam, *cams[c].film, *mp,
                                             p, vt, lam, beta, w, nUp, rng);
            }
        }
    }
    void camSpecularSplatAllVtx(const Scene& scene, const CamTarget* cams, int nCam,
                                const Vec3& p, const SpecVtx& vt, double lambda,
                                double beta, Pcg32& rng) const {
        const double w = vt.weight;
        camSpecularSplatAllVtxN(scene, cams, nCam, p, vt, &lambda, &beta, &w, 1, rng);
    }
    // Surface vertex: refract the Lambertian reflection of p through every glass sphere,
    // and reflect it in every mirror.
    void camSpecularSplatAll(const Scene& scene, const CamTarget* cams, int nCam,
                             const Vec3& p, const Vec3& n, double lambda, double beta,
                             double rho, Pcg32& rng, const Vec3* plane = nullptr) const {
        SpecVtx vt; vt.volume = false; vt.np = n; vt.weight = rho;
        if (plane) { vt.gated = true; vt.plane = *plane; }                 // see SpecVtx
        camSpecularSplatAllVtx(scene, cams, nCam, p, vt, lambda, beta, rng);
    }
    // Glossy vertex (FWD-GLOSSY-INDIRECT): the rough lobe seen in a mirror or through a glass
    // sphere, as camSplatAll's `gs` splats it directly. Its value toward the first specular point
    // replaces the Lambertian rho/pi, at each of the `nUp` wavelengths (an iridescent lobe's colour
    // turns with the direction, so it is evaluated per wavelength, not shared).
    void camSpecularSplatAllGlossy(const Scene& scene, const CamTarget* cams, int nCam,
                                   const Vec3& p, const Vec3& n, const double* lam,
                                   const double* beta, int nUp, const GlossySplat& gs,
                                   Pcg32& rng) const {
        SpecVtx vt; vt.volume = false; vt.np = n; vt.weight = 0.0; vt.glossy = &gs;
        const double w[hero::kHeroMax] = {};   // a glossy vertex's weight is in its lobe
        camSpecularSplatAllVtxN(scene, cams, nCam, p, vt, lam, beta, w, nUp, rng);
    }
    // Hero variant. The sphere REFRACTION is dispersive (glass IOR varies per λ), so it
    // cannot share one geometry — each wavelength traces its own refracted image, which
    // is exactly what makes the glass-sphere caustic-image chromatically dispersed. A
    // mirror is achromatic, so its geometry IS shared and only the reflectance varies.
    // Draws no RNG (mode B), so the bundle leaves the stream untouched.
    void camSpecularSplatAllHero(const Scene& scene, const CamTarget* cams, int nCam,
                                 const Vec3& p, const Vec3& n, const double* lam,
                                 const double* beta, const double* rho, int nUp,
                                 Pcg32& rng, const Vec3* plane = nullptr) const {
        SpecVtx vt; vt.volume = false; vt.np = n; vt.weight = rho[0];
        if (plane) { vt.gated = true; vt.plane = *plane; }                 // see SpecVtx
        camSpecularSplatAllVtxN(scene, cams, nCam, p, vt, lam, beta, rho, nUp, rng);
    }
    // Volume vertex: refract the fog in-scatter at p through every glass sphere, so the
    // glowing haze itself bends through the glass the camera flies through — and shows
    // in the mirrors.
    void camSpecularSplatVolumeAll(const Scene& scene, const Medium& med, const CamTarget* cams,
                                   int nCam, const Vec3& p, const Vec3& wIn, double lambda,
                                   double beta, Pcg32& rng) const {
        SpecVtx vt; vt.volume = true; vt.wIn = wIn; vt.g = med.g; vt.weight = med.albedo(lambda);
        camSpecularSplatAllVtx(scene, cams, nCam, p, vt, lambda, beta, rng);
    }

    // Volume (fog) analogue of camSplatAll. `med` is the medium that scattered the
    // photon here (its phase/albedo drive the in-scatter term); transmittance still
    // accounts for all media (product) inside the volume connect functions.
    void camSplatVolumeAll(const Scene& scene, const Medium& med, const CamTarget* cams,
                           int nCam, const Vec3& p, const Vec3& wIn, double lambda,
                           double beta, Pcg32& rng) const {
        for (int c = 0; c < nCam; ++c)
            if (cams[c].cam && cams[c].film) {
                if (lensCam(*cams[c].cam)) connectLensVolume(scene, med, *cams[c].cam, *cams[c].film, p, wIn, lambda, beta, rng);
                else          connectVolume(scene, med, *cams[c].cam, *cams[c].film, p, wIn, lambda, beta, rng);
            }
    }

    // Specular / wavelength-switching material interaction for the forward photon tracer:
    // the nine families that are NOT Diffuse / DiffuseTransmit (Dielectric, ThinFilm,
    // Multilayer, Mirror, Grating, HalfMirror, Filter, Glossy, Fluorescent). Mutates
    // `ray`, `beta`, `lambda` (Fluorescent Stokes shift) and the dielectric priority
    // `stk`; on absorption/termination it books the loss into `e` and returns false.
    // Returns true to keep bouncing. Shared verbatim by the scalar tracePhoton and, after
    // de-hero, by tracePhotonHero — so there is a single source of truth for these lobes.
    bool interactPhotonSpecular(const Scene& scene, const CamTarget* cams, int nCam,
                                const Material& m, const Hit& h, Ray& ray, double& beta,
                                double& lambda, MediumStack& stk, Pcg32& rng,
                                EnergyReport& e, bool firstHit = false) const {
        switch (m.type) {
            case MatType::Dielectric:
            case MatType::ThinFilm:
            case MatType::Multilayer: {
                // A dielectric body's surface, plain or coated (interfaceStep): nested bodies by
                // priority; a coating reflects with its interference reflectance (structural
                // colour), and an absorbing substrate or stack absorbs what it transmits. A
                // photon carries flux: a refraction does not rescale it. Lossless on survival
                // (absorption inside is applied per segment).
                const InterfaceStep st = interfaceStep(scene, m, h, ray.d, lambda, rng, stk,
                                                       Transport::Flux);
                if (st.absorbed) { e.absorbed += beta; return false; }
                ray = st.ray;
                return true;
            }
            case MatType::Mirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                // Russian roulette: absorb with prob (1-r), else reflect with
                // beta unchanged. Unbiased and caps path length naturally.
                if (rng.uniform() >= r) { e.absorbed += beta; return false; }
                Vec3 o = reflect(ray.d, h.n);
                if (!continueAlong(h, o, Lobe::Reflect, ray)) { e.absorbed += beta; return false; }   // across the surface
                return true;
            }
            case MatType::Grating: {
                // Diffraction grating: RR on the overall reflectivity, then
                // deflect into one stochastically-chosen order (exact grating
                // equation). Specular per order -> no camera connect (mode P /
                // the caustic on diffuse walls makes it visible).
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() >= r) { e.absorbed += beta; return false; }
                bool absorbedG;
                Ray nr = gratingDiffract(m, h, ray.d, lambda, rng, absorbedG);
                if (absorbedG) { e.absorbed += beta; return false; }
                ray = nr;
                return true;                     // beta unchanged on the chosen order
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda)); // reflect probability
                if (rng.uniform() < r) {
                    Vec3 o = reflect(ray.d, h.n);
                    if (!continueAlong(h, o, Lobe::Reflect, ray)) { e.absorbed += beta; return false; }
                } else {
                    ray = Ray{h.p + ray.d * 1e-6, ray.d}; // transmit straight
                }
                return true;                     // lossless split
            }
            case MatType::Filter: {
                // Colored gel / Wratten filter: a thin non-scattering absorber.
                // The photon passes straight through (direction unchanged) and
                // survives with probability T(lambda), else is absorbed. Russian
                // roulette on the transmittance keeps beta unchanged and unbiased —
                // the wavelength-dependent survival IS the colored transmission.
                // Specular straight-through, so no camera connect (like clear glass):
                // you see the filter's effect on whatever lies behind it.
                double t = clamp01(transmitSlot(scene, m, h, lambda));
                if (rng.uniform() >= t) { e.absorbed += beta; return false; }
                ray = Ray{h.p + ray.d * 1e-6, ray.d}; // transmit straight, unchanged
                return true;
            }
            case MatType::Glossy: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                // Russian roulette on reflectance (see Mirror).
                if (rng.uniform() >= r) { e.absorbed += beta; return false; }
                const GlossyDraw gd = drawGlossy(scene, m, h, ray.d, h.n, ray.d * -1.0, rng);
                const Vec3& o = gd.o;
                if (dot(o, h.n) <= 0) { e.absorbed += beta; return false; } // below surface
                // A photon carries the lobe's adjoint (glossyAdjointRatio) and, where the shading
                // normal tilts, Veach's shading-normal half (shadingAdjointCorr, as bdpt.h's light
                // walk takes at every non-delta vertex; 1 where ns == ng). The ledger takes what it
                // gains or loses as negative or positive absorption, so it still balances.
                const double adj = glossyPhotonAdjoint(h, ray.d * -1.0, o);
                if (!(adj > 0.0)) { e.absorbed += beta; return false; }
                e.absorbed += beta * (1.0 - adj);
                beta *= adj;
                // An iridescent lobe's interference reflectance, known only now that the pair
                // is: a second roulette, beta unchanged.
                if (gd.irid && rng.uniform() >= gd.R(lambda)) { e.absorbed += beta; return false; }
                if (!continueAlong(h, o, Lobe::Reflect, ray)) { e.absorbed += beta; return false; }
                return true;
            }
            case MatType::Fluorescent: {
                double rho, aEff; fluoroWeights(m, lambda, rho, aEff);
                Vec3 ngo = orientedGeoN(h);
                Vec3 wi = Vec3{-ray.d.x, -ray.d.y, -ray.d.z};   // toward the previous (light-side) vertex
                // A Lambertian lobe, as Diffuse's (lambertArrival).
                double sEm;
                if (!lambertArrival(h, ngo, wi, firstHit, sEm)) { e.absorbed += beta; return false; }
                e.absorbed += beta * (1.0 - sEm);
                beta *= sEm;
                // Model-B connections: the elastic channel splats at the
                // incoming lambda; the fluorescent channel samples one
                // lambda' ~ M and splats the glow (albedo aEff*Q) with the
                // camera-response evaluated at lambda' (Stokes-shifted colour).
                if (nCam > 0 && !forwardCatch) {
                    camSplatAll(scene, cams, nCam, h.p, h.n, ngo, wi, lambda, beta, rho, rng);
                    if (aEff > 0.0 && m.fluoYield > 0.0 && m.fluoEmitSampler.integral > 0.0) {
                        double pf; double lp = m.fluoEmitSampler.sample(rng, pf);   // drawn once, camera-independent
                        camSplatAll(scene, cams, nCam, h.p, h.n, ngo, wi, lp, beta, aEff * m.fluoYield, rng);
                    }
                }
                FluoroResult fr = fluoroInteract(m, lambda, rng);
                if (fr.event == FluoroEvent::Absorb) { e.absorbed += beta; return false; }
                lambda = fr.lambdaOut;              // Stokes-shifted on Reemit
                Vec3 wo;
                beta *= diffuseContinuation(h, ngo, wi, rng, wo);
                if (!continueAlong(h, wo, Lobe::Reflect, ray)) { e.absorbed += beta; return false; }
                return true;                        // beta otherwise unchanged (see above)
            }
            case MatType::Hair: {
                // Fiber BCSDF (Marschner/Chiang; src/hair.h, bridged by hair_shade.h).
                //
                // It sits in this function rather than beside Diffuse because, like the
                // dielectric family, it is wavelength-COUPLED: the absorption sigma_a is a
                // per-λ quantity, and an authored `reflect` colour is inverted into one
                // per-λ, so a hero packet cannot share one fiber interaction across C
                // wavelengths — tracePhotonHero de-heroes onto this path instead.
                //
                // The interaction is a connect-then-scatter, like Fluorescent: the lobes are
                // narrow but finite, so the vertex IS visible to a mode-B pinhole and must
                // splat before it scatters.
                const Vec3 wPrev{-ray.d.x, -ray.d.y, -ray.d.z};   // toward the light-side vertex
                const HairShade hs = hairShadeAt(scene, m, h, lambda, wPrev);
                if (nCam > 0 && !forwardCatch)
                    camSplatAllHair(scene, cams, nCam, h.p, h.n, orientedGeoN(h), wPrev,
                                    lambda, beta, hs, rng);

                double pdf = 0.0, fv = 0.0;
                const Vec3 wl = hair::sample(hs.b, hs.woLocal, rng.uniform(), rng.uniform(),
                                             rng.uniform(), rng.uniform(), pdf, fv);
                if (!(pdf > 0.0) || !(fv > 0.0)) { e.absorbed += beta; return false; }
                // The BCSDF is built so this ratio is EXACTLY T = sum_p A_p <= 1 (the total
                // Fresnel/Beer attenuation over the lobes) — the per-lobe pdf weights are
                // A_p/T and everything else cancels. So it is a deterministic survival
                // probability, and Russian-rouletting on it leaves beta untouched: the same
                // trick Mirror plays with its reflectance, but here the number is the
                // physics rather than an authored albedo.
                const double cosLong = hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
                const double T = clamp01(fv * cosLong / pdf);
                if (rng.uniform() >= T) { e.absorbed += beta; return false; }
                const Vec3 wo = hair::toWorld(hs.fr, wl);
                // TT and TRT leave through the FAR side of a real solid strand, so step
                // clear of the tube's own body (hair_shade.h); on the near side this is the
                // ordinary 1e-6 offset.
                ray = Ray{h.p + wo * 1e-6, wo, hairExitOffset(hs, h.n, wo)};
                return true;                        // beta unchanged (RR carried the weight)
            }
            default: return true;                   // unreachable (diffuse handled by callers)
        }
    }

    // Back-compat single-camera entry point (sensorFilm XOR cam+camFilm, as before).
    void tracePhoton(const Scene& scene, const Camera* cam, Film* sensorFilm,
                     Film* camFilm, Pcg32& rng, EnergyReport& e) const {
        CamTarget t{cam, camFilm};
        int nCam = (cam && camFilm) ? 1 : 0;
        tracePhoton(scene, &t, nCam, sensorFilm, rng, e);
    }

    // Trace a single photon, splatting each vertex to every camera in `cams` (nCam may
    // be 0 for the legacy contact-sensor path, which uses sensorFilm instead). Models A
    // and C require nCam<=1 (they draw RNG / consume the photon per camera); the
    // multi-camera shared pass is model B only. The caller supplies per-thread films.
    void tracePhoton(const Scene& scene, const CamTarget* cams, int nCam,
                     Film* sensorFilm, Pcg32& rng, EnergyReport& e) const {
        if (useHero) { tracePhotonHero(scene, cams, nCam, sensorFilm, rng, e); return; }
        // --- Emission ---
        // Power-weighted emitter selection: photon selects emitter k with prob
        // power_k/totalPower and carries beta = totalPower, so E[beta over the
        // selection] reproduces each emitter's true power (unbiased). For a single
        // emitter selectEmitter() draws no randomness, keeping the RNG stream (and
        // thus the image) bit-identical to the pre-multi-light engine.
        // A photon is born on a surface/environment EMITTER or inside a volumetric
        // blackbody EMITTER ("fire"). grandTotal folds both power pools; the class is
        // chosen with probability proportional to power (P(fire)=totalEmissionPower/
        // grandTotal) and the photon carries beta=grandTotal so E[beta] reproduces each
        // source's true power (unbiased carry-total scheme). When there are no emissive
        // volumes the `&&` short-circuits WITHOUT drawing an RNG, so ordinary scenes stay
        // bit-for-bit identical to the pre-fire engine.
        const double grandTotal = scene.totalPower + scene.totalEmissionPower;
        if (grandTotal <= 0.0) return;
        Vec3 origin, dir;
        double lambda, beta;
        // SPECTRAL BEAM BUNDLE (photonbeams.h). The extra stratified wavelengths this photon's
        // beam deposits will carry alongside `lambda`, and how many are still live. Filled at
        // birth for a plain SPD-sampled emitter, and dropped to zero — collapsing the beam back
        // to the classic monochromatic record — the instant the path does something the
        // wavelengths would not SHARE A CHORD through. Untouched (and therefore free) when
        // beamSpecC == 1.
        //
        // `specW[i]` is member i's running spectral weight T(specLam[i]) / T(lambda), i.e.
        // EXACTLY the quantity `foldT[k]` below carries, on a different grid: the bundle's own
        // wavelengths instead of the emitter's quadrature bins. Before 0.257.0 there was no such
        // array and the bundle had to be retired every transport iteration, because a member
        // whose weight had diverged could not be represented; every site that folds a spectral
        // factor into `foldT` now folds the same factor into `specW` in the same breath, so the
        // bundle lives exactly as long as the achromatic-path claim does.
        double specLam[kBeamSecMax];
        double specW[kBeamSecMax];
        int specSec = 0;
        // ACHROMATIC-PATH STATE (photonbeams.h, ACHROMATIC-PATH BEAMS; device twin:
        // DBeamSpec::achro/cie). The stronger, longer-lived claim beside the bundle: that
        // NOTHING on this path has depended on lambda, so the beam may be folded at the
        // emitter's mean CIE and carry no chromatic noise at all. Unlike `specSec` it is not
        // retired every step — only by an event that is itself wavelength-dependent, which a
        // scatter in an achromatic medium is not.
        Vec3 achroCie{0, 0, 0};
        bool achroPath = false;
        // SPECTRAL FOLD (scene.h: kFoldBins, Emitter::foldCie/foldLam/foldN).
        //
        // The plain achromatic fold above is all-or-nothing: the instant the path touches
        // ANYTHING wavelength-dependent the claim dies and the beam reverts to a single noisy
        // CIE(lambda_hero) sample — which is the streak-painter behind the "coloured bars".
        // Measurement on gallery_rain: 87.8% of unfolded deposits had lost the fold to a
        // SURFACE event (a diffuse albedo), 12.2% to a chromatic medium, 0% to anything else.
        // So the surface case is the whole artifact, and it does not have to be fatal.
        //
        // Generalise the fold from E_lam[CIE(lam)] to E_lam[CIE(lam) * T(lam)], where
        //
        //     T(lam) = prod_j f_j(lam) / prod_j f_j(lambda_hero)
        //
        // is the running ratio of the path's spectral factors at `lam` versus at the hero
        // wavelength the photon is actually being traced at. `foldT[k]` holds T evaluated at
        // bin k's representative wavelength `foldEm->foldLam[k]`.
        //
        // WHY THAT IS THE RIGHT QUANTITY, and why it is unbiased. A deposited beam's
        // contribution is CIE(lambda_h) * beta * G, with beta = P_emit * prod f_j(lambda_h) /
        // prod p_j. Direction pdfs are wavelength-independent on an achromatic path, so the
        // p_j factor out and the only lambda-dependence left in the whole estimator is
        // CIE * prod f_j. Replacing CIE(lambda_h) by E_lam[CIE(lam) T(lam)] therefore gives
        // E_lam[CIE(lam) prod f_j(lam)] * G / prod p_j, which is exactly the spectral integral
        // the monochromatic estimator only reaches in expectation. T(lambda_h) == 1 by
        // construction, so a path with no spectral factors folds to sum_k foldCie[k] ==
        // cieMean and reproduces the old achromatic fold BIT-FOR-BIT.
        //
        // The concrete factor at a Lambertian vertex is the albedo, which this tracer applies
        // as an ANALOG Russian roulette: survive with probability rho(lambda_h), beta
        // unchanged. Conditional on survival the fold must therefore carry
        // T_k *= rho(lam_k)/rho(lambda_h), whose expectation over the roulette is
        // rho(lambda_h) * (rho(lam_k)/rho(lambda_h)) = rho(lam_k) — the value the spectral
        // integral wants, with the survival lottery cancelled out.
        //
        // `foldEm` is the emitter whose quadrature table foldT indexes; foldT is left
        // UNINITIALISED until the first spectral factor actually arrives (foldChroma), so a
        // photon that never hits a surface pays nothing for any of this.
        const Emitter* foldEm = nullptr;
        int srcEmIdx = -1;   // chord provenance (-sunnee): the emitter this photon is born on; -1 = a volume (fire) birth
        double foldT[kFoldBins];
        bool   foldChroma = false;
        const bool volumeBirth = !scene.emissiveVolumes.empty() &&
                                 (rng.uniform() * grandTotal < scene.totalEmissionPower);
        if (volumeBirth) {
            // --- Volumetric blackbody birth (fire) ---
            // Select an emissive volume proportional to its power.
            const int vi = scene.pickFire(rng.uniform());
            const Scene::EmissiveVolume& ev = scene.emissiveVolumes[(size_t)vi];
            const Medium& fm = scene.media[ev.mediumIndex];
            // Uniform position in the grid AABB; wavelength importance-sampled from a
            // representative blackbody (Planck at emitKelvin) via ev.lamSampler.
            // beta = grandTotal·κ_e(x,λ)/(meanKe·Δλ·p(λ)) reproduces the emission
            // line-integral when splatted with the isotropic 1/(4π)/(dist²·Ω) direct
            // term (derived). With p(λ) uniform (=1/Δλ) this collapses to the plain
            // grandTotal·κ_e/meanKe; matching p(λ) to the Planck shape makes β nearly
            // constant across the band, killing the spectral colour speckle.
            origin = Vec3{ ev.bmin.x + (ev.bmax.x - ev.bmin.x) * rng.uniform(),
                           ev.bmin.y + (ev.bmax.y - ev.bmin.y) * rng.uniform(),
                           ev.bmin.z + (ev.bmax.z - ev.bmin.z) * rng.uniform() };
            double pdfLam = 0.0;
            lambda = ev.lamSampler.sample(rng, pdfLam);
            double ke = fm.emissionAt(origin, lambda);
            const double dLamE = LAMBDA_MAX - LAMBDA_MIN;
            beta = (ev.meanKe > 0.0 && pdfLam > 0.0)
                 ? grandTotal * ke / (ev.meanKe * dLamE * pdfLam) : 0.0;
            e.emitted += beta;
            if (beta <= 0.0) return;             // cold voxel: nothing to emit or transport
            // Isotropic emission direction.
            {   const double u1 = rng.uniform(), u2 = rng.uniform();
                dir = uniformSphere(u1, u2); }
            // Aimed caustic emission (causticaim.h). A fire birth point is already fixed, so
            // it is the DIRECTION that gets aimed; p_u for an isotropic birth is 1/(4pi),
            // which is the `em == nullptr` case. No-op (and no RNG draw) unless an aim map
            // is bound.
            {
                Vec3 emitNv = dir; double spotWv = 1.0;   // unread when em == nullptr
                if (!applyCausticAim(scene, nullptr, origin, dir, emitNv, spotWv, rng)) return;
            }
            // Direct-visibility emission splat (the flame seen directly by the camera).
            if (nCam > 0 && !forwardCatch)
                camSplatEmissionAll(scene, cams, nCam, origin, lambda, beta, rng);
            // Fall through to the shared transport loop so the emitted light also
            // illuminates the rest of the scene (self-scatter in the soot, walls, etc.).
        } else {
        // Power-weighted emitter selection: photon selects emitter k with prob
        // power_k/totalPower and carries beta = totalPower, so E[beta over the
        // selection] reproduces each emitter's true power (unbiased). For a single
        // emitter selectEmitter() draws no randomness, keeping the RNG stream (and
        // thus the image) bit-identical to the pre-multi-light engine.
        if (scene.emitters.empty()) return;
        int ei = scene.selectEmitter(rng);
        const Emitter& em = scene.emitters[ei];
        srcEmIdx = ei;
        double u1 = rng.uniform(), u2 = rng.uniform();
        Vec3 emitN;
        double spotW = 1.0;                      // the power's reweight: a spot's p_e/p_u, the sky's disc, a scaled light's area
        double envPdfW = 0.0;                    // env: solid-angle pdf of the sampled dir
        double emitPatW = 1.0;                   // `emit pattern:` factor at the sampled point
        if (em.shape == EmitterShape::Spot) {
            // Point spot: sample a direction uniformly in the outer cone, then
            // reweight beta by falloff*(Omega_outer/Omega_eff) so the emitted
            // distribution matches the smoothstep intensity profile (analog MC).
            origin = scene.lightPointNow(em, em.origin);
            double ct = em.spotCosOuter + u1 * (1.0 - em.spotCosOuter);
            double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
            double phi = 2.0 * PI * u2;
            Vec3 t, b; onb(em.beamDir, t, b);
            dir = scene.lightDirNow(em, t * (st * std::cos(phi)) + b * (st * std::sin(phi)) +
                                        em.beamDir * ct);
            emitN = scene.lightDirNow(em, em.beamDir);
            double omegaOuter = 2.0 * PI * (1.0 - em.spotCosOuter);
            spotW = spotFalloff(ct, em.spotCosInner, em.spotCosOuter) * omegaOuter / em.spotOmega;
        } else if (em.shape == EmitterShape::Env) {
            // Infinite environment. Sample the incoming photon direction `dir` — for
            // a constant env uniformly on the sphere (pdf 1/4pi); for an image env
            // importance-sampled from the map's luminance CDF (pdf envPdfW) — then its
            // entry point on a disk of radius R perpendicular to `dir`, centered on
            // the scene and pushed upstream so the photon starts just outside the
            // bounding sphere (disk pdf 1/(pi R^2)). For a constant env the joint pdf
            // 1/(4pi^2 R^2) = 1/envGeom makes beta = emitIntegral*envGeom exactly
            // analog (no reweight); an image env reweights beta below by
            // L(dir,lambda)/(4pi*envPdfW*avgSpd(lambda)) — which is 1 in the constant
            // case, keeping constant-env scenes bit-identical.
            if (scene.envMap) {
                // The sampler draws the SKY direction the light comes FROM; the photon travels
                // the other way. Until 0.370.0 it flew TOWARD the texel it was drawn from, so
                // every forward mode (and M's / S's photon passes) lit the scene with the
                // environment point-mirrored -- a map bright toward +x lit a sphere's -x side --
                // while the backdrop, a camera-ray lookup, showed it the right way round.
                dir = -scene.envMap->sample(u1, u2, envPdfW);
            } else {
                dir = uniformSphere(u1, u2);
            }
            // A photon with a time in a moving scene enters through a disc wide enough for the
            // movers' sweep, its power scaled by the disc's area (Scene::lightRadius).
            const double R = scene.lightRadius();
            spotW = scene.lightScale() * scene.lightScale();
            Vec3 t, b; onb(dir, t, b);
            double rd = R * std::sqrt(rng.uniform());
            double pd = 2.0 * PI * rng.uniform();
            Vec3 disk = t * (rd * std::cos(pd)) + b * (rd * std::sin(pd));
            origin = scene.sceneCenter - dir * R + disk;
            emitN = dir;
        } else if (em.shape == EmitterShape::Sun) {
            // Distant directional sun. Sample the travel direction inside the solar
            // cone (pdf 1/Omega), then the entry point on a disk of radius R
            // perpendicular to it (pdf 1/(pi R^2)) — the same upstream-disk trick the
            // env uses, but aimed instead of isotropic, so EVERY photon crosses the
            // scene rather than one in ~10^5. The joint pdf 1/(Omega*pi*R^2) is exactly
            // 1/envGeom, so beta = emitIntegral*envGeom is analog with no reweight.
            dir = em.sampleCone(em.beamDir, u1, u2);
            const double R = scene.lightRadius();             // as the env's, above
            spotW = scene.lightScale() * scene.lightScale();
            Vec3 t, b; onb(dir, t, b);
            double rd = R * std::sqrt(rng.uniform());
            double pd = 2.0 * PI * rng.uniform();
            origin = scene.sceneCenter - dir * R
                   + t * (rd * std::cos(pd)) + b * (rd * std::sin(pd));
            emitN = dir;
        } else {
            // quad: constant normal; sphere: surface point. emitterSamplePoint also
            // returns this point's `emit pattern:` factor — 1.0 (and a bit-identical
            // call) when the emitter carries no pattern.
            // A light its keys scale, or a mesh light that deforms, emits from its area at the
            // photon's time: the power it was chosen by, the instant's, grows with it (1 unchanged).
            emitPatW = emitterSamplePoint(scene, em, u1, u2, origin, emitN, &spotW);
            if (em.collimated) dir = scene.lightDirNow(em, em.beamDir);
            else if (em.normalTilted) {             // an aimed quad: intensity-proportional
                double pdfAim = 0.0;
                dir = scene.sampleAimDirNow(em, rng, pdfAim);
                if (!(pdfAim > 0.0)) return;
            } else dir = cosineHemisphere(emitN, rng);
        }
        // Aimed caustic emission (causticaim.h). Placed here, after the shape branch has
        // produced an ordinary sample and BEFORE `beta *= spotW`, because the aimed pass
        // resamples the direction and therefore recomputes spotW. In the main pass this
        // only measures the sample (no RNG draw, so every existing render stays
        // bit-for-bit) and stores the caustic MIS weight.
        if (!applyCausticAim(scene, &em, origin, dir, emitN, spotW, rng)) return;
        if ((em.shape == EmitterShape::Env || em.shape == EmitterShape::Sun) &&
            !scene.skyEntryPastHaze(origin, dir))   // a blur's wider disc
            return;
        double pdfL = 0.0;
        // Drawn through sampleAt rather than sample() so the SAME uniform variate can seed the
        // stratified secondaries below. `sample()` is literally `sampleAt(rng.uniform(), pdf)`,
        // so this consumes the identical rng draw and returns the identical wavelength — every
        // existing render stays bit-for-bit.
        const double uLam = rng.uniform();
        lambda = em.spd.sampleAt(uLam, pdfL);
        if (pdfL <= 0) return;
        // Single emitter (no fire): beta = its own power (== old lightEmitIntegral*
        // area*PI). Multiple emitters: beta = totalPower. When fire volumes coexist the
        // emitter class carries grandTotal (carry-total split, see above).
        beta = !scene.emissiveVolumes.empty() ? grandTotal
             : ((scene.emitters.size() == 1) ? em.power : scene.totalPower);
        beta *= spotW;   // 1 for a still, unscaled area light (a spot's reweight, a light's growth)
        // Image env: replace the flat power with the directional estimator. The base
        // beta carries the mean env power; multiply by L(dir,lambda)/(4pi*pdfW*mean)
        // so the photon represents the radiance actually arriving from `dir`. (No-op
        // for a constant env, so those scenes stay bit-identical.)
        if (em.shape == EmitterShape::Env && scene.envMap) {
            double denom = 4.0 * PI * envPdfW * em.spdFn(lambda);
            beta = (denom > 0.0) ? beta * (scene.envMap->radiance(-dir, lambda) / denom) : 0.0;   // -dir: the sky it came from
        }
        // An emission pattern is a pure post-multiplier on the photon's carried power:
        // the emitter is still SELECTED by its unpatterned power and the point still
        // drawn uniformly over its area, so no pdf anywhere changes and the estimator
        // stays unbiased. The cost is variance — a mostly-dark pattern spends most of
        // its photons on near-zero beta. `emitted` is credited the patterned value so
        // the energy report matches what actually leaves the surface.
        if (emitPatW != 1.0) beta *= emitPatW;
        e.emitted += beta;

        // --- Spectral bundle for the beam deposit (photonbeams.h) --------------------------
        // Hero policy (1): the C wavelengths come from ONE uniform variate, secondary i taking
        // `u + i/C` wrapped into [0,1) through the same emission CDF, so the bundle is
        // stratified over the emitter's own spectrum rather than clumped.
        //
        // Every wavelength carries the SAME power, which is why no per-wavelength weight is
        // stored anywhere. The emission sampler's pdf is p(lam) = spd(lam)/integral, and the
        // photon's beta is the emitter's total power — i.e. spd(lam)/p(lam) == integral, a
        // constant. So the C-wavelength estimate is simply beta/C at each of the C
        // wavelengths, and the beam record needs only the wavelengths themselves.
        //
        // An IMAGE environment is excluded because its beta carries the directional factor
        // L(dir,lam)/(4*pi*pdfW*spd(lam)), which is genuinely per-wavelength; so is a
        // volumetric blackbody birth, whose beta carries kappa_e(x,lam). Both keep C == 1
        // rather than being reweighted, which costs those scenes nothing they had before.
        // ONE PRECONDITION FOR BOTH (0.257.0). The bundle and the achromatic-path fold rest on
        // the same two facts — that `beta` is wavelength-independent, and that this emitter has
        // a quadrature table to decide a fold's variance against — so they are decided from one
        // predicate rather than two that drifted apart. An IMAGE environment fails it because
        // its beta carries L(dir,lam)/(4*pi*pdfW*spd(lam)); a volumetric blackbody birth never
        // reaches here at all (it takes the other branch), and its beta carries kappa_e(x,lam).
        // `foldN > 0` and a non-black cieMean are what make `foldWorthIt` answerable: without a
        // table the bundle could still be BORN but could never be REWEIGHTED, so it would have
        // to die at the first spectral factor — which is the pre-0.257.0 behaviour this change
        // exists to remove. Refusing it up front keeps one rule instead of two.
        const bool specBirthOK = beamDeposit && em.foldN > 0 &&
                                 !(em.shape == EmitterShape::Env && scene.envMap) &&
                                 (em.cieMean.x > 0.0 || em.cieMean.y > 0.0 || em.cieMean.z > 0.0);
        // The emitter whose quadrature table `foldT` indexes and whose bins `foldWorthIt`
        // reads. Set for EITHER consumer, not just the achromatic one: `-beamachro off
        // -beamspec 4` is a supported combination and its bundle needs the same table.
        if (specBirthOK) foldEm = &em;
        if (beamSpecC > 1 && specBirthOK) {
            const int C = (beamSpecC > kBeamSpecMax) ? kBeamSpecMax : beamSpecC;
            for (int i = 1; i < C; ++i) {
                double uu = uLam + (double)i / (double)C;
                if (uu >= 1.0) uu -= 1.0;
                double pI = 0.0;
                const double lI = em.spd.sampleAt(uu, pI);
                // A zero-density secondary would have to be given weight 0 while the survivors
                // kept 1/C, so drop the WHOLE bundle instead of renormalising over the
                // survivors, which would over-count them. (`wS` could express the zero, but the
                // member would then contribute nothing while still consuming one of the C
                // shares of the chord's power — a silent energy loss, not a reweighting.
                // Inverting a CDF at a uniform variate cannot land in a zero-mass bin, so this
                // is a guard, not a path.)
                if (!(pI > 0.0)) { specSec = 0; break; }
                specW[specSec] = 1.0;   // relative to the hero, which starts at parity
                specLam[specSec++] = lI;
            }
        }
        // ACHROMATIC-PATH STATE at birth. Same precondition, minus the `-beamspec` count —
        // this fold stores no extra wavelengths, so it applies at `-beamspec 1`.
        if (beamAchroOK && specBirthOK) {
            achroCie = em.cieMean;
            achroPath = true;
        }
        g_foldKill = achroPath ? FK_None : FK_NeverBorn;   // fold diagnostics

        // Direct light -> camera: makes the source itself visible. The Lambertian
        // emitter term is 1/pi, i.e. connect() with rho=1 using the light normal.
        // (Skipped in forward-catch mode; there the aperture test below handles it.)
        // A spot is a point light with no projected area, so it has no such direct
        // term (its cone illuminates surfaces, which then connect to the camera).
        // A Sun is excluded for the same reason as the env: it has no finite surface to
        // connect to, and the disc's direct view is composited by addEnvBackground.
        if (nCam > 0 && !forwardCatch &&
            em.shape != EmitterShape::Spot && em.shape != EmitterShape::Env &&
            em.shape != EmitterShape::Sun) {
            // An aimed quad: intensity Le*aimCos per unit area, so the splat's Lambertian
            // rho/PI * cos(emitN) becomes aimCos / aimNorm -- rho = PI/aimNorm about the aim --
            // and nothing reaches a camera or a mirror behind the face.
            const double rhoE = em.normalTilted ? PI / em.aimNorm : 1.0;
            const Vec3 face = em.normalTilted ? scene.aimFaceNow(em) : Vec3{0, 0, 0};
            const Vec3* plane = em.normalTilted ? &face : nullptr;
            camSplatAll(scene, cams, nCam, origin, emitN, emitN, emitN, lambda, beta, rhoE, rng, plane);
            camSpecularSplatAll(scene, cams, nCam, origin, emitN, lambda, beta, rhoE, rng, plane);
        }
        }   // end emitter-birth branch

        Ray ray{origin + dir * 1e-6, dir};
        // Nested-dielectric medium stack: the solids the photon is currently inside.
        // The current medium (for Beer-Lambert absorption and the exterior IOR at the
        // next interface) is the highest-priority entry (Schmidt & Budge 2002). Replaces
        // the old single `interior` pointer; behaves identically for a lone dielectric.
        MediumStack stk;
        auto curAbsorb = [&](double lam) -> double {           // sigma_a of the current medium
            int mi = stk.topMat();
            return (mi >= 0) ? scene.mats[mi].absorb(lam) : 0.0;
        };

        // --- SPECTRAL FOLD helpers (see the foldT declaration above) ------------------------
        //
        // The verdict itself is `foldWorthIt(*foldEm, fk, K)`, a free function at the top of
        // this header — bdpt.h's randomWalk deposits into the same beam bank and must reach
        // exactly the same decision, so there is one definition rather than one per tracer.
        //
        // Multiply the per-bin ratios f(lam_k)/f(lambda_h) into the running fold. The caller
        // has already established that every fk[k] > 0 (via foldWorthIt) and that fHero > 0.
        //
        // `fk` is TWO grids in one array: entries [0, K) are the emitter's quadrature bins and
        // feed `foldT`, entries [K, K+S) are the live bundle's own wavelengths and feed `specW`.
        // They share an array — and a caller — because they share every expensive thing: one
        // texture fetch / pattern evaluation per wavelength, one energy guard, one verdict. The
        // alternative was a second evaluation loop over the same material at three more
        // wavelengths, which is the same work done twice and two places to keep in step.
        //
        // NOTE that the VERDICT (`foldWorthIt`) is deliberately taken over the [0, K) half only.
        // The bundle's wavelengths are drawn from the same uniform variate as the hero, so a
        // test that read them would correlate the fold/no-fold decision with lambda_h and bias
        // the mixture — the exact failure the long note above foldWorthIt exists to prevent.
        // The quadrature bins are a function of the emitter alone, so they are safe to test and
        // their verdict governs both grids.
        // `K` is where the bundle's entries START in `fk`, whether or not the achromatic claim
        // is still live — the caller fills the array the same way either way, so the two halves
        // cannot drift apart when only one consumer is switched on (`-beamachro off -beamspec 4`
        // is exactly that case). Whether `foldT` is touched is read from `achroPath` here rather
        // than encoded in `K`, so there is one place that knows the layout.
        auto foldApply = [&](double fHero, const double* fk, int K, int S) {
            const double inv = 1.0 / fHero;
            if (achroPath) {
                if (!foldChroma) { for (int k = 0; k < K; ++k) foldT[k] = 1.0; foldChroma = true; }
                for (int k = 0; k < K; ++k) foldT[k] *= fk[k] * inv;
            }
            for (int i = 0; i < S; ++i) specW[i] *= fk[K + i] * inv;
        };
        // Retire BOTH spectral claims. Every event that ends the achromatic-path fold also ends
        // the bundle, because the two now live or die by the same rule: a wavelength-DIVERGENT
        // event (one after which the wavelengths no longer share a chord) kills them, and a
        // merely wavelength-DEPENDENT one is folded into `foldT`/`specW` instead. The decline
        // cases go through here too — a factor whose fold `foldWorthIt` judged a firefly risk
        // is exactly as much of a risk in `specW`, which carries the same 1/f(lambda_h).
        //
        // `g_foldKill` is only stamped while the fold was actually live, so the diagnostic
        // histogram keeps attributing each loss to the event that caused it rather than to
        // whatever came after. `specSec` is cleared unconditionally: with `-beamachro off`
        // there is no `achroPath` to guard it, and the bundle must still retire.
        auto retireSpectral = [&](FoldKill why) {
            if (achroPath) { g_foldKill = why; achroPath = false; }
            specSec = 0;
        };

        // PHOTON-BEAMS gather (CLI -beams, shared multi-camera pass only): a per-photon
        // RNG used ONLY to resample an INDEPENDENT medium-collision point for each camera's
        // volume splat (see the mediumEvent block below). Seeded from the main stream so it
        // is unique per photon and per thread; drawing it here perturbs `rng`, so this is
        // gated on beamGather && nCam>1 to keep every other mode bit-for-bit unchanged.
        const bool doBeamGather = beamGather && nCam > 1 && !forwardCatch && !scene.media.empty();
        Pcg32 crng;
        if (doBeamGather) crng.seed(((uint64_t)rng.next() << 32) ^ rng.next(),
                                    ((uint64_t)rng.next() << 32) ^ rng.next());

        // GRADIENT-INDEX (GRIN): if any medium carries an `ior` field, photons bend
        // through it via the shared Eikonal marcher (grin.h) — the same curved geometry
        // the backward/BDPT tracers use, so all transport paths agree. Gated so ordinary
        // scenes stay bit-identical (the marcher is never entered).
        const bool grinAny = grin::sceneHasGrin(scene);
        // A GRIN region bends the photon along a wavelength-dependent path and charges glass
        // absorption over the arc, so nothing downstream of a march is a shared chord. Rather
        // than reason about where the marcher was and was not entered, refuse the spectral
        // bundle for the whole photon in any GRIN scene — those scenes cannot deposit beams
        // inside the bending medium anyway (emitBeams skips it per medium).
        // The achromatic-path claim dies for the same reason and more strongly: a GRIN arc IS
        // a function of lambda, so the path's geometry differs per wavelength.
        if (grinAny) retireSpectral(FK_Grin);

        // PHOTON-BEAM deposit (mode M with -beams): store the crossed segment in the
        // view-independent beam map instead of splatting it to a camera list. GRIN media are
        // refused inside emitBeams, PER MEDIUM — a photon bends only within a gradient-index
        // region, so a fog crossing elsewhere in the same scene is still a straight chord and
        // still depositable. (This gate used to be the scene-wide `!grinAny`, which threw away
        // the beams for every ordinary medium in a scene that merely contained a GRIN lens.)
        const bool doBeamDeposit = (beamDeposit != nullptr) && !scene.media.empty();
        // Either beam path makes the photon cross media STRAIGHT: skip the analog free-flight
        // redirect below and attenuate by the crossing's transmittance instead.
        // `beamStraightOnly` is the aimed caustic pass, which must cross media by the SAME
        // rule as the main pass it is being MIS-combined with (an analog collision would set
        // sawScatter and destroy the L.S+.D classification the caustic map is defined by) —
        // but must NOT store beams, because the beam map belongs to the main pass and is
        // normalised by the main pass's nEmitted.
        const bool doBeamStraight = doBeamGather || doBeamDeposit ||
                                    (beamStraightOnly && !scene.media.empty());

        // MULTIPLE SCATTERING in the beam media (0.199.0; -beams-order). How many times this
        // photon has already scattered inside a beam-carried (non-GRIN) medium, so the order
        // cap can send it back to the single-scatter rule once it is spent.
        int beamScatters = 0;
        // Chord PROVENANCE for `-sunnee` (photonbeams.h PhotonBeam::srcEm / surf): the emitter
        // this photon was born on and the surface interactions it has made so far.
        int surfHits = 0;

        // CAUSTIC classification state (mode M's two-map split; see photonVertexKind above).
        // A deposit is a caustic iff the path so far reads L·S⁺·D — at least one FOCUS vertex
        // and no SCATTER vertex since the light. Both flags are monotone along the path: once
        // a wide scatter has destroyed the beam's coherence nothing downstream restores it, so
        // `sawScatter` is never cleared — and a diffuse deposit sets it itself, because the
        // bounce that continues past a diffuse vertex is indirect light by definition.
        bool sawFocus = false, sawScatter = false;

        for (int bounce = 0; bounce < maxBounce; ++bounce) {
            double dEvent;
            bool mediumEvent = false;
            int scatterMed = -1;   // which medium scattered (index into scene.media)
            Vec3 mp;
            // Beam multiple scattering permitted on THIS step? Re-evaluated per bounce because
            // the order cap retires it mid-path (after which the photon goes back to crossing
            // straight and booking the extinction as absorbed).
            const bool beamMS = doBeamStraight && beamMSAllowed(beamScatters);

            // GRIN curved marching pre-pass (does not consume a bounce): advance the ray
            // through any gradient-index region before the straight-ray hit test.
            //
            // MEDIA ARE INTEGRATED ALONG THE CURVE, one straight sub-segment at a time
            // (grin.h's marchSegments hook). Before 0.198.0 the marched span was skipped
            // entirely by the media code, so a scattering GRIN medium lost nearly all of
            // its scattering; the caller only ever sampled the short straight remainder.
            //
            // Along the curve every medium is transported ANALOG, `-beams` or not. That is
            // not a shortcut, it is the only thing a beam CAN'T represent: `-beams` trades
            // analog scattering for a straight chord that the gather integrates, and there
            // is no straight chord inside a bending region. Splitting the curve into its
            // ~10^4-10^5 Eikonal steps and storing a micro-beam for each would explode both
            // the beam map and the gather cost to buy an answer analog sampling already
            // gives exactly. So: curved span -> analog; straight remainder -> the normal
            // `-beams` rule below.
            if (grinAny) {
                double arc = 0.0; int whichC2 = -1;
                const bool hitInMarch = grin::marchSegments(scene, ray,
                    [&](const Vec3& so, const Vec3& sd, double slen, double& tStop) -> bool {
                        double t; int which;
                        if (!scene.media.empty() &&
                            sampleMediaCollision(scene, so, sd, slen, lambda, rng, t, which)) {
                            tStop = t; arc += t; whichC2 = which; return true;
                        }
                        arc += slen;
                        return false;
                    });
                // Glass Beer-Lambert over the marched arc: the post-march block below only
                // covers the straight remainder, so without this a dielectric enclosing a
                // GRIN region would not attenuate the curved part of the path at all.
                const double aG = curAbsorb(lambda);
                if (aG > 0.0 && arc > 0.0) beta *= std::exp(-aG * arc);
                if (hitInMarch) {
                    // The scatter below reads ray.d as the incoming direction; marchSegments
                    // already left it as this sub-segment's travel direction, and ray.o as
                    // the collision point.
                    mediumEvent = true; scatterMed = whichC2; mp = ray.o;
                }
            }

            Hit h = closestHitIn(scene, stk, ray);
            double dSurf = h.valid ? h.t : 1e30;
            // A collision found DURING the march already happened at ray.o, so nothing is
            // travelled on this iteration: zero distance for the aperture catch, the glass
            // Beer-Lambert (already charged per sub-segment above) and any beam crossing.
            const bool marchHit = mediumEvent;
            dEvent = mediumEvent ? 0.0 : dSurf;

            // Homogeneous fog: sample a free-flight collision. If it precedes the
            // surface, the photon interacts in the volume (in-scatter connect,
            // then scatter-or-absorb). Beer-Lambert transmittance is implicit in
            // the exponential free-flight, so beta is unchanged (analog MC).
            //
            // Under `-beams` with SINGLE scatter the photon does NOT redirect in a medium it
            // can cross as a straight beam — that crossing is handled by the doBeamStraight
            // block below. But a GRIN medium has no straight chord to store, so it is EXCLUDED
            // from the straight crossing and keeps full analog transport here (MedCurved),
            // exactly as it behaves without `-beams`. Before 0.198.0 the collision sampling was
            // skipped wholesale under `-beams`, so a scattering GRIN medium was charged
            // extinction by the straight crossing and never paid back: it acted purely
            // absorbing.
            //
            // Under `-beams` with MULTIPLE scattering (the 0.199.0 default) the filter goes
            // back to MedAll: EVERY medium is transported analog again. The photon scatters
            // where the free flight says it does, and the beam it deposited over this chord
            // (below) carries only the UNSCATTERED flux — the two are different orders and do
            // not overlap. See the long note on the beam block for why the beam must still run
            // the full chord to the surface even though the photon stops early.
            if (!mediumEvent && !scene.media.empty()) {
                double tMed; int which;
                if (sampleMediaCollision(scene, ray.o, ray.d, dSurf, lambda, rng, tMed, which,
                                         (doBeamStraight && !beamMS) ? MedCurved : MedAll)) {
                    dEvent = tMed; mediumEvent = true; scatterMed = which; mp = ray.o + ray.d * tMed;
                }
            }

            // Model A perspective catch: if the photon flies through the aperture
            // (nearer than the surface AND any fog collision), it lands on the film.
            if (forwardCatch && nCam > 0 && cams[0].cam && cams[0].film) {
                int px, py;
                const Camera& cc = *cams[0].cam;
                // A physical lens catches what enters its front element and comes out on the
                // sensor, and absorbs what enters and is clipped inside (lensCatch); a thin
                // lens catches what threads its disc.
                const int lc = cc.hasLens() ? cc.lensCatch(ray, dEvent, lambda, px, py)
                                            : Camera::LENS_MISSED;
                if (lc == Camera::LENS_ABSORBED) { e.absorbed += beta; return; }
                if (cc.hasLens() ? lc == Camera::LENS_CAUGHT
                                 : cc.catchPhoton(ray, dEvent, px, py)) {
                    // Same flux->film-irradiance normaliser as connectLens (model A): a
                    // caught photon deposits pupil FLUX into the cell; divide by the cell
                    // area A_cell = pixelPlaneArea()*filmDist^2 so brute-force C keeps the
                    // SAME absolute scale as the importance-sampled A (validated equal),
                    // and both now match B's radiance*camEq in absolute EV.
                    double cCell = cc.hasLens() ? cc.lensCellNorm()
                                                : 1.0 / (cc.pixelPlaneArea() * cc.filmDist * cc.filmDist);
                    // An env photon caught on its FIRST flight is the environment SEEN, so it
                    // carries the bilinear lookup every other mode's camera ray now shows
                    // (EnvMap::radianceSeen, 0.370.0) instead of the nearest texel its birth
                    // weight took from the sampler. Unbiased wherever the sampler can land; a
                    // zero-luminance texel is never drawn, so the ramp INTO one from a lit
                    // neighbour is missing -- an edge that is dark either way.
                    double seen = 1.0;
                    if (bounce == 0 && srcEmIdx >= 0 && scene.envMap &&
                        scene.emitters[srcEmIdx].shape == EmitterShape::Env) {
                        const Vec3 sky = -ray.d;
                        const double Ln = scene.envMap->radiance(sky, lambda);
                        seen = (Ln > 0.0) ? scene.envMap->radianceSeen(sky, lambda) / Ln : 0.0;
                    }
                    cams[0].film->add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * (beta * seen * cCell));
                    e.sensor += beta * seen;
                    return;
                }
            }

            // Beer-Lambert attenuation over the free path just travelled inside glass.
            // `betaPre` is the throughput at the segment start (before this attenuation),
            // so a beam-gather camera can re-apply glass absorption to ITS own resampled
            // collision distance tC instead of the photon's dEvent.
            double betaPre = beta;
            {
                double a = curAbsorb(lambda);
                // Glass absorption is spectral (that is what makes coloured glass coloured), and
                // this is one wavelength-DEPENDENT factor that is nonetheless NOT foldable into
                // `specW`, for a reason that lives downstream rather than here: `PhotonBeam`
                // stores a single scalar `absorb`, and the gather Beer-Lamberts every member of
                // the bundle with it, at the hero's coefficient, over the beam's own lead-in. A
                // reweighted bundle would therefore be right about the glass it has already
                // crossed and wrong about the glass it is still inside. Collapse both claims
                // here, BEFORE the deposit below reads them. (Making this foldable means giving
                // the record a per-member `absorb`, which is a second widening for a case — a
                // beam deposited inside coloured glass — that no scene in the suite exercises.)
                // Glass clear at `lambda` but absorbing elsewhere in the band retires it too
                // (Material::absorbChroma): the photon would go on standing for the wavelengths
                // the glass took away.
                if (a > 0.0) beta *= std::exp(-a * dEvent);
                const int cmA = stk.topMat();
                if (a > 0.0 || (cmA >= 0 && scene.mats[cmA].absorbChroma)) retireSpectral(FK_GlassAbsorb);
            }

            // PHOTON-BEAMS single-scatter gather (CLI -beams, shared multi-camera pass).
            // The photon crosses the medium in a STRAIGHT beam (the analog redirect above is
            // skipped), and each camera independently samples ONE in-scatter point along that
            // beam [ray.o, dSurf] with its OWN RNG stream `crng`, then splats it. Because the
            // deposit (the shared photon flight, traced ONCE for the whole flyby) is decoupled
            // from the per-camera gather, a volumetric flyby gets INDEPENDENT per-frame noise
            // instead of the single frozen speckle pattern that the shared point splat bakes
            // into every frame. Unbiased for SINGLE scattering: each per-camera resample is a
            // free-flight collision (pdf σ_t·Tr) over the crossing, and connectVolume's
            // albedo·phase·T_cam·β has that Tr cancelled — so E[per-camera splat] equals the
            // exact single-scatter in-scatter integral, independent of the photon's own flight.
            //
            // MULTIPLE SCATTERING (0.199.0, the default; `-beams-order 1` restores the old
            // single-scatter behaviour bit-for-bit). It is NOT inherent to photon beams that
            // they carry one order — the textbook formulation deposits a beam per straight
            // chord between scattering events, and the k-th chord carries k-th-order power.
            // Two things make that a small change here:
            //
            //  * These are LONG beams (photonbeams.h): the stored segment runs to the next
            //    SURFACE and the gather applies Tr(beam origin -> gather point) analytically.
            //    So the beam is a deterministic record of the UNSCATTERED flux along the whole
            //    chord, and it stays valid no matter where the photon itself stops.
            //  * The photon therefore goes back to plain ANALOG transport: it samples a free
            //    flight, scatters there (albedo roulette + phase sample), and the next chord
            //    deposits the next beam. E[power of chord k+1] is exactly the k-times-scattered
            //    flux, so beam k+1 supplies scattering order k+2. No overlap with beam k, which
            //    only ever carried unscattered flux.
            //
            // The one trap: do NOT truncate the beam at the sampled collision. That is the
            // SHORT-beam estimator, in which the truncation *is* the transmittance — combined
            // with a gather that also applies Tr analytically it would charge transmittance
            // twice and render the medium far too dark. Hence `dChord` below runs to dSurf
            // under MS even when the photon stopped at dEvent < dSurf.
            //
            // With SINGLE scatter the crossing instead runs to `dEvent`, not to the surface: a
            // GRIN medium is excluded from the straight rule (MedStraight below) and keeps
            // analog transport, so it can scatter the photon partway and cut the crossing
            // short. With no GRIN medium in the scene dEvent == dSurf and every draw is as it
            // was. A collision found during the GRIN march happened at ray.o itself, so there
            // is no chord at all on this step either way.
            const double dChord = marchHit ? 0.0 : (beamMS ? dSurf : dEvent);
            if (doBeamStraight && dChord > 0.0) {
                if (doBeamGather && nCam > 0 && !forwardCatch) {
                    double aC = curAbsorb(lambda);
                    for (int c = 0; c < nCam; ++c) {
                        if (!(cams[c].cam && cams[c].film)) continue;
                        double tC; int whichC;
                        if (!sampleMediaCollision(scene, ray.o, ray.d, dChord, lambda, crng, tC,
                                                  whichC, MedStraight))
                            continue;   // this camera saw no in-scatter along this beam
                        Vec3 xc = ray.o + ray.d * tC;
                        double betaC = (aC > 0.0) ? betaPre * std::exp(-aC * tC) : betaPre;
                        const Medium& smc = scene.media[whichC];
                        if (lensCam(*cams[c].cam)) connectLensVolume(scene, smc, *cams[c].cam, *cams[c].film, xc, ray.d, lambda, betaC, crng);
                        else          connectVolume(scene, smc, *cams[c].cam, *cams[c].film, xc, ray.d, lambda, betaC, crng);
                        camSpecularSplatVolumeAll(scene, smc, &cams[c], 1, xc, ray.d, lambda, betaC, crng);
                    }
                }
                // Mode M: store the crossing itself, so every camera of a flyby can gather
                // from it later without the photon knowing any camera exists.
                if (doBeamDeposit) {
                    // SPECTRAL FOLD: collapse the running per-bin ratios into the single CIE
                    // triple this beam is stored with, sum_k foldCie[k] * T_k. When the path
                    // has picked up NO spectral factor (foldChroma false) this is skipped and
                    // `achroCie` is still the emitter's cieMean, so an achromatic scene stores
                    // bit-for-bit the beams it stored before the spectral fold existed.
                    Vec3 cieF = achroCie;
                    if (achroPath && foldChroma) {
                        cieF = Vec3{0, 0, 0};
                        for (int k = 0; k < foldEm->foldN; ++k)
                            cieF += foldEm->foldCie[k] * foldT[k];
                    }
                    // The GATHER-time fold (scene.h, Scene::BowLut) needs the emitter's
                    // identity, and needs T == 1 — a path carrying per-bin weights has a
                    // spectrum the per-emitter table cannot describe. `foldChroma` is exactly
                    // that predicate, so an unweighted path names its emitter and a weighted
                    // one passes -1 and keeps the deposit-time behaviour.
                    int foldEmIdx = -1;
                    if (achroPath && !foldChroma && foldEm && !scene.emitters.empty()) {
                        const ptrdiff_t k = foldEm - &scene.emitters[0];
                        if (k >= 0 && k < (ptrdiff_t)scene.emitters.size() && k <= 32767)
                            foldEmIdx = (int)k;
                    }
                    // `beamScatters` counts medium scatters already made, so the chord being deposited
            // now is order beamScatters + 1: zero prior scatters is single scatter. Same
            // convention `-beams-order` uses -- `beamMSAllowed(n)` asks whether order n+2 is
            // still permitted, so the chord at n is order n+1.
            emitBeams(scene, ray.o, ray.d, dChord, lambda, betaPre, curAbsorb(lambda), rng,
                      beamScatters + 1,
                              beamMS ? MedAll : MedStraight, specLam, specSec,
                              achroPath ? &cieF : nullptr, foldEmIdx, specW,
                              srcEmIdx, surfHits);
                }
                if (!beamMS) {
                    // SINGLE SCATTER ONLY. Attenuate the photon by the medium extinction over
                    // the whole crossing (single-scatter transmission) so surfaces behind the
                    // fog get correctly dimmed direct light; the removed energy (out-scattered
                    // + absorbed) is booked as absorbed. The photon then continues STRAIGHT to
                    // the surface below.
                    // MedStraight: only the media that were actually crossed straight are
                    // charged here. A GRIN medium's extinction is NOT booked as absorption —
                    // it is carried by the analog free flight instead, which is what stops it
                    // from behaving as a pure absorber under `-beams`.
                    //
                    // Under MS there is deliberately nothing to do: the analog free flight
                    // above already carries the transmittance (the photon either survives the
                    // crossing with its throughput intact, or it collided and is about to
                    // scatter or be absorbed by the albedo roulette). Multiplying by Tr here as
                    // well would charge it twice.
                    double before = beta;
                    beta *= mediaTransmittance(scene, ray.o, ray.d, dChord,
                                               lambda, doBeamGather ? crng : rng, MedStraight);
                    e.absorbed += (before - beta);
                }
            }
            // BOTH SPECTRAL CLAIMS NOW USE THE SAME RULE: only a wavelength-DIVERGENT event
            // retires them. This is where the bundle used to be killed unconditionally, once
            // per transport iteration, and that line is gone (0.257.0).
            //
            // The old rule was conservative for a real reason — the record carried no
            // per-wavelength weight, so a bundle whose members' throughputs had begun to
            // diverge could not be represented at all, and the only sound thing to do was throw
            // it away before the first event that could diverge them. `PhotonBeam::wS` is that
            // missing weight, and `specW` is the tracer-side accumulator that fills it, so the
            // bundle can now do exactly what `achroPath` already did: survive an event that is
            // merely wavelength-DEPENDENT by folding its ratio in, and retire only at one that
            // is wavelength-DIVERGENT — one after which the members no longer share a chord, so
            // there is no single beam left to store them on.
            //
            // What that is worth, measured on gallery_rain: the old rule reached the light's
            // DIRECT crossing of a medium and nothing after it, which is most of a SHAFT's flux
            // but almost none of a CLOUD's — at albedo 0.9964 a photon scatters of the order of
            // 278 times inside that cloud, so one-chord-in-278 carried a bundle (36.2% of
            // deposits overall) while `achroPath`'s weaker rule reached 83.9%. The two now
            // coincide.
            //
            // A scatter in an ACHROMATIC medium is the archetypal survivable event: achromatic
            // sigma_t for the free flight that reached it, a flat albedo for the roulette
            // below, and an HG direction that depends only on `g`. A CHROMATIC one is the
            // archetypal fatal one, and not merely because its coefficients differ — its
            // `phaseSample` draws a DIRECTION as a function of lambda, so after it the
            // wavelengths are on different rays and no per-member weight could reconcile them.
            //
            // A SURFACE event is not decided here. It used to be — a pre-emptive catch-all that
            // retired the fold before the material was even known — and measurement showed that
            // single line was 87.8% of the residual "coloured bars". A diffuse albedo is
            // wavelength-dependent but not wavelength-divergent: it leaves the geometry
            // identical for every wavelength and only rescales the weight, which is exactly what
            // `foldT`/`specW` absorb. So the decision lives in the material switch, where the
            // albedo can be folded instead of thrown away. Everything in that switch which this
            // file does not explicitly fold still retires, so nothing became less conservative
            // by accident. (Device twin: render_cuda.cu, same position — still on the old
            // one-iteration rule, which is why a device-traced bundle is always equal-weight;
            // see known-issues.md, FOLD-GPU.)
            if ((achroPath || specSec > 0) && mediumEvent &&
                !mediumAchromatic(scene.media[scatterMed]))
                retireSpectral(FK_ChromaMedium);

            if (mediumEvent) {
                const Medium& sm = scene.media[scatterMed];
                // Was this collision inside the chord the beam block just covered? If so the
                // in-scatter here is ALREADY estimated — by the beam (mode M) or by each
                // camera's own resample along the same chord (modes A/B) — and splatting the
                // shared point as well would count the same scattering order twice. Only the
                // photon's REDIRECT is wanted from this event; the light it carries onward is
                // the next order, and that is delivered by the next chord's beam.
                //
                // A collision found during the GRIN march, or in a GRIN medium (which no beam
                // and no MedStraight resample ever covers), is not double counted and must
                // still splat — which is also exactly what happens with `-beams-order 1`,
                // where a non-GRIN medium can never reach this block at all.
                const bool coveredByBeam = beamMS && !marchHit && !sm.grin();
                if (nCam > 0 && !forwardCatch && !coveredByBeam) {
                    camSplatVolumeAll(scene, sm, cams, nCam, mp, ray.d, lambda, beta, rng);
                    camSpecularSplatVolumeAll(scene, sm, cams, nCam, mp, ray.d, lambda, beta, rng);
                }
                if (coveredByBeam) ++beamScatters;   // this photon just bought one more order
                // Scatter (prob albedo) or absorb; throughput unchanged on scatter.
                if (rng.uniform() >= sm.albedo(lambda)) { e.absorbed += beta; return; }
                double phPdf;   // sample the scatter direction from HG or the rainbow droplet phase
                ray = Ray{mp, sm.phaseSample(ray.d, lambda, rng, phPdf)};
                // An ANALOG collision redirects the photon by the phase function, which is wide
                // enough (even for a forward-peaked HG) that whatever focus the beam had is not
                // recoverable — so anything deposited downstream is ordinary indirect light.
                // A `-beams` STRAIGHT crossing never reaches here and correctly stays neutral:
                // it does not deflect the photon at all, so a gem seen through fog still writes
                // its caustic to the caustic map.
                sawScatter = true;
                continue;
            }

            if (!h.valid) { e.escaped += beta; return; }

            if (h.sensorId >= 0) {
                if (sensorFilm) deposit(scene.sensor, *sensorFilm, h.p, lambda, beta);
                e.sensor += beta;
                return;
            }
            ++surfHits;   // a surface interaction follows: chords after it are not direct light

            const Material* matp = &scene.mats[h.matId];
            // COMPOUND-DOUBLE-PICK: a mix or layered surface takes its deposit here, before the
            // photon's own pick (compoundGathers); the lobe's case below then deposits nothing.
            const bool compoundDeposit = compoundGathers(scene, *matp, h.fiberRadius);
            if (compoundDeposit) {
                const double w = compoundDepositWeight(scene, *matp, h, ray.d * -1.0, bounce == 0, beta);
                if (w > 0.0) depositPhoton(h.p, orientedGeoN(h), lambda, w, sawFocus && !sawScatter);
            }
            // Layered (coat over body): split the photon at the interface BEFORE the
            // switch. With prob R (Fresnel/Airy/manual) the coat reflects it as a
            // specular/glossy lobe (lossless, continue); otherwise it enters the body,
            // which picks one lobe among its children (leftover absorbs) and drives the
            // vertex exactly as that child. Energy-consistent per-photon lobe selection.
            if (matp->type == MatType::Layered) {
                const Material& cm = *matp;
                // SPECTRAL FOLD: a coat's Fresnel/Airy reflectance is exactly the kind of
                // sharply-peaked spectral factor the fold must NOT absorb — a thin-film
                // interference lobe can be near-zero at the hero wavelength and near-one two
                // bins away, so the 1/R(lambda_h) it would put into the weight is an outright
                // firefly generator. (Its iridescence is also the POINT of the material, and
                // a fold would smear the very colour it is there to produce.) Retire, exactly
                // as this vertex did before the spectral fold existed. The bundle goes with it,
                // and for a reason the fold's variance argument does not even have to reach:
                // the coat's roughness lobe is sampled once, so a reflected bundle would be
                // riding a direction chosen for the hero alone.
                retireSpectral(FK_Layered);
                // THE COAT'S SHEEN splats here, whichever lobe the coin below takes: the coat's
                // lobe toward the camera at the coat's reflectance there (GlossySplat::stack).
                // Neither of the coin's outcomes splats it: a coat reflection is a glossy bounce
                // the camera never sees at this vertex, and the body splats only itself.
                if (nCam > 0 && !forwardCatch && cm.coatChild >= 0 &&
                    glossySplats(scene.mats[(size_t)cm.coatChild]) &&
                    (specularSideSplats || !isSpecularType(cm.type))) {
                    const Vec3 wl{-ray.d.x, -ray.d.y, -ray.d.z};
                    const GlossySplat gsp{&scene, &scene.mats[(size_t)cm.coatChild], &h, h.n, wl, &cm};
                    // ...softened on the first bounce, as any glossy lobe's (glossyArrival).
                    double bS = beta * glossyArrival(scene.mats[(size_t)cm.coatChild], h, wl, bounce == 0);
                    camSplatAll(scene, cams, nCam, h.p, h.n, orientedGeoN(h), wl, lambda, bS,
                                0.0, rng, nullptr, &gsp);
                    camSpecularSplatAllGlossy(scene, cams, nCam, h.p, h.n, &lambda, &bS, 1, gsp, rng);
                }
                double R = layeredCoatReflectance(scene, cm, h, ray.d, lambda);
                if (rng.uniform() < R) {                    // coat reflection
                    double cr = materialRoughness(scene, cm, h);
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), cr, rng);
                    if (dot(o, h.n) <= 0) { e.absorbed += beta; return; }
                    // The coat's lobe is a glossy lobe (its own material, coatChild): softened on
                    // the first bounce unless it is a delta (glossyArrival), and its adjoint, as
                    // interactPhotonSpecular's.
                    const double sC = (cm.coatChild >= 0)
                        ? glossyArrival(scene.mats[(size_t)cm.coatChild], h, ray.d * -1.0, bounce == 0) : 1.0;
                    e.absorbed += beta * (1.0 - sC);
                    beta *= sC;
                    const double adj = glossyPhotonAdjoint(h, ray.d * -1.0, o);
                    if (!(adj > 0.0)) { e.absorbed += beta; return; }
                    e.absorbed += beta * (1.0 - adj);
                    beta *= adj;
                    if (!continueAlong(h, o, Lobe::Reflect, ray)) { e.absorbed += beta; return; }
                    // A clearcoat reflection is a mirror lobe: it focuses when tight (the
                    // highlight a polished coat throws IS a caustic), scatters when rough.
                    (cr <= kCausticGlossRoughness ? sawFocus : sawScatter) = true;
                    continue;                               // lossless but for the adjoint
                }
                int child = mixResolveChild(scene, cm, h, rng.uniform());  // body lobe (leftover absorbs; honours a weight map)
                if (child < 0) { e.absorbed += beta; return; }
                matp = &scene.mats[child];
            }
            // Stochastic mix: pick a child material (or absorb on the leftover
            // slice) BEFORE the switch, so the chosen child drives the vertex
            // exactly as if it were the surface material. Weights are constants,
            // so this is an unbiased per-photon lobe selection with beta unchanged.
            if (matp->type == MatType::Mix) {
                int child = mixResolveChild(scene, *matp, h, rng.uniform());
                if (child < 0) { e.absorbed += beta; return; }
                matp = &scene.mats[child];
            }
            const Material& m = *matp;
            // A light block's own surface ends the photon, as vcm.h's light walk does: it
            // reflects nothing, and a photon deposited there would be gathered by every other
            // surface within the radius (known-issues J-LAMP-PHOTONS).
            if (m.reflectsNothing) { e.absorbed += beta; return; }
            // Specular vertices skip the camera connection (a delta BSDF has no finite value to
            // connect with; the SDS limitation). A glossy one with a finite lobe splats below.
            switch (m.type) {
                case MatType::Dielectric:
                case MatType::Glossy: {
                    // SPECTRAL FOLD (FOLD-GLOSSY, 0.260.1): a glossy lobe's GEOMETRY is
                    // wavelength-free -- sampleGlossy reads the roughness and nothing else -- so
                    // every wavelength still travels the same chord out of this vertex, which is
                    // the actual test for keeping the claim (retire on wavelength DIVERGENCE,
                    // absorb wavelength DEPENDENCE into the per-bin ratios). Only its albedo
                    // varies with lambda, and that is exactly what foldT[] was built to carry,
                    // the same way Diffuse carries its reflectance. So: evaluate the albedo on
                    // the fold quadrature and the bundle, let foldWorthIt decline the saturated
                    // cases, and apply the ratio AFTER interactPhotonSpecular's survival roulette
                    // at r(lambda_hero) -- E[ r_hero * F_k r_k / r_hero ] = F_k r_k, Diffuse's
                    // arithmetic exactly. Measured before this on a fog Cornell with a coloured
                    // glossy sphere: 9.7 % of foldable beam deposits were retired as "specular".
                    // Mirrored in bdpt.h's randomWalk: the two tracers fill one PhotonBeam bank
                    // and must agree on what power / cieA / wS mean.
                    int    gK = 0, gS = specSec;
                    double gR[kFoldBins + kBeamSecMax];
                    bool   gFold = false;
                    // An iridescent glossy's albedo depends on the direction it is about to
                    // sample (the interference reflectance at the half-vector), which the fold's
                    // per-bin ratios, read before sampling, cannot carry: retire, as a layered
                    // coat does (FK_Layered), and for the same reason.
                    if (m.type == MatType::Glossy && m.iridescence && (achroPath || gS > 0)) {
                        retireSpectral(FK_DeclineGlossy); gS = 0;
                    } else if (achroPath || gS > 0) {
                        gK = foldEm->foldN;
                        for (int k = 0; k < gK + gS; ++k)
                            gR[k] = clamp01(reflectSlot(scene, m, h,
                                                        (k < gK) ? foldEm->foldLam[k] : specLam[k - gK]));
                        bool ok = gK > 0 && foldWorthIt(*foldEm, gR, gK);
                        for (int i = 0; i < gS && ok; ++i) ok = gR[gK + i] > 0.0;
                        if (ok) gFold = true;
                        else { retireSpectral(FK_DeclineGlossy); gS = 0; }
                    }
                    switch (photonVertexKind(scene, m, h)) {
                        case PV_FOCUS:   sawFocus   = true; break;
                        case PV_SCATTER: sawScatter = true; break;
                        default: break;
                    }
                    // A glossy lobe's first bounce softens toward the emitter (glossyArrival).
                    if (m.type == MatType::Glossy) {
                        const double sEm = glossyArrival(m, h, ray.d * -1.0, bounce == 0);
                        e.absorbed += beta * (1.0 - sEm);   // the ledger books what it removes
                        beta *= sEm;
                    }
                    // A glossy lobe with a finite value (glossySplats) splats to the camera as a
                    // diffuse vertex does, with the glossy BSDF toward it; the arriving beta, so
                    // before the roulette below.
                    if (nCam > 0 && !forwardCatch && glossySplats(m) &&
                        (specularSideSplats || !isSpecularType(scene.mats[h.matId].type))) {
                        const Vec3 wl{-ray.d.x, -ray.d.y, -ray.d.z};
                        const GlossySplat gsp{&scene, &m, &h, h.n, wl};
                        camSplatAll(scene, cams, nCam, h.p, h.n, orientedGeoN(h), wl, lambda, beta,
                                    0.0, rng, nullptr, &gsp);
                        camSpecularSplatAllGlossy(scene, cams, nCam, h.p, h.n, &lambda, &beta, 1, gsp, rng);
                    }
                    if (!interactPhotonSpecular(scene, cams, nCam, m, h, ray, beta, lambda, stk, rng, e,
                                                bounce == 0))
                        return;
                    if (gFold) {
                        const double rH = clamp01(reflectSlot(scene, m, h, lambda));
                        if (rH > 0.0) foldApply(rH, gR, gK, gS);
                        else retireSpectral(FK_ZeroWeight);
                    }
                    continue;
                }
                case MatType::ThinFilm:
                case MatType::Multilayer:
                case MatType::Mirror:
                case MatType::Grating:
                case MatType::HalfMirror:
                case MatType::Filter:
                case MatType::Hair:
                case MatType::Fluorescent: {
                    // Specular / wavelength-switching lobes, all handled by the shared
                    // helper (single source of truth with tracePhotonHero). Hair is here
                    // rather than with Diffuse because it does its own camera splat (its
                    // projection factor is the strand's longitudinal cosine, not dot(n,w))
                    // and its own Russian roulette on the exact lobe attenuation.
                    //
                    // Classify the vertex for the caustic split HERE rather than inside the
                    // helper: the caller is the one that holds `m`/`h`, and the helper is
                    // shared with tracePhotonHero (which does its own bookkeeping over C
                    // wavelengths), so keeping the classification out of it leaves exactly one
                    // definition of the rule (photonVertexKind) and no hidden state in the
                    // helper's signature.
                    switch (photonVertexKind(scene, m, h)) {
                        case PV_FOCUS:   sawFocus   = true; break;
                        case PV_SCATTER: sawScatter = true; break;
                        default: break;                       // PV_NEUTRAL: `filter` passes through
                    }
                    // SPECTRAL FOLD: every lobe in this group either bends the path as a
                    // function of lambda (dispersive refraction, a grating's diffraction
                    // order), interferes (thin film, multilayer), or switches the wavelength
                    // outright (fluorescence). In all three the wavelengths no longer share a
                    // chord at all, so there is no T(lam) to carry — the fold is not merely
                    // unprofitable here, it is undefined. Retire, and the bundle with it: this
                    // is the textbook wavelength-DIVERGENT vertex, the one case no per-member
                    // weight could ever rescue.
                    retireSpectral(FK_Specular);
                    if (!interactPhotonSpecular(scene, cams, nCam, m, h, ray, beta, lambda, stk, rng, e,
                                                bounce == 0))
                        return;
                    continue;
                }
                case MatType::DiffuseTransmit: {
                    // Two-lobe Lambertian: reflect albedo into the front hemisphere
                    // (+h.n, the incoming side) and transmit albedo into the back
                    // hemisphere (-h.n). Splat BOTH lobes to the camera -- each takes a camera
                    // on its own GEOMETRIC side (camSideOk tests the normal passed as ng), so
                    // passing the flipped normals for the transmit lobe images whichever side
                    // the camera is on. Because both lobes are non-specular, a directly-viewed
                    // translucent solid is VISIBLE in mode B (unlike clear dielectric).
                    double rhoR = clamp01(diffuseReflectance(scene, m, h, lambda));
                    double rhoT = clamp01(transmitSlot(scene, m, h, lambda));
                    double sum = rhoR + rhoT;
                    if (sum > 1.0) { rhoR /= sum; rhoT /= sum; sum = 1.0; }  // energy guard
                    // SPECTRAL FOLD (see foldWorthIt / foldApply). Both lobes are analog
                    // roulettes with beta unchanged, so each is foldable as
                    // T_k *= rho(lam_k)/rho(lambda_h) — but WHICH lobe is taken depends on
                    // lambda_h, and a verdict that depended on the lobe would therefore depend
                    // on lambda_h and bias the estimator. So test BOTH lobes up front and fold
                    // only if both are worth it; then apply the chosen lobe's ratios below.
                    // The energy guard is re-applied per bin because the guard changes the
                    // sampling PROBABILITY, and the probability is what the analog roulette
                    // actually uses — so it is the guarded value, not the raw slot, that the
                    // ratio has to be taken of.
                    //
                    // The SPECTRAL BUNDLE rides the same evaluation: entries [foldK, foldK+foldS)
                    // hold the same two slots at the bundle's own live wavelengths, so one pass
                    // over the material serves both grids (see foldApply). `foldS` is snapshot
                    // before the verdict because a decline retires the bundle, and the loop that
                    // filled the array must not disagree with the loop that consumes it.
                    int    foldK = 0, foldS = specSec;
                    double foldR[kFoldBins + kBeamSecMax], foldTr[kFoldBins + kBeamSecMax];
                    if (achroPath || foldS > 0) {
                        foldK = foldEm->foldN;
                        for (int k = 0; k < foldK + foldS; ++k) {
                            const double lk = (k < foldK) ? foldEm->foldLam[k]
                                                          : specLam[k - foldK];
                            double a = clamp01(diffuseReflectance(scene, m, h, lk));
                            double b = clamp01(transmitSlot(scene, m, h, lk));
                            const double s = a + b;
                            if (s > 1.0) { a /= s; b /= s; }
                            foldR[k] = a; foldTr[k] = b;
                        }
                        // One verdict, over the quadrature half only, governing both grids —
                        // and a decline retires both, because `specW` carries the very
                        // 1/f(lambda_h) whose firefly risk the verdict just weighed.
                        // `foldWorthIt` also rejects a zero bin, so the loops below can divide.
                        bool ok = foldK > 0 && foldWorthIt(*foldEm, foldR, foldK) &&
                                  foldWorthIt(*foldEm, foldTr, foldK);
                        if (ok)
                            for (int i = 0; i < foldS && ok; ++i)
                                ok = foldR[foldK + i] > 0.0 && foldTr[foldK + i] > 0.0;
                        if (!ok) { retireSpectral(FK_DeclineTransmit); foldS = 0; }
                    }
                    Vec3 ngo = orientedGeoN(h);
                    Vec3 wi = Vec3{-ray.d.x, -ray.d.y, -ray.d.z};   // toward the previous (light-side) vertex
                    // Photon-map deposit: incident flux at this translucent vertex (or at the
                    // compound it was picked from, above).
                    if (!compoundDeposit) depositPhoton(h.p, ngo, lambda, beta, sawFocus && !sawScatter);
                    sawScatter = true;   // whatever leaves this vertex is diffuse indirect light
                    if (nCam > 0 && !forwardCatch) {
                        camSplatAll(scene, cams, nCam, h.p,  h.n,  ngo, wi, lambda, beta, rhoR, rng);
                        camSplatAll(scene, cams, nCam, h.p, -h.n, -ngo, wi, lambda, beta, rhoT, rng);
                        camSpecularSplatAll(scene, cams, nCam, h.p,  h.n, lambda, beta, rhoR, rng);
                        camSpecularSplatAll(scene, cams, nCam, h.p, -h.n, lambda, beta, rhoT, rng);
                    }
                    // Analog scatter: reflect (prob rhoR), transmit (prob rhoT), else
                    // absorb — throughput unchanged on a scatter (bar the Veach adjoint
                    // correction, which is 1 for a flat/analytic surface). |cos| in the
                    // factor makes it lobe-agnostic, so (h.n, ngo) serve both lobes.
                    double u = rng.uniform();
                    if (u < rhoR) {
                        if (achroPath || foldS > 0) {
                            if (rhoR > 0.0) foldApply(rhoR, foldR, foldK, foldS);
                            else retireSpectral(FK_ZeroWeight);
                        }
                        Vec3 wo = cosineHemisphere(h.n, rng);
                        beta *= shadingAdjointCorr(wi, wo, h.n, ngo);
                        if (!continueAlong(h, wo, Lobe::Reflect, ray)) { e.absorbed += beta; return; } continue;
                    } else if (u < sum) {
                        if (achroPath || foldS > 0) {
                            if (rhoT > 0.0) foldApply(rhoT, foldTr, foldK, foldS);
                            else retireSpectral(FK_ZeroWeight);
                        }
                        Vec3 wo = cosineHemisphere(Vec3{-h.n.x, -h.n.y, -h.n.z}, rng);
                        beta *= shadingAdjointCorr(wi, wo, h.n, ngo);
                        if (!continueAlong(h, wo, Lobe::Transmit, ray)) { e.absorbed += beta; return; } continue;
                    }
                    e.absorbed += beta; return;
                }
                case MatType::Diffuse:
                default: {
                    double rho = clamp01(diffuseReflectance(scene, m, h, lambda));
                    // SPECTRAL FOLD — the vertex that the whole mechanism exists for. A
                    // Lambertian albedo is applied below as an ANALOG roulette (survive with
                    // probability rho(lambda_h), beta unchanged), so conditional on survival
                    // the fold carries T_k *= rho(lam_k)/rho(lambda_h) and its expectation
                    // over the roulette is rho(lam_k) — precisely the per-wavelength weight
                    // the spectral integral wants. Decided BEFORE the roulette so the verdict
                    // is a property of the surface rather than of this photon's luck.
                    // The SPECTRAL BUNDLE rides the same evaluation: entries [foldK, foldK+foldS)
                    // are the same albedo at the bundle's own live wavelengths (see foldApply),
                    // so a `-beamspec 4` bundle now survives a diffuse bounce carrying
                    // rho(lam_i)/rho(lambda_h) instead of being thrown away.
                    int    foldK = 0, foldS = specSec;
                    double foldRho[kFoldBins + kBeamSecMax];
                    if (achroPath || foldS > 0) {
                        foldK = foldEm->foldN;
                        for (int k = 0; k < foldK + foldS; ++k)
                            foldRho[k] = clamp01(diffuseReflectance(
                                scene, m, h, (k < foldK) ? foldEm->foldLam[k] : specLam[k - foldK]));
                        // One verdict, over the quadrature half only (it must not depend on
                        // lambda_h, and the bundle's wavelengths share the hero's variate), and
                        // a decline retires both grids — `specW` carries the same 1/rho(lambda_h)
                        // whose firefly risk the verdict just weighed.
                        bool ok = foldK > 0 && foldWorthIt(*foldEm, foldRho, foldK);
                        if (ok)
                            for (int i = 0; i < foldS && ok; ++i) ok = foldRho[foldK + i] > 0.0;
                        if (!ok) { retireSpectral(FK_DeclineDiffuse); foldS = 0; }
                    }
                    Vec3 ngo = orientedGeoN(h);
                    Vec3 wi = Vec3{-ray.d.x, -ray.d.y, -ray.d.z};   // toward the previous (light-side) vertex
                    // Light arriving below the shading normal lights nothing (bdpt::lambertLit):
                    // no deposit, no splat; and the first bounce softens toward the emitter it
                    // came from (bdpt.h emitterSoftening). Both 1 / no-ops where ns == ng.
                    double sEm;
                    if (!lambertArrival(h, ngo, wi, bounce == 0, sEm)) { e.absorbed += beta; return; }
                    e.absorbed += beta * (1.0 - sEm);
                    beta *= sEm;
                    // Photon-map deposit: incident flux at this diffuse vertex. Stored
                    // BEFORE the Russian-roulette reflect/absorb so the record captures
                    // the arriving power (direct on the first hit, indirect thereafter).
                    // Not on a strand, where no gather takes it (CURVE-GATHER), nor where the
                    // compound it was picked from took it (above).
                    if (!compoundDeposit && !gatherlessStrand(m, h.fiberRadius))
                        depositPhoton(h.p, ngo, lambda, beta * arrivalAdjoint(h, ngo, wi),
                                      sawFocus && !sawScatter);
                    sawScatter = true;   // whatever leaves this vertex is diffuse indirect light
                    if (nCam > 0 && !forwardCatch) {
                        camSplatAll(scene, cams, nCam, h.p, h.n, ngo, wi, lambda, beta, rho, rng);
                        camSpecularSplatAll(scene, cams, nCam, h.p, h.n, lambda, beta, rho, rng);
                    }
                    // Russian roulette: absorb with prob (1-rho), else scatter
                    // with beta unchanged. Unbiased; average path length ~1/(1-rho)
                    // bounces instead of running to the maxBounce cap.
                    if (rng.uniform() >= rho) { e.absorbed += beta; return; }
                    if (achroPath || foldS > 0) {
                        if (rho > 0.0) foldApply(rho, foldRho, foldK, foldS);
                        else retireSpectral(FK_ZeroWeight);
                    }
                    Vec3 wo;
                    beta *= diffuseContinuation(h, ngo, wi, rng, wo);
                    if (!continueAlong(h, wo, Lobe::Reflect, ray)) { e.absorbed += beta; return; }
                    continue;
                }
            }
        }
        e.residual += beta;
    }

    // Hero-wavelength forward photon tracer. Mirrors tracePhoton() but carries C
    // wavelengths (hero index 0 + C-1 stratified secondaries) down ONE shared BVH walk.
    // The hero drives every sampling decision with the same rng stream a single-λ photon
    // would; the secondaries ride the identical geometry, each accumulating its own
    // per-λ throughput beta[i]. Emission splits the power C ways (beta_i = base/C), so
    // the C-wavelength average is unbiased; at a dispersive / wavelength-switching lobe
    // (anything but Diffuse / DiffuseTransmit) the secondaries de-hero — terminate — and
    // the hero is boosted ×C so it alone carries a full-weight single-λ estimate onward
    // (PBRT-v4's TerminateSecondary convention; total power conserved exactly). The
    // caller gates this to scenes WITHOUT media / GRIN, so those branches are absent.
    void tracePhotonHero(const Scene& scene, const CamTarget* cams, int nCam,
                         Film* sensorFilm, Pcg32& rng, EnergyReport& e) const {
        if (scene.emitters.empty()) return;
        const int C = heroC;
        // --- Emission (geometry identical to the scalar tracer; λ-independent) ---
        int ei = scene.selectEmitter(rng);
        const Emitter& em = scene.emitters[ei];
        double u1 = rng.uniform(), u2 = rng.uniform();
        Vec3 origin, emitN, dir;
        double spotW = 1.0;
        double envPdfW = 0.0;
        double emitPatW = 1.0;                   // `emit pattern:` factor at the sampled point
        if (em.shape == EmitterShape::Spot) {
            origin = scene.lightPointNow(em, em.origin);
            double ct = em.spotCosOuter + u1 * (1.0 - em.spotCosOuter);
            double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
            double phi = 2.0 * PI * u2;
            Vec3 t, b; onb(em.beamDir, t, b);
            dir = scene.lightDirNow(em, t * (st * std::cos(phi)) + b * (st * std::sin(phi)) +
                                        em.beamDir * ct);
            emitN = scene.lightDirNow(em, em.beamDir);
            double omegaOuter = 2.0 * PI * (1.0 - em.spotCosOuter);
            spotW = spotFalloff(ct, em.spotCosInner, em.spotCosOuter) * omegaOuter / em.spotOmega;
        } else if (em.shape == EmitterShape::Env) {
            if (scene.envMap) {
                // The sampler draws the SKY direction the light comes FROM; the photon travels
                // the other way. Until 0.370.0 it flew TOWARD the texel it was drawn from, so
                // every forward mode (and M's / S's photon passes) lit the scene with the
                // environment point-mirrored -- a map bright toward +x lit a sphere's -x side --
                // while the backdrop, a camera-ray lookup, showed it the right way round.
                dir = -scene.envMap->sample(u1, u2, envPdfW);
            } else {
                dir = uniformSphere(u1, u2);
            }
            // A photon with a time in a moving scene enters through a disc wide enough for the
            // movers' sweep, its power scaled by the disc's area (Scene::lightRadius).
            const double R = scene.lightRadius();
            spotW = scene.lightScale() * scene.lightScale();
            Vec3 t, b; onb(dir, t, b);
            double rd = R * std::sqrt(rng.uniform());
            double pd = 2.0 * PI * rng.uniform();
            Vec3 disk = t * (rd * std::cos(pd)) + b * (rd * std::sin(pd));
            origin = scene.sceneCenter - dir * R + disk;
            emitN = dir;
        } else if (em.shape == EmitterShape::Sun) {
            // Distant directional sun — see the scalar tracer for the pdf argument.
            dir = em.sampleCone(em.beamDir, u1, u2);
            const double R = scene.lightRadius();             // as the env's, above
            spotW = scene.lightScale() * scene.lightScale();
            Vec3 t, b; onb(dir, t, b);
            double rd = R * std::sqrt(rng.uniform());
            double pd = 2.0 * PI * rng.uniform();
            origin = scene.sceneCenter - dir * R
                   + t * (rd * std::cos(pd)) + b * (rd * std::sin(pd));
            emitN = dir;
        } else {
            emitPatW = emitterSamplePoint(scene, em, u1, u2, origin, emitN, &spotW);   // its area now (scalar tracer)
            if (em.collimated) dir = scene.lightDirNow(em, em.beamDir);
            else if (em.normalTilted) {             // an aimed quad -- see the scalar tracer
                double pdfAim = 0.0;
                dir = scene.sampleAimDirNow(em, rng, pdfAim);
                if (!(pdfAim > 0.0)) return;
            } else dir = cosineHemisphere(emitN, rng);
        }
        // Aimed caustic emission — see the scalar tracer. Before `base *= spotW` for the
        // same reason (the aimed pass recomputes spotW).
        if (!applyCausticAim(scene, &em, origin, dir, emitN, spotW, rng)) return;
        if ((em.shape == EmitterShape::Env || em.shape == EmitterShape::Sun) &&
            !scene.skyEntryPastHaze(origin, dir))   // a blur's wider disc
            return;

        // Hero + stratified secondary wavelengths from this emitter's SPD (hero.h policy 1:
        // one base draw, C-1 wrapped strata). The hero must have a valid pdf; a dead
        // secondary carries beta 0 and simply splats nothing, so its pdf goes unread.
        double lam[hero::kHeroMax], pdfLam[hero::kHeroMax];
        if (!hero::sampleBundle(em.spd, rng.uniform(), C, lam, pdfLam)) return;
        // Per-λ throughput: base power split C ways. Image-env reweights each λ by the
        // directional radiance estimator (no-op for a constant env).
        double base = (scene.emitters.size() == 1) ? em.power : scene.totalPower;
        base *= spotW;
        double beta[hero::kHeroMax];
        for (int i = 0; i < C; ++i) beta[i] = base / C;
        if (em.shape == EmitterShape::Env && scene.envMap) {
            for (int i = 0; i < C; ++i) {
                double denom = 4.0 * PI * envPdfW * em.spdFn(lam[i]);
                beta[i] = (denom > 0.0) ? beta[i] * (scene.envMap->radiance(-dir, lam[i]) / denom) : 0.0;   // -dir: the sky
            }
        }
        // Achromatic post-multiplier — see the scalar tracer above for why this changes
        // no pdf and therefore introduces no bias.
        if (emitPatW != 1.0) for (int i = 0; i < C; ++i) beta[i] *= emitPatW;

        bool secAlive = (C > 1);
        auto activeSum = [&]() { double s = 0.0; int n = secAlive ? C : 1;
                                 for (int i = 0; i < n; ++i) s += beta[i]; return s; };
        e.emitted += activeSum();

        // Direct light -> camera (area/quad emitters only; matches the scalar tracer).
        if (nCam > 0 && !forwardCatch &&
            em.shape != EmitterShape::Spot && em.shape != EmitterShape::Env &&
            em.shape != EmitterShape::Sun) {
            // An aimed quad: rho = PI/aimNorm about the aim, gated by its face (scalar tracer).
            const double rhoE = em.normalTilted ? PI / em.aimNorm : 1.0;
            const Vec3 face = em.normalTilted ? scene.aimFaceNow(em) : Vec3{0, 0, 0};
            const Vec3* plane = em.normalTilted ? &face : nullptr;
            double rhoOne[hero::kHeroMax]; for (int i = 0; i < C; ++i) rhoOne[i] = rhoE;
            camSplatAllHero(scene, cams, nCam, origin, emitN, emitN, emitN, lam, beta, rhoOne, C, rng,
                            plane);
            camSpecularSplatAllHero(scene, cams, nCam, origin, emitN, lam, beta, rhoOne, C, rng, plane);
        }

        Ray ray{origin + dir * 1e-6, dir};
        MediumStack stk;                 // dielectric priority (Beer-Lambert uses hero λ)
        tracePhotonHeroLoop(scene, cams, nCam, sensorFilm, ray, stk, lam, beta,
                            secAlive, /*bounce0=*/0, rng, e, false, false,
                            /*envBorn=*/em.shape == EmitterShape::Env && scene.envMap != nullptr);
    }

    // Bounce loop for a hero bundle that is already sitting at (`ray`, `stk`) with
    // `secAlive ? heroC : 1` live wavelengths carrying `lamIn[]`/`betaIn[]`, resuming at
    // bounce index `bounce0`. Split out of tracePhotonHero so the `-herosplit` policy can
    // RE-ENTER it once per monochromatic sub-path that a dispersive interface fans out
    // (see the dispersive case below). Every such sub-path is spawned with
    // `secAlive == false`, and the split branch is guarded on `secAlive`, so a sub-path
    // can never split again — recursion is at most one level deep and the per-frame
    // footprint (a MediumStack plus two kHeroMax double arrays) is bounded.
    // `sawFocus`/`sawScatter` carry the caustic classification of the path that REACHED
    // (ray, stk) — see the same pair in tracePhoton. They are parameters rather than locals
    // precisely because a -herosplit sub-path resumes mid-path: it is spawned at a dispersive
    // interface, so it must inherit that vertex's FOCUS (and any earlier scatter), or every
    // split caustic would land in the wrong map.
    void tracePhotonHeroLoop(const Scene& scene, const CamTarget* cams, int nCam,
                             Film* sensorFilm, Ray ray, MediumStack stk,
                             const double* lamIn, const double* betaIn, bool secAlive,
                             int bounce0, Pcg32& rng, EnergyReport& e,
                             bool sawFocus = false, bool sawScatter = false,
                             bool envBorn = false) const {   // envBorn: mode C's direct env catch
        const int C = heroC;
        double lam[hero::kHeroMax], beta[hero::kHeroMax];
        // Copy only the LIVE entries: a monochromatic sub-path spawned by -herosplit only
        // fills slot 0 of its lamIn/betaIn, so reading all C would read indeterminate
        // values (harmless today since nUp==1 ignores them, but still UB).
        const int nLive = secAlive ? C : 1;
        for (int i = 0; i < nLive; ++i) { lam[i] = lamIn[i]; beta[i] = betaIn[i]; }
        for (int i = nLive; i < C; ++i) { lam[i] = 0.0; beta[i] = 0.0; }
        auto activeSum = [&]() { double s = 0.0; int n = secAlive ? C : 1;
                                 for (int i = 0; i < n; ++i) s += beta[i]; return s; };
        auto deHero = [&]() { if (!secAlive) return; beta[0] *= (double)C; secAlive = false; };

        for (int bounce = bounce0; bounce < maxBounce; ++bounce) {
            int nUp = secAlive ? C : 1;
            Hit h = closestHitIn(scene, stk, ray);
            double dEvent = h.valid ? h.t : 1e30;

            // Model C aperture catch: the photon physically threads the pupil.
            if (forwardCatch && nCam > 0 && cams[0].cam && cams[0].film && cams[0].cam->hasLens()) {
                // Through a physical lens. Entering the front element is geometry, the same
                // for every live wavelength; inside, the glass disperses, so each wavelength is
                // traced to its own cell, and one the lens clips is absorbed by it. A photon
                // that enters ends there, as the scalar tracer's does; one that misses flies on.
                const Camera& cc = *cams[0].cam;
                int pxs[hero::kHeroMax], pys[hero::kHeroMax];
                bool hit[hero::kHeroMax] = {};
                Vec3 lo, ld;
                if (cc.lensCatchEntry(ray, dEvent, lo, ld)) {
                    for (int i = 0; i < nUp; ++i)
                        hit[i] = cc.lensCatchImage(lo, ld, lam[i], pxs[i], pys[i]) == Camera::LENS_CAUGHT;
                    const double cCell = cc.lensCellNorm();
                    const bool envSeen = (bounce == 0 && envBorn);
                    EnvMap::Taps tp{};
                    if (envSeen) tp = scene.envMap->taps(-ray.d);
                    double sensorAdd = 0.0;
                    for (int i = 0; i < nUp; ++i) {
                        if (!hit[i]) { e.absorbed += beta[i]; continue; }   // clipped in the glass
                        double seen = 1.0;
                        if (envSeen) {
                            const double Ln = scene.envMap->radiance(-ray.d, lam[i]);
                            seen = (Ln > 0.0) ? scene.envMap->radianceAt(tp, lam[i]) / Ln : 0.0;
                        }
                        cams[0].film->add(pxs[i], pys[i], Vec3(cieX(lam[i]), cieY(lam[i]), cieZ(lam[i]))
                                                          * (beta[i] * seen * cCell));
                        sensorAdd += beta[i] * seen;
                    }
                    e.sensor += sensorAdd;
                    return;
                }
            } else if (forwardCatch && nCam > 0 && cams[0].cam && cams[0].film) {
                int px, py;
                if (cams[0].cam->catchPhoton(ray, dEvent, px, py)) {
                    const Camera& cc = *cams[0].cam;
                    double cCell = 1.0 / (cc.pixelPlaneArea() * cc.filmDist * cc.filmDist);
                    // first flight of an env photon: the env SEEN, bilinear (scalar twin above)
                    const bool envSeen = (bounce == 0 && envBorn);
                    EnvMap::Taps tp{};
                    if (envSeen) tp = scene.envMap->taps(-ray.d);
                    double sensorAdd = 0.0;
                    for (int i = 0; i < nUp; ++i) {
                        double seen = 1.0;
                        if (envSeen) {
                            const double Ln = scene.envMap->radiance(-ray.d, lam[i]);
                            seen = (Ln > 0.0) ? scene.envMap->radianceAt(tp, lam[i]) / Ln : 0.0;
                        }
                        cams[0].film->add(px, py, Vec3(cieX(lam[i]), cieY(lam[i]), cieZ(lam[i]))
                                                  * (beta[i] * seen * cCell));
                        sensorAdd += beta[i] * seen;
                    }
                    e.sensor += envSeen ? sensorAdd : activeSum();
                    return;
                }
            }

            // Beer-Lambert over the free path just travelled inside glass. A non-empty
            // stack implies we've already de-hero'd (dielectric entry de-heros), so
            // nUp == 1 whenever this is non-zero; the loop handles the general case.
            {
                int mi = stk.topMat();
                if (mi >= 0)
                    for (int i = 0; i < nUp; ++i) {
                        double a = scene.mats[mi].absorb(lam[i]);
                        if (a > 0.0) beta[i] *= std::exp(-a * dEvent);
                    }
            }

            if (!h.valid) { e.escaped += activeSum(); return; }

            if (h.sensorId >= 0) {
                if (sensorFilm)
                    for (int i = 0; i < nUp; ++i)
                        deposit(scene.sensor, *sensorFilm, h.p, lam[i], beta[i]);
                e.sensor += activeSum();
                return;
            }

            const Material* matp = &scene.mats[h.matId];
            // COMPOUND-DOUBLE-PICK (scalar twin): the compound's deposit, every live wavelength,
            // before the pick; the lobe's case below then deposits nothing.
            const bool compoundDeposit = compoundGathers(scene, *matp, h.fiberRadius);
            if (compoundDeposit) {
                const bool caus = sawFocus && !sawScatter;
                const Vec3 ngoC = orientedGeoN(h);
                const double w1 = compoundDepositWeight(scene, *matp, h, ray.d * -1.0, bounce == 0, 1.0);
                if (w1 > 0.0)
                    for (int i = 0; i < nUp; ++i) depositPhoton(h.p, ngoC, lam[i], beta[i] * w1, caus);
            }
            // Layered coat. The coat interface is NOT dispersive in DIRECTION -- the sheen is
            // a glossy lobe about the mirror direction and the body-lobe pick is a material
            // index -- so the only λ dependence is the scalar coat reflectance R(λ), and ONE
            // shared coin can serve the whole bundle: u is uniform, so P(u < R_i) == R_i
            // exactly per λ (common random numbers), with weights untouched just as in the
            // scalar twin. That keeps the bundle alive across a clearcoat instead of
            // collapsing it, which is what the backward hero loop does as of v0.115.1.
            //
            // Only a genuinely CHROMATIC coat -- a thin-film Airy stack where the coin lands
            // on different sides for different λ -- still de-heroes. A fan-out like
            // -herosplit's is NOT available here, because a forward sub-path cannot re-enter
            // this same vertex: the loop head has already run the model-C aperture catch, so
            // re-entering would deposit the photon into the film twice.
            if (matp->type == MatType::Layered) {
                const Material& cm = *matp;
                // The coat's sheen, over the bundle, before its coin (see the scalar twin).
                if (nCam > 0 && !forwardCatch && cm.coatChild >= 0 &&
                    glossySplats(scene.mats[(size_t)cm.coatChild]) &&
                    (specularSideSplats || !isSpecularType(cm.type))) {
                    const Vec3 wl{-ray.d.x, -ray.d.y, -ray.d.z};
                    const GlossySplat gsp{&scene, &scene.mats[(size_t)cm.coatChild], &h, h.n, wl, &cm};
                    double unused[hero::kHeroMax] = {};
                    const double sC = glossyArrival(scene.mats[(size_t)cm.coatChild], h, wl, bounce == 0);
                    double bS[hero::kHeroMax];
                    for (int i = 0; i < nUp; ++i) bS[i] = beta[i] * sC;
                    camSplatAllHero(scene, cams, nCam, h.p, h.n, orientedGeoN(h), wl, lam, bS,
                                    unused, nUp, rng, nullptr, &gsp);
                    camSpecularSplatAllGlossy(scene, cams, nCam, h.p, h.n, lam, bS, nUp, gsp, rng);
                }
                double Rl[hero::kHeroMax];
                for (int i = 0; i < nUp; ++i)
                    Rl[i] = layeredCoatReflectance(scene, cm, h, ray.d, lam[i]);
                const double uCoat = rng.uniform();
                const bool refl0 = uCoat < Rl[0];
                for (int i = 1; i < nUp; ++i)
                    if ((uCoat < Rl[i]) != refl0) { deHero(); nUp = 1; break; }
                if (refl0) {
                    double cr = materialRoughness(scene, cm, h);
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), cr, rng);
                    if (dot(o, h.n) <= 0) { e.absorbed += activeSum(); return; }
                    const double sC = (cm.coatChild >= 0)                             // see scalar twin
                        ? glossyArrival(scene.mats[(size_t)cm.coatChild], h, ray.d * -1.0, bounce == 0) : 1.0;
                    for (int i = 0; i < nUp; ++i) { e.absorbed += beta[i] * (1.0 - sC); beta[i] *= sC; }
                    const double adj = glossyPhotonAdjoint(h, ray.d * -1.0, o);
                    if (!(adj > 0.0)) { e.absorbed += activeSum(); return; }
                    for (int i = 0; i < nUp; ++i) { e.absorbed += beta[i] * (1.0 - adj); beta[i] *= adj; }
                    if (!continueAlong(h, o, Lobe::Reflect, ray)) { e.absorbed += activeSum(); return; }
                    (cr <= kCausticGlossRoughness ? sawFocus : sawScatter) = true;  // see scalar twin
                    continue;
                }
                int child = mixResolveChild(scene, cm, h, rng.uniform());   // honours a bound weight map
                if (child < 0) { e.absorbed += activeSum(); return; }
                matp = &scene.mats[child];
            }
            // Stochastic mix: resolve the child by the hero rng; secondaries ride along
            // the hero's chosen child (accepted approximation, see known-issues).
            if (matp->type == MatType::Mix) {
                int child = mixResolveChild(scene, *matp, h, rng.uniform());
                if (child < 0) { e.absorbed += activeSum(); return; }
                matp = &scene.mats[child];
            }
            const Material& m = *matp;
            if (m.reflectsNothing) { e.absorbed += activeSum(); return; }   // see tracePhoton

            switch (m.type) {
                case MatType::DiffuseTransmit: {
                    double rhoR[hero::kHeroMax], rhoT[hero::kHeroMax];
                    for (int i = 0; i < nUp; ++i) {
                        double rr = clamp01(diffuseReflectance(scene, m, h, lam[i]));
                        double rt = clamp01(transmitSlot(scene, m, h, lam[i]));
                        double s = rr + rt;
                        if (s > 1.0) { rr /= s; rt /= s; }        // per-λ energy guard
                        rhoR[i] = rr; rhoT[i] = rt;
                    }
                    Vec3 ngo = orientedGeoN(h);
                    Vec3 wi = Vec3{-ray.d.x, -ray.d.y, -ray.d.z};
                    // Photon-map deposit: store EVERY live wavelength as its own per-λ
                    // photon record (the gather keys off each photon's own λ). C records
                    // of base/C sum to base, and nEmitted counts PATHS, so the estimator
                    // stays energy-consistent with the scalar single-λ deposit.
                    {
                        const bool caus = sawFocus && !sawScatter;
                        if (!compoundDeposit)   // else the compound took it (above)
                            for (int i = 0; i < nUp; ++i)
                                depositPhoton(h.p, ngo, lam[i], beta[i], caus);
                        sawScatter = true;   // past a diffuse vertex it is indirect light
                    }
                    if (nCam > 0 && !forwardCatch) {
                        camSplatAllHero(scene, cams, nCam, h.p,  h.n,  ngo, wi, lam, beta, rhoR, nUp, rng);
                        camSplatAllHero(scene, cams, nCam, h.p, -h.n, -ngo, wi, lam, beta, rhoT, nUp, rng);
                        camSpecularSplatAllHero(scene, cams, nCam, h.p,  h.n, lam, beta, rhoR, nUp, rng);
                        camSpecularSplatAllHero(scene, cams, nCam, h.p, -h.n, lam, beta, rhoT, nUp, rng);
                    }
                    // Lobe pick + RR over the whole bundle (see the Diffuse case): the
                    // reflect/transmit probabilities are the per-lobe MAX over live λ, so no
                    // secondary is ever amplified. The maxima can sum past 1 (each λ alone is
                    // guarded), in which case both shrink proportionally. At nUp == 1 the two
                    // maxima are rhoR[0]/rhoT[0] and every reweight is *= 1.0.
                    double qR = hero::maxOf(rhoR, nUp), qT = hero::maxOf(rhoT, nUp);
                    double sumHero = qR + qT;
                    if (nUp > 1 && sumHero > 1.0) { qR /= sumHero; qT /= sumHero; sumHero = qR + qT; }
                    double uu = rng.uniform();
                    if (uu < qR) {                                // reflect (front)
                        // The reweight is deterministic absorption — book it, or the energy
                        // ledger loses the difference (sum/emitted would drop well below 1).
                        for (int i = 0; i < nUp; ++i) {
                            double w = rhoR[i] / qR;
                            e.absorbed += beta[i] * (1.0 - w);
                            beta[i] *= w;
                        }
                        Vec3 wo = cosineHemisphere(h.n, rng);
                        double corr = shadingAdjointCorr(wi, wo, h.n, ngo);
                        for (int i = 0; i < nUp; ++i) beta[i] *= corr;
                        if (!continueAlong(h, wo, Lobe::Reflect, ray)) { e.absorbed += activeSum(); return; } continue;
                    } else if (uu < sumHero) {                    // transmit (back)
                        for (int i = 0; i < nUp; ++i) {
                            double w = rhoT[i] / qT;
                            e.absorbed += beta[i] * (1.0 - w);
                            beta[i] *= w;
                        }
                        Vec3 wo = cosineHemisphere(Vec3{-h.n.x, -h.n.y, -h.n.z}, rng);
                        double corr = shadingAdjointCorr(wi, wo, h.n, ngo);
                        for (int i = 0; i < nUp; ++i) beta[i] *= corr;
                        if (!continueAlong(h, wo, Lobe::Transmit, ray)) { e.absorbed += activeSum(); return; } continue;
                    }
                    e.absorbed += activeSum(); return;
                }
                case MatType::Mirror:
                case MatType::Filter:
                case MatType::Glossy: {
                    switch (photonVertexKind(scene, m, h)) {      // caustic split; see tracePhoton
                        case PV_FOCUS:   sawFocus   = true; break;
                        case PV_SCATTER: sawScatter = true; break;
                        default: break;
                    }
                    // ACHROMATIC lobes (mirrors the backward tracer's radianceHero): the outgoing
                    // DIRECTION does not depend on λ, so the bundle keeps riding and only the
                    // per-λ coefficient differs. The scalar lobe survives by ANALOG Russian
                    // roulette on its coefficient; rolling that coin on the hero alone would kill
                    // live secondaries whenever c_hero == 0 (a Wratten gel is 0 over most of the
                    // spectrum) AND amplify by c_i/c_hero, so the survival probability is the MAX
                    // over live λ and survivors reweight by c_i/q <= 1. A mirror and a filter are
                    // delta lobes, with no camera connect, as in the scalar path; a glossy lobe
                    // with a finite value splats first, as the scalar path's does.
                    if (m.type == MatType::Glossy) {                // first bounce: glossyArrival
                        const double sEm = glossyArrival(m, h, ray.d * -1.0, bounce == 0);
                        for (int i = 0; i < nUp; ++i) { e.absorbed += beta[i] * (1.0 - sEm); beta[i] *= sEm; }
                    }
                    if (nCam > 0 && !forwardCatch && glossySplats(m) &&
                        (specularSideSplats || !isSpecularType(scene.mats[h.matId].type))) {
                        const Vec3 wl{-ray.d.x, -ray.d.y, -ray.d.z};
                        const GlossySplat gsp{&scene, &m, &h, h.n, wl};
                        double unused[hero::kHeroMax] = {};
                        camSplatAllHero(scene, cams, nCam, h.p, h.n, orientedGeoN(h), wl, lam, beta,
                                        unused, nUp, rng, nullptr, &gsp);
                        camSpecularSplatAllGlossy(scene, cams, nCam, h.p, h.n, lam, beta, nUp, gsp, rng);
                    }
                    double c[hero::kHeroMax];
                    for (int i = 0; i < nUp; ++i)
                        c[i] = (m.type == MatType::Filter) ? clamp01(transmitSlot(scene, m, h, lam[i]))
                                                           : clamp01(reflectSlot(scene, m, h, lam[i]));
                    const double q = hero::maxOf(c, nUp);
                    if (rng.uniform() >= q) { e.absorbed += activeSum(); return; }  // RR absorb
                    for (int i = 0; i < nUp; ++i) {                                // bounded reweight
                        double w = c[i] / q;
                        e.absorbed += beta[i] * (1.0 - w);   // deterministic part of the absorption
                        beta[i] *= w;
                    }
                    if (m.type == MatType::Mirror) {
                        if (!continueAlong(h, reflect(ray.d, h.n), Lobe::Reflect, ray)) { e.absorbed += activeSum(); return; }
                    } else if (m.type == MatType::Filter) {
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};      // direction unchanged
                    } else {
                        const GlossyDraw gd = drawGlossy(scene, m, h, ray.d, h.n, ray.d * -1.0, rng);
                        const Vec3& o = gd.o;
                        if (dot(o, h.n) <= 0) { e.absorbed += activeSum(); return; }  // below surface
                        const double adj = glossyPhotonAdjoint(h, ray.d * -1.0, o);  // see the scalar path
                        if (!(adj > 0.0)) { e.absorbed += activeSum(); return; }
                        for (int i = 0; i < nUp; ++i) { e.absorbed += beta[i] * (1.0 - adj); beta[i] *= adj; }
                        if (gd.irid) {
                            // The interference reflectance at the pair's half-vector, per λ, by
                            // the same roulette-on-the-max and bounded reweight as above.
                            double R[hero::kHeroMax];
                            for (int i = 0; i < nUp; ++i) R[i] = gd.R(lam[i]);
                            const double qR = hero::maxOf(R, nUp);
                            if (rng.uniform() >= qR) { e.absorbed += activeSum(); return; }
                            for (int i = 0; i < nUp; ++i) {
                                double w = R[i] / qR;
                                e.absorbed += beta[i] * (1.0 - w);
                                beta[i] *= w;
                            }
                        }
                        if (!continueAlong(h, o, Lobe::Reflect, ray)) { e.absorbed += activeSum(); return; }
                    }
                    continue;
                }
                case MatType::Dielectric:
                case MatType::ThinFilm:
                case MatType::Multilayer:
                case MatType::Grating:
                case MatType::HalfMirror:
                case MatType::Hair:
                case MatType::Fluorescent: {
                    switch (photonVertexKind(scene, m, h)) {      // caustic split; see tracePhoton
                        case PV_FOCUS:   sawFocus   = true; break;
                        case PV_SCATTER: sawScatter = true; break;
                        default: break;
                    }
                    // Dispersive / wavelength-switching: the outgoing direction (and, for a
                    // grating/fluorophore, the wavelength itself) depends on λ, so the bundle
                    // cannot keep riding one shared direction past this interface. A fiber
                    // belongs here for the same reason: its absorption sigma_a is per-λ, so
                    // the lobe attenuations A_p — and hence both the sampled direction and
                    // the survival probability — differ across the bundle. With -herosplit
                    // each wavelength gets its own strand interaction (that IS the coloured
                    // TT/TRT spread); otherwise the bundle de-heroes onto the scalar path.
                    if (heroSplit && secAlive && nUp > 1) {
                        // SPLIT-AT-DISPERSION (-herosplit): fan out instead of de-hero'ing.
                        // Each secondary runs the SAME interaction with its OWN λ — so it
                        // refracts along its own Snell direction / diffracts into its own
                        // grating order — and then continues as an independent monochromatic
                        // sub-path from this vertex. Weights are untouched (no ×C boost): the
                        // C sub-paths still carry base/C each, so they sum to the same total
                        // power the de-hero'd hero would have carried alone, and the energy
                        // ledger stays exact because every sub-path books its own fate.
                        for (int i = 1; i < nUp; ++i) {
                            if (beta[i] <= 0.0) continue;   // dead secondary: nothing to carry
                            double cl[hero::kHeroMax], cb[hero::kHeroMax];
                            cl[0] = lam[i]; cb[0] = beta[i];
                            MediumStack cstk = stk;         // sub-paths diverge from here on
                            Ray cray = ray;
                            if (interactPhotonSpecular(scene, cams, nCam, m, h, cray, cb[0],
                                                       cl[0], cstk, rng, e, bounce == 0))
                                tracePhotonHeroLoop(scene, cams, nCam, sensorFilm, cray, cstk,
                                                    cl, cb, /*secAlive=*/false, bounce + 1,
                                                    rng, e, sawFocus, sawScatter);
                            beta[i] = 0.0;                  // its energy is now that sub-path's
                        }
                        secAlive = false;                   // hero carries on alone, UNBOOSTED
                        if (!interactPhotonSpecular(scene, cams, nCam, m, h, ray, beta[0],
                                                    lam[0], stk, rng, e, bounce == 0))
                            return;
                        continue;
                    }
                    // Default policy: terminate secondaries, then run the shared scalar
                    // interaction on the (boosted) hero channel.
                    deHero();
                    if (!interactPhotonSpecular(scene, cams, nCam, m, h, ray, beta[0], lam[0], stk, rng, e,
                                                bounce == 0))
                        return;
                    continue;
                }
                case MatType::Diffuse:
                default: {
                    double rho[hero::kHeroMax];
                    for (int i = 0; i < nUp; ++i)
                        rho[i] = clamp01(diffuseReflectance(scene, m, h, lam[i]));
                    Vec3 ngo = orientedGeoN(h);
                    Vec3 wi = Vec3{-ray.d.x, -ray.d.y, -ray.d.z};
                    // Below the shading normal: nothing lit; the first bounce softened (as in
                    // tracePhoton).
                    double sEm;                                     // lambertArrival
                    if (!lambertArrival(h, ngo, wi, bounce == 0, sEm)) { e.absorbed += activeSum(); return; }
                    for (int i = 0; i < nUp; ++i) { e.absorbed += beta[i] * (1.0 - sEm); beta[i] *= sEm; }
                    // Photon-map deposit: store EVERY live wavelength as its own per-λ
                    // photon record (the gather keys off each photon's own λ). C records
                    // of base/C sum to base, and nEmitted counts PATHS, so the estimator
                    // stays energy-consistent with the scalar single-λ deposit.
                    {
                        const bool caus = sawFocus && !sawScatter;
                        // not on a strand (CURVE-GATHER), nor where the compound took it (above)
                        if (!compoundDeposit && !gatherlessStrand(m, h.fiberRadius))
                            for (int i = 0; i < nUp; ++i)
                                depositPhoton(h.p, ngo, lam[i], beta[i] * arrivalAdjoint(h, ngo, wi), caus);
                        sawScatter = true;   // past a diffuse vertex it is indirect light
                    }
                    if (nCam > 0 && !forwardCatch) {
                        camSplatAllHero(scene, cams, nCam, h.p, h.n, ngo, wi, lam, beta, rho, nUp, rng);
                        camSpecularSplatAllHero(scene, cams, nCam, h.p, h.n, lam, beta, rho, nUp, rng);
                    }
                    // Continuation RR over the WHOLE bundle: the survival probability is
                    // max_i rho_i, not the hero's own albedo, and every live λ reweights by
                    // rho_i/q <= 1. Rolling the coin on the hero alone (beta[i] *= rho_i/rho_0)
                    // amplifies a secondary by up to rho_max/rho_hero — on a saturated wall
                    // (redWall spans 0.05..0.75) a 15x weight spike per bounce, which cancels
                    // the whole stratification win. At nUp == 1, q == rho[0] and beta[0] *= 1.0.
                    const double q = hero::maxOf(rho, nUp);
                    if (rng.uniform() >= q) { e.absorbed += activeSum(); return; }        // RR absorb
                    for (int i = 0; i < nUp; ++i) {                                       // bounded reweight
                        double w = rho[i] / q;
                        e.absorbed += beta[i] * (1.0 - w);   // deterministic part of the absorption
                        beta[i] *= w;
                    }
                    Vec3 wo;
                    const double corr = diffuseContinuation(h, ngo, wi, rng, wo);
                    for (int i = 0; i < nUp; ++i) beta[i] *= corr;
                    if (!continueAlong(h, wo, Lobe::Reflect, ray)) { e.absorbed += activeSum(); return; }
                    continue;
                }
            }
        }
        e.residual += activeSum();
    }

    // --- One surface of a dielectric body: its optics --------------------------------------------
    // What a ray along `d` meets at `h`, a surface of material `m` -- plain (Dielectric), or coated
    // with a thin film (ThinFilm) or a multilayer stack (Multilayer) -- at wavelength L, the far
    // side having index `extIor` (exteriorIor: the enclosing medium's, or 1): which way each branch
    // goes, and how likely it is. The interface step tosses its coin on `R`, and mode M's spectral
    // gather re-evaluates the same function at each photon's wavelength (stepFactorAt), so the two
    // agree by construction.
    //   * A CLEAR surface reflects with probability R -- the Fresnel reflectance, or the coating's
    //     interference reflectance for the stack the ray actually crosses -- and otherwise refracts
    //     by Snell's law into the other medium. The index is spectral, hence dispersion. Past the
    //     critical angle it reflects with R = 1 (`tir`).
    //   * An OPAQUE one (Material::coatOpaque: a coating on an absorbing substrate, or a stack with
    //     an absorbing layer) reflects with probability R and absorbs the rest: opaque structural
    //     colour. It is one-sided, so met from inside the body it has R = 0.
    struct InterfaceOptics {
        bool entering = false;   // met from outside the material
        Vec3 nl;                 // the normal on the incidence side
        double cosI = 0.0;       // > 0
        bool opaque = false;     // reflect with R, else absorbed
        bool tir = false;        // total internal reflection: R = 1
        double eta = 1.0;        // n_incidence / n_transmission, on a clear surface
        double cosT = 0.0;       // the refraction cosine, on a clear surface short of TIR
        double R = 0.0;          // the probability of the reflected branch
    };
    static InterfaceOptics interfaceOptics(const Scene& scene, const Material& m, const Hit& h,
                                           const Vec3& d, double extIor, double L) {
        InterfaceOptics o;
        o.entering = dot(d, h.ng) < 0.0;
        o.nl = o.entering ? h.ng : -h.ng;
        o.cosI = -dot(d, o.nl);
        const double ns = m.ior(L);                     // the body's (the substrate's) index
        const int nL = (int)m.layerN.size();
        if (m.coatOpaque(L)) {
            o.opaque = true;
            if (!o.entering) return o;                  // inside the absorbing body: absorbed
            const double ks = m.substrateK(L);
            o.R = (m.type == MatType::ThinFilm)
                ? thinFilmReflectance(extIor, m.filmIor(L), ns, ks, materialFilmThickness(scene, m, h),
                                      o.cosI, L)
                : multilayerReflectance(extIor, o.cosI, L, m.layerN.data(), m.layerK.data(),
                                        m.layerThick.data(), nL, ns, ks);
            return o;
        }
        const double n1 = o.entering ? extIor : ns;     // the incidence side
        const double n2 = o.entering ? ns : extIor;     // the transmission side
        o.eta = n1 / n2;
        const double sin2t = o.eta * o.eta * (1.0 - o.cosI * o.cosI);
        if (sin2t > 1.0) { o.tir = true; o.R = 1.0; return o; }
        o.cosT = std::sqrt(1.0 - sin2t);
        if (m.type == MatType::ThinFilm) {
            // Incidence medium, film, transmission medium: reciprocal, so a ray entering and one
            // leaving see the same R.
            o.R = thinFilmReflectance(n1, m.filmIor(L), n2, 0.0, materialFilmThickness(scene, m, h),
                                      o.cosI, L);
        } else if (m.type == MatType::Multilayer) {
            // From the incidence side: a ray leaving (inside the substrate) meets the layers in
            // reverse.
            o.R = o.entering
                ? multilayerReflectance(extIor, o.cosI, L, m.layerN.data(), m.layerK.data(),
                                        m.layerThick.data(), nL, ns, 0.0)
                : multilayerReflectance(ns, o.cosI, L, m.layerN.data(), m.layerK.data(),
                                        m.layerThick.data(), nL, extIor, 0.0, /*reversed=*/true);
        } else {
            const double rs = (n1 * o.cosI - n2 * o.cosT) / (n1 * o.cosI + n2 * o.cosT);
            const double rp = (n1 * o.cosT - n2 * o.cosI) / (n1 * o.cosT + n2 * o.cosI);
            o.R = 0.5 * (rs * rs + rp * rp);
        }
        return o;
    }

    // The probability of the branch a step took, at the wavelength `o` describes, times the radiance
    // factor (n_from / n_to)^2 if it refracted -- what mode M's spectral gather folds per photon
    // (stepFactorAt, InterfaceStep::factor). A refraction that cannot happen at that wavelength --
    // past its critical angle, or into a substrate that absorbs there -- has probability 0.
    static double branchProb(const InterfaceOptics& o, bool refracted) {
        if (!refracted) return o.R;
        return (o.opaque || o.tir) ? 0.0 : 1.0 - o.R;
    }
    static double branchFactor(const InterfaceOptics& o, bool refracted) {
        const double p = branchProb(o, refracted);
        return refracted ? p * o.eta * o.eta : p;
    }

    // --- The interface step: every host walk's way through a dielectric body's surface --------
    // Which quantity a walk carries across an interface: a camera-side walk carries RADIANCE,
    // which a refraction scales by (n_from / n_to)^2 -- radiance over n^2 is what a ray conserves
    // there -- and a light-side walk carries FLUX, which it does not (RADIANCE-ETA2). The step
    // takes one, so a walk cannot leave the factor out by omission.
    enum class Transport { Radiance, Flux };

    // What a step did. `ray` continues the walk (the incoming direction, straight on, at a
    // suppressed boundary); `scale` is the throughput factor -- (n_from / n_to)^2 for a Radiance
    // walk that refracted, 1 otherwise; `whittedWeight` mode W's branch weight (1 at a
    // pass-through and at total internal reflection); `factor` the branch's probability times
    // (n_from / n_to)^2 if it refracted, at the walk's own wavelength (branchFactor: 1 at a
    // pass-through) -- what stepFactorAt gives at any other, and `prob` the branch's probability
    // alone (the same, short of the radiance factor); `extMat` the material whose index
    // the far side had (-1: air, or the flat model -- index 1); `absorbed` when an opaque coating
    // or substrate took the ray (the walk ends there).
    struct InterfaceStep {
        Ray ray;
        double scale = 1.0;
        double whittedWeight = 1.0;
        double factor = 1.0;
        double prob = 1.0;
        bool entering = false;      // met the surface from outside the material
        bool refracted = false;     // crossed into the other medium
        bool suppressible = false;  // the priority rule suppresses it wherever its coat is clear
        bool passThrough = false;   // a boundary the priority rule suppressed
        bool absorbed = false;
        int extMat = -1;
    };

    // The far-side index for a surface of material `m` whose far side is the top of `outer` (the
    // stack without m): the enclosing medium's index when both carry an explicit `priority` (air
    // always counts), else 1 -- the flat air<->glass model, so priority-free scenes render as they
    // always did. `extMat` receives that medium (-1 for index 1); `ranked` whether the rule held.
    static double exteriorIor(const Scene& scene, const Material& m, const MediumStack& outer,
                              double lambda, int& extMat, bool* ranked = nullptr) {
        const int top = outer.topMat();
        const bool rk = m.hasPriority() &&
            (outer.empty() || (top >= 0 && scene.mats[(size_t)top].hasPriority()));
        if (ranked) *ranked = rk;
        extMat = (rk && top >= 0) ? top : -1;
        return (extMat >= 0) ? scene.mats[(size_t)extMat].ior(lambda) : 1.0;
    }

    // One surface of a dielectric body -- plain (Dielectric), or coated with a thin film
    // (ThinFilm) or a multilayer stack (Multilayer) -- for every host walk; the twin of the
    // device's dDielectricStep. First the nested-dielectric PRIORITY rule (Schmidt & Budge 2002):
    // the far side's index from the enclosing medium (exteriorIor), and a boundary of no higher
    // priority than the medium it lies in suppressed -- the ray goes straight on. An opaque coat
    // is never entered, so it is no medium and the rule never suppresses it. Then the surface's
    // own optics (interfaceOptics): reflect, refract, or -- opaque -- be absorbed. A body the ray
    // enters goes on `stk` and comes off when it leaves, a coated one as a glass one does -- so
    // the walk inside knows its index (mode W's localN2), and a body inside it ranks against it.
    // `m` must be an element of scene.mats (a Mix/Layered leaf is). `whitted` asks for mode W's
    // deterministic step: the DOMINANT branch (reflect iff R >= 0.5) with that branch's
    // probability carried as weight, and no frosting -- at 1 spp a coin here is not noise but
    // salt and pepper, a glass ball rendered as a speckled blob. An opaque coat in mode W always
    // reflects and carries R, as Mirror and Filter do there.
    InterfaceStep interfaceStep(const Scene& scene, const Material& m, const Hit& h, const Vec3& d,
                                double lambda, Pcg32& rng, MediumStack& stk, Transport tr,
                                bool whitted = false) const {
        const long long mi = &m - scene.mats.data();
        if (mi < 0 || mi >= (long long)scene.mats.size())
            throw std::logic_error("Renderer::interfaceStep: material not in scene.mats");
        InterfaceStep st;
        st.entering = dot(d, h.ng) < 0.0;
        // The far side: the medium the ray comes from when it enters, the one it returns to when
        // it leaves (the stack without m).
        MediumStack after;
        // The stack knows a solid by Material::solidId, so routed copies of one material are one
        // solid however its parts were tagged.
        const int si = m.solidId((int)mi);
        if (!st.entering) { after = stk; after.popMat(si); }
        const MediumStack& outer = st.entering ? stk : after;
        bool ranked = false;
        const double extIor = exteriorIor(scene, m, outer, lambda, st.extMat, &ranked);
        st.suppressible = ranked && outer.topMat() >= 0 && m.priority <= outer.topPri();
        if (st.suppressible && !m.coatOpaque(lambda)) {                      // suppressed
            if (st.entering) stk.push(si, m.priority); else stk.popMat(si);
            st.ray = Ray{h.p + d * 1e-6, d};
            st.passThrough = true;
            st.extMat = -1;
            return st;
        }
        const InterfaceOptics o = interfaceOptics(scene, m, h, d, extIor, lambda);
        if (o.opaque) {                                 // reflect with R, else absorbed
            if (!o.entering) { st.absorbed = true; return st; }
            if (whitted) st.whittedWeight = o.R;
            else if (rng.uniform() >= o.R) { st.absorbed = true; return st; }
            const Vec3 out = normalize(reflect(d, o.nl));
            st.ray = Ray{h.p + out * 1e-6, out};
            st.factor = st.prob = o.R;
            return st;
        }
        Vec3 out;
        if (o.tir) {
            out = reflect(d, o.nl);                     // lossless: whittedWeight stays 1
        } else {
            const bool doReflect = whitted ? (o.R >= 0.5) : (rng.uniform() < o.R);
            if (whitted) st.whittedWeight = doReflect ? o.R : 1.0 - o.R;
            if (doReflect) out = reflect(d, o.nl);
            else { out = o.eta * d + o.nl * (o.eta * o.cosI - o.cosT); st.refracted = true; }
        }
        out = normalize(out);
        // Frosted glass (`roughness`): jitter the chosen lobe, rejecting a jitter that would cross
        // to the other side, so no light leaks through.
        if (m.type == MatType::Dielectric && !whitted) {
            const double rough = materialRoughness(scene, m, h);
            if (rough > 1e-3) {
                const Vec3 pert = sampleGlossy(out, rough, rng);
                if (st.refracted ? (dot(pert, o.nl) < 0.0) : (dot(pert, o.nl) > 0.0)) out = pert;
            }
        }
        st.ray = Ray{h.p + out * 1e-6, out};
        st.factor = branchFactor(o, st.refracted);
        st.prob = branchProb(o, st.refracted);
        // Radiance over n^2 is what a ray conserves across a refraction, so a camera-side walk,
        // which carries radiance, takes (n_from / n_to)^2 there, and a light-side one, which
        // carries flux, does not (RADIANCE-ETA2). It cancels for a ray that goes into a body and
        // out again; not for a camera path that ends inside one.
        if (tr == Transport::Radiance && st.refracted) st.scale = o.eta * o.eta;
        if (st.refracted) { if (st.entering) stk.push(si, m.priority); else stk.popMat(si); }
        return st;
    }

    // What mode M's spectral gather folds per photon wavelength (SpecThr) for the interface step
    // `st` that the ray `d` made at the hit `h`, at wavelength L: the probability of the branch it
    // took, had the walk been at L -- the Fresnel or interference reflectance for the same
    // incidence, or one minus it -- times the radiance factor (n_from / n_to)^2 at L if it
    // refracted; at a boundary the priority rule suppresses wherever the coat is clear, 1 for a
    // pass-through where it is clear at L and 0 for any branch the walk could not have taken
    // there (a coat opaque at one wavelength and clear at another is a medium only at the
    // second). Folded against st.factor, its value at the walk's own
    // wavelength, it re-expresses the step's coin at each photon's: a coat's interference colour
    // lives entirely in that coin, a dispersive index's partly. `prob`, if given, receives the
    // branch's probability alone at L (SpecThr::mulCoin; st.prob at the walk's wavelength).
    double stepFactorAt(const Scene& scene, const Material& m, const InterfaceStep& st,
                        const Hit& h, const Vec3& d, double L, double* prob = nullptr) const {
        double f;
        if (st.suppressible && !m.coatOpaque(L)) f = st.passThrough ? 1.0 : 0.0;
        else if (st.passThrough) f = 0.0;
        else {
            const double ne = (st.extMat >= 0) ? scene.mats[(size_t)st.extMat].ior(L) : 1.0;
            const InterfaceOptics o = interfaceOptics(scene, m, h, d, ne, L);
            if (prob) *prob = branchProb(o, st.refracted);
            return branchFactor(o, st.refracted);
        }
        if (prob) *prob = f;                             // a pass-through's factor is its probability
        return f;
    }

    // The square of the index of the medium a path is in (the top of its stack; 1 in air): what
    // turns a camera walk's throughput, which carries radiance, back into air units -- for mode
    // W's cutoff, and for its ambient fill, which is radiance in air and so n^2 brighter inside.
    static double localN2(const Scene& scene, const MediumStack& stk, double lambda) {
        const int top = stk.topMat();
        if (top < 0) return 1.0;
        const double n = scene.mats[(size_t)top].ior(lambda);
        return n * n;
    }

    // Reflective diffraction grating. The exact vector grating equation preserves
    // the tangential direction component along the grooves and shifts it by
    // m*(lambda/d) along the in-surface dispersion axis t_hat (perpendicular to the
    // grooves): v_t = u_t + m*(lambda/d)*t_hat, v_n = +sqrt(1-|v_t|^2) on the
    // incidence side (reflection). One order m is drawn stochastically among the
    // propagating orders (|v_t| < 1) with an idealised efficiency ~1/(1+|m|);
    // evanescent orders are excluded and the remaining weights renormalised, so the
    // reflected fraction is lossless (analog MC, beta unchanged). m=0 is specular.
    // The equation is reciprocal (m <-> -m), so the backward tracer reuses it.
    // Sets `absorbed` if no order propagates (degenerate grazing case).
    //
    // `whittedU` (non-null only in mode W) replaces the rng draw with one coordinate off the
    // deterministic (sIdx, bounce) lattice -- see BackwardRenderer::whittedOrderU. This is NOT
    // the dominant-branch trade the Fresnel materials make: the pick is already ANALOG (order i
    // with probability wgt[i]/wsum, throughput untouched), so a stratified u keeps the estimator
    // unbiased and only removes the per-pixel luck. It does change WHICH order a given u maps
    // to, though: the candidate list is built mm = -M..+M, so a raw u = 0 would select the most
    // NEGATIVE order. On the whitted path the walk therefore visits candidates in DESCENDING
    // efficiency (0, -1, +1, -2, +2, ...) so u = 0 gives the specular m = 0 and a 1-spp preview
    // is the undiffracted image. Total mass is the same wsum either way, so the two orders of
    // traversal agree in distribution; the stochastic path keeps its original ascending walk and
    // stays bit-identical.
    // The order normaliser at `lambda`: sum of 1/(1+|m|) over the orders that propagate for this
    // incidence, and whether `order` is one of them. gratingDiffract's own loop, kept beside it so
    // the two cannot drift: JSPEC re-expresses the order a camera walk took at other wavelengths,
    // where its probability is 1/(1+|order|) over THIS normaliser (0 if the order is evanescent).
    double gratingWsum(const Material& m, const Hit& h, const Vec3& din, double lambda,
                       int order, bool& orderLive) const {
        orderLive = false;
        Vec3 nl = dot(din, h.ng) < 0.0 ? h.ng : -h.ng;
        Vec3 g = m.grooveDir - nl * dot(m.grooveDir, nl);
        if (dot(g, g) < 1e-12)
            g = std::fabs(nl.x) < 0.9 ? cross(nl, Vec3{1, 0, 0}) : cross(nl, Vec3{0, 1, 0});
        g = normalize(g);
        Vec3 t = normalize(cross(nl, g));
        Vec3 ut = din - nl * dot(din, nl);
        int M = diffraction ? std::max(0, std::min(m.gratingMaxOrder, 32)) : 0;
        double lod = lambda / m.grooveSpacing, wsum = 0.0;
        for (int mm = -M; mm <= M; ++mm) {
            Vec3 a = ut + t * ((double)mm * lod);
            if (dot(a, a) >= 1.0) continue;
            wsum += 1.0 / (1.0 + std::abs(mm));
            if (mm == order) orderLive = true;
        }
        return wsum;
    }

    Ray gratingDiffract(const Material& m, const Hit& h, const Vec3& din,
                        double lambda, Pcg32& rng, bool& absorbed,
                        const double* whittedU = nullptr, int* pickOut = nullptr) const {
        absorbed = false;
        Vec3 nl = dot(din, h.ng) < 0.0 ? h.ng : -h.ng;      // incidence-side normal
        // Groove direction projected into the surface; dispersion axis perpendicular.
        Vec3 g = m.grooveDir - nl * dot(m.grooveDir, nl);
        if (dot(g, g) < 1e-12)
            g = std::fabs(nl.x) < 0.9 ? cross(nl, Vec3{1, 0, 0}) : cross(nl, Vec3{0, 1, 0});
        g = normalize(g);
        Vec3 t = normalize(cross(nl, g));                   // in-surface dispersion axis
        Vec3 ut = din - nl * dot(din, nl);                  // tangential incident component

        int M = diffraction ? std::max(0, std::min(m.gratingMaxOrder, 32)) : 0;
        double lod = lambda / m.grooveSpacing;              // lambda / d (dimensionless)
        int   ord[65]; double wgt[65]; int cnt = 0; double wsum = 0.0;
        int   slot[65];                                     // mm+M -> index in ord[], -1 evanescent
        if (whittedU) for (int i = 0; i <= 2 * M; ++i) slot[i] = -1;
        for (int mm = -M; mm <= M; ++mm) {
            Vec3 a = ut + t * ((double)mm * lod);
            if (dot(a, a) >= 1.0) continue;                 // evanescent -> excluded
            double w = 1.0 / (1.0 + std::abs(mm));          // idealised efficiency
            slot[mm + M] = cnt;                             // (only read on the whitted path)
            ord[cnt] = mm; wgt[cnt] = w; wsum += w; ++cnt;
        }
        if (cnt == 0 || wsum <= 0.0) { absorbed = true; return Ray{}; }
        int pick;
        if (whittedU) {
            // Deterministic: same inversion, but over the descending-efficiency traversal
            // 0, -1, +1, -2, +2, ... so u = 0 lands on the specular order.
            double xi = *whittedU * wsum, acc = 0.0;
            pick = ord[cnt - 1];                            // guard against fp round-off at u->1
            bool done = false;
            for (int k = 0; k <= M && !done; ++k) {
                for (int s = 0; s < (k == 0 ? 1 : 2); ++s) {
                    int idx = slot[(s == 0 ? -k : k) + M];
                    if (idx < 0) continue;                  // that order is evanescent
                    acc += wgt[idx]; pick = ord[idx];
                    if (xi < acc) { done = true; break; }
                }
            }
        } else {
            double xi = rng.uniform() * wsum, acc = 0.0; pick = ord[cnt - 1];
            for (int i = 0; i < cnt; ++i) { acc += wgt[i]; if (xi < acc) { pick = ord[i]; break; } }
        }
        if (pickOut) *pickOut = pick;
        Vec3 a = ut + t * ((double)pick * lod);
        Vec3 v = a + nl * std::sqrt(std::max(0.0, 1.0 - dot(a, a)));
        v = normalize(v);
        return Ray{h.p + nl * 1e-6, v};
    }
};
