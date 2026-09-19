#pragma once
// ============================================================================================
// A3 — the STOCHASTIC multi-bounce layered BSDF (position-free, after Guo/Hasan/Zhao 2018).
//
// WHAT IS WRONG WITH THE ANALYTIC MODEL, measured (TODO item 5, tools/a3_reference.py). The
// analytic coated body replaces the body's albedo `a` with `a(1-Fdr)/(1-a*Fdr)` -- the sum of the
// series in which light leaving the body is thrown back down by total internal reflection at the
// coat and scatters again. That derivation assumes the body scatters LAMBERTIAN-ly, so that a
// fixed fraction Fdr of its output lands beyond the critical angle. A directional body breaks it:
// a nearly smooth body returns light at the mirror angle, which is INSIDE the critical cone, so
// no TIR series exists at all and the correction should not be applied. Measured in a furnace
// against an explicitly traced coat:
//
//     body roughness   true    analytic   error
//        0.02         0.6288   0.4283    -31.9 %
//        0.10         0.6218   0.4283    -31.1 %
//        0.25         0.5760   0.4283    -25.6 %
//        0.45         0.4954   0.4236    -14.5 %
//
// The analytic albedo is flat in roughness because it depends only on `a`; the truth falls as the
// body roughens toward the Lambertian case the formula was derived for.
//
// THE FIX IS NOT A BETTER FORMULA. There is no closed form: the exit distribution depends on the
// body's whole lobe shape against the critical cone. So the layer stack is SIMULATED -- but
// position-free, because the coat and body are parallel homogeneous planes, so lateral position
// is irrelevant and the random walk lives purely in direction space. A walk is a handful of
// Fresnel decisions and body-BRDF samples: cheap enough to run per shading point.
//
// CONVENTIONS. Local frame, +Z is the shading normal. `wi` and `wo` point AWAY from the surface
// (both have z > 0 for reflection). Internally the walk carries PROPAGATION directions, because
// mixing the two conventions is the classic way to get a sign wrong at a reflection: reflecting a
// propagation vector off the horizontal plane flips only z, while the same event expressed in
// away-from-surface form flips x and y instead.
//
// THE MIS DECISION, made up front because it cannot be retrofitted. `pdf(wo|wi)` has no closed
// form either, and NEE/BDPT/VCM need one for their weights. The resolution: MIS weights only have
// to form a PARTITION OF UNITY -- each strategy must divide by its own true sampling density, but
// the weights themselves may use any positive function provided every strategy uses the SAME one.
// So a deterministic proxy (the analytic lobe, which is cheap and approximately right) is used for
// the weights and NEVER for the estimator. BSDF sampling avoids the problem entirely by returning
// a weight (f*cos/pdf) rather than a density. Suboptimal weights cost variance, not correctness.
// ============================================================================================
#include "linalg.h"

#include <cmath>

namespace layered {

// ---- interface helpers ---------------------------------------------------------------------

// Fresnel reflectance for an unpolarised dielectric. `cosI` is measured against +Z in the medium
// the ray starts in; `eta` is n_transmitted / n_incident. Returns 1 on total internal reflection.
inline double fresnel(double cosI, double eta) {
    cosI = (cosI < 0.0) ? -cosI : cosI;
    if (cosI > 1.0) cosI = 1.0;
    const double s2t = (1.0 - cosI * cosI) / (eta * eta);
    if (s2t >= 1.0) return 1.0;                        // TIR
    const double cosT = std::sqrt(1.0 - s2t);
    const double rs = (cosI - eta * cosT) / (cosI + eta * cosT + 1e-300);
    const double rp = (eta * cosI - cosT) / (eta * cosI + cosT + 1e-300);
    return 0.5 * (rs * rs + rp * rp);
}

// Refract the PROPAGATION direction `p` across the +Z plane. `eta` = n_t / n_i. Returns false on
// total internal reflection. `p` may point either way; the interface normal is taken to oppose it.
inline bool refractProp(const Vec3& p, double eta, Vec3& out) {
    const double ci = -p.z;                            // cos against the normal facing the ray
    const double sign = (ci < 0.0) ? -1.0 : 1.0;       // +Z if travelling down, -Z if up
    const double c = std::fabs(ci);
    const double s2t = (1.0 - c * c) / (eta * eta);
    if (s2t >= 1.0) return false;
    const double ct = std::sqrt(1.0 - s2t);
    // tangential part scales by 1/eta; normal part is rebuilt from ct
    out = Vec3{ p.x / eta, p.y / eta, -sign * ct };
    const double l = length(out);
    if (!(l > 0.0)) return false;
    out = out * (1.0 / l);
    return true;
}

inline Vec3 reflectProp(const Vec3& p) { return Vec3{ p.x, p.y, -p.z }; }

// ---- the body -------------------------------------------------------------------------------
// The body BRDF is supplied by the caller so this header stays independent of the material
// system: `Body` must provide
//     double f(const Vec3& a, const Vec3& b) const;                    // both away-from-surface, z>0
//     bool   sample(const Vec3& a, double u1, double u2, Vec3& b, double& weight) const;
//                                                    // weight = f*cos(b)/pdf, b away-from-surface
//     double pdf(const Vec3& a, const Vec3& b) const;                  // for the f-estimator's MIS

struct Params {
    double eta = 1.5;          // coat index relative to the outside
    double coatAbsorb = 0.0;   // sigma_a inside the coat, 1/m
    double coatDepth = 0.0;    // layer thickness, m (absorption needs both)
    int    maxBounce = 32;     // a white body under TIR can bounce many times; this bounds it
};

// Attenuation across one traversal of the coat at internal direction `p` (propagation).
inline double coatTransmit(const Params& P, const Vec3& p) {
    if (!(P.coatAbsorb > 0.0) || !(P.coatDepth > 0.0)) return 1.0;
    const double c = std::fabs(p.z);
    if (c < 1e-9) return 0.0;
    return std::exp(-P.coatAbsorb * P.coatDepth / c);
}

// ---- sampling -------------------------------------------------------------------------------
// Walks the stack and returns the exit direction and the throughput weight (f*cos/pdf). No
// density is produced, and none is needed: every branch is sampled with its own probability, so
// the Fresnel factors cancel against the branch probabilities and the weight carries only the
// body's own f*cos/pdf. A white body with no absorption therefore returns weight 1 ALWAYS, which
// is the white-furnace identity and the cheapest possible check that the walk is energy-correct.
template <class Body, class Rng>
inline bool sample(const Params& P, const Body& body, const Vec3& wi, Rng& rng,
                   Vec3& wo, double& weight, bool* specularCoat = nullptr) {
    if (specularCoat) *specularCoat = false;
    if (!(wi.z > 0.0)) return false;

    const double F1 = fresnel(wi.z, P.eta);
    if (rng() < F1) {                                   // reflected off the coat: its specular lobe
        wo = Vec3{ -wi.x, -wi.y, wi.z };
        weight = 1.0;
        if (specularCoat) *specularCoat = true;
        return true;
    }

    Vec3 p;                                             // propagation, heading down into the coat
    if (!refractProp(Vec3{ -wi.x, -wi.y, -wi.z }, P.eta, p)) return false;
    double thr = coatTransmit(P, p);
    Vec3 a = Vec3{ -p.x, -p.y, -p.z };                  // body's incident dir, away from body (z>0)

    for (int k = 0; k < P.maxBounce; ++k) {
        Vec3 b;
        double w = 0.0;
        if (!body.sample(a, rng(), rng(), b, w)) return false;
        thr *= w;
        if (!(thr > 0.0)) return false;

        Vec3 up = b;                                    // propagation upward toward the coat
        thr *= coatTransmit(P, up);
        if (!(thr > 0.0)) return false;

        const double F2 = fresnel(up.z, 1.0 / P.eta);   // leaving the dense medium; may be TIR
        if (rng() >= F2) {
            Vec3 outp;
            if (!refractProp(up, 1.0 / P.eta, outp)) return false;   // guarded by F2 < 1
            wo = outp;
            if (!(wo.z > 0.0)) return false;
            weight = thr;
            return true;
        }
        const Vec3 down = reflectProp(up);              // thrown back down by the coat
        thr *= coatTransmit(P, down);
        a = Vec3{ -down.x, -down.y, -down.z };
    }
    return false;                                       // bounce budget exhausted: absorb
}

// ---- evaluation -----------------------------------------------------------------------------
// A stochastic but UNBIASED estimate of f(wi, wo), by walking from wi and connecting to wo at
// every body bounce. The coat's own specular lobe is a delta and contributes nothing here; it is
// reached only by sampling, which is why `sample` reports it via `specularCoat`.
//
// The connection needs the internal direction `c` that refracts to `wo`, the Fresnel transmittance
// there, and the solid-angle Jacobian of that refraction. From Snell, sin_wo = eta*sin_c, so
// dw_wo = eta^2 (cos_c / cos_wo) dw_c. Requiring that the integral of f*cos_wo over the outside
// hemisphere equal the escaping energy then forces
//
//     f(wi, wo) = (1 - F_in) * f_body(a, c) * (1 - F_out) / eta^2
//
// with NO cos_c and NO division by cos_wo. I first wrote it with those two factors instead, on
// the assumption that the two interface crossings' eta^2 cancel; they do not, and the tests said
// so immediately -- the f-integral came out 3.376x the sampled albedo, against eta^2 = 2.25 at
// normal incidence rising with obliquity exactly as a stray cos_c/cos_wo would. The form above
// is also manifestly RECIPROCAL, since (1-F_in) and (1-F_out) swap with a and c, which the
// discarded version was not.
template <class Body, class Rng>
inline double f(const Params& P, const Body& body, const Vec3& wi, const Vec3& wo, Rng& rng) {
    if (!(wi.z > 0.0) || !(wo.z > 0.0)) return 0.0;

    Vec3 cProp;                                         // internal propagation that exits as wo
    if (!refractProp(Vec3{ -wo.x, -wo.y, -wo.z }, P.eta, cProp)) return 0.0;
    const Vec3 c = Vec3{ -cProp.x, -cProp.y, -cProp.z };            // away-from-body, z > 0
    const double F1 = fresnel(wi.z, P.eta);
    const double Fo = fresnel(c.z, 1.0 / P.eta);
    const double tOut = (1.0 - Fo) * coatTransmit(P, cProp);
    if (!(tOut > 0.0)) return 0.0;

    Vec3 p;
    if (!refractProp(Vec3{ -wi.x, -wi.y, -wi.z }, P.eta, p)) return 0.0;
    double thr = (1.0 - F1) * coatTransmit(P, p);
    Vec3 a = Vec3{ -p.x, -p.y, -p.z };

    const double invEta2 = 1.0 / (P.eta * P.eta);
    double acc = 0.0;
    for (int k = 0; k < P.maxBounce; ++k) {
        // connect: scatter from `a` straight out along `c`
        const double fb = body.f(a, c);
        if (fb > 0.0) acc += thr * fb * tOut * invEta2;

        // ... and continue the walk for the next order
        Vec3 b;
        double w = 0.0;
        if (!body.sample(a, rng(), rng(), b, w)) break;
        thr *= w;
        if (!(thr > 0.0)) break;
        Vec3 up = b;
        thr *= coatTransmit(P, up);
        const double F2 = fresnel(up.z, 1.0 / P.eta);
        if (rng() >= F2) break;                         // this walk escaped; later orders are gone
        const Vec3 down = reflectProp(up);
        thr *= coatTransmit(P, down);
        a = Vec3{ -down.x, -down.y, -down.z };
        if (!(thr > 0.0)) break;
    }
    return acc;
}

// ---- escape moments: the generalisation that actually fixes the measured defect -------------
// The analytic model is `a_eff = a(1-F)/(1-aF)`, and expanding it shows what it really assumes:
//
//     a(1-F)/(1-aF)  =  SUM_k  a^(k+1) * F^k (1-F)
//
// i.e. the number of body bounces before escape is GEOMETRIC -- each bounce escapes with the same
// probability (1-F), independent of the last. That is exactly true for a Lambertian body, whose
// output is cosine-distributed however it was lit, and it is exactly what fails for a directional
// one: a near-smooth body returns light at the mirror angle, inside the critical cone, so it
// escapes on the FIRST attempt essentially always and the higher terms vanish.
//
// So keep the expansion and drop the assumption. `escapeMoments` measures the real distribution
// P_k -- the probability of escaping after exactly k+1 body interactions -- by running the
// position-free walk. The crucial property is that P_k depends on the coat index, the incidence
// and the body's LOBE SHAPE, but NOT on the body's albedo: albedo only ever multiplies, once per
// bounce. So P_k is computed once per (material, incidence) and then serves every wavelength and
// every albedo for free, which is what makes this affordable at all.
//
// `tIo` / `tRt` fold in an absorbing layer exactly as the analytic version does: one in-and-out
// traversal always, plus one round trip per extra bounce.
template <class Body, class Rng>
inline void escapeMoments(const Params& P, const Body& body, double cosI, int K,
                          double* out, int nWalks, Rng& rng) {
    for (int k = 0; k < K; ++k) out[k] = 0.0;
    if (!(cosI > 0.0) || K <= 0 || nWalks <= 0) return;
    const double st = std::sqrt(cosI * cosI < 1.0 ? 1.0 - cosI * cosI : 0.0);
    const Vec3 wi{ st, 0.0, cosI };

    for (int i = 0; i < nWalks; ++i) {
        const double F1 = fresnel(wi.z, P.eta);
        if (rng() < F1) continue;                 // the coat's own specular lobe is not the body's
        Vec3 p;
        if (!refractProp(Vec3{ -wi.x, -wi.y, -wi.z }, P.eta, p)) continue;
        Vec3 a = Vec3{ -p.x, -p.y, -p.z };
        for (int k = 0; k < K; ++k) {
            Vec3 b;
            double w = 0.0;
            if (!body.sample(a, rng(), rng(), b, w)) break;   // the ALBEDO in w is deliberately
            //                                                   ignored: these are pure geometry
            const double F2 = fresnel(b.z, 1.0 / P.eta);
            if (rng() >= F2) { out[k] += 1.0; break; }        // escaped after k+1 interactions
            const Vec3 down = reflectProp(b);
            a = Vec3{ -down.x, -down.y, -down.z };
        }
    }
    for (int k = 0; k < K; ++k) out[k] /= (double)nWalks;
}

// a_eff from the measured moments. Reduces to `coatedAlbedo(a, F)` exactly when P is geometric,
// which is the reduction the self-test asserts against the shipped analytic function.
inline double albedoFromMoments(double a, const double* P, int K,
                                double tIo = 1.0, double tRt = 1.0) {
    if (!(a > 0.0)) return 0.0;
    double acc = 0.0, ak = a * tIo;
    for (int k = 0; k < K; ++k) {
        acc += ak * P[k];
        ak *= a * tRt;
    }
    return acc;
}

}  // namespace layered
