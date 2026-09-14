#pragma once
// ============================================================================================
//  bsdf_eval.h -- the EVALUABLE half of the material model: f(wo, wi) and its sampling density.
// ============================================================================================
//
// `render.h` knows how to SAMPLE every material (sampleGlossy, refractOrReflect, the
// cosine hemisphere). That is all a plain unidirectional tracer needs, and for a long time
// only BDPT needed the other half -- given two directions chosen by someone else, what is the
// BSDF worth and how likely was that direction? -- so `bsdfF` / `bsdfPdf` lived in `bdpt.h`.
//
// They are not BDPT-specific. Any estimator that CONNECTS to a light rather than walking into
// one needs exactly this pair, and GLOSSY-NEE (known-issues.md) needs it in two more places:
// `backward.h`'s next-event estimation, so a rough metal under a small light stops being found
// only by chance, and `photonmap_render.h`'s gather walk for the same reason. Pulling the block
// out here rather than including all of `bdpt.h` keeps `backward.h` free of the beam map, the
// surface merge map and the rest of mode J.
//
// The namespace stays `bdpt` so every existing call site is untouched by the move.
//
// CONVENTION (unchanged, and the thing to read before using these): `wo` and `wi` both point
// AWAY from the surface, `wo` toward where the subpath came from. `bsdfF` returns f such that
// on a sampled direction `f*|cos(wi)|/pdf(wi)` is the throughput factor the sampler applied --
// which for Glossy means `bsdfF` pre-divides by cos(wi) and returns `r*lobe/cos`, and for a
// fiber it pre-divides likewise (see the Hair note below).

#include "scene.h"
#include "render.h"   // sampleGlossy's lobe, materialRoughness, reflectSlot, clamp01, PI

namespace bdpt {

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

// ADJOINT BSDF, for a PARTICLE (light-subpath) vertex: f*(wo,wi) = f(wi,wo).
//
// Call this, not bsdfF, wherever a light-subpath vertex is CONNECTED to something -- the t=1
// splat to the camera, and the light endpoint of an interior connection. Those are exactly the
// sites that already carry `shadingAdjointCorr`, which is the shading-normal half of the same
// Veach rule; this is the BSDF's own half. The two are independent: shadingAdjointCorr is 1 on
// flat geometry, where this is still required.
//
// WHY IT MATTERS HERE. bsdfF pre-divides by cos(wi) (see the CONVENTION note at the top), which
// is self-consistent on a continuation, where the direction evaluated is the direction sampled.
// At a connection it is not: the Glossy lobe factor is symmetric under the swap but the
// denominator is not, so the same physical path got f = r*lobe/cos(camera-side) from the light
// subpath and f = r*lobe/cos(light-side) from the camera subpath. MIS then mixed two estimators
// that disagreed by cos(wo)/cos(wcam) -- measured as mode D reading 15 % DIM with the camera near
// the normal and bright with it grazing, a sign flip no roughness- or solid-angle-scaled error
// could produce.
//
// WHICH FORM IS RIGHT IS SETTLED BY ENERGY. Under uniform illumination L, reflectance r must
// return r*L. Dividing by the incident cosine gives \int r*lobe/cos * cos dw = r; dividing by the
// camera-side cosine does not. A white-furnace test measures mode R -- which builds paths
// camera-side and so lands on the first form -- flat to 0.04 % across roughness 0.2/0.6/0.9.
//
// A reciprocal BSDF (Diffuse's rho/PI, DiffuseTransmit) is unchanged by the swap, so switching a
// site from bsdfF to this is a no-op for them and cannot perturb a diffuse scene.
inline double bsdfFAdjoint(const Material& m, const Vec3& ns, const Vec3& wo, const Vec3& wi,
                           double lambda, const Scene& scene, const Hit* hitForTex) {
    return bsdfF(m, ns, wi, wo, lambda, scene, hitForTex);
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

} // namespace bdpt
