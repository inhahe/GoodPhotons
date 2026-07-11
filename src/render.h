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
#include <algorithm>
#include <complex>
#include "scene.h"
#include "camera.h"

struct EnergyReport {
    double emitted = 0, absorbed = 0, sensor = 0, escaped = 0, residual = 0;
};

inline double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

// Power-cosine lobe around a mirror direction (rough specular). roughness in
// [0,1]: 0 -> sharp mirror, 1 -> broad. Returns a sampled reflection direction.
inline Vec3 sampleGlossy(const Vec3& mdir, double roughness, Pcg32& rng) {
    double rr = roughness < 1e-3 ? 1e-3 : roughness;
    double e = 2.0 / (rr * rr) - 2.0; if (e < 0) e = 0;
    double u1 = rng.uniform(), u2 = rng.uniform();
    double cosT = std::pow(u1, 1.0 / (e + 1.0));
    double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
    double phi = 2.0 * PI * u2;
    Vec3 t, b; onb(mdir, t, b);
    return normalize(t * (sinT * std::cos(phi)) + b * (sinT * std::sin(phi)) + mdir * cosT);
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

// --- Henyey-Greenstein phase function (participating media) ------------------
// p(cosTheta) normalized so its integral over the sphere is 1. cosTheta is the
// cosine between the photon's propagation direction and the scattered direction;
// g in (-1,1): g>0 forward-peaked, g<0 back-scattering, g=0 isotropic.
inline double hgPhase(double cosTheta, double g) {
    double d = 1.0 + g * g - 2.0 * g * cosTheta;
    if (d < 1e-9) d = 1e-9;
    return (1.0 - g * g) / (4.0 * PI * d * std::sqrt(d));
}

// Sample a scattered direction around the propagation direction `wi` from the HG
// distribution. The sampled cosTheta has mean value g (forward for g>0), so the
// returned direction is importance-sampled proportional to the phase function.
inline Vec3 sampleHG(const Vec3& wi, double g, Pcg32& rng) {
    double u1 = rng.uniform(), u2 = rng.uniform();
    double cosT;
    if (std::fabs(g) < 1e-3) {
        cosT = 1.0 - 2.0 * u1;                          // isotropic
    } else {
        double sq = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1);
        cosT = (1.0 + g * g - sq * sq) / (2.0 * g);
    }
    double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
    double phi = 2.0 * PI * u2;
    Vec3 t, b; onb(wi, t, b);
    return normalize(t * (sinT * std::cos(phi)) + b * (sinT * std::sin(phi)) + wi * cosT);
}

// --- Thin-film interference reflectance (iridescence) ------------------------
// A thin dielectric film (index n1, thickness d nanometres) coats a substrate of
// index n2, with incident medium n0. The beams reflected off the top (n0|n1) and
// bottom (n1|n2) interfaces interfere; the round-trip optical-path phase
//   phi = 4*pi*n1*d*cos(theta1) / lambda
// makes the reflectance oscillate with wavelength AND angle -> structural colour
// (soap bubbles, oil slicks, beetle shells, anodised metal). Returns the
// unpolarised power reflectance R(lambda, theta) in [0,1]; the transmitted
// fraction is 1-R, so the film is lossless. cosI is cos of the incidence angle in
// n0; d and lambda must share units (nanometres here).
//
// This is the exact Airy multiple-beam reflectance for a single film: the full
// geometric sum over every internal round trip between the two interfaces,
// evaluated per polarisation (s and p) and averaged. It is correct at every
// thickness and angle and is naturally bounded in [0,1]. (The earlier two-beam
// form kept only the first two reflected beams; the Airy denominator below
// restores the higher-order beams, sharpening the fringes.)
//
// `k2` is the substrate's extinction coefficient: k2==0 is a transparent
// dielectric substrate (lossless, the transmitted 1-R passes through) and takes
// the exact real-valued path below (bit-identical to the pre-absorption engine).
// k2>0 is an absorbing/metallic substrate (complex index n2+i*k2): the bottom
// interface uses complex Fresnel coefficients, so the interference colour shifts
// and desaturates the way real metal-backed films do, and the transmitted light is
// absorbed (opaque). R is still the reflected power fraction in [0,1].
inline double thinFilmReflectance(double n0, double n1, double n2, double k2,
                                  double d, double cosI, double lambda) {
    cosI = clamp01(std::fabs(cosI));
    double sin0_2 = std::max(0.0, 1.0 - cosI * cosI);
    // Snell into the film: sin(theta1) = (n0/n1) sin(theta0).
    double sin1_2 = (n0 * n0) / (n1 * n1) * sin0_2;
    if (sin1_2 >= 1.0) return 1.0;                       // (n1>=n0 so this won't fire)
    double cos1 = std::sqrt(1.0 - sin1_2);
    if (k2 != 0.0) {
        // Absorbing/metallic substrate: complex index n2c = n2 + i*k2. Work with the
        // admittance q = n*cos(theta) whose transverse-momentum form q = sqrt(n^2 -
        // n0^2 sin^2 theta0) is analytic across the (now complex) substrate. The top
        // interface (n0|n1) stays real; only the bottom (n1|n2c) is complex.
        using cd = std::complex<double>;
        cd n2c(n2, k2);
        double q0 = n0 * cosI, q1 = n1 * cos1;           // real incident/film admittances
        cd q2 = std::sqrt(n2c * n2c - cd(n0 * n0 * sin0_2, 0.0));
        if (q2.imag() < 0.0) q2 = -q2;                   // decaying (absorbing) branch
        // Fresnel amplitude reflections. s-pol uses q; p-pol uses n^2/q, arranged as
        // (nb^2 qa - na^2 qb)/(...) so the real limit matches the rS/rP forms above.
        double r01s = (q0 - q1) / (q0 + q1);
        double r01p = (n1 * n1 * q0 - n0 * n0 * q1) / (n1 * n1 * q0 + n0 * n0 * q1);
        cd r12s = (cd(q1) - q2) / (cd(q1) + q2);
        cd r12p = (n2c * n2c * cd(q1) - cd(n1 * n1) * q2) /
                  (n2c * n2c * cd(q1) + cd(n1 * n1) * q2);
        double phi = (4.0 * PI * n1 * d * cos1) / lambda;
        cd p = std::exp(cd(0.0, phi));                   // round-trip phase factor e^{i*phi}
        auto Rpol = [&](double r01, cd r12) {
            cd num = cd(r01) + r12 * p;
            cd den = cd(1.0) + cd(r01) * r12 * p;
            return clamp01(std::norm(num) / std::norm(den));  // |num/den|^2
        };
        return 0.5 * (Rpol(r01s, r12s) + Rpol(r01p, r12p));
    }
    // Snell into the substrate: sin(theta2) = (n0/n2) sin(theta0).
    double sin2_2 = (n0 * n0) / (n2 * n2) * sin0_2;
    bool tir = sin2_2 >= 1.0;                            // TIR at the n1|n2 interface
    double cos2 = tir ? 0.0 : std::sqrt(1.0 - sin2_2);
    // Fresnel amplitude reflection coefficients (s- and p-polarised) at each face.
    auto rS = [](double na, double ca, double nb, double cb) {
        return (na * ca - nb * cb) / (na * ca + nb * cb);
    };
    auto rP = [](double na, double ca, double nb, double cb) {
        return (nb * ca - na * cb) / (nb * ca + na * cb);
    };
    double r01s = rS(n0, cosI, n1, cos1), r01p = rP(n0, cosI, n1, cos1);
    double r12s = tir ? 1.0 : rS(n1, cos1, n2, cos2);   // |r|=1 amplitude on TIR
    double r12p = tir ? 1.0 : rP(n1, cos1, n2, cos2);
    double phi  = (4.0 * PI * n1 * d * cos1) / lambda;   // interference phase
    double cphi = std::cos(phi);
    // Exact Airy multiple-beam power reflectance per polarisation: the geometric
    // sum over all internal round trips. num is the two-beam result; the den term
    // adds the higher-order beams and keeps R in [0,1] without clamping.
    auto Rpol = [&](double r01, double r12) {
        double num = r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * cphi;
        double den = 1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * cphi;
        return clamp01(den > 1e-12 ? num / den : num);
    };
    return 0.5 * (Rpol(r01s, r12s) + Rpol(r01p, r12p));
}

// Interface (coat) reflectance for a MatType::Layered surface at a hit. Returns the
// unpolarised power reflectance R in [0,1] used as the reflect-vs-enter probability:
//   coatModel 2 (manual): the constant coatSpecular (angle/wavelength independent).
//   coatModel 1 (thinfilm): thin-film Airy R with film index filmIor over the body's
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
        return clamp01(thinFilmReflectance(1.0, m.filmIor, ns, 0.0, thickness, cosI, lambda));
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

// --- Multilayer stack reflectance (Abeles characteristic-matrix method) ------
// Power reflectance of an ordered stack of `nLayers` thin films (per-layer real
// index nL[j], extinction kL[j], thickness dL[j] in nm) between incident medium n0
// and substrate ns+i*ks. This is the exact generalisation of the single-film Airy
// formula to N layers, handling absorbing layers/substrate via complex indices.
// The colour of a Bragg stack (beetle/Morpho/nacre) or a dichroic mirror falls out
// of the multiple-layer interference. cosI is cos of the incidence angle in n0; all
// thicknesses and lambda share units (nanometres). Returns R in [0,1].
//
// Each layer contributes the characteristic matrix
//   M_j = [[cos d_j, i sin d_j / eta_j], [i eta_j sin d_j, cos d_j]]
// with phase thickness d_j = (2 pi / lambda) q_j t_j and transverse admittance
// q_j = sqrt(n_j^2 - n0^2 sin^2 theta0); eta = q (s-pol) or n^2/q (p-pol). The
// stack product M, closed with the substrate admittance, gives r = (eta0 B - C) /
// (eta0 B + C) and R = |r|^2, averaged over the two polarisations.
inline double multilayerReflectance(double n0, double cosI, double lambda,
                                    const double* nL, const double* kL,
                                    const double* dL, int nLayers,
                                    double ns, double ks) {
    using cd = std::complex<double>;
    cosI = clamp01(std::fabs(cosI));
    double sin0_2 = std::max(0.0, 1.0 - cosI * cosI);
    double n0s = n0 * n0 * sin0_2;
    double q0 = n0 * cosI;                         // incident transverse admittance (real)
    auto admit = [](cd nsq, cd q, bool pPol) { return pPol ? nsq / q : q; };
    auto solve = [&](bool pPol) {
        cd M00(1, 0), M01(0, 0), M10(0, 0), M11(1, 0);   // identity
        for (int j = 0; j < nLayers; ++j) {
            cd nj(nL[j], kL[j]);
            cd qj = std::sqrt(nj * nj - cd(n0s, 0.0));
            if (qj.imag() < 0.0) qj = -qj;
            cd eta = admit(nj * nj, qj, pPol);
            cd delta = cd(2.0 * PI * dL[j] / lambda, 0.0) * qj;
            cd c = std::cos(delta), s = std::sin(delta);
            cd L00 = c, L01 = cd(0, 1) * s / eta, L10 = cd(0, 1) * eta * s, L11 = c;
            cd n00 = M00 * L00 + M01 * L10, n01 = M00 * L01 + M01 * L11;
            cd n10 = M10 * L00 + M11 * L10, n11 = M10 * L01 + M11 * L11;
            M00 = n00; M01 = n01; M10 = n10; M11 = n11;
        }
        cd nsub(ns, ks);
        cd qs = std::sqrt(nsub * nsub - cd(n0s, 0.0));
        if (qs.imag() < 0.0) qs = -qs;
        cd etaS = admit(nsub * nsub, qs, pPol);
        cd eta0 = pPol ? cd(n0 * n0 / q0, 0.0) : cd(q0, 0.0);
        cd B = M00 + M01 * etaS;
        cd C = M10 + M11 * etaS;
        cd r = (eta0 * B - C) / (eta0 * B + C);
        return clamp01(std::norm(r));
    };
    return 0.5 * (solve(false) + solve(true));
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

struct Renderer {
    int maxBounce = 32;          // hard safety cap; Russian roulette normally
                                 // terminates paths well before this.
    bool forwardCatch = false;   // model C: catch photons that physically fly through
                                 // the aperture (brute-force oracle, no connect/splat).
    bool lensMode     = false;   // model A: next-event splat through the finite lens
                                 // pupil (physical camera with depth of field).
    bool diffraction = true;     // when false, MatType::Grating collapses to its m=0
                                 // (specular) order — a plain mirror (CLI -diffraction).

    // Model A: map a contact-sensor hit to a pixel and deposit.
    void deposit(const Sensor& s, Film& film, const Vec3& p, double lambda, double beta) const {
        Vec3 rel = p - s.origin;
        double uu = dot(rel, s.uAxis) / dot(s.uAxis, s.uAxis);
        double vv = dot(rel, s.vAxis) / dot(s.vAxis, s.vAxis);
        if (uu < 0 || uu >= 1 || vv < 0 || vv >= 1) return;
        int px = (int)(uu * film.resX), py = (int)(vv * film.resY);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * beta);
    }

    // Model B: connect a surface vertex to the pinhole and splat onto the film.
    // f = rho/pi (Lambertian). The measurement contribution of a surface patch into
    // one pixel is  beta * f * cosSurf / (dist^2 * Omega_pix), where Omega_pix is the
    // solid angle that pixel subtends. This form is projection-general (fisheye and
    // rectilinear alike): for a rectilinear lens Omega_pix = A_pix*cosCam^3, which
    // reproduces the classic G * We = cosSurf*cosCam/dist^2 * 1/(A_pix cosCam^4).
    void connect(const Scene& scene, const Camera& cam, Film& film,
                 const Vec3& p, const Vec3& n, double lambda, double beta, double rho) const {
        Vec3 toCam = cam.eye - p;
        double dist = length(toCam);
        Vec3 wdir = toCam / dist;
        double cosSurf = dot(n, wdir);
        if (cosSurf <= 0) return;                       // camera behind surface
        int px, py; double cosCam, dist2;
        if (!cam.project(p, px, py, cosCam, dist2)) return;
        if (scene.occluded(p + n * 1e-6, wdir, dist - 2e-6)) return;

        double f = rho / PI;
        double omega = cam.pixelSolidAngle(cosCam);
        double contrib = beta * f * cosSurf / (dist2 * omega);
        // Beer-Lambert attenuation of the shadow ray through a global fog.
        if (scene.medium.enabled)
            contrib *= std::exp(-scene.medium.sigmaT(lambda) * dist);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Model B for a VOLUME scattering vertex: connect the collision point to the
    // pinhole. The surface BRDF/cosine is replaced by the phase function and the
    // single-scattering albedo; there is no surface normal. wIn is the photon's
    // propagation direction into the collision.
    //   contrib = beta * albedo * p_HG(cos) / (dist^2 * Omega_pix) * T_fog
    void connectVolume(const Scene& scene, const Camera& cam, Film& film,
                       const Vec3& p, const Vec3& wIn, double lambda, double beta) const {
        Vec3 toCam = cam.eye - p;
        double dist = length(toCam);
        Vec3 wdir = toCam / dist;
        int px, py; double cosCam, dist2;
        if (!cam.project(p, px, py, cosCam, dist2)) return;
        if (scene.occluded(p + wdir * 1e-6, wdir, dist - 2e-6)) return;

        double ph = hgPhase(dot(wIn, wdir), scene.medium.g);
        double Lambda = scene.medium.albedo(lambda);
        double omega = cam.pixelSolidAngle(cosCam);         // projection-general pixel solid angle
        double contrib = beta * Lambda * ph / (dist2 * omega);
        contrib *= std::exp(-scene.medium.sigmaT(lambda) * dist);   // fog transmittance
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
    void connectLens(const Scene& scene, const Camera& cam, Film& film,
                     const Vec3& p, const Vec3& n, double lambda, double beta, double rho,
                     Pcg32& rng) const {
        double R = cam.apertureR;
        double rr = R * std::sqrt(rng.uniform());
        double a  = 2.0 * PI * rng.uniform();
        Vec3 A = cam.eye + cam.u * (rr * std::cos(a)) + cam.v * (rr * std::sin(a));
        Vec3 toA = A - p;
        double dist = length(toA);
        if (dist < 1e-9) return;
        Vec3 wdir = toA / dist;
        double cosSurf = dot(n, wdir);
        if (cosSurf <= 0) return;                        // pupil behind the surface
        double cosLens = -dot(wdir, cam.w);              // cosine at the lens (w faces the scene)
        if (cosLens <= 1e-6) return;                     // not heading toward the film
        int px, py;
        if (!cam.lensImage(A, wdir, px, py)) return;
        if (scene.occluded(p + n * 1e-6, wdir, dist - 2e-6)) return;

        // beta * (rho/pi BRDF) * cosSurf * cosLens / dist^2 * (pi R^2 = 1/pdf_A).
        double contrib = beta * rho * cosSurf * cosLens * (R * R) / (dist * dist);
        if (scene.medium.enabled)
            contrib *= std::exp(-scene.medium.sigmaT(lambda) * dist);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Model A lens splat for a VOLUME scattering vertex (fog). As connectLens but the
    // surface BRDF*cosSurf is replaced by albedo*phase; the phase function carries no
    // 1/pi, so the pupil pdf's pi R^2 stays. wIn is the photon's incoming direction.
    void connectLensVolume(const Scene& scene, const Camera& cam, Film& film,
                           const Vec3& p, const Vec3& wIn, double lambda, double beta,
                           Pcg32& rng) const {
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
        if (scene.occluded(p + wdir * 1e-6, wdir, dist - 2e-6)) return;

        double ph = hgPhase(dot(wIn, wdir), scene.medium.g);
        double Lambda = scene.medium.albedo(lambda);
        double contrib = beta * Lambda * ph * cosLens * (PI * R * R) / (dist * dist);
        contrib *= std::exp(-scene.medium.sigmaT(lambda) * dist);
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Route a camera connection to the pinhole (model B) or the finite lens (model A).
    void camSplat(const Scene& scene, const Camera& cam, Film& film, const Vec3& p,
                  const Vec3& n, double lambda, double beta, double rho, Pcg32& rng) const {
        if (lensMode) connectLens(scene, cam, film, p, n, lambda, beta, rho, rng);
        else          connect(scene, cam, film, p, n, lambda, beta, rho);
    }

    // Splat a surface vertex to every camera target. In model B (the shared-pass case)
    // camSplat -> connect draws no RNG, so the loop is RNG-neutral; with nCam==1 this is
    // exactly the old single-camera call (model A draws its aperture sample once here).
    void camSplatAll(const Scene& scene, const CamTarget* cams, int nCam, const Vec3& p,
                     const Vec3& n, double lambda, double beta, double rho, Pcg32& rng) const {
        for (int c = 0; c < nCam; ++c)
            if (cams[c].cam && cams[c].film)
                camSplat(scene, *cams[c].cam, *cams[c].film, p, n, lambda, beta, rho, rng);
    }

    // Volume (fog) analogue of camSplatAll.
    void camSplatVolumeAll(const Scene& scene, const CamTarget* cams, int nCam, const Vec3& p,
                           const Vec3& wIn, double lambda, double beta, Pcg32& rng) const {
        for (int c = 0; c < nCam; ++c)
            if (cams[c].cam && cams[c].film) {
                if (lensMode) connectLensVolume(scene, *cams[c].cam, *cams[c].film, p, wIn, lambda, beta, rng);
                else          connectVolume(scene, *cams[c].cam, *cams[c].film, p, wIn, lambda, beta);
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
        // --- Emission ---
        // Power-weighted emitter selection: photon selects emitter k with prob
        // power_k/totalPower and carries beta = totalPower, so E[beta over the
        // selection] reproduces each emitter's true power (unbiased). For a single
        // emitter selectEmitter() draws no randomness, keeping the RNG stream (and
        // thus the image) bit-identical to the pre-multi-light engine.
        if (scene.emitters.empty()) return;
        int ei = scene.selectEmitter(rng);
        const Emitter& em = scene.emitters[ei];
        double u1 = rng.uniform(), u2 = rng.uniform();
        Vec3 origin, emitN, dir;
        double spotW = 1.0;                      // spot: p_e/p_u direction reweight (else 1)
        double envPdfW = 0.0;                    // env: solid-angle pdf of the sampled dir
        if (em.shape == EmitterShape::Spot) {
            // Point spot: sample a direction uniformly in the outer cone, then
            // reweight beta by falloff*(Omega_outer/Omega_eff) so the emitted
            // distribution matches the smoothstep intensity profile (analog MC).
            origin = em.origin;
            double ct = em.spotCosOuter + u1 * (1.0 - em.spotCosOuter);
            double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
            double phi = 2.0 * PI * u2;
            Vec3 t, b; onb(em.beamDir, t, b);
            dir = t * (st * std::cos(phi)) + b * (st * std::sin(phi)) + em.beamDir * ct;
            emitN = em.beamDir;
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
                dir = scene.envMap->sample(u1, u2, envPdfW);
            } else {
                double z = 1.0 - 2.0 * u1;
                double sr = std::sqrt(std::max(0.0, 1.0 - z * z));
                double phi = 2.0 * PI * u2;
                dir = Vec3{sr * std::cos(phi), sr * std::sin(phi), z};
            }
            Vec3 t, b; onb(dir, t, b);
            double rd = scene.sceneRadius * std::sqrt(rng.uniform());
            double pd = 2.0 * PI * rng.uniform();
            Vec3 disk = t * (rd * std::cos(pd)) + b * (rd * std::sin(pd));
            origin = scene.sceneCenter - dir * scene.sceneRadius + disk;
            emitN = dir;
        } else {
            em.samplePoint(u1, u2, origin, emitN);   // quad: constant normal; sphere: surface point
            dir = em.collimated ? em.beamDir : cosineHemisphere(emitN, rng);
        }
        double pdfL = 0.0;
        double lambda = em.spd.sample(rng, pdfL);
        if (pdfL <= 0) return;
        // Single emitter: beta = its own power (== old lightEmitIntegral*area*PI).
        // Multiple: beta = totalPower (see selection note above).
        double beta = (scene.emitters.size() == 1) ? em.power : scene.totalPower;
        beta *= spotW;   // exactly 1.0 for non-spot emitters (no bit change)
        // Image env: replace the flat power with the directional estimator. The base
        // beta carries the mean env power; multiply by L(dir,lambda)/(4pi*pdfW*mean)
        // so the photon represents the radiance actually arriving from `dir`. (No-op
        // for a constant env, so those scenes stay bit-identical.)
        if (em.shape == EmitterShape::Env && scene.envMap) {
            double denom = 4.0 * PI * envPdfW * em.spdFn(lambda);
            beta = (denom > 0.0) ? beta * (scene.envMap->radiance(dir, lambda) / denom) : 0.0;
        }
        e.emitted += beta;

        // Direct light -> camera: makes the source itself visible. The Lambertian
        // emitter term is 1/pi, i.e. connect() with rho=1 using the light normal.
        // (Skipped in forward-catch mode; there the aperture test below handles it.)
        // A spot is a point light with no projected area, so it has no such direct
        // term (its cone illuminates surfaces, which then connect to the camera).
        if (nCam > 0 && !forwardCatch &&
            em.shape != EmitterShape::Spot && em.shape != EmitterShape::Env)
            camSplatAll(scene, cams, nCam, origin, emitN, lambda, beta, 1.0, rng);

        Ray ray{origin + dir * 1e-6, dir};
        // Dielectric the photon is currently INSIDE (for Beer-Lambert interior
        // absorption / colored glass), or null in vacuum. Assumes non-nested glass.
        const Material* interior = nullptr;

        for (int bounce = 0; bounce < maxBounce; ++bounce) {
            Hit h = scene.closestHit(ray);
            double dSurf = h.valid ? h.t : 1e30;

            // Homogeneous fog: sample a free-flight collision. If it precedes the
            // surface, the photon interacts in the volume (in-scatter connect,
            // then scatter-or-absorb). Beer-Lambert transmittance is implicit in
            // the exponential free-flight, so beta is unchanged (analog MC).
            double dEvent = dSurf;
            bool mediumEvent = false;
            Vec3 mp;
            if (scene.medium.enabled) {
                double st = scene.medium.sigmaT(lambda);
                if (st > 0.0) {
                    double tMed = -std::log(1.0 - rng.uniform()) / st;
                    if (tMed < dSurf) { dEvent = tMed; mediumEvent = true; mp = ray.o + ray.d * tMed; }
                }
            }

            // Model A perspective catch: if the photon flies through the aperture
            // (nearer than the surface AND any fog collision), it lands on the film.
            if (forwardCatch && nCam > 0 && cams[0].cam && cams[0].film) {
                int px, py;
                if (cams[0].cam->catchPhoton(ray, dEvent, px, py)) {
                    cams[0].film->add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * beta);
                    e.sensor += beta;
                    return;
                }
            }

            // Beer-Lambert attenuation over the free path just travelled inside glass.
            if (interior) {
                double a = interior->absorb(lambda);
                if (a > 0.0) beta *= std::exp(-a * dEvent);
            }

            if (mediumEvent) {
                if (nCam > 0 && !forwardCatch)
                    camSplatVolumeAll(scene, cams, nCam, mp, ray.d, lambda, beta, rng);
                // Scatter (prob albedo) or absorb; throughput unchanged on scatter.
                if (rng.uniform() >= scene.medium.albedo(lambda)) { e.absorbed += beta; return; }
                ray = Ray{mp, sampleHG(ray.d, scene.medium.g, rng)};
                continue;
            }

            if (!h.valid) { e.escaped += beta; return; }

            if (h.sensorId >= 0) {
                if (sensorFilm) deposit(scene.sensor, *sensorFilm, h.p, lambda, beta);
                e.sensor += beta;
                return;
            }

            const Material* matp = &scene.mats[h.matId];
            // Layered (coat over body): split the photon at the interface BEFORE the
            // switch. With prob R (Fresnel/Airy/manual) the coat reflects it as a
            // specular/glossy lobe (lossless, continue); otherwise it enters the body,
            // which picks one lobe among its children (leftover absorbs) and drives the
            // vertex exactly as that child. Energy-consistent per-photon lobe selection.
            if (matp->type == MatType::Layered) {
                const Material& cm = *matp;
                double R = layeredCoatReflectance(scene, cm, h, ray.d, lambda);
                if (rng.uniform() < R) {                    // coat reflection
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, cm, h), rng);
                    if (dot(o, h.n) <= 0) { e.absorbed += beta; return; }
                    ray = Ray{h.p + h.n * 1e-6, o};
                    continue;                               // lossless; beta unchanged
                }
                int child = mixPickChild(cm, rng.uniform());  // body lobe (leftover absorbs)
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
            // Specular/glossy vertices skip the camera connection (delta or
            // near-delta BSDF -> ~zero connection pdf; the SDS limitation).
            switch (m.type) {
                case MatType::Dielectric: {
                    bool entering = dot(ray.d, h.ng) < 0.0;
                    bool transmitted = false;
                    ray = refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted);
                    if (transmitted) interior = entering ? &m : nullptr;  // track medium
                    continue;                       // lossless (absorption applied per-segment)
                }
                case MatType::ThinFilm: {
                    // Iridescent coated interface: specular reflect-or-refract with a
                    // thin-film interference reflectance (structural colour). With an
                    // absorbing substrate the transmitted fraction is absorbed here.
                    Ray nr;
                    if (!thinFilmInterface(scene, m, h, ray.d, lambda, rng, nr)) { e.absorbed += beta; return; }
                    ray = nr;
                    continue;                       // lossless on survival; beta unchanged
                }
                case MatType::Multilayer: {
                    // Multilayer (Bragg / dichroic) stack: specular reflect-or-
                    // refract with the Abeles full-stack reflectance. Absorbing
                    // stacks/substrates absorb the transmitted fraction here.
                    Ray nr;
                    if (!multilayerInterface(m, h, ray.d, lambda, rng, nr)) { e.absorbed += beta; return; }
                    ray = nr;
                    continue;                       // lossless on survival; beta unchanged
                }
                case MatType::Mirror: {
                    double r = clamp01(m.reflect(lambda));
                    // Russian roulette: absorb with prob (1-r), else reflect with
                    // beta unchanged. Unbiased and caps path length naturally.
                    if (rng.uniform() >= r) { e.absorbed += beta; return; }
                    Vec3 o = reflect(ray.d, h.n);
                    ray = Ray{h.p + h.n * 1e-6, o};
                    continue;
                }
                case MatType::Grating: {
                    // Diffraction grating: RR on the overall reflectivity, then
                    // deflect into one stochastically-chosen order (exact grating
                    // equation). Specular per order -> no camera connect (mode P /
                    // the caustic on diffuse walls makes it visible).
                    double r = clamp01(m.reflect(lambda));
                    if (rng.uniform() >= r) { e.absorbed += beta; return; }
                    bool absorbedG;
                    Ray nr = gratingDiffract(m, h, ray.d, lambda, rng, absorbedG);
                    if (absorbedG) { e.absorbed += beta; return; }
                    ray = nr;
                    continue;                       // beta unchanged on the chosen order
                }
                case MatType::HalfMirror: {
                    double r = clamp01(m.reflect(lambda)); // reflect probability
                    if (rng.uniform() < r) {
                        Vec3 o = reflect(ray.d, h.n);
                        ray = Ray{h.p + h.n * 1e-6, o};
                    } else {
                        ray = Ray{h.p + ray.d * 1e-6, ray.d}; // transmit straight
                    }
                    continue;                       // lossless split
                }
                case MatType::Glossy: {
                    double r = clamp01(m.reflect(lambda));
                    // Russian roulette on reflectance (see Mirror).
                    if (rng.uniform() >= r) { e.absorbed += beta; return; }
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                    if (dot(o, h.n) <= 0) { e.absorbed += beta; return; } // below surface
                    ray = Ray{h.p + h.n * 1e-6, o};
                    continue;
                }
                case MatType::Fluorescent: {
                    double rho, aEff; fluoroWeights(m, lambda, rho, aEff);
                    // Model-B connections: the elastic channel splats at the
                    // incoming lambda; the fluorescent channel samples one
                    // lambda' ~ M and splats the glow (albedo aEff*Q) with the
                    // camera-response evaluated at lambda' (Stokes-shifted colour).
                    if (nCam > 0 && !forwardCatch) {
                        camSplatAll(scene, cams, nCam, h.p, h.n, lambda, beta, rho, rng);
                        if (aEff > 0.0 && m.fluoYield > 0.0 && m.fluoEmitSampler.integral > 0.0) {
                            double pf; double lp = m.fluoEmitSampler.sample(rng, pf);   // drawn once, camera-independent
                            camSplatAll(scene, cams, nCam, h.p, h.n, lp, beta, aEff * m.fluoYield, rng);
                        }
                    }
                    FluoroResult fr = fluoroInteract(m, lambda, rng);
                    if (fr.event == FluoroEvent::Absorb) { e.absorbed += beta; return; }
                    lambda = fr.lambdaOut;              // Stokes-shifted on Reemit
                    ray = Ray{h.p + h.n * 1e-6, cosineHemisphere(h.n, rng)};
                    continue;                           // beta unchanged (see above)
                }
                case MatType::Diffuse:
                default: {
                    double rho = clamp01(diffuseReflectance(scene, m, h, lambda));
                    if (nCam > 0 && !forwardCatch) camSplatAll(scene, cams, nCam, h.p, h.n, lambda, beta, rho, rng);
                    // Russian roulette: absorb with prob (1-rho), else scatter
                    // with beta unchanged. Unbiased; average path length ~1/(1-rho)
                    // bounces instead of running to the maxBounce cap.
                    if (rng.uniform() >= rho) { e.absorbed += beta; return; }
                    ray = Ray{h.p + h.n * 1e-6, cosineHemisphere(h.n, rng)};
                    continue;
                }
            }
        }
        e.residual += beta;
    }

    // Dielectric interface: Fresnel-weighted stochastic choice of specular
    // reflection or refraction (Snell), with wavelength-dependent index -> dispersion.
    // Specular reflect-or-refract at a dielectric interface. If `transmitted` is
    // given it reports whether the photon crossed the interface (refracted) vs.
    // reflected/TIR — the caller uses this to track which medium it is now inside
    // (interior absorption). A non-zero `roughness` frosts the interface: BOTH the
    // reflected and refracted lobes are jittered by a power-cosine lobe (rough glass),
    // rejecting samples that would cross to the wrong side so no light leaks through.
    Ray refractOrReflect(const Scene& scene, const Material& m, const Hit& h, const Vec3& d,
                         double lambda, Pcg32& rng, bool* transmitted = nullptr) const {
        double ng = m.ior(lambda);
        bool entering = dot(d, h.ng) < 0.0;
        Vec3 nl = entering ? h.ng : -h.ng;      // normal on the incidence side
        double n1 = entering ? 1.0 : ng;
        double n2 = entering ? ng : 1.0;
        double eta = n1 / n2;
        double cosI = -dot(d, nl);              // > 0
        double sin2t = eta * eta * (1.0 - cosI * cosI);

        Vec3 outDir;
        bool refracted = false;
        if (sin2t > 1.0) {
            outDir = reflect(d, nl);            // total internal reflection
        } else {
            double cosT = std::sqrt(1.0 - sin2t);
            double rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
            double rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
            double R = 0.5 * (rs * rs + rp * rp);
            if (rng.uniform() < R) outDir = reflect(d, nl);
            else { outDir = eta * d + nl * (eta * cosI - cosT); refracted = true; } // Snell
        }
        outDir = normalize(outDir);
        // Frosted glass: jitter the chosen lobe, keeping it on the intended side.
        double rough = materialRoughness(scene, m, h);
        if (rough > 1e-3) {
            Vec3 pert = sampleGlossy(outDir, rough, rng);
            bool ok = refracted ? (dot(pert, nl) < 0.0) : (dot(pert, nl) > 0.0);
            if (ok) outDir = pert;
        }
        if (transmitted) *transmitted = refracted;
        return Ray{h.p + outDir * 1e-6, outDir};
    }

    // Thin-film-coated interface (iridescence). The reflection probability is the
    // thin-film interference reflectance R(lambda, theta) rather than a single
    // Fresnel R. Two substrate regimes, selected by the substrate extinction kappa:
    //   Transparent substrate (kappa==0): structurally identical to refractOrReflect
    //     -- reflect with prob R, else refract into the substrate. Lossless. Purely
    //     specular, so no camera connection and the backward tracer treats it like
    //     Dielectric (modes R/V stay valid).
    //   Absorbing/metallic substrate (kappa>0): reflect the interference fraction R,
    //     else the transmitted light is absorbed -> the photon terminates (opaque
    //     structural colour). There is no refracted ray, so this is one-sided; a hit
    //     from inside an opaque body is simply absorbed.
    // Returns false when the photon is absorbed (caller terminates the path); on
    // true, `out` is the continuation ray.
    bool thinFilmInterface(const Scene& scene, const Material& m, const Hit& h, const Vec3& d,
                           double lambda, Pcg32& rng, Ray& out) const {
        double ns = m.ior(lambda);              // substrate index (spectral -> dispersion)
        double nf = m.filmIor;                  // coating film index
        double ks = m.substrateK(lambda);       // substrate extinction (0 = transparent)
        double thickness = materialFilmThickness(scene, m, h);  // per-hit (map or constant)
        bool entering = dot(d, h.ng) < 0.0;
        Vec3 nl = entering ? h.ng : -h.ng;      // normal on the incidence side
        double cosI = -dot(d, nl);              // > 0

        if (ks > 0.0) {                         // opaque metal-backed film
            if (!entering) return false;        // inside the absorbing substrate: absorbed
            double R = thinFilmReflectance(1.0, nf, ns, ks, thickness, cosI, lambda);
            if (rng.uniform() >= R) return false;               // transmitted -> absorbed
            Vec3 o = normalize(reflect(d, nl));
            out = Ray{h.p + o * 1e-6, o};
            return true;
        }

        double nA = entering ? 1.0 : ns;        // incidence-side medium
        double nB = entering ? ns : 1.0;        // transmission-side medium
        double eta = nA / nB;
        double sin2t = eta * eta * (1.0 - cosI * cosI);
        Vec3 outDir;
        if (sin2t > 1.0) {
            outDir = reflect(d, nl);            // total internal reflection
        } else {
            double cosT = std::sqrt(1.0 - sin2t);
            // Interference reflectance for the actual stack traversed this hit:
            // incidence medium nA, coating nf, transmission medium nB. Reciprocal,
            // so entering and exiting rays see the same R (energy consistent).
            double R = thinFilmReflectance(nA, nf, nB, 0.0, thickness, cosI, lambda);
            if (rng.uniform() < R) outDir = reflect(d, nl);
            else outDir = eta * d + nl * (eta * cosI - cosT); // Snell refraction
        }
        outDir = normalize(outDir);
        out = Ray{h.p + outDir * 1e-6, outDir};
        return true;
    }

    // Multilayer thin-film stack interface (Abeles). The reflection probability is
    // the full-stack reflectance R(lambda, theta). Two regimes, like thinFilm:
    //   Lossless stack (every layer real AND transparent substrate): reflect with
    //     prob R, else refract into the substrate index (dichroic/dielectric-mirror
    //     behaviour -- a wavelength band reflects, the rest transmits).
    //   Any absorption (an absorbing layer OR absorbing substrate): reflect with
    //     prob R, else the transmitted light is absorbed -> the photon terminates
    //     (opaque structural colour: beetle/Morpho on an absorbing base).
    // Returns false when the photon is absorbed (caller terminates the path).
    bool multilayerInterface(const Material& m, const Hit& h, const Vec3& d,
                             double lambda, Pcg32& rng, Ray& out) const {
        double ns = m.ior(lambda);              // substrate index
        double ks = m.substrateK(lambda);       // substrate extinction
        int nL = (int)m.layerN.size();
        bool entering = dot(d, h.ng) < 0.0;
        Vec3 nl = entering ? h.ng : -h.ng;
        double cosI = -dot(d, nl);              // > 0
        bool anyLayerAbsorbs = false;
        for (int j = 0; j < nL; ++j) if (m.layerK[j] != 0.0) { anyLayerAbsorbs = true; break; }
        bool opaque = (ks > 0.0) || anyLayerAbsorbs;

        // For a hit from inside, reverse the stack order so the ray sees the layers
        // in traversal order (substrate-side first). Incident medium is the medium
        // the ray is actually in (air outside, substrate inside for a clear stack).
        if (opaque) {                           // one-sided: reflect-or-absorb
            if (!entering) return false;        // inside the absorbing body: absorbed
            double R = multilayerReflectance(1.0, cosI, lambda,
                                             m.layerN.data(), m.layerK.data(),
                                             m.layerThick.data(), nL, ns, ks);
            if (rng.uniform() >= R) return false;               // transmitted -> absorbed
            Vec3 o = normalize(reflect(d, nl));
            out = Ray{h.p + o * 1e-6, o};
            return true;
        }

        // Lossless: reflect-or-refract into/out of the transparent substrate.
        double nA = entering ? 1.0 : ns;
        double nB = entering ? ns : 1.0;
        double eta = nA / nB;
        double sin2t = eta * eta * (1.0 - cosI * cosI);
        Vec3 outDir;
        if (sin2t > 1.0) {
            outDir = reflect(d, nl);            // total internal reflection
        } else {
            double cosT = std::sqrt(1.0 - sin2t);
            // Evaluate the stack from the incidence side. When exiting (ray inside
            // the substrate) the stack is traversed in reverse and the incident
            // medium is ns; build reversed layer arrays for that case.
            double R;
            if (entering) {
                R = multilayerReflectance(1.0, cosI, lambda,
                                          m.layerN.data(), m.layerK.data(),
                                          m.layerThick.data(), nL, ns, 0.0);
            } else {
                std::vector<double> rn(nL), rk(nL), rd(nL);
                for (int j = 0; j < nL; ++j) { rn[j] = m.layerN[nL-1-j]; rk[j] = m.layerK[nL-1-j]; rd[j] = m.layerThick[nL-1-j]; }
                R = multilayerReflectance(ns, cosI, lambda, rn.data(), rk.data(), rd.data(), nL, 1.0, 0.0);
            }
            if (rng.uniform() < R) outDir = reflect(d, nl);
            else outDir = eta * d + nl * (eta * cosI - cosT); // Snell refraction
        }
        outDir = normalize(outDir);
        out = Ray{h.p + outDir * 1e-6, outDir};
        return true;
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
    Ray gratingDiffract(const Material& m, const Hit& h, const Vec3& din,
                        double lambda, Pcg32& rng, bool& absorbed) const {
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
        for (int mm = -M; mm <= M; ++mm) {
            Vec3 a = ut + t * ((double)mm * lod);
            if (dot(a, a) >= 1.0) continue;                 // evanescent -> excluded
            double w = 1.0 / (1.0 + std::abs(mm));          // idealised efficiency
            ord[cnt] = mm; wgt[cnt] = w; wsum += w; ++cnt;
        }
        if (cnt == 0 || wsum <= 0.0) { absorbed = true; return Ray{}; }
        double xi = rng.uniform() * wsum, acc = 0.0; int pick = ord[cnt - 1];
        for (int i = 0; i < cnt; ++i) { acc += wgt[i]; if (xi < acc) { pick = ord[i]; break; } }
        Vec3 a = ut + t * ((double)pick * lod);
        Vec3 v = a + nl * std::sqrt(std::max(0.0, 1.0 - dot(a, a)));
        v = normalize(v);
        return Ray{h.p + nl * 1e-6, v};
    }
};
