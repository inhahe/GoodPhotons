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
#include "photonmap.h"
#include "medium_stack.h"
#include "grin.h"     // shared gradient-index (GRIN) Eikonal marcher
#include "hero.h"     // hero-wavelength spectral sampling (kHeroC)

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
    bool forwardCatch = false;   // model C: catch photons that physically fly through
                                 // the aperture (brute-force oracle, no connect/splat).
    bool lensMode     = false;   // model A: next-event splat through the finite lens
                                 // pupil (physical camera with depth of field).
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

    // Photon-map deposit (ROADMAP item 1 / mode M). When non-null, every diffuse-family
    // surface vertex ALSO appends a Photon record here (view-independent radiance cache).
    // A photon pass runs with nCam==0 (no camera splat) + photonDeposit set: paths bounce
    // and deposit, but energy goes into the map instead of onto a sensor. Null in modes
    // A/B/C, so their splat behaviour is byte-for-byte unchanged.
    std::vector<Photon>* photonDeposit = nullptr;

    // Append a photon record at a diffuse/translucent vertex (no-op when the map is off).
    void depositPhoton(const Vec3& p, const Vec3& wtravel, const Vec3& n,
                       double lambda, double beta) const {
        if (!photonDeposit) return;
        photonDeposit->push_back(Photon{p, -wtravel, n, (float)beta, (float)lambda});
    }

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
    // pre-heterogeneous engine); a density field switches to delta / ratio tracking
    // with the majorant sigma_max = sigmaT(lambda) * densityMax.

    // Sample the next real collision along (o,dir) within [0,dMax]. Returns true and
    // sets tHit at a real scattering/absorption event; false if the photon reaches
    // dMax first. Delta (Woodcock) tracking for a heterogeneous medium: candidate
    // collisions at rate sigma_max, accepted as real with prob sigmaT(x)/sigma_max
    // (a rejected "null collision" just continues) — unbiased, throughput unchanged.
    bool sampleMediumCollision(const Medium& med, const Vec3& o, const Vec3& dir,
                               double dMax, double lambda, Pcg32& rng, double& tHit) const {
        double stBase = med.sigmaT(lambda);
        if (stBase <= 0.0) return false;
        double ta, tb;
        if (!med.clipToBounds(o, dir, 0.0, dMax, ta, tb)) return false;
        if (!med.heterogeneous()) {                       // exact free-flight (one draw)
            double t = ta - std::log(1.0 - rng.uniform()) / stBase;
            if (t < tb) { tHit = t; return true; }
            return false;
        }
        double sigMax = stBase * med.densityMax;
        if (sigMax <= 0.0) return false;
        double t = ta;
        for (;;) {
            t += -std::log(1.0 - rng.uniform()) / sigMax;
            if (t >= tb) return false;
            double sigT = stBase * med.densityAt(o + dir * t);
            if (rng.uniform() * sigMax < sigT) { tHit = t; return true; }  // real collision
        }                                                                 // else null collision
    }

    // Unbiased transmittance along [o, o+dir*dist] through the medium. Exact exp for a
    // homogeneous medium (clipped to its bound); ratio tracking otherwise (candidate
    // collisions at rate sigma_max, each scaling the estimate by 1 - sigmaT(x)/sigma_max).
    double mediumTransmittance(const Medium& med, const Vec3& o, const Vec3& dir,
                               double dist, double lambda, Pcg32& rng) const {
        double stBase = med.sigmaT(lambda);
        if (stBase <= 0.0) return 1.0;
        double ta, tb;
        if (!med.clipToBounds(o, dir, 0.0, dist, ta, tb)) return 1.0;   // ray never enters fog
        if (!med.heterogeneous())
            return std::exp(-stBase * (tb - ta));
        double sigMax = stBase * med.densityMax;
        if (sigMax <= 0.0) return 1.0;
        double Tr = 1.0, t = ta;
        for (;;) {
            t += -std::log(1.0 - rng.uniform()) / sigMax;
            if (t >= tb) break;
            double sigT = stBase * med.densityAt(o + dir * t);
            Tr *= 1.0 - sigT / sigMax;
        }
        return Tr;
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

    // Earliest real collision across all media within [0,dMax]. On a hit, `tHit` is the
    // distance and `whichMed` the index of the scattering medium. false if none.
    bool sampleMediaCollision(const std::vector<Medium>& media, const Vec3& o,
                              const Vec3& dir, double dMax, double lambda, Pcg32& rng,
                              double& tHit, int& whichMed) const {
        double best = dMax; int which = -1;
        for (int i = 0; i < (int)media.size(); ++i) {
            double t;
            if (sampleMediumCollision(media[i], o, dir, dMax, lambda, rng, t) && t < best) {
                best = t; which = i;
            }
        }
        if (which < 0) return false;
        tHit = best; whichMed = which; return true;
    }

    // Combined transmittance through all media = product of per-medium transmittances.
    double mediaTransmittance(const std::vector<Medium>& media, const Vec3& o,
                              const Vec3& dir, double dist, double lambda, Pcg32& rng) const {
        double Tr = 1.0;
        for (const Medium& m : media) {
            Tr *= mediumTransmittance(m, o, dir, dist, lambda, rng);
            if (Tr <= 0.0) break;
        }
        return Tr;
    }

    // Wavelength-INDEPENDENT part of the mode-B pinhole connection: project the surface
    // vertex to the film, reject/soften across the horizons, and occlusion-test the
    // shadow ray. On success returns true and fills `g` with the pixel and the geometry
    // factors, from which the caller forms the per-λ contribution as
    //   beta * (rho/PI) * cosSurf * corr / denom * stG  (× media transmittance).
    // Sharing this across the C hero wavelengths (one BVH occlusion test, one projection)
    // is the whole point of hero splatting; the scalar connect() below reuses it too so
    // its float ordering — and thus mode B — stays bit-identical.
    struct ConnGeom {
        int px = 0, py = 0;
        double cosSurf = 0, corr = 0, denom = 0, stG = 0, dist = 0;
        Vec3 wdir;
    };
    bool connectGeom(const Scene& scene, const Camera& cam, const Vec3& p, const Vec3& n,
                     const Vec3& ng, const Vec3& wi, ConnGeom& g) const {
        Vec3 toCam = cam.eye - p;
        g.dist = length(toCam);
        g.wdir = toCam / g.dist;
        g.cosSurf = dot(n, g.wdir);
        // Reject connections below the shading horizon; soften across the GEOMETRIC
        // horizon (`ng` is the geometric normal on the shading side). A smoothed shading
        // normal must not splat a vertex whose true geometry faces away from the camera,
        // but a hard cutoff there carves facet slivers at the terminator, so ramp it
        // smoothly (Chiang 2019). No-op for flat tris / analytic spheres, where ng == n
        // (stG == 1), so those scenes stay bit-identical.
        if (g.cosSurf <= 0) return false;               // camera behind shading surface
        g.stG = shadowTerminatorG(g.wdir, n, ng);
        if (g.stG <= 0.0) return false;                 // camera behind true geometry: hard cutoff
        double cosCam, dist2;
        if (!cam.project(p, g.px, g.py, cosCam, dist2)) return false;
        if (scene.occluded(p + ng * 1e-6, g.wdir, g.dist - 2e-6)) return false;
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
    void connect(const Scene& scene, const Camera& cam, Film& film,
                 const Vec3& p, const Vec3& n, const Vec3& ng, const Vec3& wi,
                 double lambda, double beta, double rho, Pcg32& rng) const {
        ConnGeom g;
        if (!connectGeom(scene, cam, p, n, ng, wi, g)) return;
        double f = rho / PI;
        double contrib = beta * f * g.cosSurf * g.corr / g.denom * g.stG;
        // Attenuation of the shadow ray through the fog (Beer-Lambert; ratio tracking
        // for a heterogeneous medium, exact exp for a homogeneous one; product over media).
        if (!scene.media.empty())
            contrib *= mediaTransmittance(scene.media, p, g.wdir, g.dist, lambda, rng);
        film.add(g.px, g.py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Hero-wavelength mode-B connection: one shared connectGeom, then splat each of the
    // `nUp` live wavelengths with its own throughput beta[i] and reflectance rho[i]. The
    // per-λ contribution uses the SAME float ordering as connect() above.
    void connectHero(const Scene& scene, const Camera& cam, Film& film,
                     const Vec3& p, const Vec3& n, const Vec3& ng, const Vec3& wi,
                     const double* lam, const double* beta, const double* rho, int nUp,
                     Pcg32& rng) const {
        ConnGeom g;
        if (!connectGeom(scene, cam, p, n, ng, wi, g)) return;
        for (int i = 0; i < nUp; ++i) {
            double f = rho[i] / PI;
            double contrib = beta[i] * f * g.cosSurf * g.corr / g.denom * g.stG;
            if (!scene.media.empty())
                contrib *= mediaTransmittance(scene.media, p, g.wdir, g.dist, lam[i], rng);
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
        if (scene.occluded(p + wdir * 1e-6, wdir, dist - 2e-6)) return;

        double ph = med.phaseValue(dot(wIn, wdir), lambda); // scattering medium's phase (HG or rainbow)
        double Lambda = med.albedo(lambda);
        double omega = cam.pixelSolidAngle(cosCam);         // projection-general pixel solid angle
        double contrib = beta * Lambda * ph / (dist2 * omega);
        contrib *= mediaTransmittance(scene.media, p, wdir, dist, lambda, rng);   // fog transmittance (all media)
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
                     const Vec3& p, const Vec3& n, const Vec3& ng, const Vec3& wi,
                     double lambda, double beta, double rho, Pcg32& rng) const {
        double R = cam.apertureR;
        double rr = R * std::sqrt(rng.uniform());
        double a  = 2.0 * PI * rng.uniform();
        Vec3 A = cam.eye + cam.u * (rr * std::cos(a)) + cam.v * (rr * std::sin(a));
        Vec3 toA = A - p;
        double dist = length(toA);
        if (dist < 1e-9) return;
        Vec3 wdir = toA / dist;
        double cosSurf = dot(n, wdir);
        // Below the shading horizon reject; soften across the geometric horizon (see
        // connect()): no-op for flat/sphere (stG == 1).
        if (cosSurf <= 0) return;                        // pupil behind the shading surface
        double stG = shadowTerminatorG(wdir, n, ng);
        if (stG <= 0.0) return;                          // pupil behind true geometry: hard cutoff
        double cosLens = -dot(wdir, cam.w);              // cosine at the lens (w faces the scene)
        if (cosLens <= 1e-6) return;                     // not heading toward the film
        int px, py;
        if (!cam.lensImage(A, wdir, px, py)) return;
        if (scene.occluded(p + ng * 1e-6, wdir, dist - 2e-6)) return;

        // beta * (rho/pi BRDF) * cosSurf * cosLens / dist^2 * (pi R^2 = 1/pdf_A).
        // cosSurf carries the Veach shading-normal adjoint correction (see connect()).
        double corr = shadingAdjointCorr(wi, wdir, n, ng);
        double contrib = beta * rho * cosSurf * corr * cosLens * (R * R) / (dist * dist) * stG;
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
            contrib *= mediaTransmittance(scene.media, p, wdir, dist, lambda, rng);
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
        if (cosSurf <= 0) return;                        // pupil behind the shading surface
        double stG = shadowTerminatorG(wdir, n, ng);
        if (stG <= 0.0) return;                          // pupil behind true geometry
        double cosLens = -dot(wdir, cam.w);
        if (cosLens <= 1e-6) return;                     // not heading toward the film
        int px, py;
        if (!cam.lensImage(A, wdir, px, py)) return;
        if (scene.occluded(p + ng * 1e-6, wdir, dist - 2e-6)) return;
        double corr = shadingAdjointCorr(wi, wdir, n, ng);
        double cellNorm = 1.0 / (cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
        for (int i = 0; i < nUp; ++i) {
            double contrib = beta[i] * rho[i] * cosSurf * corr * cosLens * (R * R) / (dist * dist) * stG;
            contrib *= cellNorm;
            if (!scene.media.empty())
                contrib *= mediaTransmittance(scene.media, p, wdir, dist, lam[i], rng);
            film.add(px, py, Vec3(cieX(lam[i]), cieY(lam[i]), cieZ(lam[i])) * contrib);
        }
    }

    // Model A lens splat for a VOLUME scattering vertex (fog). As connectLens but the
    // surface BRDF*cosSurf is replaced by albedo*phase; the phase function carries no
    // 1/pi, so the pupil pdf's pi R^2 stays. wIn is the photon's incoming direction.
    void connectLensVolume(const Scene& scene, const Medium& med, const Camera& cam, Film& film,
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

        double ph = med.phaseValue(dot(wIn, wdir), lambda); // scattering medium's phase (HG or rainbow)
        double Lambda = med.albedo(lambda);
        double contrib = beta * Lambda * ph * cosLens * (PI * R * R) / (dist * dist);
        // Same flux->film-irradiance normaliser as connectLens (see there): divide the
        // pupil flux deposited in the cell by the physical cell area so a fog vertex
        // matches B's absolute scale in absolute-EV modes (per-camera constant; auto-
        // exposed scenes unaffected).
        contrib *= 1.0 / (cam.pixelPlaneArea() * cam.filmDist * cam.filmDist);
        contrib *= mediaTransmittance(scene.media, p, wdir, dist, lambda, rng);   // all media
        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
    }

    // Route a camera connection to the pinhole (model B) or the finite lens (model A).
    void camSplat(const Scene& scene, const Camera& cam, Film& film, const Vec3& p,
                  const Vec3& n, const Vec3& ng, const Vec3& wi, double lambda, double beta,
                  double rho, Pcg32& rng) const {
        if (lensMode) connectLens(scene, cam, film, p, n, ng, wi, lambda, beta, rho, rng);
        else          connect(scene, cam, film, p, n, ng, wi, lambda, beta, rho, rng);
    }

    // Splat a surface vertex to every camera target. In model B (the shared-pass case)
    // camSplat -> connect draws no RNG, so the loop is RNG-neutral; with nCam==1 this is
    // exactly the old single-camera call (model A draws its aperture sample once here).
    void camSplatAll(const Scene& scene, const CamTarget* cams, int nCam, const Vec3& p,
                     const Vec3& n, const Vec3& ng, const Vec3& wi, double lambda, double beta,
                     double rho, Pcg32& rng) const {
        for (int c = 0; c < nCam; ++c)
            if (cams[c].cam && cams[c].film)
                camSplat(scene, *cams[c].cam, *cams[c].film, p, n, ng, wi, lambda, beta, rho, rng);
    }

    // Hero-wavelength routing (mode A finite lens or mode B pinhole), splatting all `nUp`
    // live wavelengths through one shared connection. Mode B draws no RNG (RNG-neutral
    // over the camera loop); mode A draws its single aperture sample once per camera.
    void camSplatHero(const Scene& scene, const Camera& cam, Film& film, const Vec3& p,
                      const Vec3& n, const Vec3& ng, const Vec3& wi, const double* lam,
                      const double* beta, const double* rho, int nUp, Pcg32& rng) const {
        if (lensMode) connectLensHero(scene, cam, film, p, n, ng, wi, lam, beta, rho, nUp, rng);
        else          connectHero(scene, cam, film, p, n, ng, wi, lam, beta, rho, nUp, rng);
    }
    void camSplatAllHero(const Scene& scene, const CamTarget* cams, int nCam, const Vec3& p,
                         const Vec3& n, const Vec3& ng, const Vec3& wi, const double* lam,
                         const double* beta, const double* rho, int nUp, Pcg32& rng) const {
        for (int c = 0; c < nCam; ++c)
            if (cams[c].cam && cams[c].film)
                camSplatHero(scene, *cams[c].cam, *cams[c].film, p, n, ng, wi, lam, beta, rho, nUp, rng);
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
        // Throughput at the vertex for a connection leaving toward `wP` (unit, toward
        // the sphere). Returns <0 to signal "reject" (camera-side behind a surface).
        double term(const Vec3& wP) const {
            if (volume) return weight * hgPhase(dot(wIn, wP), g);
            double cosSurf = dot(np, wP);
            return cosSurf <= 0.0 ? -1.0 : (weight / PI) * cosSurf;
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
            if (scene.occluded(ch.P1 + wE * 1e-6, wE, dE - 2e-6)) continue;

            // Fog transmittance on the two outer (vacuum-side) segments only; the
            // interior segment is solid glass (its absorption is the Beer-Lambert above).
            if (!scene.media.empty()) {
                contrib *= mediaTransmittance(scene.media, p,     wP, dP2, lambda, rng);
                contrib *= mediaTransmittance(scene.media, ch.P1, wE, dE,  lambda, rng);
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
                contrib *= mediaTransmittance(scene.media, p, wP, dP, lambda, rng);

            film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * contrib);
        }
    }

    // Splat vertex p to every camera through every smooth dielectric sphere (the
    // refracted image of p). Mode B (pinhole) only; draws no aperture RNG.
    void camSpecularSplatAllVtx(const Scene& scene, const CamTarget* cams, int nCam,
                                const Vec3& p, const SpecVtx& vt, double lambda,
                                double beta, Pcg32& rng) const {
        if (lensMode || forwardCatch) return;               // finite-lens/catch not supported
        for (const Sphere& S : scene.spheres) {
            const Material& gm = scene.mats[S.matId];
            if (gm.type != MatType::Dielectric) continue;
            double ng = gm.ior(lambda);
            for (int c = 0; c < nCam; ++c)
                if (cams[c].cam && cams[c].film)
                    connectSpecularSphere(scene, *cams[c].cam, *cams[c].film, S, gm, ng,
                                          p, vt, lambda, beta, rng);
        }
    }
    // Surface vertex: refract the Lambertian reflection of p through every glass sphere.
    void camSpecularSplatAll(const Scene& scene, const CamTarget* cams, int nCam,
                             const Vec3& p, const Vec3& n, double lambda, double beta,
                             double rho, Pcg32& rng) const {
        SpecVtx vt; vt.volume = false; vt.np = n; vt.weight = rho;
        camSpecularSplatAllVtx(scene, cams, nCam, p, vt, lambda, beta, rng);
    }
    // Hero variant: the sphere refraction is DISPERSIVE (glass IOR varies per λ), so it
    // cannot share one geometry — each wavelength traces its own refracted image, which
    // is exactly what makes the glass-sphere caustic-image chromatically dispersed. Draws
    // no RNG (mode B), so looping λ leaves the stream untouched.
    void camSpecularSplatAllHero(const Scene& scene, const CamTarget* cams, int nCam,
                                 const Vec3& p, const Vec3& n, const double* lam,
                                 const double* beta, const double* rho, int nUp,
                                 Pcg32& rng) const {
        for (int i = 0; i < nUp; ++i) {
            SpecVtx vt; vt.volume = false; vt.np = n; vt.weight = rho[i];
            camSpecularSplatAllVtx(scene, cams, nCam, p, vt, lam[i], beta[i], rng);
        }
    }
    // Volume vertex: refract the fog in-scatter at p through every glass sphere, so the
    // glowing haze itself bends through the glass the camera flies through.
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
                if (lensMode) connectLensVolume(scene, med, *cams[c].cam, *cams[c].film, p, wIn, lambda, beta, rng);
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
                                EnergyReport& e) const {
        switch (m.type) {
            case MatType::Dielectric: {
                // Nested-dielectric PRIORITY resolution (Schmidt & Budge 2002): the
                // exterior IOR is the medium the photon is currently inside (the
                // highest-priority stack entry), not a hardcoded 1.0, so glass inside
                // water refracts 1.33<->1.52. Where dielectrics overlap the higher
                // `priority` wins and the lower one's boundary is suppressed (the
                // photon passes straight through). SAFE FALLBACK: the priority rule
                // applies only when BOTH sides carry an explicit priority (air always
                // counts, IOR 1.0); otherwise this degrades to the old flat
                // air<->glass model so priority-free scenes are bit-identical.
                bool entering = dot(ray.d, h.ng) < 0.0;
                const int mi = (int)(&m - scene.mats.data());   // true index (Mix/Layered aware)
                const int pr = m.priority;               // INT_MIN if unset

                if (entering) {
                    const int outMat = stk.topMat();     // -1 == air
                    const int outPri = stk.topPri();     // INT_MIN == air
                    const bool ranked = m.hasPriority() &&
                        (stk.empty() || (outMat >= 0 && scene.mats[outMat].hasPriority()));
                    if (ranked && !stk.empty() && pr <= outPri) {   // suppressed inner surface
                        stk.push(mi, pr);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                        return true;
                    }
                    const double extIor = (ranked && outMat >= 0)
                        ? scene.mats[outMat].ior(lambda) : 1.0;
                    bool transmitted = false;
                    ray = refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                    if (transmitted) stk.push(mi, pr);
                    return true;                 // lossless (absorption applied per-segment)
                } else {
                    MediumStack after = stk;
                    after.popMat(mi);
                    const int newMat = after.topMat();   // -1 == air underneath
                    const int newPri = after.topPri();
                    const bool ranked = m.hasPriority() &&
                        (after.empty() || (newMat >= 0 && scene.mats[newMat].hasPriority()));
                    if (ranked && newMat >= 0 && pr <= newPri) {    // suppressed: still enclosed
                        stk.popMat(mi);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                        return true;
                    }
                    const double extIor = (ranked && newMat >= 0)
                        ? scene.mats[newMat].ior(lambda) : 1.0;
                    bool transmitted = false;
                    ray = refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                    if (transmitted) stk.popMat(mi);      // TIR stays inside mi
                    return true;                 // lossless (absorption applied per-segment)
                }
            }
            case MatType::ThinFilm: {
                // Iridescent coated interface: specular reflect-or-refract with a
                // thin-film interference reflectance (structural colour). With an
                // absorbing substrate the transmitted fraction is absorbed here.
                Ray nr;
                if (!thinFilmInterface(scene, m, h, ray.d, lambda, rng, nr)) { e.absorbed += beta; return false; }
                ray = nr;
                return true;                     // lossless on survival; beta unchanged
            }
            case MatType::Multilayer: {
                // Multilayer (Bragg / dichroic) stack: specular reflect-or-
                // refract with the Abeles full-stack reflectance. Absorbing
                // stacks/substrates absorb the transmitted fraction here.
                Ray nr;
                if (!multilayerInterface(m, h, ray.d, lambda, rng, nr)) { e.absorbed += beta; return false; }
                ray = nr;
                return true;                     // lossless on survival; beta unchanged
            }
            case MatType::Mirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                // Russian roulette: absorb with prob (1-r), else reflect with
                // beta unchanged. Unbiased and caps path length naturally.
                if (rng.uniform() >= r) { e.absorbed += beta; return false; }
                Vec3 o = reflect(ray.d, h.n);
                ray = Ray{h.p + h.n * 1e-6, o};
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
                    ray = Ray{h.p + h.n * 1e-6, o};
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
                double t = clamp01(m.transmit(lambda));
                if (rng.uniform() >= t) { e.absorbed += beta; return false; }
                ray = Ray{h.p + ray.d * 1e-6, ray.d}; // transmit straight, unchanged
                return true;
            }
            case MatType::Glossy: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                // Russian roulette on reflectance (see Mirror).
                if (rng.uniform() >= r) { e.absorbed += beta; return false; }
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0) { e.absorbed += beta; return false; } // below surface
                ray = Ray{h.p + h.n * 1e-6, o};
                return true;
            }
            case MatType::Fluorescent: {
                double rho, aEff; fluoroWeights(m, lambda, rho, aEff);
                Vec3 ngo = orientedGeoN(h);
                Vec3 wi = Vec3{-ray.d.x, -ray.d.y, -ray.d.z};   // toward the previous (light-side) vertex
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
                Vec3 wo = cosineHemisphere(h.n, rng);
                beta *= shadingAdjointCorr(wi, wo, h.n, ngo);   // Veach adjoint (1 when Ns==Ng)
                ray = Ray{h.p + h.n * 1e-6, wo};
                return true;                        // beta otherwise unchanged (see above)
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
            em.shape != EmitterShape::Spot && em.shape != EmitterShape::Env) {
            camSplatAll(scene, cams, nCam, origin, emitN, emitN, emitN, lambda, beta, 1.0, rng);
            camSpecularSplatAll(scene, cams, nCam, origin, emitN, lambda, beta, 1.0, rng);
        }

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

        for (int bounce = 0; bounce < maxBounce; ++bounce) {
            // GRIN curved marching pre-pass (does not consume a bounce): advance the ray
            // through any gradient-index region before the straight-ray hit test.
            if (grinAny) grin::march(scene, ray);

            Hit h = scene.closestHit(ray);
            double dSurf = h.valid ? h.t : 1e30;

            // Homogeneous fog: sample a free-flight collision. If it precedes the
            // surface, the photon interacts in the volume (in-scatter connect,
            // then scatter-or-absorb). Beer-Lambert transmittance is implicit in
            // the exponential free-flight, so beta is unchanged (analog MC).
            double dEvent = dSurf;
            bool mediumEvent = false;
            int scatterMed = -1;   // which medium scattered (index into scene.media)
            Vec3 mp;
            // In -beams (photon-beams single-scatter) mode the photon does NOT redirect in
            // the medium — it crosses in a straight beam and each camera gathers single-
            // scatter independently below — so skip the analog collision sampling here.
            if (!scene.media.empty() && !doBeamGather) {
                double tMed; int which;
                if (sampleMediaCollision(scene.media, ray.o, ray.d, dSurf, lambda, rng, tMed, which)) {
                    dEvent = tMed; mediumEvent = true; scatterMed = which; mp = ray.o + ray.d * tMed;
                }
            }

            // Model A perspective catch: if the photon flies through the aperture
            // (nearer than the surface AND any fog collision), it lands on the film.
            if (forwardCatch && nCam > 0 && cams[0].cam && cams[0].film) {
                int px, py;
                if (cams[0].cam->catchPhoton(ray, dEvent, px, py)) {
                    // Same flux->film-irradiance normaliser as connectLens (model A): a
                    // caught photon deposits pupil FLUX into the cell; divide by the cell
                    // area A_cell = pixelPlaneArea()*filmDist^2 so brute-force C keeps the
                    // SAME absolute scale as the importance-sampled A (validated equal),
                    // and both now match B's radiance*camEq in absolute EV.
                    const Camera& cc = *cams[0].cam;
                    double cCell = 1.0 / (cc.pixelPlaneArea() * cc.filmDist * cc.filmDist);
                    cams[0].film->add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * (beta * cCell));
                    e.sensor += beta;
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
                if (a > 0.0) beta *= std::exp(-a * dEvent);
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
            // Multiple scattering (a desaturating wash) is intentionally omitted: the right
            // trade for a crisp view-dependent bow / fogbow / glory / crepuscular-ray flyby.
            if (doBeamGather) {
                if (nCam > 0 && !forwardCatch) {
                    double aC = curAbsorb(lambda);
                    for (int c = 0; c < nCam; ++c) {
                        if (!(cams[c].cam && cams[c].film)) continue;
                        double tC; int whichC;
                        if (!sampleMediaCollision(scene.media, ray.o, ray.d, dSurf, lambda, crng, tC, whichC))
                            continue;   // this camera saw no in-scatter along this beam
                        Vec3 xc = ray.o + ray.d * tC;
                        double betaC = (aC > 0.0) ? betaPre * std::exp(-aC * tC) : betaPre;
                        const Medium& smc = scene.media[whichC];
                        if (lensMode) connectLensVolume(scene, smc, *cams[c].cam, *cams[c].film, xc, ray.d, lambda, betaC, crng);
                        else          connectVolume(scene, smc, *cams[c].cam, *cams[c].film, xc, ray.d, lambda, betaC, crng);
                        camSpecularSplatVolumeAll(scene, smc, &cams[c], 1, xc, ray.d, lambda, betaC, crng);
                    }
                }
                // Attenuate the photon by the medium extinction over the whole crossing
                // (single-scatter transmission) so surfaces behind the fog get correctly
                // dimmed direct light; the removed energy (out-scattered + absorbed) is booked
                // as absorbed. The photon then continues STRAIGHT to the surface below.
                double before = beta;
                beta *= mediaTransmittance(scene.media, ray.o, ray.d, dSurf, lambda, crng);
                e.absorbed += (before - beta);
            }

            if (mediumEvent) {
                const Medium& sm = scene.media[scatterMed];
                if (nCam > 0 && !forwardCatch) {
                    camSplatVolumeAll(scene, sm, cams, nCam, mp, ray.d, lambda, beta, rng);
                    camSpecularSplatVolumeAll(scene, sm, cams, nCam, mp, ray.d, lambda, beta, rng);
                }
                // Scatter (prob albedo) or absorb; throughput unchanged on scatter.
                if (rng.uniform() >= sm.albedo(lambda)) { e.absorbed += beta; return; }
                double phPdf;   // sample the scatter direction from HG or the rainbow droplet phase
                ray = Ray{mp, sm.phaseSample(ray.d, lambda, rng, phPdf)};
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
                case MatType::Dielectric:
                case MatType::ThinFilm:
                case MatType::Multilayer:
                case MatType::Mirror:
                case MatType::Grating:
                case MatType::HalfMirror:
                case MatType::Filter:
                case MatType::Glossy:
                case MatType::Fluorescent: {
                    // Specular / wavelength-switching lobes, all handled by the shared
                    // helper (single source of truth with tracePhotonHero).
                    if (!interactPhotonSpecular(scene, cams, nCam, m, h, ray, beta, lambda, stk, rng, e))
                        return;
                    continue;
                }
                case MatType::DiffuseTransmit: {
                    // Two-lobe Lambertian: reflect albedo into the front hemisphere
                    // (+h.n, the incoming side) and transmit albedo into the back
                    // hemisphere (-h.n). Splat BOTH lobes to the camera — the wrong-side
                    // lobe self-rejects inside connect() (cosSurf<=0), so passing the
                    // flipped normal for the transmit lobe just images whichever side the
                    // camera is on. Because both lobes are non-specular, a directly-viewed
                    // translucent solid is VISIBLE in mode B (unlike clear dielectric).
                    double rhoR = clamp01(diffuseReflectance(scene, m, h, lambda));
                    double rhoT = clamp01(m.transmit(lambda));
                    double sum = rhoR + rhoT;
                    if (sum > 1.0) { rhoR /= sum; rhoT /= sum; sum = 1.0; }  // energy guard
                    Vec3 ngo = orientedGeoN(h);
                    Vec3 wi = Vec3{-ray.d.x, -ray.d.y, -ray.d.z};   // toward the previous (light-side) vertex
                    // Photon-map deposit: incident flux at this translucent vertex.
                    depositPhoton(h.p, ray.d, h.n, lambda, beta);
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
                        Vec3 wo = cosineHemisphere(h.n, rng);
                        beta *= shadingAdjointCorr(wi, wo, h.n, ngo);
                        ray = Ray{h.p + h.n * 1e-6, wo}; continue;
                    } else if (u < sum) {
                        Vec3 wo = cosineHemisphere(Vec3{-h.n.x, -h.n.y, -h.n.z}, rng);
                        beta *= shadingAdjointCorr(wi, wo, h.n, ngo);
                        ray = Ray{h.p - h.n * 1e-6, wo}; continue;
                    }
                    e.absorbed += beta; return;
                }
                case MatType::Diffuse:
                default: {
                    double rho = clamp01(diffuseReflectance(scene, m, h, lambda));
                    Vec3 ngo = orientedGeoN(h);
                    Vec3 wi = Vec3{-ray.d.x, -ray.d.y, -ray.d.z};   // toward the previous (light-side) vertex
                    // Photon-map deposit: incident flux at this diffuse vertex. Stored
                    // BEFORE the Russian-roulette reflect/absorb so the record captures
                    // the arriving power (direct on the first hit, indirect thereafter).
                    depositPhoton(h.p, ray.d, h.n, lambda, beta);
                    if (nCam > 0 && !forwardCatch) {
                        camSplatAll(scene, cams, nCam, h.p, h.n, ngo, wi, lambda, beta, rho, rng);
                        camSpecularSplatAll(scene, cams, nCam, h.p, h.n, lambda, beta, rho, rng);
                    }
                    // Russian roulette: absorb with prob (1-rho), else scatter
                    // with beta unchanged. Unbiased; average path length ~1/(1-rho)
                    // bounces instead of running to the maxBounce cap.
                    if (rng.uniform() >= rho) { e.absorbed += beta; return; }
                    Vec3 wo = cosineHemisphere(h.n, rng);
                    beta *= shadingAdjointCorr(wi, wo, h.n, ngo);   // Veach adjoint (1 when Ns==Ng)
                    ray = Ray{h.p + h.n * 1e-6, wo};
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
        if (em.shape == EmitterShape::Spot) {
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
            em.samplePoint(u1, u2, origin, emitN);
            dir = em.collimated ? em.beamDir : cosineHemisphere(emitN, rng);
        }

        // Hero + stratified secondary wavelengths from this emitter's SPD (one base draw,
        // C-1 wrapped strata). The hero must have a valid pdf; a dead secondary carries
        // beta 0 and simply splats nothing.
        double lam[hero::kHeroMax];
        double u = rng.uniform();
        double pdf0 = 0.0;
        lam[0] = em.spd.sampleAt(u, pdf0);
        if (pdf0 <= 0) return;
        for (int i = 1; i < C; ++i) {
            double uu = u + (double)i / C;
            if (uu >= 1.0) uu -= 1.0;                 // wrap into [0,1)
            double pdfi = 0.0;
            lam[i] = em.spd.sampleAt(uu, pdfi);
        }
        // Per-λ throughput: base power split C ways. Image-env reweights each λ by the
        // directional radiance estimator (no-op for a constant env).
        double base = (scene.emitters.size() == 1) ? em.power : scene.totalPower;
        base *= spotW;
        double beta[hero::kHeroMax];
        for (int i = 0; i < C; ++i) beta[i] = base / C;
        if (em.shape == EmitterShape::Env && scene.envMap) {
            for (int i = 0; i < C; ++i) {
                double denom = 4.0 * PI * envPdfW * em.spdFn(lam[i]);
                beta[i] = (denom > 0.0) ? beta[i] * (scene.envMap->radiance(dir, lam[i]) / denom) : 0.0;
            }
        }

        bool secAlive = (C > 1);
        auto activeSum = [&]() { double s = 0.0; int n = secAlive ? C : 1;
                                 for (int i = 0; i < n; ++i) s += beta[i]; return s; };
        auto deHero = [&]() { if (!secAlive) return; beta[0] *= (double)C; secAlive = false; };
        e.emitted += activeSum();

        // Direct light -> camera (area/quad emitters only; matches the scalar tracer).
        if (nCam > 0 && !forwardCatch &&
            em.shape != EmitterShape::Spot && em.shape != EmitterShape::Env) {
            double rhoOne[hero::kHeroMax]; for (int i = 0; i < C; ++i) rhoOne[i] = 1.0;
            camSplatAllHero(scene, cams, nCam, origin, emitN, emitN, emitN, lam, beta, rhoOne, C, rng);
            camSpecularSplatAllHero(scene, cams, nCam, origin, emitN, lam, beta, rhoOne, C, rng);
        }

        Ray ray{origin + dir * 1e-6, dir};
        MediumStack stk;                 // dielectric priority (Beer-Lambert uses hero λ)

        for (int bounce = 0; bounce < maxBounce; ++bounce) {
            int nUp = secAlive ? C : 1;
            Hit h = scene.closestHit(ray);
            double dEvent = h.valid ? h.t : 1e30;

            // Model C aperture catch: the photon physically threads the pupil.
            if (forwardCatch && nCam > 0 && cams[0].cam && cams[0].film) {
                int px, py;
                if (cams[0].cam->catchPhoton(ray, dEvent, px, py)) {
                    const Camera& cc = *cams[0].cam;
                    double cCell = 1.0 / (cc.pixelPlaneArea() * cc.filmDist * cc.filmDist);
                    for (int i = 0; i < nUp; ++i)
                        cams[0].film->add(px, py, Vec3(cieX(lam[i]), cieY(lam[i]), cieZ(lam[i]))
                                                  * (beta[i] * cCell));
                    e.sensor += activeSum();
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
            // Layered coat: wavelength-dependent Fresnel -> de-hero, run scalar coat on hero.
            if (matp->type == MatType::Layered) {
                deHero(); nUp = 1;
                const Material& cm = *matp;
                double R = layeredCoatReflectance(scene, cm, h, ray.d, lam[0]);
                if (rng.uniform() < R) {
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, cm, h), rng);
                    if (dot(o, h.n) <= 0) { e.absorbed += beta[0]; return; }
                    ray = Ray{h.p + h.n * 1e-6, o};
                    continue;
                }
                int child = mixPickChild(cm, rng.uniform());
                if (child < 0) { e.absorbed += beta[0]; return; }
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

            switch (m.type) {
                case MatType::DiffuseTransmit: {
                    double rhoR[hero::kHeroMax], rhoT[hero::kHeroMax];
                    for (int i = 0; i < nUp; ++i) {
                        double rr = clamp01(diffuseReflectance(scene, m, h, lam[i]));
                        double rt = clamp01(m.transmit(lam[i]));
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
                    for (int i = 0; i < nUp; ++i)
                        depositPhoton(h.p, ray.d, h.n, lam[i], beta[i]);
                    if (nCam > 0 && !forwardCatch) {
                        camSplatAllHero(scene, cams, nCam, h.p,  h.n,  ngo, wi, lam, beta, rhoR, nUp, rng);
                        camSplatAllHero(scene, cams, nCam, h.p, -h.n, -ngo, wi, lam, beta, rhoT, nUp, rng);
                        camSpecularSplatAllHero(scene, cams, nCam, h.p,  h.n, lam, beta, rhoR, nUp, rng);
                        camSpecularSplatAllHero(scene, cams, nCam, h.p, -h.n, lam, beta, rhoT, nUp, rng);
                    }
                    double sumHero = rhoR[0] + rhoT[0];
                    double uu = rng.uniform();
                    if (uu < rhoR[0]) {                           // reflect (front)
                        for (int i = 1; i < nUp; ++i) beta[i] *= rhoR[i] / rhoR[0];
                        Vec3 wo = cosineHemisphere(h.n, rng);
                        double corr = shadingAdjointCorr(wi, wo, h.n, ngo);
                        for (int i = 0; i < nUp; ++i) beta[i] *= corr;
                        ray = Ray{h.p + h.n * 1e-6, wo}; continue;
                    } else if (uu < sumHero) {                    // transmit (back)
                        for (int i = 1; i < nUp; ++i) beta[i] *= rhoT[i] / rhoT[0];
                        Vec3 wo = cosineHemisphere(Vec3{-h.n.x, -h.n.y, -h.n.z}, rng);
                        double corr = shadingAdjointCorr(wi, wo, h.n, ngo);
                        for (int i = 0; i < nUp; ++i) beta[i] *= corr;
                        ray = Ray{h.p - h.n * 1e-6, wo}; continue;
                    }
                    e.absorbed += activeSum(); return;
                }
                case MatType::Dielectric:
                case MatType::ThinFilm:
                case MatType::Multilayer:
                case MatType::Mirror:
                case MatType::Grating:
                case MatType::HalfMirror:
                case MatType::Filter:
                case MatType::Glossy:
                case MatType::Fluorescent: {
                    // Dispersive / wavelength-switching: terminate secondaries, then run
                    // the shared scalar interaction on the (boosted) hero channel.
                    deHero();
                    if (!interactPhotonSpecular(scene, cams, nCam, m, h, ray, beta[0], lam[0], stk, rng, e))
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
                    // Photon-map deposit: store EVERY live wavelength as its own per-λ
                    // photon record (the gather keys off each photon's own λ). C records
                    // of base/C sum to base, and nEmitted counts PATHS, so the estimator
                    // stays energy-consistent with the scalar single-λ deposit.
                    for (int i = 0; i < nUp; ++i)
                        depositPhoton(h.p, ray.d, h.n, lam[i], beta[i]);
                    if (nCam > 0 && !forwardCatch) {
                        camSplatAllHero(scene, cams, nCam, h.p, h.n, ngo, wi, lam, beta, rho, nUp, rng);
                        camSpecularSplatAllHero(scene, cams, nCam, h.p, h.n, lam, beta, rho, nUp, rng);
                    }
                    double rhoHero = rho[0];
                    if (rng.uniform() >= rhoHero) { e.absorbed += activeSum(); return; }  // hero RR absorb
                    for (int i = 1; i < nUp; ++i) beta[i] *= rho[i] / rhoHero;            // secondary reweight
                    Vec3 wo = cosineHemisphere(h.n, rng);
                    double corr = shadingAdjointCorr(wi, wo, h.n, ngo);
                    for (int i = 0; i < nUp; ++i) beta[i] *= corr;
                    ray = Ray{h.p + h.n * 1e-6, wo};
                    continue;
                }
            }
        }
        e.residual += activeSum();
    }

    // Dielectric interface: Fresnel-weighted stochastic choice of specular
    // reflection or refraction (Snell), with wavelength-dependent index -> dispersion.
    // Specular reflect-or-refract at a dielectric interface. If `transmitted` is
    // given it reports whether the photon crossed the interface (refracted) vs.
    // reflected/TIR — the caller uses this to track which medium it is now inside
    // (interior absorption). A non-zero `roughness` frosts the interface: BOTH the
    // reflected and refracted lobes are jittered by a power-cosine lobe (rough glass),
    // rejecting samples that would cross to the wrong side so no light leaks through.
    // `extIor` is the refractive index of the medium on the NON-material side of this
    // interface — i.e. what the ray is travelling through when it enters, or what it
    // returns to when it exits. Defaults to 1.0 (vacuum/air), which reproduces the old
    // exterior-is-air behaviour bit-for-bit. Nested-dielectric callers pass the enclosing
    // medium's index (from the per-path priority stack) so glass-in-water refracts across
    // 1.33<->1.52 instead of 1.0<->1.52.
    Ray refractOrReflect(const Scene& scene, const Material& m, const Hit& h, const Vec3& d,
                         double lambda, Pcg32& rng, bool* transmitted = nullptr,
                         double extIor = 1.0) const {
        double ng = m.ior(lambda);
        bool entering = dot(d, h.ng) < 0.0;
        Vec3 nl = entering ? h.ng : -h.ng;      // normal on the incidence side
        double n1 = entering ? extIor : ng;
        double n2 = entering ? ng : extIor;
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
