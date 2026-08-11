// Fiber BCSDF (P3 stage 1) — the scattering model for a single hair / fur strand.
//
// WHAT THIS IS. A hair is not a surface, it is a dielectric CYLINDER, and light that
// enters it can leave anywhere around the circumference after any number of internal
// bounces. So the right object is not a BRDF over a hemisphere but a BCSDF over the
// whole sphere, indexed by how many times the path crossed the fiber's interior:
//
//   p = 0   R      reflected straight off the cuticle           — the white sheen
//   p = 1   TT     in one side, out the other                   — the forward glow
//   p = 2   TRT    in, bounced off the far wall, back out       — the coloured rim
//   p >= 3          everything else, folded into one residual term
//
// Marschner et al. (2003) named those lobes and gave the geometry; the formulation
// implemented here is Chiang et al. (2016) / d'Eon et al. (2011), which is the same
// physics rearranged so that it (a) conserves energy, (b) has a closed-form importance
// sampler, and (c) does not need Marschner's Gaussian-with-a-fudge azimuthal term. That
// matters practically: the raw 2003 formulation cannot be sampled and loses energy at
// high roughness, so it is only usable as a "multiply by the light" shading hack, which
// is exactly the thing a path tracer cannot do.
//
// COORDINATES. Everything here is in the LOCAL FIBER FRAME:
//
//     +x = the fiber tangent (root -> tip)
//     y, z spanning the normal plane
//
// so a direction w decomposes into a longitudinal sine `sinTheta = w.x` (the angle off
// the normal plane, +-pi/2 along the fiber) and an azimuth `phi = atan2(w.z, w.y)`
// around it. This is PBRT's convention, deliberately: the published white-furnace and
// sampling-consistency tests are stated in it, and `-checkhair` runs them, so matching
// the convention is what lets those tests be *checks* rather than re-derivations.
//
// `h` in [-1, 1] is the offset of the ray from the fiber axis in the normal plane,
// normalised by the radius — the impact parameter. It is NOT a direction; it is a
// property of where on the cylinder we are standing, and `hFromHit` recovers it from the
// surface normal the intersector already produced.
//
// SPECTRAL. This renderer is spectral, so `sigmaA` here is the absorption coefficient at
// ONE wavelength and the whole BCSDF is scalar. That is strictly simpler than PBRT's
// RGB version, which has to carry a 3-vector through the attenuation chain and then pick
// a lobe by luminance; here the caller evaluates per-lambda and the absorption is
// physically what it claims to be. Fiber absorption is also the single most wavelength-
// dependent thing about hair (eumelanin and pheomelanin are strong, smooth absorbers
// across the visible), so this is the part of the model spectral rendering most improves.
#pragma once
#include <cmath>
#include <algorithm>
#include "linalg.h"

namespace hair {

inline constexpr double kPi = 3.14159265358979323846;

// Number of explicitly-tracked scattering lobes: p = 0 (R), 1 (TT), 2 (TRT), and index 3
// as a single residual standing for every p >= 3. Marschner stopped at TRT; keeping the
// residual is what makes the model energy-conserving on a light fiber (see the furnace
// test in -checkhair §1), because for a weakly absorbing hair those tails are not small.
inline constexpr int kPMax = 3;

inline double sqr(double x) { return x * x; }
inline double safeSqrt(double x) { return std::sqrt(std::max(0.0, x)); }
inline double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

// --- Longitudinal scattering M_p --------------------------------------------
// The normalised Gaussian ON THE SPHERE (a von Mises-Fisher restricted to the
// longitudinal angle), not the flat Gaussian in theta that Marschner used. d'Eon's point
// is that the flat one integrates to more than 1 as roughness grows, which is where the
// 2003 model's energy gain comes from; this one is normalised by construction.
//
// The Bessel I0 that normalisation requires overflows for small v (a smooth fiber), so
// the small-v branch works in logs. Both branches are the same function — §2 of
// -checkhair pins them against each other across the crossover.
inline double logBesselI0(double x);

// The ascending series sum_k (x/2)^{2k} / (k!)^2. Every term is positive, so it never
// cancels -- but it needs *enough* terms. A fixed ten (the usual float-precision choice)
// is already 0.26% low by x = 10 and 4% low by x = 12, and because I0 sits inside M_p's
// normalisation that error shows up directly as a longitudinal lobe that does not
// integrate to one -- i.e. as invented or lost energy at grazing angles. So: iterate to
// double convergence, and hand large arguments to the asymptotic form, which is both
// cheaper there and immune to the eventual overflow of e^x.
inline double besselI0(double x) {
    x = std::fabs(x);
    if (x > 12.0) return std::exp(logBesselI0(x));
    const double q = 0.25 * x * x;
    double t = 1.0, sum = 1.0;
    for (int k = 1; k < 64; ++k) {
        t *= q / (double(k) * double(k));
        sum += t;
        if (t < 1e-18 * sum) break;
    }
    return sum;
}

inline double logBesselI0(double x) {
    x = std::fabs(x);
    if (x > 12.0) {
        // A&S 9.7.1: I0(x) ~ e^x / sqrt(2 pi x) * sum_k prod_j (2j-1)^2 / (k! (8x)^k).
        // Asymptotic, so it is truncated near its smallest term; at the x = 12 crossover
        // that floor is ~1e-10 relative and it only improves as x grows.
        double t = 1.0, sum = 1.0;
        for (int k = 1; k <= 12; ++k) {
            t *= sqr(2.0 * k - 1.0) / (8.0 * x * double(k));
            sum += t;
        }
        return x - 0.5 * std::log(2.0 * kPi * x) + std::log(sum);
    }
    return std::log(besselI0(x));
}

inline double Mp(double cosThetaI, double cosThetaO,
                 double sinThetaI, double sinThetaO, double v) {
    const double a = cosThetaI * cosThetaO / v;
    const double b = sinThetaI * sinThetaO / v;
    if (v <= 0.1)
        return std::exp(logBesselI0(a) - b - 1.0 / v + 0.6931471805599453 +
                        std::log(0.5 / v));
    return (std::exp(-b) * besselI0(a)) / (std::sinh(1.0 / v) * 2.0 * v);
}

// --- Azimuthal scattering N_p -----------------------------------------------
// Where the light comes OUT around the circumference, given that it went in at impact
// parameter h and refracted p times. Specular geometry fixes the exit azimuth exactly
// (`Phi` below, straight from Bravais/Snell in the normal plane); roughness smears it.
//
// The smear is a LOGISTIC, trimmed to [-pi, pi], not a Gaussian. Two reasons, both
// practical rather than aesthetic: the logistic's CDF is elementary, so it inverts in
// closed form and gives an exact importance sampler; and being trimmed to exactly one
// revolution it stays normalised on the circle instead of leaking probability past +-pi
// the way a wrapped Gaussian does unless you sum the wrap terms.
inline double Phi(int p, double gammaO, double gammaT) {
    return 2.0 * p * gammaT - 2.0 * gammaO + p * kPi;
}

inline double logistic(double x, double s) {
    x = std::fabs(x);
    const double e = std::exp(-x / s);
    return e / (s * sqr(1.0 + e));
}

inline double logisticCdf(double x, double s) { return 1.0 / (1.0 + std::exp(-x / s)); }

inline double trimmedLogistic(double x, double s, double a, double b) {
    return logistic(x, s) / (logisticCdf(b, s) - logisticCdf(a, s));
}

// Inverse CDF of the trimmed logistic on [a, b]; the sampler's whole azimuthal half.
inline double sampleTrimmedLogistic(double u, double s, double a, double b) {
    const double k = logisticCdf(b, s) - logisticCdf(a, s);
    double x = -s * std::log(1.0 / (u * k + logisticCdf(a, s)) - 1.0);
    return clampd(x, a, b);
}

inline double wrapAngle(double phi) {
    while (phi > kPi)  phi -= 2.0 * kPi;
    while (phi < -kPi) phi += 2.0 * kPi;
    return phi;
}

inline double Np(double phi, int p, double s, double gammaO, double gammaT) {
    return trimmedLogistic(wrapAngle(phi - Phi(p, gammaO, gammaT)), s, -kPi, kPi);
}

// --- Attenuation A_p ---------------------------------------------------------
// How much energy survives to leave via lobe p: Fresnel at each interface crossing, and
// Beer-Lambert along however much fiber interior the path traversed. `T` is the single-
// traversal transmittance, already raised through the chord length by the caller.
//
// The p = kPMax entry is the sum of the geometric tail sum_{p>=kPMax} (T f)^k, in closed
// form. Dropping it (as Marschner does) is what makes a blonde fiber render dark.
inline double frDielectric(double cosThetaI, double etaI, double etaT) {
    cosThetaI = clampd(cosThetaI, -1.0, 1.0);
    if (cosThetaI < 0.0) { std::swap(etaI, etaT); cosThetaI = -cosThetaI; }
    const double sinThetaI = safeSqrt(1.0 - sqr(cosThetaI));
    const double sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.0) return 1.0;                        // total internal reflection
    const double cosThetaT = safeSqrt(1.0 - sqr(sinThetaT));
    const double rParl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                         ((etaT * cosThetaI) + (etaI * cosThetaT));
    const double rPerp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                         ((etaI * cosThetaI) + (etaT * cosThetaT));
    return 0.5 * (rParl * rParl + rPerp * rPerp);
}

inline void Ap(double cosThetaO, double eta, double h, double T, double ap[kPMax + 1]) {
    const double cosGammaO = safeSqrt(1.0 - h * h);
    // The Fresnel angle is the FULL 3-D incidence on the cylinder wall, which is the
    // longitudinal and azimuthal cosines multiplied — not cosThetaO alone. Getting this
    // wrong is invisible head-on and wrong by tens of percent at grazing.
    const double f = frDielectric(cosThetaO * cosGammaO, 1.0, eta);
    ap[0] = f;                                   // R
    ap[1] = sqr(1.0 - f) * T;                    // TT
    for (int p = 2; p < kPMax; ++p) ap[p] = ap[p - 1] * T * f;
    const double denom = 1.0 - T * f;
    ap[kPMax] = (denom > 1e-12) ? ap[kPMax - 1] * f * T / denom : 0.0;
}

// --- The BCSDF ---------------------------------------------------------------
// Authored parameters. `betaM` / `betaN` are perceptual roughnesses in [0, 1] mapped to
// the model's internal variance / logistic scale by Chiang's fitted polynomials, so that
// equal steps look like equal steps; `alpha` is the cuticle scale tilt in degrees, the
// thing that separates the R and TRT highlights into the two distinct bands real hair
// shows (and the reason a hair highlight sits off the mirror direction).
struct Params {
    double eta   = 1.55;   // index of refraction of keratin
    double betaM = 0.3;    // longitudinal roughness [0, 1]
    double betaN = 0.3;    // azimuthal roughness   [0, 1]
    double alpha = 2.0;    // cuticle scale tilt, degrees
};

// Everything that depends only on the hit and the parameters — hoisted out of f()/
// sample() because a single shading point evaluates the BCSDF many times (once per light
// sample, once per wavelength) and none of this changes between those calls.
struct Bcsdf {
    double h = 0.0, gammaO = 0.0;
    double eta = 1.55, sigmaA = 0.0;
    double v[kPMax + 1] = {0, 0, 0, 0};   // longitudinal variances per lobe
    double s = 0.0;                       // azimuthal logistic scale
    double sin2kAlpha[3] = {0, 0, 0}, cos2kAlpha[3] = {1, 1, 1};
};

// `sigmaA` is per-unit-radius absorption at the wavelength being traced (the chord
// lengths below are in units of the fiber radius, so sigmaA is dimensionless).
inline Bcsdf make(const Params& pr, double h, double sigmaA) {
    Bcsdf b;
    b.h      = clampd(h, -1.0, 1.0);
    b.gammaO = std::asin(b.h);
    b.eta    = pr.eta;
    b.sigmaA = std::max(0.0, sigmaA);

    // Chiang's roughness fits. The high powers (20, 22) are not curve-fitting noise:
    // they are what makes the mapping flat over most of the range and then blow up right
    // at beta = 1, which is the "and now it is a diffuse fuzz" end.
    const double bm = clampd(pr.betaM, 1e-4, 1.0);
    const double bn = clampd(pr.betaN, 1e-4, 1.0);
    b.v[0] = sqr(0.726 * bm + 0.812 * sqr(bm) + 3.7 * std::pow(bm, 20));
    b.v[1] = 0.25 * b.v[0];
    b.v[2] = 4.0 * b.v[0];
    for (int p = 3; p <= kPMax; ++p) b.v[p] = b.v[2];
    b.s = 0.626657069 *      // sqrt(pi/8)
          (0.265 * bn + 1.194 * sqr(bn) + 5.372 * std::pow(bn, 22));

    // The tilt is applied as a rotation of the longitudinal angle by alpha for R, -alpha/2
    // for TT and -3alpha/2 for TRT, which in this parameterisation is the double-angle
    // recurrence below rather than three separate trig calls.
    b.sin2kAlpha[0] = std::sin(pr.alpha * kPi / 180.0);
    b.cos2kAlpha[0] = safeSqrt(1.0 - sqr(b.sin2kAlpha[0]));
    for (int i = 1; i < 3; ++i) {
        b.sin2kAlpha[i] = 2.0 * b.cos2kAlpha[i - 1] * b.sin2kAlpha[i - 1];
        b.cos2kAlpha[i] = sqr(b.cos2kAlpha[i - 1]) - sqr(b.sin2kAlpha[i - 1]);
    }
    return b;
}

// Apply the lobe-p cuticle tilt to the outgoing longitudinal angle.
inline void tiltO(const Bcsdf& b, int p, double sinThetaO, double cosThetaO,
                  double& sinThetaOp, double& cosThetaOp) {
    if (p == 0) {
        sinThetaOp = sinThetaO * b.cos2kAlpha[1] - cosThetaO * b.sin2kAlpha[1];
        cosThetaOp = cosThetaO * b.cos2kAlpha[1] + sinThetaO * b.sin2kAlpha[1];
    } else if (p == 1) {
        sinThetaOp = sinThetaO * b.cos2kAlpha[0] + cosThetaO * b.sin2kAlpha[0];
        cosThetaOp = cosThetaO * b.cos2kAlpha[0] - sinThetaO * b.sin2kAlpha[0];
    } else if (p == 2) {
        sinThetaOp = sinThetaO * b.cos2kAlpha[2] + cosThetaO * b.sin2kAlpha[2];
        cosThetaOp = cosThetaO * b.cos2kAlpha[2] - sinThetaO * b.sin2kAlpha[2];
    } else {
        sinThetaOp = sinThetaO;
        cosThetaOp = cosThetaO;
    }
    cosThetaOp = std::fabs(cosThetaOp);
}

// Shared geometry: the refracted azimuth and the interior chord, both of which depend
// only on wo. Factored out because f(), pdf() and sample() all need exactly this.
inline void refractGeom(const Bcsdf& b, double sinThetaO, double cosThetaO,
                        double& gammaT, double& T) {
    // Bravais' "virtual index": a ray at longitudinal angle theta refracts in the normal
    // plane as if the index were this, which is what reduces the 3-D cylinder problem to
    // the familiar 2-D circle one.
    const double etap      = safeSqrt(sqr(b.eta) - sqr(sinThetaO)) / std::max(cosThetaO, 1e-9);
    const double sinGammaT = clampd(b.h / etap, -1.0, 1.0);
    const double cosGammaT = safeSqrt(1.0 - sqr(sinGammaT));
    gammaT = std::asin(sinGammaT);
    const double sinThetaT = sinThetaO / b.eta;
    const double cosThetaT = safeSqrt(1.0 - sqr(sinThetaT));
    // Chord across the circle, lengthened by the longitudinal obliquity. This is the only
    // place the fiber's thickness enters: sigmaA is per radius, the chord is in radii.
    T = std::exp(-b.sigmaA * (2.0 * cosGammaT / std::max(cosThetaT, 1e-9)));
}

// The BCSDF value for wo -> wi, both in the local fiber frame.
//
// NOTE ON THE COSINE. The returned value is already divided by |cos theta_i|, so a caller
// integrates it as `f * |cos theta_i| dw` exactly like a surface BRDF and the two need no
// special-casing between them. That is a convention, not physics — the underlying fiber
// scattering function is per unit length — but it is the convention the published tests
// use, and it keeps the integrators from having to know what a hair is.
inline double f(const Bcsdf& b, const Vec3& wo, const Vec3& wi) {
    const double sinThetaO = clampd(wo.x, -1.0, 1.0), cosThetaO = safeSqrt(1.0 - sqr(sinThetaO));
    const double sinThetaI = clampd(wi.x, -1.0, 1.0), cosThetaI = safeSqrt(1.0 - sqr(sinThetaI));
    const double phiO = std::atan2(wo.z, wo.y);
    const double phiI = std::atan2(wi.z, wi.y);
    const double phi  = phiI - phiO;

    double gammaT, T;
    refractGeom(b, sinThetaO, cosThetaO, gammaT, T);

    double ap[kPMax + 1];
    Ap(cosThetaO, b.eta, b.h, T, ap);

    double sum = 0.0;
    for (int p = 0; p < kPMax; ++p) {
        double sinThetaOp, cosThetaOp;
        tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);
        sum += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, b.v[p]) * ap[p] *
               Np(phi, p, b.s, b.gammaO, gammaT);
    }
    // The residual is isotropic in azimuth by construction — it stands for paths that
    // have forgotten where they entered.
    sum += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, b.v[kPMax]) * ap[kPMax] *
           (0.5 / kPi);

    const double absCosI = std::fabs(cosThetaI);
    if (absCosI > 1e-9) sum /= absCosI;
    return std::isfinite(sum) ? std::max(0.0, sum) : 0.0;
}

// Discrete probability of picking each lobe, proportional to the energy it carries. Using
// the true A_p rather than a fixed split is what keeps the sampler efficient on a dark
// fiber, where TT and TRT are nearly extinct and sampling them would waste most rays.
inline void apPdf(const Bcsdf& b, double sinThetaO, double cosThetaO, double pdf[kPMax + 1]) {
    double gammaT, T;
    refractGeom(b, sinThetaO, cosThetaO, gammaT, T);
    double ap[kPMax + 1];
    Ap(cosThetaO, b.eta, b.h, T, ap);
    double total = 0.0;
    for (int p = 0; p <= kPMax; ++p) total += ap[p];
    if (total <= 1e-12) {                     // degenerate: fall back to uniform
        for (int p = 0; p <= kPMax; ++p) pdf[p] = 1.0 / (kPMax + 1);
        return;
    }
    for (int p = 0; p <= kPMax; ++p) pdf[p] = ap[p] / total;
}

inline double pdf(const Bcsdf& b, const Vec3& wo, const Vec3& wi) {
    const double sinThetaO = clampd(wo.x, -1.0, 1.0), cosThetaO = safeSqrt(1.0 - sqr(sinThetaO));
    const double sinThetaI = clampd(wi.x, -1.0, 1.0), cosThetaI = safeSqrt(1.0 - sqr(sinThetaI));
    const double phi = std::atan2(wi.z, wi.y) - std::atan2(wo.z, wo.y);

    double gammaT, Tr;
    refractGeom(b, sinThetaO, cosThetaO, gammaT, Tr);
    double ap[kPMax + 1];
    apPdf(b, sinThetaO, cosThetaO, ap);

    double sum = 0.0;
    for (int p = 0; p < kPMax; ++p) {
        double sinThetaOp, cosThetaOp;
        tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);
        sum += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, b.v[p]) * ap[p] *
               Np(phi, p, b.s, b.gammaO, gammaT);
    }
    sum += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, b.v[kPMax]) * ap[kPMax] *
           (0.5 / kPi);
    return std::isfinite(sum) ? std::max(0.0, sum) : 0.0;
}

// Importance-sample an outgoing direction. Takes four independent uniforms rather than
// PBRT's two-plus-bit-demux: the demux exists to keep a 2-D sampler's stratification, and
// this renderer hands out independent uniforms anyway, so it would only add a hazard.
//
// Returns the sampled `wi` (local frame); `pdfOut` and `fOut` are filled to match, so a
// caller's weight is simply fOut * |cos theta_i| / pdfOut.
inline Vec3 sample(const Bcsdf& b, const Vec3& wo, double u0, double u1, double u2,
                   double u3, double& pdfOut, double& fOut) {
    const double sinThetaO = clampd(wo.x, -1.0, 1.0), cosThetaO = safeSqrt(1.0 - sqr(sinThetaO));
    const double phiO = std::atan2(wo.z, wo.y);

    double gammaT, T;
    refractGeom(b, sinThetaO, cosThetaO, gammaT, T);
    double ap[kPMax + 1];
    apPdf(b, sinThetaO, cosThetaO, ap);

    int p = 0;
    { double c = 0.0;
      for (; p < kPMax; ++p) { c += ap[p]; if (u0 < c) break; } }

    double sinThetaOp, cosThetaOp;
    tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);

    // Longitudinal: exact inversion of M_p (Ou & Pellacini). The 1e-5 floor is not
    // cosmetic — log(0) here would produce a NaN direction that then poisons the film.
    u1 = std::max(u1, 1e-5);
    const double cosTheta =
        1.0 + b.v[p] * std::log(u1 + (1.0 - u1) * std::exp(-2.0 / b.v[p]));
    const double sinTheta = safeSqrt(1.0 - sqr(cosTheta));
    const double cosPhi   = std::cos(2.0 * kPi * u2);
    const double sinThetaI = clampd(-cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp,
                                    -1.0, 1.0);
    const double cosThetaI = safeSqrt(1.0 - sqr(sinThetaI));

    // Azimuthal: the specular exit angle plus a trimmed-logistic offset, or uniform for
    // the residual lobe (which has no exit angle to be specular about).
    const double dphi = (p < kPMax)
        ? Phi(p, b.gammaO, gammaT) + sampleTrimmedLogistic(u3, b.s, -kPi, kPi)
        : 2.0 * kPi * u3;
    const double phiI = phiO + dphi;

    const Vec3 wi{sinThetaI, cosThetaI * std::cos(phiI), cosThetaI * std::sin(phiI)};
    pdfOut = pdf(b, wo, wi);
    fOut   = f(b, wo, wi);
    return wi;
}

// --- Absorption from a colour ------------------------------------------------
// Authoring convenience (Chiang eq. 9): the sigmaA that makes a fiber of the given
// roughness settle at multiply-scattered reflectance `c`. It is an inverted fit to the
// FULL multiple-scattering response, not Beer-Lambert on one pass, which is why the
// roughness appears in it at all and why it is worth having rather than asking an author
// for an absorption coefficient they cannot picture.
inline double sigmaAFromReflectance(double c, double betaN) {
    c = clampd(c, 1e-4, 1.0 - 1e-6);
    const double bn = clampd(betaN, 1e-4, 1.0);
    const double d = 5.969 - 0.215 * bn + 2.532 * sqr(bn) - 10.73 * std::pow(bn, 3) +
                     5.574 * std::pow(bn, 4) + 0.245 * std::pow(bn, 5);
    return sqr(std::log(c) / d);
}

// --- Hooking the model to a hit ----------------------------------------------
// Recover the impact parameter from geometry the intersector already produced. `n` is the
// outward GEOMETRIC normal at the hit and `tangent` the fiber axis; both are what
// intersectCurveSeg writes into the Hit. Deriving h here rather than storing it on the
// Hit keeps the intersector — which is the hottest loop in a fur render and knows nothing
// about materials — free of BCSDF concerns.
//
// In the normal plane, the angle between the outward normal and the outgoing direction IS
// gamma_o, so h = sin(gamma_o) with the sign taken about the fiber axis.
inline double hFromHit(const Vec3& n, const Vec3& tangent, const Vec3& wo) {
    const Vec3 ax = normalize(tangent);
    Vec3 nPerp = n  - ax * dot(ax, n);
    Vec3 oPerp = wo - ax * dot(ax, wo);
    const double nl = length(nPerp), ol = length(oPerp);
    if (nl < 1e-12 || ol < 1e-12) return 0.0;   // exactly end-on: the axis is degenerate
    nPerp = nPerp * (1.0 / nl);
    oPerp = oPerp * (1.0 / ol);
    const double cosGamma = clampd(dot(nPerp, oPerp), -1.0, 1.0);
    const double sign = (dot(cross(nPerp, oPerp), ax) < 0.0) ? -1.0 : 1.0;
    return sign * safeSqrt(1.0 - sqr(cosGamma));
}

// World <-> local fiber frame. x is the tangent; the normal plane basis is built from the
// surface normal so that the frame is continuous around the fiber rather than flipping
// wherever an arbitrary onb() happens to.
struct Frame {
    Vec3 x, y, z;
};

inline Frame frameFromHit(const Vec3& n, const Vec3& tangent) {
    Frame fr;
    fr.x = normalize(tangent);
    Vec3 yy = n - fr.x * dot(fr.x, n);
    const double yl = length(yy);
    if (yl > 1e-12) {
        fr.y = yy * (1.0 / yl);
    } else {                                  // degenerate: any perpendicular will do
        Vec3 t, bt; onb(fr.x, t, bt); fr.y = t;
    }
    fr.z = cross(fr.x, fr.y);
    return fr;
}

inline Vec3 toLocal(const Frame& fr, const Vec3& w) {
    return Vec3{dot(w, fr.x), dot(w, fr.y), dot(w, fr.z)};
}
inline Vec3 toWorld(const Frame& fr, const Vec3& w) {
    return fr.x * w.x + fr.y * w.y + fr.z * w.z;
}

}  // namespace hair
