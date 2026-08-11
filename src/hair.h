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
#include <string>
#include "linalg.h"

namespace hair {

inline constexpr double kPi = 3.14159265358979323846;

// Number of explicitly-tracked scattering lobes: p = 0 (R), 1 (TT), 2 (TRT), and index 3
// as a single residual standing for every p >= 3. Marschner stopped at TRT; keeping the
// residual is what makes the model energy-conserving on a light fiber (see the furnace
// test in -checkhair §1), because for a weakly absorbing hair those tails are not small.
inline constexpr int kPMax = 3;

// With a medulla (stage 3) two more lobes join them, so the sampler's discrete choice and
// every per-lobe array runs over kNLobes rather than kPMax + 1:
//
//   4  TT^s    entered, was SCATTERED by the medulla on the first crossing, left
//   5  TRT^s   entered, crossed the medulla clean, bounced off the far wall, and was
//              scattered on the second crossing before leaving
//
// Yan et al. (2017) call these the scattered lobes and show they carry most of a fur
// fiber's light: the medulla blocks the specular TT almost entirely, so a fur strand's
// forward glow is TT^s, not TT. Both are identically zero when kappa = 0.
inline constexpr int kSTT   = kPMax + 1;   // 4
inline constexpr int kSTRT  = kPMax + 2;   // 5
inline constexpr int kNLobes = kPMax + 3;  // 6

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

    // --- the MEDULLA (P3 stage 3; Yan et al. 2015/2017) ----------------------
    // Animal fur is not a solid rod. It has a hollow, structured core — the medulla —
    // which SCATTERS rather than merely absorbing, and it is why plain Marschner on fur
    // reads as coloured plastic: without it, everything that enters the fiber must leave
    // along one of three specular azimuths, so a coat has no soft interior glow at all.
    //
    // `kappa` is the medullary index: medulla radius / fiber radius. It is the master
    // switch — at 0 every expression below collapses to the stage-1 solid-cylinder model
    // *bit for bit*, which is what keeps every existing hair scene unregressed.
    //
    // Real fitted values (Yan et al. 2017, Table 4): human 0.36, dog 0.68, raccoon 0.65,
    // mouse 0.66, rabbit 0.79, springbok 0.82, red fox 0.86, cat 0.87, bobcat 0.88,
    // deer 0.91. Fur is mostly medulla; human hair is mostly cortex. That single number
    // is most of the difference between "hair" and "fur".
    double kappa   = 0.0;   // medullary index, [0, 1)
    double mSigmaS = 0.0;   // medulla SCATTERING coefficient, per fiber radius
    double mSigmaA = 0.0;   // medulla ABSORPTION coefficient, per fiber radius
    double mG      = 0.0;   // Henyey-Greenstein anisotropy of medulla scattering, (-1, 1)
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

    // Medulla. `hasMedulla` is false whenever kappa or sigma_s is zero, and every code
    // path below tests it before doing anything, so a solid fiber costs one predictable
    // branch and executes the exact stage-1 arithmetic.
    bool   hasMedulla = false;
    double kappa = 0.0, mSigmaS = 0.0, mSigmaA = 0.0, mG = 0.0;
};

// Everything a single interior traversal of the fiber does to a ray: where it comes out
// azimuthally, and how much of it survives. Split out (rather than returned as two loose
// doubles, as in stage 1) because the double cylinder needs five related quantities and
// they are all computed from the same two chord lengths.
struct Chord {
    double gammaT   = 0.0;   // refracted azimuth inside the fiber
    double T        = 1.0;   // survival of the UNSCATTERED path across one traversal
    double Tsolid   = 1.0;   // the same with the medulla's extinction removed (cortex only)
    double albedoM  = 0.0;   // medulla single-scattering albedo, sigma_s / (sigma_s + sigma_a)
    double Tb       = 1.0;   // Yan eq. 20's post-scatter escape: kappa of medulla, 1-kappa of cortex
    double tauP     = 0.0;   // REDUCED optical depth (1-g)*sigma_s*chord — see scatteredSpread
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

    // A medulla that is infinitely thin, or that does not scatter, is not a medulla — in
    // either case the extra lobes carry exactly zero and the flag stays false, so the
    // whole of stage 3 costs nothing and changes nothing on a human-hair fiber.
    b.kappa   = clampd(pr.kappa, 0.0, 0.999);
    b.mSigmaS = std::max(0.0, pr.mSigmaS);
    b.mSigmaA = std::max(0.0, pr.mSigmaA);
    b.mG      = clampd(pr.mG, -0.999, 0.999);
    b.hasMedulla = (b.kappa > 0.0) && (b.mSigmaS + b.mSigmaA > 0.0);
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
//
// THE DOUBLE CYLINDER. Yan et al. (2017)'s central simplification is to give the cortex
// and the medulla the SAME index of refraction. Yan et al. (2015) let them differ, which
// is more faithful but spawns a combinatorial zoo of T-r-T / T-trt-T paths — and Yan's
// own Fig. 6 shows the two-IOR model still misses the measured profile, while producing a
// spurious dark ring at the cortex/medulla interface. With one IOR the interior ray does
// not bend at that interface at all, so the path TOPOLOGY stays exactly Marschner's
// R / TT / TRT and the medulla only changes what happens ALONG a chord. That is why this
// grafts cleanly onto the stage-1 core instead of replacing it.
//
// In the normal plane the interior ray is a chord of the unit circle whose perpendicular
// distance from the axis is |sin gammaT|. The medulla is the concentric circle of radius
// kappa, so the ray crosses it exactly when |sin gammaT| < kappa, and the two half-chords
// are sm = sqrt(kappa^2 - sin^2 gammaT) and sc = cos gammaT - sm.
inline Chord refractGeom(const Bcsdf& b, double sinThetaO, double cosThetaO) {
    Chord ch;
    // Bravais' "virtual index": a ray at longitudinal angle theta refracts in the normal
    // plane as if the index were this, which is what reduces the 3-D cylinder problem to
    // the familiar 2-D circle one.
    const double etap      = safeSqrt(sqr(b.eta) - sqr(sinThetaO)) / std::max(cosThetaO, 1e-9);
    const double sinGammaT = clampd(b.h / etap, -1.0, 1.0);
    const double cosGammaT = safeSqrt(1.0 - sqr(sinGammaT));
    ch.gammaT = std::asin(sinGammaT);
    const double sinThetaT = sinThetaO / b.eta;
    const double cosThetaT = safeSqrt(1.0 - sqr(sinThetaT));
    // Obliquity: the chord is measured in the normal plane, the ray travels it at the
    // refracted longitudinal angle. This is the only place the fiber's thickness enters —
    // sigmaA is per radius, the chord is in radii.
    const double invCosT = 1.0 / std::max(cosThetaT, 1e-9);

    if (!b.hasMedulla) {
        // Exactly the stage-1 expression, to the last bit.
        ch.T = std::exp(-b.sigmaA * (2.0 * cosGammaT * invCosT));
        ch.Tsolid = ch.T;
        return ch;
    }

    const double sm = safeSqrt(sqr(b.kappa) - sqr(sinGammaT));   // half-chord in medulla
    const double sc = std::max(0.0, cosGammaT - sm);             // half-chord in cortex
    const double mExt = b.mSigmaS + b.mSigmaA;                   // medulla extinction
    // An unscattered path crosses 2*sc of absorbing cortex and 2*sm of the medulla, where
    // BOTH absorption and scattering remove it from the specular chain (Yan Table 3).
    ch.T      = std::exp(-(2.0 * sc * b.sigmaA + 2.0 * sm * mExt) * invCosT);
    // The same fiber with the medulla replaced by cortex. Not a physical configuration —
    // it is the REFERENCE against which the medulla's removed energy is measured, which
    // is what makes the scattered lobes conserve exactly (see ApScattered).
    ch.Tsolid = std::exp(-(2.0 * cosGammaT * b.sigmaA) * invCosT);
    ch.albedoM = (mExt > 1e-12) ? b.mSigmaS / mExt : 0.0;
    // Yan eq. 20's Tb: having scattered, the light is treated as leaving from the axis,
    // so it crosses kappa of medulla and (1 - kappa) of cortex on the way out.
    ch.Tb = std::exp(-(b.kappa * b.mSigmaA + (1.0 - b.kappa) * b.sigmaA) * invCosT);
    // Similarity theory: n forward-peaked scattering events of anisotropy g randomise a
    // direction as well as n(1-g) isotropic ones, so the REDUCED depth is what sets how
    // far the exit direction has forgotten the entry direction.
    ch.tauP = (1.0 - clampd(b.mG, -0.999, 0.999)) * b.mSigmaS * (2.0 * sm) * invCosT;
    return ch;
}

// How much the medulla has smeared the exit direction, as a number in [0, 1): 0 = the
// scattered lobe still points where the specular one did, 1 = it has forgotten entirely.
//
// This is the one place where an ANALYTIC profile stands in for Yan's C^M / C^N, the pair
// of 24x16x16x720 tables they precompute by Monte-Carlo-ing the medulla and then compress
// by rank-16 tensor decomposition. Those tables are not published, and re-deriving them
// would mean shipping a 600 MB simulation or a 150 KB blob nobody can check. Instead:
// after reduced optical depth tau', the mean cosine of the transmitted-direction
// distribution decays as exp(-tau'), which is the standard similarity result. So one
// number drives both the longitudinal variance and the azimuthal logistic width, with the
// right limits at both ends and a physical parameter in between. `-checkhair` section S10
// pins it against a brute-force volumetric random walk in the medulla cylinder.
inline double scatteredSpread(const Chord& ch) {
    return 1.0 - std::exp(-std::max(0.0, ch.tauP));
}

// --- Attenuation of the SCATTERED lobes A^s_p ---------------------------------
// Yan et al. (2017) eq. 20 writes A^s_p = (1 - F) F^(p-1) Ta Tb and leaves the actual
// SCATTERED FRACTION implicit in the normalisation of their measured C^N table. With an
// analytic profile that fraction has to be supplied explicitly — and supplying it the
// obvious way (1 - exp(-sigma_s * chord), times a Fresnel guess for the exit) does not
// conserve energy, because the scattered light then pays a different toll on the way out
// than the specular light it was taken from.
//
// So compute it as a DIFFERENCE instead. Run the stage-1 chain twice: once with the real
// medulla (T), once with the medulla replaced by cortex (Tsolid). The gap between the two
// totals is, by construction, exactly the energy the medulla removed from the specular
// lobes. Multiply by the medulla's single-scattering albedo to keep only the part that was
// scattered rather than absorbed, and by Yan's Tb for the trip out.
//
// The payoff is that the white furnace still closes EXACTLY: with no absorption anywhere,
// the solid chain sums to 1 (Chiang's residual telescopes), albedoM = 1 and Tb = 1, so the
// six lobes sum to (sum of medullated chain) + (1 - sum of medullated chain) = 1. That is
// an algebraic identity, not a numerical near-miss — `-checkhair` S1 asserts it at 1e-12
// with a medulla exactly as it does without one.
//
// The split between TT^s and TRT^s follows Yan's F^(p-1): the second crossing has already
// paid one internal Fresnel reflection, so it is dimmer by F. Only two lobes are tracked,
// so the p >= 3 tail is folded into TRT^s rather than dropped.
inline void ApScattered(const Chord& ch, double cosThetaO, double eta, double h,
                        double aps[2]) {
    aps[0] = aps[1] = 0.0;
    if (ch.albedoM <= 0.0) return;
    double apMed[kPMax + 1], apSolid[kPMax + 1];
    Ap(cosThetaO, eta, h, ch.T, apMed);
    Ap(cosThetaO, eta, h, ch.Tsolid, apSolid);
    double missing = 0.0;
    for (int p = 0; p <= kPMax; ++p) missing += apSolid[p] - apMed[p];
    if (!(missing > 0.0)) return;                 // Tsolid >= T always, but be defensive
    const double total = missing * ch.albedoM * ch.Tb;
    const double cosGammaO = safeSqrt(1.0 - h * h);
    const double F = frDielectric(cosThetaO * cosGammaO, 1.0, eta);
    const double norm = 1.0 + F;
    aps[0] = total * (1.0 / norm);
    aps[1] = total * (F / norm);
}

// Exit azimuth of a scattered lobe: Yan eq. 21. One refraction into the cylinder for
// TT^s, plus one internal reflection for TRT^s — i.e. where the ray was POINTING when it
// entered the medulla, since after scattering it has no specular direction of its own.
inline double PhiS(int p, double gammaO, double gammaT) {
    return (gammaT - gammaO) + double(p - 1) * (kPi + 2.0 * gammaT);
}

// The scattered lobes reuse M_p and N_p, widened by the medulla. Variances add, so the
// longitudinal one is the cuticle's plus the medulla's; the azimuthal logistic scale grows
// toward a value at which the trimmed logistic on [-pi, pi] is flat to within a percent,
// so a thick medulla is azimuthally uniform — which is exactly what a diffusive core
// should look like.
inline double scatteredV(const Bcsdf& b, int p, double spread) {
    return b.v[p] + spread;
}
inline double scatteredS(const Bcsdf& b, double spread) {
    return b.s + 2.0 * spread;
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

    const Chord ch = refractGeom(b, sinThetaO, cosThetaO);
    const double gammaT = ch.gammaT;

    double ap[kPMax + 1];
    Ap(cosThetaO, b.eta, b.h, ch.T, ap);

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

    if (b.hasMedulla) {
        double aps[2];
        ApScattered(ch, cosThetaO, b.eta, b.h, aps);
        const double spread = scatteredSpread(ch);
        const double ss = scatteredS(b, spread);
        for (int k = 0; k < 2; ++k) {
            if (aps[k] <= 0.0) continue;
            const int p = k + 1;                       // TT^s -> 1, TRT^s -> 2
            double sinThetaOp, cosThetaOp;
            tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);
            sum += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp,
                      scatteredV(b, p, spread)) * aps[k] *
                   trimmedLogistic(wrapAngle(phi - PhiS(p, b.gammaO, gammaT)), ss, -kPi, kPi);
        }
    }

    const double absCosI = std::fabs(cosThetaI);
    if (absCosI > 1e-9) sum /= absCosI;
    return std::isfinite(sum) ? std::max(0.0, sum) : 0.0;
}

// Discrete probability of picking each lobe, proportional to the energy it carries. Using
// the true A_p rather than a fixed split is what keeps the sampler efficient on a dark
// fiber, where TT and TRT are nearly extinct and sampling them would waste most rays.
inline void apPdf(const Bcsdf& b, double sinThetaO, double cosThetaO, double pdf[kNLobes]) {
    const Chord ch = refractGeom(b, sinThetaO, cosThetaO);
    double ap[kPMax + 1];
    Ap(cosThetaO, b.eta, b.h, ch.T, ap);
    double aps[2] = {0.0, 0.0};
    if (b.hasMedulla) ApScattered(ch, cosThetaO, b.eta, b.h, aps);
    double total = 0.0;
    for (int p = 0; p <= kPMax; ++p) total += ap[p];
    total += aps[0] + aps[1];
    if (total <= 1e-12) {                     // degenerate: fall back to uniform
        const int n = b.hasMedulla ? kNLobes : (kPMax + 1);
        for (int p = 0; p < kNLobes; ++p) pdf[p] = (p < n) ? 1.0 / double(n) : 0.0;
        return;
    }
    for (int p = 0; p <= kPMax; ++p) pdf[p] = ap[p] / total;
    pdf[kSTT]  = aps[0] / total;
    pdf[kSTRT] = aps[1] / total;
}

inline double pdf(const Bcsdf& b, const Vec3& wo, const Vec3& wi) {
    const double sinThetaO = clampd(wo.x, -1.0, 1.0), cosThetaO = safeSqrt(1.0 - sqr(sinThetaO));
    const double sinThetaI = clampd(wi.x, -1.0, 1.0), cosThetaI = safeSqrt(1.0 - sqr(sinThetaI));
    const double phi = std::atan2(wi.z, wi.y) - std::atan2(wo.z, wo.y);

    const Chord ch = refractGeom(b, sinThetaO, cosThetaO);
    const double gammaT = ch.gammaT;
    double ap[kNLobes];
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
    if (b.hasMedulla) {
        const double spread = scatteredSpread(ch);
        const double ss = scatteredS(b, spread);
        for (int k = 0; k < 2; ++k) {
            const double w = ap[kSTT + k];
            if (w <= 0.0) continue;
            const int p = k + 1;
            double sinThetaOp, cosThetaOp;
            tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);
            sum += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp,
                      scatteredV(b, p, spread)) * w *
                   trimmedLogistic(wrapAngle(phi - PhiS(p, b.gammaO, gammaT)), ss, -kPi, kPi);
        }
    }
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

    const Chord ch = refractGeom(b, sinThetaO, cosThetaO);
    const double gammaT = ch.gammaT;
    double ap[kNLobes];
    apPdf(b, sinThetaO, cosThetaO, ap);

    const int nLobes = b.hasMedulla ? kNLobes : (kPMax + 1);
    int lobe = 0;
    { double c = 0.0;
      for (; lobe < nLobes - 1; ++lobe) { c += ap[lobe]; if (u0 < c) break; } }

    // A scattered lobe rides on its parent's cuticle tilt (TT^s on TT, TRT^s on TRT) and
    // uses the widened variance / logistic scale; an unscattered one is stage 1 unchanged.
    const bool  scattered = (lobe >= kSTT);
    const int   p  = scattered ? (lobe - kSTT + 1) : lobe;
    const double spread = scattered ? scatteredSpread(ch) : 0.0;
    const double vUse = scattered ? scatteredV(b, p, spread) : b.v[p];

    double sinThetaOp, cosThetaOp;
    tiltO(b, p, sinThetaO, cosThetaO, sinThetaOp, cosThetaOp);

    // Longitudinal: exact inversion of M_p (Ou & Pellacini). The 1e-5 floor is not
    // cosmetic — log(0) here would produce a NaN direction that then poisons the film.
    u1 = std::max(u1, 1e-5);
    const double cosTheta =
        1.0 + vUse * std::log(u1 + (1.0 - u1) * std::exp(-2.0 / vUse));
    const double sinTheta = safeSqrt(1.0 - sqr(cosTheta));
    const double cosPhi   = std::cos(2.0 * kPi * u2);
    const double sinThetaI = clampd(-cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp,
                                    -1.0, 1.0);
    const double cosThetaI = safeSqrt(1.0 - sqr(sinThetaI));

    // Azimuthal: the exit angle plus a trimmed-logistic offset — the specular one for an
    // unscattered lobe, Yan's entering-segment direction for a scattered one — or uniform
    // for the residual, which has no exit angle to be specular about.
    double dphi;
    if (scattered)
        dphi = PhiS(p, b.gammaO, gammaT) +
               sampleTrimmedLogistic(u3, scatteredS(b, spread), -kPi, kPi);
    else if (p < kPMax)
        dphi = Phi(p, b.gammaO, gammaT) + sampleTrimmedLogistic(u3, b.s, -kPi, kPi);
    else
        dphi = 2.0 * kPi * u3;
    const double phiI = phiO + dphi;

    const Vec3 wi{sinThetaI, cosThetaI * std::cos(phiI), cosThetaI * std::sin(phiI)};
    pdfOut = pdf(b, wo, wi);
    fOut   = f(b, wo, wi);
    return wi;
}

// --- Roughness from a measured angle ------------------------------------------
// `betaM` / `betaN` are Chiang's PERCEPTUAL [0, 1] knobs, chosen so equal steps look like
// equal steps. Every measurement in the fur literature — Yan et al. (2017) Table 4, and
// every goniophotometer paper before it — instead reports roughness as a Gaussian standard
// deviation IN DEGREES. These invert Chiang's fitted polynomials so a measured number can
// be typed in as itself.
//
// The polynomials' top terms (beta^20, beta^22) exist only to blow up in the last few
// percent before beta = 1, i.e. the "and now it is diffuse fuzz" end. Below that they are
// numerically absent, so the inversion is the quadratic root of the leading two terms,
// which is exact everywhere a real measurement lands (the largest in Yan's table is 18.94
// degrees, where the cubic-and-up terms contribute < 1e-9).
inline double invQuadFit(double target, double c1, double c2) {
    if (!(target > 0.0)) return 0.0;
    const double disc = sqr(c1) + 4.0 * c2 * target;
    return clampd((-c1 + safeSqrt(disc)) / (2.0 * c2), 0.0, 1.0);
}

// M_p's variance parameter v behaves as the square of the angular width, and make() sets
// sqrt(v) = 0.726 b + 0.812 b^2 + ...
inline double betaMFromDegrees(double deg) {
    return invQuadFit(std::fabs(deg) * kPi / 180.0, 0.726, 0.812);
}

// The azimuthal smear is a logistic of scale s, whose standard deviation is pi*s/sqrt(3);
// make() sets s = sqrt(pi/8) * (0.265 b + 1.194 b^2 + ...).
inline double betaNFromDegrees(double deg) {
    const double sTarget = std::fabs(deg) * kPi / 180.0 * 1.7320508075688772 / kPi;
    return invQuadFit(sTarget / 0.626657069, 0.265, 1.194);
}

// --- Measured species ---------------------------------------------------------
// Yan et al. (2017), "A BSSRDF Model for Efficient Rendering of Fur with Global
// Illumination", Table 4: the ten fibers they actually fit against measured
// goniophotometry. These are the whole reason the medulla exists as a feature — a fur
// fiber is not a human hair with a different colour, it is a hair with a large scattering
// core (kappa 0.65-0.91 for every animal in the table, against 0.36 for human), and that
// core is what turns a hard TRT glint into the soft forward wash a real coat has.
//
// Stored EXACTLY as the paper prints them: alpha/betaM/betaN in degrees, sigmas in
// 1/(fiber radius). The degree->perceptual conversion happens on the way out, through the
// inversions above, so a reader can diff this table against the paper line by line.
//
// The paper's tenth column `l` (the layer/lobe-count fit for their BSSRDF stage) has no
// analogue in a pure BCSDF and is deliberately not carried here.
struct Species {
    const char* name;
    double kappa;     // medullary index
    double eta;
    double alphaDeg;  // cuticle tilt
    double betaMDeg;  // longitudinal roughness, Gaussian sigma
    double betaNDeg;  // azimuthal roughness, Gaussian sigma
    double sigmaCA;   // cortex absorption
    double mSigmaS;   // medulla scattering
    double mSigmaA;   // medulla absorption
    double mG;        // medulla Henyey-Greenstein g
};

inline const Species* speciesTable(int& n) {
    static const Species kTab[] = {
        // name         kappa  eta   alpha  betaM  betaN  sig_c  sig_ms sig_ma  g
        {"bobcat",      0.88, 1.69,  5.48, 11.64,  7.49,  0.64,  1.69,  0.17,  0.44},
        {"cat",         0.87, 1.36,  3.65,  5.66,  1.34,  0.06,  2.47,  0.12,  0.60},
        {"deer",        0.91, 1.60,  3.52,  7.00,  4.53,  1.39,  2.51,  0.09,  0.46},
        {"dog",         0.68, 1.58,  2.94,  5.77, 18.94,  0.01,  2.44,  0.00,  0.26},
        {"mouse",       0.66, 1.35,  0.55,  8.39,  2.80,  0.04,  1.34,  0.06,  0.36},
        {"rabbit",      0.79, 1.47,  3.14, 11.91, 10.52,  0.24,  0.78,  0.10,  0.12},
        {"raccoon",     0.65, 1.19,  1.81,  7.44,  6.88,  0.25,  2.30,  0.14,  0.08},
        {"redfox",      0.86, 1.49,  2.64,  9.45, 17.63,  0.39,  3.15,  0.21,  0.79},
        {"springbok",   0.82, 1.48,  4.61,  8.02, 11.46,  0.32,  2.45,  0.31,  0.19},
        {"human",       0.36, 1.20,  0.70,  2.05,  3.75,  0.41,  3.49,  0.00,  0.28},
    };
    n = int(sizeof(kTab) / sizeof(kTab[0]));
    return kTab;
}

// Case-insensitive, and tolerant of the separator: "red fox", "red_fox", "redfox" and
// "RedFox" all land on the same row, because an author writing a scene should not have to
// guess which spelling the table happens to use.
inline bool findSpecies(const char* want, Species& out) {
    auto norm = [](const char* s) {
        std::string r;
        for (const char* p = s; *p; ++p) {
            const char c = *p;
            if (c == ' ' || c == '_' || c == '-') continue;
            r += (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : c;
        }
        return r;
    };
    const std::string key = norm(want);
    int n = 0;
    const Species* t = speciesTable(n);
    for (int i = 0; i < n; ++i)
        if (norm(t[i].name) == key) { out = t[i]; return true; }
    return false;
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

// --- Dual scattering (Zinke et al. 2008) --------------------------------------
// P3 stage 4. Everything above describes ONE fiber. A coat is tens of thousands of them,
// and in a light-coloured one most of what you see has crossed dozens before it reaches
// the eye — which a path tracer can do, but only by paying for a hundred-bounce random
// walk per sample. Dual scattering replaces that walk with two closed-form terms:
//
//   GLOBAL  — light that reached the shading point by scattering FORWARD through the
//             strands between it and the light. Modelled as an attenuation `T_f` (the
//             product of a per-fiber average forward attenuation `a_f`) times an angular
//             SPREAD that has gone azimuthally isotropic and stays a narrow Gaussian
//             longitudinally, with variance the SUM of the per-fiber variances.
//   LOCAL   — light that turned around in the immediate neighbourhood. Folded into a
//             single extra BCSDF lobe `f_back`, a material property, summed over all
//             paths with one or three backward scattering events.
//
// The whole method rests on six averaged quantities per fiber, each a function of one
// angle: the forward/backward average attenuation `a_f`/`a_b` (Zinke eq. 6 / 12) and the
// mean and standard deviation of the longitudinal deviation each of those imposes. This
// struct is that table.
//
// TWO THINGS ARE DONE DIFFERENTLY HERE, both deliberately.
//
// 1. The six curves are MEASURED FROM OUR OWN BCSDF by importance sampling it, not
//    integrated from a lobe decomposition. `sample()` already returns a weight whose
//    expectation is exactly `sum_p A_p`, the same quantity `-checkhair` §S1 asserts is 1
//    in a white furnace — so `a_f + a_b` is that same total, split by which azimuthal
//    half the sample landed in, and the table inherits the energy identity instead of
//    re-deriving it. It also means the medulla (stage 3) is picked up for free: TT^s and
//    TRT^s are just more lobes to the sampler, and a fur fiber's much larger forward
//    attenuation appears in `a_f` with no extra code.
//
// 2. The average backscattering shift and spread are the EXACT weighted sums, not
//    Zinke eq. 16/17. Those are stated in the paper as numerical fits "based on a power
//    series expansion with respect to a_b up to an order of three", which is a sensible
//    thing to do inside a real-time shader and a silly thing to do inside a table that is
//    built once per material. The triple sum of eq. 13 collapses: substituting
//    `k = j+1+q` makes the exponent `m = 2(i+q)` independent of `j`, so `j` contributes a
//    multiplicity of `i` and, with `n = i+q`, the whole thing becomes a single sum
//    `sum_n a_f^2n X(2n) * n(n+1)/2`. (Setting X = 1 reproduces eq. 13's closed form
//    `a_f^2/(1-a_f^2)^3` — `-checkhair` §S11 checks that, and checks the eq. 16/17 fits
//    against these sums.)
//
// Angles: every curve is indexed by an inclination in [-pi/2, pi/2]. Zinke overloads the
// symbol theta_d for both "the inclination of the illumination at a scattering event"
// (eq. 5/6) and "the difference angle (theta_o - theta_i)/2" (eq. 11-17); they are the
// same table read at two different arguments, as in the paper's own implementation.
struct Dual {
    static const int N = 48;          // inclination bins, uniform over [-pi/2, pi/2]
    double af[N], ab[N];              // average forward / backward attenuation
    double alphaF[N], alphaB[N];      // mean longitudinal deviation of each
    double betaF[N], betaB[N];        // and its standard deviation
    // Derived, so the shader does no series summation: Zinke eq. 14, and the exact
    // weighted sums the paper approximates with eq. 16 / 17.
    double Ab[N], deltaB[N], sigmaB[N];
};

// Bin centre / lookup. Linear interpolation with clamped ends — the curves are smooth and
// 48 bins over 180 degrees is finer than the angular structure they carry.
inline double dualTheta(int i) { return -0.5 * kPi + (i + 0.5) * (kPi / Dual::N); }
inline double dualLookup(const double t[Dual::N], double theta) {
    const double x = clampd((theta + 0.5 * kPi) * (Dual::N / kPi) - 0.5, 0.0, Dual::N - 1.0);
    const int i = int(x);
    const int j = (i + 1 < Dual::N) ? i + 1 : i;
    const double u = x - i;
    return t[i] * (1.0 - u) + t[j] * u;
}

// A Gaussian of the given variance, normalised in the angle. This is Zinke's `g`.
inline double dualG(double x, double var) {
    const double v = std::max(var, 1e-8);
    return std::exp(-0.5 * x * x / v) / std::sqrt(2.0 * kPi * v);
}

// The exact `sum_{n>=1} x^n w(n) X(n)` sums behind eq. 16/17, run to convergence.
// `w(n) = 1` for the single-backscatter family and `n(n+1)/2` for the triple.
inline double dualSeries(double x, bool triple, double a, double b, bool wantSigma) {
    if (!(x > 0.0) || x >= 1.0) return 0.0;
    double sum = 0.0, xn = 1.0;
    for (int n = 1; n <= 4096; ++n) {
        xn *= x;
        const double w = triple ? (0.5 * n * (n + 1.0)) : 1.0;
        // Single-backscatter path: 2n forward events plus one backward one.
        // Triple: 2n forward events plus three backward ones.
        const double val = wantSigma
            ? std::sqrt((triple ? 3.0 * b * b : b * b) + 2.0 * n * a * a)
            : ((triple ? 3.0 * b : b) + 2.0 * n * a);
        const double term = xn * w * val;
        sum += term;
        if (term < 1e-14 * (sum + 1e-30) && n > 8) break;
    }
    return sum;
}

// Zinke eq. 16 and 17 verbatim, for §S11 to compare against the exact sums above.
//
// `signFix` flips the sign of eq. 16's `a_b^2` term. The exact coefficient of `alpha_b`
// in the sum above is `(1 + 3u)/(1 + u)` with `u = a_b^2/(1 - a_f^2)^2`, whose expansion
// is `1 + 2u`; the paper prints `1 - 2u`. §S11 measures both, and the fixed one tracks the
// exact sum by two orders of magnitude better — so this is a sign typo in the paper, not a
// disagreement about the model.
inline double dualDeltaFit(double af, double ab, double alphaF, double alphaB,
                           bool signFix = false) {
    const double d = 1.0 - af * af;
    if (!(d > 1e-6)) return alphaB;
    const double u = ab * ab / (d * d);
    return alphaB * (1.0 + (signFix ? 2.0 : -2.0) * u) +
           alphaF * ((2.0 * d * d + 4.0 * af * af * ab * ab) / (d * d * d));
}
inline double dualSigmaFit(double af, double ab, double betaF, double betaB) {
    const double den = ab + ab * ab * ab * (2.0 * betaF + 3.0 * betaB);
    if (!(den > 1e-12)) return betaB;
    return (1.0 + 0.7 * af * af) *
           (ab * std::sqrt(2.0 * betaF * betaF + betaB * betaB) +
            ab * ab * ab * std::sqrt(2.0 * betaF * betaF + 3.0 * betaB * betaB)) / den;
}

// Radical inverse in base `b` — the low-discrepancy sequence the table build uses in
// place of an rng, so a material's table is identical run to run and thread to thread.
inline double dualRadical(unsigned i, unsigned b) {
    double f = 1.0 / b, r = 0.0;
    for (unsigned n = i; n > 0; n /= b) { r += f * (n % b); f /= b; }
    return r;
}

// Build the table for one fiber (one wavelength's worth of absorption). `nSamples` fiber
// samples per inclination bin; 4096 is plenty for curves this smooth (the sampler is
// importance-sampling the BCSDF, so the weights are near-constant).
inline Dual makeDual(const Params& pr, double sigmaA, int nSamples = 4096) {
    Dual d{};
    for (int i = 0; i < Dual::N; ++i) {
        const double th = dualTheta(i);
        const double sinThetaO = std::sin(th), cosThetaO = std::cos(th);
        const Vec3 wo{sinThetaO, cosThetaO, 0.0};      // phi_o = 0: the fiber is azimuthally
                                                       // symmetric, so this loses nothing
        double sF = 0.0, sB = 0.0;                     // sum of weights, forward / backward
        double m1F = 0.0, m2F = 0.0, m1B = 0.0, m2B = 0.0;   // weighted moments of the deviation
        for (int k = 0; k < nSamples; ++k) {
            // h uniform over the fiber's cross-section is exactly the near-field ->
            // far-field average the averaged attenuations are defined against.
            const double h  = 2.0 * dualRadical(k + 1, 2) - 1.0;
            const Bcsdf b = make(pr, h, sigmaA);
            double pdf = 0.0, fv = 0.0;
            const Vec3 wi = sample(b, wo, dualRadical(k + 1, 3), dualRadical(k + 1, 5),
                                   dualRadical(k + 1, 7), dualRadical(k + 1, 11), pdf, fv);
            if (!(pdf > 0.0) || !(fv > 0.0)) continue;
            const double sinThetaI = clampd(wi.x, -1.0, 1.0);
            const double cosLong = safeSqrt(1.0 - sqr(sinThetaI));
            const double w = fv * cosLong / pdf;       // E[w] = sum_p A_p
            if (!(w > 0.0) || !std::isfinite(w)) continue;
            // The deviation from the specular cone. Both directions point AWAY from the
            // fiber, and in that convention every lobe — reflected or transmitted — peaks
            // at theta_i = -theta_o, so this is zero on the cone and the cuticle tilt is
            // what makes its mean nonzero. It is the variable Marschner's M is a Gaussian
            // in, which is why variances add along a path.
            const double dev = std::asin(sinThetaI) + th;
            // Forward vs backward is azimuthal: phi = 0 is retro-reflection (the R lobe)
            // and phi = pi is straight through (TT). Zinke's front hemisphere is |phi| >
            // pi/2, i.e. wi on the far side of the fiber from wo.
            if (wi.y < 0.0) { sF += w; m1F += w * dev; m2F += w * dev * dev; }
            else            { sB += w; m1B += w * dev; m2B += w * dev * dev; }
        }
        const double inv = 1.0 / double(nSamples);
        d.af[i] = sF * inv;
        d.ab[i] = sB * inv;
        d.alphaF[i] = (sF > 0.0) ? m1F / sF : 0.0;
        d.alphaB[i] = (sB > 0.0) ? m1B / sB : 0.0;
        d.betaF[i]  = (sF > 0.0) ? safeSqrt(m2F / sF - sqr(d.alphaF[i])) : 0.0;
        d.betaB[i]  = (sB > 0.0) ? safeSqrt(m2B / sB - sqr(d.alphaB[i])) : 0.0;
    }
    // Eq. 11 / 13 / 14 and the exact forms of 16 / 17.
    for (int i = 0; i < Dual::N; ++i) {
        const double af = clampd(d.af[i], 0.0, 0.9999), ab = clampd(d.ab[i], 0.0, 1.0);
        const double x = af * af, om = 1.0 - x;
        const double A1 = ab * x / om;                       // eq. 11
        const double A3 = ab * ab * ab * x / (om * om * om); // eq. 13
        d.Ab[i] = A1 + A3;
        if (d.Ab[i] > 1e-12) {
            const double sD = ab       * dualSeries(x, false, d.alphaF[i], d.alphaB[i], false) +
                              ab*ab*ab * dualSeries(x, true,  d.alphaF[i], d.alphaB[i], false);
            const double sS = ab       * dualSeries(x, false, d.betaF[i],  d.betaB[i],  true) +
                              ab*ab*ab * dualSeries(x, true,  d.betaF[i],  d.betaB[i],  true);
            d.deltaB[i] = sD / d.Ab[i];
            d.sigmaB[i] = sS / d.Ab[i];
        } else {
            d.deltaB[i] = d.alphaB[i];
            d.sigmaB[i] = d.betaB[i];
        }
    }
    return d;
}

// The local-multiple-scattering lobe, Zinke eq. 10 + 15, in THIS renderer's normalisation.
// `thetaD` is the difference angle (theta_o - theta_i)/2, which indexes the tables; `dev`
// the deviation from the specular cone (theta_o + theta_i), which the Gaussian is in; and
// `cosThetaI` the incident longitudinal cosine. The azimuthal restriction to the backward
// half is applied by the caller, which knows phi.
//
// Zinke widens the Gaussian by the accumulated forward spread, `sigma_b^2 + sigma_f^2`,
// because in the paper the incident direction is still the light's and the spread has to be
// reintroduced analytically. Here it is not: the caller draws the arriving direction FROM
// that spread (hairSpreadDir) and evaluates this lobe at the draw, so the blur is already
// in `dev`. Adding sigma_f^2 on top would apply it twice, which is why there is no such
// argument.
//
// THE PREFACTOR IS DERIVED, NOT COPIED. The paper writes `2 A_b g / (pi cos^2 theta_d)`,
// which is in MARSCHNER's normalisation (S = M N / cos^2 theta_d). `hair.h` is CHIANG's:
// `f = sum_p A_p M_p N_p / |cos theta_i|`, with `M` a density in theta against
// `cos theta_i d theta_i` and `N` a density in phi, so that `INT f cos theta_i domega` is
// `sum_p A_p`, the albedo. Requiring the same of the backscatter lobe — a Gaussian of unit
// area in `dev`, uniform density `1/pi` over the pi-wide backward azimuths, total albedo
// `A_b` — pins the prefactor:
//     INT f_back cos theta_i domega = INT INT f_back cos^2 theta_i d theta_i d phi = A_b
//     =>  f_back = A_b g(dev - delta_b) / (pi cos^2 theta_i).
// So the `2` goes, and the cosine is the SAMPLE's rather than the difference angle's. On
// `scenes/fur_species.ftsl` that correction is worth well under a percent — this lobe is a
// few per cent of the coat's radiance, so its prefactor was never going to be visible. It
// is here because it is the normalisation this file actually uses, and because leaving a
// `cos^2 theta_d` in a denominator that is guarded at 1e-6 leaves a 1e4 firefly multiplier
// lying around on a quantity that does not need one. (The bug that DID cost 13.4x on that
// scene was the caller pairing `hair::f` with the wrong cosine — see hairDualFCos.)
inline double fBack(const Dual& d, double thetaD, double dev, double cosThetaI) {
    const double cc = std::max(cosThetaI * cosThetaI, 1e-6);
    const double Ab = dualLookup(d.Ab, thetaD);
    if (!(Ab > 0.0)) return 0.0;
    const double sb = dualLookup(d.sigmaB, thetaD);
    return Ab * dualG(dev - dualLookup(d.deltaB, thetaD), sb * sb) / (kPi * cc);
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
