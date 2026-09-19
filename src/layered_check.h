#pragma once
// ============================================================================================
// `ftrace -checklayered` — unit tests for the stochastic layered BSDF (src/layered.h).
//
// These run as pure math: no scene, no camera, no film, and therefore no fireflies and no
// sampling noise beyond what the tests themselves average away. That matters because the rig this
// replaces could not separate the model from its scene — `tools/a3_reference.py`'s lobe column
// failed its own control at +10.2 % with NO COAT PRESENT, because its two sides were two renders
// of geometrically different scenes (TODO item 10). Here both sides are the same function.
//
// Five checks, ordered so that a failure points at a cause:
//
//   1. ETA = 1 REDUCTION. A coat of index 1.0 is not a coat. Both the walk's albedo and its f
//      must collapse to the bare body's. This is the control that has to hold even when every
//      hypothesis about layering is true; if it fails, nothing below means anything.
//   2. WHITE FURNACE. A white Lambertian body with no absorption must return albedo exactly 1 at
//      every incidence: light can be delayed by total internal reflection but never destroyed.
//      Catches any mismatch between a Fresnel factor and the probability it was sampled with.
//   3. MIRROR CLOSED FORM. A perfect mirror body has no TIR series — its light leaves at the
//      mirror angle, which is inside the critical cone by construction — so the albedo collapses
//      to F + (1-F)^2 a / (1 - F a), which can be checked exactly. At a = 0.62 this is 0.626,
//      and the independent brute-force render measured 0.6288: two unrelated instruments agreeing.
//   4. f MATCHES sample. Integrating f(wi, .) cos over the hemisphere must reproduce the albedo
//      the sampler produces (excluding the coat's delta lobe, which f cannot represent). This is
//      what pins the refraction Jacobian: no wrong power of eta survives it.
//   5. RECIPROCITY. f(wi, wo) == f(wo, wi) to within the estimator's noise.
// ============================================================================================
#include "layered.h"

#include <cstdio>
#include <cmath>

namespace layered {
namespace check {

// ---- test bodies ----------------------------------------------------------------------------
struct Lambert {
    double alb = 1.0;
    double f(const Vec3&, const Vec3&) const { return alb / 3.14159265358979323846; }
    double pdf(const Vec3&, const Vec3& b) const { return b.z / 3.14159265358979323846; }
    bool sample(const Vec3&, double u1, double u2, Vec3& b, double& weight) const {
        const double r = std::sqrt(u1), phi = 2.0 * 3.14159265358979323846 * u2;
        b = Vec3{ r * std::cos(phi), r * std::sin(phi), std::sqrt(1.0 - u1 > 0 ? 1.0 - u1 : 0.0) };
        weight = alb;                       // cosine sampling: f*cos/pdf == alb exactly
        return b.z > 0.0;
    }
};

// A perfect mirror: f is a delta, so it contributes nothing to an f-estimate and the body is used
// only by the sampling tests. That asymmetry is the point of keeping it separate.
struct Mirror {
    double alb = 0.62;
    double f(const Vec3&, const Vec3&) const { return 0.0; }
    double pdf(const Vec3&, const Vec3&) const { return 0.0; }
    bool sample(const Vec3& a, double, double, Vec3& b, double& weight) const {
        b = Vec3{ -a.x, -a.y, a.z };
        weight = alb;
        return true;
    }
};

struct Rng {
    unsigned long long s;
    explicit Rng(unsigned long long seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    double operator()() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (double)((s >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
    }
};

// mean weight of the walk; `nonSpecOnly` drops the coat's delta lobe so the result is comparable
// with an integral of f
template <class Body>
inline double albedo(const Params& P, const Body& body, const Vec3& wi, int n, unsigned seed,
                     bool nonSpecOnly = false) {
    Rng rng(seed);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        Vec3 wo;
        double w = 0.0;
        bool spec = false;
        if (!sample(P, body, wi, rng, wo, w, &spec)) continue;
        if (nonSpecOnly && spec) continue;
        acc += w;
    }
    return acc / (double)n;
}

// integral of f(wi, .) cos over the upper hemisphere, by cosine-weighted quadrature
template <class Body>
inline double integrateF(const Params& P, const Body& body, const Vec3& wi, int n, unsigned seed) {
    Rng rng(seed);
    const double PI = 3.14159265358979323846;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        const double u1 = rng(), u2 = rng();
        const double r = std::sqrt(u1), phi = 2.0 * PI * u2;
        const Vec3 wo{ r * std::cos(phi), r * std::sin(phi), std::sqrt(1.0 - u1 > 0 ? 1.0 - u1 : 0.0) };
        if (!(wo.z > 0.0)) continue;
        const double pdf = wo.z / PI;
        acc += f(P, body, wi, wo, rng) * wo.z / pdf;
    }
    return acc / (double)n;
}

inline int run() {
    int fails = 0;
    const double PI = 3.14159265358979323846;
    auto report = [&](const char* name, bool ok, const char* fmt, double a, double b) {
        std::printf("  %-22s %s  ", name, ok ? "PASS" : "FAIL");
        std::printf(fmt, a, b);
        std::printf("\n");
        if (!ok) ++fails;
    };

    const Vec3 dirs[3] = { Vec3{0, 0, 1},
                           Vec3{std::sin(0.5), 0, std::cos(0.5)},
                           Vec3{std::sin(1.1), 0, std::cos(1.1)} };

    // ---- 1. eta = 1 is not a coat ------------------------------------------------------------
    {
        Params P; P.eta = 1.0;
        Lambert body; body.alb = 0.7;
        double worst = 0.0;
        for (const Vec3& wi : dirs) {
            const double a = albedo(P, body, wi, 200000, 11);
            worst = std::max(worst, std::fabs(a - 0.7) / 0.7);
        }
        report("eta=1 reduction", worst < 0.01, "worst albedo error %.3f %% (want < 1 %%)%.0s",
               100.0 * worst, 0.0);
    }

    // ---- 2. white furnace ---------------------------------------------------------------------
    {
        Params P; P.eta = 1.5;
        Lambert body; body.alb = 1.0;
        double worst = 0.0;
        for (const Vec3& wi : dirs) {
            const double a = albedo(P, body, wi, 200000, 23);
            worst = std::max(worst, std::fabs(a - 1.0));
        }
        report("white furnace", worst < 0.005, "worst |albedo - 1| = %.4f (want < 0.005)%.0s",
               worst, 0.0);
    }

    // ---- 3. mirror body: closed form ---------------------------------------------------------
    {
        Params P; P.eta = 1.5;
        Mirror body; body.alb = 0.62;
        double worst = 0.0;
        double got0 = 0.0, want0 = 0.0;
        for (const Vec3& wi : dirs) {
            const double F = fresnel(wi.z, P.eta);
            const double want = F + (1.0 - F) * (1.0 - F) * body.alb / (1.0 - F * body.alb);
            const double got = albedo(P, body, wi, 200000, 37);
            if (worst < std::fabs(got - want) / want) { got0 = got; want0 = want; }
            worst = std::max(worst, std::fabs(got - want) / want);
        }
        report("mirror closed form", worst < 0.01, "got %.4f vs %.4f", got0, want0);
    }

    // ---- 4. f integrates to the sampled albedo ----------------------------------------------
    {
        Params P; P.eta = 1.5;
        Lambert body; body.alb = 0.8;
        double worst = 0.0, gi = 0.0, ga = 0.0;
        for (const Vec3& wi : dirs) {
            const double byF = integrateF(P, body, wi, 60000, 51);
            const double byS = albedo(P, body, wi, 200000, 67, /*nonSpecOnly=*/true);
            const double e = std::fabs(byF - byS) / std::max(byS, 1e-9);
            if (e > worst) { gi = byF; ga = byS; }
            worst = std::max(worst, e);
        }
        report("f matches sample", worst < 0.02, "integral %.4f vs sampled %.4f", gi, ga);
    }

    // ---- 5. reciprocity ----------------------------------------------------------------------
    {
        Params P; P.eta = 1.5;
        Lambert body; body.alb = 0.8;
        double worst = 0.0, x = 0.0, y = 0.0;
        const Vec3 pairs[3][2] = {
            { dirs[0], dirs[1] }, { dirs[1], dirs[2] }, { dirs[0], dirs[2] } };
        for (auto& pr : pairs) {
            Rng r1(101), r2(202);
            double ab = 0.0, ba = 0.0;
            for (int i = 0; i < 40000; ++i) {
                ab += f(P, body, pr[0], pr[1], r1);
                ba += f(P, body, pr[1], pr[0], r2);
            }
            ab /= 40000.0; ba /= 40000.0;
            const double e = std::fabs(ab - ba) / std::max(0.5 * (ab + ba), 1e-12);
            if (e > worst) { x = ab; y = ba; }
            worst = std::max(worst, e);
        }
        report("reciprocity", worst < 0.03, "f(a,b) %.5f vs f(b,a) %.5f", x, y);
    }

    // ---- 6. the reduction: a Lambertian body must reproduce the SHIPPED analytic model -------
    // This is the control that matters most for the change being made. The new path generalises
    // `coatedAlbedo(a, fdr)` by replacing its assumed GEOMETRIC bounce distribution with a
    // measured one; if it is right, then in the one case the analytic form was derived for -- a
    // Lambertian body -- the two must agree to within Monte-Carlo noise. It also cross-checks two
    // independently written pieces of physics: this walk's per-direction Fresnel averaged over a
    // cosine hemisphere against the shipped Egan-Hilgeman polynomial internalFresnelDiffuse().
    {
        Params P; P.eta = 1.5; P.maxBounce = 32;
        Lambert body; body.alb = 1.0;            // albedo is deliberately absent from the moments
        const int K = 24;
        double mom[K];
        Rng rng(97);
        escapeMoments(P, body, std::cos(0.4), K, mom, 400000, rng);
        double sum = 0.0;
        for (int k = 0; k < K; ++k) sum += mom[k];
        if (sum > 0.0) for (int k = 0; k < K; ++k) mom[k] /= sum;   // condition on having entered

        const double fdr = internalFresnelDiffuse(P.eta);
        double worst = 0.0, gm = 0.0, ga = 0.0;
        for (double a : { 0.3, 0.62, 0.9 }) {
            const double byMom = albedoFromMoments(a, mom, K);
            const double byAna = coatedAlbedo(a, fdr);
            const double e = std::fabs(byMom - byAna) / std::max(byAna, 1e-9);
            if (e > worst) { gm = byMom; ga = byAna; }
            worst = std::max(worst, e);
        }
        report("lambertian reduction", worst < 0.02, "moments %.4f vs analytic %.4f", gm, ga);
    }

    // ---- 7. THE WIRING, not just the math ----------------------------------------------------
    // Checks 1-6 prove the model. This one proves the model is actually REACHED: build a Scene
    // with a layered material whose coat asks for the stochastic model, run the same
    // finalizeLayeredCoats() the loader runs, and read back what shading would see. Added because
    // the end-to-end furnace came back byte-identical to the analytic column while the model,
    // computed independently, predicted 0.60 against 0.40 -- so the defect was in the plumbing,
    // and a unit test that only exercises the math could never have located it.
    {
        Scene sc;
        Material body;
        body.type = MatType::Glossy;
        body.roughness = 0.25;
        body.reflect = constantSpectrum(0.62);
        sc.mats.push_back(body);
        Material lay;
        lay.type = MatType::Layered;
        lay.ior = iorConstant(1.5);
        lay.coatScatter = 1;
        lay.mixChildren.push_back(0);
        lay.mixWeights.push_back(1.0);
        sc.mats.push_back(lay);
        sc.finalizeLayeredCoats();
        const int cid = sc.mats[1].mixChildren[0];
        const Material& bc = sc.mats[(size_t)cid];
        const double aNew = coatedAlbedoAt(bc, 0.62, 550.0);
        const double aOld = coatedAlbedo(0.62, internalFresnelDiffuse(1.5));
        const bool ok = (bc.coatMomN > 0) && (std::fabs(aNew - aOld) / aOld > 0.10);
        std::printf("  %-22s %s  coatMomN=%d  a_eff %.4f (analytic would be %.4f)\n",
                    "scene wiring", ok ? "PASS" : "FAIL  <--", bc.coatMomN, aNew, aOld);
        if (!ok) ++fails;
    }

    (void)PI;
    std::printf("  -> layered BSDF: %d failure(s)\n", fails);
    return fails;
}

}  // namespace check
}  // namespace layered
