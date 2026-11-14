#pragma once
// ============================================================================================
// `ftrace -checkggx` -- unit tests for the glossy lobe (src/ggx.h), as pure math: no scene. Each test
// runs on the single-scattering lobe and on the lobe with its multiple scattering (`ms`, the one a
// plain glossy material shades with).
//
//   1. RECIPROCITY. fAll(a, b) == fAll(b, a), to rounding.
//   2. THE DRAW'S WEIGHT. weightAll() == fAll cos(b) / pdfAll() on drawn pairs, to rounding: the
//      factor a walk takes is the one the evaluator and the density imply.
//   3. THE SAMPLER DRAWS ITS PDF. For test functions g of the drawn direction, the mean of g over
//      draws equals the integral of g pdfAll over the hemisphere by quadrature, within the
//      estimator's noise: the sampler and the density are one distribution (what MIS rests on).
//   4. ALBEDO. The mean weight over draws equals the integral of fAll cos by quadrature; without
//      `ms` it is never above 1 (single scattering loses energy, never gains it), and with it a white
//      lobe returns what it receives -- the white furnace -- within 1.5 % (the table's resolution).
//   5. THE MIRROR LIMIT. At the least roughness 99 % of the draws lie within a few hundredths of
//      a radian of the mirror direction, and sampleUVAll(u1 == 1) is the mirror exactly.
// ============================================================================================
#include "ggx.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>

namespace ggx {
namespace check {

struct Rng {
    unsigned long long s;
    explicit Rng(unsigned long long seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    double operator()() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (double)((s >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
    }
};

inline int run() {
    int fails = 0;
    auto report = [&](const char* name, bool ms, bool ok, const char* what) {
        std::printf("  %-18s %-6s %s  %s\n", name, ms ? "(ms)" : "(ss)", ok ? "PASS" : "FAIL  <--", what);
        if (!ok) ++fails;
    };
    char buf[256];
    // A tilted normal, so no test leans on n == +z.
    const Vec3 n = normalize(Vec3{0.3, -0.2, 0.93});
    Vec3 t, bt;
    onb(n, t, bt);
    auto dirAt = [&](double cosT, double phi) {
        const double s = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
        return normalize(t * (s * std::cos(phi)) + bt * (s * std::sin(phi)) + n * cosT);
    };
    const double cosArr[3] = {1.0, std::cos(0.9), std::cos(1.4)};
    const double alphas[3] = {0.1, 0.3, alpha(1.0)};

    for (int msI = 0; msI < 2; ++msI) {
        const bool ms = (msI == 1);
        // ---- 1. reciprocity, 2. the draw's weight --------------------------------------------
        {
            Rng rng(7);
            double worstR = 0.0, worstW = 0.0;
            for (double al : alphas)
                for (double ca : cosArr) {
                    const Vec3 a = dirAt(ca, 0.7);
                    for (int i = 0; i < 20000; ++i) {
                        const Vec3 b = sampleUVAll(n, a, al, ms, rng(), rng());
                        const Pair p = pair(n, a, b, al), q = pair(n, b, a, al);
                        if (!p.ok) continue;
                        const double fa = fAll(p, ms), fb = fAll(q, ms);
                        worstR = std::max(worstR, std::fabs(fa - fb) / std::max(fa, 1e-300));
                        const double w = weightAll(p, ms), w2 = fa * p.cb / pdfAll(p, ms);
                        worstW = std::max(worstW, std::fabs(w - w2) / std::max(w2, 1e-300));
                    }
                }
            std::snprintf(buf, sizeof buf, "worst |f(a,b) - f(b,a)| / f = %.2e (want < 1e-9)", worstR);
            report("reciprocity", ms, worstR < 1e-9, buf);
            std::snprintf(buf, sizeof buf, "worst |w - f cos / pdf| / w = %.2e (want < 1e-9)", worstW);
            report("draw weight", ms, worstW < 1e-9, buf);
        }

        // ---- 3. the sampler draws its pdf, 4. albedo -----------------------------------------
        {
            // Quadrature over the hemisphere about n, uniform in (cos, phi): d omega = dcos dphi.
            const int NC = 1500, NP = 1500;
            double worstZ = 0.0, worstA = 0.0, maxAlb = 0.0, worstFurnace = 0.0;
            char worstWhat[160] = "";
            for (double al : alphas)
                for (double ca : cosArr) {
                    const Vec3 a = dirAt(ca, 0.7);
                    // The test functions: 1 (the share drawn above n), cos(b), and b's component
                    // along the tangent the arrival leans toward (an odd moment the lobe's
                    // off-specular shift moves).
                    const Vec3 lean = normalize(a - n * dot(a, n) + t * 1e-9);
                    double qI[3] = {0, 0, 0}, qAlb = 0.0;
                    for (int i = 0; i < NC; ++i) {
                        const double c = (i + 0.5) / NC;
                        for (int j = 0; j < NP; ++j) {
                            const Vec3 b = dirAt(c, 2.0 * kPi * (j + 0.5) / NP);
                            const Pair p = pair(n, a, b, al);
                            if (!p.ok) continue;
                            const double dw = (1.0 / NC) * (2.0 * kPi / NP);
                            const double pd = pdfAll(p, ms) * dw;
                            qI[0] += pd;
                            qI[1] += pd * p.cb;
                            qI[2] += pd * dot(b, lean);
                            qAlb += fAll(p, ms) * p.cb * dw;
                        }
                    }
                    Rng rng(1000 + (unsigned long long)(al * 1000) + (unsigned long long)(ca * 100) +
                            (ms ? 77 : 0));
                    const int N = 2000000;
                    double s[3] = {0, 0, 0}, s2[3] = {0, 0, 0}, sw = 0.0, sw2 = 0.0;
                    for (int k = 0; k < N; ++k) {
                        const Vec3 b = sampleUVAll(n, a, al, ms, rng(), rng());
                        const Pair p = pair(n, a, b, al);
                        if (!p.ok) continue;
                        const double g[3] = {1.0, p.cb, dot(b, lean)};
                        for (int m = 0; m < 3; ++m) { s[m] += g[m]; s2[m] += g[m] * g[m]; }
                        const double w = weightAll(p, ms);
                        sw += w; sw2 += w * w;
                    }
                    for (int m = 0; m < 3; ++m) {
                        const double mean = s[m] / N, var = s2[m] / N - mean * mean;
                        const double se = std::sqrt(std::max(var, 0.0) / N) + 1e-4 * std::fabs(qI[m]);
                        const double z = std::fabs(mean - qI[m]) / se;
                        if (z > worstZ) {
                            worstZ = z;
                            std::snprintf(worstWhat, sizeof worstWhat,
                                          "alpha %.3f cos %.2f g%d: drawn %.5f vs quadrature %.5f", al, ca,
                                          m, mean, qI[m]);
                        }
                    }
                    const double alb = sw / N;
                    const double seA = std::sqrt(std::max(sw2 / N - alb * alb, 0.0) / N) + 1e-4 * qAlb;
                    worstA = std::max(worstA, std::fabs(alb - qAlb) / seA);
                    maxAlb = std::max(maxAlb, alb);
                    worstFurnace = std::max(worstFurnace, std::fabs(alb - 1.0));
                }
            std::snprintf(buf, sizeof buf, "worst %.1f sigma (want < 5); %s", worstZ, worstWhat);
            report("sampler draws pdf", ms, worstZ < 5.0, buf);
            if (ms) {
                std::snprintf(buf, sizeof buf,
                              "worst %.1f sigma (want < 5), worst |albedo - 1| %.4f (want < 0.015)", worstA,
                              worstFurnace);
                report("white furnace", ms, worstA < 5.0 && worstFurnace < 0.015, buf);
            } else {
                std::snprintf(buf, sizeof buf, "worst %.1f sigma (want < 5), highest albedo %.4f (want <= 1)",
                              worstA, maxAlb);
                report("albedo", ms, worstA < 5.0 && maxAlb <= 1.0 + 1e-9, buf);
            }
        }

        // ---- 5. the mirror limit -------------------------------------------------------------
        // GGX's tail is long (a draw beyond the half-vector angle x has odds about alpha^2 / x^2), so
        // the test reads the 99th percentile of the draws' angle to the mirror, not the widest: about
        // 20 alpha in the reflected direction.
        {
            const double al = alpha(kMinRoughness);
            Rng rng(99);
            double worst = 0.0;
            bool exact = true;
            std::vector<double> ang(100000);
            for (double ca : cosArr) {
                const Vec3 a = dirAt(ca, 2.1);
                const Vec3 mirror = reflect(a * -1.0, n);
                const Vec3 m0 = sampleUVAll(n, a, al, ms, 1.0, 0.37);
                exact &= (m0.x == mirror.x && m0.y == mirror.y && m0.z == mirror.z);
                for (double& x : ang)
                    x = std::acos(std::min(1.0, dot(sampleUVAll(n, a, al, ms, rng(), rng()), mirror)));
                std::nth_element(ang.begin(), ang.begin() + 99000, ang.end());
                worst = std::max(worst, ang[99000]);
            }
            std::snprintf(buf, sizeof buf,
                          "99th percentile %.4f rad off the mirror (want < 0.05); u1 = 1 %s", worst,
                          exact ? "is the mirror" : "is NOT the mirror");
            report("mirror limit", ms, worst < 0.05 && exact, buf);
        }
    }

    std::printf("  -> ggx lobe: %d failure(s)\n", fails);
    return fails;
}

}  // namespace check
}  // namespace ggx
