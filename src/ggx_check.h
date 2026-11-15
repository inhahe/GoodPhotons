#pragma once
// ============================================================================================
// `ftrace -checkggx` -- unit tests for the glossy lobe (src/ggx.h), as pure math: no scene. Each test
// runs on the single-scattering lobe and on the lobe with its multiple scattering (`ms`, the one a
// plain glossy material shades with).
//
//   1. RECIPROCITY. fAll(a, b) == fAll(b, a), to rounding.
//   2. THE DRAW'S WEIGHT. drawUV's weight == fAll cos(b) / pdfAll() on its draws, to rounding: the
//      factor a walk takes is the one the evaluator and the density imply. Every test draws through
//      drawUV, the draw the tracers take (drawRough).
//   3. THE SAMPLER DRAWS ITS PDF. For test functions g of the drawn direction, the mean of g over
//      draws equals the integral of g pdfAll over the hemisphere by quadrature, within the
//      estimator's noise: the sampler and the density are one distribution (what MIS rests on).
//   4. ALBEDO. The mean weight over draws equals the integral of fAll cos by quadrature; without
//      `ms` it is never above 1 (single scattering loses energy, never gains it), and with it a white
//      lobe returns what it receives -- the white furnace -- within 1.5 % (the table's resolution).
//   4b. THE POLISHED FURNACE (ms only). The same at roughness 0.01-0.06, arrivals down to 88.6
//      degrees, where a near-mirror lobe's E dips within a cosine of about its alpha.
//   5. THE MIRROR LIMIT. At the least roughness 99 % of the draws lie within a few hundredths of
//      a radian of the mirror direction, and drawUV's u1 == 1 is the mirror exactly, weighed by the
//      lobe's albedo (mode W's sample 0, standing for the whole lobe): 1 with `ms`,
//      E(cos_a) without. A roughness past the table's span is held to it.
//   6. THE MULTIPLE SCATTERING'S OWN DRAW (ms only). msPdf integrates to 1 over the hemisphere, and
//      a draw from it (msSampleCos) weighs r = f_ms cos / msPdf over 1 - E(cos_a): mean 1 (f_ms
//      integrates to 1 - E(cos_a)), relative second moment near 1 (the draw adds little variance)
//      and no draw far heavier, wherever the branch carries energy -- the near-mirror lobe met at
//      grazing included, where a cosine draw's weight reaches (1 - E(cos_b)) / (1 - Eavg), tens.
// ============================================================================================
#include "ggx.h"
#include "linalg.h"   // Vec3
#include "rng.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>

namespace ggx {
namespace check {

using Rng = ::Lcg53;

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
                        double w = 0.0;
                        const Vec3 b = drawUV(n, a, al, ms, rng(), rng(), w);
                        const Pair p = pair(n, a, b, al), q = pair(n, b, a, al);
                        if (!p.ok) continue;
                        const double fa = fAll(p, ms), fb = fAll(q, ms);
                        worstR = std::max(worstR, std::fabs(fa - fb) / std::max(fa, 1e-300));
                        const double w2 = fa * p.cb / pdfAll(p, ms);
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
                        double w = 0.0;
                        const Vec3 b = drawUV(n, a, al, ms, rng(), rng(), w);
                        const Pair p = pair(n, a, b, al);
                        if (!p.ok) continue;
                        const double g[3] = {1.0, p.cb, dot(b, lean)};
                        for (int m = 0; m < 3; ++m) { s[m] += g[m]; s2[m] += g[m] * g[m]; }
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

        // ---- 4b. the polished furnace (ms only) -----------------------------------------------------
        // A near-mirror lobe's E dips only near grazing, within a cosine of about its alpha, so the
        // table must resolve E there at small alphas: the white furnace again, at roughness 0.01 to 0.06
        // and arrivals down to 88.6 degrees.
        if (ms) {
            const double rs[4] = {0.01, 0.02, 0.03, 0.06};
            const double cs[5] = {0.024, 0.05, 0.1, 0.3, 1.0};
            double worstP = 0.0;
            char what[96] = "";
            for (double r : rs)
                for (double ca : cs) {
                    const double al = alpha(r);
                    const Vec3 a = dirAt(ca, 0.7);
                    Rng rng(4000 + (unsigned long long)(r * 1000) + (unsigned long long)(ca * 1000));
                    const int N = 1000000;
                    double sw = 0.0;
                    for (int k = 0; k < N; ++k) {
                        double w = 0.0;
                        drawUV(n, a, al, true, rng(), rng(), w);   // w is 0 for a draw below n
                        sw += w;
                    }
                    const double e = std::fabs(sw / N - 1.0);
                    if (e > worstP) {
                        worstP = e;
                        std::snprintf(what, sizeof what, "roughness %.2f cos %.3f: albedo %.4f", r, ca, sw / N);
                    }
                }
            std::snprintf(buf, sizeof buf, "worst |albedo - 1| %.4f (want < 0.015); %s", worstP, what);
            report("polished furnace", ms, worstP < 0.015, buf);
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
                double w0 = 0.0;
                const Vec3 m0 = drawUV(n, a, al, ms, 1.0, 0.37, w0);
                exact &= (m0.x == mirror.x && m0.y == mirror.y && m0.z == mirror.z);
                for (double& x : ang) {
                    double w = 0.0;
                    x = std::acos(std::min(1.0, dot(drawUV(n, a, al, ms, rng(), rng(), w), mirror)));
                }
                std::nth_element(ang.begin(), ang.begin() + 99000, ang.end());
                worst = std::max(worst, ang[99000]);
            }
            std::snprintf(buf, sizeof buf,
                          "99th percentile %.4f rad off the mirror (want < 0.05); u1 = 1 %s", worst,
                          exact ? "is the mirror" : "is NOT the mirror");
            report("mirror limit", ms, worst < 0.05 && exact, buf);
            // drawUV's sample 0, at every test lobe and arrival.
            double worstM = 0.0;
            for (double al2 : alphas)
                for (double ca : cosArr) {
                    double w = -1.0;
                    drawUV(n, dirAt(ca, 2.1), al2, ms, 1.0, 0.37, w);
                    worstM = std::max(worstM, std::fabs(w - (ms ? 1.0 : msE(ca, al2))));
                }
            const bool held = alpha(3.0) == alpha(kMaxRoughness) && alpha(0.0) == alpha(kMinRoughness);
            std::snprintf(buf, sizeof buf, "worst |w - albedo| %.2e (want < 1e-12); roughness %s", worstM,
                          held ? "held to [0.001, 2]" : "NOT held to [0.001, 2]");
            report("mirror sample", ms, worstM < 1e-12 && held, buf);
        }

        // ---- 6. the multiple scattering's own draw -------------------------------------------------
        // Its weight r = f_ms cos / msPdf over 1 - E(cos_a) has mean 1; what the draw adds to a
        // render's variance is its relative second moment E[r^2] / E[r]^2, held near 1 wherever the
        // branch carries energy (1 - E(cos_a) above 1 %), and no draw is far heavier than the rest.
        if (ms) {
            const double alphas6[4] = {alpha(0.01), alpha(0.05), 0.3, alpha(kMaxRoughness)};
            const double cos6[5] = {1.0, 0.5, 0.15, 0.05, 0.024};
            double worstInt = 0.0, worstQ = 0.0, worstM = 0.0, hi = 0.0;
            char qWhat[96] = "", hiWhat[96] = "";
            Rng rng(123);
            for (double al : alphas6) {
                // msPdf over the hemisphere, by quadrature in (cos, phi): the azimuth is uniform.
                const int NC = 200000;
                double q = 0.0;
                for (int i = 0; i < NC; ++i) q += msPdf((i + 0.5) / NC, al) * (2.0 * kPi / NC);
                worstInt = std::max(worstInt, std::fabs(q - 1.0));
                for (double ca : cos6) {
                    const Vec3 a = dirAt(ca, 0.7);
                    const double ea = msE(ca, al);
                    if (!(1.0 - ea > 0.01)) continue;   // the branch carries next to nothing
                    double s1 = 0.0, s2 = 0.0;
                    int nk = 0;
                    for (int k = 0; k < 200000; ++k) {
                        const double cb = msSampleCos(rng(), al);
                        const Vec3 b = dirAt(cb, 2.0 * kPi * rng());
                        const Pair p = pair(n, a, b, al);
                        if (!p.ok) continue;
                        const double r = fMs(p) * p.cb / msPdf(p.cb, al) / (1.0 - ea);
                        s1 += r;
                        s2 += r * r;
                        ++nk;
                        if (r > hi) {
                            hi = r;
                            std::snprintf(hiWhat, sizeof hiWhat, "alpha %.4f cos_a %.3f cos_b %.4f", al, ca, p.cb);
                        }
                    }
                    const double m = s1 / nk, qm = (s2 / nk) / (m * m);
                    worstM = std::max(worstM, std::fabs(m - 1.0));
                    if (qm > worstQ) {
                        worstQ = qm;
                        std::snprintf(qWhat, sizeof qWhat, "alpha %.4f cos_a %.3f", al, ca);
                    }
                }
            }
            std::snprintf(buf, sizeof buf, "worst |int pdf - 1| %.1e (want < 1e-3)", worstInt);
            report("ms pdf", ms, worstInt < 1e-3, buf);
            std::snprintf(buf, sizeof buf,
                          "E[r^2]/E[r]^2 at most %.3f (want < 1.1) at %s; |E[r] - 1| at most %.3f (want < 0.05); "
                          "heaviest %.2f (want < 4) at %s",
                          worstQ, qWhat, worstM, hi, hiWhat);
            report("ms draw weight", ms, worstQ < 1.1 && worstM < 0.05 && hi < 4.0, buf);
        }
    }

    std::printf("  -> ggx lobe: %d failure(s)\n", fails);
    return fails;
}

}  // namespace check
}  // namespace ggx
