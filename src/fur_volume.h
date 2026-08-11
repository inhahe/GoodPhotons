// fur_volume.h — the aggregate scattering half of the fur LOD.  (TODO.md §P2 stage 2)
//
// `fur_grid.h` answers "how much fiber is along this ray"; this file answers "and what does
// light DO when it meets that fiber".  Together they are a participating medium that stands
// in for a coat of hair once the coat is far enough away that individual strands are not
// worth intersecting.
//
// THE MODEL.  A cell of the grid is treated as a cloud of independent fibers with a known
// density (`c`) and a known ORIENTATION DISTRIBUTION FUNCTION (the ODF).  A collision is
// then a completely ordinary fiber hit:
//
//   1. draw a tangent t from the ODF, weighted by how much cross-section that tangent
//      presents to the arriving ray — `p(t) ~ ODF(t) * sin(angle(d,t))`;
//   2. draw the impact parameter h uniformly on [-1,1] (a ray crossing a cylinder of radius
//      r has a uniform offset, which is exactly why the cross-section is `2 r l sin`);
//   3. evaluate or sample the SAME `hair::Bcsdf` a real strand would have used.
//
// So the far tier is not a different material with a fitted phase function: it is the near
// tier's own BCSDF, integrated over a distribution of strands instead of over one strand.
// A coat's colour, its medulla, its cuticle tilt and its energy conservation all carry over
// untouched, and the only thing lost is WHICH strand was hit.
//
// RECONSTRUCTING THE ODF FROM A SECOND MOMENT.  The grid stores only `T = <t t^T>`, so the
// ODF has to be reconstructed from it.  A second moment does not determine a distribution,
// so this is a modelling choice, and the one made here is the ANGULAR CENTRAL GAUSSIAN:
//
//      p(t) ~ (t^T A^-1 t)^(-3/2)      sampled as   t = normalize(sqrt(A) g),  g ~ N(0, I)
//
// with A symmetric positive semi-definite, sharing T's eigenvectors.  Three things recommend
// it over the obvious alternatives:
//
//   * ITS THREE CORNER CASES ARE EXACT, and they are exactly the three corners of the space
//     of orientation tensors.  `A = diag(1,0,0)` is a delta on one axis — locally parallel
//     strands, where real fur spends nearly all its time.  `A = I` is the uniform sphere.
//     `A = diag(1,1,0)` is the UNIFORM GREAT CIRCLE — a girdle, strands lying every which way
//     within a common plane, which is what happens over a whorl or wherever a coat sweeps
//     across curvature inside one cell.  A mixture of bipolar lobes (an earlier draft used
//     Watson lobes on T's eigenvectors) is exact at the first two and badly wrong at the
//     third: it turns a uniform ring into two orthogonal deltas, and measured 43% L1 error
//     against the population it stands for.  The ACG measures ~2%.
//   * IT IS MOMENT-MATCHED EVERYWHERE ELSE.  `A` is recovered from T by inverting
//     `m_i(A) = a_i * INTEGRAL_0^inf du / [ (1 + 2 u a_i) * PROD_j sqrt(1 + 2 u a_j) ]`,
//     which `AcgTable` does once at startup over the whole (compact, triangular) space of
//     eigenvalue triples and then interpolates.  So the phase function and the extinction
//     are read off the same tensor and cannot drift apart.
//   * IT IS TRIVIAL AND EXACT TO SAMPLE — three Gaussians and a normalise, no rejection, no
//     inverse-CDF table, no failure mode when the distribution is nearly a delta.
//
// THE SIGN.  `t t^T == (-t)(-t)^T`, so everything above is antipodally symmetric (which is
// also why the ACG is the right family and a von Mises-Fisher would have had to be
// symmetrised by hand).  The sign is therefore NOT in the second moment, and it is not
// ignorable either: the cuticle tilt `alpha` tips the R and TRT lobes toward the root, so
// reversing a tangent moves the aggregate response — by 27% on a perfectly parallel cell,
// which is the case everything else in this file gets exactly right.  So the sign comes from
// a separate quantity, the cell's FIRST moment `v = sum(r l t)/sum(r l)`, which `FurGrid`
// stores in the four bytes the unit trace made redundant.  `FurODF::orient` spends it: draw
// the axis from the ACG, then point it along `v` with probability `(1 + |v|)/2`.  The two
// moments never interfere — no choice of sign can change `T`.

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "linalg.h"
#include "rng.h"
#include "hair.h"
#include "fur_grid.h"

namespace furvol {

// ---------------------------------------------------------------------------------------
// Symmetric 3x3 eigendecomposition, cyclic Jacobi.  Small, exact enough (it converges
// quadratically and 12 sweeps is far past machine precision at this size), and — unlike the
// closed-form trigonometric solution — it produces an ORTHONORMAL eigenbasis even when two
// eigenvalues coincide, which is the common case here (an isotropic or axially symmetric
// cell) and the one the closed form handles worst.
inline void symEigen3(const double min[6], Vec3 e[3], double lam[3]) {
    // min = { xx, yy, zz, xy, xz, yz }
    double a[3][3] = {{min[0], min[3], min[4]},
                      {min[3], min[1], min[5]},
                      {min[4], min[5], min[2]}};
    double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int sweep = 0; sweep < 12; ++sweep) {
        double off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
        if (off < 1e-18) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                if (std::fabs(a[p][q]) < 1e-300) continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; ++k) {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                    const double vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
                a[p][q] = a[q][p] = 0.0;
            }
    }
    int ord[3] = {0, 1, 2};
    for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (a[ord[j]][ord[j]] > a[ord[i]][ord[i]]) std::swap(ord[i], ord[j]);
    for (int i = 0; i < 3; ++i) {
        const int k = ord[i];
        lam[i] = a[k][k];
        e[i]   = normalize(Vec3{v[0][k], v[1][k], v[2][k]});
    }
}

// ---------------------------------------------------------------------------------------
// THE BINGHAM DISTRIBUTION, and the table that inverts its second moment.
//
//      p(t) ~ exp(t^T B t)        B symmetric, sharing T's eigenvectors
//
// This is the MAXIMUM-ENTROPY distribution on the sphere with a given second moment, which
// is the precise sense in which it assumes nothing beyond what the grid stored.  Two other
// families were built and measured against explicit fiber populations first (`-checkfurvol`
// section 6, L1 error over the whole outgoing sphere):
//
//      population          Watson mixture      ACG        Bingham
//      parallel                 0.000         0.000        0.006
//      combed clump             0.114         0.257        0.080
//      isotropic                0.044         0.044        0.040
//      girdle                   0.431         0.058        0.023
//
// Bingham wins everywhere except the parallel corner, and it loses there only because BMAX
// stops the concentration short of a true delta — a 0.6% error on the one population the
// grid ALSO describes exactly, which is a fair price for halving the error on the combed
// clump that real fur is mostly made of and cutting the girdle by 20x.
//
// A mixture of bipolar Watson lobes on T's eigenvectors is exact for parallel and isotropic
// cells but turns a GIRDLE — strands lying every which way within a common plane, which is
// what a coat does over a whorl or wherever it sweeps across curvature inside one cell —
// into two orthogonal deltas.  The Angular Central Gaussian fixes the girdle (its
// `A = diag(1,1,0)` IS the uniform great circle) but its density has polynomial tails, so it
// smears a tight combed clump over twice the sphere it should.  Bingham is Gaussian-tailed
// like Watson AND covers the girdle like the ACG, because its concentration matrix is free to
// go negative; it degenerates to Watson exactly when two eigenvalues coincide.
//
// THE MOMENT MAP.  B is defined up to an additive multiple of I, so fix `b3 = 0` and carry
// `b1 >= b2 >= 0`.  Writing t3 = c and integrating out the azimuth in closed form,
//
//      Z  = 2 pi e^b1 INT_-1^1 e^(-c^2 b1) k0(x) dc
//      m1 = (1/2) INT (1-c^2) e^(-c^2 b1) (k0(x) + k1(x)) dc  /  INT e^(-c^2 b1) k0(x) dc
//      m3 =       INT   c^2   e^(-c^2 b1)  k0(x)             dc  /  (the same denominator)
//
// with `x = (1-c^2)(b1-b2)/2` and `k_n(x) = e^-x I_n(x)` the exponentially scaled modified
// Bessel functions.  Factoring `e^b1` out and using the SCALED Bessels is what keeps this
// finite: both the exponential and I0 overflow on their own well before b1 reaches the
// values a near-parallel cell needs.  The remaining integrand is a peak of half-width
// `1/sqrt(2 b1)` at c = 0, so the quadrature runs under the substitution
// `c = w sinh(kappa z)` with `kappa = asinh(1/w)`, which maps [0,1] onto [0,1] while putting
// ~40 of its nodes inside that peak no matter how sharp it is; a uniform grid in c would
// silently mis-integrate the very cells fur is mostly made of.
//
// THE INVERSE MAP is a damped Newton over the triangular space of eigenvalue triples —
// tau_1 >= tau_2 >= tau_3 >= 0 summing to 1 — sheared onto the unit square by
// `A' = tau1 - tau2` and `B' = (tau2 - tau3)/((1 - A')/2)`, whose three corners are exactly
// the three cases the model must get right.  It runs once at startup, warm-starting each
// node from its neighbour.
//
// The Newton's VARIABLES are not the b's themselves but
//
//      y_0 = log(1 + b1 - b2)          y_1 = log(1 + b2)
//
// i.e. the GAP first, then the floor, each through the log that linearises the saturation as
// b runs to infinity.  Solving in `(log(1+b1), log(1+b2))` instead looks equivalent and is
// not: the entire `tau_3 = 0` edge — every planar cell, one whole side of the triangle — then
// needs both coordinates pinned at their maximum with only their difference telling the
// points apart, and the Newton cannot see a difference of 0.08 between two numbers near 1500.
// It stalled there, returning the girdle solution for a PARALLEL cell (b = (1500, 1499.9),
// second-moment error 3.2e-1).  In the gap variables each edge of the triangle is an edge of
// the square — Watson at y_1 = 0, girdle at y_0 = 0, planar at y_1 = max — and the worst
// off-node moment error over the whole space falls from 1.15e-2 to 1.15e-3.
//
// SAMPLING is Kent, Ganeiber & Mardia (2013): an ACG proposal `Omega = I + 2 Lambda / bk`
//
// with `Lambda_i = b1 - b_i >= 0` and `bk` the root of `SUM 1/(bk + 2 Lambda_i) = 1`,
// accepted with probability `e^(-t^T Lambda t) (t^T Omega t)^(3/2) / M`.  Acceptance is 100%
// at isotropy, ~90% on a girdle and never falls below ~50%, and the proposal is itself
// sampled in closed form as `normalize(Omega^(-1/2) z)`.
struct BinghamTable {
    static const int N  = 49;    // nodes per axis of the (A', B') square
    static const int NQ = 192;   // Simpson intervals for the moment integral (even)
    static constexpr double BMAX = 1500.0;   // b beyond this is a delta to within 3e-4
    std::vector<float> y0, y1;   // log(1 + b_0 - b_1) and log(1 + b_1) at each node

    // Exponentially scaled modified Bessel functions, Abramowitz & Stegun 9.8 (|err| < 2e-7
    // relative, far inside what a moment match needs).
    static void besselK01(double x, double& k0, double& k1) {
        if (x < 3.75) {
            const double t = x / 3.75, t2 = t * t;
            const double i0 = 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492
                            + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
            const double i1 = x * (0.5 + t2 * (0.87890594 + t2 * (0.51498869 + t2 * (0.15084934
                            + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411))))));
            const double e = std::exp(-x);
            k0 = i0 * e; k1 = i1 * e;
        } else {
            const double t = 3.75 / x, r = 1.0 / std::sqrt(x);
            k0 = r * (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565
                    + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537
                    + t * (-0.01647633 + t * 0.00392377))))))));
            k1 = r * (0.39894228 + t * (-0.03988024 + t * (-0.00362018 + t * (0.00163801
                    + t * (-0.01031555 + t * (0.02282967 + t * (-0.02895312
                    + t * (0.01787654 + t * -0.00420059))))))));
        }
    }

    // The Bingham's own second moment, descending, for concentrations (b0, b1, 0).
    static void moments(double b0, double b1, double m[3]) {
        b0 = std::max(b0, b1);
        const double w   = 1.0 / std::sqrt(std::max(1.0, 2.0 * b0));   // peak half-width in c
        const double kap = std::asinh(1.0 / w);      // so that c(1) == w*sinh(kap) == 1
        double den = 0.0, n0 = 0.0, n2 = 0.0;
        for (int k = 0; k <= NQ; ++k) {
            const double z  = (double)k / NQ;
            const double c  = w * std::sinh(kap * z);
            const double dc = w * kap * std::cosh(kap * z);
            const double wt = ((k == 0 || k == NQ) ? 1.0 : ((k & 1) ? 4.0 : 2.0)) / (3.0 * NQ);
            const double s2 = std::max(0.0, 1.0 - c * c);
            double k0, k1; besselK01(0.5 * s2 * (b0 - b1), k0, k1);
            const double e = std::exp(-c * c * b0) * wt * dc;
            den += e * k0;
            n2  += e * k0 * c * c;
            n0  += e * s2 * 0.5 * (k0 + k1);
        }
        if (!(den > 0.0)) { m[0] = 1.0; m[1] = m[2] = 0.0; return; }
        m[0] = n0 / den;
        m[2] = n2 / den;
        m[1] = std::max(0.0, 1.0 - m[0] - m[2]);
    }

    // (A', B') -> the eigenvalue triple it stands for.
    static void tauAt(double Ap, double Bp, double tau[3]) {
        const double B = Bp * 0.5 * (1.0 - Ap);
        tau[2] = std::max(0.0, (1.0 - Ap - 2.0 * B) / 3.0);
        tau[1] = tau[2] + B;
        tau[0] = tau[1] + Ap;
    }

    // The concentrations a node's two log-variables stand for.  See the header: y[0] is the
    // GAP b_0 - b_1 and y[1] the floor b_1, which is what keeps the tau_3 = 0 edge solvable.
    static void decode(double yg, double yf, double& b0, double& b1) {
        b1 = std::exp(yf) - 1.0;
        b0 = b1 + std::exp(yg) - 1.0;
    }

    // Damped Newton in (log(1+gap), log(1+floor)).  `y` is both warm start and result.
    static void invert(const double tau[3], double y[2]) {
        const double yMax = std::log(1.0 + BMAX);
        auto resid = [&](const double yy[2], double r[2]) {
            double b0, b1, m[3];
            decode(yy[0], yy[1], b0, b1);
            moments(b0, b1, m);
            r[0] = m[0] - tau[0];
            r[1] = m[2] - tau[2];
            return std::fabs(r[0]) + std::fabs(r[1]);
        };
        double r[2], e = resid(y, r);
        for (int it = 0; it < 60 && e > 1e-11; ++it) {
            double J[2][2];
            for (int j = 0; j < 2; ++j) {                 // central differences
                const double h = 1e-5 * std::max(1.0, std::fabs(y[j]));
                double yp[2] = {y[0], y[1]}, ym[2] = {y[0], y[1]}, rp[2], rm[2];
                yp[j] += h; ym[j] = std::max(0.0, ym[j] - h);
                resid(yp, rp); resid(ym, rm);
                const double d = yp[j] - ym[j];
                J[0][j] = (rp[0] - rm[0]) / d;
                J[1][j] = (rp[1] - rm[1]) / d;
            }
            const double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
            if (std::fabs(det) < 1e-18) break;
            const double dy0 = -( J[1][1] * r[0] - J[0][1] * r[1]) / det;
            const double dy1 = -(-J[1][0] * r[0] + J[0][0] * r[1]) / det;
            double step = 1.0;
            bool improved = false;
            for (int ls = 0; ls < 30; ++ls) {             // backtracking line search
                double yt[2] = {std::min(yMax, std::max(0.0, y[0] + step * dy0)),
                                std::min(yMax, std::max(0.0, y[1] + step * dy1))};
                double rt[2];
                const double et = resid(yt, rt);
                if (et < e) { y[0] = yt[0]; y[1] = yt[1]; r[0] = rt[0]; r[1] = rt[1];
                              e = et; improved = true; break; }
                step *= 0.5;
            }
            if (!improved) break;
        }
    }

    BinghamTable() {
        y0.assign((size_t)N * N, 0.0f);
        y1.assign((size_t)N * N, 0.0f);
        for (int j = 0; j < N; ++j) {
            double y[2] = {0.0, 0.0};                    // warm start: isotropic, walk the row
            for (int i = 0; i < N; ++i) {
                double tau[3];
                tauAt((double)i / (N - 1), (double)j / (N - 1), tau);
                invert(tau, y);
                y0[(size_t)j * N + i] = (float)y[0];
                y1[(size_t)j * N + i] = (float)y[1];
            }
        }
    }

    // Concentrations for a descending eigenvalue triple.  b[2] is 0 by convention.
    void lookup(const double tau[3], double b[3]) const {
        const double Ap = std::min(1.0, std::max(0.0, tau[0] - tau[1]));
        const double bm = 0.5 * (1.0 - Ap);
        const double Bp = bm > 1e-12 ? std::min(1.0, std::max(0.0, (tau[1] - tau[2]) / bm)) : 0.0;
        const double fi = Ap * (N - 1), fj = Bp * (N - 1);
        int i0 = (int)fi; if (i0 > N - 2) i0 = N - 2; if (i0 < 0) i0 = 0;
        int j0 = (int)fj; if (j0 > N - 2) j0 = N - 2; if (j0 < 0) j0 = 0;
        const double fu = std::min(1.0, std::max(0.0, fi - i0));
        const double fv = std::min(1.0, std::max(0.0, fj - j0));
        const size_t r0 = (size_t)j0 * N, r1 = r0 + N;
        auto bil = [&](const std::vector<float>& t) {
            const double v0 = t[r0 + i0] + fu * (t[r0 + i0 + 1] - t[r0 + i0]);
            const double v1 = t[r1 + i0] + fu * (t[r1 + i0 + 1] - t[r1 + i0]);
            return v0 + fv * (v1 - v0);
        };
        decode(std::max(0.0, bil(y0)), std::max(0.0, bil(y1)), b[0], b[1]);
        b[2] = 0.0;
    }
};

inline const BinghamTable& bingham() { static BinghamTable t; return t; }

// ---------------------------------------------------------------------------------------
// The reconstructed orientation distribution of one cell.
struct FurODF {
    Vec3   e[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double tau[3]{1.0 / 3, 1.0 / 3, 1.0 / 3};   // T's eigenvalues, descending
    double b[3]{0, 0, 0};                       // Bingham concentrations, b[2] == 0
    double lam[3]{0, 0, 0};                     // Lambda_i = b[0] - b[i] >= 0
    double om[3]{1, 1, 1};                      // ACG proposal Omega_i = 1 + 2 lam_i / bk
    double sa[3]{1, 1, 1};                      // 1/sqrt(Omega_i): the proposal's semi-axes
    double invM = 1.0;                          // 1 / the rejection bound
    Vec3   mean{0, 0, 0};                       // unit mean tangent, or zero if unoriented
    double pAlign = 0.5;                        // P(sampled tangent points along `mean`)

    static FurODF fromCell(const FurCell& fc) {
        FurODF o;
        o.mean   = furMeanDir(fc.mdir);
        // A two-delta population with a fraction q along +v and 1-q along -v has
        // |mean| = |2q - 1|, so q = (1 + |mean|) / 2 EXACTLY.  That rule is also right at
        // both ends of the general case: a perfectly combed cell always aligns, an
        // unoriented one is a coin flip, and it is monotone in between.
        o.pAlign = 0.5 * (1.0 + furMeanCoherence(fc.mdir));
        const double m[6] = {fc.txx, fc.tyy, furTzz(fc), fc.txy, fc.txz, fc.tyz};
        symEigen3(m, o.e, o.tau);
        double s = 0.0;
        for (int i = 0; i < 3; ++i) { o.tau[i] = std::max(0.0, o.tau[i]); s += o.tau[i]; }
        if (s > 1e-30) for (int i = 0; i < 3; ++i) o.tau[i] /= s;
        bingham().lookup(o.tau, o.b);
        for (int i = 0; i < 3; ++i) o.lam[i] = std::max(0.0, o.b[0] - o.b[i]);
        // Kent's tuning constant: the unique root in (0,3] of SUM 1/(bk + 2 lam_i) = 1.
        double lo = 1e-9, hi = 3.0;
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (lo + hi);
            double f = 0.0;
            for (int i = 0; i < 3; ++i) f += 1.0 / (mid + 2.0 * o.lam[i]);
            if (f > 1.0) lo = mid; else hi = mid;
        }
        const double bk = 0.5 * (lo + hi);
        for (int i = 0; i < 3; ++i) {
            o.om[i] = 1.0 + 2.0 * o.lam[i] / bk;
            o.sa[i] = 1.0 / std::sqrt(o.om[i]);
        }
        o.invM = 1.0 / (std::exp(-(3.0 - bk) * 0.5) * std::pow(3.0 / bk, 1.5));
        return o;
    }

    // The axis part of the ODF is antipodally symmetric by construction, so `sample` draws an
    // AXIS and then chooses a sign against the cell's first moment.  Keeping the two steps
    // separate is what makes the second moment and the sign independent: `T` is untouched by
    // any choice made here, because `t t^T == (-t)(-t)^T`.
    Vec3 orient(const Vec3& t, double us) const {
        const double dm = dot(t, mean);
        if (dm == 0.0) return (us < 0.5) ? t * -1.0 : t;   // unoriented cell: a fair coin
        return ((dm > 0.0) == (us < pAlign)) ? t : t * -1.0;
    }

    // One ACG proposal from four uniforms (Box-Muller, one ordinate discarded).
    Vec3 proposal(double u0, double u1, double u2, double u3) const {
        const double r0 = std::sqrt(-2.0 * std::log(std::max(1e-300, u0)));
        const double t0 = 2.0 * hair::kPi * u1;
        const double r1 = std::sqrt(-2.0 * std::log(std::max(1e-300, u2)));
        const double t1 = 2.0 * hair::kPi * u3;
        const Vec3 v = e[0] * (sa[0] * r0 * std::cos(t0))
                     + e[1] * (sa[1] * r0 * std::sin(t0))
                     + e[2] * (sa[2] * r1 * std::cos(t1));
        const double L = length(v);
        return (L > 1e-300) ? v * (1.0 / L) : e[0];
    }

    // A tangent drawn from the ODF.  `u` seeds the first attempt (4 Gaussian uniforms, an
    // acceptance uniform and a sign uniform); retries come from `rng`.  Splitting it that way
    // lets the self-test drive the common case quasi-randomly and so measure the MODEL's
    // error rather than the sampler's, while production just passes six fresh uniforms.
    Vec3 sample(const double u[6], Pcg32& rng, int tries = 64) const {
        double a0 = u[0], a1 = u[1], a2 = u[2], a3 = u[3], ac = u[4];
        Vec3 t = e[0];
        for (int i = 0; i < tries; ++i) {
            t = proposal(a0, a1, a2, a3);
            double q = 0.0, w = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double p = dot(t, e[k]);
                q += lam[k] * p * p;
                w += om[k]  * p * p;
            }
            if (ac < std::exp(-q) * std::pow(std::max(1e-300, w), 1.5) * invM) break;
            a0 = rng.uniform(); a1 = rng.uniform(); a2 = rng.uniform(); a3 = rng.uniform();
            ac = rng.uniform();
        }
        return orient(t, u[5]);
    }

    Vec3 sample(Pcg32& rng) const {
        const double u[6] = {rng.uniform(), rng.uniform(), rng.uniform(),
                             rng.uniform(), rng.uniform(), rng.uniform()};
        return sample(u, rng);
    }
};

// A tangent drawn from `ODF(t) * sin(angle(d,t))` — the distribution of the tangent a ray
// travelling along `d` ACTUALLY meets, since a fiber's projected area carries that factor.
// Rejection is the right tool: the weight is a probability already (sin is in [0,1]), so
// there is nothing to normalise, and the acceptance rate is exactly <sin>, the quantity the
// grid's Jensen step approximates.  `tries` bounds a pathological cell (every tangent
// parallel to d); falling through returns the last draw, which is then a ~0-cross-section
// fiber and contributes nothing.
inline Vec3 sampleTangentXsec(const FurODF& odf, const Vec3& d, Pcg32& rng, int tries = 32) {
    Vec3 t{1, 0, 0};
    for (int i = 0; i < tries; ++i) {
        t = odf.sample(rng);
        const double c = dot(t, d);
        const double sn = std::sqrt(std::max(0.0, 1.0 - c * c));
        if (rng.uniform() < sn) return t;
    }
    return t;
}

// ---------------------------------------------------------------------------------------
// A VIRTUAL FIBER HIT.  `hair::Bcsdf` measures the impact parameter h from the surface
// normal at the entry point; in a volume there is no surface, so h is drawn uniformly (a ray
// crossing a cylinder has a uniform offset) and the normal that WOULD have produced it is
// reconstructed.  In the normal plane the angle between the outward normal and the outgoing
// direction is gamma_o with h = sin(gamma_o), so rotating `wo`'s perpendicular component by
// -asin(h) about the tangent lands exactly on it.  `-checkfurvol` §4 round-trips this
// through `hair::hFromHit` to prove the reconstruction is the inverse it claims to be.
inline Vec3 fiberNormalFor(const Vec3& t, const Vec3& wo, double h) {
    const Vec3 ax = normalize(t);
    Vec3 oPerp = wo - ax * dot(ax, wo);
    const double ol = length(oPerp);
    if (ol < 1e-12) { Vec3 a, b; onb(ax, a, b); return a; }   // end-on: any perpendicular
    oPerp = oPerp * (1.0 / ol);
    const Vec3 bi = cross(ax, oPerp);
    const double a = std::asin(hair::clampd(h, -1.0, 1.0));
    return oPerp * std::cos(a) - bi * std::sin(a);
}

}  // namespace furvol
