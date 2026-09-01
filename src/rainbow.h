// Airy-theory rainbow phase function for participating (water-droplet) media.
// ---------------------------------------------------------------------------
// A generic haze scatters via the single-parameter Henyey-Greenstein (HG) lobe
// (render.h `hgPhase`), which is far too smooth to ever show a rainbow. A real
// rainbow lives in the *angular fine structure* of a WATER DROPLET's scattering
// phase function p(theta, lambda): light that enters a spherical drop, reflects
// internally (p-1) times and exits piles up at a stationary deflection angle (the
// Descartes / rainbow angle), and because water's index n(lambda) disperses, that
// angle shifts with wavelength -> the colours fan out.
//
// We model this with AIRY THEORY of the rainbow (van de Hulst; Adam, Phys. Rep.
// 356 (2002)): near the rainbow the exiting wavefront is cubic, so Fresnel-Kirchhoff
// diffraction of it gives an Airy-function intensity profile. This is the standard
// tractable-yet-physical rainbow model (full Lorenz-Mie needs thousands of partial
// waves for a mm drop). It reproduces, from first principles:
//   * the exact primary (~138 deg) and secondary (~129 deg) scattering angles,
//   * their DISPERSION and reversed colour order (primary: red outer; secondary flipped),
//   * Alexander's dark band between the bows,
//   * SUPERNUMERARY arcs (the Airy side-maxima) and their (lambda/a)^(2/3) spacing,
//   * the fogbow limit: as the droplet radius shrinks the structure broadens and
//     desaturates (few/no supernumeraries), exactly as a real fogbow does.
//
// The generator tabulates a normalised spectral phase function p(mu, lambda) on a
// (wavelength x cos-angle) grid plus a per-wavelength CDF for importance sampling,
// so the render side just evaluates / samples the table like any phase function.
// The table integrates to 1 over the sphere per wavelength (2*pi * integral_-1^1 p dmu = 1),
// matching the convention of hgPhase, so pdf(omega) == p(mu).
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <functional>
#include "linalg.h"     // Vec3, onb, PI
#include "rng.h"        // Pcg32
#include "color.h"      // LAMBDA_MIN / LAMBDA_MAX

namespace rainbow {

// --- Airy function Ai(x) ----------------------------------------------------
// Tabulated once by integrating the Airy ODE  y'' = x*y  outward from x=0 with the
// exact seeds Ai(0), Ai'(0) via RK4. Stable and accurate over the modest range we
// need (the oscillatory x<0 region carries the supernumeraries; x>0 decays fast).
inline double airyAi(double x) {
    static const struct AiTable {
        const double X0 = -30.0, X1 = 12.0, DX = 0.002;
        int n = 0;
        std::vector<double> v;   // Ai at X0 + i*DX
        AiTable() {
            const double Ai0  =  0.3550280538878172;
            const double dAi0 = -0.2588194037928068;
            n = (int)std::lround((X1 - X0) / DX) + 1;
            v.assign(n, 0.0);
            int i0 = (int)std::lround((0.0 - X0) / DX);
            v[i0] = Ai0;
            // RK4 on state (y, y'), derivative f(x,y,yp) = (yp, x*y).
            auto step = [](double x, double y, double yp, double h, double& yo, double& ypo) {
                double k1y = yp,            k1p = x * y;
                double k2y = yp + 0.5*h*k1p, k2p = (x+0.5*h)*(y+0.5*h*k1y);
                double k3y = yp + 0.5*h*k2p, k3p = (x+0.5*h)*(y+0.5*h*k2y);
                double k4y = yp + h*k3p,     k4p = (x+h)*(y+h*k3y);
                yo  = y  + (h/6.0)*(k1y + 2*k2y + 2*k3y + k4y);
                ypo = yp + (h/6.0)*(k1p + 2*k2p + 2*k3p + k4p);
            };
            // Integrate toward +x (decaying — RK4 is fine over this bounded span).
            { double x = 0.0, y = Ai0, yp = dAi0;
              for (int i = i0; i < n - 1; ++i) { double yo, ypo; step(x, y, yp, DX, yo, ypo);
                  x += DX; y = yo; yp = ypo; v[i+1] = y; } }
            // Integrate toward -x (oscillatory).
            { double x = 0.0, y = Ai0, yp = dAi0;
              for (int i = i0; i > 0; --i) { double yo, ypo; step(x, y, yp, -DX, yo, ypo);
                  x -= DX; y = yo; yp = ypo; v[i-1] = y; } }
        }
        double at(double x) const {
            if (x >= X1) return 0.0;
            if (x <= X0) {
                // Below the table, use the exact large-|x| asymptotic rather than 0. Ai(-t) ~
                // t^(-1/4)/sqrt(pi) * sin((2/3) t^(3/2) + pi/4), relative error < 1e-4 for t > 30.
                // Returning 0 here used to clip the whole bow interior beyond z = -30 to nothing,
                // which mattered because the Ai^2 envelope only decays as |z|^(-1/2).
                double t = -x;
                return std::sin((2.0 / 3.0) * t * std::sqrt(t) + 0.25 * PI)
                     / (std::sqrt(std::sqrt(t)) * std::sqrt(PI));
            }
            double f = (x - X0) / DX; int i = (int)f; double t = f - i;
            if (i < 0) return v[0]; if (i >= n - 1) return v[n - 1];
            return v[i] * (1.0 - t) + v[i + 1] * t;
        }
    } table;
    return table.at(x);
}

// --- Tail integral of Ai(x)^2 -----------------------------------------------
// S(u) = integral from u to +inf of Ai(t)^2 dt. This is what makes the droplet-size
// average below both exact and cheap: the mean of Ai^2 over an interval [a,b] is
// (S(a) - S(b)) / (b - a), so a quadrature cell that straddles many Airy oscillations
// gets their exact average instead of an aliased point sample. (Point-sampling
// Ai(z*s)^2 at 64 sizes would alias badly: at z = -80 the phase sweeps ~1400 rad
// across the distribution.)
//
// Tabulated by integrating the Ai table downward from X1; below X0 the oscillation
// averages to the envelope 1/(2*pi*sqrt|t|), whose integral is sqrt|u|/pi.
inline double airyAi2Tail(double u) {
    static const struct Ai2Table {
        const double X0 = -30.0, X1 = 12.0, DX = 0.002;
        int n = 0;
        std::vector<double> s;   // S at X0 + i*DX
        Ai2Table() {
            n = (int)std::lround((X1 - X0) / DX) + 1;
            s.assign(n, 0.0);
            double acc = 0.0;
            double prev = airyAi(X1); prev *= prev;
            for (int i = n - 2; i >= 0; --i) {
                double x = X0 + i * DX;
                double cur = airyAi(x); cur *= cur;
                acc += 0.5 * (cur + prev) * DX;
                s[i] = acc; prev = cur;
            }
        }
        double at(double u) const {
            if (u >= X1) return 0.0;
            if (u <= X0) return s[0] + (std::sqrt(-u) - std::sqrt(-X0)) / PI;
            double f = (u - X0) / DX; int i = (int)f; double t = f - i;
            if (i < 0) return s[0]; if (i >= n - 1) return s[n - 1];
            return s[i] * (1.0 - t) + s[i + 1] * t;
        }
    } table;
    return table.at(u);
}

// Mean of Ai(t)^2 over t in [a, b] (any order). Exact, from the tail integral.
inline double airyAi2Mean(double a, double b) {
    if (b < a) std::swap(a, b);
    double d = b - a;
    if (d < 1e-9) { double v = airyAi(0.5 * (a + b)); return v * v; }
    return (airyAi2Tail(a) - airyAi2Tail(b)) / d;
}

// --- Rainbow geometry for one wavelength / scattering order -----------------
// order p: number of chords through the drop = (internal reflections)+1. p=2 is the
// primary bow (1 internal reflection), p=3 the secondary (2). All computed by direct
// numerical extremum-finding of the folded deflection angle, so there is no hand
// algebra to get wrong; it yields the Descartes angle theta_rb, the fold curvature h
// (rad/rad^2) and the lit-side sign.
struct BowGeom {
    bool   valid = false;
    double thetaRb = 0.0;   // rainbow (Descartes) scattering angle, radians in [0,pi]
    double h       = 0.0;   // |d^2 theta_fold / di^2| at the caustic (fold curvature)
    double sign    = -1.0;  // +/-1: lit (two-ray) side is sign*(theta-thetaRb) < 0
};

// Fold an unbounded deflection D (radians) into a scattering angle in [0, pi].
inline double foldDeflection(double D) {
    double m = std::fmod(D, 2.0 * PI);
    if (m < 0) m += 2.0 * PI;
    return (m > PI) ? (2.0 * PI - m) : m;
}
// Unfolded total deflection for incidence i, index n, order p.
inline double deflection(double i, double n, int p) {
    double s = std::sin(i) / n;
    if (s > 1.0) s = 1.0;
    double r = std::asin(s);
    return 2.0 * (i - r) + (double)(p - 1) * (PI - 2.0 * r);
}

inline BowGeom bowGeometry(double n, int p) {
    BowGeom g;
    // Rainbow incidence: cos^2 i_c = (n^2 - 1)/(p^2 - 1).
    double c2 = (n * n - 1.0) / ((double)(p * p) - 1.0);
    if (c2 < 0.0 || c2 > 1.0) return g;               // no bow for this order
    double ic = std::acos(std::sqrt(c2));
    double thetaRb = foldDeflection(deflection(ic, n, p));
    // Fold curvature from a symmetric quadratic fit near i_c.
    double d = 1e-3;                                   // rad
    double tm = foldDeflection(deflection(ic - d, n, p));
    double tp = foldDeflection(deflection(ic + d, n, p));
    double h  = std::fabs((tp + tm - 2.0 * thetaRb) / (d * d));
    // Lit side: on which side of theta_rb do the two real rays fall?
    double sign = ((tp > thetaRb) ? -1.0 : +1.0);      // z = sign*(theta-thetaRb) < 0 == lit
    g.valid = (h > 1e-6);
    g.thetaRb = thetaRb; g.h = h; g.sign = sign;
    return g;
}

// --- Droplet SIZE DISTRIBUTION (polydispersity) ------------------------------
//
// Real rain is not one droplet size, and that single fact governs how a rainbow looks. The
// bow's ANGLE is geometric — it does not depend on droplet size at all — but the Airy fold
// scale does: K = (2/h)^(1/3) * (2*pi*a/lambda)^(2/3), so z = (theta - theta_rb) * K scales as
// a^(2/3). Averaging Ai(z)^2 over a spread of a therefore smears the supernumerary train
// toward its local mean while leaving theta_rb exactly where it is. That is why a shower shows
// one clean bow and a fog (narrow distribution, tiny drops) shows supernumeraries.
//
// WHAT THIS REPLACED, AND WHY. Through 0.198.0 the stand-in for polydispersity was
// `supernumerary off`, which held the principal-lobe PEAK flat for all z < -1.02:
//
//     if (!supernumerary && z < -1.02) { peak = Ai(-1.01879); I = peak*peak; }
//
// That is not a size average, it is a plateau. A real average decays with the Airy envelope
// 1/(2*pi*sqrt|z|), and since `z < -1.02` is everything more than ~0.28 deg inside theta_rb AT
// ANY DROPLET SIZE, the clamp held the entire ~40 deg interior of the bow at full arc
// brightness. Measured (scraps/_supernum_row.py, one real 2048-bin row at droplet_um 300 /
// 550 nm, HG background included): 3.8x too bright at z=-5 rising to 14.9x at z=-80, ~9x the
// correct integrated interior mass, and — because each lambda row is renormalised to integrate
// to 1 — an arc/interior contrast of 1.01x where the correct answer is 6.40x. It did not
// render a bow at all; it rendered a uniformly bright disc with a thin coloured rim. See
// known-issues.md, "phase rainbow", problem 3.
//
// THE MODEL. A gamma distribution over droplet radius, n(a) ~ a^(k-1) exp(-Lambda a), which is
// the standard meteorological DSD; k = 1 is exactly Marshall-Palmer's exponential. Two
// different moments of it matter and they are not the same distribution:
//
//   * how much each size SCATTERS: sigma_sca ~ 2*pi*a^2, so the size mix that a photon
//     actually samples is weighted a^2 -> gamma of shape K2 = k + 2.
//   * how bright each size's BOW is: the Airy rainbow's differential cross-section scales as
//     a^(7/3) (van de Hulst), one power of a^(1/3) steeper than the total. So the mix that
//     shapes the bow is weighted a^(7/3) -> gamma of shape Kb = K2 + 1/3. Big drops make
//     disproportionately bright bows; that is not a fudge, it is the reason a heavy shower's
//     bow is more vivid than drizzle's.
//
// USER PARAMETER. `dispersion` is the relative standard deviation of the SCATTERING-weighted
// radius distribution, which is the one with the direct physical meaning ("how varied are the
// drops that are doing the scattering"). rel.sd = 1/sqrt(K2), so K2 = 1/dispersion^2, and
//
//   dispersion = 0        monodisperse — one exact droplet size. A laboratory abstraction;
//                         full supernumerary train, the pre-0.199.0 `supernumerary on` look.
//   dispersion = 0.577    Marshall-Palmer (k = 1, exponential). THE DEFAULT. Note this comes
//                         out INDEPENDENT of rain rate: MP's Lambda(R) moves the mean drop
//                         size but not the shape, so the relative width is 58% in drizzle and
//                         in a downpour alike. Verified in scraps/_supernum_mp.py at R = 1, 5
//                         and 25 mm/h.
//   dispersion -> 0.707   the ceiling (k -> 0). Beyond it the underlying n(a) has no mode.
//
// `droplet_um` remains the scattering-weighted MEAN radius, so it means the same thing at any
// dispersion and the monodisperse limit is continuous.
//
// HOW IT IS COMPUTED. The average is separable: of everything in the Airy profile only the fold
// scale K depends on a, so the size average of Ai(z)^2 is sum_i w_i * <Ai^2 over cell i>, with
// the cell spanning [z*s_lo, z*s_hi] and s = (a/abar)^(2/3) — a fixed quadrature built once per
// table build and reused for every (lambda, mu) cell.
//
// The cells are AVERAGED, not point-sampled, via airyAi2Mean(). That is not a refinement, it is
// required: at z = -80 the Airy phase sweeps ~1400 rad across a Marshall-Palmer distribution, so
// 64 point samples would alias into fixed-pattern ringing across the bow interior. Averaging each
// cell exactly makes the quadrature converge on the cell count of the *density* (smooth, 64 is
// plenty) instead of on the cell count of the *oscillation* (thousands). Cost is 2 table lookups
// per node per cell at build time; the render cost is unchanged, since the render only ever sees
// the finished (lambda x mu) table.
inline constexpr int kSizeNodes = 64;      // quadrature cells across the distribution
inline constexpr double kDispMax = 0.70;   // K2 = 1/d^2 > 2, i.e. k > 0 (n(a) still has a mode)

// --- Parameters -------------------------------------------------------------
struct Params {
    double dropletRadius_m = 0.5e-3;  // droplet radius (m). ~0.5mm rain; ~10um -> fogbow.
                                      // The SCATTERING-WEIGHTED MEAN radius when dispersion>0.
    double dispersion      = 0.577;   // relative sd of the scattering-weighted size
                                      // distribution; 0 = monodisperse, 0.577 = Marshall-
                                      // Palmer rain (the default). See the note above.
    double gForward       = 0.55;     // HG anisotropy of the smooth forward-scatter background.
    double rainbowStrength = 1.0;     // relative weight of the Airy bows vs the background.
    bool   secondary       = true;    // include the p=3 secondary bow.
    double secondaryRatio  = 0.43;    // secondary brightness relative to the primary.
    // n(lambda) of the droplet material (water by default). Callable Spectrum-like.
    std::function<double(double)> nOf = [](double lambdaNm) {
        double um = lambdaNm * 1e-3;                   // Cauchy fit for water (data/glass/water.glass)
        return 1.324 + 0.003 / (um * um);
    };
};

// --- Tabulated spectral phase function --------------------------------------
class RainbowPhase {
public:
    void build(const Params& prm) {
        p_ = prm;
        buildSizeQuadrature();
        lam0_ = LAMBDA_MIN; dLam_ = 5.0;
        nLam_ = (int)std::lround((LAMBDA_MAX - LAMBDA_MIN) / dLam_) + 1;
        nMu_  = 2048;
        pdf_.assign((size_t)nLam_ * nMu_, 0.0);
        cdf_.assign((size_t)nLam_ * nMu_, 0.0);
        const double dMu = 2.0 / (nMu_ - 1);
        for (int li = 0; li < nLam_; ++li) {
            double lambda = lam0_ + li * dLam_;
            double n = p_.nOf(lambda);
            BowGeom bow1 = bowGeometry(n, 2);
            BowGeom bow2 = p_.secondary ? bowGeometry(n, 3) : BowGeom{};
            double kSize = std::pow(2.0 * PI * p_.dropletRadius_m / (lambda * 1e-9), 2.0 / 3.0);
            double* row = &pdf_[(size_t)li * nMu_];
            for (int mi = 0; mi < nMu_; ++mi) {
                double mu = -1.0 + mi * dMu;
                if (mu > 1.0) mu = 1.0;
                double theta = std::acos(std::max(-1.0, std::min(1.0, mu)));
                // Smooth forward-scatter background (HG), keeps fog reading as fog.
                double gg = p_.gForward;
                double hgD = 1.0 + gg * gg - 2.0 * gg * mu; if (hgD < 1e-9) hgD = 1e-9;
                double bg = (1.0 - gg * gg) / (4.0 * PI * hgD * std::sqrt(hgD));
                // Airy bows.
                double bows = 0.0;
                bows += airyBow(theta, bow1, kSize, 1.0);
                if (p_.secondary) bows += airyBow(theta, bow2, kSize, p_.secondaryRatio);
                row[mi] = bg + p_.rainbowStrength * bows;
            }
            // Normalise this wavelength row so 2*pi * integral p dmu = 1, and build CDF.
            double integ = 0.0;
            for (int mi = 0; mi < nMu_ - 1; ++mi)
                integ += 0.5 * (row[mi] + row[mi + 1]) * dMu;      // trapezoid over mu
            double norm = (integ > 1e-30) ? 1.0 / (2.0 * PI * integ) : 1.0;
            double* cdf = &cdf_[(size_t)li * nMu_];
            double acc = 0.0; cdf[0] = 0.0;
            for (int mi = 0; mi < nMu_; ++mi) {
                row[mi] *= norm;
                if (mi > 0) acc += 0.5 * (row[mi] + row[mi - 1]) * dMu * (2.0 * PI); // == prob mass
                cdf[mi] = acc;
            }
            // Force exact [0,1].
            double tot = cdf[nMu_ - 1] > 1e-30 ? cdf[nMu_ - 1] : 1.0;
            for (int mi = 0; mi < nMu_; ++mi) cdf[mi] /= tot;
        }
        built_ = true;
    }

    bool built() const { return built_; }

    // --- Accessors for GPU upload: the device mirrors this exact (lambda x mu) table
    // and its per-lambda CDF, so its rainbow media match the CPU tracer bit-for-bit
    // (modulo float rounding). Layout is row-major [li*nMu + mi]; mu = -1 + mi*dMu with
    // dMu = 2/(nMu-1); lambda = lam0 + li*dLam.
    int    nLam() const { return nLam_; }
    int    nMu()  const { return nMu_; }
    double lam0() const { return lam0_; }
    double dLam() const { return dLam_; }
    const std::vector<double>& pdfTable() const { return pdf_; }
    const std::vector<double>& cdfTable() const { return cdf_; }

    // Phase value p(mu) at wavelength lambda (nm). Bilinear in (lambda, mu).
    double eval(double cosTheta, double lambda) const {
        double fl = (lambda - lam0_) / dLam_;
        int li = (int)std::floor(fl); double tl = fl - li;
        li = std::max(0, std::min(nLam_ - 2, li)); tl = std::max(0.0, std::min(1.0, tl));
        double fm = (cosTheta + 1.0) / (2.0 / (nMu_ - 1));
        int mi = (int)std::floor(fm); double tm = fm - mi;
        mi = std::max(0, std::min(nMu_ - 2, mi)); tm = std::max(0.0, std::min(1.0, tm));
        auto P = [&](int L, int M) { return pdf_[(size_t)L * nMu_ + M]; };
        double a = P(li, mi) * (1 - tm) + P(li, mi + 1) * tm;
        double b = P(li + 1, mi) * (1 - tm) + P(li + 1, mi + 1) * tm;
        return a * (1 - tl) + b * tl;
    }

    // Importance-sample a scattered direction about propagation `wi` at wavelength
    // lambda; returns the direction and sets pdfOut = p(mu) = pdf over solid angle.
    Vec3 sample(const Vec3& wi, double lambda, Pcg32& rng, double& pdfOut) const {
        int li = (int)std::lround((lambda - lam0_) / dLam_);
        li = std::max(0, std::min(nLam_ - 1, li));
        const double* cdf = &cdf_[(size_t)li * nMu_];
        double u = rng.uniform();
        // Binary search the CDF for mu.
        int lo = 0, hi = nMu_ - 1;
        while (lo + 1 < hi) { int m = (lo + hi) >> 1; if (cdf[m] < u) lo = m; else hi = m; }
        double c0 = cdf[lo], c1 = cdf[hi];
        double t = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.0;
        double dMu = 2.0 / (nMu_ - 1);
        double mu = -1.0 + (lo + t) * dMu;
        mu = std::max(-1.0, std::min(1.0, mu));
        double sinT = std::sqrt(std::max(0.0, 1.0 - mu * mu));
        double phi = 2.0 * PI * rng.uniform();
        Vec3 tb, bb; onb(wi, tb, bb);
        Vec3 dir = normalize(tb * (sinT * std::cos(phi)) + bb * (sinT * std::sin(phi)) + wi * mu);
        pdfOut = eval(mu, lambda);
        return dir;
    }

    // Console self-test: print the primary/secondary Descartes angles across the
    // spectrum and a couple of sanity checks on the Airy function + normalisation.
    static void selfTest();

private:
    // Build the droplet-size quadrature from p_.dispersion. See the long note above.
    // x = a / abar, abar = the SCATTERING-weighted (a^2-weighted) mean radius, so the
    // a^2-weighted distribution is gamma(shape K2, scale 1/K2) and has mean exactly 1 —
    // which is what makes `droplet_um` mean the same thing at every dispersion.
    // The BOW-brightness weighting is one power of a^(1/3) steeper, gamma(Kb = K2 + 1/3),
    // and that is the density the cells carry.
    void buildSizeQuadrature() {
        sLo_.clear(); sHi_.clear(); sW_.clear();
        double d = p_.dispersion;
        if (!(d > 0.0)) {                       // monodisperse: one degenerate cell
            sLo_.push_back(1.0); sHi_.push_back(1.0); sW_.push_back(1.0);
            return;
        }
        if (d > kDispMax) d = kDispMax;
        const double K2 = 1.0 / (d * d);
        const double Kb = K2 + 1.0 / 3.0;
        const double mean = Kb / K2, sd = std::sqrt(Kb) / K2;
        double x0 = mean - 8.0 * sd; if (x0 < 0.0) x0 = 0.0;
        double x1 = mean + 8.0 * sd;
        const double dx = (x1 - x0) / kSizeNodes;
        // Unnormalised gamma(Kb, 1/K2) density, in log form so large Kb cannot overflow.
        auto logDens = [&](double x) { return (Kb - 1.0) * std::log(x) - K2 * x; };
        double peak = logDens(std::max(1e-12, (Kb - 1.0) / K2));   // mode, for scaling
        double sum = 0.0;
        std::vector<double> wRaw((size_t)kSizeNodes);
        for (int i = 0; i < kSizeNodes; ++i) {
            double xm = x0 + (i + 0.5) * dx;
            double w = (xm > 1e-12) ? std::exp(logDens(xm) - peak) * dx : 0.0;
            wRaw[(size_t)i] = w; sum += w;
        }
        if (!(sum > 0.0)) {                     // degenerate guard
            sLo_.push_back(1.0); sHi_.push_back(1.0); sW_.push_back(1.0);
            return;
        }
        sLo_.reserve(kSizeNodes); sHi_.reserve(kSizeNodes); sW_.reserve(kSizeNodes);
        for (int i = 0; i < kSizeNodes; ++i) {
            double w = wRaw[(size_t)i] / sum;
            if (w < 1e-9) continue;             // skip cells that cannot matter
            double xa = x0 + i * dx, xb = xa + dx;
            sLo_.push_back(std::pow(std::max(0.0, xa), 2.0 / 3.0));
            sHi_.push_back(std::pow(std::max(0.0, xb), 2.0 / 3.0));
            sW_.push_back(w);
        }
        // Renormalise after the cull so the bow keeps exactly the monodisperse total.
        double tot = 0.0; for (double w : sW_) tot += w;
        if (tot > 0.0) for (double& w : sW_) w /= tot;
    }

    // Airy intensity of one bow at scattering angle theta (rad), averaged over the
    // droplet-size distribution. Weight w scales it.
    double airyBow(double theta, const BowGeom& bow, double kSize, double w) const {
        if (!bow.valid) return 0.0;
        double h = bow.h < 1e-4 ? 1e-4 : bow.h;
        // Airy fold scale K = ( 2*(2*pi*a/lambda)^2 / h )^(1/3) = (2/h)^(1/3) * kSize,
        // where kSize = (2*pi*a/lambda)^(2/3) is passed in. Units: rad^-1, so z below
        // is dimensionless. This bakes in the physical (lambda/a)^(2/3) bow scaling.
        double K = std::pow(2.0 / h, 1.0 / 3.0) * kSize;
        double z = bow.sign * (theta - bow.thetaRb) * K;
        if (sW_.size() == 1 && sLo_[0] == sHi_[0]) {          // monodisperse fast path
            double ai = airyAi(z * sLo_[0]);
            return w * ai * ai;
        }
        // Size average. z scales as a^(2/3), so cell i covers z in [z*sLo, z*sHi] and
        // contributes its exact mean of Ai^2 over that span (see airyAi2Mean).
        double I = 0.0;
        for (size_t i = 0; i < sW_.size(); ++i) {
            double a = z * sLo_[i], b = z * sHi_[i];
            if (a > 12.0 && b > 12.0) continue;               // deep in the dark side
            I += sW_[i] * airyAi2Mean(a, b);
        }
        return w * I;
    }

    Params p_;
    bool built_ = false;
    double lam0_ = 360.0, dLam_ = 5.0;
    int nLam_ = 0, nMu_ = 0;
    std::vector<double> pdf_, cdf_;
    // Droplet-size quadrature: cell edges in s = (a/abar)^(2/3), and the cell's mass.
    std::vector<double> sLo_, sHi_, sW_;
};

inline void RainbowPhase::selfTest() {
    std::printf("[rainbow selftest] Airy: Ai(0)=%.6f (exp 0.355028)  Ai(-1)=%.6f (exp 0.535561)  "
                "Ai(1)=%.6f (exp 0.135292)  Ai(2)=%.6f (exp 0.034924)\n",
                airyAi(0.0), airyAi(-1.0), airyAi(1.0), airyAi(2.0));
    struct { double lam; const char* name; } cols[] = {
        {400, "violet"}, {486, "blue"}, {550, "green"}, {620, "orange"}, {700, "red"} };
    Params prm;
    for (auto& c : cols) {
        double n = prm.nOf(c.lam);
        BowGeom b1 = bowGeometry(n, 2), b2 = bowGeometry(n, 3);
        std::printf("[rainbow selftest] lambda=%3.0fnm (%-6s) n=%.4f  primary=%.2f deg  secondary=%.2f deg\n",
                    c.lam, c.name, n, b1.thetaRb * 180.0 / PI, b2.thetaRb * 180.0 / PI);
    }
    // Airy tail integral: S(-inf..u) must match the envelope asymptote sqrt|u|/pi, and
    // the local mean of Ai^2 over a wide window must match the envelope 1/(2 pi sqrt|z|).
    // integral_0^inf Ai^2 = Ai'(0)^2 exactly, since d/dx[x Ai^2 - Ai'^2] = Ai^2.
    std::printf("[rainbow selftest] Ai^2 tail S(0)=%.6f (exp %.6f = Ai'(0)^2)  "
                "mean Ai^2 over [-52,-48]=%.6f (envelope %.6f)\n",
                airyAi2Tail(0.0), 0.2588194037928068 * 0.2588194037928068,
                airyAi2Mean(-52.0, -48.0), 1.0 / (2.0 * PI * std::sqrt(50.0)));
    // Droplet size distribution: the default is Marshall-Palmer, whose scattering-weighted
    // relative width is 0.577. Its size average must decay with the Airy envelope where the
    // pre-0.199.0 flat clamp held the principal-lobe peak, Ai(-1.019)^2 = 0.2869, forever.
    // Expect: within a lobe of the peak the average is near the monodisperse value; far inside
    // the bow it tracks the Airy envelope 1/(2 pi sqrt|z|) (a touch above it, by Jensen on the
    // convex s^(-1/2)), NOT the clamp, which is 3.9x too bright by z=-5 and 15x by z=-80.
    std::printf("[rainbow selftest] dispersion=%.3f (0.577 = Marshall-Palmer)  "
                "size-averaged Ai^2 (flat clamp would give 0.286928 at every z):\n",
                prm.dispersion);
    {
        RainbowPhase probe; probe.p_ = prm; probe.buildSizeQuadrature();
        BowGeom flat; flat.valid = true; flat.thetaRb = 0.0; flat.h = 2.0; flat.sign = 1.0;
        double K = std::pow(2.0 / flat.h, 1.0 / 3.0);
        for (double z : {-1.02, -5.0, -12.0, -30.0, -80.0}) {
            double v = probe.airyBow(z / K, flat, 1.0, 1.0);
            double mono = airyAi(z); mono *= mono;
            std::printf("[rainbow selftest]   z=%6.1f  avg=%.6f  mono=%.6f  envelope=%.6f\n",
                        z, v, mono, 1.0 / (2.0 * PI * std::sqrt(-z)));
        }
    }
    // Normalisation check for a mid droplet: each per-lambda phase slice must integrate
    // to 1 over the sphere, i.e. 2*pi*integral p(mu) dmu == 1.
    RainbowPhase rp; rp.build(prm);
    for (double lam : {450.0, 550.0, 650.0}) {
        double integ = 0.0; int N = 4000;
        for (int i = 0; i < N; ++i) {
            double mu = -1.0 + 2.0 * (i + 0.5) / N;
            integ += rp.eval(mu, lam) * (2.0 / N);
        }
        std::printf("[rainbow selftest] lambda=%.0fnm  2*pi*integral p dmu = %.4f (expect ~1.0)\n",
                    lam, 2.0 * PI * integ);
    }
}

} // namespace rainbow
