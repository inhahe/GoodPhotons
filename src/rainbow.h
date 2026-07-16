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
            if (x <= X0) return 0.0;
            if (x >= X1) return 0.0;
            double f = (x - X0) / DX; int i = (int)f; double t = f - i;
            if (i < 0) return v[0]; if (i >= n - 1) return v[n - 1];
            return v[i] * (1.0 - t) + v[i + 1] * t;
        }
    } table;
    return table.at(x);
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

// --- Parameters -------------------------------------------------------------
struct Params {
    double dropletRadius_m = 0.5e-3;  // droplet radius (m). ~0.5mm rain; ~10um -> fogbow.
    double gForward       = 0.55;     // HG anisotropy of the smooth forward-scatter background.
    double rainbowStrength = 1.0;     // relative weight of the Airy bows vs the background.
    bool   secondary       = true;    // include the p=3 secondary bow.
    bool   supernumerary   = true;    // keep the Airy side-maxima (else use only the main lobe).
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
    // Airy intensity of one bow at scattering angle theta (rad). Weight w scales it.
    double airyBow(double theta, const BowGeom& bow, double kSize, double w) const {
        if (!bow.valid) return 0.0;
        double h = bow.h < 1e-4 ? 1e-4 : bow.h;
        // Airy fold scale K = ( 2*(2*pi*a/lambda)^2 / h )^(1/3) = (2/h)^(1/3) * kSize,
        // where kSize = (2*pi*a/lambda)^(2/3) is passed in. Units: rad^-1, so z below
        // is dimensionless. This bakes in the physical (lambda/a)^(2/3) bow scaling.
        double K = std::pow(2.0 / h, 1.0 / 3.0) * kSize;
        double z = bow.sign * (theta - bow.thetaRb) * K;
        double ai = airyAi(z);
        double I = ai * ai;
        if (!p_.supernumerary && z < -1.02) {
            // Collapse the side-maxima: hold the principal-lobe peak for z below its max.
            double peak = airyAi(-1.01879); I = peak * peak;
        }
        return w * I;
    }

    Params p_;
    bool built_ = false;
    double lam0_ = 360.0, dLam_ = 5.0;
    int nLam_ = 0, nMu_ = 0;
    std::vector<double> pdf_, cdf_;
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
