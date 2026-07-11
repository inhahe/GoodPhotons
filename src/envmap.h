// Image-based environment lighting (Phase 3c, increment 2).
//
// An EnvMap turns an equirectangular (lat-long) RGB image — typically a Radiance
// .hdr, but any format the Texture loader handles — into an infinite directional
// emitter. Each texel's linear RGB is upsampled to a physical emission spectrum
//   L_texel(lambda) = scale * S_JH(chroma)(lambda) * illum(lambda)
// where S_JH is the Jakob-Hanika sigmoid reflectance fit of the texel's chroma and
// illum is a normalised 6504 K illuminant (the same illuminant the JH basis is fit
// under, so S_JH*illum reproduces the texel colour when integrated against the CIE
// observer). `scale` carries the (HDR) brightness. This mirrors PBRT's
// RGBIlluminantSpectrum treatment of RGB emitters.
//
// For importance sampling we build a 2D luminance distribution over the map
// (luminance * sin(theta) per texel), so directions are drawn in proportion to the
// radiance they carry — the key variance reduction for peaked skies. The class
// exposes:
//   sample(u1,u2) -> (dir, pdf_omega)   importance-sampled emission/NEE direction
//   pdf(dir)      -> pdf_omega          matching solid-angle density
//   radiance(dir, lambda)               spectral radiance in a direction
//   xyz(dir)                            directly-viewed background XYZ (= integral
//                                       of CIE(lambda)*L(dir,lambda) dlambda, the
//                                       same convention as the constant-env envXYZ)
//   avgSpd(lambda)                      sin(theta)-weighted mean radiance spectrum,
//                                       used for the emitter power / wavelength CDF
//
// Direction convention: theta measured from +y (up), v = theta/pi with row 0 at the
// top (straight up); phi = atan2(z, x), u = phi/(2pi)+0.5, with an optional rotation
// about the vertical axis. dirToUV and uvToDir are exact inverses so sampling and
// evaluation stay consistent.
#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include "linalg.h"
#include "color.h"
#include "spectrum.h"
#include "lights.h"
#include "upsample.h"
#include "texture.h"

// -------- 1D / 2D piecewise-constant distributions (Pharr, Jakob & Humphreys) ----
struct Distribution1D {
    std::vector<double> func, cdf;
    double funcInt = 0.0;
    void build(const double* f, int n) {
        func.assign(f, f + n);
        cdf.assign(n + 1, 0.0);
        for (int i = 1; i <= n; ++i) cdf[i] = cdf[i - 1] + func[i - 1] / n;
        funcInt = cdf[n];
        if (funcInt == 0.0) { for (int i = 1; i <= n; ++i) cdf[i] = (double)i / n; }
        else                { for (int i = 1; i <= n; ++i) cdf[i] /= funcInt; }
    }
    // Continuous sample in [0,1); returns the density (relative to funcInt) and the
    // integer offset of the chosen bin.
    double sampleContinuous(double u, double& pdf, int& off) const {
        int n = (int)func.size();
        int lo = 0, hi = n;                     // find last i with cdf[i] <= u
        while (lo + 1 < hi) { int m = (lo + hi) / 2; if (cdf[m] <= u) lo = m; else hi = m; }
        off = lo;
        double du = u - cdf[lo];
        double d = cdf[lo + 1] - cdf[lo];
        if (d > 0) du /= d;
        pdf = (funcInt > 0) ? func[lo] / funcInt : 0.0;
        return (lo + du) / (double)n;
    }
};

struct Distribution2D {
    std::vector<Distribution1D> cond;   // one conditional over u (columns) per row v
    Distribution1D marg;                // marginal over v (rows)
    int nu = 0, nv = 0;
    void build(const std::vector<double>& f, int nu_, int nv_) {
        nu = nu_; nv = nv_;
        cond.resize(nv);
        for (int v = 0; v < nv; ++v) cond[v].build(&f[(size_t)v * nu], nu);
        std::vector<double> m(nv);
        for (int v = 0; v < nv; ++v) m[v] = cond[v].funcInt;
        marg.build(m.data(), nv);
    }
    // Sample (u,v) in [0,1)^2; pdf is the density over the unit square.
    void sampleContinuous(double u0, double u1, double& u, double& v, double& pdf) const {
        int vo = 0, uo = 0; double dv = 0, du = 0;
        v = marg.sampleContinuous(u1, dv, vo);
        u = cond[vo].sampleContinuous(u0, du, uo);
        pdf = du * dv;
    }
    double pdf(double u, double v) const {
        int iu = std::clamp((int)(u * nu), 0, nu - 1);
        int iv = std::clamp((int)(v * nv), 0, nv - 1);
        if (marg.funcInt == 0.0) return 0.0;
        return cond[iv].func[iu] / marg.funcInt;   // density over the unit square
    }
};

struct EnvMap {
    int w = 0, h = 0;
    double rotOffset = 0.0;                 // horizontal rotation in [0,1) turns
    std::vector<std::array<double, 3>> coeff;   // per-texel JH chroma coefficients
    std::vector<double> scaleT;                 // per-texel brightness scale
    std::vector<Vec3>   xyzT;                    // per-texel spectral XYZ (background)
    Distribution2D dist;                         // luminance*sin(theta) sampler
    // Mean-radiance spectrum (for the emitter power + wavelength importance CDF).
    std::array<double, 3> avgCoeff{0, 0, 0};
    double avgScale = 0.0;
    // Normalised illuminant: illum(lambda) with integral against ybar == 1.
    Spectrum illumSpec = daylight(6504.0);
    double illumNorm = 1.0;

    double illumAt(double lambda) const { return illumNorm * std::max(0.0, illumSpec(lambda)); }

    // ---- construction -------------------------------------------------------
    bool load(const std::string& path, double rotateDeg, double intensity, std::string& err) {
        Texture tex;
        tex.encoding = TexEncoding::sRGB;   // LDR inputs honour sRGB; .hdr/.pfm force Linear
        if (!tex.load(path, err)) return false;
        if (!tex.valid()) { err = "environment map has no pixels: " + path; return false; }
        w = tex.w; h = tex.h;
        rotOffset = rotateDeg / 360.0;

        // Normalise the illuminant so a unit reflectance emits Y = 1 (1 nm grid,
        // matching cieYIntegral()'s convention used across the renderer).
        {
            double yi = 0.0;
            for (double lam = LAMBDA_MIN; lam <= LAMBDA_MAX; lam += 1.0)
                yi += std::max(0.0, illumSpec(lam)) * cieY(lam);
            illumNorm = (yi > 0.0) ? 1.0 / yi : 1.0;
        }

        const size_t nT = (size_t)w * h;
        coeff.resize(nT); scaleT.resize(nT); xyzT.resize(nT);
        std::vector<double> img(nT);        // luminance * sin(theta) for the sampler

        // Precompute a CIE*illum integration grid (5 nm) for per-texel XYZ.
        struct GridW { double lam, wx, wy, wz, il; };
        std::vector<GridW> grid;
        for (double lam = LAMBDA_MIN; lam <= LAMBDA_MAX + 1e-9; lam += 5.0)
            grid.push_back({lam, cieX(lam) * 5.0, cieY(lam) * 5.0, cieZ(lam) * 5.0, illumAt(lam)});

        Vec3 avgRgb{0, 0, 0};
        double avgW = 0.0;
        for (int row = 0; row < h; ++row) {
            double theta = (row + 0.5) / h * PI;
            double sinT = std::sin(theta);
            for (int col = 0; col < w; ++col) {
                size_t i = (size_t)row * w + col;
                Vec3 c = tex.rgb[i] * intensity;
                c.x = std::max(0.0, c.x); c.y = std::max(0.0, c.y); c.z = std::max(0.0, c.z);
                double m = std::max(c.x, std::max(c.y, c.z));
                double s = 2.0 * m;                       // PBRT-style: chroma in [0,0.5]
                std::array<double, 3> cf;
                if (m > 0.0) cf = upsample::fit(c.x / s, c.y / s, c.z / s);
                else         { cf = {0, 0, 0}; s = 0.0; }
                coeff[i] = cf; scaleT[i] = s;
                // Per-texel spectral XYZ = integral CIE(lambda)*L(lambda) dlambda.
                Vec3 xyz{0, 0, 0};
                for (const GridW& g : grid) {
                    double L = s * upsample::reflAt(cf, g.lam) * g.il;
                    xyz.x += L * g.wx; xyz.y += L * g.wy; xyz.z += L * g.wz;
                }
                xyzT[i] = xyz;
                double lum = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
                img[i] = lum * sinT;
                avgRgb += c * sinT; avgW += sinT;
            }
        }
        dist.build(img, w, h);

        // Mean-radiance spectrum from the sin(theta)-weighted average colour.
        if (avgW > 0.0) avgRgb = avgRgb * (1.0 / avgW);
        double am = std::max(avgRgb.x, std::max(avgRgb.y, avgRgb.z));
        if (am > 0.0) { avgScale = 2.0 * am; avgCoeff = upsample::fit(avgRgb.x / avgScale, avgRgb.y / avgScale, avgRgb.z / avgScale); }
        else          { avgScale = 0.0; avgCoeff = {0, 0, 0}; }
        return true;
    }

    // ---- direction <-> texture-space mapping --------------------------------
    void dirToUV(const Vec3& d, double& u, double& v) const {
        double y = std::clamp(d.y, -1.0, 1.0);
        double theta = std::acos(y);
        double phi = std::atan2(d.z, d.x);
        v = theta / PI;
        u = phi / (2.0 * PI) + 0.5 - rotOffset;
        u -= std::floor(u);                        // wrap to [0,1)
    }
    Vec3 uvToDir(double u, double v) const {
        double theta = v * PI;
        double phi = (u - 0.5 + rotOffset) * 2.0 * PI;
        double st = std::sin(theta);
        return Vec3{st * std::cos(phi), std::cos(theta), st * std::sin(phi)};
    }

    // ---- sampling / evaluation ----------------------------------------------
    // Importance-sample an incoming direction; pdfW is the solid-angle density.
    Vec3 sample(double u0, double u1, double& pdfW) const {
        double u, v, mapPdf;
        dist.sampleContinuous(u0, u1, u, v, mapPdf);
        double theta = v * PI;
        double sinT = std::sin(theta);
        pdfW = (sinT > 0.0) ? mapPdf / (2.0 * PI * PI * sinT) : 0.0;
        return uvToDir(u, v);
    }
    double pdf(const Vec3& d) const {
        double u, v; dirToUV(d, u, v);
        double theta = v * PI;
        double sinT = std::sin(theta);
        if (sinT <= 0.0) return 0.0;
        return dist.pdf(u, v) / (2.0 * PI * PI * sinT);
    }

    // Nearest-texel accessors (evaluation is per-photon/per-miss; nearest keeps it
    // cheap and matches the piecewise-constant sampler exactly).
    size_t texelOf(const Vec3& d) const {
        double u, v; dirToUV(d, u, v);
        int col = std::clamp((int)(u * w), 0, w - 1);
        int row = std::clamp((int)(v * h), 0, h - 1);
        return (size_t)row * w + col;
    }
    double radiance(const Vec3& d, double lambda) const {
        size_t i = texelOf(d);
        return scaleT[i] * upsample::reflAt(coeff[i], lambda) * illumAt(lambda);
    }
    Vec3 xyz(const Vec3& d) const { return xyzT[texelOf(d)]; }

    double avgSpd(double lambda) const {
        return avgScale * upsample::reflAt(avgCoeff, lambda) * illumAt(lambda);
    }
};
