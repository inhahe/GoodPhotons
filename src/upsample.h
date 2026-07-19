// RGB -> reflectance spectrum upsampling (Jakob & Hanika, "A Low-Dimensional
// Function Space for Efficient Spectral Upsampling", EG 2019).
//
// A reflectance is modelled as a sigmoid of a quadratic in wavelength:
//     S(λ) = 1/2 + p / (2·sqrt(1 + p²)),   p = c0·t² + c1·t + c2,
// with t a normalized wavelength. This always yields S ∈ (0,1) (a physical
// reflectance) and is smooth. Given a target linear-sRGB colour we fit the three
// coefficients (c0,c1,c2) at load time with a few Gauss-Newton steps so that the
// spectrum, viewed under D65 through the CIE observer, reproduces that colour.
//
// This replaces the crude three-lobe placeholder; it round-trips sRGB accurately
// (see checkUpsample()). No large precomputed table — the per-colour fit is a
// handful of 3×3 solves, done once per material/texel, not per photon.
#pragma once
#include <cmath>
#include <array>
#include "color.h"
#include "spectrum.h"
#include "lights.h"

namespace upsample {

// Normalized wavelength coordinate for the quadratic (keeps coefficients O(1)).
inline double tOf(double lambda) { return (lambda - 595.0) / 235.0; }

inline double sigmoid(double p) { return 0.5 + 0.5 * p / std::sqrt(1.0 + p * p); }
inline double dSigmoid(double p) { double s = 1.0 + p * p; return 0.5 / (s * std::sqrt(s)); }

// Reflectance value at λ for coefficients c.
inline double reflAt(const std::array<double, 3>& c, double lambda) {
    double t = tOf(lambda);
    double p = c[0] * t * t + c[1] * t + c[2];
    return sigmoid(p);
}

// Precomputed integration weights: wX/Y/Z(λ) = k·D65(λ)·CMF(λ)·dλ, sampled on a
// fixed grid, with k chosen so a unit reflectance integrates to the D65 white
// point (Y = 1). Built once.
struct Basis {
    static constexpr int N = 95;          // (830-360)/5 + 1
    static constexpr double step = 5.0;
    double lam[N];
    double wX[N], wY[N], wZ[N];
    Basis() {
        Spectrum d65 = daylight(6504.0);
        double kY = 0.0;
        for (int i = 0; i < N; ++i) {
            double w = LAMBDA_MIN + i * step;
            lam[i] = w;
            double e = std::max(0.0, d65(w));
            wX[i] = e * cieX(w) * step;
            wY[i] = e * cieY(w) * step;
            wZ[i] = e * cieZ(w) * step;
            kY += wY[i];
        }
        double k = (kY > 0) ? 1.0 / kY : 1.0;
        for (int i = 0; i < N; ++i) { wX[i] *= k; wY[i] *= k; wZ[i] *= k; }
    }
    // XYZ of a unit-vs-reflectance under D65.
    void integrate(const std::array<double, 3>& c, double& X, double& Y, double& Z) const {
        X = Y = Z = 0.0;
        for (int i = 0; i < N; ++i) {
            double s = reflAt(c, lam[i]);
            X += s * wX[i]; Y += s * wY[i]; Z += s * wZ[i];
        }
    }
};

inline const Basis& basis() { static Basis b; return b; }

// Linear sRGB (D65) -> XYZ. Inverse of color.h's xyzToLinearSrgb matrix.
inline void linSrgbToXyz(double r, double g, double b, double& X, double& Y, double& Z) {
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b;
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b;
}

// Fit sigmoid coefficients for a linear-sRGB colour via Gauss-Newton.
inline std::array<double, 3> fit(double r, double g, double b) {
    const Basis& B = basis();
    double tX, tY, tZ; linSrgbToXyz(r, g, b, tX, tY, tZ);
    std::array<double, 3> c{0.0, 0.0, 0.0};   // start at S ≡ 0.5 (mid gray)
    for (int iter = 0; iter < 40; ++iter) {
        // Residual and 3×3 Jacobian dXYZ/dc.
        double X = 0, Y = 0, Z = 0;
        double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (int i = 0; i < B.N; ++i) {
            double t = tOf(B.lam[i]);
            double p = c[0] * t * t + c[1] * t + c[2];
            double s = sigmoid(p), ds = dSigmoid(p);
            X += s * B.wX[i]; Y += s * B.wY[i]; Z += s * B.wZ[i];
            double dc[3] = {t * t, t, 1.0};
            for (int j = 0; j < 3; ++j) {
                J[0][j] += ds * dc[j] * B.wX[i];
                J[1][j] += ds * dc[j] * B.wY[i];
                J[2][j] += ds * dc[j] * B.wZ[i];
            }
        }
        double rX = X - tX, rY = Y - tY, rZ = Z - tZ;
        if (rX * rX + rY * rY + rZ * rZ < 1e-14) break;
        // Solve J·Δ = residual (3×3) via Cramer's rule.
        double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
                   - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
                   + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
        if (std::fabs(det) < 1e-18) break;
        double rhs[3] = {rX, rY, rZ};
        double d[3];
        for (int col = 0; col < 3; ++col) {
            double M[3][3];
            for (int a = 0; a < 3; ++a) for (int bb = 0; bb < 3; ++bb) M[a][bb] = J[a][bb];
            for (int a = 0; a < 3; ++a) M[a][col] = rhs[a];
            double dc = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                      - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                      + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
            d[col] = dc / det;
        }
        // Damped step for stability on saturated colours.
        const double relax = 1.0;
        c[0] -= relax * d[0]; c[1] -= relax * d[1]; c[2] -= relax * d[2];
    }
    return c;
}

} // namespace upsample

// Build a reflectance Spectrum from a linear-sRGB triple (Jakob-Hanika fit).
inline Spectrum rgbToReflectanceJH(double r, double g, double b) {
    r = std::clamp(r, 0.0, 1.0); g = std::clamp(g, 0.0, 1.0); b = std::clamp(b, 0.0, 1.0);
    std::array<double, 3> c = upsample::fit(r, g, b);
    return [c](double lambda) { return upsample::reflAt(c, lambda); };
}

// Reduce an arbitrary reflectance Spectrum to a linear-sRGB triple by integrating
// it under D65 through the CIE observer (the same basis rgbToReflectanceJH inverts,
// so the two round-trip). Used to interpolate colour stops in linear RGB. Not
// clamped — the caller may lerp several of these before upsampling back.
inline Vec3 reflectanceToLinearSrgbD65(const Spectrum& refl) {
    const upsample::Basis& B = upsample::basis();
    double X = 0, Y = 0, Z = 0;
    for (int i = 0; i < B.N; ++i) {
        double s = refl ? std::max(0.0, refl(B.lam[i])) : 0.0;
        X += s * B.wX[i]; Y += s * B.wY[i]; Z += s * B.wZ[i];
    }
    return xyzToLinearSrgb({X, Y, Z});
}
