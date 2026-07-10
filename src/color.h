// Spectral -> XYZ -> sRGB. CIE 1931 CMFs via the analytic multi-Gaussian fit
// (Wyman, Sloan, Shirley, JCGT 2013) — accurate and compact, no big data table.
#pragma once
#include <cmath>
#include <algorithm>
#include "linalg.h"

inline double gaussPiece(double x, double mu, double s1, double s2) {
    double t = (x - mu) * ((x < mu) ? s1 : s2);
    return std::exp(-0.5 * t * t);
}
inline double cieX(double w) {
    return 0.362 * gaussPiece(w, 442.0, 0.0624, 0.0374)
         + 1.056 * gaussPiece(w, 599.8, 0.0264, 0.0323)
         - 0.065 * gaussPiece(w, 501.1, 0.0490, 0.0382);
}
inline double cieY(double w) {
    return 0.821 * gaussPiece(w, 568.8, 0.0213, 0.0247)
         + 0.286 * gaussPiece(w, 530.9, 0.0613, 0.0322);
}
inline double cieZ(double w) {
    return 1.217 * gaussPiece(w, 437.0, 0.0845, 0.0278)
         + 0.681 * gaussPiece(w, 459.0, 0.0385, 0.0725);
}

// Wavelength integration range (nm).
constexpr double LAMBDA_MIN = 360.0;
constexpr double LAMBDA_MAX = 830.0;

// Integral of ybar over the range — luminance normaliser so an equal-energy
// spectrum has Y = 1. Computed once.
inline double cieYIntegral() {
    static double v = [] {
        double s = 0;
        for (double w = LAMBDA_MIN; w <= LAMBDA_MAX; w += 1.0) s += cieY(w);
        return s;
    }();
    return v;
}

inline Vec3 xyzToLinearSrgb(const Vec3& xyz) {
    // Standard XYZ (D65) -> linear sRGB matrix.
    return {
         3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
        -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
         0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z
    };
}

inline double srgbGamma(double c) {
    c = std::max(0.0, c);
    return (c <= 0.0031308) ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}

// Inverse of srgbGamma: decode a display-encoded sRGB channel to linear.
inline double srgbToLinear(double c) {
    c = std::clamp(c, 0.0, 1.0);
    return (c <= 0.04045) ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}
