// Analytic physical sky (Preetham et al. 2002, "A Practical Analytic Model for
// Daylight"). Generates an equirectangular (lat-long) *linear-RGB* image of a clear
// daylight sky for a given turbidity and sun direction, including a spectrally
// attenuated solar disk baked into the sky texels. The image is fed straight into an
// EnvMap (see envmap.h), so it reuses the existing importance sampler, per-texel
// Jakob-Hanika spectral upsampling, and the GPU env upload — an analytic sky lights
// the scene exactly like an image-based HDRI does, on both the CPU and GPU paths.
//
// Direction convention matches EnvMap::uvToDir: row 0 = straight up (+y), theta grows
// downward to the horizon at v=0.5 and the nadir at v=1; phi = atan2(z, x).
//
// The Perez five-parameter distribution
//   F(theta, gamma) = (1 + A e^{B/cos theta}) (1 + C e^{D gamma} + E cos^2 gamma)
// is evaluated for luminance Y and the CIE xy chromaticities, scaled by the turbidity-
// and sun-elevation-dependent zenith values (Yz, xz, yz). The result is an xyY sky
// radiance that we convert to XYZ then linear sRGB. The solar disk is added on top:
// its spectrum is a 5778 K blackbody attenuated by Rayleigh + Angstrom-aerosol optical
// depth along the sun's air mass, so low suns redden (sunset) automatically. Absolute
// magnitudes are physical (cd/m^2), then the whole image is normalised so the mean
// above-horizon sky luminance equals `intensity` (default 1) — keeping the physical
// sun/sky ratio while landing in a sane exposure range for the tone mapper.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "linalg.h"
#include "color.h"
#include "spectrum.h"
#include "lights.h"

namespace sky {

// One Perez lobe. Coefficients A..E are linear in turbidity for each of Y, x, y.
struct Perez { double A, B, C, D, E; };

inline double perezF(const Perez& p, double cosTheta, double gamma) {
    // Clamp cosTheta away from 0 so the horizon term stays finite.
    double ct = std::max(cosTheta, 1e-3);
    return (1.0 + p.A * std::exp(p.B / ct)) *
           (1.0 + p.C * std::exp(p.D * gamma) + p.E * std::cos(gamma) * std::cos(gamma));
}

// Turbidity-dependent Perez coefficients (Preetham Table, luminance + chromaticities).
inline Perez perezY(double T) { return { 0.1787*T - 1.4630, -0.3554*T + 0.4275, -0.0227*T + 5.3251, 0.1206*T - 2.5771, -0.0670*T + 0.3703 }; }
inline Perez perezX(double T) { return { -0.0193*T - 0.2592, -0.0665*T + 0.0008, -0.0004*T + 0.2125, -0.0641*T - 0.8989, -0.0033*T + 0.0452 }; }
inline Perez perezYy(double T){ return { -0.0167*T - 0.2608, -0.0950*T + 0.0092, -0.0079*T + 0.2102, -0.0441*T - 1.6537, -0.0109*T + 0.0529 }; }

// Zenith luminance (cd/m^2) and chromaticity for turbidity T and solar zenith thetaS.
inline double zenithLuminance(double T, double thetaS) {
    double chi = (4.0/9.0 - T/120.0) * (PI - 2.0*thetaS);
    double Yz = (4.0453*T - 4.9710) * std::tan(chi) - 0.2155*T + 2.4192;  // kcd/m^2
    return std::max(0.0, Yz) * 1000.0;
}
inline double zenithChromaX(double T, double ts) {
    double t2 = ts*ts, t3 = t2*ts;
    return (0.00166*t3 - 0.00375*t2 + 0.00209*ts + 0.0)      * T*T
         + (-0.02903*t3 + 0.06377*t2 - 0.03202*ts + 0.00394) * T
         + (0.11693*t3 - 0.21196*t2 + 0.06052*ts + 0.25886);
}
inline double zenithChromaY(double T, double ts) {
    double t2 = ts*ts, t3 = t2*ts;
    return (0.00275*t3 - 0.00610*t2 + 0.00317*ts + 0.0)      * T*T
         + (-0.04214*t3 + 0.08970*t2 - 0.04153*ts + 0.00516) * T
         + (0.15346*t3 - 0.26756*t2 + 0.06670*ts + 0.26688);
}

// xyY -> linear sRGB (via XYZ). Y is absolute luminance; x,y chromaticity.
inline Vec3 xyYToLinearSrgb(double x, double y, double Y) {
    if (y < 1e-6) return Vec3{0,0,0};
    double X = (x / y) * Y;
    double Z = ((1.0 - x - y) / y) * Y;
    return xyzToLinearSrgb({X, Y, Z});
}

// Kasten-Young relative optical air mass for solar zenith thetaS (radians).
inline double airMass(double thetaS) {
    double zdeg = thetaS * 180.0 / PI;
    double denom = std::cos(thetaS) + 0.50572 * std::pow(std::max(0.0, 96.07995 - zdeg), -1.6364);
    return (denom > 1e-6) ? 1.0 / denom : 40.0;
}

// Spectral radiance (relative) of the solar disk: a 5778 K blackbody attenuated by
// Rayleigh + Angstrom aerosol optical depth over `m` air masses. Returns linear-sRGB
// chromaticity+luminance ratio (unnormalised XYZ integral).
inline Vec3 sunDiskXYZ(double T, double m) {
    // Angstrom turbidity coefficient beta from Preetham's turbidity relation.
    double beta = 0.04608 * T - 0.04586;
    if (beta < 0.0) beta = 0.0;
    const double alpha = 1.3;                 // Angstrom exponent
    Spectrum bb = blackbody(5778.0);
    double X = 0, Y = 0, Z = 0;
    for (double lam = LAMBDA_MIN; lam <= LAMBDA_MAX + 1e-9; lam += 5.0) {
        double um = lam / 1000.0;
        double tauR = 0.008735 * std::pow(um, -4.08);       // Rayleigh
        double tauA = beta * std::pow(um, -alpha);          // aerosol
        double trans = std::exp(-m * (tauR + tauA));
        double L = std::max(0.0, bb(lam)) * trans;
        X += L * cieX(lam) * 5.0; Y += L * cieY(lam) * 5.0; Z += L * cieZ(lam) * 5.0;
    }
    return Vec3{X, Y, Z};
}

// Generate the equirectangular linear-RGB sky. `sunDir` need not be normalised.
// `groundAlbedo` tints the below-horizon hemisphere (a flat lit ground). `intensity`
// scales the normalised mean sky luminance (default 1).
inline std::vector<Vec3> generatePreethamSky(int w, int h, Vec3 sunDir, double turbidity,
                                             double groundAlbedo, double intensity) {
    double sl = std::sqrt(dot(sunDir, sunDir));
    Vec3 sd = (sl > 1e-9) ? sunDir * (1.0/sl) : Vec3{0, 1, 0};
    double thetaS = std::acos(std::clamp(sd.y, -1.0, 1.0));   // solar zenith angle
    double T = std::clamp(turbidity, 1.7, 10.0);

    Perez pY = perezY(T), pX = perezX(T), pYy = perezYy(T);
    double Yz = zenithLuminance(T, thetaS);
    double xz = zenithChromaX(T, thetaS);
    double yz = zenithChromaY(T, thetaS);
    double denY = perezF(pY, 1.0, thetaS);
    double denX = perezF(pX, 1.0, thetaS);
    double denYy = perezF(pYy, 1.0, thetaS);

    std::vector<Vec3> img((size_t)w * h, Vec3{0,0,0});

    // First pass: physical sky luminance/chroma per above-horizon texel, and its
    // sin(theta)-weighted mean luminance (for normalisation) and mean RGB (ground).
    double sumL = 0.0, sumW = 0.0;
    Vec3 sumRgb{0,0,0};
    for (int row = 0; row < h; ++row) {
        double theta = (row + 0.5) / h * PI;         // angle from +y
        if (theta >= PI * 0.5) continue;             // below horizon handled later
        double cosTheta = std::cos(theta);
        double sinT = std::sin(theta);
        for (int col = 0; col < w; ++col) {
            double phi = ((col + 0.5) / w - 0.5) * 2.0 * PI;
            Vec3 dir{sinT * std::cos(phi), cosTheta, sinT * std::sin(phi)};
            double cg = std::clamp(dot(dir, sd), -1.0, 1.0);
            double gamma = std::acos(cg);
            double Y = Yz  * perezF(pY,  cosTheta, gamma) / denY;
            double xx = xz * perezF(pX,  cosTheta, gamma) / denX;
            double yy = yz * perezF(pYy, cosTheta, gamma) / denYy;
            Vec3 rgb = xyYToLinearSrgb(xx, yy, std::max(0.0, Y));
            rgb.x = std::max(0.0, rgb.x); rgb.y = std::max(0.0, rgb.y); rgb.z = std::max(0.0, rgb.z);
            img[(size_t)row * w + col] = rgb;
            sumL += Y * sinT; sumW += sinT;
            sumRgb += rgb * sinT;
        }
    }
    double meanL = (sumW > 0.0) ? sumL / sumW : 1.0;
    Vec3 meanRgb = (sumW > 0.0) ? sumRgb * (1.0 / sumW) : Vec3{0,0,0};
    double norm = (meanL > 1e-9) ? intensity / meanL : intensity;

    // Ground hemisphere: a flat Lambertian lit by the (normalised) mean sky colour.
    Vec3 ground = meanRgb * (norm * std::clamp(groundAlbedo, 0.0, 1.0));

    // Second pass: apply normalisation; fill the ground; bake the solar disk.
    // The solar disk: an atmosphere-attenuated 5778 K blackbody (reddens at low sun),
    // scaled to a physical clear-air disk luminance (~1.6e9 cd/m^2 extraterrestrial
    // times the photopic transmittance), then folded into the sky's normalisation so
    // the physical sun/sky ratio is preserved.
    Vec3 sunXYZ = sunDiskXYZ(T, airMass(thetaS));      // attenuated (relative units)
    Vec3 bbXYZ  = sunDiskXYZ(0.0, 0.0);                // unattenuated reference (m=0)
    double trans = (bbXYZ.y > 1e-9) ? std::clamp(sunXYZ.y / bbXYZ.y, 0.0, 1.0) : 1.0;
    double L_sun_phys = 1.6e9 * trans;                 // physical disk luminance cd/m^2
    Vec3 sunRgb{0,0,0};
    if (sd.y > 0.0 && sunXYZ.y > 1e-9) {
        // xyzToLinearSrgb preserves luminance == XYZ.y, so rescale to L_sun_phys.
        sunRgb = xyzToLinearSrgb(sunXYZ) * ((L_sun_phys / sunXYZ.y) * norm);
        sunRgb.x = std::max(0.0, sunRgb.x); sunRgb.y = std::max(0.0, sunRgb.y); sunRgb.z = std::max(0.0, sunRgb.z);
    }
    const double sunAngRadius = 0.5 * PI/180.0 * 1.2;   // ~0.53 deg disk, slightly softened
    for (int row = 0; row < h; ++row) {
        double theta = (row + 0.5) / h * PI;
        double cosTheta = std::cos(theta);
        double sinT = std::sin(theta);
        bool below = theta >= PI * 0.5;
        for (int col = 0; col < w; ++col) {
            size_t i = (size_t)row * w + col;
            if (below) { img[i] = ground; continue; }
            img[i] = img[i] * norm;
            if (sd.y > 0.0) {
                double phi = ((col + 0.5) / w - 0.5) * 2.0 * PI;
                Vec3 dir{sinT * std::cos(phi), cosTheta, sinT * std::sin(phi)};
                double gamma = std::acos(std::clamp(dot(dir, sd), -1.0, 1.0));
                if (gamma < sunAngRadius) {
                    // Soft limb: full disk inside 0.8R, smooth to 0 at R.
                    double t = std::clamp((sunAngRadius - gamma) / (0.2 * sunAngRadius), 0.0, 1.0);
                    img[i] = img[i] + sunRgb * t;
                }
            }
        }
    }
    return img;
}

} // namespace sky
