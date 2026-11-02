// Interference reflectance of a coated surface: a single thin film (Airy) and an N-layer
// stack (Abeles). Pure functions of the optics, shared by the tracers (render.h) and the raster
// preview (raster.h), which has no tracer to call.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include "scene.h"

inline double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

// --- Thin-film interference reflectance (iridescence) ------------------------
// A thin dielectric film (index n1, thickness d nanometres) coats a substrate of
// index n2, with incident medium n0. The beams reflected off the top (n0|n1) and
// bottom (n1|n2) interfaces interfere; the round-trip optical-path phase
//   phi = 4*pi*n1*d*cos(theta1) / lambda
// makes the reflectance oscillate with wavelength AND angle -> structural colour
// (soap bubbles, oil slicks, beetle shells, anodised metal). Returns the
// unpolarised power reflectance R(lambda, theta) in [0,1]; the transmitted
// fraction is 1-R, so the film is lossless. cosI is cos of the incidence angle in
// n0; d and lambda must share units (nanometres here).
//
// This is the exact Airy multiple-beam reflectance for a single film: the full
// geometric sum over every internal round trip between the two interfaces,
// evaluated per polarisation (s and p) and averaged. It is correct at every
// thickness and angle and is naturally bounded in [0,1]. (The earlier two-beam
// form kept only the first two reflected beams; the Airy denominator below
// restores the higher-order beams, sharpening the fringes.)
//
// `k2` is the substrate's extinction coefficient: k2==0 is a transparent
// dielectric substrate (lossless, the transmitted 1-R passes through) and takes
// the exact real-valued path below (bit-identical to the pre-absorption engine).
// k2>0 is an absorbing/metallic substrate (complex index n2+i*k2): the bottom
// interface uses complex Fresnel coefficients, so the interference colour shifts
// and desaturates the way real metal-backed films do, and the transmitted light is
// absorbed (opaque). R is still the reflected power fraction in [0,1].
inline double thinFilmReflectance(double n0, double n1, double n2, double k2,
                                  double d, double cosI, double lambda) {
    cosI = clamp01(std::fabs(cosI));
    double sin0_2 = std::max(0.0, 1.0 - cosI * cosI);
    // Snell into the film: sin(theta1) = (n0/n1) sin(theta0).
    double sin1_2 = (n0 * n0) / (n1 * n1) * sin0_2;
    if (sin1_2 >= 1.0) return 1.0;                       // (n1>=n0 so this won't fire)
    double cos1 = std::sqrt(1.0 - sin1_2);
    if (k2 != 0.0) {
        // Absorbing/metallic substrate: complex index n2c = n2 + i*k2. Work with the
        // admittance q = n*cos(theta) whose transverse-momentum form q = sqrt(n^2 -
        // n0^2 sin^2 theta0) is analytic across the (now complex) substrate. The top
        // interface (n0|n1) stays real; only the bottom (n1|n2c) is complex.
        using cd = std::complex<double>;
        cd n2c(n2, k2);
        double q0 = n0 * cosI, q1 = n1 * cos1;           // real incident/film admittances
        cd q2 = std::sqrt(n2c * n2c - cd(n0 * n0 * sin0_2, 0.0));
        if (q2.imag() < 0.0) q2 = -q2;                   // decaying (absorbing) branch
        // Fresnel amplitude reflections. s-pol uses q; p-pol uses n^2/q, arranged as
        // (nb^2 qa - na^2 qb)/(...) so the real limit matches the rS/rP forms above.
        double r01s = (q0 - q1) / (q0 + q1);
        double r01p = (n1 * n1 * q0 - n0 * n0 * q1) / (n1 * n1 * q0 + n0 * n0 * q1);
        cd r12s = (cd(q1) - q2) / (cd(q1) + q2);
        cd r12p = (n2c * n2c * cd(q1) - cd(n1 * n1) * q2) /
                  (n2c * n2c * cd(q1) + cd(n1 * n1) * q2);
        double phi = (4.0 * PI * n1 * d * cos1) / lambda;
        cd p = std::exp(cd(0.0, phi));                   // round-trip phase factor e^{i*phi}
        auto Rpol = [&](double r01, cd r12) {
            cd num = cd(r01) + r12 * p;
            cd den = cd(1.0) + cd(r01) * r12 * p;
            return clamp01(std::norm(num) / std::norm(den));  // |num/den|^2
        };
        return 0.5 * (Rpol(r01s, r12s) + Rpol(r01p, r12p));
    }
    // Snell into the substrate: sin(theta2) = (n0/n2) sin(theta0).
    double sin2_2 = (n0 * n0) / (n2 * n2) * sin0_2;
    bool tir = sin2_2 >= 1.0;                            // TIR at the n1|n2 interface
    double cos2 = tir ? 0.0 : std::sqrt(1.0 - sin2_2);
    // Fresnel amplitude reflection coefficients (s- and p-polarised) at each face.
    auto rS = [](double na, double ca, double nb, double cb) {
        return (na * ca - nb * cb) / (na * ca + nb * cb);
    };
    auto rP = [](double na, double ca, double nb, double cb) {
        return (nb * ca - na * cb) / (nb * ca + na * cb);
    };
    double r01s = rS(n0, cosI, n1, cos1), r01p = rP(n0, cosI, n1, cos1);
    double r12s = tir ? 1.0 : rS(n1, cos1, n2, cos2);   // |r|=1 amplitude on TIR
    double r12p = tir ? 1.0 : rP(n1, cos1, n2, cos2);
    double phi  = (4.0 * PI * n1 * d * cos1) / lambda;   // interference phase
    double cphi = std::cos(phi);
    // Exact Airy multiple-beam power reflectance per polarisation: the geometric
    // sum over all internal round trips. num is the two-beam result; the den term
    // adds the higher-order beams and keeps R in [0,1] without clamping.
    auto Rpol = [&](double r01, double r12) {
        double num = r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * cphi;
        double den = 1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * cphi;
        return clamp01(den > 1e-12 ? num / den : num);
    };
    return 0.5 * (Rpol(r01s, r12s) + Rpol(r01p, r12p));
}

// --- Multilayer stack reflectance (Abeles characteristic-matrix method) ------
// Power reflectance of an ordered stack of `nLayers` thin films (per-layer real
// index nL[j], extinction kL[j], thickness dL[j] in nm) between incident medium n0
// and substrate ns+i*ks. This is the exact generalisation of the single-film Airy
// formula to N layers, handling absorbing layers/substrate via complex indices.
// The colour of a Bragg stack (beetle/Morpho/nacre) or a dichroic mirror falls out
// of the multiple-layer interference. cosI is cos of the incidence angle in n0; all
// thicknesses and lambda share units (nanometres). Returns R in [0,1].
//
// Each layer contributes the characteristic matrix
//   M_j = [[cos d_j, i sin d_j / eta_j], [i eta_j sin d_j, cos d_j]]
// with phase thickness d_j = (2 pi / lambda) q_j t_j and transverse admittance
// q_j = sqrt(n_j^2 - n0^2 sin^2 theta0); eta = q (s-pol) or n^2/q (p-pol). The
// stack product M, closed with the substrate admittance, gives r = (eta0 B - C) /
// (eta0 B + C) and R = |r|^2, averaged over the two polarisations. `reversed` reads the layers
// last to first, for a ray meeting the stack from its substrate side.
//
// The matrix above, with +i off its diagonal, belongs to the convention in which an absorbing
// medium's index is n - ik (Macleod's), so kL and ks enter with that sign and each admittance
// takes the root with Im(q) <= 0, the one whose field decays. With n + ik instead (Airy's
// convention, thinFilmReflectance's) an absorbing layer amplifies: a thick chromium layer gave R
// far above 1, clamped to 1, where it must tend to bulk chromium. A lossless stack reads the same
// either way (every imaginary part is zero).
inline double multilayerReflectance(double n0, double cosI, double lambda,
                                    const double* nL, const double* kL,
                                    const double* dL, int nLayers,
                                    double ns, double ks, bool reversed = false) {
    using cd = std::complex<double>;
    cosI = clamp01(std::fabs(cosI));
    double sin0_2 = std::max(0.0, 1.0 - cosI * cosI);
    double n0s = n0 * n0 * sin0_2;
    double q0 = n0 * cosI;                         // incident transverse admittance (real)
    auto admit = [](cd nsq, cd q, bool pPol) { return pPol ? nsq / q : q; };
    auto solve = [&](bool pPol) {
        cd M00(1, 0), M01(0, 0), M10(0, 0), M11(1, 0);   // identity
        for (int jj = 0; jj < nLayers; ++jj) {
            const int j = reversed ? nLayers - 1 - jj : jj;
            cd nj(nL[j], -kL[j]);                  // n - ik: see the note above
            cd qj = std::sqrt(nj * nj - cd(n0s, 0.0));
            if (qj.imag() > 0.0) qj = -qj;          // the decaying branch in that convention
            cd eta = admit(nj * nj, qj, pPol);
            cd delta = cd(2.0 * PI * dL[j] / lambda, 0.0) * qj;
            cd c = std::cos(delta), s = std::sin(delta);
            cd L00 = c, L01 = cd(0, 1) * s / eta, L10 = cd(0, 1) * eta * s, L11 = c;
            cd n00 = M00 * L00 + M01 * L10, n01 = M00 * L01 + M01 * L11;
            cd n10 = M10 * L00 + M11 * L10, n11 = M10 * L01 + M11 * L11;
            M00 = n00; M01 = n01; M10 = n10; M11 = n11;
        }
        cd nsub(ns, -ks);
        cd qs = std::sqrt(nsub * nsub - cd(n0s, 0.0));
        if (qs.imag() > 0.0) qs = -qs;
        cd etaS = admit(nsub * nsub, qs, pPol);
        cd eta0 = pPol ? cd(n0 * n0 / q0, 0.0) : cd(q0, 0.0);
        cd B = M00 + M01 * etaS;
        cd C = M10 + M11 * etaS;
        cd r = (eta0 * B - C) / (eta0 * B + C);
        return clamp01(std::norm(r));
    };
    return 0.5 * (solve(false) + solve(true));
}

// The reflectance of a rough interference reflector's coating (Material::iridescence: 1 a thin
// film, 2 a stack) at `lambda`, for light meeting it from air at the cosine `cosI`. `thickness`
// is the film's (nm) where it is mapped; a stack ignores it. 1 on any other material.
inline double coatingReflectance(const Material& m, double thickness, double lambda, double cosI) {
    if (m.iridescence == 0) return 1.0;
    const double ns = m.ior(lambda), ks = m.substrateK(lambda);
    if (m.iridescence == 1)
        return thinFilmReflectance(1.0, m.filmIor(lambda), ns, ks, thickness, cosI, lambda);
    return multilayerReflectance(1.0, cosI, lambda, m.layerN.data(), m.layerK.data(),
                                 m.layerThick.data(), (int)m.layerN.size(), ns, ks);
}
