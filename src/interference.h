// Interference reflectance of a coated surface: a single thin film (Airy) and an N-layer
// stack (Abeles). Pure functions of the optics, shared by the tracers (render.h) and the raster
// preview (raster.h), which has no tracer to call.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include "scene.h"

#include "thinfilm.h"   // clamp01, thinFilmReflectance

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
// takes the root with Im(q) <= 0, the one whose field decays. Fed n + ik instead (Airy's
// convention, thinFilmReflectance's), an absorbing layer would amplify and an absorbing
// substrate's reflection would take a conjugated phase. A lossless stack reads the same either
// way (every imaginary part is zero).
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
// film, 2 a stack) at `lambda`, for light meeting it at the cosine `cosI` from a medium of index
// `n0` (1: air). `thickness` is the film's (nm) where it is mapped; a stack ignores it. 1 on any
// other material.
inline double coatingReflectance(const Material& m, double thickness, double lambda, double cosI,
                                 double n0 = 1.0) {
    if (m.iridescence == 0) return 1.0;
    const double ns = m.ior(lambda), ks = m.substrateK(lambda);
    if (m.iridescence == 1)
        return thinFilmReflectance(n0, m.filmIor(lambda), ns, ks, thickness, cosI, lambda);
    return multilayerReflectance(n0, cosI, lambda, m.layerN.data(), m.layerK.data(),
                                 m.layerThick.data(), (int)m.layerN.size(), ns, ks);
}
