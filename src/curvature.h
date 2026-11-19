// curvature.h -- the mean curvature `curv` (O3) reads, in the scalar forms both backends share: a
// triangle's from its vertex normals (Tri::curvature, the raster previews' per pixel), and a level
// set's from its field's gradient and Hessian by central differences (an isosurface hit). The host
// (geometry.h, implicit.h, raster.h) and the device (render_cuda.cu, raster_cuda.cu) each evaluate
// their own data -- a triangle's corners, a field at the stencil's points -- and hand the numbers
// here, so the arithmetic exists once.
#pragma once
#include "unit_linkage.h"   // FT_UNIT_INLINE

#include <cmath>

#ifdef __CUDACC__
#define CURV_HD __host__ __device__
#else
#define CURV_HD
#endif

// The mean curvature H a triangle's vertex normals imply: half the trace of the map dn that takes
// its edges e1 = v1 - v0, e2 = v2 - v0 to dn1 = n1 - n0, dn2 = n2 - n0 (barycentric interpolation is
// linear, so these fix dn on the whole plane), taken in the dual basis of {e1, e2} through their
// Gram matrix -- which drops dn's out-of-plane part. A unit sphere with exact normals reads 1, a
// plane 0, and scaling a mesh by s scales H by 1/s. Positive where the normals spread apart (a bulge
// on their side). 0 on a degenerate sliver: the sine between its edges under 1e-6, at any size.
CURV_HD FT_UNIT_INLINE double triMeanCurvatureS(double e1x, double e1y, double e1z, double e2x, double e2y, double e2z,
                                        double d1x, double d1y, double d1z, double d2x, double d2y, double d2z) {
    const double gA = e1x * e1x + e1y * e1y + e1z * e1z;
    const double gB = e1x * e2x + e1y * e2y + e1z * e2z;
    const double gC = e2x * e2x + e2y * e2y + e2z * e2z;
    const double gdet = gA * gC - gB * gB;                 // = 4 * area^2 = gA gC sin^2
    if (!(gdet > 1e-12 * gA * gC)) return 0.0;
    const double e1d1 = e1x * d1x + e1y * d1y + e1z * d1z, e2d1 = e2x * d1x + e2y * d1y + e2z * d1z;
    const double e1d2 = e1x * d2x + e1y * d2y + e1z * d2z, e2d2 = e2x * d2x + e2y * d2y + e2z * d2z;
    const double trace = (gC * e1d1 - gB * e2d1 - gB * e1d2 + gA * e2d2) / gdet;
    return 0.5 * trace;
}

// The level-set stencil: point i of 13 at p + h * (x, y, z) -- p, its six axis neighbours (+x, -x,
// +y, -y, +z, -z) and six diagonals (+x+y, -x-y, +x+z, -x-z, +y+z, -y-z).
constexpr int kLevelSetStencil = 13;
CURV_HD FT_UNIT_INLINE void levelSetStencilPoint(int i, double& x, double& y, double& z) {
    x = y = z = 0.0;
    switch (i) {
        case 1: x = 1; break;            case 2: x = -1; break;
        case 3: y = 1; break;            case 4: y = -1; break;
        case 5: z = 1; break;            case 6: z = -1; break;
        case 7: x = 1; y = 1; break;     case 8: x = -1; y = -1; break;
        case 9: x = 1; z = 1; break;     case 10: x = -1; z = -1; break;
        case 11: y = 1; z = 1; break;    case 12: y = -1; z = -1; break;
        default: break;
    }
}

// The stencil's step at a point whose coordinates reach `m`, for a field whose gradient is taken at
// `gradEps`: a second difference divides the field's rounding (about eps * m) by h^2, so h stays at
// least eps^(1/4) * m (2^-13 m), where that error is 1e-7 of a unit curvature at m = 1.
CURV_HD FT_UNIT_INLINE double levelSetStep(double gradEps, double m) {
    const double h = m * 0x1p-13;
    return h > gradEps ? h : gradEps;
}

// The level set's mean curvature through the stencil's centre, H = 1/2 div(grad f / |grad f|) =
// (|g|^2 tr(Hs) - g.Hs.g) / (2 |g|^3), from the field's values f[i] at the stencil's points (step
// h); `g` its central-difference gradient. Positive where the level set bulges toward +gradient
// (out of an f < 0 inside: a sphere SDF of radius R reads 1/R). 0 where the gradient vanishes.
CURV_HD FT_UNIT_INLINE double levelSetCurvature(const double* f, double h, double& gx, double& gy, double& gz) {
    gx = (f[1] - f[2]) / (2 * h); gy = (f[3] - f[4]) / (2 * h); gz = (f[5] - f[6]) / (2 * h);
    const double h2 = h * h;
    const double hxx = (f[1] - 2 * f[0] + f[2]) / h2, hyy = (f[3] - 2 * f[0] + f[4]) / h2;
    const double hzz = (f[5] - 2 * f[0] + f[6]) / h2;
    // f(p + h(ei + ej)) + f(p - h(ei + ej)) = 2 f0 + h^2 (Hii + Hjj + 2 Hij) + O(h^4).
    const double hxy = (f[7] + f[8] - 2 * f[0]) / (2 * h2) - 0.5 * (hxx + hyy);
    const double hxz = (f[9] + f[10] - 2 * f[0]) / (2 * h2) - 0.5 * (hxx + hzz);
    const double hyz = (f[11] + f[12] - 2 * f[0]) / (2 * h2) - 0.5 * (hyy + hzz);
    const double g2 = gx * gx + gy * gy + gz * gz;
    if (!(g2 > 0.0)) return 0.0;
    const double gHg = gx * gx * hxx + gy * gy * hyy + gz * gz * hzz +
                       2.0 * (gx * gy * hxy + gx * gz * hxz + gy * gz * hyz);
    return (g2 * (hxx + hyy + hzz) - gHg) / (2.0 * g2 * sqrt(g2));
}
