#pragma once
// Which stored photons a mode-M or mode-S density estimate keeps (MERGE-PROXIMITY): photonmap.h's
// adaptive radius, the gathers in photonmap_render.h and sppm_render.h, and the footprints that
// divide their estimates (photonmap_render.h's probe, gafootprint.h). The scalar rules are
// templates both backends call, so the device's vector glue (render_cuda.cu, dOnGatherSurface)
// cannot drift from them.
//
// Not modes U and J. Their merges divide by the whole disc with no footprint to say how much of it
// is the gather's own surface, so dropping another surface's photons only darkens the disc where
// the surface ends: at the base of a box standing on a floor, 0.94 -> 0.62 of mode R in mode U and
// 1.01 -> 0.61 in mode J at radius 0.1. Their radius shrinks as the render converges, so the
// cross-surface bias they keep vanishes with it, which mode M's adaptive radius never does.
#include <algorithm>
#include <cmath>
#include <math.h>
#include "linalg.h"

#if defined(__CUDACC__)
#define FT_GP_HD __host__ __device__
#else
#define FT_GP_HD
#endif

FT_GP_HD inline float  gpSqrt(float x)  { return sqrtf(x); }
FT_GP_HD inline double gpSqrt(double x) { return ::sqrt(x); }

// MERGE-PROXIMITY: which stored photons lie on the surface a gather point is on. A gather takes the
// photons within a ball, and a second surface inside it -- the floor under a table top, once the
// radius outgrows the table's height -- would lend it light it never received. A photon belongs if
// it lies near the gather point's tangent plane: off it by at most (kGatherPlaneTol + sin(theta))
// times its distance d from the point, theta the angle between their normals, plus `slack` for
// rounding. A smooth surface bends about d*sin(theta)/2 off its tangent plane over d, so a curved
// (or faceted) one keeps its photons, where a parallel surface above or below does not.
// `withinGatherBand` takes the offset `off` (>= 0), d^2 and the cosine `c` between the normals.
constexpr double kGatherPlaneTol = 0.1;
// And it faces the gather point's way, within 60 degrees: the normal test of every photon gather,
// on true normals oriented to the side their light came from.
constexpr double kGatherFacingCos = 0.5;

template <class T> FT_GP_HD inline bool withinGatherBand(T off, T d2, T c, T slack) {
    if (off <= slack) return true;
    T q = T(1) - c * c;
    if (q < T(0)) q = T(0);
    const T k = T(kGatherPlaneTol) + gpSqrt(q);
    const T e = off - slack;
    return e * e <= k * k * d2;
}

// The slack of a gather of radius `radius` at a point on a curve of radius `fiberR` (0 off a curve):
// 1e-3 of the radius, or none of the band at all on a strand thinner than the radius. Such a strand
// -- `type diffuse` fur, grass, wire -- has no tangent plane at the radius's scale: the photons
// around a fur gather lie on the neighbouring strands, which the band would drop while the estimate
// still divides by the whole disc (on fur_creature, up to 30 % darker in mode M). A curve as thick
// as the radius is a surface at that scale, and takes the band as one. The facing test stays.
template <class T> FT_GP_HD inline T gatherBandSlack(T radius, T fiberR) {
    return (fiberR > T(0) && fiberR < radius) ? T(HUGE_VAL) : radius * T(1e-3);
}

// A photon at `pp`, true normal `pn` (on the side its light came from), for a gather at `x`, true
// normal `n`; `d2` = |pp - x|^2, which every caller has from its radius test. `slack` must be the
// same wherever one gather's photons are counted and summed (mode M's radius, its sum and its
// footprint: PhotonMap::gatherSlack), or they see different populations.
inline bool onGatherSurface(const Vec3& x, const Vec3& n, const Vec3& pp, const Vec3& pn, double d2,
                            double slack) {
    const double c = dot(pn, n);
    if (c < kGatherFacingCos) return false;
    return withinGatherBand(std::fabs(dot(pp - x, n)), d2, c, slack);
}
